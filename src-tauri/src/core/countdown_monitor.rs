//! Countdown Monitor Module (カウントダウン監視)
//!
//! Monitors and extracts the battle countdown timer from screen captures.
//! In Blue Archive, defeating the boss before time runs out is crucial.
//! Handles italic text by applying skew correction (0.25).
//!
//! # Processing Pipeline
//! 1. Extract timer ROI from full frame
//! 2. Normalize ROI to reference resolution (273px wide)
//! 3. Apply skew correction to straighten italic text
//! 4. Binarize with adaptive threshold
//! 5. Template matching OCR for digit recognition
//! 6. Parse and validate timer value
//!

use image::{imageops::FilterType, GrayImage, Luma, RgbImage};
use std::sync::OnceLock;

// Embed templates
static TEMPLATES: OnceLock<Vec<(u8, GrayImage)>> = OnceLock::new();

fn get_templates() -> &'static Vec<(u8, GrayImage)> {
    TEMPLATES.get_or_init(|| {
        let mut t = Vec::new();
        let load_tmpl = |bytes: &[u8], label: u8| {
            let img = image::load_from_memory(bytes)
                .expect("Failed to load template")
                .to_luma8();
            (label, img)
        };

        t.push(load_tmpl(include_bytes!("templates/0.png"), 0));
        t.push(load_tmpl(include_bytes!("templates/1.png"), 1));
        t.push(load_tmpl(include_bytes!("templates/2.png"), 2));
        t.push(load_tmpl(include_bytes!("templates/3.png"), 3));
        t.push(load_tmpl(include_bytes!("templates/4.png"), 4));
        t.push(load_tmpl(include_bytes!("templates/5.png"), 5));
        t.push(load_tmpl(include_bytes!("templates/6.png"), 6));
        t.push(load_tmpl(include_bytes!("templates/7.png"), 7));
        t.push(load_tmpl(include_bytes!("templates/8.png"), 8));
        t.push(load_tmpl(include_bytes!("templates/9.png"), 9));
        t.push(load_tmpl(include_bytes!("templates/colon.png"), 10)); // 10 = colon
        t.push(load_tmpl(include_bytes!("templates/dot.png"), 11)); // 11 = dot

        t
    })
}

/// Timer ROI coordinates (percentage-based for resolution independence)
pub struct TimerROI {
    /// X start position (percentage of width, 0.0 - 1.0)
    pub x_start_pct: f32,
    /// X end position (percentage of width, 0.0 - 1.0)
    pub x_end_pct: f32,
    /// Y start position (percentage of height, 0.0 - 1.0)
    pub y_start_pct: f32,
    /// Y end position (percentage of height, 0.0 - 1.0)
    pub y_end_pct: f32,
}

impl Default for TimerROI {
    fn default() -> Self {
        // Optimized coordinates for 30fps OCR performance
        // Timer "03:41.933" digits only (no clock icon)
        // Content-relative percentages (excluding window capture black bars)
        // Target pixels at 3400x1921: x=2903..3176, y=69..125
        TimerROI {
            x_start_pct: 0.8539, // 2903/3400 — skip clock icon completely
            x_end_pct: 0.9342,   // 3176/3400 — avoid brown corner
            y_start_pct: 0.0360, // 69/1921 — shifted down for vertical centering
            y_end_pct: 0.0651,   // 125/1921 — end with buffer for window scaling
        }
    }
}

/// Skew factor for italic text correction
/// Blue Archive timer uses italic font that requires this correction
/// Tested: 0.16-0.20 (under), 0.25 (optimal based on colon alignment)
pub const SKEW_FACTOR: f32 = 0.25;

/// Reference ROI width for resolution normalization.
/// Templates were created at ~3400px content width → timer ROI ≈ 273px wide.
const REFERENCE_ROI_WIDTH: u32 = 273;

/// Extract the timer region from a full frame
///
/// # Arguments
/// * `rgb_data` - RGB8 pixel data (3 bytes per pixel)
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
/// * `roi` - Region of interest coordinates
///
/// # Returns
/// Tuple of (cropped RGB data, crop width, crop height)
pub fn extract_timer_roi(
    rgb_data: &[u8],
    width: u32,
    height: u32,
    roi: &TimerROI,
) -> (Vec<u8>, u32, u32) {
    let x_start = (width as f32 * roi.x_start_pct) as u32;
    let x_end = (width as f32 * roi.x_end_pct) as u32;
    let y_start = (height as f32 * roi.y_start_pct) as u32;
    let y_end = (height as f32 * roi.y_end_pct) as u32;

    let crop_width = x_end.saturating_sub(x_start);
    let crop_height = y_end.saturating_sub(y_start);

    let mut cropped = Vec::with_capacity((crop_width * crop_height * 3) as usize);

    for y in y_start..y_end {
        for x in x_start..x_end {
            let src_idx = ((y * width + x) * 3) as usize;
            if src_idx + 2 < rgb_data.len() {
                cropped.push(rgb_data[src_idx]); // R
                cropped.push(rgb_data[src_idx + 1]); // G
                cropped.push(rgb_data[src_idx + 2]); // B
            } else {
                cropped.extend_from_slice(&[0, 0, 0]); // Black padding
            }
        }
    }

    (cropped, crop_width, crop_height)
}

/// Normalize ROI to reference resolution for consistent OCR results.
///
/// Resizes the ROI to [`REFERENCE_ROI_WIDTH`] (maintaining aspect ratio) so that
/// binarization thresholds, column-density segmentation, and template matching
/// all operate at the same scale regardless of the game window size.
///
/// Skips resizing when the input width is within ±5% of the reference to avoid
/// unnecessary interpolation on already-correct-sized images.
///
/// # Arguments
/// * `rgb_data` - RGB8 pixel data of the cropped ROI
/// * `width` - ROI width in pixels
/// * `height` - ROI height in pixels
///
/// # Returns
/// Tuple of (normalized RGB data, new width, new height)
pub fn normalize_roi(rgb_data: &[u8], width: u32, height: u32) -> (Vec<u8>, u32, u32) {
    // Skip if already within ±5% of reference width
    let ratio = width as f32 / REFERENCE_ROI_WIDTH as f32;
    if (0.95..=1.05).contains(&ratio) {
        return (rgb_data.to_vec(), width, height);
    }

    let new_width = REFERENCE_ROI_WIDTH;
    let new_height = ((height as f32) * (new_width as f32 / width as f32)).round() as u32;

    // Build an RgbImage from raw data
    let img = RgbImage::from_raw(width, height, rgb_data.to_vec())
        .expect("normalize_roi: invalid RGB buffer size");

    let resized = image::imageops::resize(&img, new_width, new_height, FilterType::Triangle);

    (resized.into_raw(), new_width, new_height)
}

/// Apply skew correction to straighten italic text
///
/// Uses horizontal shear transformation:
/// x' = x + skew_factor * (height - y)
///
/// This shifts pixels horizontally based on their vertical position,
/// effectively rotating italic text to be vertical.
///
/// # Arguments
/// * `rgb_data` - RGB8 pixel data
/// * `width` - Image width
/// * `height` - Image height
/// * `skew_factor` - Shear factor (0.45 for Blue Archive timer)
///
/// # Returns
/// Tuple of (corrected RGB data, new width, height)
pub fn apply_skew_correction(
    rgb_data: &[u8],
    width: u32,
    height: u32,
    skew_factor: f32,
) -> (Vec<u8>, u32, u32) {
    // Calculate new width needed after skew
    // Maximum horizontal shift is skew_factor * height
    let max_shift = (skew_factor * height as f32).ceil() as u32;
    let new_width = width + max_shift;

    // Create output buffer (initialize with black/transparent)
    let mut output = vec![0u8; (new_width * height * 3) as usize];

    for y in 0..height {
        // Calculate horizontal shift for this row
        // Italic text leans RIGHT (top is shifted right)
        // To correct: shift BOTTOM rows right, keep TOP rows left
        // This effectively moves top of character left relative to bottom
        let shift = (skew_factor * y as f32) as u32;

        for x in 0..width {
            let src_idx = ((y * width + x) * 3) as usize;
            let dst_x = x + shift;
            let dst_idx = ((y * new_width + dst_x) * 3) as usize;

            if src_idx + 2 < rgb_data.len() && dst_idx + 2 < output.len() {
                output[dst_idx] = rgb_data[src_idx]; // R
                output[dst_idx + 1] = rgb_data[src_idx + 1]; // G
                output[dst_idx + 2] = rgb_data[src_idx + 2]; // B
            }
        }
    }

    (output, new_width, height)
}

/// Process timer region: extract ROI and apply skew correction
///
/// # Arguments
/// * `rgb_data` - Full frame RGB8 data
/// * `width` - Frame width
/// * `height` - Frame height
///
/// # Returns
/// Tuple of (processed RGB data, width, height)
pub fn process_timer_region(rgb_data: &[u8], width: u32, height: u32) -> (Vec<u8>, u32, u32) {
    let roi = TimerROI::default();

    // Step 1: Extract timer ROI
    let (cropped, crop_w, crop_h) = extract_timer_roi(rgb_data, width, height, &roi);

    // Step 2: Normalize to reference resolution
    let (normalized, norm_w, norm_h) = normalize_roi(&cropped, crop_w, crop_h);

    // Step 3: Apply skew correction
    apply_skew_correction(&normalized, norm_w, norm_h, SKEW_FACTOR)
}

// ============================================================================
// OCR Section - Template Matching for Timer Digits
// ============================================================================

/// Target white pixel ratio for the "0" digit (calibration target)
/// Reduced to 0.15 to thin out characters and suppress background noise
const TARGET_WHITE_RATIO: f32 = 0.15;

/// Minimum column sum ratio to consider as part of a character
const COLUMN_THRESHOLD_RATIO: f32 = 0.05;

/// Expected digit positions (approximate percentage of width)
/// Format: 0M:SS.mmm (9 characters including separators)
#[allow(dead_code)]
const DIGIT_POSITIONS: [f32; 9] = [
    0.02, // 0 (always 0)
    0.13, // M (minutes)
    0.26, // : (colon)
    0.37, // S (tens of seconds)
    0.50, // S (seconds)
    0.62, // . (period)
    0.73, // m (hundreds ms)
    0.84, // m (tens ms) - only 0/3/6
    0.95, // m (ms) - only 0/3/7
];

/// Timer recognition result
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct TimerValue {
    /// Minutes (0-9, first digit always 0)
    pub minutes: u8,
    /// Seconds (0-59)
    pub seconds: u8,
    /// Milliseconds (0-999, last 2 digits: 00/33/67)
    pub milliseconds: u16,
    /// Overall confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Per-digit confidence scores
    pub digit_confidence: [f32; 7],
}

impl TimerValue {
    /// Create a new timer value
    pub fn new(minutes: u8, seconds: u8, milliseconds: u16, confidence: f32) -> Self {
        Self {
            minutes,
            seconds,
            milliseconds,
            confidence,
            digit_confidence: [1.0; 7],
        }
    }

    /// Convert to total milliseconds
    pub fn to_millis(&self) -> u32 {
        (self.minutes as u32 * 60_000) + (self.seconds as u32 * 1000) + (self.milliseconds as u32)
    }

    /// Format as string "MM:SS.mmm"
    pub fn to_string(&self) -> String {
        format!(
            "{:02}:{:02}.{:03}",
            self.minutes, self.seconds, self.milliseconds
        )
    }
}

/// Calculate adaptive threshold using first "0" digit as calibration
/// Returns optimal threshold to achieve consistent text thickness
pub fn calculate_adaptive_threshold(rgb_data: &[u8], width: u32, height: u32) -> u8 {
    // First digit "0" is located at approximately 0-12% of width
    let x_start = 0u32;
    let x_end = (width as f32 * 0.12) as u32;

    // Collect luminance values from the first digit region
    let mut luminances: Vec<u8> = Vec::new();

    for y in 0..height {
        for x in x_start..x_end {
            let idx = ((y * width + x) * 3) as usize;
            if idx + 2 < rgb_data.len() {
                let r = rgb_data[idx];
                let g = rgb_data[idx + 1];
                let b = rgb_data[idx + 2];
                let lum = (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u8;
                luminances.push(lum);
            }
        }
    }

    if luminances.is_empty() {
        return 80; // Fallback
    }

    // Sort luminances to find appropriate threshold
    luminances.sort_unstable();

    // Find threshold that gives target white ratio
    let target_idx = ((1.0 - TARGET_WHITE_RATIO) * luminances.len() as f32) as usize;
    let threshold = luminances
        .get(target_idx.min(luminances.len() - 1))
        .copied()
        .unwrap_or(80);

    threshold.clamp(40, 200)
}

/// Binarize RGB image with adaptive threshold
pub fn binarize_adaptive(rgb_data: &[u8], width: u32, height: u32, threshold: u8) -> Vec<u8> {
    let mut binary = vec![0u8; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            if idx + 2 < rgb_data.len() {
                let r = rgb_data[idx];
                let g = rgb_data[idx + 1];
                let b = rgb_data[idx + 2];

                // Luminance formula (weighted for human perception)
                let lum = (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u8;

                // White text detection
                if lum > threshold {
                    binary[(y * width + x) as usize] = 255;
                }
            }
        }
    }

    binary
}

/// Binarize RGB image with auto-calibrated threshold
pub fn binarize_for_ocr(rgb_data: &[u8], width: u32, height: u32) -> Vec<u8> {
    let threshold = calculate_adaptive_threshold(rgb_data, width, height);
    binarize_adaptive(rgb_data, width, height, threshold)
}

/// Find character boundaries by analyzing column density
pub fn find_character_columns(binary: &[u8], width: u32, height: u32) -> Vec<(u32, u32)> {
    let threshold = (height as f32 * COLUMN_THRESHOLD_RATIO) as u32;
    let mut boundaries = Vec::new();
    let mut in_char = false;
    let mut start = 0u32;

    for x in 0..width {
        // Count white pixels in this column
        let mut col_sum = 0u32;
        for y in 0..height {
            if binary[(y * width + x) as usize] > 0 {
                col_sum += 1;
            }
        }

        if col_sum > threshold {
            if !in_char {
                in_char = true;
                start = x;
            }
        } else if in_char {
            in_char = false;
            boundaries.push((start, x - 1));
        }
    }

    // Handle character extending to edge
    if in_char {
        boundaries.push((start, width - 1));
    }

    boundaries
}

/// Validate milliseconds pattern (must be X00, X33, or X67)
pub fn validate_ms_pattern(ms: u16) -> bool {
    let pattern = ms % 100;
    pattern == 0 || pattern == 33 || pattern == 67
}

/// Snap milliseconds to nearest valid pattern (00/33/67)
pub fn snap_ms_to_valid(ms: u16) -> u16 {
    let hundreds = ms / 100;
    let tens_units = ms % 100;

    let snapped = if tens_units < 17 {
        0
    } else if tens_units < 50 {
        33
    } else if tens_units < 84 {
        67
    } else {
        // Roll over to next hundred
        return ((hundreds + 1) % 10) * 100;
    };

    hundreds * 100 + snapped
}

/// Template-based digit recognition
/// Returns (digit, confidence)
pub fn recognize_digit(
    binary: &[u8],
    width: u32,
    height: u32,
    x_start: u32,
    x_end: u32,
) -> (u8, f32) {
    let seg_width = x_end.saturating_sub(x_start) + 1;
    if seg_width < 3 {
        return (0, 0.0);
    }

    // Crop segment to GrayImage
    let mut segment_img = GrayImage::new(seg_width, height);
    for y in 0..height {
        for x in 0..seg_width {
            let src_idx = (y * width + (x_start + x)) as usize;
            let val = if src_idx < binary.len() {
                binary[src_idx]
            } else {
                0
            };
            segment_img.put_pixel(x, y, Luma([val]));
        }
    }

    let templates = get_templates();
    let mut best_label = 255;
    let mut best_score = u64::MAX;
    let mut second_best_score = u64::MAX;

    // Aspect ratio of the input segment
    let ar_in = seg_width as f32 / height as f32;

    for (label, tmpl_img) in templates {
        // Validation: Check aspect ratio similarity
        // If width differs significantly, it's likely a different type of character
        // e.g. '0' (Wide) vs ':' (Narrow)
        let ar_tm = tmpl_img.width() as f32 / tmpl_img.height() as f32;
        let diff_ratio = (ar_in - ar_tm).abs() / ar_tm.max(0.01);

        // Penalty for large aspect ratio mismatch (> 50%)
        let penalty = if diff_ratio > 0.5 {
            100_000_000 // Huge penalty invalidates this match
        } else {
            0
        };

        // Resize segment to match template size
        // Using Triangle for stability
        let resized = image::imageops::resize(
            &segment_img,
            tmpl_img.width(),
            tmpl_img.height(),
            FilterType::Triangle,
        );

        // Calculate SAD (Sum of Absolute Differences)
        let mut score: u64 = penalty;
        if score < best_score {
            for (p1, p2) in resized.pixels().zip(tmpl_img.pixels()) {
                let v1 = p1.0[0] as i32;
                let v2 = p2.0[0] as i32;
                score += (v1 - v2).abs() as u64;
            }
        }

        if score < best_score {
            second_best_score = best_score;
            best_score = score;
            best_label = *label;
        } else if score < second_best_score {
            second_best_score = score;
        }
    }

    // Calculate relative confidence
    let confidence = if second_best_score > 0 {
        1.0 - (best_score as f32 / second_best_score as f32)
    } else {
        0.0
    };

    // Low confidence threshold for debugging (optional)
    if confidence < 0.05 && best_score > 5000 {
        // Potential ambiguity
    }

    (best_label, confidence)
}

/// Detect if a segment is a separator (colon or period)
pub fn is_separator(binary: &[u8], width: u32, height: u32, x_start: u32, x_end: u32) -> bool {
    let seg_width = x_end.saturating_sub(x_start) + 1;

    // Separators are typically narrow
    if seg_width > 15 {
        return false;
    }

    // Count vertical gaps
    let mid_y = height / 2;
    let mut gap_count = 0u32;

    for y in (mid_y.saturating_sub(5))..=(mid_y + 5).min(height - 1) {
        let mut has_pixel = false;
        for x in x_start..=x_end.min(width - 1) {
            let idx = (y * width + x) as usize;
            if idx < binary.len() && binary[idx] > 0 {
                has_pixel = true;
                break;
            }
        }
        if !has_pixel {
            gap_count += 1;
        }
    }

    // Colon has a gap in the middle
    gap_count >= 3
}

/// Full OCR pipeline: process image and return timer value
pub fn recognize_timer(rgb_data: &[u8], width: u32, height: u32) -> Option<TimerValue> {
    // Step 1: Process timer region (ROI + skew correction)
    let (processed, proc_w, proc_h) = process_timer_region(rgb_data, width, height);

    // Step 2: Binarize
    let binary = binarize_for_ocr(&processed, proc_w, proc_h);

    recognize_timer_from_binary(&binary, proc_w, proc_h)
}

/// OCR pipeline for pre-cropped Timer ROI data (skips ROI extraction)
/// Use this when the input is already the Timer ROI region
pub fn recognize_timer_from_roi(rgb_data: &[u8], width: u32, height: u32) -> Option<TimerValue> {
    // Step 1: Normalize to reference resolution
    let (normalized, norm_w, norm_h) = normalize_roi(rgb_data, width, height);

    // Step 2: Apply skew correction only (ROI already extracted)
    let (skewed, skew_w, skew_h) = apply_skew_correction(&normalized, norm_w, norm_h, SKEW_FACTOR);

    // Step 3: Binarize
    let binary = binarize_for_ocr(&skewed, skew_w, skew_h);

    recognize_timer_from_binary(&binary, skew_w, skew_h)
}

/// Parse timer digits (minutes, seconds, milliseconds) from a cleaned digit array.
///
/// Input: array of numeric digits (0-9) with separators already removed.
/// Expected format after cleaning: `[0] M S S m m m` (5-9 digits).
///
/// Returns `Some((minutes, seconds, snapped_ms))` or `None` if parsing fails.
fn parse_timer_digits(clean_digits: &[u8]) -> Option<(u8, u8, u16)> {
    if clean_digits.len() < 5 {
        return None;
    }

    let len = clean_digits.len();

    // Milliseconds: last 3 digits
    let ms_val = (clean_digits[len - 3] as u16 * 100)
        + (clean_digits[len - 2] as u16 * 10)
        + (clean_digits[len - 1] as u16);
    let snapped_ms = snap_ms_to_valid(ms_val);

    // Seconds: 2 digits before ms (with separator skip for 8+ digits)
    let (s1_idx, s10_idx) = if len >= 8 {
        (len - 5, len - 6)
    } else {
        (len - 4, len - 5)
    };

    if s10_idx >= clean_digits.len() {
        return None;
    }

    let seconds_val = clean_digits[s10_idx] * 10 + clean_digits[s1_idx];
    if seconds_val >= 60 {
        return None;
    }

    // Minutes: digit before seconds (with separator skip for 9+ digits)
    let mut minutes_val = 0u8;
    if len >= 7 {
        let m_idx = if len >= 9 {
            s10_idx.checked_sub(2)
        } else {
            s10_idx.checked_sub(1)
        };
        if let Some(idx) = m_idx {
            if idx < clean_digits.len() {
                minutes_val = clean_digits[idx];
            }
        }
    }

    // Minutes must be single digit (0-9)
    if minutes_val >= 10 {
        return None;
    }

    // Note: ms validation is handled by snap_ms_to_valid() which always produces
    // a valid X00/X33/X67 pattern. Explicit validate_ms_pattern() check is unnecessary.

    Some((minutes_val, seconds_val, snapped_ms))
}

/// Core OCR logic from binary data
pub fn recognize_timer_from_binary(binary: &[u8], width: u32, height: u32) -> Option<TimerValue> {
    let segments = find_character_columns(binary, width, height);

    let mut confidences: Vec<f32> = Vec::new();
    let mut clean_digits = Vec::new();
    let mut clean_confs = Vec::new();

    for (start, end) in &segments {
        let (digit, conf) = recognize_digit(binary, width, height, *start, *end);
        confidences.push(conf);
        if digit < 10 {
            clean_digits.push(digit);
            clean_confs.push(conf);
        }
    }

    let (minutes_val, seconds_val, snapped_ms) = parse_timer_digits(&clean_digits)?;

    let avg_conf = if confidences.is_empty() {
        0.0
    } else {
        confidences.iter().sum::<f32>() / confidences.len() as f32
    };

    let mut timer = TimerValue::new(minutes_val, seconds_val, snapped_ms, avg_conf);

    for (i, &conf) in clean_confs.iter().take(7).enumerate() {
        timer.digit_confidence[i] = conf;
    }

    Some(timer)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_test_image(filename: &str) -> (Vec<u8>, u32, u32) {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = format!("{}/tests/fixtures/screenshots/{}", manifest_dir, filename);
        let img = image::open(&path)
            .unwrap_or_else(|e| panic!("Failed to load test image {}: {}", path, e));
        let rgb_img = img.to_rgb8();
        let (width, height) = rgb_img.dimensions();
        let rgb_data = rgb_img.into_raw();

        // Detect and crop black bars (content anchored at top-left)
        let (cw, ch) = crate::core::visual_intercept::detect_content_bounds(&rgb_data, width, height, 3);
        let mut cropped = Vec::with_capacity((cw * ch * 3) as usize);
        for y in 0..ch {
            let start = (y * width * 3) as usize;
            let end = start + (cw * 3) as usize;
            cropped.extend_from_slice(&rgb_data[start..end]);
        }
        (cropped, cw, ch)
    }

    #[test]
    fn test_timer_roi_extraction() {
        let (rgb_data, width, height) = load_test_image("battle-active.png");

        let roi = TimerROI::default();
        let (cropped, crop_w, crop_h) = extract_timer_roi(&rgb_data, width, height, &roi);

        // Content-relative ROI: x=85.39%-93.42%, y=3.60%-6.51%
        let expected_w = ((width as f32 * 0.9342) as u32) - ((width as f32 * 0.8539) as u32);
        let expected_h = ((height as f32 * 0.0651) as u32) - ((height as f32 * 0.0360) as u32);
        assert_eq!(crop_w, expected_w, "ROI width mismatch for {}x{}", width, height);
        assert_eq!(crop_h, expected_h, "ROI height mismatch for {}x{}", width, height);
        assert_eq!(cropped.len(), (crop_w * crop_h * 3) as usize, "RGB data size mismatch");
    }

    #[test]
    fn test_skew_correction() {
        let (rgb_data, width, height) = load_test_image("battle-active.png");

        let roi = TimerROI::default();
        let (cropped, crop_w, crop_h) = extract_timer_roi(&rgb_data, width, height, &roi);

        let (corrected, new_w, new_h) =
            apply_skew_correction(&cropped, crop_w, crop_h, SKEW_FACTOR);

        // SKEW_FACTOR=0.25: max_shift = ceil(crop_h * 0.25)
        let max_shift = (crop_h as f32 * SKEW_FACTOR).ceil() as u32;
        let expected_w = crop_w + max_shift;
        assert_eq!(new_w, expected_w, "Expected skew-corrected width {}", expected_w);
        assert_eq!(new_h, crop_h, "Height should remain {}", crop_h);
        assert_eq!(corrected.len(), (new_w * new_h * 3) as usize);
    }

    #[test]
    fn test_full_processing_pipeline() {
        let images = ["battle-active.png", "battle-pause.png", "battle-slow.png"];

        for img_name in images {
            let (rgb_data, width, height) = load_test_image(img_name);
            let (processed, proc_w, proc_h) = process_timer_region(&rgb_data, width, height);

            println!("{}: Processed to {}x{}", img_name, proc_w, proc_h);

            // Verify output is valid
            assert!(proc_w > 0);
            assert!(proc_h > 0);
            assert_eq!(processed.len(), (proc_w * proc_h * 3) as usize);
        }
    }

    #[test]
    fn test_ms_validation() {
        // Valid patterns
        assert!(validate_ms_pattern(0)); // X00
        assert!(validate_ms_pattern(100)); // X00
        assert!(validate_ms_pattern(33)); // X33
        assert!(validate_ms_pattern(133)); // X33
        assert!(validate_ms_pattern(67)); // X67
        assert!(validate_ms_pattern(967)); // X67

        // Invalid patterns
        assert!(!validate_ms_pattern(34));
        assert!(!validate_ms_pattern(50));
        assert!(!validate_ms_pattern(99));
    }

    #[test]
    fn test_ms_snapping() {
        // Snap to 00
        assert_eq!(snap_ms_to_valid(0), 0);
        assert_eq!(snap_ms_to_valid(10), 0);
        assert_eq!(snap_ms_to_valid(16), 0);

        // Snap to 33
        assert_eq!(snap_ms_to_valid(17), 33);
        assert_eq!(snap_ms_to_valid(33), 33);
        assert_eq!(snap_ms_to_valid(49), 33);

        // Snap to 67
        assert_eq!(snap_ms_to_valid(50), 67);
        assert_eq!(snap_ms_to_valid(67), 67);
        assert_eq!(snap_ms_to_valid(83), 67);

        // Rollover to next hundred
        assert_eq!(snap_ms_to_valid(84), 100);
        assert_eq!(snap_ms_to_valid(99), 100);
        assert_eq!(snap_ms_to_valid(184), 200);
        assert_eq!(snap_ms_to_valid(984), 0); // 9 + 1 = 10, mod 10 = 0
    }

    #[test]
    fn test_binarization() {
        let (rgb_data, width, height) = load_test_image("battle-active.png");
        let (processed, proc_w, proc_h) = process_timer_region(&rgb_data, width, height);

        let binary = binarize_for_ocr(&processed, proc_w, proc_h);

        // Count white pixels
        let white_count = binary.iter().filter(|&&p| p > 0).count();
        let total_pixels = (proc_w * proc_h) as usize;
        let white_ratio = white_count as f32 / total_pixels as f32;

        println!("Binarization: {}x{}", proc_w, proc_h);
        println!(
            "White pixels: {} / {} ({:.1}%)",
            white_count,
            total_pixels,
            white_ratio * 100.0
        );

        // Timer text should be roughly 10-30% of the image
        assert!(
            white_ratio > 0.05,
            "Too few white pixels - threshold may be too high"
        );
        assert!(
            white_ratio < 0.50,
            "Too many white pixels - threshold may be too low"
        );
    }

    #[test]
    fn test_character_segmentation() {
        let (rgb_data, width, height) = load_test_image("battle-active.png");
        let (processed, proc_w, proc_h) = process_timer_region(&rgb_data, width, height);
        let binary = binarize_for_ocr(&processed, proc_w, proc_h);

        let boundaries = find_character_columns(&binary, proc_w, proc_h);

        println!("Found {} character segments:", boundaries.len());
        for (i, (start, end)) in boundaries.iter().enumerate() {
            let width = end - start + 1;
            println!("  [{}] x={}-{} (w={})", i, start, end, width);
        }

        // Should find 7-9 segments (0M:SS.mmm - some may merge)
        assert!(boundaries.len() >= 5, "Too few segments detected");
        assert!(boundaries.len() <= 12, "Too many segments detected");
    }

    #[test]
    fn test_timer_value() {
        let timer = TimerValue::new(3, 41, 933, 0.95);

        assert_eq!(timer.minutes, 3);
        assert_eq!(timer.seconds, 41);
        assert_eq!(timer.milliseconds, 933);
        assert_eq!(timer.to_millis(), 221933);
        assert_eq!(timer.to_string(), "03:41.933");
    }

    #[test]
    fn test_full_ocr_pipeline() {
        let test_cases = [
            ("battle-active.png", "03:43.533"),
            ("battle-pause.png", "03:45.667"),
            ("battle-slow.png", "03:39.700"),
        ];

        for (filename, expected) in test_cases {
            let (rgb_data, width, height) = load_test_image(filename);
            let result = recognize_timer(&rgb_data, width, height);

            assert!(result.is_some(), "{}: OCR returned None", filename);

            let timer = result.unwrap();
            assert_eq!(
                timer.to_string(),
                expected,
                "{}: OCR result mismatch",
                filename
            );
        }
    }

    #[test]
    fn test_is_separator_narrow_with_gap() {
        // Create a binary image with a narrow column that has a gap in the middle (colon-like)
        let width = 20u32;
        let height = 20u32;
        let mut binary = vec![0u8; (width * height) as usize];
        // Draw pixels at top and bottom of a narrow column (x=8..=10), skip middle
        for y in 0..height {
            if y < 5 || y > 14 {
                // Top and bottom: draw pixels
                for x in 8..=10 {
                    binary[(y * width + x) as usize] = 255;
                }
            }
            // Middle rows (5-14): no pixels → gap
        }
        assert!(is_separator(&binary, width, height, 8, 10));
    }

    #[test]
    fn test_is_separator_filled_column() {
        // Narrow column fully filled → not a separator (no gap)
        let width = 20u32;
        let height = 20u32;
        let mut binary = vec![0u8; (width * height) as usize];
        for y in 0..height {
            for x in 8..=10 {
                binary[(y * width + x) as usize] = 255;
            }
        }
        assert!(!is_separator(&binary, width, height, 8, 10));
    }

    #[test]
    fn test_is_separator_too_wide() {
        // Wide segment → not a separator regardless of content
        let width = 40u32;
        let height = 20u32;
        let binary = vec![0u8; (width * height) as usize];
        assert!(!is_separator(&binary, width, height, 0, 19));
    }

    #[test]
    fn test_recognize_digit_too_narrow() {
        // Segment width < 3 → returns (0, 0.0)
        let binary = vec![0u8; 100];
        let (digit, conf) = recognize_digit(&binary, 10, 10, 5, 6);
        assert_eq!(digit, 0);
        assert_eq!(conf, 0.0);
    }

    #[test]
    fn test_calculate_adaptive_threshold_empty() {
        // Empty data → fallback 80
        let threshold = calculate_adaptive_threshold(&[], 0, 0);
        assert_eq!(threshold, 80);
    }

    #[test]
    fn test_calculate_adaptive_threshold_uniform_white() {
        // All white pixels → threshold should be clamped high
        let width = 20u32;
        let height = 10u32;
        let data = vec![255u8; (width * height * 3) as usize];
        let threshold = calculate_adaptive_threshold(&data, width, height);
        // With uniform 255 luminance, threshold ≈ 255 but clamped to 200
        assert_eq!(threshold, 200);
    }

    #[test]
    fn test_calculate_adaptive_threshold_real_image() {
        // battle-active.png timer ROI: white text on dark background
        // Expected threshold is high due to high contrast (empirically 200)
        let (rgb_data, width, height) = load_test_image("battle-active.png");
        let (processed, proc_w, proc_h) = process_timer_region(&rgb_data, width, height);
        let threshold = calculate_adaptive_threshold(&processed, proc_w, proc_h);
        assert!(threshold >= 150,
            "High-contrast timer text should produce threshold >= 150, got {}", threshold);
    }

    fn analyze_digit_features(
        binary: &[u8],
        width: u32,
        height: u32,
        x_start: u32,
        x_end: u32,
    ) -> String {
        let seg_width = x_end.saturating_sub(x_start) + 1;
        if seg_width < 3 {
            return format!("Too narrow: {}", seg_width);
        }

        let mut total_pixels = 0u32;
        let mut top_half = 0u32;
        let mut _bottom_half = 0u32;
        let mut left_half = 0u32;
        let mut _right_half = 0u32;
        let mut center_col = 0u32;
        let mid_y = height / 2;
        let mid_x = x_start + seg_width / 2;

        for y in 0..height {
            for x in x_start..=x_end.min(width - 1) {
                let idx = (y * width + x) as usize;
                if idx < binary.len() && binary[idx] > 0 {
                    total_pixels += 1;
                    if y < mid_y {
                        top_half += 1;
                    } else {
                        _bottom_half += 1;
                    }
                    if x < mid_x {
                        left_half += 1;
                    } else {
                        _right_half += 1;
                    }

                    let seg_center = x_start + seg_width / 2;
                    if x >= seg_center.saturating_sub(2) && x <= seg_center + 2 {
                        center_col += 1;
                    }
                }
            }
        }

        if total_pixels == 0 {
            return "Empty".to_string();
        }

        let top_ratio = top_half as f32 / total_pixels as f32;
        let left_ratio = left_half as f32 / total_pixels as f32;
        let center_ratio = center_col as f32 / total_pixels as f32;
        let density = total_pixels as f32 / (seg_width * height) as f32;

        format!(
            "W:{} Dens:{:.2} Top:{:.2} Left:{:.2} Cent:{:.2}",
            seg_width, density, top_ratio, left_ratio, center_ratio
        )
    }

    #[test]
    fn test_binary_ocr_accuracy() {
        use std::path::PathBuf;

        // Ensure stderr output is visible
        eprintln!("DEBUG: Starting test_binary_ocr_accuracy");

        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let fixtures_dir =
            PathBuf::from(manifest_dir).join("tests/fixtures/countdown_monitor/active");

        eprintln!("DEBUG: Loading test images from {:?}", fixtures_dir);

        let paths = match std::fs::read_dir(&fixtures_dir) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping test: fixtures directory not found: {}", e);
                return;
            }
        };

        let mut total = 0;
        let mut passed = 0;
        let mut failures = Vec::new();

        for path in paths {
            let path = path.unwrap().path();
            let filename = path.file_name().unwrap().to_str().unwrap().to_string();

            if !filename.starts_with("binary_") || !filename.ends_with(".png") {
                continue;
            }

            // Expected format: binary_MM-SS-mmm.png or binary_failed_MM-SS-mmm.png
            // Parse expected values
            let content = filename
                .replace("binary_", "")
                .replace("failed_", "")
                .replace(".png", "");
            let parts: Vec<&str> = content.split('-').collect();

            if parts.len() != 3 {
                eprintln!("Skipping invalid filename format: {}", filename);
                continue;
            }

            let exp_min: u8 = parts[0].parse().unwrap_or(0);
            let exp_sec: u8 = parts[1].parse().unwrap_or(0);
            let exp_ms: u16 = parts[2].parse().unwrap_or(0);

            let img = image::open(&path).expect("Failed to open image").to_luma8();
            let width = img.width();
            let height = img.height();
            let binary = img.into_vec();

            total += 1;

            let result = recognize_timer_from_binary(&binary, width, height);

            match result {
                Some(timer) => {
                    let matches = timer.minutes == exp_min
                        && timer.seconds == exp_sec
                        && timer.milliseconds == exp_ms;

                    if matches {
                        passed += 1;
                    } else {
                        // Detailed analysis for failure
                        let segments = find_character_columns(&binary, width, height);
                        let mut debug_info = String::new();
                        for (i, (s, e)) in segments.iter().enumerate() {
                            let feat = analyze_digit_features(&binary, width, height, *s, *e);
                            let (digit, conf) = recognize_digit(&binary, width, height, *s, *e);
                            debug_info.push_str(&format!(
                                "\n    Seg{}: [{}] ({:.2}) -> {}",
                                i, digit, conf, feat
                            ));
                        }

                        failures.push(format!(
                            "{}: Expected {:02}:{:02}.{:03}, Got {:02}:{:02}.{:03}{}",
                            filename,
                            exp_min,
                            exp_sec,
                            exp_ms,
                            timer.minutes,
                            timer.seconds,
                            timer.milliseconds,
                            debug_info
                        ));
                    }
                }
                None => {
                    // Also dump features for None result
                    let segments = find_character_columns(&binary, width, height);
                    let mut debug_info = String::new();
                    for (i, (s, e)) in segments.iter().enumerate() {
                        let feat = analyze_digit_features(&binary, width, height, *s, *e);
                        let (digit, conf) = recognize_digit(&binary, width, height, *s, *e);
                        debug_info.push_str(&format!(
                            "\n    Seg{}: [{}] ({:.2}) -> {}",
                            i, digit, conf, feat
                        ));
                    }

                    failures.push(format!(
                        "{}: Failed to recognize (None){}",
                        filename, debug_info
                    ));
                }
            }
        }

        eprintln!("\n=== OCR Accuracy Results ===");
        println!(
            "Total: {}, Passed: {}, Failed: {}",
            total,
            passed,
            failures.len()
        );
        eprintln!("Accuracy: {:.1}%", (passed as f32 / total as f32) * 100.0);

        if !failures.is_empty() {
            eprintln!("\nFailures:");
            for fail in &failures {
                eprintln!("  {}", fail);
            }
            panic!("OCR accuracy test failed: {}/{} failed", failures.len(), total);
        }
    }
    #[test]
    fn test_parse_timer_digits_7_digits() {
        // 7 digits: [0, M, SS, mmm] → 0 3 4 1 9 3 3
        // len=7, so s1_idx=3, s10_idx=2, minutes from idx 1
        let result = parse_timer_digits(&[0, 3, 4, 1, 9, 3, 3]);
        assert_eq!(result, Some((3, 41, 933)));
    }

    #[test]
    fn test_parse_timer_digits_5_digits_minimum() {
        // 5 digits: [SS, mmm] → 4 1 9 3 3
        // len=5, so s1_idx=1, s10_idx=0, no minutes (len<7)
        let result = parse_timer_digits(&[4, 1, 9, 3, 3]);
        assert_eq!(result, Some((0, 41, 933)));
    }

    #[test]
    fn test_parse_timer_digits_6_digits() {
        // 6 digits: len<7, no minutes
        // s1_idx=2, s10_idx=1
        let result = parse_timer_digits(&[3, 4, 1, 9, 3, 3]);
        assert_eq!(result, Some((0, 41, 933)));
    }

    #[test]
    fn test_parse_timer_digits_8_digits_separator_skip() {
        // 8 digits: separator skip kicks in (len>=8)
        // s1_idx=3, s10_idx=2 (same formula but different branch)
        // minutes: len>=7, len<9 so m_idx = s10_idx - 1 = 1
        let result = parse_timer_digits(&[0, 3, 4, 1, 9, 3, 3, 0]);
        // ms: last 3 = [3, 3, 0] = 330 → snap to 333
        // seconds: s10_idx=2 → 4, s1_idx=3 → 1, seconds=41
        // minutes: idx 1 → 3
        assert_eq!(result, Some((3, 41, 333)));
    }

    #[test]
    fn test_parse_timer_digits_too_few() {
        assert_eq!(parse_timer_digits(&[1, 2, 3, 4]), None);
        assert_eq!(parse_timer_digits(&[1, 2, 3]), None);
        assert_eq!(parse_timer_digits(&[]), None);
    }

    #[test]
    fn test_parse_timer_digits_seconds_overflow() {
        // seconds >= 60 → None
        // 5 digits: s10_idx=0, s1_idx=1
        // seconds = digits[0]*10 + digits[1] = 6*10 + 1 = 61 → None
        let result = parse_timer_digits(&[6, 1, 0, 0, 0]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_timer_digits_ms_snapping() {
        // ms=934 should snap to 933
        let result = parse_timer_digits(&[0, 3, 4, 1, 9, 3, 4]);
        assert_eq!(result, Some((3, 41, 933)));

        // ms=700 should stay 700
        let result = parse_timer_digits(&[5, 4, 7, 0, 0]);
        assert_eq!(result, Some((0, 54, 700)));

        // ms=467 → already valid (X67 pattern), stays 467
        let result = parse_timer_digits(&[0, 0, 4, 6, 7]);
        assert_eq!(result, Some((0, 0, 467)));
    }

    #[test]
    fn test_parse_timer_digits_boundary_minutes() {
        // minutes=9: valid (single digit max)
        // 7 digits: [0, M, SS, mmm] → idx layout: 0=leading, 1=minutes, 2-3=seconds, 4-6=ms
        let result = parse_timer_digits(&[0, 9, 5, 9, 9, 3, 3]);
        assert_eq!(result, Some((9, 59, 933)));

        // minutes=10: invalid (two-digit minutes)
        // Can't happen via recognize_digit (returns 0-9), but parse_timer_digits must reject it
        let result = parse_timer_digits(&[0, 10, 5, 9, 9, 3, 3]);
        assert_eq!(result, None, "Two-digit minutes should be rejected");
    }

    #[test]
    fn test_parse_timer_digits_boundary_seconds() {
        // seconds=59: valid
        let result = parse_timer_digits(&[5, 9, 0, 0, 0]);
        assert_eq!(result, Some((0, 59, 0)));

        // seconds=60: invalid
        let result = parse_timer_digits(&[6, 0, 0, 0, 0]);
        assert_eq!(result, None, "seconds=60 should be rejected");
    }

    #[test]
    fn test_parse_timer_digits_boundary_ms() {
        // ms last two digits = 00: valid (X00 pattern)
        let result = parse_timer_digits(&[3, 0, 1, 0, 0]);
        assert_eq!(result, Some((0, 30, 100)));

        // ms last two digits = 33: valid (X33 pattern)
        let result = parse_timer_digits(&[3, 0, 1, 3, 3]);
        assert_eq!(result, Some((0, 30, 133)));

        // ms last two digits = 67: valid (X67 pattern)
        let result = parse_timer_digits(&[3, 0, 1, 6, 7]);
        assert_eq!(result, Some((0, 30, 167)));
    }

    #[test]
    #[ignore] // Template generation tool, not a test — writes to src/core/templates/
    fn test_extract_templates() {
        use std::path::PathBuf;

        eprintln!("DEBUG: Starting test_extract_templates");

        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let fixtures_dir =
            PathBuf::from(manifest_dir).join("tests/fixtures/countdown_monitor/active");
        let templates_dir = PathBuf::from(manifest_dir).join("src/core/templates");

        // Create templates dir if not exists
        let _ = std::fs::create_dir_all(&templates_dir);

        let paths = match std::fs::read_dir(&fixtures_dir) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping: fixtures directory not found: {}", e);
                return;
            }
        };

        let mut counts = [0; 12]; // 0-9, 10=colon, 11=dot

        for path in paths {
            let path = path.unwrap().path();
            let filename = path.file_name().unwrap().to_str().unwrap().to_string();

            if !filename.starts_with("binary_") || !filename.ends_with(".png") {
                continue;
            }

            // Filename format: binary_MM-SS-mmm.png
            // e.g. binary_03-41-933.png -> "03:41.933" (9 chars)
            let content = filename
                .replace("binary_", "")
                .replace("failed_", "") // Handle failed ones too if they have correct labels now
                .replace(".png", "");
            let parts: Vec<&str> = content.split('-').collect();

            if parts.len() != 3 {
                continue;
            }

            // Construct expected string sequence
            let min_str = parts[0]; // "03"
            let sec_str = parts[1]; // "41"
            let ms_str = parts[2]; // "933"

            // Expected sequence of chars
            let expected_chars = format!("{}:{}.{}", min_str, sec_str, ms_str);

            let img = image::open(&path).expect("Failed to open image").to_luma8();
            let width = img.width();
            let height = img.height();
            let binary = img.into_vec();

            let segments = find_character_columns(&binary, width, height);

            if segments.len() != expected_chars.len() {
                // Skip if segmentation doesn't match char count perfectly
                continue;
            }

            // Extract and Save
            for (i, (start, end)) in segments.iter().enumerate() {
                let char_code = expected_chars.chars().nth(i).unwrap();
                let seg_width = end - start + 1;

                // Extract segment image
                let mut seg_img_buf = Vec::new(); // Gray8 buffer
                for y in 0..height {
                    for x in *start..=*end {
                        let idx = (y * width + x) as usize;
                        let val = if idx < binary.len() { binary[idx] } else { 0 };
                        seg_img_buf.push(val);
                    }
                }

                // Determine label name
                let label = match char_code {
                    '0'..='9' => format!("{}", char_code),
                    ':' => "colon".to_string(),
                    '.' => "dot".to_string(),
                    _ => "unknown".to_string(),
                };

                // Index for uniqueness
                let idx = match char_code {
                    '0'..='9' => char_code.to_digit(10).unwrap() as usize,
                    ':' => 10,
                    '.' => 11,
                    _ => continue,
                };
                counts[idx] += 1;
                let count = counts[idx];

                // SAVE STANDARD TEMPLATE (Sample 1)
                if count == 1 {
                    // 0.png, colon.png, dot.png
                    let std_name = format!("{}.png", label);
                    let std_path = templates_dir.join(std_name);
                    image::save_buffer(
                        &std_path,
                        &seg_img_buf,
                        seg_width,
                        height,
                        image::ColorType::L8,
                    )
                    .unwrap_or_else(|e| {
                        eprintln!(
                            "Failed to save standard template {}: {}",
                            std_path.display(),
                            e
                        )
                    });
                }

                // Limit to 20 samples per char to have enough variety but save space
                if count > 20 {
                    continue;
                }

                let out_name = format!(
                    "{}_sample{}_{}.png",
                    label,
                    count,
                    filename.replace(".png", "")
                );
                let out_path = templates_dir.join(out_name);

                image::save_buffer(
                    &out_path,
                    &seg_img_buf,
                    seg_width,
                    height,
                    image::ColorType::L8,
                )
                .unwrap_or_else(|e| eprintln!("Failed to save template: {}", e));
            }
        }

        eprintln!("DEBUG: Templates extracted.");
    }

    #[test]
    fn test_min_ocr_pipeline() {
        // At minimum resolution (640x360), Timer OCR has limited success.
        // Timer ROI is only ~50px wide, too small for reliable template matching.
        let battle_images = [
            "min-battle-active.png",
            "min-battle-pause.png",
            "min-battle-slow.png",
        ];
        let mut success = 0;
        for filename in battle_images {
            let (rgb_data, width, height) = load_test_image(filename);
            let result = recognize_timer(&rgb_data, width, height);
            if let Some(timer) = result {
                println!("{} ({}x{}): {}", filename, width, height, timer.to_string());
                assert!(timer.minutes <= 59 && timer.seconds <= 59 && timer.milliseconds <= 999,
                    "{}: invalid timer values {:?}", filename, timer);
                success += 1;
            } else {
                println!("{} ({}x{}): OCR returned None (expected at min resolution)", filename, width, height);
            }
        }
        println!("Timer OCR at min resolution: {}/3 succeeded", success);
    }
}
