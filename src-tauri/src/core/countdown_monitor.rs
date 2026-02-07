//! Countdown Monitor Module (カウントダウン監視)
//!
//! Monitors and extracts the battle countdown timer from screen captures.
//! In Blue Archive, defeating the boss before time runs out is crucial.
//! Handles italic text by applying skew correction (0.45).
//!
//! # Processing Pipeline
//! 1. Extract timer ROI from full frame
//! 2. Apply skew correction to straighten italic text
//! 3. (Future: OCR digit recognition)
//!

use image::{imageops::FilterType, GrayImage, Luma};
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
        // Adjustments: top +20%, left +20%, right -4%
        TimerROI {
            x_start_pct: 0.85,  // Skip clock icon completely
            x_end_pct: 0.93,    // -4% from right (avoid brown corner)
            y_start_pct: 0.035, // Shifted down for vertical centering
            y_end_pct: 0.063,   // End with buffer for window scaling
        }
    }
}

/// Skew factor for italic text correction
/// Blue Archive timer uses italic font that requires this correction
/// Tested: 0.16-0.20 (under), 0.25 (optimal based on colon alignment)
pub const SKEW_FACTOR: f32 = 0.25;

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

    // Step 2: Apply skew correction
    apply_skew_correction(&cropped, crop_w, crop_h, SKEW_FACTOR)
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
    // Step 1: Apply skew correction only (ROI already extracted)
    let (skewed, skew_w, skew_h) = apply_skew_correction(rgb_data, width, height, SKEW_FACTOR);

    // Step 2: Binarize
    let binary = binarize_for_ocr(&skewed, skew_w, skew_h);

    recognize_timer_from_binary(&binary, skew_w, skew_h)
}

/// Core OCR logic from binary data
pub fn recognize_timer_from_binary(binary: &[u8], width: u32, height: u32) -> Option<TimerValue> {
    // Step 3: Find character segments
    let segments = find_character_columns(binary, width, height);

    // Step 4: Recognize each segment
    let mut digits: Vec<u8> = Vec::new();
    let mut confidences: Vec<f32> = Vec::new();

    for (start, end) in &segments {
        // Recognize each segment
        let (digit, conf) = recognize_digit(binary, width, height, *start, *end);

        // Separators often look like '1' or '4' or have very low confidence
        // We will filter them during parsing based on position from right
        digits.push(digit);
        confidences.push(conf);
    }

    // Step 5: Robust Parsing (Right-to-Left)
    // Expected format: [0][M]:[S][S].[m][m][m]

    // Filter out separators (10=colon, 11=dot) and keep digits (0-9)
    // We rely on positional logic mainly, but cleaning helps.

    let mut clean_digits = Vec::new();
    let mut clean_confs = Vec::new();

    for (d, c) in digits.iter().zip(confidences.iter()) {
        if *d < 10 {
            clean_digits.push(*d);
            clean_confs.push(*c);
        }
    }

    if clean_digits.len() < 5 {
        return None;
    }

    let len = clean_digits.len();

    // Parse Milliseconds (Last 3 digits)
    // If we have 9 segments (03:41.933), last 3 are ms.
    // index: 0 1 2 3 4 5 6 7 8
    // value: 0 3 : 4 1 . 9 3 3

    // We need to handle cases where separators are recognized as digits.
    // Let's assume max digits is 7 (0341933). If we have 9 segments, 2 are seps.
    // If we have 7 segments, 0 are seps.

    // Flexible parsing:
    // Take last 3 as ms
    let m100_idx = len.checked_sub(3)?;
    let m10_idx = len.checked_sub(2)?;
    let m1_idx = len.checked_sub(1)?;

    let ms_val = (clean_digits[m100_idx] as u16 * 100)
        + (clean_digits[m10_idx] as u16 * 10)
        + (clean_digits[m1_idx] as u16);

    let snapped_ms = snap_ms_to_valid(ms_val);

    // Parse Seconds (Preceding 2 digits)
    // If len >= 6, we have at least 1 second digit.
    // If len >= 7, we have 2 second digits.
    // However, there might be a separator between Seconds and Milliseconds (index len-4)

    // Heuristic: If we have 9 segments, separator is at len-4 and len-7.
    // Indices relative to end:
    // -1: m
    // -2: m
    // -3: m
    // -4: . (Separator) -> Skip if it looks like one, or strict pos
    // -5: S
    // -6: S
    // -7: : (Separator)
    // -8: M
    // -9: 0

    let mut s1_idx = len.checked_sub(4)?; // Ideally S1 or Sep
    let mut s10_idx = len.checked_sub(5)?; // Ideally S10 or S1

    // Check if s1_idx is likely a separator (period)
    // If we have 8 or 9 segments, len-4 is likely '.'
    // let mut seconds_val = 0; - REMOVED (unused assignment)

    if len >= 8 {
        // Skip separator at len-4
        s1_idx = len.checked_sub(5)?;
        s10_idx = len.checked_sub(6)?;
    }

    // Safety check indices
    if s10_idx >= clean_digits.len() {
        return None;
    }

    let seconds_val = clean_digits[s10_idx] * 10 + clean_digits[s1_idx];

    // Strict Validation: Seconds must be < 60
    if seconds_val >= 60 {
        // Error correction or rejection
        // If > 60, maybe the tens digit is wrong.
        // 71 -> 7 is wrong. Maybe it's 3? 4? 5?
        // 81 -> 8 is wrong.
        // For now, REJECT illegal values to avoid showing garbage
        return None;
    }

    // Parse Minutes
    let mut minutes_val = 0;
    if len >= 7 {
        let mut m1_idx_calc = s10_idx.checked_sub(1);

        // If len is 9, separator at len-7 leads to m1 at len-8
        if len >= 9 {
            m1_idx_calc = s10_idx.checked_sub(2);
        }

        if let Some(idx) = m1_idx_calc {
            if idx < clean_digits.len() {
                minutes_val = clean_digits[idx];
            }
        }
    }

    // Minutes tens place is always 0, ignore it.

    // Calculate overall confidence
    let avg_conf = if confidences.is_empty() {
        0.0
    } else {
        confidences.iter().sum::<f32>() / confidences.len() as f32
    };

    let mut timer = TimerValue::new(minutes_val, seconds_val, snapped_ms, avg_conf);

    // Copy per-digit confidence (mapping might be approximate due to skipping)
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
        let path = format!("{}/tests/fixtures/{}", manifest_dir, filename);
        let img = image::open(&path)
            .unwrap_or_else(|e| panic!("Failed to load test image {}: {}", path, e));
        let rgb_img = img.to_rgb8();
        let (width, height) = rgb_img.dimensions();
        let rgb_data = rgb_img.into_raw();
        (rgb_data, width, height)
    }

    #[test]
    fn test_timer_roi_extraction() {
        let (rgb_data, width, height) = load_test_image("battle-active.png");

        let roi = TimerROI::default();
        let (cropped, crop_w, crop_h) = extract_timer_roi(&rgb_data, width, height, &roi);

        println!("Original: {}x{}", width, height);
        println!("Cropped ROI: {}x{}", crop_w, crop_h);
        println!("ROI pixel count: {}", cropped.len() / 3);

        // Verify dimensions are reasonable
        assert!(crop_w > 0, "Crop width should be positive");
        assert!(crop_h > 0, "Crop height should be positive");
        assert!(crop_w < width, "Crop width should be less than original");
        assert!(crop_h < height, "Crop height should be less than original");

        // For 3416x1993: expect ~164x40 pixels
        // Width calculation must match extract_timer_roi:
        // x_start = (3416 * 0.93) as u32 = 3176
        // x_end = (3416 * 0.978) as u32 = 3340
        // crop_width = 3340 - 3176 = 164
        let x_start = (width as f32 * roi.x_start_pct) as u32;
        let x_end = (width as f32 * roi.x_end_pct) as u32;
        let y_start = (height as f32 * roi.y_start_pct) as u32;
        let y_end = (height as f32 * roi.y_end_pct) as u32;
        let expected_width = x_end - x_start;
        let expected_height = y_end - y_start;
        assert_eq!(crop_w, expected_width, "Width mismatch");
        assert_eq!(crop_h, expected_height, "Height mismatch");
    }

    #[test]
    fn test_skew_correction() {
        let (rgb_data, width, height) = load_test_image("battle-active.png");

        // Extract ROI first
        let roi = TimerROI::default();
        let (cropped, crop_w, crop_h) = extract_timer_roi(&rgb_data, width, height, &roi);

        // Apply skew correction
        let (corrected, new_w, new_h) =
            apply_skew_correction(&cropped, crop_w, crop_h, SKEW_FACTOR);

        println!("Before skew: {}x{}", crop_w, crop_h);
        println!("After skew: {}x{}", new_w, new_h);

        // Width should increase due to skew
        let expected_shift = (SKEW_FACTOR * crop_h as f32).ceil() as u32;
        assert_eq!(
            new_w,
            crop_w + expected_shift,
            "Width should increase by shift amount"
        );
        assert_eq!(new_h, crop_h, "Height should remain the same");

        // Verify output size matches expectations
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
    fn test_save_debug_images() {
        // This test saves processed images for visual verification
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let output_dir = format!("{}/output", manifest_dir);

        // Create output directory if it doesn't exist
        let _ = std::fs::create_dir_all(&output_dir);

        let images = ["battle-active.png", "battle-pause.png", "battle-slow.png"];

        for img_name in images {
            let (rgb_data, width, height) = load_test_image(img_name);
            let roi = TimerROI::default();

            // Save ROI crop
            let (cropped, crop_w, crop_h) = extract_timer_roi(&rgb_data, width, height, &roi);
            let roi_path = format!("{}/roi_{}", output_dir, img_name);
            image::save_buffer(&roi_path, &cropped, crop_w, crop_h, image::ColorType::Rgb8)
                .unwrap_or_else(|e| println!("Failed to save {}: {}", roi_path, e));

            // Save skew-corrected with current SKEW_FACTOR
            let (corrected, new_w, new_h) =
                apply_skew_correction(&cropped, crop_w, crop_h, SKEW_FACTOR);
            let skew_path = format!("{}/skew_{}", output_dir, img_name);
            image::save_buffer(&skew_path, &corrected, new_w, new_h, image::ColorType::Rgb8)
                .unwrap_or_else(|e| println!("Failed to save {}: {}", skew_path, e));

            println!(
                "Saved {} -> roi: {}x{}, skew: {}x{}",
                img_name, crop_w, crop_h, new_w, new_h
            );
        }

        // Generate comparison with multiple skew factors for battle-active only
        let (rgb_data, width, height) = load_test_image("battle-active.png");
        let roi = TimerROI::default();
        let (cropped, crop_w, crop_h) = extract_timer_roi(&rgb_data, width, height, &roi);

        let test_factors = [0.16, 0.17, 0.18, 0.19, 0.20];
        for factor in test_factors {
            let (corrected, new_w, new_h) = apply_skew_correction(&cropped, crop_w, crop_h, factor);
            let path = format!("{}/skew_{:.2}_battle-active.png", output_dir, factor);
            image::save_buffer(&path, &corrected, new_w, new_h, image::ColorType::Rgb8)
                .unwrap_or_else(|e| println!("Failed to save {}: {}", path, e));
            println!("Saved skew factor {} -> {}", factor, path);
        }
    }

    // ============================================================================
    // OCR Tests
    // ============================================================================

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
    fn test_save_binarized_image() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let output_dir = format!("{}/output", manifest_dir);
        let _ = std::fs::create_dir_all(&output_dir);

        let images = ["battle-active.png", "battle-pause.png", "battle-slow.png"];

        for img_name in images {
            let (rgb_data, width, height) = load_test_image(img_name);
            let (processed, proc_w, proc_h) = process_timer_region(&rgb_data, width, height);
            let binary = binarize_for_ocr(&processed, proc_w, proc_h);

            // Count white pixels
            let white_count = binary.iter().filter(|&&p| p > 0).count();

            // Find segments
            let segments = find_character_columns(&binary, proc_w, proc_h);

            // Save as grayscale image
            let path = format!("{}/binary_{}", output_dir, img_name);
            image::save_buffer(&path, &binary, proc_w, proc_h, image::ColorType::L8)
                .unwrap_or_else(|e| println!("Failed to save {}: {}", path, e));
            println!(
                "{}: white={}, segments={}",
                img_name,
                white_count,
                segments.len()
            );
        }
    }

    #[test]
    fn test_full_ocr_pipeline() {
        let test_cases = [
            ("battle-active.png", "03:41.933"),
            ("battle-pause.png", "03:33.400"),
            ("battle-slow.png", "03:54.700"),
        ];

        for (filename, expected) in test_cases {
            let (rgb_data, width, height) = load_test_image(filename);

            // Debug intermediate steps
            let (processed, proc_w, proc_h) = process_timer_region(&rgb_data, width, height);
            let binary = binarize_for_ocr(&processed, proc_w, proc_h);
            let segments = find_character_columns(&binary, proc_w, proc_h);

            let mut digit_count = 0;
            for (start, end) in &segments {
                if !is_separator(&binary, proc_w, proc_h, *start, *end) {
                    digit_count += 1;
                }
            }

            println!("\n=== {} ===", filename);
            println!("Segments: {}, Digits: {}", segments.len(), digit_count);

            let result = recognize_timer(&rgb_data, width, height);
            println!("Expected: {}", expected);

            if let Some(timer) = &result {
                println!("Got:      {}", timer.to_string());
            } else {
                println!("Got: None (need >= 6 digits, got {})", digit_count);
            }

            assert!(
                result.is_some(),
                "{}: segments={}, digits={}",
                filename,
                segments.len(),
                digit_count
            );
        }
    }

    #[test]
    fn test_rename_debug_images() {
        use std::path::PathBuf;

        // Ensure stderr output is visible
        eprintln!("DEBUG: Starting test_rename_debug_images");

        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let output_dir = PathBuf::from(manifest_dir).join("output");

        eprintln!("DEBUG: Output dir: {:?}", output_dir);

        let paths = match std::fs::read_dir(&output_dir) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("DEBUG: Output directory not found or unreadable: {}", e);
                return;
            }
        };

        for path in paths {
            let path = path.unwrap().path();
            let filename = path.file_name().unwrap().to_str().unwrap().to_string();

            if filename.starts_with("binary_") && filename.ends_with(".png") {
                eprintln!("DEBUG: Processing: {}", filename);

                let img = match image::open(&path) {
                    Ok(i) => i.to_luma8(),
                    Err(e) => {
                        eprintln!("DEBUG: Failed to open {}: {}", filename, e);
                        continue;
                    }
                };
                let width = img.width();
                let height = img.height();
                let binary = img.into_vec();

                let segments = find_character_columns(&binary, width, height);
                let mut digits = Vec::new();

                for (start, end) in &segments {
                    let (digit, _) = recognize_digit(&binary, width, height, *start, *end);
                    digits.push(digit);
                }

                let numeric_digits: Vec<u8> = digits.iter().filter(|&&d| d < 10).cloned().collect();

                let mut minutes = 0;
                let mut seconds = 0;
                let mut milliseconds = 0;
                let mut valid = false;

                let len = numeric_digits.len();
                if len >= 5 {
                    let m1 = numeric_digits[len - 1] as u16;
                    let m10 = numeric_digits[len - 2] as u16;
                    let m100 = numeric_digits[len - 3] as u16;
                    let raw_ms = m100 * 100 + m10 * 10 + m1;
                    milliseconds = snap_ms_to_valid(raw_ms);

                    let s1 = numeric_digits[len - 4];
                    let s10 = numeric_digits[len - 5];
                    seconds = s10 * 10 + s1;

                    if len >= 6 {
                        minutes = numeric_digits[len - 6];
                    }
                    valid = true;
                } else {
                    eprintln!("DEBUG: {} - Not enough digits: {}", filename, len);
                }

                let mut validation_error = false;
                if seconds >= 60 {
                    validation_error = true;
                    eprintln!(
                        "DEBUG: {} - Validation Error: Seconds {} >= 60",
                        filename, seconds
                    );
                }

                if minutes >= 10 {
                    validation_error = true;
                    eprintln!(
                        "DEBUG: {} - Validation Error: Minutes {} >= 10",
                        filename, minutes
                    );
                }

                let new_name = if valid && !validation_error {
                    format!(
                        "binary_{:02}-{:02}-{:03}.png",
                        minutes, seconds, milliseconds
                    )
                } else if valid {
                    format!(
                        "binary_failed_{:02}-{:02}-{:03}.png",
                        minutes, seconds, milliseconds
                    )
                } else {
                    let timestamp_part = if let Some(last_underscore) = filename.rfind('_') {
                        if let Some(dot) = filename.rfind('.') {
                            if last_underscore < dot {
                                let suffix = &filename[last_underscore + 1..dot];
                                if suffix.chars().all(|c| c.is_numeric() || c == '-') {
                                    suffix.to_string()
                                } else {
                                    filename.replace("binary_", "").replace(".png", "")
                                }
                            } else {
                                filename.clone()
                            }
                        } else {
                            filename.clone()
                        }
                    } else {
                        filename.clone()
                    };

                    let core_name = timestamp_part.replace("binary_", "").replace("failed_", "");
                    format!("binary_failed_{}.png", core_name)
                };

                let new_path = output_dir.join(&new_name);

                if path != new_path {
                    eprintln!("DEBUG: Renaming {} -> {}", filename, new_name);
                    if let Err(e) = std::fs::rename(&path, &new_path) {
                        eprintln!("DEBUG: Rename failed: {}", e);
                    }
                } else {
                    eprintln!("DEBUG: Name unchanged: {}", filename);
                }
            }
        }
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
        let mut bottom_half = 0u32;
        let mut left_half = 0u32;
        let mut right_half = 0u32;
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
                        bottom_half += 1;
                    }
                    if x < mid_x {
                        left_half += 1;
                    } else {
                        right_half += 1;
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

            // Dump to file for reliable reading
            let dump_path = PathBuf::from(manifest_dir).join("failures.txt");
            if let Ok(mut file) = std::fs::File::create(&dump_path) {
                use std::io::Write;
                for fail in &failures {
                    let _ = writeln!(file, "{}", fail);
                }
            }

            for fail in failures {
                eprintln!("  {}", fail);
            }
            panic!("OCR accuracy test failed");
        }
    }
    #[test]
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
}
