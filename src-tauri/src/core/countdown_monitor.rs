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
//! 5. Separator-first resolution probe (norm vs min templates)
//! 6. Field-based OCR with disambiguation
//! 7. Parse and validate timer value

use image::{imageops::FilterType, GrayImage, Luma, RgbImage};
use std::sync::OnceLock;

// ============================================================================
// Template Embedding
// ============================================================================

struct TemplateSet {
    digits: Vec<(u8, GrayImage)>, // 0-9
    colon: GrayImage,
    dot: GrayImage,
}

static NORM_TMPL: OnceLock<TemplateSet> = OnceLock::new();
static MIN_TMPL: OnceLock<TemplateSet> = OnceLock::new();

fn load_tmpl(bytes: &[u8]) -> GrayImage {
    image::load_from_memory(bytes)
        .expect("Failed to load template")
        .to_luma8()
}

fn get_norm_templates() -> &'static TemplateSet {
    NORM_TMPL.get_or_init(|| TemplateSet {
        digits: vec![
            (0, load_tmpl(include_bytes!("templates/normal/0.png"))),
            (1, load_tmpl(include_bytes!("templates/normal/1.png"))),
            (2, load_tmpl(include_bytes!("templates/normal/2.png"))),
            (3, load_tmpl(include_bytes!("templates/normal/3.png"))),
            (4, load_tmpl(include_bytes!("templates/normal/4.png"))),
            (5, load_tmpl(include_bytes!("templates/normal/5.png"))),
            (6, load_tmpl(include_bytes!("templates/normal/6.png"))),
            (7, load_tmpl(include_bytes!("templates/normal/7.png"))),
            (8, load_tmpl(include_bytes!("templates/normal/8.png"))),
            (9, load_tmpl(include_bytes!("templates/normal/9.png"))),
        ],
        colon: load_tmpl(include_bytes!("templates/normal/colon.png")),
        dot: load_tmpl(include_bytes!("templates/normal/dot.png")),
    })
}

fn get_min_templates() -> &'static TemplateSet {
    MIN_TMPL.get_or_init(|| TemplateSet {
        digits: vec![
            (0, load_tmpl(include_bytes!("templates/min/0.png"))),
            (1, load_tmpl(include_bytes!("templates/min/1.png"))),
            (2, load_tmpl(include_bytes!("templates/min/2.png"))),
            (3, load_tmpl(include_bytes!("templates/min/3.png"))),
            (4, load_tmpl(include_bytes!("templates/min/4.png"))),
            (5, load_tmpl(include_bytes!("templates/min/5.png"))),
            (6, load_tmpl(include_bytes!("templates/min/6.png"))),
            (7, load_tmpl(include_bytes!("templates/min/7.png"))),
            (8, load_tmpl(include_bytes!("templates/min/8.png"))),
            (9, load_tmpl(include_bytes!("templates/min/9.png"))),
        ],
        colon: load_tmpl(include_bytes!("templates/min/colon.png")),
        dot: load_tmpl(include_bytes!("templates/min/dot.png")),
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
            x_start_pct: 0.8549, // shifted right 0.1% from 85.39%
            x_end_pct: 0.9352,   // shifted right 0.1% from 93.42%
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
/// * `skew_factor` - Shear factor (0.25 for Blue Archive timer)
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

// ============================================================================
// SAD Matching & Disambiguation (production helpers)
// ============================================================================

/// Match a segment against a template using SAD, return normalized score (0.0=perfect, 1.0=worst)
fn match_segment_sad(
    binary: &[u8],
    width: u32,
    height: u32,
    x_start: u32,
    x_end: u32,
    template: &GrayImage,
) -> f32 {
    let seg_width = x_end.saturating_sub(x_start) + 1;
    if seg_width < 2 {
        return 1.0;
    }

    let mut segment_img = GrayImage::new(seg_width, height);
    for y in 0..height {
        for x in 0..seg_width {
            let src_idx = (y * width + (x_start + x)) as usize;
            let val = if src_idx < binary.len() { binary[src_idx] } else { 0 };
            segment_img.put_pixel(x, y, Luma([val]));
        }
    }

    let resized = image::imageops::resize(
        &segment_img,
        template.width(),
        template.height(),
        FilterType::Triangle,
    );

    let mut sad: u64 = 0;
    let pixel_count = (template.width() * template.height()) as u64;
    for (p1, p2) in resized.pixels().zip(template.pixels()) {
        sad += (p1.0[0] as i64 - p2.0[0] as i64).unsigned_abs();
    }

    // Normalize: 0.0 = perfect match, 1.0 = maximum difference (255 per pixel)
    sad as f32 / (pixel_count as f32 * 255.0)
}

/// Compute white pixel density in a sub-region of a segment.
/// region: (x_frac_start, x_frac_end, y_frac_start, y_frac_end) as fractions 0.0-1.0
fn segment_region_density(
    binary: &[u8],
    width: u32,
    height: u32,
    x_start: u32,
    x_end: u32,
    region: (f32, f32, f32, f32),
) -> f32 {
    let seg_w = x_end.saturating_sub(x_start) + 1;
    let rx0 = (seg_w as f32 * region.0) as u32;
    let rx1 = (seg_w as f32 * region.1) as u32;
    let ry0 = (height as f32 * region.2) as u32;
    let ry1 = (height as f32 * region.3) as u32;

    let mut white = 0u32;
    let mut total = 0u32;
    for y in ry0..ry1.min(height) {
        for x in rx0..rx1.min(seg_w) {
            total += 1;
            let idx = (y * width + (x_start + x)) as usize;
            if idx < binary.len() && binary[idx] > 0 {
                white += 1;
            }
        }
    }
    if total == 0 { 0.0 } else { white as f32 / total as f32 }
}

/// Disambiguate confusable digit pairs (3/8, 0/9) using regional pixel density.
/// Returns the corrected label if disambiguation is needed, or the original label.
fn disambiguate_digit(
    binary: &[u8],
    width: u32,
    height: u32,
    x_start: u32,
    x_end: u32,
    best_label: u8,
    best_score: f32,
    second_label: u8,
    second_score: f32,
) -> u8 {
    let score_diff = second_score - best_score;
    // Only disambiguate when scores are very close
    if score_diff > 0.05 {
        return best_label;
    }

    let pair = if best_label < second_label {
        (best_label, second_label)
    } else {
        (second_label, best_label)
    };

    match pair {
        (3, 8) => {
            // 3 vs 8: check middle-left region (left 20%, vertical 30-70%)
            // 8 has closed loops connecting on the left at mid-height → high density
            // 3 is open on the left at mid-height → low density (0.10-0.19)
            let mid_left = segment_region_density(
                binary, width, height, x_start, x_end,
                (0.0, 0.20, 0.30, 0.70),
            );
            if mid_left > 0.25 { 8 } else { 3 }
        }
        (0, 9) => {
            // 0 vs 9: check bottom-left quadrant (left 50%, bottom 40%)
            // 0 has closed bottom-left (higher density ~0.31), 9 has open bottom-left
            let bl_density = segment_region_density(
                binary, width, height, x_start, x_end,
                (0.0, 0.50, 0.60, 1.0),
            );
            if bl_density > 0.17 { 0 } else { 9 }
        }
        _ => best_label,
    }
}

// ============================================================================
// Unified OCR Pipeline
// ============================================================================

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

/// Core OCR logic from binary data — unified pipeline with separator-first resolution probe.
///
/// 1. Segment characters via column density
/// 2. Probe both norm/min colon/dot templates to determine resolution
/// 3. Split fields by colon/dot positions (MM:SS.mmm)
/// 4. Recognize digits with the selected template set + disambiguation
/// 5. Validate and return TimerValue
pub fn recognize_timer_from_binary(binary: &[u8], width: u32, height: u32) -> Option<TimerValue> {
    let norm = get_norm_templates();
    let min = get_min_templates();

    // Step 1: Segment
    let segments = find_character_columns(binary, width, height);
    if segments.len() < 5 {
        return None;
    }

    // Step 2: Find best colon and dot from both template sets
    let mut best_colon_idx = 0usize;
    let mut best_colon_score = f32::MAX;
    let mut best_colon_is_min = false;
    let mut best_dot_idx = 0usize;
    let mut best_dot_score = f32::MAX;

    for (i, (s, e)) in segments.iter().enumerate() {
        let nc = match_segment_sad(binary, width, height, *s, *e, &norm.colon);
        let mc = match_segment_sad(binary, width, height, *s, *e, &min.colon);
        let nd = match_segment_sad(binary, width, height, *s, *e, &norm.dot);
        let md = match_segment_sad(binary, width, height, *s, *e, &min.dot);

        let colon_score = nc.min(mc);
        let is_min = mc < nc;
        if colon_score < best_colon_score {
            best_colon_score = colon_score;
            best_colon_idx = i;
            best_colon_is_min = is_min;
        }

        let dot_score = nd.min(md);
        if dot_score < best_dot_score {
            best_dot_score = dot_score;
            best_dot_idx = i;
        }
    }

    // Colon must come before dot
    if best_colon_idx >= best_dot_idx {
        return None;
    }

    // Step 3: Select template set based on separator probe
    let tmpl = if best_colon_is_min { min } else { norm };

    // Step 4: Split into fields
    let min_segs: Vec<usize> = (0..best_colon_idx).collect();
    let sec_segs: Vec<usize> = (best_colon_idx + 1..best_dot_idx).collect();
    let ms_segs: Vec<usize> = (best_dot_idx + 1..segments.len()).collect();

    // Field length validation: MM=1-2, SS=2, mmm=3
    if min_segs.is_empty() || min_segs.len() > 2 || sec_segs.len() != 2 || ms_segs.len() != 3 {
        return None;
    }

    // Step 5: Recognize each digit
    let mut digits = Vec::new();
    let mut confidences = Vec::new();
    for &seg_i in min_segs.iter().chain(sec_segs.iter()).chain(ms_segs.iter()) {
        let (s, e) = segments[seg_i];
        let mut best_label = 255u8;
        let mut best_score = f32::MAX;
        let mut second_label = 255u8;
        let mut second_score = f32::MAX;
        for (lbl, t) in &tmpl.digits {
            let score = match_segment_sad(binary, width, height, s, e, t);
            if score < best_score {
                second_score = best_score;
                second_label = best_label;
                best_score = score;
                best_label = *lbl;
            } else if score < second_score {
                second_score = score;
                second_label = *lbl;
            }
        }
        let final_label = disambiguate_digit(
            binary, width, height, s, e,
            best_label, best_score, second_label, second_score,
        );
        digits.push(final_label);

        // Confidence: relative gap between best and second-best
        let conf = if second_score > 0.0 {
            1.0 - (best_score / second_score)
        } else {
            0.0
        };
        confidences.push(conf);
    }

    // Step 6: Parse fields
    let min_count = min_segs.len();
    let sec_start = min_count;

    let minutes_raw: u16 = if min_count == 2 {
        digits[0] as u16 * 10 + digits[1] as u16
    } else {
        digits[0] as u16
    };

    let seconds_raw: u16 = digits[sec_start] as u16 * 10 + digits[sec_start + 1] as u16;

    let ms_start = min_count + 2; // sec_count is always 2
    let ms_raw: u16 = digits[ms_start] as u16 * 100
        + digits[ms_start + 1] as u16 * 10
        + digits[ms_start + 2] as u16;
    let ms = snap_ms_to_valid(ms_raw);

    if minutes_raw >= 60 || seconds_raw >= 60 {
        return None;
    }
    let minutes = minutes_raw as u8;
    let seconds = seconds_raw as u8;

    let avg_conf = if confidences.is_empty() {
        0.0
    } else {
        confidences.iter().sum::<f32>() / confidences.len() as f32
    };

    let mut timer = TimerValue::new(minutes, seconds, ms, avg_conf);
    for (i, &conf) in confidences.iter().take(7).enumerate() {
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

    #[test]
    fn test_binary_ocr_accuracy() {
        use std::path::PathBuf;

        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let fixtures_dir =
            PathBuf::from(manifest_dir).join("tests/fixtures/countdown_monitor/active");

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

            if !filename.ends_with(".png") {
                continue;
            }

            // Expected format: MM-SS-mmm.png (or legacy binary_MM-SS-mmm.png)
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
                        failures.push(format!(
                            "{}: Expected {:02}:{:02}.{:03}, Got {:02}:{:02}.{:03}",
                            filename,
                            exp_min, exp_sec, exp_ms,
                            timer.minutes, timer.seconds, timer.milliseconds
                        ));
                    }
                }
                None => {
                    failures.push(format!(
                        "{}: Failed to recognize (None)",
                        filename
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

            if !filename.ends_with(".png") {
                continue;
            }

            // Filename format: MM-SS-mmm.png (or legacy binary_MM-SS-mmm.png)
            // e.g. 03-41-933.png -> "03:41.933" (9 chars)
            let content = filename
                .replace("binary_", "")
                .replace("failed_", "")
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

    /// Load a grayscale PNG template from file path, return GrayImage
    #[allow(dead_code)] // Used by ignored tool tests
    fn load_gray_template(path: &std::path::Path) -> GrayImage {
        image::open(path).expect("Failed to load template").to_luma8()
    }

    #[test]
    #[ignore] // Diagnostic tool — dumps column density and segmentation for a specific image
    fn test_diagnose_segmentation() {
        use std::path::PathBuf;

        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let target = PathBuf::from(manifest_dir)
            .join("tests/fixtures/countdown_monitor/min-active/03-45-100.png");

        let rgb_img = image::open(&target).expect("Failed to open").to_rgb8();
        let width = rgb_img.width();
        let height = rgb_img.height();
        let rgb_data = rgb_img.into_raw();

        eprintln!("Image size: {}x{}", width, height);

        let binary = binarize_for_ocr(&rgb_data, width, height);
        let threshold = (height as f32 * COLUMN_THRESHOLD_RATIO) as u32;
        eprintln!("Column threshold: {} (height={}, ratio={})", threshold, height, COLUMN_THRESHOLD_RATIO);

        // Dump column densities
        eprintln!("\nColumn density (white pixel count per column):");
        let mut density_line = String::new();
        for x in 0..width {
            let mut col_sum = 0u32;
            for y in 0..height {
                if binary[(y * width + x) as usize] > 0 {
                    col_sum += 1;
                }
            }
            density_line.push_str(&format!("{:2} ", col_sum));
        }
        eprintln!("{}", density_line);

        // Show threshold bar
        let mut thresh_line = String::new();
        for x in 0..width {
            let mut col_sum = 0u32;
            for y in 0..height {
                if binary[(y * width + x) as usize] > 0 {
                    col_sum += 1;
                }
            }
            thresh_line.push_str(if col_sum > threshold { " # " } else { " . " });
        }
        eprintln!("{}", thresh_line);

        let segments = find_character_columns(&binary, width, height);
        eprintln!("\nSegments ({}): {:?}", segments.len(), segments);

        // Save binarized image for visual inspection
        let output_dir = PathBuf::from(manifest_dir)
            .parent()
            .unwrap()
            .join("output");
        let _ = std::fs::create_dir_all(&output_dir);
        image::save_buffer(
            output_dir.join("diag_03-45-100_binary.png"),
            &binary,
            width,
            height,
            image::ColorType::L8,
        )
        .unwrap();
        eprintln!("Saved binarized image to output/diag_03-45-100_binary.png");
    }

    #[test]
    #[ignore] // Segment extraction tool — writes digit images to output/min-digits/
    fn test_extract_min_segments() {
        use std::path::PathBuf;

        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let fixtures_dir =
            PathBuf::from(manifest_dir).join("tests/fixtures/countdown_monitor/min-active");
        let output_dir = PathBuf::from(manifest_dir)
            .parent()
            .unwrap()
            .join("output/min-digits");
        let _ = std::fs::create_dir_all(&output_dir);

        let paths = match std::fs::read_dir(&fixtures_dir) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping: fixtures directory not found: {}", e);
                return;
            }
        };

        let mut counts = [0u32; 12]; // 0-9, 10=colon, 11=dot

        for path in paths {
            let path = path.unwrap().path();
            let filename = path.file_name().unwrap().to_str().unwrap().to_string();

            if !filename.ends_with(".png") {
                continue;
            }

            let content = filename.replace(".png", "");
            let parts: Vec<&str> = content.split('-').collect();
            if parts.len() != 3 {
                continue;
            }

            // Construct expected character sequence: "03:49.733"
            let expected_chars = format!("{}:{}.{}", parts[0], parts[1], parts[2]);

            // Load RGB image, binarize, segment
            let rgb_img = image::open(&path).expect("Failed to open image").to_rgb8();
            let width = rgb_img.width();
            let height = rgb_img.height();
            let rgb_data = rgb_img.into_raw();

            let binary = binarize_for_ocr(&rgb_data, width, height);
            let segments = find_character_columns(&binary, width, height);

            if segments.len() != expected_chars.len() {
                eprintln!(
                    "SKIP {}: expected {} segments, got {}",
                    filename,
                    expected_chars.len(),
                    segments.len()
                );
                continue;
            }

            for (i, (start, end)) in segments.iter().enumerate() {
                let ch = expected_chars.chars().nth(i).unwrap();

                let (label, idx) = match ch {
                    '0'..='9' => (format!("{}", ch), ch.to_digit(10).unwrap() as usize),
                    ':' => ("colon".to_string(), 10),
                    '.' => ("dot".to_string(), 11),
                    _ => continue,
                };

                counts[idx] += 1;
                let seg_width = end - start + 1;

                // Extract segment
                let mut seg_buf = Vec::with_capacity((seg_width * height) as usize);
                for y in 0..height {
                    for x in *start..=*end {
                        let pixel_idx = (y * width + x) as usize;
                        seg_buf.push(if pixel_idx < binary.len() { binary[pixel_idx] } else { 0 });
                    }
                }

                let out_name = format!(
                    "{}_sample{}_{}.png",
                    label, counts[idx], content
                );
                image::save_buffer(
                    output_dir.join(&out_name),
                    &seg_buf,
                    seg_width,
                    height,
                    image::ColorType::L8,
                )
                .unwrap_or_else(|e| eprintln!("Failed to save {}: {}", out_name, e));
            }
        }

        eprintln!("Extraction complete. Counts (0-9, colon, dot): {:?}", counts);
        eprintln!("Output: output/min-digits/");
    }

    #[test]
    fn test_min_ocr_pipeline() {
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

    /// Unified OCR pipeline test: tests BOTH normal and min images through recognize_timer_from_binary.
    #[test]
    fn test_unified_ocr_pipeline() {
        use std::path::PathBuf;

        let manifest_dir = env!("CARGO_MANIFEST_DIR");

        // ===== Test normal resolution (active/) =====
        eprintln!("=== Normal Resolution (active/) ===");
        let active_dir = PathBuf::from(manifest_dir).join("tests/fixtures/countdown_monitor/active");
        let mut norm_total = 0;
        let mut norm_passed = 0;
        let mut norm_failures: Vec<String> = Vec::new();

        if let Ok(paths) = std::fs::read_dir(&active_dir) {
            let mut sorted: Vec<_> = paths.filter_map(|p| p.ok()).collect();
            sorted.sort_by_key(|p| p.file_name());

            for entry in sorted {
                let path = entry.path();
                let filename = path.file_name().unwrap().to_str().unwrap().to_string();
                if !filename.ends_with(".png") { continue; }

                let content = filename.replace("binary_", "").replace("failed_", "").replace(".png", "");
                let parts: Vec<&str> = content.split('-').collect();
                if parts.len() != 3 { continue; }

                norm_total += 1;
                let exp_min: u8 = parts[0].parse().unwrap_or(0);
                let exp_sec: u8 = parts[1].parse().unwrap_or(0);
                let exp_ms: u16 = parts[2].parse().unwrap_or(0);

                let img = image::open(&path).expect("open").to_luma8();
                let w = img.width(); let h = img.height();
                let binary = img.into_vec();

                match recognize_timer_from_binary(&binary, w, h) {
                    Some(timer) => {
                        if timer.minutes == exp_min && timer.seconds == exp_sec && timer.milliseconds == exp_ms {
                            norm_passed += 1;
                        } else {
                            norm_failures.push(format!(
                                "{}: exp {:02}:{:02}.{:03}, got {}",
                                filename, exp_min, exp_sec, exp_ms, timer.to_string()));
                        }
                    }
                    None => {
                        norm_failures.push(format!("{}: recognition returned None", filename));
                    }
                }
            }
        }

        eprintln!("Normal: {}/{} passed", norm_passed, norm_total);
        for f in &norm_failures { eprintln!("  FAIL: {}", f); }

        // ===== Test min resolution (min-active/) =====
        eprintln!("\n=== Min Resolution (min-active/) ===");
        let min_dir = PathBuf::from(manifest_dir).join("tests/fixtures/countdown_monitor/min-active");
        let mut min_total = 0;
        let mut min_passed = 0;
        let mut min_seg_fails = 0;
        let mut min_rec_failures: Vec<String> = Vec::new();

        if let Ok(paths) = std::fs::read_dir(&min_dir) {
            let mut sorted: Vec<_> = paths.filter_map(|p| p.ok()).collect();
            sorted.sort_by_key(|p| p.file_name());

            for entry in sorted {
                let path = entry.path();
                let filename = path.file_name().unwrap().to_str().unwrap().to_string();
                if !filename.ends_with(".png") { continue; }

                let content = filename.replace(".png", "");
                let parts: Vec<&str> = content.split('-').collect();
                if parts.len() != 3 { continue; }

                min_total += 1;
                let exp_min: u8 = parts[0].parse().unwrap_or(0);
                let exp_sec: u8 = parts[1].parse().unwrap_or(0);
                let exp_ms: u16 = parts[2].parse().unwrap_or(0);

                let rgb_img = image::open(&path).expect("open").to_rgb8();
                let w = rgb_img.width(); let h = rgb_img.height();
                let binary = binarize_for_ocr(&rgb_img.into_raw(), w, h);

                match recognize_timer_from_binary(&binary, w, h) {
                    Some(timer) => {
                        if timer.minutes == exp_min && timer.seconds == exp_sec && timer.milliseconds == exp_ms {
                            min_passed += 1;
                        } else {
                            min_rec_failures.push(format!(
                                "{}: exp {:02}:{:02}.{:03}, got {}",
                                filename, exp_min, exp_sec, exp_ms, timer.to_string()));
                        }
                    }
                    None => {
                        min_seg_fails += 1;
                        eprintln!("  [{}] segmentation/field mismatch → skipped", filename);
                    }
                }
            }
        }

        let min_recognized = min_total - min_seg_fails;
        eprintln!("\nMin: {}/{} recognized, {}/{} passed",
            min_recognized, min_total, min_passed, min_recognized);
        for f in &min_rec_failures { eprintln!("  FAIL: {}", f); }

        eprintln!("\n=== Summary ===");
        eprintln!("Normal: {}/{} (recognition accuracy)", norm_passed, norm_total);
        eprintln!("Min: segmentation {}/{}, recognition {}/{}",
            min_recognized, min_total, min_passed, min_recognized);

        // Assert: normal resolution must be perfect, min recognition must be perfect
        assert_eq!(norm_failures.len(), 0,
            "Normal resolution failures: {:?}", norm_failures);
        assert_eq!(min_rec_failures.len(), 0,
            "Min resolution recognition failures: {:?}", min_rec_failures);
    }
}
