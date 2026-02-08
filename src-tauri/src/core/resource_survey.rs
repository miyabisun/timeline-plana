//! Resource Survey Module (資源調査)
//!
//! Reads the EX skill cost gauge from battle screen captures.
//! The cost gauge consists of rectangular boxes at the bottom-right of the screen,
//! each representing one unit of cost. Boxes can be Full (blue), Empty (dark),
//! or Partial (partially filled with blue).
//!
//! # Algorithm
//! 1. Apply skew correction to straighten parallelogram cost boxes
//! 2. Build blue + filled profiles — per-column pixel ratio for dual-signal classification
//! 3. Compute even grid from known max_cost — pitch = ROI width / max_cost
//! 4. Classify each box — Full/Partial/Empty using dual-signal (blue + filled)
//! 5. Calculate cost — full_count + partial_ratio = decimal cost value

/// Cost gauge reading result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CostValue {
    /// Current cost value (e.g. 5.7)
    pub current: f32,
    /// Total number of cost boxes (externally provided)
    pub max_cost: u32,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
}

/// Check if a pixel is "blue" (the filled cost box color).
///
/// Blue Archive cost boxes use a saturated blue/cyan tone.
#[inline]
fn is_blue_pixel(r: u8, g: u8, b: u8) -> bool {
    let b_i = b as i32;
    let r_i = r as i32;
    let g_i = g as i32;
    b_i > 120 && b_i > r_i + 30 && b_i > g_i
}

/// Check if a pixel belongs to a "filled" cost box across all battle states.
///
/// Broader than is_blue_pixel: catches both bright blue (active) and dark teal (pause/slow).
/// Must exclude empty-box dark purple (R≈30, G≈40, B≈70) which also has B>R.
#[inline]
fn is_filled_pixel(r: u8, g: u8, b: u8) -> bool {
    let b_i = b as i32;
    let r_i = r as i32;
    let g_i = g as i32;
    // Blue must be clearly dominant and bright enough
    b_i > 70 && b_i > r_i + 20 && b_i > g_i + 10
        && r < 160
        // Exclude very dark pixels (empty boxes have sum ≈ 130-160)
        && (r_i + g_i + b_i) > 170
}

/// Check if a pixel is part of the cyan "glow" flash effect.
///
/// When a cost box fills up (e.g. 5.9→6.0), the newly-full box briefly flashes
/// bright cyan (R≈10-90, G≈255, B≈255). These pixels fail both is_blue_pixel
/// (B not > G) and is_filled_pixel (B not > G+10) due to G ≈ B.
#[inline]
fn is_glow_pixel(r: u8, g: u8, b: u8) -> bool {
    g >= 250 && b >= 250 && r < 160
}

/// Skew factor for cost box correction.
///
/// The cost boxes are parallelograms leaning right (top-right sticks out).
/// We apply a horizontal shear to straighten them into rectangles:
/// shift = skew_factor * y
/// This moves bottom rows RIGHT, top rows stay, effectively undoing the lean.
pub const COST_SKEW_FACTOR: f32 = 0.175;

/// Apply skew correction to straighten the leaning cost boxes.
///
/// Cost boxes lean to the right (top edge shifted right relative to bottom).
/// We shift lower rows to the RIGHT to align them with upper rows.
///
/// # Returns
/// Tuple of (corrected RGB data, new width, same height)
pub fn apply_cost_skew(rgb_data: &[u8], width: u32, height: u32, skew_factor: f32) -> (Vec<u8>, u32, u32) {
    let max_shift = (skew_factor * height as f32).ceil() as u32;
    let new_width = width + max_shift;

    let mut output = vec![0u8; (new_width * height * 3) as usize];

    for y in 0..height {
        // Bottom rows get maximum shift, top rows get zero shift
        // This straightens the rightward lean of the top edge
        let shift = (skew_factor * y as f32) as u32;

        for x in 0..width {
            let src_idx = ((y * width + x) * 3) as usize;
            let dst_x = x + shift;
            let dst_idx = ((y * new_width + dst_x) * 3) as usize;

            if src_idx + 2 < rgb_data.len() && dst_idx + 2 < output.len() {
                output[dst_idx] = rgb_data[src_idx];
                output[dst_idx + 1] = rgb_data[src_idx + 1];
                output[dst_idx + 2] = rgb_data[src_idx + 2];
            }
        }
    }

    (output, new_width, height)
}

/// Build a per-column blue pixel ratio profile across the central horizontal band.
fn build_blue_profile(rgb_data: &[u8], width: u32, height: u32) -> Vec<f32> {
    let y_start = (height as f32 * 0.30) as u32;
    let y_end = (height as f32 * 0.70) as u32;
    let band_height = y_end.saturating_sub(y_start).max(1);

    let mut profile = Vec::with_capacity(width as usize);

    for x in 0..width {
        let mut blue_count = 0u32;
        for y in y_start..y_end {
            let idx = ((y * width + x) * 3) as usize;
            if idx + 2 < rgb_data.len() {
                if is_blue_pixel(rgb_data[idx], rgb_data[idx + 1], rgb_data[idx + 2]) {
                    blue_count += 1;
                }
            }
        }
        profile.push(blue_count as f32 / band_height as f32);
    }

    profile
}

/// Build a per-column glow pixel ratio profile for flash effect detection.
fn build_glow_profile(rgb_data: &[u8], width: u32, height: u32) -> Vec<f32> {
    let y_start = (height as f32 * 0.30) as u32;
    let y_end = (height as f32 * 0.70) as u32;
    let band_height = y_end.saturating_sub(y_start).max(1);

    let mut profile = Vec::with_capacity(width as usize);

    for x in 0..width {
        let mut count = 0u32;
        for y in y_start..y_end {
            let idx = ((y * width + x) * 3) as usize;
            if idx + 2 < rgb_data.len() {
                if is_glow_pixel(rgb_data[idx], rgb_data[idx + 1], rgb_data[idx + 2]) {
                    count += 1;
                }
            }
        }
        profile.push(count as f32 / band_height as f32);
    }

    profile
}

/// Build a per-column filled pixel ratio profile using the broad filled-pixel detector.
///
/// Unlike build_blue_profile which only catches active-state blue, this catches
/// filled boxes across active/pause/slow states.
fn build_filled_profile(rgb_data: &[u8], width: u32, height: u32) -> Vec<f32> {
    let y_start = (height as f32 * 0.30) as u32;
    let y_end = (height as f32 * 0.70) as u32;
    let band_height = y_end.saturating_sub(y_start).max(1);

    let mut profile = Vec::with_capacity(width as usize);

    for x in 0..width {
        let mut count = 0u32;
        for y in y_start..y_end {
            let idx = ((y * width + x) * 3) as usize;
            if idx + 2 < rgb_data.len() {
                if is_filled_pixel(rgb_data[idx], rgb_data[idx + 1], rgb_data[idx + 2]) {
                    count += 1;
                }
            }
        }
        profile.push(count as f32 / band_height as f32);
    }

    profile
}

/// Classify a box region using both filled and blue profiles.
///
/// Uses the broad filled_profile as primary signal (works across all battle states),
/// with blue_profile as a high-confidence secondary signal.
fn classify_box_broad(filled_profile: &[f32], blue_profile: &[f32], start: u32, end: u32) -> f32 {
    if start > end {
        return 0.0;
    }

    let filled_cols: Vec<f32> = (start..=end)
        .filter_map(|x| filled_profile.get(x as usize).copied())
        .collect();
    let blue_cols: Vec<f32> = (start..=end)
        .filter_map(|x| blue_profile.get(x as usize).copied())
        .collect();

    if filled_cols.is_empty() {
        return 0.0;
    }

    let avg_filled: f32 = filled_cols.iter().sum::<f32>() / filled_cols.len() as f32;
    let avg_blue: f32 = blue_cols.iter().sum::<f32>() / blue_cols.len() as f32;

    // Strong blue signal = definitely full (active state)
    if avg_blue > 0.85 {
        return 1.0;
    }

    // Partial blue detected — use blue ratio as fill level (active partial box)
    if avg_blue > 0.10 {
        // If filled is also high, use the higher signal
        let fill_ratio = avg_filled.max(avg_blue);
        if fill_ratio > 0.85 {
            return 1.0;
        }
        return fill_ratio.clamp(0.0, 0.99);
    }

    // No blue: use filled profile (pause/slow states)
    if avg_filled > 0.85 {
        1.0
    } else if avg_filled < 0.05 {
        0.0
    } else {
        // Partial via filled profile
        avg_filled.clamp(0.0, 0.99)
    }
}

/// Read the cost gauge from a pre-cropped ROI image.
///
/// # Arguments
/// * `rgb_data` - RGB8 pixel data of the cost gauge ROI
/// * `width` - ROI width in pixels
/// * `height` - ROI height in pixels
/// * `max_cost` - Known number of cost boxes (typically 10)
///
/// # Returns
/// `Some(CostValue)` if cost boxes were successfully detected, `None` otherwise.
pub fn read_cost_gauge(rgb_data: &[u8], width: u32, height: u32, max_cost: u32) -> Option<CostValue> {
    if width < 10 || height < 5 || max_cost < 2 {
        return None;
    }

    // Step 1: Apply skew correction
    let (corrected, corr_w, corr_h) = apply_cost_skew(rgb_data, width, height, COST_SKEW_FACTOR);

    // Step 2: Build profiles for dual-signal classification + glow detection
    let blue_profile = build_blue_profile(&corrected, corr_w, corr_h);
    let filled_profile = build_filled_profile(&corrected, corr_w, corr_h);
    let glow_profile = build_glow_profile(&corrected, corr_w, corr_h);

    // Step 3: Compute even grid from known max_cost
    let pitch = corr_w as f32 / max_cost as f32;
    let boxes: Vec<(u32, u32)> = (0..max_cost)
        .map(|i| {
            let start = (i as f32 * pitch).round() as u32;
            let end = (((i + 1) as f32 * pitch).round() as u32)
                .saturating_sub(1)
                .min(corr_w - 1);
            (start, end)
        })
        .collect();

    // Step 4: Classify each box with glow detection
    let mut classifications: Vec<f32> = Vec::with_capacity(boxes.len());
    let mut is_glow_box: Vec<bool> = Vec::with_capacity(boxes.len());

    for &(start, end) in &boxes {
        let cols = (end - start + 1).max(1) as f32;
        let glow_avg: f32 = (start..=end)
            .filter_map(|x| glow_profile.get(x as usize).copied())
            .sum::<f32>()
            / cols;

        // Strong glow signal → flash box, definitely full
        if glow_avg > 0.70 {
            classifications.push(1.0);
            is_glow_box.push(true);
        } else {
            let fill = classify_box_broad(&filled_profile, &blue_profile, start, end);
            classifications.push(fill);
            is_glow_box.push(false);
        }
    }

    // Step 5: Suppress glow spillover in boxes after a flash box.
    // The cyan flash bleeds into adjacent boxes. Use only the blue profile
    // (immune to cyan glow) for partial boxes immediately after a glow box.
    for i in 1..classifications.len() {
        if is_glow_box[i - 1] && classifications[i] > 0.0 && classifications[i] < 1.0 {
            let (start, end) = boxes[i];
            let cols = (end - start + 1).max(1) as f32;
            let blue_avg: f32 = (start..=end)
                .filter_map(|x| blue_profile.get(x as usize).copied())
                .sum::<f32>()
                / cols;
            classifications[i] = if blue_avg > 0.85 {
                1.0
            } else if blue_avg < 0.05 {
                0.0
            } else {
                blue_avg.clamp(0.0, 0.99)
            };
        }
    }

    // Step 6: Sum classifications
    let mut full_count: u32 = 0;
    let mut partial_ratio: f32 = 0.0;
    let mut found_partial = false;

    for &c in &classifications {
        if c >= 1.0 {
            full_count += 1;
        } else if c > 0.0 && !found_partial {
            partial_ratio = c;
            found_partial = true;
        }
    }

    let current = full_count as f32 + partial_ratio;

    // Confidence is always 1.0 since max_cost is externally guaranteed
    Some(CostValue {
        current,
        max_cost,
        confidence: 1.0,
    })
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

    /// Extract the cost gauge ROI from a full-screen image.
    /// ROI: x=64-89%, y=91.0-93.2%
    fn extract_cost_roi(rgb_data: &[u8], width: u32, height: u32) -> (Vec<u8>, u32, u32) {
        let x_start = (width as f32 * 0.64) as u32;
        let x_end = (width as f32 * 0.89) as u32;
        let y_start = (height as f32 * 0.910) as u32;
        let y_end = (height as f32 * 0.932) as u32;

        let roi_w = x_end - x_start;
        let roi_h = y_end - y_start;

        let mut roi_data = Vec::with_capacity((roi_w * roi_h * 3) as usize);
        for y in y_start..y_end {
            for x in x_start..x_end {
                let idx = ((y * width + x) * 3) as usize;
                if idx + 2 < rgb_data.len() {
                    roi_data.push(rgb_data[idx]);
                    roi_data.push(rgb_data[idx + 1]);
                    roi_data.push(rgb_data[idx + 2]);
                } else {
                    roi_data.extend_from_slice(&[0, 0, 0]);
                }
            }
        }

        (roi_data, roi_w, roi_h)
    }

    #[test]
    fn test_read_cost_gauge_active() {
        let (rgb_data, width, height) = load_test_image("battle-active.png");
        let (roi_data, roi_w, roi_h) = extract_cost_roi(&rgb_data, width, height);

        let result = read_cost_gauge(&roi_data, roi_w, roi_h, 10);
        println!("Cost result: {:?}", result);

        assert!(
            result.is_some(),
            "Should detect cost gauge in battle-active.png"
        );
        let cost = result.unwrap();
        println!(
            "Cost: {:.1} / {} (confidence: {:.2})",
            cost.current, cost.max_cost, cost.confidence
        );

        assert_eq!(cost.max_cost, 10);
        assert!(
            cost.current >= 5.0 && cost.current <= 6.5,
            "Expected current cost 5.0-6.5, got {}",
            cost.current
        );
    }

    #[test]
    fn test_read_cost_gauge_pause() {
        let (rgb_data, width, height) = load_test_image("battle-pause.png");
        let (roi_data, roi_w, roi_h) = extract_cost_roi(&rgb_data, width, height);

        let result = read_cost_gauge(&roi_data, roi_w, roi_h, 10);
        println!("Pause cost result: {:?}", result);

        assert!(result.is_some(), "Should detect cost gauge in battle-pause.png");
        let cost = result.unwrap();
        assert_eq!(cost.max_cost, 10);
        assert!(
            cost.current >= 8.0 && cost.current <= 9.5,
            "Expected current cost 8.0-9.5 in pause, got {}",
            cost.current
        );
    }

    #[test]
    fn test_read_cost_gauge_slow() {
        let (rgb_data, width, height) = load_test_image("battle-slow.png");
        let (roi_data, roi_w, roi_h) = extract_cost_roi(&rgb_data, width, height);

        let result = read_cost_gauge(&roi_data, roi_w, roi_h, 10);
        println!("Slow cost result: {:?}", result);

        assert!(result.is_some(), "Should detect cost gauge in battle-slow.png");
        let cost = result.unwrap();
        assert_eq!(cost.max_cost, 10);
        assert!(
            cost.current >= 0.5 && cost.current <= 2.5,
            "Expected current cost 0.5-2.5 in slow, got {}",
            cost.current
        );
    }

    // ====================================================================
    // Pixel classification tests
    // ====================================================================

    #[test]
    fn test_is_blue_pixel() {
        // Typical active-state blue box pixel
        assert!(is_blue_pixel(50, 100, 200));
        // Pure blue
        assert!(is_blue_pixel(0, 0, 255));
        // Near-white — not blue (B not dominant)
        assert!(!is_blue_pixel(200, 200, 210));
        // Too dark
        assert!(!is_blue_pixel(10, 10, 30));
        // Red dominant
        assert!(!is_blue_pixel(200, 50, 50));
    }

    #[test]
    fn test_is_filled_pixel() {
        // Active-state blue: R=50, G=100, B=200 → B>70, B>R+20, B>G+10, R<160, sum>170
        assert!(is_filled_pixel(50, 100, 200));
        // Pause/slow dark teal: R=40, G=80, B=110
        assert!(is_filled_pixel(40, 80, 110));
        // Empty-box dark purple: R=30, G=40, B=70 → B=70 not > 70, should be excluded
        assert!(!is_filled_pixel(30, 40, 70));
        // White border: R=200 → R >= 160, excluded
        assert!(!is_filled_pixel(200, 200, 230));
        // Very dark pixel: sum ≈ 100 < 170
        assert!(!is_filled_pixel(10, 20, 70));
        // Cyan glow: G≈B → B not > G+10, should NOT match filled
        assert!(!is_filled_pixel(50, 255, 255));
    }

    #[test]
    fn test_is_glow_pixel() {
        // Typical flash: R=10, G=255, B=255
        assert!(is_glow_pixel(10, 255, 255));
        // Flash edge: R=90, G=255, B=255
        assert!(is_glow_pixel(90, 255, 255));
        // R=150 still passes (< 160)
        assert!(is_glow_pixel(150, 255, 255));
        // R=160 fails (not < 160)
        assert!(!is_glow_pixel(160, 255, 255));
        // G=249 fails (not >= 250)
        assert!(!is_glow_pixel(10, 249, 255));
        // Normal blue pixel: G=100, not glow
        assert!(!is_glow_pixel(50, 100, 200));
        // White: R=255, excluded
        assert!(!is_glow_pixel(255, 255, 255));
    }

    // ====================================================================
    // Skew correction tests
    // ====================================================================

    #[test]
    fn test_apply_cost_skew_dimensions() {
        // 100x10 image with factor 0.175
        let width = 100u32;
        let height = 10u32;
        let rgb_data = vec![128u8; (width * height * 3) as usize];

        let (output, new_w, new_h) = apply_cost_skew(&rgb_data, width, height, 0.175);

        // max_shift = ceil(0.175 * 10) = 2
        assert_eq!(new_w, width + 2);
        assert_eq!(new_h, height);
        assert_eq!(output.len(), (new_w * new_h * 3) as usize);
    }

    #[test]
    fn test_apply_cost_skew_zero_factor() {
        let width = 10u32;
        let height = 5u32;
        let rgb_data: Vec<u8> = (0..(width * height * 3) as u8).collect();

        let (output, new_w, new_h) = apply_cost_skew(&rgb_data, width, height, 0.0);

        assert_eq!(new_w, width);
        assert_eq!(new_h, height);
        // With zero skew, output should be identical to input
        assert_eq!(output, rgb_data, "Zero skew should produce identical output");
    }

    #[test]
    fn test_apply_cost_skew_preserves_pixels() {
        // 4x3 image, factor=1.0 → shift by y pixels
        let width = 4u32;
        let height = 3u32;
        let mut rgb_data = vec![0u8; (width * height * 3) as usize];
        // Mark pixel at (0, 2) = bottom-left with distinctive color
        let idx = ((2 * width + 0) * 3) as usize;
        rgb_data[idx] = 255;     // R
        rgb_data[idx + 1] = 128; // G
        rgb_data[idx + 2] = 64;  // B

        let (output, new_w, _) = apply_cost_skew(&rgb_data, width, height, 1.0);

        // Pixel at (0, 2) should shift right by 2 → now at (2, 2)
        let dst_idx = ((2 * new_w + 2) * 3) as usize;
        assert_eq!(output[dst_idx], 255);
        assert_eq!(output[dst_idx + 1], 128);
        assert_eq!(output[dst_idx + 2], 64);
    }

    // ====================================================================
    // Profile builder tests
    // ====================================================================

    #[test]
    fn test_build_blue_profile_all_blue() {
        // Create a 10x10 image where all pixels are blue (R=0, G=0, B=200)
        let width = 10u32;
        let height = 10u32;
        let rgb_data: Vec<u8> = (0..width * height)
            .flat_map(|_| [0u8, 0, 200])
            .collect();

        let profile = build_blue_profile(&rgb_data, width, height);

        assert_eq!(profile.len(), width as usize);
        // All columns should have ratio 1.0 in the central band
        for &val in &profile {
            assert!((val - 1.0).abs() < 0.01, "Expected ~1.0, got {}", val);
        }
    }

    #[test]
    fn test_build_blue_profile_no_blue() {
        // All black pixels — no blue
        let width = 10u32;
        let height = 10u32;
        let rgb_data = vec![0u8; (width * height * 3) as usize];

        let profile = build_blue_profile(&rgb_data, width, height);

        for &val in &profile {
            assert!(val < 0.01, "Expected ~0.0, got {}", val);
        }
    }

    #[test]
    fn test_build_glow_profile() {
        // Create a 10x10 image: left half glow (R=10, G=255, B=255), right half black
        let width = 10u32;
        let height = 10u32;
        let rgb_data: Vec<u8> = (0..width * height)
            .flat_map(|i| {
                let x = i % width;
                if x < 5 { [10u8, 255, 255] } else { [0, 0, 0] }
            })
            .collect();

        let profile = build_glow_profile(&rgb_data, width, height);

        assert_eq!(profile.len(), width as usize);
        // Left half should be high
        for x in 0..5 {
            assert!(profile[x] > 0.9, "Column {} should be glow, got {}", x, profile[x]);
        }
        // Right half should be zero
        for x in 5..10 {
            assert!(profile[x] < 0.01, "Column {} should be empty, got {}", x, profile[x]);
        }
    }

    #[test]
    fn test_build_filled_profile() {
        // Create a 10x10 image: left half filled (R=50, G=80, B=150), right half empty
        let width = 10u32;
        let height = 10u32;
        let rgb_data: Vec<u8> = (0..width * height)
            .flat_map(|i| {
                let x = i % width;
                if x < 5 { [50u8, 80, 150] } else { [0, 0, 0] }
            })
            .collect();

        let profile = build_filled_profile(&rgb_data, width, height);

        assert_eq!(profile.len(), width as usize);
        for x in 0..5 {
            assert!(profile[x] > 0.9, "Column {} should be filled, got {}", x, profile[x]);
        }
        for x in 5..10 {
            assert!(profile[x] < 0.01, "Column {} should be empty, got {}", x, profile[x]);
        }
    }

    // ====================================================================
    // classify_box_broad tests
    // ====================================================================

    #[test]
    fn test_classify_box_broad_full() {
        let blue = vec![0.9, 0.95, 0.88, 0.92, 0.90];
        let filled = vec![0.9, 0.95, 0.88, 0.92, 0.90];
        assert_eq!(classify_box_broad(&filled, &blue, 0, 4), 1.0);
    }

    #[test]
    fn test_classify_box_broad_empty() {
        let blue = vec![0.0, 0.01, 0.0, 0.02, 0.0];
        let filled = vec![0.01, 0.0, 0.02, 0.01, 0.0];
        assert_eq!(classify_box_broad(&filled, &blue, 0, 4), 0.0);
    }

    #[test]
    fn test_classify_box_broad_partial() {
        let blue = vec![0.8, 0.7, 0.3, 0.02, 0.01];
        let filled = vec![0.8, 0.7, 0.3, 0.02, 0.01];
        let fill = classify_box_broad(&filled, &blue, 0, 4);
        assert!(fill > 0.0 && fill < 1.0, "Partial fill: {}", fill);
    }

    #[test]
    fn test_classify_box_broad_filled_only_full() {
        // Pause/slow: high filled, zero blue → should be full
        let blue = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let filled = vec![0.9, 0.92, 0.88, 0.91, 0.90];
        assert_eq!(classify_box_broad(&filled, &blue, 0, 4), 1.0);
    }

    #[test]
    fn test_classify_box_broad_filled_only_partial() {
        // Pause/slow partial: moderate filled, zero blue
        let blue = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let filled = vec![0.5, 0.4, 0.3, 0.0, 0.0];
        let fill = classify_box_broad(&filled, &blue, 0, 4);
        assert!(fill > 0.0 && fill < 1.0, "Expected partial, got {}", fill);
    }

    #[test]
    fn test_classify_box_broad_start_gt_end() {
        let blue = vec![0.9; 5];
        let filled = vec![0.9; 5];
        assert_eq!(classify_box_broad(&filled, &blue, 4, 2), 0.0);
    }

    // ====================================================================
    // read_cost_gauge edge case tests
    // ====================================================================

    #[test]
    fn test_read_cost_gauge_rejects_tiny_input() {
        assert!(read_cost_gauge(&[0; 30], 5, 2, 10).is_none()); // width < 10
        assert!(read_cost_gauge(&[0; 30], 10, 2, 10).is_none()); // height < 5
        assert!(read_cost_gauge(&[0; 300], 10, 10, 1).is_none()); // max_cost < 2
    }

    #[test]
    fn test_read_cost_gauge_all_black() {
        // All-black image → no filled/blue/glow → cost = 0
        let width = 100u32;
        let height = 20u32;
        let rgb_data = vec![0u8; (width * height * 3) as usize];

        let result = read_cost_gauge(&rgb_data, width, height, 10);
        assert!(result.is_some());
        let cost = result.unwrap();
        assert!(cost.current < 0.1, "All black should give ~0 cost, got {}", cost.current);
    }

    /// Load a pre-cropped cost ROI image (already extracted, no need for extract_cost_roi).
    fn load_cost_roi_image(subdir: &str, filename: &str) -> (Vec<u8>, u32, u32) {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = format!(
            "{}/tests/fixtures/resource_survey/{}/{}",
            manifest_dir, subdir, filename
        );
        let img = image::open(&path)
            .unwrap_or_else(|e| panic!("Failed to load {}: {}", path, e));
        let rgb_img = img.to_rgb8();
        let (width, height) = rgb_img.dimensions();
        let rgb_data = rgb_img.into_raw();
        (rgb_data, width, height)
    }

    // ====================================================================
    // Accuracy test harness (shared by active / slow / pause)
    // ====================================================================

    /// Run accuracy test on all `cost_roi_*.png` images in a subdirectory.
    /// Returns (pass_count, fail_count).
    fn run_accuracy_test(subdir: &str, tolerance: f32) -> (usize, usize) {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let dir = format!("{}/tests/fixtures/resource_survey/{}", manifest_dir, subdir);
        let mut entries: Vec<(String, f32)> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("Cannot read {}: {}", dir, e))
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                let val_str = name.strip_prefix("cost_roi_")?.strip_suffix(".png")?;
                let expected: f32 = val_str.parse().ok()?;
                Some((name, expected))
            })
            .collect();
        entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        println!("\n=== {} accuracy test ({} images) ===", subdir, entries.len());
        println!("{:<24} {:>8} {:>8} {:>6} {:>6}", "File", "Expected", "Got", "Diff", "OK?");
        println!("{}", "-".repeat(60));

        let (mut pass, mut fail) = (0usize, 0usize);

        for (filename, expected) in &entries {
            let (roi_data, roi_w, roi_h) = load_cost_roi_image(subdir, filename);
            let detected = read_cost_gauge(&roi_data, roi_w, roi_h, 10)
                .map(|c| c.current)
                .unwrap_or(-1.0);
            let diff = (detected - expected).abs();
            let ok = diff <= tolerance;

            if ok { pass += 1; } else { fail += 1; }

            println!(
                "{:<24} {:>8.1} {:>8.2} {:>6.2} {:>6}",
                filename, expected, detected, diff,
                if ok { "PASS" } else { "FAIL" }
            );
        }

        println!("{}", "-".repeat(60));
        println!("Result: {}/{} passed (tolerance ±{:.1})", pass, pass + fail, tolerance);
        (pass, fail)
    }

    #[test]
    fn test_accuracy_active() {
        let (_, fail) = run_accuracy_test("active", 0.5);
        assert_eq!(fail, 0, "{} images failed accuracy check", fail);
    }

    #[test]
    fn test_accuracy_slow() {
        let (_, fail) = run_accuracy_test("slow", 0.5);
        assert_eq!(fail, 0, "{} images failed accuracy check", fail);
    }

    // ====================================================================
    // Diagnostic tests (print-only, always pass)
    // ====================================================================

    #[test]
    #[ignore] // Diagnostic only: prints per-box values, no assertions
    fn test_diagnostic_cost_all_images() {
        let images = ["battle-active.png", "battle-pause.png", "battle-slow.png"];

        for img_name in images {
            let (rgb_data, width, height) = load_test_image(img_name);
            let (roi_data, roi_w, roi_h) = extract_cost_roi(&rgb_data, width, height);

            println!("\n=== {} ===", img_name);

            let (corrected, corr_w, corr_h) =
                apply_cost_skew(&roi_data, roi_w, roi_h, COST_SKEW_FACTOR);

            let blue_profile = build_blue_profile(&corrected, corr_w, corr_h);
            let filled_profile = build_filled_profile(&corrected, corr_w, corr_h);

            let pitch = corr_w as f32 / 10.0;
            println!("Grid: corr_w={}, pitch={:.1}", corr_w, pitch);

            for i in 0..10u32 {
                let start = (i as f32 * pitch).round() as u32;
                let end = (((i + 1) as f32 * pitch).round() as u32)
                    .saturating_sub(1)
                    .min(corr_w - 1);

                let blue_avg: f32 = (start..=end)
                    .filter_map(|x| blue_profile.get(x as usize).copied())
                    .sum::<f32>()
                    / (end - start + 1).max(1) as f32;
                let fill_avg: f32 = (start..=end)
                    .filter_map(|x| filled_profile.get(x as usize).copied())
                    .sum::<f32>()
                    / (end - start + 1).max(1) as f32;
                let classification = classify_box_broad(&filled_profile, &blue_profile, start, end);

                println!(
                    "  Box {}: cols {}-{} | blue={:.3} filled={:.3} class={:.2}",
                    i, start, end, blue_avg, fill_avg, classification
                );
            }

            let result = read_cost_gauge(&roi_data, roi_w, roi_h, 10);
            println!("  Result: {:?}", result);
        }
    }
}
