//! Combat Intelligence (戦闘情報分析)
//!
//! This module analyzes captured frames to determine the current battle state.
//! It acts as Plana's "eyes" - interpreting visual data to understand if battle
//! is active, paused, or in slow-motion (skill confirmation).

use image::{imageops::FilterType, GrayImage, Luma};
use std::sync::OnceLock;

/// NCC threshold for PAUSE text template matching.
/// Scores above this indicate the PAUSE dialog is visible.
/// Calibrated via diagnostic test: battle-pause ≈ 0.9+, others ≈ 0.0-0.3
pub const PAUSE_NCC_THRESHOLD: f32 = 0.6;

/// PAUSE template ROI (content-relative percentage, excluding black bars)
pub const PAUSE_ROI_X_START: f32 = 0.382;
pub const PAUSE_ROI_X_END: f32 = 0.623;
pub const PAUSE_ROI_Y_START: f32 = 0.135;
pub const PAUSE_ROI_Y_END: f32 = 0.197;

static PAUSE_TEMPLATE: OnceLock<GrayImage> = OnceLock::new();

/// Load PAUSE template (grayscale) using OnceLock lazy init.
fn get_pause_template() -> &'static GrayImage {
    PAUSE_TEMPLATE.get_or_init(|| {
        let bytes = include_bytes!("templates/pause.png");
        image::load_from_memory(bytes)
            .expect("Failed to load pause template")
            .to_luma8()
    })
}

/// Extract ROI from pixel data, convert to grayscale, and resize to target dimensions.
///
/// # Arguments
/// * `data` - Raw pixel data (RGB8 or RGBA8 depending on `bpp`)
/// * `src_width` - Full frame width
/// * `bpp` - Bytes per pixel (3 for RGB, 4 for RGBA)
/// * `roi` - (x_start, y_start, x_end, y_end) in pixels
/// * `target_size` - (width, height) to resize the result to
fn extract_grayscale_scaled(
    data: &[u8],
    src_width: u32,
    bpp: u32,
    roi: (u32, u32, u32, u32),
    target_size: (u32, u32),
) -> GrayImage {
    let (rx0, ry0, rx1, ry1) = roi;
    let roi_w = rx1 - rx0;
    let roi_h = ry1 - ry0;

    let mut gray = GrayImage::new(roi_w, roi_h);
    for y in 0..roi_h {
        for x in 0..roi_w {
            let idx = (((ry0 + y) * src_width + (rx0 + x)) * bpp) as usize;
            // Only R/G/B (idx..idx+2) are read regardless of bpp, so +2 suffices
            if idx + 2 < data.len() {
                let r = data[idx] as f32;
                let g = data[idx + 1] as f32;
                let b = data[idx + 2] as f32;
                let lum = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
                gray.put_pixel(x, y, Luma([lum]));
            }
        }
    }

    image::imageops::resize(&gray, target_size.0, target_size.1, FilterType::Triangle)
}

/// Compute Normalized Cross-Correlation between template and candidate images.
///
/// Returns a value in [-1.0, 1.0]. Higher values indicate better match.
/// Both images must have the same dimensions.
fn compute_ncc(template: &GrayImage, candidate: &GrayImage) -> f32 {
    let n = (template.width() * template.height()) as f64;
    if n == 0.0 {
        return 0.0;
    }

    // Compute means
    let mut sum_t = 0.0_f64;
    let mut sum_c = 0.0_f64;
    for (pt, pc) in template.pixels().zip(candidate.pixels()) {
        sum_t += pt.0[0] as f64;
        sum_c += pc.0[0] as f64;
    }
    let mean_t = sum_t / n;
    let mean_c = sum_c / n;

    // Compute NCC: sum((t-mean_t)*(c-mean_c)) / sqrt(sum((t-mean_t)^2) * sum((c-mean_c)^2))
    let mut numerator = 0.0_f64;
    let mut denom_t = 0.0_f64;
    let mut denom_c = 0.0_f64;

    for (pt, pc) in template.pixels().zip(candidate.pixels()) {
        let dt = pt.0[0] as f64 - mean_t;
        let dc = pc.0[0] as f64 - mean_c;
        numerator += dt * dc;
        denom_t += dt * dt;
        denom_c += dc * dc;
    }

    let denom = (denom_t * denom_c).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    (numerator / denom) as f32
}

/// Detect PAUSE dialog text using NCC template matching on RGB data.
///
/// Returns NCC score (-1.0 to 1.0). Score > PAUSE_NCC_THRESHOLD indicates PAUSE is visible.
pub fn detect_pause_text(rgb_data: &[u8], width: u32, height: u32) -> f32 {
    let template = get_pause_template();
    let target_size = (template.width(), template.height());

    let roi = (
        (width as f32 * PAUSE_ROI_X_START) as u32,
        (height as f32 * PAUSE_ROI_Y_START) as u32,
        (width as f32 * PAUSE_ROI_X_END) as u32,
        (height as f32 * PAUSE_ROI_Y_END) as u32,
    );

    let candidate = extract_grayscale_scaled(rgb_data, width, 3, roi, target_size);
    compute_ncc(template, &candidate)
}

/// Compute PAUSE NCC score from a pre-cropped RGBA ROI.
///
/// This is the consumer-side version: the producer extracts the PAUSE ROI region
/// as a dense RGBA buffer, and this function handles grayscale conversion, resize,
/// and NCC computation on the worker thread (avoiding blocking the capture callback).
///
/// Returns NCC score (-1.0 to 1.0). Score > PAUSE_NCC_THRESHOLD indicates PAUSE is visible.
pub fn compute_pause_ncc_from_roi(roi_rgba: &[u8], roi_width: u32, roi_height: u32) -> f32 {
    if roi_width == 0 || roi_height == 0 {
        return 0.0;
    }
    let template = get_pause_template();
    let target_size = (template.width(), template.height());

    let roi = (0, 0, roi_width, roi_height);
    let candidate = extract_grayscale_scaled(roi_rgba, roi_width, 4, roi, target_size);
    compute_ncc(template, &candidate)
}

/// Represents the current state of battle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BattleState {
    /// No battle detected (pause button not visible)
    Inactive,
    /// Battle in progress at normal speed
    Active,
    /// Pause menu is open (dark overlay visible)
    Paused,
    /// Skill confirmation or slow-motion state (dark overlay visible)
    Slow,
}

/// Analyzes the captured frame to determine the current battle state.
///
/// # Arguments
/// * `rgb_data` - Raw RGB8 pixel data (3 bytes per pixel)
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
///
/// # Returns
/// The detected `BattleState`
pub fn analyze_battle_state(rgb_data: &[u8], width: u32, height: u32) -> BattleState {
    // Get detection metrics
    let (blue_ratio, white_ratio) = check_pause_button_debug(rgb_data, width, height);
    let brightness = calculate_center_brightness(rgb_data, width, height);
    let pause_ncc = detect_pause_text(rgb_data, width, height);

    // Detection logic based on observed diagnostic values:
    // - Active: white_ratio ~0.21, blue_ratio ~0.00 (pure white on dark background)
    // - Home:   white_ratio ~0.43, blue_ratio ~0.26 (UI has blue-tinted elements)
    // - Paused: white_ratio ~0, brightness ~203, pause_ncc > threshold (PAUSE dialog detected)
    // - Slow: white_ratio ~0, brightness ~59 (dark overlay for skill confirmation)

    if white_ratio > 0.05 && blue_ratio < 0.10 {
        // Timer and battle UI clearly visible, no blue tint → Active battle
        // Blue elements indicate home screen UI, not battle pause button
        BattleState::Active
    } else if pause_ncc > PAUSE_NCC_THRESHOLD {
        // PAUSE dialog template matched → definitively Paused
        BattleState::Paused
    } else if white_ratio < 0.02 && blue_ratio < 0.02 {
        // ROI is dark/occluded (no white text, no blue UI) → overlay is covering it
        // This distinguishes Paused/Slow (dark overlay blocks timer) from
        // home screen (has visible UI elements with white/blue pixels)
        if brightness > 150.0 {
            BattleState::Paused
        } else if brightness < 100.0 {
            BattleState::Slow
        } else {
            BattleState::Inactive
        }
    } else {
        // Has visible elements but not battle pattern (e.g. home screen UI)
        BattleState::Inactive
    }
}

/// Check if the pause button is visible in the Wide ROI (85-100% width).
///
/// # Arguments
/// * `rgb_data` - The cropped RGB data containing the 85-100% strip.
/// * `width` - Width of the strip.
/// * `height` - Height of the strip.
pub fn check_pause_presence_in_wide_roi(rgb_data: &[u8], width: u32, height: u32) -> bool {
    // We only need to scan the RIGHT SIDE of this strip for the pause button.
    // The strip covers 0.85 to 1.00.
    // Pause button is at 0.92 to 1.00.
    // 0.92 is roughly (0.92 - 0.85) / (1.00 - 0.85) = 0.07 / 0.15 = 46.6% mark.

    // Scan from 50% to 100% of this strip to be safe.
    let roi_x_start = width / 2;
    let roi_x_end = width;

    // Height: Pause button is at top 5%.
    let roi_y_start = (height as f32 * 0.005) as u32;
    let roi_y_end = (height as f32 * 0.20) as u32; // Scan a bit deeper just in case

    let mut white_pixel_count = 0u32;
    let mut blue_pixel_count = 0u32;
    let mut total_pixels = 0u32;

    for y in (roi_y_start..roi_y_end).step_by(2) {
        for x in (roi_x_start..roi_x_end).step_by(2) {
            let idx = ((y * width + x) * 3) as usize;
            if idx + 2 >= rgb_data.len() {
                continue;
            }

            let r = rgb_data[idx] as i32;
            let g = rgb_data[idx + 1] as i32;
            let b = rgb_data[idx + 2] as i32;

            total_pixels += 1;

            // White II bars of the pause button
            let is_white = r > 200 && g > 200 && b > 200;
            if is_white {
                white_pixel_count += 1;
            }

            // Blue-tinted pixels indicate home screen UI, not battle
            let is_bluish = b > 150 && b > r + 20 && b > g;
            if is_bluish {
                blue_pixel_count += 1;
            }
        }
    }

    if total_pixels == 0 {
        return false;
    }

    let white_ratio = white_pixel_count as f32 / total_pixels as f32;
    let blue_ratio = blue_pixel_count as f32 / total_pixels as f32;

    // Active battle: white pixels present (> 5%) AND no blue tint (< 10%)
    // Home screen has blue-tinted UI elements that falsely trigger white detection
    white_ratio > 0.05 && blue_ratio < 0.10
}

/// Check if the pause button (II icon) or timer UI is visible in the top-right corner.
/// We detect this by looking for white pixels (timer text, pause button icon).
#[allow(dead_code)]
fn check_pause_button_visible(rgb_data: &[u8], width: u32, height: u32) -> bool {
    let (_blue_ratio, white_ratio) = check_pause_button_debug(rgb_data, width, height);
    // If there are significant white pixels in the top-right, battle UI is visible
    // Active: white_ratio ~0.21, Paused: white_ratio ~0 (overlay blocks)
    white_ratio > 0.05
}

/// Returns the blue and white pixel ratios in the top-right corner.
/// Used for battle state detection and diagnostics.
fn check_pause_button_debug(rgb_data: &[u8], width: u32, height: u32) -> (f32, f32) {
    // ROI: top-right corner where the pause button lives
    // 92%-100% horizontally, 0.5%-5% vertically
    let roi_x_start = (width as f32 * 0.92) as u32;
    let roi_x_end = width;
    let roi_y_start = (height as f32 * 0.005) as u32;
    let roi_y_end = (height as f32 * 0.05) as u32;

    let mut blue_pixel_count = 0u32;
    let mut white_pixel_count = 0u32;
    let mut total_pixels = 0u32;

    // Sample every 2nd pixel to speed up
    for y in (roi_y_start..roi_y_end).step_by(2) {
        for x in (roi_x_start..roi_x_end).step_by(2) {
            let idx = ((y * width + x) * 3) as usize;
            if idx + 2 >= rgb_data.len() {
                continue;
            }

            let r = rgb_data[idx] as i32;
            let g = rgb_data[idx + 1] as i32;
            let b = rgb_data[idx + 2] as i32;

            total_pixels += 1;

            let is_bluish = b > 150 && b > r + 20 && b > g;
            let is_white = r > 200 && g > 200 && b > 200;

            if is_bluish {
                blue_pixel_count += 1;
            }
            if is_white {
                white_pixel_count += 1;
            }
        }
    }

    if total_pixels == 0 {
        return (0.0, 0.0);
    }

    (
        blue_pixel_count as f32 / total_pixels as f32,
        white_pixel_count as f32 / total_pixels as f32,
    )
}

/// Calculate average brightness of the center 50% region.
fn calculate_center_brightness(rgb_data: &[u8], width: u32, height: u32) -> f64 {
    let x_start = (width as f32 * 0.25) as u32;
    let x_end = (width as f32 * 0.75) as u32;
    let y_start = (height as f32 * 0.25) as u32;
    let y_end = (height as f32 * 0.75) as u32;

    let mut total_brightness: u64 = 0;
    let mut pixel_count: u64 = 0;

    // Sample every 4th pixel to speed up calculation
    for y in (y_start..y_end).step_by(4) {
        for x in (x_start..x_end).step_by(4) {
            let idx = ((y * width + x) * 3) as usize;
            if idx + 2 >= rgb_data.len() {
                continue;
            }

            let r = rgb_data[idx] as u64;
            let g = rgb_data[idx + 1] as u64;
            let b = rgb_data[idx + 2] as u64;

            // Perceived brightness (ITU-R BT.601)
            let brightness = (299 * r + 587 * g + 114 * b) / 1000;
            total_brightness += brightness;
            pixel_count += 1;
        }
    }

    if pixel_count == 0 {
        return 0.0;
    }

    total_brightness as f64 / pixel_count as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Load a test image from the tests/fixtures directory, crop black bars, and convert to RGB8 data.
    fn load_test_image(filename: &str) -> (Vec<u8>, u32, u32) {
        // CARGO_MANIFEST_DIR points to src-tauri during cargo test
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
    fn test_battle_active() {
        let (rgb_data, width, height) = load_test_image("battle-active.png");
        let state = analyze_battle_state(&rgb_data, width, height);
        assert_eq!(
            state,
            BattleState::Active,
            "Expected Active state for battle-active.png"
        );
    }

    #[test]
    fn test_battle_paused() {
        let (rgb_data, width, height) = load_test_image("battle-pause.png");
        let state = analyze_battle_state(&rgb_data, width, height);
        assert_eq!(
            state,
            BattleState::Paused,
            "Expected Paused state for battle-pause.png, got {:?}",
            state
        );
    }

    #[test]
    fn test_battle_slow() {
        let (rgb_data, width, height) = load_test_image("battle-slow.png");
        let state = analyze_battle_state(&rgb_data, width, height);
        assert!(
            state == BattleState::Paused || state == BattleState::Slow,
            "Expected Paused or Slow state for battle-slow.png, got {:?}",
            state
        );
    }

    #[test]
    fn test_home_inactive() {
        let (rgb_data, width, height) = load_test_image("home.png");
        let state = analyze_battle_state(&rgb_data, width, height);
        assert_eq!(
            state,
            BattleState::Inactive,
            "Expected Inactive state for home.png, got {:?}",
            state
        );
    }

    #[test]
    fn test_home_noui_inactive() {
        let (rgb_data, width, height) = load_test_image("home-noui.png");
        let state = analyze_battle_state(&rgb_data, width, height);
        assert_eq!(
            state,
            BattleState::Inactive,
            "Expected Inactive state for home-noui.png, got {:?}",
            state
        );
    }

    #[test]
    fn test_pause_button_detection() {
        let (rgb_data, width, height) = load_test_image("battle-active.png");
        let visible = check_pause_button_visible(&rgb_data, width, height);
        assert!(
            visible,
            "Pause button should be visible in battle-active.png"
        );
    }

    #[test]
    fn test_center_brightness_active() {
        let (rgb_data, width, height) = load_test_image("battle-active.png");
        let brightness = calculate_center_brightness(&rgb_data, width, height);
        println!("Active brightness: {}", brightness);
        assert!(
            brightness > 80.0,
            "Active screen should have brightness > 80, got {}",
            brightness
        );
    }

    #[test]
    fn test_center_brightness_paused() {
        let (rgb_data, width, height) = load_test_image("battle-pause.png");
        let brightness = calculate_center_brightness(&rgb_data, width, height);
        println!("Paused brightness: {}", brightness);
        // Paused screen has white PAUSE menu in center, expect HIGHER brightness
        // This is different from Slow mode which has dark overlay
        assert!(
            brightness > 150.0,
            "Paused screen should have high brightness (white menu), got {}",
            brightness
        );
    }

    #[test]
    #[ignore] // Diagnostic only: prints pixel values, no assertions
    fn test_diagnostic_all_images() {
        // Diagnostic test: print all values to calibrate thresholds
        println!("\n=== DIAGNOSTIC OUTPUT ===\n");

        let images = ["battle-active.png", "battle-pause.png", "battle-slow.png", "home.png", "home-noui.png"];
        for img_name in images {
            let (rgb_data, width, height) = load_test_image(img_name);
            let brightness = calculate_center_brightness(&rgb_data, width, height);
            let pause_visible = check_pause_button_visible(&rgb_data, width, height);
            let pause_ncc = detect_pause_text(&rgb_data, width, height);
            let state = analyze_battle_state(&rgb_data, width, height);

            println!("Image: {} ({}x{})", img_name, width, height);
            let (blue_ratio, white_ratio) = check_pause_button_debug(&rgb_data, width, height);
            println!(
                "  Blue ratio: {:.4}, White ratio: {:.4}",
                blue_ratio, white_ratio
            );
            println!("  Pause NCC score: {:.6}", pause_ncc);
            println!("  Pause button visible: {}", pause_visible);
            println!("  Center brightness: {:.2}", brightness);
            println!("  Detected state: {:?}", state);

            // Sample some pixels in the ROI region
            let roi_x_start = (width as f32 * 0.975) as u32;
            let roi_x_end = width.saturating_sub(5);
            let roi_y_start = (height as f32 * 0.01) as u32;
            let roi_y_end = (height as f32 * 0.045) as u32;
            let scan_y = (roi_y_start + roi_y_end) / 2;

            println!(
                "  ROI: x={}..{}, y={}..{}, scan_y={}",
                roi_x_start, roi_x_end, roi_y_start, roi_y_end, scan_y
            );

            // Sample 10 pixels across the ROI
            let step = std::cmp::max(1, (roi_x_end - roi_x_start) / 10);
            print!("  Pixels at y={}: ", scan_y);
            for x in (roi_x_start..roi_x_end).step_by(step as usize) {
                let idx = ((scan_y * width + x) * 3) as usize;
                if idx + 2 < rgb_data.len() {
                    let r = rgb_data[idx];
                    let g = rgb_data[idx + 1];
                    let b = rgb_data[idx + 2];
                    print!("({},{},{}) ", r, g, b);
                }
            }
            println!();
            println!();
        }

        // This test always passes - it's for diagnostic output
        println!("=== END DIAGNOSTIC ===\n");
    }

    #[test]
    #[ignore] // Template generation tool — writes to src/core/templates/pause.png
    fn generate_pause_template() {
        use std::path::PathBuf;

        let (rgb_data, width, height) = load_test_image("battle-pause.png");

        // ROI: x=38-62%, y=13-19% (PAUSE dialog header)
        let roi_x_start = (width as f32 * 0.38) as u32;
        let roi_x_end = (width as f32 * 0.62) as u32;
        let roi_y_start = (height as f32 * 0.13) as u32;
        let roi_y_end = (height as f32 * 0.19) as u32;

        let roi_w = roi_x_end - roi_x_start;
        let roi_h = roi_y_end - roi_y_start;

        // Extract ROI and convert to grayscale
        let mut gray_data = Vec::with_capacity((roi_w * roi_h) as usize);
        for y in roi_y_start..roi_y_end {
            for x in roi_x_start..roi_x_end {
                let idx = ((y * width + x) * 3) as usize;
                let r = rgb_data[idx] as f32;
                let g = rgb_data[idx + 1] as f32;
                let b = rgb_data[idx + 2] as f32;
                let lum = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
                gray_data.push(lum);
            }
        }

        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let templates_dir = PathBuf::from(manifest_dir).join("src/core/templates");
        let _ = std::fs::create_dir_all(&templates_dir);

        // Save template
        let template_path = templates_dir.join("pause.png");
        image::save_buffer(
            &template_path,
            &gray_data,
            roi_w,
            roi_h,
            image::ColorType::L8,
        )
        .expect("Failed to save pause template");
        println!("Saved pause template to {:?} ({}x{})", template_path, roi_w, roi_h);

        // Also save to output/ for visual inspection
        let output_dir = PathBuf::from(manifest_dir).parent().unwrap().join("output");
        let _ = std::fs::create_dir_all(&output_dir);
        let output_path = output_dir.join("pause_template.png");
        image::save_buffer(
            &output_path,
            &gray_data,
            roi_w,
            roi_h,
            image::ColorType::L8,
        )
        .expect("Failed to save pause template to output");
        println!("Saved visual copy to {:?}", output_path);
    }

    #[test]
    fn test_min_battle_active() {
        let (rgb_data, width, height) = load_test_image("min-battle-active.png");
        let state = analyze_battle_state(&rgb_data, width, height);
        assert_eq!(state, BattleState::Active,
            "Expected Active for min-battle-active.png ({}x{}), got {:?}", width, height, state);
    }

    #[test]
    fn test_min_battle_paused() {
        let (rgb_data, width, height) = load_test_image("min-battle-pause.png");
        let state = analyze_battle_state(&rgb_data, width, height);
        assert_eq!(state, BattleState::Paused,
            "Expected Paused for min-battle-pause.png ({}x{}), got {:?}", width, height, state);
    }

    #[test]
    fn test_min_battle_slow() {
        let (rgb_data, width, height) = load_test_image("min-battle-slow.png");
        let state = analyze_battle_state(&rgb_data, width, height);
        assert!(state == BattleState::Paused || state == BattleState::Slow,
            "Expected Paused or Slow for min-battle-slow.png ({}x{}), got {:?}", width, height, state);
    }

    #[test]
    fn test_min_home_inactive() {
        let (rgb_data, width, height) = load_test_image("min-home.png");
        let state = analyze_battle_state(&rgb_data, width, height);
        assert_eq!(state, BattleState::Inactive,
            "Expected Inactive for min-home.png ({}x{}), got {:?}", width, height, state);
    }

    #[test]
    fn test_min_home_noui_inactive() {
        let (rgb_data, width, height) = load_test_image("min-home-noui.png");
        let state = analyze_battle_state(&rgb_data, width, height);
        assert_eq!(state, BattleState::Inactive,
            "Expected Inactive for min-home-noui.png ({}x{}), got {:?}", width, height, state);
    }
}
