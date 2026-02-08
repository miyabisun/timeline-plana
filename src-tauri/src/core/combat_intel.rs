//! Combat Intelligence (戦闘情報分析)
//!
//! This module analyzes captured frames to determine the current battle state.
//! It acts as Plana's "eyes" - interpreting visual data to understand if battle
//! is active, paused, or in slow-motion (skill confirmation).

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
    let (_blue_ratio, white_ratio) = check_pause_button_debug(rgb_data, width, height);
    let brightness = calculate_center_brightness(rgb_data, width, height);

    // Detection logic based on observed diagnostic values:
    // - Active: white_ratio ~0.21 (timer text visible and bright)
    // - Paused: white_ratio ~0, brightness ~203 (PAUSE menu is white, blocks timer)
    // - Slow: white_ratio ~0, brightness ~59 (dark overlay for skill confirmation)

    if white_ratio > 0.05 {
        // Timer and battle UI clearly visible → Active battle
        BattleState::Active
    } else if brightness > 150.0 {
        // Timer blocked but center is bright (PAUSE menu visible) → Paused
        BattleState::Paused
    } else if brightness < 100.0 {
        // Dark overlay (skill confirmation or slow-mo) → Slow
        BattleState::Slow
    } else {
        // In between - could be transitioning or unknown state
        // Default to Inactive to be safe
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
        }
    }

    if total_pixels == 0 {
        return false;
    }

    let white_ratio = white_pixel_count as f32 / total_pixels as f32;

    // Threshold: Active battle has clear white pixels (> 5%) for the button icon
    white_ratio > 0.05
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

    /// Load a test image from the tests/fixtures directory and convert to RGB8 data.
    fn load_test_image(filename: &str) -> (Vec<u8>, u32, u32) {
        // CARGO_MANIFEST_DIR points to src-tauri during cargo test
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
        assert!(
            state == BattleState::Paused || state == BattleState::Slow,
            "Expected Paused or Slow state for battle-pause.png, got {:?}",
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

        let images = ["battle-active.png", "battle-pause.png", "battle-slow.png"];
        for img_name in images {
            let (rgb_data, width, height) = load_test_image(img_name);
            let brightness = calculate_center_brightness(&rgb_data, width, height);
            let pause_visible = check_pause_button_visible(&rgb_data, width, height);
            let state = analyze_battle_state(&rgb_data, width, height);

            println!("Image: {} ({}x{})", img_name, width, height);
            let (blue_ratio, white_ratio) = check_pause_button_debug(&rgb_data, width, height);
            println!(
                "  Blue ratio: {:.4}, White ratio: {:.4}",
                blue_ratio, white_ratio
            );
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
}
