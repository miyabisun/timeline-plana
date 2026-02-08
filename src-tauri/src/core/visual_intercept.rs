use crate::core::combat_intel::BattleState;

use crate::state::DebugState;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};
use windows::Win32::Foundation::{HWND, POINT, RECT};
use windows::Win32::Graphics::Gdi::ClientToScreen;
use windows::Win32::UI::WindowsAndMessaging::{GetClientRect, GetWindowRect};
use windows_capture::{
    capture::{Context, GraphicsCaptureApiHandler},
    frame::Frame,
    graphics_capture_api::InternalCaptureControl,
    settings::{
        ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
        MinimumUpdateIntervalSettings, SecondaryWindowSettings, Settings,
    },
    window::Window,
};

// Struct to manage the capture state
use std::sync::atomic::{AtomicBool, Ordering};

// Struct to manage the capture state
pub struct InterceptorManager {
    // Shared flag to track if a capture is currently running
    pub is_running: Arc<AtomicBool>,
    // Flag to signal stop request (for future use)
    pub should_stop: Arc<AtomicBool>,
}

impl InterceptorManager {
    pub fn new() -> Self {
        Self {
            is_running: Arc::new(AtomicBool::new(false)),
            should_stop: Arc::new(AtomicBool::new(false)),
        }
    }
}

// Flags to pass AppHandle
#[derive(Clone, Debug)]
pub struct CaptureFlags {
    pub app_handle: AppHandle,
    pub hwnd: usize,
}

// Payload sent from Capture Thread (Producer) to Worker Thread (Consumer)
struct CapturePayload {
    rgb_data: Vec<u8>,
    roi_width: u32,
    roi_height: u32,
    /// Cost gauge ROI (bottom area)
    cost_rgb_data: Vec<u8>,
    cost_roi_width: u32,
    cost_roi_height: u32,
    /// Client area screen position and size
    client_x: i32,
    client_y: i32,
    client_width: u32,
    client_height: u32,
    /// Average brightness of center frame region (for Paused/Slow detection)
    center_brightness: f32,
}

// Status payload sent to Frontend
// Status payload moved to shittim_link
// #[derive(Clone, serde::Serialize)]
// struct CaptureStatus { ... }

// Handler for the capture event loop (Producer)
struct CaptureHandler {
    last_frame_time: Instant,
    // Conflating Queue: Mutex holds the *latest* frame. Condvar notifies consumer.
    latest_frame: Arc<(Mutex<Option<CapturePayload>>, Condvar)>,
    // Shared Stats: Updated by Producer, Read by Consumer
    shared_stats: Arc<Mutex<Option<crate::core::shittim_link::CaptureStats>>>,
    hwnd: usize,
    // FPS Counter for debugging
    fps_counter_received: u64, // Total frames received from OS
    fps_counter_accepted: u64, // Frames passed to worker (after throttle)
    fps_queue_full: u64,       // Times queue was full (worker busy)
    fps_last_log_time: Instant,
}

// Struct to hold state inside the logic loop to persist between iterations
struct LogicState {
    last_active_time: Instant,
    current_battle_state: BattleState,
}

impl GraphicsCaptureApiHandler for CaptureHandler {
    type Flags = CaptureFlags;
    type Error = Box<dyn std::error::Error + Send + Sync>;

    fn new(ctx: Context<Self::Flags>) -> Result<Self, Self::Error> {
        println!("Created CaptureHandler (Producer)");

        let app_handle = ctx.flags.app_handle.clone();
        let debug_state = app_handle.state::<Arc<DebugState>>().inner().clone();

        // Conflating Queue mechanism
        let latest_frame = Arc::new((Mutex::new(Option::<CapturePayload>::None), Condvar::new()));
        let consumer_frame = latest_frame.clone();

        // Shared Stats
        let shared_stats = Arc::new(Mutex::new(None));
        let consumer_stats = shared_stats.clone();

        // Spawn the Worker Thread (Consumer)
        thread::spawn(move || {
            println!("Worker Thread (Consumer) started");
            let (lock, cvar) = &*consumer_frame;

            // State Cache Mechanism
            // let mut frame_count: u64 = 0; // Unused
            // let mut cached_battle_state = BattleState::Inactive; (Replaced by LogicState)
            let mut logic_state = LogicState {
                last_active_time: Instant::now(),
                current_battle_state: BattleState::Inactive,
            };

            loop {
                // Wait for a new frame
                let payload = {
                    let mut guard = lock.lock().unwrap();
                    while guard.is_none() {
                        guard = cvar.wait(guard).unwrap();
                    }
                    guard.take().unwrap()
                };

                let rgb_data = payload.rgb_data;
                let width = payload.roi_width;
                let height = payload.roi_height;
                let cost_rgb_data = payload.cost_rgb_data;
                let cost_w = payload.cost_roi_width;
                let cost_h = payload.cost_roi_height;
                let client_x = payload.client_x;
                let client_y = payload.client_y;
                let client_width = payload.client_width;
                let client_height = payload.client_height;

                // Debug output directory: project root /output (not src-tauri/output)
                let debug_output_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .unwrap()
                    .join("output");

                // 1. Check for Screenshot Request
                {
                    let mut should_screenshot = debug_state.should_screenshot.lock().unwrap();
                    if *should_screenshot {
                        *should_screenshot = false;
                        println!("Visual Intercept: Screenshot requested!");
                        let _ = std::fs::create_dir_all(&debug_output_dir);
                        let path = debug_output_dir.join("debug_screenshot.png");
                        if let Some(img_buffer) =
                            image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(
                                width,
                                height,
                                rgb_data.clone(),
                            )
                        {
                            let _ = img_buffer.save(&path);
                            println!("Screenshot saved to {:?}", path);
                        }
                    }
                }

                // 2. Check for Binary Debug Request
                {
                    let mut save_binary = debug_state.should_save_binary.lock().unwrap();
                    if *save_binary {
                        println!("Saving binary timer image for debug...");
                        let (processed, proc_w, proc_h) =
                            crate::core::countdown_monitor::process_timer_region(
                                &rgb_data, width, height,
                            );
                        let binary = crate::core::countdown_monitor::binarize_for_ocr(
                            &processed, proc_w, proc_h,
                        );
                        let _ = std::fs::create_dir_all(&debug_output_dir);

                        let filename = if let Some(timer) =
                            crate::core::countdown_monitor::recognize_timer(
                                &rgb_data, width, height,
                            ) {
                            debug_output_dir.join(format!(
                                "binary_{:02}-{:02}-{:03}.png",
                                timer.minutes, timer.seconds, timer.milliseconds
                            ))
                        } else {
                            debug_output_dir.join(format!(
                                "binary_failed_{}.png",
                                std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap()
                                    .as_millis()
                            ))
                        };

                        let _ = image::save_buffer(
                            &filename,
                            &binary,
                            proc_w,
                            proc_h,
                            image::ColorType::L8,
                        );
                        println!("Saved binary image to {:?}", filename);
                        *save_binary = false;
                    }
                }

                // 3. Check for Cost ROI Debug Request
                {
                    let mut save_cost = debug_state.should_save_cost_roi.lock().unwrap();
                    if *save_cost {
                        *save_cost = false;
                        if cost_w > 0 && cost_h > 0 {
                            let _ = std::fs::create_dir_all(&debug_output_dir);
                            let (corrected, corr_w, corr_h) =
                                crate::core::resource_survey::apply_cost_skew(
                                    &cost_rgb_data,
                                    cost_w,
                                    cost_h,
                                    crate::core::resource_survey::COST_SKEW_FACTOR,
                                );
                            let now = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap();
                            let secs = now.as_secs();
                            let millis = now.subsec_millis();
                            let h = (secs / 3600) % 24;
                            let m = (secs / 60) % 60;
                            let s = secs % 60;
                            let filename = debug_output_dir.join(format!(
                                "cost_roi_{:02}{:02}{:02}_{:03}.png",
                                h, m, s, millis
                            ));
                            let _ = image::save_buffer(
                                &filename,
                                &corrected,
                                corr_w,
                                corr_h,
                                image::ColorType::Rgb8,
                            );
                            println!("Saved cost ROI to {:?}", filename);
                        }
                    }
                }

                // 4. Battle State Analysis
                // The ROI is now 85-100% of the screen.
                // Right part contains the Pause Button.
                let is_pause_visible = crate::core::combat_intel::check_pause_presence_in_wide_roi(
                    &rgb_data, width, height,
                );

                if is_pause_visible {
                    logic_state.last_active_time = Instant::now();
                    logic_state.current_battle_state = BattleState::Active;
                } else {
                    // Use center brightness to distinguish Paused vs Slow
                    // Same logic as combat_intel::analyze_battle_state()
                    let brightness = payload.center_brightness;
                    if brightness > 150.0 {
                        // Bright center = PAUSE menu visible
                        logic_state.last_active_time = Instant::now();
                        logic_state.current_battle_state = BattleState::Paused;
                    } else if brightness < 100.0
                        && logic_state.current_battle_state != BattleState::Inactive
                    {
                        // Dark overlay = skill confirmation / slow-motion
                        logic_state.last_active_time = Instant::now();
                        logic_state.current_battle_state = BattleState::Slow;
                    } else if logic_state.last_active_time.elapsed() > Duration::from_secs(2) {
                        // Persistence: Only switch to Inactive after 2s
                        logic_state.current_battle_state = BattleState::Inactive;
                    }
                    // Otherwise keep previous state (hysteresis)
                }

                // 4. Timer OCR & Cost Gauge Reading
                let mut current_timer: Option<crate::core::countdown_monitor::TimerValue> = None;
                let mut current_cost: Option<crate::core::resource_survey::CostValue> = None;

                if logic_state.current_battle_state != BattleState::Inactive {
                    // Timer Extraction
                    // We must SUB-CROP the Timer part to pass to countdown_monitor
                    // Capture is 85-100% (Width = 15%)
                    // Timer is 85-93% (Width = 8%)
                    // Timer width relative to capture width: 8/15
                    let timer_cols = (width as f32 * (0.08 / 0.15)) as u32;
                    let timer_cols = timer_cols.min(width); // Safety cap

                    // Extract logic (copying to new buffer for OCR API)
                    let mut timer_data = Vec::with_capacity((timer_cols * height * 3) as usize);
                    for y in 0..height {
                        let row_start = (y * width) as usize * 3;
                        let row_end = row_start + (timer_cols as usize * 3);
                        timer_data.extend_from_slice(&rgb_data[row_start..row_end]);
                    }

                    // Use specialized function for pre-cropped ROI
                    if let Some(timer) = crate::core::countdown_monitor::recognize_timer_from_roi(
                        &timer_data,
                        timer_cols,
                        height,
                    ) {
                        current_timer = Some(timer);
                    }

                    // Cost Gauge Reading (max_cost = 10 default, future: from timeline)
                    if cost_w > 0 && cost_h > 0 {
                        current_cost = crate::core::resource_survey::read_cost_gauge(
                            &cost_rgb_data,
                            cost_w,
                            cost_h,
                            10,
                        );
                    }
                }

                // 5. Shittim Link Sync (30FPS)
                // Read latest stats from producer
                let current_stats = {
                    let guard = consumer_stats.lock().unwrap();
                    guard.clone()
                };

                crate::core::shittim_link::sync_to_chest(
                    &app_handle,
                    &format!("{:?}", logic_state.current_battle_state),
                    current_timer,
                    current_cost,
                    30.0, // Stable FPS goal
                    current_stats,
                    crate::core::shittim_link::WindowGeometry {
                        x: client_x,
                        y: client_y,
                        width: client_width,
                        height: client_height,
                    },
                );
                // 5. MJPEG Streaming - REMOVED
            }
            // The loop is infinite, so this line is unreachable unless the thread panics or is explicitly stopped.
            // For a graceful shutdown, a shared atomic flag or another Condvar signal would be needed.
            // println!("Worker Thread finished.");
        });

        Ok(Self {
            last_frame_time: Instant::now(),
            latest_frame,
            shared_stats,
            hwnd: ctx.flags.hwnd,
            fps_counter_received: 0,
            fps_counter_accepted: 0,
            fps_queue_full: 0,
            fps_last_log_time: Instant::now(),
        })
    }

    fn on_frame_arrived(
        &mut self,
        frame: &mut Frame,
        _capture_control: InternalCaptureControl,
    ) -> Result<(), Self::Error> {
        let now = Instant::now();

        // FPS Logging: Every 10 seconds, output stats
        self.fps_counter_received += 1;
        if now.duration_since(self.fps_last_log_time) >= Duration::from_secs(10) {
            let elapsed = now.duration_since(self.fps_last_log_time).as_secs_f64();
            let received_fps = self.fps_counter_received as f64 / elapsed;
            let accepted_fps = self.fps_counter_accepted as f64 / elapsed;
            let queue_full = self.fps_queue_full;
            // let total_push = self.fps_counter_accepted;

            // Update shared stats instead of printing
            {
                let mut stats_guard = self.shared_stats.lock().unwrap();
                *stats_guard = Some(crate::core::shittim_link::CaptureStats {
                    received: received_fps,
                    accepted: accepted_fps,
                    queue_full: queue_full,
                });
            }

            // println!(
            //     "[FPS] Received: {:.1}, Accepted: {:.1}, QueueFull: {}/{}",
            //     received_fps, accepted_fps, queue_full, total_push
            // );
            self.fps_counter_received = 0;
            self.fps_queue_full = 0;
            self.fps_counter_accepted = 0;
            self.fps_last_log_time = now;
        }

        // Limit to approx 30 FPS at capture source (33ms)
        // We use 32ms to allow for slight jitter while safely dropping 60fps frames (16ms)
        if now.duration_since(self.last_frame_time) < Duration::from_millis(32) {
            return Ok(());
        }
        self.last_frame_time = now;
        self.fps_counter_accepted += 1;

        // Crop Logic
        // TODO: Refactor crop logic into helper if needed, but keeping here is minimal overhead compared to allocations
        let mut client_rect = RECT::default();
        let mut window_rect = RECT::default();
        let mut client_point = POINT { x: 0, y: 0 };
        let mut crop_x = 0;
        let mut crop_y = 0;
        let mut crop_w = frame.width();
        let mut crop_h = frame.height();

        unsafe {
            let hwnd = HWND(self.hwnd as *mut std::ffi::c_void);
            let _ = GetClientRect(hwnd, &mut client_rect);
            let _ = GetWindowRect(hwnd, &mut window_rect);
            let _ = ClientToScreen(hwnd, &mut client_point);

            if window_rect.right > window_rect.left && window_rect.bottom > window_rect.top {
                let offset_x = client_point.x - window_rect.left;
                let offset_y = client_point.y - window_rect.top;
                if offset_x >= 0 && offset_y >= 0 {
                    crop_x = offset_x as u32;
                    crop_y = offset_y as u32;
                    let cw = client_rect.right - client_rect.left;
                    let ch = client_rect.bottom - client_rect.top;
                    let fw = frame.width() as i32;
                    let fh = frame.height() as i32;
                    if (crop_x as i32 + cw) <= fw && (crop_y as i32 + ch) <= fh {
                        crop_w = cw as u32;
                        crop_h = ch as u32;
                    }
                }
            }
        }

        let mut buffer_result = frame.buffer_crop(crop_x, crop_y, crop_x + crop_w, crop_y + crop_h);
        if buffer_result.is_err() {
            buffer_result = frame.buffer(); // Fallback
        }

        if let Ok(mut buffer) = buffer_result {
            let full_width = buffer.width();
            let full_height = buffer.height();

            if let Ok(packed_slice) = buffer.as_nopadding_buffer() {
                // OPTIMIZATION: Copy Header Strip (Timer + Pause Button)
                // Timer ROI: x 85%-93%
                // Pause ROI: x 92%-100%
                // Combined: x 85%-100%
                let roi_x_start = (full_width as f32 * 0.85) as u32;
                let roi_x_end = full_width; // 100%
                let roi_y_start = (full_height as f32 * 0.035) as u32;
                let roi_y_end = (full_height as f32 * 0.063) as u32;

                let roi_width = roi_x_end.saturating_sub(roi_x_start);
                let roi_height = roi_y_end.saturating_sub(roi_y_start);

                // Cost Gauge ROI: x 64%-89%, y 91.0%-93.2%
                let cost_x_start = (full_width as f32 * 0.64) as u32;
                let cost_x_end = (full_width as f32 * 0.89) as u32;
                let cost_y_start = (full_height as f32 * 0.910) as u32;
                let cost_y_end = (full_height as f32 * 0.932) as u32;

                let cost_roi_width = cost_x_end.saturating_sub(cost_x_start);
                let cost_roi_height = cost_y_end.saturating_sub(cost_y_start);

                if roi_width > 0 && roi_height > 0 {
                    let pixel_count = (roi_width * roi_height) as usize;
                    let mut rgb_data = Vec::with_capacity(pixel_count * 3);

                    // Extract only ROI pixels (much smaller than full frame)
                    for y in roi_y_start..roi_y_end {
                        let row_start = (y * full_width + roi_x_start) as usize * 4;
                        let row_end = (y * full_width + roi_x_end) as usize * 4;
                        if row_end <= packed_slice.len() {
                            for chunk in packed_slice[row_start..row_end].chunks_exact(4) {
                                rgb_data.extend_from_slice(&chunk[0..3]);
                            }
                        }
                    }

                    // Extract Cost Gauge ROI
                    let mut cost_rgb_data = Vec::new();
                    if cost_roi_width > 0 && cost_roi_height > 0 {
                        let cost_pixel_count = (cost_roi_width * cost_roi_height) as usize;
                        cost_rgb_data.reserve(cost_pixel_count * 3);
                        for y in cost_y_start..cost_y_end {
                            let row_start = (y * full_width + cost_x_start) as usize * 4;
                            let row_end = (y * full_width + cost_x_end) as usize * 4;
                            if row_end <= packed_slice.len() {
                                for chunk in packed_slice[row_start..row_end].chunks_exact(4) {
                                    cost_rgb_data.extend_from_slice(&chunk[0..3]);
                                }
                            }
                        }
                    }

                    // Sample center brightness for Paused/Slow state detection
                    let center_brightness =
                        sample_center_brightness(packed_slice, full_width, full_height);

                    let payload = CapturePayload {
                        rgb_data,
                        roi_width,
                        roi_height,
                        cost_rgb_data,
                        cost_roi_width,
                        cost_roi_height,
                        client_x: client_point.x,
                        client_y: client_point.y,
                        client_width: full_width,
                        client_height: full_height,
                        center_brightness,
                    };

                    // Conflating Push: Overwrite any pending frame with the latest one
                    let (lock, cvar) = &*self.latest_frame;
                    let mut guard = lock.lock().unwrap();
                    if guard.is_some() {
                        self.fps_queue_full += 1; // Worker was busy, frame overwritten
                    }
                    *guard = Some(payload);
                    cvar.notify_one();
                }
            } else {
                eprintln!("Failed to get nopadding buffer");
            }
        } else {
            eprintln!("Failed to capture buffer");
        }

        Ok(())
    }

    fn on_closed(&mut self) -> Result<(), Self::Error> {
        println!("Capture session closed");
        Ok(())
    }
}

/// Sample average brightness from a 10x10 pixel region at the frame center.
///
/// Uses ITU-R BT.601 perceived brightness (0.299R + 0.587G + 0.114B).
/// Input is RGBA8 packed pixel data (4 bytes per pixel).
///
/// Returns average brightness (0.0–255.0), or 0.0 if the frame is too small.
pub fn sample_center_brightness(rgba_data: &[u8], width: u32, height: u32) -> f32 {
    let cx = width / 2;
    let cy = height / 2;
    let mut br_sum = 0u64;
    let mut br_count = 0u32;
    for dy in 0..10u32 {
        for dx in 0..10u32 {
            let sx = cx.saturating_sub(5) + dx;
            let sy = cy.saturating_sub(5) + dy;
            let idx = ((sy * width + sx) * 4) as usize;
            if idx + 2 < rgba_data.len() {
                br_sum += (299 * rgba_data[idx] as u64
                    + 587 * rgba_data[idx + 1] as u64
                    + 114 * rgba_data[idx + 2] as u64)
                    / 1000;
                br_count += 1;
            }
        }
    }
    if br_count > 0 {
        br_sum as f32 / br_count as f32
    } else {
        0.0
    }
}

pub fn start_capture(
    app_handle: AppHandle,
    hwnd: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let manager = app_handle.state::<InterceptorManager>();

    // Check if already running
    if manager.is_running.load(Ordering::SeqCst) {
        println!("Capture already running. Ignoring new request.");
        return Ok(());
    }

    // Set running flag
    manager.is_running.store(true, Ordering::SeqCst);
    manager.should_stop.store(false, Ordering::SeqCst);

    let window = Window::from_raw_hwnd(hwnd as *mut std::ffi::c_void);
    let settings = Settings::new(
        window,
        CursorCaptureSettings::Default,
        DrawBorderSettings::Default,
        SecondaryWindowSettings::Default,
        // Request 60 FPS from Windows Graphics Capture API (16ms interval)
        MinimumUpdateIntervalSettings::Custom(Duration::from_millis(16)),
        DirtyRegionSettings::Default,
        ColorFormat::Rgba8,
        CaptureFlags {
            app_handle: app_handle.clone(),
            hwnd,
        },
    );

    // Run the capture (BLOCKING)
    let result = CaptureHandler::start(settings);

    // Reset running flag when finished
    manager.is_running.store(false, Ordering::SeqCst);
    println!("Capture session ended.");

    result.map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create an RGBA8 buffer filled with a single color
    fn make_rgba_frame(width: u32, height: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let pixel_count = (width * height) as usize;
        let mut data = Vec::with_capacity(pixel_count * 4);
        for _ in 0..pixel_count {
            data.extend_from_slice(&[r, g, b, 255]);
        }
        data
    }

    #[test]
    fn test_sample_center_brightness_white() {
        // Pure white (255,255,255) → brightness = 255
        let data = make_rgba_frame(100, 100, 255, 255, 255);
        let brightness = sample_center_brightness(&data, 100, 100);
        assert!((brightness - 255.0).abs() < 1.0, "Expected ~255, got {}", brightness);
    }

    #[test]
    fn test_sample_center_brightness_black() {
        // Pure black (0,0,0) → brightness = 0
        let data = make_rgba_frame(100, 100, 0, 0, 0);
        let brightness = sample_center_brightness(&data, 100, 100);
        assert!((brightness - 0.0).abs() < 0.01, "Expected ~0, got {}", brightness);
    }

    #[test]
    fn test_sample_center_brightness_gray() {
        // Mid gray (128,128,128) → brightness ≈ 128
        let data = make_rgba_frame(100, 100, 128, 128, 128);
        let brightness = sample_center_brightness(&data, 100, 100);
        assert!((brightness - 128.0).abs() < 1.0, "Expected ~128, got {}", brightness);
    }

    #[test]
    fn test_sample_center_brightness_empty() {
        let brightness = sample_center_brightness(&[], 0, 0);
        assert_eq!(brightness, 0.0);
    }

    #[test]
    fn test_sample_center_brightness_small_frame() {
        // 5x5 frame: center is at (2,2), sampling 10x10 but only some pixels exist
        let data = make_rgba_frame(5, 5, 200, 200, 200);
        let brightness = sample_center_brightness(&data, 5, 5);
        // Should still return a value (partial sampling)
        assert!(brightness > 100.0, "Expected >100, got {}", brightness);
    }
}
