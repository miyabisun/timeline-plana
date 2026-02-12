use crate::core::combat_intel::BattleState;

use crate::state::DebugState;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};
use windows::Win32::Foundation::{HWND, POINT, RECT};
use windows::Win32::Graphics::Dwm::{DwmGetWindowAttribute, DWMWA_EXTENDED_FRAME_BOUNDS};
use windows::Win32::Graphics::Gdi::ClientToScreen;
use windows::Win32::UI::WindowsAndMessaging::GetClientRect;
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
    /// PAUSE ROI (pre-cropped RGBA for consumer-side NCC matching)
    pause_roi_rgba: Vec<u8>,
    pause_roi_width: u32,
    pause_roi_height: u32,
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
    // Debug state for full screenshot (Producer needs direct access to full frame)
    debug_state: Arc<DebugState>,
    hwnd: usize,
    // FPS Counter for debugging
    fps_counter_received: u64, // Total frames received from OS
    fps_counter_accepted: u64, // Frames passed to worker (after throttle)
    fps_queue_full: u64,       // Times queue was full (worker busy)
    fps_last_log_time: Instant,
    // Crop debug: log once on first accepted frame
    first_crop_logged: bool,
    // Reusable buffer for software crop (avoids per-frame allocation)
    crop_buffer: Vec<u8>,
    // Cached content bounds (recomputed every ~5s instead of every frame)
    cached_content_bounds: Option<(u32, u32)>,
    content_bounds_counter: u64,
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
        let producer_debug_state = debug_state.clone();

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

                // 1. Check for Binary Debug Request
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
                } else if logic_state.current_battle_state != BattleState::Inactive {
                    // Paused/Slow transitions only allowed from Active/Paused/Slow.
                    // Prevents home screen (bright but not battle) from triggering Paused.
                    // NCC template matching (only when needed, skipped during Active)
                    let pause_score =
                        crate::core::combat_intel::compute_pause_ncc_from_roi(
                            &payload.pause_roi_rgba,
                            payload.pause_roi_width,
                            payload.pause_roi_height,
                        );
                    let brightness = payload.center_brightness;
                    if pause_score > crate::core::combat_intel::PAUSE_NCC_THRESHOLD {
                        // PAUSE dialog template matched → definitively Paused
                        logic_state.last_active_time = Instant::now();
                        logic_state.current_battle_state = BattleState::Paused;
                    } else if brightness > 150.0 {
                        // Bright center = PAUSE menu visible
                        logic_state.last_active_time = Instant::now();
                        logic_state.current_battle_state = BattleState::Paused;
                    } else if brightness < 100.0 {
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
                    // Capture is 85.49%-100% (Width = 14.51%)
                    // Timer is 85.49%-93.52% (Width = 8.03%)
                    let timer_cols = (width as f32 * (0.0803 / 0.1451)) as u32;
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

                // 5. Shittim Link Sync
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
                    60.0, // Game window capture rate (WGC 60 FPS)
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
            debug_state: producer_debug_state,
            hwnd: ctx.flags.hwnd,
            fps_counter_received: 0,
            fps_counter_accepted: 0,
            fps_queue_full: 0,
            fps_last_log_time: Instant::now(),
            first_crop_logged: false,
            crop_buffer: Vec::new(),
            cached_content_bounds: None,
            content_bounds_counter: 0,
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

        // Process at ~30 FPS (33ms). WGC delivers at 60 FPS (16ms) so fresh
        // frames are always available. Consumer has 33ms to finish before the
        // next ROI arrives, keeping frontend updates low-latency.
        if now.duration_since(self.last_frame_time) < Duration::from_millis(33) {
            return Ok(());
        }
        self.last_frame_time = now;
        self.fps_counter_accepted += 1;

        // Get window geometry for crop calculation and overlay positioning.
        // DwmGetWindowAttribute(DWMWA_EXTENDED_FRAME_BOUNDS) returns the visible
        // window rect that matches the Graphics Capture API frame (excluding
        // invisible DWM shadow borders that GetWindowRect incorrectly includes).
        let mut dwm_rect = RECT::default();
        let mut client_rect = RECT::default();
        let mut client_point = POINT { x: 0, y: 0 };
        let dwm_ok;
        unsafe {
            let hwnd = HWND(self.hwnd as *mut std::ffi::c_void);
            dwm_ok = DwmGetWindowAttribute(
                hwnd,
                DWMWA_EXTENDED_FRAME_BOUNDS,
                &mut dwm_rect as *mut _ as *mut std::ffi::c_void,
                std::mem::size_of::<RECT>() as u32,
            )
            .is_ok();
            let _ = GetClientRect(hwnd, &mut client_rect);
            let _ = ClientToScreen(hwnd, &mut client_point);
        }

        // Calculate client area offset within capture frame.
        // If DWM is unavailable (remote desktop, DWM disabled), fall back to
        // no crop — full frame is processed as-is.
        let (offset_x, offset_y) = if dwm_ok {
            (
                (client_point.x - dwm_rect.left).max(0) as u32,
                (client_point.y - dwm_rect.top).max(0) as u32,
            )
        } else {
            (0, 0)
        };
        let client_width = client_rect.right as u32;
        let client_height = client_rect.bottom as u32;

        // Always use frame.buffer() — proven stable. Software crop below
        // extracts client area safely (avoids D3D11 buffer_crop issues).
        let buffer_result = frame.buffer();

        if let Ok(mut buffer) = buffer_result {
            let frame_width = buffer.width();
            let frame_height = buffer.height();

            if !self.first_crop_logged {
                self.first_crop_logged = true;
                println!(
                    "[Crop Debug] frame={}x{}, dwm=({},{},{},{}), client_pt=({},{}), client={}x{}, offset=({},{})",
                    frame_width, frame_height,
                    dwm_rect.left, dwm_rect.top, dwm_rect.right, dwm_rect.bottom,
                    client_point.x, client_point.y,
                    client_width, client_height,
                    offset_x, offset_y,
                );
            }

            if let Ok(packed_slice) = buffer.as_nopadding_buffer() {
                // Software crop: extract client area from full frame
                // This removes the title bar and window borders safely in CPU,
                // avoiding D3D11 buffer_crop which can cause allocation issues.
                // Uses self.crop_buffer to reuse allocation across frames.
                let crop_w = client_width.min(frame_width.saturating_sub(offset_x));
                let crop_h = client_height.min(frame_height.saturating_sub(offset_y));
                let (work_data, work_width, work_height): (&[u8], u32, u32) =
                    if (offset_x > 0 || offset_y > 0) && crop_w > 0 && crop_h > 0 {
                        self.crop_buffer.clear();
                        let row_bytes = crop_w as usize * 4;
                        let mut actual_h = 0u32;
                        for y in 0..crop_h {
                            let src = (offset_y + y) as usize * frame_width as usize * 4
                                + offset_x as usize * 4;
                            let end = src + row_bytes;
                            if end <= packed_slice.len() {
                                self.crop_buffer.extend_from_slice(&packed_slice[src..end]);
                                actual_h += 1;
                            } else {
                                break;
                            }
                        }
                        (&self.crop_buffer, crop_w, actual_h)
                    } else {
                        (packed_slice, frame_width, frame_height)
                    };

                // Detect content bounds (exclude black bars from game rendering)
                // Cached: recompute every 150 frames (~5s) since black bars are static
                self.content_bounds_counter += 1;
                let (content_width, content_height) =
                    if let Some(bounds) = self.cached_content_bounds {
                        if self.content_bounds_counter % 150 == 0 {
                            let b = detect_content_bounds(&work_data, work_width, work_height, 4);
                            self.cached_content_bounds = Some(b);
                            b
                        } else {
                            bounds
                        }
                    } else {
                        let b = detect_content_bounds(&work_data, work_width, work_height, 4);
                        self.cached_content_bounds = Some(b);
                        b
                    };

                // Full screenshot: save game content area (after title bar + black bar removal)
                {
                    let mut flag = self.debug_state.should_full_screenshot.lock().unwrap();
                    if *flag {
                        *flag = false;
                        let debug_output_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                            .parent()
                            .unwrap()
                            .join("output");
                        let _ = std::fs::create_dir_all(&debug_output_dir);
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap();
                        let secs = now.as_secs();
                        let millis = now.subsec_millis();
                        let h = (secs / 3600) % 24;
                        let m = (secs / 60) % 60;
                        let s = secs % 60;
                        let path = debug_output_dir.join(format!(
                            "full_screenshot_{:02}{:02}{:02}_{:03}.png",
                            h, m, s, millis
                        ));
                        // Extract content area (RGBA → RGB) using work_data with stride = work_width
                        let mut rgb = Vec::with_capacity((content_width * content_height * 3) as usize);
                        for y in 0..content_height {
                            let row_start = (y * work_width) as usize * 4;
                            let row_end = row_start + content_width as usize * 4;
                            if row_end <= work_data.len() {
                                for chunk in work_data[row_start..row_end].chunks_exact(4) {
                                    rgb.extend_from_slice(&chunk[0..3]);
                                }
                            }
                        }
                        if let Some(img) = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(
                            content_width, content_height, rgb,
                        ) {
                            let _ = img.save(&path);
                            println!("Full screenshot saved to {:?}", path);
                        }
                    }
                }

                // Timer ROI screenshot: extract timer region directly from current frame
                {
                    let mut flag = self.debug_state.should_screenshot.lock().unwrap();
                    if *flag {
                        *flag = false;
                        let debug_output_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                            .parent()
                            .unwrap()
                            .join("output");
                        let _ = std::fs::create_dir_all(&debug_output_dir);
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap();
                        let secs = now.as_secs();
                        let millis = now.subsec_millis();
                        let h = (secs / 3600) % 24;
                        let m = (secs / 60) % 60;
                        let s = secs % 60;
                        let path = debug_output_dir.join(format!(
                            "timer_roi_{:02}{:02}{:02}_{:03}.png",
                            h, m, s, millis
                        ));
                        // Extract timer ROI (85.49%-93.52%, 3.60%-6.51%) from work_data
                        let tx_start = (content_width as f32 * 0.8549) as u32;
                        let tx_end = (content_width as f32 * 0.9352) as u32;
                        let ty_start = (content_height as f32 * 0.0360) as u32;
                        let ty_end = (content_height as f32 * 0.0651) as u32;
                        let tw = tx_end.saturating_sub(tx_start);
                        let th = ty_end.saturating_sub(ty_start);
                        if tw > 0 && th > 0 {
                            let mut timer_rgb = Vec::with_capacity((tw * th * 3) as usize);
                            for y in ty_start..ty_end {
                                let row_start = (y * work_width + tx_start) as usize * 4;
                                let row_end = (y * work_width + tx_end) as usize * 4;
                                if row_end <= work_data.len() {
                                    for chunk in work_data[row_start..row_end].chunks_exact(4) {
                                        timer_rgb.extend_from_slice(&chunk[0..3]);
                                    }
                                }
                            }
                            // Apply normalize + skew correction
                            let (norm, norm_w, norm_h) =
                                crate::core::countdown_monitor::normalize_roi(
                                    &timer_rgb, tw, th,
                                );
                            let (skewed, skew_w, skew_h) =
                                crate::core::countdown_monitor::apply_skew_correction(
                                    &norm,
                                    norm_w,
                                    norm_h,
                                    crate::core::countdown_monitor::SKEW_FACTOR,
                                );
                            if let Some(img) =
                                image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(
                                    skew_w, skew_h, skewed,
                                )
                            {
                                let _ = img.save(&path);
                                println!("Timer ROI saved to {:?}", path);
                            }
                        }
                    }
                }

                // OPTIMIZATION: Copy Header Strip (Timer + Pause Button)
                // Timer ROI: x 85.49%-100%, y 3.60%-6.51%
                // ROI percentages are content-relative (excluding black bars)
                let roi_x_start = (content_width as f32 * 0.8549) as u32;
                let roi_x_end = content_width; // 100% of content
                let roi_y_start = (content_height as f32 * 0.0360) as u32;
                let roi_y_end = (content_height as f32 * 0.0651) as u32;

                let roi_width = roi_x_end.saturating_sub(roi_x_start);
                let roi_height = roi_y_end.saturating_sub(roi_y_start);

                // Cost Gauge ROI (content-relative)
                // Content-relative: x=64.49%..89.61%, y=94.68%..96.97%
                let cost_x_start = (content_width as f32 * 0.6449) as u32;
                let cost_x_end = (content_width as f32 * 0.8961) as u32;
                let cost_y_start = (content_height as f32 * 0.9468) as u32;
                let cost_y_end = (content_height as f32 * 0.9697) as u32;

                let cost_roi_width = cost_x_end.saturating_sub(cost_x_start);
                let cost_roi_height = cost_y_end.saturating_sub(cost_y_start);

                if roi_width > 0 && roi_height > 0 {
                    let pixel_count = (roi_width * roi_height) as usize;
                    let mut rgb_data = Vec::with_capacity(pixel_count * 3);

                    // Extract only ROI pixels (stride = work_width, no padding)
                    for y in roi_y_start..roi_y_end {
                        let row_start = (y * work_width + roi_x_start) as usize * 4;
                        let row_end = (y * work_width + roi_x_end) as usize * 4;
                        if row_end <= work_data.len() {
                            for chunk in work_data[row_start..row_end].chunks_exact(4) {
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
                            let row_start = (y * work_width + cost_x_start) as usize * 4;
                            let row_end = (y * work_width + cost_x_end) as usize * 4;
                            if row_end <= work_data.len() {
                                for chunk in work_data[row_start..row_end].chunks_exact(4) {
                                    cost_rgb_data.extend_from_slice(&chunk[0..3]);
                                }
                            }
                        }
                    }

                    // Sample center brightness for Paused/Slow state detection
                    // Uses content dimensions for center point, work_width for stride
                    let center_brightness =
                        sample_center_brightness(&work_data, work_width, content_width, content_height);

                    // Extract PAUSE ROI for consumer-side NCC matching (cheap crop only)
                    use crate::core::combat_intel::{
                        PAUSE_ROI_X_START, PAUSE_ROI_X_END,
                        PAUSE_ROI_Y_START, PAUSE_ROI_Y_END,
                    };
                    let pause_x_start = (content_width as f32 * PAUSE_ROI_X_START) as u32;
                    let pause_x_end = (content_width as f32 * PAUSE_ROI_X_END) as u32;
                    let pause_y_start = (content_height as f32 * PAUSE_ROI_Y_START) as u32;
                    let pause_y_end = (content_height as f32 * PAUSE_ROI_Y_END) as u32;
                    let pause_roi_width = pause_x_end.saturating_sub(pause_x_start);
                    let pause_roi_height = pause_y_end.saturating_sub(pause_y_start);

                    let mut pause_roi_rgba = Vec::with_capacity((pause_roi_width * pause_roi_height * 4) as usize);
                    for y in pause_y_start..pause_y_end {
                        let src_start = (y * work_width + pause_x_start) as usize * 4;
                        let src_end = (y * work_width + pause_x_end) as usize * 4;
                        if src_end <= work_data.len() {
                            pause_roi_rgba.extend_from_slice(&work_data[src_start..src_end]);
                        }
                    }

                    let payload = CapturePayload {
                        rgb_data,
                        roi_width,
                        roi_height,
                        cost_rgb_data,
                        cost_roi_width,
                        cost_roi_height,
                        client_x: client_point.x,
                        client_y: client_point.y,
                        client_width: content_width,
                        client_height: content_height,
                        center_brightness,
                        pause_roi_rgba,
                        pause_roi_width,
                        pause_roi_height,
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

/// Detect content boundaries by scanning from right/bottom edges inward.
///
/// Window captures often include fixed black bars (e.g. right 16px, bottom 72px).
/// This function finds where actual content ends by sampling pixel brightness.
///
/// # Arguments
/// * `data` - Raw pixel data (RGB8 or BGRA8)
/// * `width` - Full buffer width
/// * `height` - Full buffer height
/// * `bpp` - Bytes per pixel (3 for RGB, 4 for BGRA/RGBA)
///
/// # Returns
/// `(content_width, content_height)` — dimensions of the non-black content area.
/// Content is anchored at top-left (0,0). Returns full dimensions if no black bars detected.
pub fn detect_content_bounds(data: &[u8], width: u32, height: u32, bpp: u32) -> (u32, u32) {
    const SAMPLE_COUNT: u32 = 10;
    const BRIGHTNESS_THRESHOLD: u8 = 10;

    // Scan columns from right edge inward to find content_width
    // Initialize to 0; if no content found, safety clamp to 1
    let mut content_width = 0u32;
    for x in (0..width).rev() {
        let mut has_content = false;
        let step = (height / SAMPLE_COUNT).max(1);
        for i in 0..SAMPLE_COUNT {
            let y = (i * step).min(height.saturating_sub(1));
            let idx = ((y * width + x) * bpp) as usize;
            if idx + 2 < data.len() {
                let r = data[idx];
                let g = data[idx + 1];
                let b = data[idx + 2];
                if r > BRIGHTNESS_THRESHOLD || g > BRIGHTNESS_THRESHOLD || b > BRIGHTNESS_THRESHOLD {
                    has_content = true;
                    break;
                }
            }
        }
        if has_content {
            content_width = x + 1;
            break;
        }
    }

    // Scan rows from bottom edge upward to find content_height
    let mut content_height = 0u32;
    for y in (0..height).rev() {
        let mut has_content = false;
        let step = (width / SAMPLE_COUNT).max(1);
        for i in 0..SAMPLE_COUNT {
            let x = (i * step).min(width.saturating_sub(1));
            let idx = ((y * width + x) * bpp) as usize;
            if idx + 2 < data.len() {
                let r = data[idx];
                let g = data[idx + 1];
                let b = data[idx + 2];
                if r > BRIGHTNESS_THRESHOLD || g > BRIGHTNESS_THRESHOLD || b > BRIGHTNESS_THRESHOLD {
                    has_content = true;
                    break;
                }
            }
        }
        if has_content {
            content_height = y + 1;
            break;
        }
    }

    // Safety: never return zero dimensions
    (content_width.max(1), content_height.max(1))
}

/// Sample average brightness from a 10x10 pixel region at the frame center.
///
/// Uses ITU-R BT.601 perceived brightness (0.299R + 0.587G + 0.114B).
/// Input is RGBA8 packed pixel data (4 bytes per pixel).
///
/// # Arguments
/// * `rgba_data` - RGBA8 packed pixel buffer
/// * `stride_width` - Buffer stride width (full buffer row width including black bars)
/// * `content_width` - Content area width (used to compute center point)
/// * `content_height` - Content area height (used to compute center point)
///
/// Returns average brightness (0.0–255.0), or 0.0 if the frame is too small.
pub fn sample_center_brightness(rgba_data: &[u8], stride_width: u32, content_width: u32, content_height: u32) -> f32 {
    let cx = content_width / 2;
    let cy = content_height / 2;
    let mut br_sum = 0u64;
    let mut br_count = 0u32;
    for dy in 0..10u32 {
        for dx in 0..10u32 {
            let sx = cx.saturating_sub(5) + dx;
            let sy = cy.saturating_sub(5) + dy;
            let idx = ((sy * stride_width + sx) * 4) as usize;
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
        let brightness = sample_center_brightness(&data, 100, 100, 100);
        assert!((brightness - 255.0).abs() < 1.0, "Expected ~255, got {}", brightness);
    }

    #[test]
    fn test_sample_center_brightness_black() {
        // Pure black (0,0,0) → brightness = 0
        let data = make_rgba_frame(100, 100, 0, 0, 0);
        let brightness = sample_center_brightness(&data, 100, 100, 100);
        assert!((brightness - 0.0).abs() < 0.01, "Expected ~0, got {}", brightness);
    }

    #[test]
    fn test_sample_center_brightness_gray() {
        // Mid gray (128,128,128) → brightness ≈ 128
        let data = make_rgba_frame(100, 100, 128, 128, 128);
        let brightness = sample_center_brightness(&data, 100, 100, 100);
        assert!((brightness - 128.0).abs() < 1.0, "Expected ~128, got {}", brightness);
    }

    #[test]
    fn test_sample_center_brightness_empty() {
        let brightness = sample_center_brightness(&[], 0, 0, 0);
        assert_eq!(brightness, 0.0);
    }

    #[test]
    fn test_sample_center_brightness_small_frame() {
        // 5x5 frame: center is at (2,2), sampling 10x10 but only some pixels exist
        let data = make_rgba_frame(5, 5, 200, 200, 200);
        let brightness = sample_center_brightness(&data, 5, 5, 5);
        // Should still return a value (partial sampling)
        assert!(brightness > 100.0, "Expected >100, got {}", brightness);
    }

    #[test]
    fn test_detect_content_bounds_all_screenshots() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let images = [
            "battle-active.png",
            "battle-pause.png",
            "battle-slow.png",
            "home.png",
            "home-noui.png",
            "min-battle-pause.png",
            "min-home.png",
        ];

        for filename in images {
            let path = format!("{}/tests/fixtures/screenshots/{}", manifest_dir, filename);
            let img = match image::open(&path) {
                Ok(img) => img,
                Err(_) => continue,
            };
            let rgb_img = img.to_rgb8();
            let (width, height) = rgb_img.dimensions();
            let rgb_data = rgb_img.into_raw();

            let (cw, ch) = detect_content_bounds(&rgb_data, width, height, 3);

            // Check left edge: scan first 30 columns, ALL rows (dense scan)
            let mut left_bar = 0u32;
            for x in 0..width.min(30) {
                let mut all_black = true;
                for y in 0..height {
                    let idx = ((y * width + x) * 3) as usize;
                    if idx + 2 < rgb_data.len() {
                        if rgb_data[idx] > 10 || rgb_data[idx + 1] > 10 || rgb_data[idx + 2] > 10 {
                            all_black = false;
                            break;
                        }
                    }
                }
                if all_black { left_bar = x + 1; } else { break; }
            }

            // Check top edge: scan first 30 rows, ALL columns
            let mut top_bar = 0u32;
            for y in 0..height.min(30) {
                let mut all_black = true;
                for x in 0..width {
                    let idx = ((y * width + x) * 3) as usize;
                    if idx + 2 < rgb_data.len() {
                        if rgb_data[idx] > 10 || rgb_data[idx + 1] > 10 || rgb_data[idx + 2] > 10 {
                            all_black = false;
                            break;
                        }
                    }
                }
                if all_black { top_bar = y + 1; } else { break; }
            }

            println!("{}: raw={}x{}, content={}x{}, left_bar={}px, top_bar={}px, right_bar={}px, bottom_bar={}px",
                filename, width, height, cw, ch,
                left_bar, top_bar, width - cw, height - ch);
        }
    }

    #[test]
    fn test_detect_content_bounds_no_black_bars() {
        // Image with no black bars should return full dimensions
        let data = make_rgba_frame(100, 80, 128, 128, 128);
        let (cw, ch) = detect_content_bounds(&data, 100, 80, 4);
        assert_eq!((cw, ch), (100, 80));
    }

    #[test]
    fn test_detect_content_bounds_all_black() {
        // All-black image should return (1, 1) (safety minimum)
        let data = make_rgba_frame(100, 80, 0, 0, 0);
        let (cw, ch) = detect_content_bounds(&data, 100, 80, 4);
        assert_eq!((cw, ch), (1, 1));
    }
}
