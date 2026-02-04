use crate::core::combat_intel::BattleState;

use crate::state::DebugState;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager};
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
pub struct InterceptorManager {
    pub is_capturing: Arc<Mutex<bool>>,
}

impl InterceptorManager {
    pub fn new() -> Self {
        Self {
            is_capturing: Arc::new(Mutex::new(false)),
        }
    }

    pub fn start_intercept(&self, hwnd: usize) {
        println!("Starting intercept on HWND: {}", hwnd);
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
    window_width: u32,
    window_height: u32,
}

// Status payload sent to Frontend
#[derive(Clone, serde::Serialize)]
struct CaptureStatus {
    window_width: u32,
    window_height: u32,
    battle_state: String, // "Active", "Paused", "Inactive"
    fps: f64,
}

// Handler for the capture event loop (Producer)
struct CaptureHandler {
    last_frame_time: Instant,
    // Conflating Queue: Mutex holds the *latest* frame. Condvar notifies consumer.
    latest_frame: Arc<(Mutex<Option<CapturePayload>>, Condvar)>,
    hwnd: usize,
    // FPS Counter for debugging
    fps_counter_received: u64, // Total frames received from OS
    fps_counter_accepted: u64, // Frames passed to worker (after throttle)
    fps_queue_full: u64,       // Times queue was full (worker busy)
    fps_last_log_time: Instant,
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

        // Spawn the Worker Thread (Consumer)
        thread::spawn(move || {
            println!("Worker Thread (Consumer) started");
            let (lock, cvar) = &*consumer_frame;

            // State Cache Mechanism
            let mut frame_count: u64 = 0;
            let mut cached_battle_state = BattleState::Inactive;

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
                let _now = Instant::now();

                // 0. Status Update (1Hz)
                if frame_count % 30 == 0 {
                    let status = CaptureStatus {
                        window_width: payload.window_width,
                        window_height: payload.window_height,
                        battle_state: format!("{:?}", cached_battle_state),
                        fps: 30.0, // Placeholder, calculated properly elsewhere or static limit
                    };
                    let _ = app_handle.emit("capture-status", &status);
                }
                frame_count += 1;

                // 1. Check for Screenshot Request
                {
                    let mut should_screenshot = debug_state.should_screenshot.lock().unwrap();
                    if *should_screenshot {
                        *should_screenshot = false;
                        println!("Visual Intercept: Screenshot requested!");
                        let output_dir = "output";
                        let _ = std::fs::create_dir_all(output_dir);
                        let path = format!("{}/debug_screenshot.png", output_dir);
                        if let Some(img_buffer) =
                            image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(
                                width,
                                height,
                                rgb_data.clone(),
                            )
                        {
                            let _ = img_buffer.save(&path);
                            println!("Screenshot saved to {}", path);
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
                        let _ = std::fs::create_dir_all("output");

                        let filename = if let Some(timer) =
                            crate::core::countdown_monitor::recognize_timer(
                                &rgb_data, width, height,
                            ) {
                            format!(
                                "output/binary_{:02}-{:02}-{:03}.png",
                                timer.minutes, timer.seconds, timer.milliseconds
                            )
                        } else {
                            format!(
                                "output/binary_failed_{}.png",
                                std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap()
                                    .as_millis()
                            )
                        };

                        let _ = image::save_buffer(
                            &filename,
                            &binary,
                            proc_w,
                            proc_h,
                            image::ColorType::L8,
                        );
                        println!("Saved binary image to {}", filename);
                        *save_binary = false;
                    }
                }

                // 3. Battle State Analysis (SKIPPED due to Optimization)
                // We only have the Timer ROI, so we cannot analyze full-screen UI (buttons etc).
                // We assume BattleState::Active to ensure Timer OCR always runs.
                let battle_state = BattleState::Active;
                cached_battle_state = battle_state;

                // 4. Timer OCR
                // Use specialized function for pre-cropped ROI to avoid double-cropping
                if let Some(timer) = crate::core::countdown_monitor::recognize_timer_from_roi(
                    &rgb_data, width, height,
                ) {
                    let _ = app_handle.emit("timer-update", &timer);
                }
                // 5. MJPEG Streaming - REMOVED
            }
            // The loop is infinite, so this line is unreachable unless the thread panics or is explicitly stopped.
            // For a graceful shutdown, a shared atomic flag or another Condvar signal would be needed.
            // println!("Worker Thread finished.");
        });

        Ok(Self {
            last_frame_time: Instant::now(),
            latest_frame,
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
            let total_push = self.fps_counter_accepted;
            println!(
                "[FPS] Received: {:.1}, Accepted: {:.1}, QueueFull: {}/{}",
                received_fps, accepted_fps, queue_full, total_push
            );
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
                // OPTIMIZATION: Copy only Timer ROI instead of full frame
                // Timer ROI: x 85%-93%, y 3.5%-6.3% (from countdown_monitor.rs)
                let roi_x_start = (full_width as f32 * 0.85) as u32;
                let roi_x_end = (full_width as f32 * 0.93) as u32;
                let roi_y_start = (full_height as f32 * 0.035) as u32;
                let roi_y_end = (full_height as f32 * 0.063) as u32;

                let roi_width = roi_x_end.saturating_sub(roi_x_start);
                let roi_height = roi_y_end.saturating_sub(roi_y_start);

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

                    let payload = CapturePayload {
                        rgb_data,
                        roi_width: roi_width,
                        roi_height: roi_height,
                        window_width: full_width,
                        window_height: full_height,
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

pub fn start_capture(
    app_handle: AppHandle,
    hwnd: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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
        CaptureFlags { app_handle, hwnd },
    );

    CaptureHandler::start(settings)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
}
