use crate::core::mjpeg::MjpegState;
use image::codecs::jpeg::JpegEncoder;
use image::ColorType;
use std::sync::{Arc, Mutex};
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
        // Todo: Implement actual background thread handling for capture
        println!("Starting intercept on HWND: {}", hwnd);
    }
}

// Flags to pass AppHandle
#[derive(Clone, Debug)]
pub struct CaptureFlags {
    pub app_handle: AppHandle,
    pub hwnd: usize,
}

// Handler for the capture event loop
struct CaptureHandler {
    last_frame_time: Instant,
    last_stream_time: Instant,
    app_handle: AppHandle,
    hwnd: usize,
    mjpeg_state: Arc<MjpegState>,
}

impl GraphicsCaptureApiHandler for CaptureHandler {
    type Flags = CaptureFlags;
    type Error = Box<dyn std::error::Error + Send + Sync>;

    fn new(ctx: Context<Self::Flags>) -> Result<Self, Self::Error> {
        println!("Created CaptureHandler for streaming");
        let mjpeg_state = ctx
            .flags
            .app_handle
            .state::<Arc<MjpegState>>()
            .inner()
            .clone();
        Ok(Self {
            last_frame_time: Instant::now(),
            last_stream_time: Instant::now(),
            app_handle: ctx.flags.app_handle,
            hwnd: ctx.flags.hwnd,
            mjpeg_state,
        })
    }

    fn on_frame_arrived(
        &mut self,
        frame: &mut Frame,
        _capture_control: InternalCaptureControl,
    ) -> Result<(), Self::Error> {
        // println!("Frame arrived!"); // Uncomment for verbose spam
        let now = Instant::now();
        // Limit to approx 30 FPS (33ms)
        if now.duration_since(self.last_frame_time) < Duration::from_millis(33) {
            return Ok(());
        }
        self.last_frame_time = now;

        // println!("Visual Intercept: Frame captured at {:?}", now);

        // Get dimensions first to avoid borrow conflict
        // Get frame buffer (automatically crop title bar if applicable)
        // Calculate trim offsets using Win32 API
        // This is more robust than windows-capture's built-in title bar detection
        let mut client_rect = RECT::default();
        let mut window_rect = RECT::default();
        let mut client_point = POINT { x: 0, y: 0 };

        let mut crop_x = 0;
        let mut crop_y = 0;
        let mut crop_w = frame.width();
        let mut crop_h = frame.height();

        unsafe {
            let hwnd = HWND(self.hwnd as *mut std::ffi::c_void);
            // We use the HWND to determine the actual content area
            let _ = GetClientRect(hwnd, &mut client_rect);
            let _ = GetWindowRect(hwnd, &mut window_rect);
            // Map (0,0) of client area to screen coordinates
            let _ = ClientToScreen(hwnd, &mut client_point);

            if window_rect.right > window_rect.left && window_rect.bottom > window_rect.top {
                let client_screen_x = client_point.x;
                let client_screen_y = client_point.y;

                let window_screen_x = window_rect.left;
                let window_screen_y = window_rect.top;

                // Offset relative to the Window Rect (which is what frame usually captures)
                let offset_x = client_screen_x - window_screen_x;
                let offset_y = client_screen_y - window_screen_y;

                if offset_x >= 0 && offset_y >= 0 {
                    crop_x = offset_x as u32;
                    crop_y = offset_y as u32;

                    let fw = frame.width() as i32;
                    let fh = frame.height() as i32;

                    let cw = client_rect.right - client_rect.left;
                    let ch = client_rect.bottom - client_rect.top;

                    // Verify bounds
                    if (crop_x as i32 + cw) <= fw && (crop_y as i32 + ch) <= fh {
                        crop_w = cw as u32;
                        crop_h = ch as u32;
                    }
                }
            }
        }

        // Apply crop
        let mut buffer_result = frame.buffer_crop(crop_x, crop_y, crop_x + crop_w, crop_y + crop_h);

        // Fallback
        if buffer_result.is_err() {
            buffer_result = frame.buffer();
        }

        if let Ok(mut buffer) = buffer_result {
            // Get dimensions from the cropped buffer
            let width = buffer.width();
            let height = buffer.height();

            if let Ok(packed_slice) = buffer.as_nopadding_buffer() {
                // Convert RGBA to RGB (JPEG doesn't support Alpha)
                let pixel_count = (width * height) as usize;
                let mut rgb_data = Vec::with_capacity(pixel_count * 3);

                for chunk in packed_slice.chunks_exact(4) {
                    rgb_data.extend_from_slice(&chunk[0..3]);
                }

                // Stream at 15 FPS (66ms)
                if now.duration_since(self.last_stream_time) >= Duration::from_millis(66) {
                    self.last_stream_time = now;
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut encoder = JpegEncoder::new_with_quality(&mut bytes, 95);
                    match encoder.encode(&rgb_data, width, height, ColorType::Rgb8.into()) {
                        Ok(_) => {
                            self.mjpeg_state.update_frame(bytes);
                        }
                        Err(e) => eprintln!("Failed to encode frame: {}", e),
                    }
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
        MinimumUpdateIntervalSettings::Default,
        DirtyRegionSettings::Default,
        ColorFormat::Rgba8,
        CaptureFlags { app_handle, hwnd },
    );

    // This will block the thread if run directly
    CaptureHandler::start(settings)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
}
