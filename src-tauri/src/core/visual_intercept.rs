use crate::core::combat_intel::BattleState;
use crate::core::shittim_link::{sync_to_chest, ShittimPayload, WindowGeometry};
use crate::core::tactical_guidance::Observation;

use crate::state::{DebugState, GuideState};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
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

impl Default for InterceptorManager {
    fn default() -> Self {
        Self::new()
    }
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

impl LogicState {
    /// Moves the battle state on from what the frame in `payload` shows.
    fn update(&mut self, payload: &CapturePayload) {
        use crate::core::combat_intel::{
            check_pause_presence_in_wide_roi, compute_pause_ncc_from_roi, PAUSE_NCC_THRESHOLD,
        };
        // The ROI is now 85-100% of the screen.
        // Right part contains the Pause Button.
        let is_pause_visible = check_pause_presence_in_wide_roi(
            &payload.rgb_data,
            payload.roi_width,
            payload.roi_height,
        );

        if is_pause_visible {
            self.last_active_time = Instant::now();
            self.current_battle_state = BattleState::Active;
        } else if self.current_battle_state != BattleState::Inactive {
            // Paused/Slow transitions only allowed from Active/Paused/Slow.
            // Prevents home screen (bright but not battle) from triggering Paused.
            // NCC template matching (only when needed, skipped during Active)
            let pause_score = compute_pause_ncc_from_roi(
                &payload.pause_roi_rgba,
                payload.pause_roi_width,
                payload.pause_roi_height,
            );
            let brightness = payload.center_brightness;
            if pause_score > PAUSE_NCC_THRESHOLD {
                // PAUSE dialog template matched → definitively Paused
                self.last_active_time = Instant::now();
                self.current_battle_state = BattleState::Paused;
            } else if brightness > 150.0 {
                // Bright center = PAUSE menu visible
                self.last_active_time = Instant::now();
                self.current_battle_state = BattleState::Paused;
            } else if brightness < 100.0 {
                // Dark overlay = skill confirmation / slow-motion
                self.last_active_time = Instant::now();
                self.current_battle_state = BattleState::Slow;
            } else if self.last_active_time.elapsed() > Duration::from_secs(2) {
                // Persistence: Only switch to Inactive after 2s
                self.current_battle_state = BattleState::Inactive;
            }
            // Otherwise keep previous state (hysteresis)
        }
    }
}

/// A rectangle of the content area in pixels: columns `left..right`, rows `top..bottom`.
#[derive(Clone, Copy)]
struct Region {
    left: u32,
    right: u32,
    top: u32,
    bottom: u32,
}

impl Region {
    /// The part of a `width` x `height` area between the given fractions of it.
    fn of(width: u32, height: u32, (left, right): (f32, f32), (top, bottom): (f32, f32)) -> Self {
        Self {
            left: (width as f32 * left) as u32,
            right: (width as f32 * right) as u32,
            top: (height as f32 * top) as u32,
            bottom: (height as f32 * bottom) as u32,
        }
    }

    fn width(self) -> u32 {
        self.right.saturating_sub(self.left)
    }

    fn height(self) -> u32 {
        self.bottom.saturating_sub(self.top)
    }
}

/// The RGB pixels of `region` in an RGBA buffer whose rows are `stride` pixels long. Rows
/// that run past the end of the buffer are left out.
fn crop_rgb(rgba: &[u8], stride: u32, region: Region) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((region.width() * region.height() * 3) as usize);
    for y in region.top..region.bottom {
        let row_start = (y * stride + region.left) as usize * 4;
        let row_end = (y * stride + region.right) as usize * 4;
        if row_end <= rgba.len() {
            for chunk in rgba[row_start..row_end].chunks_exact(4) {
                rgb.extend_from_slice(&chunk[0..3]);
            }
        }
    }
    rgb
}

/// The folder the debug images go to: the project root's `output` (not src-tauri/output).
fn debug_output_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate folder src-tauri sits inside the project folder")
        .join("output")
}

/// Time since the Unix epoch; zero if the clock is set before it.
fn since_epoch() -> Duration {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
}

/// The time of day (UTC) as `HHMMSS_mmm`, which names the debug images.
fn clock_stamp() -> String {
    let now = since_epoch();
    let secs = now.as_secs();
    let h = (secs / 3600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    let millis = now.subsec_millis();
    format!("{h:02}{m:02}{s:02}_{millis:03}")
}

/// Waits for the producer's next frame and takes it.
fn next_payload((lock, cvar): &(Mutex<Option<CapturePayload>>, Condvar)) -> CapturePayload {
    // A panic elsewhere while holding the lock leaves at worst a stale frame, which the
    // next one replaces, so a poisoned lock is used as it is.
    let mut guard = lock.lock().unwrap_or_else(PoisonError::into_inner);
    loop {
        if let Some(payload) = guard.take() {
            return payload;
        }
        guard = cvar.wait(guard).unwrap_or_else(PoisonError::into_inner);
    }
}

/// Saves the binarized timer of the frame as `binary_MM-SS-mmm.png`, or
/// `binary_failed_<ms since epoch>.png` when it cannot be read.
fn save_binary_timer(payload: &CapturePayload) {
    use crate::core::countdown_monitor::{binarize_for_ocr, process_timer_region, recognize_timer};
    println!("Saving binary timer image for debug...");
    let (rgb_data, width, height) = (&payload.rgb_data, payload.roi_width, payload.roi_height);
    let (processed, proc_w, proc_h) = process_timer_region(rgb_data, width, height);
    let binary = binarize_for_ocr(&processed, proc_w, proc_h);
    let debug_output_dir = debug_output_dir();
    let _ = std::fs::create_dir_all(&debug_output_dir);

    let filename = if let Some(timer) = recognize_timer(rgb_data, width, height) {
        debug_output_dir.join(format!(
            "binary_{:02}-{:02}-{:03}.png",
            timer.minutes, timer.seconds, timer.milliseconds
        ))
    } else {
        debug_output_dir.join(format!("binary_failed_{}.png", since_epoch().as_millis()))
    };

    let _ = image::save_buffer(&filename, &binary, proc_w, proc_h, image::ColorType::L8);
    println!("Saved binary image to {}", filename.display());
}

/// Saves the skew-corrected cost gauge of the frame as `cost_roi_<time>.png`.
fn save_cost_roi(payload: &CapturePayload) {
    use crate::core::resource_survey::{apply_cost_skew, COST_SKEW_FACTOR};
    let (cost_w, cost_h) = (payload.cost_roi_width, payload.cost_roi_height);
    if cost_w == 0 || cost_h == 0 {
        return;
    }
    let debug_output_dir = debug_output_dir();
    let _ = std::fs::create_dir_all(&debug_output_dir);
    let (corrected, corr_w, corr_h) =
        apply_cost_skew(&payload.cost_rgb_data, cost_w, cost_h, COST_SKEW_FACTOR);
    let filename = debug_output_dir.join(format!("cost_roi_{}.png", clock_stamp()));
    let _ = image::save_buffer(
        &filename,
        &corrected,
        corr_w,
        corr_h,
        image::ColorType::Rgb8,
    );
    println!("Saved cost ROI to {}", filename.display());
}

/// Reads the timer and the cost gauge of the frame.
fn read_timer_and_cost(
    payload: &CapturePayload,
) -> (
    Option<crate::core::countdown_monitor::TimerValue>,
    Option<crate::core::resource_survey::CostValue>,
) {
    let (width, height) = (payload.roi_width, payload.roi_height);
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
        timer_data.extend_from_slice(&payload.rgb_data[row_start..row_end]);
    }

    // Use specialized function for pre-cropped ROI
    let timer =
        crate::core::countdown_monitor::recognize_timer_from_roi(&timer_data, timer_cols, height);

    // Cost Gauge Reading (max_cost = 10 default, future: from timeline)
    let cost = if payload.cost_roi_width > 0 && payload.cost_roi_height > 0 {
        crate::core::resource_survey::read_cost_gauge(
            &payload.cost_rgb_data,
            payload.cost_roi_width,
            payload.cost_roi_height,
            10,
        )
    } else {
        None
    };
    (timer, cost)
}

/// The worker (consumer): takes each latest frame, saves the debug images asked for, follows
/// the battle state, reads the timer and the cost during a battle, and sends it all to the
/// frontend. The loop never ends; a graceful shutdown would need a shared stop flag or
/// another Condvar signal.
fn run_worker(
    app_handle: &AppHandle,
    debug_state: &DebugState,
    latest_frame: &(Mutex<Option<CapturePayload>>, Condvar),
    stats: &Mutex<Option<crate::core::shittim_link::CaptureStats>>,
) {
    println!("Worker Thread (Consumer) started");
    let mut logic_state = LogicState {
        last_active_time: Instant::now(),
        current_battle_state: BattleState::Inactive,
    };

    loop {
        let payload = next_payload(latest_frame);

        if debug_state
            .should_save_binary
            .swap(false, Ordering::Relaxed)
        {
            save_binary_timer(&payload);
        }
        if debug_state
            .should_save_cost_roi
            .swap(false, Ordering::Relaxed)
        {
            save_cost_roi(&payload);
        }

        logic_state.update(&payload);

        let (current_timer, current_cost) =
            if logic_state.current_battle_state == BattleState::Inactive {
                (None, None)
            } else {
                read_timer_and_cost(&payload)
            };

        // Read latest stats from producer
        let current_stats = stats.lock().unwrap_or_else(PoisonError::into_inner).clone();

        // Follow the loaded timeline with what this frame shows.
        let frame = Observation {
            time_ms: current_timer
                .as_ref()
                .map(crate::core::countdown_monitor::TimerValue::to_millis),
            cost: current_cost.as_ref().map(|cost| cost.current),
            state: logic_state.current_battle_state,
        };
        let guidance = app_handle
            .state::<GuideState>()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_mut()
            .map(|guide| {
                guide.observe(&frame);
                guide.guidance(&frame)
            });

        sync_to_chest(
            app_handle,
            &ShittimPayload {
                battle_state: format!("{:?}", logic_state.current_battle_state),
                fps: 60.0, // Game window capture rate (WGC 60 FPS)
                stats: current_stats,
                timer: current_timer,
                cost: current_cost,
                window: WindowGeometry {
                    x: payload.client_x,
                    y: payload.client_y,
                    width: payload.client_width,
                    height: payload.client_height,
                },
                guidance,
            },
        );
    }
}

/// Where the game window's client area is, as Windows reports it now.
struct ClientArea {
    /// The visible window frame (DWM extended frame bounds), when DWM can tell. It matches
    /// the Graphics Capture API frame, unlike `GetWindowRect`, which includes the invisible
    /// DWM shadow borders.
    frame: Option<RECT>,
    /// The client area's size (`right` x `bottom`).
    size: RECT,
    /// The client area's top-left corner on the screen.
    origin: POINT,
}

impl ClientArea {
    fn of(hwnd: usize) -> Self {
        let mut frame = RECT::default();
        let mut size = RECT::default();
        let mut origin = POINT { x: 0, y: 0 };
        // SAFETY: the three locals outlive the calls that write into them, and the size
        // passed with `frame` is that of a RECT; a window handle that is no longer valid only
        // makes the calls fail.
        let frame_ok = unsafe {
            let hwnd = HWND(hwnd as *mut std::ffi::c_void);
            let ok = DwmGetWindowAttribute(
                hwnd,
                DWMWA_EXTENDED_FRAME_BOUNDS,
                (&raw mut frame).cast::<std::ffi::c_void>(),
                std::mem::size_of::<RECT>() as u32,
            )
            .is_ok();
            let _ = GetClientRect(hwnd, &raw mut size);
            let _ = ClientToScreen(hwnd, &raw mut origin);
            ok
        };
        Self {
            frame: frame_ok.then_some(frame),
            size,
            origin,
        }
    }

    /// Where the client area starts within the captured frame. If DWM is unavailable
    /// (remote desktop, DWM disabled), there is no crop: the full frame is processed as-is.
    fn offset(&self) -> (u32, u32) {
        self.frame.map_or((0, 0), |frame| {
            (
                (self.origin.x - frame.left).max(0) as u32,
                (self.origin.y - frame.top).max(0) as u32,
            )
        })
    }
}

/// The client area of a packed RGBA frame: the frame itself when the client area starts at
/// its corner, or else the client area copied into `crop_buffer`. This removes the title bar
/// and the window borders in CPU, avoiding D3D11 `buffer_crop`, which can cause allocation
/// issues; `crop_buffer` keeps its allocation across frames.
fn crop_client<'a>(
    crop_buffer: &'a mut Vec<u8>,
    packed: &'a [u8],
    (frame_width, frame_height): (u32, u32),
    (offset_x, offset_y): (u32, u32),
    (client_width, client_height): (u32, u32),
) -> (&'a [u8], u32, u32) {
    let crop_w = client_width.min(frame_width.saturating_sub(offset_x));
    let crop_h = client_height.min(frame_height.saturating_sub(offset_y));
    if (offset_x == 0 && offset_y == 0) || crop_w == 0 || crop_h == 0 {
        return (packed, frame_width, frame_height);
    }
    crop_buffer.clear();
    let row_bytes = crop_w as usize * 4;
    let mut actual_h = 0u32;
    for y in 0..crop_h {
        let src = (offset_y + y) as usize * frame_width as usize * 4 + offset_x as usize * 4;
        let end = src + row_bytes;
        if end > packed.len() {
            break;
        }
        crop_buffer.extend_from_slice(&packed[src..end]);
        actual_h += 1;
    }
    (crop_buffer, crop_w, actual_h)
}

/// Saves the game content area (after title bar + black bar removal) as
/// `full_screenshot_<time>.png`.
fn save_full_screenshot(work_data: &[u8], work_width: u32, content: Region) {
    let debug_output_dir = debug_output_dir();
    let _ = std::fs::create_dir_all(&debug_output_dir);
    let path = debug_output_dir.join(format!("full_screenshot_{}.png", clock_stamp()));
    // Extract content area (RGBA → RGB) using work_data with stride = work_width
    let rgb = crop_rgb(work_data, work_width, content);
    if let Some(img) = image::RgbImage::from_raw(content.width(), content.height(), rgb) {
        let _ = img.save(&path);
        println!("Full screenshot saved to {}", path.display());
    }
}

/// Saves the timer region of the current frame, normalized and skew-corrected, as
/// `timer_roi_<time>.png`.
fn save_timer_roi(work_data: &[u8], work_width: u32, (content_width, content_height): (u32, u32)) {
    use crate::core::countdown_monitor::{apply_skew_correction, normalize_roi, SKEW_FACTOR};
    let debug_output_dir = debug_output_dir();
    let _ = std::fs::create_dir_all(&debug_output_dir);
    let path = debug_output_dir.join(format!("timer_roi_{}.png", clock_stamp()));
    // Extract timer ROI (85.49%-93.52%, 3.60%-6.51%) from work_data
    let timer = Region::of(
        content_width,
        content_height,
        (0.8549, 0.9352),
        (0.0360, 0.0651),
    );
    if timer.width() == 0 || timer.height() == 0 {
        return;
    }
    let timer_rgb = crop_rgb(work_data, work_width, timer);
    // Apply normalize + skew correction
    let (norm, norm_w, norm_h) = normalize_roi(&timer_rgb, timer.width(), timer.height());
    let (skewed, skew_w, skew_h) = apply_skew_correction(&norm, norm_w, norm_h, SKEW_FACTOR);
    if let Some(img) = image::RgbImage::from_raw(skew_w, skew_h, skewed) {
        let _ = img.save(&path);
        println!("Timer ROI saved to {}", path.display());
    }
}

/// What the worker reads from a frame: the header strip (timer + pause button), the cost
/// gauge, the center brightness and the PAUSE dialog area. `None` when the header strip is
/// empty.
fn build_payload(
    work_data: &[u8],
    work_width: u32,
    (content_width, content_height): (u32, u32),
    origin: POINT,
) -> Option<CapturePayload> {
    use crate::core::combat_intel::{
        PAUSE_ROI_X_END, PAUSE_ROI_X_START, PAUSE_ROI_Y_END, PAUSE_ROI_Y_START,
    };
    // OPTIMIZATION: Copy Header Strip (Timer + Pause Button)
    // Timer ROI: x 85.49%-100%, y 3.60%-6.51%
    // ROI percentages are content-relative (excluding black bars)
    let roi = Region {
        right: content_width, // 100% of content
        ..Region::of(
            content_width,
            content_height,
            (0.8549, 1.0),
            (0.0360, 0.0651),
        )
    };
    if roi.width() == 0 || roi.height() == 0 {
        return None;
    }

    // Cost Gauge ROI (content-relative): x=64.49%..89.61%, y=94.68%..96.97%
    let cost = Region::of(
        content_width,
        content_height,
        (0.6449, 0.8961),
        (0.9468, 0.9697),
    );

    // Extract PAUSE ROI for consumer-side NCC matching (cheap crop only)
    let pause = Region::of(
        content_width,
        content_height,
        (PAUSE_ROI_X_START, PAUSE_ROI_X_END),
        (PAUSE_ROI_Y_START, PAUSE_ROI_Y_END),
    );
    let mut pause_roi_rgba = Vec::with_capacity((pause.width() * pause.height() * 4) as usize);
    for y in pause.top..pause.bottom {
        let src_start = (y * work_width + pause.left) as usize * 4;
        let src_end = (y * work_width + pause.right) as usize * 4;
        if src_end <= work_data.len() {
            pause_roi_rgba.extend_from_slice(&work_data[src_start..src_end]);
        }
    }

    Some(CapturePayload {
        // Extract only ROI pixels (stride = work_width, no padding)
        rgb_data: crop_rgb(work_data, work_width, roi),
        roi_width: roi.width(),
        roi_height: roi.height(),
        cost_rgb_data: crop_rgb(work_data, work_width, cost),
        cost_roi_width: cost.width(),
        cost_roi_height: cost.height(),
        client_x: origin.x,
        client_y: origin.y,
        client_width: content_width,
        client_height: content_height,
        // Sample center brightness for Paused/Slow state detection
        // Uses content dimensions for center point, work_width for stride
        center_brightness: sample_center_brightness(
            work_data,
            work_width,
            content_width,
            content_height,
        ),
        pause_roi_rgba,
        pause_roi_width: pause.width(),
        pause_roi_height: pause.height(),
    })
}

impl CaptureHandler {
    /// Counts a frame Windows delivered, and every 10 seconds hands the frame rates to the
    /// worker.
    fn count_frame(&mut self, now: Instant) {
        self.fps_counter_received += 1;
        if now.duration_since(self.fps_last_log_time) < Duration::from_secs(10) {
            return;
        }
        let elapsed = now.duration_since(self.fps_last_log_time).as_secs_f64();
        *self
            .shared_stats
            .lock()
            .unwrap_or_else(PoisonError::into_inner) =
            Some(crate::core::shittim_link::CaptureStats {
                received: self.fps_counter_received as f64 / elapsed,
                accepted: self.fps_counter_accepted as f64 / elapsed,
                queue_full: self.fps_queue_full,
            });
        self.fps_counter_received = 0;
        self.fps_queue_full = 0;
        self.fps_counter_accepted = 0;
        self.fps_last_log_time = now;
    }
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
            run_worker(&app_handle, &debug_state, &consumer_frame, &consumer_stats);
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
        self.count_frame(now);

        // Process at ~30 FPS (33ms). WGC delivers at 60 FPS (16ms) so fresh
        // frames are always available. Consumer has 33ms to finish before the
        // next ROI arrives, keeping frontend updates low-latency.
        if now.duration_since(self.last_frame_time) < Duration::from_millis(33) {
            return Ok(());
        }
        self.last_frame_time = now;
        self.fps_counter_accepted += 1;

        // Window geometry for crop calculation and overlay positioning.
        let area = ClientArea::of(self.hwnd);
        let (offset_x, offset_y) = area.offset();
        let client_width = area.size.right as u32;
        let client_height = area.size.bottom as u32;

        // Always use frame.buffer() — proven stable. Software crop below
        // extracts client area safely (avoids D3D11 buffer_crop issues).
        let Ok(mut buffer) = frame.buffer() else {
            eprintln!("Failed to capture buffer");
            return Ok(());
        };
        let frame_width = buffer.width();
        let frame_height = buffer.height();

        if !self.first_crop_logged {
            self.first_crop_logged = true;
            let dwm_rect = area.frame.unwrap_or_default();
            println!(
                "[Crop Debug] frame={}x{}, dwm=({},{},{},{}), client_pt=({},{}), client={}x{}, offset=({},{})",
                frame_width, frame_height,
                dwm_rect.left, dwm_rect.top, dwm_rect.right, dwm_rect.bottom,
                area.origin.x, area.origin.y,
                client_width, client_height,
                offset_x, offset_y,
            );
        }

        let Ok(packed_slice) = buffer.as_nopadding_buffer() else {
            eprintln!("Failed to get nopadding buffer");
            return Ok(());
        };
        let (work_data, work_width, work_height) = crop_client(
            &mut self.crop_buffer,
            packed_slice,
            (frame_width, frame_height),
            (offset_x, offset_y),
            (client_width, client_height),
        );

        // Detect content bounds (exclude black bars from game rendering)
        // Cached: recompute every 150 frames (~5s) since black bars are static
        self.content_bounds_counter += 1;
        let (content_width, content_height) = match self.cached_content_bounds {
            Some(bounds) if !self.content_bounds_counter.is_multiple_of(150) => bounds,
            _ => {
                let bounds = detect_content_bounds(work_data, work_width, work_height, 4);
                self.cached_content_bounds = Some(bounds);
                bounds
            }
        };

        if self
            .debug_state
            .should_full_screenshot
            .swap(false, Ordering::Relaxed)
        {
            let content = Region {
                left: 0,
                right: content_width,
                top: 0,
                bottom: content_height,
            };
            save_full_screenshot(work_data, work_width, content);
        }
        if self
            .debug_state
            .should_screenshot
            .swap(false, Ordering::Relaxed)
        {
            save_timer_roi(work_data, work_width, (content_width, content_height));
        }

        let content_size = (content_width, content_height);
        if let Some(payload) = build_payload(work_data, work_width, content_size, area.origin) {
            // Conflating Push: Overwrite any pending frame with the latest one
            let (lock, cvar) = &*self.latest_frame;
            let mut guard = lock.lock().unwrap_or_else(PoisonError::into_inner);
            if guard.is_some() {
                self.fps_queue_full += 1; // Worker was busy, frame overwritten
            }
            *guard = Some(payload);
            cvar.notify_one();
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
                if r > BRIGHTNESS_THRESHOLD || g > BRIGHTNESS_THRESHOLD || b > BRIGHTNESS_THRESHOLD
                {
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
                if r > BRIGHTNESS_THRESHOLD || g > BRIGHTNESS_THRESHOLD || b > BRIGHTNESS_THRESHOLD
                {
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
pub fn sample_center_brightness(
    rgba_data: &[u8],
    stride_width: u32,
    content_width: u32,
    content_height: u32,
) -> f32 {
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
                br_sum += (299 * u64::from(rgba_data[idx])
                    + 587 * u64::from(rgba_data[idx + 1])
                    + 114 * u64::from(rgba_data[idx + 2]))
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
    app_handle: &AppHandle,
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
        assert!(
            (brightness - 255.0).abs() < 1.0,
            "Expected ~255, got {brightness}"
        );
    }

    #[test]
    fn test_sample_center_brightness_black() {
        // Pure black (0,0,0) → brightness = 0
        let data = make_rgba_frame(100, 100, 0, 0, 0);
        let brightness = sample_center_brightness(&data, 100, 100, 100);
        assert!(
            (brightness - 0.0).abs() < 0.01,
            "Expected ~0, got {brightness}"
        );
    }

    #[test]
    fn test_sample_center_brightness_gray() {
        // Mid gray (128,128,128) → brightness ≈ 128
        let data = make_rgba_frame(100, 100, 128, 128, 128);
        let brightness = sample_center_brightness(&data, 100, 100, 100);
        assert!(
            (brightness - 128.0).abs() < 1.0,
            "Expected ~128, got {brightness}"
        );
    }

    #[test]
    fn test_sample_center_brightness_empty() {
        let brightness = sample_center_brightness(&[], 0, 0, 0);
        assert!(
            brightness.abs() < f32::EPSILON,
            "Expected 0, got {brightness}"
        );
    }

    #[test]
    fn test_sample_center_brightness_small_frame() {
        // 5x5 frame: center is at (2,2), sampling 10x10 but only some pixels exist
        let data = make_rgba_frame(5, 5, 200, 200, 200);
        let brightness = sample_center_brightness(&data, 5, 5, 5);
        // Should still return a value (partial sampling)
        assert!(brightness > 100.0, "Expected >100, got {brightness}");
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
            let path = format!("{manifest_dir}/tests/fixtures/screenshots/{filename}");
            let Ok(img) = image::open(&path) else {
                continue;
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
                    if idx + 2 < rgb_data.len()
                        && (rgb_data[idx] > 10 || rgb_data[idx + 1] > 10 || rgb_data[idx + 2] > 10)
                    {
                        all_black = false;
                        break;
                    }
                }
                if all_black {
                    left_bar = x + 1;
                } else {
                    break;
                }
            }

            // Check top edge: scan first 30 rows, ALL columns
            let mut top_bar = 0u32;
            for y in 0..height.min(30) {
                let mut all_black = true;
                for x in 0..width {
                    let idx = ((y * width + x) * 3) as usize;
                    if idx + 2 < rgb_data.len()
                        && (rgb_data[idx] > 10 || rgb_data[idx + 1] > 10 || rgb_data[idx + 2] > 10)
                    {
                        all_black = false;
                        break;
                    }
                }
                if all_black {
                    top_bar = y + 1;
                } else {
                    break;
                }
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
