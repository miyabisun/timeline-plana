use crate::core::countdown_monitor::TimerValue;
use crate::core::resource_survey::CostValue;
use crate::core::tactical_guidance::Guidance;
use tauri::{AppHandle, Emitter};

/// Capture performance statistics.
#[derive(Clone, serde::Serialize)]
pub struct CaptureStats {
    pub received: f64,
    pub accepted: f64,
    pub queue_full: u64,
}

/// Target window client area geometry (screen coordinates).
#[derive(Clone, serde::Serialize)]
pub struct WindowGeometry {
    /// Client area top-left X in screen pixels
    pub x: i32,
    /// Client area top-left Y in screen pixels
    pub y: i32,
    /// Client area width
    pub width: u32,
    /// Client area height
    pub height: u32,
}

/// Combined data packet sent to the Shittim Chest (Frontend) every frame.
#[derive(Clone, serde::Serialize)]
pub struct ShittimPayload {
    pub battle_state: String,
    pub fps: f64,
    pub stats: Option<CaptureStats>,
    pub timer: Option<TimerValue>,
    pub cost: Option<CostValue>,
    pub window: WindowGeometry,
    /// What to use next, once a timeline is loaded.
    pub guidance: Option<Guidance>,
}

/// Transmits tactical data to the Shittim Chest.
pub fn sync_to_chest(app_handle: &AppHandle, payload: &ShittimPayload) {
    let _ = app_handle.emit("link-sync", payload);
}
