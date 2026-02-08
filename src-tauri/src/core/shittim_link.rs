use crate::core::countdown_monitor::TimerValue;
use crate::core::resource_survey::CostValue;
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
struct ShittimPayload {
    battle_state: String,
    fps: f64,
    stats: Option<CaptureStats>,
    timer: Option<TimerValue>,
    cost: Option<CostValue>,
    window: WindowGeometry,
}

/// Transmits tactical data to the Shittim Chest.
pub fn sync_to_chest(
    app_handle: &AppHandle,
    battle_state: &str,
    timer: Option<TimerValue>,
    cost: Option<CostValue>,
    fps: f64,
    stats: Option<CaptureStats>,
    window: WindowGeometry,
) {
    let payload = ShittimPayload {
        battle_state: battle_state.to_string(),
        fps,
        stats,
        timer,
        cost,
        window,
    };

    let _ = app_handle.emit("link-sync", &payload);
}
