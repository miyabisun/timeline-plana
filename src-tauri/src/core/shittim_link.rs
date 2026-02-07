use crate::core::countdown_monitor::TimerValue;
use tauri::{AppHandle, Emitter};

/// Shittim Link Payload
/// Combined data packet sent to the Shittim Chest (Frontend) every frame.
#[derive(Clone, serde::Serialize)]
pub struct CaptureStats {
    pub received: f64,
    pub accepted: f64,
    pub queue_full: u64,
}

/// Shittim Link Payload
/// Combined data packet sent to the Shittim Chest (Frontend) every frame.
#[derive(Clone, serde::Serialize)]
struct ShittimPayload {
    /// Current Battle State ("Active", "Inactive", "Paused")
    battle_state: String,

    /// Estimated FPS of the connection
    fps: f64,

    /// Detailed Capture Stats
    stats: Option<CaptureStats>,

    /// Timer Data (Optional, only if visible)
    timer: Option<TimerValue>,
}

/// Transmits tactical data to the Shittim Chest.
///
/// Use this to sync the backend state with the frontend UI.
/// This should be called primarily from the `visual_intercept` loop.
pub fn sync_to_chest(
    app_handle: &AppHandle,
    battle_state: &str,
    timer: Option<TimerValue>,
    fps: f64,
    stats: Option<CaptureStats>,
) {
    let payload = ShittimPayload {
        battle_state: battle_state.to_string(),
        fps,
        stats,
        timer,
    };

    let _ = app_handle.emit("link-sync", &payload);
}
