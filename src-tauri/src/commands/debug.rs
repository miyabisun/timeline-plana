use crate::state::DebugState;
use std::sync::Arc;
use tauri::{command, State};

#[command]
pub fn trigger_screenshot(state: State<Arc<DebugState>>) -> Result<String, String> {
    state.request_screenshot();
    Ok("Screenshot requested".to_string())
}

#[command]
pub fn save_binary_timer_image(state: State<Arc<DebugState>>) -> Result<String, String> {
    state.request_binary_image();
    Ok("Binary timer image requested".to_string())
}
