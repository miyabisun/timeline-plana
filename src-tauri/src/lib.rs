// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
pub mod commands;
pub mod core;
pub mod state;

use crate::state::DebugState;
use std::sync::Arc;
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let debug_state = Arc::new(DebugState::new());
            // Manage the state so other threads (capture) can access it
            app.manage(debug_state);

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::timeline::greet,
            commands::target::list_potential_targets,
            commands::intercept::start_intercept_demo,
            commands::intercept::start_intercept_demo,
            commands::debug::trigger_screenshot,
            commands::debug::save_binary_timer_image
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
