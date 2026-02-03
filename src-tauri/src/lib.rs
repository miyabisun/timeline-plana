// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
pub mod commands;
pub mod core;
pub mod state;

use crate::core::mjpeg::{start_server, MjpegState};
use std::sync::Arc;
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let mjpeg_state = Arc::new(MjpegState::new());
            // Start the MJPEG server on port 12345
            start_server(mjpeg_state.clone(), 12345);
            // Manage the state so other threads (capture) can access it
            app.manage(mjpeg_state);

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::timeline::greet,
            commands::target::list_potential_targets,
            commands::intercept::start_intercept_demo
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
