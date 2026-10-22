use crate::core::visual_intercept::start_capture;
use std::thread;

#[tauri::command]
pub fn start_intercept_demo(app: tauri::AppHandle, hwnd: usize) -> Result<String, String> {
    // Spawn capture in a separate thread to avoid blocking the main Tauri event loop
    thread::spawn(move || match start_capture(&app, hwnd) {
        Ok(()) => println!("Capture finished normally"),
        Err(e) => eprintln!("Capture error: {e:?}"),
    });

    Ok("Intercept sequence initiated.".to_string())
}
