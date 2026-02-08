// App state definitions
use std::sync::Mutex;

pub struct DebugState {
    pub should_screenshot: Mutex<bool>,
    pub should_save_binary: Mutex<bool>,
    pub should_save_cost_roi: Mutex<bool>,
}

impl DebugState {
    pub fn new() -> Self {
        Self {
            should_screenshot: Mutex::new(false),
            should_save_binary: Mutex::new(false),
            should_save_cost_roi: Mutex::new(false),
        }
    }

    pub fn request_screenshot(&self) {
        let mut lock = self.should_screenshot.lock().unwrap();
        *lock = true;
    }

    pub fn request_binary_image(&self) {
        let mut lock = self.should_save_binary.lock().unwrap();
        *lock = true;
    }

    pub fn request_cost_roi_image(&self) {
        let mut lock = self.should_save_cost_roi.lock().unwrap();
        *lock = true;
    }
}
