// App state definitions
use crate::core::tactical_guidance::Guide;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// The timeline battles are guided through, once one is loaded.
pub type GuideState = Mutex<Option<Guide>>;

/// Debug image requests from the frontend; the capture side takes each one once.
#[derive(Default)]
pub struct DebugState {
    pub should_screenshot: AtomicBool,
    pub should_save_binary: AtomicBool,
    pub should_save_cost_roi: AtomicBool,
    pub should_full_screenshot: AtomicBool,
}

impl DebugState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn request_screenshot(&self) {
        self.should_screenshot.store(true, Ordering::Relaxed);
    }

    pub fn request_binary_image(&self) {
        self.should_save_binary.store(true, Ordering::Relaxed);
    }

    pub fn request_cost_roi_image(&self) {
        self.should_save_cost_roi.store(true, Ordering::Relaxed);
    }

    pub fn request_full_screenshot(&self) {
        self.should_full_screenshot.store(true, Ordering::Relaxed);
    }
}
