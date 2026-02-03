use crate::core::target_acquisition::{scan_processes, ProcessCandidate};

#[tauri::command]
pub fn list_potential_targets() -> Vec<ProcessCandidate> {
    scan_processes()
}
