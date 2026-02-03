use serde::Serialize;
use windows::Win32::Foundation::{BOOL, HWND, LPARAM};
use windows::Win32::System::Threading::{
    OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_FORMAT, PROCESS_QUERY_LIMITED_INFORMATION,
};
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GetWindowTextLengthW, GetWindowTextW, GetWindowThreadProcessId, IsWindowVisible,
};

#[derive(Debug, Serialize, Clone)]
pub struct ProcessCandidate {
    pub pid: u32,
    pub name: String,
    pub window_title: String,
    pub hwnd: usize, // Needed for windows-capture
}

pub fn scan_processes() -> Vec<ProcessCandidate> {
    let mut candidates = Vec::new();
    let candidates_ptr = &mut candidates as *mut Vec<ProcessCandidate>;

    unsafe {
        let _ = EnumWindows(Some(enum_windows_proc), LPARAM(candidates_ptr as isize));
    }

    candidates
}

unsafe extern "system" fn enum_windows_proc(hwnd: HWND, lparam: LPARAM) -> BOOL {
    if !IsWindowVisible(hwnd).as_bool() {
        return BOOL(1);
    }

    let length = GetWindowTextLengthW(hwnd);
    if length == 0 {
        return BOOL(1);
    }

    let mut buffer = vec![0u16; (length + 1) as usize];
    GetWindowTextW(hwnd, &mut buffer);
    let window_title = String::from_utf16_lossy(&buffer[..length as usize]);

    let mut pid = 0;
    GetWindowThreadProcessId(hwnd, Some(&mut pid));

    // Open process to get name
    // Use PROCESS_QUERY_LIMITED_INFORMATION for better compatibility (e.g. anti-cheat, exalted privs)
    let process_handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid);

    let mut name = String::from("Unknown");

    if let Ok(handle) = process_handle {
        let mut module_name = [0u16; 1024];
        let mut size = module_name.len() as u32;
        // QueryFullProcessImageNameW works with PROCESS_QUERY_LIMITED_INFORMATION
        unsafe {
            if QueryFullProcessImageNameW(
                handle,
                PROCESS_NAME_FORMAT(0),
                windows::core::PWSTR::from_raw(module_name.as_mut_ptr()),
                &mut size,
            )
            .is_ok()
            {
                let full_path = String::from_utf16_lossy(&module_name[..size as usize]);
                // Extract just the file name
                if let Some(file_name) = std::path::Path::new(&full_path).file_name() {
                    name = file_name.to_string_lossy().into_owned();
                } else {
                    name = full_path;
                }
            }
        }
        let _ = windows::Win32::Foundation::CloseHandle(handle);
    }

    let candidates = &mut *(lparam.0 as *mut Vec<ProcessCandidate>);
    candidates.push(ProcessCandidate {
        pid,
        name,
        window_title,
        hwnd: hwnd.0 as usize,
    });

    BOOL(1)
}
