# Target Acquisition (標的捕捉)

## Overview
Reliably identifies the Blue Archive game process on Window, distinguishing it from browser windows or videos.

## Logic
1. **EnumWindows**: Iterate through all top-level windows via Win32 API.
2. **Filter**:
   - Must be visible (`IsWindowVisible`).
   - Must have a non-empty title.
3. **Process Info**:
   - Extract PID.
   - Extract Executable Name (e.g., `BlueArchive.exe`).
4. **Candidate List**:
   - Return a list of all plausible candidates to the Frontend.
   - Frontend presents selection (or auto-selects if obvious).

## Rust Modules
- **`core::target_acquisition`**: Contains the Win32 API calls and filtering logic.
- **`commands::target`**: Tauri command wrapper.

## Dependencies
- `windows` crate (Win32 APIs)
