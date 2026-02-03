# Screen Capture & Streaming Specification

## Overview
The application must capture the game window of "Blue Archive" (BlueArchive.exe) and display it to the user with minimal latency, while simultaneously analyzing the video feed for game state (OCR).

## Implementation Details

### 1. Target Acquisition
- **Method**: Scanning active processes for `BlueArchive.exe`.
- **Validation**: Verifies the process has a visible window handle (HWND).
- **Graceful Handling**: Supports automatic reconnection if the game restarts.

### 2. Capture Pipeline
- **API**: Windows Graphics Capture API (WGC).
- **Library**: `windows-capture` (Rust).
- **Cropping**:
  - The WGC API captures the full Window Rect (including shadows and title bars).
  - We use Win32 `GetClientRect` and `ClientToScreen` to calculate the exact pixel offsets of the content area.
  - The frame is cropped *before* any analysis or encoding.
  
### 3. Dual-Rate Architecture
To balance performance and precision, the pipeline operates at two different frame rates:

| Stream | Rate | Format | Purpose |
|O---|---|---|---|
| **Internal** | **30 FPS** | Raw RGB8 | OCR, Time detection, HP tracking. Matches game logic tick rate. |
| **Output** | **15 FPS** | MJPEG (Q95) | User feedback (UI Preview). Reduced rate saves CPU/Bandwidth. |

### 4. MJPEG Server
- **Server**: Embedded `tiny_http` server.
- **Port**: 12345 (Fixed).
- **Concurrency**: Supports multiple frontend clients via `Condvar` broadcasting.
- **Endpoint**: `http://localhost:12345` (Stream root).
- **Format**: `multipart/x-mixed-replace; boundary=boundary`.

## Frontend Integration
The Frontend (React) does not handle binary data via Tauri Events. Instead, it uses a standard `<img>` tag:

```tsx
<img src="http://localhost:12345" alt="Game Stream" />
```

This offloads decoding and rendering to the browser's native engine, ensuring smooth playback without React re-renders.
