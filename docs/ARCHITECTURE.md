# Architecture

## Directory Structure

We use a **Feature-Sliced** inspired approach for Frontend and a **Modular** approach for Backend.

### Frontend (`src/`)

- **`features/`**: Code organized by business domain (e.g., `timeline`, `settings`). Each feature is self-contained.
- **`components/`**: Shared UI primitives (`ui`) and layout components (`layout`).
- **`stores/`**: Global state management (Zustand/Jotai).
- **`lib/`**: Generic utilities not tied to React.

### Backend (`src-tauri/src/`)

- **`commands/`**: The interface layer. Functions exposed to the Frontend via Tauri's invoke system. **No heavy logic here.**
- **`core/`**: Pure Rust business logic. Independent of Tauri where possible.
  - `visual_intercept.rs`: Screen capture logic using Windows Graphics Capture API.
  - `mjpeg.rs`: Lightweight HTTP server for streaming captured frames.
- **`state/`**: Application state definitions (Mutex, Arc, etc.).

## Design Principles

1. **Separation of Concerns**: Frontend handles UI/State, Backend handles System/Computation.
2. **Type Safety**: Aggressively use TypeScript and Rust types to ensure correctness.
3. **Aesthetics**: Code should be as beautiful as the UI.

## Core Systems

### Screen Capture & Streaming

We employ a high-performance, low-latency capture pipeline optimized for Blue Archive (60FPS game, 30FPS combat logic).

- **Capture Engine**: Microsoft Windows Graphics Capture API (via `windows-capture` crate).
- **Cropping**: Manual calculation of Client Area using Win32 API (`GetClientRect`, `ClientToScreen`) to exclude title bars and borders reliably across window modes.
- **Dual-Rate Architecture**:
  - **Internal Processing**: Runs at **30 FPS**. This stream allows the backend to perform precise OCR analysis (for time/HP).
  - **Video Streaming**: Runs at **15 FPS** (Quality 95 MJPEG). This stream is served via HTTP to the frontend for visual confirmation. This separation minimizes CPU/Network load while maintaining data precision.
- **Streaming Protocol**: standard MJPEG over HTTP (multipart/x-mixed-replace). Served by `tiny_http` on port `12345` (default).
