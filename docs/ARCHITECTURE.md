# Architecture

## Directory Structure

We use a **Feature-Sliced** inspired approach for Frontend and a **Modular** approach for Backend.

### Frontend (`src/`)

- **`features/`**: (Planned) Code organized by business domain.
- **`components/`**:
  - **`layout/`**: Structural components (e.g., `ConnectionPanel`, `SettingsPanel`).
  - **`ui/`**: (Planned) Reusable UI primitives.
- **`stores/`**: Global state management.
- **`lib/`**: Generic utilities not tied to React.

### Backend (`src-tauri/src/`)

- **`commands/`**: The interface layer. Functions exposed to the Frontend via Tauri's invoke system. **No heavy logic here.**
- **`core/`**: Pure Rust business logic. Independent of Tauri where possible.
  - `visual_intercept.rs`: Screen capture logic using Windows Graphics Capture API.
- **`state/`**: Application state definitions (Mutex, Arc, etc.).

## Design Principles

1. **Separation of Concerns**: Frontend handles UI/State, Backend handles System/Computation.
2. **Type Safety**: Aggressively use TypeScript and Rust types to ensure correctness.
3. **Aesthetics**: Code should be as beautiful as the UI.
4. **Theme Support**: First-class Dark Mode support (via Tailwind v4 `selector` strategy).
5. **Code Identity**: This application is Plana herself.
   - **Iron Rule**: New modules in `src-tauri/src/core/` MUST be named as actions performed by Plana to assist the Sensei (Player).
   - Examples:
     - `visual_intercept.rs` (Visual Information Interception)
     - `combat_intel.rs` (Combat Intelligence Analysis)
     - `countdown_monitor.rs` (Countdown Monitoring)
     - `shittim_link.rs` (Shittim Chest Connectivity)
     - `target_acquisition.rs` (Target Acquisition)
   - Avoid generic names like `utils.rs` or `manager.rs`. Plana doesn't "manage"; she *analyzes*, *intercepts*, *calculates*, and *assists*.

6. **Concept & Setting**:
   - **Backend (Tauri)**: **Plana** (AI Operator). She runs the core logic, intercepts the system, and calculating combat data.
   - **Frontend (React)**: **The Shittim Chest** (Performance Tablet). The interface presented to the Sensei (Player).
   - **Interaction**: Plana intercepts the "Shittim Chest" OS to overlay tactical advice and real-time battle data via the **Shittim Link** layer.

## Core Systems

### Screen Capture & Streaming

We employ a high-performance, low-latency capture pipeline optimized for Blue Archive (60FPS game, 30FPS combat logic).

- **Capture Engine**: Microsoft Windows Graphics Capture API (via `windows-capture` crate).
- **Cropping**: Manual calculation of Client Area using Win32 API (`GetClientRect`, `ClientToScreen`) to exclude title bars and borders reliably across window modes.
- **Single-Stream Architecture**:
  - **Internal Processing**: Runs at **30 FPS**. This stream allows the backend to perform precise OCR analysis (for time/HP).
  - **Streaming**: Disabled. We removed the MJPEG stream to minimize CPU/Thread contention and ensure OCR runs at peak performance.
