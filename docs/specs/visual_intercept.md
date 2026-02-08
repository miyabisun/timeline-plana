# Visual Intercept (視覚情報の傍受)

## Overview
The application captures the "Blue Archive" game window and processes it for real-time combat analysis: timer OCR, cost gauge reading, and battle state detection.

## Core Architecture: Producer-Consumer Pipeline

### 1. High-Speed Capture (Producer Thread)
- **API**: Windows Graphics Capture API (WGC).
- **Input Rate**: Forced **60 FPS** (16ms interval) using `MinimumUpdateIntervalSettings::Custom`.
- **ROI Extraction**: Instead of copying the full 4K frame (~8MB), the producer extracts only:
  - **Header Strip**: 85-100% X, 3.5-6.3% Y (~20KB) — Timer + Pause button
  - **Cost Gauge**: 64-89% X, 91.0-93.2% Y (~20KB) — EX skill cost boxes
  - **Center Brightness**: 10x10 pixel sample from frame center (~0.3KB) — Paused/Slow detection
- **Frame Gating**: Enforces **30 FPS** rhythm by dropping excess frames (`delta < 32ms`).

### 2. Zero Latency Queue
- **Mechanism**: Conflating Queue using `Arc<(Mutex<Option<CapturePayload>>, Condvar)>`.
- **Policy**: **Drop-Oldest**.
  - The worker thread always wakes up to the *absolute latest* frame.
  - If the worker is busy, intermediate frames are overwritten instantly.
  - **Latency**: Guaranteed to be at most 1 frame (approx 16-33ms).

### 3. Dedicated Worker (Consumer Thread)

#### Battle State Detection
Uses header strip ROI + center brightness to classify state:

| State | Condition |
|-------|-----------|
| **Active** | Pause button white pixels > 5% |
| **Paused** | center_brightness > 150 (white PAUSE menu) |
| **Slow** | center_brightness < 100 (dark skill confirmation overlay) |
| **Inactive** | No state detected for > 2 seconds |

State transitions use hysteresis (2s timeout before switching to Inactive).

#### Processing (for all non-Inactive states)
- **Timer OCR**: Sub-crops timer region from header strip, passes to `countdown_monitor::recognize_timer_from_roi`.
- **Cost Gauge Reading**: Passes cost ROI to `resource_survey::read_cost_gauge` with `max_cost=10`.
- **Output**: Emits combined data via `shittim_link::sync_to_chest` to Frontend.

## CapturePayload Structure

```rust
struct CapturePayload {
    rgb_data: Vec<u8>,        // Header strip ROI (RGB8)
    roi_width: u32,
    roi_height: u32,
    cost_rgb_data: Vec<u8>,   // Cost gauge ROI (RGB8)
    cost_roi_width: u32,
    cost_roi_height: u32,
    client_x: i32,            // Window position
    client_y: i32,
    client_width: u32,
    client_height: u32,
    center_brightness: f32,   // Center frame brightness (ITU-R BT.601)
}
```

## Performance Metrics
- **Capture FPS**: ~60.0 (Windows Source)
- **Worker FPS**: ~30.0 (Gated stable rate)
- **Copy Overhead**: Negligible (< 0.1ms per frame due to ROI-only copy)
