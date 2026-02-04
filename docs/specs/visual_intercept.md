# Visual Intercept (視覚情報の傍受)

## Overview
The application captures the "Blue Archive" game window and processes it for high-speed OCR (Timer) with minimal latency.  
The architecture has been highly optimized to support real-time combat analysis.

## Core Architecture: Zero Latency Pipeline

### 1. High-Speed Capture (Producer Thread)
- **API**: Windows Graphics Capture API (WGC).
- **Input Rate**: Forced **60 FPS** (16ms interval) using `MinimumUpdateIntervalSettings::Custom`.
- **Optimization (ROI Copy)**: 
  - Instead of copying the full 4K frame (~8MB), the capture thread **only extracts the Timer ROI** (~20KB).
  - This >99% reduction in bandwidth prevents the WGC callback from blocking, ensuring Windows delivers frames at stable 60 FPS.
- **Frame Gating**: 
  - Enforces a strict **30 FPS** rhythm for processing by dropping excess frames (`delta < 32ms`) before they reach the worker.

### 2. Zero Latency Queue
- **Mechanism**: Conflating Queue using `Arc<(Mutex<Option<CapturePayload>>, Condvar)>`.
- **Policy**: **Drop-Oldest**.
  - The worker thread always wakes up to the *absolute latest* frame.
  - If the worker is busy, intermediate frames are overwritten instantly.
  - **Latency**: Guaranteed to be at most 1 frame (approx 16-33ms).

### 3. Dedicated Worker (Consumer Thread)
- **Input**: Pre-cropped Timer ROI (RGB8).
- **Processing**:
  - **Direct OCR**: Passing ROI data directly to `recognize_timer_from_roi`, skipping redundant cropping/allocations.
  - **Battle State**: Assumed `Active` to maximize OCR throughput (full-screen analysis is skipped in this optimized mode).
- **Output**: Emits `timer-update` events to Frontend.

## Performance Metrics
- **Capture FPS**: ~60.0 (Windows Source)
- **Worker FPS**: ~30.0 (Gated stable rate)
- **Visibility**: Internal processing FPS is displayed in the **Debug Panel** to clear user confusion (hidden from main Connection view).
- **Copy Overhead**: Negligible (< 0.1ms per frame due to ROI-only copy)

