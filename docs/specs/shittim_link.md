# Shittim Link Specification

## Concept
**Shittim Link** is the dedicated communication channel between **Plana (Backend/Tauri)** and the **Shittim Chest (Frontend/React)**.

Because Plana operates in the background (Rust) intercepting system signals, she needs a reliable, high-frequency way to project this information onto the Sensei's tablet interface. This module encapsulates that bridge.

## Protocol

### Direction
**Unidirectional**: Plana -> Shittim Chest.
(Sensei's commands are sent via standard IPC, but tactical data flows exclusively via this Link).

### Transport
- **Mechanism**: Tauri Event System (`app_handle.emit`)
- **Event Name**: `link-sync`
- **Frequency**: **30 Hz** (Synced with the Logic Frame Rate)

## Payload Structure

The data is encoded as a JSON object with the following schema:

```json
{
  "battle_state": "Active" | "Inactive" | "Paused" | "Slow",
  "fps": number,
  "stats": {
    "received": number,
    "accepted": number,
    "queue_full": number
  },
  "timer": {
    "minutes": number,
    "seconds": number,
    "milliseconds": number
  },
  "cost": {
    "current": number,
    "max_cost": number,
    "confidence": number
  },
  "window": {
    "x": number,
    "y": number,
    "width": number,
    "height": number
  }
}
```

### Field Details

| Field | Type | Description |
|-------|------|-------------|
| `battle_state` | string | Current state: `"Active"`, `"Inactive"`, `"Paused"`, or `"Slow"` |
| `fps` | number | Estimated Link Speed |
| `stats` | object? | System diagnostics (capture FPS, processing FPS, dropped frames) |
| `timer` | object? | Remaining battle time. Present during Active/Paused/Slow states |
| `cost` | object? | EX skill cost gauge. Present during Active/Paused/Slow states |
| `window` | object | Target window client area geometry (screen coordinates) |

## Update Strategy

- The Link does not wait for requests. It pushes the latest state every logic frame (approx. 33ms).
- This ensures the UI is always "eventually consistent" with Plana's internal state without complex polling logic in the Frontend.
- **"System Linked"** status in the UI confirms the heartbeat of this signal.

## Rust Module

```rust
pub fn sync_to_chest(
    app_handle: &AppHandle,
    battle_state: &str,
    timer: Option<TimerValue>,
    cost: Option<CostValue>,
    fps: f64,
    stats: Option<CaptureStats>,
    window: WindowGeometry,
)
```
