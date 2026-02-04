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
  "battle_state": "Active" | "Inactive" | "Paused",
  "fps": number, // Estimated Link Speed
  "stats": {     // System Diagnostics
    "received": number, // Raw Capture FPS from Windows
    "accepted": number, // Processed FPS (Logic Rate)
    "queue_full": number // Dropped Frames Count
  },
  "timer": {     // Optional, only present during Active combat
    "minutes": number,
    "seconds": number,
    "milliseconds": number
  }
}
```

## Update Strategy

- The Link does not wait for requests. It pushes the latest state every logic frame (approx. 33ms).
- This ensures the UI is always "eventually consistent" with Plana's internal state without complex polling logic in the Frontend.
- **"System Linked"** status in the UI confirms the heartbeat of this signal.
