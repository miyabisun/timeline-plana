# Combat Intel (戦闘情報)

## Overview
Analyzes screen captures to detect the current battle state. This enables the application to react appropriately to pauses, slow-motion effects, and active combat phases.

## States

| State | Description | Detection Method |
|-------|-------------|------------------|
| **Active** | Normal battle in progress | Pause button white pixels > 5% in header strip ROI |
| **Paused** | PAUSE menu is open | Center brightness > 150 (white menu overlay) |
| **Slow** | Skill confirmation / slow-motion | Center brightness < 100 (dark overlay) |
| **Inactive** | No battle detected | No state detected for > 2 seconds |

## Detection Logic

### 1. Pause Button Visibility Check
- ROI: Right side of header strip (maps to ~92-100% X, 0.5-5% Y of full frame)
- Scans for white pixels (R,G,B all > 200)
- Requires > 5% white pixel ratio to confirm visibility
- **Result**: `true` → Active state

### 2. Center Brightness Check
- Source: 10x10 pixel sample from frame center (computed by Producer)
- ITU-R BT.601 perceived brightness formula
- **> 150**: Paused (white PAUSE menu dominates center)
- **< 100**: Slow (dark skill confirmation overlay)

### 3. Hysteresis
- State transitions to Inactive only after 2 seconds without Active/Paused/Slow detection
- Prevents flickering during brief transitions

## Module: `analyze_battle_state()`

Standalone function in `combat_intel.rs` that takes full-frame RGB data and returns `BattleState`:
- Uses `check_pause_button_debug()` for white_ratio (92-100% X, 0.5-5% Y)
- Uses `calculate_center_brightness()` for center 25-75% region brightness
- Same thresholds as the visual_intercept consumer

Note: `visual_intercept.rs` uses a lightweight variant of this logic (header strip + center_brightness sample) to avoid full-frame processing.

## Functions

| Function | Description |
|----------|-------------|
| `analyze_battle_state()` | フルフレームRGBから`BattleState`を判定（テスト・診断用） |
| `check_pause_presence_in_wide_roi()` | ヘッダーストリップROI（85-100% X）からポーズボタン可視性を判定。`visual_intercept.rs`のConsumerから呼ばれる主要関数 |
| `check_pause_button_visible()` | フルフレームからポーズボタン可視性を判定 |
| `check_pause_button_debug()` | ROI内の青/白ピクセル比率を返す（診断用） |
| `calculate_center_brightness()` | フレーム中央50%のITU-R BT.601輝度を算出 |

## Rust Modules
- **`core::combat_intel`**: State detection logic
- **Dependencies**: None (pure pixel analysis)
