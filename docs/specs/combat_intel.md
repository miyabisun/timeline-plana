# Combat Intel (戦闘情報)

## Overview
Analyzes screen captures to detect the current battle state. This enables the application to react appropriately to pauses, slow-motion effects, and active combat phases.

## States

| State | Description | Detection Method |
|-------|-------------|------------------|
| **Active** | Normal battle in progress | Pause button white pixels > 5% AND blue pixels < 10% in header strip ROI |
| **Paused** | PAUSE menu is open | NCC template match score > 0.6, or center brightness > 150 |
| **Slow** | Skill confirmation / slow-motion | Center brightness < 100 (dark overlay) |
| **Inactive** | No battle detected | No state detected for > 2 seconds |

## Detection Logic

### 1. Pause Button Visibility Check (→ Active)
- ROI: Right side of header strip (maps to ~92-100% X, 0.5-5% Y of full frame)
- Scans for white pixels (R,G,B all > 200) and blue pixels (B>150, B>R+20, B>G)
- Requires > 5% white pixel ratio AND < 10% blue pixel ratio
- Blue rejection prevents home screen UI (blue-tinted) from triggering Active
- **Result**: `true` → Active state

### 2. PAUSE Template Matching (→ Paused)
- ROI: Center-top of screen (x=38-62%, y=13-19%) where PAUSE dialog header appears
- Uses NCC (Normalized Cross-Correlation) template matching against `templates/pause.png`
- Template extracted from battle-pause.png, stored as grayscale
- Candidate ROI is extracted, converted to grayscale, resized to template size, then NCC computed
- **NCC > 0.6**: Paused (definitively confirmed). battle-pause scores ≈1.0, others <0.1
- Only checked when Active condition is not met

### 3. Center Brightness Check (→ Paused / Slow fallback)
- Source: 10x10 pixel sample from frame center (computed by Producer)
- ITU-R BT.601 perceived brightness formula
- Only checked when pause button is not visible AND NCC template match is below threshold
- **> 150**: Paused (white PAUSE menu dominates center)
- **< 100**: Slow (dark skill confirmation overlay)

### 4. Home Screen Rejection
- Home screen has blue-tinted UI elements (blue_ratio ~0.26) in the pause button ROI
- Active requires `blue_ratio < 0.10`, which rejects home screen
- Paused/Slow transitions only allowed from non-Inactive state (prevents home screen brightness from triggering false Paused)

### 5. Hysteresis
- State transitions to Inactive only after 2 seconds without Active/Paused/Slow detection
- Prevents flickering during brief transitions

## Module: `analyze_battle_state()`

Standalone function in `combat_intel.rs` that takes full-frame RGB data and returns `BattleState`:
- Uses `check_pause_button_debug()` for white_ratio + blue_ratio (92-100% X, 0.5-5% Y)
- Uses `detect_pause_text()` for NCC template matching (38-62% X, 13-19% Y)
- Uses `calculate_center_brightness()` for center 25-75% region brightness
- Same thresholds as the visual_intercept consumer

Note: `visual_intercept.rs` uses the same logic with RGBA variant (`detect_pause_text_rgba()` + `sample_center_brightness` + header strip ROI).

## Functions

| Function | Description |
|----------|-------------|
| `analyze_battle_state()` | フルフレームRGBから`BattleState`を判定（テスト・診断用） |
| `detect_pause_text()` | 画面中央上部(38-62% X, 13-19% Y)のPAUSEダイアログをNCCテンプレートマッチングで検出。NCCスコアを返す |
| `detect_pause_text_rgba()` | RGBA版のPAUSEテンプレートマッチング。Producer（キャプチャスレッド）から呼ばれる |
| `check_pause_presence_in_wide_roi()` | ヘッダーストリップROI（85-100% X）からポーズボタン可視性を判定。`visual_intercept.rs`のConsumerから呼ばれる主要関数 |
| `check_pause_button_visible()` | フルフレームからポーズボタン可視性を判定 |
| `check_pause_button_debug()` | ROI内の青/白ピクセル比率を返す（診断用） |
| `calculate_center_brightness()` | フレーム中央50%のITU-R BT.601輝度を算出 |

## Rust Modules
- **`core::combat_intel`**: State detection logic
- **Dependencies**: `image` crate (grayscale conversion, resize), `templates/pause.png`
