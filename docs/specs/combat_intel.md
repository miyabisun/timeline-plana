# Combat Intel (戦闘情報)

## Overview
Analyzes screen captures to detect the current battle state. This enables the application to react appropriately to pauses, slow-motion effects, and active combat phases.

## States

| State | Description | Detection Method |
|-------|-------------|------------------|
| **Active** | Normal battle in progress | Timer visible (white pixels in ROI), center brightness > 80 |
| **Paused** | PAUSE menu is open | Center brightness > 180 (white menu overlay) |
| **Slow** | Slow-motion effect active | Timer visible but center brightness < 80 (dark overlay) |
| **Inactive** | No battle detected | Timer not visible |

## Detection Logic

1. **Timer Visibility Check**
   - ROI: 92-100% X, 0.5-5% Y (top-right corner)
   - Scans for white pixels (R,G,B all > 200)
   - Requires > 5% white pixel ratio to confirm visibility

2. **Center Brightness Check**
   - ROI: 25-75% X, 25-75% Y (center region)
   - Calculates average brightness of sampled pixels
   - Used to distinguish Paused (bright) from Slow (dark)

## Rust Modules
- **`core::combat_intel`**: State detection logic
- **Dependencies**: None (pure pixel analysis)

## Integration
Called by `visual_intercept` on each captured frame to emit battle state events.
