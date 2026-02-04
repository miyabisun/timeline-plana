# Countdown Monitor (カウントダウン監視)

## Overview
Extracts and processes the battle countdown timer from screen captures. In Blue Archive, defeating the boss before time runs out is critical—time is the ultimate constraint in combat optimization.

## Processing Pipeline

```
Full Frame → ROI Extraction → Skew Correction → Template Matching → Validation
```

## ROI Coordinates (Optimized for 30fps OCR)

| Parameter | Value | For 3416x1993 |
|-----------|-------|---------------|
| X Start | 85% | 2904px |
| X End | 93% | 3177px |
| Y Start | 3.5% | 70px |
| Y End | 6.3% | 126px |

Size: ~273×56px = ~15k pixels (digits only, vertically centered)

## Skew Correction

The Blue Archive timer uses italic text (top leans right). A horizontal shear transform straightens it:

```
x' = x + 0.25 × y
```

**Skew Factor**: 0.25 (based on colon ":" alignment)

---

## OCR Strategy

### Timer Format

```
0 M : S S . m m m
│ │   │ │   │ │ │
│ │   │ │   │ │ └─ ms digit 3 (0, 3, or 7)
│ │   │ │   │ └─── ms digit 2 (0, 3, or 6)
│ │   │ │   └───── ms digit 1 (0-9)
│ │   │ └───────── seconds digit 2 (0-9)
│ │   └─────────── seconds digit 1 (0-5)
│ └─────────────── minutes digit 2 (0-9)
└───────────────── minutes digit 1 (always 0)
```

### Optimization Rules & Validation

| 桁 | 取りうる値 | 備考 |
|----|-----------|------|
| 分1桁目 | **0のみ** | コンテンツは10分未満 |
| 分2桁目 | 0-9 | |
| 秒1桁目 | 0-5 | **最大59秒** (71, 81等はエラー) |
| 秒2桁目 | 0-9 | |
| ms1桁目 | 0-9 | |
| ms2桁目 | **0, 3, 6のみ** | 30fps制約 |
| ms3桁目 | **0, 3, 7のみ** | 30fps制約 (00/33/67) |

**認識対象**: 実質6桁（分1桁目は固定、ms下2桁は3パターン）

### Validation Rules (Strict)
1. **Minutes**: Must be < 10 (Tens place = 0).
2. **Seconds**: Must be 0-59. (Tens place <= 5).
3. **Milliseconds**: Must end in 00, 33, or 67.
   - Snap values: `0-16` -> `00`, `17-49` -> `33`, `50-83` -> `67`, `84-99` -> `00` (next cycle)

**認識対象**: 実質6桁（分1桁目は固定、ms下2桁は3パターン）

### 30fps制約によるmsパターン

| パターン | 出現条件 |
|---------|---------|
| `.X00` | フレーム 0, 3, 6... |
| `.X33` | フレーム 1, 4, 7... |
| `.X67` | フレーム 2, 5, 8... |

---

## Time Speed Modes

時間予測は不可能（毎フレームOCR必須）

| 状態 | 速度倍率 | 検出方法 |
|------|---------|---------|
| ポーズ | 0x | `combat_intel` |
| スキル準備 | 0.2x | `combat_intel` |
| 速度1 | 1x | 速度ボタン検出 |
| 速度2 | 1.33x | 速度ボタン検出 |
| 速度3 | 1.67x | 速度ボタン検出 |

---

## Mouse Cursor Occlusion

マウスカーソルがタイマーを遮る可能性あり

### 対策

1. **桁単位の信頼度スコア**: 各桁独立で認識・評価
2. **低信頼度桁の補完**: 他桁から推論 or 前フレーム維持
3. **時系列整合性**: 時間は単調減少（異常検出）
4. **複数手段の複合判断**: combat_intel + OCR の相互検証

### フォールバック

| 遮蔽度 | 対応 |
|--------|------|
| 1-2桁 | 他桁・パターンから推論 |
| 3桁以上 | 前フレーム維持 + 警告 |
| 全桁 | 認識不能フラグ |

---

## Performance Estimation

### 時間予算

| 項目 | 値 |
|------|-----|
| Target FPS | 30fps |
| Frame budget | **33.3ms** |

### 処理時間試算

| 処理 | 時間 | 累積 |
|------|------|------|
| ROI抽出 | 0.1ms | 0.1ms |
| スキュー補正 | 0.2ms | 0.3ms |
| テンプレートマッチング (6桁) | 1.0ms | 1.3ms |
| 信頼度計算 | 0.1ms | 1.4ms |
| 異常値検出 | 0.1ms | **1.5ms** |

**結論**: 33.3msの約**4.5%**で完了 ✅

---

## Rust Modules

- **`core::countdown_monitor`**: ROI extraction, skew correction, OCR
- **Dependencies**: `image` crate

## Functions

| Function | Description |
|----------|-------------|
| `extract_timer_roi()` | Crops timer region from full frame |
| `apply_skew_correction()` | Applies affine shear transform |
| `process_timer_region()` | Combined pipeline |
| `recognize_digits()` | Template matching OCR (future) |
| `validate_time()` | Time validation with rules (future) |
