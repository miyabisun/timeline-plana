# Resource Survey (資源調査)

## Overview
バトル画面右下のEXスキルコストゲージをリアルタイムで読み取るモジュール。
コストゲージは菱形（平行四辺形）のボックスが横一列に並んだUI要素で、青色の充填度合いからコスト値を算出する。

## ゲーム仕様: コストシステム

### 基本仕様
- **最大コスト**: 基本10固定
- **コスト範囲**: 0.0 〜 max_cost（小数値、リアルタイムで増加）
- コストはEXスキル使用で消費され、時間経過で自動回復する

### 最大コストの変動条件

| 条件 | max_cost | 備考 |
|------|----------|------|
| 通常 | **10** | 大多数のケース |
| スペシャル生徒の固有武器Lv4 | **+0.5/人** | 固有武器Lv2で性能完成するため、Lv4まで上げる先生はほぼいない |
| 通常バトル（スペシャル枠2） | 最大11 | 10 + 0.5×2 |
| 制約解除決戦（スペシャル枠4） | 最大12 | 10 + 0.5×4 |

**実用上の結論**: max_cost = 10 と仮定して問題ない。検出結果が0や1になることはありえない。

### max_cost の外部提供

`read_cost_gauge()` は `max_cost` を外部パラメータとして受け取る。
現在はデフォルト値10を使用。将来的にはタイムライン編成情報（スペシャル枠の固有武器Lv4持ち人数）から算出可能。

---

## Processing Pipeline

```
ROI Extraction → Skew Correction → Blue/Filled Profile → Even Grid (max_cost) → Box Classification → Cost Calculation
```

## ROI Coordinates

| Parameter | Value | For 3416x1993 |
|-----------|-------|---------------|
| X Start | 64% | 2186px |
| X End | 89% | 3040px |
| Y Start | 91.0% | 1814px |
| Y End | 93.2% | 1857px |

Size: ~854×43px

## Skew Correction

コストボックスは上辺が右にせり出した平行四辺形。水平シアー変換で長方形に補正する:

```
x' = x + 0.175 × y   (下の行ほど右にシフト)
```

**Skew Factor**: 0.175（目視選定）

---

## Algorithm: 三重信号分類

### Step 1 — プロファイル構築

3つのプロファイルを構築（高さ30-70%帯をスキャン）:

| プロファイル | 判定条件 | 検出対象 |
|---|---|---|
| Blue | `B > 120, B > R+30, B > G` | Active状態の明るい青 |
| Filled (広義) | `B > 70, B > R+20, B > G+10, R < 160, sum > 170` | Pause/Slow状態の暗いティール色含む |
| Glow | `G >= 250, B >= 250, R < 160` | コストボックス充填時のシアン発光エフェクト |

### Glow Detection（発光エフェクト対応）

コストが整数値に達した瞬間（例: 5.9→6.0）、新たに満タンになったボックスが明るいシアン色（R≈10-90, G≈255, B≈255）で一時的に発光する。
このピクセルはG≈Bであるため Blue/Filled プロファイルでは検出できない。

- **Glow box**: glow_avg > 0.70 → 強制的にFull（1.0）
- **Spillover抑制**: Glow boxの直後のボックスはシアン光の漏れ込みがあるため、blue_profileのみで分類（filled_profileを無視）

### Step 2 — 等間隔グリッド生成

外部から提供された `max_cost` に基づき、ROI幅を均等分割:

```
pitch = corrected_width / max_cost
box[i] = (i * pitch, (i+1) * pitch - 1)
```

### Step 3 — 分類ロジック

| 条件 | 分類 | コスト値 |
|------|------|---------|
| avg_blue > 0.85 | Full | 1.0 |
| avg_blue 0.10-0.85 | Partial | max(avg_blue, avg_filled) |
| avg_filled > 0.85 | Full (pause/slow) | 1.0 |
| avg_filled < 0.05 | Empty | 0.0 |
| otherwise | Partial (pause/slow) | avg_filled |

### Cost Calculation
```
current = full_count + partial_ratio
max_cost = 外部パラメータ（デフォルト10）
confidence = 1.0（max_costは外部保証）
```

---

## Output

```rust
pub struct CostValue {
    pub current: f32,    // 現在コスト（例: 5.7）
    pub max_cost: u32,   // ボックス総数（外部提供）
    pub confidence: f32, // 信頼度スコア（0.0-1.0）
}
```

---

## Rust Modules

- **`core::resource_survey`**: Cost gauge reading
- **Dependencies**: `image` crate (test only)

## Functions

| Function | Description |
|----------|-------------|
| `read_cost_gauge(rgb, w, h, max_cost)` | メインAPI: RGB ROIとmax_costからコスト値を算出 |
| `apply_cost_skew()` | 平行四辺形→長方形のシアー補正 |
| `is_blue_pixel()` | 青色ピクセル判定（Active状態） |
| `is_filled_pixel()` | 広義充填ピクセル判定（全状態対応） |
| `is_glow_pixel()` | 発光エフェクトピクセル判定（シアンフラッシュ） |
| `build_blue_profile()` | 列ごとの青色比率プロファイル構築 |
| `build_filled_profile()` | 列ごとの広義充填ピクセル比率プロファイル |
| `build_glow_profile()` | 列ごとの発光ピクセル比率プロファイル |
| `classify_box_broad()` | 三重信号（blue+filled+glow）によるボックス分類 |
