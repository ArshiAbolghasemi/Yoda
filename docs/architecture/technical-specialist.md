# Technical specialist

**Role.** Form a directional view of every asset from trend, momentum, oscillator
and volume structure.

| | |
|---|---|
| Input | `features['technical']` — `(T, N, 60)`, the causal indicators from `yoda.data` |
| Output | `z_technical` `(N, 8)` and `μ̂_technical` `(N,)` |
| Enters at | `fused_μ` — it is a **directional** source (`MU_SOURCES`) |
| Code | `yoda/specialists/technical.py`, `yoda/specialists/base.py` |
| Contract | `Specialist` — `fit` / `encode` / `predict` |

## How it works

A four-stage sklearn pipeline, shared by all three specialists:

```
SimpleImputer(median, add_indicator=True) → StandardScaler → PCA(z_dim) → Ridge(α)
```

The `PCA` output is the representation `z_i`; the `Ridge` head on top of `z_i` is
the view `μ̂_i`.

The `(T, N, F)` cube is flattened to `(T·N, F)`, so the model is **cross-sectional
and shares parameters across assets**. That is what makes 36 assets trainable on
~1250 rows: per-asset models would have 36 × fewer samples each and no way to
borrow strength from the rest of the panel.

## Why `add_indicator=True` matters

Eight of the 60 indicators are NaN for every FX row, by design — Yahoo publishes no
FX volume:

`obv`, `cmf_20`, `mfi_14`, `force_index_13`, `ease_of_movement_14`, `vwap_14`,
`volume_price_trend`, `negative_volume_index`

Imputing them to zero would tell the model "volume was flat" when the truth is
"there is no volume data". The imputer instead fills with the **training** median
and appends a binary missing-indicator column, so the linear head can offset the
fill wherever the flag is set.

Two tests pin this: `test_fx_volume_indicators_stay_nan_rather_than_zero` (the
panel keeps the NaNs) and
`test_specialists_tolerate_the_all_nan_fx_volume_columns` (encoding survives them).

## Training

```python
specialist.fit(
    features["technical"][train_rows][:, universe],
    panel.targets[train_rows][:, universe],
)
```

- Rows with a non-finite target or an all-NaN feature vector are dropped.
- `n_components` is clamped to `min(z_dim, usable − 1, n_features)`.
- Imputer medians, scaler statistics, the PCA basis and the ridge coefficients are
  estimated **here and only here** — never re-estimated at inference, which is what
  keeps the training window sealed.

```
specialist_fitted name=technical rows=44059 features=60 z_dim=8
```

## Inference

`encode(X)` and `predict(X)` transform with the frozen pipeline. Both accept
`(N, F)` for a single date or `(T, N, F)` for a whole window and preserve the
leading shape. Non-finite predictions are zeroed rather than propagated into the
optimizer.

Out-of-sample rank correlation against next-day returns runs around 0.017 on the
default split — small, which is what a one-day equity forecast looks like when it
is honest.

## Settings

| Setting | Default | Effect |
|---|---|---|
| `SPECIALISTS__Z_DIM` | 8 | Width of `z_i`; also the width of `fused_z` in the RL observation |
| `SPECIALISTS__RIDGE_ALPHA` | 10.0 | Shrinkage on the view head |
| `SPECIALISTS__SEED` | 7 | PCA randomisation |

## See also

- [volatility-specialist.md](volatility-specialist.md) — same pipeline, magnitude target
- [news-specialist.md](news-specialist.md) — same pipeline, very different features
- [tail-voi-gate.md](tail-voi-gate.md) — what decides how much this view counts
