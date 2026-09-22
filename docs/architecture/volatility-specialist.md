# Volatility specialist

**Role.** Form a *risk* view — how large the next move will be, not which way.

| | |
|---|---|
| Input | `features['volatility']` — `(T, N, 8)` |
| Output | `z_volatility` `(N, 8)` and `σ̂` `(N,)`, floored at zero |
| Enters at | **the copula conditioning** — never `μ` |
| Code | `yoda/specialists/volatility.py` |
| Contract | `Specialist` — `fit` / `encode` / `predict` |

## Features

| Column | Source |
|---|---|
| `atr_7`, `atr_14`, `atr_21` | divided by adjusted close, so assets are comparable |
| `bb_width_20` | Bollinger band width |
| `keltner_width` | Keltner channel width |
| `realized_vol_5`, `realized_vol_20`, `realized_vol_60` | causal rolling std of the aligned returns, windows from `PANEL__REALIZED_VOL_WINDOWS` |

Realized volatility only — the dataset carries no options or implied-vol surface,
and inventing one would be the kind of thing that looks like a feature and behaves
like a bug.

ATR is quoted in price units, so a $500 stock and a 1.09 FX rate are not
comparable until you divide by price. `alignment.py` does that when it builds the
cube.

## How it differs from the technical specialist

Same four-stage pipeline
([technical-specialist.md](technical-specialist.md#how-it-works)), one override:

```python
def _target(self, y):  # the head regresses magnitude, not direction
    return np.abs(y)


def predict(self, X):  # a negative volatility is not a thing
    return np.maximum(super().predict(X), 0.0)
```

## Why it never touches the mean

This is the distinction that makes the Tail-VoI targets mean anything.

A volatility view does not say "buy". It says "the distribution is wider right
now". Routing it into the expected-return vector would be a category error — and,
worse, it would make *"ablate the volatility source"* test the wrong thing: you
would be measuring the loss of a mean signal that was never a mean signal.

Instead its gate weight controls `conditioning_strength`, which blends the
specialist's `σ̂` against the copula's unconditional EWMA scale:

```python
σ̂_scaled = σ̂ · √(π/2)                            # E|r| → standard deviation
regime    = (1 − s)·scale_uncond + s·σ̂_scaled     # s ∈ [0, 1]
scen      = copula.conditional(state(regime)).sample(n)
```

`s = clip(g_volatility / Σg · k, 0, 1)`. An even gate over `k` sources gives
`s = 1` — condition normally. A gate that zeroes the volatility source gives
`s = 0` — fall back to the unconditional scenario set, which is exactly what the
counterfactual generator needs when it ablates this channel.

`test_dropping_volatility_deconditions_the_scenarios` asserts the scenario standard
deviations actually change when the source is dropped.

## Training and inference

Identical to the technical specialist. The target passed in is the same
`panel.targets`; the `_target` hook takes its absolute value internally, so callers
never have to know.

```
specialist_fitted name=volatility rows=44065 features=8 z_dim=8
```

## See also

- [tail-model.md](tail-model.md) — what `σ̂` conditions
- [tail-voi-gate.md](tail-voi-gate.md) — how `g_volatility` becomes `s`
