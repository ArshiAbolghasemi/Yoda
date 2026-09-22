# Tail-VoI gate

The centerpiece. Everything else in the stack is a competent implementation of a
known technique; this is the part with a claim attached to it.

**Role.** Decide how much each information source is worth **right now**, measured
in tail risk rather than in accuracy.

| | |
|---|---|
| Input | `SpecialistOutput` for one date — `z_i` and `μ̂_i` per source |
| Output | `GateOutput(g, fused_mu, fused_z, delta_hat)` |
| Code | `yoda/tailvoi/` |
| Contract | `Gate` — `gate(spec) → GateOutput` |

## The idea

Accuracy-weighted ensembling asks *"which source is right most often?"*. That is
the wrong question for a portfolio that is judged on its tail. A source can be
mediocre on average and still be the one that keeps you out of the drawdown —
and a source can have a fine hit rate while contributing nothing when it matters.

Tail-VoI asks instead: **if I removed this source, how much worse would the tail of
the portfolio I actually chose have been?**

---

## 1. The counterfactual generator — the supervisor

`yoda/tailvoi/counterfactual.py` answers that question by brute force. For each
sampled training date:

1. Solve the full stack with an even gate → `w_full`, and score it: `ρ(I)`.
2. For each source `i`: zero its gate weight, renormalise the rest, **re-solve the
   entire downstream** → `w₋ᵢ`, and score it: `ρ(I₋ᵢ)`.
3. `Δᵢ = ρ(I₋ᵢ) − ρ(I)`.

A **positive** `Δᵢ` means the book you would have held *without* source `i` had a
worse tail — the source earned its place.

### Ablating each source at the stage it enters

This is the detail that makes the targets mean anything:

| Source | What removing it changes |
|---|---|
| `technical` | drops a contributor to `fused_μ` |
| `news` | drops a contributor to `fused_μ` |
| `volatility` | **de-conditions the copula** — the scenario set changes, `μ` does not |

Routing every ablation through `μ` would measure the loss of a mean signal the
volatility channel never provided. See
[volatility-specialist.md](volatility-specialist.md#why-it-never-touches-the-mean).

### Choosing ρ

`TAILVOI__RHO`:

| Value | ρ is | Trade-off |
|---|---|---|
| `realized` (default) | CVaR₍α₎ of the portfolio over the next `TAILVOI__RHO_WINDOW` (60) trading days | Matches the spec's wording; noisier, needs a forward window |
| `reference` | Every counterfactual book scored under one common full-information scenario set | Lower variance, no forward window, but measures model risk rather than realised risk |

A single date has no realised CVaR, which is why this is a choice at all. The
forward window is only ever taken on **training** rows, where forward returns are
labels like any other.

### Cost and sampling

`1 + K` convex solves per state (~35 ms each at
`TAILVOI__COUNTERFACTUAL_SCENARIOS` = 500), so it runs **offline on training rows
only**: 250 states × 3 sources ≈ 1000 solves ≈ 35 s per fold.

`sample_positions` takes evenly spaced dates, leaving a `rho_window` margin at the
end so every state has a full forward window. Every counterfactual starts from the
same equal-weight **reference book**, so `Δ` measures the information, not the path
the backtest happened to take.

```
counterfactual_targets states=250 rho=realized mean={'delta_technical': …, 'delta_volatility': …}
```

**Read those means.** They are the honest answer to "is this source worth anything
in the tail?" before any learning happens. On the default panel they come out
slightly *negative* — the sources are not adding tail value at these settings. That
is a finding, not a bug.

### How to position this in a paper

Leave-one-out counterfactual risk attribution over *information sources* — the
close relative of Shapley-style expected-shortfall attribution. The formula is not
the contribution; what the gate does with it is. Claiming the formula would invite
exactly the reviewer you do not want.

---

## 2. `TailVoIGate` — the amortisation

Brute force is unusable inside a backtest, so a small regressor learns it:

```
StandardScaler → MLPRegressor(hidden_layer_sizes=TAILVOI__HIDDEN, max_iter=2000)

G_φ : state summary  →  Δ̂ ∈ ℝ^K
g   = softmax(Δ̂ / τ)                    # TAILVOI__MODE=softmax
g   = 1[Δ̂ > threshold] / count          # TAILVOI__MODE=threshold
```

### The state summary

Fixed-width and cross-sectional, so it does not depend on how many assets are
investable today. Per source:

| Component | Width |
|---|---|
| mean of `z_i` across assets | `z_dim` = 8 |
| `mean\|μ̂ᵢ\|`, `std(μ̂ᵢ)`, `std(z_i)` | 3 |

`z_dim + 3 = 11` per source; 33 for three sources, 22 for two.

### Scaling

Targets are divided by `std(Δ)` before fitting, so `TAILVOI__TEMPERATURE` means the
same thing regardless of the units risk happens to be in — CVaR differences live
around `1e-3` and a raw softmax over them would be indistinguishable from uniform.
`delta_hat` is returned on `GateOutput` in original units for inspection.

Unfitted, the gate logs a warning and degrades to an even gate rather than failing.
The stack has to be assemblable before the targets exist — that is how
`build_stack` can run before stage 3.

### Training and inference

**Training:** `fit(CounterfactualTargets)`.

```
tailvoi_gate_fitted states=250 sources=technical,volatility,news train_r=0.99
```

`train_r` is the in-sample fit of `G_φ` to `Δ`. High is expected and is **not**
evidence the gate works. The evidence is the ablation below.

**Inference:** `gate(spec)` — one `summarise` call and one forward pass. Negligible
next to the convex solve that follows it.

---

## 3. Fusion — what `g` actually does

```python
fused_μ = Σ_{i ∈ MU_SOURCES}  (gᵢ / Σ_directional g) · μ̂ᵢ
fused_z = Σ_{all sources}      gᵢ · zᵢ
s       = clip(g_volatility / Σg · k, 0, 1)      # copula conditioning strength
```

`fused_z` (cross-sectionally averaged) is what the RL agent sees as its
representation of the market; `s` is what reaches the copula. `MU_SOURCES` and
`CONDITIONING_SOURCES` live in `yoda/tailvoi/base.py` and are the single place that
decides which channel is which.

---

## 4. Baseline gates — the ablation the claim rests on

| Gate | Importance signal | Why it is the right control |
|---|---|---|
| `EqualWeightGate` | none | The floor. If Tail-VoI cannot beat this, it does nothing. |
| `AccuracyGate` | Spearman IC on the training window, softmaxed | Rewards being *right*, with no notion of tail value. The volatility channel is scored against `\|r\|`, not `r`. |
| `AttentionGate` | A learned torch attention over the source summaries — one shared scorer applied per source chunk, 300 Adam epochs at lr 1e-2 | Learned, but supervised by **mean** prediction error (MSE of fused `μ` vs realized), not by risk. Isolates "learned" from "learned on the right objective". |

All four implement the identical `Gate` interface, so swapping one in changes
nothing else in the stack:

```bash
./scripts/train-tail-voli-risk.sh --gate tailvoi      # or accuracy | attention | equal_weight
```

The `gate` experiment family runs all four on the same panel, splits and intervals.
That table is the headline result — see
[inference.md](inference.md#the-experiment-harness).

## Settings

| Setting | Default | Effect |
|---|---|---|
| `TAILVOI__SOURCES` | `technical, volatility, news` | Which channels exist at all |
| `TAILVOI__TEMPERATURE` | 1.0 | Softmax sharpness over `Δ̂` |
| `TAILVOI__MODE` | `softmax` | `threshold` gives a hard on/off gate |
| `TAILVOI__HIDDEN` | `(32,)` | `G_φ` architecture |
| `TAILVOI__N_STATES` | 250 | Counterfactual states per fold — the main cost lever |
| `TAILVOI__COUNTERFACTUAL_SCENARIOS` | 500 | Scenario count inside the generator |
| `TAILVOI__RHO` / `__RHO_WINDOW` | `realized` / 60 | The risk functional |
| `TAILVOI__SEED` | 13 | |

## What to watch

The `gate_g_*` columns in `ledger.parquet` are the live gate weights at each
rebalance. **A gate pinned at a constant is a gate that has stopped contributing** —
check that before trusting any Tail-VoI result.
