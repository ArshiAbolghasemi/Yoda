# Architecture

## The one-sentence version

Several agents form independent views of the market; a gate decides how much each
view is worth *in the tail*; a copula turns that into a scenario set; a convex
solver turns the scenario set into a portfolio; and a risk policy — static rule or
RL agent — sets the three knobs the solver obeys.

```
                         ┌──────────────── the ONLY seam ─────────────────┐
                         │                                               │
  aligned panel          │        RiskParamPolicy.act(state) → (λ, B, c) │
       │                 │        ├── StaticRiskPolicy   (tail_voli_risk)│
       ▼                 │        └── RLRiskPolicy    (tail_voli_risk_rl)│
  specialists            │                                               │
  ├── technical  → μ̂ᵗ  ─┐│                                               │
  ├── volatility → σ̂   ─┼┼───────────────┐                              │
  └── news       → μ̂ⁿ  ─┘│               │                              │
       │                 │               │                              │
       ▼                 │               ▼                              ▼
  Tail-VoI gate ──g──► fused μ      copula conditioning          DRO-CVaR solver ──► w
       ▲                                 │                              ▲
       │                                 └──── scenarios ───────────────┘
  counterfactual Δ (offline, training only)
```

Everything left of the seam is shared byte-for-byte between the two pipelines.
`tail_voli_risk.py` and `tail_voli_risk_rl.py` differ in exactly one function,
`make_policy`. That is the design invariant the whole layout exists to protect.

## The components

One page each. Read the ones you are working on; they assume this page and nothing
else.

### Agents

| Agent | Role | Enters at |
|---|---|---|
| [technical-specialist.md](technical-specialist.md) | Directional view from the 60 causal indicators | `fused_μ` |
| [volatility-specialist.md](volatility-specialist.md) | Magnitude view — how wide the distribution is | the copula conditioning |
| [news-specialist.md](news-specialist.md) | Headlines → numbers, through OpenJev | `fused_μ` |
| [openjev-specialists.md](openjev-specialists.md) | Optional OpenJev 27B backend for any channel — calibrated probabilities | per channel |
| [tail-voi-gate.md](tail-voi-gate.md) | **The centerpiece.** What each source is worth *in the tail* | decides `g` |
| [risk-policy.md](risk-policy.md) | **The seam.** Chooses `(λ, B, c)` — static rule or SAC agent | drives the solver |
| [cio-agent.md](cio-agent.md) | Decides whom to trust *and* the risk stance — fills the gate and policy sockets at once | both |

### Supporting components

| Component | Role |
|---|---|
| [tail-model.md](tail-model.md) | Student-t copula: the joint distribution the views are expressed against |
| [optimizer.md](optimizer.md) | The DRO-CVaR allocator and its robustification modes |

### Procedures

| Document | Covers |
|---|---|
| [training.md](training.md) | The complete training procedure, stage by stage, both pipelines |
| [inference.md](inference.md) | The walk-forward loop, artifacts, offline scoring, live checklist |
| [experiments.md](experiments.md) | Every baseline and ablation, what each isolates, and how to read the results |

## Contracts

Every module composes through `yoda/common/types.py`. Build against these, never
against a concrete class:

| Contract | Method | Meaning |
|---|---|---|
| `Specialist` | `fit(X, y)` / `encode(X)` / `predict(X)` | One information source → representation `z_i` and view `μ̂ᵢ` |
| `TailModel` | `fit` / `conditional(state)` / `sample(n)` / `tail_stats(w, scen, α)` | The joint return distribution and its tail |
| `Gate` | `gate(spec) → GateOutput` | How much each source counts, and the fusion it implies |
| `DROCVaROptimizer` | `solve(μ, scen, rp, w_prev) → w` | Long-only weights on the simplex |
| `RiskParamPolicy` | `act(state) → RiskParams` | **The seam.** `(λ, B, c)` — never weights, never `α` |

The value objects that travel between them: `SpecialistOutput`, `TailScenarios`,
`GateOutput`, `MarketState`, `RiskParams`.

### `RiskParams` — what a policy may and may not move

```python
RiskParams(lam, budget, turnover_penalty, alpha=0.05)
```

`λ` (risk aversion), `B` (CVaR budget) and `c` (turnover penalty) are the action
space. **`α` is a fixed hyperparameter and is never an action** — an agent that can
move its own risk-measure confidence can make its risk look small by redefining
risk. `RiskParams.__post_init__` rejects `α ∉ (0,1)` and negative `λ`/`c`.

## Data flow

### 1. Panel alignment — `yoda/common/alignment.py`

`build_panel(config) → AlignedPanel`. Reads
`data/financial_dataset.parquet` (36 assets × daily rows) and produces a single
aligned tensor set on one calendar.

| Field | Shape | Notes |
|---|---|---|
| `dates` | `(T,)` | one common trading calendar |
| `returns` | `(T, N)` | simple returns; `0.0` where unavailable |
| `prices` | `(T, N)` | adjusted close, forward-filled ≤ `PANEL__MAX_FORWARD_FILL` rows |
| `available` | `(T, N)` bool | listed, priced, past per-asset warm-up |
| `features['technical']` | `(T, N, 60)` | the 60 causal indicators |
| `features['volatility']` | `(T, N, 8)` | 5 band/ATR columns + 3 realized-vol windows |
| `features['news']` | `(T, N, 7)` | OpenJev probabilities; see [news-specialist.md](news-specialist.md) |
| `targets` | `(T, N)` | forward return over `PANEL__TARGET_HORIZON` |

Default panel: **2018-01-02 → 2026-09-18, 2190 rows, 36 assets, 99.3% available.**

**Three gotchas this step exists to handle:**

- **Mixed calendars.** BTC trades 24/7, FX 24/5, equities the NYSE session.
  Everything is reindexed onto the NYSE calendar *before* returns are computed, so
  a BTC weekend move folds into Monday's return instead of vanishing. A test pins
  this (`test_btc_weekend_moves_are_folded_not_dropped`).
- **The calendar is the union of equity dates, not the intersection.** Intersecting
  truncates the panel to `DOW`'s 2019 listing and destroys the COVID stress window.
  Latecomers are handled by the `available` mask, and each fold trades only
  `panel.universe(rows)` — assets available on ≥95% of that fold's training rows.
  With the default split, `DOW` is simply out of the training universe.
- **FX volume indicators are NaN by design.** Eight of the 60 have no FX volume
  behind them. They are never imputed to zero — see
  [technical-specialist.md](technical-specialist.md#why-add_indicatortrue-matters).

### 2–5. The rest

Specialists, gate, copula and optimizer each have their own page — see
[the components](#the-components) below. The object that wires them together is
`yoda/stack.py`:

```python
stack = build_stack(panel, config, train_rows, gate)  # fits on train_rows only
state, scen, gate_out = stack.state(position, w_prev)  # one date → observation
weights = stack.allocate(state, scen, risk_params)  # → simplex weights
```

`AllocationStack` is the whole system below the seam behind four methods. The
backtest engine, the counterfactual generator and the RL environment all drive it
through the same surface, which is why none of them can disagree about what the
model saw on a given day.

## No-lookahead discipline

This is enforced structurally, not by convention:

- **Fitting** takes explicit row indices. `build_stack(panel, config, train_rows, …)`
  cannot see anything else; the walk-forward split lives in the *caller*, in one
  place (`generate_folds`).
- **Targets are labels.** `future_return_Nd` is recomputed from aligned prices and
  used only to train specialists on training rows. A test asserts the last
  `horizon` rows have no target at all.
- **Execution.** At row `t` the stack sees features up to and including `t`, the
  policy chooses `RiskParams`, the optimizer chooses the book, and that book earns
  `returns[t+1]`. Between rebalances the book *drifts* with the market rather than
  being silently rebalanced for free.
- **Preprocessing.** Imputer medians, scaler statistics and PCA bases are fitted
  inside the training window and only transformed elsewhere.

## Where the code lives

| Package | Role |
|---|---|
| `yoda.common.types` | every shared contract |
| `yoda.common.alignment` | one calendar, returns, feature cubes, availability |
| `yoda.specialists` | the three view-forming agents |
| `yoda.specialists.jev` | OpenJev client, typed decision tasks, point-in-time cache |
| `yoda.copula` | Student-t tail model, conditional scenarios, stress paths |
| `yoda.optimizer` | The DRO-CVaR solver |
| `yoda.tailvoi` | counterfactual Δ generator, the learned gate, three baseline gates |
| `yoda.policy` | **the seam** — `StaticRiskPolicy`, `RLRiskPolicy` |
| `yoda.rl` | Gym environment and SAC training (RL pipeline only) |
| `yoda.backtest` | walk-forward execution; writes artifacts, computes no metrics |
| `yoda.evaluation` | offline scoring from those artifacts, by interval |
| `yoda.pipeline` | the shared stack, the two pipelines, the experiment harness |

## Extending it

The layout has one job: make the next change local.

- **A new information source** — implement `Specialist`, add a feature cube, add the
  name to `TAILVOI__SOURCES`. The gate picks it up automatically; if it enters `μ`
  rather than the copula conditioning, add it to `MU_SOURCES` in `tailvoi/base.py`.
- **A new gate** — implement `Gate`, add a branch to `fit_gate` in
  `pipeline/stack.py`, add an arm to `default_arms`. Nothing else moves.
- **A new tail model** — implement `TailModel`. GARCH-t marginals already have their
  slot (`COPULA__MARGINALS`) and raise `NotImplementedError` until written.
- **Direct-weight RL** — a policy that emits weights instead of `RiskParams`. It
  replaces the seam *and* the `allocate` call, and touches nothing below.
