# CIO agent

One decision-maker for the whole allocation. Where the main architecture splits
the decision into three measurable pieces, this agent does all three itself.

| | |
|---|---|
| Input | `SpecialistOutput` — what the three channels reported |
| Output | the gate `g`, the risk stance `(λ, B, c)`, **and** the portfolio weights |
| Fills | `Gate` **and** `RiskParamPolicy` **and** `DROCVaROptimizer` |
| Model inference | **none** — no OpenJev, no network, fully deterministic |
| Code | `yoda/cio/agent.py` |

```
             main stack                              CIO agent
  ┌────────────────────────────┐          ┌────────────────────────────┐
  specialists                             specialists
       │                                       │
  Tail-VoI gate      → g                       │
       │                                       ├─► one agent → g
  risk policy        → (λ, B, c)                ├─►           → (λ, B, c)
       │                                       └─►           → w
  DRO-CVaR           → w
```

## Why it exists

It is the **monolithic-manager control**. A single agent reading the analysts
and producing a portfolio is the common design in multi-agent trading systems;
this repository's premise is that *decomposing* that decision is worth
something. Running both on identical splits is the only honest way to find out,
and that requires actually building the thing being argued against.

## What it costs

Two guarantees the main stack provides are **not** available here. Both are
deliberate — a control that kept them would not be a control — and both need
stating in any write-up that quotes this arm's numbers.

**1. The CVaR budget is not enforced.** DRO-CVaR imposes `CVaR_α(w) ≤ B` as a
hard constraint. The CIO solves a long-only mean-variance program instead, so
`B` is advisory and a run can breach its own budget. Realised CVaR is still
written to the ledger, so a breach is visible after the fact rather than
prevented.

**2. Attribution is gone.** Tail-VoI exists to measure what each source is
worth in the tail. When one agent consumes all three channels and emits
weights, there is no longer a quantity being measured.

What *is* still enforced: the simplex, the per-asset cap and the turnover
penalty, because those come from the convex program it solves.

## How it decides

### Trust → `g`

Each channel is scored by the **cross-sectional dispersion** of its view:

```python
score_i = (std(μ̂_i) − centre_i) / scale_i
g       = softmax(score / CIO__TEMPERATURE)
```

A channel that differentiates between assets today is saying something; one
reporting nearly the same number for everything is not, whatever its level.
Dispersion is comparable across a directional view and a risk view in a way a
raw mean is not.

`centre_i` and `scale_i` are fitted on `CIO__FIT_STATES` training dates. Without
that normalisation the softmax would compare a volatility view in daily
standard deviations against a directional view in basis points, and the larger
unit would win every day regardless of content.

### Stance → `(λ, B, c)`

A volatility-targeting rule on the current scenario tail:

```python
ratio = clip(CIO__VOL_TARGET / CVaR_now, 0.2, 5.0)
λ      = static.lam / ratio          # less risk aversion when the tail is quiet
B      = static.budget · ratio       # more headroom when the tail is quiet
c      = static.turnover_penalty     # not a CIO judgement
```

`α` is never touched — no agent redefines the risk measure it is judged by.

### Weights → `w`

Long-only mean-variance on the scenario covariance, with the `λ` and `c` it
just chose, subject to the simplex and the per-asset cap. Deliberately **not**
the DRO-CVaR program: a control that borrowed the stack's risk machinery would
not be measuring what the stack's risk machinery is worth.

## Using it

```bash
./scripts/train-tail-voli-risk.sh --gate cio
```

Selecting the CIO as the gate hands it the whole decision: `fit_gate` installs
the same object as `stack.optimizer`, so DRO-CVaR is out of the loop for that
run. The log says so explicitly:

```
cio_installed gate+policy+optimizer replaced by the CIO
```

There is no separate `--policy cio`: the CIO is all-or-nothing. A half-CIO
filling only the policy socket would be a volatility-targeting rule wearing a
misleading name, and `policy_vol_target` already covers that comparison.

## Settings

| Setting | Default | Effect |
|---|---|---|
| `CIO__TEMPERATURE` | 1.0 | Softmax sharpness over channel dispersion |
| `CIO__FIT_STATES` | 250 | Training dates sampled to normalise each channel |
| `CIO__VOL_TARGET` | 0.02 | Realised-CVaR target driving the risk stance |

## The open question

Whether one agent allocating attention, risk and capital together beats three
components each measured on its own objective is exactly the kind of claim this
repository is built to test rather than assert. `cio_full` sits in the matrix
against `policy_static`, `policy_rl` and `policy_vol_target` on identical
splits, and the answer — positive, neutral or negative — is whatever the table
says.

## See also

- [tail-voi-gate.md](tail-voi-gate.md) — the gate it replaces
- [risk-policy.md](risk-policy.md) — the seam it replaces
- [optimizer.md](optimizer.md) — the DRO-CVaR program it bypasses
- [experiments.md](experiments.md) — where its arms sit
