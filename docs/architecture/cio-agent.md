# CIO

The decision-maker that turns specialist views into a portfolio. It *contains*
the two judgements and the solver, and exposes one thing — weights.

```
                       CIO
      ┌──────────────────────────────────┐
      │  Tail-VoI gate   → g             │
      │  risk policy     → (λ, B, c)     │  static or SAC, by flag
      │  DRO-CVaR        → w             │
      └──────────────────────────────────┘
                       │
               portfolio weights
```

| | |
|---|---|
| Input | `SpecialistOutput`, the scenario set, the current `MarketState`, `w_prev` |
| Output | `CIODecision(weights, gate, risk)` |
| Code | `yoda/cio/agent.py` |

## What is fixed and what varies

The allocator is always DRO-CVaR. Two parts are selectable:

| Part | Chosen by | Options |
|---|---|---|
| gate | `TAILVOI__GATE` or `--gate` | `tailvoi` · `accuracy` · `attention` · `equal_weight` |
| risk policy | which script you run | static rule · trained SAC |

Tail-VoI is the proposal; the other three gates are the controls it has to
beat, kept selectable so the claim stays falsifiable. The risk policy is the
seam between the two pipelines:

| Policy | Chosen by | What it does |
|---|---|---|
| `StaticRiskPolicy` | `./scripts/train-tail-voli-risk.sh` | Config-driven rule |
| `RLRiskPolicy` | `./scripts/train-tail-voli-risk-rl.sh` | Trained SAC controller |

That is the whole seam. Everything else about the decision is identical between
the two pipelines, which is what makes the comparison between them a statement
about risk budgeting rather than about two different systems.

## Why it is one object

The three steps are a single decision: gate the views, choose how much risk to
carry, build the book. Keeping them behind one call means callers ask for
weights and the CIO is where the decision demonstrably happens — rather than
three loosely-coupled calls that a future change could quietly reorder.

It still records **what** it decided. `CIODecision` carries the gate weights and
the risk parameters alongside the weights, because the ledger needs them for
the regime analysis and because a decision you cannot inspect is not one you
can trust.

## What it does not do

- It does not emit weights itself — it composes the DRO-CVaR solver, so
  `w ≥ 0`, `Σw = 1`, the per-asset cap and `CVaR_α(w) ≤ B` remain enforced by
  the convex program.
- It does not re-gate. The stack gates once when it builds the `MarketState`;
  re-gating inside the CIO would pay for the same decision twice and risk the
  two copies disagreeing.
- It does not touch `α`. No component redefines the risk measure it is judged
  by.

## See also

- [tail-voi-gate.md](tail-voi-gate.md) — the gate it contains
- [risk-policy.md](risk-policy.md) — the static/SAC seam
- [optimizer.md](optimizer.md) — the DRO-CVaR program it drives
- [experiments.md](experiments.md) — the remaining families
