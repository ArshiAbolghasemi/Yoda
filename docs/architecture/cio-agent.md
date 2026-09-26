# CIO

Every specialist reports here. The CIO holds the world model it needs to read
those reports and the machinery to act on them, and exposes one thing — weights.

```
    technical ─┐
    volatility ┼─► CIO ─────────────────────────────┐
    news ──────┘   │  Tail-VoI gate     → g         │
                   │  Student-t copula  → scenarios │
                   │  risk policy       → (λ, B, c) │  static or SAC, by flag
                   │  DRO-CVaR          → w         │
                   └────────────────────────────────┘
                                  │
                          portfolio weights
```

| | |
|---|---|
| Input | a panel row (`position`) and `w_prev` |
| Output | `CIODecision(weights, gate, risk, state)` |
| Code | `yoda/cio/agent.py` |

## Construction is phased

The gate is fitted from counterfactuals that run *through* the CIO, so the
object has to exist before its gate does:

```python
cio = build_cio(panel, config, fold.train, TailVoIGate(...))  # specialists + copula
cio.gate = fit_gate(config, kind)(cio, fold.train)  # counterfactuals
cio.install_policy(make_policy(cio, fold.train, fold.val))  # static rule or SAC
```

That is not circular — the counterfactual generator *forces* gate weights
rather than asking the gate — but it does mean a CIO is usable, deliberately,
before its gate is fitted. Deciding before the policy is seated raises.

## Why `state` is public

SAC trains *against* the risk policy, so `AllocationEnv` needs the observation
a policy would act on, without one having acted. `state(position, w_prev)`
returns `(MarketState, TailScenarios, GateOutput)`; `decide` is that plus the
two judgements. The counterfactual generator uses the same surface.

## What is fixed and what varies

The allocator is always DRO-CVaR. Two parts are selectable:

| Part | Chosen by | Options |
|---|---|---|
| gate | `TAILVOI__GATE` or `--gate` | `tailvoi` · `accuracy` · `attention` · `equal_weight` |
| risk policy | which script you run | static rule · trained SAC |

The specialists, the copula and the allocator do not vary.

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
- It does not decide twice. `decide` is the only decision site: the backtest
  engine calls it and records what comes back, rather than re-running
  gate → policy → solve itself.
- It does not touch `α`. No component redefines the risk measure it is judged
  by.

## See also

- [tail-voi-gate.md](tail-voi-gate.md) — the gate it contains
- [risk-policy.md](risk-policy.md) — the static/SAC seam
- [optimizer.md](optimizer.md) — the DRO-CVaR program it drives
- [experiments.md](experiments.md) — the remaining families
