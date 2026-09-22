# Allocator — DRO-CVaR

Not an agent: the shared allocation layer both pipelines drive, and the reason the
policy seam is only three numbers wide.

| | |
|---|---|
| Input | `μ` `(N,)`, `TailScenarios`, `RiskParams`, `w_prev` `(N,)` |
| Output | long-only weights on the simplex `(N,)` |
| Code | `yoda/optimizer/dro_cvar.py`, `yoda/optimizer/classical.py` |
| Contract | `DROCVaROptimizer` — `solve(mu, scen, rp, w_prev) → w` |

## The program

```
max_w   μᵀw  −  λ · DRO-CVaR_α(w)  −  c‖w − w_prev‖₁
s.t.    w ≥ 0,  Σw = 1,  w ≤ max(OPT__MAX_WEIGHT, 1.5/N),  [CVaR_α(w) ≤ B]
```

CVaR is the Rockafellar–Uryasev convex form over the scenario set:

```
CVaR_α(w) = min_ζ  ζ + 1/(α·S) · Σ_s [(−r_sᵀw) − ζ]₊
```

so the whole thing is one convex program — cvxpy, CLARABEL, ~150 ms at 2000
scenarios and 35 assets.

The solver is **deliberately blind to where `rp` came from**: a static rule and an
RL action reach it through the identical call. That is what makes the A/C
comparison a statement about risk budgeting rather than about two different
systems.

## Robustification

`OPT__DRO`:

| Mode | Risk term |
|---|---|
| `wasserstein` (default) | `CVaR_emp(w) + (radius/α)·‖w‖₂` — the closed form of worst-case CVaR over a Wasserstein-2 ball around the empirical scenario distribution. Robustness surfaces as a **concentration penalty**, which is the behaviour you want out of it. |
| `moment` | `−μ_sᵀw + radius·√((1−α)/α)·‖Σ^½w‖₂` — worst case over all distributions matching the scenario mean and covariance |
| `none` | Plain empirical CVaR — the `static-CVaR` baseline arm |

`OPT__RADIUS` (0.001) is the ball radius / ellipsoid scale.

## The budget

`OPT__BUDGET_MODE` decides how `B` is enforced:

- `constraint` (default) — a hard `CVaR_α(w) ≤ B`
- `penalty` — a convex hinge, `OPT__BUDGET_PENALTY · max(0, CVaR − B)`

A hard budget can be infeasible. The solver then **relaxes to the penalty form
once** and, failing that, holds the previous book. It logs and never raises
mid-backtest:

```
optimizer_fallback status=infeasible failures=1
```

> Repeated fallbacks mean the budget is infeasible against the current scenario
> set, and the book is being *held* rather than chosen. Watch for this line.

## Classical baselines

`EqualWeight`, `MeanVariance` and `RiskParity` implement the identical `solve`
signature, so every baseline in the paper runs through one code path and the
backtest harness cannot tell them apart:

```python
Arm(
    "sys_risk_parity",
    "Risk-Parity",
    "system",
    gate="equal_weight",
    optimizer=RiskParity,
)
```

| Baseline | Form |
|---|---|
| `EqualWeight` | 1/N over the investable universe |
| `MeanVariance` | long-only Markowitz on the scenario covariance; `rp.lam` is risk aversion |
| `RiskParity` | equal risk contribution via the convex log-barrier form |

## Settings

| Setting | Default | Effect |
|---|---|---|
| `OPT__ALPHA` | 0.05 | The CVaR tail level. **Never an action.** |
| `OPT__DRO` | `wasserstein` | Robustification mode |
| `OPT__RADIUS` | 0.001 | Ball radius / ellipsoid scale |
| `OPT__BUDGET_MODE` | `constraint` | Hard bound vs hinge penalty |
| `OPT__BUDGET_PENALTY` | 50.0 | Hinge weight in penalty mode |
| `OPT__MAX_WEIGHT` | 0.25 | Per-asset cap, floored at `1.5/N` so the problem stays feasible |
| `OPT__SOLVER` | `CLARABEL` | Any cvxpy conic solver |

## Performance

The solve is the bottleneck of the whole system — it runs once per rebalance in a
backtest, `1 + K` times per state in the counterfactual generator, and once per
step during RL training. Scenario count is the lever: `COPULA__N_SCENARIOS` (2000)
for backtests, `TAILVOI__COUNTERFACTUAL_SCENARIOS` (500) for target generation,
`RL__ENV_SCENARIOS` (500) inside the training loop.
