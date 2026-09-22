# Tail model — Student-t copula

Not an agent: the joint distribution the agents' views are expressed against, and
the thing the optimizer actually optimises over.

| | |
|---|---|
| Input | `(T, N)` aligned returns for the fold's universe |
| Output | `TailScenarios(scenarios (S, N), R, nu)` |
| Code | `yoda/copula/student_t.py` |
| Contract | `TailModel` — `fit` / `conditional` / `sample` / `tail_stats` |

## Why a copula at all

A Gaussian covariance matrix says how assets co-move *on average*. What matters for
CVaR is how they co-move **in the tail**, and that is exactly where the Gaussian
assumption fails: correlations go to one in a crash, and a Gaussian model does not
know that.

Separating marginals from dependence lets each keep its own shape — empirical fat,
skewed marginals per asset, and a Student-t copula carrying the tail dependence a
Gaussian copula throws away.

## Fitting

1. **Devolatilise.** EWMA mean and scale (`COPULA__VOL_HALFLIFE` = 60) → standardised
   residuals.
2. **Dependence.** Kendall's τ → `sin(πτ/2)` — robust to those fat marginals, where
   Pearson is not. Computed on the full window *and* on the last
   `COPULA__CORR_WINDOW` (250) rows, blended (`COPULA__CORR_BLEND` = 0.5), then
   ridge-shrunk toward the identity (`COPULA__SHRINKAGE` = 0.1), projected onto the
   PSD cone and rescaled to unit diagonal. Kendall-implied matrices are not PSD in
   general, so that projection is load-bearing, not hygiene.
3. **ν** by grid MLE of the t-copula log-likelihood over `COPULA__NU_GRID`.

```
copula_fitted rows=1259 assets=35 nu=10.0 mean_corr=0.272
```

A low `ν` means fat joint tails. If you see `ν = 30` (the grid ceiling) the panel is
behaving Gaussian-ly and the copula is adding little — worth a look.

**Dimensionality.** 36 assets against ~1250 training rows is comfortably longer than
it is wide, so no factor reduction or asset grouping is applied; the shrinkage
handles conditioning. Past a few hundred names this is the first thing that has to
change.

## Conditioning

```python
copula.conditional(state).sample(n)
```

`MarketState.regime` carries the per-asset volatility view. `conditional` swaps it
in for the unconditional EWMA scale; a state with `regime=None` deliberately
returns the **unconditional** model, which is how the counterfactual generator
ablates the volatility channel at the stage it actually enters.

The blend between the two is set by the gate — see
[volatility-specialist.md](volatility-specialist.md#why-it-never-touches-the-mean).

## Sampling

```
MVN(0, R)  →  × √(ν/χ²_ν)  →  t-CDF  →  inverse empirical marginal per asset  →  × scale
```

Fat tails come from the data; tail *dependence* comes from the copula. Default
`COPULA__N_SCENARIOS` = 2000 per rebalance (~30 ms).

## Risk measurement

`tail_stats(w, scen, alpha)` returns the dict that travels in `MarketState` and
into the RL observation:

| Key | Meaning |
|---|---|
| `var` | VaR at `α`, in **loss** units (positive is a loss) |
| `cvar` | mean loss beyond VaR |
| `es` | the same tail as a **return** (negative) — what reports and rewards quote |
| `nu`, `vol`, `mean` | shape of the scenario set |

## Stress paths

```python
stress_scenarios(model, n, severity=2.0)
```

Re-samples with an inflated scale. The real sample contains essentially one major
crash, so a single historical stress window is a sample size of one — synthetic
tail paths are how you get more than that.

## Extending

`COPULA__MARGINALS` already has a `garch_t` slot behind the same interface; it
raises `NotImplementedError` until written. Any other `TailModel` implementation
drops in without touching the gate, the optimizer or the policies.
