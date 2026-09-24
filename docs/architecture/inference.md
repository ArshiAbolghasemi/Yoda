# Inference procedure

Inference happens in two decoupled passes. The backtest **executes and persists**;
evaluation **scores, offline, from what was persisted**. That separation is what
lets you re-score any interval — COVID, a 2022 bear window, a rolling 60-day
Sharpe — without ever re-running the model.

```
walk-forward execution  ──►  weights.parquet + ledger.parquet + run_meta.json
                                          │
                                          ▼
                              offline scoring by interval  ──►  tables + figures + report.html
```

## 1. The per-rebalance decision

One inference step, in full:

```python
state, scen, gate_out = stack.state(position, w_prev)  # ① observe
params = policy.act(state)  # ② the seam → (λ, B, c)
weights = stack.allocate(state, scen, params)  # ③ solve
```

**① Observe** (`AllocationStack.state`):

1. `specialist_output(position)` — each specialist encodes and predicts on that
   date's features, restricted to the fold's universe.
2. `gate.gate(spec)` — gate weights `g`, `fused_mu`, `fused_z`.
3. `scenarios(spec, gate, n)` — the volatility view is blended against the copula's
   unconditional scale by `conditioning_strength(gate)`, and `COPULA__N_SCENARIOS`
   (2000) joint scenarios are drawn.
4. `tail_stats(w_prev, scen, α)` — `cvar, var, es, nu, vol` of the *current* book
   under those scenarios.
5. Assemble `MarketState`.

**② Decide.** `StaticRiskPolicy` applies its rule; `RLRiskPolicy` flattens the state
to the observation vector and runs SAC with `deterministic=True`. Identical call
signature — the layers below cannot tell which ran.

**③ Solve.** One convex program → long-only weights on the simplex. If a hard CVaR
budget is infeasible the solver relaxes to the penalty form once, and failing that
holds the previous book. It logs; it never raises mid-backtest.

Cost: ~150 ms per rebalance at 2000 scenarios, dominated by the solve.

## 2. The walk-forward loop

`yoda/backtest/engine.py`. For each fold, over that fold's test rows:

```python
for step, position in enumerate(fold.test):
    rebalance = step % BACKTEST__REBALANCE_DAYS == 0
    previous  = weights
    if rebalance:
        state, scen, gate = stack.state(position, previous)
        weights = stack.allocate(state, scen, policy.act(state))
    turnover = |weights − previous|.sum()
    cost     = turnover × BACKTEST__COST_BPS / 1e4
    realized = panel.returns[position + 1][universe]      # ← earned the NEXT day
    port_return = weights @ realized − cost
    weights  = drift(weights, realized)                   # market drift, not a free rebalance
```

Three things this gets right and most backtests get wrong:

- **The book earns `returns[t+1]`**, never `returns[t]`. Features at `t` decide the
  book held over `t → t+1`.
- **Between rebalances the book drifts** with the market. Holding constant weights
  for free is an implicit daily rebalance nobody paid for.
- **Turnover is charged** at `BACKTEST__COST_BPS` (5 bp one-way) on the L1 change,
  including the drift-induced change at the next rebalance.

### Folds

`SPLIT__REFIT_DAYS = 0` → a single fold: fit on train, trade the test window.
`> 0` → rolling refit with **expanding** history: fold *k* trains on
`train + val + everything already traded` and trades the next `refit_days` rows.
Every fold re-fits specialists, copula, gate and policy from scratch. A test
asserts `fold.train.max() < fold.test.min()` for every fold.

## 3. Artifacts

Written to `data/processed/runs/<run_id>/`, schema-validated before write:

| File | Contents |
|---|---|
| `weights.parquet` | long `(date, asset, weight)` — the realised book each day, drift included. This is what you re-slice. |
| `ledger.parquet` | per step: `date, fold, port_return, turnover, cost, realized_cvar, realized_var, cash, gross, gate_g_*` |
| `run_meta.json` | `pipeline`, `label`, `gate`, `news_backend`, `policy`, `alpha`, `splits`, `rebalance_days`, `cost_bps`, `seed`, `config_hash`, `assets` |

`gate_g_*` columns are the live gate weights at each rebalance — plot them to see
whether the gate is actually moving or has collapsed to a constant.

Reload with `load_run(root, run_id)`; enumerate with `list_runs(root)`.

> **Schema note.** `run_meta.json` uses the key `pipeline` (`"tail_voli_risk"` /
> `"tail_voli_risk_rl"`). Runs written before that rename carry `version` instead.

## 4. Offline scoring

```python
from yoda.evaluation import evaluate

evaluation = evaluate(config, ["tail_voli_risk", "tail_voli_risk_rl"])
evaluation.tables["full"]  # DataFrame: run × metric
evaluation.tables["covid"]  # the same runs, COVID only
evaluation.report  # …/report.html
```

`evaluate` never re-runs a backtest. It loads artifacts, resolves intervals, and
computes one metric row per run per interval.

### Intervals

Resolved automatically and clipped to what the run actually covers:

| Kind | Source |
|---|---|
| `full` | the whole ledger |
| `train` / `val` / `test` | `SPLIT__*` |
| named regimes | `EVAL__INTERVALS` |

```dotenv
EVAL__INTERVALS=[{name="covid",start=2020-02-01,end=2020-05-31},{name="bear_2022",start=2022-01-01,end=2022-10-31}]
EVAL__ROLLING_WINDOWS=[20,60]
```

Always include the COVID-2020 window and the synthetic copula tail paths: the real
sample contains essentially one major crash, so a single stress period is a sample
size of one.

### Metrics

The nine HedgeAgents-style metrics plus tail and cost columns, per interval:

| | |
|---|---|
| `TR` | total return, `Π(1+r) − 1` |
| `ARR` | annualised, `(1+TR)^(252/n) − 1` |
| `Sharpe` | `ARR / annualised vol` |
| `Calmar` | `ARR / MaxDD` |
| `Sortino` | `ARR / annualised downside vol` |
| `MaxDD` | worst peak-to-trough |
| `Volatility` | annualised |
| `ENT` | Shannon entropy of the average book — `ln N` when even |
| `ENB` | effective number of bets |
| `CVaR` / `ES` | realised mean loss / mean return in the worst `α` tail |
| `Turnover` | mean per step |

`ENB` uses the marginal-risk-contribution decomposition, not Meucci's
minimum-torsion basis — same ranking, far cheaper, and flagged as a known
simplification in the module docstring.
It needs an asset covariance, so `evaluate` builds the panel unless you pass
`panel=`; without one, `ENB` is `NaN` and everything else still scores.

### Figures

Written next to the artifacts (single run) or to `runs/_compare/` (several), plus a
self-contained `report.html` carrying the tables and the figures:

- cumulative return, one line per run
- drawdown
- rolling Sharpe, one panel per `EVAL__ROLLING_WINDOWS` entry
- weight allocation over time, stacked area, per run

Series colours come from a validated categorical palette in fixed slot order. The
weight plot shows the largest average holdings and folds the rest into one muted
"Other" band — 36 assets, 8 categorical slots, and inventing hues is how a chart
becomes unreadable.

Pass `make_plots=False` to skip figures entirely when sweeping.

## The experiment harness

```bash
uv run main.py experiments --families gate system policy news
```

Every arm on the same panel, the same splits and the same intervals:

| Family | Arms | The question |
|---|---|---|
| `gate` | `tailvoi`, `accuracy`, `attention`, `equal_weight` | **The headline.** Does gating on tail value beat gating on accuracy, on learned attention, and on nothing? |
| `optimizer` | Wasserstein / moment / plain CVaR | Is the distributional robustness worth it? |
| `policy` | static vs RL, same gate | Does regime-adaptive risk budgeting add value over a static rule? |
| `news` | `none` vs OpenJev | Does the news channel earn its place at all? |

The `news` family asks whether `z_news` is worth anything at all. Note the
contamination caveat: OpenJev is pretrained, so it has read the future, and the
prompts' date instruction cannot remove that. Treat a large news win as a
leakage hypothesis first and a result second.

Each news backend's feature cube is built once and shared across every arm that
uses it. Arms that cannot run — no LLM endpoint configured, the `encoder` extra not
installed — are logged and skipped.

## Running inference on new data

There is no separate "predict" entry point, by design: the backtest loop *is* the
inference loop. To score a later period with an already-trained configuration,
extend `SPLIT__TEST_END` and re-run — with `SPLIT__REFIT_DAYS = 0` the fit window is
unchanged, so nothing is re-learned.

For a live book, the same three calls are all you need:

```python
panel = build_panel(config)  # refresh the dataset first
stack = build_stack(panel, config, train_rows, gate)
policy = StaticRiskPolicy(...)  # or RLRiskPolicy.load(checkpoint, ...)

today = len(panel.dates) - 1
state, scen, gate_out = stack.state(today, w_prev)
weights = stack.allocate(state, scen, policy.act(state))
book = stack.expand(weights)  # universe space → all 36 assets
```

### Live checklist

- [ ] **Persist `w_prev` between sessions.** The turnover penalty and the reported
      turnover are meaningless without the real previous book.
- [ ] **Re-check the universe.** `panel.universe(rows)` changes as assets list and
      delist; `stack.expand` maps back to full asset space with zeros elsewhere.
- [ ] **Refit on a schedule.** Nothing re-fits on its own. Use `SPLIT__REFIT_DAYS` in
      backtests to measure how fast performance decays without a refit, then match
      that cadence live.
- [ ] **Warm the news cache before the open**, not during. The backtest path never
      calls a model, but a live `build_news_features` on a cold cache will.
- [ ] **Watch `optimizer_fallback` in the logs.** Repeated fallbacks mean
      `STATIC_POLICY__BUDGET` (or the RL budget bounds) is infeasible against the
      current scenario set, and the book is being held rather than chosen.
- [ ] **Watch the `gate_g_*` columns.** A gate pinned at a constant is a gate that
      has stopped contributing — check it before trusting a Tail-VoI result.
