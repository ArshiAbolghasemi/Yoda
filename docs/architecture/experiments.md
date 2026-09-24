# Baselines and ablations

Every arm runs on the same panel, the same walk-forward splits, the same
transaction costs and the same scoring intervals. A difference between two rows
is a difference in the one thing being varied — that is the only reason the
table means anything.

30 declared arms resolve to fewer distinct runs: several arms are legitimately
the same experiment seen from different families, so execution is deduplicated
and the summary aliases the extra labels onto the one run. Details in
[Redundant arms](#redundant-arms).

---

## The architecture being ablated

```
                         panel (36 assets, one calendar)
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
  technical specialist       volatility specialist          news specialist
   z_tech, μ̂_tech              z_vol, σ̂                    z_news, μ̂_news
        └────────────────────────────┼────────────────────────────┘
                                     ▼
                               Tail-VoI GATE  →  g
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
          fused_μ (directional)               copula conditioning (risk)
                 │                                       ▼
                 │                          Student-t copula → scenarios
                 └───────────────────┬───────────────────┘
                                     ▼
                              MarketState
                                     │
                             POLICY  →  (λ, B, c)
                                     ▼
                              DRO-CVaR solver
                                     ▼
                            portfolio weights w
```

Five decision points, each independently swappable, each with its own family:

| Stage | Question | Varied by |
|---|---|---|
| specialists | What does each channel see? | `openjev` |
| which channels exist | Is this channel worth anything? | `sources` |
| gate | Whom do we trust today? | `gate` |
| policy | How hard do we act on it? | `policy` |
| allocator | Is the robustness worth it? | `optimizer` |
| horizon | Over what period? | `horizon` |

Two invariants hold across every arm. **`α` is never an action** — no agent can
redefine the risk measure it is judged by. And **nothing but the optimizer emits
weights** — `w ≥ 0`, `Σw = 1`, the per-asset cap and `CVaR_α(w) ≤ B` are
enforced by the convex program, not by an agent's good behaviour.

---

## `optimizer` — what the robustness is worth (3 arms)

**The question:** does the distributional robustness earn its place?

There are no external allocation baselines. Every arm runs the same DRO-CVaR
solver; this family varies only `OPT__DRO`, so the difference between rows is
the robustification and nothing else.

| Arm | `OPT__DRO` | Risk term |
|---|---|---|
| `opt_wasserstein` | `wasserstein` | `CVaR_emp(w) + (radius/α)·‖w‖₂` — worst case over a Wasserstein-2 ball. Robustness surfaces as a concentration penalty. |
| `opt_moment` | `moment` | `−μ_sᵀw + radius·√((1−α)/α)·‖Σ^½w‖₂` — worst case over all distributions matching the scenario mean and covariance. |
| `opt_plain_cvar` | `none` | Plain empirical CVaR. The control: the gap to `opt_wasserstein` *is* what robustness bought. |

## `gate` — the headline ablation (5 arms)

**The question:** does gating on *tail value* beat gating on accuracy, on
learned attention, and on nothing?

Everything else is pinned: same specialists, same copula, same static policy,
same DRO-CVaR solver. Only `Gate.gate(spec) → g` changes.

| Arm | Importance signal | Why it is the right control |
|---|---|---|
| `gate_equal` | none — uniform `g` | The floor. Any gate that cannot beat "weight everything equally" does nothing. |
| `gate_accuracy` | Spearman rank IC on the training window, softmaxed | Rewards being **right**, with no notion of tail value. This is the arm that isolates the paper's actual claim: that *accuracy is the wrong objective for a tail-risk portfolio*. |
| `gate_attention` | Learned torch attention over source summaries, trained on return MSE | Learned, but supervised by the **mean**, not by risk. Separates "learned" from "learned on the right objective" — without it, a TailVoI win could just be "a fitted gate beats a fixed one". |
| `gate_tailvoi` | Predicted counterfactual `Δᵢ`, softmaxed | The proposal. |
| `gate_cio` | Cross-sectional dispersion per channel, no model inference | Selecting it hands the CIO the whole decision — gate, stance and weights — so this is the monolithic control, not just a different gate. |

The three controls are chosen to strip a TailVoI win down to its actual cause.
`equal` removes gating; `accuracy` keeps learning but changes the objective;
`attention` keeps learning *and* flexibility but still optimises the mean. If
TailVoI beats all three, the remaining explanation is the tail objective itself.

### How the Tail-VoI target is built

For each sampled training date, the whole downstream is re-solved with each
source removed:

```
Δᵢ = ρ(I₋ᵢ) − ρ(I)
```

A **positive** `Δᵢ` means the book you would have held *without* source `i` had a
worse tail — it earned its place. Each source is ablated at the stage it
actually enters: dropping `technical` or `news` removes a contributor to
`fused_μ`; dropping `volatility` de-conditions the copula and changes the
scenario set. Routing every ablation through `μ` would make the volatility
target meaningless.

That is `1 + K` convex solves per state, so it runs offline on training rows
only, and a small regressor (`G_φ`) amortises it for the backtest.

---

## `policy` — risk budgeting (4 arms)

**The question:** does adapting the risk stance beat a fixed rule?

All four hold the gate at Tail-VoI and vary only
`RiskParamPolicy.act(state) → (λ, B, c)`.

| Arm | Policy | What it tests |
|---|---|---|
| `policy_static` | Config rule, optionally vol-targeted | The baseline. No learning. |
| `policy_rl` | SAC, trained against `AllocationEnv` | Reward `R − λ_CVaR·CVaR − λ_DD·DD − c·turnover`. ⚠️ `λ_CVaR` is a **fixed shaping weight**, a different quantity from the `λ` the policy chooses. |
| `policy_vol_target` | Static rule with `vol_target = 0.02` | The classical volatility-targeting schedule, as an *internal* rule rather than an external baseline. |
| `policy_cio` | OpenJev risk stance | A reasoning model instead of RL, mapped through the *same* `action_to_params` and the same bounds SAC searches — so the comparison is judgement, not reachable range. |
| `cio_full` | CIO sets the gate, the stance **and** the weights | The monolithic control. The only arm without Tail-VoI *or* DRO-CVaR, so its CVaR budget is advisory — see [cio-agent.md](cio-agent.md). |

`policy` and `gate` are orthogonal by construction: `gate` pins the policy at
static and varies the gate; `policy` pins the gate at Tail-VoI and varies the
policy.

---

## Specialists

There is no specialist-backend family any more. **Every channel is OpenJev**:
technical, volatility and news all reach the portfolio as calibrated
probabilities, and the numeric indicator heads have been removed.

One consequence to state plainly: the matrix can no longer answer *"do
probabilistic specialists add value **compared with conventional
predictors**"*, because there is no conventional arm left to compare against.
What it can still answer is whether each channel contributes at all
(`sources`), and whether Tail-VoI identifies when each is useful (`gate`, plus
the regime table).

Specialists are still scored as forecasters before any portfolio exists
(`uv run main.py specialists`) — accuracy, macro-F1, balanced accuracy,
ROC-AUC, **Brier** and **ECE**. A miscalibrated channel still produces
portfolio numbers, and those numbers would be noise wearing a result's clothes.

For reference, the numeric heads measured this on the same panel before they
were removed — technical mean daily rank IC: `mlp` +0.052, `ridge` +0.036,
`gbm` +0.026, `pcr` +0.012; volatility QLIKE: `mlp` −7.543, `gbm` −7.537,
`har` −7.464, against −7.055 for a constant scale. Those are the numbers
OpenJev would have to beat if the comparison were restored.

---

## `sources` — agent removal (10 arms)

**The question:** what is each information source actually contributing?

| Arm | Sources kept |
|---|---|
| `src_all` | technical, volatility, news |
| `src_without_technical` | volatility, news |
| `src_without_volatility` | technical, news |
| `src_without_news` | technical, volatility |
| `src_only_technical` | technical |
| `src_only_volatility` | volatility |
| `src_only_news` | news |
| `src_only_technical_volatility` | technical, volatility |
| `src_only_technical_news` | technical, news |
| `src_only_volatility_news` | volatility, news |

**Removal is genuine deletion, not zero-filling.** The channel's specialist is
never constructed, its cube is never built, and the gate renormalises over what
remains. Zero-filling would leave a dead input the gate still has to weight, and
the measured effect would be "how well does the gate ignore a zero vector"
rather than the information's marginal contribution.

Some of these degrade the stack in structurally interesting ways.
`src_only_volatility` leaves no directional source at all, so `fused_μ` is zero
and the optimizer reduces to pure minimum-CVaR — a legitimate and informative
arm, not a bug.

---

## `horizon` — prediction and rebalance period (4 arms)

`horizon_1d`, `horizon_5d`, `horizon_10d`, `horizon_20d` set both
`PANEL__TARGET_HORIZON` (what the specialists are trained to predict) and
`BACKTEST__REBALANCE_DAYS` (how often the book is rebuilt), because predicting
20 days ahead while rebalancing daily is incoherent.

Longer horizons mean less turnover and less transaction cost but a staler view.
This family is where that trade-off becomes visible.

---

## Redundant arms

Four groups of declared arms are the same configuration:

```
gate_tailvoi == policy_static == jev_none == src_all
src_without_technical  == src_only_volatility_news
src_without_volatility == src_only_technical_news
src_without_news       == src_only_technical_volatility
```

The first group is the **shared control**, appearing once per family so each
family's table reads standalone. The other three are a consequence of having
exactly three sources: "without technical" and "volatility + news only" name the
same set. That redundancy is in the experiment design itself, not the code.

Both are worth *reporting* under both names and not worth *running* twice, so
`run_experiments` deduplicates on a configuration fingerprint and the summary
aliases the extra labels onto the one run — 38 arms, 32 executions, identical
output.

---

## What gets written

```
data/processed/runs/
├── summary.csv                  one row per arm, the headline table
├── gate_weights_by_regime.csv   average Tail-VoI weight per channel per regime
└── <run_id>/
    ├── weights.parquet          date, asset, weight, weight_prev, weight_change
    ├── ledger.parquet           returns, turnover, cost, VaR, CVaR, nu,
    │                            portfolio_value, rp_lam, rp_budget,
    │                            rp_turnover_penalty, gate_g_*, tail_voi_*
    └── run_meta.json            gate, policy, backends, sources, models,
                                 splits, horizon, seed, config_hash
```

`summary.csv` columns: `experiment, tech_backend, vol_backend, news_backend,
return, sharpe, sortino, calmar, max_dd, cvar, turnover`.

Because gate weights and risk parameters are persisted **per rebalance**, the
regime analysis — *does the volatility channel earn more Tail-VoI during
stress?* — is a groupby over the ledger, not another backtest. Scoring any
interval is a re-slice of saved artifacts; the model never runs twice.

---

## Reading the results honestly

- **There are no external baselines.** Every number in this matrix is the same
  architecture with one component changed. That means the tables say which part
  of *this* system carries its weight — they do not, on their own, say the
  system beats classical allocation. If a reviewer asks that question, it needs
  a separate comparison that this repository no longer runs.
- **Run the gate family before believing anything else.** If `gate_tailvoi`
  does not beat `gate_accuracy` and `gate_attention`, the central claim has not
  survived, whatever the portfolio numbers look like.
- **Check the gate actually moves.** The `gate_g_*` columns are persisted per
  rebalance. A gate pinned at a constant has stopped contributing, and a good
  result from it is coming from somewhere else.
- **Read the counterfactual `Δ` means.** They are the honest answer to "is this
  source worth anything in the tail?" *before* any learning. On the default
  panel they currently come out slightly negative — a finding, not a bug.
- **One test window is one test window.** The measured model ordering above
  comes from a single split. Set `SPLIT__REFIT_DAYS` and re-check across folds
  before treating any ordering as settled.
- **Negative results are results.** Nothing in this design assumes OpenJev, the
  CIO, or Tail-VoI helps. Every arm is persisted either way, and the matrix is
  built so a null result is as readable as a positive one.

## See also

- [overview.md](overview.md) — the architecture and its contracts
- [tail-voi-gate.md](tail-voi-gate.md) — the gate and its counterfactual targets
- [risk-policy.md](risk-policy.md) — the policy seam
- [optimizer.md](optimizer.md) — DRO-CVaR and the classical allocators
- [openjev-specialists.md](openjev-specialists.md) — the probabilistic backend
- [inference.md](inference.md) — how a run is executed and scored
