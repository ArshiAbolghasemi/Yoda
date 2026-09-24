# Training procedure

Complete, stage by stage, for both pipelines. Stages 1–4 are **identical** and
shared; only stage 5 differs.

```bash
./scripts/train-tail-voli-risk.sh      # stages 1-5, static policy
./scripts/train-tail-voli-risk-rl.sh   # stages 1-5, SAC risk controller
```

Both accept `--gate tailvoi|accuracy|attention|equal_weight` (default:
`TAILVOI__GATE`) and `--run-id NAME`. The risk policy is what the two scripts
differ in.

## Before you start

```bash
./scripts/install-dependencies.sh     # uv sync --all-groups
cp .env.example .env                  # fill in credentials
dvc pull                              # fetch data/financial_dataset.parquet
uv run pytest                         # 50 tests, ~10s — do this first
```

If `financial_dataset.parquet` is missing, `uv run main.py data` rebuilds it from
Yahoo + Alpaca (needs `APCA_API_*`).

Set the walk-forward boundaries in `.env` — a run is fully specified by them:

```dotenv
SPLIT__TRAIN_START=2017-01-01
SPLIT__TRAIN_END=2022-12-31
SPLIT__VAL_START=2023-01-01
SPLIT__VAL_END=2023-12-31
SPLIT__TEST_START=2024-01-01
SPLIT__TEST_END=2026-09-18
SPLIT__REFIT_DAYS=0      # 0 = one fold; >0 = rolling refit, expanding history
```

Boundaries must be contiguous and strictly increasing; `SplitConfig` validates at
load and the resolved ranges are logged. They are clipped against the panel, which
starts 2018-01-02 after indicator warm-up — a `SPLIT__TRAIN_START` of 2017 is
harmless, not silently wrong.

## Where training actually happens

Everything is fitted **per fold, inside the backtest engine**, on that fold's
training rows only. This is deliberate: the walk-forward discipline lives in exactly
one place (`generate_folds` → `run_backtest`), so no stage can accidentally see
future rows.

```
run_tail_voli_risk(config)
└── run_backtest(...)
    └── for each fold:
        ├── build_stack(panel, config, fold.train, TailVoIGate())      # stages 1-2
        │   ├── build_news_features(...)      if NEWS__BACKEND != none
        │   ├── Specialist.fit(...)           per source
        │   └── StudentTCopula.fit(...)
        ├── stack.gate = fit_gate(kind, config)(stack, fold.train)      # stages 3-4
        ├── policy     = make_policy(stack, fold.train, fold.val)       # stage 5 ← THE SEAM
        └── walk fold.test, writing artifacts
```

---

## Stage 1 — Panel alignment

`build_panel(config) → AlignedPanel`. One calendar, returns, feature cubes, the
availability mask, targets. Deterministic, ~0.4 s, no fitting. Details in
[overview.md](overview.md#1-panel-alignment--yodacommonalignmentpy).

```
panel_built rows=2190 assets=36 start=2018-01-02 end=2026-09-18 calendar=equity available=0.993
```

The fold's investable universe is `panel.universe(fold.train)` — assets available on
≥95% of the training rows. With the default split that is 35 of 36 (`DOW` listed in
2019).

## Stage 2 — Specialists and the copula

**Specialists.** Each source's cube is sliced to `fold.train × universe`, flattened
to `(T·N, F)`, and fitted against `panel.targets`. Imputer medians, scaler
statistics, the PCA basis and the ridge coefficients are all estimated here.

```
specialist_fitted name=technical  rows=44059 features=60 z_dim=8
specialist_fitted name=volatility rows=44065 features=8  z_dim=8
```

The news cube is built first if `NEWS__BACKEND != none`; the first run pays the LLM
cost and every later run reads the cache ([news-specialist.md](news-specialist.md)).

**Copula.** Fitted on `panel.returns[fold.train][:, universe]`:

```
copula_fitted rows=1259 assets=35 nu=10.0 mean_corr=0.272
```

A low `ν` means fat joint tails. If you see `ν = 30` (the grid ceiling) the panel is
behaving Gaussian-ly and the copula is adding little — worth a look.

## Stage 3 — Counterfactual targets

`counterfactual_deltas(stack, fold.train, tailvoi_config, static_policy_config)`.

`TAILVOI__N_STATES` (250) evenly spaced training dates; for each, `1 + K` full
re-solves at `TAILVOI__COUNTERFACTUAL_SCENARIOS` (500). See
[tail-voi-gate.md](tail-voi-gate.md#1-the-counterfactual-generator--the-supervisor)
for the
mechanics.

```
counterfactual_targets states=250 rho=realized mean={'delta_technical': …, 'delta_volatility': …}
```

**Read those means.** They are the honest answer to "is this source worth anything
in the tail?" before any learning happens. On the default panel they come out
slightly *negative* — the sources are not adding tail value at these settings. That
is a finding, not a bug; report it.

Cost scales as `n_states × (1 + K)` solves. 250 states × 3 sources ≈ 1000 solves
≈ 35 s. Lower `TAILVOI__N_STATES` while iterating.

## Stage 4 — Train the gate

`fit_gate(config, kind)(stack, fold.train)` dispatches on the selected gate:

| gate | What is fitted | Needs the counterfactual targets? |
|---|---|---|
| `tailvoi` | `StandardScaler → MLPRegressor` on `(summary, Δ)` | yes |
| `accuracy` | Spearman IC per source over the training window | no |
| `attention` | torch attention, 300 Adam epochs | no |
| `equal_weight` | nothing | no |

```
tailvoi_gate_fitted states=250 sources=technical,volatility,news train_r=0.99
```

`train_r` is in-sample fit of `G_φ` to `Δ` — high is expected and is *not* evidence
the gate works. The evidence is the gate ablation in
[inference.md](inference.md#the-experiment-harness).

## Stage 5 — The policy (the only divergence)

### `tail_voli_risk` — static

```python
def make_policy(stack, train, val):
    return StaticRiskPolicy(config.research.static_policy, alpha=alpha)
```

No training. Tune `STATIC_POLICY__LAM`, `__BUDGET`, `__TURNOVER_PENALTY`,
`__VOL_TARGET` in `.env`.

### `tail_voli_risk_rl` — SAC

```python
def make_policy(stack, train, val):
    rows = np.concatenate([train, val])          # val is training data for the agent
    return train_sac(stack, rows, research.rl, rebalance_days=…, checkpoint=…)
```

Specialists, copula and gate are **frozen** — the agent trains against a fixed
world. `AllocationEnv` walks `train + val` at the rebalance stride; SAC learns a
3-dim continuous action. See
[risk-policy.md](risk-policy.md#rlriskpolicy--the-rl-pipeline) for the observation,
action bounds and reward.

```
env_ready rows=1509 obs_dim=85 synthetic=False
sac_train_start steps=5000 rows=1509
sac_synthetic_pass steps=256          # only if RL__SYNTHETIC_EPISODES > 0
sac_saved path=data/processed/rl/sac_risk_policy_<run_id>
```

**Budget.** Every environment step is a convex solve (~30 ms at
`RL__ENV_SCENARIOS` = 500), so wall-clock is roughly
`total_timesteps × 30 ms`: 5000 steps ≈ 2.5 min, 50 000 ≈ 25 min. The default is
tractable, not converged — raise it deliberately.

**Knobs that matter, in order:**

| Setting | Default | Why you would change it |
|---|---|---|
| `RL__TOTAL_TIMESTEPS` | 5000 | The single biggest quality/time lever |
| `RL__ENV_SCENARIOS` | 500 | Smaller = faster steps, noisier tail estimates |
| `RL__SYNTHETIC_EPISODES` | 0 | >0 adds copula-generated tail paths; the real sample has ~one crash |
| `RL__LAM_BOUNDS` / `__BUDGET_BOUNDS` | (0.1,10) / (0.01,0.20) | The action ranges — set them where the optimizer is actually responsive |
| `RL__REWARD_LAMBDA_CVAR` | 1.0 | Fixed shaping weight. **Not** the `λ` the agent chooses |

Checkpoints are written to `RL__CHECKPOINT` suffixed with the run id and reload via
`RLRiskPolicy.load(path, config, alpha, sources)`.

## Reproducibility

Seeds are per-component and config-driven: `SPECIALISTS__SEED` (7),
`COPULA__SEED` (11), `TAILVOI__SEED` (13), `RL__SEED` (17), `BACKTEST__SEED` (23).
Every run writes `run_meta.json` with the pipeline name, gate, news backend, split
boundaries, seed and a `config_hash` of the whole research config — two runs with
the same hash saw the same settings.

The LLM news channel is made reproducible by the cache, not by the model: the cache
key includes the model name and `NEWS__LLM__PROMPT_VERSION`.

## Training several arms at once

```bash
uv run main.py experiments --families gate system policy news
```

Runs every baseline and ablation on the same panel, same splits, same intervals,
and scores them side by side. One failing arm is logged and skipped rather than
sinking the sweep. Arms and families are listed in
[inference.md](inference.md#the-experiment-harness).

## A fast iteration loop

Full defaults are minutes per run. While developing, shrink three things:

```python
import dataclasses
from yoda.config import load_config
from yoda.common.alignment import build_panel
from yoda.pipeline import run_tail_voli_risk

base = load_config()
r = base.research
config = dataclasses.replace(
    base,
    research=dataclasses.replace(
        r,
        news=dataclasses.replace(r.news, backend="none"),  # no LLM
        tailvoi=dataclasses.replace(
            r.tailvoi, sources=("technical", "volatility"), n_states=40
        ),  # cheap stage 3
        split=dataclasses.replace(r.split, test_start="2025-01-01"),  # short test
    ),
)
panel = build_panel(config)  # build once, reuse across runs
run_tail_voli_risk(config, run_id="dev", panel=panel, make_plots=False)
```

That completes in ~20 s versus minutes. Every pipeline entry point accepts
`panel=`, `features=`, `optimizer=` and `make_plots=` for exactly this.
