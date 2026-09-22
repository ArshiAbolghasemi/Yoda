# Yoda — TailRiskFlow

Compact, memorable, and associated with disciplined decisions under uncertainty. It
fits a risk-aware multi-agent portfolio system that filters noisy signals, evaluates
tail risk, and communicates only valuable information before making robust
allocation decisions.

Three specialist agents form independent views of the market; a **Tail-VoI gate**
decides how much each view is worth *in the tail*; a Student-t copula turns that
into a scenario set; and a DRO-CVaR solver turns the scenario set into a portfolio.
A **risk policy** — a static rule or a trained RL agent — sets the three knobs the
solver obeys.

```
panel → specialists → Tail-VoI gate → t-copula → DRO-CVaR → portfolio
                                                      ↑
                                            RiskParamPolicy  ← the only seam
```

Two pipelines share every layer except that seam:

| Pipeline | Risk parameters `(λ, B, c)` chosen by | Everything below |
|---|---|---|
| **`tail_voli_risk`** | `StaticRiskPolicy` — fixed or vol-targeted rule | identical |
| **`tail_voli_risk_rl`** | `RLRiskPolicy` — a trained SAC agent | identical |

The RL agent never emits portfolio weights, and `α` (the CVaR tail level) is a fixed
hyperparameter, never an action.

## Documentation

Start with [architecture/overview.md](docs/architecture/overview.md) — how the
pieces compose, the shared contracts, the seam, and how data flows from parquet to
weights. Then read whichever component you are working on.

**Agents**

| Document | What it covers |
|---|---|
| [technical-specialist.md](docs/architecture/technical-specialist.md) | Directional view from the 60 causal indicators; the shared specialist pipeline and why FX NaNs are masked, not imputed |
| [volatility-specialist.md](docs/architecture/volatility-specialist.md) | Magnitude view, and why it conditions the copula instead of entering `μ` |
| [news-specialist.md](docs/architecture/news-specialist.md) | The LangGraph LLM news agent in full — graph, prompts, point-in-time cache, the three backends, cost |
| [tail-voi-gate.md](docs/architecture/tail-voi-gate.md) | **The centerpiece** — counterfactual Δ targets, the learned gate, and the three baseline gates it must beat |
| [risk-policy.md](docs/architecture/risk-policy.md) | **The seam** — the static rule, the SAC controller, the environment and its reward |

**Supporting components**

| Document | What it covers |
|---|---|
| [tail-model.md](docs/architecture/tail-model.md) | Student-t copula: fitting, conditioning, sampling, stress paths |
| [optimizer.md](docs/architecture/optimizer.md) | The DRO-CVaR program, its robustification modes, and the classical baselines |

**Procedures**

| Document | What it covers |
|---|---|
| [training.md](docs/architecture/training.md) | The **complete training procedure**, stage by stage, for both pipelines |
| [inference.md](docs/architecture/inference.md) | The **complete inference procedure**: the walk-forward loop, artifacts, offline scoring, live checklist |

**Data**

| Document | What it covers |
|---|---|
| [data/dataset.md](docs/data/dataset.md) | The `yoda.data` collection pipeline: sources, indicators, news alignment, targets |

## Setup

```bash
./scripts/install-dependencies.sh      # or: uv sync --all-groups
cp .env.example .env                   # then fill in credentials
dvc pull                               # fetch data/financial_dataset.parquet
uv run pytest                          # 50 tests, ~10s, runs on the real parquet
```

Requires Python ≥ 3.14 and [uv](https://docs.astral.sh/uv/). `.env` is git-ignored;
`.env.example` documents every setting with its default.

### Torch build selection

`torch` resolves from PyPI by default (CUDA on Linux, MPS/CPU on macOS). To pin a
specific build:

```bash
uv sync --extra cu130     # or cu128, cu126, cpu
uv sync --extra encoder   # FinBERT + sentence-transformers news backend
```

The CUDA extras are mutually exclusive. The channels cap at different torch
versions — `cu126`/`cu130` carry 2.14, `cu128` caps at 2.11 — so the lockfile pins
each to the newest build it actually publishes. `cu129` is not offered: that channel
stops at torch 2.13, below what stable-baselines3 needs here.

### LLM access for the news channel

The news agent speaks **only** the OpenAI-compatible protocol, so a self-hosted
vLLM/Ollama server, an internal gateway or OpenAI itself is a `.env` change:

```dotenv
NEWS__BACKEND=llm_agent                # or: encoder (fully local), none
NEWS__LLM__BASE_URL=https://<your-openai-compatible-endpoint>/v1
NEWS__LLM__API_KEY=your_key
NEWS__LLM__MODEL=your-chat-model
NEWS__LLM__EMBED_MODEL=your-embedding-model
```

Records are computed once and cached to parquet, so the backtest never calls a
model. Without an endpoint, set `NEWS__BACKEND=none` and everything else runs
unchanged. Details in [docs/architecture/news-specialist.md](docs/architecture/news-specialist.md).

## Running

```bash
./scripts/train-tail-voli-risk.sh      # no-RL stack, static risk policy
./scripts/train-tail-voli-risk-rl.sh   # same stack, RL risk controller
uv run main.py experiments             # every baseline and ablation, one table
uv run main.py data                    # rebuild the dataset from scratch
```

Both training scripts take `--gate tailvoi|accuracy|attention|equal_weight` and
`--run-id NAME`. Each run writes `data/processed/runs/<run_id>/` containing
`weights.parquet`, `ledger.parquet`, `run_meta.json`, four PNGs and a `report.html`.

Procedures are documented in full in
[docs/architecture/training.md](docs/architecture/training.md) and
[docs/architecture/inference.md](docs/architecture/inference.md).

## Configuration

Everything is env-driven through Dynaconf — a run is fully specified by `.env`.
Prefixes: `SPLIT__`, `EVAL__`, `PANEL__`, `SPECIALISTS__`, `NEWS__`, `COPULA__`,
`OPT__`, `TAILVOI__`, `STATIC_POLICY__`, `RL__`, `BACKTEST__`, plus `DATA__` for the
dataset pipeline. See [.env.example](.env.example).

The split boundaries are the ones you will touch first:

```dotenv
SPLIT__TRAIN_START=2017-01-01
SPLIT__TRAIN_END=2022-12-31
SPLIT__VAL_START=2023-01-01
SPLIT__VAL_END=2023-12-31
SPLIT__TEST_START=2024-01-01
SPLIT__TEST_END=2026-09-18
EVAL__INTERVALS=[{name="covid",start=2020-02-01,end=2020-05-31},{name="bear_2022",start=2022-01-01,end=2022-10-31}]
```

They must be contiguous and strictly increasing; this is validated at load.

## Dataset

`data/financial_dataset.parquet`, DVC-tracked, built by `yoda.data` from Yahoo
Finance and Alpaca news.

| | |
|---|---|
| Universe | 36 instruments — 1 crypto (`BTC-USD`), 6 FX, 29 Dow Jones equities |
| Span | 2017-01-01 → 2026-09-18, ~89k asset-day rows |
| Columns | OHLC + adjusted close + volume, 60 causal technical indicators, `headlines` + `news_count`, and `future_return_{1,5,10,20}d` targets |
| Aligned panel | 2018-01-02 → 2026-09-18, 2190 rows × 36 assets after calendar alignment and indicator warm-up |

Three properties shape the whole system and are explained in
[docs/architecture/overview.md](docs/architecture/overview.md): mixed trading calendars, FX volume
indicators that are NaN by design, and uneven news coverage.

## Project status

Defaults are conservative, not tuned. Two worth touching first:

- `STATIC_POLICY__LAM=1.0` against a daily `μ` of a few basis points means the risk
  term dominates and the book concentrates in FX. Lower `λ`, or scale `μ` to the
  rebalance horizon, if you want the directional view to matter.
- `RL__TOTAL_TIMESTEPS=5000` is a tractable default — every environment step is a
  convex solve (~30 ms). Raise it deliberately and expect minutes, not seconds.

Known limits:

- Copula marginals are empirical; GARCH-t marginals are stubbed behind the same
  interface (`COPULA__MARGINALS`) and raise until implemented.
- ENB uses the marginal-risk-contribution decomposition, not Meucci's
  minimum-torsion basis. Same ranking, far cheaper. Marked `ponytail:` in the code.
- Synthetic RL episodes reuse the real state sequence and simulate only the
  outcomes — states cannot be synthesised from the feature panel.
- Direct-weight RL is out of scope by design. It drops in at the policy level and
  touches nothing below it.

## Licence

See [LICENSE](LICENSE).
