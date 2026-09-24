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
| [news-specialist.md](docs/architecture/news-specialist.md) | The news channel — OpenJev probabilities in, directional view out |
| [tail-voi-gate.md](docs/architecture/tail-voi-gate.md) | **The centerpiece** — counterfactual Δ targets, the learned gate, and the three baseline gates it must beat |
| [risk-policy.md](docs/architecture/risk-policy.md) | **The seam** — the static rule, the SAC controller, the environment and its reward |
| [openjev-specialists.md](docs/architecture/openjev-specialists.md) | The optional OpenJev 27B backend — serving, decision tasks, caching, calibration metrics |
| [cio-agent.md](docs/architecture/cio-agent.md) | The CIO — Tail-VoI gate + risk policy + DRO-CVaR behind one call, returning portfolio weights |

**Supporting components**

| Document | What it covers |
|---|---|
| [tail-model.md](docs/architecture/tail-model.md) | Student-t copula: fitting, conditioning, sampling, stress paths |
| [optimizer.md](docs/architecture/optimizer.md) | The DRO-CVaR program and its robustification modes |

**Procedures**

| Document | What it covers |
|---|---|
| [training.md](docs/architecture/training.md) | The **complete training procedure**, stage by stage, for both pipelines |
| [inference.md](docs/architecture/inference.md) | The **complete inference procedure**: the walk-forward loop, artifacts, offline scoring, live checklist |
| [experiments.md](docs/architecture/experiments.md) | **Every baseline and ablation explained** — what each arm isolates and how to read the table |

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
uv sync --extra cu130     # or cu132, cu129, cu128, cu126, cpu
```

The CUDA extras are mutually exclusive. The channels cap at different torch
versions, so the lockfile pins each to the newest build it publishes:

| Extra | torch | vLLM | Use for |
|---|---|---|---|
| `cu130` | 2.13.0+cu130 | 0.29.0 | serving OpenJev **and** research |
| `cu132` | 2.13.0+cu132 | — | research on CUDA 13.2 |
| `cu129` · `cu126` | 2.13.0 | — | research on a CUDA 12 driver |
| `cu128` | 2.11.0 | — | research on a CUDA 12 driver, older torch |
| `cpu` | 2.13.0+cpu | — | no GPU |

**Only `cu130` carries vLLM.** Every PyPI vLLM wheel that supports Python 3.14 is
a CUDA 13 build, so it links `libcudart.so.13`; the cu12x channels ship
`libcudart.so.12` and would install a vLLM that resolves fine and then fails on
import. vLLM also pulls `torchaudio`/`torchvision`/`torchcodec`, which refuse to
load unless their CUDA matches torch's exactly — and `cu130` is the only channel
publishing all four. So `cu130` carries the whole serving set, pinned to that one
index, and every broken mix is unrepresentable rather than a runtime surprise.

`cu132` is research-only for that reason: PyTorch publishes no `+cu132`
torchaudio, so a vLLM there dies with *"PyTorch has CUDA version 13.2 whereas
TorchAudio has CUDA version 13.0"*.

Serving therefore also needs an **NVIDIA driver r580+** on the host — or
`cuda-compat-13-0` on a datacenter GPU with an older one. `llm-serve/serve.sh`
checks this before it downloads anything. See
[docs/architecture/openjev-specialists.md](docs/architecture/openjev-specialists.md).

### OpenJev (optional probabilistic specialists)

```bash
cd llm-serve && cp .env.example .env && docker compose up -d
```

Serves `openjev/openjev` through vLLM with the decision shim in front
(`:3000` → `:8000`), then point the stack at it with
`JEV__BASE_URL=http://127.0.0.1:3000`. Needs an NVIDIA GPU (~30 GB VRAM at
FP8). `cd llm-serve && ./serve.sh` does the same without containers. Details in
[docs/architecture/openjev-specialists.md](docs/architecture/openjev-specialists.md).

## Running

```bash
./scripts/train-tail-voli-risk.sh      # no-RL stack, static risk policy
./scripts/train-tail-voli-risk-rl.sh   # same stack, RL risk controller
uv run main.py specialists             # score specialist prediction quality first
./scripts/run-experiments.sh           # every ablation, one table
./scripts/run-experiments.sh --policies static rl   # ...under both controllers
uv run main.py data                    # rebuild the dataset from scratch
```

CIO agent — gate, risk stance and weights. Each run writes `data/processed/runs/<run_id>/` containing
`weights.parquet`, `ledger.parquet`, `run_meta.json`, four PNGs and a `report.html`.

Procedures are documented in full in
[docs/architecture/training.md](docs/architecture/training.md) and
[docs/architecture/inference.md](docs/architecture/inference.md).

## Configuration

Everything is env-driven through Dynaconf — a run is fully specified by `.env`.
Prefixes: `SPLIT__`, `EVAL__`, `PANEL__`, `SPECIALISTS__`, `NEWS__`, `COPULA__`,
`OPT__`, `TAILVOI__`, `STATIC_POLICY__`, `RL__`, `BACKTEST__`, `JEV__`, plus
`DATA__` for the dataset pipeline. See [.env.example](.env.example).

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

## Licence

See [LICENSE](LICENSE).
