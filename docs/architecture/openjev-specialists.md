# OpenJev specialists

**The** backend for all three specialist channels. Technical, volatility and
news each reach the portfolio as OpenJev 27B probabilities — a decision model
that answers typed questions with calibrated distributions rather than
generating text. There is no numeric alternative; the indicator cubes survive
only as the *state* each decision task is shown.

Everything below the specialists is untouched: same Tail-VoI gate, same
Student-t copula, same SAC controller, same DRO-CVaR optimizer, same
walk-forward harness.

```
                         Dataset
                            │
       ┌────────────────────┼────────────────────┐
       ▼                    ▼                    ▼
 OpenJev Technical    OpenJev Volatility    OpenJev News
       │                    │                    │
     z_tech               z_vol               z_news
       └────────────────────┼────────────────────┘
                            ▼
                     Tail-VoI Gate → copula → SAC → DRO-CVaR → weights
```

## Why a decision model rather than an LLM

The question the gate needs answered is *"how likely is each outcome, and how
much should I trust that?"* A text-generating model gives you a word; you then
guess at a probability. OpenJev reads the scores at the first output position —
one per option — and converts them with fixed calibration constants, so
`P(down)/P(flat)/P(up)` is the model's actual answer, not a parse of prose.

That is what makes Brier score and expected calibration error meaningful for
these channels, and it is why the whole distribution is stored rather than the
winning class.

## Serving

```
yoda specialists ──► OpenJev decision shim :3000 ──► vLLM :8000 ──► openjev/openjev
```

The shim speaks the TypeSafe System One protocol, so the client in
`yoda/specialists/jev/client.py` is unchanged whether it points at a local shim
or a hosted endpoint. Only `JEV__BASE_URL` moves.

### Docker Compose (recommended)

```bash
cd llm-serve
cp .env.example .env          # HF_TOKEN only if the repo is gated for you
docker compose up -d          # first run downloads ~55 GB
curl localhost:3000/v1/version
```

Three services: a one-shot `model-downloader` that pulls the weights *and*
`helper/shim.py` into a shared volume, `vllm` serving the model, and
`openjev-shim` in front of it. `vllm` has a health check and the shim waits on
it, so `up -d` is safe to run cold.

Each service declares `env_file: ../.env`, so a bare `docker compose up -d`
already reaches the containers with `HF_TOKEN`, `SHIM_TOKEN` and the readout
constants — no flag to remember. The compose-level knobs (ports, `GPU_COUNT`,
`GPU_MEMORY_UTILIZATION`) are *interpolated* rather than injected, so they fall
back to the defaults in the file; add `--env-file ../.env` as well if you want
`.env` to override those too.

Needs an NVIDIA GPU with the container toolkit. OpenJev 27B at FP8 wants roughly
30 GB of VRAM at this context length — lower `MAX_MODEL_LEN` or
`GPU_MEMORY_UTILIZATION` if you are tighter.

### Without Docker

```bash
cd llm-serve
./serve.sh                 # install, download, start both, follow logs
./serve.sh start -d        # same, but detach
./serve.sh status
./serve.sh stop
```

Reads the project's **root `.env`** — the same file the compose stack uses, so
the whole project is configured in one place. There is no `llm-serve/.env`.

vLLM is installed into `llm-serve/.venv` rather than the research environment:
it pins torch hard, and a later `uv sync` would otherwise reshuffle torch
underneath the RL code.

Both paths use the serving configuration published on the model card:

| Setting | Value |
|---|---|
| quantization | `fp8` |
| prefix caching | enabled |
| max model length | 16384 |
| max concurrent sequences | 256 |
| max logprobs | 64 |
| prefill backend | `--gdn-prefill-backend triton` |
| pinned versions | `vllm==0.29.0`, `openai==3.16.2`, `httpx==0.28.1` |

### Readout calibration

```
READOUT_T=0.85  READOUT_NOUL_T=1.829074  READOUT_NOUL_BIAS=0
READOUT_TARGETED=1  READOUT_INSTR_STYLE=pyrepr  SHIM_STAGGER=1
```

These constants were **fitted for this readout**. Changing them changes what the
probabilities mean, which is precisely what the calibration metrics depend on —
treat them as part of the model, not as tuning knobs.

### Pointing the research stack at it

```dotenv
JEV__BASE_URL=http://127.0.0.1:3000   # API root - the SDK appends /v1/systemone
JEV__API_KEY=                 # the shim's SHIM_TOKEN, if you set one
JEV__MODEL=                   # pin a version; empty records what the shim reports
JEV__PROMPT_VERSION=v1
JEV__MAX_CONCURRENCY=8
JEV__TIMEOUT=120
JEV__CACHE=processed/jev
```

Nothing about the endpoint appears in a specialist implementation.

## The three decision tasks

Each channel sends **one** `system_one` call carrying only its own
point-in-time inputs. No channel sees future returns, future indicators, or
future headlines.

### Technical → direction

| | |
|---|---|
| State | `rsi_14`, `macd_histogram`, `roc_5`, `roc_20`, `stoch_k_14`, `williams_r_14`, `cci_20`, `adx_14`, `bb_percent_20`, `keltner_percent` |
| Question | `Choice` over `down / flat / up` for the configured horizon |
| Columns | `p_down`, `p_flat`, `p_up`, `confidence`, `edge = p_up − p_down` |

`mu_hat_technical` is derived from the distribution through the channel's head
(`edge` is the signed view; the head learns its scale in return units on the
training window).

### Volatility → regime

A **separate** decision task, not a reuse of the directional one.

| | |
|---|---|
| State | `atr_14`, `bb_width_20`, `keltner_width`, `realized_vol_1`, `realized_vol_5`, `realized_vol_22` |
| Questions | `Choice` over `low / normal / high / extreme`, plus a `Noul` spike probability |
| Columns | `p_low`, `p_normal`, `p_high`, `p_extreme`, `confidence`, `p_spike`, `regime_level` |

`regime_level` is the probability-weighted regime index (0 calm → 3 crisis).
This channel represents expected *risk*; it never enters `mu` — it conditions
the copula.

### News → sentiment and tail relevance

| | |
|---|---|
| State | the processed headlines for that asset-day, plus the date |
| Questions | `Choice` over `bearish / neutral / bullish`, `Noul` downside-tail relevance, `Noul` volatility-event relevance |
| Columns | `p_bearish`, `p_neutral`, `p_bullish`, `confidence`, `sentiment`, `p_tail_risk`, `p_vol_event` |

No explanations, no recommendations — typed answers only. **Empty-headline rows
never reach the model**: they short-circuit to a deterministic uniform record,
so the no-news case is exactly reproducible and costs nothing.

## Caching

Every answer is computed once and snapshotted to
`data/processed/jev/<channel>.parquet`.

```python
key = sha256(asset | date | specialist | sha(state) | model | prompt_version | horizon)
```

All seven components matter: the same channel at a 1-day and a 20-day horizon
are different questions and must not collide, and a model or prompt rotation has
to invalidate cleanly rather than silently mixing two populations into one
feature table.

The **complete probability record** is stored, not a thresholded label — the
calibration metrics need the distribution, and so would any later
recalibration. Writes flush every 2000 rows, because losing a 79k-row build at
row 70k is a bad afternoon.

Walk-forward evaluation reads the frozen table and **never calls OpenJev inside
the backtest loop**. That is what makes 35 experiment arms affordable and
bit-for-bit reproducible.

## Evaluating the specialists before the portfolio

A miscalibrated channel still produces portfolio numbers; those numbers are
noise wearing a result's clothes. So prediction quality is scored first:

```bash
uv run main.py specialists
```

| Channel | Reported |
|---|---|
| technical | accuracy, macro-F1, balanced accuracy, ROC-AUC, **Brier**, **ECE** |
| volatility | accuracy, macro-F1, ROC-AUC, **Brier**, **ECE** |
| news | direction accuracy/F1, **sentiment↔forward-return correlation**, **Brier**, **ECE** |

Discrimination and calibration are reported separately because a channel can be
good at one and useless at the other, and the gate cares about both.

**Labels are causal.** `down/flat/up` uses a flat band scaled by each asset's
*trailing* volatility — a fixed return threshold would call FX flat almost
always and BTC flat almost never. `low/normal/high/extreme` uses trailing
quantiles of `|r|` over the previous 252 observations, so a label at `t` never
depends on anything after `t`. The warm-up rows are left unlabelled rather than
guessed.

## Where it sits in the experiment matrix

The `openjev` family runs the conventional baseline plus every non-empty subset
of channels moved onto OpenJev — 8 arms, everything else held identical:

```
Conventional · Tech · Vol · News · Tech+Vol · Tech+News · Vol+News · all three
```

The conventional side is not a straw man. Measured on this panel, the numeric
heads are: technical `mlp` IC +0.052 (t 4.36) > `ridge` +0.036 > `gbm` +0.026 >
`pcr` +0.012; volatility QLIKE `mlp` −7.543 < `gbm` −7.537 < `har` −7.464 <
`ridge` −7.431. OpenJev is being compared against the best of those, not the
default that shipped first.

See [inference.md](inference.md#the-experiment-harness) for the full matrix and
[training.md](training.md) for how a run is assembled.

## Running without a server

Every channel needs OpenJev, so a run with nothing serving will fail — loudly,
by design. Silently substituting neutral probabilities would produce results
that look fine and mean nothing.

For CI and dry runs there is an explicit escape hatch:

```dotenv
JEV__SYNTHETIC=true
```

Deterministic fake answers, no network. It logs a warning on every build and
stamps `"synthetic": true` into `run_meta.json`, so any run made this way
identifies itself. The test suite uses it — it exercises the real
cache → cube → specialist → gate path; only the answers are fabricated.

## Cost, and the honest caveats

- **Volume.** 2190 dates × 36 assets ≈ 79k asset-days per channel. The news
  channel is far cheaper because most rows are empty and skipped. Warm the cache
  once; every fold, ablation and horizon afterwards is free.
- **Licence.** `openjev/openjev` is CC BY-NC 4.0 — research use. Commercial
  application needs a conversation with the maintainers.
- **Leakage.** The prompts carry the date and instruct against post-date
  knowledge, but a pretrained model has read the future and no prompt removes
  that. This is why the conventional backends stay first-class and why the news
  channel additionally offers the fully-local `encoder` backend as a
  leakage-strict control. Treat OpenJev as **on trial**.
- **The hypothesis is open.** Nothing in this design assumes OpenJev helps. The
  research question — *do probabilistic specialists add tail-risk information
  beyond conventional predictors, and can Tail-VoI tell when?* — is answered by
  the matrix, and a negative result is a result. Every arm is persisted either
  way.
