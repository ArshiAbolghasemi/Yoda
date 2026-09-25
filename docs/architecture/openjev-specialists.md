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

vLLM ships with the **`cu129`** and **`cu130`** extras, so one `uv sync` gives
you both the research stack and the server:

```bash
uv sync --extra cu129     # CUDA 12 host (driver r570+)
uv sync --extra cu130     # CUDA 13 host (driver r580+)
```

It lives there rather than in the main dependencies because every vLLM wheel on
PyPI that supports Python 3.14 is built against **CUDA 13** — PyTorch moved its
default PyPI build to CUDA 13 at torch 2.11.0, and vLLM followed. The
`cu126`/`cu128`/`cu129` channels ship `libcudart.so.12`, so a vLLM installed
beside them resolves cleanly and then dies at import with
`ImportError: libcudart.so.13: cannot open shared object file`. Scoping it to
the matching channel makes that pairing impossible to produce.

vLLM compiles one wheel per CUDA minor version and publishes them at
`wheels.vllm.ai`; only `cu129` and `cu130` exist. PyPI carries the CUDA 13 build
alone, so each serving extra pulls vLLM from its own index rather than PyPI.
`cu126`, `cu128` and `cu132` have no vLLM build and stay research-only.

vLLM also pulls `torchaudio`, `torchvision` and `torchcodec`, each of which
asserts an **exact** CUDA match against torch at import:

> RuntimeError: Detected that PyTorch and TorchAudio were compiled with
> different CUDA versions. PyTorch has CUDA version 13.2 whereas TorchAudio has
> CUDA version 13.0.

Left to PyPI they track whatever CUDA PyTorch currently defaults to (13.0), so
they silently mismatch any non-default torch. All three are therefore pinned to
the same `cu130` index as torch. `cu132` publishes no torchaudio at all, which
is why it is research-only.

### Matching the host driver

CUDA 13 is a two-sided requirement, and only the first half comes from `uv sync`:

| Half | Provided by | Symptom when missing |
|---|---|---|
| matching `libcudart` | the channel's torch wheel | `ImportError: libcudart.so.13` |
| a driver the build accepts | the **host**, not the venv | `CUDA driver version is insufficient` |

**On a CUDA 12 host, use `cu129`** — a 12.9 build runs on an r570 driver through
CUDA minor-version compatibility, with nothing else to install. Reach for
`cu130` only when the host is r580+.

If you need the CUDA 13 build on an older datacenter GPU anyway, NVIDIA's
forward-compatibility package supplies a newer user-mode driver on top of the
older kernel module:

```bash
apt-get install -y cuda-compat-13-0
```

`serve.sh` checks `nvidia-smi` before it starts anything, adds
`/usr/local/cuda/compat` to `LD_LIBRARY_PATH` when it is present, and refuses to
launch with the exact remedy when the driver is too old and the compat libs are
absent. Failing there costs a second; failing after a 55 GB download does not.

Both paths use the serving configuration published on the model card:

| Setting | Value |
|---|---|
| quantization | `fp8` — needs SM89+, see below |
| prefix caching | enabled |
| max model length | 16384 |
| max concurrent sequences | 256 |
| KV cache | `KV_CACHE_MEMORY` bytes, else `--gpu-memory-utilization` |
| max logprobs | 64 |
| prefill backend | `--gdn-prefill-backend triton` |
| pinned versions | `vllm==0.29.0` (in the `cu130` extra), `openai==3.16.2`, `httpx==0.28.1` |

### FP8 needs SM89 or newer

The model card serves FP8, which has no hardware support below **SM89** (Ada
L40S, Hopper H100). An A100 is SM80, and vLLM's CUTLASS W8A8 kernel aborts
there:

```
RuntimeError: cutlass_scaled_mm_sm80_epilogue,
  csrc/.../quantization/w8a8/cutlass/scaled_mm_c2x.cu:89
```

vLLM does not catch this itself: `CutlassFP8ScaledMMLinearKernel.is_supported()`
returns `True` for any CUDA device without checking compute capability, so it
wins kernel selection on an A100 and then the sm80 CUTLASS dispatcher — which is
int8-only — rejects the FP8 tensors.

`serve.sh` reads the card's compute capability and picks the right path:

| Compute capability | What it does |
|---|---|
| ≥ 8.9 | native FP8 |
| < 8.9 | `VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel` — selection falls through to Marlin FP8, which supports 7.5+ |

Marlin keeps the memory win (weights stay half-size) and gives up the speed
one, which is the difference between serving and not serving on an A100. Set
`QUANTIZATION=none` in the root `.env` to serve unquantised instead — at 27B
that is ~54 GB of weights, so re-read `KV_CACHE_MEMORY` from the boot log
afterwards.

### Sizing the KV cache

By default vLLM derives the cache from `--gpu-memory-utilization`: whatever is
left after weights, activation peak and CUDA graphs. That remainder moves when
any of those three do, so the same fraction gives a different cache after a vLLM
or model update — and throughput changes with it for no visible reason.

vLLM prints the two numbers worth pinning on the first boot of a given
model/GPU pair:

```
Replace gpu_memory_utilization config with `--kv-cache-memory=57794628301`
(53.83 GiB) to fit into requested memory, or `--kv-cache-memory=67405561344`
(62.78 GiB) to fully utilize gpu memory.
```

Put the one you want in the root `.env` and `serve.sh` passes it instead of the
fraction:

```bash
KV_CACHE_MEMORY=67405561344     # A100 80GB, OpenJev fp8, max-model-len 16384
```

The number is specific to the GPU, the quantisation and the context length —
read it off your own boot log rather than copying one. Leave it empty to go back
to the fraction.

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

## The three agents

Each channel has its own prompt, stored verbatim in
`yoda/specialists/prompts/agents.py`, and returns a strict JSON view validated
against a Pydantic schema. The prompts live in one package so a revision is a
reviewable diff, and `JEV__PROMPT_VERSION` keys the cache: changing a prompt
invalidates every answer it produced.

### One common representation

All three expose the **same six core variables**, which is what makes Tail-VoI
computable over them — the gate compares channels, and can only do that if they
speak one language:

| | Variable | Range |
|---|---|---|
| `μ` | `expected_return_score` — direction | −1 … 1 |
| `m` | `expected_magnitude` — size of the move | 0 … 1 |
| `c` | `confidence` | 0 … 1 |
| `u` | `epistemic_uncertainty` | 0 … 1 |
| `τ` | `downside_tail_risk` | 0 … 1 |
| `r` | `regime_shift_probability` | 0 … 1 |

The volatility agent maps differently and deliberately: its `μ` is
`directional_bias` (its secondary view) and its `m` is `volatility_score` — the
expected *size* of the move, not its direction. Each agent then adds its own
fields on top: `trend_strength`, `signal_agreement`, `heavy_tail_score`,
`jump_risk`, `liquidity_stress`, `event_types` and so on.

| Agent | Sees | Adds beyond the core |
|---|---|---|
| news | point-in-time headlines only | `upside_tail_potential`, `volatility_impact`, `event_types` |
| technical | price, momentum, trend, volume, market structure | `trend_strength`, `momentum_score`, `volume_confirmation`, `breakout`/`breakdown_probability`, `signal_agreement` |
| volatility | realized vol, ATR, band widths, drawdowns, dispersion | `volatility_regime`, `tail_event_probability`, `tail_severity`, `negative_skew_risk`, `heavy_tail_score`, `jump_risk`, `liquidity_stress`, systemic vs idiosyncratic split |

### Numbers and prose go different ways

The numeric fields become `z_i` and feed the gate, the copula and the
optimizer. `view_summary` and the evidence lists are kept as the **semantic
message for the CIO** and never reach the numeric path — the gate must not be
able to key off free text it cannot calibrate.

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
