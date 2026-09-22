# News specialist

**Role.** Turn each asset-day's `headlines` string into numbers, then into a
directional view.

| | |
|---|---|
| Input | `features['news']` — `(T, N, 22)` |
| Output | `z_news` `(N, 8)` and `μ̂_news` `(N,)` |
| Enters at | `fused_μ` — it is a **directional** source (`MU_SOURCES`) |
| Code | `yoda/specialists/news.py`, `yoda/specialists/news_agent/` |
| Contract | `Specialist` — `fit` / `encode` / `predict` |

The specialist itself is the same pipeline as the other two
([technical-specialist.md](technical-specialist.md#how-it-works)). What makes this
channel different — and the most interesting and most suspect thing in the system —
is **where its 22 features come from**.

| `NEWS__BACKEND` | Feature source | Point-in-time safe? |
|---|---|---|
| `llm_agent` (default) | A LangGraph agent over an OpenAI-compatible model, snapshotted to a parquet cache | Prompt-level only — a pretrained model has seen the future |
| `encoder` | Frozen FinBERT sentiment + a sentence-transformer embedding, entirely local | Yes — the leakage-strict control |
| `none` | No news features at all | n/a — the `no-news` ablation arm |

All three emit the same cube shape, so the gate, optimizer and policies never learn
which one was used.

**Empty headlines.** Most asset-days have none. Those rows short-circuit to a
neutral record without a model call, and `news_count` is carried as a feature so
the model can distinguish "no news" from "neutral news".

---

## The LLM agent

### Why an agent and not a sentiment score

A headline carries several separable judgements — *what happened*, *which way it
pushes returns*, and *whether it widens the tail* — and the last of those is what
the Tail-VoI gate actually cares about. A single sentiment scalar collapses all
three. The graph keeps them separate and types the result.

### The graph

`yoda/specialists/news_agent/graph.py`, one run per asset-day with non-empty
headlines:

```
          ┌──► directional assessment ──┐
extract   │                             ├──► aggregate ──► NewsAssessment
events  ──┤                             │      (pure Python, no model call)
          └──► tail / vol relevance  ───┘
```

| Node | Model call | Output schema | Asks |
|---|---|---|---|
| `extract` | yes | `EventList` | The discrete market-relevant events, ≤ 6 short labels |
| `direction` | yes | `DirectionalView` | `sentiment ∈ [−1,1]`, `confidence ∈ [0,1]`, one-sentence rationale |
| `tail` | yes | `TailView` | `tail_risk_flag`, `vol_flag` — elevated downside tail or vol spike |
| `aggregate` | **no** | `NewsAssessment` | Assembles the record |

`direction` and `tail` fan out from `extract` and rejoin at `aggregate`, so it is
**three model calls per asset-day, not four**. Rows with no headlines never reach
the graph — `assess()` returns `NewsAssessment.neutral()` immediately.

### The record

`yoda/specialists/news_agent/schema.py` — Pydantic, validated on every response:

```python
class NewsAssessment(BaseModel):
    sentiment: float  # [-1, 1]  signed near-term return view
    confidence: float  # [0, 1]   how strongly the headlines support it
    tail_risk_flag: bool  # elevated downside-tail risk
    vol_flag: bool  # near-term volatility spike
    event_tags: list[str]  # short discrete labels
    rationale: str  # one sentence, grounded in the headlines
```

Which becomes the 22-column feature cube:

| Columns | Source |
|---|---|
| `sentiment`, `confidence`, `tail_risk_flag`, `vol_flag`, `event_count` | the record |
| `emb_0 … emb_15` | the rationale (or headlines) embedding, projected to `NEWS__LLM__EMBED_DIM` |
| `news_count` | the panel |

### Model access

`yoda/specialists/news_agent/client.py`. **OpenAI-compatible only.** Base URL, key
and model names come from the environment, so a self-hosted vLLM/Ollama server, an
internal gateway or OpenAI itself is a `.env` change and never a code change. No
provider SDK is hardcoded.

```dotenv
NEWS__LLM__BASE_URL=https://<your-openai-compatible-endpoint>/v1
NEWS__LLM__API_KEY=your_key
NEWS__LLM__MODEL=your-chat-model
NEWS__LLM__EMBED_MODEL=your-embedding-model
NEWS__LLM__TEMPERATURE=0
NEWS__LLM__MAX_CONCURRENCY=8
NEWS__LLM__PROMPT_VERSION=v1
```

Structured output is requested as `response_format={"type": "json_object"}` and
validated with Pydantic, rather than through a provider-specific schema endpoint —
JSON mode is the common denominator across compatible servers. A response that
fails to parse logs a warning and falls back to the schema's neutral defaults; it
never crashes a run. Calls retry four times with exponential backoff (`tenacity`).

**Determinism:** `temperature = 0` and a fixed `seed` where the endpoint honours it.

---

## Point-in-time discipline

Three mechanisms, in decreasing order of how much you should trust them:

**1. The cache is the point-in-time boundary.** Every record is computed once and
snapshotted; the backtest reads the frozen table and never calls a model inside the
evaluation loop. Folds and ablations are therefore deterministic and free to
repeat, and no run can accidentally condition on a later prompt version.

**2. The prompts carry the date and nothing else.** Every system prompt is *"You
are a financial news analyst working strictly as of {date}. Use only the headlines
supplied below. Do not use any knowledge of what happened after {date}."* and
instructs a neutral default when the headlines do not support a judgement.

**3. That is not enough, and the design says so.** A pretrained model has read the
future; no prompt removes that. This is why the `encoder` backend exists and why
the `news` experiment family reports `encoder` vs `llm_agent` as an explicit
leakage sensitivity check. Treat the LLM news channel as **on trial** and let the
gate decide whether `z_news` earns its place.

### The cache

`yoda/specialists/news_agent/cache.py`, written to
`data/processed/news_features.parquet` (zstd).

```python
key = sha256("|".join([asset, date, headlines, model, prompt_version]))
```

Changing the model or bumping `NEWS__LLM__PROMPT_VERSION` invalidates cleanly — old
records stay, new keys are computed, and both remain reproducible.

Raw embedding vectors are stored, **not** the projection, so changing `embed_dim`
re-projects locally instead of re-issuing thousands of embedding calls. The
projection is a seeded Gaussian random projection: data-independent by
construction, so it introduces no train/test leakage and needs no fitting. Rows
without an embedding project to zeros.

---

## The three backends

### `llm_agent` (default)

Everything above. Uncached rows are computed through a `ThreadPoolExecutor` sized
by `NEWS__LLM__MAX_CONCURRENCY`, with a `tqdm` progress bar, then flushed to
parquet in one write.

### `encoder` — the leakage-strict control

Frozen local models, no network calls:

- sentiment — `ProsusAI/finbert` (`NEWS__ENCODER_SENTIMENT_MODEL`), signed by label
  and weighted by score, neutral → 0
- embedding — `sentence-transformers/all-MiniLM-L6-v2`
  (`NEWS__ENCODER_EMBED_MODEL`), projected the same way

Requires the optional extra, and caches to `…news_features.encoder.parquet`:

```bash
uv sync --extra encoder
```

Both dependencies are imported lazily, so the package works without them installed
until this backend is selected — at which point it raises with the exact command to
fix it.

### `none` — the `no-news` arm

No news features at all. The pipeline drops `news` from `TAILVOI__SOURCES`
automatically; the gate then operates over two sources and everything downstream is
unchanged.

---

## Cost and how to control it

The panel holds ~59k asset-day rows; the non-empty subset drives cost. Budget
**3 chat calls + 1 embedding call per non-empty asset-day**, once, ever, for a
given `(model, prompt_version)`.

Ways to cut it, in order of preference:

1. **Do nothing** — the cache means you pay once across every fold, gate ablation
   and both pipelines.
2. Run `--families gate system policy` first; add `news` when you have an endpoint.
3. Restrict the panel with `SPLIT__*` and `PANEL__DATASET` while iterating.
4. Point `NEWS__LLM__BASE_URL` at a local vLLM/Ollama server — the wire protocol is
   identical, so nothing else changes.

## Building the features by hand

The cube is normally built on demand by `build_stack`. To warm the cache
separately:

```python
from yoda.common.alignment import build_panel
from yoda.config import load_config
from yoda.specialists.news import build_news_features

config = load_config()
cube, names = build_news_features(build_panel(config), config)
print(cube.shape, names)  # (T, N, 22), the column names
```

The experiment harness does exactly this once per backend and shares the cube
across every arm that uses it.

## Implementation note

The spec describes `μ̂_news` as *signed sentiment × confidence*. Here `sentiment`
and `confidence` are two of the 22 input features and the supervised ridge head
learns their mapping to forward returns instead. The head is linear in each, so it
does not represent their product exactly; add a `sentiment × confidence`
interaction column to `_llm_agent_features` if you want the spec's form literally.

## See also

- [tail-voi-gate.md](tail-voi-gate.md) — the gate is what decides whether this
  channel earns its place
- [inference.md](inference.md#the-experiment-harness) — the `news` ablation family
