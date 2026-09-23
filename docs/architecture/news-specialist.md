# News specialist

**Role.** Turn each asset-day's `headlines` string into numbers, then into a
directional view.

| | |
|---|---|
| Input | `features['news']` — `(T, N, 7)` OpenJev probabilities |
| Output | `z_news` `(N, 8)` and `μ̂_news` `(N,)` |
| Enters at | `fused_μ` — it is a **directional** source (`MU_SOURCES`) |
| Code | `yoda/specialists/news.py`, `yoda/specialists/jev/` |
| Contract | `Specialist` — `fit` / `encode` / `predict` |

The specialist itself is the same supervised head as the other two channels
([technical-specialist.md](technical-specialist.md#how-it-works)). What makes
this channel different is where its features come from: headlines reach the
portfolio through **OpenJev and nothing else**.

| `NEWS__BACKEND` | Feature source |
|---|---|
| `jev` (default) | OpenJev sentiment, downside-tail relevance and volatility-event probabilities, snapshotted to a point-in-time parquet cache |
| `none` | No news features at all — the `no-news` ablation arm. The channel is dropped from `TAILVOI__SOURCES` entirely rather than zero-filled. |

**Empty headlines.** Most asset-days have none. Those rows short-circuit to a
deterministic neutral record without a model call, and `news_count` is carried
so the model can distinguish "no news" from "neutral news".

The decision task, the prompts, the cache key and the calibration metrics are
documented in **[openjev-specialists.md](openjev-specialists.md)**.

## See also

- [openjev-specialists.md](openjev-specialists.md) — the decision protocol and serving
- [tail-voi-gate.md](tail-voi-gate.md) — what decides whether this channel earns its place
- [experiments.md](experiments.md#sources--agent-removal-10-arms) — the `no-news` arm
