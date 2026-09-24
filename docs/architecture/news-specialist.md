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

## Headline history

The numeric channels carry their past inside the indicators — a 252-day
volatility percentile, a 60-day drawdown, three momentum horizons. News had no
such summary: every headline arrived context-free, so the third downgrade in a
week read exactly like the first, and a story already priced in on Monday looked
new again on Wednesday.

Each news state now carries the recent record alongside the current day:

```json
{"date": "2020-03-18", "headlines": ["..."], "news_count": 2,
 "recent_headlines": [
   {"date": "2020-03-17", "trading_days_ago": 1, "headlines": ["..."]},
   {"date": "2020-03-12", "trading_days_ago": 4, "headlines": ["..."]}],
 "recent_headlines_days": 30}
```

| Knob | Default | Meaning |
|---|---|---|
| `JEV__NEWS_LOOKBACK` | 30 | Trading days of prior headlines; 0 restores single-day states |
| `JEV__NEWS_HISTORY_MAX` | 40 | Ceiling on past headlines per state, most recent kept |

Days with no headlines are omitted from the window rather than sent as empties,
and entries run most-recent-first so the live story is never buried. The cap
exists because thirty days of a heavily covered name would otherwise crowd out
the 16k serving context.

**This adds no model calls.** A day with no headlines of its own is still
skipped even when the window behind it is full, so history enriches the calls
already being made rather than multiplying them. It does change `input_hash`,
so it invalidates the existing cache — see
[openjev-specialists.md](openjev-specialists.md).

**Empty headlines.** Most asset-days have none. Those rows short-circuit to a
deterministic neutral record without a model call, and `news_count` is carried
so the model can distinguish "no news" from "neutral news".

The decision task, the prompts, the cache key and the calibration metrics are
documented in **[openjev-specialists.md](openjev-specialists.md)**.

## See also

- [openjev-specialists.md](openjev-specialists.md) — the decision protocol and serving
- [tail-voi-gate.md](tail-voi-gate.md) — what decides whether this channel earns its place
- [experiments.md](experiments.md#sources--agent-removal-10-arms) — the `no-news` arm
