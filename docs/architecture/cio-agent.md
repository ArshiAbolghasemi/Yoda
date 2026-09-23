# CIO agent

The specialists form views. The CIO decides **whom to trust today** and **how
much risk the book should carry**. This agent does exactly those two things, in
one OpenJev call, and nothing else.

| | |
|---|---|
| Input | the desk briefing — each channel's view, dispersion and state digest |
| Output | a trust distribution `g`, and a risk stance → `(λ, B, c)` |
| Fills | `Gate` **and** `RiskParamPolicy` |
| Code | `yoda/cio/agent.py` |

## Why the CIO does not emit weights

This is the load-bearing design decision, so it is worth being explicit.

A HedgeAgents-style manager reads the analysts and outputs a portfolio. That
concentrates three separable judgements into one opaque function, and it costs
two things this project cannot afford:

**Attribution.** Tail-VoI exists to *measure* what each information source is
worth in the tail. If a single agent consumes all three channels and emits
weights, there is no longer a quantity to measure — you cannot say whether a
good quarter came from the news channel or from the manager's mood.

**Risk guarantees.** `w ≥ 0`, `Σw = 1`, the per-asset cap and `CVaR_α(w) ≤ B`
are enforced by the convex program, not by good behaviour. An agent that emits
weights directly turns every one of those into a suggestion.

So the CIO sets *parameters*, and the DRO-CVaR solver still builds the book.
The agent that emits weights directly is the direct-weight RL variant, and it
is out of scope by design.

## The decision

One `system_one` call, two typed questions:

```python
"trust":  Choice(criteria={"technical": ..., "volatility": ..., "news": ...})
"stance": Score(criteria=[ "maximally defensive", ..., "maximally aggressive" ])
```

`ChoiceAnswer.probabilities` over the three channels **is** the gate weight
vector — no parsing, no thresholding, and it sums to one by construction.
`ScoreAnswer.score` is the probability-weighted stance on the rubric.

The briefing carries numbers only — each channel's mean view, its
cross-sectional dispersion, its strongest signal, plus a short state digest. No
prose, no recommendations, and nothing dated after the decision. A test asserts
the briefing contains no forward-looking field.

## Mapping a stance onto risk parameters

```python
action = [1 - 2·stance,  2·stance - 1,  0]
params = action_to_params(action, rl_config, alpha)     # the SAC mapping
```

The stance is deliberately routed through **the same `action_to_params` mapping
SAC uses**, against the same `RL__LAM_BOUNDS` / `RL__BUDGET_BOUNDS`. "The CIO
picks the risk parameters" and "SAC picks the risk parameters" are therefore
choices from an identical option set, and the comparison between them is about
judgement rather than about reachable ranges.

Aggressive means *less* risk aversion and a *wider* budget, so `λ` runs
backwards against the stance while `B` runs with it — asserted by a test.
Turnover cost is not a CIO judgement and stays at the configured value. `α` is
never touched.

## Using it

```bash
./scripts/train-tail-voli-risk.sh --gate cio                  # CIO sets the gate
./scripts/train-tail-voli-risk.sh --policy cio                # CIO sets the risk stance
./scripts/train-tail-voli-risk.sh --gate cio --policy cio     # both, one decision
```

With both sockets filled the *same object* serves each, so one decision per
rebalance drives the gate and the stance rather than two contradictory ones.

Three arms are in the matrix: `gate_cio` (in the `gate` family, head-to-head
against TailVoI / Accuracy / Attention / EqualWeight), `policy_cio` and
`cio_full` (in the `policy` family, beside the static rule and SAC).

## Cost, caching and failure

One call per **rebalance**, not per asset-day — roughly 136 per run at a 5-day
cadence, against ~79k for a specialist channel. Decisions are cached on demand
by `(state hash, model, prompt version, sources)` and flushed every 25 rows, so
a repeated run or a second ablation arm costs nothing.

The agent is built to **never sink a run**:

- construction never touches the network — the model id resolves lazily, and an
  unreachable endpoint records `"unresolved"` rather than raising (keys written
  under it cannot collide with real-model keys, which is the safe direction);
- a failed decision logs and falls back to an even gate;
- `act()` with no decision yet returns a neutral stance rather than inventing
  conviction.

That is what lets the test suite exercise the CIO with no server running.

## The open question

Whether a reasoning model allocates attention better than a regressor trained
on counterfactual tail-risk targets is exactly the kind of claim this repository
is built to test rather than assert. `gate_cio` sits in the same family as
`gate_tailvoi` on identical splits, and the answer — positive, neutral or
negative — is whatever the table says.

## See also

- [tail-voi-gate.md](tail-voi-gate.md) — the learned gate the CIO competes with
- [risk-policy.md](risk-policy.md) — the seam, and the SAC controller it competes with
- [openjev-specialists.md](openjev-specialists.md) — serving and the decision protocol
