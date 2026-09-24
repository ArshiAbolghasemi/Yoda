# Risk-parameter policy — the seam

**Role.** Choose `(λ, B, c)` for the optimizer. Never weights. Never `α`.

| | |
|---|---|
| Input | `MarketState` — the gated view of the market plus the current book |
| Output | `RiskParams(lam, budget, turnover_penalty, alpha)` |
| Code | `yoda/policy/`, `yoda/rl/` |
| Contract | `RiskParamPolicy` — `act(state) → RiskParams` |

This one interface is the **only** difference between the two pipelines. Compare
`yoda/pipeline/tail_voli_risk.py` with `tail_voli_risk_rl.py`: the diff is
`make_policy`. Everything below is shared byte-for-byte.

## What a policy may and may not move

```python
RiskParams(lam, budget, turnover_penalty, alpha=0.05)
```

| Field | Meaning | Action? |
|---|---|---|
| `lam` | weight on the DRO-CVaR term in the objective | **yes** |
| `budget` | `B`, the CVaR upper bound | **yes** |
| `turnover_penalty` | `c`, the L1 turnover cost weight | **yes** |
| `alpha` | the CVaR tail level | **no — fixed hyperparameter** |

An agent that can move its own risk-measure confidence can make its risk look small
by redefining risk. `RiskParams.__post_init__` rejects `α ∉ (0,1)` and negative
`λ`/`c`, and `test_alpha_is_never_an_action` asserts `α` is identical across every
extreme action.

---

## `StaticRiskPolicy` — the static pipeline

```python
RiskParams(
    lam=STATIC_POLICY__LAM,
    budget=STATIC_POLICY__BUDGET,
    turnover_penalty=STATIC_POLICY__TURNOVER_PENALTY,
    alpha=OPT__ALPHA,
)
```

Config-driven and state-independent by default. **No training step.**

With `STATIC_POLICY__VOL_TARGET > 0` it becomes the classical vol-target rule — the
budget is rescaled by `vol_target / current_CVaR`, clipped to `[0.2×, 5×]` the base
budget:

```python
budget = clip(budget · vol_target / state.tail_stats["cvar"], 0.2·budget, 5·budget)
```

That vol-targeted form is the fair static comparison for the RL controller. A fixed
budget alone would be a straw man, and beating a straw man proves nothing.

| Setting | Default |
|---|---|
| `STATIC_POLICY__LAM` | 1.0 |
| `STATIC_POLICY__BUDGET` | 0.05 |
| `STATIC_POLICY__TURNOVER_PENALTY` | 0.001 |
| `STATIC_POLICY__VOL_TARGET` | 0.0 (off) |

> `λ = 1.0` against a daily `μ` of a few basis points means the risk term dominates
> and the book concentrates in FX. That is the first thing to calibrate.

---

## `RLRiskPolicy` — the RL pipeline

A trained SAC agent (stable-baselines3) behind the same one-method interface.

### Observation

`MarketState.observation(sources)` — concatenated in fixed order, `nan_to_num`'d:

| Block | Width | Contents |
|---|---|---|
| `fused_mu` | `N` | gated expected returns per asset |
| `fused_z` | `z_dim` = 8 | cross-sectional mean of the gated representation |
| `g` | `K` | the gate weights |
| `tail_stats` | 5 | `cvar, var, es, nu, vol` of the **current** book |
| `w_prev` | `N` | the book being held |

`2N + 13 + K` total; **85** for `N = 35`, `K = 2`.

The agent therefore sees what the gate decided, what the tail currently looks like,
and where it already is — enough to answer "should I be taking more or less risk
right now?", which is the only question it is being asked.

### Action

`Box(-1, 1, (3,))`, mapped by `action_to_params` onto the configured ranges:

| Action dim | Setting | Default range |
|---|---|---|
| 0 → `lam` | `RL__LAM_BOUNDS` | 0.1 – 10.0 |
| 1 → `budget` | `RL__BUDGET_BOUNDS` | 0.01 – 0.20 |
| 2 → `turnover_penalty` | `RL__TURNOVER_BOUNDS` | 0.0 – 0.01 |

Out-of-range actions are clipped, so the bounds and `α` are inviolable by
construction rather than by the agent's good behaviour.

### The environment

`yoda/rl/allocation_env.py`. Specialists, copula and gate are **frozen** — the
agent trains against a fixed world, which is what makes the comparison against the
static policy an apples-to-apples statement about risk budgeting rather than about
the whole stack.

One episode walks the training rows at the rebalance stride, up to `max_steps`
(256). Each step: map the action → solve → apply → observe.

### Reward

```
r_t = R_p,t − λ_CVaR · CVaR_α,t − λ_DD · DD_t − c · ‖w_t − w_{t−1}‖₁
```

| Term | Setting | Default |
|---|---|---|
| `λ_CVaR` | `RL__REWARD_LAMBDA_CVAR` | 1.0 |
| `λ_DD` | `RL__REWARD_LAMBDA_DD` | 0.5 |
| `c` | `RL__REWARD_TURNOVER_COST` | 0.001 |

> ⚠️ **`λ_CVaR` is a fixed reward-shaping weight and is a different quantity from
> the `λ` the policy chooses for the optimizer objective.** One shapes learning; the
> other is the action. Conflating them is the easiest mistake to make in this file,
> which is why they live in separate config sections with separate names.

`DD_t` is the running drawdown of the episode's equity curve, so the agent is
penalised for the *path*, not only the endpoint.

### Synthetic episodes

With `RL__SYNTHETIC_EPISODES > 0` a second training pass runs on an environment
where the realised return is drawn from the conditional copula instead of history.
The 2018-2026 sample contains essentially one major crash, and an agent that has
seen one crash has not learned about crashes.

The **states remain real**; only the outcomes are simulated, because states cannot
be synthesised from the feature panel. That limitation is deliberate and belongs in
any write-up.

### Training

```python
policy = train_sac(stack, rows, config.research.rl, rebalance_days=..., checkpoint=...)
```

```
env_ready rows=1509 obs_dim=85 synthetic=False
sac_train_start steps=5000 rows=1509
sac_synthetic_pass steps=256          # only if RL__SYNTHETIC_EPISODES > 0
sac_saved path=data/processed/rl/sac_risk_policy_<run_id>
```

**Budget.** Every environment step is a convex solve (~30 ms at
`RL__ENV_SCENARIOS` = 500), so wall-clock is roughly `total_timesteps × 30 ms`:
5000 steps ≈ 2.5 min, 50 000 ≈ 25 min. The default is tractable, not converged.

Knobs that matter, in order:

| Setting | Default | Why you would change it |
|---|---|---|
| `RL__TOTAL_TIMESTEPS` | 5000 | The single biggest quality/time lever |
| `RL__ENV_SCENARIOS` | 500 | Smaller = faster steps, noisier tail estimates |
| `RL__SYNTHETIC_EPISODES` | 0 | >0 adds copula-generated tail paths |
| `RL__LAM_BOUNDS` / `__BUDGET_BOUNDS` | (0.1,10) / (0.01,0.20) | Set them where the optimizer is actually responsive |
| `RL__LEARNING_RATE`, `__BATCH_SIZE`, `__BUFFER_SIZE`, `__LEARNING_STARTS` | SB3 defaults | Standard SAC tuning |

### Inference

```python
observation = state.observation(self.sources)
action, _ = self.model.predict(observation, deterministic=True)
return action_to_params(action, self.config, self.alpha)
```

Deterministic — no exploration noise at evaluation time. Reload a checkpoint with
`RLRiskPolicy.load(path, rl_config, alpha, sources)`.

---

## Why SAC, and why not verl

The action is a 3-dim continuous `Box` from an 85-dim float observation, trained
off-policy. That is textbook continuous control: gymnasium for the environment API,
stable-baselines3 for the algorithm.

`verl` is an RL framework for **LLM post-training** — PPO/GRPO over token
sequences, built around vLLM/SGLang rollouts and FSDP/Megatron sharding. It has no
continuous action space, no off-policy replay and no SAC. The one place it would
genuinely fit is RL-training the *news agent's LLM* against downstream portfolio
reward, which is explicitly out of scope here.

The bottleneck is not the RL library anyway — it is the convex solve inside every
environment step. A warm-started or batched solver is what would move that number.

---

## Adding a direct-weight policy

A policy that emits weights instead of `RiskParams` replaces the seam *and* the
`allocate` call, and touches nothing below it. That is the whole reason this
interface is one method wide.
