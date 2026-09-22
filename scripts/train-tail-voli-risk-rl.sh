#!/usr/bin/env bash
# Same stack, SAC risk controller. Every RL step is a convex solve,
# so RL__TOTAL_TIMESTEPS drives the runtime - raise it deliberately.
# Extra flags pass through to main.py:
#   ./scripts/train-tail-voli-risk-rl.sh --gate tailvoi --run-id rl_tailvoi
set -euo pipefail

cd "$(dirname "$0")/.."
uv run --no-sync --active main.py tail-voli-risk-rl "$@"
