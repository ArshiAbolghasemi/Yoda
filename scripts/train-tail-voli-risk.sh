#!/usr/bin/env bash
# Tail-VoI gated DRO-CVaR allocation with a static risk policy.
# Fits specialists + copula, generates counterfactual targets, trains the gate,
# walk-forward backtests and scores. Extra flags pass through to main.py:
set -euo pipefail

cd "$(dirname "$0")/.."
uv run --no-sync --active main.py tail-voli-risk "$@"
