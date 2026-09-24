#!/usr/bin/env bash
# Run the ablation matrix. Extra arguments pass through to main.py.
#
#   ./scripts/run-experiments.sh                       every arm, static policy
#   ./scripts/run-experiments.sh --policies static rl  every arm under BOTH
#   ./scripts/run-experiments.sh --families gate       one family only
#
# Crossing with `rl` trains a SAC agent per fold for every arm, so it costs far
# more than twice a static sweep. Start with one family, or with --policies
# static, before committing to the full cross.
#
# Results land in data/processed/runs/: summary.csv, gate_weights_by_regime.csv
# and one folder per run with weights.parquet, ledger.parquet and run_meta.json.
set -euo pipefail

cd "$(dirname "$0")/.."
uv run --no-sync --active python main.py experiments "$@"
