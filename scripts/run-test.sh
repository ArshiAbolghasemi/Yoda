#!/usr/bin/env bash
# Lint, format-check, then run the suite. Extra arguments go to pytest:
#
#   ./scripts/test.sh                 everything
#   ./scripts/test.sh -k alignment    one area
#   ./scripts/test.sh -x -q           stop on first failure
#
# The suite reads the real data/financial_dataset.parquet and skips if it is
# absent, so run `dvc pull` first for full coverage.
set -euo pipefail

cd "$(dirname "$0")/.."
TARGETS=(yoda tests main.py)

echo "==> ruff check"
uv run --no-sync --active ruff check "${TARGETS[@]}"

echo "==> ruff format --check"
uv run --no-sync --active ruff format --check "${TARGETS[@]}"

echo "==> pytest"
# `python -m` keeps the working directory on sys.path, which is how the package
# is importable without installing it.
uv run --no-sync --active python -m pytest "$@"
