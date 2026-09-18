#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv run --no-sync --active python -m yoda.data.pipeline
