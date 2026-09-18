#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv sync --active --all-groups
