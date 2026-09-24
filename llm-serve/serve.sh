#!/usr/bin/env bash
# Serve OpenJev 27B without Docker.
#
#   TailRiskFlow specialists ──► decision shim :3000 ──► vLLM :8000 ──► openjev/openjev
#
# Reads the project's root .env, the same file docker-compose uses, so the
# whole project is configured in one place. This script never installs
# anything: vLLM is a main dependency of the project, so it runs out of the
# project environment, and if it is missing the script says what to run.
#
#   ./serve.sh              start both, wait until ready, stay in foreground
#   ./serve.sh start -d     start both in the background
#   ./serve.sh status       what is running, and whether it answers
#   ./serve.sh logs [vllm|shim]
#   ./serve.sh stop
#   ./serve.sh download     fetch the weights only
set -euo pipefail

cd "$(dirname "$0")"
ROOT=$(cd .. && pwd)

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# Serving settings live in the project's single .env at the repo root - the
# same file docker-compose reads. Missing is fatal: falling back to defaults
# would serve with no HF token and an uncalibrated readout. Loaded before the
# defaults below so anything set there wins.
[ -f "$ROOT/.env" ] || die "no $ROOT/.env - fill in the repo-root .env first"
set -a; . "$ROOT/.env"; set +a

MODEL_REPO="${MODEL_REPO:-openjev/openjev}"
MODEL_DIR="${MODEL_DIR:-./openjev}"
HOST="${SERVE_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
SHIM_PORT="${SHIM_PORT:-3000}"
SERVED_NAME="${SERVED_NAME:-qwen}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

LOGS=./logs
PIDS=./run
mkdir -p "$LOGS" "$PIDS"

# ---------------------------------------------------------------- environment

# Check only - this script installs nothing.
ensure_deps() {
  run python -c "import vllm" >/dev/null 2>&1 || die \
    "vLLM is not in the project environment. Install it yourself with
    (cd $ROOT && uv sync --extra cu130)   # or cu129 / cu126 / cpu
  vLLM publishes no macOS wheels, so serving needs a Linux box."
  command -v hf >/dev/null 2>&1 || run hf --help >/dev/null 2>&1 || die \
    "the 'hf' CLI is missing - (cd $ROOT && uv sync) should provide it"
}

run() { uv run --no-sync --active "$@"; }

download() {
  if [ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/helper/shim.py" ]; then
    say "model already present at $MODEL_DIR"
    return
  fi
  say "downloading $MODEL_REPO -> $MODEL_DIR (~55 GB on first run)"
  run hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
  [ -f "$MODEL_DIR/helper/shim.py" ] || die "shim.py missing from $MODEL_DIR"
}

# -------------------------------------------------------------------- process

alive() { [ -f "$PIDS/$1.pid" ] && kill -0 "$(cat "$PIDS/$1.pid")" 2>/dev/null; }

wait_for() { # name url timeout
  local name=$1 url=$2 limit=${3:-1800} waited=0
  printf '    waiting for %s ' "$name"
  until curl -sf "$url" >/dev/null 2>&1; do
    alive "$name" || { echo; die "$name died on startup - see $LOGS/$name.log"; }
    [ "$waited" -ge "$limit" ] && { echo; die "$name did not come up in ${limit}s"; }
    sleep 5; waited=$((waited + 5)); printf '.'
  done
  printf ' ready (%ds)\n' "$waited"
}

start_vllm() {
  alive vllm && { say "vLLM already running (pid $(cat $PIDS/vllm.pid))"; return; }
  say "starting vLLM on :$VLLM_PORT"
  nohup uv run --no-sync --active vllm serve "$MODEL_DIR" \
    --host "$HOST" --port "$VLLM_PORT" --served-model-name "$SERVED_NAME" \
    --enable-prefix-caching \
    --max-model-len 16384 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --limit-mm-per-prompt '{"image":1}' \
    --trust-remote-code \
    --max-num-seqs 256 \
    --max-logprobs 64 \
    --gdn-prefill-backend triton \
    --quantization fp8 \
    >"$LOGS/vllm.log" 2>&1 &
  echo $! >"$PIDS/vllm.pid"
  wait_for vllm "http://$HOST:$VLLM_PORT/v1/models"
}

start_shim() {
  alive shim && { say "shim already running (pid $(cat $PIDS/shim.pid))"; return; }
  say "starting OpenJev decision shim on :$SHIM_PORT"
  # Published readout calibration - these constants were fitted for this
  # readout, so changing them changes what the probabilities mean.
  VLLM="http://$HOST:$VLLM_PORT/v1" \
  TOKENIZER="$MODEL_DIR" \
  READOUT_T="${READOUT_T:-0.85}" \
  READOUT_NOUL_T="${READOUT_NOUL_T:-1.829074}" \
  READOUT_NOUL_BIAS="${READOUT_NOUL_BIAS:-0}" \
  READOUT_TARGETED="${READOUT_TARGETED:-1}" \
  READOUT_INSTR_STYLE="${READOUT_INSTR_STYLE:-pyrepr}" \
  SHIM_STAGGER="${SHIM_STAGGER:-1}" \
  SHIM_TOKEN="${SHIM_TOKEN:-}" \
  nohup uv run --no-sync --active python "$MODEL_DIR/helper/shim.py" \
    --host "$HOST" --port "$SHIM_PORT" \
    >"$LOGS/shim.log" 2>&1 &
  echo $! >"$PIDS/shim.pid"
  wait_for shim "http://$HOST:$SHIM_PORT/v1/version" 120
}

stop_one() {
  alive "$1" || { rm -f "$PIDS/$1.pid"; return; }
  local pid; pid=$(cat "$PIDS/$1.pid")
  say "stopping $1 (pid $pid)"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$PIDS/$1.pid"
}

# ----------------------------------------------------------------- subcommands

case "${1:-start}" in
  download)
    ensure_deps; download ;;

  start)
    ensure_deps; download; start_vllm; start_shim
    cat <<MSG

OpenJev is serving.
  shim     http://$HOST:$SHIM_PORT      <- point the research stack here
  vllm     http://$HOST:$VLLM_PORT/v1
  logs     $LOGS/{vllm,shim}.log

Add to the project .env (API root, the SDK appends /v1/systemone):
  JEV__BASE_URL=http://$HOST:$SHIM_PORT
MSG
    [ "${2:-}" = "-d" ] && exit 0
    say "following logs; Ctrl-C detaches (use './serve.sh stop' to shut down)"
    tail -f "$LOGS/vllm.log" "$LOGS/shim.log" ;;

  stop)
    stop_one shim; stop_one vllm; say "stopped" ;;

  status)
    for name in vllm shim; do
      if alive "$name"; then
        printf '  %-5s running (pid %s)\n' "$name" "$(cat "$PIDS/$name.pid")"
      else
        printf '  %-5s stopped\n' "$name"
      fi
    done
    curl -sf "http://$HOST:$SHIM_PORT/v1/version" >/dev/null 2>&1 \
      && say "shim answers on :$SHIM_PORT" \
      || say "shim not answering on :$SHIM_PORT" ;;

  logs)
    tail -f "$LOGS/${2:-vllm}.log" ;;

  *)
    die "unknown command: $1 (start|stop|status|logs|download)" ;;
esac
