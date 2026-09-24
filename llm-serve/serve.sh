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

# vLLM's wheel is built against CUDA 13, so it needs two separate things:
#
#   libcudart.so.13   userspace, shipped inside the cu130/cu132 torch wheel
#   driver r580+      kernel side, the host's NVIDIA driver
#
# A datacenter GPU on an older driver can still run CUDA 13 through NVIDIA's
# forward-compatibility package, which supplies a newer user-mode driver that
# talks to the older kernel module. Put it on the path if it is installed;
# without it a pre-r580 host cannot run this build at all.
CUDA_COMPAT=$(ls -d /usr/local/cuda/compat /usr/local/cuda-13*/compat 2>/dev/null | head -1 || true)
if [ -n "$CUDA_COMPAT" ]; then
  export LD_LIBRARY_PATH="$CUDA_COMPAT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

driver_major() { nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
  | head -1 | cut -d. -f1; }

# The driver story, printed once so a failure downstream is already explained.
check_driver() {
  local major; major=$(driver_major || true)   # absent nvidia-smi must not abort
  [ -n "$major" ] || die "no nvidia-smi - this script serves on an NVIDIA GPU"
  if [ "$major" -ge 580 ]; then
    say "driver r$major - CUDA 13 native"
  elif [ -n "$CUDA_COMPAT" ]; then
    say "driver r$major (<580) - using forward-compat libs in $CUDA_COMPAT"
  else
    die "driver r$major is below the r580 that CUDA 13 requires, and no
  forward-compatibility libs are installed. vLLM's wheel is a CUDA 13 build,
  so it cannot run here as-is. Either:
    * install the compat package (datacenter GPUs only, A100 included):
        apt-get install -y cuda-compat-13-0
      then re-run; this script picks it up from /usr/local/cuda/compat
    * or update the host driver to r580+."
  fi
}

# Check only - this script installs nothing. It reports what actually went
# wrong rather than guessing: "vllm is missing" and "vllm is installed but
# fails to import" are different problems with different fixes, and hiding
# the interpreter's own error makes the second one look like the first.
ensure_deps() {
  local out status
  out=$(run python -c '
import importlib, sys
print("python:", sys.executable)
try:
    import torch
    print("torch: ", torch.__version__, "cuda", torch.version.cuda)
except Exception as exc:                      # torch is what vllm pins
    print("torch:  FAILED -", exc)
importlib.import_module("vllm")
print("vllm:   ok")
' 2>&1) && return 0
  status=$?

  printf '%s\n' "$out" >&2
  case "$out" in
    *"No module named 'vllm'"*)
      die "vLLM is not in this environment. It ships with the CUDA 13 extras:
    (cd $ROOT && uv sync --extra cu132)   # or cu130" ;;
    *libcudart.so.13*)
      die "this environment has a CUDA 12 torch; vLLM needs the CUDA 13 one.
  libcudart.so.13 ships inside the cu130/cu132 torch wheels, so:
    (cd $ROOT && uv sync --extra cu132)   # or cu130
  cu126/cu128/cu129 carry libcudart.so.12 and deliberately do not pull vLLM." ;;
    *"CUDA driver version is insufficient"*|*libcuda.so*)
      die "the CUDA 13 build loaded but the driver will not accept it - see
  above. Install the forward-compat package and re-run:
    apt-get install -y cuda-compat-13-0
  or update the host driver to r580+." ;;
    *)
      die "vLLM is installed but will not import (exit $status) - see the
  error above, which names the interpreter and the torch it found." ;;
  esac
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
    check_driver; ensure_deps; download ;;

  start)
    check_driver; ensure_deps; download; start_vllm; start_shim
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
