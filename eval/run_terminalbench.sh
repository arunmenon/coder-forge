#!/usr/bin/env bash
# Terminal-Bench baseline / eval against an OpenAI-compatible endpoint (local or hosted).
# Use for the "gate on baseline" experiment: run a BASE model in-harness on a TB subset
# to see how much headroom exists before any fine-tuning.
#
# The endpoint can be your LOCAL MLX / llama.cpp server (set MODEL_ENDPOINT=http://localhost:PORT/v1)
# — no GPU rental needed for the gate.
#
# Env (see .env.example):
#   MODEL_ENDPOINT   OpenAI-compatible base URL (e.g. http://localhost:8080/v1)
#   MODEL_NAME       model id at that endpoint
#   MODEL_API_KEY    key (use "EMPTY"/"local" for a local server)
#   TB_LIMIT         number of tasks (default 25; full terminal-bench-core ~ 80-100)
#   TB_DATASET       dataset id (default terminal-bench-core)
#   TB_AGENT         agent/scaffold (default terminus)
set -euo pipefail
[[ -f .env ]] && set -a && source .env && set +a

MODEL_ENDPOINT="${MODEL_ENDPOINT:?set MODEL_ENDPOINT (e.g. http://localhost:8080/v1)}"
MODEL_NAME="${MODEL_NAME:-${DEFAULT_MODEL_NAME:-}}"   # .env wins; else BASE family default
: "${MODEL_NAME:?set MODEL_NAME in .env (or use a BASE that provides a default)}"
MODEL_API_KEY="${MODEL_API_KEY:-EMPTY}"
TB_LIMIT="${TB_LIMIT:-25}"
TB_DATASET="${TB_DATASET:-terminal-bench-core}"
TB_AGENT="${TB_AGENT:-terminus}"
RESULTS_DIR="${RESULTS_DIR:-eval/results}"
mkdir -p "$RESULTS_DIR"

echo ">> Terminal-Bench: agent=$TB_AGENT model=$MODEL_NAME limit=$TB_LIMIT dataset=$TB_DATASET"
echo ">> Docker required (each task runs in its own container). Checking..."
docker info >/dev/null 2>&1 || { echo "Docker not running" >&2; exit 1; }

# terminal-bench harness (pip install terminal-bench → `tb` CLI). See requirements-eval.txt.
command -v tb >/dev/null || { echo "Install harness: pip install terminal-bench" >&2; exit 1; }

# Route LiteLLM to the OpenAI-compatible endpoint: model must be openai/-prefixed so
# OPENAI_API_BASE takes effect (same convention as the SWE-bench mini path).
# TODO: verify the exact `tb run` flag names + dataset id for your installed terminal-bench
#       version (the CLI moves fast — confirm with `tb run --help` before the first real run).
OPENAI_API_BASE="$MODEL_ENDPOINT" \
OPENAI_API_KEY="$MODEL_API_KEY" \
  tb run \
    --agent "$TB_AGENT" \
    --model "openai/$MODEL_NAME" \
    --dataset "$TB_DATASET" \
    --n-tasks "$TB_LIMIT" \
    --output-path "$RESULTS_DIR/tb_${MODEL_NAME//\//_}" \
  | tee "$RESULTS_DIR/tb_${MODEL_NAME//\//_}.log"

echo ">> Done. Accuracy (resolved %) is in $RESULTS_DIR/tb_${MODEL_NAME//\//_}/ (results.json / summary)."
echo ">> Hold the agent + dataset fixed across base vs fine-tune runs so the delta is real."
