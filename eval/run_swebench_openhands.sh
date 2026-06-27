#!/usr/bin/env bash
# Phase 0 baseline + Phase 3 re-eval: run SWE-bench Verified through an agent harness
# against an OpenAI-compatible endpoint, on a subset, and report resolved%.
#
# The endpoint can be a hosted base model (Phase 0, no GPU needed) or your vLLM-served
# fine-tune (Phase 3). Keep HARNESS + EVAL_LIMIT identical across runs so the delta is
# real and not a harness artifact.
#
# Env (see .env.example):
#   MODEL_ENDPOINT   OpenAI-compatible base URL (e.g. https://openrouter.ai/api/v1)
#   MODEL_NAME       model id at that endpoint
#   MODEL_API_KEY    key for the endpoint
#   EVAL_LIMIT       number of instances (default 50; full Verified = 500)
#   HARNESS          openhands | mini   (default openhands)
set -euo pipefail
[[ -f .env ]] && set -a && source .env && set +a

MODEL_ENDPOINT="${MODEL_ENDPOINT:?set MODEL_ENDPOINT}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME}"
MODEL_API_KEY="${MODEL_API_KEY:-EMPTY}"
EVAL_LIMIT="${EVAL_LIMIT:-50}"
HARNESS="${HARNESS:-openhands}"
DATASET="${DATASET:-princeton-nlp/SWE-bench_Verified}"
RESULTS_DIR="${RESULTS_DIR:-eval/results}"
mkdir -p "$RESULTS_DIR"

echo ">> Harness=$HARNESS  Model=$MODEL_NAME  Limit=$EVAL_LIMIT  Dataset=$DATASET"
echo ">> Docker required (per-instance test execution). Checking..."
docker info >/dev/null 2>&1 || { echo "Docker not running" >&2; exit 1; }

if [[ "$HARNESS" == "mini" ]]; then
  # mini-swe-agent: simplest/cheapest path to a signal.
  #   pip install mini-swe-agent  (in requirements-eval.txt)
  command -v mini-extra >/dev/null || { echo "pip install mini-swe-agent" >&2; exit 1; }
  # mini routes through litellm: the model must be `openai/`-prefixed so OPENAI_API_BASE
  # takes effect (MSWEA_MODEL_API_BASE does not exist). --output is a DIRECTORY.
  MSWEA_MODEL_NAME="openai/$MODEL_NAME" \
  OPENAI_API_BASE="$MODEL_ENDPOINT" \
  OPENAI_API_KEY="$MODEL_API_KEY" \
    mini-extra swebench \
      --subset verified --split test \
      --slice ":$EVAL_LIMIT" \
      --output "$RESULTS_DIR/mini_${MODEL_NAME//\//_}"
  echo ">> Predictions: $RESULTS_DIR/mini_${MODEL_NAME//\//_}/preds.json"
  echo ">> Score separately: python -m swebench.harness.run_evaluation (or sb-cli submit)."
  exit 0
fi

# --- OpenHands path (V1 harness; matches the Nebius training trajectory format) ---
# The V1 eval harness lives in its own repo (the old evaluation/ tree was removed):
#   git clone https://github.com/OpenHands/benchmarks && cd benchmarks \
#     && git submodule update --init --recursive && make build   # uv sync
OPENHANDS_DIR="${OPENHANDS_DIR:?clone OpenHands/benchmarks and set OPENHANDS_DIR}"
LLM_CONFIG="$OPENHANDS_DIR/llm_config.json"

# Point the harness at our endpoint via a JSON LLM config (openhands.sdk.LLM).
# The `openai/` prefix routes litellm to the OpenAI-compatible path.
cat > "$LLM_CONFIG" <<EOF
{"model": "openai/${MODEL_NAME}", "base_url": "${MODEL_ENDPOINT}", "api_key": "${MODEL_API_KEY}"}
EOF
echo ">> Wrote $LLM_CONFIG (-> $MODEL_ENDPOINT)"

pushd "$OPENHANDS_DIR" >/dev/null
# Inference. Flags are NAMED (not positional). Default --agent-type is the standard
# openhands.sdk.Agent (CodeActAgent no longer exists). Prints {"output_json": "..."}.
uv run swebench-infer "$LLM_CONFIG" \
  --dataset "$DATASET" --split test \
  --workspace docker --max-iterations 100 \
  --n-limit "$EVAL_LIMIT" --num-workers 1 \
  | tee "$RESULTS_DIR/openhands_infer.log"
# Pull the emitted output path, then score it (runs the repo test suites in Docker):
OUTPUT_JSON=$(grep -o '"output_json": *"[^"]*"' "$RESULTS_DIR/openhands_infer.log" \
  | tail -1 | sed -E 's/.*"output_json": *"([^"]*)".*/\1/')
uv run swebench-eval "$OUTPUT_JSON"
popd >/dev/null

echo ">> Done. resolved% is in the swebench-eval report under eval_outputs/."
