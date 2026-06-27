#!/usr/bin/env bash
# Serve a model behind an OpenAI-compatible endpoint with the FAMILY's agentic tool-calling.
# Every family-specific serving knob (tool-call parser, reasoning parser, sampling, context,
# min vLLM) comes from config/families/<FAMILY>.yaml — no hardcoded parser. Run on the GPU pod.
#
#   FAMILY=qwen30 serve/serve_vllm.sh                        # serves the base model
#   FAMILY=qwen30 serve/serve_vllm.sh outputs/<slug>-merged  # serves your fine-tune
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAMILY="${FAMILY:-qwen30}"
# Pull the family profile into the environment (BASE_MODEL, TOOL_CALL_PARSER, ...).
eval "$(PYTHON="${PYTHON:-python3}" bash "$HERE/scripts/family.sh" export "$FAMILY")"

MODEL="${1:-$BASE_MODEL}"
PORT="${PORT:-8000}"

# Assert the family's minimum vLLM, when declared (qwen36 needs >=0.17.0 for the DeltaNet arch).
# Fail closed: missing or unparsable vLLM is an error, not a silent pass.
if [[ -n "${MIN_VLLM:-}" ]]; then
  command -v vllm >/dev/null || { echo "ERROR: family $FAMILY needs vLLM >= $MIN_VLLM but 'vllm' is not installed" >&2; exit 1; }
  have=$(vllm --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)
  [[ -n "$have" ]] || { echo "ERROR: could not determine vLLM version (need >= $MIN_VLLM)" >&2; exit 1; }
  if [[ "$(printf '%s\n%s\n' "$MIN_VLLM" "$have" | sort -V | head -1)" != "$MIN_VLLM" ]]; then
    echo "ERROR: family $FAMILY needs vLLM >= $MIN_VLLM, found $have" >&2; exit 1
  fi
fi

ARGS=( --port "$PORT" --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" )
[[ "${ENABLE_AUTO_TOOL_CHOICE:-0}" == "1" ]] && ARGS+=( --enable-auto-tool-choice )
[[ -n "${TOOL_CALL_PARSER:-}" ]] && ARGS+=( --tool-call-parser "$TOOL_CALL_PARSER" )
[[ -n "${REASONING_PARSER:-}" ]] && ARGS+=( --reasoning-parser "$REASONING_PARSER" )
# Per-family extra flags from the profile (e.g. qwen36's --mamba-ssm-cache-dtype float16).
if [[ -n "${EXTRA_ARGS:-}" && "$EXTRA_ARGS" != "[]" ]]; then
  while IFS= read -r _a; do ARGS+=( "$_a" ); done \
    < <(printf '%s' "$EXTRA_ARGS" | "${PYTHON:-python3}" -c "import json,sys;[print(x) for x in json.load(sys.stdin)]")
fi

echo ">> Serving $MODEL on :$PORT  (family=$FAMILY parser=${TOOL_CALL_PARSER:-none} max_len=$MAX_MODEL_LEN)"
vllm serve "$MODEL" "${ARGS[@]}"
