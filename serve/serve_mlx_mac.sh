#!/usr/bin/env bash
# Phase 5 — serve the quantized model locally on the Mac via MLX, OpenAI-compatible.
# Point OpenHands (or any OpenAI client) at http://localhost:8080/v1.
# MLX-native is ~2-3x faster than Ollama/llama.cpp for this MoE (~90-130 vs ~43 tok/s).
set -euo pipefail

MODEL_DIR="${1:-models/qwen3-coder-30b-a3b-mlx-4bit}"
PORT="${PORT:-8080}"
PYTHON="${PYTHON:-python3}"

if ! $PYTHON -c "import mlx_lm" 2>/dev/null; then
  echo "Install MLX: pip install mlx-lm" >&2
  exit 1
fi

# macOS caps GPU 'wired' memory at ~75% of RAM. Long 128K contexts approach that on a
# 64GB Mac and can OOM-crash via KV growth. Raise the limit to ~90% before serving.
# (Reverts on reboot.) ~57000 MB ≈ 90% of 64GB; scale to your machine.
if [[ "${RAISE_WIRED_LIMIT:-1}" == "1" ]]; then
  echo ">> Raising iogpu.wired_limit_mb (needs sudo; safe, resets on reboot)"
  sudo sysctl iogpu.wired_limit_mb=57000 || echo "   (skipped — set RAISE_WIRED_LIMIT=0 to silence)"
fi

echo ">> Serving $MODEL_DIR on :$PORT"
$PYTHON -m mlx_lm server --model "$MODEL_DIR" --port "$PORT"
