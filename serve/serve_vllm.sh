#!/usr/bin/env bash
# Serve a model behind an OpenAI-compatible endpoint with Qwen agentic tool-calling.
# Use for: Phase 0 baseline (pass the HF base id) and Phase 3 re-eval (pass the merged dir).
# Runs on the RunPod GPU pod. The endpoint is what OpenHands points at.
#
#   serve/serve_vllm.sh                                   # serves the base model
#   serve/serve_vllm.sh outputs/qwen3-coder-30b-a3b-merged  # serves your fine-tune
set -euo pipefail

MODEL="${1:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
PORT="${PORT:-8000}"
# Native context is 256K; agentic eval rarely needs >32K and long ctx costs KV memory.
MAX_LEN="${MAX_LEN:-32768}"
# bf16 30B (~61GB) fits one 80GB GPU. On a 48GB card, add: --quantization bitsandbytes
GPU_UTIL="${GPU_UTIL:-0.92}"

echo ">> Serving $MODEL on :$PORT (max_model_len=$MAX_LEN)"
# --tool-call-parser qwen3_xml is vLLM's current parser for Qwen3-Coder (streaming-
# capable; Qwen3-Coder-30B-A3B-Instruct is listed under qwen3_xml in vLLM's tool_calling
# docs). The older qwen3_coder parser is non-streaming and can emit broken tool output.
# `hermes` is for plain Qwen3 (e.g. Qwen3-8B), NOT Qwen3-Coder. No --reasoning-parser
# needed (Instruct model); the tokenizer ships a tool-compatible chat template.
vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
