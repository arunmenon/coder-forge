#!/usr/bin/env bash
# Phase 2 — QLoRA SFT cold-start on a single RunPod GPU (H100 80GB / RTX PRO 6000 96GB).
# Assumes data/sft_resolved.jsonl exists (run `make data` first) and Axolotl is installed
# (requirements-train.txt). Produces a LoRA adapter, then merges to full bf16 weights.
set -euo pipefail

CONFIG="${CONFIG:-config/qwen3_coder_30b_qlora.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3-coder-30b-a3b-qlora}"
MERGED_DIR="${MERGED_DIR:-outputs/qwen3-coder-30b-a3b-merged}"

if [[ ! -f data/sft_resolved.jsonl ]]; then
  echo "ERROR: data/sft_resolved.jsonl missing. Run: make data" >&2
  exit 1
fi

echo ">> GPU check"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

echo ">> Training QLoRA: $CONFIG"
# accelerate launch routes through Axolotl's CLI; single GPU needs no FSDP/DeepSpeed.
accelerate launch -m axolotl.cli.train "$CONFIG"

echo ">> Merging LoRA adapter -> full bf16 weights for serving/quantization"
python -m axolotl.cli.merge_lora "$CONFIG" --lora_model_dir "$OUTPUT_DIR" --output_dir "$MERGED_DIR"

echo ">> Done. Merged model at: $MERGED_DIR"
echo "   Next: serve it (serve/serve_vllm.sh $MERGED_DIR) and re-run eval (make eval),"
echo "   or quantize for the Mac (quantize/to_mlx.sh $MERGED_DIR)."
