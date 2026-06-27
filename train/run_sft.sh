#!/usr/bin/env bash
# Phase 2 — QLoRA SFT cold-start on a single RunPod GPU (H100 80GB / RTX PRO 6000 96GB).
# Trains on data/sft_train.jsonl (built by `make data` [+ `make data-terminal`]) with the
# config given by CONFIG (set per base family by the Makefile BASE switch). Produces a LoRA
# adapter, then merges to full bf16 weights.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-config/qwen3_coder_30b_qlora.yaml}"
# Derive the adapter dir from the config's own output_dir (real YAML parse — M9; a grep
# miss must NOT fall back to a path that mismatches what Axolotl writes). Override via OUTPUT_DIR=.
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR=$($PYTHON -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "$CONFIG") \
    || { echo "ERROR: could not parse output_dir from $CONFIG" >&2; exit 1; }
fi
[[ -n "$OUTPUT_DIR" ]] || { echo "ERROR: empty output_dir in $CONFIG" >&2; exit 1; }
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR%-qlora}-merged}"

if [[ ! -f data/sft_train.jsonl ]]; then
  echo "ERROR: data/sft_train.jsonl missing. Run: make data  (and optionally: make data-terminal)" >&2
  exit 1
fi

echo ">> GPU check"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

echo ">> Training QLoRA: $CONFIG  (adapter -> $OUTPUT_DIR)"
# accelerate launch routes through Axolotl's CLI; single GPU needs no FSDP/DeepSpeed.
accelerate launch -m axolotl.cli.train "$CONFIG"

echo ">> Merging LoRA adapter -> full bf16 weights ($MERGED_DIR)"
$PYTHON -m axolotl.cli.merge_lora "$CONFIG" --lora_model_dir "$OUTPUT_DIR" --output_dir "$MERGED_DIR"

echo ">> Done. Merged model at: $MERGED_DIR"
echo "   Next: serve it (serve/serve_vllm.sh $MERGED_DIR) and re-run eval (make eval),"
echo "   or quantize for the Mac (quantize/to_mlx.sh $MERGED_DIR)."
