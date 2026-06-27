#!/usr/bin/env bash
# One-command setup on a fresh RunPod GPU pod (H100 80GB SXM or RTX PRO 6000 96GB).
# Installs deps, prepares data, and (by default) runs a SMOKE TEST before the full SFT
# so you don't burn GPU hours on a config bug.
#
#   SMOKE_TEST=1 bash train/runpod_bootstrap.sh   # default: install + 10-step smoke run
#   SMOKE_TEST=0 bash train/runpod_bootstrap.sh   # install + full 3-epoch SFT
#
# Put the model + checkpoints on a PERSISTENT VOLUME (e.g. /workspace) or you lose them
# when the pod stops. Run this from the finetune/ dir on the volume.
set -euo pipefail

SMOKE_TEST="${SMOKE_TEST:-1}"
CONFIG="${CONFIG:-config/qwen3_coder_30b_qlora.yaml}"

echo ">> [1/4] GPU + deps"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
pip install -q -U pip
pip install -q -r requirements-train.txt
[[ -n "${HF_TOKEN:-}" ]] && huggingface-cli login --token "$HF_TOKEN" || \
  echo "   (set HF_TOKEN to auto-login for dataset/model access)"

echo ">> [2/4] Data"
if [[ ! -f data/sft_resolved.jsonl ]]; then
  # Skip if you rsync'd the jsonl up from your machine instead.
  python data/prepare_nebius_sft.py --output data/sft_resolved.jsonl \
    ${MAX_EXAMPLES:+--max-examples "$MAX_EXAMPLES"}
fi
TOTAL=$(wc -l < data/sft_resolved.jsonl)
echo "   $TOTAL resolved trajectories available"

if [[ "$SMOKE_TEST" == "1" ]]; then
  echo ">> [3/4] SMOKE TEST — 20 examples, 10 steps (validates the full train->merge path)"
  head -n 20 data/sft_resolved.jsonl > data/sft_smoke.jsonl
  # Override dataset + cap steps via Axolotl CLI; cheap end-to-end check.
  accelerate launch -m axolotl.cli.train "$CONFIG" \
    --datasets.0.path data/sft_smoke.jsonl \
    --max_steps 10 \
    --output_dir outputs/_smoke \
    --save_steps 10 --val_set_size 0
  echo ">> [4/4] Smoke test passed. Re-run with SMOKE_TEST=0 for the full SFT."
else
  echo ">> [3/4] FULL SFT"
  bash train/run_sft.sh
  echo ">> [4/4] Done — merged weights ready to serve (make eval) or quantize (to_mlx.sh)."
fi
