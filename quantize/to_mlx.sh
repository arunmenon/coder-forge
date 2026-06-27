#!/usr/bin/env bash
# Phase 4 — convert a (merged) HF model to MLX 4-bit for the Mac.
# Run this ON THE MAC (MLX is Apple-Silicon only). Pull the merged weights from the
# RunPod pod / HF Hub first, then quantize.
#
# IMPORTANT quant-quality note: naive RTN 4-bit drops *multi-step agentic* accuracy
# ~10-15% (ACBench) even though raw tool-call generation barely moves. mlx_lm.convert
# uses group-wise quant (better than RTN); for best fidelity consider an AWQ/DWQ recipe.
# On a 96GB+ Mac, prefer 5- or 6-bit (--q-bits 6) for tool-call/JSON robustness — it
# buys format reliability, not raw coding accuracy.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Capture explicit user overrides before pulling profile defaults (user env wins over profile).
_user_out="${OUT_DIR:-}"; _user_bits="${Q_BITS:-}"; _user_group="${Q_GROUP:-}"
if [[ -n "${FAMILY:-}" ]]; then
  eval "$(PYTHON="$PYTHON" bash "$HERE/scripts/family.sh" export "$FAMILY")"
fi
SOURCE="${1:-${MERGED_DIR:-outputs/qwen3-coder-30b-a3b-merged}}"   # HF dir or hub id
OUT_DIR="${_user_out:-${MLX_DIR:-models/qwen3-coder-30b-a3b-mlx-4bit}}"
Q_BITS="${_user_bits:-${Q_BITS:-4}}"
Q_GROUP="${_user_group:-${Q_GROUP_SIZE:-64}}"

if ! $PYTHON -c "import mlx_lm" 2>/dev/null; then
  echo "Install MLX: pip install mlx-lm" >&2
  exit 1
fi

echo ">> Quantizing $SOURCE -> $OUT_DIR (${Q_BITS}-bit, group ${Q_GROUP})"
$PYTHON -m mlx_lm convert \
  --hf-path "$SOURCE" \
  --mlx-path "$OUT_DIR" \
  --quantize \
  --q-bits "$Q_BITS" \
  --q-group-size "$Q_GROUP"

echo ">> Done. Serve with: serve/serve_mlx_mac.sh $OUT_DIR"
echo "   Expected on 64GB M4 Max: ~18-20GB weights, full 128K ctx fits, ~90-130 tok/s decode."
