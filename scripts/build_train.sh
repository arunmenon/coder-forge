#!/usr/bin/env bash
# Build the canonical training file from existing stage files, with cross-stage dedup +
# a manifest. Errors (never reads stdin) when no stage files exist.
set -euo pipefail
PYTHON="${PYTHON:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-data/sft_train.jsonl}"
shift || true

STAGES=("$@")
if [[ ${#STAGES[@]} -eq 0 ]]; then
  for f in data/sft_resolved.jsonl data/sft_terminal.jsonl; do
    [[ -f "$f" ]] && STAGES+=("$f")
  done
fi
if [[ ${#STAGES[@]} -eq 0 ]]; then
  echo "ERROR: no stage files found — run 'make data' (and/or 'make data-terminal') first." >&2
  exit 1
fi

PYTHONPATH="$HERE/src" "$PYTHON" -m coder_forge.dataprep.merge "$OUT" "${STAGES[@]}"
