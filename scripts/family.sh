#!/usr/bin/env bash
# POSIX shim around the family-profile resolver, for Make + bash consumers.
#   scripts/family.sh get <family> <dotted.key>   -> one value
#   scripts/family.sh export <family>             -> KEY=val lines (for `eval`)
#   scripts/family.sh list | doctor <family>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
exec env PYTHONPATH="$HERE/src" "$PYTHON" -m coder_forge.family "$@"
