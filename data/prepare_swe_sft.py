#!/usr/bin/env python3
"""Stage-1 SWE corpus prep (thin wrapper). All logic lives in coder_forge.dataprep so SWE
and terminal share one normalizer + curator + source registry. Run:

  python3 data/prepare_swe_sft.py --sources nebius/... nvidia/SWE-Hero-... --max-per-source 5000

(or `forge-prep-sft-swe ...` once the package is installed). See docs/REFACTOR.md.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from coder_forge.dataprep.cli import run  # noqa: E402

if __name__ == "__main__":
    run("swe")
