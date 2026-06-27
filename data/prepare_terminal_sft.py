#!/usr/bin/env python3
"""Stage-2 terminal corpus prep (thin wrapper). Logic lives in coder_forge.dataprep
(shared with the SWE prep). Run:

  python3 data/prepare_terminal_sft.py --sources Lite-Coder/LiteCoder-Terminal-SFT m-a-p/TerminalTraj

(or `forge-prep-sft-terminal ...` once the package is installed). See docs/REFACTOR.md.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from coder_forge.dataprep.cli import run  # noqa: E402

if __name__ == "__main__":
    run("terminal")
