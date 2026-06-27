"""Merge stage JSONL files into the canonical training file with CROSS-STAGE content dedup
and a reproducibility manifest. Replaces the Makefile's `cat $(ls …)` (M10 + Phase 1)."""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from .pipeline import content_hash


def merge(output: Path, stages: list[Path]) -> int:
    seen: set = set()
    written = 0
    per_stage: dict = {}
    with output.open("w", encoding="utf-8") as out_fh:
        for stage in stages:
            count = 0
            with stage.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    h = content_hash(obj.get("messages", obj))
                    if h in seen:
                        continue
                    seen.add(h)
                    out_fh.write(line + "\n")
                    written += 1
                    count += 1
            per_stage[str(stage)] = count

    sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_sha = None
    Path(str(output) + ".manifest.json").write_text(json.dumps({
        "output": str(output),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": git_sha,
        "stages": per_stage,
        "total_written": written,
        "sha256": sha256,
    }, indent=2))
    return written


def main(argv) -> None:
    if len(argv) < 2:
        raise SystemExit("usage: merge <output.jsonl> <stage1.jsonl> [stage2.jsonl ...]")
    output = Path(argv[0])
    stages = [Path(p) for p in argv[1:] if Path(p).is_file()]
    if not stages:
        raise SystemExit("ERROR: no stage files exist to merge (run `make data` first).")
    written = merge(output, stages)
    print(f">> built {output} ({written} rows after cross-stage dedup) + {output}.manifest.json")


if __name__ == "__main__":
    main(sys.argv[1:])
