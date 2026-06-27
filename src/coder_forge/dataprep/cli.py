"""Shared CLI for both prep stages: `forge-prep-sft swe|terminal` (or run the thin
data/prepare_*.py wrappers). One arg group => SWE and terminal share flag names."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import PrepOptions, run_prep, write_manifest
from .sources import DEFAULT_SOURCES, REGISTRY


def build_parser(kind: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Prepare the {kind} SFT corpus (shared engine).")
    p.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES[kind],
                   help=f"HF dataset ids (registry: {', '.join(s for s, v in REGISTRY.items() if v.kind == kind)})")
    p.add_argument("--output", type=Path,
                   default=Path(f"data/sft_{'resolved' if kind == 'swe' else 'terminal'}.jsonl"))
    p.add_argument("--append", action="store_true", help="Append (dedup seeded from existing file).")
    p.add_argument("--allow-unverified", action="store_true", help="Permit sources not in the registry.")
    # mixing (H3): per-source cap is the primary knob; --max-total is a global ceiling.
    p.add_argument("--max-per-source", type=int, default=0, help="Cap rows per source (0=off).")
    p.add_argument("--max-total", type=int, default=0, help="Global ceiling across sources (0=off).")
    # curation
    p.add_argument("--min-steps", type=int, default=1, help="Min assistant turns.")
    p.add_argument("--max-steps", type=int, default=0, help="Max assistant turns (0=off).")
    p.add_argument("--max-per-instance", type=int, default=3, help="Cap trajectories per instance (0=off).")
    p.add_argument("--max-chars-per-message", type=int, default=0)
    p.add_argument("--no-decontaminate", action="store_true")
    p.add_argument("--decontaminate-against", nargs="*", default=["princeton-nlp/SWE-bench_Verified"])
    p.add_argument("--decontaminate-split", default="test")
    p.add_argument("--require-instance-id", action="store_true",
                   help="Drop rows lacking instance_id when decontaminating (default: keep + count).")
    p.add_argument("--no-streaming", action="store_true", help="Disable streaming (download full splits).")
    p.add_argument("--no-manifest", action="store_true")
    p.add_argument("--inspect", metavar="DATASET_ID", help="Print a source's schema + first example, exit.")
    return p


def _inspect(dataset_id: str) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install deps first: pip install -r requirements-train.txt")
    spec = REGISTRY.get(dataset_id)
    loaded = load_dataset(dataset_id, name=(spec.config if spec else None), streaming=True)
    ds = loaded[next(iter(loaded.keys()))] if hasattr(loaded, "keys") else loaded
    first = next(iter(ds))
    print(f"Dataset: {dataset_id}  columns: {list(first.keys())}")
    for key, value in first.items():
        print(f"  {key}: {str(value)[:300]}")


def options_from_args(args) -> PrepOptions:
    return PrepOptions(
        max_per_source=args.max_per_source,
        max_total=args.max_total,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        max_per_instance=args.max_per_instance,
        max_chars=args.max_chars_per_message,
        decontaminate=not args.no_decontaminate,
        decontaminate_against=tuple(args.decontaminate_against),
        decontaminate_split=args.decontaminate_split,
        require_instance_id=args.require_instance_id,
        streaming=not args.no_streaming,
        allow_unverified=args.allow_unverified,
        append=args.append,
    )


def run(kind: str, argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser(kind).parse_args(argv)
    if args.inspect:
        _inspect(args.inspect)
        return
    opts = options_from_args(args)
    result = run_prep(args.sources, args.output, opts)
    print(f"Wrote {result.written} {kind} trajectories -> {args.output}")
    print(f"  per-source: {result.per_source}")
    print(f"  skipped: {result.skipped}  dropped_tool_calls: {result.dropped_tool_calls}")
    if not args.no_manifest:
        manifest = write_manifest(args.output, result, opts, [kind, *argv])
        print(f"  manifest: {manifest}")


def main_swe() -> None:
    run("swe")


def main_terminal() -> None:
    run("terminal")
