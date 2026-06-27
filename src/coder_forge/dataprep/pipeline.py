"""Curation + multi-source mixing + IO + manifest — the shared prep engine used by both
the SWE and terminal CLIs.

Fixes (docs/REFACTOR.md §2): H3 per-source caps (no global source-order cap), H4 training-split
only, H5 decontam never silently no-ops, M7 dedup + append-seeding, plus a reproducibility
manifest (§5). The per-row normalization (H1/H2/M6) lives in normalize.py.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .codecs import apply_codec
from .normalize import to_openai_messages
from .sources import is_training_split, resolve


@dataclass
class PrepOptions:
    max_per_source: int = 0
    max_total: int = 0
    min_steps: int = 1
    max_steps: int = 0
    max_per_instance: int = 3
    max_chars: int = 0
    decontaminate: bool = True
    decontaminate_against: tuple = ("princeton-nlp/SWE-bench_Verified",)
    decontaminate_split: str = "test"
    require_instance_id: bool = False
    streaming: bool = True
    allow_unverified: bool = False
    append: bool = False
    codec: str = "qwen3_xml_dict"
    allow_empty_source: bool = False


@dataclass
class PrepResult:
    written: int
    per_source: dict
    skipped: dict
    dropped_tool_calls: int
    source_ids: list
    missing_instance_id_kept: int = 0
    decontam_id_count: int = 0


def _load(dataset_id, config=None, streaming=True):
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install deps first: pip install -r requirements-train.txt")
    return load_dataset(dataset_id, name=config, streaming=streaming)


def load_eval_instance_ids(dataset_ids, split) -> set:
    """Load the eval set's instance_ids (non-streaming; it's small) for decontamination."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install deps first: pip install -r requirements-train.txt")
    ids: set = set()
    for dataset_id in dataset_ids:
        ds = load_dataset(dataset_id, split=split)
        if "instance_id" in ds.column_names:
            ids.update(str(x) for x in ds["instance_id"])
    return ids


def iter_source_rows(spec, streaming=True):
    """Yield (split_name, dataset) for TRAINING splits only (H4)."""
    loaded = _load(spec.dataset_id, spec.config, streaming)
    if hasattr(loaded, "keys"):  # (Iterable)DatasetDict
        for split_name in loaded.keys():
            if is_training_split(split_name, spec):
                yield split_name, loaded[split_name]
    else:
        yield "train", loaded


def content_hash(messages) -> str:
    return hashlib.sha1(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def seed_state(output: Path):
    """M7: rebuild the dedup set from an existing file before appending."""
    seen: set = set()
    if output.exists():
        with output.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    seen.add(content_hash(json.loads(line)["messages"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return seen, Counter()


def curate_row(row, spec, opts, eval_ids, seen, per_instance):
    """Apply the full curation pipeline to one row.
    Returns (keep, messages|None, skip_reason|None, dropped_tool_calls, undecontaminated)."""
    if spec.resolved_field and not bool(row.get(spec.resolved_field)):
        return False, None, "unresolved", 0, False

    raw_id = row.get(spec.instance_field)
    instance_id = str(raw_id) if raw_id not in (None, "") else None
    undecontaminated = False

    if opts.decontaminate and eval_ids:
        if instance_id is None:
            if opts.require_instance_id:
                return False, None, "no_instance_id", 0, False
            undecontaminated = True  # H5: kept but un-decontaminable — surfaced + counted by caller
        elif instance_id in eval_ids:
            return False, None, "contaminated", 0, False

    if opts.max_per_instance and instance_id and per_instance[instance_id] >= opts.max_per_instance:
        return False, None, "over_per_instance_cap", 0, False

    messages = row.get(spec.messages_field)
    if not isinstance(messages, list):
        return False, None, "malformed", 0, False

    norm, n_assistant, dropped = to_openai_messages(messages, spec.fmt, opts.max_chars)
    if norm is None:
        return False, None, "no_assistant", dropped, False
    if n_assistant < opts.min_steps:
        return False, None, "too_short", dropped, False
    if opts.max_steps and n_assistant > opts.max_steps:
        return False, None, "too_long", dropped, False

    norm = apply_codec(norm, opts.codec)  # family-specific tool-call shaping
    h = content_hash(norm)
    if h in seen:
        return False, None, "dup", dropped, False
    seen.add(h)
    if instance_id:
        per_instance[instance_id] += 1
    return True, norm, None, dropped, undecontaminated


def run_prep(source_ids, output: Path, opts: PrepOptions) -> PrepResult:
    specs = [resolve(s, allow_unverified=opts.allow_unverified) for s in source_ids]
    eval_ids = (load_eval_instance_ids(opts.decontaminate_against, opts.decontaminate_split)
                if opts.decontaminate else set())
    if eval_ids:
        print(f"Decontaminating against {len(eval_ids)} eval instance_ids", file=sys.stderr)
    if opts.append and opts.max_per_instance:
        print("WARNING: --append cannot re-tally per-instance counts from the existing file; "
              "--max-per-instance only applies within this run.", file=sys.stderr)

    output.parent.mkdir(parents=True, exist_ok=True)
    seen, per_instance = seed_state(output) if opts.append else (set(), Counter())
    written = dropped_total = missing_id_kept = 0
    per_source: Counter = Counter()
    skipped: Counter = Counter()

    def rows_of(spec):
        for _split, ds in iter_source_rows(spec, opts.streaming):
            for row in ds:
                yield row

    # H3: ROUND-ROBIN across sources (one raw row per source per pass) so --max-total and the
    # mix don't skew to whichever source is listed first. Per-source caps + exhaustion drop a
    # source from the rotation.
    rotation = [(spec, rows_of(spec)) for spec in specs]
    with output.open("a" if opts.append else "w", encoding="utf-8") as fh:
        while rotation:
            survivors = []
            progressed = False
            for spec, it in rotation:
                if opts.max_per_source and per_source[spec.dataset_id] >= opts.max_per_source:
                    continue  # this source has met its quota
                try:
                    row = next(it)
                except StopIteration:
                    continue  # exhausted
                progressed = True
                survivors.append((spec, it))
                keep, payload, reason, dropped, undecon = curate_row(
                    row, spec, opts, eval_ids, seen, per_instance)
                dropped_total += dropped
                if keep:
                    fh.write(json.dumps({"messages": payload}, ensure_ascii=False) + "\n")
                    written += 1
                    per_source[spec.dataset_id] += 1
                    if undecon:
                        missing_id_kept += 1
                else:
                    skipped[reason] += 1
                if opts.max_total and written >= opts.max_total:
                    survivors, progressed = [], False
                    break
            rotation = survivors
            if not progressed:
                break

    for spec in specs:
        per_source.setdefault(spec.dataset_id, 0)
    empty = [s.dataset_id for s in specs if per_source[s.dataset_id] == 0]
    if empty and not opts.allow_empty_source:
        raise SystemExit(
            f"ERROR: these requested sources contributed 0 rows: {empty}. "
            f"Check filters/caps/availability, or pass --allow-empty-source. "
            f"(skipped reasons: {dict(skipped)})")

    return PrepResult(written, dict(per_source), dict(skipped), dropped_total,
                      [s.dataset_id for s in specs], missing_id_kept, len(eval_ids))


def write_manifest(output: Path, result: PrepResult, opts: PrepOptions, argv) -> Path:
    """§5 reproducibility: JSON sidecar capturing what produced this corpus."""
    output = Path(output)
    sha256 = hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(output.parent or "."), text=True).strip()
    except Exception:
        git_sha = None
    manifest = {
        "output": str(output),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": git_sha,
        "argv": list(argv),
        "sources": result.source_ids,
        "per_source_written": result.per_source,
        "total_written": result.written,
        "skipped": result.skipped,
        "dropped_tool_calls": result.dropped_tool_calls,
        "decontaminate": opts.decontaminate,
        "decontam_id_count": result.decontam_id_count,
        "missing_instance_id_kept": result.missing_instance_id_kept,
        "codec": opts.codec,
        "sha256": sha256,
    }
    path = Path(str(output) + ".manifest.json")
    path.write_text(json.dumps(manifest, indent=2))
    return path
