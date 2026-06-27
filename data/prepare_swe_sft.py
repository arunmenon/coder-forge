#!/usr/bin/env python3
"""Stage-1 SWE corpus: pull execution-verified agentic SWE trajectories from one or more
sources, normalize to OpenAI `messages` JSONL (PRESERVING tool calls), curate, and write.

Sources are selected with --sources; each known source has a verified schema in
SWE_SOURCE_REGISTRY (column for the message list, optional `resolved` filter, optional
HF config/split). Default = Nebius only (manageable size); opt into the doc's Recipe-A
mix by listing more, e.g.:

  python data/prepare_swe_sft.py --sources \\
    nebius/SWE-rebench-openhands-trajectories \\
    nvidia/SWE-Hero-openhands-trajectories \\
    nvidia/Open-SWE-Traces \\
    SWE-Gym/OpenHands-SFT-Trajectories \\
    --max-examples 12000

Schemas verified against the HF datasets-server on 2026-06-27. Not yet wired (need custom
adapters): yoonholee/terminalbench-trajectories (`steps` + agent/reward filter) and
nvidia/Nemotron-SFT-SWE-v2 (agentless format, not parquet-indexed).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROLE_NORMALIZE = {"system": "system", "user": "user", "assistant": "assistant",
                  "tool": "tool", "function": "tool"}

# Verified schemas. resolved_field=None => source is already success-only (no filter).
# config/split override the HF load; splits=None iterates every split in the config.
SWE_SOURCE_REGISTRY = {
    "nebius/SWE-rebench-openhands-trajectories": {"messages_field": "trajectory", "resolved_field": "resolved"},
    "nvidia/SWE-Hero-openhands-trajectories":    {"messages_field": "trajectory", "resolved_field": None},
    "nvidia/Open-SWE-Traces":                    {"messages_field": "trajectory", "resolved_field": "resolved", "config": "openhands"},
    "SWE-Gym/OpenHands-SFT-Trajectories":        {"messages_field": "messages", "resolved_field": None, "split": "train.success.oss"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", nargs="+", default=["nebius/SWE-rebench-openhands-trajectories"])
    p.add_argument("--output", type=Path, default=Path("data/sft_resolved.jsonl"))
    p.add_argument("--append", action="store_true")
    p.add_argument("--inspect", metavar="DATASET_ID", help="Print one source's schema and exit.")
    p.add_argument("--instance-field", default="instance_id")
    p.add_argument("--max-examples", type=int, default=0, help="Cap total written (0=off).")
    p.add_argument("--max-chars-per-message", type=int, default=0)
    p.add_argument("--min-steps", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0, help="0=off; doc suggests ~60.")
    p.add_argument("--max-per-instance", type=int, default=3, help="0=off.")
    p.add_argument("--decontaminate-against", nargs="*", default=["princeton-nlp/SWE-bench_Verified"])
    p.add_argument("--decontaminate-split", default="test")
    p.add_argument("--no-decontaminate", action="store_true")
    return p.parse_args()


def _load(dataset_id, **kw):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Install deps first: pip install -r requirements-train.txt")
    return load_dataset(dataset_id, **kw)


def load_eval_instance_ids(dataset_ids, split) -> set[str]:
    ids: set[str] = set()
    for dataset_id in dataset_ids:
        ds = _load(dataset_id, split=split)
        if "instance_id" in ds.column_names:
            ids.update(str(x) for x in ds["instance_id"])
    return ids


def iter_source_rows(source_id: str, cfg: dict):
    """Yield rows for a source, honoring its config and single-or-all-splits."""
    name, split = cfg.get("config"), cfg.get("split")
    if split:
        yield from _load(source_id, name=name, split=split)
    else:
        dataset_dict = _load(source_id, name=name)
        for split_name in dataset_dict:
            yield from dataset_dict[split_name]


def to_openai_messages(messages, max_chars):
    """Normalize to OpenAI chat format, preserving tool_calls / tool_call_id.
    Returns (messages, n_assistant) or (None, 0)."""
    out, n_assistant = [], 0
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = ROLE_NORMALIZE.get(m.get("role"))
        if role is None:
            continue
        content = m.get("content")
        content = "" if content is None else (content if isinstance(content, str)
                                              else json.dumps(content, ensure_ascii=False))
        if max_chars and len(content) > max_chars:
            content = content[:max_chars] + "\n...[truncated]"
        msg = {"role": role, "content": content}
        if role == "assistant":
            n_assistant += 1
            if m.get("tool_calls"):
                msg["tool_calls"] = m["tool_calls"]
        elif role == "tool" and m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        if not content and not msg.get("tool_calls"):
            continue
        out.append(msg)
    return (out, n_assistant) if n_assistant else (None, 0)


def main() -> None:
    args = parse_args()
    if args.inspect:
        ds = _load(args.inspect, name=SWE_SOURCE_REGISTRY.get(args.inspect, {}).get("config"))
        ds = ds[next(iter(ds))] if hasattr(ds, "keys") else ds
        print("Columns:", ds.column_names)
        for k, v in ds[0].items():
            print(f"  {k}: {str(v)[:300]}")
        return

    eval_ids: set[str] = set()
    if not args.no_decontaminate and args.decontaminate_against:
        eval_ids = load_eval_instance_ids(args.decontaminate_against, args.decontaminate_split)
        print(f"Decontaminating against {len(eval_ids)} eval instance_ids")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written, skipped, per_instance = 0, defaultdict(int), defaultdict(int)

    with args.output.open("a" if args.append else "w", encoding="utf-8") as out_file:
        for source_id in args.sources:
            cfg = SWE_SOURCE_REGISTRY.get(source_id)
            if cfg is None:
                sys.exit(f"Unknown source {source_id}. Add it to SWE_SOURCE_REGISTRY "
                         f"(verify schema with --inspect {source_id}).")
            msg_field, resolved_field = cfg["messages_field"], cfg.get("resolved_field")
            count = 0
            for row in iter_source_rows(source_id, cfg):
                if resolved_field and not bool(row.get(resolved_field)):
                    skipped["unresolved"] += 1
                    continue
                instance_id = str(row.get(args.instance_field) or "")
                if instance_id and instance_id in eval_ids:
                    skipped["contaminated"] += 1
                    continue
                if args.max_per_instance and instance_id and per_instance[instance_id] >= args.max_per_instance:
                    skipped["over_per_instance_cap"] += 1
                    continue
                messages = row.get(msg_field)
                if not isinstance(messages, list):
                    skipped["malformed"] += 1
                    continue
                normalized, n_assistant = to_openai_messages(messages, args.max_chars_per_message)
                if normalized is None or n_assistant < args.min_steps:
                    skipped["too_short"] += 1
                    continue
                if args.max_steps and n_assistant > args.max_steps:
                    skipped["too_long"] += 1
                    continue
                out_file.write(json.dumps({"messages": normalized}, ensure_ascii=False) + "\n")
                written += 1
                count += 1
                if instance_id:
                    per_instance[instance_id] += 1
                if args.max_examples and written >= args.max_examples:
                    break
            print(f"  {source_id}: wrote {count}")
            if args.max_examples and written >= args.max_examples:
                break

    print(f"Wrote {written} SWE trajectories -> {args.output}")
    print("Skipped:", dict(skipped))


if __name__ == "__main__":
    main()
