#!/usr/bin/env python3
"""Stage-2 corpus: download dedicated TERMINAL-task trajectories, normalize them to the
same OpenAI `messages` JSONL that `prepare_nebius_sft.py` emits, and curate.

Why a separate script: terminal datasets use heterogeneous schemas (ShareGPT
`conversations` of {from,value}, OpenAI `messages`, etc.), so each source needs a small
adapter. Output is identical in shape to Stage-1, so the two mix cleanly in Axolotl
(add both files as `datasets:` entries, or concatenate).

Primary source (verified 2026-06-27): Lite-Coder/LiteCoder-Terminal-SFT — 11,255 ShareGPT
trajectories, ~27.4 avg turns, validated on the same Qwen3-30B-A3B class (+7-11pp TB).
Add more sources via --sources; unknown schemas auto-detect (sharegpt vs openai) or use
--inspect first.

Curation (per the verified data-strategy finding): KEEP failure-recovery / imperfect
traces (do NOT default to success-only); drop trivially short runs; dedup. Anti-cheating
filtering (e.g. traces exploiting cached git/conda artifacts) is dataset-specific — see
the TODO hook; it is NOT applied generically here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# ShareGPT "from" tag / OpenAI role -> normalized OpenAI role.
ROLE_NORMALIZE = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
    "observation": "tool",
    "function": "tool",
}

# Known sources -> their conversation column + format. Unlisted sources auto-detect.
SOURCE_REGISTRY = {
    "Lite-Coder/LiteCoder-Terminal-SFT": {"field": "conversations", "format": "sharegpt"},
    "m-a-p/TerminalTraj": {"field": "messages", "format": "openai"},  # 20K on Hub (verified)
    # Add verified entries as you confirm their schema with --inspect, e.g.:
    #   "<termigen-hf-id>": {"field": "messages", "format": "openai", "resolved_field": "..."},
    # yoonholee/terminalbench-trajectories needs a custom adapter (steps + agent/reward filter).
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["Lite-Coder/LiteCoder-Terminal-SFT"],
        help="HF dataset ids to pull terminal trajectories from.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("data/sft_terminal.jsonl"))
    parser.add_argument("--append", action="store_true", help="Append to --output instead of overwriting (for mixing into a Stage-1 file).")
    parser.add_argument(
        "--inspect",
        metavar="DATASET_ID",
        help="Print the schema + first example of one dataset and exit (use for new sources).",
    )
    parser.add_argument(
        "--min-assistant-turns",
        type=int,
        default=2,
        help="Drop trajectories with fewer assistant turns (trivial/short runs). CLI-Gym "
        "dropped <20-step traces; raise this to be stricter.",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Keep only resolved trajectories (needs --resolved-field). DEFAULT OFF: the "
        "verified finding is to KEEP failure-recovery/imperfect traces.",
    )
    parser.add_argument(
        "--resolved-field",
        default=None,
        help="Boolean/int success column, if the source has one and --success-only is set.",
    )
    parser.add_argument("--max-examples-per-source", type=int, default=0, help="0 = no cap.")
    parser.add_argument("--max-chars-per-message", type=int, default=0, help="0 = keep full.")
    return parser.parse_args()


def load_dataset_or_exit(dataset_id: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Install deps first: pip install -r requirements-train.txt")
    return load_dataset(dataset_id, split=split)


def detect_field_and_format(columns: list[str], dataset_id: str) -> tuple[str, str]:
    """Return (conversation_field, format) for a source, from the registry or by sniffing."""
    if dataset_id in SOURCE_REGISTRY:
        cfg = SOURCE_REGISTRY[dataset_id]
        return cfg["field"], cfg["format"]
    for candidate in ("conversations", "messages", "trajectory", "conversation"):
        if candidate in columns:
            # sharegpt uses 'conversations' of {from,value}; others are usually OpenAI-shaped.
            return candidate, ("sharegpt" if candidate == "conversations" else "openai")
    sys.exit(
        f"Could not find a conversation column in {columns} for {dataset_id}. "
        f"Run with --inspect {dataset_id} and add it to SOURCE_REGISTRY."
    )


def normalize_turn(turn: dict, fmt: str):
    """Return (role, content, tool_calls, tool_call_id) from a ShareGPT or OpenAI turn."""
    if not isinstance(turn, dict):
        return None
    if fmt == "sharegpt" or ("from" in turn and "value" in turn):
        raw_role, content, tool_calls, tool_call_id = turn.get("from"), turn.get("value"), None, None
    else:
        raw_role = turn.get("role")
        content = turn.get("content")
        tool_calls = turn.get("tool_calls")
        tool_call_id = turn.get("tool_call_id")
    role = ROLE_NORMALIZE.get(raw_role)
    if role is None:
        return None
    return role, content, tool_calls, tool_call_id


def to_openai_messages(turns, fmt: str, max_chars: int):
    """Normalize a conversation to OpenAI messages, PRESERVING structured tool calls.

    Returns (messages, n_assistant_turns) or (None, 0) if no assistant turn.
    """
    messages = []
    assistant_turns = 0
    for turn in turns or []:
        parsed = normalize_turn(turn, fmt)
        if parsed is None:
            continue
        role, content, tool_calls, tool_call_id = parsed
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if max_chars and len(content) > max_chars:
            content = content[:max_chars] + "\n...[truncated]"
        msg = {"role": role, "content": content}
        if role == "assistant":
            assistant_turns += 1
            if tool_calls:  # keep structured calls so the chat template renders <tool_call>
                msg["tool_calls"] = tool_calls
        elif role == "tool" and tool_call_id:
            msg["tool_call_id"] = tool_call_id
        if not content and not msg.get("tool_calls"):
            continue
        messages.append(msg)
    return (messages if assistant_turns else None), assistant_turns


def inspect(dataset_id: str, split: str) -> None:
    dataset = load_dataset_or_exit(dataset_id, split)
    print("Dataset:", dataset_id)
    print("Columns:", dataset.column_names)
    print("Num rows:", len(dataset))
    example = dataset[0]
    for key, value in example.items():
        preview = str(value)
        print(f"  {key}: {preview[:300]}{'...' if len(preview) > 300 else ''}")


def main() -> None:
    args = parse_args()
    if args.inspect:
        inspect(args.inspect, args.split)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    total_written = 0
    per_source_counts: dict[str, int] = {}

    with args.output.open("a" if args.append else "w", encoding="utf-8") as out_file:
        for dataset_id in args.sources:
            dataset = load_dataset_or_exit(dataset_id, args.split)
            field, fmt = detect_field_and_format(dataset.column_names, dataset_id)
            written = skipped_short = skipped_unresolved = skipped_dup = 0

            for row in dataset:
                if args.success_only:
                    if not args.resolved_field:
                        sys.exit("--success-only needs --resolved-field")
                    if not bool(row.get(args.resolved_field)):
                        skipped_unresolved += 1
                        continue
                messages, assistant_turns = to_openai_messages(
                    row.get(field), fmt, args.max_chars_per_message
                )
                if messages is None or assistant_turns < args.min_assistant_turns:
                    skipped_short += 1
                    continue
                # TODO(anti-cheat): some terminal sets contain traces that exploit cached
                # git/conda artifacts or unintended shortcuts. Detection is dataset-specific
                # (CLI-Gym removed them by hand) — add per-source heuristics here before
                # trusting a new source for the final SFT stage.
                digest = hashlib.sha1(
                    json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
                ).hexdigest()
                if digest in seen_hashes:
                    skipped_dup += 1
                    continue
                seen_hashes.add(digest)
                out_file.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                written += 1
                if args.max_examples_per_source and written >= args.max_examples_per_source:
                    break

            per_source_counts[dataset_id] = written
            total_written += written
            print(
                f"{dataset_id}: wrote {written} "
                f"(skipped {skipped_short} short, {skipped_unresolved} unresolved, {skipped_dup} dup)"
            )

    print(f"\nTotal terminal trajectories -> {args.output}: {total_written}")
    print("Mix with Stage-1 by adding this file as a second Axolotl `datasets:` entry "
          "(or concatenate with data/sft_resolved.jsonl).")


if __name__ == "__main__":
    main()
