#!/usr/bin/env python3
"""Download Nebius OpenHands trajectories, keep the resolved ones, curate, and emit an
SFT-ready chat JSONL for Axolotl.

Dataset: https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories
  ~67K execution-grounded OpenHands traces (teacher = Qwen3-Coder-480B), of which
  ~32K are `resolved` (the agent's patch passed the repo's tests).

Output: one JSON object per line, OpenAI chat format, PRESERVING structured tool calls:
    {"messages": [
       {"role": "assistant", "content": "...", "tool_calls": [{...}]},
       {"role": "tool", "content": "...", "tool_call_id": "..."}, ...]}
Axolotl reads this via `type: chat_template` + `field_messages: messages`; the model's
own chat template then renders the `<tool_call>…</tool_call>` / `<tool_response>` spans
(see swe-terminal-trajectory-datasets.md §8.1). We deliberately KEEP the structured
`tool_calls` field — dropping it would strip the core agent behavior we train on.

Curation implemented here (see RECIPES.md): filter resolved, min/max assistant steps,
cap trajectories per instance, and decontaminate instance_ids against the eval set(s).
NOT implemented (honest): anti-cheat/shortcut detection (dataset-specific) and exact
files/lines-edited bounds — left as flags/TODO, not silently claimed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROLE_NORMALIZE = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
    "function": "tool",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="nebius/SWE-rebench-openhands-trajectories")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("data/sft_resolved.jsonl"))
    parser.add_argument("--inspect", action="store_true", help="Print schema + one example, exit.")
    parser.add_argument("--resolved-field", default="resolved", help="int64 1/0 success column.")
    parser.add_argument("--messages-field", default="trajectory", help="Message-list column.")
    parser.add_argument("--instance-field", default="instance_id", help="Task-id column.")
    parser.add_argument("--max-examples", type=int, default=0, help="Cap total written (0=off).")
    parser.add_argument("--max-chars-per-message", type=int, default=0, help="Truncate obs (0=off).")
    # --- curation ---
    parser.add_argument("--min-steps", type=int, default=1, help="Drop traj with < N assistant turns.")
    parser.add_argument("--max-steps", type=int, default=0, help="Drop traj with > N assistant turns (0=off; doc suggests ~60).")
    parser.add_argument("--max-per-instance", type=int, default=3, help="Cap trajectories per task instance (0=off).")
    parser.add_argument(
        "--decontaminate-against",
        nargs="*",
        default=["princeton-nlp/SWE-bench_Verified"],
        help="HF dataset ids whose instance_ids are EXCLUDED (eval-set leakage guard).",
    )
    parser.add_argument("--decontaminate-split", default="test")
    parser.add_argument("--no-decontaminate", action="store_true", help="Skip eval-id exclusion.")
    return parser.parse_args()


def load_dataset_or_exit(dataset_id: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Install deps first: pip install -r requirements-train.txt")
    return load_dataset(dataset_id, split=split)


def load_eval_instance_ids(dataset_ids: list[str], split: str) -> set[str]:
    ids: set[str] = set()
    for dataset_id in dataset_ids:
        ds = load_dataset_or_exit(dataset_id, split)
        if "instance_id" in ds.column_names:
            ids.update(str(x) for x in ds["instance_id"])
    return ids


def to_openai_messages(messages: list[dict], max_chars: int):
    """Normalize to OpenAI chat format, PRESERVING structured tool_calls / tool_call_id.

    Returns (messages, n_assistant_turns) or (None, 0) if there is no assistant turn.
    """
    normalized = []
    n_assistant = 0
    for message in messages:
        role = ROLE_NORMALIZE.get(message.get("role"))
        if role is None:
            continue
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if max_chars and len(content) > max_chars:
            content = content[:max_chars] + "\n...[truncated]"
        out = {"role": role, "content": content}
        tool_calls = message.get("tool_calls")
        if role == "assistant":
            n_assistant += 1
            if tool_calls:
                # Keep the structured call so the chat template renders <tool_call>…</tool_call>.
                out["tool_calls"] = tool_calls
        elif role == "tool" and message.get("tool_call_id"):
            out["tool_call_id"] = message["tool_call_id"]
        # Drop only turns that carry neither text nor a tool call.
        if not content and not out.get("tool_calls"):
            continue
        normalized.append(out)
    return (normalized, n_assistant) if n_assistant else (None, 0)


def main() -> None:
    args = parse_args()
    dataset = load_dataset_or_exit(args.dataset_id, args.split)

    if args.inspect:
        print("Columns:", dataset.column_names, "\nNum rows:", len(dataset))
        for key, value in dataset[0].items():
            preview = str(value)
            print(f"  {key}: {preview[:300]}{'...' if len(preview) > 300 else ''}")
        return

    for field in (args.resolved_field, args.messages_field):
        if field not in dataset.column_names:
            sys.exit(f"'{field}' not in columns {dataset.column_names}. Run --inspect.")

    eval_ids: set[str] = set()
    if not args.no_decontaminate and args.decontaminate_against:
        eval_ids = load_eval_instance_ids(args.decontaminate_against, args.decontaminate_split)
        print(f"Decontaminating against {len(eval_ids)} eval instance_ids "
              f"from {args.decontaminate_against}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = defaultdict(int)
    per_instance = defaultdict(int)

    with args.output.open("w", encoding="utf-8") as out_file:
        for row in dataset:
            if not bool(row.get(args.resolved_field)):
                skipped["unresolved"] += 1
                continue
            instance_id = str(row.get(args.instance_field) or "")
            if instance_id and instance_id in eval_ids:
                skipped["contaminated"] += 1
                continue
            if args.max_per_instance and per_instance[instance_id] >= args.max_per_instance:
                skipped["over_per_instance_cap"] += 1
                continue
            messages = row.get(args.messages_field)
            if not isinstance(messages, list):
                skipped["malformed"] += 1
                continue
            normalized, n_assistant = to_openai_messages(messages, args.max_chars_per_message)
            if normalized is None:
                skipped["no_assistant"] += 1
                continue
            if n_assistant < args.min_steps:
                skipped["too_short"] += 1
                continue
            if args.max_steps and n_assistant > args.max_steps:
                skipped["too_long"] += 1
                continue
            out_file.write(json.dumps({"messages": normalized}, ensure_ascii=False) + "\n")
            written += 1
            per_instance[instance_id] += 1
            if args.max_examples and written >= args.max_examples:
                break

    print(f"Wrote {written} trajectories -> {args.output}")
    print("Skipped:", dict(skipped))


if __name__ == "__main__":
    main()
