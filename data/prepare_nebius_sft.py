#!/usr/bin/env python3
"""Download Nebius OpenHands trajectories, keep the resolved ones, and emit an
SFT-ready chat JSONL for Axolotl.

Dataset: https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories
  ~67K execution-grounded OpenHands traces (teacher = Qwen3-Coder-480B), of which
  ~32K are `resolved` (the agent's patch passed the repo's tests). We train only
  on the resolved trajectories so the model imitates *successful* agentic behavior.

Output: one JSON object per line, OpenAI chat format:
    {"messages": [{"role": "system"|"user"|"assistant"|"tool", "content": "..."}, ...]}
which Axolotl reads via `type: chat_template` + `field_messages: messages`
(see config/qwen3_coder_30b_qlora.yaml). Keeping native roles lets the model's
own chat template render the tool-call format it must emit at inference time.

The exact column names in the source dataset can change; run with --inspect first
to print the schema, then set --resolved-field / --messages-field accordingly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Roles we keep, normalized to OpenAI chat roles. OpenHands "function" -> "tool".
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
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print the dataset schema + one example and exit (use this first).",
    )
    parser.add_argument(
        "--resolved-field",
        default="resolved",
        help="int64 column (1=resolved/test-passing, 0=not). Verified against the dataset viewer.",
    )
    parser.add_argument(
        "--messages-field",
        default="trajectory",
        help="Column holding the OpenHands message list (List[struct] with "
        "role/content/name/tool_call_id/tool_calls). Verified against the dataset viewer.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Cap the number of resolved trajectories written (0 = no cap). "
        "Research shows meaningful gains from as few as ~500; start small.",
    )
    parser.add_argument(
        "--max-chars-per-message",
        type=int,
        default=0,
        help="Optionally truncate very long observations (0 = keep full). Long "
        "tool outputs dominate token cost; trimming trades fidelity for cheaper SFT.",
    )
    return parser.parse_args()


def load_dataset_or_exit(dataset_id: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Install deps first: pip install -r requirements-train.txt")
    return load_dataset(dataset_id, split=split)


def to_openai_messages(messages: list[dict], max_chars_per_message: int) -> list[dict] | None:
    """Normalize one trajectory's message list to OpenAI chat format.

    Returns None if the trajectory has no assistant turn to learn from.
    """
    normalized = []
    has_assistant_turn = False
    for message in messages:
        role = ROLE_NORMALIZE.get(message.get("role"))
        if role is None:
            continue
        content = message.get("content")
        # Tool/function calls may live in a structured field; serialize them so the
        # model learns the exact tool-call format it must emit at inference time.
        if not content and message.get("tool_calls"):
            content = json.dumps(message["tool_calls"], ensure_ascii=False)
        if content is None:
            continue
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if max_chars_per_message and len(content) > max_chars_per_message:
            content = content[:max_chars_per_message] + "\n...[truncated]"
        if role == "assistant":
            has_assistant_turn = True
        normalized.append({"role": role, "content": content})
    return normalized if has_assistant_turn else None


def main() -> None:
    args = parse_args()
    dataset = load_dataset_or_exit(args.dataset_id, args.split)

    if args.inspect:
        print("Columns:", dataset.column_names)
        print("Num rows:", len(dataset))
        print("\nFirst example (truncated):")
        example = dataset[0]
        for key, value in example.items():
            preview = str(value)
            print(f"  {key}: {preview[:300]}{'...' if len(preview) > 300 else ''}")
        return

    if args.resolved_field not in dataset.column_names:
        sys.exit(
            f"--resolved-field '{args.resolved_field}' not in columns "
            f"{dataset.column_names}. Run with --inspect and set it correctly."
        )
    if args.messages_field not in dataset.column_names:
        sys.exit(
            f"--messages-field '{args.messages_field}' not in columns "
            f"{dataset.column_names}. Run with --inspect and set it correctly."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_unresolved = 0
    skipped_empty = 0

    with args.output.open("w", encoding="utf-8") as out_file:
        for row in dataset:
            if not bool(row.get(args.resolved_field)):
                skipped_unresolved += 1
                continue
            messages = row.get(args.messages_field)
            if not isinstance(messages, list):
                skipped_empty += 1
                continue
            normalized = to_openai_messages(messages, args.max_chars_per_message)
            if normalized is None:
                skipped_empty += 1
                continue
            out_file.write(json.dumps({"messages": normalized}, ensure_ascii=False) + "\n")
            written += 1
            if args.max_examples and written >= args.max_examples:
                break

    print(f"Wrote {written} resolved trajectories -> {args.output}")
    print(f"Skipped: {skipped_unresolved} unresolved, {skipped_empty} empty/malformed")


if __name__ == "__main__":
    main()
