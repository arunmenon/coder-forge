"""Trajectory normalization — ONE place for role mapping, tool-call canonicalization,
content flattening, and the empty-turn guard. Imported by both SWE and terminal prep so
every correctness fix lands exactly once.

Fixes (see docs/REFACTOR.md §2):
  H1 — tool_call.function.arguments string -> dict (Qwen template iterates it as a mapping)
  H2 — never emit an empty assistant turn; keep paired `tool` results (placeholder if empty)
  M6 — flatten structured assistant content to text (don't json.dumps content blocks)
"""

from __future__ import annotations

import json

# Superset role map (ShareGPT from-tags + OpenAI roles + OpenHands variants).
ROLE_NORMALIZE = {
    "system": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "ai": "assistant",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
}

_TOOL_PLACEHOLDER = "(no output)"


def flatten_content(content) -> str:
    """M6: collapse content to text. A list of OpenAI content blocks becomes its joined
    text (non-text blocks described, not json.dumps'd into the training target)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif block.get("type"):
                    parts.append(f"[{block['type']}]")
        return "\n".join(p for p in parts if p)
    # dict or other: last-resort serialize (acceptable for non-assistant observations).
    return json.dumps(content, ensure_ascii=False)


def canonicalize_tool_calls(tool_calls):
    """H1: ensure every tool_call.function.arguments is a dict. Returns (clean_list|None, n_dropped).
    json.loads string args; drop a call whose args are malformed JSON; leave dict args untouched."""
    if not tool_calls or not isinstance(tool_calls, list):
        return None, 0
    out, dropped = [], 0
    for tc in tool_calls:
        if not isinstance(tc, dict):
            dropped += 1
            continue
        fn = tc.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
            raw = fn["arguments"].strip()
            try:
                parsed = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, ValueError):
                dropped += 1
                continue
            tc = {**tc, "function": {**fn, "arguments": parsed}}
        out.append(tc)
    return (out or None), dropped


def normalize_turn(turn, fmt: str = "openai"):
    """Return (role, raw_content, tool_calls, tool_call_id) or None for an unknown role."""
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


def to_openai_messages(turns, fmt: str = "openai", max_chars: int = 0):
    """Normalize a trajectory to OpenAI messages.

    Returns (messages, n_assistant, dropped_tool_calls), or (None, 0, dropped) when there is
    no trainable assistant turn. Guarantees: messages is non-empty and contains >=1 assistant
    turn with text or a (canonicalized) tool call; every kept tool turn has a tool_call_id and
    non-empty content.
    """
    out, n_assistant, dropped_calls = [], 0, 0
    for turn in turns or []:
        parsed = normalize_turn(turn, fmt)
        if parsed is None:
            continue
        role, content, tool_calls, tool_call_id = parsed
        text = flatten_content(content)
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"

        if role == "assistant":
            clean, d = canonicalize_tool_calls(tool_calls)
            dropped_calls += d
            msg = {"role": "assistant", "content": text}
            if clean:
                msg["tool_calls"] = clean
            if not text and not msg.get("tool_calls"):
                continue  # H2: empty assistant turn — skip and do NOT count
            n_assistant += 1
        elif role == "tool":
            if not tool_call_id:
                if not text:
                    continue  # orphan empty tool turn — nothing to keep
                msg = {"role": "tool", "content": text}
            else:
                # H2: keep the paired result even if empty so the tool_call_id keeps its pair
                msg = {"role": "tool", "content": text or _TOOL_PLACEHOLDER, "tool_call_id": tool_call_id}
        else:
            if not text:
                continue
            msg = {"role": role, "content": text}
        out.append(msg)

    if n_assistant == 0 or not any(m["role"] == "assistant" for m in out):
        return None, 0, dropped_calls
    return out, n_assistant, dropped_calls
