"""Tool-call codecs — emit the corpus's (canonical, dict-args) tool calls in the shape a
given model family's chat template expects. Selected per family via tool_call.codec.

The corpus is normalized to the canonical DICT-args form (strictly more information). A codec
is the family-specific final shaping applied at prep/render time:
  qwen3_xml_dict -> keep arguments as a dict (Qwen3-Coder iterates a mapping)
  openai_string  -> arguments as a JSON string (templates that expect the OpenAI wire form)
"""

from __future__ import annotations

import json


def _keep_dict(messages):
    return messages


def _args_to_string(messages):
    out = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            calls = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                if isinstance(fn.get("arguments"), (dict, list)):
                    fn = {**fn, "arguments": json.dumps(fn["arguments"], ensure_ascii=False)}
                    tc = {**tc, "function": fn}
                calls.append(tc)
            msg = {**msg, "tool_calls": calls}
        out.append(msg)
    return out


CODECS = {
    "qwen3_xml_dict": _keep_dict,
    "openai_string": _args_to_string,
}


def apply_codec(messages, codec: str):
    fn = CODECS.get(codec)
    if fn is None:
        raise SystemExit(f"Unknown tool_call codec '{codec}'. Known: {', '.join(CODECS)}")
    return fn(messages)
