"""Unit tests for the shared normalizer — the correctness fixes H1, H2, M6."""

from coder_forge.dataprep.normalize import (
    canonicalize_tool_calls,
    flatten_content,
    to_openai_messages,
)


def _tc(args):
    return [{"id": "c1", "type": "function", "function": {"name": "execute_bash", "arguments": args}}]


# --- H1: tool_call.function.arguments string -> dict ---

def test_h1_string_arguments_become_dict():
    calls, dropped = canonicalize_tool_calls(_tc('{"command": "ls"}'))
    assert dropped == 0
    assert calls[0]["function"]["arguments"] == {"command": "ls"}


def test_h1_dict_arguments_untouched():
    calls, dropped = canonicalize_tool_calls(_tc({"command": "ls"}))
    assert dropped == 0
    assert calls[0]["function"]["arguments"] == {"command": "ls"}


def test_h1_malformed_arguments_dropped():
    calls, dropped = canonicalize_tool_calls(_tc("{not json"))
    assert dropped == 1
    assert calls is None


def test_h1_empty_string_arguments_become_empty_dict():
    calls, _ = canonicalize_tool_calls(_tc(""))
    assert calls[0]["function"]["arguments"] == {}


def test_h1_list_arguments_dropped():
    # valid JSON but NOT a mapping — Qwen's template iterates args as a dict.
    calls, dropped = canonicalize_tool_calls(_tc("[1, 2]"))
    assert dropped == 1 and calls is None


def test_h1_scalar_arguments_dropped():
    calls, dropped = canonicalize_tool_calls(_tc("42"))
    assert dropped == 1 and calls is None


def test_h1_missing_arguments_become_empty_dict():
    calls, dropped = canonicalize_tool_calls([{"function": {"name": "f"}}])
    assert dropped == 0 and calls[0]["function"]["arguments"] == {}


def test_h1_in_full_trajectory_preserves_dict_args():
    traj = [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "searching", "tool_calls": _tc('{"command": "grep x"}')},
        {"role": "tool", "content": "found", "tool_call_id": "c1"},
    ]
    msgs, n, dropped = to_openai_messages(traj)
    assert n == 1 and dropped == 0
    assert msgs[1]["tool_calls"][0]["function"]["arguments"] == {"command": "grep x"}


# --- H2: empty assistant turns / paired tool results ---

def test_h2_empty_assistant_turn_not_counted_or_emitted():
    traj = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},          # empty, no tool call
        {"role": "assistant", "content": "real answer"},
    ]
    msgs, n, _ = to_openai_messages(traj)
    assert n == 1
    assert [m["content"] for m in msgs if m["role"] == "assistant"] == ["real answer"]


def test_h2_no_trainable_assistant_returns_none():
    traj = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}]
    msgs, n, _ = to_openai_messages(traj)
    assert msgs is None and n == 0


def test_h2_empty_tool_result_kept_with_placeholder():
    traj = [
        {"role": "assistant", "content": "run", "tool_calls": _tc("{}")},
        {"role": "tool", "content": "", "tool_call_id": "c1"},
    ]
    msgs, n, _ = to_openai_messages(traj)
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_call_id"] == "c1"
    assert tool_msgs[0]["content"]  # non-empty placeholder, pairing preserved


def test_h2_never_emits_empty_messages():
    msgs, n, _ = to_openai_messages([{"role": "assistant", "content": "  ".strip()}])
    assert msgs is None  # would otherwise be {"messages": []}


# --- M6: structured content flattened, not json-dumped ---

def test_m6_structured_assistant_content_flattened():
    traj = [{"role": "assistant", "content": [{"type": "text", "text": "hello"},
                                              {"type": "text", "text": "world"}]}]
    msgs, n, _ = to_openai_messages(traj)
    assert msgs[0]["content"] == "hello\nworld"
    assert "[{" not in msgs[0]["content"]  # not json.dumps'd


def test_m6_flatten_plain_string():
    assert flatten_content("plain") == "plain"


def test_m6_dict_content_text_extracted():
    msgs, n, _ = to_openai_messages([{"role": "assistant", "content": {"text": "hi"}}])
    assert msgs[0]["content"] == "hi" and "{" not in msgs[0]["content"]


# --- ShareGPT mapping ---

def test_sharegpt_roles_map():
    traj = [{"from": "human", "value": "task + output"}, {"from": "gpt", "value": "action"}]
    msgs, n, _ = to_openai_messages(traj, fmt="sharegpt")
    assert n == 1
    assert [m["role"] for m in msgs] == ["user", "assistant"]
