"""Unit tests for curation/mix logic (offline; the live multi-source round-robin in run_prep
is covered by the integration smoke since it needs `datasets`)."""

from collections import Counter

from coder_forge.dataprep.codecs import apply_codec
from coder_forge.dataprep.pipeline import PrepOptions, content_hash, curate_row
from coder_forge.dataprep.sources import SourceSpec, is_training_split

SPEC = SourceSpec("x/y", "trajectory", resolved_field="resolved", kind="swe")
TRAJ = [
    {"role": "user", "content": "fix"},
    {"role": "assistant", "content": "ok", "tool_calls":
        [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
    {"role": "tool", "content": "out", "tool_call_id": "c"},
]


def _row(**kw):
    base = {"resolved": 1, "instance_id": "repo__1", "trajectory": TRAJ}
    base.update(kw)
    return base


def _opts(**kw):
    base = dict(decontaminate=False, max_per_instance=0, min_steps=1)
    base.update(kw)
    return PrepOptions(**base)


def test_resolved_filter():
    keep, *_ = curate_row(_row(resolved=0), SPEC, _opts(), set(), set(), Counter())
    assert keep is False


def test_decontamination_drops_eval_instance():
    keep, _, reason, _, _ = curate_row(
        _row(instance_id="repo__leak"), SPEC, _opts(decontaminate=True),
        {"repo__leak"}, set(), Counter())
    assert keep is False and reason == "contaminated"


def test_no_instance_id_required_flag():
    keep, _, reason, _, _ = curate_row(
        _row(instance_id=None), SPEC, _opts(decontaminate=True, require_instance_id=True),
        {"x"}, set(), Counter())
    assert keep is False and reason == "no_instance_id"


def test_missing_instance_id_kept_and_flagged():
    # H5: kept when require_instance_id is off, but flagged un-decontaminable.
    keep, _, reason, _, undecon = curate_row(
        _row(instance_id=None), SPEC, _opts(decontaminate=True, require_instance_id=False),
        {"x"}, set(), Counter())
    assert keep is True and undecon is True


def test_per_instance_cap():
    keep, _, reason, _, _ = curate_row(
        _row(), SPEC, _opts(max_per_instance=3), set(), set(), Counter({"repo__1": 3}))
    assert keep is False and reason == "over_per_instance_cap"


def test_dedup():
    seen, per = set(), Counter()
    k1, *_ = curate_row(_row(), SPEC, _opts(), set(), seen, per)
    k2, _, reason, _, _ = curate_row(_row(), SPEC, _opts(), set(), seen, per)
    assert k1 is True and k2 is False and reason == "dup"


def test_min_steps():
    short = _row(trajectory=[{"role": "assistant", "content": "hi"}])
    keep, _, reason, _, _ = curate_row(short, SPEC, _opts(min_steps=2), set(), set(), Counter())
    assert keep is False and reason == "too_short"


def test_malformed_messages():
    keep, _, reason, _, _ = curate_row(_row(trajectory="notalist"), SPEC, _opts(), set(), set(), Counter())
    assert keep is False and reason == "malformed"


def test_codec_applied_in_curate():
    # openai_string codec re-serializes the dict args back to a JSON string.
    keep, payload, *_ = curate_row(_row(), SPEC, _opts(codec="openai_string"), set(), set(), Counter())
    assert keep is True
    args = [m for m in payload if m["role"] == "assistant"][0]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str)


# --- codec unit ---

def test_codec_openai_string_dict_to_string():
    msgs = [{"role": "assistant", "content": "x",
             "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]}]
    out = apply_codec(msgs, "openai_string")
    assert out[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


def test_codec_qwen_keeps_dict():
    msgs = [{"role": "assistant", "content": "x",
             "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]}]
    out = apply_codec(msgs, "qwen3_xml_dict")
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}


# --- H4: training-split allowlist ---

def test_is_training_split_excludes_test_val():
    spec = SourceSpec("x", "messages")
    assert is_training_split("train", spec) is True
    assert is_training_split("test", spec) is False
    assert is_training_split("validation", spec) is False
    assert is_training_split("minimax_m25", spec) is True


def test_is_training_split_explicit_allowlist():
    spec = SourceSpec("x", "messages", training_splits=("train.success.oss",))
    assert is_training_split("train.success.oss", spec) is True
    assert is_training_split("train", spec) is False


def test_content_hash_stable():
    assert content_hash([{"role": "user", "content": "a"}]) == content_hash([{"role": "user", "content": "a"}])
