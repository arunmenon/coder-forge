"""Unit tests for curation/mix logic (H3 mixing is exercised via run_prep in integration;
here we cover curate_row + split filtering, which need no network)."""

from collections import Counter

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
    keep, _, reason, _ = curate_row(
        _row(instance_id="repo__leak"), SPEC, _opts(decontaminate=True),
        {"repo__leak"}, set(), Counter())
    assert keep is False and reason == "contaminated"


def test_no_instance_id_required_flag():
    keep, _, reason, _ = curate_row(
        _row(instance_id=None), SPEC, _opts(decontaminate=True, require_instance_id=True),
        {"x"}, set(), Counter())
    assert keep is False and reason == "no_instance_id"


def test_per_instance_cap():
    seen, per = set(), Counter({"repo__1": 3})
    keep, _, reason, _ = curate_row(_row(), SPEC, _opts(max_per_instance=3), set(), seen, per)
    assert keep is False and reason == "over_per_instance_cap"


def test_dedup():
    seen, per = set(), Counter()
    k1, *_ = curate_row(_row(), SPEC, _opts(), set(), seen, per)
    k2, _, reason, _ = curate_row(_row(), SPEC, _opts(), set(), seen, per)
    assert k1 is True and k2 is False and reason == "dup"


def test_min_steps():
    short = _row(trajectory=[{"role": "assistant", "content": "hi"}])
    keep, _, reason, _ = curate_row(short, SPEC, _opts(min_steps=2), set(), set(), Counter())
    assert keep is False and reason == "too_short"


def test_malformed_messages():
    keep, _, reason, _ = curate_row(_row(trajectory="notalist"), SPEC, _opts(), set(), set(), Counter())
    assert keep is False and reason == "malformed"


# --- H4: training-split allowlist ---

def test_is_training_split_excludes_test_val():
    spec = SourceSpec("x", "messages")
    assert is_training_split("train", spec) is True
    assert is_training_split("test", spec) is False
    assert is_training_split("validation", spec) is False
    assert is_training_split("minimax_m25", spec) is True  # Open-SWE-Traces generator split


def test_is_training_split_explicit_allowlist():
    spec = SourceSpec("x", "messages", training_splits=("train.success.oss",))
    assert is_training_split("train.success.oss", spec) is True
    assert is_training_split("train", spec) is False


def test_content_hash_stable():
    assert content_hash([{"role": "user", "content": "a"}]) == content_hash([{"role": "user", "content": "a"}])
