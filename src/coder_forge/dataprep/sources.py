"""Verified-schema source registry — one typed SourceSpec per dataset, shared by SWE and
terminal prep. Schemas verified against the HF datasets-server (2026-06-27)."""

from __future__ import annotations

from dataclasses import dataclass, field

# Split-name substrings that must NOT enter the training corpus (H4).
NON_TRAINING_SPLIT_MARKERS = ("test", "val", "valid", "validation", "eval", "dev", "holdout")


@dataclass(frozen=True)
class SourceSpec:
    dataset_id: str
    messages_field: str
    fmt: str = "openai"                 # openai | sharegpt
    resolved_field: str | None = None   # None => already success-only (no filter)
    instance_field: str = "instance_id"
    config: str | None = None           # HF config name (e.g. Open-SWE-Traces "openhands")
    training_splits: tuple = ()         # explicit allowlist; () => auto (non-test-like only)
    kind: str = "swe"                   # swe | terminal


REGISTRY: dict[str, SourceSpec] = {
    # --- SWE (Stage 1) ---
    "nebius/SWE-rebench-openhands-trajectories": SourceSpec(
        "nebius/SWE-rebench-openhands-trajectories", "trajectory", resolved_field="resolved", kind="swe"),
    "nvidia/SWE-Hero-openhands-trajectories": SourceSpec(
        "nvidia/SWE-Hero-openhands-trajectories", "trajectory", resolved_field=None, kind="swe"),
    "nvidia/Open-SWE-Traces": SourceSpec(
        "nvidia/Open-SWE-Traces", "trajectory", resolved_field="resolved", config="openhands", kind="swe"),
    "SWE-Gym/OpenHands-SFT-Trajectories": SourceSpec(
        "SWE-Gym/OpenHands-SFT-Trajectories", "messages", resolved_field=None,
        training_splits=("train.success.oss",), kind="swe"),
    # --- Terminal (Stage 2) ---
    "Lite-Coder/LiteCoder-Terminal-SFT": SourceSpec(
        "Lite-Coder/LiteCoder-Terminal-SFT", "conversations", fmt="sharegpt", kind="terminal"),
    "m-a-p/TerminalTraj": SourceSpec(
        "m-a-p/TerminalTraj", "messages", fmt="openai", kind="terminal"),
}

DEFAULT_SOURCES = {
    "swe": ["nebius/SWE-rebench-openhands-trajectories"],
    "terminal": ["Lite-Coder/LiteCoder-Terminal-SFT"],
}


def is_training_split(name: str, spec: SourceSpec) -> bool:
    """H4: only training splits enter the corpus. Explicit allowlist wins; else exclude
    anything that looks like test/val/eval."""
    if spec.training_splits:
        return name in spec.training_splits
    return not any(marker in name.lower() for marker in NON_TRAINING_SPLIT_MARKERS)


def resolve(dataset_id: str, *, allow_unverified: bool = False) -> SourceSpec:
    spec = REGISTRY.get(dataset_id)
    if spec is not None:
        return spec
    if allow_unverified:
        # Best-effort: assume OpenAI messages; caller accepted the risk.
        return SourceSpec(dataset_id, "messages", fmt="openai")
    raise SystemExit(
        f"Unknown source '{dataset_id}'. Add it to coder_forge.dataprep.sources.REGISTRY "
        f"(verify its schema first) or pass --allow-unverified.")
