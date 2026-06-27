"""Shared data-prep library: one normalizer, one source registry, one curation+mix engine."""

from .normalize import canonicalize_tool_calls, flatten_content, normalize_turn, to_openai_messages
from .pipeline import PrepOptions, PrepResult, content_hash, run_prep, write_manifest
from .sources import DEFAULT_SOURCES, REGISTRY, SourceSpec, is_training_split, resolve

__all__ = [
    "to_openai_messages", "normalize_turn", "canonicalize_tool_calls", "flatten_content",
    "PrepOptions", "PrepResult", "run_prep", "write_manifest", "content_hash",
    "SourceSpec", "REGISTRY", "DEFAULT_SOURCES", "is_training_split", "resolve",
]
