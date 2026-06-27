"""Model-family profile loader + resolver + validator — the ONLY reader of
config/families/*.yaml. Everything per-family (Makefile, serve, quantize, eval, config
render) resolves through this, so adding a family is one profile file. See docs/REFACTOR.md §3.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
FAMILIES_DIR = REPO / "config" / "families"


def list_families() -> list[str]:
    return sorted(p.stem for p in FAMILIES_DIR.glob("*.yaml"))


def _substitute(obj, ctx):
    if isinstance(obj, str):
        for key, value in ctx.items():
            obj = obj.replace("{" + key + "}", str(value))
        return obj
    if isinstance(obj, dict):
        return {k: _substitute(v, ctx) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute(v, ctx) for v in obj]
    return obj


def load_profile(family: str) -> dict:
    path = FAMILIES_DIR / f"{family}.yaml"
    if not path.is_file():
        raise SystemExit(f"Unknown family '{family}'. Known: {', '.join(list_families())}")
    profile = yaml.safe_load(path.read_text())
    return _substitute(profile, {"id": profile["id"], "slug": profile["slug"]})


def get(family: str, dotted: str):
    node = load_profile(family)
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise SystemExit(f"key '{dotted}' not found in family '{family}'")
        node = node[part]
    return node


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def export_env(family: str) -> dict:
    """Flatten the profile to the shell vars the scripts/Makefile consume."""
    p = load_profile(family)
    return {
        "FAMILY_ID": p["id"],
        "SLUG": p["slug"],
        "BASE_MODEL": p["base_model"],
        "STATUS": p["status"],
        "CONFIG": f"config/.generated/{p['id']}.yaml",
        "OUTPUT_DIR": p["paths"]["output_dir"],
        "MERGED_DIR": p["paths"]["merged_dir"],
        "MLX_DIR": p["paths"]["mlx_dir"],
        "TOOL_CALL_PARSER": p["serve"]["tool_call_parser"],
        "REASONING_PARSER": _fmt(p["serve"].get("reasoning_parser")),
        "ENABLE_AUTO_TOOL_CHOICE": "1" if p["serve"].get("enable_auto_tool_choice") else "0",
        "MAX_MODEL_LEN": p["serve"]["max_model_len"],
        "GPU_MEMORY_UTILIZATION": p["serve"]["gpu_memory_utilization"],
        "MIN_VLLM": _fmt(p["serve"].get("min_vllm")),
        "Q_BITS": p["quantize"]["q_bits"],
        "Q_GROUP_SIZE": p["quantize"]["q_group_size"],
        "SAMPLING_JSON": json.dumps(p["serve"]["sampling"]),
        "THINKING": _fmt(p["arch"]["thinking"]),
    }


def doctor(family: str) -> int:
    """Static invariants that must hold before any GPU spend. Returns 0 (pass) / 1 (fail)."""
    p = load_profile(family)
    train, lora, arch = p["train"], p["train"]["lora"], p["arch"]
    targets = set(lora.get("target_modules", []))
    problems, warnings = [], []

    if arch.get("mtp") and train.get("sample_packing"):
        problems.append("arch.mtp=true but train.sample_packing=true (MTP conflicts with packing)")
    bad_frozen = targets & set(lora.get("frozen_modules", []))
    if bad_frozen:
        problems.append(f"frozen_modules also in target_modules: {sorted(bad_frozen)}")
    bad_full = targets & set(lora.get("full_ft_only", []))
    if bad_full:
        problems.append(f"full_ft_only modules also LoRA-targeted: {sorted(bad_full)}")
    if not targets:
        problems.append("train.lora.target_modules is empty")
    if p["status"] == "stub":
        if os.environ.get("ALLOW_STUB") == "1":
            warnings.append("status=stub overridden by ALLOW_STUB=1 — verify tooling + LoRA targets first")
        else:
            problems.append("status=stub — refusing to proceed (verify tooling + re-derive LoRA targets "
                            "from named_modules(), then set ALLOW_STUB=1 to override)")
    if arch.get("linear_attention") and lora.get("target_strategy") != "hybrid_linear_attention":
        warnings.append("linear_attention set but target_strategy is not hybrid_linear_attention")

    print(f"doctor: family={family} base={p['base_model']} status={p['status']}")
    for w in warnings:
        print(f"  WARN  {w}")
    for e in problems:
        print(f"  FAIL  {e}")
    if not problems:
        print("  PASS  static profile checks OK "
              "(deep check — target_modules vs named_modules() — needs the model on the GPU box)")
    return 1 if problems else 0


def main(argv) -> None:
    if not argv:
        raise SystemExit("usage: family.py list | get <family> <dotted.key> | export <family> | doctor <family>")
    cmd, rest = argv[0], argv[1:]
    if cmd == "list":
        print("\n".join(list_families()))
    elif cmd == "get":
        print(_fmt(get(rest[0], rest[1])))
    elif cmd == "export":
        for key, value in export_env(rest[0]).items():
            print(f"{key}={shlex.quote(_fmt(value))}")
    elif cmd == "doctor":
        raise SystemExit(doctor(rest[0]))
    else:
        raise SystemExit(f"unknown command '{cmd}'")


if __name__ == "__main__":
    main(sys.argv[1:])
