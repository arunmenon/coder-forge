# coder-forge — Refactor Spec (production-readiness + model-family extensibility)

Output of a deep multi-agent code review (correctness · extensibility · DRY · production).
North star: **add a new model family → run the existing recipes as-is.**

---

## 1. Verdict

The data-prep is **not correct enough to start a real SFT run** — it can silently corrupt
the corpus and produce an unreproducible training file. Three load-bearing bugs:
- **Tool-call `arguments` stay JSON strings** while Qwen's `tokenizer_default` template
  iterates them as a dict → train-time crash or zero-argument tool calls (corrupts the
  exact thing we train on).
- **Empty assistant turns counted before skip** → rows can be written as `{"messages": []}`,
  and empty `tool` results get dropped, breaking the `assistant.tool_calls ↔ tool` pairing.
- **`--max-examples` is a global source-order cap** → the documented Recipe-A 4-source mix
  collapses to Nebius-only.

Plus a **train/eval leak** (config-only sources iterate every split), **silent decontam
no-ops** (rows without `instance_id`), and the extensibility gap: per-family knowledge is
scattered across ~7 files, and `serve_vllm.sh` hardcodes `--tool-call-parser qwen3_xml`, so
`make serve-cloud BASE=qwen36` would serve 3.6 with the wrong parser.

---

## 2. Correctness must-fixes (before any training run)

| # | Sev | Fix |
|---|---|---|
| H1 | high | Canonicalize `tool_call.function.arguments` string→dict (`json.loads`, guard malformed, never re-parse a dict). |
| H2 | high | Increment assistant count only when a message is appended; never emit empty `messages`; keep empty `tool` turns (placeholder) so every `tool_call_id` keeps its pair. |
| H3 | high | Per-source caps/weights + round-robin/reservoir interleave; error when a requested source writes 0. |
| H4 | high | Training-split allowlist; skip `test/val/eval/dev` splits unless opted in. |
| H5 | med | Decontam/per-instance must not silently no-op on missing `instance_id` — bucket + surface; content-hash fallback; extend to terminal. |
| M6 | med | Flatten structured **assistant** content to text (don't `json.dumps` content blocks into the target). |
| M7 | med | Shared sha1 content-dedup in both prep paths; seed dedup/per-instance state on `--append`. |
| M8 | med | Terminal must require registry membership (no silent schema sniffing); auto-detect behind `--allow-unverified`. |
| M9 | low | `run_sft.sh`: parse `output_dir` with real YAML, fail hard (don't fall back to a mismatched default). |
| M10 | low | Guard Makefile `build_train` (no stdin read / empty corpus); `make clean`; `streaming=True` for large sources. |

All fixes land **once** in the shared module (§4) so SWE and terminal both inherit them.

---

## 3. Model-Family Profile Registry (CENTERPIECE)

One profile per family is the **single source of truth** for all per-family knowledge.
Every consumer reads it through one resolver.

```
config/families/<id>.yaml      # one profile per family — the ONLY per-family file
config/base.qlora.yaml         # model-agnostic Axolotl defaults (the shared ~80%)
scripts/family.py              # the only profile reader (pyyaml)
scripts/family.sh              # POSIX shim: get / export / list (for Make + bash)
```

### Profile schema (worked: `config/families/qwen30.yaml`)
```yaml
id: qwen30
slug: qwen3-coder-30b-a3b
base_model: Qwen/Qwen3-Coder-30B-A3B-Instruct
status: stable                  # stable | stub
arch:
  loader: AutoModelForCausalLM  # qwen36: AutoModelForConditionalGeneration (multimodal)
  moe: true
  shared_expert: false          # qwen36: true
  linear_attention: null        # qwen36: deltanet
  mtp: false                    # qwen36: true (conflicts with packing)
  thinking: false               # false | hybrid | true
chat_template: tokenizer_default
tool_call: { codec: qwen3_xml_dict }   # qwen3_xml_dict -> args dict | openai_string -> args JSON string
serve:
  tool_call_parser: qwen3_xml          # THE serve_vllm.sh hardcode
  reasoning_parser: null               # appended only when set (qwen36 thinking)
  enable_auto_tool_choice: true
  max_model_len: 32768
  gpu_memory_utilization: 0.92
  min_vllm: null                       # qwen36: "0.17.0"
  sampling: { temperature: 0.7, top_p: 0.8, top_k: 20, repetition_penalty: 1.05 }
quantize: { q_bits: 4, q_group_size: 64 }
train:
  lora:
    r: 32
    alpha: 64
    target_strategy: moe_expert        # dense_linear | moe_expert | hybrid_linear_attention
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
    frozen_modules: [mlp.gate]         # MoE router — never adapt
    full_ft_only: []                   # qwen36: [conv1d, A_log, dt_bias]
  sequence_len: 16384
  sample_packing: true                 # qwen36: false (MTP)
  learning_rate: 1.0e-4                # qwen36: 5.0e-5
  num_epochs: 3                        # qwen36: 1
  seed: 42
paths:
  output_dir: outputs/{slug}-qlora
  merged_dir: outputs/{slug}-merged
  mlx_dir: models/{slug}-mlx-4bit
  dataset_prepared: data/.axolotl_prepared-{id}
```

### Consumers (each reads the profile, holds zero family literals)
- **Makefile** — delete the `ifeq` block; `FAMILY ?= qwen30`; resolve `BASE_MODEL/CONFIG/
  MERGED_DIR/MLX_DIR` via `scripts/family.sh get …`. Family-count-agnostic.
- **prep / config-gen** — `tool_call.codec` picks the `ToolCallCodec`; `train.*` + `chat_template`
  feed the **generated** Axolotl YAML (base + profile deltas).
- **serve_vllm.sh** — build the vllm arg array from the profile (`--tool-call-parser`,
  optional `--reasoning-parser`, sampling); assert vLLM ≥ `min_vllm`. The `qwen3_xml` literal disappears.
- **serve_mlx_mac.sh / to_mlx.sh** — read `paths.mlx_dir/merged_dir` + `quantize.*`.
- **eval** — inject `serve.sampling` + thinking-mode into the OpenHands LLM config / `tb` flags.
- **bootstrap** — `FAMILY=` resolves `CONFIG` from the profile.

### Adding a family = ONE file
A new Qwen (`qwen37`) or a non-Qwen dense family (e.g. GLM/Llama with `llama3_json` parser
and `target_strategy: dense_linear`) is **one profile YAML, zero code edits**. The only escape
hatch: a genuinely new arch knob missing from `base.qlora.yaml` (a one-line template add).
`make doctor FAMILY=<id>` validates LoRA targets against the real `named_modules()`, the router
is frozen, vLLM ≥ min_vllm, and `packing=false` when `mtp=true` — before any GPU spend.

---

## 4. DRY / modularity (target layout)

Root cause: no Python package boundary, so the two prep scripts share code by copy-paste —
and `ROLE_NORMALIZE` has already drifted. Every §2 fix would otherwise be made twice.

```
pyproject.toml                       # PEP 621; [project.scripts]; ruff/pytest config
src/coder_forge/
  dataprep/
    normalize.py   # ONE ROLE_NORMALIZE; normalize_turn; to_openai_messages;
                   #   tool-call canonicalizer (H1); content flattener (M6)
    sources.py     # frozen SourceSpec(dataset_id, messages_field, fmt, resolved_field,
                   #   config, training_splits, instance_field, per_source_cap, weight, ...); one REGISTRY
    curate.py      # composable steps: SuccessFilter, Decontaminate, PerInstanceCap,
                   #   MinSteps, MaxSteps, TruncateContent, Dedup(sha1)
    codecs.py      # ToolCallCodec registry: qwen3_xml_dict, openai_string (keyed by family)
    io.py          # load_source (streaming-safe), write_jsonl, inspect_source, write_manifest
    cli.py         # `forge-prep-sft swe|terminal`, one shared arg group
  configs/render.py# deep-merge base.qlora.yaml + profile.train.* -> config/.generated/<id>.yaml
  family.py        # profile loader
config/base.qlora.yaml + config/families/*.yaml
scripts/
  family.sh
  lib/common.sh    # log, load_env, require_cmd, require_docker, resolve_model, litellm_route
  build_train.sh   # the ONE merge + dedup + manifest implementation
```

Deduped: the two `to_openai_messages`/`ROLE_NORMALIZE`/registry/`inspect` copies; the two
near-identical Axolotl YAMLs (→ base + ~8-line delta); the 7 shell scripts' boilerplate
(→ `common.sh`); the duplicated `cat $(ls …)`; the divergent CLI flag names.

---

## 5. Production hardening

- **Packaging** — `pyproject.toml`, `[project.scripts]` (`forge-prep-sft`, `forge-render-config`,
  `forge-family`), `[project.optional-dependencies] train/eval` replacing the flat requirements.
- **Tests** (`tests/`, pytest, network-stubbed) — unit tests for `normalize`/`curate` (incl. H1,
  H2, H3, H4, M6), and **the highest-value test**: render a fixture trajectory through
  `chat_template: tokenizer_default` and assert a real `<tool_call>` span with non-empty
  `<parameter>` tags appears (proves H1) and only assistant tokens are unmasked.
- **CI** (`.github/workflows/ci.yml`) — `py_compile`, `bash -n` over `git ls-files '*.sh'`,
  `ruff`, `pytest`, `forge-render-config --family <each>` smoke. Dependency-light (no axolotl/vllm).
- **Manifest** — `write_manifest()` per output (script+git SHA, argv, per-source rows seen/written,
  skip-reason counts, decontam id count, dedup stats, sha256, UTC). `seed` plumbed from profile.
- **Pinning** — pin `axolotl`/`transformers`/`vllm`/`terminal-bench`; commit a lockfile; Renovate.
- **Observability** — `logging` instead of `print`; per-source try/except + bounded retry + flush.
- **Docs** — fix the "ONE file" claim and the stale `prepare_nebius_sft.py` reference.

---

## 6. Phased plan

| Phase | Effort | Ships |
|---|---|---|
| **0 — Correctness via the shared normalizer** | M | A correct, leak-free corpus. Stand up `src/coder_forge/dataprep/*`, rewire both prep scripts to thin CLIs, land H1/H2/H4/H5/M6/M7 once + H3 mixer + M8/M9/M10. |
| **1 — Manifest + reproducible assembly** | S | Every `sft_train.jsonl` auditable + reproducible (build_train.sh, manifests, seed, logging). |
| **2 — Model-Family Profile Registry** (CORE) | L | "add a family = one profile file," validated by `make doctor` before GPU spend. |
| **3 — DRY finish + packaging** | M | Installable, importable, single-config-source repo (pyproject, common.sh, unified flags). |
| **4 — CI + tests + pinning** | M | Regressions in fragile integration points are gated. |
| **5 — Docs reconciliation** | S | Docs match code. |

**Critical path to a trustworthy first run: Phase 0 → 1.** **Critical path to the north star: Phase 2.**
Phases 3–5 harden and are independently shippable.
