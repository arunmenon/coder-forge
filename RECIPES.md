# RECIPES — the two training tracks (clear demarcation)

Single source of truth for **what's the same, what's different, and exactly what to
run on the VM** for each base-model family. Switch tracks with `BASE=qwen30|qwen36`.

---

## Same or different? (the one-paragraph answer)

- **Data PIPELINE: shared & model-agnostic.** `data/prepare_*.py` download → filter →
  curate → emit ONE file, `data/sft_resolved.jsonl`, in OpenAI `messages` format. You
  do **not** maintain two copies of the corpus.
- **Wire-format rendering: automatic per model.** Axolotl `chat_template:
  tokenizer_default` applies *each model's own chat template* at train time, so the
  same JSONL is rendered into Qwen3-Coder's XML tool syntax for `qwen30` and into the
  3.6 template for `qwen36`. The conversion is not something you hand-maintain twice.
- **RECIPE: different per family.** How much SFT, LR, LoRA rank/targets, packing, and
  how much weight goes to RL — these differ, and they live in the two `config/*.yaml`
  files + the table below.

> Net: **same corpus pool + same prep code; different mix/amount + different
> hyperparameters + different RL emphasis.** One genuine *data* fork: 3.6 thinking-mode
> (see Corpus §, last paragraph).

---

## Side-by-side

| | **Track A — `qwen30`** (default) | **Track B — `qwen36`** |
|---|---|---|
| Base | Qwen3-Coder-30B-A3B (Apache-2.0; ~low-teens in-harness base) | Qwen3.6-35B-A3B (~73% SWE claim; **you confirmed clearly stronger**) |
| Core strategy | **Heavy imitation SFT is the main lever** | **Light format-SFT, then RL/RFT is the main lever** — heavy SFT risks *degrading* a strong base |
| SFT corpus | Full 2-stage mix (see Corpus) | Small **verified-only** subset (~format alignment) |
| LR / LoRA rank | 1e-4 / r=32–64 | 3e-5–7e-5 / r=16–32 |
| LoRA targets | `q/k/v/o_proj` + expert `gate_proj/up_proj/down_proj` | **+ DeltaNet `in_proj_qkvz`/`in_proj_ba`/`out_proj`** (30 of 40 layers are DeltaNet — the attn-only list adapts almost nothing) |
| Sequence packing | **yes** | **no** (MTP head conflicts with packing) |
| Execution-free data | low-weight breadth OK | **drop / downweight** |
| RL emphasis | optional Phase 2b | **primary** — verifier + best-of-N + GRPO on executable envs |
| Tooling | mature (Axolotl / vLLM / MLX stable) | version-gated (**vLLM ≥0.17.0**, latest transformers / llama.cpp; multimodal `…ForConditionalGeneration` wrapper) |
| Config | `config/qwen3_coder_30b_qlora.yaml` ✅ runnable | `config/qwen36_35b_a3b_qlora.stub.yaml` ⚠️ verify-before-run |
| Status | **ship this first** | deliberate **later upgrade** — gate on a baseline first |

---

## Corpus — shared pool, per-track mix

The trajectory **pool is identical** for both families (execution-verified agentic
traces). What changes is the **mix and amount**, not the source data.

| Stage | Wired sources (verified schema, pick via `--sources`) | Prep script | Used by |
|---|---|---|---|
| **Stage 1 — SWE** | `nebius/SWE-rebench-openhands` (default ~32K) · `nvidia/SWE-Hero` (34K) · `nvidia/Open-SWE-Traces` (openhands cfg, 207K) · `SWE-Gym/OpenHands-SFT` (491, format-align) | `data/prepare_swe_sft.py` ✅ | both |
| **Stage 2 — Terminal** | `Lite-Coder/LiteCoder-Terminal-SFT` (11.3K) · `m-a-p/TerminalTraj` (20K) | `data/prepare_terminal_sft.py` ✅ | both (Terminal-Bench) |
| **Not yet wired (need a custom adapter)** | `yoonholee/terminalbench-trajectories` (`steps` + agent/reward filter) · `nvidia/Nemotron-SFT-SWE-v2` agentless (not parquet-indexed) | — ⏳ | Recipe-A scale / skill-prior |

> Recipe-A SWE mix (the doc's recommended default), in one command:
> `make data DATA_ARGS="--sources nebius/SWE-rebench-openhands-trajectories nvidia/SWE-Hero-openhands-trajectories nvidia/Open-SWE-Traces SWE-Gym/OpenHands-SFT-Trajectories --max-examples 12000"`

**Curation — exactly what the prep scripts do (no over-claiming):**
- **Stage-1 (Nebius):** keeps `resolved` trajectories (within-run recovery is preserved),
  caps `--max-per-instance 3`, applies `--min-steps` / `--max-steps`, and
  **decontaminates** instance_ids against SWE-bench Verified by default (`--no-decontaminate`
  to skip). 
- **Stage-2 (terminal):** keeps **all** traces incl. failure-recovery (no success-only
  filter — verified finding), drops trivially short runs (`--min-assistant-turns`), dedups.
- **NOT yet automated (honest):** anti-cheat/shortcut detection (dataset-specific `TODO`
  hook) and exact files/lines-edited bounds. Don't assume these are applied.

- **Track A mix (heavy):** full Stage 1 + Stage 2 (+ agentless skill prior), ~2–3 : 1
  SWE : terminal. SFT is the main lever, so use the large mix.
- **Track B mix (light):** only a few thousand **best, verified-only** traces for format
  alignment; drop execution-free data; the real gains come from RL/RFT afterward.

**The one place the DATA itself forks:** if you serve 3.6 in **thinking mode** (its 73.4
is a thinking-mode number), the ideal SFT traces are **reasoning-augmented** (scarce —
you'd synthesize per-step reasoning via rejection sampling). The Nebius/terminal traces
are *action* traces that suit **non-thinking** serving. Default plan: serve 3.6
non-thinking → the **same corpus** works.

---

## VM runbook

### Track A — `qwen30` (run this first)
```bash
cp .env.example .env                      # MODEL_* → base for the gate, then your tune
make baseline EVAL_LIMIT=50               # gate: base SWE-bench number
make baseline-terminal TB_LIMIT=25        # gate: base Terminal-Bench number
make data DATA_ARGS="--max-examples 8000" # Stage-1 SWE   -> sft_resolved.jsonl + rebuilds sft_train.jsonl
make data-terminal                        # Stage-2 term  -> sft_terminal.jsonl + rebuilds sft_train.jsonl
make sft                                  # trains on data/sft_train.jsonl (mix, no config edit needed)
make eval EVAL_LIMIT=50                   # measure the delta (same harness!)
make quantize && make serve-mac           # ship to the Mac
```

### Track B — `qwen36` (later upgrade; verify tooling first)
```bash
# 0. GATE FIRST (cheap, no training) — both thinking + non-thinking modes:
make baseline BASE=qwen36 EVAL_LIMIT=50
make baseline-terminal BASE=qwen36 TB_LIMIT=25
# If the base is already strong → SKIP heavy SFT; go straight to verifier + best-of-N + RL.
# 1. Before any training, resolve the stub's blockers (see config header):
#    - vLLM>=0.17.0 + recent transformers installed
#    - python -c "...named_modules()" → fill the real DeltaNet LoRA target names
#    - decide MTP/packing + thinking-mode
make data DATA_ARGS="--max-examples 3000" # small verified-only mix
make sft BASE=qwen36                       # LIGHT SFT (config/qwen36_35b_a3b_qlora.stub.yaml)
make eval BASE=qwen36 EVAL_LIMIT=50
# 2. Then the real lever: verifier + best-of-N + GRPO/RFT on executable envs.
```

---

## What's runnable today vs pending

- ✅ **Runnable now:** Track A end-to-end including the **2-stage corpus** (SWE +
  terminal) → SFT → eval → quantize → serve; gate experiments for **both** tracks
  (SWE + Terminal-Bench) against a local endpoint.
- ⏳ **Pending data scripts:** agentless skill-prior prep (optional, Track A). Add more
  terminal sources to Stage-2 via `prepare_terminal_sft.py --sources <id> ...` once you
  confirm their schema with `--inspect`.
- ⚠️ **Track B training:** gated on the stub blockers above — serving 3.6 is confirmed,
  fine-tuning its arch is bleeding-edge.
