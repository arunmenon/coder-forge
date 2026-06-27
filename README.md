# coder-forge

Fine-tune a ~30B open-weights model into a strong **agentic coding** model
(SWE-bench Verified / Terminal-Bench), then quantize it to run **locally on a
64–128GB Apple Silicon Mac**. Training runs on rented RunPod GPUs.

This plan is grounded in three adversarially-verified deep-research passes
(base-model selection, cost/feasibility, and a hardware-suitability matrix).
The headline conclusions are baked into the defaults below.

---

## TL;DR decisions

| Decision | Choice | Why |
| --- | --- | --- |
| Base model | **Qwen/Qwen3-Coder-30B-A3B-Instruct** | Apache-2.0 MoE, 30.5B total / **3.3B active** → fast Mac decode; purpose-built agentic function-call format; same family as the Nebius teacher (480B) → train↔serve aligned. |
| Method | **QLoRA SFT cold-start** (+ optional inference-time verifier) | SFT lifts a 32B from single-digit to ~20–38% SWE-bench Verified for **$5–320**. SOTA-scale RL is $13–30k + a container fleet → out of solo budget. |
| Training data | **Nebius `SWE-rebench-openhands-trajectories`** (resolved subset) | 67K execution-grounded OpenHands traces; filter to the ~32K resolved/test-pass ones. |
| Harness | **OpenHands** | Matches the Nebius trajectory format; serve the model behind an OpenAI-compatible endpoint and point OpenHands at it. |
| Training rig | **1× H100 80GB SXM** ($3.29/hr) or **1× RTX PRO 6000** ($2.09/hr, 96GB, Blackwell/FP4) | One GPU covers QLoRA SFT and (optional) QeRL FP4 RL. Cheapest floor: 1× 48GB A6000/L40S for QLoRA-only. |
| Local serving | **MLX 4-bit (high-quality: AWQ/DWQ/unsloth-UD)** on a **64GB M4 Max** | 64GB is the sweet spot for the MoE: full 128K context fits (~33GB), ~90–130 tok/s decode on MLX-native. |

### Realistic expected outcome (be honest about this)
- **Single-pass local SFT-only:** ~20% (≈500 trajectories) → ~38% SWE-bench
  Verified (≈8K trajectories, data-scaled).
- **+ local best-of-K verifier:** up to ~47% (slow on a Mac — multiplies latency).
- **Terminal-Bench 2.0 is brutal** for this size class (open band 23–52%,
  dominated by 100B–480B MoE; closed ~85%). Expect a dense/30B to land low-20s.
  SWE-bench is the benchmark to optimize; Terminal-Bench is a secondary signal.

---

## Pipeline

```
Phase 0  Baseline      Serve BASE model (hosted API is fine) → OpenHands → SWE-bench Verified subset → record pass@1
Phase 1  Data          Download Nebius traces → filter resolved → format to SFT chat/jsonl
Phase 2  SFT           QLoRA on 1× RunPod GPU (Axolotl) → adapter + merged weights
Phase 3  Re-eval       Serve fine-tuned model (vLLM) → same eval → measure the delta
Phase 4  Quantize      Merge → convert to MLX 4-bit (high-quality) for the Mac
Phase 5  Serve local   mlx_lm.server (OpenAI-compatible) → OpenHands on the Mac
(opt)    Verifier/RL   best-of-K verifier (cheap) before any GRPO/QeRL experiment
```

Run targets are wired in the `Makefile`. Each phase has a script under the
matching directory.

---

## Layout

```
coder-forge/
├── config/qwen3_coder_30b_qlora.yaml       # Axolotl QLoRA SFT config (30B, runnable)
├── config/qwen36_35b_a3b_qlora.stub.yaml   # 3.6-35B-A3B SFT config (STUB — see Base families)
├── data/prepare_nebius_sft.py              # Stage-1 SWE corpus (Nebius -> messages jsonl)
├── data/prepare_terminal_sft.py            # Stage-2 terminal corpus (LiteCoder/TermiGen -> jsonl)
├── train/run_sft.sh                        # launch QLoRA SFT on a single GPU
├── train/runpod_bootstrap.sh               # one-command pod setup + smoke test
├── serve/serve_vllm.sh                     # cloud OpenAI endpoint (base or tuned), Qwen tool parser
├── serve/serve_mlx_mac.sh                  # local Mac MLX OpenAI endpoint
├── eval/run_swebench_openhands.sh          # SWE-bench Verified via OpenHands / mini (subset)
├── eval/run_terminalbench.sh               # Terminal-Bench via the `tb` harness (subset)
├── eval/README.md                          # eval methodology + contamination notes
├── quantize/to_mlx.sh                      # HF → MLX 4-bit (high-quality) for the Mac
├── requirements-train.txt                  # RunPod-side training deps
├── requirements-eval.txt                   # harness/eval deps
└── .env.example                            # endpoints + API keys
```

## Base families — `BASE=qwen30` (default) | `qwen36`

Two base families are wired; switch with the `BASE` flag (selects config + model id
+ output dirs), e.g. `make baseline BASE=qwen36` or `make sft BASE=qwen36`.

> **See [`RECIPES.md`](RECIPES.md)** for the full demarcation — what's shared vs
> per-family (data pipeline is shared, recipe differs), the corpus mix per track, and
> the exact VM runbook for each. That's the doc to open when you sit down at the VM.

| | `qwen30` (default) | `qwen36` |
| --- | --- | --- |
| Model | Qwen3-Coder-30B-A3B | Qwen3.6-35B-A3B (verified real: 35B/3B, 256 experts, 30 DeltaNet + 10 attn, MTP; **73.4 SWE / 51.5 TB**, model claims) |
| Serving on Mac | ✅ mature (MLX/llama.cpp) | ✅ confirmed locally, but needs **vLLM ≥0.17.0** / latest llama.cpp |
| Fine-tuning | ✅ runnable today | ⚠️ **stub only** — bleeding-edge; multimodal `…ForConditionalGeneration` wrapper, LoRA must target the 30 DeltaNet layers (`in_proj_qkvz/in_proj_ba/out_proj`), MTP conflicts with packing |

**Recommendation (verified 2026-06-27): ship the `qwen30` pipeline first; treat
`qwen36` as a deliberate later upgrade.** A ~73% base has little headroom (heavy
imitation SFT risks *degrading* it), and its training stack is version-gated and
partly experimental. The high-value work for 3.6 is RL/RFT — prove that out cheaply
on the 30B first. **Do run the gate experiment on the 3.6 base now** (cheap, no
training): `make baseline BASE=qwen36` + `make baseline-terminal BASE=qwen36`, in
**both thinking and non-thinking modes**, to quantify the real gap.

## Quickstart

```bash
cp .env.example .env            # fill in keys/endpoints
# Phase 0 — baseline (no GPU needed; uses a hosted base-model endpoint)
make baseline EVAL_LIMIT=50
# Phase 1 — data
make data
# Phase 2 — SFT (run on a RunPod 1×H100 / RTX PRO 6000 pod)
make sft
# Phase 3 — re-eval the fine-tuned model
make eval EVAL_LIMIT=50
# Phase 4/5 — quantize + serve on the Mac
make quantize && make serve-mac
```

> Version-sensitive integration points (vLLM `qwen3_xml` tool parser, Axolotl
> `chat_template` keys, OpenHands V1 `swebench-infer`/`swebench-eval`,
> mini-swe-agent env vars, the Nebius `trajectory` column) were verified against
> current upstream docs (2026-06). Re-confirm if you install much newer versions.
> Nothing here is mocked.
