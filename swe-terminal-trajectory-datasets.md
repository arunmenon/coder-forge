# Gold-Standard Agentic-Coding Trajectory Datasets — SWE-bench + Terminal-Bench

*Counts were spot-checked against public dataset cards / pages on 2026-06-27; treat them as **approximate and version-dependent**. Some figures (model scores, environment task totals, the TerminalTraj 50,733) are **paper/model claims, not dataset-card facts** and are tagged as such. Always re-check the card before you depend on a number.*

---

## How to read this

Four different things get loosely called "trajectories." Mixing them up is the most common mistake:

1. **SFT trajectory datasets** — full agent rollouts (messages + tool calls + observations + patch). Imitation targets.
2. **RL executable environments** — repos packaged with tests so you can compute a reward. You RL *against* these; they are not trajectories.
3. **Verifier / reward-model data** — resolved + unresolved pairs, or per-step error labels, for a reranker / PRM / best-of-N selector.
4. **Task sources** — issue/PR pairs with no rollout; raw material to *generate* your own trajectories.

**Two separate notions of "format" — keep them distinct:**

- **Action space / tool vocabulary** (the "scaffold"): OpenHands tools vs SWE-agent commands vs terminal bash. **This must match your eval harness.** It is the dominant variable — bigger than dataset size.
- **Wire syntax**: the literal token format of a tool call. For this base model it is **always Qwen3-Coder's native function-call syntax** (§8), regardless of which action space you chose. You convert every trajectory's wire syntax to Qwen's; you do *not* mix action spaces.

**Execution-verified vs execution-free.** "Execution-free" rollouts were never run against tests — cheap and broad, but they teach plausible-looking behavior, not correct behavior. Keep them out of the final SFT stage or weight them down.

---

## 1 · SWE-bench — core SFT (execution-verified)

| Dataset | Trajectories | License | Scaffold | Generator | Exec-verified | Notes |
|---|---|---|---|---|---|---|
| `SWE-bench/SWE-smith-trajectories` | **5,017 trajectories** (card shows **76,002 rows**) | MIT | SWE-agent | Claude 3.7 Sonnet | Yes | See row-count note below. Trained SWE-agent-LM-32B → **40.2% SWE-bench Verified** *(model claim)*. Tasks are synthetic (SWE-smith). |
| `nebius/SWE-agent-trajectories` | **80,036** (13,389 `target==true`) | CC-BY-4.0 | SWE-agent | 3 models incl. swe-agent-llama-70b | Yes (`eval_logs`) | Resolved-only → SFT; full set → verifier + contrast. Underlies Nebius's 40.6% Verified agent *(blog claim)*. |
| `SWE-Gym/OpenHands-SFT-Trajectories` | **491** | MIT | OpenHands | gpt-4o-era | Yes (success only) | Tiny but clean. Stage-1 OpenHands tool-format alignment. |
| `SWE-Gym/OpenHands-Sampled-Trajectories` | **6,055** | ⚠ none declared | OpenHands | — | Yes (`resolved`/`test_result`) | Raw rollouts for your own filtering. |
| `nebius/SWE-rebench-openhands-trajectories` | **67,074** (32,161 successful; 3,792 issues / 1,823 repos) | CC-BY-4.0 | OpenHands v0.54 | Qwen3-Coder-480B | Yes (Docker, real issues) | Large, clean, permissive OpenHands corpus on real GitHub issues. |

> **`SWE-smith-trajectories`: 5,017 vs 76,002 explained.** The dataset card lists **76,002 rows** across 3 splits (`tool` 24.1k / `xml` 26.1k / `ticks` 25.8k) — these are the **same trajectories re-rendered in three action-formatting styles** (plus per-step row expansion). The headline **5,017** is the count of underlying agent trajectories actually used to train SWE-agent-LM-32B. Pick **one** split (matching your wire format) — don't load all three or you 3× the same data.

## 2 · SWE-bench — large-scale / multilingual SFT

| Dataset | Trajectories | License | Scaffold | Generator | Exec-verified | Notes |
|---|---|---|---|---|---|---|
| `nvidia/Open-SWE-Traces` | **207,489** | CC-BY-4.0 | SWE-agent + OpenHands | MiniMax-M2.5 + Qwen3.5-122B | Yes (`resolved`) | 9 languages. Source tasks: `nebius/SWE-rebench-V2`, permissive upstream only. Best scale-vs-license combo. **Split by the `sweagent`/`openhands` subset to keep action spaces separate.** |
| `nvidia/SWE-Hero-openhands-trajectories` | **34,269** (11,766 issues) | CC-BY-4.0 | OpenHands | Qwen3-Coder-480B | Yes (execution-based) | Execution-verified half of SWE-ZERO→SWE-HERO. |
| `nvidia/SWE-Zero-openhands-trajectories` | **318,115** (118,092 issues) | CC-BY-4.0 | OpenHands | Qwen3-Coder-480B | **No (execution-free)** | Breadth pre-SFT only; never the final SFT stage alone. |
| `nvidia/Nemotron-SFT-SWE-v2` | **256,254** = 46,278 OpenHands traj + **209,976 agentless** | CC-BY-4.0 | OpenHands + agentless | Qwen3-Coder-480B / DeepSeek-R1-0528 | Mixed | The agentless subset is the "skill-prior" data most lists miss (§3). |

## 3 · Agentless / decomposed SFT (the gap in most agentic-only lists)

Pure trajectory imitation skips a decomposition that **is reported to help**: train the sub-skills separately — fault *localization*, *repair*, *test generation* — then layer the agentic loop on top. Kimi-Dev and the Nemotron-SWE line use this as a skill prior. **Likely high ROI, but I'm not citing a controlled ablation here** — treat as "reported to lift," validate on your own eval before committing budget.

- `nvidia/Nemotron-SFT-SWE-v2` → `agentless_swe` split = **209,976** examples (ranked-file localization, patch repair, reproduction+unit-test generation), CC-BY-4.0, DeepSeek-R1-0528 generated.

## 4 · Terminal-Bench — SFT + reward data

| Dataset | Trajectories | License | Scaffold(s) | Exec-verified | Notes |
|---|---|---|---|---|---|
| `m-a-p/TerminalTraj` | **20,000 released** (paper: 50,733 generated *(paper claim)*) | ⚠ none declared | custom Dockerized terminal agent | validation code | Download is 20k, not 50,733 (arXiv:2602.01244). |
| `yoonholee/terminalbench-trajectories` | **52,104** (89 tasks; ~34,462 carry full step traces) | Apache-2.0 | **26 scaffolds** — terminus-2/3, mini-swe-agent, openhands, codex, claude-code… | Yes (`reward` 0/1) | TB 2.0 leaderboard scrape. Reward-modeling goldmine; **filter to one scaffold before imitation.** |
| `Contextbench/Tracebench` | **3,316** (verified split = 1,000) | ⚠ none declared | mini-SWE-agent / OpenHands / Terminus2 / SWE-agent | Yes (`solved` + per-stage labels) | TB + SWE-bench with stage-level failure annotations. Diagnosis/verifier, not bulk SFT. |

## 5 · RL executable environments (you RL against these, not trajectories)

| Resource | Size | Role |
|---|---|---|
| `R2E-Gym/R2E-Gym-V1` | **~8.7K tasks** *(paper claim; V1 card count is version-dependent)* | Largest procedurally-curated SWE env set. |
| `R2E-Gym/R2E-Gym-Subset` | ~4,578 envs (10 repos, non-overlapping with SWE-bench) | **The 4.5k DeepSWE RL'd on.** Decontaminated. |
| `R2E-Gym/R2EGym-SFT-Trajectories` | 3,231 (⚠ no license) | SFT bootstrap traces (Claude-3.5-Sonnet) for the same envs. |
| SWE-Gym (`SWE-Gym/SWE-Gym`, `…-Lite`, `…-Raw`) | ~2.4k instances (+Lite +Raw) | ICML-2025 executable instances + verifier recipe. |
| `nebius/SWE-bench-extra` | 6,411 issue/PR task instances | Task source for building envs. |
| `nebius/SWE-rebench` / `…-V2` | 21k+ continuously-updated, decontaminated tasks | Freshest task pool. |
| `agentica-org/DeepSWE-Preview` + rLLM/SkyRL | weights + RL recipe | RL-only on R2E-Gym → ~42% pass@1, **59% with hybrid TTS** *(model claims)*. ⚠ **Base was Qwen3-32B in *thinking* mode**, not Qwen3-Coder-30B-A3B (non-thinking) — the recipe/lesson transfers, exact behavior and reward dynamics may not. |
| `Skywork/Skywork-SWE-32B` | model + data-scaling paper | 38.0% → **47.0% with TTS** *(model claims)*. |
| **Terminal RL:** `Danau5tin/terminal-bench-rl`, Endless Terminals (~3,255 tasks), LiteCoder-Terminal (602 envs) | — | Binary pass/fail Docker reward. |

## 6 · Verifier / reward-model data

- `SWE-Gym/OpenHands-Verifier-Trajectories` — **5,272** rows (mixture/on-/off-policy), ⚠ no license. LLM-judge prompts + boolean `resolved`. Powers SWE-Gym's verifier-based test-time scaling.
- `nebius/SWE-agent-trajectories` full set — 13,389 `target==true` vs 66,647 false = ready-made positive/negative contrast.
- `yoonholee` `reward` column + `Contextbench/Tracebench` stage labels — terminal-side signal.

## 7 · Secondary / diversity tier

| Dataset | Rows | License | Scaffold | Note |
|---|---|---|---|---|
| `zai-org/SWE-Dev-train` | 20,147 (17.9k SFT + 2.28k RFT) | ⚠ none declared | OpenHands | → SWE-Dev-32B 36.6% Verified *(paper claim)*. |
| `R2E-Gym/R2EGym-SFT-Trajectories` | 3,231 | ⚠ none declared | R2E fn-calling | Repo-to-env diversity. |
| `swesynth/SWE-Synth_Moatless-SFT-Trajectories` | 3,018 | ⚠ none declared | Moatless | Synthetic. |
| `sailplane/swe-agent-trajs` | 2,294 | ⚠ none declared | SWE-agent | Older `.traj` files. |
| `AlienKevin/SWE-ZERO-96K-trajectories` | 96,237 | Apache-2.0 | mini-swe-agent | Execution-free; format imitation only. |

---

## License map (read as risk tiers, not legal advice)

None of this is legal advice; **everything below is subject to attribution, each upstream repo's license, model-output terms, and your own legal review.** CC-BY-4.0 is *not* as clean as MIT/Apache — it carries an attribution obligation and (for NVIDIA/Nebius) a "respect upstream repo license" clause.

- **Most permissive (MIT / Apache-2.0):** `SWE-smith-trajectories` (MIT), `SWE-Gym/OpenHands-SFT-Trajectories` (MIT), `yoonholee/terminalbench-trajectories` (Apache-2.0), `AlienKevin/SWE-ZERO-96K` (Apache-2.0), and the Qwen3-Coder base (Apache-2.0).
- **Commercial-plausible but with strings (CC-BY-4.0):** all `nebius/*`, all `nvidia/*`. Attribution required; NVIDIA's Nemotron card states commercial-readiness but the obligation stands. Nebius additionally flags **Llama 3.1 license applies if you use the model outputs** (some traces are Llama-generated). Re-check upstream repo license per instance before shipping weights.
- **No declared license — treat as research-only until clarified:** `SWE-Gym/OpenHands-Sampled-Trajectories`, `SWE-Gym/OpenHands-Verifier-Trajectories`, `zai-org/SWE-Dev-train`, `R2E-Gym/R2EGym-SFT-Trajectories`, `swesynth/SWE-Synth_Moatless-SFT-Trajectories`, `sailplane/swe-agent-trajs`, `m-a-p/TerminalTraj`, `Contextbench/Tracebench`.

---

## 8 · Base model — `Qwen/Qwen3-Coder-30B-A3B-Instruct` (verified)

| Spec | Value |
|---|---|
| Architecture | `qwen3_moe` MoE, causal LM |
| Total / activated params | **30.5B / 3.3B** |
| Layers | 48 |
| Attention | GQA — 32 Q heads, **4 KV heads** |
| Experts | **128 total / 8 activated** |
| Native context | **262,144 (256K)**, → 1M with YaRN |
| License | **Apache-2.0** |
| Reasoning | **Non-thinking only** — emits no `<think>` blocks |
| Recommended sampling | temp 0.7, top_p 0.8, top_k 20, rep_penalty 1.05 |

- **Non-thinking model** — the SWE-agent / OpenHands / mini-swe-agent corpora are action traces (reason-in-the-open then act), which match natively. Don't SFT on `<think>`-wrapped CoT.
- **Native tool-call format** — Qwen3-Coder ships a *specially designed* XML-style function-call format (vLLM/SGLang expose a `qwen3_coder` tool-call parser). **Re-render every trajectory into this wire syntax before SFT** — the single most-skipped step. Concrete example below.

**Hardware rough cut:** ~61 GB BF16 weights. QLoRA 4-bit fits one 80 GB A100/H100 at 32K context with care; full SFT needs FSDP/DeepSpeed across ≥4×80 GB.

### 8.1 · Concrete tool-call conversion (do this exactly)

**Raw OpenHands-style turn** (as stored in the datasets above):

```json
{ "role": "assistant",
  "content": "The test references add(); let me find the source.",
  "tool_calls": [{ "id": "call_42", "type": "function",
    "function": { "name": "execute_bash",
                  "arguments": "{\"command\": \"grep -rn 'def add' src/\"}" } }] }
{ "role": "tool", "tool_call_id": "call_42",
  "content": "src/calc.py:12:def add(a, b):" }
```

**Rendered into Qwen3-Coder native syntax** (what actually gets tokenized):

```text
<|im_start|>assistant
The test references add(); let me find the source.
<tool_call>
<function=execute_bash>
<parameter=command>grep -rn 'def add' src/</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
src/calc.py:12:def add(a, b):
</tool_response><|im_end|>
```

**Loss mask on that rendered sequence:**

| Span | Loss? |
|---|---|
| system prompt, issue text | **masked** |
| `assistant` turn — natural-language content **and** the entire `<tool_call>…</tool_call>` block | **trained** |
| `user` turn carrying `<tool_response>…</tool_response>` (the environment observation) | **masked** |

Note Qwen3-Coder returns tool output inside a **`user`** turn — so for this model the observation role is `user`, even though the source dataset called it `tool` or `ai`/`user`. Normalize during conversion; mask whatever ends up wrapping `<tool_response>`.

### 8.2 · Base-model swap → `Qwen3.6-35B-A3B`

Newer hybrid MoE: ~35B total / ~3B active, **256 experts (8 routed + 1 shared)**, **Gated DeltaNet linear-attention interleaved 3:1 with Gated Attention**, hybrid **thinking** mode, 262K native context, MTP training, Apache-2.0 (~73.4% SWE-bench Verified / 51.5% Terminal-Bench 2.0 — *model claims*). **Same datasets, same philosophy** — but lighter SFT, stricter filtering, lower LR, more RL/RFT, and re-derived LoRA targets.

**Gate everything on a pre-training benchmark.** Before any SFT, run the *base* on 50–100 SWE-bench-style tasks + 20–30 Terminal-Bench tasks **in your target harness**. If it's already strong (likely), skip heavy SFT → light SFT + RL/RFT. A stronger base has less headroom and is easier to *degrade* with noisy imitation.

| Area | 30B-A3B | 3.6-35B-A3B |
|---|---|---|
| SFT amount | moderate ok | lighter, higher-quality first |
| LR | 1e-4 LoRA | **3e-5 – 7e-5** |
| LoRA rank | 32 / 64 | start **16 / 32**, then scale |
| Data mix | more imitation | more verified + RL/RFT, less imitation noise |
| Exec-free data (SWE-Zero) | low-weight breadth | **drop / heavily downweight** |
| Tool format | `qwen3_coder` parser | re-check 3.6 chat template / tool format |
| Thinking | non-thinking | **decide: preserve thinking, or serve non-thinking** |
| Main risk | teaching agent behavior | degrading a stronger base |

**Updated SFT mix** (one harness-matched action space, converted to Qwen wire syntax):

| Data | Ratio |
|---|---:|
| Clean resolved OpenHands / SWE-agent traces matching your harness | 50% |
| `nvidia/Open-SWE-Traces` + `SWE-Hero` + Nebius OpenHands successful | 25% |
| Agentless skill prior (localization / repair / test-gen) | 15% |
| `TerminalTraj` + passed Terminal-Bench traces | 10% |

*(Carve the 491-trace format set out of the 50%; reserve failed traces for the verifier, not this mix.)*

**MoE LoRA — re-derive targets, do not reuse the 30B-A3B list.** Print `model.named_modules()` first. Only ~1 layer in 4 is standard Gated Attention with `q/k/v/o_proj`; the DeltaNet layers use different projection names — adapt those too or you touch almost nothing. 256 experts + 1 **shared expert** (always active → worth adapting). Freeze the router (`mlp.gate`-equivalent) in run 1; attention-first, then expert MLPs.

**Thinking-mode decision tree:**
- **Serve non-thinking** (simplest): disable thinking, treat 3.6 as a stronger 30B-A3B → §9 recipe transfers as-is; your action-trace data already matches.
- **Preserve thinking**: you need reasoning-augmented traces (scarce) or a synthesized per-step reasoning prefix via rejection sampling. Payoff: DeepSWE's *thinking-mode* RL recipe transfers more directly.

**Shift weight to RL.** On a 73% base the gain isn't coding ability — it's long-horizon loop completion: don't give up, don't loop, don't emit giant patches. After a light format-SFT pass, move faster to **RFT/GRPO on executable envs** (R2E-Gym-Subset, SWE-Gym; terminal via `terminal-bench-rl`) with sparse outcome reward. Re-baseline in your harness and confirm trainer/serving support for the DeltaNet+MTP arch before committing.

## 9 · SFT — pick ONE target, don't blend

The earlier scaffold warning and a single blended mix are in tension. So: **commit to one action space, build that recipe, convert everything to Qwen wire syntax (§8.1).** Three options:

**Recipe A — OpenHands-target (recommended default).** Most data, cleanest verified traces, matches a real eval harness.

| Stage | Data | ~Ratio |
|---|---|---:|
| Format alignment | `SWE-Gym/OpenHands-SFT-Trajectories` (491) | 5% |
| Core verified | `nebius/SWE-rebench-openhands-trajectories` (32k ok) + `nvidia/SWE-Hero` (34k) | 35% |
| Scale | `nvidia/Open-SWE-Traces` **openhands subset** + `Nemotron-SFT-SWE-v2` openhands traj | 30% |
| Skill prior | `Nemotron-SFT-SWE-v2` agentless (210k), capped | 10% |
| Terminal | `yoonholee` filtered to `openhands` scaffold + `TerminalTraj` (converted) | 15% |
| Negative/recovery | failed-but-good-debugging traces | 5% |

**Recipe B — SWE-agent-target.** Use if your harness is SWE-agent / SWE-agent-LM-style.

| Stage | Data | ~Ratio |
|---|---|---:|
| Core verified | `SWE-smith-trajectories` (5,017, one split) + `nebius/SWE-agent-trajectories` (`target==true`, 13,389) | 55% |
| Scale | `nvidia/Open-SWE-Traces` **sweagent subset** | 25% |
| Terminal | `TerminalTraj` + `yoonholee` filtered to mini-swe-agent | 15% |
| Negative/recovery | failed SWE-agent traces | 5% |

**Recipe C — Terminal-agent-target (Terminus / mini-swe-agent).** Use if Terminal-Bench is the priority eval.

| Stage | Data | ~Ratio |
|---|---|---:|
| Core | `m-a-p/TerminalTraj` (20k) | 45% |
| Leaderboard | `yoonholee` filtered to `terminus-2/3` (passed only) | 35% |
| Diagnosis/contrast | `Contextbench/Tracebench` terminal split | 10% |
| SWE crossover | small OpenHands or SWE-agent verified set (converted) | 10% |

**MoE LoRA targets:**

```text
# Run 1 — attention-only (cheap; captures most trajectory-discipline lift):
q_proj, k_proj, v_proj, o_proj
# Run 2 — add expert MLPs only after a clean eval (multiplies adapter size ×128 experts):
gate_proj, up_proj, down_proj
```

⚠ **Do NOT target `mlp.gate`** — that is the MoE *router*. `gate_proj` (expert MLP) is safe; `mlp.gate` destabilizes routing. Keep the router frozen in run 1.

**Hyperparameters:**

| Setting | Recommendation | Note |
|---|---|---|
| Context | start 32K → curriculum 64K/128K | *resolved* nebius traces avg **8.3K tokens**; failed ones balloon to 15K+. 32K captures most good traces. |
| LoRA rank / alpha | 32 or 64 / 64 or 128 | alpha = 2×rank |
| LR | 1e-4 LoRA, 5e-6–1e-5 full | |
| Epochs | 1 over large mix | upsample the 491-trace format set rather than relying on global epoch count |
| Packing | yes, never split a trajectory | use sample/document attention masking so packed traces don't attend across boundaries |
| Loss mask | assistant **action** tokens only | mask observations — see §8.1 |
| Precision | BF16 | |

**Filters** (the `nebius` field is literally `target`):

```text
target/resolved == true       # `target` (bool) in nebius/SWE-agent-trajectories
patch non-empty
eval/test logs passed
steps <= ~60                  # nebius resolved avg = 31
files/lines edited bounded    # resolved avg = 1.3 files / 21 lines
loop-free                     # reuse Nemotron loop_detection metadata
<= 3 trajectories per instance
decontaminate vs SWE-bench Verified (500) + Terminal-Bench eval tasks
```

## 10 · RL / RFT plan

Sequence is correct and matches the evidence: **verifier → best-of-N → GRPO on executable tasks**, and "clean SFT + test-time scaling > raw RL first" is what SWE-Gym (verifier TTS) and DeepSWE (42% → 59% with hybrid TTS) show. Build the verifier from §6.

Two refinements:

1. **Lean sparse, not shaped.** DeepSWE used **outcome-only** reward and beat shaped approaches; subgoal bonuses ("ran tests", "opened relevant file") get gamed. Keep your shaping list as filters/diagnostics, not reward terms, unless you measure a win.
2. **MoE-specific care.** Keep the router frozen in RL too; monitor expert load-balance / collapse. Use §8 sampling params for rollouts. Remember DeepSWE's published behavior came from a **thinking-mode** base — budget for your own tuning on a non-thinking base.

**Reward sources:** patch reward on `R2E-Gym-Subset` (DeepSWE's exact env) + SWE-Gym; terminal binary pass/fail via `terminal-bench-rl` / Endless Terminals.

---

## Bottom line — recommendation for your setup

For **`Qwen3-Coder-30B-A3B-Instruct`**:

1. **Pick one target harness first** — OpenHands tools or Qwen-native tool calling (Recipe A is the safe default).
2. **Convert every trajectory** into that action space **and** into Qwen3-Coder's native function-call syntax (§8.1). Mask observations.
3. **Train attention-only LoRA first**; add expert-MLP LoRA only after a clean eval shows you need it. Never touch `mlp.gate`.
4. **Execution-verified traces for final SFT**, agentless data as **skill prior**, **failed traces reserved for verifier/RL** — not vanilla imitation.
5. **Then** layer verifier + best-of-N test-time scaling; treat broad GRPO as the last step, with sparse outcome reward.

The biggest expected win is **clean, format-correct SFT + a verifier for test-time scaling**, not raw RL — Qwen3-Coder already has strong code priors; you're teaching trajectory discipline (inspect → search → edit → test → submit), not coding ability.
