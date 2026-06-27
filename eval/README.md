# Evaluation methodology

Measure progress honestly. The number that matters is **resolved% (pass@1) on
SWE-bench Verified, in a fixed harness, reported with that harness named.**

## Rules that keep the number meaningful

1. **Hold the harness fixed.** SWE-bench scores swing materially with the agent
   scaffold. Use the *same* `HARNESS` + `EVAL_LIMIT` for baseline and re-eval, or
   the delta is a harness artifact, not a model gain.
2. **Report pass@1, single-pass.** Verifier / best-of-K numbers (the 47%, 59%
   figures in the research) require running the model 8–16× plus a verifier —
   that is not what you get from a single local request. Track them separately.
3. **Watch contamination.** SWE-bench Verified instances predate the base model
   and may be in pretraining. For a clean signal, also spot-check on
   **SWE-rebench** (continuously mined, dated, contamination-flagged).
4. **Start on a subset.** `EVAL_LIMIT=50` validates the whole loop (harness →
   endpoint → Docker test execution) for a few dollars before committing to the
   full 500.

## Expected numbers (from verified research)

| Stage | SWE-bench Verified (pass@1) |
| --- | --- |
| Base Qwen3-Coder-30B-A3B, in-harness | low single digits → ~mid-teens |
| + QLoRA SFT (~500 traj) | ~20% |
| + QLoRA SFT (~8K traj, data-scaled) | ~38% |
| + local best-of-K verifier | up to ~47% (slow on Mac) |
| Open-weights frontier (reference) | GLM-5.1 ~50.7% |
| Closed frontier (reference) | ~60–63% |

## Two harnesses

- **`HARNESS=openhands`** (default) — matches the Nebius training trajectories;
  the right choice for train↔serve alignment. Heavier to stand up.
- **`HARNESS=mini`** — `mini-swe-agent`, the simplest/cheapest way to a first
  signal. Good for fast iteration; switch to OpenHands for the trajectory-aligned
  numbers you report.

## Cost / time

Each instance = many model calls + a Docker test run. A 50-instance subset is an
afternoon and a few dollars of API (Phase 0) or a couple of GPU-hours (Phase 3).
The full 500 is roughly 10× that.
