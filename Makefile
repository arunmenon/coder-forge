# Phase runner for the local agentic-coding fine-tune pipeline.
# Override knobs inline, e.g.:  make baseline EVAL_LIMIT=50  /  make sft CONFIG=...
.PHONY: help baseline data sft eval quantize serve-cloud serve-mac

EVAL_LIMIT ?= 50
MERGED_DIR ?= outputs/qwen3-coder-30b-a3b-merged
MLX_DIR    ?= models/qwen3-coder-30b-a3b-mlx-4bit

help:
	@echo "Phases:"
	@echo "  make baseline     Phase 0  SWE-bench Verified on the BASE model (hosted API ok)"
	@echo "  make data         Phase 1  Download + filter Nebius trajectories -> SFT jsonl"
	@echo "  make sft          Phase 2  QLoRA SFT on a single RunPod GPU"
	@echo "  make eval         Phase 3  Re-run the SAME eval on the fine-tune (set MODEL_* to it)"
	@echo "  make quantize     Phase 4  Merge -> MLX 4-bit (run on the Mac)"
	@echo "  make serve-mac    Phase 5  Local MLX OpenAI endpoint on the Mac"
	@echo "  make serve-cloud  Serve a model via vLLM on the GPU pod (arg: MODEL=...)"

baseline:
	EVAL_LIMIT=$(EVAL_LIMIT) bash eval/run_swebench_openhands.sh

data:
	python data/prepare_nebius_sft.py --output data/sft_resolved.jsonl $(DATA_ARGS)

sft:
	bash train/run_sft.sh

eval:
	EVAL_LIMIT=$(EVAL_LIMIT) bash eval/run_swebench_openhands.sh

quantize:
	bash quantize/to_mlx.sh $(MERGED_DIR)

serve-cloud:
	bash serve/serve_vllm.sh $(MODEL)

serve-mac:
	bash serve/serve_mlx_mac.sh $(MLX_DIR)
