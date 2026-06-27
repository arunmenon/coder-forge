# Phase runner for the local agentic-coding fine-tune pipeline.
# Switch base model family with BASE=qwen30 (default) or BASE=qwen36.
# Override knobs inline, e.g.:  make baseline EVAL_LIMIT=50  /  make sft BASE=qwen36
.PHONY: help baseline baseline-terminal data sft eval eval-terminal quantize serve-cloud serve-mac

BASE ?= qwen30

ifeq ($(BASE),qwen36)
  CONFIG     ?= config/qwen36_35b_a3b_qlora.stub.yaml
  BASE_MODEL ?= Qwen/Qwen3.6-35B-A3B
  MERGED_DIR ?= outputs/qwen36-35b-a3b-merged
  MLX_DIR    ?= models/qwen36-35b-a3b-mlx-4bit
else
  CONFIG     ?= config/qwen3_coder_30b_qlora.yaml
  BASE_MODEL ?= Qwen/Qwen3-Coder-30B-A3B-Instruct
  MERGED_DIR ?= outputs/qwen3-coder-30b-a3b-merged
  MLX_DIR    ?= models/qwen3-coder-30b-a3b-mlx-4bit
endif

EVAL_LIMIT ?= 50
TB_LIMIT   ?= 25

help:
	@echo "Base family: BASE=qwen30 (default) | qwen36   (current: $(BASE) -> $(BASE_MODEL))"
	@echo "Phases:"
	@echo "  make baseline           Phase 0  SWE-bench Verified on the BASE model (local endpoint ok)"
	@echo "  make baseline-terminal  Phase 0  Terminal-Bench on the BASE model"
	@echo "  make data               Phase 1  Download + filter Nebius trajectories -> SFT jsonl"
	@echo "  make sft                Phase 2  QLoRA SFT (CONFIG=$(CONFIG))"
	@echo "  make eval               Phase 3  SWE-bench on the fine-tune (set MODEL_* to it)"
	@echo "  make eval-terminal      Phase 3  Terminal-Bench on the fine-tune"
	@echo "  make quantize           Phase 4  Merge -> MLX 4-bit (run on the Mac)"
	@echo "  make serve-mac          Phase 5  Local MLX OpenAI endpoint ($(MLX_DIR))"
	@echo "  make serve-cloud        Serve via vLLM on the GPU pod ($(BASE_MODEL))"

baseline:
	EVAL_LIMIT=$(EVAL_LIMIT) bash eval/run_swebench_openhands.sh

baseline-terminal eval-terminal:
	TB_LIMIT=$(TB_LIMIT) bash eval/run_terminalbench.sh

data:
	python data/prepare_nebius_sft.py --output data/sft_resolved.jsonl $(DATA_ARGS)

sft:
	CONFIG=$(CONFIG) MERGED_DIR=$(MERGED_DIR) bash train/run_sft.sh

eval:
	EVAL_LIMIT=$(EVAL_LIMIT) bash eval/run_swebench_openhands.sh

quantize:
	bash quantize/to_mlx.sh $(MERGED_DIR)

serve-cloud:
	bash serve/serve_vllm.sh $(or $(MODEL),$(BASE_MODEL))

serve-mac:
	bash serve/serve_mlx_mac.sh $(MLX_DIR)
