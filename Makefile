# Phase runner for the local agentic-coding fine-tune pipeline.
# Switch base model family with BASE=qwen30 (default) or BASE=qwen36.
# Override knobs inline, e.g.:  make baseline EVAL_LIMIT=50  /  make sft BASE=qwen36
.PHONY: help baseline baseline-terminal data data-terminal sft eval eval-terminal quantize serve-cloud serve-mac clean

PYTHON ?= python3
BASE   ?= qwen30

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

# Rebuild the canonical training file: cross-stage dedup + manifest, errors if no stage files.
define build_train
	@PYTHON=$(PYTHON) bash scripts/build_train.sh data/sft_train.jsonl
endef

help:
	@echo "Base family: BASE=qwen30 (default) | qwen36   (current: $(BASE) -> $(BASE_MODEL))"
	@echo "Phases (eval uses MODEL_* from .env; BASE only supplies a fallback model id):"
	@echo "  make baseline           Phase 0  SWE-bench on the base/served model (local endpoint ok)"
	@echo "  make baseline-terminal  Phase 0  Terminal-Bench on the base/served model"
	@echo "  make data               Phase 1  Stage-1 SWE corpus -> sft_resolved + rebuild sft_train"
	@echo "  make data-terminal      Phase 1  Stage-2 terminal corpus -> sft_terminal + rebuild sft_train"
	@echo "  make sft                Phase 2  QLoRA SFT (CONFIG=$(CONFIG))"
	@echo "  make eval               Phase 3  SWE-bench on the fine-tune (point MODEL_* at it)"
	@echo "  make eval-terminal      Phase 3  Terminal-Bench on the fine-tune"
	@echo "  make quantize           Phase 4  Merge -> MLX 4-bit ($(MERGED_DIR))"
	@echo "  make serve-mac          Phase 5  Local MLX OpenAI endpoint ($(MLX_DIR))"
	@echo "  make serve-cloud        Serve via vLLM on the GPU pod ($(BASE_MODEL))"

baseline:
	DEFAULT_MODEL_NAME=$(BASE_MODEL) EVAL_LIMIT=$(EVAL_LIMIT) bash eval/run_swebench_openhands.sh

baseline-terminal eval-terminal:
	DEFAULT_MODEL_NAME=$(BASE_MODEL) TB_LIMIT=$(TB_LIMIT) bash eval/run_terminalbench.sh

data:
	$(PYTHON) data/prepare_swe_sft.py --output data/sft_resolved.jsonl $(DATA_ARGS)
	$(build_train)

data-terminal:
	$(PYTHON) data/prepare_terminal_sft.py --output data/sft_terminal.jsonl $(TERMINAL_ARGS)
	$(build_train)

sft:
	PYTHON=$(PYTHON) CONFIG=$(CONFIG) MERGED_DIR=$(MERGED_DIR) bash train/run_sft.sh

eval:
	DEFAULT_MODEL_NAME=$(BASE_MODEL) EVAL_LIMIT=$(EVAL_LIMIT) bash eval/run_swebench_openhands.sh

quantize:
	PYTHON=$(PYTHON) bash quantize/to_mlx.sh $(MERGED_DIR)

serve-cloud:
	bash serve/serve_vllm.sh $(or $(MODEL),$(BASE_MODEL))

serve-mac:
	PYTHON=$(PYTHON) bash serve/serve_mlx_mac.sh $(MLX_DIR)

clean:
	rm -f data/sft_train.jsonl data/*.manifest.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@echo "cleaned sft_train.jsonl + manifests + __pycache__ (stage files data/sft_{resolved,terminal}.jsonl kept)"
