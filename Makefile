# Phase runner. Switch model family with FAMILY=qwen30 (default) | qwen36 (BASE= is an alias).
# ALL per-family values come from config/families/<id>.yaml via scripts/family.sh — there are
# zero family-specific lines below, so adding a family is one profile file. See docs/REFACTOR.md.
.PHONY: help config doctor baseline baseline-terminal data data-terminal sft eval eval-terminal quantize serve-cloud serve-mac clean

PYTHON ?= python3
FAMILY ?= qwen30
ifdef BASE
FAMILY := $(BASE)
endif

# Resolve everything from the family profile (single source of truth).
fam = PYTHON=$(PYTHON) bash scripts/family.sh
BASE_MODEL := $(shell $(fam) get $(FAMILY) base_model)
MERGED_DIR := $(shell $(fam) get $(FAMILY) paths.merged_dir)
MLX_DIR    := $(shell $(fam) get $(FAMILY) paths.mlx_dir)
CODEC      := $(shell $(fam) get $(FAMILY) tool_call.codec)
CONFIG     := config/.generated/$(FAMILY).yaml

# Fatal guard: empty BASE_MODEL means the family didn't resolve (unknown id or missing PyYAML).
ifeq ($(strip $(BASE_MODEL)),)
$(error could not resolve FAMILY='$(FAMILY)' — unknown family or missing PyYAML; run: bash scripts/family.sh list)
endif

EVAL_LIMIT ?= 50
TB_LIMIT   ?= 25

define build_train
	@PYTHON=$(PYTHON) bash scripts/build_train.sh data/sft_train.jsonl
endef

help:
	@echo "Family: FAMILY=$(FAMILY) -> $(BASE_MODEL)   (families: $(shell $(fam) list | tr '\n' ' '))"
	@echo "  make config             Generate the Axolotl config from base + family profile"
	@echo "  make doctor             Validate the family profile before GPU spend"
	@echo "  make baseline           Phase 0  SWE-bench on the base/served model (local endpoint ok)"
	@echo "  make baseline-terminal  Phase 0  Terminal-Bench on the base/served model"
	@echo "  make data               Phase 1  Stage-1 SWE corpus -> sft_resolved + rebuild sft_train"
	@echo "  make data-terminal      Phase 1  Stage-2 terminal corpus -> sft_terminal + rebuild sft_train"
	@echo "  make sft                Phase 2  Render config + QLoRA SFT"
	@echo "  make eval               Phase 3  SWE-bench on the fine-tune"
	@echo "  make eval-terminal      Phase 3  Terminal-Bench on the fine-tune"
	@echo "  make quantize           Phase 4  Merge -> MLX 4-bit ($(MLX_DIR))"
	@echo "  make serve-mac          Phase 5  Local MLX OpenAI endpoint ($(MLX_DIR))"
	@echo "  make serve-cloud        Serve via vLLM ($(BASE_MODEL), parser from profile)"

config:
	PYTHONPATH=src $(PYTHON) -m coder_forge.configs.render $(FAMILY)

doctor:
	@$(fam) doctor $(FAMILY)

baseline:
	FAMILY=$(FAMILY) DEFAULT_MODEL_NAME=$(BASE_MODEL) EVAL_LIMIT=$(EVAL_LIMIT) bash eval/run_swebench_openhands.sh

baseline-terminal eval-terminal:
	FAMILY=$(FAMILY) DEFAULT_MODEL_NAME=$(BASE_MODEL) TB_LIMIT=$(TB_LIMIT) bash eval/run_terminalbench.sh

data:
	$(PYTHON) data/prepare_swe_sft.py --output data/sft_resolved.jsonl --codec $(CODEC) $(DATA_ARGS)
	$(build_train)

data-terminal:
	$(PYTHON) data/prepare_terminal_sft.py --output data/sft_terminal.jsonl --codec $(CODEC) $(TERMINAL_ARGS)
	$(build_train)

sft: config doctor
	PYTHON=$(PYTHON) CONFIG=$(CONFIG) MERGED_DIR=$(MERGED_DIR) bash train/run_sft.sh

eval:
	FAMILY=$(FAMILY) DEFAULT_MODEL_NAME=$(BASE_MODEL) EVAL_LIMIT=$(EVAL_LIMIT) bash eval/run_swebench_openhands.sh

quantize:
	PYTHON=$(PYTHON) FAMILY=$(FAMILY) bash quantize/to_mlx.sh $(MERGED_DIR)

serve-cloud:
	FAMILY=$(FAMILY) bash serve/serve_vllm.sh $(or $(MODEL),$(BASE_MODEL))

serve-mac:
	PYTHON=$(PYTHON) bash serve/serve_mlx_mac.sh $(MLX_DIR)

clean:
	rm -f data/sft_train.jsonl data/*.manifest.json
	rm -rf config/.generated
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@echo "cleaned sft_train.jsonl + manifests + generated configs + __pycache__"
