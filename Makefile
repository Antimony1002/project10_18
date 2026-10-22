PYTHON ?= python3
RAW_ROOT ?= data/raw
DATASET_ROOT ?= data/lerobot/towel_folding
CONFIG ?= configs/towel_folding.json
REPO_ID ?= local/towel_folding

export PYTHONPATH := ../src:src:$(PYTHONPATH)

.PHONY: test test-i2rt inspect convert validate overfit train

test:
	$(PYTHON) -m unittest discover -s tests -v

test-i2rt:
	$(PYTHON) -m pytest tests/i2rt_mcap -q

inspect:
	$(PYTHON) scripts/inspect_nedf3.py "$(RAW_ROOT)" --config "$(CONFIG)" --report reports/raw_preflight.json

convert:
	$(PYTHON) scripts/convert_nedf3_to_lerobot.py "$(RAW_ROOT)" "$(DATASET_ROOT)" --config "$(CONFIG)"

validate:
	$(PYTHON) scripts/validate_lerobot_dataset.py "$(DATASET_ROOT)" --repo-id "$(REPO_ID)"

overfit:
	DATASET_ROOT="$(abspath $(DATASET_ROOT))" REPO_ID="$(REPO_ID)" bash scripts/train_overfit.sh

train:
	DATASET_ROOT="$(abspath $(DATASET_ROOT))" REPO_ID="$(REPO_ID)" bash scripts/train_act.sh
