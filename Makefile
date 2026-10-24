# invasive-spread-forecast pipeline. Requires uv (https://docs.astral.sh/uv/) and GBIF
# credentials in the environment or .env (see .env.example).
PY := uv run python
CONFIG ?= configs/default.yaml
SPLIT_YEAR ?= 2019
export ISF_CONFIG := $(CONFIG)

.PHONY: all setup data ingest covariates clean grid panel bias figures features train evaluate holdout dashboard test lint

all: data features train evaluate

setup:
	uv sync

## Phase 1: data acquisition
ingest:
	$(PY) -m src.ingest.gbif

covariates:
	$(PY) -m src.ingest.covariates

## Phase 2: cleaning, gridding, cell x window panel, bias correction
clean: ingest covariates
	$(PY) -m src.preprocess.clean

grid: covariates
	$(PY) -m src.preprocess.grid

panel: clean grid
	$(PY) -m src.preprocess.dataset_builder --window year
	$(PY) -m src.preprocess.dataset_builder --window half

bias: panel
	$(PY) -m src.features.covariates_join
	$(PY) -m src.preprocess.bias --split-year $(SPLIT_YEAR)

figures: bias
	$(PY) -m src.viz.plots
	$(PY) -m src.viz.maps

data: figures

## Phases 3-6 (not implemented yet)
features:
	$(PY) -m src.features.build_features

train: features
	$(PY) -m src.models.train

evaluate:
	$(PY) -m src.evaluation.evaluate

# Final 2026 out-of-sample check. Run ONCE after Phase 5 is frozen (refuses to re-run).
holdout:
	$(PY) -m src.evaluation.holdout

dashboard:
	@echo "'$@' belongs to Phase 6, which is not implemented yet." && exit 1

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .
