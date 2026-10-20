IMAGE ?= waste-sorting-classifier:dev
NAME  ?= waste-sorting-classifier
PORT  ?= 8000

.PHONY: help setup format lint test check data split baseline smoke-train train

help:  ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

setup:  ## Install dependencies from uv.lock
	uv sync --locked

format:  ## Format and auto-fix lint
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Ruff lint and format check
	uv run ruff check .
	uv run ruff format --check .

test:  ## Run the test suite
	uv run pytest

check: lint test  ## Lint + tests

data:  ## Download TrashNet (SHA-256 and class counts checked)
	uv run python -m waste_sorting_classifier.data

split:  ## Recompute the grouped split (normally not needed: splits/split.csv is committed)
	uv run python -m waste_sorting_classifier.split

baseline:  ## Majority-class and linear-probe baselines on the validation split
	uv run python -m waste_sorting_classifier.baseline

smoke-train:  ## One epoch on 2 classes x 25 images on the CPU, to catch errors cheaply
	uv run python -m waste_sorting_classifier.train --output-dir artifacts/smoke \
	  --epochs 1 --limit-per-class 25 --classes glass paper

train:  ## Full training (meant for a GPU; see kaggle/train.ipynb)
	uv run python -m waste_sorting_classifier.train --output-dir artifacts/run
