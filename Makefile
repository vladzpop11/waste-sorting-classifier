IMAGE ?= waste-sorting-classifier:dev
NAME  ?= waste-sorting-classifier
PORT  ?= 8000

.PHONY: help setup format lint test check data split baseline smoke-train train export \
        evaluate fetch-model serve docker-build docker-run docker-stop docker-smoke latency errors

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

CHECKPOINT ?= artifacts/kaggle-2026-10-20/run/efficientnet_b0/model.pt

export:  ## Export CHECKPOINT to models/model.onnx, checked against PyTorch on real photos
	uv run python -m waste_sorting_classifier.export $(CHECKPOINT)

evaluate:  ## One-time test evaluation of the exported model (refuses to run again unless FORCE=1)
	uv run python -m waste_sorting_classifier.evaluate

fetch-model:  ## Download the evaluated ONNX model from the GitHub release (SHA-256 checked)
	uv run python -m waste_sorting_classifier.fetch_model

serve:  ## Run the API locally without Docker
	uv run uvicorn waste_sorting_classifier.api:app --host 127.0.0.1 --port $(PORT)

docker-build:  ## Build the API image (needs models/model.onnx)
	docker build --platform=linux/amd64 -t $(IMAGE) .

docker-run:  ## Start the API container on localhost:$(PORT) (replaces a running one)
	-@docker rm -f $(NAME) >/dev/null 2>&1
	docker run --rm -d --name $(NAME) -p 127.0.0.1:$(PORT):8000 $(IMAGE)

docker-stop:  ## Stop the API container
	docker stop $(NAME)

docker-smoke: docker-run  ## Start the container, check /health and /predict on tests/example.jpg, stop it
	@for i in $$(seq 1 30); do curl -sf localhost:$(PORT)/health >/dev/null && break; sleep 1; done
	curl -sf localhost:$(PORT)/health
	@echo
	curl -sf -X POST localhost:$(PORT)/predict -F file=@tests/example.jpg
	@echo
	docker stop $(NAME)

latency:  ## Measure /predict latency against the running API (writes reports/latency.json)
	uv run python -m waste_sorting_classifier.latency http://localhost:$(PORT)

errors:  ## Validation error gallery of the exported model (reports/figures/val_errors.png)
	uv run python -m waste_sorting_classifier.errors
