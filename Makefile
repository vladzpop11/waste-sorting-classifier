IMAGE ?= waste-sorting-classifier:dev
NAME  ?= waste-sorting-classifier
PORT  ?= 8000

.PHONY: help setup format lint test check data split baseline smoke-train train export \
        select evaluate fetch-model serve docker-build docker-run docker-stop docker-smoke latency errors demo-local web check-web

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

select:  ## Apply the selection rule (configs/config.yaml) to the downloaded Kaggle runs
	uv run python -m waste_sorting_classifier.select

export:  ## Export the selected checkpoint to models/model.onnx, checked against PyTorch on real photos
	uv run python -m waste_sorting_classifier.export

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

docker-smoke: docker-run  ## Start the container, check /health, /predict and a refused upload, stop it
	@mkdir -p .cache; trap 'docker stop $(NAME) >/dev/null' EXIT; set -e; \
	for i in $$(seq 1 30); do curl -sf localhost:$(PORT)/health >/dev/null && break; sleep 1; done; \
	curl -sf localhost:$(PORT)/health; echo; \
	curl -sf -X POST localhost:$(PORT)/predict -F file=@tests/example.jpg | tee .cache/smoke.json; echo; \
	python3 -c 'import json,sys; r=json.load(open(".cache/smoke.json")); p=r["probabilities"]; \
	assert r["label"]=="$(SMOKE_LABEL)", r; assert abs(sum(p.values())-1)<1e-4, p; print("label ok")'; \
	code=$$(curl -s -o /dev/null -w '%{http_code}' -X POST localhost:$(PORT)/predict -F file=@README.md); \
	test "$$code" = 422 && echo "non-image refused: $$code"

latency:  ## Measure /predict latency against the running API (writes reports/latency.json)
	uv run python -m waste_sorting_classifier.latency http://localhost:$(PORT)

errors:  ## Validation error gallery of the exported model (reports/figures/val_errors.png)
	uv run python -m waste_sorting_classifier.errors

demo-local:  ## The Gradio demo on localhost (needs make fetch-model)
	uv run --group demo python demo/gradio_app.py

WEB_TEST := .cache/web-test
# tests/example.jpg is plastic1.jpg, a training photo: docker-smoke expects this label.
SMOKE_LABEL := plastic

# What the static Hugging Face Space serves (web/README.md carries its settings).
WEB_SITE := web/index.html web/app.js web/preprocess.js web/image-header.js web/model-info.js \
	web/README.md web/examples web/vendor

web:  ## Prepare web/: pinned ONNX Runtime Web files and model-info.js from models/model.json
	uv run python -m waste_sorting_classifier.web_build

check-web: web  ## Browser pipeline vs Python: validation photos and edge cases (needs Node.js, Chrome)
	uv run python -m waste_sorting_classifier.web_fixtures $(WEB_TEST)/fixtures
	npm install --silent --no-audit --no-fund --prefix $(WEB_TEST) onnxruntime-web@1.30.0 puppeteer-core@24
	node web/check_preprocess.mjs $(WEB_TEST)/fixtures models/model.onnx $(WEB_TEST)/node_modules/onnxruntime-web
	rm -rf $(WEB_TEST)/site && mkdir -p $(WEB_TEST)/site $(WEB_TEST)/shots
	cp -R $(WEB_SITE) models/model.onnx $(WEB_TEST)/site/
	cd $(WEB_TEST)/site && (python3 -m http.server 8765 --bind 127.0.0.1 >/dev/null 2>&1 & echo $$! > ../server.pid)
	sleep 1; NODE_PATH=$(WEB_TEST)/node_modules node web/check_page.cjs http://127.0.0.1:8765/ \
	  $(WEB_TEST)/fixtures data/raw/images $(WEB_TEST)/shots; status=$$?; kill $$(cat $(WEB_TEST)/server.pid); exit $$status
