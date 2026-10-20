# syntax=docker/dockerfile:1
# Base images are pinned by digest; bump them deliberately.
ARG PYTHON_IMAGE=python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336

FROM ${UV_IMAGE} AS uv

# --- build stage: runtime dependencies only (ONNX Runtime, FastAPI, Pillow, NumPy) -----
FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-default-groups --no-install-project

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups

# --- runtime stage: no uv, no PyTorch -----------------------------------------------
FROM ${PYTHON_IMAGE}
RUN useradd --create-home --uid 1000 app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/app/models
WORKDIR /app

COPY --from=build /app /app
COPY configs ./configs
# The exported model (make export, or make fetch-model) and its record.
COPY models/model.onnx models/model.json ./models/

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"]

CMD ["uvicorn", "waste_sorting_classifier.api:app", "--host", "0.0.0.0", "--port", "8000", \
     "--limit-concurrency", "32", "--timeout-keep-alive", "5"]
