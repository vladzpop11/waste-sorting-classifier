"""FastAPI service: POST /predict (an image upload) and GET/HEAD /health.

Serving uses ONNX Runtime only (no PyTorch). The model is loaded at startup from MODELS_DIR
(default models/) and refused if model.onnx differs from the SHA-256 in model.json. Uploads
are limited in size and decoded defensively: only JPEG (incl. phone MPO), PNG and WebP;
pixel-count and aspect-ratio caps checked from the header before decoding; at most two
images processed at once; EXIF orientation applied. Errors always have the shape
{"detail": [{"type", "loc", "msg"}]} and never echo the upload.
"""

import io
import logging
import os
import threading
import warnings
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from waste_sorting_classifier import __version__
from waste_sorting_classifier.config import load_config, project_path
from waste_sorting_classifier.familiarity import Familiarity
from waste_sorting_classifier.onnx_model import OnnxClassifier, softmax
from waste_sorting_classifier.preprocess import check_size, load_rgb, preprocess

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
# MPO is how Pillow reports the multi-frame JPEGs many phone cameras write.
ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP"}
# Decoding and inference hold a full-size image in memory: at most this many at once.
MAX_CONCURRENT = 2
BUSY_TIMEOUT_SECONDS = 30
_slots = threading.BoundedSemaphore(MAX_CONCURRENT)


class ImageRejected(ValueError):
    """An upload refused with a message that is safe to show the client."""


class Prediction(BaseModel):
    label: str = Field(description="The most probable class")
    probabilities: dict[str, float]
    familiar: bool = Field(
        description="Whether the photo resembles the training photos (one waste item on a plain "
        "background). When false, the label is not reliable: the model has never seen photos "
        "like this one and has to pick one of its six classes anyway."
    )
    familiarity_score: float = Field(
        description="How much the photo's features resemble the training photos' (minus the "
        "Mahalanobis distance to the nearest class); familiar means at least the threshold "
        "reported by /health"
    )
    model: str


def error(status: int, type_: str, msg: str, loc: str = "body") -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"detail": [{"type": type_, "loc": [loc], "msg": msg}]}
    )


class UploadGuard:
    """ASGI middleware for POST /predict: multipart content type and a hard size limit."""

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/predict":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        content_type = headers.get(b"content-type", b"").split(b";")[0].strip().lower()
        if content_type != b"multipart/form-data":
            msg = "send the image as multipart/form-data with a field named 'file'"
            return await error(415, "unsupported_media_type", msg)(scope, receive, send)
        declared = headers.get(b"content-length", b"0")
        if not declared.isdigit() or int(declared) > MAX_UPLOAD_BYTES + 4096:
            return await self._too_large(scope, receive, send)

        body, more = b"", True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body += message.get("body", b"")
            more = message.get("more_body", False)
            if len(body) > MAX_UPLOAD_BYTES + 4096:  # allowance for multipart framing
                return await self._too_large(scope, receive, send)

        replayed = False

        async def replay() -> dict:
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)

    @staticmethod
    async def _too_large(scope: dict, receive: Callable, send: Callable) -> None:
        msg = f"upload larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
        await error(413, "upload_too_large", msg)(scope, receive, send)


def decode_image(data: bytes) -> Image.Image:
    """Open an upload safely. Size and shape are checked from the header, before decoding.

    Raises ImageRejected with a client-safe message; library errors are logged, not echoed.
    """
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            width, height = probe.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageRejected("the file is not a readable image") from exc
    if fmt not in ALLOWED_FORMATS:
        raise ImageRejected("unsupported image format; use JPEG, PNG or WebP")
    try:
        check_size(width, height)
    except ValueError as exc:
        raise ImageRejected(str(exc)) from exc
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            return load_rgb(io.BytesIO(data))
    except Exception as exc:  # any decoder failure: log the detail, show a generic message
        logger.info("upload could not be decoded: %r", exc)
        raise ImageRejected("the image could not be decoded") from exc


def models_dir() -> Path:
    return Path(os.environ.get("MODELS_DIR", project_path(load_config()["paths"]["models_dir"])))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = OnnxClassifier(models_dir(), threads=1)
    app.state.familiarity = Familiarity(models_dir(), app.state.model.meta)
    yield


app = FastAPI(
    title="Waste sorting classifier", version=__version__, lifespan=lifespan, redirect_slashes=False
)
app.add_middleware(UploadGuard)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    detail = [
        {"type": e["type"], "loc": [str(part)[:64] for part in e["loc"]], "msg": e["msg"]}
        for e in exc.errors()[:20]
    ]
    return JSONResponse(status_code=422, content={"detail": detail})


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unexpected error")
    return error(500, "internal_error", "the server could not process the request")


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    response = error(exc.status_code, "http_error", str(exc.detail), loc="path")
    response.headers.update(exc.headers or {})
    return response


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health")
@app.head("/health", include_in_schema=False)
def health() -> dict:
    model = app.state.model
    return {
        "status": "ok",
        "model": model.meta["arch"],
        "classes": model.classes,
        "onnx_sha256": model.meta["onnx_sha256"],
        "familiarity_threshold": app.state.familiarity.threshold,
    }


@app.post(
    "/predict",
    responses={
        413: {"description": f"Upload larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"},
        415: {"description": "Not a multipart/form-data upload"},
        422: {"description": "Missing file; not a readable JPEG/PNG/WebP; too large or elongated"},
        503: {"description": "Too many images being processed; retry shortly"},
    },
)
def predict(
    file: Annotated[UploadFile, File(description="A photo of one waste item")],
) -> Prediction:
    model = app.state.model
    if not _slots.acquire(timeout=BUSY_TIMEOUT_SECONDS):
        return error(503, "busy", "the server is busy; try again shortly")
    try:
        image = decode_image(file.file.read())
        logits, features = model.run(preprocess(image)[np.newaxis])
    except ImageRejected as exc:
        return error(422, "invalid_image", str(exc))
    finally:
        _slots.release()
    proba = softmax(logits)[0]
    score = float(app.state.familiarity.score(logits, features)[0])
    return Prediction(
        label=model.classes[int(proba.argmax())],
        probabilities={c: round(float(p), 6) for c, p in zip(model.classes, proba, strict=True)},
        familiar=score >= app.state.familiarity.threshold,
        familiarity_score=round(score, 4),
        model=model.meta["arch"],
    )
