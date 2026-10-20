"""FastAPI service: POST /predict (an image upload) and GET/HEAD /health.

Serving uses ONNX Runtime only (no PyTorch). The model is loaded at startup from MODELS_DIR
(default models/) and refused if model.onnx differs from the SHA-256 in model.json. Uploads
are limited in size and decoded defensively: only JPEG, PNG and WebP, a cap on pixel count
(decompression bombs), EXIF orientation applied. Errors always have the shape
{"detail": [{"type", "loc", "msg"}]} and never echo the upload.
"""

import io
import logging
import os
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
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from waste_sorting_classifier import __version__
from waste_sorting_classifier.config import load_config, project_path
from waste_sorting_classifier.onnx_model import OnnxClassifier
from waste_sorting_classifier.preprocess import load_rgb, preprocess

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 40_000_000  # e.g. 8000 x 5000; larger images are refused before decoding
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class Prediction(BaseModel):
    label: str
    probabilities: dict[str, float]
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
    """Open an upload safely; raise ValueError with a client-safe message otherwise."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            width, height = probe.size
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("the file is not a readable image") from exc
    if fmt not in ALLOWED_FORMATS:
        raise ValueError(f"unsupported image format {fmt}; use JPEG, PNG or WebP")
    if width * height > MAX_PIXELS:
        raise ValueError(f"image too large: {width}x{height} pixels")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            return load_rgb(io.BytesIO(data))
    except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("the image could not be decoded") from exc


def models_dir() -> Path:
    return Path(os.environ.get("MODELS_DIR", project_path(load_config()["paths"]["models_dir"])))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = OnnxClassifier(models_dir(), threads=1)
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
    }


@app.post(
    "/predict",
    responses={
        413: {"description": f"Upload larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"},
        415: {"description": "Not a multipart/form-data upload"},
        422: {"description": "Missing file, or not a readable JPEG/PNG/WebP image"},
    },
)
def predict(
    file: Annotated[UploadFile, File(description="A photo of one waste item")],
) -> Prediction:
    model = app.state.model
    try:
        image = decode_image(file.file.read())
    except ValueError as exc:
        return error(422, "invalid_image", str(exc))
    proba = model.predict_proba(preprocess(image)[np.newaxis])[0]
    return Prediction(
        label=model.classes[int(proba.argmax())],
        probabilities={c: round(float(p), 6) for c, p in zip(model.classes, proba, strict=True)},
        model=model.meta["arch"],
    )
