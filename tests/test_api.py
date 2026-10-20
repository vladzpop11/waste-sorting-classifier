import io
import json

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from waste_sorting_classifier.api import MAX_UPLOAD_BYTES, app
from waste_sorting_classifier.artifacts import sha256
from waste_sorting_classifier.export import export_onnx

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory):
    """A tiny ONNX model with the real input/output contract, and its model.json."""
    torch.manual_seed(0)
    tiny = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, len(CLASSES))
    ).eval()
    out = tmp_path_factory.mktemp("models")
    export_onnx(tiny, out / "model.onnx")
    meta = {"arch": "tiny", "classes": CLASSES, "onnx_sha256": sha256(out / "model.onnx")}
    (out / "model.json").write_text(json.dumps(meta))
    return out


@pytest.fixture
def client(models_dir, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(models_dir))
    with TestClient(app) as c:
        yield c


def image_bytes(fmt: str = "JPEG", size=(160, 120), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, (120, 80, 40) if mode == "RGB" else 1).save(buf, format=fmt)
    return buf.getvalue()


def upload(client, data: bytes, name="photo.jpg", content_type="image/jpeg"):
    return client.post("/predict", files={"file": (name, data, content_type)})


def assert_error(r, status: int) -> None:
    assert r.status_code == status, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, list) and detail
    assert all(set(e) == {"type", "loc", "msg"} for e in detail)


def test_health(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["classes"] == CLASSES
    assert client.head("/health").status_code == 200


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_predict_accepts_common_formats(client, fmt) -> None:
    r = upload(client, image_bytes(fmt), name=f"photo.{fmt.lower()}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] in CLASSES
    assert set(body["probabilities"]) == set(CLASSES)
    assert sum(body["probabilities"].values()) == pytest.approx(1, abs=1e-4)
    assert body["label"] == max(body["probabilities"], key=body["probabilities"].get)


def test_predict_handles_greyscale_and_transparency(client) -> None:
    assert upload(client, image_bytes("PNG", mode="L")).status_code == 200
    assert upload(client, image_bytes("PNG", mode="RGBA")).status_code == 200


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
def test_predict_rejects_other_formats(client, fmt) -> None:
    assert_error(upload(client, image_bytes(fmt)), 422)


def test_predict_rejects_non_images_without_echoing_them(client) -> None:
    secret = b"not an image: do-not-echo-42"
    r = upload(client, secret, name="secret-name.jpg")
    assert_error(r, 422)
    assert "do-not-echo-42" not in r.text and "secret-name" not in r.text


def test_predict_rejects_decompression_bombs(client) -> None:
    buf = io.BytesIO()
    Image.new("1", (9000, 9000)).save(buf, format="PNG")  # 81 megapixels in a few KB
    assert len(buf.getvalue()) < 100_000
    assert_error(upload(client, buf.getvalue(), "bomb.png", "image/png"), 422)


def test_predict_rejects_large_uploads(client) -> None:
    assert_error(upload(client, b"\xff" * (MAX_UPLOAD_BYTES + 10_000)), 413)


def test_predict_requires_multipart(client) -> None:
    r = client.post("/predict", content=image_bytes(), headers={"Content-Type": "image/jpeg"})
    assert_error(r, 415)


def test_predict_requires_the_file_field(client) -> None:
    r = client.post("/predict", files={"photo": ("a.jpg", image_bytes(), "image/jpeg")})
    assert_error(r, 422)


def test_http_errors_use_the_same_shape(client) -> None:
    assert_error(client.get("/nope"), 404)
    assert_error(client.get("/predict"), 405)
    assert client.get("/", follow_redirects=False).headers["location"] == "/docs"


def test_startup_refuses_a_tampered_model(models_dir, tmp_path, monkeypatch) -> None:
    (tmp_path / "model.onnx").write_bytes((models_dir / "model.onnx").read_bytes() + b"x")
    (tmp_path / "model.json").write_text((models_dir / "model.json").read_text())
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="does not match"), TestClient(app):
        pass
