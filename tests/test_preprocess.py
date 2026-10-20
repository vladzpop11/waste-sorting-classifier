import io

import numpy as np
import pytest
from PIL import Image

from waste_sorting_classifier.preprocess import CROP, load_rgb, preprocess


@pytest.fixture
def photo() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (384, 512, 3), dtype=np.uint8))


@pytest.mark.parametrize("size", [(512, 384), (384, 512), (300, 300), (1000, 333)])
def test_matches_torchvision_eval_transform(size) -> None:
    torchvision = pytest.importorskip("torchvision")
    t = torchvision.transforms
    reference = t.Compose(
        [
            t.Resize(256),
            t.CenterCrop(224),
            t.ToTensor(),
            t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    rng = np.random.default_rng(1)
    image = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
    np.testing.assert_allclose(preprocess(image), reference(image).numpy(), atol=1e-6)


def test_output_shape_and_type(photo) -> None:
    x = preprocess(photo)
    assert x.shape == (3, CROP, CROP) and x.dtype == np.float32


def test_exif_orientation_is_applied(photo) -> None:
    buf = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6  # stored rotated: display needs 90 degrees clockwise
    photo.save(buf, format="JPEG", exif=exif)
    buf.seek(0)
    assert load_rgb(buf).size == (384, 512)


def test_load_rgb_converts_greyscale_and_alpha() -> None:
    for mode in ["L", "RGBA", "P"]:
        buf = io.BytesIO()
        Image.new(mode, (40, 30)).save(buf, format="PNG")
        buf.seek(0)
        assert load_rgb(buf).mode == "RGB"
