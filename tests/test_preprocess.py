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


def test_16_bit_greyscale_is_scaled_not_clipped() -> None:
    values = np.linspace(0, 65535, 64 * 64).reshape(64, 64).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(values).save(buf, format="PNG")
    buf.seek(0)
    rgb = np.asarray(load_rgb(buf))
    assert rgb.min() == 0 and rgb.max() == 255
    assert len(np.unique(rgb)) > 200  # a full grey ramp, not a few clipped values


def test_transparency_is_flattened_onto_white() -> None:
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)
    rgba[..., 0] = 200  # hidden red under fully transparent pixels
    rgba[5:, :, 3] = 255  # bottom half opaque red
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    buf.seek(0)
    rgb = np.asarray(load_rgb(buf))
    assert (rgb[0, 0] == [255, 255, 255]).all()
    assert (rgb[9, 9] == [200, 0, 0]).all()


@pytest.mark.parametrize("size", [(40000, 1), (1, 40000), (311, 31), (6000, 5000)])
def test_out_of_limit_images_are_refused_before_resizing(size) -> None:
    with pytest.raises(ValueError, match="too"):
        preprocess(Image.new("RGB", size))


def test_images_at_the_limits_are_accepted() -> None:
    assert preprocess(Image.new("RGB", (310, 31))).shape == (3, 224, 224)  # 10:1 exactly
    assert preprocess(Image.new("RGB", (5000, 5000))).shape == (3, 224, 224)  # 25 MP
