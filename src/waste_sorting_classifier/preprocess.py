"""Image preprocessing shared by training, export, the API and the demo, without PyTorch.

It reproduces torchvision's standard evaluation transform exactly (tested): resize the
shorter side to 256 with bilinear filtering, centre-crop 224x224, scale to [0, 1] and
normalise with the ImageNet mean and standard deviation.

Before that, every image is brought to 8-bit RGB the same way everywhere:
- turned upright from its EXIF orientation (phones often store photos rotated);
- 16-bit greyscale scaled to 8 bits (Pillow's plain conversion would clip it to white);
- transparency flattened onto white (instead of exposing the hidden colour of transparent
  pixels, which differs between tools).
web/preprocess.js mirrors all of this for the browser demo.
"""

from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps

RESIZE = 256
CROP = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Inputs outside these limits are refused: very large images cost memory and time, and very
# elongated ones explode when the shorter side is scaled to 256 (a 40000x1 image would
# become 10,240,000 x 256).
MAX_PIXELS = 25_000_000
MAX_ASPECT = 10


def check_size(width: int, height: int) -> None:
    if width < 1 or height < 1:
        raise ValueError("the image is empty")
    if width * height > MAX_PIXELS:
        raise ValueError(f"image too large: {width}x{height} pixels (limit {MAX_PIXELS:,})")
    if max(width, height) / min(width, height) > MAX_ASPECT:
        raise ValueError(f"image too elongated: {width}x{height} (limit {MAX_ASPECT}:1)")


def to_rgb(image: Image.Image) -> Image.Image:
    """Any Pillow image -> 8-bit RGB, with the conventions described in the module docstring."""
    if image.mode in ("I;16", "I;16L", "I;16B", "I;16N", "I"):
        values = np.asarray(image, dtype=np.uint32) >> 8  # 16-bit range -> 8-bit
        image = Image.fromarray(np.clip(values, 0, 255).astype(np.uint8), mode="L")
    has_alpha = image.mode in ("RGBA", "LA", "PA", "RGBa", "La") or (
        image.mode == "P" and "transparency" in image.info
    )
    if has_alpha:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, rgba).convert("RGB")
    return image.convert("RGB")


def load_rgb(source: str | Path | BinaryIO) -> Image.Image:
    with Image.open(source) as image:
        return to_rgb(ImageOps.exif_transpose(image))


def resize_shorter_side(image: Image.Image, size: int = RESIZE) -> Image.Image:
    w, h = image.size
    if w <= h:
        new_w, new_h = size, int(size * h / w)
    else:
        new_w, new_h = int(size * w / h), size
    return image.resize((new_w, new_h), Image.Resampling.BILINEAR)


def center_crop(image: Image.Image, size: int = CROP) -> Image.Image:
    w, h = image.size
    top = int(round((h - size) / 2.0))
    left = int(round((w - size) / 2.0))
    return image.crop((left, top, left + size, top + size))


def to_array(image: Image.Image) -> np.ndarray:
    """RGB image -> normalised float32 array, channels first (3, H, W)."""
    x = np.asarray(image, dtype=np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1).copy()


def preprocess(image: Image.Image) -> np.ndarray:
    """The evaluation transform: (3, 224, 224) float32. Refuses out-of-limit images."""
    check_size(*image.size)
    return to_array(center_crop(resize_shorter_side(image)))
