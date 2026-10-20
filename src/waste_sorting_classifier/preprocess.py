"""Image preprocessing shared by training, export, the API and the demo, without PyTorch.

It reproduces torchvision's standard evaluation transform exactly (tested): resize the
shorter side to 256 with bilinear filtering, centre-crop 224x224, scale to [0, 1] and
normalise with the ImageNet mean and standard deviation. Photos are first turned upright
from their EXIF orientation, as phones often store them rotated.
"""

from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps

RESIZE = 256
CROP = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_rgb(source: str | Path | BinaryIO) -> Image.Image:
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


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
    """The evaluation transform: (3, 224, 224) float32."""
    return to_array(center_crop(resize_shorter_side(image)))
