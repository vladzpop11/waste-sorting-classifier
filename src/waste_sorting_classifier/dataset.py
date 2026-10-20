"""Images from the committed split. Evaluation uses exactly the serving preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from waste_sorting_classifier.preprocess import CROP, MEAN, STD, load_rgb, preprocess

TRAIN_TRANSFORM = T.Compose(
    [
        T.RandomResizedCrop(CROP, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),  # photos are taken from above: no natural "up"
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(MEAN.tolist(), STD.tolist()),
    ]
)


class WasteImages(Dataset):
    def __init__(self, rows: pd.DataFrame, images_dir: Path, classes: list[str], train: bool):
        self.paths = [images_dir / p for p in rows["path"]]
        self.targets = np.array([classes.index(lab) for lab in rows["label"]])
        self.train = train

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        image = load_rgb(self.paths[i])
        x = TRAIN_TRANSFORM(image) if self.train else torch.from_numpy(preprocess(image))
        return x, int(self.targets[i])
