"""PyTorch Dataset for per-view classification with numpy-space augmentation."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.augmentation import augment_classification

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ViewClassificationDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        view: str,
        image_size: int = 224,
        augment: bool = False,
        aug_params: dict | None = None,
    ):
        self.records = records
        self.view = view
        self.image_size = image_size
        self.augment = augment
        self.aug_params = aug_params

        self._path_key = {
            "full": "full_path",
            "cut_borders": "cut_borders_path",
            "border": "border_path",
            "masked": "masked_path",
        }[view]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image_path = Path(record[self._path_key])

        data = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augment:
            image = augment_classification(image, self.aug_params)

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        return {
            "image": tensor,
            "label": int(record["label"]),
            "sample_id": str(record["sample_id"]),
            "dataset": str(record.get("dataset", "")),
        }
