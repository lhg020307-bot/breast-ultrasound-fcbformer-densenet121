import random
from typing import Sequence

import cv2
import numpy as np
import torchvision.transforms.functional as TF


class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


# ── Mask utilities (ported from Compete/promax/utils/mask_ops.py) ──


def _component_stats(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    binary = (mask > 0).astype(np.uint8)
    return cv2.connectedComponentsWithStats(binary, connectivity=8)


def count_components(mask: np.ndarray) -> int:
    num_labels, _, stats, _ = _component_stats(mask)
    return int(sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 0))


def border_from_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(binary, kernel, iterations=1)
    return np.clip(dilated - eroded, 0, 1).astype(np.uint8)


def mask_quality_check(
    mask: np.ndarray,
    min_area_ratio: float = 0.0005,
    max_area_ratio: float = 0.65,
    max_components: int = 8,
) -> dict[str, object]:
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape[:2]
    total = max(1, h * w)
    area = int(binary.sum())
    area_ratio = float(area / total)
    components = count_components(binary)

    if area == 0:
        return {"is_valid": False, "area_ratio": area_ratio, "num_components": components, "reason": "empty_mask"}
    if area_ratio < min_area_ratio:
        return {"is_valid": False, "area_ratio": area_ratio, "num_components": components, "reason": "area_too_small"}
    if area_ratio > max_area_ratio:
        return {"is_valid": False, "area_ratio": area_ratio, "num_components": components, "reason": "area_too_large"}
    if components > max_components:
        return {"is_valid": False, "area_ratio": area_ratio, "num_components": components, "reason": "too_many_components"}

    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or xs.max() <= xs.min() or ys.max() <= ys.min():
        return {"is_valid": False, "area_ratio": area_ratio, "num_components": components, "reason": "invalid_bbox"}
    return {"is_valid": True, "area_ratio": area_ratio, "num_components": components, "reason": "ok"}
