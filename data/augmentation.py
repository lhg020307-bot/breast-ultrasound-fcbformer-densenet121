"""Classification augmentation adapted from Compete/promax/data/transforms.py.

Operates on numpy (H, W, C) uint8 [0, 255] RGB arrays to stay consistent
with three-classify's cv2-based preprocessing pipeline.
"""

from typing import Any

import cv2
import numpy as np

DEFAULT_AUG_PARAMS: dict[str, Any] = {
    "hflip_prob": 0.5,
    "rotate_degrees": 10,
    "scale_limit": 0.08,
    "shift_limit": 0.04,
    "brightness_limit": 0.12,
    "contrast_limit": 0.12,
    "gamma_limit": 0.10,
    "speckle_std": 0.025,
    "blur_prob": 0.08,
}


def _maybe_hflip(image: np.ndarray, prob: float) -> np.ndarray:
    if np.random.random() >= prob:
        return image
    return np.ascontiguousarray(image[:, ::-1])


def _affine(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    h, w = image.shape[:2]
    angle = np.random.uniform(
        -float(params.get("rotate_degrees", 0)),
        float(params.get("rotate_degrees", 0)),
    )
    scale_limit = float(params.get("scale_limit", 0))
    scale = 1.0 + np.random.uniform(-scale_limit, scale_limit)
    shift_limit = float(params.get("shift_limit", 0))
    tx = np.random.uniform(-shift_limit, shift_limit) * w
    ty = np.random.uniform(-shift_limit, shift_limit) * h
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    matrix[:, 2] += [tx, ty]
    return cv2.warpAffine(
        image, matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _intensity(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    arr = image.astype(np.float32) / 255.0
    brightness = np.random.uniform(
        -float(params.get("brightness_limit", 0)),
        float(params.get("brightness_limit", 0)),
    )
    contrast = 1.0 + np.random.uniform(
        -float(params.get("contrast_limit", 0)),
        float(params.get("contrast_limit", 0)),
    )
    arr = (arr - 0.5) * contrast + 0.5 + brightness
    gamma_limit = float(params.get("gamma_limit", 0))
    if gamma_limit > 0:
        gamma = 1.0 + np.random.uniform(-gamma_limit, gamma_limit)
        arr = np.power(np.clip(arr, 0.0, 1.0), gamma)
    speckle_std = float(params.get("speckle_std", 0))
    if speckle_std > 0:
        noise = np.random.normal(0.0, speckle_std, size=arr.shape).astype(np.float32)
        arr = arr + arr * noise
    if np.random.random() < float(params.get("blur_prob", 0)):
        arr = cv2.GaussianBlur(arr, (3, 3), 0)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def augment_classification(
    image: np.ndarray,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply classification augmentations to a numpy (H,W,3) uint8 RGB image."""
    if params is None:
        params = DEFAULT_AUG_PARAMS
    if not params.get("enabled", True):
        return image
    image = _maybe_hflip(image, float(params.get("hflip_prob", 0)))
    image = _affine(image, params)
    image = _intensity(image, params)
    return image
