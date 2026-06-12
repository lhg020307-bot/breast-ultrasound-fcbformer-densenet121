from __future__ import annotations
import argparse
import json
import math
import os
import sys
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
WORK_ID = "5268"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SEGMENTATION_SIZE = 352
CLASSIFICATION_SIZE = 224
MASK_THRESHOLD = 0.5
VIEW_MARGIN = 30
FOUR_VIEWS = ("full", "cut_borders", "border", "masked")
LABEL_DISPLAY = {"benign": "良性", "malignant": "恶性"}
FUSION_WEIGHTS = {
    "full": 0.45,
    "cut_borders": 0.30,
    "border": 0.05,
    "masked": 0.20,
}
PLATT_A = 0.8433612027555293
PLATT_B = -0.45303314562164687
CLASSIFICATION_THRESHOLD = 0.42433333333333334
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def first_image_file_from_clipboard(paths: list[str] | tuple[str, ...]) -> Path | None:
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path
    return None


def save_pasted_image(image, project_root: Path = PROJECT_ROOT) -> Path:
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError("clipboard content is not a PIL image")

    output_dir = Path(project_root) / "outputs" / "frontend_clipboard"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"pasted_{stamp}.png"
    image.convert("RGB").save(output_path)
    return output_path


def foreground_bbox(gray: np.ndarray, margin: int = 8) -> tuple[int, int, int, int]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 8, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, gray.shape[1], gray.shape[0]

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(gray.shape[1], x + w + margin)
    y1 = min(gray.shape[0], y + h + margin)
    return x0, y0, x1, y1


def preprocess_ultrasound(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
    median_kernel: int = 3,
    crop_margin: int = 8,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    x0, y0, x1, y1 = foreground_bbox(gray, crop_margin)
    gray = gray[y0:y1, x0:x1]

    p_low, p_high = np.percentile(gray, (1, 99))
    if p_high > p_low:
        gray = np.clip((gray.astype(np.float32) - p_low) * 255.0 / (p_high - p_low), 0, 255)
        gray = gray.astype(np.uint8)

    if median_kernel and median_kernel > 1:
        gray = cv2.medianBlur(gray, median_kernel)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), (x0, y0, x1, y1)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    flood = mask_u8.copy()
    h, w = flood.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return ((mask_u8 | holes) > 0).astype(np.uint8)


def filter_components(mask: np.ndarray, min_area: int = 50, keep_largest: bool = True) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    component_ids = [
        label_id
        for label_id in range(1, num_labels)
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area
    ]
    if not component_ids:
        return np.zeros_like(binary)

    if keep_largest:
        component_ids = [max(component_ids, key=lambda label_id: stats[label_id, cv2.CC_STAT_AREA])]
    return np.isin(labels, component_ids).astype(np.uint8)


def postprocess_mask(mask: np.ndarray, min_area: int = 50, keep_largest: bool = True) -> np.ndarray:
    return filter_components(fill_holes(mask), min_area=min_area, keep_largest=keep_largest)


def border_from_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(binary, kernel, iterations=1)
    return np.clip(dilated - eroded, 0, 1).astype(np.uint8)


def count_components(mask: np.ndarray) -> int:
    binary = (mask > 0).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return int(sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 0))


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
        reason = "empty_mask"
    elif area_ratio < min_area_ratio:
        reason = "area_too_small"
    elif area_ratio > max_area_ratio:
        reason = "area_too_large"
    elif components > max_components:
        reason = "too_many_components"
    else:
        ys, xs = np.where(binary > 0)
        invalid_bbox = len(xs) == 0 or xs.max() <= xs.min() or ys.max() <= ys.min()
        reason = "invalid_bbox" if invalid_bbox else "ok"

    return {
        "is_valid": reason == "ok",
        "area_ratio": area_ratio,
        "num_components": components,
        "reason": reason,
    }


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def center_crop_bbox(width: int, height: int, ratio: float = 0.65) -> tuple[int, int, int, int]:
    crop_w = max(1, int(width * ratio))
    crop_h = max(1, int(height * ratio))
    x0 = max(0, (width - crop_w) // 2)
    y0 = max(0, (height - crop_h) // 2)
    return x0, y0, min(width, x0 + crop_w), min(height, y0 + crop_h)


def generate_classification_views(
    image_bgr: np.ndarray,
    binary_mask: np.ndarray,
    margin: int = VIEW_MARGIN,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    h, w = image_bgr.shape[:2]
    mask = (binary_mask > 0).astype(np.uint8)
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    quality = mask_quality_check(mask)
    bbox = bbox_from_mask(mask) if quality["is_valid"] else None
    if bbox is None:
        bbox = center_crop_bbox(w, h, ratio=0.65)

    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(w, x1 + margin)
    y1 = min(h, y1 + margin)
    crop = image_bgr[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else image_bgr

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    border_img = cv2.cvtColor((gray * border_from_mask(mask)).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    masked_img = cv2.cvtColor((gray * mask).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    return {
        "full": image_bgr,
        "cut_borders": crop,
        "border": border_img,
        "masked": masked_img,
    }, quality


def compute_mask_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    pred = (pred_mask > 0).astype(np.uint8)
    target = (gt_mask > 0).astype(np.uint8)
    if pred.shape != target.shape:
        target = cv2.resize(target, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

    intersection = float((pred & target).sum())
    pred_sum = float(pred.sum())
    target_sum = float(target.sum())
    union = pred_sum + target_sum - intersection
    iou = 1.0 if union == 0 else intersection / union
    dice = 1.0 if pred_sum + target_sum == 0 else (2.0 * intersection) / (pred_sum + target_sum)
    return {"iou": float(iou), "dice": float(dice)}


def find_doctor_mask_paths(image_path: str | Path) -> list[Path]:
    image_path = Path(image_path)
    stem = image_path.stem
    if "_mask" in stem.lower():
        return []

    candidates: list[Path] = []
    for suffix in [image_path.suffix, ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        if not suffix:
            continue
        exact = image_path.with_name(f"{stem}_mask{suffix}")
        if exact.exists():
            candidates.append(exact)
    for path in image_path.parent.glob(f"{stem}_mask*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            candidates.append(path)

    unique = {str(path.resolve()).lower(): path for path in candidates}

    def sort_key(path: Path):
        name = path.stem.lower()
        exact_rank = 0 if name == f"{stem.lower()}_mask" else 1
        return exact_rank, name

    return sorted(unique.values(), key=sort_key)


def load_doctor_mask(
    image_path: str | Path,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray | None, list[Path]]:
    paths = find_doctor_mask_paths(image_path)
    if not paths:
        return None, []

    target_h, target_w = target_shape
    merged = np.zeros((target_h, target_w), dtype=np.uint8)
    loaded_paths: list[Path] = []
    for mask_path in paths:
        mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape != (target_h, target_w):
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        merged |= (mask > 0).astype(np.uint8)
        loaded_paths.append(mask_path)

    if not loaded_paths:
        return None, []
    return merged, loaded_paths


def restore_mask_to_original(
    mask: np.ndarray,
    original_shape: tuple[int, int],
    crop_box: tuple[int, int, int, int],
) -> np.ndarray:
    original_h, original_w = original_shape
    x0, y0, x1, y1 = crop_box
    restored = np.zeros((original_h, original_w), dtype=np.uint8)
    target_w = max(1, x1 - x0)
    target_h = max(1, y1 - y0)
    resized = cv2.resize((mask > 0).astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    restored[y0:y1, x0:x1] = resized[: y1 - y0, : x1 - x0]
    return restored


def overlay_prediction_with_doctor(
    image_bgr: np.ndarray,
    prediction_mask: np.ndarray,
    doctor_mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    pred = (prediction_mask > 0).astype(np.uint8)
    doctor = (doctor_mask > 0).astype(np.uint8)
    if pred.shape != (h, w):
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    if doctor.shape != (h, w):
        doctor = cv2.resize(doctor, (w, h), interpolation=cv2.INTER_NEAREST)

    color = np.zeros_like(image_bgr)
    pred_only = (pred == 1) & (doctor == 0)
    doctor_only = (doctor == 1) & (pred == 0)
    overlap = (pred == 1) & (doctor == 1)
    color[pred_only] = (0, 0, 255)
    color[doctor_only] = (0, 255, 0)
    color[overlap] = (0, 255, 255)

    output = image_bgr.copy()
    mask = pred_only | doctor_only | overlap
    output[mask] = cv2.addWeighted(image_bgr[mask], 1.0 - alpha, color[mask], alpha, 0)
    return output


def prepare_uploaded_mask(
    mask: np.ndarray,
    original_image_shape: tuple[int, int],
    preprocessed_shape: tuple[int, int],
    crop_box: tuple[int, int, int, int],
) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    original_h, original_w = original_image_shape
    if mask.shape == (original_h, original_w):
        x0, y0, x1, y1 = crop_box
        mask = mask[y0:y1, x0:x1]

    target_h, target_w = preprocessed_shape
    if mask.shape != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return postprocess_mask(mask, min_area=50, keep_largest=True)


def weighted_fusion(probabilities: dict[str, float], weights: dict[str, float]) -> float:
    total = sum(float(weights.get(view, 0.0)) for view in FOUR_VIEWS)
    if total <= 0:
        total = float(len(FOUR_VIEWS))
        weights = {view: 1.0 for view in FOUR_VIEWS}

    fused = 0.0
    for view in FOUR_VIEWS:
        fused += (float(weights.get(view, 0.0)) / total) * float(probabilities[view])
    return float(np.clip(fused, 0.0, 1.0))


def apply_platt(probability: float, a: float = PLATT_A, b: float = PLATT_B) -> float:
    clipped = float(np.clip(probability, 1e-9, 1.0 - 1e-9))
    logit = math.log(clipped / (1.0 - clipped))
    return float(1.0 / (1.0 + math.exp(-(float(a) * logit + float(b)))))


def image_metadata_from_path(path: str | Path) -> dict[str, str]:
    path = Path(path)
    parts = list(path.parts)
    filename = path.name or "未知"
    label = "未知"
    dataset = "未知"

    labels = {"benign", "malignant"}
    for index, part in enumerate(parts):
        lower = part.lower()
        if lower in labels:
            label = lower
            if index > 0:
                dataset = parts[index - 1]
            break

    return {"filename": filename, "dataset": dataset, "label": label}


def display_mask_source(mask_source: str) -> str:
    return "用户上传已有 Mask" if mask_source == "uploaded" else "分割模型自动生成"


def display_reference_label(label: str) -> str:
    return LABEL_DISPLAY.get(label.lower(), label)


def prediction_reference_warning(image_path: str | Path, predicted_label: str) -> str:
    metadata = image_metadata_from_path(image_path)
    reference_label = LABEL_DISPLAY.get(metadata["label"].lower())
    if not reference_label or reference_label == predicted_label:
        return ""
    return f"提示：文件标注为{reference_label}，模型预测为{predicted_label}；该样本可能存在域外数据或增强分布偏移。"


def view_probability_interpretation(view: str, probability: float) -> str:
    view_names = {
        "full": "完整图像",
        "cut_borders": "裁剪视图",
        "border": "边界视图",
        "masked": "Mask 区域",
    }
    name = view_names.get(view, view)
    if probability >= 0.85:
        return f"{name}高度支持恶性"
    if probability >= 0.60:
        return f"{name}倾向恶性"
    if probability >= 0.40:
        return f"{name}支持度一般"
    return f"{name}不支持恶性"


def mask_quality_summary(
    quality: dict[str, object],
    metrics: dict[str, float] | None,
) -> dict[str, str]:
    area_ratio = float(quality.get("area_ratio", 0.0) or 0.0)
    components = int(quality.get("num_components", 0) or 0)
    is_valid = bool(quality.get("is_valid", False))

    if not is_valid or area_ratio <= 0.005:
        status = "异常"
    elif components > 3 or area_ratio > 0.60:
        status = "警告"
    elif components == 1 and 0.01 <= area_ratio <= 0.60:
        status = "通过"
    else:
        status = "警告"

    return {
        "status": status,
        "area_ratio_text": f"{area_ratio * 100:.2f}%",
        "components_text": str(components),
        "iou_text": f"{metrics['iou']:.4f}" if metrics else "未计算",
        "dice_text": f"{metrics['dice']:.4f}" if metrics else "未计算",
        "reason_text": "已提供 GT Mask" if metrics else "未提供人工标注 GT Mask",
    }


def model_explanation(view_probabilities: dict[str, float] | None, predicted_label: str) -> str:
    if not view_probabilities:
        return "当前结果基于融合模型输出，未检测到完整四视图概率信息。"

    high_views = [view for view, prob in view_probabilities.items() if prob >= 0.85]
    low_view = min(view_probabilities, key=view_probabilities.get)
    parts = []
    if high_views:
        parts.append(f"{'、'.join(high_views)} 视图高度支持恶性判断")
    else:
        parts.append("四个视图均未给出很高的恶性支持度")
    parts.append(f"{low_view} 视图支持度相对较低")
    if predicted_label == "恶性":
        parts.append("融合结果仍明显倾向恶性")
    else:
        parts.append("融合结果整体倾向良性")
    return "。".join(parts) + "。"


def export_text_report(result: "InferenceResult") -> Path:
    output_dir = Path(result.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "analysis_report.txt"
    metadata = image_metadata_from_path(result.image_path)
    quality = mask_quality_summary(result.mask_quality, result.metrics)
    saved_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    view_lines = [
        f"- {view}: {prob:.4f} | {view_probability_interpretation(view, prob)}"
        for view, prob in result.view_probabilities.items()
    ]
    basic_info_lines = [
        "基本信息",
        f"输入图像: {metadata['filename']}",
        f"Mask 来源: {display_mask_source(result.mask_source)}",
    ]
    if metadata["label"] != "未知":
        basic_info_lines.insert(2, f"文件标注: {display_reference_label(metadata['label'])}")
    warning = prediction_reference_warning(result.image_path, result.predicted_label)
    text = "\n".join(
        [
            "乳腺超声智能分析系统报告",
            f"作品 ID: {result.work_id}",
            f"保存时间: {saved_time}",
            "",
            *basic_info_lines,
            "",
            "最终预测结果",
            f"最终预测: {result.predicted_label}",
            f"良性概率: {result.benign_probability:.2%}",
            f"恶性概率: {result.malignant_probability:.2%}",
            f"判别阈值: {result.threshold:.4f}",
            f"融合概率: {result.raw_fusion_probability:.4f}",
            *(["", "预测提示", warning] if warning else []),
            "",
            "四视图恶性概率",
            *view_lines,
            "",
            "Mask 质量指标",
            f"Mask 质量: {quality['status']}",
            f"病灶面积占比: {quality['area_ratio_text']}",
            f"连通区域数量: {quality['components_text']}",
            f"IoU: {quality['iou_text']}",
            f"Dice: {quality['dice_text']}",
            f"原因: {quality['reason_text']}",
            "",
            "模型解释",
            model_explanation(result.view_probabilities, result.predicted_label),
        ]
    )
    report_path.write_text(text, encoding="utf-8")
    return report_path

def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    color = np.zeros_like(image_bgr)
    color[:, :, 2] = 255
    overlay = image_bgr.copy()
    overlay[binary > 0] = cv2.addWeighted(
        image_bgr[binary > 0],
        1.0 - alpha,
        color[binary > 0],
        alpha,
        0,
    )
    return overlay


@dataclass
class InferenceResult:
    work_id: str
    image_path: str
    mask_source: str
    output_dir: str
    malignant_probability: float
    benign_probability: float
    predicted_label: str
    threshold: float
    raw_fusion_probability: float
    view_probabilities: dict[str, float]
    mask_quality: dict[str, object]
    metrics: dict[str, float] | None
    saved_files: dict[str, str]


class ModelRunner:
    def __init__(self, project_root: Path = PROJECT_ROOT, log: Callable[[str], None] | None = None):
        self.project_root = project_root
        self.log = log or (lambda message: None)
        self.torch = None
        self.device = None
        self.segmentation_model = None
        self.classification_models: dict[str, list[object]] = {}

    def _import_torch(self):
        if self.torch is not None:
            return self.torch
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "当前 Python 环境没有安装 torch/torchvision。请在训练模型使用的环境里运行：python frontend_app.py"
            ) from exc
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch

    def _segmentation_checkpoint(self) -> Path:
        candidates = [
            self.project_root / "outputs" / "models" / "segmentation" / "best.pt",
            self.project_root / "checkpoints" / "pretrained" / "FCBFormer_checkpoint.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError("没有找到分割模型权重 outputs/models/segmentation/best.pt")

    def _load_segmentation_model(self):
        if self.segmentation_model is not None:
            return self.segmentation_model
        torch = self._import_torch()
        sys.path.insert(0, str(self.project_root))
        from models.segmentation import segmentation_model

        checkpoint_path = self._segmentation_checkpoint()
        self.log(f"加载分割模型: {checkpoint_path.name}")
        model = segmentation_model.FCBFormer(size=SEGMENTATION_SIZE)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        self.segmentation_model = model
        return model

    def _classification_checkpoint_paths(self, view: str) -> list[Path]:
        ckpt_dir = self.project_root / "outputs" / "models" / "classification"
        paths = [ckpt_dir / f"cls_{view}_densenet121_fold{fold}_best.pt" for fold in range(5)]
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"分类模型权重缺失: {missing[0]}")
        return paths

    def _build_densenet121(self):
        try:
            from torchvision import models
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError(
                "当前 Python 环境没有安装 torchvision。请在训练模型使用的环境里运行：python frontend_app.py"
            ) from exc
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        return model

    def _load_classification_models(self, view: str) -> list[object]:
        if view in self.classification_models:
            return self.classification_models[view]
        torch = self._import_torch()
        models_for_view = []
        self.log(f"加载分类模型: {view}")
        for checkpoint_path in self._classification_checkpoint_paths(view):
            model = self._build_densenet121()
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.to(self.device)
            model.eval()
            models_for_view.append(model)
        self.classification_models[view] = models_for_view
        return models_for_view

    def _segmentation_tensor(self, image_bgr: np.ndarray):
        torch = self._import_torch()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (SEGMENTATION_SIZE, SEGMENTATION_SIZE), interpolation=cv2.INTER_LINEAR)
        array = image_rgb.astype(np.float32) / 255.0
        array = (array - 0.5) / 0.5
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    def _classification_tensor(self, image_bgr: np.ndarray):
        torch = self._import_torch()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (CLASSIFICATION_SIZE, CLASSIFICATION_SIZE), interpolation=cv2.INTER_LINEAR)
        array = image_rgb.astype(np.float32) / 255.0
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    def predict_mask(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self._import_torch()
        model = self._load_segmentation_model()
        tensor = self._segmentation_tensor(image_bgr)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits)
            probs = torch.nn.functional.interpolate(
                probs,
                size=image_bgr.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
        probability = np.clip(probs.squeeze().detach().cpu().numpy(), 0.0, 1.0)
        binary = (probability >= MASK_THRESHOLD).astype(np.uint8)
        binary = postprocess_mask(binary, min_area=50, keep_largest=True)
        return probability, binary

    def predict_view_probability(self, view: str, image_bgr: np.ndarray) -> float:
        torch = self._import_torch()
        tensor = self._classification_tensor(image_bgr)
        probs = []
        with torch.no_grad():
            for model in self._load_classification_models(view):
                logits = model(tensor).view(-1)
                probs.append(float(torch.sigmoid(logits).cpu().item()))
        return float(np.mean(probs))

    def run(self, image_path: str | Path, mask_path: str | Path | None = None) -> InferenceResult:
        image_path = Path(image_path)
        image_bgr = imread_unicode(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"无法读取图像: {image_path}")

        mask_source = "segmentation"
        if mask_path:
            self.log("使用已上传 mask，跳过增强和分割")
            preprocessed_bgr = image_bgr
            crop_box = (0, 0, image_bgr.shape[1], image_bgr.shape[0])
            uploaded_mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
            if uploaded_mask is None:
                raise RuntimeError(f"无法读取 mask: {mask_path}")
            binary_mask = prepare_uploaded_mask(
                uploaded_mask,
                original_image_shape=image_bgr.shape[:2],
                preprocessed_shape=preprocessed_bgr.shape[:2],
                crop_box=crop_box,
            )
            mask_probability = binary_mask.astype(np.float32)
            mask_source = "uploaded"
        else:
            self.log("预处理与增强图像")
            preprocessed_bgr, crop_box = preprocess_ultrasound(image_bgr)
            self.log("未上传 mask，使用分割模型生成 mask")
            mask_probability, binary_mask = self.predict_mask(preprocessed_bgr)

        views, quality = generate_classification_views(preprocessed_bgr, binary_mask)
        predicted_original_mask = restore_mask_to_original(
            binary_mask,
            original_shape=image_bgr.shape[:2],
            crop_box=crop_box,
        )
        doctor_mask, doctor_mask_paths = load_doctor_mask(image_path, target_shape=image_bgr.shape[:2])
        metrics = compute_mask_metrics(predicted_original_mask, doctor_mask) if doctor_mask is not None else None
        if doctor_mask is not None:
            self.log(f"已检索到医生 Mask: {doctor_mask_paths[0].name}")
            print(f"IoU: {metrics['iou']:.4f}  Dice: {metrics['dice']:.4f}")
            self.log(f"IoU: {metrics['iou']:.4f}  Dice: {metrics['dice']:.4f}")
        else:
            self.log("未检索到对应医生 Mask，IoU/Dice 未计算")

        self.log("分类并融合概率")
        view_probs = {view: self.predict_view_probability(view, views[view]) for view in FOUR_VIEWS}
        raw_fusion = weighted_fusion(view_probs, FUSION_WEIGHTS)
        malignant_prob = apply_platt(raw_fusion, PLATT_A, PLATT_B)
        predicted_label = "恶性" if malignant_prob >= CLASSIFICATION_THRESHOLD else "良性"

        output_dir = self._save_outputs(
            image_path=image_path,
            original_bgr=image_bgr,
            mask_source=mask_source,
            preprocessed_bgr=preprocessed_bgr,
            mask_probability=mask_probability,
            binary_mask=binary_mask,
            predicted_original_mask=predicted_original_mask,
            doctor_mask=doctor_mask,
            views=views,
            view_probs=view_probs,
            raw_fusion=raw_fusion,
            malignant_prob=malignant_prob,
            predicted_label=predicted_label,
            metrics=metrics,
            quality=quality,
        )
        print(f"结果输出文件夹: {output_dir}")
        self.log(f"结果输出文件夹: {output_dir}")

        saved_files = {
            "report": str(output_dir / "analysis_report.txt"),
            "original": str(output_dir / "original.png"),
            "preprocessed": str(output_dir / "preprocessed.png"),
            "predicted_mask": str(output_dir / "predicted_mask_original.png"),
            "comparison_overlay": str(output_dir / "doctor_prediction_overlay.png"),
            "result_json": str(output_dir / "result.json"),
            "view_full": str(output_dir / "view_full.png"),
            "view_cut_borders": str(output_dir / "view_cut_borders.png"),
            "view_border": str(output_dir / "view_border.png"),
            "view_masked": str(output_dir / "view_masked.png"),
        }
        if doctor_mask is not None:
            saved_files["doctor_mask"] = str(output_dir / "doctor_mask.png")

        result = InferenceResult(
            work_id=WORK_ID,
            image_path=str(image_path),
            mask_source=mask_source,
            output_dir=str(output_dir),
            malignant_probability=malignant_prob,
            benign_probability=1.0 - malignant_prob,
            predicted_label=predicted_label,
            threshold=CLASSIFICATION_THRESHOLD,
            raw_fusion_probability=raw_fusion,
            view_probabilities=view_probs,
            mask_quality=quality,
            metrics=metrics,
            saved_files=saved_files,
        )
        export_text_report(result)
        return result

    def _save_outputs(
        self,
        image_path: Path,
        original_bgr: np.ndarray,
        mask_source: str,
        preprocessed_bgr: np.ndarray,
        mask_probability: np.ndarray,
        binary_mask: np.ndarray,
        predicted_original_mask: np.ndarray,
        doctor_mask: np.ndarray | None,
        views: dict[str, np.ndarray],
        view_probs: dict[str, float],
        raw_fusion: float,
        malignant_prob: float,
        predicted_label: str,
        metrics: dict[str, float] | None,
        quality: dict[str, object],
    ) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in image_path.stem)
        output_dir = self.project_root / "outputs" / "frontend" / f"{stamp}_{safe_stem}"
        output_dir.mkdir(parents=True, exist_ok=True)

        imwrite_unicode(output_dir / "original.png", original_bgr)
        imwrite_unicode(output_dir / "preprocessed.png", preprocessed_bgr)
        imwrite_unicode(
            output_dir / "predicted_mask_original.png",
            (predicted_original_mask * 255).astype(np.uint8),
        )
        if doctor_mask is not None:
            imwrite_unicode(output_dir / "doctor_mask.png", (doctor_mask * 255).astype(np.uint8))
            compare_overlay = overlay_prediction_with_doctor(original_bgr, predicted_original_mask, doctor_mask)
        else:
            compare_overlay = overlay_mask(original_bgr, predicted_original_mask)
        imwrite_unicode(output_dir / "doctor_prediction_overlay.png", compare_overlay)

        for view, image in views.items():
            imwrite_unicode(output_dir / f"view_{view}.png", image)

        result = {
            "work_id": WORK_ID,
            "image_path": image_path.name,
            "mask_source": mask_source,
            "predicted_label": predicted_label,
            "benign_probability": 1.0 - malignant_prob,
            "malignant_probability": malignant_prob,
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "raw_fusion_probability": raw_fusion,
            "fusion_weights": FUSION_WEIGHTS,
            "platt": {"a": PLATT_A, "b": PLATT_B},
            "view_probabilities": view_probs,
            "mask_threshold": MASK_THRESHOLD,
            "mask_quality": quality,
            "segmentation_metrics": metrics,
            "doctor_mask_found": doctor_mask is not None,
            "saved_files": {
                "analysis_report": "analysis_report.txt",
                "result_json": "result.json",
                "original": "original.png",
                "preprocessed": "preprocessed.png",
                "predicted_mask": "predicted_mask_original.png",
                "doctor_mask": "doctor_mask.png" if doctor_mask is not None else None,
                "comparison_overlay": "doctor_prediction_overlay.png",
                "view_full": "view_full.png",
                "view_cut_borders": "view_cut_borders.png",
                "view_border": "view_border.png",
                "view_masked": "view_masked.png",
            },
        }
        (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_dir

class FrontendApp:
    def __init__(self):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.colors = {
            "bg": "#F3F6FA",
            "card": "#FFFFFF",
            "ink": "#0F172A",
            "muted": "#64748B",
            "accent": "#2563EB",
            "accent_dark": "#1D4ED8",
            "secondary": "#E5E7EB",
            "secondary_text": "#374151",
            "border": "#E2E8F0",
            "red": "#DC2626",
            "green": "#16A34A",
            "orange": "#F59E0B",
            "image_bg": "#F8FAFC",
        }
        self.font = ("Microsoft YaHei", 10)
        self.root = tk.Tk()
        self.root.title(f"乳腺超声智能分析系统 - 作品ID {WORK_ID}")
        self.root.geometry("1320x860")
        self.root.minsize(1120, 760)
        self.root.configure(bg=self.colors["bg"])

        self.image_path = tk.StringVar()
        self.mask_path = tk.StringVar()
        self.status_text = tk.StringVar(value="请选择或粘贴一张乳腺超声图像。")
        self.file_info_vars = {
            "filename": tk.StringVar(value="未选择"),
            "dataset": tk.StringVar(value="未知"),
            "label": tk.StringVar(value="未知"),
        }
        self.result_vars = {
            "label": tk.StringVar(value="等待预测"),
            "malignant": tk.StringVar(value="--"),
            "benign": tk.StringVar(value="--"),
            "threshold": tk.StringVar(value="--"),
            "fusion": tk.StringVar(value="--"),
            "save": tk.StringVar(value="尚未生成结果"),
        }
        self.mask_vars = {
            "status": tk.StringVar(value="未评估"),
            "area": tk.StringVar(value="--"),
            "components": tk.StringVar(value="--"),
            "iou": tk.StringVar(value="未计算"),
            "dice": tk.StringVar(value="未计算"),
            "reason": tk.StringVar(value="未提供人工标注 GT Mask"),
        }
        self.explanation_text = tk.StringVar(value="完成预测后将自动生成模型解释。")
        self.runner = ModelRunner(log=self._threadsafe_log)
        self.preview_images = {}
        self.debug_logs: list[str] = []
        self.last_result: InferenceResult | None = None
        self.prob_bar_state = {
            "value": 0.0,
            "color": self.colors["secondary"],
            "label": "--",
        }
        self._configure_style()
        self._build_ui()
        self.root.bind_all("<Control-v>", self.paste_image_from_clipboard)

    def _configure_style(self):
        style = self.ttk.Style()
        try:
            style.theme_use("clam")
        except self.tk.TclError:
            pass
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["ink"], font=self.font)
        style.configure("Treeview", rowheight=28, font=self.font)
        style.configure("Treeview.Heading", font=("Microsoft YaHei", 10, "bold"))

    def _card(self, parent, **grid):
        frame = self.tk.Frame(
            parent,
            bg=self.colors["card"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            bd=0,
            padx=8,
            pady=6,
        )
        frame.grid(**grid)
        return frame

    def _label(self, parent, text="", size=10, weight="normal", color=None, bg=None, **grid):
        label = self.tk.Label(
            parent,
            text=text,
            bg=bg or parent.cget("bg"),
            fg=color or self.colors["ink"],
            font=("Microsoft YaHei", size, weight),
            anchor="w",
            justify="left",
        )
        label.grid(**grid)
        return label

    def _button(self, parent, text, command, primary=False, outline=False, **grid):
        if primary:
            bg, fg = self.colors["accent"], "#FFFFFF"
        elif outline:
            bg, fg = self.colors["card"], self.colors["accent"]
        else:
            bg, fg = self.colors["secondary"], self.colors["secondary_text"]
        button = self.tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=self.colors["accent_dark"] if primary else "#D1D5DB",
            activeforeground="#FFFFFF" if primary else fg,
            relief="flat",
            bd=0,
            padx=8,
            pady=2,
            font=("Microsoft YaHei", 8),
            cursor="hand2",
        )
        if outline:
            button.configure(highlightbackground=self.colors["accent"], highlightthickness=1)
        button.grid(**grid)
        return button

    def _build_ui(self):
        tk = self.tk
        ttk = self.ttk
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        header = tk.Frame(root, bg=self.colors["bg"], padx=18, pady=10)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        self._label(header, "乳腺超声智能分析系统", 19, "bold", self.colors["ink"], row=0, column=0, sticky="w")
        self._label(
            header,
            f"作品 ID：{WORK_ID} · 上传原图 · 自动预处理分割 · 四视图 DenseNet121 融合分类",
            10,
            "normal",
            self.colors["muted"],
            row=1,
            column=0,
            sticky="w",
            pady=(5, 0),
        )

        main = tk.Frame(root, bg=self.colors["bg"], padx=18, pady=0)
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        self._build_image_area(main)
        self._build_right_area(main)


    def _build_image_area(self, parent):
        left = self._card(parent, row=0, column=0, sticky="nsew", padx=(0, 16))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        self._label(left, "图像预览", 13, "bold", row=0, column=0, sticky="w", pady=(0, 10))

        image_grid = self.tk.Frame(left, bg=self.colors["card"])
        image_grid.grid(row=1, column=0, sticky="nsew")
        for col in range(2):
            image_grid.columnconfigure(col, weight=1)
        for row in range(2):
            image_grid.rowconfigure(row, weight=1)

        cards = [
            ("input", "原图", "上传的数据集原始乳腺超声图像。"),
            ("doctor_mask", "医生 Mask", "自动检索到的医生/GT 标注 Mask。"),
            ("mask", "预测 Mask", "白色区域表示分割模型预测的病灶候选区域。"),
            ("overlay", "预测/医生 Mask 叠加图", "红色=预测，绿色=医生，黄色=重合区域。"),
        ]
        self.image_labels = {}
        for index, (key, title, desc) in enumerate(cards):
            panel = self._card(
                image_grid,
                row=index // 2,
                column=index % 2,
                sticky="nsew",
                padx=7,
                pady=7,
            )
            panel.columnconfigure(0, weight=1)
            panel.rowconfigure(1, weight=1)
            self._label(panel, title, 11, "bold", row=0, column=0, sticky="w")
            label = self.tk.Label(
                panel,
                bg=self.colors["image_bg"],
                fg=self.colors["muted"],
                text="暂无图像",
                anchor="center",
                font=("Microsoft YaHei", 10),
            )
            label.grid(row=1, column=0, sticky="nsew", pady=(8, 8))
            self._label(panel, desc, 9, "normal", self.colors["muted"], row=2, column=0, sticky="ew")
            self.image_labels[key] = label

    def _build_right_area(self, parent):
        right = self.tk.Frame(parent, bg=self.colors["bg"])
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        for row in range(4):
            right.rowconfigure(row, weight=0)
        right.rowconfigure(4, weight=1)
        self.right_panel = right

        self._build_input_card(right)
        self._build_result_card(right)
        self._build_view_table_card(right)
        self._build_mask_quality_card(right)

    def _build_input_card(self, parent):
        card = self._card(parent, row=0, column=0, sticky="ew", pady=(0, 8))
        for col in range(4):
            card.columnconfigure(col, weight=1)
        self._label(card, "输入", 11, "bold", row=0, column=0, columnspan=4, sticky="w")
        self._label(card, "图像：", 9, "normal", self.colors["muted"], row=1, column=0, sticky="w", pady=(6, 0))
        self._label(card, "", 9, "bold", row=1, column=1, columnspan=3, sticky="w", pady=(6, 0))
        card.grid_slaves(row=1, column=1)[0].configure(textvariable=self.file_info_vars["filename"])
        button_grid = self.tk.Frame(card, bg=self.colors["card"])
        button_grid.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        for col in range(2):
            button_grid.columnconfigure(col, weight=1)
        self._button(button_grid, "选择图片", self.select_image, row=0, column=0, sticky="ew", padx=(0, 4), pady=2)
        self._button(button_grid, "粘贴图片", self.paste_image_from_clipboard, row=0, column=1, sticky="ew", padx=(4, 0), pady=2)
        self._button(button_grid, "开始预测", self.start_inference, primary=True, row=1, column=0, sticky="ew", padx=(0, 4), pady=2)
        self._button(button_grid, "导出报告", self.export_report, outline=True, row=1, column=1, sticky="ew", padx=(4, 0), pady=2)
        self._label(card, "", 8, "normal", self.colors["muted"], row=3, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        card.grid_slaves(row=3, column=0)[0].configure(textvariable=self.status_text)

    def _build_result_card(self, parent):
        card = self._card(parent, row=1, column=0, sticky="ew", pady=(0, 8))
        for col in range(4):
            card.columnconfigure(col, weight=1)
        self._label(card, "分析结果", 12, "bold", row=0, column=0, columnspan=2, sticky="w")
        self.result_badge = self.tk.Label(
            card,
            textvariable=self.result_vars["label"],
            bg=self.colors["secondary"],
            fg=self.colors["secondary_text"],
            font=("Microsoft YaHei", 14, "bold"),
            padx=10,
            pady=4,
        )
        self.result_badge.grid(row=0, column=2, columnspan=2, sticky="e")
        metric_rows = [
            ("恶性概率：", "malignant"),
            ("良性概率：", "benign"),
            ("判别阈值：", "threshold"),
            ("融合概率：", "fusion"),
        ]
        for index, (name, key) in enumerate(metric_rows, start=0):
            row = 1 + index // 2
            col = (index % 2) * 2
            self._label(card, name, 9, "normal", self.colors["muted"], row=row, column=col, sticky="w", pady=(6, 0))
            self._label(card, "", 10 if key in {"malignant", "benign"} else 9, "bold", row=row, column=col + 1, sticky="w", pady=(6, 0))
            card.grid_slaves(row=row, column=col + 1)[0].configure(textvariable=self.result_vars[key])

        self.prob_canvas = self.tk.Canvas(card, height=18, bg=self.colors["card"], bd=0, highlightthickness=0)
        self.prob_canvas.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        self.prob_canvas.bind("<Configure>", lambda event: self._draw_probability_bar())

    def _build_view_table_card(self, parent):
        card = self._card(parent, row=2, column=0, sticky="ew", pady=(0, 8))
        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=1)
        self._label(card, "四视图恶性概率", 11, "bold", row=0, column=0, columnspan=2, sticky="w")
        self.view_table_labels = {}
        for index, view in enumerate(FOUR_VIEWS):
            row = 1 + index // 2
            col = index % 2
            prob_label = self._label(card, f"{view}: --", 9, "normal", row=row, column=col, sticky="w", pady=(6, 0))
            interp_label = self.tk.Label(card, text="", bg=self.colors["card"])
            self.view_table_labels[view] = (prob_label, interp_label)

    def _build_mask_quality_card(self, parent):
        card = self._card(parent, row=3, column=0, sticky="ew")
        for col in range(4):
            card.columnconfigure(col, weight=1)
        self._label(card, "Mask / 保存", 11, "bold", row=0, column=0, columnspan=2, sticky="w")
        self.mask_status_badge = self.tk.Label(
            card,
            textvariable=self.mask_vars["status"],
            bg=self.colors["secondary"],
            fg=self.colors["secondary_text"],
            font=("Microsoft YaHei", 10, "bold"),
            padx=8,
            pady=3,
        )
        self.mask_status_badge.grid(row=0, column=2, columnspan=2, sticky="e")
        rows = [
            ("面积：", "area"),
            ("连通域：", "components"),
        ]
        for index, (name, key) in enumerate(rows):
            row = 1 + index // 2
            col = (index % 2) * 2
            self._label(card, name, 9, "normal", self.colors["muted"], row=row, column=col, sticky="w", pady=(6, 0))
            self._label(card, "", 9, "normal", row=row, column=col + 1, sticky="w", pady=(6, 0))
            card.grid_slaves(row=row, column=col + 1)[0].configure(textvariable=self.mask_vars[key])

        self.metric_widgets = []
        metric_defs = [("IoU：", "iou", 0), ("Dice：", "dice", 2)]
        for name, key, col in metric_defs:
            name_label = self._label(card, name, 9, "normal", self.colors["muted"], row=2, column=col, sticky="w", pady=(6, 0))
            value_label = self._label(card, "", 9, "normal", row=2, column=col + 1, sticky="w", pady=(6, 0))
            value_label.configure(textvariable=self.mask_vars[key])
            name_label.grid_remove()
            value_label.grid_remove()
            self.metric_widgets.extend([name_label, value_label])

        self._label(card, "保存：", 9, "normal", self.colors["muted"], row=3, column=0, sticky="nw", pady=(6, 0))
        self._label(card, "", 8, "normal", self.colors["muted"], row=3, column=1, columnspan=3, sticky="ew", pady=(6, 0))
        card.grid_slaves(row=3, column=1)[0].configure(textvariable=self.result_vars["save"], wraplength=360)

        actions = self.tk.Frame(card, bg=self.colors["card"])
        actions.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        self._button(actions, "打开目录", self.open_output_dir, row=0, column=0, sticky="ew", padx=(0, 4))
        self._button(actions, "导出报告", self.export_report, outline=True, row=0, column=1, sticky="ew", padx=(4, 0))

    def _build_explanation_card(self, parent):
        card = self._card(parent, row=4, column=0, sticky="ew")
        card.columnconfigure(0, weight=1)
        self._label(card, "模型解释与保存信息", 13, "bold", row=0, column=0, sticky="w")
        self._label(card, "", 10, "normal", self.colors["secondary_text"], row=1, column=0, sticky="ew", pady=(10, 8))
        card.grid_slaves(row=1, column=0)[0].configure(textvariable=self.explanation_text, wraplength=440)
        self._label(card, "", 10, "normal", self.colors["muted"], row=2, column=0, sticky="ew")
        card.grid_slaves(row=2, column=0)[0].configure(textvariable=self.result_vars["save"], wraplength=440)
        actions = self.tk.Frame(card, bg=self.colors["card"])
        actions.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        self._button(actions, "打开结果目录", self.open_output_dir, row=0, column=0, sticky="ew", padx=(0, 6))
        self._button(actions, "导出报告", self.export_report, outline=True, row=0, column=1, sticky="ew", padx=(6, 0))

    def _update_file_info(self):
        metadata = image_metadata_from_path(self.image_path.get()) if self.image_path.get() else {
            "filename": "未选择",
            "dataset": "未知",
            "label": "未知",
        }
        self.file_info_vars["filename"].set(metadata["filename"])
        self.file_info_vars["dataset"].set(metadata["dataset"])
        self.file_info_vars["label"].set(metadata["label"])

    def select_image(self):
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.image_path.set(path)
            self.mask_path.set("")
            self._update_file_info()
            self.status_text.set("已选择原图，将自动预处理、分割并分类。")
            self._show_image("input", path)
            self._clear_image("doctor_mask")
            self._clear_image("mask")
            self._clear_image("overlay")

    def paste_image_from_clipboard(self, event=None):
        try:
            from PIL import Image, ImageGrab
        except ImportError:
            self.status_text.set("当前环境缺少 Pillow，无法从剪贴板粘贴图片。")
            return "break"

        clipboard = ImageGrab.grabclipboard()
        image_path = None
        if isinstance(clipboard, Image.Image):
            image_path = save_pasted_image(clipboard, PROJECT_ROOT)
        elif isinstance(clipboard, list):
            image_path = first_image_file_from_clipboard(clipboard)

        if image_path is None:
            self.status_text.set("剪贴板里没有可用图片。请复制图片文件或截图后再粘贴。")
            return "break"

        self.image_path.set(str(image_path))
        self.mask_path.set("")
        self._update_file_info()
        self.status_text.set("已从剪贴板粘贴原图，将自动预处理、分割并分类。")
        self._show_image("input", image_path)
        self._clear_image("doctor_mask")
        self._clear_image("mask")
        self._clear_image("overlay")
        return "break"

    def start_inference(self):
        image_path = self.image_path.get().strip()
        if not image_path:
            self.status_text.set("请先选择或粘贴一张乳腺超声图像。")
            return

        self.mask_path.set("")
        self.last_result = None
        self._reset_result_display()
        self.status_text.set("正在自动生成 Mask 并分类，首次加载模型会比较慢...")
        thread = threading.Thread(target=self._run_inference_thread, daemon=True)
        thread.start()

    def _run_inference_thread(self):
        try:
            result = self.runner.run(
                self.image_path.get().strip(),
                None,
            )
        except Exception as exc:
            details = traceback.format_exc()
            self.root.after(0, lambda: self._show_error(exc, details))
            return
        self.root.after(0, lambda: self._show_result(result))

    def _threadsafe_log(self, message: str):
        self.debug_logs.append(f"{datetime.now():%H:%M:%S} {message}")
        self.root.after(0, lambda: self.status_text.set(message))

    def _show_error(self, exc: Exception, details: str):
        text = str(exc)
        if "mask" in text.lower():
            message = "Mask 生成失败，请检查原图格式、清晰度或模型文件。"
        elif "classification" in text.lower() or "分类" in text:
            message = "分类预测失败，请检查模型文件和输入图像。"
        else:
            message = f"分析失败：{text}"
        self.status_text.set(message)
        self.debug_logs.append(details)
        self.result_vars["label"].set("分析失败")
        self.result_badge.configure(bg=self.colors["orange"], fg="#FFFFFF")

    def _show_result(self, result: InferenceResult):
        self.last_result = result
        self._show_image("input", result.saved_files["original"])
        if result.saved_files.get("doctor_mask"):
            self._show_image("doctor_mask", result.saved_files["doctor_mask"])
        else:
            self._clear_image("doctor_mask", "未找到医生 Mask")
        self._show_image("mask", result.saved_files["predicted_mask"])
        self._show_image("overlay", result.saved_files["comparison_overlay"])
        self._update_file_info()
        self._update_result_card(result)
        self._update_view_table(result)
        self._update_mask_quality(result)
        self.explanation_text.set(model_explanation(result.view_probabilities, result.predicted_label))
        output_name = Path(result.output_dir).name
        self.result_vars["save"].set(f"已保存：{output_name}")
        warning = prediction_reference_warning(result.image_path, result.predicted_label)
        if warning:
            status = warning
        elif result.metrics:
            status = f"分析完成，IoU={result.metrics['iou']:.4f}，Dice={result.metrics['dice']:.4f}，报告已自动保存。"
        else:
            status = "分析完成，未找到医生 Mask，IoU/Dice 未计算，报告已自动保存。"
        self.status_text.set(status)
        self._save_debug_log(result)

    def _reset_result_display(self):
        self.result_vars["label"].set("预测中")
        self.result_vars["malignant"].set("--")
        self.result_vars["benign"].set("--")
        self.result_vars["threshold"].set("--")
        self.result_vars["fusion"].set("--")
        self.result_badge.configure(bg=self.colors["secondary"], fg=self.colors["secondary_text"])
        self.prob_bar_state = {
            "value": 0.0,
            "color": self.colors["secondary"],
            "label": "--",
        }
        self._draw_probability_bar(0.0, self.colors["secondary"], "--")
        for view, labels in getattr(self, "view_table_labels", {}).items():
            labels[0].configure(text=f"{view}: --")
            labels[1].configure(text="暂无四视图概率信息", fg=self.colors["muted"])
        self.mask_vars["status"].set("未评估")
        self.mask_vars["area"].set("--")
        self.mask_vars["components"].set("--")
        self.mask_vars["iou"].set("未计算")
        self.mask_vars["dice"].set("未计算")
        self.mask_vars["reason"].set("未提供人工标注 GT Mask")
        for widget in getattr(self, "metric_widgets", []):
            widget.grid_remove()
        self.explanation_text.set("预测进行中，完成后将自动生成模型解释。")
        self.result_vars["save"].set("尚未生成结果")

    def _update_result_card(self, result: InferenceResult):
        is_malignant = result.predicted_label == "恶性"
        color = self.colors["red"] if is_malignant else self.colors["green"]
        dominant_prob = result.malignant_probability if is_malignant else result.benign_probability
        self.result_vars["label"].set(result.predicted_label)
        self.result_vars["malignant"].set(f"{result.malignant_probability:.2%}")
        self.result_vars["benign"].set(f"{result.benign_probability:.2%}")
        self.result_vars["threshold"].set(f"{result.threshold:.4f}")
        self.result_vars["fusion"].set(f"{result.raw_fusion_probability:.4f}")
        self.result_badge.configure(bg=color, fg="#FFFFFF")
        self.prob_bar_state = {
            "value": dominant_prob,
            "color": color,
            "label": f"{dominant_prob:.2%}",
        }
        self._draw_probability_bar(dominant_prob, color, f"{dominant_prob:.2%}")

    def _draw_probability_bar(self, value=None, color=None, label=None):
        if not hasattr(self, "prob_canvas"):
            return
        self.prob_canvas.delete("all")
        width = max(self.prob_canvas.winfo_width(), 1)
        height = max(self.prob_canvas.winfo_height(), 1)
        if value is None:
            value = self.prob_bar_state["value"]
        else:
            value = float(value)
        color = color or self.prob_bar_state["color"]
        label = label or self.prob_bar_state["label"]
        self.prob_canvas.create_rectangle(0, 0, width, height, fill="#E5E7EB", outline="")
        self.prob_canvas.create_rectangle(0, 0, int(width * max(0.0, min(1.0, value))), height, fill=color, outline="")
        self.prob_canvas.create_text(width // 2, height // 2, text=label, fill=self.colors["ink"], font=("Microsoft YaHei", 9, "bold"))

    def _update_view_table(self, result: InferenceResult):
        if not result.view_probabilities:
            for view, (prob_label, interp_label) in self.view_table_labels.items():
                prob_label.configure(text=f"{view}: --")
                interp_label.configure(text="暂无四视图概率信息", fg=self.colors["muted"])
            return
        for view in FOUR_VIEWS:
            prob = float(result.view_probabilities.get(view, 0.0))
            prob_label, interp_label = self.view_table_labels[view]
            prob_label.configure(text=f"{view}: {prob:.4f}")
            interp_label.configure(text=view_probability_interpretation(view, prob), fg=self.colors["ink"])

    def _update_mask_quality(self, result: InferenceResult):
        summary = mask_quality_summary(result.mask_quality, result.metrics)
        status = summary["status"]
        color = {"通过": self.colors["green"], "警告": self.colors["orange"], "异常": self.colors["red"]}.get(status, self.colors["secondary"])
        self.mask_vars["status"].set(status)
        self.mask_vars["area"].set(summary["area_ratio_text"])
        self.mask_vars["components"].set(summary["components_text"])
        self.mask_vars["iou"].set(summary["iou_text"])
        self.mask_vars["dice"].set(summary["dice_text"])
        self.mask_vars["reason"].set(summary["reason_text"])
        for widget in getattr(self, "metric_widgets", []):
            if result.metrics:
                widget.grid()
            else:
                widget.grid_remove()
        self.mask_status_badge.configure(bg=color, fg="#FFFFFF")

    def _save_debug_log(self, result: InferenceResult):
        return

    def export_report(self):
        if self.last_result is None:
            self.status_text.set("请先完成一次预测，再导出报告。")
            return
        try:
            report_path = export_text_report(self.last_result)
        except Exception as exc:
            self.status_text.set(f"报告导出失败：{exc}")
            return
        self.status_text.set(f"报告已导出：{report_path.name}")

    def open_output_dir(self):
        if self.last_result is None:
            self.status_text.set("暂无结果目录可打开。")
            return
        try:
            os.startfile(str(Path(self.last_result.output_dir)))
        except OSError as exc:
            self.status_text.set(f"无法打开结果目录：{exc}")

    def _clear_image(self, key: str, text: str = "暂无图像"):
        self.preview_images.pop(key, None)
        self.image_labels[key].configure(image="", text=text)

    def _show_image(self, key: str, path: str | Path):
        from PIL import Image, ImageTk

        image = Image.open(path).convert("RGB")
        image.thumbnail((540, 300), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.preview_images[key] = photo
        self.image_labels[key].configure(image=photo, text="")

    def run(self):
        self.root.mainloop()


def run_headless(args: argparse.Namespace) -> None:
    runner = ModelRunner(log=print)
    result = runner.run(args.image, None)
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-image frontend for FCBFormer + DenseNet121 inference.")
    parser.add_argument("--image", default=None, help="Run headless inference for one image instead of opening GUI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image:
        run_headless(args)
    else:
        FrontendApp().run()


if __name__ == "__main__":
    main()
