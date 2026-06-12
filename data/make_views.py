"""Generate 4-view dataset (full / cut_borders / border / masked) from
preprocessed images and FCBFormer predicted masks.

Reads preprocessed full images and predicted masks, then produces per-sample
views under images/finetuned/views/ and writes outputs/results/views_index.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(PROJECT_ROOT))

from models.utils import border_from_mask, mask_quality_check

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-root",
        default=str(PROJECT_ROOT / "images" / "preprocessed" / "full_image"),
    )
    parser.add_argument(
        "--mask-root",
        default=str(PROJECT_ROOT / "images" / "finetuned" / "masks"),
    )
    parser.add_argument(
        "--view-output-root",
        default=str(PROJECT_ROOT / "images" / "finetuned" / "views"),
    )
    parser.add_argument(
        "--index-output",
        default=str(PROJECT_ROOT / "outputs" / "results" / "views_index.csv"),
    )
    parser.add_argument("--margin", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    success, encoded = cv2.imencode(ext, image)
    if not success:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def _load_mask(mask_path, image_shape):
    if not mask_path.exists():
        return np.zeros(image_shape, dtype=np.uint8)
    mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(image_shape, dtype=np.uint8)
    if mask.shape != image_shape:
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def _bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _center_crop_bbox(w, h, ratio=0.65):
    crop_w = max(1, int(w * ratio))
    crop_h = max(1, int(h * ratio))
    x1 = max(0, (w - crop_w) // 2)
    y1 = max(0, (h - crop_h) // 2)
    return x1, y1, min(w, x1 + crop_w), min(h, y1 + crop_h)


def generate_views(image_path, mask_path, view_output_root, margin):
    image = imread_unicode(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return None

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = _load_mask(mask_path, (h, w))
    quality = mask_quality_check(mask)
    is_valid = bool(quality["is_valid"])

    bbox = _bbox_from_mask(mask) if is_valid else None
    if bbox is None:
        bbox = _center_crop_bbox(w, h, ratio=0.65)
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(w, x1 + margin)
    y1 = min(h, y1 + margin)
    crop = image[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else image

    border = border_from_mask(mask)
    border_img = (gray * border).astype(np.uint8)
    border_img = cv2.cvtColor(border_img, cv2.COLOR_GRAY2BGR)

    masked_img = (gray * mask).astype(np.uint8)
    masked_img = cv2.cvtColor(masked_img, cv2.COLOR_GRAY2BGR)

    sample_id = image_path.stem
    view_dirs = {
        "full": view_output_root / "full",
        "cut_borders": view_output_root / "cut_borders",
        "border": view_output_root / "border",
        "masked": view_output_root / "masked",
    }
    imwrite_unicode(view_dirs["full"] / f"{sample_id}.png", image)
    imwrite_unicode(view_dirs["cut_borders"] / f"{sample_id}.png", crop)
    imwrite_unicode(view_dirs["border"] / f"{sample_id}.png", border_img)
    imwrite_unicode(view_dirs["masked"] / f"{sample_id}.png", masked_img)

    return {
        "sample_id": sample_id,
        "dataset": str(image_path.parent.parent.name),
        "label_dir": str(image_path.parent.name),
        "full_path": str(view_dirs["full"] / f"{sample_id}.png"),
        "cut_borders_path": str(view_dirs["cut_borders"] / f"{sample_id}.png"),
        "border_path": str(view_dirs["border"] / f"{sample_id}.png"),
        "masked_path": str(view_dirs["masked"] / f"{sample_id}.png"),
        "label": 1 if image_path.parent.name.lower() == "malignant" else 0,
        "mask_quality_flag": "ok" if is_valid else "fallback",
        "mask_quality_reason": str(quality["reason"]),
    }


def main():
    args = parse_args()
    image_root = Path(args.image_root)
    mask_root = Path(args.mask_root)
    view_output_root = Path(args.view_output_root)
    index_path = Path(args.index_output)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for folder in sorted(image_root.iterdir()):
        if not folder.is_dir():
            continue
        dataset_name = folder.name
        mask_folder = mask_root / dataset_name
        for label_dir in sorted(folder.iterdir()):
            if not label_dir.is_dir():
                continue
            label_name = label_dir.name
            mask_label_dir = mask_folder / label_name

            image_paths = sorted(
                p for p in label_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if args.max_samples:
                image_paths = image_paths[: args.max_samples]

            for image_path in image_paths:
                candidates = [
                    mask_label_dir / image_path.name,
                    mask_label_dir / f"{image_path.stem}.png",
                ]
                mask_path = next((c for c in candidates if c.exists()), None)
                if mask_path is None:
                    print(f"WARNING: no mask for {image_path}")
                    continue

                row = generate_views(image_path, mask_path, view_output_root, args.margin)
                if row:
                    rows.append(row)

    fields = [
        "sample_id", "dataset", "label_dir", "label",
        "full_path", "cut_borders_path", "border_path", "masked_path",
        "mask_quality_flag", "mask_quality_reason",
    ]
    with index_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} views -> {view_output_root}")
    print(f"Views index: {index_path}")


if __name__ == "__main__":
    main()
