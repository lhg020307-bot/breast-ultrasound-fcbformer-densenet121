import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess breast ultrasound images for segmentation/classification."
    )
    parser.add_argument("--image-root", default=str(PROJECT_ROOT / "images" / "full_image"))
    parser.add_argument("--gt-mask-root", default=str(PROJECT_ROOT / "images" / "gt_masks"))
    parser.add_argument(
        "--output-image-root",
        default=str(PROJECT_ROOT / "images" / "preprocessed" / "full_image"),
    )
    parser.add_argument(
        "--output-gt-mask-root",
        default=str(PROJECT_ROOT / "images" / "preprocessed" / "gt_masks"),
    )
    parser.add_argument("--clip-limit", type=float, default=2.0)
    parser.add_argument("--tile-grid-size", type=int, default=8)
    parser.add_argument("--median-kernel", type=int, default=3)
    parser.add_argument("--crop-margin", type=int, default=8)
    return parser.parse_args()


def foreground_bbox(gray, margin):
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


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, image):
    path = Path(path)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        return False
    encoded.tofile(str(path))
    return True


def preprocess_image(image, clip_limit, tile_grid_size, median_kernel, crop_margin):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x0, y0, x1, y1 = foreground_bbox(gray, crop_margin)
    gray = gray[y0:y1, x0:x1]

    p_low, p_high = np.percentile(gray, (1, 99))
    if p_high > p_low:
        gray = np.clip((gray.astype(np.float32) - p_low) * 255.0 / (p_high - p_low), 0, 255)
        gray = gray.astype(np.uint8)

    if median_kernel and median_kernel > 1:
        gray = cv2.medianBlur(gray, median_kernel)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    gray = clahe.apply(gray)

    image_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return image_rgb, (x0, y0, x1, y1)


def process_images(args):
    image_root = Path(args.image_root)
    output_root = Path(args.output_image_root)
    count = 0

    for image_path in sorted(image_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = image_path.relative_to(image_root)
        output_path = output_root / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image = imread_unicode(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        processed, _ = preprocess_image(
            image,
            args.clip_limit,
            args.tile_grid_size,
            args.median_kernel,
            args.crop_margin,
        )
        imwrite_unicode(output_path, processed)
        count += 1

    print(f"Preprocessed {count} ultrasound images -> {output_root}")


def process_gt_masks(args):
    image_root = Path(args.image_root)
    mask_root = Path(args.gt_mask_root)
    output_root = Path(args.output_gt_mask_root)
    if not mask_root.exists():
        print(f"Skipping missing GT mask root: {mask_root}")
        return

    count = 0
    for mask_path in sorted(mask_root.rglob("*")):
        if not mask_path.is_file() or mask_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = mask_path.relative_to(mask_root)
        image_path = image_root / relative
        output_path = output_root / relative
        if not image_path.exists():
            print(f"Skipping GT mask without matching image: {mask_path}")
            continue

        image = imread_unicode(image_path, cv2.IMREAD_COLOR)
        mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            print(f"Skipping unreadable image or mask: {image_path} | {mask_path}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x0, y0, x1, y1 = foreground_bbox(gray, args.crop_margin)
        mask = mask[y0:y1, x0:x1]
        mask = (mask > 0).astype(np.uint8) * 255

        output_path.parent.mkdir(parents=True, exist_ok=True)
        imwrite_unicode(output_path, mask)
        count += 1

    print(f"Preprocessed {count} ground-truth masks -> {output_root}")


def main():
    args = parse_args()
    process_images(args)
    process_gt_masks(args)


if __name__ == "__main__":
    main()
