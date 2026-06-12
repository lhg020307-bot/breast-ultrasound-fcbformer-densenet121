import argparse
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_CLASSES = ["benign", "malignant"]
DATASET_ALIASES = {
    "BUS": ("BUS", "BUS"),
    "BUSBRA": ("BUS", "BUSBRA"),
    "BUSI": ("BUSI", "BUSI"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Crop images around generated segmentation masks.")
    parser.add_argument(
        "--dataset",
        choices=["BUSBRA", "BUSI", "BUS", "ALL"],
        default=None,
        help="Dataset shortcut. BUSBRA reads images from BUS but reads/writes masks under BUSBRA.",
    )
    parser.add_argument(
        "--image-root",
        default=str(PROJECT_ROOT / "images" / "preprocessed" / "full_image"),
    )
    parser.add_argument(
        "--mask-root",
        default=str(PROJECT_ROOT / "images" / "finetuned" / "masks"),
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "images" / "finetuned" / "cut_borders"),
    )
    parser.add_argument("--folders", nargs="+", default=None)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--margin", type=int, default=30)
    parser.add_argument("--min-area", "--min_area", type=int, default=50, dest="min_area")
    parser.add_argument("--keep-largest", "--keep_largest", action="store_true", default=True)
    parser.add_argument("--no-keep-largest", action="store_false", dest="keep_largest")
    parser.add_argument("--postprocess", action="store_true", default=True)
    parser.add_argument("--no-postprocess", action="store_false", dest="postprocess")
    parser.add_argument(
        "--empty-action",
        choices=["center_crop", "copy", "skip"],
        default="center_crop",
        help="Fallback behavior when the mask is empty after thresholding/postprocessing.",
    )
    parser.add_argument("--fallback-scale", type=float, default=0.8)
    return parser.parse_args()


def resolve_tasks(args):
    if args.dataset:
        if args.dataset == "ALL":
            return [DATASET_ALIASES["BUSBRA"], DATASET_ALIASES["BUSI"]]
        return [DATASET_ALIASES[args.dataset]]

    folders = args.folders or ["BUS", "BUSI"]
    return [(folder, folder) for folder in folders]


def resolve_class_dir(root, dataset, label):
    root = Path(root)
    dataset_dir = root / dataset / label
    if dataset_dir.exists():
        return dataset_dir
    direct_dir = root / label
    if direct_dir.exists():
        return direct_dir
    return dataset_dir


def imread_color(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imread_gray(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def imwrite_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def iter_images(folder):
    for image_path in sorted(folder.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def find_mask_file(mask_dir, image_path):
    stem = image_path.stem
    candidates = [
        mask_dir / image_path.name,
        mask_dir / f"{stem}.png",
        mask_dir / f"{stem}.npy",
        mask_dir / f"{stem}.jpg",
        mask_dir / f"{stem}.jpeg",
        mask_dir / f"{stem}.bmp",
        mask_dir / f"{stem}.tif",
        mask_dir / f"{stem}.tiff",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_mask_probability(path):
    if path.suffix.lower() == ".npy":
        mask = np.load(path).astype(np.float32)
        return np.squeeze(mask)

    mask = imread_gray(path)
    if mask is None:
        return None
    mask = mask.astype(np.float32)
    if mask.size and mask.max() > 1.0:
        mask /= 255.0
    return mask


def filter_components(mask, min_area, keep_largest):
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    component_ids = [
        label_id
        for label_id in range(1, num_labels)
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area
    ]
    if not component_ids:
        return np.zeros_like(mask)
    if keep_largest:
        component_ids = [
            max(component_ids, key=lambda label_id: stats[label_id, cv2.CC_STAT_AREA])
        ]
    return np.isin(labels, component_ids).astype(np.uint8)


def center_crop(image, scale):
    h, w = image.shape[:2]
    scale = min(max(scale, 0.05), 1.0)
    crop_h = max(1, int(round(h * scale)))
    crop_w = max(1, int(round(w * scale)))
    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    return image[y0 : y0 + crop_h, x0 : x0 + crop_w]


def bbox_from_mask(mask, margin, image_shape):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    h, w = image_shape[:2]
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(w, int(xs.max()) + margin + 1)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(h, int(ys.max()) + margin + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def fallback_image(image, action, scale):
    if action == "copy":
        return image
    if action == "center_crop":
        return center_crop(image, scale)
    return None


def main():
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    image_root = Path(args.image_root)
    mask_root = Path(args.mask_root)
    output_root = Path(args.output_root)
    tasks = resolve_tasks(args)

    print(f"Threshold: {args.threshold:.4f}")
    print(
        f"Postprocess: {args.postprocess}, keep_largest={args.keep_largest}, "
        f"min_area={args.min_area}, empty_action={args.empty_action}"
    )

    total = 0
    empty_count = 0
    missing_count = 0
    for input_dataset, output_dataset in tasks:
        for label in args.classes:
            image_dir = resolve_class_dir(image_root, input_dataset, label)
            mask_dir = resolve_class_dir(mask_root, output_dataset, label)
            save_dir = output_root / output_dataset / label

            if not image_dir.exists():
                print(f"WARNING: skipping missing image folder: {image_dir}")
                continue
            if not mask_dir.exists():
                print(f"WARNING: skipping missing mask folder: {mask_dir}")
                continue

            save_dir.mkdir(parents=True, exist_ok=True)
            count = 0
            for image_path in iter_images(image_dir):
                mask_file = find_mask_file(mask_dir, image_path)
                if mask_file is None:
                    print(f"WARNING: missing mask for image: {image_path}")
                    missing_count += 1
                    continue

                image = imread_color(image_path)
                mask_prob = read_mask_probability(mask_file)
                if image is None or mask_prob is None:
                    print(f"WARNING: unreadable image or mask: {image_path} | {mask_file}")
                    continue

                if mask_prob.shape != image.shape[:2]:
                    print(
                        f"WARNING: resizing mask from {mask_prob.shape} to {image.shape[:2]} "
                        f"for {mask_file}"
                    )
                    mask_prob = cv2.resize(
                        mask_prob,
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )

                mask = (mask_prob >= args.threshold).astype(np.uint8)
                if args.postprocess:
                    mask = filter_components(
                        mask, min_area=args.min_area, keep_largest=args.keep_largest
                    )

                bbox = bbox_from_mask(mask, margin=args.margin, image_shape=image.shape)
                if bbox is None:
                    empty_count += 1
                    print(f"WARNING: empty mask after thresholding, fallback={args.empty_action}: {mask_file}")
                    cropped = fallback_image(image, args.empty_action, args.fallback_scale)
                    if cropped is None:
                        continue
                else:
                    x0, y0, x1, y1 = bbox
                    cropped = image[y0:y1, x0:x1]

                imwrite_image(save_dir / image_path.with_suffix(".png").name, cropped)
                count += 1

            total += count
            print(f"Created {count} crops for {image_dir} -> {save_dir}")

    print(
        f"Done. Created {total} crops. Missing masks: {missing_count}. "
        f"Empty-mask fallbacks: {empty_count}."
    )


if __name__ == "__main__":
    main()
