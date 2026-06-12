import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.segmentation import segmentation_model

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_CLASSES = ["benign", "malignant"]
DATASET_ALIASES = {
    "BUS": ("BUS", "BUS"),
    "BUSBRA": ("BUS", "BUSBRA"),
    "BUSI": ("BUSI", "BUSI"),
}
DEFAULT_CHECKPOINT_PATHS = [
    PROJECT_ROOT / "outputs" / "models" / "segmentation" / "best.pt",
    PROJECT_ROOT / "checkpoints" / "pretrained" / "FCBFormer_checkpoint.pt",
    PROJECT_ROOT / "FCBFormer_checkpoint.pt",
]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate FCBFormer probability maps and binary masks.")
    parser.add_argument(
        "--dataset",
        choices=["BUSBRA", "BUSI", "BUS", "ALL"],
        default=None,
        help="Dataset shortcut. BUSBRA reads images from BUS but writes outputs under BUSBRA.",
    )
    parser.add_argument(
        "--input-root",
        default=str(PROJECT_ROOT / "images" / "preprocessed" / "full_image"),
        help="Root containing dataset/class image folders.",
    )
    parser.add_argument(
        "--output-root",
        "--mask-output-root",
        dest="mask_output_root",
        default=str(PROJECT_ROOT / "images" / "finetuned" / "masks"),
        help="Root where binary mask PNGs are saved.",
    )
    parser.add_argument(
        "--prob-output-root",
        default=str(PROJECT_ROOT / "images" / "finetuned" / "probs"),
        help="Root where raw probability .npy files are saved.",
    )
    parser.add_argument(
        "--prob-png-output-root",
        default=str(PROJECT_ROOT / "images" / "finetuned" / "prob_pngs"),
        help="Root where visual probability PNGs are saved.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--folders", nargs="+", default=None)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--size", type=int, default=352)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-probs", "--save_probs", action="store_true", dest="save_probs")
    parser.add_argument("--no-save-prob-png", action="store_true")
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--min-area", "--min_area", type=int, default=50, dest="min_area")
    parser.add_argument("--keep-largest", "--keep_largest", action="store_true", dest="keep_largest")
    return parser.parse_args()


def resolve_checkpoint(path):
    if path:
        checkpoint = Path(path)
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    for checkpoint in DEFAULT_CHECKPOINT_PATHS:
        if checkpoint.exists():
            return checkpoint
    raise FileNotFoundError("Could not find a segmentation checkpoint.")


def resolve_tasks(args):
    if args.dataset:
        if args.dataset == "ALL":
            return [DATASET_ALIASES["BUSBRA"], DATASET_ALIASES["BUSI"]]
        return [DATASET_ALIASES[args.dataset]]

    folders = args.folders or ["BUS", "BUSI"]
    return [(folder, folder) for folder in folders]


def imread_color(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def fill_holes(mask):
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    flood = mask_u8.copy()
    h, w = flood.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return ((mask_u8 | holes) > 0).astype(np.uint8)


def filter_components(mask, min_area, keep_largest):
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    component_ids = list(range(1, num_labels))
    component_ids = [
        label_id for label_id in component_ids if stats[label_id, cv2.CC_STAT_AREA] >= min_area
    ]
    if not component_ids:
        return np.zeros_like(mask)

    if keep_largest:
        component_ids = [
            max(component_ids, key=lambda label_id: stats[label_id, cv2.CC_STAT_AREA])
        ]

    filtered = np.isin(labels, component_ids)
    return filtered.astype(np.uint8)


def postprocess_mask(mask, min_area, keep_largest):
    mask = fill_holes(mask)
    return filter_components(mask, min_area=min_area, keep_largest=keep_largest)


def load_model(checkpoint_path, size):
    model = segmentation_model.FCBFormer(size=size)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    print(f"Loaded segmentation checkpoint: {checkpoint_path}")
    return model


def build_transform(size):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((size, size), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def iter_images(folder):
    for image_path in sorted(folder.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def save_outputs(prob, image_path, output_dataset, label, args):
    output_name = image_path.with_suffix(".png").name
    npy_name = image_path.with_suffix(".npy").name

    if args.save_probs:
        prob_dir = Path(args.prob_output_root) / output_dataset / label
        prob_dir.mkdir(parents=True, exist_ok=True)
        np.save(prob_dir / npy_name, prob.astype(np.float32))

        if not args.no_save_prob_png:
            prob_png_dir = Path(args.prob_png_output_root) / output_dataset / label
            prob_png = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
            imwrite_image(prob_png_dir / output_name, prob_png)

    binary = (prob >= args.threshold).astype(np.uint8)
    if args.postprocess:
        binary = postprocess_mask(binary, min_area=args.min_area, keep_largest=args.keep_largest)

    mask_dir = Path(args.mask_output_root) / output_dataset / label
    mask = (binary * 255).astype(np.uint8)
    imwrite_image(mask_dir / output_name, mask)


def main():
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    input_root = Path(args.input_root)
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    model = load_model(checkpoint_path, size=args.size)
    transform_input = build_transform(args.size)

    tasks = resolve_tasks(args)
    print(f"Device: {DEVICE}")
    print(f"Threshold: {args.threshold:.4f}")
    print(f"Postprocess: {args.postprocess}, keep_largest={args.keep_largest}, min_area={args.min_area}")

    total = 0
    with torch.no_grad():
        for input_folder, output_dataset in tasks:
            for label in args.classes:
                data_path = input_root / input_folder / label
                if not data_path.exists():
                    print(f"WARNING: skipping missing folder: {data_path}")
                    continue

                count = 0
                for image_path in iter_images(data_path):
                    image_bgr = imread_color(image_path)
                    if image_bgr is None:
                        print(f"WARNING: skipping unreadable image: {image_path}")
                        continue

                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    input_tensor = transform_input(image_rgb).unsqueeze(0).to(DEVICE)
                    logits = model(input_tensor)
                    prob_tensor = torch.sigmoid(logits)
                    prob_tensor = F.interpolate(
                        prob_tensor,
                        size=image_bgr.shape[:2],
                        mode="bilinear",
                        align_corners=False,
                    )
                    prob = prob_tensor.squeeze().detach().cpu().numpy()
                    prob = np.clip(prob, 0.0, 1.0)
                    save_outputs(prob, image_path, output_dataset, label, args)
                    count += 1

                total += count
                print(
                    f"Generated {count} masks for {data_path} -> "
                    f"{Path(args.mask_output_root) / output_dataset / label}"
                )

    print(f"Done. Generated {total} masks.")


if __name__ == "__main__":
    main()
