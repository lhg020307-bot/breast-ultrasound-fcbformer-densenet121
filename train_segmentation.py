import argparse
import json
import random
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.segmentation import segmentation_model

MIN_PRETRAINED_CHECKPOINT_BYTES = 100 * 1024 * 1024


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune FCBFormer on BUSBRA masks.")
    parser.add_argument("--image-root", default="images/full_image")
    parser.add_argument("--mask-root", default="images/gt_masks")
    parser.add_argument("--folders", nargs="+", default=["BUS"])
    parser.add_argument("--classes", nargs="+", default=["benign", "malignant"])
    parser.add_argument("--pretrained-checkpoint", default="checkpoints/pretrained/FCBFormer_checkpoint.pt")
    parser.add_argument("--output-dir", default="outputs/models/segmentation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=352)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_root=None,
        mask_root=None,
        folders=None,
        classes=None,
        size=352,
        augment=False,
        samples=None,
    ):
        self.image_root = Path(image_root) if image_root is not None else None
        self.mask_root = Path(mask_root) if mask_root is not None else None
        self.size = size
        self.augment = augment
        self.samples = list(samples) if samples is not None else []

        if samples is None:
            for folder in folders:
                for label in classes:
                    image_dir = self.image_root / folder / label
                    mask_dir = self.mask_root / folder / label
                    if not image_dir.exists() or not mask_dir.exists():
                        print(f"Skipping missing image/mask folder: {image_dir} | {mask_dir}")
                        continue
                    for image_path in sorted(image_dir.iterdir()):
                        if not image_path.is_file():
                            continue
                        mask_path = mask_dir / image_path.name
                        if mask_path.exists():
                            self.samples.append((image_path, mask_path))
                        else:
                            print(f"Skipping image without GT mask: {image_path}")

        if not self.samples:
            raise RuntimeError("No segmentation samples found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = imread_unicode(image_path, cv2.IMREAD_COLOR)
        mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Failed to read image or mask: {image_path} | {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                image = np.ascontiguousarray(np.flip(image, axis=1))
                mask = np.ascontiguousarray(np.flip(mask, axis=1))
            if random.random() < 0.5:
                image = np.ascontiguousarray(np.flip(image, axis=0))
                mask = np.ascontiguousarray(np.flip(mask, axis=0))

        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return image, mask


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def load_pretrained(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if not checkpoint_path.exists():
        print(f"Pretrained checkpoint not found, training from random init: {checkpoint_path}")
        return

    checkpoint_size = checkpoint_path.stat().st_size
    if checkpoint_size < MIN_PRETRAINED_CHECKPOINT_BYTES:
        raise RuntimeError(
            "Pretrained checkpoint is too small and is probably incomplete.\n"
            f"Path: {checkpoint_path}\n"
            f"Size: {checkpoint_size} bytes\n"
            "Please re-upload checkpoints/pretrained/FCBFormer_checkpoint.pt from the local project."
        )
    if not zipfile.is_zipfile(checkpoint_path):
        first_bytes = checkpoint_path.read_bytes()[:32]
        raise RuntimeError(
            "Pretrained checkpoint is not a valid PyTorch zip checkpoint.\n"
            f"Path: {checkpoint_path}\n"
            f"Size: {checkpoint_size} bytes\n"
            f"First bytes: {first_bytes!r}\n"
            "Please re-upload checkpoints/pretrained/FCBFormer_checkpoint.pt from the local project."
        )

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except RuntimeError as error:
        raise RuntimeError(
            "Failed to load pretrained checkpoint. The file is likely corrupted or only "
            "partially uploaded.\n"
            f"Path: {checkpoint_path}\n"
            f"Size: {checkpoint_size} bytes\n"
            "Expected local size is about 637,021,647 bytes for this project.\n"
            "Please delete the server copy and upload checkpoints/pretrained/FCBFormer_checkpoint.pt again."
        ) from error

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded pretrained segmentation checkpoint: {checkpoint_path} ({checkpoint_size} bytes)")


def dice_loss(logits, target, smooth=1.0):
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    target = target.flatten(1)
    intersection = (probs * target).sum(dim=1)
    denominator = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + smooth) / (denominator + smooth)
    return 1 - dice.mean()


def segmentation_metrics(logits, target):
    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    pred_sum = pred.sum(dim=(1, 2, 3))
    target_sum = target.sum(dim=(1, 2, 3))
    union = pred_sum + target_sum - intersection
    dice = ((2 * intersection + 1.0) / (pred_sum + target_sum + 1.0)).mean()
    iou = ((intersection + 1.0) / (union + 1.0)).mean()
    return float(dice.item()), float(iou.item())


def run_epoch(model, loader, optimizer, scaler, device, amp, train):
    model.train(train)
    losses = []
    dices = []
    ious = []

    for image, mask in loader:
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(image)
                bce = F.binary_cross_entropy_with_logits(logits, mask)
                loss = bce + dice_loss(logits, mask)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        dice, iou = segmentation_metrics(logits.detach(), mask)
        losses.append(float(loss.item()))
        dices.append(dice)
        ious.append(iou)

    return {
        "loss": float(np.mean(losses)),
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_root = PROJECT_ROOT / args.image_root
    mask_root = PROJECT_ROOT / args.mask_root
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    full_dataset = SegmentationDataset(
        image_root=image_root,
        mask_root=mask_root,
        folders=args.folders,
        classes=args.classes,
        size=args.size,
        augment=True,
    )
    indices = list(range(len(full_dataset)))
    random.Random(args.seed).shuffle(indices)
    val_size = max(1, int(len(full_dataset) * args.val_fraction))
    val_indices = set(indices[:val_size])
    train_samples = [
        sample for index, sample in enumerate(full_dataset.samples) if index not in val_indices
    ]
    val_samples = [
        sample for index, sample in enumerate(full_dataset.samples) if index in val_indices
    ]
    train_dataset = SegmentationDataset(size=args.size, augment=True, samples=train_samples)
    val_dataset = SegmentationDataset(size=args.size, augment=False, samples=val_samples)

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(args.seed + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=val_generator,
    )

    model = segmentation_model.FCBFormer(size=args.size).to(device)
    load_pretrained(model, args.pretrained_checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_dice = -1.0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, args.amp, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, scaler, device, args.amp, train=False)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(checkpoint, output_dir / "best.pt")
            print(f"Saved best checkpoint: {output_dir / 'best.pt'}")

    with (output_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    print(f"Best validation Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
