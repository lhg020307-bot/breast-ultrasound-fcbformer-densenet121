"""5-fold cross-validation classification training for Compete1.

Pure PyTorch training loop — no Lightning dependency.
Uses torchvision DenseNet121 with verified weight loading.

Output structure (Compete-style):
  outputs/checkpoints/  ← fold .pt files
  outputs/metrics/      ← OOF CSV + OOF metrics JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import models

from data.classification_dataset import ViewClassificationDataset
from data.augmentation import DEFAULT_AUG_PARAMS

PROJECT_ROOT = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(PROJECT_ROOT))
EPS = 1e-7


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
    parser = argparse.ArgumentParser(description="5-fold CV classification training")
    parser.add_argument("--view", required=True, choices=["full", "cut_borders", "border", "masked"])
    parser.add_argument("--backbone", default="densenet121")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--train-dataset", default="BUS")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--views-index", default=str(PROJECT_ROOT / "outputs" / "results" / "views_index.csv"))
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "outputs" / "models" / "classification"))
    parser.add_argument("--metrics-dir", default=str(PROJECT_ROOT / "outputs" / "results" / "oof"))
    return parser.parse_args()


def build_binary_model(backbone: str, num_classes: int = 1) -> nn.Module:
    if backbone == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if backbone == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model
    if backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported backbone: {backbone}")


def _classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp + EPS))
    sensitivity = float(tp / (tp + fn + EPS))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) >= 2 else None

    return {
        "auc": auc, "accuracy": accuracy,
        "sensitivity": sensitivity, "specificity": specificity,
        "precision": precision, "f1": f1,
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _score(metrics):
    auc = metrics.get("auc") or 0.0
    f1 = float(metrics.get("f1", 0.0))
    sensitivity = float(metrics.get("sensitivity", 0.0))
    specificity = float(metrics.get("specificity", 0.0))
    balanced = 0.5 * (sensitivity + specificity)
    return 0.70 * auc + 0.20 * f1 + 0.10 * balanced


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_dir = Path(args.checkpoint_dir)
    metrics_dir = Path(args.metrics_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    views_path = Path(args.views_index)
    if not views_path.exists():
        raise FileNotFoundError(f"views_index.csv not found: {views_path}")
    df = pd.read_csv(views_path)
    df = df[df["dataset"].astype(str).str.upper() == args.train_dataset.upper()]
    if args.max_samples:
        df = df.head(args.max_samples)

    records = df.to_dict("records")
    labels = np.asarray([int(r["label"]) for r in records])
    print(f"View: {args.view} | Backbone: {args.backbone} | Samples: {len(records)} | Device: {device}")

    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) >= 2 and counts.min() >= args.n_splits:
        splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(np.zeros(len(labels)), labels))
    else:
        splitter = KFold(n_splits=min(args.n_splits, len(records)), shuffle=True, random_state=args.seed)
        splits = list(splitter.split(np.zeros(len(labels))))
        print(f"Warning: Using KFold (not stratified). n_splits={len(splits)}")

    oof_rows = []
    fold_metrics_list = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_recs = [records[i] for i in train_idx]
        val_recs = [records[i] for i in val_idx]

        train_ds = ViewClassificationDataset(train_recs, args.view, augment=True, aug_params=DEFAULT_AUG_PARAMS)
        val_ds = ViewClassificationDataset(val_recs, args.view, augment=False)

        train_generator = torch.Generator()
        train_generator.manual_seed(args.seed + fold_idx)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=train_generator,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
        )

        model = build_binary_model(args.backbone).to(device)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)

        pos_count = sum(int(r["label"]) for r in train_recs)
        neg_count = len(train_recs) - pos_count
        pos_weight = torch.tensor(neg_count / max(pos_count, 1), dtype=torch.float32, device=device) if pos_count > 0 else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_score = float("-inf")
        best_epoch = 0
        no_improve = 0
        best_state = None
        best_val_metrics = None

        for epoch in range(args.epochs):
            model.train()
            losses = []
            for batch in train_loader:
                images = batch["image"].to(device)
                labs = batch["label"].to(device, dtype=torch.float32)
                logits = model(images).view(-1)
                loss = criterion(logits, labs)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                losses.append(float(loss.item()))

            model.eval()
            y_true, y_prob = [], []
            with torch.no_grad():
                for batch in val_loader:
                    logits = model(batch["image"].to(device)).view(-1)
                    probs = torch.sigmoid(logits).cpu().numpy().tolist()
                    y_prob.extend(probs)
                    y_true.extend(batch["label"].cpu().numpy().astype(int).tolist())

            metrics = _classification_metrics(y_true, y_prob)
            current_score = _score(metrics)

            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch + 1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_val_metrics = metrics
                no_improve = 0
            else:
                no_improve += 1

            print(f"  fold={fold_idx} epoch={epoch+1:3d} loss={np.mean(losses):.4f} "
                  f"auc={metrics.get('auc')} f1={metrics['f1']:.4f} "
                  f"best={best_score:.4f}@{best_epoch} no_imp={no_improve}")

            if no_improve >= args.patience:
                print(f"  Early stopping fold {fold_idx} at epoch {epoch+1}")
                break

        # Save best fold checkpoint
        ckpt_name = f"cls_{args.view}_{args.backbone}_fold{fold_idx}_best.pt"
        ckpt_path = ckpt_dir / ckpt_name
        torch.save({
            "state_dict": best_state,
            "view": args.view, "fold": fold_idx, "backbone": args.backbone,
            "val_metrics": best_val_metrics, "best_epoch": best_epoch,
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}")

        # Also save last checkpoint
        last_ckpt = ckpt_dir / f"cls_{args.view}_{args.backbone}_fold{fold_idx}.pt"
        torch.save({
            "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
            "view": args.view, "fold": fold_idx, "backbone": args.backbone,
            "best_epoch": best_epoch,
        }, last_ckpt)

        # Collect OOF probabilities from best model
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["image"].to(device)).view(-1)
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                for i in range(len(probs)):
                    oof_rows.append({
                        "sample_id": batch["sample_id"][i],
                        "dataset": batch["dataset"][i],
                        "y_true": int(batch["label"][i]),
                        f"prob_{args.view}": float(probs[i]),
                        "fold": fold_idx,
                        "view": args.view,
                        "model": args.backbone,
                    })

        best_val_metrics["fold"] = fold_idx
        best_val_metrics["view"] = args.view
        best_val_metrics["train_loss"] = float(np.mean(losses))
        fold_metrics_list.append(best_val_metrics)
        print(f"  fold={fold_idx} best: auc={best_val_metrics.get('auc')} f1={best_val_metrics['f1']:.4f} @ epoch={best_epoch}")

    # Save OOF CSV
    oof_path = metrics_dir / f"oof_{args.view}_{args.backbone}.csv"
    with oof_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "dataset", "y_true", f"prob_{args.view}", "fold", "view", "model"])
        writer.writeheader()
        writer.writerows(oof_rows)

    # Aggregate metrics
    oof_df = pd.DataFrame(oof_rows)
    agg = _classification_metrics(
        oof_df["y_true"].astype(int).tolist(),
        oof_df[f"prob_{args.view}"].astype(float).tolist(),
    )
    agg["view"] = args.view
    agg["backbone"] = args.backbone
    agg["n_splits"] = len(splits)
    agg["fold_metrics"] = fold_metrics_list

    json_path = metrics_dir / f"classification_oof_metrics_{args.view}_{args.backbone}.json"
    json_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nOOF Aggregate: auc={agg.get('auc')} f1={agg['f1']:.4f} acc={agg['accuracy']:.4f}")
    print(f"Saved: {oof_path}")
    print(f"Saved: {json_path}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
