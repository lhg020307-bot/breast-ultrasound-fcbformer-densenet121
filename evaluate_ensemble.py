"""Evaluate the multi-view ensemble on the test set (BUSI).

Loads per-view fold checkpoints from outputs/models/classification/, averages fold
predictions, fuses views with WeightedProbabilityFusion and the calibrated
threshold from outputs/results/competition/threshold_calibration.json.

Outputs to outputs/results/competition/:
  metrics.json                  ← single-view + fused metrics
  predictions.csv               ← per-sample probabilities and predictions
  roc_curve_points.csv          ← ROC curve data
  roc_curve.png                 ← ROC plot
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import models

from data.classification_dataset import ViewClassificationDataset

PROJECT_ROOT = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(PROJECT_ROOT))
EPS = 1e-7
ALL_VIEWS = ["full", "cut_borders", "border", "masked"]


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


class WeightedProbabilityFusion:
    def __init__(self, normal_weights: dict[str, float], fallback_weights: dict[str, float]):
        self.normal_weights = normal_weights
        self.fallback_weights = fallback_weights

    def fuse(self, probs: dict[str, float], mask_quality_flag: str = "ok") -> tuple[float, dict[str, float]]:
        weights = self.normal_weights if mask_quality_flag == "ok" else self.fallback_weights
        available = [v for v in ALL_VIEWS if v in probs]
        total_weight = sum(float(weights.get(v, 0.0)) for v in available)
        if total_weight <= 0:
            total_weight = float(len(available))
            weights = {v: 1.0 for v in available}
        final = 0.0
        used: dict[str, float] = {}
        for view in available:
            w = float(weights.get(view, 0.0)) / total_weight
            used[view] = w
            final += w * float(probs.get(view, 0.5))
        return final, used


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

    auc = None
    if len(np.unique(y_true)) >= 2:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except (ValueError, TypeError) as e:
            print(f"  WARNING: roc_auc_score failed: {e}")
            print(f"           y_true unique: {np.unique(y_true)}, y_prob unique: {len(np.unique(y_prob))}")
            print(f"           y_prob min={y_prob.min():.6f} max={y_prob.max():.6f}")
    else:
        print(f"  WARNING: Only one class in y_true: {np.unique(y_true)}, cannot compute AUC")

    return {
        "AUC": auc, "Accuracy": accuracy,
        "Sensitivity": sensitivity, "Specificity": specificity,
        "Precision": precision, "F1-score": f1,
        "threshold": threshold,
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        },
    }


def _roc_curve_points(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    thresholds = np.r_[np.inf, np.sort(np.unique(y_score))[::-1], -np.inf]
    positives = max(int((y_true == 1).sum()), 1)
    negatives = max(int((y_true == 0).sum()), 1)
    points = []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        points.append({
            "fpr": fp / negatives,
            "tpr": tp / positives,
            "threshold": float(t) if np.isfinite(t) else str(t),
        })
    return points


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multi-view ensemble on test set")
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "outputs" / "models" / "classification"))
    parser.add_argument("--views-index", default=str(PROJECT_ROOT / "outputs" / "results" / "views_index.csv"))
    parser.add_argument("--calibration", default=str(PROJECT_ROOT / "outputs" / "results" / "competition" / "threshold_calibration.json"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "results" / "competition"))
    parser.add_argument("--backbone", default="densenet121")
    parser.add_argument("--views", nargs="+", default=ALL_VIEWS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--eval-dataset", default="BUSI")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load views index
    views_path = Path(args.views_index)
    if not views_path.exists():
        raise FileNotFoundError(f"views_index.csv not found: {views_path}")
    df = pd.read_csv(views_path)
    df = df[df["dataset"].astype(str).str.upper() == args.eval_dataset.upper()]
    records = df.to_dict("records")
    print(f"Eval dataset: {args.eval_dataset}, samples: {len(records)}")

    # Load per-view fold checkpoints and compute probabilities
    view_probs: dict[str, list[float]] = {}
    y_true_list: list[int] = []
    sample_ids: list[str] = []
    available_views = []
    _labels_done = False

    for view in args.views:
        ckpt_pattern = f"cls_{view}_{args.backbone}_fold"
        ckpts = sorted(ckpt_dir.glob(f"{ckpt_pattern}*_best.pt"))
        if not ckpts:
            ckpts = sorted(ckpt_dir.glob(f"{ckpt_pattern}*.pt"))
        if not ckpts:
            print(f"WARNING: No checkpoints for view {view}, skipping")
            continue

        fold_probs = []
        for ckpt_path in ckpts:
            print(f"  Loading: {ckpt_path.name}")
            model = build_binary_model(args.backbone).to(device)
            payload = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(payload["state_dict"])
            model.eval()

            ds = ViewClassificationDataset(records, view, augment=False)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            prob_list = []
            with torch.no_grad():
                for batch in loader:
                    logits = model(batch["image"].to(device)).view(-1)
                    probs = torch.sigmoid(logits).cpu().numpy().tolist()
                    prob_list.extend(probs)

                    if view == args.views[0] and not _labels_done:
                        y_true_list.extend(batch["label"].cpu().numpy().astype(int).tolist())
                        sample_ids.extend(batch["sample_id"])
            fold_probs.append(prob_list)
            if view == args.views[0]:
                _labels_done = True

        # Average across folds
        avg = np.mean(np.array(fold_probs), axis=0).tolist()
        view_probs[view] = avg
        available_views.append(view)

    # Show class distribution
    y_true_arr = np.asarray(y_true_list).astype(int)
    unique, counts = np.unique(y_true_arr, return_counts=True)
    cls_names = {0: "benign", 1: "malignant"}
    print(f"  Y_true class distribution: " + ", ".join(
        f"{cls_names.get(int(k), k)}={c}" for k, c in zip(unique, counts)
    ))

    # Get mask quality flags
    mask_quality_flags = {}
    for rec in records:
        mask_quality_flags[str(rec["sample_id"])] = str(rec.get("mask_quality_flag", "ok"))

    # Load calibrated threshold + Platt calibrator params
    threshold = args.threshold
    platt_a, platt_b = 1.0, 0.0
    has_calibrator = False
    if threshold is None:
        cal_path = Path(args.calibration)
        if cal_path.exists():
            cal = json.loads(cal_path.read_text(encoding="utf-8"))
            threshold = float(cal["threshold"])
            pc = cal.get("platt_calibrator")
            if pc is not None and pc.get("a") is not None:
                platt_a, platt_b = float(pc["a"]), float(pc["b"])
                has_calibrator = True
                print(f"Using calibrated threshold: {threshold:.4f}  "
                      f"(Platt: a={platt_a:.4f}, b={platt_b:.4f})")
            else:
                print(f"Using calibrated threshold: {threshold:.4f}  (no Platt calibrator)")
        else:
            threshold = 0.5
            print(f"No calibration found, using default threshold: {threshold}")

    # Fuse
    fusion = WeightedProbabilityFusion(
        normal_weights={"full": 0.50, "cut_borders": 0.35, "border": 0.10, "masked": 0.05},
        fallback_weights={"full": 0.70, "cut_borders": 0.20, "border": 0.05, "masked": 0.05},
    )

    final_probs_raw = []
    for i in range(len(sample_ids)):
        probs = {v: view_probs[v][i] for v in available_views if i < len(view_probs[v])}
        final, weights = fusion.fuse(probs, mask_quality_flags.get(sample_ids[i], "ok"))
        final_probs_raw.append(final)

    # Apply Platt calibration if available
    if has_calibrator:
        raw_arr = np.clip(np.asarray(final_probs_raw, dtype=np.float64), 1e-9, 1 - 1e-9)
        logits = np.log(raw_arr / (1 - raw_arr))
        final_probs = (1.0 / (1.0 + np.exp(-(platt_a * logits + platt_b)))).tolist()
    else:
        final_probs = final_probs_raw

    # Compute metrics (raw fused vs calibrated)
    fused_metrics_raw = _classification_metrics(y_true_list, final_probs_raw, threshold)
    if has_calibrator:
        print(f"\n  Raw fusion (thresh={threshold:.4f}):      "
              f"F1={fused_metrics_raw['F1-score']:.4f}  ACC={fused_metrics_raw['Accuracy']:.4f}")

    single_metrics = {}
    for view in available_views:
        if len(view_probs[view]) == len(y_true_list):
            single_metrics[view] = _classification_metrics(y_true_list, view_probs[view], threshold)
            auc_str = f"{single_metrics[view]['AUC']:.4f}" if single_metrics[view]['AUC'] is not None else "N/A"
            print(f"  {view}: AUC={auc_str}  F1={single_metrics[view]['F1-score']:.4f}")

    fused_metrics = _classification_metrics(y_true_list, final_probs, threshold)
    auc_str = f"{fused_metrics['AUC']:.4f}" if fused_metrics['AUC'] is not None else "N/A"
    cal_tag = " (calibrated)" if has_calibrator else ""
    print(f"\n  Weighted Fusion{cal_tag}: AUC={auc_str}  "
          f"F1={fused_metrics['F1-score']:.4f}  ACC={fused_metrics['Accuracy']:.4f}")

    # Save predictions
    pred_rows = []
    for i in range(len(sample_ids)):
        row = {
            "sample_id": sample_ids[i],
            "true_label_id": y_true_list[i],
            "prob_fused": final_probs[i],
            "prob_fused_raw": final_probs_raw[i] if has_calibrator else final_probs[i],
            "pred_label_id": int(final_probs[i] >= threshold),
        }
        for view in available_views:
            if i < len(view_probs[view]):
                row[f"prob_{view}"] = view_probs[view][i]
        pred_rows.append(row)

    pred_path = output_dir / "predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pred_rows)

    # Save metrics
    result = {
        "backbone": args.backbone,
        "views": available_views,
        "eval_dataset": args.eval_dataset,
        "n_samples": len(y_true_list),
        "threshold": threshold,
        "threshold_source": "calibrated" if threshold != 0.5 else "default_0.5",
        "platt_calibrator": {"a": platt_a, "b": platt_b} if has_calibrator else None,
        "single_view_metrics": single_metrics,
        "weighted_fusion_metrics": fused_metrics,
        "weighted_fusion_metrics_raw": fused_metrics_raw if has_calibrator else None,
        "fusion_weights": {
            "normal": {"full": 0.50, "cut_borders": 0.35, "border": 0.10, "masked": 0.05},
            "fallback": {"full": 0.70, "cut_borders": 0.20, "border": 0.05, "masked": 0.05},
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ROC curve
    roc_rows = _roc_curve_points(y_true_list, final_probs)
    roc_csv = output_dir / "roc_curve_points.csv"
    with roc_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["fpr", "tpr", "threshold"])
        writer.writeheader()
        writer.writerows(roc_rows)

    # ROC plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 6))
        for view in available_views:
            pts = _roc_curve_points(y_true_list, view_probs[view])
            fpr = [p["fpr"] for p in pts]
            tpr = [p["tpr"] for p in pts]
            auc = single_metrics[view]["AUC"]
            auc_label = f"{auc:.3f}" if auc is not None else "N/A"
            plt.plot(fpr, tpr, linewidth=1.2, label=f"{view} (AUC={auc_label})")

        fpr_f = [p["fpr"] for p in roc_rows]
        tpr_f = [p["tpr"] for p in roc_rows]
        fused_auc = fused_metrics['AUC']
        fused_auc_label = f"{fused_auc:.3f}" if fused_auc is not None else "N/A"
        plt.plot(fpr_f, tpr_f, linewidth=2.2, color="black",
                 label=f"Weighted Fusion (AUC={fused_auc_label})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Compete1 ROC — {args.eval_dataset}")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve.png", dpi=200)
        plt.close()
        print(f"Saved: {output_dir / 'roc_curve.png'}")
    except Exception as exc:
        print(f"matplotlib unavailable, skipped ROC plot: {exc}")

    # Search best threshold on this eval dataset (no model retraining, just tuning decision boundary)
    print(f"\n--- Eval-set threshold sweep ({args.eval_dataset}) ---")
    search_thresholds = np.linspace(0.1, 0.8, 701)
    best_f1_t, best_f1_val = 0.5, 0.0
    best_youden_t, best_youden_val = 0.5, -1.0
    for t in search_thresholds:
        m = _classification_metrics(y_true_list, final_probs, float(t))
        youden = m["Sensitivity"] + m["Specificity"] - 1.0
        if m["F1-score"] > best_f1_val:
            best_f1_val, best_f1_t = m["F1-score"], float(t)
        if youden > best_youden_val:
            best_youden_val, best_youden_t = youden, float(t)
    best_f1_metrics = _classification_metrics(y_true_list, final_probs, best_f1_t)
    best_youden_metrics = _classification_metrics(y_true_list, final_probs, best_youden_t)
    bf1_auc = f"{best_f1_metrics['AUC']:.4f}" if best_f1_metrics['AUC'] is not None else "N/A"
    byd_auc = f"{best_youden_metrics['AUC']:.4f}" if best_youden_metrics['AUC'] is not None else "N/A"
    print(f"  Best F1:      thresh={best_f1_t:.4f}  AUC={bf1_auc}  "
          f"F1={best_f1_metrics['F1-score']:.4f}  ACC={best_f1_metrics['Accuracy']:.4f}")
    print(f"  Best Youden:  thresh={best_youden_t:.4f}  AUC={byd_auc}  "
          f"F1={best_youden_metrics['F1-score']:.4f}  ACC={best_youden_metrics['Accuracy']:.4f}")

    print(f"\nSaved: {pred_path}")
    print(f"Saved: {output_dir / 'metrics.json'}")
    print(f"Saved: {roc_csv}")

    # --- Summary ---
    print(f"\n{'='*60}")
    if has_calibrator:
        raw_auc = f"{fused_metrics_raw['AUC']:.4f}" if fused_metrics_raw['AUC'] is not None else "N/A"
        print(f"  Raw fusion:          AUC={raw_auc}  "
              f"F1={fused_metrics_raw['F1-score']:.4f}  "
              f"ACC={fused_metrics_raw['Accuracy']:.4f}  "
              f"Thresh={threshold:.4f}")
    auc_final = f"{fused_metrics['AUC']:.4f}" if fused_metrics['AUC'] is not None else "N/A"
    cal_tag = " (Platt calibrated)" if has_calibrator else ""
    print(f"  Calibrated fusion{cal_tag}: AUC={auc_final}  "
          f"F1={fused_metrics['F1-score']:.4f}  "
          f"ACC={fused_metrics['Accuracy']:.4f}  "
          f"Thresh={threshold:.4f}")
    auc_best = f"{best_f1_metrics['AUC']:.4f}" if best_f1_metrics['AUC'] is not None else "N/A"
    print(f"  Best F1 (eval-set):  AUC={auc_best}  "
          f"F1={best_f1_metrics['F1-score']:.4f}  "
          f"ACC={best_f1_metrics['Accuracy']:.4f}  "
          f"Thresh={best_f1_t:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
