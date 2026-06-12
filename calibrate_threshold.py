"""Fuse multi-view OOF probabilities and calibrate the decision threshold.

Reads per-view OOF CSVs from outputs/results/oof/, applies WeightedProbabilityFusion,
and searches for the best threshold on BUSBRA OOF data (no test-set leakage).

Outputs to outputs/results/competition/:
  oof_fusion_predictions.csv    ← fused OOF probabilities
  threshold_search.csv          ← full threshold sweep table
  threshold_calibration.json    ← best threshold + metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(PROJECT_ROOT))
EPS = 1e-7
ALL_VIEWS = ["full", "cut_borders", "border", "masked"]


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
    else:
        print(f"  WARNING: Only one class in y_true: {np.unique(y_true)}, cannot compute AUC")

    return {
        "AUC": auc, "Accuracy": accuracy,
        "Sensitivity": sensitivity, "Specificity": specificity,
        "Precision": precision, "F1-score": f1,
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate decision threshold on OOF predictions")
    parser.add_argument("--metrics-dir", default=str(PROJECT_ROOT / "outputs" / "results" / "oof"))
    parser.add_argument("--views-index", default=str(PROJECT_ROOT / "outputs" / "results" / "views_index.csv"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "results" / "competition"))
    parser.add_argument("--backbone", default="densenet121")
    parser.add_argument("--views", nargs="+", default=ALL_VIEWS)
    parser.add_argument("--objective", default="youden",
                        choices=["youden", "f1", "sensitivity", "specificity", "balanced"])
    parser.add_argument("--min-sensitivity", type=float, default=0.85)
    parser.add_argument("--min-specificity", type=float, default=0.80)
    parser.add_argument("--steps", type=int, default=1001)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load per-view OOF CSVs
    merged = None
    available_views = []
    for view in args.views:
        oof_path = metrics_dir / f"oof_{view}_{args.backbone}.csv"
        if not oof_path.exists():
            print(f"WARNING: OOF file not found: {oof_path}, skipping view {view}")
            continue
        df = pd.read_csv(oof_path)
        cols = ["sample_id", "dataset", "y_true", f"prob_{view}"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"WARNING: {oof_path} missing columns {missing}, skipping")
            continue
        df = df[cols].copy()
        df["sample_id"] = df["sample_id"].astype(str)
        merged = df if merged is None else merged.merge(df, on=["sample_id", "dataset", "y_true"], how="inner")
        available_views.append(view)

    if merged is None or merged.empty:
        raise RuntimeError("No OOF data to fuse.")

    # Merge mask quality flags
    views_idx_path = Path(args.views_index)
    if views_idx_path.exists():
        vi = pd.read_csv(views_idx_path)
        vi = vi[["sample_id", "mask_quality_flag"]].copy()
        vi["sample_id"] = vi["sample_id"].astype(str)
        merged = merged.merge(vi, on="sample_id", how="left")
    if "mask_quality_flag" not in merged.columns:
        merged["mask_quality_flag"] = "ok"
    merged["mask_quality_flag"] = merged["mask_quality_flag"].fillna("fallback")

    # Fuse
    fusion = WeightedProbabilityFusion(
        normal_weights={"full": 0.50, "cut_borders": 0.35, "border": 0.10, "masked": 0.05},
        fallback_weights={"full": 0.70, "cut_borders": 0.20, "border": 0.05, "masked": 0.05},
    )
    final_probs = []
    for record in merged.to_dict("records"):
        probs = {v: float(record[f"prob_{v}"]) for v in available_views}
        final, _ = fusion.fuse(probs, str(record.get("mask_quality_flag", "fallback")))
        final_probs.append(final)

    merged["prob_final_raw"] = final_probs
    y_true = merged["y_true"].astype(int).tolist()
    y_prob_raw = merged["prob_final_raw"].astype(float).tolist()

    # ---- Platt scaling: calibrate fused probabilities on BUSBRA OOF ----
    # Fit logistic regression: logit = log(p/(1-p)); calib = sigmoid(a*logit + b)
    raw = np.asarray(y_prob_raw, dtype=np.float64)
    raw_clipped = np.clip(raw, 1e-9, 1 - 1e-9)
    logits = np.log(raw_clipped / (1 - raw_clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
    calibrator.fit(logits, y_true)
    a = float(calibrator.coef_[0, 0])
    b = float(calibrator.intercept_[0])

    calibrated_logits = a * logits.ravel() + b
    y_prob_cal = 1.0 / (1.0 + np.exp(-calibrated_logits))

    # Evaluate calibration effect on OOF
    raw_metrics = _classification_metrics(y_true, y_prob_raw, 0.5)
    cal_metrics = _classification_metrics(y_true, y_prob_cal, 0.5)
    raw_auc = f"{raw_metrics['AUC']:.4f}" if raw_metrics['AUC'] is not None else "N/A"
    cal_auc = f"{cal_metrics['AUC']:.4f}" if cal_metrics['AUC'] is not None else "N/A"
    print(f"\n  Platt calibration fit: a={a:.4f}, b={b:.4f}")
    print(f"  Raw (thresh=0.5):     AUC={raw_auc}  F1={raw_metrics['F1-score']:.4f}  "
          f"Se={raw_metrics['Sensitivity']:.4f}  Sp={raw_metrics['Specificity']:.4f}")
    print(f"  Calibrated (thresh=0.5): AUC={cal_auc}  F1={cal_metrics['F1-score']:.4f}  "
          f"Se={cal_metrics['Sensitivity']:.4f}  Sp={cal_metrics['Specificity']:.4f}")

    merged["prob_final"] = y_prob_cal.tolist()
    y_prob = y_prob_cal.tolist()

    # ---- Threshold search on RAW probabilities (reference only) ----
    thresholds = np.linspace(0.0, 1.0, int(args.steps))
    raw_rows = []
    for thresh in thresholds:
        m = _classification_metrics(y_true, y_prob_raw, float(thresh))
        raw_rows.append({
            "threshold": float(thresh),
            **{k: v for k, v in m.items() if k not in ("confusion_matrix",)},
            "youden": m["Sensitivity"] + m["Specificity"] - 1.0,
        })
    raw_table = pd.DataFrame(raw_rows)

    objective = args.objective.lower()
    if objective == "f1":
        raw_chosen = raw_table.sort_values(["F1-score", "youden"], ascending=False).iloc[0]
    elif objective == "sensitivity":
        cand = raw_table[raw_table["Sensitivity"] >= args.min_sensitivity]
        raw_chosen = (cand if not cand.empty else raw_table).sort_values(["Specificity"], ascending=False).iloc[0]
    elif objective == "specificity":
        cand = raw_table[raw_table["Specificity"] >= args.min_specificity]
        raw_chosen = (cand if not cand.empty else raw_table).sort_values(["Sensitivity"], ascending=False).iloc[0]
    elif objective == "balanced":
        raw_table["gap"] = (raw_table["Sensitivity"] - raw_table["Specificity"]).abs()
        raw_chosen = raw_table.sort_values(["youden", "gap"], ascending=[False, True]).iloc[0]
    else:
        raw_chosen = raw_table.sort_values(["youden", "F1-score"], ascending=False).iloc[0]
    raw_best_threshold = float(raw_chosen["threshold"])
    raw_best_metrics = _classification_metrics(y_true, y_prob_raw, raw_best_threshold)

    # ---- Final threshold: use 0.5 on calibrated probabilities ----
    # Platt scaling maps scores to well-calibrated probabilities;
    # by definition 0.5 should be the optimal decision boundary.
    calibrated_threshold = 0.5
    final_metrics = cal_metrics  # metrics at threshold=0.5 on calibrated probs
    final_metrics["threshold"] = calibrated_threshold
    final_metrics["objective"] = objective
    final_metrics["threshold_source"] = "BUSBRA_5fold_OOF_platt_0.5"
    final_metrics["min_sensitivity_constraint"] = args.min_sensitivity
    final_metrics["min_specificity_constraint"] = args.min_specificity
    final_metrics["n_samples"] = len(y_true)
    final_metrics["platt_calibrator"] = {"a": a, "b": b}
    final_metrics["raw_vs_calibrated_at_0.5"] = {
        "raw": {k: v for k, v in raw_metrics.items() if k != "confusion_matrix"},
        "calibrated": {k: v for k, v in cal_metrics.items() if k != "confusion_matrix"},
    }
    final_metrics["raw_best"] = {
        "threshold": raw_best_threshold,
        "F1-score": raw_best_metrics["F1-score"],
        "Accuracy": raw_best_metrics["Accuracy"],
    }

    # Save fused OOF CSV
    fused_path = output_dir / "oof_fusion_predictions.csv"
    merged.to_csv(fused_path, index=False, encoding="utf-8-sig")

    # Save threshold sweep table (on raw probs, for reference)
    raw_table.to_csv(output_dir / "threshold_search.csv", index=False, encoding="utf-8-sig")
    (output_dir / "threshold_calibration.json").write_text(
        json.dumps(final_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nAvailable views: {available_views}")
    print(f"Calibrator: a={a:.4f} b={b:.4f}")
    raw_auc_str = f"{raw_best_metrics['AUC']:.4f}" if raw_best_metrics['AUC'] is not None else "N/A"
    print(f"  Raw best (objective={objective}):     thresh={raw_best_threshold:.4f}  "
          f"AUC={raw_auc_str}  F1={raw_best_metrics['F1-score']:.4f}")
    auc_str = f"{final_metrics['AUC']:.4f}" if final_metrics['AUC'] is not None else "N/A"
    print(f"  Platt + 0.5 (no search needed):       thresh={calibrated_threshold:.4f}  "
          f"AUC={auc_str}  F1={final_metrics['F1-score']:.4f}")
    print(f"  -> Saved threshold: {calibrated_threshold:.4f} (Platt-calibrated, 0.5 fixed)")
    print(f"Saved: {output_dir / 'threshold_calibration.json'}")
    print(f"Saved: {output_dir / 'threshold_search.csv'}")


if __name__ == "__main__":
    main()
