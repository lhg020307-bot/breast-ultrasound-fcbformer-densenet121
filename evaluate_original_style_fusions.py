"""Evaluate fixed-weight Platt fusion for 2-view and 3-view.

Fixed-weight Platt fusion:
  1. Use the fixed normal/fallback weights from the 4-view baseline.
  2. Normalize weights over available views.
  3. Fit Platt calibration on OOF fused probabilities.
  4. Use calibrated threshold 0.5.
  5. Evaluate BUSI without any BUSI-driven tuning.

This script does not retrain models and does not overwrite previous outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results" / "competition"
OOF_DIR = PROJECT_ROOT / "outputs" / "results" / "oof"
VIEWS_INDEX = PROJECT_ROOT / "outputs" / "results" / "views_index.csv"
EPS = 1e-7

ALL_VIEWS = ["full", "cut_borders", "border", "masked"]
ORIGINAL_NORMAL_WEIGHTS = {
    "full": 0.50,
    "cut_borders": 0.35,
    "border": 0.10,
    "masked": 0.05,
}
ORIGINAL_FALLBACK_WEIGHTS = {
    "full": 0.70,
    "cut_borders": 0.20,
    "border": 0.05,
    "masked": 0.05,
}

FUSION_CONFIGS = {
    "2view_fixed_weight_platt_fusion": ["full", "cut_borders"],
    "3view_fixed_weight_platt_fusion": ["full", "cut_borders", "border"],
}


def classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp + EPS))
    sensitivity = float(tp / (tp + fn + EPS))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    auc = None
    if len(np.unique(y_true)) >= 2:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except (ValueError, TypeError):
            pass

    return {
        "AUC": auc,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1-score": f1,
        "threshold": float(threshold),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
    }


def roc_curve_points(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    thresholds = np.r_[np.inf, np.sort(np.unique(y_score))[::-1], -np.inf]
    pos = max(int((y_true == 1).sum()), 1)
    neg = max(int((y_true == 0).sum()), 1)
    points = []

    for threshold in thresholds:
        pred = y_score >= threshold
        tp = int((pred & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        points.append(
            {
                "fpr": fp / neg,
                "tpr": tp / pos,
                "threshold": float(threshold) if np.isfinite(threshold) else str(threshold),
            }
        )

    return points


def original_weighted_fusion(df, views):
    probs = np.zeros(len(df), dtype=float)
    used_weights = []

    flags = df["mask_quality_flag"].astype(str).fillna("fallback").values
    for idx, flag in enumerate(flags):
        base_weights = ORIGINAL_NORMAL_WEIGHTS if flag == "ok" else ORIGINAL_FALLBACK_WEIGHTS
        total = sum(float(base_weights.get(view, 0.0)) for view in views)
        if total <= 0:
            weights = {view: 1.0 / len(views) for view in views}
        else:
            weights = {view: float(base_weights.get(view, 0.0)) / total for view in views}
        used_weights.append(weights)
        probs[idx] = sum(weights[view] * float(df.iloc[idx][f"prob_{view}"]) for view in views)

    return probs, used_weights


def apply_platt(probs, a, b):
    clipped = np.clip(np.asarray(probs, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    logits = np.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + np.exp(-(float(a) * logits + float(b))))


def load_mask_flags():
    if not VIEWS_INDEX.exists():
        return pd.DataFrame(columns=["sample_id", "mask_quality_flag"])
    df = pd.read_csv(VIEWS_INDEX)
    if "mask_quality_flag" not in df.columns:
        return pd.DataFrame(columns=["sample_id", "mask_quality_flag"])
    df = df[["sample_id", "mask_quality_flag"]].copy()
    df["sample_id"] = df["sample_id"].astype(str)
    return df.drop_duplicates("sample_id")


def load_oof_data(views, mask_flags, backbone="densenet121"):
    merged = None
    for view in views:
        path = OOF_DIR / f"oof_{view}_{backbone}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df = df.groupby(["sample_id", "dataset"], as_index=False).agg(
            {f"prob_{view}": "mean", "y_true": "first"}
        )
        df = df[["sample_id", "dataset", "y_true", f"prob_{view}"]].copy()
        df["sample_id"] = df["sample_id"].astype(str)
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["sample_id", "dataset", "y_true"], how="inner")

    merged = merged.merge(mask_flags, on="sample_id", how="left")
    merged["mask_quality_flag"] = merged["mask_quality_flag"].fillna("fallback")
    return merged


def load_busi_predictions(views, mask_flags):
    df = pd.read_csv(OUTPUT_DIR / "predictions.csv")
    df["sample_id"] = df["sample_id"].astype(str)
    keep = ["sample_id", "true_label_id"] + [f"prob_{view}" for view in views]
    df = df[keep].copy()
    df = df.merge(mask_flags, on="sample_id", how="left")
    df["mask_quality_flag"] = df["mask_quality_flag"].fillna("fallback")
    return df


def evaluate_config(name, views, mask_flags):
    oof_df = load_oof_data(views, mask_flags)
    busi_df = load_busi_predictions(views, mask_flags)

    y_oof = oof_df["y_true"].astype(int).values
    y_busi = busi_df["true_label_id"].astype(int).values

    oof_raw, _ = original_weighted_fusion(oof_df, views)
    busi_raw, busi_weights = original_weighted_fusion(busi_df, views)

    oof_clipped = np.clip(oof_raw, 1e-9, 1.0 - 1e-9)
    oof_logits = np.log(oof_clipped / (1.0 - oof_clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
    calibrator.fit(oof_logits, y_oof)
    platt_a = float(calibrator.coef_[0, 0])
    platt_b = float(calibrator.intercept_[0])

    oof_cal = apply_platt(oof_raw, platt_a, platt_b)
    busi_cal = apply_platt(busi_raw, platt_a, platt_b)

    raw_oof_metrics = classification_metrics(y_oof, oof_raw, 0.5)
    cal_oof_metrics = classification_metrics(y_oof, oof_cal, 0.5)
    raw_busi_metrics = classification_metrics(y_busi, busi_raw, 0.5)
    cal_busi_metrics = classification_metrics(y_busi, busi_cal, 0.5)

    single_metrics = {}
    for view in views:
        single_metrics[view] = classification_metrics(y_busi, busi_df[f"prob_{view}"].values, 0.5)

    result = {
        "method": "fixed_weight_platt_fusion",
        "views_used": views,
        "eval_dataset": "BUSI",
        "calibration_dataset": "OOF (BUS+BUSBRA)",
        "n_oof": int(len(y_oof)),
        "n_busi": int(len(y_busi)),
        "original_normal_weights": ORIGINAL_NORMAL_WEIGHTS,
        "original_fallback_weights": ORIGINAL_FALLBACK_WEIGHTS,
        "normalized_weights_when_mask_ok": normalize_weights(ORIGINAL_NORMAL_WEIGHTS, views),
        "normalized_weights_when_fallback": normalize_weights(ORIGINAL_FALLBACK_WEIGHTS, views),
        "platt": {"a": platt_a, "b": platt_b},
        "threshold": 0.5,
        "single_view_metrics_busi": single_metrics,
        "raw": {
            "oof_metrics": raw_oof_metrics,
            "busi_metrics": raw_busi_metrics,
        },
        "platt_calibrated": {
            "oof_metrics": cal_oof_metrics,
            "busi_metrics": cal_busi_metrics,
        },
    }

    out_json = OUTPUT_DIR / f"metrics_{name}.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    save_individual_roc(name, views, y_busi, busi_df, busi_cal, cal_busi_metrics, single_metrics)

    return result, busi_cal


def normalize_weights(weights, views):
    total = sum(float(weights.get(view, 0.0)) for view in views)
    if total <= 0:
        return {view: 1.0 / len(views) for view in views}
    return {view: float(weights.get(view, 0.0)) / total for view in views}


def save_individual_roc(name, views, y_true, busi_df, fused_probs, fused_metrics, single_metrics):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {
            "full": "#1f77b4",
            "cut_borders": "#ff7f0e",
            "border": "#9467bd",
            "masked": "#2ca02c",
        }

        plt.figure(figsize=(9, 7.5))
        for view in views:
            points = roc_curve_points(y_true, busi_df[f"prob_{view}"].values)
            auc = single_metrics[view]["AUC"]
            auc_label = "N/A" if auc is None else f"{auc:.4f}"
            plt.plot(
                [p["fpr"] for p in points],
                [p["tpr"] for p in points],
                linewidth=1.2,
                color=colors[view],
                label=f"{view} (AUC={auc_label})",
            )

        points = roc_curve_points(y_true, fused_probs)
        auc = fused_metrics["AUC"]
        auc_label = "N/A" if auc is None else f"{auc:.4f}"
        plt.plot(
            [p["fpr"] for p in points],
            [p["tpr"] for p in points],
            linewidth=2.4,
            color="black",
            label=f"Original-style fusion (AUC={auc_label})",
        )

        plt.plot([0, 1], [0, 1], linewidth=1, color="#777777", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"BUSI ROC - {len(views)}-View Fixed-Weight Platt Fusion with Single Views")
        plt.legend(loc="lower right", fontsize=7.5)
        plt.grid(alpha=0.25)
        plt.tight_layout()

        out_png = OUTPUT_DIR / f"roc_busi_{len(views)}view_fixed_weight_platt_fusion_with_single_views.png"
        plt.savefig(out_png, dpi=220)
        plt.close()
        print(f"Saved: {out_png}")

        out_csv = OUTPUT_DIR / f"roc_busi_{len(views)}view_fixed_weight_platt_fusion_with_single_views_points.csv"
        pd.DataFrame(points).to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")
    except Exception as exc:
        print(f"matplotlib unavailable, skipped ROC plot: {exc}")


def save_combined_roc(results_and_probs):
    df = pd.read_csv(OUTPUT_DIR / "predictions.csv")
    y_true = df["true_label_id"].astype(int).values

    curves = []
    for view in ALL_VIEWS:
        probs = df[f"prob_{view}"].astype(float).values
        curves.append((f"Single {view}", probs, float(roc_auc_score(y_true, probs))))

    for name, result, probs in results_and_probs:
        metrics = result["platt_calibrated"]["busi_metrics"]
        curves.append((f"{len(result['views_used'])}-view fixed-weight Platt fusion", probs, metrics["AUC"]))

    curves.append(("4-view fixed-weight Platt fusion", df["prob_fused"].astype(float).values, float(roc_auc_score(y_true, df["prob_fused"].astype(float).values))))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        styles = {
            "Single full": ("#1f77b4", "-", 1.2),
            "Single cut_borders": ("#ff7f0e", "-", 1.2),
            "Single border": ("#9467bd", "-", 1.2),
            "Single masked": ("#2ca02c", "-", 1.2),
            "2-view": ("#d62728", "--", 2.0),
            "3-view": ("#111111", "-.", 2.1),
            "4-view": ("#005f73", "-", 2.4),
        }

        plt.figure(figsize=(9.5, 8.0))
        rows = []
        for label, probs, auc in curves:
            points = roc_curve_points(y_true, probs)
            style_key = next((key for key in styles if label.startswith(key)), None)
            color, linestyle, width = styles.get(style_key, ("#444444", "-", 1.4))
            plt.plot(
                [p["fpr"] for p in points],
                [p["tpr"] for p in points],
                color=color,
                linestyle=linestyle,
                linewidth=width,
                label=f"{label} (AUC={auc:.4f})",
            )
            for point in points:
                rows.append({"curve": label, "auc": auc, **point})

        plt.plot([0, 1], [0, 1], linewidth=1, color="#777777", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("BUSI ROC - Fixed-Weight Platt Fusion Comparison: 2-View vs 3-View vs 4-View")
        plt.legend(loc="lower right", fontsize=7.0)
        plt.grid(alpha=0.25)
        plt.tight_layout()

        out_png = OUTPUT_DIR / "roc_busi_fixed_weight_platt_fusion_2view_3view_4view_comparison.png"
        plt.savefig(out_png, dpi=220)
        plt.close()
        print(f"Saved: {out_png}")

        out_csv = OUTPUT_DIR / "roc_busi_fixed_weight_platt_fusion_2view_3view_4view_comparison_points.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")
    except Exception as exc:
        print(f"matplotlib unavailable, skipped combined ROC plot: {exc}")


def print_metrics(label, metrics):
    auc = "N/A" if metrics["AUC"] is None else f"{metrics['AUC']:.4f}"
    print(
        f"  {label:<34s} AUC={auc}  ACC={metrics['Accuracy']:.4f}  "
        f"Se={metrics['Sensitivity']:.4f}  Sp={metrics['Specificity']:.4f}  "
        f"Prec={metrics['Precision']:.4f}  F1={metrics['F1-score']:.4f}"
    )


def main():
    print("=" * 78)
    print("Fixed-weight Platt fusion applied to 2-view and 3-view")
    print("Fixed baseline weights + available-view normalization + OOF Platt + threshold 0.5")
    print("=" * 78)

    mask_flags = load_mask_flags()
    results_and_probs = []

    for name, views in FUSION_CONFIGS.items():
        print(f"\n{name}: {views}")
        result, probs = evaluate_config(name, views, mask_flags)
        results_and_probs.append((name, result, probs))
        print(f"  normalized ok weights      : {result['normalized_weights_when_mask_ok']}")
        print(f"  normalized fallback weights: {result['normalized_weights_when_fallback']}")
        print(f"  Platt: a={result['platt']['a']:.4f}, b={result['platt']['b']:.4f}")
        print_metrics("BUSI raw", result["raw"]["busi_metrics"])
        print_metrics("BUSI Platt threshold=0.5", result["platt_calibrated"]["busi_metrics"])

    save_combined_roc(results_and_probs)

    metrics_4 = json.loads((OUTPUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    print("\nReference fixed-weight 4-view:")
    print_metrics("4-view fixed-weight Platt fusion", metrics_4["weighted_fusion_metrics"])

    print("\nFinal fixed-weight Platt fusion summary:")
    for name, result, _ in results_and_probs:
        print_metrics(name, result["platt_calibrated"]["busi_metrics"])
    print_metrics("4view_fixed_weight_platt_fusion", metrics_4["weighted_fusion_metrics"])


if __name__ == "__main__":
    main()
