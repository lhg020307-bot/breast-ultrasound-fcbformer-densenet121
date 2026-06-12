"""Plot single-view, 2-view, 3-view, and OOF-weighted 4-view ROC curves."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results" / "competition"
SINGLE_VIEWS = ["full", "cut_borders", "border", "masked"]


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


def weighted_probs(df, views, weights):
    total = sum(float(weights.get(view, 0.0)) for view in views)
    if total <= 0:
        total = float(len(views))
        weights = {view: 1.0 for view in views}

    probs = np.zeros(len(df), dtype=float)
    for view in views:
        probs += (float(weights[view]) / total) * df[f"prob_{view}"].astype(float).values
    return probs


def apply_platt(probs, a, b):
    clipped = np.clip(np.asarray(probs, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    logits = np.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + np.exp(-(float(a) * logits + float(b))))


def add_curve(curves, y_true, name, probs):
    curves.append(
        {
            "name": name,
            "auc": float(roc_auc_score(y_true, probs)),
            "points": roc_curve_points(y_true, probs),
        }
    )


def main():
    df = pd.read_csv(OUTPUT_DIR / "predictions.csv")
    y_true = df["true_label_id"].astype(int).values

    curves = []
    for view in SINGLE_VIEWS:
        add_curve(curves, y_true, f"Single {view}", df[f"prob_{view}"].values)

    metrics_2 = json.loads((OUTPUT_DIR / "metrics_2view.json").read_text(encoding="utf-8"))
    primary_2 = metrics_2["primary_result"]
    views_2 = primary_2["views"]
    raw_2 = weighted_probs(df, views_2, primary_2["best_weights_from_oof"])
    platt_2 = primary_2["weighted_fusion_2view_platt"]["platt"]
    prob_2 = apply_platt(raw_2, platt_2["a"], platt_2["b"])
    add_curve(curves, y_true, f"2-view {metrics_2['primary_pair_selected_on_oof']} + Platt", prob_2)

    metrics_3 = json.loads((OUTPUT_DIR / "metrics_3view.json").read_text(encoding="utf-8"))
    views_3 = metrics_3["views_used"]
    raw_3 = weighted_probs(df, views_3, metrics_3["weighted_fusion_3view_platt"]["best_weights_from_oof"])
    platt_3 = metrics_3["weighted_fusion_3view_platt"]["platt"]
    prob_3 = apply_platt(raw_3, platt_3["a"], platt_3["b"])
    add_curve(curves, y_true, "3-view full+cut_borders+masked + Platt", prob_3)

    metrics_4 = json.loads((OUTPUT_DIR / "metrics_4view.json").read_text(encoding="utf-8"))
    views_4 = metrics_4["views_used"]
    raw_4 = weighted_probs(df, views_4, metrics_4["weighted_fusion_4view_platt"]["best_weights_from_oof"])
    platt_4 = metrics_4["weighted_fusion_4view_platt"]["platt"]
    prob_4 = apply_platt(raw_4, platt_4["a"], platt_4["b"])
    add_curve(curves, y_true, "4-view OOF-weighted + Platt", prob_4)

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
        for curve in curves:
            style_key = next((key for key in styles if curve["name"].startswith(key)), None)
            color, linestyle, width = styles.get(style_key, ("#444444", "-", 1.4))
            plt.plot(
                [p["fpr"] for p in curve["points"]],
                [p["tpr"] for p in curve["points"]],
                color=color,
                linestyle=linestyle,
                linewidth=width,
                label=f"{curve['name']} (AUC={curve['auc']:.4f})",
            )

        plt.plot([0, 1], [0, 1], linewidth=1, color="#777777", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Comparison - Views, 2-view, 3-view, and 4-view Fusion")
        plt.legend(loc="lower right", fontsize=7.0)
        plt.grid(alpha=0.25)
        plt.tight_layout()

        out_png = OUTPUT_DIR / "roc_curve_all_views_2_3_4_oof_weighted.png"
        plt.savefig(out_png, dpi=220)
        plt.close()
        print(f"Saved: {out_png}")
    except Exception as exc:
        print(f"matplotlib unavailable, skipped ROC plot: {exc}")

    rows = []
    for curve in curves:
        for point in curve["points"]:
            rows.append(
                {
                    "curve": curve["name"],
                    "auc": curve["auc"],
                    "fpr": point["fpr"],
                    "tpr": point["tpr"],
                    "threshold": point["threshold"],
                }
            )
    out_csv = OUTPUT_DIR / "roc_curve_all_views_2_3_4_oof_weighted_points.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    print("\nROC curves included:")
    for curve in curves:
        print(f"  {curve['name']}: AUC={curve['auc']:.4f}")


if __name__ == "__main__":
    main()
