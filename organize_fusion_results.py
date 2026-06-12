"""Organize fusion outputs into two named result sets.

Result sets:
  fixed_weight_platt_fusion
    Fixed original weights, available-view normalization, OOF Platt, threshold 0.5.

  oof_search_weight_platt_fusion
    OOF-searched weights/thresholds with Platt calibration.

This script does not retrain models and does not remove legacy files. It only
copies metrics, regenerates standardized ROC plots, and writes summary tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent
COMPETITION_DIR = PROJECT_ROOT / "outputs" / "results" / "competition"
VIEWS_INDEX = PROJECT_ROOT / "outputs" / "results" / "views_index.csv"

FIXED_DIR = COMPETITION_DIR / "fixed_weight_platt_fusion"
OOF_SEARCH_DIR = COMPETITION_DIR / "oof_search_weight_platt_fusion"

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
LEGEND_FONT_SIZE = 18
LEGEND_MARKER_SCALE = 1.8


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def apply_platt(probs, a, b):
    clipped = np.clip(np.asarray(probs, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    logits = np.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + np.exp(-(float(a) * logits + float(b))))


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


def normalized_weights(weights, views):
    total = sum(float(weights.get(view, 0.0)) for view in views)
    if total <= 0:
        return {view: 1.0 / len(views) for view in views}
    return {view: float(weights.get(view, 0.0)) / total for view in views}


def load_mask_flags(sample_ids):
    if not VIEWS_INDEX.exists():
        return pd.Series(["fallback"] * len(sample_ids), index=sample_ids)

    flags = pd.read_csv(VIEWS_INDEX)
    if "mask_quality_flag" not in flags.columns:
        return pd.Series(["fallback"] * len(sample_ids), index=sample_ids)

    flags = flags[["sample_id", "mask_quality_flag"]].copy()
    flags["sample_id"] = flags["sample_id"].astype(str)
    flags = flags.drop_duplicates("sample_id").set_index("sample_id")["mask_quality_flag"]
    return pd.Series(sample_ids, index=sample_ids).map(flags).fillna("fallback")


def fixed_weight_probs(pred_df, views, platt=None):
    sample_ids = pred_df["sample_id"].astype(str).tolist()
    flags = load_mask_flags(sample_ids).astype(str).tolist()
    raw = np.zeros(len(pred_df), dtype=float)

    for idx, flag in enumerate(flags):
        base = ORIGINAL_NORMAL_WEIGHTS if flag == "ok" else ORIGINAL_FALLBACK_WEIGHTS
        weights = normalized_weights(base, views)
        raw[idx] = sum(weights[view] * float(pred_df.iloc[idx][f"prob_{view}"]) for view in views)

    if platt is None:
        return raw
    return apply_platt(raw, platt["a"], platt["b"])


def weighted_probs(pred_df, views, weights, platt=None):
    total = sum(float(weights.get(view, 0.0)) for view in views)
    if total <= 0:
        weights = {view: 1.0 for view in views}
        total = float(len(views))

    probs = np.zeros(len(pred_df), dtype=float)
    for view in views:
        probs += (float(weights[view]) / total) * pred_df[f"prob_{view}"].astype(float).values

    if platt is None:
        return probs
    return apply_platt(probs, platt["a"], platt["b"])


def score_metric(metrics):
    return (
        metrics["AUC"] * 15
        + metrics["Accuracy"] * 10
        + metrics["Sensitivity"] * 15
        + metrics["Specificity"] * 10
        + metrics["Precision"] * 5
        + metrics["F1-score"] * 5
    )


def pct(value):
    return f"{value * 100:.2f}%"


def make_summary(rows, out_dir, stem):
    csv_rows = []
    md_rows = []
    for rank, row in enumerate(sorted(rows, key=lambda x: x["competition_score"], reverse=True), start=1):
        metrics = row["metrics"]
        csv_rows.append(
            {
                "rank": rank,
                "method": row["method"],
                "views": row["views"],
                "threshold": metrics["threshold"],
                "AUC": metrics["AUC"],
                "Accuracy": metrics["Accuracy"],
                "Sensitivity": metrics["Sensitivity"],
                "Specificity": metrics["Specificity"],
                "Precision": metrics["Precision"],
                "F1-score": metrics["F1-score"],
                "competition_score": row["competition_score"],
            }
        )
        md_rows.append(
            [
                str(rank),
                row["method"],
                row["views"],
                f"{metrics['threshold']:.3f}",
                pct(metrics["AUC"]),
                pct(metrics["Accuracy"]),
                pct(metrics["Sensitivity"]),
                pct(metrics["Specificity"]),
                pct(metrics["Precision"]),
                pct(metrics["F1-score"]),
                f"{row['competition_score']:.2f}",
            ]
        )

    pd.DataFrame(csv_rows).to_csv(out_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")

    headers = [
        "Rank",
        "Method",
        "Views",
        "Threshold",
        "AUC",
        "Acc",
        "Sen",
        "Spec",
        "Prec",
        "F1",
        "Score",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in md_rows:
        lines.append("| " + " | ".join(row) + " |")
    (out_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_roc(output_path, title, curves, y_true, points_path=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "full": "#1f77b4",
        "cut_borders": "#ff7f0e",
        "border": "#9467bd",
        "masked": "#2ca02c",
    }

    plt.figure(figsize=(10.24, 7.68), dpi=200)
    all_points = []

    for curve in curves:
        points = roc_curve_points(y_true, curve["scores"])
        auc = float(roc_auc_score(y_true, curve["scores"]))
        color = curve.get("color", colors.get(curve["name"], "black"))
        linewidth = curve.get("linewidth", 1.6)
        linestyle = curve.get("linestyle", "-")
        plt.plot(
            [p["fpr"] for p in points],
            [p["tpr"] for p in points],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=f"{curve['label']} (AUC={auc:.4f})",
        )
        for point in points:
            all_points.append({"curve": curve["label"], "auc": auc, **point})

    plt.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.3)
    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate", fontsize=13)
    plt.title(title, fontsize=16, pad=10)
    plt.legend(
        loc="lower right",
        fontsize=LEGEND_FONT_SIZE,
        markerscale=LEGEND_MARKER_SCALE,
        frameon=True,
    )
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    if points_path is not None:
        pd.DataFrame(all_points).to_csv(points_path, index=False)


def copy_metric(src_names, dst_dir, dst_name, method_name=None):
    if isinstance(src_names, str):
        src_names = [src_names]

    src = None
    for src_name in src_names:
        candidate = COMPETITION_DIR / src_name
        if candidate.exists():
            src = candidate
            break
    if src is None:
        raise FileNotFoundError(f"None of these metric files exist: {src_names}")

    dst = dst_dir / dst_name
    data = load_json(src)
    if method_name is not None:
        data["method"] = method_name
    save_json(dst, data)
    return data


def build_fixed_set(pred_df, y_true):
    FIXED_DIR.mkdir(parents=True, exist_ok=True)

    metrics_2 = copy_metric(
        ["metrics_2view_fixed_weight_platt_fusion.json", "metrics_2view_original_fusion.json"],
        FIXED_DIR,
        "metrics_2view_fixed_weight_platt_fusion.json",
        "fixed_weight_platt_fusion",
    )
    metrics_3 = copy_metric(
        ["metrics_3view_fixed_weight_platt_fusion.json", "metrics_3view_original_fusion.json"],
        FIXED_DIR,
        "metrics_3view_fixed_weight_platt_fusion.json",
        "fixed_weight_platt_fusion",
    )
    metrics_4 = copy_metric(
        ["metrics_4view_fixed_weight_platt_fusion.json", "metrics_4view_original_fusion.json"],
        FIXED_DIR,
        "metrics_4view_fixed_weight_platt_fusion.json",
        "fixed_weight_platt_fusion",
    )

    metrics_by_view = {
        2: metrics_2,
        3: metrics_3,
        4: metrics_4,
    }

    rows = []
    fusion_curves = []
    for view_count, metrics in metrics_by_view.items():
        views = metrics["views_used"]
        platt = metrics["platt"]
        fusion_scores = fixed_weight_probs(pred_df, views, platt)
        fusion_metrics = metrics["platt_calibrated"]["busi_metrics"]
        rows.append(
            {
                "method": f"{view_count}-view fixed-weight Platt fusion",
                "views": "+".join(views),
                "metrics": fusion_metrics,
                "competition_score": score_metric(fusion_metrics),
            }
        )
        fusion_curves.append(
            {
                "name": f"{view_count}-view",
                "label": f"{view_count}-view fixed-weight Platt fusion",
                "scores": fusion_scores,
                "color": "black" if view_count == 4 else ("#d62728" if view_count == 2 else "#111111"),
                "linewidth": 3.0 if view_count == 4 else 2.2,
                "linestyle": "-" if view_count == 4 else ("--" if view_count == 2 else "-."),
            }
        )

        individual_curves = []
        for view in views:
            individual_curves.append(
                {
                    "name": view,
                    "label": view,
                    "scores": pred_df[f"prob_{view}"].astype(float).values,
                    "linewidth": 1.6,
                }
            )
        individual_curves.append(
            {
                "name": f"{view_count}-view",
                "label": "Fixed-weight Platt fusion",
                "scores": fusion_scores,
                "color": "black",
                "linewidth": 3.0,
            }
        )
        plot_roc(
            FIXED_DIR / f"roc_busi_{view_count}view_fixed_weight_platt_fusion_with_single_views.png",
            f"BUSI ROC - {view_count}-View Fixed-Weight Platt Fusion with Single Views",
            individual_curves,
            y_true,
            FIXED_DIR / f"roc_busi_{view_count}view_fixed_weight_platt_fusion_with_single_views_points.csv",
        )

    comparison_curves = []
    for view in ALL_VIEWS:
        comparison_curves.append(
            {
                "name": view,
                "label": view,
                "scores": pred_df[f"prob_{view}"].astype(float).values,
                "linewidth": 1.4,
            }
        )
    comparison_curves.extend(fusion_curves)
    plot_roc(
        FIXED_DIR / "roc_busi_fixed_weight_platt_fusion_2view_3view_4view_comparison.png",
        "BUSI ROC - Fixed-Weight Platt Fusion Comparison: 2-View vs 3-View vs 4-View",
        comparison_curves,
        y_true,
        FIXED_DIR / "roc_busi_fixed_weight_platt_fusion_2view_3view_4view_comparison_points.csv",
    )
    make_summary(rows, FIXED_DIR, "summary_fixed_weight_platt_fusion_metrics")


def build_oof_search_set(pred_df, y_true):
    OOF_SEARCH_DIR.mkdir(parents=True, exist_ok=True)

    metrics_2 = copy_metric(
        "metrics_2view.json",
        OOF_SEARCH_DIR,
        "metrics_2view_oof_search_weight_platt_fusion.json",
        "oof_search_weight_platt_fusion",
    )
    metrics_3 = copy_metric(
        "metrics_3view.json",
        OOF_SEARCH_DIR,
        "metrics_3view_oof_search_weight_platt_fusion.json",
        "oof_search_weight_platt_fusion",
    )
    metrics_4 = copy_metric(
        "metrics_4view.json",
        OOF_SEARCH_DIR,
        "metrics_4view_oof_search_weight_platt_fusion.json",
        "oof_search_weight_platt_fusion",
    )

    configs = []
    primary_2 = metrics_2["primary_result"]
    configs.append(
        {
            "view_count": 2,
            "views": primary_2["views"],
            "weights": primary_2["best_weights_from_oof"],
            "platt": primary_2["weighted_fusion_2view_platt"]["platt"],
            "metrics": primary_2["weighted_fusion_2view_platt"]["busi_metrics"],
        }
    )
    configs.append(
        {
            "view_count": 3,
            "views": metrics_3["views_used"],
            "weights": metrics_3["weighted_fusion_3view_platt"]["best_weights_from_oof"],
            "platt": metrics_3["weighted_fusion_3view_platt"]["platt"],
            "metrics": metrics_3["weighted_fusion_3view_platt"]["busi_metrics"],
        }
    )
    configs.append(
        {
            "view_count": 4,
            "views": metrics_4["views_used"],
            "weights": metrics_4["weighted_fusion_4view_platt"]["best_weights_from_oof"],
            "platt": metrics_4["weighted_fusion_4view_platt"]["platt"],
            "metrics": metrics_4["weighted_fusion_4view_platt"]["busi_metrics"],
        }
    )

    rows = []
    fusion_curves = []
    for config in configs:
        view_count = config["view_count"]
        views = config["views"]
        fusion_scores = weighted_probs(pred_df, views, config["weights"], config["platt"])
        rows.append(
            {
                "method": f"{view_count}-view OOF-search weight Platt fusion",
                "views": "+".join(views),
                "metrics": config["metrics"],
                "competition_score": score_metric(config["metrics"]),
            }
        )
        fusion_curves.append(
            {
                "name": f"{view_count}-view",
                "label": f"{view_count}-view OOF-search weight Platt fusion",
                "scores": fusion_scores,
                "color": "black" if view_count == 4 else ("#d62728" if view_count == 2 else "#111111"),
                "linewidth": 3.0 if view_count == 4 else 2.2,
                "linestyle": "-" if view_count == 4 else ("--" if view_count == 2 else "-."),
            }
        )

        individual_curves = []
        for view in views:
            individual_curves.append(
                {
                    "name": view,
                    "label": view,
                    "scores": pred_df[f"prob_{view}"].astype(float).values,
                    "linewidth": 1.6,
                }
            )
        individual_curves.append(
            {
                "name": f"{view_count}-view",
                "label": "OOF-search weight Platt fusion",
                "scores": fusion_scores,
                "color": "black",
                "linewidth": 3.0,
            }
        )
        plot_roc(
            OOF_SEARCH_DIR / f"roc_busi_{view_count}view_oof_search_weight_platt_fusion_with_single_views.png",
            f"BUSI ROC - {view_count}-View OOF-Search Weight Platt Fusion with Single Views",
            individual_curves,
            y_true,
            OOF_SEARCH_DIR / f"roc_busi_{view_count}view_oof_search_weight_platt_fusion_with_single_views_points.csv",
        )

    comparison_curves = []
    for view in ALL_VIEWS:
        comparison_curves.append(
            {
                "name": view,
                "label": view,
                "scores": pred_df[f"prob_{view}"].astype(float).values,
                "linewidth": 1.4,
            }
        )
    comparison_curves.extend(fusion_curves)
    plot_roc(
        OOF_SEARCH_DIR / "roc_busi_oof_search_weight_platt_fusion_2view_3view_4view_comparison.png",
        "BUSI ROC - OOF-Search Weight Platt Fusion Comparison: 2-View vs 3-View vs 4-View",
        comparison_curves,
        y_true,
        OOF_SEARCH_DIR / "roc_busi_oof_search_weight_platt_fusion_2view_3view_4view_comparison_points.csv",
    )
    make_summary(rows, OOF_SEARCH_DIR, "summary_oof_search_weight_platt_fusion_metrics")


def main():
    pred_df = pd.read_csv(COMPETITION_DIR / "predictions.csv")
    y_true = pred_df["true_label_id"].astype(int).values

    build_fixed_set(pred_df, y_true)
    build_oof_search_set(pred_df, y_true)

    print(f"Saved fixed-weight Platt fusion outputs to: {FIXED_DIR}")
    print(f"Saved OOF-search weight Platt fusion outputs to: {OOF_SEARCH_DIR}")


if __name__ == "__main__":
    main()
