"""Evaluate 4-view fusion with OOF-selected weights and thresholds.

This script reuses existing OOF predictions and BUSI predictions. It does not
retrain any model and does not overwrite the original metrics.json/roc_curve.png.
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
EPS = 1e-7
FOUR_VIEWS = ["full", "cut_borders", "border", "masked"]
OBJECTIVE = "f1"
MIN_SE = 0.80
MIN_SP = 0.75


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


def metric_value(metrics):
    if OBJECTIVE == "youden":
        return metrics["Sensitivity"] + metrics["Specificity"] - 1.0
    return metrics["F1-score"]


def find_best_threshold(
    y_true,
    probs,
    steps=301,
    min_sensitivity=MIN_SE,
    min_specificity=MIN_SP,
):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    pos_mask = y_true == 1
    neg_mask = ~pos_mask
    n_pos = max(int(pos_mask.sum()), 1)
    n_neg = max(int(neg_mask.sum()), 1)
    n_total = max(len(y_true), 1)

    best_t = 0.5
    best_val = -1.0
    found = False

    for t in np.linspace(0.1, 0.8, steps):
        pred = probs >= float(t)
        tp = int((pred & pos_mask).sum())
        fp = int((pred & neg_mask).sum())
        fn = n_pos - tp
        tn = n_neg - fp

        sensitivity = float(tp / (tp + fn + EPS))
        specificity = float(tn / (tn + fp + EPS))
        if sensitivity < min_sensitivity or specificity < min_specificity:
            continue

        precision = float(tp / (tp + fp + EPS))
        f1 = float(2.0 * precision * sensitivity / (precision + sensitivity + EPS))
        accuracy = float((tp + tn) / n_total)
        value = sensitivity + specificity - 1.0 if OBJECTIVE == "youden" else f1
        _ = accuracy

        if value > best_val:
            best_t = float(t)
            best_val = value
            found = True

    if not found:
        return find_best_threshold(
            y_true,
            probs,
            steps=steps,
            min_sensitivity=0.0,
            min_specificity=0.0,
        )

    return best_t, classification_metrics(y_true, probs, best_t)


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


def load_oof_data(metrics_dir, backbone="densenet121"):
    merged = None
    for view in FOUR_VIEWS:
        path = metrics_dir / f"oof_{view}_{backbone}.csv"
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        df = df.groupby(["sample_id", "dataset"], as_index=False).agg(
            {f"prob_{view}": "mean", "y_true": "first"}
        )
        df = df[["sample_id", "dataset", "y_true", f"prob_{view}"]]
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["sample_id", "dataset", "y_true"], how="inner")

    return merged


def generate_weight_candidates(step=0.05):
    n_steps = int(round(1.0 / step))
    candidates = []
    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            for k in range(n_steps + 1 - i - j):
                l = n_steps - i - j - k
                candidates.append(
                    {
                        "full": round(i * step, 4),
                        "cut_borders": round(j * step, 4),
                        "border": round(k * step, 4),
                        "masked": round(l * step, 4),
                    }
                )
    return candidates


def search_best_weights(oof_df, y_oof, step=0.05):
    candidates = generate_weight_candidates(step)
    print(f"  Grid search: {len(candidates)} weight combinations (step={step})")

    best_weights = None
    best_threshold = 0.5
    best_metrics = None
    best_value = -1.0

    for weights in candidates:
        probs = weighted_probs(oof_df, FOUR_VIEWS, weights)
        threshold, metrics = find_best_threshold(
            y_oof,
            probs,
            steps=301,
            min_sensitivity=MIN_SE,
            min_specificity=MIN_SP,
        )
        value = metric_value(metrics)
        if value > best_value:
            best_weights = weights
            best_threshold = threshold
            best_metrics = metrics
            best_value = value

    return best_weights, best_threshold, best_metrics


def print_metrics(name, threshold, metrics):
    auc = "N/A" if metrics["AUC"] is None else f"{metrics['AUC']:.4f}"
    print(
        f"  {name:<30s} thr={threshold:.4f}  AUC={auc}  "
        f"F1={metrics['F1-score']:.4f}  ACC={metrics['Accuracy']:.4f}  "
        f"Se={metrics['Sensitivity']:.4f}  Sp={metrics['Specificity']:.4f}"
    )


def save_roc_plot(output_dir, y_busi, busi_df, raw_probs, cal_probs, raw_metrics, cal_metrics):
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
        single_metrics = {}
        for view in FOUR_VIEWS:
            metrics = classification_metrics(y_busi, busi_df[f"prob_{view}"].values, 0.5)
            single_metrics[view] = metrics
            points = roc_curve_points(y_busi, busi_df[f"prob_{view}"].values)
            auc = "N/A" if metrics["AUC"] is None else f"{metrics['AUC']:.4f}"
            plt.plot(
                [p["fpr"] for p in points],
                [p["tpr"] for p in points],
                linewidth=1.2,
                color=colors[view],
                label=f"{view} (AUC={auc})",
            )

        raw_points = roc_curve_points(y_busi, raw_probs)
        raw_auc = "N/A" if raw_metrics["AUC"] is None else f"{raw_metrics['AUC']:.4f}"
        plt.plot(
            [p["fpr"] for p in raw_points],
            [p["tpr"] for p in raw_points],
            linewidth=1.8,
            linestyle="--",
            color="#d62728",
            label=f"Weighted 4-view raw (AUC={raw_auc})",
        )

        cal_points = roc_curve_points(y_busi, cal_probs)
        cal_auc = "N/A" if cal_metrics["AUC"] is None else f"{cal_metrics['AUC']:.4f}"
        plt.plot(
            [p["fpr"] for p in cal_points],
            [p["tpr"] for p in cal_points],
            linewidth=2.4,
            color="black",
            label=f"Weighted 4-view + Platt (AUC={cal_auc})",
        )

        plt.plot([0, 1], [0, 1], linewidth=1, color="#777777", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("4-View Fusion ROC - BUSI (weights and threshold selected on OOF)")
        plt.legend(loc="lower right", fontsize=7.2)
        plt.grid(alpha=0.25)
        plt.tight_layout()

        out_png = output_dir / "roc_curve_4view.png"
        plt.savefig(out_png, dpi=220)
        plt.close()
        print(f"Saved: {out_png}")

        points_path = output_dir / "roc_curve_4view_points.csv"
        pd.DataFrame(cal_points).to_csv(points_path, index=False)
        print(f"Saved: {points_path}")
    except Exception as exc:
        print(f"matplotlib unavailable, skipped ROC plot: {exc}")


def main():
    output_dir = PROJECT_ROOT / "outputs" / "results" / "competition"
    metrics_dir = PROJECT_ROOT / "outputs" / "results" / "oof"
    pred_path = output_dir / "predictions.csv"

    print("=" * 76)
    print("4-View Fusion - OOF-selected weights and thresholds")
    print(f"Objective: {OBJECTIVE} | min Se={MIN_SE} | min Sp={MIN_SP}")
    print("=" * 76)

    busi_df = pd.read_csv(pred_path)
    y_busi = busi_df["true_label_id"].astype(int).values
    print(f"BUSI eval samples: {len(busi_df)}")

    oof_df = load_oof_data(metrics_dir)
    y_oof = oof_df["y_true"].astype(int).values
    print(f"OOF samples      : {len(oof_df)}")
    unique, counts = np.unique(y_oof, return_counts=True)
    print(f"OOF class dist   : benign={counts[0]}  malignant={counts[1]}")

    single_metrics = {}
    print("\nSingle-view performance (threshold=0.5, BUSI reference)")
    for view in FOUR_VIEWS:
        metrics = classification_metrics(y_busi, busi_df[f"prob_{view}"].values, 0.5)
        single_metrics[view] = metrics
        print_metrics(view, 0.5, metrics)

    print("\nWeighted fusion 4v - grid-search weights + threshold on OOF")
    best_weights, raw_threshold, raw_oof_metrics = search_best_weights(oof_df, y_oof)
    print(f"  Best weights (OOF): {best_weights}")
    print_metrics("OOF weighted raw", raw_threshold, raw_oof_metrics)

    oof_raw = weighted_probs(oof_df, FOUR_VIEWS, best_weights)
    busi_raw = weighted_probs(busi_df, FOUR_VIEWS, best_weights)
    raw_busi_metrics = classification_metrics(y_busi, busi_raw, raw_threshold)
    print_metrics("BUSI weighted raw", raw_threshold, raw_busi_metrics)

    print("\nWeighted fusion + Platt calibration (OOF fitted, OOF threshold)")
    oof_clipped = np.clip(oof_raw, 1e-9, 1.0 - 1e-9)
    oof_logits = np.log(oof_clipped / (1.0 - oof_clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
    calibrator.fit(oof_logits, y_oof)
    platt_a = float(calibrator.coef_[0, 0])
    platt_b = float(calibrator.intercept_[0])

    oof_cal = apply_platt(oof_raw, platt_a, platt_b)
    cal_threshold, cal_oof_metrics = find_best_threshold(
        y_oof,
        oof_cal,
        min_sensitivity=MIN_SE,
        min_specificity=MIN_SP,
    )
    busi_cal = apply_platt(busi_raw, platt_a, platt_b)
    cal_busi_metrics = classification_metrics(y_busi, busi_cal, cal_threshold)

    print(f"  Platt: a={platt_a:.4f}  b={platt_b:.4f}")
    print_metrics("OOF weighted + Platt", cal_threshold, cal_oof_metrics)
    print_metrics("BUSI weighted + Platt", cal_threshold, cal_busi_metrics)

    result = {
        "eval_dataset": "BUSI",
        "calibration_dataset": "OOF (BUS+BUSBRA)",
        "n_oof": int(len(y_oof)),
        "n_busi": int(len(y_busi)),
        "views_used": FOUR_VIEWS,
        "objective": OBJECTIVE,
        "min_sensitivity_constraint": MIN_SE,
        "min_specificity_constraint": MIN_SP,
        "single_view_metrics_busi": single_metrics,
        "weighted_fusion_4view_raw": {
            "best_weights_from_oof": best_weights,
            "oof_best_threshold": raw_threshold,
            "oof_metrics_at_best": raw_oof_metrics,
            "busi_metrics": raw_busi_metrics,
        },
        "weighted_fusion_4view_platt": {
            "best_weights_from_oof": best_weights,
            "platt": {"a": platt_a, "b": platt_b},
            "oof_best_threshold": cal_threshold,
            "oof_metrics_at_best": cal_oof_metrics,
            "busi_metrics": cal_busi_metrics,
        },
    }

    out_json = output_dir / "metrics_4view.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_json}")

    save_roc_plot(
        output_dir,
        y_busi,
        busi_df,
        busi_raw,
        busi_cal,
        raw_busi_metrics,
        cal_busi_metrics,
    )

    print(f"\n{'=' * 76}")
    print("FINAL SUMMARY - 4-view OOF-weighted fusion")
    print(f"{'Method':<30s} {'Thr':>8s} {'AUC':>8s} {'F1':>8s} {'ACC':>8s} {'Se':>8s} {'Sp':>8s}")
    print("-" * 76)
    for view in FOUR_VIEWS:
        metrics = single_metrics[view]
        auc = "N/A" if metrics["AUC"] is None else f"{metrics['AUC']:.4f}"
        print(
            f"{view:<30s} {0.5:>8.4f} {auc:>8s} "
            f"{metrics['F1-score']:>8.4f} {metrics['Accuracy']:>8.4f} "
            f"{metrics['Sensitivity']:>8.4f} {metrics['Specificity']:>8.4f}"
        )
    print("-" * 76)
    for name, threshold, metrics in [
        ("Weighted 4v raw", raw_threshold, raw_busi_metrics),
        ("Weighted 4v + Platt", cal_threshold, cal_busi_metrics),
    ]:
        auc = "N/A" if metrics["AUC"] is None else f"{metrics['AUC']:.4f}"
        print(
            f"{name:<30s} {threshold:>8.4f} {auc:>8s} "
            f"{metrics['F1-score']:>8.4f} {metrics['Accuracy']:>8.4f} "
            f"{metrics['Sensitivity']:>8.4f} {metrics['Specificity']:>8.4f}"
        )
    print("=" * 76)
    print(f"Optimized weights: {best_weights}")


if __name__ == "__main__":
    main()
