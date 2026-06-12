"""Evaluate 2-view fusion using OOF-selected weights and thresholds.

This script does not retrain models. It reuses the existing normalized outputs:
  outputs/results/oof/oof_<view>_densenet121.csv
  outputs/results/competition/predictions.csv

For fairness, every fusion weight, threshold, and the primary 2-view pair are
selected on OOF only. BUSI is evaluated once after those OOF decisions.
"""

from __future__ import annotations

import itertools
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
ALL_VIEWS = ["full", "cut_borders", "border", "masked"]
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


def metric_value(metrics, objective):
    if objective == "youden":
        return metrics["Sensitivity"] + metrics["Specificity"] - 1.0
    return metrics["F1-score"]


def find_best_threshold(
    y_true,
    probs,
    objective=OBJECTIVE,
    steps=701,
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
        if sensitivity < min_sensitivity:
            continue
        if specificity < min_specificity:
            continue

        precision = float(tp / (tp + fp + EPS))
        f1 = float(2.0 * precision * sensitivity / (precision + sensitivity + EPS))
        accuracy = float((tp + tn) / n_total)
        value = sensitivity + specificity - 1.0 if objective == "youden" else f1

        if value > best_val:
            best_t = float(t)
            best_val = value
            found = True

    if not found:
        return find_best_threshold(
            y_true,
            probs,
            objective=objective,
            steps=steps,
            min_sensitivity=0.0,
            min_specificity=0.0,
        )

    best_metrics = classification_metrics(y_true, probs, best_t)
    return best_t, best_metrics


def weighted_probs(df, views, weights):
    total = sum(weights.get(v, 0.0) for v in views)
    if total <= 0:
        weights = {v: 1.0 for v in views}
        total = float(len(views))

    result = np.zeros(len(df), dtype=float)
    for view in views:
        result += (weights[view] / total) * df[f"prob_{view}"].astype(float).values
    return result


def apply_platt(probs, a, b):
    clipped = np.clip(np.asarray(probs, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    logits = np.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + np.exp(-(a * logits + b)))


def roc_curve_points(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    thresholds = np.r_[np.inf, np.sort(np.unique(y_score))[::-1], -np.inf]
    pos = max(int((y_true == 1).sum()), 1)
    neg = max(int((y_true == 0).sum()), 1)
    points = []

    for threshold in thresholds:
        pred = (y_score >= threshold).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        points.append(
            {
                "fpr": fp / neg,
                "tpr": tp / pos,
                "threshold": float(threshold) if np.isfinite(threshold) else str(threshold),
            }
        )

    return points


def load_oof_data(metrics_dir, views, backbone="densenet121"):
    merged = None
    for view in views:
        path = metrics_dir / f"oof_{view}_{backbone}.csv"
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        df = df.groupby(["sample_id", "dataset"], as_index=False).agg(
            {f"prob_{view}": "mean", "y_true": "first"}
        )
        keep = ["sample_id", "dataset", "y_true", f"prob_{view}"]
        df = df[keep]

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["sample_id", "dataset", "y_true"], how="inner")

    return merged


def search_best_weights(oof_df, y_oof, views, step=0.01):
    best_weights = None
    best_threshold = 0.5
    best_metrics = None
    best_value = -1.0

    view_a, view_b = views
    for i in range(int(1.0 / step) + 1):
        w_a = round(i * step, 4)
        w_b = round(1.0 - w_a, 4)
        weights = {view_a: w_a, view_b: w_b}
        probs = weighted_probs(oof_df, views, weights)
        threshold, metrics = find_best_threshold(
            y_oof,
            probs,
            objective=OBJECTIVE,
            steps=301,
            min_sensitivity=MIN_SE,
            min_specificity=MIN_SP,
        )
        value = metric_value(metrics, OBJECTIVE)
        if value > best_value:
            best_value = value
            best_weights = weights
            best_threshold = threshold
            best_metrics = metrics

    return best_weights, best_threshold, best_metrics


def evaluate_pair(pair, busi_df, y_busi, metrics_dir):
    views = list(pair)
    oof_df = load_oof_data(metrics_dir, views)
    y_oof = oof_df["y_true"].astype(int).values

    weights, raw_threshold, raw_oof_metrics = search_best_weights(oof_df, y_oof, views)

    oof_raw = weighted_probs(oof_df, views, weights)
    busi_raw = weighted_probs(busi_df, views, weights)
    raw_busi_metrics = classification_metrics(y_busi, busi_raw, raw_threshold)

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
        objective=OBJECTIVE,
        min_sensitivity=MIN_SE,
        min_specificity=MIN_SP,
    )
    busi_cal = apply_platt(busi_raw, platt_a, platt_b)
    cal_busi_metrics = classification_metrics(y_busi, busi_cal, cal_threshold)

    return {
        "views": views,
        "n_oof": int(len(oof_df)),
        "best_weights_from_oof": weights,
        "weighted_fusion_2view_raw": {
            "oof_best_threshold": raw_threshold,
            "oof_metrics_at_best": raw_oof_metrics,
            "busi_metrics": raw_busi_metrics,
        },
        "weighted_fusion_2view_platt": {
            "platt": {"a": platt_a, "b": platt_b},
            "oof_best_threshold": cal_threshold,
            "oof_metrics_at_best": cal_oof_metrics,
            "busi_metrics": cal_busi_metrics,
        },
        "_scores": {
            "busi_raw": busi_raw,
            "busi_cal": busi_cal,
        },
    }


def print_metrics(name, threshold, metrics):
    auc = "N/A" if metrics["AUC"] is None else f"{metrics['AUC']:.4f}"
    print(
        f"  {name:<32s} thr={threshold:.4f}  AUC={auc}  "
        f"F1={metrics['F1-score']:.4f}  ACC={metrics['Accuracy']:.4f}  "
        f"Se={metrics['Sensitivity']:.4f}  Sp={metrics['Specificity']:.4f}"
    )


def save_roc_plot(output_dir, y_busi, busi_df, pair_results, primary_key):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 7.5))
        colors = {
            "full": "#1f77b4",
            "cut_borders": "#ff7f0e",
            "border": "#9467bd",
            "masked": "#2ca02c",
        }

        single_metrics = {}
        for view in ALL_VIEWS:
            metrics = classification_metrics(y_busi, busi_df[f"prob_{view}"].values, 0.5)
            single_metrics[view] = metrics
            points = roc_curve_points(y_busi, busi_df[f"prob_{view}"].values)
            auc = "N/A" if metrics["AUC"] is None else f"{metrics['AUC']:.4f}"
            plt.plot(
                [p["fpr"] for p in points],
                [p["tpr"] for p in points],
                linewidth=1.1,
                color=colors[view],
                label=f"{view} (AUC={auc})",
            )

        primary = pair_results[primary_key]
        primary_scores = primary["_scores"]["busi_cal"]
        primary_metrics = primary["weighted_fusion_2view_platt"]["busi_metrics"]
        primary_points = roc_curve_points(y_busi, primary_scores)
        primary_auc = primary_metrics["AUC"]
        primary_auc_text = "N/A" if primary_auc is None else f"{primary_auc:.4f}"
        plt.plot(
            [p["fpr"] for p in primary_points],
            [p["tpr"] for p in primary_points],
            linewidth=2.4,
            color="black",
            label=f"Primary 2-view + Platt (AUC={primary_auc_text})",
        )

        for key, result in pair_results.items():
            if key == primary_key:
                continue
            metrics = result["weighted_fusion_2view_platt"]["busi_metrics"]
            points = roc_curve_points(y_busi, result["_scores"]["busi_cal"])
            auc = "N/A" if metrics["AUC"] is None else f"{metrics['AUC']:.4f}"
            plt.plot(
                [p["fpr"] for p in points],
                [p["tpr"] for p in points],
                linewidth=1.0,
                linestyle="--",
                alpha=0.7,
                label=f"{key} + Platt (AUC={auc})",
            )

        plt.plot([0, 1], [0, 1], linewidth=1, color="#777777", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("2-View Fusion ROC - BUSI (weights, pair, threshold selected on OOF)")
        plt.legend(loc="lower right", fontsize=6.8)
        plt.grid(alpha=0.25)
        plt.tight_layout()

        out_png = output_dir / "roc_curve_2view.png"
        plt.savefig(out_png, dpi=200)
        plt.close()

        primary_points_path = output_dir / "roc_curve_2view_primary_points.csv"
        pd.DataFrame(primary_points).to_csv(primary_points_path, index=False)
        print(f"\nSaved: {out_png}")
        print(f"Saved: {primary_points_path}")
    except Exception as exc:
        print(f"matplotlib unavailable, skipped ROC plot: {exc}")


def main():
    output_dir = PROJECT_ROOT / "outputs" / "results" / "competition"
    metrics_dir = PROJECT_ROOT / "outputs" / "results" / "oof"
    pred_path = output_dir / "predictions.csv"

    print("=" * 76)
    print("2-View Fusion - OOF-selected pair, weights, and thresholds")
    print(f"Objective: {OBJECTIVE} | min Se={MIN_SE} | min Sp={MIN_SP}")
    print("=" * 76)

    busi_df = pd.read_csv(pred_path)
    y_busi = busi_df["true_label_id"].astype(int).values
    print(f"BUSI eval samples: {len(busi_df)}")
    print(f"Candidate views  : {ALL_VIEWS}")

    pair_results = {}
    for pair in itertools.combinations(ALL_VIEWS, 2):
        key = "+".join(pair)
        print(f"\nPair: {key}")
        result = evaluate_pair(pair, busi_df, y_busi, metrics_dir)
        pair_results[key] = result

        weights = result["best_weights_from_oof"]
        print(f"  OOF weights: {weights}")
        raw = result["weighted_fusion_2view_raw"]
        cal = result["weighted_fusion_2view_platt"]
        print_metrics("BUSI weighted raw", raw["oof_best_threshold"], raw["busi_metrics"])
        print_metrics("BUSI weighted + Platt", cal["oof_best_threshold"], cal["busi_metrics"])

    primary_key = max(
        pair_results,
        key=lambda key: metric_value(
            pair_results[key]["weighted_fusion_2view_platt"]["oof_metrics_at_best"],
            OBJECTIVE,
        ),
    )
    primary = pair_results[primary_key]

    serializable_results = {}
    for key, result in pair_results.items():
        serializable = {k: v for k, v in result.items() if k != "_scores"}
        serializable_results[key] = serializable

    output = {
        "eval_dataset": "BUSI",
        "calibration_dataset": "OOF (BUS+BUSBRA)",
        "views_available": ALL_VIEWS,
        "objective": OBJECTIVE,
        "min_sensitivity_constraint": MIN_SE,
        "min_specificity_constraint": MIN_SP,
        "primary_pair_selected_on_oof": primary_key,
        "primary_result": {k: v for k, v in primary.items() if k != "_scores"},
        "all_pair_results": serializable_results,
    }

    out_json = output_dir / "metrics_2view.json"
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_json}")

    save_roc_plot(output_dir, y_busi, busi_df, pair_results, primary_key)

    print(f"\n{'=' * 76}")
    print("FINAL SUMMARY - 2-view fusion")
    print(f"Primary pair selected on OOF: {primary_key}")
    print(f"{'Pair':<26s} {'Thr':>8s} {'AUC':>8s} {'F1':>8s} {'ACC':>8s} {'Se':>8s} {'Sp':>8s}")
    print("-" * 76)
    for key, result in pair_results.items():
        cal = result["weighted_fusion_2view_platt"]
        metrics = cal["busi_metrics"]
        threshold = cal["oof_best_threshold"]
        auc = "N/A" if metrics["AUC"] is None else f"{metrics['AUC']:.4f}"
        print(
            f"{key:<26s} {threshold:>8.4f} {auc:>8s} "
            f"{metrics['F1-score']:>8.4f} {metrics['Accuracy']:>8.4f} "
            f"{metrics['Sensitivity']:>8.4f} {metrics['Specificity']:>8.4f}"
        )
    print("=" * 76)


if __name__ == "__main__":
    main()
