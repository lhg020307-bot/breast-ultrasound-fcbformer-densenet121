"""Evaluate 3-view fusion (full, cut_borders, masked) — hard voting + weighted.

Key improvements over previous version:
  1. Weight grid-search on OOF to find the best weight combination.
  2. Optimal threshold selected on OOF (not eval-set).
  3. BUSI evaluated with OOF-optimized weights and thresholds.

Reads per-view OOF CSVs and BUSI predictions.csv from existing normalized outputs.

Outputs to outputs/results/competition/:
  metrics_3view.json
  roc_curve_3view.png
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent
EPS = 1e-7
THREE_VIEWS = ["full", "cut_borders", "masked"]


# ── metrics ──────────────────────────────────────────────────────────
def classification_metrics(y_true, y_prob, threshold=0.5):
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
        except (ValueError, TypeError):
            pass

    return {
        "AUC": auc, "Accuracy": accuracy,
        "Sensitivity": sensitivity, "Specificity": specificity,
        "Precision": precision, "F1-score": f1,
        "threshold": threshold,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }


def roc_curve_points(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    thresholds = np.r_[np.inf, np.sort(np.unique(y_score))[::-1], -np.inf]
    pos = max(int((y_true == 1).sum()), 1)
    neg = max(int((y_true == 0).sum()), 1)
    pts = []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        pts.append({"fpr": fp / neg, "tpr": tp / pos,
                     "threshold": float(t) if np.isfinite(t) else str(t)})
    return pts


# ── fusion strategies ────────────────────────────────────────────────
def hard_voting(prob_dict, threshold=0.5):
    votes = [1 if prob_dict.get(v, 0.5) >= threshold else 0 for v in THREE_VIEWS]
    n_mal = sum(votes)
    n_ben = len(votes) - n_mal
    if n_mal > n_ben:
        return 1.0
    if n_ben > n_mal:
        return 0.0
    return float(prob_dict.get("full", 0.5))


def hard_voting_prob(probs_df, threshold=0.5):
    vals = []
    for _, row in probs_df.iterrows():
        pdict = {v: float(row[f"prob_{v}"]) for v in THREE_VIEWS}
        vals.append(hard_voting(pdict, threshold))
    return np.array(vals, dtype=float)


def weighted_fusion_probs(probs_df, weights):
    w = {v: weights.get(v, 0.0) for v in THREE_VIEWS}
    total = sum(w.values())
    if total <= 0:
        w = {v: 1.0 for v in THREE_VIEWS}
        total = 3.0
    result = np.zeros(len(probs_df), dtype=float)
    for v in THREE_VIEWS:
        result += (w[v] / total) * probs_df[f"prob_{v}"].astype(float).values
    return result


def apply_platt(probs, a, b):
    clipped = np.clip(np.asarray(probs, dtype=np.float64), 1e-9, 1 - 1e-9)
    logits = np.log(clipped / (1 - clipped))
    return 1.0 / (1.0 + np.exp(-(a * logits + b)))


# ── load data ────────────────────────────────────────────────────────
def load_busi_predictions(path):
    df = pd.read_csv(path)
    y_true = df["true_label_id"].astype(int).values
    return df, y_true


def load_oof_data(metrics_dir, backbone="densenet121"):
    merged = None
    for view in THREE_VIEWS:
        p = metrics_dir / f"oof_{view}_{backbone}.csv"
        if not p.exists():
            print(f"  WARNING: {p} missing, skipping {view}")
            continue
        df = pd.read_csv(p)
        group_cols = ["sample_id", "dataset"]
        df = df.groupby(group_cols, as_index=False).agg({
            f"prob_{view}": "mean",
            "y_true": "first",
        })
        keep = ["sample_id", "dataset", "y_true", f"prob_{view}"]
        df = df[[c for c in keep if c in df.columns]]
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["sample_id", "dataset"], how="inner")
    return merged


# ── threshold search ─────────────────────────────────────────────────
def find_best_threshold(y_true, probs, objective="f1", steps=701,
                        min_sensitivity=0.80, min_specificity=0.75):
    """Search threshold on [0.1, 0.8] with constraints. Returns (threshold, metrics)."""
    best_t, best_val = 0.5, -1.0
    best_metrics = None
    for t in np.linspace(0.1, 0.8, steps):
        m = classification_metrics(y_true, probs, float(t))
        if m["Sensitivity"] < min_sensitivity or m["Specificity"] < min_specificity:
            continue
        if objective == "f1":
            val = m["F1-score"]
        elif objective == "youden":
            val = m["Sensitivity"] + m["Specificity"] - 1.0
        else:
            val = m["F1-score"]
        if val > best_val:
            best_val, best_t = val, float(t)
            best_metrics = m
    if best_metrics is None:
        # no threshold satisfied constraints, pick best F1 without constraints
        return find_best_threshold(y_true, probs, "f1", steps,
                                   min_sensitivity=0.0, min_specificity=0.0)
    return best_t, best_metrics


# ── weight grid search on OOF ────────────────────────────────────────
def search_best_weights(oof_df, y_oof, objective="f1", step=0.05,
                        min_sensitivity=0.80, min_specificity=0.75):
    """Grid search weights (sum to 1) on OOF data. Returns (best_weights, best_metrics)."""
    best_weights = None
    best_val = -1.0
    best_metrics = None

    candidates = []
    n_steps = int(1.0 / step)
    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            k = n_steps - i - j
            w_full = round(i * step, 2)
            w_cut = round(j * step, 2)
            w_masked = round(k * step, 2)
            candidates.append({"full": w_full, "cut_borders": w_cut, "masked": w_masked})

    # deduplicate
    seen = set()
    unique_candidates = []
    for w in candidates:
        key = (w["full"], w["cut_borders"], w["masked"])
        if key not in seen:
            seen.add(key)
            unique_candidates.append(w)

    print(f"  Grid search: {len(unique_candidates)} weight combinations (step={step})")

    for weights in unique_candidates:
        probs = weighted_fusion_probs(oof_df, weights)
        t, m = find_best_threshold(y_oof, probs, objective, steps=301,
                                   min_sensitivity=min_sensitivity,
                                   min_specificity=min_specificity)
        if objective == "f1":
            val = m["F1-score"]
        elif objective == "youden":
            val = m["Sensitivity"] + m["Specificity"] - 1.0
        else:
            val = m["F1-score"]
        if val > best_val:
            best_val = val
            best_weights = dict(weights)
            best_metrics = m

    return best_weights, best_metrics


# ── main ─────────────────────────────────────────────────────────────
def main():
    output_dir = PROJECT_ROOT / "outputs" / "results" / "competition"
    metrics_dir = PROJECT_ROOT / "outputs" / "results" / "oof"
    pred_path = output_dir / "predictions.csv"

    # constraints
    OBJECTIVE = "f1"
    MIN_SE = 0.80
    MIN_SP = 0.75

    print("=" * 64)
    print("3-View Fusion — OOF-optimized weights & thresholds")
    print(f"  Objective: {OBJECTIVE}  |  min Se={MIN_SE}  min Sp={MIN_SP}")
    print("=" * 64)

    # ── 1. Load data ─────────────────────────────────────────────────
    busi_df, y_busi = load_busi_predictions(pred_path)
    print(f"\nBUSI eval samples : {len(busi_df)}")

    oof_df = load_oof_data(metrics_dir)
    if oof_df is None or len(oof_df) == 0:
        raise RuntimeError("No OOF data found — cannot optimize weights.")
    y_oof = oof_df["y_true"].astype(int).values
    print(f"OOF   samples      : {len(oof_df)}")
    unique, cnt = np.unique(y_oof, return_counts=True)
    print(f"OOF   class dist   : benign={cnt[0]}  malignant={cnt[1]}")

    # ── 2. Single-view (reference) ───────────────────────────────────
    print(f"\n{'─'*60}")
    print("Single-view performance (threshold=0.5, BUSI reference)")
    print("─" * 60)
    single_metrics = {}
    single_oof_metrics = {}
    for view in THREE_VIEWS:
        sm = classification_metrics(y_busi, busi_df[f"prob_{view}"].values, 0.5)
        single_metrics[view] = sm
        a = f"{sm['AUC']:.4f}" if sm['AUC'] is not None else "N/A"
        print(f"  {view:15s} AUC={a}  F1={sm['F1-score']:.4f}  "
              f"ACC={sm['Accuracy']:.4f}  Se={sm['Sensitivity']:.4f}  Sp={sm['Specificity']:.4f}")

    # ── 3. Hard voting — OOF threshold search ────────────────────────
    print(f"\n{'─'*60}")
    print("Hard voting 3v — search threshold on OOF")
    print("─" * 60)

    hv_oof = hard_voting_prob(oof_df, threshold=0.5)
    hv_best_t, hv_oof_best = find_best_threshold(y_oof, hv_oof, OBJECTIVE,
                                                  min_sensitivity=MIN_SE,
                                                  min_specificity=MIN_SP)
    print(f"  OOF best threshold: {hv_best_t:.4f}")
    a_oof = f"{hv_oof_best['AUC']:.4f}" if hv_oof_best['AUC'] is not None else "N/A"
    print(f"  OOF metrics  : AUC={a_oof}  F1={hv_oof_best['F1-score']:.4f}  "
          f"ACC={hv_oof_best['Accuracy']:.4f}  "
          f"Se={hv_oof_best['Sensitivity']:.4f}  Sp={hv_oof_best['Specificity']:.4f}")

    # apply to BUSI
    hv_busi = hard_voting_prob(busi_df, threshold=hv_best_t)
    hv_busi_metrics = classification_metrics(y_busi, hv_busi, hv_best_t)
    hv_busi_metrics["method"] = "hard_voting_3view"
    hv_busi_metrics["oof_best_threshold"] = hv_best_t
    a_hv = f"{hv_busi_metrics['AUC']:.4f}" if hv_busi_metrics['AUC'] is not None else "N/A"
    print(f"  BUSI metrics : AUC={a_hv}  F1={hv_busi_metrics['F1-score']:.4f}  "
          f"ACC={hv_busi_metrics['Accuracy']:.4f}  "
          f"Se={hv_busi_metrics['Sensitivity']:.4f}  Sp={hv_busi_metrics['Specificity']:.4f}")

    # ── 4. Weighted fusion — OOF weight + threshold search ───────────
    print(f"\n{'─'*60}")
    print("Weighted fusion 3v — grid-search weights + threshold on OOF")
    print("─" * 60)

    best_w, best_w_oof = search_best_weights(
        oof_df, y_oof, objective=OBJECTIVE, step=0.05,
        min_sensitivity=MIN_SE, min_specificity=MIN_SP)
    wf_oof_raw = weighted_fusion_probs(oof_df, best_w)
    wf_best_t, wf_oof_best = find_best_threshold(y_oof, wf_oof_raw, OBJECTIVE,
                                                  min_sensitivity=MIN_SE,
                                                  min_specificity=MIN_SP)

    print(f"  Best weights (OOF): {best_w}")
    print(f"  OOF best threshold: {wf_best_t:.4f}")
    a = f"{wf_oof_best['AUC']:.4f}" if wf_oof_best['AUC'] is not None else "N/A"
    print(f"  OOF metrics  : AUC={a}  F1={wf_oof_best['F1-score']:.4f}  "
          f"ACC={wf_oof_best['Accuracy']:.4f}  "
          f"Se={wf_oof_best['Sensitivity']:.4f}  Sp={wf_oof_best['Specificity']:.4f}")

    # apply to BUSI (raw)
    wf_busi_raw = weighted_fusion_probs(busi_df, best_w)
    wf_raw_metrics = classification_metrics(y_busi, wf_busi_raw, wf_best_t)
    wf_raw_metrics["method"] = "weighted_fusion_3view_raw"
    wf_raw_metrics["weights"] = best_w
    wf_raw_metrics["oof_best_threshold"] = wf_best_t
    a = f"{wf_raw_metrics['AUC']:.4f}" if wf_raw_metrics['AUC'] is not None else "N/A"
    print(f"  BUSI (raw)   : AUC={a}  F1={wf_raw_metrics['F1-score']:.4f}  "
          f"ACC={wf_raw_metrics['Accuracy']:.4f}  "
          f"Se={wf_raw_metrics['Sensitivity']:.4f}  Sp={wf_raw_metrics['Specificity']:.4f}")

    # ── 5. Platt calibration on OOF ──────────────────────────────────
    print(f"\n{'─'*60}")
    print("Weighted fusion + Platt calibration (OOF fitted, OOF threshold)")
    print("─" * 60)

    oof_clipped = np.clip(wf_oof_raw, 1e-9, 1 - 1e-9)
    oof_logits = np.log(oof_clipped / (1 - oof_clipped)).reshape(-1, 1)
    cal = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
    cal.fit(oof_logits, y_oof)
    platt_a = float(cal.coef_[0, 0])
    platt_b = float(cal.intercept_[0])

    oof_cal = apply_platt(wf_oof_raw, platt_a, platt_b)
    wf_cal_best_t, wf_cal_oof_best = find_best_threshold(
        y_oof, oof_cal, OBJECTIVE,
        min_sensitivity=MIN_SE, min_specificity=MIN_SP)

    print(f"  Platt: a={platt_a:.4f}  b={platt_b:.4f}")
    print(f"  OOF best threshold (calibrated): {wf_cal_best_t:.4f}")
    a = f"{wf_cal_oof_best['AUC']:.4f}" if wf_cal_oof_best['AUC'] is not None else "N/A"
    print(f"  OOF metrics (cal) : AUC={a}  F1={wf_cal_oof_best['F1-score']:.4f}  "
          f"ACC={wf_cal_oof_best['Accuracy']:.4f}  "
          f"Se={wf_cal_oof_best['Sensitivity']:.4f}  Sp={wf_cal_oof_best['Specificity']:.4f}")

    # apply to BUSI
    wf_busi_cal = apply_platt(wf_busi_raw, platt_a, platt_b)
    wf_cal_metrics = classification_metrics(y_busi, wf_busi_cal, wf_cal_best_t)
    wf_cal_metrics["method"] = "weighted_fusion_3view_platt"
    wf_cal_metrics["weights"] = best_w
    wf_cal_metrics["platt"] = {"a": platt_a, "b": platt_b}
    wf_cal_metrics["oof_best_threshold"] = wf_cal_best_t
    a = f"{wf_cal_metrics['AUC']:.4f}" if wf_cal_metrics['AUC'] is not None else "N/A"
    print(f"  BUSI (cal)    : AUC={a}  F1={wf_cal_metrics['F1-score']:.4f}  "
          f"ACC={wf_cal_metrics['Accuracy']:.4f}  "
          f"Se={wf_cal_metrics['Sensitivity']:.4f}  Sp={wf_cal_metrics['Specificity']:.4f}")

    # ── 6. Save JSON ─────────────────────────────────────────────────
    result = {
        "eval_dataset": "BUSI",
        "calibration_dataset": "OOF (BUS+BUSBRA)",
        "n_oof": int(len(y_oof)),
        "n_busi": int(len(y_busi)),
        "views_used": THREE_VIEWS,
        "objective": OBJECTIVE,
        "min_sensitivity_constraint": MIN_SE,
        "min_specificity_constraint": MIN_SP,
        "single_view_metrics_busi": single_metrics,
        "hard_voting_3view": {
            "oof_best_threshold": hv_best_t,
            "oof_metrics_at_best": hv_oof_best,
            "busi_metrics": hv_busi_metrics,
        },
        "weighted_fusion_3view_raw": {
            "best_weights_from_oof": best_w,
            "oof_best_threshold": wf_best_t,
            "oof_metrics_at_best": wf_oof_best,
            "busi_metrics": wf_raw_metrics,
        },
        "weighted_fusion_3view_platt": {
            "best_weights_from_oof": best_w,
            "platt": {"a": platt_a, "b": platt_b},
            "oof_best_threshold": wf_cal_best_t,
            "oof_metrics_at_best": wf_cal_oof_best,
            "busi_metrics": wf_cal_metrics,
        },
    }

    out_json = output_dir / "metrics_3view.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_json}")

    # ── 7. ROC plot ──────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 7.5))

        colors = {"full": "#1f77b4", "cut_borders": "#ff7f0e", "masked": "#2ca02c"}
        for view in THREE_VIEWS:
            pts = roc_curve_points(y_busi, busi_df[f"prob_{view}"].values)
            a_val = single_metrics[view]["AUC"]
            label = f"{view} (AUC={a_val:.4f})" if a_val is not None else view
            plt.plot([p["fpr"] for p in pts], [p["tpr"] for p in pts],
                     linewidth=1.2, color=colors[view], label=label)

        # hard voting
        hv_pts = roc_curve_points(y_busi, hv_busi)
        hv_auc = hv_busi_metrics["AUC"]
        plt.plot([p["fpr"] for p in hv_pts], [p["tpr"] for p in hv_pts],
                 linewidth=1.8, color="red", linestyle="--",
                 label=f"Hard Voting (AUC={hv_auc:.4f})" if hv_auc else "Hard Voting")

        # weighted raw
        wf_pts = roc_curve_points(y_busi, wf_busi_raw)
        wf_auc = wf_raw_metrics["AUC"]
        plt.plot([p["fpr"] for p in wf_pts], [p["tpr"] for p in wf_pts],
                 linewidth=1.8, color="blue", linestyle="-.",
                 label=f"Weighted 3v (AUC={wf_auc:.4f})" if wf_auc else "Weighted 3v")

        # weighted + platt
        wf_cal_pts = roc_curve_points(y_busi, wf_busi_cal)
        wf_cal_auc = wf_cal_metrics["AUC"]
        plt.plot([p["fpr"] for p in wf_cal_pts], [p["tpr"] for p in wf_cal_pts],
                 linewidth=2.2, color="black",
                 label=f"Weighted+Platt (AUC={wf_cal_auc:.4f})" if wf_cal_auc else "Weighted+Platt")

        plt.plot([0, 1], [0, 1], linewidth=1, color="#777777", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"3-View Fusion ROC — BUSI\n"
                  f"(OOF-optimized weights={best_w}, thr. via OOF)")
        plt.legend(loc="lower right", fontsize=7.5)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        out_png = output_dir / "roc_curve_3view.png"
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"Saved: {out_png}")
    except Exception as exc:
        print(f"matplotlib unavailable, skipped ROC plot: {exc}")

    # ── 8. Summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  FINAL SUMMARY — All thresholds & weights optimized on OOF")
    print(f"{'=' * 72}")
    print(f"{'Method':<28s} {'Threshold':>9s} {'AUC':>8s} {'F1':>8s} {'ACC':>8s} {'Se':>8s} {'Sp':>8s}")
    print("-" * 72)
    for view in THREE_VIEWS:
        m = single_metrics[view]
        a = f"{m['AUC']:.4f}" if m['AUC'] is not None else "N/A"
        print(f"  {view:<26s} {'0.5000':>9s} {a:>8s} {m['F1-score']:>8.4f} "
              f"{m['Accuracy']:>8.4f} {m['Sensitivity']:>8.4f} {m['Specificity']:>8.4f}")
    print("-" * 72)
    for name, thresh, m in [
        ("Hard voting 3v", hv_best_t, hv_busi_metrics),
        ("Weighted 3v raw", wf_best_t, wf_raw_metrics),
        ("Weighted 3v + Platt", wf_cal_best_t, wf_cal_metrics),
    ]:
        a = f"{m['AUC']:.4f}" if m['AUC'] is not None else "N/A"
        print(f"  {name:<26s} {thresh:>9.4f} {a:>8s} {m['F1-score']:>8.4f} "
              f"{m['Accuracy']:>8.4f} {m['Sensitivity']:>8.4f} {m['Specificity']:>8.4f}")
    print(f"{'=' * 72}")

    # Compare with original 4-view
    orig_path = output_dir / "metrics.json"
    if orig_path.exists():
        orig = json.loads(orig_path.read_text(encoding="utf-8"))
        fm = orig.get("weighted_fusion_metrics", {})
        if fm:
            a = f"{fm['AUC']:.4f}" if fm.get('AUC') is not None else "N/A"
            print(f"\n  [Ref] Original 4-view calibrated fusion: "
                  f"AUC={a}  F1={fm['F1-score']:.4f}  ACC={fm['Accuracy']:.4f}  "
                  f"(threshold={fm.get('threshold','?')})")
        fm_full = orig.get("single_view_metrics", {}).get("full", {})
        if fm_full:
            a = f"{fm_full['AUC']:.4f}" if fm_full.get('AUC') is not None else "N/A"
            print(f"  [Ref] Original single full view:          "
                  f"AUC={a}  F1={fm_full['F1-score']:.4f}  ACC={fm_full['Accuracy']:.4f}")

    print(f"\n  Optimized weights: {best_w}")
    print(f"  Platt: a={platt_a:.4f}  b={platt_b:.4f}")


if __name__ == "__main__":
    main()
