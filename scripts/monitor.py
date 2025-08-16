from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from Model.model import (
    load_raw_events,
    prepare_dataframe,
    build_user_snapshots,
    compute_reference_stats,
    label_churn_by_event,
)
import joblib
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (np.sum(p) + 1e-12)
    q = q / (np.sum(q) + 1e-12)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def population_stability_index(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref_hist, edges = np.histogram(ref, bins=bins)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_pct = ref_hist / (np.sum(ref_hist) + 1e-12)
    cur_pct = cur_hist / (np.sum(cur_hist) + 1e-12)
    psi = np.sum((cur_pct - ref_pct) * np.log((cur_pct + 1e-12) / (ref_pct + 1e-12)))
    return float(psi)


def detect_drift(reference_stats: Dict, current_Xy: pd.DataFrame, psi_threshold: float = 0.2, cat_hellinger_threshold: float = 0.2) -> Dict:
    alerts = []

    # numeric drift via PSI
    for col, st in reference_stats.get("numeric", {}).items():
        if col in current_Xy.columns and pd.api.types.is_numeric_dtype(current_Xy[col]):
            ref = np.random.normal(st["mean"], st["std"], size=min(10000, len(current_Xy)))
            cur = current_Xy[col].values.astype(float)
            val = population_stability_index(ref, cur, bins=10)
            if val > psi_threshold:
                alerts.append({"type": "data_drift", "feature": col, "metric": "psi", "value": float(val)})

    # categorical drift via Hellinger distance on freq distrib
    for col, ref_freq in reference_stats.get("categorical", {}).items():
        if col in current_Xy.columns:
            cur_freq = current_Xy[col].astype(str).value_counts(normalize=True).to_dict()
            cats = sorted(set(ref_freq.keys()) | set(cur_freq.keys()))
            ref_vec = np.array([ref_freq.get(c, 0.0) for c in cats], dtype=float)
            cur_vec = np.array([cur_freq.get(c, 0.0) for c in cats], dtype=float)
            val = hellinger(ref_vec, cur_vec)
            if val > cat_hellinger_threshold:
                alerts.append({"type": "data_drift", "feature": col, "metric": "hellinger", "value": float(val)})

    return {"alerts": alerts}


def main():
    parser = argparse.ArgumentParser(description="Simple drift monitoring")
    parser.add_argument("--ref", default="artifacts/reference_stats.json", help="Path to reference stats JSON")
    parser.add_argument("--data", default="customer_churn.json", help="Path to new batch events")
    parser.add_argument("--window", type=int, default=30, help="Observation window in days")
    parser.add_argument("--artifacts", default="artifacts", help="Artifacts dir to load model & metadata for concept drift check")
    parser.add_argument("--perf-drop", type=float, default=0.2, help="Alert if PR-AUC drops more than this fraction from reference (e.g., 0.2 = 20%)")
    args = parser.parse_args()

    with open(args.ref, "r", encoding="utf-8") as f:
        reference_stats = json.load(f)

    df = load_raw_events(args.data)
    df = prepare_dataframe(df)
    # Build proxy labels for concept drift check 
    proxy_labels = label_churn_by_event(df)
    Xy = build_user_snapshots(df=df, labels=proxy_labels, obs_window_days=args.window)

    report = detect_drift(reference_stats, Xy)
    # Concept drift / performance drift 
    try:
        art_dir = Path(args.artifacts)
        model = joblib.load(art_dir / "best_churn_pipeline.joblib")
        with open(art_dir / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        thr = float(meta.get("threshold", 0.5))
        ref_pr = float(meta.get("metrics", {}).get("val_pr_auc", 0.0))
        # Evaluate on current batch (where labels available)
        y_true = Xy["churn_label"].values.astype(int)
        X = Xy.drop(columns=["churn_label"]) 
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        else:
            scores = model.decision_function(X)
            probs = 1.0 / (1.0 + np.exp(-scores))
        y_pred = (probs >= thr).astype(int)
        cur_pr = float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) == 2 else 0.0
        cur_roc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) == 2 else float("nan")
        cur_acc = float(accuracy_score(y_true, y_pred))
        report["performance"] = {"pr_auc": cur_pr, "roc_auc": cur_roc, "accuracy": cur_acc}
        if ref_pr > 0 and (ref_pr - cur_pr) / max(ref_pr, 1e-8) > args.perf_drop:
            report.setdefault("alerts", []).append({
                "type": "concept_drift",
                "metric": "pr_auc_drop_frac",
                "value": float((ref_pr - cur_pr) / max(ref_pr, 1e-8)),
            })
    except Exception:
        # Artifacts or labels not available; skip concept drift check
        pass

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


