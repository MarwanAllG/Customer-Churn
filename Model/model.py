from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

import joblib
import json
from datetime import datetime

import mlflow
import mlflow.sklearn

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


def load_raw_events(path: str | Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df["userId"] = df["userId"].astype(str)
    df = df[df["userId"].str.strip().ne("")]
    return df


def device(user_agent: str) -> str:
    ua = str(user_agent)
    if "iPhone" in ua or "iPad" in ua or "iOS" in ua:
        return "iOS"
    if "Macintosh" in ua or "Mac OS X" in ua:
        return "Mac"
    if "Windows" in ua:
        return "Windows"
    if "Linux" in ua:
        return "Linux"
    return "Other"


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["page", "level", "auth", "userAgent"]:
        if col not in df.columns:
            df[col] = ""
    if "event_time" not in df.columns:
        if "ts" in df.columns:
            df["event_time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        else:
            df["event_time"] = pd.to_datetime(pd.Timestamp.utcnow(), utc=True)
    if "registration" in df.columns and "registration_time" not in df.columns:
        df["registration_time"] = pd.to_datetime(df["registration"], unit="ms", utc=True)
    df["userAgent"] = df["userAgent"].fillna("").astype(str)
    df['device'] = df['userAgent'].apply(device)
    df["event_date"] = pd.to_datetime(df["event_time"], utc=True).dt.date
    df["page"] = df["page"].astype(str).str.lower().str.strip()
    df["level"] = df["level"].astype(str).str.lower()
    df["auth"] = df["auth"].astype(str).str.lower().str.strip()
    return df


def label_churn_by_event(df: pd.DataFrame) -> pd.DataFrame:
    cancel_mask = (df["page"] == "cancellation confirmation")
    if not cancel_mask.any():
        return pd.DataFrame(columns=["userId", "churn_label", "churn_time"])
    first_cancel = df.loc[cancel_mask].groupby("userId")["event_time"].min()
    lbl = pd.DataFrame({"userId": df["userId"].unique()}).set_index("userId").sort_index()
    lbl["churn_label"] = 0
    lbl["churn_time"] = pd.NaT
    lbl.loc[first_cancel.index, "churn_label"] = 1
    lbl.loc[first_cancel.index, "churn_time"] = first_cancel
    return lbl.reset_index()


ACTION_PAGES = ['nextsong', 'logout', 'home', 'downgrade', 'add to playlist', 'roll advert', 'thumbs up', 'help', 'thumbs down', 'add friend','settings','save settings','upgrade','about','submit downgrade','submit upgrade','error']


def build_user_snapshots(
    df: pd.DataFrame,
    labels: pd.DataFrame,
    obs_window_days: int = 28
) -> pd.DataFrame:
    df["is_song"] = (df["page"] == "nextsong").astype(int)
    last_event_per_user = df.groupby("userId")["event_time"].max()
    lbl = labels.set_index("userId").copy()
    lbl["cutoff_time"] = lbl["churn_time"]
    lbl.loc[lbl["churn_label"] == 0, "cutoff_time"] = lbl.index.to_series().map(last_event_per_user)
    lbl["window_start"] = lbl["cutoff_time"] - pd.to_timedelta(obs_window_days, unit="D")

    features: List[Dict] = []
    for user_id, row in lbl.dropna(subset=["cutoff_time", "window_start"]).iterrows():
        mask = (df["userId"] == user_id) & (df["event_time"] < row["cutoff_time"]) & (df["event_time"] >= row["window_start"])
        wdf = df.loc[mask]
        if wdf.empty:
            wdf = df[df["userId"] == user_id].tail(100)

        n_events = len(wdf)
        n_songs = int(wdf["is_song"].sum())
        n_days_active = wdf["event_date"].nunique()
        n_sessions = wdf["sessionId"].nunique() if "sessionId" in wdf.columns else 0
        n_artists = wdf["artist"].nunique() if "artist" in wdf.columns else 0
        n_unique_songs = wdf["song"].nunique() if "song" in wdf.columns else 0

        events_per_day = n_events / n_days_active if n_days_active else 0.0
        songs_per_session = n_songs / n_sessions if n_sessions else 0.0

        page_counts = wdf["page"].value_counts()
        action_feats = {f"page_{p.replace(' ','_').lower()}": int(page_counts.get(p, 0)) for p in ACTION_PAGES}

        total_session_min = 0.0
        avg_session_min = 0.0
        if n_sessions > 0:
            sess = wdf.groupby("sessionId")["event_time"].agg(["min", "max"])
            durations = (sess["max"] - sess["min"]).dt.total_seconds() / 60.0
            total_session_min = float(durations.sum())
            avg_session_min = float(durations.mean())

        tenure_days = 0.0
        if "registration_time" in df.columns:
            user_regs = df.loc[df["userId"] == user_id, "registration_time"].dropna()
            if not user_regs.empty:
                tenure_days = (row["cutoff_time"] - user_regs.min()).days

        last_level = wdf["level"].iloc[-1] if "level" in wdf.columns and not wdf.empty else np.nan
        last_device = wdf["device"].iloc[-1] if not wdf.empty else "Other"

        rec = {
            "userId": user_id,
            "churn_label": int(row["churn_label"]),
            "cutoff_time": row["cutoff_time"],
            "window_start": row["window_start"],
            "tenure_days": float(tenure_days),
            "n_events": int(n_events),
            "n_songs": int(n_songs),
            "n_days_active": int(n_days_active),
            "n_sessions": int(n_sessions),
            "n_artists": int(n_artists),
            "n_unique_songs": int(n_unique_songs),
            "events_per_day": float(events_per_day),
            "songs_per_session": float(songs_per_session),
            "total_session_min": float(total_session_min),
            "avg_session_min": float(avg_session_min),
            "level": last_level,
            "device": last_device,
        }
        rec.update(action_feats)
        features.append(rec)

    Xy = pd.DataFrame(features)
    numeric_cols = Xy.select_dtypes(include=[np.number]).columns
    Xy[numeric_cols] = Xy[numeric_cols].fillna(0)
    return Xy


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    cat_cols = ["level", "device"]
    drop_cols = ["userId", "cutoff_time", "window_start", "churn_label"]
    num_cols = [c for c in X.columns if c not in cat_cols + drop_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(with_mean=False), num_cols),
        ],
        remainder="drop",
    )
    return pre, cat_cols, num_cols


def evaluate_with_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float, float, float]:
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        roc = float("nan")
        pr = 0.0 if pos == 0 else 1.0
        return roc, pr, 0.0, 0.5

    roc = roc_auc_score(y_true, scores)
    pr = average_precision_score(y_true, scores)

    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
    best_idx = int(np.nanargmax(f1s))
    best_thr = thresholds[max(0, min(best_idx - 1, len(thresholds) - 1))] if len(thresholds) > 0 else 0.5
    best_f1 = float(f1s[best_idx])
    return float(roc), float(pr), float(best_f1), float(best_thr)


def split_train_val_hybrid(
    Xy: pd.DataFrame,
    val_size: float = 0.2,
    min_pos_val: int = 10,
    min_neg_val: int = 50,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    Xy_sorted = Xy.sort_values("cutoff_time").reset_index(drop=True)
    for q in np.linspace(0.80, 0.60, 5):
        t_cut = Xy_sorted["cutoff_time"].quantile(q)
        train_df = Xy_sorted[Xy_sorted["cutoff_time"] <= t_cut]
        val_df = Xy_sorted[Xy_sorted["cutoff_time"] > t_cut]
        pos = int(val_df["churn_label"].sum())
        neg = int((val_df["churn_label"] == 0).sum())
        if pos >= min_pos_val and neg >= min_neg_val:
            print(f"[Split] Time-based q={q:.2f} -> val size={len(val_df)}, pos={pos}, neg={neg}")
            return train_df, val_df

    train_df, val_df = train_test_split(
        Xy_sorted, test_size=val_size, stratify=Xy_sorted["churn_label"], random_state=random_state
    )
    pos = int(val_df["churn_label"].sum())
    neg = int((val_df["churn_label"] == 0).sum())
    print(f"[Split] Fallback stratified -> val size={len(val_df)}, pos={pos}, neg={neg}")
    return train_df, val_df


def train_and_compare_models(Xy: pd.DataFrame, mlflow_experiment: Optional[str] = None) -> Tuple[Pipeline, float, Dict[str, float]]:
    train_df, val_df = split_train_val_hybrid(Xy, val_size=0.2, min_pos_val=10, min_neg_val=20)

    print("Train class balance:", train_df["churn_label"].value_counts().to_dict())
    print("Val class balance  :", val_df["churn_label"].value_counts().to_dict())

    y_train = train_df["churn_label"].values
    y_val = val_df["churn_label"].values

    X_train = train_df.drop(columns=["churn_label"])
    X_val = val_df.drop(columns=["churn_label"])

    pre, _, _ = build_preprocessor(X_train)

    models = [
        ("LogisticRegression", LogisticRegression(max_iter=400, class_weight="balanced")),
        ("RandomForest", RandomForestClassifier(n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=42)),
        ("HistGradientBoosting", HistGradientBoostingClassifier(max_depth=None, learning_rate=0.08, max_iter=300, random_state=42)),
        ("GradientBoostingClassifier", GradientBoostingClassifier(n_estimators=400, learning_rate=0.08, random_state=42)),
    ]

    best_by_pr_auc: Optional[float] = None
    best_name: Optional[str] = None
    best_pipeline: Optional[Pipeline] = None
    best_scores: Optional[np.ndarray] = None
    best_thr = 0.5
    best_f1 = 0.0
    best_roc = 0.0
    best_pr = 0.0
    best_acc = 0.0

    for name, base_model in models:
        pipe = Pipeline(steps=[("pre", pre), ("model", base_model)])
        pipe.fit(X_train, y_train)

        if hasattr(pipe.named_steps["model"], "predict_proba"):
            scores = pipe.predict_proba(X_val)[:, 1]
        else:
            scores = pipe.decision_function(X_val)

        roc, pr, f1, thr = evaluate_with_threshold(y_val, scores)
        y_pred_model = (scores >= thr).astype(int)
        acc = accuracy_score(y_val, y_pred_model)

        print(f"\n[{name}]")
        print(f" ROC-AUC : {roc:.3f}")
        print(f" PR-AUC  : {pr:.3f}")
        print(f" Best F1 : {f1:.3f} at threshold ~ {thr:.2f}")
        print(f" Accuracy: {acc*100:.2f}%")

        if mlflow is not None and mlflow_experiment is not None:
            mlflow.set_experiment(mlflow_experiment)
            with mlflow.start_run(run_name=f"candidate_{name}", nested=True):
                mlflow.log_params({"model": name})
                mlflow.log_metrics({
                    "val_roc_auc": float(roc),
                    "val_pr_auc": float(pr),
                    "val_best_f1": float(f1),
                    "val_best_thr": float(thr),
                    "val_accuracy": float(acc),
                })

        if (best_by_pr_auc is None) or (pr > best_by_pr_auc):
            best_by_pr_auc = pr
            best_name = name
            best_pipeline = pipe
            best_scores = scores
            best_thr = thr
            best_f1 = f1
            best_roc = roc
            best_pr = pr
            best_acc = acc

    metrics: Dict[str, float] = {}
    if best_scores is not None:
        y_pred = (best_scores >= best_thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        print(f"\n=== Best model: {best_name} ===")
        print(f"ROC-AUC : {best_roc:.3f} | PR-AUC: {best_pr:.3f} | F1*: {best_f1:.3f} @ thr ~ {best_thr:.2f}")
        print(f"Accuracy: {best_acc*100:.2f}%")
        print(f"Confusion Matrix @ thr {best_thr:.2f}")
        print(f"TN: {tn}  FP: {fp}  FN: {fn}  TP: {tp}")

        metrics = {
            "val_roc_auc": float(best_roc),
            "val_pr_auc": float(best_pr),
            "val_best_f1": float(best_f1),
            "val_best_thr": float(best_thr),
            "val_accuracy": float(best_acc),
            "val_tn": float(tn),
            "val_fp": float(fp),
            "val_fn": float(fn),
            "val_tp": float(tp),
        }

    if best_pipeline is None:
        raise RuntimeError("Training failed to produce a best model")

    return best_pipeline, best_thr, metrics


def compute_reference_stats(Xy: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {"numeric": {}, "categorical": {}}
    cat_cols = ["level", "device"]
    drop_cols = ["userId", "cutoff_time", "window_start", "churn_label"]
    num_cols = [c for c in Xy.columns if c not in cat_cols + drop_cols]

    for c in num_cols:
        if pd.api.types.is_numeric_dtype(Xy[c]):
            stats["numeric"][c] = {
                "mean": float(np.mean(Xy[c])),
                "std": float(np.std(Xy[c]) + 1e-8),
            }
    for c in cat_cols:
        vc = (Xy[c].astype(str).value_counts(normalize=True)).to_dict()
        stats["categorical"][c] = {str(k): float(v) for k, v in vc.items()}
    return stats


def save_artifacts(pipeline: Pipeline, threshold: float, Xy: pd.DataFrame, metrics: Optional[Dict[str, float]] = None, out_dir: str | Path = "artifacts") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = out / f"best_churn_pipeline_{timestamp}.joblib"
    joblib.dump(pipeline, model_path)

    meta = {
        "saved_at": timestamp,
        "threshold": float(threshold),
        "feature_columns": [c for c in Xy.columns if c not in ["churn_label"]],
        "metrics": metrics or {},
    }
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    stats = compute_reference_stats(Xy)
    with open(out / "reference_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    joblib.dump(pipeline, out / "best_churn_pipeline.joblib")
    return model_path


def train_entry(data_path: str = "customer_churn.json", obs_window_days: int = 30, use_mlflow: bool = True) -> None:
    df = load_raw_events(data_path)
    df = prepare_dataframe(df)

    labels = label_churn_by_event(df)
    print("\nLabel distribution (0=stay, 1=churn):")
    print(labels["churn_label"].value_counts(dropna=False))

    Xy = build_user_snapshots(df=df, labels=labels, obs_window_days=obs_window_days)
    print("\nFeature table shape:", Xy.shape)
    print("Columns:", list(Xy.columns))

    experiment = "churn_training"
    if use_mlflow and mlflow is not None:
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name="train_compare_best"):
            best_pipeline, thr, metrics = train_and_compare_models(Xy, mlflow_experiment=experiment)
            mlflow.log_params({"obs_window_days": obs_window_days})
            mlflow.log_metrics(metrics)
            mlflow.log_param("selected_threshold", float(thr))
            mlflow.sklearn.log_model(best_pipeline, artifact_path="model")
    else:
        best_pipeline, thr, metrics = train_and_compare_models(Xy, mlflow_experiment=None)

    save_artifacts(best_pipeline, thr, Xy, metrics)
    print("Artifacts saved in ./artifacts")


def build_features_for_user_events(events: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(events)
    if "userId" not in df.columns:
        df["userId"] = "0"
    df = prepare_dataframe(df)
    labels = pd.DataFrame({"userId": df["userId"].unique(), "churn_label": 0, "churn_time": pd.NaT})
    Xy = build_user_snapshots(df=df, labels=labels, obs_window_days=30)
    X = Xy.drop(columns=["churn_label"], errors="ignore")
    return X


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "customer_churn.json"
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    train_entry(path, window, use_mlflow=True)


