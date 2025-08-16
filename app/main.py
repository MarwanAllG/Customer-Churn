from __future__ import annotations

from typing import List, Dict, Optional
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

from Model.model import build_features_for_user_events


ARTIFACTS_DIR = Path("artifacts")
MODEL_FILE = ARTIFACTS_DIR / "best_churn_pipeline.joblib"
META_FILE = ARTIFACTS_DIR / "metadata.json"


class PredictRequest(BaseModel):
    events: List[Dict]
    threshold: Optional[float] = None


class PredictResponseItem(BaseModel):
    probability: float
    label: int


app = FastAPI(title="Churn Model API", version="1.0.0")


def _load_artifacts():
    if not MODEL_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError("Artifacts not found. Please run training first.")
    pipeline = joblib.load(MODEL_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    thr = float(meta.get("threshold", 0.5))
    return pipeline, thr


@app.get("/health")
def health():
    try:
        _load_artifacts()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=List[PredictResponseItem])
def predict(req: PredictRequest):
    try:
        pipeline, default_thr = _load_artifacts()
        thr = float(req.threshold) if req.threshold is not None else default_thr

        safe_events = []
        for ev in req.events:
            safe = {
                "userId": str(ev.get("userId", "0")),
                "ts": ev.get("ts", ev.get("event_time")),
                "page": ev.get("page", ""),
                "level": ev.get("level", ""),
                "auth": ev.get("auth", ""),
                "userAgent": ev.get("userAgent", ""),
                "registration": ev.get("registration", None),
                "sessionId": ev.get("sessionId", None),
                "artist": ev.get("artist", None),
                "song": ev.get("song", None),
            }
            safe_events.append(safe)

        X = build_features_for_user_events(safe_events)
        model = pipeline.named_steps.get("model", pipeline)

        if hasattr(model, "predict_proba"):
            probs = pipeline.predict_proba(X)[:, 1]
        else:
            scores = pipeline.decision_function(X)
            probs = 1.0 / (1.0 + np.exp(-scores))

        labels = (probs >= thr).astype(int)
        return [
            PredictResponseItem(probability=float(p), label=int(l))
            for p, l in zip(probs, labels)
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def root():
    return {"message": "Use POST /predict with {'events': [...]}"}


