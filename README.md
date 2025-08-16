# Churn Prediction 
---

## Objective
Identify users most likely to churn (cancel subscription) based on their recent in-app behavior.

---

## Workflow Overview
1) **Data prep & EDA** — load, clean, explore.  
2) **Modeling** — leakage-safe per-user snapshots; train multiple models; select the best by **PR-AUC/F1**.  
3) **Packaging & infra** — FastAPI endpoint, scheduled retrain script, MLflow tracking, drift checks; Docker & CI scaffolds.

---

## Quick Start

### Local — Windows (PowerShell)

```powershell
pip install -r requirements.txt

# Train (produces ./artifacts with model + metadata)
py -3.11 -m scripts.retrain --data customer_churn.json --window 30 

# Run API
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Quick checks:

```powershell
# Health
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET

# Predict (sample)
$body = @{ events = @(@{ ts = 1541106106796; page = "NextSong"; level = "free"; auth = "Logged In"; userAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)";}) } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -ContentType "application/json" -Body $body
```





---

## Data

**Format:** JSON Lines (each line = one user event).

**Columns (18):**

| Column         | Description                                           |
|----------------|-------------------------------------------------------|
| `ts`           | Event timestamp (epoch ms)                            |
| `userId`       | User identifier                                       |
| `sessionId`    | Session identifier                                    |
| `page`         | Event type (e.g., `NextSong`, `Logout`, …)            |
| `auth`         | Auth status                                           |
| `method`       | HTTP method                                           |
| `status`       | HTTP status code                                      |
| `level`        | Subscription tier (`free`/`paid`)                     |
| `itemInSession`| Order of event within session                         |
| `location`     | City, State                                           |
| `userAgent`    | Browser/OS string (used to derive coarse `device`)    |
| `firstName`    | First name                                            |
| `lastName`     | Last name                                             |
| `registration` | Registration timestamp (epoch ms)                     |
| `gender`       | User gender                                           |
| `artist`       | Artist name                                           |
| `song`         | Song name                                             |
| `length`       | Song length (sec)                                     |

**Cleaning highlights**
- Drop rows with **empty `userId`** (~15.7k rows).
- Convert `ts` & `registration` from ms → **UTC datetimes**; add `event_time` and `event_date`.
- Normalize text (`page`, `level`), derive **`device`** from `userAgent` (iOS/Android/Mac/Windows/Linux/Other).


**After cleaning (this dataset)**
- Rows: **528,005**  
- Users: **448**  
- Date range: **2018-10-01 → 2018-12-01 (UTC)**  

---

## Labeling & Leakage Control

- **Event-based label:** user is **positive (churner)** if any `page == "Cancellation Confirmation"`. First such time = `churn_time`. Others are negatives.
- **Cutoff per user:** last moment we’re allowed to “see”.
  - Churners → `cutoff_time = churn_time`
  - Non-churners → `cutoff_time = last_event_time`
- **Observation window:** `[cutoff_time − 30d, cutoff_time)` (exclusive of cutoff).  
  All features computed **inside** the window → prevents temporal **leakage**.

---

## Features (per user snapshot)

- **Volume & frequency:** `n_events`, `n_songs`, `n_sessions`, `n_days_active`, `events_per_day`
- **Session time:** `total_session_min`, `avg_session_min`
- **Diversity:** `n_unique_songs`, `n_artists`
- **Tenure:** `tenure_days = cutoff − registration`
- **Last known:** `device`, `level`, `state`
- **Behavioral counts:** `page_nextsong`, `page_logout`, `page_downgrade`, `page_submit_downgrade`, `page_error`, `page_roll_advert`, `page_thumbs_up`, …

**Encoding & scaling:** One-Hot for categoricals; StandardScaler for numerics.

**Train/Val split:** hybrid **time-aware ~80/20** by `cutoff_time` (ensure both classes in Val), fallback to stratified. One row per user; no user overlap.

---

## Results (Validation)

| Model                       | Accuracy | ROC-AUC | PR-AUC | Best F1 |
|----------------------------|:--------:|:-------:|:------:|:-------:|
| **Logistic Regression**    | **70%**  | **0.704** | **0.460** | **0.519** |
| RandomForest Classifier    | 25%      | 0.571   | 0.390  | 0.377   |
| GradientBoosting Classifier| 77%      | 0.614   | 0.357  | 0.414   |
| HistGradientBoosting       | 42%      | 0.632   | 0.421  | 0.457   |

**Selection:** **Logistic Regression** is best — highest **PR-AUC** & **F1** (most relevant under class imbalance) with strong ROC-AUC.  
Accuracy is secondary here.

---

## Quick Error Review

- **False Positives:** power users (very high events/minutes) with many downgrades/errors → raw counts inflate risk.
- **False Negatives:** very active churners right up to cancel (lack **recency/trend** features), plus some cold-start low-activity users.


---


---

## API (FastAPI) — Demo-Ready

Request example:

```json
{
  "events": [
    {
      "ts": 1541106106796,
      "page": "NextSong",
      "level": "free",
      "auth": "Logged In",
      "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
  ],
  "threshold": 0.5
}
```

Response example:

```json
[
  { "probability": 0.37, "label": 0 }
]
```

_Train once before serving so that `./artifacts` contains `best_churn_pipeline.joblib` and `metadata.json`._

---




