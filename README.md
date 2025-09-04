# Churn Prediction

## Objective
Identify users most likely to cancel their subscription so that retention actions can be prioritized. We evaluate models primarily by **PR-AUC** and **F1**, which are more informative than Accuracy under class imbalance.

---

## Workflow 
1. **Data prep** → load JSONL, clean, derive time fields, inspect imbalance/outliers.  
2. **Label & windowing (leakage-safe)** → per-user **Cutoff**; features from **[Cutoff−30d, Cutoff)** only.  
3. **Modeling** → train multiple models; threshold tuning via PR curve; pick best by **PR-AUC / F1**.  
4. **Packaging & infra** → FastAPI service, scheduled retraining, MLflow tracking, drift checks; Docker & CI scaffold.

---

## Data

**Format:** JSON Lines (one event per line).

**Columns:**

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
- Drop rows with **empty `userId`**.  
- Convert `ts`/`registration` (ms) → **UTC datetimes**; add `event_time`, `event_date`.  

**Dataset (after cleaning)**
- Rows: **528,005** • Users: **448** • Range: **2018-10-01 → 2018-12-01 (UTC)**

---

## Labeling & Leakage Control

- **Positive (churner):** user has any `page == "Cancellation Confirmation"` → first such time = `churn_time`.  
- **Cutoff (per user):**  
  - Churners → `cutoff_time = churn_time`  
  - Non-churners → `cutoff_time = last_event_time`  
- **Observation window:** **`[cutoff_time − 30d, cutoff_time)`**.  
  All features computed **inside** the window → prevents **temporal leakage**.

---

## Features (per-user snapshot)

- **Volume & frequency:** `n_events`, `n_songs`, `n_sessions`, `n_days_active`, `events_per_day`  
- **Session time:** `total_session_min`, `avg_session_min`  
- **Diversity:** `n_unique_songs`, `n_artists`  
- **Tenure:** `tenure_days = cutoff − registration`  
- **Last known state:** `level`, `state`  
- **Behavioral counts:** `page_nextsong`, `page_logout`, `page_downgrade`, `page_submit_downgrade`, `page_error`, `page_roll_advert`, `page_thumbs_up`, …  

**Encoding & scaling:** One-Hot (categoricals) + StandardScaler (numerics).

**Split:** **Time-aware ~80/20** by `cutoff_time` (ensure both classes in Val). One row per user; no overlap.

---

## Results (validation)

| Model                        | Accuracy | ROC-AUC | PR-AUC | Best F1 |
|-----------------------------|:--------:|:------:|:------:|:-------:|
| **Logistic Regression**     | **70%**  | **0.704** | **0.460** | **0.519** |
| RandomForest Classifier     | 25%      | 0.571  | 0.390  | 0.377   |
| GradientBoosting Classifier | 77%      | 0.614  | 0.357  | 0.414   |
| HistGradientBoosting        | 42%      | 0.

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




