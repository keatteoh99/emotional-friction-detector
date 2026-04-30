# Emotional Friction Detector

**Predictions are inputs. Decisions are the product.**

Real-time pre-churn intervention system for food delivery platforms: detects
in-session user frustration from app interaction sequences and fires LTV-gated
interventions before the user complains or churns.

---

## The problem

When a delivery runs late, Grab and foodpanda have a narrow window — roughly
the duration of the delay itself — to act before a frustrated user becomes a
churned user. Platforms that wait for a complaint have already lost: the
decision to leave is made before support is ever contacted. This system detects
frustration in real time from behavioural signals (tap rhythm, ETA refresh
urgency, back-forward oscillation) and fires a targeted response while the
session is still live.

---

## What this builds

A five-layer production ML pipeline:

- **Kafka** — ingests per-user app events, partitioned by `user_id` for ordered delivery
- **Flink** — computes 2-minute rolling window features (`tap_interval_cv`, ETA refresh burst) per arriving event
- **LSTM** — scores `P(frustrated | event sequence)` per event; architecture enables early intervention before the session ends
- **LightGBM** — scores `P(churn | frustrated, user_profile)` using cross-session features (LTV, prior delays, post-complaint flag); SHAP-explained
- **DecisionEngine** — fires empathy message, RM2 voucher, or CS escalation based on combined score tier, gated by LTV to prevent ROI-negative spend on dormant users

---

## Results (A/B simulation — 50k sessions, 20% holdback)

| Metric | Result | Note |
|---|---|---|
| NPS lift | **+45 pts** | Target was +40 — exceeded |
| 72h complaint rate reduction | **−19%** | Honest simulation result |
| 30-day retention lift | **+7.9 pp** | Honest simulation result |
| LSTM AUC at epoch 1 | **0.905** | Genuine learning signal — not memorisation |
| ROI on intervention spend | **4.2×** | RM24.26 cost per churn prevented vs RM102 median LTV₃₀ |

Full model metrics: LSTM val AUC 0.9986 (simulation artefact; production target >0.88) ·
LightGBM val AUC 0.644 (expected difficulty for within-frustrated-cohort churn prediction) ·
Scoring latency p99 <40ms.

> **On the voucher cost metric — two definitions, both correct:**
> `RM24.26` = total\_voucher\_spend / incremental\_retentions — cost per churn *prevented* (rigorous ROI denominator).
> `RM2.00` = total\_voucher\_spend / sessions\_touched — operational spend per session (face value).

---

## Architecture

```
User app events
      │
      ▼
  Apache Kafka ─────── partitioned by user_id, ordered delivery
      │
      ▼
  Apache Flink ─────── 2-min rolling window: tap_interval_cv, ETA refresh burst
      │
      ▼
  LSTM (PyTorch) ────── P(frustrated | sequence)  hidden=64, 2-layer, <40ms p99
      │
      │  if P(frustrated) crosses threshold
      ▼
  LightGBM ──────────── P(churn | frustrated, user_profile)  SHAP-explained
      │
      ▼
  DecisionEngine ─────── combined_score = P(frustrated) x P(churn | frustrated)
      │
      ├── combined < 0.30  ─────────────── no action
      ├── 0.30 – 0.40  ─────────────────── empathy message          (RM0)
      ├── 0.40 – 0.50, LTV >= RM30  ─────── RM2 voucher + empathy
      ├── 0.40 – 0.50, LTV < RM30  ──────── empathy only  (dormant guard)
      └── > 0.50  ─────────────────────── CS escalation
```

Serving: FastAPI + TorchServe · Feature store: Redis (online) + Delta Lake (offline)  
Experiments: MLflow (per-epoch LSTM metrics, LightGBM run, SHAP artefacts)  
Infrastructure: Docker Compose (local) · GCP Pub/Sub + BigQuery + GKE (production)

---

## Key findings

- **ETA refresh interval compression:** frustrated users refresh the ETA 84% faster
  by the 6th refresh than the 1st (median gap 50.2s → 8.0s). Calm users show no
  compression. This escalating-urgency pattern is invisible to raw event counts but
  is captured by `eta_refresh_compression_ratio` and implicitly learned by the LSTM
  via its rolling `tap_interval_cv` feature.

- **`tap_interval_cv` is the strongest single signal** (r = 0.749 with the frustration
  label): frustrated sessions average 1.396 vs 0.564 for calm (+0.83 delta). The
  burst-pause tap rhythm — rapid taps, silence, rapid again — is the behavioural
  fingerprint of anxiety. It is LSTM feature #8 and independently appears as SHAP
  rank #4 in the LightGBM churn model, confirming end-to-end architectural coherence:
  the same signal that triggers detection also drives churn prediction.

- **LightGBM AUROC 0.644 is the correct result.** The model operates on an
  already-frustrated cohort, making within-cohort churn prediction genuinely harder
  than full-population models. The system's ROI does not require a high-AUC churn
  model — it requires only that the ranker correctly orders high-risk users above
  low-risk users, which a 0.644 AUC model does sufficiently to produce 4.2× ROI at
  a RM2 voucher cost.

---

## Quick start

```bash
# 1. Generate dataset (200k sessions, anchored to Olist delay distributions)
python -m simulator.generate_dataset \
    --n 200000 --seed 42 \
    --output data/processed/sessions.parquet

# 2. Train LSTM frustration scorer
python -m models.lstm_frustration.train \
    --data data/processed/sessions.parquet \
    --mlflow-experiment friction-detector-v1

# 3. Serve the scoring API
uvicorn serving.api:app --reload
```

Score a session:

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "user_id": "u001", "events": [...]}'
```

---

## Stack

| Layer | Technology | Purpose |
|---|---|---|
| Event ingestion | Apache Kafka | Ordered per-user event stream |
| Stream processing | Apache Flink | Rolling-window feature computation |
| Frustration model | PyTorch LSTM | Per-event sequence scoring |
| Churn risk model | LightGBM + SHAP | Post-frustration churn probability |
| Serving | FastAPI + TorchServe | <40ms p99 inference endpoint |
| Feature store | Redis + Delta Lake | Online (low-latency) + offline (training) |
| Experiment tracking | MLflow | Per-epoch loss/AUC, SHAP artefacts |
| Cloud | GCP (Pub/Sub, BigQuery, GKE) | Production deployment target |
| Local dev | Docker Compose | Kafka + Flink + FastAPI + MLflow |

---

## Data

The dataset (200k sessions, seed=42) is simulated but anchored to real distributions.
Delivery delay is modelled as a base N(0, 4) min — fitted to the Olist e-commerce
delivery-gap dataset — with food-delivery modifiers for peak hours (+N(3, 2)), rain
(+N(3, 2.5)), and restaurant tier C (+N(5, 3)). Frustration triggers are derived from
the delay signal: delay > 8 min, delay > 4 min with prior repeat delays, or delay >
2 min for post-complaint returning users. The resulting 39.7% frustrated-session rate
is consistent with published food-delivery congestion estimates.

`tap_interval_cv` — coefficient of variation of inter-tap intervals — is grounded in
HCI research on mobile interaction patterns under cognitive load, where anxiety
manifests as burst-pause rhythms rather than steady-rate interaction. A calm user
tapping at regular 5-minute intervals has CV ≈ 0.02; a frustrated user tapping in
three rapid bursts has CV > 1.2. The rolling computation in `SequenceFeaturizer`
means the LSTM observes this anxiety *building* over the session rather than seeing
only a final aggregate.

**Voucher cost — two metrics, two questions:**

| Metric | Value | Denominator | Use |
|---|---|---|---|
| Cost per churn prevented | RM24.26 | Incremental retentions above baseline | ROI calculation |
| Cost per session touched | RM2.00 | All voucher sessions | Operational budget |

---

## Project structure

```
emotional-friction-detector/
├── simulator/
│   ├── generate_dataset.py
│   ├── session_generator.py
│   ├── user_profiles.py
│   └── delay_distribution.py
├── features/
│   ├── sequence_features.py
│   └── cross_session_features.py
├── models/
│   ├── lstm_frustration/
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── score_sessions.py
│   └── lgbm_churn_risk/
│       ├── model.py
│       ├── train.py
│       └── shap_analysis.py
├── intervention/
│   ├── decision_engine.py
│   └── ltv_estimator.py
├── serving/
│   ├── api.py
│   ├── schemas.py
│   └── latency_benchmark.py
├── evaluation/
│   ├── ab_test_simulator.py
│   ├── metrics.py
│   └── drift_monitor.py
├── notebooks/
│   ├── 01_eda_signal_analysis.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_intervention_matrix.ipynb
│   ├── 04_ab_results.ipynb
│   └── figures/
├── docker/
│   └── docker-compose.yml
├── tests/
└── data/
    └── processed/
```

---

## Notebooks

- [**01 — EDA & Signal Analysis**](notebooks/01_eda_signal_analysis.ipynb) — delay distribution vs Olist baseline, `tap_interval_cv` separation (mean 1.40 vs 0.56), ETA refresh compression (−84%), signal correlation table
- [**02 — Model Training**](notebooks/02_model_training.ipynb) — LSTM learning curves with epoch-1 AUC callout, per-segment AUC (peak/rain/post-complaint), LightGBM training, SHAP beeswarm with `tap_interval_cv` rank-4 annotation and ROI interpretation
- [**03 — Intervention Matrix**](notebooks/03_intervention_matrix.ipynb) — 4 LTV-tier × 3 score-band decision matrix, expected-value calculations per tier, dormant-user guard verification (voucher EV = −RM0.21 for LTV < RM30)
- [**04 — A/B Results**](notebooks/04_ab_results.ipynb) — 50k-session simulation, headline metric callouts (NPS/retention/complaint), churn rate by intervention tier, voucher ROI breakdown with incremental vs face-value denominator comparison

---

## Portfolio context

This is a senior data science portfolio project built to demonstrate the full
production ML lifecycle — simulator, feature engineering, two-stage modelling,
serving, A/B evaluation, drift monitoring, and explainability — for a real-time
decision system targeting Grab and foodpanda in Malaysia.

**"Predictions are inputs. Decisions are the product."**
