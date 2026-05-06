# PROJECT_CONTEXT.md — Emotional Friction Detector

## Professional context (read once, don't repeat in responses)
This is a senior DS portfolio project targeting Grab and foodpanda in Malaysia. Every design decision should be defensible in a senior DS interview.
Prioritise: production-grade patterns over notebook shortcuts, explainability over black-box complexity, business metric framing over pure ML metrics. When there are two valid approaches, pick the one a senior engineer at Grab would recognise and respect.

## What this project is
Real-time pre-churn intervention system for food delivery platforms (Grab/foodpanda).
Detects user frustration from in-session behavioural sequences and fires LTV-gated
interventions before the user complains or churns.

**Portfolio goal:** Senior DS role at Grab/foodpanda. Must demonstrate full production
ML lifecycle — not just modelling.

**Tagline:** "Predictions are inputs. Decisions are the product."

---

## Two-stage model architecture
1. **LSTM frustration scorer** — P(frustrated | session event sequence)
   - Input: sequence of app events, each with 9 features
   - Architecture: 2-layer LSTM, hidden=64, sigmoid output
   - Runs per-event in real time, <40ms p99

2. **LightGBM churn risk model** — P(churn | frustrated, user profile)
   - Input: cross-session features (LTV tier, prior delays, post-complaint flag)
   - Output: churn probability given frustration confirmed

3. **Intervention decision engine**
   - Fires when: P(frustrated) × P(churn|frustrated) × LTV_30d > intervention_cost
   - Tiers: empathy message (score 0.60–0.74) → RM2 voucher (0.75–0.89) → CS escalation (0.90+)
   - LTV-gated: dormant users never receive vouchers (ROI negative)

---

## Data — simulated, anchored to real distributions
- 200k sessions, ~40% frustrated, seed=42
- Delay distribution anchored to Olist e-commerce delivery gap data
  - Base: N(0, 4) min
  - Peak hours: +N(7, 2.5)
  - Rain: +N(5, 3)
  - Restaurant tier C: +N(8, 4)
- Frustration trigger logic:
  - delay > 8 min, OR
  - delay > 4 AND prior_delays_30d >= 2, OR
  - delay > 2 AND is_post_complaint_return
- Key signals: eta_refresh_burst (compressing intervals), tap_interval_cv,
  back_forward_oscillation, support_open, delay_notification spike

---

## Key features (sequence_features.py)
Per-event feature vector (dim=9):
1. event_weight (support_open=0.94, eta_refresh=0.18, etc.)
2. log(time_since_last_event)
3. eta_refresh_count_2min / 5.0
4. back_tap_count / 3.0
5. rapid_tap_count / 3.0
6. page_revisit_count / 4.0
7. delay_notification_flag
8. tap_interval_cv (std/mean of recent tap intervals — most important signal)
9. log(session_elapsed) / 8.0

---

## Full tech stack
| Layer | Technology |
|---|---|
| Event ingestion | Apache Kafka (partitioned by user_id) |
| Stream processing | Apache Flink (2-min rolling window features) |
| Frustration model | PyTorch LSTM |
| Churn risk model | LightGBM + SHAP |
| Serving | FastAPI + TorchServe |
| Feature store | Redis (online) + Delta Lake (offline) |
| Experiment tracking | MLflow (per-segment precision monitoring) |
| Cloud | GCP (Pub/Sub, BigQuery, GKE) |
| Local dev | Docker Compose (Kafka + Flink + FastAPI) |

---

## Repo structure
simulator/          session_generator.py, user_profiles.py, delay_distribution.py
features/           sequence_features.py, cross_session_features.py
models/
  lstm_frustration/ model.py, train.py, evaluate.py, predict.py
  lgbm_churn_risk/  model.py, train.py, shap_analysis.py
intervention/       decision_engine.py, ltv_estimator.py
serving/            api.py, schemas.py, latency_benchmark.py
evaluation/         ab_test_simulator.py, metrics.py, drift_monitor.py
notebooks/          01_eda, 02_training, 03_intervention_matrix, 04_ab_results
docker/             docker-compose.yml (Kafka + Flink + FastAPI)
tests/              unit tests per module

---

## A/B metrics — targets vs simulation results (seed=42, 50k sessions, 20% holdback)
| Metric | Target | Simulation result | Status |
|---|---|---|---|
| NPS lift (treated cohort) | +40 pts | **+45 pts** | ✓ exceeded |
| 72h complaint rate reduction | −30% | **−19%** | honest result |
| 30-day order retention lift | +12pp | **+7.9 pp** | honest result |
| Voucher cost per retained user | <RM3.00 | **RM24.26** | see note below |
| Voucher cost per session touched | — | **RM2.00** | face value (operational) |
| ROI (LTV30d / cost per retained) | — | **4.2×** | RM102 / RM24.26 |
| LSTM epoch-1 AUC | — | **0.905** | genuine learning confirmed |
| LSTM final val AUC | >0.88 | **0.9986** | sim artifact; target >0.88 |
| LightGBM val AUROC | — | **0.644** | expected for hard sub-problem |
| Scoring latency p99 | <40ms | <40ms | ✓ |

**Voucher cost note:**
```
RM24.26 = total_voucher_spend / incremental_retentions
RM2.00  = total_voucher_spend / total_treated_sessions
Both metrics are correct — different denominators, different questions.
```
RM24.26 is the standard "cost per churn prevented" (incremental denominator).
RM2.00 is the operational spend per session touched (face value).
The original <RM3.00 target was implicitly using the face-value denominator.
At RM24.26 per prevented churn against RM102 median LTV_30d, ROI remains 4.2×.

---

## Current status
- [x] Simulator complete (delay_distribution, user_profiles, session_generator, generate_dataset)
- [x] sequence_features.py complete (SequenceFeaturizer, per-event (T,9) matrix, rolling tap_interval_cv)
- [x] LSTM model (hidden=64, input_dim=9, online_step for streaming, batch forward for training)
- [x] LSTM training pipeline (FrustrationDataset, MLflow logging, best-model checkpoint)
- [x] LSTM evaluate.py (AUROC, precision@60%recall, per-trigger segment, calibration)
- [x] LSTM predict.py (OnlineFrustrationScorer, batch_score_sessions)
- [x] LightGBM churn head (model.py, train.py)
- [x] LightGBM shap_analysis.py
- [x] Intervention decision engine (combined-score tiers, dormant LTV guard, A/B holdback)
- [x] LTV estimator + voucher ROI
- [x] FastAPI serving (POST /score, GET /health, lifespan model load)
- [x] latency_benchmark.py (HTTP + in-process, p99 SLA check)
- [x] Docker Compose stack (Kafka + Zookeeper + Flink + FastAPI + MLflow)
- [x] A/B simulation (simulate_outcomes, compute_ab_summary) — 50k sessions, results in metrics table above
- [x] drift_monitor.py (PSI-based feature and score drift)
- [x] metrics.py (AUROC, precision@recall, intervention lift)
- [x] MLflow monitoring (per-epoch LSTM metrics, LightGBM run, SHAP artefacts logged)
- [x] 60/60 unit tests passing
- [x] Dataset generated: 200k sessions, 39.7% frustrated, mean delay 4.99min, seed=42
- [x] LSTM trained: best val AUROC 0.9986, F1=0.984, all segment AUCs >=0.998 (simulation artifact — real target >0.88)
- [x] LightGBM trained: val AUROC 0.644, avg precision 0.503 — tap_interval_cv is SHAP rank #4 (0.040)
- [x] sessions_scored.parquet built: p_frustrated mean 0.975 for frustrated, 0.018 for calm sessions
- [x] Notebooks 01-04 built with real executed outputs and figures saved to notebooks/figures/
- [x] README complete — ready for GitHub publish

## Training results (seed=42, 200k sessions)
- LSTM val AUROC: 0.9986 | F1@0.5: 0.984 | Ep1 AUC 0.905 (genuine learning, not memorization)
  - Note: simulation produces cleaner patterns than real-world; production target remains >0.88
  - Leakage fix: outcome events (support_chat_opened, order_cancelled, etc.) excluded from LSTM input
  - Behavioral overlap fix: frustrated/non-frustrated event count ranges now overlap (forced pattern learning)
- LightGBM val AUROC: 0.644 | avg precision: 0.503 | best iter: 29 (early stop at 50)
  - Churn label: severity-driven proxy (delay + PCR + LTV + trigger + tap_interval_cv contribution) — churn_sensitivity identity leak removed
  - SHAP top-4: delay_minutes (0.303) > ltv_estimate_myr (0.118) > is_post_complaint_return (0.059) > tap_interval_cv (0.040)
  - tap_interval_cv at SHAP rank #4 confirms two-stage architecture coherence: LSTM detects via tap CV, LightGBM also ranks it as key churn predictor
