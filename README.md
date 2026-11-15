# PathSnap

Distributed map-matching and route reconstruction.

Status: scaffold only. Implementation, trained models, experimental results and deployment are pending.

The deliverables PDF specifies components, not exact folder names. The folders below cover those components.

| Folder | Purpose |
|---|---|
| src/data_processing | Read, inspect, clean and preprocess .track, .nodes, .arcs and .route files. Preserve trip-level partitions. |
| src/features | Candidate generation and feature extraction. Keep candidate-to-arc correspondence; fit learned transformations on training folds only. |
| src/training | Training code for at least five distinct models. Save seeds, configurations, fitted preprocessing and artifacts. |
| src/evaluation | Group-aware cross-validation, student-test metrics, confusion matrices and classifier versus decoder comparisons. Keep reference labels separate from prediction. |
| src/optimization | Bayesian hyperparameter search using group-aware CV on the 80 training trips. Save search spaces, objectives, trials and best parameters. |
| src/route_reconstruction | HMM emissions, road-network transitions and Viterbi decoding. Handle GPS gaps, missing candidates and disconnected roads. |
| src/visualization | Folium map generation and experiment plotting code. Show GPS observations, roads, predictions and development reference routes. |
| api | FastAPI application, endpoints, model loading and server-side evaluation. Evaluator labels must never enter prediction features or UI responses. |
| ui | Laptop 2 evaluator interface. Request the ten reserved trip IDs from the API, send selection and display predicted route and metrics. |
| deployment | Installation, execution and two-laptop deployment instructions, dependency locks and launch scripts. Record actual tested commands here once implemented. |
| configs | Versioned model, feature, decoder and Bayesian search configuration. Never store secrets. |
| models | Saved trained classifier, fitted preprocessing and inference configuration. Use Git LFS or release assets if required by file size. No model has been trained yet. |
| results/metrics | Actual model comparison CSVs and classifier/HMM evaluation summaries. |
| results/figures | Plots generated from real experiment outputs. Do not fabricate training curves. |
| results/maps | Interactive Folium HTML maps from actual trips. |
| results/logs | Experiment logs, seeds, timing and provenance. |
| results/optimization | Bayesian search trial history, best parameters and tuned-versus-untuned comparisons. |
| reports | One consolidated technical report PDF plus editable source. Follow docs/report_outline.md. |
| docs/evidence | Working UI screenshots and successful API response/log screenshots from the receiving laptop. |
| notebooks | Optional EDA and experiment notebooks. Keep reusable implementation in src. |
| tests | Data validation, leakage prevention, decoder correctness and API/UI integration checks. |
| data | Local dataset location only. Raw and derived data are ignored. Use train (80 trips), student_test (10 trips), and a separate evaluator_test partition supplied at final evaluation. |

## Setup

Clone this repository, then add verified dependency and execution instructions under deployment/ as implementation is completed. No runnable application exists yet.

## Submission

Submit this public repository and one consolidated technical PDF. Use real experiment results only. Keep evaluator reference labels server-side and outside model inputs and UI responses. AI assisted with the initial folder organisation and documentation; no experimental results are claimed.

