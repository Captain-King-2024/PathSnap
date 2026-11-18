# PathSnap

Distributed map-matching and route reconstruction.

Preprocessing, road candidate generation, feature extraction, reference-route alignment, the training runner, and dataset audits are implemented. Logistic regression is trained and reproduced across three trip-grouped validation folds. It scored 80.48% top-1 on the aligned student-test observations, against 54.30% for nearest road.

The trained small MLP scored 81.10% on the same aligned student-test observations. The other comparison models are K-nearest neighbors (KNN), random forest, and histogram gradient boosting. Their results have yet to be added to this repository. Final model selection and Bayesian optimisation are pending.

Run `python -m api.server` from the repository root and open `http://127.0.0.1:8000` to review routes using logistic regression and the Viterbi decoder. See the [demo setup](api/README.md) for data prerequisites and network access.

For a browser on a second laptop, follow the [two-laptop setup](deployment/README.md).

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
| models | Trained logistic regression weights and fitted feature scaling. |
| results/training | Logistic regression and nearest-road baseline metrics. |
| results/feature_audit | Candidate-search, projection, and alignment coverage diagnostics. |
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

Follow the [training setup](src/training/README.md) to prepare the supplied dataset and reproduce the results. See the [feature audit instructions](src/features/README.md) for search checks.

## Submission

Submit this public repository and one consolidated technical PDF. Keep evaluator reference labels server-side and outside model inputs and UI responses. Disclose AI assistance in the submission as required by the hackathon.

