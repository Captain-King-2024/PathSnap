# PathSnap | Technical report

## 1. Dataset and feature engineering

PathSnap matches noisy GPS observations to directed road segments. The experiments use 80 training trips and 10 student-testing trips. Model fitting and hyperparameter selection use the training trips. The reserved evaluator inputs are intended for the final demo.

The raw files contain GPS samples (.track), network nodes (.nodes), directed arcs (.arcs), and a chronological reference route (.route). Preprocessing validates coordinates and indices, preserves observation and arc identifiers, and projects longitude and latitude into local metre coordinates.

Across both development splits, 227,577 GPS observations were retained, with zero invalid rows removed and zero duplicate timestamps. There were 7,548 timestamp gaps longer than one second. The largest measured road-length projection difference was 27.84 m on a 9.3 km arc in training trip 00000028. That arc never entered its trip's candidate set.

For each observation the search considers up to eight directed road segments within 100 m. If the radius finds none, it returns the globally nearest roads and flags the observation. The 12 inputs are distance, distance rank, position along the segment, segment length, heading cosine, heading availability, previous step length, previous interval, previous speed, gap flag, and directed degree at each end.

Reference labels are derived by monotonic geometric alignment to the provided ordered route. Points farther than 100 m from the assigned reference arc are excluded from scoring. Candidate recall among usable points was 98.60% in training and 99.72% in student testing. This is coverage against derived labels. Parallel roads and repeated segments can remain ambiguous.

Evidence: results/preprocessing/summary.json, results/feature_audit/summary.json, docs/alignment_review.md.

## 2. Model experiments

The table compares saved experiments on the student-testing partition. Cross-validation scores are reported with their fold counts because runs with different fold counts are not strictly comparable. Timing includes candidate scoring and is hardware-dependent.

| Model | Folds | CV macro F1 | Test top-1 | Test macro F1 | Log loss | Inference (s) |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 3 | 86.96% | 80.48% | 88.63% | 0.1845 | 0.033 |
| Decision Tree | 3 | 87.35% | 81.62% | 89.31% | 0.1399 | 0.007 |
| Random Forest | 5 | 87.63% | 81.72% | 89.36% | 0.1349 | 0.156 |
| Small MLP | 3 | 87.34% | 81.10% | 89.00% | 0.1357 | 0.046 |
| Baseline HGB | 5 | 87.53% | 81.86% | 89.44% | 0.1357 | 0.646 |

Logistic regression uses standardized inputs and L2-regularized logistic loss (0.001). It is interpretable and fast but its linear score has limited capacity for feature interactions. Decision Tree uses Gini splits, depth 8, and at least 40 samples per leaf. It learns threshold rules and is very fast; one tree can be sensitive to small changes in the training data.

Random Forest averages 40 bootstrapped Gini trees (depth 14, leaf size 20, sample fraction 0.25). It reduces single-tree variance and adds inference cost. Small MLP uses standardized inputs, ReLU layers of 64 and 32 units, and Adam for up to 15 epochs. It has strong log loss but relies on scaling and training settings.

Histogram Gradient Boosting fits nonlinear trees sequentially with log loss. Its baseline uses 120 iterations, 31 maximum leaves, and learning rate 0.10. The tuned HGB has the best measured student-test top-1 among the saved comparable model reports. The nearest-road rule achieved 54.30% top-1.

KNN was also tested separately with a 40,000-row fit budget. Its tuned run reached 81.00% top-1. The experiment was removed from Git at the team's request. Reported validation and student scores do not establish overfitting because matched training-set scores were not recorded.

## 3. Bayesian search and Viterbi

A Gaussian process with a Matern kernel and upper-confidence-bound acquisition ran 20 trials. The baseline and two random settings seeded the search. The objective was mean candidate macro F1 on three trip-grouped folds using up to 150,000 fit rows per fold. The search ranges were 60-240 iterations, 15-63 leaves, learning rate 0.02-0.25, minimum leaf size 10-160 and L2 penalty 0-5.

Trial 7 selected 190 iterations, 24 leaves, learning rate 0.06231, minimum leaf size 16 and L2 penalty 0.39650. Validation macro F1 improved from 87.63% to 87.69%. Both configurations were refit on all 80 training trips. On 19,143 scored student observations, top-1 improved from 81.86% to 82.04%, macro F1 from 89.44% to 89.55%, and log loss from 0.13573 to 0.13248. This run measured 0.30 s baseline and 0.39 s tuned scoring.

The Viterbi hidden state is a directed candidate arc at a GPS observation. Emission weights come from the tuned HGB candidate probabilities. The decoder adds log emissions and a transition penalty proportional to the absolute difference between GPS displacement and effective directed road distance. Bounded shortest-path search supplies intermediate road geometry. Missing nearby candidates use the flagged nearest fallback. Intervals over 15 seconds, invalid time intervals and infeasible transitions start new route pieces.

The classifier alone achieved 82.04% top-1 and 1,989 pieces under the same feasibility rule. Viterbi reached 79.27% and 103 pieces across the same 10 student trips. This is a substantial continuity improvement with a 2.77 percentage-point point-accuracy cost. Road assignments can shift to a feasible nearby route even when independent emissions favor the reference arc. The UI exposes both views.

Evidence: results/optimization/boosting_bayesian_20.json and results/metrics/hgb_decoder.json.

## 4. Geographic review and deployment

The representative view below uses actual GPS, candidate roads, reference arcs and decoded geometry for student trip 00000000. Route gaps reflect decoder boundaries. The Folium exporter supports the same layers with a layer selector.

To regenerate the interactive map, run `python -m src.visualization.folium_map` after installing the demo dependencies.

Laptop 1 hosts the tuned model and FastAPI; Laptop 2 opens the evaluator UI on the same network. GET /api/health checks readiness, GET /api/trips lists the split, and GET /api/match/{trip} returns the geometry and any available student scores. Evaluator inputs can be prepared directly from the raw files and the evaluator view requires no reference labels.

The built-in server passed loopback checks for all 10 student trips, UI HTTP 200, and HTTP 404 for an unknown trip. Ten student trips were also prepared in an isolated evaluator-format directory and one was decoded in evaluator mode without scores. FastAPI and Folium execution, the reserved evaluator trips, and actual Laptop 2 screenshots still require verification on the demo machines.

AI assistance: Codex assisted with implementation, integration checks, and report drafting. Scores and figures are drawn from the recorded experiments. The team should review this report and use its own explanation when presenting.
