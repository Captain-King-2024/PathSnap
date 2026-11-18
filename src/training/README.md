# training

Training code for six models, including a standalone Decision Tree. Run `python src/training/train.py --models tree` to train only that model.

From the repository root, run these commands in PowerShell with Python 3.11–3.13. Replace the dataset path with the local directory containing `train` and `student_test`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe src/data_processing/preprocess.py --data-root "C:\path\to\Student folder"
.\.venv\Scripts\python.exe src/features/candidates.py
.\.venv\Scripts\python.exe src/training/align_routes.py
.\.venv\Scripts\python.exe src/training/train.py
```

If processed data and features are already present, start with alignment. Teammates can split the work by passing `--models knn forest` or `--models boosting mlp`. Use the default three folds and seed across runs. Each model writes its metrics under `results/training/` and its fitted artifact under `models/`. Use `--output` to override the metrics directory and `--model-dir` to override the artifact directory. Logistic regression can run with NumPy alone using `--models logistic`.

The runner's KNN configuration uses 15 neighbors and distance weighting after standardizing features within each training fold. For a separately trained KNN run, record its actual neighbor count, weighting, distance metric, and scaling settings.

Alignment uses dynamic programming to minimize clipped squared GPS-to-segment distance while moving forward through the reference route. It allows staying on one arc and skipping arcs. GPS points more than 100 m from their aligned arc are excluded from fitting and scored metrics. These targets are approximate and may be ambiguous on loops, parallel roads and junctions. Alignment uses reference routes and is only a supervision step. Inference uses candidate features.

Cross-validation uses whole trips from the 80 training trips, balancing folds by aligned observation counts. Every model is refitted on the 80 training trips and compared on the 10 student-testing trips. Top-1 includes candidate misses among usable observations. Coverage is the fraction of usable observations whose aligned arc appears in the candidate set. Macro F1 uses binary candidate predictions with exactly one selected candidate per observation. Binary log loss uses the raw candidate probabilities. The nearest-road baseline and each model record alignment counts so the score denominator is visible.

Logs record the feature schema, settings, versions, fold trip IDs, timing and confusion counts. The logistic artifact contains NumPy weights and scaling parameters. The other artifacts contain fitted scikit-learn estimators via pickle. Load pickle files only from trusted teammates, using the scikit-learn version in their metrics file. Evaluator data is reserved for the final evaluation.

## Random Forest Model

Random Forest arc classifier for candidate ranking and emission probability calculation in map-matching.

### Workflow

1. Assemble pointwise labels from candidate features and ground truth routes:
```bash
python src/training/label_assembler.py \
    --features-root data/features/train \
    --references-root data/_references \
    --output data/training_data.npz \
    --config configs/random_forest.json
```

2. Train with 5-fold group-aware cross-validation:
```bash
PYTHONPATH=. python src/training/train_random_forest.py
```

3. Hyperparameter search with Optuna (Bayesian optimization):
```bash
PYTHONPATH=. python src/optimization/tune_random_forest.py --n-trials 50
```

4. Evaluate on student test set:
```bash
PYTHONPATH=. python src/evaluation/evaluate_random_forest.py
```

5. Plot feature importances:
```bash
python src/visualization/plot_rf_importance.py
```

### Artifacts Produced

- Model artifact: `models/random_forest/model.joblib`
- Fitted preprocessor: `models/random_forest/preprocessing.npz`
- Hyperparameters: `models/random_forest/config.json`
- Feature schema: `models/random_forest/feature_names.json`
- Training summary & CV metrics: `models/random_forest/training_summary.json`
- Cross-validation results: `results/metrics/rf_cv_scores.csv`
- Student test metrics: `results/metrics/rf_student_test_metrics.json`
