# training

Training code for classifier models in PathSnap. Save seeds, configurations, fitted preprocessing and artifacts.

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
