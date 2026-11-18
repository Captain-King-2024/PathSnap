
"""Bayesian hyperparameter search for Random Forest using Optuna.

Usage:
    python src/optimization/tune_random_forest.py --n-trials 50

Outputs
-------
results/optimization/
    rf_optuna_trials.csv      — all trial results
    rf_best_params.json       — best parameters
    rf_tuned_vs_untuned.json  — comparison summary
configs/
    random_forest_tuned.json  — updated config with best params
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# Add src/training to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
try:
    from cv_utils import get_group_kfold_splits
except ImportError:
    from src.training.cv_utils import get_group_kfold_splits


def objective(trial, X, y, groups, n_splits, seed):
    params = dict(
        n_estimators=trial.suggest_int('n_estimators', 100, 600, step=50),
        max_depth=trial.suggest_categorical('max_depth', [None, 10, 20, 30, 40]),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 16),
        max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5]),
        class_weight=trial.suggest_categorical('class_weight', ['balanced', None]),
        n_jobs=-1,
        random_state=seed,
    )
    splits = get_group_kfold_splits(X, y, groups, n_splits=n_splits, seed=seed)
    f1s = []
    for tr, va in splits:
        sc = StandardScaler().fit(X[tr])
        clf = RandomForestClassifier(**params)
        clf.fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[va]))
        f1s.append(f1_score(y[va], pred, average='macro', zero_division=0))
    return float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=Path('data/training_data.npz'))
    ap.add_argument('--config', type=Path, default=Path('configs/random_forest.json'))
    ap.add_argument('--n-trials', type=int, default=50)
    ap.add_argument('--output-dir', type=Path, default=Path('results/optimization'))
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    seed = cfg['seed']
    n_folds = cfg['cv_folds']

    with np.load(args.data, allow_pickle=False) as d:
        X, y, groups = d['X'].astype(np.float32), d['y'].astype(np.int8), d['groups']

    # Baseline (default params)
    splits = get_group_kfold_splits(X, y, groups, n_splits=n_folds, seed=seed)
    base_f1s = []
    for tr, va in splits:
        sc = StandardScaler().fit(X[tr])
        clf = RandomForestClassifier(**cfg['default_params'])
        clf.fit(sc.transform(X[tr]), y[tr])
        base_f1s.append(
            f1_score(y[va], clf.predict(sc.transform(X[va])), average='macro', zero_division=0)
        )
    baseline_f1 = float(np.mean(base_f1s))
    print(f'Baseline macro-F1 (default params): {baseline_f1:.4f}')

    study = optuna.create_study(
        direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(
        lambda t: objective(t, X, y, groups, n_folds, seed),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f'Best macro-F1: {best.value:.4f}  params: {best.params}')

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save all trials
    with (args.output_dir / 'rf_optuna_trials.csv').open('w', newline='', encoding='utf-8') as f:
        fieldnames = ['trial', 'value'] + sorted(best.params.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in study.trials:
            row = {'trial': t.number, 'value': t.value}
            row.update(t.params)
            writer.writerow(row)

    # Best params
    best_params = dict(n_jobs=-1, random_state=seed, oob_score=True, **best.params)
    (args.output_dir / 'rf_best_params.json').write_text(
        json.dumps(best_params, indent=2), encoding='utf-8'
    )

    # Comparison summary
    comparison = dict(
        baseline_macro_f1=baseline_f1,
        tuned_macro_f1=best.value,
        improvement=best.value - baseline_f1,
        best_params=best_params,
        n_trials=args.n_trials,
    )
    (args.output_dir / 'rf_tuned_vs_untuned.json').write_text(
        json.dumps(comparison, indent=2), encoding='utf-8'
    )

    # Write tuned config
    tuned_cfg = dict(cfg)
    tuned_cfg['default_params'] = best_params
    Path('configs/random_forest_tuned.json').write_text(
        json.dumps(tuned_cfg, indent=2), encoding='utf-8'
    )
    print('Saved tuned config → configs/random_forest_tuned.json')


if __name__ == '__main__':
    main()
