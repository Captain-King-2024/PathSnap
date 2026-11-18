"""Tune PathSnap's Random Forest using Optuna and trip-grouped cross-validation."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import optuna
import sklearn
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits

from src.training.train import SEED, combine, evaluate, load_trips

BASELINE = dict(n_estimators=40, max_depth=14, max_samples=0.25,
                min_samples_leaf=20, max_features='sqrt')


def parameters(trial):
    return dict(n_estimators=trial.suggest_int('n_estimators', 40, 120, step=20),
                max_depth=trial.suggest_int('max_depth', 8, 24),
                max_samples=trial.suggest_float('max_samples', 0.15, 0.50, step=0.05),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 5, 80, log=True),
                max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5]))


def estimator(params, jobs):
    return RandomForestClassifier(random_state=SEED, n_jobs=jobs, bootstrap=True, **params)


def grouped_folds(trips):
    """Assign complete trips using the training runner's split procedure."""
    folds = [[], [], []]
    sizes = [0, 0, 0]
    shuffled = np.random.default_rng(SEED).permutation(len(trips))
    for i in sorted(shuffled, key=lambda i: len(trips[i][3]), reverse=True):
        fold = int(np.argmin(sizes))
        folds[fold].append(trips[i])
        sizes[fold] += len(trips[i][3])
    return folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--objective', choices=['macro_f1', 'top1_accuracy'], default='macro_f1')
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--features', type=Path, default=Path('data/features'))
    parser.add_argument('--labels', type=Path, default=Path('data/labels'))
    args = parser.parse_args()
    if args.trials < 6 or args.threads < 1:
        parser.error('Use at least six trials and one thread.')
    first = next((args.features / 'train').glob('*/candidates.npz'), None)
    if first is None:
        parser.error('Generate candidate features and aligned labels first.')
    with np.load(first, allow_pickle=False) as data:
        names = data['feature_names'].tolist()
    train = load_trips(args.features, args.labels, 'train', names)
    folds = grouped_folds(train)
    validation_sets = [combine(fold) for fold in folds]

    def objective(trial):
        params = parameters(trial)
        values = []
        for i, validation in enumerate(validation_sets):
            x, y, _, _ = combine([t for j, fold in enumerate(folds) if j != i for t in fold])
            model = estimator(params, args.threads).fit(x, y)
            values.append(evaluate('forest', model, validation)[args.objective])
        trial.set_user_attr('fold_scores', values)
        return float(np.mean(values))

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5))
    study.enqueue_trial(BASELINE)
    with threadpool_limits(limits=1):
        study.optimize(objective, n_trials=args.trials, n_jobs=1)
        student = load_trips(args.features, args.labels, 'student_test', names)
        if {t[0] for t in train} & {t[0] for t in student}:
            raise ValueError('Train and student-test trip IDs overlap.')
        test = combine(student)
        x, y, _, _ = combine(train)
        baseline = estimator(BASELINE, args.threads).fit(x, y)
        tuned = estimator(study.best_params, args.threads).fit(x, y)
        held_out = {
            'baseline': evaluate('forest', baseline, test),
            'tuned': evaluate('forest', tuned, test),
        }
    report = dict(model='forest', objective=args.objective, seed=SEED,
                  baseline_params=BASELINE, best_params=study.best_params,
                  best_trial=study.best_trial.number, best_cv=float(study.best_value),
                  fold_trip_ids=[[trip[0] for trip in fold] for fold in folds],
                  optuna_version=optuna.__version__, sklearn_version=sklearn.__version__,
                  feature_names=names, student_test=held_out,
                  trials=[dict(number=t.number, params=t.params, value=t.value,
                               state=t.state.name, fold_scores=t.user_attrs.get('fold_scores'))
                          for t in study.trials])
    report_path = Path('results/optimization/forest_bayesian.json')
    model_path = Path('models/forest_bayesian.pkl')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    with model_path.open('wb') as handle:
        pickle.dump(dict(model=tuned, feature_names=names,
                         sklearn_version=sklearn.__version__), handle)
    print(f'Results: {report_path} | Model: {model_path}')


if __name__ == '__main__':
    main()
