"""Tune PathSnap's Histogram Gradient Boosting using Optuna and trip-grouped cross-validation."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import optuna
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from src.training.train import SEED, combine, evaluate, load_trips

BASELINE = dict(max_iter=120, max_leaf_nodes=31, learning_rate=0.1,
                min_samples_leaf=20, l2_regularization=0.0)


def parameters(trial):
    return dict(max_iter=trial.suggest_int('max_iter', 60, 240),
                max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 15, 63),
                learning_rate=trial.suggest_float('learning_rate', 0.02, 0.25, log=True),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 10, 160, log=True),
                l2_regularization=trial.suggest_float('l2_regularization', 0.0, 5.0))


def estimator(params):
    return HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, **params)


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
            model = estimator(params).fit(x, y)
            values.append(evaluate('boosting', model, validation)[args.objective])
        trial.set_user_attr('fold_scores', values)
        return float(np.mean(values))

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5))
    study.enqueue_trial(BASELINE)
    with threadpool_limits(limits=args.threads):
        study.optimize(objective, n_trials=args.trials, n_jobs=1)
        student = load_trips(args.features, args.labels, 'student_test', names)
        if {t[0] for t in train} & {t[0] for t in student}:
            raise ValueError('Train and student-test trip IDs overlap.')
        test = combine(student)
        x, y, _, _ = combine(train)
        baseline = estimator(BASELINE).fit(x, y)
        tuned = estimator(study.best_params).fit(x, y)
        held_out = {
            'baseline': evaluate('boosting', baseline, test),
            'tuned': evaluate('boosting', tuned, test),
        }
    report = dict(model='boosting', objective=args.objective, seed=SEED,
                  baseline_params=BASELINE, best_params=study.best_params,
                  best_trial=study.best_trial.number, best_cv=float(study.best_value),
                  fold_trip_ids=[[trip[0] for trip in fold] for fold in folds],
                  optuna_version=optuna.__version__, sklearn_version=sklearn.__version__,
                  feature_names=names, student_test=held_out,
                  trials=[dict(number=t.number, params=t.params, value=t.value,
                               state=t.state.name, fold_scores=t.user_attrs.get('fold_scores'))
                          for t in study.trials])
    report_path = Path('results/optimization/boosting_bayesian.json')
    model_path = Path('models/boosting_bayesian.pkl')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    with model_path.open('wb') as handle:
        pickle.dump(dict(model=tuned, feature_names=names,
                         sklearn_version=sklearn.__version__), handle)
    print(f'Results: {report_path} | Model: {model_path}')


if __name__ == '__main__':
    main()
