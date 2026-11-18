"""Reproduce the 20-trial Gaussian-process HGB experiment."""
import json
import pickle
from pathlib import Path

import numpy as np
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from threadpoolctl import threadpool_limits

from src.optimization.tune import BASELINE, estimator, grouped_folds
from src.training.train import SEED, combine, evaluate, load_trips

TRIAL_COUNT = 20
FIT_ROW_CAP = 150_000


def decode(point):
    return dict(max_iter=int(round(60 + 180 * point[0])),
                max_leaf_nodes=int(round(15 + 48 * point[1])),
                learning_rate=float(.02 * (.25 / .02) ** point[2]),
                min_samples_leaf=int(round(10 * 16 ** point[3])),
                l2_regularization=float(5 * point[4]))


def main():
    features, labels = Path('data/features'), Path('data/labels')
    first = next(iter(sorted((features / 'train').glob('*/candidates.npz'))), None)
    if first is None:
        raise FileNotFoundError('Generate candidate features and aligned labels first.')
    with np.load(first, allow_pickle=False) as data:
        names = data['feature_names'].tolist()
    train = load_trips(features, labels, 'train', names)
    folds = grouped_folds(train)
    rng = np.random.default_rng(SEED)
    fold_data = []
    for i, fold in enumerate(folds):
        x, y, _, _ = combine([trip for j, other in enumerate(folds) if i != j for trip in other])
        indices = np.sort(rng.choice(len(y), min(len(y), FIT_ROW_CAP), replace=False))
        fold_data.append((x[indices], y[indices], combine(fold)))

    baseline_point = np.array([60 / 180, 16 / 48, np.log(5) / np.log(12.5), np.log(2) / np.log(16), 0])
    points, trials = [], []
    with threadpool_limits(limits=4):
        for i in range(TRIAL_COUNT):
            if i == 0:
                point, params = baseline_point, BASELINE
            elif i < 3:
                point = rng.random(5)
                params = decode(point)
            else:
                gp = GaussianProcessRegressor(kernel=Matern(length_scale=.45, nu=2.5),
                                              alpha=1e-5, normalize_y=True, optimizer=None)
                gp.fit(points, [trial['cv_macro_f1'] for trial in trials])
                pool = rng.random((2048, 5))
                mean, std = gp.predict(pool, return_std=True)
                point = pool[np.argmax(mean + 1.5 * std)]
                params = decode(point)
            scores = [evaluate('boosting', estimator(params).fit(x, y), validation)['macro_f1']
                      for x, y, validation in fold_data]
            points.append(point)
            trials.append(dict(params=params, fold_macro_f1=scores, cv_macro_f1=float(np.mean(scores))))
            print(f"Trial {i + 1}/{TRIAL_COUNT} macro F1: {trials[-1]['cv_macro_f1']:.4f}", flush=True)

        best = max(range(len(trials)), key=lambda i: trials[i]['cv_macro_f1'])
        student = load_trips(features, labels, 'student_test', names)
        if {t[0] for t in train} & {t[0] for t in student}:
            raise ValueError('Train and student-test trip IDs overlap.')
        test = combine(student)
        x, y, _, _ = combine(train)
        baseline = estimator(BASELINE).fit(x, y)
        selected = baseline if best == 0 else estimator(trials[best]['params']).fit(x, y)
        scores = dict(baseline=evaluate('boosting', baseline, test),
                      selected=evaluate('boosting', selected, test))

    report = dict(method='Gaussian process with upper confidence bound', seed=SEED,
                  cv_fit_row_cap=FIT_ROW_CAP, objective='macro_f1',
                  fold_trip_ids=[[t[0] for t in fold] for fold in folds],
                  trials=trials, best_trial=best, student_test=scores)
    report_path = Path('results/optimization/boosting_bayesian_20.json')
    model_path = Path('models/boosting_bayesian_20.pkl')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    with model_path.open('wb') as handle:
        pickle.dump(dict(model=selected, feature_names=names, sklearn_version=sklearn.__version__), handle)
    print(f'Results saved to {report_path} and {model_path}')


if __name__ == '__main__':
    main()
