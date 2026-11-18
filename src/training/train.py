"""Run trip-grouped candidate classification experiments."""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

MODELS = ['logistic', 'knn', 'forest', 'boosting', 'mlp', 'tree']
SEED = 557


def load_trips(root, labels, split, names):
    rows = []
    paths = sorted((root / split).glob('*/candidates.npz'))
    expected = {'train': 80, 'student_test': 10}[split]
    if len(paths) != expected:
        raise ValueError(f'{split}: expected {expected} feature files, found {len(paths)}')
    for path in paths:
        trip = path.parent.name
        with np.load(path) as f, np.load(labels / split / f'{trip}.npz') as y:
            if not np.array_equal(f['observation_ids'], y['observation_ids']):
                raise ValueError(f'Observation IDs disagree for {trip}')
            if not np.array_equal(f['feature_names'], names):
                raise ValueError(f'Feature schema disagrees for {trip}')
            counts = np.diff(f['offsets'])
            usable = y['usable']
            keep = np.repeat(usable, counts)
            target = np.repeat(y['arc_ids'], counts)
            matched = f['candidate_arc_ids'][keep] == target[keep]
            rows.append((trip, f['features'][keep], matched, counts[usable], len(counts)))
    return rows


def combine(trips):
    _, x, y, counts, totals = zip(*trips)
    return np.concatenate(x), np.concatenate(y), np.concatenate(counts), sum(totals)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def fit_logistic(x, y):
    # Standardize within the fit partition and use damped Newton steps.
    mean = x.mean(axis=0, dtype=np.float64)
    scale = x.std(axis=0, dtype=np.float64)
    scale[scale == 0] = 1
    design = np.empty((len(x), x.shape[1] + 1), dtype=np.float64)
    design[:, 0] = 1
    design[:, 1:] = np.clip((x - mean) / scale, -20, 20)
    weights = np.zeros(design.shape[1])
    regularization = np.full(len(weights), 0.001)
    regularization[0] = 0
    def objective(params):
        logits = design @ params
        return np.mean(np.logaddexp(0, logits) - y * logits) + 0.5 * np.dot(regularization, params ** 2)
    for _ in range(20):
        p = sigmoid(design @ weights)
        gradient = design.T @ (p - y) / len(x) + regularization * weights
        if np.max(np.abs(gradient)) < 1e-6:
            break
        curvature = p * (1 - p)
        hessian = (design.T * curvature) @ design / len(x) + np.diag(regularization)
        step = np.linalg.solve(hessian, gradient)
        loss = objective(weights)
        for damping in 0.5 ** np.arange(20):
            trial = weights - damping * step
            if objective(trial) < loss:
                weights = trial
                break
        else:
            break
    return dict(mean=mean, scale=scale, weights=weights)


def fit(name, x, y, jobs):
    if not y.any() or y.all():
        raise ValueError('Training fold needs both candidate classes')
    if name == 'logistic':
        return fit_logistic(x, y), dict(l2=0.001, max_iter=20, standardized_clip=20)
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    if name == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=15, weights='distance',
                                           algorithm='kd_tree', n_jobs=jobs)
        model = make_pipeline(StandardScaler(), classifier)
        return model.fit(x, y), dict(scaling='StandardScaler', **classifier.get_params())
    if name == 'tree':
        model = DecisionTreeClassifier(max_depth=8, min_samples_leaf=40, random_state=SEED)
    elif name == 'forest':
        model = RandomForestClassifier(n_estimators=40, max_depth=14, max_samples=0.25,
                                       min_samples_leaf=20, n_jobs=jobs, random_state=SEED)
    elif name == 'boosting':
        model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=31,
                                               early_stopping=False, random_state=SEED)
    else:
        from sklearn.exceptions import ConvergenceWarning
        import warnings
        classifier = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=15, batch_size=2048,
                                   early_stopping=False, random_state=SEED)
        model = make_pipeline(StandardScaler(), classifier)
        with warnings.catch_warnings():
            # A fixed epoch budget is intentional. Keep the actual loss history.
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            model.fit(x, y)
        return model, dict(scaling='StandardScaler', **classifier.get_params())
    return model.fit(x, y), model.get_params()


def scores(name, model, x):
    if name == 'logistic':
        z = np.clip((x - model['mean']) / model['scale'], -20, 20)
        return sigmoid(z @ model['weights'][1:] + model['weights'][0])
    return model.predict_proba(x)[:, 1]


def metrics(prob, target, counts, total):
    starts = np.r_[0, np.cumsum(counts[:-1])]
    highest = np.maximum.reduceat(prob, starts)
    indices = np.arange(len(prob))
    # On ties choose the closest candidate using the existing candidate order.
    choices = np.minimum.reduceat(np.where(prob == np.repeat(highest, counts), indices, len(prob)), starts)
    hits = int(target[choices].sum())
    covered = int(np.count_nonzero(np.logical_or.reduceat(target, starts)))
    tp = hits
    fp = len(counts) - hits
    fn = int(target.sum()) - tp
    tn = len(target) - tp - fp - fn
    p = np.clip(prob.astype(float), 1e-8, 1 - 1e-8)
    return dict(observations=total, aligned_observations=len(counts), candidate_covered=covered,
                candidate_miss_rate=1 - covered / len(counts), top1_accuracy=hits / len(counts),
                top1_when_covered=hits / covered if covered else None,
                macro_f1=0.5 * (2 * tp / max(1, 2 * tp + fp + fn) + 2 * tn / max(1, 2 * tn + fp + fn)),
                binary_log_loss=float(-np.mean(np.where(target, np.log(p), np.log1p(-p)))),
                confusion=dict(tp=tp, fp=fp, fn=fn, tn=tn))


def evaluate(name, model, dataset):
    x, y, counts, total = dataset
    start = time.perf_counter()
    prob = scores(name, model, x)
    return dict(inference_seconds=time.perf_counter() - start, **metrics(prob, y, counts, total))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=Path('data/features'))
    parser.add_argument('--labels', type=Path, default=Path('data/labels'))
    parser.add_argument('--output', type=Path, default=Path('results/training'),
                        help='Directory for evaluation reports')
    parser.add_argument('--model-dir', type=Path, default=Path('models'),
                        help='Directory for trained model artifacts')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--folds', type=int, default=3, choices=range(2, 6))
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()

    # Allow running the script from a repository subdirectory.
    if not (args.features / 'train').is_dir():
        repo_root = Path(__file__).resolve().parents[2]
        if (repo_root / args.features / 'train').is_dir():
            args.features = repo_root / args.features
            args.labels = repo_root / args.labels
            args.output = repo_root / args.output
            args.model_dir = repo_root / args.model_dir
    if set(args.models) - {'logistic'}:
        try:
            import sklearn
        except ImportError:
            parser.error('Install requirements-training.txt or use --models logistic.')
    first = next((args.features / 'train').glob('*/candidates.npz'), None)
    if first is None:
        parser.error('Run candidate generation before training.')
    with np.load(first) as f:
        names = f['feature_names'].tolist()
    train = load_trips(args.features, args.labels, 'train', names)
    student = load_trips(args.features, args.labels, 'student_test', names)
    if {t[0] for t in train} & {t[0] for t in student}:
        raise ValueError('Train and student_test trip IDs overlap')
    folds = [[] for _ in range(args.folds)]
    sizes = [0] * args.folds
    shuffled = np.random.default_rng(SEED).permutation(len(train))
    for i in sorted(shuffled, key=lambda i: len(train[i][3]), reverse=True):
        fold = int(np.argmin(sizes))
        folds[fold].append(train[i])
        sizes[fold] += len(train[i][3])
    args.output.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    test = combine(student)
    _, y, counts, total = test
    nearest = np.zeros(len(y))
    nearest[np.r_[0, np.cumsum(counts[:-1])]] = 1
    baseline = metrics(nearest, y, counts, total)
    del baseline['binary_log_loss']  # Baseline scores are hard decisions.
    (args.output / 'nearest.json').write_text(json.dumps(baseline, indent=2))
    for name in dict.fromkeys(args.models):
        cv = []
        for i, validation in enumerate(folds):
            x, y, _, _ = combine([t for j, fold in enumerate(folds) if j != i for t in fold])
            started = time.perf_counter()
            model, _ = fit(name, x, y, args.jobs)
            elapsed = time.perf_counter() - started
            cv.append(dict(fold=i + 1, trip_ids=[t[0] for t in validation], fit_seconds=elapsed,
                           **evaluate(name, model, combine(validation))))
            print(name, 'fold', i + 1, 'top1', round(cv[-1]['top1_accuracy'], 4), flush=True)
            del x, y, model
        x, y, _, _ = combine(train)
        started = time.perf_counter()
        model, config = fit(name, x, y, args.jobs)
        fit_seconds = time.perf_counter() - started
        result = dict(model=name, configuration=config, seed=SEED, feature_names=names,
                      numpy_version=np.__version__, sklearn_version=sklearn.__version__ if name != 'logistic' else None,
                      folds=cv, cv_macro_f1=float(np.mean([f['macro_f1'] for f in cv])),
                      cv_top1=float(np.mean([f['top1_accuracy'] for f in cv])), final_fit_seconds=fit_seconds,
                      student_test=evaluate(name, model, test))
        if name == 'logistic':
            np.savez_compressed(args.model_dir / 'logistic.npz', **model, feature_names=names)
        else:
            with (args.model_dir / f'{name}.pkl').open('wb') as handle:
                pickle.dump(dict(model=model, feature_names=names, sklearn_version=sklearn.__version__), handle)
            if name == 'mlp':
                result['train_loss_by_epoch'] = model[-1].loss_curve_
        (args.output / f'{name}.json').write_text(json.dumps(result, indent=2))
        print(name, 'student_test', json.dumps(result['student_test']), flush=True)
        del x, y, model


if __name__ == '__main__':
    main()
