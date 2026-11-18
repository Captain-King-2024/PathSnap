"""Train and cross-validate a Random Forest arc classifier.

Usage (from repo root):
    python src/training/train_random_forest.py

Outputs
-------
models/random_forest/
    model.joblib          — fitted sklearn RandomForestClassifier
    preprocessing.npz     — StandardScaler parameters (fitted on train only)
    config.json           — exact hyperparameters used
    feature_names.json    — ordered feature list
    training_summary.json — CV scores, OOB, class counts, timing
results/metrics/
    rf_cv_scores.csv      — per-fold accuracy, F1, log-loss
results/logs/
    rf_training.log       — timestamped run log
"""
import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

# Ensure local imports work whether run from repo root or src/training
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from cv_utils import describe_splits, get_group_kfold_splits
except ImportError:
    from src.training.cv_utils import describe_splits, get_group_kfold_splits


# ── Logging ──────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)s  %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


# ── Feature Scaling ───────────────────────────────────────────────────────────
def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit StandardScaler on training fold only. RF is scale-invariant but
    scaling enables fair comparison with logistic/SVM baselines later."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


# ── CV Evaluation ─────────────────────────────────────────────────────────────
def run_cv(X, y, groups, params: dict, n_splits: int, seed: int, log) -> list:
    splits = get_group_kfold_splits(X, y, groups, n_splits=n_splits, seed=seed)
    describe_splits(splits, groups)

    fold_records = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        scaler = fit_scaler(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)

        t0 = time.perf_counter()
        clf = RandomForestClassifier(**params)
        clf.fit(X_tr_s, y_tr)
        elapsed = time.perf_counter() - t0

        proba = clf.predict_proba(X_va_s)
        pred = clf.predict(X_va_s)

        acc = accuracy_score(y_va, pred)
        f1_mac = f1_score(y_va, pred, average='macro', zero_division=0)
        f1_pos = f1_score(y_va, pred, average='binary', zero_division=0)
        ll = log_loss(y_va, proba)
        cm = confusion_matrix(y_va, pred).tolist()

        record = dict(
            fold=fold,
            accuracy=acc,
            macro_f1=f1_mac,
            positive_f1=f1_pos,
            log_loss=ll,
            confusion_matrix=cm,
            train_candidates=int(len(tr_idx)),
            val_candidates=int(len(va_idx)),
            train_trips=int(len(np.unique(groups[tr_idx]))),
            val_trips=int(len(np.unique(groups[va_idx]))),
            fit_seconds=round(elapsed, 2),
        )
        fold_records.append(record)
        log.info(
            f'Fold {fold}: acc={acc:.4f}  macro_F1={f1_mac:.4f}  '
            f'pos_F1={f1_pos:.4f}  log_loss={ll:.4f}  ({elapsed:.1f}s)'
        )

    return fold_records


# ── Full-data Fit ─────────────────────────────────────────────────────────────
def fit_final_model(X, y, params: dict, log):
    scaler = fit_scaler(X)
    X_s = scaler.transform(X)
    log.info(f'Fitting final model on all {len(X):,} candidates ...')
    t0 = time.perf_counter()
    clf = RandomForestClassifier(**params)
    clf.fit(X_s, y)
    elapsed = time.perf_counter() - t0
    log.info(f'Final model fitted in {elapsed:.1f}s  OOB score: {clf.oob_score_:.4f}')
    return clf, scaler


# ── Save Artifacts ────────────────────────────────────────────────────────────
def save_artifacts(
    clf,
    scaler,
    params,
    feature_names,
    fold_records,
    cfg,
    model_dir: Path,
    metrics_dir: Path,
    log,
):
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Model
    joblib.dump(clf, model_dir / 'model.joblib', compress=3)
    log.info(f'Saved model → {model_dir}/model.joblib')

    # Scaler parameters (as numpy so no joblib dependency for inference)
    np.savez_compressed(
        model_dir / 'preprocessing.npz',
        mean=scaler.mean_,
        scale=scaler.scale_,
    )
    log.info(f'Saved scaler → {model_dir}/preprocessing.npz')

    # Config snapshot
    (model_dir / 'config.json').write_text(json.dumps(params, indent=2), encoding='utf-8')
    (model_dir / 'feature_names.json').write_text(
        json.dumps(feature_names, indent=2), encoding='utf-8'
    )

    # CV summary
    mean_macro_f1 = float(np.mean([r['macro_f1'] for r in fold_records])) if fold_records else 0.0
    mean_acc = float(np.mean([r['accuracy'] for r in fold_records])) if fold_records else 0.0
    mean_ll = float(np.mean([r['log_loss'] for r in fold_records])) if fold_records else 0.0

    summary = dict(
        model='random_forest',
        cv_folds=len(fold_records),
        mean_macro_f1=mean_macro_f1,
        mean_accuracy=mean_acc,
        mean_log_loss=mean_ll,
        oob_score=float(clf.oob_score_) if hasattr(clf, 'oob_score_') and clf.oob_score_ is not None else None,
        n_estimators=clf.n_estimators,
        feature_importances={
            fn: float(fi) for fn, fi in zip(feature_names, clf.feature_importances_)
        },
        folds=fold_records,
        hyperparameters=params,
    )
    (model_dir / 'training_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8'
    )
    if fold_records:
        log.info(f'CV mean macro-F1={mean_macro_f1:.4f}  acc={mean_acc:.4f}')

    # CSV for model comparison table
    if fold_records:
        with (metrics_dir / 'rf_cv_scores.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    'fold',
                    'accuracy',
                    'macro_f1',
                    'positive_f1',
                    'log_loss',
                    'train_candidates',
                    'val_candidates',
                    'train_trips',
                    'val_trips',
                    'fit_seconds',
                ],
            )
            writer.writeheader()
            for r in fold_records:
                writer.writerow({k: r[k] for k in writer.fieldnames})
        log.info(f'Saved CV scores → {metrics_dir}/rf_cv_scores.csv')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=Path('data/training_data.npz'))
    ap.add_argument('--config', type=Path, default=Path('configs/random_forest.json'))
    ap.add_argument('--model-dir', type=Path, default=Path('models/RandomForest'))
    ap.add_argument('--metrics-dir', type=Path, default=Path('results/metrics'))
    ap.add_argument('--log-dir', type=Path, default=Path('results/logs'))
    ap.add_argument('--skip-cv', action='store_true', help='Skip CV, fit final model only')
    args = ap.parse_args()

    log = setup_logging(args.log_dir / 'rf_training.log')
    log.info('=== Random Forest Training ===')

    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    params = cfg['default_params']
    feature_names = cfg['feature_names']
    n_splits = cfg['cv_folds']
    seed = cfg['seed']

    log.info(f'Loading training data from {args.data} ...')
    with np.load(args.data, allow_pickle=False) as d:
        X = d['X'].astype(np.float32)
        y = d['y'].astype(np.int8)
        groups = d['groups']  # str array of trip_ids

    log.info(f'  Shape: {X.shape}  |  positives: {y.sum():,}  ({100*y.mean():.2f}%)')

    fold_records = []
    if not args.skip_cv:
        log.info(f'Running {n_splits}-fold group-aware CV ...')
        fold_records = run_cv(X, y, groups, params, n_splits, seed, log)

    clf, scaler = fit_final_model(X, y, params, log)
    save_artifacts(
        clf,
        scaler,
        params,
        feature_names,
        fold_records,
        cfg,
        args.model_dir,
        args.metrics_dir,
        log,
    )
    log.info('=== Done ===')


if __name__ == '__main__':
    main()
