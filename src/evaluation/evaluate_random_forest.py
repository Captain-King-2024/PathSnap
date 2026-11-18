"""Evaluate saved Random Forest on the student-test partition.

Loads model from models/random_forest/.
Loads candidate features from data/features/student_test/.
Loads reference labels from data/_references/ (server-side only).

Usage:
    python src/evaluation/evaluate_random_forest.py

Outputs
-------
results/metrics/
    rf_student_test_metrics.json
    rf_student_test_per_trip.csv
    rf_confusion_matrix.npy
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
try:
    from rf_scorer import RFScorer
except ImportError:
    from src.training.rf_scorer import RFScorer


def load_reference(trip_id: str, references_root: Path) -> set:
    ref = np.load(references_root / f'{trip_id}.npz', allow_pickle=False)
    return set(ref['route_arc_ids'].tolist())


def evaluate_trip(
    scorer: RFScorer,
    trip_id: str,
    features_root: Path,
    references_root: Path,
) -> dict:
    cand_path = features_root / trip_id / 'candidates.npz'
    result = scorer.score_trip(cand_path)
    route_set = load_reference(trip_id, references_root)

    with np.load(cand_path, allow_pickle=False) as c:
        candidate_arc_ids = c['candidate_arc_ids']

    y_true = np.array(
        [1 if int(a) in route_set else 0 for a in candidate_arc_ids], dtype=np.int8
    )
    log_em = result['log_emission']
    proba = np.exp(log_em)
    pred = (proba >= 0.5).astype(np.int8)

    # top-1 accuracy per observation (did the highest-scoring candidate match a route arc?)
    offsets = result['offsets']
    top1_hits = []
    for i in range(len(offsets) - 1):
        sl = slice(offsets[i], offsets[i + 1])
        if sl.stop <= sl.start:
            continue
        best = int(np.argmax(proba[sl]))
        top1_hits.append(int(y_true[sl][best]))
    top1_acc = float(np.mean(top1_hits)) if top1_hits else 0.0

    return dict(
        trip_id=trip_id,
        candidates=len(y_true),
        positive=int(y_true.sum()),
        accuracy=accuracy_score(y_true, pred),
        macro_f1=f1_score(y_true, pred, average='macro', zero_division=0),
        positive_f1=f1_score(y_true, pred, average='binary', zero_division=0),
        log_loss=log_loss(
            y_true, np.column_stack([1 - proba, proba]).clip(1e-7, 1 - 1e-7)
        ),
        top1_accuracy=top1_acc,
        y_true=y_true,
        pred=pred,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', type=Path, default=Path('models/RandomForest'))
    ap.add_argument('--features-root', type=Path, default=Path('data/features/student_test'))
    ap.add_argument('--references-root', type=Path, default=Path('data/_references'))
    ap.add_argument('--metrics-dir', type=Path, default=Path('results/metrics'))
    args = ap.parse_args()

    scorer = RFScorer(args.model_dir)
    trips = sorted(p.name for p in args.features_root.iterdir() if p.is_dir())

    all_true, all_pred, records = [], [], []
    for trip_id in trips:
        r = evaluate_trip(scorer, trip_id, args.features_root, args.references_root)
        all_true.append(r.pop('y_true'))
        all_pred.append(r.pop('pred'))
        records.append(r)
        print(
            f"{trip_id}: top1={r['top1_accuracy']:.3f}  "
            f"macro_F1={r['macro_f1']:.4f}  acc={r['accuracy']:.4f}"
        )

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    cm = confusion_matrix(all_true, all_pred)

    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate summary
    summary = dict(
        model='random_forest',
        partition='student_test',
        trips=len(trips),
        mean_top1_accuracy=float(np.mean([r['top1_accuracy'] for r in records])) if records else 0.0,
        overall_accuracy=float(accuracy_score(all_true, all_pred)),
        overall_macro_f1=float(
            f1_score(all_true, all_pred, average='macro', zero_division=0)
        ),
        overall_positive_f1=float(
            f1_score(all_true, all_pred, average='binary', zero_division=0)
        ),
        confusion_matrix=cm.tolist(),
        per_trip=records,
    )
    (args.metrics_dir / 'rf_student_test_metrics.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8'
    )
    np.save(args.metrics_dir / 'rf_confusion_matrix.npy', cm)

    if records:
        with (args.metrics_dir / 'rf_student_test_per_trip.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    print(f"\nOverall macro-F1 : {summary['overall_macro_f1']:.4f}")
    print(f"Overall top-1 acc: {summary['mean_top1_accuracy']:.4f}")
    print(f"Confusion matrix :\n{cm}")


if __name__ == '__main__':
    main()
