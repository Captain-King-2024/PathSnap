"""Assemble pointwise binary labels for Random Forest training.

Usage:
    python src/training/label_assembler.py \
        --features-root data/features/train \
        --references-root data/_references \
        --output data/training_data.npz \
        --config configs/random_forest.json
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_trip_labels(trip_id: str, features_root: Path, references_root: Path):
    """Return (X, y, groups) for a single trip.

    X      : float32 array, shape (n_candidates, 12)
    y      : int8 array,   shape (n_candidates,)   — 1 if arc in route else 0
    groups : str array,    shape (n_candidates,)   — trip_id for GroupKFold
    """
    cand_path = features_root / trip_id / 'candidates.npz'
    ref_path = references_root / f'{trip_id}.npz'

    with np.load(cand_path, allow_pickle=False) as c:
        features = c['features'].astype(np.float32)  # (n_cand, 12)
        candidate_arc_ids = c['candidate_arc_ids']  # (n_cand,)

    with np.load(ref_path, allow_pickle=False) as r:
        route_arc_ids = r['route_arc_ids']  # (n_route_arcs,)

    route_set = set(route_arc_ids.tolist())
    y = np.array([1 if int(a) in route_set else 0 for a in candidate_arc_ids], dtype=np.int8)
    groups = np.array([trip_id] * len(features))
    return features, y, groups


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features-root', type=Path, default=Path('data/features/train'))
    ap.add_argument('--references-root', type=Path, default=Path('data/_references'))
    ap.add_argument('--output', type=Path, default=Path('data/training_data.npz'))
    ap.add_argument('--config', type=Path, default=Path('configs/random_forest.json'))
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    expected_features = cfg['feature_names']

    trips = sorted(p.name for p in args.features_root.iterdir() if p.is_dir())
    if not trips:
        raise ValueError(f'No trips found in {args.features_root}')

    all_X, all_y, all_groups = [], [], []
    for trip_id in trips:
        X, y, g = load_trip_labels(trip_id, args.features_root, args.references_root)
        if X.shape[1] != len(expected_features):
            raise ValueError(
                f'{trip_id}: feature count mismatch: got {X.shape[1]}, '
                f'expected {len(expected_features)}'
            )
        all_X.append(X)
        all_y.append(y)
        all_groups.append(g)
        pos = int(y.sum())
        print(f'{trip_id}: {len(X)} candidates, {pos} positive ({100*pos/len(y):.1f}%)')

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups, axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=X,
        y=y,
        groups=groups,
        feature_names=np.array(expected_features),
    )
    print(f'\nSaved: {args.output}')
    print(f'  Total candidates : {len(X):,}')
    print(f'  Positive (in route): {int(y.sum()):,} ({100*y.mean():.2f}%)')
    print(f'  Negative           : {int((y==0).sum()):,}')
    print(f'  Trips              : {len(trips)}')


if __name__ == '__main__':
    main()
