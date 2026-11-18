"""Group-aware CV utilities for PathSnap training.

A 'group' is a trip_id string. All candidate rows from the same trip
are always in the same fold. This prevents leakage because the HMM
uses per-trip network topology.
"""
import numpy as np
from sklearn.model_selection import GroupKFold


def get_group_kfold_splits(X, y, groups, n_splits: int = 5, seed: int = 42):
    """Return list of (train_idx, val_idx) arrays for n_splits folds.

    Deterministic: groups are sorted before splitting.
    """
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_groups)

    # Remap groups to shuffled order so KFold assignment is deterministic
    order_map = {g: i for i, g in enumerate(shuffled)}
    group_int = np.array([order_map[g] for g in groups], dtype=np.int64)

    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(X, y, group_int))


def describe_splits(splits, groups):
    """Print a fold summary: trip counts and candidate counts."""
    unique = np.unique(groups)
    print(f'  Total trips: {len(unique)}, Total candidates: {len(groups):,}')
    for fold, (tr, va) in enumerate(splits):
        tr_trips = len(np.unique(groups[tr]))
        va_trips = len(np.unique(groups[va]))
        print(
            f'  Fold {fold}: train {tr_trips} trips ({len(tr):,} cands) '
            f'| val {va_trips} trips ({len(va):,} cands)'
        )
