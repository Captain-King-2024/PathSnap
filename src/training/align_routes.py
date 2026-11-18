"""Align GPS observations to an ordered reference route for supervision."""
import argparse
import json
from pathlib import Path

import numpy as np


def align(points, starts, ends):
    """Minimize total clipped squared distance with nondecreasing route indices."""
    vectors = ends - starts
    squared = np.sum(vectors ** 2, axis=1)
    positions = np.arange(len(starts), dtype=np.int32)
    back = np.empty((len(points), len(starts)), dtype=np.int32)
    previous = np.zeros(len(starts))
    for i, point in enumerate(points):
        prefix = np.minimum.accumulate(previous)
        best = np.maximum.accumulate(np.where(
            np.r_[True, prefix[1:] < prefix[:-1]], positions, 0))
        back[i] = best
        offset = point - starts
        fraction = np.clip(np.divide(np.sum(offset * vectors, axis=1), squared,
                                     out=np.zeros_like(squared), where=squared > 0), 0, 1)
        residual = offset - fraction[:, None] * vectors
        previous = prefix + np.minimum(np.sum(residual ** 2, axis=1), 150 ** 2)
    indices = np.empty(len(points), dtype=np.int32)
    indices[-1] = previous.argmin()
    for i in range(len(points) - 1, 0, -1):
        indices[i - 1] = back[i, indices[i]]
    diff = points - starts[indices]
    chosen_vectors, chosen_sq = vectors[indices], squared[indices]
    fraction = np.clip(np.divide(np.sum(diff * chosen_vectors, axis=1), chosen_sq,
                                 out=np.zeros(len(points)), where=chosen_sq > 0), 0, 1)
    return indices, np.linalg.norm(diff - fraction[:, None] * chosen_vectors, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path('data/processed'))
    parser.add_argument('--output', type=Path, default=Path('data/labels'))
    args = parser.parse_args()
    for split, expected in [('train', 80), ('student_test', 10)]:
        paths = sorted((args.inputs / split).glob('*/inputs.npz'))
        if len(paths) != expected:
            parser.error(f'{split}: expected {expected} trips, found {len(paths)}')
        dest = args.output / split
        dest.mkdir(parents=True, exist_ok=True)
        totals = dict(trips=len(paths), observations=0, within_100m=0)
        for path in paths:
            trip = path.parent.name
            with np.load(path) as inputs, np.load(path.parents[1] / '_references' / f'{trip}.npz') as ref:
                route = ref['route_arc_ids']
                endpoints = inputs['nodes_xy'][inputs['arcs'][route]]
                sequence, distance = align(inputs['gps_xy'], endpoints[:, 0], endpoints[:, 1])
                valid = distance <= 100
                np.savez_compressed(dest / f'{trip}.npz', observation_ids=inputs['observation_ids'],
                                    arc_ids=route[sequence], route_index=sequence,
                                    alignment_distance_m=distance, usable=valid)
            totals['observations'] += len(sequence)
            totals['within_100m'] += int(valid.sum())
        print(split, json.dumps(totals), flush=True)


if __name__ == '__main__':
    main()
