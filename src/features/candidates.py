"""Generate directed road candidates and reference-free geometric features."""
import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

FEATURE_NAMES = ['distance_m', 'distance_rank', 'projection_fraction', 'arc_length_m',
                 'heading_cosine', 'heading_available', 'previous_step_m',
                 'previous_dt_s', 'previous_speed_mps', 'gap_over_5s',
                 'start_out_degree', 'end_in_degree']


def project_to_segments(point, starts, vectors, squared_lengths):
    fraction = np.divide(np.sum((point - starts) * vectors, axis=1), squared_lengths,
                         out=np.zeros(len(starts)), where=squared_lengths > 0)
    fraction = np.clip(fraction, 0, 1)
    snapped = starts + fraction[:, None] * vectors
    return np.linalg.norm(snapped - point, axis=1), fraction, snapped


def nearest_order(distances, arc_ids, k):
    """Order the closest k arcs, resolving cutoff ties by arc ID."""
    if len(distances) <= k:
        return np.lexsort((arc_ids, distances))
    cutoff = np.partition(distances, k - 1)[k - 1]
    # Distances equal to the cutoff must be considered before breaking ties.
    eligible = np.flatnonzero(distances <= cutoff)
    return eligible[np.lexsort((arc_ids[eligible], distances[eligible]))[:k]]


def generate(inputs, radius=100.0, k=8):
    if not np.isfinite(radius) or radius <= 0 or k < 1:
        raise ValueError('radius must be positive and k must be at least one')
    points, timestamps = inputs['gps_xy'], inputs['timestamps']
    arcs, nodes = inputs['arcs'], inputs['nodes_xy']
    arc_ids = inputs['arc_ids']
    if not len(arcs):
        raise ValueError('No network arcs')
    starts = nodes[arcs[:, 0]]
    vectors = nodes[arcs[:, 1]] - starts
    squared = np.sum(vectors * vectors, axis=1)
    lengths = np.sqrt(squared)
    # Insert each radius-expanded segment bounding box into a uniform grid.
    # Every segment within radius of a point is guaranteed to be in its cell.
    grid = defaultdict(list)
    low = np.floor((np.minimum(starts, starts + vectors) - radius) / radius).astype(int)
    high = np.floor((np.maximum(starts, starts + vectors) + radius) / radius).astype(int)
    for arc, (lo, hi) in enumerate(zip(low, high)):
        for x in range(lo[0], hi[0] + 1):
            for y in range(lo[1], hi[1] + 1):
                grid[x, y].append(arc)
    grid = {cell: np.array(ids, dtype=np.int64) for cell, ids in grid.items()}
    outgoing = np.bincount(arcs[:, 0], minlength=len(nodes))
    incoming = np.bincount(arcs[:, 1], minlength=len(nodes))
    features, selected, snapped_points, offsets, fallbacks = [], [], [], [0], []
    all_arcs = np.arange(len(arcs))
    empty = np.empty(0, dtype=np.int64)
    for i, point in enumerate(points):
        cell = tuple(np.floor(point / radius).astype(int))
        pool = grid.get(cell, empty)
        distances, fractions, snaps = project_to_segments(point, starts[pool], vectors[pool], squared[pool])
        keep = distances <= radius
        fallback = not keep.any()
        if fallback:
            pool = all_arcs
            distances, fractions, snaps = project_to_segments(point, starts, vectors, squared)
        else:
            pool, distances, fractions, snaps = pool[keep], distances[keep], fractions[keep], snaps[keep]
        # Ties are deterministic and preserve directed arc identity.
        order = nearest_order(distances, arc_ids[pool], k)
        ids, ds, fs, ss = pool[order], distances[order], fractions[order], snaps[order]
        dt = float(timestamps[i] - timestamps[i - 1]) if i else 0.0
        movement = point - points[i - 1] if i else np.zeros(2)
        step = float(np.linalg.norm(movement))
        heading_ok = i > 0 and 0 < dt <= 5 and step >= 2
        heading_available = heading_ok & (lengths[ids] > 0)
        cosine = np.zeros(len(ids))
        if heading_ok:
            cosine = np.divide(vectors[ids] @ movement, lengths[ids] * step,
                               out=cosine, where=heading_available)
        matrix = np.column_stack([ds, np.arange(len(ids)), fs, lengths[ids], cosine,
                                  heading_available, np.full(len(ids), step),
                                  np.full(len(ids), dt), np.full(len(ids), step / dt if dt > 0 else 0),
                                  np.full(len(ids), dt > 5), outgoing[arcs[ids, 0]], incoming[arcs[ids, 1]]])
        features.append(matrix)
        selected.extend(arc_ids[ids])
        snapped_points.extend(ss)
        offsets.append(offsets[-1] + len(ids))
        fallbacks.append(fallback)
    result = dict(features=np.concatenate(features).astype(np.float32),
                  candidate_arc_ids=np.array(selected, dtype=np.int64),
                  snapped_xy=np.array(snapped_points), offsets=np.array(offsets, dtype=np.int64),
                  observation_ids=inputs['observation_ids'], fallback_observations=np.array(fallbacks),
                  feature_names=np.array(FEATURE_NAMES))
    if not np.isfinite(result['features']).all():
        raise ValueError('Nonfinite feature value')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=Path('data/processed'))
    parser.add_argument('--output', type=Path, default=Path('data/features'))
    parser.add_argument('--report-dir', type=Path, default=Path('results/features'))
    parser.add_argument('--radius', type=float, default=100)
    parser.add_argument('--k', type=int, default=8)
    args = parser.parse_args()
    rows = []
    for partition in ['train', 'student_test']:
        paths = sorted((args.input_root / partition).glob('*/inputs.npz'))
        if not paths:
            raise ValueError(f'No input trips in {partition}')
        for path in paths:
            started = time.perf_counter()
            with np.load(path, allow_pickle=False) as inputs:
                result = generate(inputs, args.radius, args.k)
            target = args.output / partition / path.parent.name
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target / 'candidates.npz', **result)
            rows.append(dict(partition=partition, trip_id=path.parent.name,
                             observations=len(result['observation_ids']), candidates=len(result['features']),
                             fallback_observations=int(result['fallback_observations'].sum()),
                             seconds=round(time.perf_counter() - started, 3)))
            (target / 'metadata.json').write_text(json.dumps(dict(radius_m=args.radius, k=args.k,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), feature_names=FEATURE_NAMES), indent=2))
            print(f"{partition}/{path.parent.name}: {rows[-1]['candidates']} candidates", flush=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    with (args.report_dir / 'inventory.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(trips=len(rows), observations=sum(r['observations'] for r in rows),
                   candidates=sum(r['candidates'] for r in rows),
                   fallback_observations=sum(r['fallback_observations'] for r in rows),
                   radius_m=args.radius, max_candidates=args.k, feature_names=FEATURE_NAMES)
    (args.report_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
