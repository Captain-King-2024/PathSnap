"""Check projection, network diagnostics, saved features and candidate recall."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

if __package__:
    from .candidates import generate, nearest_order, project_to_segments
else:
    from candidates import generate, nearest_order, project_to_segments


def spherical_distance(start, end):
    a, b = np.radians(start), np.radians(end)
    lon, lat = (b - a).T
    h = np.sin(lat / 2) ** 2 + np.cos(a[:, 1]) * np.cos(b[:, 1]) * np.sin(lon / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def diagnostics(inputs):
    arcs = inputs['arcs']
    n = len(inputs['nodes_xy'])
    keys = arcs[:, 0] * n + arcs[:, 1]
    reverse = arcs[:, 1] * n + arcs[:, 0]
    incoming = np.bincount(arcs[:, 1], minlength=n)
    outgoing = np.bincount(arcs[:, 0], minlength=n)
    coords = inputs['nodes_lonlat']
    gps = inputs['gps_lonlat']
    times = np.diff(inputs['timestamps'])
    gps_dist = np.linalg.norm(np.diff(inputs['gps_xy'], axis=0), axis=1)
    road_error = np.abs(inputs['arc_length_m'] - spherical_distance(coords[arcs[:, 0]], coords[arcs[:, 1]]))
    gps_error = np.abs(gps_dist - spherical_distance(gps[:-1], gps[1:]))
    return dict(sources=int(((incoming == 0) & (outgoing > 0)).sum()),
                sinks=int(((outgoing == 0) & (incoming > 0)).sum()),
                reciprocal_arcs=int(np.count_nonzero((keys != reverse) & np.isin(reverse, keys))),
                duplicate_directed_arcs=int(len(keys) - len(np.unique(keys))),
                max_in_degree=int(incoming.max()), max_out_degree=int(outgoing.max()),
                road_length_max_error_m=float(road_error.max(initial=0)),
                road_length_p95_error_m=float(np.quantile(road_error, 0.95)),
                gps_step_max_error_m=float(gps_error.max(initial=0)),
                gps_step_p95_error_m=float(np.quantile(gps_error, 0.95)) if len(gps_error) else 0,
                nonpositive_dt=int((times <= 0).sum()), gaps_over_5s=int((times > 5).sum()),
                max_gap_s=float(times.max(initial=0)))


def audit_search(inputs, saved, radius, k, rng, full_max_observations=0):
    points, arcs, nodes, ids = (inputs[key] for key in ('gps_xy', 'arcs', 'nodes_xy', 'arc_ids'))
    starts = nodes[arcs[:, 0]]
    vectors = nodes[arcs[:, 1]] - starts
    squared = np.sum(vectors ** 2, axis=1)
    fallback = saved['fallback_observations']
    exhaustive = len(points) <= full_max_observations
    sample = (np.arange(len(points)) if exhaustive else
              np.union1d(rng.choice(len(points), min(24, len(points)), replace=False), np.flatnonzero(fallback)))
    projection_s = full_sort_s = partial_sort_s = 0.0
    for i in sample:
        tick = perf_counter()
        distance, _, _ = project_to_segments(points[i], starts, vectors, squared)
        projection_elapsed = perf_counter() - tick
        pool = np.flatnonzero(distance <= radius)
        if bool(fallback[i]) != (len(pool) == 0):
            raise ValueError(f'Wrong fallback flag at observation {i}')
        if fallback[i]:
            projection_s += projection_elapsed
            tick = perf_counter()
            expected = np.lexsort((ids, distance))[:k]
            full_sort_s += perf_counter() - tick
            tick = perf_counter()
            actual = nearest_order(distance, ids, k)
            partial_sort_s += perf_counter() - tick
            np.testing.assert_array_equal(actual, expected)
        else:
            expected = pool[np.lexsort((ids[pool], distance[pool]))[:k]]
        left, right = saved['offsets'][i:i + 2]
        np.testing.assert_array_equal(saved['candidate_arc_ids'][left:right], ids[expected])
    return dict(search_checks=len(sample), exhaustive_trip=int(exhaustive), fallback_count=int(fallback.sum()),
                fallback_projection_seconds=projection_s, fallback_full_sort_seconds=full_sort_s,
                fallback_partial_sort_seconds=partial_sort_s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path('data/processed'))
    parser.add_argument('--features', type=Path, default=Path('data/features'))
    parser.add_argument('--labels', type=Path, default=Path('data/labels'))
    parser.add_argument('--output', type=Path, default=Path('results/feature_audit'))
    parser.add_argument('--full-max-observations', type=int, default=0,
                        help='Exhaustively check trips up to this size (0 samples every trip)')
    args = parser.parse_args()
    if args.full_max_observations < 0:
        parser.error('--full-max-observations must be nonnegative')
    rows = []
    rng = np.random.default_rng(557)
    for split, expected in [('train', 80), ('student_test', 10)]:
        paths = sorted((args.inputs / split).glob('*/inputs.npz'))
        if len(paths) != expected:
            parser.error(f'{split}: expected {expected} trips, found {len(paths)}')
        for path in paths:
            trip = path.parent.name
            folder = args.features / split / trip
            meta = json.loads((folder / 'metadata.json').read_text())
            with np.load(path) as file:
                inputs = dict(file)
            with np.load(folder / 'candidates.npz') as file:
                saved = dict(file)
            with np.load(args.labels / split / f'{trip}.npz') as file:
                labels = dict(file)
            radius, k = meta['radius_m'], meta['k']
            tick = perf_counter()
            regenerated = generate(inputs, radius, k)
            generation_s = perf_counter() - tick
            for key, value in regenerated.items():
                np.testing.assert_array_equal(saved[key], value, err_msg=f'{split}/{trip}: {key}')
            np.testing.assert_array_equal(saved['observation_ids'], labels['observation_ids'])
            offsets = saved['offsets']
            matches = saved['candidate_arc_ids'] == np.repeat(labels['arc_ids'], np.diff(offsets))
            covered = np.logical_or.reduceat(matches, offsets[:-1])
            usable = labels['usable']
            rows.append(dict(split=split, trip=trip, observations=len(usable),
                             aligned=int(usable.sum()), covered=int((covered & usable).sum()),
                             candidate_rows=len(saved['features']), generation_seconds=generation_s,
                             **diagnostics(inputs),
                             **audit_search(inputs, saved, radius, k, rng, args.full_max_observations)))
        print(f'{split}: audited {len(paths)} trips', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'trips.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    summed = ['observations', 'aligned', 'covered', 'candidate_rows', 'search_checks', 'fallback_count',
              'exhaustive_trip', 'generation_seconds', 'fallback_projection_seconds', 'fallback_full_sort_seconds',
              'fallback_partial_sort_seconds', 'nonpositive_dt', 'gaps_over_5s']
    for split in ('train', 'student_test'):
        group = [r for r in rows if r['split'] == split]
        stats = {key: sum(r[key] for r in group) for key in summed}
        stats.update({key: max(r[key] for r in group) for key in
                      ('max_gap_s', 'road_length_max_error_m', 'gps_step_max_error_m',
                       'road_length_p95_error_m', 'gps_step_p95_error_m')})
        stats['candidate_recall_aligned'] = stats['covered'] / stats['aligned']
        summary[split] = stats
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
