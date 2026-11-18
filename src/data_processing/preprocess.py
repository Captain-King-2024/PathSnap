"""Validate PathSnap trips and export label-separated NumPy artifacts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_table(path, columns):
    a = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if a.size == 0 or a.shape[1] != columns:
        raise ValueError(f'{path.name}: expected nonempty {columns}-column table')
    return a


def valid_coordinates(a):
    return np.isfinite(a).all(axis=1) & (np.abs(a[:, 0]) <= 180) & (np.abs(a[:, 1]) <= 90)


def project(coords, origin):
    """Local equirectangular coordinates in metres, suitable for local trips."""
    delta = np.radians(coords - origin)
    return delta * np.array([6371008.8 * np.cos(np.radians(origin[1])), 6371008.8])


def components(n, arcs):
    parent = list(range(n))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for u, v in arcs:
        parent[root(int(u))] = root(int(v))
    return len({root(i) for i in range(n)})


def read_trip_inputs(trip):
    ident = trip.name
    paths = {ext: trip / f'{ident}.{ext}' for ext in ('track', 'nodes', 'arcs')}
    track = read_table(paths['track'], 3)
    nodes = read_table(paths['nodes'], 2)
    raw_arcs = read_table(paths['arcs'], 2)
    if not valid_coordinates(nodes).all():
        raise ValueError('Invalid network coordinates; refusing to renumber nodes')
    if not np.isfinite(raw_arcs).all() or not np.equal(raw_arcs, np.floor(raw_arcs)).all():
        raise ValueError('Arc endpoints must be finite integers')
    if (raw_arcs < 0).any() or (raw_arcs >= len(nodes)).any():
        raise ValueError('Arc endpoint outside node range')
    arcs = raw_arcs.astype(np.int64)
    mask = valid_coordinates(track[:, :2]) & np.isfinite(track[:, 2])
    clean = track[mask]
    if not len(clean):
        raise ValueError('No valid GPS observations')
    if (np.diff(clean[:, 2]) < 0).any():
        raise ValueError('Timestamps decrease; refusing to reorder the trajectory silently')
    origin = np.median(nodes, axis=0)
    gps_xy, nodes_xy = project(clean[:, :2], origin), project(nodes, origin)
    lengths = np.linalg.norm(nodes_xy[arcs[:, 1]] - nodes_xy[arcs[:, 0]], axis=1)
    return dict(gps_lonlat=clean[:, :2], timestamps=clean[:, 2],
                observation_ids=np.flatnonzero(mask), gps_xy=gps_xy, nodes_lonlat=nodes,
                nodes_xy=nodes_xy, arcs=arcs, arc_ids=np.arange(len(arcs)),
                arc_length_m=lengths, projection_origin_lonlat=origin), len(track)


def process_trip(trip, destination):
    ident = trip.name
    paths = {ext: trip / f'{ident}.{ext}' for ext in ('track', 'nodes', 'arcs', 'route')}
    inputs, raw_count = read_trip_inputs(trip)
    nodes, arcs, nodes_xy, lengths = (inputs[key] for key in
                                     ('nodes_lonlat', 'arcs', 'nodes_xy', 'arc_length_m'))
    dt = np.diff(inputs['timestamps'])
    distance = np.linalg.norm(np.diff(inputs['gps_xy'], axis=0), axis=1)
    speed = np.divide(distance, dt, out=np.full(len(dt), np.nan), where=dt > 0)
    raw_route = read_table(paths['route'], 1).ravel()
    if not np.isfinite(raw_route).all() or not np.equal(raw_route, np.floor(raw_route)).all():
        raise ValueError('Route arc IDs must be finite integers')
    if (raw_route < 0).any() or (raw_route >= len(arcs)).any():
        raise ValueError('Reference arc ID outside arc range')
    route = raw_route.astype(np.int64)
    row = dict(trip_id=ident, raw_observations=raw_count, observations=len(inputs['timestamps']),
               removed_observations=raw_count - len(inputs['timestamps']), nodes=len(nodes), arcs=len(arcs),
               route_arcs=len(route), duplicate_timestamps=int((dt == 0).sum()),
               gaps_over_1s=int((dt > 1.000001).sum()), max_gap_s=float(dt.max()) if len(dt) else 0,
               zero_length_arcs=int((lengths == 0).sum()), weak_components=components(len(nodes), arcs),
               isolated_nodes=int(len(nodes) - len(np.unique(arcs))),
               reference_disconnections=int((arcs[route[:-1], 1] != arcs[route[1:], 0]).sum()),
               max_speed_mps=float(np.nanmax(speed)) if np.isfinite(speed).any() else None,
               extent_m=float(np.linalg.norm(np.ptp(nodes_xy, axis=0))))
    destination.mkdir(parents=True, exist_ok=True)
    # Inference assets deliberately contain no route/reference values.
    np.savez_compressed(destination / 'inputs.npz', **inputs)
    labels = destination.parent / '_references'
    labels.mkdir(exist_ok=True)
    np.savez_compressed(labels / f'{ident}.npz', route_arc_ids=route)
    metadata = dict(summary=row, source_sha256={k: hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
                    projection='local equirectangular; R=6371008.8m; origin=median network lon/lat',
                    policy='Preserve node/arc row IDs and observation order. Remove invalid GPS rows only; retain duplicate timestamps and gaps. Route is a sequence, not pointwise labels.')
    (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data/processed'))
    parser.add_argument('--report-dir', type=Path, default=Path('results/preprocessing'))
    args = parser.parse_args()
    rows, failures, seen = [], [], set()
    for partition, expected in [('train', 80), ('student_test', 10)]:
        trips = sorted(p for p in (args.data_root / partition).iterdir() if p.is_dir())
        if len(trips) != expected:
            raise ValueError(f'{partition}: expected {expected} trips, found {len(trips)}')
        for trip in trips:
            if trip.name in seen:
                raise ValueError(f'Trip overlaps partitions: {trip.name}')
            seen.add(trip.name)
            try:
                row = process_trip(trip, args.output / partition / trip.name)
                rows.append(dict(partition=partition, **row))
            except (ValueError, OSError) as exc:
                failures.append(dict(partition=partition, trip_id=trip.name, error=str(exc)))
    args.report_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (args.report_dir / 'inventory.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(successful_trips=len(rows), failed_trips=failures,
                   observations=sum(r['observations'] for r in rows),
                   removed_observations=sum(r['removed_observations'] for r in rows),
                   duplicate_timestamps=sum(r['duplicate_timestamps'] for r in rows),
                   gaps_over_1s=sum(r['gaps_over_1s'] for r in rows))
    (args.report_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
