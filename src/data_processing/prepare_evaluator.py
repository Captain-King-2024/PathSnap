"""Prepare the ten evaluator trips from raw track, node and arc files."""
import argparse
import re
from pathlib import Path

import numpy as np

from src.data_processing.preprocess import read_trip_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/processed/evaluator_test'))
    args = parser.parse_args()
    trips = sorted(p for p in args.data_root.iterdir()
                   if p.is_dir() and re.fullmatch(r'[0-9]{8}', p.name))
    if len(trips) != 10:
        parser.error(f'Expected ten numbered evaluator trips, found {len(trips)}')
    existing_ids = {p.parent.name for p in args.output.glob('*/inputs.npz')}
    if existing_ids - {trip.name for trip in trips}:
        parser.error('Output contains other trips. Choose an empty output directory.')
    for trip in trips:
        inputs, _ = read_trip_inputs(trip)
        destination = args.output / trip.name
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / 'inputs.npz', **inputs)
        print(f'{trip.name}: prepared {len(inputs["timestamps"])} observations')


if __name__ == '__main__':
    main()
