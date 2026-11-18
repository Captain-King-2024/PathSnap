"""Compare the selected classifier and decoder on the student test trips."""
import argparse
import json
from pathlib import Path

from api.server import DEFAULT_MODEL, Demo, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/metrics/hgb_decoder.json')
    args = parser.parse_args()
    demo = Demo(ROOT / 'data/processed', ROOT / 'data/labels',
                args.model, 'student_test')
    rows = []
    for trip in demo.trip_ids:
        result = demo.match(trip)
        if result['scores'] is None:
            raise ValueError(f'Missing aligned scores for {trip}')
        rows.append(dict(trip=trip, observations=result['observations'],
                         pieces=result['pieces'], classifier_pieces=result['classifier_pieces'],
                         **result['scores']))
        print(trip, result['scores'], flush=True)
    total = sum(row['scored'] for row in rows)
    report = {
        'model': args.model.name,
        'split': 'student_test',
        'scored_observations': total,
        'classifier_top1': sum(round(r['classifier'] * r['scored']) for r in rows) / total,
        'decoded_top1': sum(round(r['decoded'] * r['scored']) for r in rows) / total,
        'classifier_pieces': sum(r['classifier_pieces'] for r in rows),
        'decoded_pieces': sum(r['pieces'] for r in rows),
        'trips': rows,
    }
    output = args.output
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'trips'}))


if __name__ == '__main__':
    main()
