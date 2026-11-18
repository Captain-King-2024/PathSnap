"""Compare the logistic classifier and decoder on the student test trips."""
import json

from api.server import Demo, ROOT


def main():
    demo = Demo(ROOT / 'data/processed', ROOT / 'data/labels',
                ROOT / 'models/logistic.npz', 'student_test')
    rows = []
    for trip in demo.trip_ids:
        result = demo.match(trip)
        if result['scores'] is None:
            raise ValueError(f'Missing aligned scores for {trip}')
        rows.append(dict(trip=trip, observations=result['observations'],
                         pieces=result['pieces'], **result['scores']))
        print(trip, result['scores'], flush=True)
    total = sum(row['scored'] for row in rows)
    report = {
        'model': 'logistic',
        'split': 'student_test',
        'scored_observations': total,
        'classifier_top1': sum(round(r['classifier'] * r['scored']) for r in rows) / total,
        'decoded_top1': sum(round(r['decoded'] * r['scored']) for r in rows) / total,
        'trips': rows,
    }
    output = ROOT / 'results/metrics/logistic_decoder.json'
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'trips'}))


if __name__ == '__main__':
    main()
