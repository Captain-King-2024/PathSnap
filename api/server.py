"""Serve the PathSnap demo and its JSON endpoints on one origin."""
import argparse
import json
import pickle
import re
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from src.features.candidates import generate
from src.route_reconstruction.decode import connected_geometry, decode
from src.training.train import scores

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / 'models/boosting_bayesian_20.pkl'


def health(demo):
    return {'status': 'ok', 'service': 'PathSnap', 'split': demo.split,
            'trip_count': len(demo.trip_ids)}


class Demo:
    def __init__(self, data, labels, model, split):
        self.data, self.labels = data / split, labels / split
        self.trip_ids = sorted(p.parent.name for p in self.data.glob('*/inputs.npz')
                               if re.fullmatch(r'[0-9]{8}', p.parent.name))
        if not self.trip_ids:
            raise ValueError(f'No preprocessed trips in {self.data}')
        self.split = split
        if model.suffix == '.npz':
            with np.load(model, allow_pickle=False) as file:
                artifact = dict(file)
            self.model = artifact
            self.model_kind = 'logistic'
            self.model_name = 'Logistic regression'
        else:
            # Load only the trusted, preconfigured local artifact at startup.
            with model.open('rb') as file:
                artifact = pickle.load(file)
            self.model = artifact['model']
            self.model_kind = 'boosting'
            self.model_name = 'Histogram Gradient Boosting'
            if not hasattr(self.model, 'predict_proba'):
                raise ValueError('Artifact must provide candidate probabilities')
        self.feature_names = artifact['feature_names']

    @lru_cache(maxsize=4)
    def match(self, trip):
        if trip not in self.trip_ids:
            raise KeyError(trip)
        with np.load(self.data / trip / 'inputs.npz', allow_pickle=False) as file:
            inputs = dict(file)
        candidates = generate(inputs)
        if not np.array_equal(candidates['feature_names'], self.feature_names):
            raise ValueError('Model feature schema differs from candidate features')
        probability = scores(self.model_kind, self.model, candidates['features'])
        path, breaks = decode(inputs, candidates, probability)
        offsets = candidates['offsets']
        picks = [left + int(probability[left:right].argmax())
                 for left, right in zip(offsets[:-1], offsets[1:])]
        ids = candidates['candidate_arc_ids']
        selected = ids[path]
        classifier_candidates = dict(offsets=np.arange(len(picks) + 1),
                                     candidate_arc_ids=ids[picks],
                                     features=candidates['features'][picks])
        _, classifier_breaks = decode(inputs, classifier_candidates, probability[picks])
        origin = inputs['projection_origin_lonlat']
        metres_per_degree = (6371008.8 * np.pi / 180) * np.array([np.cos(np.radians(origin[1])), 1])
        snapped = candidates['snapped_xy'][path] / metres_per_degree + origin
        selected_geometry = inputs['nodes_lonlat'][inputs['arcs'][selected]]
        review_arcs = np.unique(ids)
        roads = inputs['nodes_lonlat'][inputs['arcs'][review_arcs]]
        result = dict(trip=trip, model=f'{self.model_name} + HMM/Viterbi',
                      observations=len(selected), pieces=len(breaks), breaks=breaks,
                      classifier_pieces=len(classifier_breaks),
                      gps=inputs['gps_lonlat'].tolist(), snapped=snapped.tolist(),
                      roads=roads.tolist(), matched_segments=selected_geometry.tolist(),
                      connected_segments=connected_geometry(inputs, candidates, path, breaks).tolist(),
                      classifier_segments=inputs['nodes_lonlat'][inputs['arcs'][ids[picks]]].tolist(),
                      arc_ids=selected.tolist(), scores=None)
        label_file = self.labels / f'{trip}.npz'
        if self.split == 'student_test' and label_file.is_file():
            with np.load(label_file, allow_pickle=False) as labels:
                if np.array_equal(labels['observation_ids'], inputs['observation_ids']):
                    use = labels['usable']
                    truth = labels['arc_ids'][use]
                    if len(truth):
                        result['scores'] = dict(scored=len(truth),
                            classifier=float(np.mean(ids[picks][use] == truth)),
                            decoded=float(np.mean(selected[use] == truth)))
        return result


def make_handler(demo):
    class Handler(BaseHTTPRequestHandler):
        def send(self, body, content_type, status=200):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def json(self, payload, status=200):
            self.send(json.dumps(payload, allow_nan=False).encode(), 'application/json', status)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/':
                self.send((ROOT / 'ui/index.html').read_bytes(), 'text/html; charset=utf-8')
            elif path == '/api/health':
                self.json(health(demo))
            elif path == '/api/trips':
                self.json(dict(trips=demo.trip_ids, split=demo.split))
            elif path.startswith('/api/match/'):
                try:
                    self.json(demo.match(path.removeprefix('/api/match/')))
                except KeyError:
                    self.json({'error': 'Trip not found'}, 404)
                except (ValueError, OSError):
                    self.log_error('Unable to reconstruct trip %s', path)
                    self.json({'error': 'Unable to reconstruct this trip'}, 500)
            else:
                self.json({'error': 'Endpoint not found'}, 404)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/processed')
    parser.add_argument('--labels', type=Path, default=ROOT / 'data/labels')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--split', choices=['student_test', 'evaluator_test'], default='student_test')
    args = parser.parse_args()
    demo = Demo(args.data, args.labels, args.model, args.split)
    with ThreadingHTTPServer((args.host, args.port), make_handler(demo)) as server:
        print(f'PathSnap listening on http://{args.host}:{args.port}', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
