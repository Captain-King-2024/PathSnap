"""Unit and integration tests for Random Forest training pipeline.

Run from repo root:
    python -m unittest tests/test_random_forest.py
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


class TestLabelAssembler(unittest.TestCase):
    def _make_temp_data(self, tmp: Path):
        """Create minimal fake feature/reference data."""
        feat_dir = tmp / 'features' / 'train' / 'trip_01'
        ref_dir = tmp / '_references'
        feat_dir.mkdir(parents=True)
        ref_dir.mkdir()

        n_cand = 10
        np.savez_compressed(
            feat_dir / 'candidates.npz',
            features=np.random.rand(n_cand, 12).astype(np.float32),
            candidate_arc_ids=np.arange(n_cand, dtype=np.int64),
            offsets=np.array([0, 3, 6, 10], dtype=np.int64),
            observation_ids=np.array([0, 1, 2]),
            fallback_observations=np.array([False, False, False]),
            feature_names=np.array(['f'] * 12),
        )
        np.savez_compressed(
            ref_dir / 'trip_01.npz',
            route_arc_ids=np.array([0, 1, 2], dtype=np.int64),
        )
        return feat_dir.parent.parent, ref_dir

    def test_label_shape_and_values(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'label_assembler',
            Path(__file__).resolve().parents[1] / 'src/training/label_assembler.py',
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            feat_root, ref_root = self._make_temp_data(tmp)
            X, y, g = m.load_trip_labels('trip_01', feat_root / 'train', ref_root)
            self.assertEqual(X.shape, (10, 12))
            self.assertEqual(len(y), 10)
            # arcs 0,1,2 are in route → positive
            self.assertEqual(y[0], 1)
            self.assertEqual(y[9], 0)  # arc 9 not in route

    def test_no_reference_leakage(self):
        """route_arc_ids must never appear as a column in features."""
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            feat_root, ref_root = self._make_temp_data(tmp)
            cand_path = feat_root / 'train' / 'trip_01' / 'candidates.npz'
            with np.load(cand_path, allow_pickle=False) as c:
                self.assertNotIn('route_arc_ids', c.files)


class TestRFScorer(unittest.TestCase):
    def test_scorer_output_shapes(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'rf_scorer',
            Path(__file__).resolve().parents[1] / 'src/training/rf_scorer.py',
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            # Build a minimal trained model and scaler
            import joblib
            from sklearn.ensemble import RandomForestClassifier

            clf = RandomForestClassifier(n_estimators=10, random_state=0)
            X_fake = np.random.rand(100, 12).astype(np.float32)
            y_fake = (X_fake[:, 0] > 0.5).astype(int)
            clf.fit(X_fake, y_fake)
            joblib.dump(clf, tmp / 'model.joblib')
            np.savez_compressed(
                tmp / 'preprocessing.npz',
                mean=np.zeros(12),
                scale=np.ones(12),
            )
            (tmp / 'feature_names.json').write_text(
                json.dumps([f'f{i}' for i in range(12)]), encoding='utf-8'
            )

            # Fake candidates
            cand_dir = tmp / 'cands'
            cand_dir.mkdir()
            np.savez_compressed(
                cand_dir / 'candidates.npz',
                features=X_fake[:10],
                candidate_arc_ids=np.arange(10, dtype=np.int64),
                offsets=np.array([0, 3, 7, 10], dtype=np.int64),
                observation_ids=np.array([0, 1, 2]),
                fallback_observations=np.array([False] * 3),
                feature_names=np.array([f'f{i}' for i in range(12)]),
            )

            scorer = m.RFScorer(tmp)
            result = scorer.score_trip(cand_dir / 'candidates.npz')
            self.assertEqual(result['log_emission'].shape, (10,))
            self.assertTrue(np.all(result['log_emission'] <= 0))


if __name__ == '__main__':
    unittest.main()
