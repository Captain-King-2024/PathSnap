"""Load a fitted RF model and produce per-candidate emission probabilities.

Returns a dict compatible with the HMM emission layer:
  {
    'log_emission': float32 array (n_candidates,) — log P(arc is correct | features)
    'candidate_arc_ids': int64 array (n_candidates,)
    'offsets': int64 array (n_obs + 1,)
  }

Usage:
    from rf_scorer import RFScorer
    scorer = RFScorer('models/random_forest')
    result = scorer.score_trip('data/features/student_test/trip_01/candidates.npz')
"""
import json
from pathlib import Path
import joblib
import numpy as np


class RFScorer:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.clf = joblib.load(model_dir / 'model.joblib')
        scaler_data = np.load(model_dir / 'preprocessing.npz')
        self.mean_ = scaler_data['mean']
        self.scale_ = scaler_data['scale']
        self.feature_names = json.loads(
            (model_dir / 'feature_names.json').read_text(encoding='utf-8')
        )

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def score_trip(self, candidates_path: str | Path) -> dict:
        candidates_path = Path(candidates_path)
        with np.load(candidates_path, allow_pickle=False) as c:
            features = c['features'].astype(np.float32)
            candidate_arc_ids = c['candidate_arc_ids']
            offsets = c['offsets']

        X_s = self._scale(features)
        proba = self.clf.predict_proba(X_s)  # (n_cand, 2)
        p_pos = proba[:, 1].astype(np.float32)  # P(label=1)
        # Clip for numerical safety before log
        p_pos = np.clip(p_pos, 1e-7, 1.0)
        log_emission = np.log(p_pos)

        return dict(
            log_emission=log_emission,
            candidate_arc_ids=candidate_arc_ids,
            offsets=offsets,
        )
