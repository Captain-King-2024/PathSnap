import importlib.util
import unittest
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('candidates', Path(__file__).resolve().parents[1] / 'src/features/candidates.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class CandidateTests(unittest.TestCase):
    def test_long_segment_reverse_direction_and_fallback(self):
        a = dict(nodes_xy=np.array([[0., 0.], [1000., 0.], [500., 100.]]),
                 arcs=np.array([[0, 1], [1, 0], [2, 2]]), arc_ids=np.array([42, 43, 44]),
                 gps_xy=np.array([[500., 1.], [510., 1.], [500., 10000.]]),
                 timestamps=np.array([0., 1., 20.]), observation_ids=np.array([0, 2, 3]))
        r = m.generate(a, radius=10, k=2)
        np.testing.assert_array_equal(r['candidate_arc_ids'][:2], [42, 43])
        self.assertAlmostEqual(r['features'][0, 0], 1.)
        self.assertGreater(r['features'][2, 4], 0)
        self.assertLess(r['features'][3, 4], 0)
        self.assertTrue(r['fallback_observations'][-1])
        self.assertTrue(np.isfinite(r['features']).all())
        self.assertNotIn('route_arc_ids', r)

    def test_segment_projection_clamps_endpoints(self):
        d, fraction, _ = m.project_to_segments(np.array([2., 0.]), np.array([[0., 0.]]),
                                              np.array([[1., 0.]]), np.array([1.]))
        self.assertEqual(float(d[0]), 1.)
        self.assertEqual(float(fraction[0]), 1.)


if __name__ == '__main__':
    unittest.main()
