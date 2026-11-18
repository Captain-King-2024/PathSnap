import unittest

import numpy as np

from src.route_reconstruction.decode import decode


class DecoderTests(unittest.TestCase):
    def test_disconnected_candidate_and_gap(self):
        inputs = {
            'arcs': np.array([[0, 1], [1, 2], [3, 4]]),
            'arc_length_m': np.array([10., 10., 10.]),
            'timestamps': np.array([0., 1., 50.]),
            'gps_xy': np.array([[5., 0.], [15., 0.], [20., 0.]]),
        }
        features = np.zeros((5, 3))
        features[:, 2] = 0.5
        candidates = dict(offsets=np.array([0, 1, 3, 5]),
                          candidate_arc_ids=np.array([0, 2, 1, 2, 1]), features=features)
        path, breaks = decode(inputs, candidates, np.array([.9, .95, .3, .95, .3]))
        np.testing.assert_array_equal(path, [0, 2, 3])
        self.assertEqual(breaks, [0, 2])
