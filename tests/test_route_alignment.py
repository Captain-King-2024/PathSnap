import unittest

import numpy as np

from src.training.align_routes import align
from src.training.train import metrics


class RouteAlignmentTests(unittest.TestCase):
    def test_repeated_segment_keeps_its_second_route_position(self):
        starts = np.array([[0., 0.], [100., 0.], [0., 0.]])
        ends = np.array([[0., 10.], [100., 10.], [0., 10.]])
        positions, distances = align(np.array([[0., 2.], [100., 4.], [0., 8.]]), starts, ends)
        np.testing.assert_array_equal(positions, [0, 1, 2])
        np.testing.assert_array_equal(distances, [0., 0., 0.])

    def test_candidate_miss_counts_against_top1(self):
        result = metrics(np.array([0.2, 0.8, 0.9, 0.1]),
                         np.array([False, True, False, False]), np.array([2, 2]), 3)
        self.assertEqual(result['top1_accuracy'], 0.5)
        self.assertEqual(result['candidate_miss_rate'], 0.5)
        self.assertEqual(result['top1_when_covered'], 1.0)
        self.assertEqual(result['observations'], 3)
