import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('preprocess', Path(__file__).resolve().parents[1] / 'src/data_processing/preprocess.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PreprocessingTests(unittest.TestCase):
    def test_preserves_ids_and_separates_reference(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            trip = root / 'trip'
            trip.mkdir()
            (trip / 'trip.nodes').write_text('8 49\n8.001 49\n8.002 49\n')
            (trip / 'trip.arcs').write_text('0 1\n1 2\n')
            (trip / 'trip.route').write_text('0\n1\n')
            (trip / 'trip.track').write_text('8 49 1\nnan 49 2\n8.001 49 3\n8.002 49 3\n')
            output = root / 'processed' / 'trip'
            result = module.process_trip(trip, output)
            self.assertEqual(result['removed_observations'], 1)
            self.assertEqual(result['duplicate_timestamps'], 1)
            with np.load(output / 'inputs.npz') as data:
                np.testing.assert_array_equal(data['observation_ids'], [0, 2, 3])
                np.testing.assert_array_equal(data['arcs'], [[0, 1], [1, 2]])
                self.assertNotIn('route_arc_ids', data.files)
            (trip / 'trip.arcs').write_text('0 8\n')
            with self.assertRaisesRegex(ValueError, 'outside node range'):
                module.process_trip(trip, root / 'bad')

    def test_projection_and_components(self):
        xy = module.project(np.array([[0., 0.], [.001, 0.]]), np.array([0., 0.]))
        self.assertAlmostEqual(xy[1, 0], 111.195, places=2)
        self.assertEqual(module.components(4, np.array([[0, 1], [1, 2]])), 2)


if __name__ == '__main__':
    unittest.main()
