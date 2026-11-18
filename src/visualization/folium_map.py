"""Export a layered map of one student-test trip for the technical report."""
import argparse
from pathlib import Path

import numpy as np

from api.server import DEFAULT_MODEL, ROOT, Demo


def main():
    import folium

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trip', default='00000000')
    parser.add_argument('--output', type=Path, default=Path('results/maps/student_test_map.html'))
    args = parser.parse_args()
    demo = Demo(ROOT / 'data/processed', ROOT / 'data/labels', DEFAULT_MODEL, 'student_test')
    result = demo.match(args.trip)
    gps = [point[::-1] for point in result['gps']]
    map_view = folium.Map(location=gps[0], tiles='OpenStreetMap')
    map_view.fit_bounds(gps)

    def road_layer(name, segments, color, weight):
        layer = folium.FeatureGroup(name=name).add_to(map_view)
        geometry = {'type': 'MultiLineString', 'coordinates': np.asarray(segments).tolist()}
        folium.GeoJson(geometry, style_function=lambda _: {'color': color, 'weight': weight}).add_to(layer)

    road_layer('Candidate road network (gray)', result['roads'], '#64748b', 1)
    reference = demo.data / '_references' / f'{args.trip}.npz'
    if reference.is_file():
        with np.load(reference, allow_pickle=False) as route, np.load(
                demo.data / args.trip / 'inputs.npz', allow_pickle=False) as inputs:
            segments = inputs['nodes_lonlat'][inputs['arcs'][route['route_arc_ids']]]
            road_layer('Reference route (green)', segments, '#15803d', 5)
    road_layer('HGB + Viterbi match (purple)', result['connected_segments'], '#7e22ce', 3)
    gps_layer = folium.FeatureGroup(name='Original GPS (blue)').add_to(map_view)
    for point in gps:
        folium.CircleMarker(point, radius=2, color='#2563eb', fill=True, weight=1).add_to(gps_layer)
    folium.LayerControl(collapsed=False).add_to(map_view)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    map_view.save(str(args.output))
    print(args.output)


if __name__ == '__main__':
    main()
