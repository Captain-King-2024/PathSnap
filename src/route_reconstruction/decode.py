"""Decode directed road candidates with distance-based Viterbi transitions."""
import heapq

import numpy as np


def bounded_routes(graph, source, targets, limit):
    queue = [(0.0, int(source))]
    distances = {int(source): 0.0}
    found = {}
    while queue and len(found) < len(targets):
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if node in targets:
            found[node] = distance
        for end, length in graph.get(node, ()):
            total = distance + length
            if total <= limit and total < distances.get(end, np.inf):
                distances[end] = total
                heapq.heappush(queue, (total, end))
    return found


def decode(inputs, candidates, probabilities, max_gap_s=15.0):
    """Return one candidate row per observation and boundaries between route pieces."""
    offsets = candidates['offsets']
    rows = candidates['candidate_arc_ids']
    arcs = inputs['arcs']
    lengths = inputs['arc_length_m']
    fraction = candidates['features'][:, 2]
    graph = {}
    for (start, end), length in zip(arcs, lengths):
        graph.setdefault(int(start), []).append((int(end), float(length)))
    logp = np.log(np.clip(probabilities, 1e-8, 1.0))
    back = np.full(len(rows), -1, dtype=np.int64)
    path = np.empty(len(offsets) - 1, dtype=np.int64)
    boundaries = [0]
    previous = logp[:offsets[1]]

    def close_piece(end, scores):
        best = offsets[end] + int(scores.argmax())
        for i in range(end, boundaries[-1] - 1, -1):
            path[i] = best
            best = back[best]

    for i in range(1, len(path)):
        left, middle, right = offsets[i - 1:i + 2]
        dt = inputs['timestamps'][i] - inputs['timestamps'][i - 1]
        step = np.linalg.norm(inputs['gps_xy'][i] - inputs['gps_xy'][i - 1])
        current = np.full(right - middle, -np.inf)
        if 0 < dt <= max_gap_s:
            limit = float(min(1200, max(80, step + 100, dt * 55)))
            tolerance = max(20, 0.4 * step)
            targets = set(arcs[rows[middle:right], 0].tolist())
            cache = {}
            for j in range(left, middle):
                if not np.isfinite(previous[j - left]):
                    continue
                old = rows[j]
                end = int(arcs[old, 1])
                if end not in cache:
                    cache[end] = bounded_routes(graph, end, targets, limit)
                for k in range(middle, right):
                    new = rows[k]
                    road = ((1 - fraction[j]) * lengths[old]
                            + cache[end].get(int(arcs[new, 0]), np.inf)
                            + fraction[k] * lengths[new])
                    if old == new and fraction[k] >= fraction[j]:
                        road = min(road, (fraction[k] - fraction[j]) * lengths[old])
                    if road <= limit:
                        score = previous[j - left] - abs(road - step) / tolerance
                        if score > current[k - middle]:
                            current[k - middle] = score
                            back[k] = j
        if np.isfinite(current).any():
            previous = current + logp[middle:right]
            previous -= np.max(previous)
        else:
            close_piece(i - 1, previous)
            boundaries.append(i)
            previous = logp[middle:right]
    close_piece(len(path) - 1, previous)
    return path, boundaries
