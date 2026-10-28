"""
wiring.py
=========
Turn a set of independent contour loops (from stream_gradient.contour_levels)
into a single continuous wire path, by bridging nearest points between loops
in minimum-spanning-tree order. This is a purely geometric/graph operation --
no field physics and no coordinate remapping happens here.
"""
import numpy as np
from collections import defaultdict


def loops_from_segments(segments, tol=1e-7):
    """Chain an unordered list of (2,3) segments into closed loops by matching
    endpoint coordinates (within `tol`). Returns a list of (N,3) point arrays,
    each a closed polyline. Drops any chain shorter than 4 points (not a
    meaningful loop -- usually a contour level that barely clips a mesh
    boundary)."""
    if not segments:
        return []

    def key(point):
        return tuple(np.round(point / tol).astype(np.int64))

    start_index = defaultdict(list)
    for i, seg in enumerate(segments):
        start_index[key(seg[0])].append(i)

    used = [False] * len(segments)
    loops = []
    for i0 in range(len(segments)):
        if used[i0]:
            continue
        chain = [segments[i0][0], segments[i0][1]]
        used[i0] = True
        current = segments[i0][1]
        while True:
            candidates = [j for j in start_index[key(current)] if not used[j]]
            if not candidates:
                break
            j = candidates[0]
            used[j] = True
            chain.append(segments[j][1])
            current = segments[j][1]
            if key(current) == key(chain[0]):
                break
        if len(chain) >= 4:
            loops.append(np.array(chain))
    return loops


def _nearest_pair(loop_a, loop_b, n_sample=60):
    """Cheap approximate nearest-point pair between two loops: subsample both,
    find the closest sampled pair, then map back to the closest ACTUAL points
    on each loop near that pair. Returns (index_in_a, index_in_b, distance)."""
    sa = loop_a[::max(1, len(loop_a) // n_sample)]
    sb = loop_b[::max(1, len(loop_b) // n_sample)]
    dist = np.linalg.norm(sa[:, None, :] - sb[None, :, :], axis=2)
    d_min = dist.min()
    ii, jj = np.where(dist <= d_min * 1.6 + 1e-4)
    # among near-tied candidates, prefer the one with smallest y (arbitrary
    # but deterministic tie-break, matches original behavior)
    choice = np.argmin(sa[ii, 1] + sb[jj, 1])
    i, j = ii[choice], jj[choice]
    idx_a = int(np.argmin(np.linalg.norm(loop_a - sa[i], axis=1)))
    idx_b = int(np.argmin(np.linalg.norm(loop_b - sb[j], axis=1)))
    return idx_a, idx_b, dist[i, j]


def build_bridge_tree(loops):
    """Minimum-spanning-tree over loops using nearest_pair distances as edge
    weights (greedy Prim's algorithm). Returns an adjacency dict:
    node -> [(neighbor, point_index_in_node, point_index_in_neighbor), ...]."""
    in_tree = {0}
    edges = []
    while len(in_tree) < len(loops):
        best = None
        for a in in_tree:
            for b in range(len(loops)):
                if b in in_tree:
                    continue
                ia, jb, dist = _nearest_pair(loops[a], loops[b])
                if best is None or dist < best[0]:
                    best = (dist, a, b, ia, jb)
        _, a, b, ia, jb = best
        in_tree.add(b)
        edges.append((a, b, ia, jb))
    neighbors = defaultdict(list)
    for a, b, ia, jb in edges:
        neighbors[a].append((b, ia, jb))
        neighbors[b].append((a, jb, ia))
    return neighbors


def traverse_to_single_wire(loops, neighbors, root, entry_point):
    """Depth-first walk of the bridge tree starting at `root`, entering it
    near `entry_point`, splicing every child loop in at its bridge point.
    Returns a single (N,3) continuous path visiting every loop exactly once."""
    out = []
    visited = set()

    def recurse(node, enter_near):
        visited.add(node)
        loop = loops[node]
        k = int(np.argmin(np.linalg.norm(loop - enter_near, axis=1)))
        rotated = np.vstack([loop[k:], loop[:k]])  # start the loop at its entry point

        insertions = []
        for (child, idx_on_node, idx_on_child) in neighbors[node]:
            if child in visited:
                continue
            point_on_node = loops[node][idx_on_node]
            point_on_child = loops[child][idx_on_child]
            insert_at = int(np.argmin(np.linalg.norm(rotated - point_on_node, axis=1)))
            insertions.append((insert_at, child, point_on_child, point_on_node))
        insertions.sort(key=lambda t: t[0])

        last = 0
        for insert_at, child, point_on_child, point_on_node in insertions:
            if child in visited:
                continue
            out.extend(rotated[last:insert_at + 1])
            out.append(point_on_child)
            recurse(child, point_on_child)
            out.append(point_on_node)
            last = insert_at + 1
        out.extend(rotated[last:])
        out.append(rotated[0])

    recurse(root, entry_point)
    return np.array(out)


def stitch_loops(loops, exit_near_y):
    """Convenience wrapper: build the bridge tree and traverse it, starting
    from whichever loop comes closest to y = exit_near_y (so the final path's
    start/end sit near your chosen terminal/lead-out location)."""
    neighbors = build_bridge_tree(loops)
    root = int(np.argmin([np.min(np.abs(loop[:, 1] - exit_near_y)) for loop in loops]))
    entry_idx = int(np.argmin(np.abs(loops[root][:, 1] - exit_near_y)))
    return traverse_to_single_wire(loops, neighbors, root, loops[root][entry_idx])
