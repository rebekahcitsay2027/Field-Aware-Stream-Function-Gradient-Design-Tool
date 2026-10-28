"""
former_mesh.py
===============
Load a coil-former STL into a clean, welded, consistently-oriented triangle
mesh usable by stream_gradient.py.

No coordinate shifting happens here. The only geometric operation besides
mesh cleanup is an OPTIONAL axis permutation (`axis_map`), because a CAD
tool's native part axes are frequently not the same as the physical frame
your B0 probe data is measured in -- that is a real difference between two
coordinate SYSTEMS, not an arbitrary shift, and it has nothing to do with
recentering or offsetting the ROI (which this module never does).

If you export a new STL, don't assume the axis_map below is still right --
call check_alignment() (or just look at the printed bounding boxes) to
verify the former's cavity actually overlaps your probed ROI before trusting
any design built from it.
"""
import numpy as np
from collections import defaultdict
from stl import mesh as stlmesh


class FormerMesh:
    """A welded, consistently-oriented triangle mesh plus an open-boundary mask."""
    def __init__(self, verts, tris, boundary_mask):
        self.verts = verts                   # (V,3) metres
        self.tris = tris                     # (T,3) int
        self.boundary_mask = boundary_mask   # (V,) bool; True = on the open edge (psi=0 there)


def _weld(tri, tol=1e-5):
    """Merge coincident triangle corners into shared vertices.
    tri: (T,3,3) raw STL triangle corners -> (verts, tris)."""
    flat = tri.reshape(-1, 3)
    keys = np.round(flat / tol).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    verts = np.zeros((len(uniq), 3))
    np.add.at(verts, inv, flat)
    verts /= np.bincount(inv)[:, None]
    tris = inv.reshape(-1, 3)
    degenerate = (tris[:, 0] == tris[:, 1]) | (tris[:, 1] == tris[:, 2]) | (tris[:, 0] == tris[:, 2])
    return verts, tris[~degenerate]


def _orient(tris):
    """Flood-fill consistent triangle winding across shared edges (in place on a copy)."""
    adj = defaultdict(list)
    for t, (a, b, c) in enumerate(tris):
        for u, w in [(a, b), (b, c), (c, a)]:
            adj[(min(u, w), max(u, w))].append(t)
    tris = tris.copy()
    seen = np.zeros(len(tris), bool)

    def directed_edges(t):
        return [(tris[t][i], tris[t][(i + 1) % 3]) for i in range(3)]

    for s in range(len(tris)):
        if seen[s]:
            continue
        seen[s] = True
        stack = [s]
        while stack:
            t = stack.pop()
            for (a, b) in directed_edges(t):
                for nb in adj[(min(a, b), max(a, b))]:
                    if nb == t or seen[nb]:
                        continue
                    if (a, b) in directed_edges(nb):
                        tris[nb] = tris[nb][::-1]  # neighbor traversed the shared edge the same way -> flip it
                    seen[nb] = True
                    stack.append(nb)
    return tris


def _subdivide(verts, tris):
    """One round of uniform 4-way triangle subdivision (midpoint refinement)."""
    mid = {}
    nv = list(verts)

    def midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key not in mid:
            mid[key] = len(nv)
            nv.append((verts[a] + verts[b]) / 2)
        return mid[key]

    nt = []
    for a, b, c in tris:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        nt += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
    return np.array(nv), np.array(nt)


def _boundary_mask(verts, tris):
    """True for vertices on an edge used by exactly one triangle (open boundary)."""
    edge_count = defaultdict(int)
    for a, b, c in tris:
        for u, w in [(a, b), (b, c), (c, a)]:
            edge_count[(min(u, w), max(u, w))] += 1
    bnd = np.zeros(len(verts), bool)
    for (u, w), n in edge_count.items():
        if n == 1:
            bnd[u] = bnd[w] = True
    return bnd


def _inner_face(verts, tris, toward):
    """For a closed solid STL, keep only the triangles whose outward normal
    points toward `toward` -- i.e. recover the single copper-side sheet that
    faces the sample cavity, discarding the outer shell."""
    p = verts[tris]
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    centroid = p.mean(1)
    keep = np.einsum('tc,tc->t', nrm, np.asarray(toward) - centroid) > 0
    kept_tris = tris[keep]
    used = np.unique(kept_tris)
    remap = {v: i for i, v in enumerate(used)}
    return verts[used], np.array([[remap[v] for v in t] for t in kept_tris])


def load_former(path, axis_map=(0, 1, 2), subdivide=2, weld_tol=1e-5, inner_toward=None,
                 scale=1.0, verbose=True):
    """Load an STL former into a FormerMesh.

    path         : STL file path.
    axis_map     : permutation of (0,1,2) mapping STL axes -> the same frame
                   your B0 CSV is in, e.g. (2,0,1) means
                   field_x=stl_z, field_y=stl_x, field_z=stl_y.
                   Default (0,1,2) = NO remap. Only change this if you've
                   confirmed (via the printed bounding box, or
                   check_alignment()) that your CAD export's native axes
                   don't already match your probe frame.
    subdivide    : number of uniform mesh-refinement passes (more = finer
                   current-density resolution, slower to solve).
    inner_toward : (x,y,z) of a point inside the sample cavity. If the STL
                   is a closed solid (e.g. exported as a solid part rather
                   than a single sheet), this recovers the inner (copper-side)
                   face automatically. Leave None for an already-open sheet.
    scale        : unit conversion factor applied to every vertex BEFORE
                   welding, e.g. 0.001 if your STL/STEP export is in
                   millimetres and this pipeline needs metres. This is a
                   pure unit conversion, not a coordinate shift -- position
                   0 stays position 0. Get this wrong and check_alignment()
                   will look like it passes (a millimetre-scale former bbox
                   trivially "contains" a metre-scale ROI numerically) while
                   the design itself is meaningless, so verify the printed
                   bounding box is a believable physical size in metres, not
                   just that the ROI-inside-bbox check says True.

    Beyond axis_map and scale (both real, necessary conversions between your
    CAD export's convention and the probe frame/units), nothing here shifts,
    recenters, or otherwise repositions your geometry.
    """
    raw = stlmesh.Mesh.from_file(path).vectors[..., list(axis_map)] * scale
    verts, tris = _weld(raw, weld_tol)

    # Auto-detect a closed solid (every edge shared by exactly 2 triangles)
    # and extract the inner face if so.
    edge_count = defaultdict(int)
    for a, b, c in tris:
        for u, w in [(a, b), (b, c), (c, a)]:
            edge_count[(min(u, w), max(u, w))] += 1
    is_closed = all(n == 2 for n in edge_count.values())
    if is_closed:
        target = inner_toward if inner_toward is not None else (0.0, 0.0, 0.0)
        verts, tris = _inner_face(verts, tris, target)
        if verbose:
            print(f"former_mesh: STL was a closed solid -- extracted inner face toward {target}")

    tris = _orient(tris)
    for _ in range(subdivide):
        verts, tris = _subdivide(verts, tris)
    bnd = _boundary_mask(verts, tris)

    if verbose:
        bbox_min, bbox_max = verts.min(0), verts.max(0)
        print(f"former_mesh: loaded {path}")
        print(f"  axis_map={axis_map}, subdivide={subdivide}")
        print(f"  {len(verts)} verts ({(~bnd).sum()} free, {bnd.sum()} boundary), {len(tris)} tris")
        print(f"  bounding box (m): min={np.round(bbox_min,4)} max={np.round(bbox_max,4)}")

    return FormerMesh(verts, tris, bnd)


def check_alignment(former, roi_points, label=""):
    """Print former vs. ROI bounding boxes side by side so you can eyeball
    whether axis_map (and the STL's units) are actually correct -- e.g. the
    ROI should sit INSIDE the former's cavity, not off to one side or
    outside its extent entirely."""
    fmin, fmax = former.verts.min(0), former.verts.max(0)
    rmin, rmax = roi_points.min(0), roi_points.max(0)
    print(f"check_alignment {label}:")
    print(f"  former bbox (m): x[{fmin[0]:+.4f},{fmax[0]:+.4f}] y[{fmin[1]:+.4f},{fmax[1]:+.4f}] z[{fmin[2]:+.4f},{fmax[2]:+.4f}]")
    print(f"  ROI    bbox (m): x[{rmin[0]:+.4f},{rmax[0]:+.4f}] y[{rmin[1]:+.4f},{rmax[1]:+.4f}] z[{rmin[2]:+.4f},{rmax[2]:+.4f}]")
    inside = np.all((rmin >= fmin) & (rmax <= fmax))
    print(f"  ROI fully inside former bbox: {bool(inside)}", "-- looks sane" if inside else "-- CHECK axis_map/units!")
    # A naive "inside" check passes trivially if the former is 1000x too big
    # (e.g. a millimetre-scale former numerically "contains" a metre-scale
    # ROI) -- so also flag that specific failure mode explicitly.
    former_size = fmax - fmin
    roi_size = rmax - rmin
    ratio = former_size / np.maximum(roi_size, 1e-12)
    if np.any(ratio > 50) or np.any(ratio < 1):
        print(f"  size ratio (former/ROI) per axis: {np.round(ratio,1)} "
              "-- if this is ~1000x or ~0.001x, you likely have an mm/m unit mismatch (see `scale`).")
