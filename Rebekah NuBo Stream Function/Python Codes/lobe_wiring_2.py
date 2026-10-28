"""
lobe_wiring.py
==============
Convert a solved stream function into MANUFACTURABLE, per-lobe winding paths
that import cleanly into a CAD "Path from Points" spline (e.g. Onshape
opFitSpline).

WHY THIS EXISTS
---------------
A stream-function gradient coil has several current lobes (local extrema of the
stream function psi). If every turn and every lobe is stitched into ONE
continuous wire (wiring.stitch_loops), the wire must cross over itself to hop
between lobes and between nested turns. A self-intersecting curve is rejected by
opFitSpline ("Operation created invalid geometry"), and the hops also create
sharp cusps (near-180 deg reversals) that make the fitted spline overshoot.

Within a SINGLE lobe the turns are a plain concentric spiral that never crosses
itself. So we emit one clean path per lobe. The lobe-to-lobe connections become
physical insulated jumpers made at winding time (standard for a single-layer
coil). Driving all lobes in series -- preserving each spiral's native winding
direction -- reproduces the field of the continuous design exactly.

Each exported path is:
  * on the former surface (marching-triangle contours; the seam where each turn
    opens is placed at the rim-ward extreme so crossovers stay off the ROI),
  * smooth  (per-turn periodic-spline resample),
  * cusp-free (corner relaxation; opFitSpline overshoots above ~150 deg),
  * self-intersection-free (checked; any residual crossing is nudged apart along
    the local binormal -- a tiny out-of-plane jumper lift).

PUBLIC API
----------
    paths = lobe_paths(former, psi, n_turns, wire_pitch=3e-3, roi_center=...)
        former : object with .verts (V,3 m), .tris (T,3 int), .boundary_mask
        psi    : (V,) solved stream function (stream_gradient.solve_stream)
        -> list of (N,3) float arrays in METRES, one clean spiral per lobe.
    files = write_lobe_csvs(paths, name, out_dir)   # name_lobe1.csv ...
    n     = count_self_crossings(P)                  # QA; 0 == import-safe

Requires: numpy, scipy.
"""
import os
import numpy as np
from collections import defaultdict
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

import stream_gradient as sg
from wiring import loops_from_segments


# --------------------------------------------------------------------------- #
# geometry / QA helpers
# --------------------------------------------------------------------------- #
def _seg_seg_dist(p1, p2, p3, p4):
    d1 = p2 - p1; d2 = p4 - p3; r = p1 - p3
    a = d1 @ d1; e = d2 @ d2; f = d2 @ r
    if a < 1e-18 and e < 1e-18:
        return np.linalg.norm(r)
    if a < 1e-18:
        s = 0.0; t = np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e < 1e-18:
            t = 0.0; s = np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2; den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-18 else 0.0
            t = (b * s + f) / e
            if t < 0:   t = 0.0; s = np.clip(-c / a, 0, 1)
            elif t > 1: t = 1.0; s = np.clip((b - c) / a, 0, 1)
    return np.linalg.norm((p1 + s * d1) - (p3 + t * d2))


def count_self_crossings(P, tol=5e-5, near=0.02):
    """Number of non-adjacent segment pairs that intersect (dist < tol). 0 = safe."""
    A = (P[:-1] + P[1:]) / 2.0
    tree = cKDTree(A); n = 0
    for i, j in tree.query_pairs(near):
        if abs(i - j) <= 2:
            continue
        if _seg_seg_dist(P[i], P[i + 1], P[j], P[j + 1]) < tol:
            n += 1
    return n


def _angles(P):
    v = np.diff(P, axis=0); nn = np.linalg.norm(v, axis=1); a = np.zeros(len(P))
    g = (nn[:-1] > 1e-12) & (nn[1:] > 1e-12)
    cs = np.einsum('ij,ij->i', v[:-1], v[1:])
    cs = np.where(g, cs / np.where(g, nn[:-1] * nn[1:], 1), 1)
    a[1:-1] = np.degrees(np.arccos(np.clip(cs, -1, 1)))
    return a


def _dp(P, tol):
    """Douglas-Peucker keep-mask (fewest points within `tol` of the polyline)."""
    n = len(P); keep = np.zeros(n, bool); keep[0] = keep[-1] = True; st = [(0, n - 1)]
    while st:
        i, j = st.pop()
        if j <= i + 1:
            continue
        a = P[i]; ab = P[j] - a; L2 = ab @ ab; seg = P[i + 1:j]
        if L2 < 1e-18:
            dv = np.linalg.norm(seg - a, axis=1)
        else:
            t = np.clip(((seg - a) @ ab) / L2, 0, 1); dv = np.linalg.norm(seg - (a + t[:, None] * ab), axis=1)
        k = int(np.argmax(dv))
        if dv[k] > tol:
            idx = i + 1 + k; keep[idx] = True; st += [(i, idx), (idx, j)]
    return keep


def _lap_deburr(P, thresh=115.0, iters=400):
    """Relax any corner sharper than `thresh` deg (opFitSpline overshoots on cusps).
    Moves points (Jacobi smoothing at offending vertices); adds no short segments."""
    P = P.copy()
    for _ in range(iters):
        a = _angles(P); idx = np.where(a > thresh)[0]
        if len(idx) == 0:
            break
        Q = P.copy()
        for i in idx:
            if 1 <= i < len(P) - 1:
                Q[i] = 0.5 * P[i] + 0.25 * (P[i - 1] + P[i + 1])
        P = Q
    return P


def _uncross(P, sep=0.8e-3, iters=60):
    """Nudge any self-intersecting segment pair apart along their common normal
    (a small out-of-plane jumper lift), until the path is crossing-free."""
    P = P.copy()
    for _ in range(iters):
        A = (P[:-1] + P[1:]) / 2.0; tree = cKDTree(A); fixed = 0
        for i, j in tree.query_pairs(0.02):
            if abs(i - j) <= 2:
                continue
            d1 = P[i + 1] - P[i]; d2 = P[j + 1] - P[j]; r = P[i] - P[j]
            a = d1 @ d1; e = d2 @ d2
            if a < 1e-18 or e < 1e-18:
                continue
            f = d2 @ r; c = d1 @ r; b = d1 @ d2; den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-18 else 0.0
            t = np.clip((b * s + f) / e, 0, 1); s = np.clip((b * t - c) / a, 0, 1)
            cp = (P[i] + s * d1) - (P[j] + t * d2); dist = np.linalg.norm(cp)
            if dist < sep:
                nrm = np.cross(d1, d2)
                if np.linalg.norm(nrm) < 1e-12:
                    nrm = cp
                if np.linalg.norm(nrm) < 1e-12:
                    continue
                nrm = nrm / np.linalg.norm(nrm); push = (sep - dist) / 2 + 2e-4
                P[j] += nrm * push; P[j + 1] += nrm * push
                P[i] -= nrm * push; P[i + 1] -= nrm * push
                fixed += 1
        if fixed == 0:
            break
    return P


def _smooth_loop(p, step=3e-3):
    """Periodic cubic-spline resample of one closed turn -> smooth, ~uniform points."""
    P = np.vstack([p, p[0]])
    try:
        tck, _ = splprep([P[:, 0], P[:, 1], P[:, 2]], s=len(P) * 1e-8, per=1, k=3)
        n = max(int(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)) / step), 12)
        x, y, z = splev(np.linspace(0, 1, n, endpoint=False), tck)
        return np.column_stack([x, y, z])
    except Exception:
        return p


# --------------------------------------------------------------------------- #
# lobe decomposition
# --------------------------------------------------------------------------- #
def _contour_loops(verts, tris, psi, n_turns):
    pr = psi.max() - psi.min()
    levels = np.linspace(psi.min() + pr / (2 * n_turns), psi.max() - pr / (2 * n_turns), n_turns)
    loops = []
    for _, segs in sg.contour_levels(verts, tris, psi, levels):
        for lp in loops_from_segments(segs):
            p = np.asarray(lp)
            if len(p) > 1 and np.allclose(p[0], p[-1]):
                p = p[:-1]
            if len(p) < 8:
                continue
            per = np.sum(np.linalg.norm(np.diff(np.vstack([p, p[0]]), axis=0), axis=1))
            loops.append(dict(pts=p, c=p.mean(0), per=per))
    return loops


def _lobe_seeds(verts, tris, psi, frac=0.30):
    """Lobe centres = dominant local extrema of psi on the mesh."""
    nbr = defaultdict(set)
    for a, b, c in tris:
        nbr[a] |= {b, c}; nbr[b] |= {a, c}; nbr[c] |= {a, b}
    amax = np.abs(psi).max(); seeds = []
    for v in range(len(verts)):
        nb = list(nbr[v])
        if not nb:
            continue
        if (psi[v] >= psi[nb]).all() and abs(psi[v]) > frac * amax:
            seeds.append(verts[v])
        elif (psi[v] <= psi[nb]).all() and abs(psi[v]) > frac * amax:
            seeds.append(verts[v])
    return np.array(seeds)


def lobe_paths(former, psi, n_turns, wire_pitch=3e-3, roi_center=None,
               decim_tol=0.4e-3, seed_frac=0.30):
    """Per-lobe, smooth, crossing-free winding paths (list of (N,3) arrays, metres)."""
    verts, tris = former.verts, former.tris
    loops = _contour_loops(verts, tris, psi, n_turns)
    if not loops:
        return []
    seeds = _lobe_seeds(verts, tris, psi, seed_frac)
    if len(seeds) == 0:
        seeds = np.array([max(loops, key=lambda d: d['per'])['c']])
    if roi_center is None:
        roi_center = verts.mean(0)
    for l in loops:
        l['lobe'] = int(np.argmin(np.linalg.norm(seeds - l['c'], axis=1)))
    groups = defaultdict(list)
    for l in loops:
        groups[l['lobe']].append(l)

    out = []
    for lb, lst in sorted(groups.items()):
        lst = sorted(lst, key=lambda d: -d['per'])           # outer -> inner
        outward = seeds[lb] - roi_center; nrm = np.linalg.norm(outward)
        outward = outward / nrm if nrm > 1e-9 else np.array([0.0, 0.0, 1.0])
        chunks = []
        for l in lst:
            p = _smooth_loop(l['pts'], step=wire_pitch)
            k = int(np.argmax(p @ outward))                  # open turn at rim-ward extreme
            chunks.append(np.roll(p, -k, axis=0))
        P = np.vstack(chunks)
        P = P[_dp(P, decim_tol)]                              # sparse enough for opFitSpline
        P = _lap_deburr(P)                                   # remove cusps
        P = _uncross(P)                                      # remove any self-crossing
        out.append(P)
    return out


def write_lobe_csvs(paths, name, out_dir):
    """Write name_lobe{k}.csv (metres, x,y,z). Returns [(path, npts, crossings), ...]."""
    info = []
    for k, P in enumerate(paths, 1):
        fn = os.path.join(out_dir, '%s_lobe%d.csv' % (name, k))
        np.savetxt(fn, P, header='x_m,y_m,z_m', delimiter=',', comments='', fmt='%.6f')
        info.append((fn, len(P), count_self_crossings(P)))
    return info
