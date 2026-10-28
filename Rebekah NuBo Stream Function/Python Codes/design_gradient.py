"""
design_gradient.py
===================
Full stream-function gradient design on a real STL former, targeting the
LOCAL B0 direction (not a fixed Cartesian component), for an inhomogeneous
magnet.

REWRITTEN FROM SCRATCH -- what changed from the previous version
(run_former.py, now deprecated -- kept only because this folder wouldn't
let me delete it; use this file instead):
  * No coordinate shifting/recentering of the probe CSV. The ROI is used
    exactly as measured (mm -> m conversion only). The old version did
    `roi = roi - roi.mean(0) + ROI_CENTER_FIELD`, silently recentering and
    offsetting the probed cloud to an arbitrary hardcoded point; that's
    gone. If your former's cavity isn't where you expect relative to the
    ROI, that's real information -- see check_alignment()'s printout below,
    rather than something to paper over with a coordinate shift.
  * The STL-to-probe-frame axis_map is now an explicit, documented,
    top-of-file parameter (not buried in a function call), and its
    correctness is verified by printing both bounding boxes every run
    (former_mesh.check_alignment). Change AXIS_MAP and re-run if you export
    a new STL with different native axes -- don't assume the old value.
  * Loop-stitching logic moved to wiring.py; core physics moved to
    stream_gradient.py; STL loading moved to former_mesh.py. Same
    algorithms as before (validated: R^2=0.9956 on the LR design before
    this rewrite), reorganized for clarity, not reworked out of caution.

Pipeline: STL former -> local-B0-direction sensitivity -> regularized
stream-function solve -> wire-pitch-limited turn count -> single continuous
wire -> CSV + validation plot.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import stream_gradient as sg
import former_mesh as fm
import wiring
import os

# Resolve data/output files relative to THIS file's location, so the script
# runs correctly no matter which directory you launch it from. The data
# (magnetData.csv, remeshed.stl) and all outputs live in the parent folder
# ('Stream Function'), one level up from this 'python helper codes' folder.
_DATA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIELD_CSV = os.path.join(_DATA, 'magnetData.csv')

# Former mesh: 'remeshed.stl' -- the raw_data_former surface after an
# ISOTROPIC remesh (done in Blender) to a ~7 mm uniform target edge length.
# This replaced the old raw_data_former.stl, whose STEP->STL export left the
# flat walls as a handful of huge triangles (~90% of the surface area) and the
# fillets as extreme high-aspect slivers -- a 47x edge-length / 2200x area
# range that made the stream-function basis ill-conditioned on exactly the
# surfaces the coil wraps. The remesh fixed that: ~20.8k near-equilateral
# triangles, size range 2.4x, aspect median 1.18 / max 2.6, zero slivers.
#
# UNITS: this STL is exported in METRES (bbox ~+/-0.2 m), so SCALE=1.0 (NOT
# the 0.001 the old millimetre export needed). Get this wrong and the former
# collapses to mm-scale and check_alignment() fails loudly.
#
# SUBDIVIDE=0: the mesh is already fine and uniform. The old in-code 4-way
# subdivision only scales every triangle down while PRESERVING the size ratio
# (it cannot fix non-uniformity), so running it here would just quadruple the
# triangle count and solve time for no benefit. Leave it at 0.
STL_PATH = os.path.join(_DATA, 'remeshed.stl')
AXIS_MAP = (0, 1, 2)
SCALE = 1.0
ENCODE = {'LR': 0, 'AP': 1}

# Arbitrary overall scale applied to the stream-function target (roi[:,axis]).
# NOT a coordinate shift -- it doesn't touch any position -- it's just a
# knob on how "big" the solved psi comes out, which sets how many amps end
# up flowing per turn. The resulting efficiency (mT/m/A) is mathematically
# invariant to this (gradient and required current scale together), but the
# ABSOLUTE turn_current/gradient numbers reported aren't, so this exists
# purely to keep turn_current in a physically sane range (a few amps)
# instead of an arbitrary/confusing value. Verified empirically: efficiency
# stayed at 0.5049 mT/m/A whether TARGET_SCALE was 1 or 1e-3; only
# turn_current changed (1870A vs 1.87A).
TARGET_SCALE = 1e-3

SUBDIVIDE = 0
QUAD_SUBDIV = 2
WIRE_PITCH = 2.85e-3
LAMBDA_REG = 2e-10
INNER_TOWARD = (0.0, 0.0, -0.050)  # only used if the STL turns out to be a closed solid; this former isn't
LEAD_OUT_Y = -0.240

# --- Loop post-processing: smooth each contour and shift it to its final depth
# inside the 12.5 mm U-former wall. U cross-section (mm) is from finished.step.
# LAYER_DEPTH_MM is the centre-depth of each gradient's wire, measured inward
# from the OUTER wall (LR sits shallow, AP deep so its inner edge is flush with
# the inner surface). Set LOOP_SMOOTH_SFAC = 0 to disable smoothing.
_XC, _ZC = 165.0, -76.3
_X_OUT, _Z_OUT, _R_OUT = 200.5, -111.8, 35.5
LAYER_DEPTH_MM = {'LR': 5.0, 'AP': 11.25}   # centre-depth inward from the outer wall
LOOP_SMOOTH_SFAC = 1.0                       # loop-smoothing strength (~RMS mm); 0 = off
MIN_SPACING_MM = WIRE_PITCH * 1000.0         # enforce >= this centre-to-centre spacing after
                                             # smoothing (so smoothing can't violate the wire
                                             # pitch); set 0 to disable the spacing pass
SPACING_MARGIN_MM = 0.08                     # aim this much above MIN_SPACING (absorbs CAD spline fit)
LOOP_STEP_MM = 1.5                           # resample spacing of the smoothed loops (finer -> the
                                             # curve-to-curve spacing check matches the CAD spline)


def _wall_region(x, z):
    ax = abs(x)
    if z >= _ZC and ax >= _XC: return 'wall'
    if ax <= _XC and z <= _ZC: return 'bottom'
    if ax > _XC and z < _ZC:  return 'corner'
    return 'wall'


def _layer_point(x, z, d):
    """Move point (x, z) in mm onto the constant-depth-d offset of the U wall."""
    import math
    r = _wall_region(x, z)
    if r == 'wall':   return math.copysign(_X_OUT - d, x), z
    if r == 'bottom': return x, _Z_OUT + d
    cx = math.copysign(_XC, x); dx, dz = x - cx, z - _ZC
    n = math.hypot(dx, dz) or 1.0; rr = _R_OUT - d
    return cx + dx / n * rr, _ZC + dz / n * rr


def smooth_and_layer_loop(loop_m, depth_mm, sfac=None):
    """Smooth a closed loop (N,3 metres) with a periodic spline and shift it to
    `depth_mm` centre-depth inside the U wall. Returns (M,3) metres on the target
    surface (depth exact). If depth_mm is None, only smooths. sfac defaults to
    LOOP_SMOOTH_SFAC (0 disables smoothing)."""
    from scipy.interpolate import splprep, splev
    if sfac is None:
        sfac = LOOP_SMOOTH_SFAC
    pts = np.asarray(loop_m) * 1000.0
    # drop consecutive + closing duplicate points (else the periodic fit fails)
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-3:
            keep.append(i)
    pts = pts[keep]
    if len(pts) > 2 and np.linalg.norm(pts[0] - pts[-1]) < 1e-3:
        pts = pts[:-1]
    if sfac > 0 and len(pts) >= 8:
        try:
            per = float(np.sum(np.linalg.norm(np.diff(np.vstack([pts, pts[0]]), axis=0), axis=1)))
            n = max(int(per / LOOP_STEP_MM), 40)   # resample fine so spacing check == the CAD curve
            tck, _ = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], s=len(pts) * sfac, per=1, k=3)
            xx, yy, zz = splev(np.linspace(0, 1, n, endpoint=False), tck)
            pts = np.column_stack([xx, yy, zz])
        except Exception:
            pass                                  # fall back to raw points on fit failure
    if depth_mm is not None:
        for i in range(len(pts)):
            nx, nz = _layer_point(pts[i, 0], pts[i, 2], depth_mm)
            pts[i, 0] = nx; pts[i, 2] = nz
    return pts / 1000.0


def enforce_min_spacing(loops_m, depth_mm, pitch_mm, iters=120, relax=0.9, skip_adjacent=5):
    """Keep loops >= pitch_mm apart CURVE-to-curve after smoothing -- both between
    DIFFERENT loops AND between a loop's own NON-ADJACENT parts (e.g. the two arms
    of a figure-8). Only the actually-violating points are refined each pass (fast).
    Pushes each violating point off the nearest offending SEGMENT, then re-projects
    to the wall depth. target = pitch + SPACING_MARGIN_MM absorbs the CAD spline fit;
    points within skip_adjacent along the same loop (the wire's own neighbourhood)
    are never pushed apart.

    NOTE: the design pitch has ~zero spacing margin, so this restores spacing to
    ~pitch but can't exceed it where turns are tightest. If you need a strict
    > pitch everywhere, raise WIRE_PITCH (source margin) or lower LOOP_SMOOTH_SFAC."""
    from scipy.spatial import cKDTree
    target = pitch_mm + SPACING_MARGIN_MM
    loops = [np.asarray(L) * 1000.0 for L in loops_m]
    lens = [len(L) for L in loops]
    gid = np.concatenate([[i] * n for i, n in enumerate(lens)])
    gloc = np.concatenate([np.arange(n) for n in lens])

    def _closest(p, a, b):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-18), 0.0, 1.0)
        return a + t * ab

    for _ in range(iters):
        allp = np.vstack(loops)
        pairs = cKDTree(allp).query_pairs(target, output_type='ndarray')
        if len(pairs) == 0:
            break
        same = gid[pairs[:, 0]] == gid[pairs[:, 1]]
        cd = np.abs(gloc[pairs[:, 0]] - gloc[pairs[:, 1]])
        nof = np.array([lens[x] for x in gid[pairs[:, 0]]])
        cd = np.minimum(cd, nof - cd)
        viol = np.unique(pairs[(~same) | (cd > skip_adjacent)])   # cross-loop + non-adjacent same-loop
        if len(viol) == 0:
            break
        segA = allp
        segB = np.vstack([np.roll(L, -1, axis=0) for L in loops])
        stree = cKDTree((segA + segB) / 2.0)
        for a in viol:
            p = allp[a]; push = np.zeros(3); hit = 0
            for si in stree.query_ball_point(p, target + 1.5):
                if gid[si] == gid[a]:
                    c2 = abs(gloc[si] - gloc[a]); c2 = min(c2, lens[gid[a]] - c2)
                    if c2 <= skip_adjacent:
                        continue
                c = _closest(p, segA[si], segB[si]); v = p - c; dd = np.linalg.norm(v)
                if 1e-9 < dd < target:
                    push += (v / dd) * (target - dd); hit += 1
            if hit:
                li = gid[a]; loc = gloc[a]
                q = loops[li][loc] + relax * push / hit
                if depth_mm is not None:
                    nx, nz = _layer_point(q[0], q[2], depth_mm); q = np.array([nx, q[1], nz])
                loops[li][loc] = q
    return [L / 1000.0 for L in loops]


def _fit_slope(field_values, axis_values):
    design_matrix = np.column_stack([axis_values, np.ones_like(axis_values)])
    coeffs, _, _, _ = np.linalg.lstsq(design_matrix, field_values, rcond=None)
    residual = field_values - design_matrix @ coeffs
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((field_values - field_values.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot
    return coeffs[0], r_squared


def design(name, axis, roi, bhat, former, R, lam=LAMBDA_REG, wire_pitch=WIRE_PITCH):
    C = sg.sensitivity_matrix(former.verts, former.tris, roi, quad_subdiv=QUAD_SUBDIV)
    P = bhat[:, 0:1] * C[0] + bhat[:, 1:2] * C[1] + bhat[:, 2:3] * C[2]

    psi = sg.solve_stream(P, TARGET_SCALE * roi[:, axis], R, former.boundary_mask, lam)
    psi_range = psi.max() - psi.min()
    peak_current_density = sg.max_surface_current(former.verts, former.tris, psi)
    n_turns = max(int(psi_range / (wire_pitch * peak_current_density)), 1)
    turn_current = psi_range / n_turns

    import lobe_wiring
    _paths = lobe_wiring.lobe_paths(former, psi, n_turns, wire_pitch=wire_pitch, roi_center=roi.mean(0))
    lobe_wiring.write_lobe_csvs(_paths, name, _DATA)

    levels = np.linspace(psi.min() + psi_range / (2 * n_turns),
                          psi.max() - psi_range / (2 * n_turns), n_turns)

    contours = sg.contour_levels(former.verts, former.tris, psi, levels)
    loops = []
    for _, segs in contours:
        loops.extend(wiring.loops_from_segments(segs))

    # Smooth each contour and shift it to its final depth in the U wall, then write:
    #   <name>_loops/<name>_loop###.csv   : one file per closed loop
    #   <name>_loops_layered_smooth.csv   : ALL loops in one file, separated by a
    #        BLANK ROW -> import once, one spline per loop (blank-row-delimited
    #        Path-from-Points FS). Depth = LAYER_DEPTH_MM[name], smoothing =
    #        LOOP_SMOOTH_SFAC (both set at the top of this file).
    depth = LAYER_DEPTH_MM.get(name)
    loop_dir = os.path.join(_DATA, name + '_loops')
    os.makedirs(loop_dir, exist_ok=True)
    # smooth + shift every loop, THEN enforce the minimum wire spacing so the
    # smoothing can't pull adjacent turns closer than the pitch.
    proc = [smooth_and_layer_loop(np.asarray(_lp), depth) for _lp in loops]
    if MIN_SPACING_MM > 0 and len(proc) > 1:
        proc = enforce_min_spacing(proc, depth, MIN_SPACING_MM)
    combined = ['x_m,y_m,z_m']
    for _k, _p in enumerate(proc, 1):
        _q = np.vstack([_p, _p[0]])                                    # explicitly close it
        np.savetxt(os.path.join(loop_dir, '%s_loop%03d.csv' % (name, _k)),
                   _q, header='x_m,y_m,z_m', delimiter=',', comments='', fmt='%.6f')
        combined += ['%.6f,%.6f,%.6f' % (r[0], r[1], r[2]) for r in _q]
        combined.append('')                                            # blank row = loop separator
    with open(os.path.join(_DATA, name + '_loops_layered_smooth.csv'), 'w') as _fh:
        _fh.write('\n'.join(combined) + '\n')
    _dtxt = ('layered %.2f mm' % depth) if depth is not None else 'not layered'
    print("   wrote %d loops (smoothed, %s, min spacing >= %.2f mm) -> %s_loops/  and  %s_loops_layered_smooth.csv"
          % (len(proc), _dtxt, MIN_SPACING_MM, name, name))

    wire = wiring.stitch_loops(loops, exit_near_y=LEAD_OUT_Y)

    lead_a = wire[0].copy()
    lead_a[1] = LEAD_OUT_Y - 0.02
    lead_b = wire[-1].copy()
    lead_b[1] = LEAD_OUT_Y - 0.02
    lead_b[0] = lead_b[0] + 0.006
    wire = np.vstack([lead_a, wire, lead_b])

    B_wire = sg.biot_savart_wire([(np.stack([wire[:-1], wire[1:]], axis=1), turn_current)], roi)
    projected = np.einsum('pc,pc->p', B_wire, bhat)
    slope, r_squared = _fit_slope(projected, roi[:, axis])
    wire_length = np.sum(np.linalg.norm(np.diff(wire, axis=0), axis=1))

    out_csv = os.path.join(_DATA, name + '_former_wire.csv')
    np.savetxt(out_csv,
               np.column_stack([np.arange(len(wire)), wire * 1000]),
               header='point_index,x_mm,y_mm,z_mm', delimiter=',', comments='',
               fmt=['%d', '%.3f', '%.3f', '%.3f'])

    efficiency = slope / turn_current
    metrics = dict(n_turns=n_turns, turn_current=turn_current, wire_length=wire_length,
                   gradient_per_current=slope, r_squared=r_squared, efficiency=efficiency)
    print("%s: axis=%d, %d turns @ %.4f A, %.1f m wire, gradient=%.3f mT/m (at %.3f A) -> efficiency=%.4f mT/m/A, R^2=%.4f -> %s"
          % (name, axis, n_turns, turn_current, wire_length, slope * 1e3, turn_current, efficiency * 1e3, r_squared, out_csv))
    return wire, metrics


def plot_check(name, axis, wire, roi, former):
    fig = plt.figure(figsize=(13, 4.2))

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot_trisurf(former.verts[:, 0], former.verts[:, 1], former.verts[:, 2],
                      triangles=former.tris, color=(0.6, 0.6, 0.6, 0.15), edgecolor='none')
    ax1.plot(wire[:, 0], wire[:, 1], wire[:, 2], color='tab:green', linewidth=1)
    ax1.scatter(roi[:, 0], roi[:, 1], roi[:, 2], color='tab:red', s=2)
    ax1.set_title(name + ' 3D')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    def roi_box(ax, i, j):
        lo = roi.min(0)
        hi = roi.max(0)
        xs = [lo[i], hi[i], hi[i], lo[i], lo[i]]
        ys = [lo[j], lo[j], hi[j], hi[j], lo[j]]
        ax.plot(xs, ys, color='tab:red', linewidth=1)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(wire[:, 0] * 1000, wire[:, 2] * 1000, color='tab:blue', linewidth=0.7)
    roi_box(ax2, 0, 2)
    ax2.set_title(name + ' x-z')
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('z (mm)')
    ax2.axis('equal')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(wire[:, 1] * 1000, wire[:, 2] * 1000, color='tab:blue', linewidth=0.7)
    roi_box(ax3, 1, 2)
    ax3.set_title(name + ' y-z')
    ax3.set_xlabel('y (mm)')
    ax3.set_ylabel('z (mm)')
    ax3.axis('equal')

    plt.tight_layout()
    out_png = os.path.join(_DATA, name + '_check.png')
    plt.savefig(out_png, dpi=140)
    plt.close(fig)
    print("  saved " + out_png)


def main(names=None):
    names = names or list(ENCODE.keys())

    data = np.genfromtxt(FIELD_CSV, delimiter=',', skip_header=1)
    roi = data[:, :3] * 1e-3
    B0 = data[:, 3:6]
    bhat = B0 / np.linalg.norm(B0, axis=1, keepdims=True)
    print("ROI: %d probed points, raw bounding box (m): x[%+.4f,%+.4f] y[%+.4f,%+.4f] z[%+.4f,%+.4f]"
          % (roi.shape[0], roi[:, 0].min(), roi[:, 0].max(),
             roi[:, 1].min(), roi[:, 1].max(), roi[:, 2].min(), roi[:, 2].max()))

    former = fm.load_former(STL_PATH, axis_map=AXIS_MAP, subdivide=SUBDIVIDE, inner_toward=INNER_TOWARD, scale=SCALE)
    fm.check_alignment(former, roi)

    R = sg.resistance_matrix(former.verts, former.tris)

    for name in names:
        axis = ENCODE[name]
        wire, metrics = design(name, axis, roi, bhat, former, R)
        plot_check(name, axis, wire, roi, former)

    print("done.")


if __name__ == '__main__':
    main()
