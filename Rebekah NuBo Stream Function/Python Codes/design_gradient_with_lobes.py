"""
design_gradient.py  (with per-lobe manufacturable winding export)
=================================================================
Same pipeline as before, plus: after each gradient is solved, it also writes
import-ready per-lobe winding CSVs via lobe_wiring (smooth, on-surface,
self-intersection-free -> opFitSpline safe). See lobe_wiring.py.

NOTE: this is a drop-in copy. WIRE_PITCH / LAMBDA_REG are set to 2.8 mm / 1e-10
(your last-stated values) -- confirm they match your working file before
replacing it. The ONLY functional change vs. your file is the 5-line
"per-lobe" block inside design() (marked below).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import stream_gradient as sg
import former_mesh as fm
import wiring

_DATA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIELD_CSV = os.path.join(_DATA, 'magnetData.csv')
STL_PATH = os.path.join(_DATA, 'remeshed.stl')   # isotropic-remeshed former, in METRES
AXIS_MAP = (0, 1, 2)
SCALE = 1.0
ENCODE = {'LR': 0, 'AP': 1}

TARGET_SCALE = 1e-3
SUBDIVIDE = 0
QUAD_SUBDIV = 2
WIRE_PITCH = 2.8e-3
LAMBDA_REG = 1e-10
INNER_TOWARD = (0.0, 0.0, -0.050)
LEAD_OUT_Y = -0.240


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

    # --- per-lobe manufacturable windings (import-ready, self-intersection-free) ---
    try:
        import lobe_wiring
        _paths = lobe_wiring.lobe_paths(former, psi, n_turns, wire_pitch=wire_pitch, roi_center=roi.mean(0))
        for _f, _n, _x in lobe_wiring.write_lobe_csvs(_paths, name, _DATA):
            print("  lobe -> %s (%d pts, %d self-crossings)" % (os.path.basename(_f), _n, _x))
    except Exception as _e:
        print("  [lobe_wiring skipped: %s]" % _e)
    # -----------------------------------------------------------------------------

    levels = np.linspace(psi.min() + psi_range / (2 * n_turns),
                          psi.max() - psi_range / (2 * n_turns), n_turns)

    contours = sg.contour_levels(former.verts, former.tris, psi, levels)
    loops = []
    for _, segs in contours:
        loops.extend(wiring.loops_from_segments(segs))
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
    ax1.set_title(name + ' 3D'); ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')

    def roi_box(ax, i, j):
        lo = roi.min(0); hi = roi.max(0)
        xs = [lo[i], hi[i], hi[i], lo[i], lo[i]]
        ys = [lo[j], lo[j], hi[j], hi[j], lo[j]]
        ax.plot(xs, ys, color='tab:red', linewidth=1)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(wire[:, 0] * 1000, wire[:, 2] * 1000, color='tab:blue', linewidth=0.7)
    roi_box(ax2, 0, 2); ax2.set_title(name + ' x-z'); ax2.set_xlabel('x (mm)'); ax2.set_ylabel('z (mm)'); ax2.axis('equal')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(wire[:, 1] * 1000, wire[:, 2] * 1000, color='tab:blue', linewidth=0.7)
    roi_box(ax3, 1, 2); ax3.set_title(name + ' y-z'); ax3.set_xlabel('y (mm)'); ax3.set_ylabel('z (mm)'); ax3.axis('equal')

    plt.tight_layout()
    out_png = os.path.join(_DATA, name + '_check.png')
    plt.savefig(out_png, dpi=140); plt.close(fig)
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
