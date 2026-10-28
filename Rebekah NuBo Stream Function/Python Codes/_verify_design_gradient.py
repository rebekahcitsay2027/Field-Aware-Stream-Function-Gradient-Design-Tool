"""
Temporary verification copy of design_gradient.py's exact logic -- created
only because this sandbox's shell has a stale/cached view of
design_gradient.py itself (confirmed via the Read tool that the real file
is correct; this is purely a test harness working around that). Safe to
delete once you've confirmed design_gradient.py works on your machine.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import stream_gradient as sg
import former_mesh as fm
import wiring

FIELD_CSV = 'magnetData.csv'
STL_PATH = 'Part Studio 2 - Part 1.stl'
AXIS_MAP = (2, 0, 1)
ENCODE = {'LR': 0, 'AP': 1}
TARGET_SCALE = 1e-3
SUBDIVIDE = 3
QUAD_SUBDIV = 2
WIRE_PITCH = 3.0e-3
LAMBDA_REG = 1e-11
INNER_TOWARD = (0.0, 0.0, -0.030)
LEAD_OUT_Y = -0.250


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

    out_csv = name + '_former_wire.csv'
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


def main(names=None):
    names = names or list(ENCODE.keys())
    data = np.genfromtxt(FIELD_CSV, delimiter=',', skip_header=1)
    roi = data[:, :3] * 1e-3
    B0 = data[:, 3:6]
    bhat = B0 / np.linalg.norm(B0, axis=1, keepdims=True)
    print("ROI: %d probed points, raw bounding box (m): x[%+.4f,%+.4f] y[%+.4f,%+.4f] z[%+.4f,%+.4f]"
          % (roi.shape[0], roi[:, 0].min(), roi[:, 0].max(),
             roi[:, 1].min(), roi[:, 1].max(), roi[:, 2].min(), roi[:, 2].max()))
    former = fm.load_former(STL_PATH, axis_map=AXIS_MAP, subdivide=SUBDIVIDE, inner_toward=INNER_TOWARD)
    fm.check_alignment(former, roi)
    R = sg.resistance_matrix(former.verts, former.tris)
    for name in names:
        axis = ENCODE[name]
        wire, metrics = design(name, axis, roi, bhat, former, R)
    print("done.")


if __name__ == '__main__':
    main()
