"""
make_lobe_windings.py
=====================
Regenerate MANUFACTURABLE, per-lobe winding CSVs for every gradient in
design_gradient.ENCODE, using your CURRENT design settings (whatever STL,
SCALE, WIRE_PITCH, LAMBDA_REG, etc. are set in design_gradient.py).

Run:
    python make_lobe_windings.py

Writes  <NAME>_lobe1.csv, <NAME>_lobe2.csv, ...  (metres; columns x,y,z) next to
your other outputs -- one clean, smooth, self-intersection-free spiral per lobe,
ready to import as separate "Path from Points" splines. See lobe_wiring.py for
why per-lobe (a single continuous wire self-crosses -> opFitSpline "invalid
geometry"). Connect the lobes in series at winding time, preserving each
spiral's direction, to reproduce the field of the continuous design.

This does NOT modify design_gradient.py; it reuses its constants and helpers.
"""
import numpy as np
import design_gradient as dg
import stream_gradient as sg
import former_mesh as fm
import lobe_wiring

# Optional: use the vectorized ops if present (much faster on big meshes).
try:
    import fast_ops
    sg.sensitivity_matrix = fast_ops.sensitivity_matrix
    sg.resistance_matrix = fast_ops.resistance_matrix
except Exception:
    pass


def main(names=None):
    names = names or list(dg.ENCODE.keys())

    data = np.genfromtxt(dg.FIELD_CSV, delimiter=',', skip_header=1)
    roi = data[:, :3] * 1e-3
    B0 = data[:, 3:6]
    bhat = B0 / np.linalg.norm(B0, axis=1, keepdims=True)

    former = fm.load_former(dg.STL_PATH, axis_map=dg.AXIS_MAP, subdivide=dg.SUBDIVIDE,
                            inner_toward=dg.INNER_TOWARD, scale=dg.SCALE)
    R = sg.resistance_matrix(former.verts, former.tris)

    for name in names:
        axis = dg.ENCODE[name]
        C = sg.sensitivity_matrix(former.verts, former.tris, roi, quad_subdiv=dg.QUAD_SUBDIV)
        P = bhat[:, 0:1] * C[0] + bhat[:, 1:2] * C[1] + bhat[:, 2:3] * C[2]
        psi = sg.solve_stream(P, dg.TARGET_SCALE * roi[:, axis], R, former.boundary_mask, dg.LAMBDA_REG)

        psi_range = psi.max() - psi.min()
        peak = sg.max_surface_current(former.verts, former.tris, psi)
        n_turns = max(int(psi_range / (dg.WIRE_PITCH * peak)), 1)

        paths = lobe_wiring.lobe_paths(former, psi, n_turns,
                                       wire_pitch=dg.WIRE_PITCH, roi_center=roi.mean(0))
        info = lobe_wiring.write_lobe_csvs(paths, name, dg._DATA)
        print("%s: %d lobes, %d turns total" % (name, len(paths), n_turns))
        for fn, npts, ncross in info:
            flag = "OK" if ncross == 0 else "!! %d CROSSINGS" % ncross
            print("   %-28s %5d pts  [%s]" % (fn.split('/')[-1].split('\\')[-1], npts, flag))
    print("done.")


if __name__ == '__main__':
    main()
