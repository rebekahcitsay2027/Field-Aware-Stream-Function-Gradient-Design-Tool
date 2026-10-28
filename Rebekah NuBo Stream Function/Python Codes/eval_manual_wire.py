"""
eval_manual_wire.py
===================
Evaluate the gradient (T/m per amp) and linearity (R^2) of a MANUALLY WOUND
wire path -- given as a CSV of (x,y,z) points -- against the measured B0 map,
using exactly the same physics as design_gradient.py's validation step:

    B_wire = Biot-Savart of the wire at the ROI points (per 1 A of series
             current through the single continuous conductor)
    projected = B_wire . b0hat        (component along the LOCAL main field)
    slope, R^2 = linear fit of projected vs. the encoding coordinate
    efficiency = slope   [T/m per A]  (since we drive 1 A, slope IS the
                                       gradient-per-amp = efficiency)

Because Biot-Savart is linear in current, evaluating at I = 1 A makes the
fitted slope the efficiency directly (mT/m/A) -- no need to know the coil's
real operating current or turn count.

FRAME REMAP
-----------
These wire CSVs were authored in the *CoilGen* frame, which is NOT the probe/B0
frame that magnetData.csv lives in. Per the winding's provenance, the mapping is:

    B0_x = csv_z ,   B0_y = csv_x ,   B0_z = csv_y

i.e. reorder the CSV columns (x,y,z) -> (z,x,y) = columns [2,0,1]. That is
`CSV_AXIS_MAP` below. The ROI/B0 map is used exactly as measured (mm->m only);
nothing is shifted or recentred. The printed wire-vs-ROI bounding boxes let you
confirm the wire actually surrounds the ROI after the remap (a wrong remap or a
missing origin offset shows up as a mis-registered box and a poor R^2).

Usage:
    python eval_manual_wire.py            # evaluates both entries in WIRES
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import stream_gradient as sg
import design_gradient as dg   # reuse FIELD_CSV, ENCODE, _fit_slope

# Optional: use the vectorized Biot-Savart if fast_ops.py is present (much
# faster on these multi-thousand-point paths); otherwise fall back to the
# reference implementation in stream_gradient.
try:
    import fast_ops
    sg.biot_savart_wire = fast_ops.biot_savart_wire
except Exception:
    pass

_DATA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CoilGen-frame CSV column order -> B0/probe frame:  B0=(csv_z, csv_x, csv_y)
CSV_AXIS_MAP = (2, 0, 1)

# name -> (csv filename in _DATA, encoding-axis key in dg.ENCODE)
WIRES = {
    'LR': ('wire_path_full_reduced_densified_final_LR_MW_path.csv', 'LR'),
    'AP': ('wire_path_full_reduced_v1.csv',                          'AP'),
}

AXIS_LABEL = {0: 'x (LR)', 1: 'y (AP)', 2: 'z'}


def load_field():
    data = np.genfromtxt(dg.FIELD_CSV, delimiter=',', skip_header=1)
    roi = data[:, :3] * 1e-3
    B0 = data[:, 3:6]
    bhat = B0 / np.linalg.norm(B0, axis=1, keepdims=True)
    return roi, bhat


def load_wire(csv_name):
    """Load a manual wire CSV (x,y,z in metres) and remap CoilGen->B0 frame."""
    path = csv_name if os.path.isabs(csv_name) else os.path.join(_DATA, csv_name)
    d = np.genfromtxt(path, delimiter=',', skip_header=1)
    if d.shape[1] < 3:
        raise ValueError("expected >=3 columns (x,y,z) in %s" % path)
    return d[:, list(CSV_AXIS_MAP)]          # (N,3) metres, B0 frame


def evaluate(name, csv_name, encode_key, roi, bhat, make_plot=True):
    axis = dg.ENCODE[encode_key]
    wire = load_wire(csv_name)
    segs = np.stack([wire[:-1], wire[1:]], axis=1)
    B = sg.biot_savart_wire([(segs, 1.0)], roi)          # T per 1 A
    projected = np.einsum('pc,pc->p', B, bhat)

    slope, r2 = dg._fit_slope(projected, roi[:, axis])   # T/m per A, and R^2
    length = np.sum(np.linalg.norm(np.diff(wire, axis=0), axis=1))

    # alignment check
    wmin, wmax = wire.min(0), wire.max(0)
    rmin, rmax = roi.min(0), roi.max(0)
    inside = np.all((rmin >= wmin) & (rmax <= wmax))
    print("== %s  (%s) ==" % (name, os.path.basename(csv_name)))
    print("  wire: %d pts, %.2f m; bbox(m) x[%+.3f,%+.3f] y[%+.3f,%+.3f] z[%+.3f,%+.3f]"
          % (len(wire), length, wmin[0], wmax[0], wmin[1], wmax[1], wmin[2], wmax[2]))
    print("  ROI fully inside wire bbox: %s" % bool(inside))
    # gradient along every axis -> confirms which axis this coil actually encodes
    for ax in (0, 1, 2):
        s, rr = dg._fit_slope(projected, roi[:, ax])
        tag = "  <-- encoding axis" if ax == axis else ""
        print("    grad vs %-7s: %+.4f mT/m/A   R^2=%.4f%s" % (AXIS_LABEL[ax], s * 1e3, rr, tag))
    print("  EFFICIENCY = %.4f mT/m/A   (|slope| along %s),   linearity R^2 = %.4f"
          % (abs(slope) * 1e3, AXIS_LABEL[axis], r2))
    print("  projected-field span over ROI (per A) = %.4f mT/A"
          % ((projected.max() - projected.min()) * 1e3))

    if make_plot:
        _plot(name, axis, wire, roi, projected, slope, r2)

    return dict(name=name, axis=axis, efficiency=abs(slope), r_squared=r2,
                wire_length=length, n_pts=len(wire))


def _plot(name, axis, wire, roi, projected, slope, r2):
    fig = plt.figure(figsize=(11, 4.4))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(wire[:, 0], wire[:, 1], wire[:, 2], color='tab:green', lw=0.6)
    ax1.scatter(roi[:, 0], roi[:, 1], roi[:, 2], color='tab:red', s=2)
    ax1.set_title('%s manual wire + ROI (B0 frame)' % name)
    ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')

    ax2 = fig.add_subplot(1, 2, 2)
    a = roi[:, axis] * 1000.0
    ax2.scatter(a, projected * 1e3, s=6, alpha=0.35, color='tab:blue', label='B_wire . b0hat (per A)')
    xs = np.linspace(a.min(), a.max(), 50)
    ax2.plot(xs, (slope / 1000.0 * xs + (projected.mean() - slope * roi[:, axis].mean())) * 1e3,
             'r-', lw=1.5, label='fit: %.3f mT/m/A, R^2=%.3f' % (abs(slope) * 1e3, r2))
    ax2.set_xlabel('%s position in ROI (mm)' % AXIS_LABEL[axis])
    ax2.set_ylabel('projected field per amp (mT/A)')
    ax2.set_title('%s manual-wire linearity' % name); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_DATA, 'MANUAL_%s_check.png' % name)
    fig.savefig(out, dpi=140); plt.close(fig)
    print("  saved %s" % out)


def main():
    roi, bhat = load_field()
    print("ROI: %d pts, bbox(m) x[%+.3f,%+.3f] y[%+.3f,%+.3f] z[%+.3f,%+.3f]\n"
          % (len(roi), roi[:, 0].min(), roi[:, 0].max(), roi[:, 1].min(), roi[:, 1].max(),
             roi[:, 2].min(), roi[:, 2].max()))
    for name, (csv_name, key) in WIRES.items():
        evaluate(name, csv_name, key, roi, bhat)
        print()
    print("done.")


if __name__ == '__main__':
    main()
