"""
stream_gradient.py
===================
Stream-function gradient-coil solver for an INHOMOGENEOUS magnet.

The core idea (unchanged from the previous version of this pipeline, and
still the right one): instead of targeting a fixed Cartesian field
component (e.g. Bz), target the projection of the coil's field onto the
LOCAL, PROBED main-field direction at each ROI point:

    g(r) = B_coil(r) . bhat(r),      bhat(r) = B0(r) / |B0(r)|

and drive g(r) to a linear ramp along a chosen geometric encoding axis (a
column of your raw probe coordinates -- e.g. column 0 for "LR"). In a
perfectly homogeneous magnet bhat is constant and this collapses to
"target one fixed component"; for a real, inhomogeneous magnet bhat can
tilt by tens of degrees across the ROI, so the local projection genuinely
matters (confirmed in this project: R^2 = 0.9956 for the LR design using
this approach, vs. noticeably worse fits when a single global component
was assumed instead).

No coordinate remapping or shifting happens in this module -- it operates
on whatever (x,y,z) values you hand it (mesh vertices, ROI points), in
whatever units they're already in (this project uses metres throughout).

Pipeline used by design_gradient.py:
  1. Piecewise-linear stream function psi on the former mesh vertices.
     Surface current density K = grad(psi) x n (constant per triangle).
  2. Vector Biot-Savart sensitivity C: psi -> (Bx,By,Bz) at ROI points.
  3. Project onto the local B0 direction: P = diag(bx)Cx + diag(by)Cy + diag(bz)Cz.
  4. Solve  min ||P psi - target - c*1||^2 + lambda * psi^T R psi
     (R = a power/resistance-like regularization matrix; c is a free
     constant that absorbs the unobservable B0 offset). psi = 0 on the
     open mesh boundary so contour loops close.
  5. Contour psi into discrete wire loops at evenly-spaced current levels.
  6. Stitch the loops into a single continuous wire (see wiring.py).
  7. Validate with a TRUE discrete-wire Biot-Savart of the stitched path,
     projected onto bhat, fit against the requested linear ramp (reports
     the achieved gradient and R^2 -- this is the real, honest check, not
     just how well the continuous stream function fit its own target).

Units: SI throughout (metres, Tesla, Amps). psi has units of Amps;
contour spacing between adjacent levels = current per turn.
"""
import numpy as np

MU0_4PI = 1e-7  # mu0 / 4pi  [T*m/A]


# ----------------------------------------------------------------------------
# Per-triangle geometry
# ----------------------------------------------------------------------------
def triangle_geometry(verts, tris):
    """Per-triangle area, unit normal, and the 3 in-plane basis vectors k_i
    such that the surface current density is K = sum_i psi_i * k_i
    (this is exactly grad(psi) x n for the piecewise-linear stream function).
    """
    p = verts[tris]                          # (T,3,3)
    p0, p1, p2 = p[:, 0], p[:, 1], p[:, 2]
    normal = np.cross(p1 - p0, p2 - p0)      # (T,3), magnitude = 2*area
    two_area = np.linalg.norm(normal, axis=1)
    area = 0.5 * two_area
    nhat = normal / two_area[:, None]
    edge_opposite = np.stack([p2 - p1, p0 - p2, p1 - p0], axis=1)     # (T,3,3)
    grad_lambda = np.cross(nhat[:, None, :], edge_opposite) / two_area[:, None, None]  # grad of barycentric coords
    k = np.cross(grad_lambda, nhat[:, None, :])   # (T,3,3): K contribution per unit psi at each local vertex
    return area, nhat, k


def _subtriangle_quadrature(p0, p1, p2, subdivisions):
    """Quadrature points/weights for a triangle via uniform sub-triangle
    centroids. `subdivisions` sub-triangles per edge -> subdivisions^2 points."""
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
    sub_area = area / (subdivisions * subdivisions)
    pts, wts = [], []
    for i in range(subdivisions):
        for j in range(subdivisions - i):
            b0, b1 = (i + 1/3.) / subdivisions, (j + 1/3.) / subdivisions
            pts.append((1 - b0 - b1) * p0 + b0 * p1 + b1 * p2)
            wts.append(sub_area)
            if j < subdivisions - i - 1:
                b0, b1 = (i + 2/3.) / subdivisions, (j + 2/3.) / subdivisions
                pts.append((1 - b0 - b1) * p0 + b0 * p1 + b1 * p2)
                wts.append(sub_area)
    return np.array(pts), np.array(wts)


# ----------------------------------------------------------------------------
# Sensitivity (vector Biot-Savart from the stream-function basis)
# ----------------------------------------------------------------------------
def sensitivity_matrix(verts, tris, field_pts, quad_subdiv=2):
    """C[c, p, v] = d B_c(field_pts[p]) / d psi_v   [T/A].
    Vector Biot-Savart of the piecewise-linear stream-function basis,
    evaluated by per-triangle sub-triangle quadrature."""
    n_pts, n_verts = field_pts.shape[0], verts.shape[0]
    C = np.zeros((3, n_pts, n_verts))
    _, _, kvec = triangle_geometry(verts, tris)
    corners = verts[tris]
    for t in range(tris.shape[0]):
        qp, qw = _subtriangle_quadrature(corners[t, 0], corners[t, 1], corners[t, 2], quad_subdiv)
        diff = field_pts[:, None, :] - qp[None, :, :]           # (P,Q,3)
        r_cubed = np.linalg.norm(diff, axis=2) ** 3              # (P,Q)
        r_cubed[r_cubed == 0] = np.inf
        integral = np.einsum('q,pqc->pc', qw, diff / r_cubed[:, :, None])   # (P,3)
        for i in range(3):
            v = tris[t, i]
            contribution = MU0_4PI * np.cross(kvec[t, i][None, :], integral)   # (P,3)
            C[:, :, v] += contribution.T
    return C


def resistance_matrix(verts, tris):
    """R[v,w] = sum_t (k_v . k_w) * area_t -- a power-like quadratic form used
    purely for Tikhonov regularization (psi^T R psi), not a literal resistance."""
    n_verts = verts.shape[0]
    area, _, kvec = triangle_geometry(verts, tris)
    R = np.zeros((n_verts, n_verts))
    for t in range(tris.shape[0]):
        for a in range(3):
            for b in range(3):
                R[tris[t, a], tris[t, b]] += np.dot(kvec[t, a], kvec[t, b]) * area[t]
    return R


# ----------------------------------------------------------------------------
# Solve for the stream function
# ----------------------------------------------------------------------------
def solve_stream(P_proj, target, R, boundary_mask, lam):
    """min ||P_proj psi - target - c||^2 + lam * psi^T R psi, with psi=0 on
    the mesh's open boundary (so contour loops close) and a free scalar c
    that absorbs whatever constant offset the projection can't observe.
    Returns the full psi vector (zero at boundary vertices)."""
    free = np.where(~boundary_mask)[0]
    P_free = P_proj[:, free]
    R_free = R[np.ix_(free, free)]
    ones = np.ones((P_free.shape[0], 1))
    A_aug = np.hstack([P_free, ones])
    R_aug = np.zeros((A_aug.shape[1], A_aug.shape[1]))
    R_aug[:len(free), :len(free)] = R_free   # don't regularize the free constant c
    M = A_aug.T @ A_aug + lam * R_aug
    rhs = A_aug.T @ target
    solution = np.linalg.solve(M, rhs)
    psi = np.zeros(P_proj.shape[1])
    psi[free] = solution[:len(free)]
    return psi


def max_surface_current(verts, tris, psi):
    """Peak |K| = |grad(psi) x n| over all triangles -- used to pick a wire
    pitch-safe turn count (so adjacent contour levels never demand more
    current density than a real wire pitch can carry)."""
    area, nhat, _ = triangle_geometry(verts, tris)
    p = verts[tris]
    two_area = 2 * area
    edge_opposite = np.stack([p[:, 2] - p[:, 1], p[:, 0] - p[:, 2], p[:, 1] - p[:, 0]], axis=1)
    grad_lambda = np.cross(nhat[:, None, :], edge_opposite) / two_area[:, None, None]
    grad_psi = np.einsum('tij,ti->tj', grad_lambda, psi[tris])
    return np.linalg.norm(grad_psi, axis=1).max()


# ----------------------------------------------------------------------------
# Contour psi into discrete wire loops
# ----------------------------------------------------------------------------
def contour_levels(verts, tris, psi, levels):
    """Marching-triangles contour of psi at each requested level. Segments
    are oriented along the local surface-current direction K = grad(psi) x n,
    so driving each contour loop with the level spacing as current reproduces
    the continuous surface current. Returns [(level, [seg, ...]), ...] where
    each seg is a (2,3) array [start_point, end_point]."""
    area, nhat, _ = triangle_geometry(verts, tris)
    p = verts[tris]
    two_area = 2 * area
    edge_opposite = np.stack([p[:, 2] - p[:, 1], p[:, 0] - p[:, 2], p[:, 1] - p[:, 0]], axis=1)
    grad_lambda = np.cross(nhat[:, None, :], edge_opposite) / two_area[:, None, None]
    grad_psi = np.einsum('tij,ti->tj', grad_lambda, psi[tris])
    current_dir = np.cross(grad_psi, nhat)

    out = []
    for level in levels:
        segs = []
        for ti, tri in enumerate(tris):
            vals = psi[tri] - level
            pts = verts[tri]
            crossings = []
            for a in range(3):
                b = (a + 1) % 3
                if (vals[a] > 0) != (vals[b] > 0):
                    f = vals[a] / (vals[a] - vals[b])
                    crossings.append(pts[a] + f * (pts[b] - pts[a]))
            if len(crossings) == 2:
                seg = np.array(crossings)
                if np.dot(seg[1] - seg[0], current_dir[ti]) < 0:
                    seg = seg[::-1]
                segs.append(seg)
        out.append((level, segs))
    return out


# ----------------------------------------------------------------------------
# Discrete-wire forward field (used for honest post-hoc validation)
# ----------------------------------------------------------------------------
def biot_savart_wire(segments_with_current, field_pts):
    """segments_with_current: list of (Nx2x3 segment endpoints, current I).
    Returns B (P,3) [T] -- exact finite-straight-segment Biot-Savart, no
    quadrature approximation (unlike sensitivity_matrix, which is for the
    continuous stream-function basis)."""
    B = np.zeros_like(field_pts)
    for segs, I in segments_with_current:
        if len(segs) == 0:
            continue
        segs = np.asarray(segs)
        a, b = segs[:, 0, :], segs[:, 1, :]
        for n in range(segs.shape[0]):
            r1 = field_pts - a[n]
            r2 = field_pts - b[n]
            r1n = np.linalg.norm(r1, axis=1)
            r2n = np.linalg.norm(r2, axis=1)
            cross = np.cross(r1, r2)
            denom = r1n * r2n * (r1n * r2n + np.einsum('pc,pc->p', r1, r2))
            denom[denom == 0] = np.inf
            factor = (r1n + r2n) / denom
            B += MU0_4PI * I * cross * factor[:, None]
    return B
