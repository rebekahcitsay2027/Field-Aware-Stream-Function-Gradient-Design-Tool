"""
*** DEPRECATED -- DO NOT USE. Superseded by stream_gradient.py + wiring.py. ***
Same core math, reorganized (loop-stitching split out into wiring.py) and
re-verified. run_former.py no longer imports this file. Kept only for
reference; safe to delete manually whenever you're ready (I don't have
delete permission on this folder).

gradient_streamfn.py
=====================
Self-contained stream-function gradient-coil designer for an INHOMOGENEOUS magnet.

The one idea that differs from CoilGen / standard biplanar design:
    Instead of targeting a fixed Cartesian component (B_z), we target the
    projection of the coil field onto the LOCAL, PROBED main-field direction:

        g(r) = B_coil(r) . bhat(r),     bhat(r) = B0(r) / |B0(r)|

    and we drive g(r) to a linear ramp along a chosen geometric encoding axis
    (LR = x, or AP = y). In a homogeneous magnet bhat is constant and this
    collapses to "target B_z"; here bhat tilts up to ~46 deg across the ROI,
    so the projection genuinely matters.

Pipeline:
    1. Biplanar triangle-mesh former (two planes normal to x).
    2. Piecewise-linear stream function psi on the mesh vertices.
       Surface current density K = grad(psi) x n  (constant per triangle).
    3. Vector Biot-Savart sensitivity C : psi -> (Bx,By,Bz) at ROI points.
    4. Project: P = diag(bx)Cx + diag(by)Cy + diag(bz)Cz.
    5. Solve  min ||P psi - g_target - c*1||^2 + lambda * psi^T R psi
       (R = resistance/power matrix; c = free constant = unobservable B0 offset).
       psi = 0 on the open boundary so wires close.
    6. Contour psi into discrete wires; validate with a true Biot-Savart of the
       wires, projected onto bhat, and also the full |B0+Bg| incl. concomitant.

Units: SI internally (metres, Tesla, Amps). psi has units of Amps;
contour spacing = current per turn.
"""
import numpy as np

MU0_4PI = 1e-7  # mu0 / 4pi  [T m / A]


# ----------------------------------------------------------------------------
# Mesh
# ----------------------------------------------------------------------------
class PlanarPatch:
    """A flat rectangular patch meshed into triangles, embedded in 3D."""
    def __init__(self, verts, tris, boundary_mask):
        self.verts = verts                 # (V,3) in metres
        self.tris = tris                   # (T,3) int
        self.boundary_mask = boundary_mask # (V,) bool, True = on outer edge


def make_biplanar(x_planes_mm=(-200.0, 200.0),
                  y_halfspan_mm=250.0, z_halfspan_mm=140.0,
                  z_center_mm=0.0, ny=25, nz=17):
    """Two planes normal to x. Returns combined verts/tris and boundary mask.

    z_center_mm shifts the former vertically relative to the field-map origin,
    so the ROI need not be centered in the former (e.g. ROI sits at the bottom).
    ny, nz = number of grid NODES along y and z on each plane.
    """
    ys = np.linspace(-y_halfspan_mm, y_halfspan_mm, ny) * 1e-3
    zs = (np.linspace(-z_halfspan_mm, z_halfspan_mm, nz) + z_center_mm) * 1e-3
    all_v, all_t, all_b = [], [], []
    voff = 0
    for xp in np.array(x_planes_mm) * 1e-3:
        Y, Z = np.meshgrid(ys, zs, indexing='ij')   # (ny,nz)
        V = np.column_stack([np.full(Y.size, xp), Y.ravel(), Z.ravel()])
        idx = np.arange(Y.size).reshape(ny, nz)
        tris = []
        for i in range(ny - 1):
            for j in range(nz - 1):
                a, b, c, d = idx[i, j], idx[i+1, j], idx[i+1, j+1], idx[i, j+1]
                # consistent CCW winding wrt +x normal
                tris.append([a, b, c]); tris.append([a, c, d])
        tris = np.array(tris)
        bnd = ((Y.ravel() == ys[0]) | (Y.ravel() == ys[-1]) |
               (Z.ravel() == zs[0]) | (Z.ravel() == zs[-1]))
        all_v.append(V); all_t.append(tris + voff); all_b.append(bnd)
        voff += V.shape[0]
    return PlanarPatch(np.vstack(all_v), np.vstack(all_t), np.concatenate(all_b))


# ----------------------------------------------------------------------------
# Stream-function geometry: per-triangle gradient operators
# ----------------------------------------------------------------------------
def triangle_geometry(verts, tris):
    """Return per-triangle area, unit normal, and the 3 in-plane vectors k_i
    such that surface current K = sum_i psi_i * k_i  (K = grad(psi) x n).
    grad(lambda_i) = (n x e_i)/(2A), e_i = edge opposite vertex i.
    """
    p = verts[tris]                       # (T,3,3)
    p0, p1, p2 = p[:, 0], p[:, 1], p[:, 2]
    nrm = np.cross(p1 - p0, p2 - p0)      # (T,3)
    twoA = np.linalg.norm(nrm, axis=1)
    A = 0.5 * twoA
    nhat = nrm / twoA[:, None]
    # edges opposite each local vertex
    e = np.stack([p2 - p1, p0 - p2, p1 - p0], axis=1)   # (T,3,3)
    grad = np.cross(nhat[:, None, :], e) / twoA[:, None, None]  # (T,3,3) grad lambda_i
    k = np.cross(grad, nhat[:, None, :])  # (T,3,3)  K contribution per unit psi_i
    return A, nhat, k


def _subtri_quad(p0, p1, p2, k):
    """Quadrature points + weights by uniform subdivision into k^2 subtriangles.
    Returns (Q,3) points and (Q,) weights summing to triangle area."""
    pts, wts = [], []
    # barycentric centroids of the k^2 subtriangles of a uniform refinement
    A = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
    suba = A / (k * k)
    for i in range(k):
        for j in range(k - i):
            # "upward" subtriangle
            b0 = (i + 1/3.) / k; b1 = (j + 1/3.) / k
            pts.append((1 - b0 - b1) * p0 + b0 * p1 + b1 * p2); wts.append(suba)
            if j < k - i - 1:
                b0 = (i + 2/3.) / k; b1 = (j + 2/3.) / k
                pts.append((1 - b0 - b1) * p0 + b0 * p1 + b1 * p2); wts.append(suba)
    return np.array(pts), np.array(wts)


def sensitivity_matrix(verts, tris, field_pts, ksub=3):
    """Build C: (3, P, V). C[c,p,v] = d B_c(field_pts[p]) / d psi_v   [T/A].
    Vector Biot-Savart over the piecewise-linear stream-function basis.
    """
    P = field_pts.shape[0]; V = verts.shape[0]
    C = np.zeros((3, P, V))
    A, nhat, kvec = triangle_geometry(verts, tris)
    p = verts[tris]
    for t in range(tris.shape[0]):
        qp, qw = _subtri_quad(p[t, 0], p[t, 1], p[t, 2], ksub)   # (Q,3),(Q,)
        diff = field_pts[:, None, :] - qp[None, :, :]            # (P,Q,3)
        r3 = np.linalg.norm(diff, axis=2) ** 3                   # (P,Q)
        r3[r3 == 0] = np.inf
        # I_t(r) = integral (r-r')/|r-r'|^3 dA'  ~ sum_q w_q diff/r3
        It = np.einsum('q,pqc->pc', qw, diff / r3[:, :, None])   # (P,3)
        for i in range(3):
            v = tris[t, i]
            # contribution dB = mu0/4pi * (k_i x I_t)
            contrib = MU0_4PI * np.cross(kvec[t, i][None, :], It)  # (P,3)
            C[:, :, v] += contrib.T
    return C


def resistance_matrix(verts, tris):
    """R[v,w] = sum_t (k_v . k_w) * A_t   (power ~ psi^T R psi, sigma*thickness=1)."""
    V = verts.shape[0]
    A, nhat, kvec = triangle_geometry(verts, tris)
    R = np.zeros((V, V))
    for t in range(tris.shape[0]):
        for a in range(3):
            for b in range(3):
                R[tris[t, a], tris[t, b]] += np.dot(kvec[t, a], kvec[t, b]) * A[t]
    return R


# ----------------------------------------------------------------------------
# Solve
# ----------------------------------------------------------------------------
def solve_stream(P_proj, target, R, boundary_mask, lam):
    """min ||P psi - target - c*1||^2 + lam psi^T R psi, psi=0 on boundary.
    Returns full psi (V,)."""
    V = P_proj.shape[1]
    free = ~boundary_mask
    fidx = np.where(free)[0]
    Pf = P_proj[:, fidx]
    Rf = R[np.ix_(fidx, fidx)]
    # augment with free constant c (column of ones) -> absorbs unobservable offset
    ones = np.ones((Pf.shape[0], 1))
    Aaug = np.hstack([Pf, ones])
    # regularize only psi, not c
    Raug = np.zeros((Aaug.shape[1], Aaug.shape[1]))
    Raug[:len(fidx), :len(fidx)] = Rf
    M = Aaug.T @ Aaug + lam * Raug
    rhs = Aaug.T @ target
    sol = np.linalg.solve(M, rhs)
    psi = np.zeros(V); psi[fidx] = sol[:len(fidx)]
    return psi


# ----------------------------------------------------------------------------
# Contour into wires
# ----------------------------------------------------------------------------
def contour_patch(verts, tris, psi, levels):
    """Marching-triangles contour of psi. Segments are ORIENTED along the local
    surface-current direction K = grad(psi) x n, so that driving each level loop
    with current = (level spacing) reproduces the continuous surface current.
    Returns list of (level, segs); segs is a list of (2,3) arrays [start,end]."""
    A, nhat, _ = triangle_geometry(verts, tris)
    p = verts[tris]
    # per-triangle grad(psi)
    p0, p1, p2 = p[:, 0], p[:, 1], p[:, 2]
    twoA = 2 * A
    e = np.stack([p2 - p1, p0 - p2, p1 - p0], axis=1)
    gl = np.cross(nhat[:, None, :], e) / twoA[:, None, None]      # grad lambda_i
    gpsi = np.einsum('tij,ti->tj', gl, psi[tris])                 # (T,3) grad psi
    Kdir = np.cross(gpsi, nhat)                                   # current direction
    out = []
    for lev in levels:
        segs = []
        for ti, tr in enumerate(tris):
            vals = psi[tr] - lev
            pts = verts[tr]
            cross = []
            for a in range(3):
                b = (a + 1) % 3
                if (vals[a] > 0) != (vals[b] > 0):
                    f = vals[a] / (vals[a] - vals[b])
                    cross.append(pts[a] + f * (pts[b] - pts[a]))
            if len(cross) == 2:
                s = np.array(cross)
                if np.dot(s[1] - s[0], Kdir[ti]) < 0:            # orient along K
                    s = s[::-1]
                segs.append(s)
        out.append((lev, segs))
    return out


# ----------------------------------------------------------------------------
# Discrete-wire forward field (validation) -- segments carrying current I
# ----------------------------------------------------------------------------
def biot_savart_segments(seglist_with_I, field_pts):
    """seglist_with_I: list of (Nx2x3 segment endpoints array, current I).
    Returns B (P,3) [T]."""
    B = np.zeros_like(field_pts)
    for segs, I in seglist_with_I:
        if len(segs) == 0:
            continue
        segs = np.asarray(segs)              # (N,2,3)
        a = segs[:, 0, :]; b = segs[:, 1, :]
        dl = b - a                            # (N,3)
        mid = 0.5 * (a + b)
        # finite straight-segment Biot-Savart (exact for straight wire)
        for n in range(segs.shape[0]):
            r1 = field_pts - a[n]; r2 = field_pts - b[n]
            r1n = np.linalg.norm(r1, axis=1); r2n = np.linalg.norm(r2, axis=1)
            cross = np.cross(r1, r2)
            denom = (r1n * r2n * (r1n * r2n + np.einsum('pc,pc->p', r1, r2)))
            denom[denom == 0] = np.inf
            fac = (r1n + r2n) / denom
            B += MU0_4PI * I * cross * fac[:, None]
    return B


def make_u_channel(wall_x_mm=200.0, bottom_z_mm=-140.0, top_z_mm=144.0,
                   fold_r_mm=28.0, y_halfspan_mm=250.0, n_perim=70, n_y=34):
    """U-channel (trough) former conforming to the magnet interior:
    two vertical walls at x=+/-wall_x joined by a flat bottom at z=bottom_z,
    rounded by fold_r at the lower corners, extruded along AP (y).
    Open at top (+z) and both y-ends -> single boundary loop (psi=0 there).
    Returns a PlanarPatch (generic triangle mesh)."""
    R = fold_r_mm
    xw, zb, zt = wall_x_mm, bottom_z_mm, top_z_mm
    # cross-section polyline in (x,z), top-left -> top-right, as (point, arclen)
    pts = []
    pts.append((-xw, zt))                       # top of left wall
    pts.append((-xw, zb + R))                   # bottom of left wall (start of fold)
    # left fold quarter arc: center (-xw+R, zb+R)
    for a in np.linspace(np.pi, 1.5*np.pi, 8)[1:]:
        pts.append((-xw + R + R*np.cos(a), zb + R + R*np.sin(a)))
    pts.append((xw - R, zb))                    # flat bottom
    for a in np.linspace(1.5*np.pi, 2*np.pi, 8)[1:]:
        pts.append((xw - R + R*np.cos(a), zb + R + R*np.sin(a)))
    pts.append((xw, zb + R))
    pts.append((xw, zt))                        # top of right wall
    poly = np.array(pts)
    # resample uniformly by arclength to n_perim points
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    su = np.linspace(0, s[-1], n_perim)
    xz = np.column_stack([np.interp(su, s, poly[:, 0]),
                          np.interp(su, s, poly[:, 1])])
    ys = np.linspace(-y_halfspan_mm, y_halfspan_mm, n_y)
    # build grid (n_perim x n_y)
    V = []
    for i in range(n_perim):
        for j in range(n_y):
            V.append([xz[i, 0], ys[j], xz[i, 1]])
    V = np.array(V) * 1e-3
    idx = np.arange(n_perim * n_y).reshape(n_perim, n_y)
    tris = []
    for i in range(n_perim - 1):
        for j in range(n_y - 1):
            a, b, c, d = idx[i, j], idx[i+1, j], idx[i+1, j+1], idx[i, j+1]
            tris.append([a, b, c]); tris.append([a, c, d])
    tris = np.array(tris)
    bnd = np.zeros(n_perim * n_y, bool)
    bnd[idx[0, :]] = True; bnd[idx[-1, :]] = True      # wall tops
    bnd[idx[:, 0]] = True; bnd[idx[:, -1]] = True      # y-end rims
    return PlanarPatch(V, tris, bnd)
