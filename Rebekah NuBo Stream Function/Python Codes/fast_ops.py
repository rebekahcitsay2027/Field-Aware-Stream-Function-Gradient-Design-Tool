import numpy as np
import stream_gradient as sg
MU0_4PI = 1e-7

def _quad_barycentric(subdivisions):
    bary = []; s = subdivisions
    for i in range(s):
        for j in range(s - i):
            b0, b1 = (i + 1/3.) / s, (j + 1/3.) / s
            bary.append((1 - b0 - b1, b0, b1))
            if j < s - i - 1:
                b0, b1 = (i + 2/3.) / s, (j + 2/3.) / s
                bary.append((1 - b0 - b1, b0, b1))
    return np.array(bary)

def sensitivity_matrix(verts, tris, field_pts, quad_subdiv=2, block=300):
    n_pts, n_verts = field_pts.shape[0], verts.shape[0]
    area, _, kvec = sg.triangle_geometry(verts, tris)
    corners = verts[tris]
    bary = _quad_barycentric(quad_subdiv)
    qp_all = np.einsum('qk,tkc->tqc', bary, corners)
    w = area / (quad_subdiv * quad_subdiv)
    Cv = np.zeros((n_verts, n_pts, 3))
    T = tris.shape[0]
    for start in range(0, T, block):
        sl = slice(start, min(start + block, T))
        qp = qp_all[sl]
        diff = field_pts[:, None, None, :] - qp[None, :, :, :]
        r = np.linalg.norm(diff, axis=-1); r3 = r ** 3; r3[r3 == 0] = np.inf
        integ = w[sl][None, :, None] * (diff / r3[..., None]).sum(axis=2)
        integ_bp = integ.transpose(1, 0, 2)
        for i in range(3):
            kv = kvec[sl, i, :]
            contr = MU0_4PI * np.cross(kv[:, None, :], integ_bp)
            np.add.at(Cv, tris[sl, i], contr)
    return Cv.transpose(2, 1, 0)

def resistance_matrix(verts, tris):
    n_verts = verts.shape[0]
    area, _, kvec = sg.triangle_geometry(verts, tris)
    G = np.einsum('tac,tbc->tab', kvec, kvec) * area[:, None, None]
    R = np.zeros((n_verts, n_verts))
    for a in range(3):
        for b in range(3):
            np.add.at(R, (tris[:, a], tris[:, b]), G[:, a, b])
    return R

def biot_savart_wire(segments_with_current, field_pts, block=3000):
    B = np.zeros_like(field_pts)
    for segs, I in segments_with_current:
        if len(segs) == 0:
            continue
        segs = np.asarray(segs)
        a, b = segs[:, 0, :], segs[:, 1, :]
        S = a.shape[0]
        for s0 in range(0, S, block):
            aa, bb = a[s0:s0+block], b[s0:s0+block]
            r1 = field_pts[:, None, :] - aa[None, :, :]
            r2 = field_pts[:, None, :] - bb[None, :, :]
            r1n = np.linalg.norm(r1, axis=2); r2n = np.linalg.norm(r2, axis=2)
            cross = np.cross(r1, r2)
            denom = r1n * r2n * (r1n * r2n + np.einsum('psc,psc->ps', r1, r2))
            denom[denom == 0] = np.inf
            factor = (r1n + r2n) / denom
            B += MU0_4PI * I * (cross * factor[:, :, None]).sum(axis=1)
    return B
