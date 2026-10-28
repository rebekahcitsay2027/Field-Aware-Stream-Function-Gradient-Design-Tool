"""
*** DEPRECATED -- DO NOT USE. Superseded by former_mesh.py. ***
This file recentered/shifted the probe ROI to a hardcoded offset and buried
the STL axis remap inside a function call. former_mesh.py replaces it with
no ROI shifting and an explicit, documented AXIS_MAP. Kept only for
reference; run_former.py no longer imports this file. Safe to delete
manually whenever you're ready (I don't have delete permission on this
folder).
"""
import numpy as np
from collections import defaultdict
from stl import mesh as stlmesh
from gradient_streamfn import PlanarPatch

def _weld(tri, tol=1e-5):
    """Merge coincident corners into shared vertices. tri: (T,3,3)->verts,tris."""
    flat = tri.reshape(-1, 3)
    keys = np.round(flat / tol).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    verts = np.zeros((len(uniq), 3))
    np.add.at(verts, inv, flat)
    verts /= np.bincount(inv)[:, None]
    tris = inv.reshape(-1, 3)
    good = (tris[:,0]!=tris[:,1])&(tris[:,1]!=tris[:,2])&(tris[:,0]!=tris[:,2])
    return verts, tris[good]

def _orient(verts, tris):
    """Flood-fill consistent winding across shared edges."""
    adj = defaultdict(list)
    for t,(a,b,c) in enumerate(tris):
        for u,w in [(a,b),(b,c),(c,a)]:
            adj[(min(u,w),max(u,w))].append(t)
    tris = tris.copy(); seen = np.zeros(len(tris), bool)
    def dirset(t): return [(tris[t][i],tris[t][(i+1)%3]) for i in range(3)]
    import collections
    for s in range(len(tris)):
        if seen[s]: continue
        seen[s]=True; stack=[s]
        while stack:
            t=stack.pop()
            for (a,b) in dirset(t):
                for nb in adj[(min(a,b),max(a,b))]:
                    if nb==t or seen[nb]: continue
                    # if neighbor traverses edge same direction -> flip it
                    if (a,b) in dirset(nb):
                        tris[nb]=tris[nb][::-1]
                    seen[nb]=True; stack.append(nb)
    return tris

def _subdivide(verts, tris):
    mid={}; nv=list(verts)
    def m(a,b):
        k=(min(a,b),max(a,b))
        if k not in mid: mid[k]=len(nv); nv.append((verts[a]+verts[b])/2)
        return mid[k]
    nt=[]
    for a,b,c in tris:
        ab,bc,ca=m(a,b),m(b,c),m(c,a)
        nt+=[[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]]
    return np.array(nv), np.array(nt)

def _boundary(verts, tris):
    cnt=defaultdict(int)
    for a,b,c in tris:
        for u,w in [(a,b),(b,c),(c,a)]: cnt[(min(u,w),max(u,w))]+=1
    bnd=np.zeros(len(verts),bool)
    for (u,w),n in cnt.items():
        if n==1: bnd[u]=bnd[w]=True
    return bnd

def load_former_stl(path, cols=(2,0,1), subdivide=2, tol=1e-5):
    """cols maps STL axes -> field (x,y,z). Default (2,0,1):
    field_x=stl_z, field_y=stl_x, field_z=stl_y  (your remap)."""
    tri = stlmesh.Mesh.from_file(path).vectors[..., list(cols)]  # remap, metres
    verts, tris = _weld(tri, tol)
    tris = _orient(verts, tris)
    for _ in range(subdivide):
        verts, tris = _subdivide(verts, tris)
    bnd = _boundary(verts, tris)
    return PlanarPatch(verts, tris, bnd), verts, tris

if __name__=='__main__':
    import gradient_streamfn as gs
    patch,V,T = load_former_stl('/mnt/user-data/uploads/Part_Studio_1_-_Part_1.stl',subdivide=2)
    print("welded+subdivided: %d verts (%d free, %d boundary), %d tris"
          %(len(V),(~patch.boundary_mask).sum(),patch.boundary_mask.sum(),len(T)))
    print("bbox (field frame, m): min",np.round(V.min(0),3)," max",np.round(V.max(0),3))
    # connectivity check
    A,nhat,_=gs.triangle_geometry(V,T)
    de=defaultdict(int)
    for a,b,c in T:
        for u,w in [(a,b),(b,c),(c,a)]: de[(u,w)]+=1
    bad=sum(1 for k,n in de.items() if n>1)
    print("directed-edge winding conflicts:",bad," (0 = consistent orientation)")
    print("triangle areas: min %.2e max %.2e m^2 (no zeros = clean)"%(A.min(),A.max()))


def _inner_face(verts, tris, toward=(0,0,0)):
    """Keep triangles whose normal points toward `toward` (the ROI cavity).
    Use for closed-solid STLs to recover the single copper-side sheet."""
    import gradient_streamfn as gs
    A,nh,_=gs.triangle_geometry(verts,tris)
    tc=verts[tris].mean(1)
    keep = np.einsum('tc,tc->t', nh, np.asarray(toward)-tc) > 0
    Ti=tris[keep]; used=np.unique(Ti)
    rm={v:i for i,v in enumerate(used)}
    return verts[used], np.array([[rm[v] for v in t] for t in Ti])

def load_former_stl(path, cols=(2,0,1), subdivide=2, tol=1e-5, inner_toward=None):
    """cols: STL->field axis map (default your remap). If the mesh is a closed
    solid, set inner_toward=(x,y,z) of the ROI centre to extract the copper-side
    face automatically."""
    tri = stlmesh.Mesh.from_file(path).vectors[..., list(cols)]
    verts, tris = _weld(tri, tol)
    # auto-detect closed solid -> extract inner face
    cnt=defaultdict(int)
    for a,b,c in tris:
        for u,w in [(a,b),(b,c),(c,a)]: cnt[(min(u,w),max(u,w))]+=1
    closed = sum(1 for n in cnt.values() if n==1)==0
    if closed:
        verts,tris=_inner_face(verts,tris,inner_toward if inner_toward is not None else (0,0,0))
    tris=_orient(verts,tris)
    for _ in range(subdivide): verts,tris=_subdivide(verts,tris)
    return PlanarPatch(verts,tris,_boundary(verts,tris)), verts, tris
