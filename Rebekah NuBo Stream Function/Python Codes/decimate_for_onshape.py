"""Decimate a *_former_wire.csv into an Onshape-ready point list.
Output: 3 columns (x,y,z) in METRES, matching the 'Path from Points' FeatureScript
(which reads cols 0,1,2 and multiplies by `meter`).  Douglas-Peucker keeps the
fewest points such that the path never deviates more than TOL_MM from the original.
Usage:  python decimate_for_onshape.py  LR_former_wire.csv  LR_onshape.csv  1.0
"""
import sys, numpy as np

def dp(P, tol):
    n=len(P); keep=np.zeros(n,bool); keep[0]=keep[-1]=True; stack=[(0,n-1)]
    while stack:
        i,j=stack.pop()
        if j<=i+1: continue
        a=P[i]; ab=P[j]-a; L2=ab@ab; seg=P[i+1:j]
        if L2<1e-18: dvec=np.linalg.norm(seg-a,axis=1)
        else:
            t=np.clip(((seg-a)@ab)/L2,0,1); dvec=np.linalg.norm(seg-(a+t[:,None]*ab),axis=1)
        k=int(np.argmax(dvec))
        if dvec[k]>tol:
            idx=i+1+k; keep[idx]=True; stack += [(i,idx),(idx,j)]
    return keep

def main(src, dst, tol_mm):
    d=np.genfromtxt(src, delimiter=',', skip_header=1)
    P=d[:,1:4]                       # x_mm,y_mm,z_mm (drop point_index column)
    keep=dp(P, tol_mm)
    Pm=P[keep]/1000.0                # -> metres for the FeatureScript
    np.savetxt(dst, Pm, header='x_m,y_m,z_m', delimiter=',', comments='', fmt='%.6f')
    L=np.sum(np.linalg.norm(np.diff(P,axis=0),axis=1))/1000
    print("%s: %d -> %d points (tol %.1f mm), %.1f m -> %s" % (src,len(P),keep.sum(),tol_mm,L,dst))

if __name__=='__main__':
    if len(sys.argv)>=3:
        main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv)>3 else 1.0)
    else:  # default: process both, reading run_former.py's output CSVs from this same folder
        for nm in ['LR','AP']:
            main('%s_former_wire.csv'%nm, '%s_onshape.csv'%nm, 1.0)
