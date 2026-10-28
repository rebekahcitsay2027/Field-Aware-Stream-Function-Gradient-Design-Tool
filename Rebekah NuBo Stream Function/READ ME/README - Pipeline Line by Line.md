# `design_gradient.py` — line-by-line pipeline walkthrough

This document traces **exactly** what happens when you run

NOTE:make sure your .stl is subdivided and triangulated properly before you run the stream
function on it, or else there will not be a feasible wire path for the stream function

```
python design_gradient.py
```

in execution order, following every function call into `former_mesh.py`,
`stream_gradient.py`, and `wiring.py` as it is reached. It is written to be read
next to the source files.

The physical goal: find a **surface current** on the coil former that, when it
flows, adds a magnetic field whose component *along the local main field B₀*
grows **linearly along x** across the sample region (ROI). That linear ramp is
the frequency-encoding gradient. The math trick that makes this tractable is the
stream function.

---

## 0. The mental model (read this first)

A divergence-free current confined to a surface can be written as the "curl" of
a single scalar field ψ (the **stream function**) living on that surface:

```
K(r) = ∇ψ(r) × n̂(r)        [surface current density, A/m]
```

Two consequences drive the whole algorithm:

1. **Contours of ψ are wires.** If you draw the curves ψ = constant at evenly
   spaced values, and run one turn of wire along each curve carrying a current
   equal to the spacing Δψ between adjacent contours, the resulting discrete
   winding reproduces the smooth surface current K. So *solving for ψ* and then
   *contouring it* gives you a physical coil.

2. **ψ is linear per triangle.** We represent ψ by one value ψ_v at each mesh
   vertex and interpolate linearly inside each triangle. Then ∇ψ is *constant*
   inside each triangle, so K is constant per triangle — which makes both the
   field it produces (Biot–Savart) and the regularization integral exact,
   closed-form per triangle.

The unknowns are the vertex values ψ_v. We set up a linear least-squares problem:
"pick ψ so the coil field projected on B₀ best matches a linear-in-x ramp on the
ROI, with a smoothness/effort penalty." Solve it, contour it, stitch the contours
into one wire, and verify by a true Biot–Savart of that wire.

Units are SI throughout (metres, Tesla, Amps). ψ has units of Amps; the spacing
between adjacent contour levels is the current per turn.

---

## 1. Module load and entry point

At import, `design_gradient.py` runs its top-level statements:

- `import numpy as np`, matplotlib forced to the non-interactive `'Agg'` backend
  (so it renders PNGs without a display).
- `import stream_gradient as sg`, `import former_mesh as fm`, `import wiring` —
  the three worker modules.
- The configuration constants are bound (these are the knobs you tune):
  - `FIELD_CSV = 'magnetData.csv'` — the probed B₀ map.
  - `STL_PATH = 'raw_data_former.stl'` — the former surface (STEP exported to STL).
  - `AXIS_MAP = (0, 1, 2)` — identity: the STL already sits in the CSV's frame.
  - `SCALE = 0.001` — the STL is in **mm**; multiply by 0.001 to get **metres**.
  - `ENCODE = {'LR': 0, 'AP': 1}` — maps a design name to which coordinate column
    is the encoding axis (`LR` → x = column 0).
  - `TARGET_SCALE = 1e-3` — an overall multiplier on the target ramp. It does
    **not** move any geometry; it only rescales ψ so the reported per-turn current
    lands in a sane range (a few amps). Efficiency (mT/m/A) is invariant to it.
  - `SUBDIVIDE = 2` — how many times each STL triangle is split 4-ways.
  - `QUAD_SUBDIV = 2` — quadrature resolution for the Biot–Savart integral.
  - `WIRE_PITCH = 3.0e-3` — minimum center-to-center wire spacing (3 mm), used to
    cap the turn count.
  - `LAMBDA_REG = 1e-11` — Tikhonov regularization weight.
  - `INNER_TOWARD`, `LEAD_OUT_Y` — closed-solid inner-face target (unused here,
    this former is an open sheet) and the y-height where the lead wires exit.

The last top-level line is:

```python
if __name__ == '__main__':
    main()
```

so running the file calls `main()` with no arguments.

---

## 2. `main(names=None)`  (design_gradient.py)

```python
names = names or list(ENCODE.keys())
```
`names` is `None`, so it becomes `['LR', 'AP']` — by default it designs both the
Left–Right (x) and Anterior–Posterior (y) gradients. (In our run we passed
`['LR']` to do only x.)

```python
data = np.genfromtxt(FIELD_CSV, delimiter=',', skip_header=1)
```
Reads `magnetData.csv` into an array of shape `(N, 6)`, skipping the header row.
Columns are `x(mm), y(mm), z(mm), Bx, By, Bz` (Tesla). Here N = 1331 probe points.

```python
roi = data[:, :3] * 1e-3
```
The first three columns (positions in **mm**) are converted to **metres**. `roi`
is `(1331, 3)` — the cloud of points where the field was measured, and where we
will enforce the gradient.

```python
B0 = data[:, 3:6]
```
The measured field vector at each probe point, `(1331, 3)` in Tesla.

```python
bhat = B0 / np.linalg.norm(B0, axis=1, keepdims=True)
```
The **unit vector of the local main field** at each point, `(1331, 3)`. This is
the heart of the inhomogeneous-magnet approach: instead of asking the coil to
make a linear `Bz` (which assumes a perfectly z-aligned magnet), we ask it to
make a linear **B·b̂** — the component along whatever direction B₀ actually points
at each location. In this magnet b̂ tilts by up to ~46° from x across the ROI, so
this genuinely matters.

```python
print("ROI: ...")
```
Prints the count and the raw bounding box of `roi` (metres) so you can sanity
check units and extent.

```python
former = fm.load_former(STL_PATH, axis_map=AXIS_MAP, subdivide=SUBDIVIDE,
                        inner_toward=INNER_TOWARD, scale=SCALE)
```
Builds the mesh. **→ jump to section 3.**

```python
fm.check_alignment(former, roi)
```
Prints the former and ROI bounding boxes side by side. **→ section 4.**

```python
R = sg.resistance_matrix(former.verts, former.tris)
```
Builds the regularization matrix once (it does not depend on the target).
**→ section 5.**

```python
for name in names:
    axis = ENCODE[name]
    wire, metrics = design(name, axis, roi, bhat, former, R)
    plot_check(name, axis, wire, roi, former)
```
For each requested design, look up which coordinate column is the encoding axis
(`LR` → 0), then run the solve+contour+stitch+validate in `design()`
(**→ section 6**) and draw the diagnostic figure (**→ section 12**).

```python
print("done.")
```

---

## 3. `former_mesh.load_former(...)`

Turns the STL file into a clean, welded, consistently-oriented, refined triangle
mesh with a marked open boundary.

```python
raw = stlmesh.Mesh.from_file(path).vectors[..., list(axis_map)] * scale
```
`Mesh.from_file(path).vectors` is `(T, 3, 3)` — for each of T triangles, its 3
corners as xyz. `[..., list(axis_map)]` permutes the coordinate columns (identity
here). `* scale` converts mm → m. `raw` is every triangle's corners in metres, in
the probe frame. Note: STL stores each triangle independently, so shared corners
appear as *duplicate* coordinates — that's why we weld next.

```python
verts, tris = _weld(raw, weld_tol)
```
**→ section 3a.** Merges duplicate corners into shared vertices and returns an
indexed mesh.

```python
edge_count = defaultdict(int)
for a, b, c in tris:
    for u, w in [(a, b), (b, c), (c, a)]:
        edge_count[(min(u, w), max(u, w))] += 1
is_closed = all(n == 2 for n in edge_count.values())
```
Counts how many triangles use each undirected edge. On a **closed solid** every
edge is shared by exactly 2 triangles. On an **open sheet** the rim edges are
used by only 1. This detects which kind of STL you gave it.

```python
if is_closed:
    target = inner_toward if inner_toward is not None else (0.0, 0.0, 0.0)
    verts, tris = _inner_face(verts, tris, target)
```
If it were a closed solid, keep only the inner (cavity-facing) sheet
(**→ section 3e**). Our former is an open U-channel sheet, so `is_closed` is
False and this block is skipped.

```python
tris = _orient(tris)
```
**→ section 3b.** Makes all triangle windings consistent (all normals point the
same way across shared edges) — required so K = ∇ψ × n̂ is coherent.

```python
for _ in range(subdivide):
    verts, tris = _subdivide(verts, tris)
```
**→ section 3c.** Refines the mesh `subdivide` (=2) times. Each pass splits every
triangle into 4, so triangle count grows ×4 per pass (1266 → ~5k → ~20k). Finer
triangles = finer resolvable current detail.

```python
bnd = _boundary_mask(verts, tris)
```
**→ section 3d.** Marks vertices lying on the open rim. ψ is pinned to 0 there so
that every contour loop closes on the surface.

```python
if verbose:
    ... print vert/tri counts and bounding box ...
return FormerMesh(verts, tris, bnd)
```
`FormerMesh` is a tiny container holding `verts` `(V,3)`, `tris` `(T,3)` int, and
`boundary_mask` `(V,)` bool. For our run: 12665 verts (7593 free, 5072 boundary),
20256 triangles.

### 3a. `_weld(tri, tol=1e-5)`
```python
flat = tri.reshape(-1, 3)
```
All corners flattened to `(3T, 3)`.
```python
keys = np.round(flat / tol).astype(np.int64)
```
Quantizes coordinates onto a `tol`-sized grid (1e-5 m = 10 µm) so numerically
near-identical corners get identical integer keys.
```python
uniq, inv = np.unique(keys, axis=0, return_inverse=True)
```
`uniq` = the distinct corner keys; `inv` maps each of the 3T original corners to
its unique index.
```python
verts = np.zeros((len(uniq), 3))
np.add.at(verts, inv, flat)
verts /= np.bincount(inv)[:, None]
```
Averages the *actual* (unrounded) coordinates of all corners that collapsed to
each unique key — a robust vertex position, not just the rounded one.
```python
tris = inv.reshape(-1, 3)
degenerate = (tris[:,0]==tris[:,1]) | (tris[:,1]==tris[:,2]) | (tris[:,0]==tris[:,2])
return verts, tris[~degenerate]
```
Rebuilds triangles as index triples and drops any triangle that collapsed to a
line/point (two shared corners).

### 3b. `_orient(tris)`
Builds an edge→triangles adjacency map, then flood-fills from each unvisited
triangle across shared edges. The rule:
```python
if (a, b) in directed_edges(nb):
    tris[nb] = tris[nb][::-1]
```
Two triangles sharing an edge are consistently wound **iff** they traverse that
shared edge in *opposite* directions. If a neighbor traverses it the *same* way,
its winding is flipped (`[::-1]`). Result: a globally consistent orientation, so
every triangle's normal n̂ points to the same side of the sheet.

### 3c. `_subdivide(verts, tris)`
Classic 4-way midpoint refinement. For each triangle `(a,b,c)` it creates (or
reuses, via the `mid` cache keyed on the sorted vertex pair) the three edge
midpoints `ab, bc, ca`, then replaces the triangle with four:
```python
nt += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
```
The `mid` dictionary guarantees adjacent triangles share the same midpoint vertex
(no cracks).

### 3d. `_boundary_mask(verts, tris)`
Counts undirected edge usages again; any edge used by exactly one triangle is a
boundary edge, and both its endpoints are marked `True`. These are the vertices
where ψ = 0 will be enforced.

### 3e. `_inner_face(verts, tris, toward)` (not used for this former)
For a closed solid, keeps only triangles whose outward normal points *toward* the
cavity point, i.e. recovers the single copper-side sheet and discards the outer
shell, then re-indexes.

---

## 4. `former_mesh.check_alignment(former, roi_points, label="")`
Pure diagnostics — no geometry is changed. It prints the former's and the ROI's
bounding boxes, checks `ROI fully inside former bbox`, and computes the per-axis
size ratio to catch a mm/m unit mismatch (a mm-scale former would numerically
"contain" a m-scale ROI while being physically 1000× too big). In our run the
±100 mm ROI sits well inside the former extent — alignment is sane.

---

## 5. `stream_gradient.resistance_matrix(verts, tris)`

Builds the V×V matrix used purely for Tikhonov regularization (the penalty term
ψᵀRψ ≈ ∫|K|² dA, a stand-in for coil "effort"/power). It first calls
`triangle_geometry` (**→ section 5a**), then:

```python
R = np.zeros((n_verts, n_verts))
for t in range(tris.shape[0]):
    for a in range(3):
        for b in range(3):
            R[tris[t,a], tris[t,b]] += np.dot(kvec[t,a], kvec[t,b]) * area[t]
```
For each triangle and each pair of its local vertices (a,b), it adds
`(k_a · k_b) · area` into the global matrix at those vertices. Since the basis
current from vertex v in a triangle is `ψ_v · k_v` (constant over the triangle),
`∫ K·K dA` over the triangle expands to `Σ_{a,b} ψ_a ψ_b (k_a·k_b) area`, and
summing over triangles assembles R such that `ψᵀRψ = ∫|K|² dA`. It depends only
on geometry, so it is built once and reused for every design.

*(In the fast path, `fast_ops.resistance_matrix` computes the identical matrix
with `np.einsum` + scatter-add instead of the triple Python loop.)*

### 5a. `stream_gradient.triangle_geometry(verts, tris)`
This is the geometric core used everywhere. It returns per-triangle area, unit
normal, and the three "k vectors."
```python
p = verts[tris]                       # (T,3,3) the 3 corners of each triangle
p0, p1, p2 = p[:,0], p[:,1], p[:,2]
normal = np.cross(p1 - p0, p2 - p0)   # (T,3); |normal| = 2*area
two_area = np.linalg.norm(normal, axis=1)
area = 0.5 * two_area
nhat = normal / two_area[:, None]     # unit normal per triangle
```
Then the gradients of the barycentric coordinates:
```python
edge_opposite = np.stack([p2 - p1, p0 - p2, p1 - p0], axis=1)   # (T,3,3)
grad_lambda = np.cross(nhat[:, None, :], edge_opposite) / two_area[:, None, None]
```
For a linear triangle, the barycentric coordinate λ_i (which is 1 at vertex i and
0 at the others) has the constant in-plane gradient `∇λ_i = (n̂ × e_i)/(2·area)`,
where `e_i` is the edge **opposite** vertex i (for vertex 0 that's `p2−p1`, etc.).
So `grad_lambda[t,i]` = ∇λ_i on triangle t.
```python
k = np.cross(grad_lambda, nhat[:, None, :])   # (T,3,3)
return area, nhat, k
```
`k[t,i] = ∇λ_i × n̂`. Because ψ = Σ_i ψ_i λ_i, the surface current is
`K = ∇ψ × n̂ = Σ_i ψ_i (∇λ_i × n̂) = Σ_i ψ_i k[t,i]` — i.e. **k[t,i] is the
current density contributed per unit ψ at local vertex i**, constant over the
triangle. This single quantity feeds the sensitivity matrix, the resistance
matrix, the peak-current check, and the contour orientation.

---

## 6. `design_gradient.design(name, axis, roi, bhat, former, R, ...)`

This is where a specific gradient is solved and turned into a wire.

```python
C = sg.sensitivity_matrix(former.verts, former.tris, roi, quad_subdiv=QUAD_SUBDIV)
```
Builds the **sensitivity tensor** `C` of shape `(3, P, V)`: `C[c,p,v]` = the
`c`-th component of B at ROI point p produced per unit ψ at vertex v (Tesla per
Amp). **→ section 7.** This is the expensive step.

```python
P = bhat[:, 0:1] * C[0] + bhat[:, 1:2] * C[1] + bhat[:, 2:3] * C[2]
```
Projects the vector sensitivity onto the local field direction, giving `P` of
shape `(P_pts, V)`: `P[p,v]` = how much the **B·b̂** at point p changes per unit
ψ at vertex v. This collapses the 3-component field problem into the single
scalar quantity that actually shifts the Larmor frequency. So the forward model
is simply `g = P ψ`, where g is the projected field on the ROI.

```python
psi = sg.solve_stream(P, TARGET_SCALE * roi[:, axis], R, former.boundary_mask, lam)
```
Solves the regularized least-squares problem for the vertex stream function,
targeting a ramp proportional to the encoding coordinate `roi[:,axis]` (x for LR).
**→ section 8.**

```python
psi_range = psi.max() - psi.min()
```
The total swing of ψ. The number of turns times the current per turn equals this
range (each contour level "uses up" one slice of the range).

```python
peak_current_density = sg.max_surface_current(former.verts, former.tris, psi)
```
The largest |K| = |∇ψ| anywhere on the mesh. **→ section 9.**

```python
n_turns = max(int(psi_range / (wire_pitch * peak_current_density)), 1)
```
The key discretization choice. Adjacent contour lines of ψ are separated on the
surface by roughly `Δψ / |∇ψ|`. To keep even the *densest* region's wire spacing
≥ `wire_pitch`, we need `Δψ ≥ wire_pitch · peak|∇ψ|`. Since `Δψ = psi_range /
n_turns`, that bounds `n_turns ≤ psi_range / (wire_pitch · peak|K|)`. Taking the
floor (min 1) gives the most turns you can fit without wires colliding at the
tightest spot. Here: 41 turns.

```python
turn_current = psi_range / n_turns
```
The current each turn carries (= contour spacing Δψ). Here ≈ 1.267 A.

```python
levels = np.linspace(psi.min() + psi_range/(2*n_turns),
                     psi.max() - psi_range/(2*n_turns), n_turns)
```
The `n_turns` ψ-values to contour, placed at the *centers* of equal slices (half
a step in from each end) so the turns are evenly distributed and none sits exactly
on the min/max.

```python
contours = sg.contour_levels(former.verts, former.tris, psi, levels)
```
Marching-triangles contour of ψ at each level, returning oriented segments.
**→ section 10.**

```python
loops = []
for _, segs in contours:
    loops.extend(wiring.loops_from_segments(segs))
```
Chains the loose segments at each level into closed loops. **→ section 11a.**
`loops` is a flat list of closed polylines (48 loops in our run).

```python
wire = wiring.stitch_loops(loops, exit_near_y=LEAD_OUT_Y)
```
Bridges all the separate loops into **one continuous conductor** whose ends sit
near `y = LEAD_OUT_Y`. **→ section 11b.**

```python
lead_a = wire[0].copy();  lead_a[1] = LEAD_OUT_Y - 0.02
lead_b = wire[-1].copy(); lead_b[1] = LEAD_OUT_Y - 0.02;  lead_b[0] += 0.006
wire = np.vstack([lead_a, wire, lead_b])
```
Adds two short **lead wires**: it extends the first and last points out to
`LEAD_OUT_Y − 20 mm` (and nudges the second lead 6 mm in x so the two leads don't
overlap), i.e. the terminals where you'd solder the coil to your amplifier.

```python
B_wire = sg.biot_savart_wire([(np.stack([wire[:-1], wire[1:]], axis=1), turn_current)], roi)
```
**Honest validation.** It builds explicit straight segments between consecutive
wire points, all carrying `turn_current`, and computes the true field they
produce at every ROI point via exact finite-segment Biot–Savart. **→ section 13.**
This is *not* the same approximation used to solve; it's an independent check of
the real, discretized wire.

```python
projected = np.einsum('pc,pc->p', B_wire, bhat)
```
Projects that true field onto the local B₀ direction — the quantity that encodes
frequency.

```python
slope, r_squared = _fit_slope(projected, roi[:, axis])
```
Fits `projected ≈ slope·x + const` by least squares and reports R². **→ 14.**
`slope` is the achieved gradient (T/m); R² is how linear it actually is (0.996).

```python
wire_length = np.sum(np.linalg.norm(np.diff(wire, axis=0), axis=1))
```
Total conductor length (sum of segment lengths).

```python
out_csv = name + '_former_wire.csv'
np.savetxt(out_csv, np.column_stack([np.arange(len(wire)), wire * 1000]),
           header='point_index,x_mm,y_mm,z_mm', delimiter=',', comments='',
           fmt=['%d', '%.3f', '%.3f', '%.3f'])
```
Writes the full wire path as `point_index, x_mm, y_mm, z_mm`.

```python
efficiency = slope / turn_current
metrics = dict(...)
print("%s: axis=%d, %d turns @ %.4f A, ...")
return wire, metrics
```
`efficiency` = gradient per amp (mT/m/A) — the current-independent figure of
merit. Prints the summary line and returns the wire and metrics.

---

## 7. `stream_gradient.sensitivity_matrix(verts, tris, field_pts, quad_subdiv=2)`

Computes `C[c,p,v] = ∂B_c(field_pts[p]) / ∂ψ_v` — the field each vertex's basis
current makes at each ROI point.

```python
n_pts, n_verts = field_pts.shape[0], verts.shape[0]
C = np.zeros((3, n_pts, n_verts))
_, _, kvec = triangle_geometry(verts, tris)   # per-triangle k vectors
corners = verts[tris]
for t in range(tris.shape[0]):
    qp, qw = _subtriangle_quadrature(corners[t,0], corners[t,1], corners[t,2], quad_subdiv)
```
For triangle t, `_subtriangle_quadrature` (**→ 7a**) returns integration points
`qp` inside the triangle and their weights `qw` (weights sum to the triangle's
area).
```python
    diff = field_pts[:, None, :] - qp[None, :, :]   # (P,Q,3) = r − r'
    r_cubed = np.linalg.norm(diff, axis=2) ** 3
    r_cubed[r_cubed == 0] = np.inf                  # avoid /0 if a probe sits on the mesh
    integral = np.einsum('q,pqc->pc', qw, diff / r_cubed)   # (P,3)
```
This evaluates the geometric Biot–Savart integral `∫ (r−r')/|r−r'|³ dA'` over the
triangle by quadrature, for every field point p at once. `integral[p]` is a 3-vector.
```python
    for i in range(3):
        v = tris[t, i]
        contribution = MU0_4PI * np.cross(kvec[t, i][None, :], integral)   # (P,3)
        C[:, :, v] += contribution.T
return C
```
The Biot–Savart law for a surface current is
`B(r) = (μ₀/4π) ∫ K(r') × (r−r')/|r−r'|³ dA'`. Since the vertex-i basis current is
the constant vector `k[t,i]` per unit ψ, its field contribution is
`(μ₀/4π) · k[t,i] × integral`. `MU0_4PI = 1e-7`. Each contribution is accumulated
into column v of C (a vertex is shared by several triangles, so contributions add).

### 7a. `_subtriangle_quadrature(p0, p1, p2, subdivisions)`
Splits the triangle into `subdivisions²` small sub-triangles and returns each
sub-triangle's **centroid** as an integration point with weight = sub-triangle
area. For `subdivisions = 2` this yields 4 points, and their weights sum to the
full triangle area. More points → more accurate near-field integral, slower.

*(The fast path `fast_ops.sensitivity_matrix` produces the identical C by
batching all triangles through vectorized NumPy with a scatter-add over the
shared vertex axis, instead of the Python `for t` loop. Verified to machine
precision, ~10 s vs. several minutes.)*

---

## 8. `stream_gradient.solve_stream(P_proj, target, R, boundary_mask, lam)`

Solves for ψ. The problem:

```
minimize   ‖ P·ψ − target − c·1 ‖²  +  λ·ψᵀRψ
subject to ψ = 0 on boundary vertices
```

with a free scalar `c`. Line by line:
```python
free = np.where(~boundary_mask)[0]
```
Indices of the interior (free) vertices — the actual unknowns. Boundary vertices
are fixed at 0.
```python
P_free = P_proj[:, free]
R_free = R[np.ix_(free, free)]
```
Restrict the forward operator and the regularizer to the free unknowns.
```python
ones = np.ones((P_free.shape[0], 1))
A_aug = np.hstack([P_free, ones])
```
Append a column of ones. The extra unknown `c` (its coefficient) is a **free
constant** that absorbs the DC offset the projected field can't observe: only the
*slope* of the target ramp is physical, not its absolute level, so we let the fit
choose the offset instead of forcing it.
```python
R_aug = np.zeros((A_aug.shape[1], A_aug.shape[1]))
R_aug[:len(free), :len(free)] = R_free    # do NOT regularize c
```
The regularizer applies to ψ but not to `c`.
```python
M = A_aug.T @ A_aug + lam * R_aug
rhs = A_aug.T @ target
solution = np.linalg.solve(M, rhs)
```
These are the **normal equations** of the regularized least-squares problem
(`(AᵀA + λR)x = Aᵀb`). `M` is `(free+1)×(free+1)` and symmetric positive-definite;
`np.linalg.solve` factorizes and solves it. This dense solve on ~7600 unknowns is
the second heaviest step (~8 s).
```python
psi = np.zeros(P_proj.shape[1])
psi[free] = solution[:len(free)]
return psi
```
Scatter the solved interior values back into a full length-V vector (boundary
entries stay 0), dropping the `c` component. Result: ψ at every mesh vertex.

The role of `λ` (`LAMBDA_REG = 1e-11`): larger λ → smoother, lower-effort current
(fewer wiggles, easier to wind) at the cost of gradient fidelity; smaller λ →
tighter fit but noisier, denser wires. It trades off the data term against ∫|K|².

---

## 9. `stream_gradient.max_surface_current(verts, tris, psi)`

Finds peak |K| to size the turn count.
```python
area, nhat, _ = triangle_geometry(verts, tris)
p = verts[tris]
two_area = 2 * area
edge_opposite = np.stack([p[:,2]-p[:,1], p[:,0]-p[:,2], p[:,1]-p[:,0]], axis=1)
grad_lambda = np.cross(nhat[:,None,:], edge_opposite) / two_area[:,None,None]
grad_psi = np.einsum('tij,ti->tj', grad_lambda, psi[tris])
return np.linalg.norm(grad_psi, axis=1).max()
```
It recomputes the per-triangle `∇λ_i`, contracts with the triangle's three ψ
values to get the constant `∇ψ` per triangle, and returns the largest `|∇ψ|`.
Because `K = ∇ψ × n̂` and n̂ ⟂ ∇ψ with |n̂| = 1, `|K| = |∇ψ|` — so this is exactly
the peak surface current density. Feeds the `n_turns` formula in section 6.

---

## 10. `stream_gradient.contour_levels(verts, tris, psi, levels)`

Marching-triangles extraction of the wire paths.
```python
area, nhat, _ = triangle_geometry(verts, tris)
... grad_lambda ...
grad_psi = np.einsum('tij,ti->tj', grad_lambda, psi[tris])   # ∇ψ per triangle
current_dir = np.cross(grad_psi, nhat)                       # K direction per triangle
```
Precomputes the local current direction `K = ∇ψ × n̂` in every triangle, used to
orient segments so current flows consistently.
```python
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
```
For each triangle, subtract the level from the three vertex ψ values. Walk the
three edges; wherever ψ − level changes sign across an edge, the contour crosses
it, and linear interpolation gives the crossing point (`f` is the fractional
position). A triangle with a contour passing through it has exactly 2 such
crossings.
```python
        if len(crossings) == 2:
            seg = np.array(crossings)
            if np.dot(seg[1] - seg[0], current_dir[ti]) < 0:
                seg = seg[::-1]
            segs.append(seg)
    out.append((level, segs))
return out
```
Form the segment from the 2 crossings, and **flip it if needed** so it points
along `current_dir` (the physical current direction). This orientation is what
lets `loops_from_segments` chain them and lets the coil reproduce the surface
current with the correct sign. Returns a list of `(level, [segments])`.

---

## 11. Wiring: from segments to one continuous conductor

### 11a. `wiring.loops_from_segments(segments, tol=1e-7)`
Chains an unordered pile of oriented `(2,3)` segments into closed polylines.
```python
def key(point): return tuple(np.round(point / tol).astype(np.int64))
start_index = defaultdict(list)
for i, seg in enumerate(segments):
    start_index[key(seg[0])].append(i)
```
Builds a lookup from a quantized start point to the segments starting there.
```python
for i0 in range(len(segments)):
    if used[i0]: continue
    chain = [segments[i0][0], segments[i0][1]]
    ...
    while True:
        candidates = [j for j in start_index[key(current)] if not used[j]]
        if not candidates: break
        j = candidates[0]; used[j] = True
        chain.append(segments[j][1]); current = segments[j][1]
        if key(current) == key(chain[0]): break
    if len(chain) >= 4:
        loops.append(np.array(chain))
```
Starting from an unused segment, it repeatedly finds the segment whose start
matches the current end and appends it, until it returns to the loop start.
Chains shorter than 4 points (a level that barely clips the mesh edge) are
discarded. Output: a list of closed loops for that level.

### 11b. `wiring.stitch_loops(loops, exit_near_y)`
```python
neighbors = build_bridge_tree(loops)
root = int(np.argmin([np.min(np.abs(loop[:,1] - exit_near_y)) for loop in loops]))
entry_idx = int(np.argmin(np.abs(loops[root][:,1] - exit_near_y)))
return traverse_to_single_wire(loops, neighbors, root, loops[root][entry_idx])
```
Builds a tree connecting the separate loops (**→ 11c**), chooses as the **root**
the loop that comes closest to the chosen lead-out height `exit_near_y`, and
walks the tree into a single path (**→ 11e**) entering at that closest point. The
result's two ends therefore sit near the terminal location.

### 11c. `wiring.build_bridge_tree(loops)`
Greedy minimum-spanning-tree (Prim's algorithm) over the loops, using the
loop-to-loop nearest-point distance as edge weight:
```python
in_tree = {0}
while len(in_tree) < len(loops):
    # find the closest (in-tree loop a, out-of-tree loop b) pair
    ... ia, jb, dist = _nearest_pair(loops[a], loops[b]) ...
    # add the cheapest such b, record the bridge points
```
It returns an adjacency dict: for each loop, its tree neighbors together with the
**specific point index on each loop** where they should be bridged. Physically,
these bridges are the short cross-overs that connect one turn to the next.

### 11d. `wiring._nearest_pair(loop_a, loop_b, n_sample=60)`
A cheap approximate closest-point finder: subsample both loops to ≤60 points,
compute the pairwise distances, take the minimum, then among near-ties break
deterministically toward smaller y and map back to the closest *actual* points on
each full loop. Returns `(index_in_a, index_in_b, distance)`.

### 11e. `wiring.traverse_to_single_wire(loops, neighbors, root, entry_point)`
Depth-first traversal that splices every loop into one polyline:
```python
def recurse(node, enter_near):
    loop = loops[node]
    k = argmin distance(loop − enter_near)
    rotated = np.vstack([loop[k:], loop[:k]])   # start this loop at its entry point
    # for each not-yet-visited child, find where on THIS loop to branch off,
    # sort those branch points by position along the loop, and at each one:
    #   emit loop points up to the branch, hop to the child's bridge point,
    #   recurse into the child, hop back, continue the loop
    ...
    out.append(rotated[0])   # close the loop back to its start
```
Walking the loop, at each bridge point it detours into the child loop (fully
traversing it via recursion) and returns, so the entire tree becomes **one
continuous, non-self-retracing path** that visits every loop exactly once.
Returns the `(N,3)` wire.

---

## 12. `design_gradient.plot_check(name, axis, wire, roi, former)`

Diagnostics only. Creates a 3-panel figure:
- **3D**: the former surface (`plot_trisurf`, translucent grey) with the wire
  (green) and the ROI points (red).
- **x–z**: the wire projected to the x–z plane, plus the ROI bounding box.
- **y–z**: the wire projected to the y–z plane, plus the ROI box.

`roi_box(ax, i, j)` just draws the min/max rectangle of the ROI in the chosen two
coordinates. Saves `name + '_check.png'` at 140 dpi. For an x-gradient, the y–z
panel shows the nested, y-symmetric loop pattern characteristic of an LR gradient.

---

## 13. `stream_gradient.biot_savart_wire(segments_with_current, field_pts)`

Exact field of the **discrete** wire, used for validation (no quadrature
approximation).
```python
for segs, I in segments_with_current:
    a, b = segs[:,0,:], segs[:,1,:]
    for n in range(segs.shape[0]):
        r1 = field_pts - a[n];  r2 = field_pts - b[n]
        r1n = norm(r1); r2n = norm(r2)
        cross = np.cross(r1, r2)
        denom = r1n*r2n*(r1n*r2n + einsum('pc,pc->p', r1, r2))
        factor = (r1n + r2n) / denom
        B += MU0_4PI * I * cross * factor[:, None]
```
This is the closed-form field of a **finite straight segment** carrying current
I: for endpoints a,b and field point r, with r1 = r−a, r2 = r−b, the field is
`(μ₀I/4π) · (r1×r2)·(|r1|+|r2|) / ( |r1||r2|(|r1||r2| + r1·r2) )`. Summed over all
segments and both leads, this is the true field the manufactured winding would
produce. *(The fast path chunks this over segments with vectorized NumPy — needed
because the raw contoured wire has ~377k segments — giving an identical result;
for validation we run it on the 0.5 mm-decimated path.)*

---

## 14. `design_gradient._fit_slope(field_values, axis_values)`
```python
design_matrix = np.column_stack([axis_values, np.ones_like(axis_values)])
coeffs, _, _, _ = np.linalg.lstsq(design_matrix, field_values, rcond=None)
residual = field_values - design_matrix @ coeffs
ss_res = np.sum(residual ** 2)
ss_tot = np.sum((field_values - field_values.mean()) ** 2)
r_squared = 1 - ss_res / ss_tot
return coeffs[0], r_squared
```
Ordinary least-squares line fit of the projected field against the encoding
coordinate. `coeffs[0]` is the slope (the achieved gradient, T/m); `r_squared`
is the standard coefficient of determination, i.e. the fraction of the field's
variation explained by a straight line in x — the linearity of the gradient.

---

## 15. End-to-end data flow (one glance)

```
magnetData.csv ──► roi (P×3, m), B0 (P×3, T) ──► b̂ (P×3, unit)
raw_data_former.stl ─(weld, orient, subdivide ×2, boundary)─► mesh: verts(V×3), tris(T×3), bnd(V)

triangle_geometry(mesh) ─► area, n̂, k        (constant surface-current basis per triangle)
        │
        ├─► resistance_matrix ─► R (V×V)                       [regularizer, built once]
        └─► sensitivity_matrix(mesh, roi) ─► C (3×P×V)          [field per unit ψ]

C · b̂  ─► P (P×V)                                              [projected forward model g = Pψ]
solve_stream(P, TARGET_SCALE·x_roi, R, bnd, λ) ─► ψ (V)         [regularized least squares]

max|∇ψ| ─► n_turns, turn_current, levels
contour_levels(mesh, ψ, levels) ─► oriented segments
loops_from_segments ─► closed loops ─► stitch_loops ─► single wire (+leads)

biot_savart_wire(wire, turn_current) at roi ─► B_wire ─► B_wire·b̂
_fit_slope(B_wire·b̂, x_roi) ─► gradient (slope), R²
                              ─► efficiency = slope / turn_current
save *_former_wire.csv ; plot_check ─► *_check.png
```

**Result for this run (LR / x):** 41 turns @ 1.267 A, gradient 0.937 mT/m,
efficiency 0.739 mT/m/A, R² = 0.996, wire length ≈ 75.7 m.
