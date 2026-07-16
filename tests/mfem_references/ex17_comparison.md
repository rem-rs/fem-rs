# ex17 DG SIP Linear Elasticity: C++/Rust Sequence Comparison

**C++ source**: `/home/quan/works/mfem/examples/ex17.cpp` (MFEM 4.9)
**Rust source**: `fem-rs/examples/mfem_ex17_dg_elasticity.rs`
**DG assembler**: `fem-rs/crates/assembly/src/dg/dg_elasticity.rs`
**Date**: 2026-07-16

---

## Pipeline Overview

```
C++ MFEM ex17                          Rust mfem_ex17_dg_elasticity.rs
─────────────────────                  ─────────────────────────────────
1. CLI args (mesh,ref,order,α,κ,vis)  1. CLI args (mesh,ref,order,α,κ)  ❌ no -vis flag
2. Mesh(mesh_file,1,1)                2. read_mfem_file → Mesh2d        ✔ same
3. UniformRefinement (auto ≈5k)       3. refine_uniform (auto ≈5k)     ✔ same formula
4. mesh.SetCurvature(order)           4. (skipped)                      ❌ no NURBS→curved
5. DG_FECollection(GL)                5. L2Space (default basis)        ❌ diff basis family
   FiniteElementSpace(dim)                                                ✔ vector space
6. dir_bdr markers (attrs 1,2)        6. dirichlet_attrs [1,2]          ✔ same
7. x.ProjectCoefficient(init_x)       7. x = zero                       ❌ no initial guess
8. lambda/mu PWConstCoefficient       8. per-elem lambda/mu arrays      ✔ equivalent
9. b.AddBdrFaceIntegrator(             9. assemble_dg_elasticity_       ❓ hand-coded RHS
   DGElasticityDirichletLFIntegrator)     dirichlet_rhs()
10.a.AddDomainIntegrator(             10. assemble_volume()             ❓ hand-coded volume
     ElasticityIntegrator)
   a.AddInteriorFaceIntegrator(          assemble_interior_face_stress() ❓ hand-coded interior
     DGElasticityIntegrator)
   a.AddBdrFaceIntegrator(               assemble_boundary_face_stress() ❓ hand-coded boundary
     DGElasticityIntegrator, dir_bdr)
11.a.FormLinearSystem(→A,X,B)         11. (skip FormLinearSystem)       ❌ no FormLinearSystem
12.PCG (α=-1) or GMRES (α≠-1)         12. PCG only                     ❌ no GMRES path
   GSSmoother preconditioner              solve_pcg_gssmoother          ✔ same precond
13.a.RecoverFEMSolution               13. (skip)                       ❌ no recover
14.mesh.SetNodalFESpace(&fespace)     14. (skip)                       ❌ no nodal space
15.Save displaced.mesh + sol.gf       15. Output ||u||_L2 + checksum   ❌ different output
16.GLVis stress visualization         16. (skip)                       ❌ no viz
```

---

## Step-by-Step Detailed Comparison

### Step 1: CLI Arguments

| Parameter | C++ | Rust | Match |
|-----------|-----|------|-------|
| `-m` / `--mesh` | `../data/beam-tri.mesh` | `data/beam-tri.mesh` (via CARGO_MANIFEST_DIR) | ✔ (path resolution differs) |
| `-r` / `--refine` | int, default -1 | i32, default -1 | ✔ |
| `-o` / `--order` | int, default 1 | u8, default 1 | ✔ |
| `-a` / `--alpha` | real_t, default -1.0 | f64, default -1.0 | ✔ |
| `-k` / `--kappa` | real_t, default -1.0 | f64, default -1.0 | ✔ |
| `-vis` / `-no-vis` | bool, default true | ❌ missing | ❌ |

**Issue**: Rust CLI has no `-vis`/`-no-vis` flag. C++ defaults `visualization=1`.

### Step 2: Mesh Reading

```cpp
// C++
Mesh mesh(mesh_file, 1, 1);  // generate_edges=1, refine=1
int dim = mesh.Dimension();
```

```rust
// Rust
let mfem = read_mfem_file(mesh_file).expect("...");
let mesh = mfem.mesh2d.expect("...");
let dim = 2;
```

- C++ `refine=1` parameter affects initial mesh — **verify if this is needed**.
- Rust hardcodes `dim = 2`; C++ uses `mesh.Dimension()`. For beam-tri.mesh both are 2D.

### Step 3: Uniform Refinement

Same formula: `floor(log(5000/nelems) / log(2) / dim)` — **confirmed matching**.

### Step 4: NURBS → Curved Conversion (MISSING in Rust)

```cpp
// C++
if (mesh.NURBSext) { mesh.SetCurvature(order); }
```

Rust has no equivalent. For non-NURBS meshes (beam-tri.mesh, beam-quad.mesh) this is a no-op.

### Step 5: DG Finite Element Space ⚠️ CRITICAL DIFFERENCE

```cpp
// C++ — Gauss-Lobatto basis
DG_FECollection fec(order, dim, BasisType::GaussLobatto);
FiniteElementSpace fespace(&mesh, &fec, dim);
```

```rust
// Rust — default basis (Gauss-Legendre?)
let space = L2Space::new(mesh.clone(), order);
let n_scalar = space.n_dofs();
let n_total = dim * n_scalar;
```

**Critical**: C++ uses **Gauss-Lobatto** nodal basis (`BasisType::GaussLobatto`). This gives:
- Nodes include boundary points at ±1
- Sparser matrix (fewer off-diagonal couplings)
- Different basis values than Gauss-Legendre

Rust `L2Space` likely uses **Gauss-Legendre** or equally-spaced nodal basis. The comment at line 82-84 acknowledges this:
```rust
// L2Space currently uses default basis; Gauss-Lobatto not directly available via this API.
```

**Impact**: Even with identical assembly code, matrix values and solution will differ due to basis mismatch.

### Step 6: Dirichlet Boundary Markers ✔ Equivalent

C++:
```cpp
Array<int> dir_bdr(mesh.bdr_attributes.Max());
dir_bdr = 0;
dir_bdr[0] = 1;  // attribute 1
dir_bdr[1] = 1;  // attribute 2
```

Rust:
```rust
let dirichlet_attrs = [1, 2];
```

Both mark boundary attributes 1 and 2 for weak Dirichlet BC. **Equivalent**.

### Step 7: Initial Guess ❌ MISSING in Rust

```cpp
// C++
GridFunction x(&fespace);
VectorFunctionCoefficient init_x(dim, InitDisplacement);
x.ProjectCoefficient(init_x);
// ...
a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
// X is NOT zero — it starts from projected InitDisplacement
```

```rust
// Rust
let mut x = vec![0.0_f64; n_total];
// Starts from zero
```

**Important difference**: C++ projects the Dirichlet BC onto the solution as an **initial guess** via `FormLinearSystem`. The iterative solver in C++ starts from `X ≈ u_D` while Rust starts from `X = 0`.

Initial residual is therefore very different:
- C++: `r0 = B - A*X_init` where `X_init ≈ u_D`  
- Rust: `r0 = B` (since X=0)

This affects iteration count but should **not** affect the final converged solution (assuming well-posed system).

### Step 8: Material Constants (Lambda/Mu) ✔ Equivalent

C++ uses `PWConstCoefficient` (attribute-based):
```cpp
Vector lambda(mesh.attributes.Max());
lambda = 1.0;
lambda(0) = 50.0;  // attribute 1 → λ=50
```

Rust uses per-element arrays:
```rust
let mut lambda_elem = vec![1.0_f64; n_elem];
for e in mesh.elem_iter() {
    if attr == 1 { lambda_elem[e] = 50.0; }
}
```

MFEM's `PWConstCoefficient` interpolates from attribute-value arrays. Rust maps attribute 1 to per-element indices. These produce the **same values at integration points** (assuming correct attribute→element mapping in the mesh).

### Step 9: RHS Assembly — Dirichlet BC ⚠️ NEEDS VERIFICATION

**C++** (MFEM library):
```cpp
b.AddBdrFaceIntegrator(
    new DGElasticityDirichletLFIntegrator(
        init_x, lambda_c, mu_c, alpha, kappa), dir_bdr);
b.Assemble();
```

The MFEM `DGElasticityDirichletLFIntegrator` assembles the RHS from weak Dirichlet BC. It computes:
```
L(v) = ∫ (κ/h)·u_D·v − α·σ(v)·n·u_D  ds
```

(The consistency term `σ(u_D)·n·v` is handled by the matrix boundary integrator and applied via `FormLinearSystem` which combines A and b.)

**Rust** (hand-coded):
```rust
fn assemble_dg_elasticity_dirichlet_rhs(...) {
    // For each Dirichlet boundary face:
    // Penalty: -(κ/h) · u_D_comp · φ_a
    // Symmetry: +α · Σ_i (σ(φ_a·e_comp)·n)_i · u_D_i
}
```

Sign convention:
- Rust penalty: `rhs -= w_f * pen * phi_a * u_d` → negative sign
- Rust symmetry: `rhs += w_f * alpha * dot` where `dot = Σ_i sn_flux[comp][i] * u_D_i`

**Need to verify** against MFEM's `DGElasticityDirichletLFIntegrator` source. The `FormLinearSystem` in C++ combines A and b, effectively computing `B = b - A*X_init` where `X_init` is the projected Dirichlet BC. The Rust code does NOT do this subtraction — it assembles RHS directly.

### Step 10: Matrix Assembly ⚠️ NEEDS FORMULA VERIFICATION

**Volume term:**

C++ uses `ElasticityIntegrator(lambda_c, mu_c)`:
- Standard linear elasticity kernel: `∫ 2μ·ε(u):ε(v) + λ·div(u)·div(v) dx`

Rust `assemble_volume()`:
```rust
// Block-diagonal: μ·δᵢⱼ·∇φ_a·∇φ_b (i=j only)
coo.add(row, dofs[b]*dim+i, w * mu * nabla_ab);
// Cross + div-div: μ·∂ⱼφ_a·∂ᵢφ_b + λ·∂ᵢφ_a·∂ⱼφ_b
val = mu * ga[j] * gb[i] + lam * ga[i] * gb[j];
```

The formula is:
```
K[(a,i),(b,j)] = ∫ [μ·δᵢⱼ·∇φ_a·∇φ_b + μ·∂ⱼφ_a·∂ᵢφ_b + λ·∂ᵢφ_a·∂ⱼφ_b] dx
```

The standard elasticity kernel is:
```
∫ 2μ·ε(u):ε(v) + λ·div(u)·div(v) dx
  = ∫ μ·(∂ᵢuⱼ·∂ᵢvⱼ + ∂ᵢuⱼ·∂ⱼvᵢ) + λ·∂ᵢuᵢ·∂ⱼvⱼ dx
```

Using u = φ_b·eⱼ (so uₖ = φ_b·δⱼₖ, ∂ᵢuₖ = ∂ᵢφ_b·δⱼₖ) and v = φ_a·eᵢ:
```
∫ μ·(∂ₖφ_a·∂ₖφ_b·δᵢⱼ + ∂ᵢφ_a·∂ⱼφ_b) + λ·∂ᵢφ_a·∂ⱼφ_b dx
```

So: `μ·δᵢⱼ·∇φ_a·∇φ_b + μ·∂ⱼφ_a·∂ᵢφ_b + λ·∂ᵢφ_a·∂ⱼφ_b`

This matches the Rust code. **✔ Formula matches**.

**Interior face term:**

C++ uses `DGElasticityIntegrator(lambda_c, mu_c, alpha, kappa)` — AssembleBlock builds the consistency term, then:
```cpp
elmat := -elmat + alpha*elmat^T + jmat
```

Rust `assemble_interior_face_stress()` builds four blocks (LL, LR, RL, RR) directly using the SIP stress formulation:
```
K += −∫ {σ(u)·n}·[[v]] − α∫ {σ(v)·n}·[[u]] + ∫ (κ/h)[[u]]·[[v]] ds
```

LL block:
```rust
t1 = -0.5 * snr[b][j][i] * phi_l[a];   // -{σ(u)·n}·[[v]]
t2 = 0.5 * alpha * snl[a][i][j] * phi_l[b];  // +α{σ(v)·n}·[[u]]
t3 = pen * phi_l[a] * phi_l[b] * δᵢⱼ;    // penalty
```

**Need to verify**: Sign conventions between MFEM's `elmat := -elmat + alpha*elmat^T + jmat` and Rust's direct SIP formula.

For the (1,1) block with α=-1:
- C++: `elmat = -consistency + (-1)*consistency^T + jmat = -(consistency + consistency^T) + jmat`
- Rust: `t1 + t2 + t3 = -0.5*snr + 0.5*(-1)*snl + pen` ... wait, α=-1 so t2 = -0.5*snl, and t1 = -0.5*snr

Actually for the LL block of the Rust formulation: the normal computation means snl and snr may differ in sign because σ·n involves the normal and `n_L = -n_R` on interior faces. So the averaging in the consistency term and its interaction with the sign needs careful verification.

**Boundary face term:**

Rust `assemble_boundary_face_stress()`:
```rust
t1 = -sn[b][j][i] * phi[a];
t2 = alpha * sn[a][i][j] * phi[b];
t3 = pen * phi[a] * phi[b] * δᵢⱼ;
```

This gives: `K_bdr = -σ(φ_b·e_j)·n)_i·φ_a + α·σ(φ_a·e_i)·n)_j·φ_b + (κ/h)·φ_a·φ_b·δᵢⱼ`

The MFEM version (no element 2) gives the same by computing `elmat = consistency` then `elmat = -elmat + alpha*elmat^T + jmat`:
```
K_bdr = -consistency + alpha*consistency^T + penalty
```
where `consistency = σ(φ_b·e_j)·n)_i·φ_a`.

So:
```
K_bdr = -σ(u)·n·v + α·σ(v)·n·u + penalty
```

This matches the Rust formula. **✔ Boundary formula matches**.

**But**: The C++ `no2 = False` path does NOT multiply `w` by 1/2 (interior faces use `w = ip.weight/2`, boundary uses `w = ip.weight`). Rust uses `w_f = q_face.weights[qi] * h_f` for both interior and boundary faces — **different weight computation**.

### Step 11: FormLinearSystem ❌ MISSING in Rust

```cpp
// C++
a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
```

This does three things:
1. `X = x` (copy initial guess from projected GridFunction)
2. `B = b` (copy RHS)
3. **Modifies system** for essential BC — but since `ess_tdof_list` is empty, no modification happens

Since `ess_tdof_list` is empty (all Dirichlet is weak in DG), FormLinearSystem is essentially:
```
X = x  (initial guess)
B = b  (rhs from DirichletLFIntegrator)
A = a  (matrix from ElasticityIntegrator + DGElasticityIntegrator)
```

Rust skips this entirely, starting from zero initial guess.

### Step 12: Solver ⚠️ INCOMPLETE

**C++**:
```cpp
if (alpha == -1.0)
    PCG(A, M, B, X, 3, 5000, rtol*rtol, 0.0);
else
    GMRES(A, M, B, X, 3, 5000, 100, rtol*rtol, 0.0);
```

**Rust**:
```rust
// Always PCG, no GMRES path
let res = solve_pcg_gssmoother(&a_mat, &rhs, &mut x, &cfg);
```

- For `α = -1` (symmetric SIP): both use PCG ✔
- For `α ≠ -1` (non-symmetric): C++ uses GMRES, Rust still uses PCG ❌
- C++ tolerances: `rtol*rtol = 1e-12`, `atol = 0.0`
- Rust: `rtol: 1e-12`, `atol: 0.0` — ✔ matches
- C++ max_iter: 5000, Rust: 5000 — ✔
- C++ print_level: "3" (print every 3rd), Rust: `verbose: false` — different but irrelevant

### Step 14-16: Post-processing ❌ MISSING in Rust

| Feature | C++ | Rust | |
|---------|-----|------|---|
| `RecoverFEMSolution` | ✔ | ❌ | Essential for correct GridFunction extraction |
| `mesh.SetNodalFESpace` | ✔ | ❌ | Needed for displaced mesh output |
| displaced.mesh output | ✔ | ❌ | |
| sol.gf output | ✔ | ❌ | |
| GLVis visualization | ✔ | ❌ | |
| Stress component calc | ✔ | ❌ | `StressCoefficient` |

---

## Normal Computation ⚠️ POTENTIAL DIFFERENCE

**C++** (MFEM):
```cpp
CalcOrtho(Trans.Jacobian(), nor);
```
For 2D, `CalcOrtho` on the 2×1 face Jacobian gives:
```
nor(0) =  J(1,0) = dy/dξ
nor(1) = -J(0,0) = -dx/dξ
```
This normal is **NOT normalized** — its length `|nor| = sqrt((dx/dξ)² + (dy/dξ)²)` is the physical-to-reference edge length ratio.

Then `ip.weight * |nor|` gives the physical quadrature weight.

**Rust**:
```rust
fn face_geom_2d(...) -> (f64, Vec<f64>) {
    let len = dx.hypot(dy);  // physical edge length
    (len, vec![-dy/len, dx/len])  // UNIT normal
}
```

Rust's normal is **unit-normalized** and `h_f = len` is the edge length.

**Penalty scale**: In C++, the penalty coefficient is:
```
jmatcoef = κ * (nor·nor) * wLM
```

The face Jacobian `nor` has length equal to the edge length in physical space (for 2D), so `nor·nor = len²`.

The physical quadrature weight is `ip.weight * |nor| = ip.weight * len`.

And `wLM = (λ+2μ) * w / Weight_elem` where `w = ip.weight * len` for boundary faces.

Plus `1/Weight_elem` ≈ `1/det(J_elem)` which is the 2D element area Jacobian determinant.

So C++ penalty at QP = `κ * len² * (λ+2μ) * ip.weight * len / det(J_elem)` = `κ * (λ+2μ) * ip.weight * len³ / det(J_elem)`

Rust penalty at QP = `κ * (λ+2μ) / h_f * w_f` where `w_f = q_face.weights[qi] * h_f`:
= `κ * (λ+2μ) / len * q_w * len` = `κ * (λ+2μ) * q_w`

C++ face QP weight uses `ip.weight` (weight on reference face). With Gauss-Lobatto quadrature on [-1,1], the ip.weight sum is 2 (the length of the reference interval). Rust's `q_face.weights[qi]` are quadrature weights on [-1,1] with same sum.

**This is a key difference**: The C++ penalty factor includes `len² * len / det(J)` = `len³/det(J)` which is NOT simply `1/len`. The Rust penalty uses `1/len * len = 1` factor from the face weight times the `1/h_f` in the penalty.

Actually wait, let me reconsider. In C++ MFEM:
- `nor` is not the face Jacobian. Let me look at CalcOrtho again for a face element transformation...

Actually, `Trans.Jacobian()` for a face element transformation gives the Jacobian of the transformation from the reference face (1D) to the physical face (2D). This is a 2×1 matrix:
```
J = [dx/dξ; dy/dξ]
```

The "normal" computed by CalcOrtho for 2D is:
```
nor = [dy/dξ, -dx/dξ]^T
```

But this is NOT the unit normal. It's the edge derivative rotated by 90°.

The physical length of the edge element at the integration point is `|J| = sqrt((dx/dξ)² + (dy/dξ)²)`.

The reference face integration in MFEM uses `∫_{face_ref} f(x(ξ)) * |J(ξ)| dξ`.

The quadrature weight `ip.weight` already includes `dξ` (the reference coordinate weight).

So the physical integration weight at the QP is `ip.weight * |J(ξ)|`.

Now, `nor·nor = (dy/dξ)² + (-dx/dξ)² = (dx/dξ)² + (dy/dξ)² = |J(ξ)|²`.

And `w = ip.weight` for boundary or `ip.weight/2` for interior.
`w1 = w / Trans.Elem1->Weight()` where `Trans.Elem1->Weight()` = det(J_elem) evaluated at the QP mapped from the face.

So for penalty on boundary:
```
jmatcoef = κ * |J|² * [(λ+2μ) * ip.weight / det(J_elem)]
```

And jmat adds `jmatcoef * shape * shape` to the (d,d) block.

So the `jmatcoef * shape * shape` contribution integrated over the face becomes:
```
∫ κ * |J|² * (λ+2μ) / det(J_elem) * φ_a * φ_b * dξ
```

The full penalty term should be `∫ (κ/h_f) * (λ+2μ) * u · v ds` on the physical face.

`h_f` is typically the face size. MFEM uses `h_f = 1/|J|` ... no, I think the h in the MFEM penalty is actually encoded in the `jmatcoef`.

Actually, in MFEM's approach, the h is already factored into the relationship between the reference and physical face. The term `wLM` has dimensions of `(λ+2μ) * dξ / det(J_elem)`, and `jmatcoef = κ * |J|² * wLM`.

The multiplication by `shape * shape` and summation uses the quadrature weight `w_q` (from `ir`).

But wait — the face integration already uses the physical face measure through the trans Jacobian. Let me look at how MFEM integrates face terms...

Actually, I think I'm overcomplicating this. In MFEM's `FaceElementTransformations`, the integration point `ip` is in the **reference face** coordinate. The face Jacobian `Trans.Jacobian()` maps from reference face to physical face. The normal `nor` computed by CalcOrtho also absorbs this Jacobian.

The key point is that in MFEM's face integration, the `ip.weight` is the Gauss weight in the **reference** domain, and the physical face measure is accounted for by the `Trans` Jacobian determinant (which is `|nor| = sqrt(nor·nor)`).

So `w * ip.weight` (with the weight factored through `Trans.Elem1->Weight()`) is NOT exactly the same as Rust's `h_f * q_face.weights[qi]`.

**This is probably the biggest source of numerical difference between C++ and Rust**.

---

## Summary of Issues by Severity

### Critical (affects numerical values)
1. **Basis function mismatch**: Gauss-Lobatto (C++) vs default (Rust) — different nodal basis
2. **Normal/penalty weighting**: MFEM's CalcOrtho-based normal with Jacobian scaling vs Rust's unit-normal + edge length

### Significant (may affect values)
3. **RHS assembly sign convention**: Need to verify `assemble_dg_elasticity_dirichlet_rhs` vs `DGElasticityDirichletLFIntegrator`
4. **Interior face formula**: Verify SIP stress formula against MFEM's `AssembleBlock` + `elmat := -elmat + alpha*elmat^T + jmat`

### Moderate (affects iteration count only)
5. **Initial guess**: C++ uses projected Dirichlet BC, Rust uses zero
6. **No GMRES path**: Rust always uses PCG (only works for α=-1)

### Missing features (no numerical impact on core solve)
7. No NURBS→curved conversion
8. No `FormLinearSystem` (empty ess_tdof makes it a no-op anyway)
9. No `RecoverFEMSolution`
10. No displaced mesh / solution file output
11. No GLVis visualization
12. No stress coefficient computation

---

## Verification Strategy

To 1:1 match C++ values, prioritize:

1. **Basis**: Implement Gauss-Lobatto for L2Space, or verify that the current default basis produces same values for order=1 on triangles (with Gauss-Lobatto, order=1 means equally-spaced nodes at ±1 on each edge, which is the same as equally-spaced on triangles).

2. **Penalty formula**: Compare `pen * phi * phi` contributions:
   - C++: `jmatcoef = kappa * (nor*nor) * wLM` with MFEM's CalcOrtho
   - Rust: `pen = kappa * (lam+2*mu) / h_f` with unit normal and edge length

3. **Quadrature rule**: Verify Rust's face quadrature uses the same order and rule as C++ (`2*max(order1, order2)`).

4. **RH**: Verify sign convention by testing with known Dirichlet BC (e.g., u_D = 0 → RHS should be zero).
