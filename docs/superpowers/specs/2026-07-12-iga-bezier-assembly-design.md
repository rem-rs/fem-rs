# IGA Bezier Element-by-Element Assembly

**Date**: 2026-07-12
**Status**: Draft
**Author**: Claude (superpowers:brainstorming → writing-plans)

## Overview

Replace the Cox-de Boor B-spline evaluation inside IGA assembly with **Bernstein basis evaluation + extraction operator** (Borden et al., CMAME 2011). This is the foundational step for efficient IGA element-by-element assembly on both CPU and GPU.

## Motivation

The existing IGA assembly (`crates/assembly/src/iga/iga.rs`) evaluates NURBS basis functions at each quadrature point via Cox-de Boor recursion. This is:

1. **O(p²) per direction per point** — unnecessary when extraction is precomputed once per element.
2. **Not GPU-friendly** — Cox-de Boor is a sequential triangular scheme.
3. **Not element-local** — global basis indexing mixes with assembly scatter.

Bezier extraction solves all three: precompute `C_e` for each element (one-time), then assembly evaluates Bernstein polynomials (O(p) per direction per point) and applies `C_e^T` (dense `(p+1)ᵈ × (p+1)ᵈ` matrix multiply).

## Changes

### A. Fix `compute_extraction_2d` — 2 lines changed

**File**: `crates/element/src/bezier_extraction.rs` (L194-L196)

Replace calls to `compute_extraction_1d` (identity-only, no non-uniform support) with `compute_extraction_1d_full` (Chebyshev point-matching for any knot vector):

```diff
- let ext_u = compute_extraction_1d(&pd.kv_u)?;
- let ext_v = compute_extraction_1d(&pd.kv_v)?;
+ let ext_u = compute_extraction_1d_full(&pd.kv_u)?;
+ let ext_v = compute_extraction_1d_full(&pd.kv_v)?;
```

**Impact**: `compute_extraction_2d` now correctly handles non-uniform knot vectors (e.g. `[0,0,0,0.2,0.5,0.8,1,1,1]`).

### B. New file `crates/assembly/src/iga/iga_bezier.rs`

#### B.1. Core helper: `bezier_eval_2d`

```rust
pub fn bezier_eval_2d(
    pd: &NurbsPatch2DData,
    ext: &BezierExtraction2D,
    elem_u: usize,
    elem_v: usize,
    xi: f64,      // ∈ [0,1]
    eta: f64,     // ∈ [0,1]
    phi: &mut [f64],
    phys_grads: &mut [f64],
) -> (f64 /* det_j */);
```

**Flow** (one Gauss point):

1. Get the extraction matrix `C` from `ext.matrices[ev * ext.n_elements_u + eu]`.
2. Evaluate Bernstein basis + parametric gradients on `[0,1]²` via `eval_bernstein_2d(p,q,xi,eta, phi_b, grads_b)`.
3. Apply extraction to get B-spline values/grads via `apply_extraction_2d(C, ...)` → `phi_n`, `grads_n`.
4. If NURBS weights exist, apply rational weighting:
   - `W = Σ_a phi_n[a] * w[a]`
   - `R[a] = phi_n[a] * w[a] / W`
   - `dR/du[a] = (w[a] * dN/du[a] * W - w[a] * N[a] * dW/du) / W²` (same for dR/dv)
5. Compute physical Jacobian from control points:
   - `J[i][j] = Σ_a x_A[i] * dR_A/dξ_j`
   - `det_j = |det(J)|`
6. Transform to physical gradients: `∇_x R = J^{-T} · ∇_ξ R`

**Complexity**: O((p+1)²(q+1)²) per Gauss point — dominated by the extraction matmul, same as `apply_extraction_2d`. Idential to Cox-de Boor in FLOPs for uniform knots; faster for non-uniform due to precomputation.

#### B.2. Core helper: `bezier_eval_3d`

Same pattern using `eval_bernstein_3d` + `apply_extraction_3d`. The 3D extraction matmul is O((p+1)²(q+1)²(r+1)²) per Gauss point.

#### B.3. Assembly functions

Each function:
1. Precomputes `BezierExtraction2D/3D` for each patch (one-time, O(n_elements × p³)).
2. Iterates over elements (knot spans) — for each element, looks up its `C_e`.
3. For each Gauss point, calls `bezier_eval_2d/3d`.
4. Assembles element contributions into global matrix/vector.

```rust
// 2-D
pub fn assemble_iga_diffusion_2d_bezier(mesh: &NurbsMesh2D, kappa: f64, quad_order: u8) -> CsrMatrix<f64>;
pub fn assemble_iga_mass_2d_bezier(mesh: &NurbsMesh2D, rho: f64, quad_order: u8) -> CsrMatrix<f64>;
pub fn assemble_iga_load_2d_bezier(mesh: &NurbsMesh2D, source: impl Fn(&[f64]) -> f64, quad_order: u8) -> Vec<f64>;

// 3-D
pub fn assemble_iga_diffusion_3d_bezier(mesh: &NurbsMesh3D, kappa: f64, quad_order: u8) -> CsrMatrix<f64>;
pub fn assemble_iga_mass_3d_bezier(mesh: &NurbsMesh3D, rho: f64, quad_order: u8) -> CsrMatrix<f64>;
pub fn assemble_iga_load_3d_bezier(mesh: &NurbsMesh3D, source: impl Fn(&[f64]) -> f64, quad_order: u8) -> Vec<f64>;
```

Future functions (elasticity, multi-patch, nonlinear) can reuse `bezier_eval_2d/3d` the same way `iga.rs` reuses `physical_grads_2d/3d`.

### C. Register in `mod.rs`

```rust
pub mod iga_bezier;
pub use iga_bezier::*;
```

## Testing

Every assembly function gets a test comparing against the Cox-de Boor reference:

1. **Uniform knots** (identity extraction): Bezier and Cox-de Boor results must match to 1e-14.
2. **Non-uniform knots**: Both implementations must match to 1e-10.
3. **Partition of unity**: Mass row sum equals domain volume.
4. **Source integral**: Load vector sum for f=1 equals domain volume.
5. **Symmetry**: Stiffness matrix must be symmetric.
6. **Positivity**: Diagonal entries of stiffness matrix must be positive.

## File Change Summary

| File | Change |
|------|--------|
| `crates/element/src/bezier_extraction.rs` | L194-L195: `compute_extraction_1d` → `compute_extraction_1d_full` (2 lines) |
| `crates/assembly/src/iga/iga_bezier.rs` | **New** — ~400 lines: helpers + assembly functions + tests |
| `crates/assembly/src/iga/mod.rs` | Add `pub mod iga_bezier; pub use iga_bezier::*;` |
