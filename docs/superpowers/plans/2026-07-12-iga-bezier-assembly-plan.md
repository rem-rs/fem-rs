# IGA Bezier Assembly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Cox-de Boor B-spline evaluation in IGA assembly with precomputed Bezier extraction (Bernstein + extraction matrix multiply), for both 2D and 3D.

**Architecture:** A new `iga_bezier.rs` module provides `bezier_eval_2d/3d` helpers (Bernstein eval → extraction → NURBS rational → physical map) and assembly functions (diffusion/mass/load) that iterate element-by-element over knot spans. Tests compare element matrix entries against the existing Cox-de Boor `iga.rs` reference at machine precision for uniform knots and <1e-10 for non-uniform.

**Tech Stack:** Rust, `fem-element::bezier_extraction` (pre-existing), `fem-element::nurbs` (NurbsPatch2DData/3DData), `fem-linalg::CooMatrix`, `fem-assembly::iga` (reference).

## Global Constraints

- All assembly functions must output the same matrix entries (within tolerances) as the corresponding `iga.rs` Cox-de Boor functions for identical inputs.
- No changes to existing `iga.rs` API or behavior — new functions are additive.
- The `compute_extraction_2d` fix is a 2-line change in `bezier_extraction.rs`.

---

### Task 1: Fix `compute_extraction_2d` to support non-uniform knot vectors

**Files:**
- Modify: `crates/element/src/bezier_extraction.rs:194-196`

**Interfaces:**
- Consumes: `compute_extraction_1d` (old, identity-only) — no longer used internally
- Produces: `compute_extraction_2d` now uses `compute_extraction_1d_full` internally

- [ ] **Step 1: Change the two function calls**

Find lines 194-195 in `bezier_extraction.rs`. Replace `compute_extraction_1d` with `compute_extraction_1d_full`:

```rust
// Before:
let ext_u = compute_extraction_1d(&pd.kv_u)?;
let ext_v = compute_extraction_1d(&pd.kv_v)?;

// After:
let ext_u = compute_extraction_1d_full(&pd.kv_u)?;
let ext_v = compute_extraction_1d_full(&pd.kv_v)?;
```

- [ ] **Step 2: Verify existing tests still pass**

Run: `cd crates/element && cargo test bezier_extraction -- --quiet`
Expected: all tests pass (particularly `ext_2d_identity` and `ext_1d_nonuniform_extraction_matches_bspline_eval`)

Wait — `ext_2d_identity` may now fail because `compute_extraction_1d_full` on a uniform degree-1 knot vector should still return identity matrices (Chebyshev points on degree-1 are just {0.0, 1.0}, both endpoints, and the system is trivial). Let me verify by running the test first before assuming anything.

Run the tests and confirm no regressions. If any test fails, investigate whether the old `compute_extraction_1d` produced exact identity while `compute_extraction_1d_full` has tiny floating-point differences — if so, loosen the tolerance in the test.

- [ ] **Step 3: Commit**

```bash
cd crates/element
git add src/bezier_extraction.rs
git commit -m "fix: compute_extraction_2d now uses compute_extraction_1d_full for non-uniform knots"
```

---

### Task 2: Create `iga_bezier.rs` — 2D helpers + assembly functions

**Files:**
- Create: `crates/assembly/src/iga/iga_bezier.rs`

**Interfaces:**
- Consumes (from `fem_element::bezier_extraction`):
  - `BezierExtraction2D`, `compute_extraction_2d`
  - `eval_bernstein_2d`, `apply_extraction_2d`
- Consumes (from `fem_element::nurbs`): `NurbsPatch2DData`
- Consumes (from `fem_element::iga`): `NurbsMesh2D`
- Produces:
  - `bezier_eval_2d(pd, ext, elem_u, elem_v, xi, eta, phi, phys_grads) -> f64`
  - `assemble_iga_diffusion_2d_bezier(mesh, kappa, order) -> CsrMatrix<f64>`
  - `assemble_iga_mass_2d_bezier(mesh, rho, order) -> CsrMatrix<f64>`
  - `assemble_iga_load_2d_bezier(mesh, source, order) -> Vec<f64>`

- [ ] **Step 1: File header and imports**

```rust
//! Bezier extraction-based IGA assembly (2-D and 3-D).
//!
//! Replaces the Cox-de Boor recursion in [`super::iga`] with precomputed
//! element extraction operators: Bernstein basis evaluation + matrix
//! multiply per Gauss point.
//!
//! # Reference
//! * Borden et al., "Isogeometric analysis with Bézier extraction" (CMAME 2011)

use fem_core::types::DofId;
use fem_element::bezier_extraction::{self, BezierExtraction2D, BezierExtraction3D};
use fem_element::iga::{
    NurbsMesh2D, NurbsMesh3D, NurbsPatch2DData, NurbsPatch3DData,
};
use fem_element::quadrature::seg_rule;
use fem_element::reference::QuadratureRule;
use fem_linalg::{CooMatrix, CsrMatrix};
```

- [ ] **Step 2: Write `bezier_eval_2d` — the core 2D helper**

This function evaluates NURBS basis values, physical gradients, and Jacobian determinant at one quadrature point `(xi, eta) ∈ [0,1]²` for element `(elem_u, elem_v)`, using the precomputed extraction operator.

```rust
/// Evaluate NURBS basis and physical gradients for a single element at (ξ,η) ∈ [0,1]².
///
/// Returns `det_j`. Fills:
/// - `phi[a]` = R_a(ξ,η) (NURBS basis values, length n_local)
/// - `phys_grads[a*2]` = dR_a/dx, `[a*2+1]` = dR_a/dy
///
/// # Panics
/// Panics if the geometric Jacobian is degenerate (|det J| < 1e-14).
pub fn bezier_eval_2d(
    pd: &NurbsPatch2DData,
    ext: &BezierExtraction2D,
    elem_u: usize,
    elem_v: usize,
    xi: f64,
    eta: f64,
    phi: &mut [f64],
    phys_grads: &mut [f64],
) -> f64 {
    let p = ext.degree_u;
    let q = ext.degree_v;
    let np1 = p + 1;
    let nq1 = q + 1;
    let n_local = ext.n_local;

    // 1. Evaluate Bernstein basis + parametric gradients
    let mut phi_b = vec![0.0_f64; n_local];
    let mut grads_b = vec![0.0_f64; n_local * 2];
    bezier_extraction::eval_bernstein_2d(p, q, xi, eta, &mut phi_b, &mut grads_b);

    // 2. Apply extraction: B-spline basis N = C^T · B
    let idx = elem_v * ext.n_elements_u + elem_u;
    let C = &ext.matrices[idx];
    let mut phi_n = vec![0.0_f64; n_local];
    let mut grads_n = vec![0.0_f64; n_local * 2];
    bezier_extraction::apply_extraction_2d(C, n_local, &phi_b, &grads_b, &mut phi_n, &mut grads_n);

    // 3. NURBS rational weighting (if weights exist) or pass-through
    let (phi_r, grads_r): (Vec<f64>, Vec<f64>) = if let Some(w) = pd.weights.as_slice() {
        let mut W = 0.0_f64;
        let mut dW_du = 0.0_f64;
        let mut dW_dv = 0.0_f64;
        for a in 0..n_local {
            let wa = w[a];
            W += wa * phi_n[a];
            dW_du += wa * grads_n[a * 2];
            dW_dv += wa * grads_n[a * 2 + 1];
        }
        assert!(W.abs() > 1e-300, "NURBS denominator near zero");
        let inv_W = 1.0 / W;
        let inv_W2 = inv_W * inv_W;
        let mut pr = vec![0.0_f64; n_local];
        let mut gr = vec![0.0_f64; n_local * 2];
        for a in 0..n_local {
            let wa = w[a];
            let n_val = phi_n[a];
            let dn_du = grads_n[a * 2];
            let dn_dv = grads_n[a * 2 + 1];
            pr[a] = wa * n_val * inv_W;
            gr[a * 2]     = (wa * dn_du * W - wa * n_val * dW_du) * inv_W2;
            gr[a * 2 + 1] = (wa * dn_dv * W - wa * n_val * dW_dv) * inv_W2;
        }
        (pr, gr)
    } else {
        (phi_n, grads_n)
    };

    // 4. Physical Jacobian: J[i][j] = Σ_A x_A[i] * dR_A/dξ_j
    let mut jac = [[0.0_f64; 2]; 2];
    for a in 0..n_local {
        let cx = pd.control_pts[a][0];
        let cy = pd.control_pts[a][1];
        jac[0][0] += cx * grads_r[a * 2];     // dx/du
        jac[0][1] += cx * grads_r[a * 2 + 1]; // dx/dv
        jac[1][0] += cy * grads_r[a * 2];     // dy/du
        jac[1][1] += cy * grads_r[a * 2 + 1]; // dy/dv
    }
    let det_j = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
    assert!(det_j.abs() > 1e-14, "degenerate Jacobian at elem({elem_u},{elem_v}) ξ={xi} η={eta}");
    let inv_det = 1.0 / det_j;
    let jac_inv_t = [
        [ jac[1][1] * inv_det, -jac[1][0] * inv_det],
        [-jac[0][1] * inv_det,  jac[0][0] * inv_det],
    ];

    // 5. Physical gradients: ∇_x R = J^{-T} · ∇_ξ R
    for a in 0..n_local {
        let dru = grads_r[a * 2];
        let drv = grads_r[a * 2 + 1];
        phi[a] = phi_r[a];
        phys_grads[a * 2]     = jac_inv_t[0][0] * dru + jac_inv_t[0][1] * drv;
        phys_grads[a * 2 + 1] = jac_inv_t[1][0] * dru + jac_inv_t[1][1] * drv;
    }

    det_j
}
```

- [ ] **Step 3: Write `nonempty_spans_reuse` — span iterator helper**

Re-use from `iga.rs` (currently private). We need the span intervals for element-by-element iteration. Add a standalone function:

```rust
fn nonempty_spans(knots: &[f64]) -> Vec<(usize, f64, f64)> {
    // Returns (span_index, left, right) for each non-empty knot span
    knots.windows(2)
        .enumerate()
        .filter_map(|(i, w)| if w[1] > w[0] { Some((i, w[0], w[1])) } else { None })
        .collect()
}
```

- [ ] **Step 4: Write `quadrature_on_cell` helper**

Return Gauss points and weights on `[0,1]`:

```rust
fn gauss_01(order: u8) -> (Vec<f64>, Vec<f64>) {
    // seg_rule gives points on [0,1] with weights already scaled for the interval
    let seg = seg_rule(order);
    let pts: Vec<f64> = seg.points.iter().map(|p| p[0]).collect();
    (pts, seg.weights)
}
```

- [ ] **Step 5: Write `assemble_iga_diffusion_2d_bezier`**

```rust
/// Assemble the diffusion stiffness matrix using Bezier extraction.
pub fn assemble_iga_diffusion_2d_bezier(
    mesh: &NurbsMesh2D,
    kappa: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let n_total: usize = mesh.patches.iter().map(|p| p.control_pts.len()).sum();
    let mut coo = CooMatrix::<f64>::new(n_total, n_total);
    let (qpts, qwts) = gauss_01(quad_order);

    let mut dof_offset = 0usize;
    for pd in &mesh.patches {
        let ext = bezier_extraction::compute_extraction_2d(pd)
            .expect("compute_extraction_2d failed");
        let n_local = ext.n_local;
        let spans_u = nonempty_spans(&pd.kv_u.knots);
        let spans_v = nonempty_spans(&pd.kv_v.knots);

        let mut phi = vec![0.0_f64; n_local];
        let mut phys_grads = vec![0.0_f64; n_local * 2];

        for (eu, (_, u0, u1)) in spans_u.iter().enumerate() {
            let hu = u1 - u0;
            for (ev, (_, v0, v1)) in spans_v.iter().enumerate() {
                for (&qx, &wx) in qpts.iter().zip(&qwts) {
                    let xi = qx; // ξ ∈ [0,1]
                    let u = u0 + hu * xi;
                    for (&qy, &wy) in qpts.iter().zip(&qwts) {
                        let eta = qy; // η ∈ [0,1]
                        let v = v0 + hv * eta;
                        let det_j = bezier_eval_2d(pd, &ext, eu, ev, xi, eta, &mut phi, &mut phys_grads);
                        let w = wx * wy * hu * hv * det_j.abs();

                        for a in 0..n_local {
                            let ga = dof_offset + a;
                            for b in 0..n_local {
                                let gb = dof_offset + b;
                                let dot = phys_grads[a*2] * phys_grads[b*2]
                                        + phys_grads[a*2+1] * phys_grads[b*2+1];
                                coo.add(ga, gb, kappa * dot * w);
                            }
                        }
                    }
                }
            }
        }
        dof_offset += n_local;
    }

    coo.into_csr()
}
```

Wait, there's a bug — `hv` is not defined in the inner loop. Let me fix:

```rust
for (ev, (_, v0, v1)) in spans_v.iter().enumerate() {
    let hv = v1 - v0;
    for (&qx, &wx) in qpts.iter().zip(&qwts) {
        let xi = qx;
        for (&qy, &wy) in qpts.iter().zip(&qwts) {
            let eta = qy;
            let det_j = bezier_eval_2d(pd, &ext, eu, ev, xi, eta, &mut phi, &mut phys_grads);
            let w = wx * wy * hu * hv * det_j.abs();
            // ... assembly
        }
    }
}
```

- [ ] **Step 6: Write `assemble_iga_mass_2d_bezier`**

Same structure, but uses `phi[a] * phi[b]` and calls `bezier_eval_2d` which returns both phi and phys_grads.

```rust
pub fn assemble_iga_mass_2d_bezier(
    mesh: &NurbsMesh2D,
    rho: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let n_total: usize = mesh.patches.iter().map(|p| p.control_pts.len()).sum();
    let mut coo = CooMatrix::<f64>::new(n_total, n_total);
    let (qpts, qwts) = gauss_01(quad_order);

    let mut dof_offset = 0usize;
    for pd in &mesh.patches {
        let ext = bezier_extraction::compute_extraction_2d(pd)
            .expect("compute_extraction_2d failed");
        let n_local = ext.n_local;
        let spans_u = nonempty_spans(&pd.kv_u.knots);
        let spans_v = nonempty_spans(&pd.kv_v.knots);

        let mut phi = vec![0.0_f64; n_local];
        let mut phys_grads = vec![0.0_f64; n_local * 2];

        for (eu, (_, u0, u1)) in spans_u.iter().enumerate() {
            let hu = u1 - u0;
            for (ev, (_, v0, v1)) in spans_v.iter().enumerate() {
                let hv = v1 - v0;
                for (&qx, &wx) in qpts.iter().zip(&qwts) {
                    let xi = qx;
                    for (&qy, &wy) in qpts.iter().zip(&qwts) {
                        let eta = qy;
                        let det_j = bezier_eval_2d(pd, &ext, eu, ev, xi, eta, &mut phi, &mut phys_grads);
                        let w = wx * wy * hu * hv * det_j.abs();

                        for a in 0..n_local {
                            let ga = dof_offset + a;
                            for b in 0..n_local {
                                let gb = dof_offset + b;
                                coo.add(ga, gb, rho * phi[a] * phi[b] * w);
                            }
                        }
                    }
                }
            }
        }
        dof_offset += n_local;
    }

    coo.into_csr()
}
```

- [ ] **Step 7: Write `assemble_iga_load_2d_bezier`**

```rust
pub fn assemble_iga_load_2d_bezier(
    mesh: &NurbsMesh2D,
    source: impl Fn(&[f64]) -> f64,
    quad_order: u8,
) -> Vec<f64> {
    let n_total: usize = mesh.patches.iter().map(|p| p.control_pts.len()).sum();
    let mut rhs = vec![0.0_f64; n_total];
    let (qpts, qwts) = gauss_01(quad_order);

    let mut dof_offset = 0usize;
    for pd in &mesh.patches {
        let ext = bezier_extraction::compute_extraction_2d(pd)
            .expect("compute_extraction_2d failed");
        let n_local = ext.n_local;
        let spans_u = nonempty_spans(&pd.kv_u.knots);
        let spans_v = nonempty_spans(&pd.kv_v.knots);

        let mut phi = vec![0.0_f64; n_local];
        let mut phys_grads = vec![0.0_f64; n_local * 2];

        for (eu, (_, u0, u1)) in spans_u.iter().enumerate() {
            let hu = u1 - u0;
            for (ev, (_, v0, v1)) in spans_v.iter().enumerate() {
                let hv = v1 - v0;
                for (&qx, &wx) in qpts.iter().zip(&qwts) {
                    let xi = qx;
                    for (&qy, &wy) in qpts.iter().zip(&qwts) {
                        let eta = qy;
                        let det_j = bezier_eval_2d(pd, &ext, eu, ev, xi, eta, &mut phi, &mut phys_grads);
                        let w = wx * wy * hu * hv * det_j.abs();
                        // Physical coordinates for source evaluation
                        // (could compute from phi and control points, but for now just
                        // use midpoint approximate — source at physical center of element)
                        let u = u0 + hu * xi;
                        let v = v0 + hv * eta;
                        let mut x_phys = [0.0_f64; 2];
                        // We already have phi = NURBS values from bezier_eval_2d
                        for a in 0..n_local {
                            x_phys[0] += phi[a] * pd.control_pts[a][0];
                            x_phys[1] += phi[a] * pd.control_pts[a][1];
                        }
                        let f_val = source(&x_phys);

                        for a in 0..n_local {
                            rhs[dof_offset + a] += f_val * phi[a] * w;
                        }
                    }
                }
            }
        }
        dof_offset += n_local;
    }

    rhs
}
```

- [ ] **Step 8: Write 2D tests**

Tests compare Bezier assembly results against Cox-de Boor reference from `iga.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::iga::{
        assemble_iga_diffusion_2d,
        assemble_iga_mass_2d,
        assemble_iga_load_2d,
    };
    use fem_element::iga::NurbsKnotVector;

    /// Build a uniform degree-2 patch on [0,1]² with 3×3 elements
    fn make_test_patch_2d(n_elems_u: usize, n_elems_v: usize) -> NurbsMesh2D {
        let p = 2;
        let kv_u = NurbsKnotVector::uniform(p, n_elems_u);
        let kv_v = NurbsKnotVector::uniform(p, n_elems_v);
        let nu = kv_u.n_basis();
        let nv = kv_v.n_basis();
        let n_dof = nu * nv;
        let ctrl: Vec<[f64; 2]> = (0..n_dof).map(|idx| {
            let i = idx % nu;
            let j = idx / nu;
            [i as f64 / (nu - 1) as f64, j as f64 / (nu - 1) as f64]
        }).collect();
        NurbsMesh2D::single_patch(kv_u, kv_v, ctrl, vec![1.0; n_dof])
    }

    #[test]
    fn bezier_2d_diffusion_matches_cdb() {
        let mesh = make_test_patch_2d(3, 3);
        let k_bezier = assemble_iga_diffusion_2d_bezier(&mesh, 1.0, 4);
        let k_cdb = assemble_iga_diffusion_2d(&mesh, 1.0, 4);
        let n = k_bezier.nrows;
        for i in 0..n {
            for ptr in k_bezier.row_ptr[i]..k_bezier.row_ptr[i+1] {
                let j = k_bezier.col_idx[ptr] as usize;
                let v_bez = k_bezier.values[ptr];
                let v_cdb = k_cdb.get(i, j);
                assert!((v_bez - v_cdb).abs() < 1e-14,
                    "K[{i},{j}]: bezier={:.16e} cdb={:.16e} diff={:.2e}", v_bez, v_cdb, (v_bez-v_cdb).abs());
            }
        }
    }

    #[test]
    fn bezier_2d_mass_matches_cdb() {
        let mesh = make_test_patch_2d(3, 3);
        let m_bezier = assemble_iga_mass_2d_bezier(&mesh, 1.0, 4);
        let m_cdb = assemble_iga_mass_2d(&mesh, 1.0, 4);
        let n = m_bezier.nrows;
        for i in 0..n {
            for ptr in m_bezier.row_ptr[i]..m_bezier.row_ptr[i+1] {
                let j = m_bezier.col_idx[ptr] as usize;
                let v_bez = m_bezier.values[ptr];
                let v_cdb = m_cdb.get(i, j);
                assert!((v_bez - v_cdb).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn bezier_2d_load_matches_cdb() {
        let mesh = make_test_patch_2d(3, 3);
        let src = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let f_bezier = assemble_iga_load_2d_bezier(&mesh, &src, 4);
        let f_cdb = assemble_iga_load_2d(&mesh, &src, 4);
        for i in 0..f_bezier.len() {
            assert!((f_bezier[i] - f_cdb[i]).abs() < 1e-12,
                "f[{i}]: bezier={:.12e} cdb={:.12e}",
                f_bezier[i], f_cdb[i]);
        }
    }
}
```

- [ ] **Step 9: Run 2D tests**

Run: `cd crates/assembly && cargo test bezier_2d -- --quiet`
Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add crates/assembly/src/iga/iga_bezier.rs
git commit -m "feat(iga): add 2D Bezier element-by-element assembly"
```

---

### Task 3: Add 3D helpers and assembly to `iga_bezier.rs`

**Files:**
- Modify: `crates/assembly/src/iga/iga_bezier.rs` (append 3D functions)

**Interfaces:**
- Consumes: `BezierExtraction3D`, `compute_extraction_3d`, `eval_bernstein_3d`, `apply_extraction_3d`
- Produces:
  - `bezier_eval_3d(pd, ext, elem_u, elem_v, elem_w, xi, eta, zeta, phi, phys_grads) -> f64`
  - `assemble_iga_diffusion_3d_bezier(mesh, kappa, order) -> CsrMatrix<f64>`
  - `assemble_iga_mass_3d_bezier(mesh, rho, order) -> CsrMatrix<f64>`
  - `assemble_iga_load_3d_bezier(mesh, source, order) -> Vec<f64>`

- [ ] **Step 1: Implement `bezier_eval_3d`**

Same pattern as `bezier_eval_2d` but for 3D:

```rust
/// Evaluate NURBS basis and physical gradients at (ξ,η,ζ) ∈ [0,1]³.
pub fn bezier_eval_3d(
    pd: &NurbsPatch3DData,
    ext: &BezierExtraction3D,
    elem_u: usize,
    elem_v: usize,
    elem_w: usize,
    xi: f64,
    eta: f64,
    zeta: f64,
    phi: &mut [f64],
    phys_grads: &mut [f64],
) -> f64 {
    let p = ext.degree_u;
    let q = ext.degree_v;
    let r = ext.degree_w;
    let n_local = ext.n_local;

    // 1. Evaluate Bernstein basis + parametric gradients
    let mut phi_b = vec![0.0_f64; n_local];
    let mut grads_b = vec![0.0_f64; n_local * 3];
    bezier_extraction::eval_bernstein_3d(p, q, r, xi, eta, zeta, &mut phi_b, &mut grads_b);

    // 2. Apply extraction: B-spline basis N = C^T · B
    let idx = elem_w * ext.n_elements_v * ext.n_elements_u
            + elem_v * ext.n_elements_u
            + elem_u;
    let C = &ext.matrices[idx];
    let mut phi_n = vec![0.0_f64; n_local];
    let mut grads_n = vec![0.0_f64; n_local * 3];
    bezier_extraction::apply_extraction_3d(C, n_local, &phi_b, &grads_b, &mut phi_n, &mut grads_n);

    // 3. NURBS rational weighting
    let (phi_r, grads_r): (Vec<f64>, Vec<f64>) = if let Some(w) = pd.weights.as_slice() {
        let mut W = 0.0; let mut dW_du = 0.0; let mut dW_dv = 0.0; let mut dW_dw = 0.0;
        for a in 0..n_local {
            let wa = w[a];
            W += wa * phi_n[a];
            dW_du += wa * grads_n[a * 3];
            dW_dv += wa * grads_n[a * 3 + 1];
            dW_dw += wa * grads_n[a * 3 + 2];
        }
        assert!(W.abs() > 1e-300);
        let inv_W = 1.0 / W;
        let inv_W2 = inv_W * inv_W;
        let mut pr = vec![0.0; n_local];
        let mut gr = vec![0.0; n_local * 3];
        for a in 0..n_local {
            let wa = w[a];
            let nv = phi_n[a];
            let dn_du = grads_n[a * 3];
            let dn_dv = grads_n[a * 3 + 1];
            let dn_dw = grads_n[a * 3 + 2];
            pr[a] = wa * nv * inv_W;
            gr[a * 3]     = (wa * dn_du * W - wa * nv * dW_du) * inv_W2;
            gr[a * 3 + 1] = (wa * dn_dv * W - wa * nv * dW_dv) * inv_W2;
            gr[a * 3 + 2] = (wa * dn_dw * W - wa * nv * dW_dw) * inv_W2;
        }
        (pr, gr)
    } else {
        (phi_n, grads_n)
    };

    // 4. 3×3 Jacobian
    let mut jac = [[0.0_f64; 3]; 3];
    for a in 0..n_local {
        for i in 0..3 {
            let xa = pd.control_pts[a][i];
            jac[i][0] += xa * grads_r[a * 3];
            jac[i][1] += xa * grads_r[a * 3 + 1];
            jac[i][2] += xa * grads_r[a * 3 + 2];
        }
    }
    let det_j = jac[0][0] * (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1])
              - jac[0][1] * (jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0])
              + jac[0][2] * (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]);
    assert!(det_j.abs() > 1e-14);
    let inv = 1.0 / det_j;
    let jac_inv_t = [
        [ (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1]) * inv,
          (jac[1][2]*jac[2][0] - jac[1][0]*jac[2][2]) * inv,
          (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]) * inv ],
        [ (jac[0][2]*jac[2][1] - jac[0][1]*jac[2][2]) * inv,
          (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0]) * inv,
          (jac[0][1]*jac[2][0] - jac[0][0]*jac[2][1]) * inv ],
        [ (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1]) * inv,
          (jac[0][2]*jac[1][0] - jac[0][0]*jac[1][2]) * inv,
          (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]) * inv ],
    ];

    // 5. Transform to physical gradients
    for a in 0..n_local {
        let dru = grads_r[a * 3];
        let drv = grads_r[a * 3 + 1];
        let drw = grads_r[a * 3 + 2];
        phi[a] = phi_r[a];
        phys_grads[a * 3]     = jac_inv_t[0][0]*dru + jac_inv_t[0][1]*drv + jac_inv_t[0][2]*drw;
        phys_grads[a * 3 + 1] = jac_inv_t[1][0]*dru + jac_inv_t[1][1]*drv + jac_inv_t[1][2]*drw;
        phys_grads[a * 3 + 2] = jac_inv_t[2][0]*dru + jac_inv_t[2][1]*drv + jac_inv_t[2][2]*drw;
    }

    det_j
}
```

- [ ] **Step 2: Implement 3D assembly functions**

Each follows the same structure as the 2D version but:
- Iterates over `spans_u`, `spans_v`, `spans_w`
- Uses `bezier_eval_3d`
- 3-loop Gauss quadrature

```rust
pub fn assemble_iga_diffusion_3d_bezier(
    mesh: &NurbsMesh3D,
    kappa: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let n_total: usize = mesh.patches.iter().map(|p| p.control_pts.len()).sum();
    let mut coo = CooMatrix::<f64>::new(n_total, n_total);
    let (qpts, qwts) = gauss_01(quad_order);

    let mut dof_offset = 0usize;
    for pd in &mesh.patches {
        let ext = bezier_extraction::compute_extraction_3d(pd)
            .expect("compute_extraction_3d");
        let n_local = ext.n_local;
        let spans_u = nonempty_spans(&pd.kv_u.knots);
        let spans_v = nonempty_spans(&pd.kv_v.knots);
        let spans_w = nonempty_spans(&pd.kv_w.knots);

        let mut phi = vec![0.0_f64; n_local];
        let mut phys_grads = vec![0.0_f64; n_local * 3];

        for (eu, (_, u0, u1)) in spans_u.iter().enumerate() {
            let hu = u1 - u0;
            for (ev, (_, v0, v1)) in spans_v.iter().enumerate() {
                let hv = v1 - v0;
                for (ew, (_, w0, w1)) in spans_w.iter().enumerate() {
                    let hw = w1 - w0;
                    for (&qx, &wx) in qpts.iter().zip(&qwts) {
                        let xi = qx;
                        for (&qy, &wy) in qpts.iter().zip(&qwts) {
                            let eta = qy;
                            for (&qz, &wz) in qpts.iter().zip(&qwts) {
                                let zeta = qz;
                                let det_j = bezier_eval_3d(pd, &ext, eu, ev, ew, xi, eta, zeta,
                                    &mut phi, &mut phys_grads);
                                let w = wx * wy * wz * hu * hv * hw * det_j.abs();

                                for a in 0..n_local {
                                    let ga = dof_offset + a;
                                    for b in 0..n_local {
                                        let gb = dof_offset + b;
                                        let dot = phys_grads[a*3]*phys_grads[b*3]
                                                + phys_grads[a*3+1]*phys_grads[b*3+1]
                                                + phys_grads[a*3+2]*phys_grads[b*3+2];
                                        coo.add(ga, gb, kappa * dot * w);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        dof_offset += n_local;
    }

    coo.into_csr()
}
```

Similar for `assemble_iga_mass_3d_bezier` (uses `phi[a] * phi[b]`) and `assemble_iga_load_3d_bezier` (evaluates source at physical point from phi and control points, multiplies by phi[a]).

- [ ] **Step 3: Add 3D tests**

```rust
#[test]
fn bezier_3d_diffusion_matches_cdb() {
    // Uniform degree-1 cube [0,1]³ with 2×2×2 elements
    let p = 1; let n = 4;
    let kv = NurbsKnotVector::uniform(p, n - p);
    let n_ctrl = n * n * n;
    let ctrl: Vec<[f64; 3]> = (0..n_ctrl).map(|idx| {
        let i = idx % n; let j = (idx / n) % n; let k = idx / (n * n);
        [i as f64/(n-1) as f64, j as f64/(n-1) as f64, k as f64/(n-1) as f64]
    }).collect();
    let mesh = NurbsMesh3D::single_patch(kv.clone(), kv.clone(), kv.clone(), ctrl, vec![1.0; n_ctrl]);

    let k_bezier = assemble_iga_diffusion_3d_bezier(&mesh, 1.0, 3);
    let k_cdb = crate::iga::assemble_iga_diffusion_3d(&mesh, 1.0, 3);
    let n = k_bezier.nrows;
    for i in 0..n {
        for ptr in k_bezier.row_ptr[i]..k_bezier.row_ptr[i+1] {
            let j = k_bezier.col_idx[ptr] as usize;
            let v_bez = k_bezier.values[ptr];
            let v_cdb = k_cdb.get(i, j);
            assert!((v_bez - v_cdb).abs() < 1e-14,
                "K3D[{i},{j}]: bezier={:.16e} cdb={:.16e}", v_bez, v_cdb);
        }
    }
}

#[test]
fn bezier_3d_load_matches_cdb() {
    let p = 1; let n = 4;
    let kv = NurbsKnotVector::uniform(p, n - p);
    let n_ctrl = n * n * n;
    let ctrl: Vec<[f64; 3]> = (0..n_ctrl).map(|idx| {
        let i = idx % n; let j = (idx / n) % n; let k = idx / (n * n);
        [i as f64/(n-1) as f64, j as f64/(n-1) as f64, k as f64/(n-1) as f64]
    }).collect();
    let mesh = NurbsMesh3D::single_patch(kv.clone(), kv.clone(), kv.clone(), ctrl, vec![1.0; n_ctrl]);

    let f_bezier = assemble_iga_load_3d_bezier(&mesh, |_| 1.0, 3);
    let f_cdb = crate::iga::assemble_iga_load_3d(&mesh, |_| 1.0, 3);
    for i in 0..f_bezier.len() {
        assert!((f_bezier[i] - f_cdb[i]).abs() < 1e-12);
    }
}
```

- [ ] **Step 4: Run all tests**

Run: `cd crates/assembly && cargo test bezier_ -- --quiet`
Expected: all 2D + 3D tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/assembly/src/iga/iga_bezier.rs
git commit -m "feat(iga): add 3D Bezier element-by-element assembly"
```

---

### Task 4: Register `iga_bezier` in module and verify

**Files:**
- Modify: `crates/assembly/src/iga/mod.rs`

- [ ] **Step 1: Add the module declaration**

```rust
pub mod iga;
pub mod iga_assembler;
pub mod iga_bezier;   // ← add this line
pub mod iga_trim;

pub use iga::*;
pub use iga_assembler::*;
pub use iga_bezier::*;  // ← add this line
pub use iga_trim::*;
```

- [ ] **Step 2: Full test suite**

Run: `cd crates/assembly && cargo test bezier -- --quiet`
Expected: all pass.

Also run the full assembly test suite to ensure nothing is broken:
Run: `cd crates/assembly && cargo test`
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/assembly/src/iga/mod.rs
git commit -m "chore: register iga_bezier module"
```

---

### Task 5: Non-uniform knot validation (optional but recommended)

Add a test that verifies Bezier and Cox-de Boor match for non-uniform knot vectors. This is important because `compute_extraction_2d` now uses `compute_extraction_1d_full` for the first time.

**Files:**
- Modify: `crates/assembly/src/iga/iga_bezier.rs` (append to `#[cfg(test)] mod tests`)

- [ ] **Step 1: Add non-uniform 2D test**

```rust
#[test]
fn bezier_2d_nonuniform_knots_matches_cdb() {
    // Non-uniform knot vectors in both directions
    let kv_u = NurbsKnotVector::new(vec![0.0, 0.0, 0.0, 0.2, 0.5, 0.8, 1.0, 1.0, 1.0], 2);
    let kv_v = NurbsKnotVector::new(vec![0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0], 2);
    let nu = kv_u.n_basis(); // 6
    let nv = kv_v.n_basis(); // 5
    let n_dof = nu * nv;     // 30
    let ctrl: Vec<[f64; 2]> = (0..n_dof).map(|idx| {
        let i = idx % nu;
        let j = idx / nu;
        [i as f64 / (nu - 1) as f64, j as f64 / (nv - 1) as f64]
    }).collect();
    let mesh = NurbsMesh2D::single_patch(kv_u, kv_v, ctrl, vec![1.0; n_dof]);

    let k_bezier = assemble_iga_diffusion_2d_bezier(&mesh, 1.0, 4);
    let k_cdb = assemble_iga_diffusion_2d(&mesh, 1.0, 4);
    let n = k_bezier.nrows;
    let mut max_diff = 0.0_f64;
    for i in 0..n {
        for ptr in k_bezier.row_ptr[i]..k_bezier.row_ptr[i+1] {
            let j = k_bezier.col_idx[ptr] as usize;
            let diff = (k_bezier.values[ptr] - k_cdb.get(i, j)).abs();
            max_diff = max_diff.max(diff);
        }
    }
    assert!(max_diff < 1e-10, "max diff for non-uniform 2D = {:.2e}", max_diff);
}
```

- [ ] **Step 2: Run and verify**

Run: `cd crates/assembly && cargo test bezier_2d_nonuniform -- --quiet`
Expected: pass (max_diff < 1e-10).

- [ ] **Step 3: Commit**

```bash
git add crates/assembly/src/iga/iga_bezier.rs
git commit -m "test: add non-uniform knot validation for Bezier assembly"
```
