//! Bezier extraction-based IGA assembly (2-D).
//!
//! Replaces the Cox-de Boor recursion in [`super::iga`] with precomputed
//! element extraction operators: Bernstein basis evaluation + matrix
//! multiply per Gauss point.
//!
//! # Reference
//! * Borden et al., "Isogeometric analysis with Bézier extraction" (CMAME 2011)

use fem_element::bezier_extraction::{self, BezierExtraction2D};
use fem_element::iga::{NurbsMesh2D, NurbsPatch2DData};
use fem_element::quadrature::seg_rule;
use fem_linalg::{CooMatrix, CsrMatrix};

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Return (span_index, left, right) for each non-empty knot span.
fn nonempty_spans(knots: &[f64]) -> Vec<(usize, f64, f64)> {
    knots
        .windows(2)
        .enumerate()
        .filter_map(|(i, w)| {
            if w[1] > w[0] {
                Some((i, w[0], w[1]))
            } else {
                None
            }
        })
        .collect()
}

/// Gauss–Legendre points and weights on [0, 1].
fn gauss_01(order: u8) -> (Vec<f64>, Vec<f64>) {
    let seg = seg_rule(order);
    let pts: Vec<f64> = seg.points.iter().map(|p| p[0]).collect();
    (pts, seg.weights)
}

/// Build a local→global DOF mapping for an element, assuming row-major
/// ordering (v slowest, u fastest) matching the control-pt layout.
///
/// `nu` = total number of basis functions in the u-direction for the patch.
/// `dof_offset` = patch-level DOF offset (0 for first patch).
fn element_dof_map(
    span_idx_u: usize,
    span_idx_v: usize,
    p: usize,
    q: usize,
    nu: usize,
    dof_offset: usize,
    out: &mut [usize],
) {
    let np1 = p + 1;
    let active_u_base = span_idx_u - p;
    let active_v_base = span_idx_v - q;
    for j in 0..=q {
        for i in 0..=p {
            let local = j * np1 + i;
            let global = dof_offset + (active_v_base + j) * nu + (active_u_base + i);
            out[local] = global;
        }
    }
}

// ─── Core evaluator ──────────────────────────────────────────────────────────

/// Evaluate NURBS basis and physical gradients for a single element at (ξ,η) ∈ [0,1]².
///
/// Returns `det_j`. Fills:
/// - `phi[a]` = R_a(ξ,η) (NURBS basis values, length n_local, indexed by local DOF)
/// - `phys_grads[a*2]` = dR_a/dx, `[a*2+1]` = dR_a/dy (local DOF indexing)
///
/// `local_to_global` maps local DOF index `a` (0..n_local-1) to the global
/// control-point / weight index for that active basis function.
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
    local_to_global: &[usize],
) -> f64 {
    let p = ext.degree_u;
    let q = ext.degree_v;
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
    bezier_extraction::apply_extraction_2d(
        C, n_local, &phi_b, &grads_b, &mut phi_n, &mut grads_n,
    );

    // 3. NURBS rational weighting using global weights
    let w = pd.weights.as_slice();
    let mut W = 0.0_f64;
    let mut dW_du = 0.0_f64;
    let mut dW_dv = 0.0_f64;
    for a in 0..n_local {
        let wa = w[local_to_global[a]];
        W += wa * phi_n[a];
        dW_du += wa * grads_n[a * 2];
        dW_dv += wa * grads_n[a * 2 + 1];
    }
    assert!(W.abs() > 1e-300, "NURBS denominator near zero");
    let inv_W = 1.0 / W;
    let inv_W2 = inv_W * inv_W;
    let mut phi_r = vec![0.0_f64; n_local];
    let mut grads_r = vec![0.0_f64; n_local * 2];
    for a in 0..n_local {
        let wa = w[local_to_global[a]];
        let n_val = phi_n[a];
        let dn_du = grads_n[a * 2];
        let dn_dv = grads_n[a * 2 + 1];
        phi_r[a] = wa * n_val * inv_W;
        grads_r[a * 2] = (wa * dn_du * W - wa * n_val * dW_du) * inv_W2;
        grads_r[a * 2 + 1] = (wa * dn_dv * W - wa * n_val * dW_dv) * inv_W2;
    }

    // 4. Physical Jacobian: J[i][j] = Σ_A x_A[i] * dR_A/dξ_j
    let mut jac = [[0.0_f64; 2]; 2];
    for a in 0..n_local {
        let gi = local_to_global[a];
        let cx = pd.control_pts[gi][0];
        let cy = pd.control_pts[gi][1];
        jac[0][0] += cx * grads_r[a * 2]; // dx/du
        jac[0][1] += cx * grads_r[a * 2 + 1]; // dx/dv
        jac[1][0] += cy * grads_r[a * 2]; // dy/du
        jac[1][1] += cy * grads_r[a * 2 + 1]; // dy/dv
    }
    let det_j = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
    assert!(
        det_j.abs() > 1e-14,
        "degenerate Jacobian at elem({elem_u},{elem_v}) ξ={xi} η={eta}"
    );
    let inv_det = 1.0 / det_j;
    let jac_inv_t = [
        [jac[1][1] * inv_det, -jac[1][0] * inv_det],
        [-jac[0][1] * inv_det, jac[0][0] * inv_det],
    ];

    // 5. Physical gradients: ∇_x R = J^{-T} · ∇_ξ R
    for a in 0..n_local {
        let dru = grads_r[a * 2];
        let drv = grads_r[a * 2 + 1];
        phi[a] = phi_r[a];
        phys_grads[a * 2] = jac_inv_t[0][0] * dru + jac_inv_t[0][1] * drv;
        phys_grads[a * 2 + 1] = jac_inv_t[1][0] * dru + jac_inv_t[1][1] * drv;
    }

    det_j
}

// ─── 2-D assembly ────────────────────────────────────────────────────────────

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
        let p = ext.degree_u;
        let q = ext.degree_v;
        let nu = pd.kv_u.n_basis();

        let spans_u = nonempty_spans(&pd.kv_u.knots);
        let spans_v = nonempty_spans(&pd.kv_v.knots);

        let mut phi = vec![0.0_f64; n_local];
        let mut phys_grads = vec![0.0_f64; n_local * 2];
        let mut l2g = vec![0usize; n_local];

        for (eu, (span_u, u0, u1)) in spans_u.iter().enumerate() {
            for (ev, (span_v, v0, v1)) in spans_v.iter().enumerate() {
                element_dof_map(*span_u, *span_v, p, q, nu, dof_offset, &mut l2g);
                for (&qx, &wx) in qpts.iter().zip(&qwts) {
                    let xi = qx; // ξ ∈ [0,1]
                    for (&qy, &wy) in qpts.iter().zip(&qwts) {
                        let eta = qy; // η ∈ [0,1]
                        let det_j = bezier_eval_2d(
                            pd, &ext, eu, ev, xi, eta, &mut phi, &mut phys_grads, &l2g,
                        );
                        // det_j = |det(J_ξ→x)| already includes the element size
                        // because the Jacobian maps from [0,1]² to physical coords.
                        let w = wx * wy * det_j.abs();

                        for a in 0..n_local {
                            let ga = l2g[a];
                            for b in 0..n_local {
                                let gb = l2g[b];
                                let dot = phys_grads[a * 2] * phys_grads[b * 2]
                                    + phys_grads[a * 2 + 1] * phys_grads[b * 2 + 1];
                                coo.add(ga, gb, kappa * dot * w);
                            }
                        }
                    }
                }
            }
        }
        dof_offset += pd.control_pts.len();
    }

    coo.into_csr()
}

/// Assemble the mass matrix using Bezier extraction.
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
        let p = ext.degree_u;
        let q = ext.degree_v;
        let nu = pd.kv_u.n_basis();

        let spans_u = nonempty_spans(&pd.kv_u.knots);
        let spans_v = nonempty_spans(&pd.kv_v.knots);

        let mut phi = vec![0.0_f64; n_local];
        let mut phys_grads = vec![0.0_f64; n_local * 2];
        let mut l2g = vec![0usize; n_local];

        for (eu, (span_u, u0, u1)) in spans_u.iter().enumerate() {
            for (ev, (span_v, v0, v1)) in spans_v.iter().enumerate() {
                element_dof_map(*span_u, *span_v, p, q, nu, dof_offset, &mut l2g);
                for (&qx, &wx) in qpts.iter().zip(&qwts) {
                    let xi = qx;
                    for (&qy, &wy) in qpts.iter().zip(&qwts) {
                        let eta = qy;
                        let det_j = bezier_eval_2d(
                            pd, &ext, eu, ev, xi, eta, &mut phi, &mut phys_grads, &l2g,
                        );
                        let w = wx * wy * det_j.abs();

                        for a in 0..n_local {
                            let ga = l2g[a];
                            for b in 0..n_local {
                                let gb = l2g[b];
                                coo.add(ga, gb, rho * phi[a] * phi[b] * w);
                            }
                        }
                    }
                }
            }
        }
        dof_offset += pd.control_pts.len();
    }

    coo.into_csr()
}

/// Assemble the load vector using Bezier extraction.
///
/// `source` receives the physical coordinate `&[x, y]` and returns the source value.
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
        let p = ext.degree_u;
        let q = ext.degree_v;
        let nu = pd.kv_u.n_basis();

        let spans_u = nonempty_spans(&pd.kv_u.knots);
        let spans_v = nonempty_spans(&pd.kv_v.knots);

        let mut phi = vec![0.0_f64; n_local];
        let mut phys_grads = vec![0.0_f64; n_local * 2];
        let mut l2g = vec![0usize; n_local];

        for (eu, (span_u, u0, u1)) in spans_u.iter().enumerate() {
            for (ev, (span_v, v0, v1)) in spans_v.iter().enumerate() {
                element_dof_map(*span_u, *span_v, p, q, nu, dof_offset, &mut l2g);
                for (&qx, &wx) in qpts.iter().zip(&qwts) {
                    let xi = qx;
                    for (&qy, &wy) in qpts.iter().zip(&qwts) {
                        let eta = qy;
                        let det_j = bezier_eval_2d(
                            pd, &ext, eu, ev, xi, eta, &mut phi, &mut phys_grads, &l2g,
                        );
                        let w = wx * wy * det_j.abs();

                        // Physical coordinates for source evaluation
                        let mut x_phys = [0.0_f64; 2];
                        for a in 0..n_local {
                            let gi = l2g[a];
                            x_phys[0] += phi[a] * pd.control_pts[gi][0];
                            x_phys[1] += phi[a] * pd.control_pts[gi][1];
                        }
                        let f_val = source(&x_phys);

                        for a in 0..n_local {
                            rhs[l2g[a]] += f_val * phi[a] * w;
                        }
                    }
                }
            }
        }
        dof_offset += pd.control_pts.len();
    }

    rhs
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iga::{
        assemble_iga_diffusion_2d, assemble_iga_load_2d, assemble_iga_mass_2d,
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
        let ctrl: Vec<[f64; 2]> = (0..n_dof)
            .map(|idx| {
                let i = idx % nu;
                let j = idx / nu;
                [
                    i as f64 / (nu - 1) as f64,
                    j as f64 / (nv - 1) as f64,
                ]
            })
            .collect();
        NurbsMesh2D::single_patch(kv_u, kv_v, ctrl, vec![1.0; n_dof])
    }

    #[test]
    fn bezier_2d_diffusion_matches_cdb() {
        let mesh = make_test_patch_2d(3, 3);
        let k_bezier = assemble_iga_diffusion_2d_bezier(&mesh, 1.0, 4);
        let k_cdb = assemble_iga_diffusion_2d(&mesh, 1.0, 4);
        assert_eq!(k_bezier.nrows, k_cdb.nrows);
        let n = k_bezier.nrows;
        for i in 0..n {
            for ptr in k_bezier.row_ptr[i]..k_bezier.row_ptr[i + 1] {
                let j = k_bezier.col_idx[ptr] as usize;
                let v_bez = k_bezier.values[ptr];
                let v_cdb = k_cdb.get(i, j);
                assert!(
                    (v_bez - v_cdb).abs() < 1e-14,
                    "K[{i},{j}]: bezier={:.16e} cdb={:.16e} diff={:.2e}",
                    v_bez,
                    v_cdb,
                    (v_bez - v_cdb).abs()
                );
            }
        }
    }

    #[test]
    fn bezier_2d_mass_matches_cdb() {
        let mesh = make_test_patch_2d(3, 3);
        let m_bezier = assemble_iga_mass_2d_bezier(&mesh, 1.0, 4);
        let m_cdb = assemble_iga_mass_2d(&mesh, 1.0, 4);
        assert_eq!(m_bezier.nrows, m_cdb.nrows);
        let n = m_bezier.nrows;
        for i in 0..n {
            for ptr in m_bezier.row_ptr[i]..m_bezier.row_ptr[i + 1] {
                let j = m_bezier.col_idx[ptr] as usize;
                let v_bez = m_bezier.values[ptr];
                let v_cdb = m_cdb.get(i, j);
                assert!(
                    (v_bez - v_cdb).abs() < 1e-14,
                    "M[{i},{j}]: bezier={:.16e} cdb={:.16e}",
                    v_bez,
                    v_cdb
                );
            }
        }
    }

    #[test]
    fn bezier_2d_load_matches_cdb() {
        let mesh = make_test_patch_2d(3, 3);
        let src = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let f_bezier = assemble_iga_load_2d_bezier(&mesh, &src, 4);
        let f_cdb = assemble_iga_load_2d(&mesh, &src, 4);
        assert_eq!(f_bezier.len(), f_cdb.len());
        for i in 0..f_bezier.len() {
            assert!(
                (f_bezier[i] - f_cdb[i]).abs() < 1e-12,
                "f[{i}]: bezier={:.12e} cdb={:.12e}",
                f_bezier[i],
                f_cdb[i]
            );
        }
    }
}
