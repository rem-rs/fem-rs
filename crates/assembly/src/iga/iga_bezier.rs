//! Bezier extraction-based IGA assembly (2-D and 3-D).
//!
//! Replaces the Cox-de Boor recursion in [`super::iga`] with precomputed
//! element extraction operators: Bernstein basis evaluation + matrix
//! multiply per Gauss point.
//!
//! # Reference
//! * Borden et al., "Isogeometric analysis with Bézier extraction" (CMAME 2011)

use fem_element::bezier_extraction::{self, BezierExtraction2D, BezierExtraction3D};
use fem_element::iga::{NurbsMesh2D, NurbsMesh3D, NurbsPatch2DData, NurbsPatch3DData};
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
    debug_assert!(span_idx_u >= p && span_idx_v >= q, "span index must be >= degree");
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

/// Build a local→global DOF mapping for a 3-D element, assuming row-major
/// ordering (w slowest, u fastest) matching the control-pt layout.
///
/// `nu`, `nv` = total basis functions in the u/v directions for the patch.
/// `dof_offset` = patch-level DOF offset (0 for first patch).
fn element_dof_map_3d(
    span_u: usize,
    span_v: usize,
    span_w: usize,
    p: usize,
    q: usize,
    r: usize,
    nu: usize,
    nv: usize,
    dof_offset: usize,
    out: &mut [usize],
) {
    debug_assert!(span_u >= p && span_v >= q && span_w >= r, "span index must be >= degree");
    let np1 = p + 1;
    let nq1 = q + 1;
    let active_u_base = span_u - p;
    let active_v_base = span_v - q;
    let active_w_base = span_w - r;
    for k in 0..=r {
        for j in 0..=q {
            for i in 0..=p {
                let local = k * nq1 * np1 + j * np1 + i;
                let global = dof_offset
                    + (active_w_base + k) * nu * nv
                    + (active_v_base + j) * nu
                    + (active_u_base + i);
                out[local] = global;
            }
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

// ─── 3-D core evaluator ──────────────────────────────────────────────────────

/// Evaluate NURBS basis and physical gradients for a single element at (ξ,η,ζ) ∈ [0,1]³.
///
/// Returns `det_j`. Fills:
/// - `phi[a]` = R_a(ξ,η,ζ) (NURBS basis values, length n_local)
/// - `phys_grads[a*3..a*3+3]` = ∇R_a (physical gradient, length n_local*3)
///
/// `local_to_global` maps local DOF index `a` (0..n_local-1) to the global
/// control-point / weight index for that active basis function.
///
/// # Panics
/// Panics if the geometric Jacobian is degenerate (|det J| < 1e-14).
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
    local_to_global: &[usize],
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
    bezier_extraction::apply_extraction_3d(
        C, n_local, &phi_b, &grads_b, &mut phi_n, &mut grads_n,
    );

    // 3. NURBS rational weighting using global weights
    let w = pd.weights.as_slice();
    let mut W = 0.0_f64;
    let mut dW_du = 0.0_f64;
    let mut dW_dv = 0.0_f64;
    let mut dW_dw = 0.0_f64;
    for a in 0..n_local {
        let wa = w[local_to_global[a]];
        W += wa * phi_n[a];
        dW_du += wa * grads_n[a * 3];
        dW_dv += wa * grads_n[a * 3 + 1];
        dW_dw += wa * grads_n[a * 3 + 2];
    }
    assert!(W.abs() > 1e-300, "NURBS denominator near zero");
    let inv_W = 1.0 / W;
    let inv_W2 = inv_W * inv_W;
    let mut phi_r = vec![0.0_f64; n_local];
    let mut grads_r = vec![0.0_f64; n_local * 3];
    for a in 0..n_local {
        let wa = w[local_to_global[a]];
        let nv = phi_n[a];
        let dn_du = grads_n[a * 3];
        let dn_dv = grads_n[a * 3 + 1];
        let dn_dw = grads_n[a * 3 + 2];
        phi_r[a] = wa * nv * inv_W;
        grads_r[a * 3]     = (wa * dn_du * W - wa * nv * dW_du) * inv_W2;
        grads_r[a * 3 + 1] = (wa * dn_dv * W - wa * nv * dW_dv) * inv_W2;
        grads_r[a * 3 + 2] = (wa * dn_dw * W - wa * nv * dW_dw) * inv_W2;
    }

    // 4. 3×3 Jacobian
    let mut jac = [[0.0_f64; 3]; 3];
    for a in 0..n_local {
        let gi = local_to_global[a];
        for i in 0..3 {
            let xa = pd.control_pts[gi][i];
            jac[i][0] += xa * grads_r[a * 3];
            jac[i][1] += xa * grads_r[a * 3 + 1];
            jac[i][2] += xa * grads_r[a * 3 + 2];
        }
    }
    let det_j = jac[0][0] * (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1])
              - jac[0][1] * (jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0])
              + jac[0][2] * (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]);
    assert!(
        det_j.abs() > 1e-14,
        "degenerate Jacobian at elem({elem_u},{elem_v},{elem_w}) \
         ξ={xi} η={eta} ζ={zeta}"
    );
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

// ─── 3-D assembly ────────────────────────────────────────────────────────────

/// Assemble the 3-D diffusion stiffness matrix using Bezier extraction.
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
            .expect("compute_extraction_3d failed");
        let n_local = ext.n_local;
        let p = ext.degree_u;
        let q = ext.degree_v;
        let r = ext.degree_w;
        let nu = pd.kv_u.n_basis();
        let nv = pd.kv_v.n_basis();

        let spans_u = nonempty_spans(&pd.kv_u.knots);
        let spans_v = nonempty_spans(&pd.kv_v.knots);
        let spans_w = nonempty_spans(&pd.kv_w.knots);

        let mut phi = vec![0.0_f64; n_local];
        let mut phys_grads = vec![0.0_f64; n_local * 3];
        let mut l2g = vec![0usize; n_local];

        for (eu, (span_u, _, _)) in spans_u.iter().enumerate() {
            for (ev, (span_v, _, _)) in spans_v.iter().enumerate() {
                for (ew, (span_w, _, _)) in spans_w.iter().enumerate() {
                    element_dof_map_3d(
                        *span_u, *span_v, *span_w, p, q, r, nu, nv, dof_offset, &mut l2g,
                    );
                    for (&qx, &wx) in qpts.iter().zip(&qwts) {
                        let xi = qx;
                        for (&qy, &wy) in qpts.iter().zip(&qwts) {
                            let eta = qy;
                            for (&qz, &wz) in qpts.iter().zip(&qwts) {
                                let zeta = qz;
                                let det_j = bezier_eval_3d(
                                    pd, &ext, eu, ev, ew, xi, eta, zeta,
                                    &mut phi, &mut phys_grads, &l2g,
                                );
                                let w = wx * wy * wz * det_j.abs();

                                for a in 0..n_local {
                                    let ga = l2g[a];
                                    for b in 0..n_local {
                                        let gb = l2g[b];
                                        let dot = phys_grads[a * 3] * phys_grads[b * 3]
                                            + phys_grads[a * 3 + 1] * phys_grads[b * 3 + 1]
                                            + phys_grads[a * 3 + 2] * phys_grads[b * 3 + 2];
                                        coo.add(ga, gb, kappa * dot * w);
                                    }
                                }
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

/// Assemble the 3-D mass matrix using Bezier extraction.
pub fn assemble_iga_mass_3d_bezier(
    mesh: &NurbsMesh3D,
    rho: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let n_total: usize = mesh.patches.iter().map(|p| p.control_pts.len()).sum();
    let mut coo = CooMatrix::<f64>::new(n_total, n_total);
    let (qpts, qwts) = gauss_01(quad_order);

    let mut dof_offset = 0usize;
    for pd in &mesh.patches {
        let ext = bezier_extraction::compute_extraction_3d(pd)
            .expect("compute_extraction_3d failed");
        let n_local = ext.n_local;
        let p = ext.degree_u;
        let q = ext.degree_v;
        let r = ext.degree_w;
        let nu = pd.kv_u.n_basis();
        let nv = pd.kv_v.n_basis();

        let spans_u = nonempty_spans(&pd.kv_u.knots);
        let spans_v = nonempty_spans(&pd.kv_v.knots);
        let spans_w = nonempty_spans(&pd.kv_w.knots);

        let mut phi = vec![0.0_f64; n_local];
        let mut phys_grads = vec![0.0_f64; n_local * 3];
        let mut l2g = vec![0usize; n_local];

        for (eu, (span_u, _, _)) in spans_u.iter().enumerate() {
            for (ev, (span_v, _, _)) in spans_v.iter().enumerate() {
                for (ew, (span_w, _, _)) in spans_w.iter().enumerate() {
                    element_dof_map_3d(
                        *span_u, *span_v, *span_w, p, q, r, nu, nv, dof_offset, &mut l2g,
                    );
                    for (&qx, &wx) in qpts.iter().zip(&qwts) {
                        let xi = qx;
                        for (&qy, &wy) in qpts.iter().zip(&qwts) {
                            let eta = qy;
                            for (&qz, &wz) in qpts.iter().zip(&qwts) {
                                let zeta = qz;
                                let det_j = bezier_eval_3d(
                                    pd, &ext, eu, ev, ew, xi, eta, zeta,
                                    &mut phi, &mut phys_grads, &l2g,
                                );
                                let w = wx * wy * wz * det_j.abs();

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
            }
        }
        dof_offset += pd.control_pts.len();
    }

    coo.into_csr()
}

/// Assemble the 3-D load vector using Bezier extraction.
///
/// `source` receives the physical coordinate `&[x, y, z]` and returns the source value.
pub fn assemble_iga_load_3d_bezier(
    mesh: &NurbsMesh3D,
    source: impl Fn(&[f64]) -> f64,
    quad_order: u8,
) -> Vec<f64> {
    let n_total: usize = mesh.patches.iter().map(|p| p.control_pts.len()).sum();
    let mut rhs = vec![0.0_f64; n_total];
    let (qpts, qwts) = gauss_01(quad_order);

    let mut dof_offset = 0usize;
    for pd in &mesh.patches {
        let ext = bezier_extraction::compute_extraction_3d(pd)
            .expect("compute_extraction_3d failed");
        let n_local = ext.n_local;
        let p = ext.degree_u;
        let q = ext.degree_v;
        let r = ext.degree_w;
        let nu = pd.kv_u.n_basis();
        let nv = pd.kv_v.n_basis();

        let spans_u = nonempty_spans(&pd.kv_u.knots);
        let spans_v = nonempty_spans(&pd.kv_v.knots);
        let spans_w = nonempty_spans(&pd.kv_w.knots);

        let mut phi = vec![0.0_f64; n_local];
        let mut phys_grads = vec![0.0_f64; n_local * 3];
        let mut l2g = vec![0usize; n_local];

        for (eu, (span_u, _, _)) in spans_u.iter().enumerate() {
            for (ev, (span_v, _, _)) in spans_v.iter().enumerate() {
                for (ew, (span_w, _, _)) in spans_w.iter().enumerate() {
                    element_dof_map_3d(
                        *span_u, *span_v, *span_w, p, q, r, nu, nv, dof_offset, &mut l2g,
                    );
                    for (&qx, &wx) in qpts.iter().zip(&qwts) {
                        let xi = qx;
                        for (&qy, &wy) in qpts.iter().zip(&qwts) {
                            let eta = qy;
                            for (&qz, &wz) in qpts.iter().zip(&qwts) {
                                let zeta = qz;
                                let det_j = bezier_eval_3d(
                                    pd, &ext, eu, ev, ew, xi, eta, zeta,
                                    &mut phi, &mut phys_grads, &l2g,
                                );
                                let w = wx * wy * wz * det_j.abs();

                                // Physical coordinates for source evaluation
                                let mut x_phys = [0.0_f64; 3];
                                for a in 0..n_local {
                                    let gi = l2g[a];
                                    x_phys[0] += phi[a] * pd.control_pts[gi][0];
                                    x_phys[1] += phi[a] * pd.control_pts[gi][1];
                                    x_phys[2] += phi[a] * pd.control_pts[gi][2];
                                }
                                let f_val = source(&x_phys);

                                for a in 0..n_local {
                                    rhs[l2g[a]] += f_val * phi[a] * w;
                                }
                            }
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
        assemble_iga_diffusion_2d, assemble_iga_diffusion_3d,
        assemble_iga_load_2d, assemble_iga_load_3d,
        assemble_iga_mass_2d, assemble_iga_mass_3d,
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

    // ─── 3-D tests ─────────────────────────────────────────────────────────────

    /// Build a uniform degree-1 patch on [0,1]³ with 2×2×2 elements
    fn make_test_patch_3d() -> NurbsMesh3D {
        let p = 1;
        let n = 4; // n_basis = 4 → 2 elements per direction
        let kv = NurbsKnotVector::uniform(p, n - p);
        let n_ctrl = n * n * n;
        let ctrl: Vec<[f64; 3]> = (0..n_ctrl)
            .map(|idx| {
                let i = idx % n;
                let j = (idx / n) % n;
                let k = idx / (n * n);
                [
                    i as f64 / (n - 1) as f64,
                    j as f64 / (n - 1) as f64,
                    k as f64 / (n - 1) as f64,
                ]
            })
            .collect();
        NurbsMesh3D::single_patch(kv.clone(), kv.clone(), kv.clone(), ctrl, vec![1.0; n_ctrl])
    }

    #[test]
    fn bezier_3d_diffusion_matches_cdb() {
        let mesh = make_test_patch_3d();
        let k_bezier = assemble_iga_diffusion_3d_bezier(&mesh, 1.0, 3);
        let k_cdb = assemble_iga_diffusion_3d(&mesh, 1.0, 3);
        assert_eq!(k_bezier.nrows, k_cdb.nrows);
        let n = k_bezier.nrows;
        for i in 0..n {
            for ptr in k_bezier.row_ptr[i]..k_bezier.row_ptr[i + 1] {
                let j = k_bezier.col_idx[ptr] as usize;
                let v_bez = k_bezier.values[ptr];
                let v_cdb = k_cdb.get(i, j);
                assert!(
                    (v_bez - v_cdb).abs() < 1e-14,
                    "K3D[{i},{j}]: bezier={:.16e} cdb={:.16e} diff={:.2e}",
                    v_bez,
                    v_cdb,
                    (v_bez - v_cdb).abs()
                );
            }
        }
    }

    #[test]
    fn bezier_3d_load_matches_cdb() {
        let mesh = make_test_patch_3d();
        let f_bezier = assemble_iga_load_3d_bezier(&mesh, |_| 1.0, 3);
        let f_cdb = assemble_iga_load_3d(&mesh, |_| 1.0, 3);
        assert_eq!(f_bezier.len(), f_cdb.len());
        for i in 0..f_bezier.len() {
            assert!(
                (f_bezier[i] - f_cdb[i]).abs() < 1e-12,
                "f3D[{i}]: bezier={:.12e} cdb={:.12e}",
                f_bezier[i],
                f_cdb[i]
            );
        }
    }

    #[test]
    fn bezier_3d_mass_matches_cdb() {
        let mesh = make_test_patch_3d();
        let m_bezier = assemble_iga_mass_3d_bezier(&mesh, 1.0, 3);
        let m_cdb = assemble_iga_mass_3d(&mesh, 1.0, 3);
        assert_eq!(m_bezier.nrows, m_cdb.nrows);
        let n = m_bezier.nrows;
        for i in 0..n {
            for ptr in m_bezier.row_ptr[i]..m_bezier.row_ptr[i + 1] {
                let j = m_bezier.col_idx[ptr] as usize;
                let v_bez = m_bezier.values[ptr];
                let v_cdb = m_cdb.get(i, j);
                assert!(
                    (v_bez - v_cdb).abs() < 1e-14,
                    "M3D[{i},{j}]: bezier={:.16e} cdb={:.16e}",
                    v_bez, v_cdb
                );
            }
        }
    }

    #[test]
    fn bezier_2d_nonuniform_knots_matches_cdb() {
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
}
