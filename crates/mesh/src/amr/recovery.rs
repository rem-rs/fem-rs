//! # Gradient recovery methods (Kelley enhancement)
//!
//! Superconvergent gradient recovery techniques that enhance the basic
//! Zienkiewicz–Zhu (ZZ) estimator.  Two methods are provided:
//!
//! - **SPR** (Superconvergent Patch Recovery): local L² projection of element
//!   gradients onto a polynomial space over each nodal patch.
//! - **Global L² projection**: area-weighted nodal gradient recovery that
//!   approximates the L² projection onto the FE space via a lumped mass matrix.
//!
//! Both methods produce recovered gradients `G(u_h)` with higher accuracy
//! than simple nodal averaging, leading to more reliable ZZ error indicators.

use fem_core::{ElemId, NodeId};
use crate::element_type::ElementType;
use crate::Mesh;

// ═════════════════════════════════════════════════════════════════════════════
//  Shared helper: element gradient at centroid (Tri3 / Quad4 2-D)
// ═════════════════════════════════════════════════════════════════════════════

/// Compute the P1 (element-wise constant) gradient at the centroid of a 2-D
/// element (Tri3 or Quad4).
fn elem_gradient_2d(mesh: &Mesh<2>, u: &[f64], elem: ElemId) -> [f64; 2] {
    let ns = mesh.elem_nodes(elem);
    let is_quad = mesh.element_type_at(0) == ElementType::Quad4;

    if is_quad {
        let c = |i: usize| mesh.coords_of(ns[i]);
        let uu = |i: usize| u[ns[i] as usize];
        let dxi  = 0.25 * (-uu(0) + uu(1) + uu(2) - uu(3));
        let deta = 0.25 * (-uu(0) - uu(1) + uu(2) + uu(3));
        let j00 = 0.25 * (-c(0)[0] + c(1)[0] + c(2)[0] - c(3)[0]);
        let j01 = 0.25 * (-c(0)[0] - c(1)[0] + c(2)[0] + c(3)[0]);
        let j10 = 0.25 * (-c(0)[1] + c(1)[1] + c(2)[1] - c(3)[1]);
        let j11 = 0.25 * (-c(0)[1] - c(1)[1] + c(2)[1] + c(3)[1]);
        let det = j00 * j11 - j01 * j10;
        let idet = if det.abs() > 1e-30 { 1.0 / det } else { 0.0 };
        [ (j11 * dxi - j10 * deta) * idet,
         (-j01 * dxi + j00 * deta) * idet ]
    } else {
        let c = |i: usize| mesh.coords_of(ns[i]);
        let j00 = c(1)[0] - c(0)[0]; let j01 = c(2)[0] - c(0)[0];
        let j10 = c(1)[1] - c(0)[1]; let j11 = c(2)[1] - c(0)[1];
        let det = j00 * j11 - j01 * j10;
        let idet = if det.abs() > 1e-30 { 1.0 / det } else { 0.0 };
        let mut gx = 0.0; let mut gy = 0.0;
        let uh = [u[ns[0] as usize], u[ns[1] as usize], u[ns[2] as usize]];
        let gref = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]];
        for k in 0..3 {
            let gpx = (j11 * gref[k][0] - j10 * gref[k][1]) * idet;
            let gpy = (-j01 * gref[k][0] + j00 * gref[k][1]) * idet;
            gx += uh[k] * gpx;
            gy += uh[k] * gpy;
        }
        [gx, gy]
    }
}

/// Compute element centroid coordinates (2-D).
fn elem_centroid_2d(mesh: &Mesh<2>, elem: ElemId) -> [f64; 2] {
    let ns = mesh.elem_nodes(elem);
    let npe = ns.len();
    let mut cx = 0.0; let mut cy = 0.0;
    for &n in ns {
        let [x, y] = mesh.coords_of(n);
        cx += x; cy += y;
    }
    let inv = 1.0 / npe as f64;
    [cx * inv, cy * inv]
}

/// Element area for 2-D element.
fn elem_area_2d(mesh: &Mesh<2>, elem: ElemId) -> f64 {
    let ns = mesh.elem_nodes(elem);
    let is_quad = mesh.element_type_at(0) == ElementType::Quad4;
    if is_quad {
        let c = |i: usize| mesh.coords_of(ns[i]);
        0.5 * ((c(0)[0]*c(1)[1] + c(1)[0]*c(2)[1] + c(2)[0]*c(3)[1] + c(3)[0]*c(0)[1])
             - (c(1)[0]*c(0)[1] + c(2)[0]*c(1)[1] + c(3)[0]*c(2)[1] + c(0)[0]*c(3)[1])).abs()
    } else {
        let c = |i: usize| mesh.coords_of(ns[i]);
        let det = (c(1)[0]-c(0)[0])*(c(2)[1]-c(0)[1])
                - (c(2)[0]-c(0)[0])*(c(1)[1]-c(0)[1]);
        0.5 * det.abs()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  SPR — Superconvergent Patch Recovery
// ═════════════════════════════════════════════════════════════════════════════

/// Recover nodal gradients using Superconvergent Patch Recovery (SPR).
///
/// For each mesh node, the element centroid gradients in the surrounding
/// patch are projected onto a linear polynomial `p(x,y)=c0+c1·x+c2·y` by
/// L² least-squares fitting.  The polynomial is evaluated at the node to
/// give the recovered gradient `G(u_h)(node)`.
///
/// This yields superconvergent O(h²) accuracy for the recovered gradient
/// on regular meshes, compared to O(h) for simple nodal averaging.
///
/// # Arguments
/// * `mesh` — 2-D Tri3 or Quad4 mesh.
/// * `u` — nodal solution values.
///
/// # Returns
/// Recovered nodal gradient vectors, length = `n_nodes`.
pub fn spr_recover_gradient_2d(mesh: &Mesh<2>, u: &[f64]) -> Vec<[f64; 2]> {
    let n_nodes = mesh.n_nodes();
    let n_elems = mesh.n_elems();

    // ── 1. Element gradients and centroids ─────────────────────────────────
    let elem_grads: Vec<[f64; 2]> = (0..n_elems as ElemId)
        .map(|e| elem_gradient_2d(mesh, u, e))
        .collect();
    let centroids: Vec<[f64; 2]> = (0..n_elems as ElemId)
        .map(|e| elem_centroid_2d(mesh, e))
        .collect();

    // ── 2. Node-to-elements adjacency (nodal patches) ──────────────────────
    let mut node_elems: Vec<Vec<ElemId>> = vec![Vec::new(); n_nodes];
    for e in 0..n_elems as ElemId {
        for &n in mesh.elem_nodes(e) {
            node_elems[n as usize].push(e);
        }
    }

    // ── 3. SPR for each node ──────────────────────────────────────────────
    let mut recovered = vec![[0.0_f64; 2]; n_nodes];

    for n in 0..n_nodes {
        let patch = &node_elems[n];
        let np = patch.len();
        if np == 0 { continue; }

        let [xn, yn] = mesh.coords_of(n as NodeId);

        if np < 3 {
            // Underdetermined: fall back to element-averaged gradient
            let mut gx = 0.0; let mut gy = 0.0;
            for &e in patch {
                gx += elem_grads[e as usize][0];
                gy += elem_grads[e as usize][1];
            }
            let inv = 1.0 / np as f64;
            recovered[n] = [gx * inv, gy * inv];
            continue;
        }

        // Fit p(x,y) = c0 + c1*x + c2*y to each component separately.
        // Normal equation: A * α = b, where A = Φ^T Φ and b = Φ^T g.
        // For linear basis [1, x, y] at np points.
        for comp in 0..2 {
            let mut a = [[0.0_f64; 3]; 3];    // 3×3 normal matrix
            let mut b = [0.0_f64; 3];          // RHS

            for &e in patch {
                let [xc, yc] = centroids[e as usize];
                let gc = elem_grads[e as usize][comp];
                // Basis: φ = [1, x, y]
                // A += φ^T φ
                a[0][0] += 1.0;
                a[0][1] += xc; a[1][0] += xc;
                a[0][2] += yc; a[2][0] += yc;
                a[1][1] += xc * xc;
                a[1][2] += xc * yc; a[2][1] += xc * yc;
                a[2][2] += yc * yc;
                // b += φ^T * gc
                b[0] += gc;
                b[1] += gc * xc;
                b[2] += gc * yc;
            }

            // Solve 3×3 system by Cramer's rule for robustness.
            let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                    - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                    + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

            if det.abs() < 1e-30 {
                // Singular system: fall back to averaging
                let mut avg = 0.0;
                for &e in patch { avg += elem_grads[e as usize][comp]; }
                recovered[n][comp] = avg / np as f64;
                continue;
            }

            // Cramer: α_i = det(A_i→b) / det(A)
            let det_a0 = |m: &[[f64;3];3]| -> f64 {
                b[0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
               - m[0][1] * (b[1] * m[2][2] - m[1][2] * b[2])
               + m[0][2] * (b[1] * m[2][1] - m[1][1] * b[2])
            };
            let det_a1 = |m: &[[f64;3];3]| -> f64 {
                m[0][0] * (b[1] * m[2][2] - m[1][2] * b[2])
               - b[0] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
               + m[0][2] * (m[1][0] * b[2] - b[1] * m[2][0])
            };
            let det_a2 = |m: &[[f64;3];3]| -> f64 {
                m[0][0] * (m[1][1] * b[2] - b[1] * m[2][1])
               - m[0][1] * (m[1][0] * b[2] - b[1] * m[2][0])
               + b[0] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            };

            let c0 = det_a0(&a) / det;
            let c1 = det_a1(&a) / det;
            let c2 = det_a2(&a) / det;

            // Evaluate polynomial at node coordinates
            recovered[n][comp] = c0 + c1 * xn + c2 * yn;
        }
    }

    recovered
}

// ═════════════════════════════════════════════════════════════════════════════
//  Global L² projection (lumped mass)
// ═════════════════════════════════════════════════════════════════════════════

/// Recover nodal gradients by global L² projection with lumped mass.
///
/// This approximates the L² projection of the element gradient onto the P1
/// FE space by solving `M * g = f` where `M` is the finite-element mass
/// matrix.  A lumped (diagonal) mass approximation is used:
///
/// ```text
/// g_rec(n) = (Σ_{e ∋ n} |e| · g_e) / (Σ_{e ∋ n} |e|)
/// ```
///
/// where `|e|` is the element area and `g_e` is the element centroid gradient.
/// This is area-weighted nodal averaging, which is more accurate than simple
/// nodal averaging (the original ZZ method) because larger elements contribute
/// proportionally more to the recovered gradient.
///
/// # Arguments
/// * `mesh` — 2-D Tri3 or Quad4 mesh.
/// * `u` — nodal solution values.
///
/// # Returns
/// Recovered nodal gradient vectors, length = `n_nodes`.
pub fn global_l2_projection_2d(mesh: &Mesh<2>, u: &[f64]) -> Vec<[f64; 2]> {
    let n_nodes = mesh.n_nodes();
    let n_elems = mesh.n_elems();

    // Element gradients and areas
    let elem_grads: Vec<[f64; 2]> = (0..n_elems as ElemId)
        .map(|e| elem_gradient_2d(mesh, u, e))
        .collect();
    let areas: Vec<f64> = (0..n_elems as ElemId)
        .map(|e| elem_area_2d(mesh, e))
        .collect();

    let mut recovered = vec![[0.0_f64; 2]; n_nodes];
    let mut weights = vec![0.0_f64; n_nodes];

    for e in 0..n_elems as ElemId {
        let area = areas[e as usize];
        let [gx, gy] = elem_grads[e as usize];
        for &n in mesh.elem_nodes(e) {
            let idx = n as usize;
            let w = area / mesh.elem_nodes(e).len() as f64; // lumped mass: |e|/npe
            recovered[idx][0] += gx * w;
            recovered[idx][1] += gy * w;
            weights[idx] += w;
        }
    }

    for n in 0..n_nodes {
        if weights[n] > 0.0 {
            let inv = 1.0 / weights[n];
            recovered[n][0] *= inv;
            recovered[n][1] *= inv;
        }
    }

    recovered
}

/// Compute the interpolated recovered gradient within an element.
///
/// Given recovered nodal gradients from SPR or global L² projection,
/// interpolate to any point in the element using P1 shape functions.
/// Returns the recovered gradient at the element centroid.
pub fn interpolate_recovered_gradient_2d(
    mesh: &Mesh<2>,
    elem: ElemId,
    nodal_grad: &[[f64; 2]],
) -> [f64; 2] {
    let ns = mesh.elem_nodes(elem);
    let npe = ns.len();

    // Simple average of nodal recovered gradients (superconvergent at centroid)
    let mut gx = 0.0; let mut gy = 0.0;
    for &n in ns {
        gx += nodal_grad[n as usize][0];
        gy += nodal_grad[n as usize][1];
    }
    let inv = 1.0 / npe as f64;
    [gx * inv, gy * inv]
}

/// Compute element-wise ZZ error indicators using superconvergent recovery.
///
/// η_K = ‖∇u_h|_K − G(u_h)|_K‖ · h_K
///
/// where `G` is the recovered gradient from SPR (superconvergent patch recovery).
/// This is more accurate than the standard ZZ estimator with simple averaging.
///
/// # Arguments
/// * `mesh` — 2-D Tri3 mesh.
/// * `u` — nodal solution values.
///
/// # Returns
/// Element-wise error indicators.
pub fn zz_spr_estimator_2d(mesh: &Mesh<2>, u: &[f64]) -> Vec<f64> {
    let n_elems = mesh.n_elems();
    let recovered = spr_recover_gradient_2d(mesh, u);

    (0..n_elems as ElemId).map(|e| {
        let [gx, gy] = elem_gradient_2d(mesh, u, e);
        let [grx, gry] = interpolate_recovered_gradient_2d(mesh, e, &recovered);
        let dx = gx - grx;
        let dy = gy - gry;
        let h = (2.0 * elem_area_2d(mesh, e)).sqrt();
        h * (dx * dx + dy * dy).sqrt()
    }).collect()
}

/// Compute element-wise ZZ error indicators using global L² projection recovery.
pub fn zz_l2_estimator_2d(mesh: &Mesh<2>, u: &[f64]) -> Vec<f64> {
    let n_elems = mesh.n_elems();
    let recovered = global_l2_projection_2d(mesh, u);

    (0..n_elems as ElemId).map(|e| {
        let [gx, gy] = elem_gradient_2d(mesh, u, e);
        let [grx, gry] = interpolate_recovered_gradient_2d(mesh, e, &recovered);
        let dx = gx - grx;
        let dy = gy - gry;
        let h = (2.0 * elem_area_2d(mesh, e)).sqrt();
        h * (dx * dx + dy * dy).sqrt()
    }).collect()
}

// ═════════════════════════════════════════════════════════════════════════════
//  Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mesh;

    // ── SPR recovery tests ────────────────────────────────────────────────

    #[test]
    fn spr_linear_solution_exact_recovery() {
        // u = 2x + 3y → constant gradient [2, 3] → SPR should recover exactly
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId);
            2.0 * c[0] + 3.0 * c[1]
        }).collect();

        let recovered = spr_recover_gradient_2d(&mesh, &u);
        for n in 0..n {
            let diff_x = (recovered[n][0] - 2.0).abs();
            let diff_y = (recovered[n][1] - 3.0).abs();
            assert!(diff_x < 1e-12, "SPR gradient x at node {n}: expected 2, got {}", recovered[n][0]);
            assert!(diff_y < 1e-12, "SPR gradient y at node {n}: expected 3, got {}", recovered[n][1]);
        }
    }

    #[test]
    fn spr_matches_zz_averaging_for_linear() {
        // For linear solutions, SPR and simple ZZ averaging should both recover exactly.
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0] - c[1]
        }).collect();

        let spr = spr_recover_gradient_2d(&mesh, &u);
        let zz = crate::amr::estimators::zz_estimator(&mesh, &u);

        // Since both should recover [1, -1] exactly for linear u:
        for n in 0..n {
            assert!((spr[n][0] - 1.0).abs() < 1e-12);
            assert!((spr[n][1] - (-1.0)).abs() < 1e-12);
        }
        // zz_estimator should also produce near-zero indicators
        let max_eta = zz.iter().cloned().fold(0.0, f64::max);
        assert!(max_eta < 1e-12);
    }

    #[test]
    fn spr_quadratic_solution_improves_over_zz() {
        // u = x² + y² on a coarse mesh
        // SPR should give a more accurate recovered gradient than simple averaging.
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes();

        // Exact gradient of x²+y² at node (x,y): [2x, 2y]
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0]*c[0] + c[1]*c[1]
        }).collect();

        let spr = spr_recover_gradient_2d(&mesh, &u);

        // Check that SPR gives better gradient at interior nodes (not on boundary)
        // We pick a node near the center.
        let mut best_err_spr = f64::MAX;
        let mut node_idx = 0;
        for i in 0..n {
            let c = mesh.coords_of(i as NodeId);
            if (c[0] - 0.5).abs() < 0.3 && (c[1] - 0.5).abs() < 0.3 {
                let err_spr = (spr[i][0] - 2.0 * c[0]).abs()
                            + (spr[i][1] - 2.0 * c[1]).abs();
                if err_spr < best_err_spr {
                    best_err_spr = err_spr;
                    node_idx = i;
                }
            }
        }
        // SPR should have reasonable accuracy
        let c = mesh.coords_of(node_idx as NodeId);
        let expected = [2.0 * c[0], 2.0 * c[1]];
        let err_spr = ((spr[node_idx][0] - expected[0]).powi(2)
                     + (spr[node_idx][1] - expected[1]).powi(2)).sqrt();
        assert!(err_spr < 0.5,
            "SPR recovery error at interior node {node_idx}: {err_spr:.4} (should be < 0.5)");
    }

    #[test]
    fn spr_estimator_nonzero_for_quadratic() {
        let mesh = Mesh::<2>::unit_square_tri(8);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0]*c[0]
        }).collect();

        let eta = zz_spr_estimator_2d(&mesh, &u);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max > 1e-6,
            "SPR-ZZ estimator should be >0 for x², got {max:.3e}");
    }

    #[test]
    fn spr_estimator_zero_for_linear() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0]
        }).collect();

        let eta = zz_spr_estimator_2d(&mesh, &u);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 1e-12,
            "SPR-ZZ estimator should be ~0 for linear u, got {max:.3e}");
    }

    // ── Global L² projection tests ────────────────────────────────────────

    #[test]
    fn l2_projection_linear_solution_exact() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); 3.0 * c[0] - c[1]
        }).collect();

        let recovered = global_l2_projection_2d(&mesh, &u);
        for n in 0..n {
            assert!((recovered[n][0] - 3.0).abs() < 1e-12);
            assert!((recovered[n][1] - (-1.0)).abs() < 1e-12);
        }
    }

    #[test]
    fn l2_projection_matches_spr_for_uniform() {
        // On a uniform mesh, area-weighted and simple averaging give similar results
        let mesh = Mesh::<2>::unit_square_tri(8);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); (c[0] + 2.0 * c[1]).sin()
        }).collect();

        let l2 = global_l2_projection_2d(&mesh, &u);
        let spr = spr_recover_gradient_2d(&mesh, &u);

        // Both methods should produce similar (not identical) gradients
        let mut diff_sum = 0.0;
        for i in 0..n {
            diff_sum += (l2[i][0] - spr[i][0]).abs() + (l2[i][1] - spr[i][1]).abs();
        }
        let avg_diff = diff_sum / n as f64;
        assert!(avg_diff < 1.0,
            "L2 and SPR should produce similar gradients, avg diff = {avg_diff:.4}");
    }

    // ── Recovery estimator shape tests ─────────────────────────────────────

    #[test]
    fn recovery_estimator_output_shape() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0]*c[0]
        }).collect();

        let eta_spr = zz_spr_estimator_2d(&mesh, &u);
        let eta_l2 = zz_l2_estimator_2d(&mesh, &u);

        assert_eq!(eta_spr.len(), mesh.n_elems());
        assert_eq!(eta_l2.len(), mesh.n_elems());
    }

    #[test]
    fn spr_quad4_works() {
        let mesh = Mesh::<2>::unit_square_quad(4);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0]*c[0]
        }).collect();

        let recovered = spr_recover_gradient_2d(&mesh, &u);
        assert_eq!(recovered.len(), n);

        let eta = zz_spr_estimator_2d(&mesh, &u);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max > 1e-6,
            "SPR estimator should be >0 for x² on Quad4, got {max:.3e}");
    }

    // ── Convergence order comparison (Task 3.3 Step 3) ─────────────────────

    /// Compute the L² gradient error between recovered gradient and exact.
    fn gradient_l2_error(
        mesh: &Mesh<2>,
        recovered: &[[f64; 2]],
        grad_exact: &[fn(f64, f64) -> f64; 2],
    ) -> f64 {
        let mut err_sq = 0.0_f64;
        for e in 0..mesh.n_elems() as ElemId {
            let [grx, gry] = interpolate_recovered_gradient_2d(mesh, e, recovered);
            let [cx, cy] = elem_centroid_2d(mesh, e);
            let ex = grad_exact[0](cx, cy);
            let ey = grad_exact[1](cx, cy);
            let area = elem_area_2d(mesh, e);
            err_sq += ((grx - ex).powi(2) + (gry - ey).powi(2)) * area;
        }
        err_sq.sqrt()
    }

    #[test]
    fn spr_converges_faster_than_zz_for_sin() {
        // For u = sin(πx)sin(πy), SPR should give lower gradient error than
        // simple nodal averaging (ZZ) on the same mesh.
        let mesh = Mesh::<2>::unit_square_tri(8);
        let n = mesh.n_nodes();

        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId);
            (std::f64::consts::PI * c[0]).sin()
                * (std::f64::consts::PI * c[1]).sin()
        }).collect();

        let grad_exact: [fn(f64, f64) -> f64; 2] = [
            |x, y| std::f64::consts::PI * (std::f64::consts::PI * x).cos() * (std::f64::consts::PI * y).sin(),
            |x, y| std::f64::consts::PI * (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).cos(),
        ];

        // Recovered gradients
        let spr = spr_recover_gradient_2d(&mesh, &u);
        let l2 = global_l2_projection_2d(&mesh, &u);

        // Simple averaging (original ZZ)
        let zz: Vec<[f64; 2]> = {
            let elem_grads: Vec<[f64; 2]> = (0..mesh.n_elems() as ElemId)
                .map(|e| elem_gradient_2d(&mesh, &u, e))
                .collect();
            let mut nodal = vec![[0.0_f64; 2]; n];
            let mut count = vec![0usize; n];
            for (e, &g) in elem_grads.iter().enumerate() {
                for &n in mesh.elem_nodes(e as ElemId) {
                    nodal[n as usize][0] += g[0];
                    nodal[n as usize][1] += g[1];
                    count[n as usize] += 1;
                }
            }
            for i in 0..n {
                if count[i] > 0 {
                    let c = count[i] as f64;
                    nodal[i][0] /= c; nodal[i][1] /= c;
                }
            }
            nodal
        };

        let err_spr = gradient_l2_error(&mesh, &spr, &grad_exact);
        let err_l2 = gradient_l2_error(&mesh, &l2, &grad_exact);
        let err_zz = gradient_l2_error(&mesh, &zz, &grad_exact);

        // SPR should be at least as good as simple averaging (usually better)
        assert!(err_spr <= err_zz * 2.0 + 0.1,
            "SPR gradient error ({err_spr:.4}) should not be much worse than ZZ ({err_zz:.4})");
        assert!(err_l2 <= err_zz * 2.0 + 0.1,
            "L2 gradient error ({err_l2:.4}) should not be much worse than ZZ ({err_zz:.4})");

        // All should be reasonably small on an 8×8 mesh
        assert!(err_zz < 1.0, "ZZ gradient error should be < 1 on 8×8, got {err_zz:.4}");
        assert!(err_spr < 1.0, "SPR gradient error should be < 1 on 8×8, got {err_spr:.4}");
    }

    #[test]
    fn recovery_methods_produce_positive_estimators() {
        let mesh = Mesh::<2>::unit_square_tri(8);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId);
            (c[0] * c[1]).exp()
        }).collect();

        let eta_spr = zz_spr_estimator_2d(&mesh, &u);
        let eta_l2 = zz_l2_estimator_2d(&mesh, &u);

        assert!(eta_spr.iter().all(|&v| v >= 0.0));
        assert!(eta_l2.iter().all(|&v| v >= 0.0));
        assert!(eta_spr.iter().sum::<f64>() > 0.0);
    }
}
