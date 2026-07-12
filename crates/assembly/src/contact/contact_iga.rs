//! IGA-specific Nitsche contact for 2D linear elasticity (Phase 2.5).
//!
//! Provides:
//! - [`iga_boundary_edges_2d`] -- enumerate boundary elements on a NURBS patch edge
//! - [`iga_boundary_normal_2d`] -- outward unit normal at a NURBS boundary point
//! - [`assemble_nitsche_contact_2d`] -- Nitsche contact matrix and RHS assembly
//!
//! The Nitsche form enforces normal-displacement continuity across a conforming
//! NURBS interface using a penalty-like (incomplete) Nitsche method.  Only the
//! penalty term is implemented; the stress-consistency terms (σₙ) are reserved
//! for a follow-up.

use fem_element::iga::{NurbsPatch2D, NurbsPatch2DData};
use fem_element::quadrature::seg_rule;
use fem_element::ReferenceElement;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_space::IgaBoundary2D;

// ─── helpers ──────────────────────────────────────────────────────────────────

/// Return non‑empty knot spans `(start, end)` from a knot‑value slice.
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

// ─── public API ───────────────────────────────────────────────────────────────

/// Enumerate boundary elements on a NURBS patch edge.
///
/// For each knot span on the given boundary edge, return a tuple
/// `(edge_param_coord, local_knot_span_index)`.
///
/// | side               | iteration direction | returned coordinate |
/// |--------------------|---------------------|---------------------|
/// | `VMin` / `VMax`    | *u*-spans           | mid‑point *u*       |
/// | `UMin` / `UMax`    | *v*-spans           | mid‑point *v*       |
pub fn iga_boundary_edges_2d(
    pd: &NurbsPatch2DData,
    side: IgaBoundary2D,
) -> Vec<(f64, usize)> {
    let kv = match side {
        IgaBoundary2D::UMin | IgaBoundary2D::UMax => &pd.kv_v,
        IgaBoundary2D::VMin | IgaBoundary2D::VMax => &pd.kv_u,
    };
    nonempty_spans(&kv.knots)
        .into_iter()
        .map(|(idx, a, b)| (0.5 * (a + b), idx))
        .collect()
}

/// Compute the outward unit normal at a point on a NURBS patch boundary.
///
/// `xi` is the full 2‑D parametric coordinate `[u, v]` **on the boundary**.
/// The normal is derived from the NURBS Jacobian `J`:
///
/// | side   | tangent       | outward normal                     |
/// |--------|---------------|------------------------------------|
/// | `VMin` | `(dx/du, dy/du)` | `(dy/du, -dx/du) / |tangent|` |
/// | `VMax` | `(dx/du, dy/du)` | `(-dy/du, dx/du) / |tangent|` |
/// | `UMin` | `(dx/dv, dy/dv)` | `(-dy/dv, dx/dv) / |tangent|` |
/// | `UMax` | `(dx/dv, dy/dv)` | `(dy/dv, -dx/dv) / |tangent|` |
pub fn iga_boundary_normal_2d(
    pd: &NurbsPatch2DData,
    xi: &[f64],
    side: IgaBoundary2D,
) -> [f64; 2] {
    let map = crate::iga::physical_map_2d(pd, xi);
    let jac = &map.jac;

    match side {
        IgaBoundary2D::VMin => {
            // Bottom edge v=0: outward = (dy/du, -dx/du)
            let tx = jac[0][0];
            let ty = jac[1][0];
            let len = (tx * tx + ty * ty).sqrt().max(1e-30);
            [ty / len, -tx / len]
        }
        IgaBoundary2D::VMax => {
            // Top edge v=1: outward = (-dy/du, dx/du)
            let tx = jac[0][0];
            let ty = jac[1][0];
            let len = (tx * tx + ty * ty).sqrt().max(1e-30);
            [-ty / len, tx / len]
        }
        IgaBoundary2D::UMin => {
            // Left edge u=0: outward = (-dy/dv, dx/dv)
            let tx = jac[0][1];
            let ty = jac[1][1];
            let len = (tx * tx + ty * ty).sqrt().max(1e-30);
            [-ty / len, tx / len]
        }
        IgaBoundary2D::UMax => {
            // Right edge u=1: outward = (dy/dv, -dx/dv)
            let tx = jac[0][1];
            let ty = jac[1][1];
            let len = (tx * tx + ty * ty).sqrt().max(1e-30);
            [ty / len, -tx / len]
        }
    }
}

/// Assemble the Nitsche contact contribution for 2‑D linear elasticity between
/// two NURBS patches at a **conforming** interface.
///
/// # Simplified (penalty) form
///
/// For frictionless contact only the penalty term of the symmetric Nitsche
/// method is assembled (the stress‑consistency terms are omitted):
///
/// ```text
/// K_contact[a, b] ← γ (N_a·n)(N_b·n) · w · |J_b|
/// f_contact[a]   ← γ·gap·(N_a·n)          · w · |J_b|
/// ```
///
/// where `n` is the outward normal of the **slave** surface,
/// `gap = (x_master − x_slave)·n` is the signed initial gap,
/// and `γ` is the Nitsche penalty parameter.
///
/// # DOF ordering
///
/// The returned matrix and RHS vector order **slave DOFs first**, then **master
/// DOFs**.  For control‑point index `c`:
///
/// | range                 | meaning                    |
/// |-----------------------|----------------------------|
/// | `2·c`                 | slave, x‑component         |
/// | `2·c + 1`             | slave, y‑component         |
/// | `2·n_slave + 2·c`     | master, x‑component        |
/// | `2·n_slave + 2·c + 1` | master, y‑component        |
///
/// # Panics
///
/// Panics if the boundary‑edge pair is non‑conforming (not opposite edges,
/// e.g. `(VMax, VMin)` or `(UMin, UMax)`).
///
/// # Usage
///
/// ```ignore
/// let (k_nitsche, f_nitsche) = assemble_nitsche_contact_2d(
///     &pd_slave, IgaBoundary2D::VMax,
///     &pd_master, IgaBoundary2D::VMin,
///     lambda, mu, gamma, 4,
/// );
/// // Combine with elasticity:
/// //   K_total = block_diag(K_slave, K_master) + k_nitsche
/// //   RHS_total = [f_slave; f_master] + f_nitsche
/// ```
pub fn assemble_nitsche_contact_2d(
    pd_slave: &NurbsPatch2DData,
    side_slave: IgaBoundary2D,
    pd_master: &NurbsPatch2DData,
    side_master: IgaBoundary2D,
    lambda: f64,
    mu: f64,
    gamma: f64,
    quad_order: u8,
) -> (CsrMatrix<f64>, Vec<f64>) {
    let n_slave = pd_slave.control_pts.len();
    let n_master = pd_master.control_pts.len();
    let n_total = 2 * (n_slave + n_master);
    let mut coo = CooMatrix::new(n_total, n_total);
    let mut rhs = vec![0.0; n_total];

    // Unused in the simplified form – kept in the signature for future
    // extension to the full symmetric Nitsche with stress terms.
    let _ = lambda;
    let _ = mu;

    // Build NURBS patch evaluators and cache sizes.
    let patch_slave = NurbsPatch2D::new(
        pd_slave.kv_u.clone(),
        pd_slave.kv_v.clone(),
        pd_slave.weights.clone(),
    );
    let patch_master = NurbsPatch2D::new(
        pd_master.kv_u.clone(),
        pd_master.kv_v.clone(),
        pd_master.weights.clone(),
    );
    let n_dof_s = patch_slave.n_dofs();
    let n_dof_m = patch_master.n_dofs();

    // Determine the parametric direction that runs **along** the interface,
    // and the fixed parametric coordinate on each side.
    let (along_kv, fixed_slave, fixed_master) = match (side_slave, side_master) {
        (IgaBoundary2D::VMin, IgaBoundary2D::VMax) => (&pd_slave.kv_u, 0.0, 1.0),
        (IgaBoundary2D::VMax, IgaBoundary2D::VMin) => (&pd_slave.kv_u, 1.0, 0.0),
        (IgaBoundary2D::UMin, IgaBoundary2D::UMax) => (&pd_slave.kv_v, 0.0, 1.0),
        (IgaBoundary2D::UMax, IgaBoundary2D::UMin) => (&pd_slave.kv_v, 1.0, 0.0),
        _ => panic!(
            "Nitsche contact: side_slave={side_slave:?} and side_master={side_master:?} \
             must be opposite edges (e.g. VMax/VMin or UMin/UMax)"
        ),
    };

    let is_uhorizontal = matches!(side_slave, IgaBoundary2D::VMin | IgaBoundary2D::VMax);

    // 1‑D Gauss–Legendre quadrature on [0, 1].
    let seg = seg_rule(quad_order);
    let qpts = &seg.points; // each is Vec<f64> holding ξ ∈ [0,1]
    let qwts = &seg.weights;

    // Pre‑allocate scratch buffers.
    let mut basis_s = vec![0.0; n_dof_s];
    let mut basis_m = vec![0.0; n_dof_m];

    // Gradient scratch (used for the stress terms in future).
    let _grad_s = vec![0.0; n_dof_s * 2];
    let _grad_m = vec![0.0; n_dof_m * 2];

    // Iterate over non‑empty knot spans along the interface direction.
    for (_, t0, t1) in nonempty_spans(&along_kv.knots) {
        let h = t1 - t0; // parametric span length

        for (qp, qw) in qpts.iter().zip(qwts.iter()) {
            let xi = qp[0]; // ξ ∈ [0, 1]
            let t = t0 + h * xi; // parameter along the edge

            // ── Slave evaluation at (u, v) on the boundary ────────
            let xi_s = if is_uhorizontal {
                [t, fixed_slave]
            } else {
                [fixed_slave, t]
            };

            patch_slave.eval_basis(&xi_s, &mut basis_s);
            let map_s = crate::iga::physical_map_2d(pd_slave, &xi_s);
            let n = iga_boundary_normal_2d(pd_slave, &xi_s, side_slave);

            // |tangent| of the boundary curve
            let tang_len = if is_uhorizontal {
                let tx = map_s.jac[0][0];
                let ty = map_s.jac[1][0];
                (tx * tx + ty * ty).sqrt()
            } else {
                let tx = map_s.jac[0][1];
                let ty = map_s.jac[1][1];
                (tx * tx + ty * ty).sqrt()
            };
            let w = qw * h * tang_len;

            // ── Master evaluation at the corresponding point ──────
            let xi_m = if is_uhorizontal {
                [t, fixed_master]
            } else {
                [fixed_master, t]
            };

            patch_master.eval_basis(&xi_m, &mut basis_m);
            let map_m = crate::iga::physical_map_2d(pd_master, &xi_m);

            // Signed gap: master minus slave projected onto n.
            let gap = (map_m.x_phys[0] - map_s.x_phys[0]) * n[0]
                + (map_m.x_phys[1] - map_s.x_phys[1]) * n[1];

            // ── Assemble the four blocks of the Nitsche penalty ───
            //
            // The jump form of the Nitsche penalty:
            //   ∫_Γc γ ( [[u]]·n ) ( [[v]]·n ) dΓ
            // with [[u]] = u_s − u_m, [[v]] = v_s − v_m.
            //
            //   S‑S  block: +γ (R_a·n)(R_b·n)
            //   S‑M  block: −γ (R_a·n)(R_b·n)
            //   M‑S  block: −γ (R_a·n)(R_b·n)
            //   M‑M  block: +γ (R_a·n)(R_b·n)
            //
            // RHS (from the gap, moved to RHS):
            //   −γ·gap·(v_s·n − v_m·n)

            let dof_off_m = 2 * n_slave;

            // Pre‑compute R_a·n for every slave DOF.
            let rn_s: Vec<[f64; 2]> = (0..n_dof_s)
                .map(|a| {
                    [
                        basis_s[a] * n[0],
                        basis_s[a] * n[1],
                    ]
                })
                .collect();
            let rn_m: Vec<[f64; 2]> = (0..n_dof_m)
                .map(|a| {
                    [
                        basis_m[a] * n[0],
                        basis_m[a] * n[1],
                    ]
                })
                .collect();

            // ── S‑S block ──────────────────────────────────
            for a in 0..n_dof_s {
                let ga_sx = a * 2;
                let ga_sy = a * 2 + 1;
                let rna = &rn_s[a];

                // RHS: −γ·gap·(v_s·n)
                rhs[ga_sx] += -gamma * gap * rna[0] * w;
                rhs[ga_sy] += -gamma * gap * rna[1] * w;

                for b in 0..n_dof_s {
                    let gb_sx = b * 2;
                    let gb_sy = b * 2 + 1;
                    let rnb = &rn_s[b];
                    let k = gamma * w;
                    coo.add(ga_sx, gb_sx, k * rna[0] * rnb[0]);
                    coo.add(ga_sx, gb_sy, k * rna[0] * rnb[1]);
                    coo.add(ga_sy, gb_sx, k * rna[1] * rnb[0]);
                    coo.add(ga_sy, gb_sy, k * rna[1] * rnb[1]);
                }
            }

            // ── S‑M block ──────────────────────────────────
            for a in 0..n_dof_s {
                let ga_sx = a * 2;
                let ga_sy = a * 2 + 1;
                let rna = &rn_s[a];
                for b in 0..n_dof_m {
                    let gb_mx = dof_off_m + b * 2;
                    let gb_my = dof_off_m + b * 2 + 1;
                    let rnb = &rn_m[b];
                    let k = -gamma * w;
                    coo.add(ga_sx, gb_mx, k * rna[0] * rnb[0]);
                    coo.add(ga_sx, gb_my, k * rna[0] * rnb[1]);
                    coo.add(ga_sy, gb_mx, k * rna[1] * rnb[0]);
                    coo.add(ga_sy, gb_my, k * rna[1] * rnb[1]);
                }
            }

            // ── M‑S block ──────────────────────────────────
            for a in 0..n_dof_m {
                let ga_mx = dof_off_m + a * 2;
                let ga_my = dof_off_m + a * 2 + 1;
                let rna = &rn_m[a];

                // RHS: +γ·gap·(v_m·n)
                rhs[ga_mx] += gamma * gap * rna[0] * w;
                rhs[ga_my] += gamma * gap * rna[1] * w;

                for b in 0..n_dof_s {
                    let gb_sx = b * 2;
                    let gb_sy = b * 2 + 1;
                    let rnb = &rn_s[b];
                    let k = -gamma * w;
                    coo.add(ga_mx, gb_sx, k * rna[0] * rnb[0]);
                    coo.add(ga_mx, gb_sy, k * rna[0] * rnb[1]);
                    coo.add(ga_my, gb_sx, k * rna[1] * rnb[0]);
                    coo.add(ga_my, gb_sy, k * rna[1] * rnb[1]);
                }
            }

            // ── M‑M block ──────────────────────────────────
            for a in 0..n_dof_m {
                let ga_mx = dof_off_m + a * 2;
                let ga_my = dof_off_m + a * 2 + 1;
                let rna = &rn_m[a];
                for b in 0..n_dof_m {
                    let gb_mx = dof_off_m + b * 2;
                    let gb_my = dof_off_m + b * 2 + 1;
                    let rnb = &rn_m[b];
                    let k = gamma * w;
                    coo.add(ga_mx, gb_mx, k * rna[0] * rnb[0]);
                    coo.add(ga_mx, gb_my, k * rna[0] * rnb[1]);
                    coo.add(ga_my, gb_mx, k * rna[1] * rnb[0]);
                    coo.add(ga_my, gb_my, k * rna[1] * rnb[1]);
                }
            }
        }
    }

    (coo.into_csr(), rhs)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use fem_element::iga::{NurbsKnotVector, NurbsMesh2D};
    use fem_solver::solve_cg;

    /// Helper: build a unit‑square NURBS patch with degree‑1, 2×2 elements.
    fn unit_square_patch(x0: f64, y0: f64) -> NurbsPatch2DData {
        let kv = NurbsKnotVector::uniform(1, 2); // degree 1, 2 elements → 3 basis fns
        let nu = kv.n_basis(); // 3
        let nv = nu;
        let ctrl: Vec<[f64; 2]> = (0..nv)
            .flat_map(|j| {
                (0..nu).map(move |i| {
                    [
                        x0 + i as f64 / (nu - 1) as f64,
                        y0 + j as f64 / (nv - 1) as f64,
                    ]
                })
            })
            .collect();
        let weights = vec![1.0; nu * nv];
        NurbsPatch2DData {
            kv_u: kv.clone(),
            kv_v: kv,
            control_pts: ctrl,
            weights,
            tag: 1,
        }
    }

    // ── boundary edge tools ──────────────────────────────────────────────────

    #[test]
    fn iga_boundary_edges_2d_returns_correct_count() {
        let pd = unit_square_patch(0.0, 0.0);
        // Degree 1, 2 elements → 2 non‑empty u‑spans and 2 v‑spans.
        let edges_u = iga_boundary_edges_2d(&pd, IgaBoundary2D::VMin);
        assert_eq!(edges_u.len(), 2);
        assert!((edges_u[0].0 - 0.25).abs() < 1e-14);
        assert!((edges_u[1].0 - 0.75).abs() < 1e-14);

        let edges_v = iga_boundary_edges_2d(&pd, IgaBoundary2D::UMin);
        assert_eq!(edges_v.len(), 2);
    }

    #[test]
    fn iga_boundary_normal_2d_unit_square() {
        let pd = unit_square_patch(0.0, 0.0);
        let tol = 1e-13;

        // Bottom edge mid‑point: n should be (0, -1)
        let n = iga_boundary_normal_2d(&pd, &[0.5, 0.0], IgaBoundary2D::VMin);
        assert!((n[0] - 0.0).abs() < tol && (n[1] - (-1.0)).abs() < tol,
            "VMin normal: ({}, {})", n[0], n[1]);

        // Top edge mid‑point: n should be (0, 1)
        let n = iga_boundary_normal_2d(&pd, &[0.5, 1.0], IgaBoundary2D::VMax);
        assert!((n[0] - 0.0).abs() < tol && (n[1] - 1.0).abs() < tol,
            "VMax normal: ({}, {})", n[0], n[1]);

        // Left edge mid‑point: n should be (-1, 0)
        let n = iga_boundary_normal_2d(&pd, &[0.0, 0.5], IgaBoundary2D::UMin);
        assert!((n[0] - (-1.0)).abs() < tol && (n[1] - 0.0).abs() < tol,
            "UMin normal: ({}, {})", n[0], n[1]);

        // Right edge mid‑point: n should be (1, 0)
        let n = iga_boundary_normal_2d(&pd, &[1.0, 0.5], IgaBoundary2D::UMax);
        assert!((n[0] - 1.0).abs() < tol && (n[1] - 0.0).abs() < tol,
            "UMax normal: ({}, {})", n[0], n[1]);
    }

    // ── Nitsche contact assembly ─────────────────────────────────────────────

    #[test]
    fn nitsche_contact_2d_assembles_correct_size() {
        // Two blocks: slave at [0,1]×[0,1], master at [0,1]×[1,2].
        let pd_slave = unit_square_patch(0.0, 0.0);
        let pd_master = unit_square_patch(0.0, 1.0);

        let gamma = 1e5;
        let quad_order = 4;

        let (k, f) = assemble_nitsche_contact_2d(
            &pd_slave,
            IgaBoundary2D::VMax,
            &pd_master,
            IgaBoundary2D::VMin,
            100.0,
            50.0,
            gamma,
            quad_order,
        );

        let n_dofs = 2 * (pd_slave.control_pts.len() + pd_master.control_pts.len());
        assert_eq!(k.nrows, n_dofs);
        assert_eq!(k.ncols, n_dofs);
        assert_eq!(f.len(), n_dofs);
        assert!(f.iter().all(|v| v.is_finite()), "RHS has non‑finite entries");
    }

    #[test]
    fn nitsche_contact_2d_matrix_is_symmetric() {
        let pd_slave = unit_square_patch(0.0, 0.0);
        let pd_master = unit_square_patch(0.0, 1.0);

        let (k, _) = assemble_nitsche_contact_2d(
            &pd_slave,
            IgaBoundary2D::VMax,
            &pd_master,
            IgaBoundary2D::VMin,
            100.0,
            50.0,
            1e5,
            4,
        );

        // Check symmetry: K[i,j] ≈ K[j,i]
        let n = k.nrows;
        for i in 0..n.min(50) {
            let row_start = k.row_ptr[i];
            let row_end = k.row_ptr[i + 1];
            for p in row_start..row_end {
                let j = k.col_idx[p as usize] as usize;
                let kij = k.values[p as usize];
                // Find K[j,i]
                let j_start = k.row_ptr[j];
                let j_end = k.row_ptr[j + 1];
                let mut kji = 0.0;
                for q in j_start..j_end {
                    if k.col_idx[q as usize] as usize == i {
                        kji = k.values[q as usize];
                        break;
                    }
                }
                let diff = (kij - kji).abs();
                let max_val = kij.abs().max(kji.abs()).max(1e-30);
                assert!(
                    diff / max_val < 1e-12,
                    "Symmetry broken at ({i},{j}): kij={kij:.6e}, kji={kji:.6e}, diff={diff:.6e}"
                );
            }
        }
    }

    #[test]
    fn nitsche_contact_2d_rhs_is_zero_with_zero_gap() {
        // For conforming interface with zero initial gap, the RHS should be zero.
        let pd_slave = unit_square_patch(0.0, 0.0);
        let pd_master = unit_square_patch(0.0, 1.0);

        let (_, f) = assemble_nitsche_contact_2d(
            &pd_slave,
            IgaBoundary2D::VMax,
            &pd_master,
            IgaBoundary2D::VMin,
            100.0,
            50.0,
            1e5,
            4,
        );

        let fnorm: f64 = f.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            fnorm < 1e-12,
            "Expected zero RHS for zero gap, got |f| = {fnorm:.3e}"
        );
    }

    // ── Full two‑block contact solve ─────────────────────────────────────────

    #[test]
    fn nitsche_contact_2d_two_blocks() {
        // Two unit‑square NURBS patches stacked vertically:
        //   bottom (slave):  [0,1]×[0,1]
        //   top   (master):  [0,1]×[1,2]
        // Contact interface at y = 1.
        // The top block is compressed downward (prescribed u_y = −0.05 on top face).

        use fem_linalg::SolverConfig;
        use fem_solver::solve_cg;

        let pd_slave = unit_square_patch(0.0, 0.0);
        let pd_master = unit_square_patch(0.0, 1.0);

        let lambda = 100.0;
        let mu = 50.0;
        let gamma = 1e6;
        let quad_order = 4;

        // --- Assemble individual elasticity matrices ---
        let mesh_bot = NurbsMesh2D::single_patch(
            pd_slave.kv_u.clone(),
            pd_slave.kv_v.clone(),
            pd_slave.control_pts.clone(),
            pd_slave.weights.clone(),
        );
        let mesh_top = NurbsMesh2D::single_patch(
            pd_master.kv_u.clone(),
            pd_master.kv_v.clone(),
            pd_master.control_pts.clone(),
            pd_master.weights.clone(),
        );

        let k_bot = crate::iga::assemble_iga_elasticity_2d(&mesh_bot, lambda, mu, quad_order);
        let k_top = crate::iga::assemble_iga_elasticity_2d(&mesh_top, lambda, mu, quad_order);

        let n_slave = pd_slave.control_pts.len(); // 9
        let n_master = pd_master.control_pts.len(); // 9
        let n_dof_s = 2 * n_slave; // 18
        let n_dof_m = 2 * n_master; // 18
        let n_total = n_dof_s + n_dof_m; // 36

        // --- Build block‑diagonal elasticity matrix ---
        let mut coo_total = CooMatrix::new(n_total, n_total);
        // K_bot → rows/cols [0, n_dof_s)
        for i in 0..n_dof_s {
            let rs = k_bot.row_ptr[i];
            let re = k_bot.row_ptr[i + 1];
            for p in rs..re {
                let j = k_bot.col_idx[p as usize] as usize;
                let v = k_bot.values[p as usize];
                coo_total.add(i, j, v);
            }
        }
        // K_top → rows/cols [n_dof_s, n_total)
        for i in 0..n_dof_m {
            let rs = k_top.row_ptr[i];
            let re = k_top.row_ptr[i + 1];
            for p in rs..re {
                let j = k_top.col_idx[p as usize] as usize;
                let v = k_top.values[p as usize];
                coo_total.add(n_dof_s + i, n_dof_s + j, v);
            }
        }
        let mut k_total = coo_total.into_csr();

        // --- Add Nitsche contact ---
        let (k_nitsche, f_nitsche) = assemble_nitsche_contact_2d(
            &pd_slave,
            IgaBoundary2D::VMax,
            &pd_master,
            IgaBoundary2D::VMin,
            lambda,
            mu,
            gamma,
            quad_order,
        );

        // Merge K_nitsche into K_total
        let mut coo_k = CooMatrix::new(n_total, n_total);
        for i in 0..n_total {
            let rs = k_total.row_ptr[i];
            let re = k_total.row_ptr[i + 1];
            for p in rs..re {
                let j = k_total.col_idx[p as usize] as usize;
                let v = k_total.values[p as usize];
                coo_k.add(i, j, v);
            }
        }
        for i in 0..n_total {
            let rs = k_nitsche.row_ptr[i];
            let re = k_nitsche.row_ptr[i + 1];
            for p in rs..re {
                let j = k_nitsche.col_idx[p as usize] as usize;
                let v = k_nitsche.values[p as usize];
                coo_k.add(i, j, v);
            }
        }
        k_total = coo_k.into_csr();

        let mut rhs_total = vec![0.0; n_total];

        // --- Apply boundary conditions ---
        // Slave bottom face (VMin, v=0): control points 0,1,2 → fix u=0
        // Master top face (VMax, v=1): control points 6,7,8 → prescribe u_y = −0.05 (x free)
        let presc_y = -0.05;
        let mut prescribed_val = vec![0.0; n_total];

        // Slave bottom (j=0): all DOFs fixed (u_x = u_y = 0)
        for c in 0..3 {
            prescribed_val[2 * c] = 0.0; // u_x
            prescribed_val[2 * c + 1] = 0.0; // u_y
        }
        // Master top (j=2): u_y prescribed, u_x free (leave as 0 in prescribed)
        for c in 6..9 {
            let y_dof = n_dof_s + 2 * c + 1;
            prescribed_val[y_dof] = presc_y;
        }

        // Mark eliminated (prescribed) DOFs
        let eliminated: Vec<bool> = prescribed_val.iter().map(|&v| v != 0.0 || {
            // Also mark DOFs with exactly 0 that are on the slave bottom
            let c = /* figure out if on bottom */ false; // handled below
            false
        }).collect();
        // Simpler: just explicitly list constrained DOFs
        let mut eliminated = vec![false; n_total];
        for c in 0..3 { eliminated[2*c] = true; eliminated[2*c+1] = true; }
        for c in 6..9 { let d = n_dof_s + 2*c + 1; eliminated[d] = true; }

        // Build modified system: for each free DOF i, keep entries K[i,j]
        // and subtract K[i,j] * u_j^prescribed from RHS for prescribed j.
        let mut coo_mod = CooMatrix::new(n_total, n_total);
        for i in 0..n_total {
            if eliminated[i] {
                coo_mod.add(i, i, 1.0);
                rhs_total[i] = prescribed_val[i];
            } else {
                let rs = k_total.row_ptr[i];
                let re = k_total.row_ptr[i + 1];
                for p in rs..re {
                    let j = k_total.col_idx[p as usize] as usize;
                    let v = k_total.values[p as usize];
                    if eliminated[j] {
                        rhs_total[i] -= v * prescribed_val[j];
                    } else {
                        coo_mod.add(i, j, v);
                    }
                }
                // Add Nitsche RHS for free DOFs
                rhs_total[i] += f_nitsche[i];
            }
        }
        k_total = coo_mod.into_csr();

        // --- Solve ---
        let mut u = vec![0.0; n_total];
        let cfg = SolverConfig {
            rtol: 1e-10,
            max_iter: 2000,
            ..Default::default()
        };
        let res = solve_cg(&k_total, &rhs_total, &mut u, &cfg);
        assert!(res.is_ok(), "CG solver failed: {res:?}");

        // --- Verify no penetration at the contact interface ---
        // Contact interface: slave VMax (v=1) and master VMin (v=0).
        // For each slave control point on the top edge (j=2): indices 6,7,8
        //   u_n = u_slave·n_slave - u_master·n_slave
        //   At the interface: n_slave = (0, 1) (pointing up from slave)
        //   u_n = u_y_slave - u_y_master
        //   No penetration: u_n ≥ −gap = 0 → u_y_slave ≥ u_y_master
        let slave_top_cps: Vec<usize> = (6..9).collect(); // j=2 in 3×3 grid
        let master_bot_cps: Vec<usize> = (0..3).collect(); // j=0 in 3×3 grid

        for (&cs, &cm) in slave_top_cps.iter().zip(master_bot_cps.iter()) {
            let u_s_y = u[2 * cs + 1];
            let u_m_y = u[n_dof_s + 2 * cm + 1];
            // Slave should not penetrate master: u_s_y ≥ u_m_y (n = (0,1))
            assert!(
                u_s_y + 1e-12 >= u_m_y,
                "Penetration at interface: u_slave_y={:.6e} < u_master_y={:.6e}",
                u_s_y,
                u_m_y
            );
        }

        // The top block should have compressed (negative u_y on top face).
        for &c in &[6, 7, 8] {
            let u_top_y = u[n_dof_s + 2 * c + 1];
            assert!(
                u_top_y < 0.0,
                "Expected negative displacement on top face, got {:.6e}",
                u_top_y
            );
        }

        // The bottom block should be compressed (negative u_y on top face).
        for &c in &[6, 7, 8] {
            let u_bot_y = u[2 * c + 1];
            assert!(
                u_bot_y < 0.0,
                "Expected negative displacement on slave top, got {:.6e}",
                u_bot_y
            );
        }
    }
}
