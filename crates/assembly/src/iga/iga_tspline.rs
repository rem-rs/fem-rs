//! T-spline finite element assembly (Phase 3.3).
//!
//! Provides element-by-element assembly of the diffusion stiffness matrix
//! and mass matrix for a 2-D T-spline mesh using direct basis evaluation
//! (without Bezier extraction).
//!
//! # Assembly loop
//!
//! For each cell (element):
//! 1. Find active vertices whose T-spline blending functions overlap the cell.
//! 2. For each Gauss quadrature point, evaluate all active blending functions
//!    and their parametric gradients.
//! 3. Weight by the Jacobian determinant (from the physical map through
//!    control-point coordinates) and accumulate into the global matrix.
//!
//! # DOF numbering
//!
//! Global DOF `a` corresponds to [`TVertex`](fem_element::tmesh::TVertex) at
//! index `a` in [`TMesh2D::vertices`](fem_element::tmesh::TMesh2D::vertices).
//! T-junction positions have no DOF — they are not in the vertex list.

use fem_element::quadrature::gauss_legendre_01;
use fem_element::tmesh::{TCell, TMesh2D};
use fem_linalg::{CooMatrix, CsrMatrix};

// ─── Physical map helpers ───────────────────────────────────────────────────

/// Compute the physical Jacobian determinant for a T-spline cell at parametric
/// point `(u, v)`.
///
/// Uses the control-point coordinates to build the physical map:
///   x(u,v) = Σ_a R_a(u,v) * x_a
///
/// Returns `(x, y, det_j, jac_inv_t)`.
fn tspline_physical_map(
    tmesh: &TMesh2D,
    cell: &TCell,
    u: f64,
    v: f64,
    phi: &[f64],
    grads_u: &[f64],
    grads_v: &[f64],
) -> (f64, f64, f64, [[f64; 2]; 2]) {
    let active = tmesh.find_active_vertices(cell);

    // Physical coordinates and Jacobian.
    let mut x = 0.0;
    let mut y = 0.0;
    let mut dx_du = 0.0;
    let mut dx_dv = 0.0;
    let mut dy_du = 0.0;
    let mut dy_dv = 0.0;

    for (a, &vidx) in active.iter().enumerate() {
        let vtx = &tmesh.vertices[vidx];
        let w = vtx.weight;
        let r = phi[a] * w; // NURBS-weighted value
        let dru = grads_u[a] * w;
        let drv = grads_v[a] * w;

        x += r * vtx.x;
        y += r * vtx.y;
        dx_du += dru * vtx.x;
        dx_dv += drv * vtx.x;
        dy_du += dru * vtx.y;
        dy_dv += drv * vtx.y;
    }

    let det_j = dx_du * dy_dv - dx_dv * dy_du;
    let inv_det = 1.0 / det_j;

    // J^{-T}: [ [dy_dv, -dy_du], [-dx_dv, dx_du] ] / det
    let jac_inv_t = [
        [dy_dv * inv_det, -dy_du * inv_det],
        [-dx_dv * inv_det, dx_du * inv_det],
    ];

    (x, y, det_j, jac_inv_t)
}

// ─── Diffusion assembly ─────────────────────────────────────────────────────

/// Assemble the diffusion stiffness matrix for a 2-D T-spline mesh.
///
/// ```text
/// K_{ab} = ∫_Ω κ ∇R_a · ∇R_b dΩ
/// ```
///
/// # Arguments
///
/// * `tmesh` — the T-mesh (vertices, cells, knot vectors).
/// * `kappa` — diffusion coefficient (constant).
/// * `quad_order` — Gauss quadrature order (number of points per direction).
///
/// # Returns
///
/// The global stiffness matrix as a CSR matrix.
pub fn assemble_tspline_diffusion_2d(
    tmesh: &TMesh2D,
    kappa: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let n_dofs = tmesh.vertices.len();
    let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);

    let n_q = ((quad_order as usize + 2) / 2).max(1);
    let (qpts, qwts) = gauss_legendre_01(n_q);

    for cell in &tmesh.cells {
        let u0 = tmesh.unique_u[cell.iu_min];
        let u1 = tmesh.unique_u[cell.iu_max];
        let v0 = tmesh.unique_v[cell.iv_min];
        let v1 = tmesh.unique_v[cell.iv_max];
        let hu = u1 - u0;
        let hv = v1 - v0;

        let active = tmesh.find_active_vertices(cell);
        let n_active = active.len();

        let mut phi = vec![0.0; n_active];
        let mut grads_u = vec![0.0; n_active];
        let mut grads_v = vec![0.0; n_active];

        for (i, &xi) in qpts.iter().enumerate() {
            for (j, &eta) in qpts.iter().enumerate() {
                tmesh.eval_cell(cell, xi, eta, &mut phi, &mut grads_u, &mut grads_v);

                // Physical coordinates at this point.
                let (_x, _y, det_j, jac_inv_t) = tspline_physical_map(
                    tmesh, cell, u0 + xi * hu, v0 + eta * hv,
                    &phi, &grads_u, &grads_v,
                );

                // Parametric → physical gradient transformation:
                // ∇_x R = J^{-T} · ∇_ξ R
                let w = qwts[i] * qwts[j] * hu * hv * det_j.abs();

                for a in 0..n_active {
                    let dru = grads_u[a];
                    let drv = grads_v[a];
                    // Physical gradients
                    let drx = jac_inv_t[0][0] * dru + jac_inv_t[0][1] * drv;
                    let dry = jac_inv_t[1][0] * dru + jac_inv_t[1][1] * drv;

                    let ga = active[a];
                    for b in 0..n_active {
                        let drub = grads_u[b];
                        let drvb = grads_v[b];
                        let drxb = jac_inv_t[0][0] * drub + jac_inv_t[0][1] * drvb;
                        let dryb = jac_inv_t[1][0] * drub + jac_inv_t[1][1] * drvb;

                        let dot = drx * drxb + dry * dryb;
                        coo.add(ga, active[b], kappa * dot * w);
                    }
                }
            }
        }
    }

    coo.into_csr()
}

/// Assemble the mass matrix for a 2-D T-spline mesh.
///
/// ```text
/// M_{ab} = ∫_Ω ρ R_a R_b dΩ
/// ```
pub fn assemble_tspline_mass_2d(
    tmesh: &TMesh2D,
    rho: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let n_dofs = tmesh.vertices.len();
    let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);

    let n_q = ((quad_order as usize + 2) / 2).max(1);
    let (qpts, qwts) = gauss_legendre_01(n_q);

    for cell in &tmesh.cells {
        let u0 = tmesh.unique_u[cell.iu_min];
        let u1 = tmesh.unique_u[cell.iu_max];
        let v0 = tmesh.unique_v[cell.iv_min];
        let v1 = tmesh.unique_v[cell.iv_max];
        let hu = u1 - u0;
        let hv = v1 - v0;

        let active = tmesh.find_active_vertices(cell);
        let n_active = active.len();

        let mut phi = vec![0.0; n_active];
        let mut grads_u = vec![0.0; n_active];
        let mut grads_v = vec![0.0; n_active];

        for (i, &xi) in qpts.iter().enumerate() {
            for (j, &eta) in qpts.iter().enumerate() {
                tmesh.eval_cell(cell, xi, eta, &mut phi, &mut grads_u, &mut grads_v);

                let (_x, _y, det_j, _) = tspline_physical_map(
                    tmesh, cell, u0 + xi * hu, v0 + eta * hv,
                    &phi, &grads_u, &grads_v,
                );

                let w = qwts[i] * qwts[j] * hu * hv * det_j.abs();

                for a in 0..n_active {
                    let ga = active[a];
                    for b in 0..n_active {
                        coo.add(ga, active[b], rho * phi[a] * phi[b] * w);
                    }
                }
            }
        }
    }

    coo.into_csr()
}

/// Assemble the load vector for a 2-D T-spline mesh.
///
/// ```text
/// f_a = ∫_Ω source(x,y) R_a dΩ
/// ```
///
/// `source` receives the physical coordinate `(x, y)` and returns the source value.
pub fn assemble_tspline_load_2d(
    tmesh: &TMesh2D,
    source: impl Fn(f64, f64) -> f64,
    quad_order: u8,
) -> Vec<f64> {
    let n_dofs = tmesh.vertices.len();
    let mut rhs = vec![0.0; n_dofs];

    let n_q = ((quad_order as usize + 2) / 2).max(1);
    let (qpts, qwts) = gauss_legendre_01(n_q);

    for cell in &tmesh.cells {
        let u0 = tmesh.unique_u[cell.iu_min];
        let u1 = tmesh.unique_u[cell.iu_max];
        let v0 = tmesh.unique_v[cell.iv_min];
        let v1 = tmesh.unique_v[cell.iv_max];
        let hu = u1 - u0;
        let hv = v1 - v0;

        let active = tmesh.find_active_vertices(cell);
        let n_active = active.len();

        let mut phi = vec![0.0; n_active];
        let mut grads_u = vec![0.0; n_active];
        let mut grads_v = vec![0.0; n_active];

        for (i, &xi) in qpts.iter().enumerate() {
            for (j, &eta) in qpts.iter().enumerate() {
                tmesh.eval_cell(cell, xi, eta, &mut phi, &mut grads_u, &mut grads_v);

                let (x_phys, y_phys, det_j, _) = tspline_physical_map(
                    tmesh, cell, u0 + xi * hu, v0 + eta * hv,
                    &phi, &grads_u, &grads_v,
                );

                let w = qwts[i] * qwts[j] * hu * hv * det_j.abs();
                let f_val = source(x_phys, y_phys);

                for a in 0..n_active {
                    rhs[active[a]] += f_val * phi[a] * w;
                }
            }
        }
    }

    rhs
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_element::tmesh::uniform_tspline_2d;

    /// For a uniform tensor-product T-mesh equal to a B-spline, the T-spline
    /// stiffness matrix should match the standard IGA diffusion matrix.
    #[test]
    fn tspline_diffusion_2d_matches_iga_bspline() {
        let nu = 4;
        let nv = 4;
        let pu = 1;
        let pv = 1;

        // Build a T-spline mesh equivalent to a 4×4 Q1 B-spline.
        let tmesh = uniform_tspline_2d(nu, nv, pu, pv);

        let k_tspline = assemble_tspline_diffusion_2d(&tmesh, 1.0, 3);

        // Build the analogous IGA B-spline mesh.
        use fem_element::iga::{NurbsKnotVector, NurbsMesh2D};
        let kv = NurbsKnotVector::uniform(pu, nu - pu);
        let n_basis = kv.n_basis();
        assert_eq!(n_basis, nu);
        let ctrl: Vec<[f64; 2]> = (0..n_basis * n_basis)
            .map(|idx| {
                let i = idx % n_basis;
                let j = idx / n_basis;
                [
                    i as f64 / (n_basis - 1).max(1) as f64,
                    j as f64 / (n_basis - 1).max(1) as f64,
                ]
            })
            .collect();
        let mesh = NurbsMesh2D::single_patch(kv.clone(), kv.clone(), ctrl, vec![1.0; n_basis * n_basis]);
        let k_bspline = crate::iga::iga::assemble_iga_diffusion_2d(&mesh, 1.0, 3);

        // Both matrices should be identical.
        assert_eq!(k_tspline.nrows, k_bspline.nrows);
        assert_eq!(k_tspline.ncols, k_bspline.ncols);

        // Compare entries in the dense representation.
        let n = k_tspline.nrows;
        let mut dense_t = vec![0.0; n * n];
        let mut dense_b = vec![0.0; n * n];
        for i in 0..n {
            for p in k_tspline.row_ptr[i]..k_tspline.row_ptr[i + 1] {
                dense_t[i * n + k_tspline.col_idx[p] as usize] = k_tspline.values[p];
            }
            for p in k_bspline.row_ptr[i]..k_bspline.row_ptr[i + 1] {
                dense_b[i * n + k_bspline.col_idx[p] as usize] = k_bspline.values[p];
            }
        }
        for i in 0..n {
            for j in 0..n {
                let diff = (dense_t[i * n + j] - dense_b[i * n + j]).abs();
                assert!(
                    diff < 1e-10,
                    "Mismatch at ({i},{j}): T-spline={:.10e}, B-spline={:.10e}",
                    dense_t[i * n + j],
                    dense_b[i * n + j],
                );
            }
        }
    }

    /// Stiffness matrix symmetry for a uniform T-spline mesh.
    #[test]
    fn tspline_stiffness_2d_is_symmetric() {
        let tmesh = uniform_tspline_2d(5, 5, 2, 2);
        let k = assemble_tspline_diffusion_2d(&tmesh, 1.0, 4);
        let n = k.nrows;
        let mut dense = vec![0.0; n * n];
        for i in 0..n {
            for p in k.row_ptr[i]..k.row_ptr[i + 1] {
                dense[i * n + k.col_idx[p] as usize] = k.values[p];
            }
        }
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i * n + j] - dense[j * n + i]).abs();
                assert!(
                    diff < 1e-12,
                    "K[{i},{j}] != K[{j},{i}]: diff={diff:.3e}",
                );
            }
        }
    }

    /// Mass matrix row sums should total the domain area (partition of unity).
    #[test]
    fn tspline_mass_2d_row_sum_equals_area() {
        // Use degree 1: unique knot values equal the Greville abscissae,
        // so the physical map is exactly the identity.
        let tmesh = uniform_tspline_2d(5, 5, 1, 1);
        let m = assemble_tspline_mass_2d(&tmesh, 1.0, 4);
        let n = m.nrows;
        let total: f64 = (0..n)
            .map(|a| {
                (0..n)
                    .map(|b| {
                        m.values[m.row_ptr[a]..m.row_ptr[a + 1]]
                            .iter()
                            .zip(&m.col_idx[m.row_ptr[a]..m.row_ptr[a + 1]])
                            .find(|(_, &c)| c as usize == b)
                            .map(|(v, _)| *v)
                            .unwrap_or(0.0)
                    })
                    .sum::<f64>()
            })
            .sum();
        assert!(
            (total - 1.0).abs() < 1e-8,
            "Total mass = {total:.6}, expected 1.0",
        );
    }

    /// Load vector with unit source should sum to area = 1.0.
    #[test]
    fn tspline_load_2d_unit_source_sums_to_area() {
        // Use degree 1: unique knot values equal the Greville abscissae,
        // so the physical map is exactly the identity.
        let tmesh = uniform_tspline_2d(5, 5, 1, 1);
        let rhs = assemble_tspline_load_2d(&tmesh, |_, _| 1.0, 4);
        let total: f64 = rhs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-8,
            "Load sum = {total:.6}, expected 1.0",
        );
    }

    /// Solve a Poisson problem with a known manufactured solution on a
    /// T-spline mesh and verify that refining the mesh reduces the error.
    #[test]
    fn tspline_poisson_2d_l2_error_decreases_with_refinement() {
        fn l2_error(nu: usize, nv: usize, pu: usize, pv: usize) -> f64 {
            let tmesh = uniform_tspline_2d(nu, nv, pu, pv);
            let n_dofs = tmesh.vertices.len();

            let mut k = assemble_tspline_diffusion_2d(&tmesh, 1.0, 4);
            let rhs = assemble_tspline_load_2d(&tmesh, |x, y| {
                2.0 * y * (1.0 - y) + 2.0 * x * (1.0 - x)
            }, 4);

            // Boundary DOFs: i=0, i=nu-1, j=0, j=nv-1
            let mut bc_dofs = Vec::new();
            for (idx, vtx) in tmesh.vertices.iter().enumerate() {
                if vtx.iu == 0 || vtx.iu == nu - 1 || vtx.iv == 0 || vtx.iv == nv - 1 {
                    bc_dofs.push(idx);
                }
            }
            bc_dofs.sort_unstable();
            bc_dofs.dedup();

            // Apply Dirichlet BCs by zeroing rows/cols.
            {
                let bc_set: std::collections::HashSet<usize> =
                    bc_dofs.iter().copied().collect();
                for &d in &bc_dofs {
                    if d < n_dofs {
                        for p in k.row_ptr[d]..k.row_ptr[d + 1] {
                            let col = k.col_idx[p] as usize;
                            k.values[p] = if col == d { 1.0 } else { 0.0 };
                        }
                    }
                }
                for i in 0..n_dofs {
                    if bc_set.contains(&i) {
                        continue;
                    }
                    for p in k.row_ptr[i]..k.row_ptr[i + 1] {
                        let col = k.col_idx[p] as usize;
                        if bc_set.contains(&col) {
                            k.values[p] = 0.0;
                        }
                    }
                }
            }
            let mut rhs_bc = rhs.clone();
            for &d in &bc_dofs {
                if d < n_dofs {
                    rhs_bc[d] = 0.0;
                }
            }

            // Solve via dense LU.
            use nalgebra::{DMatrix, DVector};
            let mut dense = DMatrix::<f64>::zeros(n_dofs, n_dofs);
            for i in 0..n_dofs {
                for p in k.row_ptr[i]..k.row_ptr[i + 1] {
                    dense[(i, k.col_idx[p] as usize)] = k.values[p];
                }
            }
            let b = DVector::from_column_slice(&rhs_bc);
            let u = dense.lu().solve(&b).unwrap_or_else(|| DVector::zeros(n_dofs));
            let u_vec: Vec<f64> = u.iter().cloned().collect();

            // Compute L2 error (use fixed 4-point Gauss for degree-1 integrand).
            let n_q = 4;
            let (qpts, qwts) = gauss_legendre_01(n_q);
            let mut err_sq = 0.0;
            for cell in &tmesh.cells {
                let u0 = tmesh.unique_u[cell.iu_min];
                let u1 = tmesh.unique_u[cell.iu_max];
                let v0 = tmesh.unique_v[cell.iv_min];
                let v1 = tmesh.unique_v[cell.iv_max];
                let hu = u1 - u0;
                let hv = v1 - v0;
                let active = tmesh.find_active_vertices(cell);
                let n_active = active.len();
                let mut phi = vec![0.0; n_active];
                let mut gu = vec![0.0; n_active];
                let mut gv = vec![0.0; n_active];
                for (i, &xi) in qpts.iter().enumerate() {
                    for (j, &eta) in qpts.iter().enumerate() {
                        tmesh.eval_cell(cell, xi, eta, &mut phi, &mut gu, &mut gv);

                        let (x_phys, y_phys, det_j, _) = tspline_physical_map(
                            &tmesh,
                            cell,
                            u0 + xi * hu,
                            v0 + eta * hv,
                            &phi,
                            &gu,
                            &gv,
                        );
                        let w = qwts[i] * qwts[j] * hu * hv * det_j.abs();
                        let u_exact = x_phys * (1.0 - x_phys) * y_phys * (1.0 - y_phys);
                        let u_h: f64 = phi.iter().zip(&active).map(|(r, &aidx)| r * u_vec[aidx]).sum();
                        err_sq += (u_exact - u_h).powi(2) * w;
                    }
                }
            }
            err_sq.sqrt()
        }

        // Test with degree 1 (should have O(h^2) convergence).
        let e_coarse = l2_error(4, 4, 1, 1);
        let e_fine = l2_error(8, 8, 1, 1);
        assert!(
            e_fine < e_coarse,
            "L2 error should decrease: coarse={e_coarse:.3e}, fine={e_fine:.3e}",
        );
        let ratio = e_coarse / e_fine;
        assert!(
            ratio > 1.5,
            "Expected at least O(h) convergence; got ratio={ratio:.2}",
        );
    }
}
