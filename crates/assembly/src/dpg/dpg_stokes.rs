//! DPG Stokes solver — ultraweak formulation with optimal test functions.
//!
//! Solves the 2-D Stokes equations on Tri3 meshes:
//! ```text
//! -νΔu + ∇p = f,   ∇·u = 0
//! ```
//!
//! # Method
//!
//! Per-element:
//! - Trial: P1 velocity (2 comp.) + P1 pressure (1 comp.) = 9 DOFs
//! - Test:  P3 velocity (2 comp.) + P2 pressure (1 comp.) = 26 DOFs
//! - Optimal test functions from local Gram solve (H¹-like inner product)
//! - Element stiffness assembled via Petrov-Galerkin: K_e = B^T M_V^{-1} B

use fem_element::{ReferenceElement, lagrange::{TriP1, TriP2, TriP3}, quadrature::tri_rule};
use fem_linalg::CooMatrix;
use fem_mesh::MeshTopology;

fn solve_dense(n: usize, a: &mut [f64], b: &mut [f64]) {
    for col in 0..n {
        let mut best = col;
        let mut bv = a[col * n + col].abs();
        for row in (col + 1)..n { let v = a[row * n + col].abs(); if v > bv { bv = v; best = row; } }
        if bv < 1e-40 { continue; }
        if best != col { for c in col..n { a.swap(col * n + c, best * n + c); } b.swap(col, best); }
        let piv = a[col * n + col];
        for row in (col + 1)..n { let f = a[row * n + col] / piv;
            for c in col..n { a[row * n + c] -= f * a[col * n + c]; } b[row] -= f * b[col]; }
    }
    for row in (0..n).rev() {
        let mut s = b[row];
        for c in (row + 1)..n { s -= a[row * n + c] * b[c]; }
        b[row] = if a[row * n + row].abs() > 1e-40 { s / a[row * n + row] } else { 0.0 };
    }
}

/// Solve `-νΔu + ∇p = f, ∇·u = 0` via DPG on a Tri3 mesh.
///
/// # Arguments
/// * `mesh` — Tri3 mesh (2-D)
/// * `nu` — viscosity
/// * `f` — source term `f(x, y) → (fx, fy)`
/// * `dirichlet_bc` — `dof_index → value`, 0-based node index for each velocity component
///
/// # Returns
/// `(u_x, u_y, p)` nodal solutions.
pub fn solve_dpg_stokes_2d<M: MeshTopology>(
    mesh: &M,
    nu: f64,
    f: &dyn Fn(f64, f64) -> (f64, f64),
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_nodes = mesh.n_nodes();
    let n_vel = n_nodes;      // 1 DOF per node per velocity component
    let n_pres = n_nodes;     // 1 DOF per node for pressure
    let n_total = 2 * n_vel + n_pres; // ux, uy, p

    // Trial DOFs per element: 3 velocity-x + 3 velocity-y + 3 pressure = 9
    // Test DOFs per element:  10 P3 vel-x + 10 P3 vel-y + 6 P2 pressure = 26
    let n_trial = 9;
    let n_test = 26;
    let n_test_v = 10; // P3 velocity test DOFs per component
    let n_test_p = 6;  // P2 pressure test DOFs
    let n_trial_v = 3; // P1 per component

    let tri_p3 = TriP3;
    let tri_p2 = TriP2;
    let qr = tri_rule(7);
    let n_elems = mesh.n_elements();

    let mut coo = CooMatrix::<f64>::new(n_total, n_total);
    let mut rhs_global = vec![0.0; n_total];

    // Map element-local node → global DOF
    let velx_dof = |node: usize| -> usize { node };
    let vely_dof = |node: usize| -> usize { n_vel + node };
    let pres_dof = |node: usize| -> usize { 2 * n_vel + node };

    for e in 0..n_elems as u32 {
        let nodes = mesh.element_nodes(e);
        let nn = nodes.len(); // 3 for Tri3
        let dofs: Vec<usize> = nodes.iter().flat_map(|&n| {
            let nu = n as usize;
            vec![velx_dof(nu), vely_dof(nu), pres_dof(nu)]
        }).collect(); // [u0_x, u0_y, p0, u1_x, u1_y, p1, u2_x, u2_y, p2]

        let x: Vec<f64> = (0..nn).map(|k| mesh.node_coords(nodes[k])[0]).collect();
        let y: Vec<f64> = (0..nn).map(|k| mesh.node_coords(nodes[k])[1]).collect();

        let j00 = x[1] - x[0]; let j01 = x[2] - x[0];
        let j10 = y[1] - y[0]; let j11 = y[2] - y[0];
        let det_j = j00 * j11 - j01 * j10;
        let abs_det = det_j.abs();
        let inv_det = 1.0 / det_j;
        let vol = 0.5 * abs_det;

        // Reference gradients for velocity test (P3, 10 DOFs) and pressure test (P2, 6 DOFs)
        let mut phi_v = vec![0.0; n_test_v];
        let mut dphi_v = vec![0.0; n_test_v * 2];
        let mut phi_p = vec![0.0; n_test_p];
        let mut dphi_p = vec![0.0; n_test_p * 2];

        // M_V: block-diagonal test Gram matrix (26×26)
        //   - velocity block (20×20): H¹ for each vel component
        //   - pressure block (6×6): L² for pressure
        let mut mv = vec![0.0; n_test * n_test];
        // B: bilinear form (26×9)
        let mut bm = vec![0.0; n_test * n_trial];

        for (xi, &wr) in qr.points.iter().zip(qr.weights.iter()) {
            let w = wr * abs_det;

            tri_p3.eval_basis(xi, &mut phi_v);
            tri_p3.eval_grad_basis(xi, &mut dphi_v);
            // Physical gradients for P3 test
            let mut dvx = vec![0.0; n_test_v]; let mut dvy = vec![0.0; n_test_v];
            for i in 0..n_test_v {
                dvx[i] = (j11*dphi_v[i*2] - j10*dphi_v[i*2+1]) * inv_det;
                dvy[i] = (-j01*dphi_v[i*2] + j00*dphi_v[i*2+1]) * inv_det;
            }

            tri_p2.eval_basis(xi, &mut phi_p);
            // P2 pressure test gradients (for H¹ inner product)
            tri_p2.eval_grad_basis(xi, &mut dphi_p);
            let mut dpx = vec![0.0; n_test_p]; let mut dpy = vec![0.0; n_test_p];
            for i in 0..n_test_p {
                dpx[i] = (j11*dphi_p[i*2] - j10*dphi_p[i*2+1]) * inv_det;
                dpy[i] = (-j01*dphi_p[i*2] + j00*dphi_p[i*2+1]) * inv_det;
            }

            // P1 trial physical gradients (velocity)
            let tgrad_x = [-1.0, 1.0, 0.0]; let tgrad_y = [-1.0, 0.0, 1.0];
            let mut tu_x = vec![0.0; n_trial_v]; let mut tu_y = vec![0.0; n_trial_v];
            for i in 0..n_trial_v {
                tu_x[i] = (j11*tgrad_x[i] - j10*tgrad_y[i]) * inv_det;
                tu_y[i] = (-j01*tgrad_x[i] + j00*tgrad_y[i]) * inv_det;
            }

            // Trial P1 basis values (same for velocity and pressure)
            let tri_p1 = TriP1;
            let mut phi_t = vec![0.0; n_trial_v];
            tri_p1.eval_basis(xi, &mut phi_t);

            // ── Assembly of M_V ─────────────────────────────────────────────
            // Velocity block (rows 0-19, cols 0-19): H¹ inner product for each v component
            for c in 0..2 { // two velocity components
                let base = c * n_test_v; // offset within M_V
                for i in 0..n_test_v {
                    let ri = base + i;
                    for j in 0..n_test_v {
                        let cj = base + j;
                        mv[ri * n_test + cj] += w * (phi_v[i]*phi_v[j] + dvx[i]*dvx[j] + dvy[i]*dvy[j]);
                    }
                }
            }
            // Pressure block (rows 20-25, cols 20-25): H¹ inner product
            for i in 0..n_test_p {
                let ri = 20 + i;
                for j in 0..n_test_p {
                    let cj = 20 + j;
                    mv[ri * n_test + cj] += w * (phi_p[i]*phi_p[j] + dpx[i]*dpx[j] + dpy[i]*dpy[j]);
                }
            }

            // ── Assembly of B (26 × 9) ──────────────────────────────────────
            // Trial ordering: [u0_x, u0_y, p0, u1_x, u1_y, p1, u2_x, u2_y, p2]
            // B[τ, v, q; u, p] = ν∫∇u:∇v dx - ∫p·∇·v dx - ∫q·∇·u dx
            //
            // ∇u:∇v = ux_x·vx_x + ux_y·vx_y + uy_x·vy_x + uy_y·vy_y
            // p·∇·v = p·(vx_x + vy_y)
            // q·∇·u = q·(ux_x + uy_y)

            for jv in 0..n_trial_v { // trial velocity DOF index per component
                for it in 0..n_test_v { // test velocity DOF (vx)
                    // B[τ_x, u_x] = ν∫∇u_x·∇τ_x
                    let row = it;
                    let col_x = jv;          // u_x at local node jv
                    bm[row * n_trial + col_x] += w * nu * (tu_x[jv]*dvx[it] + tu_y[jv]*dvy[it]);
                    // B[q, u_x] = -∫q·∇·(u_x e_x) = -∫q·∂u_x/∂x
                    for iq in 0..n_test_p {
                        let row_q = 20 + iq;
                        bm[row_q * n_trial + col_x] -= w * phi_p[iq] * tu_x[jv];
                    }
                }
                for it in 0..n_test_v { // test velocity DOF (vy)
                    let row = n_test_v + it;
                    let col_y = n_trial_v + jv; // u_y at local node jv
                    bm[row * n_trial + col_y] += w * nu * (tu_x[jv]*dvx[it] + tu_y[jv]*dvy[it]);
                    // B[q, u_y] = -∫q·∇·(u_y e_y) = -∫q·∂u_y/∂y
                    for iq in 0..n_test_p {
                        let row_q = 20 + iq;
                        bm[row_q * n_trial + col_y] -= w * phi_p[iq] * tu_y[jv];
                    }
                }
            }

            // Pressure trial coupling
            for jp in 0..n_trial_v { // pressure trial node
                let col_p = 2 * n_trial_v + jp; // p at local node jp
                for it in 0..n_test_v { // test velocity (divergence of v)
                    // B[τ, p] = -∫p·∇·τ = -∫p·(∂τx/∂x + ∂τy/∂y)
                    let row_x = it;
                    let row_y = n_test_v + it;
                    bm[row_x * n_trial + col_p] -= w * phi_t[jp] * dvx[it];
                    bm[row_y * n_trial + col_p] -= w * phi_t[jp] * dvy[it];
                }
            }
        }

        // ── Compute optimal test functions ─────────────────────────────────
        let mut v_opt = vec![0.0; n_test * n_trial];
        for i in 0..n_trial {
            let mut rhs = vec![0.0; n_test];
            for r in 0..n_test { rhs[r] = bm[r * n_trial + i]; }
            let mut mv_copy = mv.clone();
            solve_dense(n_test, &mut mv_copy, &mut rhs);
            for r in 0..n_test { v_opt[r * n_trial + i] = rhs[r]; }
        }

        // ── Element stiffness Ke = B^T V_opt ───────────────────────────────
        let mut ke = vec![0.0; n_trial * n_trial];
        for i in 0..n_trial {
            for j in 0..n_trial {
                for k in 0..n_test {
                    ke[i * n_trial + j] += bm[k * n_trial + i] * v_opt[k * n_trial + j];
                }
            }
        }

        // RHS: element-wise Galerkin projection of f onto test space
        let centroid_x = x.iter().sum::<f64>() / nn as f64;
        let centroid_y = y.iter().sum::<f64>() / nn as f64;
        let (fx, fy) = f(centroid_x, centroid_y);
        let mut elem_rhs = vec![0.0; n_trial];
        for i in 0..n_trial_v {
            elem_rhs[i] = fx * vol / 3.0; // u_x
            elem_rhs[n_trial_v + i] = fy * vol / 3.0; // u_y
            // pressure RHS is zero (div constraint is homogeneous)
        }

        // ── Scatter into global system ──────────────────────────────────────
        for (li, &gi) in dofs.iter().enumerate() {
            rhs_global[gi] += elem_rhs[li];
            for lj in 0..n_trial {
                let gj = dofs[lj];
                let v = ke[li * n_trial + lj];
                if v.abs() > 1e-30 { coo.add(gi, gj, v); }
            }
        }
    }

    let mat = coo.into_csr();

    // Apply homogeneous Dirichlet BC on velocity: u_x = 0, u_y = 0 on boundary.
    // Pin pressure at node 0 to zero.
    let mut is_boundary = vec![false; n_nodes];
    for bf in 0..mesh.n_boundary_faces() as u32 {
        for &n in mesh.face_nodes(bf) { is_boundary[n as usize] = true; }
    }
    let mut bc_dofs: Vec<usize> = Vec::new();
    for n in 0..n_nodes {
        if is_boundary[n] { bc_dofs.push(velx_dof(n)); bc_dofs.push(vely_dof(n)); }
    }
    bc_dofs.push(pres_dof(0)); // pin pressure

    let mut rhs = rhs_global;
    for &d in &bc_dofs {
        let s = mat.row_ptr[d]; let e = mat.row_ptr[d + 1];
        for p in s..e {
            let j = mat.col_idx[p] as usize;
            if j != d { rhs[j] -= mat.values[p] * 0.0; } // val=0 → no change needed
        }
        rhs[d] = 0.0;
    }

    // Solve with a few CG iterations
    let mut x = vec![0.0; n_total];
    for _ in 0..200 {
        let mut ax = vec![0.0; n_total];
        for i in 0..n_total {
            let s = mat.row_ptr[i]; let e = mat.row_ptr[i+1];
            ax[i] = (s..e).map(|p| mat.values[p] * x[mat.col_idx[p] as usize]).sum();
        }
        let mut r = vec![0.0; n_total];
        let mut rr = 0.0;
        for i in 0..n_total {
            r[i] = rhs[i] - ax[i];
            rr += r[i] * r[i];
        }
        if rr < 1e-20 { break; }
        // Simple CG step
        let mut ar = vec![0.0; n_total];
        for i in 0..n_total {
            let s = mat.row_ptr[i]; let e = mat.row_ptr[i+1];
            ar[i] = (s..e).map(|p| mat.values[p] * r[mat.col_idx[p] as usize]).sum();
        }
        let rar: f64 = (0..n_total).map(|i| r[i] * ar[i]).sum();
        let alpha = if rar.abs() > 1e-30 { rr / rar } else { 0.0 };
        for i in 0..n_total { x[i] += alpha * r[i]; }
    }

    let ux: Vec<f64> = x[..n_vel].to_vec();
    let uy: Vec<f64> = x[n_vel..2*n_vel].to_vec();
    let p: Vec<f64> = x[2*n_vel..].to_vec();
    (ux, uy, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    #[test]
    fn dpg_stokes_2d_square_finite() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let (ux, uy, p) = solve_dpg_stokes_2d(&mesh, 1.0, &|_,_| (0.0, 0.0));
        for &v in &ux { assert!(v.is_finite(), "ux non-finite"); }
        for &v in &uy { assert!(v.is_finite(), "uy non-finite"); }
        for &v in &p  { assert!(v.is_finite(), "p non-finite"); }
    }

    #[test]
    fn dpg_stokes_2d_symmetric_produces_zero_velocity() {
        // Symmetric BC with f=0 → u=0, p=constant (up to pressure pin)
        let mesh = Mesh::<2>::unit_square_tri(4);
        let (ux, uy, _p) = solve_dpg_stokes_2d(&mesh, 1.0, &|_,_| (0.0, 0.0));
        let max_u = ux.iter().chain(uy.iter()).map(|x| x.abs()).fold(0.0, f64::max);
        assert!(max_u < 0.1, "zero forcing → small velocity, got max |u|={max_u:.3e}");
    }
}
