//! DPG linear elasticity solver — primal formulation.
//!
//! Solves `-∇·σ = f` with σ = λ(tr ε)I + 2μ ε on Tri3 meshes.
//!
//! # Method
//!
//! Per-element DPG with optimal test functions:
//! - Trial: P1 displacement (2 components = 6 DOFs)
//! - Test:  P3 displacement (2 components = 20 DOFs)
//! - Bilinear form: a(u,v) = ∫ λ(∇·u)(∇·v) + 2μ ε(u):ε(v) dx
//! - Optimal test functions via local H¹ Gram matrix solve

use fem_element::{ReferenceElement, lagrange::{TriP1, TriP3}, quadrature::tri_rule};
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

/// Solve 2-D linear elasticity via DPG on a Tri3 mesh.
///
/// # Arguments
/// * `mesh` — Tri3 mesh
/// * `lambda`, `mu` — Lamé parameters
/// * `f` — body force `f(x,y) → (fx, fy)`
///
/// # Returns
/// `(u_x, u_y)` nodal displacement vectors.
pub fn solve_dpg_elasticity_2d<M: MeshTopology>(
    mesh: &M,
    lambda: f64,
    mu: f64,
    f: &dyn Fn(f64, f64) -> (f64, f64),
) -> (Vec<f64>, Vec<f64>) {
    let n_nodes = mesh.n_nodes();
    let n_elems = mesh.n_elements();
    let n_total = 2 * n_nodes;
    let n_test = 20;  // P3 × 2 components
    let n_trial = 6;  // P1 × 2 components
    let n_v = 10;     // P3 DOFs per component

    let tri_p3 = TriP3;
    let qr = tri_rule(7);

    let mut coo = CooMatrix::<f64>::new(n_total, n_total);
    let mut rhs_global = vec![0.0; n_total];

    let dof_ux = |n: usize| n;
    let dof_uy = |n: usize| n_nodes + n;

    for e in 0..n_elems as u32 {
        let nodes = mesh.element_nodes(e);
        let nn = nodes.len();
        let dofs: Vec<usize> = nodes.iter().flat_map(|&n| {
            let nu = n as usize; vec![dof_ux(nu), dof_uy(nu)]
        }).collect();

        let x: Vec<f64> = (0..nn).map(|k| mesh.node_coords(nodes[k])[0]).collect();
        let y: Vec<f64> = (0..nn).map(|k| mesh.node_coords(nodes[k])[1]).collect();

        let j00 = x[1] - x[0]; let j01 = x[2] - x[0];
        let j10 = y[1] - y[0]; let j11 = y[2] - y[0];
        let det_j = j00 * j11 - j01 * j10;
        let abs_det = det_j.abs();
        let inv_det = 1.0 / det_j;
        let vol = 0.5 * abs_det;

        let mut phi = vec![0.0; n_v];
        let mut dphi = vec![0.0; n_v * 2];
        let mut mv = vec![0.0; n_test * n_test];
        let mut bm = vec![0.0; n_test * n_trial];

        for (xi, &wr) in qr.points.iter().zip(qr.weights.iter()) {
            let w = wr * abs_det;

            tri_p3.eval_basis(xi, &mut phi);
            tri_p3.eval_grad_basis(xi, &mut dphi);

            let mut dpx = vec![0.0; n_v]; let mut dpy = vec![0.0; n_v];
            for i in 0..n_v {
                dpx[i] = (j11*dphi[i*2] - j10*dphi[i*2+1]) * inv_det;
                dpy[i] = (-j01*dphi[i*2] + j00*dphi[i*2+1]) * inv_det;
            }

            // P1 trial
            let tri_p1 = TriP1;
            tri_p1.eval_basis(xi, &mut phi);
            let tgrad_x = [-1.0, 1.0, 0.0]; let tgrad_y = [-1.0, 0.0, 1.0];
            let mut tdpx = [0.0; 3]; let mut tdpy = [0.0; 3];
            for i in 0..3 {
                tdpx[i] = (j11*tgrad_x[i] - j10*tgrad_y[i]) * inv_det;
                tdpy[i] = (-j01*tgrad_x[i] + j00*tgrad_y[i]) * inv_det;
            }

            // M_V: H¹ inner product (2 components)
            for c in 0..2 {
                let base = c * n_v;
                for i in 0..n_v {
                    let ri = base + i;
                    for j in 0..n_v {
                        let cj = base + j;
                        mv[ri * n_test + cj] += w * (phi[i]*phi[j] + dpx[i]*dpx[j] + dpy[i]*dpy[j]);
                    }
                }
            }

            // B: elasticity bilinear form
            // ε(u):ε(v) = ε_xx·ε_xx + 2·ε_xy·ε_xy + ε_yy·ε_yy
            // where ε_xx(v) = ∂vx/∂x, ε_yy(v) = ∂vy/∂y, ε_xy(v) = ½(∂vx/∂y + ∂vy/∂x)
            //
            // Similarly ∇·u = ∂ux/∂x + ∂uy/∂y
            // TriP3 test: 0..9 = vx, 10..19 = vy
            // TriP1 trial: 0..2 = ux, 3..5 = uy

            for ti in 0..3 { // trial node index
                // Trial ux at node ti
                let col_ux = ti;
                // Trial uy at node ti
                let col_uy = 3 + ti;

                for si in 0..n_v { // test shape function index
                    // Test vx
                    let row_vx = si;
                    // Test vy
                    let row_vy = n_v + si;

                    // Strain components for trial:
                    // ε_xx(ux) = ∂ux/∂x = tdpx[ti]
                    // ε_yy(ux) = 0
                    // ε_xy(ux) = ½·∂ux/∂y = ½·tdpy[ti]
                    // ε_xx(uy) = 0
                    // ε_yy(uy) = ∂uy/∂y = tdpy[ti]
                    // ε_xy(uy) = ½·∂uy/∂x = ½·tdpx[ti]

                    // Strain components for test vx:
                    // ε_xx(vx) = dpx[si]
                    // ε_xy(vx) = ½·dpy[si]
                    // For test vy:
                    // ε_yy(vy) = dpy[si]
                    // ε_xy(vy) = ½·dpx[si]

                    // Divergence:
                    // ∇·ux = tdpx[ti]; ∇·uy = tdpy[ti]
                    // ∇·vx = dpx[si];  ∇·vy = dpy[si]

                    // λ(∇·ux)(∇·vx) + 2μ[ε_xx(ux)·ε_xx(vx) + 2·ε_xy(ux)·ε_xy(vx)]
                    let b_ux_vx = lambda * tdpx[ti] * dpx[si]
                        + 2.0 * mu * (tdpx[ti]*dpx[si] + 2.0*0.25*tdpy[ti]*dpy[si]);
                    bm[row_vx * n_trial + col_ux] += w * b_ux_vx;

                    // λ(∇·uy)(∇·vx) + 2μ[ε_xx(uy)·ε_xx(vx) + 2·ε_xy(uy)·ε_xy(vx)]
                    let b_uy_vx = lambda * tdpy[ti] * dpx[si]
                        + 2.0 * mu * (2.0*0.25*tdpx[ti]*dpy[si]);
                    bm[row_vx * n_trial + col_uy] += w * b_uy_vx;

                    // λ(∇·ux)(∇·vy) + 2μ[ε_yy(ux)·ε_yy(vy) + 2·ε_xy(ux)·ε_xy(vy)]
                    let b_ux_vy = lambda * tdpx[ti] * dpy[si]
                        + 2.0 * mu * (2.0*0.25*tdpy[ti]*dpx[si]);
                    bm[row_vy * n_trial + col_ux] += w * b_ux_vy;

                    // λ(∇·uy)(∇·vy) + 2μ[ε_yy(uy)·ε_yy(vy) + 2·ε_xy(uy)·ε_xy(vy)]
                    let b_uy_vy = lambda * tdpy[ti] * dpy[si]
                        + 2.0 * mu * (tdpy[ti]*dpy[si] + 2.0*0.25*tdpx[ti]*dpx[si]);
                    bm[row_vy * n_trial + col_uy] += w * b_uy_vy;
                }
            }

            // Restore phi for M_V (P3 values)
            tri_p3.eval_basis(xi, &mut phi);
        }

        // Solve optimal test functions
        let mut v_opt = vec![0.0; n_test * n_trial];
        for i in 0..n_trial {
            let mut rhs = vec![0.0; n_test];
            for r in 0..n_test { rhs[r] = bm[r * n_trial + i]; }
            let mut mv_copy = mv.clone();
            solve_dense(n_test, &mut mv_copy, &mut rhs);
            for r in 0..n_test { v_opt[r * n_trial + i] = rhs[r]; }
        }

        // Ke = B^T V_opt
        let mut ke = vec![0.0; n_trial * n_trial];
        for i in 0..n_trial {
            for j in 0..n_trial {
                ke[i * n_trial + j] = (0..n_test).map(|k| bm[k*n_trial+i] * v_opt[k*n_trial+j]).sum();
            }
        }

        // RHS
        let cx = x.iter().sum::<f64>() / 3.0;
        let cy = y.iter().sum::<f64>() / 3.0;
        let (fx, fy) = f(cx, cy);
        for i in 0..3 {
            rhs_global[dof_ux(nodes[i] as usize)] += fx * vol / 3.0;
            rhs_global[dof_uy(nodes[i] as usize)] += fy * vol / 3.0;
        }

        // Scatter
        for (li, &gi) in dofs.iter().enumerate() {
            for lj in 0..n_trial {
                let gj = dofs[lj];
                let v = ke[li * n_trial + lj];
                if v.abs() > 1e-30 { coo.add(gi, gj, v); }
            }
        }
    }

    let mat = coo.into_csr();
    let mut rhs = rhs_global;

    // Dirichlet BC: fix boundary ux=uy=0
    let mut bc_dofs: Vec<usize> = Vec::new();
    for bf in 0..mesh.n_boundary_faces() as u32 {
        for &n in mesh.face_nodes(bf) {
            bc_dofs.push(dof_ux(n as usize));
            bc_dofs.push(dof_uy(n as usize));
        }
    }
    bc_dofs.sort_unstable(); bc_dofs.dedup();

    for &d in &bc_dofs {
        let s = mat.row_ptr[d]; let e = mat.row_ptr[d + 1];
        for p in s..e { let j = mat.col_idx[p] as usize; if j != d { rhs[j] -= mat.values[p] * 0.0; } }
        rhs[d] = 0.0;
    }

    // CG solver
    let mut x = vec![0.0; n_total];
    for _ in 0..400 {
        let mut ax = vec![0.0; n_total];
        for i in 0..n_total {
            let s = mat.row_ptr[i]; let e = mat.row_ptr[i + 1];
            ax[i] = (s..e).map(|p| mat.values[p] * x[mat.col_idx[p] as usize]).sum();
        }
        let mut r = vec![0.0; n_total];
        let mut rr = 0.0;
        for i in 0..n_total { r[i] = rhs[i] - ax[i]; rr += r[i] * r[i]; }
        if rr < 1e-24 { break; }
        let mut ar = vec![0.0; n_total];
        for i in 0..n_total {
            let s = mat.row_ptr[i]; let e = mat.row_ptr[i + 1];
            ar[i] = (s..e).map(|p| mat.values[p] * r[mat.col_idx[p] as usize]).sum();
        }
        let rar: f64 = (0..n_total).map(|i| r[i] * ar[i]).sum();
        let alpha = if rar.abs() > 1e-30 { rr / rar } else { 0.0 };
        for i in 0..n_total { x[i] += alpha * r[i]; }
        for &d in &bc_dofs { x[d] = 0.0; }
    }

    let ux: Vec<f64> = x[..n_nodes].to_vec();
    let uy: Vec<f64> = x[n_nodes..].to_vec();
    (ux, uy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn dpg_elasticity_2d_finite() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let e = 100.0; let nu = 0.3;
        let lam = e * nu / ((1.0+nu)*(1.0-2.0*nu));
        let mu = e / (2.0*(1.0+nu));
        let (ux, uy) = solve_dpg_elasticity_2d(&mesh, lam, mu, &|_,_| (0.0, 0.0));
        for &v in &ux { assert!(v.is_finite()); }
        for &v in &uy { assert!(v.is_finite()); }
    }

    #[test]
    fn dpg_elasticity_gravity_finite() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let e = 1000.0; let nu = 0.3;
        let lam = e * nu / ((1.0+nu)*(1.0-2.0*nu));
        let mu = e / (2.0*(1.0+nu));
        let (ux, uy) = solve_dpg_elasticity_2d(&mesh, lam, mu, &|_,_| (0.0, -1.0));
        for &v in &ux { assert!(v.is_finite()); }
        for &v in &uy { assert!(v.is_finite()); }
        // Under gravity load, vertical displacement should be negative somewhere
        let max_uy = uy.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_uy <= 0.0, "gravity → downward displacement, got max_uy={max_uy:.3e}");
    }
}
