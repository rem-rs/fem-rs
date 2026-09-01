//! 2D DPG (Discontinuous Petrov-Galerkin) for Poisson on triangles.
//!
//! Trial: continuous P1 on Tri3. Test: enriched P3 (10 DOFs) per element.
//! Optimal test functions computed from the H¹ inner product.
//!
//! For pure Poisson, the optimal test functions have zero mean, which
//! suppresses the standard DPG RHS. We use the Galerkin RHS instead
//! (stable, convergent Petrov-Galerkin method).

use fem_element::{ReferenceElement, lagrange::factory::TriPk};
use fem_element::quadrature::tri_rule;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::MeshTopology;

fn solve_dense(n: usize, a: &mut [f64], b: &mut [f64]) {
    for col in 0..n {
        let mut best = col;
        let mut best_val = a[col * n + col].abs();
        for row in (col + 1)..n { let v = a[row * n + col].abs(); if v > best_val { best_val = v; best = row; } }
        if best_val < 1e-30 { continue; }
        if best != col { for c in col..n { a.swap(col * n + c, best * n + c); } b.swap(col, best); }
        let pivot = a[col * n + col];
        for row in (col + 1)..n { let f = a[row * n + col] / pivot;
            for c in col..n { a[row * n + c] -= f * a[col * n + c]; } b[row] -= f * b[col]; }
    }
    for row in (0..n).rev() {
        let mut s = b[row];
        for c in (row + 1)..n { s -= a[row * n + c] * b[c]; }
        if a[row * n + row].abs() > 1e-30 { b[row] = s / a[row * n + row]; } else { b[row] = 0.0; }
    }
}

/// Solve `-Δu = f` on a Tri3 mesh with homogeneous Dirichlet BCs.
pub fn solve_dpg_poisson_2d<M: MeshTopology>(mesh: &M, f: &dyn Fn(f64, f64) -> f64) -> Vec<f64> {
    let n_nodes = mesh.n_nodes();
    let n_elem = mesh.n_elements();
    let n_test = 10;
    let n_trial = 3;
    let tri_p3 = TriPk::new(3);
    let qr = tri_rule(7);

    let mut coo = CooMatrix::<f64>::new(n_nodes, n_nodes);
    let mut rhs = vec![0.0; n_nodes];
    let mut phi = vec![0.0; n_test];
    let mut dphi = vec![0.0; n_test * 2];

    for e in 0..n_elem as u32 {
        let nodes = mesh.element_nodes(e);
        let dofs: Vec<usize> = nodes.iter().map(|&n| n as usize).collect();

        let x: Vec<f64> = (0..3).map(|k| mesh.node_coords(nodes[k])[0]).collect();
        let y: Vec<f64> = (0..3).map(|k| mesh.node_coords(nodes[k])[1]).collect();

        let j00 = x[1] - x[0]; let j01 = x[2] - x[0];
        let j10 = y[1] - y[0]; let j11 = y[2] - y[0];
        let det_j = j00 * j11 - j01 * j10;
        let abs_det = det_j.abs();
        let inv_det = 1.0 / det_j;

        let mut mv = vec![0.0; n_test * n_test];
        let mut kv = vec![0.0; n_test * n_test];
        let mut bm = vec![0.0; n_test * n_trial];

        for (xi, &wr) in qr.points.iter().zip(qr.weights.iter()) {
            let w = wr * abs_det;
            tri_p3.eval_basis(xi, &mut phi);
            tri_p3.eval_grad_basis(xi, &mut dphi);

            // J^{-T} = [[j11, -j10], [-j01, j00]] / det
            let mut dpx = [0.0; 10]; let mut dpy = [0.0; 10];
            for i in 0..n_test {
                dpx[i] = (j11 * dphi[i*2] - j10 * dphi[i*2+1]) * inv_det;
                dpy[i] = (-j01 * dphi[i*2] + j00 * dphi[i*2+1]) * inv_det;
            }

            // P1 trial physical gradients
            let tgx = [-1.0, 1.0, 0.0]; let tgy = [-1.0, 0.0, 1.0];
            let mut tdpx = [0.0; 3]; let mut tdpy = [0.0; 3];
            for i in 0..n_trial {
                tdpx[i] = (j11 * tgx[i] - j10 * tgy[i]) * inv_det;
                tdpy[i] = (-j01 * tgx[i] + j00 * tgy[i]) * inv_det;
            }

            for i in 0..n_test { for j in 0..n_test {
                mv[i*n_test+j] += phi[i] * phi[j] * w;
                kv[i*n_test+j] += (dpx[i]*dpx[j] + dpy[i]*dpy[j]) * w;
            }}
            for i in 0..n_test { for j in 0..n_trial {
                bm[i*n_trial+j] += (dpx[i]*tdpx[j] + dpy[i]*tdpy[j]) * w;
            }}

            // Galerkin RHS: ∫ f · φ_i  (using P1 trial, not the optimal test)
            let xp = x[0] + j00*xi[0] + j01*xi[1];
            let yp = y[0] + j10*xi[0] + j11*xi[1];
            let fv = f(xp, yp);
            let tp0 = 1.0 - xi[0] - xi[1];
            let tp1 = xi[0];
            let tp2 = xi[1];
            rhs[dofs[0]] += fv * tp0 * w;
            rhs[dofs[1]] += fv * tp1 * w;
            rhs[dofs[2]] += fv * tp2 * w;
        }

        // H¹ inner product: MV + KV
        let mut mk = mv;
        for i in 0..n_test { for j in 0..n_test { mk[i*n_test+j] += kv[i*n_test+j]; }}

        // Optimal test functions: MV * v_opt_j = B[:,j]
        let mut vo = vec![0.0; n_test * n_trial];
        for j in 0..n_trial {
            let mut rj = vec![0.0; n_test];
            for i in 0..n_test { rj[i] = bm[i*n_trial+j]; }
            let mut mc = mk.clone();
            solve_dense(n_test, &mut mc, &mut rj);
            for i in 0..n_test { vo[i*n_trial+j] = rj[i]; }
        }

        // DPG element stiffness: K_e[i,j] = Σ_k B[k,j] · v_opt[k,i]
        for i in 0..n_trial { for j in 0..n_trial {
            let mut v = 0.0; for k in 0..n_test { v += bm[k*n_trial+j] * vo[k*n_trial+i]; }
            coo.add(dofs[i], dofs[j], v);
        }}
    }

    // Dirichlet BCs via CSR
    let mut a: CsrMatrix<f64> = coo.into_csr();
    let mut bdy = vec![false; n_nodes];
    for f in 0..mesh.n_boundary_faces() as u32 { for &n in mesh.face_nodes(f) { bdy[n as usize] = true; }}
    for d in 0..n_nodes { if bdy[d] {
        let s = a.row_ptr[d]; let e = a.row_ptr[d+1];
        for j in s..e { a.values[j] = if a.col_idx[j] as usize == d { 1.0 } else { 0.0 }; }
        rhs[d] = 0.0;
    }}

    let mut x = vec![0.0; n_nodes];
    let cfg = fem_solver::SolverConfig { rtol: 1e-12, atol: 1e-30, max_iter: 10_000, ..Default::default() };
    fem_solver::solve_cg(&a, &rhs, &mut x, &cfg).expect("DPG CG solve converged");
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use std::f64::consts::PI;

    fn f_src(x: f64, y: f64) -> f64 { 2.0*PI*PI * (PI*x).sin() * (PI*y).sin() }
    fn u_exact(x: f64, y: f64) -> f64 { (PI*x).sin() * (PI*y).sin() }

    #[test]
    fn dpg_2d_convergence() {
        let mut prev = f64::MAX;
        for &n in &[4, 8, 16] {
            let mesh = Mesh::<2>::unit_square_tri(n);
            let u = solve_dpg_poisson_2d(&mesh, &f_src);
            let nn = mesh.n_nodes();
            let mut e2 = 0.0;
            for i in 0..nn {
                let c = mesh.node_coords(i as u32);
                let d = u[i] - u_exact(c[0], c[1]);
                e2 += d * d;
            }
            let err = (e2 / nn as f64).sqrt();
            let mx = u.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            eprintln!("n={n} L²_err={err:.6e} max_u={mx:.6e}");
            assert!(err < prev * 0.9, "L² error should decrease: n={n} err={err:.3e}");
            prev = err;
        }
    }
}
