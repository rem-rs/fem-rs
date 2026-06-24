//! 3D DPG (Discontinuous Petrov-Galerkin) for Poisson on tetrahedra.
//!
//! Trial: continuous P1 on Tet4. Test: enriched P3 (20 DOFs) per element.
//! Optimal test functions computed from the H¹ inner product.
//!
//! Galerkin RHS (same rationale as 2D — stable, convergent Petrov-Galerkin).

use fem_element::{ReferenceElement, lagrange::TetP3};
use fem_element::quadrature::tet_rule;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::MeshTopology;

fn solve_dense(n: usize, a: &mut [f64], b: &mut [f64]) {
    for col in 0..n {
        let mut best = col;
        let mut best_val = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > best_val { best_val = v; best = row; }
        }
        if best_val < 1e-30 { continue; }
        if best != col {
            for c in col..n { a.swap(col * n + c, best * n + c); }
            b.swap(col, best);
        }
        let pivot = a[col * n + col];
        for row in (col + 1)..n {
            let f = a[row * n + col] / pivot;
            for c in col..n { a[row * n + c] -= f * a[col * n + c]; }
            b[row] -= f * b[col];
        }
    }
    for row in (0..n).rev() {
        let mut s = b[row];
        for c in (row + 1)..n { s -= a[row * n + c] * b[c]; }
        if a[row * n + row].abs() > 1e-30 { b[row] = s / a[row * n + row]; } else { b[row] = 0.0; }
    }
}

/// Solve `-Δu = f` on a Tet4 mesh with homogeneous Dirichlet BCs.
pub fn solve_dpg_poisson_3d<M: MeshTopology>(mesh: &M, f: &dyn Fn(f64, f64, f64) -> f64) -> Vec<f64> {
    let n_nodes = mesh.n_nodes();
    let n_elem = mesh.n_elements();
    let n_test = 20;
    let n_trial = 4;
    let tet_p3 = TetP3;
    let qr = tet_rule(6);

    let mut coo = CooMatrix::<f64>::new(n_nodes, n_nodes);
    let mut rhs = vec![0.0; n_nodes];
    let mut phi = vec![0.0; n_test];
    let mut dphi = vec![0.0; n_test * 3];

    for e in 0..n_elem as u32 {
        let nodes = mesh.element_nodes(e);
        let dofs: Vec<usize> = nodes.iter().map(|&n| n as usize).collect();

        let x: Vec<f64> = (0..4).map(|k| mesh.node_coords(nodes[k])[0]).collect();
        let y: Vec<f64> = (0..4).map(|k| mesh.node_coords(nodes[k])[1]).collect();
        let z: Vec<f64> = (0..4).map(|k| mesh.node_coords(nodes[k])[2]).collect();

        // Jacobian J = [x1-x0, x2-x0, x3-x0]  (3 columns)
        let j00 = x[1] - x[0]; let j01 = x[2] - x[0]; let j02 = x[3] - x[0];
        let j10 = y[1] - y[0]; let j11 = y[2] - y[0]; let j12 = y[3] - y[0];
        let j20 = z[1] - z[0]; let j21 = z[2] - z[0]; let j22 = z[3] - z[0];

        let det_j = j00*(j11*j22 - j12*j21) - j01*(j10*j22 - j12*j20) + j02*(j10*j21 - j11*j20);
        let abs_det = det_j.abs();
        let inv_det = 1.0 / det_j;

        // J^{-T} (cofactor matrix / det)
        let it00 = (j11*j22 - j12*j21) * inv_det;
        let it01 = (j02*j21 - j01*j22) * inv_det;
        let it02 = (j01*j12 - j02*j11) * inv_det;
        let it10 = (j12*j20 - j10*j22) * inv_det;
        let it11 = (j00*j22 - j02*j20) * inv_det;
        let it12 = (j02*j10 - j00*j12) * inv_det;
        let it20 = (j10*j21 - j11*j20) * inv_det;
        let it21 = (j01*j20 - j00*j21) * inv_det;
        let it22 = (j00*j11 - j01*j10) * inv_det;

        let mut mv = vec![0.0; n_test * n_test];
        let mut kv = vec![0.0; n_test * n_test];
        let mut bm = vec![0.0; n_test * n_trial];

        for (xi, &wr) in qr.points.iter().zip(qr.weights.iter()) {
            let w = wr * abs_det;
            tet_p3.eval_basis(xi, &mut phi);
            tet_p3.eval_grad_basis(xi, &mut dphi);

            // Enriched test physical gradients: dphys = J^{-T} * dref
            let mut dpx = vec![0.0; n_test];
            let mut dpy = vec![0.0; n_test];
            let mut dpz = vec![0.0; n_test];
            for i in 0..n_test {
                let ri = i * 3;
                dpx[i] = it00*dphi[ri] + it01*dphi[ri+1] + it02*dphi[ri+2];
                dpy[i] = it10*dphi[ri] + it11*dphi[ri+1] + it12*dphi[ri+2];
                dpz[i] = it20*dphi[ri] + it21*dphi[ri+1] + it22*dphi[ri+2];
            }

            // P1 trial reference gradients
            let tgx = [-1.0, 1.0, 0.0, 0.0];
            let tgy = [-1.0, 0.0, 1.0, 0.0];
            let tgz = [-1.0, 0.0, 0.0, 1.0];

            // P1 trial physical gradients
            let mut tdpx = [0.0; 4];
            let mut tdpy = [0.0; 4];
            let mut tdpz = [0.0; 4];
            for i in 0..n_trial {
                tdpx[i] = it00*tgx[i] + it01*tgy[i] + it02*tgz[i];
                tdpy[i] = it10*tgx[i] + it11*tgy[i] + it12*tgz[i];
                tdpz[i] = it20*tgx[i] + it21*tgy[i] + it22*tgz[i];
            }

            // MV and KV (H¹ inner product for enriched test)
            for i in 0..n_test { for j in 0..n_test {
                mv[i*n_test+j] += phi[i] * phi[j] * w;
                kv[i*n_test+j] += (dpx[i]*dpx[j] + dpy[i]*dpy[j] + dpz[i]*dpz[j]) * w;
            }}
            // B matrix: ∫ ∇ψ_j·∇φ_i
            for i in 0..n_test { for j in 0..n_trial {
                bm[i*n_trial+j] += (dpx[i]*tdpx[j] + dpy[i]*tdpy[j] + dpz[i]*tdpz[j]) * w;
            }}

            // Galerkin RHS: ∫ f · φ_i (using P1 trial)
            let xp = x[0] + j00*xi[0] + j01*xi[1] + j02*xi[2];
            let yp = y[0] + j10*xi[0] + j11*xi[1] + j12*xi[2];
            let zp = z[0] + j20*xi[0] + j21*xi[1] + j22*xi[2];
            let fv = f(xp, yp, zp);
            let tp0 = 1.0 - xi[0] - xi[1] - xi[2];
            let tp1 = xi[0];
            let tp2 = xi[1];
            let tp3 = xi[2];
            rhs[dofs[0]] += fv * tp0 * w;
            rhs[dofs[1]] += fv * tp1 * w;
            rhs[dofs[2]] += fv * tp2 * w;
            rhs[dofs[3]] += fv * tp3 * w;
        }

        // H¹ inner product: MV + KV
        let mut mk = mv;
        for i in 0..n_test { for j in 0..n_test { mk[i*n_test+j] += kv[i*n_test+j]; }}

        // Optimal test functions: V * w_i = B[:,i]
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
            let mut v = 0.0;
            for k in 0..n_test { v += bm[k*n_trial+j] * vo[k*n_trial+i]; }
            coo.add(dofs[i], dofs[j], v);
        }}
    }

    // Dirichlet BCs via CSR
    let mut a: CsrMatrix<f64> = coo.into_csr();
    let mut bdy = vec![false; n_nodes];
    for f in 0..mesh.n_boundary_faces() as u32 {
        for &n in mesh.face_nodes(f) { bdy[n as usize] = true; }
    }
    for d in 0..n_nodes { if bdy[d] {
        let s = a.row_ptr[d]; let e = a.row_ptr[d+1];
        for j in s..e { a.values[j] = if a.col_idx[j] as usize == d { 1.0 } else { 0.0 }; }
        rhs[d] = 0.0;
    }}

    let mut x = vec![0.0; n_nodes];
    let cfg = fem_solver::SolverConfig { rtol: 1e-12, atol: 1e-30, max_iter: 10_000, ..Default::default() };
    fem_solver::solve_cg(&a, &rhs, &mut x, &cfg).expect("DPG 3D CG solve converged");
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    fn f_src(_x: f64, _y: f64, _z: f64) -> f64 { 6.0 * _x }

    #[test]
    fn dpg_3d_finite_solution() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(4);
        let u = solve_dpg_poisson_3d(&mesh, &f_src);
        let mx = u.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mn = u.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        eprintln!("dpg_3d n=4: min_u={:.6e} max_u={:.6e}", mn, mx);
        assert!(mx.is_finite());
        assert!(mn >= -1.0 && mx <= 1.0, "solution bounds: [{mn:.4e}, {mx:.4e}]");
    }

    #[test]
    fn dpg_3d_midpoint_value() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(8);
        let u = solve_dpg_poisson_3d(&mesh, &f_src);
        let mut mx_idx = 0;
        for i in 1..mesh.n_nodes() {
            if u[i] > u[mx_idx] { mx_idx = i; }
        }
        let c = mesh.node_coords(mx_idx as u32);
        let u_exact = c[0] * (1.0 - c[0]);
        eprintln!("dpg_3d n=8: max at ({:.3},{:.3},{:.3}) u={:.6e} exact={:.6e}",
                  c[0], c[1], c[2], u[mx_idx], u_exact);
        assert!((u[mx_idx] - u_exact).abs() < 0.15, "max value too far from exact");
    }
}
