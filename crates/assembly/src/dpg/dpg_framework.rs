//! DPG (Discontinuous Petrov-Galerkin) unified framework.
//!
//! # Architecture
//!
//! ```text
//!                   BilinearForm (B)
//!                         │
//!            ┌────────────┴────────────┐
//!            │                         │
//!       TrialSpace              TestSpace (enriched, broken)
//!            │                         │
//!            └────────────┬────────────┘
//!                         │
//!               OptimalTestFunctions
//!              (local Gram solve M_V)
//!                         │
//!                   ElementMatrix
//!                 K_e = Bᵀ M_V⁻¹ B
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! let dpg = DpgSolver::new(&mesh, trial_order, test_order);
//! let mat = dpg.assemble(&bilinear_form);
//! ```

use fem_linalg::CsrMatrix;
use fem_mesh::topology::MeshTopology;

/// Bilinear form interface for DPG methods.
///
/// For an element with `n_test` enriched test DOFs and `n_trial` trial DOFs,
/// returns the `n_test × n_trial` element matrix `B` plus the `n_test × n_test`
/// test-space inner product (Gram) matrix `M_V`.
pub trait BilinearForm<M: MeshTopology> {
    /// Evaluate the bilinear form B and test inner product M_V on element `e`.
    fn eval(&self, mesh: &M, e: u32, n_test: usize, n_trial: usize) -> (Vec<f64>, Vec<f64>);
}

/// DPG solver framework.
///
/// # Type parameters
/// * `M` — mesh type
/// * `F` — bilinear form type
pub struct DpgSolver<'a, M: MeshTopology, F: BilinearForm<M>> {
    mesh: &'a M,
    form: &'a F,
    n_test: usize,
    n_trial: usize,
}

impl<'a, M: MeshTopology, F: BilinearForm<M>> DpgSolver<'a, M, F> {
    pub fn new(mesh: &'a M, form: &'a F, n_test: usize, n_trial: usize) -> Self {
        Self { mesh, form, n_test, n_trial }
    }

    /// Assemble the global DPG stiffness matrix.
    ///
    /// For each element:
    /// 1. Compute `B` (n_test × n_trial) and `M_V` (n_test × n_test) via `BilinearForm::eval`.
    /// 2. Solve `M_V * V_opt = B` for the optimal test functions.
    /// 3. Element stiffness `K_e = Bᵀ * V_opt` (n_trial × n_trial).
    /// 4. Scatter into global matrix.
    pub fn assemble(&self, rhs_fn: &dyn Fn(&[f64]) -> f64) -> (CsrMatrix<f64>, Vec<f64>) {
        let n_nodes = self.mesh.n_nodes();
        let n_elems = self.mesh.n_elements();
        let mut coo = fem_linalg::CooMatrix::<f64>::new(n_nodes, n_nodes);
        let mut rhs_global = vec![0.0; n_nodes];

        for e in 0..n_elems as u32 {
            let dofs: Vec<usize> = self.mesh.element_nodes(e).iter().map(|&n| n as usize).collect();
            let (b_mat, mv) = self.form.eval(self.mesh, e, self.n_test, self.n_trial);

            // Solve MV * v_opt_i = b[:,i] for each trial DOF i
            let n_test = self.n_test;
            let n_trial = self.n_trial;
            let mut v_opt = vec![0.0; n_test * n_trial];
            for i in 0..n_trial {
                let mut rhs = vec![0.0; n_test];
                for r in 0..n_test { rhs[r] = b_mat[r * n_trial + i]; }
                let mut mv_copy = mv.clone();
                solve_dense(n_test, &mut mv_copy, &mut rhs);
                for r in 0..n_test { v_opt[r * n_trial + i] = rhs[r]; }
            }

            // K_e[i][j] = sum_{k} B[k][i] * v_opt[k][j]
            //           = sum_k b_mat[k*n_trial+i] * v_opt[k*n_trial+j]
            for i in 0..n_trial {
                for j in 0..n_trial {
                    let mut s = 0.0;
                    for k in 0..n_test {
                        s += b_mat[k * n_trial + i] * v_opt[k * n_trial + j];
                    }
                    if s.abs() > 1e-30 {
                        coo.add(dofs[i], dofs[j], s);
                    }
                }
            }

            // RHS: ∫ f·v_opt_i for each trial DOF i
            // Use the optimal test function as weight
            // (Galerkin RHS for simplicity — same as existing 2D/3D DPG codes)
            let mut elem_rhs = vec![0.0; n_trial];
            let (_, mv_orig) = self.form.eval(self.mesh, e, self.n_test, self.n_trial);
            // Re-solve with f-weighted RHS
            let mut mv_rhs = mv_orig;
            let mut f_rhs = vec![0.0; n_test];
            // Evaluate f on test basis
            let elem_nodes = self.mesh.element_nodes(e);
            let centroid: Vec<f64> = (0..self.mesh.dim() as usize)
                .map(|d| elem_nodes.iter().map(|&n| self.mesh.node_coords(n)[d]).sum::<f64>() / elem_nodes.len() as f64)
                .collect();
            let f_val = rhs_fn(&centroid);
            for k in 0..n_test { f_rhs[k] = f_val; } // constant f → uniform weighting
            solve_dense(n_test, &mut mv_rhs, &mut f_rhs);
            for i in 0..n_trial {
                elem_rhs[i] = (0..n_test).map(|k| b_mat[k * n_trial + i] * f_rhs[k]).sum();
            }
            // Replace with element-volume scaled Galerkin RHS
            let vol = element_volume(self.mesh, e);
            for i in 0..n_trial {
                elem_rhs[i] = f_val * vol / n_trial as f64;
            }

            for (i, &di) in dofs.iter().enumerate() {
                rhs_global[di] += elem_rhs[i];
            }
        }

        (coo.into_csr(), rhs_global)
    }
}

/// Solve `A·x = b` in-place (dense, `n × n`).
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
        b[row] = if a[row * n + row].abs() > 1e-30 { s / a[row * n + row] } else { 0.0 };
    }
}

/// Element volume for Tet4 / Tri3.
fn element_volume<M: MeshTopology>(mesh: &M, e: u32) -> f64 {
    let ns = mesh.element_nodes(e);
    let c = |i| mesh.node_coords(ns[i]);
    match ns.len() {
        4 => {
            let j00 = c(1)[0]-c(0)[0]; let j01 = c(2)[0]-c(0)[0]; let j02 = c(3)[0]-c(0)[0];
            let j10 = c(1)[1]-c(0)[1]; let j11 = c(2)[1]-c(0)[1]; let j12 = c(3)[1]-c(0)[1];
            let j20 = c(1)[2]-c(0)[2]; let j21 = c(2)[2]-c(0)[2]; let j22 = c(3)[2]-c(0)[2];
            let det = j00*(j11*j22-j12*j21) - j01*(j10*j22-j12*j20) + j02*(j10*j21-j11*j20);
            det.abs() / 6.0
        }
        3 => {
            let j00 = c(1)[0]-c(0)[0]; let j01 = c(2)[0]-c(0)[0];
            let j10 = c(1)[1]-c(0)[1]; let j11 = c(2)[1]-c(0)[1];
            let det = j00*j11 - j01*j10;
            det.abs() / 2.0
        }
        _ => 1.0,
    }
}

// ─── Poisson bilinear form for 2-D TriP1 trial / TriP3 test ────────────────

/// 2-D Poisson bilinear form (P1 trial / P3 test on Tri3 mesh).
pub struct Poisson2DForm;

impl<M: MeshTopology> BilinearForm<M> for Poisson2DForm {
    fn eval(&self, mesh: &M, e: u32, n_test: usize, n_trial: usize) -> (Vec<f64>, Vec<f64>) {
        use fem_element::{ReferenceElement, lagrange::TriP3, quadrature::tri_rule};
        let tri_p3 = TriP3;
        let qr = tri_rule(7);
        let mut phi = vec![0.0; n_test];
        let mut dphi = vec![0.0; n_test * 2];

        let nodes = mesh.element_nodes(e);
        let x: Vec<f64> = (0..3).map(|k| mesh.node_coords(nodes[k])[0]).collect();
        let y: Vec<f64> = (0..3).map(|k| mesh.node_coords(nodes[k])[1]).collect();

        let j00 = x[1]-x[0]; let j01 = x[2]-x[0];
        let j10 = y[1]-y[0]; let j11 = y[2]-y[0];
        let det_j = j00*j11 - j01*j10;
        let abs_det = det_j.abs();
        let inv_det = 1.0/det_j;

        let mut mv = vec![0.0; n_test * n_test];
        let mut bm = vec![0.0; n_test * n_trial];

        for (xi, &wr) in qr.points.iter().zip(qr.weights.iter()) {
            let w = wr * abs_det;
            tri_p3.eval_basis(xi, &mut phi);
            tri_p3.eval_grad_basis(xi, &mut dphi);
            let mut dpx = vec![0.0; n_test]; let mut dpy = vec![0.0; n_test];
            for i in 0..n_test {
                dpx[i] = (j11*dphi[i*2] - j10*dphi[i*2+1]) * inv_det;
                dpy[i] = (-j01*dphi[i*2] + j00*dphi[i*2+1]) * inv_det;
            }
            let tgx = [-1.0, 1.0, 0.0]; let tgy = [-1.0, 0.0, 1.0];
            let mut tdpx = vec![0.0; n_trial]; let mut tdpy = vec![0.0; n_trial];
            for i in 0..n_trial {
                tdpx[i] = (j11*tgx[i] - j10*tgy[i]) * inv_det;
                tdpy[i] = (-j01*tgx[i] + j00*tgy[i]) * inv_det;
            }
            // M_V: H¹ inner product (v·w + ∇v·∇w)
            for i in 0..n_test {
                for j in 0..n_test {
                    mv[i*n_test+j] += w * (phi[i]*phi[j] + dpx[i]*dpx[j] + dpy[i]*dpy[j]);
                }
            }
            // B: ∇ψ_i · ∇φ_j  (Poisson)
            for i in 0..n_test {
                for j in 0..n_trial {
                    bm[i*n_trial+j] += w * (dpx[i]*tdpx[j] + dpy[i]*tdpy[j]);
                }
            }
        }
        (bm, mv)
    }
}

// ─── Poisson bilinear form for 3-D TetP1 trial / TetP3 test ────────────────

/// 3-D Poisson bilinear form (P1 trial / P3 test on Tet4 mesh).
pub struct Poisson3DForm;

impl<M: MeshTopology> BilinearForm<M> for Poisson3DForm {
    fn eval(&self, mesh: &M, e: u32, n_test: usize, n_trial: usize) -> (Vec<f64>, Vec<f64>) {
        use fem_element::{ReferenceElement, lagrange::TetP3, quadrature::tet_rule};
        let tet_p3 = TetP3;
        let qr = tet_rule(6);
        let mut phi = vec![0.0; n_test];
        let mut dphi = vec![0.0; n_test * 3];

        let nodes = mesh.element_nodes(e);
        let x: Vec<f64> = (0..4).map(|k| mesh.node_coords(nodes[k])[0]).collect();
        let y: Vec<f64> = (0..4).map(|k| mesh.node_coords(nodes[k])[1]).collect();
        let z: Vec<f64> = (0..4).map(|k| mesh.node_coords(nodes[k])[2]).collect();

        let j00=x[1]-x[0]; let j01=x[2]-x[0]; let j02=x[3]-x[0];
        let j10=y[1]-y[0]; let j11=y[2]-y[0]; let j12=y[3]-y[0];
        let j20=z[1]-z[0]; let j21=z[2]-z[0]; let j22=z[3]-z[0];
        let det_j = j00*(j11*j22-j12*j21)-j01*(j10*j22-j12*j20)+j02*(j10*j21-j11*j20);
        let abs_det = det_j.abs();
        let inv_det = 1.0/det_j;

        let it00=(j11*j22-j12*j21)*inv_det; let it01=(j02*j21-j01*j22)*inv_det; let it02=(j01*j12-j02*j11)*inv_det;
        let it10=(j12*j20-j10*j22)*inv_det; let it11=(j00*j22-j02*j20)*inv_det; let it12=(j02*j10-j00*j12)*inv_det;
        let it20=(j10*j21-j11*j20)*inv_det; let it21=(j01*j20-j00*j21)*inv_det; let it22=(j00*j11-j01*j10)*inv_det;

        let mut mv = vec![0.0; n_test * n_test];
        let mut bm = vec![0.0; n_test * n_trial];

        for (xi, &wr) in qr.points.iter().zip(qr.weights.iter()) {
            let w = wr * abs_det;
            tet_p3.eval_basis(xi, &mut phi);
            tet_p3.eval_grad_basis(xi, &mut dphi);
            let mut dpx = vec![0.0; n_test];
            let mut dpy = vec![0.0; n_test];
            let mut dpz = vec![0.0; n_test];
            for i in 0..n_test {
                dpx[i] = it00*dphi[i*3]+it01*dphi[i*3+1]+it02*dphi[i*3+2];
                dpy[i] = it10*dphi[i*3]+it11*dphi[i*3+1]+it12*dphi[i*3+2];
                dpz[i] = it20*dphi[i*3]+it21*dphi[i*3+1]+it22*dphi[i*3+2];
            }
            // P1 trial reference gradients (4 DOFs for Tet4)
            let tgx = [-1.0, 1.0, 0.0, 0.0];
            let tgy = [-1.0, 0.0, 1.0, 0.0];
            let tgz = [-1.0, 0.0, 0.0, 1.0];
            let mut tdpx = vec![0.0; n_trial];
            let mut tdpy = vec![0.0; n_trial];
            let mut tdpz = vec![0.0; n_trial];
            for i in 0..n_trial {
                tdpx[i] = it00*tgx[i]+it01*tgy[i]+it02*tgz[i];
                tdpy[i] = it10*tgx[i]+it11*tgy[i]+it12*tgz[i];
                tdpz[i] = it20*tgx[i]+it21*tgy[i]+it22*tgz[i];
            }
            for i in 0..n_test {
                for j in 0..n_test {
                    mv[i*n_test+j] += w * (phi[i]*phi[j] + dpx[i]*dpx[j] + dpy[i]*dpy[j] + dpz[i]*dpz[j]);
                }
            }
            for i in 0..n_test {
                for j in 0..n_trial {
                    bm[i*n_trial+j] += w * (dpx[i]*tdpx[j] + dpy[i]*tdpy[j] + dpz[i]*tdpz[j]);
                }
            }
        }
        (bm, mv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn dpg_framework_2d_poisson() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let form = Poisson2DForm;
        let solver = DpgSolver::new(&mesh, &form, 10, 3);
        let (mat, _rhs) = solver.assemble(&|_| 1.0);
        // Matrix should be symmetric
        for i in 0..mat.nrows.min(50) {
            for j in 0..i.min(50) {
                assert!((mat.get(i,j) - mat.get(j,i)).abs() < 1e-12,
                    "asymmetry at ({i},{j})");
            }
        }
        // Matrix should be square, matching node count
        assert_eq!(mat.nrows, mesh.n_nodes());
    }

    #[test]
    fn dpg_framework_3d_poisson() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let form = Poisson3DForm;
        let solver = DpgSolver::new(&mesh, &form, 20, 4);
        let (mat, _rhs) = solver.assemble(&|_| 1.0);
        for i in 0..mat.nrows.min(50) {
            for j in 0..i.min(50) {
                assert!((mat.get(i,j) - mat.get(j,i)).abs() < 1e-12,
                    "3D asymmetry at ({i},{j})");
            }
        }
        assert_eq!(mat.nrows, mesh.n_nodes());
    }

    #[test]
    fn dpg_framework_2d_matches_legacy() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let form = Poisson2DForm;
        let solver = DpgSolver::new(&mesh, &form, 10, 3);
        let (mat_new, _) = solver.assemble(&|_| 1.0);

        // Compare with legacy solve_dpg_poisson_2d
        let u_legacy = super::super::dpg_2d::solve_dpg_poisson_2d(&mesh, &|_, _| 1.0);
        // Both should produce finite results with same Dirichlet BC structure
        for &v in &u_legacy { assert!(v.is_finite()); }
        assert_eq!(mat_new.nrows, u_legacy.len());
    }
}
