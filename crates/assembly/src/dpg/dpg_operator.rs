//! DPG normal equations operator and Shat preconditioner builder.
//!
//! # DPG Normal Operator
//! The DPG normal equations form the system `A x = b` where:
//! - `A = B^T * S^{-1} * B`, with `B = [B0, Bhat]`
//! - `b = B^T * S^{-1} * F`
//!
//! A is never formed explicitly; `DpgNormalOperator` applies it matrix-free.
//!
//! # Shat Preconditioner Block
//! `Shat = Bhat^T * S^{-1} * Bhat` is formed via sparse triple product
//! `RAP(Bhat, Sinv, Bhat)`, matching MFEM's `RAP(matBhat, matSinv, matBhat)`.

use std::sync::Arc;

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;

use super::sinv::SinvBuilder;

// ─── Assemble S^{-1} as a global sparse matrix (MFEM InverseIntegrator path) ─

/// Assemble the per-element (M+K)^{-1} blocks into a single sparse matrix,
/// matching MFEM's `InverseIntegrator(Sum(Diffusion, Mass))` assembled via
/// `BilinearForm::Assemble()`.
pub fn assemble_sinv_sparse<M: MeshTopology>(
    sinv: &SinvBuilder<M>,
) -> CsrMatrix<f64> {
    let n = sinv.n_dofs_total();
    let nt = sinv.n_per_elem();
    let mut coo = CooMatrix::new(n, n);
    for e in 0..sinv.n_elements() {
        let block = sinv.elem_inverse(e as u32);
        let dofs = sinv.elem_dofs(e as u32);
        for i in 0..nt {
            let gi = dofs[i];
            for j in 0..nt {
                let v = block[i * nt + j];
                if v.abs() > 1e-30 {
                    coo.add(gi, dofs[j], v);
                }
            }
        }
    }
    coo.into_csr()
}

// ─── DpgNormalOperator ───────────────────────────────────────────────────────

/// Matrix-free normal equations operator for the DPG 2×2 block system.
///
/// Applies `A = B^T * S^{-1} * B` where `B = [B0, Bhat]` without forming A
/// explicitly.  Each `apply` call computes:
///
/// 1. `t0 = B * x = B0 * x_trial + Bhat * x_trace`
/// 2. `t1 = S^{-1} * t0`
/// 3. `y0 = B0^T * t1`
///    `y1 = Bhat^T * t1`
/// 4. Zero Dirichlet DOFs in y
pub struct DpgNormalOperator<M: MeshTopology> {
    n_total: usize,
    n_test: usize,
    n_trial: usize,
    n_trace: usize,
    b0: CsrMatrix<f64>,
    b0_t: CsrMatrix<f64>,
    bhat: CsrMatrix<f64>,
    bhat_t: CsrMatrix<f64>,
    sinv: Arc<SinvBuilder<M>>,
    ess_dofs: Arc<Vec<usize>>,
}

impl<M: MeshTopology> DpgNormalOperator<M> {
    /// Create a new DPG normal operator.
    ///
    /// # Arguments
    /// * `b0` — mixed bilinear form: trial × test (n_test × n_trial)
    /// * `bhat` — trace coupling matrix: test × trace (n_test × n_trace)
    /// * `sinv` — block-diagonal S^{-1} on the test space
    /// * `ess_dofs` — Dirichlet DOF indices (in the trial block)
    pub fn new(
        b0: CsrMatrix<f64>,
        bhat: CsrMatrix<f64>,
        sinv: SinvBuilder<M>,
        ess_dofs: Vec<usize>,
    ) -> Self {
        let n_trial = b0.ncols;
        let n_trace = bhat.ncols;
        let n_test = b0.nrows;
        let n_total = n_trial + n_trace;
        let b0_t = b0.transpose();
        let bhat_t = bhat.transpose();
        DpgNormalOperator {
            n_total,
            n_test,
            n_trial,
            n_trace,
            b0,
            b0_t,
            bhat,
            bhat_t,
            sinv: Arc::new(sinv),
            ess_dofs: Arc::new(ess_dofs),
        }
    }

    /// Apply the normal equation operator: `y = A * x`.
    ///
    /// `x` is the concatenated vector `[x_trial; x_trace]`.
    /// `y` is the concatenated vector `[y_trial; y_trace]`.
    pub fn apply(&self, x: &[f64], y: &mut [f64]) {
        y.fill(0.0);

        // 1. t0 = B * x = B0 * x_trial + Bhat * x_trace
        let mut t0 = vec![0.0; self.n_test];
        self.b0.spmv(&x[..self.n_trial], &mut t0);
        if self.n_trace > 0 {
            let mut tt = vec![0.0; self.n_test];
            self.bhat.spmv(&x[self.n_trial..], &mut tt);
            for i in 0..self.n_test {
                t0[i] += tt[i];
            }
        }

        // 2. t1 = S^{-1} * t0
        let mut t1 = vec![0.0; self.n_test];
        self.sinv.apply(&t0, &mut t1);

        // 3. y = B^T * t1
        // y_trial = B0^T * t1
        self.b0_t.spmv(&t1, &mut y[..self.n_trial]);

        if self.n_trace > 0 {
            // y_trace = Bhat^T * t1
            self.bhat_t.spmv(&t1, &mut y[self.n_trial..]);
        }

        // 4. Zero Dirichlet DOFs
        for &d in self.ess_dofs.iter() {
            if d < self.n_total {
                y[d] = 0.0;
            }
        }
    }

    /// Return a closure suitable for `solve_pcg_operator_precond`.
    pub fn as_closure(&self) -> impl Fn(&[f64], &mut [f64]) + '_ {
        |x, y| self.apply(x, y)
    }

    /// Total system size (n_trial + n_trace).
    pub fn n_total(&self) -> usize {
        self.n_total
    }

    /// Compute the DPG a posteriori error estimator using stored operators.
    ///
    /// Returns `||B*x - F||_{S^{-1}}` where `x = [x_trial; x_trace]`.
    pub fn compute_residual(&self, f_test: &[f64], x: &[f64]) -> f64 {
        let n_test = self.n_test;
        let mut r = vec![0.0; n_test];

        // r = B0 * x_trial
        self.b0.spmv(&x[..self.n_trial], &mut r);
        // r += Bhat * x_trace
        if self.n_trace > 0 {
            let mut bt = vec![0.0; n_test];
            self.bhat.spmv(&x[self.n_trial..], &mut bt);
            for i in 0..n_test {
                r[i] += bt[i];
            }
        }
        // r -= F
        for i in 0..n_test {
            r[i] -= f_test[i];
        }

        // e = S^{-1} * r
        let mut e = vec![0.0; n_test];
        self.sinv.apply(&r, &mut e);

        // sqrt(r · e)
        r.iter().zip(e.iter()).map(|(a, b)| a * b).sum::<f64>().abs().sqrt()
    }
}

// ─── build_shat ──────────────────────────────────────────────────────────────

/// Build `Shat = Bhat^T * S^{-1} * Bhat` as an explicit sparse matrix.
///
/// Used as the (1,1) block of the block-diagonal DPG preconditioner.
/// Uses the sparse triple product `RAP(Bhat, Sinv, Bhat)` via
/// `CsrMatrix::rap_product`, matching MFEM's `RAP(matBhat, matSinv, matBhat)`.
pub fn build_shat<M: MeshTopology>(
    bhat: &CsrMatrix<f64>,
    sinv: &SinvBuilder<M>,
    n_trace: usize,
) -> CsrMatrix<f64> {
    let mat_sinv = assemble_sinv_sparse(sinv);
    bhat.rap_product(&mat_sinv, bhat)
}

// ─── DPG residual (error estimator) ─────────────────────────────────────────

/// Compute the DPG a posteriori error estimator.
///
/// Returns `||B*x - F||_{S^{-1}} = sqrt(r · S^{-1} · r)`
/// where `r = B0 * x_trial + Bhat * x_trace - F` is the residual in the
/// test space, and `S^{-1}` is the inverse of the test-space Gram matrix.
///
/// This is the energy norm of the residual, a built-in error estimator
/// in the DPG method.
pub fn compute_dpg_residual<M: MeshTopology>(
    b0: &CsrMatrix<f64>,
    bhat: &CsrMatrix<f64>,
    sinv: &SinvBuilder<M>,
    f_test: &[f64],
    x_trial: &[f64],
    x_trace: &[f64],
) -> f64 {
    // r = B0 * x_trial + Bhat * x_trace - F
    let n_test = f_test.len();
    let mut r = vec![0.0; n_test];

    b0.spmv(x_trial, &mut r);
    if !x_trace.is_empty() {
        let mut bt = vec![0.0; n_test];
        bhat.spmv(x_trace, &mut bt);
        for i in 0..n_test {
            r[i] += bt[i];
        }
    }
    for i in 0..n_test {
        r[i] -= f_test[i];
    }

    // e = S^{-1} * r
    let mut e = vec![0.0; n_test];
    sinv.apply(&r, &mut e);

    // sqrt(r · e) = sqrt(r · S^{-1} · r)
    let val: f64 = r.iter().zip(e.iter()).map(|(a, b)| a * b).sum();
    val.abs().sqrt()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::standard::{DiffusionIntegrator, DomainSourceIntegrator};
    use crate::{Assembler, MixedAssembler, MixedBilinearIntegrator};
    use crate::integrator::QpData;
    use crate::dpg::{SinvBuilder, assemble_bhat};
    use fem_mesh::{refine_uniform, Mesh};
    use fem_space::{H1Space, L2Space, DpgTraceSpace, boundary_dofs};
    use fem_space::fe_space::FESpace;

    // Replicate the MixedDiffusion from the example
    struct MixedDiffusion;
    impl MixedBilinearIntegrator for MixedDiffusion {
        fn add_to_element_matrix(&self, qp_row: &QpData<'_>, qp_col: &QpData<'_>, m: &mut [f64]) {
            let nr = qp_row.n_dofs; let nc = qp_col.n_dofs; let d = qp_col.dim; let w = qp_col.weight;
            for k in 0..d {
                for i in 0..nr {
                    let gik = qp_row.grad_phys[i * d + k];
                    for j in 0..nc {
                        m[i * nc + j] += w * gik * qp_col.grad_phys[j * d + k];
                    }
                }
            }
        }
    }

    fn run_dpg_solve(n: usize) -> (usize, f64) {
        let mesh = Mesh::<2>::unit_square_tri(n);

        // Refine to get enough elements
        let rl = { let ne = mesh.n_elems() as f64;
            (10000.0 / ne).ln().max(0.0) / (2.0_f64).ln() / 2.0_f64 } as usize;
        let mesh = if rl > 0 { let mut m = mesh; for _ in 0..rl { m = refine_uniform(&m); } m } else { mesh };

        // Spaces
        let t_order = 1u8;
        let tr_order = 0u8;
        let te_order = 2u8; // enriched test = trial + 1

        let x0 = H1Space::new(mesh.clone(), t_order);
        let test = L2Space::new(mesh.clone(), te_order);
        let trace = DpgTraceSpace::new(mesh.clone(), tr_order);

        let s0_sz = x0.n_dofs();
        let s1_sz = trace.n_dofs();
        let st_sz = test.n_dofs();

        // F on test space
        let qo = (te_order as u8 * 2 + 2).max(3);
        let f_test = Assembler::assemble_linear(&test, &[&DomainSourceIntegrator::new(|_| 1.0)], qo);

        // B0
        let ess_tags: Vec<i32> = mesh.unique_boundary_tags();
        let dm = x0.dof_manager();
        let ess_dofs: Vec<u32> = boundary_dofs(&mesh as &dyn MeshTopology, dm, &ess_tags);
        let mut b0 = MixedAssembler::assemble_bilinear(&test, &x0, &[&MixedDiffusion], qo);
        let ess_usize: Vec<usize> = ess_dofs.iter().map(|&d| d as usize).collect();

        // Zero BC columns of B0
        for &d in &ess_dofs {
            let c = d as usize;
            for row in 0..b0.nrows {
                for p in b0.row_ptr[row]..b0.row_ptr[row + 1] {
                    if b0.col_idx[p] as usize == c {
                        b0.values[p] = 0.0;
                    }
                }
            }
        }

        // Bhat
        let qf = (te_order as u8 * 2).max(2);
        let bhat = assemble_bhat(&test, &trace, qf);

        // Sinv
        let sinv = SinvBuilder::build(&test, qo);

        // S0
        let mut s0_mat = Assembler::assemble_bilinear(&x0, &[&DiffusionIntegrator { kappa: 1.0 }], qo);
        let mut zr = vec![0.0; s0_sz];
        fem_space::apply_dirichlet(&mut s0_mat, &mut zr, &ess_dofs, &vec![0.0; ess_dofs.len()]);

        // RHS: b = B^T * S^{-1} * F
        let mut sf = vec![0.0; st_sz];
        sinv.apply(&f_test, &mut sf);
        let mut rhs = vec![0.0; s0_sz + s1_sz];
        for i in 0..s0_sz { rhs[i] = b0_t(&b0, i, &sf); }
        for i in 0..st_sz {
            let v = sf[i];
            if v.abs() < 1e-30 { continue; }
            for p in bhat.row_ptr[i]..bhat.row_ptr[i + 1] {
                rhs[s0_sz + bhat.col_idx[p] as usize] += bhat.values[p] * v;
            }
        }
        for &d in &ess_dofs { rhs[d as usize] = 0.0; }

        // Shat
        let shat = build_shat(&bhat, &sinv, s1_sz);

        // Preconditioner: inner CG solves for S0^{-1} and Shat^{-1}
        use fem_solver::SolverConfig;
        let inner_cfg = SolverConfig { rtol: 1e-3, max_iter: 200, verbose: false, ..Default::default() };

        let s0_mat_ref = &s0_mat;
        let shat_ref = &shat;
        let ess_precond = ess_usize.clone();

        let precond = move |r: &[f64], z: &mut [f64]| {
            fem_solver::solve_cg_operator(s0_sz, s0_sz, |x, y| s0_mat_ref.spmv(x, y), &r[..s0_sz], &mut z[..s0_sz], &inner_cfg).ok();
            if s1_sz > 0 {
                fem_solver::solve_cg_operator(s1_sz, s1_sz, |x, y| shat_ref.spmv(x, y), &r[s0_sz..], &mut z[s0_sz..], &inner_cfg).ok();
            }
            for &d in &ess_precond { z[d] = 0.0; }
        };

        // Normal operator
        let op = DpgNormalOperator::new(b0, bhat, sinv, ess_usize);
        let n_tot = op.n_total();

        // Solve
        let mut x = vec![0.0; n_tot];
        // rtol uses the squared-norm criterion (nom ≤ nom0·rtol², MFEM PCG);
        // 1e-12 would demand 1e-24 relative accuracy and exhaust max_iter.
        let cfg = SolverConfig { rtol: 1e-6, atol: 0.0, max_iter: 200, verbose: false, ..Default::default() };
        let result = fem_solver::solve_pcg_operator_precond(n_tot, op.as_closure(), &rhs, &mut x, precond, &cfg);
        let res = result.expect("PCG solve failed");

        (res.iterations, res.final_residual)
    }

    fn b0_t(b0: &CsrMatrix<f64>, col: usize, y: &[f64]) -> f64 {
        let mut s = 0.0;
        for row in 0..b0.nrows {
            for p in b0.row_ptr[row]..b0.row_ptr[row + 1] {
                if b0.col_idx[p] as usize == col {
                    s += b0.values[p] * y[row];
                    break;
                }
            }
        }
        s
    }

    #[test]
    fn dpg_solve_converges() {
        let (iters, final_res) = run_dpg_solve(2);
        eprintln!("DPG solve: iters={iters}, final_res={final_res:.3e}");
        assert!(final_res < 1e-6, "PCG did not converge enough: res={final_res:.3e}");
    }
}
