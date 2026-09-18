//! Bramble–Pasciak solver assembly helpers and solver.
//!
//! 1:1 port of MFEM `miniapps/solvers/bramble_pasciak.{hpp,cpp}` —
//! [`BramblePasciakSolver`] (block-operator construction, `Init`, `Mult` and
//! `ConstructMassPreconditioner`) and, in [`crate::bpcg::solve_bpcg`], the
//! `BPCGSolver::Mult` iteration body.
//!
//! For the Darcy saddle-point system `[[M, Bᵀ], [B, 0]]`, Bramble–Pasciak
//! preconditioning requires an SPD `Q` such that `M − Q` stays SPD.  MFEM
//! builds `Q` from the *element* mass matrices `M_T`:
//!
//! ```text
//!     M_T x = λ · diag(M_T) x          (generalized eigenproblem, per element)
//!     Q_T  = α · λ_min · diag(M_T)     (0 < α < 1, q_scaling)
//! ```
//!
//! so that on every element `Q_T` is a scaled copy of the diagonal of `M_T`.
//! [`element_q_scaling`] computes the per-element factor `α·λ_min` following
//! MFEM's no-LAPACK path exactly:
//! 1. form `M̄ = D^{-1/2} M_T D^{-1/2}` (`DenseMatrix::InvSymmetricScaling`),
//! 2. smallest eigenvalue of `M̄` by the inverse power method on `M̄^{-1}`,
//!    started from `x = Vector::Randomize(696383552 + 779345·elem)` — i.e.
//!    glibc `srand` + `rand()/2³¹` ([`crate::geometric_mg::GlibcRand`]) —
//!    iterating until the relative change of the Rayleigh value is `≤ 1e-12`
//!    (at most 1000 iterations, as MFEM).
//!
//! [`element_q_block`] assembles the per-element block
//! `Q_T = α·λ_min·diag(M_T)` (MFEM `Q_i.Diag(diag_i)`); the global scatter is
//! `fem_assembly::Assembler::assemble_from_element_matrices`, the port of
//! `qVarf.AssembleElementMatrix(i, Q_i, 1)` (the solver crate itself is
//! assembly-free, so the caller supplies the per-element matrices).
//!
//! [`BramblePasciakSolver`] then transforms the block operator with
//! `X = A·N − Id`, `N = diag(invQ, 0)`, and solves `X·A x = X·b` either with
//! BPCG (`use_bpcg = true`, the implicit iteration of [`crate::bpcg`]) or
//! with regular PCG on `mop_ = (A·N − Id)·A` preconditioned by
//! `cpc = diag(diag(M)⁻¹, M1)` (`use_bpcg = false`).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use fem_linalg::dense::{lu_factor, lu_solve};
use fem_linalg::{CsrMatrix, PrintLevel, SolverConfig, SolverError};

use crate::block::BlockSystem;
use crate::bpcg::solve_bpcg;
use crate::darcy_solvers::{
    schur_complement_bmb_diag, IterSolveParameters, SchurApprox, SchurMode,
};
use crate::geometric_mg::GlibcRand;
use crate::{solve_cg_mfem, SliOptions};

/// Per-element scaling factor `q_scaling · λ_min` of MFEM
/// `BramblePasciakSolver::ConstructMassPreconditioner`.
///
/// # Arguments
/// * `m_elem` — row-major `n × n` element mass matrix `M_T` (SPD).
/// * `n`      — element matrix dimension.
/// * `q_scaling` — `α`, must lie in `(0, 1)` (MFEM `MFEM_ASSERT`).
/// * `elem_index` — global element index; seeds the power iteration exactly
///   like MFEM (`696383552 + 779345·elem`).
///
/// # Returns
/// `q_scaling · λ_min` where `λ_min` is the smallest eigenvalue of
/// `M_T x = λ diag(M_T) x`.
///
/// # Panics
/// If `q_scaling` is outside `(0, 1)`, the element matrix is singular, or the
/// inverse power iteration fails to converge in 1000 iterations (MFEM calls
/// `MFEM_ASSERT`/`MFEM_VERIFY` and aborts in the same situations).
pub fn element_q_scaling(m_elem: &[f64], n: usize, q_scaling: f64, elem_index: usize) -> f64 {
    assert!(
        (q_scaling > 0.0) && (q_scaling < 1.0),
        "Invalid Q-scaling factor: q_scaling = {q_scaling}"
    );
    assert_eq!(m_elem.len(), n * n, "element matrix must be n×n row-major");

    // D = diag(M_T); form M̄ = D^{-1/2} · M_T · D^{-1/2} in place
    // (MFEM DenseMatrix::InvSymmetricScaling).
    let mut m = m_elem.to_vec();
    let mut inv_sqrt = vec![0.0f64; n];
    for i in 0..n {
        inv_sqrt[i] = 1.0 / m[i * n + i].sqrt();
    }
    for j in 0..n {
        for i in 0..n {
            m[j * n + i] *= inv_sqrt[i] * inv_sqrt[j];
        }
    }

    // Inverse power iteration for the smallest eigenvalue of M̄, i.e. the
    // largest eigenvalue of M̄^{-1} (MFEM no-LAPACK branch).
    let mut piv = vec![0usize; n];
    lu_factor(&mut m, n, &mut piv)
        .expect("element mass matrix must be non-singular for the power method");

    // x.Randomize(696383552 + 779345 * elem): glibc rand()/2^31 draws.
    let seed = 696_383_552u64.wrapping_add(779_345 * elem_index as u64) as u32;
    let mut rng = GlibcRand::new(seed);
    let mut x: Vec<f64> = (0..n).map(|_| rng.rand_real()).collect();

    const REL_TOL: f64 = 1e-12;
    const MAX_ITER: usize = 1000;
    let mut eval = 0.0f64;
    let mut eval_prev;
    let mut mx = vec![0.0f64; n];
    let mut iter = 0usize;
    let mut converged = false;
    loop {
        eval_prev = eval;
        // M̄^{-1} · x (LU factors of `m` already computed).
        mx.copy_from_slice(&x);
        lu_solve(&m, n, &piv, &mut mx);
        eval = mx.iter().map(|v| v * v).sum::<f64>().sqrt();
        let inv_eval = 1.0 / eval;
        for (xi, mxi) in x.iter_mut().zip(mx.iter()) {
            *xi = *mxi * inv_eval;
        }
        iter += 1;
        let rel = (eval - eval_prev).abs() / eval.abs();
        if rel <= REL_TOL {
            converged = true;
            break;
        }
        if iter >= MAX_ITER {
            break;
        }
    }
    assert!(
        converged,
        "Inverse power method did not converge.\n\t iter      = {iter}\n\t \
         eval_i    = {eval}\n\t eval_prev = {eval_prev}\n\t rel change = {}",
        (eval - eval_prev).abs() / eval.abs()
    );

    let lambda_min = 1.0 / eval;
    q_scaling * lambda_min
}

/// Per-element block `Q_T = q_scaling·λ_min·diag(M_T)` of MFEM
/// `BramblePasciakSolver::ConstructMassPreconditioner`
/// (`scaling = q_scaling*eval_i; diag_i.Set(scaling, diag_i);
/// Q_i.Diag(diag_i.GetData(), diag_i.Size())`).
///
/// # Arguments
/// * `m_elem` — row-major `n × n` element mass matrix `M_T` (SPD, in the
///   element's own dof convention — the diagonal is sign-invariant).
/// * `n`      — element matrix dimension.
/// * `q_scaling` — `α`, must lie in `(0, 1)`.
/// * `elem_index` — global element index (seeds the power iteration).
///
/// # Returns
/// The row-major `n × n` diagonal block `Q_T`; scatter it globally with
/// `fem_assembly::Assembler::assemble_from_element_matrices`.
///
/// # Panics
/// Like [`element_q_scaling`] (out-of-range `q_scaling`, singular element
/// matrix, or a non-converging power iteration).
pub fn element_q_block(m_elem: &[f64], n: usize, q_scaling: f64, elem_index: usize) -> Vec<f64> {
    let scaling = element_q_scaling(m_elem, n, q_scaling, elem_index);
    let mut q_i = vec![0.0_f64; n * n];
    for i in 0..n {
        q_i[i * n + i] = scaling * m_elem[i * n + i];
    }
    q_i
}

/// Parameters for the [`BramblePasciakSolver`] method — MFEM
/// `blocksolvers::BPSParameters` (`IterSolveParameters` base +
/// `use_bpcg`/`q_scaling`, bramble_pasciak.hpp:56).
#[derive(Debug, Clone)]
pub struct BpsParameters {
    /// MFEM `IterSolveParameters` base (defaults: `print_level = 0`,
    /// `max_iter = 500`, `abs_tol = 1e-12`, `rel_tol = 1e-9`).
    pub iter: IterSolveParameters,
    /// MFEM `use_bpcg`: whether to use BPCG (rather than regular PCG on the
    /// transformed operator).
    pub use_bpcg: bool,
    /// MFEM `q_scaling`: scaling (> 0 and < 1) of the `Q` preconditioner.
    pub q_scaling: f64,
}

impl Default for BpsParameters {
    fn default() -> Self {
        Self {
            iter: IterSolveParameters::default(),
            use_bpcg: true,
            q_scaling: 0.5,
        }
    }
}

/// Serial port of MFEM `blocksolvers::BramblePasciakSolver` for the Darcy
/// saddle system `A = [[M, Bᵀ], [B, 0]]`.
///
/// Solves `X·A x = X·b` with `X = A·N − Id`, `N = diag(invQ, 0)`
/// (bramble_pasciak.cpp:94-123 `Init`):
/// * `use_bpcg = true` — the implicit Bramble–Pasciak CG of MFEM
///   `BPCGSolver::Mult` ([`crate::bpcg::solve_bpcg`]) on the *untransformed*
///   operator `A` with particular preconditioner `P = cpc·tri`
///   (`tri = [[I, 0], [B·invQ, −I]]`, `cpc = diag(invQ, M1)`) and
///   incomplete preconditioner `N`;
/// * `use_bpcg = false` — regular PCG (MFEM `CGSolver`) on the transformed
///   operator `mop_ = (A·N − Id)·A` with block-diagonal preconditioner
///   `cpc = diag(diag(M)⁻¹, M1)`, applied to the transformed right-hand side
///   `map_·b = (A·N − Id)·b` (bramble_pasciak.cpp:199-212).
///
/// Serial cut of the C++ form-based constructor: `M0 = HypreDiagScale(M)`
/// (Jacobi) and `S = B·diag(M)⁻¹·Bᵀ` with `M1` selected by [`SchurMode`]
/// (C++ hard-wires `HypreBoomerAMG(S)`; use `SchurMode::Amg` to match).
/// [`Mult`][Self::mult] mirrors `BramblePasciakSolver::Mult`, including the
/// `ess_zero_dofs_` post-zeroing of `SetEssZeroDofs`.
pub struct BramblePasciakSolver {
    n_u: usize,
    n_p: usize,
    n: usize,
    /// Flat operator `A = [[M, Bᵀ], [B, 0]]` (MFEM `oop_` / `temp_oop`).
    flat_a: CsrMatrix<f64>,
    /// `B` (`n_p × n_u`) for the `B·invQ` block of `tri` (MFEM `BinvQ`).
    b: CsrMatrix<f64>,
    /// `diag(Q)⁻¹` (MFEM `HypreDiagScale(Q)`, the `invQ` operator).
    inv_q: Vec<f64>,
    /// `diag(M)` — `M0 = HypreDiagScale(M)`, block 0 of `cpc_` in the cg
    /// branch.
    m_diag: Vec<f64>,
    /// Schur-block solver `M1` (MFEM `HypreBoomerAMG(S)`).
    m1: SchurApprox,
    use_bpcg: bool,
    param: IterSolveParameters,
    ess_zero_dofs: Vec<usize>,
    last_iters: AtomicUsize,
    last_converged: AtomicBool,
}

/// MFEM legacy print level → fem-rs [`PrintLevel`] (MFEM
/// `IterativeSolver::SetPrintLevel`).
fn print_level_from(level: i32) -> PrintLevel {
    match level {
        i32::MIN..=-1 => PrintLevel::Silent,
        0 => PrintLevel::WarningsOnly,
        1 => PrintLevel::Iterations,
        2 => PrintLevel::Summary,
        _ => PrintLevel::FirstAndLast,
    }
}

impl BramblePasciakSolver {
    /// MFEM `BramblePasciakSolver::Init` for the user-provided `M`, `B`, `Q`
    /// (the second C++ constructor; the form-based one just assembles these
    /// first — in the serial cut the caller assembles them, with `Q` via
    /// [`element_q_block`] + `Assembler::assemble_from_element_matrices`).
    ///
    /// `m1_mode` selects the `M1` Schur-block solver (`SchurMode::Amg`
    /// reproduces the C++ `HypreBoomerAMG` choice).
    pub fn new(
        m: &CsrMatrix<f64>,
        b: &CsrMatrix<f64>,
        q: &CsrMatrix<f64>,
        param: BpsParameters,
        m1_mode: SchurMode,
    ) -> Self {
        assert!(m.nrows == m.ncols, "BP: M must be square");
        assert!(m.ncols == b.ncols, "BP: B must be (n_p × n_u)");
        assert!(
            q.nrows == m.nrows && q.ncols == m.ncols,
            "BP: Q must have the shape of M"
        );
        let n_u = m.nrows;
        let n_p = b.nrows;
        let n = n_u + n_p;

        // S = B·diag(M)⁻¹·Bᵀ (MFEM: InvScaleRows on Bᵀ, ParMult) and
        // M1 = BoomerAMG(S) / exact / Jacobi.
        let m_diag: Vec<f64> = (0..n_u)
            .map(|i| {
                let d = m.get(i, i);
                assert!(d != 0.0, "BP: zero diagonal in M at row {i}");
                d
            })
            .collect();
        let s = schur_complement_bmb_diag(b, &m_diag);
        let m1 = SchurApprox::new(&s, m1_mode);

        // invQ = diag(Q)⁻¹ (MFEM HypreDiagScale(Q)).
        let inv_q: Vec<f64> = (0..n_u)
            .map(|i| {
                let d = q.get(i, i);
                assert!(d != 0.0, "BP: zero diagonal in Q at row {i}");
                1.0 / d
            })
            .collect();

        // oop_ (bpcg branch) / temp_oop (cg branch): [[M, Bᵀ], [B, 0]].
        let flat_a = BlockSystem {
            a: m.clone(),
            bt: b.transpose(),
            b: b.clone(),
            c: None,
        }
        .to_flat_csr();

        Self {
            n_u,
            n_p,
            n,
            flat_a,
            b: b.clone(),
            inv_q,
            m_diag,
            m1,
            use_bpcg: param.use_bpcg,
            param: param.iter,
            ess_zero_dofs: Vec::new(),
            last_iters: AtomicUsize::new(0),
            last_converged: AtomicBool::new(false),
        }
    }

    /// MFEM `SetEssZeroDofs`: solution dofs forced to zero after the solve.
    pub fn set_ess_zero_dofs(&mut self, dofs: &[usize]) {
        self.ess_zero_dofs = dofs.to_vec();
    }

    /// MFEM `GetNumIterations`.
    pub fn num_iterations(&self) -> usize {
        self.last_iters.load(Ordering::Relaxed)
    }

    /// Whether the last [`mult`][Self::mult] converged.
    pub fn converged(&self) -> bool {
        self.last_converged.load(Ordering::Relaxed)
    }

    /// Block sizes `[0, n_u, n_u + n_p]` (MFEM `DarcySolver::offsets_`).
    pub fn offsets(&self) -> [usize; 3] {
        [0, self.n_u, self.n]
    }

    /// MFEM `BramblePasciakSolver::Mult`: solve, then zero the ess dofs.
    pub fn mult(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.n, "BP: rhs size mismatch");
        assert_eq!(y.len(), self.n, "BP: solution size mismatch");
        if self.use_bpcg {
            // MFEM BPCGSolver with iterative_mode == false: x = 0, r = b
            // (solve_bpcg's iterative_mode = true contract with x = 0 gives
            // the identical iterate sequence).
            y.fill(0.0);
            let cfg = SolverConfig {
                rtol: self.param.rel_tol,
                atol: self.param.abs_tol,
                max_iter: self.param.max_iter,
                verbose: false,
                print_level: print_level_from(self.param.print_level),
            };
            match solve_bpcg(
                self.n,
                |v, w| self.apply_a(v, w),
                |v, w| self.apply_ppc(v, w),
                |v, w| self.apply_ipc(v, w),
                x,
                y,
                &cfg,
            ) {
                Ok(r) => {
                    self.last_iters.store(r.iterations, Ordering::Relaxed);
                    self.last_converged.store(true, Ordering::Relaxed);
                }
                Err(e) => {
                    self.last_converged.store(false, Ordering::Relaxed);
                    if let SolverError::ConvergenceFailed { max_iter, .. } = e {
                        self.last_iters.store(max_iter, Ordering::Relaxed);
                    }
                }
            }
        } else {
            // map_·x = (A·N − Id)·x (the transformed right-hand side), then
            // PCG on mop_ = map_·A with cpc_ = diag(diag(M)⁻¹, M1).
            let mut transformed = vec![0.0_f64; self.n];
            self.apply_map(x, &mut transformed);
            let opts = SliOptions {
                rel_tol: self.param.rel_tol,
                abs_tol: self.param.abs_tol,
                max_iter: self.param.max_iter as i32,
                print_level: self.param.print_level,
            };
            let res = solve_cg_mfem(
                self.n,
                |v, w| self.apply_mop(v, w),
                &transformed,
                y,
                Some(|v: &[f64], w: &mut [f64]| self.apply_cpc(v, w)),
                &opts,
                false,
                None,
            );
            self.last_iters
                .store(res.iterations.max(0) as usize, Ordering::Relaxed);
            self.last_converged.store(res.converged, Ordering::Relaxed);
        }
        for &dof in &self.ess_zero_dofs {
            y[dof] = 0.0;
        }
    }

    /// `y = A·v` (flat `[[M, Bᵀ], [B, 0]]`).
    fn apply_a(&self, v: &[f64], w: &mut [f64]) {
        self.flat_a.spmv(v, w);
    }

    /// `y = N·v = (invQ·v_u, 0)` (MFEM `ipc_`, both branches).
    fn apply_ipc(&self, v: &[f64], w: &mut [f64]) {
        for i in 0..self.n_u {
            w[i] = self.inv_q[i] * v[i];
        }
        for wi in &mut w[self.n_u..] {
            *wi = 0.0;
        }
    }

    /// `y = map_·v = A·N·v − v` (MFEM `map_ = SumOperator(temp_AN, 1, id, −1)`).
    fn apply_map(&self, v: &[f64], w: &mut [f64]) {
        let mut nv = vec![0.0_f64; self.n];
        self.apply_ipc(v, &mut nv);
        self.apply_a(&nv, w);
        for (wi, &vi) in w.iter_mut().zip(v) {
            *wi -= vi;
        }
    }

    /// `y = mop_·v = map_·(A·v) = (A·N − Id)·A·v` (MFEM `mop_`).
    fn apply_mop(&self, v: &[f64], w: &mut [f64]) {
        let mut av = vec![0.0_f64; self.n];
        self.apply_a(v, &mut av);
        self.apply_map(&av, w);
    }

    /// `y = P·v = cpc·tri·v` (MFEM `ppc_` of the bpcg branch) with
    /// `tri = [[I, 0], [B·invQ, −I]]`, `cpc = diag(invQ, M1)`.
    fn apply_ppc(&self, v: &[f64], w: &mut [f64]) {
        // tri·v: (v_u, B·invQ·v_u − v_p)
        let invq_u: Vec<f64> = self.inv_q
            .iter()
            .zip(&v[..self.n_u])
            .map(|(qi, vi)| qi * vi)
            .collect();
        let mut bpu = vec![0.0_f64; self.n_p];
        self.b.spmv(&invq_u, &mut bpu);
        let tri_p: Vec<f64> = bpu
            .iter()
            .zip(&v[self.n_u..])
            .map(|(&bi, &pi)| bi - pi)
            .collect();
        // cpc·(tri·v): (invQ·(tri·v)_u, M1·(tri·v)_p)
        for i in 0..self.n_u {
            w[i] = self.inv_q[i] * v[i];
        }
        self.m1.apply(&tri_p, &mut w[self.n_u..]);
    }

    /// `y = cpc·v = (diag(M)⁻¹·v_u, M1·v_p)` (MFEM `cpc_` of the cg branch).
    fn apply_cpc(&self, v: &[f64], w: &mut [f64]) {
        for i in 0..self.n_u {
            w[i] = v[i] / self.m_diag[i];
        }
        self.m1.apply(&v[self.n_u..], &mut w[self.n_u..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q_scaling_matches_analytic_3x3() {
        // M = [[3,1,0],[1,3,1],[0,1,3]] with D = diag(3,3,3): the generalized
        // eigenproblem M x = λ D x reduces to the plain one for M̄ = M/3 whose
        // eigenvalues are {1, (3±√2)/3}; λ_min = (3−√2)/3.
        let m: [f64; 9] = [3.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 3.0];
        let lam_min = (3.0 - std::f64::consts::SQRT_2) / 3.0;
        let got = element_q_scaling(&m, 3, 0.5, 7);
        let expect = 0.5 * lam_min;
        assert!(
            (got - expect).abs() < 1e-10,
            "scaling {got:.15e} != 0.5·λ_min {expect:.15e}"
        );
        // Same value regardless of the power-iteration seed (element index).
        for elem in 0..16usize {
            let g = element_q_scaling(&m, 3, 0.5, elem);
            assert!(
                (g - expect).abs() < 1e-10,
                "elem {elem}: scaling {g:.15e} != {expect:.15e}"
            );
        }
    }

    #[test]
    fn q_scaling_matches_analytic_2x2() {
        // M = [[2,-0.5],[-0.5,2]], D = diag(2,2): M̄ = M/2 has eigenvalues
        // {1.25, 0.75} → λ_min = 0.75.
        let m: [f64; 4] = [2.0, -0.5, -0.5, 2.0];
        let got = element_q_scaling(&m, 2, 0.5, 0);
        assert!((got - 0.375).abs() < 1e-10, "scaling {got:.15e} != 0.375");
    }

    #[test]
    #[should_panic(expected = "Invalid Q-scaling")]
    fn q_scaling_rejects_out_of_range() {
        let m: [f64; 4] = [2.0, 0.0, 0.0, 2.0];
        let _ = element_q_scaling(&m, 2, 1.5, 0);
    }

    // ─── BramblePasciakSolver (both branches) ───────────────────────────────

    use fem_linalg::CooMatrix;

    use super::BpsParameters;

    /// Deterministic SPD `M` (strictly diagonally dominant tridiagonal, so
    /// `M − 0.5·diag(M)` is SPD) and full-row-rank `B` — the same fixture as
    /// `bpcg.rs::saddle_data`.
    fn saddle_data(n_u: usize, n_p: usize) -> (CsrMatrix<f64>, CsrMatrix<f64>) {
        let mut coo_m = CooMatrix::<f64>::new(n_u, n_u);
        for i in 0..n_u {
            coo_m.add(i, i, 2.0);
            if i > 0 {
                coo_m.add(i, i - 1, -0.5);
                coo_m.add(i - 1, i, -0.5);
            }
        }
        let mut coo_b = CooMatrix::<f64>::new(n_p, n_u);
        for r in 0..n_p {
            for c in 0..n_u {
                let v = ((r + 1) * (c + 1) % 5) as f64 * 0.1 + 0.01 * (r as f64 + c as f64);
                if v != 0.0 {
                    coo_b.add(r, c, v);
                }
            }
        }
        (coo_m.into_csr(), coo_b.into_csr())
    }

    /// Flat saddle operator `[[M, Bᵀ], [B, 0]]` action (test reference).
    fn apply_a_ref(m: &CsrMatrix<f64>, b: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
        let n_u = m.nrows;
        m.spmv(&x[..n_u], &mut y[..n_u]);
        let mut t = vec![0.0_f64; n_u];
        b.transpose().spmv(&x[n_u..], &mut t);
        for (yi, ti) in y[..n_u].iter_mut().zip(&t) {
            *yi += *ti;
        }
        b.spmv(&x[..n_u], &mut y[n_u..]);
    }

    /// Both solver branches (`use_bpcg` true/false) recover the exact solution
    /// of a small saddle system whose `Q` is `0.5·diag(M)` (the test stand-in
    /// for the elementwise `λ_min` scaling; `M − Q` stays SPD) with an exact
    /// dense Schur inverse (`SchurMode::Dense`).
    #[test]
    fn bp_solver_both_branches_recover_exact_solution() {
        let n_u = 6;
        let n_p = 3;
        let n = n_u + n_p;
        let (m, b) = saddle_data(n_u, n_p);

        let x_star: Vec<f64> =
            (0..n).map(|i| (i as f64 + 1.0) / n as f64 * 0.5 - 0.2).collect();
        let mut rhs = vec![0.0_f64; n];
        apply_a_ref(&m, &b, &x_star, &mut rhs);

        // Q = 0.5·diag(M), assembled as a diagonal CSR.
        let mut coo_q = CooMatrix::<f64>::new(n_u, n_u);
        for i in 0..n_u {
            coo_q.add(i, i, 0.5 * m.get(i, i));
        }
        let q = coo_q.into_csr();

        for use_bpcg in [true, false] {
            let param = BpsParameters {
                iter: IterSolveParameters {
                    print_level: -1,
                    max_iter: 200,
                    abs_tol: 1e-14,
                    rel_tol: 1e-10,
                },
                use_bpcg,
                q_scaling: 0.5,
            };
            let solver = BramblePasciakSolver::new(&m, &b, &q, param, SchurMode::Dense);
            let mut sol = vec![0.0_f64; n];
            solver.mult(&rhs, &mut sol);
            assert!(solver.converged(), "use_bpcg={use_bpcg}: did not converge");
            assert!(
                solver.num_iterations() < 200,
                "use_bpcg={use_bpcg}: too many iterations"
            );
            let err = sol
                .iter()
                .zip(&x_star)
                .map(|(a, e)| (a - e).abs())
                .fold(0.0_f64, f64::max);
            assert!(err < 1e-6, "use_bpcg={use_bpcg}: error too large {err:.3e}");
        }
    }

    /// `SetEssZeroDofs` forces the listed dofs to zero in the output (both
    /// branches), as MFEM `BramblePasciakSolver::Mult` does.
    #[test]
    fn bp_solver_ess_zero_dofs() {
        let n_u = 6;
        let n_p = 3;
        let n = n_u + n_p;
        let (m, b) = saddle_data(n_u, n_p);
        let mut rhs = vec![1.0_f64; n];
        apply_a_ref(&m, &b, &vec![1.0; n], &mut rhs);

        let mut coo_q = CooMatrix::<f64>::new(n_u, n_u);
        for i in 0..n_u {
            coo_q.add(i, i, 0.5 * m.get(i, i));
        }
        let q = coo_q.into_csr();

        for use_bpcg in [true, false] {
            let param = BpsParameters {
                iter: IterSolveParameters {
                    print_level: -1,
                    ..IterSolveParameters::default()
                },
                use_bpcg,
                q_scaling: 0.5,
            };
            let mut solver = BramblePasciakSolver::new(&m, &b, &q, param, SchurMode::Dense);
            solver.set_ess_zero_dofs(&[0, n_u]);
            let mut sol = vec![0.5_f64; n]; // nonzero guess must be overwritten too
            solver.mult(&rhs, &mut sol);
            assert_eq!(sol[0], 0.0, "use_bpcg={use_bpcg}: u ess dof not zeroed");
            assert_eq!(sol[n_u], 0.0, "use_bpcg={use_bpcg}: p ess dof not zeroed");
        }
    }

    /// `element_q_block` produces `α·λ_min·diag(M_T)` — diagonal, with the
    /// diagonal equal to `element_q_scaling(M_T)·diag(M_T)`.
    #[test]
    fn element_q_block_is_scaled_diagonal() {
        let m: [f64; 9] = [3.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 3.0];
        let block = element_q_block(&m, 3, 0.5, 3);
        let scaling = element_q_scaling(&m, 3, 0.5, 3);
        for (i, row) in block.chunks(3).enumerate() {
            for (j, &v) in row.iter().enumerate() {
                let expect = if i == j { scaling * m[i * 3 + i] } else { 0.0 };
                assert_eq!(v, expect, "Q_T[{i}][{j}]");
            }
        }
    }
}
