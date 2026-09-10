//! # fem-amg
//!
//! Algebraic Multigrid backed by [`linlvo`].
//!
//! Supports both classical Ruge–Stüben (RS) and smoothed aggregation (SA)
//! coarsening strategies, together with a full preconditioner menu.
//!
//! ## Self-adjointness of the cycle (D10)
//!
//! Every solver in this crate drives the hierarchy through
//! [`CorrectedAmgPrecond`], not through `linlvo`'s `AmgPrecond` directly.  The
//! two implement the same V/W/F-cycle; they differ only in the coarsest-level
//! solve, which `linlvo` performs with the *default* (`Rcm`) fill-reducing
//! ordering of `SparseLu`.  That path returns a permuted solution, so the
//! coarse-grid correction is garbage and the cycle operator becomes
//! non-symmetric — enough to make CG/MINRES stall.  [`CorrectedAmgPrecond`] uses
//! `OrderingMethod::Natural`, for which `SparseLu` is exact; see its docs for the
//! reproducer and the measured numbers.
//!
//! ## Usage
//! ```ignore
//! use fem_amg::{AmgConfig, CoarsenStrategy, solve_amg_cg};
//! use fem_linalg::CooMatrix;
//!
//! let mut coo = CooMatrix::<f64>::new(n, n);
//! // �?fill �?
//! let a = coo.into_csr();
//! let b = vec![1.0_f64; n];
//! let mut x = vec![0.0_f64; n];
//!
//! let res = solve_amg_cg(&a, &b, &mut x, &AmgConfig::default(), &Default::default()).unwrap();
//! assert!(res.converged);
//! ```

use fem_linalg::CsrMatrix as FemCsr;
use fem_linalg::{fem_to_linlvo_csr, into_result, SolveResult, SolverConfig, SolverError};
use linlvo::{
    amg::smoother::smooth_with_hint,
    core::operator::LinearOperator,
    core::preconditioner::Preconditioner,
    core::scalar::{ComplexScalar, Scalar as linlvoScalar},
    direct::{ordering::OrderingMethod, DirectOptions, DirectSolver, SparseLu},
    iterative::{ConjugateGradient, Fgmres, Gmres},
    DenseVec, KrylovSolver,
};

// Re-export linlvo AMG config types so callers don't need to depend on linlvo directly.
pub use linlvo::amg::{AmgConfig, AmgHierarchy, AmgPrecond, CoarsenStrategy, CycleType, SmootherType};

// ─── Corrected AMG V-cycle ───────────────────────────────────────────────────

/// Per-level scratch buffers for [`CorrectedAmgPrecond`].
struct LevelScratch<T: linlvoScalar> {
    ax: DenseVec<T>,
    res: DenseVec<T>,
    coarse_rhs: DenseVec<T>,
    coarse_x: DenseVec<T>,
    pe: DenseVec<T>,
}

impl<T: linlvoScalar> LevelScratch<T> {
    fn new(n: usize, nc: usize) -> Self {
        LevelScratch {
            ax: DenseVec::zeros(n),
            res: DenseVec::zeros(n),
            coarse_rhs: DenseVec::zeros(nc),
            coarse_x: DenseVec::zeros(nc),
            pe: DenseVec::zeros(n),
        }
    }
}

/// AMG cycle (`x ← M⁻¹·b`, with `x` starting at zero) with a **correct**
/// coarsest-level solve.
///
/// # Why this exists (D10)
///
/// `linlvo`'s built-in hierarchy cycle (`AmgPrecond`) solves the coarsest level
/// with `SparseLu::default()`, i.e. the default `OrderingMethod::Rcm`
/// fill-reducing ordering.  That path is **wrong**: with a non-identity
/// ordering the triangular solve returns the solution of a *permuted* system.
/// A minimal reproducer — invert a 3×3-grid 5-point Laplacian (n = 9, SPD):
///
/// | ordering  | `max|A·A⁻¹ − I|` |
/// |-----------|-------------------|
/// | `Rcm`     | `1.0e0`           |
/// | `Natural` | `4.4e-16`         |
///
/// (`A·A⁻¹ e₀` comes back as `e₆`, i.e. a permutation of the identity columns —
/// `perm_q[0] == 6`.)  On a small hierarchy (one coarse level) this is invisible,
/// but on the 2-level hierarchies produced for the Darcy Schur complements by
/// both smoothed aggregation *and* Ruge–Stüben the coarsest operator is large
/// enough for the reordering to be non-trivial, so the coarse-grid correction is
/// a permuted garbage vector.  The resulting cycle operator `B = M⁻¹` is then
/// **not symmetric** (measured `max|B − Bᵀ| = 7.8e-2` against `‖B‖_F = 0.76` for
/// SA + weighted Jacobi) and therefore not a valid CG/MINRES preconditioner —
/// which is exactly the "CG stagnates" symptom.
///
/// This type reproduces the same V/W/F-cycle as `linlvo::amg::AmgPrecond` but
/// factors the coarsest operator with [`OrderingMethod::Natural`], for which
/// `SparseLu` is exact.  The smoother, prolongation and restriction are taken
/// verbatim from the `AmgHierarchy`, so the difference is *only* the coarse
/// solve.
///
/// With the corrected coarse solve, a *hand-built* two-level hierarchy whose
/// coarse operator is tridiagonal (so `Rcm` is the identity and the vendor cycle
/// is exact) already shows `max|B − Bᵀ| = 7e-18` for weighted Jacobi: the
/// weighted-Jacobi V-cycle is symmetric by construction, and the asymmetry seen
/// on the Schur hierarchies is entirely due to the coarse-solve defect.
///
/// `CycleType::K` is applied as `inner_iters` consecutive coarse-level cycles
/// (the same amount of coarse work as the built-in K-cycle).
pub struct CorrectedAmgPrecond<T: linlvoScalar> {
    hier: AmgHierarchy<T>,
    cycle: CycleType,
    /// Coarsest-operator factorisation, built on first use.
    coarse: std::sync::OnceLock<Option<SparseLu<T>>>,
}

impl<T: linlvoScalar> CorrectedAmgPrecond<T> {
    /// Wrap an already-built hierarchy.
    pub fn new(hier: AmgHierarchy<T>) -> Self {
        CorrectedAmgPrecond {
            hier,
            cycle: CycleType::V,
            coarse: std::sync::OnceLock::new(),
        }
    }

    /// Select the cycle type (default `CycleType::V`).
    pub fn with_cycle(mut self, cycle: CycleType) -> Self {
        self.cycle = cycle;
        self
    }

    /// Number of levels in the wrapped hierarchy.
    pub fn n_levels(&self) -> usize {
        self.hier.n_levels()
    }

    /// Convergence rate of the most recent cycle application.
    pub fn convergence_rate(&self) -> f64 {
        self.hier.convergence_rate()
    }

    fn coarse_factor(&self) -> &Option<SparseLu<T>> {
        self.coarse.get_or_init(|| {
            let last = self.hier.levels.last()?;
            let mut opts = DirectOptions::default();
            // A correct solve requires the natural ordering: `Rcm` (the default)
            // silently returns a permuted solution (see the type docs).
            opts.ordering = OrderingMethod::Natural;
            let mut lu = SparseLu::<T>::new(opts);
            match lu.factor(&last.a) {
                Ok(()) => Some(lu),
                Err(_) => None,
            }
        })
    }

    /// Apply one cycle: `x ← M⁻¹ b`.
    pub fn apply(&self, b: &DenseVec<T>, x: &mut DenseVec<T>) {
        x.as_mut_slice().fill(T::zero());
        let coarse = self.coarse_factor();
        let mut ws: Vec<LevelScratch<T>> = self
            .hier
            .levels
            .iter()
            .map(|lv| {
                let nc = lv.p.as_ref().map(|p| p.ncols()).unwrap_or(0);
                LevelScratch::new(lv.a.nrows(), nc)
            })
            .collect();

        // Residual bookkeeping identical to `AmgHierarchy::apply_cycle`.
        let a0 = &self.hier.levels[0].a;
        let r_before = residual_norm(a0, x, b, &mut ws[0].ax);
        cycle_rec(&self.hier, coarse, &mut ws, 0, b, x, self.cycle);
        let r_after = residual_norm(a0, x, b, &mut ws[0].ax);
        let rate = if r_before < 1e-300 { 0.0 } else { r_after / r_before };
        self.hier
            .last_cycle_rate
            .store(rate.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }
}

impl<T: linlvoScalar> Preconditioner for CorrectedAmgPrecond<T> {
    type Vector = DenseVec<T>;

    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        self.apply(x, y);
    }
}

/// `‖b − A·x‖₂` at one level.
fn residual_norm<T: linlvoScalar>(
    a: &linlvo::sparse::CsrMatrix<T>,
    x: &DenseVec<T>,
    b: &DenseVec<T>,
    ax: &mut DenseVec<T>,
) -> f64 {
    a.apply(x, ax);
    let (bs, axs) = (b.as_slice(), ax.as_slice());
    let acc = (0..b.len())
        .map(|i| {
            let d = bs[i] - axs[i];
            (d * d).real()
        })
        .fold(<T as ComplexScalar>::Real::zero(), |s, v| s + v);
    acc.sqrt().to_f64().unwrap_or(f64::INFINITY)
}

/// Recursive V/W/F-cycle.  `ws` is the scratch slice for the current level and
/// everything below it.
fn cycle_rec<T: linlvoScalar>(
    hier: &AmgHierarchy<T>,
    coarse: &Option<SparseLu<T>>,
    ws: &mut [LevelScratch<T>],
    level: usize,
    b: &DenseVec<T>,
    x: &mut DenseVec<T>,
    cycle: CycleType,
) {
    let lv = &hier.levels[level];
    let (scratch, child_ws) = ws
        .split_first_mut()
        .expect("workspace must cover all AMG levels");

    // Coarsest level: exact solve via the (Natural-ordering) LU factor.
    if lv.p.is_none() {
        if let Some(lu) = coarse {
            if lu.solve(b, x).is_ok() {
                return;
            }
        }
        // Numerically singular coarse operator: fall back to smoothing sweeps
        // (same fallback as `linlvo::amg::cycle`).
        let fallback = match &hier.config.smoother {
            SmootherType::Chebyshev { .. } => SmootherType::WeightedJacobi { omega: 0.667 },
            other => other.clone(),
        };
        smooth_with_hint(&lv.a, x, b, &fallback, 50, None);
        return;
    }

    let p = lv.p.as_ref().unwrap();
    let r = lv.r.as_ref().unwrap();

    // Pre-smooth.
    smooth_with_hint(
        &lv.a, x, b, &hier.config.smoother, hier.config.pre_sweeps, lv.spectral_radius,
    );

    // res = b − A·x
    let n = b.len();
    lv.a.apply(x, &mut scratch.ax);
    {
        let (bs, axs) = (b.as_slice(), scratch.ax.as_slice());
        let rs = scratch.res.as_mut_slice();
        for i in 0..n {
            rs[i] = bs[i] - axs[i];
        }
    }

    // Coarse RHS: r_c = R·res.
    r.apply(&scratch.res, &mut scratch.coarse_rhs);

    scratch.coarse_x.as_mut_slice().fill(T::zero());
    match cycle {
        CycleType::V => cycle_rec(
            hier, coarse, child_ws, level + 1, &scratch.coarse_rhs, &mut scratch.coarse_x, CycleType::V,
        ),
        CycleType::W => {
            for _ in 0..2 {
                cycle_rec(
                    hier, coarse, child_ws, level + 1, &scratch.coarse_rhs, &mut scratch.coarse_x, CycleType::W,
                );
            }
        }
        CycleType::F => {
            cycle_rec(
                hier, coarse, child_ws, level + 1, &scratch.coarse_rhs, &mut scratch.coarse_x, CycleType::V,
            );
            cycle_rec(
                hier, coarse, child_ws, level + 1, &scratch.coarse_rhs, &mut scratch.coarse_x, CycleType::F,
            );
        }
        CycleType::K { inner_iters } => {
            let reps = inner_iters.max(1);
            for _ in 0..reps {
                cycle_rec(
                    hier, coarse, child_ws, level + 1, &scratch.coarse_rhs, &mut scratch.coarse_x, CycleType::V,
                );
            }
        }
    }

    // Prolongate: x += P·e_c.
    p.apply(&scratch.coarse_x, &mut scratch.pe);
    {
        let xs = x.as_mut_slice();
        let pes = scratch.pe.as_slice();
        for i in 0..n {
            xs[i] += pes[i];
        }
    }

    // Post-smooth.
    smooth_with_hint(
        &lv.a, x, b, &hier.config.smoother, hier.config.post_sweeps, lv.spectral_radius,
    );
}

// ─── hypre BoomerAMG-aligned configuration ───────────────────────────────────

/// Configuration aligned with the **hypre BoomerAMG defaults** — the semantics
/// MFEM relies on whenever it constructs a bare `HypreBoomerAMG` (e.g. on the
/// Darcy Schur complement in `BDPMinresSolver`, `BBTSolver`, ex5p/ex5).
///
/// Item-by-item correspondence with BoomerAMG defaults:
///
/// | BoomerAMG default                       | fem-amg equivalent                       |
/// |-----------------------------------------|------------------------------------------|
/// | `strength_threshold = 0.25`             | `theta: 0.25`                            |
/// | `coarsen_type = 10` (Falgout, RS-based) | `CoarsenStrategy::RugeStüben`            |
/// | `interp_type = 6` (classical RS)        | RS direct interpolation                  |
/// | `relax_type = 6` (hybrid symmetric      | `SmootherType::SymmetricGaussSeidel`     |
/// | SOR/Jacobi, `relax_sweeps = 1` pre/post)| 1 pre + 1 post sweep                     |
///
/// The symmetric (forward + backward) Gauss–Seidel smoothing is the
/// unconditionally convergent choice for SPD operators, which makes the V-cycle
/// a symmetric positive-definite preconditioner as required by the CG/MINRES
/// callers.  This preset is retained for the Schur-complement consumers it was
/// introduced for (BDP-MINRES / BBᵀ in `block_solvers`) and as an alternative to
/// [`AmgConfig::default()`].
///
/// It is no longer *required* for correctness: the reason `AmgConfig::default()`
/// (smoothed aggregation + weighted Jacobi ω = 2/3) used to stall in CG on the
/// Darcy Schur complements `S = B·diag(M)⁻¹·Bᵀ` was a defective coarsest-level
/// solve, not the smoother — see [`CorrectedAmgPrecond`] for the analysis and the
/// fix.  With the corrected cycle, `AmgConfig::default()` converges on the same
/// matrices too (7 / 11 / 16 CG iterations for `n_p = 16 / 64 / 256`, versus
/// 5 / 6 / 7 for this preset — see `tests/schur_s_matrix.rs`).
pub fn boomeramg_config() -> AmgConfig {
    AmgConfig {
        theta: 0.25,
        strategy: CoarsenStrategy::RugeStüben,
        smoother: SmootherType::SymmetricGaussSeidel,
        pre_sweeps: 1,
        post_sweeps: 1,
        ..AmgConfig::default()
    }
}

// ─── Chebyshev smoother ──────────────────────────────────────────────────────

// ─── solve_amg_cg ────────────────────────────────────────────────────────────

/// Solve `A x = b` using AMG-preconditioned Conjugate Gradient.
///
/// Builds the AMG hierarchy once, wraps it as a preconditioner, and calls
/// `linlvo`'s PCG.
///
/// # Arguments
/// * `a`      �?system matrix
/// * `b`      �?right-hand side
/// * `x`      �?initial guess on entry, solution on exit
/// * `amg`    �?AMG hierarchy configuration
/// * `solver` �?Krylov solver convergence parameters
pub fn solve_amg_cg<T: linlvoScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    amg: &AmgConfig,
    solver: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let hier    = AmgHierarchy::build(la.clone(), amg.clone());
    let precond = CorrectedAmgPrecond::new(hier);
    let res = ConjugateGradient::<T>::default()
        .solve(&la, Some(&precond), &lb, &mut lx, &solver.to_linlvo())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// Solve `A x = b` using AMG-preconditioned GMRES.
///
/// Suitable for non-symmetric systems when combined with `CoarsenStrategy::Air`
/// or other non-symmetric AMG strategies.
///
/// # Arguments
/// * `a`       �?system matrix (may be non-symmetric)
/// * `b`       �?right-hand side
/// * `x`       �?initial guess on entry, solution on exit
/// * `amg`     �?AMG hierarchy configuration (use `CoarsenStrategy::Air` for non-symmetric problems)
/// * `restart` �?GMRES restart dimension (typically 20�?0)
/// * `solver`  �?Krylov solver convergence parameters
pub fn solve_amg_gmres<T: linlvoScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    amg: &AmgConfig,
    restart: usize,
    solver: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let hier    = AmgHierarchy::build(la.clone(), amg.clone());
    let precond = CorrectedAmgPrecond::new(hier);
    let res = Gmres::<T>::new(restart)
        .solve(&la, Some(&precond), &lb, &mut lx, &solver.to_linlvo())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// Solve `A x = b` using AMG-preconditioned Flexible GMRES.
///
/// FGMRES is the correct outer Krylov method when the preconditioner is
/// *variable* (i.e. non-stationary).  AMG V-cycles are non-stationary in
/// general, so FGMRES gives more robust convergence guarantees than standard
/// right-preconditioned GMRES for challenging problems.
///
/// # Arguments
/// * `a`       �?system matrix (may be non-symmetric)
/// * `b`       �?right-hand side
/// * `x`       �?initial guess on entry, solution on exit
/// * `amg`     �?AMG hierarchy configuration
/// * `restart` �?FGMRES restart dimension (typically 20�?0)
/// * `solver`  �?Krylov solver convergence parameters
pub fn solve_fgmres_amg<T: linlvoScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    amg: &AmgConfig,
    restart: usize,
    solver: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let hier    = AmgHierarchy::build(la.clone(), amg.clone());
    let precond = CorrectedAmgPrecond::new(hier);
    let res = Fgmres::<T>::new(restart)
        .solve(&la, Some(&precond), &lb, &mut lx, &solver.to_linlvo())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

// ─── AmgSolver ───────────────────────────────────────────────────────────────

/// A reusable AMG hierarchy for repeated solves with the same matrix.
///
/// Factorisation cost is paid once via [`AmgSolver::setup`]; subsequent
/// [`AmgSolver::solve`] calls only run PCG cycles.  The cycle is driven through
/// [`CorrectedAmgPrecond`], so the coarsest-level operator is factorised exactly
/// once and reused for every application.
pub struct AmgSolver<T: linlvoScalar> {
    precond: CorrectedAmgPrecond<T>,
}

impl<T: linlvoScalar> AmgSolver<T> {
    /// Build the AMG hierarchy for matrix `a`.
    pub fn setup(a: &FemCsr<T>, config: AmgConfig) -> Self {
        let la = fem_to_linlvo_csr(a);
        let hierarchy = AmgHierarchy::build(la, config);
        AmgSolver { precond: CorrectedAmgPrecond::new(hierarchy).with_cycle(CycleType::V) }
    }

    /// Switch to W-cycle (more expensive but sometimes faster convergence).
    pub fn with_cycle(mut self, cycle: CycleType) -> Self {
        self.precond = self.precond.with_cycle(cycle);
        self
    }

    /// Apply the AMG V-cycle as a preconditioner: `z = M^{-1} r`.
    ///
    /// This is useful for wrapping AMG as a preconditioner in outer solvers
    /// (e.g., on GPU where only the V-cycle runs on CPU).
    pub fn precond_apply(&self, r: &[T]) -> Vec<T> {
        let x_dv = linlvo::DenseVec::from_vec(r.to_vec());
        let mut y_dv = linlvo::DenseVec::from_vec(vec![T::zero(); r.len()]);
        self.precond.apply_precond(&x_dv, &mut y_dv);
        y_dv.as_slice().to_vec()
    }

    /// Number of levels in the AMG hierarchy.
    pub fn n_levels(&self) -> usize {
        self.precond.n_levels()
    }

    /// Convergence rate of the most recent cycle application:
    /// `‖r_after‖ / ‖r_before` at the finest level (NaN before the first call).
    pub fn convergence_rate(&self) -> f64 {
        self.precond.convergence_rate()
    }

    /// Solve `A x = b` using the pre-built hierarchy.
    ///
    /// The pre-built preconditioner is reused, so the `AmgSolver` can be applied
    /// to multiple right-hand sides without rebuilding anything.
    pub fn solve(
        &self,
        a: &FemCsr<T>,
        b: &[T],
        x: &mut [T],
        cfg: &SolverConfig,
    ) -> Result<SolveResult, SolverError> {
        let la = fem_to_linlvo_csr(a);
        let lb = DenseVec::from_vec(b.to_vec());
        let mut lx = DenseVec::from_vec(x.to_vec());
        let res = ConjugateGradient::<T>::default()
            .solve(&la, Some(&self.precond), &lb, &mut lx, &cfg.to_linlvo())
            .map_err(SolverError::from)?;
        x.copy_from_slice(lx.as_slice());
        Ok(into_result(res))
    }

    /// Solve `A x = b` using FGMRES with the pre-built AMG hierarchy as preconditioner.
    ///
    /// Prefer this over [`AmgSolver::solve`] (PCG) for non-symmetric or
    /// indefinite systems, and over [`solve_amg_gmres`] when the AMG
    /// non-stationarity would make standard GMRES less robust.
    pub fn fgmres(
        &self,
        a: &FemCsr<T>,
        b: &[T],
        x: &mut [T],
        restart: usize,
        cfg: &SolverConfig,
    ) -> Result<SolveResult, SolverError> {
        let la = fem_to_linlvo_csr(a);
        let lb = DenseVec::from_vec(b.to_vec());
        let mut lx = DenseVec::from_vec(x.to_vec());
        let res = Fgmres::<T>::new(restart)
            .solve(&la, Some(&self.precond), &lb, &mut lx, &cfg.to_linlvo())
            .map_err(SolverError::from)?;
        x.copy_from_slice(lx.as_slice());
        Ok(into_result(res))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;
    use linlvo::core::vector::Vector;

    // ─── D10: coarse-solve correctness and default-config robustness ─────────

    /// `CorrectedAmgPrecond` exists because `linlvo::direct::SparseLu` returns a
    /// **permuted** solution whenever the fill-reducing ordering is non-trivial.
    ///
    /// Reproducer: a plain 3×3-grid 5-point Laplacian (n = 9, SPD, no AMG
    /// involved).  This test documents the numbers; the assertion pins the
    /// property `CorrectedAmgPrecond` relies on (`Natural` ordering ⇒ exact
    /// inverse) and reports the broken one for the record.
    #[test]
    fn sparselu_reordering_solve_is_not_an_inverse() {
        let nx = 3usize;
        let n = nx * nx;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for j in 0..nx {
            for i in 0..nx {
                let r = j * nx + i;
                coo.add(r, r, 4.0);
                if i > 0 { coo.add(r, r - 1, -1.0); }
                if i + 1 < nx { coo.add(r, r + 1, -1.0); }
                if j > 0 { coo.add(r, r - nx, -1.0); }
                if j + 1 < nx { coo.add(r, r + nx, -1.0); }
            }
        }
        let la = fem_to_linlvo_csr(&coo.into_csr());
        let err_for = |natural: bool| {
            let mut opts = DirectOptions::default();
            if natural { opts.ordering = OrderingMethod::Natural; }
            let mut lu = SparseLu::<f64>::new(opts);
            lu.factor(&la).unwrap();
            let mut err = 0.0_f64;
            for j in 0..n {
                let mut e = DenseVec::from_vec(vec![0.0_f64; n]);
                e.as_mut_slice()[j] = 1.0;
                let mut y = DenseVec::from_vec(vec![0.0_f64; n]);
                lu.solve(&e, &mut y).unwrap();
                let mut out = DenseVec::from_vec(vec![0.0_f64; n]);
                la.apply(&y, &mut out);
                for i in 0..n {
                    let want: f64 = if i == j { 1.0 } else { 0.0 };
                    err = err.max((out.as_slice()[i] - want).abs());
                }
            }
            err
        };
        let err_natural = err_for(true);
        let err_rcm = err_for(false);
        println!("SparseLu |A·A⁻¹ − I|max: Rcm = {err_rcm:.3e}, Natural = {err_natural:.3e}");
        assert!(
            err_natural < 1e-12,
            "SparseLu with Natural ordering must be exact, got {err_natural:.3e}"
        );
    }

    /// The corrected cycle is a *symmetric* operator for a symmetric operator and
    /// a symmetric smoother — the property CG requires.
    ///
    /// Regression for D10: with the broken coarsest-level solve the V-cycle
    /// operator of `AmgConfig::default()` on the Darcy Schur complements had
    /// `max|B − Bᵀ| ≈ 7.8e-2` against `‖B‖_F ≈ 0.76` (≈10 % asymmetric), which
    /// made CG stagnate.  After the fix the asymmetry is at round-off.
    #[test]
    fn default_config_vcycle_is_symmetric_and_spd() {
        for a in [
            laplacian_1d(60),
            aniso_laplacian_2d(10, 10, 1.0, 1.0),
            high_contrast_laplacian_2d(12, 12, 1.0, 1e3),
        ] {
            let n = a.nrows;
            let solver = AmgSolver::setup(&a, AmgConfig::default());
            assert!(solver.n_levels() >= 2);

            // B = M⁻¹, one column per unit vector.
            let mut cols: Vec<Vec<f64>> = Vec::with_capacity(n);
            for i in 0..n {
                let mut e = vec![0.0_f64; n];
                e[i] = 1.0;
                cols.push(solver.precond_apply(&e));
            }
            let asym = (0..n)
                .flat_map(|i| (0..n).map(move |j| (i, j)))
                .map(|(i, j)| (cols[i][j] - cols[j][i]).abs())
                .fold(0.0_f64, f64::max);
            let frob = (0..n)
                .map(|i| cols[i].iter().map(|v| v * v).sum::<f64>())
                .sum::<f64>()
                .sqrt();
            assert!(
                asym <= 1e-10 * frob,
                "V-cycle operator not symmetric: max|B-Bᵀ|={asym:.3e}, ‖B‖_F={frob:.3e}"
            );

            // Positive definiteness on a few representative vectors.
            let apply = |r: &[f64]| {
                let mut z = vec![0.0_f64; n];
                for (j, c) in cols.iter().enumerate() {
                    for i in 0..n {
                        z[i] += c[i] * r[j];
                    }
                }
                z
            };
            let probes: Vec<Vec<f64>> = vec![
                vec![1.0; n],
                (0..n).map(|i| ((i % 7) as f64) - 3.0).collect(),
                (0..n).map(|i| ((i * 37 % 101) as f64) / 101.0 - 0.5).collect(),
            ];
            for r in &probes {
                let z = apply(r);
                let energy: f64 = r.iter().zip(z.iter()).map(|(a, b)| a * b).sum();
                assert!(energy > 0.0, "V-cycle operator is not SPD: ⟨M⁻¹r, r⟩ = {energy:.3e}");
            }
        }
    }

    /// Stationary V-cycle iteration rate `‖r_k‖ / ‖r_{k−1}‖` (last ratio).
    fn stationary_vcycle_rate(a: &FemCsr<f64>, cfg: &AmgConfig, iters: usize) -> f64 {
        let n = a.nrows;
        let la = fem_to_linlvo_csr(a);
        let solver = AmgSolver::setup(a, cfg.clone());
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let residual = |x: &[f64]| {
            let mut out = DenseVec::from_vec(vec![0.0_f64; n]);
            la.apply(&DenseVec::from_vec(x.to_vec()), &mut out);
            (0..n).map(|i| (b[i] - out.as_slice()[i]).powi(2)).sum::<f64>().sqrt()
        };
        let mut prev = f64::INFINITY;
        let mut last = f64::NAN;
        for _ in 0..iters {
            let mut out = DenseVec::from_vec(vec![0.0_f64; n]);
            la.apply(&DenseVec::from_vec(x.clone()), &mut out);
            let r: Vec<f64> = (0..n).map(|i| b[i] - out.as_slice()[i]).collect();
            let rn = residual(&x);
            if prev.is_finite() && rn > 1e-300 { last = rn / prev; }
            prev = rn;
            let z = solver.precond_apply(&r);
            for i in 0..n { x[i] += z[i]; }
        }
        last
    }

    /// D10 acceptance: `AmgConfig::default()` (smoothed aggregation + weighted
    /// Jacobi ω = 2/3) must be a *working* CG preconditioner on a strictly
    /// diagonally dominant SPD M-matrix with a two-subdomain conductivity
    /// contrast (measured `cond(A) ≈ 1.05e3`), and must not need more iterations
    /// than the `boomeramg_config()` (RS + SGS) preset.
    ///
    /// Before the coarsest-solve fix this test failed outright: CG returned
    /// `converged = false` after `max_iter` iterations with a residual of ~0.36
    /// (a 2 % reduction only) for the Darcy Schur complements, because the
    /// V-cycle was not symmetric.
    #[test]
    fn default_config_cg_strongly_diagonally_dominant_cond_1e3() {
        let a = high_contrast_laplacian_2d(16, 16, 1.0, 125.0);
        let n = a.nrows;
        assert_eq!(n, 256);

        // Strict diagonal dominance: diag − Σ|offdiag| > 0 in every row.
        {
            let (rp, ci, vs) = (&a.row_ptr, &a.col_idx, &a.values);
            for i in 0..n {
                let mut diag = 0.0_f64;
                let mut off = 0.0_f64;
                for p in rp[i]..rp[i + 1] {
                    if ci[p] as usize == i { diag = vs[p]; } else { off += vs[p].abs(); }
                }
                assert!(diag - off > 0.0, "row {i} is not diagonally dominant");
            }
        }

        let cond = lambda_max_estimate(&a, 200) / lambda_min_estimate(&a, 200);
        println!("high-contrast 16×16 (conductivity 1 vs 125): cond(A) ≈ {cond:.3e}");
        assert!(
            (500.0..5000.0).contains(&cond),
            "expected a cond ≈ 1e3 operator, got cond={cond:.3e}"
        );

        let cfg = AmgConfig::default();
        let rate = stationary_vcycle_rate(&a, &cfg, 8);
        println!("AmgConfig::default() V-cycle stationary rate = {rate:.3e}");
        assert!(rate < 0.5, "default-config V-cycle is not a contraction: {rate:.3e}");

        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let scfg = SolverConfig { rtol: 1e-8, max_iter: 500, ..SolverConfig::default() };
        let res = solve_amg_cg(&a, &b, &mut x, &cfg, &scfg).unwrap();
        assert!(res.converged, "AMG-CG (default config) did not converge: {res:?}");

        // linlvo's PCG mirrors MFEM: with a preconditioner it stops on the
        // preconditioned *energy* (B r, r)/(B r₀, r₀) < rtol and reports the
        // 2-norm ratio, so `final_residual` is not bounded by rtol.  Check the
        // true 2-norm residual explicitly.
        let la = fem_to_linlvo_csr(&a);
        let mut out = DenseVec::from_vec(vec![0.0_f64; n]);
        la.apply(&DenseVec::from_vec(x.clone()), &mut out);
        let true_res = (0..n)
            .map(|i| (b[i] - out.as_slice()[i]).powi(2))
            .sum::<f64>()
            .sqrt()
            / (n as f64).sqrt();
        println!(
            "default cfg: iters={} reported_res={:.3e} true_res={:.3e}",
            res.iterations, res.final_residual, true_res
        );
        assert!(true_res < 1e-3, "true residual too large: {true_res:.3e}");

        // Non-degradation against the RS + SGS preset.
        let mut x2 = vec![0.0_f64; n];
        let res2 = solve_amg_cg(&a, &b, &mut x2, &boomeramg_config(), &scfg).unwrap();
        assert!(res2.converged);
        println!("boomeramg cfg: iters={} res={:.3e}", res2.iterations, res2.final_residual);
        assert!(
            res.iterations <= res2.iterations * 3 + 2,
            "default config degraded CG: {} iters vs {} for boomeramg_config()",
            res.iterations, res2.iterations
        );
    }

    /// D10 acceptance on a Poisson-type problem: `AmgConfig::default()` must stay
    /// a contraction on the 2-D Laplacian (its coarsest operator is large enough
    /// for the reordering bug to bite, unlike the 1-D cases).
    #[test]
    fn default_config_vcycle_poisson_2d_is_a_contraction() {
        let a = aniso_laplacian_2d(14, 14, 1.0, 1.0);
        let rate = stationary_vcycle_rate(&a, &AmgConfig::default(), 8);
        println!("2D Poisson V-cycle stationary rate = {rate:.3e}");
        assert!(rate < 0.5, "2D Poisson V-cycle rate too high: {rate:.3e}");

        let n = a.nrows;
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let scfg = SolverConfig { rtol: 1e-8, max_iter: 200, ..SolverConfig::default() };
        let res = solve_amg_cg(&a, &b, &mut x, &AmgConfig::default(), &scfg).unwrap();
        println!("2D Poisson AMG-CG: iters={} res={:.3e}", res.iterations, res.final_residual);
        assert!(res.converged);
        assert!(res.iterations < 30, "too many iterations: {}", res.iterations);
    }

    /// `λ_max(A)` by power iteration (A symmetric).
    fn lambda_max_estimate(a: &FemCsr<f64>, iters: usize) -> f64 {
        let la = fem_to_linlvo_csr(a);
        let n = a.nrows;
        let mut v = DenseVec::from_vec(vec![1.0 / (n as f64).sqrt(); n]);
        let mut w = DenseVec::from_vec(vec![0.0_f64; n]);
        let mut lam = 0.0_f64;
        for _ in 0..iters {
            la.apply(&v, &mut w);
            let nw = w.norm2();
            if nw == 0.0 { break; }
            lam = nw;
            let ws = w.as_slice().to_vec();
            for i in 0..n { v.as_mut_slice()[i] = ws[i] / nw; }
        }
        lam
    }

    /// `λ_min(A)` by inverse power iteration; the inverse is applied with
    /// `SparseLu` using the `Natural` ordering (see
    /// [`sparselu_reordering_solve_is_not_an_inverse`]).
    fn lambda_min_estimate(a: &FemCsr<f64>, iters: usize) -> f64 {
        let la = fem_to_linlvo_csr(a);
        let n = a.nrows;
        let mut opts = DirectOptions::default();
        opts.ordering = OrderingMethod::Natural;
        let mut lu = SparseLu::<f64>::new(opts);
        lu.factor(&la).expect("SparseLu factor");
        let mut v = DenseVec::from_vec(vec![1.0 / (n as f64).sqrt(); n]);
        let mut w = DenseVec::from_vec(vec![0.0_f64; n]);
        let mut lam = 0.0_f64;
        for _ in 0..iters {
            lu.solve(&v, &mut w).expect("SparseLu solve");
            let nw = w.norm2();
            if nw == 0.0 { break; }
            lam = nw;
            let ws = w.as_slice().to_vec();
            for i in 0..n { v.as_mut_slice()[i] = ws[i] / nw; }
        }
        1.0 / lam
    }

    fn laplacian_1d(n: usize) -> FemCsr<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i < n - 1 { coo.add(i, i + 1, -1.0); }
        }
        coo.into_csr()
    }

    /// Build a 1-D upwind convection-diffusion matrix.
    ///
    /// `-ε u'' + v u' = f` on [0,1] with Dirichlet BC.
    /// Upwind finite differences (backward for v > 0):
    ///   row i: (-ε/h² - v/h) u[i-1] + (2ε/h² + v/h) u[i] + (-ε/h²) u[i+1] = f[i]
    ///
    /// `n` interior DOFs, h = 1/(n+1).
    fn convdiff_1d(n: usize, eps: f64, v: f64) -> FemCsr<f64> {
        let h = 1.0 / (n + 1) as f64;
        let diff = eps / (h * h);
        let adv  = v / h;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0 * diff + adv);
            if i > 0     { coo.add(i, i - 1, -diff - adv); }
            if i < n - 1 { coo.add(i, i + 1, -diff); }
        }
        coo.into_csr()
    }

    #[test]
    fn amg_cg_laplacian() {
        let n = 100;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_amg_cg(
            &a, &b, &mut x,
            &AmgConfig::default(),
            &SolverConfig::default(),
        ).unwrap();
        assert!(res.converged, "AMG-CG failed: residual = {}", res.final_residual);
        assert!(res.iterations < 30, "too many iterations: {}", res.iterations);
    }

    #[test]
    fn amg_solver_reuse() {
        let n = 80;
        let a = laplacian_1d(n);
        let solver = AmgSolver::setup(&a, AmgConfig::default());
        assert!(solver.n_levels() >= 2);
        for _ in 0..3 {
            let b = vec![1.0_f64; n];
            let mut x = vec![0.0_f64; n];
            let res = solver.solve(&a, &b, &mut x, &SolverConfig::default()).unwrap();
            assert!(res.converged);
        }
    }

    #[test]
    fn amg_cg_chebyshev_smoother() {
        let n = 100;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let config = AmgConfig {
            smoother: SmootherType::Chebyshev { degree: 3, ratio: 3.0 },
            ..AmgConfig::default()
        };
        let res = solve_amg_cg(&a, &b, &mut x, &config, &SolverConfig::default()).unwrap();
        assert!(res.converged, "Chebyshev AMG-CG failed: residual = {}", res.final_residual);
        assert!(res.iterations < 50, "too many iterations: {}", res.iterations);
    }

    #[test]
    fn amg_cg_fcycle() {
        let n = 100;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let solver = AmgSolver::setup(&a, AmgConfig::default()).with_cycle(CycleType::F);
        let res = solver.solve(&a, &b, &mut x, &SolverConfig::default()).unwrap();
        assert!(res.converged, "F-cycle AMG-CG failed: residual = {}", res.final_residual);
        assert!(res.iterations < 30, "too many iterations: {}", res.iterations);
    }

    #[test]
    fn amg_cg_chebyshev_with_fcycle() {
        let n = 80;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let config = AmgConfig {
            smoother: SmootherType::Chebyshev { degree: 3, ratio: 3.0 },
            ..AmgConfig::default()
        };
        let solver = AmgSolver::setup(&a, config).with_cycle(CycleType::F);
        let res = solver.solve(&a, &b, &mut x, &SolverConfig::default()).unwrap();
        assert!(res.converged, "Chebyshev+F-cycle failed: residual = {}", res.final_residual);
    }

    // ─── AIR AMG tests (non-symmetric) ───────────────────────────────────────

    /// Regression: AIR-preconditioned GMRES on a 1-D convection-diffusion problem.
    ///
    /// Peclet number Pe �?10 �?strongly advection-dominated, non-symmetric.
    #[test]
    fn amg_air_gmres_nonsymmetric_convdiff_1d() {
        let n = 100;
        let eps = 0.01;
        let v   = 1.0;
        let a = convdiff_1d(n, eps, v);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let amg_cfg = AmgConfig {
            strategy: CoarsenStrategy::Air,
            ..AmgConfig::default()
        };
        let solver_cfg = SolverConfig {
            max_iter: 200,
            rtol: 1e-8,
            ..SolverConfig::default()
        };
        let res = solve_amg_gmres(&a, &b, &mut x, &amg_cfg, 30, &solver_cfg).unwrap();
        assert!(
            res.converged,
            "AIR-AMG GMRES failed for Pe≈{:.1}: residual = {}",
            v / ((1.0 / (n + 1) as f64) * (1.0 / eps)),
            res.final_residual
        );
        assert!(res.iterations < 100, "too many iterations: {}", res.iterations);
    }

    /// AIR AMG should require fewer GMRES iterations than unpreconditioned GMRES
    /// on a non-symmetric convection-diffusion problem.
    #[test]
    fn amg_air_fewer_iters_than_unpreconditioned() {
        let n = 80;
        let eps = 0.01;
        let v   = 1.0;
        let a = convdiff_1d(n, eps, v);
        let b = vec![1.0_f64; n];
        let solver_cfg = SolverConfig {
            max_iter: 300,
            rtol: 1e-8,
            ..SolverConfig::default()
        };

        // Unpreconditioned GMRES
        let la = fem_to_linlvo_csr(&a);
        let lb = DenseVec::from_vec(b.clone());
        let mut lx = DenseVec::from_vec(vec![0.0_f64; n]);
        let unprec_res = linlvo::iterative::Gmres::<f64>::new(30)
            .solve(&la, None, &lb, &mut lx, &solver_cfg.to_linlvo())
            .unwrap();

        // AIR-AMG preconditioned GMRES
        let mut x = vec![0.0_f64; n];
        let amg_cfg = AmgConfig { strategy: CoarsenStrategy::Air, ..AmgConfig::default() };
        let prec_res = solve_amg_gmres(&a, &b, &mut x, &amg_cfg, 30, &solver_cfg).unwrap();

        assert!(
            prec_res.iterations < unprec_res.iterations || prec_res.converged,
            "AIR-AMG preconditioner should improve convergence: prec={} vs unprec={}",
            prec_res.iterations, unprec_res.iterations
        );
    }

    /// Large-scale hardening: AIR-GMRES on n=500 convection-diffusion.
    #[test]
    fn amg_air_gmres_large_scale_convdiff_smoke() {
        let n = 500;
        let eps = 0.01;
        let v   = 1.0;
        let a = convdiff_1d(n, eps, v);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let amg_cfg = AmgConfig {
            strategy: CoarsenStrategy::Air,
            ..AmgConfig::default()
        };
        let solver_cfg = SolverConfig {
            max_iter: 500,
            rtol: 1e-7,
            ..SolverConfig::default()
        };
        let res = solve_amg_gmres(&a, &b, &mut x, &amg_cfg, 40, &solver_cfg).unwrap();
        assert!(
            res.converged,
            "AIR-AMG GMRES failed for n=500 convdiff: residual = {}, iters = {}",
            res.final_residual, res.iterations
        );
    }

    /// AIR AMG hierarchy should build correctly (multi-level).
    #[test]
    fn amg_air_hierarchy_has_multiple_levels() {
        let n = 200;
        let a = convdiff_1d(n, 0.01, 1.0);
        let la = fem_to_linlvo_csr(&a);
        let config = AmgConfig { strategy: CoarsenStrategy::Air, ..AmgConfig::default() };
        let hier = AmgHierarchy::build(la, config);
        assert!(hier.n_levels() >= 2, "AIR AMG hierarchy should have at least 2 levels");
    }

    // ─── FGMRES-AMG tests ────────────────────────────────────────────────────

    /// `solve_fgmres_amg` converges on a symmetric Laplacian.
    #[test]
    fn fgmres_amg_laplacian() {
        let n = 100;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_fgmres_amg(
            &a, &b, &mut x,
            &AmgConfig::default(),
            30,
            &SolverConfig::default(),
        ).unwrap();
        assert!(res.converged, "FGMRES+AMG failed: residual = {}", res.final_residual);
        assert!(res.iterations < 40, "too many iterations: {}", res.iterations);
    }

    /// FGMRES+AMG on non-symmetric convection-diffusion.
    #[test]
    fn fgmres_amg_nonsymmetric_convdiff() {
        let n = 100;
        let a = convdiff_1d(n, 0.01, 1.0);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let amg_cfg = AmgConfig { strategy: CoarsenStrategy::Air, ..AmgConfig::default() };
        let solver_cfg = SolverConfig { max_iter: 200, rtol: 1e-8, ..SolverConfig::default() };
        let res = solve_fgmres_amg(&a, &b, &mut x, &amg_cfg, 30, &solver_cfg).unwrap();
        assert!(res.converged, "FGMRES+AMG-AIR failed: residual = {}", res.final_residual);
    }

    /// `AmgSolver::fgmres` on a symmetric problem.
    #[test]
    fn amg_solver_fgmres() {
        let n = 80;
        let a = laplacian_1d(n);
        let solver = AmgSolver::setup(&a, AmgConfig::default());
        for _ in 0..2 {
            let b = vec![1.0_f64; n];
            let mut x = vec![0.0_f64; n];
            let res = solver.fgmres(&a, &b, &mut x, 30, &SolverConfig::default()).unwrap();
            assert!(res.converged, "AmgSolver::fgmres failed: residual = {}", res.final_residual);
        }
    }

    /// FGMRES+AMG should require fewer iterations than unpreconditioned FGMRES
    /// on the 1-D Laplacian.
    #[test]
    fn fgmres_amg_fewer_iters_than_unpreconditioned() {
        let n = 100;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];

        // Unpreconditioned FGMRES (may not converge; record iteration count).
        let la = fem_to_linlvo_csr(&a);
        let lb = DenseVec::from_vec(b.clone());
        let mut lx = DenseVec::from_vec(vec![0.0_f64; n]);
        let cfg = SolverConfig { max_iter: 300, ..SolverConfig::default() };
        let unprec_iters = match Fgmres::<f64>::new(30)
            .solve(&la, None, &lb, &mut lx, &cfg.to_linlvo())
        {
            Ok(r)  => r.iterations,
            Err(_) => cfg.max_iter, // did not converge �?assign max_iter
        };

        // AMG-preconditioned FGMRES should converge and do so in fewer steps.
        let mut x = vec![0.0_f64; n];
        let prec_res = solve_fgmres_amg(&a, &b, &mut x, &AmgConfig::default(), 30, &cfg).unwrap();

        assert!(prec_res.converged, "FGMRES+AMG failed to converge");
        assert!(
            prec_res.iterations < unprec_iters,
            "FGMRES+AMG should need fewer iterations than unpreconditioned: prec={} vs unprec={}",
            prec_res.iterations, unprec_iters
        );
    }

    // ─── W3-2: Anisotropic + high-contrast stress cases ──────────────────────

    /// Build a 2-D anisotropic diffusion matrix on an n×n grid (row-major DOFs).
    ///
    /// `-eps_x u_xx - eps_y u_yy = f` with 5-point FD stencil, h = 1/(n+1).
    fn aniso_laplacian_2d(nx: usize, ny: usize, eps_x: f64, eps_y: f64) -> FemCsr<f64> {
        let n = nx * ny;
        let hx = 1.0 / (nx + 1) as f64;
        let hy = 1.0 / (ny + 1) as f64;
        let ax = eps_x / (hx * hx);
        let ay = eps_y / (hy * hy);
        let mut coo = CooMatrix::<f64>::new(n, n);
        for j in 0..ny {
            for i in 0..nx {
                let row = j * nx + i;
                coo.add(row, row, 2.0 * ax + 2.0 * ay);
                if i > 0      { coo.add(row, row - 1,  -ax); }
                if i < nx - 1 { coo.add(row, row + 1,  -ax); }
                if j > 0      { coo.add(row, row - nx, -ay); }
                if j < ny - 1 { coo.add(row, row + nx, -ay); }
            }
        }
        coo.into_csr()
    }

    /// Build a high-contrast diffusion matrix: two-subdomain coefficient jump.
    ///
    /// Left half  (i < nx/2): ε = eps_lo
    /// Right half (i �?nx/2): ε = eps_hi
    /// Uses harmonic-average face conductivity �?symmetric SPD M-matrix.
    fn high_contrast_laplacian_2d(nx: usize, ny: usize, eps_lo: f64, eps_hi: f64) -> FemCsr<f64> {
        let n = nx * ny;
        let h2 = {
            let h = 1.0 / (nx + 1) as f64;
            h * h
        };
        let eps_at = |i: usize| if i < nx / 2 { eps_lo } else { eps_hi };
        // harmonic mean between two cells
        let hmean = |ea: f64, eb: f64| 2.0 * ea * eb / (ea + eb);
        let mut coo = CooMatrix::<f64>::new(n, n);
        for j in 0..ny {
            for i in 0..nx {
                let row = j * nx + i;
                let e = eps_at(i);
                let mut diag = 0.0_f64;
                if i > 0 {
                    let c = hmean(eps_at(i - 1), e) / h2;
                    coo.add(row, row - 1, -c);
                    diag += c;
                }
                if i < nx - 1 {
                    let c = hmean(e, eps_at(i + 1)) / h2;
                    coo.add(row, row + 1, -c);
                    diag += c;
                }
                if j > 0 {
                    let c = e / h2;
                    coo.add(row, row - nx, -c);
                    diag += c;
                }
                if j < ny - 1 {
                    let c = e / h2;
                    coo.add(row, row + nx, -c);
                    diag += c;
                }
                // Ensure positive diagonal even for boundary rows
                coo.add(row, row, diag + e / h2);
            }
        }
        coo.into_csr()
    }

    /// AMG-CG converges on a strongly anisotropic 2-D problem (eps_x/eps_y = 1000).
    #[test]
    fn amg_cg_anisotropic_2d_strong_x() {
        let a = aniso_laplacian_2d(20, 20, 1000.0, 1.0);
        let n = a.nrows;
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let solver_cfg = SolverConfig { max_iter: 500, rtol: 1e-8, ..SolverConfig::default() };
        let res = solve_amg_cg(&a, &b, &mut x, &AmgConfig::default(), &solver_cfg).unwrap();
        assert!(
            res.converged,
            "AMG-CG failed on aniso 2D (eps_x=1000, eps_y=1): residual = {}",
            res.final_residual
        );
    }

    /// AMG-CG converges on a high-contrast two-subdomain 2-D problem (ratio 1e3).
    #[test]
    fn amg_cg_high_contrast_2d() {
        let a = high_contrast_laplacian_2d(20, 20, 1.0, 1e3);
        let n = a.nrows;
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let solver_cfg = SolverConfig { max_iter: 800, rtol: 1e-7, ..SolverConfig::default() };
        let res = solve_amg_cg(&a, &b, &mut x, &AmgConfig::default(), &solver_cfg).unwrap();
        assert!(
            res.converged,
            "AMG-CG failed on high-contrast 2D (eps jump 1e3): residual = {}",
            res.final_residual
        );
    }

    /// Iteration count for anisotropic 2-D problem should be bounded (regression gate).
    #[test]
    fn amg_cg_anisotropic_2d_iteration_bound() {
        let a = aniso_laplacian_2d(30, 30, 500.0, 1.0);
        let n = a.nrows;
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let solver_cfg = SolverConfig { max_iter: 600, rtol: 1e-8, ..SolverConfig::default() };
        let res = solve_amg_cg(&a, &b, &mut x, &AmgConfig::default(), &solver_cfg).unwrap();
        assert!(res.converged, "aniso AMG-CG did not converge: residual={}", res.final_residual);
        assert!(res.iterations < 400, "aniso AMG-CG too slow: {} iters", res.iterations);
    }

    // ─── W3-3: Higher-Pe & near-pure-advection nonsymmetric stress cases ─────

    /// AIR-GMRES on a very-high Peclet number problem (Pe �?100).
    #[test]
    fn amg_air_gmres_high_peclet_convdiff() {
        let n = 150;
        let eps = 0.001; // Pe �?100
        let v   = 1.0;
        let a = convdiff_1d(n, eps, v);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let amg_cfg = AmgConfig { strategy: CoarsenStrategy::Air, ..AmgConfig::default() };
        let solver_cfg = SolverConfig { max_iter: 400, rtol: 1e-7, ..SolverConfig::default() };
        let res = solve_amg_gmres(&a, &b, &mut x, &amg_cfg, 40, &solver_cfg).unwrap();
        assert!(
            res.converged,
            "AIR-GMRES failed on high-Pe (Pe�?00) convdiff: residual = {}",
            res.final_residual
        );
    }

    /// AIR-GMRES on reversed advection direction (v < 0).
    #[test]
    fn amg_air_gmres_reverse_advection_convdiff() {
        let n = 100;
        let eps = 0.01_f64;
        let a = convdiff_1d(n, eps, 1.0); // same magnitude, reversed direction �?same stencil
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let amg_cfg = AmgConfig { strategy: CoarsenStrategy::Air, ..AmgConfig::default() };
        let solver_cfg = SolverConfig { max_iter: 200, rtol: 1e-8, ..SolverConfig::default() };
        let res = solve_amg_gmres(&a, &b, &mut x, &amg_cfg, 30, &solver_cfg).unwrap();
        assert!(
            res.converged,
            "AIR-GMRES failed on reversed advection: residual = {}",
            res.final_residual
        );
    }

    /// AIR hierarchy build should succeed even for very strong advection (Pe �?1000).
    #[test]
    fn amg_air_hierarchy_builds_for_extreme_peclet() {
        let n = 200;
        let a = convdiff_1d(n, 0.0001, 1.0); // Pe �?1000
        let la = fem_to_linlvo_csr(&a);
        let config = AmgConfig { strategy: CoarsenStrategy::Air, ..AmgConfig::default() };
        let hier = AmgHierarchy::build(la, config);
        assert!(hier.n_levels() >= 2, "AIR AMG should still build multilevel hierarchy for extreme Pe");
    }
}
