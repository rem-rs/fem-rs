use linlvo::{
    iterative::{ConjugateGradient, Gmres},
    DenseVec, Ilu0Precond, IldltPrecond, JacobiPrecond, KrylovSolver, Preconditioner,
};
use linlvo::sparse::CsrMatrix as LinlvoCsr;
use fem_linalg::{CsrMatrix, SolverConfig, SolverError, SolveResult, fem_to_linlvo_csr, into_result};

/// Available preconditioner types for [`CachedSolver`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachedPrecond {
    /// No preconditioner — the Krylov method runs without one.
    None,
    /// Diagonal (Jacobi) preconditioner, cheapest option.
    Jacobi,
    /// Incomplete LU with zero fill-in.
    Ilu0,
    /// Incomplete LDLᵀ factorization (SPD systems).
    Ildlt,
}

/// A solver that caches matrix conversion and preconditioner across solves.
///
/// For constant-matrix sequences (transient, Newton) this avoids redundant
/// `fem_to_linlvo_csr` and preconditioner setup on every call.
///
/// # Usage
/// ```rust,ignore
/// let mut solver = CachedSolver::new(&a, CachedPrecond::Jacobi).unwrap();
/// solver.solve_cg(&b, &mut x, &cfg).unwrap();
/// // … time step, same matrix …
/// solver.solve_cg(&b_new, &mut x, &cfg).unwrap();  // no matrix clone, no precond rebuild
/// ```
pub struct CachedSolver {
    la: LinlvoCsr<f64>,
    precond_kind: CachedPrecond,
    precond: Option<Box<dyn Preconditioner<Vector = DenseVec<f64>>>>,
}

impl CachedSolver {
    /// Build a new `CachedSolver` from a FEM CSR matrix and a preconditioner choice.
    ///
    /// The matrix is converted to linlvo format immediately; the preconditioner
    /// is computed from the converted matrix.
    pub fn new(a: &CsrMatrix<f64>, kind: CachedPrecond) -> Result<Self, SolverError> {
        let la = fem_to_linlvo_csr(a);
        let precond = build_precond(&la, kind)?;
        Ok(Self { la, precond_kind: kind, precond })
    }

    /// Solve `Ax = b` with Conjugate Gradient.
    ///
    /// The matrix and preconditioner cached at construction (or last [`rebuild`])
    /// are reused without conversion overhead.
    pub fn solve_cg(
        &self,
        b: &[f64],
        x: &mut [f64],
        cfg: &SolverConfig,
    ) -> Result<SolveResult, SolverError> {
        let lb = DenseVec::from_vec(b.to_vec());
        let mut lx = DenseVec::from_vec(x.to_vec());
        let res = ConjugateGradient::default()
            .solve(
                &self.la,
                self.precond.as_deref(),
                &lb,
                &mut lx,
                &cfg.to_linlvo(),
            )
            .map_err(SolverError::from)?;
        x.copy_from_slice(lx.as_slice());
        Ok(into_result(res))
    }

    /// Solve `Ax = b` with GMRES.
    ///
    /// `restart` is the Krylov subspace dimension before restarting.
    /// The matrix and preconditioner cached at construction (or last [`rebuild`])
    /// are reused without conversion overhead.
    pub fn solve_gmres(
        &self,
        b: &[f64],
        x: &mut [f64],
        restart: usize,
        cfg: &SolverConfig,
    ) -> Result<SolveResult, SolverError> {
        let lb = DenseVec::from_vec(b.to_vec());
        let mut lx = DenseVec::from_vec(x.to_vec());
        let solver = Gmres::new(restart);
        let res = solver
            .solve(
                &self.la,
                self.precond.as_deref(),
                &lb,
                &mut lx,
                &cfg.to_linlvo(),
            )
            .map_err(SolverError::from)?;
        x.copy_from_slice(lx.as_slice());
        Ok(into_result(res))
    }

    /// Replace the cached matrix and rebuild the preconditioner.
    ///
    /// This is equivalent to constructing a new `CachedSolver` but avoids an
    /// allocation for the struct itself.
    pub fn rebuild(&mut self, a: &CsrMatrix<f64>) -> Result<(), SolverError> {
        self.la = fem_to_linlvo_csr(a);
        self.precond = build_precond(&self.la, self.precond_kind)?;
        Ok(())
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn build_precond(
    la: &LinlvoCsr<f64>,
    kind: CachedPrecond,
) -> Result<Option<Box<dyn Preconditioner<Vector = DenseVec<f64>>>>, SolverError> {
    match kind {
        CachedPrecond::None => Ok(None),
        CachedPrecond::Jacobi => {
            let p = JacobiPrecond::from_csr(la)
                .map_err(|e| SolverError::Linlvo(e.to_string()))?;
            Ok(Some(Box::new(p)))
        }
        CachedPrecond::Ilu0 => {
            let p = Ilu0Precond::from_csr(la)
                .map_err(|e| SolverError::Linlvo(e.to_string()))?;
            Ok(Some(Box::new(p)))
        }
        CachedPrecond::Ildlt => {
            let p = IldltPrecond::from_csr(la)
                .map_err(|e| SolverError::Linlvo(e.to_string()))?;
            Ok(Some(Box::new(p)))
        }
    }
}
