use fem_linalg::CsrMatrix as FemCsr;
use fem_linalg::SolverError;

// ─── Macros to eliminate repetitive solver boilerplate ─────────────────────

macro_rules! solve_iterative_simple {
    ($name:ident, $solver:ty, $doc:literal) => {
        #[doc = $doc]
        pub fn $name<T: linlvoScalar>(
            a: &FemCsr<T>, b: &[T], x: &mut [T], cfg: &SolverConfig,
        ) -> Result<SolveResult, SolverError> {
            check_dims(a, b, x)?;
            let la = fem_to_linlvo_csr(a);
            let lb = DenseVec::from_vec(b.to_vec());
            let mut lx = DenseVec::from_vec(x.to_vec());
            let res = <$solver>::default()
                .solve(&la, None, &lb, &mut lx, &cfg.to_linlvo())
                .map_err(SolverError::from)?;
            x.copy_from_slice(lx.as_slice());
            Ok(into_result(res))
        }
    };
}

macro_rules! solve_iterative_restart {
    ($name:ident, $solver:ty, $doc:literal $(, $arg:ident: $ty:ty)*) => {
        #[doc = $doc]
        pub fn $name<T: linlvoScalar>(
            a: &FemCsr<T>, b: &[T], x: &mut [T],
            $($arg: $ty),*,
            cfg: &SolverConfig,
        ) -> Result<SolveResult, SolverError> {
            check_dims(a, b, x)?;
            let la = fem_to_linlvo_csr(a);
            let lb = DenseVec::from_vec(b.to_vec());
            let mut lx = DenseVec::from_vec(x.to_vec());
            let solver = <$solver>::new($($arg),*);
            let res = solver
                .solve(&la, None, &lb, &mut lx, &cfg.to_linlvo())
                .map_err(SolverError::from)?;
            x.copy_from_slice(lx.as_slice());
            Ok(into_result(res))
        }
    };
}

macro_rules! solve_precond_simple {
    ($name:ident, $solver:ty, $precond:ty, $doc:literal) => {
        #[doc = $doc]
        pub fn $name<T: linlvoScalar>(
            a: &FemCsr<T>, b: &[T], x: &mut [T], cfg: &SolverConfig,
        ) -> Result<SolveResult, SolverError> {
            check_dims(a, b, x)?;
            let la = fem_to_linlvo_csr(a);
            let lb = DenseVec::from_vec(b.to_vec());
            let mut lx = DenseVec::from_vec(x.to_vec());
            let prec = <$precond>::from_csr(&la).map_err(|e| SolverError::Linlvo(e.to_string()))?;
            let res = <$solver>::default()
                .solve(&la, Some(&prec), &lb, &mut lx, &cfg.to_linlvo())
                .map_err(SolverError::from)?;
            x.copy_from_slice(lx.as_slice());
            Ok(into_result(res))
        }
    };
}

macro_rules! solve_precond_restart {
    ($name:ident, $solver:ty, $precond:ty, $doc:literal) => {
        #[doc = $doc]
        pub fn $name<T: linlvoScalar>(
            a: &FemCsr<T>, b: &[T], x: &mut [T],
            restart: usize,
            cfg: &SolverConfig,
        ) -> Result<SolveResult, SolverError> {
            check_dims(a, b, x)?;
            let la = fem_to_linlvo_csr(a);
            let lb = DenseVec::from_vec(b.to_vec());
            let mut lx = DenseVec::from_vec(x.to_vec());
            let prec = <$precond>::from_csr(&la).map_err(|e| SolverError::Linlvo(e.to_string()))?;
            let solver = <$solver>::new(restart);
            let res = solver
                .solve(&la, Some(&prec), &lb, &mut lx, &cfg.to_linlvo())
                .map_err(SolverError::from)?;
            x.copy_from_slice(lx.as_slice());
            Ok(into_result(res))
        }
    };
}

macro_rules! solve_direct {
    ($name:ident, $solver:ty, $doc:literal) => {
        #[doc = $doc]
        pub fn $name<T: linlvoScalar>(
            a: &FemCsr<T>, b: &[T],
        ) -> Result<Vec<T>, SolverError> {
            let la = fem_to_linlvo_csr(a);
            let lb = DenseVec::from_vec(b.to_vec());
            let mut lx = DenseVec::zeros(b.len());
            let mut solver = <$solver>::default();
            solver.factor(&la).map_err(|e| SolverError::Linlvo(e.to_string()))?;
            solver.solve(&lb, &mut lx).map_err(|e| SolverError::Linlvo(e.to_string()))?;
            Ok(lx.into_vec())
        }
    };
}

pub(crate) fn check_dims<T>(a: &FemCsr<T>, b: &[T], x: &[T]) -> Result<(), SolverError> {
    if a.nrows != b.len() || a.ncols != x.len() {
        return Err(SolverError::DimensionMismatch {
            rows: a.nrows,
            cols: a.ncols,
            rhs:  b.len(),
        });
    }
    Ok(())
}
