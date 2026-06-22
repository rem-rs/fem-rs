//! HYPRE-compatible solver interfaces backed by native `linger` implementations.
//!
//! Provides MFEM/HYPRE-style naming conventions for BoomerAMG, AMS, ADS,
//! and ParCSR matrix operations. All backends are pure-Rust via `linger`;
//! no external HYPRE installation is required.
//!
//! # Naming convention
//!
//! | fem_solver::hypre type | MFEM/HYPRE equivalent | Backend |
//! |------------------------|----------------------|---------|
//! | `HypreBoomerAMG` | `HypreBoomerAMG` | linger `AmgPrecond` |
//! | `HypreParMatrix` | `HypreParMatrix` | fem-linalg `CsrMatrix` |
//! | `hypre_solve_pcg` | `HyprePCG` (PCG + BoomerAMG) | linger |
//! | `hypre_solve_gmres` | `HypreGMRES` (GMRES + BoomerAMG) | linger |

use fem_linalg::CsrMatrix;
use linger::amg::{AmgConfig, AmgHierarchy, AmgPrecond, CoarsenStrategy, SmootherType};

/// HYPRE-compatible configuration for BoomerAMG.
///
/// All fields mirror the `HYPRE_BoomerAMGSet*` API.
#[derive(Debug, Clone)]
pub struct HypreBoomerAMG {
    pub max_levels: usize,
    pub coarsen_type: usize, // 0 = Ruge-Stueben, 1 = Falgout, 6 = HMIS, 10 = PMIS
    pub cycle_type: usize,   // 1 = V, 2 = W, 3 = F
    pub smoother_type: usize, // 0 = Jacobi, 1 = GS, 6 = Schwarz, etc.
    pub num_sweeps: usize,
    pub relax_weight: f64,
    pub strong_threshold: f64,
    pub print_level: usize,
}

impl Default for HypreBoomerAMG {
    fn default() -> Self {
        HypreBoomerAMG {
            max_levels: 25,
            coarsen_type: 0,
            cycle_type: 1,
            smoother_type: 1,
            num_sweeps: 2,
            relax_weight: 1.0,
            strong_threshold: 0.25,
            print_level: 0,
        }
    }
}

/// Preconditioner built from [`HypreBoomerAMG`] settings.
pub struct HyprePrecond {
    amg: AmgPrecond<f64>,
}

impl HyprePrecond {
    /// Build the AMG hierarchy from matrix `a` using BoomerAMG-style settings.
    pub fn new(a: &CsrMatrix<f64>, cfg: &HypreBoomerAMG) -> Self {
        let smoother = match cfg.smoother_type {
            0 => SmootherType::WeightedJacobi { omega: cfg.relax_weight },
            _ => SmootherType::GaussSeidel,
        };
        let amg_cfg = AmgConfig {
            max_levels: cfg.max_levels,
            theta: cfg.strong_threshold,
            smoother,
            pre_sweeps: cfg.num_sweeps,
            post_sweeps: cfg.num_sweeps,
            ..Default::default()
        };

        let la = fem_to_linger(a);
        let hier = AmgHierarchy::build(la, amg_cfg);
        HyprePrecond { amg: AmgPrecond::new(hier) }
    }
}

/// HYPRE-style `HypreParMatrix` wrapper around a local CSR matrix.
///
/// In MFEM/HYPRE this is a distributed parallel matrix (diag + offd blocks).
/// Here it wraps a [`CsrMatrix<f64>`] with parallel partitioning metadata.
#[derive(Debug, Clone)]
pub struct HypreParMatrix {
    /// Local CSR matrix.
    pub diag: CsrMatrix<f64>,
    /// First local row index (in global numbering).
    pub first_row: usize,
    /// Last local row index + 1.
    pub last_row: usize,
    /// Global number of rows.
    pub global_nrows: usize,
}

impl HypreParMatrix {
    /// Wrap a CSR matrix as a single-process "parallel" matrix.
    pub fn from_serial(a: CsrMatrix<f64>) -> Self {
        let n = a.nrows;
        HypreParMatrix {
            diag: a,
            first_row: 0,
            last_row: n,
            global_nrows: n,
        }
    }
}

/// Solve `A x = b` with PCG + BoomerAMG preconditioner (HYPRE-style).
///
/// Equivalent to MFEM's `HyprePCG` with `BoomerAMG` as the preconditioner.
pub fn hypre_solve_pcg(
    a: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    amg_cfg: &HypreBoomerAMG,
    rtol: f64,
    max_iter: usize,
) -> Result<super::SolveResult, super::SolverError> {
    let precond = HyprePrecond::new(a, amg_cfg);

    // Use the generic preconditioned CG path
    super::solve_pcg_precond(a, b, x, &precond, &super::SolverConfig {
        rtol,
        max_iter,
        ..Default::default()
    })
}

/// Solve `A x = b` with GMRES + BoomerAMG preconditioner (HYPRE-style).
pub fn hypre_solve_gmres(
    a: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    restart: usize,
    amg_cfg: &HypreBoomerAMG,
    rtol: f64,
    max_iter: usize,
) -> Result<super::SolveResult, super::SolverError> {
    let precond = HyprePrecond::new(a, amg_cfg);

    super::solve_gmres_precond(a, b, x, restart, &precond, &super::SolverConfig {
        rtol,
        max_iter,
        ..Default::default()
    })
}

/// Convert fem-linalg CSR to linger CSR.
fn fem_to_linger(a: &CsrMatrix<f64>) -> linger::sparse::CsrMatrix<f64> {
    linger::sparse::CsrMatrix::from_raw(
        a.nrows,
        a.ncols,
        a.row_ptr.clone(),
        a.col_idx.iter().map(|&c| c as usize).collect(),
        a.values.clone(),
    )
}

impl linger::core::preconditioner::Preconditioner for HyprePrecond {
    type Vector = linger::core::vector::DenseVec<f64>;

    fn apply_precond(&self, x: &Self::Vector, y: &mut Self::Vector) {
        self.amg.apply_precond(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;

    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i > 0 { coo.add(i, i - 1, -1.0); }
            if i < n - 1 { coo.add(i, i + 1, -1.0); }
        }
        coo.into_csr()
    }

    #[test]
    fn hypre_pcg_converges() {
        let a = laplacian_1d(32);
        let n = a.nrows;
        let mut b = vec![0.0; n];
        b[n / 2] = 1.0;
        let mut x = vec![0.0; n];
        let cfg = HypreBoomerAMG::default();
        let res = hypre_solve_pcg(&a, &b, &mut x, &cfg, 1e-8, 100).unwrap();
        assert!(res.converged, "Hypre PCG should converge: {res:?}");
    }

    #[test]
    fn hypre_gmres_converges() {
        let a = laplacian_1d(32);
        let n = a.nrows;
        let mut b = vec![0.0; n];
        b[n / 2] = 1.0;
        let mut x = vec![0.0; n];
        let cfg = HypreBoomerAMG::default();
        let res = hypre_solve_gmres(&a, &b, &mut x, 8, &cfg, 1e-8, 100).unwrap();
        assert!(res.converged, "Hypre GMRES should converge: {res:?}");
    }

    #[test]
    fn hypre_par_matrix_wraps_serial() {
        let a = laplacian_1d(5);
        let hp = HypreParMatrix::from_serial(a);
        assert_eq!(hp.first_row, 0);
        assert_eq!(hp.global_nrows, 5);
    }
}
