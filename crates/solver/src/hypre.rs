//! HYPRE-compatible solver interfaces backed by native `linlvo` implementations.
//!
//! Provides MFEM/HYPRE-style naming conventions for BoomerAMG, AMS, ADS,
//! and ParCSR matrix operations. All backends are pure-Rust via `linlvo`;
//! no external HYPRE installation is required.
//!
//! # Naming convention
//!
//! | fem_solver::hypre type | MFEM/HYPRE equivalent | Backend |
//! |------------------------|----------------------|---------|
//! | `HypreBoomerAMG` | `HypreBoomerAMG` | linlvo `AmgPrecond` |
//! | `HypreParMatrix` | `HypreParMatrix` | fem-linalg `CsrMatrix` |
//! | `hypre_solve_pcg` | `HyprePCG` (PCG + BoomerAMG) | linlvo |
//! | `hypre_solve_gmres` | `HypreGMRES` (GMRES + BoomerAMG) | linlvo |

use fem_linalg::CsrMatrix;
use linlvo::amg::{AmgConfig, AmgHierarchy, AmgPrecond, CoarsenStrategy, SmootherType};

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

        let la = fem_to_linlvo(a);
        let hier = AmgHierarchy::build(la, amg_cfg);
        HyprePrecond { amg: AmgPrecond::new(hier) }
    }
}

/// HYPRE-style `HypreParMatrix` wrapper around a local CSR matrix.
///
/// In MFEM/HYPRE this is a distributed parallel matrix (diag + offd blocks).
/// Here it wraps a [`CsrMatrix<f64>`] with parallel partitioning metadata.
///
/// **Deprecated**: use [`fem_parallel::ParCsrMatrix`] for true distributed storage,
/// or `fem_linalg::CsrMatrix` for serial storage.  This type is kept for
/// backward compatibility with MFEM-style code.
#[derive(Debug, Clone)]
#[deprecated(since = "0.2.0", note = "use fem_parallel::ParCsrMatrix or fem_linalg::CsrMatrix directly")]
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
///
/// **Deprecated**: use `fem_parallel::par_solve_pcg_amg` for distributed systems,
/// or `fem_solver::solve_pcg_precond` with a custom preconditioner for serial systems.
#[deprecated(since = "0.2.0", note = "use fem_solver::solve_pcg_precond with AmgPrecond, or fem_parallel::par_solve_pcg_amg for distributed")]
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
///
/// **Deprecated**: use `fem_solver::solve_gmres_precond` with `AmgPrecond`,
/// or `fem_parallel::par_solve_gmres_amg` for distributed systems.
#[deprecated(since = "0.2.0", note = "use fem_solver::solve_gmres_precond with AmgPrecond")]
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

/// Convert fem-linalg CSR to linlvo CSR.
fn fem_to_linlvo(a: &CsrMatrix<f64>) -> linlvo::sparse::CsrMatrix<f64> {
    linlvo::sparse::CsrMatrix::from_raw(
        a.nrows,
        a.ncols,
        a.row_ptr.clone(),
        a.col_idx.iter().map(|&c| c as usize).collect(),
        a.values.clone(),
    )
}

impl linlvo::core::preconditioner::Preconditioner for HyprePrecond {
    type Vector = linlvo::core::vector::DenseVec<f64>;

    fn apply_precond(&self, x: &Self::Vector, y: &mut Self::Vector) {
        self.amg.apply_precond(x, y)
    }
}

#[cfg(test)]
#[allow(deprecated)]
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

    /// Poisson 2D: hypre (BoomerAMG+PCG) === direct CG
    #[test]
    fn hypre_poisson_2d_matches_native_cg() {
        use fem_assembly::{
            Assembler,
            standard::{DiffusionIntegrator, DomainSourceIntegrator},
        };
        use fem_mesh::SimplexMesh;
        use fem_space::{
            H1Space, fe_space::FESpace,
            constraints::{apply_dirichlet, boundary_dofs},
        };

        let n = 16;
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        let space = H1Space::new(mesh, 1);
        let dofs = space.n_dofs();
        let quad = 3;

        // Assemble
        let mat_orig = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], quad);
        let source = DomainSourceIntegrator::new(|x: &[f64]| {
            2.0 * std::f64::consts::PI * std::f64::consts::PI
                * (std::f64::consts::PI * x[0]).sin()
                * (std::f64::consts::PI * x[1]).sin()
        });
        let mut rhs_hypre = Assembler::assemble_linear(&space, &[&source], quad);
        let mut rhs_native = rhs_hypre.clone();
        let dm = space.dof_manager();
        let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
        let bnd_vals = vec![0.0_f64; bnd.len()];

        let mut mat_hypre = mat_orig.clone();
        let mut mat_native = mat_orig;
        apply_dirichlet(&mut mat_hypre, &mut rhs_hypre, &bnd, &bnd_vals);
        apply_dirichlet(&mut mat_native, &mut rhs_native, &bnd, &bnd_vals);

        // Hypre solve: BoomerAMG + PCG
        let mut x_hypre = vec![0.0; dofs];
        let amg_cfg = HypreBoomerAMG::default();
        let res_h = hypre_solve_pcg(&mat_hypre, &rhs_hypre, &mut x_hypre, &amg_cfg, 1e-8, 2000).unwrap();
        assert!(res_h.converged, "Hypre PCG not converged");

        // Native CG
        let mut x_native = vec![0.0; dofs];
        let cfg = crate::SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 10_000, verbose: false, ..crate::SolverConfig::default() };
        let res_c = crate::solve_cg(&mat_native, &rhs_native, &mut x_native, &cfg).unwrap();
        assert!(res_c.converged, "Native CG not converged");

        // Solutions should match to solver tolerance
        let max_diff = x_hypre.iter().zip(x_native.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_diff < 1e-6,
            "hypre PCG vs native CG: max|diff| = {max_diff:.3e}");

        // Both solvers should have converged
        assert!(res_h.converged && res_c.converged, "both solvers must converge");
    }

}
