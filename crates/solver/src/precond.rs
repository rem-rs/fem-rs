use fem_linalg::CsrMatrix as FemCsr;
use linlvo::{
    core::scalar::Scalar as linlvoScalar,
    iterative::{ConjugateGradient, Gmres},
    precond::{AmsPrecond, AmsConfig, AdsPrecond, AdsConfig},
    sparse::CsrMatrix as linlvoCsr,
    DenseVec, KrylovSolver,
};
use fem_linalg::{fem_to_linlvo_csr, into_result, SolverConfig, SolverError, SolveResult};

use crate::macros::check_dims;

// ─── ILU family preconditioner selector ────────────────────────────────────

/// ILU family preconditioner selector.
///
/// Pass one of these variants to [`solve_precond_kind`] to choose the
/// incomplete-factorisation strategy without changing the calling code.
///
/// | Variant | Fill strategy | Typical use |
/// |---------|---------------|-------------|
/// | `Ilu0`  | Sparsity of `A` | Cheap, SPD or diagonally dominant |
/// | `Iluk(k)` | Level-of-fill — k | Better quality for moderate fill |
/// | `Ilut { tau, fill }` | Drop tolerance + fill bound | Non-symmetric, harder systems |
#[derive(Debug, Clone, Default)]
pub enum PrecondKind {
    /// ILU(0): no extra fill (fastest build, lowest quality).
    #[default]
    Ilu0,
    /// ILU(k): allow fill-in entries up to level `k`.
    /// `k = 0` equals ILU(0); larger `k` approaches exact LU.
    Iluk(usize),
    /// ILUT(τ, p): drop entries smaller than `tau × ‖row‖₂`;
    /// keep at most `fill` off-diagonal entries per row in L and U.
    Ilut {
        /// Relative drop tolerance (e.g. `0.01`).
        tau:  f64,
        /// Max off-diagonal fill per row in each factor.
        fill: usize,
    },
}

/// Unified ILU-family GMRES dispatcher.
///
/// Selects the preconditioner at runtime from a [`PrecondKind`] value.
/// Useful when the choice of preconditioner should be a configuration
/// parameter rather than a compile-time decision.
///
/// # Example
/// ```rust,ignore
/// use fem_solver::{solve_precond_kind, PrecondKind, SolverConfig};
///
/// let res = solve_precond_kind(&a, &b, &mut x, 30,
///     PrecondKind::Ilut { tau: 0.01, fill: 20 },
///     &SolverConfig::default())?;
/// ```
pub fn solve_precond_kind<T: linlvoScalar>(
    a:       &FemCsr<T>,
    b:       &[T],
    x:       &mut [T],
    restart: usize,
    kind:    PrecondKind,
    cfg:     &SolverConfig,
) -> Result<SolveResult, SolverError> {
    match kind {
        PrecondKind::Ilu0             => crate::solve_gmres_ilu0(a, b, x, restart, cfg),
        PrecondKind::Iluk(k)          => crate::solve_gmres_iluk(a, b, x, restart, k, cfg),
        PrecondKind::Ilut { tau, fill } => crate::solve_gmres_ilut(a, b, x, restart, tau, fill, cfg),
    }
}

// ─── Auxiliary-space Maxwell Solver (AMS) ──────────────────────────────────

/// Configuration for AMS (Auxiliary-space Maxwell Solver) preconditioner.
///
/// AMS is the Hiptmair-Xu preconditioner for H(curl) problems (Maxwell).
/// It uses a multigrid V-cycle on the auxiliary nodal space plus
/// a stationary correction on the edge space.
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct AmsSolverConfig {
    pub inner_cfg: SolverConfig,
    pub ams_cfg: AmsConfig,
}

/// Solve an H(curl) system using PCG with AMS preconditioner.
///
/// # Arguments
/// * `a`       — H(curl) stiffness matrix (edge DOFs)
/// * `g`       — Discrete gradient matrix (vertices -> edges)
/// * `b`       — right-hand side
/// * `x`       — initial guess on entry, solution on exit
/// * `cfg`     — solver configuration
///
/// # Type parameters
/// The discrete gradient `g` is passed as a linlvo CsrMatrix to match internal types.
/// Convert using `fem_to_linlvo_csr`.
pub fn solve_pcg_ams<T: linlvoScalar>(
    a: &FemCsr<T>,
    g: &linlvoCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &AmsSolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());

    let ams = AmsPrecond::<T>::new(&la, g, cfg.ams_cfg.clone())
        .map_err(|e| SolverError::Linlvo(e.to_string()))?;

    let res = ConjugateGradient::<T>::default()
        .solve(&la, Some(&ams), &lb, &mut lx, &cfg.inner_cfg.to_linlvo())
        .map_err(SolverError::from)?;

    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// Solve an H(curl) system using GMRES with AMS preconditioner.
///
/// Use this for non-symmetric H(curl) problems (e.g., with absorbing BCs).
pub fn solve_gmres_ams<T: linlvoScalar>(
    a: &FemCsr<T>,
    g: &linlvoCsr<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    cfg: &AmsSolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());

    let ams = AmsPrecond::<T>::new(&la, g, cfg.ams_cfg.clone())
        .map_err(|e| SolverError::Linlvo(e.to_string()))?;

    let solver = Gmres::<T>::new(restart);
    let res = solver
        .solve(&la, Some(&ams), &lb, &mut lx, &cfg.inner_cfg.to_linlvo())
        .map_err(SolverError::from)?;

    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

// ─── Auxiliary-space Divergence Solver (ADS) ───────────────────────────────

/// Configuration for ADS (Auxiliary-space Divergence Solver) preconditioner.
///
/// ADS is the Hiptmair-Xu preconditioner for H(div) problems (Darcy flow).
/// It combines auxiliary-space cycles on the edge space (via curl) and
/// nodal space (via gradient) for robust H(div) preconditioning.
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct AdsSolverConfig {
    pub inner_cfg: SolverConfig,
    pub ads_cfg: AdsConfig,
}

/// Solve an H(div) system using PCG with ADS preconditioner.
///
/// # Arguments
/// * `a`       — H(div) stiffness matrix (face DOFs)
/// * `c`       — Discrete curl matrix (edges -> faces)
/// * `g`       — Discrete gradient matrix (vertices -> edges)
/// * `b`       — right-hand side
/// * `x`       — initial guess on entry, solution on exit
/// * `cfg`     — solver configuration
///
/// # Notes
/// Both `c` and `g` should be converted to linlvo format using `fem_to_linlvo_csr`.
pub fn solve_pcg_ads<T: linlvoScalar>(
    a: &FemCsr<T>,
    c: &linlvoCsr<T>,
    g: &linlvoCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &AdsSolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());

    let ads = AdsPrecond::<T>::new(&la, c, g, cfg.ads_cfg.clone())
        .map_err(|e| SolverError::Linlvo(e.to_string()))?;

    let res = ConjugateGradient::<T>::default()
        .solve(&la, Some(&ads), &lb, &mut lx, &cfg.inner_cfg.to_linlvo())
        .map_err(SolverError::from)?;

    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// Solve an H(div) system using GMRES with ADS preconditioner.
///
/// Use this for non-symmetric H(div) problems.
pub fn solve_gmres_ads<T: linlvoScalar>(
    a: &FemCsr<T>,
    c: &linlvoCsr<T>,
    g: &linlvoCsr<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    cfg: &AdsSolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());

    let ads = AdsPrecond::<T>::new(&la, c, g, cfg.ads_cfg.clone())
        .map_err(|e| SolverError::Linlvo(e.to_string()))?;

    let solver = Gmres::<T>::new(restart);
    let res = solver
        .solve(&la, Some(&ads), &lb, &mut lx, &cfg.inner_cfg.to_linlvo())
        .map_err(SolverError::from)?;

    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}
