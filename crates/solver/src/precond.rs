use fem_linalg::CsrMatrix as FemCsr;
use fem_linalg::{fem_to_linlvo_csr, into_result, SolveResult, SolverConfig, SolverError};
use linlvo::{
    core::scalar::Scalar as linlvoScalar,
    iterative::{ConjugateGradient, Fgmres, Gmres},
    precond::{AdsConfig, AdsPrecond, AmsConfig, AmsPrecond},
    sparse::CsrMatrix as linlvoCsr,
    DenseVec, KrylovSolver, Preconditioner,
};

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
        tau: f64,
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
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    kind: PrecondKind,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    match kind {
        PrecondKind::Ilu0 => crate::solve_gmres_ilu0(a, b, x, restart, cfg),
        PrecondKind::Iluk(k) => crate::solve_gmres_iluk(a, b, x, restart, k, cfg),
        PrecondKind::Ilut { tau, fill } => {
            crate::solve_gmres_ilut(a, b, x, restart, tau, fill, cfg)
        }
    }
}

/// Lift real-valued (`f64`) interpolation blocks to the solve scalar type.
///
/// The auxiliary-space interpolation blocks (`Pi`) are assembled as `f64`
/// matrices by the assembly layer; the Krylov solve runs in `T`.
fn lift_blocks<T: linlvoScalar>(blocks: &[linlvoCsr<f64>]) -> Vec<linlvoCsr<T>> {
    blocks
        .iter()
        .map(|m| {
            linlvoCsr::from_raw(
                m.nrows(),
                m.ncols(),
                m.row_ptr().to_vec(),
                m.col_idx().to_vec(),
                m.values()
                    .iter()
                    .map(|&v| <T as linlvoScalar>::from_f64(v))
                    .collect(),
            )
        })
        .collect()
}

// ─── Auxiliary-space Maxwell Solver (AMS) ──────────────────────────────────

/// Configuration for AMS (Auxiliary-space Maxwell Solver) preconditioner.
///
/// AMS is the Hiptmair-Xu preconditioner for H(curl) problems (Maxwell).
/// It uses a multigrid V-cycle on the auxiliary nodal space plus
/// a stationary correction on the edge space.
#[derive(Debug, Clone, Default)]
pub struct AmsSolverConfig {
    pub inner_cfg: SolverConfig,
    pub ams_cfg: AmsConfig,
    /// Face (curl) auxiliary-space interpolation blocks `Pi = [Pix, Piy, Piz]`
    /// (each `n_edges × n_nodes`), the analogue of MFEM's
    /// `HypreAMS::MakeGradientAndInterpolation` handing `id_ND` component
    /// blocks to `HYPRE_AMSSetInterpolations` (`None` = two-space AMS, the
    /// historical `AmsPrecond::new` behaviour).  Real-valued `f64` blocks;
    /// converted internally to the solve scalar type.
    pub pi: Option<Vec<linlvoCsr<f64>>>,
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
    let la = fem_to_linlvo_csr(a);
    let mut lx: Vec<T> = x.to_vec();

    // `pi` is stored real-valued; lift to the scalar type for `with_pi`.
    let pi_lifted: Option<Vec<linlvoCsr<T>>> = cfg.pi.as_ref().map(|blocks| lift_blocks(blocks));
    let ams = match &pi_lifted {
        Some(pi) => AmsPrecond::<T>::with_pi(&la, g, pi, cfg.ams_cfg.clone())
            .map_err(|e| SolverError::Linlvo(e.to_string()))?,
        None => AmsPrecond::<T>::new(&la, g, cfg.ams_cfg.clone())
            .map_err(|e| SolverError::Linlvo(e.to_string()))?,
    };

    // linger's CG converges on the preconditioned ENERGY `(B r, r)/(B r₀, r₀)`,
    // which can dip below rtol (even via `|·|` across an energy zero) while
    // the true residual `‖b − A x‖/‖b‖` is still 1–2 orders of magnitude too
    // large.  Restart PCG from the current iterate, recomputing the true
    // residual each round, until the true relative residual meets the
    // tolerance or the PCG iteration budget is exhausted.
    let mut total_iters = 0usize;
    let mut final_residual = f64::INFINITY;
    let mut converged = false;
    let b_norm = b
        .iter()
        .fold(T::zero(), |s, &v| s + v * v)
        .abs()
        .sqrt();
    let tol = <T as linlvoScalar>::from_f64(cfg.inner_cfg.rtol);
    let atol = <T as linlvoScalar>::from_f64(cfg.inner_cfg.atol.max(0.0));

    for _restart in 0..8 {
        // r = b − A x (true residual).
        let mut ax = vec![T::zero(); la.nrows()];
        la.spmv(&lx, &mut ax);
        let mut r = vec![T::zero(); la.nrows()];
        for i in 0..la.nrows() {
            r[i] = b[i] - ax[i];
        }

        let r_norm = r
            .iter()
            .fold(T::zero(), |s, &v| s + v * v)
            .sqrt();

        final_residual = num_traits::ToPrimitive::to_f64(&(r_norm / b_norm))
            .unwrap_or(f64::INFINITY);
        if r_norm <= tol * b_norm + atol {
            converged = true;
            break;
        }
        if total_iters >= cfg.inner_cfg.max_iter {
            break;
        }

        // FGMRES round: robust to a non-SPD AMS cycle (V11 + SGS), unlike
        // plain CG which breaks down when (p, A p) -> 0 mid-iteration.
        let mut dx = DenseVec::zeros(la.nrows());
        let res = Fgmres::<T>::new(30)
            .solve(
                &la,
                Some(&ams),
                &DenseVec::from_vec(r),
                &mut dx,
                &cfg.inner_cfg.to_linlvo(),
            )
            .map_err(SolverError::from)?;
        total_iters += res.iterations;
        let xs = lx.as_mut_slice();
        let dxs = dx.as_slice();
        for i in 0..la.nrows() {
            xs[i] = xs[i] + dxs[i];
        }
        if res.converged && final_residual <= cfg.inner_cfg.rtol {
            converged = true;
            break;
        }
    }

    if final_residual <= cfg.inner_cfg.rtol {
        converged = true;
    }
    x.copy_from_slice(&lx);
    Ok(SolveResult {
        converged,
        iterations: total_iters,
        final_residual,
    })
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
#[derive(Debug, Clone, Default)]
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

    let ads = AdsPrecond::<T>::new(&la, c, g, cfg.ads_cfg.clone())
        .map_err(|e| SolverError::Linlvo(e.to_string()))?;

    // Same true-residual restart driver as `solve_pcg_ams` (see there): the
    // preconditioned energy criterion can report convergence well before the
    // true residual meets the tolerance.
    let b_norm = b
        .iter()
        .fold(T::zero(), |s, &v| s + v * v)
        .sqrt();
    let tol = <T as linlvoScalar>::from_f64(cfg.inner_cfg.rtol);
    let atol = <T as linlvoScalar>::from_f64(cfg.inner_cfg.atol.max(0.0));

    let mut lx: Vec<T> = x.to_vec();
    let mut total_iters = 0usize;
    let mut final_residual = f64::INFINITY;
    let mut converged = false;

    for _restart in 0..8 {
        let mut ax = vec![T::zero(); la.nrows()];
        la.spmv(&lx, &mut ax);
        let mut r = vec![T::zero(); la.nrows()];
        for i in 0..la.nrows() {
            r[i] = b[i] - ax[i];
        }

        let r_norm = r
            .iter()
            .fold(T::zero(), |s, &v| s + v * v)
            .sqrt();

        final_residual = num_traits::ToPrimitive::to_f64(&(r_norm / b_norm))
            .unwrap_or(f64::INFINITY);
        if r_norm <= tol * b_norm + atol {
            converged = true;
            break;
        }
        if total_iters >= cfg.inner_cfg.max_iter {
            break;
        }

        // PCG for the correction dx (zero initial guess).
        let mut dx = DenseVec::zeros(la.nrows());
        let res = ConjugateGradient::<T>::default()
            .solve(
                &la,
                Some(&ads),
                &DenseVec::from_vec(r),
                &mut dx,
                &cfg.inner_cfg.to_linlvo(),
            )
            .map_err(SolverError::from)?;
        total_iters += res.iterations;
        let xs = lx.as_mut_slice();
        let dxs = dx.as_slice();
        for i in 0..la.nrows() {
            xs[i] = xs[i] + dxs[i];
        }
    }

    if final_residual <= cfg.inner_cfg.rtol {
        converged = true;
    }
    x.copy_from_slice(&lx);
    Ok(SolveResult {
        converged,
        iterations: total_iters,
        final_residual,
    })
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

// ─── hypre ADS (five-level multiplicative subspace cycle) ───────────────────

/// Configuration for the hypre-faithful ADS preconditioner
/// ([`linlvo::precond::AdsHyprePrecond`], the MFEM `HypreADS` structure) —
/// distinct from the additive three-term [`AdsSolverConfig`].
#[derive(Debug, Clone, Default)]
pub struct AdsHypreSolverConfig {
    pub inner_cfg: SolverConfig,
    pub ads_cfg: linlvo::precond::AdsHypreConfig,
    /// The three edge (`ND`) interpolation blocks `Pi_nd` (`n_edges × n_nodes`)
    /// for the inner AMS (`B_C`), real-valued like [`AmsSolverConfig::pi`].
    pub pi_nd: Vec<linlvoCsr<f64>>,
    /// The three face (`RT`) interpolation blocks `Pi_rt` (`n_faces × n_nodes`).
    pub pi_rt: Vec<linlvoCsr<f64>>,
}

/// A reusable PCG + hypre-ADS solver, the fem-rs analogue of MFEM's
/// `HyprePCG` + `HypreADS` pair: the (expensive) preconditioner setup is done
/// **once** in [`Self::new`] and reused for every [`Self::solve`], exactly
/// like MFEM's `if (ads_a2 == NULL) { ads_a2 = new HypreADS(*A2, ...); }`
/// (joule_solver.cpp:580-590).
///
/// The operator must not change between setup and solve (the usual
/// preconditioner contract).
pub struct AdsHypreSolver<T: linlvoScalar> {
    la: linlvoCsr<T>,
    ads: linlvo::precond::AdsHyprePrecond<T>,
    inner_cfg: SolverConfig,
}

impl<T: linlvoScalar> AdsHypreSolver<T> {
    /// Set up the hypre-ADS preconditioner for `a`.
    ///
    /// `c`/`g` are linlvo matrices (see [`fem_linalg::fem_to_linlvo_csr`]);
    /// the interpolation blocks are real-valued `f64` and lifted to `T`.
    pub fn new(
        a: &FemCsr<T>,
        c: &linlvoCsr<T>,
        g: &linlvoCsr<T>,
        cfg: &AdsHypreSolverConfig,
    ) -> Result<Self, SolverError> {
        let la = fem_to_linlvo_csr(a);
        let pi_nd = lift_blocks::<T>(&cfg.pi_nd);
        let pi_rt = lift_blocks::<T>(&cfg.pi_rt);
        let ads = linlvo::precond::AdsHyprePrecond::<T>::new(
            &la,
            c,
            g,
            &pi_rt,
            &pi_nd,
            cfg.ads_cfg.clone(),
        )
        .map_err(|e| SolverError::Linlvo(e.to_string()))?;
        Ok(AdsHypreSolver {
            la,
            ads,
            inner_cfg: cfg.inner_cfg.clone(),
        })
    }

    /// Setup diagnostics of the hypre-ADS preconditioner.
    pub fn profile(&self) -> &linlvo::precond::AdsHypreProfile {
        self.ads.profile()
    }

    /// Solve `A x = b` with PCG + the hypre-ADS preconditioner.
    ///
    /// Same true-residual restart driver as [`solve_pcg_ads`] (see there): the
    /// preconditioned energy criterion can report convergence well before the
    /// true residual meets the tolerance.
    pub fn solve(&self, b: &[T], x: &mut [T]) -> Result<SolveResult, SolverError> {
        let la = &self.la;
        if b.len() != la.nrows() || x.len() != la.nrows() {
            return Err(SolverError::Linlvo(
                "AdsHypreSolver::solve: b/x length does not match the operator".into(),
            ));
        }
        let b_norm = b.iter().fold(T::zero(), |s, &v| s + v * v).sqrt();
        let tol = <T as linlvoScalar>::from_f64(self.inner_cfg.rtol);
        let atol = <T as linlvoScalar>::from_f64(self.inner_cfg.atol.max(0.0));

        let mut lx: Vec<T> = x.to_vec();
        let mut total_iters = 0usize;
        let mut final_residual = f64::INFINITY;
        let mut converged = false;

        for _restart in 0..8 {
            let mut ax = vec![T::zero(); la.nrows()];
            la.spmv(&lx, &mut ax);
            let mut r = vec![T::zero(); la.nrows()];
            for i in 0..la.nrows() {
                r[i] = b[i] - ax[i];
            }

            let r_norm = r.iter().fold(T::zero(), |s, &v| s + v * v).sqrt();

            final_residual = num_traits::ToPrimitive::to_f64(&(r_norm / b_norm))
                .unwrap_or(f64::INFINITY);
            if r_norm <= tol * b_norm + atol {
                converged = true;
                break;
            }
            if total_iters >= self.inner_cfg.max_iter {
                break;
            }

            // PCG for the correction dx (zero initial guess).
            let mut dx = DenseVec::zeros(la.nrows());
            let res = ConjugateGradient::<T>::default()
                .solve(
                    la,
                    Some(&self.ads),
                    &DenseVec::from_vec(r),
                    &mut dx,
                    &self.inner_cfg.to_linlvo(),
                )
                .map_err(SolverError::from)?;
            total_iters += res.iterations;
            let xs = lx.as_mut_slice();
            let dxs = dx.as_slice();
            for i in 0..la.nrows() {
                xs[i] = xs[i] + dxs[i];
            }
        }

        if final_residual <= self.inner_cfg.rtol {
            converged = true;
        }
        x.copy_from_slice(&lx);
        Ok(SolveResult {
            converged,
            iterations: total_iters,
            final_residual,
        })
    }
}

/// One-shot H(div) solve with PCG + the hypre-ADS preconditioner (builds the
/// preconditioner for this call only — prefer [`AdsHypreSolver`] when the
/// operator is reused across many right-hand sides).
///
/// # Arguments
/// * `a`   — H(div) stiffness matrix (face DOFs)
/// * `c`   — Discrete curl matrix (edges -> faces)
/// * `g`   — Discrete gradient matrix (vertices -> edges)
/// * `b`   — right-hand side
/// * `x`   — initial guess on entry, solution on exit
/// * `cfg` — solver configuration (interpolation blocks + hypre parameters)
pub fn solve_pcg_ads_hypre<T: linlvoScalar>(
    a: &FemCsr<T>,
    c: &linlvoCsr<T>,
    g: &linlvoCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &AdsHypreSolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let solver = AdsHypreSolver::<T>::new(a, c, g, cfg)?;
    solver.solve(b, x)
}

/// Block-diagonal preconditioner for complex (2x2 block) systems.
pub struct BlockDiagPrecond<P: Preconditioner<Vector = DenseVec<f64>>> {
    pub inner: P,
    pub n: usize,
}

impl<P: Preconditioner<Vector = DenseVec<f64>>> Preconditioner for BlockDiagPrecond<P> {
    type Vector = DenseVec<f64>;
    fn apply_precond(&self, x: &DenseVec<f64>, z: &mut DenseVec<f64>) {
        let n = self.n;
        let xs = x.as_slice();
        let zs = z.as_mut_slice();
        let x_re = DenseVec::from_vec(xs[..n].to_vec());
        let mut z_re = DenseVec::zeros(n);
        self.inner.apply_precond(&x_re, &mut z_re);
        zs[..n].copy_from_slice(z_re.as_slice());
        let x_im = DenseVec::from_vec(xs[n..].to_vec());
        let mut z_im = DenseVec::zeros(n);
        self.inner.apply_precond(&x_im, &mut z_im);
        zs[n..].copy_from_slice(z_im.as_slice());
    }
}

/// Scaled preconditioner — wraps a precond and scales output by `scale`.
/// Equivalent to C++ `ScaledOperator(pc, s)`.
pub struct ScaledPrecond<P> {
    pub inner: P,
    pub scale: f64,
}

impl<P: Preconditioner<Vector = DenseVec<f64>>> Preconditioner for ScaledPrecond<P> {
    type Vector = DenseVec<f64>;
    fn apply_precond(&self, x: &DenseVec<f64>, z: &mut DenseVec<f64>) {
        self.inner.apply_precond(x, z);
        for v in z.as_mut_slice().iter_mut() {
            *v *= self.scale;
        }
    }
}

/// Block-diagonal preconditioner with separate preconds for real and imaginary blocks.
/// Equivalent to C++ `BlockDiagonalPreconditioner` with two distinct diagonal blocks.
pub struct BlockDiagPrecondPair<
    P1: Preconditioner<Vector = DenseVec<f64>>,
    P2: Preconditioner<Vector = DenseVec<f64>>,
> {
    pub pre_re: P1,
    pub pre_im: P2,
    pub n: usize,
}

impl<P1: Preconditioner<Vector = DenseVec<f64>>, P2: Preconditioner<Vector = DenseVec<f64>>>
    Preconditioner for BlockDiagPrecondPair<P1, P2>
{
    type Vector = DenseVec<f64>;
    fn apply_precond(&self, x: &DenseVec<f64>, z: &mut DenseVec<f64>) {
        let n = self.n;
        let xs = x.as_slice();
        let zs = z.as_mut_slice();
        let x_re = DenseVec::from_vec(xs[..n].to_vec());
        let mut z_re = DenseVec::zeros(n);
        self.pre_re.apply_precond(&x_re, &mut z_re);
        zs[..n].copy_from_slice(z_re.as_slice());
        let x_im = DenseVec::from_vec(xs[n..].to_vec());
        let mut z_im = DenseVec::zeros(n);
        self.pre_im.apply_precond(&x_im, &mut z_im);
        zs[n..].copy_from_slice(z_im.as_slice());
    }
}
