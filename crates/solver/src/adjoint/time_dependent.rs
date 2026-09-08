//! Adjoint (reverse-mode) time integration with checkpointed forward solves —
//! the Rust counterpart of MFEM's `TimeDependentAdjointOperator` plus the
//! adjoint mode of `CVODESSolver` (CVODES semantics), implemented on this
//! crate's own Nordsieck BDF (no SUNDIALS dependency).
//!
//! # Problem setup
//!
//! For the forward ODE `y' = f(t, y; p)` on `[t0, tf]` and a goal functional
//!
//! ```text
//! G(p) = ∫_{t0}^{tf} g(t, y, p) dt
//! ```
//!
//! the adjoint state `λ` solves the *backward-in-time* ODE
//!
//! ```text
//! dλ/dt = fB(t, y(t), λ) = -(∂f/∂y)ᵀ λ - (∂g/∂y)ᵀ,   λ(tf) = 0
//! ```
//!
//! (this is exactly the rate MFEM's `AdjointRateMult` must supply), and the
//! gradient is accumulated along the backward integration through the
//! quadrature-sensitivity rate (MFEM's `QuadratureSensitivityMult`)
//!
//! ```text
//! dG/dp = ∫_{t0}^{tf} qBdot(t, y(t), λ) dt,   qBdot = (∂g/∂p)ᵀ - (∂f/∂p)ᵀ λ.
//! ```
//!
//! Sensitivities w.r.t. the initial data are `dG/dy0 = λ(t0)`.
//!
//! # CVODES workflow mirrored here
//!
//! | CVODES / MFEM `CVODESSolver`            | this module                                     |
//! |-----------------------------------------|-------------------------------------------------|
//! | `Init` + forward `Step` loop            | [`AdjointSolver::run_forward`]                  |
//! | `InitAdjointSolve(ncheck, interp)`      | [`AdjointConfig::steps_per_checkpoint`] + [`AdjointConfig::interpolation`] |
//! | `InitB` + `InitQuadIntegrationB`        | [`AdjointSolver::init_adjoint`]                 |
//! | `StepB(w, t, dt)`                       | [`AdjointSolver::step_adjoint_to`]              |
//! | `GetForwardSolution(tB)`                | [`AdjointSolver::forward_solution_at`]          |
//! | `EvalQuadIntegration` / `EvalQuadIntegrationB` | [`AdjointSolver::quadrature`] / [`AdjointSolver::quad_sensitivity`] |
//! | `CV_HERMITE` / `CV_POLYNOMIAL`          | [`Interpolation::Hermite`] / [`Interpolation::Polynomial`] |
//!
//! # Checkpointing with bisection ("检查点二分")
//!
//! During the forward run a checkpoint (the full Nordsieck state) is sealed
//! every [`AdjointConfig::steps_per_checkpoint`] accepted steps; the dense
//! per-step data inside a checkpoint interval is *discarded*. During the
//! backward run, whenever the forward solution `y(tB)` is needed inside an
//! interval `[t_i, t_{i+1}]`, the interval is re-integrated from the right
//! checkpoint — but only down to the midpoint of the not-yet-covered stretch
//! (a bisection of the remaining uncovered span). If the backward integration
//! later needs times below the covered window, the window is extended by
//! re-integrating to the next bisection target. Each extension halves the
//! remaining span, so the total re-integration work per interval stays within
//! 2× a full re-integration — the same cost model as the CVODES checkpointing
//! machinery — while typically doing much less work when backward steps are
//! coarse.
//!
//! Setting [`AdjointConfig::store_all_forward_steps`] keeps the dense data of
//! every accepted forward step instead (全量存储 reference mode): no
//! re-integration ever happens (see the
//! `checkpoint_bisection_matches_full_storage` test).
//!
//! # Numerical trade-offs vs CVODES
//!
//! * The forward problem is integrated with the same variable-order (1 up to
//!   `max_order`), variable-step Nordsieck BDF in *both* directions. CVODES'
//!   forward CV_ADAMS (Adams–Moulton) mode is not reproduced; like-for-like
//!   comparisons should use the BDF mode (`-no-a` in the MFEM miniapps).
//!   Order selection raises the order after three consecutive small local
//!   errors and lowers it immediately on large ones.
//! * Forward and backward quadratures are accumulated by 3-point
//!   Gauss–Legendre quadrature of the BDF interpolant (CVODES integrates
//!   quadrature variables alongside the solution with their own error test).
//!   This is high-order accurate for the polynomial interpolants and does not
//!   drive step-size selection — the quadrature tolerances of
//!   `InitQuadIntegration` have no counterpart here.
//! * The implicit Newton correction solves `A δ = -F` with
//!   `A = l1·I - h·∂f/∂y` (identity form). MFEM examples often provide the
//!   mass-matrix form `M - γJ` via `SUNImplicitSetupB`; the converged BDF
//!   step is identical — only the inner linear-solver path differs.
//! * Vector absolute tolerances (`SetSVtolerances`) are supported on both the
//!   forward and the adjoint problem.

use std::cell::{Cell, RefCell};

use fem_linalg::{CooMatrix, CsrMatrix};

use crate::bdf::{pascal_coeff, ERROR_WEIGHTS, L_COEFFS};
use crate::butcher::i_step_controller;
use crate::{solve_gmres, SolverConfig};

// ─── Operator trait ──────────────────────────────────────────────────────────

/// Time-dependent operator with adjoint rate equations — the equivalent of
/// MFEM's `TimeDependentAdjointOperator`.
///
/// The forward rate [`Self::mult`]/[`Self::jac`] drive the checkpointed
/// forward integration; [`Self::adjoint_rate_mult`]/[`Self::adjoint_jac`]
/// drive the backward integration, which receives the forward solution `y(t)`
/// at the current (backward) time. The quadrature hooks are optional.
pub trait TimeDependentAdjointOperator: Send + Sync {
    /// Dimension of the forward state.
    fn dim(&self) -> usize;

    /// Forward rate equation (MFEM `Mult`): `ydot = f(t, y)`.
    fn mult(&self, t: f64, y: &[f64], ydot: &mut [f64]);

    /// Jacobian of the forward rate: `J = ∂f/∂y` at `(t, y)`.
    fn jac(&self, t: f64, y: &[f64]) -> CsrMatrix<f64>;

    /// Dimension of the adjoint state (MFEM `GetAdjointHeight`). Defaults to
    /// [`Self::dim`].
    fn adjoint_dim(&self) -> usize {
        self.dim()
    }

    /// Adjoint rate equation (MFEM `AdjointRateMult`): `yBdot = fB(t, y, yB)`,
    /// the right-hand side of the backward-in-time adjoint ODE.
    fn adjoint_rate_mult(&self, t: f64, y: &[f64], yb: &[f64], ybdot: &mut [f64]);

    /// Jacobian of the adjoint rate w.r.t. `yb`: `JB = ∂fB/∂yB` at `(t, y, yB)`.
    fn adjoint_jac(&self, t: f64, y: &[f64], yb: &[f64]) -> CsrMatrix<f64>;

    /// Dimension of the forward quadrature vector (MFEM `InitQuadIntegration`
    /// size). Zero disables the forward quadrature.
    fn quad_dim(&self) -> usize {
        0
    }

    /// Forward quadrature rate (MFEM `QuadratureIntegration`): `qdot = g(t, y)`.
    fn quadrature_integration(&self, _t: f64, _y: &[f64], _qdot: &mut [f64]) {}

    /// Backward quadrature-sensitivity rate (MFEM `QuadratureSensitivityMult`):
    /// `qBdot = (∂g/∂p)ᵀ - (∂f/∂p)ᵀ yB` evaluated with both `y` and `yB` at
    /// the same time. The vector length is problem-defined (number of
    /// parameters) and must match the `qb0` passed to
    /// [`AdjointSolver::init_adjoint`].
    fn quadrature_sensitivity_mult(&self, _t: f64, _y: &[f64], _yb: &[f64], _qbdot: &mut [f64]) {}
}

// ─── Public configuration ────────────────────────────────────────────────────

/// Interpolation of the forward solution during the backward integration
/// (CVODES `CV_HERMITE` / `CV_POLYNOMIAL`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    /// Cubic Hermite interpolation on the two endpoints of the enclosing
    /// re-integration sub-step (uses `y` and `f = y'` at both ends).
    Hermite,
    /// Evaluation of the Nordsieck (Taylor) polynomial of the enclosing
    /// re-integration sub-step (uses the stored Nordsieck vector).
    Polynomial,
}

/// Tolerances of one BDF run (CVODES `SetSVtolerances` semantics: scalar
/// relative + vector absolute).
#[derive(Clone, Debug)]
pub struct AdjointTolerances {
    /// Scalar relative tolerance.
    pub rtol: f64,
    /// Vector absolute tolerance (one entry per unknown).
    pub abstol: Vec<f64>,
}

impl AdjointTolerances {
    /// Scalar-abstol convenience constructor.
    pub fn scalar(rtol: f64, abstol: f64, n: usize) -> Self {
        AdjointTolerances { rtol, abstol: vec![abstol; n] }
    }
}

/// BDF configuration for one integration direction.
#[derive(Clone, Debug)]
pub struct AdjointBdfConfig {
    /// Local error tolerances.
    pub tol: AdjointTolerances,
    /// Initial step size.
    pub dt0: f64,
    /// Smallest allowed step.
    pub dt_min: f64,
    /// Largest allowed step.
    pub dt_max: f64,
    /// Maximum BDF order (1–6).
    pub max_order: usize,
    /// Safety cap on attempted steps.
    pub max_steps: u64,
    /// Max Newton iterations per step.
    pub newton_max_iter: u32,
}

impl AdjointBdfConfig {
    /// Config with the given tolerances and default CVODES-like controller
    /// knobs (`dt_min = 1e-14`, unbounded `dt_max`, `max_order = 5`).
    pub fn new(tol: AdjointTolerances, dt0: f64) -> Self {
        AdjointBdfConfig {
            tol,
            dt0,
            dt_min: 1e-14,
            dt_max: f64::INFINITY,
            max_order: 5,
            max_steps: 2_000_000,
            newton_max_iter: 6,
        }
    }
}

/// Full configuration of [`AdjointSolver`].
#[derive(Clone, Debug)]
pub struct AdjointConfig {
    /// Forward BDF settings.
    pub forward: AdjointBdfConfig,
    /// Adjoint (backward) BDF settings.
    pub adjoint: AdjointBdfConfig,
    /// Number of accepted forward steps between checkpoints
    /// (`InitAdjointSolve(ncheck, _)`).
    pub steps_per_checkpoint: usize,
    /// Forward-solution interpolation used during the backward integration.
    pub interpolation: Interpolation,
    /// Keep the dense solution of every forward step (全量存储 reference
    /// mode; disables checkpoint re-integration).
    pub store_all_forward_steps: bool,
}

// ─── Run statistics (PrintInfo analogue) ─────────────────────────────────────

/// Counters of one BDF run. The analogue of the CVODES statistics printed by
/// MFEM's `CVODESSolver::PrintInfo` (different counter definitions; see the
/// module docs for the trade-offs vs CVODES).
#[derive(Clone, Debug, Default)]
pub struct AdjointRunStats {
    /// Attempted steps.
    pub n_steps: u64,
    /// Accepted steps.
    pub n_accepted: u64,
    /// Rejected steps (local error test or Newton failure).
    pub n_rejected: u64,
    /// Final BDF order.
    pub final_order: usize,
    /// Final step size.
    pub final_dt: f64,
}

impl AdjointRunStats {
    /// One-line summary in the spirit of `CVODESSolver::PrintInfo`.
    pub fn summary(&self) -> String {
        format!(
            "steps: {} accepted: {} rejected: {} final order: {} final dt: {:.3e}",
            self.n_steps,
            self.n_accepted,
            self.n_rejected,
            self.final_order,
            self.final_dt,
        )
    }
}

// ─── Nordsieck BDF core ──────────────────────────────────────────────────────

/// Closure bundle describing one ODE system `y' = rhs(t, y)` for the
/// Nordsieck BDF driver.
struct BdfProblem<'b> {
    n: usize,
    rtol: f64,
    atol: Vec<f64>,
    dt_min: f64,
    dt_max: f64,
    max_order: usize,
    max_steps: u64,
    newton_max_iter: u32,
    rhs: &'b dyn Fn(f64, &[f64], &mut [f64]),
    jac: &'b dyn Fn(f64, &[f64]) -> CsrMatrix<f64>,
}

/// Nordsieck history `z = [y, h·y', h²y''/2, …]` plus controller state.
struct BdfHistory {
    t: f64,
    z: Vec<Vec<f64>>,
    order: usize,
    dt: f64,
    /// Step size the stored Nordsieck vector is currently scaled with
    /// (`z[i] = h_scale^i·y^(i)/i!`); rescaled whenever the attempted `h`
    /// differs (CVODE's step-size rescaling of the Nordsieck array).
    h_scale: f64,
    /// Step size of the last accepted step (for quadrature sub-intervals).
    last_h: f64,
    /// Correction δ of the most recent accepted step and its step size
    /// (δ ∝ h^{k+1}, must be rescaled before order comparisons).
    last_delta: Option<(Vec<f64>, f64)>,
    /// Correction δ of the previous accepted step and its step size (for the
    /// order-raising test, cf. CVODE's saved `acor` history).
    prev_delta: Option<(Vec<f64>, f64)>,
    /// Accepted steps since the last order change (order-change hysteresis).
    ord_since: u32,
    reject_streak: u32,
}

impl BdfHistory {
    fn new(t0: f64, y0: &[f64], dt0: f64, rhs: &dyn Fn(f64, &[f64], &mut [f64])) -> Self {
        let n = y0.len();
        let mut ydot = vec![0.0; n];
        rhs(t0, y0, &mut ydot);
        let z1: Vec<f64> = ydot.iter().map(|&v| dt0 * v).collect();
        BdfHistory {
            t: t0,
            z: vec![y0.to_vec(), z1],
            order: 1,
            dt: dt0,
            h_scale: dt0,
            last_h: dt0,
            last_delta: None,
            prev_delta: None,
            ord_since: 0,
            reject_streak: 0,
        }
    }
}

/// Weighted root-mean-square norm: `sqrt(mean((e/(atol+rtol|y|))²))`.
fn wrms_vec(e: &[f64], y: &[f64], atol: &[f64], rtol: f64) -> f64 {
    let mut sum = 0.0;
    for i in 0..e.len() {
        let scale = (atol[i] + rtol * y[i].abs()).max(1e-300);
        let r = e[i] / scale;
        sum += r * r;
    }
    (sum / e.len() as f64).sqrt()
}

/// Build `l1·I − h·J` as CSR.
fn newton_matrix(l1: f64, h_alpha: f64, jac: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let n = jac.nrows;
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        coo.add(i, i, l1);
    }
    for i in 0..n {
        for p in jac.row_ptr[i]..jac.row_ptr[i + 1] {
            let j = jac.col_idx[p] as usize;
            coo.add(i, j, -h_alpha * jac.values[p]);
        }
    }
    coo.into_csr()
}

/// Outcome of one attempted BDF step.
enum StepOutcome {
    /// Newton converged; carries the corrected Nordsieck vector, the
    /// correction δ and the WRMS local error estimate. The caller decides
    /// whether to commit (local error test) — `hist` is left untouched.
    Accepted {
        z_new: Vec<Vec<f64>>,
        delta: Vec<f64>,
        err: f64,
    },
    /// Newton did not converge — retry with a much smaller step.
    NewtonFailed,
}

/// One Nordsieck BDF step of size `h` from `hist.t`.
///
/// Corrector (Nordsieck delta form, cf. CVODE/BCP): with the Pascal-predicted
/// `ẑ`, solve for the correction `δ` (update `z⁺ = ẑ + l·δ`)
///
/// ```text
/// F(δ) = l1·δ + ẑ1 − h·f(t+h, ẑ0 + δ) = 0
/// ```
///
/// (with `l1 = L_COEFFS[k][1] = 1/α_k`, so the converged step satisfies the
/// Nordsieck invariant `z1⁺ = h·f`, equivalent to BDF-k), by modified Newton
/// iteration with matrix `A = l1·I − h·J`. On success `hist` is advanced to
/// `t + h` with the corrected Nordsieck vector.
fn bdf_step(prob: &BdfProblem, hist: &mut BdfHistory, h: f64) -> StepOutcome {
    let k = hist.order;
    let n = prob.n;
    let l1 = L_COEFFS[k][1];
    let tn1 = hist.t + h;

    // 1. Pascal-triangle prediction ẑ[i] = Σ_j C(j,i)·z[j].
    let mut zp: Vec<Vec<f64>> = (0..=k)
        .map(|i| {
            let mut v = vec![0.0; n];
            for j in i..=k {
                let c = pascal_coeff(k, i, j);
                if c != 0.0 {
                    for d in 0..n {
                        v[d] += c * hist.z[j][d];
                    }
                }
            }
            v
        })
        .collect();

    // Newton matrix A = l1·I − h·J (J frozen at the predicted point —
    // modified Newton; the converged step satisfies the true BDF residual).
    let jac = (prob.jac)(tn1, &zp[0]);
    let sys = newton_matrix(l1, h, &jac);
    let cfg = SolverConfig {
        rtol: 1e-11,
        atol: 0.0,
        max_iter: 500,
        verbose: false,
        ..SolverConfig::default()
    };

    // 2. Newton iteration on δ.
    let mut delta = vec![0.0; n];
    let mut y = zp[0].clone();
    let mut converged = false;
    for _ in 0..prob.newton_max_iter {
        let mut fy = vec![0.0; n];
        (prob.rhs)(tn1, &y, &mut fy);

        // Residual F = l1·δ + ẑ1 − h·f(y); right-hand side −F. (Note
        // l1 = L_COEFFS[k][1] = 1/α_k, so the converged step satisfies the
        // Nordsieck invariant z1⁺ = h·f(t+h, y⁺) — equivalent to BDF-k.)
        let mut rhs_lin = vec![0.0; n];
        for i in 0..n {
            rhs_lin[i] = -l1 * delta[i] - zp[1][i] + h * fy[i];
        }

        let mut dcorr = vec![0.0; n];
        if solve_gmres(&sys, &rhs_lin, &mut dcorr, 30, &cfg).is_err() {
            return StepOutcome::NewtonFailed;
        }
        let corr_norm = wrms_vec(&dcorr, &y, &prob.atol, prob.rtol);
        for i in 0..n {
            delta[i] += dcorr[i];
            y[i] = zp[0][i] + delta[i];
        }
        if corr_norm <= 0.33 {
            converged = true;
            break;
        }
    }
    if !converged {
        return StepOutcome::NewtonFailed;
    }

    // 3. Nordsieck update z⁺ = ẑ + l·δ, and commit.
    for (i, z_i) in zp.iter_mut().enumerate() {
        let li = L_COEFFS[k][i];
        if li != 0.0 {
            for d in 0..n {
                z_i[d] += li * delta[d];
            }
        }
    }
    // 4. Local truncation error estimate: e = ERROR_WEIGHTS[k][k]·δ.
    let ew = ERROR_WEIGHTS[k][k];
    let yerr: Vec<f64> = (0..n).map(|d| ew * delta[d]).collect();
    let err = wrms_vec(&yerr, &zp[0], &prob.atol, prob.rtol);
    StepOutcome::Accepted { z_new: zp, delta, err }
}

/// Adaptive BDF integration from `hist.t` to `t_target`.
///
/// `on_accepted` is invoked after every accepted step.
fn bdf_integrate(
    prob: &BdfProblem,
    hist: &mut BdfHistory,
    t_target: f64,
    stats: &mut AdjointRunStats,
    mut on_accepted: impl FnMut(&BdfHistory),
) -> Result<(), String> {
    // ~16 ulp of the target time: loose enough to absorb round-off in t + h,
    // tight enough that steps of any meaningful size always make progress.
    let tol_t = 16.0 * f64::EPSILON * (1.0 + t_target.abs());
    while (t_target - hist.t) > tol_t {
        if stats.n_steps >= prob.max_steps {
            return Err(format!(
                "adjoint BDF: max_steps ({}) exhausted at t = {:.6e}",
                prob.max_steps, hist.t
            ));
        }
        stats.n_steps += 1;

        let h = hist
            .dt
            .min(prob.dt_max)
            .min(t_target - hist.t)
            .max(prob.dt_min);

        let saved_z = hist.z.clone();
        let saved_h_scale = hist.h_scale;
        // Rescale the Nordsieck array to the attempted step size:
        // z[i] = h_scale^i·y^(i)/i! → z[i]·(h/h_scale)^i.
        let ratio = h / hist.h_scale;
        if ratio != 1.0 {
            let mut pow = ratio;
            for zi in hist.z.iter_mut().skip(1) {
                for v in zi.iter_mut() {
                    *v *= pow;
                }
                pow *= ratio;
            }
            hist.h_scale = h;
        }
        match bdf_step(prob, hist, h) {
            StepOutcome::Accepted { z_new, delta, err } if err <= 1.0 => {
                // Commit the accepted step atomically: Nordsieck vector,
                // time, step size and the correction history.
                hist.z = z_new;
                hist.t += h;
                hist.last_h = h;
                hist.prev_delta = hist.last_delta.take();
                hist.last_delta = Some((delta, h));
                stats.n_accepted += 1;
                hist.reject_streak = 0;
                on_accepted(hist);
                // Order selection from correction-based estimates (CVODE-like
                // Shampine-style tests; constants simplified — see module docs):
                // * raise k→k+1 when the h-rescaled change of the correction
                //   between the last two steps (∝ the h^{k+2} truncation
                //   term, i.e. the order k+1 local error) is well below the
                //   projected order-k error;
                // * lower k→k−1 when the highest stored Nordsieck component
                //   z[k] ∝ h^k·y^(k) (order k−1 truncation term) is well
                //   below it. Both tests require ≥2 steps spent at the
                //   current order (hysteresis).
                hist.ord_since += 1;
                // Raise after ≥3 steps at the current order; lower only after
                // ≥5 (z[k] accumulates from the Nordsieck corrections after a
                // raise and needs a few steps before the lower test is
                // meaningful — a freshly padded z[k+1] ≈ 0 would otherwise
                // undo the raise immediately).
                if hist.ord_since >= 3 {
                    let k = hist.order;
                    let kp1 = (k + 1) as i32;
                    let h_next = i_step_controller(h, err.max(1e-15), k as u8)
                        .clamp(prob.dt_min, prob.dt_max);
                    let est_cur = err * (h_next / h).powi(kp1);
                    let mut new_order = k;
                    let mut est_high = f64::INFINITY;
                    if k < prob.max_order {
                        if let (Some((d1, h1)), Some((d0, h0))) =
                            (&hist.last_delta, &hist.prev_delta)
                        {
                            let r1 = (h_next / h1).powi(kp1);
                            let r0 = (h_next / h0).powi(kp1);
                            let dd: Vec<f64> = d1
                                .iter()
                                .zip(d0.iter())
                                .map(|(a, b)| r1 * a - r0 * b)
                                .collect();
                            est_high = wrms_vec(&dd, &hist.z[0], &prob.atol, prob.rtol);
                            if est_high < 0.5 * est_cur {
                                new_order = k + 1;
                            }
                        }
                    }
                    if new_order == k && k > 1 && hist.ord_since >= 5 {
                        let zk_scale = (h_next / hist.h_scale).powi(k as i32) / (k as f64);
                        let zk: Vec<f64> =
                            hist.z[k].iter().map(|&v| zk_scale * v).collect();
                        let est_low = wrms_vec(&zk, &hist.z[0], &prob.atol, prob.rtol);
                        if est_low < 0.5 * est_cur && est_low < est_high {
                            new_order = k - 1;
                        }
                    }
                    if new_order != k {
                        hist.order = new_order;
                        if new_order > k {
                            // Pad the new component with its first-step
                            // estimate z[k+1] ≈ l_{k+1}·δ (a zero pad would
                            // degrade the predictor and trip the error test).
                            let pad: Vec<f64> = hist
                                .last_delta
                                .as_ref()
                                .map(|(d, _)| {
                                    d.iter()
                                        .map(|&v| L_COEFFS[new_order][new_order] * v)
                                        .collect()
                                })
                                .unwrap_or_else(|| vec![0.0; prob.n]);
                            hist.z.push(pad);
                        } else {
                            hist.z.truncate(new_order + 1);
                        }
                        hist.ord_since = 0;
                        hist.dt = i_step_controller(h, err.max(1e-15), new_order as u8)
                            .clamp(prob.dt_min, prob.dt_max);
                    } else {
                        hist.dt = h_next;
                    }
                } else {
                    hist.dt = i_step_controller(h, err.max(1e-15), hist.order as u8)
                        .clamp(prob.dt_min, prob.dt_max);
                }
            }
            StepOutcome::Accepted { .. } => {
                // Local error too large — retry with a smaller step and lower
                // the order by one (CVODE lowers one order per rejection).
                // The step was never committed; only undo the pre-step
                // Nordsieck rescale.
                stats.n_rejected += 1;
                hist.reject_streak += 1;
                hist.z = saved_z;
                hist.h_scale = saved_h_scale;
                hist.order = (hist.order - 1).max(1);
                hist.z.truncate(hist.order + 1);
                hist.dt = (h * 0.5).max(prob.dt_min);
            }
            StepOutcome::NewtonFailed => {
                stats.n_rejected += 1;
                hist.reject_streak += 1;
                hist.z = saved_z;
                hist.h_scale = saved_h_scale;
                hist.order = 1;
                hist.z.truncate(2);
                hist.dt = (h * 0.25).max(prob.dt_min);
            }
        }
        if hist.reject_streak > 60 {
            return Err(format!(
                "adjoint BDF: >60 consecutive rejections at t = {:.6e}",
                hist.t
            ));
        }
    }
    // Snap to the target within round-off.
    if (hist.t - t_target).abs() <= tol_t {
        hist.t = t_target;
    }
    Ok(())
}

// ─── Quadrature helpers ──────────────────────────────────────────────────────

/// 3-point Gauss–Legendre: abscissae `s` on `[-1, 0]` (covering the BDF step
/// that just ended at `t_new`, i.e. `[t_new − h, t_new]`) and weights (sum 1).
const GAUSS3_S: [f64; 3] = [-0.8872983346207417, -0.5, -0.1127016653792583];
const GAUSS3_W: [f64; 3] = [5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0];

/// Evaluate the Nordsieck polynomial `y(t_a + s·h) = Σ s^i·z[i]`.
fn nordsieck_eval(z: &[Vec<f64>], s: f64, out: &mut [f64]) {
    for d in 0..out.len() {
        out[d] = z[0][d];
    }
    let mut sp = s;
    for zi in &z[1..] {
        for (d, o) in out.iter_mut().enumerate() {
            *o += sp * zi[d];
        }
        sp *= s;
    }
}

// ─── Checkpoints / dense windows ─────────────────────────────────────────────

/// A dense sample of the forward solution at one accepted BDF step.
struct DensePoint {
    t: f64,
    /// RHS at `t` (needed by Hermite interpolation).
    f: Vec<f64>,
    /// Nordsieck vector centered at `t` (needed by polynomial interpolation
    /// and for cheap window extension).
    z: Vec<Vec<f64>>,
}

/// A sealed forward checkpoint: enough to restart the forward integrator
/// (CVODES stores the whole integrator state at each checkpoint).
struct Checkpoint {
    t: f64,
    y: Vec<f64>,
    f: Vec<f64>,
    z: Vec<Vec<f64>>,
    dt: f64,
    order: usize,
    /// Dense per-step samples inside the interval ending at this checkpoint
    /// (kept only in the `store_all_forward_steps` mode).
    dense: Vec<DensePoint>,
}

/// Re-integration window over one checkpoint interval: the forward problem is
/// re-integrated *forward in time from the left checkpoint*, and the
/// uncovered right-hand stretch is bisected (target = midpoint of the
/// remaining span), so the total re-integration work per interval stays
/// within 2× a full re-integration.
struct Window {
    /// (left checkpoint time, right checkpoint time).
    interval: (f64, f64),
    /// Dense points in generation (ascending-time) order.
    points: Vec<DensePoint>,
    /// Nordsieck state at `covered_from` (highest covered time) for extension.
    z_low: Vec<Vec<f64>>,
    dt_low: f64,
    order_low: usize,
    /// Lowest time with dense data available.
    covered_from: f64,
    /// Next bisection target (midpoint of the uncovered remaining span).
    next_target: f64,
}

/// Immutable-after-forward data: checkpoints plus the lazily built
/// re-integration window. The window sits in a `RefCell` so the backward
/// RHS/Jacobian closures can call [`ForwardData::solution_at`] through
/// shared references.
struct ForwardData<'a, O: TimeDependentAdjointOperator + ?Sized> {
    op: &'a O,
    t0: f64,
    tf: f64,
    interp: Interpolation,
    full_dense: bool,
    checkpoints: Vec<Checkpoint>,
    window: RefCell<Option<Window>>,
    /// Forward re-integration steps spent on checkpoint windows (overhead
    /// accounting).
    n_reint_steps: Cell<u64>,
}

impl<O: TimeDependentAdjointOperator + ?Sized> ForwardData<'_, O> {
    /// Interpolated forward solution `y(t)` for `t ∈ [t0, tf]`, using the
    /// dense storage or (lazily, with bisection) the checkpoint windows.
    fn solution_at(&self, t: f64, cfg: &AdjointBdfConfig) -> Vec<f64> {
        let cps = &self.checkpoints;
        let tf = self.tf;
        let tol_t = 16.0 * f64::EPSILON * (1.0 + tf.abs());
        if t >= tf - tol_t {
            return cps.last().expect("no checkpoints (run_forward not called?)").y.clone();
        }
        if t <= self.t0 + tol_t {
            return cps[0].y.clone();
        }
        let i = cps.partition_point(|c| c.t <= t) - 1;
        let left = &cps[i];
        let right = &cps[i + 1];
        if right.t - left.t <= tol_t {
            return left.y.clone();
        }

        // Ascending-time point list covering `t`: (t, y, f, z) views. Built
        // and consumed branch-locally so the window RefCell guard's lifetime
        // stays contained.
        if self.full_dense {
            let mut pts: Vec<(f64, &[f64], &[f64], &[Vec<f64>])> = Vec::new();
            pts.push((left.t, &left.y, &left.f, &left.z));
            for dp in &right.dense {
                pts.push((dp.t, &dp.z[0], &dp.f, &dp.z));
            }
            pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            return interpolate_points(&pts, t, self.interp);
        }
        let mut w = self.window.borrow_mut();
        let rebuild = !matches!(w.as_ref(), Some(w) if w.interval == (left.t, right.t));
        if rebuild {
            *w = Some(Window {
                interval: (left.t, right.t),
                points: Vec::new(),
                z_low: left.z.clone(),
                dt_low: left.dt,
                order_low: left.order,
                covered_from: left.t,
                next_target: left.t + (right.t - left.t) * 0.5,
            });
        }
        let op = self.op;
        let rhs = move |t: f64, y: &[f64], out: &mut [f64]| op.mult(t, y, out);
        let jac = move |t: f64, y: &[f64]| op.jac(t, y);
        let prob = BdfProblem {
            n: op.dim(),
            rtol: cfg.tol.rtol,
            atol: cfg.tol.abstol.clone(),
            dt_min: cfg.dt_min,
            dt_max: cfg.dt_max,
            max_order: cfg.max_order,
            max_steps: cfg.max_steps,
            newton_max_iter: cfg.newton_max_iter,
            rhs: &rhs,
            jac: &jac,
        };
        let mut wstats = AdjointRunStats::default();
        {
            let win = w.as_mut().unwrap();
            // Bisection extension: re-integrate the forward problem forward in
            // time from `covered_from` up to the next bisection target (never
            // beyond `t` or `right.t`).
            while win.covered_from < t - tol_t {
                let target = win
                    .next_target
                    .min(t)
                    .min(right.t)
                    .max(win.covered_from);
                if target - win.covered_from <= tol_t {
                    // Sub-tolerance stretch: snap the window up (the
                    // interpolation bracket still covers `t`).
                    win.covered_from = target;
                    break;
                }
                let mut hist = BdfHistory {
                    t: win.covered_from,
                    z: win.z_low.clone(),
                    order: win.order_low,
                    dt: win.dt_low,
                    h_scale: win.dt_low,
                    last_h: win.dt_low,
                    last_delta: None,
                    prev_delta: None,
                    ord_since: 0,
                    reject_streak: 0,
                };
                let mut newpts: Vec<DensePoint> = Vec::new();
                let steps_before = wstats.n_steps;
                bdf_integrate(&prob, &mut hist, target, &mut wstats, |hh| {
                    let mut f = vec![0.0; op.dim()];
                    rhs(hh.t, &hh.z[0], &mut f);
                    newpts.push(DensePoint { t: hh.t, f, z: hh.z.clone() });
                })
                .unwrap_or_else(|e| panic!("adjoint checkpoint re-integration failed: {e}"));
                self.n_reint_steps
                    .set(self.n_reint_steps.get() + (wstats.n_steps - steps_before));
                win.points.extend(newpts);
                win.z_low = hist.z.clone();
                win.dt_low = hist.dt;
                win.order_low = hist.order;
                win.covered_from = hist.t;
                let remaining = right.t - win.covered_from;
                win.next_target = right.t - remaining * 0.5;
                if win.covered_from >= right.t - tol_t {
                    win.covered_from = right.t;
                    break;
                }
            }
        }
        let mut pts: Vec<(f64, &[f64], &[f64], &[Vec<f64>])> = Vec::new();
        pts.push((left.t, &left.y, &left.f, &left.z));
        {
            let win = w.as_ref().unwrap();
            for dp in &win.points {
                pts.push((dp.t, &dp.z[0], &dp.f, &dp.z));
            }
        }
        pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        interpolate_points(&pts, t, self.interp)
    }
}

/// Hermite or Nordsieck-polynomial interpolation on an ascending-time point
/// list that covers `t`: `(t, y, f, z)` views.
fn interpolate_points(
    pts: &[(f64, &[f64], &[f64], &[Vec<f64>])],
    t: f64,
    mode: Interpolation,
) -> Vec<f64> {
    debug_assert!(pts.len() >= 2, "interpolation needs at least 2 points");
    let n = pts[0].1.len();
    // Bracketing pair: largest j with pts[j].t <= t (clamped).
    let j = pts.partition_point(|p| p.0 <= t).max(2) - 2;
    let (ta, ya, fa, _za) = pts[j];
    let (tb, yb, fb, zb) = pts[j + 1];
    let h = tb - ta;
    debug_assert!(h > 0.0);
    match mode {
        Interpolation::Hermite => {
            let s = (t - ta) / h;
            let s2 = s * s;
            let s3 = s2 * s;
            let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
            let h10 = s3 - 2.0 * s2 + s;
            let h01 = -2.0 * s3 + 3.0 * s2;
            let h11 = s3 - s2;
            let mut out = vec![0.0; n];
            for d in 0..n {
                out[d] = h00 * ya[d] + h01 * yb[d] + h * (h10 * fa[d] + h11 * fb[d]);
            }
            out
        }
        Interpolation::Polynomial => {
            // Nordsieck polynomial stored at the RIGHT bracket point `tb`:
            // the step producing `tb` ran from `ta` with h > 0, so its
            // Taylor expansion at tb covers [tb − h, tb]:
            // y(t) = Σ ((t − tb)/h)^i · zb[i], s ∈ [−1, 0].
            let s = (t - tb) / h;
            let mut out = vec![0.0; n];
            nordsieck_eval(zb, s, &mut out);
            out
        }
    }
}

// ─── Solver ──────────────────────────────────────────────────────────────────

/// Checkpointed adjoint time-integration driver — the equivalent of MFEM's
/// `CVODESSolver` used in adjoint mode. See the [module docs](self).
pub struct AdjointSolver<'a, O: TimeDependentAdjointOperator + ?Sized> {
    fw: ForwardData<'a, O>,
    cfg: AdjointConfig,
    n: usize,
    nb: usize,
    nq: usize,
    /// Current backward-integration time `tB`.
    tb: f64,
    /// Current adjoint state at `tb`.
    yb: Vec<f64>,
    /// Forward quadrature accumulated by [`Self::run_forward`].
    quad: Vec<f64>,
    /// Accumulated backward quadrature (dG/dp) at `tb`.
    qb: Vec<f64>,
    /// Adjoint BDF history (persists across `step_adjoint_to` calls).
    bhist: Option<BdfHistory>,
    /// Forward integration statistics.
    pub forward_stats: AdjointRunStats,
    /// Adjoint integration statistics.
    pub adjoint_stats: AdjointRunStats,
}

impl<'a, O: TimeDependentAdjointOperator + ?Sized> AdjointSolver<'a, O> {
    /// Create a solver for `op` with configuration `cfg`.
    pub fn new(op: &'a O, cfg: AdjointConfig) -> Self {
        let n = op.dim();
        let nb = op.adjoint_dim();
        let nq = op.quad_dim();
        assert_eq!(
            cfg.forward.tol.abstol.len(),
            n,
            "forward abstol must have one entry per unknown"
        );
        assert_eq!(
            cfg.adjoint.tol.abstol.len(),
            nb,
            "adjoint abstol must have one entry per adjoint unknown"
        );
        AdjointSolver {
            fw: ForwardData {
                op,
                t0: f64::NAN,
                tf: f64::NAN,
                interp: cfg.interpolation,
                full_dense: cfg.store_all_forward_steps,
                checkpoints: Vec::new(),
                window: RefCell::new(None),
                n_reint_steps: Cell::new(0),
            },
            cfg,
            n,
            nb,
            nq,
            tb: f64::NAN,
            yb: Vec::new(),
            quad: Vec::new(),
            qb: Vec::new(),
            bhist: None,
            forward_stats: AdjointRunStats::default(),
            adjoint_stats: AdjointRunStats::default(),
        }
    }

    /// Number of sealed forward checkpoints.
    pub fn n_checkpoints(&self) -> usize {
        self.fw.checkpoints.len()
    }

    /// Forward re-integration steps spent on checkpoint windows during
    /// backward runs (checkpointing overhead accounting).
    pub fn n_reint_steps(&self) -> u64 {
        self.fw.n_reint_steps.get()
    }

    /// Integrate the forward problem from `t0` to `tf`, sealing checkpoints
    /// every [`AdjointConfig::steps_per_checkpoint`] accepted steps and
    /// accumulating the forward quadrature starting from `q0`.
    pub fn run_forward(&mut self, t0: f64, tf: f64, y0: &[f64], q0: &[f64]) -> Result<(), String> {
        assert_eq!(y0.len(), self.n, "y0 size mismatch");
        assert_eq!(q0.len(), self.nq, "q0 size mismatch");
        assert!(tf > t0, "forward integration requires tf > t0");
        let op = self.fw.op;
        let cfg = self.cfg.clone();
        let n = self.n;
        let nq = self.nq;
        let keep_dense = cfg.store_all_forward_steps;
        let ncheck = cfg.steps_per_checkpoint.max(1);

        let rhs = move |t: f64, y: &[f64], out: &mut [f64]| op.mult(t, y, out);
        let jac = move |t: f64, y: &[f64]| op.jac(t, y);
        let prob = BdfProblem {
            n,
            rtol: cfg.forward.tol.rtol,
            atol: cfg.forward.tol.abstol.clone(),
            dt_min: cfg.forward.dt_min,
            dt_max: cfg.forward.dt_max,
            max_order: cfg.forward.max_order,
            max_steps: cfg.forward.max_steps,
            newton_max_iter: cfg.forward.newton_max_iter,
            rhs: &rhs,
            jac: &jac,
        };

        let mut hist = BdfHistory::new(t0, y0, cfg.forward.dt0, &rhs);
        let mut f0 = vec![0.0; n];
        rhs(t0, y0, &mut f0);

        let mut stats = AdjointRunStats::default();
        let mut quad = q0.to_vec();
        let mut dense: Vec<DensePoint> = Vec::new();
        let mut steps_in_interval = 0usize;
        let mut checkpoints: Vec<Checkpoint> = Vec::new();
        checkpoints.push(Checkpoint {
            t: t0,
            y: y0.to_vec(),
            f: f0,
            z: hist.z.clone(),
            dt: cfg.forward.dt0,
            order: 1,
            dense: Vec::new(),
        });

        let mut qdot = vec![0.0; nq.max(1)];
        let mut qtmp = vec![0.0; n];
        {
            let cp = &mut checkpoints;
            bdf_integrate(&prob, &mut hist, tf, &mut stats, |hh| {
                let h = hh.last_h;
                let mut f = vec![0.0; n];
                rhs(hh.t, &hh.z[0], &mut f);
                dense.push(DensePoint { t: hh.t, f: f.clone(), z: hh.z.clone() });
                // Forward quadrature over the accepted step [t-h, t].
                if nq > 0 {
                    for g in 0..3 {
                        let s = GAUSS3_S[g];
                        nordsieck_eval(&hh.z, s, &mut qtmp);
                        op.quadrature_integration(hh.t + s * h, &qtmp, &mut qdot);
                        for (q, qd) in quad.iter_mut().zip(qdot.iter()) {
                            *q += h * GAUSS3_W[g] * qd;
                        }
                    }
                }
                steps_in_interval += 1;
                if steps_in_interval >= ncheck {
                    steps_in_interval = 0;
                    let taken = std::mem::take(&mut dense);
                    cp.push(Checkpoint {
                        t: hh.t,
                        y: hh.z[0].clone(),
                        f,
                        z: hh.z.clone(),
                        dt: hh.dt,
                        order: hh.order,
                        dense: if keep_dense { taken } else { Vec::new() },
                    });
                }
            })?;

            // Seal the final (partial) checkpoint.
            let final_dense = std::mem::take(&mut dense);
            let mut ff = vec![0.0; n];
            rhs(hist.t, &hist.z[0], &mut ff);
            cp.push(Checkpoint {
                t: hist.t,
                y: hist.z[0].clone(),
                f: ff,
                z: hist.z.clone(),
                dt: hist.dt,
                order: hist.order,
                dense: if keep_dense { final_dense } else { Vec::new() },
            });
        }

        self.fw.t0 = t0;
        self.fw.tf = hist.t;
        self.fw.checkpoints = checkpoints;
        stats.final_order = hist.order;
        stats.final_dt = hist.dt;
        self.forward_stats = stats;
        self.quad = quad;
        Ok(())
    }

    /// Forward quadrature value (CVODES `EvalQuadIntegration`).
    pub fn quadrature(&self) -> &[f64] {
        &self.quad
    }

    /// Forward solution at the final forward time.
    pub fn forward_final(&self) -> &[f64] {
        &self.fw.checkpoints.last().expect("run_forward not called").y
    }

    /// Interpolated forward solution at an arbitrary time in `[t0, tf]`
    /// (CVODES `GetForwardSolution`).
    pub fn forward_solution_at(&self, t: f64) -> Vec<f64> {
        self.fw.solution_at(t, &self.cfg.forward)
    }

    /// Initialize the adjoint problem at `tB = tf` with the final-time adjoint
    /// value `yb0` (usually zero) and the backward quadrature `qb0` (usually
    /// zero; CVODES `InitB` + `InitQuadIntegrationB`).
    pub fn init_adjoint(&mut self, yb0: &[f64], qb0: &[f64]) {
        assert_eq!(yb0.len(), self.nb, "yb0 size mismatch");
        assert!(!self.fw.tf.is_nan(), "run_forward must be called first");
        self.tb = self.fw.tf;
        self.yb = yb0.to_vec();
        self.qb = qb0.to_vec();
        self.bhist = None;
    }

    /// Integrate the adjoint problem backward from the current `tB` down to
    /// `tb_end` (CVODES `StepB` loop). May be called repeatedly (e.g. first
    /// to an output time, then to `t0`); the BDF history persists.
    pub fn step_adjoint_to(&mut self, tb_end: f64) -> Result<(), String> {
        let tf = self.fw.tf;
        let tau_start = tf - self.tb;
        let tau_end = tf - tb_end;
        assert!(
            !self.tb.is_nan(),
            "init_adjoint must be called before step_adjoint_to"
        );
        assert!(
            tau_end > tau_start + 1e-12 * (1.0 + tf.abs()),
            "step_adjoint_to requires tb_end < current tB"
        );

        // One-time adjoint history initialization at tB = tf.
        if self.bhist.is_none() {
            let op = self.fw.op;
            let y0 = self.yb.clone();
            let mut ybdot = vec![0.0; self.nb];
            let y_here = self.fw.solution_at(self.tb, &self.cfg.forward);
            op.adjoint_rate_mult(self.tb, &y_here, &y0, &mut ybdot);
            let z1: Vec<f64> = ybdot.iter().map(|&v| self.cfg.adjoint.dt0 * v).collect();
            self.bhist = Some(BdfHistory {
                t: 0.0,
                z: vec![y0, z1],
                order: 1,
                dt: self.cfg.adjoint.dt0,
                h_scale: self.cfg.adjoint.dt0,
                last_h: self.cfg.adjoint.dt0,
                last_delta: None,
                prev_delta: None,
                ord_since: 0,
                reject_streak: 0,
            });
        }

        // Backward problem: in τ = tf − t the adjoint ODE dyB/dt = fB(t, y, yB)
        // becomes dyB/dτ = −fB(tf−τ, y(tf−τ), yB), so both the rate and its
        // Jacobian enter with a flipped sign. The forward solution y(t) is
        // fetched through the (shared) checkpoint store.
        let fw = &self.fw;
        let op = self.fw.op;
        let cfgf = self.cfg.forward.clone();
        let cfgf2 = cfgf.clone();
        let cfgf3 = cfgf.clone();
        let rhs = move |tau: f64, yb: &[f64], out: &mut [f64]| {
            let t = tf - tau;
            let y = fw.solution_at(t, &cfgf);
            op.adjoint_rate_mult(t, &y, yb, out);
            for v in out.iter_mut() {
                *v = -*v;
            }
        };
        let jac = move |tau: f64, yb: &[f64]| {
            let t = tf - tau;
            let y = fw.solution_at(t, &cfgf2);
            let j = op.adjoint_jac(t, &y, yb);
            CsrMatrix {
                nrows: j.nrows,
                ncols: j.ncols,
                row_ptr: j.row_ptr,
                col_idx: j.col_idx,
                values: j.values.iter().map(|v| -v).collect(),
            }
        };
        let cfg = self.cfg.adjoint.clone();
        let prob = BdfProblem {
            n: self.nb,
            rtol: cfg.tol.rtol,
            atol: cfg.tol.abstol.clone(),
            dt_min: cfg.dt_min,
            dt_max: cfg.dt_max,
            max_order: cfg.max_order,
            max_steps: cfg.max_steps,
            newton_max_iter: cfg.newton_max_iter,
            rhs: &rhs,
            jac: &jac,
        };

        let mut stats = std::mem::take(&mut self.adjoint_stats);
        let mut qb = std::mem::take(&mut self.qb);
        let has_q = !qb.is_empty();
        let mut ytmp = vec![0.0; self.nb];
        let mut qd = vec![0.0; qb.len()];
        {
            let hist = self.bhist.as_mut().unwrap();
            bdf_integrate(&prob, hist, tau_end, &mut stats, |hh| {
                // Backward quadrature over the accepted τ-step: the
                // corresponding t-window is [tf − t_τ, tf − (t_τ − h)] in
                // positive t orientation.
                if has_q {
                    let h = hh.last_h;
                    for g in 0..3 {
                        let s = GAUSS3_S[g];
                        nordsieck_eval(&hh.z, s, &mut ytmp);
                        let t_g = tf - (hh.t + s * h);
                        let y_g = fw.solution_at(t_g, &cfgf3);
                        op.quadrature_sensitivity_mult(t_g, &y_g, &ytmp, &mut qd);
                        for (q, qv) in qb.iter_mut().zip(qd.iter()) {
                            *q += h * GAUSS3_W[g] * qv;
                        }
                    }
                }
            })?;
            self.yb = hist.z[0].clone();
            self.tb = tf - hist.t;
            stats.final_order = hist.order;
            stats.final_dt = hist.dt;
        }
        self.adjoint_stats = stats;
        self.qb = qb;
        Ok(())
    }

    /// Current adjoint state `λ(tB)` (CVODES `w` after `StepB`).
    pub fn adjoint(&self) -> &[f64] {
        &self.yb
    }

    /// Current backward time `tB`.
    pub fn adjoint_time(&self) -> f64 {
        self.tb
    }

    /// Accumulated backward quadrature `dG/dp` at the current `tB`
    /// (CVODES `EvalQuadIntegrationB`).
    pub fn quad_sensitivity(&self) -> &[f64] {
        &self.qb
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// CSR helper from triplets.
    fn triplets(n: usize, entries: &[(usize, usize, f64)]) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for &(i, j, v) in entries {
            coo.add(i, j, v);
        }
        coo.into_csr()
    }

    /// BDF config for one direction with scalar tolerances.
    fn bdf_cfg(n: usize, rtol: f64, atol: f64, dt0: f64, dt_max: f64) -> AdjointBdfConfig {
        let mut c = AdjointBdfConfig::new(AdjointTolerances::scalar(rtol, atol, n), dt0);
        c.dt_max = dt_max;
        c
    }

    /// Linear self-adjoint heat rod: `y' = -p·L y` (L = SPD 1-D Laplacian),
    /// running cost `g(t, y) = c·y`. The adjoint is `λ' = p·L λ - c` and
    /// `dG/dp = ∫ ⟨L y, λ⟩ dt`, `dG/dy0 = λ(t0)`.
    struct HeatRod {
        p: f64,
        lap: CsrMatrix<f64>,
        c: Vec<f64>,
    }

    impl HeatRod {
        fn laplacian(n: usize) -> CsrMatrix<f64> {
            let mut e: Vec<(usize, usize, f64)> = Vec::new();
            for i in 0..n {
                e.push((i, i, 2.0));
                if i > 0 {
                    e.push((i, i - 1, -1.0));
                }
                if i + 1 < n {
                    e.push((i, i + 1, -1.0));
                }
            }
            triplets(n, &e)
        }

        fn scaled_lap(&self, s: f64) -> CsrMatrix<f64> {
            let n = self.lap.nrows;
            let mut e: Vec<(usize, usize, f64)> = Vec::new();
            for i in 0..n {
                for p in self.lap.row_ptr[i]..self.lap.row_ptr[i + 1] {
                    e.push((i, self.lap.col_idx[p] as usize, s * self.lap.values[p]));
                }
            }
            triplets(n, &e)
        }
    }

    impl TimeDependentAdjointOperator for HeatRod {
        fn dim(&self) -> usize {
            self.lap.nrows
        }
        fn mult(&self, _t: f64, y: &[f64], ydot: &mut [f64]) {
            self.lap.spmv(y, ydot);
            for v in ydot.iter_mut() {
                *v *= -self.p;
            }
        }
        fn jac(&self, _t: f64, _y: &[f64]) -> CsrMatrix<f64> {
            self.scaled_lap(-self.p)
        }
        fn adjoint_rate_mult(&self, _t: f64, _y: &[f64], yb: &[f64], ybdot: &mut [f64]) {
            self.lap.spmv(yb, ybdot);
            for (d, yd) in ybdot.iter_mut().enumerate() {
                *yd = self.p * *yd - self.c[d];
            }
        }
        fn adjoint_jac(&self, _t: f64, _y: &[f64], _yb: &[f64]) -> CsrMatrix<f64> {
            self.scaled_lap(self.p)
        }
        fn quad_dim(&self) -> usize {
            1
        }
        fn quadrature_integration(&self, _t: f64, y: &[f64], qdot: &mut [f64]) {
            qdot[0] = self.c.iter().zip(y).map(|(&c, &y)| c * y).sum();
        }
        fn quadrature_sensitivity_mult(&self, _t: f64, y: &[f64], yb: &[f64], qbdot: &mut [f64]) {
            // qBdot = (∂g/∂p)ᵀ + (∂f/∂p)ᵀ λ = (−L y)ᵀ λ = −⟨L y, λ⟩
            // (CVODES convention: dG/dp = ∫ qBdot dt, cf. the Roberts C++
            // miniapp whose qBdot = +f_pᵀλ).
            let mut ly = vec![0.0; y.len()];
            self.lap.spmv(y, &mut ly);
            qbdot[0] = -ly.iter().zip(yb).map(|(&l, &b)| l * b).sum::<f64>();
        }
    }

    fn rod(p: f64, n: usize) -> HeatRod {
        HeatRod { p, lap: HeatRod::laplacian(n), c: vec![1.0; n] }
    }

    fn rod_solver<'a>(
        op: &'a HeatRod,
        t_final: f64,
        interp: Interpolation,
        ckpt: usize,
        dense: bool,
    ) -> AdjointSolver<'a, HeatRod> {
        let n = op.dim();
        let cfg = AdjointConfig {
            forward: bdf_cfg(n, 1e-10, 1e-12, 1e-4, t_final),
            adjoint: bdf_cfg(n, 1e-10, 1e-12, 1e-4, t_final),
            steps_per_checkpoint: ckpt,
            interpolation: interp,
            store_all_forward_steps: dense,
        };
        AdjointSolver::new(op, cfg)
    }

    /// Forward + backward run; returns (G, λ(0), dG/dp).
    fn run_rod(
        op: &HeatRod,
        t_final: f64,
        interp: Interpolation,
        ckpt: usize,
        dense: bool,
    ) -> (f64, Vec<f64>, f64) {
        let n = op.dim();
        let mut solver = rod_solver(op, t_final, interp, ckpt, dense);
        let y0 = vec![1.0; n];
        solver.run_forward(0.0, t_final, &y0, &[0.0]).unwrap();
        let g = solver.quadrature()[0];
        solver.init_adjoint(&vec![0.0; n], &[0.0]);
        solver.step_adjoint_to(0.0).unwrap();
        let l0 = solver.adjoint().to_vec();
        let dgdp = solver.quad_sensitivity()[0];
        (g, l0, dgdp)
    }

    /// 线性自伴随问题：sensitivity = vᵀλ(0) 对照复合步进有限差分。
    #[test]
    fn linear_self_adjoint_sensitivity_vs_direct_differentiation() {
        let n = 6;
        let t_final = 0.5;
        let (_, l0, _) = run_rod(&rod(1.0, n), t_final, Interpolation::Hermite, 3, false);
        // NOTE: v must be asymmetric — the symmetric rod gives a symmetric
        // λ(0), and a direction with v[i] = v[n−1-i] cancels exactly.
        let v: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).sin().exp()).collect();
        let vt_l0: f64 = v.iter().zip(&l0).map(|(&a, &b)| a * b).sum();

        // G(y0 ± ε·v) by re-running the forward problem (compound stepping).
        let goal = |dy: &[f64]| -> f64 {
            let op = rod(1.0, n);
            let mut solver = rod_solver(&op, t_final, Interpolation::Hermite, 3, false);
            let y0: Vec<f64> = (0..n).map(|i| 1.0 + dy[i]).collect();
            solver.run_forward(0.0, t_final, &y0, &[0.0]).unwrap();
            solver.quadrature()[0]
        };
        let eps = 1e-6;
        let plus: Vec<f64> = (0..n).map(|i| eps * v[i]).collect();
        let minus: Vec<f64> = plus.iter().map(|x| -x).collect();
        let fd = (goal(&plus) - goal(&minus)) / (2.0 * eps);
        let rel = (vt_l0 - fd).abs() / fd.abs();
        assert!(
            rel < 1e-7,
            "vᵀλ(0)={vt_l0:.12e} vs FD dG/dy0={fd:.12e} (rel {rel:.2e})"
        );
    }

    /// 伴随 dG/dp（quadrature-sensitivity 累积）对照参数 p 的中心差分。
    #[test]
    fn heat_parameter_sensitivity_vs_finite_difference() {
        let n = 6;
        let t_final = 0.5;
        let p0 = 1.0;
        let (_, _, dgdp_adj) = run_rod(&rod(p0, n), t_final, Interpolation::Hermite, 3, false);

        let eps = 1e-6 * p0;
        let g_plus = run_rod(&rod(p0 + eps, n), t_final, Interpolation::Hermite, 3, false).0;
        let g_minus = run_rod(&rod(p0 - eps, n), t_final, Interpolation::Hermite, 3, false).0;
        let fd = (g_plus - g_minus) / (2.0 * eps);
        let rel = (dgdp_adj - fd).abs() / fd.abs();
        assert!(
            rel < 1e-6,
            "adjoint dG/dp={dgdp_adj:.12e} vs FD={fd:.12e} (rel {rel:.2e})"
        );
    }

    /// 非线性 2 变量系统（Robertson 风格），检查点二分 vs 全量存储必须一致。
    struct TwoSpecies {
        p: f64,
    }

    impl TimeDependentAdjointOperator for TwoSpecies {
        fn dim(&self) -> usize {
            2
        }
        // y1' = -p·y1 + y2²,  y2' = p·y1 - 2·y2
        fn mult(&self, _t: f64, y: &[f64], ydot: &mut [f64]) {
            ydot[0] = -self.p * y[0] + y[1] * y[1];
            ydot[1] = self.p * y[0] - 2.0 * y[1];
        }
        fn jac(&self, _t: f64, _y: &[f64]) -> CsrMatrix<f64> {
            triplets(
                2,
                &[(0, 0, -self.p), (0, 1, 0.0), (1, 0, self.p), (1, 1, -2.0)],
            )
        }
        // λ' = -Jᵀλ - (∂g/∂y)ᵀ, g = y1:
        // λ1' = p·λ1 - p·λ2 - 1,  λ2' = -2·y2·λ1 + 2·λ2
        fn adjoint_rate_mult(&self, _t: f64, y: &[f64], yb: &[f64], ybdot: &mut [f64]) {
            ybdot[0] = self.p * yb[0] - self.p * yb[1] - 1.0;
            ybdot[1] = -2.0 * y[1] * yb[0] + 2.0 * yb[1];
        }
        fn adjoint_jac(&self, _t: f64, y: &[f64], _yb: &[f64]) -> CsrMatrix<f64> {
            triplets(
                2,
                &[
                    (0, 0, self.p),
                    (0, 1, -self.p),
                    (1, 0, -2.0 * y[1]),
                    (1, 1, 2.0),
                ],
            )
        }
        fn quad_dim(&self) -> usize {
            1
        }
        fn quadrature_integration(&self, _t: f64, y: &[f64], qdot: &mut [f64]) {
            qdot[0] = y[0];
        }
        // qBdot = (∂g/∂p)ᵀ + (∂f/∂p)ᵀλ = (−y1, y1)·λ = y1·(λ2 − λ1)
        fn quadrature_sensitivity_mult(&self, _t: f64, y: &[f64], yb: &[f64], qbdot: &mut [f64]) {
            qbdot[0] = y[0] * (yb[1] - yb[0]);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn two_species_cfg(
        t_final: f64,
        rtol: f64,
        atol: f64,
        interp: Interpolation,
        ckpt: usize,
        dense: bool,
    ) -> AdjointConfig {
        AdjointConfig {
            forward: bdf_cfg(2, rtol, atol, 1e-5, t_final),
            adjoint: bdf_cfg(2, rtol, atol, 1e-5, t_final),
            steps_per_checkpoint: ckpt,
            interpolation: interp,
            store_all_forward_steps: dense,
        }
    }

    /// TwoSpecies 的一次完整伴随求解：返回 (G, λ(0), dG/dp)。
    #[allow(clippy::too_many_arguments)]
    fn run_two_species(
        p: f64,
        t_final: f64,
        rtol: f64,
        atol: f64,
        interp: Interpolation,
        ckpt: usize,
        dense: bool,
    ) -> (f64, Vec<f64>, f64) {
        let op = TwoSpecies { p };
        let cfg = two_species_cfg(t_final, rtol, atol, interp, ckpt, dense);
        let mut solver = AdjointSolver::new(&op, cfg);
        let y0 = vec![1.0, 0.0];
        solver.run_forward(0.0, t_final, &y0, &[0.0]).unwrap();
        let g = solver.quadrature()[0];
        solver.init_adjoint(&[0.0, 0.0], &[0.0]);
        solver.step_adjoint_to(0.0).unwrap();
        (g, solver.adjoint().to_vec(), solver.quad_sensitivity()[0])
    }

    /// 检查点二分 vs 全量存储：λ(0) 与 dG/dp 必须一致（插值精度内）。
    #[test]
    fn checkpoint_bisection_matches_full_storage() {
        let t_final = 0.5;
        // 全量存储（无重积分）
        let (g_dense, l_dense, grad_dense) =
            run_two_species(2.0, t_final, 1e-9, 1e-12, Interpolation::Hermite, 50, true);
        // 检查点二分（每 2 步一个 checkpoint，区间内重积分 + 二分扩展）
        let (g_ckpt, l_ckpt, grad_ckpt) =
            run_two_species(2.0, t_final, 1e-9, 1e-12, Interpolation::Hermite, 2, false);

        // The checkpoint path sees Hermite-interpolated forward data in the
        // backward RHS; agreement to interpolation accuracy (~1e-7 here) is
        // the consistency requirement.
        for i in 0..2 {
            let rel = (l_ckpt[i] - l_dense[i]).abs() / l_dense[i].abs().max(1e-300);
            assert!(
                rel < 1e-6,
                "λ(0)[{i}]: dense={:.12e} ckpt={:.12e} (rel {rel:.2e})",
                l_dense[i],
                l_ckpt[i]
            );
        }
        let relg = (grad_ckpt - grad_dense).abs() / grad_dense.abs();
        assert!(
            relg < 1e-6,
            "dG/dp: dense={grad_dense:.12e} ckpt={grad_ckpt:.12e} (rel {relg:.2e})"
        );
        let relq = (g_ckpt - g_dense).abs() / g_dense.abs();
        assert!(relq < 1e-10, "G: dense={g_dense:.12e} ckpt={g_ckpt:.12e} (rel {relq:.2e})");
    }

    /// Polynomial 插值模式（CV_POLYNOMIAL 对位）与 Hermite 一样给出一致的梯度。
    #[test]
    fn checkpoint_bisection_polynomial_interpolation_agrees() {
        let t_final = 0.5;
        let (_, l_dense, grad_dense) =
            run_two_species(2.0, t_final, 1e-9, 1e-12, Interpolation::Hermite, 50, true);
        let (_, l_poly, grad_poly) =
            run_two_species(2.0, t_final, 1e-9, 1e-12, Interpolation::Polynomial, 2, false);
        for i in 0..2 {
            let rel = (l_poly[i] - l_dense[i]).abs() / l_dense[i].abs().max(1e-300);
            assert!(
                rel < 5e-5,
                "Polynomial λ(0)[{i}]: dense={:.12e} poly={:.12e} (rel {rel:.2e})",
                l_dense[i],
                l_poly[i]
            );
        }
        let relg = (grad_poly - grad_dense).abs() / grad_dense.abs();
        assert!(
            relg < 5e-5,
            "Polynomial dG/dp: dense={grad_dense:.12e} poly={grad_poly:.12e} (rel {relg:.2e})"
        );
    }

    /// 分段 backward 积分（先 T→T/2 再 T/2→0）与一步到 0 一致。
    #[test]
    fn split_backward_run_matches_single_run() {
        let t_final = 0.5;
        let op = TwoSpecies { p: 2.0 };
        let rtol = 1e-9;
        let atol = 1e-12;
        let (_, l_single, grad_single) =
            run_two_species(2.0, t_final, rtol, atol, Interpolation::Hermite, 50, false);

        let cfg = two_species_cfg(t_final, rtol, atol, Interpolation::Hermite, 50, false);
        let mut solver = AdjointSolver::new(&op, cfg);
        solver
            .run_forward(0.0, t_final, &[1.0, 0.0], &[0.0])
            .unwrap();
        solver.init_adjoint(&[0.0, 0.0], &[0.0]);
        solver.step_adjoint_to(t_final / 2.0).unwrap();
        solver.step_adjoint_to(0.0).unwrap();
        for i in 0..2 {
            let rel = (solver.adjoint()[i] - l_single[i]).abs() / l_single[i].abs().max(1e-300);
            // Splitting the backward run changes the BDF step sequence; the
            // two paths agree to the integration tolerance, not bitwise.
            assert!(rel < 1e-7, "split λ(0)[{i}] mismatch (rel {rel:.2e})");
        }
        let relg = (solver.quad_sensitivity()[0] - grad_single).abs() / grad_single.abs();
        assert!(relg < 1e-7, "split dG/dp mismatch (rel {relg:.2e})");
    }

    /// GetForwardSolution：检查点二分的插值解与全量存储一致到插值精度；
    /// Hermite 与 Polynomial 插值互相一致。
    #[test]
    fn forward_solution_interpolation_accuracy() {
        let t_final = 0.5;
        let op = TwoSpecies { p: 2.0 };
        // 全量存储参考解。
        let cfg_ref = two_species_cfg(t_final, 1e-9, 1e-12, Interpolation::Hermite, 50, true);
        let solver_ref = AdjointSolver::new(&op, cfg_ref);
        let mut sref = solver_ref;
        sref.run_forward(0.0, t_final, &[1.0, 0.0], &[0.0]).unwrap();

        // 检查点二分 + Hermite。
        let cfg_h = two_species_cfg(t_final, 1e-9, 1e-12, Interpolation::Hermite, 2, false);
        let mut sh = AdjointSolver::new(&op, cfg_h);
        sh.run_forward(0.0, t_final, &[1.0, 0.0], &[0.0]).unwrap();

        // 检查点二分 + Polynomial。
        let cfg_p = two_species_cfg(t_final, 1e-9, 1e-12, Interpolation::Polynomial, 2, false);
        let mut sp = AdjointSolver::new(&op, cfg_p);
        sp.run_forward(0.0, t_final, &[1.0, 0.0], &[0.0]).unwrap();

        let t_probe = [0.37, 0.123, 0.4567, 0.01, 0.4999];
        for &tp in &t_probe {
            let y_ref = sref.forward_solution_at(tp);
            let y_h = sh.forward_solution_at(tp);
            let y_p = sp.forward_solution_at(tp);
            for i in 0..2 {
                let relh = (y_h[i] - y_ref[i]).abs() / y_ref[i].abs().max(1e-300);
                assert!(relh < 1e-7, "Hermite y({tp})[{i}]: ref={:.12e} got={:.12e} (rel {relh:.2e})", y_ref[i], y_h[i]);
                let relp = (y_p[i] - y_ref[i]).abs() / y_ref[i].abs().max(1e-300);
                assert!(relp < 1e-7, "Polynomial y({tp})[{i}]: ref={:.12e} got={:.12e} (rel {relp:.2e})", y_ref[i], y_p[i]);
            }
        }
    }
}
