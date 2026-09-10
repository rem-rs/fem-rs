//! Additional MFEM-matched Runge–Kutta and multistep integrators (1:1 port of
//! MFEM `linalg/ode.hpp` / `ode.cpp` classes that were missing from this crate).
//!
//! Explicit: [`Rk2`] (family with parameter `a`), [`Rk6`] and [`Rk8`] (Verner
//! "efficient" pairs, via a generic explicit tableau stepper), [`Ab5`]
//! (5-step Adams–Bashforth with RK6 startup).
//!
//! Implicit: [`ImplicitMidpoint`], [`Sdirk34`], [`Sdirk33`], [`Esdirk32`],
//! [`Esdirk33`] — all solve `(I − γ·dt·J) k = f(t_stage, u_stage)` per stage,
//! matching MFEM's `TimeDependentOperator::ImplicitSolve` slope semantics.

use super::implicit::identity_minus_dt_jac;
use super::traits::{ImplicitTimeStepper, TimeStepper};
use crate::{solve_gmres, SolverConfig};

// ─── Shared: implicit stage solve ────────────────────────────────────────────

/// Solve one DIRK stage for the slope `k` in MFEM `ImplicitSolve(gamma*dt, x, k)`
/// semantics: `(I − gamma·dt·J(t_stage, u_stage)) k = f(t_stage, u_stage)`.
fn solve_stage_slope<F, J>(rhs: &F, jac_fn: &J, t_stage: f64, u_stage: &[f64], gamma_dt: f64, k: &mut [f64])
where
    F: Fn(f64, &[f64], &mut [f64]),
    J: Fn(f64, &[f64]) -> fem_linalg::CsrMatrix<f64>,
{
    let n = u_stage.len();
    let jac = jac_fn(t_stage, u_stage);
    let sys = identity_minus_dt_jac(&jac, gamma_dt);
    let mut b = vec![0.0_f64; n];
    rhs(t_stage, u_stage, &mut b);
    let cfg = SolverConfig {
        rtol: 1e-10,
        atol: 0.0,
        max_iter: 500,
        verbose: false,
        ..SolverConfig::default()
    };
    k.fill(0.0);
    solve_gmres(&sys, &b, k, 30, &cfg).expect("DIRK stage linear solve failed");
}

// ─── RK2 (family of 2nd-order explicit methods) ──────────────────────────────

/// A family of explicit second-order RK2 methods (MFEM `RK2Solver`).
///
/// Butcher tableau
/// ```text
///   0   |
///   a   |  a
/// ------+--------
///       | 1-b  b      b = 1/(2a)
/// ```
/// Some choices for the parameter `a`:
/// - `a = 1/2` — the midpoint method ([`Rk2::midpoint`]),
/// - `a = 1`   — Heun's method ([`Rk2::heun`]),
/// - `a = 2/3` — default, has minimal truncation error ([`Rk2::default`]).
#[derive(Debug, Clone, Copy)]
pub struct Rk2 {
    /// Stage time fraction / sub-step weight; must be nonzero.
    pub a: f64,
}

impl Default for Rk2 {
    fn default() -> Self {
        Rk2 { a: 2.0 / 3.0 }
    }
}

impl Rk2 {
    /// Midpoint method (`a = 1/2`).
    pub fn midpoint() -> Self {
        Rk2 { a: 0.5 }
    }

    /// Heun's method (`a = 1`).
    pub fn heun() -> Self {
        Rk2 { a: 1.0 }
    }
}

impl TimeStepper for Rk2 {
    fn step<F>(&self, t: f64, dt: f64, u: &mut [f64], rhs: F)
    where
        F: Fn(f64, &[f64], &mut [f64]),
    {
        let n = u.len();
        let b = 0.5 / self.a;
        let mut dxdt = vec![0.0_f64; n];
        let mut x1 = vec![0.0_f64; n];

        rhs(t, u, &mut dxdt);
        for i in 0..n {
            x1[i] = u[i] + (1.0 - b) * dt * dxdt[i]; // x1 = x + (1-b)*dt*k1
            u[i] += self.a * dt * dxdt[i]; // x  = x + a*dt*k1 (stage state)
        }
        rhs(t + self.a * dt, u, &mut dxdt);
        for i in 0..n {
            u[i] = x1[i] + b * dt * dxdt[i]; // x = x1 + b*dt*k2
        }
    }
}

// ─── Generic explicit tableau stepper (MFEM `ExplicitRKSolver`) ──────────────

/// Advance one step with a generic explicit RK tableau (MFEM `ExplicitRKSolver`).
///
/// Tableau layout (row `i` of `a` has `i` entries, packed row-major):
/// ```text
///   0     |
///   c[0]  | a[0]
///   c[1]  | a[1] a[2]
///   ...   |   ...
///   c[s-2]|  ...  a[s(s-1)/2-1]
/// --------+---------------------
///         | b[0] b[1] ... b[s-1]
/// ```
fn explicit_tableau_step<F>(a: &[f64], b: &[f64], c: &[f64], t: f64, dt: f64, u: &mut [f64], rhs: &F)
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    let n = u.len();
    let s = b.len();
    let mut k: Vec<Vec<f64>> = (0..s).map(|_| vec![0.0_f64; n]).collect();
    let mut y = vec![0.0_f64; n];

    rhs(t, u, &mut k[0]);
    let mut l = 0_usize;
    for i in 1..s {
        y.copy_from_slice(u);
        for (yi, k0) in y.iter_mut().zip(&k[0]) {
            *yi += a[l] * dt * k0;
        }
        l += 1;
        for j in 1..i {
            for (yi, kj) in y.iter_mut().zip(&k[j]) {
                *yi += a[l] * dt * kj;
            }
            l += 1;
        }
        rhs(t + c[i - 1] * dt, &y, &mut k[i]);
    }
    for (i, ki) in k.iter().enumerate() {
        for (xi, kij) in u.iter_mut().zip(ki) {
            *xi += b[i] * dt * kij;
        }
    }
}

// ─── RK6: 8-stage, 6th order (Verner "efficient" 9-stage 6(5) pair) ─────────

/// An 8-stage, 6th order explicit RK method (MFEM `RK6Solver`).
///
/// From Verner's "efficient" 9-stage 6(5) pair.
pub struct Rk6;

/// `RK6Solver::a` — packed lower-triangular A (28 entries).
const RK6_A: [f64; 28] = [
    0.6e-1,
    0.1923996296296296296296296296296296296296e-1,
    0.7669337037037037037037037037037037037037e-1,
    0.35975e-1,
    0.,
    0.107925,
    1.318683415233148260919747276431735612861,
    0.,
    -5.042058063628562225427761634715637693344,
    4.220674648395413964508014358283902080483,
    -41.87259166432751461803757780644346812905,
    0.,
    159.4325621631374917700365669070346830453,
    -122.1192135650100309202516203389242140663,
    5.531743066200053768252631238332999150076,
    -54.43015693531650433250642051294142461271,
    0.,
    207.0672513650184644273657173866509835987,
    -158.6108137845899991828742424365058599469,
    6.991816585950242321992597280791793907096,
    -0.1859723106220323397765171799549294623692e-1,
    -54.66374178728197680241215648050386959351,
    0.,
    207.9528062553893734515824816699834244238,
    -159.2889574744995071508959805871426654216,
    7.018743740796944434698170760964252490817,
    -0.1833878590504572306472782005141738268361e-1,
    -0.5119484997882099077875432497245168395840e-3,
];

/// `RK6Solver::b` — output weights (8 entries).
const RK6_B: [f64; 8] = [
    0.3438957868357036009278820124728322386520e-1,
    0.,
    0.,
    0.2582624555633503404659558098586120858767,
    0.4209371189673537150642551514069801967032,
    4.405396469669310170148836816197095664891,
    -176.4831190242986576151740942499002125029,
    172.3641334014150730294022582711902413315,
];

/// `RK6Solver::c` — abscissae for stages 1..7 (stage 0 is at `t`).
const RK6_C: [f64; 7] = [
    0.6e-1,
    0.9593333333333333333333333333333333333333e-1,
    0.1439,
    0.4973,
    0.9725,
    0.9995,
    1.,
];

impl TimeStepper for Rk6 {
    fn step<F>(&self, t: f64, dt: f64, u: &mut [f64], rhs: F)
    where
        F: Fn(f64, &[f64], &mut [f64]),
    {
        explicit_tableau_step(&RK6_A, &RK6_B, &RK6_C, t, dt, u, &rhs);
    }
}

// ─── RK8: 12-stage, 8th order (Verner "efficient" 13-stage 8(7) pair) ───────

/// A 12-stage, 8th order explicit RK method (MFEM `RK8Solver`).
///
/// From Verner's "efficient" 13-stage 8(7) pair.
pub struct Rk8;

/// `RK8Solver::a` — packed lower-triangular A (66 entries).
const RK8_A: [f64; 66] = [
    0.5e-1,
    -0.69931640625e-2,
    0.1135556640625,
    0.399609375e-1,
    0.,
    0.1198828125,
    0.3613975628004575124052940721184028345129,
    0.,
    -1.341524066700492771819987788202715834917,
    1.370126503900035259414693716084313000404,
    0.490472027972027972027972027972027972028e-1,
    0.,
    0.,
    0.2350972042214404739862988335493427143122,
    0.180855592981356728810903963653454488485,
    0.6169289044289044289044289044289044289044e-1,
    0.,
    0.,
    0.1123656831464027662262557035130015442303,
    -0.3885046071451366767049048108111244567456e-1,
    0.1979188712522045855379188712522045855379e-1,
    -1.767630240222326875735597119572145586714,
    0.,
    0.,
    -62.5,
    -6.061889377376669100821361459659331999758,
    5.650823198222763138561298030600840174201,
    65.62169641937623283799566054863063741227,
    -1.180945066554970799825116282628297957882,
    0.,
    0.,
    -41.50473441114320841606641502701994225874,
    -4.434438319103725011225169229846100211776,
    4.260408188586133024812193710744693240761,
    43.75364022446171584987676829438379303004,
    0.787142548991231068744647504422630755086e-2,
    -1.281405999441488405459510291182054246266,
    0.,
    0.,
    -45.04713996013986630220754257136007322267,
    -4.731362069449576477311464265491282810943,
    4.514967016593807841185851584597240996214,
    47.44909557172985134869022392235929015114,
    0.1059228297111661135687393955516542875228e-1,
    -0.5746842263844616254432318478286296232021e-2,
    -1.724470134262485191756709817484481861731,
    0.,
    0.,
    -60.92349008483054016518434619253765246063,
    -5.95151837622239245520283276706185486829,
    5.556523730698456235979791650843592496839,
    63.98301198033305336837536378635995939281,
    0.1464202825041496159275921391759452676003e-1,
    0.6460408772358203603621865144977650714892e-1,
    -0.7930323169008878984024452548693373291447e-1,
    -3.301622667747079016353994789790983625569,
    0.,
    0.,
    -118.011272359752508566692330395789886851,
    -10.14142238845611248642783916034510897595,
    9.139311332232057923544012273556827000619,
    123.3759428284042683684847180986501894364,
    4.623244378874580474839807625067630924792,
    -3.383277738068201923652550971536811240814,
    4.527592100324618189451265339351129035325,
    -5.828495485811622963193088019162985703755,
];

/// `RK8Solver::b` — output weights (12 entries).
const RK8_B: [f64; 12] = [
    0.4427989419007951074716746668098518862111e-1,
    0.,
    0.,
    0.,
    0.,
    0.3541049391724448744815552028733568354121,
    0.2479692154956437828667629415370663023884,
    -15.69420203883808405099207034271191213468,
    25.08406496555856261343930031237186278518,
    -31.73836778626027646833156112007297739997,
    22.93828327398878395231483560344797018313,
    -0.2361324633071542145259900641263517600737,
];

/// `RK8Solver::c` — abscissae for stages 1..11 (stage 0 is at `t`).
const RK8_C: [f64; 11] = [
    0.5e-1,
    0.1065625,
    0.15984375,
    0.39,
    0.465,
    0.155,
    0.943,
    0.901802041735856958259707940678372149956,
    0.909,
    0.94,
    1.,
];

impl TimeStepper for Rk8 {
    fn step<F>(&self, t: f64, dt: f64, u: &mut [f64], rhs: F)
    where
        F: Fn(f64, &[f64], &mut [f64]),
    {
        explicit_tableau_step(&RK8_A, &RK8_B, &RK8_C, t, dt, u, &rhs);
    }
}

// ─── Implicit midpoint ───────────────────────────────────────────────────────

/// Implicit midpoint method (MFEM `ImplicitMidpointSolver`).
///
/// `u_{n+1} = u_n + dt·f(t_n + dt/2, (u_n + u_{n+1})/2)`, i.e. the one-stage
/// implicit RK with `c = γ = b = 1/2`.  Second order, A-stable, not L-stable;
/// it also preserves quadratic invariants and is symplectic for Hamiltonian
/// systems.
pub struct ImplicitMidpoint;

impl ImplicitTimeStepper for ImplicitMidpoint {
    fn step_implicit<F, J>(&self, t: f64, dt: f64, u: &mut [f64], rhs: F, jac_fn: J)
    where
        F: Fn(f64, &[f64], &mut [f64]),
        J: Fn(f64, &[f64]) -> fem_linalg::CsrMatrix<f64>,
    {
        let n = u.len();
        let mut k = vec![0.0_f64; n];
        solve_stage_slope(&rhs, &jac_fn, t + dt / 2.0, u, dt / 2.0, &mut k);
        for (xi, ki) in u.iter_mut().zip(&k) {
            *xi += dt * ki;
        }
    }
}

// ─── SDIRK34: 3-stage, 4th order SDIRK (A-stable, not L-stable) ──────────────

/// Three stage, singly diagonal implicit Runge–Kutta (SDIRK) method of order 4
/// (MFEM `SDIRK34Solver`).  A-stable, not L-stable.
///
/// Butcher tableau
/// ```text
///    a   |    a
///   1/2  |  1/2-a    a
///   1-a  |   2a    1-4a    a
/// -------+--------------------
///        |    b     1-2b    b
/// ```
/// with `a = 1/√3·cos(π/18) + 1/2 > 1` (two stage times fall outside [t, t+dt]).
pub struct Sdirk34;

impl ImplicitTimeStepper for Sdirk34 {
    fn step_implicit<F, J>(&self, t: f64, dt: f64, u: &mut [f64], rhs: F, jac_fn: J)
    where
        F: Fn(f64, &[f64], &mut [f64]),
        J: Fn(f64, &[f64]) -> fem_linalg::CsrMatrix<f64>,
    {
        let n = u.len();
        let a = 1.0 / 3.0_f64.sqrt() * (std::f64::consts::PI / 18.0).cos() + 0.5;
        let b = 1.0 / (6.0 * (2.0 * a - 1.0) * (2.0 * a - 1.0));

        let mut k = vec![0.0_f64; n];
        let mut y = vec![0.0_f64; n];
        let mut z = vec![0.0_f64; n];

        // Stage 1 at t + a*dt
        solve_stage_slope(&rhs, &jac_fn, t + a * dt, u, a * dt, &mut k);
        for i in 0..n {
            y[i] = u[i] + (0.5 - a) * dt * k[i];
            z[i] = u[i] + 2.0 * a * dt * k[i];
            u[i] += b * dt * k[i];
        }

        // Stage 2 at t + dt/2
        solve_stage_slope(&rhs, &jac_fn, t + dt / 2.0, &y, a * dt, &mut k);
        for i in 0..n {
            z[i] += (1.0 - 4.0 * a) * dt * k[i];
            u[i] += (1.0 - 2.0 * b) * dt * k[i];
        }

        // Stage 3 at t + (1-a)*dt
        solve_stage_slope(&rhs, &jac_fn, t + (1.0 - a) * dt, &z, a * dt, &mut k);
        for i in 0..n {
            u[i] += b * dt * k[i];
        }
    }
}

// ─── SDIRK33: 3-stage, 3rd order SDIRK (L-stable) ────────────────────────────

/// Three stage, singly diagonal implicit Runge–Kutta (SDIRK) method of order 3
/// (MFEM `SDIRK33Solver`).  L-stable.
///
/// Butcher tableau
/// ```text
///    a  |   a
///    c  |  c-a    a
///    1  |   b    1-a-b   a
/// ------+----------------
///       |   b    1-a-b   a
/// ```
pub struct Sdirk33;

impl ImplicitTimeStepper for Sdirk33 {
    fn step_implicit<F, J>(&self, t: f64, dt: f64, u: &mut [f64], rhs: F, jac_fn: J)
    where
        F: Fn(f64, &[f64], &mut [f64]),
        J: Fn(f64, &[f64]) -> fem_linalg::CsrMatrix<f64>,
    {
        let n = u.len();
        let a = 0.435866521508458999416019_f64;
        let b = 1.20849664917601007033648_f64;
        let c = 0.717933260754229499708010_f64;

        let mut k = vec![0.0_f64; n];
        let mut y = vec![0.0_f64; n];

        // Stage 1 at t + a*dt
        solve_stage_slope(&rhs, &jac_fn, t + a * dt, u, a * dt, &mut k);
        for i in 0..n {
            y[i] = u[i] + (c - a) * dt * k[i];
            u[i] += b * dt * k[i];
        }

        // Stage 2 at t + c*dt
        solve_stage_slope(&rhs, &jac_fn, t + c * dt, &y, a * dt, &mut k);
        for i in 0..n {
            u[i] += (1.0 - a - b) * dt * k[i];
        }

        // Stage 3 at t + dt
        solve_stage_slope(&rhs, &jac_fn, t + dt, u, a * dt, &mut k);
        for i in 0..n {
            u[i] += a * dt * k[i];
        }
    }
}

// ─── ESDIRK32: 3-stage, 2nd order ESDIRK (L-stable) ──────────────────────────

/// Three stage, explicitly (first stage) diagonal implicit Runge–Kutta (ESDIRK)
/// method of order 2 (MFEM `ESDIRK32Solver`).  L-stable.
///
/// Butcher tableau
/// ```text
///    0   |    0      0     0
///    2a  |    a      a     0
///    1   |  1-b-a    b     a
/// -------+--------------------
///        |  1-b-a    b     a
/// ```
pub struct Esdirk32;

impl ImplicitTimeStepper for Esdirk32 {
    fn step_implicit<F, J>(&self, t: f64, dt: f64, u: &mut [f64], rhs: F, jac_fn: J)
    where
        F: Fn(f64, &[f64], &mut [f64]),
        J: Fn(f64, &[f64]) -> fem_linalg::CsrMatrix<f64>,
    {
        let n = u.len();
        let a = (2.0 - 2.0_f64.sqrt()) / 2.0;
        let b = (1.0 - 2.0 * a) / (4.0 * a);

        let mut k = vec![0.0_f64; n];
        let mut y = vec![0.0_f64; n];
        let mut z = vec![0.0_f64; n];

        // Stage 1 (explicit) at t
        rhs(t, u, &mut k);
        for i in 0..n {
            y[i] = u[i] + a * dt * k[i];
            z[i] = u[i] + (1.0 - b - a) * dt * k[i];
            u[i] += (1.0 - b - a) * dt * k[i];
        }

        // Stage 2 at t + 2a*dt
        solve_stage_slope(&rhs, &jac_fn, t + 2.0 * a * dt, &y, a * dt, &mut k);
        for i in 0..n {
            z[i] += b * dt * k[i];
            u[i] += b * dt * k[i];
        }

        // Stage 3 at t + dt
        solve_stage_slope(&rhs, &jac_fn, t + dt, &z, a * dt, &mut k);
        for i in 0..n {
            u[i] += a * dt * k[i];
        }
    }
}

// ─── ESDIRK33: 3-stage, 3rd order ESDIRK (A-stable) ──────────────────────────

/// Three stage, explicitly (first stage) diagonal implicit Runge–Kutta (ESDIRK)
/// method of order 3 (MFEM `ESDIRK33Solver`).  A-stable.
///
/// Butcher tableau
/// ```text
///    0   |      0          0        0
///    2a  |      a          a        0
///    1   |    1-b-a        b        a
/// -------+----------------------------
///        |  1-b_2-b_3     b_2      b_3
/// ```
pub struct Esdirk33;

impl ImplicitTimeStepper for Esdirk33 {
    fn step_implicit<F, J>(&self, t: f64, dt: f64, u: &mut [f64], rhs: F, jac_fn: J)
    where
        F: Fn(f64, &[f64], &mut [f64]),
        J: Fn(f64, &[f64]) -> fem_linalg::CsrMatrix<f64>,
    {
        let n = u.len();
        let a = (3.0 + 3.0_f64.sqrt()) / 6.0;
        let b = (1.0 - 2.0 * a) / (4.0 * a);
        let b_2 = 1.0 / (12.0 * a * (1.0 - 2.0 * a));
        let b_3 = (1.0 - 3.0 * a) / (3.0 * (1.0 - 2.0 * a));

        let mut k = vec![0.0_f64; n];
        let mut y = vec![0.0_f64; n];
        let mut z = vec![0.0_f64; n];

        // Stage 1 (explicit) at t
        rhs(t, u, &mut k);
        for i in 0..n {
            y[i] = u[i] + a * dt * k[i];
            z[i] = u[i] + (1.0 - b - a) * dt * k[i];
            u[i] += (1.0 - b_2 - b_3) * dt * k[i];
        }

        // Stage 2 at t + 2a*dt
        solve_stage_slope(&rhs, &jac_fn, t + 2.0 * a * dt, &y, a * dt, &mut k);
        for i in 0..n {
            z[i] += b * dt * k[i];
            u[i] += b_2 * dt * k[i];
        }

        // Stage 3 at t + dt
        solve_stage_slope(&rhs, &jac_fn, t + dt, &z, a * dt, &mut k);
        for i in 0..n {
            u[i] += b_3 * dt * k[i];
        }
    }
}

// ─── AB5: 5-step, 5th order Adams–Bashforth (MFEM `AB5Solver`) ───────────────

/// A 5-stage, 5th order explicit Adams–Bashforth multistep method
/// (MFEM `AB5Solver`).
///
/// `u_{n+1} = u_n + dt·Σ_{i=0..4} a_i·f_{n-i}` with the classical 5-step
/// weights `a = {1901, -2774, 2616, -1274, 251}/720`.  Startup (the first 4
/// steps) uses the [`Rk6`] Runge–Kutta method to build the RHS history,
/// exactly as in MFEM's `AdamsBashforthSolver`.
///
/// The method requires a constant time step; like MFEM, a changed `dt` purges
/// the history and rebuilds it with RK6.
pub struct Ab5;

/// RHS history state for [`Ab5`].
///
/// `history` holds `f(t_{n-i}, u_{n-i})`, most recent last (max 5 entries).
pub struct Ab5State {
    /// Stored RHS evaluations, most recent last.
    history: Vec<Vec<f64>>,
    /// Last time step seen; a change purges the history (MFEM `CheckTimestep`).
    last_dt: f64,
}

impl Default for Ab5State {
    fn default() -> Self {
        Self::new()
    }
}

/// Adams–Bashforth 5-step weights (MFEM `AB5Solver::a`).
const AB5_A: [f64; 5] = [
    1901.0 / 720.0,
    -2774.0 / 720.0,
    2616.0 / 720.0,
    -1274.0 / 720.0,
    251.0 / 720.0,
];

impl Ab5State {
    /// Create an empty history (startup will use RK6).
    pub fn new() -> Self {
        Ab5State {
            history: Vec::new(),
            last_dt: -1.0,
        }
    }
}

impl Ab5 {
    /// Advance `u` from time `t` by step `dt`, updating `state`.
    pub fn step<F>(&self, t: f64, dt: f64, u: &mut [f64], state: &mut Ab5State, rhs: F)
    where
        F: Fn(f64, &[f64], &mut [f64]),
    {
        // MFEM CheckTimestep: a changed dt purges the history.
        if state.last_dt < 0.0 {
            state.last_dt = dt;
        } else if (dt - state.last_dt).abs() > 10.0 * f64::EPSILON {
            state.history.clear();
            state.last_dt = dt;
        }

        let n = u.len();
        if state.history.len() >= AB5_A.len() - 1 {
            // Full AB5 step: push f_n, then combine the 5 most recent RHS values.
            let mut f0 = vec![0.0_f64; n];
            rhs(t, u, &mut f0);
            state.history.push(f0);
            if state.history.len() > AB5_A.len() {
                state.history.remove(0);
            }
            let h = &state.history;
            for i in 0..n {
                let mut acc = 0.0_f64;
                for (j, aj) in AB5_A.iter().enumerate() {
                    acc += aj * h[h.len() - 1 - j][i];
                }
                u[i] += dt * acc;
            }
        } else {
            // Startup: store f_n and take an RK6 step (MFEM uses RK6Solver).
            let mut f0 = vec![0.0_f64; n];
            rhs(t, u, &mut f0);
            state.history.push(f0);
            let rk6 = Rk6;
            rk6.step(t, dt, u, &rhs);
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;

    /// dU/dt = cos(t)·U, U(0) = 1  →  U(t) = exp(sin t).  Non-autonomous, so it
    /// exercises the time-dependence of the stages.
    fn cos_rhs() -> impl Fn(f64, &[f64], &mut [f64]) {
        |t, u, dudt| {
            dudt[0] = t.cos() * u[0];
        }
    }

    /// Exact Jacobian of cos(t)·u: cos(t) as a 1x1 CSR matrix.
    fn exact_cos_jac(t: f64) -> fem_linalg::CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(1, 1);
        coo.add(0, 0, t.cos());
        coo.into_csr()
    }

    /// dU/dt = U (exact Jacobian 1), U(0) = 1  →  U(t) = e^t.
    fn exp_rhs() -> impl Fn(f64, &[f64], &mut [f64]) {
        |_t, u, dudt| {
            dudt[0] = u[0];
        }
    }

    /// Exact Jacobian of -λ·u as a 1x1 CSR matrix.
    fn exact_jac_scalar(lambda: f64) -> fem_linalg::CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(1, 1);
        coo.add(0, 0, lambda);
        coo.into_csr()
    }

    /// Integrate from 0 to `t_end` (must be a multiple of `dt`) with fixed step.
    fn run_explicit<S: TimeStepper>(s: &S, nsteps: usize, dt: f64, rhs: impl Fn(f64, &[f64], &mut [f64])) -> f64 {
        let mut u = vec![1.0_f64];
        let mut t = 0.0_f64;
        for _ in 0..nsteps {
            s.step(t, dt, &mut u, &rhs);
            t += dt;
        }
        u[0]
    }

    fn run_implicit<S: ImplicitTimeStepper>(
        s: &S,
        nsteps: usize,
        dt: f64,
        rhs: impl Fn(f64, &[f64], &mut [f64]),
        jac: impl Fn(f64, &[f64]) -> fem_linalg::CsrMatrix<f64>,
    ) -> f64 {
        let mut u = vec![1.0_f64];
        let mut t = 0.0_f64;
        for _ in 0..nsteps {
            s.step_implicit(t, dt, &mut u, &rhs, &jac);
            t += dt;
        }
        u[0]
    }

    /// Empirical order from two step sizes: p = log2(err(dt) / err(dt/2)).
    fn order2(e_coarse: f64, e_fine: f64) -> f64 {
        (e_coarse / e_fine).log2()
    }

    // Reference values below were produced by the MFEM 4.10 C++ classes
    // (serial build) on the same scalar problems; they pin the 1:1 port.
    //   explicit: relative agreement at machine precision
    //   implicit: fem-rs solves stages with GMRES (rtol 1e-10), so 1e-8 rel

    #[test]
    fn rk2_reference_and_order() {
        let rhs = cos_rhs();
        // MFEM RK2Solver(a) on y' = cos(t) y, t_end = 1.
        for (name, solver, ref50, ref100) in [
            ("Rk2(2/3)", Rk2 { a: 2.0 / 3.0 }, 2.31973202004195755e0, 2.31976563981177319e0),
            ("Rk2(1/2)", Rk2 { a: 0.5 }, 2.31979128140761404e0, 2.31978053598191147e0),
            ("Rk2(1)", Rk2 { a: 1.0 }, 2.31961306462422989e0, 2.31973579268920771e0),
        ] {
            let u50 = run_explicit(&solver, 50, 0.02, &rhs);
            let u100 = run_explicit(&solver, 100, 0.01, &rhs);
            assert!(
                (u50 - ref50).abs() <= 1e-13 * ref50.abs(),
                "{name}: dt=0.02 got {u50:.17e}, MFEM {ref50:.17e}"
            );
            assert!(
                (u100 - ref100).abs() <= 1e-13 * ref100.abs(),
                "{name}: dt=0.01 got {u100:.17e}, MFEM {ref100:.17e}"
            );
            let exact = (1.0_f64).sin().exp();
            let p = order2((u50 - exact).abs(), (u100 - exact).abs());
            println!("{name}: order = {p:.3}");
            assert!(p > 1.8, "{name}: measured order {p:.3}, expected ~2");
        }
    }

    #[test]
    fn rk6_reference_and_order() {
        // Order measured on y' = y (on y' = cos(t)·y RK6's leading error term
        // nearly cancels, so C++ itself shows erratic local orders there).
        let e_rhs = exp_rhs();
        let u_c = run_explicit(&Rk6, 5, 0.2, &e_rhs);
        let u_f = run_explicit(&Rk6, 10, 0.1, &e_rhs);
        // MFEM RK6Solver reference values.
        assert!(
            (u_c - 2.71828182822302722e0).abs() <= 1e-13,
            "RK6 dt=0.2 got {u_c:.17e}, MFEM 2.71828182822302722e+00"
        );
        assert!(
            (u_f - 2.71828182845469257e0).abs() <= 1e-13,
            "RK6 dt=0.1 got {u_f:.17e}, MFEM 2.71828182845469257e+00"
        );
        let p = order2((u_c - std::f64::consts::E).abs(), (u_f - std::f64::consts::E).abs());
        println!("Rk6: order = {p:.3}");
        assert!(p > 5.0, "Rk6: measured order {p:.3}, expected ~6");
    }

    #[test]
    fn rk8_reference_and_order() {
        // y' = cos(t) y integrated to t_end = 4 (MFEM reference values).
        let rhs = cos_rhs();
        let u_c = run_explicit(&Rk8, 8, 0.5, &rhs);
        let u_f = run_explicit(&Rk8, 16, 0.25, &rhs);
        assert!(
            (u_c - 4.69164186184372722e-1).abs() <= 1e-13,
            "RK8 dt=0.5 got {u_c:.17e}, MFEM 4.69164186184372722e-01"
        );
        assert!(
            (u_f - 4.69164185874983652e-1).abs() <= 1e-13,
            "RK8 dt=0.25 got {u_f:.17e}, MFEM 4.69164185874983652e-01"
        );
        let exact = 4.0_f64.sin().exp();
        let p = order2((u_c - exact).abs(), (u_f - exact).abs());
        println!("Rk8: order = {p:.3}");
        assert!(p > 6.5, "Rk8: measured order {p:.3}, expected ~8");
    }

    #[test]
    fn ab5_reference_and_order() {
        let rhs = cos_rhs();
        let mut state = Ab5State::new();
        let mut u = vec![1.0_f64];
        let mut t = 0.0_f64;
        for _ in 0..20 {
            Ab5.step(t, 0.05, &mut u, &mut state, &rhs);
            t += 0.05;
        }
        // MFEM AB5Solver reference (RK6 startup, classical 5-step weights).
        assert!(
            (u[0] - 2.31977313300567278e0).abs() <= 1e-12,
            "AB5 dt=0.05 got {:.17e}, MFEM 2.31977313300567278e+00",
            u[0]
        );
        let exact = (1.0_f64).sin().exp();
        let err_c = (u[0] - exact).abs();

        let mut state = Ab5State::new();
        let mut u = vec![1.0_f64];
        let mut t = 0.0_f64;
        for _ in 0..40 {
            Ab5.step(t, 0.025, &mut u, &mut state, &rhs);
            t += 0.025;
        }
        assert!(
            (u[0] - 2.31977669796512753e0).abs() <= 1e-12,
            "AB5 dt=0.025 got {:.17e}, MFEM 2.31977669796512753e+00",
            u[0]
        );
        let p = order2(err_c, (u[0] - exact).abs());
        println!("Ab5: order = {p:.3}");
        assert!(p > 4.3, "Ab5: measured order {p:.3}, expected ~5");
    }

    #[test]
    fn implicit_midpoint_reference_order_stability() {
        let crhs = cos_rhs();
        let cjac = |t: f64, _u: &[f64]| exact_cos_jac(t);

        // Stiff y' = -1000 y, dt = 0.1: A-stable (not L-stable) → |R|^n bounded.
        let s_rhs = |_t: f64, u: &[f64], dudt: &mut [f64]| {
            dudt[0] = -1000.0 * u[0];
        };
        let s_jac = |_t: f64, _u: &[f64]| exact_jac_scalar(-1000.0);
        let u_stiff = run_implicit(&ImplicitMidpoint, 10, 0.1, &s_rhs, &s_jac);
        println!("ImplicitMidpoint stiff: u = {u_stiff:.17e}");
        assert!(
            (u_stiff - 6.70284288004420858e-1).abs() <= 1e-8,
            "ImplicitMidpoint stiff got {u_stiff:.17e}, MFEM 6.70284288004420858e-01"
        );
        assert!(u_stiff.abs() < 1.0, "ImplicitMidpoint: |u| grew: {u_stiff}");

        // Accuracy on the cos problem.
        let u_c = run_implicit(&ImplicitMidpoint, 50, 0.02, &crhs, &cjac);
        let u_f = run_implicit(&ImplicitMidpoint, 100, 0.01, &crhs, &cjac);
        assert!(
            (u_c - 2.31985907367899546e0).abs() <= 1e-6,
            "ImpMid dt=0.02 got {u_c:.17e}"
        );
        assert!(
            (u_f - 2.31979738597408680e0).abs() <= 1e-6,
            "ImpMid dt=0.01 got {u_f:.17e}"
        );
        let exact = (1.0_f64).sin().exp();
        let p = order2((u_c - exact).abs(), (u_f - exact).abs());
        println!("ImplicitMidpoint: order = {p:.3}");
        assert!(p > 1.8, "ImplicitMidpoint: measured order {p:.3}, expected ~2");
    }

    #[test]
    fn sdirk34_reference_order_stability() {
        let s_rhs = |_t: f64, u: &[f64], dudt: &mut [f64]| {
            dudt[0] = -1000.0 * u[0];
        };
        let s_jac = |_t: f64, _u: &[f64]| exact_jac_scalar(-1000.0);
        let u_stiff = run_implicit(&Sdirk34, 10, 0.1, &s_rhs, &s_jac);
        println!("Sdirk34 stiff: u = {u_stiff:.17e}");
        assert!(
            (u_stiff - 6.80469393086526562e-3).abs() <= 1e-8,
            "Sdirk34 stiff got {u_stiff:.17e}, MFEM 6.80469393086526562e-03"
        );

        let crhs = cos_rhs();
        let cjac = |t: f64, _u: &[f64]| exact_cos_jac(t);
        let u_c = run_implicit(&Sdirk34, 20, 0.05, &crhs, &cjac);
        let u_f = run_implicit(&Sdirk34, 40, 0.025, &crhs, &cjac);
        assert!((u_c - 2.31977687369539654e0).abs() <= 1e-6, "Sdirk34 dt=0.05 got {u_c:.17e}");
        assert!((u_f - 2.31977682731907553e0).abs() <= 1e-6, "Sdirk34 dt=0.025 got {u_f:.17e}");
        let exact = (1.0_f64).sin().exp();
        let p = order2((u_c - exact).abs(), (u_f - exact).abs());
        println!("Sdirk34: order = {p:.3}");
        assert!(p > 3.5, "Sdirk34: measured order {p:.3}, expected ~4");
    }

    #[test]
    fn sdirk33_reference_order_stability() {
        let s_rhs = |_t: f64, u: &[f64], dudt: &mut [f64]| {
            dudt[0] = -1000.0 * u[0];
        };
        let s_jac = |_t: f64, _u: &[f64]| exact_jac_scalar(-1000.0);
        let u_stiff = run_implicit(&Sdirk33, 10, 0.1, &s_rhs, &s_jac);
        println!("Sdirk33 stiff: u = {u_stiff:.17e}");
        assert!(u_stiff.abs() < 1e-6, "Sdirk33 (L-stable): not decaying: {u_stiff:.3e}");

        let crhs = cos_rhs();
        let cjac = |t: f64, _u: &[f64]| exact_cos_jac(t);
        let u_c = run_implicit(&Sdirk33, 20, 0.05, &crhs, &cjac);
        let u_f = run_implicit(&Sdirk33, 40, 0.025, &crhs, &cjac);
        assert!((u_c - 2.31977389762229746e0).abs() <= 1e-6, "Sdirk33 dt=0.05 got {u_c:.17e}");
        assert!((u_f - 2.31977645767178675e0).abs() <= 1e-6, "Sdirk33 dt=0.025 got {u_f:.17e}");
        let exact = (1.0_f64).sin().exp();
        let p = order2((u_c - exact).abs(), (u_f - exact).abs());
        println!("Sdirk33: order = {p:.3}");
        assert!(p > 2.7, "Sdirk33: measured order {p:.3}, expected ~3");
    }

    #[test]
    fn esdirk32_reference_order_stability() {
        let s_rhs = |_t: f64, u: &[f64], dudt: &mut [f64]| {
            dudt[0] = -1000.0 * u[0];
        };
        let s_jac = |_t: f64, _u: &[f64]| exact_jac_scalar(-1000.0);
        let u_stiff = run_implicit(&Esdirk32, 10, 0.1, &s_rhs, &s_jac);
        println!("Esdirk32 stiff: u = {u_stiff:.17e}");
        assert!(u_stiff.abs() < 1e-6, "Esdirk32 (L-stable): not decaying: {u_stiff:.3e}");

        let crhs = cos_rhs();
        let cjac = |t: f64, _u: &[f64]| exact_cos_jac(t);
        let u_c = run_implicit(&Esdirk32, 50, 0.02, &crhs, &cjac);
        let u_f = run_implicit(&Esdirk32, 100, 0.01, &crhs, &cjac);
        assert!((u_c - 2.31972962244026437e0).abs() <= 1e-6, "Esdirk32 dt=0.02 got {u_c:.17e}");
        assert!((u_f - 2.31976501099869736e0).abs() <= 1e-6, "Esdirk32 dt=0.01 got {u_f:.17e}");
        let exact = (1.0_f64).sin().exp();
        let p = order2((u_c - exact).abs(), (u_f - exact).abs());
        println!("Esdirk32: order = {p:.3}");
        assert!(p > 1.8, "Esdirk32: measured order {p:.3}, expected ~2");
    }

    #[test]
    fn esdirk33_reference_order_stability() {
        let s_rhs = |_t: f64, u: &[f64], dudt: &mut [f64]| {
            dudt[0] = -1000.0 * u[0];
        };
        let s_jac = |_t: f64, _u: &[f64]| exact_jac_scalar(-1000.0);
        // A-stable, not L-stable: at z = -100 the amplification is ~ -0.707/step
        // (MFEM C++ gives 3.01708389844956848e-02 after 10 steps — matched).
        let u_stiff = run_implicit(&Esdirk33, 10, 0.1, &s_rhs, &s_jac);
        println!("Esdirk33 stiff: u = {u_stiff:.17e}");
        assert!(
            (u_stiff - 3.01708389844956848e-2).abs() <= 1e-8,
            "Esdirk33 stiff got {u_stiff:.17e}, MFEM 3.01708389844956848e-02"
        );

        let crhs = cos_rhs();
        let cjac = |t: f64, _u: &[f64]| exact_cos_jac(t);
        let u_c = run_implicit(&Esdirk33, 50, 0.02, &crhs, &cjac);
        let u_f = run_implicit(&Esdirk33, 100, 0.01, &crhs, &cjac);
        assert!((u_c - 2.31977946840327620e0).abs() <= 1e-6, "Esdirk33 dt=0.02 got {u_c:.17e}");
        assert!((u_f - 2.31977715376558269e0).abs() <= 1e-6, "Esdirk33 dt=0.01 got {u_f:.17e}");
        let exact = (1.0_f64).sin().exp();
        let p = order2((u_c - exact).abs(), (u_f - exact).abs());
        println!("Esdirk33: order = {p:.3}");
        assert!(p > 2.7, "Esdirk33: measured order {p:.3}, expected ~3");
    }
}
