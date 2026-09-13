//! MFEM's `ODESolver` family (`linalg/ode.hpp`, `linalg/ode.cpp`) — the time
//! integrators the electromagnetics miniapps drive (`joule`, `maxwell`), in
//! their **class** form rather than the `TimeStepper` closure form of the
//! sibling modules.
//!
//! ```text
//!   ODESolver                 (trait, linalg/ode.hpp:120)   -> OdeSolver
//!   ├── BackwardEulerSolver                                  -> BackwardEulerSolver
//!   ├── ImplicitMidpointSolver                               -> ImplicitMidpointSolver
//!   ├── SDIRK23Solver(gamma_opt)                             -> Sdirk23Solver
//!   ├── SDIRK33Solver                                        -> Sdirk33Solver
//!   └── SDIRK34Solver                                        -> Sdirk34Solver
//!   SIASolver                 (linalg/ode.hpp:761)          -> SiavSolver
//!   └── SIAVSolver(order)                                    -> (same type)
//! ```
//!
//! The distinguishing feature of this family — and the reason it exists next to
//! the closure-based [`TimeStepper`](super::TimeStepper)s — is that it talks to
//! a *stateful operator object*:
//!
//! ```text
//!   f->SetTime(t);                 // the operator carries the stage time
//!   f->ImplicitSolve(dt, x, k);    // solve  k = f(x + dt·k, t)   (SLOPE), or
//!                                  //        k = u(t+dt)          (STATE)
//!   f->Mult(x, y);                 // y = f(t, x)   (the explicit branch)
//! ```
//!
//! [`TimeDependentOperator`] is the Rust form of that object.  All six solvers
//! below are line-by-line ports of `linalg/ode.cpp`: the same Butcher
//! coefficients, the same stage order, the same `t` bookkeeping (including
//! `SIAV`'s `t += a_[i]*dt` *inside* the stage loop) and the same branch on the
//! operator's implicit-variable type.
//!
//! Numerical agreement with MFEM is checked in
//! `crates/solver/tests/d89_ode_solvers.rs` against per-step values dumped from
//! the C++ library (stiff 2×2 system, `≤ 1e-13`).

/// MFEM `TimeDependentOperator` restricted to what the ODE solvers use.
///
/// # `ImplicitSolve` contract
/// `implicit_solve(dt, x, k)` solves, for the *slope* convention (the default,
/// `ImplicitVariableType::SLOPE`) [MFEM `ode.hpp:556`]:
/// ```text
///   k = f(t, x + dt·k)          ⇒   (I − dt·∂f/∂x) k = f(t, x)   (linearised)
/// ```
/// or, when [`TimeDependentOperator::implicit_var_is_state`] is `true`
/// (`ImplicitVariableType::STATE`), it returns the advanced *state*
/// `k = u(t + dt)` directly.  The solvers of this module apply
/// [`compute_slope_from_state`] in that case, exactly as `ode.cpp` does.
pub trait TimeDependentOperator {
    /// `f->Height()` — the size of the state vector.
    fn size(&self) -> usize;

    /// `f->SetTime(t)` — the stage time the next `Mult`/`ImplicitSolve` uses.
    fn set_time(&mut self, t: f64);

    /// `f->Mult(x, y)`: `y = f(t, x)`.
    fn mult(&self, x: &[f64], y: &mut [f64]);

    /// `f->ImplicitSolve(dt, x, k)`: see the trait documentation.
    fn implicit_solve(&mut self, dt: f64, x: &[f64], k: &mut [f64]);

    /// `f->isExplicit()` — MFEM's `Type::EXPLICIT` flag, which decides whether
    /// `SIAVSolver::Step` calls `Mult` or `ImplicitSolve`.
    fn is_explicit(&self) -> bool { false }

    /// `f->ImplicitVarTypeIsState()` — `true` when `implicit_solve` returns the
    /// advanced state instead of the slope.
    fn implicit_var_is_state(&self) -> bool { false }
}

/// `ODE solver: k currently holds u(t+dt); convert it to the slope
/// `du/dt = (u(t+dt) − u(t))/dt` (MFEM `ODESolver::ComputeSlopeFromState`).
pub fn compute_slope_from_state(dt: f64, u: &[f64], k: &mut [f64]) {
    let fac = 1.0 / dt;
    for (ki, &ui) in k.iter_mut().zip(u.iter()) {
        *ki -= ui;
        *ki *= fac;
    }
}

/// MFEM `ODESolver`: a time-stepping scheme driven by a
/// [`TimeDependentOperator`].
///
/// `step` deliberately has no return value and takes `dt` by value: every
/// solver in this family is non-adaptive (MFEM's `Step` only writes back `dt`
/// for the adaptive members, which are not part of this set).
pub trait OdeSolver<F: TimeDependentOperator> {
    /// `ODESolver::Init` — size the internal stage vectors from `f`.
    fn init(&mut self, f: &F);

    /// `ODESolver::Step(x, t, dt)`: advance `x` from `t` to `t + dt`.
    fn step(&mut self, f: &mut F, x: &mut [f64], t: &mut f64, dt: f64);

    /// `ODESolver::Run(x, t, dt, tf)`: `while (t < tf) Step(x, t, dt);`
    ///
    /// # Panics
    /// Panics if the step is not positive and `t` has not reached `tf` (MFEM
    /// would spin forever).
    fn run(&mut self, f: &mut F, x: &mut [f64], t: &mut f64, dt: f64, tf: f64) {
        while *t < tf {
            assert!(dt > 0.0, "OdeSolver::run: non-positive step in an infinite loop");
            self.step(f, x, t, dt);
        }
    }
}

// ─── Backward Euler ─────────────────────────────────────────────────────────

/// Backward Euler, L-stable (MFEM `BackwardEulerSolver`, `ode.cpp:682`).
pub struct BackwardEulerSolver {
    k: Vec<f64>,
}

impl Default for BackwardEulerSolver {
    fn default() -> Self { Self::new() }
}

impl BackwardEulerSolver {
    /// Create the solver; the stage vector is sized by [`OdeSolver::init`].
    pub fn new() -> Self { BackwardEulerSolver { k: Vec::new() } }
}

impl<F: TimeDependentOperator> OdeSolver<F> for BackwardEulerSolver {
    fn init(&mut self, f: &F) { self.k = vec![0.0; f.size()]; }

    fn step(&mut self, f: &mut F, x: &mut [f64], t: &mut f64, dt: f64) {
        f.set_time(*t + dt);
        f.implicit_solve(dt, x, &mut self.k); // k = f(x + dt*k, t + dt)
        if f.implicit_var_is_state() {
            x.copy_from_slice(&self.k); // x = u_{i+1}
        } else {
            for (xi, &ki) in x.iter_mut().zip(self.k.iter()) {
                *xi += dt * ki;
            }
        }
        *t += dt;
    }
}

// ─── Implicit midpoint ──────────────────────────────────────────────────────

/// Implicit midpoint method, A-stable but not L-stable (MFEM
/// `ImplicitMidpointSolver`, `ode.cpp:705`).
pub struct ImplicitMidpointSolver {
    k: Vec<f64>,
}

impl Default for ImplicitMidpointSolver {
    fn default() -> Self { Self::new() }
}

impl ImplicitMidpointSolver {
    /// Create the solver; the stage vector is sized by [`OdeSolver::init`].
    pub fn new() -> Self { ImplicitMidpointSolver { k: Vec::new() } }
}

impl<F: TimeDependentOperator> OdeSolver<F> for ImplicitMidpointSolver {
    fn init(&mut self, f: &F) { self.k = vec![0.0; f.size()]; }

    fn step(&mut self, f: &mut F, x: &mut [f64], t: &mut f64, dt: f64) {
        f.set_time(*t + dt / 2.0);
        f.implicit_solve(dt / 2.0, x, &mut self.k);
        if f.implicit_var_is_state() {
            for (xi, &ki) in x.iter_mut().zip(self.k.iter()) {
                *xi = -(*xi) + 2.0 * ki; // x.Neg(); x.Add(2.0, k)
            }
        } else {
            for (xi, &ki) in x.iter_mut().zip(self.k.iter()) {
                *xi += dt * ki;
            }
        }
        *t += dt;
    }
}

// ─── SDIRK23 (two-stage, parameterised by gamma) ─────────────────────────────

/// Which `gamma` the 2-stage SDIRK23 tableau uses — MFEM's `gamma_opt`
/// (`ode.cpp:729`).
///
/// ```text
///   gamma |   gamma
///   1-gamma | 1-2·gamma   gamma
///   --------+--------------------
///           |   1/2        1/2
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sdirk23Gamma {
    /// `(3 − √3)/6` — 3rd order, **not** A-stable (`gamma_opt = 0`).
    Order3,
    /// `(3 + √3)/6` — 3rd order, A-stable, not L-stable (default, `gamma_opt = 1`).
    AStable,
    /// `(2 − √2)/2` — 2nd order, L-stable (`gamma_opt = 2`).
    LStable,
    /// `(2 + √2)/2` — 2nd order, L-stable; both solves are outside
    /// `[t, t+dt]` since `gamma > 1` (`gamma_opt = 3`).
    LStableOutside,
}

impl Sdirk23Gamma {
    /// The `gamma` value itself.
    pub fn gamma(self) -> f64 {
        match self {
            Sdirk23Gamma::Order3 => (3.0 - 3.0_f64.sqrt()) / 6.0,
            Sdirk23Gamma::AStable => (3.0 + 3.0_f64.sqrt()) / 6.0,
            Sdirk23Gamma::LStable => (2.0 - 2.0_f64.sqrt()) / 2.0,
            Sdirk23Gamma::LStableOutside => (2.0 + 2.0_f64.sqrt()) / 2.0,
        }
    }
}

/// Two-stage singly diagonally implicit RK, 3rd order (MFEM `SDIRK23Solver`,
/// `ode.cpp:749`).
pub struct Sdirk23Solver {
    gamma: f64,
    k: Vec<f64>,
    y: Vec<f64>,
}

impl Default for Sdirk23Solver {
    fn default() -> Self { Self::new(Sdirk23Gamma::AStable) }
}

impl Sdirk23Solver {
    /// Create the solver with MFEM's `gamma_opt` choice.
    pub fn new(gamma_opt: Sdirk23Gamma) -> Self {
        Sdirk23Solver { gamma: gamma_opt.gamma(), k: Vec::new(), y: Vec::new() }
    }

    /// The `gamma` actually in use.
    pub fn gamma(&self) -> f64 { self.gamma }
}

impl<F: TimeDependentOperator> OdeSolver<F> for Sdirk23Solver {
    fn init(&mut self, f: &F) {
        self.k = vec![0.0; f.size()];
        self.y = vec![0.0; f.size()];
    }

    fn step(&mut self, f: &mut F, x: &mut [f64], t: &mut f64, dt: f64) {
        let g = self.gamma;
        let n = x.len();

        f.set_time(*t + g * dt);
        f.implicit_solve(g * dt, x, &mut self.k);
        if f.implicit_var_is_state() {
            compute_slope_from_state(g * dt, x, &mut self.k);
        }
        // y = x + (1 - 2g) dt k  ... and x += dt/2 k (x is used as the accumulator)
        for i in 0..n {
            self.y[i] = x[i] + (1.0 - 2.0 * g) * dt * self.k[i];
            x[i] += dt / 2.0 * self.k[i];
        }

        f.set_time(*t + (1.0 - g) * dt);
        f.implicit_solve(g * dt, &self.y, &mut self.k);
        if f.implicit_var_is_state() {
            compute_slope_from_state(g * dt, &self.y, &mut self.k);
        }
        for (xi, &ki) in x.iter_mut().zip(self.k.iter()) {
            *xi += dt / 2.0 * ki;
        }
        *t += dt;
    }
}

// ─── SDIRK33 ────────────────────────────────────────────────────────────────

/// Three-stage singly diagonally implicit RK of order 3, L-stable (MFEM
/// `SDIRK33Solver`, `ode.cpp:833`).
///
/// ```text
///   a | a
///   c | c-a    a
///   1 | b    1-a-b   a
///   --+----------------
///     | b    1-a-b   a
/// ```
pub struct Sdirk33Solver {
    k: Vec<f64>,
    y: Vec<f64>,
}

impl Default for Sdirk33Solver {
    fn default() -> Self { Self::new() }
}

impl Sdirk33Solver {
    /// Create the solver (the tableau has no free parameters).
    pub fn new() -> Self { Sdirk33Solver { k: Vec::new(), y: Vec::new() } }
}

impl<F: TimeDependentOperator> OdeSolver<F> for Sdirk33Solver {
    fn init(&mut self, f: &F) {
        self.k = vec![0.0; f.size()];
        self.y = vec![0.0; f.size()];
    }

    fn step(&mut self, f: &mut F, x: &mut [f64], t: &mut f64, dt: f64) {
        let a = 0.435866521508458999416019_f64;
        let b = 1.20849664917601007033648_f64;
        let c = 0.717933260754229499708010_f64;
        let n = x.len();

        f.set_time(*t + a * dt);
        f.implicit_solve(a * dt, x, &mut self.k);
        if f.implicit_var_is_state() {
            compute_slope_from_state(a * dt, x, &mut self.k);
        }
        for i in 0..n {
            self.y[i] = x[i] + (c - a) * dt * self.k[i];
            x[i] += b * dt * self.k[i];
        }

        f.set_time(*t + c * dt);
        f.implicit_solve(a * dt, &self.y, &mut self.k);
        if f.implicit_var_is_state() {
            compute_slope_from_state(a * dt, &self.y, &mut self.k);
        }
        for (xi, &ki) in x.iter_mut().zip(self.k.iter()) {
            *xi += (1.0 - a - b) * dt * ki;
        }

        f.set_time(*t + dt);
        f.implicit_solve(a * dt, x, &mut self.k);
        if f.implicit_var_is_state() {
            compute_slope_from_state(a * dt, x, &mut self.k);
        }
        for (xi, &ki) in x.iter_mut().zip(self.k.iter()) {
            *xi += a * dt * ki;
        }
        *t += dt;
    }
}

// ─── SDIRK34 ────────────────────────────────────────────────────────────────

/// Three-stage singly diagonally implicit RK of order 4, A-stable, not L-stable
/// (MFEM `SDIRK34Solver`, `ode.cpp:784`).
///
/// ```text
///   a   | a
///   1/2 | 1/2-a    a
///   1-a | 2a       1-4a   a
///   ----+--------------------
///       | b        1-2b   b
/// ```
/// Two of the three solves lie outside `[t, t+dt]` (`c1 = a > 1`, `c3 = 1-a < 0`).
pub struct Sdirk34Solver {
    k: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
}

impl Default for Sdirk34Solver {
    fn default() -> Self { Self::new() }
}

impl Sdirk34Solver {
    /// Create the solver (the tableau has no free parameters).
    pub fn new() -> Self {
        Sdirk34Solver { k: Vec::new(), y: Vec::new(), z: Vec::new() }
    }
}

impl<F: TimeDependentOperator> OdeSolver<F> for Sdirk34Solver {
    fn init(&mut self, f: &F) {
        self.k = vec![0.0; f.size()];
        self.y = vec![0.0; f.size()];
        self.z = vec![0.0; f.size()];
    }

    fn step(&mut self, f: &mut F, x: &mut [f64], t: &mut f64, dt: f64) {
        let a = 1.0 / 3.0_f64.sqrt() * (std::f64::consts::PI / 18.0).cos() + 0.5;
        let b = 1.0 / (6.0 * (2.0 * a - 1.0) * (2.0 * a - 1.0));
        let n = x.len();

        f.set_time(*t + a * dt);
        f.implicit_solve(a * dt, x, &mut self.k);
        if f.implicit_var_is_state() {
            compute_slope_from_state(a * dt, x, &mut self.k);
        }
        for i in 0..n {
            self.y[i] = x[i] + (0.5 - a) * dt * self.k[i];
            self.z[i] = x[i] + (2.0 * a) * dt * self.k[i];
            x[i] += b * dt * self.k[i];
        }

        f.set_time(*t + dt / 2.0);
        f.implicit_solve(a * dt, &self.y, &mut self.k);
        if f.implicit_var_is_state() {
            compute_slope_from_state(a * dt, &self.y, &mut self.k);
        }
        for i in 0..n {
            self.z[i] += (1.0 - 4.0 * a) * dt * self.k[i];
            x[i] += (1.0 - 2.0 * b) * dt * self.k[i];
        }

        f.set_time(*t + (1.0 - a) * dt);
        f.implicit_solve(a * dt, &self.z, &mut self.k);
        if f.implicit_var_is_state() {
            compute_slope_from_state(a * dt, &self.z, &mut self.k);
        }
        for (xi, &ki) in x.iter_mut().zip(self.k.iter()) {
            *xi += b * dt * ki;
        }
        *t += dt;
    }
}

// ─── SIAV (symplectic, variable order) ──────────────────────────────────────

/// One of the two state vectors of a symplectic (SIA) system.
///
/// In MFEM these are `Vector`s (`SIASolver::Step`) or `ParVector`s
/// (`maxwell.cpp`); the trait exists so the step below can be written once
/// while a host keeps ownership of its storage — including the ghost exchange a
/// `ParVector` needs after every update ([`SiaState::sync`], no-op for a serial
/// `Vec`).
pub trait SiaState {
    /// Number of stored entries.
    fn len(&self) -> usize;

    /// Whether the vector is empty.
    fn is_empty(&self) -> bool { self.len() == 0 }

    /// Values the `F_`/`P_` operators read (for a `ParVector`, ghost entries
    /// included — `ParVector::as_slice`).
    fn slice(&self) -> &[f64];

    /// Values the operators write.
    fn slice_mut(&mut self) -> &mut [f64];

    /// `v.Add(c, y)` — the entries of `y` are added to the matching entries of
    /// `v` (the rest of `v`, if any, is left alone).
    fn add_scaled(&mut self, c: f64, y: &[f64]) {
        for (vi, &yi) in self.slice_mut().iter_mut().zip(y.iter()) {
            *vi += c * yi;
        }
    }

    /// Propagate the new values to any redundant storage — MFEM's
    /// `ParVector::update_ghosts()`; a no-op for a serial vector.
    fn sync(&mut self) {}
}

impl SiaState for Vec<f64> {
    fn len(&self) -> usize { self.as_slice().len() }
    fn slice(&self) -> &[f64] { self.as_slice() }
    fn slice_mut(&mut self) -> &mut [f64] { self.as_mut_slice() }
}

/// Variable-order (1–4) symplectic integration algorithm (MFEM `SIAVSolver`,
/// `ode.cpp:1109`), for Hamiltonian systems written as
/// ```text
///   dq/dt = P p        (P_ : Operator,        `p_mult`)
///   dp/dt = F q        (F_ : TimeDependentOperator, explicit or implicit)
/// ```
/// The tables are copied verbatim from `ode.cpp:1109-1149`.
pub struct SiavSolver {
    order: usize,
    a: Vec<f64>,
    b: Vec<f64>,
}

impl SiavSolver {
    /// Create the solver of the given order (1–4).
    ///
    /// # Panics
    /// Panics if `order` is not 1–4 (MFEM: `MFEM_ASSERT(false, "Unsupported
    /// order in SIAVSolver")`).
    pub fn new(order: usize) -> Self {
        let (a, b) = match order {
            1 => (vec![1.0], vec![1.0]),
            2 => (vec![0.5, 0.5], vec![0.0, 1.0]),
            3 => (
                vec![2.0 / 3.0, -2.0 / 3.0, 1.0],
                vec![7.0 / 24.0, 0.75, -1.0 / 24.0],
            ),
            4 => {
                let c2 = 2.0_f64.powf(1.0 / 3.0);
                (
                    vec![
                        (2.0 + c2 + 1.0 / c2) / 6.0,
                        (1.0 - c2 - 1.0 / c2) / 6.0,
                        (1.0 - c2 - 1.0 / c2) / 6.0,
                        (2.0 + c2 + 1.0 / c2) / 6.0,
                    ],
                    vec![
                        0.0,
                        1.0 / (2.0 - c2),
                        1.0 / (1.0 - c2 * c2),
                        1.0 / (2.0 - c2),
                    ],
                )
            }
            o => panic!("SiavSolver::new: unsupported order {o} (must be 1..=4)"),
        };
        SiavSolver { order, a, b }
    }

    /// Order of the integrator.
    pub fn order(&self) -> usize { self.order }

    /// The `a_` table (the `q`-update weights; `Σ a_i = 1`).
    pub fn a(&self) -> &[f64] { &self.a }

    /// The `b_` table (the `p`-update weights).
    pub fn b(&self) -> &[f64] { &self.b }

    /// MFEM `SIASolver::Step(q, p, t, dt)` (`ode.cpp:1151`).
    ///
    /// `f` is `F_` (used on the first argument of the system — MFEM passes `q`:
    /// `F_->Mult(q, dp_)` / `F_->ImplicitSolve(b_i·dt, q, dp_)`), `p_mult` is
    /// `P_->Mult(p, dq_)`, and `dp`/`dq` are the solver's own scratch vectors
    /// (`SIASolver::dp_`, `dq_`).
    ///
    /// The `t += a_[i] * dt` of the C++ loop is repeated verbatim **inside** the
    /// stage loop, so `F_` sees the staggered stage times.
    pub fn step<F, PM, S>(
        &self,
        f: &mut F,
        p_mult: PM,
        q: &mut S,
        p: &mut S,
        dp: &mut S,
        dq: &mut S,
        t: &mut f64,
        dt: f64,
    ) where
        F: TimeDependentOperator,
        PM: Fn(&[f64], &mut [f64]),
        S: SiaState,
    {
        for i in 0..self.order {
            if self.b[i] != 0.0 {
                f.set_time(*t);
                if f.is_explicit() {
                    f.mult(q.slice(), dp.slice_mut());
                } else {
                    f.implicit_solve(self.b[i] * dt, q.slice(), dp.slice_mut());
                }
                p.add_scaled(self.b[i] * dt, dp.slice());
                p.sync();
            }

            p_mult(p.slice(), dq.slice_mut());
            q.add_scaled(self.a[i] * dt, dq.slice());
            q.sync();

            *t += self.a[i] * dt;
        }
    }

    /// Serial convenience wrapper around [`SiavSolver::step`]: the scratch
    /// vectors `dp`/`dq` are allocated here (MFEM sizes them in
    /// `SIASolver::Init` from `F_->Height()` / `P_->Height()`).
    pub fn step_vecs<F, PM>(
        &self,
        f: &mut F,
        p_mult: PM,
        q: &mut Vec<f64>,
        p: &mut Vec<f64>,
        t: &mut f64,
        dt: f64,
    ) where
        F: TimeDependentOperator,
        PM: Fn(&[f64], &mut [f64]),
    {
        let mut dp = vec![0.0; f.size()];
        let mut dq = vec![0.0; p.len()];
        self.step(f, p_mult, q, p, &mut dp, &mut dq, t, dt);
    }
}

// ─── A small matrix operator (for tests and simple drivers) ─────────────────

/// A dense serial `P` matrix — the `Operator` role of `SIAVSolver::Init`.
///
/// Kept here (rather than in `fem-linalg`) because only the SIAV step needs an
/// operator whose `Mult` is called as a plain slice-to-slice map.
#[derive(Debug, Clone)]
pub struct DenseOperator {
    n: usize,
    a: Vec<f64>,
}

impl DenseOperator {
    /// Build the operator `y = A x` from a row-major `n × n` matrix.
    ///
    /// # Panics
    /// Panics if `a.len() != n * n`.
    pub fn new(n: usize, a: Vec<f64>) -> Self {
        assert_eq!(a.len(), n * n, "DenseOperator::new: expected {n}x{n} entries, got {}", a.len());
        DenseOperator { n, a }
    }

    /// Build from a 2-D slice (rows).
    pub fn from_rows(rows: &[&[f64]]) -> Self {
        let n = rows.len();
        let mut a = Vec::with_capacity(n * n);
        for r in rows {
            assert_eq!(r.len(), n, "DenseOperator::from_rows: not square");
            a.extend_from_slice(r);
        }
        DenseOperator { n, a }
    }

    /// Size of the operator.
    pub fn size(&self) -> usize { self.n }

    /// `y = A x`.
    ///
    /// # Panics
    /// Panics if `x.len() < n` or `y.len() < n` (MFEM's `Operator::Mult`
    /// asserts the same).
    pub fn mult(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.n {
            let mut s = 0.0;
            for j in 0..self.n {
                s += self.a[i * self.n + j] * x[j];
            }
            y[i] = s;
        }
    }

    /// `(I − dt·A) k = A x` — the `ImplicitSolve` of a linear operator whose
    /// `f(u) = A u`.  `LinearOp` below uses it.
    pub fn implicit_solve(&self, dt: f64, x: &[f64], k: &mut [f64]) {
        let n = self.n;
        let mut ax = vec![0.0; n];
        self.mult(x, &mut ax);
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                m[i * n + j] = -dt * self.a[i * n + j];
            }
            m[i * n + i] += 1.0;
        }
        solve_dense(n, &m, &ax, k);
    }
}

/// A square `*_FESpace`-free linear operator `f(u) = A u` used as the `F_` of a
/// [`SiavSolver`] or as the operator of the implicit solvers.
///
/// `implicit_solve` solves `(I − dt·A) k = A x` with Gaussian elimination with
/// partial pivoting (small dense systems only — the FE operators of the
/// miniapps implement [`TimeDependentOperator`] themselves).
#[derive(Debug, Clone)]
pub struct LinearOp {
    a: DenseOperator,
    implicit_state: bool,
    explicit: bool,
}

impl LinearOp {
    /// Wrap a dense matrix as a [`TimeDependentOperator`].
    pub fn new(a: DenseOperator) -> Self {
        LinearOp { a, implicit_state: false, explicit: false }
    }

    /// Mark the operator `Type::EXPLICIT` (`f->isExplicit()`), as MFEM's
    /// `MaxwellSolver` is (`miniapps/electromagnetics/maxwell.cpp` never passes
    /// `Type::IMPLICIT`).  `SIAVSolver::Step` then calls `Mult` and never
    /// `ImplicitSolve`.
    pub fn as_explicit(mut self) -> Self {
        self.explicit = true;
        self
    }

    /// Switch `ImplicitSolve` to `ImplicitVariableType::STATE` semantics
    /// (`k = u(t+dt)`, MFEM `ImplicitVarTypeIsState()`).
    pub fn with_state_implicit(mut self, yes: bool) -> Self {
        self.implicit_state = yes;
        self
    }
}

impl TimeDependentOperator for LinearOp {
    fn size(&self) -> usize { self.a.size() }

    /// The operator is autonomous: `SetTime` is accepted and ignored, exactly
    /// like the constant-coefficient operators of the miniapps.
    fn set_time(&mut self, _t: f64) {}

    fn mult(&self, x: &[f64], y: &mut [f64]) { self.a.mult(x, y); }

    fn implicit_solve(&mut self, dt: f64, x: &[f64], k: &mut [f64]) {
        if self.implicit_state {
            // k = u(t+dt) = x + dt * A(x + dt k) ⇒ (I − dt A) k = x
            let n = self.a.size();
            let mut m = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    m[i * n + j] = -dt * self.a.a[i * n + j];
                }
                m[i * n + i] += 1.0;
            }
            solve_dense(n, &m, x, k);
        } else {
            self.a.implicit_solve(dt, x, k);
        }
    }

    fn is_explicit(&self) -> bool { self.explicit }

    fn implicit_var_is_state(&self) -> bool { self.implicit_state }
}

/// Solve `M k = b` for a small dense `n × n` system (Gauss–Jordan with partial
/// pivoting, the same algorithm as MFEM's `DenseMatrix::Invert` + `Mult`).
fn solve_dense(n: usize, m: &[f64], b: &[f64], k: &mut [f64]) {
    let mut a = m.to_vec();
    let mut rhs = b.to_vec();
    for col in 0..n {
        // pivot
        let mut piv = col;
        for r in (col + 1)..n {
            if a[r * n + col].abs() > a[piv * n + col].abs() {
                piv = r;
            }
        }
        if piv != col {
            for c in 0..n {
                a.swap(col * n + c, piv * n + c);
            }
            rhs.swap(col, piv);
        }
        let d = a[col * n + col];
        assert!(d != 0.0, "solve_dense: singular system");
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r * n + col] / d;
            if f == 0.0 {
                continue;
            }
            for c in col..n {
                a[r * n + c] -= f * a[col * n + c];
            }
            rhs[r] -= f * rhs[col];
        }
    }
    for i in 0..n {
        k[i] = rhs[i] / a[i * n + i];
    }
}

