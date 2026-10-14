//! Round-153 pin: [`NavierSolver::reassemble_operators(dt)`] — the mid-run
//! refresh of the `Setup`-time operator block (fem-rs extension; see the
//! `fem_solver::navier` module docs).
//!
//! # What is under test
//!
//! The hook must (a) reassemble exactly the `Setup` operator block
//! (`mv`/`sp`/`mv_diag`/`h_diag`, `sp_amg` when `pressure_amg` is set) so a
//! variable-coefficient discretization sees the field-moved matrices from
//! the very next step, and (b) leave every piece of solver STATE untouched —
//! `dthist` and the velocity history (`un`/`un_next`/`unm1`/`unm2`), the
//! pressure state (`pn`, `pnm1`, `pnm2`) — so the BDF time order never drops
//! (the failure mode a mid-run `setup()` re-entry would cause).
//!
//! # Referees
//!
//! - `hook_is_bit_identical_no_op_on_constant_dt`: two identical
//!   constant-coefficient solvers, one never calling the hook and one
//!   calling it before every step — every state vector, every BDF history
//!   entry and every iteration counter must stay bit-identical after each of
//!   5 accepted steps (the strongest form of "the hook disturbs nothing").
//! - `hook_preserves_history_under_variable_dt`: same lockstep with a
//!   varying `dt` sequence.  The hook refreshes `h_diag` through the `1/dt`
//!   Setup convention, so with varying `dt` the H-solve preconditioner
//!   legitimately differs from the never-hooked run — the BDF HISTORY
//!   (`dthist`, the state vectors) must still match to the solve
//!   tolerances, and `dthist` itself must stay bit-identical (that is the
//!   pin).
//! - `hook_refreshes_matrices_consumed_by_step`: flip the disc's coefficient
//!   scales after `setup` and show (i) a step without the hook uses the
//!   stale matrices, (ii) a step after the hook uses the fresh ones —
//!   bit-identical to a solver whose `setup` ran with the fresh scales.
//! - `hook_rebuilds_pressure_amg_and_steps`: the `sp_amg` rebuild arm
//!   (`NavierConfig::pressure_amg = true`) survives repeated hook calls.

use fem_linalg::CsrMatrix;
use fem_solver::navier::{NavierConfig, NavierDiscretization, NavierSolver};
use std::cell::{Cell, RefCell};

// ─── A compact variable-coefficient discretization ──────────────────────────

/// `nv = 4` velocity dofs, `np = 3` pressure dofs; the mass and pressure
/// matrices carry a `Cell`-scale the tests flip between steps (the stand-in
/// for a transported-field coefficient), and every assembly call is counted.
struct VarDisc {
    nv: usize,
    np: usize,
    vel_ess: Vec<usize>,
    pres_ess: Vec<usize>,
    mv_scale: Cell<f64>,
    sp_scale: Cell<f64>,
    mv_calls: Cell<u32>,
    sp_calls: Cell<u32>,
    h_calls: RefCell<Vec<(f64, f64)>>,
}

impl VarDisc {
    fn new(dirichlet_velocity: bool) -> Self {
        VarDisc {
            nv: 4,
            np: 3,
            vel_ess: if dirichlet_velocity { vec![0] } else { vec![] },
            pres_ess: if dirichlet_velocity { vec![] } else { vec![0] },
            mv_scale: Cell::new(1.0),
            sp_scale: Cell::new(1.0),
            mv_calls: Cell::new(0),
            sp_calls: Cell::new(0),
            h_calls: RefCell::new(Vec::new()),
        }
    }

    /// Symmetric tridiagonal `[off, diag, off]` matrix, scaled.
    fn tridiag(n: usize, diag: f64, off: f64, scale: f64) -> CsrMatrix<f64> {
        let mut m = CsrMatrix::new_empty(n, n);
        let mut rp = vec![0usize];
        let mut ci = Vec::new();
        let mut va = Vec::new();
        for r in 0..n {
            if r > 0 {
                ci.push((r - 1) as u32);
                va.push(off * scale);
            }
            ci.push(r as u32);
            va.push(diag * scale);
            if r + 1 < n {
                ci.push((r + 1) as u32);
                va.push(off * scale);
            }
            rp.push(ci.len());
        }
        m.row_ptr = rp;
        m.col_idx = ci;
        m.values = va;
        m
    }

    /// The pure-Neumann 1-D Laplacian `[1,−1; −1,2,−1; …; −1,1]` (singular
    /// with the constant nullspace — the closed-domain pressure operator
    /// shape the kernel's `OrthoSolver` path expects), scaled.
    fn neumann_laplacian(n: usize, scale: f64) -> CsrMatrix<f64> {
        let mut m = CsrMatrix::new_empty(n, n);
        let mut rp = vec![0usize];
        let mut ci = Vec::new();
        let mut va = Vec::new();
        for r in 0..n {
            if r > 0 {
                ci.push((r - 1) as u32);
                va.push(-scale);
            }
            ci.push(r as u32);
            va.push(if r == 0 || r + 1 == n { scale } else { 2.0 * scale });
            if r + 1 < n {
                ci.push((r + 1) as u32);
                va.push(-scale);
            }
            rp.push(ci.len());
        }
        m.row_ptr = rp;
        m.col_idx = ci;
        m.values = va;
        m
    }
}

impl NavierDiscretization for VarDisc {
    fn n_vel(&self) -> usize {
        self.nv
    }
    fn n_pres(&self) -> usize {
        self.np
    }
    fn vel_ess_dofs(&self) -> &[usize] {
        &self.vel_ess
    }
    fn pres_ess_dofs(&self) -> &[usize] {
        &self.pres_ess
    }

    fn assemble_mass_velocity(&self) -> CsrMatrix<f64> {
        self.mv_calls.set(self.mv_calls.get() + 1);
        Self::tridiag(self.nv, 4.0, -1.0, self.mv_scale.get())
    }
    fn assemble_pressure_laplace(&self) -> CsrMatrix<f64> {
        self.sp_calls.set(self.sp_calls.get() + 1);
        Self::neumann_laplacian(self.np, self.sp_scale.get())
    }
    fn assemble_divergence(&self) -> CsrMatrix<f64> {
        // np × nv, deterministic band structure.
        let mut m = CsrMatrix::new_empty(self.np, self.nv);
        let mut rp = vec![0usize];
        let mut ci = Vec::new();
        let mut va = Vec::new();
        for r in 0..self.np {
            for c in 0..self.nv {
                ci.push(c as u32);
                va.push(if c == r { 1.0 } else { 0.25 });
            }
            rp.push(ci.len());
        }
        m.row_ptr = rp;
        m.col_idx = ci;
        m.values = va;
        m
    }
    fn assemble_gradient(&self) -> CsrMatrix<f64> {
        // nv × np — the same band structure transposed (the toy convention;
        // no physical D/G consistency is needed for these pins).
        let mut m = CsrMatrix::new_empty(self.nv, self.np);
        let mut rp = vec![0usize];
        let mut ci = Vec::new();
        let mut va = Vec::new();
        for r in 0..self.nv {
            for c in 0..self.np {
                ci.push(c as u32);
                va.push(if c == r { 1.0 } else { 0.25 });
            }
            rp.push(ci.len());
        }
        m.row_ptr = rp;
        m.col_idx = ci;
        m.values = va;
        m
    }
    fn assemble_helmholtz(&self, mass_coeff: f64, visc_coeff: f64) -> CsrMatrix<f64> {
        self.h_calls.borrow_mut().push((mass_coeff, visc_coeff));
        // The field-weighted mass (the stand-in for M_ρ) + the constant
        // viscous block — so both the H MATRIX (rebuilt in `step`) and the
        // H DIAGONAL snapshot (built by `setup`/the hook) carry `mv_scale`.
        let m = Self::tridiag(self.nv, 4.0, -1.0, self.mv_scale.get() * mass_coeff);
        let k = Self::tridiag(self.nv, 2.0, -0.5, visc_coeff);
        m.add(&k)
    }
    fn convection_residual(&self, u: &[f64], out: &mut [f64]) {
        for (o, u) in out.iter_mut().zip(u.iter()) {
            *o = 0.5 * u;
        }
    }
    fn curl_curl(&self, u: &[f64]) -> Vec<f64> {
        let s: f64 = u.iter().sum::<f64>() / u.len() as f64;
        vec![s; u.len()]
    }
    fn project_velocity_bdr(&self, t: f64, out: &mut [f64]) {
        for &d in &self.vel_ess {
            out[d] = 1.0 + t;
        }
    }
    fn project_pressure_bdr(&self, t: f64, out: &mut [f64]) {
        for &d in &self.pres_ess {
            out[d] = 2.0 + t;
        }
    }
    fn assemble_ftext_bdr(&self, ftext: &[f64]) -> Vec<f64> {
        vec![ftext[0] * 0.25; self.np]
    }
    fn assemble_g_bdr(&self, t: f64) -> Vec<f64> {
        vec![0.5 + t; self.np]
    }
    fn mean_zero(&self, v: &mut [f64]) {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        for x in v.iter_mut() {
            *x -= mean;
        }
    }
    fn compute_cfl(&self, u: &[f64], dt: f64) -> f64 {
        dt * u.iter().fold(0.0_f64, |m, &v| m.max(v.abs()))
    }
    fn eliminate_bc(
        &self,
        mat: &mut CsrMatrix<f64>,
        rhs: &mut [f64],
        ess: &[usize],
        values: &[f64],
    ) {
        // MFEM DIAG_KEEP `EliminateRowCol` (same double as the navier unit
        // tests: reactions from the TRUE column entries `A[other, dof]`).
        for (k, &d) in ess.iter().enumerate() {
            let mut diag = 0.0;
            for j in mat.row_ptr[d]..mat.row_ptr[d + 1] {
                let c = mat.col_idx[j] as usize;
                if c == d {
                    diag = mat.values[j];
                } else {
                    if let Some(p) = mat.find_entry(c, d) {
                        let a_cd = mat.values[p];
                        if a_cd != 0.0 {
                            rhs[c] -= a_cd * values[k];
                            mat.values[p] = 0.0;
                        }
                    }
                }
            }
            for j in mat.row_ptr[d]..mat.row_ptr[d + 1] {
                if mat.col_idx[j] as usize != d {
                    mat.values[j] = 0.0;
                }
            }
            rhs[d] = diag * values[k];
        }
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn new_solver(disc: VarDisc) -> NavierSolver<VarDisc> {
    let cfg = NavierConfig { verbose: false, ..Default::default() };
    let mut s = NavierSolver::new(disc, 0.5, cfg);
    s.velocity_mut().copy_from_slice(&[1.0, -0.5, 0.25, 0.75]);
    s
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter().zip(b).fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

fn assert_bits_eq(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}: entry {i} differs ({x} vs {y})"
        );
    }
}

// ─── Referees ───────────────────────────────────────────────────────────────

/// Constant `dt`: the hook must be a bit-identical no-op on every piece of
/// state, history and solve bookkeeping, step after step.
#[test]
fn hook_is_bit_identical_no_op_on_constant_dt() {
    let mut no_hook = new_solver(VarDisc::new(true));
    let mut hooked = new_solver(VarDisc::new(true));
    no_hook.setup(0.1);
    hooked.setup(0.1);
    let mut t_a = 0.0_f64;
    let mut t_b = 0.0_f64;
    for step in 0..5 {
        hooked.reassemble_operators(0.1);
        no_hook.step(&mut t_a, 0.1, step, false);
        hooked.step(&mut t_b, 0.1, step, false);
        assert_bits_eq(&[t_a], &[t_b], "time");
        assert_bits_eq(&no_hook.dthist(), &hooked.dthist(), "dthist");
        assert_bits_eq(no_hook.velocity(), hooked.velocity(), "velocity (un)");
        assert_bits_eq(
            no_hook.provisional_velocity(),
            hooked.provisional_velocity(),
            "un_next",
        );
        assert_bits_eq(no_hook.pressure(), hooked.pressure(), "pressure (pn)");
        assert_eq!(
            (
                no_hook.iter_mvsolve(),
                no_hook.iter_spsolve(),
                no_hook.iter_hsolve()
            ),
            (
                hooked.iter_mvsolve(),
                hooked.iter_spsolve(),
                hooked.iter_hsolve()
            ),
            "iteration counters diverged at step {step}"
        );
        assert_bits_eq(
            &[no_hook.res_mvsolve(), no_hook.res_spsolve(), no_hook.res_hsolve()],
            &[hooked.res_mvsolve(), hooked.res_spsolve(), hooked.res_hsolve()],
            "sub-solve residual norms",
        );
    }
    // The hook ran (5 calls) in addition to the setup assembly.
    assert_eq!(hooked.discretization().mv_calls.get(), 6);
    assert_eq!(hooked.discretization().sp_calls.get(), 6);
}

/// Varying `dt`: the BDF history must be preserved bit-exactly (the pin);
/// the trajectories must match to the sub-solve tolerances (the `h_diag`
/// refresh legitimately changes the H preconditioner with the new `dt`).
#[test]
fn hook_preserves_history_under_variable_dt() {
    let dts = [0.1, 0.08, 0.12, 0.1, 0.09];
    let mut no_hook = new_solver(VarDisc::new(true));
    let mut hooked = new_solver(VarDisc::new(true));
    no_hook.setup(dts[0]);
    hooked.setup(dts[0]);
    let mut t_a = 0.0_f64;
    let mut t_b = 0.0_f64;
    for (step, dt) in dts.iter().enumerate() {
        hooked.reassemble_operators(*dt);
        no_hook.step(&mut t_a, *dt, step as i32, false);
        hooked.step(&mut t_b, *dt, step as i32, false);
        assert_bits_eq(&no_hook.dthist(), &hooked.dthist(), "dthist");
        assert_bits_eq(&[t_a], &[t_b], "time");
        // The states agree to the sub-solve tolerances (rtol_hsolve = 1e-8
        // dominates; the h_diag preconditioner difference moves the H-solve
        // iterates at that scale, never the BDF history).
        let du = max_abs_diff(no_hook.velocity(), hooked.velocity());
        let dp = max_abs_diff(no_hook.pressure(), hooked.pressure());
        assert!(
            du < 1e-6 && dp < 1e-3,
            "step {step}: hooked state drifted beyond the solve tolerances \
             (du = {du:.3e}, dp = {dp:.3e})"
        );
    }
}

/// The hook must feed the NEXT step the fresh variable-coefficient matrices:
/// without it the step consumes the stale ones; with it the step is
/// bit-identical to a solver whose `setup` ran with the fresh coefficients.
#[test]
fn hook_refreshes_matrices_consumed_by_step() {
    let run = |flip_before_setup: bool, call_hook: bool| -> (Vec<f64>, Vec<f64>, u32, u32) {
        let disc = VarDisc::new(true);
        if flip_before_setup {
            disc.mv_scale.set(3.0);
            disc.sp_scale.set(2.0);
        }
        let mut s = new_solver(disc);
        s.setup(0.1);
        if !flip_before_setup {
            s.discretization().mv_scale.set(3.0);
            s.discretization().sp_scale.set(2.0);
        }
        if call_hook {
            s.reassemble_operators(0.1);
        }
        let mut t = 0.0;
        s.step(&mut t, 0.1, 0, false);
        (
            s.velocity().to_vec(),
            s.pressure().to_vec(),
            s.discretization().mv_calls.get(),
            s.discretization().sp_calls.get(),
        )
    };

    let (u_stale, p_stale, mv_stale, sp_stale) = run(false, false);
    let (u_fresh, p_fresh, mv_fresh, sp_fresh) = run(false, true);
    let (u_ref, p_ref, mv_ref, sp_ref) = run(true, false);

    // (i) Without the hook the step runs on the setup-time (stale) matrices.
    assert!(
        max_abs_diff(&u_stale, &u_fresh) > 1e-10,
        "the flipped coefficients never entered the step — the hook (or the \
         step) is not consuming the reassembled matrices"
    );
    assert!(
        max_abs_diff(&p_stale, &p_fresh) > 1e-10,
        "the flipped pressure laplacian never entered the step"
    );
    // (ii) With the hook the step is bit-identical to the fresh-setup run.
    assert_bits_eq(&u_ref, &u_fresh, "velocity (fresh setup vs hook)");
    assert_bits_eq(&p_ref, &p_fresh, "pressure (fresh setup vs hook)");
    // Assembly-call ledger: setup + hook = 2 with the flip-after-setup arm.
    assert_eq!((mv_stale, sp_stale), (1, 1));
    assert_eq!((mv_fresh, sp_fresh), (2, 2));
    assert_eq!((mv_ref, sp_ref), (1, 1));
}

/// The `pressure_amg` arm: the hook rebuilds `SpInvPC` (the AMG V-cycle on
/// the reassembled `sp`) and the solver keeps stepping.
#[test]
fn hook_rebuilds_pressure_amg_and_steps() {
    let disc = VarDisc::new(true);
    let mut s = NavierSolver::new(
        disc,
        0.5,
        NavierConfig { verbose: false, pressure_amg: true, ..Default::default() },
    );
    s.velocity_mut().copy_from_slice(&[1.0, -0.5, 0.25, 0.75]);
    s.setup(0.1);
    let mut t = 0.0_f64;
    for step in 0..3 {
        s.discretization().mv_scale.set(1.0 + 0.5 * step as f64);
        s.discretization().sp_scale.set(1.0 + 0.25 * step as f64);
        s.reassemble_operators(0.1);
        s.step(&mut t, 0.1, step, false);
        assert!(
            s.velocity().iter().all(|v| v.is_finite()),
            "AMG-hooked step {step}: non-finite velocity"
        );
        assert!(
            s.pressure().iter().all(|v| v.is_finite()),
            "AMG-hooked step {step}: non-finite pressure"
        );
    }
    assert_eq!(s.discretization().sp_calls.get(), 4);
}
