//! adjoint_cvodes_roberts — 1:1 port of MFEM `miniapps/adjoint/cvsRoberts_ASAi_dns.cpp`
//! (a port of the SUNDIALS CVODES serial example `cvsRoberts_ASAi_dns`).
//!
//! The C++ original requires SUNDIALS CVODES; this port uses fem-solver's own
//! checkpointed adjoint time-integration kernel
//! (`fem_solver::adjoint::time_dependent`) with CVODES-equivalent semantics:
//! forward BDF + checkpoints (`InitAdjointSolve(150, CV_HERMITE)` analog),
//! backward adjoint BDF, and backward quadrature-sensitivity accumulation.
//!
//! Adjoint sensitivity example problem (Robertson chemical kinetics):
//!
//! ```text
//!     dy1/dt = -p1*y1 + p2*y2*y3
//!     dy2/dt =  p1*y1 - p2*y2*y3 - p3*(y2)^2
//!     dy3/dt =  p3*(y2)^2
//! ```
//!
//! on t ∈ [0, 4e7], y(0) = (1, 0, 0), p = (0.04, 1e4, 3e7). The goal is
//! `G = ∫_0^{t_final} y3 dt`, whose gradient dG/dp is obtained from the adjoint
//! system solved backward in time. Run: `adjoint_cvodes_roberts -dt 0.01`.
//!
//! Numerical differences vs CVODES (documented in the kernel docs): the
//! forward problem uses the same variable-order/variable-step BDF as the
//! backward problem (CVODES' forward CV_ADAMS mode is not reproduced), and
//! quadratures are accumulated by Gauss quadrature of the BDF interpolant.

use fem_solver::adjoint::{
    AdjointBdfConfig, AdjointConfig, AdjointSolver, AdjointTolerances, Interpolation,
    TimeDependentAdjointOperator,
};
use fem_linalg::CsrMatrix;

/// Format like C `printf("%.*g", prec, x)` — matches `std::ostream` with
/// `cout.precision(prec)` used by the C++ miniapp.
fn fmt_g_prec(x: f64, p: i32) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    if !x.is_finite() {
        return format!("{x}");
    }
    let s = format!("{:.*e}", (p - 1) as usize, x);
    let epos = s.find('e').unwrap();
    let exp: i32 = s[epos + 1..].parse().unwrap();
    if exp < -4 || exp >= p {
        let mant = s[..epos].trim_end_matches('0').trim_end_matches('.');
        format!("{}e{}{:02}", mant, if exp < 0 { "-" } else { "+" }, exp.abs())
    } else {
        let digits = (p - 1 - exp).max(0) as usize;
        if digits == 0 {
            // Integer rendering: the trailing zeros are significant digits.
            return format!("{:.0}", x);
        }
        let f = format!("{:.*}", digits, x);
        f.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// MFEM `Vector::Print` analogue: `width` space-separated `%g` values per line.
fn print_vec(nameless: &[f64], width: usize) {
    let prec = 8; // C++: cout.precision(8)
    for (i, &v) in nameless.iter().enumerate() {
        print!("{}", fmt_g_prec(v, prec));
        if (i + 1) % width == 0 || i + 1 == nameless.len() {
            println!();
        } else {
            print!(" ");
        }
    }
}

fn dense_csr(n: usize, a: &[[f64; 3]; 3]) -> CsrMatrix<f64> {
    use fem_linalg::CooMatrix;
    let mut coo = CooMatrix::<f64>::new(n, n);
    for (i, row) in a.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 {
                coo.add(i, j, v);
            }
        }
    }
    coo.into_csr()
}

/// `RobertsTDAOperator` — MFEM's `TimeDependentAdjointOperator` implementation
/// of the Robertson rate equations (1:1 with the C++ miniapp).
struct RobertsTDAOperator {
    p: [f64; 3],
}

impl RobertsTDAOperator {
    /// Jacobian of the forward rate, `J = ∂f/∂y` (the C++ builds the adjoint
    /// Newton matrix by applying `AdjointRateMult` to unit vectors; here both
    /// Jacobians are analytic — the converged BDF steps are identical).
    fn forward_jac(&self, y: &[f64]) -> [[f64; 3]; 3] {
        let [p1, p2, p3] = self.p;
        [
            [-p1, p2 * y[2], p2 * y[1]],
            [p1, -p2 * y[2] - 2.0 * p3 * y[1], -p2 * y[1]],
            [0.0, 2.0 * p3 * y[1], 0.0],
        ]
    }

    /// Jacobian of the adjoint rate w.r.t. `yB`, `JB = ∂fB/∂yB`.
    fn adjoint_jac_dense(&self, y: &[f64]) -> [[f64; 3]; 3] {
        let [p1, p2, p3] = self.p;
        [
            [p1, -p1, 0.0],
            [-p2 * y[2], p2 * y[2] + 2.0 * p3 * y[1], -2.0 * p3 * y[1]],
            [-p2 * y[1], p2 * y[1], 0.0],
        ]
    }
}

impl TimeDependentAdjointOperator for RobertsTDAOperator {
    fn dim(&self) -> usize {
        3
    }

    // cvsRoberts_ASAi_dns rate equation
    fn mult(&self, _t: f64, x: &[f64], y: &mut [f64]) {
        y[0] = -self.p[0] * x[0] + self.p[1] * x[1] * x[2];
        y[2] = self.p[2] * x[1] * x[1];
        y[1] = -y[0] - y[2];
    }

    fn jac(&self, _t: f64, y: &[f64]) -> CsrMatrix<f64> {
        dense_csr(3, &self.forward_jac(y))
    }

    // cvsRoberts_ASAi_dns adjoint rate equation
    fn adjoint_rate_mult(&self, _t: f64, y: &[f64], yb: &[f64], ybdot: &mut [f64]) {
        let l21 = yb[1] - yb[0];
        let l32 = yb[2] - yb[1];
        let p1 = self.p[0];
        let p2 = self.p[1];
        let p3 = self.p[2];
        ybdot[0] = -p1 * l21;
        ybdot[1] = p2 * y[2] * l21 - 2.0 * p3 * y[1] * l32;
        ybdot[2] = p2 * y[1] * l21 - 1.0;
    }

    fn adjoint_jac(&self, _t: f64, y: &[f64], _yb: &[f64]) -> CsrMatrix<f64> {
        dense_csr(3, &self.adjoint_jac_dense(y))
    }

    fn quad_dim(&self) -> usize {
        1
    }

    // cvsRoberts_ASAi_dns quadrature rate equation
    fn quadrature_integration(&self, _t: f64, y: &[f64], qdot: &mut [f64]) {
        qdot[0] = y[2];
    }

    // cvsRoberts_ASAi_dns quadrature sensitivity rate equation
    fn quadrature_sensitivity_mult(&self, _t: f64, y: &[f64], yb: &[f64], qbdot: &mut [f64]) {
        let l21 = yb[1] - yb[0];
        let l32 = yb[2] - yb[1];
        let y23 = y[1] * y[2];

        qbdot[0] = y[0] * l21;
        qbdot[1] = -y23 * l21;
        qbdot[2] = y[1] * y[1] * l32;
    }
}

fn main() {
    // Parse command-line options (C++ OptionsParser analog).
    let mut t_final: f64 = 4e7;
    let mut dt: f64 = 0.01;
    let mut reltol: f64 = 1e-4;
    let mut reltolb: f64 = 1e-4; // CVODES default rel_tolB
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().unwrap_or_default();
        match arg.as_str() {
            "-tf" | "--t-final" => t_final = val().parse().unwrap_or(t_final),
            "-dt" | "--time-step" => dt = val().parse().unwrap_or(dt),
            "-rtol" | "--rel-tol" => reltol = val().parse().unwrap_or(reltol),
            "-rtolb" | "--rel-tol-b" => reltolb = val().parse().unwrap_or(reltolb),
            _ => {}
        }
    }
    let abstol_v = [1.0e-8_f64, 1.0e-14, 1.0e-6];

    let precision = 8;
    println!("Options:");
    println!("   -dt: {}", fmt_g_prec(dt, precision));

    // Define material parameters p.
    let p = [0.04, 1.0e4, 3.0e7];

    // The original cvsRoberts_ASAi_dns problem is a fixed sized problem of
    // size 3. Define the solution vector: y(0) = (1, 0, 0).
    let mut u = vec![1.0_f64, 0.0, 0.0];

    let adv = RobertsTDAOperator { p };

    // Create the solver; forward tolerances = (reltol, abstol_v) via
    // SetSVtolerances; adjoint tolerances = CVODES defaults
    // (rel_tolB = 1e-4, abs_tolB = 1e-9). InitAdjointSolve(150, CV_HERMITE).
    let cfg = AdjointConfig {
        forward: AdjointBdfConfig {
            tol: AdjointTolerances { rtol: reltol, abstol: abstol_v.to_vec() },
            dt0: 1e-6,
            ..AdjointBdfConfig::new(AdjointTolerances::scalar(reltol, 1.0, 3), 1e-6)
        },
        adjoint: AdjointBdfConfig::new(AdjointTolerances::scalar(reltolb, 1e-9, 3), 1e-6),
        steps_per_checkpoint: 150,
        interpolation: Interpolation::Hermite,
        store_all_forward_steps: false,
    };
    let mut cvodes = AdjointSolver::new(&adv, cfg);

    // Initialize the quadrature result and integrate the forward problem.
    let t = 0.0;
    cvodes.run_forward(t, t_final, &u, &[0.0]).expect("forward solve failed");
    u.copy_from_slice(cvodes.forward_final());

    println!("BDF statistics (PrintInfo analog): {}", cvodes.forward_stats.summary());

    println!("Final Solution: {}", fmt_g_prec(t_final, precision));
    print_vec(&u, 8);

    let mut q = [0.0];
    q[0] = cvodes.quadrature()[0];
    println!(" Final Quadrature ");
    print_vec(&q, 8);

    // Solve the adjoint problem at different points in time.
    let mut w = vec![0.0_f64; 3];
    let tbout1: f64 = 40.;
    let mut dg_dp = vec![0.0_f64; 3];

    cvodes.init_adjoint(&w, &dg_dp);

    // Results at time TBout1: StepB(w, t, max(dt, t_final - TBout1)).
    cvodes.step_adjoint_to(tbout1).expect("adjoint step to TBout1 failed");
    w.copy_from_slice(cvodes.adjoint());
    println!("t: {}", fmt_g_prec(cvodes.adjoint_time(), precision));
    println!("w:");
    print_vec(&w, 8);

    let ufwd = cvodes.forward_solution_at(cvodes.adjoint_time());
    println!("u:");
    print_vec(&ufwd, 8);

    // Results at T0: StepB(w, t, max(dt, t - 0)).
    cvodes.step_adjoint_to(0.0).expect("adjoint step to T0 failed");
    w.copy_from_slice(cvodes.adjoint());
    println!("t: {}", fmt_g_prec(cvodes.adjoint_time(), precision));
    println!("w:");
    print_vec(&w, 8);

    let ufwd = cvodes.forward_solution_at(cvodes.adjoint_time());
    println!("u:");
    print_vec(&ufwd, 8);

    // Evaluate Sensitivity: EvalQuadIntegrationB(t, dG_dp).
    dg_dp.copy_from_slice(cvodes.quad_sensitivity());
    println!("dG/dp:");
    print_vec(&dg_dp, 8);

    println!(
        "Adjoint BDF statistics (PrintInfo analog): {}",
        cvodes.adjoint_stats.summary()
    );
    println!("Checkpoint re-integration steps: {}", cvodes.n_reint_steps());
}
