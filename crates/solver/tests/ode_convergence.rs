use fem_solver::{
    ForwardEuler, Rk4,
    ImplicitEuler, Sdirk2,
    Bdf2, Bdf2State,
    TimeStepper, ImplicitTimeStepper,
    ImexOperator, ImexTimeStepper,
    SdcConfig, SdcIntegrator,
    DaeState, DaeIntegrator, DaeNewtonConfig,
};
use fem_linalg::{CooMatrix, CsrMatrix};

// u' = -λ u  ->  u(t) = exp(-λ t)
fn decay_rhs(lambda: f64) -> impl Fn(f64, &[f64], &mut [f64]) {
    move |_t, u, dudt| { dudt[0] = -lambda * u[0]; }
}

fn decay_jac(n: usize, lambda: f64) -> impl Fn(f64, &[f64]) -> CsrMatrix<f64> {
    move |_t, _u| {
        let mut coo = CooMatrix::<f64>::new(n, n);
        coo.add(0, 0, -lambda);
        coo.into_csr()
    }
}

fn integrate_explicit<S: TimeStepper>(
    stepper: &S, rhs: impl Fn(f64, &[f64], &mut [f64]),
    dt: f64, t_end: f64,
) -> f64 {
    let mut u = vec![1.0];
    let mut t = 0.0;
    while t < t_end - 1e-14 {
        let h = dt.min(t_end - t);
        stepper.step(t, h, &mut u, &rhs);
        t += h;
    }
    u[0]
}

fn integrate_implicit<S: ImplicitTimeStepper>(
    stepper: &S,
    rhs: impl Fn(f64, &[f64], &mut [f64]),
    jac: impl Fn(f64, &[f64]) -> CsrMatrix<f64>,
    dt: f64, t_end: f64,
) -> f64 {
    let mut u = vec![1.0];
    let mut t = 0.0;
    while t < t_end - 1e-14 {
        let h = dt.min(t_end - t);
        stepper.step_implicit(t, h, &mut u, &rhs, &jac);
        t += h;
    }
    u[0]
}

fn observed_order(errors: &[f64], dts: &[f64]) -> f64 {
    let r = errors[0] / errors[1];
    let h_ratio = dts[0] / dts[1];
    r.log2() / h_ratio.log2()
}

const T_END: f64 = 0.5;
const LAMBDA: f64 = 2.0;

#[test]
fn forward_euler_is_first_order() {
    let rhs = decay_rhs(LAMBDA);
    let fe = ForwardEuler;
    let dts = [0.002, 0.001];
    let errors: Vec<f64> = dts.iter().map(|&dt: &f64| {
        let u = integrate_explicit(&fe, &rhs, dt, T_END);
        (u - (-LAMBDA * T_END).exp()).abs()
    }).collect();
    let order = observed_order(&errors, &dts);
    assert!(order > 0.85 && order < 1.5, "ForwardEuler order={order:.2} (expected ~1)");
}

#[test]
fn rk4_is_fourth_order() {
    let rhs = decay_rhs(LAMBDA);
    let rk4 = Rk4;
    let dts = [0.05, 0.025];
    let errors: Vec<f64> = dts.iter().map(|&dt: &f64| {
        let u = integrate_explicit(&rk4, &rhs, dt, T_END);
        (u - (-LAMBDA * T_END).exp()).abs()
    }).collect();
    let order = observed_order(&errors, &dts);
    assert!(order > 3.5, "RK4 order={order:.2} (expected ~4)");
}

#[test]
fn implicit_euler_is_first_order() {
    let rhs = decay_rhs(LAMBDA);
    let jac = decay_jac(1, LAMBDA);
    let ie = ImplicitEuler;
    let dts = [0.01, 0.005];
    let errors: Vec<f64> = dts.iter().map(|&dt: &f64| {
        let u = integrate_implicit(&ie, &rhs, &jac, dt, T_END);
        (u - (-LAMBDA * T_END).exp()).abs()
    }).collect();
    let order = observed_order(&errors, &dts);
    assert!(order > 0.85 && order < 1.5, "ImplicitEuler order={order:.2} (expected ~1)");
}

#[test]
fn sdirk2_is_second_order() {
    let rhs = decay_rhs(LAMBDA);
    let jac = decay_jac(1, LAMBDA);
    let sdirk2 = Sdirk2;
    let dts = [0.05, 0.025];
    let errors: Vec<f64> = dts.iter().map(|&dt: &f64| {
        let u = integrate_implicit(&sdirk2, &rhs, &jac, dt, T_END);
        (u - (-LAMBDA * T_END).exp()).abs()
    }).collect();
    let order = observed_order(&errors, &dts);
    assert!(order > 1.7, "SDIRK2 order={order:.2} (expected ~2)");
}

#[test]
fn bdf2_is_second_order() {
    let rhs = decay_rhs(LAMBDA);
    let jac = decay_jac(1, LAMBDA);
    let bdf2 = Bdf2;
    let dts = [0.05, 0.025];
    let errors: Vec<f64> = dts.iter().map(|&dt: &f64| {
        let mut u = vec![1.0];
        let mut state = Bdf2State::new();
        let mut t = 0.0;
        while t < T_END - 1e-14 {
            let h = dt.min(T_END - t);
            bdf2.step_implicit(t, h, &mut u, &mut state, &rhs, &jac);
            t += h;
        }
        (u[0] - (-LAMBDA * T_END).exp()).abs()
    }).collect();
    let order = observed_order(&errors, &dts);
    assert!(order > 1.7, "BDF2 order={order:.2} (expected ~2)");
}

struct SplitDecay {
    lambda: f64,
}

impl ImexOperator for SplitDecay {
    fn explicit(&self, _t: f64, _u: &[f64], out: &mut [f64]) {
        out[0] = 0.0;
    }
    fn implicit(&self, _t: f64, u: &[f64], out: &mut [f64]) {
        out[0] = -self.lambda * u[0];
    }
    fn jac_implicit(&self, _t: f64, _u: &[f64]) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(1, 1);
        coo.add(0, 0, -self.lambda);
        coo.into_csr()
    }
}

#[test]
fn imex_ssp2_second_order() {
    let op = SplitDecay { lambda: 5.0 };
    let driver = ImexTimeStepper;
    let t_end = 0.5;
    let exact = (-op.lambda * t_end).exp();
    let dts = [0.05, 0.025];
    let errors: Vec<f64> = dts.iter().map(|&dt: &f64| {
        let mut u = vec![1.0];
        driver.integrate_ssp2(&op, 0.0, t_end, &mut u, dt);
        (u[0] - exact).abs()
    }).collect();
    let order = observed_order(&errors, &dts);
    assert!(order > 1.5, "IMEX SSP2 order={order:.2} (expected ~2)");
}

#[test]
fn imex_rk3_third_order() {
    let op = SplitDecay { lambda: 5.0 };
    let driver = ImexTimeStepper;
    let t_end = 0.5;
    let exact = (-op.lambda * t_end).exp();
    let dts = [0.05, 0.025];
    let errors: Vec<f64> = dts.iter().map(|&dt: &f64| {
        let mut u = vec![1.0];
        driver.integrate_rk3(&op, 0.0, t_end, &mut u, dt);
        (u[0] - exact).abs()
    }).collect();
    let order = observed_order(&errors, &dts);
    assert!(order > 2.5, "IMEX RK3 order={order:.2} (expected ~3)");
}

#[test]
fn sdc_converges_with_more_sweeps() {
    let rhs = decay_rhs(LAMBDA);
    let t_end = 0.5;
    let exact = (-LAMBDA * t_end).exp();

    let coarse = SdcIntegrator::new(SdcConfig { m: 2, k: 1 });
    let fine   = SdcIntegrator::new(SdcConfig { m: 2, k: 3 });

    let mut u_coarse = vec![1.0];
    let mut u_fine   = vec![1.0];
    let dt: f64 = 0.05;
    let mut t = 0.0;
    while t < t_end - 1e-14 {
        let h = dt.min(t_end - t);
        coarse.step(t, h, &mut u_coarse, &rhs);
        fine  .step(t, h, &mut u_fine,   &rhs);
        t += h;
    }

    let err_coarse = (u_coarse[0] - exact).abs();
    let err_fine   = (u_fine[0] - exact).abs();
    assert!(err_fine < err_coarse,
        "SDC: 3 sweeps (err={err_fine:.4e}) should be more accurate than 1 sweep (err={err_coarse:.4e})");
}

fn simple_dae_res(_t: f64, y: &[f64], yp: &[f64], res: &mut [f64]) {
    res[0] = yp[0] - 2.0 * y[0];
    res[1] = y[0] + 1.0 - y[1];
}

fn simple_dae_jac(_t: f64, _y: &[f64], _yp: &[f64]) -> (CsrMatrix<f64>, CsrMatrix<f64>) {
    let mut df_dy = CooMatrix::<f64>::new(2, 2);
    df_dy.add(0, 0, -2.0);
    df_dy.add(1, 0, 1.0);
    df_dy.add(1, 1, -1.0);
    let mut df_dyp = CooMatrix::<f64>::new(2, 2);
    df_dyp.add(0, 0, 1.0);
    (df_dy.into_csr(), df_dyp.into_csr())
}

#[test]
fn dae_simple_step_constraint_satisfied() {
    let y0 = vec![1.0, 2.0];
    let yp0 = vec![2.0, 0.0];
    let mut state = DaeState::new(0.0, &y0, &yp0, 0.05);
    let newton = DaeNewtonConfig::default();
    let result = DaeIntegrator::step(&mut state, 0.0, &simple_dae_res, &simple_dae_jac, &newton);
    assert!(result.is_ok(), "DAE step failed: {:?}", result.err());
    let y = state.y();
    let expected_y = (2.0 * 0.05_f64).exp();
    assert!((y[0] - expected_y).abs() < 0.01, "y={}, expected={}", y[0], expected_y);
    assert!((y[1] - (y[0] + 1.0)).abs() < 1e-12, "constraint violated: z={}, y+1={}", y[1], y[0] + 1.0);
}
