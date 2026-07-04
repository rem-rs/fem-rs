use fem_assembly::{
    LbfgsConfig, LbfgsSolver,
    TrustRegionConfig, TrustRegionSolver,
    NonlinearForm,
};
use fem_linalg::{CooMatrix, CsrMatrix};

struct TestQuadratic {
    n: usize,
    coeffs: Vec<f64>,
}

impl NonlinearForm for TestQuadratic {
    fn residual(&self, u: &[f64], _rhs: &[f64], r: &mut [f64]) {
        for i in 0..self.n {
            let a = self.coeffs[3 * i];
            let b = self.coeffs[3 * i + 1];
            let c = self.coeffs[3 * i + 2];
            r[i] = a * u[i] * u[i] + b * u[i] + c;
        }
    }
    fn jacobian(&self, u: &[f64]) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(self.n, self.n);
        for i in 0..self.n {
            coo.add(i, i, 2.0 * self.coeffs[3 * i] * u[i] + self.coeffs[3 * i + 1]);
        }
        coo.into_csr()
    }
    fn n_dofs(&self) -> usize { self.n }
}

// ── LBFGS tests ──────────────────────────────────────────────────────────────

#[test]
fn lbfgs_scalar_quadratic() {
    let form = TestQuadratic { n: 1, coeffs: vec![1.0, -4.0, 3.0] };
    let rhs = vec![0.0];
    let cfg = LbfgsConfig { rtol: 1e-12, verbose: false, ..Default::default() };
    let solver = LbfgsSolver::new(cfg);
    let mut u = vec![10.0_f64];
    let res = solver.solve(&form, &rhs, &mut u).unwrap();
    assert!(res.converged);
    assert!((u[0] - 1.0).abs() < 1e-6 || (u[0] - 3.0).abs() < 1e-6);
}

#[test]
fn lbfgs_two_variable() {
    // F₀ = u₀² − 2  (root ±√2), F₁ = u₁² − 27  (root ±√27 ≈ 5.196)
    let coeffs = vec![1.0, 0.0, -2.0, 1.0, 0.0, -27.0];
    let form = TestQuadratic { n: 2, coeffs };
    let rhs = vec![0.0; 2];
    let cfg = LbfgsConfig { rtol: 1e-8, history: 15, max_iter: 200, verbose: false, ..Default::default() };
    let solver = LbfgsSolver::new(cfg);
    let mut u = vec![3.0_f64, 3.0_f64];
    let res = solver.solve(&form, &rhs, &mut u).unwrap();
    assert!(res.converged, "LBFGS 2-var: {} iters res={:.3e}", res.iterations, res.final_residual);
}

#[test]
fn lbfgs_converges_on_simple_problems() {
    for (coeffs, start) in [
        (vec![1.0, -4.0, 3.0], vec![100.0]),
        (vec![1.0, 0.0, -2.0], vec![3.0]),
        (vec![2.0, -4.0, 0.0], vec![-5.0]),
    ] {
        let n = coeffs.len() / 3;
        let form = TestQuadratic { n, coeffs };
        let rhs = vec![0.0; n];
        let cfg = LbfgsConfig { rtol: 1e-6, history: 15, max_iter: 200, verbose: false, ..Default::default() };
        let solver = LbfgsSolver::new(cfg);
        let mut u = start;
        let res = solver.solve(&form, &rhs, &mut u);
        assert!(res.is_ok(), "LBFGS failed: start={:?}", u);
    }
}

// ── Trust-region tests ───────────────────────────────────────────────────────

#[test]
fn trust_region_scalar_quadratic() {
    let form = TestQuadratic { n: 1, coeffs: vec![1.0, -2.0, 0.0] };
    let rhs = vec![0.0];
    let cfg = TrustRegionConfig { rtol: 1e-10, verbose: false, ..Default::default() };
    let solver = TrustRegionSolver::new(cfg);
    let mut u = vec![10.0_f64];
    let res = solver.solve(&form, &rhs, &mut u).unwrap();
    assert!(res.converged);
}

#[test]
fn trust_region_handles_indefinite() {
    let form = TestQuadratic { n: 1, coeffs: vec![-1.0, 0.0, 4.0] };
    let rhs = vec![0.0];
    let cfg = TrustRegionConfig { rtol: 1e-10, verbose: false, ..Default::default() };
    let solver = TrustRegionSolver::new(cfg);
    let mut u = vec![2.0_f64];
    let res = solver.solve(&form, &rhs, &mut u).unwrap();
    assert!(res.converged, "TR indefinite: {} iters res={:.3e}", res.iterations, res.final_residual);
}

#[test]
fn trust_region_convex_converges() {
    // Convex residual: F(u) = u²-2, J = 2u (positive for u>0)
    let form = TestQuadratic { n: 1, coeffs: vec![1.0, 0.0, -2.0] };
    let rhs = vec![0.0];
    let cfg = TrustRegionConfig { rtol: 1e-8, verbose: false, ..Default::default() };
    let solver = TrustRegionSolver::new(cfg);
    let mut u = vec![10.0_f64];
    let res = solver.solve(&form, &rhs, &mut u).unwrap();
    assert!(res.converged, "TR convex: {} iters res={:.3e}", res.iterations, res.final_residual);
}
