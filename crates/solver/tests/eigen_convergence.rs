use fem_solver::{
    lobpcg, krylov_schur, LobpcgConfig,
};
use fem_linalg::{CooMatrix, CsrMatrix};

/// 1-D Laplacian tridiagonal matrix [-1, 2, -1] of size n.
fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        coo.add(i, i, 2.0);
        if i > 0   { coo.add(i, i-1, -1.0); }
        if i < n-1 { coo.add(i, i+1, -1.0); }
    }
    coo.into_csr()
}

/// Identity matrix.
fn identity(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n { coo.add(i, i, 1.0); }
    coo.into_csr()
}

/// Exact k-th eigenvalue (ascending) of the 1-D Laplacian of size n.
fn exact_laplacian_eigenvalue(k: usize, n: usize) -> f64 {
    2.0 - 2.0 * (std::f64::consts::PI * k as f64 / (n as f64 + 1.0)).cos()
}

// ─── Spectrum verification ──────────────────────────────────────────────

#[test]
fn lobpcg_smallest_three_match_analytical() {
    let n = 30;
    let a = laplacian_1d(n);
    let cfg = LobpcgConfig { max_iter: 500, tol: 1e-8, verbose: false, nullspace_skip: 0.0 };
    let res = lobpcg(&a, None, 3, &cfg).unwrap();

    for k in 0..3 {
        let exact = exact_laplacian_eigenvalue(k + 1, n);
        let err = (res.eigenvalues[k] - exact).abs();
        assert!(err < 1e-6,
            "λ[{k}] computed={:.8e}, exact={exact:.8e}, err={err:.2e}", res.eigenvalues[k]);
    }
}

#[test]
fn lobpcg_first_eigenvalue_converges_with_tolerance() {
    // For a 10×10 matrix, eigen gap is large enough that we can get high accuracy.
    let n = 10;
    let a = laplacian_1d(n);
    let exact = exact_laplacian_eigenvalue(1, n);

    for &tol in &[1e-4, 1e-6, 1e-8] {
        let cfg = LobpcgConfig { max_iter: 500, tol, verbose: false, nullspace_skip: 0.0 };
        let res = lobpcg(&a, None, 1, &cfg).unwrap();
        let err = (res.eigenvalues[0] - exact).abs();
        assert!(err < tol * 10.0,
            "tol={tol:.0e}: eigenvalue err={err:.2e} (should be ~tol)");
    }
}

#[test]
fn lobpcg_eigenvalues_are_sorted_ascending() {
    let n = 20;
    let a = laplacian_1d(n);
    let cfg = LobpcgConfig { max_iter: 500, tol: 1e-8, verbose: false, nullspace_skip: 0.0 };
    let res = lobpcg(&a, None, 5, &cfg).unwrap();
    for i in 0..res.eigenvalues.len() - 1 {
        assert!(res.eigenvalues[i] < res.eigenvalues[i + 1],
            "λ[{i}]={:.6e} should be < λ[{}]={:.6e}", res.eigenvalues[i], i+1, res.eigenvalues[i+1]);
    }
}

#[test]
fn lobpcg_eigenvectors_are_orthonormal() {
    let n = 20;
    let a = laplacian_1d(n);
    let cfg = LobpcgConfig { max_iter: 500, tol: 1e-8, verbose: false, nullspace_skip: 0.0 };
    let res = lobpcg(&a, None, 4, &cfg).unwrap();
    let xtx = res.eigenvectors.transpose() * &res.eigenvectors;
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let err = (xtx[(i, j)] - expected).abs();
            assert!(err < 1e-8,
                "XᵀX[{i},{j}] = {:.2e}, expected {expected}", xtx[(i,j)]);
        }
    }
}

#[test]
fn lobpcg_rayleigh_quotient_matches_eigenvalue() {
    // For each eigenpair (λ�? v�?, verify (vᵢᵀ A v�? / (vᵢᵀ v�? �?λ�?
    let n = 20;
    let a = laplacian_1d(n);
    let cfg = LobpcgConfig { max_iter: 500, tol: 1e-8, verbose: false, nullspace_skip: 0.0 };
    let res = lobpcg(&a, None, 3, &cfg).unwrap();

    for i in 0..3 {
        // Extract column as Vec for manual Rayleigh quotient
        let v_vec: Vec<f64> = (0..n).map(|r| res.eigenvectors[(r, i)]).collect();
        let v_dot = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(a, b)| a * b).sum::<f64>();
        let mut av = vec![0.0; n];
        // Manual SpMV with 1D Laplacian
        for r in 0..n {
            av[r] = 2.0 * v_vec[r];
            if r > 0   { av[r] -= v_vec[r - 1]; }
            if r < n-1 { av[r] -= v_vec[r + 1]; }
        }
        let rq = v_dot(&av, &v_vec) / v_dot(&v_vec, &v_vec);
        let err = (rq - res.eigenvalues[i]).abs();
        assert!(err < 1e-10,
            "Rayleigh quotient mismatch for λ[{i}]: rq={rq:.10e}, λ={:.10e}, err={err:.2e}",
            res.eigenvalues[i]);
    }
}

// ─── Generalized eigenvalue ─────────────────────────────────────────────

#[test]
fn lobpcg_generalized_known_spectrum() {
    // A = diag(1, 2, ..., 8), B = I
    // Eigenvalues are 1, 2, ..., 8
    let n = 8;
    let mut coo_a = CooMatrix::<f64>::new(n, n);
    for i in 0..n { coo_a.add(i, i, (i + 1) as f64); }
    let a = coo_a.into_csr();
    let b = identity(n);
    let cfg = LobpcgConfig { max_iter: 300, tol: 1e-10, verbose: false, nullspace_skip: 0.0 };
    let res = lobpcg(&a, Some(&b), 3, &cfg).unwrap();

    assert!((res.eigenvalues[0] - 1.0).abs() < 1e-8, "λ₀={:.6e} �?1", res.eigenvalues[0]);
    assert!((res.eigenvalues[1] - 2.0).abs() < 1e-8, "λ�?{:.6e} �?2", res.eigenvalues[1]);
    assert!((res.eigenvalues[2] - 3.0).abs() < 1e-8, "λ�?{:.6e} �?3", res.eigenvalues[2]);
}

// ─── Krylov-Schur ───────────────────────────────────────────────────────

#[test]
fn krylov_schur_returns_k_eigenvalues() {
    // Krylov-Schur is a hybrid method; validate it runs and returns k values.
    let n = 15;
    let a = laplacian_1d(n);
    let res = krylov_schur(&a, 3, Some(10)).unwrap();
    assert_eq!(res.eigenvalues.len(), 3, "should return 3 eigenvalues");
    // Eigenvalues should be positive (1D Laplacian is SPD)
    for &lam in &res.eigenvalues {
        assert!(lam > 0.0, "KrylovSchur eigenvalue should be positive: {lam}");
    }
}

// ─── Large problem smoke ────────────────────────────────────────────────

#[test]
fn lobpcg_scales_to_100x100() {
    let n = 100;
    let a = laplacian_1d(n);
    let cfg = LobpcgConfig { max_iter: 1000, tol: 1e-6, verbose: false, nullspace_skip: 0.0 };
    let res = lobpcg(&a, None, 3, &cfg).unwrap();
    for k in 0..3 {
        let exact = exact_laplacian_eigenvalue(k + 1, n);
        let err = (res.eigenvalues[k] - exact).abs();
        assert!(err < 1e-4,
            "n={n}, λ[{k}] computed={:.6e}, exact={exact:.6e}, err={err:.2e}", res.eigenvalues[k]);
    }
}

// ─── Residual verification ──────────────────────────────────────────────

#[test]
fn lobpcg_eigenpair_residual_small() {
    // For each eigenpair, verify ‖A v - λ v�?/ ‖A�?is small.
    let n = 30;
    let a = laplacian_1d(n);
    let cfg = LobpcgConfig { max_iter: 500, tol: 1e-8, verbose: false, nullspace_skip: 0.0 };
    let res = lobpcg(&a, None, 3, &cfg).unwrap();

    // Estimate ‖A�?as the largest eigenvalue (Gershgorin bound: 4 for 1D Laplacian)
    let a_norm = 4.0;

    for i in 0..3 {
        let v_vec: Vec<f64> = (0..n).map(|r| res.eigenvectors[(r, i)]).collect();
        let mut av = vec![0.0; n];
        for r in 0..n {
            av[r] = 2.0 * v_vec[r];
            if r > 0   { av[r] -= v_vec[r - 1]; }
            if r < n-1 { av[r] -= v_vec[r + 1]; }
        }
        // residual = ‖Av - λv‖₂ / ‖A�?
        let res_norm: f64 = av.iter().zip(v_vec.iter())
            .map(|(a, b)| (a - res.eigenvalues[i] * b).powi(2)).sum::<f64>().sqrt();
        let residual = res_norm / a_norm;
        assert!(residual < 1e-6,
            "residual for λ[{i}]={:.6e}: ‖Av-λv�?‖A�?{residual:.2e}", res.eigenvalues[i]);
    }
}
