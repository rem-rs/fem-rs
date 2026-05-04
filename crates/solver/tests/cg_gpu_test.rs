#![cfg(feature = "gpu")]

use fem_linalg::CsrMatrix;
use fem_linalg_gpu::GpuContext;
use fem_solver::{SolverConfig, cg_gpu::solve_cg_gpu};

/// Build a 1D Poisson matrix (tridiagonal [2, -1, 0, ...; -1, 2, -1, ...]).
fn poisson_1d(n: usize) -> (CsrMatrix<f64>, Vec<f64>, Vec<f64>) {
    let nnz = 3 * n - 2;
    let mut row_ptr = vec![0usize; n + 1];
    let mut col_idx = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);

    for i in 0..n {
        row_ptr[i + 1] = row_ptr[i];
        if i > 0 {
            col_idx.push(i as u32 - 1);
            values.push(-1.0);
            row_ptr[i + 1] += 1;
        }
        col_idx.push(i as u32);
        values.push(2.0);
        row_ptr[i + 1] += 1;
        if i + 1 < n {
            col_idx.push(i as u32 + 1);
            values.push(-1.0);
            row_ptr[i + 1] += 1;
        }
    }

    let a = CsrMatrix { nrows: n, ncols: n, row_ptr, col_idx, values };

    // Exact solution: x_i = sin(pi * i / (n-1))
    let pi = std::f64::consts::PI;
    let x_exact: Vec<f64> = (0..n).map(|i| (pi * i as f64 / (n as f64 - 1.0)).sin()).collect();

    // RHS b = A * x_exact
    let mut b = vec![0.0f64; n];
    for i in 0..n {
        let start = a.row_ptr[i];
        let end = a.row_ptr[i + 1];
        let mut s = 0.0;
        for k in start..end {
            s += a.values[k] * x_exact[a.col_idx[k] as usize];
        }
        b[i] = s;
    }

    (a, b, x_exact)
}

#[test]
fn cg_gpu_solves_poisson_1d() {
    let gpu = GpuContext::new_sync().expect("gpu context");
    let n = 64;
    let (a, b, x_exact) = poisson_1d(n);

    let cfg = SolverConfig {
        rtol: 1e-10,
        atol: 0.0,
        max_iter: 200,
        verbose: false,
        print_level: fem_solver::PrintLevel::Silent,
    };

    let mut x = vec![0.0f64; n];
    let result = solve_cg_gpu(&gpu, &a, &b, &mut x, &cfg).expect("CG should converge");

    assert!(result.converged, "CG did not converge in {} iters", result.iterations);
    assert!(result.iterations <= n, "CG took {} iterations (expected <= {n})", result.iterations);

    let mut max_err = 0.0f64;
    for i in 0..n {
        let err = (x[i] - x_exact[i]).abs();
        if err > max_err { max_err = err; }
    }
    assert!(max_err < 1e-8, "max error {max_err} > 1e-8");
}
