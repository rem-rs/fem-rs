#![cfg(feature = "gpu")]

use fem_core::Scalar;
use fem_linalg::CsrMatrix;
use fem_linalg_gpu::GpuContext;
use fem_solver::{
    cg_gpu::{solve_cg_gpu, solve_cg_gpu_f32},
    SolverConfig,
};

/// Build a 1D Poisson matrix (tridiagonal [2, -1, 0, ...; -1, 2, -1, ...]).
fn poisson_1d<T: Scalar>(n: usize) -> (CsrMatrix<T>, Vec<T>, Vec<T>) {
    let nnz = 3 * n - 2;
    let mut row_ptr = vec![0usize; n + 1];
    let mut col_idx = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);

    for i in 0..n {
        row_ptr[i + 1] = row_ptr[i];
        if i > 0 {
            col_idx.push(i as u32 - 1);
            values.push(T::from_f64(-1.0));
            row_ptr[i + 1] += 1;
        }
        col_idx.push(i as u32);
        values.push(T::from_f64(2.0));
        row_ptr[i + 1] += 1;
        if i + 1 < n {
            col_idx.push(i as u32 + 1);
            values.push(T::from_f64(-1.0));
            row_ptr[i + 1] += 1;
        }
    }

    let a = CsrMatrix {
        nrows: n,
        ncols: n,
        row_ptr,
        col_idx,
        values,
    };

    // Exact solution: x_i = sin(pi * i / (n-1))
    let pi = std::f64::consts::PI;
    let x_exact: Vec<T> = (0..n)
        .map(|i| T::from_f64((pi * i as f64 / (n as f64 - 1.0)).sin()))
        .collect();

    // RHS b = A * x_exact
    let mut b = vec![T::zero(); n];
    for i in 0..n {
        let start = a.row_ptr[i];
        let end = a.row_ptr[i + 1];
        let mut s = T::zero();
        for k in start..end {
            s += a.values[k] * x_exact[a.col_idx[k] as usize];
        }
        b[i] = s;
    }

    (a, b, x_exact)
}

fn max_error<T: Scalar>(x: &[T], x_exact: &[T]) -> f64 {
    x.iter()
        .zip(x_exact.iter())
        .map(|(lhs, rhs)| (*lhs - *rhs).abs().to_f64().unwrap())
        .fold(0.0_f64, f64::max)
}

#[test]
fn cg_gpu_solves_poisson_1d() {
    let gpu = GpuContext::new_sync().expect("gpu context");
    let n = 64;
    if gpu.features.native_f64 {
        let (a, b, x_exact) = poisson_1d::<f64>(n);
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 200,
            verbose: false,
            print_level: fem_solver::PrintLevel::Silent,
        };

        let mut x = vec![0.0f64; n];
        let result = solve_cg_gpu(&gpu, &a, &b, &mut x, &cfg).expect("CG should converge");
        assert!(
            result.converged,
            "CG did not converge in {} iters",
            result.iterations
        );
        assert!(
            result.iterations <= n,
            "CG took {} iterations (expected <= {n})",
            result.iterations
        );
        assert!(max_error(&x, &x_exact) < 1e-8, "max error too large");
    } else {
        let (a, b, x_exact) = poisson_1d::<f32>(n);
        let cfg = SolverConfig {
            rtol: 1e-5,
            atol: 1e-6,
            max_iter: 200,
            verbose: false,
            print_level: fem_solver::PrintLevel::Silent,
        };

        let mut x = vec![0.0f32; n];
        let result = solve_cg_gpu_f32(&gpu, &a, &b, &mut x, &cfg).expect("CG f32 should converge");
        assert!(
            result.converged,
            "CG did not converge in {} iters",
            result.iterations
        );
        assert!(
            result.iterations <= n,
            "CG took {} iterations (expected <= {n})",
            result.iterations
        );
        assert!(max_error(&x, &x_exact) < 5e-4, "max error too large");
    }
}
