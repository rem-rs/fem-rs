#![cfg(feature = "gpu")]

use fem_core::Scalar;
use fem_linalg::CsrMatrix;
use fem_linalg_gpu::GpuContext;
use fem_solver::{SolverConfig, gmres_gpu::{GmresGpuWorkspace, solve_gmres_gpu, solve_gmres_gpu_f32}};

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

    let a = CsrMatrix { nrows: n, ncols: n, row_ptr, col_idx, values };
    let pi = std::f64::consts::PI;
    let x_exact: Vec<T> = (0..n)
        .map(|i| T::from_f64((pi * i as f64 / (n as f64 - 1.0)).sin()))
        .collect();

    let mut b = vec![T::zero(); n];
    for i in 0..n {
        let start = a.row_ptr[i];
        let end = a.row_ptr[i + 1];
        let mut sum = T::zero();
        for k in start..end {
            sum += a.values[k] * x_exact[a.col_idx[k] as usize];
        }
        b[i] = sum;
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
fn gmres_gpu_solves_poisson_1d() {
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
        let result = solve_gmres_gpu(&gpu, &a, &b, &mut x, &cfg).expect("GMRES should converge");
        assert!(result.converged, "GMRES did not converge in {} iters", result.iterations);
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
        let result = solve_gmres_gpu_f32(&gpu, &a, &b, &mut x, &cfg).expect("GMRES f32 should converge");
        assert!(result.converged, "GMRES did not converge in {} iters", result.iterations);
        assert!(max_error(&x, &x_exact) < 5e-4, "max error too large");
    }
}

#[test]
fn gmres_gpu_profile_fixed_iters_reports_segment_times() {
    let gpu = GpuContext::new_sync().expect("gpu context");
    let n = 64;

    if gpu.features.native_f64 {
        let (a, b, _) = poisson_1d::<f64>(n);
        let mut workspace = GmresGpuWorkspace::new(&gpu, &a, &b);
        let zero_x = vec![0.0f64; n];
        let profile = workspace.profile_fixed_iters(&gpu, &zero_x, 8);
        println!(
            "GMRES profile f64: total={:?}, residual={:?}, basis_seed={:?}, arnoldi_spmv={:?}, arnoldi_orth={:?}, arnoldi_norm={:?}, solution_update={:?}, finalization={:?}, final_residual={}",
            profile.total_phase,
            profile.residual_phase,
            profile.basis_seed_phase,
            profile.arnoldi_spmv_phase,
            profile.arnoldi_orthogonalization_phase,
            profile.arnoldi_normalization_phase,
            profile.solution_update_phase,
            profile.finalization_phase,
            profile.final_residual,
        );

        assert_eq!(profile.iterations, 8);
        assert!(profile.total_phase.as_nanos() > 0);
        assert!(profile.residual_phase.as_nanos() > 0);
        assert!(profile.basis_seed_phase.as_nanos() > 0);
        assert!(profile.arnoldi_spmv_phase.as_nanos() > 0);
        assert!(profile.arnoldi_orthogonalization_phase.as_nanos() > 0);
        assert!(profile.arnoldi_normalization_phase.as_nanos() > 0);
        assert!(profile.solution_update_phase.as_nanos() > 0);
    } else {
        let (a, b, _) = poisson_1d::<f32>(n);
        let mut workspace = GmresGpuWorkspace::new(&gpu, &a, &b);
        let zero_x = vec![0.0f32; n];
        let profile = workspace.profile_fixed_iters(&gpu, &zero_x, 8);
        println!(
            "GMRES profile f32: total={:?}, residual={:?}, basis_seed={:?}, arnoldi_spmv={:?}, arnoldi_orth={:?}, arnoldi_norm={:?}, solution_update={:?}, finalization={:?}, final_residual={}",
            profile.total_phase,
            profile.residual_phase,
            profile.basis_seed_phase,
            profile.arnoldi_spmv_phase,
            profile.arnoldi_orthogonalization_phase,
            profile.arnoldi_normalization_phase,
            profile.solution_update_phase,
            profile.finalization_phase,
            profile.final_residual,
        );

        assert_eq!(profile.iterations, 8);
        assert!(profile.total_phase.as_nanos() > 0);
        assert!(profile.residual_phase.as_nanos() > 0);
        assert!(profile.basis_seed_phase.as_nanos() > 0);
        assert!(profile.arnoldi_spmv_phase.as_nanos() > 0);
        assert!(profile.arnoldi_orthogonalization_phase.as_nanos() > 0);
        assert!(profile.arnoldi_normalization_phase.as_nanos() > 0);
        assert!(profile.solution_update_phase.as_nanos() > 0);
    }
}