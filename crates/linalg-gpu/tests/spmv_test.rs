// crates/linalg-gpu/tests/spmv_test.rs
use fem_core::Scalar;
use fem_linalg::CsrMatrix;
use fem_linalg_gpu::{GpuContext, GpuCsrMatrix, GpuVector, SpmvPipeline};

fn ctx() -> GpuContext {
    GpuContext::new_sync().expect("gpu context")
}

/// 3×3 SPD matrix, manually verified.
fn tiny_spd<T: Scalar>() -> CsrMatrix<T> {
    CsrMatrix {
        nrows: 3,
        ncols: 3,
        row_ptr: vec![0, 2, 3, 5],
        col_idx: vec![0u32, 2, 1, 0, 2],
        values: vec![2.0, 1.0, 3.0, 1.0, 4.0].into_iter().map(T::from_f64).collect(),
    }
}

fn cpu_spmv<T: Scalar>(a: &CsrMatrix<T>, x: &[T], y: &mut [T]) {
    for row in 0..a.nrows {
        let start = a.row_ptr[row];
        let end = a.row_ptr[row + 1];
        let mut s = T::zero();
        for k in start..end {
            s += a.values[k] * x[a.col_idx[k] as usize];
        }
        y[row] = s;
    }
}

fn assert_close<T: Scalar>(actual: T, expected: T, tol: f64, label: &str) {
    let diff = (actual - expected).abs();
    assert!(diff < T::from_f64(tol), "{label}: actual={actual} expected={expected} diff={diff}");
}

fn run_spmv_matches_cpu<T: Scalar>(gpu: &GpuContext, tol: f64) {
    let cpu_mat = tiny_spd::<T>();
    let gpu_mat = GpuCsrMatrix::<T>::from_cpu(gpu, &cpu_mat);
    let x_host = [T::from_f64(1.0), T::from_f64(2.0), T::from_f64(3.0)];
    let x = GpuVector::from_slice(gpu, &x_host);
    let gpu_y = GpuVector::<T>::zeros(gpu, 3);

    let pipeline = SpmvPipeline::new(&gpu.device, gpu.features.native_f64);
    pipeline.spmv(gpu, 1.0, &gpu_mat, &x, 0.0, &gpu_y);

    let gpu_result = gpu_y.read_to_cpu(gpu);

    let mut cpu_result = vec![T::zero(); 3];
    cpu_spmv(&cpu_mat, &x_host, &mut cpu_result);

    for i in 0..3 {
        assert_close(gpu_result[i], cpu_result[i], tol, &format!("row {i}"));
    }
}

fn run_spmv_with_alpha_beta<T: Scalar>(gpu: &GpuContext, tol: f64) {
    let cpu_mat = tiny_spd::<T>();
    let gpu_mat = GpuCsrMatrix::<T>::from_cpu(gpu, &cpu_mat);
    let x = GpuVector::from_slice(gpu, &[T::from_f64(1.0), T::from_f64(0.0), T::from_f64(0.0)]);
    let gpu_y = GpuVector::from_slice(gpu, &[T::from_f64(2.0), T::from_f64(2.0), T::from_f64(2.0)]);

    let pipeline = SpmvPipeline::new(&gpu.device, gpu.features.native_f64);
    pipeline.spmv(gpu, 3.0, &gpu_mat, &x, 0.5, &gpu_y);

    let gpu_result = gpu_y.read_to_cpu(gpu);
    assert_close(gpu_result[0], T::from_f64(7.0), tol, "y[0]");
    assert_close(gpu_result[1], T::from_f64(1.0), tol, "y[1]");
    assert_close(gpu_result[2], T::from_f64(4.0), tol, "y[2]");
}

#[test]
fn spmv_matches_cpu() {
    let gpu = ctx();
    if gpu.features.native_f64 {
        run_spmv_matches_cpu::<f64>(&gpu, 1e-14);
    } else {
        run_spmv_matches_cpu::<f32>(&gpu, 1e-5);
    }
}

#[test]
fn spmv_with_alpha_beta() {
    let gpu = ctx();
    if gpu.features.native_f64 {
        run_spmv_with_alpha_beta::<f64>(&gpu, 1e-14);
    } else {
        run_spmv_with_alpha_beta::<f32>(&gpu, 1e-5);
    }
}
