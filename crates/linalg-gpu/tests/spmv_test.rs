// crates/linalg-gpu/tests/spmv_test.rs
use fem_linalg::CsrMatrix;
use fem_linalg_gpu::{GpuContext, GpuCsrMatrix, GpuVector, SpmvPipeline};

fn ctx() -> GpuContext {
    GpuContext::new_sync().expect("gpu context")
}

/// 3×3 SPD matrix, manually verified.
fn tiny_spd() -> CsrMatrix<f64> {
    CsrMatrix {
        nrows: 3,
        ncols: 3,
        row_ptr: vec![0, 2, 3, 5],
        col_idx: vec![0u32, 2, 1, 0, 2],
        values: vec![2.0, 1.0, 3.0, 1.0, 4.0],
    }
}

fn cpu_spmv(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    for row in 0..a.nrows {
        let start = a.row_ptr[row];
        let end = a.row_ptr[row + 1];
        let mut s = 0.0;
        for k in start..end {
            s += a.values[k] * x[a.col_idx[k] as usize];
        }
        y[row] = s;
    }
}

#[test]
fn spmv_matches_cpu() {
    let gpu = ctx();
    let cpu_mat = tiny_spd();
    let gpu_mat = GpuCsrMatrix::<f64>::from_cpu(&gpu, &cpu_mat);
    let x = GpuVector::from_slice(&gpu, &[1.0, 2.0, 3.0]);
    let gpu_y = GpuVector::<f64>::zeros(&gpu, 3);

    let pipeline = SpmvPipeline::new(&gpu.device, gpu.features.native_f64);
    pipeline.spmv(&gpu,
        1.0, &gpu_mat, &x,
        0.0, &gpu_y,
    );

    let gpu_result = gpu_y.read_to_cpu(&gpu);

    let mut cpu_result = vec![0.0; 3];
    cpu_spmv(&cpu_mat, &[1.0, 2.0, 3.0], &mut cpu_result);

    for i in 0..3 {
        let diff = (gpu_result[i] - cpu_result[i]).abs();
        assert!(diff < 1e-14, "row {i}: gpu={} cpu={} diff={}", gpu_result[i], cpu_result[i], diff);
    }
}

#[test]
fn spmv_with_alpha_beta() {
    let gpu = ctx();
    let cpu_mat = tiny_spd();
    let gpu_mat = GpuCsrMatrix::<f64>::from_cpu(&gpu, &cpu_mat);
    let x = GpuVector::from_slice(&gpu, &[1.0, 0.0, 0.0]);
    // Start y = [2, 2, 2]
    let gpu_y = GpuVector::from_slice(&gpu, &[2.0, 2.0, 2.0]);

    let pipeline = SpmvPipeline::new(&gpu.device, gpu.features.native_f64);
    // y = 3*A*x + 0.5*y
    pipeline.spmv(&gpu,
        3.0, &gpu_mat, &x,
        0.5, &gpu_y,
    );

    let gpu_result = gpu_y.read_to_cpu(&gpu);

    // A*x (column 0) = [2, 0, 1]
    // y = 3*[2,0,1] + 0.5*[2,2,2] = [6+1, 0+1, 3+1] = [7, 1, 4]
    assert!((gpu_result[0] - 7.0).abs() < 1e-14);
    assert!((gpu_result[1] - 1.0).abs() < 1e-14);
    assert!((gpu_result[2] - 4.0).abs() < 1e-14);
}
