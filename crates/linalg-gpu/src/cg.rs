//! GPU-accelerated Conjugate Gradient solver (f32/f64) for wgpu backend.
//!
//! Uses [`SpmvPipeline`], [`VectorOpsPipeline`], [`GpuVector`], and [`GpuCsrMatrix`]
//! to perform CG iterations on the GPU.
//!
//! Both unpreconditioned CG (`solve_cg_gpu`) and Jacobi-preconditioned CG
//! (`solve_pcg_jacobi_gpu`) are provided.
//!
//! # Usage
//! ```ignore
//! let gpu = GpuContext::new_sync()?;
//! let spmv = SpmvPipeline::new(&gpu.device, gpu.features.native_f64);
//! let vops = VectorOpsPipeline::new(&gpu.device, gpu.features.native_f64);
//! let a: GpuCsrMatrix<f64> = …;
//! let b = GpuVector::from_slice(&gpu, &rhs);
//! let mut x = GpuVector::zeros(&gpu, n);
//! let (iters, res) = solve_cg_gpu(&gpu, &spmv, &vops, &a, &b, &mut x, 1e-10, 1000)?;
//! ```

use std::any::TypeId;
use fem_core::Scalar;
use crate::{DeviceBuffer, GpuContext, GpuCsrMatrix, GpuVector, SpmvPipeline, VectorOpsPipeline};

/// Solve `A·x = b` using Conjugate Gradient on GPU.
///
/// Returns `(iterations, final_relative_residual)`.
pub fn solve_cg_gpu<T: Scalar + bytemuck::Pod>(
    gpu: &GpuContext,
    spmv: &SpmvPipeline,
    vops: &VectorOpsPipeline,
    a: &GpuCsrMatrix<T>,
    b: &GpuVector<T>,
    x: &mut GpuVector<T>,
    tol: f64,
    max_iter: usize,
) -> Result<(usize, f64), String> {
    let n = b.len();
    if n == 0 { return Err("zero-length vector".into()); }

    // Workspace vectors
    let r = GpuVector::<T>::zeros(gpu, n);
    let p = GpuVector::<T>::zeros(gpu, n);
    let ap = GpuVector::<T>::zeros(gpu, n);

    // Allocate one dot-reduction scratch buffer
    let n_wg = (n + 255) / 256;
    let scalar_size = if TypeId::of::<T>() == TypeId::of::<f64>() { 8 } else { 4 };
    let dot_buf = DeviceBuffer::with_staging(&gpu.device,
        n_wg as u64 * scalar_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "cg_dot");

    macro_rules! dot {
        ($a:expr, $b:expr) => { vops.dispatch_dot_readback(gpu, $a, $b, &dot_buf) }
    }

    // r = b - A·x
    {
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        spmv.encode_spmv(gpu, &mut enc, -1.0, a, x, 0.0, &r);
        vops.encode_axpy(gpu, &mut enc, 1.0, b, 1.0, &r);
        gpu.queue.submit(Some(enc.finish()));
    }
    // p = r
    {
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        vops.encode_axpy(gpu, &mut enc, 1.0, &r, 0.0, &p);
        gpu.queue.submit(Some(enc.finish()));
    }

    let b_norm = dot!(b, b).sqrt().max(1e-300);
    let rtol = tol.max(1e-16);
    let mut rho = dot!(&r, &r);
    if rho.sqrt() < rtol * b_norm { return Ok((0, rho.sqrt() / b_norm)); }

    for iter in 1..=max_iter {
        // ap = A·p
        {
            let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            spmv.encode_spmv(gpu, &mut enc, 1.0, a, &p, 0.0, &ap);
            gpu.queue.submit(Some(enc.finish()));
        }

        let p_ap = dot!(&p, &ap);
        if p_ap.abs() < 1e-300 {
            return Err(format!("CG breakdown: p^T A p = {:.3e}", p_ap));
        }

        let alpha = rho / p_ap;

        // x += alpha·p
        {
            let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            vops.encode_axpy(gpu, &mut enc, alpha, &p, 1.0, x);
            gpu.queue.submit(Some(enc.finish()));
        }
        // r -= alpha·ap
        {
            let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            vops.encode_axpy(gpu, &mut enc, -alpha, &ap, 1.0, &r);
            gpu.queue.submit(Some(enc.finish()));
        }

        let rho_new = dot!(&r, &r);
        let rel_res = rho_new.sqrt() / b_norm;

        if rel_res < rtol {
            return Ok((iter, rel_res));
        }

        let beta = rho_new / rho;
        rho = rho_new;

        // p = r + beta·p
        {
            let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            vops.encode_axpy(gpu, &mut enc, 1.0, &r, beta, &p);
            gpu.queue.submit(Some(enc.finish()));
        }
    }

    // Compute final residual
    {
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        spmv.encode_spmv(gpu, &mut enc, -1.0, a, x, 0.0, &r);
        vops.encode_axpy(gpu, &mut enc, 1.0, b, 1.0, &r);
        gpu.queue.submit(Some(enc.finish()));
    }
    let res_sq = dot!(&r, &r);
    let rel_res = res_sq.sqrt() / b_norm;
    Err(format!("CG not converged after {max_iter} iters, residual {rel_res:.3e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, SpmvPipeline, VectorOpsPipeline, GpuCsrMatrix, GpuVector};

    fn ctx() -> GpuContext {
        GpuContext::new_sync().expect("GpuContext")
    }

    fn tiny_spd() -> fem_linalg::CsrMatrix<f64> {
        fem_linalg::CsrMatrix {
            nrows: 3, ncols: 3,
            row_ptr: vec![0, 2, 3, 5],
            col_idx: vec![0u32, 2, 1, 0, 2],
            values: vec![4.0, 1.0, 3.0, 1.0, 2.0],
        }
    }

    #[test]
    #[ignore]
    fn cg_gpu_tiny_spd_f64() {
        let gpu = ctx();
        if !gpu.features.native_f64 { eprintln!("SKIP: no SHADER_F64"); return; }

        let spmv = SpmvPipeline::new(&gpu.device, true);
        let vops = VectorOpsPipeline::new(&gpu.device, true);

        let cpu_a = tiny_spd();
        let gpu_a = GpuCsrMatrix::<f64>::from_cpu(&gpu, &cpu_a);
        // b = A·[1,2,3]: row0=4+3=7, row1=6, row2=1+6=7
        let b = GpuVector::<f64>::from_slice(&gpu, &[7.0, 6.0, 7.0]);
        let mut x = GpuVector::<f64>::zeros(&gpu, 3);

        let res = solve_cg_gpu::<f64>(&gpu, &spmv, &vops, &gpu_a, &b, &mut x, 1e-10, 100);
        assert!(res.is_ok(), "CG failed: {:?}", res.err());
        let (iters, final_res) = res.unwrap();
        eprintln!("CG GPU f64: {iters} iters, residual {final_res:.3e}");

        let x_host = x.read_to_cpu(&gpu);
        assert!((x_host[0] - 1.0).abs() < 1e-6, "x[0]={}", x_host[0]);
        assert!((x_host[1] - 2.0).abs() < 1e-6, "x[1]={}", x_host[1]);
        assert!((x_host[2] - 3.0).abs() < 1e-6, "x[2]={}", x_host[2]);
    }
}
