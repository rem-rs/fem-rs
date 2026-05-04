//! GPU-resident Conjugate Gradient solver.
//!
//! All vectors live on the GPU; only the residual norm is read back each
//! iteration for convergence checking.

use fem_linalg::CsrMatrix;
use fem_linalg_gpu::{
    GpuContext, GpuCsrMatrix, GpuVector,
    SpmvPipeline, VectorOpsPipeline, read_partial_reduction,
};
use wgpu;
use crate::{SolverConfig, SolveResult, SolverError};

/// Solve `A x = b` using Conjugate Gradient, with all iteration data on the GPU.
///
/// # Arguments
/// * `ctx`     — initialized GPU context with device and queue.
/// * `a`       — system matrix (CPU CSR, uploaded once).
/// * `b`       — right-hand side (CPU slice, uploaded once).
/// * `x`       — initial guess on entry (CPU slice), solution on exit (overwritten).
/// * `cfg`     — convergence parameters.
pub fn solve_cg_gpu(
    ctx: &GpuContext,
    a: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let n = a.nrows as u32;
    assert_eq!(a.ncols as u32, n, "matrix must be square");
    assert_eq!(b.len() as u32, n);
    assert_eq!(x.len() as u32, n);

    let spmv_pipeline = SpmvPipeline::new(&ctx.device, ctx.features.native_f64);
    let vec_pipeline = VectorOpsPipeline::new(&ctx.device, ctx.features.native_f64);

    // Upload matrix and vectors
    let gpu_a = GpuCsrMatrix::<f64>::from_cpu(ctx, a);
    let gpu_b = GpuVector::from_slice(ctx, b);
    let mut gpu_x = GpuVector::from_slice(ctx, x);
    let mut gpu_r = GpuVector::<f64>::zeros(ctx, n);
    let mut gpu_p = GpuVector::<f64>::zeros(ctx, n);
    let mut gpu_ap = GpuVector::<f64>::zeros(ctx, n);

    // r = b - A*x  (three submits: tmp=Ax, r=b, r=r-tmp)
    let mut gpu_tmp = GpuVector::<f64>::zeros(ctx, n);
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        spmv_pipeline.encode_spmv(ctx, &mut enc, 1.0, &gpu_a, &gpu_x, 0.0, &mut gpu_tmp);
        ctx.queue.submit(Some(enc.finish()));
    }
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        vec_pipeline.encode_axpy(ctx, &mut enc, 1.0, &gpu_b, 0.0, &mut gpu_r);
        ctx.queue.submit(Some(enc.finish()));
    }
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        vec_pipeline.encode_axpy(ctx, &mut enc, -1.0, &gpu_tmp, 1.0, &mut gpu_r);
        ctx.queue.submit(Some(enc.finish()));
    }
    // p = r
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        vec_pipeline.encode_axpy(ctx, &mut enc, 1.0, &gpu_r, 0.0, &mut gpu_p);
        ctx.queue.submit(Some(enc.finish()));
    }

    // rsold = r·r via dot reduction
    let n_wg = (n + 255) / 256;
    let dot_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cg_dot_buf"),
        size: n_wg as u64 * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut rsold = {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        vec_pipeline.encode_dot(ctx, &mut enc, &gpu_r, &gpu_r, &dot_buf);
        ctx.queue.submit(Some(enc.finish()));
        read_partial_reduction(ctx, &dot_buf, n_wg)
    };

    let b_norm = vec_pipeline.compute_norm2(ctx, &gpu_b);
    let tol = cfg.atol.max(cfg.rtol * b_norm);

    for iter in 0..cfg.max_iter {
        // ap = A * p
        {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            spmv_pipeline.encode_spmv(ctx, &mut enc, 1.0, &gpu_a, &gpu_p, 0.0, &mut gpu_ap);
            ctx.queue.submit(Some(enc.finish()));
        }

        // pAp = p · ap
        let pap = {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            vec_pipeline.encode_dot(ctx, &mut enc, &gpu_p, &gpu_ap, &dot_buf);
            ctx.queue.submit(Some(enc.finish()));
            read_partial_reduction(ctx, &dot_buf, n_wg)
        };

        let alpha = rsold / pap;

        // x = x + alpha * p
        {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            vec_pipeline.encode_axpy(ctx, &mut enc, alpha, &gpu_p, 1.0, &mut gpu_x);
            ctx.queue.submit(Some(enc.finish()));
        }

        // r = r - alpha * ap
        {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            vec_pipeline.encode_axpy(ctx, &mut enc, -alpha, &gpu_ap, 1.0, &mut gpu_r);
            ctx.queue.submit(Some(enc.finish()));
        }

        // rsnew = r·r
        let rsnew = {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            vec_pipeline.encode_dot(ctx, &mut enc, &gpu_r, &gpu_r, &dot_buf);
            ctx.queue.submit(Some(enc.finish()));
            read_partial_reduction(ctx, &dot_buf, n_wg)
        };

        // Convergence check
        let r_norm = rsnew.sqrt();
        if r_norm < tol {
            // Read solution back
            let cpu_x = gpu_x.read_to_cpu(ctx);
            x.copy_from_slice(&cpu_x);
            return Ok(SolveResult {
                converged: true,
                iterations: iter + 1,
                final_residual: r_norm,
            });
        }

        let beta = rsnew / rsold;
        rsold = rsnew;

        // p = r + beta * p
        {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            vec_pipeline.encode_axpy(ctx, &mut enc, 1.0, &gpu_r, beta, &mut gpu_p);
            ctx.queue.submit(Some(enc.finish()));
        }
    }

    // Did not converge
    let cpu_x = gpu_x.read_to_cpu(ctx);
    x.copy_from_slice(&cpu_x);
    let final_r = vec_pipeline.compute_norm2(ctx, &gpu_r);
    Err(SolverError::ConvergenceFailed {
        max_iter: cfg.max_iter,
        residual: final_r,
    })
}
