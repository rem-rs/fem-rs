//! GPU-resident restarted GMRES solver.
//!
//! Uses modified Gram-Schmidt for Arnoldi with m-step restart.
//! The Hessenberg least-squares is solved on CPU (tiny, O(m²)).

use fem_linalg::CsrMatrix;
use fem_linalg_gpu::{
    GpuContext, GpuCsrMatrix, GpuVector,
    SpmvPipeline, VectorOpsPipeline, read_partial_reduction,
};
use wgpu;
use crate::{SolverConfig, SolveResult, SolverError};

/// Default GMRES restart dimension.
const DEFAULT_RESTART: usize = 30;

/// Solve `A x = b` using restarted GMRES on the GPU.
pub fn solve_gmres_gpu(
    ctx: &GpuContext,
    a: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let n = a.nrows as u32;
    let restart = DEFAULT_RESTART.min(n as usize);

    let spmv = SpmvPipeline::new(&ctx.device, ctx.features.native_f64);
    let vops = VectorOpsPipeline::new(&ctx.device, ctx.features.native_f64);

    let gpu_a = GpuCsrMatrix::<f64>::from_cpu(ctx, a);
    let gpu_b = GpuVector::from_slice(ctx, b);
    let mut gpu_x = GpuVector::from_slice(ctx, x);

    let b_norm = vops.compute_norm2(ctx, &gpu_b);
    let tol = cfg.atol.max(cfg.rtol * b_norm);
    let n_wg = (n + 255) / 256;
    let dot_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gmres_dot"),
        size: n_wg as u64 * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut iter_count = 0usize;

    for _outer in 0..cfg.max_iter / restart + 1 {
        // r = b - A*x
        let gpu_r = compute_residual(ctx, &spmv, &vops, &gpu_a, &gpu_b, &gpu_x, n);
        let r_norm = vops.compute_norm2(ctx, &gpu_r);

        if r_norm < tol {
            let cpu_x = gpu_x.read_to_cpu(ctx);
            x.copy_from_slice(&cpu_x);
            return Ok(SolveResult { converged: true, iterations: iter_count, final_residual: r_norm });
        }

        // Arnoldi basis vectors: V[0..restart] each of length n
        let mut basis: Vec<GpuVector<f64>> = Vec::with_capacity(restart + 1);
        // V[0] = r / r_norm (scale in place)
        {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            vops.encode_axpy(ctx, &mut enc, 1.0 / r_norm, &gpu_r, 0.0, &gpu_r);
            ctx.queue.submit(Some(enc.finish()));
        }
        basis.push(gpu_r);

        // Hessenberg matrix (CPU, small)
        let mut h = vec![0.0f64; (restart + 1) * restart];
        let mut s = vec![0.0f64; restart + 1];
        let mut cs = vec![0.0f64; restart];
        let mut sn = vec![0.0f64; restart];

        s[0] = r_norm;
        let mut gmres_r_norm = r_norm;
        let mut j = 0usize;

        for jj in 0..restart {
            j = jj;
            // w = A * V[j]
            let mut gpu_w = GpuVector::<f64>::zeros(ctx, n);
            {
                let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                spmv.encode_spmv(ctx, &mut enc, 1.0, &gpu_a, &basis[jj], 0.0, &mut gpu_w);
                ctx.queue.submit(Some(enc.finish()));
            }

            // Modified Gram-Schmidt
            for i in 0..=jj {
                let dot_val = {
                    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    vops.encode_dot(ctx, &mut enc, &gpu_w, &basis[i], &dot_buf);
                    ctx.queue.submit(Some(enc.finish()));
                    read_partial_reduction(ctx, &dot_buf, n_wg)
                };
                h[i * restart + jj] = dot_val;
                // w = w - h_ij * V[i]
                {
                    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    vops.encode_axpy(ctx, &mut enc, -dot_val, &basis[i], 1.0, &mut gpu_w);
                    ctx.queue.submit(Some(enc.finish()));
                }
            }

            let w_norm = vops.compute_norm2(ctx, &gpu_w);
            h[(jj + 1) * restart + jj] = w_norm;

            // Apply previous Givens rotations to the new column of H
            for i in 0..jj {
                let hi = h[i * restart + jj];
                let hi1 = h[(i + 1) * restart + jj];
                h[i * restart + jj]     =  cs[i] * hi + sn[i] * hi1;
                h[(i + 1) * restart + jj] = -sn[i] * hi + cs[i] * hi1;
            }

            // Compute new Givens rotation
            let h_jj = h[jj * restart + jj];
            let h_j1j = h[(jj + 1) * restart + jj];
            let denom = (h_jj * h_jj + h_j1j * h_j1j).sqrt();
            if denom < 1e-30 {
                break;
            }
            cs[jj] = h_jj / denom;
            sn[jj] = h_j1j / denom;
            h[jj * restart + jj] = denom;
            h[(jj + 1) * restart + jj] = 0.0;

            // Apply to s
            let sj = s[jj];
            let sj1 = s[jj + 1];
            s[jj]     =  cs[jj] * sj + sn[jj] * sj1;
            s[jj + 1] = -sn[jj] * sj + cs[jj] * sj1;

            gmres_r_norm = s[jj + 1].abs();
            iter_count += 1;

            if gmres_r_norm < tol {
                break;
            }

            // V[j+1] = w / w_norm
            if w_norm > 1e-15 {
                let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                vops.encode_axpy(ctx, &mut enc, 1.0 / w_norm, &gpu_w, 0.0, &gpu_w);
                ctx.queue.submit(Some(enc.finish()));
            }
            basis.push(gpu_w);
        }

        // Back-substitute on CPU: solve H * y = s
        let mut y = vec![0.0f64; j + 1];
        for ii in (0..=j).rev() {
            let mut sum = s[ii];
            for kk in ii + 1..=j {
                sum -= h[ii * restart + kk] * y[kk];
            }
            y[ii] = sum / h[ii * restart + ii];
        }

        // x = x + sum(y[i] * V[i])
        for i in 0..=j {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            vops.encode_axpy(ctx, &mut enc, y[i], &basis[i], 1.0, &mut gpu_x);
            ctx.queue.submit(Some(enc.finish()));
        }

        if gmres_r_norm < tol {
            let cpu_x = gpu_x.read_to_cpu(ctx);
            x.copy_from_slice(&cpu_x);
            return Ok(SolveResult { converged: true, iterations: iter_count, final_residual: gmres_r_norm });
        }
    }

    let cpu_x = gpu_x.read_to_cpu(ctx);
    x.copy_from_slice(&cpu_x);
    Err(SolverError::ConvergenceFailed { max_iter: cfg.max_iter, residual: 0.0 })
}

fn compute_residual(
    ctx: &GpuContext,
    spmv: &SpmvPipeline,
    vops: &VectorOpsPipeline,
    a: &GpuCsrMatrix<f64>,
    b: &GpuVector<f64>,
    x: &GpuVector<f64>,
    n: u32,
) -> GpuVector<f64> {
    let mut ax = GpuVector::<f64>::zeros(ctx, n);
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        spmv.encode_spmv(ctx, &mut enc, 1.0, a, x, 0.0, &mut ax);
        ctx.queue.submit(Some(enc.finish()));
    }
    let mut r = GpuVector::<f64>::zeros(ctx, n);
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        vops.encode_axpy(ctx, &mut enc, 1.0, b, 0.0, &mut r);
        ctx.queue.submit(Some(enc.finish()));
    }
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        vops.encode_axpy(ctx, &mut enc, -1.0, &ax, 1.0, &mut r);
        ctx.queue.submit(Some(enc.finish()));
    }
    r
}
