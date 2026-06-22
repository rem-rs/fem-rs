//! GPU-resident Conjugate Gradient solver.
//!
//! All vectors live on the GPU; only the residual norm is read back each
//! iteration for convergence checking.

use fem_core::Scalar;
use fem_linalg::CsrMatrix;
use fem_linalg_gpu::{
    DeviceBuffer, GpuContext, GpuCsrMatrix, GpuVector,
    SpmvPipeline, VectorOpsPipeline,
};
use wgpu;
use crate::{SolverConfig, SolveResult, SolverError};

/// Reusable GPU CG workspace that keeps uploaded data and temporary buffers on device.
pub struct CgGpuWorkspace<T: Scalar> {
    n: u32,
    spmv_pipeline: SpmvPipeline,
    vec_pipeline: VectorOpsPipeline,
    gpu_a: GpuCsrMatrix<T>,
    gpu_b: GpuVector<T>,
    gpu_x: GpuVector<T>,
    gpu_r: GpuVector<T>,
    gpu_p: GpuVector<T>,
    gpu_ap: GpuVector<T>,
    gpu_tmp: GpuVector<T>,
    dot_buf: DeviceBuffer,
    b_norm: f64,
}

impl<T: Scalar> CgGpuWorkspace<T> {
    /// Build a reusable GPU CG workspace for a fixed matrix and right-hand side.
    pub fn new(ctx: &GpuContext, a: &CsrMatrix<T>, b: &[T]) -> Self {
        let n = a.nrows as u32;
        assert_eq!(a.ncols as u32, n, "matrix must be square");
        assert_eq!(b.len() as u32, n);

        let spmv_pipeline = SpmvPipeline::new(&ctx.device, ctx.features.native_f64);
        let vec_pipeline = VectorOpsPipeline::new(&ctx.device, ctx.features.native_f64);
        let gpu_a = GpuCsrMatrix::<T>::from_cpu(ctx, a);
        let gpu_b = GpuVector::from_slice(ctx, b);
        let gpu_x = GpuVector::<T>::zeros(ctx, n);
        let gpu_r = GpuVector::<T>::zeros(ctx, n);
        let gpu_p = GpuVector::<T>::zeros(ctx, n);
        let gpu_ap = GpuVector::<T>::zeros(ctx, n);
        let gpu_tmp = GpuVector::<T>::zeros(ctx, n);
        let n_wg = (n + 255) / 256;
        let dot_buf = DeviceBuffer::with_staging(
            &ctx.device,
            n_wg as u64 * std::mem::size_of::<T>() as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "cg_dot_buf",
        );
        let b_norm = vec_pipeline.compute_norm2(ctx, &gpu_b);

        Self {
            n,
            spmv_pipeline,
            vec_pipeline,
            gpu_a,
            gpu_b,
            gpu_x,
            gpu_r,
            gpu_p,
            gpu_ap,
            gpu_tmp,
            dot_buf,
            b_norm,
        }
    }

    /// Solve using the pre-uploaded matrix/rhs and reusable temporaries.
    pub fn solve(&mut self, ctx: &GpuContext, x: &mut [T], cfg: &SolverConfig) -> Result<SolveResult, SolverError> {
        assert_eq!(x.len() as u32, self.n);
        self.gpu_x.write_from_slice(ctx, x);
        solve_cg_gpu_prepared(
            ctx,
            &self.spmv_pipeline,
            &self.vec_pipeline,
            &self.gpu_a,
            &self.gpu_b,
            &self.gpu_x,
            &self.gpu_r,
            &self.gpu_p,
            &mut self.gpu_ap,
            &self.gpu_tmp,
            &self.dot_buf,
            self.b_norm,
            None,
            x,
            cfg,
        )
    }

    /// Run a fixed number of CG iterations without early convergence exit.
    /// Returns the final residual norm after the last iteration.
    pub fn solve_fixed_iters(&mut self, ctx: &GpuContext, x: &mut [T], iterations: usize) -> f64 {
        assert_eq!(x.len() as u32, self.n);
        self.gpu_x.write_from_slice(ctx, x);
        let cfg = SolverConfig {
            rtol: 0.0,
            atol: 0.0,
            max_iter: iterations,
            verbose: false,
            print_level: crate::PrintLevel::Silent,
        };
        match solve_cg_gpu_prepared(
            ctx,
            &self.spmv_pipeline,
            &self.vec_pipeline,
            &self.gpu_a,
            &self.gpu_b,
            &self.gpu_x,
            &self.gpu_r,
            &self.gpu_p,
            &mut self.gpu_ap,
            &self.gpu_tmp,
            &self.dot_buf,
            self.b_norm,
            Some(iterations),
            x,
            &cfg,
        ) {
            Ok(result) => result.final_residual,
            Err(SolverError::ConvergenceFailed { residual, .. }) => residual,
            Err(err) => panic!("fixed-iteration CG run failed unexpectedly: {err}"),
        }
    }
}

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
    solve_cg_gpu_impl(ctx, a, b, x, cfg)
}

/// Solve `A x = b` using Conjugate Gradient in GPU `f32` mode.
pub fn solve_cg_gpu_f32(
    ctx: &GpuContext,
    a: &CsrMatrix<f32>,
    b: &[f32],
    x: &mut [f32],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    solve_cg_gpu_impl(ctx, a, b, x, cfg)
}

fn solve_cg_gpu_impl<T: Scalar>(
    ctx: &GpuContext,
    a: &CsrMatrix<T>,
    b: &[T],
    x: &mut [T],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let mut workspace = CgGpuWorkspace::new(ctx, a, b);
    workspace.solve(ctx, x, cfg)
}

#[allow(clippy::too_many_arguments)]
fn solve_cg_gpu_prepared<T: Scalar>(
    ctx: &GpuContext,
    spmv_pipeline: &SpmvPipeline,
    vec_pipeline: &VectorOpsPipeline,
    gpu_a: &GpuCsrMatrix<T>,
    gpu_b: &GpuVector<T>,
    gpu_x: &GpuVector<T>,
    gpu_r: &GpuVector<T>,
    gpu_p: &GpuVector<T>,
    gpu_ap: &mut GpuVector<T>,
    gpu_tmp: &GpuVector<T>,
    dot_buf: &DeviceBuffer,
    b_norm: f64,
    fixed_iterations: Option<usize>,
    x: &mut [T],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    // r = b - A*x  (three submits: tmp=Ax, r=b, r=r-tmp)
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        spmv_pipeline.encode_spmv(ctx, &mut enc, 1.0, gpu_a, gpu_x, 0.0, gpu_tmp);
        ctx.queue.submit(Some(enc.finish()));
    }
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        vec_pipeline.encode_axpy(ctx, &mut enc, 1.0, gpu_b, 0.0, gpu_r);
        ctx.queue.submit(Some(enc.finish()));
    }
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        vec_pipeline.encode_axpy(ctx, &mut enc, -1.0, gpu_tmp, 1.0, gpu_r);
        ctx.queue.submit(Some(enc.finish()));
    }
    // p = r
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        vec_pipeline.encode_axpy(ctx, &mut enc, 1.0, gpu_r, 0.0, gpu_p);
        ctx.queue.submit(Some(enc.finish()));
    }

    let mut rsold = vec_pipeline.dispatch_dot_readback(ctx, gpu_r, gpu_r, dot_buf);
    let tol = cfg.atol.max(cfg.rtol * b_norm);

    let total_iterations = fixed_iterations.unwrap_or(cfg.max_iter);
    for iter in 0..total_iterations {
        // ap = A * p
        {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            spmv_pipeline.encode_spmv(ctx, &mut enc, 1.0, gpu_a, gpu_p, 0.0, gpu_ap);
            ctx.queue.submit(Some(enc.finish()));
        }

        // pAp = p · ap
        let pap = vec_pipeline.dispatch_dot_readback(ctx, gpu_p, gpu_ap, dot_buf);

        let alpha = rsold / pap;

        // x = x + alpha * p
        {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            vec_pipeline.encode_axpy(ctx, &mut enc, alpha, gpu_p, 1.0, gpu_x);
            ctx.queue.submit(Some(enc.finish()));
        }

        // r = r - alpha * ap
        {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            vec_pipeline.encode_axpy(ctx, &mut enc, -alpha, gpu_ap, 1.0, gpu_r);
            ctx.queue.submit(Some(enc.finish()));
        }

        // rsnew = r·r
        let rsnew = vec_pipeline.dispatch_dot_readback(ctx, gpu_r, gpu_r, dot_buf);

        let r_norm = rsnew.sqrt();
        if fixed_iterations.is_none() && r_norm < tol {
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
            vec_pipeline.encode_axpy(ctx, &mut enc, 1.0, gpu_r, beta, gpu_p);
            ctx.queue.submit(Some(enc.finish()));
        }
    }

    // Did not converge
    let cpu_x = gpu_x.read_to_cpu(ctx);
    x.copy_from_slice(&cpu_x);
    let final_r = vec_pipeline.compute_norm2(ctx, gpu_r);
    Err(SolverError::ConvergenceFailed {
        max_iter: total_iterations,
        residual: final_r,
    })
}
