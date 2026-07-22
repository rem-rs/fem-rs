//! GPU-native Conjugate Gradient (CG) solver.
//!
//! All vectors live on the GPU; only the residual norm is read back each
//! iteration for convergence checking.
//!
//! Relies on [`super::gpu_base::GpuSolverBase`] for shared resources.

use fem_core::Scalar;
use fem_linalg::CsrMatrix;
use fem_linalg_gpu::{GpuContext, GpuVector, GpuJacobiPrecond, DeviceBuffer};
use wgpu;
use crate::{SolverConfig, SolveResult, SolverError};
use super::gpu_base::GpuSolverBase;

// ═══════════════════════════════════════════════════════════════════════════════
// CG
// ═══════════════════════════════════════════════════════════════════════════════

pub struct CgGpuWorkspace<T: Scalar> {
    base: GpuSolverBase<T>,
    gpu_p: GpuVector<T>,
    gpu_ap: GpuVector<T>,
    gpu_tmp: GpuVector<T>,
}

impl<T: Scalar> CgGpuWorkspace<T> {
    pub fn new(ctx: &GpuContext, a: &CsrMatrix<T>, b: &[T]) -> Self {
        let base = GpuSolverBase::new(ctx, a, b);
        let n = base.n;
        Self { base, gpu_p: GpuVector::zeros(ctx, n), gpu_ap: GpuVector::zeros(ctx, n), gpu_tmp: GpuVector::zeros(ctx, n) }
    }

    pub fn solve(&mut self, ctx: &GpuContext, x: &mut [T], cfg: &SolverConfig) -> Result<SolveResult, SolverError> {
        self.base.gpu_x.write_from_slice(ctx, x);
        let (res, sol) = self.solve_cg_prepared(ctx, cfg, false);
        x.copy_from_slice(&sol);
        res
    }

    pub fn solve_fixed_iters(&mut self, ctx: &GpuContext, x: &mut [T], cfg: &SolverConfig) -> Result<SolveResult, SolverError> {
        self.base.gpu_x.write_from_slice(ctx, x);
        let (res, sol) = self.solve_cg_prepared(ctx, cfg, true);
        x.copy_from_slice(&sol);
        res
    }

    fn solve_cg_prepared(&mut self, ctx: &GpuContext, cfg: &SolverConfig, fixed_iters: bool) -> (Result<SolveResult, SolverError>, Vec<T>) {
        let max_iter = cfg.max_iter;
        let tol = (cfg.rtol.max(1e-30) * self.base.b_norm).max(cfg.atol.max(1e-30));

        // r = b - A*x
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cg_init") });
        self.base.compute_residual(ctx, &mut enc);
        ctx.queue.submit([enc.finish()]);

        let mut r_norm = self.base.norm2(ctx, &self.base.gpu_r);
        if r_norm <= tol && !fixed_iters {
            return (Ok(SolveResult { converged: true, iterations: 0, final_residual: r_norm }), self.base.read_solution(ctx));
        }

        // p = r
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cg_init_p") });
        self.base.vops.encode_axpy(ctx, &mut enc, 1.0, &self.base.gpu_r, 0.0, &self.gpu_p);
        ctx.queue.submit([enc.finish()]);

        let mut rsold = self.base.dot_readback(ctx, &self.base.gpu_r, &self.base.gpu_r);
        let mut iters = 0u32;

        for _ in 0..max_iter {
            // tmp = A·p
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cg_spmv") });
            self.base.spmv.encode_spmv(ctx, &mut enc, 1.0, &self.base.gpu_a, &self.gpu_p, 0.0, &self.gpu_tmp);
            ctx.queue.submit([enc.finish()]);

            let pap = self.base.dot_readback(ctx, &self.gpu_p, &self.gpu_tmp);
            let alpha = rsold / pap;

            // x = x + α·p,  r = r - α·A·p
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cg_update") });
            self.base.vops.encode_axpy(ctx, &mut enc, alpha, &self.gpu_p, 1.0, &self.base.gpu_x);
            self.base.vops.encode_axpy(ctx, &mut enc, -alpha, &self.gpu_tmp, 1.0, &self.base.gpu_r);
            ctx.queue.submit([enc.finish()]);

            iters += 1;
            if !fixed_iters {
                r_norm = self.base.norm2(ctx, &self.base.gpu_r);
                if r_norm <= tol { break; }
            }

            let rsnew = self.base.dot_readback(ctx, &self.base.gpu_r, &self.base.gpu_r);
            let beta = rsnew / rsold;
            rsold = rsnew;

            // p = r + β·p
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cg_update_p") });
            self.base.vops.encode_axpy(ctx, &mut enc, -beta, &self.gpu_p, 0.0, &self.gpu_p);
            self.base.vops.encode_axpy(ctx, &mut enc, 1.0, &self.base.gpu_r, 1.0, &self.gpu_p);
            ctx.queue.submit([enc.finish()]);
        }

        (Ok(SolveResult { converged: r_norm <= tol, iterations: iters as usize, final_residual: r_norm }), self.base.read_solution(ctx))
    }
}

pub fn solve_cg_gpu(ctx: &GpuContext, a: &CsrMatrix<f64>, b: &[f64], x: &mut [f64], cfg: &SolverConfig) -> Result<SolveResult, SolverError> {
    let mut ws = CgGpuWorkspace::<f64>::new(ctx, a, b);
    ws.solve(ctx, x, cfg)
}

pub fn solve_cg_gpu_f32(ctx: &GpuContext, a: &CsrMatrix<f32>, b: &[f32], x: &mut [f32], cfg: &SolverConfig) -> Result<SolveResult, SolverError> {
    let mut ws = CgGpuWorkspace::<f32>::new(ctx, a, b);
    ws.solve(ctx, x, cfg)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Preconditioned CG (Jacobi)
// ═══════════════════════════════════════════════════════════════════════════════

pub struct PcgGpuWorkspace<T: Scalar> {
    base: GpuSolverBase<T>,
    gpu_z: GpuVector<T>,
    gpu_p: GpuVector<T>,
    gpu_ap: GpuVector<T>,
    precond: GpuJacobiPrecond<T>,
    inner_dot_buf: DeviceBuffer,
}

impl<T: Scalar> PcgGpuWorkspace<T> {
    pub fn new(ctx: &GpuContext, a: &CsrMatrix<T>, b: &[T]) -> Self {
        let base = GpuSolverBase::new(ctx, a, b);
        let n = base.n;
        let precond = GpuJacobiPrecond::<T>::from_matrix(ctx, a);
        let n_wg = n.div_ceil(256);
        Self {
            base,
            gpu_z: GpuVector::zeros(ctx, n), gpu_p: GpuVector::zeros(ctx, n), gpu_ap: GpuVector::zeros(ctx, n),
            precond,
            inner_dot_buf: DeviceBuffer::with_staging(
                &ctx.device, n_wg as u64 * std::mem::size_of::<T>() as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, "pcg_dot_buf"),
        }
    }

    pub fn solve(&mut self, ctx: &GpuContext, x: &mut [T], cfg: &SolverConfig) -> Result<SolveResult, SolverError> {
        self.base.gpu_x.write_from_slice(ctx, x);
        let (res, sol) = self.solve_pcg_prepared(ctx, cfg, false);
        x.copy_from_slice(&sol);
        res
    }

    pub fn solve_fixed_iters(&mut self, ctx: &GpuContext, x: &mut [T], cfg: &SolverConfig) -> Result<SolveResult, SolverError> {
        self.base.gpu_x.write_from_slice(ctx, x);
        let (res, sol) = self.solve_pcg_prepared(ctx, cfg, true);
        x.copy_from_slice(&sol);
        res
    }

    fn solve_pcg_prepared(&mut self, ctx: &GpuContext, cfg: &SolverConfig, fixed_iters: bool) -> (Result<SolveResult, SolverError>, Vec<T>) {
        let max_iter = cfg.max_iter;
        let tol = (cfg.rtol.max(1e-30) * self.base.b_norm).max(cfg.atol.max(1e-30));

        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pcg_init") });
        self.base.compute_residual(ctx, &mut enc);
        ctx.queue.submit([enc.finish()]);

        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pcg_apply_z") });
        self.precond.encode_apply(ctx, &mut enc, &self.base.gpu_r, &mut self.gpu_z);
        ctx.queue.submit([enc.finish()]);

        let mut rho = self.base.dot_readback(ctx, &self.base.gpu_r, &self.gpu_z);
        let mut r_norm = self.base.norm2(ctx, &self.base.gpu_r);

        if r_norm <= tol && !fixed_iters {
            return (Ok(SolveResult { converged: true, iterations: 0, final_residual: r_norm }), self.base.read_solution(ctx));
        }

        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pcg_init_p") });
        self.base.vops.encode_axpy(ctx, &mut enc, 1.0, &self.gpu_z, 0.0, &self.gpu_p);
        ctx.queue.submit([enc.finish()]);

        let mut iters = 0u32;
        for _ in 0..max_iter {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pcg_spmv") });
            self.base.spmv.encode_spmv(ctx, &mut enc, 1.0, &self.base.gpu_a, &self.gpu_p, 0.0, &self.gpu_ap);
            ctx.queue.submit([enc.finish()]);

            let pap = self.base.dot_readback(ctx, &self.gpu_p, &self.gpu_ap);
            let alpha = rho / pap;

            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pcg_update") });
            self.base.vops.encode_axpy(ctx, &mut enc, alpha, &self.gpu_p, 1.0, &self.base.gpu_x);
            self.base.vops.encode_axpy(ctx, &mut enc, -alpha, &self.gpu_ap, 1.0, &self.base.gpu_r);
            ctx.queue.submit([enc.finish()]);

            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pcg_apply_z2") });
            self.precond.encode_apply(ctx, &mut enc, &self.base.gpu_r, &mut self.gpu_z);
            ctx.queue.submit([enc.finish()]);

            iters += 1;
            if !fixed_iters {
                r_norm = self.base.norm2(ctx, &self.base.gpu_r);
                if r_norm <= tol { break; }
            }

            let rho_new = self.base.dot_readback(ctx, &self.base.gpu_r, &self.gpu_z);
            let beta = rho_new / rho;
            rho = rho_new;

            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pcg_update_p") });
            self.base.vops.encode_axpy(ctx, &mut enc, -beta, &self.gpu_p, 0.0, &self.gpu_p);
            self.base.vops.encode_axpy(ctx, &mut enc, 1.0, &self.gpu_z, 1.0, &self.gpu_p);
            ctx.queue.submit([enc.finish()]);
        }

        (Ok(SolveResult { converged: r_norm <= tol, iterations: iters as usize, final_residual: r_norm }), self.base.read_solution(ctx))
    }
}

pub fn solve_pcg_gpu(ctx: &GpuContext, a: &CsrMatrix<f64>, b: &[f64], x: &mut [f64], cfg: &SolverConfig) -> Result<SolveResult, SolverError> {
    let mut ws = PcgGpuWorkspace::<f64>::new(ctx, a, b);
    ws.solve(ctx, x, cfg)
}

pub fn solve_pcg_gpu_f32(ctx: &GpuContext, a: &CsrMatrix<f32>, b: &[f32], x: &mut [f32], cfg: &SolverConfig) -> Result<SolveResult, SolverError> {
    let mut ws = PcgGpuWorkspace::<f32>::new(ctx, a, b);
    ws.solve(ctx, x, cfg)
}
