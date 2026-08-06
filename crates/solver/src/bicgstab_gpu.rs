// MFEM: BiCGSTABSolver (GPU)
//! GPU-native BiCGSTAB solver.
//!
//! Suitable for non-symmetric systems. All vectors live on the GPU; only dot
//! products and scalar updates are handled on CPU via readback.
//!
//! Relies on [`super::gpu_base::GpuSolverBase`] for shared resources.

use super::gpu_base::GpuSolverBase;
use crate::{SolveResult, SolverConfig, SolverError};
use fem_core::Scalar;
use fem_linalg::CsrMatrix;
use fem_linalg_gpu::{GpuContext, GpuVector};
use wgpu;

/// Reusable GPU BiCGSTAB workspace.
///
/// Algorithm (Saad, 2003):
///   r₀ = b − Ax₀,  r̂₀ = r₀
///   ρ₀ = α = ω₀ = 1
///   p₀ = v₀ = 0
///   for i = 1,2,… until convergence:
///     ρᵢ = (r̂₀, rᵢ₋₁)
///     β = (ρᵢ/ρᵢ₋₁)(α/ωᵢ₋₁)
///     pᵢ = rᵢ₋₁ + β(pᵢ₋₁ − ωᵢ₋₁·vᵢ₋₁)
///     vᵢ = A·pᵢ
///     α = ρᵢ / (r̂₀, vᵢ)
///     s  = rᵢ₋₁ − α·vᵢ
///     t  = A·s
///     ω = (t, s) / (t, t)
///     xᵢ = xᵢ₋₁ + α·pᵢ + ω·s
///     rᵢ = s − ω·t
pub struct BicgstabGpuWorkspace<T: Scalar> {
    base: GpuSolverBase<T>,
    gpu_r0: GpuVector<T>, // fixed shadow residual r̂₀
    gpu_p: GpuVector<T>,  // direction p
    gpu_v: GpuVector<T>,  // v = A·p
    gpu_s: GpuVector<T>,  // s = r − α·v
    gpu_t: GpuVector<T>,  // t = A·s
}

impl<T: Scalar> BicgstabGpuWorkspace<T> {
    pub fn new(ctx: &GpuContext, a: &CsrMatrix<T>, b: &[T]) -> Self {
        let base = GpuSolverBase::new(ctx, a, b);
        let n = base.n;
        Self {
            base,
            gpu_r0: GpuVector::zeros(ctx, n),
            gpu_p: GpuVector::zeros(ctx, n),
            gpu_v: GpuVector::zeros(ctx, n),
            gpu_s: GpuVector::zeros(ctx, n),
            gpu_t: GpuVector::zeros(ctx, n),
        }
    }

    pub fn solve(
        &mut self,
        ctx: &GpuContext,
        x: &mut [T],
        cfg: &SolverConfig,
    ) -> Result<SolveResult, SolverError> {
        self.base.gpu_x.write_from_slice(ctx, x);
        self.solve_bicgstab(ctx, x, cfg, false)
    }

    pub fn solve_fixed_iters(
        &mut self,
        ctx: &GpuContext,
        x: &mut [T],
        cfg: &SolverConfig,
    ) -> Result<SolveResult, SolverError> {
        self.base.gpu_x.write_from_slice(ctx, x);
        self.solve_bicgstab(ctx, x, cfg, true)
    }

    fn solve_bicgstab(
        &mut self,
        ctx: &GpuContext,
        _x: &mut [T],
        cfg: &SolverConfig,
        fixed_iters: bool,
    ) -> Result<SolveResult, SolverError> {
        let max_iter = cfg.max_iter;
        let tol = (cfg.rtol.max(1e-30) * self.base.b_norm).max(cfg.atol.max(1e-30));

        // r₀ = b − A·x₀
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bicgstab_init"),
            });
        self.base.compute_residual(ctx, &mut enc);
        ctx.queue.submit([enc.finish()]);

        let mut r_norm = self.base.norm2(ctx, &self.base.gpu_r);
        if r_norm <= tol && !fixed_iters {
            _x.copy_from_slice(&self.base.read_solution(ctx));
            return Ok(SolveResult {
                converged: true,
                iterations: 0,
                final_residual: r_norm,
            });
        }

        // r̂₀ = r₀  (fixed shadow residual, stays constant throughout)
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bicgstab_r0"),
            });
        self.base
            .vops
            .encode_axpy(ctx, &mut enc, 1.0, &self.base.gpu_r, 0.0, &self.gpu_r0);
        ctx.queue.submit([enc.finish()]);

        let mut rho: f64 = 1.0;
        let mut alpha: f64 = 1.0;
        let mut omega: f64 = 1.0;
        let mut iters = 0u32;

        for _ in 0..max_iter {
            let rho_prev = rho;
            rho = self.base.dot_readback(ctx, &self.gpu_r0, &self.base.gpu_r);
            if rho.abs() < 1e-30 {
                break;
            }

            let beta = (rho / rho_prev) * (alpha / omega);

            // p = r + β·(p − ω·v)
            // Step 1: p = p − ω·v  (p_old becomes intermediate)
            // Step 2: p = β·p        (scale the intermediate)
            // Step 3: p = p + r      (add r)
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bicgstab_p"),
                });
            self.base
                .vops
                .encode_axpy(ctx, &mut enc, -omega, &self.gpu_v, 1.0, &self.gpu_p);
            self.base
                .vops
                .encode_axpy(ctx, &mut enc, beta, &self.gpu_p, 0.0, &self.gpu_p);
            self.base
                .vops
                .encode_axpy(ctx, &mut enc, 1.0, &self.base.gpu_r, 1.0, &self.gpu_p);
            ctx.queue.submit([enc.finish()]);

            // v = A·p
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bicgstab_spmv"),
                });
            self.base.spmv.encode_spmv(
                ctx,
                &mut enc,
                1.0,
                &self.base.gpu_a,
                &self.gpu_p,
                0.0,
                &self.gpu_v,
            );
            ctx.queue.submit([enc.finish()]);

            let r0_dot_v = self.base.dot_readback(ctx, &self.gpu_r0, &self.gpu_v);
            if r0_dot_v.abs() < 1e-30 {
                break;
            }
            alpha = rho / r0_dot_v;

            // s = r − α·v   (store in gpu_s)
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bicgstab_s"),
                });
            self.base
                .vops
                .encode_axpy(ctx, &mut enc, -alpha, &self.gpu_v, 1.0, &self.base.gpu_r);
            self.base
                .vops
                .encode_axpy(ctx, &mut enc, 1.0, &self.base.gpu_r, 0.0, &self.gpu_s);
            ctx.queue.submit([enc.finish()]);

            // t = A·s
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bicgstab_t"),
                });
            self.base.spmv.encode_spmv(
                ctx,
                &mut enc,
                1.0,
                &self.base.gpu_a,
                &self.gpu_s,
                0.0,
                &self.gpu_t,
            );
            ctx.queue.submit([enc.finish()]);

            let t_dot_t = self.base.dot_readback(ctx, &self.gpu_t, &self.gpu_t);
            let t_dot_s = self.base.dot_readback(ctx, &self.gpu_t, &self.gpu_s);
            omega = if t_dot_t.abs() > 1e-30 {
                t_dot_s / t_dot_t
            } else {
                0.0
            };

            // x = x + α·p + ω·s
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bicgstab_x"),
                });
            self.base
                .vops
                .encode_axpy(ctx, &mut enc, alpha, &self.gpu_p, 1.0, &self.base.gpu_x);
            self.base
                .vops
                .encode_axpy(ctx, &mut enc, omega, &self.gpu_s, 1.0, &self.base.gpu_x);
            ctx.queue.submit([enc.finish()]);

            // r = s − ω·t
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bicgstab_r"),
                });
            self.base
                .vops
                .encode_axpy(ctx, &mut enc, -omega, &self.gpu_t, 1.0, &self.gpu_s);
            self.base
                .vops
                .encode_axpy(ctx, &mut enc, 1.0, &self.gpu_s, 0.0, &self.base.gpu_r);
            ctx.queue.submit([enc.finish()]);

            iters += 1;
            if !fixed_iters {
                r_norm = self.base.norm2(ctx, &self.base.gpu_r);
                if r_norm <= tol {
                    break;
                }
            }
        }

        _x.copy_from_slice(&self.base.read_solution(ctx));
        Ok(SolveResult {
            converged: r_norm <= tol,
            iterations: iters as usize,
            final_residual: r_norm,
        })
    }
}

pub fn solve_bicgstab_gpu(
    ctx: &GpuContext,
    a: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let mut ws = BicgstabGpuWorkspace::<f64>::new(ctx, a, b);
    ws.solve(ctx, x, cfg)
}

pub fn solve_bicgstab_gpu_f32(
    ctx: &GpuContext,
    a: &CsrMatrix<f32>,
    b: &[f32],
    x: &mut [f32],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let mut ws = BicgstabGpuWorkspace::<f32>::new(ctx, a, b);
    ws.solve(ctx, x, cfg)
}
