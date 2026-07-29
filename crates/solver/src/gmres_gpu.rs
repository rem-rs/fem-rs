// MFEM: GMRESSolver (GPU)
//! GPU-native GMRES solver with restart (modified Gram-Schmidt Arnoldi).
//!
//! All vectors live on the GPU; only dot products and the small Hessenberg
//! least-squares problem (O(m²)) are handled on CPU.
//!
//! Relies on [`super::gpu_base::GpuSolverBase`] for shared resources.

use fem_core::Scalar;
use fem_linalg::CsrMatrix;
use fem_linalg_gpu::{GpuContext, GpuVector};
use std::time::{Duration, Instant};
use wgpu;
use crate::{SolverConfig, SolveResult, SolverError};
use super::gpu_base::GpuSolverBase;

const DEFAULT_RESTART: usize = 30;

/// Helper: convert Scalar to f64.
#[inline(always)]
fn sc_to_f64<T: Scalar>(v: T) -> f64 { v.to_f64().unwrap() }

/// Timing breakdown for a fixed-iteration GMRES GPU run.
#[derive(Clone, Debug, Default)]
pub struct GmresGpuProfile {
    pub iterations: usize,
    pub residual_phase: Duration,
    pub basis_seed_phase: Duration,
    pub arnoldi_spmv_phase: Duration,
    pub arnoldi_orthogonalization_phase: Duration,
    pub arnoldi_normalization_phase: Duration,
    pub solution_update_phase: Duration,
    pub finalization_phase: Duration,
    pub total_phase: Duration,
    pub final_residual: f64,
}

/// Reusable GPU GMRES workspace.
pub struct GmresGpuWorkspace<T: Scalar> {
    base: GpuSolverBase<T>,
    restart: usize,
    gpu_w: GpuVector<T>,
    basis: Vec<GpuVector<T>>,
    // CPU-side Hessenberg arrays
    h: Vec<f64>,
    s: Vec<f64>,
    cs: Vec<f64>,
    sn: Vec<f64>,
    y: Vec<f64>,
}

impl<T: Scalar> GmresGpuWorkspace<T> {
    pub fn new(ctx: &GpuContext, a: &CsrMatrix<T>, b: &[T]) -> Self {
        let base = GpuSolverBase::new(ctx, a, b);
        let n = base.n;
        let m = DEFAULT_RESTART;
        let basis = (0..=m).map(|_| GpuVector::<T>::zeros(ctx, n)).collect();
        Self {
            base, restart: m,
            gpu_w: GpuVector::zeros(ctx, n),
            basis,
            h: vec![0.0; (m + 1) * m], s: vec![0.0; m + 1],
            cs: vec![0.0; m], sn: vec![0.0; m], y: vec![0.0; m],
        }
    }

    fn m(&self) -> usize { self.restart }

    pub fn solve(&mut self, ctx: &GpuContext, x: &mut [T], cfg: &SolverConfig) -> Result<SolveResult, SolverError> {
        self.base.gpu_x.write_from_slice(ctx, x);
        self.solve_gmres(ctx, x, cfg, false)
    }

    pub fn solve_fixed_iters(&mut self, ctx: &GpuContext, x: &mut [T], cfg: &SolverConfig) -> Result<SolveResult, SolverError> {
        self.base.gpu_x.write_from_slice(ctx, x);
        self.solve_gmres(ctx, x, cfg, true)
    }

    // ── Performance measurement helpers ─────────────────────────────────────

    pub fn measure_fixed_iters(&mut self, ctx: &GpuContext, x: &mut [T], cfg: &SolverConfig) -> GmresGpuProfile {
        self.measure(ctx, x, cfg, false, false)
    }

    pub fn measure_fixed_iters_arnoldi_only(&mut self, ctx: &GpuContext, x: &mut [T], cfg: &SolverConfig) -> GmresGpuProfile {
        self.measure(ctx, x, cfg, true, false)
    }

    pub fn measure_fixed_iters_spmv_only(&mut self, ctx: &GpuContext, x: &mut [T], cfg: &SolverConfig) -> GmresGpuProfile {
        self.measure(ctx, x, cfg, true, true)
    }

    pub fn profile_fixed_iters(&mut self, ctx: &GpuContext, x: &mut [T], cfg: &SolverConfig) -> GmresGpuProfile {
        self.profile(ctx, x, cfg)
    }

    // ── Core solver ─────────────────────────────────────────────────────────

    fn solve_gmres(&mut self, ctx: &GpuContext, _x: &mut [T], cfg: &SolverConfig, fixed_iters: bool) -> Result<SolveResult, SolverError> {
        let max_iter = cfg.max_iter;
        let tol = (cfg.rtol.max(1e-30) * self.base.b_norm).max(cfg.atol.max(1e-30));
        let m = self.m();

        // r = b - A*x
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_init") });
        self.base.compute_residual(ctx, &mut enc);
        ctx.queue.submit([enc.finish()]);

        let mut r_norm = self.base.norm2(ctx, &self.base.gpu_r);
        if r_norm <= tol && !fixed_iters {
            _x.copy_from_slice(&self.base.read_solution(ctx));
            return Ok(SolveResult { converged: true, iterations: 0, final_residual: r_norm });
        }

        self.seed_basis(ctx, r_norm);
        self.s[0] = r_norm;

        let mut h = vec![0.0_f64; (m + 1) * m];
        let mut s = vec![0.0_f64; m + 1];
        s[0] = r_norm;
        let mut cs = vec![0.0_f64; m];
        let mut sn = vec![0.0_f64; m];
        let mut y = vec![0.0_f64; m];
        let mut iter_count = 0u32;

        'outer: loop {
            for j in 0..m {
                // w = A·basis[j]
                let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_arnoldi") });
                self.base.spmv.encode_spmv(ctx, &mut enc, 1.0, &self.base.gpu_a, &self.basis[j], 0.0, &self.gpu_w);
                ctx.queue.submit([enc.finish()]);

                // MGS orthogonalization
                for i in 0..=j {
                    let hij = self.base.dot_readback(ctx, &self.gpu_w, &self.basis[i]);
                    h[i * m + j] = sc_to_f64(hij);
                    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_mgs") });
                    self.base.vops.encode_axpy(ctx, &mut enc, -h[i * m + j], &self.basis[i], 1.0, &self.gpu_w);
                    ctx.queue.submit([enc.finish()]);
                }

                let w_norm = self.base.norm2(ctx, &self.gpu_w);
                h[(j + 1) * m + j] = w_norm;

                // Givens rotation
                for i in 0..j {
                    let tmp = cs[i] * h[i * m + j] + sn[i] * h[(i + 1) * m + j];
                    h[(i + 1) * m + j] = -sn[i] * h[i * m + j] + cs[i] * h[(i + 1) * m + j];
                    h[i * m + j] = tmp;
                }
                let hjj = h[j * m + j];
                let hjp1j = h[(j + 1) * m + j];
                let sq = (hjj * hjj + hjp1j * hjp1j).sqrt();
                cs[j] = hjj / sq;
                sn[j] = hjp1j / sq;
                h[j * m + j] = sq;
                h[(j + 1) * m + j] = 0.0;
                s[j + 1] = -sn[j] * s[j];
                s[j] = cs[j] * s[j];

                if iter_count + 1 < max_iter as u32 {
                    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_normalize") });
                    self.base.vops.encode_axpy(ctx, &mut enc, 1.0 / w_norm, &self.gpu_w, 0.0, &self.basis[j + 1]);
                    ctx.queue.submit([enc.finish()]);
                }

                iter_count += 1;

                if !fixed_iters {
                    let r_norm_j = s[j + 1].abs();
                    if r_norm_j <= tol || iter_count >= max_iter as u32 {
                        let n_mini = j + 1;
                        for i in (0..n_mini).rev() {
                            let mut sum = s[i];
                            for k in (i + 1)..n_mini { sum -= h[i * m + k] * y[k]; }
                            y[i] = sum / h[i * m + i];
                        }
                        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_solution") });
                        for i in 0..n_mini {
                            self.base.vops.encode_axpy(ctx, &mut enc, y[i], &self.basis[i], 1.0, &self.base.gpu_x);
                        }
                        ctx.queue.submit([enc.finish()]);
                        _x.copy_from_slice(&self.base.read_solution(ctx));
                        return Ok(SolveResult { converged: r_norm_j <= tol, iterations: iter_count as usize, final_residual: r_norm_j });
                    }
                }
            }

            if iter_count >= max_iter as u32 { break 'outer; }

            // Restart
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_restart") });
            self.base.compute_residual(ctx, &mut enc);
            ctx.queue.submit([enc.finish()]);
            r_norm = self.base.norm2(ctx, &self.base.gpu_r);
            if r_norm <= tol && !fixed_iters {
                _x.copy_from_slice(&self.base.read_solution(ctx));
                return Ok(SolveResult { converged: true, iterations: iter_count as usize, final_residual: r_norm });
            }
            self.seed_basis(ctx, r_norm);
            s[0] = r_norm;
            for i in 1..=m { s[i] = 0.0; }
            h.iter_mut().for_each(|x| *x = 0.0);
        }

        _x.copy_from_slice(&self.base.read_solution(ctx));
        let final_res = if fixed_iters { self.base.norm2(ctx, &self.base.gpu_r) } else { r_norm };
        Ok(SolveResult { converged: final_res <= tol, iterations: iter_count as usize, final_residual: final_res })
    }

    fn seed_basis(&self, ctx: &GpuContext, r_norm: f64) {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_seed") });
        self.base.vops.encode_axpy(ctx, &mut enc, 1.0 / r_norm, &self.base.gpu_r, 0.0, &self.basis[0]);
        ctx.queue.submit([enc.finish()]);
    }

    // ── Profiling ───────────────────────────────────────────────────────────

    fn measure(&mut self, ctx: &GpuContext, x: &mut [T], cfg: &SolverConfig, skip_orth: bool, skip_update: bool) -> GmresGpuProfile {
        self.base.gpu_x.write_from_slice(ctx, x);
        let total_start = Instant::now();
        let mut prof = GmresGpuProfile::default();
        let target = cfg.max_iter;
        let m = self.m();

        let t0 = Instant::now();
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_measure_init") });
        self.base.compute_residual(ctx, &mut enc);
        ctx.queue.submit([enc.finish()]);
        prof.residual_phase = t0.elapsed();

        let t0 = Instant::now();
        let r_norm = self.base.norm2(ctx, &self.base.gpu_r);
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_measure_seed") });
        self.base.vops.encode_axpy(ctx, &mut enc, 1.0 / r_norm, &self.base.gpu_r, 0.0, &self.basis[0]);
        ctx.queue.submit([enc.finish()]);
        prof.basis_seed_phase = t0.elapsed();

        let mut iter_count = 0usize;
        while iter_count < target {
            for j in 0..m {
                let t_spmv = Instant::now();
                let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_measure_spmv") });
                self.base.spmv.encode_spmv(ctx, &mut enc, 1.0, &self.base.gpu_a, &self.basis[j], 0.0, &self.gpu_w);
                ctx.queue.submit([enc.finish()]);
                prof.arnoldi_spmv_phase += t_spmv.elapsed();

                if skip_orth {
                    let w_norm = self.base.norm2(ctx, &self.gpu_w);
                    if iter_count + 1 < target {
                        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_measure_skip") });
                        self.base.vops.encode_axpy(ctx, &mut enc, 1.0 / w_norm, &self.gpu_w, 0.0, &self.basis[j + 1]);
                        ctx.queue.submit([enc.finish()]);
                    }
                    iter_count += 1;
                    if iter_count >= target { break; }
                    continue;
                }

                let t_orth = Instant::now();
                for i in 0..=j {
                    let hij = self.base.dot_readback(ctx, &self.gpu_w, &self.basis[i]);
                    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_measure_mgs") });
                    self.base.vops.encode_axpy(ctx, &mut enc, -sc_to_f64(hij), &self.basis[i], 1.0, &self.gpu_w);
                    ctx.queue.submit([enc.finish()]);
                }
                prof.arnoldi_orthogonalization_phase += t_orth.elapsed();

                let t_norm = Instant::now();
                let _w_norm = self.base.norm2(ctx, &self.gpu_w);
                prof.arnoldi_normalization_phase += t_norm.elapsed();

                if iter_count + 1 < target {
                    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_measure_norm") });
                    self.base.vops.encode_axpy(ctx, &mut enc, 1.0 / _w_norm, &self.gpu_w, 0.0, &self.basis[j + 1]);
                    ctx.queue.submit([enc.finish()]);
                }

                iter_count += 1;
                if iter_count >= target { break; }
            }
            if iter_count >= target { break; }

            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_measure_restart") });
            self.base.compute_residual(ctx, &mut enc);
            ctx.queue.submit([enc.finish()]);
            let r_norm = self.base.norm2(ctx, &self.base.gpu_r);
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_measure_reseed") });
            self.base.vops.encode_axpy(ctx, &mut enc, 1.0 / r_norm, &self.base.gpu_r, 0.0, &self.basis[0]);
            ctx.queue.submit([enc.finish()]);
        }

        prof.iterations = iter_count;

        if !skip_update {
            let t_sol = Instant::now();
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_measure_solution") });
            for i in 0..m.min(iter_count) {
                self.base.vops.encode_axpy(ctx, &mut enc, self.y[i], &self.basis[i], 1.0, &self.base.gpu_x);
            }
            ctx.queue.submit([enc.finish()]);
            prof.solution_update_phase = t_sol.elapsed();
        }

        let t_final = Instant::now();
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_measure_final") });
        self.base.compute_residual(ctx, &mut enc);
        ctx.queue.submit([enc.finish()]);
        prof.final_residual = self.base.norm2(ctx, &self.base.gpu_r);
        prof.finalization_phase = t_final.elapsed();
        prof.total_phase = total_start.elapsed();
        prof
    }

    fn profile(&mut self, ctx: &GpuContext, x: &mut [T], cfg: &SolverConfig) -> GmresGpuProfile {
        self.base.gpu_x.write_from_slice(ctx, x);
        let total_start = Instant::now();
        let mut prof = GmresGpuProfile::default();
        let target = cfg.max_iter;
        let m = self.m();

        let t0 = Instant::now();
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_profile_init") });
        self.base.compute_residual(ctx, &mut enc);
        ctx.queue.submit([enc.finish()]);
        prof.residual_phase = t0.elapsed();

        let t0 = Instant::now();
        let r_norm = self.base.norm2(ctx, &self.base.gpu_r);
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_profile_seed") });
        self.base.vops.encode_axpy(ctx, &mut enc, 1.0 / r_norm, &self.base.gpu_r, 0.0, &self.basis[0]);
        ctx.queue.submit([enc.finish()]);
        prof.basis_seed_phase = t0.elapsed();

        let mut h = vec![0.0_f64; (m + 1) * m];
        let mut s = vec![0.0_f64; m + 1];
        s[0] = r_norm;
        let mut cs = vec![0.0_f64; m];
        let mut sn = vec![0.0_f64; m];

        let mut iter_count = 0usize;
        while iter_count < target {
            for j in 0..m {
                let t_spmv = Instant::now();
                let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_profile_spmv") });
                self.base.spmv.encode_spmv(ctx, &mut enc, 1.0, &self.base.gpu_a, &self.basis[j], 0.0, &self.gpu_w);
                ctx.queue.submit([enc.finish()]);
                prof.arnoldi_spmv_phase += t_spmv.elapsed();

                let t_orth = Instant::now();
                for i in 0..=j {
                    let hij = self.base.dot_readback(ctx, &self.gpu_w, &self.basis[i]);
                    h[i * m + j] = sc_to_f64(hij);
                    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_profile_mgs") });
                    self.base.vops.encode_axpy(ctx, &mut enc, -h[i * m + j], &self.basis[i], 1.0, &self.gpu_w);
                    ctx.queue.submit([enc.finish()]);
                }
                prof.arnoldi_orthogonalization_phase += t_orth.elapsed();

                let t_norm = Instant::now();
                let w_norm = self.base.norm2(ctx, &self.gpu_w);
                h[(j + 1) * m + j] = w_norm;
                prof.arnoldi_normalization_phase += t_norm.elapsed();

                for i in 0..j {
                    let tmp = cs[i] * h[i * m + j] + sn[i] * h[(i + 1) * m + j];
                    h[(i + 1) * m + j] = -sn[i] * h[i * m + j] + cs[i] * h[(i + 1) * m + j];
                    h[i * m + j] = tmp;
                }
                let hjj = h[j * m + j];
                let hjp1j = h[(j + 1) * m + j];
                let sq = (hjj * hjj + hjp1j * hjp1j).sqrt();
                cs[j] = hjj / sq; sn[j] = hjp1j / sq;
                h[j * m + j] = sq; h[(j + 1) * m + j] = 0.0;
                s[j + 1] = -sn[j] * s[j]; s[j] = cs[j] * s[j];

                if iter_count + 1 < target {
                    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_profile_norm") });
                    self.base.vops.encode_axpy(ctx, &mut enc, 1.0 / w_norm, &self.gpu_w, 0.0, &self.basis[j + 1]);
                    ctx.queue.submit([enc.finish()]);
                }
                iter_count += 1;
                if iter_count >= target { break; }
            }
            if iter_count >= target { break; }

            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_profile_restart") });
            self.base.compute_residual(ctx, &mut enc);
            ctx.queue.submit([enc.finish()]);
            let r_norm = self.base.norm2(ctx, &self.base.gpu_r);
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_profile_reseed") });
            self.base.vops.encode_axpy(ctx, &mut enc, 1.0 / r_norm, &self.base.gpu_r, 0.0, &self.basis[0]);
            ctx.queue.submit([enc.finish()]);
            s[0] = r_norm;
            for i in 1..=m { s[i] = 0.0; }
            h.iter_mut().for_each(|x| *x = 0.0);
        }

        prof.iterations = iter_count;
        let n_mini = if iter_count % m == 0 { m } else { iter_count % m };

        let t_sol = Instant::now();
        for i in (0..n_mini).rev() {
            let mut sum = s[i];
            for j in (i + 1)..n_mini { sum -= h[i * m + j] * self.y[j]; }
            self.y[i] = sum / h[i * m + i];
        }
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_profile_solution") });
        for i in 0..n_mini {
            self.base.vops.encode_axpy(ctx, &mut enc, self.y[i], &self.basis[i], 1.0, &self.base.gpu_x);
        }
        ctx.queue.submit([enc.finish()]);
        prof.solution_update_phase = t_sol.elapsed();

        let t_final = Instant::now();
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gmres_profile_final") });
        self.base.compute_residual(ctx, &mut enc);
        ctx.queue.submit([enc.finish()]);
        prof.final_residual = self.base.norm2(ctx, &self.base.gpu_r);
        prof.finalization_phase = t_final.elapsed();
        prof.total_phase = total_start.elapsed();
        prof
    }
}

pub fn solve_gmres_gpu(ctx: &GpuContext, a: &CsrMatrix<f64>, b: &[f64], x: &mut [f64], cfg: &SolverConfig) -> Result<SolveResult, SolverError> {
    let mut ws = GmresGpuWorkspace::<f64>::new(ctx, a, b);
    ws.solve(ctx, x, cfg)
}

pub fn solve_gmres_gpu_f32(ctx: &GpuContext, a: &CsrMatrix<f32>, b: &[f32], x: &mut [f32], cfg: &SolverConfig) -> Result<SolveResult, SolverError> {
    let mut ws = GmresGpuWorkspace::<f32>::new(ctx, a, b);
    ws.solve(ctx, x, cfg)
}
