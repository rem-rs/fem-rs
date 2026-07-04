//! GPU-resident restarted GMRES solver.
//!
//! Uses modified Gram-Schmidt for Arnoldi with m-step restart.
//! The Hessenberg least-squares is solved on CPU (tiny, O(m²)).

use fem_core::Scalar;
use fem_linalg::CsrMatrix;
use fem_linalg_gpu::{
    DeviceBuffer, GpuContext, GpuCsrMatrix, GpuVector, SpmvPipeline, VectorOpsPipeline,
};
use std::time::{Duration, Instant};
use wgpu;
use crate::{SolverConfig, SolveResult, SolverError};

/// Default GMRES restart dimension.
const DEFAULT_RESTART: usize = 30;

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

/// Reusable GPU GMRES workspace that keeps uploaded data and temporary buffers on device.
pub struct GmresGpuWorkspace<T: Scalar> {
    n: u32,
    restart: usize,
    spmv: SpmvPipeline,
    vops: VectorOpsPipeline,
    gpu_a: GpuCsrMatrix<T>,
    gpu_b: GpuVector<T>,
    gpu_x: GpuVector<T>,
    gpu_ax: GpuVector<T>,
    gpu_r: GpuVector<T>,
    gpu_w: GpuVector<T>,
    basis: Vec<GpuVector<T>>,
    dot_buf: DeviceBuffer,
    b_norm: f64,
    h: Vec<f64>,
    s: Vec<f64>,
    cs: Vec<f64>,
    sn: Vec<f64>,
    y: Vec<f64>,
}

impl<T: Scalar> GmresGpuWorkspace<T> {
    /// Build a reusable GPU GMRES workspace for a fixed matrix and right-hand side.
    pub fn new(ctx: &GpuContext, a: &CsrMatrix<T>, b: &[T]) -> Self {
        let n = a.nrows as u32;
        assert_eq!(a.ncols as u32, n, "matrix must be square");
        assert_eq!(b.len() as u32, n);

        let restart = DEFAULT_RESTART.min(n as usize);
        let spmv = SpmvPipeline::new(&ctx.device, ctx.features.native_f64);
        let vops = VectorOpsPipeline::new(&ctx.device, ctx.features.native_f64);
        let gpu_a = GpuCsrMatrix::<T>::from_cpu(ctx, a);
        let gpu_b = GpuVector::from_slice(ctx, b);
        let gpu_x = GpuVector::<T>::zeros(ctx, n);
        let gpu_ax = GpuVector::<T>::zeros(ctx, n);
        let gpu_r = GpuVector::<T>::zeros(ctx, n);
        let gpu_w = GpuVector::<T>::zeros(ctx, n);
        let mut basis = Vec::with_capacity(restart + 1);
        for _ in 0..=restart {
            basis.push(GpuVector::<T>::zeros(ctx, n));
        }

        let n_wg = n.div_ceil(256);
        let dot_buf = DeviceBuffer::with_staging(
            &ctx.device,
            n_wg as u64 * std::mem::size_of::<T>() as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "gmres_dot",
        );
        let b_norm = vops.compute_norm2(ctx, &gpu_b);
        let h = vec![0.0f64; (restart + 1) * restart];
        let s = vec![0.0f64; restart + 1];
        let cs = vec![0.0f64; restart];
        let sn = vec![0.0f64; restart];
        let y = vec![0.0f64; restart];

        Self {
            n,
            restart,
            spmv,
            vops,
            gpu_a,
            gpu_b,
            gpu_x,
            gpu_ax,
            gpu_r,
            gpu_w,
            basis,
            dot_buf,
            b_norm,
            h,
            s,
            cs,
            sn,
            y,
        }
    }

    /// Solve using the pre-uploaded matrix/rhs and reusable temporaries.
    pub fn solve(&mut self, ctx: &GpuContext, x: &mut [T], cfg: &SolverConfig) -> Result<SolveResult, SolverError> {
        assert_eq!(x.len() as u32, self.n);
        self.gpu_x.write_from_slice(ctx, x);
        solve_gmres_gpu_prepared(
            ctx,
            self.restart,
            &self.spmv,
            &self.vops,
            &self.gpu_a,
            &self.gpu_b,
            &self.gpu_x,
            &mut self.gpu_ax,
            &mut self.gpu_r,
            &mut self.gpu_w,
            &mut self.basis,
            &self.dot_buf,
            self.b_norm,
            &mut self.h,
            &mut self.s,
            &mut self.cs,
            &mut self.sn,
            &mut self.y,
            None,
            true,
            true,
            true,
            x,
            cfg,
            None,
        )
    }

    /// Run a fixed number of GMRES iterations without early convergence exit.
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
        match solve_gmres_gpu_prepared(
            ctx,
            self.restart,
            &self.spmv,
            &self.vops,
            &self.gpu_a,
            &self.gpu_b,
            &self.gpu_x,
            &mut self.gpu_ax,
            &mut self.gpu_r,
            &mut self.gpu_w,
            &mut self.basis,
            &self.dot_buf,
            self.b_norm,
            &mut self.h,
            &mut self.s,
            &mut self.cs,
            &mut self.sn,
            &mut self.y,
            Some(iterations),
            true,
            true,
            true,
            x,
            &cfg,
            None,
        ) {
            Ok(result) => result.final_residual,
            Err(SolverError::ConvergenceFailed { residual, .. }) => residual,
            Err(err) => panic!("fixed-iteration GMRES run failed unexpectedly: {err}"),
        }
    }

    /// Run a fixed number of GMRES iterations for benchmarking without reading
    /// the final solution back to the CPU.
    pub fn measure_fixed_iters(&mut self, ctx: &GpuContext, initial_x: &[T], iterations: usize) -> f64 {
        self.measure_fixed_iters_internal(ctx, initial_x, iterations, true, true)
    }

    /// Run a fixed number of GMRES iterations for benchmarking while skipping
    /// the final `x += V y` update. This isolates Arnoldi/readback cost.
    pub fn measure_fixed_iters_arnoldi_only(&mut self, ctx: &GpuContext, initial_x: &[T], iterations: usize) -> f64 {
        self.measure_fixed_iters_internal(ctx, initial_x, iterations, false, true)
    }

    /// Run a fixed number of GMRES iterations for benchmarking while keeping the
    /// SpMV and basis-normalization work but skipping orthogonalization.
    pub fn measure_fixed_iters_spmv_only(&mut self, ctx: &GpuContext, initial_x: &[T], iterations: usize) -> f64 {
        self.measure_fixed_iters_internal(ctx, initial_x, iterations, false, false)
    }

    /// Run a fixed number of GMRES iterations and return a timing breakdown for
    /// the real GPU solver path.
    pub fn profile_fixed_iters(&mut self, ctx: &GpuContext, initial_x: &[T], iterations: usize) -> GmresGpuProfile {
        assert_eq!(initial_x.len() as u32, self.n);
        self.gpu_x.write_from_slice(ctx, initial_x);
        let cfg = SolverConfig {
            rtol: 0.0,
            atol: 0.0,
            max_iter: iterations,
            verbose: false,
            print_level: crate::PrintLevel::Silent,
        };
        let mut profile = GmresGpuProfile::default();
        let residual = match solve_gmres_gpu_prepared(
            ctx,
            self.restart,
            &self.spmv,
            &self.vops,
            &self.gpu_a,
            &self.gpu_b,
            &self.gpu_x,
            &mut self.gpu_ax,
            &mut self.gpu_r,
            &mut self.gpu_w,
            &mut self.basis,
            &self.dot_buf,
            self.b_norm,
            &mut self.h,
            &mut self.s,
            &mut self.cs,
            &mut self.sn,
            &mut self.y,
            Some(iterations),
            false,
            true,
            true,
            &mut [],
            &cfg,
            Some(&mut profile),
        ) {
            Ok(result) => result.final_residual,
            Err(SolverError::ConvergenceFailed { residual, .. }) => residual,
            Err(err) => panic!("fixed-iteration GMRES profiling failed unexpectedly: {err}"),
        };
        profile.final_residual = residual;
        profile
    }

    fn measure_fixed_iters_internal(
        &mut self,
        ctx: &GpuContext,
        initial_x: &[T],
        iterations: usize,
        apply_solution_update: bool,
        perform_orthogonalization: bool,
    ) -> f64 {
        assert_eq!(initial_x.len() as u32, self.n);
        self.gpu_x.write_from_slice(ctx, initial_x);
        let cfg = SolverConfig {
            rtol: 0.0,
            atol: 0.0,
            max_iter: iterations,
            verbose: false,
            print_level: crate::PrintLevel::Silent,
        };
        match solve_gmres_gpu_prepared(
            ctx,
            self.restart,
            &self.spmv,
            &self.vops,
            &self.gpu_a,
            &self.gpu_b,
            &self.gpu_x,
            &mut self.gpu_ax,
            &mut self.gpu_r,
            &mut self.gpu_w,
            &mut self.basis,
            &self.dot_buf,
            self.b_norm,
            &mut self.h,
            &mut self.s,
            &mut self.cs,
            &mut self.sn,
            &mut self.y,
            Some(iterations),
            false,
            apply_solution_update,
            perform_orthogonalization,
            &mut [],
            &cfg,
            None,
        ) {
            Ok(result) => result.final_residual,
            Err(SolverError::ConvergenceFailed { residual, .. }) => residual,
            Err(err) => panic!("fixed-iteration GMRES measurement failed unexpectedly: {err}"),
        }
    }
}

/// Solve `A x = b` using restarted GMRES on the GPU.
pub fn solve_gmres_gpu(
    ctx: &GpuContext,
    a: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    solve_gmres_gpu_impl(ctx, a, b, x, cfg)
}

/// Solve `A x = b` using restarted GMRES on the GPU with `f32` buffers.
pub fn solve_gmres_gpu_f32(
    ctx: &GpuContext,
    a: &CsrMatrix<f32>,
    b: &[f32],
    x: &mut [f32],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    solve_gmres_gpu_impl(ctx, a, b, x, cfg)
}

fn solve_gmres_gpu_impl<T: Scalar>(
    ctx: &GpuContext,
    a: &CsrMatrix<T>,
    b: &[T],
    x: &mut [T],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let mut workspace = GmresGpuWorkspace::new(ctx, a, b);
    workspace.solve(ctx, x, cfg)
}

#[allow(clippy::too_many_arguments)]
fn solve_gmres_gpu_prepared<T: Scalar>(
    ctx: &GpuContext,
    restart: usize,
    spmv: &SpmvPipeline,
    vops: &VectorOpsPipeline,
    a: &GpuCsrMatrix<T>,
    b: &GpuVector<T>,
    gpu_x: &GpuVector<T>,
    gpu_ax: &mut GpuVector<T>,
    gpu_r: &mut GpuVector<T>,
    gpu_w: &mut GpuVector<T>,
    basis: &mut [GpuVector<T>],
    dot_buf: &DeviceBuffer,
    b_norm: f64,
    h: &mut [f64],
    s: &mut [f64],
    cs: &mut [f64],
    sn: &mut [f64],
    y: &mut [f64],
    fixed_iterations: Option<usize>,
    read_back_solution: bool,
    apply_solution_update: bool,
    perform_orthogonalization: bool,
    x: &mut [T],
    cfg: &SolverConfig,
    mut profile: Option<&mut GmresGpuProfile>,
) -> Result<SolveResult, SolverError> {
    let total_start = Instant::now();
    let tol = cfg.atol.max(cfg.rtol * b_norm);
    let mut iter_count = 0usize;
    let target_iterations = fixed_iterations.unwrap_or(cfg.max_iter);
    let mut last_residual = f64::INFINITY;

    while iter_count < target_iterations {
        let local_restart = restart.min(target_iterations - iter_count);
        let residual_start = Instant::now();
        compute_residual_into(ctx, spmv, vops, a, b, gpu_x, gpu_ax, gpu_r);
        let r_norm = vops.compute_norm2(ctx, gpu_r);
        if let Some(ref mut profile) = profile {
            profile.residual_phase += residual_start.elapsed();
        }
        last_residual = r_norm;

        if fixed_iterations.is_none() && r_norm < tol {
            if read_back_solution {
                let cpu_x = gpu_x.read_to_cpu(ctx);
                x.copy_from_slice(&cpu_x);
            }
            if let Some(ref mut profile) = profile {
                profile.iterations = iter_count;
                profile.final_residual = r_norm;
                profile.total_phase = total_start.elapsed();
            }
            return Ok(SolveResult { converged: true, iterations: iter_count, final_residual: r_norm });
        }

        h.fill(0.0);
        s.fill(0.0);
        cs.fill(0.0);
        sn.fill(0.0);
        y.fill(0.0);

        let basis_seed_start = Instant::now();
        {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            vops.encode_axpy(ctx, &mut enc, 1.0 / r_norm, gpu_r, 0.0, &basis[0]);
            ctx.queue.submit(Some(enc.finish()));
        }
        if let Some(ref mut profile) = profile {
            profile.basis_seed_phase += basis_seed_start.elapsed();
        }

        s[0] = r_norm;
        let mut gmres_r_norm = r_norm;
        let mut j = 0usize;

        for jj in 0..local_restart {
            j = jj;
            let spmv_start = Instant::now();
            {
                let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                spmv.encode_spmv(ctx, &mut enc, 1.0, a, &basis[jj], 0.0, gpu_w);
                ctx.queue.submit(Some(enc.finish()));
            }
            if let Some(ref mut profile) = profile {
                profile.arnoldi_spmv_phase += spmv_start.elapsed();
            }

            if perform_orthogonalization {
                let orth_start = Instant::now();
                for i in 0..=jj {
                    let dot_val = vops.dispatch_dot_readback(ctx, gpu_w, &basis[i], dot_buf);
                    h[i * restart + jj] = dot_val;
                    {
                        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                        vops.encode_axpy(ctx, &mut enc, -dot_val, &basis[i], 1.0, gpu_w);
                        ctx.queue.submit(Some(enc.finish()));
                    }
                }
                if let Some(ref mut profile) = profile {
                    profile.arnoldi_orthogonalization_phase += orth_start.elapsed();
                }
            }

            let normalize_start = Instant::now();
            let w_norm = vops.compute_norm2(ctx, gpu_w);
            h[(jj + 1) * restart + jj] = w_norm;

            for i in 0..jj {
                let hi = h[i * restart + jj];
                let hi1 = h[(i + 1) * restart + jj];
                h[i * restart + jj] = cs[i] * hi + sn[i] * hi1;
                h[(i + 1) * restart + jj] = -sn[i] * hi + cs[i] * hi1;
            }

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

            let sj = s[jj];
            let sj1 = s[jj + 1];
            s[jj] = cs[jj] * sj + sn[jj] * sj1;
            s[jj + 1] = -sn[jj] * sj + cs[jj] * sj1;

            gmres_r_norm = s[jj + 1].abs();
            last_residual = gmres_r_norm;
            iter_count += 1;

            if fixed_iterations.is_none() && gmres_r_norm < tol {
                break;
            }

            if w_norm > 1e-15 {
                let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                vops.encode_axpy(ctx, &mut enc, 1.0 / w_norm, gpu_w, 0.0, &basis[jj + 1]);
                ctx.queue.submit(Some(enc.finish()));
            } else {
                let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                vops.encode_axpy(ctx, &mut enc, 1.0, gpu_w, 0.0, &basis[jj + 1]);
                ctx.queue.submit(Some(enc.finish()));
            }
            if let Some(ref mut profile) = profile {
                profile.arnoldi_normalization_phase += normalize_start.elapsed();
            }
        }

        if apply_solution_update {
            let solution_update_start = Instant::now();
            for ii in (0..=j).rev() {
                let mut sum = s[ii];
                for kk in ii + 1..=j {
                    sum -= h[ii * restart + kk] * y[kk];
                }
                y[ii] = sum / h[ii * restart + ii];
            }

            for i in 0..=j {
                let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                vops.encode_axpy(ctx, &mut enc, y[i], &basis[i], 1.0, gpu_x);
                ctx.queue.submit(Some(enc.finish()));
            }
            if let Some(ref mut profile) = profile {
                profile.solution_update_phase += solution_update_start.elapsed();
            }
        }

        if fixed_iterations.is_none() && gmres_r_norm < tol {
            if read_back_solution {
                let cpu_x = gpu_x.read_to_cpu(ctx);
                x.copy_from_slice(&cpu_x);
            }
            if let Some(ref mut profile) = profile {
                profile.iterations = iter_count;
                profile.final_residual = gmres_r_norm;
                profile.total_phase = total_start.elapsed();
            }
            return Ok(SolveResult { converged: true, iterations: iter_count, final_residual: gmres_r_norm });
        }
    }

    if fixed_iterations.is_some() && !read_back_solution {
        if let Some(ref mut profile) = profile {
            profile.iterations = iter_count;
            profile.final_residual = last_residual;
            profile.total_phase = total_start.elapsed();
        }
        return Err(SolverError::ConvergenceFailed { max_iter: target_iterations, residual: last_residual });
    }

    let finalization_start = Instant::now();
    compute_residual_into(ctx, spmv, vops, a, b, gpu_x, gpu_ax, gpu_r);
    let final_residual = vops.compute_norm2(ctx, gpu_r);
    if read_back_solution {
        let cpu_x = gpu_x.read_to_cpu(ctx);
        x.copy_from_slice(&cpu_x);
    }
    if let Some(ref mut profile) = profile {
        profile.iterations = iter_count;
        profile.final_residual = final_residual;
        profile.finalization_phase += finalization_start.elapsed();
        profile.total_phase = total_start.elapsed();
    }
    Err(SolverError::ConvergenceFailed { max_iter: target_iterations, residual: final_residual })
}

#[allow(clippy::too_many_arguments)]
fn compute_residual_into<T: Scalar>(
    ctx: &GpuContext,
    spmv: &SpmvPipeline,
    vops: &VectorOpsPipeline,
    a: &GpuCsrMatrix<T>,
    b: &GpuVector<T>,
    x: &GpuVector<T>,
    ax: &mut GpuVector<T>,
    r: &mut GpuVector<T>,
) {
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        spmv.encode_spmv(ctx, &mut enc, 1.0, a, x, 0.0, ax);
        ctx.queue.submit(Some(enc.finish()));
    }
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        vops.encode_axpy(ctx, &mut enc, 1.0, b, 0.0, r);
        ctx.queue.submit(Some(enc.finish()));
    }
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        vops.encode_axpy(ctx, &mut enc, -1.0, ax, 1.0, r);
        ctx.queue.submit(Some(enc.finish()));
    }
}
