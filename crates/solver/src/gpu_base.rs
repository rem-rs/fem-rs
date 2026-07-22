//! Shared GPU solver core — holds pipelines, matrix, RHS, and basic vectors
//! that are common across all GPU iterative solvers (CG, GMRES, BiCGSTAB, etc.).
//!
//! The pattern is composition: each solver workspace embeds a `GpuSolverBase<T>`
//! and adds its own solver-specific GPU vectors and CPU-side scratch arrays.

use fem_core::Scalar;
use fem_linalg::CsrMatrix;
use fem_linalg_gpu::{
    DeviceBuffer, GpuContext, GpuCsrMatrix, GpuVector,
    SpmvPipeline, VectorOpsPipeline,
};
use wgpu;

/// Shared resources for a GPU-resident iterative solver.
///
/// Owns the matrix, RHS, solution vector, residual vector, and all wgpu
/// pipelines so that derived workspaces (CG, GMRES, …) only need to add
/// their own temporary vectors and solver-specific logic.
///
/// The scalar parameter `T` should be `f32` or `f64` and determines the
/// precision of matrix/vector storage. Scalar operation coefficients
/// (α, β for SpMV/axpy) are always `f64` regardless of `T`; they are
/// cast inside the wgpu pipeline as needed.
pub struct GpuSolverBase<T: Scalar> {
    /// Problem size.
    pub n: u32,
    /// Matrix–vector product pipeline.
    pub spmv: SpmvPipeline,
    /// Vector operations pipeline (axpy, dot, norm2).
    pub vops: VectorOpsPipeline,
    /// Matrix stored on GPU.
    pub gpu_a: GpuCsrMatrix<T>,
    /// Right-hand side on GPU.
    pub gpu_b: GpuVector<T>,
    /// Solution vector on GPU (initial guess on entry, solution on exit).
    pub gpu_x: GpuVector<T>,
    /// Residual vector on GPU.
    pub gpu_r: GpuVector<T>,
    /// Staging buffer for dot-product reduction readback.
    pub dot_buf: DeviceBuffer,
    /// Precomputed ‖b‖₂ for convergence checks.
    pub b_norm: f64,
}

impl<T: Scalar> GpuSolverBase<T> {
    /// Build the shared GPU resources for a fixed matrix and right-hand side.
    pub fn new(ctx: &GpuContext, a: &CsrMatrix<T>, b: &[T]) -> Self {
        let n = a.nrows as u32;
        assert_eq!(a.ncols as u32, n, "matrix must be square");
        assert_eq!(b.len() as u32, n);

        let spmv = SpmvPipeline::new(&ctx.device, ctx.features.native_f64);
        let vops = VectorOpsPipeline::new(&ctx.device, ctx.features.native_f64);
        let gpu_a = GpuCsrMatrix::<T>::from_cpu(ctx, a);
        let gpu_b = GpuVector::from_slice(ctx, b);
        let gpu_x = GpuVector::<T>::zeros(ctx, n);
        let gpu_r = GpuVector::<T>::zeros(ctx, n);
        let n_wg = n.div_ceil(256);
        let dot_buf = DeviceBuffer::with_staging(
            &ctx.device,
            n_wg as u64 * std::mem::size_of::<T>() as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "solver_base_dot_buf",
        );
        let b_norm = vops.compute_norm2(ctx, &gpu_b);

        Self { n, spmv, vops, gpu_a, gpu_b, gpu_x, gpu_r, dot_buf, b_norm }
    }

    /// Compute the initial residual `r = b − A·x` in a single encoder.
    /// Coefficients 1.0 and -1.0 are always f64; the pipeline casts internally.
    pub fn compute_residual(&self, ctx: &GpuContext, enc: &mut wgpu::CommandEncoder) {
        self.spmv.encode_spmv(ctx, enc, 1.0, &self.gpu_a, &self.gpu_x, 0.0, &self.gpu_r);
        self.vops.encode_axpy(ctx, enc, 1.0, &self.gpu_b, 0.0, &self.gpu_r);
        self.vops.encode_axpy(ctx, enc, -1.0, &self.gpu_r, 1.0, &self.gpu_r);
    }

    /// Submit a simple encoder and wait for completion.
    pub fn submit_and_wait(&self, ctx: &GpuContext, enc: wgpu::CommandEncoder) {
        ctx.queue.submit([enc.finish()]);
        ctx.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Compute the 2-norm of a GPU vector (synchronous readback).
    pub fn norm2(&self, ctx: &GpuContext, v: &GpuVector<T>) -> f64 {
        self.vops.compute_norm2(ctx, v)
    }

    /// Compute the dot product of two GPU vectors (synchronous readback, returns f64).
    pub fn dot_readback(&self, ctx: &GpuContext, u: &GpuVector<T>, v: &GpuVector<T>) -> f64 {
        self.vops.dispatch_dot_readback(ctx, u, v, &self.dot_buf)
    }

    /// Read the solution vector back to CPU.
    pub fn read_solution(&self, ctx: &GpuContext) -> Vec<T> {
        self.gpu_x.read_to_cpu(ctx)
    }
}
