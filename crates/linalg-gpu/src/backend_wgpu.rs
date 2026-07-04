//! wgpu backend implementation (default).
//!
//! Re-exports the existing wgpu-based types and provides the
//! [`Backend`] trait implementation.

use fem_core::Scalar;
use crate::context::GpuContext;
use crate::csr::GpuCsrMatrix;
use crate::vector::GpuVector;
use crate::spmv_pipeline::SpmvPipeline;
use crate::vector_pipeline::VectorOpsPipeline;
use crate::jacobi::GpuJacobiPrecond;

/// wgpu backend context.
pub struct WgpuBackend {
    pub ctx: GpuContext,
    spmv: SpmvPipeline,
    vops: VectorOpsPipeline,
}

impl WgpuBackend {
    pub fn new() -> Result<Self, crate::GpuError> {
        let ctx = GpuContext::new_sync()?;
        let spmv = SpmvPipeline::new(&ctx.device, ctx.features.native_f64);
        let vops = VectorOpsPipeline::new(&ctx.device, ctx.features.native_f64);
        Ok(Self { ctx, spmv, vops })
    }

    pub fn device(&self) -> &wgpu::Device { &self.ctx.device }
    pub fn queue(&self) -> &wgpu::Queue { &self.ctx.queue }
    pub fn spmv_pipeline(&self) -> &SpmvPipeline { &self.spmv }
    pub fn vops_pipeline(&self) -> &VectorOpsPipeline { &self.vops }
    pub fn features(&self) -> &crate::GpuFeatures { &self.ctx.features }
    pub fn native_f64(&self) -> bool { self.ctx.features.native_f64 }
    pub fn ctx(&self) -> &GpuContext { &self.ctx }
}

/// Compute `y = A * x` using the wgpu SpMV pipeline.
pub fn wgpu_spmv<T: Scalar>(
    backend: &WgpuBackend,
    alpha: f64,
    a: &GpuCsrMatrix<T>,
    x: &GpuVector<T>,
    beta: f64,
    y: &GpuVector<T>,
) {
    let mut enc = backend.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    backend.spmv.encode_spmv(&backend.ctx, &mut enc, alpha, a, x, beta, y);
    backend.queue().submit(Some(enc.finish()));
}

/// Compute `y = alpha * x + beta * y` using the wgpu vector ops pipeline.
pub fn wgpu_axpy<T: Scalar>(
    backend: &WgpuBackend,
    alpha: f64,
    x: &GpuVector<T>,
    beta: f64,
    y: &GpuVector<T>,
) {
    let mut enc = backend.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    backend.vops.encode_axpy(&backend.ctx, &mut enc, alpha, x, beta, y);
    backend.queue().submit(Some(enc.finish()));
}

/// Compute dot(a, b) on wgpu.
pub fn wgpu_dot<T: Scalar>(
    backend: &WgpuBackend,
    a: &GpuVector<T>,
    b: &GpuVector<T>,
) -> f64 {
    let n_wg = a.len().div_ceil(256);
    let dot_buf = crate::DeviceBuffer::with_staging(
        backend.device(),
        n_wg as u64 * std::mem::size_of::<T>() as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "backend_dot",
    );
    backend.vops.dispatch_dot_readback(&backend.ctx, a, b, &dot_buf)
}

/// Apply Jacobi preconditioner on wgpu.
pub fn wgpu_apply_jacobi<T: Scalar>(
    backend: &WgpuBackend,
    precond: &GpuJacobiPrecond<T>,
    r: &GpuVector<T>,
    z: &GpuVector<T>,
) {
    let mut enc = backend.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    precond.encode_apply(&backend.ctx, &mut enc, r, z);
    backend.queue().submit(Some(enc.finish()));
}
