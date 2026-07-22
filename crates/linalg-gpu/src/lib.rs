pub mod assembly;
pub mod iga_assembly;
pub mod pa_apply;
pub mod cg;
#[cfg(feature = "amg")]
pub mod amg_precond;
pub mod backend;
pub mod backend_wgpu;
#[cfg(feature = "cuda")]
pub mod backend_cuda;
pub mod buffer;
pub mod context;
pub mod csr;
pub mod direct;
pub mod jacobi;
pub mod spmv_pipeline;
pub mod vector;
pub mod vector_pipeline;

pub use assembly::{
    assemble_poisson_2d_p1,
    assemble_poisson_2d_p2,
    assemble_poisson_2d_q1,
    assemble_poisson_3d_p1,
    assemble_poisson_3d_hex8,
    assemble_mass_3d_hex8,
    assemble_mass_2d_tri3,
    assemble_mass_2d_quad4,
    assemble_mass_3d_tet4,
    assemble_elasticity_2d_tri3,
    assemble_elasticity_3d_tet4,
    assemble_elasticity_3d_hex8,
    assemble_poisson_2d_p1_gpu,
    assemble_mass_2d_tri3_gpu,
    assemble_elasticity_2d_tri3_gpu,
    assemble_poisson_2d_p1_f64,
    triplets_f64_to_gpu_csr,
};
pub use iga_assembly::{
    assemble_iga_bezier_diffusion_2d,
    assemble_iga_bezier_mass_2d,
    assemble_iga_bezier_diffusion_3d,
    assemble_iga_bezier_mass_3d,
    GpuIgaBezier2DElement,
    GpuIgaBezier3DElement,
};
pub use buffer::DeviceBuffer;
pub use context::{GpuContext, GpuFeatures};
pub use csr::GpuCsrMatrix;
pub use direct::{solve_dense_gpu, GpuDenseMatrix};
pub use jacobi::GpuJacobiPrecond;
pub use cg::{solve_cg_gpu, solve_pcg_jacobi_gpu};
#[cfg(feature = "amg")]
pub use amg_precond::GpuAmgPrecond;
pub use spmv_pipeline::SpmvPipeline;
pub use vector::GpuVector;
pub use vector_pipeline::{
    VectorOpsPipeline,
    read_partial_reduction,
    read_partial_reduction_staged,
};

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("no suitable wgpu adapter found")]
    NoAdapter,
    #[error("wgpu device request failed: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),
    #[error("buffer creation failed: {0}")]
    Buffer(#[from] wgpu::BufferAsyncError),
    #[error("f64 not supported by GPU and no emulation path compiled")]
    F64Unavailable,
}

pub type GpuResult<T> = Result<T, GpuError>;
