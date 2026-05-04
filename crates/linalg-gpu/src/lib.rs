pub mod buffer;
pub mod context;
pub mod csr;
pub mod spmv_pipeline;
pub mod vector;
pub mod vector_pipeline;

pub use buffer::DeviceBuffer;
pub use context::{GpuContext, GpuFeatures};
pub use csr::GpuCsrMatrix;
pub use spmv_pipeline::SpmvPipeline;
pub use vector::GpuVector;
pub use vector_pipeline::{VectorOpsPipeline, read_partial_reduction};

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
