//! GPU-resident dense vector.
//!
//! Wraps a `DeviceBuffer` with a known length and provides
//! host↔device transfer methods.

use crate::buffer::DeviceBuffer;

/// A dense vector resident on the GPU.
///
/// `T` must implement `bytemuck::Pod` (typically `f32`).
pub struct GpuVector<T: bytemuck::Pod> {
    _buf: DeviceBuffer<T>,
    _len: usize,
}
