//! GPU device buffer abstraction.
//!
//! Wraps a `wgpu::Buffer` with type-safe read/write operations,
//! staging-belt upload/download, and bytemuck-compatible element types.

use wgpu::BufferUsages;

/// A GPU-resident buffer typed over `T`.
///
/// `T` must implement `bytemuck::Pod` (plain-old-data) so that
/// the buffer contents can be safely transmuted to/from `&[u8]`.
pub struct DeviceBuffer<T: bytemuck::Pod> {
    _buf: wgpu::Buffer,
    _len: usize,
    _usage: BufferUsages,
    _phantom: std::marker::PhantomData<T>,
}
