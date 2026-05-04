//! GPU context: adapter, device, and queue management.
//!
//! Owns the wgpu `Instance`, `Adapter`, `Device`, and `Queue`.
//! Provides the entry point for all GPU operations in the crate.

/// Manages the wgpu instance, adapter, device, and queue.
///
/// Created once per application (or per GPU) and shared across
/// all GPU-accelerated linear-algebra operations.
pub struct GpuContext {
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    _device: wgpu::Device,
    _queue: wgpu::Queue,
}
