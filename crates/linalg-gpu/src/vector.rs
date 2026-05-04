//! GPU-resident dense vector.
//!
//! Wraps a `DeviceBuffer` with a known length and provides
//! host↔device transfer methods.

use std::marker::PhantomData;
use fem_core::Scalar;
use crate::buffer::DeviceBuffer;
use crate::GpuContext;

/// Dense vector resident on the GPU.
pub struct GpuVector<T: Scalar> {
    len: u32,
    buffer: DeviceBuffer,
    _marker: PhantomData<T>,
}

impl<T: Scalar> GpuVector<T> {
    /// Create a zero-initialized vector of length `len` on the GPU.
    pub fn zeros(ctx: &GpuContext, len: u32) -> Self {
        let size = len as u64 * std::mem::size_of::<T>() as u64;
        let buffer = DeviceBuffer::new(
            &ctx.device,
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "gpu_vector",
        );
        // Zero fill via queue write
        let zeros = vec![0u8; size as usize];
        ctx.queue.write_buffer(buffer.buffer(), 0, &zeros);
        Self { len, buffer, _marker: PhantomData }
    }

    /// Upload from a CPU slice.
    pub fn from_slice(ctx: &GpuContext, data: &[T]) -> Self {
        let buffer = DeviceBuffer::from_bytes(
            &ctx.device,
            &ctx.queue,
            data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "gpu_vector",
        );
        Self {
            len: data.len() as u32,
            buffer,
            _marker: PhantomData,
        }
    }

    /// Read back to CPU. Blocks until the copy completes.
    /// Only use for convergence checks (once per iteration).
    pub fn read_to_cpu(&self, ctx: &GpuContext) -> Vec<T> {
        let size = self.len as u64 * std::mem::size_of::<T>() as u64;
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(self.buffer.buffer(), 0, &staging, 0, size);
        ctx.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    /// Length of the vector.
    pub fn len(&self) -> u32 { self.len }

    /// Raw buffer reference for pipeline binding.
    pub fn buffer(&self) -> &wgpu::Buffer { self.buffer.buffer() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuContext;

    fn ctx() -> GpuContext {
        GpuContext::new_sync().expect("gpu context")
    }

    #[test]
    fn zeros_is_zero() {
        let gpu = ctx();
        let v: GpuVector<f64> = GpuVector::zeros(&gpu, 10);
        let cpu = v.read_to_cpu(&gpu);
        assert_eq!(cpu.len(), 10);
        for &x in &cpu { assert_eq!(x, 0.0); }
    }

    #[test]
    fn from_slice_roundtrip() {
        let gpu = ctx();
        let data: Vec<f64> = vec![1.0, 2.0, 3.14159, -5.0];
        let v = GpuVector::from_slice(&gpu, &data);
        let cpu = v.read_to_cpu(&gpu);
        assert_eq!(cpu.len(), data.len());
        for (a, b) in data.iter().zip(cpu.iter()) {
            assert!((a - b).abs() < 1e-15, "mismatch: {a} vs {b}");
        }
    }
}
