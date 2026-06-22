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
        let buffer = DeviceBuffer::with_staging(
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
        let buffer = DeviceBuffer::from_bytes_with_staging(
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

    /// Overwrite the vector contents from a CPU slice.
    pub fn write_from_slice(&self, ctx: &GpuContext, data: &[T]) {
        assert_eq!(data.len() as u32, self.len, "slice length must match gpu vector length");
        ctx.queue.write_buffer(self.buffer.buffer(), 0, bytemuck::cast_slice(data));
    }

    /// Fill the vector with zeros.
    pub fn fill_zero(&self, ctx: &GpuContext) {
        let size = self.len as u64 * std::mem::size_of::<T>() as u64;
        let zeros = vec![0u8; size as usize];
        ctx.queue.write_buffer(self.buffer.buffer(), 0, &zeros);
    }

    /// Read back to CPU. Blocks until the copy completes.
    /// Only use for convergence checks (once per iteration).
    pub fn read_to_cpu(&self, ctx: &GpuContext) -> Vec<T> {
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.buffer.encode_copy_to_staging(&mut encoder);
        ctx.queue.submit(Some(encoder.finish()));

        let staging = self.buffer.staging();
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

    #[test]
    fn repeated_readback_reuses_staging() {
        let gpu = ctx();
        let data: Vec<f64> = vec![0.5, -2.0, 9.25, 4.0];
        let v = GpuVector::from_slice(&gpu, &data);

        let first = v.read_to_cpu(&gpu);
        let second = v.read_to_cpu(&gpu);

        assert_eq!(first, data);
        assert_eq!(second, data);
    }
}
