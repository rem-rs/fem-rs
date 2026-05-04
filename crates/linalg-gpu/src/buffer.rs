//! GPU device buffer with optional staging for readback.
//!
//! Wraps a `wgpu::Buffer` and provides helper methods to upload slices
//! of `bytemuck::Pod` data and read back via a staging buffer.

use wgpu::util::DeviceExt;

/// Owned wgpu buffer with optional staging for readback.
///
/// Wraps a `wgpu::Buffer` and provides helper methods to upload slices
/// of `bytemuck::Pod` data and read back via a staging buffer.
pub struct DeviceBuffer {
    buffer: wgpu::Buffer,
    size: u64,
    usage: wgpu::BufferUsages,
    /// Present only when readback was requested.
    staging: Option<wgpu::Buffer>,
}

impl DeviceBuffer {
    /// Create a buffer with the given size and usage, initialized to zero.
    pub fn new(device: &wgpu::Device, size: u64, usage: wgpu::BufferUsages, label: &str) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });
        Self { buffer, size, usage, staging: None }
    }

    /// Create a buffer from a byte slice (one-shot upload via staging).
    pub fn from_bytes<T: bytemuck::Pod>(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        data: &[T],
        usage: wgpu::BufferUsages,
        label: &str,
    ) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage,
        });
        Self {
            buffer,
            size: (data.len() * std::mem::size_of::<T>()) as u64,
            usage,
            staging: None,
        }
    }

    /// Create with a staging buffer for CPU readback.
    pub fn with_staging(device: &wgpu::Device, size: u64, usage: wgpu::BufferUsages, label: &str) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("{label}_buffer").as_str()),
            size,
            usage,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("{label}_staging").as_str()),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            size,
            usage,
            staging: Some(staging),
        }
    }

    /// Raw wgpu buffer reference.
    pub fn buffer(&self) -> &wgpu::Buffer { &self.buffer }

    /// Buffer usage flags.
    pub fn usage(&self) -> wgpu::BufferUsages { self.usage }

    /// Size in bytes.
    pub fn size(&self) -> u64 { self.size }

    /// Staging buffer reference (panics if not created with `with_staging`).
    pub fn staging(&self) -> &wgpu::Buffer {
        self.staging.as_ref().expect("DeviceBuffer has no staging buffer")
    }

    /// Encode a copy from self to the staging buffer.
    pub fn encode_copy_to_staging(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(ref staging) = self.staging {
            encoder.copy_buffer_to_buffer(&self.buffer, 0, staging, 0, self.size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device() -> (wgpu::Device, wgpu::Queue) {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .expect("need a wgpu adapter");
            adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .expect("need a wgpu device")
        })
    }

    #[test]
    fn buffer_from_f64_slice() {
        let (device, queue) = test_device();
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let buf = DeviceBuffer::from_bytes(
            &device, &queue, &data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "test_f64",
        );
        assert_eq!(buf.size(), 4 * 8);
    }

    #[test]
    fn buffer_with_staging_has_staging() {
        let (device, _queue) = test_device();
        let buf = DeviceBuffer::with_staging(
            &device,
            1024,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "test_staging",
        );
        let _staging = buf.staging(); // does not panic
    }
}
