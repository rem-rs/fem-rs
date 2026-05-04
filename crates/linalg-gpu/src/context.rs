//! GPU context: adapter, device, and queue management.
//!
//! Owns the wgpu `Device` and `Queue`.
//! Provides the entry point for all GPU operations in the crate.

/// Manages the wgpu device, queue, and pre-compiled compute pipelines.
///
/// Created once per process via [`GpuContext::new()`] (async). Pipelines are
/// compiled at construction time to avoid latency during solve iterations.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub features: GpuFeatures,
}

/// Detected GPU capabilities that affect shader selection.
#[derive(Debug, Clone)]
pub struct GpuFeatures {
    /// Native `f64` in WGSL (`wgpu::Features::SHADER_F64`).
    pub native_f64: bool,
    /// Maximum storage buffer size.
    pub max_buffer_size: u64,
    /// Maximum compute workgroups per dimension.
    pub max_compute_workgroups_per_dim: u32,
}

impl GpuContext {
    /// Initialize wgpu, select adapter, create device, and detect features.
    pub async fn new() -> Result<Self, crate::GpuError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(crate::GpuError::NoAdapter)?;

        let adapter_features = adapter.features();
        let has_f64 = adapter_features.contains(wgpu::Features::SHADER_F64);
        let limits = adapter.limits();

        let required_features = if has_f64 {
            wgpu::Features::SHADER_F64
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("fem-linalg-gpu"),
                    required_features,
                    required_limits: wgpu::Limits {
                        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                None,
            )
            .await?;

        let gpu_features = GpuFeatures {
            native_f64: has_f64,
            max_buffer_size: limits.max_storage_buffer_binding_size as u64,
            max_compute_workgroups_per_dim: limits.max_compute_workgroups_per_dimension,
        };

        if !has_f64 {
            log::warn!("SHADER_F64 not available; f64 emulation path not yet implemented — f64 buffers will error");
        }

        Ok(Self { device, queue, features: gpu_features })
    }

    /// Synchronous init using pollster (for non-async contexts).
    pub fn new_sync() -> Result<Self, crate::GpuError> {
        pollster::block_on(Self::new())
    }
}
