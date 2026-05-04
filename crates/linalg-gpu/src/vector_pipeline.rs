// crates/linalg-gpu/src/vector_pipeline.rs
use std::borrow::Cow;
use fem_core::Scalar;
use wgpu::util::DeviceExt;
use crate::{GpuContext, GpuVector};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AxpyParams {
    alpha: f64,
    beta: f64,
    len: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DotParams {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Pre-compiled vector operations pipelines.
pub struct VectorOpsPipeline {
    axpy_pipeline: wgpu::ComputePipeline,
    axpy_bind_group_layout: wgpu::BindGroupLayout,
    dot_pipeline: wgpu::ComputePipeline,
    dot_bind_group_layout: wgpu::BindGroupLayout,
}

impl VectorOpsPipeline {
    pub fn new(device: &wgpu::Device, native_f64: bool) -> Self {
        let shader_source = if native_f64 {
            Cow::Borrowed(include_str!("vector_ops.wgsl"))
        } else {
            panic!("f64 emulation path not yet implemented");
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vector_ops_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source),
        });

        // Axpy bind group layout
        let axpy_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("axpy_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let axpy_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("axpy_layout"),
            bind_group_layouts: &[&axpy_bgl],
            push_constant_ranges: &[],
        });

        let axpy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("axpy_pipeline"),
            layout: Some(&axpy_layout),
            module: &shader,
            entry_point: Some("axpy_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Dot bind group layout (with result buffer)
        let dot_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dot_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let dot_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dot_layout"),
            bind_group_layouts: &[&dot_bgl],
            push_constant_ranges: &[],
        });

        let dot_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dot_pipeline"),
            layout: Some(&dot_layout),
            module: &shader,
            entry_point: Some("dot_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            axpy_pipeline,
            axpy_bind_group_layout: axpy_bgl,
            dot_pipeline,
            dot_bind_group_layout: dot_bgl,
        }
    }

    /// Encode axpy: `y = alpha * x + beta * y`
    pub fn encode_axpy<T: Scalar>(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        alpha: f64,
        x: &GpuVector<T>,
        beta: f64,
        y: &GpuVector<T>,
    ) {
        assert_eq!(x.len(), y.len());
        let params = AxpyParams { alpha, beta, len: x.len(), _pad: 0 };
        let uniform_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("axpy_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("axpy_bg"),
            layout: &self.axpy_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y.buffer().as_entire_binding() },
            ],
        });

        let workgroups = (x.len() + 255) / 256;
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("axpy_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.axpy_pipeline);
        cpass.set_bind_group(0, &bg, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Encode dot product: result stored in `result_buf` (a GPU buffer of
    /// length n_workgroups, requires a second-pass reduction on CPU).
    ///
    /// The caller should allocate `result_buf` with `n_workgroups * 8` bytes
    /// where `n_workgroups = (len + 255) / 256`.
    pub fn encode_dot<T: Scalar>(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuVector<T>,
        b: &GpuVector<T>,
        result_buf: &wgpu::Buffer,
    ) {
        assert_eq!(a.len(), b.len());
        let n_workgroups = (a.len() + 255u32) / 256u32;
        let params = DotParams { len: a.len(), _pad0: 0, _pad1: 0, _pad2: 0 };
        let uniform_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dot_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dot_bg"),
            layout: &self.dot_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: result_buf.as_entire_binding() },
            ],
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dot_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.dot_pipeline);
        cpass.set_bind_group(0, &bg, &[]);
        cpass.dispatch_workgroups(n_workgroups, 1, 1);
    }
}
