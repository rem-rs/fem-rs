// crates/linalg-gpu/src/spmv_pipeline.rs
use std::borrow::Cow;
use fem_core::Scalar;
use wgpu::util::DeviceExt;
use crate::{GpuContext, GpuCsrMatrix, GpuVector};

/// SpMV parameter uniform layout: alpha (f64), beta (f64), nrows (u32), _pad (u32)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SpmvParams {
    alpha: f64,
    beta: f64,
    nrows: u32,
    _pad: u32,
}

/// Pre-compiled SpMV compute pipeline.
pub struct SpmvPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl SpmvPipeline {
    pub fn new(device: &wgpu::Device, native_f64: bool) -> Self {
        let shader_source = if native_f64 {
            Cow::Borrowed(include_str!("spmv.wgsl"))
        } else {
            panic!("f64 emulation path not yet implemented");
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("spmv_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("spmv_bind_group_layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("spmv_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("spmv_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline, bind_group_layout }
    }

    /// Encode an SpMV dispatch: `y = alpha * A * x + beta * y`.
    pub fn encode_spmv<T: Scalar>(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        alpha: f64,
        mat: &GpuCsrMatrix<T>,
        x: &GpuVector<T>,
        beta: f64,
        y: &GpuVector<T>,
    ) {
        let params = SpmvParams {
            alpha,
            beta,
            nrows: mat.nrows,
            _pad: 0,
        };
        let uniform_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("spmv_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spmv_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: mat.row_ptr_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: mat.col_idx_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: mat.values_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: x.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: y.buffer().as_entire_binding() },
            ],
        });

        let workgroup_count = (mat.nrows + 255) / 256;
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("spmv_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroup_count, 1, 1);
        }
    }

    /// Compute `y = A * x` (convenience shorthand that creates a command encoder, dispatches, and submits).
    pub fn spmv<T: Scalar>(
        &self,
        ctx: &GpuContext,
        alpha: f64,
        mat: &GpuCsrMatrix<T>,
        x: &GpuVector<T>,
        beta: f64,
        y: &GpuVector<T>,
    ) {
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.encode_spmv(ctx, &mut encoder, alpha, mat, x, beta, y);
        ctx.queue.submit(Some(encoder.finish()));
    }
}
