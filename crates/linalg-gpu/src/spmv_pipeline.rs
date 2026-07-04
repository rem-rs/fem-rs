// crates/linalg-gpu/src/spmv_pipeline.rs
use std::any::TypeId;
use std::borrow::Cow;
use fem_core::Scalar;
use crate::{GpuContext, GpuCsrMatrix, GpuVector};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SpmvParamsF64 {
    alpha: f64,
    beta: f64,
    nrows: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SpmvParamsF32 {
    alpha: f32,
    beta: f32,
    nrows: u32,
    _pad: u32,
}

/// Pre-compiled SpMV compute pipelines for supported scalar widths.
pub struct SpmvPipeline {
    pipeline_f64: Option<wgpu::ComputePipeline>,
    pipeline_f32: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer_f64: Option<wgpu::Buffer>,
    params_buffer_f32: wgpu::Buffer,
}

impl SpmvPipeline {
    pub fn new(device: &wgpu::Device, native_f64: bool) -> Self {
        let shader_f32 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("spmv_shader_f32"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("spmv_f32.wgsl"))),
        });
        let shader_f64 = native_f64.then(|| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("spmv_shader_f64"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("spmv.wgsl"))),
            })
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

        let pipeline_f32 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("spmv_pipeline_f32"),
            layout: Some(&pipeline_layout),
            module: &shader_f32,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_f64 = shader_f64.as_ref().map(|shader| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("spmv_pipeline_f64"),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let params_buffer_f32 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spmv_params_f32"),
            size: std::mem::size_of::<SpmvParamsF32>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buffer_f64 = native_f64.then(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("spmv_params_f64"),
                size: std::mem::size_of::<SpmvParamsF64>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        Self {
            pipeline_f64,
            pipeline_f32,
            bind_group_layout,
            params_buffer_f64,
            params_buffer_f32,
        }
    }

    /// Encode an SpMV dispatch: `y = alpha * A * x + beta * y`.
    #[allow(clippy::too_many_arguments)]
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
        let (pipeline, params_binding) = if TypeId::of::<T>() == TypeId::of::<f64>() {
            let params = SpmvParamsF64 { alpha, beta, nrows: mat.nrows, _pad: 0 };
            let params_buffer = self
                .params_buffer_f64
                .as_ref()
                .expect("f64 SpMV requested on adapter without SHADER_F64 support");
            ctx.queue.write_buffer(params_buffer, 0, bytemuck::bytes_of(&params));
            (
                self.pipeline_f64
                    .as_ref()
                    .expect("f64 SpMV requested on adapter without SHADER_F64 support"),
                params_buffer.as_entire_binding(),
            )
        } else if TypeId::of::<T>() == TypeId::of::<f32>() {
            let params = SpmvParamsF32 {
                alpha: alpha as f32,
                beta: beta as f32,
                nrows: mat.nrows,
                _pad: 0,
            };
            ctx.queue.write_buffer(&self.params_buffer_f32, 0, bytemuck::bytes_of(&params));
            (&self.pipeline_f32, self.params_buffer_f32.as_entire_binding())
        } else {
            panic!("unsupported scalar type for SpMV pipeline");
        };

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spmv_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_binding },
                wgpu::BindGroupEntry { binding: 1, resource: mat.row_ptr_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: mat.col_idx_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: mat.values_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: x.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: y.buffer().as_entire_binding() },
            ],
        });

        let workgroup_count = mat.nrows.div_ceil(256);
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("spmv_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroup_count, 1, 1);
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
