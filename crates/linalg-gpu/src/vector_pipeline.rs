// crates/linalg-gpu/src/vector_pipeline.rs
use std::any::TypeId;
use std::borrow::Cow;
use std::sync::Mutex;
use fem_core::Scalar;
use crate::{DeviceBuffer, GpuContext, GpuVector};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AxpyParamsF64 {
    alpha: f64,
    beta: f64,
    len: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AxpyParamsF32 {
    alpha: f32,
    beta: f32,
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

/// Pre-compiled vector operations pipelines for supported scalar widths.
pub struct VectorOpsPipeline {
    axpy_pipeline_f64: Option<wgpu::ComputePipeline>,
    axpy_pipeline_f32: wgpu::ComputePipeline,
    axpy_bind_group_layout: wgpu::BindGroupLayout,
    axpy_params_buffer_f64: Option<wgpu::Buffer>,
    axpy_params_buffer_f32: wgpu::Buffer,
    dot_pipeline_f64: Option<wgpu::ComputePipeline>,
    dot_pipeline_f32: wgpu::ComputePipeline,
    dot_bind_group_layout: wgpu::BindGroupLayout,
    dot_params_buffer: wgpu::Buffer,
    dot_reduction_scratch: Mutex<Option<DeviceBuffer>>,
}

impl VectorOpsPipeline {
    pub fn new(device: &wgpu::Device, native_f64: bool) -> Self {
        let shader_f32 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vector_ops_shader_f32"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("vector_ops_f32.wgsl"))),
        });
        let shader_f64 = native_f64.then(|| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vector_ops_shader_f64"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("vector_ops.wgsl"))),
            })
        });

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

        let axpy_pipeline_f32 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("axpy_pipeline_f32"),
            layout: Some(&axpy_layout),
            module: &shader_f32,
            entry_point: Some("axpy_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let axpy_pipeline_f64 = shader_f64.as_ref().map(|shader| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("axpy_pipeline_f64"),
                layout: Some(&axpy_layout),
                module: shader,
                entry_point: Some("axpy_main"),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let axpy_params_buffer_f32 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("axpy_params_f32"),
            size: std::mem::size_of::<AxpyParamsF32>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let axpy_params_buffer_f64 = native_f64.then(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("axpy_params_f64"),
                size: std::mem::size_of::<AxpyParamsF64>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

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

        let dot_pipeline_f32 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dot_pipeline_f32"),
            layout: Some(&dot_layout),
            module: &shader_f32,
            entry_point: Some("dot_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let dot_pipeline_f64 = shader_f64.as_ref().map(|shader| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("dot_pipeline_f64"),
                layout: Some(&dot_layout),
                module: shader,
                entry_point: Some("dot_main"),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let dot_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dot_params"),
            size: std::mem::size_of::<DotParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            axpy_pipeline_f64,
            axpy_pipeline_f32,
            axpy_bind_group_layout: axpy_bgl,
            axpy_params_buffer_f64,
            axpy_params_buffer_f32,
            dot_pipeline_f64,
            dot_pipeline_f32,
            dot_bind_group_layout: dot_bgl,
            dot_params_buffer,
            dot_reduction_scratch: Mutex::new(None),
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
        let (pipeline, params_binding) = if TypeId::of::<T>() == TypeId::of::<f64>() {
            let params = AxpyParamsF64 { alpha, beta, len: x.len(), _pad: 0 };
            let params_buffer = self
                .axpy_params_buffer_f64
                .as_ref()
                .expect("f64 AXPY requested on adapter without SHADER_F64 support");
            ctx.queue.write_buffer(params_buffer, 0, bytemuck::bytes_of(&params));
            (
                self.axpy_pipeline_f64
                    .as_ref()
                    .expect("f64 AXPY requested on adapter without SHADER_F64 support"),
                params_buffer.as_entire_binding(),
            )
        } else if TypeId::of::<T>() == TypeId::of::<f32>() {
            let params = AxpyParamsF32 {
                alpha: alpha as f32,
                beta: beta as f32,
                len: x.len(),
                _pad: 0,
            };
            ctx.queue.write_buffer(&self.axpy_params_buffer_f32, 0, bytemuck::bytes_of(&params));
            (&self.axpy_pipeline_f32, self.axpy_params_buffer_f32.as_entire_binding())
        } else {
            panic!("unsupported scalar type for vector ops pipeline");
        };

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("axpy_bg"),
            layout: &self.axpy_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_binding },
                wgpu::BindGroupEntry { binding: 1, resource: x.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y.buffer().as_entire_binding() },
            ],
        });

        let workgroups = x.len().div_ceil(256);
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("axpy_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bg, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Encode dot product: result stored in `result_buf` with one partial per workgroup.
    pub fn encode_dot<T: Scalar>(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuVector<T>,
        b: &GpuVector<T>,
        result_buf: &wgpu::Buffer,
    ) {
        assert_eq!(a.len(), b.len());
        let n_workgroups = a.len().div_ceil(256u32);
        let params = DotParams { len: a.len(), _pad0: 0, _pad1: 0, _pad2: 0 };
        ctx.queue.write_buffer(&self.dot_params_buffer, 0, bytemuck::bytes_of(&params));
        let pipeline = if TypeId::of::<T>() == TypeId::of::<f64>() {
            self.dot_pipeline_f64
                .as_ref()
                .expect("f64 dot requested on adapter without SHADER_F64 support")
        } else if TypeId::of::<T>() == TypeId::of::<f32>() {
            &self.dot_pipeline_f32
        } else {
            panic!("unsupported scalar type for vector ops pipeline");
        };

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dot_bg"),
            layout: &self.dot_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.dot_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: result_buf.as_entire_binding() },
            ],
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dot_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bg, &[]);
        cpass.dispatch_workgroups(n_workgroups, 1, 1);
    }

    /// Dispatch dot(a,b), copy the partial reductions into the staging buffer,
    /// and return the CPU-reduced scalar with a single queue submission.
    pub fn dispatch_dot_readback<T: Scalar>(
        &self,
        ctx: &GpuContext,
        a: &GpuVector<T>,
        b: &GpuVector<T>,
        result_buf: &DeviceBuffer,
    ) -> f64 {
        let partial_count = a.len().div_ceil(256);
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.encode_dot(ctx, &mut enc, a, b, result_buf.buffer());
        result_buf.encode_copy_to_staging(&mut enc);
        ctx.queue.submit(Some(enc.finish()));

        if TypeId::of::<T>() == TypeId::of::<f64>() {
            read_partial_reduction_mapped_f64(ctx, result_buf.staging(), partial_count)
        } else if TypeId::of::<T>() == TypeId::of::<f32>() {
            read_partial_reduction_mapped_f32(ctx, result_buf.staging(), partial_count) as f64
        } else {
            panic!("unsupported scalar type for vector ops pipeline");
        }
    }

    /// Dispatch multiple dot(a, b_i) reductions with a single queue submission,
    /// then map and CPU-reduce each staged result buffer.
    pub fn dispatch_dot_readback_batch<T: Scalar>(
        &self,
        ctx: &GpuContext,
        a: &GpuVector<T>,
        others: &[GpuVector<T>],
        result_bufs: &[DeviceBuffer],
        out: &mut [f64],
    ) {
        assert_eq!(others.len(), result_bufs.len());
        assert_eq!(others.len(), out.len());
        let partial_count = a.len().div_ceil(256);
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        for (other, result_buf) in others.iter().zip(result_bufs.iter()) {
            assert_eq!(a.len(), other.len());
            self.encode_dot(ctx, &mut enc, a, other, result_buf.buffer());
            result_buf.encode_copy_to_staging(&mut enc);
        }
        ctx.queue.submit(Some(enc.finish()));

        if TypeId::of::<T>() == TypeId::of::<f64>() {
            for (slot, result_buf) in out.iter_mut().zip(result_bufs.iter()) {
                *slot = read_partial_reduction_mapped_f64(ctx, result_buf.staging(), partial_count);
            }
        } else if TypeId::of::<T>() == TypeId::of::<f32>() {
            for (slot, result_buf) in out.iter_mut().zip(result_bufs.iter()) {
                *slot = read_partial_reduction_mapped_f32(ctx, result_buf.staging(), partial_count) as f64;
            }
        } else {
            panic!("unsupported scalar type for vector ops pipeline");
        }
    }

    /// Compute ||v||₂ by dispatching dot(v,v) and reading back the partial reduction.
    pub fn compute_norm2<T: Scalar>(&self, ctx: &GpuContext, v: &GpuVector<T>) -> f64 {
        let n_wg = v.len().div_ceil(256);
        let required_size = n_wg as u64 * std::mem::size_of::<T>() as u64;
        let mut scratch = self.dot_reduction_scratch.lock().unwrap();
        let needs_resize = scratch
            .as_ref()
            .map(|buf| buf.size() < required_size)
            .unwrap_or(true);
        if needs_resize {
            *scratch = Some(DeviceBuffer::with_staging(
                &ctx.device,
                required_size,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                "norm2_tmp",
            ));
        }
        let dot_buf = scratch.as_ref().expect("norm2 scratch buffer should exist");
        self.dispatch_dot_readback(ctx, v, v, dot_buf).sqrt()
    }
}

/// Read back a GPU partial-reduction buffer (post dot dispatch) and sum on CPU.
pub fn read_partial_reduction(ctx: &GpuContext, result_buf: &wgpu::Buffer, n_wg: u32) -> f64 {
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reduce_staging"),
        size: n_wg as u64 * 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_buffer_to_buffer(result_buf, 0, &staging, 0, n_wg as u64 * 8);
    ctx.queue.submit(Some(enc.finish()));

    let mapped = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    mapped.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();

    let view = mapped.get_mapped_range();
    let partials: &[f64] = bytemuck::cast_slice(&view);
    let sum: f64 = partials.iter().sum();
    drop(view);
    staging.unmap();
    sum
}

/// Read back a GPU partial-reduction buffer through a reusable staging buffer.
pub fn read_partial_reduction_staged(
    ctx: &GpuContext,
    result_buf: &DeviceBuffer,
    partial_count: u32,
) -> f64 {
    read_partial_reduction_staged_f64(ctx, result_buf, partial_count)
}

fn read_partial_reduction_staged_f64(
    ctx: &GpuContext,
    result_buf: &DeviceBuffer,
    partial_count: u32,
) -> f64 {
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    result_buf.encode_copy_to_staging(&mut enc);
    ctx.queue.submit(Some(enc.finish()));
    read_partial_reduction_mapped_f64(ctx, result_buf.staging(), partial_count)
}

fn read_partial_reduction_mapped_f64(
    ctx: &GpuContext,
    staging: &wgpu::Buffer,
    partial_count: u32,
) -> f64 {
    let mapped = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    mapped.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();

    let view = mapped.get_mapped_range();
    let partials: &[f64] = bytemuck::cast_slice(&view);
    let sum: f64 = partials.iter().take(partial_count as usize).sum();
    drop(view);
    staging.unmap();
    sum
}

fn read_partial_reduction_mapped_f32(
    ctx: &GpuContext,
    staging: &wgpu::Buffer,
    partial_count: u32,
) -> f32 {
    let mapped = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    mapped.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();

    let view = mapped.get_mapped_range();
    let partials: &[f32] = bytemuck::cast_slice(&view);
    let sum: f32 = partials.iter().take(partial_count as usize).copied().sum();
    drop(view);
    staging.unmap();
    sum
}
