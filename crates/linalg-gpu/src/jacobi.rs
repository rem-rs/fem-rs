//! GPU Jacobi (diagonal) preconditioner.
//!
//! Applies `z_i = (1/A_ii) * r_i` as a preconditioner for CG/GMRES.

use std::any::TypeId;
use fem_core::Scalar;
use fem_linalg::CsrMatrix;
use crate::{DeviceBuffer, GpuContext, GpuVector};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct JacobiParams {
    n: u32,
}

/// GPU-resident Jacobi preconditioner.
///
/// Stores the inverse diagonal of the system matrix.
/// Applies `z = D^{-1} r` on the GPU.
pub struct GpuJacobiPrecond<T: Scalar> {
    n: u32,
    diag_inv: DeviceBuffer,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Scalar> GpuJacobiPrecond<T> {
    /// Build from a CPU CSR matrix. Extracts the diagonal and builds the GPU pipeline.
    pub fn from_matrix(ctx: &GpuContext, a: &CsrMatrix<T>) -> Self {
        let n = a.nrows as u32;
        let diag_inv_values: Vec<T> = (0..a.nrows)
            .map(|i| {
                let d = a.get(i, i);
                if d.abs() > T::from_f64(1e-14) {
                    T::from_f64(1.0) / d
                } else {
                    T::from_f64(1.0)
                }
            })
            .collect();

        let diag_inv = DeviceBuffer::from_bytes_with_staging(
            &ctx.device, &ctx.queue, &diag_inv_values,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "jacobi_diag_inv",
        );

        let is_f64 = TypeId::of::<T>() == TypeId::of::<f64>();
        let shader = if is_f64 {
            ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("jacobi_shader_f64"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    include_str!("jacobi_f64.wgsl"),
                )),
            })
        } else {
            ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("jacobi_shader_f32"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    include_str!("jacobi_f32.wgsl"),
                )),
            })
        };

        let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jacobi_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new(8),
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
            ],
        });

        let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jacobi_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jacobi_pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params = JacobiParams { n };
        let params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jacobi_params"),
            size: std::mem::size_of::<JacobiParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        Self { n, diag_inv, pipeline, bgl, params_buf, _marker: std::marker::PhantomData }
    }

    /// Encode `z = D^{-1} * r` into an existing command encoder.
    pub fn encode_apply(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        r: &GpuVector<T>,
        z: &GpuVector<T>,
    ) {
        assert_eq!(r.len(), self.n);
        assert_eq!(z.len(), self.n);

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jacobi_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.diag_inv.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: r.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: z.buffer().as_entire_binding() },
            ],
        });

        let wg = self.n.div_ceil(256);
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jacobi_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &bg, &[]);
        cpass.dispatch_workgroups(wg, 1, 1);
    }
}
