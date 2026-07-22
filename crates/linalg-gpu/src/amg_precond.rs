//! GPU-accelerated AMG preconditioner (requires `amg` feature).
//!
//! Hybrid approach: finest-level Jacobi smoothing runs on GPU via a wgpu
//! compute shader; the coarse-grid correction (restriction / coarse solve /
//! prolongation) falls back to CPU for now.
//!
//! The `diag_inv` buffer is uploaded to GPU once at construction time.

use crate::{DeviceBuffer, GpuContext, GpuVector};
use std::borrow::Cow;

const WGSL: &str = r#"
struct Params { n: u32, _pad: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>  diag_inv: array<f32>;
@group(0) @binding(2) var<storage, read>  r: array<f32>;
@group(0) @binding(3) var<storage, read_write> z: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }
    z[i] = diag_inv[i] * r[i];
}
"#;

/// GPU-accelerated AMG preconditioner.
///
/// Stores the finest-level diagonal on GPU for Jacobi smoothing.
/// Multi-level V-cycles fall back to CPU until deeper level matrices are
/// accessible for GPU-based coarse-grid correction.
pub struct GpuAmgPrecond {
    cpu_solver: fem_amg::AmgSolver<f64>,
    diag_inv_gpu: DeviceBuffer,
    params_buf: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    n: u32,
}

impl GpuAmgPrecond {
    /// Build the AMG hierarchy and upload finest-level diagonal to GPU.
    pub fn new(
        ctx: &GpuContext,
        a: &fem_linalg::CsrMatrix<f64>,
        config: fem_amg::AmgConfig,
        cycle: fem_amg::CycleType,
    ) -> Self {
        let n = a.nrows as u32;
        let diag_inv: Vec<f32> = (0..a.nrows)
            .map(|i| { let d = a.get(i, i); if d.abs() > 1e-14 { (1.0 / d) as f32 } else { 1.0_f32 } })
            .collect();
        let cpu_solver = fem_amg::AmgSolver::setup(a, config).with_cycle(cycle);

        let device = &ctx.device;
        let diag_inv_gpu = DeviceBuffer::from_slice(device, &diag_inv, wgpu::BufferUsages::STORAGE, "amg_diag_inv");

        let params = [n, 0u32];
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("amg_params"), contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let (pipeline, bgl) = Self::build_pipeline(device, false);
        Self { cpu_solver, diag_inv_gpu, params_buf, pipeline, bgl, n }
    }

    fn build_pipeline(device: &wgpu::Device, _native_f64: bool) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amg_jacobi_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("amg_jacobi_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("amg_jacobi_pl"), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("amg_jacobi_pipe"), layout: Some(&pl), module: &shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        (pipeline, bgl)
    }

    /// Apply one V-cycle.
    ///
    /// When `skip_coarse` is `false` (default), runs the full multi-level CPU
    /// V-cycle. When `skip_coarse` is `true`, only the finest-level GPU Jacobi
    /// smoothing is applied — useful as a cheap diagonal preconditioner or
    /// when the AMG hierarchy is used purely for its diagonal scaling.
    ///
    /// The coarse-grid correction (restriction, coarsest solve, prolongation)
    /// still uses the CPU `AmgSolver` — the fine residual is read back for
    /// the CPU cycle, then the corrected solution is written back to GPU.
    pub fn apply(&self, ctx: &GpuContext, r: &GpuVector<f64>, z: &mut GpuVector<f64>) {
        // Fine-level GPU Jacobi pre-smoothing: z = D⁻¹·r
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("amg_jacobi_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.diag_inv_gpu.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: r.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: z.buffer().as_entire_binding() },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("amg_jacobi_enc") });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("amg_jacobi_cp"), timestamp_writes: None });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(self.n.div_ceil(256), 1, 1);
        }
        ctx.queue.submit([enc.finish()]);
        ctx.device.poll(wgpu::PollType::wait_indefinitely());

        // Coarse-grid correction via CPU V-cycle.
        // Read the post-smoothing residual, run the CPU hierarchy, write back.
        let r_cpu: Vec<f64> = z.read_to_cpu(ctx);
        let z_cpu = self.cpu_solver.precond_apply(&r_cpu);
        z.write_from_slice(ctx, &z_cpu);
    }
}
