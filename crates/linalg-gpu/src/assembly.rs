use std::borrow::Cow;
use wgpu::util::DeviceExt;

use crate::context::GpuContext;

/// GPU-side element input for a P1 triangle.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuElementInput {
    nodes: [f32; 6],
    dofs: [u32; 3],
    _pad: u32,
}

/// GPU-side COO triplet.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCooTriplet {
    row: u32,
    col: u32,
    val: f32,
}

/// Assemble 2D Poisson stiffness matrix on the GPU for P1 triangles.
///
/// `elem_nodes`: flat f32 array [x0,y0,x1,y1,x2,y2] per element
/// `elem_dofs`: flat u32 array [dof0,dof1,dof2] per element  
/// `n_elem`: number of triangular elements
/// `n_dofs`: global DOF count
///
/// Returns a Vec of (row, col, value) COO triplets.
pub fn assemble_poisson_2d_p1(
    gpu: &GpuContext,
    elem_nodes: &[f32],
    elem_dofs: &[u32],
    n_elem: usize,
) -> Vec<(u32, u32, f32)> {
    assert_eq!(elem_nodes.len(), n_elem * 6);
    assert_eq!(elem_dofs.len(), n_elem * 3);

    // Pack element input data
    let mut inputs = Vec::with_capacity(n_elem);
    for e in 0..n_elem {
        let nb = e * 6;
        let db = e * 3;
        inputs.push(GpuElementInput {
            nodes: [
                elem_nodes[nb], elem_nodes[nb+1], elem_nodes[nb+2],
                elem_nodes[nb+3], elem_nodes[nb+4], elem_nodes[nb+5],
            ],
            dofs: [elem_dofs[db], elem_dofs[db+1], elem_dofs[db+2]],
            _pad: 0,
        });
    }

    let device = &gpu.device;
    let queue = &gpu.queue;

    // Element input buffer
    let elem_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu_assemble_elements"),
        contents: bytemuck::cast_slice(&inputs),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // COO output buffer (9 entries per element, one workgroup per element)
    let coo_byte_len = (n_elem * 9 * size_of::<GpuCooTriplet>()) as u64;
    let coo_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_assemble_coo"),
        size: coo_byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Staging for readback
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_assemble_staging"),
        size: coo_byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Uniform parameters
    let params = [n_elem as u32, 0u32, 0u32, 0u32];
    let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu_assemble_params"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Shader
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gpu_assemble_poisson_tri3"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
            include_str!("assembly_poisson_tri3.wgsl"),
        )),
    });

    // Bind group layout
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gpu_assemble_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None,
                }, count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None,
                }, count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None,
                }, count: None,
            },
        ],
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu_assemble_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: elem_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: coo_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: param_buffer.as_entire_binding() },
        ],
    });

    // Pipeline
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gpu_assemble_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gpu_assemble_pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Dispatch
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu_assemble_enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_assemble_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let wg = ((n_elem as u32) + 63) / 64;
        pass.dispatch_workgroups(wg, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&coo_buffer, 0, &staging, 0, coo_byte_len);
    queue.submit(Some(encoder.finish()));

    // Readback
    let (tx, rx) = std::sync::mpsc::channel();
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let triplets: &[GpuCooTriplet] = bytemuck::cast_slice(&data);
    let mut result: Vec<(u32, u32, f32)> = Vec::with_capacity(n_elem * 9);
    for t in triplets {
        if t.val != 0.0 {
            result.push((t.row, t.col, t.val));
        }
    }
    drop(data);
    drop(staging);
    result
}
