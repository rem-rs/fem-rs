use std::borrow::Cow;
use wgpu::util::DeviceExt;

use crate::context::GpuContext;
use crate::csr::GpuCsrMatrix;

/// GPU-side element input for a P1 triangle.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuElementInput {
    nodes: [f32; 6],
    dofs: [u32; 3],
    _pad: u32,
}

/// GPU-side element input for Tri3 elasticity (3 nodes × 2 interleaved DOFs).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuElementInputElasticTri3 {
    nodes: [f32; 6],
    dofs: [u32; 6],
}

/// GPU-side element input for a P2 triangle (6 nodes, 12 coords + 6 DOFs).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuElementInputTri6 {
    nodes: [f32; 12],
    dofs: [u32; 6],
    _pad0: u32,
    _pad1: u32,
}

/// GPU-side element input for a Q1 quadrilateral (4 nodes, 8 coords + 4 DOFs).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuElementInputQuad4 {
    nodes: [f32; 8],
    dofs: [u32; 4],
}

/// GPU-side element input for a TET4 tetrahedron (4 nodes, 12 coords + 4 DOFs).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuElementInputTet4 {
    nodes: [f32; 12],
    dofs: [u32; 4],
    _pad0: u32,
    _pad1: u32,
}

/// GPU-side element input for a Hex8 hexahedron (8 nodes, 24 coords + 8 DOFs).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuElementInputHex8 {
    nodes: [f32; 24],
    dofs: [u32; 8],
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

/// GPU-side COO triplet.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCooTriplet {
    row: u32,
    col: u32,
    val: f32,
}

fn run_assembly_shader(
    gpu: &GpuContext,
    elem_bytes: &[u8],
    n_elem: usize,
    entries_per_elem: usize,
    shader_wgsl: &str,
) -> Vec<(u32, u32, f32)> {
    let device = &gpu.device;
    let queue = &gpu.queue;

    let elem_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu_assemble_elements"),
        contents: &elem_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let triplet_byte_len = (n_elem * entries_per_elem * size_of::<GpuCooTriplet>()) as u64;
    let coo_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_assemble_coo"),
        size: triplet_byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_assemble_staging"),
        size: triplet_byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let params = [n_elem as u32, 0u32, 0u32, 0u32];
    let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu_assemble_params"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gpu_assemble_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_wgsl)),
    });

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
    encoder.copy_buffer_to_buffer(&coo_buffer, 0, &staging, 0, triplet_byte_len);
    queue.submit(Some(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let triplets: &[GpuCooTriplet] = bytemuck::cast_slice(&data);
    let mut result: Vec<(u32, u32, f32)> = Vec::with_capacity(n_elem * entries_per_elem);
    for t in triplets {
        if t.val != 0.0 {
            result.push((t.row, t.col, t.val));
        }
    }
    drop(data);
    drop(staging);
    result
}

fn triplets_to_gpu_csr_f64(
    gpu: &GpuContext,
    triplets: Vec<(u32, u32, f32)>,
    n: usize,
) -> GpuCsrMatrix<f64> {
    use fem_linalg::CooMatrix;
    let mut coo = CooMatrix::new(n, n);
    for (r, c, v) in triplets {
        if v != 0.0 {
            coo.add(r as usize, c as usize, v as f64);
        }
    }
    let cpu_csr = coo.into_csr();
    GpuCsrMatrix::from_cpu(gpu, &cpu_csr)
}

/// Assemble 2D Poisson stiffness matrix on GPU, returning a GPU-resident f64 CSR.
pub fn assemble_poisson_2d_p1_gpu(
    gpu: &GpuContext,
    elem_nodes: &[f32],
    elem_dofs: &[u32],
    n_elem: usize,
    n_dofs: usize,
) -> GpuCsrMatrix<f64> {
    let triplets = assemble_poisson_2d_p1(gpu, elem_nodes, elem_dofs, n_elem);
    triplets_to_gpu_csr_f64(gpu, triplets, n_dofs)
}

/// Assemble 2D mass matrix on GPU, returning a GPU-resident f64 CSR.
pub fn assemble_mass_2d_tri3_gpu(
    gpu: &GpuContext,
    elem_nodes: &[f32],
    elem_dofs: &[u32],
    n_elem: usize,
    n_dofs: usize,
) -> GpuCsrMatrix<f64> {
    let triplets = assemble_mass_2d_tri3(gpu, elem_nodes, elem_dofs, n_elem);
    triplets_to_gpu_csr_f64(gpu, triplets, n_dofs)
}

/// Assemble 2D elasticity stiffness matrix on GPU, returning a GPU-resident f64 CSR.
pub fn assemble_elasticity_2d_tri3_gpu(
    gpu: &GpuContext,
    elem_nodes: &[f32],
    elem_dofs: &[u32],
    n_elem: usize,
    n_dofs: usize,
    lambda: f32,
    mu: f32,
) -> GpuCsrMatrix<f64> {
    let triplets = assemble_elasticity_2d_tri3(gpu, elem_nodes, elem_dofs, n_elem, lambda, mu);
    triplets_to_gpu_csr_f64(gpu, triplets, n_dofs)
}
pub fn assemble_poisson_2d_p1(
    gpu: &GpuContext,
    elem_nodes: &[f32],
    elem_dofs: &[u32],
    n_elem: usize,
) -> Vec<(u32, u32, f32)> {
    assert_eq!(elem_nodes.len(), n_elem * 6);
    assert_eq!(elem_dofs.len(), n_elem * 3);

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

    run_assembly_shader(
        gpu,
        bytemuck::cast_slice(&inputs),
        n_elem,
        9,
        include_str!("assembly_poisson_tri3.wgsl"),
    )
}

/// Assemble 2D Poisson stiffness matrix on the GPU for P2 triangles (6-node).
pub fn assemble_poisson_2d_p2(
    gpu: &GpuContext,
    elem_nodes: &[f32],
    elem_dofs: &[u32],
    n_elem: usize,
) -> Vec<(u32, u32, f32)> {
    assert_eq!(elem_nodes.len(), n_elem * 12);
    assert_eq!(elem_dofs.len(), n_elem * 6);

    let mut inputs = Vec::with_capacity(n_elem);
    for e in 0..n_elem {
        let nb = e * 12;
        let db = e * 6;
        let mut nodes = [0.0f32; 12];
        nodes.copy_from_slice(&elem_nodes[nb..nb+12]);
        inputs.push(GpuElementInputTri6 {
            nodes,
            dofs: [
                elem_dofs[db], elem_dofs[db+1], elem_dofs[db+2],
                elem_dofs[db+3], elem_dofs[db+4], elem_dofs[db+5],
            ],
            _pad0: 0,
            _pad1: 0,
        });
    }

    run_assembly_shader(
        gpu,
        bytemuck::cast_slice(&inputs),
        n_elem,
        36,
        include_str!("assembly_poisson_tri6.wgsl"),
    )
}

/// Assemble 2D Poisson stiffness matrix on the GPU for Q1 quadrilaterals.
pub fn assemble_poisson_2d_q1(
    gpu: &GpuContext,
    elem_nodes: &[f32],
    elem_dofs: &[u32],
    n_elem: usize,
) -> Vec<(u32, u32, f32)> {
    assert_eq!(elem_nodes.len(), n_elem * 8);
    assert_eq!(elem_dofs.len(), n_elem * 4);

    let mut inputs = Vec::with_capacity(n_elem);
    for e in 0..n_elem {
        let nb = e * 8;
        let db = e * 4;
        let mut nodes = [0.0f32; 8];
        nodes.copy_from_slice(&elem_nodes[nb..nb+8]);
        inputs.push(GpuElementInputQuad4 {
            nodes,
            dofs: [elem_dofs[db], elem_dofs[db+1], elem_dofs[db+2], elem_dofs[db+3]],
        });
    }

    run_assembly_shader(
        gpu,
        bytemuck::cast_slice(&inputs),
        n_elem,
        16,
        include_str!("assembly_poisson_quad4.wgsl"),
    )
}

/// Assemble 3D Poisson stiffness matrix on the GPU for TET4 tetrahedra.
pub fn assemble_poisson_3d_p1(
    gpu: &GpuContext,
    elem_nodes: &[f32],
    elem_dofs: &[u32],
    n_elem: usize,
) -> Vec<(u32, u32, f32)> {
    assert_eq!(elem_nodes.len(), n_elem * 12);
    assert_eq!(elem_dofs.len(), n_elem * 4);

    let mut inputs = Vec::with_capacity(n_elem);
    for e in 0..n_elem {
        let nb = e * 12;
        let db = e * 4;
        let mut nodes = [0.0f32; 12];
        nodes.copy_from_slice(&elem_nodes[nb..nb+12]);
        inputs.push(GpuElementInputTet4 {
            nodes,
            dofs: [elem_dofs[db], elem_dofs[db+1], elem_dofs[db+2], elem_dofs[db+3]],
            _pad0: 0,
            _pad1: 0,
        });
    }

    run_assembly_shader(
        gpu,
        bytemuck::cast_slice(&inputs),
        n_elem,
        16,
        include_str!("assembly_poisson_tet4.wgsl"),
    )
}

/// Assemble 3D Poisson stiffness matrix on the GPU for Hex8 hexahedra.
pub fn assemble_poisson_3d_hex8(
    gpu: &GpuContext,
    elem_nodes: &[f32],
    elem_dofs: &[u32],
    n_elem: usize,
) -> Vec<(u32, u32, f32)> {
    assert_eq!(elem_nodes.len(), n_elem * 24);
    assert_eq!(elem_dofs.len(), n_elem * 8);

    let mut inputs = Vec::with_capacity(n_elem);
    for e in 0..n_elem {
        let nb = e * 24;
        let db = e * 8;
        let mut nodes = [0.0f32; 24];
        nodes.copy_from_slice(&elem_nodes[nb..nb+24]);
        inputs.push(GpuElementInputHex8 {
            nodes,
            dofs: [
                elem_dofs[db], elem_dofs[db+1], elem_dofs[db+2], elem_dofs[db+3],
                elem_dofs[db+4], elem_dofs[db+5], elem_dofs[db+6], elem_dofs[db+7],
            ],
            _pad0: 0, _pad1: 0, _pad2: 0, _pad3: 0,
        });
    }

    run_assembly_shader(
        gpu,
        bytemuck::cast_slice(&inputs),
        n_elem,
        64,
        include_str!("assembly_poisson_hex8.wgsl"),
    )
}

/// Assemble 3D Hex8 mass matrix on the GPU.
pub fn assemble_mass_3d_hex8(
    gpu: &GpuContext,
    elem_nodes: &[f32],
    elem_dofs: &[u32],
    n_elem: usize,
) -> Vec<(u32, u32, f32)> {
    assert_eq!(elem_nodes.len(), n_elem * 24);
    assert_eq!(elem_dofs.len(), n_elem * 8);

    let mut inputs = Vec::with_capacity(n_elem);
    for e in 0..n_elem {
        let nb = e * 24;
        let db = e * 8;
        let mut nodes = [0.0f32; 24];
        nodes.copy_from_slice(&elem_nodes[nb..nb+24]);
        inputs.push(GpuElementInputHex8 {
            nodes,
            dofs: [
                elem_dofs[db], elem_dofs[db+1], elem_dofs[db+2], elem_dofs[db+3],
                elem_dofs[db+4], elem_dofs[db+5], elem_dofs[db+6], elem_dofs[db+7],
            ],
            _pad0: 0, _pad1: 0, _pad2: 0, _pad3: 0,
        });
    }

    run_assembly_shader(
        gpu,
        bytemuck::cast_slice(&inputs),
        n_elem,
        64,
        include_str!("assembly_mass_hex8.wgsl"),
    )
}

/// Assemble 2D linear elasticity stiffness matrix on the GPU for P1 triangles.
///
/// Uses constant-strain triangle (CST) with single-point quadrature.
/// Interleaved DOF layout: [u_x0, u_y0, u_x1, u_y1, u_x2, u_y2].
/// `lambda` and `mu` are the Lamé parameters.
pub fn assemble_elasticity_2d_tri3(
    gpu: &GpuContext,
    elem_nodes: &[f32],
    elem_dofs: &[u32],
    n_elem: usize,
    lambda: f32,
    mu: f32,
) -> Vec<(u32, u32, f32)> {
    assert_eq!(elem_nodes.len(), n_elem * 6);
    assert_eq!(elem_dofs.len(), n_elem * 6);

    let mut inputs = Vec::with_capacity(n_elem);
    for e in 0..n_elem {
        let nb = e * 6;
        let db = e * 6;
        let mut nodes = [0.0f32; 6];
        nodes.copy_from_slice(&elem_nodes[nb..nb+6]);
        inputs.push(GpuElementInputElasticTri3 {
            nodes,
            dofs: [
                elem_dofs[db], elem_dofs[db+1], elem_dofs[db+2],
                elem_dofs[db+3], elem_dofs[db+4], elem_dofs[db+5],
            ],
        });
    }

    let device = &gpu.device;
    let queue = &gpu.queue;

    let elem_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu_assemble_elasticity_elements"),
        contents: bytemuck::cast_slice(&inputs),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let entries_per_elem = 36;
    let triplet_byte_len = (n_elem * entries_per_elem * size_of::<GpuCooTriplet>()) as u64;
    let coo_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_assemble_elasticity_coo"),
        size: triplet_byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_assemble_elasticity_staging"),
        size: triplet_byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Params: [n_elements (u32), lambda (f32), mu (f32)]
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct ElasticityParams {
        n_elems: u32,
        lambda: f32,
        mu: f32,
    }
    let params = ElasticityParams { n_elems: n_elem as u32, lambda, mu };
    let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu_assemble_elasticity_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gpu_assemble_elasticity_shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
            include_str!("assembly_elasticity_tri3.wgsl"),
        )),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gpu_assemble_elasticity_bgl"),
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
        label: Some("gpu_assemble_elasticity_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: elem_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: coo_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: param_buffer.as_entire_binding() },
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gpu_assemble_elasticity_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gpu_assemble_elasticity_pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu_assemble_elasticity_enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_assemble_elasticity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let wg = ((n_elem as u32) + 63) / 64;
        pass.dispatch_workgroups(wg, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&coo_buffer, 0, &staging, 0, triplet_byte_len);
    queue.submit(Some(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let triplets: &[GpuCooTriplet] = bytemuck::cast_slice(&data);
    let mut result: Vec<(u32, u32, f32)> = Vec::with_capacity(n_elem * entries_per_elem);
    result.extend(triplets.iter().map(|t| (t.row, t.col, t.val)));
    drop(data);
    drop(staging);
    result
}

/// Assemble 2D mass matrix on the GPU for P1 triangles.
pub fn assemble_mass_2d_tri3(
    gpu: &GpuContext,
    elem_nodes: &[f32],
    elem_dofs: &[u32],
    n_elem: usize,
) -> Vec<(u32, u32, f32)> {
    assert_eq!(elem_nodes.len(), n_elem * 6);
    assert_eq!(elem_dofs.len(), n_elem * 3);
    let mut inputs = Vec::with_capacity(n_elem);
    for e in 0..n_elem {
        let nb = e * 6; let db = e * 3;
        inputs.push(GpuElementInput { nodes: [
            elem_nodes[nb], elem_nodes[nb+1], elem_nodes[nb+2],
            elem_nodes[nb+3], elem_nodes[nb+4], elem_nodes[nb+5],
        ], dofs: [elem_dofs[db], elem_dofs[db+1], elem_dofs[db+2]], _pad: 0 });
    }
    run_assembly_shader(gpu, bytemuck::cast_slice(&inputs), n_elem, 9, include_str!("assembly_mass_tri3.wgsl"))
}

/// Assemble 2D mass matrix on the GPU for Q1 quadrilaterals.
pub fn assemble_mass_2d_quad4(
    gpu: &GpuContext,
    elem_nodes: &[f32],
    elem_dofs: &[u32],
    n_elem: usize,
) -> Vec<(u32, u32, f32)> {
    assert_eq!(elem_nodes.len(), n_elem * 8);
    assert_eq!(elem_dofs.len(), n_elem * 4);
    let mut inputs = Vec::with_capacity(n_elem);
    for e in 0..n_elem {
        let nb = e * 8; let db = e * 4;
        let mut nodes = [0.0f32; 8];
        nodes.copy_from_slice(&elem_nodes[nb..nb+8]);
        inputs.push(GpuElementInputQuad4 { nodes,
            dofs: [elem_dofs[db], elem_dofs[db+1], elem_dofs[db+2], elem_dofs[db+3]] });
    }
    run_assembly_shader(gpu, bytemuck::cast_slice(&inputs), n_elem, 16, include_str!("assembly_mass_quad4.wgsl"))
}

/// Assemble 3D mass matrix on the GPU for TET4 tetrahedra.
pub fn assemble_mass_3d_tet4(
    gpu: &GpuContext,
    elem_nodes: &[f32],
    elem_dofs: &[u32],
    n_elem: usize,
) -> Vec<(u32, u32, f32)> {
    assert_eq!(elem_nodes.len(), n_elem * 12);
    assert_eq!(elem_dofs.len(), n_elem * 4);
    let mut inputs = Vec::with_capacity(n_elem);
    for e in 0..n_elem {
        let nb = e * 12; let db = e * 4;
        let mut nodes = [0.0f32; 12];
        nodes.copy_from_slice(&elem_nodes[nb..nb+12]);
        inputs.push(GpuElementInputTet4 { nodes,
            dofs: [elem_dofs[db], elem_dofs[db+1], elem_dofs[db+2], elem_dofs[db+3]],
            _pad0: 0, _pad1: 0 });
    }
    run_assembly_shader(gpu, bytemuck::cast_slice(&inputs), n_elem, 16, include_str!("assembly_mass_tet4.wgsl"))
}

// ─── Tet4 elasticity assembly ─────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuElementInputElasticTet4 {
    nodes: [f32; 12],
    dofs: [u32; 12],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ElasticityParams {
    n_elems: u32,
    lambda: f32,
    mu: f32,
}

fn run_elasticity_shader(
    gpu: &GpuContext,
    elem_bytes: &[u8],
    n_elem: usize,
    entries_per_elem: usize,
    lambda: f32,
    mu: f32,
    shader_wgsl: &str,
) -> Vec<(u32, u32, f32)> {
    let device = &gpu.device;
    let queue = &gpu.queue;

    let elem_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu_elasticity_elements"),
        contents: &elem_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let triplet_byte_len = (n_elem * entries_per_elem * size_of::<GpuCooTriplet>()) as u64;
    let coo_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_elasticity_coo"), size: triplet_byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_elasticity_staging"), size: triplet_byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let params = ElasticityParams { n_elems: n_elem as u32, lambda, mu };
    let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu_elasticity_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gpu_elasticity_shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_wgsl)),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gpu_elasticity_bgl"),
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
        label: Some("gpu_elasticity_bg"), layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: elem_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: coo_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: param_buffer.as_entire_binding() },
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gpu_elasticity_pl"), bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gpu_elasticity_pipeline"), layout: Some(&pl),
        module: &shader, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gpu_elasticity_enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_elasticity_pass"), timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(((n_elem as u32) + 63) / 64, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&coo_buffer, 0, &staging, 0, triplet_byte_len);
    queue.submit(Some(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let triplets: &[GpuCooTriplet] = bytemuck::cast_slice(&data);
    let mut result: Vec<(u32, u32, f32)> = Vec::with_capacity(n_elem * entries_per_elem);
    result.extend(triplets.iter().map(|t| (t.row, t.col, t.val)));
    drop(data); drop(staging);
    result
}

/// Assemble 3D Tet4 elasticity stiffness matrix on GPU (constant strain tetrahedron).
pub fn assemble_elasticity_3d_tet4(
    gpu: &GpuContext,
    elem_nodes: &[f32],
    elem_dofs: &[u32],
    n_elem: usize,
    lambda: f32,
    mu: f32,
) -> Vec<(u32, u32, f32)> {
    assert_eq!(elem_nodes.len(), n_elem * 12);
    assert_eq!(elem_dofs.len(), n_elem * 12);
    let mut inputs = Vec::with_capacity(n_elem);
    for e in 0..n_elem {
        let nb = e * 12; let db = e * 12;
        let mut nodes = [0.0f32; 12]; nodes.copy_from_slice(&elem_nodes[nb..nb+12]);
        inputs.push(GpuElementInputElasticTet4 { nodes,
            dofs: [elem_dofs[db], elem_dofs[db+1], elem_dofs[db+2], elem_dofs[db+3],
                   elem_dofs[db+4], elem_dofs[db+5], elem_dofs[db+6], elem_dofs[db+7],
                   elem_dofs[db+8], elem_dofs[db+9], elem_dofs[db+10], elem_dofs[db+11]] });
    }
    run_elasticity_shader(gpu, bytemuck::cast_slice(&inputs), n_elem, 144, lambda, mu,
        include_str!("assembly_elasticity_tet4.wgsl"))
}

/// Return GpuCsrMatrix<f64> from Tet4 GPU elasticity.
pub fn assemble_elasticity_3d_tet4_gpu(
    gpu: &GpuContext,
    elem_nodes: &[f32],
    elem_dofs: &[u32],
    n_elem: usize,
    n_dofs: usize,
    lambda: f32,
    mu: f32,
) -> GpuCsrMatrix<f64> {
    let triplets = assemble_elasticity_3d_tet4(gpu, elem_nodes, elem_dofs, n_elem, lambda, mu);
    triplets_to_gpu_csr_f64(gpu, triplets, n_dofs)
}

/// Assemble 3D Hex8 elasticity stiffness matrix on GPU (8-point Gauss quadrature).
pub fn assemble_elasticity_3d_hex8(
    gpu: &GpuContext,
    elem_nodes: &[f32],
    elem_dofs: &[u32],
    n_elem: usize,
    lambda: f32,
    mu: f32,
) -> Vec<(u32, u32, f32)> {
    assert_eq!(elem_nodes.len(), n_elem * 24);
    assert_eq!(elem_dofs.len(), n_elem * 24);
    // Build raw bytes manually since the struct needs 24 nodes + 24 dofs
    let bytes_per_elem = 24 * 4 + 24 * 4; // f32[24] + u32[24]
    let mut raw = Vec::<u8>::with_capacity(n_elem * bytes_per_elem);
    for e in 0..n_elem {
        let nb = e * 24; let db = e * 24;
        raw.extend_from_slice(bytemuck::cast_slice(&elem_nodes[nb..nb+24]));
        raw.extend_from_slice(bytemuck::cast_slice(&elem_dofs[db..db+24]));
    }
    run_elasticity_shader(gpu, &raw, n_elem, 576, lambda, mu,
        include_str!("assembly_elasticity_hex8.wgsl"))
}

/// Return GpuCsrMatrix<f64> from Hex8 GPU elasticity.
pub fn assemble_elasticity_3d_hex8_gpu(
    gpu: &GpuContext,
    elem_nodes: &[f32],
    elem_dofs: &[u32],
    n_elem: usize,
    n_dofs: usize,
    lambda: f32,
    mu: f32,
) -> GpuCsrMatrix<f64> {
    let triplets = assemble_elasticity_3d_hex8(gpu, elem_nodes, elem_dofs, n_elem, lambda, mu);
    triplets_to_gpu_csr_f64(gpu, triplets, n_dofs)
}
