#![cfg(feature = "gpu")]

use fem_linalg_gpu::{GpuContext, assemble_elasticity_2d_tri3};
use fem_linalg::CooMatrix;
use fem_mesh::{SimplexMesh, MeshTopology};
use fem_space::{VectorH1Space, FESpace};
use fem_assembly::standard::ElasticityIntegrator;
use fem_assembly::Assembler;

/// Build a unit-square mesh with 2 triangles and extract node coordinates
/// and DOFs in the format expected by the GPU assembly function.
fn build_tri3_mesh_data(
) -> (Vec<f32>, Vec<u32>, SimplexMesh<2>, VectorH1Space<SimplexMesh<2>>) {
    let mesh = SimplexMesh::<2>::unit_square_tri(2);
    let space = VectorH1Space::new(mesh.clone(), 1, 2);

    let n_elem = mesh.n_elements();
    let mut elem_nodes_f32 = Vec::with_capacity(n_elem * 6);
    let mut elem_dofs_u32 = Vec::with_capacity(n_elem * 6);

    for e in 0..n_elem as u32 {
        let nodes = mesh.element_nodes(e);
        let dofs: Vec<u32> = space.element_dofs(e).iter().map(|&d| d as u32).collect();
        for &nid in nodes {
            let c = mesh.node_coords(nid);
            elem_nodes_f32.push(c[0] as f32);
            elem_nodes_f32.push(c[1] as f32);
        }
        elem_dofs_u32.extend_from_slice(&dofs);
    }

    (elem_nodes_f32, elem_dofs_u32, mesh, space)
}

#[test]
fn elasticity_gpu_matches_cpu() {
    let gpu = GpuContext::new_sync().expect("GPU context");

    let (elem_nodes, elem_dofs, mesh, space) = build_tri3_mesh_data();
    let n_elem = mesh.n_elements();

    // GPU assembly
    let lambda = 1.0f32;
    let mu = 1.0f32;
    let gpu_triplets = assemble_elasticity_2d_tri3(
        &gpu, &elem_nodes, &elem_dofs, n_elem, lambda, mu,
    );

    let n_dofs = space.n_dofs();
    let mut gpu_coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
    for (r, c, v) in &gpu_triplets {
        gpu_coo.add(*r as usize, *c as usize, *v as f64);
    }
    let gpu_mat = gpu_coo.into_csr();

    // CPU assembly
    let cpu_integ = ElasticityIntegrator { lambda: lambda as f64, mu: mu as f64 };
    let cpu_mat = Assembler::assemble_bilinear(&space, &[&cpu_integ], 3);

    assert_eq!(gpu_mat.nrows, cpu_mat.nrows);
    assert_eq!(gpu_mat.nnz(), cpu_mat.nnz());

    let mut max_diff = 0.0f64;
    let mut max_rel = 0.0f64;
    for i in 0..cpu_mat.nrows {
        let start = cpu_mat.row_ptr[i];
        let end = cpu_mat.row_ptr[i + 1];
        for k in start..end {
            let col = cpu_mat.col_idx[k] as usize;
            let cpu_val = cpu_mat.values[k];
            let gpu_val = gpu_mat.get(i, col);
            let diff = (cpu_val - gpu_val).abs();
            max_diff = max_diff.max(diff);
            let denom = cpu_val.abs().max(1e-15);
            max_rel = max_rel.max(diff / denom);
        }
    }
    eprintln!("elasticity max_abs_diff={:.2e} max_rel_diff={:.2e}", max_diff, max_rel);
    assert!(max_rel < 5e-3, "relative error too large: {:.2e}", max_rel);
}
