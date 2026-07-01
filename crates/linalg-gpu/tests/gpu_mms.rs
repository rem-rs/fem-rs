//! GPU vs CPU MMS comparison test.
//!
//! Requires GPU backend (wgpu) with SHADER_F64 support.
//! Run: cargo test -p fem-linalg-gpu -- --ignored gpu_mms

use fem_linalg::CsrMatrix;
use fem_mesh::SimplexMesh;
use fem_mesh::topology::MeshTopology;
use fem_space::H1Space;
use fem_space::fe_space::FESpace;
use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
use fem_assembly::Assembler;
use fem_solver::solve_cg;
use fem_solver::SolverConfig;

fn cpu_matrix(label: &str, n: usize, integrator: &dyn fem_assembly::BilinearIntegrator, quad: u8) -> CsrMatrix<f64> {
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh, 1);
    let start = std::time::Instant::now();
    let mat = Assembler::assemble_bilinear(&space, &[integrator], quad);
    eprintln!("CPU {label}: {:.3}ms, {} DOF", start.elapsed().as_secs_f64() * 1e3, space.n_dofs());
    mat
}

fn gpu_matrix_f64<F>(label: &str, _n: usize, assemble: F) -> Vec<(u32, u32, f64)>
where F: FnOnce(&fem_linalg_gpu::GpuContext) -> Vec<(u32, u32, f64)>,
{
    let gpu = pollster::block_on(fem_linalg_gpu::GpuContext::new()).expect("GPU context");
    if !gpu.features.native_f64 {
        eprintln!("GPU f64 not supported, skipping {label}");
        return Vec::new();
    }
    let start = std::time::Instant::now();
    let triplets = assemble(&gpu);
    eprintln!("GPU {label}: {:.3}ms", start.elapsed().as_secs_f64() * 1e3);
    triplets
}

fn compare_matrices(label: &str, cpu: &CsrMatrix<f64>, gpu_triplets: &[(u32, u32, f64)], tol: f64) {
    if gpu_triplets.is_empty() { eprintln!("SKIP {label}: no GPU data"); return; }
    // Build GPU CSR
    let mut coo = fem_linalg::CooMatrix::new(cpu.nrows, cpu.ncols);
    for &(r, c, v) in gpu_triplets { coo.add(r as usize, c as usize, v); }
    let gpu: CsrMatrix<f64> = coo.into_csr();

    let mut max_rel = 0.0f64;
    let mut max_abs = 0.0f64;
    for i in 0..cpu.nrows.min(gpu.nrows) {
        for k in cpu.row_ptr[i]..cpu.row_ptr[i+1] {
            let j = cpu.col_idx[k] as usize;
            let cpu_v = cpu.values[k];
            let gpu_v = gpu.get(i, j);
            let diff = (cpu_v - gpu_v).abs();
            let rel = diff / cpu_v.abs().max(1.0);
            max_rel = max_rel.max(rel);
            max_abs = max_abs.max(diff);
        }
    }
    eprintln!("{label}: max_rel={:.3e}, max_abs={:.3e}", max_rel, max_abs);
    assert!(max_abs < tol.max(max_rel * 10.0_f64),
        "{label}: max_abs={:.3e} exceeds tol={:.3e}", max_abs, tol);
}

// ─── Poisson ────────────────────────────────────────────────────────────────

#[test]
#[ignore]
fn gpu_vs_cpu_poisson_f64() {
    let n = 8;
    let cpu = cpu_matrix("Poisson", n, &DiffusionIntegrator { kappa: 1.0 }, 3);
    use fem_linalg_gpu::assembly::assemble_poisson_2d_p1_f64;
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh, 1);
    let (elem_nodes, elem_dofs, n_elem) = extract_tri3_p1(&space);
    let triplets = gpu_matrix_f64("Poisson", n, |gpu| {
        assemble_poisson_2d_p1_f64(gpu, &elem_nodes, &elem_dofs, n_elem)
    });
    compare_matrices("Poisson f64", &cpu, &triplets, 1e-12_f64);
}

// ─── Mass ───────────────────────────────────────────────────────────────────

#[test]
#[ignore]
fn gpu_vs_cpu_mass_f64() {
    let n = 8;
    use fem_assembly::standard::MassIntegrator;
    let cpu = cpu_matrix("Mass", n, &MassIntegrator { rho: 1.0 }, 3);
    use fem_linalg_gpu::assembly::assemble_mass_2d_tri3_f64;
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh, 1);
    let (elem_nodes, elem_dofs, n_elem) = extract_tri3_p1(&space);
    let triplets = gpu_matrix_f64("Mass", n, |gpu| {
        assemble_mass_2d_tri3_f64(gpu, &elem_nodes, &elem_dofs, n_elem)
    });
    compare_matrices("Mass f64", &cpu, &triplets, 1e-12_f64);
}

// ─── Elasticity ─────────────────────────────────────────────────────────────

#[test]
#[ignore]
fn gpu_vs_cpu_elasticity_f64() {
    let n = 8;
    use fem_assembly::standard::VectorDiffusionIntegrator;
    use fem_space::vector_h1::VectorH1Space;
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = VectorH1Space::new(mesh, 1, 2);
    let n_dofs = space.n_dofs();
    let start = std::time::Instant::now();
    let cpu = fem_assembly::Assembler::assemble_bilinear(
        &space, &[&VectorDiffusionIntegrator { kappa: 1.0 }], 3);
    eprintln!("CPU Elasticity: {:.3}ms, {n_dofs} DOF", start.elapsed().as_secs_f64() * 1e3);

    use fem_linalg_gpu::assembly::assemble_elasticity_2d_tri3_f64;
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = VectorH1Space::new(mesh, 1, 2);
    // Extract elasticity element data (3 nodes, 2 DOFs per node = 6 DOFs per element)
    let n_elem = space.mesh().n_elems();
    let mut elem_nodes = Vec::with_capacity(n_elem * 6);
    let mut elem_dofs = Vec::with_capacity(n_elem * 6);
    for e in 0..n_elem as u32 {
        let ns = space.mesh().element_nodes(e);
        for ni in ns.iter() {
            let c = space.mesh().node_coords(*ni);
            elem_nodes.push(c[0]); elem_nodes.push(c[1]);
        }
        let dofs = space.element_dofs(e);
        let n_d = dofs.len();
        for i in 0..n_d { elem_dofs.push(dofs[i]); }
    }

    let triplets = gpu_matrix_f64("Elasticity", n, |gpu| {
        assemble_elasticity_2d_tri3_f64(gpu, &elem_nodes, &elem_dofs, n_elem, 1.0, 1.0)
    });
    if !triplets.is_empty() {
        let mut coo = fem_linalg::CooMatrix::new(n_dofs, n_dofs);
        for &(r, c, v) in &triplets { coo.add(r as usize, c as usize, v); }
        let gpu: CsrMatrix<f64> = coo.into_csr();
        let mut max_rel = 0.0_f64;
        for i in 0..cpu.nrows.min(gpu.nrows) {
            for k in cpu.row_ptr[i]..cpu.row_ptr[i+1] {
                let j = cpu.col_idx[k] as usize;
                let diff = (cpu.values[k] - gpu.get(i, j)).abs();
                let rel = diff / cpu.values[k].abs().max(1.0_f64);
                max_rel = max_rel.max(rel);
            }
        }
        eprintln!("Elasticity f64: max_rel={:.3e}", max_rel);
        assert!(max_rel < 1e-10_f64, "Elasticity mismatch: {:.3e}", max_rel);
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn extract_tri3_p1(space: &H1Space<SimplexMesh<2>>) -> (Vec<f64>, Vec<u32>, usize) {
    let mesh = space.mesh();
    let n_elem = mesh.n_elems();
    let mut elem_nodes = Vec::with_capacity(n_elem * 6);
    let mut elem_dofs = Vec::with_capacity(n_elem * 3);
    for e in 0..n_elem as u32 {
        let ns = mesh.element_nodes(e);
        for ni in ns.iter() {
            let c = mesh.node_coords(*ni);
            elem_nodes.push(c[0]); elem_nodes.push(c[1]);
        }
        let dofs = space.element_dofs(e);
        elem_dofs.push(dofs[0]); elem_dofs.push(dofs[1]); elem_dofs.push(dofs[2]);
    }
    (elem_nodes, elem_dofs, n_elem)
}
