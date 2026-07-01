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
use fem_space::constraints::boundary_dofs;

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

/// Extract Tet4 P1 element data: node coords (12 per element) + DOF ids (4 per element).
fn extract_tet4_p1(space: &H1Space<SimplexMesh<3>>) -> (Vec<f64>, Vec<u32>, usize) {
    let mesh = space.mesh();
    let n_elem = mesh.n_elems();
    let mut elem_nodes = Vec::with_capacity(n_elem * 12);
    let mut elem_dofs = Vec::with_capacity(n_elem * 4);
    for e in 0..n_elem as u32 {
        let ns = mesh.element_nodes(e);
        for ni in ns.iter() {
            let c = mesh.node_coords(*ni);
            elem_nodes.push(c[0]); elem_nodes.push(c[1]); elem_nodes.push(c[2]);
        }
        let dofs = space.element_dofs(e);
        for i in 0..dofs.len() { elem_dofs.push(dofs[i]); }
    }
    (elem_nodes, elem_dofs, n_elem)
}

// ─── Tet4 3D tests ─────────────────────────────────────────────────────────

#[test]
#[ignore]
fn gpu_vs_cpu_poisson_tet4_f64() {
    let n = 4;
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;
    use fem_assembly::standard::DiffusionIntegrator;
    use fem_assembly::Assembler;
    let mesh = SimplexMesh::<3>::unit_cube_tet(n);
    let space = H1Space::new(mesh, 1);
    let cpu = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
    eprintln!("CPU Poisson Tet4: {} DOF", space.n_dofs());

    let mesh = SimplexMesh::<3>::unit_cube_tet(n);
    let space = H1Space::new(mesh, 1);
    let (elem_nodes, elem_dofs, n_elem) = extract_tet4_p1(&space);

    use fem_linalg_gpu::assembly::assemble_poisson_3d_tet4_f64;
    let gpu = pollster::block_on(fem_linalg_gpu::GpuContext::new()).expect("GPU context");
    if !gpu.features.native_f64 { eprintln!("SKIP: no SHADER_F64"); return; }

    let triplets = assemble_poisson_3d_tet4_f64(&gpu, &elem_nodes, &elem_dofs, n_elem);
    let mut coo = fem_linalg::CooMatrix::new(cpu.nrows, cpu.ncols);
    for &(r,c,v) in &triplets { coo.add(r as usize, c as usize, v); }
    let gpu_mat: fem_linalg::CsrMatrix<f64> = coo.into_csr();

    let mut max_rel = 0.0_f64;
    for i in 0..cpu.nrows.min(gpu_mat.nrows) {
        for k in cpu.row_ptr[i]..cpu.row_ptr[i+1] {
            let j = cpu.col_idx[k] as usize;
            let diff = (cpu.values[k] - gpu_mat.get(i, j)).abs();
            let rel = diff / cpu.values[k].abs().max(1.0_f64);
            max_rel = max_rel.max(rel);
        }
    }
    eprintln!("Poisson Tet4 f64: max_rel={:.3e}", max_rel);
    assert!(max_rel < 1e-12_f64, "Tet4 Poisson mismatch: {:.3e}", max_rel);
}

#[test]
#[ignore]
fn gpu_vs_cpu_mass_tet4_f64() {
    let n = 4;
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;
    use fem_assembly::standard::MassIntegrator;
    use fem_assembly::Assembler;
    let mesh = SimplexMesh::<3>::unit_cube_tet(n);
    let space = H1Space::new(mesh, 1);
    let cpu = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 3);
    eprintln!("CPU Mass Tet4: {} DOF", space.n_dofs());

    let mesh = SimplexMesh::<3>::unit_cube_tet(n);
    let space = H1Space::new(mesh, 1);
    let (elem_nodes, elem_dofs, n_elem) = extract_tet4_p1(&space);

    use fem_linalg_gpu::assembly::assemble_mass_3d_tet4_f64;
    let gpu = pollster::block_on(fem_linalg_gpu::GpuContext::new()).expect("GPU context");
    if !gpu.features.native_f64 { eprintln!("SKIP: no SHADER_F64"); return; }

    let triplets = assemble_mass_3d_tet4_f64(&gpu, &elem_nodes, &elem_dofs, n_elem);
    let mut coo = fem_linalg::CooMatrix::new(cpu.nrows, cpu.ncols);
    for &(r,c,v) in &triplets { coo.add(r as usize, c as usize, v); }
    let gpu_mat: fem_linalg::CsrMatrix<f64> = coo.into_csr();

    let mut max_rel = 0.0_f64;
    for i in 0..cpu.nrows.min(gpu_mat.nrows) {
        for k in cpu.row_ptr[i]..cpu.row_ptr[i+1] {
            let j = cpu.col_idx[k] as usize;
            let diff = (cpu.values[k] - gpu_mat.get(i, j)).abs();
            let rel = diff / cpu.values[k].abs().max(1.0_f64);
            max_rel = max_rel.max(rel);
        }
    }
    eprintln!("Mass Tet4 f64: max_rel={:.3e}", max_rel);
    assert!(max_rel < 1e-12_f64, "Tet4 Mass mismatch: {:.3e}", max_rel);
}

#[test]
#[ignore]
fn gpu_vs_cpu_elasticity_tet4_f64() {
    let n = 3;
    use fem_mesh::SimplexMesh;
    use fem_space::vector_h1::VectorH1Space;
    use fem_assembly::standard::VectorDiffusionIntegrator;
    use fem_assembly::Assembler;
    let mesh = SimplexMesh::<3>::unit_cube_tet(n);
    let space = VectorH1Space::new(mesh, 1, 3);
    let cpu = Assembler::assemble_bilinear(&space, &[&VectorDiffusionIntegrator { kappa: 1.0 }], 3);
    let n_dofs = cpu.nrows;
    eprintln!("CPU Elasticity Tet4: {} DOF", n_dofs);

    let mesh = SimplexMesh::<3>::unit_cube_tet(n);
    let space = VectorH1Space::new(mesh, 1, 3);
    let n_elem = space.mesh().n_elems();
    let mut elem_nodes = Vec::with_capacity(n_elem * 12);
    let mut elem_dofs = Vec::with_capacity(n_elem * 12);
    for e in 0..n_elem as u32 {
        let ns = space.mesh().element_nodes(e);
        for ni in ns.iter() {
            let c = space.mesh().node_coords(*ni);
            elem_nodes.push(c[0]); elem_nodes.push(c[1]); elem_nodes.push(c[2]);
        }
        let dofs = space.element_dofs(e);
        for i in 0..dofs.len() { elem_dofs.push(dofs[i]); }
    }

    use fem_linalg_gpu::assembly::assemble_elasticity_3d_tet4_f64;
    let gpu = pollster::block_on(fem_linalg_gpu::GpuContext::new()).expect("GPU context");
    if !gpu.features.native_f64 { eprintln!("SKIP: no SHADER_F64"); return; }

    let triplets = assemble_elasticity_3d_tet4_f64(&gpu, &elem_nodes, &elem_dofs, n_elem, 1.0, 1.0);
    let mut coo = fem_linalg::CooMatrix::new(n_dofs, n_dofs);
    for &(r,c,v) in &triplets { coo.add(r as usize, c as usize, v); }
    let gpu_mat: fem_linalg::CsrMatrix<f64> = coo.into_csr();

    let mut max_rel = 0.0_f64;
    for i in 0..cpu.nrows.min(gpu_mat.nrows) {
        for k in cpu.row_ptr[i]..cpu.row_ptr[i+1] {
            let j = cpu.col_idx[k] as usize;
            let diff = (cpu.values[k] - gpu_mat.get(i, j)).abs();
            let rel = diff / cpu.values[k].abs().max(1.0_f64);
            max_rel = max_rel.max(rel);
        }
    }
    eprintln!("Elasticity Tet4 f64: max_rel={:.3e}", max_rel);
    assert!(max_rel < 1e-10_f64, "Tet4 Elasticity mismatch: {:.3e}", max_rel);
}

// ─── End-to-end: GPU assembly → GPU solve vs CPU assembly → CPU solve ───────

#[test]
#[ignore]
fn e2e_gpu_assemble_solve_poisson_2d() {
    let n = 8;
    // ── CPU path ───────────────────────────────────────────────────────────
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh.clone(), 1);
    let dm = fem_space::DofManager::new(&mesh, 1);
    let a_cpu = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
    let f = |x: &[f64]| 2.0 * std::f64::consts::PI.powi(2)
        * (std::f64::consts::PI * x[0]).sin() * (std::f64::consts::PI * x[1]).sin();
    let mut rhs_cpu = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(f)], 3);
    let bdofs: Vec<usize> = boundary_dofs(&mesh, &dm, &[1, 2, 3, 4]).iter()
        .map(|&d| d as usize).collect();
    let mut a_cpu_mut = a_cpu;
    fem_space::constraints::apply_dirichlet(&mut a_cpu_mut, &mut rhs_cpu, &bdofs.iter().map(|&d| d as u32).collect::<Vec<_>>(), &vec![0.0; bdofs.len()]);
    let mut u_cpu = vec![0.0; space.n_dofs()];
    solve_cg(&a_cpu_mut, &rhs_cpu, &mut u_cpu, &SolverConfig { rtol: 1e-8, ..Default::default() })
        .expect("CPU CG solve");

    // ── GPU path ───────────────────────────────────────────────────────────
    let gpu = pollster::block_on(fem_linalg_gpu::GpuContext::new()).expect("GPU context");
    if !gpu.features.native_f64 { eprintln!("SKIP: no SHADER_F64"); return; }

    // 1. GPU assembly via f64 shader
    let mesh_g = SimplexMesh::<2>::unit_square_tri(n);
    let space_g = H1Space::new(mesh_g, 1);
    let (elem_nodes, elem_dofs, n_elem) = extract_tri3_p1(&space_g);

    use fem_linalg_gpu::assembly::assemble_poisson_2d_p1_f64;
    let triplets = assemble_poisson_2d_p1_f64(&gpu, &elem_nodes, &elem_dofs, n_elem);

    let n_dofs = space_g.n_dofs() as u32;
    let mut coo_gpu = fem_linalg::CooMatrix::new(n_dofs as usize, n_dofs as usize);
    for &(r, c, v) in &triplets { if v != 0.0 { coo_gpu.add(r as usize, c as usize, v); } }
    let csr_gpu: CsrMatrix<f64> = coo_gpu.into_csr();

    // Apply Dirichlet BCs
    let mut a_bc = csr_gpu;
    let mut rhs_gpu = vec![0.0; n_dofs as usize];
    for &d in &bdofs { a_bc.apply_dirichlet_row_zeroing(d, 0.0, &mut rhs_gpu); rhs_gpu[d] = 0.0; }

    use fem_linalg_gpu::{GpuVector, GpuCsrMatrix, SpmvPipeline, VectorOpsPipeline, solve_cg_gpu};
    let spmv = SpmvPipeline::new(&gpu.device, true);
    let vops = VectorOpsPipeline::new(&gpu.device, true);
    let gpu_mat_bc = GpuCsrMatrix::<f64>::from_cpu(&gpu, &a_bc);
    let b_gpu = GpuVector::<f64>::from_slice(&gpu, &rhs_gpu);
    let mut x_gpu = GpuVector::<f64>::zeros(&gpu, n_dofs);

    let result = solve_cg_gpu::<f64>(&gpu, &spmv, &vops, &gpu_mat_bc, &b_gpu, &mut x_gpu, 1e-8, 2000);
    assert!(result.is_ok(), "GPU CG solve failed: {:?}", result.err());
    let u_gpu = x_gpu.read_to_cpu(&gpu);

    // ── Compare ────────────────────────────────────────────────────────────
    let diff_norm: f64 = u_cpu.iter().zip(u_gpu.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    let ref_norm: f64 = u_cpu.iter().map(|a| a.powi(2)).sum();
    let err = (diff_norm / ref_norm.max(1e-300)).sqrt();
    eprintln!("E2E Poisson 2D f64: CPU DOF={}, GPU DOF={}, rel_error={:.3e}",
        u_cpu.len(), u_gpu.len(), err);
    assert!(err < 1e-6, "E2E GPU vs CPU solution mismatch: {:.3e}", err);
}

// ─── Hex8 end-to-end tests ─────────────────────────────────────────────────

/// Extract Hex8 P1 element data: 8 nodes × 3 coords + 8 DOFs per element.
fn extract_hex8_p1(space: &H1Space<SimplexMesh<3>>) -> (Vec<f64>, Vec<u32>, usize) {
    let mesh = space.mesh();
    let n_elem = mesh.n_elems();
    let mut elem_nodes = Vec::with_capacity(n_elem * 24);
    let mut elem_dofs = Vec::with_capacity(n_elem * 8);
    for e in 0..n_elem as u32 {
        let ns = mesh.element_nodes(e);
        for ni in ns.iter() {
            let c = mesh.node_coords(*ni);
            elem_nodes.push(c[0]); elem_nodes.push(c[1]); elem_nodes.push(c[2]);
        }
        let dofs = space.element_dofs(e);
        for i in 0..dofs.len() { elem_dofs.push(dofs[i]); }
    }
    (elem_nodes, elem_dofs, n_elem)
}

#[test]
#[ignore]
fn gpu_vs_cpu_poisson_hex8_f64() {
    let n = 4;
    let mesh = SimplexMesh::<3>::unit_cube_hex(n);
    let space = H1Space::new(mesh, 1);
    let cpu = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
    eprintln!("CPU Poisson Hex8: {} DOF", space.n_dofs());

    let mesh = SimplexMesh::<3>::unit_cube_hex(n);
    let space = H1Space::new(mesh, 1);
    let (elem_nodes, elem_dofs, n_elem) = extract_hex8_p1(&space);

    use fem_linalg_gpu::assembly::assemble_poisson_3d_hex8_f64;
    let gpu = pollster::block_on(fem_linalg_gpu::GpuContext::new()).expect("GPU context");
    if !gpu.features.native_f64 { eprintln!("SKIP: no SHADER_F64"); return; }

    let triplets = assemble_poisson_3d_hex8_f64(&gpu, &elem_nodes, &elem_dofs, n_elem);
    let mut coo = fem_linalg::CooMatrix::new(cpu.nrows, cpu.ncols);
    for &(r,c,v) in &triplets { coo.add(r as usize, c as usize, v); }
    let gpu_mat: CsrMatrix<f64> = coo.into_csr();

    let mut max_rel = 0.0_f64;
    for i in 0..cpu.nrows.min(gpu_mat.nrows) {
        for k in cpu.row_ptr[i]..cpu.row_ptr[i+1] {
            let j = cpu.col_idx[k] as usize;
            let diff = (cpu.values[k] - gpu_mat.get(i, j)).abs();
            let rel = diff / cpu.values[k].abs().max(1.0_f64);
            max_rel = max_rel.max(rel);
        }
    }
    eprintln!("Poisson Hex8 f64: max_rel={:.3e}", max_rel);
    assert!(max_rel < 1e-12_f64, "Hex8 Poisson mismatch: {:.3e}", max_rel);
}

#[test]
#[ignore]
fn e2e_hex8_assemble_solve_poisson_f64() {
    let n = 3;
    // ── CPU reference ─────────────────────────────────────────────────────
    let mesh = SimplexMesh::<3>::unit_cube_hex(n);
    let space = H1Space::new(mesh.clone(), 1);
    let dm = fem_space::DofManager::new(&mesh, 1);
    let a_cpu = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
    let f = |x: &[f64]| 3.0 * std::f64::consts::PI.powi(2) * (std::f64::consts::PI * x[0]).sin()
        * (std::f64::consts::PI * x[1]).sin() * (std::f64::consts::PI * x[2]).sin();
    let mut rhs_cpu = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(f)], 3);
    let bdofs: Vec<usize> = boundary_dofs(&mesh, &dm, &[1, 2, 3, 4, 5, 6]).iter()
        .map(|&d| d as usize).collect();
    let mut a_cpu_mut = a_cpu;
    let bdofs_u32: Vec<u32> = bdofs.iter().map(|&d| d as u32).collect();
    fem_space::constraints::apply_dirichlet(&mut a_cpu_mut, &mut rhs_cpu, &bdofs_u32, &vec![0.0; bdofs.len()]);
    let mut u_cpu = vec![0.0; space.n_dofs()];
    solve_cg(&a_cpu_mut, &rhs_cpu, &mut u_cpu, &SolverConfig { rtol: 1e-8, ..Default::default() })
        .expect("CPU Hex8 CG");

    // ── GPU path ──────────────────────────────────────────────────────────
    let gpu = pollster::block_on(fem_linalg_gpu::GpuContext::new()).expect("GPU context");
    if !gpu.features.native_f64 { eprintln!("SKIP: no SHADER_F64"); return; }

    let mesh_g = SimplexMesh::<3>::unit_cube_hex(n);
    let space_g = H1Space::new(mesh_g, 1);
    let (elem_nodes, elem_dofs, n_elem) = extract_hex8_p1(&space_g);

    use fem_linalg_gpu::assembly::assemble_poisson_3d_hex8_f64;
    let triplets = assemble_poisson_3d_hex8_f64(&gpu, &elem_nodes, &elem_dofs, n_elem);

    let n_dofs = space_g.n_dofs() as u32;
    let mut coo = fem_linalg::CooMatrix::new(n_dofs as usize, n_dofs as usize);
    for &(r,c,v) in &triplets { if v != 0.0 { coo.add(r as usize, c as usize, v); } }
    let csr_gpu: CsrMatrix<f64> = coo.into_csr();

    let mut a_bc = csr_gpu;
    let mut rhs_gpu = vec![0.0; n_dofs as usize];
    for &d in &bdofs { a_bc.apply_dirichlet_row_zeroing(d, 0.0, &mut rhs_gpu); rhs_gpu[d] = 0.0; }

    use fem_linalg_gpu::{GpuVector, GpuCsrMatrix, SpmvPipeline, VectorOpsPipeline, solve_cg_gpu};
    let spmv = SpmvPipeline::new(&gpu.device, true);
    let vops = VectorOpsPipeline::new(&gpu.device, true);
    let gpu_mat = GpuCsrMatrix::<f64>::from_cpu(&gpu, &a_bc);
    let b_gpu = GpuVector::<f64>::from_slice(&gpu, &rhs_gpu);
    let mut x_gpu = GpuVector::<f64>::zeros(&gpu, n_dofs);

    let result = solve_cg_gpu::<f64>(&gpu, &spmv, &vops, &gpu_mat, &b_gpu, &mut x_gpu, 1e-8, 2000);
    assert!(result.is_ok(), "Hex8 GPU CG failed: {:?}", result.err());
    let u_gpu = x_gpu.read_to_cpu(&gpu);

    let diff: f64 = u_cpu.iter().zip(u_gpu.iter()).map(|(a,b)| (a-b).powi(2)).sum();
    let ref_: f64 = u_cpu.iter().map(|a| a.powi(2)).sum();
    let err = (diff / ref_.max(1e-300)).sqrt();
    eprintln!("E2E Hex8 Poisson 3D f64: rel_error={:.3e}", err);
    assert!(err < 1e-6_f64, "Hex8 E2E mismatch: {:.3e}", err);
}
