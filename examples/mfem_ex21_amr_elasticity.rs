//! Example 21 — AMR for linear elasticity (1:1 translation of MFEM ex21).
//!
//! Multi-material cantilever beam with adaptive mesh refinement,
//! ZZ error estimator, and PCG+GSSmoother solver.
//!
//! Supports 2D (Tri3/Quad4) and 3D (Tet4/Hex8/Prism6).
//!
//! Usage:
//!   cargo run --example mfem_ex21_amr_elasticity
//!   cargo run --example mfem_ex21_amr_elasticity -- -m data/beam-tri.mesh -o 2
//!   cargo run --example mfem_ex21_amr_elasticity -- -m data/beam-tet.mesh -o 2
//!   cargo run --example mfem_ex21_amr_elasticity -- -m data/beam-hex.mesh -o 2 -sc

#![allow(non_snake_case)]

use fem_assembly::assembler::face_dofs_p1;
use fem_assembly::static_cond::condense_global;
use fem_assembly::postproc::coefficient::PWConstCoeff;
use fem_assembly::postproc::error_estimate::zz_estimator;
use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::standard::{ElasticityIntegrator, NeumannIntegrator};
use fem_assembly::Assembler;
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_file_3d, write_mfem_file_with_coords, write_mfem_gf_file};
use fem_linalg::SolverConfig;
use fem_mesh::element_type::ElementType;
use fem_mesh::{Mesh, MeshTopology};
use fem_solver::solve_pcg_gssmoother;
use fem_solver::solve_sparse_lu;
use fem_space::constraints::boundary_dofs;
use fem_space::constraints::prolong::prolongate_pk_hanging;
use fem_space::dof_manager::DofManager;
use fem_space::{FESpace, VectorH1Space};

/// Threshold refiner: mark elements with error above a fraction of total.
fn mark_elements(eta: &[f64], fraction: f64) -> Vec<u32> {
    let total: f64 = eta.iter().sum();
    if total <= 0.0 { return Vec::new(); }
    let target = fraction * total;
    let mut idx: Vec<usize> = (0..eta.len()).collect();
    idx.sort_by(|&a, &b| eta[b].partial_cmp(&eta[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut cum = 0.0_f64;
    let mut marked = Vec::new();
    for &i in &idx { marked.push(i as u32); cum += eta[i]; if cum >= target { break; } }
    marked
}

fn main() {
    // 1. Parse command-line options (matching MFEM ex21)
    let mut mesh_file = "data/beam-tri.mesh".to_string();
    let mut order = 1u8;
    let mut static_cond = false;
    let mut _flux_averaging = 0i32;
    let mut visualization = false;
    let mut use_direct = false;
    let max_dofs = 50000usize;
    let max_amr_itr = 20usize;

    let mut i = std::env::args().skip(1);
    while let Some(arg) = i.next() {
        match arg.as_str() {
            "-h" | "--help" => { eprintln!("Usage: ex21 [-m mesh] [-o order] [-sc/-no-sc] [-f 0|1] [-vis/-no-vis]"); return; }
            "-m" | "--mesh" => mesh_file = i.next().unwrap_or_default(),
            "-o" | "--order" => order = i.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-sc" | "--static-condensation" => static_cond = true,
            "-no-sc" | "--no-static-condensation" => static_cond = false,
            "-f" | "--flux-averaging" => _flux_averaging = i.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "-vis" | "--visualization" => visualization = true,
            "-no-vis" | "--no-visualization" => visualization = false,
            "-direct" | "--direct-solver" => use_direct = true,
            _ => {}
        }
    }
    println!("Options used:\n   --mesh {mesh_file}\n   --order {order}");

    // 2. Read mesh — try 2D first, then 3D
    let mfem_data = read_mfem_file(&mesh_file).expect("failed to read mesh");
    if let Some(mesh2d) = mfem_data.mesh2d {
        run_2d(mesh2d, order, static_cond, visualization, use_direct, max_dofs, max_amr_itr);
    } else if let Some(mesh3d) = mfem_data.mesh3d {
        run_3d(mesh3d, order, static_cond, visualization, use_direct, max_dofs, max_amr_itr);
    } else {
        eprintln!("Mesh file must be 2D or 3D");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2D implementation
// ═══════════════════════════════════════════════════════════════════════════════

fn run_2d(
    mut mesh: Mesh<2>, order: u8, static_cond: bool,
    visualization: bool, use_direct: bool, max_dofs: usize, max_amr_itr: usize,
) {
    use fem_mesh::amr::{closure_refine, refine_nonconforming_quad};
    let dim = 2usize;
    let is_quad = matches!(mesh.element_type(0), ElementType::Quad4);
    let quad_order = order * 2 + 1;
    let lambda_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let mu_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let elasticity = ElasticityIntegrator::new(lambda_coeff, mu_coeff);
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() };

    let mut x_prev: Option<Vec<f64>> = None;
    let mut prev_mesh: Option<Mesh<2>> = None;
    let mut prev_dm: Option<DofManager> = None;

    for it in 0..=max_amr_itr {
        let space = VectorH1Space::new(mesh.clone(), order, dim as u8);
        let n_dofs = space.n_dofs();
        let n_scalar = space.n_scalar_dofs();
        println!("\nAMR iteration {it}\nNumber of unknowns: {n_dofs}");

        let dm = space.scalar_dof_manager();
        let ess_bdr = boundary_dofs(space.mesh(), dm, &[1]);
        let mut ess: Vec<usize> = Vec::new();
        for &d in &ess_bdr { ess.push(d as usize); ess.push(d as usize + n_scalar); }

        let fdofs = face_dofs_p1(space.mesh());
        let neumann = NeumannIntegrator::new(|_: &[f64], _: &[f64]| -1.0e-2);
        let traction_y = Assembler::assemble_boundary_linear(
            n_scalar, space.mesh(), &fdofs, order, &[&neumann], &[2], quad_order,
        );
        let mut rhs = vec![0.0_f64; n_dofs];
        for (i, &v) in traction_y.iter().enumerate() { rhs[n_scalar + i] += v; }

        let mut mat = Assembler::assemble_bilinear(&space, &[&elasticity], quad_order);
        for &d in &ess { mat.apply_dirichlet_row_zeroing(d, 0.0, &mut rhs); }

        let mut x = if let (Some(prev_u), Some(ref pmesh), Some(ref pdm)) = (&x_prev, &prev_mesh, &prev_dm) {
            if !is_quad && pdm.n_dofs > 0 {
                let mut u_new = vec![0.0_f64; n_dofs];
                for c in 0..2 {
                    let comp_prev = &prev_u[c * pdm.n_dofs..(c + 1) * pdm.n_dofs];
                    let comp_new = prolongate_pk_hanging::<Mesh<2>>(pmesh, pdm, dm, comp_prev);
                    for (j, &v) in comp_new.iter().enumerate() { u_new[c * n_scalar + j] = v; }
                }
                u_new
            } else { vec![0.0_f64; n_dofs] }
        } else { vec![0.0_f64; n_dofs] };

        // Static condensation + solve (shared by both dims)
        let (solve_mat, solve_rhs, backsub) = setup_solve(&mat, &rhs, dm, n_scalar, n_dofs, static_cond);
        let mut solve_x = vec![0.0_f64; solve_mat.nrows];
        if use_direct {
            match solve_sparse_lu(&solve_mat, &solve_rhs) { Ok(x_lu) => { solve_x = x_lu; } Err(e) => eprintln!("LU: {e}"), }
        } else {
            let _ = solve_pcg_gssmoother(&solve_mat, &solve_rhs, &mut solve_x, &cfg);
        }
        x = backsolve(backsub, solve_x, n_dofs);

        // ZZ estimator
        let gf = GridFunction::new(&space, x.clone());
        let est = zz_estimator(&gf);
        let max_err = est.eta.iter().cloned().fold(0.0_f64, f64::max);
        println!("  Max err: {max_err:.6e}");

        let marked = mark_elements(&est.eta, 0.7);
        println!("  Marked {} elements", marked.len());

        if marked.is_empty() || n_dofs > max_dofs {
            if n_dofs > max_dofs { println!("Reached max DOFs. Stop."); }
            if marked.is_empty() { println!("No elements marked. Stop."); }
            x_prev = Some(x); break;
        }

        prev_mesh = Some(mesh.clone());
        prev_dm = Some(space.scalar_dof_manager().clone());

        if is_quad {
            let (new_mesh, _hanging) = refine_nonconforming_quad(&mesh, &marked, None);
            x_prev = Some(x);
            mesh = new_mesh;
        } else {
            let new_mesh = closure_refine(&mesh, &marked, 20, None);
            x_prev = Some(x);
            mesh = new_mesh;
        }
    }

    // Output
    write_output_2d(&mesh, &x_prev, order, dim, visualization);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3D implementation
// ═══════════════════════════════════════════════════════════════════════════════

fn run_3d(
    mut mesh: Mesh<3>, order: u8, static_cond: bool,
    visualization: bool, use_direct: bool, max_dofs: usize, max_amr_itr: usize,
) {
    use fem_mesh::amr::refine_nonconforming_3d;
    let dim = 3usize;
    let quad_order = order * 2 + 1;
    // MFEM ex21: lambda/mu use attribute 0 and 1 (index 0/1).
    // For 3D meshes attributes may differ; fallback to uniform if fewer attributes.
    let lambda_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let mu_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let elasticity = ElasticityIntegrator::new(lambda_coeff, mu_coeff);
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() };

    let mut x_prev: Option<Vec<f64>> = None;
    let mut prev_mesh: Option<Mesh<3>> = None;
    let mut prev_dm: Option<DofManager> = None;

    for it in 0..=max_amr_itr {
        let space = VectorH1Space::new(mesh.clone(), order, dim as u8);
        let n_dofs = space.n_dofs();
        let n_scalar = space.n_scalar_dofs();
        println!("\nAMR iteration {it}\nNumber of unknowns: {n_dofs}");

        let dm = space.scalar_dof_manager();
        let ess_bdr = boundary_dofs(space.mesh(), dm, &[1]);
        let mut ess: Vec<usize> = Vec::new();
        for &d in &ess_bdr { ess.push(d as usize); ess.push(d as usize + n_scalar); }

        let fdofs = face_dofs_p1(space.mesh());
        let neumann = NeumannIntegrator::new(|_: &[f64], _: &[f64]| -1.0e-2);
        // 3D: traction in z-direction (index dim-1 = 2)
        let traction_z = Assembler::assemble_boundary_linear(
            n_scalar, space.mesh(), &fdofs, order, &[&neumann], &[2], quad_order,
        );
        let mut rhs = vec![0.0_f64; n_dofs];
        for (i, &v) in traction_z.iter().enumerate() { rhs[(dim - 1) * n_scalar + i] += v; }

        let mut mat = Assembler::assemble_bilinear(&space, &[&elasticity], quad_order);
        for &d in &ess { mat.apply_dirichlet_row_zeroing(d, 0.0, &mut rhs); }

        let mut x = if let (Some(prev_u), Some(ref pmesh), Some(ref pdm)) = (&x_prev, &prev_mesh, &prev_dm) {
            if pdm.n_dofs > 0 {
                let mut u_new = vec![0.0_f64; n_dofs];
                for c in 0..dim {
                    let comp_prev = &prev_u[c * pdm.n_dofs..(c + 1) * pdm.n_dofs];
                    let comp_new = prolongate_pk_hanging::<Mesh<3>>(pmesh, pdm, dm, comp_prev);
                    for (j, &v) in comp_new.iter().enumerate() { u_new[c * n_scalar + j] = v; }
                }
                u_new
            } else { vec![0.0_f64; n_dofs] }
        } else { vec![0.0_f64; n_dofs] };

        let (solve_mat, solve_rhs, backsub) = setup_solve(&mat, &rhs, dm, n_scalar, n_dofs, static_cond);
        let mut solve_x = vec![0.0_f64; solve_mat.nrows];
        if use_direct {
            match solve_sparse_lu(&solve_mat, &solve_rhs) { Ok(x_lu) => { solve_x = x_lu; } Err(e) => eprintln!("LU: {e}"), }
        } else {
            let _ = solve_pcg_gssmoother(&solve_mat, &solve_rhs, &mut solve_x, &cfg);
        }
        x = backsolve(backsub, solve_x, n_dofs);

        let gf = GridFunction::new(&space, x.clone());
        let est = zz_estimator(&gf);
        let max_err = est.eta.iter().cloned().fold(0.0_f64, f64::max);
        println!("  Max err: {max_err:.6e}");

        let marked = mark_elements(&est.eta, 0.7);
        println!("  Marked {} elements", marked.len());

        if marked.is_empty() || n_dofs > max_dofs {
            if n_dofs > max_dofs { println!("Reached max DOFs. Stop."); }
            if marked.is_empty() { println!("No elements marked. Stop."); }
            x_prev = Some(x); break;
        }

        prev_mesh = Some(mesh.clone());
        prev_dm = Some(space.scalar_dof_manager().clone());

        // 3D non-conforming refinement (handles Tet/Hex/Prism)
        let (new_mesh, _edge_hanging, _face_hanging) = refine_nonconforming_3d(&mesh, &marked, None);
        x_prev = Some(x);
        mesh = new_mesh;
    }

    write_output_3d(&mesh, &x_prev, order, dim, visualization);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Shared helpers
// ═══════════════════════════════════════════════════════════════════════════════

fn setup_solve(
    mat: &CsrMatrix<f64>, rhs: &[f64], dm: &DofManager,
    n_scalar: usize, n_dofs: usize, static_cond: bool,
) -> (CsrMatrix<f64>, Vec<f64>, Option<fem_assembly::static_cond::GlobalBacksolve>) {
    if static_cond {
        let bs = dm.bubble_dof_start;
        let interior: Vec<usize> = (0..n_dofs).filter(|&d| (d % n_scalar) >= bs).collect();
        if !interior.is_empty() {
            let (cmat, crhs, bs) = condense_global(mat, rhs, &interior);
            println!("  SC: {}=>{}, {} interior DOFs eliminated", n_dofs, cmat.nrows, interior.len());
            return (cmat, crhs, Some(bs));
        }
    }
    (mat.clone(), rhs.to_vec(), None)
}

fn backsolve(
    backsub: Option<fem_assembly::static_cond::GlobalBacksolve>,
    solve_x: Vec<f64>, n_dofs: usize,
) -> Vec<f64> {
    if let Some(bs) = &backsub {
        match bs.backsolve(&solve_x, 1e-12, 2000) {
            Ok(u_i) => {
                let mut full = vec![0.0_f64; n_dofs];
                for (k, &g) in bs.boundary.iter().enumerate() { full[g] = solve_x[k]; }
                for (k, &g) in bs.interior.iter().enumerate() { full[g] = u_i[k]; }
                return full;
            }
            Err(e) => eprintln!("SC backsolve error: {e}"),
        }
    }
    solve_x
}

fn write_output_2d(mesh: &Mesh<2>, x_prev: &Option<Vec<f64>>, order: u8, dim: usize, visualization: bool) {
    let space_final = VectorH1Space::new(mesh.clone(), order, dim as u8);
    let n_scalar_final = space_final.n_scalar_dofs();
    let nn = mesh.n_nodes() as usize;

    write_mfem_file("ex21_reference.mesh", mesh).expect("write ref");
    println!("Wrote ex21_reference.mesh");

    if let Some(ref u_final) = x_prev {
        let mut def_coords = Vec::with_capacity(nn * dim);
        for n in 0..nn {
            let c = mesh.node_coords(n as u32);
            let ux = if n < n_scalar_final { u_final[n] } else { 0.0 };
            let uy = if n + n_scalar_final < u_final.len() { u_final[n_scalar_final + n] } else { 0.0 };
            def_coords.push(c[0] + ux); def_coords.push(c[1] + uy);
        }
        write_mfem_file_with_coords("ex21_deformed.mesh", mesh, &def_coords).expect("write deformed");
        println!("Wrote ex21_deformed.mesh");

        write_mfem_gf_file("ex21_displacement.sol", dim, u_final, "H1", order, dim, 16).expect("write sol");
        println!("Wrote ex21_displacement.sol");

        if visualization {
            write_mfem_file_with_coords("ex21_vis.mesh", mesh, &def_coords).expect("write vis mesh");
            write_mfem_gf_file("ex21_vis.sol", dim, u_final, "H1", order, dim, 16).expect("write vis sol");
            println!("  GLVis: glvis -m ex21_vis.mesh -g ex21_vis.sol");
        }
    }
}

fn write_output_3d(mesh: &Mesh<3>, x_prev: &Option<Vec<f64>>, order: u8, dim: usize, visualization: bool) {
    let space_final = VectorH1Space::new(mesh.clone(), order, dim as u8);
    let n_scalar_final = space_final.n_scalar_dofs();
    let nn = mesh.n_nodes() as usize;

    use fem_io::mfem::write_mfem_file_3d;
    // For 3D output, we need to use write_mfem_file_3d or write the mesh directly
    // write_mfem_file_3d creates a dummy 2D mesh internally
    write_mfem_file_3d("ex21_reference.mesh", mesh).expect("write ref");
    println!("Wrote ex21_reference.mesh (3D)");

    if let Some(ref u_final) = x_prev {
        let mut def_coords = vec![0.0_f64; nn * dim];
        for n in 0..nn {
            let c = mesh.node_coords(n as u32);
            for d in 0..dim {
                let u_d = if n < n_scalar_final { u_final[d * n_scalar_final + n] } else { 0.0 };
                def_coords[n * dim + d] = c[d] + u_d;
            }
        }
        // For 3D deformed mesh, create a displaced clone
        let mut displaced = mesh.clone();
        displaced.coords.copy_from_slice(&def_coords);
        write_mfem_file_3d("ex21_deformed.mesh", &displaced).expect("write deformed");
        println!("Wrote ex21_deformed.mesh (3D)");

        write_mfem_gf_file("ex21_displacement.sol", dim, u_final, "H1", order, dim, 16).expect("write sol");
        println!("Wrote ex21_displacement.sol");

        if visualization {
            write_mfem_gf_file("ex21_vis.sol", dim, u_final, "H1", order, dim, 16).expect("write vis sol");
            // For 3D vis, just write the displaced mesh
            write_mfem_file_3d("ex21_vis.mesh", &displaced).expect("write vis mesh");
            println!("  GLVis: glvis -m ex21_vis.mesh -g ex21_vis.sol");
        }
    }
}

// Needed for CsrMatrix in setup_solve signature
use fem_linalg::CsrMatrix;
