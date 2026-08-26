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
use fem_assembly::postproc::error_estimate::zz_estimator_stress;
use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::standard::{ElasticityIntegrator, NeumannIntegrator};
use fem_assembly::Assembler;
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_file_3d, write_mfem_file_with_coords, write_mfem_gf_file};
use fem_linalg::{CsrMatrix, SolverConfig};
use fem_mesh::element_type::ElementType;
use fem_mesh::{Mesh, MeshTopology};
use fem_solver::solve_pcg_gssmoother;
use fem_solver::solve_sparse_lu;
use fem_space::constraints::boundary_dofs;
use fem_space::constraints::prolong::prolongate_pk_hanging;
use fem_space::dof_manager::DofManager;
use fem_space::{FESpace, VectorH1Space};

fn mark_elements(eta: &[f64], fraction: f64) -> Vec<u32> {
    // MFEM ThresholdRefiner (ex21): threshold = total_fraction · ‖η‖_∞
    // (total_norm_p = ∞), mark every element with ηᵉ ≥ threshold.
    // A cumulative/Dörfler marking changes the AMR trajectory.
    let max_err = eta.iter().cloned().fold(0.0_f64, f64::max);
    if max_err <= 0.0 { return Vec::new(); }
    let threshold = fraction * max_err;
    (0..eta.len())
        .filter(|&i| eta[i] >= threshold)
        .map(|i| i as u32)
        .collect()
}

// ─── Macro: single AMR loop body for Mesh<2> or Mesh<3> ────────────────────
macro_rules! amr_loop {
    ($mesh:expr, $order:expr, $dim:expr, $static_cond:expr, $use_direct:expr,
     $cfg:expr, $elasticity:expr, $quad_order:expr, $max_dofs:expr, $max_amr_itr:expr,
     $refine_fn:expr, $write_ref:expr, $write_def:expr, $vis_path:expr) => {{
        let mut mesh = $mesh;
        let order = $order;
        let dim = $dim;
        let use_direct = $use_direct;
        let quad_order = $quad_order;
        let cfg = $cfg;
        let elasticity = $elasticity;
        let quad_order_u8 = quad_order as u8;
        let max_dofs = $max_dofs;
        let max_amr_itr = $max_amr_itr;
        let static_cond = $static_cond;
        let visualize = $vis_path;

        let mut x_prev: Option<Vec<f64>> = None;
        let mut prev_mesh = None;
        let mut prev_dm: Option<DofManager> = None;

        for it in 0..=max_amr_itr {
            let space = VectorH1Space::new(mesh.clone(), order, dim as u8);
            let n_dofs = space.n_dofs();
            let n_scalar = space.n_scalar_dofs();
            println!("\nAMR iteration {it}\nNumber of unknowns: {n_dofs}");

            let dm = space.scalar_dof_manager();
            let ess_bdr = boundary_dofs(space.mesh(), &dm, &[1]);
            let mut ess: Vec<usize> = Vec::new();
            for &d in &ess_bdr { ess.push(d as usize); ess.push(d as usize + n_scalar); }

            let fdofs = face_dofs_p1(space.mesh());
            let neumann = NeumannIntegrator::new(|_: &[f64], _: &[f64]| -1.0e-2);
            let traction = Assembler::assemble_boundary_linear(
                n_scalar, space.mesh(), &fdofs, order, &[&neumann], &[2], quad_order_u8,
            );
            let mut rhs = vec![0.0_f64; n_dofs];
            for (i, &v) in traction.iter().enumerate() { rhs[(dim - 1) * n_scalar + i] += v; }

            let mut mat = Assembler::assemble_bilinear(&space, &[&elasticity], quad_order_u8);
            for &d in &ess { mat.apply_dirichlet_row_zeroing(d, 0.0, &mut rhs); }

            let mut x = if let (Some(prev_u), Some(ref pmesh), Some(ref pdm)) = (&x_prev, &prev_mesh, &prev_dm) {
                if pdm.n_dofs > 0 {
                    let mut u_new = vec![0.0_f64; n_dofs];
                    for c in 0..dim {
                        let cp = &prev_u[c * pdm.n_dofs..(c + 1) * pdm.n_dofs];
                        let cn = prolongate_pk_hanging(pmesh, pdm, &dm, cp);
                        for (j, &v) in cn.iter().enumerate() { u_new[c * n_scalar + j] = v; }
                    }
                    u_new
                } else { vec![0.0_f64; n_dofs] }
            } else { vec![0.0_f64; n_dofs] };

            // Static condensation
            let (solve_mat, solve_rhs, backsub) = if static_cond {
                let bs = dm.bubble_dof_start;
                let interior: Vec<usize> = (0..n_dofs).filter(|&d| (d % n_scalar) >= bs).collect();
                if !interior.is_empty() {
                    let (cmat, crhs, bs) = condense_global(&mat, &rhs, &interior);
                    println!("  SC: {}=>{}, {} int DOFs", n_dofs, cmat.nrows, interior.len());
                    (cmat, crhs, Some(bs))
                } else { (mat.clone(), rhs.clone(), None) }
            } else { (mat.clone(), rhs.clone(), None) };

            let mut solve_x = vec![0.0_f64; solve_mat.nrows];
            if use_direct {
                match solve_sparse_lu(&solve_mat, &solve_rhs) {
                    Ok(x_lu) => { solve_x = x_lu; println!("  Direct solve"); }
                    Err(e) => eprintln!("LU error: {e}"),
                }
            } else {
                let _ = solve_pcg_gssmoother(&solve_mat, &solve_rhs, &mut solve_x, &cfg);
            }

            x = if let Some(ref bs) = backsub {
                match bs.backsolve(&solve_x, 1e-12, 2000) {
                    Ok(u_i) => {
                        let mut full = vec![0.0_f64; n_dofs];
                        for (k, &g) in bs.boundary.iter().enumerate() { full[g] = solve_x[k]; }
                        for (k, &g) in bs.interior.iter().enumerate() { full[g] = u_i[k]; }
                        full
                    }
                    Err(e) => { eprintln!("SC backsolve: {e}"); solve_x }
                }
            } else { solve_x };

            // ZZ estimator
            let gf = GridFunction::new(&space, x.clone());
            let est = zz_estimator_stress(&gf, &|t: i32| if t == 1 { 50.0 } else { 1.0 }, &|t: i32| if t == 1 { 50.0 } else { 1.0 });
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

            // Refine
            mesh = $refine_fn(mesh, &marked);
            x_prev = Some(x);
        }

        // Output
        let ns_final = {
            let space_final = VectorH1Space::new(mesh.clone(), order, dim as u8);
            space_final.n_scalar_dofs()
        };
        let nn = mesh.n_nodes() as usize;

        ($write_ref)(&mesh).expect("write ref");
        println!("Wrote ex21_reference.mesh");

        if let Some(ref u_final) = x_prev {
            let mut def_coords = vec![0.0_f64; nn * dim];
            for n in 0..nn {
                let c = mesh.node_coords(n as u32);
                for d in 0..dim {
                    let u_d = if (n as usize) < ns_final
                        && d * ns_final + (n as usize) < u_final.len() {
                        u_final[d * ns_final + (n as usize)]
                    } else { 0.0 };
                    def_coords[(n as usize) * dim + d] = c[d] + u_d;
                }
            }

            ($write_def)(&mesh, &def_coords).expect("write deformed");
            println!("Wrote ex21_deformed.mesh");

            write_mfem_gf_file("ex21_displacement.sol", dim, u_final, "H1", order, dim, 16)
                .expect("write sol");
            println!("Wrote ex21_displacement.sol");

            if visualize {
                write_mfem_gf_file("ex21_vis.sol", dim, u_final, "H1", order, dim, 16)
                    .expect("write vis sol");
                ($write_def)(&mesh, &def_coords).expect("write vis");
                println!("  glvis -m ex21_vis.mesh -g ex21_vis.sol");
            }
        }
    }};
}

fn main() {
    // 1. CLI (matching MFEM ex21)
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
            "-h" | "--help" => { eprintln!("Usage: ex21 [-m mesh] [-o order] [-sc/-no-sc] [-f 0|1] [-vis/-no-vis] [-direct]"); return; }
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

    // 2. Read mesh and dispatch
    let mfem_data = read_mfem_file(&mesh_file).expect("failed to read mesh");

    if let Some(mesh2d) = mfem_data.mesh2d {
        amr_loop_2d(mesh2d, order, static_cond, use_direct, max_dofs, max_amr_itr, visualization);
    } else if let Some(mesh3d) = mfem_data.mesh3d {
        amr_loop_3d(mesh3d, order, static_cond, use_direct, max_dofs, max_amr_itr, visualization);
    } else {
        panic!("Mesh must be 2D or 3D");
    }
}

// ─── 2D entry: macro expansion with 2D-specific refinement + output ────────

fn amr_loop_2d(mesh: Mesh<2>, order: u8, static_cond: bool, use_direct: bool,
               max_dofs: usize, max_amr_itr: usize, visualization: bool) {
    use fem_mesh::amr::{closure_refine, refine_nonconforming_quad};
    let dim = 2usize;
    let quad_order = order as usize * 2 + 1;
    let lambda_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let mu_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let elasticity = ElasticityIntegrator::new(lambda_coeff, mu_coeff);
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() };

    // Detect element type to choose conforming (Tri) vs non-conforming (Quad) refinement
    let is_quad = matches!(mesh.element_type(0), ElementType::Quad4);

    amr_loop!(mesh, order, dim, static_cond, use_direct, cfg, elasticity, quad_order,
              max_dofs, max_amr_itr,
              // Refinement function
              |m: Mesh<2>, marked: &[u32]| {
                  if is_quad {
                      let (new_mesh, _h) = refine_nonconforming_quad(&m, marked, None);
                      new_mesh
                  } else {
                      closure_refine(&m, marked, 20, None)
                  }
              },
              // Write reference mesh
              |m: &Mesh<2>| write_mfem_file("ex21_reference.mesh", m),
              // Write deformed mesh
              |m: &Mesh<2>, coords: &[f64]| write_mfem_file_with_coords("ex21_deformed.mesh", m, coords),
              visualization);
}

// ─── 3D entry ────────────────────────────────────────────────────────────

fn amr_loop_3d(mesh: Mesh<3>, order: u8, static_cond: bool, use_direct: bool,
               max_dofs: usize, max_amr_itr: usize, visualization: bool) {
    use fem_mesh::amr::refine_nonconforming_3d;
    let dim = 3usize;
    let quad_order = order as usize * 2 + 1;
    let lambda_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let mu_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let elasticity = ElasticityIntegrator::new(lambda_coeff, mu_coeff);
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() };

    amr_loop!(mesh, order, dim, static_cond, use_direct, cfg, elasticity, quad_order,
              max_dofs, max_amr_itr,
              // Refinement function (non-conforming for all 3D element types)
              |m: Mesh<3>, marked: &[u32]| {
                  let (new_mesh, _, _) = refine_nonconforming_3d(&m, marked, None);
                  new_mesh
              },
              // Write reference mesh (3D)
              |m: &Mesh<3>| write_mfem_file_3d("ex21_reference.mesh", m),
              // Write deformed mesh (3D)
              |m: &Mesh<3>, coords: &[f64]| {
                  let mut displaced = m.clone();
                  displaced.coords.copy_from_slice(coords);
                  write_mfem_file_3d("ex21_deformed.mesh", &displaced)
              },
              visualization);
}
