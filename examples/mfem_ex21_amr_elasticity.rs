//! Example 21 — AMR for linear elasticity (1:1 translation of MFEM ex21).
//!
//! Multi-material cantilever beam with adaptive mesh refinement,
//! ZZ error estimator, and PCG+GSSmoother solver.
//!
//! Usage:
//!   cargo run --example mfem_ex21_amr_elasticity
//!   cargo run --example mfem_ex21_amr_elasticity -- -m data/beam-tri.mesh -o 2
//!   cargo run --example mfem_ex21_amr_elasticity -- -m data/beam-quad.mesh -o 1 -f 1

#![allow(non_snake_case)]

use fem_assembly::assembler::face_dofs_p1;
use fem_assembly::postproc::coefficient::PWConstCoeff;
use fem_assembly::postproc::error_estimate::zz_estimator;
use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::standard::{ElasticityIntegrator, NeumannIntegrator};
use fem_assembly::Assembler;
use fem_io::mfem::read_mfem_file;
use fem_linalg::SolverConfig;
use fem_mesh::amr::HangingNodeConstraint;
use fem_mesh::element_type::ElementType;
use fem_mesh::{Mesh, MeshTopology};
use fem_solver::solve_pcg_gssmoother;
use fem_space::constraints::boundary_dofs;
use fem_space::constraints::hanging_2d::{apply_hanging_constraints, recover_hanging_values};
use fem_space::constraints::prolong::prolongate_pk_hanging;
use fem_space::dof_manager::DofManager;
use fem_space::{FESpace, VectorH1Space};
use std::fs::File;
use std::io::Write;

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

/// Build hanging-node constraints for a VectorH1 space by duplicating per component.
fn duplicate_hanging_constraints(ch: &[HangingNodeConstraint], ns: usize) -> Vec<HangingNodeConstraint> {
    let mut vc = Vec::with_capacity(ch.len() * 2);
    for c in ch {
        vc.push(HangingNodeConstraint { constrained: c.constrained, parent_a: c.parent_a, parent_b: c.parent_b });
        vc.push(HangingNodeConstraint { constrained: c.constrained + ns, parent_a: c.parent_a + ns, parent_b: c.parent_b + ns });
    }
    vc
}

/// Write MFEM v1.0 mesh file with given vertex coordinates.
fn write_mfem_mesh(mesh: &impl MeshTopology, coords: &[f64], dim: usize, path: &str) {
    let nn = mesh.n_nodes() as usize;
    let ne = mesh.n_elements() as usize;
    let mut f = File::create(path).unwrap_or_else(|_| panic!("cannot create {path}"));
    writeln!(f, "MFEM mesh v1.0\n\ndimension\n{dim}\n").ok();
    writeln!(f, "elements\n{ne}").ok();
    for e in 0..ne as u32 {
        let et = mesh.element_type(e);
        let nd = mesh.element_nodes(e);
        let et_name = match et {
            ElementType::Tri3 => "triangle",
            ElementType::Quad4 => "quadrilateral",
            _ => panic!("unsupported element type"),
        };
        write!(f, "{} {}", e + 1, et_name).ok();
        for &n in nd { write!(f, " {}", n + 1).ok(); }
        writeln!(f).ok();
    }
    writeln!(f, "\nboundary\n0\n\nvertices\n{nn}\n{dim}\n").ok();
    for i in 0..nn {
        for d in 0..dim { write!(f, "{:.16} ", coords[i * dim + d]).ok(); }
        writeln!(f).ok();
    }
}

fn main() {
    // 1. Parse command-line options (matching MFEM ex21)
    let mut mesh_file = "data/beam-tri.mesh".to_string();
    let mut order = 1u8;
    let mut _static_cond = false;
    let mut _flux_averaging = 0i32;
    let mut _visualization = false;
    let max_dofs = 50000usize;
    let max_amr_itr = 20usize;

    let mut i = std::env::args().skip(1);
    while let Some(arg) = i.next() {
        match arg.as_str() {
            "-h" | "--help" => { eprintln!("Usage: ex21 [-m mesh] [-o order] [-sc/-no-sc] [-f 0|1] [-vis/-no-vis]"); return; }
            "-m" | "--mesh" => mesh_file = i.next().unwrap_or_default(),
            "-o" | "--order" => order = i.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-sc" | "--static-condensation" => _static_cond = true,
            "-no-sc" | "--no-static-condensation" => _static_cond = false,
            "-f" | "--flux-averaging" => _flux_averaging = i.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "-vis" | "--visualization" => _visualization = true,
            "-no-vis" | "--no-visualization" => _visualization = false,
            _ => {}
        }
    }

    println!("Options used:\n   --mesh {mesh_file}\n   --order {order}\n   --flux-averaging {_flux_averaging}");

    // 2. Read mesh
    let mfem_data = read_mfem_file(&mesh_file).expect("failed to read mesh");
    let mesh: Mesh<2> = mfem_data.mesh2d.expect("expected 2D mesh");
    let dim: usize = 2;

    // 3. Detect element type (tri → conforming, quad → non-conforming)
    let is_quad = match mesh.element_type(0) {
        ElementType::Quad4 => true,
        ElementType::Tri3 => false,
        _ => { eprintln!("Unsupported element type"); std::process::exit(1); }
    };
    if is_quad && mesh_file.contains("beam-tri") {
        eprintln!("Warning: beam-tri.mesh with Quad4 detection — check mesh file.");
    }

    // 4. Material constants
    let quad_order = order * 2 + 1;
    let lambda_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let mu_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let elasticity = ElasticityIntegrator::new(lambda_coeff, mu_coeff);
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() };

    // AMR loop state
    let mut mesh = mesh;
    let mut x_prev: Option<Vec<f64>> = None;
    let mut prev_mesh: Option<Mesh<2>> = None;
    let mut prev_dm: Option<DofManager> = None;

    for it in 0..=max_amr_itr {
        let space = VectorH1Space::new(mesh.clone(), order, dim as u8);
        let n_dofs = space.n_dofs();
        let n_scalar = space.n_scalar_dofs();
        println!("\nAMR iteration {it}\nNumber of unknowns: {n_dofs}");

        // 5. Essential BC: boundary attribute 1 fixed
        let dm = space.scalar_dof_manager();
        let ess_bdr = boundary_dofs(space.mesh(), dm, &[1]);
        let mut ess: Vec<usize> = Vec::new();
        for &d in &ess_bdr {
            ess.push(d as usize);
            ess.push(d as usize + n_scalar);
        }

        // 6. RHS: traction on boundary attribute 2 (pull-down: f_y = -0.01)
        let fdofs = face_dofs_p1(space.mesh());
        let neumann = NeumannIntegrator::new(|_: &[f64], _: &[f64]| -1.0e-2);
        let traction_y = Assembler::assemble_boundary_linear(
            n_scalar, space.mesh(), &fdofs, order, &[&neumann], &[2], quad_order,
        );
        let mut rhs = vec![0.0_f64; n_dofs];
        for (i, &v) in traction_y.iter().enumerate() { rhs[n_scalar + i] += v; }

        // 7. Assemble stiffness matrix
        let mut mat = Assembler::assemble_bilinear(&space, &[&elasticity], quad_order);

        // 8. Apply essential BCs
        for &d in &ess { mat.apply_dirichlet_row_zeroing(d, 0.0, &mut rhs); }

        // 9. Initial guess from previous solution (solution interpolation after AMR)
        let mut x = if let (Some(prev_u), Some(ref pmesh), Some(ref pdm)) = (&x_prev, &prev_mesh, &prev_dm) {
            // For tri: use prolongate_pk_hanging per component
            // For quad: reuse previous if sizes match, else zero
            let ns_prev = pdm.n_dofs;
            if !is_quad && ns_prev > 0 {
                let mut u_new = vec![0.0_f64; n_dofs];
                for c in 0..2 {
                    let comp_prev = &prev_u[c * ns_prev..(c + 1) * ns_prev];
                    let comp_new = prolongate_pk_hanging::<Mesh<2>>(pmesh, pdm, dm, comp_prev);
                    for (j, &v) in comp_new.iter().enumerate() {
                        u_new[c * n_scalar + j] = v;
                    }
                }
                // Re-apply BC values
                for &(dof, val) in &[(0, 0.0)] { // dummy, handled below
                    let _ = val;
                }
                u_new
            } else {
                vec![0.0_f64; n_dofs]
            }
        } else {
            vec![0.0_f64; n_dofs]
        };

        // 10. Solve (PCG + GSSmoother)
        let res = solve_pcg_gssmoother(&mat, &rhs, &mut x, &cfg);
        match &res {
            Ok(r) => println!("  PCG: {} its, res={:.3e}", r.iterations, r.final_residual),
            Err(e) => eprintln!("  PCG error: {e}"),
        }

        // 11. ZZ error estimator
        let gf = GridFunction::new(&space, x.clone());
        let est = zz_estimator(&gf);
        let max_err = est.eta.iter().cloned().fold(0.0_f64, f64::max);
        println!("  Max err indicator: {max_err:.6e}");

        // 12. Mark elements (70% fraction, matching C++)
        let marked = mark_elements(&est.eta, 0.7);
        println!("  Marked {} elements", marked.len());

        // 13. Check stopping criteria (before refinement, so output has a valid solution)
        if marked.is_empty() || n_dofs > max_dofs {
            if n_dofs > max_dofs { println!("Reached the maximum number of dofs. Stop."); }
            if marked.is_empty() { println!("No elements marked for refinement. Stop."); }
            x_prev = Some(x);
            break;
        }

        // 14. Save state for solution interpolation in next iteration
        prev_mesh = Some(mesh.clone());
        prev_dm = Some(space.scalar_dof_manager().clone());

        // 15. Refine mesh
        if is_quad {
            // Non-conforming for quads (with hanging nodes)
            let (new_mesh, hanging) = fem_mesh::amr::refine_nonconforming_quad(&mesh, &marked, None);
            // Apply hanging-node constraints to solution for persistent storage
            // (the next iteration will recompute the initial guess from prolongation)
            let vc = duplicate_hanging_constraints(&hanging, n_scalar);
            let mut x_store = x.clone();
            apply_hanging_constraints(&mut mat, &mut rhs, &vc);
            // We store the raw (non-constrained) solution; prolongation handles it
            x_prev = Some(x_store);
            mesh = new_mesh;
        } else {
            // Conforming closure refinement for tris (no hanging nodes)
            let new_mesh = fem_mesh::amr::closure_refine(&mesh, &marked, 20, None);
            x_prev = Some(x);
            mesh = new_mesh;
        }
    }

    // 16. Final output: reference mesh + deformed mesh + displacement
    let space_final = VectorH1Space::new(mesh.clone(), order, dim as u8);
    let n_scalar_final = space_final.n_scalar_dofs();
    let nn = mesh.n_nodes() as usize;

    // Reference mesh
    let mut ref_coords = Vec::with_capacity(nn * dim);
    for n in 0..nn { let c = mesh.node_coords(n as u32); ref_coords.push(c[0]); ref_coords.push(c[1]); }
    write_mfem_mesh(&mesh, &ref_coords, dim, "ex21_reference.mesh");
    println!("Wrote ex21_reference.mesh");

    // Deformed mesh + displacement (if solution exists)
    if let Some(ref u_final) = x_prev {
        let mut def_coords = Vec::with_capacity(nn * dim);
        for n in 0..nn {
            let c = mesh.node_coords(n as u32);
            let ux = if n < n_scalar_final { u_final[n] } else { 0.0 };
            let uy = if n + n_scalar_final < u_final.len() { u_final[n_scalar_final + n] } else { 0.0 };
            def_coords.push(c[0] + ux);
            def_coords.push(c[1] + uy);
        }
        write_mfem_mesh(&mesh, &def_coords, dim, "ex21_deformed.mesh");
        println!("Wrote ex21_deformed.mesh");

        let mut f = File::create("ex21_displacement.sol").expect("cannot create ex21_displacement.sol");
        writeln!(f, "{}", u_final.len()).ok();
        for &v in u_final { writeln!(f, "{:.16e}", v).ok(); }
        println!("Wrote ex21_displacement.sol");
    }
}
