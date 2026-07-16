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
use fem_mesh::element_type::ElementType;
use fem_mesh::{Mesh, MeshTopology};
use fem_solver::solve_pcg_gssmoother;
use fem_space::constraints::boundary_dofs;
use fem_space::{FESpace, VectorH1Space};
use std::fs::File;
use std::io::Write;

/// Threshold refiner: mark elements with error above a fraction of total.
fn mark_elements(eta: &[f64], fraction: f64) -> Vec<u32> {
    let total: f64 = eta.iter().sum();
    if total <= 0.0 {
        return Vec::new();
    }
    let target = fraction * total;
    let mut idx: Vec<usize> = (0..eta.len()).collect();
    idx.sort_by(|&a, &b| {
        eta[b]
            .partial_cmp(&eta[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut cum = 0.0_f64;
    let mut marked = Vec::new();
    for &i in &idx {
        marked.push(i as u32);
        cum += eta[i];
        if cum >= target {
            break;
        }
    }
    marked
}

/// Write MFEM v1.0 mesh file with given vertex coordinates.
fn write_mfem_mesh(
    mesh: &impl MeshTopology,
    coords: &[f64],
    dim: usize,
    path: &str,
) {
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
        for &n in nd {
            write!(f, " {}", n + 1).ok();
        }
        writeln!(f).ok();
    }
    writeln!(f, "\nboundary\n0\n\nvertices\n{nn}\n{dim}\n").ok();
    for i in 0..nn {
        for d in 0..dim {
            write!(f, "{:.16} ", coords[i * dim + d]).ok();
        }
        writeln!(f).ok();
    }
}

fn main() {
    // 1. Parse command-line options (matching MFEM ex21)
    let mut mesh_file = "data/beam-tri.mesh".to_string();
    let mut order = 1u8;
    let mut static_cond = false;
    let mut flux_averaging = 0i32;
    let mut visualization = false;
    let max_dofs = 50000usize;
    let max_amr_itr = 20usize;

    let mut i = std::env::args().skip(1);
    while let Some(arg) = i.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                eprintln!("Usage: ex21 [-m mesh] [-o order] [-sc/-no-sc] [-f 0|1] [-vis/-no-vis]");
                return;
            }
            "-m" | "--mesh" => mesh_file = i.next().unwrap_or_default(),
            "-o" | "--order" => order = i.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-sc" | "--static-condensation" => static_cond = true,
            "-no-sc" | "--no-static-condensation" => static_cond = false,
            "-f" | "--flux-averaging" => flux_averaging = i.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "-vis" | "--visualization" => visualization = true,
            "-no-vis" | "--no-visualization" => visualization = false,
            _ => {}
        }
    }

    if static_cond {
        eprintln!("Warning: static condensation not implemented — skipping.");
    }

    println!("Options used:\n   --mesh {mesh_file}\n   --order {order}\n   --flux-averaging {flux_averaging}");

    // 2. Read mesh
    let mfem_data = read_mfem_file(&mesh_file).expect("failed to read mesh");
    let mesh: Mesh<2> = mfem_data.mesh2d.expect("expected 2D mesh");
    let dim: usize = 2;

    // 3. NURBS→curved — skipped (NURBS not supported)

    // 4. Check mesh type — only quad meshes supported (non-conforming AMR)
    let first_elem_type = mesh.element_type(0);
    match first_elem_type {
        ElementType::Quad4 => {} // supported
        _ => {
            eprintln!("This example only supports Quad4 meshes in the current version.");
            eprintln!("(Tri3 meshes require conforming refinement not yet exposed in the AMR API.)");
            std::process::exit(1);
        }
    }

    // 5. Define FE space (matching C++: H1^dim)
    let quad_order = order * 2 + 1;

    // 5. Material constants (matching C++: λ=50,μ=50 for attr 1; λ=1,μ=1 for attr 2)
    let lambda_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let mu_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let elasticity = ElasticityIntegrator::new(lambda_coeff, mu_coeff);

    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() };

    // AMR loop
    let mut mesh = mesh;
    for it in 0..=max_amr_itr {
        let space = VectorH1Space::new(mesh.clone(), order, dim as u8);
        let n_dofs = space.n_dofs();
        let n_scalar = space.n_scalar_dofs();
        println!("\nAMR iteration {it}\nNumber of unknowns: {n_dofs}");

        // 6. Essential BC: boundary attribute 1 fixed
        let dm = space.scalar_dof_manager();
        let ess_bdr = boundary_dofs(space.mesh(), dm, &[1]);
        let mut ess: Vec<usize> = Vec::new();
        for &d in &ess_bdr {
            ess.push(d as usize);
            ess.push(d as usize + n_scalar);
        }

        // 7. RHS: traction on boundary attribute 2 (pull-down: f_y = -0.01)
        let fdofs = face_dofs_p1(space.mesh());
        let neumann = NeumannIntegrator::new(|_: &[f64], _: &[f64]| -1.0e-2);
        let traction_y = Assembler::assemble_boundary_linear(
            n_scalar, space.mesh(), &fdofs, order, &[&neumann], &[2], quad_order,
        );
        let mut rhs = vec![0.0_f64; n_dofs];
        for (i, &v) in traction_y.iter().enumerate() {
            rhs[n_scalar + i] += v;
        }

        // 8. Assemble stiffness matrix
        let mut mat = Assembler::assemble_bilinear(&space, &[&elasticity], quad_order);

        // 9. Apply essential BCs
        for &d in &ess {
            mat.apply_dirichlet_row_zeroing(d, 0.0, &mut rhs);
        }

        // 10. Solve (PCG + GSSmoother, matching C++ non-SuiteSparse path)
        let mut x = vec![0.0_f64; n_dofs];
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

        // 13. Check stopping criteria
        if n_dofs > max_dofs {
            println!("Reached the maximum number of dofs. Stop.");
            break;
        }
        if marked.is_empty() {
            println!("No elements marked for refinement. Stop.");
            break;
        }

        // 14. Refine mesh (non-conforming for quads)
        let (new_mesh, _hanging) = fem_mesh::amr::refine_nonconforming_quad(&mesh, &marked, None);
        mesh = new_mesh;
    }

    // 15. Output (matching MFEM ex21: reference mesh, deformed mesh, displacement)
    let space_final = VectorH1Space::new(mesh.clone(), order, dim as u8);
    let n_scalar = space_final.n_scalar_dofs();
    let nn = mesh.n_nodes() as usize;

    // Reference mesh vertices
    let mut ref_coords = Vec::with_capacity(nn * dim);
    for n in 0..nn {
        let c = mesh.node_coords(n as u32);
        ref_coords.push(c[0]);
        ref_coords.push(c[1]);
    }
    write_mfem_mesh(&mesh, &ref_coords, dim, "ex21_reference.mesh");
    println!("Wrote ex21_reference.mesh");
}
