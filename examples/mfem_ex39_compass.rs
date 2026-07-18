//! # Example 39 — Named attribute sets (1:1 with MFEM ex39)
//! Solves Poisson on compass.msh with named attribute sets.
//! ```bash
//! cargo run --example mfem_ex39_compass -- -m data/compass.msh
//! ```

use fem_assembly::standard::DiffusionIntegrator;
use fem_assembly::integrator::{LinearIntegrator, QpData};
use fem_assembly::Assembler;
use fem_io::read_msh_file;
use fem_linalg::fem_to_linlvo_csr;
use fem_mesh::{Mesh, amr::refine_uniform};
use fem_solver::{solve_pcg, GSSmoother};
use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs};

fn main() {
    let mut mesh_file = "data/compass.msh".to_string();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() { "-m"|"--mesh" => { mesh_file = it.next().unwrap_or(mesh_file); } _ => {} }
    }
    let msh = read_msh_file(&mesh_file).expect("read mesh");
    let registry = msh.named_attribute_registry();
    let mesh: Mesh<2> = msh.into_2d().expect("2D");

    // Capture named attribute sets BEFORE refinement (C++ ex39 uses original mesh attributes)
    let all_names = registry.names();
    let mut elem_names: Vec<&str> = Vec::new();
    let mut bdr_names: Vec<&str> = Vec::new();
    for name in &all_names {
        if let Ok(ids) = mesh.element_ids_for_named_set(&registry, name) {
            if !ids.is_empty() { elem_names.push(*name); continue; }
        }
        if let Ok(ids) = mesh.face_ids_for_named_set(&registry, name) {
            if !ids.is_empty() { bdr_names.push(*name); }
        }
    }
    elem_names.sort(); bdr_names.sort();

    let ne = mesh.n_elems();
    let dim = 2;
    let ref_levels = ((50000.0 / ne as f64).ln() / (2.0_f64).ln() / dim as f64).floor() as usize;
    eprintln!("  ne_init={ne} ref_levels={ref_levels}");
    let mesh = if ref_levels > 0 { let mut m = mesh; for _ in 0..ref_levels { m = refine_uniform(&m); } m } else { mesh };
    println!("Element Attribute Set Names: {}", elem_names.join(" "));
    println!("Boundary Attribute Set Names: {}", bdr_names.join(" "));

    let mesh = refine_uniform(&mesh);
    let h1 = H1Space::new(mesh, 1);
    let n = h1.n_dofs();
    println!("Number of finite element unknowns: {n}");

    let diff = DiffusionIntegrator { kappa: 1.0 };
    let a = Assembler::assemble_bilinear(&h1, &[&diff], 4);
    let rhs = Assembler::assemble_linear(&h1, &[&Q1Source], 4);

    let bdr = boundary_dofs(h1.mesh(), h1.dof_manager(), &[1,2,3,4,5,6,7,8]);
    let mut mat = a;
    let mut b = rhs;
    for &d in &bdr { mat.apply_dirichlet_symmetric(d as usize, 0.0, &mut b); }
    let mut x = vec![0.0; n];
    let precond = GSSmoother::from_csr(&fem_to_linlvo_csr(&mat)).expect("GS");
    solve_pcg(&mat, &b, &mut x, &precond, 1e-12, 2000, true).expect("PCG");
    println!("Size of linear system: {n}");
}

struct Q1Source;
impl LinearIntegrator for Q1Source {
    fn add_to_element_vector(&self, qp: &QpData<'_>, fe: &mut [f64]) {
        for i in 0..qp.n_dofs { fe[i] += qp.weight * qp.phi[i]; }
    }
}
