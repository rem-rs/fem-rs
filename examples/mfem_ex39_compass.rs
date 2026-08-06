//! # Example 39 — Named attribute sets (1:1 with MFEM ex39)
//!
//! Solves `−∇·(κ ∇u) = 1_s` on `compass.msh` where:
//! - the source region `s` is given by name (`-src`, default `Rose Even`),
//! - the diffusion coefficient is a sum of three pieces:
//!   `1e-6` everywhere + `1.0` on `Base` + `2.0` on `Rose Even`
//!   (the `Base`/`Rose Even` markers come from named attribute sets),
//! - homogeneous Dirichlet BCs on the named boundary set `-ess` (default
//!   `Boundary`).
//!
//! This demonstrates fem-rs named attribute sets: reading them from the
//! GMSH file, creating new sets from existing ones
//! (`SetAttributeSet`/`AddToAttributeSet`), converting a set to a marker
//! array (`GetAttributeSetMarker`), and using markers in assembly and
//! essential-BC selection.
//!
//! ```bash
//! cargo run --example mfem_ex39_compass -- -no-vis
//! cargo run --example mfem_ex39_compass -- -ess "Southern Boundary" -no-vis
//! cargo run --example mfem_ex39_compass -- -src Base -no-vis
//! ```

use fem_assembly::assembler::Assembler;
use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
use fem_io::read_msh_file;
use fem_mesh::Mesh;
use fem_solver::{solve_pcg_gssmoother, SolverConfig};
use fem_space::constraints::boundary_dofs;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

fn main() {
    // 1. Parse command-line options (MFEM OptionsParser semantics).
    let mut mesh_file = "data/compass.msh".to_string();
    let mut order = 1;
    let mut source_name = "Rose Even".to_string();
    let mut ess_name = "Boundary".to_string();
    let mut visualization = true;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-m" | "--mesh" => mesh_file = args.next().expect("-m needs a value"),
            "-o" | "--order" => order = args.next().expect("-o needs a value").parse().expect("order"),
            "-src" | "--source-attr-name" => source_name = args.next().expect("-src needs a value"),
            "-ess" | "--ess-attr-name" => ess_name = args.next().expect("-ess needs a value"),
            "-vis" | "--visualization" => visualization = true,
            "-no-vis" | "--no-visualization" => visualization = false,
            _ => {}
        }
    }
    println!("Options used:");
    println!("   --mesh {}", mesh_file);
    println!("   --order {order}");
    println!("   --source-attr-name {}", source_name);
    println!("   --ess-attr-name {}", ess_name);
    if !visualization { println!("   --no-visualization"); }

    // 2. Read the mesh from the given GMSH mesh file.
    let msh = read_msh_file(&mesh_file).expect("read mesh");
    let mut registry = msh.named_attribute_registry();
    let mut bdr_registry = registry.clone();
    let mesh: Mesh<2> = msh.into_2d().expect("2D mesh");

    // 3. Refine the mesh: largest `ref_levels` with at most 50,000 elements.
    let dim = 2;
    let ne = mesh.n_elems();
    let ref_levels = ((50000.0 / ne as f64).ln() / (2.0_f64).ln() / dim as f64).floor() as usize;
    let mut mesh = mesh;
    // MFEM's Mesh(filename, 1, 1) constructor passes `refine = 1`, which
    // rotates each triangle so its longest edge is (0,1) BEFORE any uniform
    // refinement — this ordering affects the edge-midpoint numbering of every
    // refinement level (and hence the GS-sweep order), so replicate it.
    fem_mesh::amr::mark_tri_mesh_for_refinement(&mut mesh);
    for _ in 0..ref_levels {
        mesh = fem_mesh::amr::refine_uniform(&mesh);
    }

    // 4a. Display attribute set names contained in the initial mesh.
    // MFEM prints names sorted, each quoted with a leading space.
    // `registry` holds both element-side and boundary-side sets (a name may
    // appear in either side); element set names are those with element tags.
    let mut elem_names: Vec<&str> = registry.names()
        .into_iter()
        .filter(|n| !registry.element_set(n).is_empty())
        .collect();
    elem_names.sort();
    print!("Element Attribute Set Names: ");
    for n in &elem_names { print!(" \"{n}\""); }
    println!();

    let mut bdr_names: Vec<&str> = registry.names()
        .into_iter()
        .filter(|n| !registry.boundary_set(n).is_empty())
        .collect();
    bdr_names.sort();
    print!("Boundary Attribute Set Names: ");
    for n in &bdr_names { print!(" \"{n}\""); }
    println!();
    // 4b. Define new element regions based on existing element attribute sets.
    let n_even = registry.element_set("N Even").to_vec();
    let n_odd  = registry.element_set("N Odd").to_vec();
    let s_even = registry.element_set("S Even").to_vec();
    let s_odd  = registry.element_set("S Odd").to_vec();
    let e_even = registry.element_set("E Even").to_vec();
    let e_odd  = registry.element_set("E Odd").to_vec();
    let w_even = registry.element_set("W Even").to_vec();
    let w_odd  = registry.element_set("W Odd").to_vec();

    // North point = N Even ∪ N Odd
    registry.set_attribute_set("North", &n_even);
    registry.add_to_attribute_set("North", &n_odd);
    // South point
    registry.set_attribute_set("South", &s_even);
    registry.add_to_attribute_set("South", &s_odd);
    // East point
    registry.set_attribute_set("East", &e_even);
    registry.add_to_attribute_set("East", &e_odd);
    // West point
    registry.set_attribute_set("West", &w_even);
    registry.add_to_attribute_set("West", &w_odd);
    // "a" sides of the compass rose
    registry.set_attribute_set("Rose Even", &n_even);
    registry.add_to_attribute_set("Rose Even", &s_even);
    registry.add_to_attribute_set("Rose Even", &e_even);
    registry.add_to_attribute_set("Rose Even", &w_even);
    // "b" sides of the compass rose
    registry.set_attribute_set("Rose Odd", &n_odd);
    registry.add_to_attribute_set("Rose Odd", &s_odd);
    registry.add_to_attribute_set("Rose Odd", &e_odd);
    registry.add_to_attribute_set("Rose Odd", &w_odd);
    // Full compass rose
    let rose_even = registry.element_set("Rose Even").to_vec();
    let rose_odd  = registry.element_set("Rose Odd").to_vec();
    registry.set_attribute_set("Rose", &rose_even);
    registry.add_to_attribute_set("Rose", &rose_odd);

    // 4c. Define new boundary regions based on existing boundary sets.
    let nne = bdr_registry.boundary_set("NNE").to_vec();
    let nnw = bdr_registry.boundary_set("NNW").to_vec();
    let ene = bdr_registry.boundary_set("ENE").to_vec();
    let ese = bdr_registry.boundary_set("ESE").to_vec();
    let sse = bdr_registry.boundary_set("SSE").to_vec();
    let ssw = bdr_registry.boundary_set("SSW").to_vec();
    let wnw = bdr_registry.boundary_set("WNW").to_vec();
    let wsw = bdr_registry.boundary_set("WSW").to_vec();

    bdr_registry.set_boundary_attribute_set("Northern Boundary", &nne);
    bdr_registry.add_to_boundary_attribute_set("Northern Boundary", &nnw);
    bdr_registry.set_boundary_attribute_set("Southern Boundary", &sse);
    bdr_registry.add_to_boundary_attribute_set("Southern Boundary", &ssw);
    bdr_registry.set_boundary_attribute_set("Eastern Boundary", &ene);
    bdr_registry.add_to_boundary_attribute_set("Eastern Boundary", &ese);
    bdr_registry.set_boundary_attribute_set("Western Boundary", &wnw);
    bdr_registry.add_to_boundary_attribute_set("Western Boundary", &wsw);

    let n_bound = bdr_registry.boundary_set("Northern Boundary").to_vec();
    let s_bound = bdr_registry.boundary_set("Southern Boundary").to_vec();
    let e_bound = bdr_registry.boundary_set("Eastern Boundary").to_vec();
    let w_bound = bdr_registry.boundary_set("Western Boundary").to_vec();
    bdr_registry.set_boundary_attribute_set("Boundary", &n_bound);
    bdr_registry.add_to_boundary_attribute_set("Boundary", &s_bound);
    bdr_registry.add_to_boundary_attribute_set("Boundary", &e_bound);
    bdr_registry.add_to_boundary_attribute_set("Boundary", &w_bound);

    // 5. Define a finite element space on the mesh.
    let space = H1Space::new(mesh.clone(), order as u8);
    let n = space.n_dofs();
    println!("Number of finite element unknowns: {n}");

    // 6. Essential (Dirichlet) boundary dofs from the named boundary set.
    let max_bdr_attr = mesh.face_tags.iter().copied().max().unwrap_or(0);
    let ess_marker = bdr_registry.boundary_set_marker(&ess_name, max_bdr_attr);
    let ess_tags: Vec<i32> = (0..ess_marker.len() as i32)
        .filter(|&i| ess_marker[i as usize] == 1)
        .map(|i| i + 1)
        .collect();
    let ess_bdr = if ess_tags.is_empty() {
        Vec::new()
    } else {
        boundary_dofs(&mesh, space.dof_manager(), &ess_tags)
    };

    // 7. Linear form b: ∫ 1_s v dx with the source marker.
    let max_elem_attr = mesh.elem_tags.iter().copied().max().unwrap_or(0);
    let source_marker = registry.element_set_marker(&source_name, max_elem_attr);
    let source = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
    let qo = (2 * order + 1) as u8;
    let mut rhs = Assembler::assemble_linear_marked(
        &space,
        &[(&source, Some(source_marker.as_slice()))],
        qo,
    );

    // 8. Solution vector x — zero initial guess.
    let mut x = vec![0.0_f64; n];

    // 9. Bilinear form: κ = 1e-6 everywhere + 1.0 on Base + 2.0 on Rose Even.
    let default_coef = DiffusionIntegrator { kappa: 1.0e-6 };
    let base_coef = DiffusionIntegrator { kappa: 1.0 };
    let rose_coef = DiffusionIntegrator { kappa: 2.0 };
    let base_marker = registry.element_set_marker("Base", max_elem_attr);
    let rose_marker = registry.element_set_marker("Rose Even", max_elem_attr);
    let mut a = Assembler::assemble_bilinear_marked(
        &space,
        &[
            (&default_coef, None),
            (&base_coef, Some(base_marker.as_slice())),
            (&rose_coef, Some(rose_marker.as_slice())),
        ],
        qo,
    );

    // 10. Form the linear system with the essential BCs (MFEM FormLinearSystem).
    for &d in &ess_bdr {
        let du = d as usize;
        let mut dummy = vec![0.0; n];
        a.apply_dirichlet_symmetric(du, 0.0, &mut dummy);
        if let Some(k) = a.find_entry(du, du) { a.values[k] = 1.0; }
        rhs[du] = 0.0;
    }
    println!("Size of linear system: {}", a.nrows);

    // 11. Solve: PCG with symmetric Gauss-Seidel preconditioner
    //     (MFEM: PCG(A, M, B, X, 1, 800, 1e-12, 0.0) — the global PCG wrapper
    //     sets CGSolver rel_tol = sqrt(1e-12) = 1e-6).
    let cfg = SolverConfig {
        rtol: 1e-6, atol: 0.0, max_iter: 800, verbose: true,
        ..Default::default()
    };
    let _res = solve_pcg_gssmoother(&a, &rhs, &mut x, &cfg).expect("PCG");

    // 12. Recover the solution (already in x for non-hybrid assembly).

    // 13. Save the refined mesh and the solution.
    let _ = fem_io::mfem::write_mfem_file("refined.mesh", &mesh);
    let _ = fem_io::mfem::write_mfem_gf_file("sol.gf", 2, &x, "H1", order as u8, 1, 16);

    // 14. (GLVis visualization is not supported; prints nothing.)
    let _ = visualization;
}
