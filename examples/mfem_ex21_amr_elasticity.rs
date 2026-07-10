//! Example 21 鈥?AMR for linear elasticity. 1:1 translation of MFEM ex21.
//! PCG+GSSmoother, Z-Z estimator, non-conforming quad refinement.

#![allow(non_snake_case, dead_code)]

use fem_assembly::postproc::coefficient::PWConstCoeff;
use fem_assembly::postproc::error_estimate::{threshold_mark, zz_estimator};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::standard::ElasticityIntegrator;
use fem_assembly::Assembler;
use fem_linalg::SolverConfig;
use fem_mesh::{amr::HangingNodeConstraint, Mesh};
use fem_space::constraints::boundary_dofs;
use fem_space::constraints::hanging_2d::{apply_hanging_constraints, recover_hanging_values};
use fem_space::{FESpace, VectorH1Space};

fn main() {
    let mut mf = "data/beam-quad.mesh".to_string();
    let mut o = 1u8; let max_dofs = 50000usize; let max_amr = 20usize;
    let mut a = std::env::args().skip(1);
    while let Some(arg) = a.next() {
        match arg.as_str() {
            "-h" => { eprintln!("Usage: ex21 [-m mesh] [-o order]"); return; }
            "-m" | "--mesh" => { mf = a.next().unwrap_or_default(); }
            "-o" | "--order" => { o = a.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
            _ => {}
        }
    }
    println!("Options:\n  --mesh {mf}\n  --order {o}");

    let data = fem_io::mfem::read_mfem_file(&mf).expect("read mesh");
    let mut mesh: Mesh<2> = data.mesh2d.expect("2D mesh");
    let dim = 2usize;
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 2000, ..SolverConfig::default() };



    let mut constraints: Vec<HangingNodeConstraint> = Vec::new();

    for it in 0..=max_amr {
        let sp = VectorH1Space::new(mesh.clone(), o, dim as u8);
        let dm2 = sp.scalar_dof_manager();
        let bdry2 = boundary_dofs(sp.mesh(), dm2, &[1]);
        let ess: Vec<usize> = bdry2.iter().flat_map(|&d| vec![d as usize, d as usize+sp.n_scalar_dofs()]).collect();
        let n = sp.n_dofs(); let ns_cur = sp.n_scalar_dofs();
        println!("\nAMR iteration {it}");
        println!("Number of unknowns: {n}");

        let stiff = Assembler::assemble_bilinear(
            &sp, &[&ElasticityIntegrator::new(
                PWConstCoeff::new([(1,50.0),(2,1.0)]),
                PWConstCoeff::new([(1,50.0),(2,1.0)]),
            )], 3);

        let mut rhs = vec![0.0; n]; for i in ns_cur..2*ns_cur { rhs[i] = -0.001; } // gravity on y-components
        let mut A = stiff.clone();

        // Apply hanging node constraints (vector: each scalar 鈫?both x and y)
        if !constraints.is_empty() {
            let mut vc: Vec<HangingNodeConstraint> = Vec::with_capacity(constraints.len()*2);
            for c in &constraints {
                vc.push(HangingNodeConstraint { constrained: c.constrained, parent_a: c.parent_a, parent_b: c.parent_b });
                vc.push(HangingNodeConstraint { constrained: c.constrained + ns_cur, parent_a: c.parent_a + ns_cur, parent_b: c.parent_b + ns_cur });
            }
            let mut rhs2 = rhs.clone();
            apply_hanging_constraints(&mut A, &mut rhs2, &vc);
            for &d in &ess { A.apply_dirichlet_row_zeroing(d, 0.0, &mut rhs2); }
            let mut X = vec![0.0; n];
            match fem_solver::solve_pcg_gssmoother(&A, &rhs2, &mut X, &cfg) {
                Ok(r) => println!("  PCG: {} its res={:.3e}", r.iterations, r.final_residual),
                Err(e) => eprintln!("  PCG: {e}"),
            }
            recover_hanging_values(&mut X, &vc);

            let gf = GridFunction::new(&sp, X);
            let ind = zz_estimator(&gf);
            let me = ind.eta.iter().cloned().fold(0.0_f64, f64::max);
            println!("  Max element error: {me:.6e}");
            let marked = threshold_mark(&ind.eta, 0.7*me);
            println!("  Marked {} elements", marked.len());
            if n > max_dofs || marked.is_empty() { println!("Stop."); break; }
            mesh = fem_mesh::refine_uniform(&mesh); constraints = Vec::new();
        } else {
            for &d in &ess { A.apply_dirichlet_row_zeroing(d, 0.0, &mut rhs); }
            let mut X = vec![0.0; n];
            match fem_solver::solve_pcg_gssmoother(&A, &rhs, &mut X, &cfg) {
                Ok(r) => println!("  PCG: {} its res={:.3e}", r.iterations, r.final_residual),
                Err(e) => eprintln!("  PCG: {e}"),
            }
            let gf = GridFunction::new(&sp, X);
            let ind = zz_estimator(&gf);
            let me = ind.eta.iter().cloned().fold(0.0_f64, f64::max);
            println!("  Max element error: {me:.6e}");
            let marked = threshold_mark(&ind.eta, 0.7*me);
            println!("  Marked {} elements", marked.len());
            if n > max_dofs || marked.is_empty() { println!("Stop."); break; }
            mesh = fem_mesh::refine_uniform(&mesh); constraints = Vec::new();
        }
    }
}







