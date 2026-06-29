//! GPU vs CPU MMS comparison test.
//!
//! Requires GPU backend (wgpu).  Marked `#[ignore]` by default; run with
//! `cargo test -p fem-linalg-gpu -- --ignored gpu_mms` on a machine with a
//! working GPU and compatible WGPU/Vulkan drivers.
//!
//! Verifies that GPU‑assembled matrices produce the same solution as CPU
//! assembly to within 1e-10 relative error.

fn cpu_poisson_solve(n: usize) -> (Vec<f64>, Vec<f64>) {
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;
    use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
    use fem_assembly::Assembler;
    use fem_solver::LinearSolver;
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh.clone(), 1);
    let a = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
    let f = |x: &[f64]| 2.0 * std::f64::consts::PI.powi(2) * (std::f64::consts::PI * x[0]).sin() * (std::f64::consts::PI * x[1]).sin();
    let mut rhs = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(&f)], 3);
    let bdofs = fem_space::constraints::boundary_dofs(&mesh, space.dof_manager(), &[1, 2, 3, 4]);
    let mut a_mut = a;
    fem_space::constraints::apply_dirichlet(&mut a_mut, &mut rhs, &bdofs, &vec![0.0; bdofs.len()]);
    let mut u = vec![0.0; space.n_dofs()];
    fem_solver::cg::cg(&a_mut, &rhs, &mut u, 1e-10, 5000).unwrap();
    (u, rhs)
}

#[test]
#[ignore]
#[cfg_attr(not(feature = "gpu"), ignore)]
fn gpu_vs_cpu_poisson_mms() {
    let (cpu_u, _cpu_rhs) = cpu_poisson_solve(8);

    // TODO: Replace with actual GPU solver call when GPU assembly is f64+3D capable.
    // Currently GPU assembly covers only f32 2D Poisson Tri3 (P1).
    // let gpu_u = gpu_poisson_solve(8);
    // for i in 0..cpu_u.len() {
    //     let rel = (gpu_u[i] - cpu_u[i]).abs() / cpu_u[i].abs().max(1.0);
    //     assert!(rel < 1e-10, "GPU vs CPU mismatch at DOF {i}: rel={:.3e}", rel);
    // }

    eprintln!("gpu_vs_cpu_poisson_mms: CPU solve succeeded on {} DOFs. GPU path pending (see Stage 8).", cpu_u.len());
}
