//! Example 29p — Poisson on a 2D surface embedded in 3D (parallel)
//!
//! ## Reference
//! MFEM ex29p: https://github.com/mfem/mfem/blob/master/examples/ex29p.cpp

use fem_assembly::{
    Assembler, standard::{DiffusionIntegrator, DomainSourceIntegrator},
    boundary::surface::{SurfaceDiffusionIntegrator, SurfaceAssembler},
};
use fem_mesh::Mesh;
use fem_parallel::{
    WorkerConfig, launcher::native::ThreadLauncher,
    par_partition::partition_mesh, par_space::ParallelFESpace,
    ParAssembler, ParVector, par_solve_pcg_jacobi,
};
use fem_solver::SolverConfig;
use fem_space::{H1Space, constraints::boundary_dofs, fe_space::FESpace};

fn main() {
    let args = parse_args();
    let launcher = ThreadLauncher::new(WorkerConfig::new(args.np));
    launcher.launch(move |comm| {
        let mesh = Mesh::<3>::cylinder_surface(8, 8); // 2D surface mesh in R³
        let par_mesh = partition_mesh(&mesh, &comm);
        let local_mesh = par_mesh.local_mesh().clone();

        let order = 1u8; let qo = 3;
        let space = H1Space::new(local_mesh.clone(), order);
        let par_space = ParallelFESpace::new(space, &par_mesh, comm.clone());
        let dm = par_space.local_space().dof_manager();

        let ess = boundary_dofs(&local_mesh, dm, &local_mesh.unique_boundary_tags());
        let a = ParAssembler::assemble_bilinear(&par_space, &[&DiffusionIntegrator { kappa: 1.0 }], qo);
        let mut rhs = ParAssembler::assemble_linear(&par_space, &[&DomainSourceIntegrator(|_| 1.0)], qo);
        let mut u = ParVector::zeros_like(&rhs);
        let cfg = SolverConfig { rtol: 1e-8, max_iter: 5000, ..SolverConfig::default() };
        let _ = par_solve_pcg_jacobi(&a, &rhs, &mut u, &cfg);
        if comm.is_root() { println!("pex29: solved on 2D surface, ||u|| = {:.6e}", u.global_norm()); }
    });
}

struct Args { np: usize }
fn parse_args() -> Args { Args { np: 2 } }
