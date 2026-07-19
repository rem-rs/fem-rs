//! Example 39p — Named attribute sets (parallel Poisson on unit square)
//!
//! ## Reference: MFEM ex39p

use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
use fem_mesh::Mesh;
use fem_parallel::{
    WorkerConfig, launcher::native::ThreadLauncher,
    par_partition::partition_mesh, par_space::ParallelFESpace,
    ParAssembler, ParVector, par_solve_pcg_jacobi,
};
use fem_solver::SolverConfig;
use fem_space::{H1Space, constraints::boundary_dofs};

fn main() {
    let args = parse_args();
    let launcher = ThreadLauncher::new(WorkerConfig::new(args.np));
    launcher.launch(move |comm| {
        let mesh = Mesh::<2>::unit_square_tri(args.n);
        let par_mesh = partition_mesh(&mesh, &comm);
        let local_mesh = par_mesh.local_mesh().clone();
        let space = H1Space::new(local_mesh.clone(), 1);
        let ps = ParallelFESpace::new(space, &par_mesh, comm.clone());
        let dm = ps.local_space().dof_manager();
        let ess = boundary_dofs(&local_mesh, dm, &[1]);

        let mut a = ParAssembler::assemble_bilinear(&ps, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
        let mut rhs = ParAssembler::assemble_linear(&ps, &[&DomainSourceIntegrator::new(|_: &[f64]| 1.0)], 3);
        for &d in &ess { a.apply_dirichlet_par(d as usize, 0.0, &mut rhs); }
        let mut u = ParVector::zeros_like(&rhs);
        let _ = par_solve_pcg_jacobi(&a, &rhs, &mut u, &SolverConfig::default());
        if comm.is_root() { println!("pex39: ||u|| = {:.6e}", u.global_norm()); }
    });
}

struct Args { n: usize, np: usize }
fn parse_args() -> Args {
    let mut a = Args { n: 16, np: 2 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => a.n = it.next().and_then(|s| s.parse().ok()).unwrap_or(16),
            "--np" => a.np = it.next().and_then(|s| s.parse().ok()).unwrap_or(2),
            _ => {}
        }
    }
    a
}
