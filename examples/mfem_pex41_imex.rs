//! Example 41p — DG advection-diffusion assembly (parallel)
//!
//! ## Reference: MFEM ex41p

use fem_assembly::standard::MassIntegrator;
use fem_mesh::Mesh;
use fem_parallel::{
    WorkerConfig, launcher::native::ThreadLauncher,
    par_partition::partition_mesh, par_space::ParallelFESpace,
    ParAssembler,
};
use fem_space::L2Space;

fn main() {
    let args = parse_args();
    let launcher = ThreadLauncher::new(WorkerConfig::new(args.np));
    launcher.launch(move |comm| {
        let mesh = Mesh::<2>::unit_square_tri(args.n);
        let par_mesh = partition_mesh(&mesh, &comm);
        let local_mesh = par_mesh.local_mesh().clone();
        let space = L2Space::new(local_mesh.clone(), 1);
        let ps = ParallelFESpace::new(space, &par_mesh, comm.clone());
        let _m = ParAssembler::assemble_bilinear(&ps, &[&MassIntegrator { rho: 1.0 }], 3);
        if comm.is_root() { println!("pex41: DG assembly OK"); }
    });
}

struct Args { n: usize, np: usize }
fn parse_args() -> Args { Args { n: 8, np: 2 } }
