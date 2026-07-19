//! Example 31p — Electromagnetic diffusion, restricted H(curl) (parallel)
//!
//! ## Reference: MFEM ex31p

use fem_assembly::standard::DiffusionIntegrator;
use fem_mesh::Mesh;
use fem_parallel::{
    WorkerConfig, launcher::native::ThreadLauncher,
    par_partition::partition_mesh, par_space::ParallelFESpace,
    ParAssembler,
};
use fem_space::RestrictedHCurlSpace;

fn main() {
    let args = parse_args();
    let launcher = ThreadLauncher::new(WorkerConfig::new(args.np));
    launcher.launch(move |comm| {
        let mesh = Mesh::<2>::unit_square_tri(args.n);
        let par_mesh = partition_mesh(&mesh, &comm);
        let local_mesh = par_mesh.local_mesh().clone();
        let space = RestrictedHCurlSpace::new(local_mesh.clone(), 1, 3);
        let ps = ParallelFESpace::new(space, &par_mesh, comm.clone());
        let _a = ParAssembler::assemble_bilinear(&ps, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
        if comm.is_root() { println!("pex31: restricted H(curl) OK"); }
    });
}

struct Args { n: usize, np: usize }
fn parse_args() -> Args { Args { n: 8, np: 2 } }
