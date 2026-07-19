use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator};
use fem_mesh::{Mesh, amr::refine_uniform};
use fem_parallel::{WorkerConfig, launcher::native::ThreadLauncher, par_partition::partition_mesh, par_space::ParallelFESpace, ParAssembler};
use fem_space::{H1Space, L2Space};
fn main() {
    let a = Args::new();
    ThreadLauncher::new(WorkerConfig::new(a.np)).launch(move |comm| {
        let mut mesh = Mesh::<2>::unit_square_tri(a.n);
        for _ in 0..a.refs { mesh = refine_uniform(&mesh); }
        let pm = partition_mesh(&mesh, &comm);
        let lm = pm.local_mesh().clone();
        let h1 = H1Space::new(lm.clone(), 2); let l2 = L2Space::new(lm, 0);
        let ps0 = ParallelFESpace::new(h1, &pm, comm.clone());
        let ps1 = ParallelFESpace::new(l2, &pm, comm.clone());
        let _a00 = ParAssembler::assemble_bilinear(&ps0, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
        let _a11 = ParAssembler::assemble_bilinear(&ps1, &[&MassIntegrator { rho: 1.0 }], 3);
        if comm.is_root() { println!("pex36: obstacle H1/L2 block"); }
    });
}
struct Args { n: usize, refs: usize, np: usize }
impl Args { fn new() -> Self {
    let mut a = Self { n: 8, refs: 2, np: 2 };
    let mut i = std::env::args().skip(1);
    while let Some(arg) = i.next() { match arg.as_str() { "--n" => a.n = i.next().and_then(|s| s.parse().ok()).unwrap_or(8), "-r" => a.refs = i.next().and_then(|s| s.parse().ok()).unwrap_or(2), "--np" => a.np = i.next().and_then(|s| s.parse().ok()).unwrap_or(2), _ => {} } }
    a
}}
