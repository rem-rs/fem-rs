use fem_assembly::standard::DiffusionIntegrator;
use fem_mesh::{Mesh, amr::refine_uniform};
use fem_parallel::{WorkerConfig, launcher::native::ThreadLauncher, par_partition::partition_mesh, par_space::ParallelFESpace, ParAssembler};
use fem_space::{HDivSpace, L2Space};
fn main() {
    let a = Args::new();
    ThreadLauncher::new(WorkerConfig::new(a.np)).launch(move |comm| {
        let mut mesh = Mesh::<2>::unit_square_tri(a.n);
        for _ in 0..a.refs { mesh = refine_uniform(&mesh); }
        let pm = partition_mesh(&mesh, &comm);
        let lm = pm.local_mesh().clone();
        let v = HDivSpace::new(lm.clone(), 1); let s = L2Space::new(lm, 0);
        let pv = ParallelFESpace::new(v, &pm, comm.clone());
        let _ps = ParallelFESpace::new(s, &pm, comm.clone());
        let _m = ParAssembler::assemble_bilinear(&pv, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
        if comm.is_root() { println!("pex40: HDiv/L2 assembled"); }
    });
}
struct Args { n: usize, refs: usize, np: usize }
impl Args { fn new() -> Self {
    let mut a = Self { n: 8, refs: 2, np: 2 };
    let mut i = std::env::args().skip(1);
    while let Some(arg) = i.next() { match arg.as_str() { "--n" => a.n = i.next().and_then(|s| s.parse().ok()).unwrap_or(8), "-r" => a.refs = i.next().and_then(|s| s.parse().ok()).unwrap_or(2), "--np" => a.np = i.next().and_then(|s| s.parse().ok()).unwrap_or(2), _ => {} } }
    a
}}
