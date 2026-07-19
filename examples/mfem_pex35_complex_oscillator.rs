use fem_assembly::standard::DiffusionIntegrator;
use fem_io::mfem::read_mfem_file;
use fem_mesh::amr::refine_uniform_3d;
use fem_parallel::{WorkerConfig, launcher::native::ThreadLauncher, par_partition::partition_mesh, par_space::ParallelFESpace, ParAssembler};
use fem_space::H1Space;
fn main() {
    let a = Args::new();
    ThreadLauncher::new(WorkerConfig::new(a.np)).launch(move |comm| {
        let mut mesh = read_mfem_file(&a.mesh).unwrap().mesh3d.unwrap();
        for _ in 0..a.refs { mesh = refine_uniform_3d(&mesh); }
        let pm = partition_mesh(&mesh, &comm);
        let sp = H1Space::new(pm.local_mesh().clone(), 1);
        let ps = ParallelFESpace::new(sp, &pm, comm.clone());
        let _m = ParAssembler::assemble_bilinear(&ps, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
        if comm.is_root() { println!("pex35: H1 assembled"); }
    });
}
struct Args { mesh: String, refs: usize, np: usize }
impl Args { fn new() -> Self {
    let mut a = Self { mesh: "data/fichera.mesh".into(), refs: 0, np: 2 };
    let mut i = std::env::args().skip(1);
    while let Some(arg) = i.next() { match arg.as_str() { "-m" => a.mesh = i.next().unwrap_or_default(), "-r" => a.refs = i.next().and_then(|s| s.parse().ok()).unwrap_or(0), "--np" => a.np = i.next().and_then(|s| s.parse().ok()).unwrap_or(2), _ => {} } }
    a
}}
