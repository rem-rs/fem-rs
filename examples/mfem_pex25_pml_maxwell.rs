use fem_parallel::{WorkerConfig, launcher::native::ThreadLauncher, par_partition::partition_mesh, par_space::ParallelFESpace};
use fem_space::HCurlSpace;
fn main() {
    let a = Args::new();
    ThreadLauncher::new(WorkerConfig::new(a.np)).launch(move |comm| {
        let mesh = fem_mesh::Mesh::<3>::unit_cube_hex(2);
        let pm = partition_mesh(&mesh, &comm);
        let sp = HCurlSpace::new(pm.local_mesh().clone(), 1);
        let _ps = ParallelFESpace::new(sp, &pm, comm.clone());
        if comm.is_root() { println!("pex34: HCurl space created"); }
    });
}
struct Args { np: usize }
impl Args { fn new() -> Self {
    let mut a = Self { np: 2 };
    let mut i = std::env::args().skip(1);
    while let Some(arg) = i.next() { match arg.as_str() { "--np" => a.np = i.next().and_then(|s| s.parse().ok()).unwrap_or(2), _ => {} } }
    a
}}
