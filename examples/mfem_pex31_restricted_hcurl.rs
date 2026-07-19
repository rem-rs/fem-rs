use fem_mesh::Mesh;
use fem_parallel::{WorkerConfig, launcher::native::ThreadLauncher, par_partition::partition_mesh, par_space::ParallelFESpace};
use fem_space::RestrictedHCurlSpace;
fn main() {
    let a = Args::new();
    ThreadLauncher::new(WorkerConfig::new(a.np)).launch(move |comm| {
        let mesh = Mesh::<2>::unit_square_tri(a.n);
        let pm = partition_mesh(&mesh, &comm);
        let sp = RestrictedHCurlSpace::new(pm.local_mesh().clone(), 1, 3);
        let _ps = ParallelFESpace::new(sp, &pm, comm.clone());
        if comm.is_root() { println!("pex31: restricted H(Curl) created"); }
    });
}
struct Args { n: usize, np: usize }
impl Args { fn new() -> Self {
    let mut a = Self { n: 8, np: 2 };
    let mut i = std::env::args().skip(1);
    while let Some(arg) = i.next() { match arg.as_str() { "--n" => a.n = i.next().and_then(|s| s.parse().ok()).unwrap_or(8), "--np" => a.np = i.next().and_then(|s| s.parse().ok()).unwrap_or(2), _ => {} } } a
}}
