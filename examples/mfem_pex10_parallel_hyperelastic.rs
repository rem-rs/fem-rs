//! Parallel hyperelastic (pex10) 3-D. Usage: --n 4 --ranks 2
use std::sync::{Arc, Mutex};
use fem_assembly::standard::VectorDiffusionIntegrator;
use fem_mesh::Mesh;
use fem_parallel::{
    ParAssembler, ParallelFESpace, par_partition::partition_mesh,
    launcher::native::ThreadLauncher, WorkerConfig,
};
use fem_space::H1Space;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n = a.iter().position(|x| x == "--n").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(4);
    let r = a.iter().position(|x| x == "--ranks").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(2);
    let mesh = Arc::new(Mesh::<3>::unit_cube_tet(n));
    let res = Arc::new(Mutex::new(None)); let rs = Arc::clone(&res);
    ThreadLauncher::new(WorkerConfig::new(r)).launch(move |c| {
        let pm = partition_mesh(&mesh, &c); let lm = pm.local_mesh().clone();
        let ps = ParallelFESpace::new(H1Space::new(lm, 1), &pm, c.clone());
        let vd = VectorDiffusionIntegrator { kappa: 1.0 };
        let _mat = ParAssembler::assemble_bilinear(&ps, &[&vd], 3);
        *rs.lock().unwrap() = Some(ps.n_global_dofs());
    });
    println!("pex10: dofs={}", res.lock().unwrap().unwrap_or(0));
}
