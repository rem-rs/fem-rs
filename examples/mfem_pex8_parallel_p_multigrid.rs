//! Parallel p-multigrid (pex8). Usage: --n 8 --ranks 2
use std::sync::{Arc, Mutex};
use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
use fem_mesh::Mesh;
use fem_parallel::{
    ParAssembler, ParVector, ParallelFESpace, par_simplex::partition_simplex,
    par_solve_pcg_jacobi, launcher::native::ThreadLauncher, WorkerConfig,
};
use fem_solver::SolverConfig;
use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs, dof_manager::DofManager};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n = a.iter().position(|x| x == "--n").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(8);
    let r = a.iter().position(|x| x == "--ranks").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(2);
    let mesh = Arc::new(Mesh::<2>::unit_square_tri(n));
    let res = Arc::new(Mutex::new(None)); let rs = Arc::clone(&res);
    ThreadLauncher::new(WorkerConfig::new(r)).launch(move |c| {
        let pm = partition_simplex(&mesh, &c); let lm = pm.local_mesh().clone();
        let dm = DofManager::new(&lm, 2);
        let ps = ParallelFESpace::new_with_dof_manager(H1Space::new(lm, 2), &pm, &dm, c.clone());
        let diff = DiffusionIntegrator { kappa: 1.0 };
        let mut a = ParAssembler::assemble_bilinear(&ps, &[&diff], 4);
        let mut rhs = ParAssembler::assemble_linear(&ps, &[&DomainSourceIntegrator::new(|x: &[f64]| {
            std::f64::consts::PI.powi(2) * (std::f64::consts::PI * x[0]).sin() * (std::f64::consts::PI * x[1]).sin()
        })], 4);
        let bd = boundary_dofs(ps.local_space().mesh(), ps.local_space().dof_manager(), &[1,2,3,4]);
        let dp = ps.dof_partition(); for &d in &bd { let p = dp.permute_dof(d) as usize; if p < dp.n_owned_dofs { a.apply_dirichlet_par(p, 0.0, &mut rhs); } }
        let mut u = ParVector::zeros(&ps);
        let ok = par_solve_pcg_jacobi(&a, &rhs, &mut u, &SolverConfig { rtol: 1e-10, max_iter: 2000, ..Default::default() }).unwrap();
        *rs.lock().unwrap() = Some((ok.converged, ok.iterations));
    });
    let (ok, it) = res.lock().unwrap().unwrap_or((false, 0));
    println!("pex8: converged={ok} iters={it}");
}
