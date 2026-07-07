//! Parallel PML Helmholtz (pex25). Usage: --n 12 --ranks 2 --k 5.0
use std::sync::{Arc, Mutex};
use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator, DomainSourceIntegrator};
use fem_mesh::Mesh;
use fem_parallel::{
    ParAssembler, ParVector, ParallelFESpace, par_partition::partition_mesh,
    par_solve_gmres_jacobi, launcher::native::ThreadLauncher, WorkerConfig,
};
use fem_solver::SolverConfig;
use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n = a.iter().position(|x| x == "--n").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(12);
    let r = a.iter().position(|x| x == "--ranks").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(2);
    let k = a.iter().position(|x| x == "--k").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(5.0);
    let mesh = Arc::new(Mesh::<2>::unit_square_tri(n));
    let res = Arc::new(Mutex::new(None)); let rs = Arc::clone(&res);
    ThreadLauncher::new(WorkerConfig::new(r)).launch(move |c| {
        let pm = partition_mesh(&mesh, &c); let lm = pm.local_mesh().clone();
        let ps = ParallelFESpace::new(H1Space::new(lm, 1), &pm, c.clone());
        // Helmholtz: −Δu − k²u = f
        let mut a = ParAssembler::assemble_bilinear(&ps, &[
            &DiffusionIntegrator { kappa: 1.0 },
            &MassIntegrator { rho: -(k * k) },
        ], 3);
        let mut rhs = ParAssembler::assemble_linear(&ps, &[&DomainSourceIntegrator::new(|x: &[f64]| {
            (-10.0 * ((x[0]-0.5).powi(2) + (x[1]-0.5).powi(2))).exp()
        })], 3);
        let bd = boundary_dofs(ps.local_space().mesh(), ps.local_space().dof_manager(), &ps.local_space().mesh().unique_boundary_tags());
        let dp = ps.dof_partition(); for &d in &bd { let p = dp.permute_dof(d) as usize; if p < dp.n_owned_dofs { a.apply_dirichlet_par(p, 0.0, &mut rhs); } }
        let mut u = ParVector::zeros(&ps);
        let ok = par_solve_gmres_jacobi(&a, &rhs, &mut u, 30, &SolverConfig { rtol: 1e-8, max_iter: 3000, ..Default::default() }).unwrap();
        *rs.lock().unwrap() = Some((ok.converged, ok.iterations, ps.n_global_dofs()));
    });
    let (ok, it, dof) = res.lock().unwrap().unwrap_or((false, 0, 0));
    println!("pex25(k={k}): dofs={dof} converged={ok} iters={it}");
}
