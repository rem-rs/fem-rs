//! Parallel incompressible hyperelastic (pex19). Usage: --n 8 --ranks 2
use std::sync::{Arc, Mutex};
use fem_assembly::standard::VectorDiffusionIntegrator;
use fem_mesh::Mesh;
use fem_parallel::{
    ParAssembler, ParVector, ParallelFESpace, par_partition::partition_mesh,
    par_solve_pcg_jacobi, launcher::native::ThreadLauncher, WorkerConfig,
};
use fem_solver::SolverConfig;
use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n = a.iter().position(|x| x == "--n").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(8);
    let r = a.iter().position(|x| x == "--ranks").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(2);
    let mesh = Arc::new(Mesh::<2>::unit_square_tri(n));
    let res = Arc::new(Mutex::new(None)); let rs = Arc::clone(&res);
    ThreadLauncher::new(WorkerConfig::new(r)).launch(move |c| {
        let pm = partition_mesh(&mesh, &c); let lm = pm.local_mesh().clone();
        let ps = ParallelFESpace::new(H1Space::new(lm, 1), &pm, c.clone());
        let vd = VectorDiffusionIntegrator { kappa: 1.0 };
        let mut a = ParAssembler::assemble_bilinear(&ps, &[&vd], 3);
        let mut u = ParVector::zeros(&ps);
        let mut rhs = ParVector::zeros(&ps);
        for v in rhs.owned_slice_mut().iter_mut() { *v = 1.0; }
        rhs.update_ghosts();
        let bd = boundary_dofs(ps.local_space().mesh(), ps.local_space().dof_manager(), &[1,2,3,4]);
        let dp = ps.dof_partition(); for &d in &bd { let p = dp.permute_dof(d) as usize; if p < dp.n_owned_dofs { a.apply_dirichlet_par(p, 0.0, &mut rhs); } }
        let ok = par_solve_pcg_jacobi(&a, &rhs, &mut u, &SolverConfig { rtol: 1e-6, max_iter: 1000, ..Default::default() }).unwrap();
        *rs.lock().unwrap() = Some((ok.converged, ok.iterations, ps.n_global_dofs()));
    });
    let (ok, it, dof) = res.lock().unwrap().unwrap_or((false, 0, 0));
    println!("pex19: dofs={dof} converged={ok} iters={it}");
}
