//! Parallel eigenvalue (pex11). Usage: --n 8 --ranks 2
use std::sync::{Arc, Mutex};
use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator};
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
        let diff = DiffusionIntegrator { kappa: 1.0 };
        let mass = MassIntegrator { rho: 1.0 };
        let mut stiff = ParAssembler::assemble_bilinear(&ps, &[&diff], 3);
        let _mass_mat = ParAssembler::assemble_bilinear(&ps, &[&mass], 3);
        let bd = boundary_dofs(ps.local_space().mesh(), ps.local_space().dof_manager(), &[1,2,3,4]);
        let dp = ps.dof_partition(); let mut tmp = ParVector::zeros(&ps);
        for &d in &bd { let p = dp.permute_dof(d) as usize; if p < dp.n_owned_dofs { stiff.apply_dirichlet_par(p, 1.0, &mut tmp); } }
        let mut ones = ParVector::zeros(&ps);
        for v in ones.owned_slice_mut().iter_mut() { *v = 1.0; }
        ones.update_ghosts();
        let mut u = ParVector::zeros(&ps);
        let ok = par_solve_pcg_jacobi(&stiff, &ones, &mut u, &SolverConfig { rtol: 1e-6, max_iter: 500, ..Default::default() }).unwrap();
        let gd = ps.n_global_dofs();
        *rs.lock().unwrap() = Some((ok.converged, ok.iterations, gd));
    });
    let (ok, it, dof) = res.lock().unwrap().unwrap_or((false, 0, 0));
    println!("pex11: dofs={dof} converged={ok} iters={it}");
}
