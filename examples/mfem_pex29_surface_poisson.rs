use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
use fem_mesh::Mesh;
use fem_parallel::{
    WorkerConfig, launcher::native::ThreadLauncher,
    par_partition::partition_mesh, par_space::ParallelFESpace,
    ParAssembler, ParVector, par_solve_pcg_jacobi,
};
use fem_solver::SolverConfig;
use fem_space::{H1Space, constraints::boundary_dofs};
fn main() {
    let a = Args::new();
    ThreadLauncher::new(WorkerConfig::new(a.np)).launch(move |comm| {
        let mesh = Mesh::<2>::unit_square_tri(a.n);
        let pm = partition_mesh(&mesh, &comm);
        let lm = pm.local_mesh().clone();
        let sp = H1Space::new(lm.clone(), 1);
        let ps = ParallelFESpace::new(sp, &pm, comm.clone());
        let dm = ps.local_space().dof_manager();
        let ess = boundary_dofs(&lm, dm, &lm.unique_boundary_tags());
        let mut a_mat = ParAssembler::assemble_bilinear(&ps, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
        let mut rhs = ParAssembler::assemble_linear(&ps, &[&DomainSourceIntegrator::new(|_: &[f64]| 1.0)], 3);
        for &d in &ess { a_mat.apply_dirichlet_par(d as usize, 0.0, &mut rhs); }
        let mut u = ParVector::zeros_like(&rhs);
        let _ = par_solve_pcg_jacobi(&a_mat, &rhs, &mut u, &SolverConfig::default());
        if comm.is_root() { println!("pex29: surface Poisson, ||u||={:.6e}", u.global_norm()); }
    });
}
struct Args { n: usize, np: usize }
impl Args { fn new() -> Self {
    let mut a = Self { n: 16, np: 2 };
    let mut i = std::env::args().skip(1);
    while let Some(arg) = i.next() { match arg.as_str() { "--n" => a.n = i.next().and_then(|s| s.parse().ok()).unwrap_or(16), "--np" => a.np = i.next().and_then(|s| s.parse().ok()).unwrap_or(2), _ => {} } }
    a
}}
