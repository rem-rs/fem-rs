//! Example 37p — Topology optimization baseline (parallel)
//!
//! ## Reference: MFEM ex37p

use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
use fem_mesh::{Mesh, amr::refine_uniform, topology::MeshTopology};
use fem_parallel::{
    WorkerConfig, launcher::native::ThreadLauncher,
    par_partition::partition_mesh, par_space::ParallelFESpace,
    ParAssembler, ParVector, par_solve_pcg_jacobi,
};
use fem_solver::SolverConfig;
use fem_space::{H1Space, constraints::boundary_dofs};

fn main() {
    let args = parse_args();
    let launcher = ThreadLauncher::new(WorkerConfig::new(args.np));
    launcher.launch(move |comm| {
        let mut mesh = Mesh::make_cartesian_2d(3, 1, 3.0, 1.0);
        // Remap left edge to tag 1, others to tag 2
        {
            let mut new_tags = Vec::new();
            for bf in 0..mesh.n_faces() {
                let nds = mesh.bface_nodes(bf as u32);
                let avg_x = nds.iter().map(|&n| mesh.node_coords(n)[0]).sum::<f64>() / nds.len() as f64;
                new_tags.push(if (avg_x - 0.0).abs() < 1e-10 { 1 } else { 2 });
            }
            mesh.face_tags = new_tags;
        }
        for _ in 0..args.refs { mesh = refine_uniform(&mesh); }
        let par_mesh = partition_mesh(&mesh, &comm);
        let local_mesh = par_mesh.local_mesh().clone();
        let space = H1Space::new(local_mesh.clone(), 1);
        let ps = ParallelFESpace::new(space, &par_mesh, comm.clone());
        let dm = ps.local_space().dof_manager();

        let ess = boundary_dofs(&local_mesh, dm, &[1]);
        let mut a = ParAssembler::assemble_bilinear(&ps, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
        let mut rhs = ParAssembler::assemble_linear(&ps, &[&DomainSourceIntegrator::new(|x: &[f64]|
            if (x[0] - 2.9).powi(2) + (x[1] - 0.5).powi(2) <= 0.05_f64.powi(2) { 1.0 } else { 0.0 }
        )], 3);
        for &d in &ess { a.apply_dirichlet_par(d as usize, 0.0, &mut rhs); }
        let mut u = ParVector::zeros_like(&rhs);
        let _ = par_solve_pcg_jacobi(&a, &rhs, &mut u, &SolverConfig::default());
        if comm.is_root() { println!("pex37: ||u|| = {:.6e}", u.global_norm()); }
    });
}

struct Args { refs: usize, np: usize }
fn parse_args() -> Args {
    let mut a = Args { refs: 2, np: 2 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-r" | "--refine" => a.refs = it.next().and_then(|s| s.parse().ok()).unwrap_or(2),
            "--np" => a.np = it.next().and_then(|s| s.parse().ok()).unwrap_or(2),
            _ => {}
        }
    }
    a
}
