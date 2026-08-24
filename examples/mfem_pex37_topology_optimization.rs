//!
//! Parallel topology optimization baseline (pex37).
//!
//! Multi-material cantilever beam: -div(sigma(u)) = f with SIP-DG.
//! Matches MFEM ex37p core static solve.
//!
//! ## Parallel layout (pex12 pattern)
//! - serial mesh read on every rank, `-rs` serial uniform refinements;
//! - `partition_mesh` + `-rp` parallel uniform refinements;
//! - VectorH1Space byNODES (MFEM block layout);
//! - `ParAssembler` + AMG V-cycle (C++ uses BoomerAMG).

use fem_assembly::standard::ElasticityIntegrator;
use fem_io::mfem::read_mfem_file;
use fem_linalg::CooMatrix;
use fem_mesh::amr::refine_uniform;
use fem_mesh::Mesh;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_refine::par_uniform_refine;
use fem_parallel::par_solve_pcg_amg;
use fem_parallel::{
    ParAmgConfig, ParAssembler, ParVector, ParallelFESpace, SmootherType, WorkerConfig,
};
use fem_solver::SolverConfig;
use fem_space::{VectorH1Space, fe_space::FESpace};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_workers = args.iter().position(|a| a == "--ranks").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(1);
    let mesh_file = args.iter().position(|a| a == "-m").and_then(|i| args.get(i + 1)).map(|s| s.as_str()).unwrap_or("data/beam-tri.mesh");
    let ser_ref = args.iter().position(|a| a == "-rs").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(2);
    let par_ref = args.iter().position(|a| a == "-rp").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(0);
    let order: u8 = args.iter().position(|a| a == "-o").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(1);

    let mfem = read_mfem_file(mesh_file).expect("failed to read mesh");
    let mut mesh: Mesh<2> = mfem.mesh2d.expect("expected 2D mesh");
    let dim = 2usize;
    for _ in 0..ser_ref { mesh = refine_uniform(&mesh); }
    let mesh = std::sync::Arc::new(mesh);

    let result = std::sync::Arc::new(std::sync::Mutex::new(None));
    let result_slot = result.clone();
    let mesh_arc = mesh.clone();

    ThreadLauncher::new(WorkerConfig::new(n_workers)).launch(move |comm| {
        let rank = comm.rank();
        let par_mesh = partition_mesh(&mesh_arc, &comm);
        let local_mesh = par_mesh.local_mesh().clone();

        let vec_space = VectorH1Space::new(local_mesh.clone(), order, dim as u8);
        let ps = ParallelFESpace::new_vector(vec_space.clone(), &par_mesh, dim, comm.clone());
        let n_owned = ps.dof_partition().n_owned_dofs;
        let ghost = ps.dof_ghost_exchange_arc();

        let n_elem = local_mesh.n_elems() as usize;
        let mut lambda = vec![1.0f64; n_elem];
        let mut mu = vec![1.0f64; n_elem];
        for e in local_mesh.elem_iter() {
            let attr = local_mesh.elem_tags[e as usize];
            if attr == 1 { lambda[e as usize] = 50.0; mu[e as usize] = 50.0; }
        }

        let qo = 2 * order + 1;
        let integ = ElasticityIntegrator::new(
            fem_assembly::postproc::coefficient::VecCoeff::from_values(lambda.clone()),
            fem_assembly::postproc::coefficient::VecCoeff::from_values(mu.clone()),
        );
        let a_local = ParAssembler::assemble_bilinear(&ps, &[&integ], qo);
        let a_mat = fem_parallel::ParCsrMatrix::from_local_matrix(&a_local, n_owned, ghost.clone(), comm.clone());

        let mut rhs = ParVector::zeros(&ps);

        let scalar_dm = ps.local_space().scalar_dof_manager();
        let n_scalar = ps.local_space().n_scalar_dofs();
        let bnd_scalar = fem_space::constraints::boundary_dofs(ps.local_space().mesh(), scalar_dm, &[1, 2]);
        let mut clamped: Vec<usize> = Vec::with_capacity(bnd_scalar.len() * 2);
        for &d in &bnd_scalar {
            clamped.push(d as usize);
            clamped.push(d as usize + n_scalar);
        }
        a_mat.eliminate_diag_symmetric(&clamped, 1.0);
        for &pid in &clamped {
            if pid < n_owned { rhs.as_slice_mut()[pid] = 0.0; }
        }

        if rank == 0 { println!("Number of unknowns: {}", ps.n_global_dofs()); }

        let amg_cfg = ParAmgConfig {
            smoother: SmootherType::SymmetricGaussSeidel,
            n_pre_smooth: 2, n_post_smooth: 2,
            smoothed_prolongation: true, block_size: dim,
            use_global_aggregation: true, ..ParAmgConfig::default()
        };
        let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() };

        let mut u = ParVector::zeros(&ps);
        let res = par_solve_pcg_amg(&a_mat, &rhs, &mut u, &amg_cfg, &cfg).expect("solve failed");

        let sol_norm = u.global_norm();
        let sol_sum = comm.allreduce_sum_f64(u.as_slice()[..n_owned].iter().sum::<f64>());
        let checksum = comm.allreduce_sum_f64(
            (0..n_owned).map(|pid| (ps.dof_partition().global_dof(pid as u32) as f64 + 1.0) * u.as_slice()[pid]).sum::<f64>()
        );

        if rank == 0 {
            *result_slot.lock().unwrap() = Some((ps.n_global_dofs(), res.iterations, res.final_residual, sol_norm, sol_sum, checksum));
        }
    });

    if let Some((dofs, iters, residual, norm, sum, checksum)) = *result.lock().unwrap() {
        println!("Number of unknowns: {dofs}");
        println!("  PCG: {iters} iters, residual = {residual:.3e}");
        println!("  ||u|| = {norm:.6}, sum = {sum:.6}, checksum = {checksum:.6}");
    }
}
