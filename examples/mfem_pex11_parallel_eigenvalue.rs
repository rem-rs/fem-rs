//!
//! Parallel Example 11 — Laplacian Eigenvalue (1:1 port of MFEM ex11p.cpp).
//!
//! Solves -Delta u = lambda u with homogeneous Dirichlet BCs using LOBPCG.
//!
//! Usage:
//!   cargo run --release --example mfem_pex11_parallel_eigenvalue -- --ranks 2
//!   cargo run --release --example mfem_pex11_parallel_eigenvalue -- --ranks 4 -n 5

use std::sync::Arc;

use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator};
use fem_io::mfem::read_mfem_file;
use fem_linalg::CooMatrix;
use fem_mesh::{refine_uniform, Mesh};
use fem_parallel::{
    par_partition::partition_mesh,
    launcher::native::ThreadLauncher,
    ParAssembler, ParallelFESpace, ParVector, ParCsrMatrix, ParAmgHierarchy, ParAmgConfig, SmootherType,
    par_lobpcg,
    WorkerConfig,
};
use fem_space::{H1Space, fe_space::FESpace};
use fem_space::constraints::boundary_dofs;
use fem_space::dof_manager::DofManager;

struct Args {
    mesh_file: String,
    order: u8,
    ranks: usize,
    nev: usize,
}

fn parse_args() -> Args {
    let mut mesh_file = "data/star.mesh".to_string();
    let mut order = 1u8;
    let mut ranks = 2usize;
    let mut nev = 5usize;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => mesh_file = it.next().unwrap_or(mesh_file),
            "-o" | "--order" => order = it.next().and_then(|s| s.parse().ok()).unwrap_or(order),
            "--ranks" => ranks = it.next().and_then(|s| s.parse().ok()).unwrap_or(ranks),
            "-n" | "--num-eigs" => nev = it.next().and_then(|s| s.parse().ok()).unwrap_or(nev),
            _ => {}
        }
    }
    Args { mesh_file, order, ranks, nev }
}

fn eliminate_ess_diag(a: &ParCsrMatrix, ess: &[usize], diag_val: f64) -> ParCsrMatrix {
    let no = a.n_owned();
    let nt = no + a.n_ghost();
    let mut coo = CooMatrix::new(nt, nt);
    for r in 0..a.n_owned() {
        let d = a.diag_block();
        for k in d.row_ptr[r]..d.row_ptr[r + 1] {
            coo.add(r, d.col_idx[k] as usize, d.values[k]);
        }
        let o = a.offd_block();
        for k in o.row_ptr[r]..o.row_ptr[r + 1] {
            coo.add(r, (o.col_idx[k] as usize) + no, o.values[k]);
        }
    }
    let mut loc = coo.into_csr();
    for &p in ess {
        loc.eliminate_essential_bc_diag_symmetric(p, diag_val);
    }
    ParCsrMatrix::from_local_matrix(&loc, no, a.ghost_exchange_arc(), a.comm().clone())
}

fn main() {
    let args = parse_args();

    // Read mesh and refine
    let reader = read_mfem_file(&args.mesh_file).expect("Failed to read mesh");
    let mut mesh: Mesh<2> = reader.mesh2d.expect("Mesh must be 2D");
    for _ in 0..2 {
        mesh = refine_uniform(&mesh);
    }

    let launcher = ThreadLauncher::new(WorkerConfig::new(args.ranks));

    launcher.launch(move |comm| {
        let rank = comm.rank();

        // Partition mesh
        let par_mesh = partition_mesh(&mesh, &comm);

        // Create H1 space
        let local_mesh = par_mesh.local_mesh().clone();
        let dm = DofManager::new(&local_mesh, args.order);
        let local_space = H1Space::new(local_mesh, args.order);
        let par_space = if args.order >= 2 {
            ParallelFESpace::new_with_dof_manager(local_space, &par_mesh, &dm, comm.clone())
        } else {
            ParallelFESpace::new(local_space, &par_mesh, comm.clone())
        };

        let par_space = Arc::new(par_space);

        // Assemble Laplacian (A) and Mass (M) matrices
        let diff = DiffusionIntegrator { kappa: 1.0 };
        let mass = MassIntegrator { rho: 1.0 };
        let quad_order = if args.order >= 2 { 4 } else { 3 };

        let mut a = ParAssembler::assemble_bilinear(&par_space, &[&diff], quad_order);
        let mut m = ParAssembler::assemble_bilinear(&par_space, &[&mass], quad_order);

        // Apply Dirichlet BCs
        let bc_dm = par_space.local_space().dof_manager();
        let bc_dofs = boundary_dofs(
            par_space.local_space().mesh(),
            bc_dm,
            &par_space.local_space().mesh().unique_boundary_tags(),
        );
        let dof_part = par_space.dof_partition();
        let mut ess: Vec<usize> = bc_dofs.iter()
            .map(|&d| dof_part.permute_dof(d) as usize)
            .filter(|&p| p < dof_part.n_owned_dofs)
            .collect();
        ess.sort_unstable();
        ess.dedup();

        a = eliminate_ess_diag(&a, &ess, 1.0);
        m = eliminate_ess_diag(&m, &ess, f64::MIN_POSITIVE);

        // Build AMG preconditioner
        let amg_cfg = ParAmgConfig {
            smoother: SmootherType::SymmetricGaussSeidel,
            n_pre_smooth: 2,
            n_post_smooth: 2,
            smoothed_prolongation: true,
            ..Default::default()
        };
        let comm_amg = par_space.comm().clone();
        let amg = ParAmgHierarchy::build(&a, &comm_amg, amg_cfg);
        let pv_pre = par_space.clone();
        let precond = move |r: &[f64], z: &mut [f64]| {
            let n = r.len();
            let mut b = ParVector::zeros(&pv_pre);
            let mut x = ParVector::zeros(&pv_pre);
            for i in 0..n { b.as_slice_mut()[i] = r[i]; }
            amg.vcycle(&b, &mut x);
            for i in 0..n { z[i] = x.as_slice()[i]; }
        };

        let ess_c = ess.clone();
        let proj = move |block: &mut [ParVector]| {
            for v in block { for &p in &ess_c { if p < v.owned_slice_mut().len() { v.owned_slice_mut()[p] = 0.0; } } }
        };

        if rank == 0 { eprintln!("Solving for eigenvalues using ParLOBPCG"); }

        let res = par_lobpcg(&a, Some(&m), args.nev, &precond, Some(&proj), 0.0, 100, 1e-8);

        if rank == 0 {
            println!("Number of unknowns: {}", par_space.n_global_dofs());
            println!("\n  Computed eigenvalues:");
            let mut sorted = res.eigenvalues.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            for (i, &lam) in sorted.iter().enumerate() {
                println!("  {:<6}  {:>24.14e}  {:>16.6e}", i + 1, lam, lam.sqrt() / (2.0 * std::f64::consts::PI));
            }
        }
    });
}
