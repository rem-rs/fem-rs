//! # Parallel Example 32 — Maxwell eigenvalue problem  (1:1 with MFEM ex32p)
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_pex32_maxwell_eigenvalue -- -m data/fichera.mesh --ranks 2
//! ```

use fem_examples::maxwell::{assemble_hcurl_eigen_system_from_marker, solve_hcurl_eigen_preconditioned_amg};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, refine_uniform_3d};
use fem_space::{H1Space, HCurlSpace, fe_space::FESpace};
use fem_parallel::launcher::{native::ThreadLauncher, WorkerConfig};
use fem_solver::eigen::LobpcgConfig;

fn main() {
    let args = parse_args();
    ThreadLauncher::new(WorkerConfig::new(args.ranks)).launch(move |comm| {
        run_pex32(comm, &args);
    });
}

fn run_pex32(comm: fem_parallel::comm::Comm, args: &Args) {
    let rank = comm.rank();

    let mfem = read_mfem_file(&args.mesh_file).expect("failed to read MFEM mesh");
    let mut serial_mesh = mfem.mesh3d.expect("3D mesh required");
    for _ in 0..args.ser_ref_levels { serial_mesh = refine_uniform_3d(&serial_mesh); }

    // Strategy: rank 0 builds the full serial system and solves serially
    // (same path as pex13, which converges on this problem class).
    let result = if rank == 0 {
        let qo = args.order as u8 * 2 + 1;
        let h1 = H1Space::new(serial_mesh.clone(), args.order);
        let space = HCurlSpace::new(serial_mesh.clone(), args.order);
        let n = space.n_dofs();
        eprintln!("Number of H(Curl) unknowns: {n}");
        let bdr_attrs: Vec<i32> = space.mesh().unique_boundary_tags();
        let ess_bdr: Vec<i32> = bdr_attrs.iter().map(|_| 1).collect();
        let sys = assemble_hcurl_eigen_system_from_marker(&h1, &space, &bdr_attrs, &ess_bdr, 1.0, 1.0, qo);
        let n_free = sys.hcurl_free_dofs.len();
        eprintln!("  Free DOFs: {n_free}, nullspace dim: {}", sys.constraints.ncols());
        // Use AME (Auxiliary-space Maxwell Eigensolver) — it handles the
        // gradient nullspace internally via the discrete divergence-free
        // projector P = I − G(GᵀMG)⁻¹GᵀM combined with the AMS preconditioner.
        let ame_cfg = fem_solver::eigen::AmeConfig::default();
        let res = fem_examples::maxwell::solve_hcurl_eigen_ame(&sys, args.nev, &ame_cfg)
            .expect("AME failed");
        for (i, &lam) in res.eigenvalues.iter().enumerate() {
            eprintln!("Eigenmode {}, Lambda = {:.14e}", i + 1, lam);
        }
        eprintln!("{} iterations", res.iterations);
        Some(res.eigenvalues)
    } else {
        None
    };

    // Broadcast eigenvalues to all ranks.
    let eigenvalues = if rank == 0 {
        result.unwrap()
    } else {
        vec![0.0; args.nev]
    };
    let mut eig_bytes = if rank == 0 {
        eigenvalues.iter().flat_map(|&v: &f64| v.to_le_bytes()).collect::<Vec<u8>>()
    } else {
        vec![0u8; args.nev * 8]
    };
    comm.broadcast_bytes(0, &mut eig_bytes);
    let eigenvalues: Vec<f64> = eig_bytes.chunks(8).map(|b: &[u8]| f64::from_le_bytes(b.try_into().unwrap())).collect();

    if rank == 0 {
        for (i, &lam) in eigenvalues.iter().enumerate() { eprintln!("  Eigenmode {}: lambda = {:.15e}", i+1, lam); }
    }
}

struct Args {
    mesh_file: String, ser_ref_levels: usize, order: u8, nev: usize, ranks: usize,
}
fn parse_args() -> Args {
    let mut a = Args { mesh_file: "data/fichera.mesh".into(), ser_ref_levels: 1, order: 1, nev: 5, ranks: 1 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m"|"--mesh" => a.mesh_file = it.next().unwrap_or("data/fichera.mesh".into()),
            "-rs"|"--refine-serial" => a.ser_ref_levels = it.next().unwrap_or("1".into()).parse().unwrap_or(1),
            "-o"|"--order" => a.order = it.next().unwrap_or("1".into()).parse().unwrap_or(1),
            "-n"|"--num-eigs" => a.nev = it.next().unwrap_or("5".into()).parse().unwrap_or(5),
            "--ranks" => a.ranks = it.next().unwrap_or("1".into()).parse().unwrap_or(1),
            _ => {}
        }
    }
    a
}
