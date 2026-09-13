//! # Parallel Example 32 — Maxwell eigenvalue problem  (partial port of MFEM ex32p)
//!
//! ## Status (round 31, D137): 3-D only — the C++ default mesh is **2-D**
//!
//! C++ `ex32p` default: `../data/inline-quad.mesh` (2-D), solved with the
//! **restricted** H(curl) space `ND_R2D_FECollection` (in-plane Nédélec +
//! out-of-plane H¹ z-component) and `HypreAME` + `HypreAMS`.  This port only
//! has the plain **3-D** `ND_FECollection` path, so a 2-D input — including
//! the default — is refused with an explicit message and exit status 3
//! instead of solving a different problem.
//!
//! Ported path (1:1 with the C++ 3-D branch): `-m data/fichera.mesh`.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_pex32_maxwell_eigenvalue -- -m data/fichera.mesh --ranks 2
//! ```

use fem_examples::maxwell::assemble_hcurl_eigen_system_from_marker;
use fem_io::mfem::read_mfem_file;
use fem_mesh::refine_uniform_3d;
use fem_space::{H1Space, HCurlSpace, fe_space::FESpace};
use fem_parallel::launcher::{native::ThreadLauncher, WorkerConfig};

fn main() {
    let args = parse_args();
    ThreadLauncher::new(WorkerConfig::new(args.ranks)).launch(move |comm| {
        run_pex32(comm, &args);
    });
}

fn run_pex32(comm: fem_parallel::comm::Comm, args: &Args) {
    let rank = comm.rank();

    let mfem = read_mfem_file(&args.mesh_file).unwrap_or_else(|e| {
        eprintln!(
            "mfem_pex32_maxwell_eigenvalue: cannot read mesh file '{}': {e} — exiting with \
             status 3",
            args.mesh_file
        );
        std::process::exit(3)
    });
    // D137: never assume the dimension — the C++ default is a 2-D mesh.
    let mut serial_mesh = if let Some(m3) = mfem.mesh3d {
        m3
    } else if mfem.mesh2d.is_some() {
        if rank == 0 {
            eprintln!(
                "mfem_pex32_maxwell_eigenvalue: 2-D mesh '{}' is not ported (D137 — the port only \
                 implements the plain 3-D ND path).\n\
                 \x20 C++ ex32p's default mesh is data/inline-quad.mesh (2-D) and it solves it with \
                 the *restricted* H(curl) space\n\
                 \x20 ND_R2D_FECollection (in-plane Nedelec + out-of-plane H1) and HypreAME, which \
                 this port does not implement;\n\
                 \x20 solving the same mesh as a plain 2-D ND problem would not be comparable to \
                 C++.\n\
                 \x20 Pass a 3-D mesh for the ported path (e.g. -m data/fichera.mesh), or use the \
                 C++ ex32p for the 2-D default.",
                args.mesh_file
            );
        }
        std::process::exit(3)
    } else {
        if rank == 0 {
            eprintln!(
                "mfem_pex32_maxwell_eigenvalue: mesh file '{}' contains neither a 2-D nor a 3-D \
                 mesh — exiting with status 3",
                args.mesh_file
            );
        }
        std::process::exit(3)
    };
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
    let mut a = Args { mesh_file: "data/inline-quad.mesh".into(), ser_ref_levels: 1, order: 1, nev: 5, ranks: 1 };
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
