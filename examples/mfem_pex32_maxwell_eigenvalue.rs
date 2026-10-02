//! # Parallel Example 32 — Maxwell eigenvalue problem  (1:1 port of MFEM ex32p)
//!
//! ## Status (round 106, D999): 3-D ND path + the 2-D **default** branch ported
//!
//! C++ `ex32p` default: `../data/inline-quad.mesh` (2-D), solved with the
//! **restricted** H(curl) space `ND_R2D_FECollection` (in-plane Nédélec +
//! out-of-plane H¹ z-component) and `HypreAME` + `HypreAMS`.  Both branches
//! are ported:
//!
//! - **2-D** (the C++ default): `fem_space::embedded_r2d::HCurlR2dSpace`
//!   (the `ND_R2D` space-layer equivalent, bit-verified against C++ ex31)
//!   + the ε-tensor pencil from [`fem_examples::maxwell::assemble_ex32p_r2d_eigen_system`]
//!   + AME.  The 1-D branch (`ND_R1D`, C++ dim == 1) remains a declared gap
//!   and is refused with exit status 3.
//! - **3-D**: plain `ND_FECollection` path, e.g. `-m data/fichera.mesh`.
//!
//! ## Usage
//! ```text
//! cargo run --release --example mfem_pex32_maxwell_eigenvalue -- -m data/fichera.mesh -rs 1 --ranks 2
//! cargo run --release --example mfem_pex32_maxwell_eigenvalue -- --ranks 1   (2-D default, -rs 2)
//! ```

use fem_examples::maxwell::assemble_hcurl_eigen_system_tensor_from_marker;
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::amr::refine_uniform;
use fem_mesh::refine_uniform_3d;
use fem_space::embedded_r2d::HDivR2dSpace;
use fem_space::{H1Space, HCurlSpace, HDivSpace, fe_space::FESpace};
use fem_parallel::launcher::{native::ThreadLauncher, WorkerConfig};

/// C `%.*e` spelling (two-digit exponent): `1.26794609394135e+00`, matching
/// HYPRE's `Eigenvalue lambda   %.14e` printout used for the C++ 对拍.
fn c_e14(v: f64) -> String {
    let s = format!("{v:.14e}");
    let (mant, exp) = s.split_once('e').unwrap();
    let exp_val: i32 = exp.parse().unwrap();
    format!("{mant}e{}{:02}", if exp_val < 0 { "-" } else { "+" }, exp_val.abs())
}

/// The ex32p anisotropic ε tensor
/// `[[2, 1/√2, 0], [1/√2, 2, 1/√2], [0, 1/√2, 2]]` (`M_SQRT1_2`), row-major.
fn ex32p_epsilon() -> [f64; 9] {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    [2.0, s, 0.0, s, 2.0, s, 0.0, s, 2.0]
}

fn print_options(args: &Args) {
    println!("Options used:");
    println!("   --mesh {}", args.mesh_file);
    println!("   --refine-serial {}", args.ser_ref_levels);
    println!("   --refine-parallel 0");
    println!("   --order {}", args.order);
    println!("   --num-eigs {}", args.nev);
    println!("   --no-visualization");
}

fn print_eigen_table(eigenvalues: &[f64]) {
    for lam in eigenvalues {
        println!("Eigenvalue lambda   {}", c_e14(*lam));
    }
}

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

    // Dimension dispatch (D137/D999): never assume the dimension — the C++
    // default mesh is 2-D and takes the restricted ND_R2D branch.
    if let Some(mut mesh2d) = mfem.mesh2d {
        for _ in 0..args.ser_ref_levels { mesh2d = refine_uniform(&mesh2d); }
        let eigenvalues = if rank == 0 {
            print_options(args);
            let shift = mesh2d.unique_boundary_tags().is_empty();
            let sys = fem_examples::maxwell::assemble_ex32p_r2d_eigen_system(
                &mesh2d,
                args.order,
                &ex32p_epsilon(),
                shift,
            );
            println!("Number of H(Curl) unknowns: {}", sys.n_dofs);
            let rt = HDivR2dSpace::new(mesh2d.clone(), args.order.saturating_sub(1));
            println!("Number of H(Div) unknowns: {}", rt.n_dofs());
            if shift {
                println!("Computing eigenvalues shifted by 1");
            }
            eprintln!(
                "  Free DOFs: {}, ess: {}",
                sys.n_dofs - sys.n_ess,
                sys.n_ess
            );
            // AME (Auxiliary-space Maxwell Eigensolver) — the free-DOF
            // elimination is detected from the eliminated M diagonal; the
            // gradient nullspace is skipped by the relative threshold
            // (round-106 D997 semantic pin: the unconstrained positive
            // spectrum, exactly what HYPRE AME converges to).
            let cfg = fem_solver::eigen::AmeConfig::default();
            let g_dummy = CsrMatrix::new_empty(1, 1); // unused by the dense path
            let res = fem_solver::eigen::ame_solve(&sys.a, &sys.m, &g_dummy, &cfg)
                .expect("AME failed");
            eprintln!("{} iterations", res.iterations);
            print_eigen_table(&res.eigenvalues);
            res.eigenvalues
        } else {
            Vec::new()
        };
        broadcast_eigenvalues(comm, args, rank, eigenvalues);
        return;
    }
    if mfem.mesh1d.is_some() {
        if rank == 0 {
            eprintln!(
                "mfem_pex32_maxwell_eigenvalue: 1-D mesh '{}' is not ported (the C++ dim == 1 \
                 branch uses ND_R1D_FECollection + the shifted pencil; the port implements the \
                 2-D ND_R2D default and the 3-D ND path) — exiting with status 3",
                args.mesh_file
            );
        }
        std::process::exit(3)
    }
    let Some(mut serial_mesh) = mfem.mesh3d else {
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

    if rank == 0 {
        print_options(args);
    }

    // Strategy: rank 0 builds the full serial system and solves serially
    // (same path as pex13, which converges on this problem class).
    let result = if rank == 0 {
        // MFEM VectorFEMassIntegrator rule: Trans.OrderW() + 2*GetOrder() = 2+2k
        // on affine hexes — the caller-selected order for the (fixed-order,
        // round-31 D127) tensor mass integrator; the CurlCurlIntegrator picks
        // its own MFEM rule regardless.
        let qo = args.order * 2 + 2;
        let h1 = H1Space::new(serial_mesh.clone(), args.order);
        let space = HCurlSpace::new(serial_mesh.clone(), args.order);
        let n = space.n_dofs();
        // C++ prints the H(Div) companion space size next to the H(Curl) one.
        let rt = HDivSpace::new(serial_mesh.clone(), args.order.saturating_sub(1));
        println!("Number of H(Curl) unknowns: {n}");
        println!("Number of H(Div) unknowns: {}", rt.n_dofs());
        // C++ ex32p: the anisotropic ε tensor as the VectorFEMassIntegrator
        // coefficient of the mass matrix (D996).
        let epsilon = fem_assembly::postproc::coefficient::ConstantMatrixCoeff(
            ex32p_epsilon().to_vec(),
        );
        let bdr_attrs: Vec<i32> = space.mesh().unique_boundary_tags();
        let ess_bdr: Vec<i32> = bdr_attrs.iter().map(|_| 1).collect();
        let sys = assemble_hcurl_eigen_system_tensor_from_marker(
            &h1, &space, &bdr_attrs, &ess_bdr, 1.0, epsilon, qo,
        );
        let n_free = sys.hcurl_free_dofs.len();
        eprintln!("  Free DOFs: {n_free}, nullspace dim: {}", sys.constraints.ncols());
        // Use AME (Auxiliary-space Maxwell Eigensolver) — it handles the
        // gradient nullspace internally via the discrete divergence-free
        // projector P = I − G(GᵀMG)⁻¹GᵀM combined with the AMS preconditioner.
        let ame_cfg = fem_solver::eigen::AmeConfig::default();
        let res = fem_examples::maxwell::solve_hcurl_eigen_ame(&sys, args.nev, &ame_cfg)
            .expect("AME failed");
        for lam in &res.eigenvalues {
            println!("Eigenvalue lambda   {}", c_e14(*lam));
        }
        eprintln!("{} iterations", res.iterations);
        Some(res.eigenvalues)
    } else {
        None
    };

    // Broadcast eigenvalues to all ranks.
    let eigenvalues = match result {
        Some(v) => v,
        None => vec![0.0; args.nev],
    };
    broadcast_eigenvalues(comm, args, rank, eigenvalues);
}

fn broadcast_eigenvalues(
    comm: fem_parallel::comm::Comm,
    args: &Args,
    rank: i32,
    mut eigenvalues: Vec<f64>,
) {
    let n_bytes = args.nev * 8;
    let mut eig_bytes = if rank == 0 && eigenvalues.len() == args.nev {
        eigenvalues.iter().flat_map(|&v: &f64| v.to_le_bytes()).collect::<Vec<u8>>()
    } else {
        vec![0u8; n_bytes]
    };
    comm.broadcast_bytes(0, &mut eig_bytes);
    eigenvalues = eig_bytes.chunks(8).map(|b: &[u8]| f64::from_le_bytes(b.try_into().unwrap())).collect();
    let _ = eigenvalues; // consumed by every rank (visualization is not ported)
}

struct Args {
    mesh_file: String, ser_ref_levels: usize, order: u8, nev: usize, ranks: usize,
}
fn parse_args() -> Args {
    let mut a = Args { mesh_file: "data/inline-quad.mesh".into(), ser_ref_levels: 2, order: 1, nev: 5, ranks: 1 };
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
