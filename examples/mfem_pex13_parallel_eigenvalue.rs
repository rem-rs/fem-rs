//!
//! Parallel Maxwell eigenvalue (pex13).
//!
//! Solves curl curl E = lambda E with homogeneous Dirichlet BC.
//! Strategy: rank 0 runs serial solve, broadcasts eigenvalues.
//!
//! Usage:
//! ```text
//! cargo run --release --example mfem_pex13_parallel_eigenvalue
//! cargo run --release --example mfem_pex13_parallel_eigenvalue -- --ranks 4
//! ```

use fem_amg::AmgConfig;
use fem_examples::maxwell::{assemble_hcurl_eigen_system_from_marker, solve_hcurl_eigen_ame};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::{SolverConfig, eigen::LobpcgConfig};
use fem_space::{H1Space, HCurlSpace, fe_space::FESpace};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::WorkerConfig;

struct Args {
    mesh: String,
    ser_ref_levels: usize,
    order: u8,
    nev: usize,
}

impl Args {
    fn parse() -> Self {
        let mut a = Args { mesh: "data/beam-tet.mesh".into(), ser_ref_levels: 0, order: 1, nev: 5 };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" | "--mesh" => a.mesh = it.next().unwrap_or(a.mesh),
                "-rs" | "--refine-serial" => { a.ser_ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(0); }
                "-o" | "--order" => { a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
                "-n" | "--num-eigs" => { a.nev = it.next().and_then(|v| v.parse().ok()).unwrap_or(5); }
                _ => {}
            }
        }
        a
    }
}

fn main() {
    let args = Args::parse();
    let n_workers: usize = std::env::args()
        .position(|a| a == "--ranks")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    println!("=== fem-rs mfem_pex13: Parallel Maxwell Eigenvalue ===");
    println!("  Workers: {}, Mesh: {}, Refine: {}, Order: {}", n_workers, args.mesh, args.ser_ref_levels, args.order);

    let result = std::sync::Arc::new(std::sync::Mutex::new(None::<(usize, Vec<f64>, String)>));
    let result_slot = result.clone();

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();

        let (n_total, eigenvalues, status) = if rank == 0 {
            let mfem = read_mfem_file(&args.mesh).expect("failed to read mesh");
            let mut mesh3d: Option<Mesh<3>> = None;
            let mut mesh2d: Option<Mesh<2>> = None;
            if let Some(m) = mfem.mesh2d {
                let mut m = m;
                for _ in 0..args.ser_ref_levels { m = refine_uniform(&m); }
                mesh2d = Some(m);
            } else if let Some(m) = mfem.mesh3d {
                let mut m = m;
                for _ in 0..args.ser_ref_levels { m = fem_mesh::amr::refine_uniform_3d(&m); }
                mesh3d = Some(m);
            } else {
                panic!("Mesh must be 2D or 3D");
            }

            let qo = args.order as u8 * 2 + 1;
            let (n, eigenvalues) = if let Some(m) = mesh3d {
                let h1 = H1Space::new(m.clone(), args.order);
                let space = HCurlSpace::new(m.clone(), args.order);
                let n = space.n_dofs();
                println!("Number of unknowns: {n}");
                let bdr_attrs: Vec<i32> = space.mesh().unique_boundary_tags();
                let ess_bdr: Vec<i32> = bdr_attrs.iter().map(|_| 1).collect();
                let sys = assemble_hcurl_eigen_system_from_marker(&h1, &space, &bdr_attrs, &ess_bdr, 1.0, 1.0, qo);
                let n_free = sys.hcurl_free_dofs.len();
                println!("  Free DOFs: {n_free}, nullspace dim: {}", sys.constraints.ncols());
                let eig_cfg = fem_solver::eigen::AmeConfig::default();
                let result = solve_hcurl_eigen_ame(&sys, args.nev, &eig_cfg)
                    .expect("AME failed");
                for (i, &lam) in result.eigenvalues.iter().enumerate() {
                    println!("Eigenmode {}, Lambda = {:.14e}", i + 1, lam);
                }
                println!("{} iterations", result.iterations);
                (n, result.eigenvalues)
            } else if let Some(m) = mesh2d {
                let h1 = H1Space::new(m.clone(), args.order);
                let space = HCurlSpace::new(m.clone(), args.order);
                let n = space.n_dofs();
                println!("Number of unknowns: {n}");
                let bdr_attrs: Vec<i32> = space.mesh().unique_boundary_tags();
                let ess_bdr: Vec<i32> = bdr_attrs.iter().map(|_| 1).collect();
                let sys = assemble_hcurl_eigen_system_from_marker(&h1, &space, &bdr_attrs, &ess_bdr, 1.0, 1.0, qo);
                let n_free = sys.hcurl_free_dofs.len();
                println!("  Free DOFs: {n_free}, nullspace dim: {}", sys.constraints.ncols());
                let eig_cfg = fem_solver::eigen::AmeConfig::default();
                let result = solve_hcurl_eigen_ame(&sys, args.nev, &eig_cfg)
                    .expect("AME failed");
                for (i, &lam) in result.eigenvalues.iter().enumerate() {
                    println!("Eigenmode {}, Lambda = {:.14e}", i + 1, lam);
                }
                println!("{} iterations", result.iterations);
                (n, result.eigenvalues)
            } else {
                panic!("Mesh must be 2D or 3D");
            };

            (n, eigenvalues, "Done".to_string())
        } else {
            (0, vec![], "".to_string())
        };

        // Broadcast n_total
        let mut n_bytes = if rank == 0 { (n_total as u64).to_le_bytes().to_vec() } else { vec![0u8; 8] };
        comm.broadcast_bytes(0, &mut n_bytes);
        let n_total: usize = u64::from_le_bytes(n_bytes.try_into().unwrap()) as usize;

        // Broadcast eigenvalues
        let nev = if rank == 0 { eigenvalues.len() } else { 0 };
        let mut nev_bytes = if rank == 0 { (nev as u64).to_le_bytes().to_vec() } else { vec![0u8; 8] };
        comm.broadcast_bytes(0, &mut nev_bytes);
        let nev: usize = u64::from_le_bytes(nev_bytes.try_into().unwrap()) as usize;

        let mut eig_bytes = if rank == 0 { eigenvalues.iter().flat_map(|&v| v.to_le_bytes()).collect::<Vec<u8>>() } else { vec![0u8; nev * 8] };
        comm.broadcast_bytes(0, &mut eig_bytes);
        let eigenvalues: Vec<f64> = eig_bytes.chunks(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();

        if rank == 0 { *result_slot.lock().unwrap() = Some((n_total, eigenvalues, status)); }
    });

    let (n_total, eigenvalues, status) = result.lock().unwrap().take().unwrap_or((0, vec![], "".to_string()));
    println!("Number of unknowns: {}", n_total);
    for (i, &lam) in eigenvalues.iter().enumerate() {
        println!("Eigenmode {}, Lambda = {:.14e}", i + 1, lam);
    }
    println!("{}", status);
    println!("=== Done ===");
}
