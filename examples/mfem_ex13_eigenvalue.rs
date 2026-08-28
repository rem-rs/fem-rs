//! MFEM Example 13 — Maxwell Cavity Eigenvalue (1:1 translation of ex13p).
//!
//! Supports two solver backends:
//! - **AMG-LOBPCG** (default): gradient constraints + AMG preconditioner
//! - **AME** (`--ame`): Auxiliary-space Maxwell Eigensolver (LPBPCG + AMS +
//!   discrete divergence-free projection) — matches HYPRE AME.

use std::fs::File;
use std::io::Write;
use fem_amg::AmgConfig;
use fem_examples::maxwell::{assemble_hcurl_eigen_system_from_marker, solve_hcurl_eigen_preconditioned_amg, solve_hcurl_eigen_ame};
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_file_3d};
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::{SolverConfig, eigen::{LobpcgConfig, AmeConfig}};
use fem_space::{H1Space, HCurlSpace, fe_space::FESpace};

struct Args { mesh: String, ser_ref_levels: usize, order: u8, nev: usize, use_ame: bool, }
impl Args {
    fn parse() -> Self {
        // MFEM ex13p defaults: beam-tet.mesh, ser_ref_levels=2, par_ref_levels=1
        // (total 3 uniform refinements in the serial port), order=1, nev=5.
        let mut a = Args { mesh: "data/beam-tet.mesh".to_string(), ser_ref_levels: 2, order: 1, nev: 5, use_ame: false };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" | "--mesh" => a.mesh = it.next().unwrap_or(a.mesh),
                "-rs" | "--refine-serial" => { a.ser_ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(0) }
                "-o" | "--order" => { a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1) }
                "-n" | "--num-eigs" => { a.nev = it.next().and_then(|v| v.parse().ok()).unwrap_or(5) }
                "--ame" => a.use_ame = true,
                "-no-vis" => {} _ => {}
            }
        }
        a
    }
}

fn main() {
    let args = Args::parse();
    let mfem = read_mfem_file(&args.mesh).expect("failed to read MFEM mesh");
    let is_3d: bool;
    let mut mesh3d: Option<Mesh<3>> = None;
    let mut mesh2d: Option<Mesh<2>> = None;
    if let Some(m) = mfem.mesh2d {
        is_3d = false;
        let mut m = m;
        for _ in 0..args.ser_ref_levels { m = refine_uniform(&m); }
        mesh2d = Some(m);
    } else if let Some(m) = mfem.mesh3d {
        is_3d = true;
        let mut m = m;
        for _ in 0..args.ser_ref_levels { m = fem_mesh::amr::refine_uniform_3d(&m); }
        mesh3d = Some(m);
    } else {
        panic!("MFEM mesh must be 2D or 3D");
    }

    let qo = args.order as u8 * 2 + 1;

    if is_3d {
        let mesh = mesh3d.take().unwrap();
        let h1 = H1Space::new(mesh.clone(), args.order);
        let space = HCurlSpace::new(mesh, args.order);
        let n = space.n_dofs();
        println!("Options used:");
        println!("   --mesh {}", args.mesh);
        println!("   --refine-serial {}", args.ser_ref_levels);
        println!("   --order {}", args.order);
        println!("   --num-eigs {}", args.nev);
        println!("   --ame {}", args.use_ame);
        println!("   --no-visualization");
        println!("Number of unknowns: {n}");

        let bdr_attrs: Vec<i32> = space.mesh().unique_boundary_tags();
        let ess_bdr: Vec<i32> = bdr_attrs.iter().map(|_| 1).collect();
        let sys = assemble_hcurl_eigen_system_from_marker(
            &h1, &space, &bdr_attrs, &ess_bdr, 1.0, 1.0, qo,
        );
        let n_free = sys.hcurl_free_dofs.len();
        println!("  Free DOFs: {n_free}, nullspace dim: {}", sys.constraints.ncols());
        println!("\nSolving generalized eigenvalue problem with preconditioning");
        let result = if args.use_ame {
            let ame_cfg = AmeConfig { nev: args.nev, verbose: true, ..AmeConfig::default() };
            solve_hcurl_eigen_ame(&sys, args.nev, &ame_cfg).expect("AME solve failed")
        } else {
            let eig_cfg = LobpcgConfig { max_iter: 200, tol: 1e-8, verbose: true, ..LobpcgConfig::default() };
            let inner_cfg = SolverConfig { rtol: 1e-2, atol: 1e-12, max_iter: 20, verbose: false, ..SolverConfig::default() };
            solve_hcurl_eigen_preconditioned_amg(&sys, args.nev, &eig_cfg, AmgConfig::default(), &inner_cfg)
                .expect("LOBPCG failed")
        };
        for (i, &lam) in result.eigenvalues.iter().enumerate() {
            println!("Eigenmode {}, Lambda = {:.14e}", i + 1, lam);
        }
        println!("{} iterations", result.iterations);
        {
            write_mfem_file_3d("refined.mesh", space.mesh()).expect("mesh write failed");
            eprintln!("  Saved refined mesh -> 'refined.mesh'");
            for i in 0..result.eigenvalues.len() {
                let mf = format!("mode_{:02}.dat", i);
                let mut fo = File::create(&mf).unwrap_or_else(|e| panic!("cannot create {mf}: {e}"));
                for r in 0..n_free { writeln!(fo, "{:.14e}", result.eigenvectors[(r, i)]).expect("write"); }
                eprintln!("  Saved eigenmode {:>2} (λ = {:.14e}) -> '{mf}'", i+1, result.eigenvalues[i]);
            }
        }
        eprintln!("\n  Done.");
    } else {
        let mesh = mesh2d.take().unwrap();
        let h1 = H1Space::new(mesh.clone(), args.order);
        let space = HCurlSpace::new(mesh, args.order);
        let n = space.n_dofs();
        println!("Options used:");
        println!("   --mesh {}", args.mesh);
        println!("   --refine-serial {}", args.ser_ref_levels);
        println!("   --order {}", args.order);
        println!("   --num-eigs {}", args.nev);
        println!("   --ame {}", args.use_ame);
        println!("   --no-visualization");
        println!("Number of unknowns: {n}");

        let bdr_attrs: Vec<i32> = space.mesh().unique_boundary_tags();
        let ess_bdr: Vec<i32> = bdr_attrs.iter().map(|_| 1).collect();
        let sys = assemble_hcurl_eigen_system_from_marker(
            &h1, &space, &bdr_attrs, &ess_bdr, 1.0, 1.0, qo,
        );
        let n_free = sys.hcurl_free_dofs.len();
        println!("  Free DOFs: {n_free}, nullspace dim: {}", sys.constraints.ncols());
        println!("\nSolving generalized eigenvalue problem with preconditioning");
        let result = if args.use_ame {
            let ame_cfg = AmeConfig { nev: args.nev, verbose: true, ..AmeConfig::default() };
            solve_hcurl_eigen_ame(&sys, args.nev, &ame_cfg).expect("AME solve failed")
        } else {
            let eig_cfg = LobpcgConfig { max_iter: 200, tol: 1e-8, verbose: true, ..LobpcgConfig::default() };
            let inner_cfg = SolverConfig { rtol: 1e-2, atol: 1e-12, max_iter: 20, verbose: false, ..SolverConfig::default() };
            solve_hcurl_eigen_preconditioned_amg(&sys, args.nev, &eig_cfg, AmgConfig::default(), &inner_cfg)
                .expect("LOBPCG failed")
        };
        for (i, &lam) in result.eigenvalues.iter().enumerate() {
            println!("Eigenmode {}, Lambda = {:.14e}", i + 1, lam);
        }
        println!("{} iterations", result.iterations);
        {
            write_mfem_file("refined.mesh", space.mesh()).expect("mesh write failed");
            eprintln!("  Saved refined mesh -> 'refined.mesh'");
            for i in 0..result.eigenvalues.len() {
                let mf = format!("mode_{:02}.dat", i);
                let mut fo = File::create(&mf).unwrap_or_else(|e| panic!("cannot create {mf}: {e}"));
                for r in 0..n_free { writeln!(fo, "{:.14e}", result.eigenvectors[(r, i)]).expect("write"); }
                eprintln!("  Saved eigenmode {:>2} (λ = {:.14e}) -> '{mf}'", i+1, result.eigenvalues[i]);
            }
        }
        eprintln!("\n  Done.");
    }
}
