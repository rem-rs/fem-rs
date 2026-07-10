//! MFEM Example 13 — Maxwell Cavity Eigenvalue (1:1 translation of ex13p).
//! Uses solve_hcurl_eigen_preconditioned_amg (now with AMS + nullspace_skip).

use std::fs::File;
use std::io::Write;
use fem_amg::AmgConfig;
use fem_examples::maxwell::{assemble_hcurl_eigen_system_from_marker, solve_hcurl_eigen_preconditioned_amg};
use fem_io::mfem::{read_mfem_file, write_mfem};
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::{SolverConfig, eigen::LobpcgConfig};
use fem_space::{H1Space, HCurlSpace, fe_space::FESpace};

struct Args { mesh: String, ser_ref_levels: usize, order: u8, nev: usize, }
impl Args {
    fn parse() -> Self {
        let mut a = Args { mesh: "data/beam-tri.mesh".to_string(), ser_ref_levels: 0, order: 1, nev: 5 };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" | "--mesh" => a.mesh = it.next().unwrap_or(a.mesh),
                "-rs" | "--refine-serial" => { a.ser_ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(0) }
                "-o" | "--order" => { a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1) }
                "-n" | "--num-eigs" => { a.nev = it.next().and_then(|v| v.parse().ok()).unwrap_or(5) }
                "-no-vis" => {} _ => {}
            }
        }
        a
    }
}

fn main() {
    let args = Args::parse();
    let mfem = read_mfem_file(&args.mesh).expect("failed to read MFEM mesh");
    let mut mesh: Mesh<2> = mfem.mesh2d.expect("MFEM mesh must be 2D");
    for _ in 0..args.ser_ref_levels { mesh = refine_uniform(&mesh); }

    let h1 = H1Space::new(mesh.clone(), args.order);
    let space = HCurlSpace::new(mesh, args.order);
    let n = space.n_dofs();
    let qo = args.order as u8 * 2 + 1;

    println!("Options used:");
    println!("   --mesh {}", args.mesh);
    println!("   --refine-serial {}", args.ser_ref_levels);
    println!("   --order {}", args.order);
    println!("   --num-eigs {}", args.nev);
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

    let eig_cfg = LobpcgConfig { max_iter: 200, tol: 1e-8, verbose: true, ..LobpcgConfig::default() };
    let inner_cfg = SolverConfig { rtol: 1e-2, atol: 1e-12, max_iter: 20, verbose: false, ..SolverConfig::default() };
    let result = solve_hcurl_eigen_preconditioned_amg(
        &sys, args.nev, &eig_cfg, AmgConfig::default(), &inner_cfg,
    ).expect("LOBPCG failed");

    for (i, &lam) in result.eigenvalues.iter().enumerate() {
        println!("Eigenmode {}, Lambda = {:.14e}", i + 1, lam);
    }
    println!("{} iterations", result.iterations);

    {
        let mut f = File::create("refined.mesh").expect("cannot create refined.mesh");
        write_mfem(&mut f, space.mesh(), None).expect("mesh write failed");
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
