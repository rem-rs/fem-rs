//! Poisson/Darcy mixed-method solver — serial cut of MFEM
//! `miniapps/hdiv-linear-solver/darcy.cpp` (PAR-only, 1:1 physics).
//!
//! Solves `-Δp = f` (optionally `α p − Δp = f`) in mixed form
//!
//! ```text
//!      −u − grad p = 0
//!      α p + div u = f
//! ```
//!
//! with natural boundary conditions on the flux `u` and the Dirichlet
//! condition on `p` folded into the flux-equation right-hand side
//! (`VectorFEBoundaryFluxLFIntegrator` of the exact solution).  The exact
//! solution and RHS follow `miniapps/solvers/lor_mms.hpp`:
//! `p = sin(πx)·sin(πy)`, `f = (α + 2π²)·sin(πx)·sin(πy)` (2-D).
//!
//! The saddle-point system is solved with MINRES + matrix-free-style
//! block-diagonal preconditioning via the serial port of
//! `HdivSaddlePointSolver` (see `hdiv_linear_solver.rs` for the deviations
//! from the C++ matrix-free/change-of-basis implementation).
//!
//! Serial cut: `-d/--device` is accepted and ignored; `-rp` counts as extra
//! uniform refinements (C++ `LoadParMesh` refines `-rs` before
//! partitioning and `-rp` after, which for 1 rank is the same mesh).
//! The boundary-flux RHS `b_rt = ∮ p_D (v·n) ds` is assembled directly
//! through `VectorBoundaryAssembler` + `HdivNormalFluxIntegrator` (the
//! `VectorFEBoundaryFluxLFIntegrator` analogue); the previous lifting
//! workaround (`b_rt = Dᵀ·p_src − R·q_src`) was removed once the boundary
//! assembler gained owner-type-aware isoparametric geometry (kernel fix D13).

use std::time::Instant;

#[path = "hdiv_linear_solver.rs"]
mod hdiv_linear_solver;

use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::standard::DomainSourceIntegrator;
use fem_assembly::{Assembler, HdivNormalFluxIntegrator, VectorBoundaryAssembler};
use fem_io::mfem::read_mfem_file;
use fem_mesh::MeshTopology;
use fem_mesh::refine_uniform;
use hdiv_linear_solver::{HdivSaddlePointSolver, Mode};

/// `lor_mms.hpp` exact solution `u = sin(πx)·sin(πy)` (2-D).
fn u_exact(x: &[f64]) -> f64 {
    (std::f64::consts::PI * x[0]).sin() * (std::f64::consts::PI * x[1]).sin()
}

/// `lor_mms.hpp` RHS `f(α) = α·u + 2π²·u` (2-D).
fn f_exact(x: &[f64], alpha: f64) -> f64 {
    let pi2 = std::f64::consts::PI * std::f64::consts::PI;
    (alpha + 2.0 * pi2) * u_exact(x)
}

struct Args {
    mesh: String,
    ser_ref: u32,
    par_ref: u32,
    order: u8,
    alpha: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: "../data/star.mesh".to_string(),
        ser_ref: 1,
        par_ref: 1,
        order: 3,
        alpha: 0.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next().unwrap_or_else(|| a.mesh.clone()),
            "-rs" | "--serial-refine" => {
                a.ser_ref = it.next().and_then(|v| v.parse().ok()).unwrap_or(1);
            }
            "-rp" | "--parallel-refine" => {
                a.par_ref = it.next().and_then(|v| v.parse().ok()).unwrap_or(1);
            }
            "-o" | "--order" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(3),
            "-a" | "--alpha" => a.alpha = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.0),
            // C++ `Device device(device_config)` — serial CPU cut, ignored.
            "-d" | "--device" => {
                let _ = it.next();
            }
            other => {
                eprintln!("unrecognized option '{other}' (darcy serial cut)");
                std::process::exit(2);
            }
        }
    }
    a
}

fn main() {
    let args = parse_args();
    run(&args);
}

/// The miniapp body (C++ `main` after `ParseCheck`); returns the computed
/// L2 error so the round-105 pin tests can assert against the C++ reference
/// without reparsing the CLI.
fn run(args: &Args) -> f64 {
    // MFEM `args.ParseCheck()` → `OptionsParser::PrintOptions` (optparser.cpp
    // 255-272): the "Options used:" echo, byte-for-byte (`--alpha 0` is
    // `ostream << double`, i.e. `%g`).
    print_options(args);
    // MFEM `Device device(device_config); device.Print()` on the serial CPU
    // cut (linalg/device.cpp: "Device configuration" / "Memory configuration").
    println!("Device configuration: cpu");
    println!("Memory configuration: host-std");

    let mesh0 = read_mfem_file(&args.mesh)
        .unwrap_or_else(|e| panic!("cannot read mesh '{}': {e}", args.mesh))
        .mesh2d
        .expect("darcy serial cut supports 2-D meshes only (C++ also handles 3-D)");
    // MFEM `LoadParMesh`: `-rs` refinements before partitioning, `-rp` after —
    // for one rank both are plain uniform refinements.
    let mut mesh = mesh0;
    for _ in 0..args.ser_ref {
        mesh = refine_uniform(&mesh);
    }
    for _ in 0..args.par_ref {
        mesh = refine_uniform(&mesh);
    }

    assert!(args.order >= 1, "-o must be >= 1");
    // C++ `RT_FECollection fec_rt(order-1, dim)`, `L2_FECollection(order-1, dim)`.
    let rt_order = args.order - 1;
    let rt = fem_space::HDivSpace::new(mesh.clone(), rt_order);
    let l2 = fem_space::L2Space::new(mesh.clone(), rt_order);
    let n_rt = rt.n_dofs();
    let n_l2 = l2.n_dofs();
    println!("\nRT DOFs: {n_rt}\nL2 DOFs: {n_l2}");

    // Volume/interior blocks are polynomial and integrated exactly with the
    // 2k+1 rule (matches C++ to rounding).
    let qo = (2 * args.order as usize + 1).max(2) as u8;
    // The two linear forms carry SMOOTH (non-polynomial) data, so their
    // quadrature order is part of the 1:1 semantics: MFEM defaults are
    // `DomainLFIntegrator(f)`  = oa·GetOrder + ob with (a=2, b=0) and
    // `VectorFEBoundaryFluxLFIntegrator(u)` = (a=2, b=0), i.e. order
    // 2·(element order) = 2·rt_order on both sides.
    let lf_qo = (2 * rt_order).max(1) as u8;

    // `b_l2 = (f, v)` on the L2 space (MFEM `DomainLFIntegrator(f_coeff)`).
    let b_l2 = Assembler::assemble_linear(
        &l2,
        &[&DomainSourceIntegrator::new(|x: &[f64]| f_exact(x, args.alpha))],
        lf_qo,
    );

    // `HdivSaddlePointSolver(mesh, fes_rt, fes_l2, alpha_coeff, one, ess={}, DARCY)`.
    let solver = HdivSaddlePointSolver::new(&mesh, &l2, &rt, args.alpha, 1.0, &[], Mode::Darcy);
    let offs = solver.offsets();
    assert_eq!(offs, [0, n_l2, n_l2 + n_rt]);

    // `b_rt = ∮ p_D (v·n) ds` on every boundary attribute (MFEM
    // `b_rt.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(u_coeff))`
    // with the default all-boundary attribute marker).
    let bdr_tags: Vec<i32> = (0..mesh.n_boundary_faces() as u32)
        .map(|f| mesh.face_tag(f))
        .collect();
    let b_rt = VectorBoundaryAssembler::assemble_boundary_linear(
        &rt,
        &[&HdivNormalFluxIntegrator { g: u_exact }],
        &bdr_tags,
        lf_qo,
    );

    let mut b = vec![0.0_f64; offs[2]];
    b[..n_l2].copy_from_slice(&b_l2);
    b[n_l2..].copy_from_slice(&b_rt);
    let mut x = vec![0.0_f64; offs[2]];

    print!("\nSaddle point solver... ");
    use std::io::Write as _;
    std::io::stdout().flush().unwrap();
    let t0 = Instant::now();
    solver.mult(&b, &mut x);
    println!(
        "Done.\nIterations: {}\nElapsed: {}",
        solver.num_iterations(),
        fem_solver::fmt_g(t0.elapsed().as_secs_f64())
    );
    if !solver.converged() {
        eprintln!("MINRES did not converge to the requested tolerance");
    }

    let p = GridFunction::new(&l2, x[..n_l2].to_vec());
    let error = p.compute_l2_error(&u_exact, qo);
    println!("L2 error: {}", fem_solver::fmt_g(error));
    error
}

/// MFEM `OptionsParser::PrintOptions` (optparser.cpp:331-360): the option echo
/// after `ParseCheck`, `   --long-name value` per entry in `AddOption` order.
/// The `alpha` DOUBLE goes through `fem_solver::fmt_g` (`ostream <<` is `%g`
/// at the default precision 6).
fn print_options(a: &Args) {
    println!("Options used:");
    println!("   --device cpu");
    println!("   --mesh {}", a.mesh);
    println!("   --serial-refine {}", a.ser_ref);
    println!("   --parallel-refine {}", a.par_ref);
    println!("   --order {}", a.order);
    println!("   --alpha {}", fem_solver::fmt_g(a.alpha));
}

#[cfg(test)]
mod d105_pins {
    use super::*;

    /// Round-105 pin against the serial C++ MFEM reference (`mpirun -np 1`,
    /// GCC 13 / hypre BoomerAMG, MFEM 4.10; probe `L2ERR17` dump in
    /// `tmp/d105spde/probe_default.txt`).  Both sides run MINRES (rtol 1e-12)
    /// on the same saddle-point system; only the Schur-complement AMG differs
    /// (hypre BoomerAMG vs fem-rs smoothed aggregation), so the achieved L2
    /// errors agree to the solve tolerance — measured rel deviation 5.6e-10 at
    /// port time; the pin holds 1e-8.
    #[test]
    fn darcy_star_default_l2_error_matches_cpp() {
        let args = Args {
            mesh: concat!(env!("CARGO_MANIFEST_DIR"), "/../data/star.mesh").to_string(),
            ser_ref: 1,
            par_ref: 1,
            order: 3,
            alpha: 0.0,
        };
        let error = run(&args);
        // C++ probe: `L2ERR17 4.69700081530774298e-04`; the Rust side computed
        // 4.697000815308006e-04 (rel 5.6e-10, MINRES tolerance scale).
        let want = 4.69700081530774298e-04;
        assert!(
            (error - want).abs() <= 1e-8 * want,
            "darcy L2 error {error:.17e} vs C++ {want:.17e}"
        );
    }

    /// `--alpha` echo and the printed L2 error must survive `%g` formatting:
    /// C++ `cout << 0.0004697000815...` gives `0.0004697` (byte-checked
    /// against cpp_darcy_star.log in round-105).
    #[test]
    fn fmt_g_matches_ostream_double() {
        assert_eq!(fem_solver::fmt_g(4.697000815308006e-04), "0.0004697");
        assert_eq!(fem_solver::fmt_g(0.08), "0.08");
        assert_eq!(fem_solver::fmt_g(-3.168211e4), "-31682.1");
    }

    /// DOF counts on the default `-rs 1 -rp 1` star mesh (C++ `RT DOFs: 5880`,
    /// `L2 DOFs: 2880` — byte-checked round-105).
    #[test]
    fn dof_counts_match_cpp() {
        let mesh_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/star.mesh");
        let mesh0 = read_mfem_file(mesh_path).unwrap().mesh2d.unwrap();
        let mut mesh = refine_uniform(&mesh0);
        mesh = refine_uniform(&mesh);
        let rt = fem_space::HDivSpace::new(mesh.clone(), 2);
        let l2 = fem_space::L2Space::new(mesh, 2);
        assert_eq!(rt.n_dofs(), 5880);
        assert_eq!(l2.n_dofs(), 2880);
    }
}
