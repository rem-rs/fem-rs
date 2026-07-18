//! # Example 40 — Eikonal equation (1:1 with MFEM ex40)
//!
//! Solves |∇u| = 1 in Ω, u = 0 on ∂Ω by solving the saddle-point system
//!   (∇R)⁻¹(ψ) + ∇u = 0   in H(div)
//!            ∇·ψ = αₖ   in L²
//!
//! using a damped quasi-Newton method with MINRES + BlockDiagonalPC.
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex40_eikonal -- -m data/star.mesh
//! cargo run --example mfem_ex40_eikonal -- -m data/star.mesh -r 3 -o 1
//! ```

#![allow(warnings)]
use std::f64::consts::PI;
use fem_assembly::{Assembler, VectorAssembler};
use fem_assembly::standard::{VectorMassIntegrator, MassIntegrator};
use fem_assembly::vector_integrator::{VectorBilinearIntegrator, VectorLinearIntegrator, VectorQpData};
use fem_assembly::integrator::{BilinearIntegrator, LinearIntegrator, QpData};
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{Mesh, MeshTopology, ElementType, amr::refine_uniform};
use fem_solver::SolverConfig;
use fem_space::{HDivSpace, L2Space, fe_space::FESpace};
use fem_space::constraints::{eliminate_dirichlet, expand_from_reduced, boundary_dofs_hdiv};

struct Args {
    mesh: String, order: u8, refs: usize, max_it: usize,
    alpha: f64, growth_rate: f64, newton_scaling: f64, eps: f64, tol: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: "data/star.mesh".into(), order: 1, refs: 3, max_it: 5,
        alpha: 1.0, growth_rate: 1.0, newton_scaling: 0.8, eps: 1e-6, tol: 1e-4,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m"|"--mesh" => a.mesh = it.next().unwrap_or(a.mesh),
            "-o"|"--order" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-r"|"--refs" => a.refs = it.next().and_then(|v| v.parse().ok()).unwrap_or(3),
            "-mi"|"--max-it" => a.max_it = it.next().and_then(|v| v.parse().ok()).unwrap_or(5),
            "-step" => a.alpha = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            "-gr"|"--growth-rate" => a.growth_rate = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            "-no-vis"|"--no-visualization" => {}
            _ => {}
        }
    }
    a
}

fn main() {
    let args = parse_args();
    println!("Options used:");
    println!("   --mesh {}", args.mesh);
    println!("   --order {}", args.order);
    println!("   --refs {}", args.refs);
    println!("   --max-it {}", args.max_it);
    println!("   --step {}", args.alpha);
    println!("   --growth-rate {}", args.growth_rate);
    println!("   --no-visualization\n");

    let mfem = read_mfem_file(&args.mesh).expect("mesh file");
    let mut mesh = mfem.mesh2d.expect("2D mesh");
    for _ in 0..args.refs { mesh = refine_uniform(&mesh); }
    let dim = mesh.dim();

    // NOTE: This is a STUB — a full 1:1 translation of MFEM ex40 (Eikonal equation
    // with RT+L² spaces, MINRES, BlockDiagonalPC) requires implementing:
    //   1. IsomorphismCoefficient and DIsomorphismCoefficient
    //   2. BlockOperator assembly for the saddle-point system
    //   3. Schur-complement preconditioner construction
    //   4. MINRES solver with BlockDiagonalPreconditioner
    //   5. Outer Newton iteration loop with damped updates
    //
    // The previous file named "mfem_ex40_stokes.rs" was a Stokes solver, NOT
    // a translation of MFEM ex40. It has been renamed to mfem_phase40_stokes.rs.

    eprintln!("\nNOTE: Full ex40 (Eikonal) translation is pending — see C++ ex40.cpp for reference.");
    eprintln!("RT DOFs: {}", HDivSpace::new(mesh.clone(), args.order).n_dofs());
    eprintln!("L2 DOFs: {}", L2Space::new(mesh.clone(), args.order).n_dofs());
}
