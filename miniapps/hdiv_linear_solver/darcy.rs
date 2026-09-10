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
//! The boundary-flux RHS `b_rt` is assembled through the lifting
//! `b_rt = Dᵀ·p_src − R·q_src` (L² projections of the exact pressure/flux):
//! fem-rs' `VectorBoundaryAssembler` boundary-trace path is simplex-owner
//! only and produces wrong DOF contributions on quad meshes (kernel gap),
//! while the lifting above equals `∮ p_D (v·n) ds` up to the common
//! `O(h^(k+1))` projection error.

use std::time::Instant;

#[path = "hdiv_linear_solver.rs"]
mod hdiv_linear_solver;

use fem_assembly::postproc::coefficient::FnVectorCoeff;
use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::standard::{DomainSourceIntegrator, MassIntegrator, VectorDomainLFIntegrator};
use fem_assembly::vector_assembler::VectorAssembler;
use fem_assembly::Assembler;
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
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

    let qo = (2 * args.order as usize + 1).max(2) as u8;

    // `b_l2 = (f, v)` on the L2 space (MFEM `DomainLFIntegrator(f_coeff)`).
    let b_l2 = Assembler::assemble_linear(
        &l2,
        &[&DomainSourceIntegrator::new(|x: &[f64]| f_exact(x, args.alpha))],
        qo,
    );

    // `HdivSaddlePointSolver(mesh, fes_rt, fes_l2, alpha_coeff, one, ess={}, DARCY)`.
    let solver = HdivSaddlePointSolver::new(&mesh, &l2, &rt, args.alpha, 1.0, &[], Mode::Darcy);
    let offs = solver.offsets();
    assert_eq!(offs, [0, n_l2, n_l2 + n_rt]);

    // `b_rt = ∮ p_D (v·n) ds` (MFEM `VectorFEBoundaryFluxLFIntegrator`).
    // Serial-cut lifting: `fem_assembly::VectorBoundaryAssembler` evaluates
    // boundary traces through a simplex-only owner inverse map, which is
    // wrong on quad/polygon meshes (kernel gap, see report).  The consistent
    // alternative uses only validated volume assemblies: with `p_src`/`q_src`
    // the L² projections of the exact pressure/flux,
    //
    //     b_rt = Dᵀ·p_src − R·q_src  =  ⟨p_D, v·n⟩_∂Ω + O(h^(k+1))
    //
    // so the discrete system is satisfied exactly by the projected exact
    // solution (same accuracy class as the C++ discretization error).
    let d_e = solver.d_e_matrix();
    let r_e = solver.r_e_matrix();
    let rhs_p = Assembler::assemble_linear(&l2, &[&DomainSourceIntegrator::new(u_exact)], qo);
    let rhs_q = VectorAssembler::assemble_linear(
        &rt,
        &[&VectorDomainLFIntegrator { f: FnVectorCoeff(q_exact) }],
        qo,
    );
    let p_src = solve_spd(&l2_mass(&l2, qo), &rhs_p);
    let q_src = solve_spd(r_e, &rhs_q);
    let dt = d_e.transpose();
    let mut b_rt = vec![0.0_f64; n_rt];
    dt.spmv(&p_src, &mut b_rt);
    let mut rq = vec![0.0_f64; n_rt];
    r_e.spmv(&q_src, &mut rq);
    for (b, r) in b_rt.iter_mut().zip(&rq) {
        *b -= r;
    }

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
        t0.elapsed().as_secs_f64()
    );
    if !solver.converged() {
        eprintln!("MINRES did not converge to the requested tolerance");
    }

    let p = GridFunction::new(&l2, x[..n_l2].to_vec());
    let error = p.compute_l2_error(&u_exact, qo);
    println!("L2 error: {error}");
}

/// Exact flux `q = −∇p` for the `lor_mms` pressure (2-D).
fn q_exact(x: &[f64], out: &mut [f64]) {
    let pi = std::f64::consts::PI;
    out[0] = -pi * (pi * x[0]).cos() * (pi * x[1]).sin();
    out[1] = -pi * (pi * x[0]).sin() * (pi * x[1]).cos();
}

/// AMG-preconditioned CG solve of an SPD mass system (MFEM
/// `ParFiniteElementSpace` projection machinery stand-in).
fn solve_spd(a: &CsrMatrix<f64>, b: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0_f64; a.nrows];
    fem_amg::solve_amg_cg(
        a,
        b,
        &mut x,
        &fem_amg::AmgConfig::default(),
        &fem_linalg::SolverConfig {
            rtol: 1e-13,
            atol: 1e-14,
            max_iter: 500,
            verbose: false,
            print_level: fem_linalg::PrintLevel::Silent,
        },
    )
    .expect("projection mass solve failed");
    x
}

/// L² mass matrix on the L² space.
fn l2_mass(l2: &fem_space::L2Space<fem_mesh::Mesh<2>>, qo: u8) -> CsrMatrix<f64> {
    Assembler::assemble_bilinear(l2, &[&MassIntegrator { rho: 1.0 }], qo)
}
