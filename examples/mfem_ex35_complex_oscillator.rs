//! # Example 35 — Complex-valued harmonic oscillator (1:1 with MFEM ex35p)
#![allow(warnings)]
//!
//! Solves -Div(a Grad u) - ω² b u + i ω c u = 0 with port BCs.
//! Variants: 0 = H1 (scalar), 1 = HCurl (vector), 2 = HDiv (vector)
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex35_complex_oscillator -- -p 0 -o 2
//! cargo run --example mfem_ex35_complex_oscillator -- -p 1 -o 1
//! cargo run --example mfem_ex35_complex_oscillator -- -p 2 -o 1
//! ```

use std::f64::consts::PI;
use fem_assembly::{Assembler, VectorAssembler};
use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator, CurlCurlIntegrator, VectorMassIntegrator};
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CooMatrix, CsrMatrix, fem_to_linlvo_csr, SolveResult};
use fem_mesh::{Mesh, MeshTopology, ElementType, amr::refine_uniform};
use fem_space::{H1Space, HCurlSpace, HDivSpace, SpaceType,
    fe_space::FESpace, constraints::boundary_dofs_hcurl, constraints::boundary_dofs_hdiv, constraints::boundary_dofs};

struct Args { mesh: String, order: u8, refs: usize, prob: u8, mu: f64, eps: f64, sig: f64, omega: f64 }
fn parse_args() -> Args {
    let mut a = Args { mesh: "data/beam-tri.mesh".into(), order: 1, refs: 1, prob: 0, mu: 1.0, eps: 1.0, sig: 2.0, omega: 1.0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m"|"--mesh" => a.mesh = it.next().unwrap_or(a.mesh),
            "-o"|"--order" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-r"|"--refs" => a.refs = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-p"|"--prob" => a.prob = it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "-mu"|"--mu-const" => a.mu = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            "-eps"|"--epsilon-const" => a.eps = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            "-sig"|"--sigma-const" => a.sig = it.next().and_then(|v| v.parse().ok()).unwrap_or(2.0),
            "-f"|"--frequency" => a.omega = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            "-no-vis"|"--no-visualization" => {}
            _ => {}
        }
    }
    a
}

fn main() {
    let args = parse_args();
    let omega = args.omega * PI;
    let omega2 = omega * omega;
    let a_const = 1.0 / args.mu;     // a = 1/μ
    let b_const = args.eps;           // b = ε
    let c_const = args.sig;           // c = σ

    println!("Options used:");
    println!("   --mesh {}", args.mesh);
    println!("   --order {}", args.order);
    println!("   --refs {}", args.refs);
    println!("   --prob {}", args.prob);
    println!("   --mu-const {}", args.mu);
    println!("   --epsilon-const {}", args.eps);
    println!("   --sigma-const {}", args.sig);
    println!("   --frequency {}", args.omega);
    println!("   --no-visualization\n");

    let mfem = read_mfem_file(&args.mesh).expect("mesh");
    let mut mesh: Mesh<2> = mfem.mesh2d.expect("2D");
    for _ in 0..args.refs { mesh = refine_uniform(&mesh); }

    match args.prob {
        0 => solve_h1(&mesh, args.order, a_const, b_const, c_const, omega2, omega),
        1 => solve_hcurl(&mesh, args.order, a_const, b_const, c_const, omega2, omega),
        2 => solve_hdiv(&mesh, args.order, a_const, b_const, c_const, omega2, omega),
        _ => panic!("unsupported problem type"),
    }
}

/// Build complex system matrix: K = real_mat + i * imag_mat
fn build_complex_system(real_mat: &CsrMatrix<f64>, imag_mat: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    // Complex system stored as block matrix [real, -imag; imag, real]
    let n = real_mat.nrows;
    let mut coo = CooMatrix::<f64>::new(2*n, 2*n);
    for r in 0..n {
        for p in real_mat.row_ptr[r]..real_mat.row_ptr[r+1] {
            let c = real_mat.col_idx[p] as usize;
            let v = real_mat.values[p];
            if v != 0.0 { coo.add(r, c, v); coo.add(n+r, n+c, v); }
        }
        for p in imag_mat.row_ptr[r]..imag_mat.row_ptr[r+1] {
            let c = imag_mat.col_idx[p] as usize;
            let v = imag_mat.values[p];
            if v != 0.0 { coo.add(r, n+c, -v); coo.add(n+r, c, v); }
        }
    }
    coo.into_csr()
}

fn solve_h1(mesh: &Mesh<2>, order: u8, a: f64, b: f64, _c: f64, omega2: f64, omega: f64) {
    let qo = order as u8 * 4;
    let h1 = H1Space::new(mesh.clone(), order);
    let n = h1.n_dofs();
    println!("Number of H1 unknowns: {n}");

    // Real part: K = a·∇·∇ - ω²·b·I  (K = stiffness - ω²·mass)
    let stiff = Assembler::assemble_bilinear(&h1, &[&DiffusionIntegrator{kappa:a}], qo);
    let mass = Assembler::assemble_bilinear(&h1, &[&MassIntegrator{rho: b}], qo);
    let mut real_coo = CooMatrix::new(n, n);
    for r in 0..n { for p in stiff.row_ptr[r]..stiff.row_ptr[r+1] { real_coo.add(r, stiff.col_idx[p]as usize, stiff.values[p]); }}
    for r in 0..n { for p in mass.row_ptr[r]..mass.row_ptr[r+1] { real_coo.add(r, mass.col_idx[p]as usize, -omega2 * mass.values[p]); }}
    let real_mat = real_coo.into_csr();

    // Imag part: ω·c·M (damping)
    let mut imag_coo = CooMatrix::new(n, n);
    for r in 0..n { for p in mass.row_ptr[r]..mass.row_ptr[r+1] { imag_coo.add(r, mass.col_idx[p]as usize, omega * mass.values[p]); }}
    let imag_mat = imag_coo.into_csr();

    let sys_mat = build_complex_system(&real_mat, &imag_mat);
    let n2 = 2*n;

    // Dirichlet BC on all boundaries (u=0, real and imag parts)
    let bdr = boundary_dofs(mesh, h1.dof_manager(), &mesh.unique_boundary_tags());
    let mut rhs = vec![0.0; n2];
    let mut mat = sys_mat;
    for &d in &bdr {
        mat.apply_dirichlet_symmetric(d as usize, 0.0, &mut rhs);
        mat.apply_dirichlet_symmetric(n + d as usize, 0.0, &mut rhs);
    }
    let mut x = vec![0.0; n2];
    let precond = fem_solver::GSSmoother::from_csr(&fem_to_linlvo_csr(&mat)).expect("GS");
    fem_solver::solve_pcg(&mat, &rhs, &mut x, &precond, 1e-10, 5000, true).expect("PCG");
    println!("Size of linear system: {n2}");
    println!("  max|Re(u)| = {:.6e}", x[..n].iter().map(|v|v.abs()).fold(0.0_f64, f64::max));
    println!("  max|Im(u)| = {:.6e}", x[n..].iter().map(|v|v.abs()).fold(0.0_f64, f64::max));
}

fn solve_hcurl(mesh: &Mesh<2>, order: u8, a: f64, b: f64, _c: f64, omega2: f64, omega: f64) {
    let qo = order as u8 * 4;
    let nd = HCurlSpace::new(mesh.clone(), order);
    let n = nd.n_dofs();
    println!("Number of H(Curl) unknowns: {n}");

    let curl = CurlCurlIntegrator{mu:a};
    let mass = VectorMassIntegrator{alpha:b};
    let a_mat = VectorAssembler::assemble_bilinear(&nd, &[&curl, &mass], qo);
    let m_mat = VectorAssembler::assemble_bilinear(&nd, &[&VectorMassIntegrator{alpha:b}], qo);

    let mut real_coo = CooMatrix::new(n, n);
    for r in 0..n { for p in a_mat.row_ptr[r]..a_mat.row_ptr[r+1] { real_coo.add(r, a_mat.col_idx[p]as usize, a_mat.values[p]); }}
    for r in 0..n { for p in m_mat.row_ptr[r]..m_mat.row_ptr[r+1] { real_coo.add(r, m_mat.col_idx[p]as usize, -omega2 * m_mat.values[p]); }}
    let real_mat = real_coo.into_csr();

    let mut imag_coo = CooMatrix::new(n, n);
    for r in 0..n { for p in m_mat.row_ptr[r]..m_mat.row_ptr[r+1] { imag_coo.add(r, m_mat.col_idx[p]as usize, omega * m_mat.values[p]); }}
    let imag_mat = imag_coo.into_csr();

    let sys_mat = build_complex_system(&real_mat, &imag_mat);
    let n2 = 2*n;

    let bdr = boundary_dofs_hcurl(mesh, &nd, &mesh.unique_boundary_tags());
    let mut rhs = vec![0.0; n2];
    let mut mat = sys_mat;
    for &d in &bdr { mat.apply_dirichlet_symmetric(d as usize, 0.0, &mut rhs); mat.apply_dirichlet_symmetric(n + d as usize, 0.0, &mut rhs); }
    let mut x = vec![0.0; n2];
    let precond = fem_solver::GSSmoother::from_csr(&fem_to_linlvo_csr(&mat)).expect("GS");
    fem_solver::solve_pcg(&mat, &rhs, &mut x, &precond, 1e-10, 5000, true).expect("PCG");
    println!("Size of linear system: {n2}");
}

fn solve_hdiv(mesh: &Mesh<2>, order: u8, a: f64, b: f64, _c: f64, omega2: f64, omega: f64) {
    let qo = order as u8 * 4;
    let rt = HDivSpace::new(mesh.clone(), order);
    let n = rt.n_dofs();
    println!("Number of H(Div) unknowns: {n}");

    let mass = VectorMassIntegrator{alpha:b};
    let m_mat = VectorAssembler::assemble_bilinear(&rt, &[&mass], qo);
    let mut real_coo = CooMatrix::new(n, n);
    for r in 0..n { for p in m_mat.row_ptr[r]..m_mat.row_ptr[r+1] { real_coo.add(r, m_mat.col_idx[p]as usize, -omega2 * m_mat.values[p]); }}
    let real_mat = real_coo.into_csr();

    let mut imag_coo = CooMatrix::new(n, n);
    for r in 0..n { for p in m_mat.row_ptr[r]..m_mat.row_ptr[r+1] { imag_coo.add(r, m_mat.col_idx[p]as usize, omega * m_mat.values[p]); }}
    let imag_mat = imag_coo.into_csr();

    let sys_mat = build_complex_system(&real_mat, &imag_mat);
    let n2 = 2*n;

    let bdr = boundary_dofs_hdiv(mesh, &rt, &mesh.unique_boundary_tags());
    let mut rhs = vec![0.0; n2];
    let mut mat = sys_mat;
    for &d in &bdr { mat.apply_dirichlet_symmetric(d as usize, 0.0, &mut rhs); mat.apply_dirichlet_symmetric(n + d as usize, 0.0, &mut rhs); }
    let mut x = vec![0.0; n2];
    let precond = fem_solver::GSSmoother::from_csr(&fem_to_linlvo_csr(&mat)).expect("GS");
    fem_solver::solve_pcg(&mat, &rhs, &mut x, &precond, 1e-10, 5000, true).expect("PCG");
    println!("Size of linear system: {n2}");
}
