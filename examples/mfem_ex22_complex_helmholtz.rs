//! MFEM Example 22 — Complex Helmholtz (3 variants)
//!
//! Translates MFEM C++ ex22 1:1, supporting:
//!   -p 0: Scalar H1   field:  -Div(a Grad u) - ω² b u + i ω c u = 0
//!   -p 1: Vector H(Curl) field: Curl(a Curl u) - ω² b u + i ω c u = 0
//!   -p 2: Vector H(Div)  field: -Grad(a Div u) - ω² b u + i ω c u = 0
//!
//! Each is driven by a forced oscillation at angular frequency ω imposed on
//! all boundaries (essential / Dirichlet).  On "inline-*" meshes the exact
//! solution u(x) = exp(-iκ·x_{dim-1}) is known.
//!
//! Solver: GMRES + block-diagonal preconditioner
//!   - p=0: DSmoother (Jacobi) on K - ω²εM + ωσM
//!   - p=1: GSSmoother (Gauss-Seidel) on CurlCurl + ω²εM + ωσM
//!   - p=2: DSmoother (Jacobi) on GradDiv - ω²εM + ωσM
//!   Second block = s × first block,  s = (p!=1) ? 1.0 : -1.0
//!
//! Outputs: refined.mesh, sol_r.gf, sol_i.gf, sol_z.gf
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex22_complex_helmholtz -- -p 0 -m data/inline-quad.mesh
//! cargo run --example mfem_ex22_complex_helmholtz -- -p 1 -m data/inline-quad.mesh -o 2
//! cargo run --example mfem_ex22_complex_helmholtz -- -p 2 -m data/inline-quad.mesh -o 2
//! cargo run --example mfem_ex22_complex_helmholtz -- -p 0 -m data/inline-hex.mesh
//! cargo run --example mfem_ex22_complex_helmholtz -- -p 0 -m data/star.mesh -r 1 -o 2 -sigma 10.0
//! ```

#![allow(non_snake_case, dead_code, unused_imports)]

use std::fs::File;
use std::io::Write;

use fem_assembly::complex::{ComplexAssembler, ComplexGridFunction, ComplexSystem};
use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator,
                              CurlCurlIntegrator, GradDivIntegrator, VectorMassIntegrator};
use fem_assembly::VectorAssembler;
use fem_io::mfem::{read_mfem_file, write_mfem, write_mfem_file_3d};
use fem_linalg::{CsrMatrix, fem_to_linlvo_csr};
use fem_element::ReferenceElement;
use fem_mesh::{refine_uniform, topology::MeshTopology, Mesh};
use fem_space::{FESpace, H1Space, HCurlSpace, HDivSpace};
use fem_space::constraints::{boundary_dofs, boundary_dofs_hcurl, boundary_dofs_hdiv};
use fem_solver::{linlvoPreconditioner, DenseVec, GSSmoother, right_preconditioned_gmres,
                 solve_gmres};
use fem_solver::SolverConfig as SolverCfg;
use linlvo::JacobiPrecond;

// ─── CLI struct ───────────────────────────────────────────────────────────

#[derive(Debug)]
struct Config {
    mesh_file: String,
    ref_levels: usize,
    order: usize,
    prob: usize,     // 0=H1, 1=HCurl, 2=HDiv
    mu: f64,
    epsilon: f64,
    sigma: f64,
    omega: f64,
    herm_conv: bool,
}

fn parse_args() -> Config {
    let mut cfg = Config {
        mesh_file: "data/inline-quad.mesh".to_string(),
        ref_levels: 0,
        order: 1,
        prob: 0,
        mu: 1.0,
        epsilon: 1.0,
        sigma: 20.0,
        omega: 10.0,
        herm_conv: true,
    };
    let mut a_coef: f64 = 0.0;
    let mut freq: f64 = -1.0;

    let mut i = std::env::args().skip(1);
    while let Some(a) = i.next() {
        match a.as_str() {
            "-h" | "--help" => {
                eprintln!("Usage: ex22 [OPTIONS]");
                eprintln!("  -m/--mesh         Mesh file (default: data/inline-quad.mesh)");
                eprintln!("  -r/--refine       Refinement levels (default: 0)");
                eprintln!("  -o/--order        Polynomial order (default: 1)");
                eprintln!("  -p/--problem-type 0=H1, 1=HCurl, 2=HDiv (default: 0)");
                eprintln!("  -mu/--permeability μ (default: 1.0)");
                eprintln!("  -eps/--permittivity ε (default: 1.0)");
                eprintln!("  -sigma/--conductivity σ (default: 20.0)");
                eprintln!("  --omega           Angular frequency ω (default: 10.0)");
                eprintln!("  -f/--frequency    Frequency in Hz (overrides --omega)");
                eprintln!("  -a/--stiffness-coef a = 1/μ (default: 0 → use μ)");
                eprintln!("  -herm/--no-herm   Hermitian convention (default: true)");
                std::process::exit(0);
            }
            "-m" | "--mesh" => { cfg.mesh_file = i.next().unwrap_or_default(); }
            "-r" | "--refine" => { cfg.ref_levels = i.next().and_then(|v| v.parse().ok()).unwrap_or(0); }
            "-o" | "--order" => { cfg.order = i.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
            "-p" | "--problem-type" => { cfg.prob = i.next().and_then(|v| v.parse().ok()).unwrap_or(0); }
            "-mu" | "--permeability" => { cfg.mu = i.next().and_then(|v| v.parse().ok()).unwrap_or(1.0); }
            "-eps" | "--permittivity" => { cfg.epsilon = i.next().and_then(|v| v.parse().ok()).unwrap_or(1.0); }
            "-sigma" | "--conductivity" => { cfg.sigma = i.next().and_then(|v| v.parse().ok()).unwrap_or(20.0); }
            "--omega" => { cfg.omega = i.next().and_then(|v| v.parse().ok()).unwrap_or(10.0); }
            "-f" | "--frequency" => { freq = i.next().and_then(|v| v.parse().ok()).unwrap_or(-1.0); }
            "-a" | "--stiffness-coef" => { a_coef = i.next().and_then(|v| v.parse().ok()).unwrap_or(0.0); }
            "-herm" | "--hermitian" => { cfg.herm_conv = true; }
            "-no-herm" | "--no-hermitian" => { cfg.herm_conv = false; }
            _ => {}
        }
    }
    if a_coef != 0.0 {
        cfg.mu = 1.0 / a_coef;
    }
    if freq > 0.0 {
        cfg.omega = 2.0 * std::f64::consts::PI * freq;
    }
    cfg
}

// ─── Exact solution helpers ───────────────────────────────────────────────

/// Complex wavenumber κ = sqrt(μ·ω·(ε·ω - iσ)).
fn complex_kappa(mu: f64, epsilon: f64, sigma: f64, omega: f64) -> (f64, f64) {
    let ar = epsilon * omega;
    let ai = -sigma;
    let k2r = mu * omega * ar;
    let k2i = mu * omega * ai;
    let r = (k2r * k2r + k2i * k2i).sqrt().sqrt();
    let t = 0.5 * k2i.atan2(k2r);
    (r * t.cos(), r * t.sin())
}

/// u0_exact(x) = exp(-iκ·x_{dim-1}): returns (re, im).
fn u0_exact(x: &[f64], mu: f64, epsilon: f64, sigma: f64, omega: f64) -> (f64, f64) {
    let (kr, ki) = complex_kappa(mu, epsilon, sigma, omega);
    let z = x[x.len() - 1];
    let e = (ki * z).exp();
    (e * (-kr * z).cos(), e * (-kr * z).sin())
}

/// Check if mesh filename starts with "inline-".
fn is_inline_mesh(path: &str) -> bool {
    std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.starts_with("inline-"))
        .unwrap_or(false)
}

/// Compute Jacobian matrix (2×2) for a linear element (Tri3, Quad4) at reference point xi.
/// Uses the gradient of the scalar Lagrange basis.
fn element_jacobian(mesh: &Mesh<2>, e: u32, xi: &[f64]) -> (nalgebra::DMatrix<f64>, [f64; 2]) {
    let en = mesh.element_nodes(e);
    let n_nodes = en.len();
    let mut J = nalgebra::DMatrix::<f64>::zeros(2, 2);
    let mut xp = [0.0_f64; 2];

    // Use P1 Lagrange reference element for Jacobian computation
    use fem_element::lagrange::TriP1;
    let ref_e = TriP1;
    let mut phi = vec![0.0; 3];
    let mut grad = vec![0.0; 6]; // 3 nodes × 2 grad components
    ref_e.eval_basis(xi, &mut phi);
    ref_e.eval_grad_basis(xi, &mut grad);

    for k in 0..n_nodes.min(3) {
        let xk = mesh.node_coords(en[k]);
        for a in 0..2 {
            for b in 0..2 {
                J[(a, b)] += xk[a] * grad[k * 2 + b];
            }
            xp[a] += xk[a] * phi[k];
        }
    }
    (J, xp)
}

// ─── Main ─────────────────────────────────────────────────────────────────

fn main() {
    let cfg = parse_args();

    println!("Options used:");
    println!("   --mesh {}", cfg.mesh_file);
    println!("   --refine {}", cfg.ref_levels);
    println!("   --order {}", cfg.order);
    println!("   --problem-type {}", cfg.prob);
    println!("   --mu {}", cfg.mu);
    println!("   --epsilon {}", cfg.epsilon);
    println!("   --sigma {}", cfg.sigma);
    println!("   --omega {}", cfg.omega);
    println!("   --hermitian {}", cfg.herm_conv);

    let exact_sol_known = is_inline_mesh(&cfg.mesh_file);
    if exact_sol_known {
        println!("Identified a mesh with known exact solution");
    }

    let omega = cfg.omega;
    let mu = cfg.mu;
    let epsilon = cfg.epsilon;
    let sigma = cfg.sigma;
    let stiffness_coef = 1.0 / mu;
    let mass_coef = -omega * omega * epsilon;
    let loss_coef = omega * sigma;
    let quad_order = (2 * cfg.order + 1) as u8;

    // Read mesh — detect 2D or 3D
    let mesh_file = &cfg.mesh_file;
    let data = read_mfem_file(mesh_file).expect("read mesh");

    if let Some(mesh2d) = data.mesh2d {
        let mesh = mesh2d;
        if mesh.dim() == 1 && cfg.prob != 0 {
            println!("Switching to problem type 0, H1 basis functions, for 1 dimensional mesh.");
            return;
        }
        let mesh = if cfg.ref_levels > 0 {
            let mut m = mesh;
            for _ in 0..cfg.ref_levels { m = refine_uniform(&m); }
            m
        } else { mesh };
        match cfg.prob {
            0 => solve_p0(&mesh, &cfg, omega, mu, epsilon, sigma,
                          stiffness_coef, mass_coef, loss_coef, quad_order, exact_sol_known),
            1 => solve_p1(&mesh, &cfg, omega, mu, epsilon, sigma,
                          stiffness_coef, mass_coef, loss_coef, quad_order, exact_sol_known),
            2 => solve_p2(&mesh, &cfg, omega, mu, epsilon, sigma,
                          stiffness_coef, mass_coef, loss_coef, quad_order, exact_sol_known),
            _ => unreachable!(),
        }
    } else if let Some(mesh3d) = data.mesh3d {
        let mesh = if cfg.ref_levels > 0 {
            let mut m = mesh3d;
            for _ in 0..cfg.ref_levels { m = fem_mesh::refine_uniform_3d(&m); }
            m
        } else { mesh3d };
        match cfg.prob {
            0 => solve_p0_3d(&mesh, &cfg, omega, mu, epsilon, sigma,
                             stiffness_coef, mass_coef, loss_coef, quad_order, exact_sol_known),
            1 => solve_p1_3d(&mesh, &cfg, omega, mu, epsilon, sigma,
                             stiffness_coef, mass_coef, loss_coef, quad_order, exact_sol_known),
            2 => solve_p2_3d(&mesh, &cfg, omega, mu, epsilon, sigma,
                             stiffness_coef, mass_coef, loss_coef, quad_order, exact_sol_known),
            _ => unreachable!(),
        }
    } else {
        panic!("No mesh found in file: {}", mesh_file);
    }
}

// ─── p=0: Scalar H1 ───────────────────────────────────────────────────────

fn solve_p0(mesh: &Mesh<2>, cfg: &Config, omega: f64,
            mu: f64, epsilon: f64, sigma: f64,
            stiffness_coef: f64, mass_coef: f64, loss_coef: f64,
            quad_order: u8, exact_sol_known: bool) {
    use fem_element::lagrange::{TriP1, TriP2, TriP3,
                                 quad::{QuadQ1, QuadQ2, QuadQ3, QuadQ4}};

    let space = H1Space::new(mesh.clone(), cfg.order as u8);
    let n = space.n_dofs();
    println!("Number of finite element unknowns: {}", n);

    let dm = space.dof_manager();

    let all_tags: Vec<i32> = mesh.unique_boundary_tags();
    let ess_bdr: Vec<usize> = if !all_tags.is_empty() {
        boundary_dofs(mesh, dm, &all_tags).into_iter().map(|d| d as usize).collect()
    } else {
        vec![]
    };

    let mut sys = ComplexAssembler::assemble(
        &space,
        &[&DiffusionIntegrator { kappa: stiffness_coef }],
        &[&MassIntegrator { rho: mass_coef }],
        &[&MassIntegrator { rho: loss_coef }],
        omega, quad_order,
    );
    println!("Size of linear system: {}", sys.n_total());

    let mut rhs = vec![0.0; 2 * n];

    if exact_sol_known {
        let u_proj: Vec<f64> = (0..n).map(|d| {
            let c = dm.dof_coord(d as u32);
            let (re, _im) = u0_exact(&c[..mesh.dim() as usize], mu, epsilon, sigma, omega);
            re
        }).collect();
        let bc_re: Vec<f64> = ess_bdr.iter().map(|&d| u_proj[d]).collect();
        let bc_im: Vec<f64> = ess_bdr.iter().map(|_| 0.0).collect();
        sys.apply_dirichlet(&ess_bdr, &bc_re, &bc_im, &mut rhs);
    } else {
        let bc_re = vec![0.0; ess_bdr.len()];
        let bc_im = vec![0.0; ess_bdr.len()];
        sys.apply_dirichlet(&ess_bdr, &bc_re, &bc_im, &mut rhs);
    }

    let flat = sys.to_flat_csr();

    // Preconditioner: DSmoother (Jacobi) on k_re
    let pc_linlvo = fem_to_linlvo_csr(&sys.k_re);
    let jacobi = JacobiPrecond::from_csr(&pc_linlvo).expect("Jacobi setup");
    let s: f64 = if cfg.herm_conv { 1.0 } else { -1.0 };

    let mut X = vec![0.0; sys.n_total()];
    let pre = |r: &[f64], z: &mut [f64]| {
        let vr = DenseVec::from(r[..n].to_vec());
        let mut zr = DenseVec::zeros(n);
        jacobi.apply_precond(&vr, &mut zr);
        for i in 0..n { z[i] = zr[i]; }
        let vi = DenseVec::from(r[n..].to_vec());
        let mut zi = DenseVec::zeros(n);
        jacobi.apply_precond(&vi, &mut zi);
        for i in 0..n { z[n + i] = s * zi[i]; }
    };
    match right_preconditioned_gmres(&flat, &rhs, &mut X, 50,
        &SolverCfg { rtol: 1e-12, atol: 0.0, max_iter: 1000, ..SolverCfg::default() }, &pre) {
        Ok(r) => println!("  GMRES: {} its  ||r||/||b|| = {:.3e}",
                          r.iterations, r.final_residual),
        Err(e) => eprintln!("  GMRES: {e}"),
    }

    if exact_sol_known {
        let gf = ComplexGridFunction::from_flat(&X);
        let mut er2 = 0.0; let mut ei2 = 0.0;
        for e in 0..mesh.n_elements() as u32 {
            let et = mesh.element_type(e);
            let re = ref_element_h1(et, cfg.order as u8);
            let nld = re.n_dofs();
            let q = re.quadrature(quad_order);
            let mut phi = vec![0.0; nld];
            let mut gr = vec![0.0; nld * 2];
            let en = mesh.element_nodes(e);
            let ed: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
            let mut J = nalgebra::DMatrix::<f64>::zeros(2, 2);
            for (qi, xi) in q.points.iter().enumerate() {
                re.eval_basis(xi, &mut phi);
                re.eval_grad_basis(xi, &mut gr);
                J.fill(0.0);
                let mut xp = [0.0; 2];
                for k in 0..en.len() {
                    let xk = mesh.node_coords(en[k]);
                    for a in 0..2 { for b in 0..2 { J[(a, b)] += xk[a] * gr[k * 2 + b]; } }
                    for a in 0..2 { xp[a] += xk[a] * phi[k]; }
                }
                let w = q.weights[qi] * J.determinant().abs();
                let mut ur = 0.0; let mut ui = 0.0;
                for a in 0..nld { ur += gf.u_re[ed[a]] * phi[a]; ui += gf.u_im[ed[a]] * phi[a]; }
                let (er, ei) = u0_exact(&xp, mu, epsilon, sigma, omega);
                let d1 = ur - er; er2 += w * d1 * d1;
                let d2 = ui - ei; ei2 += w * d2 * d2;
            }
        }
        println!("\n|| Re(u_h-u) ||_{{L^2}} = {:.6e}", er2.sqrt());
        println!("|| Im(u_h-u) ||_{{L^2}} = {:.6e}\n", ei2.sqrt());
        save_output(mesh, &gf);
    } else {
        let gf = ComplexGridFunction::from_flat(&X);
        save_output(mesh, &gf);
    }
}

// ─── p=1: H(Curl) ─────────────────────────────────────────────────────────

fn solve_p1(mesh: &Mesh<2>, cfg: &Config, omega: f64,
            mu: f64, epsilon: f64, sigma: f64,
            stiffness_coef: f64, mass_coef: f64, loss_coef: f64,
            quad_order: u8, exact_sol_known: bool) {
    use fem_element::VectorReferenceElement;
    use fem_element::nedelec::TriND1;

    let space = HCurlSpace::new(mesh.clone(), cfg.order as u8);
    let n = space.n_dofs();
    println!("Number of finite element unknowns: {}", n);

    let all_tags: Vec<i32> = mesh.unique_boundary_tags();
    let ess_bdr: Vec<usize> = if !all_tags.is_empty() {
        boundary_dofs_hcurl(mesh, &space, &all_tags)
            .into_iter().map(|d| d as usize).collect()
    } else {
        vec![]
    };
    println!("  Essential BC DOFs: {}/{}", ess_bdr.len(), n);

    let curl_curl = CurlCurlIntegrator { mu: stiffness_coef };
    let vec_mass_re = VectorMassIntegrator { alpha: mass_coef };
    let vec_mass_im = VectorMassIntegrator { alpha: loss_coef };

    let mut sys = ComplexAssembler::assemble_vector(
        &space,
        &[&curl_curl],
        &[&vec_mass_re],
        &[&vec_mass_im],
        omega, quad_order,
    );
    println!("Size of linear system: {}", sys.n_total());

    // pcOp: CurlCurl(1/μ) + ω²ε·M_vec + ωσ·M_vec
    let neg_mass_coef = omega * omega * epsilon;
    let pc_vec_mass1 = VectorMassIntegrator { alpha: neg_mass_coef };
    let pc_vec_mass2 = VectorMassIntegrator { alpha: loss_coef };
    let pc_mat = VectorAssembler::assemble_bilinear(
        &space, &[&curl_curl, &pc_vec_mass1, &pc_vec_mass2], quad_order);
    let pc_linlvo = fem_to_linlvo_csr(&pc_mat);
    let gsmoother = GSSmoother::from_csr(&pc_linlvo).expect("GSSmoother setup");

    let s: f64 = if cfg.herm_conv { -1.0 } else { 1.0 };

    let mut rhs = vec![0.0; 2 * n];
    if exact_sol_known {
        let u_proj = space.interpolate_vector(&|x| {
            let (re, _im) = u0_exact(x, mu, epsilon, sigma, omega);
            let mut v = vec![0.0; x.len()];
            v[0] = re;
            v
        });
        let u_proj_im = space.interpolate_vector(&|x| {
            let (_re, im) = u0_exact(x, mu, epsilon, sigma, omega);
            let mut v = vec![0.0; x.len()];
            v[0] = im;
            v
        });
        let u_proj_all: Vec<f64> = u_proj.as_slice().to_vec();
        let u_proj_all_im: Vec<f64> = u_proj_im.as_slice().to_vec();
        let bc_re: Vec<f64> = ess_bdr.iter().map(|&d| u_proj_all[d]).collect();
        let bc_im: Vec<f64> = ess_bdr.iter().map(|&d| u_proj_all_im[d]).collect();
        sys.apply_dirichlet(&ess_bdr, &bc_re, &bc_im, &mut rhs);
    } else {
        let bc_re = vec![0.0; ess_bdr.len()];
        let bc_im = vec![0.0; ess_bdr.len()];
        sys.apply_dirichlet(&ess_bdr, &bc_re, &bc_im, &mut rhs);
    }
    let flat = sys.to_flat_csr();
    let mut X = vec![0.0; sys.n_total()];

    let pre = |r: &[f64], z: &mut [f64]| {
        let vr = DenseVec::from(r[..n].to_vec());
        let mut zr = DenseVec::zeros(n);
        gsmoother.apply_precond(&vr, &mut zr);
        for i in 0..n { z[i] = zr[i]; }
        let vi = DenseVec::from(r[n..].to_vec());
        let mut zi = DenseVec::zeros(n);
        gsmoother.apply_precond(&vi, &mut zi);
        for i in 0..n { z[n + i] = s * zi[i]; }
    };
    match right_preconditioned_gmres(&flat, &rhs, &mut X, 50,
        &SolverCfg { rtol: 1e-12, atol: 0.0, max_iter: 1000, ..SolverCfg::default() }, &pre) {
        Ok(r) => println!("  GMRES: {} its  ||r||/||b|| = {:.3e}",
                          r.iterations, r.final_residual),
        Err(e) => eprintln!("  GMRES: {e}"),
    }

    let gf = ComplexGridFunction::from_flat(&X);
    if exact_sol_known {
        let (er2, ei2) = l2_error_hcurl(mesh, &space, &gf.u_re, &gf.u_im,
                                         mu, epsilon, sigma, omega, quad_order);
        println!("\n|| Re(u_h-u) ||_{{L^2}} = {:.6e}", er2);
        println!("|| Im(u_h-u) ||_{{L^2}} = {:.6e}\n", ei2);
    }
    save_output(mesh, &gf);
}

// ─── p=2: H(Div) ──────────────────────────────────────────────────────────

fn solve_p2(mesh: &Mesh<2>, cfg: &Config, omega: f64,
            mu: f64, epsilon: f64, sigma: f64,
            stiffness_coef: f64, mass_coef: f64, loss_coef: f64,
            quad_order: u8, exact_sol_known: bool) {
    let rt_order = if cfg.order >= 1 { cfg.order as u8 - 1 } else { 0 };
    let space = HDivSpace::new(mesh.clone(), rt_order);
    let n = space.n_dofs();
    println!("Number of finite element unknowns: {}", n);

    let all_tags: Vec<i32> = mesh.unique_boundary_tags();
    let ess_bdr: Vec<usize> = if !all_tags.is_empty() {
        boundary_dofs_hdiv(mesh, &space, &all_tags)
            .into_iter().map(|d| d as usize).collect()
    } else {
        vec![]
    };
    println!("  Essential BC DOFs: {}/{}", ess_bdr.len(), n);

    let grad_div = GradDivIntegrator { kappa: stiffness_coef };
    let vec_mass_re = VectorMassIntegrator { alpha: mass_coef };
    let vec_mass_im = VectorMassIntegrator { alpha: loss_coef };

    let mut sys = ComplexAssembler::assemble_vector(
        &space,
        &[&grad_div],
        &[&vec_mass_re],
        &[&vec_mass_im],
        omega, quad_order,
    );
    println!("Size of linear system: {}", sys.n_total());

    // pcOp = k_re (for p=2: pcOp = GradDiv(1/μ) - ω²ε·M + ωσ·M = k_re)
    let pc_linlvo = fem_to_linlvo_csr(&sys.k_re);
    let jacobi = JacobiPrecond::from_csr(&pc_linlvo).expect("Jacobi setup");
    let s: f64 = if cfg.herm_conv { 1.0 } else { -1.0 };

    let mut rhs = vec![0.0; 2 * n];
    if exact_sol_known {
        let u_proj = space.interpolate_vector(&|x| {
            let (re, _im) = u0_exact(x, mu, epsilon, sigma, omega);
            let mut v = vec![0.0; x.len()];
            v[x.len() - 1] = re;
            v
        });
        let u_proj_im = space.interpolate_vector(&|x| {
            let (_re, im) = u0_exact(x, mu, epsilon, sigma, omega);
            let mut v = vec![0.0; x.len()];
            v[x.len() - 1] = im;
            v
        });
        let u_proj_all: Vec<f64> = u_proj.as_slice().to_vec();
        let u_proj_all_im: Vec<f64> = u_proj_im.as_slice().to_vec();
        let bc_re: Vec<f64> = ess_bdr.iter().map(|&d| u_proj_all[d]).collect();
        let bc_im: Vec<f64> = ess_bdr.iter().map(|&d| u_proj_all_im[d]).collect();
        sys.apply_dirichlet(&ess_bdr, &bc_re, &bc_im, &mut rhs);
    } else {
        let bc_re = vec![0.0; ess_bdr.len()];
        let bc_im = vec![0.0; ess_bdr.len()];
        sys.apply_dirichlet(&ess_bdr, &bc_re, &bc_im, &mut rhs);
    }
    let flat = sys.to_flat_csr();
    let mut X = vec![0.0; sys.n_total()];

    let pre = |r: &[f64], z: &mut [f64]| {
        let vr = DenseVec::from(r[..n].to_vec());
        let mut zr = DenseVec::zeros(n);
        jacobi.apply_precond(&vr, &mut zr);
        for i in 0..n { z[i] = zr[i]; }
        let vi = DenseVec::from(r[n..].to_vec());
        let mut zi = DenseVec::zeros(n);
        jacobi.apply_precond(&vi, &mut zi);
        for i in 0..n { z[n + i] = s * zi[i]; }
    };
    match right_preconditioned_gmres(&flat, &rhs, &mut X, 50,
        &SolverCfg { rtol: 1e-12, atol: 0.0, max_iter: 1000, ..SolverCfg::default() }, &pre) {
        Ok(r) => println!("  GMRES: {} its  ||r||/||b|| = {:.3e}",
                          r.iterations, r.final_residual),
        Err(e) => eprintln!("  GMRES: {e}"),
    }

    let gf = ComplexGridFunction::from_flat(&X);
    if exact_sol_known {
        let (er2, ei2) = l2_error_hdiv(mesh, &space, &gf.u_re, &gf.u_im,
                                         mu, epsilon, sigma, omega, quad_order);
        println!("\n|| Re(u_h-u) ||_{{L^2}} = {:.6e}", er2);
        println!("|| Im(u_h-u) ||_{{L^2}} = {:.6e}\n", ei2);
    }
    save_output(mesh, &gf);
}

// ─── L² error helpers ─────────────────────────────────────────────────────

/// L² error for H(Curl) vector fields on Tri3 mesh.
fn l2_error_hcurl(mesh: &Mesh<2>, space: &HCurlSpace<Mesh<2>>,
                   u_re: &[f64], u_im: &[f64],
                   mu: f64, epsilon: f64, sigma: f64, omega: f64,
                   quad_order: u8) -> (f64, f64) {
    use fem_element::VectorReferenceElement;
    use fem_element::nedelec::TriND1;

    let mut er2 = 0.0; let mut ei2 = 0.0;
    for e in 0..mesh.n_elements() as u32 {
        let re = TriND1;
        let nld = re.n_dofs();
        let q = re.quadrature(quad_order);
        let mut phi = vec![0.0; nld * 2];
        let _en = mesh.element_nodes(e);
        let ed: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        for (qi, xi) in q.points.iter().enumerate() {
            re.eval_basis_vec(xi, &mut phi);
            let (J, xp) = element_jacobian(mesh, e, xi);
            let det_j = J.determinant();
            let w = q.weights[qi] * det_j.abs();

            // Numerical solution
            let mut uh_re = [0.0; 2];
            let mut uh_im = [0.0; 2];
            for a in 0..nld {
                uh_re[0] += u_re[ed[a]] * phi[a * 2];
                uh_re[1] += u_re[ed[a]] * phi[a * 2 + 1];
                uh_im[0] += u_im[ed[a]] * phi[a * 2];
                uh_im[1] += u_im[ed[a]] * phi[a * 2 + 1];
            }
            // Exact: [u0_exact, 0]
            let (er, ei) = u0_exact(&xp, mu, epsilon, sigma, omega);
            er2 += w * ((uh_re[0] - er).powi(2) + (uh_re[1] - 0.0).powi(2));
            ei2 += w * ((uh_im[0] - ei).powi(2) + (uh_im[1] - 0.0).powi(2));
        }
    }
    (er2.sqrt(), ei2.sqrt())
}

/// L² error for H(Div) vector fields on Tri3 mesh.
fn l2_error_hdiv(mesh: &Mesh<2>, space: &HDivSpace<Mesh<2>>,
                  u_re: &[f64], u_im: &[f64],
                  mu: f64, epsilon: f64, sigma: f64, omega: f64,
                  quad_order: u8) -> (f64, f64) {
    use fem_element::VectorReferenceElement;
    use fem_element::raviart_thomas::TriRT0;

    let mut er2 = 0.0; let mut ei2 = 0.0;
    for e in 0..mesh.n_elements() as u32 {
        let re = TriRT0;
        let nld = re.n_dofs();
        let q = re.quadrature(quad_order);
        let mut phi = vec![0.0; nld * 2];
        let _en = mesh.element_nodes(e);
        let ed: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        for (qi, xi) in q.points.iter().enumerate() {
            re.eval_basis_vec(xi, &mut phi);
            let (J, xp) = element_jacobian(mesh, e, xi);
            let det_j = J.determinant();
            let w = q.weights[qi] * det_j.abs();

            let mut uh_re = [0.0; 2];
            let mut uh_im = [0.0; 2];
            for a in 0..nld {
                uh_re[0] += u_re[ed[a]] * phi[a * 2];
                uh_re[1] += u_re[ed[a]] * phi[a * 2 + 1];
                uh_im[0] += u_im[ed[a]] * phi[a * 2];
                uh_im[1] += u_im[ed[a]] * phi[a * 2 + 1];
            }
            // Exact: [0, u0_exact]
            let (er, ei) = u0_exact(&xp, mu, epsilon, sigma, omega);
            er2 += w * ((uh_re[0] - 0.0).powi(2) + (uh_re[1] - er).powi(2));
            ei2 += w * ((uh_im[0] - 0.0).powi(2) + (uh_im[1] - ei).powi(2));
        }
    }
    (er2.sqrt(), ei2.sqrt())
}

// ─── Output ───────────────────────────────────────────────────────────────

fn ref_element_h1(et: fem_mesh::element_type::ElementType, order: u8) -> Box<dyn ReferenceElement> {
    use fem_element::lagrange::*;
    match (et, order) {
        (fem_mesh::element_type::ElementType::Tri3, 1) => Box::new(TriP1),
        (fem_mesh::element_type::ElementType::Tri3, 2) => Box::new(TriP2),
        (fem_mesh::element_type::ElementType::Tri3, 3) => Box::new(TriP3),
        (fem_mesh::element_type::ElementType::Quad4, 1) => Box::new(QuadQ1),
        (fem_mesh::element_type::ElementType::Quad4, 2) => Box::new(QuadQ2),
        (fem_mesh::element_type::ElementType::Quad4, 3) => Box::new(QuadQ3),
        (fem_mesh::element_type::ElementType::Quad4, 4) => Box::new(QuadQ4),
        _ => Box::new(QuadQ1),
    }
}

fn save_output(mesh: &Mesh<2>, gf: &ComplexGridFunction) {
    if let Ok(mut f) = File::create("refined.mesh") {
        write_mfem(&mut f, mesh, None).ok();
    }
    if let Ok(mut f) = File::create("sol_r.gf") {
        for &v in &gf.u_re {
            writeln!(f, "{:.14e}", v).ok();
        }
    }
    if let Ok(mut f) = File::create("sol_i.gf") {
        for &v in &gf.u_im {
            writeln!(f, "{:.14e}", v).ok();
        }
    }
    if let Ok(mut f) = File::create("sol_z.gf") {
        for i in 0..gf.u_re.len() {
            writeln!(f, "{:.14e}", gf.u_re[i]).ok();
            writeln!(f, "{:.14e}", gf.u_im[i]).ok();
        }
    }
    let amp: Vec<f64> = gf.amplitude();
    let max_amp = amp.iter().cloned().fold(0.0_f64, f64::max);
    let min_amp = amp.iter().cloned().fold(f64::MAX, f64::min);
    let mean_amp = amp.iter().sum::<f64>() / amp.len() as f64;
    println!("Solution amplitude: max={:.6e} min={:.6e} mean={:.6e}",
             max_amp, min_amp, mean_amp);
    println!("Wrote refined.mesh, sol_r.gf, sol_i.gf, sol_z.gf");
}

// ═══════════════════════════════════════════════════════════════════════════
// 3D support
// ═══════════════════════════════════════════════════════════════════════════

/// Compute 3D Jacobian (3x3) for a linear element (Tet4, Hex8).
fn element_jacobian_3d(mesh: &Mesh<3>, e: u32, xi: &[f64]) -> (nalgebra::DMatrix<f64>, [f64; 3]) {
    let et = mesh.element_type(e);
    // Use element-appropriate linear reference element for Jacobian
    let (re, n_max): (Box<dyn ReferenceElement>, usize) = match et {
        fem_mesh::element_type::ElementType::Hex8 => {
            (Box::new(fem_element::lagrange::HexQ1), 8)
        }
        _ => {
            (Box::new(fem_element::lagrange::TetP1), 4)
        }
    };
    let mut phi = vec![0.0; n_max];
    let mut grad = vec![0.0; n_max * 3];
    re.eval_basis(xi, &mut phi);
    re.eval_grad_basis(xi, &mut grad);
    let en = mesh.element_nodes(e);
    let n = en.len().min(n_max);
    let mut J = nalgebra::DMatrix::<f64>::zeros(3, 3);
    let mut xp = [0.0_f64; 3];
    for k in 0..n {
        let xk = mesh.node_coords(en[k]);
        for a in 0..3 {
            for b in 0..3 { J[(a, b)] += xk[a] * grad[k * 3 + b]; }
            xp[a] += xk[a] * phi[k];
        }
    }
    (J, xp)
}

fn ref_element_h1_3d(et: fem_mesh::element_type::ElementType, order: u8) -> Box<dyn ReferenceElement> {
    use fem_element::lagrange::*;
    match (et, order) {
        (fem_mesh::element_type::ElementType::Tet4, 1) => Box::new(TetP1),
        (fem_mesh::element_type::ElementType::Tet4, 2) => Box::new(TetP2),
        (fem_mesh::element_type::ElementType::Tet4, 3) => Box::new(TetP3),
        (fem_mesh::element_type::ElementType::Hex8, 1) => Box::new(HexQ1),
        (fem_mesh::element_type::ElementType::Hex8, 2) => Box::new(HexQ2),
        (fem_mesh::element_type::ElementType::Hex8, 3) => Box::new(HexQ3),
        _ => Box::new(TetP1),
    }
}

// ─── p=0: H1 3D ───────────────────────────────────────────────────────────

fn solve_p0_3d(mesh: &Mesh<3>, cfg: &Config, omega: f64,
               mu: f64, epsilon: f64, sigma: f64,
               stiffness_coef: f64, mass_coef: f64, loss_coef: f64,
               quad_order: u8, exact_sol_known: bool) {
    let space = H1Space::new(mesh.clone(), cfg.order as u8);
    let n = space.n_dofs();
    println!("Number of finite element unknowns: {}", n);

    let dm = space.dof_manager();
    let all_tags: Vec<i32> = mesh.unique_boundary_tags();
    let ess_bdr: Vec<usize> = if !all_tags.is_empty() {
        boundary_dofs(mesh, dm, &all_tags).into_iter().map(|d| d as usize).collect()
    } else { vec![] };

    let mut sys = ComplexAssembler::assemble(
        &space,
        &[&DiffusionIntegrator { kappa: stiffness_coef }],
        &[&MassIntegrator { rho: mass_coef }],
        &[&MassIntegrator { rho: loss_coef }],
        omega, quad_order,
    );
    println!("Size of linear system: {}", sys.n_total());

    let mut rhs = vec![0.0; 2 * n];
    if exact_sol_known {
        let u_proj: Vec<f64> = (0..n).map(|d| {
            let c = dm.dof_coord(d as u32);
            let (re, _im) = u0_exact(&c[..mesh.dim() as usize], mu, epsilon, sigma, omega);
            re
        }).collect();
        let bc_re: Vec<f64> = ess_bdr.iter().map(|&d| u_proj[d]).collect();
        let bc_im: Vec<f64> = ess_bdr.iter().map(|_| 0.0).collect();
        sys.apply_dirichlet(&ess_bdr, &bc_re, &bc_im, &mut rhs);
    } else {
        let bc_re = vec![0.0; ess_bdr.len()];
        let bc_im = vec![0.0; ess_bdr.len()];
        sys.apply_dirichlet(&ess_bdr, &bc_re, &bc_im, &mut rhs);
    }

    let flat = sys.to_flat_csr();
    let pc_linlvo = fem_to_linlvo_csr(&sys.k_re);
    let jacobi = JacobiPrecond::from_csr(&pc_linlvo).expect("Jacobi setup");
    let s: f64 = if cfg.herm_conv { 1.0 } else { -1.0 };

    let mut X = vec![0.0; sys.n_total()];
    let pre = |r: &[f64], z: &mut [f64]| {
        let vr = DenseVec::from(r[..n].to_vec()); let mut zr = DenseVec::zeros(n);
        jacobi.apply_precond(&vr, &mut zr); for i in 0..n { z[i] = zr[i]; }
        let vi = DenseVec::from(r[n..].to_vec()); let mut zi = DenseVec::zeros(n);
        jacobi.apply_precond(&vi, &mut zi); for i in 0..n { z[n + i] = s * zi[i]; }
    };
    match right_preconditioned_gmres(&flat, &rhs, &mut X, 50,
        &SolverCfg { rtol: 1e-12, atol: 0.0, max_iter: 1000, ..SolverCfg::default() }, &pre) {
        Ok(r) => println!("  GMRES: {} its  ||r||/||b|| = {:.3e}",
                          r.iterations, r.final_residual),
        Err(e) => eprintln!("  GMRES: {e}"),
    }

    let gf = ComplexGridFunction::from_flat(&X);
    if exact_sol_known {
        let mut er2 = 0.0; let mut ei2 = 0.0;
        for e in 0..mesh.n_elements() as u32 {
            let et = mesh.element_type(e);
            let re = ref_element_h1_3d(et, cfg.order as u8);
            let nld = re.n_dofs();
            let q = re.quadrature(quad_order);
            let mut phi = vec![0.0; nld];
            let mut gr = vec![0.0; nld * 3];
            let ed: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
            for (qi, xi) in q.points.iter().enumerate() {
                re.eval_basis(xi, &mut phi);
                re.eval_grad_basis(xi, &mut gr);
                let mut J = nalgebra::DMatrix::<f64>::zeros(3, 3);
                let mut xp = [0.0; 3];
                for k in 0..mesh.element_nodes(e).len() {
                    let xk = mesh.node_coords(mesh.element_nodes(e)[k]);
                    for a in 0..3 { for b in 0..3 { J[(a, b)] += xk[a] * gr[k * 3 + b]; } }
                    for a in 0..3 { xp[a] += xk[a] * phi[k]; }
                }
                let w = q.weights[qi] * J.determinant().abs();
                let mut ur = 0.0; let mut ui = 0.0;
                for a in 0..nld { ur += gf.u_re[ed[a]] * phi[a]; ui += gf.u_im[ed[a]] * phi[a]; }
                let (er, ei) = u0_exact(&xp, mu, epsilon, sigma, omega);
                let d1 = ur - er; er2 += w * d1 * d1;
                let d2 = ui - ei; ei2 += w * d2 * d2;
            }
        }
        println!("\n|| Re(u_h-u) ||_{{L^2}} = {:.6e}", er2.sqrt());
        println!("|| Im(u_h-u) ||_{{L^2}} = {:.6e}\n", ei2.sqrt());
    }
    save_output_3d(mesh, &gf);
}

// ─── p=1: HCurl 3D ────────────────────────────────────────────────────────

fn solve_p1_3d(mesh: &Mesh<3>, cfg: &Config, omega: f64,
               mu: f64, epsilon: f64, sigma: f64,
               stiffness_coef: f64, mass_coef: f64, loss_coef: f64,
               quad_order: u8, exact_sol_known: bool) {
    let space = HCurlSpace::new(mesh.clone(), cfg.order as u8);
    let n = space.n_dofs();
    println!("Number of finite element unknowns: {}", n);

    let all_tags: Vec<i32> = mesh.unique_boundary_tags();
    let ess_bdr: Vec<usize> = if !all_tags.is_empty() {
        boundary_dofs_hcurl(mesh, &space, &all_tags).into_iter().map(|d| d as usize).collect()
    } else { vec![] };
    println!("  Essential BC DOFs: {}/{}", ess_bdr.len(), n);

    let curl_curl = CurlCurlIntegrator { mu: stiffness_coef };
    let vec_mass_re = VectorMassIntegrator { alpha: mass_coef };
    let vec_mass_im = VectorMassIntegrator { alpha: loss_coef };

    let mut sys = ComplexAssembler::assemble_vector(
        &space, &[&curl_curl], &[&vec_mass_re], &[&vec_mass_im],
        omega, quad_order,
    );
    println!("Size of linear system: {}", sys.n_total());

    let neg_mass_coef = omega * omega * epsilon;
    let pc_mat = VectorAssembler::assemble_bilinear(
        &space, &[&curl_curl,
                   &VectorMassIntegrator { alpha: neg_mass_coef },
                   &VectorMassIntegrator { alpha: loss_coef }], quad_order);
    let pc_linlvo = fem_to_linlvo_csr(&pc_mat);
    let gsmoother = GSSmoother::from_csr(&pc_linlvo).expect("GSSmoother");
    let s: f64 = if cfg.herm_conv { -1.0 } else { 1.0 };

    let mut rhs = vec![0.0; 2 * n];
    if exact_sol_known {
        let u_proj = space.interpolate_vector(&|x| {
            let (re, _im) = u0_exact(x, mu, epsilon, sigma, omega);
            let mut v = vec![0.0; x.len()]; v[0] = re; v
        });
        let u_proj_im = space.interpolate_vector(&|x| {
            let (_re, im) = u0_exact(x, mu, epsilon, sigma, omega);
            let mut v = vec![0.0; x.len()]; v[0] = im; v
        });
        let bc_re: Vec<f64> = ess_bdr.iter().map(|&d| u_proj.as_slice()[d]).collect();
        let bc_im: Vec<f64> = ess_bdr.iter().map(|&d| u_proj_im.as_slice()[d]).collect();
        sys.apply_dirichlet(&ess_bdr, &bc_re, &bc_im, &mut rhs);
    } else {
        sys.apply_dirichlet(&ess_bdr, &vec![0.0; ess_bdr.len()], &vec![0.0; ess_bdr.len()], &mut rhs);
    }
    let flat = sys.to_flat_csr();
    let mut X = vec![0.0; sys.n_total()];
    let pre = |r: &[f64], z: &mut [f64]| {
        let vr = DenseVec::from(r[..n].to_vec()); let mut zr = DenseVec::zeros(n);
        gsmoother.apply_precond(&vr, &mut zr); for i in 0..n { z[i] = zr[i]; }
        let vi = DenseVec::from(r[n..].to_vec()); let mut zi = DenseVec::zeros(n);
        gsmoother.apply_precond(&vi, &mut zi); for i in 0..n { z[n + i] = s * zi[i]; }
    };
    match right_preconditioned_gmres(&flat, &rhs, &mut X, 50,
        &SolverCfg { rtol: 1e-12, atol: 0.0, max_iter: 1000, ..SolverCfg::default() }, &pre) {
        Ok(r) => println!("  GMRES: {} its  ||r||/||b|| = {:.3e}", r.iterations, r.final_residual),
        Err(e) => eprintln!("  GMRES: {e}"),
    }
    let gf = ComplexGridFunction::from_flat(&X);
    if exact_sol_known {
        let (er2, ei2) = l2_error_hcurl_3d(mesh, &space, &gf.u_re, &gf.u_im,
                                            mu, epsilon, sigma, omega, quad_order);
        println!("\n|| Re(u_h-u) ||_{{L^2}} = {:.6e}", er2);
        println!("|| Im(u_h-u) ||_{{L^2}} = {:.6e}\n", ei2);
    }
    save_output_3d(mesh, &gf);
}

// ─── p=2: HDiv 3D ─────────────────────────────────────────────────────────

fn solve_p2_3d(mesh: &Mesh<3>, cfg: &Config, omega: f64,
               mu: f64, epsilon: f64, sigma: f64,
               stiffness_coef: f64, mass_coef: f64, loss_coef: f64,
               quad_order: u8, exact_sol_known: bool) {
    let rt_order = if cfg.order >= 1 { cfg.order as u8 - 1 } else { 0 };
    let space = HDivSpace::new(mesh.clone(), rt_order);
    let n = space.n_dofs();
    println!("Number of finite element unknowns: {}", n);

    let all_tags: Vec<i32> = mesh.unique_boundary_tags();
    let ess_bdr: Vec<usize> = if !all_tags.is_empty() {
        boundary_dofs_hdiv(mesh, &space, &all_tags).into_iter().map(|d| d as usize).collect()
    } else { vec![] };
    println!("  Essential BC DOFs: {}/{}", ess_bdr.len(), n);

    let grad_div = GradDivIntegrator { kappa: stiffness_coef };
    let vec_mass_re = VectorMassIntegrator { alpha: mass_coef };
    let vec_mass_im = VectorMassIntegrator { alpha: loss_coef };
    let mut sys = ComplexAssembler::assemble_vector(
        &space, &[&grad_div], &[&vec_mass_re], &[&vec_mass_im],
        omega, quad_order,
    );
    println!("Size of linear system: {}", sys.n_total());

    let pc_linlvo = fem_to_linlvo_csr(&sys.k_re);
    let jacobi = JacobiPrecond::from_csr(&pc_linlvo).expect("Jacobi setup");
    let s: f64 = if cfg.herm_conv { 1.0 } else { -1.0 };

    let mut rhs = vec![0.0; 2 * n];
    if exact_sol_known {
        let u_proj = space.interpolate_vector(&|x| {
            let (re, _im) = u0_exact(x, mu, epsilon, sigma, omega);
            let mut v = vec![0.0; x.len()]; v[x.len()-1] = re; v
        });
        let u_proj_im = space.interpolate_vector(&|x| {
            let (_re, im) = u0_exact(x, mu, epsilon, sigma, omega);
            let mut v = vec![0.0; x.len()]; v[x.len()-1] = im; v
        });
        let bc_re: Vec<f64> = ess_bdr.iter().map(|&d| u_proj.as_slice()[d]).collect();
        let bc_im: Vec<f64> = ess_bdr.iter().map(|&d| u_proj_im.as_slice()[d]).collect();
        sys.apply_dirichlet(&ess_bdr, &bc_re, &bc_im, &mut rhs);
    } else {
        sys.apply_dirichlet(&ess_bdr, &vec![0.0; ess_bdr.len()], &vec![0.0; ess_bdr.len()], &mut rhs);
    }
    let flat = sys.to_flat_csr();
    let mut X = vec![0.0; sys.n_total()];
    let pre = |r: &[f64], z: &mut [f64]| {
        let vr = DenseVec::from(r[..n].to_vec()); let mut zr = DenseVec::zeros(n);
        jacobi.apply_precond(&vr, &mut zr); for i in 0..n { z[i] = zr[i]; }
        let vi = DenseVec::from(r[n..].to_vec()); let mut zi = DenseVec::zeros(n);
        jacobi.apply_precond(&vi, &mut zi); for i in 0..n { z[n + i] = s * zi[i]; }
    };
    match right_preconditioned_gmres(&flat, &rhs, &mut X, 50,
        &SolverCfg { rtol: 1e-12, atol: 0.0, max_iter: 1000, ..SolverCfg::default() }, &pre) {
        Ok(r) => println!("  GMRES: {} its  ||r||/||b|| = {:.3e}", r.iterations, r.final_residual),
        Err(e) => eprintln!("  GMRES: {e}"),
    }
    let gf = ComplexGridFunction::from_flat(&X);
    if exact_sol_known {
        let (er2, ei2) = l2_error_hdiv_3d(mesh, &space, &gf.u_re, &gf.u_im,
                                           mu, epsilon, sigma, omega, quad_order);
        println!("\n|| Re(u_h-u) ||_{{L^2}} = {:.6e}", er2);
        println!("|| Im(u_h-u) ||_{{L^2}} = {:.6e}\n", ei2);
    }
    save_output_3d(mesh, &gf);
}

// ─── 3D L² error helpers ──────────────────────────────────────────────────

fn l2_error_hcurl_3d(mesh: &Mesh<3>, space: &HCurlSpace<Mesh<3>>,
                      u_re: &[f64], u_im: &[f64],
                      mu: f64, epsilon: f64, sigma: f64, omega: f64,
                      quad_order: u8) -> (f64, f64) {
    use fem_element::VectorReferenceElement;
    use fem_element::nedelec::{TetND1, HexND1};
    let mut er2 = 0.0; let mut ei2 = 0.0;
    for e in 0..mesh.n_elements() as u32 {
        let et = mesh.element_type(e);
        let re: &dyn VectorReferenceElement = match et {
            fem_mesh::element_type::ElementType::Hex8 => &HexND1,
            _ => &TetND1,
        };
        let nld = re.n_dofs();
        let q = re.quadrature(quad_order);
        let mut phi = vec![0.0; nld * 3];
        let ed: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        for (qi, xi) in q.points.iter().enumerate() {
            re.eval_basis_vec(xi, &mut phi);
            let (J, xp) = element_jacobian_3d(mesh, e, xi);
            let w = q.weights[qi] * J.determinant().abs();
            let mut uh_re = [0.0; 3]; let mut uh_im = [0.0; 3];
            for a in 0..nld {
                for c in 0..3 {
                    uh_re[c] += u_re[ed[a]] * phi[a * 3 + c];
                    uh_im[c] += u_im[ed[a]] * phi[a * 3 + c];
                }
            }
            let (er, ei) = u0_exact(&xp, mu, epsilon, sigma, omega);
            er2 += w * ((uh_re[0] - er).powi(2) + uh_re[1].powi(2) + uh_re[2].powi(2));
            ei2 += w * ((uh_im[0] - ei).powi(2) + uh_im[1].powi(2) + uh_im[2].powi(2));
        }
    }
    (er2.sqrt(), ei2.sqrt())
}

fn l2_error_hdiv_3d(mesh: &Mesh<3>, space: &HDivSpace<Mesh<3>>,
                     u_re: &[f64], u_im: &[f64],
                     mu: f64, epsilon: f64, sigma: f64, omega: f64,
                     quad_order: u8) -> (f64, f64) {
    use fem_element::VectorReferenceElement;
    use fem_element::raviart_thomas::{TetRT0, HexRT0};
    let mut er2 = 0.0; let mut ei2 = 0.0;
    for e in 0..mesh.n_elements() as u32 {
        let et = mesh.element_type(e);
        let re: &dyn VectorReferenceElement = match et {
            fem_mesh::element_type::ElementType::Hex8 => &HexRT0,
            _ => &TetRT0,
        };
        let nld = re.n_dofs();
        let q = re.quadrature(quad_order);
        let mut phi = vec![0.0; nld * 3];
        let ed: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        for (qi, xi) in q.points.iter().enumerate() {
            re.eval_basis_vec(xi, &mut phi);
            let (J, xp) = element_jacobian_3d(mesh, e, xi);
            let w = q.weights[qi] * J.determinant().abs();
            let mut uh_re = [0.0; 3]; let mut uh_im = [0.0; 3];
            for a in 0..nld {
                for c in 0..3 {
                    uh_re[c] += u_re[ed[a]] * phi[a * 3 + c];
                    uh_im[c] += u_im[ed[a]] * phi[a * 3 + c];
                }
            }
            let (er, ei) = u0_exact(&xp, mu, epsilon, sigma, omega);
            // Exact: [0, 0, u0] for HDiv (last component)
            er2 += w * (uh_re[0].powi(2) + uh_re[1].powi(2) + (uh_re[2] - er).powi(2));
            ei2 += w * (uh_im[0].powi(2) + uh_im[1].powi(2) + (uh_im[2] - ei).powi(2));
        }
    }
    (er2.sqrt(), ei2.sqrt())
}

fn save_output_3d(mesh: &Mesh<3>, gf: &ComplexGridFunction) {
    use fem_io::mfem::write_mfem_file_3d;
    let _ = write_mfem_file_3d("refined.mesh", mesh);
    if let Ok(mut f) = File::create("sol_r.gf") {
        for &v in &gf.u_re { writeln!(f, "{:.14e}", v).ok(); }
    }
    if let Ok(mut f) = File::create("sol_i.gf") {
        for &v in &gf.u_im { writeln!(f, "{:.14e}", v).ok(); }
    }
    if let Ok(mut f) = File::create("sol_z.gf") {
        for i in 0..gf.u_re.len() {
            writeln!(f, "{:.14e}", gf.u_re[i]).ok();
            writeln!(f, "{:.14e}", gf.u_im[i]).ok();
        }
    }
    let amp: Vec<f64> = gf.amplitude();
    let max_amp = amp.iter().cloned().fold(0.0_f64, f64::max);
    let min_amp = amp.iter().cloned().fold(f64::MAX, f64::min);
    let mean_amp = amp.iter().sum::<f64>() / amp.len() as f64;
    println!("Solution amplitude: max={:.6e} min={:.6e} mean={:.6e}", max_amp, min_amp, mean_amp);
    println!("Wrote refined.mesh, sol_r.gf, sol_i.gf, sol_z.gf");
}
