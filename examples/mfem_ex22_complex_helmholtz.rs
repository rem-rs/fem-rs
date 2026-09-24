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

use fem_assembly::complex::{ComplexAssembler, ComplexGridFunction};
use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator,
                              CurlCurlIntegrator, GradDivIntegrator, VectorMassIntegrator};
use fem_assembly::VectorAssembler;
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_gf_file};
use fem_linalg::fem_to_linlvo_csr;
use fem_mesh::{element_jacobian_at, refine_uniform, topology::MeshTopology, Mesh};
use fem_space::{FESpace, H1Space, HCurlSpace, HDivSpace};
use fem_space::constraints::{boundary_dofs, boundary_dofs_hcurl, boundary_dofs_hdiv};
use fem_solver::{linlvoPreconditioner, DenseVec, GSSmoother, right_preconditioned_gmres};
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
/// MFEM: u0_real_exact / u0_imag_exact
fn u0_exact(x: &[f64], mu: f64, epsilon: f64, sigma: f64, omega: f64) -> (f64, f64) {
    let (kr, ki) = complex_kappa(mu, epsilon, sigma, omega);
    let z = x[x.len() - 1];
    let e = (ki * z).exp();
    (e * (-kr * z).cos(), e * (-kr * z).sin())
}

/// u1_exact for H(Curl): v[0] = u0_exact, other components = 0.
/// MFEM: u1_real_exact / u1_imag_exact
fn u1_exact(x: &[f64], dim: usize, mu: f64, epsilon: f64, sigma: f64, omega: f64) -> (Vec<f64>, Vec<f64>) {
    let (re, im) = u0_exact(x, mu, epsilon, sigma, omega);
    let mut vr = vec![0.0; dim];
    let mut vi = vec![0.0; dim];
    vr[0] = re;
    vi[0] = im;
    (vr, vi)
}

/// u2_exact for H(Div): v[dim-1] = u0_exact, other components = 0.
/// MFEM: u2_real_exact / u2_imag_exact
fn u2_exact(x: &[f64], dim: usize, mu: f64, epsilon: f64, sigma: f64, omega: f64) -> (Vec<f64>, Vec<f64>) {
    let (re, im) = u0_exact(x, mu, epsilon, sigma, omega);
    let mut vr = vec![0.0; dim];
    let mut vi = vec![0.0; dim];
    vr[dim - 1] = re;
    vi[dim - 1] = im;
    (vr, vi)
}

/// Check if mesh filename starts with "inline-".
fn is_inline_mesh(path: &str) -> bool {
    std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.starts_with("inline-"))
        .unwrap_or(false)
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
    // Pass the raw physical coefficients (1/μ, ε, σ) — ComplexAssembler
    // applies the frequency factors itself: k_re = K − ω²·ε·M,
    // k_im = ω·σ·M (matching C++ ex22's SesquilinearForm with
    // massCoef(−ω²ε) and lossCoef(ωσ)).  Passing −ω²ε / ωσ here would
    // double-count the frequency (ω⁴ε, ω²σ) and corrupt the solution.
    let stiffness_coef = 1.0 / mu;
    let mass_coef = epsilon;
    let loss_coef = sigma;
    let quad_order = (2 * cfg.order + 1) as u8;

    // Read mesh — detect 2D or 3D
    let mesh_file = &cfg.mesh_file;
    let data = read_mfem_file(mesh_file).expect("read mesh");

    if let Some(mesh2d) = data.mesh2d {
        let mesh = mesh2d;
        if mesh.dim() == 1 && cfg.prob != 0 {
            println!("Switching to problem type 0, H1 basis functions, for 1 dimensional mesh.");
            // Fall through to solve_p0 below
        }
        let mesh = if cfg.ref_levels > 0 {
            let mut m = mesh;
            for _ in 0..cfg.ref_levels { m = refine_uniform(&m); }
            m
        } else { mesh };
        match cfg.prob {
            0 => solve_2d_p0(&mesh, &cfg, omega, mu, epsilon, sigma,
                             stiffness_coef, mass_coef, loss_coef, quad_order, exact_sol_known, 2),
            1 => solve_2d_p1(&mesh, &cfg, omega, mu, epsilon, sigma,
                             stiffness_coef, mass_coef, loss_coef, quad_order, exact_sol_known, 2),
            2 => solve_2d_p2(&mesh, &cfg, omega, mu, epsilon, sigma,
                             stiffness_coef, mass_coef, loss_coef, quad_order, exact_sol_known, 2),
            _ => unreachable!(),
        }
    } else if let Some(mesh3d) = data.mesh3d {
        let mesh = if cfg.ref_levels > 0 {
            let mut m = mesh3d;
            for _ in 0..cfg.ref_levels { m = fem_mesh::refine_uniform_3d(&m); }
            m
        } else { mesh3d };
        match cfg.prob {
            0 => solve_3d_p0(&mesh, &cfg, omega, mu, epsilon, sigma,
                             stiffness_coef, mass_coef, loss_coef, quad_order, exact_sol_known, 3),
            1 => solve_3d_p1(&mesh, &cfg, omega, mu, epsilon, sigma,
                             stiffness_coef, mass_coef, loss_coef, quad_order, exact_sol_known, 3),
            2 => solve_3d_p2(&mesh, &cfg, omega, mu, epsilon, sigma,
                             stiffness_coef, mass_coef, loss_coef, quad_order, exact_sol_known, 3),
            _ => unreachable!(),
        }
    } else {
        panic!("No mesh found in file: {}", mesh_file);
    }
}

// ─── p=0: Scalar H1 ───────────────────────────────────────────────────────

fn solve_2d_p0(mesh: &Mesh<2>, cfg: &Config, omega: f64,
               mu: f64, epsilon: f64, sigma: f64,
               stiffness_coef: f64, mass_coef: f64, loss_coef: f64,
               quad_order: u8, exact_sol_known: bool, _dim: usize) {
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
        // Project BOTH real and imaginary parts of the exact solution onto the
        // boundary DOFs (C++ ex22: ProjectBdrCoefficient(u0_r, u0_i)).  The
        // imaginary part was previously zeroed, which badly corrupted the
        // p=0 (H1) solution since the complex system couples re/im.
        let u_proj = space.interpolate(&|x| {
            let (re, _im) = u0_exact(x, mu, epsilon, sigma, omega);
            re
        }).as_slice().to_vec();
        let u_proj_im = space.interpolate(&|x| {
            let (_re, im) = u0_exact(x, mu, epsilon, sigma, omega);
            im
        }).as_slice().to_vec();
        let bc_re: Vec<f64> = ess_bdr.iter().map(|&d| u_proj[d]).collect();
        let bc_im: Vec<f64> = ess_bdr.iter().map(|&d| u_proj_im[d]).collect();
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

    let gf = ComplexGridFunction::from_flat(&X);
    if exact_sol_known {
        // D693: use the core `ComplexGridFunction::compute_l2_error` (D681
        // verdict) — the mapped point and integration weight come from the
        // mesh's element transformation, not from the solution basis' corner
        // slots, which degenerate on Q2 quads (the former inline evaluator
        // gave 5.06e-1 where C++ prints 5.64364e-3).  Quadrature order
        // follows MFEM's ComputeL2Error convention (2*order + 3).
        let (er, ei) = gf.compute_l2_error(
            &|x| u0_exact(x, mu, epsilon, sigma, omega).0,
            &|x| u0_exact(x, mu, epsilon, sigma, omega).1,
            2 * cfg.order as u8 + 3, &space);
        println!("\n|| Re(u_h-u) ||_{{L^2}} = {:.6e}", er);
        println!("|| Im(u_h-u) ||_{{L^2}} = {:.6e}\n", ei);
    }
    save_output(mesh, &gf);
}

// ─── p=1: H(Curl) ─────────────────────────────────────────────────────────

fn solve_2d_p1(mesh: &Mesh<2>, cfg: &Config, omega: f64,
               mu: f64, epsilon: f64, sigma: f64,
               stiffness_coef: f64, mass_coef: f64, loss_coef: f64,
               quad_order: u8, exact_sol_known: bool, _dim: usize) {
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

    // pcOp (C++ ex22 9a): CurlCurl(1/μ) + ω²ε·M_vec + ωσ·M_vec.
    // D694: the loss term carries its own ω here (MFEM lossCoef(ω·σ)); the
    // system's k_im gets its ω inside ComplexAssembler, so the `loss_coef`
    // passed to the sesquilinear form is the bare σ.
    let neg_mass_coef = omega * omega * epsilon;
    let pc_vec_mass1 = VectorMassIntegrator { alpha: neg_mass_coef };
    let pc_vec_mass2 = VectorMassIntegrator { alpha: omega * loss_coef };
    let mut pc_mat = VectorAssembler::assemble_bilinear(
        &space, &[&curl_curl, &pc_vec_mass1, &pc_vec_mass2], quad_order);
    // D695: MFEM `pcOp->SetDiagonalPolicy(DIAG_ONE)` +
    // `pcOp->FormSystemMatrix(ess_tdof_list)` — zero the essential rows and
    // columns, put 1 on the diagonal, before handing the matrix to the
    // Gauss-Seidel smoother (the zero `value` leaves the dummy rhs untouched).
    let mut pc_rhs = vec![0.0; n];
    for &d in &ess_bdr {
        pc_mat.apply_dirichlet_symmetric(d, 0.0, &mut pc_rhs);
    }
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
        // Use the correct H(Curl) L² error (Piola-transformed, Quad4-aware)
        // from fem_examples::maxwell — the local l2_error_hcurl is hardcoded
        // to TriND1 without the Piola transform and is wrong on quad meshes.
        let er2 = fem_examples::maxwell::l2_error_hcurl_exact(&space, &gf.u_re, |x| {
            let (re, _im) = u0_exact(x, mu, epsilon, sigma, omega);
            [re, 0.0]
        });
        let ei2 = fem_examples::maxwell::l2_error_hcurl_exact(&space, &gf.u_im, |x| {
            let (_re, im) = u0_exact(x, mu, epsilon, sigma, omega);
            [im, 0.0]
        });
        println!("\n|| Re(u_h-u) ||_{{L^2}} = {:.6e}", er2);
        println!("|| Im(u_h-u) ||_{{L^2}} = {:.6e}\n", ei2);
    }
    save_output(mesh, &gf);
}

// ─── p=2: H(Div) ──────────────────────────────────────────────────────────

fn solve_2d_p2(mesh: &Mesh<2>, cfg: &Config, omega: f64,
               mu: f64, epsilon: f64, sigma: f64,
               stiffness_coef: f64, mass_coef: f64, loss_coef: f64,
               quad_order: u8, exact_sol_known: bool, _dim: usize) {
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
                                         mu, epsilon, sigma, omega, 2 * cfg.order as u8 + 3);
        println!("\n|| Re(u_h-u) ||_{{L^2}} = {:.6e}", er2);
        println!("|| Im(u_h-u) ||_{{L^2}} = {:.6e}\n", ei2);
    }
    save_output(mesh, &gf);
}

// ─── L² error helpers ─────────────────────────────────────────────────────

/// L² error for H(Div) vector fields in 2-D (Tri3/Quad4).
///
/// D748: the reference element is the one **paired with the space's own
/// DOF/slot tables** (`fem_assembly::paired_vector_reference_element`, i.e.
/// the assembler's dispatch).  The former hard-wired `TriRTk::new(0)` /
/// `QuadRTk::new(0)` evaluated every order ≥ 1 H(div) space with the RT0
/// basis (4 slots consumed, the rest of the element's DOFs ignored), so
/// `-p 2 -o 2` (RT1) printed 3.497500e-1 against MFEM's 1.595990e-2 (22×).
/// The element's DOF count is asserted against the space's slot table so a
/// future pairing drift fails loudly instead of printing a wrong row.
/// The quadrature order argument is the MFEM `ComputeL2Error` convention
/// (`2*order + 3`), the same convention the ND evaluator uses.
fn l2_error_hdiv(mesh: &Mesh<2>, space: &HDivSpace<Mesh<2>>,
                  u_re: &[f64], u_im: &[f64],
                  mu: f64, epsilon: f64, sigma: f64, omega: f64,
                  quad_order: u8) -> (f64, f64) {
    use fem_mesh::element_type::ElementType;

    // H(Div) Piola (contravariant, 2-D): φ_phys = J·φ_ref / det J.
    // Reference domain: TriRTk on [0,1]² triangle (affine map), QuadRTk on
    // [0,1]² quad (bilinear map — NOTE element_jacobian_at uses the [-1,1]^2
    // QuadQ1 basis and cannot be paired with these [0,1] quadrature points;
    // we build the [0,1]^2 map here instead).
    let rt_order = space.order();
    let mut er2 = 0.0; let mut ei2 = 0.0;
    for e in 0..mesh.n_elements() as u32 {
        let et = mesh.element_type(e);
        let nodes = mesh.element_nodes(e);
        let ed: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);
        let re = fem_assembly::paired_vector_reference_element(
            space.space_type(), et, 2, rt_order);
        let nld = re.n_dofs();
        assert_eq!(
            ed.len(), nld,
            "l2_error_hdiv: {et:?} slot table has {} dofs, the paired RT{rt_order} element \
             needs {nld} — the space and the evaluator disagree",
            ed.len(),
        );
        let q = re.quadrature(quad_order);
        let mut phi = vec![0.0; nld * 2];
        match et {
            ElementType::Tri3 => {
                let x0 = mesh.node_coords(nodes[0]);
                let x1 = mesh.node_coords(nodes[1]);
                let x2 = mesh.node_coords(nodes[2]);
                let (j00, j01) = (x1[0] - x0[0], x2[0] - x0[0]);
                let (j10, j11) = (x1[1] - x0[1], x2[1] - x0[1]);
                let det = j00 * j11 - j01 * j10;
                for (qi, xi) in q.points.iter().enumerate() {
                    re.eval_basis_vec(xi, &mut phi);
                    let w = q.weights[qi] * det.abs();
                    let xp = [x0[0] + j00 * xi[0] + j01 * xi[1],
                              x0[1] + j10 * xi[0] + j11 * xi[1]];
                    let mut uh_re = [0.0; 2];
                    let mut uh_im = [0.0; 2];
                    for a in 0..nld {
                        let s = signs[a];
                        let px = (j00 * phi[a * 2] + j01 * phi[a * 2 + 1]) / det;
                        let py = (j10 * phi[a * 2] + j11 * phi[a * 2 + 1]) / det;
                        uh_re[0] += s * u_re[ed[a]] * px;
                        uh_re[1] += s * u_re[ed[a]] * py;
                        uh_im[0] += s * u_im[ed[a]] * px;
                        uh_im[1] += s * u_im[ed[a]] * py;
                    }
                    let (er, ei) = u0_exact(&xp, mu, epsilon, sigma, omega);
                    er2 += w * ((uh_re[0] - 0.0).powi(2) + (uh_re[1] - er).powi(2));
                    ei2 += w * ((uh_im[0] - 0.0).powi(2) + (uh_im[1] - ei).powi(2));
                }
            }
            ElementType::Quad4 => {
                let xc: Vec<Vec<f64>> = (0..4)
                    .map(|k| mesh.node_coords(nodes[k]).to_vec())
                    .collect();
                // [0,1]^2 bilinear map (must match QuadRTk's [0,1] domain)
                #[allow(non_snake_case)]
                let n = |k: usize, xi: f64, eta: f64| -> f64 {
                    match k {
                        0 => (1.0 - xi) * (1.0 - eta),
                        1 => xi * (1.0 - eta),
                        2 => xi * eta,
                        3 => (1.0 - xi) * eta,
                        _ => 0.0,
                    }
                };
                for (qi, xi_eta) in q.points.iter().enumerate() {
                    let (xi, eta) = (xi_eta[0], xi_eta[1]);
                    re.eval_basis_vec(&[xi, eta], &mut phi);
                    // dx/dxi, dx/deta
                    let (j00, j01) = (
                        -(1.0 - eta) * xc[0][0] + (1.0 - eta) * xc[1][0] + eta * xc[2][0] - eta * xc[3][0],
                        -(1.0 - xi)  * xc[0][0] - xi * xc[1][0] + xi * xc[2][0] + (1.0 - xi) * xc[3][0],
                    );
                    let (j10, j11) = (
                        -(1.0 - eta) * xc[0][1] + (1.0 - eta) * xc[1][1] + eta * xc[2][1] - eta * xc[3][1],
                        -(1.0 - xi)  * xc[0][1] - xi * xc[1][1] + xi * xc[2][1] + (1.0 - xi) * xc[3][1],
                    );
                    let det = j00 * j11 - j01 * j10;
                    let w = q.weights[qi] * det.abs();
                    let xp = [
                        xc[0][0] * n(0, xi, eta) + xc[1][0] * n(1, xi, eta)
                            + xc[2][0] * n(2, xi, eta) + xc[3][0] * n(3, xi, eta),
                        xc[0][1] * n(0, xi, eta) + xc[1][1] * n(1, xi, eta)
                            + xc[2][1] * n(2, xi, eta) + xc[3][1] * n(3, xi, eta),
                    ];
                    let mut uh_re = [0.0; 2];
                    let mut uh_im = [0.0; 2];
                    for a in 0..nld {
                        let s = signs[a];
                        let px = (j00 * phi[a * 2] + j01 * phi[a * 2 + 1]) / det;
                        let py = (j10 * phi[a * 2] + j11 * phi[a * 2 + 1]) / det;
                        uh_re[0] += s * u_re[ed[a]] * px;
                        uh_re[1] += s * u_re[ed[a]] * py;
                        uh_im[0] += s * u_im[ed[a]] * px;
                        uh_im[1] += s * u_im[ed[a]] * py;
                    }
                    let (er, ei) = u0_exact(&xp, mu, epsilon, sigma, omega);
                    er2 += w * ((uh_re[0] - 0.0).powi(2) + (uh_re[1] - er).powi(2));
                    ei2 += w * ((uh_im[0] - 0.0).powi(2) + (uh_im[1] - ei).powi(2));
                }
            }
            _ => panic!("l2_error_hdiv: unsupported element type {et:?}"),
        }
    }
    (er2.sqrt(), ei2.sqrt())
}

// ─── Output ───────────────────────────────────────────────────────────────

fn save_output(mesh: &Mesh<2>, gf: &ComplexGridFunction) {
    write_mfem_file("refined.mesh", mesh).ok();
    // Real part, imaginary part, and complex (interleaved) output
    write_mfem_gf_file("sol_r.gf", 2, &gf.u_re, "H1", 0, 1, 14).ok();
    write_mfem_gf_file("sol_i.gf", 2, &gf.u_im, "H1", 0, 1, 14).ok();
    let n = gf.u_re.len();
    let mut z = vec![0.0_f64; 2 * n];
    for i in 0..n { z[2 * i] = gf.u_re[i]; z[2 * i + 1] = gf.u_im[i]; }
    write_mfem_gf_file("sol_z.gf", 2, &z, "H1", 0, 2, 14).ok();
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
    let (J, xp) = element_jacobian_at(mesh, e, xi, 3);
    (J, [xp[0], xp[1], xp[2]])
}

// ─── p=0: H1 3D ───────────────────────────────────────────────────────────

fn solve_3d_p0(mesh: &Mesh<3>, cfg: &Config, omega: f64,
               mu: f64, epsilon: f64, sigma: f64,
               stiffness_coef: f64, mass_coef: f64, loss_coef: f64,
               quad_order: u8, exact_sol_known: bool, dim: usize) {
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
        // D718: BOTH components of the essential BC come from the nodal
        // projection of the exact solution (C++ `ProjectBdrCoefficient(u0_r,
        // u0_i, ess_bdr)` evaluates re AND im at the boundary nodes).  The
        // imaginary half was silently zeroed here, homogenizing it — the
        // 2-D path carried both components since D693, 3-D was the straggler.
        let u_proj: Vec<f64> = (0..n).map(|d| {
            let c = dm.dof_coord(d as u32);
            let (re, _im) = u0_exact(&c[..dim], mu, epsilon, sigma, omega);
            re
        }).collect();
        let u_proj_im: Vec<f64> = (0..n).map(|d| {
            let c = dm.dof_coord(d as u32);
            let (_re, im) = u0_exact(&c[..dim], mu, epsilon, sigma, omega);
            im
        }).collect();
        let bc_re: Vec<f64> = ess_bdr.iter().map(|&d| u_proj[d]).collect();
        let bc_im: Vec<f64> = ess_bdr.iter().map(|&d| u_proj_im[d]).collect();
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
        // D693: same fix as the 2-D path — core `compute_l2_error` (D681),
        // quadrature order = MFEM ComputeL2Error convention (2*order + 3).
        let (er, ei) = gf.compute_l2_error(
            &|x| u0_exact(x, mu, epsilon, sigma, omega).0,
            &|x| u0_exact(x, mu, epsilon, sigma, omega).1,
            2 * cfg.order as u8 + 3, &space);
        println!("\n|| Re(u_h-u) ||_{{L^2}} = {:.6e}", er);
        println!("|| Im(u_h-u) ||_{{L^2}} = {:.6e}\n", ei);
    }
    save_output_3d(mesh, &gf);
}

// ─── p=1: HCurl 3D ────────────────────────────────────────────────────────

fn solve_3d_p1(mesh: &Mesh<3>, cfg: &Config, omega: f64,
               mu: f64, epsilon: f64, sigma: f64,
               stiffness_coef: f64, mass_coef: f64, loss_coef: f64,
               quad_order: u8, exact_sol_known: bool, _dim: usize) {
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

    // D694: pc loss term carries its own ω (MFEM lossCoef(ω·σ)); see 2-D p1.
    let neg_mass_coef = omega * omega * epsilon;
    let mut pc_mat = VectorAssembler::assemble_bilinear(
        &space, &[&curl_curl,
                   &VectorMassIntegrator { alpha: neg_mass_coef },
                   &VectorMassIntegrator { alpha: omega * loss_coef }], quad_order);
    // D695: MFEM DIAG_ONE elimination of the essential dofs on the pc matrix
    // before the smoother (same as the 2-D p1 path).
    let mut pc_rhs = vec![0.0; n];
    for &d in &ess_bdr {
        pc_mat.apply_dirichlet_symmetric(d, 0.0, &mut pc_rhs);
    }
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
                                            mu, epsilon, sigma, omega, 2 * cfg.order as u8 + 3);
        println!("\n|| Re(u_h-u) ||_{{L^2}} = {:.6e}", er2);
        println!("|| Im(u_h-u) ||_{{L^2}} = {:.6e}\n", ei2);
    }
    save_output_3d(mesh, &gf);
}

// ─── p=2: HDiv 3D ─────────────────────────────────────────────────────────

fn solve_3d_p2(mesh: &Mesh<3>, cfg: &Config, omega: f64,
               mu: f64, epsilon: f64, sigma: f64,
               stiffness_coef: f64, mass_coef: f64, loss_coef: f64,
               quad_order: u8, exact_sol_known: bool, _dim: usize) {
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
                                           mu, epsilon, sigma, omega, 2 * cfg.order as u8 + 3);
        println!("\n|| Re(u_h-u) ||_{{L^2}} = {:.6e}", er2);
        println!("|| Im(u_h-u) ||_{{L^2}} = {:.6e}\n", ei2);
    }
    save_output_3d(mesh, &gf);
}

// ─── 3D L² error helpers ──────────────────────────────────────────────────

/// L² error for H(Curl) vector fields in 3-D.
///
/// D748 (D674 family): the reference element is the one **paired with the
/// space's own DOF/slot tables** (`fem_assembly::paired_vector_reference_element`
/// — the assembler's dispatch).  The former hard-wired `HexNDk::new(1)` /
/// `TetNDk::new(1)` evaluated every order ≥ 2 space with the ND1 basis (4/6
/// slots consumed, the rest ignored) and every non-Hex8/Tet4 cell with a tet
/// basis, so `-p 1 -o 2` printed 4.177593e-1 (hex) / 2.674999e-1 (tet) against
/// MFEM's 1.563990e-2 / 5.189320e-2 (27× / 5.2×).  The element's DOF count is
/// asserted against the space's slot table so a future pairing drift fails
/// loudly instead of printing a wrong row.  The quadrature order argument is
/// the MFEM `ComputeL2Error` convention (`2*order + 3`).
fn l2_error_hcurl_3d(mesh: &Mesh<3>, space: &HCurlSpace<Mesh<3>>,
                      u_re: &[f64], u_im: &[f64],
                      mu: f64, epsilon: f64, sigma: f64, omega: f64,
                      quad_order: u8) -> (f64, f64) {
    let nd_order = space.order();
    let mut er2 = 0.0; let mut ei2 = 0.0;
    for e in 0..mesh.n_elements() as u32 {
        let et = mesh.element_type(e);
        let re = fem_assembly::paired_vector_reference_element(
            space.space_type(), et, 3, nd_order);
        let nld = re.n_dofs();
        let q = re.quadrature(quad_order);
        let mut phi = vec![0.0; nld * 3];
        let ed: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        assert_eq!(
            ed.len(), nld,
            "l2_error_hcurl_3d: {et:?} slot table has {} dofs, the paired ND{nd_order} element \
             needs {nld} — the space and the evaluator disagree",
            ed.len(),
        );
        // Canonical → element-local DOF values: the signed gather plus the
        // per-face 2×2 rotation (`u_local = T·u_canon`, D37).  Tet NDk face
        // DOFs are stored as the shared face's canonical functionals, so
        // summing `signs·u` against the element's own basis is wrong on every
        // element whose face order differs from the face-creating one — the
        // tet row stayed 2.7× off after the basis fix until this rotation was
        // applied (hex faces reduce to the identity).
        let ur = fem_assembly::vector_assembler::element_local_dofs_canonical(space, e, u_re);
        let ui = fem_assembly::vector_assembler::element_local_dofs_canonical(space, e, u_im);
        for (qi, xi) in q.points.iter().enumerate() {
            re.eval_basis_vec(xi, &mut phi);
            let (J, xp) = element_jacobian_3d(mesh, e, xi);
            let det = J.determinant();
            let w = q.weights[qi] * det.abs();
            // Covariant Piola: u_h(phys) = J^{-T} · u_ref.  MFEM's
            // NDInterpolator/nedelec element transforms use this on every
            // element, not the raw reference basis.
            let jit = J.try_inverse().unwrap_or_default().transpose();
            let mut uh_re = [0.0; 3]; let mut uh_im = [0.0; 3];
            for a in 0..nld {
                let vx = jit[(0, 0)] * phi[a * 3] + jit[(0, 1)] * phi[a * 3 + 1] + jit[(0, 2)] * phi[a * 3 + 2];
                let vy = jit[(1, 0)] * phi[a * 3] + jit[(1, 1)] * phi[a * 3 + 1] + jit[(1, 2)] * phi[a * 3 + 2];
                let vz = jit[(2, 0)] * phi[a * 3] + jit[(2, 1)] * phi[a * 3 + 1] + jit[(2, 2)] * phi[a * 3 + 2];
                uh_re[0] += ur[a] * vx;
                uh_re[1] += ur[a] * vy;
                uh_re[2] += ur[a] * vz;
                uh_im[0] += ui[a] * vx;
                uh_im[1] += ui[a] * vy;
                uh_im[2] += ui[a] * vz;
            }
            let (er, ei) = u0_exact(&xp, mu, epsilon, sigma, omega);
            er2 += w * ((uh_re[0] - er).powi(2) + uh_re[1].powi(2) + uh_re[2].powi(2));
            ei2 += w * ((uh_im[0] - ei).powi(2) + uh_im[1].powi(2) + uh_im[2].powi(2));
        }
    }
    (er2.sqrt(), ei2.sqrt())
}

/// L² error for H(Div) vector fields in 3-D.
///
/// D748 (D674 family): same fix as the H(Curl) 3-D and H(Div) 2-D evaluators —
/// the reference element comes from
/// `fem_assembly::paired_vector_reference_element` (the assembler's dispatch,
/// which keeps D330's Gauss-Legendre nodal pair for Hex8 via
/// `HexRTk::new_gauss_legendre`) instead of the hard-wired `...RTk(0)` /
/// `TetRTk::new(0)`, which silently evaluated every order ≥ 1 space with the
/// RT0 basis (`-p 2 -o 2` hex: 4.568788e-1 vs MFEM 1.595990e-2).  The
/// element's DOF count is asserted against the space's slot table.
fn l2_error_hdiv_3d(mesh: &Mesh<3>, space: &HDivSpace<Mesh<3>>,
                     u_re: &[f64], u_im: &[f64],
                     mu: f64, epsilon: f64, sigma: f64, omega: f64,
                     quad_order: u8) -> (f64, f64) {
    let rt_order = space.order();
    let mut er2 = 0.0; let mut ei2 = 0.0;
    for e in 0..mesh.n_elements() as u32 {
        let et = mesh.element_type(e);
        let re = fem_assembly::paired_vector_reference_element(
            space.space_type(), et, 3, rt_order);
        let nld = re.n_dofs();
        let q = re.quadrature(quad_order);
        let mut phi = vec![0.0; nld * 3];
        let ed: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);
        assert_eq!(
            ed.len(), nld,
            "l2_error_hdiv_3d: {et:?} slot table has {} dofs, the paired RT{rt_order} element \
             needs {nld} — the space and the evaluator disagree",
            ed.len(),
        );
        for (qi, xi) in q.points.iter().enumerate() {
            re.eval_basis_vec(xi, &mut phi);
            let (J, xp) = element_jacobian_3d(mesh, e, xi);
            let det = J.determinant();
            let w = q.weights[qi] * det.abs();
            // Contravariant Piola: u_h(phys) = J·u_ref / det(J).
            let id = 1.0 / det;
            let mut uh_re = [0.0; 3]; let mut uh_im = [0.0; 3];
            for a in 0..nld {
                let s = signs[a];
                let px = id * (J[(0, 0)] * phi[a * 3] + J[(0, 1)] * phi[a * 3 + 1] + J[(0, 2)] * phi[a * 3 + 2]);
                let py = id * (J[(1, 0)] * phi[a * 3] + J[(1, 1)] * phi[a * 3 + 1] + J[(1, 2)] * phi[a * 3 + 2]);
                let pz = id * (J[(2, 0)] * phi[a * 3] + J[(2, 1)] * phi[a * 3 + 1] + J[(2, 2)] * phi[a * 3 + 2]);
                uh_re[0] += s * u_re[ed[a]] * px;
                uh_re[1] += s * u_re[ed[a]] * py;
                uh_re[2] += s * u_re[ed[a]] * pz;
                uh_im[0] += s * u_im[ed[a]] * px;
                uh_im[1] += s * u_im[ed[a]] * py;
                uh_im[2] += s * u_im[ed[a]] * pz;
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
    write_mfem_gf_file("sol_r.gf", 3, &gf.u_re, "H1", 0, 1, 14).ok();
    write_mfem_gf_file("sol_i.gf", 3, &gf.u_im, "H1", 0, 1, 14).ok();
    let n = gf.u_re.len();
    let mut z = vec![0.0_f64; 2 * n];
    for i in 0..n { z[2 * i] = gf.u_re[i]; z[2 * i + 1] = gf.u_im[i]; }
    write_mfem_gf_file("sol_z.gf", 3, &z, "H1", 0, 2, 14).ok();
    let amp: Vec<f64> = gf.amplitude();
    let max_amp = amp.iter().cloned().fold(0.0_f64, f64::max);
    let min_amp = amp.iter().cloned().fold(f64::MAX, f64::min);
    let mean_amp = amp.iter().sum::<f64>() / amp.len() as f64;
    println!("Solution amplitude: max={:.6e} min={:.6e} mean={:.6e}", max_amp, min_amp, mean_amp);
    println!("Wrote refined.mesh, sol_r.gf, sol_i.gf, sol_z.gf");
}
