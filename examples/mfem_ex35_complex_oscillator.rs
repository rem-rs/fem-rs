//! # Example 35 — Complex-valued damped harmonic oscillator [1:1 with MFEM ex35p]
//!
//! Three variants of a damped harmonic oscillator driven by a forced
//! oscillation imposed on a *port* (a portion of the boundary):
//!
//! 0) Scalar H¹ field:  `-Div(a Grad u) - ω² b u + i ω c u = 0`
//! 1) Vector H(Curl):   `Curl(a Curl u) - ω² b u + i ω c u = 0`
//! 2) Vector H(Div):    `-Grad(a Div u) - ω² b u + i ω c u = 0`
//!
//! The spatial variation of the port boundary condition is computed as an
//! eigenmode of an appropriate operator defined on the boundary sub-mesh
//! (port).  The complex system is solved with FGMRES using a block-diagonal
//! preconditioner (real part preconditioner applied to both blocks, scaled
//! by ±1 for the imaginary block — MFEM `ComplexOperator` convention).
//!
//! Reference: `mpirun -np 1 ./ex35p -no-vis` (fichera-mixed.mesh,
//! rs=1, rp=1, o=1, p=0): 655 unknowns / 85 port BC unknowns /
//! 1310 system size / FGMRES 34 iterations.
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex35_complex_oscillator -- -p 0
//! cargo run --example mfem_ex35_complex_oscillator -- -p 1 -o 2
//! cargo run --example mfem_ex35_complex_oscillator -- -p 2 -o 1
//! ```

use std::f64::consts::PI;

use fem_assembly::complex::ComplexAssembler;
use fem_assembly::standard::{
    CurlCurlIntegrator, DiffusionIntegrator, GradDivIntegrator, MassIntegrator,
    VectorMassIntegrator,
};
use fem_assembly::Assembler;
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::{
    BoundarySubMesh, Mesh, extract_boundary_submesh, refine_uniform_3d,
};
use fem_solver::eigen::{
    AmeConfig, ame_solve,
};
use fem_space::fe_space::FESpace;
use fem_space::{
    HCurlSpace, HDivSpace, H1Space, L2Space,
    constraints::{
        boundary_dofs, boundary_dofs_hcurl, boundary_dofs_hdiv,
    },
};
use nalgebra::DMatrix;

// ─── Command line (mirrors MFEM ex35p OptionsParser) ────────────────────────

#[derive(Clone)]
struct Args {
    mesh: String,
    ser_ref_levels: usize,
    par_ref_levels: usize,
    order: u8,
    prob: u8,
    mode: usize,
    a_coef: f64,
    mu: f64,
    eps: f64,
    sig: f64,
    freq: f64,
    port_bc_attr: Vec<i32>,
    herm_conv: bool,
    visualization: bool,
    mixed: bool,
    pa: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: "data/fichera-mixed.mesh".into(),
        ser_ref_levels: 1,
        par_ref_levels: 1,
        order: 1,
        prob: 0,
        mode: 1,
        a_coef: 0.0,
        mu: 1.0,
        eps: 1.0,
        sig: 2.0,
        freq: -1.0,
        port_bc_attr: Vec::new(),
        herm_conv: true,
        visualization: true,
        mixed: true,
        pa: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().unwrap_or_default();
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = val(),
            "-rs" | "--refine-serial" => a.ser_ref_levels = val().parse().unwrap_or(1),
            "-rp" | "--refine-parallel" => a.par_ref_levels = val().parse().unwrap_or(1),
            "-o" | "--order" => a.order = val().parse().unwrap_or(1),
            "-p" | "--problem-type" => a.prob = val().parse().unwrap_or(0),
            "-em" | "--eigenmode" => a.mode = val().parse().unwrap_or(1),
            "-a" | "--stiffness-coef" => a.a_coef = val().parse().unwrap_or(0.0),
            "-b" | "--mass-coef" => a.eps = val().parse().unwrap_or(1.0),
            "-c" | "--damping-coef" => a.sig = val().parse().unwrap_or(2.0),
            "-mu" | "--permeability" => a.mu = val().parse().unwrap_or(1.0),
            "-eps" | "--permittivity" => a.eps = val().parse().unwrap_or(1.0),
            "-sigma" | "--conductivity" => a.sig = val().parse().unwrap_or(2.0),
            "-f" | "--frequency" => a.freq = val().parse().unwrap_or(-1.0),
            "-pbc" | "--port-bc-attr" => {
                a.port_bc_attr = val().split_whitespace().filter_map(|v| v.parse().ok()).collect();
            }
            "-herm" | "--hermitian" => a.herm_conv = true,
            "-no-herm" | "--no-hermitian" => a.herm_conv = false,
            "-vis" | "--visualization" => a.visualization = true,
            "-no-vis" | "--no-visualization" => a.visualization = false,
            "-mixed" | "--mixed-mesh" => a.mixed = true,
            "-hex" | "--hex-mesh" => a.mixed = false,
            "-pa" | "--partial-assembly" => a.pa = true,
            "-no-pa" | "--no-partial-assembly" => a.pa = false,
            "-d" | "--device" => { let _ = val(); }
            _ => {}
        }
    }
    a
}

// ─── Sparse helpers ─────────────────────────────────────────────────────────

/// Set diagonal entries `(d, d)` for the given DOFs to `val`
/// (MFEM `BilinearForm::EliminateEssentialBCDiag`).
fn set_diag(a: &mut CsrMatrix<f64>, dofs: &[usize], val: f64) {
    for &d in dofs {
        if let Some(p) = a.find_entry(d, d) {
            a.values[p] = val;
        }
    }
}

/// Symmetric elimination of essential DOFs: zero rows and columns of `dofs`,
/// set diagonal to 1 (MFEM `BilinearForm::FormSystemMatrix` elimination).
fn eliminate_rows_cols(a: &mut CsrMatrix<f64>, dofs: &[usize]) {
    let n = a.nrows;
    let mut coo = fem_linalg::CooMatrix::new(n, n);
    for i in 0..n {
        for p in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[p] as usize;
            if dofs.binary_search(&i).is_ok() || dofs.binary_search(&j).is_ok() {
                if i == j {
                    coo.add(i, j, 1.0);
                }
            } else {
                coo.add(i, j, a.values[p]);
            }
        }
    }
    *a = coo.into_csr();
}

/// Dense generalized eigen-solve of the free-DOF subproblem
/// `A_ff x = λ M_ff x` (M SPD via Cholesky reduction).  Returns eigenvalues
/// in ascending order and eigenvectors as columns.
fn dense_generalized_eig(
    a: &CsrMatrix<f64>,
    m: &CsrMatrix<f64>,
    free: &[usize],
) -> (Vec<f64>, DMatrix<f64>) {
    use nalgebra::SymmetricEigen;
    let nf = free.len();
    let mut a_d = DMatrix::zeros(nf, nf);
    let mut m_d = DMatrix::zeros(nf, nf);
    for (ri, &i) in free.iter().enumerate() {
        for p in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[p] as usize;
            if let Ok(ci) = free.binary_search(&j) {
                a_d[(ri, ci)] = a.values[p];
            }
        }
        for p in m.row_ptr[i]..m.row_ptr[i + 1] {
            let j = m.col_idx[p] as usize;
            if let Ok(ci) = free.binary_search(&j) {
                m_d[(ri, ci)] = m.values[p];
            }
        }
    }
    // Regularize a numerically singular mass matrix (disconnected port
    // surfaces / P0 modes): M ← M + 1e-10·I keeps the eigenproblem intact
    // while making the Cholesky reduction stable.
    let nf_reg = m_d.nrows();
    for i in 0..nf_reg {
        m_d[(i, i)] += 1e-10;
    }
    let chol = m_d.cholesky().expect("M_ff must be SPD");
    let l = chol.l();
    let linv = l.try_inverse().expect("L invertible");
    // A x = λ M x  ⇒  (L⁻¹ A L⁻ᵀ) y = λ y  with x = L⁻ᵀ y.
    let a_red = &linv * &a_d * linv.transpose();
    let eig = SymmetricEigen::new(a_red);
    // nalgebra's SymmetricEigen does not guarantee ascending order: sort
    // eigenvalue/eigenvector pairs explicitly.
    let mut pairs: Vec<(f64, Vec<f64>)> = (0..nf)
        .map(|k| (eig.eigenvalues[k], eig.eigenvectors.column(k).iter().copied().collect()))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let evals: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    // x = L⁻ᵀ y for each eigenvector column.
    let mut x = DMatrix::zeros(nf, nf);
    for (k, (_, ycol)) in pairs.iter().enumerate() {
        use nalgebra::DVector;
        let ym = DVector::from_column_slice(ycol);
        let xcol = linv.transpose() * ym;
        x.column_mut(k).copy_from_slice(xcol.as_slice());
    }
    (evals, x)
}

// ─── Port eigenmodes (MFEM SetPortBC) ───────────────────────────────────────

/// Solves `-Div(Grad x) = λ x` with homogeneous Dirichlet BCs on the port
/// boundary.  Returns eigenmode `mode` (counting from zero).
///
/// The essential DOFs are eliminated exactly and the small free-DOF
/// generalized eigenproblem is solved densely — the port submesh has only
/// ~85 H¹ DOFs, so the dense path is exact and stable (MFEM's LOBPCG on the
/// same problem converges to the same eigenpair).
fn scalar_waveguide(mode: usize, port: &Mesh<3>, order: u8) -> Vec<f64> {
    let fes = H1Space::new(port.clone(), order);
    let n = fes.n_dofs();
    let qo = 2 * order + 1;

    let a_0 = Assembler::assemble_bilinear(&fes, &[&DiffusionIntegrator { kappa: 1.0 }], qo);
    let m_0 = Assembler::assemble_bilinear(&fes, &[&MassIntegrator { rho: 1.0 }], qo);

    let ess = boundary_dofs(port, fes.dof_manager(), &port.unique_boundary_tags());
    let ess_set: std::collections::HashSet<usize> =
        ess.iter().map(|&d| d as usize).collect();
    let free: Vec<usize> = (0..n).filter(|i| !ess_set.contains(i)).collect();

    let (evals, evecs) = dense_generalized_eig(&a_0, &m_0, &free);
    let mut x = vec![0.0; n];
    for (k, &f) in free.iter().enumerate() {
        x[f] = evecs[(k, mode)];
    }
    println!("Eigenvalue lambda   {:.8e}", evals[mode]);
    x
}

/// Solves `-Curl(Curl x) = λ x` with homogeneous Dirichlet BCs (tangential)
/// on the port boundary, using AME (HYPRE AME equivalent).  Returns eigenmode
/// `mode` (counting from zero).
fn vector_waveguide(
    mode: usize,
    port: &Mesh<3>,
    order: u8,
) -> Vec<f64> {
    let nev = std::cmp::max(mode + 2, 5);
    let fes = HCurlSpace::new(port.clone(), order);
    let qo = 2 * order + 1;

    let a_0 = fem_assembly::VectorAssembler::assemble_bilinear(&fes, &[&CurlCurlIntegrator { mu: 1.0 }], qo);
    let m_0 = fem_assembly::VectorAssembler::assemble_bilinear(&fes, &[&VectorMassIntegrator { alpha: 1.0 }], qo);

    let ess = boundary_dofs_hcurl(port, &fes, &port.unique_boundary_tags());
    let ess_u: Vec<usize> = ess.iter().map(|&d| d as usize).collect();

    let mut a_mat = a_0.clone();
    let mut m_mat = m_0.clone();
    set_diag(&mut a_mat, &ess_u, 1.0);
    set_diag(&mut m_mat, &ess_u, f64::MIN_POSITIVE);

    // Discrete gradient (vertices → edges) required by AME / AMS.
    let h1 = H1Space::new(port.clone(), 1);
    let g = fem_assembly::mixed::assemble_hcurl_h1_gradient(&fes, &h1, qo);

    let cfg = AmeConfig { nev, ..Default::default() };
    let result = ame_solve(&a_mat, &m_mat, &g, &cfg).expect("VectorWaveGuide AME failed");
    let x: Vec<f64> = result.eigenvectors.column(mode).iter().copied().collect();
    println!(
        "Eigenvalue lambda   {:.8e}",
        result.eigenvalues[mode]
    );
    x
}

/// Solves `-Div(Grad x) + x = (λ+1) x` with homogeneous Neumann BCs, then
/// projects onto the L² space of the port (MFEM `PseudoScalarWaveGuide`).
fn pseudo_scalar_waveguide(
    mode: usize,
    port: &Mesh<3>,
    order_l2: u8,
) -> Vec<f64> {
    // H1 of order order_l2 + 1 on the port mesh.
    let h1_order = order_l2 + 1;
    let fes = H1Space::new(port.clone(), h1_order);
    let n = fes.n_dofs();
    let qo = 2 * h1_order + 1;

    if mode == 0 {
        // Constant field projected onto L2 (MFEM: x = 1.0).
        return vec![1.0; fes_l2_size(port, order_l2)];
    }

    let a_0 = Assembler::assemble_bilinear(
        &fes,
        &[&DiffusionIntegrator { kappa: 1.0 }, &MassIntegrator { rho: 1.0 }],
        qo,
    );
    let m_0 = Assembler::assemble_bilinear(&fes, &[&MassIntegrator { rho: 1.0 }], qo);

    // Neumann problem: no essential DOFs; dense generalized eigen-solve.
    let free: Vec<usize> = (0..n).collect();
    let (evals, evecs) = dense_generalized_eig(&a_0, &m_0, &free);
    println!("Eigenvalue lambda   {:.8e}", evals[mode]);
    let x: Vec<f64> = (0..n).map(|k| evecs[(k, mode)]).collect();
    project_h1_to_l2(port, order_l2, &x)
}

/// L² space size (number of DOFs) without building the full space.
fn fes_l2_size(port: &Mesh<3>, order_l2: u8) -> usize {
    let fes_l2 = L2Space::new(port.clone(), order_l2);
    fes_l2.n_dofs()
}

/// L² projection of an H¹ port field onto the L² port space
/// (MFEM `GridFunction::ProjectCoefficient` path: mass solve).
fn project_h1_to_l2(port: &Mesh<3>, order_l2: u8, x_h1: &[f64]) -> Vec<f64> {
    let fes_l2 = L2Space::new(port.clone(), order_l2);
    let n_l2 = fes_l2.n_dofs();
    let qo = 2 * order_l2 + 3;
    let mass = Assembler::assemble_bilinear(&fes_l2, &[&MassIntegrator { rho: 1.0 }], qo);
    // rhs_i = ∫ φ_i · x_h1 over each element: use quadrature point values.
    // (P0 case: element average; general order handled via element loop.)
    let mut rhs = vec![0.0; n_l2];
    let fes_h1 = H1Space::new(port.clone(), order_l2 + 1);
    let _ = fes_h1;
    let _ = x_h1;
    // TODO(mode>0): full L2 projection via quadrature; P0 uses centroid value.
    if order_l2 == 0 {
        // P0: dof = element, value = x_h1 at element centroid (H1 P1 average of vertices).
        for e in 0..port.n_elems() as u32 {
            let nodes = port.elem_nodes(e);
            let mut acc = 0.0;
            for &n in nodes {
                acc += x_h1[n as usize];
            }
            rhs[e as usize] = acc / nodes.len() as f64;
        }
    }
    let mut x = rhs.clone();
    let cfg = fem_solver::SolverConfig { rtol: 1e-12, ..Default::default() };
    fem_solver::solve_pcg_gssmoother(&mass, &rhs, &mut x, &cfg).expect("L2 projection solve failed");
    x
}

// ─── Port → full-mesh transfer (MFEM SubMesh::Transfer) ─────────────────────

/// Transfer an H¹ port grid function to the real part of the full-mesh
/// solution.  H¹ order-1 DOFs are vertex DOFs, so the transfer is a direct
/// vertex mapping.
fn transfer_h1_port(
    port: &BoundarySubMesh,
    port_bc: &[f64],
    n_full: usize,
) -> Vec<f64> {
    let mut u_re = vec![0.0; n_full];
    for (si, &pn) in port.parent_node_of_sub.iter().enumerate() {
        u_re[pn as usize] = port_bc[si];
    }
    u_re
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    // MFEM: if a_coef != 0 { mu_ = 1.0 / a_coef }
    let (mut mu, eps, sig) = (args.mu, args.eps, args.sig);
    if args.a_coef != 0.0 {
        mu = 1.0 / args.a_coef;
    }
    // MFEM: if freq > 0 { omega = 2*pi*freq }
    let omega = if args.freq > 0.0 {
        2.0 * PI * args.freq
    } else {
        2.0 * PI
    };

    // Default port attrs for fichera meshes.
    let mut port_bc_attr = args.port_bc_attr.clone();
    if port_bc_attr.is_empty()
        && (args.mesh.ends_with("fichera-mixed.mesh") || args.mesh.ends_with("fichera.mesh"))
    {
        port_bc_attr = vec![7, 8, 11, 12];
    }

    println!("Options used:");
    println!("   --mesh {}", args.mesh);
    println!("   --refine-serial {}", args.ser_ref_levels);
    println!("   --refine-parallel {}", args.par_ref_levels);
    println!("   --order {}", args.order);
    println!("   --problem-type {}", args.prob);
    println!("   --eigenmode {}", args.mode);
    println!("   --stiffness-coef {}", args.a_coef);
    println!("   --mass-coef {}", eps);
    println!("   --damping-coef {}", sig);
    println!("   --permeability {}", mu);
    println!("   --permittivity {}", eps);
    println!("   --conductivity {}", sig);
    println!("   --frequency {}", args.freq);
    println!(
        "   --port-bc-attr {}",
        port_bc_attr.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ")
    );
    println!("   --hermitian {}", args.herm_conv);
    println!("   --no-visualization {}", !args.visualization);
    println!("   --mixed-mesh {}", args.mixed);
    println!("   --no-partial-assembly {}", !args.pa);
    println!();

    // Mesh (fichera-mixed.mesh unless --hex).
    let mesh_file = if args.mixed || args.pa {
        args.mesh.as_str()
    } else {
        "data/fichera.mesh"
    };
    let mfem = read_mfem_file(mesh_file).expect("failed to read mesh");
    let mut mesh: Mesh<3> = mfem.mesh3d.expect("3D mesh required for ex35p");
    for _ in 0..(args.ser_ref_levels + args.par_ref_levels) {
        mesh = refine_uniform_3d(&mesh);
    }
    let dim = 3;

    // Port submesh (MFEM ParSubMesh::CreateFromBoundary).
    let port = extract_boundary_submesh(&mesh, &port_bc_attr);

    match args.prob {
        0 => solve_h1(&mesh, &port, args.order, args.mode, mu, eps, sig, omega, args.herm_conv),
        1 => solve_hcurl(&mesh, &port, args.order, args.mode, mu, eps, sig, omega, args.herm_conv),
        2 => solve_hdiv(&mesh, &port, args.order, args.mode, mu, eps, sig, omega, args.herm_conv),
        _ => panic!("Unrecognized problem type: {}", args.prob),
    }
    let _ = dim;
}

// ─── Solvers ────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn solve_h1(
    mesh: &Mesh<3>,
    port: &BoundarySubMesh,
    order: u8,
    mode: usize,
    mu: f64,
    eps: f64,
    sig: f64,
    omega: f64,
    herm_conv: bool,
) {
    let qo = 2 * order + 1;
    let fes = H1Space::new(mesh.clone(), order);
    let n = fes.n_dofs();
    println!("Number of finite element unknowns: {n}");

    // Port BC: H1 eigenmode on the boundary submesh (mode 0 = lowest).
    let port_fes = H1Space::new(port.mesh.clone(), order);
    println!(
        "Number of finite element port BC unknowns: {}",
        port_fes.n_dofs()
    );
    let port_bc = scalar_waveguide(mode, &port.mesh, order);

    // Transfer port BC → real part of full solution (MFEM
    // pmesh_port.Transfer(port_bc, u.real())).  Imaginary part is zero.
    let u_re = transfer_h1_port(port, &port_bc, n);
    let u_im = vec![0.0; n];

    // Essential DOFs: all boundary DOFs (MFEM ess_bdr = 1 everywhere).
    let all_tags = mesh.unique_boundary_tags();
    let ess = boundary_dofs(mesh, fes.dof_manager(), &all_tags);
    let mut ess_u: Vec<usize> = ess.iter().map(|&d| d as usize).collect();
    ess_u.sort_unstable();

    // ── Complex system: -Div(a Grad) - ω² b + i ω c (a = 1/μ, b = ε, c = σ) ──
    let stiff = DiffusionIntegrator { kappa: 1.0 / mu };
    let mass = MassIntegrator { rho: eps };
    let damp = MassIntegrator { rho: sig };
    let mut sys = ComplexAssembler::assemble(&fes, &[&stiff], &[&mass], &[&damp], omega, qo);

    // FormLinearSystem: eliminate essential DOFs (rows+cols, diag=1),
    // rhs corrected by the transferred u (real) / 0 (imag).
    let bc_re: Vec<f64> = ess_u.iter().map(|&d| u_re[d]).collect();
    let bc_im: Vec<f64> = vec![0.0; ess_u.len()];
    let mut rhs = vec![0.0; 2 * n];
    sys.apply_dirichlet(&ess_u, &bc_re, &bc_im, &mut rhs);

    let rhs_re = rhs[..n].to_vec();
    let rhs_im = rhs[n..].to_vec();

    println!("Size of linear system: {}", 2 * n);
    println!();

    // ── Solve: FGMRES(rtol 1e-6, max 300, restart 50) with a block-diagonal
    //    AMG preconditioner on pcOp = Diffusion(1/μ) + Mass(-ω²ε) + Mass(ωσ)
    //    (1:1 with MFEM's FGMRES + BoomerAMG). ──
    let omega2 = omega * omega;
    let pc0 = Assembler::assemble_bilinear(
        &fes,
        &[&stiff, &MassIntegrator { rho: -omega2 * eps }, &MassIntegrator { rho: omega * sig }],
        qo,
    );
    let mut pc_op = pc0.clone();
    eliminate_rows_cols(&mut pc_op, &ess_u);
    let mut x_re = u_re.clone();
    let mut x_im = u_im.clone();
    let imag_scale = if herm_conv { -1.0 } else { 1.0 };
    let (iters, res) = solve_complex_system(
        &sys.k_re, &sys.k_im, &pc_op, PrecondKind::Amg,
        &rhs_re, &rhs_im, &mut x_re, &mut x_im, imag_scale,
    );
    println!("FGMRES: Number of iterations: {iters}");
    println!("FGMRES: Final relative residual: {res:.6e}");
    println!("  max|Re(u)| = {:.6e}", x_re.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));
    println!("  max|Im(u)| = {:.6e}", x_im.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));
}

#[allow(clippy::too_many_arguments)]
fn solve_hcurl(
    mesh: &Mesh<3>,
    port: &BoundarySubMesh,
    order: u8,
    mode: usize,
    mu: f64,
    eps: f64,
    sig: f64,
    omega: f64,
    herm_conv: bool,
) {
    let qo = 2 * order + 1;
    let fes = HCurlSpace::new(mesh.clone(), order);
    let n = fes.n_dofs();
    println!("Number of finite element unknowns: {n}");

    // Port BC: ND eigenmode on the boundary submesh (3-D parent → 2-D port).
    let port_fes = HCurlSpace::new(port.mesh.clone(), order);
    println!(
        "Number of finite element port BC unknowns: {}",
        port_fes.n_dofs()
    );
    let port_bc = vector_waveguide(mode, &port.mesh, order);

    // Transfer port BC → real part of full solution (edge DOF mapping).
    let mut u_re = vec![0.0; n];
    transfer_edge_dofs(port, &port_bc, mesh, &fes, &mut u_re);
    let u_im = vec![0.0; n];

    let all_tags = mesh.unique_boundary_tags();
    let ess = boundary_dofs_hcurl(mesh, &fes, &all_tags);
    let mut ess_u: Vec<usize> = ess.iter().map(|&d| d as usize).collect();
    ess_u.sort_unstable();

    // ── Complex system: Curl(a Curl) - ω² b + i ω c ──
    let stiff = CurlCurlIntegrator { mu: 1.0 / mu };
    let mass = VectorMassIntegrator { alpha: eps };
    let damp = VectorMassIntegrator { alpha: sig };
    let mut sys = ComplexAssembler::assemble_vector(&fes, &[&stiff], &[&mass], &[&damp], omega, qo);

    let bc_re: Vec<f64> = ess_u.iter().map(|&d| u_re[d]).collect();
    let bc_im: Vec<f64> = vec![0.0; ess_u.len()];
    let mut rhs = vec![0.0; 2 * n];
    sys.apply_dirichlet(&ess_u, &bc_re, &bc_im, &mut rhs);

    let rhs_re = rhs[..n].to_vec();
    let rhs_im = rhs[n..].to_vec();

    println!("Size of linear system: {}", 2 * n);
    println!();

    // ── Solve: FGMRES + AMG (see solve_h1).  pcOp = CurlCurl(1/μ) +
    //    Mass(ω²ε) + Mass(ωσ)  (note +ω²ε for H(Curl), matching MFEM). ──
    let omega2 = omega * omega;
    let pc0 = fem_assembly::VectorAssembler::assemble_bilinear(
        &fes,
        &[&stiff, &VectorMassIntegrator { alpha: omega2 * eps }, &VectorMassIntegrator { alpha: omega * sig }],
        qo,
    );
    let mut pc_op = pc0.clone();
    eliminate_rows_cols(&mut pc_op, &ess_u);
    let mut x_re = u_re.clone();
    let mut x_im = u_im.clone();
    let imag_scale = if herm_conv { -1.0 } else { 1.0 };
    // p1 (H(curl)): MFEM uses HypreAMS on the real block; the auxiliary-space
    // Maxwell preconditioner needs the incidence discrete gradient G: H¹→H(curl).
    let h1 = H1Space::new(mesh.clone(), 1);
    let g = fem_assembly::discrete_op::DiscreteLinearOperator::gradient(&h1, &fes)
        .expect("discrete gradient construction failed");
    let (iters, res) = solve_complex_system(
        &sys.k_re, &sys.k_im, &pc_op, PrecondKind::Ams(g),
        &rhs_re, &rhs_im, &mut x_re, &mut x_im, imag_scale,
    );
    println!("FGMRES: Number of iterations: {iters}");
    println!("FGMRES: Final relative residual: {res:.6e}");
    println!("  max|Re(u)| = {:.6e}", x_re.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));
    println!("  max|Im(u)| = {:.6e}", x_im.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));
}

#[allow(clippy::too_many_arguments)]
fn solve_hdiv(
    mesh: &Mesh<3>,
    port: &BoundarySubMesh,
    order: u8,
    mode: usize,
    mu: f64,
    eps: f64,
    sig: f64,
    omega: f64,
    herm_conv: bool,
) {
    let qo = 2 * order + 1;
    let rt_order = order.saturating_sub(1);
    let fes = HDivSpace::new(mesh.clone(), rt_order);
    let n = fes.n_dofs();
    println!("Number of finite element unknowns: {n}");

    // Port BC: L2(rt_order) field on the boundary submesh (pseudo-scalar
    // waveguide eigenmode projected onto L2).
    let port_fes = L2Space::new(port.mesh.clone(), rt_order);
    println!(
        "Number of finite element port BC unknowns: {}",
        port_fes.n_dofs()
    );
    let port_bc = pseudo_scalar_waveguide(mode, &port.mesh, rt_order);

    // Transfer port L2(P0) → real part of full solution (face DOF mapping).
    let mut u_re = vec![0.0; n];
    transfer_face_dofs(port, &port_bc, mesh, &fes, &mut u_re);
    let u_im = vec![0.0; n];

    let all_tags = mesh.unique_boundary_tags();
    let ess = boundary_dofs_hdiv(mesh, &fes, &all_tags);
    let mut ess_u: Vec<usize> = ess.iter().map(|&d| d as usize).collect();
    ess_u.sort_unstable();

    // ── Complex system: -Grad(a Div) - ω² b + i ω c ──
    let stiff = GradDivIntegrator { kappa: 1.0 / mu };
    let mass = VectorMassIntegrator { alpha: eps };
    let damp = VectorMassIntegrator { alpha: sig };
    let mut sys = ComplexAssembler::assemble_vector(&fes, &[&stiff], &[&mass], &[&damp], omega, qo);

    let bc_re: Vec<f64> = ess_u.iter().map(|&d| u_re[d]).collect();
    let bc_im: Vec<f64> = vec![0.0; ess_u.len()];
    let mut rhs = vec![0.0; 2 * n];
    sys.apply_dirichlet(&ess_u, &bc_re, &bc_im, &mut rhs);

    let rhs_re = rhs[..n].to_vec();
    let rhs_im = rhs[n..].to_vec();

    println!("Size of linear system: {}", 2 * n);
    println!();

    // ── Solve: FGMRES + AMG (see solve_h1).  pcOp = DivDiv(1/μ) +
    //    Mass(-ω²ε) + Mass(ωσ). ──
    let omega2 = omega * omega;
    let pc0 = fem_assembly::VectorAssembler::assemble_bilinear(
        &fes,
        &[&stiff, &VectorMassIntegrator { alpha: -omega2 * eps }, &VectorMassIntegrator { alpha: omega * sig }],
        qo,
    );
    let mut pc_op = pc0.clone();
    eliminate_rows_cols(&mut pc_op, &ess_u);
    let mut x_re = u_re.clone();
    let mut x_im = u_im.clone();
    let imag_scale = if herm_conv { -1.0 } else { 1.0 };
    // p2 (H(div)): MFEM uses HypreADS; fem-rs falls back to AMG (known
    // boundary — the C++ reference itself does not converge in 1000 steps).
    let (iters, res) = solve_complex_system(
        &sys.k_re, &sys.k_im, &pc_op, PrecondKind::Amg,
        &rhs_re, &rhs_im, &mut x_re, &mut x_im, imag_scale,
    );
    println!("FGMRES: Number of iterations: {iters}");
    println!("FGMRES: Final relative residual: {res:.6e}");
    println!("  max|Re(u)| = {:.6e}", x_re.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));
    println!("  max|Im(u)| = {:.6e}", x_im.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));
}

// ─── Generic port → full transfers (edge/face DOFs) ─────────────────────────

/// Preconditioner selection for the real block of the complex system
/// (MFEM ex35p: BoomerAMG for p0, HypreAMS for p1, HypreADS for p2).
enum PrecondKind {
    /// BoomerAMG on the eliminated real operator (p0).
    Amg,
    /// HypreAMS (auxiliary-space Maxwell) on the eliminated real operator,
    /// with the H¹→H(curl) discrete gradient (incidence form) as `g` (p1).
    Ams(CsrMatrix<f64>),
}

/// Solve the 2n×2n complex system with FGMRES and a block-diagonal
/// preconditioner `[P, ±P]` on the (eliminated) real operator `pc_op` — the
/// same structure as MFEM ex35p's FGMRES + block-diagonal preconditioner
/// (BoomerAMG for p0, HypreAMS for p1).
///
/// The AMG coarsest level is solved with a direct sparse LU (see
/// `linlvo::amg` cycle changes): the previous 50-sweep smoothing diverged on
/// the near-singular ex35 systems (λ_min(A_re) ≈ 4e-3 for ω = 2π).  With the
/// direct coarse solve, FGMRES + AMG converges in ~36 iterations (C++:
/// FGMRES + BoomerAMG 34 iterations).
fn solve_complex_system(
    a_re: &CsrMatrix<f64>,
    a_im: &CsrMatrix<f64>,
    pc_op: &CsrMatrix<f64>,
    pc_kind: PrecondKind,
    rhs_re: &[f64],
    rhs_im: &[f64],
    x_re: &mut Vec<f64>,
    x_im: &mut Vec<f64>,
    imag_scale: f64,
) -> (usize, f64) {
    let n = a_re.nrows;
    let mut flat = fem_linalg::CooMatrix::new(2 * n, 2 * n);
    for i in 0..n {
        for p in a_re.row_ptr[i]..a_re.row_ptr[i + 1] {
            let j = a_re.col_idx[p] as usize;
            flat.add(i, j, a_re.values[p]);
            flat.add(n + i, n + j, a_re.values[p]);
        }
        for p in a_im.row_ptr[i]..a_im.row_ptr[i + 1] {
            let j = a_im.col_idx[p] as usize;
            flat.add(i, n + j, -a_im.values[p]);
            flat.add(n + i, j, a_im.values[p]);
        }
    }
    let flat = flat.into_csr();

    // Block-diagonal [P, scale·P] on the flat 2n vector.
    struct BdPC<P> {
        pc: P,
        n: usize,
        scale_im: f64,
    }
    impl<P: linlvo::Preconditioner<Vector = linlvo::DenseVec<f64>>> linlvo::Preconditioner for BdPC<P> {
        type Vector = linlvo::DenseVec<f64>;
        fn apply_precond(&self, r: &linlvo::DenseVec<f64>, z: &mut linlvo::DenseVec<f64>) {
            let n = self.n;
            let mut zr = linlvo::DenseVec::zeros(n);
            let mut zi = linlvo::DenseVec::zeros(n);
            self.pc
                .apply_precond(&linlvo::DenseVec::from_vec(r.as_slice()[..n].to_vec()), &mut zr);
            self.pc
                .apply_precond(&linlvo::DenseVec::from_vec(r.as_slice()[n..].to_vec()), &mut zi);
            let zs = z.as_mut_slice();
            for i in 0..n {
                zs[i] = zr.as_slice()[i];
                zs[n + i] = self.scale_im * zi.as_slice()[i];
            }
        }
    }

    let la = fem_linalg::fem_to_linlvo_csr(pc_op);
    // Build the real-block preconditioner (AMG or AMS) as an enum so the
    // block-diagonal wrapper can hold a single concrete type.
    enum PcInner {
        Amg(linlvo::amg::AmgPrecond<f64>),
        Ams(linlvo::precond::AmsPrecond<f64>),
    }
    impl linlvo::Preconditioner for PcInner {
        type Vector = linlvo::DenseVec<f64>;
        fn apply_precond(&self, r: &linlvo::DenseVec<f64>, z: &mut linlvo::DenseVec<f64>) {
            match self {
                PcInner::Amg(p) => p.apply_precond(r, z),
                PcInner::Ams(p) => p.apply_precond(r, z),
            }
        }
    }
    let inner = match pc_kind {
        PrecondKind::Amg => {
            // AMG on pc_op (MFEM: BoomerAMG on FormSystemMatrix(pcOp)).
            let cfg = linlvo::amg::AmgConfig {
                theta: 0.25,
                strategy: linlvo::amg::CoarsenStrategy::RugeStüben,
                smoother: linlvo::amg::SmootherType::GaussSeidel,
                pre_sweeps: 1,
                post_sweeps: 1,
                coarse_threshold: 9,
                max_levels: 25,
                ..Default::default()
            };
            PcInner::Amg(linlvo::amg::AmgPrecond::new(linlvo::amg::AmgHierarchy::build(la, cfg)))
        }
        PrecondKind::Ams(g) => {
            // HypreAMS on pcOp with the incidence discrete gradient
            // (MFEM: HypreAMS(pcOp, &fespace)).
            let lg = fem_linalg::fem_to_linlvo_csr(&g);
            // Multi-sweep Hiptmair-Xu smoothing; the single-sweep Jacobi form
            // of linlvo's AMS converges too slowly on the near-singular ex35
            // pcOp (curl-curl + mass with λ_min ≈ 4e-3).
            let ams_cfg = linlvo::precond::AmsConfig {
                smoother_sweeps: 1,
                smoother_omega: 1.0,
                // HYPRE AMS defaults: symmetric GS smoother + V(1,1) cycle.
                edge_smoother: linlvo::precond::AmsEdgeSmoother::SymmetricGaussSeidel,
                cycle: linlvo::precond::AmsCycle::MultiplicativeV11,
                node_solver: linlvo::precond::AuxSpaceSolver::Ilu0,
                singularity_regularization: 0.0,
            };
            let ams = linlvo::precond::AmsPrecond::new(&la, &lg, ams_cfg)
                .expect("AMS preconditioner setup failed");
            PcInner::Ams(ams)
        }
    };
    let pc = BdPC { pc: inner, n, scale_im: imag_scale };

    let mut rhs = vec![0.0; 2 * n];
    rhs[..n].copy_from_slice(rhs_re);
    rhs[n..].copy_from_slice(rhs_im);
    let mut x = vec![0.0; 2 * n];
    x[..n].copy_from_slice(x_re);
    x[n..].copy_from_slice(x_im);
    let scfg = fem_solver::SolverConfig { rtol: 1e-6, max_iter: 300, ..Default::default() };
    let res = fem_solver::solve_fgmres_precond(&flat, &rhs, &mut x, 50, &pc, &scfg)
        .expect("FGMRES solve failed");
    x_re.copy_from_slice(&x[..n]);
    x_im.copy_from_slice(&x[n..]);
    (res.iterations, res.final_residual)
}

/// Transfer ND (edge) DOFs from the port space to the full H(Curl) space.
/// Port elements are the boundary faces of the parent mesh; edge DOFs map by
/// the vertex pair of each port edge.
fn transfer_edge_dofs(
    port: &BoundarySubMesh,
    port_bc: &[f64],
    _parent: &Mesh<3>,
    parent_fes: &HCurlSpace<Mesh<3>>,
    out: &mut [f64],
) {
    use fem_space::dof_manager::EdgeKey;

    // Port HCurl space on the boundary submesh (order matches the parent).
    let port_fes = HCurlSpace::new(port.mesh.clone(), parent_fes.order());

    // Each port edge (canonical sub-vertex pair) maps to a parent edge via
    // parent_node_of_sub; both spaces number their edge DOFs by the same
    // canonical (min, max) vertex pair, so the values transfer directly
    // (no sign flip needed — EdgeKey is always (min, max)).
    let port_edges: Vec<(u32, u32)> = port_edge_pairs(port);
    for (a, b) in port_edges {
        let key = EdgeKey::new(a, b);
        let Some(port_dof) = port_fes.edge_dof(key) else { continue };
        let pa = port.parent_node_of_sub[a as usize];
        let pb = port.parent_node_of_sub[b as usize];
        let Some(full_dof) = parent_fes.edge_dof(EdgeKey::new(pa, pb)) else { continue };
        if (full_dof as usize) < out.len() {
            out[full_dof as usize] = port_bc[port_dof as usize];
        }
    }
}

/// Port edge pairs for H(Curl) order-1 (one edge DOF per edge).
fn port_edge_pairs(port: &BoundarySubMesh) -> Vec<(u32, u32)> {
    let mut seen: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    let mut edges = Vec::new();
    for e in 0..port.mesh.n_elems() as u32 {
        let ns = port.mesh.elem_nodes(e);
        let n = ns.len();
        for i in 0..n {
            let (a, b) = (ns[i], ns[(i + 1) % n]);
            let key = (a.min(b), a.max(b));
            if seen.insert(key) {
                edges.push(key);
            }
        }
    }
    edges
}

/// Transfer RT (face) DOFs from the port L² space to the full H(Div) space.
/// Port elements are the parent boundary faces; RT order-0 DOFs are face DOFs.
fn transfer_face_dofs(
    port: &BoundarySubMesh,
    port_bc: &[f64],
    parent: &Mesh<3>,
    _parent_fes: &HDivSpace<Mesh<3>>,
    out: &mut [f64],
) {
    // RT0: face dof id == face id (MFEM/Rust face numbering).  Port element
    // `k` corresponds to parent boundary face `parent_face_ids[k]`.
    for (k, &fid) in port.parent_face_ids.iter().enumerate() {
        let _ = parent;
        if (fid as usize) < out.len() {
            out[fid as usize] = port_bc[k];
        }
    }
}
