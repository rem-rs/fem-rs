//! Example 19 — 1:1 translation of MFEM ex19
//! Quasi-static incompressible neo-Hookean hyperelasticity (mixed u/p).
//!
//! Solves H(x) = 0 via MFEM's `NewtonSolver` semantics (full steps, no line
//! search) where each Jacobian solve is a **left-preconditioned** MFEM-faithful
//! `GMRESSolver` (`mfem_gmres`) with the example's `JacobianPreconditioner`
//! block elimination (mass CG+GSSmoother, stiffness GMRES+GSSmoother,
//! gamma = 1e-5), monitored exactly like `GeneralResidualMonitor`.
//!
//! BCs (matching MFEM ex19):
//!   Boundary attribute 1: u = 0 (fixed)
//!   Boundary attribute 2: u_x = 0, u_y = 0.25·x (prescribed shear)
//!
//! Usage:
//!   cargo run --example mfem_ex19_hyperelastic_incomp
//!   cargo run --example mfem_ex19_hyperelastic_incomp -- -m data/beam-quad.mesh -o 2 -r 0
//!   cargo run --example mfem_ex19_hyperelastic_incomp -- -mu 1.0 -rel 1e-4 -abs 1e-6 -it 500

#![allow(non_snake_case)]

use fem_assembly::physics::mixed_hyperelasticity::MixedHyperelasticityForm;
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_gf_file};
use fem_linalg::{BlockMatrix, CooMatrix, CsrMatrix, SolverConfig};
use fem_solver::block_operator::{
    mfem_gmres, mfem_gmres_monitored, IterativeSolverMonitor, MfemBlockOperator, MfemGmresConfig,
};
use fem_solver::smoother::{GsSmoother, GsType};
use fem_solver::{fmt_g, solve_pcg_gssmoother};
use fem_mesh::{geometry_jacobian, refine_uniform, MeshTopology};
use fem_space::{constraints::boundary_dofs, fe_space::FESpace, H1Space, VectorH1Space};

/// MFEM ex19 numbers the displacement space with `Ordering::byVDIM`
/// (`ex19.cpp`: `FiniteElementSpace R_space(mesh, &quad_coll, dim,
/// Ordering::byVDIM)`), whose true-DOF index is `dim·s + c` for scalar DOF `s`
/// and component `c`.  `VectorH1Space` numbers its DOFs component-major
/// (`c·ns + s`), so the two differ by this fixed permutation.
///
/// Re-ordering the assembled system into MFEM's indexing puts every Krylov dot
/// product, Gauss-Seidel sweep and residual norm in the reference summation
/// order.  Measured on the Newton-0 anchor: `‖z_u‖` moves from `2.3164065128747025`
/// (blocked, 5e-11 off MFEM) to `2.3164065129917222` (byVDIM, ~1 ulp off MFEM's
/// `2.3164065129917213`).  (Library follow-up: `VectorH1Space` should expose the
/// ordering directly — see the rr157 lane2 report.)
fn byvdim_perm(ns: usize, dim: usize) -> Vec<usize> {
    let mut p = vec![0usize; ns * dim];
    for s in 0..ns {
        for c in 0..dim {
            p[c * ns + s] = dim * s + c;
        }
    }
    p
}

/// `A'[pr[i]][pc[j]] = A[i][j]` for a CSR matrix.
fn permute_csr(
    a: &CsrMatrix<f64>,
    pr: &[usize],
    pc: &[usize],
    nrow: usize,
    ncol: usize,
) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(nrow, ncol);
    for i in 0..a.nrows {
        for k in a.row_ptr[i]..a.row_ptr[i + 1] {
            coo.add(pr[i], pc[a.col_idx[k] as usize], a.values[k]);
        }
    }
    coo.into_csr()
}

/// Re-index the `[K_uu K_up; K_pu K_pp]` block matrix into MFEM's byVDIM
/// ordering (the pressure block is scalar and keeps its own ordering).
fn permute_block_matrix(jac: &BlockMatrix, perm: &[usize], nu: usize, np: usize) -> BlockMatrix {
    let ident: Vec<usize> = (0..np).collect();
    let mut bm = BlockMatrix::new_square(vec![nu, np]);
    if let Some(k) = jac.get(0, 0) {
        bm.set(0, 0, permute_csr(k, perm, perm, nu, nu));
    }
    if let Some(k) = jac.get(0, 1) {
        bm.set(0, 1, permute_csr(k, perm, &ident, nu, np));
    }
    if let Some(k) = jac.get(1, 0) {
        bm.set(1, 0, permute_csr(k, &ident, perm, np, nu));
    }
    if let Some(k) = jac.get(1, 1) {
        bm.set(1, 1, permute_csr(k, &ident, &ident, np, np));
    }
    bm
}


/// Euclidean norm of a slice.
fn nr(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// ex19.cpp `GeneralResidualMonitor` — 1:1 port (`IterativeSolverMonitor`
/// subclass printing `"{prefix} iteration {it,setw(2)} : ||r|| = {norm}"`,
/// with `||r||/||r_0||` appended for `it > 0`; `norm0` is captured at the
/// `it == 0` print).
struct GeneralResidualMonitor {
    prefix: &'static str,
    print_level: i32,
    norm0: f64,
}

impl GeneralResidualMonitor {
    fn new(prefix: &'static str, print_level: i32) -> Self {
        Self { prefix, print_level, norm0: 0.0 }
    }
}

impl IterativeSolverMonitor for GeneralResidualMonitor {
    fn monitor_residual(&mut self, it: usize, norm: f64, final_: bool) {
        if (self.print_level == 1 && !final_) || (self.print_level == 3 && (final_ || it == 0)) {
            let mut line = format!("{} iteration {:>2} : ||r|| = {}", self.prefix, it, fmt_g(norm));
            if it > 0 {
                line.push_str(&format!(",  ||r||/||r_0|| = {}", fmt_g(norm / self.norm0)));
            } else {
                self.norm0 = norm;
            }
            println!("{line}");
        }
    }
}

/// ex19.cpp `JacobianPreconditioner::Mult` — 1:1 port (shared by the Newton
/// loop and the anchor test):
///
/// ```text
/// mass_pcg->Mult(pres_in, pres_out);        // CG + GSSmoother
/// pres_out *= -gamma;
/// jacobian->GetBlock(0,1).Mult(pres_out, temp);
/// subtract(disp_in, temp, temp2);
/// stiff_pcg->Mult(temp2, disp_out);         // GMRES + GSSmoother on K_uu
/// ```
#[allow(clippy::too_many_arguments)]
fn apply_jacobian_preconditioner(
    r: &[f64],
    z: &mut [f64],
    kuu: &CsrMatrix<f64>,
    kup: &CsrMatrix<f64>,
    mp: &CsrMatrix<f64>,
    stiff_prec: &GsSmoother,
    gamma: f64,
    nu: usize,
    np: usize,
    s_cfg: &SolverConfig,
    stiff_cfg: MfemGmresConfig,
) {
    let mut pres_out = vec![0.0_f64; np];
    let _ = solve_pcg_gssmoother(mp, &r[nu..], &mut pres_out, s_cfg);
    for v in pres_out.iter_mut() {
        *v *= -gamma;
    }
    let mut temp = vec![0.0_f64; nu];
    for i in 0..nu {
        // MFEM `SparseMatrix::Mult` — row storage order.
        let mut s = 0.0;
        for k in kup.row_ptr[i]..kup.row_ptr[i + 1] {
            s += kup.values[k] * pres_out[kup.col_idx[k] as usize];
        }
        temp[i] = s;
    }
    let mut temp2 = vec![0.0_f64; nu];
    for i in 0..nu {
        temp2[i] = r[i] - temp[i];
    }
    let mut disp_out = vec![0.0_f64; nu];
    let _ = mfem_gmres(
        kuu,
        &temp2,
        &mut disp_out,
        stiff_cfg,
        Some(&|rr: &[f64], zz: &mut [f64]| stiff_prec.mult(rr, zz)),
    );
    z[..nu].copy_from_slice(&disp_out);
    z[nu..].copy_from_slice(&pres_out);
}

// ─── Pressure mass matrix (MFEM: BilinearForm(MassIntegrator) on pressure space)
//
// Quadrature: MFEM `MassIntegrator::GetRule` picks
// `order = p + p + Trans.OrderW()` (bilininteg.cpp:1464) and beam-tet is
// straight-sided, i.e. `OrderW() == 0` — so the integration rule is the
// order-`2*p` one, NOT the order-`2*order+3` rule the mixed form uses.
fn build_pressure_mass(
    mesh: impl fem_mesh::MeshTopology + 'static,
    p_order: u8,
    quad_order: u8,
    np: usize,
) -> CsrMatrix<f64> {
    use fem_space::fe_space::FESpace;
    let space = H1Space::new(mesh.clone_mesh(), p_order);
    let mut coo = CooMatrix::<f64>::new(np, np);
    let ne = mesh.n_elements() as usize;
    for e in 0..ne {
        let et = mesh.element_type(e as u32);
        let ref_elem = et.ref_elem(p_order);
        let n_ldofs = ref_elem.n_dofs();
        let edofs: Vec<usize> = space.element_dofs(e as u32)
            .iter().map(|&d| d as usize).collect();
        let q = ref_elem.quadrature(quad_order);
        let mut phi = vec![0.0_f64; n_ldofs];
        let mut me = vec![0.0_f64; n_ldofs * n_ldofs];
        for (qi, xi) in q.points.iter().enumerate() {
            ref_elem.eval_basis(xi, &mut phi);
            let (det_j, _ji) = geometry_jacobian(&mesh, e as u32, xi, mesh.dim() as usize);
            let w = q.weights[qi] * det_j.abs();
            for i in 0..n_ldofs {
                for j in 0..n_ldofs {
                    me[i * n_ldofs + j] += w * phi[i] * phi[j];
                }
            }
        }
        for a in 0..n_ldofs {
            for b in 0..n_ldofs {
                coo.add(edofs[a], edofs[b], me[a * n_ldofs + b]);
            }
        }
    }
    coo.into_csr()
}

/// Format an f64 like C++ `os << value` with default precision (6 significant
/// digits, scientific notation with a 2-digit signed exponent when the decimal
/// exponent is < -4 or >= 6) — MFEM's OptionsParser::WriteValue prints option
/// doubles this way, e.g. 1e-6 → "1e-06", 0.0001 → "0.0001", 1.0 → "1".
fn cpp_double(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    let exp = v.abs().log10().floor() as i32;
    if exp < -4 || exp >= 6 {
        let mant = v / 10f64.powi(exp);
        let mut s = format!("{:.5}", mant);
        if s.contains('.') {
            while s.ends_with('0') {
                s.pop();
            }
            if s.ends_with('.') {
                s.pop();
            }
        }
        format!("{s}e{}{:02}", if exp < 0 { "-" } else { "+" }, exp.abs())
    } else {
        let decimals = (5 - exp).max(0) as usize;
        let mut s = format!("{v:.decimals$}");
        if s.contains('.') {
            while s.ends_with('0') {
                s.pop();
            }
            if s.ends_with('.') {
                s.pop();
            }
        }
        s
    }
}

fn main() {
    let args = Args::parse();
    // C++ ex19 prints `args.PrintOptions(cout)` (ex19.cpp:219) as its very
    // first output; the echo mirrors OptionsParser::PrintOptions byte-for-byte
    // (replaces this port's former ad-hoc banner line).
    println!("Options used:");
    println!("   --mesh {}", args.mesh);
    println!("   --refine {}", args.refine);
    println!("   --order {}", args.order);
    println!("   {}", if args.visualization { "--visualization" } else { "--no-visualization" });
    println!("   --relative-tolerance {}", cpp_double(args.rel_tol));
    println!("   --absolute-tolerance {}", cpp_double(args.abs_tol));
    println!("   --newton-iterations {}", args.max_iter);
    println!("   --shear-modulus {}", cpp_double(args.mu));

    // 1. Read mesh (2D or 3D)
    let mfem = read_mfem_file(&args.mesh).expect("failed to read mesh");
    let dim: usize;
    let mesh3d: bool;
    let mesh: Box<dyn fem_mesh::MeshTopology + Send + Sync> = if let Some(m2) = mfem.mesh2d {
        mesh3d = false;
        dim = 2;
        let mut m = m2;
        for _ in 0..args.refine {
            m = refine_uniform(&m);
        }
        Box::new(m)
    } else if let Some(m3) = mfem.mesh3d {
        mesh3d = true;
        dim = 3;
        let mut m = m3;
        for _ in 0..args.refine {
            m = fem_mesh::amr::refine_uniform_3d(&m);
        }
        Box::new(m)
    } else {
        panic!("ex19: mesh must be 2D or 3D (got {})", args.mesh);
    };
    let dim_u8 = dim as u8;
    let order = args.order;
    let p_order = if order > 1 { order - 1 } else { 1 };

    // 2. FE spaces (Taylor-Hood: VectorH1^dim for u, H1 for p)
    let u_space = VectorH1Space::new(mesh.clone_mesh(), order, dim_u8);
    let p_space = H1Space::new(mesh.clone_mesh(), p_order);
    let nu = u_space.n_dofs();
    let np = p_space.n_dofs();
    let ns = u_space.n_scalar_dofs(); // scalar DOFs per component
    // MFEM's byVDIM displacement indexing (see `byvdim_perm`): every assembled
    // vector/matrix and every norm below is expressed in this order.
    let perm = byvdim_perm(ns, dim);
    let residual_byvdim = |ru: &[f64], rp: &[f64]| -> Vec<f64> {
        let mut v = vec![0.0_f64; nu + np];
        for i in 0..nu {
            v[perm[i]] = ru[i];
        }
        v[nu..].copy_from_slice(rp);
        v
    };
    println!("***********************************************************");
    println!("dim(u) = {nu}");
    println!("dim(p) = {np}");
    println!("dim(u+p) = {}", nu + np);
    println!("***********************************************************");

    // 3. Dirichlet BCs (matching MFEM ex19)
    //    Attr 1: fixed (u=0). Attr 2: u_x=0, u_y=0.25*x
    let dm = u_space.scalar_dof_manager();
    let attr1 = boundary_dofs(u_space.mesh(), dm, &[1]);
    let attr2 = boundary_dofs(u_space.mesh(), dm, &[2]);
    let mut du: Vec<(usize, f64)> = Vec::new();
    for &d in &attr1 {
        // All components zero
        for c in 0..dim {
            du.push((d as usize + c * ns, 0.0));
        }
    }
    for &d in &attr2 {
        let x = dm.dof_coord(d as u32)[0]; // x-coordinate
        du.push((d as usize, 0.0));         // u_x = 0
        du.push((d as usize + ns, 0.25 * x)); // u_y = 0.25*x
        if mesh3d {
            du.push((d as usize + 2 * ns, 0.0)); // u_z = 0
        }
    }

    // 4. Initial guess: InitialDeformation = ReferenceConfiguration + shear
    //    u(x) = x_def - x_ref  ->  u_x = 0, u_y = 0.25*x[0]
    let mut u = vec![0.0_f64; nu];
    let mut p = vec![0.0_f64; np];
    for s in 0..ns {
        let xc = dm.dof_coord(s as u32);
        let x = xc[0];
        // Component-major: idx = comp * ns + s
        u[0 * ns + s] = 0.0;         // u_x = 0 (no offset from reference)
        u[1 * ns + s] = 0.25 * x;    // u_y = 0.25*x
    }
    // Apply BC values to the DOF vector (essential BC elimination)
    for &(dof, val) in &du {
        u[dof] = val;
    }

    // 5. Pre-compute element DOF tables
    let ne = mesh.n_elements() as usize;
    let elem_dofs_u: Vec<Vec<usize>> = (0..ne)
        .map(|e| u_space.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
        .collect();
    let elem_dofs_p: Vec<Vec<usize>> = (0..ne)
        .map(|e| p_space.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
        .collect();

    // 6. Build MixedHyperelasticityForm (moves residual/jacobian to library;
    //    the integrator's own rule is 2·order+3, nonlininteg.cpp:642)
    let dim_u = dim;
    let form = MixedHyperelasticityForm::new(
        Box::new(mesh.clone_mesh()),
        dim_u, order, p_order, args.mu, nu, np, ns,
        elem_dofs_u.clone(), elem_dofs_p.clone(), du.clone(),
    );

    // 7. Pressure mass matrix (built once, used in preconditioner).
    //    MFEM BilinearForm(MassIntegrator): rule order = 2*p + OrderW(=0).
    let p_mass = build_pressure_mass(mesh.clone_mesh(), p_order, 2 * p_order, np);

    // 8. Initial residual.  MFEM: `oper->Mult(x, r)` before the Newton loop,
    //    `norm0 = norm = Norm(r)` with `Norm = sqrt(Dot(r, r))`.
    let mut ru = vec![0.0_f64; nu];
    let mut rp = vec![0.0_f64; np];
    form.residual(&u, &p, &mut ru, &mut rp);

    // 9. Newton loop — MFEM `NewtonSolver::Mult` (solvers.cpp): full steps
    //    (`ComputeScalingFactor == 1`, no line search), convergence
    //    `norm <= norm_goal` with `norm_goal = max(rel_tol·norm0, abs_tol)`
    //    fixed once from the initial residual, and the residual recomputed
    //    after every accepted step.
    //
    //    Inner solve: `GMRESSolver` (rtol/atol 1e-12, 300 it, kdim 50,
    //    printlevel -1, iterative_mode = false) with the example's
    //    `JacobianPreconditioner` (block elimination, gamma = 1e-5):
    //      mass:  CG + GSSmoother (rtol/atol 1e-12, 200 it)
    //      stiff: GMRES + GSSmoother (rtol/atol 1e-8, 200 it)
    let s_cfg = SolverConfig {
        rtol: 1e-12,
        atol: 1e-12,
        max_iter: 200,
        verbose: false,
        ..SolverConfig::default()
    };
    let stiff_cfg = MfemGmresConfig {
        kdim: 50,
        max_iter: 200,
        rel_tol: 1e-8,
        abs_tol: 1e-8,
        print_level: 0,
        iterative_mode: false,
    };
    let j_cfg = MfemGmresConfig {
        kdim: 50,
        max_iter: 300,
        rel_tol: 1e-12,
        abs_tol: 1e-12,
        print_level: -1,
        iterative_mode: false,
    };
    let gamma = 1e-5;

    let mut newton_monitor = GeneralResidualMonitor::new("Newton", 1);
    let mut j_monitor = GeneralResidualMonitor::new("  GMRES", 3);

    let mut norm = nr(&residual_byvdim(&ru, &rp));
    let norm_goal = (args.rel_tol * norm).max(args.abs_tol);

    let mut converged = false;
    let mut it = 0usize;
    loop {
        // MFEM `Monitor(it, norm, r, x)` — Reset at it == 0, residual hook,
        // then the monitor's convergence vote ahead of the solver's own test.
        if it == 0 {
            newton_monitor.reset();
        }
        newton_monitor.monitor_residual(it, norm, false);
        if newton_monitor.has_converged() || norm <= norm_goal {
            converged = true;
            break;
        }
        if it >= args.max_iter {
            break;
        }

        // grad = &oper->GetGradient(x); prec->SetOperator(*grad).
        let (_sizes, jac) = form.jacobian_blocks(&u, &p);
        // Re-order into MFEM's byVDIM indexing (same matrix, reference
        // summation order for the preconditioner and Krylov sweeps).
        let jac = permute_block_matrix(&jac, &perm, nu, np);
        let bop = MfemBlockOperator::new(&jac);
        let kuu = jac.get(0, 0).expect("Jacobian block (0,0)");
        let kup = jac.get(0, 1).expect("Jacobian block (0,1)");
        let stiff_prec = GsSmoother::new(kuu, GsType::Symmetric, 1);
        let mp = &p_mass;
        let s_cfg_ref = &s_cfg;
        let jacobian_preconditioner = move |r: &[f64], z: &mut [f64]| {
            apply_jacobian_preconditioner(
                r, z, kuu, kup, mp, &stiff_prec, gamma, nu, np, s_cfg_ref, stiff_cfg,
            );
        };

        // c = J⁻¹ r — left-preconditioned GMRES, `iterative_mode = false`
        // (c starts from zero and b = r, not −r).
        let b = residual_byvdim(&ru, &rp);
        let mut c = vec![0.0_f64; nu + np];
        let _ = mfem_gmres_monitored(
            &bop,
            &b,
            &mut c,
            j_cfg,
            Some(&jacobian_preconditioner),
            &mut j_monitor,
        );

        // MFEM `add(x, -c_scale, c, x)` with `c_scale == 1`.  c[ess] is
        // exactly zero — the eliminated Jacobian rows, the zeroed residual
        // entries and the preconditioner all map ess components to zero, so
        // every Krylov vector vanishes there — hence the essential values
        // survive without re-application, exactly as in C++.
        for i in 0..nu {
            u[i] -= c[perm[i]];
        }
        for i in 0..np {
            p[i] -= c[nu + i];
        }

        form.residual(&u, &p, &mut ru, &mut rp);
        norm = nr(&residual_byvdim(&ru, &rp));
        it += 1;
    }
    // MFEM end-of-`Mult`: `Monitor(final_iter, final_norm, r, x, true)` —
    // the level-1 Newton monitor prints nothing on the final call.
    newton_monitor.monitor_residual(it, norm, true);

    // MFEM: MFEM_VERIFY(newton_solver.GetConverged(), ...)
    assert!(converged, "Newton Solver did not converge.");

    // 10. Save output using fem-io library (matching MFEM SwapNodes+Print + GridFunction::Save)
    // Deformed mesh: displace nodes then write via write_mfem_file
    {
        let dim_u = dim;
        // Rebuild a displaced node list through the trait interface, then
        // downcast to the concrete Mesh<D> for serialisation.
        let nn = mesh.n_nodes() as usize;
        let mut coords_disp: Vec<f64> = Vec::with_capacity(nn * dim_u);
        for n in 0..nn {
            let c = mesh.node_coords(n as u32);
            for d in 0..dim_u {
                let mut v = if d < c.len() { c[d] } else { 0.0 };
                if n < ns {
                    v += u[d * ns + n];
                }
                coords_disp.push(v);
            }
        }
        if mesh3d {
            let mut m = mesh.as_any().downcast_ref::<fem_mesh::Mesh<3>>().expect("3D").clone();
            m.coords = coords_disp;
            fem_io::mfem::write_mfem_file_3d("deformed.mesh", &m)
                .expect("cannot write deformed.mesh");
        } else {
            let mut m = mesh.as_any().downcast_ref::<fem_mesh::Mesh<2>>().expect("2D").clone();
            m.coords = coords_disp;
            write_mfem_file("deformed.mesh", &m)
                .expect("cannot write deformed.mesh");
        }
    }

    // Pressure solution (MFEM: p_gf.Save(pressure_ofs))
    write_mfem_gf_file("pressure.sol", dim as usize, &p, "H1", p_order, 1, 8)
        .expect("cannot write pressure.sol");

    // Deformation (MFEM: x_def = x_gf - x_ref; x_def.Save(deformation_ofs))
    write_mfem_gf_file("deformation.sol", dim as usize, &u, "H1", order, dim as usize, 8)
        .expect("cannot write deformation.sol");
}

struct Args {
    mesh: String,
    refine: usize,
    order: u8,
    mu: f64,
    rel_tol: f64,
    abs_tol: f64,
    max_iter: usize,
    #[allow(dead_code)]
    visualization: bool,
}

impl Args {
    fn parse() -> Self {
        let mut a = Self {
            // MFEM ex19.cpp:186 default mesh = beam-tet (round-135 D1256 closure:
            // the old Rust-only beam-quad default was a 1:1 fidelity deviation).
            mesh: "data/beam-tet.mesh".into(),
            refine: 0,
            order: 2,
            mu: 1.0,
            rel_tol: 1e-4,
            abs_tol: 1e-6,
            max_iter: 500,
            visualization: true,  // MFEM defaults: enabled
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" => a.mesh = it.next().unwrap_or_default(),
                "-r" => a.refine = it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
                "-o" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(2),
                "-mu" => a.mu = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0),
                "-rel" => a.rel_tol = it.next().and_then(|v| v.parse().ok()).unwrap_or(1e-4),
                "-abs" => a.abs_tol = it.next().and_then(|v| v.parse().ok()).unwrap_or(1e-6),
                "-it" => a.max_iter = it.next().and_then(|v| v.parse().ok()).unwrap_or(500),
                "-vis" => a.visualization = true,
                "-no-vis" => a.visualization = false,
                _ => {}
            }
        }
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    #[test]
    fn zero_state_zero_residual() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let dim = 2;
        let order = 1;
        let p_order = 1;
        let mu = 1.0;

        let u_space = VectorH1Space::new(mesh.clone(), order, dim);
        let p_space = H1Space::new(mesh.clone(), p_order);
        let nu = u_space.n_dofs();
        let np = p_space.n_dofs();
        let ns = u_space.n_scalar_dofs();

        // BC: all boundaries fixed (u=0) so zero displacement is equilibrium
        let dm = u_space.scalar_dof_manager();
        let all_bdr = boundary_dofs(u_space.mesh(), dm, &[1, 2, 3, 4]);
        let mut du: Vec<(usize, f64)> = Vec::new();
        for &d in &all_bdr {
            du.push((d as usize, 0.0));
            du.push((d as usize + ns, 0.0));
        }

        // Initial guess: zero
        let mut u = vec![0.0_f64; nu];
        let mut p = vec![0.0_f64; np];
        for &(dof, val) in &du { u[dof] = val; }

        let ne = mesh.n_elements() as usize;
        let elem_dofs_u: Vec<Vec<usize>> = (0..ne)
            .map(|e| u_space.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
            .collect();
        let elem_dofs_p: Vec<Vec<usize>> = (0..ne)
            .map(|e| p_space.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
            .collect();
        let quad_order = 5;

        let mut ru = vec![0.0_f64; nu];
        let mut rp = vec![0.0_f64; np];
        let form = MixedHyperelasticityForm::new(
            Box::new(mesh.clone()), dim as usize, order, p_order, mu,
            nu, np, ns, elem_dofs_u, elem_dofs_p, du,
        );
        form.residual(&u, &p, &mut ru, &mut rp);
        let r0 = nr(&[ru.as_slice(), rp.as_slice()].concat());
        // With all boundaries fixed and zero displacement+pressure, the internal
        // residual should be exactly zero (F=I, J-1=0)
        assert!(r0 < 1e-14, "zero state should have zero residual, got {r0}");
    }

    /// rr148 referee anchor: on the default beam-tet profile MFEM's outer
    /// GMRES opens with `||M r|| = 2.31641` (its left-preconditioned residual
    /// at iteration 0, x = initial guess).  Reproducing that print pins the
    /// whole preconditioner chain bitwise through 6 significant digits:
    /// Jacobian assembly (K_uu/K_up with row elimination), the pressure mass
    /// matrix (MassIntegrator rule order 2·p), mass CG+GSSmoother, the −γ
    /// scaling, K_up·z_p, and the stiffness GMRES+GSSmoother.  Run with CWD =
    /// fem-rs root (data/beam-tet.mesh).
    #[test]
    fn newton0_preconditioned_residual_anchor() {
        // Test CWD = fem-rs/examples (the fem-examples package root).
        let mfem = read_mfem_file("../data/beam-tet.mesh").expect("read ../data/beam-tet.mesh");
        let m3 = mfem.mesh3d.expect("beam-tet is a 3D mesh");
        let order: u8 = 2;
        let p_order: u8 = order - 1;
        let dim = 3_usize;

        let u_space = VectorH1Space::new(m3.clone(), order, dim as u8);
        let p_space = H1Space::new(m3.clone(), p_order);
        let nu = u_space.n_dofs();
        let np = p_space.n_dofs();
        let ns = u_space.n_scalar_dofs();

        let dm = u_space.scalar_dof_manager();
        let attr1 = boundary_dofs(u_space.mesh(), dm, &[1]);
        let attr2 = boundary_dofs(u_space.mesh(), dm, &[2]);
        let mut du: Vec<(usize, f64)> = Vec::new();
        for &d in &attr1 {
            for c in 0..dim {
                du.push((d as usize + c * ns, 0.0));
            }
        }
        for &d in &attr2 {
            let x = dm.dof_coord(d as u32)[0];
            du.push((d as usize, 0.0));
            du.push((d as usize + ns, 0.25 * x));
            du.push((d as usize + 2 * ns, 0.0));
        }

        // Initial guess: u_x = 0, u_y = 0.25·x, p = 0, with BC values set.
        let mut u = vec![0.0_f64; nu];
        let mut p = vec![0.0_f64; np];
        for s in 0..ns {
            u[1 * ns + s] = 0.25 * dm.dof_coord(s as u32)[0];
        }
        for &(dof, val) in &du {
            u[dof] = val;
        }

        let ne = m3.n_elements() as usize;
        let elem_dofs_u: Vec<Vec<usize>> = (0..ne)
            .map(|e| u_space.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
            .collect();
        let elem_dofs_p: Vec<Vec<usize>> = (0..ne)
            .map(|e| p_space.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
            .collect();
        let elem_dofs_u_probe = elem_dofs_u.clone();
        let du_probe = du.clone();
        let mesh_for_mass = m3.clone();
        let mesh_for_kuu = m3.clone();
        let form = MixedHyperelasticityForm::new(
            Box::new(m3),
            dim, order, p_order, 1.0, nu, np, ns,
            elem_dofs_u, elem_dofs_p, du,
        );
        // MFEM MassIntegrator rule: order = 2·p + OrderW(=0, straight tets).
        let p_mass = build_pressure_mass(mesh_for_mass, p_order, 2 * p_order, np);

        let mut ru = vec![0.0_f64; nu];
        let mut rp = vec![0.0_f64; np];
        form.residual(&u, &p, &mut ru, &mut rp);

        let (_sizes, jac) = form.jacobian_blocks(&u, &p);
        let kuu = jac.get(0, 0).expect("J(0,0)");
        let kup = jac.get(0, 1).expect("J(0,1)");
        let stiff_prec = GsSmoother::new(kuu, GsType::Symmetric, 1);
        let s_cfg = SolverConfig {
            rtol: 1e-12,
            atol: 1e-12,
            max_iter: 200,
            verbose: false,
            ..SolverConfig::default()
        };
        let stiff_cfg = MfemGmresConfig {
            kdim: 50,
            max_iter: 200,
            rel_tol: 1e-8,
            abs_tol: 1e-8,
            print_level: 0,
            iterative_mode: false,
        };

        let mut b = vec![0.0_f64; nu + np];
        b[..nu].copy_from_slice(&ru);
        b[nu..].copy_from_slice(&rp);
        let mut z = vec![0.0_f64; nu + np];
        apply_jacobian_preconditioner(
            &b, &mut z, kuu, kup, &p_mass, &stiff_prec, 1e-5, nu, np, &s_cfg, stiff_cfg,
        );

        // Stage-by-stage diagnostics (compare against the C++ stage probe
        // `ex19_stages.cpp`): localize the first diverging preconditioner
        // stage without touching the shared apply function.
        let rp = &b[nu..];
        let mut pres_out = vec![0.0_f64; np];
        let _ = solve_pcg_gssmoother(&p_mass, rp, &mut pres_out, &s_cfg);
        println!(
            "[stage] ||r||        = {:?}",
            nr(&b)
        );
        println!(
            "[stage] ||Mp^-1 rp|| = {:?}  z0={:?}",
            nr(&pres_out),
            pres_out[0]
        );
        for v in pres_out.iter_mut() {
            *v *= -1e-5;
        }
        let mut temp = vec![0.0_f64; nu];
        for i in 0..nu {
            let mut s = 0.0;
            for k in kup.row_ptr[i]..kup.row_ptr[i + 1] {
                s += kup.values[k] * pres_out[kup.col_idx[k] as usize];
            }
            temp[i] = s;
        }
        println!("[stage] ||Kup z_p||  = {:?}", nr(&temp));
        let mut temp2 = vec![0.0_f64; nu];
        for i in 0..nu {
            temp2[i] = b[i] - temp[i];
        }
        println!("[stage] ||temp2||    = {:?}", nr(&temp2));
        let mut disp_out = vec![0.0_f64; nu];
        let _ = mfem_gmres(
            kuu,
            &temp2,
            &mut disp_out,
            stiff_cfg,
            Some(&|rr: &[f64], zz: &mut [f64]| stiff_prec.mult(rr, zz)),
        );
        println!("[stage] ||z_u||      = {:?}", nr(&disp_out));
        println!("[stage] ||M r||      = {:?}", nr(&z));

        // ── rr148 root-cause probe: MFEM-faithful K_uu ────────────────────────
        // MFEM `IncompressibleNeoHookeanIntegrator::AssembleElementGrad`
        // (nonlininteg.cpp:620-691) accumulates THREE terms into (0,0):
        //   T1: dJ·(μ·F − p·F⁻ᵀ)(i,l)·F⁻ᵀ(j,n)·DS_a(l)·DS_b(n)·w
        //   T2: dJ·μ·DS_a(l)·DS_b(n)·w           (i_dim == j_dim && n == l)
        //   T3: dJ·p·F⁻ᵀ(i,n)·F⁻ᵀ(j,l)·DS_a(l)·DS_b(n)·w
        // fem-assembly's `MixedHyperelasticityForm::jacobian_blocks` implements
        // only T2 and T3 — **T1 is missing entirely**, and the (0,1) block
        // additionally lacks the dJ factor (nonlininteg.cpp:684 assembles
        // (0,1) −= dJ·F⁻ᵀ·DS·Sh·w, the form writes −w·F⁻ᵀ·DS·Sh).  At the
        // Newton-0 state p = 0 ⇒ T3 vanishes and only T1+T2 remain.
        //
        // This probe assembles the MFEM-faithful K_uu locally (no
        // fem-assembly change, which is round-148 lane 3's crate) and asserts
        // the preconditioner anchor: ‖K_uu⁻¹ r_u‖ then opens MFEM's outer
        // GMRES at 2.31641 — the value the C++ oracle prints.
        {
            use fem_mesh::xform_grads;
            let mu = 1.0_f64;
            let et = mesh_for_kuu.element_type(0);
            let ref_u = et.ref_elem(order);
            let n_du = ref_u.n_dofs();
            let vd = n_du * dim;
            let q = ref_u.quadrature(2 * order + 3); // nonlininteg.cpp:642
            let mut coo = CooMatrix::<f64>::new(nu, nu);
            let mut gr_u = vec![0.0_f64; n_du * dim];
            let mut gp_u = vec![0.0_f64; n_du * dim];
            for e in 0..mesh_for_kuu.n_elements() as usize {
                let eu = &elem_dofs_u_probe[e];
                let mut ue = vec![0.0_f64; vd];
                for (k, &g) in eu.iter().enumerate() {
                    ue[k] = u[g];
                }
                let mut ke = vec![0.0_f64; vd * vd];
                for (qi, xi) in q.points.iter().enumerate() {
                    ref_u.eval_grad_basis(xi, &mut gr_u);
                    let (det_j, ji) = geometry_jacobian(&mesh_for_kuu, e as u32, xi, dim);
                    xform_grads(&ji, &gr_u, &mut gp_u, n_du, dim);
                    let mut f = nalgebra::DMatrix::<f64>::identity(dim, dim);
                    for k in 0..n_du {
                        for i in 0..dim {
                            for j in 0..dim {
                                f[(i, j)] += ue[k * dim + i] * gp_u[k * dim + j];
                            }
                        }
                    }
                    let dj = f.determinant();
                    let ft = f
                        .clone()
                        .try_inverse()
                        .expect("F invertible")
                        .transpose();
                    let w = q.weights[qi] * det_j;
                    for a in 0..n_du {
                        for i in 0..dim {
                            for b in 0..n_du {
                                for j in 0..dim {
                                    let mut v = 0.0_f64;
                                    for n in 0..dim {
                                        for l in 0..dim {
                                            // T1 (pres = 0 at Newton 0): μ only.
                                            v += dj
                                                * (mu * f[(i, l)])
                                                * ft[(j, n)]
                                                * gp_u[a * dim + l]
                                                * gp_u[b * dim + n];
                                            if j == i && n == l {
                                                // T2.
                                                v += dj
                                                    * mu
                                                    * gp_u[a * dim + l]
                                                    * gp_u[b * dim + n];
                                            }
                                            // T3: pres = 0 at Newton 0.
                                        }
                                    }
                                    ke[(a * dim + i) * vd + (b * dim + j)] += v * w;
                                }
                            }
                        }
                    }
                }
                for a in 0..vd {
                    for bb in 0..vd {
                        coo.add(eu[a], eu[bb], ke[a * vd + bb]);
                    }
                }
            }
            let mut kuu_mfem = coo.into_csr();
            // Essential rows → identity (same row-zeroing convention the form
            // applies; the ess columns are immaterial — every solution/Krylov
            // vector vanishes there because r_u[ess] = 0).
            let mut scratch = vec![0.0_f64; nu];
            for &(dof, _) in &du_probe {
                kuu_mfem.apply_dirichlet_row_zeroing(dof, 0.0, &mut scratch);
            }
            let stiff_prec_mfem = GsSmoother::new(&kuu_mfem, GsType::Symmetric, 1);
            let mut disp_mfem = vec![0.0_f64; nu];
            let _ = mfem_gmres(
                &kuu_mfem,
                &temp2,
                &mut disp_mfem,
                stiff_cfg,
                Some(&|rr: &[f64], zz: &mut [f64]| stiff_prec_mfem.mult(rr, zz)),
            );
            println!("[mfem-Kuu] ||z_u||    = {:?}", nr(&disp_mfem));
            println!("[mfem-Kuu] ||M r||    = {:?}", nr(&disp_mfem));

            // THE anchor: MFEM's outer GMRES opens at ||M r|| = 2.31641 on
            // this matrix.  With the current fem-assembly K_uu (T1 missing)
            // the same solve yields 6.50731 (printed above as
            // `[stage] ||M r||`) — the rr127 "solver structural debt" is this
            // missing Jacobian term, not the solvers.
            assert_eq!(
                fmt_g(nr(&disp_mfem)),
                "2.31641",
                "Newton-0 ||M r|| anchor with MFEM-faithful K_uu"
            );

            // ── (4) dof-layout pin (rr157 lane2) ─────────────────────────────
            // MFEM ex19 numbers the displacement space `Ordering::byVDIM`.
            // Re-ordering the assembled system into that indexing puts every
            // Gauss-Seidel sweep / Krylov dot product in MFEM's summation order
            // and closes the `‖z_u‖` gap from 5e-11 (blocked) to ~1 ulp: the
            // oracle's full-precision value is `2.3164065129917213`.
            {
                let perm = byvdim_perm(ns, dim);
                let ident: Vec<usize> = (0..np).collect();
                let kuu_p = permute_csr(kuu, &perm, &perm, nu, nu);
                let kup_p = permute_csr(kup, &perm, &ident, nu, np);
                let mut b_p = vec![0.0_f64; nu + np];
                for i in 0..nu {
                    b_p[perm[i]] = b[i];
                }
                b_p[nu..].copy_from_slice(&b[nu..]);
                let stiff_p = GsSmoother::new(&kuu_p, GsType::Symmetric, 1);
                let mut z_p = vec![0.0_f64; nu + np];
                apply_jacobian_preconditioner(
                    &b_p, &mut z_p, &kuu_p, &kup_p, &p_mass, &stiff_p, 1e-5, nu, np, &s_cfg,
                    stiff_cfg,
                );
                let zu = nr(&z_p[..nu]);
                println!("[byvdim] ||z_u||    = {zu:?}  (MFEM 2.3164065129917213)");
                assert!(
                    (zu - 2.3164065129917213).abs() < 1e-13,
                    "byVDIM Newton-0 ||z_u|| = {zu:?}, MFEM = 2.3164065129917213"
                );
            }
        }
    }
}
