//! # Example 31 — Anisotropic Maxwell (1:1 with MFEM ex31)
//!
//! Solves ∇×(∇×E) + Σ·E = f with a full 3×3 anisotropic tensor Σ and
//! PEC BC on a 1‑D, 2‑D or 3‑D mesh.  Uses `HCurlSpace` (in‑plane) +
//! `H1Space` (z‑component) to match MFEM's `ND_R2D_FECollection`.
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex31_anisotropic_maxwell -- -m data/beam-tri.mesh -o 1 -r 1
//! ```

#![allow(unused_variables, unused_mut)]
use std::f64::consts::{PI, SQRT_2};

use fem_assembly::standard::{CurlCurlIntegrator, DiffusionIntegrator, MassIntegrator,
    VectorMassTensorIntegrator};
use fem_assembly::coefficient::ConstantMatrixCoeff;
use fem_assembly::{VectorAssembler, Assembler, FixedOrder};
use fem_assembly::postproc::grid_function::project_bdr_coefficient_tangent_2d;
use fem_element::{VectorReferenceElement, ReferenceElement,
    nedelec::{TriND1, QuadND1}, lagrange::{TriP1, QuadQ1}};
use fem_io::mfem::read_mfem_file;
use fem_linalg::CooMatrix;
use fem_mesh::{ElementType, Mesh, MeshTopology, amr::refine_uniform};
use fem_solver::SolverConfig;
use fem_space::{HCurlSpace, H1Space,
    fe_space::FESpace, constraints::{boundary_dofs_hcurl, boundary_dofs}};

// ─── MFEM ex31 coefficients (2‑D case) ──────────────────────────────

const A0: f64 = 1.1; const A1: f64 = 1.2; const A2: f64 = 1.3;
const PHI1: f64 = 0.4 * PI; const PHI2: f64 = 0.9 * PI;

/// Σ = [[2, 1/√2, 0], [1/√2, 2, 1/√2], [0, 1/√2, 2]]
const SXX: f64 = 2.0; const SXY: f64 = 1.0 / SQRT_2;
const SYY: f64 = 2.0; const SYZ: f64 = 1.0 / SQRT_2; const SZZ: f64 = 2.0;

// ─── CLI ────────────────────────────────────────────────────────────

struct Args { mesh: Option<String>, n: usize, ref_levels: usize, order: u8, freq: f64 }
fn parse_args() -> Args {
    let mut a = Args { mesh: None, n: 16, ref_levels: 2, order: 1, freq: 1.0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next(),
            "--n" => a.n = it.next().and_then(|v| v.parse().ok()).unwrap_or(16),
            "-r" | "--refine" => a.ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(2),
            "-o" | "--order" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-f" | "--frequency" => a.freq = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            _ => {}
        }
    }
    a
}

// ─── Exact solution ─────────────────────────────────────────────────

fn exact_e(x: &[f64], kappa: f64) -> [f64; 3] {
    let u = (kappa / SQRT_2) * (x[0] + x[1]);
    [A0 * u.sin(), A1 * (u + PHI1).sin(), A2 * (u + PHI2).sin()]
}

fn exact_curl(x: &[f64], kappa: f64) -> [f64; 3] {
    let u = (kappa / SQRT_2) * (x[0] + x[1]);
    let (c0, c4, c9) = (u.cos(), (u + PHI1).cos(), (u + PHI2).cos());
    let a = kappa / SQRT_2;
    [A2 * c9 * a, -A2 * c9 * a, A1 * c4 * a - A0 * c0 * a]
}

/// Matches C++ ex31 f_exact for dim==2.
fn source_3d(x: &[f64], kappa: f64) -> [f64; 3] {
    let k2 = kappa * kappa;
    let u = (kappa / SQRT_2) * (x[0] + x[1]);
    let (s0, s4, s9) = (u.sin(), (u + PHI1).sin(), (u + PHI2).sin());
    let f0 = 0.55 * (4.0 + k2) * s0 + 0.6 * (SQRT_2 - k2) * s4;
    let f1 = 0.55 * (SQRT_2 - k2) * s0 + 0.6 * (4.0 + k2) * s4 + 0.65 * SQRT_2 * s9;
    let f2 = 0.6 * SQRT_2 * s4 + 1.3 * (2.0 + k2) * s9;
    [f0, f1, f2]
}

// ─── Element reference dispatcher ───────────────────────────────────

type JacobianFn = fn(
    &Mesh<2>, u32, &[u32], &[f64],
) -> (f64, f64, f64, f64, f64, f64); // (inv_det, jit00, jit01, jit10, jit11, det_j)

fn affine_jac(mesh: &Mesh<2>, _e: u32, nodes: &[u32], _xi: &[f64]) -> (f64, f64, f64, f64, f64, f64) {
    let x0 = mesh.node_coords(nodes[0]);
    let x1 = mesh.node_coords(nodes[1]);
    let x2 = mesh.node_coords(nodes[2]);
    let (j00, j01) = (x1[0] - x0[0], x2[0] - x0[0]);
    let (j10, j11) = (x1[1] - x0[1], x2[1] - x0[1]);
    let det = j00 * j11 - j01 * j10;
    let inv = 1.0 / det;
    (inv, j11 * inv, -j10 * inv, -j01 * inv, j00 * inv, det.abs())
}

fn isoparametric_jac(mesh: &Mesh<2>, _e: u32, nodes: &[u32], xi: &[f64]) -> (f64, f64, f64, f64, f64, f64) {
    let geo = QuadQ1;
    let n_geo = geo.n_dofs();
    let mut grad = vec![0.0_f64; n_geo * 2];
    geo.eval_grad_basis(xi, &mut grad);
    let mut j = nalgebra::DMatrix::<f64>::zeros(2, 2);
    let mut xp = vec![0.0_f64; 2];
    for k in 0..n_geo {
        let xk = mesh.node_coords(nodes[k]);
        for i in 0..2 { xp[i] += (1.0 / n_geo as f64) * xk[i]; } // placeholder
        for i in 0..2 { for d in 0..2 { j[(i, d)] += xk[i] * grad[k * 2 + d]; } }
    }
    let det = j.determinant();
    let inv = 1.0 / det;
    (inv, j[(1,1)] * inv, -j[(1,0)] * inv, -j[(0,1)] * inv, j[(0,0)] * inv, det.abs())
}

fn setup_element_ref(et: ElementType, _order: u8) -> (usize, &'static dyn VectorReferenceElement, &'static dyn ReferenceElement, usize, JacobianFn) {
    match et {
        ElementType::Tri3 => (3, &TriND1 as &dyn VectorReferenceElement, &TriP1 as &dyn ReferenceElement, 3, affine_jac as JacobianFn),
        ElementType::Quad4 => (4, &QuadND1 as &dyn VectorReferenceElement, &QuadQ1 as &dyn ReferenceElement, 4, isoparametric_jac as JacobianFn),
        _ => panic!("unsupported element type {et:?}"),
    }
}

// ─── Main ───────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    let kappa = args.freq * PI;

    println!("Options used:");
    match &args.mesh {
        Some(p) => println!("   --mesh {p}"),
        None => println!("   --mesh (built-in {0}x{0} tri)", args.n),
    }
    println!("   --refine {}", args.ref_levels);
    println!("   --order {}", args.order);
    println!("   --frequency {}", args.freq);
    println!();

    // 1. Build mesh.
    let base_mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_tri(args.n)
    };
    let mesh = if args.ref_levels > 0 {
        let mut m = base_mesh;
        for _ in 0..args.ref_levels { m = refine_uniform(&m); }
        m
    } else { base_mesh };

    // 2. Spaces.
    let quad_order = args.order * 2 + 2;
    let nd_space = HCurlSpace::new(mesh.clone(), args.order);
    let z_space = H1Space::new(mesh.clone(), args.order);
    let n_nd = nd_space.n_dofs();
    let n_h1 = z_space.n_dofs();
    let n_total = n_nd + n_h1;
    println!("DOFs: H(Curl)={n_nd}  H¹(z)={n_h1}  total={n_total}");

    // 3. In‑plane block: curlcurl + Σ_xy.
    let curl_curl = CurlCurlIntegrator { mu: 1.0 };
    let sigma_2d = ConstantMatrixCoeff(vec![SXX, SXY, SXY, SYY]);
    let vec_mass = VectorMassTensorIntegrator { alpha: sigma_2d };
    let a_nd = VectorAssembler::assemble_bilinear(
        &nd_space, &[&curl_curl, &vec_mass], quad_order,
    );

    // 4. Z‑component block: -∇² + Σ_zz.
    //    MFEM's CurlCurlIntegrator on the ND_R2D (Pk) space uses integration
    //    order 2p-2 = 0 (single-point rule) for the curl contribution to the
    //    z-component; replicate that with FixedOrder so the z-block matches
    //    bit-for-bit (ex29/ex30 Araw style).
    let laplace = FixedOrder::new(DiffusionIntegrator { kappa: 1.0 }, 0);
    let z_mass = MassIntegrator { rho: SZZ };
    let a_z = Assembler::assemble_bilinear(&z_space, &[&laplace, &z_mass], quad_order);

    // 5. Coupling block: Σ_yz · ∫ E_y · E_z dx.
    let mut coupling_coo = CooMatrix::<f64>::new(n_nd, n_h1);
    for e in 0..mesh.n_elements() as u32 {
        let nd_dofs: Vec<usize> = nd_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let h1_dofs: Vec<usize> = z_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = nd_space.mesh().element_nodes(e);
        let signs = nd_space.element_signs(e);
        let (n_ld, rnd, rh1, n_lh1, jac_fn) = setup_element_ref(mesh.element_type(e), args.order);
        let q = rnd.quadrature(quad_order);
        let mut np = vec![0.0; n_ld * 2];
        let mut hp = vec![0.0; n_lh1];
        let mut em = vec![0.0_f64; n_ld * n_lh1];
        for (qi, xi) in q.points.iter().enumerate() {
            let (_, jit00, jit01, jit10, jit11, det) = jac_fn(&mesh, e, nodes, xi);
            let w = q.weights[qi] * det * SYZ;
            rnd.eval_basis_vec(xi, &mut np);
            rh1.eval_basis(xi, &mut hp);
            for i in 0..n_ld {
                let py = signs[i] * (jit10 * np[i * 2] + jit11 * np[i * 2 + 1]);
                if py.abs() < 1e-15 { continue; }
                for j in 0..n_lh1 { em[i * n_lh1 + j] += w * py * hp[j]; }
            }
        }
        for (li, &ri) in nd_dofs.iter().enumerate() {
            for (lj, &cj) in h1_dofs.iter().enumerate() {
                let v = em[li * n_lh1 + lj];
                if v != 0.0 { coupling_coo.add(ri, cj, v); }
            }
        }
    }
    let coupling = coupling_coo.into_csr();

    // 6. Combine blocks.
    let mut sys_coo = CooMatrix::<f64>::new(n_total, n_total);
    for r in 0..n_nd { for k in a_nd.row_ptr[r]..a_nd.row_ptr[r+1] { sys_coo.add(r, a_nd.col_idx[k] as usize, a_nd.values[k]); } }
    for r in 0..n_h1 { let rr = n_nd + r; for k in a_z.row_ptr[r]..a_z.row_ptr[r+1] { sys_coo.add(rr, n_nd + a_z.col_idx[k] as usize, a_z.values[k]); } }
    for r in 0..coupling.nrows {
        for k in coupling.row_ptr[r]..coupling.row_ptr[r+1] {
            let c = coupling.col_idx[k] as usize; let v = coupling.values[k];
            if v != 0.0 { sys_coo.add(r, n_nd + c, v); sys_coo.add(n_nd + c, r, v); }
        }
    }
    let sys_mat = sys_coo.into_csr();

    // 7. RHS.  MFEM's VectorFEDomainLFIntegrator uses IntRules order
    //    2·el.GetOrder() = 2 for the (non-polynomial) source; replicate.
    let src_nd = FixedOrder::new(FnVectorSource(Box::new(move |x| { let f = source_3d(x, kappa); [f[0], f[1]] })), 2);
    let rhs_nd = VectorAssembler::assemble_linear(&nd_space, &[&src_nd], quad_order);
    let src_z = FixedOrder::new(FnScalarSource(Box::new(move |x| source_3d(x, kappa)[2])), 2);
    let rhs_z = Assembler::assemble_linear(&z_space, &[&src_z], quad_order);
    let mut rhs = vec![0.0_f64; n_total];
    for i in 0..n_nd { rhs[i] = rhs_nd[i]; }
    for i in 0..n_h1 { rhs[n_nd + i] = rhs_z[i]; }

    // 8. BC (non-homogeneous Dirichlet with exact solution).
    let nd_bdr = boundary_dofs_hcurl(&mesh, &nd_space, &mesh.unique_boundary_tags());
    let h1_bdr = boundary_dofs(&mesh, z_space.dof_manager(), &mesh.unique_boundary_tags());
    eprintln!("  BC DOFs: H(Curl)={}  H¹(z)={}", nd_bdr.len(), h1_bdr.len());
    let mut x = vec![0.0_f64; n_total];
    project_bdr_coefficient_tangent_2d(&mut x[..n_nd], &nd_space,
        &|x: &[f64], out: &mut [f64]| { let e = exact_e(x, kappa); out[0] = e[0]; out[1] = e[1]; },
        &mesh.unique_boundary_tags());
    for &d in &h1_bdr { let c = z_space.dof_manager().dof_coord(d); x[n_nd + d as usize] = exact_e(c, kappa)[2]; }
    let mut mat = sys_mat;
    for &d in &nd_bdr { mat.apply_dirichlet_keep_diag(d as usize, x[d as usize], &mut rhs); }
    for &d in &h1_bdr { mat.apply_dirichlet_keep_diag(n_nd + d as usize, x[n_nd + d as usize], &mut rhs); }

    // 9. Solve.  MFEM: PCG(*A, M, B, X, 1, 500, 1e-12, 0.0) with GSSmoother —
    // the free-function wrapper calls CGSolver::SetRelTol(sqrt(1e-12)) = 1e-6.
    // solve_pcg_gssmoother is the bit-for-bit MFEM CGSolver+GSSmoother port
    // (fwd+back GS sweeps); linlvo's GSSmoother is not bit-identical.
    let cfg = SolverConfig {
        rtol: 1e-6,
        max_iter: 500,
        verbose: true,
        ..Default::default()
    };
    fem_solver::solve_pcg_gssmoother(&mat, &rhs, &mut x, &cfg).expect("PCG");

    // 10. H(Curl) error.
    let mut err2 = 0.0_f64;
    for e in 0..mesh.n_elements() as u32 {
        let nd_dofs: Vec<usize> = nd_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let h1_dofs: Vec<usize> = z_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);
        let signs = nd_space.element_signs(e);
        let (n_ld, rnd, rh1, n_lh1, jac_fn) = setup_element_ref(mesh.element_type(e), args.order);
        let qord = 2 * args.order as u8 + 3;
        let q = rnd.quadrature(qord);
        let mut pn = vec![0.0; n_ld * 2];
        let mut ph = vec![0.0; n_lh1];
        let mut cn = vec![0.0; n_ld];
        for (qi, xi) in q.points.iter().enumerate() {
            let (inv_det, jit00, jit01, jit10, jit11, det) = jac_fn(&mesh, e, nodes, xi);
            let w = q.weights[qi] * det;
            let xp = if mesh.element_type(e) == ElementType::Quad4 {
                // Isoparametric: compute through QuadQ1 geometry
                let geo = QuadQ1; let ng = geo.n_dofs(); let mut phi = vec![0.0; ng];
                geo.eval_basis(xi, &mut phi);
                let mut p = [0.0_f64; 2];
                for k in 0..ng { let c = mesh.node_coords(nodes[k]); p[0] += phi[k] * c[0]; p[1] += phi[k] * c[1]; }
                p
            } else {
                let x0 = mesh.node_coords(nodes[0]);
                let x1 = mesh.node_coords(nodes[1]);
                let x2 = mesh.node_coords(nodes[2]);
                [x0[0] + (x1[0]-x0[0])*xi[0] + (x2[0]-x0[0])*xi[1],
                 x0[1] + (x1[1]-x0[1])*xi[0] + (x2[1]-x0[1])*xi[1]]
            };
            rnd.eval_basis_vec(xi, &mut pn);
            rh1.eval_basis(xi, &mut ph);
            rnd.eval_curl(xi, &mut cn);
            let mut eh = [0.0_f64; 3];
            for i in 0..n_ld {
                let s = signs[i];
                // MFEM CalcVShape_ND (fe_base.cpp): shape = vshape_ref · J⁻¹
                // (row vector right-multiplied), so
                //   φx = J⁻¹₀₀·φx + J⁻¹₁₀·φy = jit00·φx + jit01·φy
                //   φy = J⁻¹₀₁·φx + J⁻¹₁₁·φy = jit10·φx + jit11·φy
                // (jit00=j11/det, jit01=−j10/det=J⁻¹₁₀, jit10=−j01/det=J⁻¹₀₁,
                //  jit11=j00/det).
                eh[0] += s * x[nd_dofs[i]] * (jit00 * pn[i * 2] + jit01 * pn[i * 2 + 1]);
                eh[1] += s * x[nd_dofs[i]] * (jit10 * pn[i * 2] + jit11 * pn[i * 2 + 1]);
            }
            for j in 0..n_lh1 { eh[2] += x[n_nd + h1_dofs[j]] * ph[j]; }
            let mut ce = [0.0_f64; 3];
            for i in 0..n_ld { ce[2] += signs[i] * x[nd_dofs[i]] * cn[i]; }
            ce[2] *= inv_det;
            let mut gr = vec![0.0_f64; n_lh1 * 2];
            rh1.eval_grad_basis(xi, &mut gr);
            for j in 0..n_lh1 {
                // ∇z_phys = J⁻ᵀ·∇z_ref:  ∂z/∂x = jit00·gx + jit10·gy,
                // ∂z/∂y = jit01·gx + jit11·gy;  curl_x = ∂Ez/∂y, curl_y = −∂Ez/∂x.
                let dx = jit00 * gr[j * 2] + jit10 * gr[j * 2 + 1];
                let dy = jit01 * gr[j * 2] + jit11 * gr[j * 2 + 1];
                ce[0] += x[n_nd + h1_dofs[j]] * dy;
                ce[1] -= x[n_nd + h1_dofs[j]] * dx;
            }
            let (ee, ec) = (exact_e(&xp, kappa), exact_curl(&xp, kappa));
            for c in 0..3 { let d = eh[c] - ee[c]; err2 += w * d * d; let dc = ce[c] - ec[c]; err2 += w * dc * dc; }
        }
    }
    let hcurl_err = err2.sqrt();
    println!("\n|| E_h - E ||_{{H(Curl)}} = {hcurl_err:.10e}");
}

// ─── Helper integrators ─────────────────────────────────────────────

use fem_assembly::vector_integrator::{VectorLinearIntegrator, VectorQpData};
struct FnVectorSource(Box<dyn Fn(&[f64]) -> [f64; 2] + Send + Sync>);
impl VectorLinearIntegrator for FnVectorSource {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, fe: &mut [f64]) {
        let f = (self.0)(qp.x_phys);
        for i in 0..qp.n_dofs { fe[i] += qp.weight * (qp.phi_vec[i*2]*f[0] + qp.phi_vec[i*2+1]*f[1]); }
    }
}
use fem_assembly::integrator::{LinearIntegrator, QpData};
struct FnScalarSource(Box<dyn Fn(&[f64]) -> f64 + Send + Sync>);
impl LinearIntegrator for FnScalarSource {
    fn add_to_element_vector(&self, qp: &QpData<'_>, fe: &mut [f64]) {
        let f = (self.0)(qp.x_phys);
        for i in 0..qp.n_dofs { fe[i] += qp.weight * qp.phi[i] * f; }
    }
}
