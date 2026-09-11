//! # Example 3 — Maxwell Electromagnetic Diffusion  (1:1 with MFEM ex3)
//!
//! Solves ∇×(∇×E) + E = f with non-homogeneous Dirichlet BC.
//! Supports both 2D and 3D (matching MFEM ex3.cpp `dim` dispatch).
//!
//! Default: data/beam-tet.mesh (3D, 32016 unknowns). Use `-m data/star.mesh` for 2D.
//!
//! # Canonical (shared-face) DOF basis — D48/D58
//!
//! The **default assembly entry points**
//! ([`VectorAssembler::assemble_bilinear`] / `assemble_linear`) apply the
//! canonical shared-face rotation themselves (`A ← Sᵀ·A·S`, `b ← Sᵀ·b`) and
//! the solution is rotated back per element (`u_local = T·u_canon`,
//! [`fem_assembly::vector_assembler::nd_element_local_dofs`]) exactly where
//! MFEM calls `DofTransformation::InvTransformPrimal`
//! (`GridFunction::GetVectorValues`).  This is a no-op for spaces without
//! shared-face DOF pairs (2-D, hex, ND1) and is what makes the tet `NDk`
//! (k ≥ 2) discretisation conforming.  `-no-canonical` selects the historical
//! element-local pair (non-conforming, kept for the A/B comparison) built
//! from the explicit per-element accumulators.
//!
//! # Hex NDk quadrature + solver parity — D55
//!
//! MFEM's `ND_HexahedronElement` is a **tensor (Qk)** element, so
//! `CurlCurlIntegrator` assembles it with the `2k` rule (the `2k-2` Pk rule
//! applies to simplices only), and `ND_HexahedronElement` gets **no**
//! `DofTransformation` at all (`DofTransformationForGeometry` returns NULL
//! for tensor-product geometries): hex quad faces are pure signed
//! permutations of the shared functionals, which is exactly what
//! `HCurlSpace`'s `match_face_dof` already encodes.  fem-rs's
//! `CurlCurlIntegrator::integration_order` reports the Pk rule `2k-2`
//! unconditionally, under-integrating the hex curl-curl block; this example
//! assembles that integrator itself with the geometry-correct `2k` rule (see
//! [`assemble_mat_mfem_rule`]).  With it, the eliminated beam-hex ND2 systems
//! match a C++ harness entry-invariant to ~1e-13 and the exact (dense)
//! Galerkin L2 agrees to 14 digits (`crates/assembly/tests/`
//! `d55_hex_nd2_system_regression.rs`).  On the *refined* beam the printed
//! `|| E_h - E ||` is PCG stopping-sensitive (MFEM's own 500-iteration run
//! prints 1.50666e-4 vs its 291-iteration 2.5343e-4); fem-rs's converged
//! 1.50664e-4 matches the former to 1.6e-5 relative.
//!
//! # 2-D solver parity — D57
//!
//! Two 2-D-path defects were fixed in round 18:
//!
//! * [`l2_err_2d`] applied `J^{-1}` instead of `J^{-T}` in the covariant
//!   Piola reconstruction (invisible on right triangles with diagonal
//!   Jacobians, wrong on sheared ones);
//! * `solve_report_2d` re-assembled a *fresh, un-eliminated* matrix and
//!   solved it against the *eliminated* right-hand side — a different linear
//!   system whose solution violates the essential boundary condition.
//!   `solve_report_2d` now receives the eliminated matrix, exactly like the
//!   3-D `solve_report`.  Evidence: `crates/assembly/tests/`
//!   `d57_ex3_2d_regression.rs` (2-triangle square, refinement 3, every
//!   pipeline stage matches the C++ `probe2d` harness to <=1e-9) and
//!   `beam-tri` o1 solving to 8.01477893043346e-2 vs the C++
//!   0.08014778969562512.

use std::f64::consts::PI;

use fem_assembly::{
    VectorAssembler, DiscreteLinearOperator,
    vector_assembler::{
        accumulate_vector_bilinear_element_blocks, accumulate_vector_linear_element,
        nd_element_local_dofs, nd_element_local_dofs_signed,
    },
    vector_integrator::{VectorLinearIntegrator, VectorQpData},
    standard::{CurlCurlIntegrator, VectorMassIntegrator},
};
use fem_element::VectorReferenceElement;
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_file_3d, write_mfem_gf_file};
use fem_linalg::CooMatrix;
use fem_mesh::{Mesh, MeshTopology, amr::refine_uniform};
use fem_solver::{solve_pcg, SolverConfig};
use fem_space::{HCurlSpace, fe_space::FESpace, constraints::{boundary_dofs_hcurl, form_linear_system}};

fn main() {
    let args = parse_args();
    println!("Options used:");
    println!("   --mesh {}", args.mesh.as_deref().unwrap_or("(built-in beam-tet)"));
    println!("   --order {}", args.order);
    println!("   --frequency {}", args.freq);
    println!("   --no-static-condensation");
    println!("   --no-partial-assembly");
    println!("   --conforming");
    println!("   --device cpu");
    println!("   --no-visualization");
    println!("Device configuration: cpu");
    println!("Memory configuration: host-std");

    let mfem = if let Some(ref path) = args.mesh {
        read_mfem_file(path).expect("failed to read MFEM mesh")
    } else {
        read_mfem_file("data/beam-tet.mesh").expect("failed to read data/beam-tet.mesh")
    };

    if let Some(mesh3d) = mfem.mesh3d {
        solve_3d(&args, mesh3d);
    } else if let Some(mesh2d) = mfem.mesh2d {
        solve_2d(&args, mesh2d);
    } else {
        panic!("MFEM file must contain a 2D or 3D mesh");
    }
}

// ─── D48/D58 assembly/reconstruction mode switch ────────────────────────────

/// Load-vector counterpart of [`assemble_mat_mfem_rule`].
fn assemble_linear<M: MeshTopology>(
    canonical: bool,
    sp: &HCurlSpace<M>,
    ints: &[&dyn VectorLinearIntegrator],
    qo: u8,
) -> Vec<f64> {
    if canonical {
        return VectorAssembler::assemble_linear(sp, ints, qo);
    }
    let n = sp.n_dofs();
    let space_order = sp.element_order(0);
    if ints.iter().any(|i| i.integration_order(space_order).is_some()) {
        let mut acc = vec![0.0_f64; n];
        for integ in ints {
            let qo_i = integ.integration_order(space_order).unwrap_or(qo);
            let v = assemble_linear_element_local(sp, std::slice::from_ref(integ), qo_i);
            for i in 0..n { acc[i] += v[i]; }
        }
        return acc;
    }
    assemble_linear_element_local(sp, ints, qo)
}

/// Element-local (non-canonical) load accumulation of one integrator list.
fn assemble_linear_element_local<M: MeshTopology>(
    sp: &HCurlSpace<M>,
    ints: &[&dyn VectorLinearIntegrator],
    qo: u8,
) -> Vec<f64> {
    let mut rhs = vec![0.0_f64; sp.n_dofs()];
    for e in 0..sp.mesh().n_elements() as u32 {
        accumulate_vector_linear_element(sp, e, ints, qo, &mut rhs);
    }
    rhs
}

/// Element-local DOF values for the selected mode: `u_local = T·u_canon` on the
/// canonical basis (MFEM `DofTransformation::InvTransformPrimal`), or the
/// scalar-sign gather that matches the element-local assembly.
fn element_local_dofs<M: MeshTopology>(
    canonical: bool,
    sp: &HCurlSpace<M>,
    e: u32,
    x: &[f64],
) -> Vec<f64> {
    if canonical {
        nd_element_local_dofs(sp, e, x)
    } else {
        nd_element_local_dofs_signed(sp, e, x)
    }
}

/// `curl curl + mass` assembled with MFEM's **per-geometry** integration rules
/// (D55).
///
/// MFEM `CurlCurlIntegrator::AssembleElementMatrix` (fem/bilininteg.cpp) picks
/// `order = 2*el.GetOrder()` for Qk (tensor) elements and
/// `2*el.GetOrder() - 2` for Pk simplices; `VectorFEMassIntegrator` picks
/// `Trans.OrderW() + 2*el.GetOrder()` (= `2k` on the affine meshes ex3 runs).
/// fem-rs's `CurlCurlIntegrator::integration_order` currently reports the Pk
/// rule `2k-2` for *every* geometry and the assemble entry points re-dispatch
/// through it, which **under-integrates** the hex (Qk) NDk curl-curl block:
/// at k=2 it assembles with 2 Gauss points per direction where the integrand
/// needs degree 4 (3 points).  That is D55's real defect — on beam-hex the
/// solve gave 1.12e-1 vs the C++ 1.15e-1 unrefined and 1.51e-4 vs 2.53e-4
/// refined.  Until the core dispatch is geometry-aware, ex3 assembles the two
/// integrators itself with MFEM's rules:
///
/// * curl-curl at `2k` — MFEM's Qk rule exactly; on Pk simplices it merely
///   over-integrates an integrand MFEM integrates exactly, so the assembled
///   matrix is the same and the beam-tet baseline is preserved;
/// * mass at `2k+3` — [`VectorMassIntegrator`]'s own (exact, bit-matching)
///   choice, unchanged from the dispatched path.
///
/// Both branches keep the D48 face-block rotation (canonical basis) through
/// [`VectorAssembler::accumulate_vector_bilinear_element_blocks`].
fn assemble_mat_mfem_rule<M: MeshTopology>(
    canonical: bool,
    sp: &HCurlSpace<M>,
    mu: f64,
    sigma: f64,
    order: u8,
) -> fem_linalg::CsrMatrix<f64> {
    let n = sp.n_dofs();
    let mut coo = CooMatrix::<f64>::new(n, n);
    let cc = CurlCurlIntegrator { mu };
    let mass = VectorMassIntegrator { alpha: sigma };
    let qo_cc = 2 * order;
    let qo_mass = 2 * order + 3;
    let no_blocks: [fem_space::hcurl::FaceDofBlock; 0] = [];
    for e in 0..sp.mesh().n_elements() as u32 {
        let blocks: &[fem_space::hcurl::FaceDofBlock] = if canonical {
            sp.element_face_blocks(e)
        } else {
            &no_blocks
        };
        accumulate_vector_bilinear_element_blocks(
            sp, e, &[&cc], qo_cc, &mut coo, blocks,
        );
        accumulate_vector_bilinear_element_blocks(
            sp, e, &[&mass], qo_mass, &mut coo, blocks,
        );
    }
    coo.into_csr()
}

// ─── 2D ─────────────────────────────────────────────────────────────────────

fn solve_2d(args: &Args, mut mesh: Mesh<2>) {
    let dim = 2;
    let n_ref = ((50000.0 / mesh.n_elems() as f64).ln() / 2.0_f64.ln() / dim as f64).floor() as usize;
    for _ in 0..n_ref { mesh = refine_uniform(&mesh); }

    let space = HCurlSpace::new(mesh, args.order);
    let n_dofs = space.n_dofs();
    println!("\nNumber of finite element unknowns: {n_dofs}");

    let tags = space.mesh().unique_boundary_tags();
    let ess_bdr = boundary_dofs_hcurl(space.mesh(), &space, &tags);

    let kappa = args.freq * PI;
    let qo = args.order as u8 * 2;
    let mut rhs = assemble_linear(args.canonical, &space, &[&Src2D { kappa }], qo);
    let u_proj = project_2d(&space, kappa);
    let bc_vals: Vec<f64> = ess_bdr.iter().map(|&d| u_proj[d as usize]).collect();

    let mut mat =
        assemble_mat_mfem_rule(args.canonical, &space, 1.0, 1.0, args.order.max(1) as u8);
    let mut x = u_proj.clone();
    form_linear_system(&mut mat, &mut rhs, &mut x, &ess_bdr, &bc_vals);

    solve_report_2d(&space, &mut x, &rhs, args, kappa, &mat);
    write_mfem_file("refined.mesh", space.mesh()).unwrap();
    write_mfem_gf_file("sol.gf", dim, &x, "ND", args.order, dim, 8).unwrap();
}

struct Src2D { kappa: f64 }
impl VectorLinearIntegrator for Src2D {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f: &mut [f64]) {
        let x = qp.x_phys;
        let c = 1.0 + self.kappa * self.kappa;
        let fx = c * (self.kappa * x[1]).sin();
        let fy = c * (self.kappa * x[0]).sin();
        for i in 0..qp.n_dofs { f[i] += qp.weight * (qp.phi_vec[i*2]*fx + qp.phi_vec[i*2+1]*fy); }
    }
}

fn exact_2d(x: &[f64], k: f64) -> [f64; 2] { [(k*x[1]).sin(), (k*x[0]).sin()] }

fn project_2d(space: &HCurlSpace<Mesh<2>>, k: f64) -> Vec<f64> {
    space.interpolate_vector(&|x| exact_2d(x, k).to_vec()).into_vec()
}

fn l2_err_2d<F>(mesh: &Mesh<2>, sp: &HCurlSpace<Mesh<2>>, u: &[f64], ex: &F, canonical: bool) -> f64
where
    F: Fn(&[f64]) -> [f64; 2],
{
    use fem_element::nedelec::TriNDk;
    let k = sp.order() as usize;
    let mut e2 = 0.0;
    for e in mesh.elem_iter() {
        // Order-generic reference element: with `TriNDk::new(1)` the first
        // three DOFs of an ND2 element were contracted with the ND1 basis,
        // i.e. the reported error was meaningless for `-o 2` (D48).
        let r = TriNDk::new(k);
        let n = r.n_dofs();
        // MFEM `GridFunction::ComputeL2Error` (fem/gridfunc.cpp:3410):
        // intorder = 2*fe->GetOrder() + 3.
        let q = r.quadrature((2 * k + 3) as u8);
        let mut p = vec![0.0; n*2];
        // Reconstruction on the element basis (`u_local = T·u_canon`).
        let uloc = element_local_dofs(canonical, sp, e, u);
        let nd = mesh.elem_nodes(e);
        let x0 = mesh.node_coords(nd[0]);
        let x1 = mesh.node_coords(nd[1]);
        let x2 = mesh.node_coords(nd[2]);
        // Jacobian: J = [x1-x0 | x2-x0] (columns are edge vectors)
        let j00 = x1[0]-x0[0]; let j01 = x2[0]-x0[0];
        let j10 = x1[1]-x0[1]; let j11 = x2[1]-x0[1];
        let det_j = j00*j11 - j01*j10;
        let inv_det = 1.0 / det_j;
        // H(curl) covariant Piola: phi_phys = J^{-T} * phi_ref.
        // J = [[j00, j01], [j10, j11]], so
        // J^{-T} = (J^{-1})^T = [[j11, -j10], [-j01, j00]] / det — the
        // off-diagonal entries of the adjugate must be *transposed*; this
        // evaluator used to apply J^{-1} (D57: the 2-D L2 error was wrong on
        // any non-right triangle, e.g. beam-tri 32.96 vs C++ 8.01e-2).
        let jt00 =  j11*inv_det; let jt01 = -j10*inv_det;
        let jt10 = -j01*inv_det; let jt11 =  j00*inv_det;
        for (qi, xi) in q.points.iter().enumerate() {
            r.eval_basis_vec(xi, &mut p);
            let w = q.weights[qi] * det_j.abs();
            let mut uh = [0.0; 2];
            for a in 0..n {
                uh[0] += uloc[a] * (jt00*p[a*2] + jt01*p[a*2+1]);
                uh[1] += uloc[a] * (jt10*p[a*2] + jt11*p[a*2+1]);
            }
            let xp = [
                (1.0-xi[0]-xi[1])*x0[0]+xi[0]*x1[0]+xi[1]*x2[0],
                (1.0-xi[0]-xi[1])*x0[1]+xi[0]*x1[1]+xi[1]*x2[1]
            ];
            let e = ex(&xp);
            e2 += w * ((uh[0]-e[0]).powi(2) + (uh[1]-e[1]).powi(2));
        }
    }
    e2.sqrt()
}

fn solve_report_2d(
    sp: &HCurlSpace<Mesh<2>>,
    x: &mut [f64],
    b: &[f64],
    a: &Args,
    k: f64,
    mat: &fem_linalg::CsrMatrix<f64>,
) {
    if a.no_ams {
        let precond = fem_solver::GSSmoother::from_csr(&fem_linalg::fem_to_linlvo_csr(mat)).unwrap();
        // MFEM prints "PCG: No convergence!" and still reports the error; do
        // the same instead of aborting, so the L^2 number stays comparable.
        match solve_pcg(mat, b, x, &precond, 1e-12, 500, true) {
            Ok(r) => println!("PCG+GSSmoother: {} iters, ||r||/||b|| = {:.3e}", r.iterations, r.final_residual),
            Err(_) => println!("PCG: No convergence!"),
        }
    } else {
        use fem_solver::{solve_pcg_ams, AmsSolverConfig, AmsConfig};
        use fem_linalg::fem_to_linlvo_csr as ftl;
        let g = DiscreteLinearOperator::gradient(&fem_space::H1Space::new(sp.mesh().clone(), 1), sp).unwrap();
        let r = solve_pcg_ams(mat, &ftl(&g), b, x, &AmsSolverConfig {
            inner_cfg: SolverConfig { rtol: 1e-12, atol: 1e-20, max_iter: 2000, verbose: true, ..SolverConfig::default() },
            ams_cfg: AmsConfig::default(),
        }).unwrap();
        println!("PCG+AMS: {} iters, ||r||/||b|| = {:.3e}", r.iterations, r.final_residual);
    }
    println!("\n|| E_h - E ||_{{L^2}} = {:.14e}\n", l2_err_2d(sp.mesh(), sp, x, &|xi| exact_2d(xi, k), a.canonical));
}

// ─── 3D ─────────────────────────────────────────────────────────────────────

fn solve_3d(args: &Args, mut mesh: Mesh<3>) {
    let dim = 3;
    let n_ref = ((50000.0 / mesh.n_elems() as f64).ln() / 2.0_f64.ln() / dim as f64).floor() as usize;
    for _ in 0..n_ref { mesh = fem_mesh::amr::refine_uniform_3d(&mesh); }

    let space = HCurlSpace::new(mesh, args.order);
    let n_dofs = space.n_dofs();
    println!("\nNumber of finite element unknowns: {n_dofs}");

    let tags = space.mesh().unique_boundary_tags();
    let ess_bdr = boundary_dofs_hcurl(space.mesh(), &space, &tags);

    let kappa = args.freq * PI;
    let qo = args.order as u8 * 2;
    let mut rhs = assemble_linear(args.canonical, &space, &[&Src3D { kappa }], qo);
    let u_proj = project_3d(&space, kappa);
    let bc_vals: Vec<f64> = ess_bdr.iter().map(|&d| u_proj[d as usize]).collect();

    let mut mat =
        assemble_mat_mfem_rule(args.canonical, &space, 1.0, 1.0, args.order.max(1) as u8);
    let mut x = u_proj.clone();
    form_linear_system(&mut mat, &mut rhs, &mut x, &ess_bdr, &bc_vals);

    solve_report(&space, &mut x, &rhs, args, &mat);
    println!("\n|| E_h - E ||_{{L^2}} = {:.14e}\n", l2_err_3d(space.mesh(), &space, &x, &|xi| exact_3d(xi, kappa), args.canonical));
    write_mfem_file_3d("refined.mesh", space.mesh()).unwrap();
    write_mfem_gf_file("sol.gf", dim, &x, "ND", args.order, dim, 8).unwrap();
}

struct Src3D { kappa: f64 }
impl VectorLinearIntegrator for Src3D {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f: &mut [f64]) {
        let x = qp.x_phys;
        let c = 1.0 + self.kappa * self.kappa;
        let fx = c * (self.kappa * x[1]).sin();
        let fy = c * (self.kappa * x[2]).sin();
        let fz = c * (self.kappa * x[0]).sin();
        for i in 0..qp.n_dofs { f[i] += qp.weight * (qp.phi_vec[i*3]*fx + qp.phi_vec[i*3+1]*fy + qp.phi_vec[i*3+2]*fz); }
    }
}

fn exact_3d(x: &[f64], k: f64) -> [f64; 3] { [(k*x[1]).sin(), (k*x[2]).sin(), (k*x[0]).sin()] }

fn project_3d(space: &HCurlSpace<Mesh<3>>, k: f64) -> Vec<f64> {
    space.interpolate_vector(&|x| exact_3d(x, k).to_vec()).into_vec()
}

/// Physical Jacobian of element `e` at `xi` and the physical point.
///
/// Handles the two volume geometries ex3 can produce: the affine `Tet4`
/// (`J = [P1-P0 | P2-P0 | P3-P0]`) and the **trilinear** `Hex8` (MFEM's vertex
/// order = bottom face CCW then top face CCW on the `[-1,1]³` cube).  The hex
/// case used to fall into the tet branch and read nodes 0..3 (a coplanar,
/// hence singular, "Jacobian"), so the hex `L²` error was `||E||` exactly.
fn jac_3d(mesh: &Mesh<3>, e: u32, xi: &[f64]) -> (nalgebra::DMatrix<f64>, [f64; 3]) {
    let n = mesh.element_nodes(e);
    if n.len() == 8 {
        const C: [[f64; 3]; 8] = [
            [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
        ];
        let mut j = nalgebra::DMatrix::<f64>::zeros(3, 3);
        let mut xp = [0.0_f64; 3];
        for (i, c) in C.iter().enumerate() {
            let p = mesh.node_coords(n[i]);
            let a = [
                1.0 + xi[0] * c[0],
                1.0 + xi[1] * c[1],
                1.0 + xi[2] * c[2],
            ];
            // dN_i/dξ_c = c_c (1 + ξ_d ξ_i^d)(1 + ξ_e ξ_i^e) / 8
            let dn = [
                c[0] * a[1] * a[2] / 8.0,
                a[0] * c[1] * a[2] / 8.0,
                a[0] * a[1] * c[2] / 8.0,
            ];
            let ni = a[0] * a[1] * a[2] / 8.0;
            for d in 0..3 {
                for cc in 0..3 {
                    j[(d, cc)] += p[d] * dn[cc];
                }
                xp[d] += p[d] * ni;
            }
        }
        return (j, xp);
    }
    let x0 = mesh.node_coords(n[0]); let x1 = mesh.node_coords(n[1]);
    let x2 = mesh.node_coords(n[2]); let x3 = mesh.node_coords(n[3]);
    let (a, b, c) = (xi[0], xi[1], xi[2]);
    let j = nalgebra::dmatrix![
        -x0[0]+x1[0], -x0[0]+x2[0], -x0[0]+x3[0];
        -x0[1]+x1[1], -x0[1]+x2[1], -x0[1]+x3[1];
        -x0[2]+x1[2], -x0[2]+x2[2], -x0[2]+x3[2]
    ];
    let xp = [
        (1.0-a-b-c)*x0[0]+a*x1[0]+b*x2[0]+c*x3[0],
        (1.0-a-b-c)*x0[1]+a*x1[1]+b*x2[1]+c*x3[1],
        (1.0-a-b-c)*x0[2]+a*x1[2]+b*x2[2]+c*x3[2],
    ];
    (j, xp)
}

fn l2_err_3d<F>(mesh: &Mesh<3>, sp: &HCurlSpace<Mesh<3>>, u: &[f64], ex: &F, canonical: bool) -> f64
where
    F: Fn(&[f64]) -> [f64; 3],
{
    use fem_element::nedelec::{TetNDk, HexNDk};
    let k = sp.order() as usize;
    let mut e2 = 0.0;
    for e in mesh.elem_iter() {
        // Order-generic reference element (D48): `*NDk::new(1)` contracted the
        // first 6 DOFs of an ND2 element with the ND1 basis.
        let r: &dyn VectorReferenceElement = match mesh.element_type(e) {
            fem_mesh::element_type::ElementType::Hex8 => &HexNDk::new(k),
            _ => &TetNDk::new(k),
        };
        let n = r.n_dofs();
        // MFEM `GridFunction::ComputeL2Error` (fem/gridfunc.cpp:3463):
        // intorder = 2*fe->GetOrder() + 3.
        let q = r.quadrature((2 * k + 3) as u8);
        let mut p = vec![0.0; n*3];
        // Reconstruction on the element basis (`u_local = T·u_canon`), the
        // exact analogue of MFEM's `doftrans.InvTransformPrimal` in
        // `GridFunction::GetVectorValues`.
        let uloc = element_local_dofs(canonical, sp, e, u);
        for (qi, xi) in q.points.iter().enumerate() {
            r.eval_basis_vec(xi, &mut p);
            let (j, xp) = jac_3d(mesh, e, xi);
            let w = q.weights[qi] * j.determinant().abs();
            let jt = j.try_inverse().unwrap_or_default().transpose();
            let mut uh = [0.0; 3];
            for a in 0..n {
                for c in 0..3 {
                    let mut v = 0.0;
                    for kk in 0..3 { v += jt[(c,kk)] * p[a*3+kk]; }
                    uh[c] += uloc[a] * v;
                }
            }
            let ex = ex(&xp);
            e2 += w * ((uh[0]-ex[0]).powi(2) + (uh[1]-ex[1]).powi(2) + (uh[2]-ex[2]).powi(2));
        }
    }
    e2.sqrt()
}

fn solve_report(sp: &HCurlSpace<Mesh<3>>, x: &mut [f64], b: &[f64], a: &Args, mat: &fem_linalg::CsrMatrix<f64>) {
    if a.no_ams {
        // NOTE (D55): MFEM's `PCG()` helper starts `CGSolver` from x = 0
        // (`iterative_mode` false), while `solve_pcg` here keeps the projected
        // x as the initial guess.  On this ill-conditioned beam-hex ND2 system
        // the two PCG paths stop at *different* iterates — MFEM's Gauss-Seidel
        // sweep is dof-ordering dependent, so no fem-rs ordering can reproduce
        // its trajectory bit-for-bit — and the printed L2 is
        // stopping-sensitive: MFEM at its own 291-iteration stop prints
        // 2.5343e-4, but the same C++ pipeline iterated further (500 iters)
        // prints 1.50666e-4, matching fem-rs's converged 1.50664e-4 to 1.6e-5
        // relative.  The discretisations are identical: the *eliminated
        // systems* match C++ entry-invariant to 1e-13 and the exact (dense)
        // Galerkin L2 on the unrefined beam agrees to 14 digits
        // (1.14704645256018e-1 both sides, see
        // `crates/assembly/tests/d55_hex_nd2_system_regression.rs`).
        let precond = fem_solver::GSSmoother::from_csr(&fem_linalg::fem_to_linlvo_csr(mat)).unwrap();
        // MFEM prints "PCG: No convergence!" and still reports the error; do
        // the same instead of aborting, so the L^2 number stays comparable.
        match solve_pcg(mat, b, x, &precond, 1e-12, 500, true) {
            Ok(r) => println!("PCG+GSSmoother: {} iters, ||r||/||b|| = {:.3e}", r.iterations, r.final_residual),
            Err(_) => println!("PCG: No convergence!"),
        }
    } else {
        use fem_solver::{solve_pcg_ams, AmsSolverConfig, AmsConfig};
        use fem_linalg::fem_to_linlvo_csr as ftl;
        let g = DiscreteLinearOperator::gradient(&fem_space::H1Space::new(sp.mesh().clone(), 1), sp).unwrap();
        let r = solve_pcg_ams(mat, &ftl(&g), b, x, &AmsSolverConfig {
            inner_cfg: SolverConfig { rtol: 1e-12, atol: 1e-20, max_iter: 2000, verbose: true, ..SolverConfig::default() },
            ams_cfg: AmsConfig::default(),
        }).unwrap();
        println!("PCG+AMS: {} iters, ||r||/||b|| = {:.3e}", r.iterations, r.final_residual);
    }
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Default)]
struct Args { mesh: Option<String>, order: u8, freq: f64, no_ams: bool, vis: bool, canonical: bool }

fn parse_args() -> Args {
    // MFEM ex3 defaults: GSSmoother + PCG (no AMS, max_iter=500, rtol=1e-12).
    // `canonical` defaults to true: MFEM always applies its DofTransformation,
    // so the canonical basis is the 1:1 translation (D48).
    let mut a = Args { order: 1, freq: 1.0, no_ams: true, canonical: true, ..Args::default() };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => { a.mesh = it.next(); }
            "-o" | "--order" => { a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
            "-f" | "--frequency" => { a.freq = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0); }
            "-no-ams" => { a.no_ams = true; }
            "-ams" => { a.no_ams = false; }
            "-canonical" => { a.canonical = true; }
            "-no-canonical" => { a.canonical = false; }
            "-vis" => { a.vis = true; }
            "-no-vis" => { a.vis = false; }
            _ => {}
        }
    }
    a
}
