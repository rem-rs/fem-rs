//! # Example 31 — Anisotropic Maxwell (1:1 with MFEM ex31, 2-D order-1 path)
//!
//! Solves the definite Maxwell equation `curl μ⁻¹ curl E + Σ·E = f` with the
//! anisotropic tensor `Σ = [[2, 1/√2, 0], [1/√2, 2, 1/√2], [0, 1/√2, 2]]`,
//! all-boundary PEC data (`sol.ProjectCoefficient(E_exact)` + `FormLinearSystem`
//! DIAG_KEEP elimination) and GS-preconditioned PCG — MFEM `ex31.cpp` step by
//! step, with the C++ print format (`Options used:`, `Number of H(Curl)
//! unknowns:`, the PCG `(B r, r)` log and the final `|| E_h - E ||_{H(Curl)}`
//! line in `%g` style via [`fem_solver::fmt_g`]).
//!
//! MFEM's `ND_R2D_FECollection` restricted space is reproduced with the
//! combined `[H¹(z) | H(curl)(xy)]` space: the in-plane components live on the
//! Nédélec edge DOFs, the out-of-plane (z) component on continuous H¹ vertex
//! DOFs.  The global DOF layout is `[z vertex DOFs 0..n_verts | in-plane ND
//! edge DOFs n_verts..]`, matching MFEM's `ND_R2D GetElementVDofs` (verified
//! against `tools/ex31_cpp_helper/ex31_dump.cpp`: 833 = 289 vertex + 544 edge
//! DOFs on the default mesh; element-0 vdofs `0 81 225 84 | 289 290 -292 -293`).
//!
//! ## Status (round 31, D375 closes D128)
//! **Verified against C++ MFEM 4.10 (`$HOME/mfem410_ser/examples/ex31.cpp`):**
//! * `cargo run --release --example mfem_ex31_anisotropic_maxwell --
//!   -m data/inline-quad.mesh -r 2 -no-vis` →
//!   `|| E_h - E ||_{H(Curl)} = 0.181455` — the **entire stdout** (options
//!   block, all 75 PCG `(B r, r)` lines, `Average reduction factor =
//!   0.829075`, final error line) is byte-identical to the C++ run.
//! * triangle meshes agree too: `inline-tri.mesh -r 2` → 0.312913 and
//!   `star.mesh -r 2` → 0.858735, both identical to C++ (star's sheared
//!   triangles exercise the non-diagonal Jacobian path of the curl error).
//! * the dump harness (`tools/ex31_cpp_helper/compare_ex31_systems.py`) agrees
//!   on raw `A`, raw `b`, eliminated `A`/`B`/`X0` and the solution `x` to
//!   ≤ 1e-13 (dumped by `examples/mfem_ex31_dump.rs`, same assembly code).
//!
//! **D384 (raw-A structural zeros):** the `Σ_yz ∫ E_y φ_z` coupling rows of
//! the horizontally-polarized edge functions are *identically zero*; MFEM's
//! assembly drops those element entries (`SparseMatrix::AddSubMatrix` skips
//! `a == 0.0` unless the mirror entry is nonzero), while the pre-D384 port
//! accumulated `≤ 3e-18` rounding noise that passed the `v != 0.0` check and
//! entered the COO (784 spurious nnz = 392 mirrored pairs on the default
//! mesh).  The coupling add now requires `|v| > 1e-12` — nine orders below
//! the smallest physical coupling entry, six above the measured noise, and
//! the margin grows on finer meshes (v scales like h).  Residual, documented
//! nnz difference: C++ *keeps* 392 of its own ≤ 5e-18 noise entries at
//! mirror-nonzero positions (the `skip_zeros == 1` mirror rule), which this
//! port does not reproduce — reproducing them would mean deliberately
//! assembling noise; every compared A/x value still agrees to ≤ 1.3e-13 and
//! the printed metrics are unchanged.
//!
//! **Honest gaps (refused with exit status 3):**
//! * `-o > 1`: the order-p restricted space (ND_R2D edge moments + higher-order
//!   H¹ z-block) is not ported; only the order-1 local basis exists (C++ ex31
//!   supports any order).
//! * 1-D and 3-D meshes: C++ ex31's `ND_R1D_FECollection` / full `ND`
//!   branches are not ported — 2-D meshes only.
//! * nonconforming AMR meshes (`MFEM NC mesh v1.0`, e.g. `amr-quad.mesh`) are
//!   refused by `fem_io`'s reader (C++ ex31 handles them via constraint
//!   tables, which the restricted space does not port).
//! * `-vis`: GLVis streaming is not wired for this example.
//! * curved/second-order element types (Tri6, Quad8, …) are refused by
//!   `setup_element_ref` (only straight-sided Tri3/Quad4 are implemented).
//!
//! The parallel sibling `examples/mfem_pex31_restricted_hcurl.rs` (MFEM ex31p)
//! shares this element machinery.

use std::f64::consts::{PI, SQRT_2};
use std::fs::File;
use std::io::{BufWriter, Write};

use fem_assembly::standard::{CurlCurlIntegrator, DiffusionIntegrator, MassIntegrator,
    VectorMassTensorIntegrator};
use fem_assembly::coefficient::ConstantMatrixCoeff;
use fem_assembly::postproc::grid_function::project_bdr_coefficient_tangent_2d;
use fem_assembly::{VectorAssembler, Assembler, FixedOrder};
use fem_core::types::DofId;
use fem_element::{VectorReferenceElement, ReferenceElement,
    nedelec::{TriNDk, QuadNDk}, lagrange::{TriP1, QuadQk}};
use fem_io::mfem::{read_mfem_file, write_mfem};
use fem_linalg::CooMatrix;
use fem_mesh::{ElementType, Mesh, MeshTopology, amr::refine_uniform};
use fem_solver::{SolverConfig, fmt_g, solve_pcg_gssmoother};
use fem_space::{HCurlSpace, H1Space,
    fe_space::FESpace, constraints::{boundary_dofs_hcurl, boundary_dofs}};

// ─── MFEM ex31 coefficients (2‑D case) ──────────────────────────────

const A0: f64 = 1.1; const A1: f64 = 1.2; const A2: f64 = 1.3;
const PHI1: f64 = 0.4 * PI; const PHI2: f64 = 0.9 * PI;

/// Σ = [[2, 1/√2, 0], [1/√2, 2, 1/√2], [0, 1/√2, 2]]
const SXX: f64 = 2.0; const SXY: f64 = 1.0 / SQRT_2;
const SYY: f64 = 2.0; const SYZ: f64 = 1.0 / SQRT_2; const SZZ: f64 = 2.0;

// ─── CLI (mirrors ex31 OptionsParser) ───────────────────────────────

struct Args { mesh_file: String, ref_levels: usize, order: u8, freq: f64, visualization: bool }
fn parse_args() -> Args {
    let mut a = Args {
        mesh_file: "data/inline-quad.mesh".into(),
        ref_levels: 2, order: 1, freq: 1.0, visualization: true,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh_file = it.next().unwrap_or_default(),
            "-r" | "--refine" => a.ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(2),
            "-o" | "--order" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-f" | "--frequency" => a.freq = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            "-vis" | "--visualization" => a.visualization = true,
            "-no-vis" | "--no-visualization" => a.visualization = false,
            _ => {}
        }
    }
    a
}

// ─── Exact solution (1:1 with ex31 E_exact / CurlE_exact / f_exact, dim == 2) ──

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

fn source_3d(x: &[f64], kappa: f64) -> [f64; 3] {
    let k2 = kappa * kappa;
    let u = (kappa / SQRT_2) * (x[0] + x[1]);
    let (s0, s4, s9) = (u.sin(), (u + PHI1).sin(), (u + PHI2).sin());
    let f0 = 0.55 * (4.0 + k2) * s0 + 0.6 * (SQRT_2 - k2) * s4;
    let f1 = 0.55 * (SQRT_2 - k2) * s0 + 0.6 * (4.0 + k2) * s4 + 0.65 * SQRT_2 * s9;
    let f2 = 0.6 * SQRT_2 * s4 + 1.3 * (2.0 + k2) * s9;
    [f0, f1, f2]
}

// ─── Element reference helpers (2-D, straight Tri3 / Quad4) ─────────

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
    // Geometry element on [0,1]^2 (MFEM BiLinear2DFiniteElement), matching
    // QuadQk used by the assembler for the H1 z-space (QuadQ1 is [-1,1]^2 and
    // would mix reference domains with QuadND1's [0,1]^2 quadrature).
    let geo = QuadQk::new(1);
    let n_geo = geo.n_dofs();
    let mut grad = vec![0.0_f64; n_geo * 2];
    geo.eval_grad_basis(xi, &mut grad);
    let mut j = nalgebra::DMatrix::<f64>::zeros(2, 2);
    for k in 0..n_geo {
        let xk = mesh.node_coords(nodes[k]);
        for i in 0..2 { for d in 0..2 { j[(i, d)] += xk[i] * grad[k * 2 + d]; } }
    }
    let det = j.determinant();
    let inv = 1.0 / det;
    (inv, j[(1,1)] * inv, -j[(1,0)] * inv, -j[(0,1)] * inv, j[(0,0)] * inv, det.abs())
}

/// Local reference elements of the `[z | nd]` combined space for one element.
///
/// Returns `(n_nd_ldofs, nd_ref, h1_ref, n_h1_ldofs, jacobian)`.
///
/// `n_nd_ldofs` **must** come from `nd_ref.n_dofs()`: the Whitney 1-form of
/// `QuadNDk::new(1)` fills `2 · n_dofs() = 8` slots (4 edge DOFs × 2
/// components), while `TriNDk::new(1)` fills 6 (3 × 2).  A hard-coded 3 here
/// (the pre-round-31 typo for `Quad4`) under-sizes every `n_ld * 2` scratch
/// buffer and trips `QuadNDk::eval_basis_vec`'s `values[6]` write (D137b).
fn setup_element_ref(et: ElementType, order: u8) -> (usize, &'static dyn VectorReferenceElement, Box<dyn ReferenceElement>, usize, JacobianFn) {
    // Only order 1 is wired up (see the `-o > 1` refusal in `main`).
    assert_eq!(order, 1, "setup_element_ref only implements order 1");
    match et {
        ElementType::Tri3 => {
            // Leak to get 'static lifetime (acceptable for singleton reference elements)
            let nd: &'static TriNDk = Box::leak(Box::new(TriNDk::new(1)));
            (nd.n_dofs(), nd as &dyn VectorReferenceElement, Box::new(TriP1), 3, affine_jac as JacobianFn)
        },
        ElementType::Quad4 => {
            let nd: &'static QuadNDk = Box::leak(Box::new(QuadNDk::new(1)));
            (nd.n_dofs(), nd as &dyn VectorReferenceElement, Box::new(QuadQk::new(1)), 4, isoparametric_jac as JacobianFn)
        },
        _ => {
            // Not a panic: an unsupported element type (e.g. a curved Tri6 /
            // Quad9 mesh) is a declared gap of this port.
            eprintln!(
                "mfem_ex31_anisotropic_maxwell: unsupported element type {et:?} (only straight-sided \
                 Tri3 and Quad4 are implemented) — exiting with status 3"
            );
            std::process::exit(3)
        }
    }
}

/// Physical coordinates of the quadrature point on element `e` (Tri: affine,
/// Quad: isoparametric Q1 geometry on [0,1]²).
fn phys_point(mesh: &Mesh<2>, et: ElementType, nodes: &[u32], xi: &[f64]) -> [f64; 2] {
    if et == ElementType::Quad4 {
        let geo = QuadQk::new(1);
        let ng = geo.n_dofs();
        let mut phi = vec![0.0; ng];
        geo.eval_basis(xi, &mut phi);
        let mut p = [0.0_f64; 2];
        for k in 0..ng {
            let c = mesh.node_coords(nodes[k]);
            p[0] += phi[k] * c[0];
            p[1] += phi[k] * c[1];
        }
        p
    } else {
        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        [x0[0] + (x1[0]-x0[0])*xi[0] + (x2[0]-x0[0])*xi[1],
         x0[1] + (x1[1]-x0[1])*xi[0] + (x2[1]-x0[1])*xi[1]]
    }
}

// ─── Main ───────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    // Honest gaps: order-p restricted space and GLVis streaming are not
    // ported; refuse instead of degrading (C++ ex31 supports both).
    if args.order != 1 {
        eprintln!(
            "mfem_ex31_anisotropic_maxwell: -o {} (order-p restricted H(curl) space) is not \
             ported; only the order-1 local basis exists (C++ ex31 supports any order). \
             Re-run with -o 1.",
            args.order
        );
        std::process::exit(3);
    }
    if args.visualization {
        eprintln!(
            "mfem_ex31_anisotropic_maxwell: GLVis visualization (-vis) is not ported for this \
             example. Re-run with -no-vis."
        );
        std::process::exit(3);
    }

    println!("Options used:");
    println!("   --mesh {}", args.mesh_file);
    println!("   --refine {}", args.ref_levels);
    println!("   --order {}", args.order);
    println!("   --frequency {}", fmt_g(args.freq));
    println!("   --no-visualization");
    let kappa = args.freq * PI;

    // 2. Read the mesh (triangles, quadrilaterals or mixed).  This port is the
    //    2-D restricted H(curl) path (ND_R2D_FECollection); C++ ex31's 1-D
    //    (ND_R1D) and 3-D (ND) branches are declared gaps — refuse them.
    let mfem = match read_mfem_file(&args.mesh_file) {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "mfem_ex31_anisotropic_maxwell: cannot read mesh file '{}': {e} — exiting with \
                 status 3",
                args.mesh_file
            );
            std::process::exit(3)
        }
    };
    let base_mesh: Mesh<2> = match mfem.mesh2d {
        Some(m) => m,
        None => {
            eprintln!(
                "mfem_ex31_anisotropic_maxwell: mesh '{}' is not 2-D; this port implements only \
                 the 2-D restricted H(curl) path (C++ ex31's 1-D ND_R1D and 3-D ND branches are \
                 not ported) — exiting with status 3",
                args.mesh_file
            );
            std::process::exit(3)
        }
    };

    // 3. Uniform refinements.
    let mesh = if args.ref_levels > 0 {
        let mut m = base_mesh;
        for _ in 0..args.ref_levels { m = refine_uniform(&m); }
        m
    } else { base_mesh };

    // 4. The restricted H(curl) space as `[z | nd]`: H¹ (z component, vertex
    //    DOFs) + Nédélec (in-plane, edge DOFs).  Global layout matches MFEM's
    //    single ND_R2D FiniteElementSpace::GetElementVDofs: vertex (z) DOFs
    //    first (dof = vertex id), then edge (in-plane) DOFs (dof = n_verts +
    //    edge id) — verified against the C++ dump harness (833 = 289 + 544).
    let quad_order = args.order * 2 + 2;
    let nd_space = HCurlSpace::new(mesh.clone(), args.order);
    let z_space = H1Space::new(mesh.clone(), args.order);
    let n_nd = nd_space.n_dofs();
    let n_h1 = z_space.n_dofs();
    let n_total = n_nd + n_h1;
    println!("Number of H(Curl) unknowns: {n_total}");

    // 5. Essential (Dirichlet) DOFs: ALL boundary attributes (PEC).
    let bdr_tags = mesh.unique_boundary_tags();
    let nd_bdr = boundary_dofs_hcurl(&mesh, &nd_space, &bdr_tags);
    let h1_bdr = boundary_dofs(&mesh, z_space.dof_manager(), &bdr_tags);

    // 6. Linear form b = (f, φ_i) — MFEM VectorFEDomainLFIntegrator uses the
    //    order-2·GetOrder() rule for the (non-polynomial) source; the H¹ z
    //    part assembles through the scalar assembler.
    let src_nd = FixedOrder::new(FnVectorSource(Box::new(move |x| {
        let f = source_3d(x, kappa); [f[0], f[1]]
    })), 2);
    let rhs_nd = VectorAssembler::assemble_linear(&nd_space, &[&src_nd], quad_order);
    let src_z = FixedOrder::new(FnScalarSource(Box::new(move |x| source_3d(x, kappa)[2])), 2);
    let rhs_z = Assembler::assemble_linear(&z_space, &[&src_z], quad_order);
    let mut b = vec![0.0_f64; n_total];
    for i in 0..n_h1 { b[i] = rhs_z[i]; }
    for i in 0..n_nd { b[n_h1 + i] = rhs_nd[i]; }

    // 7. Initialize the solution by projecting the exact solution (all DOFs of
    //    the data path: boundary values below, interior left at zero — MFEM
    //    FormLinearSystem's default copy_interior = 0 zeroes the interior of
    //    the PCG initial guess X anyway).
    let mut x = vec![0.0_f64; n_total];
    project_bdr_coefficient_tangent_2d(&mut x[n_h1..], &nd_space,
        &|x: &[f64], out: &mut [f64]| { let e = exact_e(x, kappa); out[0] = e[0]; out[1] = e[1]; },
        &bdr_tags);
    for &d in &h1_bdr { let c = z_space.dof_manager().dof_coord(d); x[d as usize] = exact_e(c, kappa)[2]; }

    // 8. Bilinear form curl μ⁻¹ curl + Σ:
    //    in-plane block (ND): CurlCurlIntegrator (MFEM Pk rule 2p-2 = 0) +
    //    VectorMassTensorIntegrator (MFEM rule OrderW + 2p = 2, affine);
    //    z block (H¹): the curl-curl Ez part is a plain -∇² under that same
    //    single-point rule (FixedOrder 0) plus the Σ_zz mass under the order-2
    //    rule; coupling Σ_yz ∫ E_y φ_z (order-2 rule).  All integrands here
    //    are polynomials these rules integrate exactly, so each block matches
    //    C++ MFEM entry for entry (dump harness: max |diff| 5.7e-14 on A).
    let cc0 = FixedOrder::new(CurlCurlIntegrator { mu: 1.0 }, 0);
    let vm2 = FixedOrder::new(VectorMassTensorIntegrator { alpha: ConstantMatrixCoeff(vec![SXX, SXY, SXY, SYY]) }, 2);
    let a_nd = VectorAssembler::assemble_bilinear(&nd_space, &[&cc0, &vm2], quad_order);

    let laplace = FixedOrder::new(DiffusionIntegrator { kappa: 1.0 }, 0);
    let z_mass = FixedOrder::new(MassIntegrator { rho: SZZ }, 2);
    let a_z = Assembler::assemble_bilinear(&z_space, &[&laplace, &z_mass], quad_order);

    let mut coupling_coo = CooMatrix::<f64>::new(n_nd, n_h1);
    for e in 0..mesh.n_elements() as u32 {
        let nd_dofs: Vec<usize> = nd_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let h1_dofs: Vec<usize> = z_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = nd_space.mesh().element_nodes(e);
        let signs = nd_space.element_signs(e);
        let (n_ld, rnd, rh1, n_lh1, jac_fn) = setup_element_ref(mesh.element_type(e), args.order);
        let q = rnd.quadrature(2); // MFEM VectorFEMassIntegrator rule: OrderW + 2p = 2
        let mut np = vec![0.0; n_ld * 2];
        let mut hp = vec![0.0; n_lh1];
        let mut em = vec![0.0_f64; n_ld * n_lh1];
        for (qi, xi) in q.points.iter().enumerate() {
            let (_, _jit00, _jit01, jit10, jit11, det) = jac_fn(&mesh, e, nodes, xi);
            let w = q.weights[qi] * det * SYZ;
            rnd.eval_basis_vec(xi, &mut np);
            rh1.eval_basis(xi, &mut hp);
            for i in 0..n_ld {
                let py = signs[i] * (jit10 * np[i * 2] + jit11 * np[i * 2 + 1]);
                for j in 0..n_lh1 { em[i * n_lh1 + j] += w * py * hp[j]; }
            }
        }
        for (li, &ri) in nd_dofs.iter().enumerate() {
            for (lj, &cj) in h1_dofs.iter().enumerate() {
                let v = em[li * n_lh1 + lj];
                // D384: the coupling rows of horizontally-polarized edge
                // functions are *identically zero* (E_y ≡ 0); their rounding
                // noise is ≤ 3e-18 on this mesh while the smallest physical
                // coupling entry is ≈ 3.7e-3, so anything below 1e-12 is
                // pattern noise, not signal.  MFEM's assembly drops the same
                // positions at exact zero (`SparseMatrix::AddSubMatrix`
                // skip_zeros); the pex31 port filters the equivalent rows at
                // generation.  1e-12 is nine orders below signal and six
                // above the measured noise, and v scales down with h² so the
                // margin only grows on finer meshes.
                if v.abs() > 1e-12 { coupling_coo.add(ri, cj, v); }
            }
        }
    }
    let coupling = coupling_coo.into_csr();

    // 9. Combine the blocks into the single [z | nd] system (z rows/cols first,
    //    in-plane ND offset by n_h1) and apply the essential BCs with MFEM's
    //    default DIAG_KEEP policy (BilinearForm::FormLinearSystem):
    //    keep the diagonal, zero the rest of the BC rows/cols, adjust the RHS.
    let mut sys_coo = CooMatrix::<f64>::new(n_total, n_total);
    for r in 0..n_nd {
        let rr = n_h1 + r;
        for k in a_nd.row_ptr[r]..a_nd.row_ptr[r + 1] {
            sys_coo.add(rr, n_h1 + a_nd.col_idx[k] as usize, a_nd.values[k]);
        }
    }
    for r in 0..n_h1 {
        for k in a_z.row_ptr[r]..a_z.row_ptr[r + 1] {
            sys_coo.add(r, a_z.col_idx[k] as usize, a_z.values[k]);
        }
    }
    for r in 0..coupling.nrows {
        for k in coupling.row_ptr[r]..coupling.row_ptr[r + 1] {
            let c = coupling.col_idx[k] as usize;
            let v = coupling.values[k];
            if v != 0.0 { sys_coo.add(n_h1 + r, c, v); sys_coo.add(c, n_h1 + r, v); }
        }
    }
    let mut mat = sys_coo.into_csr();

    let mut bdr_dofs: Vec<DofId> = nd_bdr.iter().map(|&d| (n_h1 + d as usize) as DofId).collect();
    let mut bdr_vals: Vec<f64> = nd_bdr.iter().map(|&d| x[n_h1 + d as usize]).collect();
    for &d in &h1_bdr { bdr_dofs.push(d); bdr_vals.push(x[d as usize]); }
    // MFEM 4.10 BilinearForm default diag_policy = DIAG_KEEP (bilinearform.hpp):
    // EliminateVDofs keeps the diagonal and zeroes the rest of the BC rows/cols,
    // then EliminateVDofsInRHS adjusts the RHS.  Apply per-DOF in the same way.
    for (&dof, &val) in bdr_dofs.iter().zip(bdr_vals.iter()) {
        mat.apply_dirichlet_keep_diag(dof as usize, val, &mut b);
    }

    // 10. Solve A X = B with PCG + GSSmoother.  C++ calls the free-function
    //     wrapper PCG(*A, M, B, X, 1, 500, 1e-12, 0.0), which sets
    //     SetRelTol(sqrt(1e-12)) = 1e-6 (linalg/solvers.cpp) — solve_pcg_gssmoother
    //     is the bit-for-bit MFEM CGSolver+GSSmoother port (fwd+back GS sweeps,
    //     convergence on (B r, r) ≤ rtol²·(B r, r)₀).
    let cfg = SolverConfig {
        rtol: 1e-6,
        max_iter: 500,
        verbose: true,
        ..Default::default()
    };
    solve_pcg_gssmoother(&mat, &b, &mut x, &cfg).expect("PCG");

    // 13. H(Curl) norm of the error (2-D: in-plane ND + z-H1 evaluation; MFEM
    //     ComputeHCurlError uses the 2·order + 3 rule).
    let hcurl_err = compute_hcurl_error(&mesh, &nd_space, &z_space, &x, args.order, kappa);
    println!("\n|| E_h - E ||_{{H(Curl)}} = {}\n", fmt_g(hcurl_err));

    // 14. Save the refined mesh and the solution (GLVis inputs, precision 8).
    {
        let mut mesh_f = File::create("refined.mesh").expect("cannot create refined.mesh");
        write_mfem(&mut mesh_f, &mesh, None).expect("mesh write failed");
        let sol_f = File::create("sol.gf").expect("cannot create sol.gf");
        let mut w = BufWriter::new(sol_f);
        for &v in &x { writeln!(w, "{:.8e}", v).expect("sol write failed"); }
    }
}

// ─── H(Curl) error (2-D: in-plane ND + z-H1, exact 3-component evaluation) ────

fn compute_hcurl_error(
    mesh: &Mesh<2>,
    nd_space: &HCurlSpace<Mesh<2>>,
    z_space: &H1Space<Mesh<2>>,
    x: &[f64],
    order: u8,
    kappa: f64,
) -> f64 {
    let n_h1 = z_space.n_dofs();
    let mut err2 = 0.0_f64;
    for e in 0..mesh.n_elements() as u32 {
        let nd_dofs: Vec<usize> = nd_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let h1_dofs: Vec<usize> = z_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);
        let signs = nd_space.element_signs(e);
        let (n_ld, rnd, rh1, n_lh1, jac_fn) = setup_element_ref(mesh.element_type(e), order);
        let qord = 2 * order + 3;
        let q = rnd.quadrature(qord);
        let mut pn = vec![0.0; n_ld * 2];
        let mut ph = vec![0.0; n_lh1];
        let mut cn = vec![0.0; n_ld];
        for (qi, xi) in q.points.iter().enumerate() {
            let (inv_det, jit00, jit01, jit10, jit11, det) = jac_fn(mesh, e, nodes, xi);
            let w = q.weights[qi] * det;
            let xp = phys_point(mesh, mesh.element_type(e), nodes, xi);
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
                eh[0] += s * x[n_h1 + nd_dofs[i]] * (jit00 * pn[i * 2] + jit01 * pn[i * 2 + 1]);
                eh[1] += s * x[n_h1 + nd_dofs[i]] * (jit10 * pn[i * 2] + jit11 * pn[i * 2 + 1]);
            }
            for j in 0..n_lh1 { eh[2] += x[h1_dofs[j]] * ph[j]; }
            let mut ce = [0.0_f64; 3];
            for i in 0..n_ld { ce[2] += signs[i] * x[n_h1 + nd_dofs[i]] * cn[i]; }
            ce[2] *= inv_det;
            let mut gr = vec![0.0_f64; n_lh1 * 2];
            rh1.eval_grad_basis(xi, &mut gr);
            for j in 0..n_lh1 {
                // ∇z_phys = J⁻¹·∇z_ref (MFEM GetCurl: grad_hat · J⁻¹, column
                // convention): ∂z/∂x = J⁻¹(0,0)·gξ + J⁻¹(1,0)·gη = jit00·gξ +
                // jit01·gη, ∂z/∂y = J⁻¹(0,1)·gξ + J⁻¹(1,1)·gη = jit10·gξ +
                // jit11·gη (jit00=j11/det, jit01=−j10/det, jit10=−j01/det,
                // jit11=j00/det).  curl_x = ∂Ez/∂y, curl_y = −∂Ez/∂x.  The
                // transposed convention agrees on the axis-aligned inline-quad
                // elements but breaks on the sheared triangles of split
                // squares (star/inline-tri) — verified against C++ on both.
                let dx = jit00 * gr[j * 2] + jit01 * gr[j * 2 + 1];
                let dy = jit10 * gr[j * 2] + jit11 * gr[j * 2 + 1];
                ce[0] += x[h1_dofs[j]] * dy;
                ce[1] -= x[h1_dofs[j]] * dx;
            }
            let (ee, ec) = (exact_e(&xp, kappa), exact_curl(&xp, kappa));
            for c in 0..3 {
                let d = eh[c] - ee[c]; err2 += w * d * d;
                let dc = ce[c] - ec[c]; err2 += w * dc * dc;
            }
        }
    }
    err2.sqrt()
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
