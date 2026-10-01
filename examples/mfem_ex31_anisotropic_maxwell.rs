//! # Example 31 — Anisotropic Maxwell (1:1 with MFEM ex31, 2-D path)
//!
//! Solves the definite Maxwell equation `curl μ⁻¹ curl E + Σ·E = f` with the
//! anisotropic tensor `Σ = [[2, 1/√2, 0], [1/√2, 2, 1/√2], [0, 1/√2, 2]]`,
//! all-boundary PEC data (`sol.ProjectCoefficient(E_exact)` + `FormLinearSystem`
//! DIAG_KEEP elimination) and GS-preconditioned PCG — MFEM `ex31.cpp` step by
//! step, with the C++ print format (`Options used:`, `Number of H(Curl)
//! unknowns:`, the PCG `(B r, r)` log and the final `|| E_h - E ||_{H(Curl)}`
//! line in `%g` style via [`fem_solver::fmt_g`]).
//!
//! ## Round 102 — the collection-level `ND_R2D` port (supersedes the round-31
//! order-1-only restricted space)
//!
//! The discretization is now the real `ND_R2D_FECollection(order, 2)`:
//!
//! * space: [`fem_space::embedded_r2d::HCurlR2dSpace`] — MFEM's
//!   `FiniteElementSpace` dof layout for the embedded collection (vertex
//!   z-dofs, `2p−1`-dof edge blocks through the collection's `SegDofOrd`
//!   orientation tables, element interiors; pinned slot-for-slot against
//!   `tmp/d102r2d/probe_space.cpp`),
//! * elements: [`fem_element::embedded::NdR2dTri`] /
//!   [`fem_element::embedded::NdR2dQuad`] — 1:1 ports of MFEM's
//!   `ND_R2D_TriangleElement` / `ND_R2D_QuadrilateralElement` (pinned against
//!   `tmp/d102r2d/probe_r2d.cpp`: shapes, curls, dof sites and `Project` to
//!   6.6e-15 relative over 822 comparisons),
//! * assembly: the C++ integrator rules verbatim — `CurlCurlIntegrator`
//!   (Pk: `2p−2`), `VectorFEMassIntegrator` (`OrderW + 2p`, `OrderW = 0`
//!   affine tri / `1` bilinear quad), `VectorFEDomainLFIntegrator` (`2p`),
//!   `ComputeHCurlError` (`2p + 3`) — with the D384 raw-A structural-zero
//!   filter applied to the assembled matrix (see the history section).
//!
//! Every order `-o >= 1` is supported (C++ ex31 parity); the round-31
//! `[H¹(z) | H(curl)(xy)]` block machinery and its order-1-only local basis
//! are gone — the Σ couplings (`Σ_xz`, `Σ_yz`, …) now flow through the single
//! element matrix exactly as in MFEM.
//!
//! ## Status (round 102)
//! **Verified against C++ MFEM 4.10 (`$HOME/work/d102r2d/ex31_cpp`, built
//! from $HOME/mfem410_ser/examples/ex31.cpp):** the **entire stdout is
//! byte-identical** (options block, every PCG `(B r, r)` line, `Average
//! reduction factor`, the final `|| E_h - E ||_{H(Curl)}`) on 12 runs —
//! `inline-quad / hexagon / star / inline-tri` × `-o 1 / 2 / 3` at `-r 2`
//! (e.g. inline-quad: 833 / 3201 / 7105 unknowns → `0.181455` /
//! `0.00295871` / `0.000180716`; `diff` = 0 lines on every run).  The
//! round-31 base was byte-identical only at `-o 1` through the
//! `[H1(z) | H(curl)(xy)]` block path; round 102 replaces it with the
//! collection port and unlocks every order — the acceptance line was
//! 数值一致级 ≤ 1e-10, the delivered parity is bitwise at the print level.
//! Evidence: `tmp/d102r2d/REPORT.md`.
//!
//! **D384 (raw-A structural zeros, history):** the `Σ_yz ∫ E_y φ_z` coupling
//! rows of the horizontally-polarized edge functions are *identically zero*
//! in the order-1 element basis; MFEM's assembly drops those element entries
//! (`SparseMatrix::AddSubMatrix` skips `a == 0.0` unless the mirror entry is
//! nonzero), while accumulating the products in `f64` picks up `≤ 3e-18`
//! rounding noise that entered the COO as spurious nnz.  The assembled matrix
//! here keeps the same `|v| <= 1e-12` filter; MFEM's own ≤ 2.4e-17 noise
//! entries at mirror-nonzero positions are deliberately not reproduced
//! (measured on inline-quad -r2 -o1: 1960 C++-only entries, max |v| 2.3e-17,
//! every compared A/b/x value agreeing to ≤ 4.5e-16).
//!
//! **Honest gaps (refused with exit status 3):**
//! * 3-D meshes (full `ND` collection): not wired in this example — 2-D and
//!   1-D meshes are ported (1-D = `ND_R1D_FECollection`, D901 closed round
//!   104; 3-D remains refused).
//! * nonconforming AMR meshes (`MFEM NC mesh v1.0`, e.g. `amr-quad.mesh`) are
//!   refused by `fem_io`'s reader (C++ ex31 handles them via constraint
//!   tables, which this example does not port).
//! * `-vis`: GLVis streaming is not wired for this example.
//! * curved/second-order element types (Tri6, Quad8, …) are refused by
//!   `elem_jac_at` (only straight-sided Tri3/Quad4 are implemented).
//!
//! The parallel sibling `examples/mfem_pex31_restricted_hcurl.rs` (MFEM ex31p)
//! shares the round-31 element machinery and keeps its own status.

use std::f64::consts::{PI, SQRT_2};
use std::fs::File;
use std::io::{BufWriter, Write};

use fem_assembly::{eliminate_ess_tdofs, ElimPolicy};
use fem_core::types::DofId;
use fem_element::embedded::{Jac2D, NdR2dQuad, NdR2dTri};
use fem_element::quadrature::{gauss_legendre_01, quad_rule_01, tri_rule};
use fem_element::reference::QuadratureRule;
use fem_io::mfem::{read_mfem_file, write_mfem};
use fem_linalg::CooMatrix;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{amr::refine_uniform, amr::refine_uniform_1d, ElementType, Mesh};
use fem_solver::{fmt_g, solve_pcg_gssmoother, SolverConfig};
use fem_space::embedded_r1d::HCurlR1dSpace;
use fem_space::embedded_r2d::HCurlR2dSpace;

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

// ─── Element local engine (straight-sided Tri3 / Quad4) ─────────────────────

/// The ND_R2D reference element of cell `e` (in-plane ND + z-directed H1
/// parts; the dof tables are the MFEM elements 1:1).
enum Cell {
    Tri(NdR2dTri),
    Quad(NdR2dQuad),
}

impl Cell {
    fn new(mesh: &Mesh<2>, e: u32, p: u8) -> Self {
        match mesh.element_type(e) {
            ElementType::Tri3 => Cell::Tri(NdR2dTri::new(p as usize)),
            ElementType::Quad4 => Cell::Quad(NdR2dQuad::new(p as usize)),
            other => panic!(
                "mfem_ex31_anisotropic_maxwell: unsupported element type {other:?} (only \
                 straight-sided Tri3 and Quad4 are implemented)"
            ),
        }
    }
    fn n_dofs(&self) -> usize {
        match self {
            Cell::Tri(el) => el.n_dofs(),
            Cell::Quad(el) => el.n_dofs(),
        }
    }
    fn eval_vshape_phys(&self, xi: &[f64], j: &Jac2D, out: &mut [f64]) {
        match self {
            Cell::Tri(el) => el.eval_vshape_phys(xi, j, out),
            Cell::Quad(el) => el.eval_vshape_phys(xi, j, out),
        }
    }
    fn eval_curl_phys(&self, xi: &[f64], j: &Jac2D, out: &mut [f64]) {
        match self {
            Cell::Tri(el) => el.eval_curl_phys(xi, j, out),
            Cell::Quad(el) => el.eval_curl_phys(xi, j, out),
        }
    }
}

/// Jacobian + physical point of element `e` at the reference point `xi`
/// (affine Tri3 map / the MFEM `BiLinear2DFiniteElement` map on `[0,1]²`).
fn elem_jac_at(mesh: &Mesh<2>, e: u32, xi: [f64; 2]) -> (Jac2D, [f64; 2]) {
    let nodes = mesh.element_nodes(e);
    match mesh.element_type(e) {
        ElementType::Tri3 => {
            let x0 = mesh.node_coords(nodes[0]);
            let x1 = mesh.node_coords(nodes[1]);
            let x2 = mesh.node_coords(nodes[2]);
            let (j00, j01) = (x1[0] - x0[0], x2[0] - x0[0]);
            let (j10, j11) = (x1[1] - x0[1], x2[1] - x0[1]);
            let det = j00 * j11 - j01 * j10;
            (
                Jac2D { j00, j01, j10, j11, det },
                [
                    x0[0] + xi[0] * (x1[0] - x0[0]) + xi[1] * (x2[0] - x0[0]),
                    x0[1] + xi[0] * (x1[1] - x0[1]) + xi[1] * (x2[1] - x0[1]),
                ],
            )
        }
        ElementType::Quad4 => {
            let c: [[f64; 2]; 4] = [
                { let q = mesh.node_coords(nodes[0]); [q[0], q[1]] },
                { let q = mesh.node_coords(nodes[1]); [q[0], q[1]] },
                { let q = mesh.node_coords(nodes[2]); [q[0], q[1]] },
                { let q = mesh.node_coords(nodes[3]); [q[0], q[1]] },
            ];
            let (x, y) = (xi[0], xi[1]);
            let phi = [(1.0 - x) * (1.0 - y), x * (1.0 - y), x * y, (1.0 - x) * y];
            let grad = [[y - 1.0, x - 1.0], [1.0 - y, -x], [y, x], [-y, 1.0 - x]];
            let mut p = [0.0_f64; 2];
            let mut j = [0.0_f64; 4];
            for k in 0..4 {
                p[0] += phi[k] * c[k][0];
                p[1] += phi[k] * c[k][1];
                j[0] += grad[k][0] * c[k][0];
                j[1] += grad[k][1] * c[k][0];
                j[2] += grad[k][0] * c[k][1];
                j[3] += grad[k][1] * c[k][1];
            }
            (
                Jac2D { j00: j[0], j01: j[1], j10: j[2], j11: j[3], det: j[0] * j[3] - j[1] * j[2] },
                p,
            )
        }
        other => panic!(
            "mfem_ex31_anisotropic_maxwell: unsupported element type {other:?} (only \
             straight-sided Tri3 and Quad4 are implemented)"
        ),
    }
}

/// MFEM `IsoparametricTransformation::OrderW()` for the straight-sided cells:
/// `0` for the affine simplex, `Qk order 1 · dim − 1 = 1` for the bilinear
/// quad.
fn order_w(mesh: &Mesh<2>, e: u32) -> u8 {
    if mesh.element_type(e) == ElementType::Quad4 { 1 } else { 0 }
}

fn cell_rule(mesh: &Mesh<2>, e: u32, order: u8) -> QuadratureRule {
    if mesh.element_type(e) == ElementType::Quad4 {
        quad_rule_01(order)
    } else {
        tri_rule(order)
    }
}

// ─── Main ───────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

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

    // 2. Read the mesh.  1-D meshes take the ND_R1D path (`run_1d`, D901);
    //    2-D meshes take the restricted ND_R2D path below; 3-D (full ND) is a
    //    declared gap.
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
    if let Some(m1) = mfem.mesh1d {
        return run_1d(m1, &args, kappa);
    }
    let base_mesh: Mesh<2> = match mfem.mesh2d {
        Some(m) => m,
        None => {
            eprintln!(
                "mfem_ex31_anisotropic_maxwell: mesh '{}' is not 2-D; this port implements the \
                 1-D (ND_R1D) and 2-D (ND_R2D) restricted H(curl) paths (C++ ex31's 3-D ND \
                 branch is not ported) — exiting with status 3",
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

    // 4. The ND_R2D space (MFEM dof layout 1:1).
    let p = args.order;
    let space = HCurlR2dSpace::new(mesh.clone(), p);
    let n_total = space.n_dofs();
    println!("Number of H(Curl) unknowns: {n_total}");

    // 5. Essential (Dirichlet) DOFs: ALL boundary attributes (PEC).
    let bdr_tags = mesh.unique_boundary_tags();
    let bdr_dofs: Vec<DofId> = space.boundary_dofs(&bdr_tags);

    // 6/8. Assemble curl μ⁻¹ curl (Pk rule 2p−2) + Σ-mass (rule OrderW + 2p)
    //       and the linear form b = (f, φ_i) (rule 2p) with the C++ rules.
    let mut sys_coo = CooMatrix::<f64>::new(n_total, n_total);
    let mut b = vec![0.0_f64; n_total];
    for e in 0..mesh.n_elements() as u32 {
        let cell = Cell::new(&mesh, e, p);
        let nd = cell.n_dofs();
        let el_dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);

        let mut k_elem = vec![0.0_f64; nd * nd];
        let mut phi = vec![0.0_f64; nd * 3];
        let mut curl = vec![0.0_f64; nd * 3];
        // Pass 0: CurlCurlIntegrator (Pk: 2p−2).  Pass 1: VectorFEMassIntegrator
        // with the 3×3 Σ (OrderW + 2p).
        for pass in 0..2 {
            let qord: u8 = if pass == 0 { 2 * p - 2 } else { order_w(&mesh, e) + 2 * p };
            let q = cell_rule(&mesh, e, qord);
            for (qi, xiq) in q.points.iter().enumerate() {
                let xi = [xiq[0], xiq[1]];
                let (j, _) = elem_jac_at(&mesh, e, xi);
                let w = q.weights[qi] * j.det;
                if pass == 0 {
                    cell.eval_curl_phys(&xi, &j, &mut curl);
                    for i in 0..nd {
                        for jj in 0..nd {
                            let mut acc = 0.0;
                            for c in 0..3 {
                                acc += curl[i * 3 + c] * curl[jj * 3 + c];
                            }
                            k_elem[i * nd + jj] += w * acc;
                        }
                    }
                } else {
                    cell.eval_vshape_phys(&xi, &j, &mut phi);
                    for i in 0..nd {
                        // (Σ φ_i) — Σ symmetric
                        let sphi = [
                            SXX * phi[i * 3] + SXY * phi[i * 3 + 1],
                            SXY * phi[i * 3] + SYY * phi[i * 3 + 1] + SYZ * phi[i * 3 + 2],
                            SYZ * phi[i * 3 + 1] + SZZ * phi[i * 3 + 2],
                        ];
                        for jj in 0..nd {
                            let mut acc = 0.0;
                            for c in 0..3 {
                                acc += sphi[c] * phi[jj * 3 + c];
                            }
                            k_elem[i * nd + jj] += w * acc;
                        }
                    }
                }
            }
        }

        // RHS pass (VectorFEDomainLFIntegrator rule 2p).
        let q = cell_rule(&mesh, e, 2 * p);
        for (qi, xiq) in q.points.iter().enumerate() {
            let xi = [xiq[0], xiq[1]];
            let (j, xp) = elem_jac_at(&mesh, e, xi);
            let w = q.weights[qi] * j.det;
            let fv = source_3d(&xp, kappa);
            cell.eval_vshape_phys(&xi, &j, &mut phi);
            for i in 0..nd {
                let mut acc = 0.0;
                for c in 0..3 {
                    acc += phi[i * 3 + c] * fv[c];
                }
                b[el_dofs[i]] += signs[i] * w * acc;
            }
        }

        // Scatter with the orientation signs (D384: drop |v| <= 1e-12
        // structural zeros exactly like MFEM's SparseMatrix::AddSubMatrix).
        for i in 0..nd {
            for jj in 0..nd {
                let v = signs[i] * signs[jj] * k_elem[i * nd + jj];
                if v.abs() > 1e-12 {
                    sys_coo.add(el_dofs[i], el_dofs[jj], v);
                }
            }
        }
    }
    let mut mat = sys_coo.into_csr();

    // 7. Initialize the solution by projecting the exact solution (MFEM
    //    GridFunction::ProjectCoefficient over the ND_R2D space); only the
    //    boundary values are consumed by the DIAG_KEEP elimination.
    let mut x = space.project_with_jac(
        &|x: &[f64]| exact_e(x, kappa),
        &|e, xi| {
            let (j, xp) = elem_jac_at(&mesh, e, xi);
            (j, xp)
        },
    );

    // 9. FormLinearSystem: MFEM 4.10 BilinearForm default DIAG_KEEP policy —
    //    keep the diagonal, zero the rest of the BC rows/cols, adjust the RHS
    //    from the projected boundary values (the D706 serial core entry).
    eliminate_ess_tdofs(&mut mat, &bdr_dofs, &x, &mut b, ElimPolicy::DiagKeep);
    // MFEM FormLinearSystem default `copy_interior = 0`
    // (`X.SetSubVectorComplement(ess_tdof_list, 0.0)`): the PCG initial guess
    // keeps only the essential entries.
    {
        let ess: std::collections::HashSet<usize> =
            bdr_dofs.iter().map(|&d| d as usize).collect();
        for (i, xi) in x.iter_mut().enumerate() {
            if !ess.contains(&i) {
                *xi = 0.0;
            }
        }
    }

    // 10. Solve A X = B with PCG + GSSmoother (C++ PCG(*A, M, B, X, 1, 500,
    //     1e-12, 0.0): SetRelTol(sqrt(1e-12)) = 1e-6, convergence on (B r, r)).
    let cfg = SolverConfig {
        rtol: 1e-6,
        max_iter: 500,
        verbose: true,
        ..Default::default()
    };
    solve_pcg_gssmoother(&mat, &b, &mut x, &cfg).expect("PCG");

    // 13. H(Curl) norm of the error (MFEM ComputeHCurlError: rule 2p+3 for
    //     both the field and the curl part).
    let hcurl_err = compute_hcurl_error(&mesh, &space, &x, p, kappa);
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

// ─── H(Curl) error (MFEM ComputeHCurlError, 2·order + 3 rule) ───────────────

fn compute_hcurl_error(
    mesh: &Mesh<2>,
    space: &HCurlR2dSpace<Mesh<2>>,
    x: &[f64],
    p: u8,
    kappa: f64,
) -> f64 {
    let mut err2 = 0.0_f64;
    for e in 0..mesh.n_elements() as u32 {
        let cell = Cell::new(mesh, e, p);
        let el_dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);
        let q = cell_rule(mesh, e, 2 * p + 3);
        let nd = el_dofs.len();
        let mut phi = vec![0.0_f64; nd * 3];
        let mut curl = vec![0.0_f64; nd * 3];
        for (qi, xiq) in q.points.iter().enumerate() {
            let xi = [xiq[0], xiq[1]];
            let (j, xp) = elem_jac_at(mesh, e, xi);
            let w = q.weights[qi] * j.det;
            let mut eh = [0.0_f64; 3];
            let mut ce = [0.0_f64; 3];
            cell.eval_vshape_phys(&xi, &j, &mut phi);
            cell.eval_curl_phys(&xi, &j, &mut curl);
            for i in 0..nd {
                let c = x[el_dofs[i]] * signs[i];
                for d in 0..3 {
                    eh[d] += c * phi[i * 3 + d];
                    ce[d] += c * curl[i * 3 + d];
                }
            }
            let (ee, ec) = (exact_e(&xp, kappa), exact_curl(&xp, kappa));
            for d in 0..3 {
                let d0 = eh[d] - ee[d]; err2 += w * d0 * d0;
                let dc = ce[d] - ec[d]; err2 += w * dc * dc;
            }
        }
    }
    err2.sqrt()
}

// ─── dim == 1 branch (ND_R1D_FECollection, D901) ────────────────────────────

/// `E_exact` for `dim == 1` (all three components vary along the segment).
fn exact_e_1d(x: f64, kappa: f64) -> [f64; 3] {
    [
        A0 * (kappa * x).sin(),
        A1 * (kappa * x + PHI1).sin(),
        A2 * (kappa * x + PHI2).sin(),
    ]
}

/// `CurlE_exact` for `dim == 1` (curl acts on the y/z derivatives).
fn exact_curl_1d(x: f64, kappa: f64) -> [f64; 3] {
    [
        0.0,
        -A2 * (kappa * x + PHI2).cos() * kappa,
        A1 * (kappa * x + PHI1).cos() * kappa,
    ]
}

/// `f_exact` for `dim == 1` — the **manufactured source** −ΔE − κ²E + ΣE
/// (MFEM ex31.cpp `f_exact`, the `dim == 1` branch), NOT the field itself:
/// the RHS linear form assembles `(f_exact, φ_i)`.
fn exact_source_1d(x: f64, kappa: f64) -> [f64; 3] {
    let s0 = (kappa * x).sin();
    let s4 = (kappa * x + PHI1).sin();
    let s9 = (kappa * x + PHI2).sin();
    let k2 = kappa * kappa;
    [
        2.2 * s0 + 1.2 * SQRT_2.recip() * s4,
        1.2 * (2.0 + k2) * s4 + SQRT_2.recip() * (A0 * s0 + A2 * s9),
        A2 * (2.0 + k2) * s9 + 1.2 * SQRT_2.recip() * s4,
    ]
}

/// Gauss–Legendre segment rule with MFEM's point count `N = (order+2)/2`
/// (`IntRules.Get(SEGMENT, order)`), returned on the reference `[0, 1]`.
fn seg_rule(order: u8) -> (Vec<f64>, Vec<f64>) {
    gauss_legendre_01((order as usize + 2) / 2)
}

fn run_1d(mut mesh: Mesh<1>, args: &Args, kappa: f64) {
    // 3. Uniform refinements — MFEM `Mesh::LocalRefinement` `Dim == 1`:
    //    midpoint split, new vertex `cnv + j` (element-major), the parent
    //    keeps `[v0, mid]` and `[mid, v1]` is appended.
    for _ in 0..args.ref_levels {
        mesh = refine_uniform_1d(&mesh);
    }
    let p = args.order;
    let space = HCurlR1dSpace::new(mesh.clone(), p);
    let n_total = space.n_dofs();
    println!("Number of H(Curl) unknowns: {n_total}");

    // 5. Essential dofs: all boundary POINTs (PEC).
    let bdr_tags = mesh.unique_boundary_tags();
    let bdr_dofs: Vec<DofId> = space.boundary_dofs(&bdr_tags);

    // Vertex coordinates (straight 1-D segments: J00 = x1 − x0, x = x0 + ξ·J00).
    let vx: Vec<f64> = (0..mesh.n_nodes() as u32)
        .map(|v| mesh.coords_of(v)[0])
        .collect();
    let seg = space.segment_element();
    let nd = seg.n_dofs();

    // 6/8. Assemble curl μ⁻¹ curl (rule 2p−2) + Σ-mass (rule OrderW + 2p) and
    //      b = (f, φ_i) (rule 2p) — the segment instantiations of the same
    //      integrator defaults the 2-D branch pins.
    let mut sys_coo = CooMatrix::<f64>::new(n_total, n_total);
    let mut b = vec![0.0_f64; n_total];
    let mut curl = vec![0.0_f64; nd * 3];
    let mut phi = vec![0.0_f64; nd * 3];
    for e in 0..mesh.n_elements() as u32 {
        let v = mesh.element_nodes(e);
        let (x0, x1) = (vx[v[0] as usize], vx[v[1] as usize]);
        let j00 = x1 - x0;
        let el_dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();

        let mut k_elem = vec![0.0_f64; nd * nd];
        // Pass 0: CurlCurlIntegrator (Pk rule 2p−2).  CalcPhysCurlShape:
        // reference curl scaled by 1/J (whole array).
        let (xs, ws) = seg_rule(2 * p - 2);
        for (qi, &xi) in xs.iter().enumerate() {
            let w = ws[qi] * j00;
            seg.eval_curl_ref(xi, &mut curl);
            for c in 0..nd * 3 {
                curl[c] /= j00;
            }
            for i in 0..nd {
                for j in 0..nd {
                    let mut acc = 0.0;
                    for c in 0..3 {
                        acc += curl[i * 3 + c] * curl[j * 3 + c];
                    }
                    k_elem[i * nd + j] += w * acc;
                }
            }
        }
        // Pass 1: VectorFEMassIntegrator with the 3×3 Σ (rule OrderW + 2p,
        // OrderW = |J00| for a straight segment).  CalcVShape(Trans): the
        // x-directed rows scale by J⁻¹ = 1/J00 (covariant), y/z rows as-is.
        let (xs, ws) = seg_rule(1 + 2 * p);
        for (qi, &xi) in xs.iter().enumerate() {
            let w = ws[qi] * j00;
            seg.eval_vshape_ref(xi, &mut phi);
            for i in 0..nd {
                phi[i * 3] /= j00;
            }
            for i in 0..nd {
                let sphi = [
                    SXX * phi[i * 3] + SXY * phi[i * 3 + 1],
                    SXY * phi[i * 3] + SYY * phi[i * 3 + 1] + SYZ * phi[i * 3 + 2],
                    SYZ * phi[i * 3 + 1] + SZZ * phi[i * 3 + 2],
                ];
                for j in 0..nd {
                    let mut acc = 0.0;
                    for c in 0..3 {
                        acc += sphi[c] * phi[j * 3 + c];
                    }
                    k_elem[i * nd + j] += w * acc;
                }
            }
        }

        // RHS pass (VectorFEDomainLFIntegrator, rule 2p) with the
        // manufactured source f_exact.
        let (xs, ws) = seg_rule(2 * p);
        for (qi, &xi) in xs.iter().enumerate() {
            let xp = x0 + xi * j00;
            let w = ws[qi] * j00;
            seg.eval_vshape_ref(xi, &mut phi);
            for i in 0..nd {
                phi[i * 3] /= j00;
            }
            let fv = exact_source_1d(xp, kappa);
            for i in 0..nd {
                let mut acc = 0.0;
                for c in 0..3 {
                    acc += phi[i * 3 + c] * fv[c];
                }
                b[el_dofs[i]] += w * acc;
            }
        }

        // Scatter (no signed dofs in 1-D; D384 drop |v| <= 1e-12 like MFEM
        // AddSubMatrix skip_zeros).
        for i in 0..nd {
            for j in 0..nd {
                let v = k_elem[i * nd + j];
                if v.abs() > 1e-12 {
                    sys_coo.add(el_dofs[i], el_dofs[j], v);
                }
            }
        }
    }
    let mut mat = sys_coo.into_csr();

    // 7. Project the exact solution (ND_R1D_SegmentElement::Project:
    //    x-dofs get J00·E_x, y/z-dofs plain point values).
    let mut x = space.project(
        &|xp| exact_e_1d(xp[0], kappa),
        &|e, xi| {
            let v = mesh.element_nodes(e);
            let (x0, x1) = (vx[v[0] as usize], vx[v[1] as usize]);
            (x1 - x0, [x0 + xi * (x1 - x0)])
        },
    );

    // 9. FormLinearSystem: DIAG_KEEP + copy_interior = 0 (MFEM defaults).
    eliminate_ess_tdofs(&mut mat, &bdr_dofs, &x, &mut b, ElimPolicy::DiagKeep);
    {
        let ess: std::collections::HashSet<usize> =
            bdr_dofs.iter().map(|&d| d as usize).collect();
        for (i, xi) in x.iter_mut().enumerate() {
            if !ess.contains(&i) {
                *xi = 0.0;
            }
        }
    }

    // 10. Solve with PCG + GSSmoother (PCG(*A, M, B, X, 1, 500, 1e-12, 0.0)).
    let cfg = SolverConfig {
        rtol: 1e-6,
        max_iter: 500,
        verbose: true,
        ..Default::default()
    };
    solve_pcg_gssmoother(&mat, &b, &mut x, &cfg).expect("PCG");

    // 13. H(Curl) norm of the error (rule 2p + 3, same as the 2-D branch).
    let mut err2 = 0.0_f64;
    let q = seg_rule(2 * p + 3);
    for e in 0..mesh.n_elements() as u32 {
        let v = mesh.element_nodes(e);
        let (x0, x1) = (vx[v[0] as usize], vx[v[1] as usize]);
        let j00 = x1 - x0;
        let el_dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        for (qi, &xi) in q.0.iter().enumerate() {
            let xp = x0 + xi * j00;
            let w = q.1[qi] * j00;
            seg.eval_vshape_ref(xi, &mut phi);
            for i in 0..nd {
                phi[i * 3] /= j00;
            }
            seg.eval_curl_ref(xi, &mut curl);
            for c in 0..nd * 3 {
                curl[c] /= j00;
            }
            let mut eh = [0.0_f64; 3];
            let mut ce = [0.0_f64; 3];
            for i in 0..nd {
                let c = x[el_dofs[i]];
                for d in 0..3 {
                    eh[d] += c * phi[i * 3 + d];
                    ce[d] += c * curl[i * 3 + d];
                }
            }
            let (ee, ec) = (exact_e_1d(xp, kappa), exact_curl_1d(xp, kappa));
            for d in 0..3 {
                let d0 = eh[d] - ee[d];
                err2 += w * d0 * d0;
                let dc = ce[d] - ec[d];
                err2 += w * dc * dc;
            }
        }
    }
    println!("\n|| E_h - E ||_{{H(Curl)}} = {}\n", fmt_g(err2.sqrt()));

    // 14. Save the solution (precision 8).  `refined.mesh` is not written:
    //     fem_io has no Mesh<1> mesh-structure writer (D946's io lane), only
    //     the `nodes`-section writer; the stdout comparison is unaffected.
    let sol_f = File::create("sol.gf").expect("cannot create sol.gf");
    let mut w = BufWriter::new(sol_f);
    for &v in &x {
        writeln!(w, "{:.8e}", v).expect("sol write failed");
    }
}
