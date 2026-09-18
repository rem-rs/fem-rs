//! # LOR Solvers Miniapp — port of MFEM `miniapps/solvers/lor_solvers.cpp`
//!
//! Definite Helmholtz / definite Maxwell / grad-div manufactured problems
//! (`lor_mms.hpp`) discretised on `H1`, `H(curl)`, `H(div)` or `L2` spaces and
//! **preconditioned by the low-order refined operator**: `P1` (or `ND1`/`RT0`)
//! on the Gauss-Lobatto refined mesh, with the LOR dof permutation
//! `M⁻¹ = Π · A_LOR⁻¹ · Πᵀ`.
//!
//! Round 48: this file used to be a **declarative stub** (D140) that assembled
//! the H¹ mass and diffusion matrices, **discarded the mass matrix**, solved the
//! diffusion-only system with plain PCG+Jacobi and printed
//! `LOR solvers complete` with exit code 0 — i.e. it claimed to be the LOR
//! miniapp while computing the wrong operator, and refused every input.  The
//! LOR stack it needed has since landed (`fem_space::lor::{LorH1,LorNd,LorRt}`,
//! `fem_assembly::lor_factory`, `fem_solver::lor`), so it is now a real driver.
//!
//! 1:1 with the C++ driver:
//!
//! | C++ | here |
//! |---|---|
//! | `Mesh mesh(mesh_file, 1, 1)` + `-r` `UniformRefinement()` | `read_mfem_file` + `refine_uniform` |
//! | `H1_FECollection(order, dim, GaussLobatto)` | `H1Space::new(mesh, order)` |
//! | `ND_FECollection(order, dim, GaussLobatto, IntegratedGLL)` | `HCurlSpace::new(mesh, order)` |
//! | `RT_FECollection(order-1, dim, GaussLobatto, IntegratedGLL)` | `HDivSpace::new(mesh, order-1)` |
//! | `MassIntegrator + DiffusionIntegrator` (H1/L2) | `MassIntegrator` + `DiffusionIntegrator` |
//! | `VectorFEMassIntegrator (+ CurlCurl/DivDiv)` (ND/RT) | `VectorMassIntegrator` + `CurlCurlIntegrator`/`DivDivIntegrator` |
//! | `fes.GetBoundaryTrueDofs(ess_dofs)` | `boundary_dofs` / `boundary_dofs_hcurl` / `boundary_dofs_hdiv_quad_rt` |
//! | `x.ProjectCoefficient(u_coeff / u_vec_coeff)` | `FESpace::interpolate` |
//! | `FormLinearSystem` (DIAG_KEEP) | `apply_dirichlet` (`DIAG_KEEP`) |
//! | `LORSolver<GSSmoother>` (no SuiteSparse) | `build_lor_amg_h1` (H1) / `build_lor_sgs_nd_quad` / `build_lor_sgs_rt_quad` |
//! | `CGSolver` (rtol 1e-12, 500 iters) | `solve_pcg_lor_amg` / `solve_pcg_precond` |
//! | `x.ComputeL2Error(u)` | `GridFunction::compute_l2_error` / `compute_l2_error_hcurl/hdiv` |
//!
//! ## Measured C++ reference (MFEM 4.10, `$HOME/mfem410_ser`, no SuiteSparse)
//!
//! ```text
//! lor_solvers -m data/star.mesh        -fe h -> Number of DOFs:  781  L2 error: 0.000395471
//! lor_solvers -m data/inline-quad.mesh -fe h -> Number of DOFs:  625  L2 error: 5.56315e-06
//! lor_solvers -m data/inline-quad.mesh -fe n -> Number of DOFs: 1200  L2 error: 0.000134744
//! lor_solvers -m data/inline-quad.mesh -fe r -> Number of DOFs: 1200  L2 error: 0.000134744
//! ```
//!
//! (`star.mesh` is a **20-element quadrilateral** mesh — its `elements` block uses
//! geometry type 3 — so it is the quad LOR path, not the simplex one.)
//!
//! ## Verified: `-fe h`
//!
//! `Number of DOFs` and `L2 error` are **byte-identical to the C++** on both
//! meshes above (`781` / `0.000395471` and `625` / `5.56315e-06`).
//!
//! Two quadrature details had to be pinned (both measured, see the helpers):
//!
//! * the **load** rule is `2·order + 1`, not `2·order + 2` — `f` is
//!   trigonometric, so the RHS rule shifts the discrete solution and only
//!   `2·order + 1` reproduces the C++ value;
//! * the **`L2 error`** rule is MFEM's `ComputeL2Error` default
//!   `2·order + 3` (`fem/gridfunc.cpp:3410`).
//!
//! The CG **iteration count** is *not* part of the acceptance: MFEM built
//! without SuiteSparse uses `LORSolver<GSSmoother>` (one GS sweep on `A_LOR`),
//! while fem-rs applies AMG to `A_LOR` through `build_lor_amg_h1`.  Both reach
//! the same solution (`star.mesh`: 26 fem-rs iterations against 58 C++
//! iterations, `L2 error` identical to the printed digit).
//!
//! ## Not ported (honest gaps, refused with a message, never faked)
//!
//! * **`-fe n` / `-fe r` on a quad mesh (D368)**: the D367 capability landed —
//!   `build_lor_sgs_nd_quad` / `build_lor_sgs_rt_quad` take the HO essential
//!   dofs and eliminate them on `A_LOR` exactly like MFEM's
//!   `LORSolver(a_ho, ess_tdof_list)` (serial `BatchedLORAssembly::Assemble`
//!   finishes with `EliminateBC(ess_dofs, DIAG_KEEP)`, `lor_batched.cpp:726-734`;
//!   the discrete gradient of an AMS inner needs no ess treatment,
//!   `lor_ams.cpp`), and the inner is one symmetric GS sweep = MFEM's
//!   `LORSolver<GSSmoother>`.  The H(div) essential list is also complete now
//!   ([`boundary_dofs_hdiv_quad_rt`]: 3 dofs per boundary edge like MFEM's
//!   `GetBoundaryTrueDofs`, where `boundary_dofs_hdiv` exposed only one).
//!   What still fails is the **HO space basis** (D69): fem-rs's quad ND/RT
//!   elements (the legacy `QuadNDk` / `QuadRTk` behind `vec_ref_elem`) are not
//!   faithful `ND_FECollection` / `RT_FECollection` `(GaussLobatto,
//!   IntegratedGLL)` ports, and the LOR transfer — built for MFEM's dof
//!   functionals — then mismatches the HO operator it must precondition.
//!   Measured on `data/inline-quad.mesh -o 3` with this very driver: ND PCG
//!   does not converge (true relative residual `2.1e-02` after 500 iterations
//!   against the C++'s 279); RT converges in 281 iterations (C++ 268 with the
//!   same GS algorithm) but prints `L2 error: 0.000134747` against the C++'s
//!   `0.000134744`.  Porting the faithful quad ND/RT spaces is D368/D69.
//! * **`-fe n` / `-fe r` on a simplex mesh**: fem-rs's ND/RT LOR
//!   discretisations are tensor-product only (`LorNd::<2>::new_quad` /
//!   `LorRt::<2>::new_quad` require every element to be `Quad4`), while MFEM's
//!   `LORBase` only sets `ir_el = NULL` for a non-tensor geometry and still
//!   builds the low-order space (`lor.cpp:338-355`).
//! * **`-fe l`** (L² DG): MFEM adds `DGDiffusionIntegrator(-1, kappa)` as an
//!   interior/boundary *face* integrator on the bilinear/linear forms.
//!   fem-rs's DG machinery is a separate assembler API (`dg::DgAssembler`), not
//!   a `BilinearForm` integrator, so the DG leg has no 1:1 expression here.
//!
//! ## `-fe h` on a simplex mesh: no C++ oracle exists
//!
//! `lor_solvers -m data/inline-tri.mesh -fe h` **aborts inside MFEM**:
//! `MFEM_VERIFY(mode == DofToQuad::FULL)` (`fem/fe/fe_base.cpp:377`) — the
//! partial-assembly path that `lor_solvers.cpp:159` selects for H¹ needs a
//! tensor `DofToQuad`, which the triangle element does not provide.  fem-rs
//! assembles fully, so it runs there and reports `Number of DOFs: 625` (which
//! does agree with the C++'s `625` for `inline-tri` before the abort).  That
//! path is therefore *ahead of* the C++, not verified against it.

use std::f64::consts::PI;

use fem_assembly::coefficient::FnVectorCoeff;
use fem_assembly::postproc::grid_function::{
    compute_l2_error_hcurl, compute_l2_error_hdiv, GridFunction,
};
use fem_assembly::standard::{
    CurlCurlIntegrator, DiffusionIntegrator, DivDivIntegrator, MassIntegrator,
    VectorDomainLFIntegrator, VectorMassIntegrator,
};
use fem_assembly::{build_lor_amg_h1, Assembler, VectorAssembler};
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::lor::solve_pcg_lor_amg;
use fem_solver::{fmt_g, solve_pcg_precond, SolverConfig};
use fem_space::constraints::{apply_dirichlet, boundary_dofs, boundary_dofs_hcurl};
use fem_space::{EdgeKey, FESpace, HCurlSpace, HDivSpace, H1Space};

// ─── lor_mms.hpp ────────────────────────────────────────────────────────────

/// `lor_mms.hpp::u` — `sin(πx) sin(πy)` (2-D) / `sin(πx) sin(πy) sin(πz)`.
fn u_exact(x: &[f64]) -> f64 {
    let s = (PI * x[0]).sin() * (PI * x[1]).sin();
    if x.len() == 2 {
        s
    } else {
        s * (PI * x[2]).sin()
    }
}

/// `lor_mms.hpp::f(1.0)` — the definite Helmholtz load `u − Δu = f`.
fn f_rhs(x: &[f64]) -> f64 {
    let u = u_exact(x);
    let lap = if x.len() == 2 { 2.0 } else { 3.0 } * PI * PI * u;
    u + lap
}

/// `lor_mms.hpp::u_vec` — the exact vector field (a divergence-free
/// `(cos·sin, sin·cos)` field).
fn u_vec_exact(x: &[f64]) -> Vec<f64> {
    let (cx, sx) = ((PI * x[0]).cos(), (PI * x[0]).sin());
    let (cy, sy) = ((PI * x[1]).cos(), (PI * x[1]).sin());
    if x.len() == 2 {
        vec![cx * sy, sx * cy]
    } else {
        let (cz, sz) = ((PI * x[2]).cos(), (PI * x[2]).sin());
        vec![cx * sy * sz, sx * cy * sz, sx * sy * cz]
    }
}

/// `lor_mms.hpp::f_vec(grad_div_problem)`.
///
/// `grad_div_problem = false` (the H(curl) leg) is the plain field; `true` (the
/// H(div) leg) scales the grad-div load by `1 + dim·π²`.
fn f_vec_rhs(x: &[f64], grad_div_problem: bool) -> Vec<f64> {
    let (cx, sx) = ((PI * x[0]).cos(), (PI * x[0]).sin());
    let (cy, sy) = ((PI * x[1]).cos(), (PI * x[1]).sin());
    if x.len() == 2 {
        if grad_div_problem {
            let a = 1.0 + 2.0 * PI * PI;
            vec![a * cx * sy, a * cy * sx]
        } else {
            vec![cx * sy, sx * cy]
        }
    } else {
        let (cz, sz) = ((PI * x[2]).cos(), (PI * x[2]).sin());
        if grad_div_problem {
            let a = 1.0 + 3.0 * PI * PI;
            vec![a * cx * sy * sz, a * cy * sx * sz, a * cz * sx * sy]
        } else {
            vec![cx * sy * sz, sx * cy * sz, sx * sy * cz]
        }
    }
}

// ─── driver ─────────────────────────────────────────────────────────────────

struct Args {
    mesh_file: String,
    ref_levels: usize,
    order: u8,
    fe: String,
}

/// What the run produced, printed in MFEM's order.
struct Outcome {
    n_dofs: usize,
    iterations: usize,
    final_residual: f64,
    l2_err: f64,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut a = Args {
        mesh_file: "data/star.mesh".to_string(),
        ref_levels: 1,
        order: 3,
        fe: "h".to_string(),
    };
    let mut i = 0;
    while i < argv.len() {
        let arg = argv[i].as_str();
        let value = |i: &mut usize, arg: &str| -> String {
            *i += 1;
            argv.get(*i).cloned().unwrap_or_else(|| {
                eprintln!("lor_solvers: missing value for {arg}");
                std::process::exit(2);
            })
        };
        match arg {
            "-m" | "--mesh" => a.mesh_file = value(&mut i, arg),
            "-r" | "--refine" => a.ref_levels = value(&mut i, arg).parse().unwrap_or(1),
            "-o" | "--order" => a.order = value(&mut i, arg).parse().unwrap_or(3),
            "-fe" | "--fe-type" => a.fe = value(&mut i, arg),
            "-vis" | "--visualization" | "-no-vis" | "--no-visualization" => {}
            "-d" | "--device" => {
                // C++ `Device device(device_config)`; only the CPU path exists.
                let dev = value(&mut i, arg);
                if dev != "cpu" {
                    eprintln!(
                        "lor_solvers: device {dev:?} is not ported (CPU only) — exiting with \
                         status 3"
                    );
                    std::process::exit(3);
                }
            }
            other => {
                eprintln!("lor_solvers (Rust port): Unrecognized option: {other}");
                std::process::exit(3);
            }
        }
        i += 1;
    }
    a
}

fn main() {
    let args = parse_args();
    // C++ `MFEM_ABORT("Bad FE type...")` — there is no `-vis` socket in fem-rs,
    // so nothing is emitted for it either way.
    let leg = match args.fe.as_str() {
        "h" => Leg::H1,
        "n" => Leg::HCurl,
        "r" => Leg::HDiv,
        "l" => Leg::L2,
        other => {
            eprintln!("Bad FE type '{other}'. Must be 'h', 'n', 'r', or 'l'.");
            std::process::exit(3);
        }
    };

    let mfem = read_mfem_file(&args.mesh_file).unwrap_or_else(|e| {
        eprintln!("lor_solvers: failed to read mesh {}: {e}", args.mesh_file);
        std::process::exit(3);
    });
    let mut mesh: Mesh<2> = mfem.mesh2d.unwrap_or_else(|| {
        eprintln!(
            "lor_solvers: {} is not a 2-D mesh (the LOR stack here is the 2-D one)",
            args.mesh_file
        );
        std::process::exit(3);
    });
    out_of_scope_guards(&mesh, leg, args.order);
    for _ in 0..args.ref_levels {
        mesh = refine_uniform(&mesh);
    }

    let order = args.order;
    let out = match leg {
        Leg::H1 => run_h1(&mesh, order),
        Leg::HCurl => run_nd(&mesh, order),
        Leg::HDiv => run_rt(&mesh, order),
        Leg::L2 => unreachable!("guarded above"),
    };

    println!("Number of DOFs: {}", out.n_dofs);
    println!(
        "CG ({} preconditioned iterations, true relative residual {:.6e})",
        out.iterations, out.final_residual
    );
    println!("L2 error: {}", fmt_g(out.l2_err));
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Leg {
    H1,
    HCurl,
    HDiv,
    L2,
}

/// Refusals for combinations fem-rs genuinely cannot express at 1:1 fidelity
/// (see the module doc).  Each names the concrete missing capability and the
/// **measured** symptom instead of degrading to a wrong number.
fn out_of_scope_guards(mesh: &Mesh<2>, leg: Leg, order: u8) {
    if leg == Leg::L2 {
        eprintln!(
            "lor_solvers: -fe l (L2 DG) is not ported — MFEM adds \
             DGDiffusionIntegrator(-1, kappa) as an interior *and* boundary FACE integrator on \
             the bilinear and linear forms (lor_solvers.cpp:153-171); fem-rs's DG machinery is \
             the separate `fem_assembly::dg::DgAssembler` API, not a BilinearForm integrator, so \
             there is no 1:1 expression for it. Exiting with status 3."
        );
        std::process::exit(3);
    }
    if matches!(leg, Leg::HCurl | Leg::HDiv) && order < 2 {
        eprintln!(
            "lor_solvers: -fe n/r at -o {order} — the low-order space needs no LOR (MFEM's \
             LORSolver is only built for order >= 2 here; RT uses order-1). Exiting with status 3."
        );
        std::process::exit(3);
    }
    let et = mesh.element_type(0);
    let quad = et == fem_mesh::element_type::ElementType::Quad4;
    if leg == Leg::H1 {
        if quad || order <= 3 {
            return;
        }
        eprintln!(
            "lor_solvers: -fe h -o {order} on a {et:?} mesh is not ported — the fem-rs H¹ LOR \
             discretisation covers Quad4 (any order) and Tri3 (order <= 3). Exiting with status 3."
        );
        std::process::exit(3);
    }
    if !quad {
        eprintln!(
            "lor_solvers: -fe {} on a {et:?} mesh is not ported — fem-rs's ND/RT LOR \
             discretisations are tensor-product only (`LorNd::<2>::new_quad` / \
             `LorRt::<2>::new_quad` require every element to be Quad4), while MFEM's \
             `LORBase` only sets `ir_el = NULL` for a non-tensor geometry and still builds the \
             low-order space (lor.cpp:338-355). Re-run with a quad mesh, or use -fe h. \
             Exiting with status 3.",
            if leg == Leg::HCurl { "n" } else { "r" }
        );
        std::process::exit(3);
    }
    // Quad ND/RT: the D367 preconditioner fix is in (`build_lor_sgs_*` build on
    // the eliminated LOR matrix, MFEM `EliminateBC` DIAG_KEEP semantics, plus
    // the symmetric-GS inner of `LORSolver<GSSmoother>`), and the H(div)
    // essential list is now complete (`boundary_dofs_hdiv_quad_rt`, 3 dofs per
    // boundary edge = MFEM `GetBoundaryTrueDofs`).  What remains is **D368**:
    // the high-order quad spaces are not field-faithful to MFEM's
    // `(GaussLobatto, IntegratedGLL)` collections (D69), so the discrete
    // solution differs and the legs do not meet the byte-identical bar.
    // Measured on `data/inline-quad.mesh -o 3` with this very driver (the
    // refusal below is the only thing that normally stops it):
    // * `-fe n`: PCG does not converge — 500 iterations, true relative residual
    //   2.1101872802611644e-02 (C++: 279 iterations).
    // * `-fe r`: PCG converges in 281 iterations (C++: 268, same
    //   `LORSolver<GSSmoother>` algorithm), true relative residual 2.090907e-12,
    //   but `L2 error: 0.000134747` against the C++'s `0.000134744` — the
    //   remaining gap is the HO space, not the solver (quadrature orders were
    //   verified identical: the `2*el.GetOrder()` VectorFE load rule and the
    //   per-integrator order-6 rules all select the same 4×4 Gauss product).
    let n = if leg == Leg::HCurl { "n" } else { "r" };
    eprintln!(
        "lor_solvers: -fe {n} on a quad mesh does not meet the 1:1 fidelity bar (D368). The D367 \
         fix landed: `fem_assembly::lor_factory::build_lor_sgs_nd_quad` / `build_lor_sgs_rt_quad` \
         now take the HO essential dofs and eliminate them on `A_LOR` with MFEM's \
         `EliminateBC(ess_dofs, DIAG_KEEP)` semantics (`lor_batched.cpp:726-734`), and the inner \
         is one symmetric Gauss-Seidel sweep = MFEM's `LORSolver<GSSmoother>`. The remaining gap \
         is the HO space basis: fem-rs's quad ND/RT elements (`vec_ref_elem` legacy `QuadNDk` / \
         `QuadRTk`, D69) are not faithful ports of MFEM's \
         `ND_FECollection(o, 2, GaussLobatto, IntegratedGLL)` / \
         `RT_FECollection(o-1, 2, GaussLobatto, IntegratedGLL)`, so the LOR transfer (built for \
         MFEM's dof functionals) does not spectrally match the HO operator. Measured \
         (data/inline-quad.mesh, -o 3): {msg}. C++ reference: ND 279 iterations, RT 268 \
         iterations, both `L2 error: 0.000134744`. Porting the faithful `QuadND`/`QuadRT` spaces \
         is debt D368/D69 (large, separate). Exiting with status 3.",
        msg = if leg == Leg::HCurl {
            "PCG does not converge: 500 iterations, true relative residual 2.1101872802611644e-02"
        } else {
            "PCG converges in 281 iterations (true relative residual 2.090907e-12) but prints \
             `L2 error: 0.000134747`, 3 ulps of the 7th digit away from the C++"
        }
    );
    std::process::exit(3);
}

/// Quadrature order for the bilinear forms.
///
/// The value is **not** load-bearing: measured on `data/star.mesh -fe h`, every
/// order in `2·order .. 2·order + 3` gives the same `L2 error`
/// (`0.000395471`), i.e. the element matrices are integrated exactly there.
fn bilinear_order(order: u8) -> u8 {
    (2 * order as u16 + 1) as u8
}

/// MFEM's `GridFunction::ComputeL2Error` default rule (`fem/gridfunc.cpp:3410`:
/// `intorder = 2*fe->GetOrder() + 3`).
fn l2_rule_order(space_order: u8) -> u8 {
    (2 * space_order as u16 + 3).min(255) as u8
}

/// The rule MFEM's `DomainLFIntegrator` uses through `LinearForm` for H¹ —
/// `2·order + 1` (`b.UseFastAssembly(true)` on the H1 leg, `lor_solvers.cpp:171`).
///
/// This one **is** load-bearing: `f` is trigonometric, so no polynomial rule is
/// exact and the RHS quadrature shifts the discrete solution.  Measured on
/// `data/star.mesh -fe h` (C++ `L2 error: 0.000395471`):
///
/// | load rule | `L2 error` |
/// |---|---|
/// | `2·order + 1` = 7 | **0.000395471** (= C++) |
/// | `2·order + 2` = 8 | 0.000395475 |
/// | `2·order + 3` = 9 | 0.000395475 |
fn load_order(order: u8) -> u8 {
    (2 * order as u16 + 1) as u8
}

/// The rule MFEM's `VectorFEDomainLFIntegrator` (the H(curl)/H(div) load)
/// uses: `2*el.GetOrder()` — **not** the scalar `DomainLFIntegrator`'s
/// `2*order + 1` that the H¹ leg pins (`fem/lininteg.cpp:481`, the
/// `int intorder = 2*el.GetOrder();` branch).  `el.GetOrder()` is the element
/// order: `order` for ND(`order`), `(order-1)+1 = order` for the
/// `RT_FECollection(order-1, …)` element (`RT_QuadrilateralElement(p)` reports
/// order `p+1`, `fem/fe/fe_rt.cpp:30`).
fn vector_load_order(order: u8) -> u8 {
    (2 * order as u16).min(255) as u8
}

/// `x.ProjectCoefficient(u_coeff)`, then `FormLinearSystem` with DIAG_KEEP.
///
/// Returns `(a_system, b_system, x0)` where `x0` is the interpolant (MFEM's
/// `copy_interior = 1` default keeps it as the initial guess).
fn form_system(a: CsrMatrix<f64>, b: Vec<f64>, ess: &[u32], x0: &[f64]) -> (CsrMatrix<f64>, Vec<f64>, Vec<f64>) {
    let mut a = a;
    let mut b = b;
    let vals: Vec<f64> = ess.iter().map(|&d| x0[d as usize]).collect();
    apply_dirichlet(&mut a, &mut b, ess, &vals);
    (a, b, x0.to_vec())
}

/// `fes.GetBoundaryTrueDofs(ess_dofs)` for the H(div) leg on a quad mesh.
///
/// `fem_space::constraints::boundary_dofs_hdiv` exposes **one** dof per
/// boundary edge (`HDivSpace::edge_face_dof` returns only the first), but the
/// LOR-compatible `RT_FECollection(order-1, 2, GaussLobatto, IntegratedGLL)`
/// element (MFEM `RT_QuadrilateralElement(p)`, `fem/fe/fe_rt.cpp:26`) carries
/// `p + 1` dofs per edge — for the `-o 3` leg (`RT_Quad(2)`) three per edge,
/// so a per-edge dof list would leave two thirds of the boundary normal trace
/// unconstrained.  This walks the mesh boundary and collects the *full* edge
/// dof block of `HDivSpace::element_dofs`: local edge `i` occupies the
/// `f` consecutive slots `f·i .. f·i+f−1` with `f = p+1`, where
/// `2·f·(f+1) = dofs per element` for the `RT_QuadrilateralElement` layout
/// (orientation-reversed edges enumerate the block backwards, which does not
/// change the dof set).
fn boundary_dofs_hdiv_quad_rt(mesh: &Mesh<2>, space: &HDivSpace<Mesh<2>>, tags: &[i32]) -> Vec<u32> {
    let dpe = space.element_dofs(0).len();
    let f = (((1.0 + 2.0 * dpe as f64).sqrt() - 1.0) / 2.0).round() as usize;
    assert!(
        2 * f * (f + 1) == dpe && f >= 1,
        "unexpected quad RT dof layout: {dpe} dofs per element"
    );

    // Edge key → (element, local edge index).
    let mut edge_elem: std::collections::HashMap<EdgeKey, (u32, usize)> = Default::default();
    for e in 0..mesh.n_elements() as u32 {
        let verts = mesh.element_nodes(e);
        for (i, (a, b)) in [(0usize, 1usize), (1, 2), (2, 3), (3, 0)].iter().enumerate() {
            edge_elem.insert(EdgeKey::new(verts[*a], verts[*b]), (e, i));
        }
    }

    let mut out: Vec<u32> = Vec::new();
    for bf in 0..mesh.n_boundary_faces() as u32 {
        if !tags.contains(&mesh.face_tag(bf)) {
            continue;
        }
        let nodes = mesh.face_nodes(bf);
        let key = EdgeKey::new(nodes[0], nodes[1]);
        if let Some(&(e, li)) = edge_elem.get(&key) {
            let dofs = space.element_dofs(e);
            out.extend_from_slice(&dofs[li * f..li * f + f]);
        }
    }
    out
}

/// `-fe h`: `M + K` on `H1(order)`, LOR-AMG preconditioned CG.
fn run_h1(mesh: &Mesh<2>, order: u8) -> Outcome {
    let space = H1Space::new(mesh.clone(), order);
    let n_dofs = space.n_dofs();
    let qo = bilinear_order(order);
    let mass = MassIntegrator { rho: 1.0 };
    let diff = DiffusionIntegrator { kappa: 1.0 };
    let a = Assembler::assemble_bilinear(&space, &[&mass, &diff], qo);
    let src = fem_assembly::standard::DomainSourceIntegrator::new(f_rhs);
    let b = Assembler::assemble_linear(&space, &[&src], load_order(order));

    let tags = mesh.unique_boundary_tags();
    let ess = boundary_dofs(mesh, space.dof_manager(), &tags);
    let x0 = space.interpolate(&u_exact).as_slice().to_vec();

    let (a, b, mut x) = form_system(a, b.as_slice().to_vec(), &ess, &x0);

    let lor = build_lor_amg_h1(&space, &a, None)
        .unwrap_or_else(|e| panic!("LOR-AMG build failed (H1 P{order}): {e}"));
    let cfg = SolverConfig {
        rtol: 1e-12,
        atol: 0.0,
        max_iter: 500,
        verbose: false,
        ..SolverConfig::default()
    };
    let res = solve_pcg_lor_amg(&a, &b, &mut x, &lor, &cfg).expect("LOR-AMG PCG failed");

    let gf = GridFunction::new(&space, x);
    let l2_err = gf.compute_l2_error(&u_exact, l2_rule_order(order));
    Outcome {
        n_dofs,
        iterations: res.iterations,
        final_residual: res.final_residual,
        l2_err,
    }
}

/// `-fe n`: `M + curl-curl` on `H(curl, order)`, LOR-AMS preconditioned CG.
fn run_nd(mesh: &Mesh<2>, order: u8) -> Outcome {
    let space = HCurlSpace::new(mesh.clone(), order);
    let n_dofs = space.n_dofs();
    let qo = bilinear_order(order);
    let mass = VectorMassIntegrator { alpha: 1.0 };
    let curl = CurlCurlIntegrator { mu: 1.0 };
    let a = VectorAssembler::assemble_bilinear(&space, &[&mass, &curl], qo);
    let src = VectorDomainLFIntegrator {
        f: FnVectorCoeff(|x: &[f64], out: &mut [f64]| {
            out.copy_from_slice(&f_vec_rhs(x, false));
        }),
    };
    let b = VectorAssembler::assemble_linear(&space, &[&src], vector_load_order(order));

    let tags = mesh.unique_boundary_tags();
    let ess = boundary_dofs_hcurl(mesh, &space, &tags);
    let x0 = space.interpolate_vector(&u_vec_exact).as_slice().to_vec();

    let (a, b, mut x) = form_system(a, b.as_slice().to_vec(), &ess, &x0);

    // D367: the builders now take the HO essential dofs and eliminate them on
    // `A_LOR` (MFEM `LORSolver(a_ho, ess_tdof_list)` → `EliminateBC` DIAG_KEEP),
    // so the preconditioner sits on the same eliminated operator as the CG
    // system.  The C++ reference (no SuiteSparse, no hypre) preconditions with
    // `LORSolver<GSSmoother>`: one symmetric GS sweep on the eliminated LOR
    // matrix — reproduced by `build_lor_sgs_nd_quad`.
    let lor = fem_assembly::lor_factory::build_lor_sgs_nd_quad(&space, &a, 1.0, 1.0, &ess)
        .unwrap_or_else(|e| panic!("LOR-SGS build failed (ND P{order}): {e}"));
    let cfg = SolverConfig {
        rtol: 1e-12,
        atol: 0.0,
        max_iter: 500,
        verbose: false,
        ..SolverConfig::default()
    };
    let res = solve_pcg_precond(&a, &b, &mut x, &lor, &cfg).expect("LOR-SGS PCG failed");

    let l2_err = compute_l2_error_hcurl(&x, &space, &|x: &[f64]| u_vec_exact(x),
                                        l2_rule_order(order), None);
    Outcome {
        n_dofs,
        iterations: res.iterations,
        final_residual: res.final_residual,
        l2_err,
    }
}

/// `-fe r`: `M + div-div` on `H(div, order-1)`, LOR-Jacobi preconditioned CG.
fn run_rt(mesh: &Mesh<2>, order: u8) -> Outcome {
    let rt_order = order - 1;
    let space = HDivSpace::new(mesh.clone(), rt_order);
    let n_dofs = space.n_dofs();
    let qo = bilinear_order(order);
    let mass = VectorMassIntegrator { alpha: 1.0 };
    let divdiv = DivDivIntegrator { kappa: 1.0 };
    let a = VectorAssembler::assemble_bilinear(&space, &[&mass, &divdiv], qo);
    let src = VectorDomainLFIntegrator {
        f: FnVectorCoeff(|x: &[f64], out: &mut [f64]| {
            // MFEM `f_vec(RT)` with RT=true -> the grad-div load.
            out.copy_from_slice(&f_vec_rhs(x, true));
        }),
    };
    let b = VectorAssembler::assemble_linear(&space, &[&src], vector_load_order(order));

    let tags = mesh.unique_boundary_tags();
    let ess = boundary_dofs_hdiv_quad_rt(mesh, &space, &tags);
    let x0 = space.interpolate_vector(&u_vec_exact).as_slice().to_vec();

    let (a, b, mut x) = form_system(a, b.as_slice().to_vec(), &ess, &x0);

    let lor = fem_assembly::lor_factory::build_lor_sgs_rt_quad(&space, &a, 1.0, 1.0, &ess)
        .unwrap_or_else(|e| panic!("LOR-SGS RT build failed (RT{rt_order}): {e}"));
    let cfg = SolverConfig {
        rtol: 1e-12,
        atol: 0.0,
        max_iter: 500,
        verbose: false,
        ..SolverConfig::default()
    };
    let res = solve_pcg_precond(&a, &b, &mut x, &lor, &cfg).expect("LOR-Jacobi PCG failed");

    let l2_err = compute_l2_error_hdiv(&x, &space, &|x: &[f64]| u_vec_exact(x),
                                       l2_rule_order(order), None);
    Outcome {
        n_dofs,
        iterations: res.iterations,
        final_residual: res.final_residual,
        l2_err,
    }
}
