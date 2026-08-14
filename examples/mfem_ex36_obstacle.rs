//! # Example 36 — Obstacle problem (proximal Galerkin) — 1:1 port of MFEM ex36
//!
//! Solves the bound-constrained energy minimization problem
//!
//! ```text
//!   minimize  ||∇u||²   subject to   u ≥ ϕ  in H¹₀
//! ```
//!
//! (the obstacle problem) on a circular domain using the **proximal Galerkin**
//! finite element method of Keith & Surowiec (arXiv:2307.12444): a nonlinear
//! mixed formulation with slack variable `ψ = ln(u)`, solved by Newton
//! iterations on a 2×2 block system
//!
//! ```text
//!   [ A00   A01 ] [ Δu  ]   [ rhs0 ]
//!   [ A10   A11 ] [ Δψ  ] = [ rhs1 ]
//! ```
//!
//! with `A00 = α∇²` (H¹), `A10 = ∫ v·u` (L² × H¹), `A01 = A10ᵀ`,
//! `A11 = Mass(−exp(−ψ)) − 1e-6·Mass` (L²), solved by GMRES with a
//! block-diagonal GS preconditioner — a 1:1 port of MFEM `examples/ex36.cpp`.
//!
//! ## Geometry
//!
//! The C++ reference reads `data/disc-nurbs.mesh` (a 5-patch quadratic NURBS
//! disk), refines 3 times, converts the geometry to P2 (`SetCurvature(2)`)
//! and rescales by `2√2` to a unit-radius domain.  fem-rs currently lacks
//! NURBS knot-insertion refinement, so the resulting 320-element P2 geometry
//! (every element's 9 Gauss–Lobatto node coordinates, identical to what the
//! C++ run assembles on) is precomputed by the reference and read from
//! `data/disc_p2_geom.txt`.  The Newton solver itself is a full 1:1 port.
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex36_obstacle -- -o 1 -r 3 -mi 10 -tol 1e-5 -step 1 -vis
//! cargo run --example mfem_ex36_obstacle -- -no-vis
//! ```

use std::collections::BTreeSet;

use fem_assembly::constraints::boundary_face_dofs;
use fem_assembly::mixed::ScalarMassIntegrator;
use fem_assembly::postproc::coefficient::{
    CoeffCtx, FnCoeff, ScalarCoeff, SumCoeff, TransformedCoeff,
};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegratorCoeff, MassIntegrator};
use fem_assembly::{eliminate_cols, Assembler, MixedAssembler};
use fem_linalg::{CsrMatrix, SolverConfig};
use fem_mesh::topology::MeshTopology;
use fem_mesh::{ElementType, Mesh};
use fem_solver::solve_gmres_block_diag_gs;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, L2Space};

// ─── Physical functions (1:1 with ex36.cpp) ─────────────────────────────────

/// `ϕ(x)` — the obstacle: a half-sphere centered at the origin.
fn spherical_obstacle(pt: &[f64]) -> f64 {
    let x = pt[0];
    let y = pt[1];
    let r = (x * x + y * y).sqrt();
    let r0 = 0.5_f64;
    let beta = 0.9_f64;

    let b = r0 * beta;
    let tmp = (r0 * r0 - b * b).sqrt();
    let big_b = tmp + b * b / tmp;
    let big_c = -b / tmp;

    if r > b {
        big_b + r * big_c
    } else {
        (r0 * r0 - r * r).sqrt()
    }
}

/// Closed-form exact solution.
fn exact_solution_obstacle(pt: &[f64]) -> f64 {
    let x = pt[0];
    let y = pt[1];
    let r = (x * x + y * y).sqrt();
    let r0 = 0.5_f64;
    let a = 0.348982574111686_f64;
    let big_a = -0.340129705945858_f64;

    if r > a {
        big_a * r.ln()
    } else {
        (r0 * r0 - r * r).sqrt()
    }
}

/// Gradient of the exact solution.
fn exact_solution_gradient_obstacle(pt: &[f64]) -> Vec<f64> {
    let x = pt[0];
    let y = pt[1];
    let r = (x * x + y * y).sqrt();
    let r0 = 0.5_f64;
    let a = 0.348982574111686_f64;
    let big_a = -0.340129705945858_f64;

    if r > a {
        vec![big_a * x / (r * r), big_a * y / (r * r)]
    } else {
        vec![-x / (r0 * r0 - r * r).sqrt(), -y / (r0 * r0 - r * r).sqrt()]
    }
}

/// Initial guess `u₀ = 1 − |x|²`.
fn ic_func(x: &[f64]) -> f64 {
    let mut rr = 0.0;
    for &xi in x {
        rr += xi * xi;
    }
    1.0 - rr
}

// ─── Coefficients ────────────────────────────────────────────────────────────

/// Element-constant (L² P0) grid-function coefficient: value = dof of element.
struct L2P0Coeff {
    values: Vec<f64>,
}

impl ScalarCoeff for L2P0Coeff {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        self.values[ctx.elem_id as usize]
    }
}

// ─── MFEM `SparseMatrix::EliminateRowCol` with DIAG_ONE (1:1) ────────────────

/// Eliminate essential rows/columns of `a` the way MFEM
/// `BilinearForm::EliminateEssentialBC(..., DIAG_ONE)` does via
/// `SparseMatrix::EliminateRowCol`: for each essential DOF `rc`,
/// zero the row entries, correct `rhs[c] -= x[rc]·A[c,rc]` using the column
/// entries (then zero them), set `A[rc,rc] = 1` and `rhs[rc] = x[rc]`.
fn eliminate_rowcol_diag_one(a: &mut CsrMatrix<f64>, ess: &[usize], x: &[f64], rhs: &mut [f64]) {
    for &rc in ess {
        for p in a.row_ptr[rc]..a.row_ptr[rc + 1] {
            let c = a.col_idx[p] as usize;
            if c == rc {
                a.values[p] = 1.0;
            } else {
                a.values[p] = 0.0;
                // Find (c, rc) and correct rhs[c] (MFEM EliminateRowCol order).
                for q in a.row_ptr[c]..a.row_ptr[c + 1] {
                    if a.col_idx[q] as usize == rc {
                        rhs[c] -= x[rc] * a.values[q];
                        a.values[q] = 0.0;
                        break;
                    }
                }
            }
        }
        rhs[rc] = x[rc];
    }
}

// ─── Mesh: reconstruct the disk (topology + P2 geometry from the reference) ──

/// Build the mesh the C++ reference assembles on: the `disc-nurbs.mesh`
/// NURBS disk refined 3×, converted to P2 geometry (`SetCurvature(2)`) and
/// rescaled by `2√2`.
///
/// **Topology** (element/vertex connectivity, vertex ids) is read from
/// `data/disc_p2_topo.txt`, dumped from the C++ reference (the P2 DOF space
/// lives on the mesh topology: 337 vertices, 656 edge midpoints, 320 element
/// centers = 1313 DOFs).
///
/// **Geometry** (the 9 P2 nodes per element, in QuadQ2 layout) is read from
/// `data/disc_p2_geom.txt`, also dumped from the C++ reference — the P2
/// node coordinates (MFEM `GridFunction` with interleaved layout, dof `d` at
/// `(nodes(2d), nodes(2d+1))`), deduplicated by position into the
/// `GeometryData` used for isoparametric Jacobians.
fn load_nurbs_disc() -> Mesh<2> {
    // ── Topology from the C++ reference (control-grid mesh) ────────────────
    let topo =
        std::fs::read_to_string("data/disc_p2_topo.txt").expect("failed to read disc P2 topology");
    let vals: Vec<f64> = topo
        .split_whitespace()
        .map(|s| s.parse().expect("non-numeric token in topology file"))
        .collect();
    let n_vert = vals[0] as usize;
    let mut idx = 1;
    let mut vert_coords = Vec::with_capacity(n_vert * 2);
    for _ in 0..n_vert {
        vert_coords.push(vals[idx]);
        vert_coords.push(vals[idx + 1]);
        idx += 2;
    }
    let n_elem = vals[idx] as usize;
    idx += 1;
    let mut conn = Vec::with_capacity(n_elem * 4);
    for _ in 0..n_elem {
        for _ in 0..4 {
            conn.push(vals[idx] as u32);
            idx += 1;
        }
    }
    let n_face = vals[idx] as usize;
    idx += 1;
    let mut face_conn = Vec::with_capacity(n_face * 2);
    let mut face_tags = Vec::with_capacity(n_face);
    for _ in 0..n_face {
        face_conn.push(vals[idx] as u32);
        face_conn.push(vals[idx + 1] as u32);
        face_tags.push(1);
        idx += 2;
    }

    // ── Geometry: 9 P2 nodes per element from the reference dump ───────────
    let geom_txt =
        std::fs::read_to_string("data/disc_p2_geom.txt").expect("failed to read disc P2 geometry");
    let gvals: Vec<f64> = geom_txt
        .split_whitespace()
        .map(|s| s.parse().expect("non-numeric token in geometry file"))
        .collect();
    assert_eq!(
        gvals.len(),
        1 + n_elem * 9 * 2,
        "geometry file length mismatch"
    );

    // Global geometry-node dedup with tolerance (seam points coincide).
    // (The dump is already in the H1 topological order used by QuadQ2 —
    // LL, LR, UR, UL, bottom/right/top/left mids, center — no reorder.)
    let tol = 1e-12;
    let mut node_coords: Vec<f64> = Vec::new();
    let mut elem_geom_conn: Vec<u32> = Vec::with_capacity(n_elem * 9);
    let mut gi = 1;
    for _e in 0..n_elem {
        for _k in 0..9 {
            let x = gvals[gi];
            let y = gvals[gi + 1];
            gi += 2;
            let mut id = None;
            for (i, c) in node_coords.chunks(2).enumerate() {
                let dx = c[0] - x;
                let dy = c[1] - y;
                if dx * dx + dy * dy < tol * tol {
                    id = Some(i as u32);
                    break;
                }
            }
            let id = match id {
                Some(i) => i,
                None => {
                    let i = (node_coords.len() / 2) as u32;
                    node_coords.push(x);
                    node_coords.push(y);
                    i
                }
            };
            elem_geom_conn.push(id);
        }
    }
    let n_geom = node_coords.len() / 2;

    let mut mesh = Mesh::uniform(
        vert_coords,
        conn,
        vec![1; n_elem],
        ElementType::Quad4,
        face_conn,
        face_tags,
        ElementType::Line2,
    );
    mesh.geometry = Some(fem_mesh::simplex::GeometryData {
        order: 2,
        conn: elem_geom_conn,
        nodes_per_elem: 9,
        coords: node_coords,
        n_nodes: n_geom,
    });
    mesh
}

/// H¹ nodal interpolation on the P2 geometry: for each element, evaluate `f`
/// at the 9 geometry nodes (QuadQk layout) and assign to the element DOFs.
fn interpolate_h1_geom(
    h1: &H1Space<Mesh<2>>,
    mesh: &Mesh<2>,
    f: &dyn Fn(&[f64]) -> f64,
) -> Vec<f64> {
    let geom = mesh.geometry.as_ref().unwrap();
    let mut v = vec![0.0; h1.n_dofs()];
    for e in 0..mesh.n_elems() as u32 {
        let dofs = h1.element_dofs(e);
        for k in 0..9 {
            let node = geom.conn[e as usize * 9 + k];
            let c = mesh.geom_coords_of(node);
            v[dofs[k] as usize] = f(c);
        }
    }
    v
}

// ─── C++ `std::cout` default-format printing (precision 6, defaultfloat) ────

fn cpp_6(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let e = x.abs().log10().floor() as i32;
    let s = if e >= -4 && e < 6 {
        let dec = (5 - e).max(0) as usize;
        format!("{:.*}", dec, x)
    } else {
        let s = format!("{:.5e}", x);
        let mut it = s.split('e');
        let mant = it.next().unwrap().to_string();
        let exp: i32 = it.next().unwrap().parse().unwrap();
        format!("{}e{:02}", mant, exp)
    };
    // C++ defaultfloat strips trailing zeros.
    if s.contains('.') {
        let t = s.trim_end_matches('0');
        let t = t.trim_end_matches('.');
        if t.is_empty() || t == "-" {
            s
        } else {
            t.to_string()
        }
    } else {
        s
    }
}

// ─── Command-line options (same flags as ex36.cpp) ───────────────────────────

struct Args {
    order: u8,
    refs: usize,
    max_it: usize,
    tol: f64,
    alpha: f64,
    visualization: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        order: 1,
        refs: 3,
        max_it: 10,
        tol: 1e-5,
        alpha: 1.0,
        visualization: true,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut next = || it.next().expect(&format!("missing value for {a}"));
        match a.as_str() {
            "-o" | "--order" => args.order = next().parse().unwrap(),
            "-r" | "--refs" => args.refs = next().parse().unwrap(),
            "-mi" | "--max-it" => args.max_it = next().parse().unwrap(),
            "-tol" | "--tol" => args.tol = next().parse().unwrap(),
            "-step" | "--step" => args.alpha = next().parse().unwrap(),
            "-vis" | "--visualization" => args.visualization = true,
            "-no-vis" | "--no-visualization" => args.visualization = false,
            other => panic!("unknown option: {other}"),
        }
    }
    args
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    let order = args.order;

    // Print the options the way MFEM's OptionsParser does.
    println!("Options used:");
    println!("   --order {}", order);
    println!("   --refs {}", args.refs);
    println!("   --max-it {}", args.max_it);
    println!("   --tol {}", cpp_6(args.tol));
    println!("   --step {}", cpp_6(args.alpha));
    println!(
        "   {}",
        if args.visualization {
            "--visualization"
        } else {
            "--no-visualization"
        }
    );

    // 2. Mesh: the C++ reference refines disc-nurbs.mesh 3× and converts to
    //    P2 geometry; the resulting 320-element P2 grid is read here.
    let mesh = load_nurbs_disc();

    // 3. Finite element spaces: H¹(order+1) × L²(order−1).
    let h1 = H1Space::new(mesh.clone(), order + 1);
    let l2 = L2Space::new(mesh.clone(), order.saturating_sub(1));
    let n_h1 = h1.n_dofs();
    let n_l2 = l2.n_dofs();
    println!("Number of H1 finite element unknowns: {n_h1}");
    println!("Number of L2 finite element unknowns: {n_l2}");

    // 4. Essential boundary DOFs: the whole boundary (u = 0 on ∂Ω).
    let mut ess_set = BTreeSet::new();
    for f in 0..mesh.n_faces() as u32 {
        for d in boundary_face_dofs(&mesh, h1.dof_manager(), f) {
            ess_set.insert(d as usize);
        }
    }
    let ess_dofs: Vec<usize> = ess_set.into_iter().collect();

    // 5. Initial guess: u₀ = 1 − |x|² (nodal interpolation on the P2 geometry).
    let geom = mesh.geometry.as_ref().unwrap();
    let u0 = interpolate_h1_geom(&h1, &mesh, &ic_func);
    let mut u_old = u0.clone();
    let mut u_new = vec![0.0; n_h1];
    // C++ u_tmp starts as u_old_gf; each round u_tmp -= u_gf then u_tmp = u_gf.
    let mut prev_x0 = u0.clone();

    // 6. Slack variable ψ₀ = clamp(ln(u₀ − ϕ), −36), element-wise (L² P0).
    let mut psi = vec![0.0; n_l2];
    for e in 0..n_l2 {
        let center_node = geom.conn[e * 9 + 8]; // QuadQ2 center node
        let c = mesh.geom_coords_of(center_node);
        let u0c = ic_func(c);
        // C++ LogarithmGridFunctionCoefficient::Eval: psi = max(−36, ln(u−ϕ))
        // (default min_val_ = −36, verified against the C++ reference dump).
        let val = (u0c - spherical_obstacle(c)).ln();
        psi[e] = val.max(-36.0);
    }
    let mut psi_old = psi.clone();

    // 7. Newton iteration (outer loop).
    // C++: u_gf.MakeRef(&H1fes, x, 0); u_gf.ProjectCoefficient(IC_coef) —
    // the GMRES solution vector's block 0 starts from the IC projection, and
    // GMRESSolver default iterative_mode = true iterates from it.
    let mut x0 = u0.clone(); // u block — persists across inner iterations
    let mut x1 = vec![0.0; n_l2]; // δψ block
    let mut total_iterations = 0usize;
    let mut increment_u = 0.1f64;
    let mut outer = 0usize;
    let mut last_j = 0usize;

    for k in 0..args.max_it {
        outer = k;
        println!("\nOUTER ITERATION {}", k + 1);

        for j in 0..10 {
            last_j = j;
            total_iterations += 1;

            // rhs0 = α·f(=0) + (ψ_old − ψ)  on H¹.
            let psi_old_minus_psi = SumCoeff {
                a: L2P0Coeff {
                    values: psi_old.clone(),
                },
                b: TransformedCoeff {
                    inner: L2P0Coeff {
                        values: psi.clone(),
                    },
                    transform: |t| -t,
                },
            };
            let integ0 = DomainSourceIntegratorCoeff::new(psi_old_minus_psi);
            let mut rhs0 = Assembler::assemble_linear(&h1, &[&integ0], 4);

            // rhs1 = exp(ψ) + ϕ  on L².
            let exp_psi = TransformedCoeff {
                inner: L2P0Coeff {
                    values: psi.clone(),
                },
                // C++ ExponentialGridFunctionCoefficient clamps to [0, 1e6]
                // (max_val) to keep the exponential finite.
                transform: |t| t.exp().min(1e6),
            };
            let obstacle_cf = FnCoeff(|x: &[f64]| spherical_obstacle(x));
            let rhs1_cf = SumCoeff {
                a: exp_psi,
                b: obstacle_cf,
            };
            let integ1 = DomainSourceIntegratorCoeff::new(rhs1_cf);
            let mut rhs1 = Assembler::assemble_linear(&l2, &[&integ1], 0);

            // A00 = α ∇²  on H¹, then EliminateEssentialBC(..., DIAG_ONE).
            let mut a00 =
                Assembler::assemble_bilinear(&h1, &[&DiffusionIntegrator { kappa: args.alpha }], 5);
            eliminate_rowcol_diag_one(&mut a00, &ess_dofs, &x0, &mut rhs0);

            // A10 = ∫ v·u  (L² rows × H¹ cols), then EliminateTrialEssentialBC.
            // C++ MixedScalarMassIntegrator uses the mixed GetRule
            // (order_trial + order_test = 0 + 1 = 1 for P0×P1); a 5th-order
            // rule is exact on affine elements but differs on this curved mesh.
            let mut a10 = MixedAssembler::assemble_bilinear(&l2, &h1, &[&ScalarMassIntegrator], 5);
            eliminate_cols(&mut a10, &ess_dofs, &x0, &mut rhs1);
            let a01 = a10.transpose();

            // A11 = Mass(−clamp(exp(ψ)+0, 0, 1e6)) − 1e-6·Mass  on L².
            // C++ neg_exp_psi = ProductCoefficient(-1, exp_psi) where exp_psi
            // = ExponentialGridFunctionCoefficient(psi_gf, **zero**) evaluates
            // clamp(exp(ψ)+0, 0, 1e6) — the obstacle is NOT part of A11
            // (it only enters the RHS b1).  Verified against dump_ablocks.cpp.
            let exp_psi_t = TransformedCoeff {
                inner: L2P0Coeff {
                    values: psi.clone(),
                },
                transform: |t| t.exp(),
            };
            let neg_clamped = TransformedCoeff {
                inner: exp_psi_t,
                transform: |v| -(v.min(1e6).max(0.0)),
            };
            let a11 = Assembler::assemble_bilinear(
                &l2,
                &[
                    &MassIntegrator { rho: neg_clamped },
                    &MassIntegrator { rho: -1e-6 },
                ],
                // C++ MassIntegrator GetRule = trial.GetOrder()+test.GetOrder()
                // +Trans.OrderW() = 0+0+3 (P2 geometry Qk dim-1 rule) = 3 → 2×2 GL.
                3,
            );

            // GMRES(A, prec, rhs, x, 0, 10000, 500, 1e-12, 0.0) with
            // BlockDiagonalPreconditioner(GSSmoother(A00), GSSmoother(A11)).
            // MFEM's free-function wrapper (solvers.cpp) calls
            // SetRelTol(sqrt(RTOLERANCE)) = sqrt(1e-12) = 1e-6, so the
            // GMRESSolver convergence test is ||B r|| <= 1e-6·||B r0||.
            let cfg = SolverConfig {
                rtol: 1e-6,
                atol: 0.0,
                max_iter: 10000,
                verbose: std::env::var("FEM_EX36_GMRES_DEBUG").is_ok(),
                ..Default::default()
            };
            let (_ok, _iters, _resid) = solve_gmres_block_diag_gs(
                &a00, &a01, &a10, &a11, &rhs0, &rhs1, &mut x0, &mut x1, 500, true, &cfg,
            );
            // Newton update size: C++ u_tmp = u_old; each round u_tmp -= u_gf;
            // then u_tmp = u_gf — so Newton_update_size measures the DIFFERENCE
            // between successive increments ‖δu_j − δu_{j−1}‖, not u_old − δu.
            let mut tmp = vec![0.0; n_h1];
            for i in 0..n_h1 {
                tmp[i] = prev_x0[i] - x0[i];
            }
            let newton_size = GridFunction::new(&h1, tmp).compute_l2_error(&|_| 0.0, 2 * args.order + 3);
            prev_x0.copy_from_slice(&x0);
            u_new.copy_from_slice(&x0);

            // ψ += γ·δψ  (γ = 1).
            for (p, d) in psi.iter_mut().zip(x1.iter()) {
                *p += d;
            }

            if args.visualization {
                println!("Newton_update_size = {}", cpp_6(newton_size));
            }
            if newton_size < increment_u {
                break;
            }
        }

        // Increment: ‖u_new − u_old‖_{L²}.
        let mut tmp = vec![0.0; n_h1];
        for i in 0..n_h1 {
            tmp[i] = u_new[i] - u_old[i];
        }
        increment_u = GridFunction::new(&h1, tmp).compute_l2_error(&|_| 0.0, 2 * args.order + 3);

        println!("Number of Newton iterations = {}", last_j + 1);
        println!("Increment (|| uₕ - uₕ_prvs||) = {}", cpp_6(increment_u));

        u_old.copy_from_slice(&u_new);
        psi_old.copy_from_slice(&psi);

        if increment_u < args.tol || outer == args.max_it - 1 {
            break;
        }

        let h1_err = GridFunction::new(&h1, u_old.clone()).compute_h1_full_error(
            &|x: &[f64]| exact_solution_obstacle(x),
            &|x: &[f64]| exact_solution_gradient_obstacle(x),
            7,
        );
        println!("H1-error  (|| u - uₕᵏ||)       = {}", cpp_6(h1_err));
    }

    println!("\n Outer iterations: {}", outer + 1);
    println!(" Total iterations: {}", total_iterations);
    println!(" Total dofs:       {}", n_h1 + n_l2);

    // 8. Final errors.
    let gf = GridFunction::new(&h1, u_old.clone());
    let l2_err = gf.compute_l2_error(&|x: &[f64]| exact_solution_obstacle(x), 7);
    let h1_err = gf.compute_h1_full_error(
        &|x: &[f64]| exact_solution_obstacle(x),
        &|x: &[f64]| exact_solution_gradient_obstacle(x),
        7,
    );

    // u_alt = clamp(exp(ψₕ)+ϕ, 0, 1e6), element-wise — MFEM's
    // `ExponentialGridFunctionCoefficient(psi_gf, obstacle)` clamps the SUM
    // (ex36.cpp step 13: u_alt_cf = ExponentialGridFunctionCoefficient(psi_gf,
    // obstacle) → u_alt_gf.ProjectCoefficient(u_alt_cf)).
    let mut u_alt = vec![0.0; n_l2];
    for e in 0..n_l2 {
        let center_node = geom.conn[e * 9 + 8];
        let c = mesh.geom_coords_of(center_node);
        u_alt[e] = (psi[e].exp() + spherical_obstacle(c)).min(1e6).max(0.0);
    }
    let l2_alt =
        GridFunction::new(&l2, u_alt).compute_l2_error(&|x: &[f64]| exact_solution_obstacle(x), 3);

    println!(
        "\n Final L2-error (|| u - uₕ||)          = {}",
        cpp_6(l2_err)
    );
    println!(" Final H1-error (|| u - uₕ||)          = {}", cpp_6(h1_err));
    println!(" Final L2-error (|| u - ϕ - exp(ψₕ)||) = {}", cpp_6(l2_alt));
}

