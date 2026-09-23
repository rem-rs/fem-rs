//! Example 5 — Mixed Darcy (1:1 with MFEM ex5)
//!
//! Solves the saddle point system:
//!   k·u + ∇p = f,  −∇·u = g  in Ω,  −p = p̄ on ∂Ω
//! Exact: u = (−eˣ sin y, −eˣ cos y), p = eˣ sin y.
//! RT H(div) for velocity, L₂ for pressure.
//!
//! Solver structure mirrors MFEM ex5:
//!   MINRES + BlockDiagonalPreconditioner(DSmoother(M), GSSmoother(S))
//! where S = B diag(M)^{-1} B^T.

use std::time::Instant;

use fem_assembly::mixed::{assemble_hdiv_l2_mixed, HDivL2DivIntegrator};
use fem_assembly::hdiv_error::compute_hdiv_l2_error_order;
use fem_assembly::standard::VectorMassIntegrator;
use fem_assembly::VectorAssembler;
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_gf_file};
use fem_mesh::{refine_uniform, Mesh};
use fem_space::{HDivSpace, L2Space, fe_space::FESpace};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_solver::darcy_solvers::{IterSolveParameters, mfem_minres, schur_complement_bmb_diag};
use fem_solver::smoother::{GsSmoother, GsType};
use fem_solver::fmt_g;

fn main() {
    let args = parse_args();
    let mesh = read_mfem_file(args.mesh.as_deref().unwrap_or("../data/star.mesh")).unwrap();
    let mesh: Mesh<2> = mesh.mesh2d.unwrap();
    let dim = 2;

    // C++: ref_levels so the final mesh has ≤ 10 000 elements
    let rl = ((10000.0 / mesh.n_elems() as f64).ln() / (2.0_f64).ln() / dim as f64).floor() as usize;
    let mesh = if rl > 0 { let mut m = mesh; for _ in 0..rl { m = refine_uniform(&m); } m } else { mesh };
    let u_sp = HDivSpace::new(mesh.clone(), args.order);
    let p_sp = L2Space::new(mesh, args.order);
    let n_u = u_sp.n_dofs(); let n_p = p_sp.n_dofs();

    // C++: print block dimensions with separator lines
    println!("***********************************************************");
    println!("dim(R) = {n_u}");
    println!("dim(W) = {n_p}");
    println!("dim(R+W) = {}", n_u + n_p);
    println!("***********************************************************");

    // ── Assemble ─────────────────────────────────────────────────────────
    // MFEM: VectorFEMassIntegrator default order = Trans.OrderW() + 2*GetOrder()
    // For RT1: 1 + 2*1 = 3. DivDivIntegrator: max(2*order-2, 0) = 0 for order=1.
    // Use qo = 2*order + 1 to match MFEM ex5 default.
    let qo = (2 * args.order as usize + 1).max(2) as u8;

    // M = ∫ (u·v) dx   (mass matrix)
    let mm = VectorAssembler::assemble_bilinear(&u_sp, &[&VectorMassIntegrator{alpha:1.0}], qo);
    // B  = −∫ div(u) q dx   (divergence, negated)
    let mut mb = assemble_hdiv_l2_mixed(&p_sp, &u_sp, &[&HDivL2DivIntegrator], qo);
    for v in &mut mb.values { *v *= -1.0; } // C++: B *= -1

    // ── RHS: natural BC −p = p_exact → ∫ (−p_exact)·(v·n) ds ──────────
    let tags: Vec<i32> = u_sp.mesh().unique_boundary_tags();
    let fu = if !tags.is_empty() {
        // MFEM `VectorFEBoundaryFluxLFIntegrator` semantics: the RT boundary
        // DOFs are GL-nodal edge traces, so the RHS is the *reference* L²
        // projection ∫₀¹ g·φ_k dξ = w_k·g(ξ_k) (GL weights on [0,1], no |J|, no normal).
        fn neg_p_exact(x: &[f64]) -> f64 { -p_exact(x) }
        assemble_ex5_bdr_rhs(&u_sp, &tags, &neg_p_exact)
    } else {
        vec![0.0; n_u]
    };
    let gp = vec![0.0; n_p]; // g = 0 in 2D

    // ── Flat saddle operator [[M, Bᵀ], [B, 0]] and RHS ───────────────────
    let bt = mb.transpose(); // B^T
    let n = n_u + n_p;
    let mut rhs = Vec::with_capacity(n);
    rhs.extend(fu); rhs.extend(gp);
    let mut x = vec![0.0; n];

    // ── Preconditioner: BlockDiagonalPreconditioner(DSmoother(M), GSSmoother(S))
    //   C++ ex5.cpp: MinvBt = Transpose(B) scaled by 1/diag(M), S = B·MinvBt,
    //   invM = new DSmoother(M) (= diag(M)⁻¹), invS = new GSSmoother(*S)
    //   (symmetric, 1 sweep, zero start), both iterative_mode = false.
    let m_diag: Vec<f64> = (0..n_u).map(|i| mm.get(i, i)).collect();
    let s = schur_complement_bmb_diag(&mb, &m_diag);
    let gs = GsSmoother::new(&s, GsType::Symmetric, 1);

    // ── Solve with MINRESSolver ──────────────────────────────────────────
    //   C++ ex5.cpp: MINRESSolver, SetAbsTol(1e-10), SetRelTol(1e-6),
    //   SetMaxIter(1000), SetPrintLevel(1) — the exact `mfem_minres` port
    //   (linalg/solvers.cpp MINRESSolver::Mult, incl. the ‖r‖_B print format).
    let param = IterSolveParameters {
        print_level: 1,
        max_iter: 1000,  // MFEM ex5: maxIter = 1000
        abs_tol: 1e-10,
        rel_tol: 1e-6,
    };
    let apply_op = |v: &[f64], w: &mut [f64]| {
        // [[M, Bᵀ], [B, 0]] · v
        mm.spmv(&v[..n_u], &mut w[..n_u]);
        let mut bt_vp = vec![0.0; n_u];
        bt.spmv(&v[n_u..], &mut bt_vp);
        for (wi, bi) in w[..n_u].iter_mut().zip(&bt_vp) {
            *wi += bi;
        }
        mb.spmv(&v[..n_u], &mut w[n_u..]);
    };
    let apply_prec = |v: &[f64], w: &mut [f64]| {
        // diag(DSmoother(M), GSSmoother(S)) · v
        for (wi, (&vi, &di)) in w[..n_u].iter_mut().zip(v[..n_u].iter().zip(&m_diag)) {
            *wi = vi / di;
        }
        gs.mult(&v[n_u..], &mut w[n_u..]);
    };

    let start = Instant::now();
    let (iters, converged, final_norm) =
        mfem_minres(n, &apply_op, Some(&apply_prec), &rhs, &mut x, &param);
    let elapsed = start.elapsed();

    // C++: solver.GetConverged() / GetNumIterations() / GetFinalNorm() print
    // (default ostream precision — `fmt_g`).
    if converged {
        println!("MINRES converged in {iters} iterations with a residual norm of {}.", fmt_g(final_norm));
    } else {
        println!("MINRES did not converge in {iters} iterations. Residual norm is {}.", fmt_g(final_norm));
    }
    println!("MINRES solver took {}s.", fmt_g(elapsed.as_secs_f64()));

    // ── L² errors (matching C++ MFEM ex5 exactly) ────────────────────
    // MFEM: order_quad = max(2, 2*order+1);
    //        err_u = u.ComputeL2Error(ucoeff, irs);
    //        norm_u = ComputeLpNorm(2., ucoeff, *mesh, irs);
    let order_quad = std::cmp::max(2, 2 * args.order + 1);

    let p_ex_fn = |x: &[f64]| -> f64 { x[0].exp() * x[1].sin() };

    // Pressure L² error (scalar L2 space — MFEM ComputeL2Error)
    let p_gf = GridFunction::new(&p_sp, x[n_u..].to_vec());
    let ep = p_gf.compute_l2_error(&p_ex_fn, order_quad);
    // Normalize by ||p_ex|| (MFEM: ComputeLpNorm(2., pcoeff, *mesh, irs))
    let p_zero = GridFunction::new(&p_sp, vec![0.0; n_p]);
    let np = p_zero.compute_l2_error(&p_ex_fn, order_quad);

    // Velocity L² error (H(div) vector field — contravariant Piola).
    // MFEM: u.ComputeL2Error(ucoeff, irs) with irs = IntRules.Get(geom,
    // order_quad); `nu` plays the role of MFEM's ComputeLpNorm(2., ucoeff,
    // *mesh, irs) via the zero-field error (same integral).
    // D639: this used to be an example-local reconstruction hardcoded to
    // `TriRTk::new(0)` + triangle quadrature + the first 3 element DOFs.
    // star.mesh is a *quad* mesh (RT1-quad = 12 DOFs/element, bilinear
    // geometry), so the reconstructed field was garbage and the printed
    // ratio was solution-insensitive (1.211582e0 vs 1.211583e0 for two
    // different solutions; C++: 1.43587e-4).  The core routine picks the
    // per-element reference element/geometry exactly like the assembler.
    let u_ex_vec = |x: &[f64]| -> Vec<f64> {
        vec![-(x[0].exp() * x[1].sin()), -(x[0].exp() * x[1].cos())]
    };
    let eu = compute_hdiv_l2_error_order(&u_sp, &x[..n_u], &u_ex_vec, order_quad);
    let nu = compute_hdiv_l2_error_order(&u_sp, &vec![0.0; n_u], &u_ex_vec, order_quad);

    // MFEM output format (default ostream precision — `fmt_g`):
    //   "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u
    println!("|| u_h - u_ex || / || u_ex || = {}", fmt_g(eu / nu.max(1e-32)));
    println!("|| p_h - p_ex || / || p_ex || = {}", fmt_g(ep / np.max(1e-32)));

    // ── Output ──────────────────────────────────────────────────────────
    write_mfem_file("ex5.mesh", u_sp.mesh()).expect("mesh write failed");
    write_mfem_gf_file("sol_u.gf", dim, &x[..n_u], "H1", args.order, dim, 14).expect("write sol_u");
    write_mfem_gf_file("sol_p.gf", dim, &x[n_u..], "H1", args.order, 1, 14).expect("write sol_p");
    eprintln!("  Wrote ex5.mesh, sol_u.gf, sol_p.gf");
}

fn assemble_ex5_bdr_rhs(
    space: &HDivSpace<Mesh<2>>,
    tags: &[i32],
    g: &dyn Fn(&[f64]) -> f64,
) -> Vec<f64> {
    use fem_mesh::MeshTopology;
    let mesh = space.mesh();
    let n_dofs = space.n_dofs();
    let mut rhs = vec![0.0; n_dofs];

    // GL 2-point rule on [0,1] (MFEM `IntRules.Get(SEGMENT, 2)` for RT1)
    let xi = [0.5 * (1.0 - 1.0 / 3.0f64.sqrt()), 0.5 * (1.0 + 1.0 / 3.0f64.sqrt())];
    let wts = [0.5, 0.5];

    for f in 0..mesh.n_boundary_faces() as u32 {
        if !tags.contains(&mesh.face_tag(f)) { continue; }
        let nodes = mesh.face_nodes(f);
        if nodes.len() < 2 { continue; }
        let pa = mesh.node_coords(nodes[0]);
        let pb = mesh.node_coords(nodes[1]);
        let (a, b) = (nodes[0], nodes[1]);
        let key = if a < b { (a, b) } else { (b, a) };
        let Some(first) = space.edge_face_dof(fem_space::dof_manager::EdgeKey::new(key.0, key.1)) else { continue };
        let first = first as usize;

        let face_forward = a < b;
        let cor = if face_forward { 1 } else { -1 };
        for k in 0..2 {
            let t = xi[k];
            let xp = [pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1])];
            let global = if cor > 0 { first + k } else { first + (1 - k) };
            let sgn = if cor > 0 { 1.0 } else { -1.0 };
            rhs[global] += sgn * wts[k] * (g)(&xp);
        }
    }
    rhs
}

fn p_exact(x: &[f64]) -> f64 { x[0].exp() * x[1].sin() }

struct Args { mesh: Option<String>, order: u8, visualization: bool }

fn parse_args() -> Args {
    let mut a = Args { mesh: None, order: 1, visualization: true };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => { a.mesh = it.next(); }
            "-o" | "--order" => { a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
            "-vis" | "--visualization" => { a.visualization = true; }
            "-no-vis" | "--no-visualization" => { a.visualization = false; }
            _ => {}
        }
    }
    a
}
