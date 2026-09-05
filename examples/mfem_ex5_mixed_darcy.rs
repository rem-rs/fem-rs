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

use std::fs::File;
use std::io::Write;
use std::time::Instant;

use fem_assembly::mixed::{assemble_hdiv_l2_mixed, HDivL2DivIntegrator};
use fem_assembly::standard::VectorMassIntegrator;
use fem_assembly::{
    VectorAssembler,
    vector_integrator::{VectorLinearIntegrator, VectorQpData},
};
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_gf_file};
use fem_mesh::{refine_uniform, Mesh, MeshTopology};
use fem_linalg::{CooMatrix, fem_to_linlvo_csr};
use fem_solver::block::BlockSystem;
use fem_space::{HDivSpace, L2Space, fe_space::FESpace};
use fem_assembly::postproc::grid_function::GridFunction;
use linlvo::{
    precond::{BlockDiagonalPreconditioner, GaussSeidelSmoother, JacobiPrecond, SplitMode},
    DenseVec, KrylovSolver, Minres, SolverParams, VerboseLevel,
};

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

    // ── Build flat block system ─────────────────────────────────────────
    let mm_clone = mm.clone();
    let mb_clone = mb.clone();
    let bt = mb.transpose(); // B^T
    let flat = BlockSystem { a: mm, bt, b: mb, c: None }.to_flat_csr();
    let n = n_u + n_p;
    let mut rhs = Vec::with_capacity(n);
    rhs.extend(fu); rhs.extend(gp);
    let mut x = vec![0.0; n];

    // ── Build preconditioner ────────────────────────────────────────────
    // C++:  BlockDiagonalPreconditioner darcyPrec(block_offsets);
    //       darcyPrec.SetDiagonalBlock(0, new DSmoother(M));     // diag(M)⁻¹
    //       darcyPrec.SetDiagonalBlock(1, new GSSmoother(*S));   // GS on S

    let mm = mm_clone;
    let mb = mb_clone;
    let diag_m: Vec<f64> = (0..n_u).map(|i| mm.get(i, i).max(1e-30)).collect();

    // S = B diag(M)^{-1} B^T (exact, no regularization — matches C++)
    let bt_t = mb.transpose();
    let mut minvbt_coo = CooMatrix::<f64>::new(n_u, n_p);
    for i in 0..n_u {
        let inv_d = 1.0 / diag_m[i];
        for ptr in bt_t.row_ptr[i]..bt_t.row_ptr[i+1] {
            let j = bt_t.col_idx[ptr] as usize;
            minvbt_coo.add(i, j, bt_t.values[ptr] * inv_d);
        }
    }
    let minvbt = minvbt_coo.into_csr();
    let s = mb.multiply(&minvbt);

    let m_linlvo = fem_to_linlvo_csr(&mm);
    let jacobi = JacobiPrecond::from_csr(&m_linlvo)
        .expect("JacobiPrecond on mass matrix failed");

    // C++: GSSmoother(*S) — Gauss-Seidel on exact Schur complement
    let s_linlvo = fem_to_linlvo_csr(&s);
    let gs = GaussSeidelSmoother::from_csr(&s_linlvo)
        .expect("GaussSeidelSmoother on Schur complement failed");

    let prec: BlockDiagonalPreconditioner<f64> = BlockDiagonalPreconditioner::new(
        n, n_u, SplitMode::BlockJacobi, Box::new(jacobi), Box::new(gs),
    );

    // ── Solve with preconditioned MINRES ────────────────────────────────
    let flat_linlvo = fem_to_linlvo_csr(&flat);
    let lb = DenseVec::from_vec(rhs);
    let mut lx = DenseVec::zeros(n);

    let params = SolverParams {
        rtol:  1e-6,
        atol:  1e-10,
        max_iter: 1000,  // MFEM ex5: maxIter = 1000
        verbose: VerboseLevel::Iterations,
        check_interval: 1,
    };

    let solver = Minres::<f64>::default();
    let start = Instant::now();
    let result = solver.solve(&flat_linlvo, Some(&prec), &lb, &mut lx, &params);
    let elapsed = start.elapsed();

    if let Ok(ref res) = result {
        if res.final_residual.is_finite() {
            x.copy_from_slice(lx.as_slice());
        }
    }

    match result {
        Ok(res) => {
            println!();
            if res.converged {
                // MFEM format: "MINRES converged in N iterations with a residual norm of X.XXXe-XX."
                println!("MINRES converged in {} iterations with a residual norm of {:.3e}.", res.iterations, res.final_residual);
            } else {
                println!("MINRES did not converge in {} iterations. Residual norm is {:.3e}.", res.iterations, res.final_residual);
            }
            println!("MINRES solver took {:.4}s.", elapsed.as_secs_f64());
        }
        Err(e) => println!("\nMINRES error: {e}"),
    }

    // ── L² errors (matching C++ MFEM ex5 exactly) ────────────────────
    // MFEM: order_quad = max(2, 2*order+1);
    //        err_u = u.ComputeL2Error(ucoeff, irs);
    //        norm_u = ComputeLpNorm(2., ucoeff, *mesh, irs);
    let order_quad = std::cmp::max(2, 2 * args.order + 1);

    let u_ex_fn = |x: &[f64]| -> [f64; 2] {
        [-(x[0].exp() * x[1].sin()), -(x[0].exp() * x[1].cos())]
    };
    let p_ex_fn = |x: &[f64]| -> f64 { x[0].exp() * x[1].sin() };

    // Pressure L² error (scalar L2 space — MFEM ComputeL2Error)
    let p_gf = GridFunction::new(&p_sp, x[n_u..].to_vec());
    let ep = p_gf.compute_l2_error(&p_ex_fn, order_quad);
    // Normalize by ||p_ex|| (MFEM: ComputeLpNorm(2., pcoeff, *mesh, irs))
    let p_zero = GridFunction::new(&p_sp, vec![0.0; n_p]);
    let np = p_zero.compute_l2_error(&p_ex_fn, order_quad);

    // Velocity L² error (H(div) vector field — contravariant Piola)
    let eu = compute_hdiv_l2_error_2d_simple(&u_sp, &x[..n_u], &u_ex_fn);
    let nu = compute_hdiv_l2_error_2d_simple(&u_sp, &vec![0.0; n_u], &u_ex_fn);

    // MFEM output format: "|| u_h - u_ex || / || u_ex || = %.6e"
    println!("|| u_h - u_ex || / || u_ex || = {:.6e}", eu / nu.max(1e-32));
    println!("|| p_h - p_ex || / || p_ex || = {:.6e}", ep / np.max(1e-32));

    // ── Output ──────────────────────────────────────────────────────────
    write_mfem_file("ex5.mesh", u_sp.mesh()).expect("mesh write failed");
    write_mfem_gf_file("sol_u.gf", dim, &x[..n_u], "H1", args.order, dim, 14).expect("write sol_u");
    write_mfem_gf_file("sol_p.gf", dim, &x[n_u..], "H1", args.order, 1, 14).expect("write sol_p");
    eprintln!("  Wrote ex5.mesh, sol_u.gf, sol_p.gf");
}

fn compute_hdiv_l2_error_2d_simple<F>(space: &HDivSpace<Mesh<2>>, u: &[f64], ex: &F) -> f64
where
    F: Fn(&[f64]) -> [f64; 2],
{
    use fem_element::raviart_thomas::TriRTk;
    use fem_element::reference::VectorReferenceElement;
    use fem_mesh::ElementTransformation;

    let mut e2 = 0.0;
    let ref_elem = TriRTk::new(0);
    let n_ldofs = ref_elem.n_dofs();
    let q = ref_elem.quadrature(6);
    let mut ref_phi = vec![0.0_f64; n_ldofs * 2];

    for e in space.mesh().elem_iter() {
        let nodes = space.mesh().elem_nodes(e);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);
        let tr = ElementTransformation::from_simplex_nodes(space.mesh(), nodes);
        let jac = tr.jacobian();
        let det_j = tr.det_j();
        let inv_det = 1.0 / det_j;

        for (qi, xi) in q.points.iter().enumerate() {
            ref_elem.eval_basis_vec(xi, &mut ref_phi);
            let w = q.weights[qi] * det_j.abs();

            // Contravariant Piola: φ_phys = (1/det(J)) · J · φ_ref
            let mut fh = [0.0_f64; 2];
            for i in 0..n_ldofs {
                let s = signs[i];
                let r0 = ref_phi[i * 2];
                let r1 = ref_phi[i * 2 + 1];
                let px = s * (jac[(0, 0)] * r0 + jac[(0, 1)] * r1) * inv_det;
                let py = s * (jac[(1, 0)] * r0 + jac[(1, 1)] * r1) * inv_det;
                fh[0] += u[dofs[i]] * px;
                fh[1] += u[dofs[i]] * py;
            }

            let xp = [tr.map_to_physical(xi)[0], tr.map_to_physical(xi)[1]];
            let exact = ex(&xp);
            e2 += w * ((fh[0] - exact[0]).powi(2) + (fh[1] - exact[1]).powi(2));
        }
    }
    e2.sqrt()
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
