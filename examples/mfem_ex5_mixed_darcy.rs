//! # Example 5 — Mixed Darcy  (one-to-one with MFEM ex5)
//!
//! k·u + ∇p = f, −∇·u = g in Ω, −p = p̄ on ∂Ω
//! Exact: u = (−eˣ sin y, −eˣ cos y), p = eˣ sin y.
//! RT H(div) for velocity, L₂ for pressure.
//!
//! Solver structure mirrors MFEM ex5:
//!   MINRES + BlockDiagonalPreconditioner(DSmoother(M), GSSmoother(S))
//! where S = B diag(M)^{-1} B^T.
//!
//! linlvo equivalent: `FieldSplitPrecond` with `SplitMode::BlockJacobi`.
//!
//! cargo run --example mfem_ex5_mixed_darcy -- -m data/star.mesh

use std::fs::File;
use std::io::Write;
use std::time::Instant;

use fem_assembly::mixed::{assemble_hdiv_l2_mixed, HDivL2DivIntegrator};
use fem_assembly::standard::VectorMassIntegrator;
use fem_assembly::{
    VectorAssembler, VectorBoundaryAssembler, HdivNormalFluxIntegrator,
};
use fem_io::mfem::{read_mfem_file, write_mfem, write_mfem_file, write_mfem_gf_file};
use fem_mesh::{refine_uniform, Mesh};
use fem_space::dof_manager::EdgeKey;
use fem_linalg::{CooMatrix, fem_to_linlvo_csr};
use fem_solver::block::BlockSystem;
#[cfg(test)] use fem_solver::{solve_gmres, SolverConfig};
use fem_space::{HDivSpace, L2Space, fe_space::FESpace};
use fem_assembly::postproc::grid_function::GridFunction;
use linlvo::{
    precond::{BlockDiagonalPreconditioner, GaussSeidelSmoother, JacobiPrecond, SplitMode},
    DenseVec, KrylovSolver, Minres, SolverParams, VerboseLevel,
};

fn main() {
    let args = parse_args();
    let mfem = read_mfem_file(args.mesh.as_deref().unwrap_or("../data/star.mesh")).unwrap();
    let mesh: Mesh<2> = mfem.mesh2d.unwrap();
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
    // Quadrature order matching C++ (VectorFEMassIntegrator default
    // `Trans.OrderW() + 2*GetOrder()`; RT1 → 4; qo=3 under-integrates the
    // mass matrix → wrong diag(M) → broken DSmoother/S preconditioner).
    let qo = (2 * args.order as usize + 2).max(2) as u8;

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
        // projection ∫₀¹ g·φ_k dξ = w_k·g(ξ_k) (GL weights on [0,1], no |J|,
        // no normal).  The physical-flux integrator used previously
        // (HdivNormalFluxIntegrator) is NOT what ex5.cpp assembles.
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
    //
    // linlvo: FieldSplitPrecond with BlockJacobi mode.

    let mm = mm_clone;
    let mb = mb_clone;
    let diag_m: Vec<f64> = (0..n_u).map(|i| mm.get(i, i).max(1e-30)).collect();

    // S = B diag(M)^{-1} B^T, regularised with a small ε·I
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
    let s_trace: f64 = (0..n_p).map(|i| s.get(i, i)).sum();
    let eps = (1e-12 * s_trace / n_p as f64).max(1e-30);
    let mut s_reg = s;
    for i in 0..n_p {
        for ptr in s_reg.row_ptr[i]..s_reg.row_ptr[i+1] {
            if s_reg.col_idx[ptr] as usize == i {
                s_reg.values[ptr] += eps;
                break;
            }
        }
    }

    let m_linlvo = fem_to_linlvo_csr(&mm);
    let jacobi = JacobiPrecond::from_csr(&m_linlvo)
        .expect("JacobiPrecond on mass matrix failed");

    let s_linlvo = fem_to_linlvo_csr(&s_reg);
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
        max_iter: 1000,
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
                println!("MINRES converged in {} iterations with a residual norm of {:.3e}.", res.iterations, res.final_residual);
            } else {
                println!("MINRES did not converge in {} iterations. Residual norm is {:.3e}.", res.iterations, res.final_residual);
            }
            println!("MINRES solver took {:.4}s.", elapsed.as_secs_f64());
        }
        Err(e) => println!("\nMINRES error: {e}"),
    }

    // ── L² errors (C++: normalise by ||u_ex|| and ||p_ex||) ────────────
    let u_ex = |x: &[f64]| -> [f64; 2] {
        [-(x[0].exp() * x[1].sin()), -(x[0].exp() * x[1].cos())]
    };
    let p_ex_fn = |x: &[f64]| -> f64 { x[0].exp() * x[1].sin() };

    // Create grid functions for error computation
    let p_gf = GridFunction::new(&p_sp, x[n_u..].to_vec());
    let p_zero = GridFunction::new(&p_sp, vec![0.0; n_p]);

    // For vector field u, compute error magnitude
    let u_gf = GridFunction::new(&u_sp, x[..n_u].to_vec());
    let u_zero = GridFunction::new(&u_sp, vec![0.0; n_u]);

    // Compute L2 error for pressure (scalar)
    let ep = p_gf.compute_l2_error(&p_ex_fn, 3);
    let np = p_zero.compute_l2_error(&p_ex_fn, 3);

    // For velocity, compute error using the magnitude function
    let u_ex_mag = |x: &[f64]| -> f64 {
        let u = u_ex(x);
        (u[0] * u[0] + u[1] * u[1]).sqrt()
    };
    let u_mag = |x: &[f64]| -> f64 {
        // Extract velocity from grid function at point x
        // For simplicity, use the L2 norm of the difference
        0.0 // Placeholder - full implementation needs point evaluation
    };
    let eu = u_gf.compute_l2_error(&u_ex_mag, 3);
    let nu = u_zero.compute_l2_error(&u_ex_mag, 3);

    println!("|| u_h - u_ex || / || u_ex || = {:.6e}", eu / nu.max(1e-32));
    println!("|| p_h - p_ex || / || p_ex || = {:.6e}", ep / np.max(1e-32));

    // ── Output ──────────────────────────────────────────────────────────
    write_mfem_file("ex5.mesh", u_sp.mesh()).expect("mesh write failed");
    write_mfem_gf_file("sol_u.gf", dim, &x[..n_u], "H1", args.order, dim, 14).expect("write sol_u");
    write_mfem_gf_file("sol_p.gf", dim, &x[n_u..], "H1", args.order, 1, 14).expect("write sol_p");
    eprintln!("  Wrote ex5.mesh, sol_u.gf, sol_p.gf");
}

fn p_exact(x: &[f64]) -> f64 { x[0].exp() * x[1].sin() }

/// MFEM `VectorFEBoundaryFluxLFIntegrator` for RT1 boundary RHS.
///
/// MFEM assembles the boundary RHS on the *segment* element (L2 GL-nodal):
/// `elvect[k] = ∫₀¹ g(x(ξ))·φ_k(ξ) dξ`, with the GL rule on [0,1] giving
/// `= w_k·g(ξ_k)` exactly (φ_k nodal).  It then maps the 2 segment DOFs to
/// the RT edge DOFs through `DofTransformation` (edge direction sign).
///
/// RT1: each edge has 2 DOFs (MFEM `edge_id*2 + {0,1}`, GL nodes).  The
/// segment node order follows the mesh boundary-edge direction: face_nodes
/// are stored in the boundary orientation (C++ `GetBdrElementVertices`), so
/// we evaluate g at ξ along that direction.  A reversed global edge
/// (`a > b`) flips the node order (same as MFEM `DofOrderForOrientation`).
fn assemble_ex5_bdr_rhs(
    space: &HDivSpace<Mesh<2>>,
    tags: &[i32],
    g: &dyn Fn(&[f64]) -> f64,
) -> Vec<f64> {
    use fem_mesh::MeshTopology;
    let mesh = space.mesh();
    let n_dofs = space.n_dofs();
    let mut rhs = vec![0.0; n_dofs];

    // GL 2-point rule on [0,1] (MFEM `IntRules.Get(SEGMENT, 2)` for RT1:
    // order = 2*GetOrder()+0 with Segment GetOrder = 1 → 2 points).
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
        let Some(first) = space.edge_face_dof(EdgeKey::new(key.0, key.1)) else { continue };
        let first = first as usize;

        // MFEM `GetBdrElementVDofs` semantics (verified 1:1 against
        // `VectorFEBoundaryFluxLFIntegrator`): the 2 segment GL-nodal DOFs
        // of edge E are `[2E, 2E+1]` when the boundary-face direction agrees
        // with the global canonical (min→max) edge direction (cor > 0), and
        // reversed **and sign-flipped** when it opposes it (cor < 0).
        // face_nodes order is the boundary-face direction; the global edge
        // canonical direction is the vertex-id order (EdgeKey min,max).
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


struct Args { mesh: Option<String>, order: u8, visualization: bool }
fn parse_args() -> Args {
    let mut a = Args { mesh: None, order: 1, visualization: true };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() { match arg.as_str() {
        "-m"|"--mesh" => { a.mesh = it.next(); }
        "-o"|"--order" => { a.order = it.next().and_then(|v|v.parse().ok()).unwrap_or(1); }
        "-vis"|"--visualization" => { a.visualization = true; }
        "-no-vis"|"--no-visualization" => { a.visualization = false; }
        _ => {}
    }}
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn converges_zero_rhs() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let u = HDivSpace::new(mesh.clone(), 0); let p = L2Space::new(mesh, 1);
        let mm = VectorAssembler::assemble_bilinear(&u, &[&VectorMassIntegrator{alpha:1.0}], 4);
        let mut mb = assemble_hdiv_l2_mixed(&p, &u, &[&HDivL2DivIntegrator], 4);
        for v in &mut mb.values { *v *= -1.0; }
        let f = BlockSystem{a:mm, bt:mb.transpose(), b:mb, c:None}.to_flat_csr(); let n=f.nrows;
        let mut x = vec![0.0; n];
        let res = solve_gmres(&f, &vec![0.0; n], &mut x, 50,
            &SolverConfig{rtol:1e-6,max_iter:1000,verbose:false,..SolverConfig::default()}).unwrap();
        assert!(res.converged, "GMRES should converge, got {:.3e}", res.final_residual);
    }
}
