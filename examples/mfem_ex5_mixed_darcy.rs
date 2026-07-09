//! # Example 5 — Mixed Darcy  (one-to-one with MFEM ex5)
//!
//! k·u + ∇p = f, −∇·u = g in Ω, −p = p̄ on ∂Ω
//! Exact: u = (−eˣ sin y, −eˣ cos y), p = eˣ sin y.
//! RT H(div) for velocity, L₂ for pressure.
//!
//! cargo run --example mfem_ex5_mixed_darcy -- -m data/star.mesh

use std::fs::File;
use std::io::Write;
use fem_assembly::mixed::{assemble_hdiv_l2_mixed, HDivL2DivIntegrator};
use fem_assembly::standard::VectorMassIntegrator;
use fem_assembly::{
    VectorAssembler, VectorBoundaryAssembler,
    boundary::vector_boundary::{VectorBdQpData, VectorBoundaryLinearIntegrator},
};
use fem_io::mfem::{read_mfem_file, write_mfem};
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::{SolverConfig, block::{BlockSystem, MinresSolver}};
use fem_space::{HDivSpace, L2Space, fe_space::FESpace};

fn main() {
    let args = parse_args();
    let mfem = read_mfem_file(args.mesh.as_deref().unwrap_or("../data/star.mesh")).unwrap();
    let mesh: Mesh<2> = mfem.mesh2d.unwrap();
    let dim = 2;
    let rl = ((10000.0 / mesh.n_elems() as f64).ln() / (2.0_f64).ln() / dim as f64).floor() as usize;
    let mesh = if rl > 0 { let mut m = mesh; for _ in 0..rl { m = refine_uniform(&m); } m } else { mesh };
    let u_sp = HDivSpace::new(mesh.clone(), args.order);
    let p_sp = L2Space::new(mesh, args.order);
    let n_u = u_sp.n_dofs(); let n_p = p_sp.n_dofs();
    println!("\ndim(R) = {n_u}\ndim(W) = {n_p}");

    let qo = args.order * 2 + 2;
    let mm = VectorAssembler::assemble_bilinear(&u_sp, &[&VectorMassIntegrator{alpha:1.0}], qo);
    let mut mb = assemble_hdiv_l2_mixed(&p_sp, &u_sp, &[&HDivL2DivIntegrator], qo);
    for v in &mut mb.values { *v *= -1.0; } // C++ B *= -1

    // RHS: natural BC −p = p_exact → ∫ (−p_exact)·(v·n) ds per boundary face DOF
    let tags: Vec<i32> = u_sp.mesh().unique_boundary_tags();
    let fu = if !tags.is_empty() {
        fn neg_p_exact(x: &[f64]) -> f64 { -p_exact(x) }
        let nf_int = HdivNormalFluxIntegrator { g: neg_p_exact };
        VectorBoundaryAssembler::assemble_boundary_linear(&u_sp, &[&nf_int], &tags, qo)
    } else {
        vec![0.0; n_u]
    };
    let gp = vec![0.0; n_p]; // g = 0 in 2D

    // Build and solve block system
    let flat = BlockSystem{a:mm, bt:mb.transpose(), b:mb, c:None}.to_flat_csr();
    let mut rhs = Vec::with_capacity(n_u + n_p);
    rhs.extend(fu); rhs.extend(gp);
    let mut x = vec![0.0; rhs.len()];
    let cfg = SolverConfig{rtol:1e-6, max_iter:2000, verbose:false, ..SolverConfig::default()};
    match MinresSolver::solve(&flat, &rhs, &mut x, &cfg) {
        Ok(r) => println!("\nMINRES: converged={}, iters={}, residual={:.3e}", r.converged, r.iterations, r.final_residual),
        Err(e) => println!("\nMINRES error: {e}"),
    }

    // Diagnostic: solution norms
    let sol_norm_u = fem_assembly::hdiv_error::compute_hdiv_l2_error(&u_sp, &x[..n_u], |_|[0.0,0.0]);
    let sol_norm_p = fem_assembly::hdiv_error::compute_l2_error_scalar(&p_sp, &x[n_u..], |_|0.0);
    println!("  ||u_h|| = {:.6e}  ||p_h|| = {:.6e}", sol_norm_u, sol_norm_p);

    // L2 errors
    let eu = fem_assembly::hdiv_error::compute_hdiv_l2_error(&u_sp, &x[..n_u], |x| [-(x[0].exp()*x[1].sin()),-(x[0].exp()*x[1].cos())]);
    let nu_ = sol_norm_u;
    let ep = fem_assembly::hdiv_error::compute_l2_error_scalar(&p_sp, &x[n_u..], |x| x[0].exp()*x[1].sin());
    let np_ = fem_assembly::hdiv_error::compute_l2_error_scalar(&p_sp, &x[n_u..], |_|0.0);
    println!("||u_h−u_ex||/||u_ex|| = {:.6e}", eu/nu_.max(1e-32));
    println!("||p_h−p_ex||/||p_ex|| = {:.6e}", ep/np_.max(1e-32));

    // Output
    let mut mf = File::create("ex5.mesh").unwrap();
    write_mfem(&mut mf, u_sp.mesh(), None).unwrap();
    let mut uf = File::create("sol_u.gf").unwrap();
    for &v in &x[..n_u] { writeln!(uf, "{:.14e}", v).unwrap(); }
    let mut pf = File::create("sol_p.gf").unwrap();
    for &v in &x[n_u..] { writeln!(pf, "{:.14e}", v).unwrap(); }
    eprintln!("  Wrote ex5.mesh, sol_u.gf, sol_p.gf");
}

fn p_exact(x: &[f64]) -> f64 { x[0].exp() * x[1].sin() }

/// ∫ g * (v·n) ds — H(div) natural boundary condition flux integrator.
/// 1:1 translation of MFEM's VectorFEBoundaryFluxLFIntegrator.
struct HdivNormalFluxIntegrator { g: fn(&[f64]) -> f64 }
impl VectorBoundaryLinearIntegrator for HdivNormalFluxIntegrator {
    fn add_to_face_vector(&self, qp: &VectorBdQpData, f_elem: &mut [f64]) {
        let g_val = (self.g)(&qp.x_phys);
        for i in 0..qp.n_dofs {
            let vn = qp.phi_vec[i*2]*qp.normal[0] + qp.phi_vec[i*2+1]*qp.normal[1];
            f_elem[i] += g_val * vn * qp.weight;
        }
    }
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
