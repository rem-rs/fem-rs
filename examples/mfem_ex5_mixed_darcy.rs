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
use fem_assembly::VectorAssembler;
use fem_io::mfem::{read_mfem_file, write_mfem};
use fem_mesh::{refine_uniform, Mesh, MeshTopology};
use fem_solver::{SolverConfig, block::BlockSystem, solve_gmres};
use fem_space::{EdgeKey, HDivSpace, L2Space, fe_space::FESpace};

fn main() {
    let args = parse_args();
    let mfem = read_mfem_file(args.mesh.as_deref().unwrap_or("../data/star.mesh")).unwrap();
    let mesh: Mesh<2> = mfem.mesh2d.unwrap();
    let dim = 2;
    let rl = ((10000.0 / mesh.n_elems() as f64).ln() / (2.0_f64).ln() / dim as f64).floor() as usize;
    let mesh = if rl > 0 { let mut m = mesh; for _ in 0..rl { m = refine_uniform(&m); } m } else { mesh };
    let u_sp = HDivSpace::new(mesh.clone(), if args.order >= 1 { args.order - 1 } else { 0 });
    let p_sp = L2Space::new(mesh, args.order);
    let n_u = u_sp.n_dofs(); let n_p = p_sp.n_dofs();
    println!("\ndim(R) = {n_u}\ndim(W) = {n_p}");

    let qo = args.order * 2 + 2;
    let mm = VectorAssembler::assemble_bilinear(&u_sp, &[&VectorMassIntegrator{alpha:1.0}], qo);
    let mut mb = assemble_hdiv_l2_mixed(&p_sp, &u_sp, &[&HDivL2DivIntegrator], qo);
    for v in &mut mb.values { *v *= -1.0; }

    // RHS: natural BC −p = p_exact → ∫ p_exact · (v·n) ds
    let tags: Vec<i32> = u_sp.mesh().unique_boundary_tags();
    let mut fu = vec![0.0; n_u];
    if !tags.is_empty() {
        for f in 0..u_sp.mesh().n_boundary_faces() as u32 {
            if !tags.contains(&u_sp.mesh().face_tag(f)) { continue; }
            let n = u_sp.mesh().face_nodes(f);
            if n.len() < 2 { continue; }
            let pa = u_sp.mesh().node_coords(n[0]);
            let pb = u_sp.mesh().node_coords(n[1]);
            let tx = pb[0] - pa[0]; let ty = pb[1] - pa[1];
            let mid = [0.5*(pa[0]+pb[0]), 0.5*(pa[1]+pb[1])];
            let ek = EdgeKey::new(n[0], n[1]);
            if let Some(dof) = u_sp.edge_face_dof(ek) {
                let edge_len = (tx*tx + ty*ty).sqrt();
                // Boundary term: ∫ (−p) · (v·n) ds ≈ −p(mid)·edge_len
                fu[dof as usize] += -p_exact(&mid) * edge_len;
            }
        }
    }
    let gp = vec![0.0; n_p]; // g = 0 in 2D

    // Build and solve block system
    let flat = BlockSystem{a:mm, bt:mb.transpose(), b:mb, c:None}.to_flat_csr();
    let mut rhs = Vec::with_capacity(n_u + n_p);
    rhs.extend(fu); rhs.extend(gp);
    let mut x = vec![0.0; rhs.len()];
    let cfg = SolverConfig{rtol:1e-6, max_iter:2000, verbose:false, ..SolverConfig::default()};
    match solve_gmres(&flat, &rhs, &mut x, 50, &cfg) {
        Ok(r) => println!("\nGMRES: converged={}, iters={}, residual={:.3e}", r.converged, r.iterations, r.final_residual),
        Err(e) => println!("\nGMRES error: {e}"),
    }

    // L2 errors
    let eu = fem_assembly::hdiv_error::compute_hdiv_l2_error(&u_sp, &x[..n_u], |x| [-(x[0].exp()*x[1].sin()),-(x[0].exp()*x[0].cos())]);
    let nu_ = fem_assembly::hdiv_error::compute_hdiv_l2_error(&u_sp, &x[..n_u], |_|[0.0,0.0]);
    let ep = l2s(&p_sp, &x[n_u..], |x| x[0].exp()*x[1].sin());
    let np_ = l2s(&p_sp, &x[n_u..], |_|0.0);
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

fn l2s<F: Fn(&[f64])->f64>(s: &L2Space<Mesh<2>>, uh: &[f64], ex: F) -> f64 {
    use fem_element::{ReferenceElement, lagrange::TriP1};
    use fem_mesh::ElementTransformation;
    let m = s.mesh(); let r = TriP1; let q = r.quadrature(6); let n = r.n_dofs(); let mut p = vec![0.0; n]; let mut e2 = 0.0;
    for el in m.elem_iter() {
        let d: Vec<usize> = s.element_dofs(el).iter().map(|&d| d as usize).collect();
        let no = m.element_nodes(el); let tr = ElementTransformation::from_simplex_nodes(m, no); let dj = tr.det_j();
        for (qi, xi) in q.points.iter().enumerate() {
            let w = q.weights[qi] * dj.abs(); let xp = tr.map_to_physical(xi); r.eval_basis(xi, &mut p);
            let mut vh = 0.0; for i in 0..n { vh += uh[d[i]] * p[i]; } let ve = ex(&xp); e2 += w * (vh - ve) * (vh - ve);
        }
    }
    e2.sqrt()
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
