//! 3D IGA Poisson: -Δu = 1 on unit cube, u=0 on ∂Ω.
use fem_assembly::iga::{assemble_iga_diffusion_3d, assemble_iga_load_3d};
use fem_element::iga::NurbsKnotVector;
use fem_element::nurbs::NurbsMesh3D;
use fem_solver::{SolverConfig, solve_cg};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let nu: usize = a.iter().position(|x| x=="--nu").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(6);
    let nv: usize = a.iter().position(|x| x=="--nv").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(6);
    let nw: usize = a.iter().position(|x| x=="--nw").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(6);
    let p: usize = a.iter().position(|x| x=="--p").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(1);
    let kv = NurbsKnotVector::uniform(p, nu-p);
    let mut ctrl = Vec::with_capacity(nu*nv*nw);
    for k in 0..nw { for j in 0..nv { for i in 0..nu { ctrl.push([i as f64/(nu-1) as f64, j as f64/(nv-1) as f64, k as f64/(nw-1) as f64]); }}}
    let mesh = NurbsMesh3D::single_patch(kv.clone(), kv.clone(), kv.clone(), ctrl, vec![1.0; nu*nv*nw]);
    let stiff = assemble_iga_diffusion_3d(&mesh, 1.0, 3);
    let rhs = assemble_iga_load_3d(&mesh, &|_: &[f64]| 1.0, 3);
    let n = stiff.nrows; let mut u = vec![0.0; n];
    solve_cg(&stiff, &rhs, &mut u, &SolverConfig{rtol:1e-8,max_iter:20000,..Default::default()}).unwrap();
    let norm: f64 = u.iter().map(|x|x*x).sum::<f64>().sqrt();
    println!("IGA Poisson 3D: dofs={n} sol_norm={norm:.6e}");
}
