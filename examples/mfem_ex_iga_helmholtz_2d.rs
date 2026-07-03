//! 2D IGA Helmholtz: -Δu + k²u = f on unit square, u=0 on ∂Ω.
use fem_assembly::iga::{assemble_iga_diffusion_2d, assemble_iga_mass_2d, assemble_iga_load_2d};
use fem_element::iga::NurbsKnotVector;
use fem_element::nurbs::NurbsMesh2D;
use fem_linalg::CsrMatrix;
use fem_solver::{SolverConfig, solve_cg};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let nu: usize = a.iter().position(|x| x=="--nu").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(8);
    let nv: usize = a.iter().position(|x| x=="--nv").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(8);
    let p: usize = a.iter().position(|x| x=="--p").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(2);
    let k: f64 = a.iter().position(|x| x=="--k").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(5.0);
    let kv = NurbsKnotVector::uniform(p, nu - p);
    let mut ctrl = Vec::with_capacity(nu*nv);
    for j in 0..nv { for i in 0..nu { ctrl.push([i as f64/(nu-1) as f64, j as f64/(nv-1) as f64]); }}
    let mesh = NurbsMesh2D::single_patch(kv.clone(), kv.clone(), ctrl, vec![1.0; nu*nv]);
    let stiff = assemble_iga_diffusion_2d(&mesh, 1.0, 4);
    let mass = assemble_iga_mass_2d(&mesh, k*k, 4);
    let a: CsrMatrix<f64> = stiff.add(&mass);
    let rhs = assemble_iga_load_2d(&mesh, &|x: &[f64]| (std::f64::consts::PI*x[0]).sin()*(std::f64::consts::PI*x[1]).sin(), 4);
    let n = a.nrows; let mut u = vec![0.0; n];
    solve_cg(&a, &rhs, &mut u, &SolverConfig{rtol:1e-10,max_iter:20000,..Default::default()}).unwrap();
    let norm: f64 = u.iter().map(|x|x*x).sum::<f64>().sqrt();
    println!("IGA Helmholtz 2D: dofs={n} k={k} sol_norm={norm:.6e}");
}
