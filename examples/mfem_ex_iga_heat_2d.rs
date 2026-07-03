//! 2D IGA heat: ∂u/∂t-κΔu=0, implicit Euler. Usage: --nu 12 --nv 12 --p 1 --dt 0.01 --steps 10
use fem_assembly::iga::{assemble_iga_diffusion_2d, assemble_iga_mass_2d};
use fem_element::iga::NurbsKnotVector;
use fem_element::nurbs::NurbsMesh2D;
use fem_solver::{SolverConfig, solve_cg};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let nu: usize = a.iter().position(|x|x=="--nu").and_then(|i|a.get(i+1)).and_then(|s|s.parse().ok()).unwrap_or(12);
    let nv: usize = a.iter().position(|x|x=="--nv").and_then(|i|a.get(i+1)).and_then(|s|s.parse().ok()).unwrap_or(12);
    let p: usize = a.iter().position(|x|x=="--p").and_then(|i|a.get(i+1)).and_then(|s|s.parse().ok()).unwrap_or(1);
    let dt: f64 = a.iter().position(|x|x=="--dt").and_then(|i|a.get(i+1)).and_then(|s|s.parse().ok()).unwrap_or(0.01);
    let ns: usize = a.iter().position(|x|x=="--steps").and_then(|i|a.get(i+1)).and_then(|s|s.parse().ok()).unwrap_or(10);
    let kv = NurbsKnotVector::uniform(p, nu-p);
    let mut ctrl = Vec::with_capacity(nu*nv);
    for j in 0..nv { for i in 0..nu { ctrl.push([i as f64/(nu-1) as f64, j as f64/(nv-1) as f64]); }}
    let mesh = NurbsMesh2D::single_patch(kv.clone(), kv.clone(), ctrl, vec![1.0; nu*nv]);
    let mass = assemble_iga_mass_2d(&mesh, 1.0, 3);
    let dt_stiff = assemble_iga_diffusion_2d(&mesh, dt, 3); // kappa = dt
    let sys = mass.add(&dt_stiff);
    let n = sys.nrows;
    let mut y = vec![0.0_f64; n];
    let mut u = vec![0.0_f64; n];
    for j in 0..nv { for i in 0..nu {
        u[j*nu+i] = (std::f64::consts::PI*i as f64/(nu-1) as f64).sin()*(std::f64::consts::PI*j as f64/(nv-1) as f64).sin();
    }}
    for step in 1..=ns {
        mass.spmv(&u, &mut y);
        solve_cg(&sys, &y, &mut u, &SolverConfig{rtol:1e-8,max_iter:10000,..Default::default()}).unwrap();
        println!("  step {step}/{ns}: ||u||={:.6e}", u.iter().map(|x|x*x).sum::<f64>().sqrt());
    }
    println!("IGA Heat 2D: dofs={n} dt={dt} steps={ns}");
}
