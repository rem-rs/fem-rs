//! 3D IGA Poisson: -Δu = 1 on unit cube, u=0 on ∂Ω.
//!
//! Also provides an MMS smoke test under `#[cfg(test)]`.
use fem_assembly::iga::{assemble_iga_diffusion_3d, assemble_iga_load_3d};
use fem_element::iga::NurbsKnotVector;
use fem_element::nurbs::NurbsMesh3D;
use fem_solver::{SolverConfig, solve_cg};
use std::f64::consts::PI;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iga_poisson_3d_mms() {
        let p = 2;
        let nu = 6; let nv = 6; let nw = 6;
        let kv = NurbsKnotVector::uniform(p, nu-p);
        let mut ctrl = Vec::with_capacity(nu*nv*nw);
        for k in 0..nw { for j in 0..nv { for i in 0..nu {
            ctrl.push([i as f64/(nu-1) as f64, j as f64/(nv-1) as f64, k as f64/(nw-1) as f64]);
        }}}
        let mesh = NurbsMesh3D::single_patch(kv.clone(), kv.clone(), kv.clone(), ctrl, vec![1.0; nu*nv*nw]);
        let mut stiff = assemble_iga_diffusion_3d(&mesh, 1.0, 4);
        let n = stiff.nrows;

        // Source: f = 3π²·sin(πx)sin(πy)sin(πz) with manufactured u = sin(πx)sin(πy)sin(πz)
        let src = |x: &[f64]| 3.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin() * (PI * x[2]).sin();
        let mut rhs = assemble_iga_load_3d(&mesh, &src, 4);

        // Symmetric Dirichlet elimination: u=0 on all 6 NURBS patch faces
        let mut is_bnd = vec![false; n];
        for k in 0..nw { for j in 0..nv { for i in 0..nu {
            let idx = k * nu * nv + j * nu + i;
            if i == 0 || i == nu-1 || j == 0 || j == nv-1 || k == 0 || k == nw-1 {
                is_bnd[idx] = true;
            }
        }}}
        for d in 0..n {
            if is_bnd[d] {
                // Zero column entries (symmetrically eliminate)
                for i in 0..n {
                    if i == d { continue; }
                    for p in stiff.row_ptr[i]..stiff.row_ptr[i+1] {
                        if stiff.col_idx[p] as usize == d {
                            stiff.values[p] = 0.0;
                        }
                    }
                }
                // Zero row entries + set diagonal
                for p in stiff.row_ptr[d]..stiff.row_ptr[d+1] {
                    let col = stiff.col_idx[p] as usize;
                    stiff.values[p] = if col == d { 1.0 } else { 0.0 };
                }
                rhs[d] = 0.0;
            }
        }

        let mut u = vec![0.0; n];
        solve_cg(&stiff, &rhs, &mut u,
            &SolverConfig{rtol:1e-8,max_iter:5000,..Default::default()})
            .expect("IGA 3D CG failed");

        let norm: f64 = u.iter().map(|x|x*x).sum::<f64>().sqrt();
        eprintln!("  [IGA 3D Poisson MMS] p={}, n_ctrl={}, ||u||={:.6e}", p, n, norm);
        assert!(norm > 0.0 && norm < 10.0, "||u||={:.6e} outside range", norm);

        fem_regression::regression("iga_poisson_3d_mms")
            .check_with("l2_norm", norm, 1e-6, 1e-10)
            .check_with("n_dofs", n as f64, 1e-6, 0.5)
            .finalize();
    }
}
