//! 3D Maxwell DPG solver.

use fem_linalg::complex_csr::{ComplexCoo, ComplexCsr};
use fem_solver::complex_ams::solve_cg_complex;

fn build_maxwell_dpg_3d(
    nx: usize, ny: usize, nz: usize,
    omega: f64, j_re: f64, j_im: f64,
) -> (ComplexCsr, Vec<f64>, Vec<f64>) {
    let hx = 1.0 / nx as f64;
    let hy = 1.0 / ny as f64;
    let hz = 1.0 / nz as f64;
    let n_vertices = (nx + 1) * (ny + 1) * (nz + 1);
    let n_e = 3 * n_vertices;
    let n_h = 3 * n_vertices;
    let n_trial = n_e + n_h;
    let quad_pts = [-0.5773502691896258, 0.5773502691896258];
    let quad_wts = [1.0, 1.0];
    let mut a_coo = ComplexCoo::new(n_trial, n_trial);
    let mut b_re = vec![0.0f64; n_trial];
    let mut b_im = vec![0.0f64; n_trial];

    for i in 0..n_e {
        let vtx = i / 3;
        let ix = vtx % (nx + 1);
        let iy = (vtx / (nx + 1)) % (ny + 1);
        let iz = vtx / ((nx + 1) * (ny + 1));
        if ix == 0 || ix == nx || iy == 0 || iy == ny || iz == 0 || iz == nz {
            a_coo.add(i, i, 1.0, 0.0);
        }
    }

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let v000 = iz * (nx + 1) * (ny + 1) + iy * (nx + 1) + ix;
                let v100 = v000 + 1;
                let v110 = v000 + (nx + 1) + 1;
                let v010 = v000 + (nx + 1);
                let v001 = v000 + (nx + 1) * (ny + 1);
                let v101 = v001 + 1;
                let v111 = v001 + (nx + 1) + 1;
                let v011 = v001 + (nx + 1);
                let e_dofs = [3*v000, 3*v100, 3*v110, 3*v010, 3*v001, 3*v101, 3*v111, 3*v011];
                let h_dofs = [n_e + 3*v000, n_e + 3*v100, n_e + 3*v110, n_e + 3*v010, n_e + 3*v001, n_e + 3*v101, n_e + 3*v111, n_e + 3*v011];

                for i in 0..8 {
                    for comp in 0..3 {
                        let e_row = e_dofs[i] + comp;
                        let on_boundary = ix == 0 || ix == nx || iy == 0 || iy == ny || iz == 0 || iz == nz;
                        for j in 0..8 {
                            let mut a_val_re = 0.0f64;
                            for (xi, wi) in quad_pts.iter().zip(quad_wts.iter()) {
                                for (eta, wj) in quad_pts.iter().zip(quad_wts.iter()) {
                                    for (zeta, wk) in quad_pts.iter().zip(quad_wts.iter()) {
                                        let w = *wi * *wj * *wk;
                                        let jac = hx * hy * hz / 8.0;
                                        let term = w * jac;
                                        a_val_re += term;
                                    }
                                }
                            }
                            if !on_boundary {
                                a_coo.add(e_row, e_dofs[j] + comp, a_val_re, 0.0);
                            }
                            a_coo.add(e_row, h_dofs[j] + comp, a_val_re, 0.0);
                        }
                        if !on_boundary {
                            let mut b_val_re = 0.0f64;
                            for (xi, wi) in quad_pts.iter().zip(quad_wts.iter()) {
                                for (eta, wj) in quad_pts.iter().zip(quad_wts.iter()) {
                                    for (zeta, wk) in quad_pts.iter().zip(quad_wts.iter()) {
                                        let w = *wi * *wj * *wk;
                                        let jac = hx * hy * hz / 8.0;
                                        b_val_re += w * jac * j_re;
                                    }
                                }
                            }
                            b_re[e_row] += b_val_re;
                        }
                    }
                }
            }
        }
    }
    let a = a_coo.into_complex_csr();
    (a, b_re, b_im)
}

fn main() {
    let nx = 2; let ny = 2; let nz = 2;
    let omega = 2.0 * std::f64::consts::PI;
    let j_re = 1.0; let j_im = 0.0;
    println!("Solving 3D Maxwell DPG");
    let (a, b_re, b_im) = build_maxwell_dpg_3d(nx, ny, nz, omega, j_re, j_im);
    println!("  System size: {} DOFs", a.nrows);
    let mut x_re = vec![0.0f64; a.nrows];
    let mut x_im = vec![0.0f64; a.nrows];
    match solve_cg_complex(&a, &b_re, &b_im, &mut x_re, &mut x_im, 1e-10, 500) {
        Ok((iters, res)) => println!("  Converged in {} iterations, residual = {:.6e}", iters, res),
        Err(e) => eprintln!("  Solver failed: {}", e),
    }
}
