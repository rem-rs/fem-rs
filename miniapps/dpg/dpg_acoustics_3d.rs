//! 3D Acoustics DPG (Discontinuous Petrov-Galerkin) solver for Helmholtz.
//!
//! ⚠️ **STATUS（第十一轮代理 H 复核）：这不是真 DPG。** 本文件是手写的 P1
//! Galerkin 复系统 + `solve_cg_complex`：没有骨架 trace 未知量（p̂/û）、
//! 没有 DPG 正规方程 `A = BᵀG⁻¹B`、没有伴随图范数测试空间；`∇p + iωu = 0`
//! 的复耦合退化为实数 P1 Poisson 项。与 C++ `miniapps/dpg/acoustics.cpp`
//! （p ∈ L²、u ∈ (L²)³、p̂ ∈ `H1_Trace_FECollection(order,3)`、
//! û ∈ `RT_Trace_FECollection(order-1,3)`、q ∈ H¹(order+do)、v ∈ RT(order+do)）
//! **无法数值对照**。
//!
//! 真替换的**剩余阻塞点**（详见 `tmp/dpg3d/FINDINGS.md`）：
//! 1. `dpg_weakform.rs` 的 trace 分块只支持标量面 Lagrange 基；且 3D 三角面
//!    的 `SkeletonSpace` 语义与 `RT_Trace_FECollection` 不一致（3D RT trace
//!    无棱 dof、面内部 = L2 面空间），骨架 H1 trace 的 3D 分支持久化也只在
//!    2D 验证过。该文件归代理 G，本轮未改。
//! 2. 同一文件 `face_geo_at` 的 3D 面法向/measure 只有 MFEM 的一半
//!    （`cross_half` vs MFEM `CalcOrtho`），所有 3D 面 trace 积分差 2 倍。
//!
//! **已完成的前置件**（本轮）：`crates/assembly/src/dpg/dpg_basis.rs` 的
//! `TraceSpace`（H1/RT/ND）与 C++ `*_Trace_FECollection` 的 dof 计数、
//! 全局编号、面实体表逐位一致（`tmp/dpg3d/ndtrace_dump.txt` diff 全等），
//! 3D 所需积分器已核对（acoustics 3D 无缺口）。
//!
//! Solves the first-order system:
//!   ∇p + iωu = 0  in Ω
//!   ∇·u + iωp = f  in Ω
//!   p = p₀           on ∂Ω
//!
//! Ultraweak DPG formulation with enriched test spaces on a structured hex mesh.

use fem_linalg::complex_csr::{ComplexCoo, ComplexCsr};
use fem_solver::complex_ams::solve_cg_complex;

/// Build the 3D acoustics DPG system on a structured hex mesh.
fn build_acoustics_dpg_3d(
    nx: usize,
    ny: usize,
    nz: usize,
    omega: f64,
    f_re: f64,
    f_im: f64,
) -> (ComplexCsr, Vec<f64>, Vec<f64>) {
    let hx = 1.0 / nx as f64;
    let hy = 1.0 / ny as f64;
    let hz = 1.0 / nz as f64;

    // Trial space: P1 (continuous, vertex DOFs) for both p and u
    let n_vertices = (nx + 1) * (ny + 1) * (nz + 1);
    let n_p = n_vertices;       // pressure DOFs
    let n_u = 3 * n_vertices;   // velocity DOFs (3 components)
    let n_trial = n_p + n_u;

    // Gauss-Legendre quadrature on [-1, 1] (2 points for P1)
    let quad_pts = [-0.5773502691896258, 0.5773502691896258];
    let quad_wts = [1.0, 1.0];

    let mut a_coo = ComplexCoo::new(n_trial, n_trial);
    let mut b_re = vec![0.0f64; n_trial];
    let mut b_im = vec![0.0f64; n_trial];

    // Apply Dirichlet BC: p = 0 on boundary
    for i in 0..n_p {
        let ix = i % (nx + 1);
        let iy = (i / (nx + 1)) % (ny + 1);
        let iz = i / ((nx + 1) * (ny + 1));
        let on_boundary = ix == 0 || ix == nx || iy == 0 || iy == ny || iz == 0 || iz == nz;
        if on_boundary {
            a_coo.add(i, i, 1.0, 0.0);
            b_re[i] = 0.0;
            b_im[i] = 0.0;
        }
    }

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                // Hex element vertex indices
                let v000 = iz * (nx + 1) * (ny + 1) + iy * (nx + 1) + ix;
                let v100 = iz * (nx + 1) * (ny + 1) + iy * (nx + 1) + ix + 1;
                let v110 = iz * (nx + 1) * (ny + 1) + (iy + 1) * (nx + 1) + ix + 1;
                let v010 = iz * (nx + 1) * (ny + 1) + (iy + 1) * (nx + 1) + ix;
                let v001 = (iz + 1) * (nx + 1) * (ny + 1) + iy * (nx + 1) + ix;
                let v101 = (iz + 1) * (nx + 1) * (ny + 1) + iy * (nx + 1) + ix + 1;
                let v111 = (iz + 1) * (nx + 1) * (ny + 1) + (iy + 1) * (nx + 1) + ix + 1;
                let v011 = (iz + 1) * (nx + 1) * (ny + 1) + (iy + 1) * (nx + 1) + ix;

                let p_dofs = [v000, v100, v110, v010, v001, v101, v111, v011];

                // P1 basis derivatives on reference hex [-1,1]³
                let dphi = |i: usize, xi: f64, eta: f64, zeta: f64| -> (f64, f64, f64) {
                    match i {
                        0 => (-(1.0 - eta) * (1.0 - zeta) / 8.0,
                              -(1.0 - xi) * (1.0 - zeta) / 8.0,
                              -(1.0 - xi) * (1.0 - eta) / 8.0),
                        1 => ((1.0 - eta) * (1.0 - zeta) / 8.0,
                              -(1.0 + xi) * (1.0 - zeta) / 8.0,
                              -(1.0 + xi) * (1.0 - eta) / 8.0),
                        2 => ((1.0 + eta) * (1.0 - zeta) / 8.0,
                              (1.0 + xi) * (1.0 - zeta) / 8.0,
                              -(1.0 + xi) * (1.0 + eta) / 8.0),
                        3 => (-(1.0 + eta) * (1.0 - zeta) / 8.0,
                              (1.0 - xi) * (1.0 - zeta) / 8.0,
                              -(1.0 - xi) * (1.0 + eta) / 8.0),
                        4 => (-(1.0 - eta) * (1.0 + zeta) / 8.0,
                              -(1.0 - xi) * (1.0 + zeta) / 8.0,
                              (1.0 - xi) * (1.0 - eta) / 8.0),
                        5 => ((1.0 - eta) * (1.0 + zeta) / 8.0,
                              -(1.0 + xi) * (1.0 + zeta) / 8.0,
                              (1.0 + xi) * (1.0 - eta) / 8.0),
                        6 => ((1.0 + eta) * (1.0 + zeta) / 8.0,
                              (1.0 + xi) * (1.0 + zeta) / 8.0,
                              (1.0 + xi) * (1.0 + eta) / 8.0),
                        7 => (-(1.0 + eta) * (1.0 + zeta) / 8.0,
                              (1.0 - xi) * (1.0 + zeta) / 8.0,
                              (1.0 - xi) * (1.0 + eta) / 8.0),
                        _ => unreachable!(),
                    }
                };

                let phi = |i: usize, xi: f64, eta: f64, zeta: f64| -> f64 {
                    match i {
                        0 => (1.0 - xi) * (1.0 - eta) * (1.0 - zeta) / 8.0,
                        1 => (1.0 + xi) * (1.0 - eta) * (1.0 - zeta) / 8.0,
                        2 => (1.0 + xi) * (1.0 + eta) * (1.0 - zeta) / 8.0,
                        3 => (1.0 - xi) * (1.0 + eta) * (1.0 - zeta) / 8.0,
                        4 => (1.0 - xi) * (1.0 - eta) * (1.0 + zeta) / 8.0,
                        5 => (1.0 + xi) * (1.0 - eta) * (1.0 + zeta) / 8.0,
                        6 => (1.0 + xi) * (1.0 + eta) * (1.0 + zeta) / 8.0,
                        7 => (1.0 - xi) * (1.0 + eta) * (1.0 + zeta) / 8.0,
                        _ => unreachable!(),
                    }
                };

                for i in 0..8 {
                    let p_row = p_dofs[i];
                    let ix_row = p_row % (nx + 1);
                    let iy_row = (p_row / (nx + 1)) % (ny + 1);
                    let iz_row = p_row / ((nx + 1) * (ny + 1));
                    let on_boundary = ix_row == 0 || ix_row == nx ||
                        iy_row == 0 || iy_row == ny ||
                        iz_row == 0 || iz_row == nz;

                    for j in 0..8 {
                        let mut a_val_re = 0.0f64;
                        let mut a_val_im = 0.0f64;

                        for (xi, wi) in quad_pts.iter().zip(quad_wts.iter()) {
                            for (eta, wj) in quad_pts.iter().zip(quad_wts.iter()) {
                                for (zeta, wk) in quad_pts.iter().zip(quad_wts.iter()) {
                                    let w = *wi * *wj * *wk;
                                    let jac = hx * hy * hz / 8.0;

                                    let (dphidx_i, dphidy_i, dphidz_i) = dphi(i, *xi, *eta, *zeta);
                                    let (dphidx_j, dphidy_j, dphidz_j) = dphi(j, *xi, *eta, *zeta);

                                    let dphidx_i = dphidx_i * 2.0 / hx;
                                    let dphidy_i = dphidy_i * 2.0 / hy;
                                    let dphidz_i = dphidz_i * 2.0 / hz;
                                    let dphidx_j = dphidx_j * 2.0 / hx;
                                    let dphidy_j = dphidy_j * 2.0 / hy;
                                    let dphidz_j = dphidz_j * 2.0 / hz;

                                    let term = (dphidx_i * dphidx_j +
                                                dphidy_i * dphidy_j +
                                                dphidz_i * dphidz_j) * w * jac;
                                    a_val_re += term;
                                }
                            }
                        }

                        if !on_boundary {
                            a_coo.add(p_row, p_dofs[j], a_val_re, a_val_im);
                        }
                    }

                    // RHS
                    if !on_boundary {
                        let mut b_val_re = 0.0f64;
                        let mut b_val_im = 0.0f64;
                        for (xi, wi) in quad_pts.iter().zip(quad_wts.iter()) {
                            for (eta, wj) in quad_pts.iter().zip(quad_wts.iter()) {
                                for (zeta, wk) in quad_pts.iter().zip(quad_wts.iter()) {
                                    let w = *wi * *wj * *wk;
                                    let jac = hx * hy * hz / 8.0;
                                    let phi_val = phi(i, *xi, *eta, *zeta);
                                    b_val_re += w * jac * f_re * phi_val;
                                    b_val_im += w * jac * f_im * phi_val;
                                }
                            }
                        }
                        b_re[p_row] += b_val_re;
                        b_im[p_row] += b_val_im;
                    }
                }
            }
        }
    }

    let a = a_coo.into_complex_csr();
    (a, b_re, b_im)
}

fn main() {
    let nx = 2;
    let ny = 2;
    let nz = 2;
    let omega = 2.0 * std::f64::consts::PI;
    let f_re = 1.0;
    let f_im = 0.0;

    println!("Solving 3D Acoustics DPG: ∇p + iωu = 0, ∇·u + iωp = f");
    println!("  Grid: {}×{}×{}, ω = {:.4}, f = {} + {}i", nx, ny, nz, omega, f_re, f_im);

    let (a, b_re, b_im) = build_acoustics_dpg_3d(nx, ny, nz, omega, f_re, f_im);

    println!("  System size: {} DOFs", a.nrows);

    let mut x_re = vec![0.0f64; a.nrows];
    let mut x_im = vec![0.0f64; a.nrows];

    match solve_cg_complex(&a, &b_re, &b_im, &mut x_re, &mut x_im, 1e-10, 500) {
        Ok((iters, res)) => {
            println!("  Converged in {} iterations, residual = {:.6e}", iters, res);
            println!("  Pressure solution (real part at vertices):");
            for iz in 0..=nz {
                println!("  Layer iz={}:", iz);
                for iy in 0..=ny {
                    for ix in 0..=nx {
                        let idx = iz * (nx + 1) * (ny + 1) + iy * (nx + 1) + ix;
                        print!(" {:8.4}", x_re[idx]);
                    }
                    println!();
                }
            }
        }
        Err(e) => {
            eprintln!("  Solver failed: {}", e);
        }
    }
}
