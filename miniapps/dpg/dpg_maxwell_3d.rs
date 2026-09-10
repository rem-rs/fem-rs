//! 3D Maxwell DPG solver.
//!
//! ⚠️ **STATUS（第十一轮代理 H 复核）：这不是真 DPG。** 本文件是手写的
//! P1（节点）Galerkin 复系统 + `solve_cg_complex`，注释中的 "DPG" 名不副实：
//! `omega` 参数未使用、虚部恒 0、没有骨架 trace 未知量、没有 DPG 正规方程
//! `A = BᵀG⁻¹B`。与 C++ `miniapps/dpg/maxwell.cpp` 的 3D 分支（E,H ∈ (L²)³、
//! Ê,Ĥ ∈ `ND_Trace_FECollection(order,3)`、F,G ∈ `ND_FECollection(order+do,3)`、
//! 复块算子 + 伴随图范数测试空间）**无法数值对照**。
//!
//! 真替换的**剩余阻塞点**（详见 `tmp/dpg3d/FINDINGS.md`）：
//! 1. `dpg_weakform.rs` 只有标量面 Lagrange trace 支持（`eval_face_lagrange`
//!    + `FaceVals::phi`）；ND trace 需要 `FaceVals::vec_phi` 与面-棱共享编号。
//!    该文件归代理 G，本轮未改。
//! 2. 同一文件 `face_geo_at` 的 3D 面法向带 0.5 因子（MFEM `CalcOrtho` 无），
//!    3D 面 trace 积分整体差 2 倍。
//! 3. `HexNDk` 是 MFEM `ND_HexahedronElement` 的**另一组基**（[−1,1]³ +
//!    IntegratedGLL vs [0,1]³ + GaussLegendre）：空间相同（已用单测
//!    `dpg_basis::trace_tests::nd_hex_span_is_tensor_nedelec` 钉死），
//!    解不受影响，但单元矩阵不能逐位对照。
//!
//! **已完成的前置件**（本轮，均在 `crates/assembly/src/dpg/dpg_basis.rs`）：
//! `TraceSpace`（H1/RT/ND 三类 3D 骨架空间，面/棱实体表、全局 dof 编号、
//! 面内 dof 顺序与 `EncodeDof` 符号编码、面单元基值、切向协变物理映射）
//! 与 C++ 逐位一致；3D 所需积分器已核对补齐
//! （新增 `DpgTransposedMixedCurlIntegrator` = `(E,∇×F)`）。

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
