//! L2-projection Zienkiewicz–Zhu error estimator — 1:1 port of MFEM's
//! `L2ZZErrorEstimator` (fem/pgridfunc.cpp), validated bit-for-bit against
//! MFEM ex6p on `star.mesh` (it0/it1 element errors match C++ exactly).
//!
//! Algorithm (matches MFEM exactly, all on the `[0,1]²` reference quad):
//! 1. The discontinuous flux `σ_h = ∇u_h` — a **Q1 bilinear field** per
//!    element (NOT constant: the quad solution basis is bilinear), evaluated
//!    with the correct geometric Jacobian at the L2 flux-space nodes.
//! 2. L2-project `σ_h` into the smooth H(div) RT0 space: solve `A x = b`
//!    with `A_ij = ∫ φ_i · φ_j` (MFEM `VectorFEMassIntegrator`) and
//!    `b_i = ∫ φ_i · σ_h` (`VectorFEDomainLFIntegrator`), PCG tol 1e-12,
//!    max 200 (C++: HyprePCG + BoomerAMG).
//! 3. Per-element error `η_K = ∫_K |σ_h − Qσ_h|₂ dx` — MFEM
//!    `ComputeElementLpDistance` with the `L2ZienkiewiczZhuEstimator` default
//!    `local_norm_p = 1` (an L1 norm of the pointwise L2 distance, NO square
//!    root — using the L2 norm instead scales η by ~√∫ and changes marking).
//!
//! Quadrature: the RT0/L2 spaces live on `[0,1]²` (MFEM intrules use the 1-D
//! Gauss points 0.2113…/0.7887…; `Geometry::SQUARE` integrates on `[0,1]²`).
//! `QuadRT0::quadrature(2)` gives 2 points/dim (4 points) — the same rule
//! MFEM uses for `VectorFEMassIntegrator` (order `1+1+1 = 3`), the flux load
//! (order 2) and the error integral (order `2·1+1 = 3`).
//!
//! Currently supports 2-D **Quad4** meshes (ex6p / pex6 use `star.mesh`).

use fem_core::ElemId;
use fem_element::raviart_thomas::QuadRT0;
use fem_element::VectorReferenceElement;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_solver::{SolverConfig, solve_pcg_gssmoother};
use fem_space::{FESpace, HDivSpace};

/// Element-wise L2(→RT0) ZZ error indicators `η_K` for a P1 solution `u`
/// (one value per **node**, length `n_nodes`) on a Quad4 mesh.
///
/// Returns one `η_K` per element.
pub fn l2_zz_estimator(mesh: &fem_mesh::Mesh<2>, u: &[f64]) -> Vec<f64> {
    assert_eq!(
        mesh.element_type_at(0),
        fem_mesh::element_type::ElementType::Quad4,
        "l2_zz_estimator currently supports Quad4 meshes only"
    );
    let n_elems = mesh.n_elems();

    // ── 0. RT0 space (dof layout + orientation signs) ────────────────────────
    let rt: HDivSpace<fem_mesh::Mesh<2>> = HDivSpace::new(mesh.clone(), 0);
    let n_rt_dofs = rt.n_dofs();

    // 4-point Gauss-Legendre rule on [0,1]² (MFEM IntRules for the SQUARE:
    // 1-D points 0.2113…/0.7887…, 2 points/dim).  Everything — the Q1 flux
    // gradient, the RT0 basis and the geometry Jacobian — lives on [0,1]².
    let qr = QuadRT0.quadrature(2);
    // Q1 (H1) shape-function gradients on [0,1]² (MFEM solution basis):
    //   N0=(1-x)(1-y) N1=x(1-y) N2=xy N3=(1-x)y
    let dn = |x: f64, y: f64| -> [[f64; 2]; 4] {
        [
            [-(1.0 - y), -(1.0 - x)],
            [1.0 - y, -x],
            [y, x],
            [-y, 1.0 - x],
        ]
    };

    // ── 1. Element loop: element matrices, flux gradients, error integrands ──
    let mut coo = Vec::<(usize, usize, f64)>::new();
    let mut b = vec![0.0_f64; n_rt_dofs];
    let mut phi_qp = vec![vec![0.0_f64; 4 * 8]; n_elems]; // [e][qp*8 + i*2 + c]
    let mut grad_qp = vec![vec![[0.0_f64; 2]; 4]; n_elems]; // per QP flux
    let mut wdet_qp = vec![vec![0.0_f64; 4]; n_elems]; // weight·|detJ|
    let mut elem_rt_dofs = Vec::<Vec<u32>>::with_capacity(n_elems);

    for e in 0..n_elems as ElemId {
        let nodes = mesh.elem_nodes(e);
        let c = |i: usize| mesh.coords_of(nodes[i]);
        let ue = [
            u[nodes[0] as usize],
            u[nodes[1] as usize],
            u[nodes[2] as usize],
            u[nodes[3] as usize],
        ];
        let rt_dofs: Vec<u32> = rt.element_dofs(e).to_vec();
        let signs: Vec<f64> = rt.element_signs(e).to_vec();
        elem_rt_dofs.push(rt_dofs.clone());

        let mut phi = [0.0_f64; 8]; // physical RT0 basis at current QP
        let mut phi_ref = [0.0_f64; 8];
        for (q, xi) in qr.points.iter().enumerate() {
            let (x, y) = (xi[0], xi[1]);
            // Bilinear Q1 mapping Jacobian at (x, y) ∈ [0,1]² (MFEM geometry).
            let j00 = -(1.0 - y) * c(0)[0] + (1.0 - y) * c(1)[0] + y * c(2)[0] - y * c(3)[0];
            let j01 = -(1.0 - x) * c(0)[0] - x * c(1)[0] + x * c(2)[0] + (1.0 - x) * c(3)[0];
            let j10 = -(1.0 - y) * c(0)[1] + (1.0 - y) * c(1)[1] + y * c(2)[1] - y * c(3)[1];
            let j11 = -(1.0 - x) * c(0)[1] - x * c(1)[1] + x * c(2)[1] + (1.0 - x) * c(3)[1];
            let det = j00 * j11 - j01 * j10;
            let inv_det = 1.0 / det;

            // Physical gradient ∇u_h = J^{-T} · Σ_j u_j ∇N_j(x, y) ([0,1]² Q1).
            let d = dn(x, y);
            let g_ref0 = ue[0] * d[0][0] + ue[1] * d[1][0] + ue[2] * d[2][0] + ue[3] * d[3][0];
            let g_ref1 = ue[0] * d[0][1] + ue[1] * d[1][1] + ue[2] * d[2][1] + ue[3] * d[3][1];
            let gx = (j11 * g_ref0 - j10 * g_ref1) * inv_det;
            let gy = (-j01 * g_ref0 + j00 * g_ref1) * inv_det;
            grad_qp[e as usize][q] = [gx, gy];

            // RT0 physical basis via contravariant Piola + orientation signs
            // on the [0,1]² reference domain (MFEM RT0Quad / intrules).
            QuadRT0.eval_basis_vec(xi, &mut phi_ref);
            for i in 0..4 {
                let s = signs[i];
                phi[i * 2] = (j00 * phi_ref[i * 2] + j01 * phi_ref[i * 2 + 1]) * inv_det * s;
                phi[i * 2 + 1] = (j10 * phi_ref[i * 2] + j11 * phi_ref[i * 2 + 1]) * inv_det * s;
            }
            let w = qr.weights[q] * det.abs();
            wdet_qp[e as usize][q] = w;
            for i in 0..4 {
                phi_qp[e as usize][q * 8 + i * 2] = phi[i * 2];
                phi_qp[e as usize][q * 8 + i * 2 + 1] = phi[i * 2 + 1];
            }

            // Mass + load contributions at this QP.
            for i in 0..4 {
                for j in 0..4 {
                    let dot = phi[i * 2] * phi[j * 2] + phi[i * 2 + 1] * phi[j * 2 + 1];
                    coo.push((rt_dofs[i] as usize, rt_dofs[j] as usize, w * dot));
                }
            }
            for i in 0..4 {
                b[rt_dofs[i] as usize] += w * (phi[i * 2] * gx + phi[i * 2 + 1] * gy);
            }
        }
    }

    // ── 2. Global RT0 mass matrix + PCG solve (L2 projection) ────────────────
    let mut cm = CooMatrix::new(n_rt_dofs, n_rt_dofs);
    for (i, j, v) in coo {
        cm.add(i, j, v);
    }
    let a = cm.into_csr_sorted();
    let mut x = vec![0.0_f64; n_rt_dofs];
    let cfg = SolverConfig {
        rtol: 1e-12,
        max_iter: 200,
        verbose: false,
        ..SolverConfig::default()
    };
    solve_pcg_gssmoother(&a, &b, &mut x, &cfg).expect("RT0 L2 projection solve failed");

    // ── 3. Per-element error (MFEM `ComputeElementLpDistance` with the
    //        L2ZienkiewiczZhuEstimator default `local_norm_p = 1`):
    //        η_K = ∫_K |σ_h − Qσ_h|₂ dx  (L1, no square root) ────────────────
    let mut eta = vec![0.0_f64; n_elems];
    for e in 0..n_elems as ElemId {
        let rt_dofs = &elem_rt_dofs[e as usize];
        let mut err = 0.0;
        for q in 0..4 {
            let w = wdet_qp[e as usize][q];
            let [gx, gy] = grad_qp[e as usize][q];
            let phi = &phi_qp[e as usize][q * 8..q * 8 + 8];
            let mut sx = 0.0;
            let mut sy = 0.0;
            for i in 0..4 {
                let xv = x[rt_dofs[i] as usize];
                sx += xv * phi[i * 2];
                sy += xv * phi[i * 2 + 1];
            }
            let (dx, dy) = (gx - sx, gy - sy);
            err += w * (dx * dx + dy * dy).sqrt();
        }
        eta[e as usize] = err;
    }
    eta
}
