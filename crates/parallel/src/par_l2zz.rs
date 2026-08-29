//! Parallel L2-projection Zienkiewicz–Zhu error estimator.
//!
//! Parallel counterpart of [`fem_assembly::postproc::l2_zz::l2_zz_estimator`]
//! (itself a 1:1 port of MFEM's `L2ZienkiewiczZhuEstimator`, validated
//! bit-for-bit against ex6p on `star.mesh`).
//!
//! Why a parallel version is needed (pex6 deep-water):
//! the serial estimator solves the RT0 L2-projection **locally on each rank's
//! local mesh**.  On a partitioned mesh the RT0 space is not the global one —
//! cross-rank face DOFs appear as independent unknowns per rank, so the
//! recovered smooth flux `Qσ_h` differs from C++'s *global* projection and
//! the element error indicators (and hence the AMR marking) drift
//! (`np2 it2`: marked 12 vs np1/C++ 40).
//!
//! This version assembles the RT0 mass matrix and load **across ranks**:
//!  1. The local `HDivSpace` (RT0) is built on the local mesh (owned + ghost
//!     elements), exactly like the serial estimator's element loop.
//!  2. The element matrix/vector contributions are permuted into the parallel
//!     `[owned|ghost]` DOF ordering (`DofPartition::permute_dof` +
//!     `sign_correction`, same machinery as [`crate::ParAssembler`]).
//!  3. `par_solve_pcg_amg` solves the *global* RT0 system; ghost DOFs are
//!     refreshed from their owning ranks.
//!  4. Per-element errors are computed from the global projection (the exact
//!     same quadrature/math as the serial estimator, so `np1` is unchanged).
//!
//! The mesh must be a 2-D **Quad4** mesh (same restriction as the serial
//! estimator).  Hanging-node RT0 flux continuity on non-conforming meshes is
//! left to the underlying `HDivSpace` (identical to the serial path, which is
//! validated against C++ on star.mesh through `it2`).

use std::sync::Arc;

use fem_core::Rank;
use fem_element::raviart_thomas::QuadRT0;
use fem_element::VectorReferenceElement;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::Mesh;
use fem_solver::SolverConfig;
use fem_space::{FESpace, HDivSpace};

use crate::comm::Comm;
use crate::dof_partition::DofPartition;
use crate::par_amg::{ParAmgConfig, SmootherType, par_solve_pcg_amg};
use crate::par_assembler::{permute_csr, permute_vec};
use crate::par_csr::ParCsrMatrix;
use crate::par_mesh::ParallelMesh;
use crate::par_space::ParallelFESpace;
use crate::par_vector::ParVector;

/// Element-wise L2(→RT0) ZZ error indicators for a P1 solution `u`
/// (one value per **node**, length `n_nodes`) on a partitioned Quad4 mesh.
///
/// Returns one `η_K` per **local** element (owned + ghost, same order as the
/// local mesh's element array — `[0, n_owned_elems)` are the owned elements).
///
/// The RT0 L2 projection is solved **globally** (cross-rank mass matrix +
/// load, `par_solve_pcg_amg`), matching C++'s global
/// `L2ZienkiewiczZhuEstimator` — this is what keeps `np2` marking aligned
/// with `np1`/C++ in the AMR loop.
pub fn l2_zz_estimator_parallel(
    par_mesh: &ParallelMesh<Mesh<2>>,
    comm: &Comm,
    u_dm: &[f64],
) -> Vec<f64> {
    let local_mesh = par_mesh.local_mesh();
    assert_eq!(
        local_mesh.element_type_at(0),
        fem_mesh::element_type::ElementType::Quad4,
        "l2_zz_estimator_parallel currently supports Quad4 meshes only"
    );
    let n_local_elems = local_mesh.n_elems();

    // ── 0. Parallel RT0 space (dof layout + orientation signs) ─────────────
    // The local space is the serial HDivSpace on the local mesh; the parallel
    // wrapper partitions its DOFs across ranks (cross-rank face DOFs are
    // owned by the lowest rank and mirrored as ghosts elsewhere).
    let rt_local: HDivSpace<Mesh<2>> = HDivSpace::new(local_mesh.clone(), 0);
    let rt_par = ParallelFESpace::new_for_edge_space(rt_local.clone(), par_mesh, comm.clone());
    let dp = rt_par.dof_partition();
    let n_total_dofs = dp.n_total_dofs();
    let n_owned_dofs = dp.n_owned_dofs;
    let n_rt_dofs = rt_local.n_dofs();

    // 4-point Gauss-Legendre rule on [0,1]² (same as the serial estimator).
    let qr = QuadRT0.quadrature(2);
    let dn = |x: f64, y: f64| -> [[f64; 2]; 4] {
        [
            [-(1.0 - y), -(1.0 - x)],
            [1.0 - y, -x],
            [y, x],
            [-y, 1.0 - x],
        ]
    };

    // ── 1. Element loop: same math as the serial estimator ─────────────────
    // Local (dm-order) COO entries: the parallel system is assembled by
    // permuting these into partition order afterwards (permute_csr also
    // applies the RT0 orientation sign corrections).
    let mut coo = Vec::<(usize, usize, f64)>::new();
    let mut b = vec![0.0_f64; n_rt_dofs];
    let mut phi_qp = vec![vec![0.0_f64; 4 * 8]; n_local_elems]; // [e][qp*8 + i*2 + c]
    let mut grad_qp = vec![vec![[0.0_f64; 2]; 4]; n_local_elems]; // per QP flux
    let mut wdet_qp = vec![vec![0.0_f64; 4]; n_local_elems]; // weight·|detJ|
    let mut elem_rt_dofs = Vec::<Vec<u32>>::with_capacity(n_local_elems);

    for e in 0..n_local_elems as fem_core::ElemId {
        let nodes = local_mesh.elem_nodes(e);
        let c = |i: usize| local_mesh.coords_of(nodes[i]);
        let ue = [
            u_dm[nodes[0] as usize],
            u_dm[nodes[1] as usize],
            u_dm[nodes[2] as usize],
            u_dm[nodes[3] as usize],
        ];
        let rt_dofs: Vec<u32> = rt_local.element_dofs(e).to_vec();
        let signs: Vec<f64> = rt_local.element_signs(e).to_vec();
        elem_rt_dofs.push(rt_dofs.clone());

        let mut phi = [0.0_f64; 8]; // physical RT0 basis at current QP
        let mut phi_ref = [0.0_f64; 8];
        for (q, xi) in qr.points.iter().enumerate() {
            let (x, y) = (xi[0], xi[1]);
            let j00 = -(1.0 - y) * c(0)[0] + (1.0 - y) * c(1)[0] + y * c(2)[0] - y * c(3)[0];
            let j01 = -(1.0 - x) * c(0)[0] - x * c(1)[0] + x * c(2)[0] + (1.0 - x) * c(3)[0];
            let j10 = -(1.0 - y) * c(0)[1] + (1.0 - y) * c(1)[1] + y * c(2)[1] - y * c(3)[1];
            let j11 = -(1.0 - x) * c(0)[1] - x * c(1)[1] + x * c(2)[1] + (1.0 - x) * c(3)[1];
            let det = j00 * j11 - j01 * j10;
            let inv_det = 1.0 / det;

            let d = dn(x, y);
            let g_ref0 = ue[0] * d[0][0] + ue[1] * d[1][0] + ue[2] * d[2][0] + ue[3] * d[3][0];
            let g_ref1 = ue[0] * d[0][1] + ue[1] * d[1][1] + ue[2] * d[2][1] + ue[3] * d[3][1];
            let gx = (j11 * g_ref0 - j10 * g_ref1) * inv_det;
            let gy = (-j01 * g_ref0 + j00 * g_ref1) * inv_det;
            grad_qp[e as usize][q] = [gx, gy];

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

    // ── 2. Global RT0 mass matrix + load (cross-rank) + PCG/AMG solve ──────
    let mut cm = CooMatrix::new(n_rt_dofs, n_rt_dofs);
    for (i, j, v) in coo {
        cm.add(i, j, v);
    }
    let local_a = cm.into_csr_sorted();
    let permuted_a = permute_csr(&local_a, dp);
    let a_mat = ParCsrMatrix::from_local_matrix(
        &permuted_a,
        n_owned_dofs,
        rt_par.dof_ghost_exchange_arc(),
        comm.clone(),
    );
    let permuted_b = permute_vec(&b, dp);
    let rhs = ParVector::from_local_raw(
        permuted_b,
        n_owned_dofs,
        rt_par.dof_ghost_exchange_arc(),
        comm.clone(),
    );
    let mut x = ParVector::zeros_like(&rhs);
    let amg_cfg = ParAmgConfig {
        smoother: SmootherType::SymmetricGaussSeidel,
        ..Default::default()
    };
    let cfg = SolverConfig {
        rtol: 1e-12,
        max_iter: 200,
        verbose: false,
        ..SolverConfig::default()
    };
    par_solve_pcg_amg(&a_mat, &rhs, &mut x, &amg_cfg, &cfg)
        .expect("parallel RT0 L2 projection solve failed");
    x.update_ghosts();

    // ── 3. Per-element error (same math as the serial estimator) ───────────
    // The global solution x is in partition order; map it back to dm order
    // (sign-corrected) so the per-element recovery uses the global flux
    // coefficients.
    let mut x_dm = vec![0.0_f64; n_rt_dofs];
    let needs_sign = dp.needs_sign_correction();
    for pid in 0..n_total_dofs {
        let dm = dp.unpermute_dof(pid as u32) as usize;
        let s = if needs_sign {
            dp.sign_correction(dm as u32)
        } else {
            1.0
        };
        x_dm[dm] = x.as_slice()[pid] * s;
    }

    let mut eta = vec![0.0_f64; n_local_elems];
    for e in 0..n_local_elems {
        let rt_dofs = &elem_rt_dofs[e];
        let mut err = 0.0;
        for q in 0..4 {
            let w = wdet_qp[e][q];
            let [gx, gy] = grad_qp[e][q];
            let phi = &phi_qp[e][q * 8..q * 8 + 8];
            let mut sx = 0.0;
            let mut sy = 0.0;
            for i in 0..4 {
                let xv = x_dm[rt_dofs[i] as usize];
                sx += xv * phi[i * 2];
                sy += xv * phi[i * 2 + 1];
            }
            let (dx, dy) = (gx - sx, gy - sy);
            err += w * (dx * dx + dy * dy).sqrt();
        }
        eta[e] = err;
    }
    eta
}
