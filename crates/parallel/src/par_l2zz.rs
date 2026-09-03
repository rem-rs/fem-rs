//! Parallel L2-projection Zienkiewicz–Zhu error estimator.
//!
//! Parallel counterpart of [`fem_assembly::postproc::l2_zz_rt1::l2_zz_estimator`]
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
use fem_element::raviart_thomas::QuadRTk;
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
    hanging_edges: &[(u32, u32, u32)],
    creation_order: &std::collections::HashMap<u32, u32>,
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
    // `new_for_edge_space_ordered`: after an AMR partition rebuild the node
    // gids are NOT in MFEM's UpdateVertices creation order, which would flip
    // the RT0 edge orientation (`ga < gb` test) and negate the recovered
    // flux (np2 pex6: the whole RT0 solution comes out negated → eta/marking
    // drift).  Measuring the orientation against the creation order keeps it
    // rank-invariant and MFEM-consistent.
    let rt_par = ParallelFESpace::new_for_edge_space_ordered(
        rt_local.clone(),
        par_mesh,
        comm.clone(),
        Some(creation_order),
    );
    let dp = rt_par.dof_partition();
    let n_total_dofs = dp.n_total_dofs();
    let n_owned_dofs = dp.n_owned_dofs;
    let n_rt_dofs = rt_local.n_dofs();

    // 4-point Gauss-Legendre rule on [0,1]² (same as the serial estimator).
    let qr = QuadRTk::new(0).quadrature(2);
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

            QuadRTk::new(0).eval_basis_vec(xi, &mut phi_ref);
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

    // ── 2b. Hanging-flux constraints (MFEM true-dof semantics) ─────────────
    //  On a non-conforming mesh the RT0 space has a slave flux DOF on each
    //  fine half-edge: flux continuity requires `u_fine = ±0.5·u_coarse`
    //  (same flux density over half the edge length).  MFEM eliminates these
    //  slave DOFs via the conforming prolongation P (true-dof space); Rust's
    //  HDivSpace keeps them as independent DOFs, so the parallel assembly
    //  across a hanging edge does not match C++ (pex6 deep-water: np2 it2
    //  marked 12 vs np1/C++ 40).
    //
    //  Enforce the constraint in the LOCAL (dm-order) mass matrix without
    //  changing the DOF count (DofPartition/permutation/global assembly stay
    //  untouched): the slave row becomes `x_s − c·x_m = 0` (row s: 1 at s,
    //  −c at m; RHS 0); free rows keep the original equations.  The solution
    //  then satisfies u_s = c·u_m by construction, identical to MFEM's
    //  true-dof solution on the free DOFs.  The slave/master pair always
    //  lives on the same rank (both half-edges and the coarse edge are local
    //  once the ghost layer is complete), so this is purely local and
    //  consistent across ranks.
    let mut slave_deps: Vec<(u32, f64, u32)> = Vec::new(); // (slave, coef, master)
    let mut edge_of_dof: Vec<(u32, u32)> = vec![(u32::MAX, u32::MAX); n_rt_dofs];
    let partition = par_mesh.partition();
    {
        // RT0 edge dof → endpoint node ids (local), via first element.
        for e in 0..n_local_elems as fem_core::ElemId {
            let ns = local_mesh.elem_nodes(e);
            for (li, (ia, ib)) in [(0usize, 1usize), (1, 2), (2, 3), (3, 0)]
                .iter()
                .enumerate()
            {
                let d = elem_rt_dofs[e as usize][li] as usize;
                if edge_of_dof[d] == (u32::MAX, u32::MAX) {
                    edge_of_dof[d] = (ns[*ia], ns[*ib]);
                }
            }
        }
        let coords = |n: u32| -> [f64; 2] {
            let c = local_mesh.coords_of(n);
            [c[0], c[1]]
        };
        // 1. Master edges: edge (a,b) with some other node m at its midpoint.
        let mut node_list: Vec<u32> = edge_of_dof
            .iter()
            .flat_map(|&(a, b)| [a, b])
            .filter(|&n| n != u32::MAX)
            .collect();
        node_list.sort_unstable();
        node_list.dedup();
        let mut master_edges: Vec<(u32, u32, u32)> = Vec::new(); // (a, b, mid), LOCAL ids
        // (a) Globally-merged hanging edges (from the caller): convert global
        // node ids → local ids.  Ranks lacking the coarse element (nodes not
        // in the ghost layer) simply skip that edge — the rank holding it
        // imposes the constraint; from_local_matrix folds across ranks.
        {
            let partition = par_mesh.partition();
            for &(ga, gb, gmid) in hanging_edges {
                if let (Some(a), Some(b), Some(m)) = (
                    partition.local_node(ga),
                    partition.local_node(gb),
                    partition.local_node(gmid),
                ) {
                    master_edges.push((a.min(b), a.max(b), m));
                }
            }
        }
        // (b) Local geometric scan (covers edges the caller's global merge
        //     missed, e.g. single-rank or no-hanging_edges path).
        for &(a, b) in edge_of_dof.iter() {
            if a == u32::MAX {
                continue;
            }
            let (mx, my) = {
                let ca = coords(a);
                let cb = coords(b);
                (0.5 * (ca[0] + cb[0]), 0.5 * (ca[1] + cb[1]))
            };
            for &m in &node_list {
                if m == a || m == b {
                    continue;
                }
                let cm = coords(m);
                if (cm[0] - mx).abs() < 1e-9 && (cm[1] - my).abs() < 1e-9 {
                    master_edges.push((a.min(b), a.max(b), m));
                    break;
                }
            }
        }
        master_edges.sort_unstable();
        master_edges.dedup_by(|x, y| x.0 == y.0 && x.1 == y.1);
        // 2. Slave half-edge dofs: (a,m) or (m,b) of a master (a,b,m).
        let master_dof: std::collections::HashMap<(u32, u32), u32> = master_edges
            .iter()
            .filter_map(|&(a, b, _m)| {
                edge_of_dof
                    .iter()
                    .position(|&(x, y)| x.min(y) == a && x.max(y) == b)
                    .map(|i| ((a, b), i as u32))
            })
            .collect();
        // Slave sign measured in the MFEM `UpdateVertices` node creation
        // order (`creation_order`: gid → order, threaded across AMR rounds
        // by the caller — a partition rebuild renumbers local ids per rank,
        // so the local-id test `lo == pa.min(pb)` flips across ranks, np2
        // it3: same physical slave edge gets +1 on one rank, −1 on the
        // other → marked 1 vs np1/C++ 25).  The coarse edge's "min"
        // endpoint is the one created first.
        let part = par_mesh.partition();
        let gid_of = |n: u32| part.global_node(n);
        for d in 0..n_rt_dofs as u32 {
            let (a, b) = edge_of_dof[d as usize];
            if a == u32::MAX {
                continue;
            }
            // Slave detection on LOCAL ids (the endpoint *set* is rank
            // invariant even though the ordering is not).
            let (lo, hi) = (a.min(b), a.max(b));
            for &(pa, pb, mid) in &master_edges {
                // slave edge endpoints = {master endpoint, midpoint}
                let is_slave = (lo == pa.min(pb) || lo == pa.max(pb) || lo == mid)
                    && (hi == pa.min(pb) || hi == pa.max(pb) || hi == mid)
                    && (lo == mid) != (hi == mid)
                    && lo != hi;
                if !is_slave {
                    continue;
                }
                if let Some(&md) = master_dof.get(&(pa, pb)) {
                    // sign: +1 when the slave half-edge departs from the
                    // coarse edge's first-created (MFEM "min") endpoint,
                    // −1 otherwise.
                    let (gpa, gpb) = (gid_of(pa), gid_of(pb));
                    let min_end = if creation_order[&gpa] < creation_order[&gpb] {
                        gpa
                    } else {
                        gpb
                    };
                    let coarse_end = if gid_of(lo) == gpa || gid_of(lo) == gpb {
                        gid_of(lo)
                    } else {
                        gid_of(hi)
                    };
                    let sign = if coarse_end == min_end { 1.0 } else { -1.0 };
                    slave_deps.push((d, 0.5 * sign, md));
                }
            }
        }
        slave_deps.sort_by_key(|x| (x.0, x.2));
        slave_deps.dedup();
    }
    let mut permuted_a = permute_csr(&local_a, dp);
    let mut permuted_b = permute_vec(&b, dp);
    let mut a_mat = ParCsrMatrix::from_local_matrix(
        &permuted_a,
        n_owned_dofs,
        rt_par.dof_ghost_exchange_arc(),
        comm.clone(),
    );
    let mut rhs = ParVector::from_local_raw(
        permuted_b,
        n_owned_dofs,
        rt_par.dof_ghost_exchange_arc(),
        comm.clone(),
    );
    if !slave_deps.is_empty() {
        // RT0 hanging-flux constraints as MFEM true-dof elimination
        // (PᵀKP / Pᵀf with cross-rank folding): each slave flux DOF is
        // constrained to ±0.5·master.  A row-replacement (slave row →
        // x_s−c·x_m=0) is NOT equivalent at 2nd+ level hanging edges (the
        // A_sm/A_ss/b_s contributions folded into the master rows are
        // dropped; serial pex6 it3: row-replacement marked 40 vs PᵀAP 25 =
        // C++).
        use fem_mesh::amr::HangingNodeConstraint;
        let constraints: Vec<HangingNodeConstraint> = slave_deps
            .iter()
            .map(|&(s, c, m)| {
                // Physical flux-continuity coefficient directly: the RT0
                // DOF value IS the physical flux (∫σ·n̂ ds), so the
                // permuted-space constraint is x_s^p = c·x_m^p with no
                // per-rank sign correction.  Multiplying by
                // sign_s·sign_m would depend on which element first saw
                // each DOF locally — a per-rank choice — and flip the
                // weight on ranks whose local node ordering differs
                // (np2 pex6 it2: sign_m=-1 on one rank vs +1 on the
                // serial mesh → PᵀKP wrong → marked 5 vs 40).
                HangingNodeConstraint::new_weighted(
                    s as usize,
                    m as usize,
                    m as usize, // unused second parent (coeff 0)
                    c,
                    0.0,
                    Vec::new(),
                )
            })
            .collect();
        a_mat.apply_hanging_constraints(&constraints, &mut rhs, dp);
    }
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
    let res = par_solve_pcg_amg(&a_mat, &rhs, &mut x, &amg_cfg, &cfg)
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
    // Slave flux values: apply_hanging_constraints eliminates slave DOFs in
    // the true-dof system (their rows become identity with 0 RHS, so the
    // solver leaves them 0); recover u_s = c0·u_master (chained at 2nd+ level
    // where the master is itself a slave), in dm order.
    //
    // The recovery runs in the local (dm) basis: x_dm = S·x_p, so the
    // physical constraint x_s^p = c·x_m^p becomes
    // x_s_dm = c·S_s·S_m·x_m_dm (S = per-DOF sign_correction, ±1).  On the
    // serial mesh S=1 for the constrained DOFs and c alone is right; after a
    // partition rebuild the local node ordering flips S_m on some ranks and
    // the missing S product negates the recovered slave flux (np2 pex6 it2:
    // elements next to a hanging edge get η ≈ 6.9e-3 vs 2.9e-4 → marked 9).
    if !slave_deps.is_empty() {
        let master_of: std::collections::HashMap<u32, (u32, f64)> = slave_deps
            .iter()
            .map(|&(s, c, m)| (s, (m, c)))
            .collect();
        for &(s, c0, m0) in &slave_deps {
            let sign_s = if needs_sign {
                dp.sign_correction(s)
            } else {
                1.0
            };
            let sign_m = if needs_sign {
                dp.sign_correction(m0)
            } else {
                1.0
            };
            let mut coef = c0 * sign_s * sign_m;
            let mut cur = m0;
            let mut guard = 0;
            while let Some(&(m, c)) = master_of.get(&cur) {
                let sign_cur = if needs_sign {
                    dp.sign_correction(cur)
                } else {
                    1.0
                };
                let sign_m2 = if needs_sign {
                    dp.sign_correction(m)
                } else {
                    1.0
                };
                coef *= c * sign_cur * sign_m2;
                cur = m;
                guard += 1;
                assert!(guard < 64, "hanging-flux dependency cycle");
            }
            x_dm[s as usize] = coef * x_dm[cur as usize];
        }
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
