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
use fem_element::raviart_thomas::QuadRTk;
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
    let qr = QuadRTk::new(0).quadrature(2);
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

    // ── 2b. Hanging-flux constraints (MFEM true-dof semantics) ─────────────
    // On a non-conforming mesh the RT0 space has a slave flux DOF on each
    // fine half-edge: flux continuity requires `u_fine = ±0.5·u_coarse`
    // (same flux density over half the edge length).  MFEM eliminates these
    // slave DOFs via the conforming prolongation P (true-dof space); a plain
    // serial solve with the slave DOFs as independent unknowns does NOT
    // reproduce C++ at 2nd+ level hanging edges (pex6 it3: np1 marked 10 vs
    // C++ 25 — the flux field of the slave edges decouples from the master).
    // At 1st-level hanging edges the slave value happens to come out as
    // ±0.5·master automatically, which is why it0-it2 aligned without this
    // constraint.
    //
    // Enforce the constraint in the local (dm-order) mass matrix: the slave
    // row becomes `x_s − c·x_m = 0` (row s: 1 at s, −c at m; RHS 0).  The
    // slave/master pair is detected purely from topology+coordinates (any
    // edge whose midpoint is a mesh node is a master; its two halves are
    // slaves), which covers 2nd+ level edges (m2 = mid(a,m1) where m1 is
    // itself hanging — the parent edge (a,m1) need not be an element edge).
    let mut slave_deps: Vec<(u32, f64, u32)> = Vec::new(); // (slave, coef, master)
    let mut edge_of_dof: Vec<(u32, u32)> = vec![(u32::MAX, u32::MAX); n_rt_dofs];
    {
        // RT0 edge dof → endpoint node ids (local), via first element.
        for e in 0..n_elems as ElemId {
            let ns = mesh.elem_nodes(e);
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
            let c = mesh.coords_of(n);
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
        let mut master_edges: Vec<(u32, u32, u32)> = Vec::new(); // (a, b, mid)
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
        for d in 0..n_rt_dofs as u32 {
            let (a, b) = edge_of_dof[d as usize];
            if a == u32::MAX {
                continue;
            }
            for &(pa, pb, mid) in &master_edges {
                // slave edge endpoints = {master endpoint, midpoint}
                let (lo, hi) = (a.min(b), a.max(b));
                let is_slave = (lo == pa.min(pb) || lo == pa.max(pb) || lo == mid)
                    && (hi == pa.min(pb) || hi == pa.max(pb) || hi == mid)
                    && (lo == mid) != (hi == mid)
                    && lo != hi;
                if !is_slave {
                    continue;
                }
                if let Some(&md) = master_dof.get(&(pa, pb)) {
                    // sign: slave's low endpoint equals master's low endpoint
                    // → +0.5, else −0.5 (MFEM: one half-edge +0.5, other −0.5)
                    let sign = if lo == pa.min(pb) { 1.0 } else { -1.0 };
                    slave_deps.push((d, 0.5 * sign, md));
                }
            }
        }
        slave_deps.sort_by_key(|x| (x.0, x.2));
        slave_deps.dedup();
    }
    let a = a;
    let mut x = vec![0.0_f64; n_rt_dofs];
    if !slave_deps.is_empty() && std::env::var("L2ZZ_NOCONSTRAINT").is_err() {
        // ── true-dof elimination (MFEM PᵀAP semantics) ─────────────────────
        // MFEM's estimator solves the L2 projection on the *true-dof* space:
        // `A_true = Pᵀ A P`, `b_true = Pᵀ b` with P the conforming
        // prolongation (slave flux dof s = ±0.5·master, chained at 2nd+ level
        // where the master is itself a slave).  A row-replacement
        // approximation (slave row → x_s − c·x_m = 0) is NOT equivalent: it
        // drops the A_sm/A_ss/b_s contributions folded into the master rows
        // by PᵀAP, which only goes unnoticed while the slave values happen to
        // equal ±0.5·master anyway (1st-level edges).  pex6 it3 (2nd-level
        // hanging edges): row-replacement marked 40 vs C++ 25, PᵀAP should
        // match C++.
        let slave_rows: std::collections::HashSet<u32> =
            slave_deps.iter().map(|(s, _, _)| *s).collect();
        // Free (true) dofs, in ascending order — their index in the true
        // space is their position here.
        let free_dofs: Vec<u32> = (0..n_rt_dofs as u32)
            .filter(|d| !slave_rows.contains(d))
            .collect();
        // Chain-expand each slave to its ultimate free master (2nd+ level:
        // master may itself be a slave).
        let master_of: std::collections::HashMap<u32, (u32, f64)> = slave_deps
            .iter()
            .map(|&(s, c, m)| (s, (m, c)))
            .collect();
        let mut p_entries: Vec<(u32, usize, f64)> = Vec::new(); // (full dof, true idx, coef)
        for (i, &f) in free_dofs.iter().enumerate() {
            p_entries.push((f, i, 1.0));
        }
        for &(s, c0, m0) in &slave_deps {
            let mut coef = c0;
            let mut cur = m0;
            let mut guard = 0;
            while let Some(&(m, c)) = master_of.get(&cur) {
                coef *= c;
                cur = m;
                guard += 1;
                assert!(guard < 64, "hanging-flux dependency cycle");
            }
            let idx = free_dofs.binary_search(&cur).expect("slave chain ends at slave");
            p_entries.push((s, idx, coef));
        }
        // A_true = Pᵀ A P, b_true = Pᵀ b.
        let n_true = free_dofs.len();
        let mut coo_true = CooMatrix::new(n_true, n_true);
        let mut b_true = vec![0.0_f64; n_true];
        for &(p_row, p_ti, p_v) in &p_entries {
            for k in a.row_ptr[p_row as usize]..a.row_ptr[p_row as usize + 1] {
                let col = a.col_idx[k] as usize;
                let av = a.values[k];
                // column of A is a full dof — find its true index via P
                if let Some((q_ti, q_v)) = p_entries.iter().find(|&&(r, _, _)| r as usize == col).map(|&(_, ti, v)| (ti, v)) {
                    coo_true.add(p_ti, q_ti, p_v * av * q_v);
                }
            }
            // b_true = Pᵀ b: only free rows of b contribute through P (slave
            // rows of b are folded with their coef — b itself has no slave
            // rows, it is assembled per-element into all dofs; MFEM Pᵀb sums
            // b_master + c·b_slave).
            let bv = b[p_row as usize];
            if bv != 0.0 {
                b_true[p_ti] += p_v * bv;
            }
        }
        let a_true = coo_true.into_csr_sorted();
        let mut y = vec![0.0_f64; n_true];
        let cfg = SolverConfig {
            rtol: 1e-12,
            max_iter: 200,
            verbose: false,
            ..SolverConfig::default()
        };
        solve_pcg_gssmoother(&a_true, &b_true, &mut y, &cfg)
            .expect("RT0 L2 projection solve failed");
        // x = P y
        for &(p_row, p_ti, p_v) in &p_entries {
            x[p_row as usize] += p_v * y[p_ti];
        }
    } else {
        let cfg = SolverConfig {
            rtol: 1e-12,
            max_iter: 200,
            verbose: false,
            ..SolverConfig::default()
        };
        solve_pcg_gssmoother(&a, &b, &mut x, &cfg).expect("RT0 L2 projection solve failed");
    }

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
