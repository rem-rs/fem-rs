//! RT1 L2-projection Zienkiewicz–Zhu error estimator for P2 Quad4 solutions.
//!
//! 1:1 math with MFEM's `L2ZienkiewiczZhuEstimator` (used by `ex15p.cpp`):
//!   1. The discontinuous flux `σ_h = ∇u_h` — a Q2 field whose components
//!      live in the L2(order=2) flux space (∇(Q2) ⊂ P1⊗P2, P2⊗P1 per
//!      component), evaluated pointwise (MFEM `ComputeElementFlux` evaluates
//!      at the flux-space nodes; since σ_h ∈ flux space this equals the L²
//!      projection exactly).
//!   2. L2-project `σ_h` into the smooth H(div) RT1 space: solve `A x = b`
//!      with `A_ij = ∫ φ_i · φ_j` (`VectorFEMassIntegrator`) and
//!      `b_i = ∫ φ_i · σ_h` (`VectorFEDomainLFIntegrator`), PCG tol 1e-12,
//!      max 200 (C++: HyprePCG + BoomerAMG).
//!   3. Per-element error `η_K = ∫_K |σ_h − Qσ_h|₂ dx` — MFEM
//!      `ComputeElementLpDistance` with the L2ZZ default `local_norm_p = 1`
//!      (an L1 norm of the pointwise L2 distance, NO square root) on
//!      quadrature order `2·max(2,1)+1 = 5`.
//!
//! Everything lives on the `[0,1]²` reference square (MFEM `SQUARE`): the
//! bilinear geometry Jacobian `J_01` (same as `l2_zz.rs`), the QuadRT1 basis
//! (contravariant Piola), and the Q2 solution basis is evaluated through its
//! `[-1,1]²` implementation (`QuadQ2`) with the chain-rule factor 2 on the
//! reference gradients.
//!
//! The RT1 space on a non-conforming mesh has a **slave edge DOF pair** on
//! each fine half-edge: flux continuity makes the two fine-edge linear
//! traces a 2×2 combination of the coarse-edge pair (NOT the RT0 ±½
//! pointwise rule — the fine Gauss points sit at t = 1/4, 3/4 of the master
//! edge).  MFEM builds this via `GetTransferMatrix` (master/slave FE of the
//! same RT1 collection); the 2×2 matrices are derived analytically here and
//! applied as a conforming-P (PᵀAP) elimination, chained across levels.
//!
//! The module is deliberately **partition-agnostic**: callers supply the
//! global edge-DOF numbering and the element→global-id mapping, so the same
//! code serves the serial estimator and the cross-rank global assembly in
//! `fem_parallel::par_l2zz_rt1`.

use std::collections::HashMap;

use fem_core::ElemId;
use fem_element::lagrange::QuadQ2;
use fem_element::raviart_thomas::QuadRT1;
use fem_element::{ReferenceElement, VectorReferenceElement};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{Mesh, topology::MeshTopology};
use fem_solver::{SolverConfig, solve_pcg_dsmoother};

/// RT1 slave→master transfer matrices (fine edge = GL points of the coarse
/// edge at t = 1/4, 3/4, expressed in the coarse-edge linear basis).
///
/// `M_first`: slave half-edge `(p, m)` of master `(p, q)` split at `m`;
/// `M_second`: slave half-edge `(m, q)`.  Both in the global `min→max`
/// direction (a fine half-edge's min→max direction always agrees with the
/// master's, so no extra sign appears).
pub const RT1_M_FIRST: [[f64; 2]; 2] = [
    [0.5915063509461096, -0.0915063509461096],
    [0.3415063509461096, 0.15849364905389035],
];
pub const RT1_M_SECOND: [[f64; 2]; 2] = [
    [0.15849364905389035, 0.34150635094610965],
    [-0.0915063509461096, 0.5915063509461096],
];

/// One RT1 hanging-flux constraint: `slave_base + {0,1}` DOFs =
/// `M · (master_base + {0,1})` DOFs (2×2).
#[derive(Debug, Clone, Copy)]
pub struct Rt1Slave {
    pub slave_base: u32,
    pub master_base: u32,
    pub m: [[f64; 2]; 2],
}

/// Deterministic global edge numbering: `(min,max)` edge key → dof base
/// (2 consecutive DOFs).  Keys sorted ascending; base = index·2.
/// `edges` must contain the *same* set on every rank (the caller gathers
/// the global edge set).
pub fn build_edge_dof_map(edges: &[(u32, u32)]) -> HashMap<(u32, u32), u32> {
    let mut keys: Vec<(u32, u32)> = edges.iter().copied().collect();
    keys.sort_unstable();
    keys.dedup();
    keys.into_iter()
        .enumerate()
        .map(|(i, k)| (k, (i as u32) * 2))
        .collect()
}

/// Build the RT1 hanging constraints from a list of master edges
/// `(parent_a, parent_b, midpoint)` (global node ids, as returned by
/// `detect_hanging_quad` merged across ranks).
///
/// Handles **multi-level** hanging edges: a slave edge's immediate parent
/// may itself be a slave (its parent edge is no longer a current element
/// edge), in which case the constraint chains up to the ultimate current
/// (coarse) master edge, multiplying the 2×2 transfer matrices.
///
/// `coords` maps global node ids to coordinates, used to orient the master
/// and slave edges: the RT edge dofs are normal traces, so when the slave
/// edge's min→max normal opposes the master's, the transfer matrix picks up
/// a − sign (`M ·= −1`).
pub fn rt1_hanging_constraints(
    hang: &[(u32, u32, u32)],
    edge_dof: &HashMap<(u32, u32), u32>,
    coords: &HashMap<u32, [f64; 2]>,
) -> Vec<Rt1Slave> {
    // Normal of the edge (a,b) in the min→max tangent's CCW direction.
    fn normal(coords: &HashMap<u32, [f64; 2]>, a: u32, b: u32) -> [f64; 2] {
        let ca = coords[&a];
        let cb = coords[&b];
        let (tx, ty) = (cb[0] - ca[0], cb[1] - ca[1]);
        [-ty, tx]
    }
    // parent_of: (slave edge key) -> (parent edge key, transfer M).
    // A quad split is always at the geometric midpoint, so the half-edge
    // toward `lo` uses M_FIRST and toward `hi` M_SECOND.  The slave edge's
    // dofs run min→max; when the new midpoint's gid places the edge
    // backwards relative to the master direction, the M rows must be
    // swapped to match the slave dof order, and the whole M picks up the
    // sign of (normal_master · normal_slave).
    let swap_rows = |m: [[f64; 2]; 2]| [[m[1][0], m[1][1]], [m[0][0], m[0][1]]];
    let mut parent_of: HashMap<(u32, u32), ((u32, u32), [[f64; 2]; 2])> = HashMap::new();
    for &(pa, pb, mid) in hang {
        let (lo, hi) = (pa.min(pb), pa.max(pb));
        let parent = (lo, hi);
        let n_m = normal(coords, lo, hi);
        // First half-edge (toward lo): slave dof order min→max.
        let (s1a, s1b) = if lo < mid { (lo, mid) } else { (mid, lo) };
        let mut m1 = if lo < mid { RT1_M_FIRST } else { swap_rows(RT1_M_FIRST) };
        if normal(coords, s1a, s1b)[0] * n_m[0] + normal(coords, s1a, s1b)[1] * n_m[1] < 0.0 {
            m1 = [[-m1[0][0], -m1[0][1]], [-m1[1][0], -m1[1][1]]];
        }
        parent_of.insert((s1a, s1b), (parent, m1));
        // Second half-edge (toward hi).
        let (s2a, s2b) = if mid < hi { (mid, hi) } else { (hi, mid) };
        let mut m2 = if mid < hi { RT1_M_SECOND } else { swap_rows(RT1_M_SECOND) };
        if normal(coords, s2a, s2b)[0] * n_m[0] + normal(coords, s2a, s2b)[1] * n_m[1] < 0.0 {
            m2 = [[-m2[0][0], -m2[0][1]], [-m2[1][0], -m2[1][1]]];
        }
        parent_of.insert((s2a, s2b), (parent, m2));
    }
    let mut out: Vec<Rt1Slave> = Vec::new();
    let mut n_missing = 0usize;
    for (&edge, &slave_base) in edge_dof {
        let Some(&((mut cur), mut m)) = parent_of.get(&edge) else { continue };
        // Chain to the ultimate master that is a current element edge.
        let mut guard = 0;
        while !edge_dof.contains_key(&cur) {
            let Some(&((pp), mm)) = parent_of.get(&cur) else {
                break; // no further parent — leave as-is (should not happen)
            };
            m = mat_mul(m, mm);
            cur = pp;
            guard += 1;
            assert!(guard < 64, "RT1 hanging chain too deep");
        }
        if let Some(&master_base) = edge_dof.get(&cur) {
            out.push(Rt1Slave { slave_base, master_base, m });
        } else {
            n_missing += 1;
        }
    }
    if std::env::var("PEX15_DBG").is_ok() {
        eprintln!("[dbg-rt1] constraints: out={} chained_missing_master={} parent_of_keys={}",
            out.len(), n_missing, parent_of.len());
        // Do the parent_of slave edges exist as current edges?
        let in_edges = parent_of.keys().filter(|k| edge_dof.contains_key(k)).count();
        let not_in = parent_of.len() - in_edges;
        eprintln!("[dbg-rt1] parent_of slaves in edge_dof: {in_edges} / {} (missing {not_in})", parent_of.len());
        let sample: Vec<(u32, u32)> = parent_of.keys().filter(|k| !edge_dof.contains_key(k)).copied().take(3).collect();
        eprintln!("[dbg-rt1]   sample missing slave edges: {sample:?}");
    }
    out
}

/// Chain-level expansion: a slave whose (immediate) master is itself a
/// slave is resolved to its ultimate free master (the 2×2 matrices
/// multiply).  `slaves` is the output of [`rt1_hanging_constraints`]
/// together with the *skipped* level-2 candidates — build it from the full
/// `hang` list and pass every slave edge that exists in `edge_dof`.
///
/// The returned map `slave_base → (free_master_base, M)` has one entry per
/// constrained slave base.
pub fn expand_rt1_slaves(
    slaves: &[Rt1Slave],
) -> HashMap<u32, (u32, [[f64; 2]; 2])> {
    let mut master_of: HashMap<u32, (u32, [[f64; 2]; 2])> = HashMap::new();
    for s in slaves {
        master_of.insert(s.slave_base, (s.master_base, s.m));
    }
    let mut out: HashMap<u32, (u32, [[f64; 2]; 2])> = HashMap::new();
    for &s in slaves {
        let mut m = s.m;
        let mut cur = s.master_base;
        let mut guard = 0;
        while let Some(&(mm, mmat)) = master_of.get(&cur) {
            m = mat_mul(m, mmat);
            cur = mm;
            guard += 1;
            assert!(guard < 64, "RT1 hanging-flux dependency cycle");
        }
        out.entry(s.slave_base).or_insert((cur, m));
    }
    out
}

fn mat_mul(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]
}

/// Assemble the global RT1 mass matrix + load for a set of elements.
///
/// `elem_dofs[e]` = P2 H1 dof ids (dm order) of element `e`;
/// `u` = P2 solution values (dm order);
/// `elem_gid` maps a local element id to its global id (interior DOF base =
/// `gid·4 + n_edges·2`);
/// `edge_dof` = global edge map from [`build_edge_dof_map`];
/// `b` = global load vector (length `n_edges·2 + n_global_elems·4`),
/// accumulated in place.
///
/// Returns the COO triplets (global dof ids, 0-based).
pub fn assemble_rt1_system(
    mesh: &Mesh<2>,
    node_gid: &[u32],
    elem_dofs: &[Vec<u32>],
    u: &[f64],
    elems: &[ElemId],
    elem_gid: &dyn Fn(ElemId) -> u32,
    edge_dof: &HashMap<(u32, u32), u32>,
    n_edges_global: u32,
    b: &mut [f64],
) -> Vec<(u32, u32, f64)> {
    let n_int_base = n_edges_global * 2;
    let qr = QuadRT1.quadrature(8);
    let mut coo: Vec<(u32, u32, f64)> = Vec::new();
    if std::env::var("PEX15_DBG").is_ok() {
        let mut min_det = f64::MAX;
        let mut min_e: ElemId = 0;
        for &e in elems {
            let nodes = mesh.elem_nodes(e);
            let c = |i: usize| mesh.coords_of(nodes[i]);
            let j00 = -(1.0 - 0.5) * c(0)[0] + (1.0 - 0.5) * c(1)[0]
                + 0.5 * c(2)[0] - 0.5 * c(3)[0];
            let j01 = -(1.0 - 0.5) * c(0)[0] - 0.5 * c(1)[0]
                + 0.5 * c(2)[0] + (1.0 - 0.5) * c(3)[0];
            let j10 = -(1.0 - 0.5) * c(0)[1] + (1.0 - 0.5) * c(1)[1]
                + 0.5 * c(2)[1] - 0.5 * c(3)[1];
            let j11 = -(1.0 - 0.5) * c(0)[1] - 0.5 * c(1)[1]
                + 0.5 * c(2)[1] + (1.0 - 0.5) * c(3)[1];
            let d = (j00 * j11 - j01 * j10).abs();
            if d < min_det {
                min_det = d;
                min_e = e;
            }
        }
        eprintln!("[dbg-rt1] elem min|det| = {min_det:.6e} at elem {min_e} (gid {})", elem_gid(min_e));
    }

    for &e in elems {
        let gid = elem_gid(e);
        let nodes = mesh.elem_nodes(e);
        if std::env::var("PEX15_DBG").is_ok() && (gid == 40 || gid == 41 || gid == 183) {
            eprintln!("[dbg-rt1] assemble elem {e} gid {gid} nodes {nodes:?} n_int_base={n_int_base} interior_base={}",
                n_int_base + gid * 4);
        }
        let c = |i: usize| mesh.coords_of(nodes[i]);
        let ue: Vec<f64> = elem_dofs[e as usize]
            .iter()
            .map(|&d| u[d as usize])
            .collect();
        // Local edge (li,lj) → global direction + sign + dof pos flip.
        // QuadRT1 reference edge dofs run along the local edge (li→lj):
        // dof {0,1} = Gauss points A,B along that direction.  Global dofs
        // run along min→max; when the local direction opposes it, the pair
        // is reversed and both components change sign (opposite outward
        // normal) — MFEM `DofOrderForOrientation` + `EncodeDof`.
        let mut gd = [0u32; 12]; // global dof ids, 0 = invalid
        let mut gs = [0.0f64; 12]; // sign
        for (li, lj) in [(0usize, 1usize), (1, 2), (2, 3), (3, 0)] {
            let (ga, gb) = (node_gid[nodes[li] as usize], node_gid[nodes[lj] as usize]);
            let key = (ga.min(gb), ga.max(gb));
            let base = edge_dof[&key];
            let same_dir = ga < gb; // global min→max direction (independent of the local edge (li,lj));
            let li2 = if same_dir { 0 } else { 1 };
            let lj2 = if same_dir { 1 } else { 0 };
            let sign = if same_dir { 1.0 } else { -1.0 };
            let face = li; // QuadRT1 edge dof pairs: face (0,1)=dofs 0,1; (1,2)=2,3; ...
            gd[face * 2] = base + li2;
            gd[face * 2 + 1] = base + lj2;
            gs[face * 2] = sign;
            gs[face * 2 + 1] = sign;
        }
        for k in 0..4 {
            gd[8 + k] = n_int_base + gid * 4 + k as u32;
            gs[8 + k] = 1.0;
        }

        // Quadrature: flux gradients, RT1 physical basis, element integrals.
        let mut phi_ref = [0.0f64; 24];
        let mut phi = [[0.0f64; 2]; 12];
        let mut grad_m11 = [0.0f64; 18];
        for q in 0..qr.points.len() {
            let xi = &qr.points[q];
            // Bilinear geometry Jacobian on [0,1]² (MFEM SQUARE), same
            // formula as l2_zz.rs.
            let j00 = -(1.0 - xi[1]) * c(0)[0] + (1.0 - xi[1]) * c(1)[0]
                + xi[1] * c(2)[0] - xi[1] * c(3)[0];
            let j01 = -(1.0 - xi[0]) * c(0)[0] - xi[0] * c(1)[0]
                + xi[0] * c(2)[0] + (1.0 - xi[0]) * c(3)[0];
            let j10 = -(1.0 - xi[1]) * c(0)[1] + (1.0 - xi[1]) * c(1)[1]
                + xi[1] * c(2)[1] - xi[1] * c(3)[1];
            let j11 = -(1.0 - xi[0]) * c(0)[1] - xi[0] * c(1)[1]
                + xi[0] * c(2)[1] + (1.0 - xi[0]) * c(3)[1];
            let det = j00 * j11 - j01 * j10;
            let inv_det = 1.0 / det;
            let w = qr.weights[q] * det.abs();

            // RT1 reference basis at xi ∈ [0,1]², contravariant Piola.
            QuadRT1.eval_basis_vec(xi, &mut phi_ref);
            for i in 0..12 {
                phi[i][0] = (j00 * phi_ref[i * 2] + j01 * phi_ref[i * 2 + 1]) * inv_det;
                phi[i][1] = (j10 * phi_ref[i * 2] + j11 * phi_ref[i * 2 + 1]) * inv_det;
            }

            // Physical gradient σ_h = ∇u_h: Q2 basis on [-1,1]² at
            // ξ = 2·xi − 1, chain-rule ×2 into the [0,1]² reference, then
            // J^{-T}.  (MFEM evaluates at the flux nodes with the same J.)
            let eta = [2.0 * xi[0] - 1.0, 2.0 * xi[1] - 1.0];
            QuadQ2.eval_grad_basis(&eta, &mut grad_m11);
            let mut g_ref = [0.0f64; 2];
            for j in 0..9 {
                g_ref[0] += ue[j] * grad_m11[j * 2] * 2.0;
                g_ref[1] += ue[j] * grad_m11[j * 2 + 1] * 2.0;
            }
            let gx = (j11 * g_ref[0] - j10 * g_ref[1]) * inv_det;
            let gy = (-j01 * g_ref[0] + j00 * g_ref[1]) * inv_det;

            // Element mass + load (global dof ids).  The global basis is
            // φ^g = gs·φ^elem, so the mass matrix carries gs[i]·gs[j]
            // (NOT ±·± = + — different edges have different signs) and the
            // load carries gs[i].
            for i in 0..12 {
                let gi = gd[i];
                let si = gs[i];
                b[gi as usize] += si * w * (phi[i][0] * gx + phi[i][1] * gy);
                for j in 0..12 {
                    let gj = gd[j];
                    let v = w * (phi[i][0] * phi[j][0] + phi[i][1] * phi[j][1]);
                    coo.push((gi, gj, v * si * gs[j]));
                }
            }
        }
    }
    coo
}

/// Apply the (chained) RT1 hanging constraints as a conforming-P true-dof
/// elimination: `A_true = Pᵀ A P`, `b_true = Pᵀ b`, solve with PCG, and
/// return the full-space solution `x = P y`.
///
/// `n_dofs` = number of global RT1 dofs; `slaves` = the chained
/// slave→free-master map from [`expand_rt1_slaves`].
pub fn solve_rt1_projection(
    coo: &[(u32, u32, f64)],
    b: &[f64],
    n_dofs: usize,
    slaves: &HashMap<u32, (u32, [[f64; 2]; 2])>,
) -> Vec<f64> {
    let mut cm = CooMatrix::new(n_dofs, n_dofs);
    for &(i, j, v) in coo {
        cm.add(i as usize, j as usize, v);
    }
    let a = cm.into_csr_sorted();

    if slaves.is_empty() {
        let mut x = vec![0.0f64; n_dofs];
        let cfg = SolverConfig {
            rtol: 1e-12,
            max_iter: 200,
            verbose: false,
            ..SolverConfig::default()
        };
        solve_pcg_dsmoother(&a, b, &mut x, &cfg).expect("RT1 projection solve failed");
        return x;
    }

    // P entries: free dof → itself; slave base → (free master base, M) with
    // the slave DOF pair expressed via the 2×2 matrix.
    let slave_rows: std::collections::HashSet<u32> = slaves.keys().copied().collect();
    let free_bases: Vec<u32> = (0..n_dofs as u32)
        .step_by(2)
        .filter(|b| !slave_rows.contains(b))
        .collect();
    // Free dof indices: free_base→true idx (bases are 2-aligned).
    let free_idx: HashMap<u32, usize> = free_bases
        .iter()
        .enumerate()
        .map(|(i, &b)| (b, i * 2))
        .collect();
    let n_true = free_bases.len() * 2;

    // A_true = Pᵀ A P, b_true = Pᵀ b — row/column pairs of A map to true
    // indices via P: dof d in base b: if free → (b, pos); if slave → the
    // master pair through M (both slave DOFs depend on both master DOFs).
    let mut coo_true = CooMatrix::new(n_true, n_true);
    let mut b_true = vec![0.0f64; n_true];
    for &(i, j, v) in coo {
        // expand column j to (true_idx_j, weight_j)
        let jbase = j - (j % 2);
        let jpos = (j % 2) as usize;
        let j_exp: Vec<(usize, f64)> = match slaves.get(&jbase) {
            Some(&(mb, m)) => {
                let ti = free_idx[&mb];
                [(ti, m[jpos][0]), (ti + 1, m[jpos][1])].to_vec()
            }
            None => {
                let ti = free_idx[&jbase] + jpos;
                vec![(ti, 1.0)]
            }
        };
        let ibase = i - (i % 2);
        let ipos = (i % 2) as usize;
        let i_exp: Vec<(usize, f64)> = match slaves.get(&ibase) {
            Some(&(mb, m)) => {
                let ti = free_idx[&mb];
                [(ti, m[ipos][0]), (ti + 1, m[ipos][1])].to_vec()
            }
            None => {
                let ti = free_idx[&ibase] + ipos;
                vec![(ti, 1.0)]
            }
        };
        for &(ii, wi) in &i_exp {
            for &(jj, wj) in &j_exp {
                coo_true.add(ii, jj, v * wi * wj);
            }
        }
    }
    // b_true: Pᵀ b — one pass over the dofs (NOT inside the matrix-entry
    // loop, which would re-add each b[i] once per column and blow it up by
    // the row degree).  Free rows contribute directly; slave rows fold into
    // the master rows with M.
    for i in 0..n_dofs as u32 {
        if b[i as usize] == 0.0 {
            continue;
        }
        let ibase = i - (i % 2);
        let ipos = (i % 2) as usize;
        let i_exp: Vec<(usize, f64)> = match slaves.get(&ibase) {
            Some(&(mb, m)) => {
                let ti = free_idx[&mb];
                [(ti, m[ipos][0]), (ti + 1, m[ipos][1])].to_vec()
            }
            None => {
                let ti = free_idx[&ibase] + ipos;
                vec![(ti, 1.0)]
            }
        };
        for &(ii, wi) in &i_exp {
            b_true[ii] += wi * b[i as usize];
        }
    }
    let a_true = coo_true.into_csr_sorted();
    if std::env::var("PEX15_DBG").is_ok() {
        let mut dmin = f64::MAX;
        let mut dmax = 0.0f64;
        let mut n_zero_rows = 0usize;
        for i in 0..n_true {
            let mut row_norm2 = 0.0f64;
            for k in a_true.row_ptr[i]..a_true.row_ptr[i + 1] {
                let v = a_true.values[k];
                row_norm2 += v * v;
                if a_true.col_idx[k] as usize == i {
                    let av = v.abs();
                    dmin = dmin.min(av);
                    dmax = dmax.max(av);
                }
            }
            if row_norm2 == 0.0 {
                n_zero_rows += 1;
                if n_zero_rows <= 20 {
                    let base = i / 2;
                    eprintln!("[dbg-rt1]   zero row {i} (free base {base} -> orig base {})", free_bases[base]);
                }
            }
        }
        eprintln!("[dbg-rt1] A_true diag: min={dmin:.3e} max={dmax:.3e} n_true={n_true} zero_rows={n_zero_rows}");
        // Rebuild as dense to check eigenvalues (small systems only).
        if n_true <= 4000 {
            let mut mat = nalgebra::DMatrix::<f64>::zeros(n_true, n_true);
            for i in 0..n_true {
                for k in a_true.row_ptr[i]..a_true.row_ptr[i + 1] {
                    mat[(i, a_true.col_idx[k] as usize)] = a_true.values[k];
                }
            }
            let ev = mat.symmetric_eigenvalues();
            let evmin = ev.iter().cloned().fold(f64::MAX, f64::min);
            eprintln!("[dbg-rt1] A_true eigen min = {evmin:.4e}");
        }
    }
    let mut y = vec![0.0f64; n_true];
    let cfg = SolverConfig {
        rtol: 1e-12,
        max_iter: 200,
        verbose: false,
        ..SolverConfig::default()
    };
    let r = solve_pcg_dsmoother(&a_true, &b_true, &mut y, &cfg)
        .expect("RT1 projection solve failed");
    if std::env::var("PEX15_DBG").is_ok() {
        let ymax = y.iter().cloned().fold(0.0f64, |m, v| m.max(v.abs()));
        let bmax = b_true.iter().cloned().fold(0.0f64, |m, v| m.max(v.abs()));
        let bi = b_true.iter().position(|&v| v.abs() == bmax).unwrap_or(0);
        let slaves_in = slaves.len();
        let n_free = free_bases.len();
        eprintln!("[dbg-rt1] PCG: iters={} res={:.3e} ymax={ymax:.3e} b_true_max={bmax:.3e} at {bi} slaves={slaves_in} n_free={n_free}",
            r.iterations, r.final_residual);
    }

    // x = P y
    let mut x = vec![0.0f64; n_dofs];
    for (i, &b_) in free_bases.iter().enumerate() {
        let ti = i * 2;
        x[b_ as usize] = y[ti];
        x[b_ as usize + 1] = y[ti + 1];
    }
    for (&sb, &(mb, m)) in slaves {
        let ti = free_idx[&mb];
        x[sb as usize] = m[0][0] * y[ti] + m[0][1] * y[ti + 1];
        x[sb as usize + 1] = m[1][0] * y[ti] + m[1][1] * y[ti + 1];
    }
    x
}

/// Per-element error `η_K = ∫_K |σ_h − Qσ_h|₂ dx` (L1 over the points,
/// MFEM `ComputeElementLpDistance`, p = 1, quadrature order 5).
///
/// `x_global` = full RT1 solution (values in *global* dof order, including
/// the slave expansion); `elem_gid`, `edge_dof`, `n_edges_global` must
/// match the assembly.  Returns one η per element in `elems` order.
pub fn compute_eta(
    mesh: &Mesh<2>,
    node_gid: &[u32],
    elem_dofs: &[Vec<u32>],
    u: &[f64],
    elems: &[ElemId],
    x_global: &[f64],
    elem_gid: &dyn Fn(ElemId) -> u32,
    edge_dof: &HashMap<(u32, u32), u32>,
    n_edges_global: u32,
) -> Vec<f64> {
    let n_int_base = n_edges_global * 2;
    let qr = QuadRT1.quadrature(8);
    let mut eta = Vec::with_capacity(elems.len());
    for &e in elems {
        let gid = elem_gid(e);
        let nodes = mesh.elem_nodes(e);
        let c = |i: usize| mesh.coords_of(nodes[i]);
        let ue: Vec<f64> = elem_dofs[e as usize]
            .iter()
            .map(|&d| u[d as usize])
            .collect();
        let mut gd = [0u32; 12];
        let mut gs = [1.0f64; 12];
        for (li, lj) in [(0usize, 1usize), (1, 2), (2, 3), (3, 0)] {
            let (ga, gb) = (node_gid[nodes[li] as usize], node_gid[nodes[lj] as usize]);
            let key = (ga.min(gb), ga.max(gb));
            let base = edge_dof[&key];
            let same_dir = ga < gb; // global min→max direction (independent of the local edge (li,lj));
            let li2 = if same_dir { 0 } else { 1 };
            let lj2 = if same_dir { 1 } else { 0 };
            let sign = if same_dir { 1.0 } else { -1.0 };
            gd[li * 2] = base + li2;
            gd[li * 2 + 1] = base + lj2;
            gs[li * 2] = sign;
            gs[li * 2 + 1] = sign;
        }
        for k in 0..4 {
            gd[8 + k] = n_int_base + gid * 4 + k as u32;
        }

        let mut phi_ref = [0.0f64; 24];
        let mut phi = [[0.0f64; 2]; 12];
        let mut grad_m11 = [0.0f64; 18];
        let mut err = 0.0f64;
        let mut sum_flux = 0.0f64;
        let mut sum_qs = 0.0f64;
        for q in 0..qr.points.len() {
            let xi = &qr.points[q];
            let j00 = -(1.0 - xi[1]) * c(0)[0] + (1.0 - xi[1]) * c(1)[0]
                + xi[1] * c(2)[0] - xi[1] * c(3)[0];
            let j01 = -(1.0 - xi[0]) * c(0)[0] - xi[0] * c(1)[0]
                + xi[0] * c(2)[0] + (1.0 - xi[0]) * c(3)[0];
            let j10 = -(1.0 - xi[1]) * c(0)[1] + (1.0 - xi[1]) * c(1)[1]
                + xi[1] * c(2)[1] - xi[1] * c(3)[1];
            let j11 = -(1.0 - xi[0]) * c(0)[1] - xi[0] * c(1)[1]
                + xi[0] * c(2)[1] + (1.0 - xi[0]) * c(3)[1];
            let det = j00 * j11 - j01 * j10;
            let inv_det = 1.0 / det;
            let w = qr.weights[q] * det.abs();

            QuadRT1.eval_basis_vec(xi, &mut phi_ref);
            for i in 0..12 {
                phi[i][0] = (j00 * phi_ref[i * 2] + j01 * phi_ref[i * 2 + 1]) * inv_det;
                phi[i][1] = (j10 * phi_ref[i * 2] + j11 * phi_ref[i * 2 + 1]) * inv_det;
            }

            let eta_pt = [2.0 * xi[0] - 1.0, 2.0 * xi[1] - 1.0];
            QuadQ2.eval_grad_basis(&eta_pt, &mut grad_m11);
            let mut g_ref = [0.0f64; 2];
            for j in 0..9 {
                g_ref[0] += ue[j] * grad_m11[j * 2] * 2.0;
                g_ref[1] += ue[j] * grad_m11[j * 2 + 1] * 2.0;
            }
            let gx = (j11 * g_ref[0] - j10 * g_ref[1]) * inv_det;
            let gy = (-j01 * g_ref[0] + j00 * g_ref[1]) * inv_det;

            let mut sx = 0.0;
            let mut sy = 0.0;
            for i in 0..12 {
                // x is in *global* dof values; the element basis representation
                // is x·gs (φ^g = gs·φ^elem).
                let v = x_global[gd[i] as usize] * gs[i];
                sx += v * phi[i][0];
                sy += v * phi[i][1];
            }
            sum_flux += w * (gx * gx + gy * gy).sqrt();
            sum_qs += w * (sx * sx + sy * sy).sqrt();
            let (dx, dy) = (gx - sx, gy - sy);
            err += w * (dx * dx + dy * dy).sqrt();
        }
        if std::env::var("PEX15_DBG").is_ok() && err > 1.0 {
            eprintln!("[dbg-eta] elem gid={gid} err={err:.4e} ∫|σ|={sum_flux:.4e} ∫|Qσ|={sum_qs:.4e}", );
        }
        eta.push(err);
    }
    eta
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::H1Space;
    use fem_space::fe_space::FESpace;

    fn run_estimator_eta(mesh: &Mesh<2>, u: &[f64]) -> Vec<f64> {
        let space = H1Space::new(mesh.clone(), 2u8);
        let elem_dofs: Vec<Vec<u32>> = (0..mesh.n_elems())
            .map(|e| space.element_dofs(e as ElemId).to_vec())
            .collect();
        let node_gid: Vec<u32> = (0..mesh.n_nodes() as u32).collect();
        let mut edges: Vec<(u32, u32)> = Vec::new();
        for e in 0..mesh.n_elems() as u32 {
            let ns = mesh.elem_nodes(e as ElemId);
            for (li, lj) in [(0usize, 1usize), (1, 2), (2, 3), (3, 0)] {
                let (a, b) = (ns[li], ns[lj]);
                edges.push((a.min(b), a.max(b)));
            }
        }
        edges.sort_unstable();
        edges.dedup();
        let n_edges = edges.len() as u32;
        let edge_dof = build_edge_dof_map(&edges);
        let elems: Vec<ElemId> = (0..mesh.n_elems()).map(|e| e as ElemId).collect();
        let elem_gid = |e: ElemId| e as u32;
        let n_dofs = (n_edges * 2 + mesh.n_elems() as u32 * 4) as usize;
        let mut b = vec![0.0f64; n_dofs];
        let coo = assemble_rt1_system(
            mesh, &node_gid, &elem_dofs, u, &elems, &elem_gid, &edge_dof, n_edges, &mut b,
        );
        let slaves = HashMap::new();
        let x = solve_rt1_projection(&coo, &b, n_dofs, &slaves);
        compute_eta(
            mesh, &node_gid, &elem_dofs, u, &elems, &x, &elem_gid, &edge_dof, n_edges,
        )
    }

    #[test]
    fn rt1_projection_exact_for_linear() {
        // u = x + y is linear → σ_h = ∇u = (1,1) constant ∈ RT1 →
        // projection exact → η = 0 on every element.
        let mesh = Mesh::<2>::make_cartesian_2d(2, 2, 1.0, 1.0);
        let space = H1Space::new(mesh.clone(), 2u8);
        let u = space.interpolate(&|pt: &[f64]| pt[0] + pt[1]);
        let eta = run_estimator_eta(&mesh, u.as_slice());
        for (e, &v) in eta.iter().enumerate() {
            assert!(v < 1e-10, "element {e}: η = {v} (linear should be exact)");
        }
    }

    #[test]
    fn rt1_hanging_linear_exact() {
        // 2×1 grid with the left element refined → hanging edges at x=0.5.
        // u = x + y linear → σ_h = (1,1) ∈ RT1 → the projection (with the
        // RT1 hanging-flux constraints) must be exact → η = 0 everywhere.
        let coarse = Mesh::<2>::make_cartesian_2d(2, 1, 1.0, 1.0);
        let mut nc = fem_mesh::amr::NCStateQuad::new();
        let (mesh, _c, _m) = nc.refine(&coarse, &[0], 0);
        let space = H1Space::new(mesh.clone(), 2u8);
        let u = space.interpolate(&|pt: &[f64]| pt[0] + pt[1]);

        let elem_dofs: Vec<Vec<u32>> = (0..mesh.n_elems())
            .map(|e| space.element_dofs(e as ElemId).to_vec())
            .collect();
        let node_gid: Vec<u32> = (0..mesh.n_nodes() as u32).collect();
        let mut edges: Vec<(u32, u32)> = Vec::new();
        for e in 0..mesh.n_elems() as u32 {
            let ns = mesh.elem_nodes(e as ElemId);
            for (li, lj) in [(0usize, 1usize), (1, 2), (2, 3), (3, 0)] {
                let (a, b) = (ns[li], ns[lj]);
                edges.push((a.min(b), a.max(b)));
            }
        }
        edges.sort_unstable();
        edges.dedup();
        let n_edges = edges.len() as u32;
        let edge_dof = build_edge_dof_map(&edges);
        let elems: Vec<ElemId> = (0..mesh.n_elems()).map(|e| e as ElemId).collect();
        let elem_gid = |e: ElemId| e as u32;
        let n_dofs = (n_edges * 2 + mesh.n_elems() as u32 * 4) as usize;
        let mut b = vec![0.0f64; n_dofs];
        let coo = assemble_rt1_system(
            &mesh, &node_gid, &elem_dofs, u.as_slice(), &elems, &elem_gid, &edge_dof,
            n_edges, &mut b,
        );
        // Hanging constraints from the NC mesh topology.
        let hc = fem_mesh::amr::detect_hanging_quad(&mesh);
        let hang: Vec<(u32, u32, u32)> = hc
            .iter()
            .map(|c| (c.parent_a as u32, c.parent_b as u32, c.constrained as u32))
            .collect();
        let mut gcoords: HashMap<u32, [f64; 2]> = HashMap::new();
        for n in 0..mesh.n_nodes() as u32 {
            let c = mesh.coords_of(n as fem_core::NodeId);
            gcoords.insert(n, [c[0], c[1]]);
        }
        let raw = rt1_hanging_constraints(&hang, &edge_dof, &gcoords);
        assert!(!raw.is_empty(), "expected hanging constraints");
        let slaves = expand_rt1_slaves(&raw);
        let x = solve_rt1_projection(&coo, &b, n_dofs, &slaves);
        let eta = compute_eta(
            &mesh, &node_gid, &elem_dofs, u.as_slice(), &elems, &x, &elem_gid,
            &edge_dof, n_edges,
        );
        for (e, &v) in eta.iter().enumerate() {
            assert!(v < 1e-10, "element {e}: η = {v} (linear should be exact on NC mesh)");
        }
    }

    #[test]
    fn rt1_projection_quadratic_nonzero() {
        // u = x² is not in the RT1 gradient space → η > 0.
        let mesh = Mesh::<2>::make_cartesian_2d(2, 2, 1.0, 1.0);
        let space = H1Space::new(mesh.clone(), 2u8);
        let u = space.interpolate(&|pt: &[f64]| pt[0] * pt[0]);
        let eta = run_estimator_eta(&mesh, u.as_slice());
        let total: f64 = eta.iter().sum();
        assert!(total > 0.0 && total.is_finite(), "quadratic η sum = {total}");
    }
}
