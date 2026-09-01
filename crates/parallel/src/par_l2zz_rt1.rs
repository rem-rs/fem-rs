//! Cross-rank RT1 L2-projection ZZ estimator for P2 Quad4 solutions.
//!
//! Parallel counterpart of
//! [`fem_assembly::postproc::l2_zz_rt1`] (1:1 math with MFEM ex15p's
//! `L2ZienkiewiczZhuEstimator`).  The RT1 space is H(div)-continuous, so the
//! smooth-flux projection must be solved on the **global** space  a plain
//! per-rank local solve gives different smooth fluxes than C++ and drifts
//! the AMR marking (pex6 deep-water, RT0 case).
//!
//! Strategy (partition-agnostic, no parallel RT1 FESpace needed):
//!   1. Every rank collects its **owned** elements' edges (global node ids)
//!      and allgathers them  all ranks build the identical global edgeDOF
//!      numbering ([`build_edge_dof_map`], deterministic sort).
//!   2. Every rank assembles its owned elements' RT1 mass/load
//!      contributions into *global* dof ids.
//!   3. Rank 0 gathers all contributions, sorts the COO deterministically
//!      (stable by (row,col) so the summation order  and hence the bit
//!      pattern  is identical for every np), applies the RT1 hanging-flux
//!      constraints (22 P52AP) and solves the global projection with PCG.
//!   4. The global RT1 solution is broadcast; every rank computes its owned
//!      elements'  = |_h 61 Q_h|.
//!
//! Because the assembled global system is the same matrix for every np
//! (deterministic ordering), np1 == np2 == np4 bit-for-bit.

use std::collections::HashMap;

use fem_assembly::postproc::l2_zz_rt1::{
    assemble_rt1_system, build_edge_dof_map, compute_eta, expand_rt1_slaves,
    rt1_hanging_constraints, solve_rt1_projection,
};
use fem_core::ElemId;
use fem_mesh::{Mesh, topology::MeshTopology};
use fem_space::H1Space;
use fem_space::fe_space::FESpace;

use crate::comm::Comm;
use crate::par_mesh::ParallelMesh;

/// RT1 L2ZZ error indicators for a P2 solution `u` (dm order, local dofs)
/// on a partitioned Quad4 mesh.
///
/// `hang_global` = globally-merged hanging edges `(pa, pb, mid)` (global
/// node ids  the caller merges `detect_hanging_quad` across ranks like
/// pex6 does).
///
/// Returns one `_K` per **owned** element (the first
/// `partition.n_owned_elems` local elements).
pub fn l2_zz_rt1_estimator_global(
    par_mesh: &ParallelMesh<Mesh<2>>,
    comm: &Comm,
    u_dm: &[f64],
    hang_global: &[(u32, u32, u32)],
) -> Vec<f64> {
    let local_mesh = par_mesh.local_mesh();
    let partition = par_mesh.partition();
    let n_owned = partition.n_owned_elems;
    let n_local = n_owned + partition.n_ghost_elems;
    // The mesh may contain extra (non-DOF) nodes beyond the partition table
    // (pex6 note) — iterate only over the partitioned nodes.
    let n_nodes_local = partition.global_node_ids.len();
    let node_gid: Vec<u32> = (0..n_nodes_local)
        .map(|n| partition.global_node(n as u32))
        .collect();

    // P2 H1 space on the local mesh (dm order)  element dof extraction.
    let space = H1Space::new(local_mesh.clone(), 2u8);
    let elem_dofs: Vec<Vec<u32>> = (0..n_local)
        .map(|e| space.element_dofs(e as ElemId).to_vec())
        .collect();

    let elem_gid = |e: ElemId| partition.global_elem(e);

    //  1. Global edge set (owned elements only) 
    let mut local_edges: Vec<(u32, u32)> = Vec::new();
    for e in 0..n_owned as u32 {
        let ns = local_mesh.elem_nodes(e as ElemId);
        for (li, lj) in [(0usize, 1usize), (1, 2), (2, 3), (3, 0)] {
            let ga = node_gid[ns[li] as usize];
            let gb = node_gid[ns[lj] as usize];
            local_edges.push((ga.min(gb), ga.max(gb)));
        }
    }
    let mut global_edges = gather_edges(comm, &local_edges);
    global_edges.sort_unstable();
    global_edges.dedup();
    if std::env::var("PEX15_DBG").is_ok() {
        // Which global edges are NOT referenced by any owned element?
        let used: std::collections::HashSet<(u32, u32)> = local_edges.iter().copied().collect();
        let unused: Vec<(u32, u32)> = global_edges.iter().copied().filter(|e| !used.contains(e)).collect();
        eprintln!("[dbg-rt1] local_edges={} global_edges={} unused_by_owned={}",
            local_edges.len(), global_edges.len(), unused.len());
        for &e in unused.iter().take(5) {
            eprintln!("[dbg-rt1]   unused edge {e:?}");
        }
    }
    let edge_dof = build_edge_dof_map(&global_edges);
    let n_edges_global = global_edges.len() as u32;
    let n_elems_global = comm.allreduce_sum_i64(n_owned as i64) as u32;
    let n_dofs = (n_edges_global * 2 + n_elems_global * 4) as usize;

    //  2. Owned-element assembly (global dof ids) 
    let owned: Vec<ElemId> = (0..n_owned).map(|e| e as ElemId).collect();
    let mut b_local = vec![0.0f64; n_dofs];
    if std::env::var("PEX15_DBG").is_ok() {
        // Check global elem id uniqueness (dup gids = rebuild bug).
        let mut seen: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
        for e in 0..n_owned as u32 {
            seen.entry(partition.global_elem(e)).or_default().push(e);
        }
        let dups: Vec<_> = seen.iter().filter(|(_, v)| v.len() > 1).collect();
        if !dups.is_empty() {
            eprintln!("[dbg-rt1] DUP ELEM GIDS: {}", dups.len());
            for (g, v) in dups.iter().take(5) {
                eprintln!("[dbg-rt1]   gid {g} local elems {v:?}");
            }
        } else {
            eprintln!("[dbg-rt1] elem gids unique ({} owned)", n_owned);
        }
        eprintln!("[dbg-rt1] n_owned={n_owned} n_local={n_local} n_dofs={n_dofs} n_edges={n_edges_global} n_int_base={}",
            n_edges_global * 2);
    }
    let coo_local = assemble_rt1_system(
        local_mesh, &node_gid, &elem_dofs, u_dm, &owned, &elem_gid, &edge_dof,
        n_edges_global, &mut b_local,
    );

    //  3. Rank 0 gathers + deterministic sort + constraints + solve 
    let rank = comm.rank();
    let size = comm.size();
    let (all_coo, all_b): (Vec<Vec<(u32, u32, f64)>>, Vec<Vec<f64>>) = if size > 1 {
        let mut coo_payload = Vec::with_capacity(coo_local.len() * 20);
        for &(i, j, v) in &coo_local {
            coo_payload.extend_from_slice(&i.to_le_bytes());
            coo_payload.extend_from_slice(&j.to_le_bytes());
            coo_payload.extend_from_slice(&v.to_le_bytes());
        }
        let coo_sends: Vec<(fem_core::Rank, Vec<u8>)> =
            (0..size as i32).map(|r| (r, coo_payload.clone())).collect();
        let mut b_payload = Vec::with_capacity(b_local.len() * 8);
        for &v in &b_local {
            b_payload.extend_from_slice(&v.to_le_bytes());
        }
        let b_sends: Vec<(fem_core::Rank, Vec<u8>)> =
            (0..size as i32).map(|r| (r, b_payload.clone())).collect();
        let coo_recv = comm.alltoallv_bytes(&coo_sends);
        let b_recv = comm.alltoallv_bytes(&b_sends);
        let mut coos = vec![Vec::new(); size as usize];
        let mut bs = vec![Vec::new(); size as usize];
        for (src, bytes) in coo_recv {
            let mut v = Vec::with_capacity(bytes.len() / 16);
            for chunk in bytes.chunks_exact(16) {
                v.push((
                    u32::from_le_bytes(chunk[0..4].try_into().unwrap()),
                    u32::from_le_bytes(chunk[4..8].try_into().unwrap()),
                    f64::from_le_bytes(chunk[8..16].try_into().unwrap()),
                ));
            }
            coos[src as usize] = v;
        }
        for (src, bytes) in b_recv {
            let mut v = vec![0.0f64; bytes.len() / 8];
            for (k, chunk) in bytes.chunks_exact(8).enumerate() {
                v[k] = f64::from_le_bytes(chunk.try_into().unwrap());
            }
            bs[src as usize] = v;
        }
        (coos, bs)
    } else {
        (vec![coo_local.clone()], vec![b_local.clone()])
    };

    let slaves = {
        // Global node coordinates for edge-orientation signs.  In np>1 the
        // local partition table covers only the local nodes, but the merged
        // hang_global constraints reference cross-rank nodes — collect every
        // rank's node coordinates (gid → coords) so all constraint endpoints
        // resolve (pex15 np2: coords[&a] missing → panic).
        let mut gcoords: HashMap<u32, [f64; 2]> = HashMap::new();
        for n in 0..n_nodes_local as u32 {
            let c = local_mesh.coords_of(n as fem_core::NodeId);
            gcoords.insert(node_gid[n as usize], [c[0], c[1]]);
        }
        if comm.size() > 1 {
            let mut payload = Vec::with_capacity(n_nodes_local * 20);
            for &g in &node_gid {
                let c = gcoords[&g];
                payload.extend_from_slice(&g.to_le_bytes());
                payload.extend_from_slice(&c[0].to_le_bytes());
                payload.extend_from_slice(&c[1].to_le_bytes());
            }
            let sends: Vec<(fem_core::Rank, Vec<u8>)> =
                (0..comm.size() as i32).map(|r| (r, payload.clone())).collect();
            for (_src, bytes) in comm.alltoallv_bytes(&sends) {
                for chunk in bytes.chunks_exact(20) {
                    let g = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
                    let x = f64::from_le_bytes(chunk[4..12].try_into().unwrap());
                    let y = f64::from_le_bytes(chunk[12..20].try_into().unwrap());
                    gcoords.entry(g).or_insert([x, y]);
                }
            }
        }
        let raw = rt1_hanging_constraints(hang_global, &edge_dof, &gcoords);
        if std::env::var("PEX15_DBG").is_ok() && rank == 0 {
            let n_first = raw.iter().filter(|s| s.m == fem_assembly::postproc::l2_zz_rt1::RT1_M_FIRST).count();
            let n_second = raw.iter().filter(|s| s.m == fem_assembly::postproc::l2_zz_rt1::RT1_M_SECOND).count();
            eprintln!("[dbg-rt1] hang_global={} raw_slaves={} first={n_first} second={n_second}",
                hang_global.len(), raw.len());
        }
        expand_rt1_slaves(&raw)
    };
    if std::env::var("PEX15_DBG").is_ok() && rank == 0 {
        for (sb, &(mb, m)) in slaves.iter().take(5) {
            eprintln!("[dbg-rt1] slave {sb} <- master {mb} M=[[{:.4},{:.4}],[{:.4},{:.4}]]",
                m[0][0], m[0][1], m[1][0], m[1][1]);
        }
    }

    let x_global: Vec<f64> = if rank == 0 {
        // Deterministic merge: concatenate in rank order, then stable-sort
        // by (row, col) so the accumulation order is identical for any np.
        let mut coo_all: Vec<(u32, u32, f64)> = Vec::new();
        for c in &all_coo {
            coo_all.extend_from_slice(c);
        }
        coo_all.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        let mut b_all = vec![0.0f64; n_dofs];
        for bb in &all_b {
            for (i, &v) in bb.iter().enumerate() {
                b_all[i] += v;
            }
        }
        if std::env::var("PEX15_DBG").is_ok() {
            let nz = coo_all.len();
            let bn: f64 = b_all.iter().map(|v| v * v).sum::<f64>().sqrt();
            let bmax = b_all.iter().cloned().fold(0.0f64, |m, v| if v.is_finite() { m.max(v.abs()) } else { m });
            let bnan = b_all.iter().filter(|v| !v.is_finite()).count();
            let conan = coo_all.iter().filter(|&&(_, _, v)| !v.is_finite()).count();
            eprintln!("[dbg-rt1] n_dofs={n_dofs} nz={nz} ||b||={bn:.6e} bmax={bmax:.4e} bnan={bnan} conan={conan} slaves={}", slaves.len());
            // Diagonal range of A.
            let mut diag: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
            for &(i, j, v) in &coo_all {
                if i == j {
                    *diag.entry(i).or_insert(0.0) += v;
                }
            }
            let dmin = diag.values().cloned().fold(f64::MAX, f64::min);
            let dmax = diag.values().cloned().fold(0.0f64, f64::max);
            eprintln!("[dbg-rt1] diag: min={dmin:.3e} max={dmax:.3e}");
            for b in [3386u32, 3388, 3958, 3960] {
                eprintln!("[dbg-rt1]   orig base {b} diag = {:?}", diag.get(&b));
            }
        }
        solve_rt1_projection(&coo_all, &b_all, n_dofs, &slaves)
    } else {
        vec![0.0f64; n_dofs]
    };

    //  4. Broadcast + owned-element  
    let mut x = x_global;
    if size > 1 {
        let mut buf = x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        comm.broadcast_bytes(0, &mut buf);
        x = buf
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
    }
    if std::env::var("PEX15_DBG").is_ok() && rank == 0 {
        let xnan = x.iter().filter(|v| !v.is_finite()).count();
        let xmax = x.iter().cloned().fold(0.0f64, |m, v| if v.is_finite() { m.max(v.abs()) } else { m });
        let xi = x.iter().position(|&v| v.abs() == xmax).unwrap_or(0);
        eprintln!("[dbg-rt1] x nan={xnan} max={xmax:.4e} at dof {xi} (n_edges*2={}) len={}",
            n_edges_global * 2, x.len());
    }
    let _ = &all_b;

    let eta = compute_eta(
        local_mesh, &node_gid, &elem_dofs, u_dm, &owned, &x, &elem_gid, &edge_dof,
        n_edges_global,
    );
    eta
}

/// Allgather a list of edge keys; returns the merged (unsorted, possibly
/// duplicated) list replicated on every rank.
fn gather_edges(comm: &Comm, local: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let size = comm.size();
    if size <= 1 {
        return local.to_vec();
    }
    let mut payload = Vec::with_capacity(local.len() * 8);
    for &(a, b) in local {
        payload.extend_from_slice(&a.to_le_bytes());
        payload.extend_from_slice(&b.to_le_bytes());
    }
    let sends: Vec<(fem_core::Rank, Vec<u8>)> =
        (0..size as i32).map(|r| (r, payload.clone())).collect();
    let recv = comm.alltoallv_bytes(&sends);
    let mut out = Vec::new();
    for (_src, bytes) in recv {
        for chunk in bytes.chunks_exact(8) {
            out.push((
                u32::from_le_bytes(chunk[0..4].try_into().unwrap()),
                u32::from_le_bytes(chunk[4..8].try_into().unwrap()),
            ));
        }
    }
    out
}
