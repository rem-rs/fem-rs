//! Parallel DPG trace (skeleton) space.
//!
//! Wraps the serial [`DpgTraceSpace`] of the local mesh and assigns
//! **globally consistent** trace DOF ids: every skeleton face gets a global
//! face id from the sorted set of face keys (the key of a face is its sorted
//! pair of global node ids), and each face carries `order + 1` consecutive
//! DOFs starting at `global_face_id * dofs_per_face`.  All ranks compute the
//! same global numbering, so a shared face has the same DOF ids on both
//! sides.
//!
//! Local DOFs are laid out in the usual `[owned | ghost]` compact segments
//! (owned = owner of the face's smaller endpoint node), with a dedicated
//! [`GhostExchange`] for trace-DOF values.

use std::collections::BTreeSet;
use std::sync::Arc;

use fem_core::Rank;
use fem_mesh::topology::MeshTopology;
use fem_space::DpgTraceSpace;

use crate::comm::Comm;
use crate::ghost::{GhostChannelDef, GhostExchange};
use crate::par_mesh::ParallelMesh;

/// Tag base for the trace-face key exchange.
const DPG_TRACE_TAG: i32 = 0x3C00;

/// Parallel DPG trace space over a partitioned mesh.
pub struct ParDpgTraceSpace<M: MeshTopology> {
    /// Serial trace space of the local mesh (face geometry/adjacency).
    local: DpgTraceSpace<M>,
    /// Global DOF base of each local face (`dpf` consecutive DOFs).
    face_global_dof_base: Vec<u32>,
    /// Owner rank of each local face (owner of the smaller endpoint node).
    face_owner: Vec<Rank>,
    /// Local compact segment → global DOF id (`[owned | ghost]`).
    local_to_global: Vec<u32>,
    /// Compact segment index of each local face's DOFs (flat, `dpf` each).
    face_local_dofs: Vec<u32>,
    n_owned_dofs: usize,
    n_ghost_dofs: usize,
    n_global_dofs: usize,
    ghost_exchange: Arc<GhostExchange>,
    /// Ghost DOFs in the compact segment: `(local_compact_id, owner)`.
    ghost_dof_owners: Vec<(u32, Rank)>,
}

impl<M: MeshTopology> ParDpgTraceSpace<M> {
    /// Build the parallel trace space from a partitioned mesh.
    ///
    /// `comm.size() == 1` fast path: the local space is used directly with an
    /// identity numbering (global == local).
    pub fn new(
        local_mesh: M,
        order: u8,
        par_mesh: &ParallelMesh<M>,
        comm: &Comm,
    ) -> Self {
        let local = DpgTraceSpace::new(local_mesh, order);
        let dpf = local.dofs_per_face();
        let n_local_faces = local.n_faces();
        let rank = comm.rank();
        let partition = par_mesh.partition();

        // 1. Local face keys (sorted global node-id pairs).
        let gid_of = |n: u32| partition.global_node(n);
        let mut local_face_keys: Vec<(u32, u32)> = Vec::with_capacity(n_local_faces);
        for f in 0..n_local_faces {
            let nodes = match local.face_info(f) {
                fem_space::FaceInfo::Boundary { nodes, .. } => nodes,
                fem_space::FaceInfo::Interior { nodes, .. } => nodes,
            };
            let a = gid_of(nodes[0]);
            let b = gid_of(nodes[1]);
            local_face_keys.push((a.min(b), a.max(b)));
        }

        // 2. Global face numbering: sorted union of all ranks' face keys.
        let global_keys = if comm.size() > 1 {
            let mut payload = Vec::with_capacity(local_face_keys.len() * 8);
            for &(a, b) in &local_face_keys {
                payload.extend_from_slice(&a.to_le_bytes());
                payload.extend_from_slice(&b.to_le_bytes());
            }
            let sends: Vec<(Rank, Vec<u8>)> = (0..comm.size() as i32)
                .map(|r| (r, payload.clone()))
                .collect();
            let mut set: BTreeSet<(u32, u32)> = BTreeSet::new();
            for (_src, bytes) in comm.alltoallv_bytes(&sends) {
                for chunk in bytes.chunks_exact(8) {
                    let a = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
                    let b = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
                    set.insert((a, b));
                }
            }
            set.into_iter().collect::<Vec<_>>()
        } else {
            let set: BTreeSet<(u32, u32)> = local_face_keys.iter().copied().collect();
            set.into_iter().collect()
        };
        let n_global_faces = global_keys.len();
        let global_id_of = |key: (u32, u32)| global_keys.binary_search(&key).unwrap() as u32;

        // 3. Local face → global DOF base + owner; build the global-dof list.
        let mut face_global_dof_base = vec![0u32; n_local_faces];
        let mut face_owner = vec![0i32; n_local_faces];
        let mut face_local_dofs = vec![u32::MAX; n_local_faces * dpf];
        let mut dofs: Vec<(u32, Rank)> = Vec::with_capacity(n_local_faces * dpf); // (global dof, owner)
        for f in 0..n_local_faces {
            let key = local_face_keys[f];
            let fid = global_id_of(key);
            let base = fid as usize * dpf;
            face_global_dof_base[f] = base as u32;
            let min_local = partition
                .local_node(key.0)
                .expect("par_dpg_trace: face endpoint not present locally");
            let owner = partition.node_owner[min_local as usize];
            face_owner[f] = owner;
            for k in 0..dpf {
                dofs.push(((base + k) as u32, owner));
            }
        }

        // 4. Compact [owned | ghost] layout (sorted by global id within each
        //    segment) and the local face → compact index map.
        dofs.sort_by_key(|&(g, o)| (o != rank, g));
        let n_owned_dofs = dofs.iter().filter(|&&(_, o)| o == rank).count();
        let local_to_global: Vec<u32> = dofs.iter().map(|&(g, _)| g).collect();
        for f in 0..n_local_faces {
            for k in 0..dpf {
                let g = face_global_dof_base[f] as usize + k;
                let pos = dofs
                    .iter()
                    .position(|&(gg, _)| gg as usize == g)
                    .expect("par_dpg_trace: dof not in local layout");
                face_local_dofs[f * dpf + k] = pos as u32;
            }
        }
        let n_ghost_dofs = dofs.len() - n_owned_dofs;

        // 5. Ghost-exchange channels for trace-DOF values.
        let ghost_exchange = build_trace_ghost_exchange(comm, &dofs, rank, n_owned_dofs);
        let ghost_dof_owners: Vec<(u32, Rank)> = dofs
            .iter()
            .enumerate()
            .skip(n_owned_dofs)
            .map(|(i, &(_, o))| (i as u32, o))
            .collect();

        ParDpgTraceSpace {
            local,
            face_global_dof_base,
            face_owner,
            local_to_global,
            face_local_dofs,
            n_owned_dofs,
            n_ghost_dofs,
            n_global_dofs: n_global_faces * dpf,
            ghost_exchange: Arc::new(ghost_exchange),
            ghost_dof_owners,
        }
    }

    /// The serial local trace space.
    pub fn local(&self) -> &DpgTraceSpace<M> {
        &self.local
    }

    /// Local compact DOF indices (segment order) of skeleton face `face_idx`.
    pub fn face_dofs_local(&self, face_idx: usize) -> &[u32] {
        let dpf = self.local.dofs_per_face();
        &self.face_local_dofs[face_idx * dpf..(face_idx + 1) * dpf]
    }

    /// Owner rank of face `face_idx`.
    pub fn face_owner(&self, face_idx: usize) -> Rank {
        self.face_owner[face_idx]
    }

    /// Global DOF id of a local compact DOF.
    pub fn global_dof(&self, local_compact: u32) -> u32 {
        self.local_to_global[local_compact as usize]
    }

    /// Local compact id of an *owned* global DOF, if this rank owns it.
    pub fn owned_local_dof(&self, global: u32) -> Option<u32> {
        self.local_to_global[..self.n_owned_dofs]
            .binary_search(&global)
            .ok()
            .map(|i| i as u32)
    }

    /// Local compact segment → global id mapping.
    pub fn local_to_global(&self) -> &[u32] {
        &self.local_to_global
    }

    /// Ghost DOFs in the compact segment: `(local_compact_id, owner_rank)`.
    pub fn ghost_dofs(&self) -> &[(u32, Rank)] {
        &self.ghost_dof_owners
    }

    /// Number of owned trace DOFs on this rank.
    pub fn n_owned_dofs(&self) -> usize {
        self.n_owned_dofs
    }

    /// Number of ghost trace DOFs on this rank.
    pub fn n_ghost_dofs(&self) -> usize {
        self.n_ghost_dofs
    }

    /// Total number of trace DOFs across all ranks.
    pub fn n_global_dofs(&self) -> usize {
        self.n_global_dofs
    }

    /// Total local trace DOFs (owned + ghost).
    pub fn n_local_dofs(&self) -> usize {
        self.n_owned_dofs + self.n_ghost_dofs
    }

    /// Ghost-exchange handle for trace-DOF values.
    pub fn ghost_exchange_arc(&self) -> Arc<GhostExchange> {
        Arc::clone(&self.ghost_exchange)
    }
}

/// Build trace-DOF ghost-exchange channels: each rank requests the global
/// DOF ids it needs as ghosts from their owners; owners reply with the local
/// compact indices to send.
fn build_trace_ghost_exchange(
    comm: &Comm,
    dofs: &[(u32, Rank)], // sorted [owned | ghost], (global dof, owner)
    rank: Rank,
    n_owned: usize,
) -> GhostExchange {
    if comm.size() <= 1 {
        return GhostExchange::from_trivial();
    }
    let _ = rank;

    // Requests: ghost dof → owner (global ids).
    let mut requests: std::collections::BTreeMap<Rank, Vec<u32>> = Default::default();
    let mut recv_local: std::collections::BTreeMap<Rank, Vec<u32>> = Default::default();
    for (local_id, &(g, o)) in dofs.iter().enumerate().skip(n_owned) {
        requests.entry(o).or_default().push(g);
        recv_local.entry(o).or_default().push(local_id as u32);
    }

    // Send requests to owners (alltoallv, collective).
    let sends: Vec<(Rank, Vec<u8>)> = requests
        .iter()
        .map(|(&dest, gids)| {
            let mut b = Vec::with_capacity(gids.len() * 4);
            for &g in gids {
                b.extend_from_slice(&g.to_le_bytes());
            }
            (dest, b)
        })
        .collect();
    let incoming = comm.alltoallv_bytes(&sends);

    // Owner side: map requested global ids to local compact indices.
    let mut send_local: std::collections::BTreeMap<Rank, Vec<u32>> = Default::default();
    for (requester, bytes) in &incoming {
        let mut idx = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            let g = u32::from_le_bytes(chunk.try_into().unwrap());
            let pos = dofs[..n_owned]
                .iter()
                .position(|&(gg, _)| gg == g)
                .expect("par_dpg_trace: requested dof not owned");
            idx.push(pos as u32);
        }
        send_local.insert(*requester, idx);
    }

    let mut channels = Vec::new();
    let mut all_ranks: BTreeSet<Rank> = send_local.keys().copied().collect();
    all_ranks.extend(recv_local.keys().copied());
    for r in all_ranks {
        channels.push(GhostChannelDef {
            rank: r,
            send_local_ids: send_local.remove(&r).unwrap_or_default(),
            recv_local_ids: recv_local.remove(&r).unwrap_or_default(),
        });
    }
    GhostExchange::from_channels(channels)
}
