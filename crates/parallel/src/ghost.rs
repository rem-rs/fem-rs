//! Ghost (halo) node exchange infrastructure.
//!
//! [`GhostExchange`] pre-computes the send/receive communication pattern from a
//! [`MeshPartition`] so that halo updates — propagating owned-node values to
//! ghost copies on neighbouring ranks — can be performed efficiently and
//! repeatedly.
//!
//! ## Halo update protocol
//!
//! 1. **Build once**: `GhostExchange::from_partition(&partition, &comm)`.
//! 2. **Forward per step**: `exchange.forward(&comm, data)` — owner → ghost.
//! 3. **Reverse (optional)**: `exchange.reverse(&comm, data)` — ghost → owner
//!    accumulation (needed for parallel FEM assembly).
//!
//! Communication is fully delegated to [`Comm`]'s backend, so the same code
//! runs against the native MPI backend, the serial stub, and the in-process
//! channel backend without any `#[cfg]` guards here.

use std::collections::{HashMap, HashSet};
use fem_core::Rank;
use crate::partition::MeshPartition;
use crate::comm::Comm;

// ── NeighbourChannel ─────────────────────────────────────────────────────────

/// Send/receive specification for one neighbouring rank.
#[derive(Debug, Clone)]
struct NeighbourChannel {
    /// Peer rank.
    rank: Rank,
    /// Local node IDs this rank sends TO `rank` (owned nodes needed as ghosts
    /// on `rank`).  Populated by the alltoallv setup collective.
    send_local_ids: Vec<u32>,
    /// Local ghost-slot IDs that will be overwritten with data FROM `rank`.
    recv_local_ids: Vec<u32>,
}

// ── GhostExchange ─────────────────────────────────────────────────────────────

/// Pre-computed communication pattern for halo (ghost node) updates.
///
/// Construct via [`GhostExchange::from_partition`]; then call
/// [`forward`](GhostExchange::forward) or [`reverse`](GhostExchange::reverse)
/// once per solve step.
#[derive(Debug, Clone)]
pub struct GhostExchange {
    channels: Vec<NeighbourChannel>,
}

impl GhostExchange {
    // ── construction ─────────────────────────────────────────────────────────

    /// Create a trivial (no-op) ghost exchange with no neighbours.
    ///
    /// Useful for serial / single-rank contexts or testing.
    pub fn from_trivial() -> Self {
        GhostExchange { channels: Vec::new() }
    }

    /// Build the exchange pattern from a partition.
    ///
    /// When `comm.size() == 1` (serial or single-rank) this returns an empty
    /// no-op exchange immediately.
    ///
    /// For multi-rank runs an `alltoallv` collective distributes ghost-node
    /// requests to their owners so that every rank learns which of its owned
    /// nodes are needed as ghosts elsewhere (`send_local_ids`).
    ///
    /// ## Algorithm
    /// 1. Each rank groups its ghost nodes by owner rank and serialises their
    ///    **global** node IDs as request payloads.
    /// 2. `alltoallv_bytes` delivers each request to its target owner.
    /// 3. Each owner maps the requested global IDs back to local IDs — these
    ///    become `send_local_ids` for the corresponding channel.
    /// 4. `recv_local_ids` are always known locally from the partition.
    pub fn from_partition(partition: &MeshPartition, comm: &Comm) -> Self {
        // ── fast path: serial / single-rank ──────────────────────────────────
        if comm.size() == 1 {
            return GhostExchange { channels: Vec::new() };
        }

        // ── group ghosts by owner rank ────────────────────────────────────────
        // recv_slots[owner] = list of local ghost-slot IDs owned by `owner`.
        // requests[owner]   = list of global node IDs we need from `owner`.
        let mut recv_slots: HashMap<Rank, Vec<u32>> = HashMap::new();
        let mut requests:   HashMap<Rank, Vec<u32>> = HashMap::new();

        for (local_id, owner) in partition.ghost_nodes() {
            let gid = partition.global_node(local_id);
            recv_slots.entry(owner).or_default().push(local_id);
            requests.entry(owner).or_default().push(gid);
        }

        // ── send requests via alltoallv ───────────────────────────────────────
        // Each request payload is a contiguous array of u32 global node IDs
        // encoded as little-endian bytes.
        let sends: Vec<(Rank, Vec<u8>)> = requests
            .iter()
            .map(|(&owner, gids)| {
                let bytes = gids
                    .iter()
                    .flat_map(|&g| g.to_le_bytes())
                    .collect::<Vec<u8>>();
                (owner, bytes)
            })
            .collect();

        let received = comm.alltoallv_bytes(&sends);

        // ── build send_local_ids from received requests ───────────────────────
        // `received` contains `(requester_rank, encoded_global_ids)` for every
        // rank that needs owned nodes from us.
        let mut send_map: HashMap<Rank, Vec<u32>> = HashMap::new();
        for (requester, bytes) in received {
            // Decode u32 global IDs.
            debug_assert_eq!(bytes.len() % 4, 0, "alltoallv payload must be u32-aligned");
            let requested_gids: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let local_ids: Vec<u32> = requested_gids
                .iter()
                .map(|&gid| {
                    partition
                        .local_node(gid)
                        .unwrap_or_else(|| panic!(
                            "GhostExchange: rank {} requested global node {} \
                             but this rank does not own it",
                            requester, gid
                        ))
                })
                .collect();
            send_map.insert(requester, local_ids);
        }

        // ── assemble channels ─────────────────────────────────────────────────
        // A channel exists for every neighbour we either receive from OR send to.
        let mut all_neighbors: HashSet<Rank> = HashSet::new();
        all_neighbors.extend(recv_slots.keys().copied());
        all_neighbors.extend(send_map.keys().copied());

        let channels = all_neighbors
            .into_iter()
            .map(|neighbor| NeighbourChannel {
                rank:           neighbor,
                send_local_ids: send_map.remove(&neighbor).unwrap_or_default(),
                recv_local_ids: recv_slots.remove(&neighbor).unwrap_or_default(),
            })
            .collect();

        GhostExchange { channels }
    }

    // ── operations ───────────────────────────────────────────────────────────

    /// **Forward update**: push owned-node values into ghost slots on neighbours.
    ///
    /// `data` must have length `>= partition.n_total_nodes()`.
    /// After the call `data[ghost_id]` on every rank mirrors the owner's value.
    ///
    /// Tag scheme: use `TAG_FWD + sender_rank` so both ends of a channel
    /// independently arrive at the same tag without coordination.
    pub fn forward(&self, comm: &Comm, data: &mut [f64]) {
        if self.channels.is_empty() {
            return; // serial / single-rank
        }

        const TAG_FWD: i32 = 0x1000;
        let my_rank = comm.rank();

        // Phase 1: send all owned values to neighbours (non-blocking pushes).
        for ch in &self.channels {
            let send_tag = TAG_FWD + my_rank;
            let send_buf: Vec<u8> = ch
                .send_local_ids
                .iter()
                .flat_map(|&lid| data[lid as usize].to_le_bytes())
                .collect();
            comm.send_bytes(ch.rank, send_tag, &send_buf);
        }

        // Phase 2: receive ghost values from all neighbours.
        for ch in &self.channels {
            let recv_tag = TAG_FWD + ch.rank;
            let recv_buf = comm.recv_bytes(ch.rank, recv_tag);
            for (j, &lid) in ch.recv_local_ids.iter().enumerate() {
                let start = j * 8;
                data[lid as usize] = f64::from_le_bytes(
                    recv_buf[start..start + 8].try_into().unwrap(),
                );
            }
        }
    }

    /// **Reverse update**: accumulate ghost contributions back to owned slots.
    ///
    /// Zeroes ghost slots after accumulation.
    ///
    /// Tag scheme: use `TAG_REV + sender_rank` symmetrically to `forward`.
    pub fn reverse(&self, comm: &Comm, data: &mut [f64]) {
        if self.channels.is_empty() {
            return;
        }

        const TAG_REV: i32 = 0x2000;
        let my_rank = comm.rank();

        // Phase 1: send all ghost values back to their owners (zero locally).
        for ch in &self.channels {
            let send_tag = TAG_REV + my_rank;
            let send_buf: Vec<u8> = ch
                .recv_local_ids
                .iter()
                .flat_map(|&lid| {
                    let val = data[lid as usize];
                    data[lid as usize] = 0.0;
                    val.to_le_bytes()
                })
                .collect();
            comm.send_bytes(ch.rank, send_tag, &send_buf);
        }

        // Phase 2: receive contributions from all neighbours into owned slots.
        for ch in &self.channels {
            let recv_tag = TAG_REV + ch.rank;
            let recv_buf = comm.recv_bytes(ch.rank, recv_tag);
            for (j, &lid) in ch.send_local_ids.iter().enumerate() {
                let start = j * 8;
                let val = f64::from_le_bytes(
                    recv_buf[start..start + 8].try_into().unwrap(),
                );
                data[lid as usize] += val;
            }
        }
    }

    // ── queries ───────────────────────────────────────────────────────────────

    /// Number of neighbouring ranks this rank communicates with.
    pub fn n_neighbours(&self) -> usize {
        self.channels.len()
    }

    /// `true` if the exchange is trivially empty (serial / single-rank).
    pub fn is_trivial(&self) -> bool {
        self.channels.is_empty()
    }
}
