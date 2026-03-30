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
//! runs against the native MPI backend, the serial stub, and the future Web
//! Worker backend without any `#[cfg]` guards here.

use std::collections::HashMap;
use fem_core::Rank;
use crate::partition::MeshPartition;
use crate::comm::Comm;

// ── NeighbourChannel ─────────────────────────────────────────────────────────

/// Send/receive specification for one neighbouring rank.
#[derive(Debug, Clone)]
struct NeighbourChannel {
    /// Peer rank.
    rank: Rank,
    /// Local node indices this rank sends to `rank` (owned nodes needed as
    /// ghosts on `rank`).  Filled by the setup collective in Phase 10.
    send_local_ids: Vec<u32>,
    /// Local ghost-slot indices that will be overwritten with data from `rank`.
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

    /// Build the exchange pattern from a partition.
    ///
    /// When `comm.size() == 1` (serial or single-rank) this returns an empty
    /// no-op exchange immediately.
    ///
    /// For multi-rank runs a small collective determines which owned nodes must
    /// be sent to which neighbours (Phase 10 full implementation).
    pub fn from_partition(partition: &MeshPartition, comm: &Comm) -> Self {
        // ── fast path: serial or single-rank ─────────────────────────────────
        if comm.size() == 1 {
            return GhostExchange { channels: Vec::new() };
        }

        // ── multi-rank: group ghost nodes by owner rank ───────────────────────
        //
        // recv_local_ids are already known from the partition (ghost slots).
        // send_local_ids require an AllToAll so every rank learns which of its
        // owned nodes are needed as ghosts elsewhere.
        //
        // Phase 10 plan:
        //   1. Each rank serialises (global_node_id, local_slot) for each ghost.
        //   2. alltoallv_bytes to deliver these requests to their owners.
        //   3. Each owner records the requested local IDs as send_local_ids.
        //
        // For now we only populate recv_local_ids; the forward/reverse impls
        // are stubs until Phase 10 wires the collective.

        let mut recv_map: HashMap<Rank, Vec<u32>> = HashMap::new();
        for (local_id, owner) in partition.ghost_nodes() {
            recv_map.entry(owner).or_default().push(local_id);
        }

        let channels = recv_map
            .into_iter()
            .map(|(rank, recv_local_ids)| NeighbourChannel {
                rank,
                send_local_ids: Vec::new(), // filled by setup collective in Phase 10
                recv_local_ids,
            })
            .collect();

        GhostExchange { channels }
    }

    // ── operations ───────────────────────────────────────────────────────────

    /// **Forward update**: push owned-node values into ghost slots on neighbours.
    ///
    /// `data` must have length `>= partition.n_total_nodes()`.
    /// After the call `data[ghost_id]` on every rank mirrors the owner's value.
    pub fn forward(&self, comm: &Comm, data: &mut [f64]) {
        if self.channels.is_empty() {
            return; // serial or single-rank: nothing to do
        }

        // Phase 10: for each channel, pack send_local_ids values into a byte
        // buffer, comm.send_bytes(channel.rank, TAG_FORWARD, &buf), then
        // comm.recv_bytes(channel.rank, TAG_FORWARD) into recv slots.
        //
        // Tag scheme: use channel index to avoid tag collisions.
        const TAG_FWD: i32 = 0x1000;
        for (i, ch) in self.channels.iter().enumerate() {
            let tag = TAG_FWD + i as i32;
            // Pack owned values destined for this neighbour.
            let send_buf: Vec<u8> = ch
                .send_local_ids
                .iter()
                .flat_map(|&lid| data[lid as usize].to_le_bytes())
                .collect();
            comm.send_bytes(ch.rank, tag, &send_buf);

            // Receive ghost values from this neighbour.
            let recv_buf = comm.recv_bytes(ch.rank, tag);
            for (j, &lid) in ch.recv_local_ids.iter().enumerate() {
                let start = j * 8;
                let val = f64::from_le_bytes(
                    recv_buf[start..start + 8].try_into().unwrap(),
                );
                data[lid as usize] = val;
            }
        }
    }

    /// **Reverse update**: accumulate ghost contributions back to owned slots.
    ///
    /// Zeroes ghost slots after accumulation.
    pub fn reverse(&self, comm: &Comm, data: &mut [f64]) {
        if self.channels.is_empty() {
            return;
        }

        const TAG_REV: i32 = 0x2000;
        for (i, ch) in self.channels.iter().enumerate() {
            let tag = TAG_REV + i as i32;
            // Pack ghost values to send back to owner.
            let send_buf: Vec<u8> = ch
                .recv_local_ids
                .iter()
                .flat_map(|&lid| {
                    let val = data[lid as usize];
                    data[lid as usize] = 0.0; // zero ghost after sending
                    val.to_le_bytes()
                })
                .collect();
            comm.send_bytes(ch.rank, tag, &send_buf);

            // Receive contributions into owned slots.
            let recv_buf = comm.recv_bytes(ch.rank, tag);
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
