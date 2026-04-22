//! Ghost (halo) node exchange infrastructure.
//!
//! [`GhostExchange`] pre-computes the send/receive communication pattern from a
//! [`MeshPartition`] so that halo updates — propagating owned-node values to
//! ghost copies on neighbouring ranks — can be performed efficiently and
//! repeatedly.
//!
//! Forward/reverse transfers use blocking P2P on [`Comm`] unless the backend is
//! native MPI ([`Comm::is_native_mpi`]) and the `mpi` feature is enabled: then
//! [`GhostExchange::forward_overlapping`] / [`GhostExchange::reverse_overlapping`]
//! use non-blocking `MPI_Isend` / `MPI_Irecv` inside an `rsmpi` request scope so
//! local work can run between posting and completion (e.g. diagonal SpMV while
//! the halo is in flight).
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
        self.forward_overlapping(comm, data, |_data| {});
    }

    /// Like [`forward`](GhostExchange::forward), but on native MPI runs
    /// non-blocking P2P and invokes `overlap` after all sends/receives are
    /// posted (typically diagonal / local work that does not read ghost DOFs).
    ///
    /// `overlap` receives the same `data` buffer so callers can overlap work
    /// without capturing the parent [`ParVector`](crate::par_vector::ParVector).
    pub fn forward_overlapping<F: FnOnce(&mut [f64])>(
        &self,
        comm: &Comm,
        data: &mut [f64],
        overlap: F,
    ) {
        if self.channels.is_empty() {
            overlap(data);
            return;
        }

        #[cfg(all(feature = "mpi", not(target_arch = "wasm32")))]
        if comm.is_native_mpi() {
            mpi_forward_overlap(&self.channels, comm.rank(), data, overlap);
            return;
        }

        overlap(data);
        self.forward_blocking(comm, data);
    }

    fn forward_blocking(&self, comm: &Comm, data: &mut [f64]) {
        const TAG_FWD: i32 = 0x1000;
        let my_rank = comm.rank();

        for ch in &self.channels {
            let send_tag = TAG_FWD + my_rank;
            let send_buf: Vec<u8> = ch
                .send_local_ids
                .iter()
                .flat_map(|&lid| data[lid as usize].to_le_bytes())
                .collect();
            comm.send_bytes(ch.rank, send_tag, &send_buf);
        }

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
        self.reverse_overlapping(comm, data, |_data| {});
    }

    /// Like [`reverse`](GhostExchange::reverse), with an `overlap` hook between
    /// posting and completing non-blocking transfers on native MPI.
    ///
    /// On the native path, ghost slots are zeroed while building send buffers
    /// before `overlap` runs; the callback must not rely on ghost values.
    pub fn reverse_overlapping<F: FnOnce(&mut [f64])>(
        &self,
        comm: &Comm,
        data: &mut [f64],
        overlap: F,
    ) {
        if self.channels.is_empty() {
            overlap(data);
            return;
        }

        #[cfg(all(feature = "mpi", not(target_arch = "wasm32")))]
        if comm.is_native_mpi() {
            mpi_reverse_overlap(&self.channels, comm.rank(), data, overlap);
            return;
        }

        overlap(data);
        self.reverse_blocking(comm, data);
    }

    fn reverse_blocking(&self, comm: &Comm, data: &mut [f64]) {
        const TAG_REV: i32 = 0x2000;
        let my_rank = comm.rank();

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

// ── native MPI non-blocking (rsmpi) ───────────────────────────────────────────

#[cfg(all(feature = "mpi", not(target_arch = "wasm32")))]
fn mpi_forward_overlap<F: FnOnce(&mut [f64])>(
    channels: &[NeighbourChannel],
    my_rank: Rank,
    data: &mut [f64],
    overlap: F,
) {
    use ::mpi::request;
    use ::mpi::topology::SimpleCommunicator;
    use ::mpi::traits::{Communicator, Destination, Source};

    const TAG_FWD: i32 = 0x1000;
    let world = SimpleCommunicator::world();

    // Buffers must outlive the rsmpi `LocalScope` (see `mpi::request` docs).
    let mut recv_bufs: Vec<Vec<u8>> = channels
        .iter()
        .map(|ch| vec![0u8; ch.recv_local_ids.len().saturating_mul(8)])
        .collect();

    let send_bufs: Vec<Vec<u8>> = channels
        .iter()
        .map(|ch| {
            ch.send_local_ids
                .iter()
                .flat_map(|&lid| data[lid as usize].to_le_bytes())
                .collect()
        })
        .collect();

    request::scope(|scope| {
        let mut recv_reqs = Vec::with_capacity(channels.len());
        let n_ch = channels.len();
        debug_assert_eq!(recv_bufs.len(), n_ch);
        let recv_ptr: *mut Vec<u8> = recv_bufs.as_mut_ptr();
        for i in 0..n_ch {
            let ch = &channels[i];
            let recv_tag = TAG_FWD + ch.rank;
            // Disjoint `&mut` into `recv_bufs`; safe because each `i` is unique.
            let recv_buf = unsafe { &mut *recv_ptr.add(i) };
            let req = world
                .process_at_rank(ch.rank)
                .immediate_receive_into_with_tag(scope, recv_buf, recv_tag);
            recv_reqs.push(req);
        }

        let mut send_reqs = Vec::with_capacity(channels.len());
        for (i, ch) in channels.iter().enumerate() {
            let send_tag = TAG_FWD + my_rank;
            let sreq = world
                .process_at_rank(ch.rank)
                .immediate_send_with_tag(scope, &send_bufs[i], send_tag);
            send_reqs.push(sreq);
        }

        overlap(data);

        for req in recv_reqs {
            req.wait();
        }
        for req in send_reqs {
            req.wait();
        }
    });

    for (ch, recv_buf) in channels.iter().zip(recv_bufs.iter()) {
        for (j, &lid) in ch.recv_local_ids.iter().enumerate() {
            let start = j * 8;
            data[lid as usize] = f64::from_le_bytes(
                recv_buf[start..start + 8].try_into().unwrap(),
            );
        }
    }
}

#[cfg(all(feature = "mpi", not(target_arch = "wasm32")))]
fn mpi_reverse_overlap<F: FnOnce(&mut [f64])>(
    channels: &[NeighbourChannel],
    my_rank: Rank,
    data: &mut [f64],
    overlap: F,
) {
    use ::mpi::request;
    use ::mpi::topology::SimpleCommunicator;
    use ::mpi::traits::{Communicator, Destination, Source};

    const TAG_REV: i32 = 0x2000;
    let world = SimpleCommunicator::world();

    let mut recv_bufs: Vec<Vec<u8>> = channels
        .iter()
        .map(|ch| vec![0u8; ch.send_local_ids.len().saturating_mul(8)])
        .collect();

    let mut send_bufs: Vec<Vec<u8>> = Vec::with_capacity(channels.len());
    for ch in channels {
        let send_buf: Vec<u8> = ch
            .recv_local_ids
            .iter()
            .flat_map(|&lid| {
                let val = data[lid as usize];
                data[lid as usize] = 0.0;
                val.to_le_bytes()
            })
            .collect();
        send_bufs.push(send_buf);
    }

    request::scope(|scope| {
        let mut recv_reqs = Vec::with_capacity(channels.len());
        let n_ch = channels.len();
        debug_assert_eq!(recv_bufs.len(), n_ch);
        let recv_ptr: *mut Vec<u8> = recv_bufs.as_mut_ptr();
        for i in 0..n_ch {
            let ch = &channels[i];
            let recv_tag = TAG_REV + ch.rank;
            let recv_buf = unsafe { &mut *recv_ptr.add(i) };
            let req = world
                .process_at_rank(ch.rank)
                .immediate_receive_into_with_tag(scope, recv_buf, recv_tag);
            recv_reqs.push(req);
        }

        let mut send_reqs = Vec::with_capacity(channels.len());
        for (i, ch) in channels.iter().enumerate() {
            let send_tag = TAG_REV + my_rank;
            let sreq = world
                .process_at_rank(ch.rank)
                .immediate_send_with_tag(scope, &send_bufs[i], send_tag);
            send_reqs.push(sreq);
        }

        overlap(data);

        for req in recv_reqs {
            req.wait();
        }
        for req in send_reqs {
            req.wait();
        }
    });

    for (ch, recv_buf) in channels.iter().zip(recv_bufs.iter()) {
        for (j, &lid) in ch.send_local_ids.iter().enumerate() {
            let start = j * 8;
            let val = f64::from_le_bytes(
                recv_buf[start..start + 8].try_into().unwrap(),
            );
            data[lid as usize] += val;
        }
    }
}
