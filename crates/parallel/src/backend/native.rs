//! Native (non-wasm32) communication backends.
//!
//! Two implementations are provided:
//!
//! * [`SerialBackend`] — single-rank no-op, always available.
//! * [`NativeMpiBackend`] — wraps `rsmpi` (`mpi` crate), requires the `mpi`
//!   feature flag and an MPI installation at link time.

use fem_core::Rank;
use super::CommBackend;

// ── SerialBackend ─────────────────────────────────────────────────────────────

/// Single-rank backend used when the `mpi` feature is disabled.
///
/// All collective ops degenerate to identity operations; point-to-point ops
/// panic because there is no other rank to communicate with.
pub struct SerialBackend;

impl CommBackend for SerialBackend {
    fn rank(&self) -> Rank { 0 }
    fn size(&self) -> usize { 1 }
    fn barrier(&self) {}

    fn allreduce_sum_f64(&self, local: f64) -> f64 { local }
    fn allreduce_sum_i64(&self, local: i64) -> i64 { local }

    fn broadcast_bytes(&self, _root: Rank, _buf: &mut Vec<u8>) {
        // Single rank: caller already holds the data.
    }

    fn send_bytes(&self, dest: Rank, _tag: i32, _data: &[u8]) {
        panic!("SerialBackend: cannot send to rank {dest} — only one rank exists");
    }

    fn recv_bytes(&self, src: Rank, _tag: i32) -> Vec<u8> {
        panic!("SerialBackend: cannot receive from rank {src} — only one rank exists");
    }

    fn alltoallv_bytes(&self, _sends: &[(Rank, Vec<u8>)]) -> Vec<(Rank, Vec<u8>)> {
        // No neighbours: nothing to send or receive.
        Vec::new()
    }
}

// ── NativeMpiBackend ──────────────────────────────────────────────────────────

/// rsmpi-backed implementation of [`CommBackend`].
///
/// Wraps `MPI_COMM_WORLD`.  Sub-communicators can be added in Phase 10.
///
/// Requires the `mpi` feature; will not compile without it.
#[cfg(feature = "mpi")]
pub struct NativeMpiBackend {
    rank: Rank,
    size: i32,
}

#[cfg(feature = "mpi")]
impl NativeMpiBackend {
    /// Construct from a live `mpi::environment::Universe`.
    pub fn from_world(universe: &::mpi::environment::Universe) -> Self {
        use ::mpi::traits::Communicator;
        let w = universe.world();
        NativeMpiBackend {
            rank: w.rank(),
            size: w.size(),
        }
    }
}

#[cfg(feature = "mpi")]
impl CommBackend for NativeMpiBackend {
    fn rank(&self) -> Rank { self.rank }
    fn size(&self) -> usize { self.size as usize }

    fn barrier(&self) {
        use ::mpi::topology::SystemCommunicator;
        use ::mpi::traits::Communicator;
        SystemCommunicator::world().barrier();
    }

    fn allreduce_sum_f64(&self, local: f64) -> f64 {
        use ::mpi::collective::SystemOperation;
        use ::mpi::topology::SystemCommunicator;
        use ::mpi::traits::CommunicatorCollectives;
        let mut result = 0.0_f64;
        SystemCommunicator::world()
            .all_reduce_into(&local, &mut result, &SystemOperation::sum());
        result
    }

    fn allreduce_sum_i64(&self, local: i64) -> i64 {
        use ::mpi::collective::SystemOperation;
        use ::mpi::topology::SystemCommunicator;
        use ::mpi::traits::CommunicatorCollectives;
        let mut result = 0_i64;
        SystemCommunicator::world()
            .all_reduce_into(&local, &mut result, &SystemOperation::sum());
        result
    }

    fn broadcast_bytes(&self, root: Rank, buf: &mut Vec<u8>) {
        use ::mpi::topology::SystemCommunicator;
        use ::mpi::traits::{CommunicatorCollectives, Root};
        let world = SystemCommunicator::world();
        let root_proc = world.process_at_rank(root);
        // Step 1: broadcast payload length so non-root ranks can allocate.
        let mut len = buf.len();
        root_proc.broadcast_into(&mut len);
        if self.rank != root {
            buf.resize(len, 0u8);
        }
        // Step 2: broadcast payload.
        root_proc.broadcast_into(buf.as_mut_slice());
    }

    fn send_bytes(&self, dest: Rank, tag: i32, data: &[u8]) {
        use ::mpi::topology::SystemCommunicator;
        use ::mpi::traits::Communicator;
        SystemCommunicator::world()
            .process_at_rank(dest)
            .send_with_tag(data, tag);
    }

    fn recv_bytes(&self, src: Rank, tag: i32) -> Vec<u8> {
        use ::mpi::topology::SystemCommunicator;
        use ::mpi::traits::Communicator;
        let (msg, _status) = SystemCommunicator::world()
            .process_at_rank(src)
            .receive_vec_with_tag::<u8>(tag);
        msg
    }

    fn alltoallv_bytes(&self, sends: &[(Rank, Vec<u8>)]) -> Vec<(Rank, Vec<u8>)> {
        use ::mpi::topology::SystemCommunicator;
        use ::mpi::traits::{Communicator, CommunicatorCollectives};

        let world = SystemCommunicator::world();
        let n     = self.size as usize;

        // Step 1: AllToAll on message counts so each rank knows how many bytes
        // it will receive from each peer (0 = no message expected).
        let mut send_counts = vec![0i64; n];
        for (dest, data) in sends {
            send_counts[*dest as usize] = data.len() as i64;
        }
        let mut recv_counts = vec![0i64; n];
        world.all_to_all_into(&send_counts, &mut recv_counts);

        // Step 2: sparse Isend / blocking-receive exchange.
        //
        // Tag scheme: sender encodes its own rank so that per-source tags are
        // unique within the TAG_A2AV namespace.  We send-then-receive to avoid
        // MPI deadlock (standard "first-half sends, second-half receives"
        // ordering is not needed here because we use MPI_Send which is allowed
        // to buffer, and receive counts are already known).
        const TAG_A2AV: i32 = 0x3000;

        // Post all sends first (MPI may buffer or rendezvous internally).
        for (dest, data) in sends {
            let tag = TAG_A2AV + self.rank;
            world.process_at_rank(*dest).send_with_tag(data.as_slice(), tag);
        }

        // Collect incoming messages in sender-rank order.
        let mut results: Vec<(Rank, Vec<u8>)> = Vec::new();
        for src in 0..n {
            if recv_counts[src] == 0 {
                continue;
            }
            let tag = TAG_A2AV + src as i32;
            let (msg, _status) = world
                .process_at_rank(src as i32)
                .receive_vec_with_tag::<u8>(tag);
            results.push((src as Rank, msg));
        }
        results
    }
}
