//! jsmpi-backed communication backend for `wasm32` targets.
//!
//! Replaces the previous SAB-based stub with real MPI-style communication
//! via the jsmpi library, which uses Web Workers + SharedArrayBuffer.
//!
//! ## Two modes
//! - `JsMpiBackend::serial()` — rank-0/size-1 stub (same as before)
//! - `JsMpiBackend::from_communicator(comm)` — real multi-worker backend

use fem_core::Rank;
use jsmpi::traits::{Communicator, Destination, Source};
use jsmpi::collective::Root;
use super::CommBackend;

// ── JsMpiBackend ─────────────────────────────────────────────────────────────

/// jsmpi-backed MPI communication backend.
///
/// Construct via:
/// * [`JsMpiBackend::serial()`] — rank-0/size-1 serial stub (always safe).
/// * [`JsMpiBackend::from_communicator()`] — real multi-worker backend wrapping
///   a jsmpi [`SimpleCommunicator`](jsmpi::SimpleCommunicator).
pub struct JsMpiBackend {
    rank: i32,
    size: i32,
    /// `None` = serial stub; `Some` = real jsmpi communicator.
    comm: Option<jsmpi::SimpleCommunicator>,
}

impl JsMpiBackend {
    /// Rank-0 / size-1 stub for serial WASM builds.
    ///
    /// All collective ops degenerate to no-ops; point-to-point ops panic.
    pub fn serial() -> Self {
        JsMpiBackend { rank: 0, size: 1, comm: None }
    }

    /// Construct a live multi-worker backend from a jsmpi communicator.
    pub fn from_communicator(comm: jsmpi::SimpleCommunicator) -> Self {
        let rank = comm.rank();
        let size = comm.size();
        JsMpiBackend { rank, size, comm: Some(comm) }
    }

    /// `true` if this is the stub single-rank path (no real communicator).
    #[inline]
    pub fn is_stub(&self) -> bool {
        self.comm.is_none()
    }
}

// ── Send + Sync ──────────────────────────────────────────────────────────────
//
// wasm32-unknown-unknown is single-threaded; there is no concurrent access.
// These impls are required so `Box<dyn CommBackend>` (which requires
// `CommBackend: Send + Sync`) can hold a `JsMpiBackend`.
//
// SAFETY: wasm32-unknown-unknown has a single-threaded execution model.
//         No other thread can access a `JsMpiBackend` instance.
unsafe impl Send for JsMpiBackend {}
unsafe impl Sync for JsMpiBackend {}

// ── CommBackend impl ─────────────────────────────────────────────────────────

impl CommBackend for JsMpiBackend {
    fn rank(&self) -> Rank { self.rank as Rank }
    fn size(&self) -> usize { self.size as usize }

    fn barrier(&self) {
        if let Some(comm) = &self.comm {
            comm.barrier();
        }
        // serial stub: nothing to synchronise
    }

    fn allreduce_sum_f64(&self, local: f64) -> f64 {
        if let Some(comm) = &self.comm {
            let mut result = 0.0_f64;
            comm.process_at_rank(0).all_reduce_sum_into(&local, &mut result);
            result
        } else {
            local
        }
    }

    fn allreduce_sum_i64(&self, local: i64) -> i64 {
        if let Some(comm) = &self.comm {
            let mut result = 0_i64;
            comm.process_at_rank(0).all_reduce_sum_into(&local, &mut result);
            result
        } else {
            local
        }
    }

    fn broadcast_bytes(&self, root: Rank, buf: &mut Vec<u8>) {
        if let Some(comm) = &self.comm {
            comm.process_at_rank(root).broadcast_into(buf);
        }
        // serial stub: buf already contains the right data
    }

    fn send_bytes(&self, dest: Rank, tag: i32, data: &[u8]) {
        if let Some(comm) = &self.comm {
            comm.process_at_rank(dest).send_with_tag(&data.to_vec(), tag);
        } else {
            panic!("JsMpiBackend (stub): cannot send to rank {dest}");
        }
    }

    fn recv_bytes(&self, src: Rank, tag: i32) -> Vec<u8> {
        if let Some(comm) = &self.comm {
            let (data, _): (Vec<u8>, _) = comm.process_at_rank(src).receive_with_tag(tag);
            data
        } else {
            panic!("JsMpiBackend (stub): cannot receive from rank {src}");
        }
    }

    fn alltoallv_bytes(&self, sends: &[(Rank, Vec<u8>)]) -> Vec<(Rank, Vec<u8>)> {
        let comm = match &self.comm {
            Some(c) => c,
            None => return Vec::new(), // serial stub
        };

        let n = comm.size() as usize;
        let my_rank = comm.rank();

        // Build send map: dest_rank -> payload
        let mut send_map: std::collections::HashMap<i32, &[u8]> = std::collections::HashMap::new();
        for (dest, data) in sends {
            send_map.insert(*dest, data);
        }

        // Step 1: exchange counts via tagged send/recv.
        const TAG_CNT: i32 = 0x4000;
        const TAG_DATA: i32 = 0x5000;

        // Send my counts to all peers
        for r in 0..n {
            let r = r as i32;
            if r == my_rank { continue; }
            let cnt = send_map.get(&r).map(|d| d.len()).unwrap_or(0) as u64;
            comm.process_at_rank(r).send_with_tag(&cnt, TAG_CNT + my_rank);
        }

        // Receive counts from all peers
        let mut recv_counts: Vec<usize> = vec![0; n];
        for r in 0..n {
            let r = r as i32;
            if r == my_rank { continue; }
            let (cnt, _): (u64, _) = comm.process_at_rank(r).receive_with_tag(TAG_CNT + r);
            recv_counts[r as usize] = cnt as usize;
        }

        // Step 2: exchange data using offset-ring to avoid deadlock.
        let mut results: Vec<(Rank, Vec<u8>)> = Vec::new();
        for offset in 1..n {
            let send_to = ((my_rank as usize + offset) % n) as i32;
            let recv_from = ((my_rank as usize + n - offset) % n) as i32;

            // Send if we have data for send_to
            if let Some(&data) = send_map.get(&send_to) {
                if !data.is_empty() {
                    comm.process_at_rank(send_to).send_with_tag(&data.to_vec(), TAG_DATA + my_rank);
                }
            }

            // Recv if recv_from has data for us
            if recv_counts[recv_from as usize] > 0 {
                let (data, _): (Vec<u8>, _) = comm.process_at_rank(recv_from).receive_with_tag(TAG_DATA + recv_from);
                results.push((recv_from, data));
            }
        }
        results
    }
}
