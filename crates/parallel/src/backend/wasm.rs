//! Web Worker–based communication backend for `wasm32` targets.
//!
//! ## Architecture (planned)
//!
//! Each MPI "process" runs as an independent Web Worker loading the same WASM
//! module.  Workers share a `SharedArrayBuffer` (SAB) arena that holds:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  SAB layout (allocated by the launcher, passed to each worker)  │
//! ├──────────┬──────────┬─────────────────────────────────────────┐  │
//! │ ctrl[0]  │ ctrl[1]  │  ring-buffer slots [rank_0 … rank_N-1]  │  │
//! │  barrier │ msg meta │  (one slot = header + payload bytes)     │  │
//! └──────────┴──────────┴─────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Workers block using `Atomics.wait()` (allowed inside dedicated workers) and
//! signal peers with `Atomics.notify()`.  This gives semantics equivalent to
//! MPI blocking sends/receives without any round-trip through the main thread.
//!
//! ## Current status
//!
//! This module provides a **stub** that compiles for `wasm32-unknown-unknown`
//! but panics on any real communication call.  The actual implementation will
//! be wired up by `fem-wasm` once:
//!
//! 1. The `WorkerLauncher` spawns N workers and allocates the SAB.
//! 2. Each worker receives `(rank, size, sab_handle)` through its `onmessage`
//!    init payload.
//! 3. `WasmWorkerBackend::from_init_msg(rank, size, sab)` builds a live backend.
//!
//! Until then, `WasmWorkerBackend::single()` provides a rank-0 / size-1 stub
//! suitable for serial WASM builds.

use fem_core::Rank;
use super::CommBackend;

// ── SharedArena ───────────────────────────────────────────────────────────────

/// Placeholder for the `SharedArrayBuffer` handle that the launcher passes to
/// each worker.
///
/// In the full implementation this wraps a `js_sys::SharedArrayBuffer` and
/// exposes typed views (`Int32Array`, `Uint8Array`) for Atomics operations.
/// Kept as `()` until the `wasm-bindgen` integration is wired in `fem-wasm`.
pub struct SharedArena(
    // Future: js_sys::SharedArrayBuffer
    (),
);

// ── WasmWorkerBackend ─────────────────────────────────────────────────────────

/// Web Worker simulation of an MPI process.
///
/// Construct via:
/// * [`WasmWorkerBackend::single()`] — rank-0/size-1 serial stub (always safe).
/// * [`WasmWorkerBackend::from_init_msg()`] — real multi-worker backend
///   (panics until the SAB implementation is complete).
pub struct WasmWorkerBackend {
    rank: u32,
    size: u32,
    /// Shared memory arena; `None` in the serial stub path.
    arena: Option<SharedArena>,
}

impl WasmWorkerBackend {
    /// Rank-0 / size-1 stub for serial WASM builds.
    ///
    /// All collective ops degenerate to no-ops; point-to-point ops panic.
    pub fn single() -> Self {
        WasmWorkerBackend { rank: 0, size: 1, arena: None }
    }

    /// Construct a live multi-worker backend from the init message sent by the
    /// [`WorkerLauncher`](super::super::launcher::wasm::WorkerLauncher).
    ///
    /// # Parameters
    /// * `rank`  — this worker's rank (0-based).
    /// * `size`  — total number of workers.
    /// * `arena` — shared `SharedArrayBuffer` arena (currently a placeholder).
    ///
    /// # Panics
    /// Panics with a `todo!` until the SAB-based implementation is complete.
    pub fn from_init_msg(rank: u32, size: u32, arena: SharedArena) -> Self {
        WasmWorkerBackend {
            rank,
            size,
            arena: Some(arena),
        }
    }

    /// `true` if this is the stub single-rank path (no real shared memory).
    #[inline]
    pub fn is_stub(&self) -> bool {
        self.arena.is_none()
    }
}

// ── Send + Sync ──────────────────────────────────────────────────────────────
//
// wasm32-unknown-unknown is single-threaded; there is no concurrent access.
// These impls are required so `Box<dyn CommBackend>` (which requires
// `CommBackend: Send + Sync`) can hold a `WasmWorkerBackend`.
//
// SAFETY: wasm32-unknown-unknown has a single-threaded execution model.
//         No other thread can access a `WasmWorkerBackend` instance.
unsafe impl Send for WasmWorkerBackend {}
unsafe impl Sync for WasmWorkerBackend {}

// ── CommBackend impl ─────────────────────────────────────────────────────────

impl CommBackend for WasmWorkerBackend {
    fn rank(&self) -> Rank { self.rank as Rank }
    fn size(&self) -> usize { self.size as usize }

    fn barrier(&self) {
        if self.is_stub() {
            return; // single rank: nothing to synchronise
        }
        // Full implementation: spin on an atomic counter in the SAB.
        // Each worker atomically increments a shared "arrived" counter; the
        // last arrival resets it and notifies all waiters.
        todo!("WasmWorkerBackend::barrier — SAB+Atomics impl pending")
    }

    fn allreduce_sum_f64(&self, local: f64) -> f64 {
        if self.is_stub() { return local; }
        // Full implementation: write local value to rank's slot in the SAB,
        // barrier, read all slots and sum, barrier again.
        todo!("WasmWorkerBackend::allreduce_sum_f64 — SAB impl pending")
    }

    fn allreduce_sum_i64(&self, local: i64) -> i64 {
        if self.is_stub() { return local; }
        todo!("WasmWorkerBackend::allreduce_sum_i64 — SAB impl pending")
    }

    fn broadcast_bytes(&self, _root: Rank, _buf: &mut Vec<u8>) {
        if self.is_stub() { return; }
        // Full implementation: root writes payload length + bytes into SAB,
        // barrier, all ranks read out the data.
        todo!("WasmWorkerBackend::broadcast_bytes — SAB impl pending")
    }

    fn send_bytes(&self, dest: Rank, _tag: i32, _data: &[u8]) {
        if self.is_stub() {
            panic!("WasmWorkerBackend (stub): cannot send to rank {dest}");
        }
        // Full implementation: write header + payload into dest's ring-buffer
        // slot in the SAB, then Atomics.notify() the destination worker.
        todo!("WasmWorkerBackend::send_bytes — SAB impl pending")
    }

    fn recv_bytes(&self, src: Rank, _tag: i32) -> Vec<u8> {
        if self.is_stub() {
            panic!("WasmWorkerBackend (stub): cannot receive from rank {src}");
        }
        // Full implementation: Atomics.wait() on this rank's ring-buffer slot
        // until a matching message appears, then copy out the payload.
        todo!("WasmWorkerBackend::recv_bytes — SAB impl pending")
    }

    fn alltoallv_bytes(&self, _sends: &[(Rank, Vec<u8>)]) -> Vec<(Rank, Vec<u8>)> {
        if self.is_stub() { return Vec::new(); }
        todo!("WasmWorkerBackend::alltoallv_bytes — SAB impl pending")
    }
}
