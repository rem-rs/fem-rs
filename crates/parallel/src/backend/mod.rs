//! Abstract communication backend.
//!
//! [`CommBackend`] is the single trait that all MPI-like transport layers must
//! implement.  This indirection lets the rest of the codebase compile and run
//! identically against three concrete backends:
//!
//! | Backend | When active | Status |
//! |---------|-------------|--------|
//! | [`SerialBackend`]       | non-wasm32, no `mpi` feature | ✅ complete |
//! | [`NativeMpiBackend`]    | non-wasm32, `mpi` feature    | ✅ Phase 10 |
//! | [`JsMpiBackend`]        | `wasm32` target              | ✅ via jsmpi |
//!
//! ## Byte-level API
//!
//! All point-to-point and collective operations move raw `[u8]` so that higher
//! layers can serialise any `T: bytemuck::Pod` without coupling the backend to
//! a specific numeric type.  [`Comm`](super::comm::Comm) exposes typed
//! convenience wrappers on top of these primitives.

use fem_core::Rank;

// ── CommBackend ───────────────────────────────────────────────────────────────

/// Transport-layer abstraction for MPI-style collective and point-to-point ops.
///
/// # Object safety
/// The trait is object-safe so that `Comm` can hold a `dyn CommBackend`
/// without monomorphising over the transport layer.
///
/// # Thread safety
/// `CommBackend` requires `Send + Sync` on all targets so that
/// [`Comm`] (which wraps a `Box<dyn CommBackend>` in an `Arc`) can itself be
/// `Send + Sync`.  This is necessary for [`ParallelMesh`] to implement
/// [`MeshTopology`] (which has `Send + Sync` supertraits).
///
/// On `wasm32` the target is single-threaded, so `Send + Sync` carry no real
/// cost.  [`JsMpiBackend`](wasm::JsMpiBackend) provides `unsafe impl
/// Send/Sync` since wasm32-unknown-unknown never has actual concurrent access.
pub trait CommBackend: Send + Sync {
    // ── topology ─────────────────────────────────────────────────────────────

    /// Rank of the calling process (0-based).
    fn rank(&self) -> Rank;

    /// Total number of processes.
    fn size(&self) -> usize;

    // ── synchronisation ──────────────────────────────────────────────────────

    /// Global barrier (`MPI_Barrier`).
    fn barrier(&self);

    // ── collectives ──────────────────────────────────────────────────────────

    /// AllReduce sum of a single `f64` value across all ranks.
    fn allreduce_sum_f64(&self, local: f64) -> f64;

    /// AllReduce sum of a single `i64` value across all ranks.
    fn allreduce_sum_i64(&self, local: i64) -> i64;

    /// Broadcast a byte buffer from `root` to all ranks in-place.
    ///
    /// On non-root ranks `buf` is resized to match the root's length before
    /// the payload is copied in.
    fn broadcast_bytes(&self, root: Rank, buf: &mut Vec<u8>);

    // ── point-to-point ───────────────────────────────────────────────────────

    /// Blocking send of raw bytes to `dest` with message tag `tag`.
    fn send_bytes(&self, dest: Rank, tag: i32, data: &[u8]);

    /// Blocking receive of raw bytes from `src` with message tag `tag`.
    ///
    /// Returns a newly-allocated `Vec<u8>` containing the message payload.
    fn recv_bytes(&self, src: Rank, tag: i32) -> Vec<u8>;

    // ── all-to-all ───────────────────────────────────────────────────────────

    /// Sparse variable-length all-to-all.
    ///
    /// Each entry `(dest_rank, payload)` in `sends` will be delivered to
    /// `dest_rank`.  Returns the list of `(src_rank, payload)` pairs received
    /// from all ranks that sent to this rank.
    ///
    /// This is the primitive used by [`GhostExchange`](super::ghost::GhostExchange)
    /// to exchange ghost-node ownership metadata.
    fn alltoallv_bytes(&self, sends: &[(Rank, Vec<u8>)]) -> Vec<(Rank, Vec<u8>)>;

    /// Split this communicator into sub-communicators based on `color`.
    ///
    /// Processes with the same `color` end up in the same sub-communicator.
    /// Within each sub-communicator, ranks are ordered by `key`.
    ///
    /// Returns `(new_rank, new_size, new_backend)` for the calling process's
    /// sub-communicator.  The default implementation panics — backends that
    /// support splitting must override this.
    fn split(&self, _color: i32, _key: i32) -> Box<dyn CommBackend> {
        panic!("CommBackend::split not supported by this backend");
    }
}

// ── platform modules ─────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
pub mod native;

/// In-process channel backend for [`ThreadLauncher`] (non-wasm32 only).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod channel;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
