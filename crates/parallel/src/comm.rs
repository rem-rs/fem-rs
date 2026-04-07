//! [`Universe`] and [`Comm`] — the two public handles for the parallel environment.
//!
//! ## Entry points
//!
//! | Target | Recommended entry point |
//! |--------|------------------------|
//! | non-wasm32 + `mpi` feature  | [`launcher::native::MpiLauncher::init()`](super::launcher::native::MpiLauncher) |
//! | non-wasm32, no `mpi` feature | [`launcher::native::ThreadLauncher`](super::launcher::native::ThreadLauncher) or `Universe::init()` |
//! | `wasm32` | [`launcher::wasm::WorkerLauncher`](super::launcher::wasm::WorkerLauncher) or `Universe::init()` for serial |
//!
//! `Universe::init()` is the legacy entry point; it internally creates the
//! appropriate backend for the current build configuration.

use fem_core::Rank;
use bytemuck::Pod;

use crate::backend::CommBackend;

// `CommBackend: Send + Sync` (supertrait), so `Box<dyn CommBackend>` is
// automatically `Send + Sync` on all targets, including wasm32 (where
// WasmWorkerBackend carries unsafe impls because the target is single-threaded).
type BackendBox = Box<dyn CommBackend>;

// ── Comm ─────────────────────────────────────────────────────────────────────

/// Handle to a communicator (equivalent to `MPI_Comm`).
///
/// Currently wraps `MPI_COMM_WORLD`; sub-communicators can be added in
/// Phase 10 via `Comm::split`.
///
/// `Comm` is cheaply cloneable: the internal backend is reference-counted.
pub struct Comm {
    inner: std::sync::Arc<BackendBox>,
}

// Manual Clone because BackendBox is not Clone, but Arc is.
impl Clone for Comm {
    fn clone(&self) -> Self {
        Comm { inner: std::sync::Arc::clone(&self.inner) }
    }
}

impl Comm {
    /// Construct from any `CommBackend` implementation.
    ///
    /// This is the integration point for custom backends and for the launchers.
    pub fn from_backend(backend: BackendBox) -> Self {
        Comm { inner: std::sync::Arc::new(backend) }
    }

    // ── topology ─────────────────────────────────────────────────────────────

    /// Rank of the calling process (0-based).
    #[inline]
    pub fn rank(&self) -> Rank { self.inner.rank() }

    /// Total number of processes in this communicator.
    #[inline]
    pub fn size(&self) -> usize { self.inner.size() }

    /// `true` iff this is rank 0 (the "root" process).
    #[inline]
    pub fn is_root(&self) -> bool { self.inner.rank() == 0 }

    // ── synchronisation ──────────────────────────────────────────────────────

    /// Global barrier — blocks until every rank has called this.
    #[inline]
    pub fn barrier(&self) { self.inner.barrier(); }

    // ── typed collectives ────────────────────────────────────────────────────

    /// AllReduce sum of a single `f64` across all ranks.
    #[inline]
    pub fn allreduce_sum_f64(&self, local: f64) -> f64 {
        self.inner.allreduce_sum_f64(local)
    }

    /// AllReduce sum of a single `i64` across all ranks.
    #[inline]
    pub fn allreduce_sum_i64(&self, local: i64) -> i64 {
        self.inner.allreduce_sum_i64(local)
    }

    /// Broadcast a `usize` from `root` to all ranks.
    pub fn broadcast_usize(&self, root: Rank, val: usize) -> usize {
        let mut buf = (val as u64).to_le_bytes().to_vec();
        self.inner.broadcast_bytes(root, &mut buf);
        u64::from_le_bytes(buf.try_into().expect("broadcast_usize: wrong buffer length")) as usize
    }

    // ── typed point-to-point ─────────────────────────────────────────────────

    /// Send a slice of `Pod` values to `dest`.
    pub fn send<T: Pod>(&self, dest: Rank, tag: i32, data: &[T]) {
        self.inner.send_bytes(dest, tag, bytemuck::cast_slice(data));
    }

    /// Receive a `Vec<T: Pod>` from `src`.
    pub fn recv<T: Pod>(&self, src: Rank, tag: i32) -> Vec<T> {
        let bytes = self.inner.recv_bytes(src, tag);
        // Ensure alignment for T (copy into aligned allocation if needed).
        let n = bytes.len() / std::mem::size_of::<T>();
        let mut out: Vec<T> = Vec::with_capacity(n);
        // SAFETY: T: Pod, so any bit pattern is valid; we manually fill.
        for i in 0..n {
            let start = i * std::mem::size_of::<T>();
            let end = start + std::mem::size_of::<T>();
            let t: T = *bytemuck::from_bytes(&bytes[start..end]);
            out.push(t);
        }
        out
    }

    /// Broadcast a raw byte buffer from `root` to all ranks in-place.
    ///
    /// On non-root ranks `buf` is overwritten with the root's contents.
    #[inline]
    pub fn broadcast_bytes(&self, root: Rank, buf: &mut Vec<u8>) {
        self.inner.broadcast_bytes(root, buf);
    }

    // ── raw byte operations (for GhostExchange + tests) ───────────────────────

    /// Raw byte send — blocking point-to-point.
    #[inline]
    pub fn send_bytes(&self, dest: Rank, tag: i32, data: &[u8]) {
        self.inner.send_bytes(dest, tag, data);
    }

    /// Raw byte receive — blocking point-to-point.
    #[inline]
    pub fn recv_bytes(&self, src: Rank, tag: i32) -> Vec<u8> {
        self.inner.recv_bytes(src, tag)
    }

    /// Sparse all-to-all variable-length byte exchange.
    #[inline]
    pub fn alltoallv_bytes(
        &self,
        sends: &[(Rank, Vec<u8>)],
    ) -> Vec<(Rank, Vec<u8>)> {
        self.inner.alltoallv_bytes(sends)
    }

    /// Split this communicator into sub-communicators.
    ///
    /// Processes with the same `color` end up in the same sub-communicator.
    /// Within each sub-communicator, ranks are ordered by `key`.
    ///
    /// Returns a new `Comm` handle for the sub-communicator containing this rank.
    pub fn split(&self, color: i32, key: i32) -> Comm {
        let backend = self.inner.split(color, key);
        Comm { inner: std::sync::Arc::new(backend) }
    }
}

// ── Universe ─────────────────────────────────────────────────────────────────

/// MPI execution environment handle.
///
/// Exactly one `Universe` may exist per process.  Drop it to trigger
/// `MPI_Finalize` (or the equivalent teardown for other backends).
///
/// Prefer using a typed [`Launcher`](super::launcher::Launcher) instead of
/// `Universe` directly; `Universe` is kept for backwards compatibility and
/// for quick one-off scripts.
pub struct Universe {
    #[cfg(all(not(target_arch = "wasm32"), feature = "mpi"))]
    mpi_universe: ::mpi::environment::Universe,
}

impl Universe {
    /// Initialise the parallel environment.
    ///
    /// Returns `None` if already initialised.  On `wasm32` this always
    /// succeeds (returning a stub single-rank universe).
    pub fn init() -> Option<Self> {
        #[cfg(all(not(target_arch = "wasm32"), feature = "mpi"))]
        {
            ::mpi::initialize().map(|mpi_universe| Universe { mpi_universe })
        }

        #[cfg(not(all(not(target_arch = "wasm32"), feature = "mpi")))]
        {
            Some(Universe {})
        }
    }

    /// Return a handle to the world communicator.
    pub fn world(&self) -> Comm {
        #[cfg(all(not(target_arch = "wasm32"), feature = "mpi"))]
        {
            use crate::backend::native::NativeMpiBackend;
            return Comm::from_backend(Box::new(
                NativeMpiBackend::from_world(&self.mpi_universe),
            ));
        }

        #[cfg(target_arch = "wasm32")]
        {
            use crate::backend::wasm::JsMpiBackend;
            return Comm::from_backend(Box::new(JsMpiBackend::serial()));
        }

        #[cfg(all(not(target_arch = "wasm32"), not(feature = "mpi")))]
        {
            use crate::backend::native::SerialBackend;
            Comm::from_backend(Box::new(SerialBackend))
        }
    }
}
