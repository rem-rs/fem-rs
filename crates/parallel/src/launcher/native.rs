//! Native (non-wasm32) launchers.
//!
//! * [`MpiLauncher`] — thin wrapper around `mpi::initialize()`.  The actual
//!   multi-process spawning is done externally by `mpirun` / `mpiexec`; this
//!   launcher only initialises the MPI runtime inside the already-started
//!   process.
//!
//! * [`ThreadLauncher`] — spawns N OS threads each running the user function
//!   with a fake in-process communicator backed by channels.  No MPI install
//!   required; suitable for unit tests and CI environments.

use crate::comm::Comm;
use crate::launcher::{Launcher, WorkerConfig};

// ── MpiLauncher ───────────────────────────────────────────────────────────────

/// Initialises the MPI runtime and exposes `MPI_COMM_WORLD`.
///
/// Requires the `mpi` feature.  When the feature is disabled this type still
/// exists but `init()` returns a single-rank serial launcher instead (so
/// code that calls `MpiLauncher::init()` compiles without `#[cfg]` guards).
pub struct MpiLauncher {
    #[cfg(feature = "mpi")]
    universe: ::mpi::environment::Universe,
}

impl Launcher for MpiLauncher {
    fn init() -> Option<Self> {
        #[cfg(feature = "mpi")]
        {
            ::mpi::initialize().map(|universe| MpiLauncher { universe })
        }
        #[cfg(not(feature = "mpi"))]
        {
            Some(MpiLauncher {})
        }
    }

    fn world_comm(&self) -> Comm {
        #[cfg(feature = "mpi")]
        {
            use crate::backend::native::NativeMpiBackend;
            Comm::from_backend(Box::new(NativeMpiBackend::from_world(&self.universe)))
        }
        #[cfg(not(feature = "mpi"))]
        {
            use crate::backend::native::SerialBackend;
            Comm::from_backend(Box::new(SerialBackend))
        }
    }
}

// ── ThreadLauncher ────────────────────────────────────────────────────────────

/// In-process multi-threaded launcher for testing without an MPI install.
///
/// Spawns `config.n_workers` OS threads.  Each thread receives a [`Comm`]
/// backed by a channel-based fake communicator.
///
/// # Limitations
/// * Collective ops (`allreduce`, `barrier`) are implemented via a
///   `Mutex`+`Condvar` rendezvous; they are correct but not high-performance.
/// * Not suitable for production — use [`MpiLauncher`] for real runs.
///
/// # Status
/// The channel backend is a **Phase 10 TODO**.  Currently `launch` runs `f`
/// once with a single-rank serial comm (equivalent to `n_workers = 1`).
pub struct ThreadLauncher {
    config: WorkerConfig,
}

impl ThreadLauncher {
    pub fn new(config: WorkerConfig) -> Self {
        ThreadLauncher { config }
    }

    /// Spawn `config.n_workers` threads, each executing `f(comm)`.
    ///
    /// Blocks until all threads complete.
    pub fn launch<F>(&self, f: F)
    where
        F: Fn(Comm) + Send + Sync + 'static,
    {
        let n = self.config.n_workers;
        if n <= 1 {
            // Fast path: no spawning needed for a single worker.
            use crate::backend::native::SerialBackend;
            f(Comm::from_backend(Box::new(SerialBackend)));
            return;
        }

        // Phase 10: build shared channel backend, spawn threads.
        // For now fall back to serial (single thread).
        log::warn!(
            "ThreadLauncher: multi-worker channel backend not yet implemented; \
             running single-threaded (n_workers forced to 1)"
        );
        use crate::backend::native::SerialBackend;
        f(Comm::from_backend(Box::new(SerialBackend)));
    }
}

impl Launcher for ThreadLauncher {
    fn init() -> Option<Self> {
        Some(ThreadLauncher::new(WorkerConfig::new(1)))
    }

    fn world_comm(&self) -> Comm {
        use crate::backend::native::SerialBackend;
        Comm::from_backend(Box::new(SerialBackend))
    }
}
