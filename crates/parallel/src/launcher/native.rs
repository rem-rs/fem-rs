//! Native (non-wasm32) launchers.
//!
//! * [`MpiLauncher`] — thin wrapper around `mpi::initialize()`.  The actual
//!   multi-process spawning is done externally by `mpirun` / `mpiexec`; this
//!   launcher only initialises the MPI runtime inside the already-started
//!   process.
//!
//! * [`ThreadLauncher`] — spawns N OS threads each running the user function
//!   with a fake in-process communicator backed by [`ChannelBackend`].  No MPI
//!   install required; suitable for unit tests and CI environments.

use std::sync::Arc;
use crate::comm::Comm;
use crate::launcher::{Launcher, WorkerConfig};

// ── Rayon / MPI thread-count coordination ────────────────────────────────────

/// Called once after MPI init to cap Rayon threads to `physical_cpus / mpi_size`.
///
/// Prevents over-subscription when running multiple MPI ranks on one node.
/// A floor of 1 thread is enforced.  Subsequent calls (from re-entrant code or
/// tests) are silently ignored because Rayon's global pool is already set.
///
/// The env var `RAYON_NUM_THREADS` takes precedence when already set; Rayon
/// reads it itself during `build_global`, so we skip our calculation in that
/// case.
#[cfg(all(not(target_arch = "wasm32"), feature = "mpi"))]
fn configure_rayon_for_mpi(mpi_size: usize) {
    use std::sync::OnceLock;
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        // Respect explicit user override.
        if std::env::var_os("RAYON_NUM_THREADS").is_some() {
            return;
        }
        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let per_rank = (logical / mpi_size).max(1);
        // Silently ignore build_global errors (e.g. pool already configured in tests).
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(per_rank)
            .build_global();
        log::debug!(
            "fem-parallel: MPI size={mpi_size}, logical CPUs={logical} \
             → Rayon threads/rank={per_rank}"
        );
    });
}

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
            use ::mpi::traits::Communicator;
            use crate::backend::native::NativeMpiBackend;
            let mpi_size = self.universe.world().size() as usize;
            configure_rayon_for_mpi(mpi_size);
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
/// backed by [`ChannelBackend`] — a shared `Mutex`/`Condvar` in-process
/// communicator that correctly implements all collective and point-to-point
/// operations.
///
/// # Usage
/// ```no_run
/// use fem_parallel::launcher::native::ThreadLauncher;
/// use fem_parallel::WorkerConfig;
///
/// ThreadLauncher::new(WorkerConfig::new(4)).launch(|comm| {
///     println!("rank {} / {}", comm.rank(), comm.size());
/// });
/// ```
pub struct ThreadLauncher {
    config: WorkerConfig,
}

impl ThreadLauncher {
    pub fn new(config: WorkerConfig) -> Self {
        ThreadLauncher { config }
    }

    /// Spawn `config.n_workers` threads, each executing `f(comm)`.
    ///
    /// Blocks until all threads complete.  For `n_workers == 1` no new thread
    /// is spawned (the closure runs on the calling thread with a
    /// [`SerialBackend`](crate::backend::native::SerialBackend)).
    pub fn launch<F>(&self, f: F)
    where
        F: Fn(Comm) + Send + Sync + 'static,
    {
        let n = self.config.n_workers;

        if n <= 1 {
            use crate::backend::native::SerialBackend;
            f(Comm::from_backend(Box::new(SerialBackend)));
            return;
        }

        // Build the shared channel state and wrap `f` in an Arc for sharing.
        use crate::backend::channel::{ChannelBackend, ChannelShared};
        let shared = ChannelShared::new(n);
        let f_arc  = Arc::new(f);

        let handles: Vec<_> = (0..n as i32)
            .map(|rank| {
                let shared_clone = Arc::clone(&shared);
                let f_clone      = Arc::clone(&f_arc);
                std::thread::spawn(move || {
                    let backend = ChannelBackend::new(rank, shared_clone);
                    let comm    = Comm::from_backend(Box::new(backend));
                    f_clone(comm);
                })
            })
            .collect();

        for h in handles {
            h.join().expect("ThreadLauncher: worker thread panicked");
        }
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
