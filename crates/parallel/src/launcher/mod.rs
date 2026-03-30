//! Process / worker launcher abstraction.
//!
//! A [`Launcher`] is responsible for initialising the parallel environment and
//! handing the caller a [`Comm`](super::comm::Comm) that is already bound to
//! the correct rank.
//!
//! ## Platform implementations
//!
//! | Launcher | Target | Description |
//! |----------|--------|-------------|
//! | [`MpiLauncher`](native::MpiLauncher)     | non-wasm32 + `mpi` feature | Wraps `mpi::initialize()`. MPI itself handles multi-process spawn (via `mpirun`). |
//! | [`ThreadLauncher`](native::ThreadLauncher) | non-wasm32               | Spawns `N` OS threads sharing a fake in-process communicator. Useful for unit tests without an MPI install. |
//! | [`WorkerLauncher`](wasm::WorkerLauncher)  | `wasm32`                  | Spawns `N` Web Workers and hands each one a `SharedArrayBuffer` arena. (Stub — implementation in `fem-wasm`.) |
//!
//! ## Usage (native MPI example)
//!
//! ```ignore
//! use fem_parallel::launcher::native::MpiLauncher;
//! use fem_parallel::Launcher;
//!
//! // mpirun -n 4 ./my_solver
//! let launcher = MpiLauncher::init().expect("MPI already initialised");
//! let comm = launcher.world_comm();
//! println!("rank {} of {}", comm.rank(), comm.size());
//! ```

use crate::comm::Comm;

// ── Launcher trait ────────────────────────────────────────────────────────────

/// Common interface for all parallel environment initialisers.
///
/// Implementors set up the communication infrastructure and return a
/// [`Comm`] that callers use to exchange data.
pub trait Launcher: Sized {
    /// Initialise the environment.  Returns `None` if the environment has
    /// already been initialised (mirrors `mpi::initialize()` semantics).
    fn init() -> Option<Self>;

    /// Return the world communicator (equivalent to `MPI_COMM_WORLD`).
    fn world_comm(&self) -> Comm;
}

// ── WorkerConfig ─────────────────────────────────────────────────────────────

/// Configuration for multi-worker / multi-process launches.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Number of parallel workers (MPI ranks / Web Workers / OS threads).
    pub n_workers: usize,
    /// Optional stack size per worker in bytes (platform-dependent).
    pub stack_bytes: Option<usize>,
}

impl WorkerConfig {
    pub fn new(n_workers: usize) -> Self {
        WorkerConfig { n_workers, stack_bytes: None }
    }
}

// ── platform modules ─────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
pub mod native;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

