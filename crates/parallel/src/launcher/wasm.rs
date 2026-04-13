//! Web Worker launcher for `wasm32` targets.
//!
//! ## Architecture
//!
//! ```text
//!  JS / main thread
//!  ┌──────────────────────────────────────────────────────────────┐
//!  │  WorkerLauncher::spawn_async(...)                            │
//!  │    → jsmpi::launcher::create_job(...)                        │
//!  │    → returns WasmJob handle immediately                      │
//!  │                                                              │
//!  │  Each worker loads wasm module, calls jsmpi_main()           │
//!  └──────────────────────────────────────────────────────────────┘
//!
//!  Inside each worker (wasm32 context)
//!  ┌──────────────────────────────────────────────────────────────┐
//!  │  onmessage({ rank, size, comm })                             │
//!  │    → worker.js sets __jsmpi_rank / __jsmpi_size              │
//!  │    → imports module, calls jsmpi_main()                      │
//!  │                                                              │
//!  │  jsmpi_main():                                               │
//!  │    init = WorkerInitMsg::from_jsmpi_env()                    │
//!  │    comm = init.into_comm()                                   │
//!  │    … run parallel solver …                                   │
//!  │    jsmpi::runtime::mark_finished()                           │
//!  └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! Communication is handled by the jsmpi crate which provides MPI-style
//! messaging via Web Workers + SharedArrayBuffer.
//!
//! ## Usage
//!
//! **Main thread (JS glue)**:
//! ```ignore
//! let launcher = WorkerLauncher::new(WorkerConfig::new(4));
//! let job = launcher.spawn_async(
//!     "worker.js",
//!     "fem_wasm.js",
//!     |kind, msg| web_sys::console::log_1(&format!("[{kind}] {msg}").into()),
//!     |done, total| web_sys::console::log_1(&format!("{done}/{total} finished").into()),
//! );
//! // job is dropped → workers are terminated (or keep alive as needed)
//! ```
//!
//! **Inside each worker**:
//! ```ignore
//! #[wasm_bindgen]
//! pub fn jsmpi_main() {
//!     let init = WorkerInitMsg::from_jsmpi_env().unwrap();
//!     let comm = init.into_comm();
//!     // … use comm for parallel computation …
//!     jsmpi::runtime::mark_finished().ok();
//! }
//! ```

use crate::backend::wasm::JsMpiBackend;
use crate::comm::Comm;
use crate::launcher::{Launcher, WorkerConfig};
use jsmpi::topology::Communicator;

// ── WorkerLauncher ────────────────────────────────────────────────────────────

/// Spawns Web Workers and coordinates their shared-memory communication arena.
///
/// Intended to be constructed from JS (via `wasm-bindgen`) or from a WASM
/// entry-point that has already received the init message from the host page.
pub struct WorkerLauncher {
    config: WorkerConfig,
}

impl WorkerLauncher {
    pub fn new(config: WorkerConfig) -> Self {
        WorkerLauncher { config }
    }

    /// Spawn `config.n_workers` Web Workers and launch `f` on each one.
    ///
    /// # Single-rank mode
    /// If `n_workers <= 1`, runs `f` synchronously on this thread.
    ///
    /// # Multi-rank mode
    /// Panics — use [`spawn_async`](WorkerLauncher::spawn_async) instead.
    /// The browser main thread cannot block, so synchronous multi-worker
    /// launch is not possible.
    pub fn spawn<F>(&self, _f: F)
    where
        F: Fn(Comm) + 'static,
    {
        if self.config.n_workers <= 1 {
            // Fast path: no workers needed for a single process.
            let comm = self.world_comm();
            _f(comm);
            return;
        }
        todo!(
            "WorkerLauncher::spawn — synchronous multi-worker launch is not supported \
             in the browser. Use spawn_async() from the main thread, or call \
             WorkerInitMsg::from_jsmpi_env() inside each Web Worker's jsmpi_main()."
        )
    }

    /// Asynchronously spawn Web Workers for parallel computation.
    ///
    /// Returns a [`WasmJob`] handle immediately.  The actual solver logic
    /// runs inside each worker's `jsmpi_main()` export (called by worker.js).
    ///
    /// # Arguments
    /// * `worker_url` — URL of the worker script (e.g. `"worker.js"`).
    /// * `module_url` — URL of the WASM module entry (e.g. `"fem_wasm.js"`).
    /// * `on_log` — callback `(kind, text)` for log/transport/error messages.
    /// * `on_complete` — callback `(finished_ranks, total_ranks)` when all done.
    ///
    /// # Single-rank fast path
    /// If `n_workers <= 1`, returns a no-op `WasmJob` (no workers spawned).
    pub fn spawn_async(
        &self,
        worker_url: &str,
        module_url: &str,
        on_log: impl Fn(String, String) + 'static,
        on_complete: impl Fn(u32, u32) + 'static,
    ) -> WasmJob {
        if self.config.n_workers <= 1 {
            return WasmJob { inner: None };
        }
        let job = jsmpi::launcher::create_job(
            worker_url,
            module_url,
            self.config.n_workers as u32,
            on_log,
            |_state| {},  // state changes handled internally
            on_complete,
        );
        WasmJob { inner: Some(job) }
    }
}

impl Launcher for WorkerLauncher {
    fn init() -> Option<Self> {
        Some(WorkerLauncher::new(WorkerConfig::new(1)))
    }

    fn world_comm(&self) -> Comm {
        Comm::from_backend(Box::new(JsMpiBackend::serial()))
    }
}

// ── WasmJob ──────────────────────────────────────────────────────────────────

/// Handle to a set of running Web Workers.
///
/// Drop or call [`terminate`](WasmJob::terminate) to stop all workers.
/// Constructed by [`WorkerLauncher::spawn_async`].
pub struct WasmJob {
    inner: Option<jsmpi::launcher::Job>,
}

impl WasmJob {
    /// Terminate all workers immediately.
    pub fn terminate(&self) {
        if let Some(ref job) = self.inner {
            job.terminate();
        }
    }

    /// `true` if this is the single-rank no-op job (no workers spawned).
    pub fn is_noop(&self) -> bool {
        self.inner.is_none()
    }
}

// ── WorkerInitMsg ─────────────────────────────────────────────────────────────

/// Typed representation of the init message a worker receives from the launcher.
///
/// In the full implementation this is deserialised from the `MessageEvent` data
/// passed by `postMessage({ rank, size, comm })`.
///
/// Fields are public so `fem-wasm` can construct this from `wasm-bindgen`
/// callbacks without requiring a dependency back on `fem-parallel`'s internals.
pub struct WorkerInitMsg {
    pub rank: u32,
    pub size: u32,
    /// Optional jsmpi communicator for multi-worker mode.
    pub comm: Option<jsmpi::SimpleCommunicator>,
}

impl WorkerInitMsg {
    /// Build from the live jsmpi browser environment.
    ///
    /// Must be called inside a Web Worker after jsmpi's `worker.js` has set
    /// `__jsmpi_rank` / `__jsmpi_size` on the global scope and routed the
    /// init message.
    ///
    /// # Errors
    /// Returns an error string if jsmpi initialisation fails (e.g. the
    /// global environment variables are not set).
    pub fn from_jsmpi_env() -> Result<Self, String> {
        let universe = jsmpi::initialize()
            .map_err(|e| format!("jsmpi init failed: {e}"))?;
        let world = universe.world();
        let rank = world.rank() as u32;
        let size = world.size() as u32;
        Ok(WorkerInitMsg {
            rank,
            size,
            comm: if size > 1 { Some(world) } else { None },
        })
    }

    /// Build a [`Comm`] from this init message.
    ///
    /// If `comm` is `Some`, creates a real multi-worker backend via jsmpi.
    /// Otherwise falls back to a serial stub.
    pub fn into_comm(self) -> Comm {
        let backend = match self.comm {
            Some(communicator) => JsMpiBackend::from_communicator(communicator),
            None => JsMpiBackend::serial(),
        };
        Comm::from_backend(Box::new(backend))
    }
}
