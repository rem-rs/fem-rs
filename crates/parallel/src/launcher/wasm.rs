//! Web Worker launcher for `wasm32` targets.
//!
//! ## Architecture
//!
//! ```text
//!  JS / main thread
//!  ┌──────────────────────────────────────────────────────────────┐
//!  │  WorkerLauncher::spawn(n)                                    │
//!  │    1. create jsmpi runtime for each rank                     │
//!  │    2. for rank in 0..n:                                      │
//!  │         worker = new Worker("fem_solver.js")                 │
//!  │         worker.postMessage({ rank, size: n, comm })          │
//!  │    3. wait for all workers to post "ready"                   │
//!  │    4. post "go" broadcast → workers call fem_entry(comm)     │
//!  └──────────────────────────────────────────────────────────────┘
//!
//!  Inside each worker (wasm32 context)
//!  ┌──────────────────────────────────────────────────────────────┐
//!  │  onmessage({ rank, size, comm })                             │
//!  │    backend = JsMpiBackend::from_communicator(comm)            │
//!  │    comm    = Comm::from_backend(backend)                     │
//!  │    fem_entry(comm)   // user solver entry point              │
//!  └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! Communication is handled by the jsmpi crate which provides MPI-style
//! messaging via Web Workers + SharedArrayBuffer.
//!
//! ## Current status
//!
//! `WorkerLauncher::spawn` for multi-worker mode still panics with `todo!`.
//! The `JsMpiBackend` is fully wired; `WorkerInitMsg::into_comm` can create
//! a live multi-worker backend when given a jsmpi `SimpleCommunicator`.
//! Wiring up the JS-side worker spawning is deferred to `fem-wasm`.

use crate::backend::wasm::JsMpiBackend;
use crate::comm::Comm;
use crate::launcher::{Launcher, WorkerConfig};

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
    /// # Panics
    /// Currently panics with `todo!` — implementation requires `wasm-bindgen`
    /// integration in `fem-wasm`.
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
            "WorkerLauncher::spawn — wasm-bindgen + SharedArrayBuffer integration \
             pending in fem-wasm Phase 11"
        )
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
