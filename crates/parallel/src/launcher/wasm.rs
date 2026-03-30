//! Web Worker launcher for `wasm32` targets.
//!
//! ## Planned architecture
//!
//! ```text
//!  JS / main thread
//!  ┌──────────────────────────────────────────────────────────────┐
//!  │  WorkerLauncher::spawn(n)                                    │
//!  │    1. allocate SharedArrayBuffer(arena_bytes)                │
//!  │    2. for rank in 0..n:                                      │
//!  │         worker = new Worker("fem_solver.js")                 │
//!  │         worker.postMessage({ rank, size: n, sab })           │
//!  │    3. wait for all workers to post "ready"                   │
//!  │    4. post "go" broadcast → workers call fem_entry(comm)     │
//!  └──────────────────────────────────────────────────────────────┘
//!
//!  Inside each worker (wasm32 context)
//!  ┌──────────────────────────────────────────────────────────────┐
//!  │  onmessage({ rank, size, sab })                              │
//!  │    backend = WasmWorkerBackend::from_init_msg(rank, size, sab)│
//!  │    comm    = Comm::from_backend(backend)                     │
//!  │    fem_entry(comm)   // user solver entry point              │
//!  └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! The SAB is divided into fixed-size slots: one control word (for barriers and
//! message metadata) per rank plus a payload ring-buffer.  Blocking is achieved
//! via `Atomics.wait()` (permitted inside dedicated workers under COOP/COEP
//! HTTP headers) and `Atomics.notify()`.
//!
//! ## Current status
//!
//! This module is a **stub**.  The types exist and the code compiles for
//! `wasm32-unknown-unknown`, but `WorkerLauncher::spawn` panics with `todo!`.
//! Wiring up the `wasm-bindgen` + `js-sys` + `web-sys` calls is deferred to
//! the `fem-wasm` crate integration (Phase 11).

use crate::backend::wasm::WasmWorkerBackend;
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
        Comm::from_backend(Box::new(WasmWorkerBackend::single()))
    }
}

// ── WorkerInitMsg ─────────────────────────────────────────────────────────────

/// Typed representation of the init message a worker receives from the launcher.
///
/// In the full implementation this is deserialised from the `MessageEvent` data
/// passed by `postMessage({ rank, size, sab })`.
///
/// Fields are public so `fem-wasm` can construct this from `wasm-bindgen`
/// callbacks without requiring a dependency back on `fem-parallel`'s internals.
pub struct WorkerInitMsg {
    pub rank: u32,
    pub size: u32,
    // Future: pub sab: js_sys::SharedArrayBuffer,
}

impl WorkerInitMsg {
    /// Build a [`Comm`] from this init message.
    ///
    /// # Panics
    /// Panics if `size > 1` until the SAB backend is implemented.
    pub fn into_comm(self) -> Comm {
        let backend = if self.size <= 1 {
            WasmWorkerBackend::single()
        } else {
            // The SharedArena handle will be passed here once wasm-bindgen
            // integration is complete.
            todo!("WorkerInitMsg::into_comm — SharedArena pending")
        };
        Comm::from_backend(Box::new(backend))
    }
}
