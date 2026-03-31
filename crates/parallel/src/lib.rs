//! # fem-parallel
//!
//! MPI-parallel infrastructure: abstract backend, distributed mesh primitives,
//! halo exchange, and process launchers.
//!
//! ## Feature flags
//!
//! | Feature | Effect |
//! |---------|--------|
//! | `mpi`   | Links against rsmpi (`mpi 0.8`); enables [`NativeMpiBackend`](backend::native::NativeMpiBackend) and [`MpiLauncher`](launcher::native::MpiLauncher). |
//! | *(none)*| Serial / WASM stub — same public API, no MPI install required. |
//!
//! ## Target matrix
//!
//! | `cfg` | Active backend | Launcher |
//! |-------|---------------|----------|
//! | non-wasm32 + `mpi` feature | `NativeMpiBackend` (rsmpi) | `MpiLauncher` |
//! | non-wasm32, no `mpi`       | `SerialBackend`             | `ThreadLauncher` |
//! | `wasm32`                   | `WasmWorkerBackend` (stub)  | `WorkerLauncher` |
//!
//! ## Quick start
//!
//! ```ignore
//! // native MPI build (mpirun -n 4 ./solver)
//! use fem_parallel::launcher::{Launcher, native::MpiLauncher};
//! let launcher = MpiLauncher::init().unwrap();
//! let comm = launcher.world_comm();
//! println!("rank {} / {}", comm.rank(), comm.size());
//! ```
//!
//! ```ignore
//! // WASM single-rank (serial stub until Web Worker backend lands)
//! use fem_parallel::launcher::{Launcher, wasm::WorkerLauncher};
//! let launcher = WorkerLauncher::init().unwrap();
//! let comm = launcher.world_comm();
//! assert_eq!(comm.size(), 1);
//! ```

pub mod backend;
pub mod comm;
pub mod ghost;
pub mod launcher;
pub mod par_mesh;
pub mod par_simplex;
pub mod partition;

// Flat re-exports for ergonomic `use fem_parallel::*`.
pub use comm::{Comm, Universe};
pub use ghost::GhostExchange;
pub use launcher::{Launcher, WorkerConfig};
pub use par_mesh::ParallelMesh;
pub use par_simplex::partition_simplex;
pub use partition::MeshPartition;
