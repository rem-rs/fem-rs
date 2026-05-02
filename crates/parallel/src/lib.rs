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
//!
//! ## Environment variables
//!
//! See [`env`] — notably [`FEM_PARALLEL_LOCAL_RAYON_MIN`](env::FEM_PARALLEL_LOCAL_RAYON_MIN) for
//! thresholds on local Rayon parallelism before MPI collectives,
//! [`FEM_LINALG_SPMV_PARALLEL_MIN_ROWS`] for local CSR SpMV threading (`fem-linalg`, re-exported),
//! and [`FEM_ASSEMBLY_PARALLEL_MIN_ELEMS`] for local volume assembly (`fem-assembly`, re-exported).
//!
//! Halo exchange for [`ParCsrMatrix`](crate::par_csr::ParCsrMatrix) uses non-blocking MPI
//! when [`Comm::is_native_mpi`](comm::Comm::is_native_mpi) is true, overlapping the diagonal
//! SpMV with in-flight receives/sends (see [`GhostExchange::forward_overlapping`](crate::ghost::GhostExchange::forward_overlapping)).

pub mod backend;
pub mod comm;
pub mod dof_partition;
pub mod env;
pub mod ghost;
pub mod launcher;
pub mod mesh_serde;
pub mod metis;
pub mod par_amg;
pub mod par_assembler;
pub mod par_csr;
pub mod par_mesh;
pub mod par_mixed_assembler;
pub mod par_ras;
pub mod par_simplex;
pub mod par_solver;
pub mod par_space;
pub mod par_vector;
pub mod par_vector_assembler;
pub mod partition;

#[cfg(test)]
mod mpi_test_env;
#[cfg(feature = "hdf5")]
pub mod par_hdf5;
#[cfg(feature = "hdf5")]
pub mod checkpoint;

// Flat re-exports for ergonomic `use fem_parallel::*`.
pub use comm::{Comm, Universe};
pub use env::{local_rayon_min, FEM_PARALLEL_LOCAL_RAYON_MIN};
pub use fem_assembly::{assembly_parallel_min_elems, FEM_ASSEMBLY_PARALLEL_MIN_ELEMS};
pub use fem_linalg::{spmv_parallel_min_rows, FEM_LINALG_SPMV_PARALLEL_MIN_ROWS};
pub use dof_partition::DofPartition;
pub use ghost::GhostExchange;
pub use launcher::{Launcher, WorkerConfig};
pub use metis::{MetisPartitioner, MetisOptions, partition_simplex_metis, partition_simplex_metis_streaming};
pub use par_assembler::ParAssembler;
pub use par_amg::{ParAmgConfig, ParAmgHierarchy, par_solve_pcg_amg};
pub use par_csr::ParCsrMatrix;
pub use par_mesh::ParallelMesh;
pub use par_mixed_assembler::ParMixedAssembler;
pub use par_ras::{
	RasConfig, RasHpcDiagnostics, RasLocalSolverKind, RasPrecond, par_solve_gmres_ras,
	par_solve_pcg_ras, summarize_ras_hpc,
};
pub use par_simplex::{partition_simplex, partition_simplex_streaming};
pub use par_solver::{par_solve_cg, par_solve_gmres_jacobi, par_solve_pcg_jacobi, par_solve_minres};
pub use par_space::ParallelFESpace;
pub use par_vector::ParVector;
pub use par_vector_assembler::ParVectorAssembler;
pub use partition::MeshPartition;

#[cfg(feature = "hdf5")]
pub use par_hdf5::{par_write_mesh_and_fields, ParHdf5Options, ParallelWriteMode};
#[cfg(feature = "hdf5")]
pub use checkpoint::{write_checkpoint, read_checkpoint, CheckpointData};
