#![allow(clippy::needless_range_loop)]

//! # fem-parallel
//!
//! MPI-parallel infrastructure: abstract backend, distributed mesh primitives,
//! halo exchange, and process launchers.
//!
//! ## Feature flags
//!
//! | Feature | Effect |
//! |---------|--------|
//! | `mpi`   | Links rsmpi (`mpi` 0.8): real process MPI in `backend::native`, [`MpiLauncher`](launcher::native::MpiLauncher). |
//! | *(none)*| Serial / WASM stub — same public API, no MPI install required. |
//!
//! ## Target matrix
//!
//! | `cfg` | Active backend | Launcher |
//! |-------|---------------|----------|
//! | non-wasm32 + `mpi` feature | rsmpi world communicator | [`MpiLauncher`](launcher::native::MpiLauncher) |
//! | non-wasm32, no `mpi`       | `SerialBackend`             | `ThreadLauncher` |
//! | `wasm32`                   | `WasmWorkerBackend` (stub)  | `WorkerLauncher` |
//!
//! ## MPI (multi-process) scenarios
//!
//! ### When to use which launcher
//!
//! - **[`MpiLauncher`](launcher::native::MpiLauncher)** — **real** MPI: you start *N* OS
//!   processes with `mpirun` / `mpiexec` / your scheduler; each process runs `main`,
//!   calls `MpiLauncher::init()`, and receives a [`Comm`] tied to `MPI_COMM_WORLD`.
//! - **`ThreadLauncher`** — **fake** multi-rank inside one process: good for CI, laptops
//!   without an MPI install, and debugging partition / ghost / solver logic. The
//!   `mfem_pex*` examples use this by default; the mathematics and collectives match
//!   multi-rank semantics, but there is only one address space.
//!
//! ### Build and run (native MPI)
//!
//! 1. Install an MPI implementation (MS-MPI on Windows, OpenMPI / Intel MPI on Linux, etc.)
//!    and ensure `mpicc` / libraries are on `PATH` / `LIB` so `rsmpi` can link.
//! 2. Enable the crate feature: e.g. `cargo build -p fem-parallel --features mpi` or
//!    depend on `fem-parallel = { path = "...", features = ["mpi"] }` from your binary.
//! 3. Launch *N* processes from outside Rust:  
//!    `mpiexec -n 4 target/release/your_solver [args]`  
//!    (Do not expect `ThreadLauncher` to create OS processes — only `MpiLauncher` + a
//!    process manager does that.)
//!
//! ### `Comm` and MPI `Universe` lifetime
//!
//! [`MpiLauncher`](launcher::native::MpiLauncher) owns `mpi::environment::Universe`. If
//! you drop the launcher while still using the [`Comm`] it created, `MPI_Finalize` may run
//! too early. **Keep the launcher alive** for the whole solve (e.g. store it in `main` next
//! to the [`Comm`]), or use a process-wide `OnceLock` holding `Universe` (see the
//! `mpi_test_env` module in this crate’s source for the unit-test pattern).
//!
//! ### Typical distributed FEM pipeline
//!
//! 1. **Partition** the global mesh: [`partition_mesh`](par_partition::partition_mesh),
//!    [`partition_mesh_metis`](metis::partition_mesh_metis), or **streaming** variants
//!    when only rank 0 holds the full mesh — see [`partition_mesh_streaming`](par_partition::partition_mesh_streaming)
//!    and [`mesh_serde::encode_submesh`](mesh_serde::encode_submesh).
//! 2. **Ghost layout**: [`GhostExchange::from_partition`](ghost::GhostExchange::from_partition)
//!    (uses an all-to-all style setup so each rank knows halo send/recv lists).
//! 3. **FE space**: [`ParallelFESpace`](par_space::ParallelFESpace) — local mesh includes
//!    one layer of ghost elements so **volume assembly is local**; [`ParAssembler`](par_assembler::ParAssembler)
//!    / [`ParVectorAssembler`](par_vector_assembler::ParVectorAssembler) then strip to owned rows.
//! 4. **Operator apply**: [`ParCsrMatrix`](par_csr::ParCsrMatrix) SpMV + halo refresh. With
//!    [`Comm::is_native_mpi`](comm::Comm::is_native_mpi) `== true`, use
//!    [`GhostExchange::forward_overlapping`](ghost::GhostExchange::forward_overlapping) so
//!    non-blocking receives overlap local diagonal work.
//! 5. **Collectives** on [`Comm`]: barriers, `allreduce_sum_*`, broadcast, byte vectors for
//!    custom metadata — same calls work on `ThreadLauncher`’s channel backend for tests.
//!
//! ### I/O and checkpoints
//!
//! Parallel HDF5 / MPI-IO lives in **`fem-io-hdf5-parallel`** (see that crate’s README):
//! enable `hdf5-mpi` / workspace `io_hdf5_mpi` when building examples that write collective
//! checkpoints under real MPI.
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
//! See [`mod@env`] — notably [`FEM_PARALLEL_LOCAL_RAYON_MIN`] for
//! thresholds on local Rayon parallelism before MPI collectives,
//! [`FEM_LINALG_SPMV_PARALLEL_MIN_ROWS`] for local CSR SpMV threading (`fem-linalg`, re-exported),
//! and [`FEM_ASSEMBLY_PARALLEL_MIN_ELEMS`] for local volume assembly (`fem-assembly`, re-exported).
//!
//! Halo exchange for [`ParCsrMatrix`] uses non-blocking MPI
//! when [`Comm::is_native_mpi`](comm::Comm::is_native_mpi) is true, overlapping the diagonal
//! SpMV with in-flight receives/sends (see [`GhostExchange::forward_overlapping`](crate::ghost::GhostExchange::forward_overlapping)).
//!
//! Streaming partition ([`partition_mesh_streaming`]) serialises sub-meshes with
//! [`mesh_serde::encode_submesh`](mesh_serde::encode_submesh); **wire format v2** carries
//! mixed volume elements and mixed boundary faces (e.g. `Prism6` / `Pyramid5` with Tri3+Quad4
//! boundaries) so cylinder- or cone-like GMSH meshes round-trip correctly across ranks.

pub mod backend;
pub mod comm;
pub mod dof_partition;
pub mod env;
pub mod forest;
pub mod ghost;
pub mod gpu_mpi;
pub mod launcher;
pub mod mesh_serde;
pub mod metis;
pub mod par_amg;
pub mod par_amr;
pub mod par_refine;
pub mod par_assembler;
pub mod par_csr;
pub mod par_mesh;
pub mod par_mixed_assembler;
pub mod par_ras;
pub mod par_partition;
pub mod par_solver;
pub mod par_direct;
pub mod par_dpg_trace;
pub mod par_ams;
pub mod par_discrete_operator;
pub mod par_lobpcg;
pub mod par_space;
pub mod par_vector;
pub mod par_vector_assembler;
pub mod par_complex_csr;
pub mod par_block_csr;
pub mod partition;
pub mod rcb;
pub mod sfc;
pub mod shared_entities;
pub mod par_mesh_builder;

#[cfg(test)]
mod mpi_test_env;
#[cfg(feature = "hdf5")]
pub mod par_hdf5;
pub mod pmesh;
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
pub use metis::{MetisPartitioner, MetisOptions, partition_mesh_metis, partition_mesh_metis_streaming};
pub use par_assembler::ParAssembler;
pub use par_vector_assembler::ParVectorAssembler;
pub use par_amg::{
    ParAmgConfig, ParAmgHierarchy, SmootherType, par_solve_pcg_amg,
    par_solve_pcg_block_diag,
};
pub use par_ams::ParAmsPrecond;
pub use par_csr::ParCsrMatrix;
pub use par_discrete_operator::ParDiscreteLinearOperator;
pub use par_lobpcg::{par_lobpcg, ParLobpcgResult};
pub use par_mesh::ParallelMesh;
pub use par_mixed_assembler::ParMixedAssembler;
pub use par_ras::{
	RasConfig, RasHpcDiagnostics, RasLocalSolverKind, RasPrecond, par_solve_gmres_ras,
	par_solve_pcg_ras, summarize_ras_hpc,
};
pub use par_partition::{partition_mesh, partition_mesh_replicated, partition_mesh_streaming};
pub use par_solver::{par_solve_cg, par_solve_gmres_jacobi, par_solve_pcg_jacobi, par_solve_pcg_precond, par_solve_minres};
pub use par_space::ParallelFESpace;
pub use par_vector::ParVector;
pub use par_vector::ParComplexVector;
pub use partition::MeshPartition;
pub use shared_entities::{SharedEntities, SharedEntity};

#[cfg(feature = "hdf5")]
pub use par_hdf5::{par_write_mesh_and_fields, ParHdf5Options, ParallelWriteMode};
// .pmesh format is independent of HDF5
pub use pmesh::{write_pmesh, read_pmesh};
#[cfg(feature = "hdf5")]
pub use checkpoint::{write_checkpoint, read_checkpoint, CheckpointData};
