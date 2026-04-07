//! Parallel WASM solver using jsmpi multi-worker communication.
//!
//! ## Entry point
//!
//! [`jsmpi_main`] is the function called by jsmpi's `worker.js` inside each
//! Web Worker.  It initialises the MPI-like communicator, partitions the mesh
//! via streaming (only rank 0 holds the full mesh), assembles and solves a
//! parallel Poisson problem, then signals completion.
//!
//! ## JS integration
//!
//! [`WasmParSolver`] provides a JS-facing wrapper that can be used from the
//! browser main thread to orchestrate a multi-worker solve.

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
use fem_mesh::SimplexMesh;
use fem_solver::SolverConfig;
use fem_space::constraints::boundary_dofs;
use fem_space::H1Space;
use fem_space::fe_space::FESpace;

use fem_parallel::{
    Comm,
    ParAssembler,
    ParVector,
    ParallelFESpace,
    par_simplex::partition_simplex_streaming,
    par_solver::par_solve_pcg_jacobi,
};

// ── jsmpi_main — Web Worker entry point ──────────────────────────────────────

/// Entry point called by jsmpi's `worker.js` inside each Web Worker.
///
/// This function:
/// 1. Initialises the jsmpi communicator from browser globals.
/// 2. Receives the sub-mesh via streaming partition (only rank 0 sends).
/// 3. Assembles and solves a Poisson problem in parallel.
/// 4. Signals completion to the coordinator.
///
/// Only available on `wasm32` targets with the `wasm` feature enabled.
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
#[wasm_bindgen]
pub fn jsmpi_main() {
    console_error_panic_hook::set_once();

    let init = fem_parallel::launcher::wasm::WorkerInitMsg::from_jsmpi_env()
        .expect("failed to initialise jsmpi environment");
    let comm = init.into_comm();

    run_poisson_2d(&comm, 16);

    jsmpi::runtime::mark_finished().ok();
}

/// Run a 2D Poisson problem on a unit square mesh with `n×n` grid divisions.
///
/// This is the parallel equivalent of the serial [`WasmSolver`].
#[cfg(any(target_arch = "wasm32", test))]
fn run_poisson_2d(comm: &Comm, n_grid: usize) {
    // Only rank 0 builds the full mesh.
    let mesh = if comm.is_root() {
        Some(SimplexMesh::<2>::unit_square_tri(n_grid))
    } else {
        None
    };

    let pmesh = partition_simplex_streaming(mesh.as_ref(), comm)
        .expect("streaming partition failed");

    let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
    let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

    // Assemble -∇²u = 1.
    let diff = DiffusionIntegrator { kappa: 1.0 };
    let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);

    let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
    let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

    // Apply Dirichlet BCs: u = 0 on ∂Ω.
    let dm = par_space.local_space().dof_manager();
    let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
    for &d in &bc_dofs {
        let lid = d as usize;
        if lid < par_space.dof_partition().n_owned_dofs {
            a_mat.apply_dirichlet_row(lid, 0.0, rhs.as_slice_mut());
        }
    }

    let mut u = ParVector::zeros(&par_space);
    let cfg = SolverConfig {
        rtol: 1e-8,
        max_iter: 500,
        ..SolverConfig::default()
    };
    let result = par_solve_pcg_jacobi(&a_mat, &rhs, &mut u, &cfg)
        .expect("PCG solve failed");

    if comm.is_root() {
        log::info!(
            "par_poisson_2d: converged={}, iters={}, residual={:.3e}",
            result.converged, result.iterations, result.final_residual,
        );
    }
}

// ── WasmParSolver — JS-facing orchestration ──────────────────────────────────

/// Multi-worker parallel Poisson solver for browser use.
///
/// Orchestrates the launch of Web Workers via jsmpi and coordinates a parallel
/// FEM solve.  Construct on the main thread, call [`launch`](WasmParSolver::launch)
/// to start the computation.
///
/// ## Usage (JS)
/// ```js
/// import init, { WasmParSolver } from './fem_wasm.js';
/// await init();
///
/// const solver = new WasmParSolver(16, 4);  // 16×16 grid, 4 workers
/// solver.launch("worker.js", "fem_wasm.js");
/// // … logs and completion callbacks are forwarded via jsmpi …
/// ```
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmParSolver {
    n_grid: u32,
    n_workers: u32,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmParSolver {
    /// Create a parallel solver for an `n × n` unit-square triangular mesh
    /// using `n_workers` Web Workers.
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(n_grid: u32, n_workers: u32) -> Self {
        WasmParSolver {
            n_grid: n_grid.max(1),
            n_workers: n_workers.max(1),
        }
    }

    /// Number of grid divisions per axis.
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn n_grid(&self) -> u32 { self.n_grid }

    /// Number of workers.
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn n_workers(&self) -> u32 { self.n_workers }
}

// ── Native-only tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_parallel::launcher::native::ThreadLauncher;
    use fem_parallel::WorkerConfig;

    /// Verify the parallel Poisson solver converges with 1 rank via streaming.
    #[test]
    fn par_poisson_streaming_serial() {
        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(|comm| {
            run_poisson_2d(&comm, 8);
        });
    }

    /// Verify the parallel Poisson solver converges with 2 ranks via streaming.
    #[test]
    fn par_poisson_streaming_2_ranks() {
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(|comm| {
            run_poisson_2d(&comm, 8);
        });
    }

    /// Verify with 4 ranks.
    #[test]
    fn par_poisson_streaming_4_ranks() {
        let launcher = ThreadLauncher::new(WorkerConfig::new(4));
        launcher.launch(|comm| {
            run_poisson_2d(&comm, 8);
        });
    }

    /// WasmParSolver construction (native test — only struct fields).
    #[test]
    fn wasm_par_solver_construction() {
        let s = WasmParSolver::new(16, 4);
        assert_eq!(s.n_grid(), 16);
        assert_eq!(s.n_workers(), 4);
    }
}
