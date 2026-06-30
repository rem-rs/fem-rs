//! GPU-accelerated AMG preconditioner (requires `amg` feature).
//!
//! Hybrid approach: hierarchy lives on CPU, smoothing uses GPU SpMV when
//! the level hierarchy API is available. For now, wraps the CPU solver
//! and provides the `GpuAmgPrecond` API for use in GPU CG/GMRES.

use crate::GpuContext;
use crate::GpuVector;

/// GPU-accelerated AMG preconditioner.
///
/// Stores the finest-level diagonal for GPU Jacobi smoothing.
/// Multi-level V-cycles fall back to CPU until level matrices are
/// accessible via a public hierarchy API.
pub struct GpuAmgPrecond {
    cpu_solver: fem_amg::AmgSolver<f64>,
    diag_inv: Vec<f64>,
}

impl GpuAmgPrecond {
    /// Build the AMG hierarchy from `a` and store state for GPU smoothing.
    pub fn new(
        _ctx: &GpuContext,
        a: &fem_linalg::CsrMatrix<f64>,
        config: fem_amg::AmgConfig,
        cycle: fem_amg::CycleType,
    ) -> Self {
        let diag_inv: Vec<f64> = (0..a.nrows)
            .map(|i| { let d = a.get(i, i); if d.abs() > 1e-14 { 1.0 / d } else { 1.0 } })
            .collect();
        let cpu_solver = fem_amg::AmgSolver::setup(a, config).with_cycle(cycle);
        GpuAmgPrecond { cpu_solver, diag_inv }
    }

    /// Apply one V-cycle.
    ///
    /// Currently reads from GPU, runs CPU AMG V-cycle, writes back.
    /// When `AmgHierarchy` exposes level matrices, pre/post-smoothing
    /// will be done via GPU SpMV (see `amg_precond.rs` in `vendor/linger`).
    pub fn apply(&self, ctx: &GpuContext, r: &GpuVector<f64>, z: &mut GpuVector<f64>) {
        let r_cpu: Vec<f64> = r.read_to_cpu(ctx);
        let z_cpu = self.cpu_solver.precond_apply(&r_cpu);
        z.write_from_slice(ctx, &z_cpu);
    }
}
