//! GPU AMG preconditioner (requires `amg` feature).

use crate::{GpuContext, GpuVector};

/// GPU-resident AMG preconditioner wrapping a CPU hierarchy.
pub struct GpuAmgPrecond {
    solver: fem_amg::AmgSolver<f64>,
}

impl GpuAmgPrecond {
    pub fn new(a: &fem_linalg::CsrMatrix<f64>, config: fem_amg::AmgConfig, cycle: fem_amg::CycleType) -> Self {
        Self { solver: fem_amg::AmgSolver::setup(a, config).with_cycle(cycle) }
    }

    pub fn apply(&self, ctx: &GpuContext, r: &GpuVector<f64>, z: &GpuVector<f64>) {
        let cpu_z = self.solver.precond_apply(&r.read_to_cpu(ctx));
        z.write_from_slice(ctx, &cpu_z);
    }
}
