//! Parallel finite element assembly.
//!
//! [`ParAssembler`] wraps the serial [`Assembler`] and leverages a one-layer
//! ghost-element overlap in the local mesh so that each rank's owned DOF rows
//! receive the full assembled contributions without any inter-rank exchange.

use fem_space::fe_space::FESpace;
use fem_assembly::assembler::Assembler;
use fem_assembly::integrator::{BilinearIntegrator, LinearIntegrator};

use crate::par_csr::ParCsrMatrix;
use crate::par_space::ParallelFESpace;
use crate::par_vector::ParVector;

/// Parallel assembly driver.
///
/// All methods are associated functions (no `self`).
///
/// The local mesh includes a one-layer ghost-element overlap, so serial
/// assembly on the local mesh produces complete owned-row contributions.
/// No ghost-row exchange is needed.
pub struct ParAssembler;

impl ParAssembler {
    /// Parallel bilinear form assembly.
    ///
    /// 1. Serial assembly on the local mesh (owned + ghost elements).
    /// 2. Split into `ParCsrMatrix` — only owned rows are retained.
    pub fn assemble_bilinear<S: FESpace>(
        par_space: &ParallelFESpace<S>,
        integrators: &[&dyn BilinearIntegrator],
        quad_order: u8,
    ) -> ParCsrMatrix {
        // Serial assembly on local mesh (includes ghost elements).
        let local_mat = Assembler::assemble_bilinear(
            par_space.local_space(), integrators, quad_order,
        );

        // ParCsrMatrix::from_local_matrix keeps only owned rows and splits
        // columns into diag (owned) and offd (ghost) blocks.
        ParCsrMatrix::from_local_matrix(
            &local_mat,
            par_space.dof_partition().n_owned_dofs,
            par_space.dof_ghost_exchange_arc(),
            par_space.comm().clone(),
        )
    }

    /// Parallel linear form assembly.
    ///
    /// 1. Serial assembly on the local mesh (owned + ghost elements).
    /// 2. Wrap in `ParVector` — only owned entries are meaningful.
    pub fn assemble_linear<S: FESpace>(
        par_space: &ParallelFESpace<S>,
        integrators: &[&dyn LinearIntegrator],
        quad_order: u8,
    ) -> ParVector {
        // Serial assembly produces the complete RHS for owned DOFs (thanks
        // to the ghost-element overlap).  No reverse exchange needed.
        let local_rhs = Assembler::assemble_linear(
            par_space.local_space(), integrators, quad_order,
        );

        ParVector::from_local(local_rhs, par_space)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::native::ThreadLauncher;
    use crate::launcher::WorkerConfig;
    use crate::par_simplex::partition_simplex;
    use crate::par_space::ParallelFESpace;
    use fem_assembly::assembler::Assembler;
    use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;

    #[test]
    fn par_assembly_rhs_integral() {
        // Parallel assembly of ∫ 1 dx over the unit square should give global
        // sum = 1.0.
        let mesh = SimplexMesh::<2>::unit_square_tri(8);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            // Sum owned entries across all ranks.
            let local_sum: f64 = rhs.owned_slice().iter().sum();
            let global_sum = comm.allreduce_sum_f64(local_sum);

            assert!(
                (global_sum - 1.0).abs() < 1e-10,
                "rank {}: global ∫1 dx = {global_sum}, expected 1.0",
                comm.rank()
            );
        });
    }

    #[test]
    fn par_assembly_stiffness_diagonal_positive() {
        // The diagonal of the Laplacian stiffness matrix should be positive.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);

            let diag = a_mat.diagonal();
            for (i, &d) in diag.iter().enumerate() {
                assert!(d > 0.0,
                    "rank {}: diagonal[{i}] = {d}, expected positive",
                    comm.rank()
                );
            }
        });
    }

    #[test]
    fn par_assembly_serial_matches() {
        // Compare parallel assembly (1 rank) with serial assembly.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let mesh2 = mesh.clone();

        // Serial assembly.
        let serial_space = H1Space::new(mesh.clone(), 1);
        let diff = DiffusionIntegrator { kappa: 1.0 };
        let serial_mat = Assembler::assemble_bilinear(&serial_space, &[&diff], 2);

        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh2, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let par_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);

            // Compare diag block with serial matrix.
            let n = par_mat.n_owned;
            for i in 0..n {
                for j in 0..n {
                    let par_val = par_mat.diag.get(i, j);
                    let ser_val = serial_mat.get(i, j);
                    assert!(
                        (par_val - ser_val).abs() < 1e-12,
                        "mismatch at ({i},{j}): par={par_val}, serial={ser_val}"
                    );
                }
            }
        });
    }
}
