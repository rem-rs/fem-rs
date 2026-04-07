//! Parallel finite element assembly.
//!
//! [`ParAssembler`] wraps the serial [`Assembler`] and leverages a one-layer
//! ghost-element overlap in the local mesh so that each rank's owned DOF rows
//! receive the full assembled contributions without any inter-rank exchange.

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_space::fe_space::FESpace;
use fem_assembly::assembler::Assembler;
use fem_assembly::integrator::{BilinearIntegrator, LinearIntegrator};

use crate::par_csr::ParCsrMatrix;
use crate::par_space::ParallelFESpace;
use crate::par_vector::ParVector;
use crate::dof_partition::DofPartition;

/// Parallel assembly driver.
///
/// The local mesh includes a one-layer ghost-element overlap, so serial
/// assembly on the local mesh produces complete owned-row contributions.
/// No ghost-row exchange is needed.
pub struct ParAssembler;

impl ParAssembler {
    /// Parallel bilinear form assembly.
    ///
    /// 1. Serial assembly on the local mesh (owned + ghost elements).
    /// 2. Permute to [owned|ghost] DOF ordering if needed (P2+).
    /// 3. Split into `ParCsrMatrix` — only owned rows are retained.
    pub fn assemble_bilinear<S: FESpace>(
        par_space: &ParallelFESpace<S>,
        integrators: &[&dyn BilinearIntegrator],
        quad_order: u8,
    ) -> ParCsrMatrix {
        let local_mat = Assembler::assemble_bilinear(
            par_space.local_space(), integrators, quad_order,
        );

        let dof_part = par_space.dof_partition();
        let permuted_mat = if dof_part.needs_permutation() {
            permute_csr(&local_mat, dof_part)
        } else {
            local_mat
        };

        ParCsrMatrix::from_local_matrix(
            &permuted_mat,
            dof_part.n_owned_dofs,
            par_space.dof_ghost_exchange_arc(),
            par_space.comm().clone(),
        )
    }

    /// Parallel linear form assembly.
    ///
    /// 1. Serial assembly on the local mesh (owned + ghost elements).
    /// 2. Permute to [owned|ghost] DOF ordering if needed (P2+).
    /// 3. Wrap in `ParVector` — only owned entries are meaningful.
    pub fn assemble_linear<S: FESpace>(
        par_space: &ParallelFESpace<S>,
        integrators: &[&dyn LinearIntegrator],
        quad_order: u8,
    ) -> ParVector {
        let local_rhs = Assembler::assemble_linear(
            par_space.local_space(), integrators, quad_order,
        );

        let dof_part = par_space.dof_partition();
        let permuted_rhs = if dof_part.needs_permutation() {
            permute_vec(&local_rhs, dof_part)
        } else {
            local_rhs
        };

        ParVector::from_local_raw(
            permuted_rhs,
            dof_part.n_owned_dofs,
            par_space.dof_ghost_exchange_arc(),
            par_space.comm().clone(),
        )
    }
}

/// Permute a CSR matrix from DofManager ordering to partition [owned|ghost] ordering.
fn permute_csr(mat: &CsrMatrix<f64>, dof_part: &DofPartition) -> CsrMatrix<f64> {
    let n = dof_part.n_total_dofs();
    let mut coo = CooMatrix::<f64>::new(n, n);

    for row in 0..mat.nrows {
        let new_row = dof_part.permute_dof(row as u32) as usize;
        for k in mat.row_ptr[row]..mat.row_ptr[row + 1] {
            let col = mat.col_idx[k] as usize;
            let new_col = dof_part.permute_dof(col as u32) as usize;
            let val = mat.values[k];
            if val != 0.0 {
                coo.add(new_row, new_col, val);
            }
        }
    }

    coo.into_csr()
}

/// Permute a vector from DofManager ordering to partition [owned|ghost] ordering.
fn permute_vec(vec: &[f64], dof_part: &DofPartition) -> Vec<f64> {
    let n = dof_part.n_total_dofs();
    let mut out = vec![0.0; n];
    for (i, &v) in vec.iter().enumerate() {
        let new_i = dof_part.permute_dof(i as u32) as usize;
        out[new_i] = v;
    }
    out
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
    use fem_space::dof_manager::DofManager;

    #[test]
    fn par_assembly_rhs_integral_p1() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

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
    fn par_assembly_rhs_integral_p2() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_mesh = pmesh.local_mesh().clone();
            let dm = DofManager::new(&local_mesh, 2);
            let local_space = H1Space::new(local_mesh, 2);
            let par_space = ParallelFESpace::new_with_dof_manager(
                local_space, &pmesh, &dm, comm.clone(),
            );

            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let rhs = ParAssembler::assemble_linear(&par_space, &[&source], 4);

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
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let mesh2 = mesh.clone();

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

    #[test]
    fn par_assembly_stiffness_diagonal_positive_p2() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_mesh = pmesh.local_mesh().clone();
            let dm = DofManager::new(&local_mesh, 2);
            let local_space = H1Space::new(local_mesh, 2);
            let par_space = ParallelFESpace::new_with_dof_manager(
                local_space, &pmesh, &dm, comm.clone(),
            );

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 4);

            let diag = a_mat.diagonal();
            for (i, &d) in diag.iter().enumerate() {
                assert!(d > 0.0,
                    "rank {}: diagonal[{i}] = {d}, expected positive",
                    comm.rank()
                );
            }
        });
    }
}
