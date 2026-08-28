//! Parallel finite element assembly.
//!
//! [`ParAssembler`] wraps the serial [`Assembler`] and leverages a one-layer
//! ghost-element overlap in the local mesh so that each rank's owned DOF rows
//! receive the full assembled contributions without any inter-rank exchange.

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_space::fe_space::FESpace;
use fem_assembly::assembler::Assembler;
use fem_assembly::integrator::{
    BilinearIntegrator, BoundaryBilinearIntegrator, BoundaryLinearIntegrator, LinearIntegrator,
};
use fem_core::types::DofId;

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
    /// Parallel bilinear form assembly with per-integrator element markers
    /// (named attribute sets).  Same semantics as
    /// [`Assembler::assemble_bilinear_marked`] on the local mesh (owned +
    /// ghost elements; the partition migrates `elem_tags`), then permute and
    /// wrap in a `ParCsrMatrix`.
    ///
    /// Used by pex39 (ex39p): κ = 1e-6 everywhere + 1.0 on `Base` + 2.0 on
    /// `Rose Even`, each piece restricted to a named attribute set marker.
    pub fn assemble_bilinear_marked<S: FESpace>(
        par_space: &ParallelFESpace<S>,
        integrators: &[(&dyn BilinearIntegrator, Option<&[i32]>)],
        quad_order: u8,
    ) -> ParCsrMatrix {
        let local_mat = Assembler::assemble_bilinear_marked(
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

    /// Parallel linear form assembly with a per-integrator element marker
    /// (named attribute sets).  Same semantics as
    /// [`Assembler::assemble_linear_marked`] on the local mesh, then permute
    /// and wrap in a `ParVector`.
    pub fn assemble_linear_marked<S: FESpace>(
        par_space: &ParallelFESpace<S>,
        integrators: &[(&dyn LinearIntegrator, Option<&[i32]>)],
        quad_order: u8,
    ) -> ParVector {
        let local_rhs = Assembler::assemble_linear_marked(
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

    /// Parallel boundary linear form assembly (e.g. Neumann traction on a
    /// boundary attribute).  Serial boundary assembly on the local mesh
    /// (owned + ghost faces), then permute and wrap in a `ParVector`.
    ///
    /// A boundary face is assembled only by the rank owning its adjacent
    /// element, but the face's *vertex* DOFs can be owned by other ranks (they
    /// are ghost slots locally).  The ghost-slot contributions are accumulated
    /// back into their owners with the reverse ghost exchange before wrapping,
    /// so every owned DOF receives its full boundary load.
    pub fn assemble_boundary_linear<S: FESpace>(
        par_space: &ParallelFESpace<S>,
        n_dofs: usize,
        face_dofs: &(dyn Fn(u32) -> Vec<DofId> + Sync),
        order: u8,
        integrators: &[&dyn BoundaryLinearIntegrator],
        tags: &[i32],
        quad_order: u8,
    ) -> ParVector {
        let local_rhs = Assembler::assemble_boundary_linear(
            n_dofs,
            par_space.local_space().mesh(),
            face_dofs,
            order,
            integrators,
            tags,
            quad_order,
        );

        let dof_part = par_space.dof_partition();
        let mut permuted_rhs = if dof_part.needs_permutation() {
            permute_vec(&local_rhs, dof_part)
        } else {
            local_rhs
        };

        let ghost = par_space.dof_ghost_exchange_arc();
        ghost.reverse(par_space.comm(), &mut permuted_rhs);

        ParVector::from_local_raw(
            permuted_rhs,
            dof_part.n_owned_dofs,
            ghost,
            par_space.comm().clone(),
        )
    }

    /// Parallel boundary bilinear form assembly (e.g. the Robin mass
    /// `∫_Γ a·u·v ds` on a boundary attribute).  Serial boundary assembly on
    /// the local mesh (owned + ghost faces), then permute and wrap in a
    /// `ParCsrMatrix` — only owned rows are retained.
    ///
    /// The local mesh keeps every boundary face of the owned elements (the
    /// face is assigned to the rank owning the adjacent element), so each
    /// global boundary face contributes to exactly one rank's matrix.
    ///
    /// The face's *vertex* DOFs can be owned by other ranks; the rows of the
    /// local matrix that belong to ghost DOFs are exchanged to their owning
    /// ranks and added there, so every owned row receives its full boundary
    /// contribution (the standard `from_local_matrix` drops ghost rows).
    pub fn assemble_boundary_bilinear<S: FESpace>(
        par_space: &ParallelFESpace<S>,
        n_dofs: usize,
        face_dofs: &(dyn Fn(u32) -> Vec<DofId> + Sync),
        order: u8,
        integrators: &[&dyn BoundaryBilinearIntegrator],
        tags: &[i32],
        quad_order: u8,
    ) -> ParCsrMatrix {
        let local_mat = Assembler::assemble_boundary_bilinear(
            n_dofs,
            par_space.local_space().mesh(),
            face_dofs,
            order,
            integrators,
            tags,
            quad_order,
        );

        let dof_part = par_space.dof_partition();
        let permuted_mat = if dof_part.needs_permutation() {
            permute_csr(&local_mat, dof_part)
        } else {
            local_mat
        };

        let n_owned = dof_part.n_owned_dofs;
        let comm = par_space.comm();
        let rank = comm.rank();
        let n_ranks = comm.size() as i32;
        let n_local = dof_part.n_total_dofs();

        // 1. Owned rows → diag/offd; ghost rows → collect for exchange.
        let mut diag_coo = CooMatrix::<f64>::new(n_owned, n_owned);
        let mut offd_coo = CooMatrix::<f64>::new(n_owned, n_local.saturating_sub(n_owned));
        // (owner rank, global row id, (global col id, value) entries)
        let mut ghost_rows: Vec<(i32, u32, Vec<(u32, f64)>)> = Vec::new();
        for row in 0..n_local {
            if row < n_owned {
                for k in permuted_mat.row_ptr[row]..permuted_mat.row_ptr[row + 1] {
                    let col = permuted_mat.col_idx[k] as usize;
                    let val = permuted_mat.values[k];
                    if val == 0.0 { continue; }
                    if col < n_owned {
                        diag_coo.add(row, col, val);
                    } else {
                        offd_coo.add(row, col - n_owned, val);
                    }
                }
            } else {
                let owner = dof_part.dof_owner(row as u32);
                let global_row = dof_part.global_dof(row as u32);
                let mut entries: Vec<(u32, f64)> = Vec::new();
                for k in permuted_mat.row_ptr[row]..permuted_mat.row_ptr[row + 1] {
                    let val = permuted_mat.values[k];
                    if val != 0.0 {
                        entries.push((dof_part.global_dof(permuted_mat.col_idx[k]), val));
                    }
                }
                if !entries.is_empty() {
                    ghost_rows.push((owner, global_row, entries));
                }
            }
        }

        // 2. Alltoall the ghost rows to their owners.
        let mut sends: Vec<(i32, Vec<u8>)> = Vec::new();
        for r in 0..n_ranks {
            if r == rank { continue; }
            let mut bytes = Vec::new();
            for (owner, grow, entries) in &ghost_rows {
                if *owner != r { continue; }
                bytes.extend_from_slice(&grow.to_le_bytes());
                bytes.extend_from_slice(&(entries.len() as u32).to_le_bytes());
                for (c, v) in entries {
                    bytes.extend_from_slice(&c.to_le_bytes());
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
            }
            sends.push((r, bytes));
        }
        let incoming = comm.alltoallv_bytes(&sends);
        for (_, bytes) in incoming {
            let mut i = 0usize;
            while i + 8 <= bytes.len() {
                let grow = u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
                let ne = u32::from_le_bytes(bytes[i + 4..i + 8].try_into().unwrap()) as usize;
                i += 8;
                let mut entries = Vec::with_capacity(ne);
                for _ in 0..ne {
                    let c = u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
                    let v = f64::from_le_bytes(bytes[i + 4..i + 12].try_into().unwrap());
                    i += 12;
                    entries.push((c, v));
                }
                let Some(local_row) = dof_part.local_dof(grow) else { continue; };
                let local_row = local_row as usize;
                debug_assert!(local_row < n_owned, "ghost-row exchange must target an owned row");
                for (gc, v) in entries {
                    let Some(local_col) = dof_part.local_dof(gc) else { continue; };
                    let local_col = local_col as usize;
                    if local_col < n_owned {
                        diag_coo.add(local_row, local_col, v);
                    } else {
                        offd_coo.add(local_row, local_col - n_owned, v);
                    }
                }
            }
        }

        let diag = diag_coo.into_csr();
        let offd = offd_coo.into_csr();
        ParCsrMatrix::from_blocks(
            diag,
            offd,
            n_owned,
            n_local.saturating_sub(n_owned),
            par_space.dof_ghost_exchange_arc(),
            comm.clone(),
        )
    }
}

/// Permute a CSR matrix from DofManager ordering to partition [owned|ghost] ordering.
pub fn permute_csr(mat: &CsrMatrix<f64>, dof_part: &DofPartition) -> CsrMatrix<f64> {
    let n = dof_part.n_total_dofs();
    let mut coo = CooMatrix::<f64>::new(n, n);
    let needs_sign = dof_part.needs_sign_correction();

    for row in 0..mat.nrows {
        let new_row = dof_part.permute_dof(row as u32) as usize;
        let sr = if needs_sign {
            dof_part.sign_correction(row as u32)
        } else {
            1.0
        };
        for k in mat.row_ptr[row]..mat.row_ptr[row + 1] {
            let col = mat.col_idx[k] as usize;
            let new_col = dof_part.permute_dof(col as u32) as usize;
            let sc = if needs_sign {
                dof_part.sign_correction(col as u32)
            } else {
                1.0
            };
            let val = mat.values[k] * sr * sc;
            if val != 0.0 {
                coo.add(new_row, new_col, val);
            }
        }
    }

    coo.into_csr()
}

/// Permute a vector from DofManager ordering to partition [owned|ghost] ordering.
pub fn permute_vec(vec: &[f64], dof_part: &DofPartition) -> Vec<f64> {
    let n = dof_part.n_total_dofs();
    let mut out = vec![0.0; n];
    let needs_sign = dof_part.needs_sign_correction();
    for (i, &v) in vec.iter().enumerate() {
        let new_i = dof_part.permute_dof(i as u32) as usize;
        let si = if needs_sign {
            dof_part.sign_correction(i as u32)
        } else {
            1.0
        };
        out[new_i] = v * si;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::native::ThreadLauncher;
    use crate::launcher::WorkerConfig;
    use crate::par_partition::partition_mesh;
    use crate::par_space::ParallelFESpace;
    use fem_assembly::assembler::Assembler;
    use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
    use fem_mesh::Mesh;
    use fem_space::H1Space;
    use fem_space::dof_manager::DofManager;

    #[test]
    fn par_assembly_rhs_integral_p1() {
        let mesh = Mesh::<2>::unit_square_tri(8);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
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
        let mesh = Mesh::<2>::unit_square_tri(8);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
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
        let mesh = Mesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
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
        let mesh = Mesh::<2>::unit_square_tri(4);
        let mesh2 = mesh.clone();

        let serial_space = H1Space::new(mesh.clone(), 1);
        let diff = DiffusionIntegrator { kappa: 1.0 };
        let serial_mat = Assembler::assemble_bilinear(&serial_space, &[&diff], 2);

        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh2, &comm);
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
        let mesh = Mesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
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
