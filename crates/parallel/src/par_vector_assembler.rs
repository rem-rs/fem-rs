//! Parallel assembly for vector finite element spaces (H(curl), H(div)).
//!
//! [`ParVectorAssembler`] wraps the serial [`VectorAssembler`] and leverages a
//! one-layer ghost-element overlap in the local mesh so that each rank's owned
//! DOF rows receive the full assembled contributions without any inter-rank exchange.

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_space::fe_space::FESpace;
use fem_assembly::vector_assembler::VectorAssembler;
use fem_assembly::vector_integrator::{VectorBilinearIntegrator, VectorLinearIntegrator};

use crate::par_csr::ParCsrMatrix;
use crate::par_space::ParallelFESpace;
use crate::par_vector::ParVector;
use crate::dof_partition::DofPartition;

/// Parallel assembly driver for vector FE spaces (H(curl), H(div)).
///
/// Follows the same pattern as [`ParAssembler`](crate::par_assembler::ParAssembler):
/// serial assembly on local mesh (with ghost overlap), optional DOF permutation,
/// then split into `ParCsrMatrix`.
pub struct ParVectorAssembler;

impl ParVectorAssembler {
    /// Parallel bilinear form assembly for vector spaces.
    ///
    /// 1. Serial vector assembly on the local mesh (owned + ghost elements).
    /// 2. Permute to [owned|ghost] DOF ordering if needed.
    /// 3. Split into `ParCsrMatrix` — only owned rows are retained.
    pub fn assemble_bilinear<S: FESpace>(
        par_space: &ParallelFESpace<S>,
        integrators: &[&dyn VectorBilinearIntegrator],
        quad_order: u8,
    ) -> ParCsrMatrix {
        let local_mat = VectorAssembler::assemble_bilinear(
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

    /// Parallel linear form assembly for vector spaces.
    ///
    /// 1. Serial vector assembly on the local mesh (owned + ghost elements).
    /// 2. Permute to [owned|ghost] DOF ordering if needed.
    /// 3. Wrap in `ParVector` — only owned entries are meaningful.
    pub fn assemble_linear<S: FESpace>(
        par_space: &ParallelFESpace<S>,
        integrators: &[&dyn VectorLinearIntegrator],
        quad_order: u8,
    ) -> ParVector {
        let local_rhs = VectorAssembler::assemble_linear(
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

/// Permute a CSR matrix from local space ordering to partition [owned|ghost] ordering.
///
/// For H(curl)/H(div) spaces, also applies the sign correction `d_i * d_j`
/// stored in [`DofPartition::sign_corrections`] so that the matrix is
/// expressed in the globally consistent edge-orientation basis.
fn permute_csr(mat: &CsrMatrix<f64>, dof_part: &DofPartition) -> CsrMatrix<f64> {
    let n = dof_part.n_total_dofs();
    let mut coo = CooMatrix::<f64>::new(n, n);

    for row in 0..mat.nrows {
        let new_row = dof_part.permute_dof(row as u32) as usize;
        let d_row = dof_part.sign_correction(row as u32);
        for k in mat.row_ptr[row]..mat.row_ptr[row + 1] {
            let col = mat.col_idx[k] as usize;
            let new_col = dof_part.permute_dof(col as u32) as usize;
            let d_col = dof_part.sign_correction(col as u32);
            let val = mat.values[k] * d_row * d_col;
            if val != 0.0 {
                coo.add(new_row, new_col, val);
            }
        }
    }

    coo.into_csr()
}

/// Permute a vector from local space ordering to partition [owned|ghost] ordering.
///
/// For H(curl)/H(div) spaces, also applies the sign correction `d_i`
/// so that the vector is in the globally consistent edge-orientation basis.
fn permute_vec(vec: &[f64], dof_part: &DofPartition) -> Vec<f64> {
    let n = dof_part.n_total_dofs();
    let mut out = vec![0.0; n];
    for (i, &v) in vec.iter().enumerate() {
        let new_i = dof_part.permute_dof(i as u32) as usize;
        out[new_i] = v * dof_part.sign_correction(i as u32);
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
    use fem_assembly::standard::{CurlCurlIntegrator, VectorMassIntegrator};
    use fem_mesh::SimplexMesh;
    use fem_space::HCurlSpace;

    #[test]
    fn par_vector_assembly_hcurl_diagonal_positive() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = HCurlSpace::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new_for_edge_space(
                local_space, &pmesh, comm.clone(),
            );

            let curl_curl = CurlCurlIntegrator { mu: 1.0 };
            let vec_mass = VectorMassIntegrator { alpha: 1.0 };
            let a_mat = ParVectorAssembler::assemble_bilinear(
                &par_space, &[&curl_curl, &vec_mass], 4,
            );

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
    fn par_vector_assembly_hcurl_global_dof_count() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let serial_space = HCurlSpace::new(mesh.clone(), 1);
        let serial_n = serial_space.n_dofs();

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = HCurlSpace::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new_for_edge_space(
                local_space, &pmesh, comm.clone(),
            );
            assert_eq!(par_space.n_global_dofs(), serial_n,
                "global DOFs should match serial");
        });
    }

    #[test]
    fn par_vector_assembly_hcurl_serial_matches() {
        // Single-rank parallel should produce matrix with same diagonal set.
        use fem_assembly::VectorAssembler;

        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let serial_space = HCurlSpace::new(mesh.clone(), 1);
        let curl_curl = CurlCurlIntegrator { mu: 1.0 };
        let vec_mass = VectorMassIntegrator { alpha: 1.0 };
        let serial_mat = VectorAssembler::assemble_bilinear(
            &serial_space, &[&curl_curl, &vec_mass], 4,
        );

        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = HCurlSpace::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new_for_edge_space(
                local_space, &pmesh, comm.clone(),
            );

            let curl_curl = CurlCurlIntegrator { mu: 1.0 };
            let vec_mass = VectorMassIntegrator { alpha: 1.0 };
            let par_mat = ParVectorAssembler::assemble_bilinear(
                &par_space, &[&curl_curl, &vec_mass], 4,
            );

            let n = par_mat.n_owned;
            assert_eq!(n, serial_mat.nrows, "row count mismatch");

            // Diagonals should be the same set of values (permuted order).
            let mut par_diag: Vec<f64> = par_mat.diagonal();
            let mut ser_diag: Vec<f64> = serial_mat.diagonal();
            par_diag.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ser_diag.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (i, (p, s)) in par_diag.iter().zip(ser_diag.iter()).enumerate() {
                assert!((p - s).abs() < 1e-10,
                    "diagonal mismatch at sorted pos {i}: par={p:.6e}, serial={s:.6e}");
            }

            // Also verify the Frobenius norm is the same.
            let par_frob: f64 = par_mat.diag.values.iter().map(|v| v*v).sum::<f64>().sqrt();
            let ser_frob: f64 = serial_mat.values.iter().map(|v| v*v).sum::<f64>().sqrt();
            assert!((par_frob - ser_frob).abs() < 1e-8,
                "Frobenius norm mismatch: par={par_frob:.6e}, serial={ser_frob:.6e}");
        });
    }

    #[test]
    fn par_vector_assembly_hcurl_ghost_exchange() {
        // Verify ghost exchange works correctly for edge DOF spaces.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let serial_space = HCurlSpace::new(mesh.clone(), 1);
        let serial_n = serial_space.n_dofs();

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = HCurlSpace::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new_for_edge_space(
                local_space, &pmesh, comm.clone(),
            );

            let dp = par_space.dof_partition();
            let n_owned = dp.n_owned_dofs;
            let n_local = dp.n_total_dofs();

            // Set owned DOFs to their global DOF ID.
            let mut data = vec![-1.0_f64; n_local];
            for lid in 0..n_owned {
                let gid = dp.global_dof(lid as u32);
                data[lid] = gid as f64;
            }

            // Forward exchange: fill ghost slots.
            par_space.forward_dof_exchange(&mut data);

            // After exchange, ghost DOFs should equal their global DOF ID.
            for lid in n_owned..n_local {
                let expected = dp.global_dof(lid as u32) as f64;
                assert!(
                    (data[lid] - expected).abs() < 1e-14,
                    "rank {}: ghost DOF local={lid} expected {expected}, got {}",
                    comm.rank(), data[lid]
                );
            }
        });
    }

    #[test]
    fn par_vector_assembly_hcurl_solve_converges() {
        use crate::par_solver::par_solve_pcg_jacobi;
        use fem_solver::SolverConfig;
        use fem_space::constraints::boundary_dofs_hcurl;

        let mesh = SimplexMesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = HCurlSpace::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new_for_edge_space(
                local_space, &pmesh, comm.clone(),
            );
            let dp = par_space.dof_partition();

            let curl_curl = CurlCurlIntegrator { mu: 1.0 };
            let vec_mass = VectorMassIntegrator { alpha: 1.0 };
            let mut a_mat = ParVectorAssembler::assemble_bilinear(
                &par_space, &[&curl_curl, &vec_mass], 4,
            );

            use fem_assembly::vector_integrator::{VectorLinearIntegrator, VectorQpData};
            struct MaxwellSource;
            impl VectorLinearIntegrator for MaxwellSource {
                fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f: &mut [f64]) {
                    let x = qp.x_phys;
                    let coeff = 1.0 + std::f64::consts::PI * std::f64::consts::PI;
                    let fx = coeff * (std::f64::consts::PI * x[1]).sin();
                    let fy = coeff * (std::f64::consts::PI * x[0]).sin();
                    for i in 0..qp.n_dofs {
                        let dot = qp.phi_vec[i * 2] * fx + qp.phi_vec[i * 2 + 1] * fy;
                        f[i] += qp.weight * dot;
                    }
                }
            }
            let mut rhs = ParVectorAssembler::assemble_linear(&par_space, &[&MaxwellSource], 4);

            // Apply n×E = 0 on boundary.
            let bnd = boundary_dofs_hcurl(
                par_space.local_space().mesh(), par_space.local_space(), &[1, 2, 3, 4],
            );
            for &d in &bnd {
                let pid = dp.permute_dof(d) as usize;
                if pid < dp.n_owned_dofs {
                    a_mat.apply_dirichlet_par(pid, 0.0, &mut rhs);
                }
            }

            let mut u = ParVector::zeros(&par_space);
            let cfg = SolverConfig { rtol: 1e-8, max_iter: 5000, verbose: false, ..SolverConfig::default() };
            let res = par_solve_pcg_jacobi(&a_mat, &rhs, &mut u, &cfg).unwrap();

            assert!(res.converged,
                "rank {}: PCG did not converge: {} iters, res={:.3e}",
                comm.rank(), res.iterations, res.final_residual
            );
        });
    }
}
