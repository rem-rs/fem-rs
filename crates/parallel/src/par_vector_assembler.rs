//! Parallel assembly for vector finite element spaces (H(curl), H(div)).
//!
//! [`ParVectorAssembler`] wraps the serial [`VectorAssembler`] and leverages a
//! one-layer ghost-element overlap in the local mesh so that each rank's owned
//! DOF rows receive the full assembled contributions without any inter-rank exchange.

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;
use fem_assembly::vector_assembler::VectorAssembler;
use fem_assembly::vector_integrator::{VectorBilinearIntegrator, VectorLinearIntegrator};
use fem_assembly::boundary::vector_boundary::{
    VectorBoundaryAssembler, VectorBoundaryBilinearIntegrator,
};

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

    /// Parallel **boundary** bilinear assembly for vector spaces
    /// (H(curl) / H(div)): `∫_Γ γ (n×u)·(n×v) dS` and friends.
    ///
    /// Mirrors MFEM's `ParBilinearForm(HCurlFESpace_)->AddBoundaryIntegrator(
    /// integ, marker)` + `ParallelAssemble()` — maxwell's
    /// `hCurlLosses_`/`M1Losses_` (the `-abcs` / conductive-loss path).
    ///
    /// Unlike the volume path ([`Self::assemble_bilinear`]), a boundary face's
    /// DOFs can be owned by another rank, so the ghost rows of the locally
    /// assembled boundary matrix are exchanged to their owners and added there
    /// ([`crate::par_assembler::finalize_boundary_matrix`], shared with the
    /// scalar `ParAssembler::assemble_boundary_bilinear`) — every owned row of
    /// the result carries its complete boundary contribution.
    ///
    /// Returns a `ParCsrMatrix` over the **same** DOF partition as
    /// [`Self::assemble_bilinear`], so the two can be summed with
    /// [`Self::add_boundary_bilinear`] / [`Self::add_bilinear`].
    pub fn assemble_boundary_bilinear<S>(
        par_space: &ParallelFESpace<S>,
        integrators: &[&dyn VectorBoundaryBilinearIntegrator],
        tags: &[i32],
        quad_order: u8,
    ) -> ParCsrMatrix
    where
        S: FESpace + Sync,
        S::Mesh: MeshTopology + Sync,
    {
        Self::boundary_bilinear_impl(par_space, integrators, tags, quad_order, 1.0)
    }

    /// `a += factor · (vector boundary bilinear form)` on an already-assembled
    /// `ParCsrMatrix`.
    ///
    /// MFEM `ParBilinearForm::AddBoundaryIntegrator` + a later
    /// `FormSystemMatrix(…, A1[dt])` accumulates the boundary operator into an
    /// existing matrix; this is the same operation for a time-stepped family
    /// such as maxwell's `A1[dt] = M1(ε) + 0.5·dt·L` with
    /// `L = M1(σ) + M1(η⁻¹)|_ABC`:
    ///
    /// ```ignore
    /// let mut a1 = ParVectorAssembler::assemble_bilinear(&nd, &[&mass_eps], qo);
    /// ParVectorAssembler::add_bilinear(&mut a1, &nd, &[&mass_sigma], qo, 0.5 * dt);
    /// ParVectorAssembler::add_boundary_bilinear(
    ///     &mut a1, &nd, &[&TangentialMassIntegrator { gamma: eta_inv }], &abc_tags, qo, 0.5 * dt);
    /// ```
    pub fn add_boundary_bilinear<S>(
        a: &mut ParCsrMatrix,
        par_space: &ParallelFESpace<S>,
        integrators: &[&dyn VectorBoundaryBilinearIntegrator],
        tags: &[i32],
        quad_order: u8,
        factor: f64,
    ) where
        S: FESpace + Sync,
        S::Mesh: MeshTopology + Sync,
    {
        let delta = Self::boundary_bilinear_impl(par_space, integrators, tags, quad_order, factor);
        add_blocks_into(a, &delta);
    }

    /// `a += factor · (vector volume bilinear form)` on an already-assembled
    /// `ParCsrMatrix` (the domain half of the same `A1[dt]` family; see
    /// [`Self::add_boundary_bilinear`]).
    pub fn add_bilinear<S: FESpace>(
        a: &mut ParCsrMatrix,
        par_space: &ParallelFESpace<S>,
        integrators: &[&dyn VectorBilinearIntegrator],
        quad_order: u8,
        factor: f64,
    ) {
        let local_mat = VectorAssembler::assemble_bilinear(
            par_space.local_space(), integrators, quad_order,
        );
        let dof_part = par_space.dof_partition();
        let permuted = permute_csr_scaled(&local_mat, dof_part, factor);
        let delta = ParCsrMatrix::from_local_matrix(
            &permuted,
            dof_part.n_owned_dofs,
            par_space.dof_ghost_exchange_arc(),
            par_space.comm().clone(),
        );
        add_blocks_into(a, &delta);
    }

    /// Shared body of [`Self::assemble_boundary_bilinear`] /
    /// [`Self::add_boundary_bilinear`].
    fn boundary_bilinear_impl<S>(
        par_space: &ParallelFESpace<S>,
        integrators: &[&dyn VectorBoundaryBilinearIntegrator],
        tags: &[i32],
        quad_order: u8,
        factor: f64,
    ) -> ParCsrMatrix
    where
        S: FESpace + Sync,
        S::Mesh: MeshTopology + Sync,
    {
        let local_mat = VectorBoundaryAssembler::assemble_boundary_bilinear(
            par_space.local_space(),
            integrators,
            tags,
            quad_order,
        );
        let dof_part = par_space.dof_partition();
        let permuted = permute_csr_scaled(&local_mat, dof_part, factor);
        crate::par_assembler::finalize_boundary_matrix(par_space, &permuted)
    }
}

/// `a += b` on the `diag`/`offd` blocks.
///
/// `CsrMatrix::axpby` builds the **union** of the two sparsity patterns, so the
/// result is the sum of the two operators regardless of which entries each
/// assembly produced (a boundary matrix only touches boundary DOFs).
fn add_blocks_into(a: &mut ParCsrMatrix, b: &ParCsrMatrix) {
    let d = a.diag_block_mut();
    *d = d.axpby(1.0, b.diag_block(), 1.0);
    let o = a.offd_block_mut();
    *o = o.axpby(1.0, b.offd_block(), 1.0);
}

/// Permute a CSR matrix from local space ordering to partition [owned|ghost] ordering.
///
/// For H(curl)/H(div) spaces, also applies the sign correction `d_i * d_j`
/// stored in [`DofPartition::sign_corrections`] so that the matrix is
/// expressed in the globally consistent edge-orientation basis.
fn permute_csr(mat: &CsrMatrix<f64>, dof_part: &DofPartition) -> CsrMatrix<f64> {
    permute_csr_scaled(mat, dof_part, 1.0)
}

/// [`permute_csr`] with an overall factor applied to every entry — used by the
/// `add_bilinear` / `add_boundary_bilinear` accumulation paths so the factor
/// never has to be applied to a `ParCsrMatrix` (which has no `scale`).
fn permute_csr_scaled(
    mat: &CsrMatrix<f64>,
    dof_part: &DofPartition,
    factor: f64,
) -> CsrMatrix<f64> {
    let n = dof_part.n_total_dofs();
    let mut coo = CooMatrix::<f64>::new(n, n);

    for row in 0..mat.nrows {
        let new_row = dof_part.permute_dof(row as u32) as usize;
        let d_row = dof_part.sign_correction(row as u32);
        for k in mat.row_ptr[row]..mat.row_ptr[row + 1] {
            let col = mat.col_idx[k] as usize;
            let new_col = dof_part.permute_dof(col as u32) as usize;
            let d_col = dof_part.sign_correction(col as u32);
            let val = mat.values[k] * d_row * d_col * factor;
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
    use crate::par_partition::partition_mesh;
    use crate::par_space::ParallelFESpace;
    use fem_assembly::standard::{CurlCurlIntegrator, VectorMassIntegrator};
    use fem_assembly::{TangentialMassIntegrator, VectorAssembler, VectorBoundaryAssembler};
    use fem_mesh::Mesh;
    use fem_space::HCurlSpace;

    /// B2 / D88: the vector **boundary** assembly + append path must reproduce
    /// the serial family `A1 = M1(ε) + 0.5·dt·(M1(σ) + M1(η⁻¹)|_ABC)` — the
    /// matrix family maxwell's `-abcs` / conductive-loss branch needs.
    ///
    /// The parallel matrix is compared against the serial one **through the DOF
    /// partition** (`m_par[permute(d)][permute(c)] == m_ser[d][c]·s(d)·s(c)`),
    /// which is exactly the basis change `permute_csr_scaled` performs; at one
    /// rank the local mesh is the whole mesh, so this is a value-level identity
    /// for every entry of the accumulated family.
    #[test]
    fn vector_boundary_append_matches_serial_family_one_rank() {
        const DT: f64 = 0.25;
        const EPS: f64 = 1.0;
        const SIGMA: f64 = 0.1;
        const ETA_INV: f64 = 0.5;
        const ABC_TAG: i32 = 2; // right edge (unit_square_* convention)

        let mesh = Mesh::<2>::unit_square_tri(4);
        let mesh_ser = mesh.clone();

        // Serial reference in DofManager order.
        let ser_space = HCurlSpace::new(mesh_ser, 1);
        let n = ser_space.n_dofs();
        let ser = VectorAssembler::assemble_bilinear(
            &ser_space,
            &[&VectorMassIntegrator { alpha: EPS }],
            3,
        )
        .axpby(
            1.0,
            &VectorAssembler::assemble_bilinear(
                &ser_space,
                &[&VectorMassIntegrator { alpha: SIGMA }],
                3,
            ),
            0.5 * DT,
        )
        .axpby(
            1.0,
            &VectorBoundaryAssembler::assemble_boundary_bilinear(
                &ser_space,
                &[&TangentialMassIntegrator { gamma: ETA_INV }],
                &[ABC_TAG],
                3,
            ),
            0.5 * DT,
        );

        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let local_space = HCurlSpace::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new_for_edge_space(
                local_space, &pmesh, comm.clone(),
            );

            // A1[dt] = M1(ε); + 0.5·dt·M1(σ); + 0.5·dt·M1(η⁻¹)|_Γ
            let mut a1 = ParVectorAssembler::assemble_bilinear(
                &par_space,
                &[&VectorMassIntegrator { alpha: EPS }],
                3,
            );
            ParVectorAssembler::add_bilinear(
                &mut a1,
                &par_space,
                &[&VectorMassIntegrator { alpha: SIGMA }],
                3,
                0.5 * DT,
            );
            ParVectorAssembler::add_boundary_bilinear(
                &mut a1,
                &par_space,
                &[&TangentialMassIntegrator { gamma: ETA_INV }],
                &[ABC_TAG],
                3,
                0.5 * DT,
            );

            // The boundary piece must be non-empty, otherwise the test would
            // pass on the volume path alone.
            let bnd = ParVectorAssembler::assemble_boundary_bilinear(
                &par_space,
                &[&TangentialMassIntegrator { gamma: ETA_INV }],
                &[ABC_TAG],
                3,
            );
            assert!(bnd.diag_block().nnz() > 0, "boundary matrix is empty");

            let local = a1.to_local_matrix();
            let part = par_space.dof_partition();
            let n_local = par_space.local_space().n_dofs();
            assert_eq!(n_local, n, "1 rank: local space is the whole space");

            let mut max_dev = 0.0_f64;
            for d in 0..n_local as u32 {
                let p = part.permute_dof(d) as usize;
                let sd = part.sign_correction(d);
                for c in 0..n_local as u32 {
                    let q = part.permute_dof(c) as usize;
                    let want = ser.get(d as usize, c as usize) * sd * part.sign_correction(c);
                    let got = local.get(p, q);
                    let dev = (want - got).abs();
                    if dev.is_nan() || !got.is_finite() {
                        max_dev = f64::INFINITY;
                    } else {
                        max_dev = max_dev.max(dev);
                    }
                }
            }
            assert!(
                max_dev < 1e-12,
                "rank {}: A1[dt] family (serial vs parallel add path) max dev {max_dev:.3e}",
                comm.rank()
            );
        });
    }

    /// The append semantics themselves, at 2 and 4 ranks: `a += factor·B` must
    /// leave `a`'s own entries untouched and add exactly `factor·B` — where `B`
    /// comes from the independent (fresh) boundary assembly.  This exercises the
    /// ghost-row exchange into owners at every rank count, which is the part of
    /// the boundary path that only exists in parallel.
    #[test]
    fn vector_boundary_append_is_exact_at_multiple_ranks() {
        const DT: f64 = 0.25;
        const ABC_TAG: i32 = 3; // top edge
        let mesh = Mesh::<2>::unit_square_tri(4);

        for n_ranks in [2usize, 4] {
            let mesh = mesh.clone();
            ThreadLauncher::new(WorkerConfig::new(n_ranks)).launch(move |comm| {
                let pmesh = partition_mesh(&mesh, &comm);
                let local_space = HCurlSpace::new(pmesh.local_mesh().clone(), 1);
                let par_space = ParallelFESpace::new_for_edge_space(
                    local_space, &pmesh, comm.clone(),
                );

                let base = ParVectorAssembler::assemble_bilinear(
                    &par_space,
                    &[&VectorMassIntegrator { alpha: 1.0 }],
                    3,
                );
                let bnd = ParVectorAssembler::assemble_boundary_bilinear(
                    &par_space,
                    &[&TangentialMassIntegrator { gamma: 0.5 }],
                    &[ABC_TAG],
                    3,
                );

                let mut acc = base.clone_vec();
                ParVectorAssembler::add_boundary_bilinear(
                    &mut acc,
                    &par_space,
                    &[&TangentialMassIntegrator { gamma: 0.5 }],
                    &[ABC_TAG],
                    3,
                    0.5 * DT,
                );

                let m_base = base.to_local_matrix();
                let m_bnd = bnd.to_local_matrix();
                let m_acc = acc.to_local_matrix();
                let nr = m_acc.nrows;
                let nc = m_acc.ncols;
                let mut max_dev = 0.0_f64;
                for r in 0..nr {
                    for c in 0..nc {
                        let want = m_base.get(r, c) + 0.5 * DT * m_bnd.get(r, c);
                        let got = m_acc.get(r, c);
                        let dev = (want - got).abs();
                        if dev.is_nan() || !got.is_finite() {
                            max_dev = f64::INFINITY;
                        } else {
                            max_dev = max_dev.max(dev);
                        }
                    }
                }
                assert!(
                    max_dev < 1e-12,
                    "rank {} of {}: a += factor·B max dev {max_dev:.3e}",
                    comm.rank(),
                    comm.size()
                );
            });
        }
    }

    #[test]
    fn par_vector_assembly_hcurl_diagonal_positive() {
        let mesh = Mesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
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
        let mesh = Mesh::<2>::unit_square_tri(4);
        let serial_space = HCurlSpace::new(mesh.clone(), 1);
        let serial_n = serial_space.n_dofs();

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
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

        let mesh = Mesh::<2>::unit_square_tri(2);
        let serial_space = HCurlSpace::new(mesh.clone(), 1);
        let curl_curl = CurlCurlIntegrator { mu: 1.0 };
        let vec_mass = VectorMassIntegrator { alpha: 1.0 };
        let serial_mat = VectorAssembler::assemble_bilinear(
            &serial_space, &[&curl_curl, &vec_mass], 4,
        );

        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
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
        let mesh = Mesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
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

        let mesh = Mesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
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
