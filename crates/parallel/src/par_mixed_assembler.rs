//! Parallel mixed bilinear form assembly.
//!
//! [`ParMixedAssembler`] wraps the serial [`MixedAssembler`] and produces a
//! rectangular `CsrMatrix` split into owned/ghost row partitions for parallel
//! saddle-point systems.

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_space::fe_space::FESpace;
use fem_assembly::mixed::{
    MixedAssembler, MixedBilinearIntegrator, HDivL2Integrator, assemble_hdiv_l2_mixed,
    assemble_hcurl_h1_gradient, assemble_hcurl_hdiv_weak_curl,
    assemble_hcurl_h1_weak_div, HCurlH1WeakDiv, HCurlH1WeakDivIntegrator,
};

use crate::dof_partition::DofPartition;
use crate::par_space::ParallelFESpace;

/// Parallel mixed bilinear form assembler.
///
/// Produces a rectangular `CsrMatrix` where rows correspond to the row-space's
/// **owned** DOFs and columns span the full column-space local DOF range
/// (owned + ghost).
pub struct ParMixedAssembler;

impl ParMixedAssembler {
    /// Assemble a mixed bilinear form `b(u, v)` in parallel.
    ///
    /// - `row_par_space` — parallel row/test space (determines owned rows).
    /// - `col_par_space` — parallel column/trial space (full local columns).
    ///
    /// Both spaces must share the same local mesh (with ghost overlap).
    ///
    /// Returns a `CsrMatrix` with `n_owned_row` rows and `n_total_col` columns.
    pub fn assemble_bilinear<SR: FESpace, SC: FESpace>(
        row_par_space: &ParallelFESpace<SR>,
        col_par_space: &ParallelFESpace<SC>,
        integrators: &[&dyn MixedBilinearIntegrator],
        quad_order: u8,
    ) -> CsrMatrix<f64> {
        // Local mixed assembly (Rayon volume loop when `fem-assembly/parallel` and
        // `n_elements >= FEM_ASSEMBLY_PARALLEL_MIN_ELEMS`).
        let local_mat = MixedAssembler::assemble_bilinear(
            row_par_space.local_space(),
            col_par_space.local_space(),
            integrators,
            quad_order,
        );

        // Permute if needed.
        let row_part = row_par_space.dof_partition();
        let col_part = col_par_space.dof_partition();

        let needs_perm = row_part.needs_permutation() || col_part.needs_permutation();
        let permuted_mat = if needs_perm {
            permute_rect_csr(&local_mat, row_part, col_part)
        } else {
            local_mat
        };

        // Keep only owned rows (discard ghost rows).
        let n_owned_rows = row_part.n_owned_dofs;
        let n_total_cols = col_part.n_total_dofs();
        extract_owned_rows(&permuted_mat, n_owned_rows, n_total_cols)
    }

    /// Parallel mixed assembly for HDiv × L2 (Darcy divergence coupling),
    /// via the dedicated [`assemble_hdiv_l2_mixed`] path (the generic
    /// [`MixedAssembler`] skips vector-valued spaces).
    ///
    /// `row_par_space` is the L2 (pressure) space, `col_par_space` the HDiv
    /// (velocity) space.  Returns `n_owned_row × n_total_col` CSR.
    pub fn assemble_hdiv_l2<SR: FESpace, SC: FESpace>(
        row_par_space: &ParallelFESpace<SR>,
        col_par_space: &ParallelFESpace<SC>,
        integrators: &[&dyn HDivL2Integrator],
        quad_order: u8,
    ) -> CsrMatrix<f64> {
        let local_mat = assemble_hdiv_l2_mixed(
            row_par_space.local_space(),
            col_par_space.local_space(),
            integrators,
            quad_order,
        );

        let row_part = row_par_space.dof_partition();
        let col_part = col_par_space.dof_partition();
        let needs_perm = row_part.needs_permutation() || col_part.needs_permutation();
        let permuted_mat = if needs_perm {
            permute_rect_csr(&local_mat, row_part, col_part)
        } else {
            local_mat
        };

        // Keep ALL local rows (owned + ghost L2 rows): the transpose Bᵀ needs
        // the ghost-L2 columns to pair with cross-rank B entries (B[j][c] with
        // j on this rank and c owned elsewhere).  Dropping ghost rows made A01
        // miss those columns → (Au)·v ≠ u·(Av) for the saddle-point operator.
        permuted_mat
    }

    /// Parallel mixed assembly for the H¹ × H(curl) gradient coupling
    /// `(∇p, v)` (MFEM `MixedVectorGradientIntegrator`), via the serial
    /// [`assemble_hcurl_h1_gradient`] path.
    ///
    /// Returns a CSR with `n_owned_row` (H(curl)) rows and `n_total_col`
    /// (H¹) columns.
    pub fn assemble_hcurl_h1_gradient<M: fem_mesh::topology::MeshTopology + Clone + 'static>(
        h1_par: &ParallelFESpace<fem_space::H1Space<M>>,
        hcurl_par: &ParallelFESpace<fem_space::HCurlSpace<M>>,
        quad_order: u8,
    ) -> CsrMatrix<f64> {
        let local_mat = assemble_hcurl_h1_gradient(
            hcurl_par.local_space(),
            h1_par.local_space(),
            quad_order,
        );
        let row_part = hcurl_par.dof_partition();
        let col_part = h1_par.dof_partition();
        let needs_perm = row_part.needs_permutation() || col_part.needs_permutation();
        let permuted_mat = if needs_perm {
            permute_rect_csr(&local_mat, row_part, col_part)
        } else {
            local_mat
        };
        // Keep only owned H(curl) rows.
        let n_owned_rows = row_part.n_owned_dofs;
        let n_total_cols = col_part.n_total_dofs();
        extract_owned_rows(&permuted_mat, n_owned_rows, n_total_cols)
    }

    /// Parallel mixed assembly for the H(curl) × H(div) curl coupling
    /// `(curl v, w)` (MFEM `MixedVectorCurlIntegrator`), via the serial
    /// [`assemble_hcurl_hdiv_weak_curl`] path.
    ///
    /// Returns a CSR with `n_owned_row` (H(div)) rows and `n_total_col`
    /// (H(curl)) columns.
    pub fn assemble_hcurl_hdiv_curl<M: fem_mesh::topology::MeshTopology + Clone + 'static>(
        nd_par: &ParallelFESpace<fem_space::HCurlSpace<M>>,
        rt_par: &ParallelFESpace<fem_space::HDivSpace<M>>,
        quad_order: u8,
    ) -> CsrMatrix<f64> {
        Self::assemble_hcurl_hdiv_curl_with_coeff(nd_par, rt_par, quad_order, 1.0_f64)
    }

    /// Parallel mixed assembly for H(curl) × H(div) curl coupling with coefficient.
    pub fn assemble_hcurl_hdiv_curl_with_coeff<M, C>(
        nd_par: &ParallelFESpace<fem_space::HCurlSpace<M>>,
        rt_par: &ParallelFESpace<fem_space::HDivSpace<M>>,
        quad_order: u8,
        coeff: C,
    ) -> CsrMatrix<f64>
    where
        M: fem_mesh::topology::MeshTopology + Clone + 'static,
        C: fem_assembly::postproc::coefficient::ScalarCoeff,
    {
        let local_mat = assemble_hcurl_hdiv_weak_curl(
            nd_par.local_space(),
            rt_par.local_space(),
            quad_order,
            coeff,
        );
        let row_part = rt_par.dof_partition();
        let col_part = nd_par.dof_partition();
        let needs_perm = row_part.needs_permutation() || col_part.needs_permutation();
        let permuted_mat = if needs_perm {
            permute_rect_csr(&local_mat, row_part, col_part)
        } else {
            local_mat
        };
        let n_owned_rows = row_part.n_owned_dofs;
        let n_total_cols = col_part.n_total_dofs();
        extract_owned_rows(&permuted_mat, n_owned_rows, n_total_cols)
    }

    /// Parallel mixed assembly for the **H(curl)-row** weak curl
    /// `B[i,j] = ∫ ν · w_j · curl(ψ_i) dx` (rows `i` over the H(curl) space,
    /// columns `j` over the H(div) space) — MFEM
    /// `ParMixedBilinearForm(HDivFESpace_, HCurlFESpace_)` +
    /// `MixedVectorWeakCurlIntegrator`, i.e. maxwell's `WeakCurlMuInv_`.
    ///
    /// This is the multi-rank-correct counterpart of taking
    /// [`Self::assemble_hcurl_hdiv_curl_with_coeff`] and calling
    /// `.transpose()` at the call site: that transposes an already
    /// ghost-row-truncated matrix and therefore yields rows for every *local*
    /// H(curl) DOF, ghost rows carrying only this rank's element
    /// contributions.  Here the local matrix is assembled in the
    /// H(curl)-row orientation, permuted with the **H(curl)** partition as the
    /// row partition, and only the owned H(curl) rows are kept.
    ///
    /// Returns a CSR with `n_owned_row` (H(curl)) rows and `n_total_col`
    /// (H(div), owned + ghost) columns.
    pub fn assemble_hdiv_hcurl_curl_with_coeff<M, C>(
        nd_par: &ParallelFESpace<fem_space::HCurlSpace<M>>,
        rt_par: &ParallelFESpace<fem_space::HDivSpace<M>>,
        quad_order: u8,
        coeff: C,
    ) -> CsrMatrix<f64>
    where
        M: fem_mesh::topology::MeshTopology + Clone + 'static,
        C: fem_assembly::postproc::coefficient::ScalarCoeff,
    {
        let local_mat = fem_assembly::mixed::assemble_hdiv_hcurl_weak_curl(
            nd_par.local_space(),
            rt_par.local_space(),
            quad_order,
            coeff,
        );
        // Rows are H(curl) DOFs, columns H(div) DOFs — the partitions must be
        // passed in that same order (the row partition is what `permute_rect_csr`
        // reorders to [owned | ghost], and what the extraction below keeps).
        let row_part = nd_par.dof_partition();
        let col_part = rt_par.dof_partition();
        let needs_perm = row_part.needs_permutation() || col_part.needs_permutation();
        let permuted_mat = if needs_perm {
            permute_rect_csr(&local_mat, row_part, col_part)
        } else {
            local_mat
        };
        let n_owned_rows = row_part.n_owned_dofs;
        let n_total_cols = col_part.n_total_dofs();
        extract_owned_rows(&permuted_mat, n_owned_rows, n_total_cols)
    }

    /// Parallel mixed assembly for the H(curl) × H¹ weak divergence
    /// `b(v, u) = -∫ v (∇·u) dx` (MFEM `VectorFEWeakDivergenceIntegrator`
    /// on `ParMixedBilinearForm(HCurlFESpace_, H1FESpace_)`), via the serial
    /// [`fem_assembly::mixed::assemble_hcurl_h1_weak_div`] path.
    ///
    /// Returns a CSR with `n_owned_row` (H¹) rows and `n_total_col`
    /// (H(curl)) columns.
    pub fn assemble_hcurl_h1_weak_div<M>(
        h1_par: &ParallelFESpace<fem_space::H1Space<M>>,
        nd_par: &ParallelFESpace<fem_space::HCurlSpace<M>>,
        quad_order: u8,
    ) -> CsrMatrix<f64>
    where
        M: fem_mesh::topology::MeshTopology + Clone + 'static,
    {
        let integ = HCurlH1WeakDiv::new(1.0_f64);
        let local_mat = assemble_hcurl_h1_weak_div(
            h1_par.local_space(),
            nd_par.local_space(),
            &[&integ],
            quad_order,
        );
        let row_part = h1_par.dof_partition();
        let col_part = nd_par.dof_partition();
        let needs_perm = row_part.needs_permutation() || col_part.needs_permutation();
        let permuted_mat = if needs_perm {
            permute_rect_csr(&local_mat, row_part, col_part)
        } else {
            local_mat
        };
        let n_owned_rows = row_part.n_owned_dofs;
        let n_total_cols = col_part.n_total_dofs();
        extract_owned_rows(&permuted_mat, n_owned_rows, n_total_cols)
    }
    ///
    /// `eps` may be a spatially-varying [`fem_assembly::postproc::coefficient::ScalarCoeff`]
    /// (Volta `-ds` dielectric sphere / `-pwe` piecewise-constant ε).
    ///
    /// Returns a CSR with `n_owned_row` (H(div)) rows and `n_total_col`
    /// (H(curl)) columns.
    pub fn assemble_hcurl_hdiv_mass<M, C>(
        nd_par: &ParallelFESpace<fem_space::HCurlSpace<M>>,
        rt_par: &ParallelFESpace<fem_space::HDivSpace<M>>,
        quad_order: u8,
        eps: C,
    ) -> CsrMatrix<f64>
    where
        M: fem_mesh::topology::MeshTopology + Clone + 'static,
        C: fem_assembly::postproc::coefficient::ScalarCoeff,
    {
        let local_mat = fem_assembly::mixed::assemble_hcurl_hdiv_mixed(
            nd_par.local_space(),
            rt_par.local_space(),
            quad_order,
            eps,
        );
        let row_part = rt_par.dof_partition();
        let col_part = nd_par.dof_partition();
        let needs_perm = row_part.needs_permutation() || col_part.needs_permutation();
        let permuted_mat = if needs_perm {
            permute_rect_csr(&local_mat, row_part, col_part)
        } else {
            local_mat
        };
        let n_owned_rows = row_part.n_owned_dofs;
        let n_total_cols = col_part.n_total_dofs();
        extract_owned_rows(&permuted_mat, n_owned_rows, n_total_cols)
    }
}

/// Permute a rectangular CSR matrix using row and column DOF partitions.
/// Permute a rectangular local CSR matrix from DofManager order to the
/// partition `[owned | ghost]` order (row and column partitions may differ).
pub fn permute_rect_csr(
    mat: &CsrMatrix<f64>,
    row_part: &DofPartition,
    col_part: &DofPartition,
) -> CsrMatrix<f64> {
    let nr = row_part.n_total_dofs();
    let nc = col_part.n_total_dofs();
    let mut coo = CooMatrix::<f64>::new(nr, nc);
    let row_sign = row_part.needs_sign_correction();
    let col_sign = col_part.needs_sign_correction();

    for row in 0..mat.nrows {
        let new_row = row_part.permute_dof(row as u32) as usize;
        let sr = if row_sign {
            row_part.sign_correction(row as u32)
        } else {
            1.0
        };
        for k in mat.row_ptr[row]..mat.row_ptr[row + 1] {
            let col = mat.col_idx[k] as usize;
            let new_col = col_part.permute_dof(col as u32) as usize;
            let sc = if col_sign {
                col_part.sign_correction(col as u32)
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

/// Extract the first `n_owned_rows` rows from a CSR matrix.
fn extract_owned_rows(
    mat: &CsrMatrix<f64>,
    n_owned_rows: usize,
    n_cols: usize,
) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n_owned_rows, n_cols);
    for row in 0..n_owned_rows.min(mat.nrows) {
        for k in mat.row_ptr[row]..mat.row_ptr[row + 1] {
            let col = mat.col_idx[k] as usize;
            let val = mat.values[k];
            if val != 0.0 && col < n_cols {
                coo.add(row, col, val);
            }
        }
    }
    coo.into_csr()
}

// ─── ParMixedSesquilinearForm — parallel complex mixed bilinear form ─────────
//
// Mirrors MFEM 4.10's ParMixedSesquilinearForm: parallel complex-valued mixed
// bilinear forms with two different spaces (trial and test).

use fem_assembly::complex::Convention;

/// Parallel complex-valued mixed bilinear form `b(u, v)` where `u ∈ U` (trial)
/// and `v ∈ V` (test) are different parallel spaces.
///
/// Produces a `MixedComplexSystem` with rectangular matrices:
/// - `k_re`: n_owned_test × n_trial (real part)
/// - `k_im`: n_owned_test × n_trial (imaginary part)
pub struct ParMixedSesquilinearForm<'a, 'b, 'c, S1: FESpace + Send + Sync, S2: FESpace + Send + Sync> {
    trial_par_space: &'a ParallelFESpace<S1>,
    test_par_space: &'a ParallelFESpace<S2>,
    conv: Convention,
    quad_order: u8,
    pairs: Vec<(
        &'b dyn MixedBilinearIntegrator,
        Option<&'c dyn MixedBilinearIntegrator>,
    )>,
}

impl<'a, 'b, 'c, S1: FESpace + Send + Sync, S2: FESpace + Send + Sync>
    ParMixedSesquilinearForm<'a, 'b, 'c, S1, S2>
{
    pub fn new(
        trial_par_space: &'a ParallelFESpace<S1>,
        test_par_space: &'a ParallelFESpace<S2>,
        conv: Convention,
        quad_order: u8,
    ) -> Self {
        ParMixedSesquilinearForm {
            trial_par_space,
            test_par_space,
            conv,
            quad_order,
            pairs: Vec::new(),
        }
    }

    /// Add a mixed integrator pair (real and imaginary parts).
    pub fn add_mixed_domain_integrator_pair(
        &mut self,
        re_integ: &'b dyn MixedBilinearIntegrator,
        im_integ: Option<&'c dyn MixedBilinearIntegrator>,
    ) {
        self.pairs.push((re_integ, im_integ));
    }

    /// Assemble the parallel mixed complex system.
    pub fn assemble(self) -> ParMixedComplexSystem {
        let mut re_all: Vec<&dyn MixedBilinearIntegrator> = Vec::new();
        let mut im_all: Vec<&dyn MixedBilinearIntegrator> = Vec::new();
        for (re, im) in &self.pairs {
            re_all.push(*re);
            if let Some(im) = im {
                im_all.push(*im);
            }
        }

        let n_trial = self.trial_par_space.local_space().n_dofs();
        let n_test = self.test_par_space.local_space().n_dofs();

        let k_re = if !re_all.is_empty() {
            ParMixedAssembler::assemble_bilinear(
                self.test_par_space,
                self.trial_par_space,
                &re_all,
                self.quad_order,
            )
        } else {
            CsrMatrix::new_empty(n_test, n_trial)
        };

        let k_im = if !im_all.is_empty() {
            ParMixedAssembler::assemble_bilinear(
                self.test_par_space,
                self.trial_par_space,
                &im_all,
                self.quad_order,
            )
        } else {
            CsrMatrix::new_empty(n_test, n_trial)
        };

        ParMixedComplexSystem {
            k_re,
            k_im,
            n_trial,
            n_test,
            omega: 0.0,
        }
    }
}

/// Parallel complex system with rectangular matrices (mixed spaces).
#[derive(Debug, Clone)]
pub struct ParMixedComplexSystem {
    /// Real part: n_owned_test × n_trial.
    pub k_re: CsrMatrix<f64>,
    /// Imaginary part: n_owned_test × n_trial.
    pub k_im: CsrMatrix<f64>,
    /// Number of trial DOFs.
    pub n_trial: usize,
    /// Number of test DOFs.
    pub n_test: usize,
    /// Frequency (for Helmholtz-type problems).
    pub omega: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn par_mixed_complex_system_shape() {
        // ParMixedComplexSystem should have correct dimensions
        let k_re = CsrMatrix::<f64>::new_empty(5, 10);
        let k_im = CsrMatrix::<f64>::new_empty(5, 10);
        let sys = ParMixedComplexSystem {
            k_re,
            k_im,
            n_trial: 10,
            n_test: 5,
            omega: 0.0,
        };
        assert_eq!(sys.n_trial, 10);
        assert_eq!(sys.n_test, 5);
        assert_eq!(sys.k_re.nrows, 5);
        assert_eq!(sys.k_re.ncols, 10);
    }

    /// B1 / D88: the **owned-H(curl)-row** weak curl
    /// ([`ParMixedAssembler::assemble_hdiv_hcurl_curl_with_coeff`]) must be the
    /// rank-local transpose of the H(div)-row mixed curl, *permuted into the
    /// partition basis and truncated to the owned H(curl) rows*.
    ///
    /// The reference is the full local (DofManager-order) transposed assembly —
    /// `B_par[permute(d)][permute(c)] == B_loc[d][c]·s_nd(d)·s_rt(c)` — which
    /// pins four independent things at once: the **row space** really is
    /// H(curl) (swapping the two partitions breaks every entry), the row
    /// partition used for reordering is the H(curl) one, the sign convention
    /// matches `permute_rect_csr`'s (the signs of the *right* space), and the
    /// column range keeps the ghost H(div) DOFs.
    ///
    /// Runs at 1, 2 and 4 ranks.  At one rank this is bit-for-bit the statement
    /// "equals `assemble_hcurl_hdiv_curl_with_coeff(…).transpose()`" (identity
    /// permutation, no ghosts), which is the acceptance criterion for the
    /// single-rank limit; at 2+ ranks the same identity is checked through the
    /// real permutation and with non-empty ghost blocks.
    #[test]
    fn owned_hcurl_rows_match_local_transpose_at_all_rank_counts() {
        use crate::dof_partition::DofPartition;
        use crate::launcher::native::ThreadLauncher;
        use crate::launcher::WorkerConfig;
        use crate::par_partition::partition_mesh;
        use crate::par_space::ParallelFESpace;
        use fem_mesh::Mesh;
        use fem_space::fe_space::FESpace as _;
        use fem_space::{HCurlSpace, HDivSpace};
        use std::sync::{Arc, Mutex};

        let mesh = Mesh::<3>::unit_cube_hex(2);
        for n_ranks in [1usize, 2, 4] {
            let seen = Arc::new(Mutex::new(Vec::new()));
            let seen_rank = Arc::clone(&seen);
            let mesh = mesh.clone();
            ThreadLauncher::new(WorkerConfig::new(n_ranks)).launch(move |comm| {
                let pmesh = partition_mesh(&mesh, &comm);
                let lm = pmesh.local_mesh().clone();
                let nd = HCurlSpace::new(lm.clone(), 1);
                let rt = HDivSpace::new(lm.clone(), 0);
                let nd_par =
                    ParallelFESpace::new_for_edge_space(nd, &pmesh, comm.clone());
                // 3-D H(div) is face-based → the face partition.
                let rt_part = DofPartition::from_face_space(&rt, pmesh.partition(), &comm);
                let rt_par =
                    ParallelFESpace::new_with_dof_partition(rt, rt_part, comm.clone());

                let b_par =
                    ParMixedAssembler::assemble_hdiv_hcurl_curl_with_coeff(
                        &nd_par,
                        &rt_par,
                        3,
                        1.0_f64,
                    );
                let b_loc = fem_assembly::mixed::assemble_hdiv_hcurl_weak_curl(
                    nd_par.local_space(),
                    rt_par.local_space(),
                    3,
                    1.0_f64,
                );

                let ndp = nd_par.dof_partition();
                let rtp = rt_par.dof_partition();
                // Owned H(curl) rows × all local H(div) columns.
                assert_eq!(b_par.nrows, ndp.n_owned_dofs, "row count = owned H(curl)");
                assert_eq!(b_par.ncols, rtp.n_total_dofs(), "column count = local H(div)");

                let n_nd_local = nd_par.local_space().n_dofs();
                let n_rt_local = rt_par.local_space().n_dofs();
                assert_eq!(b_loc.nrows, n_nd_local);
                assert_eq!(b_loc.ncols, n_rt_local);

                let mut max_dev = 0.0_f64;
                let mut checked_rows = 0usize;
                for d in 0..n_nd_local as u32 {
                    let p = ndp.permute_dof(d) as usize;
                    if p >= ndp.n_owned_dofs {
                        continue; // ghost H(curl) row — deliberately dropped
                    }
                    checked_rows += 1;
                    let sr = ndp.sign_correction(d);
                    for c in 0..n_rt_local as u32 {
                        let q = rtp.permute_dof(c) as usize;
                        let want =
                            b_loc.get(d as usize, c as usize) * sr * rtp.sign_correction(c);
                        let got = b_par.get(p, q);
                        let dev = (want - got).abs();
                        if dev.is_nan() || !got.is_finite() {
                            max_dev = f64::INFINITY;
                        } else {
                            max_dev = max_dev.max(dev);
                        }
                    }
                }
                assert!(checked_rows > 0, "rank {}: no owned H(curl) rows", comm.rank());
                assert!(
                    max_dev < 1e-12,
                    "rank {} ({} ranks): owned-H(curl)-row weak curl vs local \
                     transposed assembly max dev {max_dev:.3e}",
                    comm.rank(),
                    comm.size()
                );
                seen_rank.lock().unwrap().push(comm.rank());
            });
            assert_eq!(
                seen.lock().unwrap().len(),
                n_ranks,
                "{n_ranks}-rank launch must visit every rank"
            );
        }
    }

    /// D88 gap pin: at 2 ranks the call-site workaround (transpose of the
    /// **already row-truncated** H(div)-row matrix) exposes rows for *all*
    /// local H(curl) DOFs, ghost rows included — rows the new path drops.  This
    /// is why the new entry point exists rather than a `.transpose()` at the
    /// call site.
    #[test]
    fn old_transpose_workaround_exposes_ghost_hcurl_rows() {
        use crate::dof_partition::DofPartition;
        use crate::launcher::native::ThreadLauncher;
        use crate::launcher::WorkerConfig;
        use crate::par_partition::partition_mesh;
        use crate::par_space::ParallelFESpace;
        use fem_mesh::Mesh;
        use fem_space::{HCurlSpace, HDivSpace};

        let mesh = Mesh::<3>::unit_cube_hex(2);
        ThreadLauncher::new(WorkerConfig::new(2)).launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let lm = pmesh.local_mesh().clone();
            let nd = ParallelFESpace::new_for_edge_space(
                HCurlSpace::new(lm.clone(), 1),
                &pmesh,
                comm.clone(),
            );
            let rt_local = HDivSpace::new(lm, 0);
            let rt_part = DofPartition::from_face_space(&rt_local, pmesh.partition(), &comm);
            let rt = ParallelFESpace::new_with_dof_partition(rt_local, rt_part, comm.clone());

            let old = ParMixedAssembler::assemble_hcurl_hdiv_curl_with_coeff(
                &nd,
                &rt,
                3,
                1.0_f64,
            )
            .transpose();
            let new =
                ParMixedAssembler::assemble_hdiv_hcurl_curl_with_coeff(&nd, &rt, 3, 1.0_f64);

            assert_eq!(new.nrows, nd.dof_partition().n_owned_dofs);
            assert!(nd.dof_partition().n_ghost_dofs > 0, "expected ghost H(curl) DOFs");
            assert_eq!(
                old.nrows,
                nd.dof_partition().n_total_dofs(),
                "the transpose workaround keeps ghost H(curl) rows"
            );
            assert!(old.nrows > new.nrows, "the two paths must differ in shape");
        });
    }
}
