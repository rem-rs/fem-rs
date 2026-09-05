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
        let local_mat = assemble_hcurl_hdiv_weak_curl(
            nd_par.local_space(),
            rt_par.local_space(),
            quad_order,
            1.0,
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

    /// Parallel mixed assembly for the H(curl) × H(div) mass coupling
    /// `∫ ε · ψ_j · w_i dx` (MFEM `VectorFEMassIntegrator` on
    /// `ParMixedBilinearForm(HCurlFESpace_, HDivFESpace_)` — Volta's
    /// `hCurlHDivEps_` / `hCurlHDiv_`), via the serial
    /// [`fem_assembly::mixed::assemble_hcurl_hdiv_mixed`] path.
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
}
