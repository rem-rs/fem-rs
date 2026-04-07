//! Parallel mixed bilinear form assembly.
//!
//! [`ParMixedAssembler`] wraps the serial [`MixedAssembler`] and produces a
//! rectangular `CsrMatrix` split into owned/ghost row partitions for parallel
//! saddle-point systems.

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_space::fe_space::FESpace;
use fem_assembly::mixed::{MixedAssembler, MixedBilinearIntegrator};

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
        // Serial mixed assembly on local mesh.
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
}

/// Permute a rectangular CSR matrix using row and column DOF partitions.
fn permute_rect_csr(
    mat: &CsrMatrix<f64>,
    row_part: &DofPartition,
    col_part: &DofPartition,
) -> CsrMatrix<f64> {
    let nr = row_part.n_total_dofs();
    let nc = col_part.n_total_dofs();
    let mut coo = CooMatrix::<f64>::new(nr, nc);

    for row in 0..mat.nrows {
        let new_row = row_part.permute_dof(row as u32) as usize;
        for k in mat.row_ptr[row]..mat.row_ptr[row + 1] {
            let col = mat.col_idx[k] as usize;
            let new_col = col_part.permute_dof(col as u32) as usize;
            let val = mat.values[k];
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
