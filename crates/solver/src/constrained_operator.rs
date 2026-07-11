//! RectangularConstrainedOperator — wraps a prolongation/restriction operator
//! to enforce essential BC DOFs (matching MFEM `RectangularConstrainedOperator`).
//!
//! When prolongating a coarse-grid correction to the fine grid, this operator
//! ensures that fine-grid BC DOFs remain zero.  When restricting a fine-grid
//! residual to the coarse grid, coarse-grid BC DOFs are zeroed out.

/// Rectangular-constrained operator for prolongation with BC enforcement.
///
/// `Mult(x_coarse, y_fine)`: `y_fine = P * x_coarse` with fine BC DOFs zeroed and
/// coarse BC DOFs set to 0 in the input before multiplying.
///
/// `MultTranspose(x_fine, y_coarse)`: `y_coarse = P^T * x_fine` with coarse BC DOFs
/// zeroed and fine BC DOFs set to 0 in the input before multiplying.
pub struct RectangularConstrainedOperator {
    /// The underlying prolongation matrix (fine_rows × coarse_cols).
    pub mat: fem_linalg::CsrMatrix<f64>,
    /// Essential BC DOFs on the fine (row) space.
    pub ess_fine: Vec<u32>,
    /// Essential BC DOFs on the coarse (column) space.
    pub ess_coarse: Vec<u32>,
}

impl RectangularConstrainedOperator {
    /// Prolongation: `y_fine += P * x_coarse`, with BC enforcement.
    ///
    /// Equivalent to MFEM `RectangularConstrainedOperator::Mult`:
    ///   temp = (x_coarse with ess_coarse zeroed)
    ///   y_fine = P * temp
    ///   zero ess_fine in y_fine
    pub fn prolong(&self, x_coarse: &[f64], y_fine: &mut [f64]) {
        // Zero coarse BC entries in the input
        let mut x = x_coarse.to_vec();
        for &d in &self.ess_coarse {
            if (d as usize) < x.len() { x[d as usize] = 0.0; }
        }
        // y_fine = P * x
        self.mat.spmv(&x, y_fine);
        // Zero fine BC entries in the output
        for &d in &self.ess_fine {
            if (d as usize) < y_fine.len() { y_fine[d as usize] = 0.0; }
        }
    }

    /// Restriction: `y_coarse += P^T * x_fine`, with BC enforcement.
    ///
    /// Equivalent to MFEM `RectangularConstrainedOperator::MultTranspose`:
    ///   temp = (x_fine with ess_fine zeroed)
    ///   y_coarse = P^T * temp
    ///   zero ess_coarse in y_coarse
    pub fn restrict(&self, x_fine: &[f64], y_coarse: &mut Vec<f64>) {
        // Zero fine BC entries in the input
        let mut x = x_fine.to_vec();
        for &d in &self.ess_fine {
            if (d as usize) < x.len() { x[d as usize] = 0.0; }
        }
        // y_coarse = P^T * x (transpose SpMV)
        let n_coarse = self.mat.ncols;
        y_coarse.clear();
        y_coarse.resize(n_coarse, 0.0);
        for row in 0..self.mat.nrows {
            let val = x[row];
            if val == 0.0 { continue; }
            for p in self.mat.row_ptr[row]..self.mat.row_ptr[row + 1] {
                y_coarse[self.mat.col_idx[p] as usize] += self.mat.values[p] * val;
            }
        }
        // Zero coarse BC entries in the output
        for &d in &self.ess_coarse {
            if (d as usize) < y_coarse.len() { y_coarse[d as usize] = 0.0; }
        }
    }
}
