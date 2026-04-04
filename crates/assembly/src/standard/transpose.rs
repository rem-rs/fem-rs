//! Transpose integrator wrapper.
//!
//! Wraps any [`BilinearIntegrator`] and transposes its element matrix
//! contribution: the entry that would go to `K[i,j]` is placed at `K[j,i]`.

use crate::integrator::{BilinearIntegrator, QpData};

/// Wraps a bilinear integrator, transposing its element matrix contribution.
///
/// If `A` produces a non-symmetric element matrix, `TransposeIntegrator(A)`
/// produces its transpose.  If `A` is already symmetric, this is a no-op.
///
/// # Example
/// ```rust,ignore
/// use fem_assembly::standard::{ConvectionIntegrator, TransposeIntegrator};
/// // The adjoint convection operator:
/// let adj = TransposeIntegrator(ConvectionIntegrator { velocity: b });
/// ```
pub struct TransposeIntegrator<I: BilinearIntegrator>(pub I);

impl<I: BilinearIntegrator> BilinearIntegrator for TransposeIntegrator<I> {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let mut tmp = vec![0.0_f64; n * n];
        self.0.add_to_element_matrix(qp, &mut tmp);
        for i in 0..n {
            for j in 0..n {
                k_elem[i * n + j] += tmp[j * n + i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::Assembler;
    use crate::standard::DiffusionIntegrator;
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;

    /// Transpose of a symmetric integrator (diffusion) should give the same matrix.
    #[test]
    fn transpose_of_symmetric_is_same() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let integ = DiffusionIntegrator { kappa: 1.0 };
        let trans = TransposeIntegrator(DiffusionIntegrator { kappa: 1.0 });

        let mat_a = Assembler::assemble_bilinear(&space, &[&integ], 2);
        let mat_t = Assembler::assemble_bilinear(&space, &[&trans], 2);

        let da = mat_a.to_dense();
        let dt = mat_t.to_dense();
        let n = mat_a.nrows;
        for i in 0..n {
            for j in 0..n {
                let diff = (da[i * n + j] - dt[i * n + j]).abs();
                assert!(diff < 1e-12, "diff at ({i},{j}) = {diff}");
            }
        }
    }
}
