//! Diffusion (Laplacian) bilinear form integrator.
//!
//! Computes the element contribution to
//!
//! ```text
//! a(u, v) = ∫_Ω κ ∇u · ∇v dx
//! ```

use crate::coefficient::{CoeffCtx, ScalarCoeff};
use crate::integrator::{BilinearIntegrator, QpData};

/// Bilinear integrator for the scalar diffusion operator `κ ∇u · ∇v`.
///
/// For `κ = 1` this is the standard Laplacian stiffness matrix.
///
/// The coefficient `κ` is generic over [`ScalarCoeff`], with `f64` as the
/// default for full backwards compatibility:
///
/// ```rust,ignore
/// // Constant (unchanged):
/// let integ = DiffusionIntegrator { kappa: 1.0 };
///
/// // Spatially varying:
/// let integ = DiffusionIntegrator { kappa: FnCoeff(|x: &[f64]| 1.0 + x[0]) };
///
/// // Piecewise constant per material:
/// let integ = DiffusionIntegrator { kappa: PWConstCoeff::new([(1, 1.0), (2, 100.0)]) };
/// ```
pub struct DiffusionIntegrator<C: ScalarCoeff = f64> {
    /// Scalar conductivity / diffusivity coefficient.
    pub kappa: C,
}

impl<C: ScalarCoeff> BilinearIntegrator for DiffusionIntegrator<C> {
    /// `K_elem[i,j] += w · κ(x) · (∇φᵢ · ∇φⱼ)`
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n   = qp.n_dofs;
        let d   = qp.dim;
        let ctx = CoeffCtx::from_qp(
            qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag,
            Some(qp.phi), qp.elem_dofs,
        );
        let w_k = qp.weight * self.kappa.eval(&ctx);

        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..d {
                    dot += qp.grad_phys[i * d + k] * qp.grad_phys[j * d + k];
                }
                k_elem[i * n + j] += w_k * dot;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::Assembler;
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;

    /// The stiffness matrix from a DiffusionIntegrator on the reference triangle
    /// (one element) should be symmetric positive semi-definite with a known
    /// row-sum of zero (constant functions are in the kernel of ∇).
    #[test]
    fn stiffness_row_sum_zero_single_element() {
        // Build a 1-element mesh (one triangle).
        let mesh  = SimplexMesh::<2>::unit_square_tri(1);
        let space = H1Space::new(mesh, 1);
        let integ = DiffusionIntegrator { kappa: 1.0 };
        let mat   = Assembler::assemble_bilinear(&space, &[&integ], 2);

        // Row sums must be ≈ 0 (Neumann compatibility).
        let dense = mat.to_dense();
        for row in 0..mat.nrows {
            let s: f64 = (0..mat.ncols).map(|c| dense[row * mat.ncols + c]).sum();
            assert!(s.abs() < 1e-12, "row {row} sum = {s}");
        }
    }

    /// Symmetry check: K[i,j] == K[j,i].
    #[test]
    fn stiffness_is_symmetric() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let integ = DiffusionIntegrator { kappa: 1.0 };
        let mat   = Assembler::assemble_bilinear(&space, &[&integ], 2);
        let dense = mat.to_dense();
        let n = mat.nrows;
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i * n + j] - dense[j * n + i]).abs();
                assert!(diff < 1e-12, "K[{i},{j}] - K[{j},{i}] = {diff}");
            }
        }
    }

    /// DiffusionIntegrator with FnCoeff for spatially-varying kappa.
    #[test]
    fn spatially_varying_kappa() {
        use crate::coefficient::FnCoeff;
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let integ = DiffusionIntegrator { kappa: FnCoeff(|x: &[f64]| 1.0 + x[0]) };
        let mat   = Assembler::assemble_bilinear(&space, &[&integ], 2);
        // Row sums should still be ~0 (∇·(κ∇c) = 0 for constant c, but only exact
        // for constant κ; for varying κ the discrete row-sums may have small nonzero values).
        // At least the matrix should be symmetric.
        let dense = mat.to_dense();
        let n = mat.nrows;
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i * n + j] - dense[j * n + i]).abs();
                assert!(diff < 1e-12, "K[{i},{j}] - K[{j},{i}] = {diff}");
            }
        }
    }
}
