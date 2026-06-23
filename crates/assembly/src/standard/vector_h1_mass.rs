//! Vector H¹ mass-matrix bilinear form integrator.
//!
//! Computes the element contribution to
//!
//! ```text
//! m(u, v) = ∫_Ω κ u · v dx = κ Σᵢ ∫ uᵢ · vᵢ dx
//! ```
//!
//! Acts on `VectorH1Space` with interleaved DOFs `[u_x(0), u_y(0), …]`.

use crate::coefficient::{CoeffCtx, ScalarCoeff};
use crate::integrator::{BilinearIntegrator, QpData};

/// Bilinear integrator for the vector mass operator `κ u · v`.
///
/// Each component is treated independently (no cross-coupling between
/// u_x and u_y), just like [`VectorDiffusionIntegrator`].
///
/// Unlike [`MassIntegrator`] (which only works for scalar H¹ spaces),
/// this correctly handles `VectorH1Space` by dividing the total DOF
/// count by `dim` to obtain the scalar per-component DOF count.
///
/// # Example
/// ```rust,ignore
/// use fem_assembly::standard::VectorH1MassIntegrator;
/// let integ = VectorH1MassIntegrator { kappa: 1.0 };
/// ```
pub struct VectorH1MassIntegrator<C: ScalarCoeff = f64> {
    /// Scalar reaction / mass coefficient.
    pub kappa: C,
}

impl<C: ScalarCoeff> BilinearIntegrator for VectorH1MassIntegrator<C> {
    /// `M[(k,a),(l,b)] += δ_{ab} · w · κ · φ_k · φ_l`
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let dim     = qp.dim;
        let n       = qp.n_dofs;         // total DOFs = n_nodes * dim
        let n_nodes = n / dim;
        let ctx = CoeffCtx::from_qp(
            qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag,
            Some(qp.phi), qp.elem_dofs,
        );
        let w_k = qp.weight * self.kappa.eval(&ctx);

        for k in 0..n_nodes {
            for l in 0..n_nodes {
                let contrib = w_k * qp.phi[k] * qp.phi[l];
                // Only same-component pairs (a == b) contribute.
                for a in 0..dim {
                    let row = k * dim + a;
                    let col = l * dim + a;
                    k_elem[row * n + col] += contrib;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::Assembler;
    use fem_mesh::SimplexMesh;
    use fem_space::VectorH1Space;

    #[test]
    fn vector_h1_mass_is_symmetric() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 2, 2);
        let integ = VectorH1MassIntegrator { kappa: 1.0 };
        let mat   = Assembler::assemble_bilinear(&space, &[&integ], 5);
        let dense = mat.to_dense();
        let n = mat.nrows;
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i * n + j] - dense[j * n + i]).abs();
                assert!(diff < 1e-12, "M[{i},{j}] - M[{j},{i}] = {diff}");
            }
        }
    }

    #[test]
    fn vector_h1_mass_diagonal_positive() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 2, 2);
        let integ = VectorH1MassIntegrator { kappa: 1.0 };
        let mat   = Assembler::assemble_bilinear(&space, &[&integ], 5);
        for i in 0..mat.nrows {
            let diag = mat.get(i, i);
            assert!(diag > 0.0, "M[{i},{i}] = {diag} ≤ 0");
        }
    }

    #[test]
    fn vector_h1_mass_with_diffusion_is_spd() {
        // Combined with VectorDiffusionIntegrator, the result should be SPD.
        use crate::standard::VectorDiffusionIntegrator;
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 2, 2);
        let diff  = VectorDiffusionIntegrator { kappa: 0.5 };
        let mass  = VectorH1MassIntegrator { kappa: 1.5 };
        let mat   = Assembler::assemble_bilinear(&space, &[&diff, &mass], 5);
        let dense = mat.to_dense();
        let n = mat.nrows;
        // Check symmetry
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i * n + j] - dense[j * n + i]).abs();
                assert!(diff < 1e-12, "K[{i},{j}] - K[{j},{i}] = {diff}");
            }
        }
        // Check positive diagonal
        for i in 0..n {
            assert!(dense[i * n + i] > 0.0, "K[{i},{i}] ≤ 0");
        }
    }
}
