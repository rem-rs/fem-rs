//! Vector diffusion (vector Laplacian) bilinear form integrator.
//!
//! Computes the element contribution to
//!
//! ```text
//! a(u, v) = ∫_Ω κ ∇uᵢ · ∇vᵢ dx   (summed over components i)
//! ```
//!
//! This is the component-wise Laplacian on a vector field, acting on
//! `VectorH1Space` with interleaved DOFs `[u_x(0), u_y(0), …]`.

use crate::coefficient::{CoeffCtx, ScalarCoeff};
use crate::integrator::{BilinearIntegrator, QpData};

/// Bilinear integrator for the vector Laplacian `κ Σᵢ ∇uᵢ · ∇vᵢ`.
///
/// Unlike [`ElasticityIntegrator`], this treats each component independently
/// (no cross-coupling between u_x and u_y).
///
/// # Example
/// ```rust,ignore
/// use fem_assembly::standard::VectorDiffusionIntegrator;
/// let integ = VectorDiffusionIntegrator { kappa: 1.0 };
/// ```
pub struct VectorDiffusionIntegrator<C: ScalarCoeff = f64> {
    /// Scalar diffusivity coefficient.
    pub kappa: C,
}

impl<C: ScalarCoeff> BilinearIntegrator for VectorDiffusionIntegrator<C> {
    /// `K[(k,a),(l,b)] += δ_{ab} · w · κ · (∇φ_k · ∇φ_l)`
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
                // ∇φ_k · ∇φ_l  (scalar basis gradient dot product)
                let mut dot = 0.0;
                for d in 0..dim {
                    dot += qp.grad_phys[k * dim + d] * qp.grad_phys[l * dim + d];
                }
                let contrib = w_k * dot;
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
    fn vector_diffusion_is_symmetric() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let integ = VectorDiffusionIntegrator { kappa: 1.0 };
        let mat   = Assembler::assemble_bilinear(&space, &[&integ], 3);
        let dense = mat.to_dense();
        let n = mat.nrows;
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i * n + j] - dense[j * n + i]).abs();
                assert!(diff < 1e-12, "K[{i},{j}] - K[{j},{i}] = {diff}");
            }
        }
    }

    #[test]
    fn vector_diffusion_row_sums_zero() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let integ = VectorDiffusionIntegrator { kappa: 1.0 };
        let mat   = Assembler::assemble_bilinear(&space, &[&integ], 3);
        let dense = mat.to_dense();
        let n = mat.nrows;
        for row in 0..n {
            let s: f64 = (0..n).map(|c| dense[row * n + c]).sum();
            assert!(s.abs() < 1e-10, "row {row} sum = {s}");
        }
    }
}
