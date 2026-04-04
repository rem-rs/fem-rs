//! Mass-matrix bilinear form integrator.
//!
//! Computes the element contribution to
//!
//! ```text
//! m(u, v) = ∫_Ω ρ u v dx
//! ```

use crate::coefficient::{CoeffCtx, ScalarCoeff};
use crate::integrator::{BilinearIntegrator, QpData};

/// Bilinear integrator for the scalar mass operator `ρ u v`.
///
/// For `ρ = 1` this is the standard L² mass matrix.
///
/// # Example
/// ```
/// # use fem_assembly::standard::MassIntegrator;
/// let integ = MassIntegrator { rho: 1.0 };
/// ```
pub struct MassIntegrator<C: ScalarCoeff = f64> {
    /// Scalar density / reaction coefficient.
    pub rho: C,
}

impl<C: ScalarCoeff> BilinearIntegrator for MassIntegrator<C> {
    /// `M_elem[i,j] += w · ρ(x) · φᵢ · φⱼ`
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let ctx = CoeffCtx::from_qp(
            qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag,
            Some(qp.phi), qp.elem_dofs,
        );
        let w_rho = qp.weight * self.rho.eval(&ctx);
        for i in 0..n {
            for j in 0..n {
                k_elem[i * n + j] += w_rho * qp.phi[i] * qp.phi[j];
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

    /// The mass matrix must be symmetric.
    #[test]
    fn mass_is_symmetric() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let integ = MassIntegrator { rho: 1.0 };
        let mat   = Assembler::assemble_bilinear(&space, &[&integ], 2);
        let dense = mat.to_dense();
        let n = mat.nrows;
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i * n + j] - dense[j * n + i]).abs();
                assert!(diff < 1e-12, "M[{i},{j}] - M[{j},{i}] = {diff}");
            }
        }
    }

    /// The L² norm of the constant function 1 is the domain area (= 1 for unit square).
    /// That is, `1^T M 1 ≈ 1`.
    #[test]
    fn mass_norm_of_one_is_domain_area() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(8);
        let space = H1Space::new(mesh, 1);
        let integ = MassIntegrator { rho: 1.0 };
        let mat   = Assembler::assemble_bilinear(&space, &[&integ], 3);

        let n = mat.nrows;
        let ones = vec![1.0_f64; n];
        let mut y = vec![0.0_f64; n];
        mat.spmv(&ones, &mut y);
        let s: f64 = y.iter().sum();
        assert!((s - 1.0).abs() < 1e-10, "1^T M 1 = {s}, expected ≈ 1");
    }
}
