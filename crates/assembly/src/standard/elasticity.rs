//! Linear elasticity bilinear form integrator.
//!
//! Computes the element contribution to the symmetric bilinear form
//!
//! ```text
//! a(u, v) = ∫_Ω σ(u) : ε(v) dx
//!          = ∫_Ω [ λ (∇·u)(∇·v) + 2μ ε(u):ε(v) ] dx
//! ```
//!
//! where `ε(u) = ½(∇u + (∇u)ᵀ)` is the symmetric strain tensor and
//! `σ(u) = λ tr(ε)I + 2με` is the Cauchy stress (Lamé parameters λ, μ).
//!
//! # DOF convention
//! Element DOFs must be **interleaved** (node-major):
//! `[u_x(0), u_y(0), u_x(1), u_y(1), …]` as produced by [`VectorH1Space`].

use crate::coefficient::{CoeffCtx, ScalarCoeff};
use crate::integrator::{BilinearIntegrator, QpData};

/// Bilinear integrator for the isotropic linear elasticity operator.
///
/// # Parameters
/// - `lambda`: first Lamé parameter (related to bulk modulus)
/// - `mu`:     shear modulus (second Lamé parameter)
///
/// # Relation to Young's modulus and Poisson ratio
/// - `E  = mu * (3*lambda + 2*mu) / (lambda + mu)`
/// - `nu = lambda / (2*(lambda + mu))`
pub struct ElasticityIntegrator<C1: ScalarCoeff = f64, C2: ScalarCoeff = f64> {
    /// First Lamé parameter.
    pub lambda: C1,
    /// Second Lamé parameter (shear modulus).
    pub mu: C2,
}

impl<C1: ScalarCoeff, C2: ScalarCoeff> BilinearIntegrator for ElasticityIntegrator<C1, C2> {
    /// Accumulates the element elasticity matrix assuming interleaved DOFs.
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let dim   = qp.dim;
        let n     = qp.n_dofs;              // total DOFs (n_nodes * dim)
        let n_nodes = n / dim;
        let w = qp.weight;
        let ctx = CoeffCtx::from_qp(
            qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag,
            Some(qp.phi), qp.elem_dofs,
        );
        let lam = self.lambda.eval(&ctx);
        let mu  = self.mu.eval(&ctx);

        for k in 0..n_nodes {
            for a in 0..dim {
                let row = k * dim + a;
                let grad_k: Vec<f64> = (0..dim).map(|d| qp.grad_phys[k * dim + d]).collect();
                let div_ka = grad_k[a];

                for l in 0..n_nodes {
                    for b in 0..dim {
                        let col = l * dim + b;
                        let grad_l: Vec<f64> = (0..dim).map(|d| qp.grad_phys[l * dim + d]).collect();
                        let div_lb = grad_l[b];

                        let vol = lam * div_ka * div_lb;

                        let mut shear = 0.0;
                        for i in 0..dim {
                            for j in 0..dim {
                                let eps_ka_ij =
                                    0.5 * (if j == a { grad_k[i] } else { 0.0 }
                                         + if i == a { grad_k[j] } else { 0.0 });
                                let eps_lb_ij =
                                    0.5 * (if j == b { grad_l[i] } else { 0.0 }
                                         + if i == b { grad_l[j] } else { 0.0 });
                                shear += eps_ka_ij * eps_lb_ij;
                            }
                        }

                        k_elem[row * n + col] += w * (vol + 2.0 * mu * shear);
                    }
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

    /// The assembled elasticity matrix should be symmetric.
    #[test]
    fn elasticity_matrix_symmetric() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let integ = ElasticityIntegrator { lambda: 1.0, mu: 1.0 };
        let mat   = Assembler::assemble_bilinear(&space, &[&integ], 3);
        let dense = mat.to_dense();
        let n = mat.nrows;
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i * n + j] - dense[j * n + i]).abs();
                assert!(diff < 1e-11, "K[{i},{j}] - K[{j},{i}] = {diff}");
            }
        }
    }

    /// Patch test: constant strain state (rigid body) is in the kernel.
    /// Row sums of the elasticity matrix should be ≈ 0 for any component.
    #[test]
    fn elasticity_row_sums_zero() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(2);
        let space = VectorH1Space::new(mesh, 1, 2);
        let integ = ElasticityIntegrator { lambda: 1.0, mu: 0.5 };
        let mat   = Assembler::assemble_bilinear(&space, &[&integ], 3);
        let dense = mat.to_dense();
        let n = mat.nrows;
        for row in 0..n {
            let s: f64 = (0..n).map(|c| dense[row * n + c]).sum();
            assert!(s.abs() < 1e-10, "row {row} sum = {s}");
        }
    }
}
