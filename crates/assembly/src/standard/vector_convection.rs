//! Vector convection bilinear form integrator.
//!
//! Computes the element contribution to
//!
//! ```text
//! a(u, v) = ∫_Ω (w · ∇)u · v dx
//! ```
//!
//! where `w` is a known vector velocity field (from a previous Picard iterate)
//! and `u`, `v` are vector test/trial functions from a `VectorH1Space`.
//!
//! The DOFs are assumed to be interleaved: `[φ₀_x, φ₀_y, φ₁_x, φ₁_y, ...]`.
//! Only same-component pairs contribute (no cross-coupling between u_x and u_y).

use crate::integrator::{BilinearIntegrator, QpData};

/// Bilinear integrator for `∫ (w · ∇u_a) v_a dx` on a `VectorH1Space`.
///
/// `w` is a known velocity DOF vector from the global `VectorH1Space`.
/// At each quadrature point, `w(x)` is evaluated from the element DOFs and
/// scalar basis functions.
///
/// # Example
/// ```rust,ignore
/// let integ = VectorConvectionIntegrator::new(&w_prev, n_scalar);
/// ```
pub struct VectorConvectionIntegrator<'a> {
    /// Global DOF vector for the convecting velocity (length = n_total_dofs).
    w_dofs: &'a [f64],
    /// Number of scalar DOFs per component in the VectorH1Space.
    n_scalar: usize,
}

impl<'a> VectorConvectionIntegrator<'a> {
    /// Create a new integrator.
    ///
    /// - `w_dofs` — the velocity DOF vector from the previous Picard iteration.
    /// - `n_scalar` — number of scalar DOFs per component (`VectorH1Space::n_scalar_dofs()`).
    pub fn new(w_dofs: &'a [f64], n_scalar: usize) -> Self {
        VectorConvectionIntegrator { w_dofs, n_scalar }
    }
}

impl<'a> BilinearIntegrator for VectorConvectionIntegrator<'a> {
    /// `K[(k,a),(l,a)] += w_qp · φ_k · (w_qp · ∇φ_l)`
    ///
    /// where `w_qp` is the convecting velocity evaluated at the quadrature point.
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let dim     = qp.dim;
        let n       = qp.n_dofs;         // total element DOFs = n_nodes * dim
        let n_nodes = n / dim;

        // Evaluate w(x) at this quadrature point from the DOF vector.
        // w_a(x) = Σ_k φ_k(x) · w_dofs[global_dof_of(k, a)]
        // In VectorH1Space global layout: component a, scalar DOF s → global DOF = a * n_scalar + s
        // In element interleaved layout: local DOF (k, a) → elem_dofs[k * dim + a]
        let elem_dofs = qp.elem_dofs
            .expect("VectorConvectionIntegrator requires elem_dofs in QpData");

        let mut w_qp = [0.0_f64; 3];
        for k in 0..n_nodes {
            let phi_k = qp.phi[k];
            for a in 0..dim {
                let global_dof = elem_dofs[k * dim + a] as usize;
                w_qp[a] += phi_k * self.w_dofs[global_dof];
            }
        }

        // Now assemble: K[(k,a),(l,a)] += w · φ_k · (w_qp · ∇φ_l)
        let w = qp.weight;
        for k in 0..n_nodes {
            let phi_k = qp.phi[k];
            for l in 0..n_nodes {
                // w_qp · ∇φ_l = Σ_d w_qp[d] * grad_phys[l * dim + d]
                let mut w_dot_grad_l = 0.0;
                for d in 0..dim {
                    w_dot_grad_l += w_qp[d] * qp.grad_phys[l * dim + d];
                }
                let contrib = w * phi_k * w_dot_grad_l;
                // Same-component only (δ_{ab})
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
    use fem_space::{VectorH1Space, fe_space::FESpace};

    #[test]
    fn vector_convection_is_non_symmetric() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        // Uniform w = (1, 0) via DOF vector
        let n_scalar = space.n_scalar_dofs();
        let mut w = vec![0.0_f64; n];
        for i in 0..n_scalar { w[i] = 1.0; } // w_x = 1, w_y = 0
        let integ = VectorConvectionIntegrator::new(&w, n_scalar);
        let mat = Assembler::assemble_bilinear(&space, &[&integ], 3);
        let dense = mat.to_dense();
        let nn = mat.nrows;
        let mut has_asymmetry = false;
        for i in 0..nn {
            for j in 0..nn {
                if (dense[i * nn + j] - dense[j * nn + i]).abs() > 1e-12 {
                    has_asymmetry = true;
                    break;
                }
            }
            if has_asymmetry { break; }
        }
        assert!(has_asymmetry, "vector convection should be non-symmetric");
    }

    #[test]
    fn vector_convection_zero_velocity_is_zero_matrix() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        let w = vec![0.0_f64; n];
        let n_scalar = space.n_scalar_dofs();
        let integ = VectorConvectionIntegrator::new(&w, n_scalar);
        let mat = Assembler::assemble_bilinear(&space, &[&integ], 3);
        let dense = mat.to_dense();
        let max_val: f64 = dense.iter().map(|&v| v.abs()).fold(0.0, f64::max);
        assert!(max_val < 1e-14, "zero velocity should give zero matrix, got max={max_val}");
    }
}
