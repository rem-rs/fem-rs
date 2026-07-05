//! Boundary mass bilinear form integrator.
//!
//! Computes the boundary contribution to
//!
//! ```text
//! a_Γ(u, v) = ∫_Γ α u v ds
//! ```
//!
//! Used for Robin boundary conditions, penalty methods, and absorbing BCs.

boundary_scalar_bilinear!(BoundaryMassIntegrator, alpha,
    "Bilinear integrator for the boundary mass operator `α u v` on Γ.

# Example
```rust,ignore
use fem_assembly::standard::BoundaryMassIntegrator;
let integ = BoundaryMassIntegrator { alpha: 1.0 };
```", |qp, k_face, n, w| {
    for i in 0..n {
        for j in 0..n {
            k_face[i * n + j] += w * qp.phi[i] * qp.phi[j];
        }
    }
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::{Assembler, face_dofs_p1};
    use fem_mesh::SimplexMesh;
    use fem_space::{H1Space, fe_space::FESpace};

    /// 1^T M_Γ 1 should equal the boundary length (= 4 for unit square).
    #[test]
    fn boundary_mass_integral_equals_perimeter() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let n = space.n_dofs();
        let integ = BoundaryMassIntegrator { alpha: 1.0 };
        let tags: Vec<i32> = (1..=4).collect();

        let mat = Assembler::assemble_boundary_bilinear(
            n, space.mesh(), &face_dofs_p1(space.mesh()), 1,
            &[&integ], &tags, 3,
        );

        let ones = vec![1.0_f64; n];
        let mut y = vec![0.0_f64; n];
        mat.spmv(&ones, &mut y);
        let total: f64 = y.iter().sum();
        assert!((total - 4.0).abs() < 1e-10, "1^T M_Γ 1 = {total}, expected 4.0");
    }

    /// Boundary mass matrix should be symmetric.
    #[test]
    fn boundary_mass_is_symmetric() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let n = space.n_dofs();
        let integ = BoundaryMassIntegrator { alpha: 1.0 };
        let tags: Vec<i32> = (1..=4).collect();

        let mat = Assembler::assemble_boundary_bilinear(
            n, space.mesh(), &face_dofs_p1(space.mesh()), 1,
            &[&integ], &tags, 3,
        );

        let dense = mat.to_dense();
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i * n + j] - dense[j * n + i]).abs();
                assert!(diff < 1e-12, "M[{i},{j}] - M[{j},{i}] = {diff}");
            }
        }
    }
}
