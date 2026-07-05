//! Vector H¹ mass-matrix bilinear form integrator.
//!
//! Computes the element contribution to
//!
//! ```text
//! m(u, v) = ∫_Ω κ u · v dx = κ Σᵢ ∫ uᵢ · vᵢ dx
//! ```
//!
//! Acts on `VectorH1Space` with interleaved DOFs `[u_x(0), u_y(0), …]`.

scalar_bilinear_integrator!(VectorH1MassIntegrator, kappa,
    "Bilinear integrator for the vector mass operator `κ u · v`.

Each component is treated independently (no cross-coupling between
u_x and u_y), just like [`VectorDiffusionIntegrator`].

Unlike [`MassIntegrator`] (which only works for scalar H¹ spaces),
this correctly handles `VectorH1Space` by dividing the total DOF
count by `dim` to obtain the scalar per-component DOF count.

# Example
```rust,ignore
use fem_assembly::standard::VectorH1MassIntegrator;
let integ = VectorH1MassIntegrator { kappa: 1.0 };
```", |qp, k_elem, n, w| {
    let dim     = qp.dim;
    let n_nodes = n / dim;
    for k in 0..n_nodes {
        for l in 0..n_nodes {
            let contrib = w * qp.phi[k] * qp.phi[l];
            for a in 0..dim {
                let row = k * dim + a;
                let col = l * dim + a;
                k_elem[row * n + col] += contrib;
            }
        }
    }
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::Assembler;
    use fem_mesh::SimplexMesh;
    use fem_space::VectorH1Space;

    /// The vector mass matrix must be symmetric.
    #[test]
    fn vector_h1_mass_is_symmetric() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let integ = VectorH1MassIntegrator { kappa: 1.0 };
        let mat   = Assembler::assemble_bilinear(&space, &[&integ], 3);
        let dense = mat.to_dense();
        let n = mat.nrows;
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i * n + j] - dense[j * n + i]).abs();
                assert!(diff < 1e-12, "M[{i},{j}] - M[{j},{i}] = {diff}");
            }
        }
    }
}
