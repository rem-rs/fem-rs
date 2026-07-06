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

scalar_bilinear_integrator!(VectorDiffusionIntegrator, kappa,
    "Bilinear integrator for the vector Laplacian `κ Σᵢ ∇uᵢ · ∇vᵢ`.

Unlike [`ElasticityIntegrator`], this treats each component independently
(no cross-coupling between u_x and u_y).

# Example
```rust,ignore
use fem_assembly::standard::VectorDiffusionIntegrator;
let integ = VectorDiffusionIntegrator { kappa: 1.0 };
```", |qp, k_elem, n, w| {
    let dim     = qp.dim;
    let n_nodes = n / dim;
    // Outer product over spatial dimensions: same-component coupling.
    for d in 0..dim {
        for k in 0..n_nodes {
            let g_kd = qp.grad_phys[k * dim + d];
            for l in 0..n_nodes {
                let contrib = w * g_kd * qp.grad_phys[l * dim + d];
                for a in 0..dim {
                    let row = k * dim + a;
                    let col = l * dim + a;
                    k_elem[row * n + col] += contrib;
                }
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

    /// The vector diffusion matrix should be symmetric.
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
}
