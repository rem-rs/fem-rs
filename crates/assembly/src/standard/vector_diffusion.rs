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

MFEM twin: `VectorDiffusionIntegrator::AssembleElementMatrix`
(`fem/bilininteg.cpp:3049`) computes the `nd × nd` scalar matrix
`pelmat = w·dshapedxt·dshapedxtᵀ` **once** per quadrature point (the
`AddMult_a_AAt` bit pattern, identical to [`DiffusionIntegrator`]) and adds
that same matrix to each of the `vdim` diagonal blocks
(`elmat.AddMatrix(pelmat, dof·k, dof·k)`).  This kernel reproduces that
shape: one accumulation per node pair, scattered to every same-component
block — so the diagonal blocks are bit-identical to the scalar
`DiffusionIntegrator` on the same space.

# Example
```rust,ignore
use fem_assembly::standard::VectorDiffusionIntegrator;
let integ = VectorDiffusionIntegrator { kappa: 1.0 };
```", |qp, k_elem, n, w| {
    let dim     = qp.dim;
    let n_nodes = n / dim;
    // MFEM `AddMult_a_AAt` pair accumulation (see `diffusion.rs`), computed
    // once per node pair and then added to each diagonal (same-component)
    // block — local dof (node k, component a) = `k·dim + a`, the layout of
    // `VectorH1Space`'s interleaved element DOF list.
    for k in 0..n_nodes {
        for l in 0..k {
            let mut d = 0.0_f64;
            for c in 0..dim {
                d += qp.grad_phys[k * dim + c] * qp.grad_phys[l * dim + c];
            }
            let ad = w * d;
            for a in 0..dim {
                let row = k * dim + a;
                let col = l * dim + a;
                k_elem[row * n + col] += ad;
                k_elem[col * n + row] += ad;
            }
        }
        let mut d = 0.0_f64;
        for c in 0..dim {
            d += qp.grad_phys[k * dim + c] * qp.grad_phys[k * dim + c];
        }
        for a in 0..dim {
            let row = k * dim + a;
            k_elem[row * n + row] += w * d;
        }
    }
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::Assembler;
    use fem_mesh::Mesh;
    use fem_space::VectorH1Space;

    /// The vector diffusion matrix should be symmetric.
    #[test]
    fn vector_diffusion_is_symmetric() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
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
