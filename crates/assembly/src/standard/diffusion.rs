//! Diffusion (Laplacian) bilinear form integrator.
//!
//! Computes the element contribution to
//!
//! ```text
//! a(u, v) = ∫_Ω κ ∇u · ∇v dx
//! ```

scalar_bilinear_integrator!(DiffusionIntegrator, kappa,
    "Bilinear integrator for the scalar diffusion operator `κ ∇u · ∇v`.

For `κ = 1` this is the standard Laplacian stiffness matrix.

The coefficient `κ` is generic over [`ScalarCoeff`], with `f64` as the
default for full backwards compatibility:

```rust,ignore
// Constant (unchanged):
let integ = DiffusionIntegrator { kappa: 1.0 };

// Spatially varying:
let integ = DiffusionIntegrator { kappa: FnCoeff(|x: &[f64]| 1.0 + x[0]) };

// Piecewise constant per material:
let integ = DiffusionIntegrator { kappa: PWConstCoeff::new([(1, 1.0), (2, 100.0)]) };
```", |qp, k_elem, n, w| {
    // MFEM `AddMult_a_AAt` (densemat.cpp): for each pair (i,j) first
    // accumulate d = Σ_k A(i,k)·A(j,k) with plain multiply-then-add (NOT FMA;
    // serial g++ builds do not fuse), then scale by a once:
    //   AAt(i,j) += (d *= a);  AAt(j,i) += d;   (i > j)
    //   AAt(i,i) += a * d;
    // This bit-matches the reference integration (a per-point FMA loop with
    // `a` folded into each term differs by ~1 ulp on shared entries).
    let dim = qp.dim;
    for i in 0..n {
        for j in 0..i {
            let mut d = 0.0_f64;
            for k in 0..dim {
                d += qp.grad_phys[i * dim + k] * qp.grad_phys[j * dim + k];
            }
            let ad = w * d;
            k_elem[i * n + j] += ad;
            k_elem[j * n + i] += ad;
        }
        let mut d = 0.0_f64;
        for k in 0..dim {
            d += qp.grad_phys[i * dim + k] * qp.grad_phys[i * dim + k];
        }
        k_elem[i * n + i] += w * d;
    }
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::Assembler;
    use fem_mesh::Mesh;
    use fem_space::H1Space;

    /// The stiffness matrix from a DiffusionIntegrator on the reference triangle
    /// (one element) should be symmetric positive semi-definite with a known
    /// row-sum of zero (constant functions are in the kernel of ∇).
    #[test]
    fn stiffness_row_sum_zero_single_element() {
        let mesh  = Mesh::<2>::unit_square_tri(1);
        let space = H1Space::new(mesh, 1);
        let integ = DiffusionIntegrator { kappa: 1.0 };
        let mat   = Assembler::assemble_bilinear(&space, &[&integ], 2);

        let dense = mat.to_dense();
        for row in 0..mat.nrows {
            let s: f64 = (0..mat.ncols).map(|c| dense[row * mat.ncols + c]).sum();
            assert!(s.abs() < 1e-12, "row {row} sum = {s}");
        }
    }

    /// Symmetry check: K[i,j] == K[j,i].
    #[test]
    fn stiffness_is_symmetric() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
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
        use crate::postproc::coefficient::FnCoeff;
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let integ = DiffusionIntegrator { kappa: FnCoeff(|x: &[f64]| 1.0 + x[0]) };
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
}
