//! Mass-matrix bilinear form integrator.
//!
//! Computes the element contribution to
//!
//! ```text
//! m(u, v) = ∫_Ω ρ u v dx
//! ```

scalar_bilinear_integrator_phys!(MassIntegrator, rho,
    "Bilinear integrator for the scalar mass operator `ρ u v`.

For `ρ = 1` this is the standard L² mass matrix.

# Example
```
# use fem_assembly::standard::MassIntegrator;
let integ = MassIntegrator { rho: 1.0 };
```", |qp, k_elem, n, w| {
    // MFEM AddMult_a_VVt: avi = a*v(i) once, the symmetric (j,i) entry
    // shares the SAME avivj product (bit-identical).
    for i in 0..n {
        let avi = w * qp.phi[i];
        for j in 0..i {
            let avivj = avi * qp.phi[j];
            k_elem[i * n + j] += avivj;
            k_elem[j * n + i] += avivj;
        }
        k_elem[i * n + i] += avi * qp.phi[i];
    }
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::Assembler;
    use fem_mesh::Mesh;
    use fem_space::H1Space;

    /// The mass matrix must be symmetric.
    #[test]
    fn mass_is_symmetric() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
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
        let mesh  = Mesh::<2>::unit_square_tri(8);
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
