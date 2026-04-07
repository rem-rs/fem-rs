//! Sum integrator wrapper.
//!
//! Wraps multiple [`BilinearIntegrator`]s and applies them all to the same
//! element matrix.  This is equivalent to what the assembler already does
//! when given a slice of integrators, but packaged as a single integrator
//! for use in contexts that accept one integrator (e.g. [`TransposeIntegrator`]).

use crate::integrator::{BilinearIntegrator, QpData};

/// Applies multiple bilinear integrators in sequence, summing their contributions.
///
/// # Example
/// ```rust,ignore
/// use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator, SumIntegrator};
/// let integ = SumIntegrator(vec![
///     Box::new(DiffusionIntegrator { kappa: 1.0 }),
///     Box::new(MassIntegrator { rho: 1.0 }),
/// ]);
/// ```
pub struct SumIntegrator(pub Vec<Box<dyn BilinearIntegrator>>);

impl BilinearIntegrator for SumIntegrator {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        for integ in &self.0 {
            integ.add_to_element_matrix(qp, k_elem);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::Assembler;
    use crate::standard::{DiffusionIntegrator, MassIntegrator};
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;

    /// SumIntegrator(D, M) should equal assembling with &[&D, &M].
    #[test]
    fn sum_equals_separate_assembly() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);

        let d = DiffusionIntegrator { kappa: 1.0 };
        let m = MassIntegrator { rho: 1.0 };

        // Assembled with two integrators (the standard way).
        let mat_sep = Assembler::assemble_bilinear(&space, &[&d, &m], 3);

        // Assembled with SumIntegrator.
        let sum = SumIntegrator(vec![
            Box::new(DiffusionIntegrator { kappa: 1.0 }),
            Box::new(MassIntegrator { rho: 1.0 }),
        ]);
        let mat_sum = Assembler::assemble_bilinear(&space, &[&sum], 3);

        let ds = mat_sep.to_dense();
        let du = mat_sum.to_dense();
        let n = mat_sep.nrows;
        for i in 0..n {
            for j in 0..n {
                let diff = (ds[i * n + j] - du[i * n + j]).abs();
                assert!(diff < 1e-12, "diff at ({i},{j}) = {diff}");
            }
        }
    }
}
