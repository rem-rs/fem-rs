//! Domain-source linear form integrator.
//!
//! Computes the element contribution to
//!
//! ```text
//! F(v) = ∫_Ω f(x) v dx
//! ```

use crate::integrator::{LinearIntegrator, QpData};

/// Linear integrator for the domain source term `∫ f(x) v dx`.
///
/// The source function `f` may depend on the physical coordinates `x`.
///
/// # Example
/// ```
/// # use fem_assembly::standard::DomainSourceIntegrator;
/// // f(x, y) = 2π² sin(πx) sin(πy)
/// let integ = DomainSourceIntegrator::new(|x| {
///     use std::f64::consts::PI;
///     2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
/// });
/// ```
pub struct DomainSourceIntegrator<F>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    f: F,
}

impl<F> DomainSourceIntegrator<F>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    /// Create a new source integrator with the given forcing function.
    pub fn new(f: F) -> Self { DomainSourceIntegrator { f } }
}

impl<F> LinearIntegrator for DomainSourceIntegrator<F>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    /// `f_elem[i] += w · f(x) · φᵢ`
    fn add_to_element_vector(&self, qp: &QpData<'_>, f_elem: &mut [f64]) {
        let fval = (self.f)(qp.x_phys);
        let w_f  = qp.weight * fval;
        for i in 0..qp.n_dofs {
            f_elem[i] += w_f * qp.phi[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::Assembler;
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;

    /// ∫_Ω 1 dx over the unit square should be ≈ 1.
    #[test]
    fn source_constant_one_integrates_to_area() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(8);
        let space = H1Space::new(mesh, 1);
        let integ = DomainSourceIntegrator::new(|_x| 1.0);
        let rhs   = Assembler::assemble_linear(&space, &[&integ], 3);
        let s: f64 = rhs.iter().sum();
        assert!((s - 1.0).abs() < 1e-10, "∫1 dx = {s}, expected ≈ 1");
    }
}
