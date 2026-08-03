//! Domain-source linear form integrator.
//!
//! Computes the element contribution to
//!
//! ```text
//! F(v) = ∫_Ω f(x) v dx
//! ```

use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff};

domain_linear_closure!(DomainSourceIntegrator,
    "Linear integrator for the domain source term `∫ f(x) v dx`.

The source function `f` may depend on the physical coordinates `x`.

# Example
```
# use fem_assembly::standard::DomainSourceIntegrator;
// f(x, y) = 2π² sin(πx) sin(πy)
let integ = DomainSourceIntegrator::new(|x| {
    use std::f64::consts::PI;
    2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
});
```", |qp, f_elem, n, w| {
    for i in 0..n {
        f_elem[i] += w * qp.phi[i];
    }
});

/// Domain source integrator with an arbitrary [`ScalarCoeff`] coefficient.
///
/// MFEM equivalent: `DomainLFIntegrator(Coefficient)` — computes
/// `F(v) = ∫_Ω f(x) v dx` where `f` is evaluated through a [`CoeffCtx`]
/// (GridFunction values via `phi`/`elem_dofs`, element tags, …).
pub struct DomainSourceIntegratorCoeff<C: ScalarCoeff> {
    /// The source coefficient `f`.
    pub f: C,
}

impl<C: ScalarCoeff> DomainSourceIntegratorCoeff<C> {
    /// Create the integrator from a coefficient.
    pub fn new(f: C) -> Self {
        DomainSourceIntegratorCoeff { f }
    }
}

impl<C: ScalarCoeff> LinearIntegrator for DomainSourceIntegratorCoeff<C> {
    fn add_to_element_vector(&self, qp: &QpData<'_>, f_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let ctx = CoeffCtx::from_qp(
            qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag,
            Some(qp.phi), qp.elem_dofs,
        );
        let w = qp.weight * self.f.eval(&ctx);
        for i in 0..n {
            f_elem[i] += w * qp.phi[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::Assembler;
    use fem_mesh::Mesh;
    use fem_space::H1Space;

    /// ∫_Ω 1 dx over the unit square should be ≈ 1.
    #[test]
    fn source_constant_one_integrates_to_area() {
        let mesh  = Mesh::<2>::unit_square_tri(8);
        let space = H1Space::new(mesh, 1);
        let integ = DomainSourceIntegrator::new(|_x| 1.0);
        let rhs   = Assembler::assemble_linear(&space, &[&integ], 3);
        let s: f64 = rhs.iter().sum();
        assert!((s - 1.0).abs() < 1e-10, "∫1 dx = {s}, expected ≈ 1");
    }
}
