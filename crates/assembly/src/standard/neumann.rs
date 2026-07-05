//! Neumann boundary linear form integrator.
//!
//! Computes the boundary contribution to the load vector:
//!
//! ```text
//! F_Γ(v) = ∫_Γ g(x) v ds
//! ```
//!
//! where `g` is the prescribed outward normal flux (or any boundary data).

boundary_linear_closure!(NeumannIntegrator,
    "Linear integrator for a Neumann (natural) boundary condition `∫_Γ g(x) v ds`.

`g` may depend on the physical position `x` and optionally the outward unit normal.

# Example
```
# use fem_assembly::standard::NeumannIntegrator;
// Constant flux g = 1.0 on the boundary.
let integ = NeumannIntegrator::new(|_x, _n| 1.0);
```", |qp, f_face, n, w| {
    for i in 0..n {
        f_face[i] += w * qp.phi[i];
    }
});
