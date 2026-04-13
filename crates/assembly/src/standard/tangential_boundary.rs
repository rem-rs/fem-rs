//! Tangential boundary linear form integrator for H(curl) spaces.
//!
//! Computes boundary load contributions of the form
//!
//! ```text
//! F(v) = ∫_Γ g · (n×v) dS    (3-D)
//! F(v) = ∫_Γ g (n×v) ds      (2-D)
//! ```
//!
//! where:
//! - `n` is the outward unit normal,
//! - `v` is an H(curl) test function,
//! - `g` is prescribed tangential boundary data.
//!
//! This is the linear-form counterpart of `TangentialMassIntegrator` and is
//! useful for:
//! - Silver-Muller / impedance boundary-condition right-hand sides,
//! - non-homogeneous tangential trace loading,
//! - boundary-driven Maxwell test problems.

use crate::coefficient::CoeffCtx;
use crate::vector_boundary::{VectorBdQpData, VectorBoundaryLinearIntegrator};

/// Tangential boundary load integrator for H(curl) spaces.
///
/// The closure receives `(x_phys, normal, out)`.
///
/// - In 2-D, write the scalar load into `out[0]` for
///   `∫_Γ g (n×v) ds`.
/// - In 3-D, write the 3-vector load into `out[0..3]` for
///   `∫_Γ g · (n×v) dS`.
///
/// # Example (2-D)
/// ```rust,ignore
/// use fem_assembly::standard::TangentialTraceLFIntegrator;
/// let integ = TangentialTraceLFIntegrator::new(|_x, _n, out| {
///     out[0] = 1.0;
/// });
/// ```
///
/// # Example (3-D)
/// ```rust,ignore
/// use fem_assembly::standard::TangentialTraceLFIntegrator;
/// let integ = TangentialTraceLFIntegrator::new(|x, _n, out| {
///     out[0] = x[1];
///     out[1] = -x[0];
///     out[2] = 0.0;
/// });
/// ```
pub struct TangentialTraceLFIntegrator<F>
where
    F: Fn(&CoeffCtx<'_>, &[f64], &mut [f64]) + Send + Sync,
{
    g: F,
}

impl<F> TangentialTraceLFIntegrator<F>
where
    F: Fn(&CoeffCtx<'_>, &[f64], &mut [f64]) + Send + Sync,
{
    /// Create a new tangential boundary load integrator.
    pub fn new(g: F) -> Self {
        TangentialTraceLFIntegrator { g }
    }
}

impl<F> VectorBoundaryLinearIntegrator for TangentialTraceLFIntegrator<F>
where
    F: Fn(&CoeffCtx<'_>, &[f64], &mut [f64]) + Send + Sync,
{
    fn add_to_face_vector(&self, qp: &VectorBdQpData<'_>, f_face: &mut [f64]) {
        let ctx = CoeffCtx::from_qp(qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag, None, None);
        let n_dofs = qp.n_dofs;

        if qp.dim == 2 {
            let mut g = [0.0_f64; 1];
            (self.g)(&ctx, qp.normal, &mut g);
            let gval = g[0];

            let nx = qp.normal[0];
            let ny = qp.normal[1];
            for i in 0..n_dofs {
                let phi_x = qp.phi_vec[i * 2];
                let phi_y = qp.phi_vec[i * 2 + 1];
                let nx_phi = phi_x * ny - phi_y * nx;
                f_face[i] += qp.weight * gval * nx_phi;
            }
        } else {
            let mut g = [0.0_f64; 3];
            (self.g)(&ctx, qp.normal, &mut g);
            let [nx, ny, nz] = [qp.normal[0], qp.normal[1], qp.normal[2]];

            for i in 0..n_dofs {
                let [phi_x, phi_y, phi_z] = [
                    qp.phi_vec[i * 3],
                    qp.phi_vec[i * 3 + 1],
                    qp.phi_vec[i * 3 + 2],
                ];
                let cross = [
                    ny * phi_z - nz * phi_y,
                    nz * phi_x - nx * phi_z,
                    nx * phi_y - ny * phi_x,
                ];
                let dot = g[0] * cross[0] + g[1] * cross[1] + g[2] * cross[2];
                f_face[i] += qp.weight * dot;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_boundary::VectorBoundaryAssembler;
    use fem_mesh::SimplexMesh;
    use fem_space::{EdgeKey, HCurlSpace};

    #[test]
    fn tangential_trace_zero_load_gives_zero_rhs() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let space = HCurlSpace::new(mesh, 1);
        let integ = TangentialTraceLFIntegrator::new(|_ctx, _n, out| {
            out[0] = 0.0;
        });

        let rhs = VectorBoundaryAssembler::assemble_boundary_linear(&space, &[&integ], &[1, 2, 3, 4], 4);
        assert!(rhs.iter().all(|v| v.abs() < 1e-14), "expected zero rhs, got {rhs:?}");
    }

    #[test]
    fn tangential_trace_unit_load_on_bottom_edge_matches_edge_moment() {
        let mesh = SimplexMesh::<2>::unit_square_tri(1);
        let space = HCurlSpace::new(mesh, 1);
        let integ = TangentialTraceLFIntegrator::new(|_ctx, _n, out| {
            out[0] = 1.0;
        });

        let rhs = VectorBoundaryAssembler::assemble_boundary_linear(&space, &[&integ], &[1], 4);
        let bottom = space.edge_dof(EdgeKey::new(0, 1)).expect("missing bottom edge dof") as usize;

        for (i, &val) in rhs.iter().enumerate() {
            if i == bottom {
                assert!((val + 1.0).abs() < 1e-12, "bottom edge load = {val}, expected -1");
            } else {
                assert!(val.abs() < 1e-12, "unexpected load at dof {i}: {val}");
            }
        }
    }
}