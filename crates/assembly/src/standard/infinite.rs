//! Infinite-element integrator for unbounded-domain truncation.
//!
//! Implements a simple mapped infinite element: elements tagged as "infinite"
//! receive a geometric decay weighting that approximates the Sommerfeld
//! radiation condition for acoustic/electromagnetic wave problems.
//!
//! The decay applied is `w(r) = (r₀ / r)^p` where:
//! - `r` is the distance from the origin/pole to the quadrature point
//! - `r₀` is a reference radius (the start of the infinite element layer)
//! - `p` is the decay exponent (p = dim for standard wave problems)
//!
//! # Usage
//! ```rust,ignore
//! use fem_assembly::standard::{InfiniteDomainIntegrator, DiffusionIntegrator};
//!
//! // Standard diffusion for most elements, decay-weighted for infinite elements
//! let inner = DiffusionIntegrator { kappa: 1.0 };
//! let inf = InfiniteDomainIntegrator::new(inner, [0.0, 0.0], 1.0, None);
//! let mat = Assembler::assemble_bilinear(&space, &[&inf], 3);
//! ```

use crate::integrator::{BilinearIntegrator, QpData};
use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff};

/// Infinite-element bilinear form integrator.
///
/// Wraps a base integrator and applies geometric decay weighting to elements
/// whose tag matches `inf_tag` (default: elements with tag < 0 are infinite).
///
/// The decay factor `(r₀ / r)^p` is applied to the element matrix entries,
/// where `r` is the distance from `pole` to the quadrature point.
pub struct InfiniteDomainIntegrator<I: BilinearIntegrator> {
    /// Base integrator (e.g., DiffusionIntegrator or MassIntegrator).
    pub inner: I,
    /// Pole/origin for the spherical/cylindrical decay.
    pub pole: Vec<f64>,
    /// Reference radius (start of infinite layer). If `None`, uses `r₀ = 1.0`.
    pub r0: f64,
    /// Decay exponent. Default: `dim` (1D=1, 2D=1, 3D=2 for acoustics).
    pub power: f64,
    /// Element tag for infinite elements (default: tag < 0 means infinite).
    pub inf_tag: Option<i32>,
}

impl<I: BilinearIntegrator> InfiniteDomainIntegrator<I> {
    /// Create a new infinite-domain integrator.
    ///
    /// * `inner` — base integrator for the element interior
    /// * `pole` — center coordinates for the spherical/cylindrical decay
    /// * `r0` — reference radius; distance at which decay starts
    /// * `power` — decay exponent; if `None`, defaults to `dim`
    pub fn new(inner: I, pole: Vec<f64>, r0: f64, power: Option<f64>) -> Self {
        Self {
            inner,
            pole,
            r0: r0.max(1e-30),
            power: power.unwrap_or(1.0),
            inf_tag: None,
        }
    }

    /// Set the element tag for infinite elements.
    pub fn with_inf_tag(mut self, tag: i32) -> Self {
        self.inf_tag = Some(tag);
        self
    }
}

impl<I: BilinearIntegrator> BilinearIntegrator for InfiniteDomainIntegrator<I> {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        // Check if this element is infinite
        let is_infinite = match self.inf_tag {
            Some(tag) => qp.elem_tag == tag,
            None => qp.elem_tag < 0, // default: negative tags = infinite
        };

        if !is_infinite {
            // Pass through to inner integrator
            self.inner.add_to_element_matrix(qp, k_elem);
            return;
        }

        // Compute decay factor at this quadrature point
        let r = distance(&self.pole, qp.x_phys);
        let decay = if r > self.r0 {
            (self.r0 / r).powi(2) // 1/r² decay for 3D wave problems
        } else {
            1.0 // inside the reference sphere, no decay
        };

        // Scale the weight by the decay factor
        let original_weight = qp.weight;
        // We can't modify qp directly (it's immutable), so we patch k_elem after
        // First, let the inner integrator compute with the original weight
        self.inner.add_to_element_matrix(qp, k_elem);

        // Then apply the decay factor by scaling the element matrix entries.
        // This is correct only if the inner integrator produces entries
        // proportional to qp.weight (which all standard integrators do).
        if (decay - 1.0).abs() > 1e-15 {
            let n = qp.n_dofs;
            for entry in k_elem.iter_mut() {
                *entry *= decay;
            }
        }
    }
}

/// Compute Euclidean distance between a pole and a point.
fn distance(pole: &[f64], x: &[f64]) -> f64 {
    let d = pole.len().min(x.len());
    let mut sq = 0.0_f64;
    for i in 0..d {
        let di = x[i] - pole[i];
        sq += di * di;
    }
    sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::standard::{DiffusionIntegrator, MassIntegrator};
    use crate::Assembler;
    use fem_mesh::Mesh;
    use fem_space::H1Space;

    #[test]
    fn infinite_element_different_from_standard() {
        // Create a mesh. All elements have tag 1 by default.
        let mesh = Mesh::<2>::unit_square_tri(8);
        let space = H1Space::new(mesh, 1);

        // Standard mass
        let standard = MassIntegrator { rho: 1.0 };
        let k_std = Assembler::assemble_bilinear(&space, &[&standard], 3);

        // Infinite mass: treat tag 1 as infinite, pole at origin, r0 = 0.3
        let inf = InfiniteDomainIntegrator::new(standard, vec![0.0, 0.0], 0.3, None)
            .with_inf_tag(1);
        let k_inf = Assembler::assemble_bilinear(&space, &[&inf], 3);

        // They should differ because of the geometric decay
        let d_std = k_std.to_dense();
        let d_inf = k_inf.to_dense();
        let mut diff = 0.0;
        for i in 0..k_std.nrows {
            for j in 0..k_std.ncols {
                diff += (d_std[i * k_std.nrows + j] - d_inf[i * k_std.nrows + j]).abs();
            }
        }
        assert!(diff > 1e-10, "infinite and standard should differ, diff={:.3e}", diff);
    }

    #[test]
    fn infinite_tag_filtering() {
        // All elements have tag 1 → no infinite elements with inf_tag=Some(-1)
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);

        let standard = MassIntegrator { rho: 1.0 };
        let k_std = Assembler::assemble_bilinear(&space, &[&standard], 3);

        // With inf_tag = -1 (no elements match), should be identical to standard
        let inf = InfiniteDomainIntegrator::new(standard, vec![0.0, 0.0], 1.0, None)
            .with_inf_tag(-1);
        let k_inf = Assembler::assemble_bilinear(&space, &[&inf], 3);

        let d_std = k_std.to_dense();
        let d_inf = k_inf.to_dense();
        let mut diff = 0.0;
        for i in 0..k_std.nrows {
            for j in 0..k_std.ncols {
                diff += (d_std[i * k_std.nrows + j] - d_inf[i * k_std.nrows + j]).abs();
            }
        }
        assert!(diff < 1e-12, "with no matching tags, infinite=standard");
    }

    #[test]
    fn infinite_matrix_symmetric() {
        let mesh = Mesh::<2>::unit_square_tri(6);
        let space = H1Space::new(mesh, 1);

        let inner = DiffusionIntegrator { kappa: 1.0 };
        let inf = InfiniteDomainIntegrator::new(inner, vec![0.0, 0.0], 0.3, None);
        let k = Assembler::assemble_bilinear(&space, &[&inf], 3);
        let dense = k.to_dense();
        let n = k.nrows;
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i * n + j] - dense[j * n + i]).abs();
                assert!(diff < 1e-12, "asymmetry at ({i},{j}): {:.3e}", diff);
            }
        }
    }
}
