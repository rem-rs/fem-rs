//! Mass scaling for explicit dynamics.
//!
//! Increases the critical time step in explicit time integration by
//! selectively adding artificial mass to small or stiff elements.
//!
//! # Theory
//!
//! The critical time step for central difference is:
//! ```text
//! Δt_crit = min_e (L_e / c_e)
//! ```
//! where `L_e` is the element characteristic length and `c_e` is the
//! wave speed. Mass scaling increases `L_e / c_e` by adding artificial
//! mass to elements whose stable time step is below the target.
//!
//! # Usage
//!
//! ```rust,ignore
//! use fem_solver::mass_scaling::*;
//!
//! let cfg = MassScalingConfig::new(1e-6)  // target Δt = 1 μs
//!     .max_scale_factor(100.0);           // limit scaling
//! let mass_scaled = cfg.apply(&mass_original, &element_dt, &element_dof_map);
//! ```

/// Mass scaling configuration.
#[derive(Debug, Clone)]
pub struct MassScalingConfig {
    /// Target critical time step (s).
    pub target_dt: f64,
    /// Maximum allowable mass scale factor per element.
    pub max_scale_factor: f64,
    /// Scaling method.
    pub method: MassScalingMethod,
}

/// Scaling method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MassScalingMethod {
    /// Scale only elements whose stable time step < target_dt.
    Selective,
    /// Scale all elements uniformly to match target_dt.
    Uniform,
}

impl MassScalingConfig {
    pub fn new(target_dt: f64) -> Self {
        Self {
            target_dt,
            max_scale_factor: 1000.0,
            method: MassScalingMethod::Selective,
        }
    }

    pub fn max_scale_factor(mut self, f: f64) -> Self { self.max_scale_factor = f; self }
    pub fn method(mut self, m: MassScalingMethod) -> Self { self.method = m; self }

    /// Apply mass scaling.
    ///
    /// `mass_original` — diagonal entries of the original lumped mass matrix
    /// `element_dt` — stable time step for each element (computed from element size & wave speed)
    /// `element_dofs` — for each element, the list of DOF indices it contributes to
    ///
    /// Returns the scaled mass vector.
    pub fn apply(
        &self,
        mass_original: &[f64],
        element_dt: &[f64],
        element_dofs: &[Vec<usize>],
    ) -> Vec<f64> {
        let n_dofs = mass_original.len();
        let mut mass_scaled = mass_original.to_vec();

        match self.method {
            MassScalingMethod::Selective => {
                let target = self.target_dt.max(1e-30);
                for (e, &dt_e) in element_dt.iter().enumerate() {
                    if dt_e >= target { continue; }
                    let factor = (target / dt_e.max(1e-30)).powi(2);
                    let factor = factor.min(self.max_scale_factor);

                    // Add scaled mass to element DOFs
                    if let Some(dofs) = element_dofs.get(e) {
                        let added_mass = (factor - 1.0) * mass_original.iter().map(|m| *m).sum::<f64>() / n_dofs as f64;
                        for &dof in dofs {
                            if dof < n_dofs {
                                mass_scaled[dof] += added_mass / dofs.len().max(1) as f64;
                            }
                        }
                    }
                }
            }
            MassScalingMethod::Uniform => {
                let target = self.target_dt.max(1e-30);
                let min_dt = element_dt.iter().fold(f64::MAX, |a, &b| a.min(b));
                if min_dt > 1e-30 {
                    let factor = (target / min_dt).powi(2);
                    let factor = factor.min(self.max_scale_factor);
                    for m in &mut mass_scaled {
                        *m *= factor;
                    }
                }
            }
        }

        mass_scaled
    }
}

/// Compute element characteristic length for common element types.
///
/// `volume` — element volume (or area in 2D)
/// `n_dims` — spatial dimension (2 or 3)
/// Returns characteristic length as `volume^(1/n_dims)`.
pub fn element_characteristic_length(volume: f64, n_dims: usize) -> f64 {
    if volume < 1e-30 { return 1e-6; }
    volume.powf(1.0 / n_dims as f64)
}

/// Compute element stable time step.
///
/// `char_length` — element characteristic length (m)
/// `wave_speed` — dilatational wave speed (m/s) = sqrt(E(1-ν)/((1+ν)(1-2ν)ρ))
pub fn element_stable_dt(char_length: f64, wave_speed: f64) -> f64 {
    if wave_speed < 1e-30 { return f64::MAX; }
    char_length / wave_speed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selective_scaling_increases_small_mass() {
        let mass = vec![1.0; 10];
        let elem_dt = vec![1e-7, 1e-6]; // first element below target
        let dof_map = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]];
        let cfg = MassScalingConfig::new(5e-7).method(MassScalingMethod::Selective);
        let scaled = cfg.apply(&mass, &elem_dt, &dof_map);
        // DOFs of element 0 should have increased mass
        for i in 0..4 {
            assert!(scaled[i] > 1.0, "scaled[{}] = {}", i, scaled[i]);
        }
    }

    #[test]
    fn uniform_scaling_scales_all() {
        let mass = vec![1.0; 10];
        let elem_dt = vec![1e-7, 1e-6];
        let dof_map = vec![vec![0, 1], vec![2, 3]];
        let cfg = MassScalingConfig::new(5e-7).method(MassScalingMethod::Uniform);
        let scaled = cfg.apply(&mass, &elem_dt, &dof_map);
        for &m in &scaled {
            assert!(m > 1.0, "scaled = {}", m);
        }
    }

    #[test]
    fn no_scaling_when_target_already_met() {
        let mass = vec![1.0; 5];
        let elem_dt = vec![1e-5, 2e-5];
        let dof_map = vec![vec![0, 1], vec![2, 3]];
        let cfg = MassScalingConfig::new(1e-6);
        let scaled = cfg.apply(&mass, &elem_dt, &dof_map);
        for i in 0..5 {
            assert!((scaled[i] - 1.0).abs() < 1e-10, "scaled[{}] = {}", i, scaled[i]);
        }
    }

    #[test]
    fn char_length_is_reasonable() {
        let l = element_characteristic_length(1.0, 3);
        assert!((l - 1.0).abs() < 1e-10);
        let l2 = element_characteristic_length(0.001, 3);
        assert!((l2 - 0.1).abs() < 0.01);
    }

    #[test]
    fn stable_dt_scales_with_length() {
        let dt1 = element_stable_dt(1.0, 5000.0);
        let dt2 = element_stable_dt(0.5, 5000.0);
        assert!((dt1 - 2.0 * dt2).abs() < 1e-15);
    }
}
