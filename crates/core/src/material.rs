//! Material constitutive model trait — Abaqus UMAT/VUMAT analog.
//!
//! Defines [`MaterialModel`] for small-strain implicit/explicit analysis,
//! and [`FiniteStrainMaterial`] for finite-strain (hyperelastic) analysis.
//!
//! # Usage
//!
//! ```rust,ignore
//! use fem_core::material::{MaterialModel, MaterialResponse};
//!
//! struct MySteel { E: f64, nu: f64 }
//!
//! impl MaterialModel for MySteel {
//!     fn name(&self) -> &str { "MySteel" }
//!     fn n_state_vars(&self) -> usize { 0 }
//!     fn n_props(&self) -> usize { 2 }
//!     fn init_state(&self) -> Vec<f64> { vec![] }
//!
//!     fn update_stress(&self, strain: &[f64], state: &[f64],
//!                       dt: f64, is_3d: bool) -> MaterialResponse {
//!         // ... linear elasticity implementation
//!     }
//!
//!     fn update_stress_explicit(&self, strain: &[f64], state: &[f64],
//!                                 dt: f64, is_3d: bool) -> (Vec<f64>, Vec<f64>) {
//!         let resp = self.update_stress(strain, state, dt, is_3d);
//!         (resp.stress, resp.state)
//!     }
//! }
//! ```

/// Result of a material constitutive update (UMAT-style).
///
/// Contains the updated stress, consistent tangent stiffness, and
/// updated state variables at an integration point.
#[derive(Debug, Clone)]
pub struct MaterialResponse {
    /// Stress in Voigt notation:
    /// - 2D (plane strain): `[σxx, σyy, τxy]`
    /// - 3D: `[σxx, σyy, σzz, τxy, τyz, τzx]`
    pub stress: Vec<f64>,
    /// Consistent tangent stiffness (Jacobian `∂Δσ/∂Δε`), row-major.
    /// - 2D: 3×3 = 9 entries
    /// - 3D: 6×6 = 36 entries
    pub tangent: Vec<f64>,
    /// Updated state variables at this integration point.
    pub state: Vec<f64>,
}

// ─── Small-strain material model (UMAT/VUMAT) ──────────────────────────────

/// Small-strain constitutive model trait.
///
/// Equivalent to Abaqus UMAT (implicit) and VUMAT (explicit).
/// - UMAT: implement [`update_stress`] (stress + consistent tangent)
/// - VUMAT: implement [`update_stress_explicit`] (stress only, no tangent)
///
/// The strain is the **total strain** at the current increment.
/// For elasto-plastic models, the stress update must integrate the
/// constitutive rate form internally.
pub trait MaterialModel: Send + Sync {
    /// User-visible name of this material model.
    fn name(&self) -> &str;

    /// Number of state variables per integration point.
    fn n_state_vars(&self) -> usize;

    /// Number of material property parameters.
    fn n_props(&self) -> usize;

    /// Initialize state variables at an integration point.
    fn init_state(&self) -> Vec<f64>;

    /// UMAT-mode: update stress and compute consistent tangent.
    ///
    /// # Arguments
    /// * `strain` — total strain (Voigt) at current increment
    /// * `state` — state variables at start of increment
    /// * `dt` — time step size
    /// * `is_3d` — `true` for 3D, `false` for 2D plane strain
    ///
    /// # Returns
    /// [`MaterialResponse`] with updated stress in Voigt, consistent tangent
    /// (row-major, n×n), and updated state variables.
    fn update_stress(
        &self,
        strain: &[f64],
        state: &[f64],
        dt: f64,
        is_3d: bool,
    ) -> MaterialResponse;

    /// VUMAT-mode: update stress only (no tangent needed).
    ///
    /// Default implementation calls [`update_stress`] and discards the tangent.
    /// Override for efficiency if tangent computation is expensive.
    fn update_stress_explicit(
        &self,
        strain: &[f64],
        state: &[f64],
        dt: f64,
        is_3d: bool,
    ) -> (Vec<f64>, Vec<f64>) {
        let resp = self.update_stress(strain, state, dt, is_3d);
        (resp.stress, resp.state)
    }
}

// ─── Finite-strain material model ──────────────────────────────────────────

/// Deformation gradient for 3D finite-strain kinematics.
pub type DeformationGradient = [[f64; 3]; 3];

/// Finite-strain constitutive model trait.
///
/// For hyperelastic materials (NeoHookean, Mooney-Rivlin, etc.) that
/// take the deformation gradient `F` as input and return Cauchy stress
/// plus spatial tangent (in Voigt form).
pub trait FiniteStrainMaterial: Send + Sync {
    /// User-visible name.
    fn name(&self) -> &str;

    /// Number of state variables per integration point.
    fn n_state_vars(&self) -> usize;

    /// Initialize state at an integration point.
    fn init_state(&self) -> Vec<f64>;

    /// Compute Cauchy stress and spatial consistent tangent from `F`.
    ///
    /// # Arguments
    /// * `F` — deformation gradient `∂x/∂X` (3×3)
    /// * `state` — previous state variables
    /// * `dt` — time step
    ///
    /// # Returns
    /// * Cauchy stress in Voigt: `[σxx, σyy, σzz, σxy, σyz, σzx]`
    /// * Spatial tangent moduli `c_ijkl` in Voigt (6×6, row-major, 36 entries)
    /// * Updated state variables
    fn update_cauchy_stress(
        &self,
        F: &DeformationGradient,
        state: &[f64],
        dt: f64,
    ) -> MaterialResponse;
}

// ─── Built-in helpers ──────────────────────────────────────────────────────

/// Number of stress/strain components for a given spatial dimension.
pub fn n_voigt_components(is_3d: bool) -> usize {
    if is_3d { 6 } else { 3 }
}

/// Build the linear elasticity stiffness matrix from E and ν.
///
/// Returns the 6×6 or 3×3 stiffness in Voigt notation, row-major.
pub fn linear_elastic_stiffness(E: f64, nu: f64, is_3d: bool) -> Vec<f64> {
    if is_3d {
        let lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let mu = E / (2.0 * (1.0 + nu));
        let mut D = vec![0.0; 36];
        // Diagonal
        for i in 0..3 { D[i * 6 + i] = lam + 2.0 * mu; }
        // Off-diagonal
        for i in 0..3 { for j in 0..3 { if i != j { D[i * 6 + j] = lam; } } }
        // Shear
        D[3 * 6 + 3] = mu;
        D[4 * 6 + 4] = mu;
        D[5 * 6 + 5] = mu;
        D
    } else {
        // Plane strain
        let factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let c11 = factor * (1.0 - nu);
        let c12 = factor * nu;
        let c33 = factor * (1.0 - 2.0 * nu) / 2.0;
        vec![c11, c12, 0.0, c12, c11, 0.0, 0.0, 0.0, c33]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_components_2d() {
        assert_eq!(n_voigt_components(false), 3);
    }

    #[test]
    fn n_components_3d() {
        assert_eq!(n_voigt_components(true), 6);
    }

    #[test]
    fn linear_elastic_3d_symmetric() {
        let D = linear_elastic_stiffness(200e9, 0.3, true);
        assert_eq!(D.len(), 36);
        // Check symmetry
        for i in 0..6 {
            for j in 0..6 {
                assert!((D[i * 6 + j] - D[j * 6 + i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn linear_elastic_2d_plane_strain() {
        let D = linear_elastic_stiffness(200e9, 0.3, false);
        assert_eq!(D.len(), 9);
        assert!(D[0] > D[1]); // c11 > c12
    }

    #[test]
    fn default_explicit_equals_implicit() {
        // A trivial elastic material to test default VUMAT
        struct TestElastic { E: f64, nu: f64 }
        impl MaterialModel for TestElastic {
            fn name(&self) -> &str { "test" }
            fn n_state_vars(&self) -> usize { 0 }
            fn n_props(&self) -> usize { 2 }
            fn init_state(&self) -> Vec<f64> { vec![] }
            fn update_stress(&self, strain: &[f64], _state: &[f64],
                              _dt: f64, is_3d: bool) -> MaterialResponse {
                let D = linear_elastic_stiffness(self.E, self.nu, is_3d);
                let n = if is_3d { 6 } else { 3 };
                let mut stress = vec![0.0; n];
                for i in 0..n {
                    for j in 0..n {
                        stress[i] += D[i * n + j] * strain[j];
                    }
                }
                MaterialResponse { stress, tangent: D, state: vec![] }
            }
        }
        let mat = TestElastic { E: 200e9, nu: 0.3 };
        let strain = vec![0.001, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (s_imp, _) = mat.update_stress_explicit(&strain, &[], 0.01, true);
        let resp = mat.update_stress(&strain, &[], 0.01, true);
        assert_eq!(s_imp.len(), resp.stress.len());
        for i in 0..s_imp.len() {
            assert!((s_imp[i] - resp.stress[i]).abs() < 1e-12);
        }
    }
}
