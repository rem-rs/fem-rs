//! Thermal radiation boundary integrators.
//!
//! Provides integrators for Stefan–Boltzmann radiation boundary conditions.
//!
//! ## Ambient radiation to a fixed-temperature environment
//!
//! The radiative heat flux from a surface at temperature `T` to an ambient
//! environment at `T₀` is:
//!
//! ```text
//! q_rad = ε · σ · (T⁴ − T₀⁴)
//! ```
//!
//! where `ε` is emissivity and `σ` is the Stefan–Boltzmann constant.
//!
//! The weak form residual is:
//!
//! ```text
//! r_i = ∫_Γ ε·σ·(T⁴ − T₀⁴) · φ_i  dΓ
//! ```
//!
//! Newton linearization gives the tangent stiffness:
//!
//! ```text
//! K_ij = ∫_Γ 4·ε·σ·T³ · φ_i · φ_j  dΓ
//! ```
//!
//! ## Usage in a Newton loop
//!
//! ```rust,ignore
//! use fem_assembly::standard::{RadiationTangentIntegrator, RadiationResidualIntegrator};
//!
//! let t_fn = |x: &[f64]| -> f64 { uh.eval(x) };   // current temperature field
//! let tan = RadiationTangentIntegrator::new(0.8, t_fn);
//! let res = RadiationResidualIntegrator::new(0.8, 300.0, t_fn);
//! ```

use crate::integrator::{BdQpData, BoundaryBilinearIntegrator, BoundaryLinearIntegrator};
use crate::postproc::coefficient::*;

// ─── Stefan–Boltzmann constant ───────────────────────────────────────────────

/// Stefan–Boltzmann constant (W·m⁻²·K⁻⁴).
pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

// ─── Tangent stiffness (bilinear) ────────────────────────────────────────────

/// Radiation tangent stiffness: `∫ 4·ε·σ·T(x)³ · φ_i · φ_j  dΓ`.
///
/// Here `T(x)` is the temperature at the **previous** Newton step.
pub struct RadiationTangentIntegrator<F: Fn(&[f64]) -> f64 + Send + Sync> {
    pub emissivity: f64,
    pub temperature: F,
}

impl<F: Fn(&[f64]) -> f64 + Send + Sync> RadiationTangentIntegrator<F> {
    pub fn new(emissivity: f64, temperature: F) -> Self {
        Self { emissivity, temperature }
    }
}

impl<F: Fn(&[f64]) -> f64 + Send + Sync> BoundaryBilinearIntegrator
    for RadiationTangentIntegrator<F>
{
    fn add_to_face_matrix(&self, qp: &BdQpData<'_>, k_face: &mut [f64]) {
        let n = qp.n_dofs;
        let t = (self.temperature)(qp.x_phys);
        let coeff = 4.0 * self.emissivity * STEFAN_BOLTZMANN * t.powi(3);
        let w = qp.weight * coeff;
        // K_ij += w · φ_i · φ_j  (row-major)
        for i in 0..n {
            let row_off = i * n;
            for j in 0..n {
                k_face[row_off + j] += w * qp.phi[i] * qp.phi[j];
            }
        }
    }
}

// ─── Residual (linear form) ──────────────────────────────────────────────────

/// Radiation residual: `∫ ε·σ·(T(x)⁴ − T₀⁴) · φ_i  dΓ`.
///
/// This is the RHS contribution for a Newton step:
/// `f_rad = ∫ ε·σ·(T_prev⁴ − T₀⁴)·φ_i dΓ`.
pub struct RadiationResidualIntegrator<F: Fn(&[f64]) -> f64 + Send + Sync> {
    pub emissivity: f64,
    pub ambient_temp: f64,
    pub temperature: F,
}

impl<F: Fn(&[f64]) -> f64 + Send + Sync> RadiationResidualIntegrator<F> {
    pub fn new(emissivity: f64, ambient_temp: f64, temperature: F) -> Self {
        Self { emissivity, ambient_temp, temperature }
    }
}

impl<F: Fn(&[f64]) -> f64 + Send + Sync> BoundaryLinearIntegrator
    for RadiationResidualIntegrator<F>
{
    fn add_to_face_vector(&self, qp: &BdQpData<'_>, f_face: &mut [f64]) {
        let n = qp.n_dofs;
        let t = (self.temperature)(qp.x_phys);
        let flux = self.emissivity * STEFAN_BOLTZMANN * (t.powi(4) - self.ambient_temp.powi(4));
        let w = qp.weight * flux;
        for i in 0..n {
            f_face[i] += w * qp.phi[i];
        }
    }
}

// ─── Enclosure radiation (view factor) utilities ─────────────────────────────

/// Simple view factor: parallel coaxial disks (analytical formula).
///
/// Returns `F_{1→2}` for two parallel coaxial disks of radii `r1`, `r2`
/// separated by distance `d`.
pub fn view_factor_coaxial_disks(r1: f64, r2: f64, d: f64) -> f64 {
    let r1 = r1 / d;
    let r2 = r2 / d;
    let x = 1.0 + (1.0 + r2 * r2) / (r1 * r1);
    0.5 * (x - (x * x - 4.0 * (r2 / r1).powi(2)).sqrt())
}

/// Simple view factor: two parallel coaxial rectangles.
pub fn view_factor_parallel_rectangles(
    a: f64, b: f64, // rectangle half-dimensions
    c: f64,         // separation distance
) -> f64 {
    // Standard analytical formula for F_{1→2} of parallel rectangles
    let x = a / c;
    let y = b / c;
    let xy = x * y;
    let x2 = x * x;
    let y2 = y * y;
    let tmp = 1.0 + x2 + y2;
    (2.0 / std::f64::consts::PI / (x * y)) * (
        (xy * (1.0 + x2 + y2) / (1.0 + x2) / (1.0 + y2)).ln() / 4.0
        + x * (1.0 + y2).sqrt() * (x / (1.0 + y2).sqrt()).atan()
        + y * (1.0 + x2).sqrt() * (y / (1.0 + x2).sqrt()).atan()
        - x * (x).atan() - y * (y).atan()
    ).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stefan_boltzmann_constant_is_positive() {
        assert!(STEFAN_BOLTZMANN > 0.0);
    }

    #[test]
    fn view_factor_coaxial_disks_same() {
        // Two identical coaxial disks at d = r: F = 0.5*(3 - √5) ≈ 0.382
        let f = view_factor_coaxial_disks(1.0, 1.0, 1.0);
        assert!((f - 0.382).abs() < 0.01, "f = {f}");
    }

    #[test]
    fn view_factor_coaxial_disks_far() {
        // Two disks far apart: F → 0
        let f = view_factor_coaxial_disks(1.0, 1.0, 100.0);
        assert!(f < 0.001, "f = {f}");
    }

    #[test]
    fn view_factor_parallel_rect_same() {
        // Two identical parallel squares, small separation
        let f = view_factor_parallel_rectangles(1.0, 1.0, 0.1);
        assert!(f > 0.8 && f < 1.0, "f = {f}");
    }

    #[test]
    fn view_factor_sum_leq_one() {
        // View factor from a small surface to a large surface should be ≤ 1
        let f = view_factor_coaxial_disks(0.1, 1.0, 1.0);
        assert!(f <= 1.0 && f >= 0.0, "f = {f}");
    }
}
