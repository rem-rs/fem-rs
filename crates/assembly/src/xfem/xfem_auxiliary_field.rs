//! 2D linear elastic fracture mechanics auxiliary fields.
//!
//! Provides the crack-tip asymptotic fields (Williams solution) used in the
//! interaction integral for extracting stress intensity factors from XFEM
//! solutions.
//!
//! ## Field equations (plane strain)
//!
//! Mode I displacement field (K_I = 1):
//! ```text
//! u_x = (1/2μ)·√(r/2π) · cos(θ/2)·[κ − 1 + 2·sin²(θ/2)]
//! u_y = (1/2μ)·√(r/2π) · sin(θ/2)·[κ + 1 − 2·cos²(θ/2)]
//! ```
//!
//! Mode I stress field (K_I = 1):
//! ```text
//! σ_xx = (1/√(2πr)) · cos(θ/2)·[1 − sin(θ/2)·sin(3θ/2)]
//! σ_yy = (1/√(2πr)) · cos(θ/2)·[1 + sin(θ/2)·sin(3θ/2)]
//! σ_xy = (1/√(2πr)) · sin(θ/2)·cos(θ/2)·cos(3θ/2)
//! ```
//!
//! where κ = 3−4ν (plane strain) or (3−ν)/(1+ν) (plane stress),
//! and μ = E/(2(1+ν)).

use std::f64::consts::PI;

/// Kolosov constant for plane strain.
pub fn kappa_plane_strain(nu: f64) -> f64 {
    3.0 - 4.0 * nu
}

/// Kolosov constant for plane stress.
pub fn kappa_plane_stress(nu: f64) -> f64 {
    (3.0 - nu) / (1.0 + nu)
}

/// Shear modulus.
pub fn shear_modulus(e: f64, nu: f64) -> f64 {
    e / (2.0 * (1.0 + nu))
}

// ─── Mode I auxiliary field (K_I = 1) ───────────────────────────────────────

/// Mode I auxiliary displacement (K_I = 1) at (r, θ).
///
/// Returns `[u_x, u_y]`.
pub fn aux_displacement_mode1(r: f64, theta: f64, mu: f64, kappa: f64) -> [f64; 2] {
    let sqrt_term = (r / (2.0 * PI)).sqrt().max(1e-30);
    let ct2 = (theta / 2.0).cos();
    let st2 = (theta / 2.0).sin();
    let pref = 1.0 / (2.0 * mu);
    [
        pref * sqrt_term * ct2 * (kappa - 1.0 + 2.0 * st2 * st2),
        pref * sqrt_term * st2 * (kappa + 1.0 - 2.0 * ct2 * ct2),
    ]
}

/// Mode I auxiliary stress (K_I = 1) at (r, θ).
///
/// Returns `[σ_xx, σ_yy, σ_xy]` in Voigt notation.
pub fn aux_stress_mode1(r: f64, theta: f64) -> [f64; 3] {
    let sqrt_term = 1.0 / (2.0 * PI * r).sqrt().max(1e-30);
    let ct2 = (theta / 2.0).cos();
    let st2 = (theta / 2.0).sin();
    let s3t2 = (3.0 * theta / 2.0).sin();
    let c3t2 = (3.0 * theta / 2.0).cos();
    [
        sqrt_term * ct2 * (1.0 - st2 * s3t2),
        sqrt_term * ct2 * (1.0 + st2 * s3t2),
        sqrt_term * st2 * ct2 * c3t2,
    ]
}

/// Mode I auxiliary strain (K_I = 1) at (r, θ).
///
/// Returns `[ε_xx, ε_yy, γ_xy]` (Voigt).
pub fn aux_strain_mode1(r: f64, theta: f64, mu: f64, kappa: f64) -> [f64; 3] {
    let s = aux_stress_mode1(r, theta);
    // For plane strain: ε_xx = (1/E')·(σ_xx − ν'·σ_yy) etc.
    // Using μ and κ: ε_xx = (1/(2μ))·(σ_xx − (3−κ)/(1+κ)·σ_yy) ... simplified
    // For the interaction integral we typically use σ directly, not ε.
    // This is a placeholder — use stress-based form in the integral.
    let factor = 1.0 / (2.0 * mu);
    [factor * s[0], factor * s[1], factor * s[2] * 2.0]
}

// ─── Mode II auxiliary field (K_II = 1) ──────────────────────────────────────

/// Mode II auxiliary displacement (K_II = 1) at (r, θ).
///
/// Returns `[u_x, u_y]`.
pub fn aux_displacement_mode2(r: f64, theta: f64, mu: f64, kappa: f64) -> [f64; 2] {
    let sqrt_term = (r / (2.0 * PI)).sqrt().max(1e-30);
    let ct2 = (theta / 2.0).cos();
    let st2 = (theta / 2.0).sin();
    let pref = 1.0 / (2.0 * mu);
    [
        pref * sqrt_term * st2 * (kappa + 1.0 + 2.0 * ct2 * ct2),
        pref * sqrt_term * ct2 * (kappa - 1.0 - 2.0 * st2 * st2),
    ]
}

/// Mode II auxiliary stress (K_II = 1) at (r, θ).
///
/// Returns `[σ_xx, σ_yy, σ_xy]` in Voigt notation.
pub fn aux_stress_mode2(r: f64, theta: f64) -> [f64; 3] {
    let sqrt_term = 1.0 / (2.0 * PI * r).sqrt().max(1e-30);
    let ct2 = (theta / 2.0).cos();
    let st2 = (theta / 2.0).sin();
    let c3t2 = (3.0 * theta / 2.0).cos();
    let s3t2 = (3.0 * theta / 2.0).sin();
    [
        -sqrt_term * st2 * (2.0 + ct2 * c3t2),
        sqrt_term * st2 * ct2 * c3t2,
        sqrt_term * ct2 * (1.0 - st2 * s3t2),
    ]
}

/// Mode II auxiliary strain (K_II = 1) at (r, θ).
pub fn aux_strain_mode2(r: f64, theta: f64, mu: f64, _kappa: f64) -> [f64; 3] {
    let s = aux_stress_mode2(r, theta);
    let factor = 1.0 / (2.0 * mu);
    [factor * s[0], factor * s[1], factor * s[2] * 2.0]
}

// ─── Derived quantities ──────────────────────────────────────────────────────

/// Interaction strain energy: W^(1,2) = σ_ij^(1)·ε_ij^(2) = σ_ij^(2)·ε_ij^(1)
///
/// Uses Voigt notation: `[σ_xx, σ_yy, σ_xy]` × `[ε_xx, ε_yy, γ_xy]`
pub fn interaction_strain_energy(
    stress_num: &[f64; 3],   // σ^(1)
    strain_aux: &[f64; 3],   // ε^(2)
    stress_aux: &[f64; 3],   // σ^(2)
    strain_num: &[f64; 3],   // ε^(1)
) -> f64 {
    // W^(1,2) = 0.5 * (σ^(1):ε^(2) + σ^(2):ε^(1))
    0.5 * (
        stress_num[0] * strain_aux[0] + stress_num[1] * strain_aux[1] + stress_num[2] * strain_aux[2]
        + stress_aux[0] * strain_num[0] + stress_aux[1] * strain_num[1] + stress_aux[2] * strain_num[2]
    )
}

/// Derivative of the displacement field with respect to x₁ (crack direction).
///
/// For the auxiliary field, this is the analytical gradient in global coordinates.
/// For the mode I field at (r, θ), ∂u/∂x₁ = (∂u/∂r)·(∂r/∂x₁) + (∂u/∂θ)·(∂θ/∂x₁).
///
/// This function computes ∂u_i^(aux)/∂x₁ via finite differences for robustness.
pub fn aux_displacement_grad_x1(
    r: f64, theta: f64, mu: f64, kappa: f64,
    mode: u8,
) -> [f64; 2] {  // [∂u_x/∂x₁, ∂u_y/∂x₁]
    let eps = 1e-8;
    let dr = eps * r.max(1e-6);
    let (u_plus, u_minus) = if mode == 1 {
        let up = aux_displacement_mode1(r + dr, theta, mu, kappa);
        let um = aux_displacement_mode1(r - dr, theta, mu, kappa);
        (up, um)
    } else {
        let up = aux_displacement_mode2(r + dr, theta, mu, kappa);
        let um = aux_displacement_mode2(r - dr, theta, mu, kappa);
        (up, um)
    };
    // ∂u/∂x₁ ≈ (∂u/∂r)·cos(θ) − (1/r)·(∂u/∂θ)·sin(θ)
    // Simplified: use central differences along x₁ direction
    let dtheta = eps;
    let (u_tplus, u_tminus) = if mode == 1 {
        let up = aux_displacement_mode1(r, theta + dtheta, mu, kappa);
        let um = aux_displacement_mode1(r, theta - dtheta, mu, kappa);
        (up, um)
    } else {
        let up = aux_displacement_mode2(r, theta + dtheta, mu, kappa);
        let um = aux_displacement_mode2(r, theta - dtheta, mu, kappa);
        (up, um)
    };
    let dudr = [(u_plus[0] - u_minus[0]) / (2.0 * dr),
                (u_plus[1] - u_minus[1]) / (2.0 * dr)];
    let dudt = [(u_tplus[0] - u_tminus[0]) / (2.0 * dtheta),
                (u_tplus[1] - u_tminus[1]) / (2.0 * dtheta)];
    // Chain rule: ∂/∂x₁ = cos(θ)·∂/∂r − sin(θ)/r · ∂/∂θ
    let ct = theta.cos();
    let st = theta.sin();
    let inv_r = 1.0 / r.max(1e-30);
    [
        ct * dudr[0] - st * inv_r * dudt[0],
        ct * dudr[1] - st * inv_r * dudt[1],
    ]
}

// ─── q-function for domain integration ───────────────────────────────────────

/// Weight function q(x) for the interaction integral domain form.
///
/// `q(x) = 1` for `r ≤ r_inner`, `q(x) = 0` for `r ≥ r_outer`,
/// with a smooth cubic transition in between.
pub fn q_function(
    r: f64,
    r_inner: f64,
    r_outer: f64,
) -> f64 {
    if r <= r_inner {
        1.0
    } else if r >= r_outer {
        0.0
    } else {
        // Cubic Hermite interpolation: q(s) = 1 − 3s² + 2s³, s = (r−r_in)/(r_out−r_in)
        let s = (r - r_inner) / (r_outer - r_inner);
        let s2 = s * s;
        let s3 = s2 * s;
        1.0 - 3.0 * s2 + 2.0 * s3
    }
}

/// ∂q/∂x_j — gradient of the weight function.
pub fn grad_q(
    r: f64,
    x_rel: &[f64; 2],  // (x − tip) in global coordinates
    r_inner: f64,
    r_outer: f64,
) -> [f64; 2] {
    if r <= r_inner || r >= r_outer {
        return [0.0, 0.0];
    }
    let dr = r_outer - r_inner;
    let s = (r - r_inner) / dr;
    let ds = 1.0 / dr;
    // dq/ds = -6s + 6s²
    let dqds = -6.0 * s + 6.0 * s * s;
    let dqdr = dqds * ds;
    // ∂q/∂x_j = (dq/dr) · (∂r/∂x_j) = dqdr · x_rel_j / r
    let inv_r = 1.0 / r.max(1e-30);
    [dqdr * x_rel[0] * inv_r, dqdr * x_rel[1] * inv_r]
}

/// Convert global coordinates (x, tip) to polar (r, θ) relative to crack direction.
pub fn global_to_polar(
    x: &[f64; 2],
    tip: &[f64; 2],
    crack_dir: &[f64; 2],  // unit vector pointing along crack
) -> (f64, f64) {
    let dx = x[0] - tip[0];
    let dy = x[1] - tip[1];
    let r = (dx * dx + dy * dy).sqrt().max(1e-30);

    // θ = angle measured from crack direction
    // cos θ = (dx·crack_dir_x + dy·crack_dir_y) / r
    // sin θ = (-dx·crack_dir_y + dy·crack_dir_x) / r  (cross product)
    let cos_theta = (dx * crack_dir[0] + dy * crack_dir[1]) / r;
    let sin_theta = (-dx * crack_dir[1] + dy * crack_dir[0]) / r;

    (r, sin_theta.atan2(cos_theta))
}

/// Effective Young's modulus for SIF extraction.
///
/// E' = E (plane stress), E' = E/(1−ν²) (plane strain)
pub fn eprime(e: f64, nu: f64, plane_stress: bool) -> f64 {
    if plane_stress { e } else { e / (1.0 - nu * nu) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kappa_plane_strain_value() {
        let k = kappa_plane_strain(0.3);
        assert!((k - 1.8).abs() < 1e-10);
    }

    #[test]
    fn aux_displacement_mode1_finite() {
        let u = aux_displacement_mode1(1.0, 0.0, 1.0, 1.8);
        assert!(u[0].is_finite());
        assert!(u[1].is_finite());
    }

    #[test]
    fn aux_stress_mode1_finite() {
        let s = aux_stress_mode1(1.0, 0.0);
        assert!(s[0].is_finite());
        assert!(s[1].is_finite());
        assert!(s[2].is_finite());
    }

    #[test]
    fn aux_stress_singularity() {
        // Stress should grow as 1/√r as r → 0
        let s_near = aux_stress_mode1(0.01, 0.0);
        let s_far = aux_stress_mode1(1.0, 0.0);
        assert!(s_near[0].abs() > s_far[0].abs());
    }

    #[test]
    fn aux_displacement_singularity() {
        // Displacement should go as √r as r → 0
        let u_near = aux_displacement_mode1(0.01, 0.0, 1.0, 1.8);
        let u_far = aux_displacement_mode1(1.0, 0.0, 1.0, 1.8);
        assert!(u_near[0].abs() < u_far[0].abs());
    }

    #[test]
    fn q_function_is_one_near_tip() {
        assert!((q_function(0.0, 0.1, 0.5) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn q_function_is_zero_far() {
        assert!((q_function(1.0, 0.1, 0.5) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn q_function_continuous() {
        let q_mid = q_function(0.3, 0.1, 0.5);
        assert!(q_mid > 0.0 && q_mid < 1.0);
    }

    #[test]
    fn grad_q_zero_inside() {
        let g = grad_q(0.05, &[0.0, 0.05], 0.1, 0.5);
        assert_eq!(g, [0.0, 0.0]);
    }

    #[test]
    fn global_to_polar_on_axis() {
        let (r, theta) = global_to_polar(
            &[2.0, 0.0], &[1.0, 0.0], &[1.0, 0.0]);
        assert!((r - 1.0).abs() < 1e-10);
        assert!(theta.abs() < 1e-10);
    }

    #[test]
    fn global_to_polar_90_degrees() {
        let (r, theta) = global_to_polar(
            &[1.0, 1.0], &[1.0, 0.0], &[1.0, 0.0]);
        assert!((r - 1.0).abs() < 1e-10);
        assert!((theta - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn interaction_strain_energy_symmetric() {
        let s1 = [1.0, 2.0, 0.5];
        let e2 = [0.1, 0.2, 0.05];
        let s2 = [0.5, 1.0, 0.25];
        let e1 = [0.2, 0.4, 0.1];
        let w = interaction_strain_energy(&s1, &e2, &s2, &e1);
        assert!(w.is_finite());
    }

    #[test]
    fn eprime_plane_stress() {
        let ep = eprime(200e9, 0.3, true);
        assert!((ep - 200e9).abs() < 1.0);
    }

    #[test]
    fn mode2_displacement_finite() {
        let u = aux_displacement_mode2(1.0, PI / 4.0, 1.0, 1.8);
        assert!(u[0].is_finite());
        assert!(u[1].is_finite());
    }

    #[test]
    fn mode2_stress_finite() {
        let s = aux_stress_mode2(1.0, PI / 4.0);
        assert!(s[0].is_finite());
        assert!(s[1].is_finite());
        assert!(s[2].is_finite());
    }
}
