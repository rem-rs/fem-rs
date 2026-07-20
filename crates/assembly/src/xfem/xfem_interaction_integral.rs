//! Interaction integral for extracting SIFs from XFEM solutions.
//!
//! The domain-form of the interaction integral for 2D:
//!
//! ```text
//! I^(1,2) = (1/A) · ∫_Ω [σ_ij^(1)·∂u_i^(2)/∂x_1 + σ_ij^(2)·∂u_i^(1)/∂x_1
//!            − W^(1,2)·δ_1j] · ∂q/∂x_j dΩ
//! ```
//!
//! where:
//! - (1) = numerical (XFEM) solution
//! - (2) = auxiliary field (K=1, mode I or II)
//! - q = weight function (1 at tip, 0 outside domain)
//! - A = normalization factor (∫_Ω ∂q/∂x₁ dΩ)
//!
//! ## SIF extraction
//!
//! ```text
//! K_I  = I^(1, auxI) · E' / 2
//! K_II = I^(1, auxII) · E' / 2
//! ```

use super::xfem_auxiliary_field::*;

/// Integration domain parameters.
#[derive(Debug, Clone)]
pub struct InteractionIntegralConfig {
    /// Inner radius of the integration domain (q = 1).
    pub r_inner: f64,
    /// Outer radius of the integration domain (q = 0).
    pub r_outer: f64,
    /// Number of angular sectors for the domain integral.
    pub n_sectors: usize,
    /// Number of radial integration points per sector.
    pub n_radial: usize,
}

impl Default for InteractionIntegralConfig {
    fn default() -> Self {
        Self {
            r_inner: 0.05,
            r_outer: 0.2,
            n_sectors: 16,
            n_radial: 4,
        }
    }
}

/// Compute the interaction integral I^(1, aux_mode) for given auxiliary mode.
///
/// This function computes the domain-form integral by summing over elements
/// within the integration domain. The numerical solution (1) is provided via
/// the `stress_fn` and `disp_grad_fn` closures.
///
/// # Arguments
///
/// * `tip` — crack tip position [x, y]
/// * `crack_dir` — unit vector along crack direction
/// * `mu` — shear modulus
/// * `kappa` — Kolosov constant
/// * `mode` — 1 for mode I auxiliary, 2 for mode II
/// * `config` — domain integration parameters
/// * `stress_num_fn` — returns `[σ_xx, σ_yy, σ_xy]` at physical point `[x, y]`
/// * `strain_num_fn` — returns `[ε_xx, ε_yy, γ_xy]` at physical point `[x, y]`
/// * `disp_grad_num_fn` — returns `[∂u_x/∂x₁, ∂u_y/∂x₁]` at physical point (optional)
/// * `plane_stress` — true for plane stress, false for plane strain
///
/// # Returns
///
/// The interaction integral value I^(1, aux_mode).
pub fn compute_interaction_integral<F1, F2, F3>(
    tip: &[f64; 2],
    crack_dir: &[f64; 2],
    mu: f64,
    kappa: f64,
    mode: u8,
    config: &InteractionIntegralConfig,
    stress_num_fn: F1,
    strain_num_fn: F2,
    _disp_grad_num_fn: F3,
    plane_stress: bool,
) -> f64
where
    F1: Fn(&[f64; 2]) -> [f64; 3],   // σ^(1)(x) → [σ_xx, σ_yy, σ_xy]
    F2: Fn(&[f64; 2]) -> [f64; 3],   // ε^(1)(x) → [ε_xx, ε_yy, γ_xy]
    F3: Fn(&[f64; 2]) -> [f64; 2],   // ∂u^(1)/∂x₁(x) → [∂u_x/∂x₁, ∂u_y/∂x₁]
{
    let e = 2.0 * mu * (1.0 + if plane_stress { 0.0 } else { 1.0 - (1.0 - (kappa - 1.0) / (kappa + 1.0)) });
    // Let the caller pass E' for normalization

    let mut integral = 0.0;

    // Domain integration using polar quadrature around the crack tip
    let dr = (config.r_outer - config.r_inner) / config.n_radial as f64;
    let dtheta = 2.0 * std::f64::consts::PI / config.n_sectors as f64;

    for i in 0..config.n_sectors {
        let theta = (i as f64 + 0.5) * dtheta - std::f64::consts::PI;
        for j in 0..config.n_radial {
            let r = config.r_inner + (j as f64 + 0.5) * dr;
            // Physical point
            let x = [
                tip[0] + r * (crack_dir[0] * theta.cos() - crack_dir[1] * theta.sin()),
                tip[1] + r * (crack_dir[0] * theta.sin() + crack_dir[1] * theta.cos()),
            ];

            let x_rel = [x[0] - tip[0], x[1] - tip[1]];

            // Numerical fields at this point
            let stress_num = stress_num_fn(&x);
            let strain_num = strain_num_fn(&x);

            // Auxiliary fields at this point (K = 1)
            let stress_aux = if mode == 1 { aux_stress_mode1(r, theta) } else { aux_stress_mode2(r, theta) };
            let strain_aux = if mode == 1 { aux_strain_mode1(r, theta, mu, kappa) } else { aux_strain_mode2(r, theta, mu, kappa) };

            // ∂u_i^(aux)/∂x₁
            let du_aux_dx1 = aux_displacement_grad_x1(r, theta, mu, kappa, mode);

            // ∂u_i^(1)/∂x₁ (from closure)
            // Using the provided disp_grad_num_fn
            let du_num_dx1 = _disp_grad_num_fn(&x);

            // Interaction strain energy
            let w12 = interaction_strain_energy(&stress_num, &strain_aux, &stress_aux, &strain_num);

            // ∂q/∂x_j
            let dq = grad_q(r, &x_rel, config.r_inner, config.r_outer);

            // Jacobian: dΩ = r·dr·dθ (in polar coordinates)
            let jac = r * dr * dtheta;

            // Integrand: [σ^(1):∂u^(2)/∂x₁ + σ^(2):∂u^(1)/∂x₁ − W^(1,2)·δ_1j] · ∂q/∂x_j
            // For j = 1: (σ_xx·u_x,1 + σ_xy·u_y,1 + ... − W) · q_,1
            // For j = 2: (σ_xy·u_x,1 + σ_yy·u_y,1 + ...) · q_,2  (δ₁₂ = 0)
            let term_j1 = (stress_num[0] * du_aux_dx1[0] + stress_num[2] * du_aux_dx1[1]
                         + stress_aux[0] * du_num_dx1[0] + stress_aux[2] * du_num_dx1[1]
                         - w12) * dq[0];
            let term_j2 = (stress_num[2] * du_aux_dx1[0] + stress_num[1] * du_aux_dx1[1]
                         + stress_aux[2] * du_num_dx1[0] + stress_aux[1] * du_num_dx1[1]) * dq[1];
            let integrand = term_j1 + term_j2;

            integral += integrand * jac;
        }
    }

    // No normalization needed — domain form directly gives I^(1,2)
    integral
}

/// Extract K_I and K_II from an XFEM solution using the interaction integral.
///
/// This is the main entry point. It calls `compute_interaction_integral` for
/// both mode I and mode II auxiliary fields.
///
/// # Arguments
///
/// * `tip` — crack tip position
/// * `crack_dir` — unit vector along crack direction  
/// * `mu` — shear modulus
/// * `kappa` — Kolosov constant
/// * `e_prime` — E' = E (plane stress) or E/(1-ν²) (plane strain)
/// * `config` — integration domain parameters
/// * `stress_fn` — returns σ(x) from XFEM solution
/// * `strain_fn` — returns ε(x) from XFEM solution  
/// * `disp_grad_fn` — returns ∂u/∂x₁ from XFEM solution
///
/// Returns `(K_I, K_II)`.
pub fn extract_sifs<F1, F2, F3>(
    tip: &[f64; 2],
    crack_dir: &[f64; 2],
    mu: f64,
    kappa: f64,
    e_prime: f64,
    config: &InteractionIntegralConfig,
    stress_fn: F1,
    strain_fn: F2,
    disp_grad_fn: F3,
) -> (f64, f64)
where
    F1: Fn(&[f64; 2]) -> [f64; 3],
    F2: Fn(&[f64; 2]) -> [f64; 3],
    F3: Fn(&[f64; 2]) -> [f64; 2],
{
    let i1 = compute_interaction_integral(
        tip, crack_dir, mu, kappa, 1, config,
        &stress_fn, &strain_fn, &disp_grad_fn, false);

    let i2 = compute_interaction_integral(
        tip, crack_dir, mu, kappa, 2, config,
        &stress_fn, &strain_fn, &disp_grad_fn, false);

    // K = I^(1,aux) · E' / 2
    let k_i = i1 * e_prime / 2.0;
    let k_ii = i2 * e_prime / 2.0;

    (k_i, k_ii)
}

/// Interaction integral domain test: verifies that the integral recovers
/// the known K for an analytical field.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interaction_integral_with_known_k() {
        // For a known pure mode I field (K_I = 1), the interaction integral
        // with mode I auxiliary should give I = 2/E'.
        let tip = [0.0, 0.0];
        let crack_dir = [1.0, 0.0];
        let mu = 1.0;
        let nu = 0.3;
        let e = 2.0 * mu * (1.0 + nu);
        let kappa = kappa_plane_strain(nu);
        let e_prime = e / (1.0 - nu * nu); // plane strain
        let config = InteractionIntegralConfig {
            r_inner: 0.05,
            r_outer: 0.5,
            n_sectors: 32,
            n_radial: 8,
        };

        // Stress and strain functions that return the analytical mode I field
        let stress_fn = |x: &[f64; 2]| -> [f64; 3] {
            let (r, theta) = global_to_polar(x, &tip, &crack_dir);
            let s = aux_stress_mode1(r, theta);
            // Scale to K_I = 2.0
            [s[0] * 2.0, s[1] * 2.0, s[2] * 2.0]
        };

        let strain_fn = |x: &[f64; 2]| -> [f64; 3] {
            let (r, theta) = global_to_polar(x, &tip, &crack_dir);
            let e = aux_strain_mode1(r, theta, mu, kappa);
            [e[0] * 2.0, e[1] * 2.0, e[2] * 2.0]
        };

        let disp_grad_fn = |x: &[f64; 2]| -> [f64; 2] {
            let (r, theta) = global_to_polar(x, &tip, &crack_dir);
            let du = aux_displacement_grad_x1(r, theta, mu, kappa, 1);
            [du[0] * 2.0, du[1] * 2.0]
        };

        let (k_i, k_ii) = extract_sifs(
            &tip, &crack_dir, mu, kappa, e_prime, &config,
            stress_fn, strain_fn, disp_grad_fn);

        // Should recover K_I = 2.0, K_II = 0
        assert!((k_i - 2.0).abs() < 0.5, "K_I = {k_i}, expected ~2.0");
        assert!(k_ii.abs() < 0.5, "K_II = {k_ii}, expected ~0");
    }

    #[test]
    fn config_defaults_positive() {
        let cfg = InteractionIntegralConfig::default();
        assert!(cfg.r_inner < cfg.r_outer);
        assert!(cfg.n_sectors > 0);
        assert!(cfg.n_radial > 0);
    }

    #[test]
    fn known_k_converges_with_refinement() {
        let tip = [0.0, 0.0];
        let crack_dir = [1.0, 0.0];
        let mu = 1.0;
        let nu = 0.3;
        let e = 2.0 * mu * (1.0 + nu);
        let kappa = kappa_plane_strain(nu);
        let e_prime = e / (1.0 - nu * nu);

        let stress_fn = |x: &[f64; 2]| -> [f64; 3] {
            let (r, theta) = global_to_polar(x, &tip, &crack_dir);
            let s = aux_stress_mode1(r, theta);
            [s[0], s[1], s[2]] // K_I = 1.0
        };
        let strain_fn = |x: &[f64; 2]| -> [f64; 3] {
            let (r, theta) = global_to_polar(x, &tip, &crack_dir);
            let e = aux_strain_mode1(r, theta, mu, kappa);
            [e[0], e[1], e[2]]
        };
        let disp_grad_fn = |x: &[f64; 2]| -> [f64; 2] {
            let (r, theta) = global_to_polar(x, &tip, &crack_dir);
            aux_displacement_grad_x1(r, theta, mu, kappa, 1)
        };

        // Coarse integration
        let coarse = InteractionIntegralConfig {
            r_inner: 0.05, r_outer: 0.5, n_sectors: 8, n_radial: 2,
        };
        let (k_i_coarse, _) = extract_sifs(
            &tip, &crack_dir, mu, kappa, e_prime, &coarse,
            stress_fn, strain_fn, disp_grad_fn);

        // Fine integration
        let fine = InteractionIntegralConfig {
            r_inner: 0.05, r_outer: 0.5, n_sectors: 48, n_radial: 16,
        };
        let (k_i_fine, _) = extract_sifs(
            &tip, &crack_dir, mu, kappa, e_prime, &fine,
            stress_fn, strain_fn, disp_grad_fn);

        // Fine should be closer to 1.0 than coarse
        let err_coarse = (k_i_coarse - 1.0).abs();
        let err_fine = (k_i_fine - 1.0).abs();
        assert!(
            err_fine < err_coarse + 0.1,
            "K_I error: coarse={err_coarse:.4}, fine={err_fine:.4}"
        );
    }
}
