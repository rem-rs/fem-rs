//! XFEM crack propagation driver.
//!
//! Provides utilities for propagating a crack described by a level set:
//! computing propagation direction (maximum hoop stress), updating the
//! level set, and re‑detecting enrichment.
//!
//! # Usage
//! ```rust,ignore
//! use fem_assembly::xfem_crack::{CrackPropagationConfig, propagate_crack_front};
//! use fem_assembly::xfem::{XfemLevelSet, detect_enriched_nodes};
//!
//! let cfg = CrackPropagationConfig::default();
//! let (new_ls, angle) = propagate_crack_front(&old_ls, &cfg, 0);
//! ```

use super::xfem::XfemLevelSet;

/// Material and numerical parameters for crack propagation.
#[derive(Debug, Clone)]
pub struct CrackPropagationConfig {
    /// Mode‑I fracture toughness K_IC (stress intensity factor).
    pub k_ic: f64,
    /// Maximum propagation angle increment in degrees (per step).
    pub max_delta_theta_deg: f64,
    /// Crack extension length per propagation step.
    pub delta_a: f64,
    /// Number of sectors for the hoop‑stress search.
    pub n_theta: usize,
    /// Radius (in mesh units) around tip where SIFs are sampled.
    pub sample_radius: f64,
}

impl Default for CrackPropagationConfig {
    fn default() -> Self {
        Self {
            k_ic: 1.0,
            max_delta_theta_deg: 30.0,
            delta_a: 0.05,
            n_theta: 37,
            sample_radius: 0.1,
        }
    }
}

/// Result of a single propagation step.
#[derive(Debug, Clone)]
pub struct PropagationResult {
    /// Updated crack level set.
    pub crack_ls: XfemLevelSet,
    /// Propagation angle in radians (CCW from current crack direction).
    pub theta_c: f64,
    /// Equivalent SIF (MPa√m) for the given K_IC comparison.
    pub k_eq: f64,
}

/// Approximate the stress intensity factors K_I and K_II at a crack tip
/// from the crack‑tip displacement field (simplified estimator).
///
/// This uses the relative displacement of two symmetric points behind
/// the crack tip to estimate the SIFs — a rough but fast approximation.
/// For production work use the interaction integral.
pub fn estimate_sifs(
    _u: &[f64], _tip: [f64; 2], _crack_dir: [f64; 2],
    _mu: f64, _nu: f64,
) -> (f64, f64) {
    // Placeholder: returns low values.
    // In a full implementation, sample displacement jump across the crack
    // faces behind the tip and use linear elastic fracture mechanics formulas.
    (0.1, 0.0)
}

/// Compute the propagation angle using the **maximum hoop stress** criterion
/// (Erdogan–Sih):
///
/// ```text
/// K_I·sin(θ) + K_II·(3·cos(θ) − 1) = 0
/// θ_c = 2·atan((K_I ± √(K_I² + 8·K_II²)) / (4·K_II))
/// ```
///
/// Returns `θ_c` in radians (negative = clockwise from crack direction).
pub fn max_hoop_stress_angle(k_i: f64, k_ii: f64) -> f64 {
    if k_i.abs() < 1e-30 && k_ii.abs() < 1e-30 {
        return 0.0;
    }
    // Solve for the root closest to θ=0 with negative sign for K_II > 0
    let discriminant = (k_i * k_i + 8.0 * k_ii * k_ii).sqrt();
    let denom = 4.0 * k_ii;
    let theta1 = if denom.abs() > 1e-30 {
        2.0 * ((-k_i + discriminant) / denom).atan()
    } else {
        0.0
    };
    // Pick the root that gives positive hoop stress
    let cos_t1 = (theta1 / 2.0).cos().powi(3);
    let sigma_theta1 = k_i * cos_t1 - 1.5 * k_ii * theta1.cos() * (theta1 / 2.0).sin();
    if sigma_theta1 > 0.0 { theta1 } else { -theta1 }
}

/// Equivalent mode‑I SIF using the **maximum hoop stress** criterion:
/// `K_eq = cos(θ_c/2)·[K_I·cos²(θ_c/2) − 1.5·K_II·sin(θ_c)]`.
pub fn equivalent_k(k_i: f64, k_ii: f64, theta_c: f64) -> f64 {
    let ch = (theta_c / 2.0).cos();
    let _sh = (theta_c / 2.0).sin();
    ch * (k_i * ch * ch - 1.5 * k_ii * theta_c.sin())
}

/// Propagate one tip of a `CrackLine` by one step.
///
/// - `tip_index`: 0 for the first tip (`x1`), 1 for the second (`x2`).
/// - Returns the new level set and propagation info.
pub fn propagate_crack_front(
    crack_ls: &XfemLevelSet,
    cfg: &CrackPropagationConfig,
    tip_index: usize,
    k_i: f64,
    k_ii: f64,
    _mu: f64,
    _nu: f64,
) -> PropagationResult {
    match crack_ls {
        XfemLevelSet::CrackLine { x1, x2 } => {
            let tip = if tip_index == 0 { *x1 } else { *x2 };
            // Crack direction: always AWAY from the other tip.
            let crack_dir = if tip_index == 0 {
                let dx = x1[0] - x2[0];
                let dy = x1[1] - x2[1];
                let len = (dx * dx + dy * dy).sqrt().max(1e-30);
                [dx / len, dy / len]
            } else {
                let dx = x2[0] - x1[0];
                let dy = x2[1] - x1[1];
                let len = (dx * dx + dy * dy).sqrt().max(1e-30);
                [dx / len, dy / len]
            };

            // Propagation angle (max hoop stress)
            let theta_c = max_hoop_stress_angle(k_i, k_ii);
            // Clamp to max allowed angle
            let max_rad = cfg.max_delta_theta_deg.to_radians();
            let theta_c = theta_c.clamp(-max_rad, max_rad);

            // New tip position
            let new_tip = [
                tip[0] + cfg.delta_a * (crack_dir[0] * theta_c.cos() - crack_dir[1] * theta_c.sin()),
                tip[1] + cfg.delta_a * (crack_dir[0] * theta_c.sin() + crack_dir[1] * theta_c.cos()),
            ];

            // Build updated crack line
            let new_ls = if tip_index == 0 {
                XfemLevelSet::CrackLine { x1: new_tip, x2: *x2 }
            } else {
                XfemLevelSet::CrackLine { x1: *x1, x2: new_tip }
            };

            let k_eq = equivalent_k(k_i, k_ii, theta_c);
            PropagationResult { crack_ls: new_ls, theta_c, k_eq }
        }
        _ => {
            // Not a CrackLine → no propagation
            PropagationResult {
                crack_ls: crack_ls.clone(),
                theta_c: 0.0,
                k_eq: 0.0,
            }
        }
    }
}

#[cfg(test)]
#[allow(unused_variables)]
mod tests {
    use super::*;

    #[test]
    fn max_hoop_stress_pure_mode_i_is_zero() {
        // Pure mode I: K_II = 0 → angle should be 0
        let theta = max_hoop_stress_angle(1.0, 0.0);
        assert!(theta.abs() < 1e-12, "θ should be 0 for pure mode I, got {theta}");
    }

    #[test]
    fn max_hoop_stress_pure_mode_ii_negative() {
        // Pure mode II (K_I=0): θ ≈ -70.5° (≈ -1.23 rad)
        let theta = max_hoop_stress_angle(0.0, 1.0);
        assert!(theta < 0.0, "θ should be negative for K_II > 0");
        assert!((theta - (-1.23_f64)).abs() < 0.05, "θ≈-1.23 rad for pure mode II, got {theta}");
    }

    #[test]
    fn propagate_crack_line_moves_tip() {
        let ls = XfemLevelSet::CrackLine { x1: [0.0, 0.5], x2: [0.5, 0.5] };
        let cfg = CrackPropagationConfig { delta_a: 0.1, ..Default::default() };
        let result = propagate_crack_front(&ls, &cfg, 1, 1.0, 0.1, 1.0, 0.3);
        match &result.crack_ls {
            XfemLevelSet::CrackLine { x1, x2 } => {
                let dx = x2[0] - 0.5;
                let dy = x2[1] - 0.5;
                let dist = (dx*dx + dy*dy).sqrt();
                assert!(dist > 0.0, "tip should move");
                assert!(result.theta_c != 0.0, "mixed-mode should produce non-zero angle");
            }
            _ => panic!("should be CrackLine"),
        }
    }

    #[test]
    fn equivalent_k_pure_mode_i() {
        let k_eq = equivalent_k(1.0, 0.0, 0.0);
        assert!((k_eq - 1.0).abs() < 1e-12, "K_eq should equal K_I for pure mode I");
    }

    #[test]
    fn propagate_front_extends_crack_length() {
        let ls = XfemLevelSet::CrackLine { x1: [0.0, 0.5], x2: [0.3, 0.5] };
        let cfg = CrackPropagationConfig { delta_a: 0.05, ..Default::default() };
        let result = propagate_crack_front(&ls, &cfg, 1, 0.5, 0.0, 1.0, 0.3);
        match &result.crack_ls {
            XfemLevelSet::CrackLine { x1, x2 } => {
                let orig_len = 0.3;
                let new_len = ((x2[0]-x1[0]).powi(2) + (x2[1]-x1[1]).powi(2)).sqrt();
                assert!(new_len > orig_len - 1e-10, "crack should lengthen: {new_len} > {orig_len}");
            }
            _ => panic!("CrackLine expected"),
        }
    }
}
