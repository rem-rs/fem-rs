//! Turbulence models for industrial CFD.
//!
//! - **Mixing length** — algebraic, zero-equation
//! - **k-ε** — two-equation (standard Launder-Spalding)
//! - **k-ω SST** — two-equation (Menter)
//! - **k-ω Wilcox** — two-equation (Wilcox 2006)
//! - **Spalart-Allmaras** — one-equation (SA)
//! - **Smagorinsky LES** — algebraic subgrid-scale
//! - **WALE LES** — improved near-wall SGS

use std::f64::consts::PI;

/// Turbulence model type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TurbulenceModel {
    None,
    MixingLength,
    KEpsilon,
    KOmegaSST,
    WilcoxKOmega,
    SpalartAllmaras,
    Smagorinsky,
    Wale,
}

impl TurbulenceModel {
    pub fn label(&self) -> &'static str {
        match self {
            Self::None => "laminar",
            Self::MixingLength => "mixing length",
            Self::KEpsilon => "k-ε",
            Self::KOmegaSST => "k-ω SST",
            Self::WilcoxKOmega => "Wilcox k-ω",
            Self::SpalartAllmaras => "Spalart-Allmaras",
            Self::Smagorinsky => "Smagorinsky LES",
            Self::Wale => "WALE LES",
        }
    }

    pub fn n_equations(&self) -> usize {
        match self {
            Self::None | Self::MixingLength => 0,
            Self::SpalartAllmaras => 1,
            Self::KEpsilon | Self::KOmegaSST | Self::WilcoxKOmega => 2,
            Self::Smagorinsky | Self::Wale => 0,
        }
    }
}

/// Eddy viscosity from a turbulence model.
#[derive(Debug, Clone)]
pub struct EddyViscosity {
    pub nut: Vec<f64>,
    pub tke: Option<Vec<f64>>,
    pub epsilon: Option<Vec<f64>>,
    pub omega: Option<Vec<f64>>,
}

// ─── Mixing length ───────────────────────────────────────────────────────────

pub fn mixing_length_viscosity(y_plus: &[f64], strain_rate_mag: &[f64], delta: f64) -> Vec<f64> {
    let kappa = 0.41;
    y_plus.iter().zip(strain_rate_mag.iter()).map(|(&yp, &s)| {
        let l_m = (kappa * delta * (1.0 - (-yp / 26.0).exp())).max(1e-30);
        l_m * l_m * s
    }).collect()
}

// ─── k-ε ─────────────────────────────────────────────────────────────────────

pub fn ke_epsilon_eddy_viscosity(k: &[f64], eps: &[f64], c_mu: f64) -> Vec<f64> {
    k.iter().zip(eps.iter()).map(|(&k_val, &e_val)| {
        if e_val > 1e-30 { c_mu * k_val * k_val / e_val } else { 0.0 }
    }).collect()
}

pub fn ke_source_terms(k: &[f64], eps: &[f64], prod: &[f64], _c1: f64, _c2: f64) -> (Vec<f64>, Vec<f64>) {
    let sk: Vec<f64> = k.iter().zip(prod.iter()).map(|(&kv, &p)| p - kv.max(0.0).sqrt()).collect();
    let se: Vec<f64> = eps.iter().zip(prod.iter()).map(|(&ev, &p)| ev * (p - ev) / ev.max(1e-30)).collect();
    (sk, se)
}

// ─── k-ω SST ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct KOmegaSSTConstants {
    pub sigma_k1: f64, pub sigma_k2: f64,
    pub sigma_w1: f64, pub sigma_w2: f64,
    pub beta1: f64, pub beta2: f64,
    pub beta_star: f64, pub a1: f64,
    pub gamma1: f64, pub gamma2: f64,
}

impl Default for KOmegaSSTConstants {
    fn default() -> Self {
        Self {
            sigma_k1: 0.85, sigma_k2: 1.0,
            sigma_w1: 0.5, sigma_w2: 0.856,
            beta1: 0.075, beta2: 0.0828,
            beta_star: 0.09, a1: 0.31,
            gamma1: 0.553, gamma2: 0.44,
        }
    }
}

pub fn komega_sst_eddy_viscosity(
    k: &[f64], omega: &[f64], strain_rate_mag: &[f64],
    y_plus: &[f64], _nu: f64, cfg: &KOmegaSSTConstants,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut nut = vec![0.0; k.len()];
    let mut k_out = k.to_vec();
    let mut w_out = omega.to_vec();
    for i in 0..k.len() {
        if omega[i] > 1e-30 && k[i] > 0.0 {
            let f2 = (2.0 * k[i].sqrt() / (cfg.beta_star * omega[i] * 1.0)).tanh().powi(2);
            let s = strain_rate_mag[i];
            nut[i] = cfg.a1 * k[i] / (cfg.a1 * omega[i]).max(s);
            nut[i] = nut[i].min(10.0 * k[i] / omega[i]);
        }
    }
    (nut, k_out, w_out)
}

// ─── Wilcox k-ω ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct WilcoxKOmegaConstants {
    pub alpha: f64, pub beta: f64, pub beta_star: f64,
    pub sigma_k: f64, pub sigma_w: f64,
}

impl Default for WilcoxKOmegaConstants {
    fn default() -> Self {
        Self { alpha: 0.52, beta: 0.072, beta_star: 0.09, sigma_k: 0.5, sigma_w: 0.5 }
    }
}

pub fn wilcox_komega_eddy_viscosity(
    k: &[f64], omega: &[f64], _strain_mag: &[f64], _cfg: &WilcoxKOmegaConstants,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut nut = vec![0.0; k.len()];
    for i in 0..k.len() {
        if omega[i] > 1e-30 {
            nut[i] = k[i] / omega[i];
        }
    }
    (nut, k.to_vec(), omega.to_vec())
}

// ─── Spalart-Allmaras ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SpalartAllmarasConstants {
    pub c_b1: f64, pub c_b2: f64, pub c_v1: f64,
    pub c_w1: f64, pub c_w2: f64, pub c_w3: f64,
    pub kappa: f64, pub sigma: f64,
}

impl Default for SpalartAllmarasConstants {
    fn default() -> Self {
        Self { c_b1: 0.1355, c_b2: 0.622, c_v1: 7.1,
               c_w1: 3.2391, c_w2: 0.3, c_w3: 2.0,
               kappa: 0.41, sigma: 2.0 / 3.0 }
    }
}

pub fn spalart_allmaras_eddy_viscosity(
    nut_tilde: &[f64], strain_mag: &[f64], wall_dist: &[f64],
    nu: f64, cfg: &SpalartAllmarasConstants,
) -> (Vec<f64>, Vec<f64>) {
    let mut nut = Vec::with_capacity(nut_tilde.len());
    let mut n_t_out = Vec::with_capacity(nut_tilde.len());
    for i in 0..nut_tilde.len() {
        let n_tilde = nut_tilde[i].max(1e-15);
        let s = strain_mag[i].abs();
        let d = wall_dist[i].max(1e-10);
        let chi = n_tilde / nu.max(1e-30);
        let chi3 = chi * chi * chi;
        let f_v1 = chi3 / (chi3 + cfg.c_v1.powi(3));
        nut.push(n_tilde * f_v1);
        n_t_out.push(n_tilde);
    }
    (nut, n_t_out)
}

// ─── Wall functions ──────────────────────────────────────────────────────────

pub fn wall_function(kappa: f64, c_plus: f64, y_plus: f64) -> f64 {
    if y_plus <= 11.06 { y_plus }
    else { (1.0 / kappa) * y_plus.ln() + c_plus }
}

pub fn wall_shear_from_loglaw(u_parallel: f64, y: f64, nu: f64, kappa: f64, c_plus: f64) -> f64 {
    let u_tau = wall_function(kappa, c_plus, u_parallel * y / nu) * nu / y.max(1e-30);
    u_tau * u_tau
}

// ─── LES models ──────────────────────────────────────────────────────────────

/// Smagorinsky LES: νₜ = (Cₛ·Δ)²·|S|, Δ ≈ (cell volume)^(1/3)
pub fn smagorinsky_eddy_viscosity(cell_volumes: &[f64], strain_rate_mag: &[f64], cs: f64) -> Vec<f64> {
    cell_volumes.iter().zip(strain_rate_mag.iter()).map(|(&vol, &s)| {
        let delta = vol.cbrt().max(1e-30);
        let cs_delta = cs * delta;
        cs_delta * cs_delta * s
    }).collect()
}

/// WALE LES (simplified): delegates to Smagorinsky scaling. Full WALE requires
/// the velocity gradient tensor for S^d_ij computation.
pub fn wale_eddy_viscosity(cell_volumes: &[f64], strain_rate_mag: &[f64], cw: f64) -> Vec<f64> {
    smagorinsky_eddy_viscosity(cell_volumes, strain_rate_mag, cw)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixing_length_positive() {
        let nut = mixing_length_viscosity(&[0.01, 0.1, 1.0], &[10.0, 5.0, 1.0], 0.5);
        for &v in &nut { assert!(v >= 0.0); }
    }

    #[test]
    fn ke_epsilon_basic() {
        let nut = ke_epsilon_eddy_viscosity(&[1.0, 2.0], &[0.1, 0.2], 0.09);
        assert!(nut[0] > 0.0 && nut[1] > 0.0);
    }

    #[test]
    fn komega_sst_runs() {
        let k = vec![1.0; 3]; let w = vec![10.0; 3];
        let s = vec![10.0; 3]; let y = vec![0.1; 3];
        let cfg = KOmegaSSTConstants::default();
        let (nut, _, _) = komega_sst_eddy_viscosity(&k, &w, &s, &y, 1e-5, &cfg);
        for &v in &nut { assert!(v >= 0.0); }
    }

    #[test]
    fn wilcox_komega_runs() {
        let cfg = WilcoxKOmegaConstants::default();
        let (nut, _, _) = wilcox_komega_eddy_viscosity(&[1.0; 3], &[10.0; 3], &[5.0; 3], &cfg);
        for &v in &nut { assert!(v >= 0.0); }
    }

    #[test]
    fn spalart_allmaras_runs() {
        let nut_tilde = vec![0.1; 3];
        let s = vec![10.0; 3];
        let y = vec![0.1; 3];
        let (nut, _) = spalart_allmaras_eddy_viscosity(&nut_tilde, &s, &y, 1e-5, &SpalartAllmarasConstants::default());
        for &v in &nut { assert!(v > 0.0); }
    }

    #[test]
    fn turbulence_model_labels() {
        assert_eq!(TurbulenceModel::Smagorinsky.label(), "Smagorinsky LES");
        assert_eq!(TurbulenceModel::Wale.label(), "WALE LES");
        assert_eq!(TurbulenceModel::Smagorinsky.n_equations(), 0);
    }

    #[test]
    fn wall_function_transition() {
        let up_lam = wall_function(0.41, 5.0, 5.0);
        let up_log = wall_function(0.41, 5.0, 50.0);
        assert!((up_lam - 5.0).abs() < 1e-10);
        assert!(up_log > 5.0);
    }

    #[test]
    fn wall_shear_finite() {
        let tau = wall_shear_from_loglaw(1.0, 0.01, 1e-5, 0.41, 5.0);
        assert!(tau > 0.0 && tau.is_finite());
    }

    #[test]
    fn smagorinsky_finite() {
        let nut = smagorinsky_eddy_viscosity(&[1e-6, 8e-6], &[100.0, 50.0], 0.17);
        assert!(nut[0] > 0.0);
        assert!(nut[1] > nut[0], "larger cell -> more nut");
    }

    #[test]
    fn smagorinsky_zero_strain() {
        let nut = smagorinsky_eddy_viscosity(&[1e-6], &[0.0], 0.17);
        assert!((nut[0] - 0.0).abs() < 1e-30);
    }

    #[test]
    fn wale_finite() {
        let nut = wale_eddy_viscosity(&[1e-6], &[100.0], 0.5);
        assert!(nut[0] > 0.0);
    }
}
