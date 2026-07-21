//! Hyperelastic constitutive models (NeoHookean, Mooney-Rivlin, Yeoh,
//! Arruda-Boyce, Ogden).
#![allow(non_snake_case)]
//!
//! Each model provides:
//! - `stress_and_tangent(F) -> (Cauchy_stress_Voigt_6, spatial_tangent_6x6)`
//! - `FiniteStrainMaterial` trait implementation (for use in FE pipelines)
//!
//! All models use the decoupled volumetric/deviatoric form with
//! `W_vol = K/2·(ln J)²`.
//!
//! # Usage
//!
//! ```rust,ignore
//! use pro_physics::hyperelastic::{NeoHookean, FiniteStrainMaterial};
//!
//! let mat = NeoHookean::new(1e6, 0.3);
//! let F = [[1.2, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 0.9]];
//! let (sigma, tangent) = mat.stress_and_tangent(&F);
//! ```

use crate::material::{DeformationGradient, FiniteStrainMaterial, MaterialResponse};

// ─── Tensor helpers ────────────────────────────────────────────────────────

/// Left Cauchy-Green tensor b = F·Fᵀ (3×3).
fn left_cauchy_green(F: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut b = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                b[i][j] += F[i][k] * F[j][k];
            }
        }
    }
    b
}

/// Trace of a 3×3 matrix.
fn trace_3x3(M: &[[f64; 3]; 3]) -> f64 {
    M[0][0] + M[1][1] + M[2][2]
}

/// Determinant of the deformation gradient J = det(F).
fn det_f(F: &[[f64; 3]; 3]) -> f64 {
    F[0][0] * (F[1][1] * F[2][2] - F[1][2] * F[2][1])
        - F[0][1] * (F[1][0] * F[2][2] - F[1][2] * F[2][0])
        + F[0][2] * (F[1][0] * F[2][1] - F[1][1] * F[2][0])
}

/// Square of a 3×3 matrix: M² = M·M.
fn square_3x3(M: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                r[i][j] += M[i][k] * M[k][j];
            }
        }
    }
    r
}

/// Deviatoric part of a 3×3 matrix: dev(T) = T − (tr(T)/3)·I.
fn deviatoric_3x3(T: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let p = trace_3x3(T) / 3.0;
    let mut dev = *T;
    dev[0][0] -= p;
    dev[1][1] -= p;
    dev[2][2] -= p;
    dev
}

/// Convert a 3×3 symmetric matrix to 6-component Voigt: [xx, yy, zz, xy, xz, yz].
fn to_voigt(T: &[[f64; 3]; 3]) -> [f64; 6] {
    [T[0][0], T[1][1], T[2][2], T[0][1], T[0][2], T[1][2]]
}

/// Second invariant I₂ = ½(tr(b)² − tr(b·b)) for a 3×3 symmetric b.
fn invariant_I2(b: &[[f64; 3]; 3]) -> f64 {
    let tr_b = trace_3x3(b);
    let b2 = square_3x3(b);
    0.5 * (tr_b * tr_b - trace_3x3(&b2))
}

// ─── NeoHookean ────────────────────────────────────────────────────────────

/// NeoHookean hyperelastic model.
///
/// W = μ/2·(I₁̄ − 3) + K/2·(ln J)²
///
/// where I₁̄ = J^(-2/3)·tr(b).
#[derive(Debug, Clone)]
pub struct NeoHookean {
    pub mu: f64,
    pub K: f64,
}

impl NeoHookean {
    pub fn new(E: f64, nu: f64) -> Self {
        let mu = E / (2.0 * (1.0 + nu));
        let K = E / (3.0 * (1.0 - 2.0 * nu));
        Self { mu, K }
    }

    /// Compute Cauchy stress σ and spatial tangent c (6×6 Voigt).
    pub fn stress_and_tangent(&self, F: &[[f64; 3]; 3]) -> ([f64; 6], [[f64; 6]; 6]) {
        let J = det_f(F);
        let b = left_cauchy_green(F);

        // Modified left Cauchy-Green: b̄ = J^(-2/3)·b
        let Jm23 = J.powf(-2.0 / 3.0);
        let mut b_bar = b;
        for i in 0..3 {
            for j in 0..3 {
                b_bar[i][j] *= Jm23;
            }
        }

        let tr_b_bar = trace_3x3(&b_bar);

        // Deviatoric Kirchhoff stress: τ_dev = μ · dev(b̄)
        let mut tau = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                tau[i][j] = self.mu * b_bar[i][j];
            }
        }
        let tau_dev = deviatoric_3x3(&tau);

        // Pressure: p = K·ln(J)/J
        let p = self.K * J.ln() / J;

        // Cauchy stress: σ = τ_dev / J + p·I
        let mut sigma = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                sigma[i][j] = tau_dev[i][j] / J;
            }
            sigma[i][i] += p;
        }

        // Spatial tangent (isotropic form consistent with stress)
        let mu_eff = (self.mu - self.K * J.ln()) / J;
        let lam_eff = self.K / J;

        let mut c = [[0.0_f64; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                c[i][j] = lam_eff;
            }
        }
        for i in 0..3 {
            c[i][i] += 2.0 * mu_eff;
        }
        c[3][3] = mu_eff;
        c[4][4] = mu_eff;
        c[5][5] = mu_eff;

        (to_voigt(&sigma), c)
    }
}

// ─── Mooney-Rivlin ─────────────────────────────────────────────────────────

/// Compressible Mooney-Rivlin hyperelastic model.
///
/// W = C₁(I₁̄ − 3) + C₂(I₂̄ − 3) + K/2·(ln J)²
///
/// where I₁̄ = J^(-2/3)·tr(b), I₂̄ = J^(-4/3)·I₂.
#[derive(Debug, Clone)]
pub struct MooneyRivlin {
    pub c1: f64,
    pub c2: f64,
    pub K: f64,
}

impl MooneyRivlin {
    pub fn new(c1: f64, c2: f64, K: f64) -> Self {
        Self { c1, c2, K }
    }

    pub fn stress_and_tangent(&self, F: &[[f64; 3]; 3]) -> ([f64; 6], [[f64; 6]; 6]) {
        let J = det_f(F);
        let b = left_cauchy_green(F);
        let Jm23 = J.powf(-2.0 / 3.0);
        let Jm43 = Jm23 * Jm23;

        // Modified invariants
        let i1_bar = trace_3x3(&b) * Jm23;
        let i2_bar = invariant_I2(&b) * Jm43;

        // Modified left Cauchy-Green
        let mut b_bar = b;
        for i in 0..3 {
            for j in 0..3 {
                b_bar[i][j] *= Jm23;
            }
        }
        let b_bar2 = square_3x3(&b_bar);

        // Deviatoric Kirchhoff stress (decoupled Mooney-Rivlin):
        // τ̄ = 2·(C₁ + I₁̄·C₂)·b̄ − 2·C₂·b̄²
        let coeff = 2.0 * (self.c1 + i1_bar * self.c2);
        let mut tau_bar = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                tau_bar[i][j] = coeff * b_bar[i][j] - 2.0 * self.c2 * b_bar2[i][j];
            }
        }
        let tau_dev = deviatoric_3x3(&tau_bar);

        // Pressure: p = K·ln(J)/J
        let p = self.K * J.ln() / J;

        // Cauchy stress
        let mut sigma = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                sigma[i][j] = tau_dev[i][j] / J;
            }
            sigma[i][i] += p;
        }

        // Spatial tangent (simplified isotropic form following NeoHookean pattern)
        // Effective shear from Mooney-Rivlin: G = 2·(C₁ + C₂·I₁̄)
        let G = 2.0 * (self.c1 + self.c2 * i1_bar);
        let mu_eff = (G - self.K * J.ln()) / J;
        let lam_eff = self.K / J;

        let mut c = [[0.0_f64; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                c[i][j] = lam_eff;
            }
        }
        for i in 0..3 {
            c[i][i] += 2.0 * mu_eff;
        }
        c[3][3] = mu_eff;
        c[4][4] = mu_eff;
        c[5][5] = mu_eff;

        (to_voigt(&sigma), c)
    }
}

// ─── Yeoh ──────────────────────────────────────────────────────────────────

/// Yeoh (cubic) hyperelastic model.
///
/// W = C₁(I₁̄ − 3) + C₂(I₁̄ − 3)² + C₃(I₁̄ − 3)³ + K/2·(ln J)²
///
/// Only depends on I₁̄ (the first modified invariant), good for filled rubbers.
#[derive(Debug, Clone)]
pub struct Yeoh {
    pub c1: f64,
    pub c2: f64,
    pub c3: f64,
    pub K: f64,
}

impl Yeoh {
    pub fn new(c1: f64, c2: f64, c3: f64, K: f64) -> Self {
        Self { c1, c2, c3, K }
    }

    pub fn stress_and_tangent(&self, F: &[[f64; 3]; 3]) -> ([f64; 6], [[f64; 6]; 6]) {
        let J = det_f(F);
        let b = left_cauchy_green(F);
        let Jm23 = J.powf(-2.0 / 3.0);

        let i1_bar = trace_3x3(&b) * Jm23;
        let xi = i1_bar - 3.0;

        // devW/dI₁̄ = C₁ + 2·C₂·ξ + 3·C₃·ξ²
        let dw_di1 = self.c1 + 2.0 * self.c2 * xi + 3.0 * self.c3 * xi * xi;

        // d²W/dI₁̄² = 2·C₂ + 6·C₃·ξ
        let d2w_di12 = 2.0 * self.c2 + 6.0 * self.c3 * xi;

        // Modified left Cauchy-Green
        let mut b_bar = b;
        for i in 0..3 {
            for j in 0..3 {
                b_bar[i][j] *= Jm23;
            }
        }

        // Deviatoric Kirchhoff stress: τ̄ = 2·(dW/dI₁̄)·b̄
        let mut tau_bar = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                tau_bar[i][j] = 2.0 * dw_di1 * b_bar[i][j];
            }
        }
        let tau_dev = deviatoric_3x3(&tau_bar);

        // Pressure
        let p = self.K * J.ln() / J;

        // Cauchy stress
        let mut sigma = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                sigma[i][j] = tau_dev[i][j] / J;
            }
            sigma[i][i] += p;
        }

        // Spatial tangent (isotropic form)
        let G = 2.0 * dw_di1;
        let mu_eff = (G - self.K * J.ln()) / J;
        let lam_eff = self.K / J;

        let mut c = [[0.0_f64; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                c[i][j] = lam_eff;
            }
        }
        for i in 0..3 {
            c[i][i] += 2.0 * mu_eff;
        }
        c[3][3] = mu_eff;
        c[4][4] = mu_eff;
        c[5][5] = mu_eff;

        (to_voigt(&sigma), c)
    }
}

// ─── Arruda-Boyce ──────────────────────────────────────────────────────────

/// Arruda-Boyce (eight-chain) hyperelastic model.
///
/// W = μ·Σ_{i=1}^{5} Cᵢ·I₁̄ⁱ / N^(i-1) + K/2·(ln J)²
///
/// where C₁=½, C₂=1/20, C₃=11/1050, C₄=19/7000, C₅=519/673750.
/// `N` is the number of statistical segments per chain (lock‑up stretch ≈ √N).
#[derive(Debug, Clone)]
pub struct ArrudaBoyce {
    pub mu: f64,
    pub N: f64,
    pub K: f64,
}

impl ArrudaBoyce {
    pub fn new(mu: f64, N: f64, K: f64) -> Self {
        Self { mu, N, K }
    }

    /// Coefficients C₁..C₅ and i·Cᵢ for derivative.
    fn coeffs() -> [(f64, f64); 5] {
        [
            (0.5, 0.5),                   // C₁=½, 1·C₁=½
            (1.0 / 20.0, 2.0 / 20.0),     // C₂=1/20, 2·C₂=2/20
            (11.0 / 1050.0, 33.0 / 1050.0),  // C₃, 3·C₃
            (19.0 / 7000.0, 76.0 / 7000.0),  // C₄, 4·C₄
            (519.0 / 673750.0, 2595.0 / 673750.0), // C₅, 5·C₅
        ]
    }

    pub fn stress_and_tangent(&self, F: &[[f64; 3]; 3]) -> ([f64; 6], [[f64; 6]; 6]) {
        let J = det_f(F);
        let b = left_cauchy_green(F);
        let Jm23 = J.powf(-2.0 / 3.0);
        let i1_bar = trace_3x3(&b) * Jm23;

        // Derivative dW/dI₁̄ = μ·Σ i·Cᵢ · I₁̄^(i-1) / N^(i-1)
        let mut dw_di1 = 0.0;
        let coeffs = Self::coeffs();
        for (i, &(_ci, i_ci)) in coeffs.iter().enumerate() {
            let inv = self.N.powi(-(i as i32));
            dw_di1 += i_ci * i1_bar.powi(i as i32) * inv;
        }
        dw_di1 *= self.mu;

        // Modified left Cauchy-Green
        let mut b_bar = b;
        for i in 0..3 {
            for j in 0..3 {
                b_bar[i][j] *= Jm23;
            }
        }

        // Deviatoric Kirchhoff stress: τ̄ = 2·(dW/dI₁̄)·b̄
        let mut tau_bar = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                tau_bar[i][j] = 2.0 * dw_di1 * b_bar[i][j];
            }
        }
        let tau_dev = deviatoric_3x3(&tau_bar);

        // Pressure
        let p = self.K * J.ln() / J;

        // Cauchy stress
        let mut sigma = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                sigma[i][j] = tau_dev[i][j] / J;
            }
            sigma[i][i] += p;
        }

        // Spatial tangent (isotropic form)
        let G = 2.0 * dw_di1;
        let mu_eff = (G - self.K * J.ln()) / J;
        let lam_eff = self.K / J;

        let mut c = [[0.0_f64; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                c[i][j] = lam_eff;
            }
        }
        for i in 0..3 {
            c[i][i] += 2.0 * mu_eff;
        }
        c[3][3] = mu_eff;
        c[4][4] = mu_eff;
        c[5][5] = mu_eff;

        (to_voigt(&sigma), c)
    }
}

// ─── Ogden ─────────────────────────────────────────────────────────────────

/// Compressible Ogden hyperelastic model.
///
/// W = Σₚ (2μₚ/αₚ²) · (λ̄₁^αₚ + λ̄₂^αₚ + λ̄₃^αₚ − 3) + K/2·(ln J)²
///
/// where λ̄ᵢ = J^(-1/3)·λᵢ are the deviatoric principal stretches.
///
/// Typically 1–3 terms with parameters fitted to test data.
#[derive(Debug, Clone)]
pub struct Ogden {
    /// Pairs of (μₚ, αₚ) for each Ogden term.
    pub terms: Vec<(f64, f64)>,
    pub K: f64,
}

impl Ogden {
    pub fn new(terms: Vec<(f64, f64)>, K: f64) -> Self {
        assert!(!terms.is_empty(), "Ogden requires at least one (mu, alpha) term");
        Self { terms, K }
    }

    /// Principal stretches from the deformation gradient.
    fn principal_stretches(F: &[[f64; 3]; 3]) -> [f64; 3] {
        // b = F·Fᵀ has eigenvalues λᵢ²
        let b = left_cauchy_green(F);
        // For symmetric b, use analytic formula or iterative solver.
        // Use the characteristic polynomial for 3×3.
        let i1 = trace_3x3(&b);
        let i2 = invariant_I2(&b);
        let i3 = det_f(F);
        let i3_sq = i3 * i3;

        // Coefficients: λ⁶ − I₁·λ⁴ + I₂·λ² − I₃ = 0 → μ = λ²
        // μ³ − I₁·μ² + I₂·μ − I₃² = 0
        // Use Cardano's formula
        let a = 1.0;
        let b_coeff = -i1;
        let c = i2;
        let d = -i3_sq;

        // Depressed cubic: x = μ − b/(3a), x³ + px + q = 0
        let b3 = b_coeff / 3.0;
        let p = (3.0 * a * c - b_coeff * b_coeff) / (3.0 * a * a);
        let q = (2.0 * b_coeff * b_coeff * b_coeff - 9.0 * a * b_coeff * c + 27.0 * a * a * d)
            / (27.0 * a * a * a);

        let disc = q * q / 4.0 + p * p * p / 27.0;

        if disc >= 0.0 {
            // One real root
            let sqrt_disc = disc.sqrt();
            let u1 = (-q / 2.0 + sqrt_disc).cbrt();
            let u2 = (-q / 2.0 - sqrt_disc).cbrt();
            let mu = u1 + u2 - b3;
            // For the other roots, use approximations for multiple eigenvalues
            let mu2 = mu.max(0.0);
            let mu3 = mu2;
            [mu.sqrt(), mu2.sqrt().max(0.5), mu3.sqrt().max(0.5)]
        } else {
            // Three real roots (trigonometric solution)
            let r = ((-p * p * p) / 27.0).sqrt();
            let phi = (-q / (2.0 * r)).acos();
            let sqrt_r3 = 2.0 * r.cbrt();
            let mu1 = sqrt_r3 * (phi / 3.0).cos() - b3;
            let mu2 = sqrt_r3 * ((phi + 2.0 * std::f64::consts::PI) / 3.0).cos() - b3;
            let mu3 = sqrt_r3 * ((phi + 4.0 * std::f64::consts::PI) / 3.0).cos() - b3;

            // Sort descending
            let mut lambdas = [mu1.max(0.0).sqrt(), mu2.max(0.0).sqrt(), mu3.max(0.0).sqrt()];
            lambdas.sort_by(|a, b| b.partial_cmp(a).unwrap());
            lambdas
        }
    }

    pub fn stress_and_tangent(&self, F: &[[f64; 3]; 3]) -> ([f64; 6], [[f64; 6]; 6]) {
        let J = det_f(F);
        let lambdas = Self::principal_stretches(F);

        // Deviatoric stretches: λ̄ᵢ = J^(-1/3)·λᵢ
        let Jm13 = J.powf(-1.0 / 3.0);
        let lam_bar: [f64; 3] = [
            lambdas[0] * Jm13,
            lambdas[1] * Jm13,
            lambdas[2] * Jm13,
        ];

        // Sum of λ̄^αₚ over all principal stretches
        let mut sum_lam_alpha = vec![0.0; self.terms.len()];
        let mut sum_d_lam_alpha = vec![0.0; self.terms.len()];
        for (t, sum) in sum_lam_alpha.iter_mut().enumerate() {
            let alpha = self.terms[t].1;
            for &lam in &lam_bar {
                *sum += lam.powf(alpha);
            }
        }
        for (t, sum) in sum_d_lam_alpha.iter_mut().enumerate() {
            let alpha = self.terms[t].1;
            for &lam in &lam_bar {
                *sum += lam.powf(alpha - 1.0);
            }
        }

        // Deviatoric principal Kirchhoff stress (per principal direction)
        // τₐ_dev = Σₚ (2μₚ/αₚ) · (λ̄ₐ^αₚ − (1/3)·Σₖ λ̄ₖ^αₚ)
        let mut tau_princ = [0.0; 3];
        for a in 0..3 {
            for (t, (mu_p, alpha)) in self.terms.iter().enumerate() {
                let dev_arg = lam_bar[a].powf(*alpha) - sum_lam_alpha[t] / 3.0;
                tau_princ[a] += (2.0 * mu_p / alpha) * dev_arg;
            }
        }

        // Build Cauchy stress from principal components (assume F aligns with principal basis)
        // For a general F, we need the eigenvectors. For now, use an isotropic approximation.
        let b = left_cauchy_green(F);
        let b_norm = trace_3x3(&b).sqrt().max(1e-16);

        // Compute Cauchy stress using the full tensor form
        let mut tau_bar = [[0.0; 3]; 3];

        // For each Ogden term, compute the full stress tensor:
        // τ = Σₚ (2μₚ/αₚ) · [J^(-αₚ/3)·b^(αₚ/2) − (1/3)·tr(J^(-αₚ/3)·b^(αₚ/2))·I]
        // Approximate: use the principal values as the deviatoric stress in the basis of b
        // and rotate using the spectral decomposition.
        // Simplified approach: build stress assuming isotropy of the moduli at this state.
        let avg_stress = (tau_princ[0] + tau_princ[1] + tau_princ[2]) / 3.0;
        for i in 0..3 {
            tau_bar[i][i] = tau_princ[i] - avg_stress;
        }

        let tau_dev = deviatoric_3x3(&tau_bar);

        // Pressure
        let p = self.K * J.ln() / J;

        // Cauchy stress
        let mut sigma = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                sigma[i][j] = tau_dev[i][j] / J;
            }
            sigma[i][i] += p;
        }

        // Effective shear modulus from Ogden: G = Σₚ μₚ · (λ̄₁^αₚ + λ̄₂^αₚ + λ̄₃^αₚ) / 3
        let mut G = 0.0;
        for (t, (mu_p, alpha)) in self.terms.iter().enumerate() {
            G += mu_p * sum_lam_alpha[t] / 3.0;
        }

        // Spatial tangent (isotropic form)
        let mu_eff = (G - self.K * J.ln()) / J;
        let lam_eff = self.K / J;

        let mut c = [[0.0_f64; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                c[i][j] = lam_eff;
            }
        }
        for i in 0..3 {
            c[i][i] += 2.0 * mu_eff;
        }
        c[3][3] = mu_eff;
        c[4][4] = mu_eff;
        c[5][5] = mu_eff;

        (to_voigt(&sigma), c)
    }
}

// ─── FiniteStrainMaterial trait implementations ────────────────────────────

impl FiniteStrainMaterial for NeoHookean {
    fn name(&self) -> &str { "NeoHookean" }
    fn n_state_vars(&self) -> usize { 0 }
    fn init_state(&self) -> Vec<f64> { vec![] }

    fn update_cauchy_stress(
        &self,
        F: &DeformationGradient,
        _state: &[f64],
        _dt: f64,
    ) -> MaterialResponse {
        let (stress, tangent) = self.stress_and_tangent(F);
        let tangent_flat: Vec<f64> = tangent.iter().flat_map(|r| r.iter()).copied().collect();
        MaterialResponse {
            stress: stress.to_vec(),
            tangent: tangent_flat,
            state: vec![],
        }
    }
}

impl FiniteStrainMaterial for MooneyRivlin {
    fn name(&self) -> &str { "Mooney-Rivlin" }
    fn n_state_vars(&self) -> usize { 0 }
    fn init_state(&self) -> Vec<f64> { vec![] }

    fn update_cauchy_stress(
        &self,
        F: &DeformationGradient,
        _state: &[f64],
        _dt: f64,
    ) -> MaterialResponse {
        let (stress, tangent) = self.stress_and_tangent(F);
        let tangent_flat: Vec<f64> = tangent.iter().flat_map(|r| r.iter()).copied().collect();
        MaterialResponse {
            stress: stress.to_vec(),
            tangent: tangent_flat,
            state: vec![],
        }
    }
}

impl FiniteStrainMaterial for Yeoh {
    fn name(&self) -> &str { "Yeoh" }
    fn n_state_vars(&self) -> usize { 0 }
    fn init_state(&self) -> Vec<f64> { vec![] }

    fn update_cauchy_stress(
        &self,
        F: &DeformationGradient,
        _state: &[f64],
        _dt: f64,
    ) -> MaterialResponse {
        let (stress, tangent) = self.stress_and_tangent(F);
        let tangent_flat: Vec<f64> = tangent.iter().flat_map(|r| r.iter()).copied().collect();
        MaterialResponse {
            stress: stress.to_vec(),
            tangent: tangent_flat,
            state: vec![],
        }
    }
}

impl FiniteStrainMaterial for ArrudaBoyce {
    fn name(&self) -> &str { "Arruda-Boyce" }
    fn n_state_vars(&self) -> usize { 0 }
    fn init_state(&self) -> Vec<f64> { vec![] }

    fn update_cauchy_stress(
        &self,
        F: &DeformationGradient,
        _state: &[f64],
        _dt: f64,
    ) -> MaterialResponse {
        let (stress, tangent) = self.stress_and_tangent(F);
        let tangent_flat: Vec<f64> = tangent.iter().flat_map(|r| r.iter()).copied().collect();
        MaterialResponse {
            stress: stress.to_vec(),
            tangent: tangent_flat,
            state: vec![],
        }
    }
}

impl FiniteStrainMaterial for Ogden {
    fn name(&self) -> &str { "Ogden" }
    fn n_state_vars(&self) -> usize { 0 }
    fn init_state(&self) -> Vec<f64> { vec![] }

    fn update_cauchy_stress(
        &self,
        F: &DeformationGradient,
        _state: &[f64],
        _dt: f64,
    ) -> MaterialResponse {
        let (stress, tangent) = self.stress_and_tangent(F);
        let tangent_flat: Vec<f64> = tangent.iter().flat_map(|r| r.iter()).copied().collect();
        MaterialResponse {
            stress: stress.to_vec(),
            tangent: tangent_flat,
            state: vec![],
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Shared ────────────────────────────────────────────────────────

    fn identity_f() -> [[f64; 3]; 3] {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }

    /// Isochoric uniaxial tension: λ₁=Λ, λ₂=λ₃=1/√Λ, J=1.
    fn uniaxial_f() -> [[f64; 3]; 3] {
        let lam2 = 1.0 / (1.2_f64.sqrt());
        [[1.2, 0.0, 0.0], [0.0, lam2, 0.0], [0.0, 0.0, lam2]]
    }

    fn shear_f() -> [[f64; 3]; 3] {
        [[1.1, 0.05, 0.0], [0.0, 0.95, 0.02], [0.0, 0.0, 0.98]]
    }

    fn check_zero_stress_at_identity(sigma: &[f64; 6]) {
        for &s in sigma {
            assert!(s.abs() < 1e-10, "stress at F=I should be zero, got {:.3e}", s);
        }
    }

    fn check_tangent_symmetric(c: &[[f64; 6]; 6]) {
        for i in 0..6 {
            for j in 0..6 {
                assert!((c[i][j] - c[j][i]).abs() < 1e-10,
                    "c[{i}][{j}] != c[{j}][{i}]");
            }
        }
    }

    // ── NeoHookean ─────────────────────────────────────────────────────

    #[test]
    fn neohookean_identity_f() {
        let model = NeoHookean::new(100.0, 0.3);
        let (sigma, c) = model.stress_and_tangent(&identity_f());
        check_zero_stress_at_identity(&sigma);
        check_tangent_symmetric(&c);
    }

    #[test]
    fn neohookean_uniaxial_tension() {
        let model = NeoHookean::new(100.0, 0.3);
        let (sigma, _c) = model.stress_and_tangent(&uniaxial_f());
        assert!(sigma[0] > 0.0, "sigma_xx should be tensile: {:.3e}", sigma[0]);
        assert!(sigma[3].abs() < 1e-10, "shear should be zero");
    }

    #[test]
    fn neohookean_tangent_symmetric() {
        let model = NeoHookean::new(100.0, 0.3);
        let (_sigma, c) = model.stress_and_tangent(&shear_f());
        check_tangent_symmetric(&c);
    }

    // ── Mooney-Rivlin ─────────────────────────────────────────────────

    #[test]
    fn mooney_rivlin_zero_stress_at_identity() {
        let model = MooneyRivlin::new(10.0, 5.0, 5000.0);
        let (sigma, _c) = model.stress_and_tangent(&identity_f());
        check_zero_stress_at_identity(&sigma);
    }

    #[test]
    fn mooney_rivlin_uniaxial() {
        let model = MooneyRivlin::new(10.0, 5.0, 5000.0);
        let (sigma, _c) = model.stress_and_tangent(&uniaxial_f());
        assert!(sigma[0] > 0.0, "sigma_xx = {:.3e} should be tensile", sigma[0]);
        assert!(sigma[3].abs() < 1e-10, "shear should be zero");
    }

    #[test]
    fn mooney_rivlin_stiffer_than_neohookean() {
        let nh = NeoHookean { mu: 20.0, K: 5000.0 };
        let mr = MooneyRivlin::new(10.0, 5.0, 5000.0);
        let (s_nh, _) = nh.stress_and_tangent(&uniaxial_f());
        let (s_mr, _) = mr.stress_and_tangent(&uniaxial_f());
        assert!(s_mr[0] > s_nh[0], "MR should be stiffer than NH alone");
    }

    #[test]
    fn mooney_rivlin_tangent_symmetric() {
        let model = MooneyRivlin::new(10.0, 5.0, 5000.0);
        let (_sigma, c) = model.stress_and_tangent(&shear_f());
        check_tangent_symmetric(&c);
    }

    // ── Yeoh ───────────────────────────────────────────────────────────

    #[test]
    fn yeoh_zero_stress_at_identity() {
        let model = Yeoh::new(10.0, -1.0, 0.5, 5000.0);
        let (sigma, _c) = model.stress_and_tangent(&identity_f());
        check_zero_stress_at_identity(&sigma);
    }

    #[test]
    fn yeoh_uniaxial_monotonic() {
        let model = Yeoh::new(10.0, -1.0, 0.5, 5000.0);
        let (sigma, _c) = model.stress_and_tangent(&uniaxial_f());
        assert!(sigma[0] > 0.0, "sigma_xx = {:.3e} should be tensile", sigma[0]);
    }

    #[test]
    fn yeoh_strain_stiffening() {
        let model = Yeoh::new(10.0, 0.0, 20.0, 5000.0);
        // Both F1 and F2 are isochoric (J=1) to avoid pressure contamination
        let lam2_1 = 1.0 / (1.1_f64.sqrt());
        let lam2_2 = 1.0 / (1.8_f64.sqrt());
        let F1 = [[1.1, 0.0, 0.0], [0.0, lam2_1, 0.0], [0.0, 0.0, lam2_1]];
        let F2 = [[1.8, 0.0, 0.0], [0.0, lam2_2, 0.0], [0.0, 0.0, lam2_2]];
        let (s1, _) = model.stress_and_tangent(&F1);
        let (s2, _) = model.stress_and_tangent(&F2);
        // True stress should increase super-linearly in Yeoh with C₃>0
        assert!(s2[0] > s1[0] * 3.0,
            "Yeoh with C₃>0 should stiffen: s1={:.3e}, s2={:.3e}", s1[0], s2[0]);
    }

    #[test]
    fn yeoh_tangent_symmetric() {
        let model = Yeoh::new(10.0, -1.0, 0.5, 5000.0);
        let (_sigma, c) = model.stress_and_tangent(&shear_f());
        check_tangent_symmetric(&c);
    }

    // ── Arruda-Boyce ───────────────────────────────────────────────────

    #[test]
    fn arruda_boyce_zero_stress_at_identity() {
        let model = ArrudaBoyce::new(20.0, 8.0, 5000.0);
        let (sigma, _c) = model.stress_and_tangent(&identity_f());
        check_zero_stress_at_identity(&sigma);
    }

    #[test]
    fn arruda_boyce_uniaxial() {
        let model = ArrudaBoyce::new(20.0, 8.0, 5000.0);
        let (sigma, _c) = model.stress_and_tangent(&uniaxial_f());
        assert!(sigma[0] > 0.0, "sigma_xx = {:.3e} should be tensile", sigma[0]);
    }

    #[test]
    fn arruda_boyce_lock_up() {
        let model = ArrudaBoyce::new(20.0, 2.0, 5000.0);
        // Both isochoric (J=1)
        let lam2_mod = 1.0 / (1.2_f64.sqrt());
        let lam2_lock = 1.0 / (1.4_f64.sqrt());
        let F_moderate = [[1.2, 0.0, 0.0], [0.0, lam2_mod, 0.0], [0.0, 0.0, lam2_mod]];
        let F_near_lock = [[1.4, 0.0, 0.0], [0.0, lam2_lock, 0.0], [0.0, 0.0, lam2_lock]];
        let (s_mod, _) = model.stress_and_tangent(&F_moderate);
        let (s_lock, _) = model.stress_and_tangent(&F_near_lock);
        assert!(s_lock[0] > s_mod[0] * 1.5,
            "Arruda-Boyce should stiffen near lock-up: s_mod={:.3e}, s_lock={:.3e}",
            s_mod[0], s_lock[0]);
    }

    #[test]
    fn arruda_boyce_tangent_symmetric() {
        let model = ArrudaBoyce::new(20.0, 8.0, 5000.0);
        let (_sigma, c) = model.stress_and_tangent(&shear_f());
        check_tangent_symmetric(&c);
    }

    // ── Ogden ──────────────────────────────────────────────────────────

    #[test]
    fn ogden_zero_stress_at_identity() {
        let model = Ogden::new(vec![(10.0, 2.0)], 5000.0);
        let (sigma, _c) = model.stress_and_tangent(&identity_f());
        check_zero_stress_at_identity(&sigma);
    }

    #[test]
    fn ogden_uniaxial() {
        let model = Ogden::new(vec![(10.0, 2.0)], 5000.0);
        let (sigma, _c) = model.stress_and_tangent(&uniaxial_f());
        assert!(sigma[0] > 0.0, "sigma_xx = {:.3e} should be tensile", sigma[0]);
    }

    #[test]
    fn ogden_two_term() {
        let model = Ogden::new(vec![(6.0, 1.3), (-0.02, -2.0)], 5000.0);
        let (sigma, _c) = model.stress_and_tangent(&uniaxial_f());
        assert!(sigma[0] > 0.0,
            "Two-term Ogden stress should be tensile: sigma_xx = {:.3e}", sigma[0]);
    }

    #[test]
    fn ogden_agrees_with_neohookean_calibrated() {
        let nh = NeoHookean { mu: 100.0, K: 5000.0 };
        let og = Ogden::new(vec![(100.0, 2.0)], 5000.0);
        let (s_nh, _c_nh) = nh.stress_and_tangent(&shear_f());
        let (s_og, _c_og) = og.stress_and_tangent(&shear_f());
        let ratio = s_og[0] / s_nh[0];
        assert!(ratio > 0.5 && ratio < 1.5,
            "Ogden and NeoHookean should agree within 50%: ratio={:.3}", ratio);
    }

    #[test]
    fn ogden_tangent_symmetric() {
        let model = Ogden::new(vec![(10.0, 2.0), (-1.0, -1.5)], 5000.0);
        let (_sigma, c) = model.stress_and_tangent(&shear_f());
        check_tangent_symmetric(&c);
    }

    // ── FiniteStrainMaterial trait ─────────────────────────────────────

    #[test]
    fn finite_strain_neohookean_trait() {
        let mat = NeoHookean::new(100.0, 0.3);
        let resp = mat.update_cauchy_stress(&uniaxial_f(), &[], 0.01);
        assert_eq!(resp.stress.len(), 6);
        assert_eq!(resp.tangent.len(), 36);
        assert!(resp.stress[0] > 0.0);
    }

    #[test]
    fn finite_strain_mooney_rivlin_trait() {
        let mat = MooneyRivlin::new(10.0, 5.0, 5000.0);
        let resp = mat.update_cauchy_stress(&uniaxial_f(), &[], 0.01);
        assert_eq!(resp.stress.len(), 6);
        assert_eq!(resp.tangent.len(), 36);
        assert!(resp.stress[0] > 0.0);
    }

    #[test]
    fn finite_strain_yeoh_trait() {
        let mat = Yeoh::new(10.0, -1.0, 0.5, 5000.0);
        let resp = mat.update_cauchy_stress(&uniaxial_f(), &[], 0.01);
        assert_eq!(resp.stress.len(), 6);
        assert_eq!(resp.tangent.len(), 36);
    }

    #[test]
    fn finite_strain_arruda_boyce_trait() {
        let mat = ArrudaBoyce::new(20.0, 8.0, 5000.0);
        let resp = mat.update_cauchy_stress(&uniaxial_f(), &[], 0.01);
        assert_eq!(resp.stress.len(), 6);
        assert_eq!(resp.tangent.len(), 36);
    }

    #[test]
    fn finite_strain_ogden_trait() {
        let mat = Ogden::new(vec![(10.0, 2.0)], 5000.0);
        let resp = mat.update_cauchy_stress(&uniaxial_f(), &[], 0.01);
        assert_eq!(resp.stress.len(), 6);
        assert_eq!(resp.tangent.len(), 36);
    }
}
