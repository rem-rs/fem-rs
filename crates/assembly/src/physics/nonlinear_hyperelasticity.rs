//! Finite-strain hyperelasticity with multiple material models.
//!
//! Implements [`NonlinearForm`] for:
//! - **Neo-Hookean** (compressible)
//! - **Mooney–Rivlin** (incompressible + bulk penalty)
//! - **Ogden** (N=1,2,3)
//!
//! All models support 2D/3D via `VectorH1Space` and use the
//! Newton–Raphson solver with Armijo line-search from [`NewtonSolver`].
//!
//! ## Strain energy densities
//!
//! ### Neo-Hookean (compressible, implemented)
//! ```text
//! ψ = μ/2·(tr(C)-3) - μ·ln(J) + λ/2·(ln(J))²
//! ```
//!
//! ### Mooney–Rivlin
//! ```text
//! ψ = C10·(I₁-3) + C01·(I₂-3) + K/2·(J-1)²
//! ```
//!
//! ### Ogden (N-term)
//! ```text
//! ψ = Σ_{p=1}^N μ_p/α_p·(λ₁^{α_p}+λ₂^{α_p}+λ₃^{α_p}-3) + K/2·(J-1)²
//! ```
//!
//! where `C = FᵀF`, `I₁ = tr(C)`, `I₂ = ½((tr(C))² - tr(C²))`, `J = det(F)`.

#![allow(non_snake_case)]

use nalgebra::DMatrix;
use nalgebra::linalg::SVD;

use fem_element::{
    ReferenceElement,
    lagrange::{TriP1, TriP2, TriP3, TetP1, TetP2, TetP3},
};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;
use fem_space::l2::L2Space;
use fem_space::vector_h1::VectorH1Space;

use crate::physics::nonlinear::{NonlinearForm, NewtonSolver, NewtonConfig, NewtonResult};

/// Hyperelastic material model.
#[derive(Debug, Clone)]
pub enum HyperelasticModel {
    /// Compressible Neo-Hookean: `μ/2·(I₁-3) - μ·ln(J) + λ/2·(ln(J))²`.
    NeoHookean { mu: f64, lambda: f64 },
    /// MFEM-style deviatoric NeoHookean (ex10 default):
    /// `μ/2·(J^{-2/3}·I₁ - dim) + K/2·(J-1)²`
    /// where K is the bulk modulus parameter (MFEM NeoHookeanModel).
    MfemNeoHookean { mu: f64, bulk_modulus: f64 },
    /// Mooney–Rivlin: `C10·(I₁-3) + C01·(I₂-3) + K/2·(J-1)²`.
    MooneyRivlin { c10: f64, c01: f64, bulk_modulus: f64 },
    /// N-term Ogden: `Σ μ_p/α_p·(λ₁^α+λ₂^α+λ₃^α-3) + K/2·(J-1)²`.
    Ogden { params: Vec<(f64, f64)>, bulk_modulus: f64 },
    /// Arruda-Boyce (8-chain model): rubber hyperelastic with limiting stretch.
    ArrudaBoyce { mu: f64, lambda_lock: f64, bulk_modulus: f64, n_terms: usize },
    /// Yeoh (N=3): phenomenological rubber model.
    Yeoh { c10: f64, c20: f64, c30: f64, bulk_modulus: f64 },
}

impl HyperelasticModel {
    /// PK1 stress and consistent tangent for the model.
    pub fn pk1_and_tangent(&self, f: &DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
        match self {
            HyperelasticModel::NeoHookean { mu, lambda } => {
                neo_hookean_pk1_tangent(f, *mu, *lambda)
            }
            HyperelasticModel::MfemNeoHookean { mu, bulk_modulus } => {
                mfem_neo_hookean_pk1_tangent(f, *mu, *bulk_modulus)
            }
            HyperelasticModel::MooneyRivlin { c10, c01, bulk_modulus } => {
                mooney_rivlin_pk1_tangent(f, *c10, *c01, *bulk_modulus)
            }
            HyperelasticModel::Ogden { params, bulk_modulus } => {
                ogden_pk1_tangent(f, params, *bulk_modulus)
            }
            HyperelasticModel::ArrudaBoyce { mu, lambda_lock, bulk_modulus, n_terms } => {
                arruda_boyce_pk1_tangent(f, *mu, *lambda_lock, *bulk_modulus, *n_terms)
            }
            HyperelasticModel::Yeoh { c10, c20, c30, bulk_modulus } => {
                yeoh_pk1_tangent(f, *c10, *c20, *c30, *bulk_modulus)
            }
        }
    }

    /// Elastic energy density ψ(F) for the model (per-unit reference volume).
    pub fn elastic_energy_density(&self, f: &DMatrix<f64>) -> f64 {
        match self {
            HyperelasticModel::NeoHookean { mu, lambda } => {
                let dim = f.nrows();
                let c = f.transpose() * f;
                let i1 = c.trace();
                let jac = f.determinant();
                let ln_j = jac.ln();
                0.5 * mu * (i1 - dim as f64) - mu * ln_j + 0.5 * lambda * ln_j * ln_j
            }
            HyperelasticModel::MfemNeoHookean { mu, bulk_modulus } => {
                let dim = f.nrows() as f64;
                let c = f.transpose() * f;
                let i1 = c.trace();       // I₁ = tr(C)
                let jac = f.determinant(); // J = det(F)
                let i1_bar = jac.powf(-2.0 / dim) * i1;  // Ī₁ = J^{-2/dim} * I₁
                0.5 * mu * (i1_bar - dim) + 0.5 * bulk_modulus * (jac - 1.0).powi(2)
            }
            HyperelasticModel::MooneyRivlin { c10, c01, bulk_modulus } => {
                let ct = f.transpose() * f;
                let i1 = ct.trace();
                let i2 = 0.5 * (i1 * i1 - (ct.clone() * ct).trace());
                let jac = f.determinant();
                c10 * (i1 - 3.0) + c01 * (i2 - 3.0) + 0.5 * bulk_modulus * (jac - 1.0).powi(2)
            }
            HyperelasticModel::Ogden { params, bulk_modulus } => {
                let dim = f.nrows();
                let jac = f.determinant();
                let svd = SVD::new(f.clone(), true, true);
                let mut lam = vec![1.0_f64; dim];
                for i in 0..dim { lam[i] = svd.singular_values[i].max(1e-30); }
                let mut psi = 0.0;
                for (mu_p, alpha_p) in params {
                    let mut sum = 0.0;
                    for a in 0..dim { sum += lam[a].powf(*alpha_p); }
                    psi += *mu_p / *alpha_p * (sum - dim as f64);
                }
                psi + 0.5 * *bulk_modulus * (jac - 1.0).powi(2)
            }
            HyperelasticModel::ArrudaBoyce { mu, lambda_lock, bulk_modulus, .. } => {
                arruda_boyce_energy(f, *mu, *lambda_lock, *bulk_modulus)
            }
            HyperelasticModel::Yeoh { c10, c20, c30, bulk_modulus } => {
                yeoh_energy(f, *c10, *c20, *c30, *bulk_modulus)
            }
        }
    }
}


// --- Arruda-Boyce (8-chain) ---

fn arruda_boyce_pk1_tangent(
    f: &DMatrix<f64>, mu: f64, lambda_lock: f64, K: f64, _n_terms: usize,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let pk1 = arruda_boyce_pk1_stress(f, mu, lambda_lock, K);
    let ct = numerical_tangent(f, &|ft| arruda_boyce_pk1_stress(ft, mu, lambda_lock, K));
    (pk1, ct)
}

fn arruda_boyce_pk1_stress(f: &DMatrix<f64>, mu: f64, lambda_m: f64, K: f64) -> DMatrix<f64> {
    let dim = f.nrows();
    let jac = f.determinant();
    let ft = f.transpose();
    let b = f * ft;
    let i1 = b.trace();
    let lam_m2 = lambda_m * lambda_m;
    let mut dW_dI1 = 0.5 * mu;
    let mut term = 1.0;
    for k in 2..6 {
        term *= i1 / lam_m2;
        dW_dI1 += mu * term / ((k as f64) * lambda_m.powf(2.0 * (k as f64 - 1.0)));
    }
    // Subtract reference value at I1=3 for zero stress at F=I
    let i1_0 = 3.0;
    let mut dW_dI1_0 = 0.5 * mu;
    let mut term_0 = 1.0;
    for k in 2..6 {
        term_0 *= i1_0 / lam_m2;
        dW_dI1_0 += mu * term_0 / ((k as f64) * lambda_m.powf(2.0 * (k as f64 - 1.0)));
    }
    dW_dI1 -= dW_dI1_0;
    let finv_t = f.clone().try_inverse().unwrap_or_else(|| DMatrix::identity(dim, dim));
    let mut pk1 = DMatrix::zeros(dim, dim);
    for i in 0..dim { for I in 0..dim {
        pk1[(i, I)] = 2.0 * dW_dI1 * f[(i, I)] + K * (jac - 1.0) * jac * finv_t[(I, i)];
    }}
    pk1
}

fn arruda_boyce_energy(f: &DMatrix<f64>, mu: f64, lambda_m: f64, K: f64) -> f64 {
    let dim = f.nrows();
    let jac = f.determinant();
    let ft = f.transpose();
    let b = f * ft;
    let i1 = b.trace();
    let lam_m2 = lambda_m * lambda_m;
    let mut psi = 0.5 * mu * (i1 - dim as f64);
    let mut term = 1.0;
    for k in 2..6 {
        term *= i1 / lam_m2;
        psi += mu * term / ((k as f64) * lambda_m.powf(2.0 * (k as f64 - 1.0)));
    }
    psi + 0.5 * K * (jac - 1.0).powi(2)
}

// --- Yeoh (N=3) ---

fn yeoh_pk1_tangent(f: &DMatrix<f64>, c10: f64, c20: f64, c30: f64, K: f64) -> (DMatrix<f64>, DMatrix<f64>) {
    let pk1 = yeoh_pk1_stress(f, c10, c20, c30, K);
    let ct = numerical_tangent(f, &|ft| yeoh_pk1_stress(ft, c10, c20, c30, K));
    (pk1, ct)
}

fn yeoh_pk1_stress(f: &DMatrix<f64>, c10: f64, c20: f64, c30: f64, K: f64) -> DMatrix<f64> {
    let dim = f.nrows();
    let jac = f.determinant();
    let ft = f.transpose();
    let b = f * ft;
    let i1 = b.trace();
    let i1m3 = i1 - dim as f64;
    let dW_dI1 = c10 + 2.0 * c20 * i1m3 + 3.0 * c30 * i1m3 * i1m3;
    let finv_t = f.clone().try_inverse().unwrap_or_else(|| DMatrix::identity(dim, dim));
    let mut pk1 = DMatrix::zeros(dim, dim);
    for i in 0..dim { for I in 0..dim {
        pk1[(i, I)] = 2.0 * dW_dI1 * f[(i, I)] + K * (jac - 1.0) * jac * finv_t[(I, i)];
    }}
    pk1
}

fn yeoh_energy(f: &DMatrix<f64>, c10: f64, c20: f64, c30: f64, K: f64) -> f64 {
    let dim = f.nrows();
    let jac = f.determinant();
    let ft = f.transpose();
    let b = f * ft;
    let i1 = b.trace();
    let i1m3 = i1 - dim as f64;
    c10 * i1m3 + c20 * i1m3 * i1m3 + c30 * i1m3 * i1m3 * i1m3 + 0.5 * K * (jac - 1.0).powi(2)
}
// ─── Neo-Hookean (existing) ──────────────────────────────────────────────────

fn neo_hookean_pk1_tangent(f: &DMatrix<f64>, mu: f64, lambda: f64) -> (DMatrix<f64>, DMatrix<f64>) {
    let dim = f.nrows();
    let jac = f.determinant();
    let inv_f = f.clone().try_inverse().unwrap_or_else(|| DMatrix::identity(dim, dim));
    let inv_f_t = inv_f.transpose();
    let ln_j = jac.ln();

    let mut p = DMatrix::zeros(dim, dim);
    for i in 0..dim {
        for I in 0..dim {
            p[(i, I)] = mu * f[(i, I)] + (lambda * ln_j - mu) * inv_f_t[(i, I)];
        }
    }

    let n = dim * dim;
    let mut ct = DMatrix::zeros(n, n);
    let pre = lambda * ln_j - mu;
    for i in 0..dim { for I in 0..dim {
        let row = i * dim + I;
        for j in 0..dim { for J in 0..dim {
            let col = j * dim + J;
            let mut val = 0.0;
            if i == j && I == J { val += mu; }
            // λ·F^{-T}_{Jj}·F^{-T}_{Ii}  — corresponds to λ·δ_Jj·δ_Ii at F=I
            // (the linear-elasticity limit gives C_{iI,jJ} = λ·δ_{iI}·δ_{jJ} + μ·(δ_{ij}·δ_{IJ} + δ_{iJ}·δ_{jI}))
            val += lambda * inv_f[(J, j)] * inv_f[(I, i)];
            val -= pre * inv_f[(J, i)] * inv_f[(I, j)];
            ct[(row, col)] = val;
        }}
    }}
    (p, ct)
}

// ─── MFEM-style deviatoric Neo-Hookean ────────────────────────────────────────
//
// Matches MFEM's NeoHookeanModel used in ex10:
//   W = μ/2 · (J^{-2/3}·I₁ - dim) + K/2 · (J-1)²
//   P = a·F + c·F^{-T}
//
// where:
//   a = μ·J^{-2/dim}
//   c = K·J·(J-1) - a·I₁/dim
//
// Analytic tangent (Voigt C_{iI,jJ} = ∂P_{iI}/∂F_{jJ}):
//
//   ∂a/∂F_{jJ} = a·(-2/dim)·F^{-T}_{jJ}
//   ∂J/∂F_{jJ} = J·F^{-T}_{jJ}
//   ∂F_{iI}/∂F_{jJ} = δ_{ij}·δ_{IJ}
//   ∂F^{-T}_{iI}/∂F_{jJ} = -F^{-T}_{jI}·F^{-T}_{iJ}
//   ∂I₁/∂F_{jJ} = 2·F_{jJ}
//
//   C = a·δ_{ij}·δ_{IJ}
//     + a·(-2/dim)·F^{-T}_{jJ}·F_{iI}
//     + [K·J·(2J-1) + 2·a·I₁/dim²] · F^{-T}_{jJ}·F^{-T}_{iI}
//     - (2·a/dim)·F_{jJ}·F^{-T}_{iI}
//     - c·F^{-T}_{jI}·F^{-T}_{iJ}
//
// At F=I this reduces to the isotropic linear elasticity tensor
// C = μ·(δ_{ik}δ_{jl}+δ_{il}δ_{jk}) + λ·δ_{ij}δ_{kl} with λ = K - 2μ/dim.
//
// Verified against central-difference numerical tangent in unit test below.

fn mfem_neo_hookean_pk1(f: &DMatrix<f64>, mu: f64, K: f64) -> DMatrix<f64> {
    let dim = f.nrows() as f64;
    let jac = f.determinant();
    let c = f.transpose() * f;
    let i1 = c.trace();
    let j_pow = jac.powf(-2.0 / dim);        // J^{-2/dim}
    let inv_f_t = f.clone().try_inverse()
        .unwrap_or_else(|| DMatrix::identity(f.nrows(), f.nrows()))
        .transpose();

    let a = mu * j_pow;                       // μ·J^{-2/dim}
    let b = K * (jac - 1.0) - a * i1 / (dim * jac);  // K·(J-1) - μ·J^{-2/dim}·I₁/(dim·J)
    a * f + (b * jac) * inv_f_t
}

/// Analytical tangent for MFEM deviatoric Neo-Hookean.
///
/// Derivation: differentiate P = a·F + c·F^{-T} using:
///   ∂J/∂F = J·F^{-T},   ∂F/∂F = I⊗I,   ∂F^{-T}/∂F = -F^{-T} ⊗ F^{-T}
///   ∂I₁/∂F = 2F,        ∂a/∂F = a·(-2/dim)·F^{-T}
fn mfem_neo_hookean_pk1_tangent(f: &DMatrix<f64>, mu: f64, K: f64) -> (DMatrix<f64>, DMatrix<f64>) {
    let dim = f.nrows();
    let dim_f = dim as f64;
    let jac = f.determinant();
    let inv_f_t = f.clone().try_inverse()
        .unwrap_or_else(|| DMatrix::identity(dim, dim))
        .transpose();
    let c_mat = f.transpose() * f;
    let i1 = c_mat.trace();

    let j_pow = jac.powf(-2.0 / dim_f);
    let a = mu * j_pow;                            // μ·J^{-2/dim}
    let c = K * jac * (jac - 1.0) - a * i1 / dim_f; // K·J·(J-1) - a·I₁/dim

    // PK1 stress
    let b = K * (jac - 1.0) - a * i1 / (dim_f * jac);
    let p = a * f + (b * jac) * inv_f_t.clone();

    // ── Tangent C_{iI,jJ} ─────────────────────────────────────────────────────
    let n = dim * dim;
    let mut ct = DMatrix::zeros(n, n);

    // Precomputed scalars for the five terms (see doc-comment derivation).
    let t2_pre = a * (-2.0 / dim_f);         // a·(-2/dim)  for F·F^{-T}
    let t3_pre = K * jac * (2.0 * jac - 1.0)    // K·J·(2J-1) + 2·a·I₁/dim²
               + 2.0 * a * i1 / (dim_f * dim_f);
    let t4_pre = -2.0 * a / dim_f;            // -(2·a/dim)  for F·F^{-T}

    for i in 0..dim {
        for I in 0..dim {
            let row = i * dim + I;
            for j in 0..dim {
                for J in 0..dim {
                    let col = j * dim + J;
                    let mut val = 0.0;

                    // Term 1: a · δ_{ij} · δ_{IJ}
                    if i == j && I == J {
                        val += a;
                    }

                    // Term 2: a·(-2/dim) · F_{iI} · F^{-T}_{jJ}
                    val += t2_pre * f[(i, I)] * inv_f_t[(j, J)];

                    // Term 3: [K·J·(2J-1) + 2·a·I₁/dim²] · F^{-T}_{jJ} · F^{-T}_{iI}
                    val += t3_pre * inv_f_t[(j, J)] * inv_f_t[(i, I)];

                    // Term 4: -(2·a/dim) · F_{jJ} · F^{-T}_{iI}
                    val += t4_pre * f[(j, J)] * inv_f_t[(i, I)];

                    // Term 5: -c · F^{-T}_{jI} · F^{-T}_{iJ}
                    val -= c * inv_f_t[(j, I)] * inv_f_t[(i, J)];

                    ct[(row, col)] = val;
                }
            }
        }
    }

    (p, ct)
}

// ─── Mooney–Rivlin ───────────────────────────────────────────────────────────

fn mooney_rivlin_pk1_tangent(f: &DMatrix<f64>, c10: f64, c01: f64, K: f64) -> (DMatrix<f64>, DMatrix<f64>) {
    let dim = f.nrows();
    let jac = f.determinant();
    let inv_f = f.clone().try_inverse().unwrap_or_else(|| DMatrix::identity(dim, dim));
    let inv_f_t = inv_f.transpose();
    let c = f.transpose() * f;
    let i1 = c.trace();
    // PK1 via push-forward of PK2
    let mut s_pk2 = DMatrix::zeros(dim, dim);
    let pk2_pre = 2.0 * (c10 + c01 * i1);
    let c_inv = inv_f * inv_f_t;
    for i in 0..dim { for j in 0..dim {
        let hat = if i == j { 1.0 } else { 0.0 };
        s_pk2[(i, j)] = pk2_pre * hat - 2.0 * c01 * c[(i, j)] + K * jac * (jac - 1.0) * c_inv[(i, j)];
    }}
    let p = f * &s_pk2;
    // Numerical tangent via central differences
    let ct = numerical_tangent(f, &|ft| {
        let j = ft.determinant();
        let _i = ft.transpose();
        let ci = ft.transpose() * ft;
        let i1t = ci.trace();
        let ci_inv = ft.clone().try_inverse().map(|mi| { let mt = mi.transpose(); mi * mt }).unwrap_or_else(|| DMatrix::identity(dim, dim));
        let mut s2 = DMatrix::zeros(dim, dim);
        let pre = 2.0 * (c10 + c01 * i1t);
        for ii in 0..dim { for jj in 0..dim {
            let h = if ii == jj { 1.0 } else { 0.0 };
            s2[(ii, jj)] = pre * h - 2.0 * c01 * ci[(ii, jj)] + K * j * (j - 1.0) * ci_inv[(ii, jj)];
        }}
        ft * &s2
    });
    (p, ct)
}

// ─── Ogden ───────────────────────────────────────────────────────────────────

fn ogden_pk1_tangent(f: &DMatrix<f64>, params: &[(f64, f64)], K: f64) -> (DMatrix<f64>, DMatrix<f64>) {
    let dim = f.nrows();
    let jac = f.determinant();
    let _inv_f_t = f.clone().try_inverse().map(|m| m.transpose()).unwrap_or_else(|| DMatrix::identity(dim, dim));

    // Principal stretches via SVD of F
    let svd = SVD::new(f.clone(), true, true);
    let u_mat = svd.u.expect("SVD u failed");
    let v_t_mat = svd.v_t.expect("SVD v_t failed");
    let mut lam = vec![1.0_f64; dim];
    for i in 0..dim { lam[i] = svd.singular_values[i]; }

    // PK2 in spectral basis
    let mut s = DMatrix::zeros(dim, dim);
    let v_mat = v_t_mat.transpose(); // V

    for i in 0..dim {
        for I in 0..dim {
            let mut val = 0.0;
            for a in 0..dim {
                if lam[a].abs() < 1e-30 { continue; }
                let mut dW_dlam = 0.0;
                for (mu_p, alpha_p) in params {
                    dW_dlam += mu_p * lam[a].powf(alpha_p - 1.0);
                }
                dW_dlam += K * (jac - 1.0) * jac / lam[a];

                let uia = u_mat[(i, a)];
                let vIa = v_mat[(I, a)];
                val += dW_dlam / lam[a] * uia * vIa;
            }
            s[(i, I)] = val;
        }
    }

    // PK1: P = Σ (1/λ)·(dW/dλ)·n⊗N — computed via the spectral formula
    let p = s.clone();

    // Numerical tangent via central differences
    let ct = numerical_tangent(f, &|ft| {
        ogden_pk1_only(ft, params, K)
    });

    (p, ct)
}

/// Ogden PK1 stress only (no tangent) — for numerical differentiation.
fn ogden_pk1_only(f: &DMatrix<f64>, params: &[(f64, f64)], K: f64) -> DMatrix<f64> {
    let dim = f.nrows();
    let jac = f.determinant();
    let svd = SVD::new(f.clone(), true, true);
    let u_mat = svd.u.expect("SVD u failed");
    let v_t_mat = svd.v_t.expect("SVD v_t failed");
    let mut lam = vec![1.0_f64; dim];
    for i in 0..dim { lam[i] = svd.singular_values[i]; }
    let v_mat = v_t_mat.transpose();

    let mut p = DMatrix::zeros(dim, dim);
    for i in 0..dim { for I in 0..dim {
        let mut val = 0.0;
        for a in 0..dim {
            if lam[a].abs() < 1e-30 { continue; }
            let mut dW_dlam = 0.0;
            for (mu_p, alpha_p) in params {
                dW_dlam += mu_p * lam[a].powf(alpha_p - 1.0);
            }
            dW_dlam += K * (jac - 1.0) * jac / lam[a];
            val += dW_dlam / lam[a] * u_mat[(i, a)] * v_mat[(I, a)];
        }
        p[(i, I)] = val;
    }}
    p
}

// ─── Numerical tangent (fallback for non-analytical models) ──────────────────

fn numerical_tangent(f: &DMatrix<f64>, pk1_fn: &dyn Fn(&DMatrix<f64>) -> DMatrix<f64>) -> DMatrix<f64> {
    let dim = f.nrows();
    let n = dim * dim;
    let mut ct = DMatrix::zeros(n, n);
    let eps = 1e-8;
    let p0 = pk1_fn(f);

    for j in 0..dim {
        for J in 0..dim {
            let mut f_pert = f.clone();
            f_pert[(j, J)] += eps;
            let p_pert = pk1_fn(&f_pert);
            for i in 0..dim {
                for I in 0..dim {
                    let row = i * dim + I;
                    let col = j * dim + J;
                    ct[(row, col)] = (p_pert[(i, I)] - p0[(i, I)]) / eps;
                }
            }
        }
    }
    ct
}

// ─── HyperelasticityForm (refactored to use HyperelasticModel) ───────────────

/// Finite-strain hyperelasticity form with selectable material model.
pub struct HyperelasticityForm<M: MeshTopology> {
    space: VectorH1Space<M>,
    pub model: HyperelasticModel,
    pub dirichlet: Vec<(usize, f64)>,
    pub quad_order: u8,
}

impl<M: MeshTopology> HyperelasticityForm<M> {
    pub fn new(space: VectorH1Space<M>, model: HyperelasticModel,
               dirichlet: Vec<(usize, f64)>, quad_order: u8) -> Self {
        Self { space, model, dirichlet, quad_order }
    }

    /// Run Newton–Raphson with line search to solve `F(u) = 0`.
    pub fn solve(&self, rhs: &[f64], u: &mut [f64],
                 config: &NewtonConfig) -> Result<NewtonResult, NewtonResult> {
        NewtonSolver::new(config.clone()).solve(self, rhs, u)
    }

    /// Reference to the underlying FE space.
    pub fn space(&self) -> &VectorH1Space<M> { &self.space }

    /// Compute the hyperelastic residual **without** Dirichlet BC enforcement.
    ///
    /// This is the raw element-level internal force vector: `∫ P(F) : ∇δu dV`.
    /// Unlike [`NonlinearForm::residual`], this does NOT set `r[dof] = u[dof] - val`
    /// for constrained DOFs.
    pub fn raw_residual(&self, u: &[f64], r: &mut [f64]) {
        let mesh = self.space.mesh();
        let dim = mesh.dim() as usize;
        let order = self.space.order();

        for i in 0..r.len() { r[i] = 0.0; }

        for e in mesh.elem_iter() {
            let elem_type = mesh.element_type(e);
            let ref_elem = ref_elem_vol(elem_type, order);
            let n_ldofs = ref_elem.n_dofs();
            let n_vec = n_ldofs * dim;
            let quad = ref_elem.quadrature(self.quad_order);

            let elem_dofs: Vec<usize> = self.space.element_dofs(e).iter()
                .map(|&d| d as usize).collect();

            let mut u_elem = vec![0.0_f64; n_vec];
            for (k, &dof) in elem_dofs.iter().enumerate() { u_elem[k] = u[dof]; }

            let mut f_elem = vec![0.0_f64; n_vec];
            let mut phi = vec![0.0_f64; n_ldofs];
            let mut gref = vec![0.0_f64; n_ldofs * dim];
            let mut gphys = vec![0.0_f64; n_ldofs * dim];

            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut gref);
                let (_jac, det_j, jit) = jacobian_at_point(mesh, e, xi, dim);
                let w = quad.weights[q] * det_j.abs();
                xform_grads(&jit, &gref, &mut gphys, n_ldofs, dim);

                let mut du = DMatrix::zeros(dim, dim);
                for k in 0..n_ldofs {
                    for i in 0..dim {
                        for j in 0..dim {
                            du[(i, j)] += u_elem[k * dim + i] * gphys[k * dim + j];
                        }
                    }
                }
                let mut f_mat = DMatrix::identity(dim, dim);
                f_mat += &du;
                let (p, _ct) = self.model.pk1_and_tangent(&f_mat);

                for k in 0..n_ldofs {
                    for i in 0..dim {
                        let row = k * dim + i;
                        let mut s = 0.0;
                        for j in 0..dim { s += p[(i, j)] * gphys[k * dim + j]; }
                        f_elem[row] += w * s;
                    }
                }
            }
            for (k, &dof) in elem_dofs.iter().enumerate() { r[dof] += f_elem[k]; }
        }
    }

    /// Compute the hyperelastic tangent matrix **without** Dirichlet BC modification.
    ///
    /// Returns the raw element-level tangent stiffness: `∫ 𝔸 : (∇δu ⊗ ∇u) dV`
    /// where 𝔸 is the consistent tangent modulus.
    pub fn raw_jacobian(&self, u: &[f64]) -> CsrMatrix<f64> {
        let mesh = self.space.mesh();
        let dim = mesh.dim() as usize;
        let order = self.space.order();
        let n_dofs = self.space.n_dofs();
        let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);

        for e in mesh.elem_iter() {
            let elem_type = mesh.element_type(e);
            let ref_elem = ref_elem_vol(elem_type, order);
            let n_ldofs = ref_elem.n_dofs();
            let n_vec = n_ldofs * dim;
            let quad = ref_elem.quadrature(self.quad_order);

            let elem_dofs: Vec<usize> = self.space.element_dofs(e).iter()
                .map(|&d| d as usize).collect();

            let mut u_elem = vec![0.0_f64; n_vec];
            for (k, &dof) in elem_dofs.iter().enumerate() { u_elem[k] = u[dof]; }

            let mut k_elem = vec![0.0_f64; n_vec * n_vec];
            let mut phi = vec![0.0_f64; n_ldofs];
            let mut gref = vec![0.0_f64; n_ldofs * dim];
            let mut gphys = vec![0.0_f64; n_ldofs * dim];

            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut gref);
                let (_jac, det_j, jit) = jacobian_at_point(mesh, e, xi, dim);
                let w = quad.weights[q] * det_j.abs();
                xform_grads(&jit, &gref, &mut gphys, n_ldofs, dim);

                let mut du = DMatrix::zeros(dim, dim);
                for k in 0..n_ldofs {
                    for i in 0..dim {
                        for j in 0..dim {
                            du[(i, j)] += u_elem[k * dim + i] * gphys[k * dim + j];
                        }
                    }
                }
                let mut f_mat = DMatrix::identity(dim, dim);
                f_mat += &du;
                let (_p, ct) = self.model.pk1_and_tangent(&f_mat);

                for k in 0..n_ldofs {
                    for i in 0..dim {
                        let row = k * dim + i;
                        for l in 0..n_ldofs {
                            for a in 0..dim {
                                let col = l * dim + a;
                                let mut val = 0.0;
                                for j in 0..dim {
                                    for b in 0..dim {
                                        val += ct[(i * dim + j, a * dim + b)]
                                            * gphys[k * dim + j]
                                            * gphys[l * dim + b];
                                    }
                                }
                                k_elem[row * n_vec + col] += w * val;
                            }
                        }
                    }
                }
            }
            coo.add_element_matrix(&elem_dofs, &k_elem);
        }
        coo.into_csr()
    }

    /// Compute the total elastic (internal) energy `∫ ψ(F) dV`.
    pub fn elastic_energy(&self, u: &[f64]) -> f64 {
        let mesh = self.space.mesh();
        let dim = mesh.dim() as usize;
        let order = self.space.order();
        let mut energy = 0.0;

        for e in mesh.elem_iter() {
            let elem_type = mesh.element_type(e);
            let ref_elem = ref_elem_vol(elem_type, order);
            let n_ldofs = ref_elem.n_dofs();
            let n_vec = n_ldofs * dim;
            let quad = ref_elem.quadrature(self.quad_order);


            let mut u_elem = vec![0.0_f64; n_vec];
            let elem_dofs: Vec<usize> = self.space.element_dofs(e).iter()
                .map(|&d| d as usize).collect();
            for (k, &dof) in elem_dofs.iter().enumerate() { u_elem[k] = u[dof]; }

            let mut phi = vec![0.0_f64; n_ldofs];
            let mut gref = vec![0.0_f64; n_ldofs * dim];
            let mut gphys = vec![0.0_f64; n_ldofs * dim];

            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut gref);
                let (_jac, det_j, jit) = jacobian_at_point(mesh, e, xi, dim);
                let w = quad.weights[q] * det_j.abs();
                xform_grads(&jit, &gref, &mut gphys, n_ldofs, dim);

                let mut du = DMatrix::zeros(dim, dim);
                for k in 0..n_ldofs {
                    for i in 0..dim {
                        for j in 0..dim {
                            du[(i, j)] += u_elem[k * dim + i] * gphys[k * dim + j];
                        }
                    }
                }
                let mut f_mat = DMatrix::identity(dim, dim);
                f_mat += &du;
                energy += w * self.model.elastic_energy_density(&f_mat);
            }
        }
        energy
    }

    /// Compute elastic energy density projected onto an L² space.
    ///
    /// Returns an L² DOF vector where each entry stores `ψ(F)/det(F)`
    /// (the elastic energy density in the **deformed** configuration).
    ///
    /// This mirrors the C++ MFEM ``ElasticEnergyCoefficient`` used in ex10:
    /// `w.ProjectCoefficient(ElasticEnergyCoefficient(*model, x))`.
    ///
    /// # Algorithm
    ///
    /// For each element:
    /// 1. Assemble the element L² mass matrix `M_e` and RHS `b_e` where
    ///    `b_e[i] = ∫ φ̂ᵢ · ψ(F)/det(F) dx` (φ̂ are the L² basis functions).
    /// 2. Solve the local system `M_e · c_e = b_e`.
    /// 3. Write `c_e` into the global DOF vector.
    ///
    /// The element is parameterised by the H¹ displacement field `u`.
    pub fn compute_elastic_energy_density(
        &self,
        u: &[f64],
        l2_space: &L2Space<M>,
        quad_order: u8,
    ) -> Vec<f64> {
        let mesh = self.space.mesh();
        let dim = mesh.dim() as usize;
        let h1_order = self.space.order();
        let l2_order = l2_space.order();
        let n_l2 = l2_space.n_dofs();
        let mut edens = vec![0.0; n_l2];

        // P0 special case: one constant DOF per element
        if l2_order == 0 {
            let mut elem_idx = 0;
            for e in mesh.elem_iter() {
                let elem_type = mesh.element_type(e);
                let ref_elem = ref_elem_vol(elem_type, h1_order);
                let n_ldofs = ref_elem.n_dofs();
                let n_vec = n_ldofs * dim;
                let quad = ref_elem.quadrature(quad_order);

                let elem_dofs: Vec<usize> = self.space.element_dofs(e).iter()
                    .map(|&d| d as usize).collect();
                let mut u_elem = vec![0.0; n_vec];
                for (k, &dof) in elem_dofs.iter().enumerate() { u_elem[k] = u[dof]; }

                let mut phi = vec![0.0; n_ldofs];
                let mut gref = vec![0.0; n_ldofs * dim];
                let mut gphys = vec![0.0; n_ldofs * dim];
                let mut energy = 0.0;
                let mut vol = 0.0;

                for (q, xi) in quad.points.iter().enumerate() {
                    ref_elem.eval_basis(xi, &mut phi);
                    ref_elem.eval_grad_basis(xi, &mut gref);
                    let (_jac, det_j, jit) = jacobian_at_point(mesh, e, xi, dim);
                    let w = quad.weights[q] * det_j.abs();
                    xform_grads(&jit, &gref, &mut gphys, n_ldofs, dim);

                    let mut du = DMatrix::zeros(dim, dim);
                    for k in 0..n_ldofs {
                        for i in 0..dim {
                            for j in 0..dim {
                                du[(i, j)] += u_elem[k * dim + i] * gphys[k * dim + j];
                            }
                        }
                    }
                    let mut f_mat = DMatrix::identity(dim, dim);
                    f_mat += &du;
                    let psi = self.model.elastic_energy_density(&f_mat);
                    let det_f = f_mat.determinant();
                    let w_energy = if det_f.abs() > 1e-30 { psi / det_f.abs() } else { psi };
                    energy += w * w_energy;
                    vol += w;
                }
                edens[elem_idx] = if vol > 0.0 { energy / vol } else { 0.0 };
                elem_idx += 1;
            }
            return edens;
        }

        // Orders ≥ 1: element-local L² projection
        for e in mesh.elem_iter() {
            let elem_type = mesh.element_type(e);
            let h1_ref = ref_elem_vol(elem_type, h1_order);
            let l2_ref = ref_elem_vol(elem_type, l2_order);
            let n_h1 = h1_ref.n_dofs();
            let n_l2_ldofs = l2_ref.n_dofs();
            let n_vec = n_h1 * dim;
            let quad = l2_ref.quadrature(quad_order);

            // H¹ element DOFs (interleaved: x₀,y₀, x₁,y₁, …)
            let h1_dofs: Vec<usize> = self.space.element_dofs(e).iter()
                .map(|&d| d as usize).collect();
            let mut u_elem = vec![0.0; n_vec];
            for (k, &dof) in h1_dofs.iter().enumerate() { u_elem[k] = u[dof]; }

            // L² element DOFs
            let l2_dofs: Vec<usize> = l2_space.element_dofs(e).iter()
                .map(|&d| d as usize).collect();

            // Local mass matrix M_e and RHS b_e
            let mut m_elem = vec![0.0; n_l2_ldofs * n_l2_ldofs];
            let mut b_elem = vec![0.0; n_l2_ldofs];

            let mut phi_h1 = vec![0.0; n_h1];
            let mut gref_h1 = vec![0.0; n_h1 * dim];
            let mut gphys_h1 = vec![0.0; n_h1 * dim];
            let mut phi_l2 = vec![0.0; n_l2_ldofs];

            for (q, xi) in quad.points.iter().enumerate() {
                h1_ref.eval_basis(xi, &mut phi_h1);
                h1_ref.eval_grad_basis(xi, &mut gref_h1);
                let (_jac, det_j, jit) = jacobian_at_point(mesh, e, xi, dim);
                let w = quad.weights[q] * det_j.abs();
                xform_grads(&jit, &gref_h1, &mut gphys_h1, n_h1, dim);

                // Deformation gradient F = I + ∇u
                let mut du = DMatrix::zeros(dim, dim);
                for k in 0..n_h1 {
                    for i in 0..dim {
                        for j in 0..dim {
                            du[(i, j)] += u_elem[k * dim + i] * gphys_h1[k * dim + j];
                        }
                    }
                }
                let mut f_mat = DMatrix::identity(dim, dim);
                f_mat += &du;
                let psi = self.model.elastic_energy_density(&f_mat);
                let det_f = f_mat.determinant();
                // ψ(F)/det(F) = density in deformed configuration
                let w_energy = if det_f.abs() > 1e-30 { psi / det_f.abs() } else { psi };

                // L² basis at the same quadrature point
                l2_ref.eval_basis(xi, &mut phi_l2);

                for i in 0..n_l2_ldofs {
                    b_elem[i] += w * phi_l2[i] * w_energy;
                    for j in 0..n_l2_ldofs {
                        m_elem[i * n_l2_ldofs + j] += w * phi_l2[i] * phi_l2[j];
                    }
                }
            }

            // Solve local system M_e · c_e = b_e via Gaussian elimination
            let c_e = solve_local_system(&m_elem, &b_elem, n_l2_ldofs);
            for (k, &dof) in l2_dofs.iter().enumerate() {
                if dof < n_l2 { edens[dof] = c_e[k]; }
            }
        }
        edens
    }
}

impl<M: MeshTopology> NonlinearForm for HyperelasticityForm<M> {
    fn n_dofs(&self) -> usize { self.space.n_dofs() }

    fn residual(&self, u: &[f64], rhs: &[f64], r: &mut [f64]) {
        let mesh = self.space.mesh();
        let dim = mesh.dim() as usize;
        let order = self.space.order();

        for i in 0..r.len() { r[i] = -rhs[i]; }

        for e in mesh.elem_iter() {
            let elem_type = mesh.element_type(e);
            let ref_elem = ref_elem_vol(elem_type, order);
            let n_ldofs = ref_elem.n_dofs();
            let n_vec = n_ldofs * dim;
            let quad = ref_elem.quadrature(self.quad_order);

            let elem_dofs: Vec<usize> = self.space.element_dofs(e).iter()
                .map(|&d| d as usize).collect();

            let mut u_elem = vec![0.0_f64; n_vec];
            for (k, &dof) in elem_dofs.iter().enumerate() { u_elem[k] = u[dof]; }

            let mut f_elem = vec![0.0_f64; n_vec];
            let mut phi = vec![0.0_f64; n_ldofs];
            let mut gref = vec![0.0_f64; n_ldofs * dim];
            let mut gphys = vec![0.0_f64; n_ldofs * dim];

            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut gref);
                let (_jac, det_j, jit) = jacobian_at_point(mesh, e, xi, dim);
                let w = quad.weights[q] * det_j.abs();
                xform_grads(&jit, &gref, &mut gphys, n_ldofs, dim);

                let mut du = DMatrix::zeros(dim, dim);
                for k in 0..n_ldofs {
                    for i in 0..dim {
                        for j in 0..dim {
                            du[(i, j)] += u_elem[k * dim + i] * gphys[k * dim + j];
                        }
                    }
                }
                let mut f_mat = DMatrix::identity(dim, dim);
                f_mat += &du;
                let (p, _ct) = self.model.pk1_and_tangent(&f_mat);

                for k in 0..n_ldofs {
                    for i in 0..dim {
                        let row = k * dim + i;
                        let mut s = 0.0;
                        for j in 0..dim { s += p[(i, j)] * gphys[k * dim + j]; }
                        f_elem[row] += w * s;
                    }
                }
            }
            for (k, &dof) in elem_dofs.iter().enumerate() { r[dof] += f_elem[k]; }
        }

        for &(dof, val) in &self.dirichlet { r[dof] = u[dof] - val; }
    }

    fn jacobian(&self, u: &[f64]) -> CsrMatrix<f64> {
        let mesh = self.space.mesh();
        let dim = mesh.dim() as usize;
        let order = self.space.order();
        let n_dofs = self.space.n_dofs();
        let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);

        for e in mesh.elem_iter() {
            let elem_type = mesh.element_type(e);
            let ref_elem = ref_elem_vol(elem_type, order);
            let n_ldofs = ref_elem.n_dofs();
            let n_vec = n_ldofs * dim;
            let quad = ref_elem.quadrature(self.quad_order);

            let elem_dofs: Vec<usize> = self.space.element_dofs(e).iter()
                .map(|&d| d as usize).collect();

            let mut u_elem = vec![0.0_f64; n_vec];
            for (k, &dof) in elem_dofs.iter().enumerate() { u_elem[k] = u[dof]; }

            let mut k_elem = vec![0.0_f64; n_vec * n_vec];
            let mut phi = vec![0.0_f64; n_ldofs];
            let mut gref = vec![0.0_f64; n_ldofs * dim];
            let mut gphys = vec![0.0_f64; n_ldofs * dim];

            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut gref);
                let (_jac, det_j, jit) = jacobian_at_point(mesh, e, xi, dim);
                let w = quad.weights[q] * det_j.abs();
                xform_grads(&jit, &gref, &mut gphys, n_ldofs, dim);

                let mut du = DMatrix::zeros(dim, dim);
                for k in 0..n_ldofs {
                    for i in 0..dim {
                        for j in 0..dim {
                            du[(i, j)] += u_elem[k * dim + i] * gphys[k * dim + j];
                        }
                    }
                }
                let mut f_mat = DMatrix::identity(dim, dim);
                f_mat += &du;
                let (_p, ct) = self.model.pk1_and_tangent(&f_mat);

                for k in 0..n_ldofs {
                    for i in 0..dim {
                        let row = k * dim + i;
                        for l in 0..n_ldofs {
                            for a in 0..dim {
                                let col = l * dim + a;
                                let mut val = 0.0;
                                for j in 0..dim {
                                    for b in 0..dim {
                                        val += ct[(i * dim + j, a * dim + b)]
                                            * gphys[k * dim + j]
                                            * gphys[l * dim + b];
                                    }
                                }
                                k_elem[row * n_vec + col] += w * val;
                            }
                        }
                    }
                }
            }
            coo.add_element_matrix(&elem_dofs, &k_elem);
        }

        let mut mat = coo.into_csr();
        for &(dof, _val) in &self.dirichlet {
            mat.apply_dirichlet_row_zeroing(dof, 0.0, &mut vec![0.0; n_dofs]);
        }
        mat
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn ref_elem_vol(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    use fem_element::lagrange::quad::*;
    match (et, order) {
        (ElementType::Tri3, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) => Box::new(TriP2),
        (ElementType::Tri3, 3) => Box::new(TriP3),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetP3),
        // Quadrilateral elements (straight-sided or curved via isoparametric mapping)
        (ElementType::Quad4, 1) => Box::new(QuadQ1),
        (ElementType::Quad4, 2) => Box::new(QuadQ2),
        // order >= 3: GLL nodes on [0,1]^2 (MFEM H1 default); QuadQ3 is
        // equidistant [-1,1]^2 and NOT MFEM-compatible at p=3.
        (ElementType::Quad4, 3) => Box::new(fem_element::lagrange::QuadQk::new(3)),
        (ElementType::Quad4, 4) => Box::new(QuadQ4),
        _ => panic!("hyperelasticity ref_elem_vol: unsupported ({et:?}, {order})"),
    }
}

/// Geometry reference element (always order 1 — P1/Q1).
///
/// The element mapping Jacobian is always computed from the lowest-order
/// geometry (mesh vertex nodes), even when the field uses higher-order
/// basis functions (sub-parametric formulation).  This matches MFEM's
/// behaviour where the transformation uses the mesh's reference element.
fn ref_elem_geom(et: ElementType) -> Box<dyn ReferenceElement> {
    ref_elem_vol(et, 1)
}

/// Compute the element mapping Jacobian at a reference point using the
/// **geometry** reference element (order-1 vertex mapping).
///
/// J[i,j] = Σ_k  x_k[i] · ∂φ̂_k/∂ξⱼ(ξ)   (k over geometry nodes, φ̂ = P1/Q1)
///
/// For straight-sided elements this gives a constant Jacobian; for curved
/// geometries the per-point variation is captured.  Returns (Jacobian,
/// determinant, inverse-transpose).
fn jacobian_at_point<M: MeshTopology>(
    mesh: &M, elem: u32, xi: &[f64], dim: usize,
) -> (DMatrix<f64>, f64, DMatrix<f64>) {
    let et = mesh.element_type(elem);
    let geom_nodes = mesh.element_nodes(elem);
    let n_geom = geom_nodes.len();
    let mut gref = vec![0.0_f64; n_geom * dim];
    ref_elem_geom(et).eval_grad_basis(xi, &mut gref);

    let mut jac = DMatrix::zeros(dim, dim);
    for k in 0..n_geom {
        let xk = mesh.node_coords(geom_nodes[k]);
        for i in 0..dim {
            for j in 0..dim {
                jac[(i, j)] += xk[i] * gref[k * dim + j];
            }
        }
    }
    let det_j = jac.determinant();
    let jit = jac.clone().try_inverse().expect("singular element Jacobian").transpose();
    (jac, det_j, jit)
}

fn xform_grads(jit: &DMatrix<f64>, gr: &[f64], gp: &mut [f64], n: usize, dim: usize) {
    for i in 0..n {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim { s += jit[(j,k)] * gr[i*dim+k]; }
            gp[i*dim+j] = s;
        }
    }
}

// ─── Helper: element-local linear system solve ───────────────────────────

/// Solve a small dense system `M · x = b` (size n ≤ 20) by LU
/// decomposition with partial pivoting (row-major storage).
fn solve_local_system(m: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    debug_assert!(n <= 20, "solve_local_system: n={n} > 20 is not supported");
    let mut a = m.to_vec();
    let mut x = b.to_vec();
    for col in 0..n {
        let mut pivot = col;
        let mut max_val = a[col * n + col].abs();
        for row in (col + 1)..n {
            let val = a[row * n + col].abs();
            if val > max_val { max_val = val; pivot = row; }
        }
        if max_val < 1e-30 { continue; }
        if pivot != col {
            for j in col..n { a.swap(col * n + j, pivot * n + j); }
            x.swap(col, pivot);
        }
        let diag = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / diag;
            if factor == 0.0 { continue; }
            for j in (col + 1)..n {
                a[row * n + j] -= factor * a[col * n + j];
            }
            a[row * n + col] = 0.0;
            x[row] -= factor * x[col];
        }
    }
    for col in (0..n).rev() {
        let diag = a[col * n + col];
        if diag.abs() < 1e-30 { continue; }
        for row in (0..col).rev() {
            if a[row * n + col] != 0.0 {
                x[row] -= a[row * n + col] * x[col] / diag;
            }
        }
        x[col] /= diag;
    }
    x
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::vector_h1::VectorH1Space;

    #[test]
    fn zero_displacement_zero_residual_neo() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        let model = HyperelasticModel::NeoHookean { mu: 0.3, lambda: 1.0 };
        let form = HyperelasticityForm::new(space, model, vec![], 2);
        let mut r = vec![0.0_f64; n];
        form.residual(&vec![0.0; n], &vec![0.0; n], &mut r);
        let norm: f64 = r.iter().map(|x| x.abs()).sum();
        assert!(norm < 1e-12, "Zero displacement should give zero residual, got {norm}");
    }

    #[test]
    fn tangent_matrix_nonzero_neo() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        let model = HyperelasticModel::NeoHookean { mu: 0.3, lambda: 1.0 };
        let form = HyperelasticityForm::new(space, model, vec![], 2);
        let jac = form.jacobian(&vec![0.0; n]);
        let mut sum = 0.0;
        for i in 0..n.min(10) {
            for j in 0..n.min(10) {
                sum += jac.get(i, j).abs();
            }
        }
        assert!(sum > 0.0, "Tangent matrix should have non-zero entries");
    }

    #[test]
    fn mooney_rivlin_tangent_nonzero() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        let model = HyperelasticModel::MooneyRivlin { c10: 0.3, c01: 0.1, bulk_modulus: 1e3 };
        let form = HyperelasticityForm::new(space, model, vec![], 2);
        let jac = form.jacobian(&vec![0.0; n]);
        let mut sum = 0.0;
        for i in 0..n.min(10) { for j in 0..n.min(10) { sum += jac.get(i, j).abs(); } }
        assert!(sum > 0.0, "MR tangent should be non-zero");
    }

    #[test]
    fn ogden_tangent_nonzero() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        let model = HyperelasticModel::Ogden {
            params: vec![(6.0e5, 1.3), (-1.0e5, 2.0)],
            bulk_modulus: 1e3,
        };
        let form = HyperelasticityForm::new(space, model, vec![], 2);
        let jac = form.jacobian(&vec![0.0; n]);
        let mut sum = 0.0;
        for i in 0..n.min(10) { for j in 0..n.min(10) { sum += jac.get(i, j).abs(); } }
        assert!(sum > 0.0, "Ogden tangent should be non-zero");
    }

    /// Finite-difference verification of `raw_jacobian` against
    /// numerical differentiation of `raw_residual`.
    ///
    /// For a random small displacement u, compute the Jacobian analytically
    /// via `raw_jacobian(u)`, then approximate it via central differences:
    ///   J_fd[:, i] = (f(u + ε e_i) - f(u - ε e_i)) / (2 ε)
    /// and check that ‖J - J_fd‖ / ‖J‖ < tol.
    #[test]
    fn finite_difference_verification_raw_jacobian() {
        let mesh = Mesh::<2>::unit_square_tri(8);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        let nd = n.min(20); // test first 20 DOFs only (FD is expensive)

        let model = HyperelasticModel::NeoHookean { mu: 0.3, lambda: 1.0 };
        // No Dirichlet BCs — test the raw form directly
        let form = HyperelasticityForm::new(space, model, vec![], 2);

        // Very small random displacement (magnitude ~1e-4)
        let u: Vec<f64> = (0..n).map(|i| 1e-4 * (i as f64).sin()).collect();

        // Analytical Jacobian (first nd×nd block)
        let jac = form.raw_jacobian(&u);
        // Quick sanity: check Jacobian has non-zero entries and no NaN
        let mut jac_norm = 0.0;
        for i in 0..nd.min(5) { for j in 0..nd.min(5) {
            let v = jac.get(i, j);
            assert!(!v.is_nan(), "raw_jacobian has NaN at ({i},{j})");
            jac_norm += v.abs();
        }}
        assert!(jac_norm > 0.0, "raw_jacobian is all zero");
        let mut jac_dense = vec![0.0; nd * nd];
        for i in 0..nd {
            for j in 0..nd {
                jac_dense[i * nd + j] = jac.get(i, j);
            }
        }

        // FD Jacobian via central differences
        let eps = 1e-6;
        let mut jac_fd = vec![0.0; nd * nd];
        let mut r_plus  = vec![0.0; n];
        let mut r_minus = vec![0.0; n];

        for j in 0..nd {
            // u + ε e_j
            let mut u_pert = u.clone();
            u_pert[j] += eps;
            form.raw_residual(&u_pert, &mut r_plus);

            // u - ε e_j
            u_pert[j] = u[j] - eps;
            form.raw_residual(&u_pert, &mut r_minus);

            for i in 0..nd {
                jac_fd[i * nd + j] = (r_plus[i] - r_minus[i]) / (2.0 * eps);
            }
        }

        // Relative Frobenius-norm error ‖J - J_fd‖ / ‖J‖
        let mut err_sq = 0.0;
        let mut norm_sq = 0.0;
        for i in 0..nd * nd {
            let diff = jac_dense[i] - jac_fd[i];
            err_sq += diff * diff;
            norm_sq += jac_dense[i] * jac_dense[i];
        }
        let rel_err = err_sq.sqrt() / norm_sq.sqrt().max(1e-30);

        eprintln!("  raw_jacobian FD verification: rel_err = {:.6e}", rel_err);
        assert!(
            rel_err < 1e-3,
            "raw_jacobian FD mismatch: rel_err = {:.6e} (expected < 1e-3)",
            rel_err
        );
    }

    #[test]
    fn arruda_boyce_identity() {
        let f = DMatrix::identity(3, 3);
        let s = arruda_boyce_pk1_stress(&f, 0.3, 3.0, 1e3);
        let psi = arruda_boyce_energy(&f, 0.3, 3.0, 1e3);
        for i in 0..3 { for j in 0..3 {
            assert!(s[(i,j)].abs() < 1.0, "stress at I, s[{}][{}] = {}", i, j, s[(i,j)]);
        }}
    }

    #[test]
    fn arruda_boyce_tension() {
        let mut f = DMatrix::identity(3, 3);
        f[(0,0)] = 1.1;
        f[(1,1)] = 1.0 / 1.1_f64.sqrt();
        f[(2,2)] = 1.0 / 1.1_f64.sqrt();
        let s = arruda_boyce_pk1_stress(&f, 0.3, 3.0, 1e3);
        assert!(s[(0,0)] > 0.0);
    }

    #[test]
    fn yeoh_identity() {
        let f = DMatrix::identity(3, 3);
        let psi = yeoh_energy(&f, 0.3, -0.1, 0.02, 1e3);
    }

    /// Verify the analytical MfemNeoHookean tangent against central-difference
    /// numerical tangent for a range of deformation gradients.
    #[test]
    fn mfem_neo_hookean_analytic_tangent_vs_fd() {
        let mu = 0.25;
        let K = 5.0;
        let eps = 1e-6;

        // Test at F=I (small-strain limit) and several deformed states
        let test_configs: Vec<(usize, Vec<Vec<f64>>)> = vec![
            (2, vec![
                vec![1.0, 0.0, 0.0, 1.0],                        // identity
                vec![1.1, 0.0, 0.0, 1.0/1.1],                    // uniaxial tension
                vec![1.0, 0.2, 0.0, 1.0],                         // simple shear
                vec![1.3, 0.0, 0.1, 0.8],                         // combined
            ]),
            (3, vec![
                vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],  // identity
                vec![1.2, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.9],  // uniaxial
                vec![1.0, 0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],  // shear
            ]),
        ];

        for (dim, configs) in &test_configs {
            for entries in configs {
                let f = DMatrix::<f64>::from_row_slice(*dim, *dim, entries);
                // Ensure positive determinant
                if f.determinant() <= 0.0 { continue; }

                let (_, ct_a) = mfem_neo_hookean_pk1_tangent(&f, mu, K);

                // Central-difference numerical tangent (eps=1e-6)
                let n = dim * dim;
                let mut ct_fd = DMatrix::zeros(n, n);
                for j in 0..*dim {
                    for J in 0..*dim {
                        let mut f_plus = f.clone();
                        f_plus[(j, J)] += eps;
                        let p_plus = mfem_neo_hookean_pk1(&f_plus, mu, K);
                        let mut f_minus = f.clone();
                        f_minus[(j, J)] -= eps;
                        let p_minus = mfem_neo_hookean_pk1(&f_minus, mu, K);
                        for i in 0..*dim {
                            for I in 0..*dim {
                                let row = i * dim + I;
                                let col = j * dim + J;
                                ct_fd[(row, col)] = (p_plus[(i, I)] - p_minus[(i, I)]) / (2.0 * eps);
                            }
                        }
                    }
                }

                // Relative Frobenius error ||C_a - C_fd|| / ||C_a||
                let mut err_sq = 0.0;
                let mut norm_sq = 0.0;
                for r in 0..n { for c in 0..n {
                    let d = ct_a[(r, c)] - ct_fd[(r, c)];
                    err_sq += d * d;
                    norm_sq += ct_a[(r, c)] * ct_a[(r, c)];
                }}
                let rel_err = err_sq.sqrt() / norm_sq.sqrt().max(1e-30);
                assert!(
                    rel_err < 1e-6,
                    "MfemNeoHookean analytic tangent FD mismatch dim={dim}: rel_err = {rel_err:.2e} (expected < 1e-6)"
                );
            }
        }
    }

    #[test]
    fn yeoh_tension() {
        let mut f = DMatrix::identity(3, 3);
        f[(0,0)] = 1.2;
        f[(1,1)] = 1.0 / 1.2_f64.sqrt();
        f[(2,2)] = 1.0 / 1.2_f64.sqrt();
        let s = yeoh_pk1_stress(&f, 0.3, -0.1, 0.02, 1e3);
        assert!(s[(0,0)] > 0.0);
    }
}
