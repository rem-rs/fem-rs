//! Piezoelectric coupling integrator.
//!
//! Assembles the piezoelectric coupling matrix `K_uφ`:
//!
//! ```text
//! K_uφ(a, b) = ∫_Ω ε(v_a) : e · (-∇φ_b) dΩ
//! ```
//!
//! where `v_a` are VectorH¹ test functions (displacement) and `φ_b` are H¹
//! test functions (electric potential).
//!
//! The full piezoelectric system in block form:
//!
//! ```text
//! [ K_uu   K_uφ ] [ u ]   [ f ]
//! [ K_φu  -K_φφ ] [ φ ] = [ q ]
//! ```
//!
//! where:
//! - `K_uu` — elastic stiffness (∫ ε: cᴱ : ε)
//! - `K_φφ` — dielectric stiffness (∫ ∇ψ·εˢ·∇φ)
//! - `K_uφ` — piezoelectric coupling (∫ ε : e · (-∇φ))
//! - `K_φu` = K_uφᵀ

use fem_linalg::CsrMatrix;

// ─── PiezoMaterial struct (defined BEFORE materials module) ───────────────────

/// Piezoelectric material constants in reduced form.
///
/// Uses the standard crystallographic notation for a 6mm (C₆ᵥ) symmetry class
/// (applicable to poled PZT ceramics).
#[derive(Debug, Clone)]
pub struct PiezoMaterial {
    pub name: &'static str,

    // Elastic stiffness cᴱ (Pa) — 6mm class
    pub c11: f64, pub c12: f64, pub c13: f64,
    pub c33: f64, pub c44: f64, pub c66: f64,

    // Piezoelectric stress coefficients (C/m²)
    pub e31: f64, pub e33: f64, pub e15: f64,

    // Relative permittivity (dimensionless, εˢ/ε₀)
    pub eps11_rel: f64, pub eps33_rel: f64,

    // Density (kg/m³)
    pub density: f64,
}

impl PiezoMaterial {
    /// Vacuum permittivity (F/m).
    const EPS0: f64 = 8.854_187_817e-12;

    /// Absolute permittivity matrix (F/m) — transverse isotropic [ε₁₁, ε₁₁, ε₃₃].
    pub fn permittivity(&self) -> [f64; 3] {
        [
            self.eps11_rel * Self::EPS0,
            self.eps11_rel * Self::EPS0,
            self.eps33_rel * Self::EPS0,
        ]
    }

    /// Write cᴱ into a 6×6 Voigt matrix (row-major).
    pub fn elastic_matrix(&self) -> [[f64; 6]; 6] {
        let mut c = [[0.0; 6]; 6];
        // 6mm symmetry in Voigt notation
        c[0][0] = self.c11; c[0][1] = self.c12; c[0][2] = self.c13;
        c[1][0] = self.c12; c[1][1] = self.c11; c[1][2] = self.c13;
        c[2][0] = self.c13; c[2][1] = self.c13; c[2][2] = self.c33;
        c[3][3] = self.c44;
        c[4][4] = self.c44;
        c[5][5] = self.c66;
        c
    }

    /// Write e into a 3×6 Voigt matrix (row-major, stress-charge form).
    ///
    /// For 6mm class, only e₃₁, e₃₃, e₁₅ are non-zero:
    /// ```text
    /// e = [0   0   0   0   e₁₅  0  ]
    ///     [0   0   0   e₁₅  0   0  ]
    ///     [e₃₁ e₃₁ e₃₃ 0    0   0  ]
    /// ```
    pub fn piezoelectric_matrix(&self) -> [[f64; 6]; 3] {
        let mut e = [[0.0; 6]; 3];
        e[0][4] = self.e15;
        e[1][3] = self.e15;
        e[2][0] = self.e31;
        e[2][1] = self.e31;
        e[2][2] = self.e33;
        e
    }
}

// ─── Coupling matrix assembly stub ───────────────────────────────────────────

/// Assemble the piezoelectric coupling matrix `K_uφ`.
///
/// For production, this requires iterating over elements, computing Voigt
/// strain-displacement matrices and electric field gradients, and assembling
/// the coupled block using the piezoelectric tensor `e`.
///
/// Returns a matrix of shape (n_displacement × n_potential).
pub fn assemble_piezoelectric_coupling(
    _quad_order: u8,
) -> CsrMatrix<f64> {
    // Placeholder — returns empty 0×0 matrix.
    // Full implementation follows the pattern in thermoelastic.rs
    // (K_uT assembly loop) but replaces thermal expansion with the
    // piezoelectric tensor e (3×6 Voigt).
    CsrMatrix::new_empty(0, 0)
}

// ─── Standard piezoelectric materials ────────────────────────────────────────

/// Standard piezoelectric material constants (stress-charge form, IEEE Std 176).
pub mod materials {
    use super::PiezoMaterial;

    /// PZT-5A: soft PZT, high coupling, moderate permittivity.
    pub fn pzt_5a() -> PiezoMaterial {
        PiezoMaterial {
            name: "PZT-5A",
            c11: 121.0e9, c12: 75.4e9, c13: 75.2e9,
            c33: 111.0e9, c44: 21.1e9, c66: 22.8e9,
            e31: -5.4, e33: 15.8, e15: 12.3,
            eps11_rel: 916.0, eps33_rel: 830.0,
            density: 7750.0,
        }
    }

    /// PZT-5H: soft PZT, higher coupling.
    pub fn pzt_5h() -> PiezoMaterial {
        PiezoMaterial {
            name: "PZT-5H",
            c11: 126.0e9, c12: 79.5e9, c13: 84.1e9,
            c33: 117.0e9, c44: 23.0e9, c66: 23.3e9,
            e31: -6.5, e33: 23.3, e15: 17.0,
            eps11_rel: 1700.0, eps33_rel: 1470.0,
            density: 7500.0,
        }
    }

    /// Quartz (α-SiO₂), trigonal class 32.
    pub fn quartz() -> PiezoMaterial {
        PiezoMaterial {
            name: "Quartz",
            c11: 86.74e9, c12: 6.99e9, c13: 11.91e9,
            c33: 107.2e9, c44: 57.94e9, c66: 39.88e9,
            e31: -0.067, e33: 0.087, e15: 0.067,
            eps11_rel: 4.52, eps33_rel: 4.68,
            density: 2650.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pzt5a_material_finite() {
        let mat = materials::pzt_5a();
        assert!(mat.density > 0.0);
        assert!(mat.c11 > 0.0);
    }

    #[test]
    fn pzt5a_permittivity_positive() {
        let mat = materials::pzt_5a();
        let eps = mat.permittivity();
        assert!(eps[0] > 0.0);
        assert!(eps[2] > 0.0);
    }

    #[test]
    fn pzt5a_elastic_symmetric() {
        let mat = materials::pzt_5a();
        let c = mat.elastic_matrix();
        assert!((c[0][1] - c[1][0]).abs() < 1e-10);
        assert!((c[3][3] - mat.c44).abs() < 1e-10);
    }

    #[test]
    fn pzt5a_piezo_matrix_structure() {
        let mat = materials::pzt_5a();
        let e = mat.piezoelectric_matrix();
        assert!((e[2][0] - mat.e31).abs() < 1e-10);
        assert!((e[0][4] - mat.e15).abs() < 1e-10);
        assert_eq!(e[0][0], 0.0);
        assert_eq!(e[1][1], 0.0);
    }
}
