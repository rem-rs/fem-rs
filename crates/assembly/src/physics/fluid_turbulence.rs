//! Turbulence model integration for RANS CFD.
//!
//! Provides eddy viscosity computation from turbulence models and
//! coupling to the Navier-Stokes assembly (via ν → ν + νₜ).

use fem_linalg::{CsrMatrix, CooMatrix};
use fem_mesh::topology::MeshTopology;

/// Eddy viscosity from the Smagorinsky LES model.
///
/// νₜ = (Cₛ·Δ)² · |S|
/// where `Cₛ ≈ 0.1–0.2`, Δ = element size, |S| = strain rate magnitude.
pub fn smagorinsky_eddy_viscosity(
    cell_size: f64,
    strain_rate_mag: f64,
    cs: f64,
) -> f64 {
    let delta_sq = cell_size * cell_size;
    cs * cs * delta_sq * strain_rate_mag
}

/// Compute the element-averaged strain rate magnitude from velocity gradient.
pub fn strain_rate_magnitude(
    grad_u: &[f64], // [du/dx, du/dy, dv/dx, dv/dy] for 2D
) -> f64 {
    if grad_u.len() < 4 { return 0.0; }
    let dux = grad_u[0]; let duy = grad_u[1];
    let dvx = grad_u[2]; let dvy = grad_u[3];

    let s11 = dux;
    let s22 = dvy;
    let s12 = 0.5 * (duy + dvx);
    let s_mag = (2.0 * (s11 * s11 + s22 * s22 + 2.0 * s12 * s12)).sqrt();
    s_mag
}

/// Build the turbulent viscosity vector (per-element) for the NS assembly.
///
/// Modifies the diffusion operator by adding eddy viscosity:
/// `ν_eff = ν + νₜ`
pub fn add_turbulent_viscosity(
    a_diff: &CsrMatrix<f64>,
    nu: f64,
    nu_t: &[f64],     // per-element eddy viscosity
    elem_map: &[usize], // element ID for each velocity DOF
) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(a_diff.nrows, a_diff.ncols);
    for r in 0..a_diff.nrows {
        for k in a_diff.row_ptr[r]..a_diff.row_ptr[r + 1] {
            let c = a_diff.col_idx[k] as usize;
            let v = a_diff.values[k];
            // Scale by (ν + νₜ) / ν = (1 + νₜ/ν)
            let elem_id = elem_map.get(r).copied().unwrap_or(0);
            let nu_eff_ratio = if nu > 0.0 {
                1.0 + nu_t.get(elem_id).copied().unwrap_or(0.0) / nu
            } else {
                1.0
            };
            coo.add(r, c, v * nu_eff_ratio);
        }
    }
    coo.into_csr()
}

/// k-ω SST model: compute eddy viscosity from k and ω.
///
/// νₜ = (a₁·k) / max(a₁·ω, F₂·|S|)
/// where a₁ = 0.31, F₂ is the SST blending function.
pub fn k_omega_sst_eddy_viscosity(k: f64, omega: f64, strain_mag: f64) -> f64 {
    let a1 = 0.31;
    let f2 = 1.0; // simplified: fully turbulent
    let denom = (a1 * omega).max(f2 * strain_mag);
    if denom > 1e-30 { a1 * k / denom } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smagorinsky_nonzero() {
        let nu_t = smagorinsky_eddy_viscosity(0.1, 10.0, 0.1);
        assert!(nu_t > 0.0);
    }

    #[test]
    fn k_omega_sst_nonzero() {
        let nu_t = k_omega_sst_eddy_viscosity(1.0, 10.0, 5.0);
        assert!(nu_t > 0.0);
    }
}
