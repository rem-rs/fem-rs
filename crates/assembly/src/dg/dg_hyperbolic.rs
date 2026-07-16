//! DG time-domain operator for hyperbolic conservation laws.
//!
//! Provides [`FluxFunction`] trait, [`EulerFlux`], [`RusanovFlux`],
//! and [`DgHyperbolicConservationLaws`].
//!
//! ## Reference
//! MFEM examples/ex18.hpp — DGHyperbolicConservationLaws

/// Physical flux function for hyperbolic conservation laws.
pub trait FluxFunction: Send + Sync {
    fn num_equations(&self) -> usize;
    fn compute_flux(&self, state: &[f64], point: &[f64], flux_out: &mut [f64]);
    fn max_speed(&self, state: &[f64], normal: &[f64]) -> f64;
    fn numerical_flux(&self, ql: &[f64], qr: &[f64], normal: &[f64]) -> Vec<f64>;
}

// ─── EulerFlux ──────────────────────────────────────────────────────────────────

/// Convert conserved variables to primitive variables.
fn cons_to_prim(q: &[f64], gamma: f64) -> (f64, f64, f64, f64) {
    let rho = q[0].max(1e-14);
    let u = q[1] / rho;
    let v = q[2] / rho;
    let ke = 0.5 * rho * (u * u + v * v);
    let p = ((gamma - 1.0) * (q[3] - ke)).max(1e-14);
    (rho, u, v, p)
}

/// Convert primitive variables to conserved variables.
fn prim_to_cons(rho: f64, u: f64, v: f64, p: f64, gamma: f64) -> [f64; 4] {
    let e = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);
    [rho, rho * u, rho * v, e]
}

/// 2-D compressible Euler flux (4 equations).
///
/// Conserved variables: [ρ, ρu, ρv, E]
/// γ (specific heat ratio) defaults to 1.4 (air).
pub struct EulerFlux {
    pub gamma: f64,
}

impl Default for EulerFlux {
    fn default() -> Self {
        Self { gamma: 1.4 }
    }
}

impl FluxFunction for EulerFlux {
    fn num_equations(&self) -> usize {
        4
    }

    fn compute_flux(&self, state: &[f64], _point: &[f64], flux_out: &mut [f64]) {
        let (rho, u, v, p) = cons_to_prim(state, self.gamma);
        // flux_out interleaved by dim then equation:
        //   [F_x[ρ], F_y[ρ], F_x[ρu], F_y[ρu], F_x[ρv], F_y[ρv], F_x[E], F_y[E]]
        let E = state[3];
        flux_out[0] = state[1];                         // F_x[ρ]:  ρu
        flux_out[1] = state[2];                         // F_y[ρ]:  ρv
        flux_out[2] = rho * u * u + p;                  // F_x[ρu]: ρu² + p
        flux_out[3] = rho * u * v;                      // F_y[ρu]: ρuv
        flux_out[4] = rho * u * v;                      // F_x[ρv]: ρuv
        flux_out[5] = rho * v * v + p;                  // F_y[ρv]: ρv² + p
        flux_out[6] = u * (E + p);                      // F_x[E]:  u(E + p)
        flux_out[7] = v * (E + p);                      // F_y[E]:  v(E + p)
    }

    fn max_speed(&self, state: &[f64], normal: &[f64]) -> f64 {
        let (rho, u, v, p) = cons_to_prim(state, self.gamma);
        let a = (self.gamma * p / rho).sqrt();
        let vn = u * normal[0] + v * normal[1];
        let nlen = (normal[0] * normal[0] + normal[1] * normal[1]).sqrt();
        (if nlen > 0.0 { (vn / nlen).abs() } else { 0.0 }) + a
    }

    fn numerical_flux(&self, ql: &[f64], qr: &[f64], normal: &[f64]) -> Vec<f64> {
        let mut fl = [0.0_f64; 8];
        let mut fr = [0.0_f64; 8];
        self.compute_flux(ql, &[0.0, 0.0], &mut fl);
        self.compute_flux(qr, &[0.0, 0.0], &mut fr);
        // F_n = F_x · n_x + F_y · n_y  (component-wise for each equation)
        let mut fnl = [0.0_f64; 4];
        let mut fnr = [0.0_f64; 4];
        for eq in 0..4 {
            fnl[eq] = fl[eq * 2] * normal[0] + fl[eq * 2 + 1] * normal[1];
            fnr[eq] = fr[eq * 2] * normal[0] + fr[eq * 2 + 1] * normal[1];
        }
        let c = self.max_speed(ql, normal).max(self.max_speed(qr, normal));
        // ½(F_n(L) + F_n(R)) - ½·c·(qR - qL)
        let mut f = vec![0.0_f64; 4];
        for eq in 0..4 {
            f[eq] = 0.5 * (fnl[eq] + fnr[eq]) - 0.5 * c * (qr[eq] - ql[eq]);
        }
        f
    }
}

// ─── RusanovFlux ────────────────────────────────────────────────────────────────

/// Rusanov (local Lax-Friedrichs) numerical flux.
///
/// Wraps any `FluxFunction` — delegates `compute_flux` and `max_speed`
/// to the inner function, and `numerical_flux` calls `inner.numerical_flux`.
pub struct RusanovFlux<F: FluxFunction> {
    pub inner: F,
}

impl<F: FluxFunction> FluxFunction for RusanovFlux<F> {
    fn num_equations(&self) -> usize {
        self.inner.num_equations()
    }

    fn compute_flux(&self, state: &[f64], point: &[f64], flux_out: &mut [f64]) {
        self.inner.compute_flux(state, point, flux_out);
    }

    fn max_speed(&self, state: &[f64], normal: &[f64]) -> f64 {
        self.inner.max_speed(state, normal)
    }

    fn numerical_flux(&self, ql: &[f64], qr: &[f64], normal: &[f64]) -> Vec<f64> {
        self.inner.numerical_flux(ql, qr, normal)
    }
}
