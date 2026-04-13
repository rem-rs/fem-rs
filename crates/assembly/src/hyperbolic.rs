//! Baseline hyperbolic-form utilities for 1-D conservative systems.
//!
//! This module provides a practical starter for DG/FV-style explicit transport
//! stepping used by Euler examples.

/// Numerical flux options for interface coupling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericalFlux {
    /// Local Lax-Friedrichs (Rusanov) flux.
    LaxFriedrichs,
    /// Roe approximate Riemann flux with a small entropy fix.
    Roe,
}

/// Baseline hyperbolic integrator for 1-D Euler equations.
#[derive(Debug, Clone, Copy)]
pub struct HyperbolicFormIntegrator {
    /// Ratio of specific heats.
    pub gamma: f64,
    /// Interface numerical flux scheme.
    pub flux: NumericalFlux,
}

impl Default for HyperbolicFormIntegrator {
    fn default() -> Self {
        Self {
            gamma: 1.4,
            flux: NumericalFlux::LaxFriedrichs,
        }
    }
}

impl HyperbolicFormIntegrator {
    /// Convert primitive variables `(rho, u, p)` to conservative state.
    pub fn prim_to_cons(&self, rho: f64, u: f64, p: f64) -> [f64; 3] {
        let e = p / (self.gamma - 1.0) + 0.5 * rho * u * u;
        [rho, rho * u, e]
    }

    /// Convert conservative state to primitive `(rho, u, p)`.
    pub fn cons_to_prim(&self, q: &[f64; 3]) -> (f64, f64, f64) {
        let rho = q[0].max(1e-14);
        let u = q[1] / rho;
        let kinetic = 0.5 * rho * u * u;
        let p = ((self.gamma - 1.0) * (q[2] - kinetic)).max(1e-14);
        (rho, u, p)
    }

    /// Physical 1-D Euler flux `F(U)`.
    pub fn physical_flux_1d(&self, q: &[f64; 3]) -> [f64; 3] {
        let (rho, u, p) = self.cons_to_prim(q);
        [rho * u, rho * u * u + p, u * (q[2] + p)]
    }

    /// Maximum characteristic speed `|u|+a` for CFL estimation.
    pub fn max_wave_speed_1d(&self, q: &[f64; 3]) -> f64 {
        let (rho, u, p) = self.cons_to_prim(q);
        let a = (self.gamma * p / rho).sqrt();
        u.abs() + a
    }

    /// Interface numerical flux between left and right conservative states.
    pub fn numerical_flux_1d(&self, ql: &[f64; 3], qr: &[f64; 3]) -> [f64; 3] {
        match self.flux {
            NumericalFlux::LaxFriedrichs => self.lax_friedrichs_flux(ql, qr),
            NumericalFlux::Roe => self.roe_flux(ql, qr),
        }
    }

    fn lax_friedrichs_flux(&self, ql: &[f64; 3], qr: &[f64; 3]) -> [f64; 3] {
        let fl = self.physical_flux_1d(ql);
        let fr = self.physical_flux_1d(qr);
        let alpha = self.max_wave_speed_1d(ql).max(self.max_wave_speed_1d(qr));

        [
            0.5 * (fl[0] + fr[0]) - 0.5 * alpha * (qr[0] - ql[0]),
            0.5 * (fl[1] + fr[1]) - 0.5 * alpha * (qr[1] - ql[1]),
            0.5 * (fl[2] + fr[2]) - 0.5 * alpha * (qr[2] - ql[2]),
        ]
    }

    fn roe_flux(&self, ql: &[f64; 3], qr: &[f64; 3]) -> [f64; 3] {
        let fl = self.physical_flux_1d(ql);
        let fr = self.physical_flux_1d(qr);

        let (rho_l, u_l, p_l) = self.cons_to_prim(ql);
        let (rho_r, u_r, p_r) = self.cons_to_prim(qr);
        let h_l = (ql[2] + p_l) / rho_l;
        let h_r = (qr[2] + p_r) / rho_r;

        let sr_l = rho_l.sqrt();
        let sr_r = rho_r.sqrt();
        let denom = (sr_l + sr_r).max(1e-14);
        let u_t = (sr_l * u_l + sr_r * u_r) / denom;
        let h_t = (sr_l * h_l + sr_r * h_r) / denom;
        let a_t2 = ((self.gamma - 1.0) * (h_t - 0.5 * u_t * u_t)).max(1e-14);
        let a_t = a_t2.sqrt();

        let du0 = qr[0] - ql[0];
        let du1 = qr[1] - ql[1];
        let du2 = qr[2] - ql[2];

        let alpha2 = ((self.gamma - 1.0) / a_t2)
            * ((h_t - u_t * u_t) * du0 + u_t * du1 - du2);
        let alpha1 = (du0 * (u_t + a_t) - du1 - a_t * alpha2) / (2.0 * a_t);
        let alpha3 = du0 - alpha1 - alpha2;

        let mut l1 = (u_t - a_t).abs();
        let mut l2 = u_t.abs();
        let mut l3 = (u_t + a_t).abs();
        let eps = 0.05 * a_t;
        if l1 < eps { l1 = 0.5 * (l1 * l1 / eps + eps); }
        if l2 < eps { l2 = 0.5 * (l2 * l2 / eps + eps); }
        if l3 < eps { l3 = 0.5 * (l3 * l3 / eps + eps); }

        let r1 = [1.0, u_t - a_t, h_t - u_t * a_t];
        let r2 = [1.0, u_t, 0.5 * u_t * u_t];
        let r3 = [1.0, u_t + a_t, h_t + u_t * a_t];

        let diss0 = l1 * alpha1 * r1[0] + l2 * alpha2 * r2[0] + l3 * alpha3 * r3[0];
        let diss1 = l1 * alpha1 * r1[1] + l2 * alpha2 * r2[1] + l3 * alpha3 * r3[1];
        let diss2 = l1 * alpha1 * r1[2] + l2 * alpha2 * r2[2] + l3 * alpha3 * r3[2];

        [
            0.5 * (fl[0] + fr[0]) - 0.5 * diss0,
            0.5 * (fl[1] + fr[1]) - 0.5 * diss1,
            0.5 * (fl[2] + fr[2]) - 0.5 * diss2,
        ]
    }

    /// Compute semi-discrete finite-volume residual with periodic boundaries.
    pub fn fv_residual_periodic(&self, q: &[[f64; 3]], dx: f64, out: &mut [[f64; 3]]) {
        assert_eq!(q.len(), out.len());
        let n = q.len();
        let inv_dx = 1.0 / dx;

        for i in 0..n {
            let il = if i == 0 { n - 1 } else { i - 1 };
            let ir = if i + 1 == n { 0 } else { i + 1 };

            let f_l = self.numerical_flux_1d(&q[il], &q[i]);
            let f_r = self.numerical_flux_1d(&q[i], &q[ir]);
            out[i][0] = -(f_r[0] - f_l[0]) * inv_dx;
            out[i][1] = -(f_r[1] - f_l[1]) * inv_dx;
            out[i][2] = -(f_r[2] - f_l[2]) * inv_dx;
        }
    }

    /// Advance one SSP-RK2 step for periodic 1-D FV state.
    pub fn step_ssprk2_periodic(&self, q: &mut [[f64; 3]], dx: f64, dt: f64) {
        let n = q.len();
        let mut k1 = vec![[0.0; 3]; n];
        let mut q1 = q.to_vec();
        self.fv_residual_periodic(q, dx, &mut k1);
        for i in 0..n {
            q1[i][0] = q[i][0] + dt * k1[i][0];
            q1[i][1] = q[i][1] + dt * k1[i][1];
            q1[i][2] = q[i][2] + dt * k1[i][2];
        }

        let mut k2 = vec![[0.0; 3]; n];
        self.fv_residual_periodic(&q1, dx, &mut k2);
        for i in 0..n {
            q[i][0] = 0.5 * q[i][0] + 0.5 * (q1[i][0] + dt * k2[i][0]);
            q[i][1] = 0.5 * q[i][1] + 0.5 * (q1[i][1] + dt * k2[i][1]);
            q[i][2] = 0.5 * q[i][2] + 0.5 * (q1[i][2] + dt * k2[i][2]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{HyperbolicFormIntegrator, NumericalFlux};

    #[test]
    fn lax_flux_consistency() {
        let h = HyperbolicFormIntegrator::default();
        let q = h.prim_to_cons(1.0, 2.0, 1.0);
        let f_num = h.numerical_flux_1d(&q, &q);
        let f_phy = h.physical_flux_1d(&q);
        for i in 0..3 {
            assert!((f_num[i] - f_phy[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn roe_flux_is_finite() {
        let h = HyperbolicFormIntegrator { gamma: 1.4, flux: NumericalFlux::Roe };
        let ql = h.prim_to_cons(1.0, 0.75, 1.0);
        let qr = h.prim_to_cons(0.125, 0.0, 0.1);
        let f = h.numerical_flux_1d(&ql, &qr);
        assert!(f.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn periodic_mass_conservation() {
        let h = HyperbolicFormIntegrator::default();
        let n = 64;
        let dx = 1.0 / n as f64;
        let mut q = vec![[0.0; 3]; n];
        for (i, qi) in q.iter_mut().enumerate() {
            let x = (i as f64 + 0.5) * dx;
            let rho = 1.0 + 0.1 * (2.0 * std::f64::consts::PI * x).sin();
            *qi = h.prim_to_cons(rho, 1.0, 1.0);
        }

        let m0: f64 = q.iter().map(|qi| qi[0]).sum::<f64>() * dx;
        h.step_ssprk2_periodic(&mut q, dx, 2e-3);
        let m1: f64 = q.iter().map(|qi| qi[0]).sum::<f64>() * dx;
        assert!((m1 - m0).abs() < 1e-10, "mass drift too large: {}", (m1 - m0).abs());
    }
}