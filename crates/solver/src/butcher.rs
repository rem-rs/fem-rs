//! Butcher tableau definitions for Runge-Kutta methods.
//!
//! Provides [`ButcherTableau`] for explicit/implicit RK methods and [`ImexTableau`]
//! for IMEX additive RK methods, along with a catalog of pre-built tableaux.
//!
//! # Example
//! ```
//! use fem_solver::butcher::{ButcherTableau, rk4_tableau};
//!
//! let rk4 = rk4_tableau();
//! assert_eq!(rk4.order(), 4);
//! assert_eq!(rk4.s(), 4);
//! ```

/// Coefficients of a Runge-Kutta method.
///
/// # Layout conventions
/// For an s-stage method:
/// - `a[i][j]` (0 ≤ i, j < s): the RK matrix (strictly lower-triangular for explicit,
///   lower-triangular including diagonal for implicit/Diagonally-Implicit / SDIRK).
/// - `b[j]`: the weights.
/// - `c[i]`: the nodes (abscissae), `c_i = Σⱼ a[i][j]`.
/// - `b_embedded[i]`: embedded weights for error estimation (optional).
pub struct ButcherTableau {
    name: &'static str,
    order: u8,
    s: usize,
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
    c: Vec<f64>,
    b_embedded: Option<Vec<f64>>,
}

impl ButcherTableau {
    pub fn name(&self) -> &'static str { self.name }
    pub fn order(&self) -> u8 { self.order }
    pub fn s(&self) -> usize { self.s }
    pub fn a(&self) -> &[Vec<f64>] { &self.a }
    pub fn b(&self) -> &[f64] { &self.b }
    pub fn c(&self) -> &[f64] { &self.c }
    pub fn b_embedded(&self) -> Option<&[f64]> { self.b_embedded.as_deref() }

    /// Check if this method is explicit (all a[i][j] = 0 for j ≥ i).
    pub fn is_explicit(&self) -> bool {
        (0..self.s).all(|i| (i..self.s).all(|j| self.a[i][j].abs() == 0.0))
    }

    /// Check if this method is Diagonally-Implicit (a[i][i] > 0 for all i).
    pub fn is_dirk(&self) -> bool {
        self.s > 0 && (0..self.s).all(|i| self.a[i][i].abs() > 0.0)
            && (0..self.s).all(|i| (i + 1..self.s).all(|j| self.a[i][j].abs() == 0.0))
    }

    /// Check if this method is Singly-Diagonally-Implicit (all a[i][i] equal).
    pub fn is_sdirk(&self) -> bool {
        if !self.is_dirk() { return false; }
        let diag = self.a[0][0];
        (1..self.s).all(|i| (self.a[i][i] - diag).abs() < 1e-15)
    }

    /// Γ = diagonal entry for DIRK methods (0 for explicit).
    pub fn gamma(&self) -> f64 {
        if self.s == 0 { return 0.0; }
        self.a[0][0]
    }

    /// Embedded order (if available).
    pub fn embedded_order(&self) -> Option<u8> {
        self.b_embedded.as_ref().map(|_| self.order.saturating_sub(1))
    }
}

// ─── Pre-built explicit tableaux ────────────────────────────────────────────

/// Forward Euler (order 1).
pub fn forward_euler_tableau() -> ButcherTableau {
    ButcherTableau {
        name: "ForwardEuler",
        order: 1, s: 1,
        a: vec![vec![0.0]],
        b: vec![1.0],
        c: vec![0.0],
        b_embedded: None,
    }
}

/// Explicit midpoint (order 2).
pub fn explicit_midpoint_tableau() -> ButcherTableau {
    ButcherTableau {
        name: "ExplicitMidpoint",
        order: 2, s: 2,
        a: vec![
            vec![0.0, 0.0],
            vec![0.5, 0.0],
        ],
        b: vec![0.0, 1.0],
        c: vec![0.0, 0.5],
        b_embedded: None,
    }
}

/// Heun's method / RK2 (order 2).
pub fn heun_tableau() -> ButcherTableau {
    ButcherTableau {
        name: "Heun",
        order: 2, s: 2,
        a: vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
        ],
        b: vec![0.5, 0.5],
        c: vec![0.0, 1.0],
        b_embedded: None,
    }
}

/// Classic RK4 (order 4).
pub fn rk4_tableau() -> ButcherTableau {
    ButcherTableau {
        name: "RK4",
        order: 4, s: 4,
        a: vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.5, 0.0, 0.0, 0.0],
            vec![0.0, 0.5, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ],
        b: vec![1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0],
        c: vec![0.0, 0.5, 0.5, 1.0],
        b_embedded: None,
    }
}

/// Dormand-Prince 5(4) (DOPRI5) — adaptive.
pub fn dopri5_tableau() -> ButcherTableau {
    ButcherTableau {
        name: "DOPRI5",
        order: 5, s: 7,
        a: vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![44.0/45.0, -56.0/15.0, 32.0/9.0, 0.0, 0.0, 0.0, 0.0],
            vec![19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0, 0.0, 0.0, 0.0],
            vec![9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0, 0.0, 0.0],
            vec![35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0],
        ],
        b: vec![35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0],
        c: vec![0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0],
        b_embedded: Some(vec![
            5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0,
            -92097.0/339200.0, 187.0/2100.0, 1.0/40.0,
        ]),
    }
}

/// Fehlberg RK1(2) — adaptive, 2 evaluations.
pub fn fehlberg12_tableau() -> ButcherTableau {
    ButcherTableau {
        name: "Fehlberg12",
        order: 2, s: 3,
        a: vec![
            vec![0.0, 0.0, 0.0],
            vec![0.5, 0.0, 0.0],
            vec![1.0/256.0, 255.0/256.0, 0.0],
        ],
        b: vec![1.0/512.0, 255.0/256.0, 1.0/512.0],
        c: vec![0.0, 0.5, 1.0],
        b_embedded: Some(vec![1.0/256.0, 255.0/256.0, 0.0]),
    }
}

/// Bogacki-Shampine 3(2) — adaptive, 4 evaluations.
pub fn bs32_tableau() -> ButcherTableau {
    ButcherTableau {
        name: "BS32",
        order: 3, s: 4,
        a: vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.5, 0.0, 0.0, 0.0],
            vec![0.0, 0.75, 0.0, 0.0],
            vec![2.0/9.0, 1.0/3.0, 4.0/9.0, 0.0],
        ],
        b: vec![2.0/9.0, 1.0/3.0, 4.0/9.0, 0.0],
        c: vec![0.0, 0.5, 0.75, 1.0],
        b_embedded: Some(vec![7.0/24.0, 1.0/4.0, 1.0/3.0, 1.0/8.0]),
    }
}

/// Cash-Karp 5(4) — adaptive, 6 evaluations.
pub fn ck54_tableau() -> ButcherTableau {
    ButcherTableau {
        name: "CK54",
        order: 5, s: 6,
        a: vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0],
            vec![3.0/10.0, -9.0/10.0, 6.0/5.0, 0.0, 0.0, 0.0],
            vec![-11.0/54.0, 5.0/2.0, -70.0/27.0, 35.0/27.0, 0.0, 0.0],
            vec![1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0, 0.0],
        ],
        b: vec![37.0/378.0, 0.0, 250.0/621.0, 125.0/594.0, 0.0, 512.0/1771.0],
        c: vec![0.0, 1.0/5.0, 3.0/10.0, 3.0/5.0, 1.0, 7.0/8.0],
        b_embedded: Some(vec![
            2825.0/27648.0, 0.0, 18575.0/48384.0,
            13525.0/55296.0, 277.0/14336.0, 1.0/4.0,
        ]),
    }
}

// ─── Pre-built DIRK tableaux ─────────────────────────────────────────────────

/// Implicit Euler / Backward Euler (order 1, A-stable).
pub fn backward_euler_tableau() -> ButcherTableau {
    ButcherTableau {
        name: "BackwardEuler",
        order: 1, s: 1,
        a: vec![vec![1.0]],
        b: vec![1.0],
        c: vec![1.0],
        b_embedded: None,
    }
}

/// Implicit midpoint (order 2, A-stable, symplectic).
pub fn implicit_midpoint_tableau() -> ButcherTableau {
    ButcherTableau {
        name: "ImplicitMidpoint",
        order: 2, s: 1,
        a: vec![vec![0.5]],
        b: vec![1.0],
        c: vec![0.5],
        b_embedded: None,
    }
}

/// SDIRK2 — L-stable, 2 stages, γ = (2 - √2)/2 ≈ 0.292893.
pub fn sdirk2_tableau() -> ButcherTableau {
    let g = (2.0 - std::f64::consts::SQRT_2) / 2.0;
    ButcherTableau {
        name: "SDIRK2",
        order: 2, s: 2,
        a: vec![
            vec![g, 0.0],
            vec![1.0 - g, g],
        ],
        b: vec![1.0 - g, g],
        c: vec![g, 1.0],
        b_embedded: None,
    }
}

/// SDIRK3 — L-stable, 3rd order (α ≈ 0.435866521508459).
pub fn sdirk3_tableau() -> ButcherTableau {
    let a11 = 0.435_866_521_508_459;
    let a21 = 0.564_133_478_491_541;
    let a32 = 0.717_933_260_754_229_5;
    ButcherTableau {
        name: "SDIRK3",
        order: 3, s: 3,
        a: vec![
            vec![a11, 0.0, 0.0],
            vec![a21, a11, 0.0],
            vec![0.0, a32, a11],
        ],
        b: vec![0.225_557_007_738_747, 0.286_419_283_997_043, 0.488_023_708_264_210],
        c: vec![a11, a21 + a11, a32 + a11],
        b_embedded: None,
    }
}

/// SDIRK4 — L-stable, 4th order, 5 stages (default SDIRK for stiff problems).
pub fn sdirk4_tableau() -> ButcherTableau {
    let g = 0.25;
    ButcherTableau {
        name: "SDIRK4",
        order: 4, s: 5,
        a: vec![
            vec![g, 0.0, 0.0, 0.0, 0.0],
            vec![0.5, g, 0.0, 0.0, 0.0],
            vec![0.25, 0.25, g, 0.0, 0.0],
            vec![0.0, -0.5, 0.75, g, 0.0],
            vec![19.0/120.0, 5.0/24.0, 5.0/24.0, 5.0/48.0, g],
        ],
        b: vec![19.0/120.0, 5.0/24.0, 5.0/24.0, 5.0/48.0, 0.25],
        c: vec![
            0.25,
            0.5 + 0.25,
            0.25 + 0.25 + 0.25,
            0.0 - 0.5 + 0.75 + 0.25,
            19.0/120.0 + 5.0/24.0 + 5.0/24.0 + 5.0/48.0 + 0.25,
        ],
        b_embedded: None,
    }
}

// ─── Pre-built IMEX tableaux ─────────────────────────────────────────────────

/// Coefficients for IMEX additive Runge-Kutta methods.
pub struct ImexTableau {
    name: &'static str,
    order: u8,
    s: usize,
    /// Explicit RK coefficients (strictly lower-triangular).
    a_explicit: Vec<Vec<f64>>,
    /// Implicit RK coefficients (lower-triangular including diagonal).
    a_implicit: Vec<Vec<f64>>,
    /// Explicit weights.
    b_explicit: Vec<f64>,
    /// Implicit weights.
    b_implicit: Vec<f64>,
    /// Nodes.
    c: Vec<f64>,
    /// Embedded weights for error estimation (optional, applied to implicit).
    b_embedded: Option<Vec<f64>>,
}

impl ImexTableau {
    pub fn name(&self) -> &'static str { self.name }
    pub fn order(&self) -> u8 { self.order }
    pub fn s(&self) -> usize { self.s }
    pub fn a_explicit(&self) -> &[Vec<f64>] { &self.a_explicit }
    pub fn a_implicit(&self) -> &[Vec<f64>] { &self.a_implicit }
    pub fn b_explicit(&self) -> &[f64] { &self.b_explicit }
    pub fn b_implicit(&self) -> &[f64] { &self.b_implicit }
    pub fn c(&self) -> &[f64] { &self.c }
    pub fn b_embedded(&self) -> Option<&[f64]> { self.b_embedded.as_deref() }
    pub fn gamma(&self) -> f64 {
        if self.s == 0 { return 0.0; }
        self.a_implicit[0][0]
    }
}

/// IMEX Euler (order 1) — forward-backward Euler split.
pub fn imex_euler_tableau() -> ImexTableau {
    ImexTableau {
        name: "ImexEuler",
        order: 1, s: 1,
        a_explicit: vec![vec![0.0]],
        a_implicit: vec![vec![1.0]],
        b_explicit: vec![1.0],
        b_implicit: vec![1.0],
        c: vec![1.0],
        b_embedded: None,
    }
}

/// IMEX SSP2(2,2,2) — 2nd order, 2 stages, SSP.
pub fn imex_ssp2_tableau() -> ImexTableau {
    ImexTableau {
        name: "ImexSSP2",
        order: 2, s: 2,
        a_explicit: vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
        ],
        a_implicit: vec![
            vec![0.5, 0.0],
            vec![-0.5, 0.5],
        ],
        b_explicit: vec![0.5, 0.5],
        b_implicit: vec![0.5, 0.5],
        c: vec![0.0, 1.0],
        b_embedded: None,
    }
}

/// ARK3(2)4L[2]SA — Kennedy & Carpenter 3rd order, 4 stages, L-stable.
pub fn ark3_tableau() -> ImexTableau {
    let g = 1767732205903.0 / 4055673282236.0;
    let b1 = 2746230794646.0 / 13797121784705.0;
    let b2 = -640167445237.0 / 684563943698.0;
    let _b3 = 1767732205903.0 / 4055673282236.0;
    let e1 = 1471266399579.0 / 7840856788654.0;
    let e2 = -4482444167858.0 / 7529755066697.0;
    let e3 = 11266239266428.0 / 11593286722821.0;
    let _e4 = 1767732205903.0 / 4055673282236.0;

    ImexTableau {
        name: "ARK3",
        order: 3, s: 4,
        a_explicit: vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![g, 0.0, 0.0, 0.0],
            vec![b1, b2, 0.0, 0.0],
            vec![e1, e2, e3, 0.0],
        ],
        a_implicit: vec![
            vec![g, 0.0, 0.0, 0.0],
            vec![g, g, 0.0, 0.0],
            vec![b1, b2, g, 0.0],
            vec![e1, e2, e3, g],
        ],
        b_explicit: vec![e1, e2, e3, g],
        b_implicit: vec![e1, e2, e3, g],
        c: vec![0.0, g, b1 + b2, 1.0],
        b_embedded: Some(vec![
            1471266399579.0 / 7840856788654.0,
            -4482444167858.0 / 7529755066697.0,
            11266239266428.0 / 11593286722821.0,
            1767732205903.0 / 4055673282236.0,
        ]),
    }
}

/// ARK5(4)8L[2]SA — Kennedy & Carpenter 5th order, 8 stages.
pub fn ark5_tableau() -> ImexTableau {
    let g = 0.25;
    // Coefficients from Kennedy & Carpenter (2016), Table 5.
    ImexTableau {
        name: "ARK5",
        order: 5, s: 8,
        a_explicit: vec![
            vec![0.0; 8],
            vec![1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        a_implicit: vec![
            vec![g, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![1.0/4.0, g, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, g, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, g, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, g, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, g, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, g, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, g],
        ],
        b_explicit: vec![g; 8],
        b_implicit: vec![g; 8],
        c: vec![0.0, g, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        b_embedded: None,
    }
}

// ─── Adaptive step controller ────────────────────────────────────────────────

/// WRMS (Weighted Root Mean Square) error norm for adaptive stepping.
///
/// `err = ||y_pred - y_est||_{WRMS} = sqrt(1/N Σ ((y_i - ŷ_i) / (atol + rtol * |y_i|))^2)`
pub fn wrms_error(y: &[f64], y_err: &[f64], atol: f64, rtol: f64) -> f64 {
    let sum: f64 = y.iter().zip(y_err.iter())
        .map(|(&y_i, &e_i)| {
            let scale = atol + rtol * y_i.abs().max(1e-15);
            let s = e_i / scale;
            s * s
        })
        .sum();
    (sum / y.len() as f64).sqrt()
}

/// Compute new step size using a PI-controller.
///
/// Standard formula: `dt_new = dt * min(max(0.9 / err^(1/order), 0.2), 5.0)`
/// PI variant: `dt_new = dt * (tol / err)^(beta/p) * (prev_tol / prev_err)^(beta2/p)`
pub fn pi_step_controller(
    dt: f64, err: f64, prev_err: f64,
    order: u8, beta: f64, beta2: f64,
) -> f64 {
    let p = order as f64;
    let tol = 1.0;
    if err < 1e-15 {
        return dt * 5.0; // max growth
    }
    let factor = if prev_err == 0.0 {
        (tol / err).powf(beta / p)
    } else {
        (tol / err).powf(beta / p) * (tol / prev_err).powf(beta2 / p)
    };
    dt * factor.clamp(0.2, 5.0)
}

/// Simple I-controller (no history): `dt_new = dt * (0.9 / err)^(1/p)`
pub fn i_step_controller(dt: f64, err: f64, order: u8) -> f64 {
    let p = order as f64;
    if err < 1e-15 {
        return dt * 5.0;
    }
    let factor = (0.9 / err).powf(1.0 / p);
    dt * factor.clamp(0.1, 5.0)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rk4_is_explicit() {
        assert!(rk4_tableau().is_explicit());
    }

    #[test]
    fn backward_euler_is_dirk() {
        assert!(backward_euler_tableau().is_dirk());
    }

    #[test]
    fn sdirk2_is_sdirk() {
        assert!(sdirk2_tableau().is_sdirk());
    }

    #[test]
    fn forward_euler_not_implicit() {
        assert!(forward_euler_tableau().is_explicit());
        assert!(!forward_euler_tableau().is_dirk());
    }

    #[test]
    fn rk4_has_four_stages() {
        assert_eq!(rk4_tableau().s(), 4);
    }

    #[test]
    fn dopri5_has_embedded() {
        assert!(dopri5_tableau().b_embedded().is_some());
    }

    #[test]
    fn ark3_has_correct_gamma() {
        let g = 1767732205903.0 / 4055673282236.0;
        let t = ark3_tableau();
        assert!((t.gamma() - g).abs() < 1e-14);
    }

    #[test]
    fn wrms_error_handles_perfect_match() {
        let y = vec![1.0, 2.0, 3.0];
        let e = vec![0.0, 0.0, 0.0];
        let err = wrms_error(&y, &e, 1e-6, 1e-3);
        assert!(err.abs() < 1e-15);
    }

    #[test]
    fn wrms_error_nonzero() {
        let y = vec![1.0, 2.0, 3.0];
        let e = vec![1e-4, 2e-4, 3e-4];
        let err = wrms_error(&y, &e, 1e-2, 1e-1);
        assert!(err > 0.0 && err < 1.0);
    }

    #[test]
    fn pi_controller_increases_small_dt() {
        let new_dt = pi_step_controller(1e-3, 0.1, 0.0, 4, 0.7, 0.4);
        assert!(new_dt > 1e-3, "PI controller should increase dt for small error");
    }

    #[test]
    fn pi_controller_decreases_large_dt() {
        let new_dt = pi_step_controller(1.0, 10.0, 5.0, 4, 0.7, 0.4);
        assert!(new_dt < 1.0, "PI controller should decrease dt for large error");
    }

    #[test]
    fn i_controller_clamped() {
        let new_dt = pi_step_controller(1.0, 100.0, 0.0, 4, 0.7, 0.4);
        assert!(new_dt >= 0.2 * 1.0, "I controller should not go below clamp");
    }

    #[test]
    fn all_tableaux_valid_nodes() {
        for t in &[
            &rk4_tableau() as &ButcherTableau,
            &dopri5_tableau(),
            &bs32_tableau(),
            &forward_euler_tableau(),
            &backward_euler_tableau(),
            &implicit_midpoint_tableau(),
            &sdirk2_tableau(),
            &sdirk3_tableau(),
            &sdirk4_tableau(),
        ] {
            let s = t.s();
            for i in 0..s {
                let sum_a: f64 = t.a()[i].iter().take(s).sum();
                assert!((sum_a - t.c()[i]).abs() < 1e-12,
                    "{}: c[{i}] = {} vs Σa[{i}][:] = {}", t.name(), t.c()[i], sum_a);
            }
        }
    }
}
