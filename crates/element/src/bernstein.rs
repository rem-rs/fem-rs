//! Bernstein polynomial basis.
//!
//! B_{i,p}(t) = C(p,i) · t^i · (1-t)^{p-i},  t ∈ [0,1]
//!
//! Used for Bézier curves, IGA, and as an alternative well-conditioned
//! basis for high-order finite elements.

/// Evaluate all degree-p Bernstein basis polynomials at t ∈ [0,1].
/// values[i] = B_{i,p}(t)
pub fn bernstein_vals(p: usize, t: f64) -> Vec<f64> {
    let mut v = vec![0.0; p + 1];
    match p {
        0 => v[0] = 1.0,
        1 => {
            v[0] = 1.0 - t;
            v[1] = t;
        }
        _ => {
            // de Casteljau / recurrence: B_{i,p} = (1-t)·B_{i,p-1} + t·B_{i-1,p-1}
            let mut prev = vec![1.0]; // p=0
            for deg in 1..=p {
                let mut cur = vec![0.0; deg + 1];
                cur[0] = (1.0 - t) * prev[0];
                for i in 1..deg {
                    cur[i] = (1.0 - t) * prev[i] + t * prev[i - 1];
                }
                cur[deg] = t * prev[deg - 1];
                prev = cur;
            }
            v.copy_from_slice(&prev);
        }
    }
    v
}

/// Evaluate all degree-p Bernstein basis derivatives at t ∈ [0,1].
/// ders[i] = d/dt B_{i,p}(t)
pub fn bernstein_ders(p: usize, t: f64) -> Vec<f64> {
    if p == 0 {
        return vec![0.0];
    }
    let v_low = bernstein_vals(p - 1, t);
    let mut d = Vec::with_capacity(p + 1);
    let pf = p as f64;
    d.push(-pf * v_low[0]);
    for i in 1..p {
        d.push(pf * (v_low[i - 1] - v_low[i]));
    }
    d.push(pf * v_low[p - 1]);
    d
}

/// Evaluate all degree-p Bernstein basis second derivatives at t ∈ [0,1].
/// dders[i] = d²/dt² B_{i,p}(t)
///
/// Uses the closed-form recurrence:
///   d²B_{p,i}/dt² = p·(p-1)·(B_{p-2,i-2} - 2·B_{p-2,i-1} + B_{p-2,i})
/// where B_{p,i} = 0 for i < 0 or i > p.
pub fn bernstein_dders(p: usize, t: f64) -> Vec<f64> {
    if p < 2 {
        return vec![0.0; p + 1];
    }
    let v_low = bernstein_vals(p - 2, t);
    let pf = p as f64 * (p - 1) as f64;
    let mut d = Vec::with_capacity(p + 1);
    for i in 0..=p {
        let b1 = if i >= 2 { v_low[i - 2] } else { 0.0 };
        let b2 = if i >= 1 && i <= p - 1 {
            v_low[i - 1]
        } else {
            0.0
        };
        let b3 = if i <= p - 2 { v_low[i] } else { 0.0 };
        d.push(pf * (b1 - 2.0 * b2 + b3));
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bernstein_p0() {
        let v = bernstein_vals(0, 0.5);
        assert!((v[0] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn bernstein_p1() {
        let v = bernstein_vals(1, 0.3);
        assert!((v[0] - 0.7).abs() < 1e-14);
        assert!((v[1] - 0.3).abs() < 1e-14);
    }

    #[test]
    fn bernstein_p2_endpoints() {
        let v0 = bernstein_vals(2, 0.0);
        assert!((v0[0] - 1.0).abs() < 1e-14);
        assert!((v0[1] - 0.0).abs() < 1e-14);
        assert!((v0[2] - 0.0).abs() < 1e-14);
        let v1 = bernstein_vals(2, 1.0);
        assert!((v1[2] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn bernstein_partition_of_unity() {
        for p in 0..=8 {
            for &t in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
                let v = bernstein_vals(p, t);
                let s: f64 = v.iter().sum();
                assert!((s - 1.0).abs() < 1e-13, "p={p} t={t}: sum={s}");
            }
        }
    }

    #[test]
    fn bernstein_ders_fd() {
        let h = 1e-8;
        for p in 1..=5 {
            for &t in &[0.1, 0.3, 0.5, 0.7, 0.9] {
                let d_analytic = bernstein_ders(p, t);
                let vp = bernstein_vals(p, t + h);
                let vm = bernstein_vals(p, t - h);
                for i in 0..=p {
                    let fd = (vp[i] - vm[i]) / (2.0 * h);
                    assert!(
                        (d_analytic[i] - fd).abs() < 1e-7,
                        "p={p} t={t} i={i}: analytic={} fd={}",
                        d_analytic[i],
                        fd
                    );
                }
            }
        }
    }

    #[test]
    fn bernstein_dders_fd() {
        let h = 1e-8;
        for p in 2..=6 {
            // Avoid t very close to 0 or 1 where high-degree derivatives amplify FD error
            for &t in &[0.1, 0.3, 0.5, 0.7] {
                let dd_analytic = bernstein_dders(p, t);
                let dp = bernstein_ders(p, t + h);
                let dm = bernstein_ders(p, t - h);
                for i in 0..=p {
                    let fd = (dp[i] - dm[i]) / (2.0 * h);
                    assert!(
                        (dd_analytic[i] - fd).abs() < 1e-6,
                        "p={p} t={t} i={i}: analytic={} fd={}",
                        dd_analytic[i],
                        fd
                    );
                }
            }
        }
    }

    #[test]
    fn bernstein_dders_p0_p1_zero() {
        for p in 0..=1 {
            let dd = bernstein_dders(p, 0.5);
            for i in 0..=p {
                assert!((dd[i] - 0.0).abs() < 1e-14, "p={p} i={i}: dd={}", dd[i]);
            }
        }
    }

}
