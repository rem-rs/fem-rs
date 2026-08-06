#[derive(Clone, Debug)]
pub struct KnotVector {
    knots: Vec<f64>,
}

impl KnotVector {
    pub fn new_clamped(knots: Vec<f64>) -> Result<Self, String> {
        if knots.len() < 2 {
            return Err("knot vector must contain at least two knots".to_string());
        }
        if knots.iter().any(|k| !k.is_finite()) {
            return Err("knot vector entries must be finite".to_string());
        }
        if knots.windows(2).any(|w| w[0] > w[1]) {
            return Err("knot vector must be nondecreasing".to_string());
        }
        Ok(Self { knots })
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.knots
    }
}

#[derive(Clone, Debug)]
pub struct BsplineBasis {
    degree: usize,
    knots: KnotVector,
}

impl BsplineBasis {
    pub fn new(degree: usize, knots: KnotVector) -> Result<Self, String> {
        let n_knots = knots.as_slice().len();
        if n_knots < 2 * (degree + 1) {
            return Err("knot vector too short for requested degree".to_string());
        }
        let n_basis = n_knots.saturating_sub(degree + 1);
        if n_basis == 0 {
            return Err("invalid basis size".to_string());
        }
        if n_basis <= degree {
            return Err("invalid basis size for degree".to_string());
        }

        let u0 = knots.as_slice()[0];
        let u1 = knots.as_slice()[n_knots - 1];
        if u0 >= u1 {
            return Err("invalid knot range".to_string());
        }

        const CLAMP_EPS: f64 = 1e-12;
        let left_mult = knots
            .as_slice()
            .iter()
            .take_while(|&&u| (u - u0).abs() <= CLAMP_EPS)
            .count();
        let right_mult = knots
            .as_slice()
            .iter()
            .rev()
            .take_while(|&&u| (u - u1).abs() <= CLAMP_EPS)
            .count();

        if left_mult < degree + 1 || right_mult < degree + 1 {
            return Err(
                "clamped knot vector must repeat first/last knot at least degree+1 times"
                    .to_string(),
            );
        }

        Ok(Self { degree, knots })
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn n_basis(&self) -> usize {
        self.knots.as_slice().len() - self.degree - 1
    }

    pub fn find_span(&self, u: f64) -> usize {
        let p = self.degree;
        let n = self.n_basis() - 1;
        let knots = self.knots.as_slice();

        if u >= knots[n + 1] {
            return n;
        }
        if u <= knots[p] {
            return p;
        }

        let mut low = p;
        let mut high = n + 1;
        let mut mid = (low + high) / 2;
        while u < knots[mid] || u >= knots[mid + 1] {
            if u < knots[mid] {
                high = mid;
            } else {
                low = mid;
            }
            mid = (low + high) / 2;
        }
        mid
    }

    pub fn active_basis_indices(&self, span: usize) -> Vec<usize> {
        let first = span + 1 - (self.degree + 1);
        (first..=span).collect()
    }

    pub fn nonzero_values(&self, u: f64) -> Result<Vec<(usize, f64)>, String> {
        self.check_param(u, "nonzero_values")?;
        let p = self.degree;
        let span = self.find_span(u);
        let (values, _) = self.local_basis_and_first_derivatives(span, u);

        let first = span + 1 - (p + 1);
        Ok(values
            .into_iter()
            .enumerate()
            .map(|(j, v)| (first + j, v))
            .collect())
    }

    pub fn nonzero_derivatives(&self, u: f64) -> Result<Vec<(usize, f64)>, String> {
        self.check_param(u, "nonzero_derivatives")?;
        let p = self.degree;
        let span = self.find_span(u);
        let (_, derivatives) = self.local_basis_and_first_derivatives(span, u);
        let first = span + 1 - (p + 1);
        Ok(derivatives
            .into_iter()
            .enumerate()
            .map(|(j, dv)| (first + j, dv))
            .collect())
    }

    fn check_param(&self, u: f64, eval_name: &str) -> Result<(), String> {
        if !u.is_finite() {
            return Err(format!("{eval_name}: parameter must be finite, got u={u}"));
        }
        let knots = self.knots.as_slice();
        let umin = knots[self.degree];
        let umax = knots[self.n_basis()];
        if u < umin || u > umax {
            return Err(format!(
                "{eval_name}: parameter out of knot span range, got u={u}, valid=[{umin}, {umax}]"
            ));
        }
        Ok(())
    }

    fn local_basis_and_first_derivatives(&self, span: usize, u: f64) -> (Vec<f64>, Vec<f64>) {
        let p = self.degree;
        let knots = self.knots.as_slice();

        let mut ndu = vec![vec![0.0; p + 1]; p + 1];
        let mut left = vec![0.0; p + 1];
        let mut right = vec![0.0; p + 1];
        ndu[0][0] = 1.0;

        for j in 1..=p {
            left[j] = u - knots[span + 1 - j];
            right[j] = knots[span + j] - u;
            let mut saved = 0.0;
            for r in 0..j {
                ndu[j][r] = right[r + 1] + left[j - r];
                let temp = if ndu[j][r].abs() > f64::EPSILON {
                    ndu[r][j - 1] / ndu[j][r]
                } else {
                    0.0
                };
                ndu[r][j] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            ndu[j][j] = saved;
        }

        let mut values = vec![0.0; p + 1];
        for j in 0..=p {
            values[j] = ndu[j][p];
        }

        if p == 0 {
            return (values, vec![0.0]);
        }

        let mut ders = vec![0.0; p + 1];
        let mut a = vec![vec![0.0; p + 1]; 2];
        for r in 0..=p {
            let mut s1 = 0usize;
            let mut s2 = 1usize;
            a[0][0] = 1.0;
            let mut d = 0.0;

            let rk = r as isize - 1;
            let pk = p as isize - 1;
            if r >= 1 {
                a[s2][0] = if ndu[(pk + 1) as usize][rk as usize].abs() > f64::EPSILON {
                    a[s1][0] / ndu[(pk + 1) as usize][rk as usize]
                } else {
                    0.0
                };
                d = a[s2][0] * ndu[rk as usize][pk as usize];
            }

            let j1 = if rk >= -1 { 1 } else { (-rk) as usize };
            let j2 = if (r as isize) - 1 <= pk {
                0
            } else {
                (p as isize) - (r as isize)
            } as usize;

            for j in j1..=j2 {
                let denom = ndu[(pk + 1) as usize][(rk + j as isize) as usize];
                a[s2][j] = if denom.abs() > f64::EPSILON {
                    (a[s1][j] - a[s1][j - 1]) / denom
                } else {
                    0.0
                };
                d += a[s2][j] * ndu[(rk + j as isize) as usize][pk as usize];
            }

            if r <= pk as usize {
                a[s2][1] = if ndu[(pk + 1) as usize][r].abs() > f64::EPSILON {
                    -a[s1][0] / ndu[(pk + 1) as usize][r]
                } else {
                    0.0
                };
                d += a[s2][1] * ndu[r][pk as usize];
            }

            ders[r] = d * (p as f64);
            std::mem::swap(&mut s1, &mut s2);
        }

        (values, ders)
    }

    /// Insert a knot `u` into the basis (Böhm algorithm).
    pub fn insert_knot(&self, u: f64) -> Result<Self, String> {
        let p = self.degree;
        let knots = self.knots.as_slice();
        // n / n_knots are retained for clarity even though the current
        // implementation returns a fresh basis rather than updating
        // control-point rows in place.
        let _n = self.n_basis();
        let _n_knots = knots.len();

        // Find insertion span
        let k = self.find_span(u);
        let mult = knots.iter().filter(|&&xi| (xi - u).abs() < 1e-12).count();
        if mult > p {
            return Err("knot multiplicity already at maximum".to_string());
        }

        // New knot vector
        let mut new_knots = knots.to_vec();
        new_knots.insert(k + 1, u);
        let new_kv = KnotVector { knots: new_knots };

        // For the basis functions, we just return a new BsplineBasis with the
        // refined knot vector. Control point refinement is handled by the caller
        // (the NURBS patch).
        BsplineBasis::new(p, new_kv)
    }

    /// Refine by inserting multiple knots.
    pub fn refine_by_knots(&self, new_knots: &[f64]) -> Result<Self, String> {
        let mut basis = self.clone();
        for &u in new_knots {
            basis = basis.insert_knot(u)?;
        }
        Ok(basis)
    }
}

#[derive(Clone, Debug)]
pub struct NurbsBasis {
    bspline: BsplineBasis,
    weights: Vec<f64>,
}

impl NurbsBasis {
    pub fn new(bspline: BsplineBasis, weights: Vec<f64>) -> Result<Self, String> {
        if weights.len() != bspline.n_basis() {
            return Err("weights length must match number of basis functions".to_string());
        }
        if weights.iter().any(|w| !w.is_finite() || *w <= 0.0) {
            return Err("weights must be finite and strictly positive".to_string());
        }
        Ok(Self { bspline, weights })
    }

    pub fn nonzero_values(&self, u: f64) -> Result<Vec<(usize, f64)>, String> {
        let bvals = self.bspline.nonzero_values(u)?;
        let mut w_sum = 0.0;
        for (i, n) in &bvals {
            w_sum += self.weights[*i] * *n;
        }
        if w_sum.abs() <= f64::EPSILON {
            return Err("NURBS denominator is zero".to_string());
        }

        Ok(bvals
            .into_iter()
            .map(|(i, n)| (i, self.weights[i] * n / w_sum))
            .collect())
    }

    pub fn nonzero_derivatives(&self, u: f64) -> Result<Vec<(usize, f64)>, String> {
        let bvals = self.bspline.nonzero_values(u)?;
        let bders = self.bspline.nonzero_derivatives(u)?;

        let mut w = 0.0;
        let mut w_der = 0.0;
        for ((i, n), (_, dn)) in bvals.iter().zip(bders.iter()) {
            let wi = self.weights[*i];
            w += wi * *n;
            w_der += wi * *dn;
        }
        if w.abs() <= f64::EPSILON {
            return Err("NURBS denominator is zero".to_string());
        }

        Ok(bvals
            .into_iter()
            .zip(bders)
            .map(|((i, n), (_, dn))| {
                let wi = self.weights[i];
                let num = wi * dn * w - wi * n * w_der;
                (i, num / (w * w))
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::{BsplineBasis, KnotVector, NurbsBasis};

    fn basis_value(values: &[(usize, f64)], idx: usize) -> f64 {
        values
            .iter()
            .find_map(|(i, v)| (*i == idx).then_some(*v))
            .unwrap_or(0.0)
    }

    #[test]
    fn partition_of_unity_quadratic() {
        let kv = KnotVector::new_clamped(vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]).unwrap();
        let basis = BsplineBasis::new(2, kv).unwrap();
        let points = [0.0, 0.1, 0.25, 0.5, 0.8, 1.0];

        for &u in &points {
            let sum: f64 = basis
                .nonzero_values(u)
                .unwrap()
                .iter()
                .map(|(_, v)| *v)
                .sum();
            assert!((sum - 1.0).abs() < 1e-12, "u={u}, sum={sum}");
        }
    }

    #[test]
    fn local_support_active_count_interior() {
        let kv = KnotVector::new_clamped(vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]).unwrap();
        let basis = BsplineBasis::new(2, kv).unwrap();
        let span = basis.find_span(0.25);
        let active = basis.active_basis_indices(span);
        assert_eq!(active.len(), basis.degree() + 1);
        assert_eq!(active, vec![0, 1, 2]);
    }

    #[test]
    fn derivative_matches_finite_difference() {
        let kv = KnotVector::new_clamped(vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]).unwrap();
        let basis = BsplineBasis::new(2, kv).unwrap();
        let u = 0.37;
        let h = 1e-6;
        let idx = 1;

        let analytic = basis_value(&basis.nonzero_derivatives(u).unwrap(), idx);
        let vp = basis_value(&basis.nonzero_values(u + h).unwrap(), idx);
        let vm = basis_value(&basis.nonzero_values(u - h).unwrap(), idx);
        let fd = (vp - vm) / (2.0 * h);
        assert!((analytic - fd).abs() < 1e-5, "analytic={analytic}, fd={fd}");
    }

    #[test]
    fn endpoint_behavior_uses_last_span() {
        let kv = KnotVector::new_clamped(vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]).unwrap();
        let basis = BsplineBasis::new(2, kv).unwrap();
        let vals = basis.nonzero_values(1.0).unwrap();
        assert_eq!(vals.len(), 3);
        assert!((basis_value(&vals, basis.n_basis() - 1) - 1.0).abs() < 1e-12);
        let sum: f64 = vals.iter().map(|(_, v)| *v).sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn invalid_input_errors_are_reported() {
        assert!(KnotVector::new_clamped(vec![0.0, 0.3, 0.2, 1.0]).is_err());

        let kv = KnotVector::new_clamped(vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]).unwrap();
        let basis = BsplineBasis::new(2, kv).unwrap();
        let err = basis.nonzero_values(1.2).unwrap_err();
        assert!(err.contains("u=1.2"));
        assert!(err.contains("valid=[0"));
    }

    #[test]
    fn clamped_multiplicity_is_epsilon_tolerant() {
        let kv =
            KnotVector::new_clamped(vec![0.0, 5e-13, 9e-13, 0.5, 1.0 - 9e-13, 1.0 - 5e-13, 1.0])
                .unwrap();
        assert!(BsplineBasis::new(2, kv).is_ok());
    }

    #[test]
    fn unit_weights_nurbs_matches_bspline() {
        let kv = KnotVector::new_clamped(vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]).unwrap();
        let bspline = BsplineBasis::new(2, kv).unwrap();
        let nurbs = NurbsBasis::new(bspline.clone(), vec![1.0; bspline.n_basis()]).unwrap();

        for u in [0.0, 0.2, 0.5, 0.9, 1.0] {
            let b_vals = bspline.nonzero_values(u).unwrap();
            let n_vals = nurbs.nonzero_values(u).unwrap();
            for ((bi, bv), (ni, nv)) in b_vals.iter().zip(n_vals.iter()) {
                assert_eq!(*bi, *ni);
                assert!((*bv - *nv).abs() < 1e-12);
            }

            let b_der = bspline.nonzero_derivatives(u).unwrap();
            let n_der = nurbs.nonzero_derivatives(u).unwrap();
            for ((bi, bd), (ni, nd)) in b_der.iter().zip(n_der.iter()) {
                assert_eq!(*bi, *ni);
                assert!((*bd - *nd).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn knot_insertion_increases_basis_count() {
        let kv = KnotVector::new_clamped(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let basis = BsplineBasis::new(2, kv).unwrap();
        let n0 = basis.n_basis();
        let refined = basis.insert_knot(0.5).unwrap();
        let n1 = refined.n_basis();
        assert_eq!(
            n1,
            n0 + 1,
            "inserting one knot should add one basis function"
        );
        // Partition of unity must still hold
        for u in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let sum: f64 = refined
                .nonzero_values(u)
                .unwrap()
                .iter()
                .map(|(_, v)| *v)
                .sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "partition of unity fails at u={u}: sum={sum}"
            );
        }
    }

    #[test]
    fn multiple_knot_insertion_preserves_shape() {
        let kv = KnotVector::new_clamped(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let basis = BsplineBasis::new(2, kv).unwrap();
        // Insert several knots
        let refined = basis.refine_by_knots(&[0.2, 0.4, 0.6, 0.8]).unwrap();
        assert_eq!(refined.n_basis(), basis.n_basis() + 4);
        // Check partition of unity
        for u in [0.0, 0.1, 0.5, 0.9, 1.0] {
            let sum: f64 = refined
                .nonzero_values(u)
                .unwrap()
                .iter()
                .map(|(_, v)| *v)
                .sum();
            assert!((sum - 1.0).abs() < 1e-12);
        }
    }
}

// ─── Backward-compat re-exports (from the `nurbs` module) ────────────────
//
// These types originally lived in `fem_element::nurbs` and are re-exported
// here so that consumers can migrate by changing `fem_element::nurbs::X` to
// `fem_element::iga::X`.  The old `nurbs` module is kept as legacy; new code
// should prefer `fem_element::iga` types where possible.
//
// The old `KnotVector` (which stored degree) is NOT re-exported as `KnotVector`
// because the new `iga::KnotVector` (pure knot sequence) supersedes it.
// Code that needs the degree-aware knot vector uses `NurbsKnotVector` (the
// re-exported old type) temporarily during migration, or constructs a
// `BsplineBasis::new(deg, kv)` from the new `KnotVector`.

pub use crate::nurbs::{
    greville_abscissae, BSplineBasis1D as NurbsBSplineBasis1D, KnotVector as NurbsKnotVector,
    NurbsMesh2D, NurbsMesh3D, NurbsPatch2D, NurbsPatch2DData, NurbsPatch3D, NurbsPatch3DData,
};
