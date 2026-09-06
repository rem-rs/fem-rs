//! NURBS mesh working layer — port of MFEM `mesh/nurbs.hpp/cpp` `KnotVector`.
//!
//! Methods faithfully reproduce the C++ implementations:
//! - `KnotVector::Print` → `print()`
//! - `KnotVector::GetGreville` → `greville_abscissae()`
//! - `KnotVector::DegreeElevate` → `degree_elevate()`
//! - `KnotVector::GetBotella` → `botella_abscissae()`
//! - `KnotVector::GetDemko` → `demko_abscissae()`

use std::fmt;

/// KnotVector — faithful port of MFEM `KnotVector` (mesh/nurbs.hpp).
#[derive(Debug, Clone)]
pub struct NurbsKnotVector {
    /// Order of the B-spline basis functions (degree + 1).
    order: i32,
    /// Number of control points.
    num_cp: i32,
    /// Knot values (length = num_cp + order + 1).
    knots: Vec<f64>,
}

const MAX_ORDER: usize = 10;

impl NurbsKnotVector {
    /// Create a KnotVector from order, number of control points, and knot values.
    pub fn new(order: i32, num_cp: i32, knots: Vec<f64>) -> Self {
        Self { order, num_cp, knots }
    }

    /// Return the order.
    pub fn order(&self) -> i32 { self.order }

    /// Return the number of control points.
    pub fn num_cp(&self) -> i32 { self.num_cp }

    /// Return the number of knots (including multiplicities).
    pub fn len(&self) -> usize { self.knots.len() }

    /// Return a reference to the knot values.
    pub fn values(&self) -> &[f64] { &self.knots }

    /// MFEM `KnotVector::Print`:
    /// ```text
    /// <order> <num_cp> <knot[0]> <knot[1]> ... <knot[n]>
    /// ```
    pub fn print(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("{} {} ", self.order, self.num_cp));
        for (i, k) in self.knots.iter().enumerate() {
            if i > 0 { s.push(' '); }
            s.push_str(&format!("{:.15e}", k));
        }
        s.push('\n');
        s
    }

    /// MFEM `KnotVector::GetGreville(i)`:
    /// `greville(i) = (knot[i+1] + ... + knot[i+Order]) / Order`
    pub fn greville(&self, i: usize) -> f64 {
        let mut sum = 0.0;
        for j in 1..(self.order + 1) as usize {
            sum += self.knots[i + j];
        }
        sum / self.order as f64
    }

    /// All Greville abscissae.
    pub fn greville_abscissae(&self) -> Vec<f64> {
        let ncp = self.num_cp as usize;
        (0..ncp).map(|i| self.greville(i)).collect()
    }

    /// MFEM `KnotVector::DegreeElevate(t)`:
    /// Creates a new KnotVector with elevated order by repeating endpoints.
    pub fn degree_elevate(&self, t: i32) -> Self {
        assert!(t >= 0, "degree elevate factor must be non-negative");
        let new_order = self.order + t;
        let new_ncp = self.num_cp + t;
        let new_size = (new_ncp + new_order + 1) as usize;
        let mut new_knots = vec![0.0; new_size];

        // First new_order+1 knots = first knot
        for i in 0..=new_order as usize {
            new_knots[i] = self.knots[0];
        }
        // Middle knots: shifted by t
        for i in (new_order as usize + 1)..new_ncp as usize {
            new_knots[i] = self.knots[i - t as usize];
        }
        // Last new_order+1 knots = last knot
        let last = self.knots[self.knots.len() - 1];
        for i in 0..=new_order as usize {
            new_knots[new_ncp as usize + i] = last;
        }

        Self {
            order: new_order,
            num_cp: new_ncp,
            knots: new_knots,
        }
    }

    // -------------------------------------------------------------------------
    // Helper methods — B-spline basis evaluation (from "The NURBS Book")
    // -------------------------------------------------------------------------

    /// Return the index of the knot span containing parameter u.
    /// Faithful port of MFEM `KnotVector::GetSpan`.
    /// Returns `mid` such that `knot[mid] <= u < knot[mid+1]` (or mid = ncp-1 at the right end).
    fn get_span(&self, u: f64) -> usize {
        let order = self.order as usize;
        let ncp = self.num_cp as usize;

        if u >= self.knots[ncp] {
            // At the right end: return ncp - 1
            ncp - 1
        } else if u <= self.knots[0] {
            // At the left end: return Order
            order
        } else {
            // If u < knot[order], the correct span is order
            // (this handles the case where the first Greville point is below knot[order])
            if u < self.knots[order] {
                return order;
            }
            // Binary search in [order, ncp)
            let mut low = order;
            let mut high = ncp;
            let mut mid = (low + high) / 2;
            while u < self.knots[mid] || u >= self.knots[mid + 1] {
                if u < self.knots[mid] {
                    high = mid;
                } else {
                    low = mid;
                }
                let new_mid = (low + high) / 2;
                if new_mid == mid {
                    break; // Prevent infinite loop
                }
                mid = new_mid;
            }
            mid
        }
    }

    /// Return the reference coordinate in [0,1] for parameter u
    /// in the element beginning at knot `ni`.
    fn get_ref_point(&self, u: f64, ni: usize) -> f64 {
        (u - self.knots[ni]) / (self.knots[ni + 1] - self.knots[ni])
    }

    /// Calculate the nonvanishing shape function values for the element
    /// corresponding to knot index `i` and element reference coordinate `xi`.
    /// Algorithm A2.2 from "The NURBS Book" (Piegl & Tiller, 2nd ed., p.70).
    fn calc_shape(&self, shape: &mut [f64], i: i32, xi: f64) {
        let p = self.order as usize;
        let ip = (i + p as i32) as usize;
        let u = self.knots[ip] + xi * (self.knots[ip + 1] - self.knots[ip]);

        let mut left = [0.0f64; MAX_ORDER + 1];
        let mut right = [0.0f64; MAX_ORDER + 1];

        shape[0] = 1.0;
        for j in 1..=p {
            left[j] = u - self.knots[ip + 1 - j];
            right[j] = self.knots[ip + j] - u;
            let mut saved = 0.0;
            for r in 0..j {
                let tmp = shape[r] / (right[r + 1] + left[j - r]);
                shape[r] = saved + right[r + 1] * tmp;
                saved = left[j - r] * tmp;
            }
            shape[j] = saved;
        }
    }

    /// Calculate first derivatives of the nonvanishing shape functions.
    /// Algorithm A2.3 from "The NURBS Book" (Piegl & Tiller, 2nd ed., p.72).
    fn calc_dshape(&self, grad: &mut [f64], i: i32, xi: f64) {
        let p = self.order as usize;
        // Match C++: ip = (i >= 0) ? (i + p) : (-1 - i + p)
        let ip = if i >= 0 { (i as usize) + p } else { ((-1 - i) as usize) + p };
        // Match C++: u = GetKnotLocation((i >= 0) ? xi : 1. - xi, ip)
        let xi = if i >= 0 { xi } else { 1.0 - xi };
        let u = self.knots[ip] + xi * (self.knots[ip + 1] - self.knots[ip]);

        let mut ndu = [[0.0f64; MAX_ORDER + 1]; MAX_ORDER + 1];
        let mut left = [0.0f64; MAX_ORDER + 1];
        let mut right = [0.0f64; MAX_ORDER + 1];

        ndu[0][0] = 1.0;
        for j in 1..=p {
            left[j] = u - self.knots[ip - j + 1];
            right[j] = self.knots[ip + j] - u;
            let mut saved = 0.0;
            for r in 0..j {
                ndu[j][r] = right[r + 1] + left[j - r];
                let temp = ndu[r][j - 1] / ndu[j][r];
                ndu[r][j] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            ndu[j][j] = saved;
        }

        for r in 0..=p {
            let mut d = 0.00;
            let rk = r as i32 - 1;
            let pk = p as i32 - 1;
            if r >= 1 {
                d = ndu[rk as usize][pk as usize] / ndu[p][rk as usize];
            }
            if r <= pk as usize {
                d -= ndu[r][pk as usize] / ndu[p][r];
            }
            grad[r] = d;
        }

        // Scale by derivative chain rule
        let scale = p as f64 * (self.knots[ip + 1] - self.knots[ip]);
        for r in 0..=p {
            grad[r] *= scale;
        }
    }

    /// Calculate second derivatives of the nonvanishing shape functions.
    /// Uses central finite differences on the first derivative.
    fn calc_d2shape(&self, grad2: &mut [f64], i: i32, xi: f64) {
        let p = self.order as usize;
        let h = 1e-6;

        // Compute d2N/dxi2 using central finite differences of dN/dxi
        let mut gradp = vec![0.0f64; MAX_ORDER + 1];
        let mut gradm = vec![0.0f64; MAX_ORDER + 1];

        // Clamp xi to [h, 1-h] to avoid boundary issues
        let xi = xi.max(h).min(1.0 - h);

        // Forward point
        self.calc_dshape(&mut gradp, i, xi + h);
        // Backward point
        self.calc_dshape(&mut gradm, i, xi - h);

        let dx = 2.0 * h;
        for r in 0..=p {
            grad2[r] = (gradp[r] - gradm[r]) / dx;
        }
    }

    // -------------------------------------------------------------------------
    // Botella — Newton iteration for shape function maxima
    // -------------------------------------------------------------------------

    /// MFEM `KnotVector::GetBotella(i)`:
    /// Returns the knot location where the i-th shape function is maximum.
    /// Uses Newton iteration with the Greville point as starting value.
    pub fn botella(&self, i: usize) -> f64 {
        const ITEMAX: usize = 10;
        const TOL: f64 = 1e-8;

        let order = self.order as usize;
        let mut grad = vec![0.0; order + 1];
        let mut hess = vec![0.0; order + 1];

        // Initial guess: Greville point
        let mut u = self.greville(i);

        // Check for a repeated knot — revert to Greville point
        if self.knots[i + 1] == self.knots[i + self.order as usize] {
            return u;
        }

        for _iter in 0..ITEMAX {
            let ks = self.get_span(u);
            let xi = self.get_ref_point(u, ks);
            // The shape function index within the span: o = order - (ks - i)
            let o = order - (ks - i);

            self.calc_dshape(&mut grad, (ks - order) as i32, xi);
            self.calc_d2shape(&mut hess, (ks - order) as i32, xi);

            // Newton step: u -= (grad[o]/hess[o]) * (knot[ks+1] - knot[ks])
            let dk = self.knots[ks + 1] - self.knots[ks];
            let h = hess[o];
            if h.abs() > 1e-6 {
                u -= (grad[o] / h) * dk;
            } else {
                // Second derivative is close to zero (linear basis function)
                // The maximum is at the Greville point, stop iterating
                break;
            }

            // Clamp u to valid range
            u = u.max(self.knots[order]).min(self.knots[self.num_cp as usize]);

            if grad[o].abs() < TOL {
                break;
            }
        }

        u
    }

    /// All Botella abscissae.
    pub fn botella_abscissae(&self) -> Vec<f64> {
        let ncp = self.num_cp as usize;
        (0..ncp).map(|i| self.botella(i)).collect()
    }

    // -------------------------------------------------------------------------
    // GetInterpolant — global curve interpolation
    // -------------------------------------------------------------------------

    /// MFEM `KnotVector::GetInterpolant`:
    /// Global curve interpolation through the points `x` at the knot locations `u`.
    /// The control points of the interpolated curve are returned in `a`.
    /// Uses a collocation matrix approach (Algorithm A9.1 from "The NURBS Book").
    pub fn get_interpolant(&self, u: &[f64], x: &[f64], a: &mut [f64]) {
        let ncp = self.num_cp as usize;
        let order = self.order as usize;

        // Build the collocation matrix A where A[i][j] = N_{j,order}(u[i])
        let mut matrix = vec![vec![0.0f64; ncp]; ncp];
        let mut shape = vec![0.0; order + 1];

        for i in 0..ncp {
            let ks = self.get_span(u[i]);
            let xi = self.get_ref_point(u[i], ks);
            let j_start = ks - order; // shape function index
            self.calc_shape(&mut shape, j_start as i32, xi);
            for p in 0..=order {
                let col = j_start + p;
                if col < ncp {
                    matrix[i][col] = shape[p];
                }
            }
        }

        // Solve A * a = x via Gaussian elimination with partial pivoting
        let n = ncp;
        let mut aug = vec![vec![0.0f64; n + 1]; n];
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = matrix[i][j];
            }
            aug[i][n] = x[i];
        }

        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_val = aug[col][col].abs();
            let mut max_row = col;
            for row in (col + 1)..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }
            if max_val < 1e-15 {
                continue; // Singular or near-singular
            }
            if max_row != col {
                aug.swap(col, max_row);
            }

            // Eliminate below
            for row in (col + 1)..n {
                let factor = aug[row][col] / aug[col][col];
                for j in col..=n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        // Back substitution
        for i in (0..n).rev() {
            let mut sum = aug[i][n];
            for j in (i + 1)..n {
                sum -= aug[i][j] * a[j];
            }
            a[i] = sum / aug[i][i];
        }
    }

    // -------------------------------------------------------------------------
    // Demko — Remez iteration for Chebyshev spline extrema
    // -------------------------------------------------------------------------

    /// MFEM `KnotVector::ComputeDemko()`:
    /// Computes the Demko abscissae (Chebyshev spline extrema points) via
    /// Remez iteration.
    pub fn demko_abscissae(&self) -> Vec<f64> {
        const ITEMAX1: usize = 50;
        const ITEMAX2: usize = 50;
        const TOL1: f64 = 1e-10;
        const TOL2: f64 = 1e-8;

        let ncp = self.num_cp as usize;
        let order = self.order as usize;

        // Initial alternating values for interpolation
        let x: Vec<f64> = (0..ncp).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();

        // Initialize demko points with Greville abscissae
        let mut demko: Vec<f64> = (0..ncp).map(|i| self.greville(i)).collect();

        let mut a = vec![0.0; ncp];
        let mut anew = vec![0.0; ncp];
        let mut sh = vec![0.0; order + 1];
        let mut shgrad = vec![0.0; order + 1];
        let mut shhess = vec![0.0; order + 1];

        // Get initial interpolant
        self.get_interpolant(&demko, &x, &mut anew);

        for _iter1 in 0..ITEMAX1 {
            // a = anew
            a.copy_from_slice(&anew);

            for i in 0..ncp {
                // Check for a repeated knot
                if self.knots[i + 1] == self.knots[i + order] {
                    continue;
                }

                let mut u = demko[i];

                // Find location of extremum
                for _iter2 in 0..ITEMAX2 {
                    let ks = self.get_span(u);
                    let xi = self.get_ref_point(u, ks);

                    self.calc_shape(&mut sh, (ks - order) as i32, xi);
                    self.calc_dshape(&mut shgrad, (ks - order) as i32, xi);
                    self.calc_d2shape(&mut shhess, (ks - order) as i32, xi);

                    let mut val = 0.0;
                    let mut grad = 0.0;
                    let mut hess = 0.0;
                    for p in 0..=order {
                        let idx = ks - order + p;
                        if idx < ncp {
                            val += a[idx] * sh[p];
                            grad += a[idx] * shgrad[p];
                            hess += a[idx] * shhess[p];
                        }
                    }

                    if grad.abs() < TOL2 {
                        break;
                    }

                    let dk = self.knots[ks + 1] - self.knots[ks];
                    if hess.abs() < 3.0f64.powi(order as i32) {
                        u += 0.25 * 0.45f64.powi(order as i32)
                            * val.signum() * grad * dk;
                    } else {
                        u -= (grad / hess) * dk;
                    }
                }

                // Update
                demko[i] = u;
            }

            // Correct order of demko vector (bubble sort pass)
            for i in 0..(ncp - 1) {
                if demko[i] > demko[i + 1] {
                    demko.swap(i, i + 1);
                }
            }

            // Find new interpolant and compare with old
            self.get_interpolant(&demko, &x, &mut anew);

            // Check convergence: |a - anew|_2 < tol1
            let diff_norm: f64 = a.iter().zip(anew.iter())
                .map(|(ai, bi)| (ai - bi).powi(2))
                .sum::<f64>()
                .sqrt();

            if diff_norm < TOL1 {
                break;
            }
        }

        demko
    }
}

impl fmt::Display for NurbsKnotVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knot_vector_print() {
        let kv = NurbsKnotVector::new(2, 4, vec![0.0, 0.0, 0.5, 1.0, 1.0]);
        let s = kv.print();
        assert!(s.starts_with("2 4 "));
        assert!(s.contains("0.000000000000000e0"), "got: {}", s);
        assert!(s.contains("5.000000000000000e-1"));
        assert!(s.ends_with("\n"));
    }

    #[test]
    fn greville_abscissae() {
        let kv = NurbsKnotVector::new(2, 3, vec![0.0, 0.0, 0.5, 1.0, 1.0]);
        let g = kv.greville_abscissae();
        assert_eq!(g.len(), 3);
        assert!((g[0] - 0.25).abs() < 1e-14, "g[0] = {}", g[0]);
        assert!((g[1] - 0.75).abs() < 1e-14, "g[1] = {}", g[1]);
        assert!((g[2] - 1.0).abs() < 1e-14, "g[2] = {}", g[2]);
    }

    #[test]
    fn degree_elevate() {
        let kv = NurbsKnotVector::new(2, 2, vec![0.0, 0.0, 1.0, 1.0]);
        let elevated = kv.degree_elevate(1);
        assert_eq!(elevated.order(), 3);
        assert_eq!(elevated.num_cp(), 3);
        assert_eq!(elevated.len(), 7);
        // First new_order+1=4 knots = first knot (0), last 4 knots = last knot (1)
        // Middle loop is empty (new_order+1=4 > new_ncp=3)
        assert_eq!(elevated.knots, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn botella_abscissae() {
        // Fully clamped quadratic B-spline: knots [0,0,0, 0.5, 1,1,1]
        let kv = NurbsKnotVector::new(2, 4, vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]);
        let bot = kv.botella_abscissae();
        let grev = kv.greville_abscissae();
        assert_eq!(bot.len(), 4);
        // Botella points should be monotonically increasing
        for i in 1..bot.len() {
            assert!(bot[i] >= bot[i - 1], "bot[{}]={} < bot[{}]={}", i, bot[i], i - 1, bot[i - 1]);
        }
        // For non-uniform knots, Botella can differ from Greville by up to the knot spacing
        for i in 0..bot.len() {
            assert!((bot[i] - grev[i]).abs() < 0.3,
                "bot[{}]={} vs grev[{}]={}", i, bot[i], i, grev[i]);
        }
    }

    #[test]
    fn botella_repeated_knot() {
        // When knot[i+1] == knot[i+order], Botella should return Greville
        let kv = NurbsKnotVector::new(2, 4, vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]);
        // For i=0, knot[1]=0 == knot[2]=0 (repeated), so Botella should return Greville
        let bot_first = kv.botella(0);
        let grev_first = kv.greville(0);
        assert!((bot_first - grev_first).abs() < 1e-14,
            "bot_first={} should equal grev_first={}", bot_first, grev_first);
    }

    #[test]
    fn demko_abscissae() {
        // Fully clamped quadratic B-spline
        let kv = NurbsKnotVector::new(2, 4, vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]);
        let demko = kv.demko_abscissae();
        assert_eq!(demko.len(), 4);
        // Demko points should be monotonically increasing
        for i in 1..demko.len() {
            assert!(demko[i] >= demko[i - 1],
                "demko[{}]={} < demko[{}]={}", i, demko[i], i - 1, demko[i - 1]);
        }
        // Demko points should be within [0, 1]
        for &d in &demko {
            assert!(d >= -1e-10 && d <= 1.0 + 1e-10, "demko out of range: {}", d);
        }
    }

    #[test]
    fn demko_uniform_cubic() {
        // Fully clamped cubic B-spline: knots [0,0,0,0, 0.5, 1,1,1,1]
        let kv = NurbsKnotVector::new(3, 5, vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]);
        let demko = kv.demko_abscissae();
        assert_eq!(demko.len(), 5);
        for i in 1..demko.len() {
            assert!(demko[i] >= demko[i - 1],
                "demko[{}]={} < demko[{}]={}", i, demko[i], i - 1, demko[i - 1]);
        }
    }

    #[test]
    fn get_interpolant_identity() {
        // Fully clamped quadratic B-spline
        let kv = NurbsKnotVector::new(2, 4, vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]);
        let grev = kv.greville_abscissae();
        let x: Vec<f64> = grev.clone();
        let mut a = vec![0.0; 4];
        kv.get_interpolant(&grev, &x, &mut a);
        // The control points should be close to the identity for uniform knots
        for i in 0..4 {
            assert!((a[i] - x[i]).abs() < 0.1,
                "a[{}]={} vs x[{}]={}", i, a[i], i, x[i]);
        }
    }
}
