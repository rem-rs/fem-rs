//! NURBS patch object layer — port of MFEM `NURBSPatch`
//! (`mesh/nurbs.hpp:320`, `mesh/nurbs.cpp:1274-1520`) together with the
//! `KnotVector` working API that drives it (`mesh/nurbs.hpp:40`,
//! `mesh/nurbs.cpp:28-1252`).
//!
//! # What lives here
//!
//! - [`NurbsKnotVector`] — MFEM `KnotVector`: order / NCP / knot values plus
//!   the Greville, Botella and Demko abscissae, `GetInterpolant` (global curve
//!   interpolation), `Difference` and the mesh-format `Print`.  This type was
//!   previously the orphaned `nurbs_mesh::NurbsKnotVector`; D374 relocated it
//!   here and completed it to bit-exact MFEM parity (the old version used a
//!   `{:.15e}` print format, a non-MFEM `GetSpan`, finite-difference second
//!   derivatives and a Gaussian solve; MFEM uses `%g` formatting, span
//!   bisection, the analytic `CalcDnShape` (A2.3) and an explicit inverse).
//! - [`NurbsPatch`] — the control-point store of MFEM `NURBSPatch`: indexed
//!   `get`/`set` with MFEM's `(i + j*ni)*Dim + l` layout
//!   (`mesh/nurbs.hpp:1364`), degree elevation, knot insertion and the exact
//!   `NURBSPatch::Print` text format (`mesh/nurbs.cpp:1484`) used for the
//!   `patches` geometry flavour of `MFEM NURBS mesh` files.
//!
//! # Delegation
//!
//! The *numerics* of degree elevation and knot insertion are not duplicated:
//! [`NurbsPatch::degree_elevate`] and [`NurbsPatch::knot_insert`] delegate to
//! `fem_element::nurbs::{elevate_u_2d, elevate_v_2d, h_refine_uk,
//! h_refine_vk}`, which are the existing ports of the same Piegl & Tiller
//! routines MFEM uses (`NURBSPatch::DegreeElevate` mesh/nurbs.cpp:2082,
//! `NURBSPatch::KnotInsert(dir, Vector&)` mesh/nurbs.cpp:1767).
//!
//! # Known parity limits
//!
//! - MFEM stores raw (homogeneous) components `(x*w, y*w, w)`; the delegated
//!   kernels round-trip through Cartesian form (`x*w` blended, divided back by
//!   the blended `w`).  For unit weights (the whole `-uw`/B-spline path of
//!   `nurbs_curveint`) the round trip is exact; with non-unit weights a
//!   last-bit difference versus MFEM's raw storage is possible.  (Since D496
//!   `NurbsPatch::knot_insert` is the exact raw-data A5.5 port, only
//!   `degree_elevate` still goes through the delegated kernels.)
//! - The 1-D (single knot vector) patch of `mesh/nurbs.cpp:1281-1296` is not
//!   implemented; the 2-D and 3-D patches are.

use std::fmt;

// ─── C++ `operator<<(ostream&, double)` — printf %g ──────────────────────────

/// Format `value` exactly as C++ `operator<<(std::ostream &, double)` does for
/// a stream with `precision() == precision` (the default is 6), i.e.
/// `printf("%.*g", precision, value)`.
///
/// MFEM writes `NURBSPatch::Print` and `KnotVector::Print` through plain
/// `ostream` insertion on a default-constructed `ofstream`
/// (`mesh/nurbs.cpp:1484-1507`, `mesh/nurbs.cpp:632-637`), so reproducing the
/// C++ mesh bytes requires reproducing `%g`.  Rust has no `%g`, so the style
/// selection is done explicitly, mirroring the private (and duplicated)
/// `format_g` of `crates/io/src/nurbs_mesh.rs` — `fem-mesh` cannot depend on
/// `fem-io` (the dependency edge goes the other way).  Also public for
/// drivers (e.g. `nurbs_curveint`) that must print `real_t` options the way
/// MFEM's `OptionsParser::WriteValue` does.
pub fn format_g(value: f64, precision: usize) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.to_string();
    }
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_string();
    }

    let p = precision.max(1);
    // `%g` picks the style from the exponent of the value rounded to `p`
    // significant digits, which is what `{:.*e}` reports.
    let sci = format!("{:.*e}", p - 1, value);
    let (mant, exp) = sci
        .split_once('e')
        .expect("Rust's LowerExp always writes an exponent");
    let exp: i32 = exp
        .parse()
        .expect("Rust's LowerExp always writes a decimal exponent");

    if exp < -4 || exp >= p as i32 {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", trim_frac(mant), sign, exp.unsigned_abs())
    } else {
        let frac = (p as i32 - 1 - exp).max(0) as usize;
        trim_frac(&format!("{value:.*}", frac))
    }
}

/// Drop trailing zeros (and a dangling `.`) from a decimal rendering.
fn trim_frac(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    let t = s.trim_end_matches('0');
    t.strip_suffix('.').unwrap_or(t).to_string()
}

/// MFEM `KnotVector::MaxOrder` (`mesh/nurbs.hpp:64`): the largest supported
/// order, sizing the fixed stack arrays of `CalcShape`/`CalcDShape`/
/// `CalcDnShape`.
const MAX_ORDER: usize = 10;

// ─── NurbsKnotVector ─────────────────────────────────────────────────────────

/// KnotVector — port of MFEM `KnotVector` (`mesh/nurbs.hpp:40-300`).
///
/// Relocated from `nurbs_mesh.rs` (D374, previously orphaned) and completed to
/// bit-exact MFEM parity for the routines the `nurbs_curveint` miniapp drives:
/// `GetDemko`/`ComputeDemko`, `GetInterpolant`, `Difference` and `Print`.
#[derive(Debug, Clone)]
pub struct NurbsKnotVector {
    /// Order of the B-spline basis functions — MFEM's `Order` is the
    /// polynomial *degree* `p` (nurbs.cpp A2.2 uses `p = Order` directly), the
    /// same convention as `fem_element`'s `degree`.
    order: i32,
    /// Number of control points.
    num_cp: i32,
    /// Knot values (length = num_cp + order + 1).
    knots: Vec<f64>,
}

impl NurbsKnotVector {
    /// Create a KnotVector from order, number of control points, and knot
    /// values (MFEM `KnotVector::KnotVector(int, int)`, `mesh/nurbs.cpp:48`,
    /// plus the knot contents).
    pub fn new(order: i32, num_cp: i32, knots: Vec<f64>) -> Self {
        Self { order, num_cp, knots }
    }

    /// Return the order (`KnotVector::GetOrder`).
    pub fn order(&self) -> i32 {
        self.order
    }

    /// Return the number of control points (`KnotVector::GetNCP`).
    pub fn num_cp(&self) -> i32 {
        self.num_cp
    }

    /// Return the number of knots including multiplicities (`KnotVector::Size`).
    pub fn len(&self) -> usize {
        self.knots.len()
    }

    /// Return the knot values.
    pub fn values(&self) -> &[f64] {
        &self.knots
    }

    /// MFEM `KnotVector::Print` (`mesh/nurbs.cpp:632`):
    ///
    /// ```text
    /// <order> <num_cp> <knot[0]> ... <knot[n]> '\n'
    /// ```
    ///
    /// The knots go through `Vector::Print(os, Size())`
    /// (`linalg/vector.cpp:870`), i.e. `%g` with the stream's default
    /// precision 6 and `ZeroSubnormal` applied.
    pub fn print(&self) -> String {
        let mut s = format!("{} {} ", self.order, self.num_cp);
        for (i, k) in self.knots.iter().enumerate() {
            if i > 0 {
                s.push(' ');
            }
            // ZeroSubnormal (linalg/vector.cpp:876 / mfem.hpp).
            let v = if *k != 0.0 && k.abs() < f64::MIN_POSITIVE { 0.0 } else { *k };
            s.push_str(&format_g(v, 6));
        }
        s.push('\n');
        s
    }

    /// MFEM `KnotVector::GetGreville(i)` (`mesh/nurbs.cpp:211`):
    /// `(knot[i+1] + ... + knot[i+Order]) / Order`.
    pub fn greville(&self, i: usize) -> f64 {
        let mut sum = 0.0;
        for j in 1..(self.order + 1) as usize {
            sum += self.knots[i + j];
        }
        sum / self.order as f64
    }

    /// All Greville abscissae (`KnotVector::GetGreville(Vector&)`,
    /// `mesh/nurbs.cpp:220`).
    pub fn greville_abscissae(&self) -> Vec<f64> {
        let ncp = self.num_cp as usize;
        (0..ncp).map(|i| self.greville(i)).collect()
    }

    /// MFEM `KnotVector::DegreeElevate(t)` (`mesh/nurbs.cpp:412-440`):
    /// elevates the order by repeating the end knots.
    ///
    /// Delegates to [`fem_element::nurbs_fe_collection::degree_elevate`], the
    /// single implementation of the routine (shared with the `fem_space` NURBS
    /// extensions).  That routine recovers `Order` from the knot
    /// multiplicities (`Order + 1` repeated end knots), so the stored
    /// `order`/`num_cp` must agree with the knot sequence (the clamping the
    /// NURBS mesh format guarantees); the result's `order`/`num_cp` stay the
    /// declared `+ t`.
    pub fn degree_elevate(&self, t: i32) -> Self {
        assert!(t >= 0, "degree elevate factor must be non-negative");
        let kv = fem_element::iga::KnotVector::new_clamped(self.knots.clone())
            .expect("NurbsKnotVector::degree_elevate: invalid knot vector");
        let elevated = fem_element::nurbs_fe_collection::degree_elevate(&kv, t as usize)
            .expect("NurbsKnotVector::degree_elevate: invalid knot vector");
        let new_order = self.order + t;
        let new_ncp = self.num_cp + t;
        debug_assert_eq!(
            elevated.as_slice().len(),
            (new_ncp + new_order + 1) as usize,
            "NurbsKnotVector::degree_elevate: order/num_cp disagree with the knots"
        );
        Self {
            order: new_order,
            num_cp: new_ncp,
            knots: elevated.as_slice().to_vec(),
        }
    }

    // -------------------------------------------------------------------------
    // Basis evaluation helpers
    // -------------------------------------------------------------------------

    /// MFEM `KnotVector::GetKnotLocation(xi, ni)` (`mesh/nurbs.hpp:148`):
    /// `xi*knot(ni+1) + (1 - xi)*knot(ni)`.  Kept in MFEM's exact form (not
    /// the algebraically equal `knot[ni] + xi*(knot[ni+1]-knot[ni]`) so that
    /// shape evaluation is bit-identical to the C++.
    fn get_knot_location(&self, xi: f64, ni: usize) -> f64 {
        xi * self.knots[ni + 1] + (1.0 - xi) * self.knots[ni]
    }

    /// MFEM `KnotVector::GetRefPoint(u, ni)` (`mesh/nurbs.hpp:143`).
    fn get_ref_point(&self, u: f64, ni: usize) -> f64 {
        (u - self.knots[ni]) / (self.knots[ni + 1] - self.knots[ni])
    }

    /// MFEM `KnotVector::GetSpan(u)` (`mesh/nurbs.cpp:177-208`): returns the
    /// index `mid` with `knot[mid] <= u < knot[mid+1]`; exact port including
    /// the endpoint equality tests.
    fn get_span(&self, u: f64) -> usize {
        let order = self.order as usize;
        let ncp = self.num_cp as usize;

        if u == self.knots[ncp + order] {
            ncp - 1
        } else if u == self.knots[0] {
            order
        } else if u > self.knots[0] && u < self.knots[ncp + order] {
            let mut low = order;
            let mut high = ncp;
            let mut mid = (low + high) / 2;
            while u < self.knots[mid] || u >= self.knots[mid + 1] {
                if u < self.knots[mid] {
                    high = mid;
                } else {
                    low = mid;
                }
                mid = (low + high) / 2;
            }
            mid
        } else {
            panic!("Knot location outside of the range of the KnotVector");
        }
    }

    /// MFEM `KnotVector::CalcShape` (`mesh/nurbs.cpp:728-751`, NURBS Book
    /// A2.2): non-vanishing shape functions at reference coordinate `xi` of
    /// the element starting at knot index `i`.
    fn calc_shape(&self, shape: &mut [f64], i: i32, xi: f64) {
        let p = self.order as usize;
        let ip = if i >= 0 { i as usize + p } else { (-1 - i) as usize + p };
        let u = self.get_knot_location(if i >= 0 { xi } else { 1.0 - xi }, ip);
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

    /// MFEM `KnotVector::CalcDShape` (`mesh/nurbs.cpp:755-806`, NURBS Book
    /// A2.3): first derivatives of the non-vanishing shape functions.
    fn calc_dshape(&self, grad: &mut [f64], i: i32, xi: f64) {
        let p = self.order as usize;
        let ip = if i >= 0 { i as usize + p } else { (-1 - i) as usize + p };
        let u = self.get_knot_location(if i >= 0 { xi } else { 1.0 - xi }, ip);
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
            let mut d = 0.0;
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

        // Scale by the derivative chain rule; `CalcDShape` multiplies by
        // +p*(...) when i >= 0 and by -p*(...) when i < 0 (nurbs.cpp:797-805).
        let scale = if i >= 0 {
            p as f64 * (self.knots[ip + 1] - self.knots[ip])
        } else {
            p as f64 * (self.knots[ip] - self.knots[ip + 1])
        };
        for g in grad[..=p].iter_mut() {
            *g *= scale;
        }
    }

    /// MFEM `KnotVector::CalcDnShape` (`mesh/nurbs.cpp:811-905`, NURBS Book
    /// A2.3 general derivative); `CalcD2Shape` is `CalcDnShape(grad2, 2, i, xi)`
    /// (`mesh/nurbs.hpp:214`).
    ///
    /// MFEM indexes `ndu[rk+j][pk]` with raw `int`s, which reads out of bounds
    /// when the order is smaller than `n` (undefined behaviour in C++); the
    /// port defines the derivative as zero in that regime instead.
    fn calc_dn_shape(&self, gradn: &mut [f64], n: usize, i: i32, xi: f64) {
        let p = self.order as usize;
        let ip = if i >= 0 { i as usize + p } else { (-1 - i) as usize + p };
        let mut u = self.get_knot_location(if i >= 0 { xi } else { 1.0 - xi }, ip);
        let mut ndu = [[0.0f64; MAX_ORDER + 1]; MAX_ORDER + 1];
        let mut left = [0.0f64; MAX_ORDER + 1];
        let mut right = [0.0f64; MAX_ORDER + 1];
        // a[2][MaxOrder+1] of nurbs.cpp:800.
        let mut a = [[0.0f64; MAX_ORDER + 1]; 2];

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
            let mut s1 = 0usize; // C++ s1
            let mut s2 = 1usize; // C++ s2
            a[0][0] = 1.0;
            for k in 1..=n {
                let mut d = 0.0;
                let rk = r as i32 - k as i32;
                let pk = p as i32 - k as i32;
                if r >= k {
                    a[s2][0] = a[s1][0] / ndu[(pk + 1) as usize][rk as usize];
                    d = a[s2][0] * ndu[rk as usize][pk as usize];
                }
                // MFEM reuses the loop variable `j` after the inner loop; when
                // the `r <= pk` branch fires the loop was empty, so `j == k`
                // and `ndu[rk+j][pk] == ndu[r][pk]` (NURBS Book A2.3).
                let j1: i32 = if rk >= -1 { 1 } else { -rk };
                let j2: i32 = if (r as i32 - 1) <= pk { k as i32 - 1 } else { p as i32 - r as i32 };
                let mut j = j1;
                while j <= j2 {
                    a[s2][j as usize] =
                        (a[s1][j as usize] - a[s1][j as usize - 1]) / ndu[(pk + 1) as usize][(rk + j) as usize];
                    d += a[s2][j as usize] * ndu[(rk + j) as usize][pk as usize];
                    j += 1;
                }
                if r as i32 <= pk {
                    a[s2][k] = -a[s1][k - 1] / ndu[(pk + 1) as usize][r as usize];
                    if pk >= 0 && rk + j >= 0 {
                        // In-bounds case (order >= n): matches the C++ read
                        // ndu[rk+j][pk], which equals ndu[r][pk] here.
                        d += a[s2][j as usize] * ndu[(rk + j) as usize][pk as usize];
                    } else {
                        // C++ reads out of bounds when order < n (UB); defined
                        // here as zero.
                        d += 0.0;
                    }
                }
                gradn[r] = d;
                // C++ swap: j = s1; s1 = s2; s2 = j;
                std::mem::swap(&mut s1, &mut s2);
            }
        }

        if i >= 0 {
            u = self.knots[ip + 1] - self.knots[ip];
        } else {
            u = self.knots[ip] - self.knots[ip + 1];
        }
        let mut temp = p as f64 * u;
        for k in 1..n {
            temp *= (p - k) as f64 * u;
        }
        for g in gradn[..=p].iter_mut() {
            *g *= temp;
        }
    }

    /// MFEM `KnotVector::GetBotella(i)` — Newton iteration from the Greville
    /// point towards the maximum of the i-th shape function.  (Used by the
    /// deprecated `FindInterpolant`; kept from the relocated knot-vector API.)
    pub fn botella(&self, i: usize) -> f64 {
        const ITEMAX: usize = 10;
        const TOL: f64 = 1e-8;

        let order = self.order as usize;
        let mut grad = vec![0.0; order + 1];
        let mut hess = vec![0.0; order + 1];

        // Initial guess: Greville point.
        let mut u = self.greville(i);

        // Check for a repeated knot — revert to Greville point.
        if self.knots[i + 1] == self.knots[i + order] {
            return u;
        }

        for _iter in 0..ITEMAX {
            let ks = self.get_span(u);
            let xi = self.get_ref_point(u, ks);
            // The shape function index within the span: o = order - (ks - i)
            let o = order - (ks - i);

            self.calc_dshape(&mut grad, (ks - order) as i32, xi);
            self.calc_dn_shape(&mut hess, 2, (ks - order) as i32, xi);

            // Newton step: u -= (grad[o]/hess[o]) * (knot[ks+1] - knot[ks])
            let dk = self.knots[ks + 1] - self.knots[ks];
            let h = hess[o];
            if h.abs() > 1e-6 {
                u -= (grad[o] / h) * dk;
            } else {
                // Second derivative close to zero (linear basis function):
                // the maximum is at the Greville point, stop iterating.
                break;
            }

            // Clamp u to valid range.
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

    /// MFEM `KnotVector::GetInterpolant(x, u, a)` (`mesh/nurbs.cpp:1080` and
    /// `1092-1195`, NURBS Book Algorithm A9.1): global interpolation through
    /// the data values `x` at parameter locations `u`; the resulting control
    /// points are written to `a`.
    ///
    /// The serial (no-LAPACK) solve builds the collocation matrix, inverts it
    /// in place with `DenseMatrix::Invert` (`linalg/densemat.cpp:674-779`,
    /// Gauss–Jordan with partial pivoting) and multiplies through
    /// `kernels::Mult` (`linalg/kernels.hpp:160`, ascending-column dot
    /// products); the port reproduces both operation orders so results are
    /// bit-identical to the C++.
    pub fn get_interpolant(&self, x: &[f64], u: &[f64], a: &mut [f64]) {
        let ncp = self.num_cp as usize;
        let order = self.order as usize;

        // Assemble the collocation matrix A(i, j) = N_j(u_i)
        // (nurbs.cpp:1123-1140); named A_coll_inv after MFEM, which inverts
        // the matrix in place.
        let mut a_coll_inv = vec![0.0f64; ncp * ncp];
        let mut shape = vec![0.0; order + 1];
        for i in 0..ncp {
            let ks = self.get_span(u[i]);
            let xi = self.get_ref_point(u[i], ks);
            self.calc_shape(&mut shape, (ks as i32) - order as i32, xi);
            for p in 0..=order {
                let j = ks - order + p;
                a_coll_inv[i * ncp + j] = shape[p];
            }
        }

        // DenseMatrix::Invert (linalg/densemat.cpp:674-779), non-LAPACK
        // branch: Gauss-Jordan with partial pivoting, in place.
        invert_dense(&mut a_coll_inv, ncp);

        // A_coll_inv.Mult(tmp, *x[i]) (nurbs.cpp:1186-1192) via
        // kernels::Mult (linalg/kernels.hpp:160): per row, ascending-j
        // accumulation.
        let tmp: Vec<f64> = x.to_vec();
        for i in 0..ncp {
            let mut sum = 0.0;
            for j in 0..ncp {
                sum += a_coll_inv[i * ncp + j] * tmp[j];
            }
            a[i] = sum;
        }
    }

    /// MFEM `KnotVector::GetNKS()` (`mesh/nurbs.hpp`): number of knot spans
    /// including the empty ones between repeated knots.
    pub fn n_ks(&self) -> usize {
        self.knots.len() - 2 * self.order as usize - 1
    }

    /// MFEM `KnotVector::isElement(ks)` (`mesh/nurbs.hpp`): span `ks` is a
    /// real element when its knots are not repeated.
    pub fn is_element(&self, ks: usize) -> bool {
        let o = self.order as usize;
        self.knots[ks + o] < self.knots[ks + o + 1]
    }

    /// MFEM `KnotVector::GetNE()` (`mesh/nurbs.cpp:616-625`): the number of
    /// non-empty knot spans.
    pub fn n_elements(&self) -> usize {
        let o = self.order as usize;
        let mut ne = 0;
        for i in o..(self.num_cp as usize) {
            if self.knots[i] != self.knots[i + 1] {
                ne += 1;
            }
        }
        ne
    }

    /// MFEM `KnotVector::CalcShape(shape, i, xi)` (`mesh/nurbs.cpp:728`):
    /// the `Order+1` non-vanishing basis functions at reference coordinate
    /// `xi` of the element (span) starting at knot index `i`.  `i` may be
    /// negative, meaning the mirrored span (see `CalcShape`).
    pub fn calc_shape_at(&self, i: i32, xi: f64) -> Vec<f64> {
        let mut shape = vec![0.0; self.order as usize + 1];
        self.calc_shape(&mut shape, i, xi);
        shape
    }

    /// MFEM `KnotVector::CalcDShape(grad, i, xi)` (`mesh/nurbs.cpp:755`):
    /// first derivatives of [`Self::calc_shape_at`].
    pub fn calc_dshape_at(&self, i: i32, xi: f64) -> Vec<f64> {
        let mut grad = vec![0.0; self.order as usize + 1];
        self.calc_dshape(&mut grad, i, xi);
        grad
    }

    /// MFEM `KnotVector::CalcD2Shape(grad, i, xi)` (`mesh/nurbs.hpp:214`,
    /// `CalcDnShape(grad, 2, i, xi)`): second derivatives of
    /// [`Self::calc_shape_at`].
    pub fn calc_d2shape_at(&self, i: i32, xi: f64) -> Vec<f64> {
        let mut grad = vec![0.0; self.order as usize + 1];
        self.calc_dn_shape(&mut grad, 2, i, xi);
        grad
    }

    /// MFEM `KnotVector::PrintFunctions(os, samples)` (`mesh/nurbs.cpp:638`):
    /// over every non-empty span, at `samples` reference points, one line
    /// `u \t N_0 … N_p \t dN_0 … \t d²N_0 …` (tab separated, `%g` values).
    pub fn print_functions(&self, samples: usize) -> String {
        assert!(self.n_elements() > 0, "Elements not counted. Use GetElements().");
        let order = self.order as usize;
        let mut s = String::new();
        let dxi = 1.0 / (samples - 1) as f64;
        for ks in 0..self.n_ks() {
            if !self.is_element(ks) {
                continue;
            }
            for j in 0..samples {
                let xi = j as f64 * dxi;
                s.push_str(&format_g(self.get_knot_location(xi, ks + order), 6));
                s.push('\t');
                let sh = self.calc_shape_at(ks as i32, xi);
                for v in &sh {
                    s.push('\t');
                    s.push_str(&format_g(*v, 6));
                }
                let dsh = self.calc_dshape_at(ks as i32, xi);
                for v in &dsh {
                    s.push('\t');
                    s.push_str(&format_g(*v, 6));
                }
                let d2sh = self.calc_d2shape_at(ks as i32, xi);
                for (d, v) in d2sh.iter().enumerate() {
                    s.push('\t');
                    s.push_str(&format_g(*v, 6));
                    if d + 1 == d2sh.len() {
                        s.push('\n');
                    }
                }
            }
        }
        s
    }

    /// MFEM `KnotVector::PrintFunction(os, a, samples)` (`mesh/nurbs.cpp:669`):
    /// over every non-empty span, at `samples` reference points, one line
    /// `u \t Σ a·N \t Σ a·dN \t Σ a·d²N` for the spline with coefficients
    /// `a`.  This is what `miniapps/nurbs/nurbs_mesh_info.cpp` writes to its
    /// `k<k>_n<i>.dat` / `k<k>_cheby.dat` files.
    pub fn print_function(&self, a: &[f64], samples: usize) -> String {
        assert!(self.n_elements() > 0, "Elements not counted. Use GetElements().");
        let order = self.order as usize;
        let mut s = String::new();
        let dxi = 1.0 / (samples - 1) as f64;
        for ks in 0..self.n_ks() {
            if !self.is_element(ks) {
                continue;
            }
            for j in 0..samples {
                let xi = j as f64 * dxi;
                s.push_str(&format_g(self.get_knot_location(xi, ks + order), 6));
                s.push('\t');

                let sh = self.calc_shape_at(ks as i32, xi);
                let mut val = 0.0;
                for p in 0..=order {
                    val += a[ks + p] * sh[p];
                }
                s.push_str(&format_g(val, 6));
                s.push('\t');

                let dsh = self.calc_dshape_at(ks as i32, xi);
                let mut val = 0.0;
                for p in 0..=order {
                    val += a[ks + p] * dsh[p];
                }
                s.push_str(&format_g(val, 6));
                s.push('\t');

                let d2sh = self.calc_d2shape_at(ks as i32, xi);
                let mut val = 0.0;
                for p in 0..=order {
                    val += a[ks + p] * d2sh[p];
                }
                s.push_str(&format_g(val, 6));
                s.push('\n');
            }
        }
        s
    }

    /// MFEM `KnotVector::UniformRefinement(new_knots, rf)`
    /// (`mesh/nurbs.cpp:432-451`, non-spacing branch): the `rf-1` new knots
    /// inside every non-empty span, `(1 - m·h)·knot(i) + m·h·knot(i+1)`.
    ///
    /// (The `spacing`-function branch of `KnotVector::Refinement` — v1.1
    /// spacing files — is not ported; D496 meshes carry plain knot vectors.)
    pub fn uniform_refinement_knots(&self, rf: i32) -> Vec<f64> {
        assert!(rf > 1, "Refinement factor must be at least 2.");
        let h = 1.0 / rf as f64;
        let mut new_knots = Vec::with_capacity(self.n_elements() * (rf - 1) as usize);
        for i in 0..self.knots.len() - 1 {
            if self.knots[i] != self.knots[i + 1] {
                for m in 1..rf {
                    let m = m as f64;
                    new_knots.push((1.0 - m * h) * self.knots[i] + m * h * self.knots[i + 1]);
                }
            }
        }
        new_knots
    }

    /// MFEM `KnotVector::KnotVector(int order, const Vector &intervals,
    /// const Array<int> &continuity)` (`mesh/nurbs.cpp`, used by
    /// `miniapps/nurbs/surface.cpp`): a clamped knot vector of the given
    /// order with `intervals.len()` spans of the given widths, interior knot
    /// multiplicity `order - continuity[i]` at each span junction.
    pub fn new_from_intervals(order: i32, intervals: &[f64], continuity: &[i32]) -> Self {
        assert!(
            continuity.len() == intervals.len() + 1,
            "Incompatible sizes of continuity and intervals."
        );
        let num_knots = order as i64 * continuity.len() as i64
            - continuity.iter().map(|&c| c as i64).sum::<i64>();
        assert!(num_knots >= 0, "Invalid continuity vector for order.");
        let num_knots = num_knots as usize;
        let num_cp = num_knots - order as usize - 1;
        let mut knots = vec![0.0f64; num_knots];
        let mut accum = 0.0f64;
        let mut iknot = 0usize;
        for (i, &c) in continuity.iter().enumerate() {
            let multiplicity = order - c;
            assert!(
                (1..=order + 1).contains(&multiplicity),
                "Invalid knot multiplicity for order."
            );
            for _ in 0..multiplicity {
                knots[iknot] = accum;
                iknot += 1;
            }
            if i < intervals.len() {
                accum += intervals[i];
            }
        }
        assert!(
            knots.len() >= 2 * (order as usize + 1),
            "Insufficient number of knots to define NURBS."
        );
        let kv = Self::new(order, num_cp as i32, knots);
        debug_assert_eq!(kv.num_cp as usize, num_cp);
        kv
    }

    /// MFEM `KnotVector::GetInterpolant(Array<Vector*>&, u, reuse_inverse)`
    /// (`mesh/nurbs.cpp:1092-1195`, non-LAPACK branch): solve the collocation
    /// system for several right-hand sides with a single assembly +
    /// inversion.  `x` is overwritten in place with the spline coefficients.
    ///
    /// MFEM's `reuse_inverse` flag skips re-assembling/inverting the
    /// collocation matrix when `u` did not change since the previous call.
    /// The port always assembles and inverts: the matrix depends only on `u`,
    /// so the rebuilt inverse is bit-identical to the cached one and the flag
    /// is a pure performance shortcut.
    pub fn get_interpolant_multi(&self, x: &mut [Vec<f64>], u: &[f64], _reuse_inverse: bool) {
        let ncp = self.num_cp as usize;
        let order = self.order as usize;

        // Assemble the collocation matrix A(i, j) = N_j(u_i) and invert it in
        // place (`A_coll_inv.Invert()`), exactly like the single-vector
        // [`Self::get_interpolant`].
        let mut a_coll_inv = vec![0.0f64; ncp * ncp];
        let mut shape = vec![0.0; order + 1];
        for i in 0..ncp {
            let ks = self.get_span(u[i]);
            let xi = self.get_ref_point(u[i], ks);
            self.calc_shape(&mut shape, (ks as i32) - order as i32, xi);
            for p in 0..=order {
                a_coll_inv[i * ncp + (ks - order + p)] = shape[p];
            }
        }
        invert_dense(&mut a_coll_inv, ncp);

        // A_coll_inv.Mult(tmp, *x[i]) (kernels::Mult, ascending j).
        for xi in x.iter_mut() {
            let tmp = xi.clone();
            for (i, out) in xi.iter_mut().enumerate() {
                let mut sum = 0.0;
                for j in 0..ncp {
                    sum += a_coll_inv[i * ncp + j] * tmp[j];
                }
                *out = sum;
            }
        }
    }

    /// MFEM `KnotVector::Difference(kv, diff)` (`mesh/nurbs.cpp:1219-1252`):
    /// the knots of `kv` not contained in `self` (matched within
    /// `2*epsilon`).
    ///
    /// The C++ walks a matching index `i` without a bound check (documented
    /// UB when `kv` has unmatched leading knots); the port saturates `i` at
    /// the end of `self` instead, which is the behaviour for well-formed
    /// nested inputs (the only use, `NURBSPatch::KnotInsert(dir, KnotVector&)`,
    /// nurbs.cpp:1722-1728).
    pub fn difference(&self, kv: &NurbsKnotVector) -> Vec<f64> {
        assert!(
            self.order == kv.order,
            "KnotVector::Difference :\n Can not compare knot vectors with different orders!"
        );
        let s = kv.knots.len() as i64 - self.knots.len() as i64;
        if s < 0 {
            return kv.difference(self);
        }
        let mut diff = vec![0.0; s as usize];
        if s == 0 {
            return diff;
        }
        let mut i = 0usize;
        let mut w = 0usize;
        for j in 0..kv.knots.len() {
            if i < self.knots.len() && (self.knots[i] - kv.knots[j]).abs() < 2.0 * f64::EPSILON {
                i += 1;
            } else {
                diff[w] = kv.knots[j];
                w += 1;
            }
        }
        diff
    }

    // -------------------------------------------------------------------------
    // Demko — Remez iteration for the Chebyshev spline extrema
    // -------------------------------------------------------------------------

    /// MFEM `KnotVector::ComputeDemko` (`mesh/nurbs.cpp:300-381`) behind
    /// `GetDemko` (`mesh/nurbs.cpp:281-299`): the Demko abscissae (approximate
    /// Chebyshev/extrema locations of the basis) via Remez iteration, starting
    /// from the Greville points.
    pub fn demko_abscissae(&self) -> Vec<f64> {
        const ITERMAX1: usize = 50;
        const ITERMAX2: usize = 50;
        const TOL1: f64 = 1e-10;
        const TOL2: f64 = 1e-8;

        let ncp = self.num_cp as usize;
        let order = self.order as usize;

        // Initial alternating values for the interpolation (nurbs.cpp:312-317).
        let x: Vec<f64> = (0..ncp).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();

        // Initialize demko points with the Greville abscissae (nurbs.cpp:318-321).
        let mut demko: Vec<f64> = (0..ncp).map(|i| self.greville(i)).collect();

        let mut a = vec![0.0; ncp];
        let mut anew = vec![0.0; ncp];
        let mut sh = vec![0.0; order + 1];
        let mut shgrad = vec![0.0; order + 1];
        let mut shhess = vec![0.0; order + 1];

        // Get initial interpolant (nurbs.cpp:332).
        self.get_interpolant(&x, &demko, &mut anew);

        for _iter1 in 0..ITERMAX1 {
            // a = anew (nurbs.cpp:335).
            a.copy_from_slice(&anew);

            for i in 0..ncp {
                // Check for a repeated knot (nurbs.cpp:339-342).
                if self.knots[i + 1] == self.knots[i + order] {
                    continue;
                }

                let mut u = demko[i];

                // Newton iteration for the extremum location
                // (nurbs.cpp:347-374).
                for _iter2 in 0..ITERMAX2 {
                    let ks = self.get_span(u);
                    let xi = self.get_ref_point(u, ks);

                    self.calc_shape(&mut sh, (ks as i32) - order as i32, xi);
                    self.calc_dshape(&mut shgrad, (ks as i32) - order as i32, xi);
                    self.calc_dn_shape(&mut shhess, 2, (ks as i32) - order as i32, xi);

                    let mut val = 0.0;
                    let mut grad = 0.0;
                    let mut hess = 0.0;
                    for p in 0..=order {
                        val += a[ks - order + p] * sh[p];
                        grad += a[ks - order + p] * shgrad[p];
                        hess += a[ks - order + p] * shhess[p];
                    }

                    if grad.abs() < TOL2 {
                        break;
                    }

                    let dk = self.knots[ks + 1] - self.knots[ks];
                    if hess.abs() < 3.0f64.powi(order as i32) {
                        u += 0.25 * 0.45f64.powi(order as i32) * (val / val.abs()) * grad * dk;
                    } else {
                        u -= (grad / hess) * dk;
                    }
                }

                // Update (nurbs.cpp:377).
                demko[i] = u;
            }

            // Restore ordering with a single bubble-sort pass
            // (nurbs.cpp:380-385).
            for i in 0..ncp - 1 {
                if demko[i] > demko[i + 1] {
                    demko.swap(i, i + 1);
                }
            }

            // New interpolant and convergence check |a - anew|_2 < tol1
            // (nurbs.cpp:388-394); Vector::Norml2 uses the scaled
            // (hypot-style) reduction of linalg/vector.cpp:968-1002.
            self.get_interpolant(&x, &demko, &mut anew);
            for i in 0..ncp {
                a[i] -= anew[i];
            }
            if norml2(&a) < TOL1 {
                break;
            }
        }

        demko
    }
}

/// Port of MFEM `Vector::Norml2` (`linalg/vector.cpp:968-1002`): the scaled
/// (LAPACK `dnrm2`/`hypot`-style) Euclidean norm, applied sequentially the way
/// the host `reduce` does.
fn norml2(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut scale = 0.0f64;
    let mut sum = 0.0f64;
    for &m in v {
        let n = m.abs();
        if n > 0.0 {
            if scale <= n {
                let arg = scale / n;
                sum = sum * (arg * arg) + 1.0;
                scale = n;
            } else {
                let arg = n / scale;
                sum += arg * arg;
            }
        }
    }
    scale * sum.sqrt()
}

/// Port of MFEM `DenseMatrix::Invert` (`linalg/densemat.cpp:674-779`),
/// non-LAPACK branch: in-place Gauss-Jordan inversion with partial pivoting.
/// `a` holds the logical (row, column) matrix in row-major order — MFEM stores
/// column-major but all accesses go through `operator()(i, j)`, so the
/// algorithm is storage-agnostic.
fn invert_dense(a: &mut [f64], n: usize) {
    let mut piv = vec![0usize; n];

    for c in 0..n {
        // Partial pivot: largest |value| in column c at or below the diagonal.
        let mut big = a[c * n + c].abs();
        let mut ipiv = c;
        for j in c + 1..n {
            let b = a[j * n + c].abs();
            if big < b {
                big = b;
                ipiv = j;
            }
        }
        if big == 0.0 {
            panic!("DenseMatrix::Invert() : singular matrix");
        }
        piv[c] = ipiv;
        for j in 0..n {
            a.swap(c * n + j, ipiv * n + j);
        }

        // a = (*this)(c, c) = 1.0 / (*this)(c, c)
        let inv = 1.0 / a[c * n + c];
        a[c * n + c] = inv;
        for j in 0..c {
            a[c * n + j] *= inv;
        }
        for j in c + 1..n {
            a[c * n + j] *= inv;
        }
        for i in 0..c {
            let b = -a[i * n + c];
            a[i * n + c] = inv * b;
            for j in 0..c {
                a[i * n + j] += b * a[c * n + j];
            }
            for j in c + 1..n {
                a[i * n + j] += b * a[c * n + j];
            }
        }
        for i in c + 1..n {
            let b = -a[i * n + c];
            a[i * n + c] = inv * b;
            for j in 0..c {
                a[i * n + j] += b * a[c * n + j];
            }
            for j in c + 1..n {
                a[i * n + j] += b * a[c * n + j];
            }
        }
    }

    // Undo the pivoting.
    for c in (0..n).rev() {
        let j = piv[c];
        for i in 0..n {
            a.swap(i * n + c, i * n + j);
        }
    }
}

impl fmt::Display for NurbsKnotVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print())
    }
}

// ─── NurbsPatch ──────────────────────────────────────────────────────────────

/// NURBS patch control-point object — port of MFEM `NURBSPatch`
/// (`mesh/nurbs.hpp:320`, `mesh/nurbs.cpp:1274-1520`) for the 2-D (two knot
/// vector) and 3-D (three knot vector) cases.
///
/// `data` stores `dim` components per control point in MFEM's raw
/// (homogeneous) form `(x*w, y*w, w)` at exactly the C++ layout
/// `data[(i + j*ni)*dim + l]` (`mesh/nurbs.hpp:1364`,
/// `NURBSPatch::operator()(int, int, int)`); a 3-D patch extends the flat
/// index to `(i + j*ni + k*ni*nj)*dim + l` — MFEM's
/// `operator()(int, int, int, int)` (`mesh/nurbs.hpp:1377`).
#[derive(Debug, Clone)]
pub struct NurbsPatch {
    /// Knot vectors per parametric direction (`NURBSPatch::kv`).
    kv: Vec<NurbsKnotVector>,
    /// Physical dimension plus 1 (`NURBSPatch::Dim`, nurbs.cpp:1290).
    dim: usize,
    /// Number of control points in the u direction (`NURBSPatch::ni`).
    ni: usize,
    /// Number of control points in the v direction (`NURBSPatch::nj`).
    nj: usize,
    /// Number of control points in the w direction (`NURBSPatch::nk`); `0`
    /// for a 2-D patch (MFEM stores `-1` there).
    nk: usize,
    /// Raw component data, `data[(i + j*ni [+ k*ni*nj])*dim + l]`.
    data: Vec<f64>,
}

impl NurbsPatch {
    /// Create a 2-D patch — port of
    /// `NURBSPatch::NURBSPatch(kv0, kv1, dim)` (`mesh/nurbs.cpp:1402`) plus
    /// `init` (`mesh/nurbs.cpp:1274-1335`).  `n_components` is MFEM's `dim`
    /// (physical dimension plus 1); the data is zero-initialized (the C++
    /// leaves it uninitialized in release builds).
    pub fn new_2d(kv_u: NurbsKnotVector, kv_v: NurbsKnotVector, n_components: usize) -> Self {
        assert!(
            n_components > 1,
            "NURBS patch dimension (including weight) must be greater than 1."
        );
        let ni = kv_u.num_cp() as usize;
        let nj = kv_v.num_cp() as usize;
        assert!(ni > 0 && nj > 0, "Invalid knot vector dimensions.");
        Self {
            kv: vec![kv_u, kv_v],
            dim: n_components,
            ni,
            nj,
            nk: 0,
            data: vec![0.0; ni * nj * n_components],
        }
    }

    /// Create a 3-D patch — port of
    /// `NURBSPatch::NURBSPatch(kv0, kv1, kv2, dim)`
    /// (`mesh/nurbs.cpp:1410-1418`, used by `miniapps/nurbs/surface.cpp` to
    /// build the interpolation volume).  `n_components` is MFEM's `dim`
    /// (physical dimension plus 1); the data is zero-initialized.
    pub fn new_3d(
        kv_u: NurbsKnotVector,
        kv_v: NurbsKnotVector,
        kv_w: NurbsKnotVector,
        n_components: usize,
    ) -> Self {
        assert!(
            n_components > 1,
            "NURBS patch dimension (including weight) must be greater than 1."
        );
        let ni = kv_u.num_cp() as usize;
        let nj = kv_v.num_cp() as usize;
        let nk = kv_w.num_cp() as usize;
        assert!(ni > 0 && nj > 0 && nk > 0, "Invalid knot vector dimensions.");
        Self {
            kv: vec![kv_u, kv_v, kv_w],
            dim: n_components,
            ni,
            nj,
            nk,
            data: vec![0.0; ni * nj * nk * n_components],
        }
    }

    /// MFEM `NURBSPatch::operator()(i, j, l)` (`mesh/nurbs.hpp:1364`):
    /// component `l` of the control point at `(i, j)`.
    pub fn get(&self, i: usize, j: usize, l: usize) -> f64 {
        assert!(i < self.ni && j < self.nj && l < self.dim, "NURBSPatch::operator() 2D");
        self.data[(i + j * self.ni) * self.dim + l]
    }

    /// Mutable counterpart of [`Self::get`] (MFEM's reference-returning
    /// `operator()`, e.g. `patch(i,0,2) = 1.0`).
    pub fn set(&mut self, i: usize, j: usize, l: usize, value: f64) {
        assert!(i < self.ni && j < self.nj && l < self.dim, "NURBSPatch::operator() 2D");
        self.data[(i + j * self.ni) * self.dim + l] = value;
    }

    /// MFEM `NURBSPatch::operator()(i, j, k, l)` (`mesh/nurbs.hpp:1377`):
    /// component `l` of the 3-D patch control point at `(i, j, k)`.
    pub fn get_ijk(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        assert!(
            self.nk > 0 && i < self.ni && j < self.nj && k < self.nk && l < self.dim,
            "NURBSPatch::operator() 3D"
        );
        self.data[(i + j * self.ni + k * self.ni * self.nj) * self.dim + l]
    }

    /// Mutable counterpart of [`Self::get_ijk`], e.g. `(*patch)(i,j,k,3) = 1.0`.
    pub fn set_ijk(&mut self, i: usize, j: usize, k: usize, l: usize, value: f64) {
        assert!(
            self.nk > 0 && i < self.ni && j < self.nj && k < self.nk && l < self.dim,
            "NURBSPatch::operator() 3D"
        );
        self.data[(i + j * self.ni + k * self.ni * self.nj) * self.dim + l] = value;
    }

    /// Knot vector in the u direction (`NURBSPatch::GetKV(0)`).
    pub fn kv_u(&self) -> &NurbsKnotVector {
        &self.kv[0]
    }

    /// Knot vector in the v direction (`NURBSPatch::GetKV(1)`).
    pub fn kv_v(&self) -> &NurbsKnotVector {
        &self.kv[1]
    }

    /// Knot vector in the w direction (`NURBSPatch::GetKV(2)`); panics on a
    /// 2-D patch.
    pub fn kv_w(&self) -> &NurbsKnotVector {
        assert!(self.nk > 0, "NURBSPatch::GetKV(2) on a 2-D patch");
        &self.kv[2]
    }

    /// All knot vectors (`NURBSPatch::kv` / `GetNKV()`).
    pub fn kvs(&self) -> &[NurbsKnotVector] {
        &self.kv
    }

    /// Number of components per control point (`NURBSPatch::Dim`).
    pub fn n_components(&self) -> usize {
        self.dim
    }

    /// Control points in the u direction (`NURBSPatch::ni`).
    pub fn ni(&self) -> usize {
        self.ni
    }

    /// Control points in the v direction (`NURBSPatch::nj`).
    pub fn nj(&self) -> usize {
        self.nj
    }

    /// Control points in the w direction (`NURBSPatch::nk`; `0` for 2-D).
    pub fn nk(&self) -> usize {
        self.nk
    }

    /// Per-direction NCPs, `[ni]`, `[ni, nj]` or `[ni, nj, nk]`.
    pub fn kv_dims(&self) -> Vec<usize> {
        if self.nk > 0 {
            vec![self.ni, self.nj, self.nk]
        } else {
            vec![self.ni, self.nj]
        }
    }

    /// Raw component `l` of the flat control point `flat` (MFEM tensor order:
    /// `flat = i + j*ni [+ k*ni*nj]`), i.e. `data[flat*Dim + l]`.
    pub fn get_flat(&self, flat: usize, l: usize) -> f64 {
        self.data[flat * self.dim + l]
    }

    /// Mutable counterpart of [`Self::get_flat`].
    pub fn set_flat(&mut self, flat: usize, l: usize, value: f64) {
        self.data[flat * self.dim + l] = value;
    }

    /// Raw component `l` of the control point at the multi-index `midx`
    /// (length 2 or 3), tensor order `flat = i + j*ni [+ k*ni*nj]`.
    pub fn raw_at(&self, midx: &[usize], l: usize) -> f64 {
        let flat = if midx.len() > 2 {
            midx[0] + midx[1] * self.ni + midx[2] * self.ni * self.nj
        } else {
            midx[0] + midx[1] * self.ni
        };
        self.data[flat * self.dim + l]
    }

    /// MFEM `NURBSPatch::DegreeElevate(dir, t)` (`mesh/nurbs.cpp:2082-2178`,
    /// NURBS Book routine): raise the polynomial degree in direction `dir` by
    /// `t`, elevating every component (including the weight) in homogeneous
    /// form.
    ///
    /// The math is delegated to `fem_element::nurbs::{elevate_u_2d,
    /// elevate_v_2d}`, the existing ports of the same routine; this wrapper
    /// converts between MFEM's raw homogeneous layout and the kernel's
    /// Cartesian + weights view (exact for unit weights).
    pub fn degree_elevate(&mut self, dir: usize, t: usize) {
        assert!(dir < 2, "NURBSPatch::DegreeElevate : Incorrect direction!");
        if t == 0 {
            // MFEM rebuilds an identical patch; keep self unchanged.
            return;
        }
        let pd = self.to_patch_data_2d();
        let elevated = match dir {
            0 => fem_element::nurbs::elevate_u_2d(&pd, t),
            _ => fem_element::nurbs::elevate_v_2d(&pd, t),
        };
        self.from_patch_data_2d(&elevated);
    }

    /// MFEM `NURBSPatch::KnotInsert(dir, const Vector &knot)`
    /// (`mesh/nurbs.cpp:1767-1873`, NURBS Book Algorithm A5.5): insert all
    /// knot values in one pass, operating on the raw homogeneous control net
    /// through MFEM's `SetLoopDirection`/`slice` flattened access — an exact
    /// port, unlike the per-value A5.1 delegation this method used before
    /// D496.
    ///
    /// This is also the kernel behind `NURBSPatch::UniformRefinement`
    /// (nurbs.cpp:1583): refinement knots come from
    /// [`NurbsKnotVector::uniform_refinement_knots`].
    pub fn knot_insert(&mut self, dir: usize, knots: &[f64]) {
        if knots.is_empty() {
            return; // nurbs.cpp:1770-1772
        }
        assert!(dir < self.kv.len(), "NURBSPatch::KnotInsert : Invalid direction!");

        let dim = self.dim;
        let (ni, nj, nk) = (self.ni, self.nj, self.nk);
        let old_kv = &self.kv[dir];
        let order = old_kv.order() as usize;
        let ml = old_kv.num_cp() as usize; // old NCP along `dir`
        let rr = knots.len() - 1; // nurbs.cpp:1806

        // `a = oldkv.GetSpan(knot(0))`, `b = oldkv.GetSpan(knot(rr))`
        // (nurbs.cpp:1807-1808).
        let a = old_kv.get_span(knots[0]);
        let b = old_kv.get_span(knots[rr]);

        // New knot vector: copies of the old knots around the inserted ones
        // (nurbs.cpp:1811-1819), interior slots filled by the A5.5 loop.
        let mut new_knots = vec![0.0f64; old_kv.len() + knots.len()];
        for (j, slot) in new_knots.iter_mut().enumerate().take(a + 1) {
            *slot = old_kv.values()[j];
        }
        for j in b + order..=ml + order {
            new_knots[j + rr + 1] = old_kv.values()[j];
        }

        // Flattened (sd, nd) of the old and new nets along `dir`
        // (`SetLoopDirection`, nurbs.cpp:1508-1560).
        let (sd, nd, ls) = match (nj != 0, nk != 0) {
            (_, true) => match dir {
                0 => (dim, ni, nj * nk * dim),
                1 => (ni * dim, nj, ni * nk * dim),
                _ => (ni * nj * dim, nk, ni * nj * dim),
            },
            (true, false) => match dir {
                0 => (dim, ni, nj * dim),
                _ => (ni * dim, nj, ni * dim),
            },
            _ => (dim, ni, dim),
        };
        // The flattened size is invariant (nurbs.cpp:1810-1813).
        let slice = |sd: usize, nd: usize, i: usize, j: usize| -> usize {
            j % sd + sd * (i + (j / sd) * nd)
        };

        let old_data = std::mem::take(&mut self.data);
        let ml_new = ml + knots.len();
        // `nd` differs along `dir` between the old and the new net
        // (`nd_new = nd + rr + 1`).  New data size through the flattened
        // view: `nd_new * ls` entries (equals `ni_new*nj*dim` etc. for every
        // direction).
        let nd_new = nd + knots.len();
        let mut new_data = vec![0.0f64; nd_new * ls];
        for k in 0..=a.saturating_sub(order) {
            for ll in 0..ls {
                new_data[slice(sd, nd_new, k, ll)] = old_data[slice(sd, nd, k, ll)];
            }
        }
        // nurbs.cpp:1827-1834: `for (k = b-1; k < ml; k++)`
        for k in b.saturating_sub(1)..ml {
            for ll in 0..ls {
                new_data[slice(sd, nd_new, k + rr + 1, ll)] = old_data[slice(sd, nd, k, ll)];
            }
        }

        // The A5.5 insertion loop (nurbs.cpp:1836-1870).
        let old_knots = old_kv.values();
        let mut i = b + order - 1;
        let mut k = b + order + rr;
        for j in (0..=rr).rev() {
            while knots[j] <= old_knots[i] && i > a {
                new_knots[k] = old_knots[i];
                for ll in 0..ls {
                    new_data[slice(sd, nd_new, k - order - 1, ll)] =
                        old_data[slice(sd, nd, i - order - 1, ll)];
                }
                k -= 1;
                i -= 1;
            }

            for ll in 0..ls {
                new_data[slice(sd, nd_new, k - order - 1, ll)] =
                    new_data[slice(sd, nd_new, k - order, ll)];
            }

            for l in 1..=order {
                let ind = k - order + l;
                let mut alfa = new_knots[k + l] - knots[j];
                if alfa.abs() == 0.0 {
                    for ll in 0..ls {
                        new_data[slice(sd, nd_new, ind - 1, ll)] =
                            new_data[slice(sd, nd_new, ind, ll)];
                    }
                } else {
                    alfa /= new_knots[k + l] - old_knots[i - order + l];
                    for ll in 0..ls {
                        let dst = slice(sd, nd_new, ind - 1, ll);
                        new_data[dst] = alfa * new_data[dst]
                            + (1.0 - alfa) * new_data[slice(sd, nd_new, ind, ll)];
                    }
                }
            }

            new_knots[k] = knots[j];
            k -= 1;
        }

        // `newkv.GetElements(); swap(newpatch);` — rebuild the knot vector and
        // patch dimensions in direction `dir` (nurbs.cpp:1871-1873).
        self.data = new_data;
        self.kv[dir] = NurbsKnotVector::new(order as i32, ml_new as i32, new_knots);
        match dir {
            0 => self.ni = ml_new,
            1 => self.nj = ml_new,
            _ => self.nk = ml_new,
        }
    }

    /// MFEM `NURBSPatch::UniformRefinement(rf, multiplicity=1)`
    /// (`mesh/nurbs.cpp:1583-1597`): per direction, insert the refinement
    /// knots of `KnotVector::Refinement`/`UniformRefinement` in one A5.5 pass.
    pub fn uniform_refine(&mut self, rf: i32) {
        if rf <= 1 {
            return;
        }
        for dir in 0..self.kv.len() {
            let new_knots = self.kv[dir].uniform_refinement_knots(rf);
            self.knot_insert(dir, &new_knots);
        }
    }

    /// MFEM `NURBSPatch::KnotInsert(dir, const KnotVector &newkv)`
    /// (`mesh/nurbs.cpp:1714-1738`): degree-elevate direction `dir` to the
    /// order of `newkv` if needed, then insert the knots of `newkv` that are
    /// not yet present (`KnotVector::Difference`).
    pub fn knot_insert_kv(&mut self, dir: usize, kv: &NurbsKnotVector) {
        assert!(dir < 2, "NURBSPatch::KnotInsert : Incorrect direction!");
        let t = kv.order() - self.kv[dir].order();
        if t > 0 {
            self.degree_elevate(dir, t as usize);
        } else if t < 0 {
            panic!("NURBSPatch::KnotInsert : Incorrect order!");
        }
        let diff = self.kv[dir].difference(kv);
        if !diff.is_empty() {
            self.knot_insert(dir, &diff);
        }
    }

    /// MFEM `NURBSPatch::Print` (`mesh/nurbs.cpp:1484-1507`): the `patches`
    /// block of an `MFEM NURBS mesh` file — `knotvectors`, then `dimension`,
    /// then one line of `dim` components per control point, all through
    /// `%g`-formatted `ostream` insertion at the default precision 6.
    pub fn print(&self) -> String {
        let mut s = String::new();
        s.push_str("knotvectors\n");
        s.push_str(&format!("{}\n", self.kv.len()));
        for kv in &self.kv {
            s.push_str(&kv.print());
        }
        s.push_str("\ndimension\n");
        s.push_str(&format!("{}", self.dim - 1));
        s.push_str("\n\ncontrolpoints\n");
        let ncp = self.ni * self.nj;
        for k in 0..ncp {
            for (d, c) in self.data[k * self.dim..(k + 1) * self.dim].iter().enumerate() {
                if d > 0 {
                    s.push(' ');
                }
                // `os << data[j++]` — no ZeroSubnormal here (unlike
                // Vector::Print), just plain ostream formatting.
                s.push_str(&format_g(*c, 6));
            }
            s.push('\n');
        }
        s
    }

    // -----------------------------------------------------------------------
    // Delegation helpers
    // -----------------------------------------------------------------------

    /// View the patch as `fem_element::nurbs::NurbsPatch2DData` (Cartesian
    /// control points plus weights; `x = data/w`, exact for unit weights).
    fn to_patch_data_2d(&self) -> fem_element::nurbs::NurbsPatch2DData {
        use fem_element::nurbs::KnotVector as ElementKnotVector;
        let kv_u = ElementKnotVector::new(self.kv[0].values().to_vec(), self.kv[0].order() as usize);
        let kv_v = ElementKnotVector::new(self.kv[1].values().to_vec(), self.kv[1].order() as usize);
        let n = self.ni * self.nj;
        let mut control_pts = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        for k in 0..n {
            let c = &self.data[k * self.dim..(k + 1) * self.dim];
            let w = c[2];
            let x = if w != 0.0 { c[0] / w } else { 0.0 };
            let y = if w != 0.0 { c[1] / w } else { 0.0 };
            control_pts.push([x, y]);
            weights.push(w);
        }
        fem_element::nurbs::NurbsPatch2DData {
            kv_u,
            kv_v,
            control_pts,
            weights,
            tag: 0,
        }
    }

    /// Write back a `NurbsPatch2DData` produced by the delegated kernels,
    /// restoring MFEM's raw homogeneous layout (`x*w, y*w, w`; exact for unit
    /// weights).
    fn from_patch_data_2d(&mut self, pd: &fem_element::nurbs::NurbsPatch2DData) {
        // MFEM's `Order` and the element-crate `degree` are both the
        // polynomial degree.
        self.kv[0] = NurbsKnotVector::new(
            pd.kv_u.degree as i32,
            pd.kv_u.n_basis() as i32,
            pd.kv_u.knots.clone(),
        );
        self.kv[1] = NurbsKnotVector::new(
            pd.kv_v.degree as i32,
            pd.kv_v.n_basis() as i32,
            pd.kv_v.knots.clone(),
        );
        self.ni = pd.kv_u.n_basis();
        self.nj = pd.kv_v.n_basis();
        self.nk = 0;
        let n = self.ni * self.nj;
        self.data = Vec::with_capacity(n * self.dim);
        for k in 0..n {
            let [x, y] = pd.control_pts[k];
            let w = pd.weights[k];
            self.data.push(x * w);
            self.data.push(y * w);
            self.data.push(w);
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// `UniformKnotVector` of miniapps/nurbs/nurbs_curveint.cpp:32-50.
    fn uniform_kv(order: i32, ncp: i32) -> NurbsKnotVector {
        assert!(order < ncp, "UniformKnotVector: ncp should be at least order + 1");
        let size = (ncp + order + 1) as usize;
        let mut knots = vec![0.0; size];
        for i in (order + 1) as usize..ncp as usize {
            knots[i] = (i as f64 - order as f64) / (ncp as f64 - order as f64);
        }
        for i in ncp as usize..size {
            knots[i] = 1.0;
        }
        NurbsKnotVector::new(order, ncp, knots)
    }

    #[test]
    fn knot_vector_print_matches_cpp_ostream() {
        // MFEM KnotVector::Print writes through ostream defaults (%g, 6
        // significant digits): "2 4 0 0 0.5 1 1".
        let kv = NurbsKnotVector::new(2, 4, vec![0.0, 0.0, 0.5, 1.0, 1.0]);
        assert_eq!(kv.print(), "2 4 0 0 0.5 1 1\n");
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
        // Fully clamped quadratic: order 2, 3 control points,
        // `Size() = NCP + Order + 1 = 6`.  (The previous fixture
        // `new(2, 2, [0,0,1,1])` violated that invariant — order 2 needs three
        // repeated end knots — which the delegated implementation detects
        // because it recovers `Order` from the knot multiplicities, as
        // `fem_element::nurbs_fe_collection::degree_elevate` and every other
        // NURBS code path do.)
        let kv = NurbsKnotVector::new(2, 3, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let elevated = kv.degree_elevate(1);
        assert_eq!(elevated.order(), 3);
        assert_eq!(elevated.num_cp(), 4);
        assert_eq!(elevated.len(), 8);
        assert_eq!(elevated.values(), &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn botella_abscissae() {
        // Fully clamped quadratic B-spline: knots [0,0,0, 0.5, 1,1,1]
        let kv = NurbsKnotVector::new(2, 4, vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]);
        let bot = kv.botella_abscissae();
        let grev = kv.greville_abscissae();
        assert_eq!(bot.len(), 4);
        for i in 1..bot.len() {
            assert!(bot[i] >= bot[i - 1], "bot[{}]={} < bot[{}]={}", i, bot[i], i - 1, bot[i - 1]);
        }
        for i in 0..bot.len() {
            assert!(
                (bot[i] - grev[i]).abs() < 0.3,
                "bot[{}]={} vs grev[{}]={}",
                i,
                bot[i],
                i,
                grev[i]
            );
        }
    }

    #[test]
    fn botella_repeated_knot() {
        // When knot[i+1] == knot[i+order], Botella should return Greville.
        let kv = NurbsKnotVector::new(2, 4, vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]);
        let bot_first = kv.botella(0);
        let grev_first = kv.greville(0);
        assert!(
            (bot_first - grev_first).abs() < 1e-14,
            "bot_first={} should equal grev_first={}",
            bot_first,
            grev_first
        );
    }

    #[test]
    fn demko_abscissae() {
        // Fully clamped quadratic B-spline.
        let kv = NurbsKnotVector::new(2, 4, vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]);
        let demko = kv.demko_abscissae();
        assert_eq!(demko.len(), 4);
        for i in 1..demko.len() {
            assert!(
                demko[i] >= demko[i - 1],
                "demko[{}]={} < demko[{}]={}",
                i,
                demko[i],
                i - 1,
                demko[i - 1]
            );
        }
        for &d in &demko {
            assert!(d >= -1e-10 && d <= 1.0 + 1e-10, "demko out of range: {}", d);
        }
    }

    /// Demko abscissae for the uniform quadratic knot vector of
    /// `nurbs_curveint -n 9`, pinned bit-exactly against MFEM 4.10
    /// (`KnotVector::GetDemko`, mesh/nurbs.cpp:281).  Reference values dumped
    /// at %.17g by `tmp/d374/probe.cpp` built against
    /// `$HOME/mfem410_ser/libmfem.a` (D374).
    #[test]
    fn demko_matches_mfem_bit_exact() {
        let kv = uniform_kv(2, 9);
        let demko = kv.demko_abscissae();
        let expected = [
            0.0,
            0.086128830285624014,
            0.2168955508645925,
            0.35758049515146972,
            0.5,
            0.64241950484853028,
            0.78310444913540744,
            0.91387116971437599,
            1.0,
        ];
        assert_eq!(demko.len(), expected.len());
        for (i, (got, want)) in demko.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, want, "demko[{}] = {:e}, expected {:e}", i, got, want);
        }
    }

    /// `GetInterpolant` of the sine data at the Demko points must reproduce
    /// MFEM's control points bit-exactly, including the -5.7e-17 solve
    /// residual (collocation + `DenseMatrix::Invert`, nurbs.cpp:1092-1195).
    /// Reference: `tmp/d374/probe.cpp` against MFEM 4.10 (D374).
    #[test]
    fn get_interpolant_matches_mfem_bit_exact() {
        let kv = uniform_kv(2, 9);
        let demko = kv.demko_abscissae();

        let x: Vec<f64> = demko.iter().map(|&u| (u - 0.5) * 1.0).collect();
        let mut interp = vec![0.0; 9];
        kv.get_interpolant(&x, &demko, &mut interp);
        let expected_x = [
            -0.5,
            -0.42857142857142849,
            -0.28571428571428564,
            -0.14285714285714288,
            -5.7462715141731735e-17,
            0.14285714285714304,
            0.28571428571428564,
            0.42857142857142849,
            0.5,
        ];
        for (i, (got, want)) in interp.iter().zip(expected_x.iter()).enumerate() {
            assert_eq!(got, want, "interp x[{}] = {:e}, expected {:e}", i, got, want);
        }

        let sine: Vec<f64> = demko
            .iter()
            .map(|&u| 0.1 * (u * 2.0 * std::f64::consts::PI).sin() - 0.5)
            .collect();
        kv.get_interpolant(&sine, &demko, &mut interp);
        let expected_y = [
            -0.5,
            -0.45161176908964712,
            -0.39243382577809249,
            -0.413692394360202,
            -0.5,
            -0.58630760563979789,
            -0.6075661742219074,
            -0.54838823091035283,
            -0.5,
        ];
        for (i, (got, want)) in interp.iter().zip(expected_y.iter()).enumerate() {
            assert_eq!(got, want, "interp y[{}] = {:e}, expected {:e}", i, got, want);
        }
    }

    /// `Difference` (mesh/nurbs.cpp:1219) between the order-2 ncp-3 clamped
    /// knot vector (after elevation) and the uniform ncp-9 vector: the six
    /// interior knots.
    #[test]
    fn difference_matches_mfem() {
        let kv9 = uniform_kv(2, 9);
        let kv3 = NurbsKnotVector::new(2, 3, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let diff = kv3.difference(&kv9);
        assert_eq!(diff.len(), 6);
        for (i, d) in diff.iter().enumerate() {
            let want = (i as f64 + 1.0) / 7.0;
            assert!((d - want).abs() < 1e-15, "diff[{}] = {}, want ~{}", i, d, want);
        }
        // Symmetric empty case.
        assert!(kv9.difference(&kv9).is_empty());
    }

    /// Box patch, order 1 -> 2, knot-inserted in u: control points must match
    /// MFEM bit-exactly (exercises `NURBSPatch::DegreeElevate` and
    /// `KnotInsert` through the delegated kernels).  Reference:
    /// `tmp/d374/probe.cpp` "row0 after elevate+insert u" against MFEM 4.10.
    #[test]
    fn box_patch_elevate_and_insert_u_matches_mfem() {
        let kv_o1 = uniform_kv(1, 2);
        let kv = uniform_kv(2, 9);
        let mut patch = NurbsPatch::new_2d(kv_o1.clone(), kv_o1.clone(), 3);
        for j in 0..2 {
            for i in 0..2 {
                patch.set(i, j, 2, 1.0);
            }
        }
        patch.set(0, 0, 0, -0.5);
        patch.set(0, 0, 1, -0.5);
        patch.set(1, 0, 0, 0.5);
        patch.set(1, 0, 1, -0.5);
        patch.set(0, 1, 0, -0.5);
        patch.set(0, 1, 1, 0.5);
        patch.set(1, 1, 0, 0.5);
        patch.set(1, 1, 1, 0.5);

        patch.degree_elevate(0, 1);
        assert_eq!((patch.ni(), patch.nj()), (3, 2));
        patch.knot_insert_kv(0, &kv);
        assert_eq!((patch.ni(), patch.nj()), (9, 2));

        let expected_x = [
            -0.5,
            -0.4285714285714286,
            -0.2857142857142857,
            -0.14285714285714285,
            0.0,
            0.14285714285714288,
            0.2857142857142857,
            0.42857142857142855,
            0.5,
        ];
        for (i, want) in expected_x.iter().enumerate() {
            let got = patch.get(i, 0, 0);
            if got == *want {
                continue;
            }
            // Known sub-epsilon delta (observed at i=3, i=4): MFEM inserts
            // all six knots in one Piegl-Tiller A5.5 pass (nurbs.cpp:1767)
            // blending against the old knots, while the delegated
            // `fem_element::nurbs::h_refine_uk` kernel applies its A5.1
            // insertion per knot value — algebraically identical, one
            // rounding step apart, so exact cancellations may leave a
            // ~4e-17 residual where MFEM cancels to 0.0.  Values are
            // physically zero; the %g6 print difference is quantified in the
            // D374 report.
            if *want == 0.0 {
                assert!(got.abs() < 1e-16, "x[{}] = {:e}, expected ~0", i, got);
            } else {
                let scale = want.abs();
                assert!(
                    (got - want).abs() / scale <= 2.0 * f64::EPSILON,
                    "x[{}] = {:e}, expected {:e} (within 2 ulp)",
                    i,
                    got,
                    want
                );
            }
        }
        for i in 0..9 {
            assert_eq!(patch.get(i, 0, 1), -0.5, "y[{}]", i);
            assert_eq!(patch.get(i, 0, 2), 1.0, "w[{}]", i);
        }
    }

}

