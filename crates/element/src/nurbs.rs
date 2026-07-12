#![allow(non_snake_case)]
//! B-spline and NURBS finite elements for isogeometric analysis (IGA).
//!
//! # Overview
//!
//! This module provides:
//! - [`KnotVector`] — a B-spline knot sequence with the Cox-de Boor recursion.
//! - [`BSplineBasis1D`] — 1-D B-spline basis on a single knot vector.
//! - [`NurbsPatch2D`] — 2-D NURBS patch implementing [`ReferenceElement`].
//! - [`NurbsPatch3D`] — 3-D NURBS patch implementing [`ReferenceElement`].
//! - [`NurbsMesh`] — a collection of NURBS patches with inter-patch connectivity.
//!
//! # Mathematical background
//!
//! B-spline basis functions $N_{i,p}(\xi)$ are defined recursively by the
//! **Cox-de Boor** formula:
//!
//! $$N_{i,0}(\xi) = \begin{cases} 1 & \text{if } \xi \in [\Xi_i, \Xi_{i+1}) \\ 0 & \text{otherwise} \end{cases}$$
//!
//! $$N_{i,p}(\xi) = \frac{\xi - \Xi_i}{\Xi_{i+p} - \Xi_i} N_{i,p-1}(\xi) + \frac{\Xi_{i+p+1} - \xi}{\Xi_{i+p+1} - \Xi_{i+1}} N_{i+1,p-1}(\xi)$$
//!
//! NURBS basis functions are the rational enrichment:
//!
//! $$R_{i,p}(\xi) = \frac{N_{i,p}(\xi) w_i}{\sum_j N_{j,p}(\xi) w_j}$$
//!
//! In 2-D (tensor product): $R_{ij,p,q}(\xi,\eta) = R_{i,p}(\xi) R_{j,q}(\eta)$, etc.
//!
//! # References
//! - Piegl & Tiller, *The NURBS Book* (2nd ed., 1997).
//! - Hughes, Cottrell & Bazilevs, *Isogeometric Analysis: CAD, Finite Elements,
//!   NURBS, Exact Geometry and Mesh Refinement*, CMAME 2005.

use crate::quadrature::{hex_rule, quad_rule};
use crate::reference::{QuadratureRule, ReferenceElement};

// ─── KnotVector ───────────────────────────────────────────────────────────────

/// A B-spline knot vector $\Xi = \{\xi_0, \xi_1, \ldots, \xi_{m}\}$.
///
/// The knot vector must be non-decreasing.  Clamped (open) knot vectors have
/// $p+1$ equal knots at each end; those are the standard choice for IGA since
/// they yield interpolating boundary conditions.
///
/// # Example
/// ```rust,ignore
/// let kv = KnotVector::uniform(1, 5); // degree 1, 5 elements on [0,1]
/// ```
#[derive(Debug, Clone)]
pub struct KnotVector {
    /// Knot values (non-decreasing).
    pub knots: Vec<f64>,
    /// Polynomial degree $p$.
    pub degree: usize,
}

impl KnotVector {
    /// Create a knot vector from an existing sequence and degree.
    ///
    /// # Panics
    /// Panics if `knots.len() < degree + 2` (need at least `p+2` knots to have
    /// one non-empty span) or if `knots` is not non-decreasing.
    pub fn new(knots: Vec<f64>, degree: usize) -> Self {
        assert!(knots.len() >= degree + 2,
            "KnotVector: need at least {} knots for degree {}, got {}",
            degree + 2, degree, knots.len());
        for i in 1..knots.len() {
            assert!(knots[i] >= knots[i - 1],
                "KnotVector: knots must be non-decreasing; knots[{}]={} < knots[{}]={}",
                i, knots[i], i-1, knots[i-1]);
        }
        KnotVector { knots, degree }
    }

    /// Construct a **uniform clamped** knot vector on `[0, 1]` with `n_elems`
    /// elements (spans) of polynomial degree `p`.
    ///
    /// Knot structure: `[0]*p+1, 1/n, 2/n, ..., (n-1)/n, [1]*p+1`.
    /// Length: `n_elems + 2*p + 1`.
    pub fn uniform(degree: usize, n_elems: usize) -> Self {
        assert!(n_elems >= 1, "n_elems must be ≥ 1");
        let mut knots = Vec::new();
        // p+1 leading zeros
        knots.extend(std::iter::repeat_n(0.0, degree + 1));
        // interior knots
        for i in 1..n_elems {
            knots.push(i as f64 / n_elems as f64);
        }
        // p+1 trailing ones
        knots.extend(std::iter::repeat_n(1.0, degree + 1));
        KnotVector { knots, degree }
    }

    /// Number of basis functions: `n_knots - degree - 1`.
    pub fn n_basis(&self) -> usize {
        self.knots.len() - self.degree - 1
    }

    /// Number of non-empty spans (elements) in the knot vector.
    pub fn n_spans(&self) -> usize {
        self.knots.windows(2).filter(|w| w[1] > w[0]).count()
    }

    /// Find the knot span index $i$ such that $\Xi_i \leq \xi < \Xi_{i+1}$.
    ///
    /// At the right endpoint returns the last non-empty span.
    /// Uses binary search: O(log n).
    pub fn find_span(&self, xi: f64) -> usize {
        let n = self.n_basis() - 1; // highest basis index
        let p = self.degree;
        let knots = &self.knots;

        // Clamp to domain.
        if xi >= knots[n + 1] { return n; }
        if xi <= knots[p] { return p; }

        let mut lo = p;
        let mut hi = n + 1;
        let mut mid = (lo + hi) / 2;
        while xi < knots[mid] || xi >= knots[mid + 1] {
            if xi < knots[mid] {
                hi = mid;
            } else {
                lo = mid;
            }
            mid = (lo + hi) / 2;
        }
        mid
    }

    /// Evaluate all $p+1$ non-zero B-spline basis functions at `xi`.
    ///
    /// Returns `N[0..=p]` where `N[j] = N_{span-p+j, p}(xi)`.
    ///
    /// Uses the triangular de Boor scheme; $O(p^2)$.
    pub fn basis_funs(&self, span: usize, xi: f64) -> Vec<f64> {
        let p = self.degree;
        let knots = &self.knots;
        let mut n = vec![0.0_f64; p + 1];
        let mut left  = vec![0.0_f64; p + 1];
        let mut right = vec![0.0_f64; p + 1];

        n[0] = 1.0;
        for j in 1..=p {
            left[j]  = xi - knots[span + 1 - j];
            right[j] = knots[span + j] - xi;
            let mut saved = 0.0_f64;
            for r in 0..j {
                let denom = right[r + 1] + left[j - r];
                let temp = if denom.abs() < 1e-300 { 0.0 } else { n[r] / denom };
                n[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            n[j] = saved;
        }
        n
    }

    /// Evaluate B-spline basis values **and** their first derivatives at `xi`.
    ///
    /// Returns `(N, dN)` where each has length `p+1`.
    /// `dN[j] = dN_{span-p+j,p}/dxi`.
    ///
    /// Uses the triangular `ndu` table (Algorithm A2.3 from Piegl & Tiller,
    /// simplified to first derivatives only).
    pub fn basis_funs_and_ders(&self, span: usize, xi: f64) -> (Vec<f64>, Vec<f64>) {
        let p = self.degree;
        let knots = &self.knots;

        // Build the full triangular scheme (ndu table).
        // ndu[i][j]: for j < i it holds the knot difference; for j >= i it
        // holds the basis value N_{span-j+i, j} (column-major by degree).
        //
        // We use the standard layout from the NURBS Book:
        //   ndu[j][r] = N_{span-j+r, j}  (the r-th basis of degree j)
        // and ndu is sized (p+1) x (p+1).
        let mut ndu    = vec![vec![0.0_f64; p + 1]; p + 1];
        let mut left   = vec![0.0_f64; p + 1];
        let mut right  = vec![0.0_f64; p + 1];

        ndu[0][0] = 1.0;
        for j in 1..=p {
            left[j]  = xi - knots[span + 1 - j];
            right[j] = knots[span + j] - xi;
            let mut saved = 0.0_f64;
            for r in 0..j {
                // Store the denominator in the lower triangular part.
                ndu[j][r] = right[r + 1] + left[j - r];
                let temp = if ndu[j][r].abs() < 1e-300 { 0.0 } else { ndu[r][j - 1] / ndu[j][r] };
                ndu[r][j] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            ndu[j][j] = saved;
        }

        // Extract basis values N_{span-p+r, p} = ndu[r][p], r = 0..=p.
        let n_vals: Vec<f64> = (0..=p).map(|r| ndu[r][p]).collect();

        if p == 0 {
            return (n_vals, vec![0.0; p + 1]);
        }

        // Compute first derivatives using the two-row "a" working array.
        // Algorithm A2.3, k=1 (first derivative only).
        let mut dn = vec![0.0_f64; p + 1];

        // a[s][j]: working array, two rows.
        let mut a = vec![vec![0.0_f64; p + 1]; 2];
        for r in 0..=p {
            let mut s1 = 0usize;
            let mut s2 = 1usize;
            a[s1][0] = 1.0;

            // Compute 1st derivative contribution for basis function r.
            let rk = r as i64 - 1;
            let pk = p as i64 - 1;

            let j1: usize = if rk >= 0 { 1 } else { ((-rk) as usize).min(1) };
            let j2: usize = if (r as i64 - 1) <= pk {
                0
            } else {
                (r - 1) - p  // this is the "p - r" range lower bound
            };
            // Simpler: direct formula using ndu.
            // dN_{span-p+r,p}/dxi = p * (N_{span-p+r,p-1}/(Xi_{span+r}-Xi_{span-p+r})
            //                            - N_{span-p+r+1,p-1}/(Xi_{span+r+1}-Xi_{span-p+r+1}))
            // N_{span-p+r,p-1} = ndu[r][p-1]  (r=0..p-1)
            // N_{span-p+r+1,p-1} = ndu[r+1][p-1]  (r=0..p-1; for r=p, this is 0)
            let _ = (j1, j2, s2, rk, pk, &mut a); // suppress unused warnings

            let i = span as i64 - p as i64 + r as i64;
            let n_ip_m1  = if r > 0 { ndu[r - 1][p - 1] } else { 0.0 };
            let n_ip1_m1 = if r < p { ndu[r][p - 1] } else { 0.0 };
            let d1 = if r > 0 {
                let denom = ndu[p][r - 1]; // stored denominator = Xi_{span+r} - Xi_{span-p+r}
                if denom.abs() > 1e-300 { n_ip_m1 / denom } else { 0.0 }
            } else { 0.0 };
            let d2 = if r < p {
                let denom = ndu[p][r]; // stored denominator = Xi_{span+r+1} - Xi_{span-p+r+1}
                if denom.abs() > 1e-300 { n_ip1_m1 / denom } else { 0.0 }
            } else { 0.0 };
            let _ = i;
            dn[r] = p as f64 * (d1 - d2);

            // Reset a for next r iteration.
            a[s1][0] = 0.0;
            s1 = 1 - s1; s2 = 1 - s2;
            let _ = (s1, s2);
        }

        (n_vals, dn)
    }

    /// Evaluate B-spline basis values, first derivatives, **and** second derivatives.
    ///
    /// Returns `(N, dN, ddN)` where each has length `p+1`.
    ///
    /// For degree ≤ 1, the second derivatives are identically zero.
    /// For degree ≥ 2, uses centered finite differences on the basis values
    /// (second-order accurate, O(eps²) error).  The evaluation point is clamped
    /// to stay within the current knot span for robustness near span boundaries.
    pub fn basis_funs_and_ders2(&self, span: usize, xi: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let p = self.degree;
        let (n, dn) = self.basis_funs_and_ders(span, xi);

        // Degree ≤ 1: piecewise linear/constant → zero second derivative
        if p <= 1 {
            return (n, dn, vec![0.0; p + 1]);
        }

        let knots = &self.knots;

        // Clamp the evaluation window so we stay within [knots[span], knots[span+1]]
        let xi0 = knots[span];
        let xi1 = knots[span + 1];
        let h = (xi1 - xi0).max(1e-14);
        let eps = 1e-6_f64.min(h * 0.25);

        let xi_p = (xi + eps).min(xi1 - 1e-14);
        let xi_m = (xi - eps).max(xi0 + 1e-14);

        let (np, _) = self.basis_funs_and_ders(span, xi_p);
        let (nm, _) = self.basis_funs_and_ders(span, xi_m);

        let inv_eps2 = 1.0 / (eps * eps);
        let ddn: Vec<f64> = (0..=p).map(|j| (np[j] - 2.0 * n[j] + nm[j]) * inv_eps2).collect();
        (n, dn, ddn)
    }
}

// ─── BSplineBasis1D ───────────────────────────────────────────────────────────

/// 1-D B-spline basis on a single knot vector.
///
/// Provides evaluation of all basis functions and their derivatives at a
/// given parametric coordinate.
#[derive(Debug, Clone)]
pub struct BSplineBasis1D {
    /// The underlying knot vector.
    pub kv: KnotVector,
}

impl BSplineBasis1D {
    pub fn new(kv: KnotVector) -> Self { BSplineBasis1D { kv } }

    /// Number of basis functions.
    pub fn n_basis(&self) -> usize { self.kv.n_basis() }

    /// Evaluate all basis functions at `xi`.
    ///
    /// Returns a vector of length `n_basis()` where only `p+1` entries are
    /// non-zero (those supported on the knot span containing `xi`).
    pub fn eval(&self, xi: f64) -> Vec<f64> {
        let n = self.n_basis();
        let span = self.kv.find_span(xi);
        let local = self.kv.basis_funs(span, xi);
        let p = self.kv.degree;
        let mut vals = vec![0.0_f64; n];
        for j in 0..=p {
            vals[span - p + j] = local[j];
        }
        vals
    }

    /// Evaluate all basis functions and their first derivatives at `xi`.
    ///
    /// Returns `(values, derivatives)`, each of length `n_basis()`.
    pub fn eval_with_ders(&self, xi: f64) -> (Vec<f64>, Vec<f64>) {
        let n = self.n_basis();
        let span = self.kv.find_span(xi);
        let (local_n, local_dn) = self.kv.basis_funs_and_ders(span, xi);
        let p = self.kv.degree;
        let mut vals = vec![0.0_f64; n];
        let mut ders = vec![0.0_f64; n];
        for j in 0..=p {
            vals[span - p + j] = local_n[j];
            ders[span - p + j] = local_dn[j];
        }
        (vals, ders)
    }

    /// Evaluate all basis functions, first derivatives, **and** second derivatives at `xi`.
    ///
    /// Returns `(values, derivatives, second_derivatives)`, each of length `n_basis()`.
    pub fn eval_with_ders2(&self, xi: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = self.n_basis();
        let span = self.kv.find_span(xi);
        let (local_n, local_dn, local_ddn) = self.kv.basis_funs_and_ders2(span, xi);
        let p = self.kv.degree;
        let mut vals = vec![0.0_f64; n];
        let mut ders = vec![0.0_f64; n];
        let mut dders = vec![0.0_f64; n];
        for j in 0..=p {
            vals[span - p + j] = local_n[j];
            ders[span - p + j] = local_dn[j];
            dders[span - p + j] = local_ddn[j];
        }
        (vals, ders, dders)
    }
}

// ─── NurbsPatch2D ─────────────────────────────────────────────────────────────

/// A 2-D NURBS patch element implementing [`ReferenceElement`].
///
/// The reference domain is $[0,1]^2$ (parameterised by $(u, v)$).
/// The `n_u × n_v` control points and their weights define the rational
/// B-spline map.
///
/// DOF ordering: lexicographic $(i, j)$ where $i$ is the $u$-index (fast)
/// and $j$ is the $v$-index (slow): DOF index = `j * n_u + i`.
#[derive(Debug, Clone)]
pub struct NurbsPatch2D {
    /// B-spline basis in the $u$ direction.
    pub basis_u: BSplineBasis1D,
    /// B-spline basis in the $v$ direction.
    pub basis_v: BSplineBasis1D,
    /// NURBS weights $w_{ij}$ in DOF order (length `n_u * n_v`).
    pub weights: Vec<f64>,
}

impl NurbsPatch2D {
    /// Create a new 2-D NURBS patch.
    ///
    /// # Arguments
    /// * `kv_u`, `kv_v` — knot vectors for the two parametric directions.
    /// * `weights`       — rational weights, length `n_u * n_v` (DOF order).
    ///
    /// # Panics
    /// Panics if `weights.len() != kv_u.n_basis() * kv_v.n_basis()`.
    pub fn new(kv_u: KnotVector, kv_v: KnotVector, weights: Vec<f64>) -> Self {
        let n_u = kv_u.n_basis();
        let n_v = kv_v.n_basis();
        assert_eq!(weights.len(), n_u * n_v,
            "NurbsPatch2D: weights.len()={} != n_u*n_v={}",
            weights.len(), n_u * n_v);
        for &w in &weights {
            assert!(w > 0.0, "NURBS weights must be positive");
        }
        NurbsPatch2D {
            basis_u: BSplineBasis1D::new(kv_u),
            basis_v: BSplineBasis1D::new(kv_v),
            weights,
        }
    }

    /// Create a uniform B-spline (all weights = 1) patch.
    pub fn uniform(kv_u: KnotVector, kv_v: KnotVector) -> Self {
        let n = kv_u.n_basis() * kv_v.n_basis();
        Self::new(kv_u, kv_v, vec![1.0; n])
    }

    /// Number of DOFs in $u$.
    pub fn n_u(&self) -> usize { self.basis_u.n_basis() }
    /// Number of DOFs in $v$.
    pub fn n_v(&self) -> usize { self.basis_v.n_basis() }

    /// Evaluate NURBS Hessians (second derivatives) at `xi = [u, v]` in
    /// **parametric** coordinates.
    ///
    /// `hessians` must have length `n_dofs * 4`.  Layout per DOF `a`:
    /// - `hessians[a*4 + 0]` = d²R/du²
    /// - `hessians[a*4 + 1]` = d²R/dudv
    /// - `hessians[a*4 + 2]` = d²R/dvdu  (symmetric, same as dudv)
    /// - `hessians[a*4 + 3]` = d²R/dv²
    ///
    /// Uses the quotient rule applied twice (see Phase 2.6 notes).
    /// The underlying B-spline second derivatives are computed via
    /// centered finite differences on the basis values.
    pub fn eval_hessian_basis(&self, xi: &[f64], hessians: &mut [f64]) {
        let (u, v) = (xi[0], xi[1]);
        let n_u = self.n_u();
        let n_v = self.n_v();
        let n_dofs = n_u * n_v;

        let (nu, dnu, ddnu) = self.basis_u.eval_with_ders2(u);
        let (nv, dnv, ddnv) = self.basis_v.eval_with_ders2(v);

        // B-spline tensor-product values and parametric derivatives
        let mut b = vec![0.0_f64; n_dofs];
        let mut db_du = vec![0.0_f64; n_dofs];
        let mut db_dv = vec![0.0_f64; n_dofs];
        let mut d2b_du2 = vec![0.0_f64; n_dofs];
        let mut d2b_dv2 = vec![0.0_f64; n_dofs];
        let mut d2b_dudv = vec![0.0_f64; n_dofs];

        for j in 0..n_v {
            for i in 0..n_u {
                let dof = j * n_u + i;
                b[dof] = nu[i] * nv[j];
                db_du[dof] = dnu[i] * nv[j];
                db_dv[dof] = nu[i] * dnv[j];
                d2b_du2[dof] = ddnu[i] * nv[j];
                d2b_dv2[dof] = nu[i] * ddnv[j];
                d2b_dudv[dof] = dnu[i] * dnv[j];
            }
        }

        // Denominator W = Σ w_k B_k and its parametric derivatives
        let w = &self.weights;
        let mut W = 0.0_f64;
        let mut dW_du = 0.0_f64;
        let mut dW_dv = 0.0_f64;
        let mut d2W_du2 = 0.0_f64;
        let mut d2W_dv2 = 0.0_f64;
        let mut d2W_dudv = 0.0_f64;

        for a in 0..n_dofs {
            let wa = w[a];
            W += wa * b[a];
            dW_du += wa * db_du[a];
            dW_dv += wa * db_dv[a];
            d2W_du2 += wa * d2b_du2[a];
            d2W_dv2 += wa * d2b_dv2[a];
            d2W_dudv += wa * d2b_dudv[a];
        }

        assert!(W.abs() > 1e-300, "NURBS denominator near zero at ({u},{v})");
        let inv_W = 1.0 / W;
        let inv_W2 = inv_W * inv_W;

        // NURBS rational quotient rule for values, first and second derivatives
        for a in 0..n_dofs {
            let wa = w[a];
            let n_val = b[a];
            let dn_du = db_du[a];
            let dn_dv = db_dv[a];
            let d2n_du2 = d2b_du2[a];
            let d2n_dv2 = d2b_dv2[a];
            let d2n_dudv = d2b_dudv[a];

            // First derivatives (same as eval_grad_basis)
            let dr_du = (wa * dn_du * W - wa * n_val * dW_du) * inv_W2;
            let dr_dv = (wa * dn_dv * W - wa * n_val * dW_dv) * inv_W2;

            // Second derivatives via repeated quotient rule:
            //   R'' = [w*N''*W - w*N*W''] / W²  -  2*W'*R' / W
            let d2r_du2 =
                (wa * d2n_du2 * W - wa * n_val * d2W_du2) * inv_W2 - 2.0 * dW_du * dr_du * inv_W;
            let d2r_dv2 =
                (wa * d2n_dv2 * W - wa * n_val * d2W_dv2) * inv_W2 - 2.0 * dW_dv * dr_dv * inv_W;
            let d2r_dudv = (wa * d2n_dudv * W - wa * n_val * d2W_dudv) * inv_W2
                - (dW_du * dr_dv + dW_dv * dr_du) * inv_W;

            hessians[a * 4] = d2r_du2;
            hessians[a * 4 + 1] = d2r_dudv;
            hessians[a * 4 + 2] = d2r_dudv; // symmetric
            hessians[a * 4 + 3] = d2r_dv2;
        }
    }
}

impl ReferenceElement for NurbsPatch2D {
    fn dim(&self) -> u8 { 2 }

    fn order(&self) -> u8 {
        self.basis_u.kv.degree.max(self.basis_v.kv.degree) as u8
    }

    fn n_dofs(&self) -> usize { self.n_u() * self.n_v() }

    /// Evaluate all NURBS basis functions $R_{ij}(u,v)$ at reference point `xi = [u, v]`.
    ///
    /// `values` must have length `n_u * n_v`.
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (u, v) = (xi[0], xi[1]);
        let n_u = self.n_u();
        let n_v = self.n_v();
        let n = n_u * n_v;

        let nu = self.basis_u.eval(u);
        let nv = self.basis_v.eval(v);

        // Tensor-product B-spline values and weighted sum (denominator).
        let mut w_sum = 0.0_f64;
        for j in 0..n_v {
            for i in 0..n_u {
                let dof = j * n_u + i;
                let b = nu[i] * nv[j];
                values[dof] = b * self.weights[dof];
                w_sum += values[dof];
            }
        }

        // Normalise by the denominator.
        if w_sum.abs() > 1e-300 {
            let inv_w = 1.0 / w_sum;
            for v in values[..n].iter_mut() { *v *= inv_w; }
        }
    }

    /// Evaluate gradients $\nabla R_{ij}(u,v)$ at `xi = [u, v]`.
    ///
    /// `grads` must have length `n_dofs * 2`.  Layout: `grads[dof*2] = dR/du`,
    /// `grads[dof*2+1] = dR/dv`.
    ///
    /// Uses the quotient rule:
    /// $\nabla R_A = \frac{w_A (\nabla B_A) W - w_A B_A \nabla W}{W^2}$
    /// where $W = \sum_k w_k B_k$.
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (u, v) = (xi[0], xi[1]);
        let n_u = self.n_u();
        let n_v = self.n_v();

        let (nu,  dnu)  = self.basis_u.eval_with_ders(u);
        let (nv,  dnv)  = self.basis_v.eval_with_ders(v);

        // Compute tensor-product B-splines, their weighted values, and the
        // denominator W and its gradient ∇W.
        let mut b   = vec![0.0_f64; n_u * n_v]; // B_ij = N_i * M_j
        let mut db_du = vec![0.0_f64; n_u * n_v]; // dB/du = dN_i/du * M_j
        let mut db_dv = vec![0.0_f64; n_u * n_v]; // dB/dv = N_i * dM_j/dv

        let mut w_sum    = 0.0_f64;
        let mut dw_du = 0.0_f64;
        let mut dw_dv = 0.0_f64;

        for j in 0..n_v {
            for i in 0..n_u {
                let dof = j * n_u + i;
                let w = self.weights[dof];
                b[dof]    = nu[i] * nv[j];
                db_du[dof] = dnu[i] * nv[j];
                db_dv[dof] = nu[i]  * dnv[j];
                w_sum    += w * b[dof];
                dw_du += w * db_du[dof];
                dw_dv += w * db_dv[dof];
            }
        }

        let w2 = w_sum * w_sum;
        let inv_w2 = if w2 > 1e-300 { 1.0 / w2 } else { 0.0 };

        for j in 0..n_v {
            for i in 0..n_u {
                let dof = j * n_u + i;
                let w = self.weights[dof];
                let w_b = w * b[dof];
                // dR/du = (w * dB/du * W - w*B * dW/du) / W²
                grads[dof * 2]     = (w * db_du[dof] * w_sum - w_b * dw_du) * inv_w2;
                // dR/dv = (w * dB/dv * W - w*B * dW/dv) / W²
                grads[dof * 2 + 1] = (w * db_dv[dof] * w_sum - w_b * dw_dv) * inv_w2;
            }
        }
    }

    /// Gauss-Legendre tensor-product quadrature rule on $[0,1]^2$.
    fn quadrature(&self, order: u8) -> QuadratureRule { quad_rule(order) }

    /// Reference-domain DOF coordinates.
    ///
    /// The Greville abscissae $\bar{\xi}_i = (\xi_{i+1} + \ldots + \xi_{i+p}) / p$
    /// are the canonical DOF coordinates for B-splines.
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let n_u = self.n_u();
        let n_v = self.n_v();
        let gu = greville_abscissae(&self.basis_u.kv);
        let gv = greville_abscissae(&self.basis_v.kv);
        let mut coords = Vec::with_capacity(n_u * n_v);
        for j in 0..n_v {
            for i in 0..n_u {
                coords.push(vec![gu[i], gv[j]]);
            }
        }
        coords
    }
}

// ─── NurbsPatch3D ─────────────────────────────────────────────────────────────

/// A 3-D NURBS patch implementing [`ReferenceElement`].
///
/// DOF ordering: lexicographic $(i, j, k)$ — `i` fast, `k` slow:
/// DOF index = `k * n_u * n_v + j * n_u + i`.
#[derive(Debug, Clone)]
pub struct NurbsPatch3D {
    pub basis_u: BSplineBasis1D,
    pub basis_v: BSplineBasis1D,
    pub basis_w: BSplineBasis1D,
    pub weights: Vec<f64>,
}

impl NurbsPatch3D {
    pub fn new(
        kv_u: KnotVector,
        kv_v: KnotVector,
        kv_w: KnotVector,
        weights: Vec<f64>,
    ) -> Self {
        let n = kv_u.n_basis() * kv_v.n_basis() * kv_w.n_basis();
        assert_eq!(weights.len(), n,
            "NurbsPatch3D: weights.len()={} != n_u*n_v*n_w={}", weights.len(), n);
        for &w in &weights {
            assert!(w > 0.0, "NURBS weights must be positive");
        }
        NurbsPatch3D {
            basis_u: BSplineBasis1D::new(kv_u),
            basis_v: BSplineBasis1D::new(kv_v),
            basis_w: BSplineBasis1D::new(kv_w),
            weights,
        }
    }

    pub fn uniform(kv_u: KnotVector, kv_v: KnotVector, kv_w: KnotVector) -> Self {
        let n = kv_u.n_basis() * kv_v.n_basis() * kv_w.n_basis();
        Self::new(kv_u, kv_v, kv_w, vec![1.0; n])
    }

    pub fn n_u(&self) -> usize { self.basis_u.n_basis() }
    pub fn n_v(&self) -> usize { self.basis_v.n_basis() }
    pub fn n_w(&self) -> usize { self.basis_w.n_basis() }
}

impl ReferenceElement for NurbsPatch3D {
    fn dim(&self) -> u8 { 3 }

    fn order(&self) -> u8 {
        [self.basis_u.kv.degree, self.basis_v.kv.degree, self.basis_w.kv.degree]
            .into_iter().max().unwrap() as u8
    }

    fn n_dofs(&self) -> usize { self.n_u() * self.n_v() * self.n_w() }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (u, v, w) = (xi[0], xi[1], xi[2]);
        let n_u = self.n_u();
        let n_v = self.n_v();
        let n_w = self.n_w();
        let n = n_u * n_v * n_w;

        let nu = self.basis_u.eval(u);
        let nv = self.basis_v.eval(v);
        let nw = self.basis_w.eval(w);

        let mut w_sum = 0.0_f64;
        for k in 0..n_w {
            for j in 0..n_v {
                for i in 0..n_u {
                    let dof = k * n_u * n_v + j * n_u + i;
                    let b = nu[i] * nv[j] * nw[k];
                    values[dof] = b * self.weights[dof];
                    w_sum += values[dof];
                }
            }
        }
        if w_sum.abs() > 1e-300 {
            let inv_w = 1.0 / w_sum;
            for v in values[..n].iter_mut() { *v *= inv_w; }
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (u, v, w) = (xi[0], xi[1], xi[2]);
        let n_u = self.n_u();
        let n_v = self.n_v();
        let n_w = self.n_w();

        let (nu,  dnu)  = self.basis_u.eval_with_ders(u);
        let (nv,  dnv)  = self.basis_v.eval_with_ders(v);
        let (nw,  dnw)  = self.basis_w.eval_with_ders(w);

        let n_dofs = n_u * n_v * n_w;
        let mut b     = vec![0.0_f64; n_dofs];
        let mut db_du = vec![0.0_f64; n_dofs];
        let mut db_dv = vec![0.0_f64; n_dofs];
        let mut db_dw = vec![0.0_f64; n_dofs];

        let mut w_sum = 0.0; let mut dw_du = 0.0; let mut dw_dv = 0.0; let mut dw_dw = 0.0;

        for k in 0..n_w {
            for j in 0..n_v {
                for i in 0..n_u {
                    let dof = k * n_u * n_v + j * n_u + i;
                    let wt  = self.weights[dof];
                    b[dof]    = nu[i] * nv[j] * nw[k];
                    db_du[dof] = dnu[i] * nv[j] * nw[k];
                    db_dv[dof] = nu[i]  * dnv[j] * nw[k];
                    db_dw[dof] = nu[i]  * nv[j]  * dnw[k];
                    w_sum += wt * b[dof];
                    dw_du += wt * db_du[dof];
                    dw_dv += wt * db_dv[dof];
                    dw_dw += wt * db_dw[dof];
                }
            }
        }

        let w2 = w_sum * w_sum;
        let inv_w2 = if w2 > 1e-300 { 1.0 / w2 } else { 0.0 };

        for k in 0..n_w {
            for j in 0..n_v {
                for i in 0..n_u {
                    let dof = k * n_u * n_v + j * n_u + i;
                    let wt  = self.weights[dof];
                    let w_b = wt * b[dof];
                    grads[dof * 3]     = (wt * db_du[dof] * w_sum - w_b * dw_du) * inv_w2;
                    grads[dof * 3 + 1] = (wt * db_dv[dof] * w_sum - w_b * dw_dv) * inv_w2;
                    grads[dof * 3 + 2] = (wt * db_dw[dof] * w_sum - w_b * dw_dw) * inv_w2;
                }
            }
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { hex_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let n_u = self.n_u();
        let n_v = self.n_v();
        let n_w = self.n_w();
        let gu = greville_abscissae(&self.basis_u.kv);
        let gv = greville_abscissae(&self.basis_v.kv);
        let gw = greville_abscissae(&self.basis_w.kv);
        let mut coords = Vec::with_capacity(n_u * n_v * n_w);
        for k in 0..n_w {
            for j in 0..n_v {
                for i in 0..n_u {
                    coords.push(vec![gu[i], gv[j], gw[k]]);
                }
            }
        }
        coords
    }
}

// ─── NurbsMesh ────────────────────────────────────────────────────────────────

/// A multi-patch NURBS mesh.
///
/// Stores the IGA control mesh as a collection of 2-D or 3-D patches.
/// Each patch carries control-point coordinates, weights, and its own
/// knot vectors.  Inter-patch connectivity (shared boundaries) is stored
/// as a list of matched face/edge pairs.
///
/// # Usage
///
/// ```rust,ignore
/// use fem_element::nurbs::{KnotVector, NurbsMesh2D};
///
/// // Single-patch square on [0,1]^2 with Q1 (bilinear = degree-1 B-spline)
/// let kv = KnotVector::uniform(1, 1);
/// let mut mesh = NurbsMesh2D::single_patch(kv.clone(), kv.clone(),
///     vec![[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]], vec![1.0;4]);
/// ```
#[derive(Debug, Clone)]
pub struct NurbsMesh2D {
    /// One entry per patch.
    pub patches: Vec<NurbsPatch2DData>,
    /// Inter-patch matched edge pairs: `(patch_a, edge_a, patch_b, edge_b)`.
    pub edge_connectivity: Vec<(usize, usize, usize, usize)>,
}

/// Geometric data for one 2-D NURBS patch.
#[derive(Debug, Clone)]
pub struct NurbsPatch2DData {
    /// Knot vector in $u$ direction.
    pub kv_u: KnotVector,
    /// Knot vector in $v$ direction.
    pub kv_v: KnotVector,
    /// Control-point coordinates in DOF order: `control_pts[dof] = [x, y]`.
    pub control_pts: Vec<[f64; 2]>,
    /// Rational weights in DOF order.
    pub weights: Vec<f64>,
    /// Physical tag / material ID.
    pub tag: i32,
}

impl NurbsMesh2D {
    /// Build a single-patch mesh from control point data.
    pub fn single_patch(
        kv_u: KnotVector,
        kv_v: KnotVector,
        control_pts: Vec<[f64; 2]>,
        weights: Vec<f64>,
    ) -> Self {
        NurbsMesh2D {
            patches: vec![NurbsPatch2DData { kv_u, kv_v, control_pts, weights, tag: 1 }],
            edge_connectivity: Vec::new(),
        }
    }

    /// Number of patches.
    pub fn n_patches(&self) -> usize { self.patches.len() }

    /// Total number of control points (DOFs) across all patches.
    /// Note: shared boundary DOFs are counted once per patch (no deduplication).
    pub fn n_control_pts_total(&self) -> usize {
        self.patches.iter().map(|p| p.control_pts.len()).sum()
    }

    /// Get the `NurbsPatch2D` reference element for patch `i`.
    pub fn patch_element(&self, patch_idx: usize) -> NurbsPatch2D {
        let pd = &self.patches[patch_idx];
        NurbsPatch2D::new(pd.kv_u.clone(), pd.kv_v.clone(), pd.weights.clone())
    }
}

/// Geometric data for one 3-D NURBS patch.
#[derive(Debug, Clone)]
pub struct NurbsPatch3DData {
    pub kv_u: KnotVector,
    pub kv_v: KnotVector,
    pub kv_w: KnotVector,
    pub control_pts: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub tag: i32,
}

/// A multi-patch NURBS mesh in 3-D.
#[derive(Debug, Clone)]
pub struct NurbsMesh3D {
    pub patches: Vec<NurbsPatch3DData>,
    pub face_connectivity: Vec<(usize, usize, usize, usize)>,
}

impl NurbsMesh3D {
    pub fn single_patch(
        kv_u: KnotVector,
        kv_v: KnotVector,
        kv_w: KnotVector,
        control_pts: Vec<[f64; 3]>,
        weights: Vec<f64>,
    ) -> Self {
        NurbsMesh3D {
            patches: vec![NurbsPatch3DData { kv_u, kv_v, kv_w, control_pts, weights, tag: 1 }],
            face_connectivity: Vec::new(),
        }
    }

    pub fn n_patches(&self) -> usize { self.patches.len() }

    pub fn patch_element(&self, patch_idx: usize) -> NurbsPatch3D {
        let pd = &self.patches[patch_idx];
        NurbsPatch3D::new(
            pd.kv_u.clone(), pd.kv_v.clone(), pd.kv_w.clone(),
            pd.weights.clone(),
        )
    }
}

impl NurbsPatch3DData {
    /// Get the `NurbsPatch3D` reference element for this patch data.
    pub fn patch_element_ref(&self) -> NurbsPatch3D {
        NurbsPatch3D::new(
            self.kv_u.clone(), self.kv_v.clone(), self.kv_w.clone(),
            self.weights.clone(),
        )
    }
}

// ─── Helper: Greville abscissae ──────────────────────────────────────────────

/// Compute Greville abscissae for a knot vector.
///
/// $\bar{\xi}_i = \frac{1}{p} \sum_{k=1}^{p} \Xi_{i+k}$, for $i = 0, \ldots, n-1$.
pub fn greville_abscissae(kv: &KnotVector) -> Vec<f64> {
    let n = kv.n_basis();
    let p = kv.degree;
    let knots = &kv.knots;
    (0..n).map(|i| {
        if p == 0 {
            // Midpoint of the span.
            0.5 * (knots[i] + knots[i + 1])
        } else {
            let sum: f64 = (1..=p).map(|k| knots[i + k]).sum();
            sum / p as f64
        }
    }).collect()
}

// ─── Degree elevation ────────────────────────────────────────────────────────

fn binom(n: usize, k: usize) -> f64 {
    if k > n { return 0.0; }
    let k = k.min(n - k);
    if k == 0 { return 1.0; }
    let mut r = 1.0;
    for i in 1..=k { r = r * (n - k + i) as f64 / i as f64; }
    r
}

fn bezalfs(p: usize, t: usize) -> Vec<Vec<f64>> {
    let ph = p + t;
    let mut a = vec![vec![0.0; p + 1]; ph + 1];
    a[0][0] = 1.0;
    a[ph][p] = 1.0;
    for i in 1..ph {
        let d = binom(ph, i);
        if d.abs() <= 1e-300 { continue; }
        for j in (if i >= t { i - t } else { 0 })..=p.min(i) {
            a[i][j] = binom(p, j) * binom(t, i - j) / d;
        }
    }
    for i in 1..ph {
        for j in 0..p {
            a[ph - i][p - j] = a[i][j];
        }
    }
    a
}

#[allow(unused_assignments, non_snake_case, dead_code)]
pub(crate) fn elevate_curve_1d(knots: &[f64], ctrl: &[f64], p: usize, t: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(t > 0);
    let n = ctrl.len() - 1;
    let m = knots.len() - 1;
    let ph = p + t;
    assert_eq!(m, n + p + 1);
    let bz = bezalfs(p, t);
    let ne = n - p + 1;
    let nb = n + t * ne;
    let mb = nb + ph + 1;
    let mut U = vec![0.0; mb + 1];
    let mut Q = vec![0.0; nb + 1];
    for j in 0..=ph { U[j] = knots[0]; }
    Q[0] = ctrl[0];
    let mut bp = vec![0.0; p + 1];
    let mut eb = vec![0.0; ph + 1];
    let mut nx = vec![0.0; p.saturating_sub(1).max(1)];
    let mut al = vec![0.0; p.saturating_sub(1).max(1)];
    for j in 0..=p { bp[j] = ctrl[j]; }
    let mut a = p as isize;
    let mut b = (p + 1) as isize;
    let mut kd = (ph + 1) as isize;
    let mut ci = 1_isize;
    let mut or = -1_isize;
    let mut r = -1_isize;
    let mut ua = knots[0];
    while b < m as isize {
        let i0 = b;
        while b < m as isize && (knots[b as usize] - knots[(b + 1) as usize]).abs() < 1e-14 { b += 1; }
        let mul = (b - i0 + 1) as usize;
        let ub = knots[b as usize];
        or = r;
        r = p as isize - mul as isize;
        let ru = if r > 0 { r as usize } else { 0 };
        let lbz = if or > 0 { ((or + 2) / 2) as usize } else { 1 };
        let rbz = if r > 0 { ph - (r as usize + 1) / 2 } else { ph };
        if r > 0 {
            let nu = ub - ua;
            for k in (mul + 1..=p).rev() {
                al[k - mul - 1] = nu / (knots[(a + k as isize) as usize] - ua);
            }
            for j in 1..=ru {
                let s = mul + j;
                for k in (s..=p).rev() {
                    bp[k] = al[k - s] * bp[k] + (1.0 - al[k - s]) * bp[k - 1];
                }
                nx[ru - j] = bp[p];
            }
        }
        for i in lbz..=ph {
            let mut s = 0.0;
            for j in (if i >= t { i - t } else { 0 })..=p.min(i) {
                s += bz[i][j] * bp[j];
            }
            eb[i] = s;
        }
        if or > 1 {
            let mut fi = kd - 2;
            let mut la = kd;
            let dn = ub - ua;
            let _bet = (ub - U[(kd - 1) as usize]) / dn;
            for tr in 1..or as usize {
                let mut ib = fi;
                let mut jb = la;
                let mut kj = la - kd + 1;
                while jb - ib > tr as isize {
                    if ib < ci {
                        let af = (ub - U[ib as usize]) / (ua - U[ib as usize]);
                        let iu = ib as usize;
                        Q[iu] = af * Q[iu] - (1.0 - af) * Q[if ib > 0 { (ib - 1) as usize } else { 0 }];
                    }
                    if jb >= lbz as isize {
                        let jtr = jb - tr as isize;
                        let kju = kj as usize;
                        let gm = (ub - U[jtr as usize]) / dn;
                        eb[kju] = gm * eb[kju] + (1.0 - gm) * eb[(kj + 1) as usize];
                    }
                    ib += 1;
                    jb -= 1;
                    kj -= 1;
                }
                fi -= 1;
                la += 1;
            }
        }
        if a != p as isize {
            for _ in 0..if or > 0 { ph.saturating_sub(or as usize) } else { ph } {
                U[kd as usize] = ua;
                kd += 1;
            }
        }
        for j in lbz..=rbz {
            if (ci as usize) <= nb {
                Q[ci as usize] = eb[j];
                ci += 1;
            }
        }
        if b < m as isize {
            for j in 0..ru { bp[j] = nx[j]; }
            for j in ru..=p {
                bp[j] = ctrl[(b - p as isize + j as isize) as usize];
            }
            a = b;
            b += 1;
            ua = ub;
        } else {
            for i in 0..=ph {
                U[(kd + i as isize) as usize] = knots[m];
            }
        }
    }
    (U, Q)
}

#[allow(non_snake_case)]
pub fn elevate_degree_curve_2d(kv: &KnotVector, ctrl: &[[f64; 2]], w: &[f64], t: usize) -> (KnotVector, Vec<[f64; 2]>, Vec<f64>) {
    let p = kv.degree; let n = ctrl.len();
    let (U, x) = elevate_curve_1d(&kv.knots, &(0..n).map(|i| ctrl[i][0] * w[i]).collect::<Vec<_>>(), p, t);
    let (_, y) = elevate_curve_1d(&kv.knots, &(0..n).map(|i| ctrl[i][1] * w[i]).collect::<Vec<_>>(), p, t);
    let (_, ww) = elevate_curve_1d(&kv.knots, &w.to_vec(), p, t);
    let nn = x.len();
    (KnotVector::new(U, p + t), (0..nn).map(|i| { let w = ww[i]; if w.abs() > 1e-300 { [x[i] / w, y[i] / w] } else { [0.0, 0.0] } }).collect(), ww)
}

#[allow(non_snake_case)]
pub fn elevate_curve_3d(kv: &KnotVector, ctrl: &[[f64; 3]], w: &[f64], t: usize) -> (KnotVector, Vec<[f64; 3]>, Vec<f64>) {
    let p = kv.degree; let n = ctrl.len();
    let (U, x) = elevate_curve_1d(&kv.knots, &(0..n).map(|i| ctrl[i][0] * w[i]).collect::<Vec<_>>(), p, t);
    let (_, y) = elevate_curve_1d(&kv.knots, &(0..n).map(|i| ctrl[i][1] * w[i]).collect::<Vec<_>>(), p, t);
    let (_, z) = elevate_curve_1d(&kv.knots, &(0..n).map(|i| ctrl[i][2] * w[i]).collect::<Vec<_>>(), p, t);
    let (_, ww) = elevate_curve_1d(&kv.knots, &w.to_vec(), p, t);
    let nn = x.len();
    (KnotVector::new(U, p + t), (0..nn).map(|i| { let w = ww[i]; if w.abs() > 1e-300 { [x[i] / w, y[i] / w, z[i] / w] } else { [0.0, 0.0, 0.0] } }).collect(), ww)
}

pub fn elevate_u_2d(pd: &NurbsPatch2DData, t: usize) -> NurbsPatch2DData {
    if t == 0 { return pd.clone(); }
    let nu = pd.kv_u.n_basis(); let nv = pd.kv_v.n_basis(); let p = pd.kv_u.degree;
    let n = nu * nv; let mut xw = vec![0.0; n]; let mut yw = vec![0.0; n]; let mut ww = vec![0.0; n];
    for j in 0..nv { for i in 0..nu { let idx = j * nu + i; xw[idx] = pd.control_pts[idx][0] * pd.weights[idx]; yw[idx] = pd.control_pts[idx][1] * pd.weights[idx]; ww[idx] = pd.weights[idx]; } }
    let mut nx = Vec::new(); let mut ny = Vec::new(); let mut nw = Vec::new(); let mut nkv = None;
    for j in 0..nv {
        let (U, rx) = elevate_curve_1d(&pd.kv_u.knots, &(0..nu).map(|i| xw[j * nu + i]).collect::<Vec<_>>(), p, t);
        let (_, ry) = elevate_curve_1d(&pd.kv_u.knots, &(0..nu).map(|i| yw[j * nu + i]).collect::<Vec<_>>(), p, t);
        let (_, rw) = elevate_curve_1d(&pd.kv_u.knots, &(0..nu).map(|i| ww[j * nu + i]).collect::<Vec<_>>(), p, t);
        if nkv.is_none() { nkv = Some(KnotVector::new(U, p + t)); }
        nx.extend(rx); ny.extend(ry); nw.extend(rw);
    }
    let nkv = nkv.unwrap(); let nnu = nkv.n_basis();
    NurbsPatch2DData { kv_u: nkv, kv_v: pd.kv_v.clone(), control_pts: (0..nnu * nv).map(|i| { let w = nw[i]; if w.abs() > 1e-300 { [nx[i] / w, ny[i] / w] } else { [0.0, 0.0] } }).collect(), weights: nw, tag: pd.tag }
}

pub fn elevate_v_2d(pd: &NurbsPatch2DData, t: usize) -> NurbsPatch2DData { /* same pattern for v-direction */
    if t == 0 { return pd.clone(); }
    let nu = pd.kv_u.n_basis(); let nv = pd.kv_v.n_basis(); let q = pd.kv_v.degree;
    let n = nu * nv; let mut xw = vec![0.0; n]; let mut yw = vec![0.0; n]; let mut ww = vec![0.0; n];
    for idx in 0..n { xw[idx] = pd.control_pts[idx][0] * pd.weights[idx]; yw[idx] = pd.control_pts[idx][1] * pd.weights[idx]; ww[idx] = pd.weights[idx]; }
    let mut nx = Vec::new(); let mut ny = Vec::new(); let mut nw = Vec::new(); let mut nkv = None;
    for i in 0..nu {
        let (V, rx) = elevate_curve_1d(&pd.kv_v.knots, &(0..nv).map(|j| xw[j * nu + i]).collect::<Vec<_>>(), q, t);
        let (_, ry) = elevate_curve_1d(&pd.kv_v.knots, &(0..nv).map(|j| yw[j * nu + i]).collect::<Vec<_>>(), q, t);
        let (_, rw) = elevate_curve_1d(&pd.kv_v.knots, &(0..nv).map(|j| ww[j * nu + i]).collect::<Vec<_>>(), q, t);
        if nkv.is_none() { nkv = Some(KnotVector::new(V, q + t)); }
        nx.extend(rx); ny.extend(ry); nw.extend(rw);
    }
    let nkv = nkv.unwrap(); let nnv = nkv.n_basis();
    let ctrl: Vec<[f64; 2]> = (0..nu * nnv).map(|idx| { let j = idx / nu; let i = idx % nu; let w = nw[i * nnv + j]; if w.abs() > 1e-300 { [nx[i * nnv + j] / w, ny[i * nnv + j] / w] } else { [0.0, 0.0] } }).collect();
    let wgt: Vec<f64> = (0..nu * nnv).map(|idx| { let j = idx / nu; let i = idx % nu; nw[i * nnv + j] }).collect();
    NurbsPatch2DData { kv_u: pd.kv_u.clone(), kv_v: nkv, control_pts: ctrl, weights: wgt, tag: pd.tag }
}

pub fn elevate_deg_2d(pd: &NurbsPatch2DData, tu: usize, tv: usize) -> NurbsPatch2DData {
    elevate_v_2d(&elevate_u_2d(pd, tu), tv)
}

impl NurbsMesh2D {
    pub fn elevate_degree(&self, tu: usize, tv: usize) -> Self {
        NurbsMesh2D { patches: self.patches.iter().map(|p| elevate_deg_2d(p, tu, tv)).collect(), edge_connectivity: self.edge_connectivity.clone() }
    }
}

// ─── Knot insertion ──────────────────────────────────────────────────────────

pub(crate) fn insert_knot_1d(knots: &[f64], ctrl: &[f64], p: usize, u: f64) -> (Vec<f64>, Vec<f64>) {
    let n = ctrl.len() - 1; assert_eq!(knots.len() - 1, n + p + 1);
    let k = {
        let (mut lo, mut hi, mut mid) = (p, n + 1, (p + n + 1) / 2);
        while u < knots[mid] || u >= knots[mid + 1] {
            if u < knots[mid] { hi = mid; } else { lo = mid; }
            mid = (lo + hi) / 2;
        }
        mid
    };
    assert!(knots.iter().filter(|&&x| (x - u).abs() < 1e-12).count() <= p);
    let mut nk = knots.to_vec(); nk.insert(k + 1, u);
    let mut nc = Vec::with_capacity(n + 2);
    for i in 0..=k - p { nc.push(ctrl[i]); }
    for i in (k - p + 1)..=k {
        let a = if (knots[i + p] - knots[i]).abs() > 1e-300 { (u - knots[i]) / (knots[i + p] - knots[i]) } else { 0.0 };
        nc.push(a * ctrl[i] + (1.0 - a) * ctrl[i - 1]);
    }
    for i in (k + 1)..=n + 1 { nc.push(ctrl[i - 1]); }
    (nk, nc)
}

pub fn insert_knot_curve_2d(kv: &KnotVector, ctrl: &[[f64; 2]], w: &[f64], u: f64) -> (KnotVector, Vec<[f64; 2]>, Vec<f64>) {
    let p = kv.degree; let n = ctrl.len();
    let (nk, x) = insert_knot_1d(&kv.knots, &(0..n).map(|i| ctrl[i][0] * w[i]).collect::<Vec<_>>(), p, u);
    let (_, y) = insert_knot_1d(&kv.knots, &(0..n).map(|i| ctrl[i][1] * w[i]).collect::<Vec<_>>(), p, u);
    let (_, ww) = insert_knot_1d(&kv.knots, &w.to_vec(), p, u);
    let nn = x.len();
    let nc: Vec<[f64; 2]> = (0..nn).map(|i| { let w = ww[i]; if w.abs() > 1e-300 { [x[i] / w, y[i] / w] } else { [0.0, 0.0] } }).collect();
    (KnotVector::new(nk, p), nc, ww)
}

pub fn h_refine_uk(pd: &NurbsPatch2DData, u: &[f64]) -> NurbsPatch2DData {
    let nv = pd.kv_v.n_basis();
    let mut r = pd.clone();
    for &v in u {
        let nu = r.kv_u.n_basis();
        let p = r.kv_u.degree;
        let kv = r.kv_u.knots.clone(); // save original knot vector
        // Insert knot into each v-row using the SAME original knot vector
        let mut new_ctrl = Vec::new();
        let mut new_wgt = Vec::new();
        for j in 0..nv {
            let row_x: Vec<f64> = (0..nu).map(|i| r.control_pts[j*nu+i][0] * r.weights[j*nu+i]).collect();
            let row_y: Vec<f64> = (0..nu).map(|i| r.control_pts[j*nu+i][1] * r.weights[j*nu+i]).collect();
            let row_w: Vec<f64> = (0..nu).map(|i| r.weights[j*nu+i]).collect();
            let (nk, rx) = insert_knot_1d(&kv, &row_x, p, v);
            let (_, ry) = insert_knot_1d(&kv, &row_y, p, v);
            let (_, rw) = insert_knot_1d(&kv, &row_w, p, v);
            r.kv_u = KnotVector::new(nk, p); // update kv (same for all rows)
            for i in 0..r.kv_u.n_basis() {
                let w = rw[i];
                new_ctrl.push(if w.abs() > 1e-300 { [rx[i]/w, ry[i]/w] } else { [0.0,0.0] });
                new_wgt.push(w);
            }
        }
        r.control_pts = new_ctrl;
        r.weights = new_wgt;
    }
    r
}

pub fn h_refine_vk(pd: &NurbsPatch2DData, v: &[f64]) -> NurbsPatch2DData {
    let nu = pd.kv_u.n_basis(); let pv = pd.kv_v.degree;
    let mut kk = pd.kv_v.clone(); let mut cc = pd.control_pts.to_vec(); let mut ww = pd.weights.to_vec();
    for &u in v {
        let cv = kk.n_basis();
        let nn = cv + 1; // each insertion adds one basis function
        let orig_knots = kk.knots.clone(); // all columns use the SAME original knot vector
        kk = KnotVector::new(orig_knots.clone(), pv); // reset to original (will be updated after loop)
        let mut new_cc = Vec::with_capacity(nu * nn);
        let mut new_ww = Vec::with_capacity(nu * nn);
        for i in 0..nu {
            let cx: Vec<f64> = (0..cv).map(|j| cc[j*nu+i][0]*ww[j*nu+i]).collect();
            let cy: Vec<f64> = (0..cv).map(|j| cc[j*nu+i][1]*ww[j*nu+i]).collect();
            let cw: Vec<f64> = (0..cv).map(|j| ww[j*nu+i]).collect();
            let (nk, rx) = insert_knot_1d(&orig_knots, &cx, pv, u);
            let (_, ry) = insert_knot_1d(&orig_knots, &cy, pv, u);
            let (_, rw) = insert_knot_1d(&orig_knots, &cw, pv, u);
            if i == 0 { kk = KnotVector::new(nk, pv); }
            for j in 0..nn {
                let w = rw[j];
                if w.abs() > 1e-300 {
                    new_cc.push([rx[j] / w, ry[j] / w]);
                } else {
                    new_cc.push([0.0, 0.0]);
                }
                new_ww.push(w);
            }
        }
        cc = new_cc;
        ww = new_ww;
    }
    NurbsPatch2DData { kv_u: pd.kv_u.clone(), kv_v: kk, control_pts: cc, weights: ww, tag: pd.tag }
}

pub fn h_refine2d(pd: &NurbsPatch2DData, uk: &[f64], vk: &[f64]) -> NurbsPatch2DData { h_refine_vk(&h_refine_uk(pd, uk), vk) }

// ─── Spacing function ─────────────────────────────────────────────────────────

/// Compute the local element size at parametric point `(u, v)` for a 2-D NURBS patch.
///
/// Returns `sqrt(|det J|)` where `J` is the 2×2 Jacobian of the physical map,
/// which is the characteristic element size `h` at that point.
pub fn nurbs_spacing_2d(pd: &NurbsPatch2DData, u: f64, v: f64) -> f64 {
    let patch = NurbsPatch2D::new(pd.kv_u.clone(), pd.kv_v.clone(), pd.weights.clone());
    let n_dofs = patch.n_dofs();
    let mut grads = vec![0.0; n_dofs * 2];
    patch.eval_grad_basis(&[u, v], &mut grads);
    // Compute physical Jacobian J = sum_A P_A ⊗ ∇R_A
    let (mut j00, mut j01, mut j10, mut j11) = (0.0, 0.0, 0.0, 0.0);
    for dof in 0..n_dofs {
        let px = pd.control_pts[dof][0];
        let py = pd.control_pts[dof][1];
        let dud = grads[dof * 2];
        let dvd = grads[dof * 2 + 1];
        j00 += px * dud; j01 += px * dvd;
        j10 += py * dud; j11 += py * dvd;
    }
    let det = (j00 * j11 - j01 * j10).abs();
    det.sqrt()
}

/// Compute the local element size at parametric point `(u, v, w)` for a 3-D NURBS patch.
///
/// Returns `cbrt(|det J|)` where `J` is the 3×3 Jacobian.
pub fn nurbs_spacing_3d(pd: &NurbsPatch3DData, u: f64, v: f64, w: f64) -> f64 {
    let patch = NurbsPatch3D::new(pd.kv_u.clone(), pd.kv_v.clone(), pd.kv_w.clone(), pd.weights.clone());
    let n_dofs = patch.n_dofs();
    let mut grads = vec![0.0; n_dofs * 3];
    patch.eval_grad_basis(&[u, v, w], &mut grads);
    // 3×3 Jacobian
    let mut J = [[0.0; 3]; 3];
    for dof in 0..n_dofs {
        for r in 0..3 {
            for c in 0..3 {
                J[r][c] += pd.control_pts[dof][r] * grads[dof * 3 + c];
            }
        }
    }
    let det = (J[0][0]*(J[1][1]*J[2][2] - J[1][2]*J[2][1])
             - J[0][1]*(J[1][0]*J[2][2] - J[1][2]*J[2][0])
             + J[0][2]*(J[1][0]*J[2][1] - J[1][1]*J[2][0])).abs();
    det.cbrt()
}

/// Compute the spacing at the center of each non-empty knot span of a 2-D NURBS patch.
///
/// Returns `Vec` of `((span_u, span_v), h)` where each span is identified by its
/// knot-span index in the u and v directions.
pub fn nurbs_span_sizes_2d(pd: &NurbsPatch2DData) -> Vec<((usize, usize), f64)> {
    let span_centers_u: Vec<f64> = pd.kv_u.knots.windows(2)
        .filter(|w| w[1] > w[0])
        .map(|w| 0.5 * (w[0] + w[1]))
        .collect();
    let span_centers_v: Vec<f64> = pd.kv_v.knots.windows(2)
        .filter(|w| w[1] > w[0])
        .map(|w| 0.5 * (w[0] + w[1]))
        .collect();
    let mut result = Vec::with_capacity(span_centers_u.len() * span_centers_v.len());
    for (iu, &cu) in span_centers_u.iter().enumerate() {
        for (iv, &cv) in span_centers_v.iter().enumerate() {
            result.push(((iu, iv), nurbs_spacing_2d(pd, cu, cv)));
        }
    }
    result
}

// ─── 3-D knot insertion (h-refinement) ─────────────────────────────────────────

/// Insert knots into the u-direction of a 3-D NURBS patch.
///
/// For each knot value in `uvals`, iterates over all (w, v) rows and inserts
/// the knot into the u-direction of each row using [`insert_knot_1d`].
/// The rational weight is handled by multiplying coordinates by weight before
/// insertion and dividing after.
pub fn h_refine_uk_3d(pd: &NurbsPatch3DData, uvals: &[f64]) -> NurbsPatch3DData {
    let nv = pd.kv_v.n_basis();
    let nw = pd.kv_w.n_basis();
    let mut r = pd.clone();
    for &u in uvals {
        let nu = r.kv_u.n_basis();
        let p = r.kv_u.degree;
        let kv = r.kv_u.knots.clone();
        let mut new_ctrl = Vec::new();
        let mut new_wgt = Vec::new();
        for k in 0..nw {
            for j in 0..nv {
                let row_x: Vec<f64> = (0..nu)
                    .map(|i| r.control_pts[k * nu * nv + j * nu + i][0] * r.weights[k * nu * nv + j * nu + i])
                    .collect();
                let row_y: Vec<f64> = (0..nu)
                    .map(|i| r.control_pts[k * nu * nv + j * nu + i][1] * r.weights[k * nu * nv + j * nu + i])
                    .collect();
                let row_z: Vec<f64> = (0..nu)
                    .map(|i| r.control_pts[k * nu * nv + j * nu + i][2] * r.weights[k * nu * nv + j * nu + i])
                    .collect();
                let row_w: Vec<f64> = (0..nu)
                    .map(|i| r.weights[k * nu * nv + j * nu + i])
                    .collect();
                let (nk, rx) = insert_knot_1d(&kv, &row_x, p, u);
                let (_, ry) = insert_knot_1d(&kv, &row_y, p, u);
                let (_, rz) = insert_knot_1d(&kv, &row_z, p, u);
                let (_, rw) = insert_knot_1d(&kv, &row_w, p, u);
                r.kv_u = KnotVector::new(nk, p);
                for i in 0..r.kv_u.n_basis() {
                    let w = rw[i];
                    new_ctrl.push(
                        if w.abs() > 1e-300 {
                            [rx[i] / w, ry[i] / w, rz[i] / w]
                        } else {
                            [0.0, 0.0, 0.0]
                        },
                    );
                    new_wgt.push(w);
                }
            }
        }
        r.control_pts = new_ctrl;
        r.weights = new_wgt;
    }
    r
}

/// Insert knots into the v-direction of a 3-D NURBS patch.
///
/// For each knot value in `vvals`, iterates over all (w, u) columns and inserts
/// the knot into the v-direction of each column using [`insert_knot_1d`].
pub fn h_refine_vk_3d(pd: &NurbsPatch3DData, vvals: &[f64]) -> NurbsPatch3DData {
    let nu = pd.kv_u.n_basis();
    let nw = pd.kv_w.n_basis();
    let mut r = pd.clone();
    for &v in vvals {
        let nv = r.kv_v.n_basis();
        let p = r.kv_v.degree;
        let kv = r.kv_v.knots.clone();
        // Dummy call to get new knot vector (nk depends only on kv and u, not ctrl).
        let (nk, _) = insert_knot_1d(&kv, &vec![0.0; nv], p, v);
        let nnv = nk.len() - p - 1; // n_basis after insertion
        let mut new_ctrl = vec![[0.0; 3]; nw * nu * nnv];
        let mut new_wgt = vec![0.0; nw * nu * nnv];
        r.kv_v = KnotVector::new(nk, p);
        for k in 0..nw {
            for i in 0..nu {
                let col_x: Vec<f64> = (0..nv)
                    .map(|j| r.control_pts[k * nu * nv + j * nu + i][0] * r.weights[k * nu * nv + j * nu + i])
                    .collect();
                let col_y: Vec<f64> = (0..nv)
                    .map(|j| r.control_pts[k * nu * nv + j * nu + i][1] * r.weights[k * nu * nv + j * nu + i])
                    .collect();
                let col_z: Vec<f64> = (0..nv)
                    .map(|j| r.control_pts[k * nu * nv + j * nu + i][2] * r.weights[k * nu * nv + j * nu + i])
                    .collect();
                let col_w: Vec<f64> = (0..nv)
                    .map(|j| r.weights[k * nu * nv + j * nu + i])
                    .collect();
                let (_, rx) = insert_knot_1d(&kv, &col_x, p, v);
                let (_, ry) = insert_knot_1d(&kv, &col_y, p, v);
                let (_, rz) = insert_knot_1d(&kv, &col_z, p, v);
                let (_, rw) = insert_knot_1d(&kv, &col_w, p, v);
                for j in 0..nnv {
                    let w = rw[j];
                    let idx = k * nu * nnv + j * nu + i;
                    new_ctrl[idx] = if w.abs() > 1e-300 {
                        [rx[j] / w, ry[j] / w, rz[j] / w]
                    } else {
                        [0.0, 0.0, 0.0]
                    };
                    new_wgt[idx] = w;
                }
            }
        }
        r.control_pts = new_ctrl;
        r.weights = new_wgt;
    }
    r
}

/// Insert knots into the w-direction of a 3-D NURBS patch.
///
/// For each knot value in `wvals`, iterates over all (v, u) columns and inserts
/// the knot into the w-direction of each column using [`insert_knot_1d`].
pub fn h_refine_wk_3d(pd: &NurbsPatch3DData, wvals: &[f64]) -> NurbsPatch3DData {
    let nu = pd.kv_u.n_basis();
    let nv = pd.kv_v.n_basis();
    let mut r = pd.clone();
    for &w in wvals {
        let nw = r.kv_w.n_basis();
        let p = r.kv_w.degree;
        let kv = r.kv_w.knots.clone();
        // Dummy call to get new knot vector.
        let (nk, _) = insert_knot_1d(&kv, &vec![0.0; nw], p, w);
        let nnw = nk.len() - p - 1; // n_basis after insertion
        let mut new_ctrl = vec![[0.0; 3]; nnw * nu * nv];
        let mut new_wgt = vec![0.0; nnw * nu * nv];
        r.kv_w = KnotVector::new(nk, p);
        for j in 0..nv {
            for i in 0..nu {
                let col_x: Vec<f64> = (0..nw)
                    .map(|k| r.control_pts[k * nu * nv + j * nu + i][0] * r.weights[k * nu * nv + j * nu + i])
                    .collect();
                let col_y: Vec<f64> = (0..nw)
                    .map(|k| r.control_pts[k * nu * nv + j * nu + i][1] * r.weights[k * nu * nv + j * nu + i])
                    .collect();
                let col_z: Vec<f64> = (0..nw)
                    .map(|k| r.control_pts[k * nu * nv + j * nu + i][2] * r.weights[k * nu * nv + j * nu + i])
                    .collect();
                let col_w: Vec<f64> = (0..nw)
                    .map(|k| r.weights[k * nu * nv + j * nu + i])
                    .collect();
                let (_, rx) = insert_knot_1d(&kv, &col_x, p, w);
                let (_, ry) = insert_knot_1d(&kv, &col_y, p, w);
                let (_, rz) = insert_knot_1d(&kv, &col_z, p, w);
                let (_, rw) = insert_knot_1d(&kv, &col_w, p, w);
                for k in 0..nnw {
                    let wgt = rw[k];
                    let idx = k * nu * nv + j * nu + i;
                    new_ctrl[idx] = if wgt.abs() > 1e-300 {
                        [rx[k] / wgt, ry[k] / wgt, rz[k] / wgt]
                    } else {
                        [0.0, 0.0, 0.0]
                    };
                    new_wgt[idx] = wgt;
                }
            }
        }
        r.control_pts = new_ctrl;
        r.weights = new_wgt;
    }
    r
}

/// Insert knots into all three directions of a 3-D NURBS patch.
///
/// Composition order: u first, then v, then w (matching the 2-D pattern).
pub fn h_refine_3d(
    pd: &NurbsPatch3DData,
    uvals: &[f64],
    vvals: &[f64],
    wvals: &[f64],
) -> NurbsPatch3DData {
    h_refine_wk_3d(&h_refine_vk_3d(&h_refine_uk_3d(pd, uvals), vvals), wvals)
}

impl NurbsMesh3D {
    /// Refine each patch by inserting knots in u, v, w.
    ///
    /// Face connectivity is preserved unchanged.
    pub fn h_refine(&self, uvals: &[f64], vvals: &[f64], wvals: &[f64]) -> Self {
        NurbsMesh3D {
            patches: self
                .patches
                .iter()
                .map(|p| h_refine_3d(p, uvals, vvals, wvals))
                .collect(),
            face_connectivity: self.face_connectivity.clone(),
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── KnotVector tests ──────────────────────────────────────────────────────

    #[test]
    fn knot_vector_uniform_degree1() {
        let kv = KnotVector::uniform(1, 3);
        // p=1, n_elems=3 → [0,0, 1/3, 2/3, 1,1]
        assert_eq!(kv.knots.len(), 6);
        assert_eq!(kv.n_basis(), 4);   // 6 - 1 - 1 = 4
        assert_eq!(kv.n_spans(), 3);
        assert!((kv.knots[2] - 1.0/3.0).abs() < 1e-15);
    }

    #[test]
    fn knot_vector_uniform_degree2() {
        let kv = KnotVector::uniform(2, 4);
        // p=2, n_elems=4 → [0,0,0, 1/4, 2/4, 3/4, 1,1,1]
        assert_eq!(kv.knots.len(), 9);
        assert_eq!(kv.n_basis(), 6);
        assert_eq!(kv.n_spans(), 4);
    }

    #[test]
    fn find_span_is_correct() {
        let kv = KnotVector::uniform(2, 4);
        // interior spans: [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1]
        assert_eq!(kv.find_span(0.0),   2);
        assert_eq!(kv.find_span(0.125), 2);
        assert_eq!(kv.find_span(0.25),  3);
        assert_eq!(kv.find_span(0.5),   4);
        assert_eq!(kv.find_span(0.75),  5);
        assert_eq!(kv.find_span(1.0),   5); // clamped to last non-empty span
    }

    #[test]
    fn basis_funs_sum_to_one() {
        let kv = KnotVector::uniform(2, 5);
        for xi in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] {
            let span = kv.find_span(xi);
            let n = kv.basis_funs(span, xi);
            let sum: f64 = n.iter().sum();
            assert!((sum - 1.0).abs() < 1e-14, "xi={xi}: basis sum = {sum}");
        }
    }

    #[test]
    fn bspline_basis1d_partition_of_unity() {
        let kv = KnotVector::uniform(3, 6);
        let basis = BSplineBasis1D::new(kv);
        for xi in [0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0] {
            let vals = basis.eval(xi);
            let sum: f64 = vals.iter().sum();
            assert!((sum - 1.0).abs() < 1e-13, "xi={xi}: sum = {sum}");
        }
    }

    #[test]
    fn bspline_basis1d_derivatives_finite_diff() {
        let kv = KnotVector::uniform(2, 4);
        let basis = BSplineBasis1D::new(kv);
        let h = 1e-6;
        for xi in [0.1, 0.35, 0.6, 0.85] {
            let (_, dn) = basis.eval_with_ders(xi);
            let n_p = basis.eval(xi + h);
            let n_m = basis.eval(xi - h);
            for i in 0..basis.n_basis() {
                let fd = (n_p[i] - n_m[i]) / (2.0 * h);
                assert!((dn[i] - fd).abs() < 1e-5,
                    "xi={xi}, dof={i}: analytic={:.6} fd={:.6}", dn[i], fd);
            }
        }
    }

    // ── NurbsPatch2D tests ────────────────────────────────────────────────────

    #[test]
    fn nurbs2d_partition_of_unity() {
        let kv = KnotVector::uniform(2, 3);
        let patch = NurbsPatch2D::uniform(kv.clone(), kv.clone());
        let n = patch.n_dofs();
        let mut vals = vec![0.0; n];
        for &u in &[0.05, 0.25, 0.5, 0.75, 0.95] {
            for &v in &[0.05, 0.25, 0.5, 0.75, 0.95] {
                patch.eval_basis(&[u, v], &mut vals);
                let sum: f64 = vals.iter().sum();
                assert!((sum - 1.0).abs() < 1e-12,
                    "u={u}, v={v}: sum = {sum}");
            }
        }
    }

    #[test]
    fn nurbs2d_grad_finite_difference() {
        let kv = KnotVector::uniform(2, 3);
        let patch = NurbsPatch2D::uniform(kv.clone(), kv.clone());
        let n = patch.n_dofs();
        let h = 1e-6;
        let u0 = 0.4;
        let v0 = 0.6;

        let mut grads = vec![0.0; n * 2];
        patch.eval_grad_basis(&[u0, v0], &mut grads);

        let mut vp = vec![0.0; n];
        let mut vm = vec![0.0; n];

        // dR/du: finite diff
        patch.eval_basis(&[u0 + h, v0], &mut vp);
        patch.eval_basis(&[u0 - h, v0], &mut vm);
        for i in 0..n {
            let fd = (vp[i] - vm[i]) / (2.0 * h);
            assert!((grads[i * 2] - fd).abs() < 1e-5,
                "dof={i}: dR/du analytic={:.6} fd={:.6}", grads[i * 2], fd);
        }

        // dR/dv: finite diff
        patch.eval_basis(&[u0, v0 + h], &mut vp);
        patch.eval_basis(&[u0, v0 - h], &mut vm);
        for i in 0..n {
            let fd = (vp[i] - vm[i]) / (2.0 * h);
            assert!((grads[i * 2 + 1] - fd).abs() < 1e-5,
                "dof={i}: dR/dv analytic={:.6} fd={:.6}", grads[i * 2 + 1], fd);
        }
    }

    #[test]
    fn nurbs2d_dof_coords_count() {
        let kv_u = KnotVector::uniform(1, 4);
        let kv_v = KnotVector::uniform(2, 3);
        let patch = NurbsPatch2D::uniform(kv_u, kv_v);
        let coords = patch.dof_coords();
        assert_eq!(coords.len(), patch.n_dofs());
        for c in &coords {
            assert_eq!(c.len(), 2);
            assert!(c[0] >= 0.0 && c[0] <= 1.0);
            assert!(c[1] >= 0.0 && c[1] <= 1.0);
        }
    }

    // ── NurbsPatch3D tests ────────────────────────────────────────────────────

    #[test]
    fn nurbs3d_partition_of_unity() {
        let kv = KnotVector::uniform(1, 2);
        let patch = NurbsPatch3D::uniform(kv.clone(), kv.clone(), kv.clone());
        let n = patch.n_dofs();
        let mut vals = vec![0.0; n];
        for &u in &[0.1, 0.5, 0.9] {
            for &v in &[0.1, 0.5, 0.9] {
                for &w in &[0.1, 0.5, 0.9] {
                    patch.eval_basis(&[u, v, w], &mut vals);
                    let sum: f64 = vals.iter().sum();
                    assert!((sum - 1.0).abs() < 1e-12,
                        "u={u},v={v},w={w}: sum={sum}");
                }
            }
        }
    }

    #[test]
    fn nurbs3d_grad_finite_difference() {
        let kv = KnotVector::uniform(1, 2);
        let patch = NurbsPatch3D::uniform(kv.clone(), kv.clone(), kv.clone());
        let n = patch.n_dofs();
        let h = 1e-6;
        // Avoid knot boundaries (xi=0.5) where C^0 continuity causes FD mismatch.
        let pt = [0.3, 0.4, 0.6];

        let mut grads = vec![0.0; n * 3];
        patch.eval_grad_basis(&pt, &mut grads);

        let mut vp = vec![0.0; n];
        let mut vm = vec![0.0; n];

        for dir in 0..3 {
            let mut pt_p = pt;
            let mut pt_m = pt;
            pt_p[dir] += h;
            pt_m[dir] -= h;
            patch.eval_basis(&pt_p, &mut vp);
            patch.eval_basis(&pt_m, &mut vm);
            for i in 0..n {
                let fd = (vp[i] - vm[i]) / (2.0 * h);
                let an = grads[i * 3 + dir];
                assert!((an - fd).abs() < 1e-5,
                    "dir={dir}, dof={i}: analytic={:.6} fd={:.6}", an, fd);
            }
        }
    }

    // ── NurbsMesh tests ───────────────────────────────────────────────────────

    #[test]
    fn nurbs_mesh2d_single_patch_square() {
        let kv = KnotVector::uniform(1, 1);
        let pts = vec![[0.0f64, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let mesh = NurbsMesh2D::single_patch(kv.clone(), kv.clone(), pts, vec![1.0; 4]);
        assert_eq!(mesh.n_patches(), 1);
        assert_eq!(mesh.n_control_pts_total(), 4);

        let elem = mesh.patch_element(0);
        assert_eq!(elem.n_dofs(), 4);
    }

    #[test]
    fn greville_abscissae_degree1_uniform() {
        let kv = KnotVector::uniform(1, 4);
        // p=1, n_basis=5, knots=[0,0,0.25,0.5,0.75,1,1]
        // grev[i] = knots[i+1]
        let g = greville_abscissae(&kv);
        assert_eq!(g.len(), 5);
        assert!((g[0] - 0.0).abs() < 1e-15);
        assert!((g[1] - 0.25).abs() < 1e-15);
        assert!((g[2] - 0.5).abs() < 1e-15);
        assert!((g[3] - 0.75).abs() < 1e-15);
        assert!((g[4] - 1.0).abs() < 1e-15);
    }

    fn eval_1d(kv: &KnotVector, ctrl: &[f64], xi: f64) -> f64 {
        let s = kv.find_span(xi); let b = kv.basis_funs(s, xi); let p = kv.degree;
        (0..=p).map(|j| b[j] * ctrl[s-p+j]).sum()
    }

    #[test]
    fn elevate_degree_curve_preserves_shape() {
        let kv = KnotVector::uniform(1, 4);
        let c: Vec<f64> = (0..5).map(|i| i as f64/4.0).collect();
        let (nk, nc) = elevate_curve_1d(&kv.knots, &c, 1, 1);
        let nkv = KnotVector::new(nk, 2);
        assert_eq!(nc.len(), 9);
        for i in 0..=20 { let x = i as f64/20.0; assert!((eval_1d(&kv,&c,x)-eval_1d(&nkv,&nc,x)).abs()<1e-12); }
    }

    #[test]
    fn elevate_degree_curve_multi_step() {
        let kv = KnotVector::uniform(1, 3);
        let c: Vec<f64> = (0..4).map(|i| i as f64/3.0).collect();
        let (nk, nc) = elevate_curve_1d(&kv.knots, &c, 1, 2);
        let nkv = KnotVector::new(nk, 3);
        for i in 0..=20 { let x = i as f64/20.0; assert!((eval_1d(&kv,&c,x)-eval_1d(&nkv,&nc,x)).abs()<1e-12); }
    }

    #[test]
    fn elevate_surface_2d_increases_degree() {
        let pd = NurbsPatch2DData {
            kv_u: KnotVector::uniform(1, 3), kv_v: KnotVector::uniform(1, 2),
            control_pts: (0..12).map(|i|[i as f64/12.0,(i%4)as f64/3.0]).collect(),
            weights: vec![1.0; 12], tag: 1,
        };
        assert_eq!(elevate_u_2d(&pd, 1).kv_u.degree, 2);
        assert_eq!(elevate_v_2d(&pd, 1).kv_v.degree, 2);
        assert_eq!(elevate_deg_2d(&pd, 1, 1).kv_u.degree, 2);
        assert_eq!(elevate_deg_2d(&pd, 1, 1).kv_v.degree, 2);
    }

    #[test]
    fn h_refine_knot_increases_cp() {
        let kv = KnotVector::uniform(1, 4);
        let c: Vec<f64> = (0..5).map(|i| i as f64/4.0).collect();
        let (nk, nc) = insert_knot_1d(&kv.knots, &c, 1, 0.5);
        let nkv = KnotVector::new(nk, 1);
        assert_eq!(nc.len(), 6);
        for i in 0..=20 { let x = i as f64/20.0; assert!((eval_1d(&kv,&c,x)-eval_1d(&nkv,&nc,x)).abs()<1e-14); }
    }

    #[test]
    fn h_refine_2d_u_increases_cp() {
        let pd = NurbsPatch2DData {
            kv_u: KnotVector::uniform(1, 3), kv_v: KnotVector::uniform(1, 2),
            control_pts: (0..12).map(|i|[i as f64/12.0,(i%4)as f64/3.0]).collect(),
            weights: vec![1.0; 12], tag: 1,
        };
        let cp_old = pd.control_pts.len();
        let r = h_refine_uk(&pd, &[0.5]);
        assert_eq!(r.control_pts.len(), cp_old + pd.kv_v.n_basis());
    }

    #[test]
    fn spacing_2d_unit_square() {
        let pd = NurbsPatch2DData {
            kv_u: KnotVector::uniform(1, 1), kv_v: KnotVector::uniform(1, 1),
            control_pts: vec![[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]],
            weights: vec![1.0; 4], tag: 1,
        };
        let h = nurbs_spacing_2d(&pd, 0.5, 0.5);
        assert!((h - 1.0).abs() < 1e-12, "unit square spacing={h}");
    }

    #[test]
    fn spacing_2d_stretched() {
        let pd = NurbsPatch2DData {
            kv_u: KnotVector::uniform(1, 1), kv_v: KnotVector::uniform(1, 1),
            control_pts: vec![[0.0,0.0],[2.0,0.0],[0.0,3.0],[2.0,3.0]],
            weights: vec![1.0; 4], tag: 1,
        };
        // det J = 2*3 = 6 for a uniform 2×3 rectangle
        let h = nurbs_spacing_2d(&pd, 0.5, 0.5);
        assert!((h - (6.0_f64).sqrt()).abs() < 1e-12, "stretched spacing={h}");
    }

    #[test]
    fn spacing_span_sizes_match_patch_count() {
        // deg 2 with 3 spans → n_basis = 5, deg 1 with 2 spans → n_basis = 3
        let pd = NurbsPatch2DData {
            kv_u: KnotVector::uniform(2, 3), kv_v: KnotVector::uniform(1, 2),
            control_pts: (0..15).map(|i|[i as f64/5.0,(i%5)as f64/3.0]).collect(),
            weights: vec![1.0; 15], tag: 1,
        };
        let sizes = nurbs_span_sizes_2d(&pd);
        assert_eq!(sizes.len(), 3 * 2); // 3 u-spans × 2 v-spans
    }

    // ── 3-D h-refinement tests ─────────────────────────────────────────────

    #[test]
    fn h_refine_3d_u_increases_cp() {
        // uniform(1,1) → n_basis = 2 in each direction → 8 total CP
        let pd = NurbsPatch3DData {
            kv_u: KnotVector::uniform(1, 1),
            kv_v: KnotVector::uniform(1, 1),
            kv_w: KnotVector::uniform(1, 1),
            control_pts: (0..8).map(|i| [i as f64, (i % 2) as f64, (i / 4) as f64]).collect(),
            weights: vec![1.0; 8],
            tag: 1,
        };
        let cp_old = pd.control_pts.len(); // 8
        let r = h_refine_uk_3d(&pd, &[0.5]);
        // nw*nv = 2*2 = 4 new control points added (one per (w,v) row)
        assert_eq!(r.control_pts.len(), cp_old + pd.kv_v.n_basis() * pd.kv_w.n_basis());
    }

    #[test]
    fn h_refine_3d_v_increases_cp() {
        // uniform(1,1) → n_basis = 2 in each direction → 8 total CP
        let pd = NurbsPatch3DData {
            kv_u: KnotVector::uniform(1, 1),
            kv_v: KnotVector::uniform(1, 1),
            kv_w: KnotVector::uniform(1, 1),
            control_pts: (0..8).map(|i| [i as f64, (i % 2) as f64, (i / 4) as f64]).collect(),
            weights: vec![1.0; 8],
            tag: 1,
        };
        let cp_old = pd.control_pts.len(); // 8
        let r = h_refine_vk_3d(&pd, &[0.5]);
        // nw*nu = 2*2 = 4 new control points added
        assert_eq!(r.control_pts.len(), cp_old + pd.kv_u.n_basis() * pd.kv_w.n_basis());
    }

    #[test]
    fn h_refine_3d_w_increases_cp() {
        // uniform(1,1) → n_basis = 2 in each direction → 8 total CP
        let pd = NurbsPatch3DData {
            kv_u: KnotVector::uniform(1, 1),
            kv_v: KnotVector::uniform(1, 1),
            kv_w: KnotVector::uniform(1, 1),
            control_pts: (0..8).map(|i| [i as f64, (i % 2) as f64, (i / 4) as f64]).collect(),
            weights: vec![1.0; 8],
            tag: 1,
        };
        let cp_old = pd.control_pts.len(); // 8
        let r = h_refine_wk_3d(&pd, &[0.5]);
        // nv*nu = 2*2 = 4 new control points added
        assert_eq!(r.control_pts.len(), cp_old + pd.kv_u.n_basis() * pd.kv_v.n_basis());
    }

    #[test]
    fn h_refine_3d_all_directions_increases_cp() {
        // uniform(1,1) → n_basis = 2 in each direction → 8 total CP
        let pd = NurbsPatch3DData {
            kv_u: KnotVector::uniform(1, 1),
            kv_v: KnotVector::uniform(1, 1),
            kv_w: KnotVector::uniform(1, 1),
            control_pts: (0..8).map(|i| [i as f64, (i % 2) as f64, (i / 4) as f64]).collect(),
            weights: vec![1.0; 8],
            tag: 1,
        };
        let r = h_refine_3d(&pd, &[0.5], &[0.5], &[0.5]);
        // Each direction adds one knot → nu=3, nv=3, nw=3 → total = 27
        assert_eq!(r.control_pts.len(), 27);
    }

    // ── Second derivative tests ──────────────────────────────────────────────

    #[test]
    fn bspline_second_derivatives_match_fd() {
        // Degree 1 is piecewise linear (exact zero second derivative).
        // The FD comparison is noise-dominated (1e-16 / h^2 ~ 2e-4), so skip it.
        for degree in 2..=4 {
            let kv = KnotVector::uniform(degree, 5);
            let basis = BSplineBasis1D::new(kv);
            let eps = 1e-6;
            for xi in [0.05, 0.15, 0.35, 0.5, 0.65, 0.85, 0.95] {
                let (_, _, ddn) = basis.eval_with_ders2(xi);
                let (np, _) = basis.eval_with_ders(xi + eps);
                let (nm, _) = basis.eval_with_ders(xi - eps);
                let n0 = basis.eval(xi);
                // Second-order central difference on basis values
                for j in 0..basis.n_basis() {
                    let fd = (np[j] - 2.0 * n0[j] + nm[j]) / (eps * eps);
                    assert!(
                        (ddn[j] - fd).abs() < 1e-4,
                        "degree={degree}, xi={xi}, j={j}: analytic={:.6e} fd={:.6e}",
                        ddn[j],
                        fd
                    );
                }
            }
        }
    }

    #[test]
    fn nurbs2d_hessian_is_symmetric() {
        let kv = KnotVector::uniform(2, 3);
        let patch = NurbsPatch2D::uniform(kv.clone(), kv.clone());
        let n = patch.n_dofs();
        let mut hess = vec![0.0; n * 4];
        patch.eval_hessian_basis(&[0.4, 0.6], &mut hess);
        for a in 0..n {
            assert!(
                (hess[a * 4 + 1] - hess[a * 4 + 2]).abs() < 1e-14,
                "d2R/dudv != d2R/dvdu at dof {a}"
            );
        }
    }

    #[test]
    fn nurbs2d_hessian_sum_to_zero_linear() {
        // For a linear NURBS (degree 1), second derivatives should be zero
        // because the basis functions are piecewise linear.
        // Use interior point away from knots (0.5 is a knot for uniform(1,4))
        let kv = KnotVector::uniform(1, 4);
        let patch = NurbsPatch2D::uniform(kv.clone(), kv.clone());
        let n = patch.n_dofs();
        let mut hess = vec![0.0; n * 4];
        patch.eval_hessian_basis(&[0.3, 0.7], &mut hess);
        for a in 0..n {
            assert!(
                hess[a * 4].abs() < 1e-12,
                "d2R/du2 not zero for degree 1 at dof {a}: {}",
                hess[a * 4]
            );
            assert!(
                hess[a * 4 + 3].abs() < 1e-12,
                "d2R/dv2 not zero for degree 1 at dof {a}: {}",
                hess[a * 4 + 3]
            );
        }
    }

    #[test]
    fn nurbs2d_hessian_fd_check() {
        let kv = KnotVector::uniform(2, 3);
        let patch = NurbsPatch2D::uniform(kv.clone(), kv.clone());
        let n = patch.n_dofs();
        let h = 1e-6;
        let u0 = 0.4;
        let v0 = 0.6;

        let mut hess = vec![0.0; n * 4];
        patch.eval_hessian_basis(&[u0, v0], &mut hess);

        // d2R/du2 via FD on first derivatives in u
        let mut gp = vec![0.0; n * 2];
        let mut gm = vec![0.0; n * 2];
        patch.eval_grad_basis(&[u0 + h, v0], &mut gp);
        patch.eval_grad_basis(&[u0 - h, v0], &mut gm);
        for a in 0..n {
            let fd = (gp[a * 2] - gm[a * 2]) / (2.0 * h);
            assert!(
                (hess[a * 4] - fd).abs() < 5e-4,
                "a={a}: d2R/du2 analytic={:.6e} fd={:.6e}",
                hess[a * 4],
                fd
            );
        }

        // d2R/dv2 via FD on first derivatives in v
        patch.eval_grad_basis(&[u0, v0 + h], &mut gp);
        patch.eval_grad_basis(&[u0, v0 - h], &mut gm);
        for a in 0..n {
            let fd = (gp[a * 2 + 1] - gm[a * 2 + 1]) / (2.0 * h);
            assert!(
                (hess[a * 4 + 3] - fd).abs() < 5e-4,
                "a={a}: d2R/dv2 analytic={:.6e} fd={:.6e}",
                hess[a * 4 + 3],
                fd
            );
        }

        // d2R/dudv via FD on dR/du w.r.t. v
        patch.eval_grad_basis(&[u0, v0 + h], &mut gp);
        patch.eval_grad_basis(&[u0, v0 - h], &mut gm);
        for a in 0..n {
            let fd = (gp[a * 2] - gm[a * 2]) / (2.0 * h);
            assert!(
                (hess[a * 4 + 1] - fd).abs() < 5e-4,
                "a={a}: d2R/dudv analytic={:.6e} fd={:.6e}",
                hess[a * 4 + 1],
                fd
            );
        }
    }

    #[test]
    fn bspline_second_derivatives_nonuniform() {
        let kv =
            KnotVector::new(vec![0.0, 0.0, 0.0, 0.2, 0.5, 0.8, 1.0, 1.0, 1.0], 2);
        let basis = BSplineBasis1D::new(kv);
        let eps = 1e-6;
        for xi in [0.05, 0.15, 0.35, 0.65, 0.85, 0.95] {
            let (_, _, ddn) = basis.eval_with_ders2(xi);
            let (np, _) = basis.eval_with_ders((xi + eps).min(0.9999));
            let (nm, _) = basis.eval_with_ders((xi - eps).max(0.0001));
            let n0 = basis.eval(xi);
            for j in 0..basis.n_basis() {
                let fd = (np[j] - 2.0 * n0[j] + nm[j]) / (eps * eps);
                assert!(
                    (ddn[j] - fd).abs() < 1e-4,
                    "xi={xi}, j={j}: analytic={:.6e} fd={:.6e}",
                    ddn[j],
                    fd
                );
            }
        }
    }
}
