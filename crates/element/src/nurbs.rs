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
        for _ in 0..=degree { knots.push(0.0); }
        // interior knots
        for i in 1..n_elems {
            knots.push(i as f64 / n_elems as f64);
        }
        // p+1 trailing ones
        for _ in 0..=degree { knots.push(1.0); }
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
}
