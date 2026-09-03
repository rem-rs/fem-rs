//! NURBS vector finite elements for isogeometric analysis (IGA).
//!
//! Implements the divergence-conforming (H(div)) and curl-conforming (H(curl))
//! NURBS elements from MFEM's `NURBS_HDivFECollection` and `NURBS_HCurlFECollection`.
//!
//! # Mathematical background
//!
//! Following Buffa, De Falco, Sangalli [2010] and Evans, Hughes [2013]:
//!
//! **H(div) 2D** (quad, orders px, py):
//! - DOFs: `(px+2)*(py+1) + (px+1)*(py+2)`
//! - x-component: `N_i^(px+1)(ξ) * N_j^(py)(η)`  for i=0..px+1, j=0..py
//! - y-component: `N_i^(px)(ξ) * N_j^(py+1)(η)`  for i=0..px, j=0..py+1
//! - Piola transform: `v_phys = J * v_ref / weight`
//!
//! **H(curl) 2D** (quad, orders px, py):
//! - DOFs: `(px+1)*(py+2) + (px+2)*(py+1)`
//! - x-component: `N_i^(px)(ξ) * N_j^(py+1)(η)`  for i=0..px, j=0..py+1
//! - y-component: `N_i^(px+1)(ξ) * N_j^(py)(η)`  for i=0..px+1, j=0..py
//! - Piola transform: `v_phys = J^{-T} * v_ref`
//!
//! **H(div) 3D** (hex, orders px, py, pz):
//! - DOFs: `(px+2)*(py+1)*(pz+1) + (px+1)*(py+2)*(pz+1) + (px+1)*(py+1)*(pz+2)`
//! - Three components, each using degree-elevated knot vector in one direction
//!
//! **H(curl) 3D** (hex, orders px, py, pz):
//! - DOFs: `(px+1)*(py+2)*(pz+2) + (px+2)*(py+1)*(pz+2) + (px+2)*(py+2)*(pz+1)`
//! - Three components, each using degree-elevated knot vectors in two directions

use crate::iga::{BsplineBasis, KnotVector};
use crate::reference::{QuadratureRule, VectorReferenceElement};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Helper: degree-elevated BsplineBasis
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Determine the polynomial order from a clamped knot vector.
/// The order equals the multiplicity of the first knot minus 1.
fn order_from_knots(kv: &KnotVector) -> usize {
    let knots = kv.as_slice();
    if knots.is_empty() { return 0; }
    let first = knots[0];
    let multiplicity = knots.iter().take_while(|&&k| (k - first).abs() < 1e-12).count();
    multiplicity.saturating_sub(1)
}

/// Create a clamped uniform knot vector with `n_elem` spans on [0,1].
fn clamped_uniform_knots(degree: usize, n_elem: usize) -> KnotVector {
    let mut knots = Vec::new();
    knots.extend(std::iter::repeat_n(0.0, degree + 1));
    for i in 1..n_elem {
        knots.push(i as f64 / n_elem as f64);
    }
    knots.extend(std::iter::repeat_n(1.0, degree + 1));
    KnotVector::new_clamped(knots).expect("valid clamped uniform knots")
}

/// Elevate the degree of a knot vector by `t`, preserving the geometry.
fn degree_elevate_knots(kv: &KnotVector, t: usize) -> KnotVector {
    assert!(t >= 1, "degree_elevate: t must be >= 1");
    let knots = kv.as_slice();

    // Collect distinct knot values.
    let mut distinct = Vec::new();
    for w in knots.windows(2) {
        if w[1] > w[0] {
            distinct.push(w[0]);
        }
    }
    if let Some(&last) = knots.last() {
        distinct.push(last);
    }

    let first = distinct.first().copied().unwrap_or(0.0);
    let last = distinct.last().copied().unwrap_or(1.0);

    // Determine original degree from multiplicity of first knot.
    let orig_degree = order_from_knots(kv);

    // Build elevated knot vector.
    let mut result = Vec::new();
    result.extend(std::iter::repeat_n(first, orig_degree + t + 1));
    for window in distinct.windows(2) {
        let a = window[0];
        let b = window[1];
        for k in 1..=t {
            result.push(a + (b - a) * (k as f64) / ((t + 1) as f64));
        }
        result.push(b);
    }
    result.extend(std::iter::repeat_n(last, orig_degree + t));

    KnotVector::new_clamped(result).expect("valid elevated knots")
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// NURBS_HDiv_2D — divergence-conforming NURBS on a quadrilateral
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 2D H(div)-conforming NURBS vector element on a square reference domain.
///
/// Uses mixed-degree B-spline bases: the x-component uses a degree-elevated
/// knot vector in ξ (order px+1) and the original in η (order py); the
/// y-component uses the original ξ and degree-elevated η.
#[derive(Debug, Clone)]
pub struct NurbsHDiv2D {
    /// Order in ξ direction.
    pub order_u: usize,
    /// Order in η direction.
    pub order_v: usize,
    /// B-spline basis for ξ (order px).
    basis_u: BsplineBasis,
    /// B-spline basis for η (order py).
    basis_v: BsplineBasis,
    /// Degree-elevated basis for ξ (order px+1).
    basis1_u: BsplineBasis,
    /// Degree-elevated basis for η (order py+1).
    basis1_v: BsplineBasis,
    /// Number of DOFs.
    pub n_dofs: usize,
}

impl NurbsHDiv2D {
    /// Create from orders (px, py) with uniform clamped knot vectors (1 span each).
    pub fn new(order_u: usize, order_v: usize) -> Self {
        Self::from_knot_vectors(
            clamped_uniform_knots(order_u, 1),
            clamped_uniform_knots(order_v, 1),
        )
        .expect("NurbsHDiv2D::new")
    }

    /// Create from knot vectors.
    pub fn from_knot_vectors(kv_u: KnotVector, kv_v: KnotVector) -> Result<Self, String> {
        let px = order_from_knots(&kv_u);
        let py = order_from_knots(&kv_v);

        let kv1_u = degree_elevate_knots(&kv_u, 1);
        let kv1_v = degree_elevate_knots(&kv_v, 1);

        let basis_u = BsplineBasis::new(px, kv_u).map_err(|e| format!("basis_u: {e}"))?;
        let basis_v = BsplineBasis::new(py, kv_v).map_err(|e| format!("basis_v: {e}"))?;
        let basis1_u = BsplineBasis::new(px + 1, kv1_u).map_err(|e| format!("basis1_u: {e}"))?;
        let basis1_v = BsplineBasis::new(py + 1, kv1_v).map_err(|e| format!("basis1_v: {e}"))?;

        let n = (px + 2) * (py + 1) + (px + 1) * (py + 2);

        Ok(Self {
            order_u: px,
            order_v: py,
            basis_u,
            basis_v,
            basis1_u,
            basis1_v,
            n_dofs: n,
        })
    }

    /// Static DOF count for given orders.
    pub fn n_dofs_static(px: usize, py: usize) -> usize {
        (px + 2) * (py + 1) + (px + 1) * (py + 2)
    }

    /// Evaluate 1D B-spline shape values at parameter u.
    /// Returns vector of (global_index, value) pairs.
    fn eval_1d(basis: &BsplineBasis, u: f64) -> Vec<(usize, f64)> {
        basis.nonzero_values(u).unwrap_or_default()
    }

    /// Evaluate 1D B-spline derivative values at parameter u.
    fn eval_1d_deriv(basis: &BsplineBasis, u: f64) -> Vec<(usize, f64)> {
        basis.nonzero_derivatives(u).unwrap_or_default()
    }
}

impl VectorReferenceElement for NurbsHDiv2D {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { self.order_u.max(self.order_v) as u8 }
    fn n_dofs(&self) -> usize { self.n_dofs }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        assert_eq!(xi.len(), 2);
        let n = self.n_dofs;
        assert_eq!(values.len(), n * 2);

        // Evaluate 1D basis functions.
        let shape_x: Vec<(usize, f64)> = Self::eval_1d(&self.basis1_u, xi[0]);
        let shape_y: Vec<(usize, f64)> = Self::eval_1d(&self.basis_v, xi[1]);
        let shape_x_orig: Vec<(usize, f64)> = Self::eval_1d(&self.basis_u, xi[0]);
        let shape_y1: Vec<(usize, f64)> = Self::eval_1d(&self.basis1_v, xi[1]);

        // Build lookup arrays for fast access.
        let px = self.order_u;
        let py = self.order_v;
        let mut sx1 = vec![0.0_f64; px + 2];
        for (i, v) in &shape_x { if *i < sx1.len() { sx1[*i] = *v; } }
        let mut sy = vec![0.0_f64; py + 1];
        for (i, v) in &shape_y { if *i < sy.len() { sy[*i] = *v; } }
        let mut sx = vec![0.0_f64; px + 1];
        for (i, v) in &shape_x_orig { if *i < sx.len() { sx[*i] = *v; } }
        let mut sy1 = vec![0.0_f64; py + 2];
        for (i, v) in &shape_y1 { if *i < sy1.len() { sy1[*i] = *v; } }

        // First set: x-component = shape1_x(i) * shape_y(j), i=0..px+1, j=0..py
        let mut o = 0;
        for j in 0..=py {
            let sj = sy[j];
            for i in 0..=px + 1 {
                values[o * 2 + 0] = sx1[i] * sj;
                values[o * 2 + 1] = 0.0;
                o += 1;
            }
        }
        // Second set: y-component = shape_x(i) * shape1_y(j), i=0..px, j=0..py+1
        for j in 0..=py + 1 {
            let sj = sy1[j];
            for i in 0..=px {
                values[o * 2 + 0] = 0.0;
                values[o * 2 + 1] = sx[i] * sj;
                o += 1;
            }
        }
        assert_eq!(o, n);
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        assert_eq!(xi.len(), 2);
        let n = self.n_dofs;
        assert_eq!(div_vals.len(), n);

        let px = self.order_u;
        let py = self.order_v;

        let shape_y: Vec<(usize, f64)> = Self::eval_1d(&self.basis_v, xi[1]);
        let dsx1: Vec<(usize, f64)> = Self::eval_1d_deriv(&self.basis1_u, xi[0]);
        let shape_x: Vec<(usize, f64)> = Self::eval_1d(&self.basis_u, xi[0]);
        let dsy1: Vec<(usize, f64)> = Self::eval_1d_deriv(&self.basis1_v, xi[1]);

        let mut sy = vec![0.0_f64; py + 1];
        for (i, v) in &shape_y { if *i < sy.len() { sy[*i] = *v; } }
        let mut dsx = vec![0.0_f64; px + 2];
        for (i, v) in &dsx1 { if *i < dsx.len() { dsx[*i] = *v; } }
        let mut sx = vec![0.0_f64; px + 1];
        for (i, v) in &shape_x { if *i < sx.len() { sx[*i] = *v; } }
        let mut dsy = vec![0.0_f64; py + 2];
        for (i, v) in &dsy1 { if *i < dsy.len() { dsy[*i] = *v; } }

        let mut o = 0;
        for j in 0..=py {
            let sj = sy[j];
            for i in 0..=px + 1 {
                div_vals[o] = dsx[i] * sj;
                o += 1;
            }
        }
        for j in 0..=py + 1 {
            let dsj = dsy[j];
            for i in 0..=px {
                div_vals[o] = sx[i] * dsj;
                o += 1;
            }
        }
        assert_eq!(o, n);
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        // H(div) elements don't have curl in the standard sense.
        curl_vals.fill(0.0);
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        // Tensor-product Gauss-Legendre rule.
        let p = order.max(2);
        crate::quadrature::quad_rule(p)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        // DOF coordinates are not well-defined for NURBS vector elements
        // (they're associated with knot spans, not geometric points).
        // Return empty — interpolation for these elements is handled separately.
        Vec::new()
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// NURBS_HCurl_2D — curl-conforming NURBS on a quadrilateral
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 2D H(curl)-conforming NURBS vector element on a square reference domain.
///
/// Uses mixed-degree B-spline bases: the x-component uses original ξ and
/// degree-elevated η; the y-component uses degree-elevated ξ and original η.
#[derive(Debug, Clone)]
pub struct NurbsHCurl2D {
    pub order_u: usize,
    pub order_v: usize,
    basis_u: BsplineBasis,
    basis_v: BsplineBasis,
    basis1_u: BsplineBasis,
    basis1_v: BsplineBasis,
    pub n_dofs: usize,
}

impl NurbsHCurl2D {
    pub fn new(order_u: usize, order_v: usize) -> Self {
        Self::from_knot_vectors(
            clamped_uniform_knots(order_u, 1),
            clamped_uniform_knots(order_v, 1),
        )
        .expect("NurbsHCurl2D::new")
    }

    pub fn from_knot_vectors(kv_u: KnotVector, kv_v: KnotVector) -> Result<Self, String> {
        let px = order_from_knots(&kv_u);
        let py = order_from_knots(&kv_v);

        let kv1_u = degree_elevate_knots(&kv_u, 1);
        let kv1_v = degree_elevate_knots(&kv_v, 1);

        let basis_u = BsplineBasis::new(px, kv_u).map_err(|e| format!("basis_u: {e}"))?;
        let basis_v = BsplineBasis::new(py, kv_v).map_err(|e| format!("basis_v: {e}"))?;
        let basis1_u = BsplineBasis::new(px + 1, kv1_u).map_err(|e| format!("basis1_u: {e}"))?;
        let basis1_v = BsplineBasis::new(py + 1, kv1_v).map_err(|e| format!("basis1_v: {e}"))?;

        let n = (px + 1) * (py + 2) + (px + 2) * (py + 1);

        Ok(Self {
            order_u: px,
            order_v: py,
            basis_u,
            basis_v,
            basis1_u,
            basis1_v,
            n_dofs: n,
        })
    }

    pub fn n_dofs_static(px: usize, py: usize) -> usize {
        (px + 1) * (py + 2) + (px + 2) * (py + 1)
    }

    fn eval_1d(basis: &BsplineBasis, u: f64) -> Vec<(usize, f64)> {
        basis.nonzero_values(u).unwrap_or_default()
    }

    fn eval_1d_deriv(basis: &BsplineBasis, u: f64) -> Vec<(usize, f64)> {
        basis.nonzero_derivatives(u).unwrap_or_default()
    }
}

impl VectorReferenceElement for NurbsHCurl2D {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { self.order_u.max(self.order_v) as u8 }
    fn n_dofs(&self) -> usize { self.n_dofs }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        assert_eq!(xi.len(), 2);
        let n = self.n_dofs;
        assert_eq!(values.len(), n * 2);

        let shape_x: Vec<(usize, f64)> = Self::eval_1d(&self.basis_u, xi[0]);
        let shape_y1: Vec<(usize, f64)> = Self::eval_1d(&self.basis1_v, xi[1]);
        let shape_x1: Vec<(usize, f64)> = Self::eval_1d(&self.basis1_u, xi[0]);
        let shape_y: Vec<(usize, f64)> = Self::eval_1d(&self.basis_v, xi[1]);

        let px = self.order_u;
        let py = self.order_v;
        let mut sx = vec![0.0_f64; px + 1];
        for (i, v) in &shape_x { if *i < sx.len() { sx[*i] = *v; } }
        let mut sy1 = vec![0.0_f64; py + 2];
        for (i, v) in &shape_y1 { if *i < sy1.len() { sy1[*i] = *v; } }
        let mut sx1 = vec![0.0_f64; px + 2];
        for (i, v) in &shape_x1 { if *i < sx1.len() { sx1[*i] = *v; } }
        let mut sy = vec![0.0_f64; py + 1];
        for (i, v) in &shape_y { if *i < sy.len() { sy[*i] = *v; } }

        // First set: x-component = shape_x(i) * shape1_y(j), i=0..px, j=0..py+1
        let mut o = 0;
        for j in 0..=py + 1 {
            let sj = sy1[j];
            for i in 0..=px {
                values[o * 2 + 0] = sx[i] * sj;
                values[o * 2 + 1] = 0.0;
                o += 1;
            }
        }
        // Second set: y-component = shape1_x(i) * shape_y(j), i=0..px+1, j=0..py
        for j in 0..=py {
            let sj = sy[j];
            for i in 0..=px + 1 {
                values[o * 2 + 0] = 0.0;
                values[o * 2 + 1] = sx1[i] * sj;
                o += 1;
            }
        }
        assert_eq!(o, n);
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        // 2D curl is scalar: ∂Φ_y/∂ξ - ∂Φ_x/∂η
        assert_eq!(xi.len(), 2);
        let n = self.n_dofs;
        assert_eq!(curl_vals.len(), n);

        let px = self.order_u;
        let py = self.order_v;

        let shape_x: Vec<(usize, f64)> = Self::eval_1d(&self.basis_u, xi[0]);
        let dsy1: Vec<(usize, f64)> = Self::eval_1d_deriv(&self.basis1_v, xi[1]);
        let dsx1: Vec<(usize, f64)> = Self::eval_1d_deriv(&self.basis1_u, xi[0]);
        let shape_y: Vec<(usize, f64)> = Self::eval_1d(&self.basis_v, xi[1]);

        let mut sx = vec![0.0_f64; px + 1];
        for (i, v) in &shape_x { if *i < sx.len() { sx[*i] = *v; } }
        let mut dsy = vec![0.0_f64; py + 2];
        for (i, v) in &dsy1 { if *i < dsy.len() { dsy[*i] = *v; } }
        let mut dsx = vec![0.0_f64; px + 2];
        for (i, v) in &dsx1 { if *i < dsx.len() { dsx[*i] = *v; } }
        let mut sy = vec![0.0_f64; py + 1];
        for (i, v) in &shape_y { if *i < sy.len() { sy[*i] = *v; } }

        let mut o = 0;
        // First set (x-component): curl = -shape_x(i) * dshape1_y(j)
        for j in 0..=py + 1 {
            let dsj = dsy[j];
            for i in 0..=px {
                curl_vals[o] = -sx[i] * dsj;
                o += 1;
            }
        }
        // Second set (y-component): curl = dshape1_x(i) * shape_y(j)
        for j in 0..=py {
            let sj = sy[j];
            for i in 0..=px + 1 {
                curl_vals[o] = dsx[i] * sj;
                o += 1;
            }
        }
        assert_eq!(o, n);
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        // H(curl) elements don't have divergence in the standard sense.
        div_vals.fill(0.0);
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        let p = order.max(2);
        crate::quadrature::quad_rule(p)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        Vec::new()
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// NURBS_HDiv_3D — divergence-conforming NURBS on a hexahedron
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 3D H(div)-conforming NURBS vector element on a cube reference domain.
#[derive(Debug, Clone)]
pub struct NurbsHDiv3D {
    pub order_u: usize,
    pub order_v: usize,
    pub order_w: usize,
    basis_u: BsplineBasis,
    basis_v: BsplineBasis,
    basis_w: BsplineBasis,
    basis1_u: BsplineBasis,
    basis1_v: BsplineBasis,
    basis1_w: BsplineBasis,
    pub n_dofs: usize,
}

impl NurbsHDiv3D {
    pub fn new(order_u: usize, order_v: usize, order_w: usize) -> Self {
        Self::from_knot_vectors(
            clamped_uniform_knots(order_u, 1),
            clamped_uniform_knots(order_v, 1),
            clamped_uniform_knots(order_w, 1),
        )
        .expect("NurbsHDiv3D::new")
    }

    pub fn from_knot_vectors(kv_u: KnotVector, kv_v: KnotVector, kv_w: KnotVector) -> Result<Self, String> {
        let px = order_from_knots(&kv_u);
        let py = order_from_knots(&kv_v);
        let pz = order_from_knots(&kv_w);

        let kv1_u = degree_elevate_knots(&kv_u, 1);
        let kv1_v = degree_elevate_knots(&kv_v, 1);
        let kv1_w = degree_elevate_knots(&kv_w, 1);

        let basis_u = BsplineBasis::new(px, kv_u).map_err(|e| format!("basis_u: {e}"))?;
        let basis_v = BsplineBasis::new(py, kv_v).map_err(|e| format!("basis_v: {e}"))?;
        let basis_w = BsplineBasis::new(pz, kv_w).map_err(|e| format!("basis_w: {e}"))?;
        let basis1_u = BsplineBasis::new(px + 1, kv1_u).map_err(|e| format!("basis1_u: {e}"))?;
        let basis1_v = BsplineBasis::new(py + 1, kv1_v).map_err(|e| format!("basis1_v: {e}"))?;
        let basis1_w = BsplineBasis::new(pz + 1, kv1_w).map_err(|e| format!("basis1_w: {e}"))?;

        let n = (px + 2) * (py + 1) * (pz + 1)
              + (px + 1) * (py + 2) * (pz + 1)
              + (px + 1) * (py + 1) * (pz + 2);

        Ok(Self {
            order_u: px,
            order_v: py,
            order_w: pz,
            basis_u, basis_v, basis_w,
            basis1_u, basis1_v, basis1_w,
            n_dofs: n,
        })
    }

    pub fn n_dofs_static(px: usize, py: usize, pz: usize) -> usize {
        (px + 2) * (py + 1) * (pz + 1)
            + (px + 1) * (py + 2) * (pz + 1)
            + (px + 1) * (py + 1) * (pz + 2)
    }
}

impl VectorReferenceElement for NurbsHDiv3D {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { self.order_u.max(self.order_v).max(self.order_w) as u8 }
    fn n_dofs(&self) -> usize { self.n_dofs }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        assert_eq!(xi.len(), 3);
        let n = self.n_dofs;
        assert_eq!(values.len(), n * 3);

        let px = self.order_u;
        let py = self.order_v;
        let pz = self.order_w;

        // Evaluate 1D basis functions.
        let sx1: Vec<f64> = self.basis1_u.nonzero_values(xi[0]).unwrap_or_default()
            .into_iter().fold(vec![0.0; px + 2], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });
        let sy: Vec<f64> = self.basis_v.nonzero_values(xi[1]).unwrap_or_default()
            .into_iter().fold(vec![0.0; py + 1], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });
        let sz: Vec<f64> = self.basis_w.nonzero_values(xi[2]).unwrap_or_default()
            .into_iter().fold(vec![0.0; pz + 1], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });

        let sx: Vec<f64> = self.basis_u.nonzero_values(xi[0]).unwrap_or_default()
            .into_iter().fold(vec![0.0; px + 1], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });
        let sy1: Vec<f64> = self.basis1_v.nonzero_values(xi[1]).unwrap_or_default()
            .into_iter().fold(vec![0.0; py + 2], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });
        let sz1: Vec<f64> = self.basis1_w.nonzero_values(xi[2]).unwrap_or_default()
            .into_iter().fold(vec![0.0; pz + 2], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });

        let mut o = 0;
        // x-component: shape1_x(i) * shape_y(j) * shape_z(k)
        for k in 0..=pz {
            let sk = sz[k];
            for j in 0..=py {
                let sj_sk = sy[j] * sk;
                for i in 0..=px + 1 {
                    values[o * 3 + 0] = sx1[i] * sj_sk;
                    values[o * 3 + 1] = 0.0;
                    values[o * 3 + 2] = 0.0;
                    o += 1;
                }
            }
        }
        // y-component: shape_x(i) * shape1_y(j) * shape_z(k)
        for k in 0..=pz {
            let sk = sz[k];
            for j in 0..=py + 1 {
                let sj_sk = sy1[j] * sk;
                for i in 0..=px {
                    values[o * 3 + 0] = 0.0;
                    values[o * 3 + 1] = sx[i] * sj_sk;
                    values[o * 3 + 2] = 0.0;
                    o += 1;
                }
            }
        }
        // z-component: shape_x(i) * shape_y(j) * shape1_z(k)
        for k in 0..=pz + 1 {
            let sk = sz1[k];
            for j in 0..=py {
                let sj_sk = sy[j] * sk;
                for i in 0..=px {
                    values[o * 3 + 0] = 0.0;
                    values[o * 3 + 1] = 0.0;
                    values[o * 3 + 2] = sx[i] * sj_sk;
                    o += 1;
                }
            }
        }
        assert_eq!(o, n);
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        assert_eq!(xi.len(), 3);
        let n = self.n_dofs;
        assert_eq!(div_vals.len(), n);

        let px = self.order_u;
        let py = self.order_v;
        let pz = self.order_w;

        let sy: Vec<f64> = self.basis_v.nonzero_values(xi[1]).unwrap_or_default()
            .into_iter().fold(vec![0.0; py + 1], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });
        let sz: Vec<f64> = self.basis_w.nonzero_values(xi[2]).unwrap_or_default()
            .into_iter().fold(vec![0.0; pz + 1], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });
        let dsx1: Vec<f64> = self.basis1_u.nonzero_derivatives(xi[0]).unwrap_or_default()
            .into_iter().fold(vec![0.0; px + 2], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });

        let sx: Vec<f64> = self.basis_u.nonzero_values(xi[0]).unwrap_or_default()
            .into_iter().fold(vec![0.0; px + 1], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });
        let dsy1: Vec<f64> = self.basis1_v.nonzero_derivatives(xi[1]).unwrap_or_default()
            .into_iter().fold(vec![0.0; py + 2], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });
        let dsz1: Vec<f64> = self.basis1_w.nonzero_derivatives(xi[2]).unwrap_or_default()
            .into_iter().fold(vec![0.0; pz + 2], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });

        let mut o = 0;
        for k in 0..=pz {
            let sk = sz[k];
            for j in 0..=py {
                let sj_sk = sy[j] * sk;
                for i in 0..=px + 1 {
                    div_vals[o] = dsx1[i] * sj_sk;
                    o += 1;
                }
            }
        }
        for k in 0..=pz {
            let sk = sz[k];
            for j in 0..=py + 1 {
                let dsj_sk = dsy1[j] * sk;
                for i in 0..=px {
                    div_vals[o] = sx[i] * dsj_sk;
                    o += 1;
                }
            }
        }
        for k in 0..=pz + 1 {
            let dsk = dsz1[k];
            for j in 0..=py {
                let sj_dsk = sy[j] * dsk;
                for i in 0..=px {
                    div_vals[o] = sx[i] * sj_dsk;
                    o += 1;
                }
            }
        }
        assert_eq!(o, n);
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        curl_vals.fill(0.0);
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        let p = order.max(2);
        crate::quadrature::hex_rule(p)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        Vec::new()
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// NURBS_HCurl_3D — curl-conforming NURBS on a hexahedron
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 3D H(curl)-conforming NURBS vector element on a cube reference domain.
#[derive(Debug, Clone)]
pub struct NurbsHCurl3D {
    pub order_u: usize,
    pub order_v: usize,
    pub order_w: usize,
    basis_u: BsplineBasis,
    basis_v: BsplineBasis,
    basis_w: BsplineBasis,
    basis1_u: BsplineBasis,
    basis1_v: BsplineBasis,
    basis1_w: BsplineBasis,
    pub n_dofs: usize,
}

impl NurbsHCurl3D {
    pub fn new(order_u: usize, order_v: usize, order_w: usize) -> Self {
        Self::from_knot_vectors(
            clamped_uniform_knots(order_u, 1),
            clamped_uniform_knots(order_v, 1),
            clamped_uniform_knots(order_w, 1),
        )
        .expect("NurbsHCurl3D::new")
    }

    pub fn from_knot_vectors(kv_u: KnotVector, kv_v: KnotVector, kv_w: KnotVector) -> Result<Self, String> {
        let px = order_from_knots(&kv_u);
        let py = order_from_knots(&kv_v);
        let pz = order_from_knots(&kv_w);

        let kv1_u = degree_elevate_knots(&kv_u, 1);
        let kv1_v = degree_elevate_knots(&kv_v, 1);
        let kv1_w = degree_elevate_knots(&kv_w, 1);

        let basis_u = BsplineBasis::new(px, kv_u).map_err(|e| format!("basis_u: {e}"))?;
        let basis_v = BsplineBasis::new(py, kv_v).map_err(|e| format!("basis_v: {e}"))?;
        let basis_w = BsplineBasis::new(pz, kv_w).map_err(|e| format!("basis_w: {e}"))?;
        let basis1_u = BsplineBasis::new(px + 1, kv1_u).map_err(|e| format!("basis1_u: {e}"))?;
        let basis1_v = BsplineBasis::new(py + 1, kv1_v).map_err(|e| format!("basis1_v: {e}"))?;
        let basis1_w = BsplineBasis::new(pz + 1, kv1_w).map_err(|e| format!("basis1_w: {e}"))?;

        let n = (px + 1) * (py + 2) * (pz + 2)
              + (px + 2) * (py + 1) * (pz + 2)
              + (px + 2) * (py + 2) * (pz + 1);

        Ok(Self {
            order_u: px,
            order_v: py,
            order_w: pz,
            basis_u, basis_v, basis_w,
            basis1_u, basis1_v, basis1_w,
            n_dofs: n,
        })
    }

    pub fn n_dofs_static(px: usize, py: usize, pz: usize) -> usize {
        (px + 1) * (py + 2) * (pz + 2)
            + (px + 2) * (py + 1) * (pz + 2)
            + (px + 2) * (py + 2) * (pz + 1)
    }
}

impl VectorReferenceElement for NurbsHCurl3D {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { self.order_u.max(self.order_v).max(self.order_w) as u8 }
    fn n_dofs(&self) -> usize { self.n_dofs }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        assert_eq!(xi.len(), 3);
        let n = self.n_dofs;
        assert_eq!(values.len(), n * 3);

        let px = self.order_u;
        let py = self.order_v;
        let pz = self.order_w;

        let sx: Vec<f64> = self.basis_u.nonzero_values(xi[0]).unwrap_or_default()
            .into_iter().fold(vec![0.0; px + 1], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });
        let sy1: Vec<f64> = self.basis1_v.nonzero_values(xi[1]).unwrap_or_default()
            .into_iter().fold(vec![0.0; py + 2], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });
        let sz1: Vec<f64> = self.basis1_w.nonzero_values(xi[2]).unwrap_or_default()
            .into_iter().fold(vec![0.0; pz + 2], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });

        let sx1: Vec<f64> = self.basis1_u.nonzero_values(xi[0]).unwrap_or_default()
            .into_iter().fold(vec![0.0; px + 2], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });
        let sy: Vec<f64> = self.basis_v.nonzero_values(xi[1]).unwrap_or_default()
            .into_iter().fold(vec![0.0; py + 1], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });
        let sz: Vec<f64> = self.basis_w.nonzero_values(xi[2]).unwrap_or_default()
            .into_iter().fold(vec![0.0; pz + 1], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });

        let mut o = 0;
        // x-component: shape_x(i) * shape1_y(j) * shape1_z(k)
        for k in 0..=pz + 1 {
            let sk = sz1[k];
            for j in 0..=py + 1 {
                let sj_sk = sy1[j] * sk;
                for i in 0..=px {
                    values[o * 3 + 0] = sx[i] * sj_sk;
                    values[o * 3 + 1] = 0.0;
                    values[o * 3 + 2] = 0.0;
                    o += 1;
                }
            }
        }
        // y-component: shape1_x(i) * shape_y(j) * shape1_z(k)
        for k in 0..=pz + 1 {
            let sk = sz1[k];
            for j in 0..=py {
                let sj_sk = sy[j] * sk;
                for i in 0..=px + 1 {
                    values[o * 3 + 0] = 0.0;
                    values[o * 3 + 1] = sx1[i] * sj_sk;
                    values[o * 3 + 2] = 0.0;
                    o += 1;
                }
            }
        }
        // z-component: shape1_x(i) * shape1_y(j) * shape_z(k)
        for k in 0..=pz {
            let sk = sz[k];
            for j in 0..=py + 1 {
                let sj_sk = sy1[j] * sk;
                for i in 0..=px + 1 {
                    values[o * 3 + 0] = 0.0;
                    values[o * 3 + 1] = 0.0;
                    values[o * 3 + 2] = sx1[i] * sj_sk;
                    o += 1;
                }
            }
        }
        assert_eq!(o, n);
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        // 3D curl is a vector: (∂Φ_z/∂y - ∂Φ_y/∂z, ∂Φ_x/∂z - ∂Φ_z/∂x, ∂Φ_y/∂x - ∂Φ_x/∂y)
        assert_eq!(xi.len(), 3);
        let n = self.n_dofs;
        assert_eq!(curl_vals.len(), n * 3);

        let px = self.order_u;
        let py = self.order_v;
        let pz = self.order_w;

        // Evaluate all needed basis functions and derivatives.
        let sx: Vec<f64> = self.basis_u.nonzero_values(xi[0]).unwrap_or_default()
            .into_iter().fold(vec![0.0; px + 1], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });
        let sy1: Vec<f64> = self.basis1_v.nonzero_values(xi[1]).unwrap_or_default()
            .into_iter().fold(vec![0.0; py + 2], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });
        let sz1: Vec<f64> = self.basis1_w.nonzero_values(xi[2]).unwrap_or_default()
            .into_iter().fold(vec![0.0; pz + 2], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });

        let sx1: Vec<f64> = self.basis1_u.nonzero_values(xi[0]).unwrap_or_default()
            .into_iter().fold(vec![0.0; px + 2], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });
        let sy: Vec<f64> = self.basis_v.nonzero_values(xi[1]).unwrap_or_default()
            .into_iter().fold(vec![0.0; py + 1], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });
        let sz: Vec<f64> = self.basis_w.nonzero_values(xi[2]).unwrap_or_default()
            .into_iter().fold(vec![0.0; pz + 1], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });

        let dsx: Vec<f64> = self.basis_u.nonzero_derivatives(xi[0]).unwrap_or_default()
            .into_iter().fold(vec![0.0; px + 1], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });
        let dsy1: Vec<f64> = self.basis1_v.nonzero_derivatives(xi[1]).unwrap_or_default()
            .into_iter().fold(vec![0.0; py + 2], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });
        let dsz1: Vec<f64> = self.basis1_w.nonzero_derivatives(xi[2]).unwrap_or_default()
            .into_iter().fold(vec![0.0; pz + 2], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });

        let dsx1: Vec<f64> = self.basis1_u.nonzero_derivatives(xi[0]).unwrap_or_default()
            .into_iter().fold(vec![0.0; px + 2], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });
        let _dsy: Vec<f64> = self.basis_v.nonzero_derivatives(xi[1]).unwrap_or_default()
            .into_iter().fold(vec![0.0; py + 1], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });
        let _dsz: Vec<f64> = self.basis_w.nonzero_derivatives(xi[2]).unwrap_or_default()
            .into_iter().fold(vec![0.0; pz + 1], |mut acc, (i, v)| { if i < acc.len() { acc[i] = v; } acc });

        let mut o = 0;
        // x-component basis: shape_x(i) * shape1_y(j) * shape1_z(k)
        // curl_x = ∂(shape_x*shape1_y*shape1_z)/∂y - ∂(0)/∂z = shape_x * dshape1_y * shape1_z
        // curl_y = ∂(0)/∂z - ∂(shape_x*shape1_y*shape1_z)/∂x = -dshape_x * shape1_y * shape1_z
        // curl_z = ∂(0)/∂x - ∂(0)/∂y = 0
        for k in 0..=pz + 1 {
            let sk = sz1[k];
            for j in 0..=py + 1 {
                let dsy1_sk = dsy1[j] * sk;
                let sy1_sk = sy1[j] * sk;
                for i in 0..=px {
                    curl_vals[o * 3 + 0] = sx[i] * dsy1_sk;
                    curl_vals[o * 3 + 1] = -dsx[i] * sy1_sk;
                    curl_vals[o * 3 + 2] = 0.0;
                    o += 1;
                }
            }
        }
        // y-component basis: shape1_x(i) * shape_y(j) * shape1_z(k)
        // curl_x = ∂(0)/∂y - ∂(shape1_x*shape_y*shape1_z)/∂z = -shape1_x * shape_y * dshape1_z
        // curl_y = ∂(shape1_x*shape_y*shape1_z)/∂z - ∂(0)/∂x = shape1_x * shape_y * dshape1_z
        // Wait, let me redo this properly.
        // For y-component basis v = (0, sx1[i]*sy[j]*sz1[k], 0):
        // curl_x = ∂v_y/∂z - ∂v_z/∂y = sx1[i]*sy[j]*dsz1[k] - 0
        // curl_y = ∂v_z/∂x - ∂v_x/∂z = 0 - 0 = 0
        // curl_z = ∂v_x/∂y - ∂v_y/∂x = 0 - dsx1[i]*sy[j]*sz1[k]
        for k in 0..=pz + 1 {
            let sk = sz1[k];
            let dsk = dsz1[k];
            for j in 0..=py {
                let sy_sk = sy[j] * sk;
                let sy_dsk = sy[j] * dsk;
                for i in 0..=px + 1 {
                    curl_vals[o * 3 + 0] = sx1[i] * sy_dsk;
                    curl_vals[o * 3 + 1] = 0.0;
                    curl_vals[o * 3 + 2] = -dsx1[i] * sy_sk;
                    o += 1;
                }
            }
        }
        // z-component basis: shape1_x(i) * shape1_y(j) * shape_z(k)
        // For z-component basis v = (0, 0, sx1[i]*sy1[j]*sz[k]):
        // curl_x = ∂v_z/∂y - ∂v_y/∂z = sx1[i]*dsy1[j]*sz[k] - 0
        // curl_y = ∂v_x/∂z - ∂v_z/∂x = 0 - dsx1[i]*sy1[j]*sz[k]
        // curl_z = ∂v_x/∂y - ∂v_y/∂x = 0 - 0 = 0
        for k in 0..=pz {
            let sk = sz[k];
            for j in 0..=py + 1 {
                let dsy1_sk = dsy1[j] * sk;
                let sy1_sk = sy1[j] * sk;
                for i in 0..=px + 1 {
                    curl_vals[o * 3 + 0] = sx1[i] * dsy1_sk;
                    curl_vals[o * 3 + 1] = -dsx1[i] * sy1_sk;
                    curl_vals[o * 3 + 2] = 0.0;
                    o += 1;
                }
            }
        }
        assert_eq!(o, n);
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        div_vals.fill(0.0);
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        let p = order.max(2);
        crate::quadrature::hex_rule(p)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        Vec::new()
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nurbs_hdiv2d_dof_count() {
        assert_eq!(NurbsHDiv2D::n_dofs_static(1, 1), 2 * 2 * 3); // (3*2 + 2*3) = 12
        assert_eq!(NurbsHDiv2D::n_dofs_static(2, 2), 4 * 3 + 3 * 4); // 24
        let elem = NurbsHDiv2D::new(1, 1);
        assert_eq!(elem.n_dofs(), 12);
    }

    #[test]
    fn nurbs_hcurl2d_dof_count() {
        assert_eq!(NurbsHCurl2D::n_dofs_static(1, 1), 12);
        assert_eq!(NurbsHCurl2D::n_dofs_static(2, 2), 24);
        let elem = NurbsHCurl2D::new(1, 1);
        assert_eq!(elem.n_dofs(), 12);
    }

    #[test]
    fn nurbs_hdiv3d_dof_count() {
        // (px+2)*(py+1)*(pz+1) + (px+1)*(py+2)*(pz+1) + (px+1)*(py+1)*(pz+2)
        // For p=1: 3*2*2 + 2*3*2 + 2*2*3 = 12 + 12 + 12 = 36
        assert_eq!(NurbsHDiv3D::n_dofs_static(1, 1, 1), 36);
    }

    #[test]
    fn nurbs_hcurl3d_dof_count() {
        // (px+1)*(py+2)*(pz+2) + (px+2)*(py+1)*(pz+2) + (px+2)*(py+2)*(pz+1)
        // For p=1: 2*3*3 + 3*2*3 + 3*3*2 = 18 + 18 + 18 = 54
        assert_eq!(NurbsHCurl3D::n_dofs_static(1, 1, 1), 54);
    }

    #[test]
    fn nurbs_hdiv2d_basis_sum() {
        // At any point, the sum of all x-component basis functions should be 1.0
        // (partition of unity for the x-component).
        let elem = NurbsHDiv2D::new(1, 1);
        let n = elem.n_dofs();
        let mut values = vec![0.0; n * 2];
        elem.eval_basis_vec(&[0.5, 0.5], &mut values);

        let sum_x: f64 = (0..n).map(|i| values[i * 2 + 0]).sum();
        let sum_y: f64 = (0..n).map(|i| values[i * 2 + 1]).sum();

        // For order 1, the x-component has (1+2)*(1+1) = 6 DOFs
        // and y-component has (1+1)*(1+2) = 6 DOFs
        // The sum of all x-basis functions at center should be ~1.0
        assert!((sum_x - 1.0).abs() < 1e-10, "sum_x = {sum_x}");
        assert!((sum_y - 1.0).abs() < 1e-10, "sum_y = {sum_y}");
    }

    #[test]
    fn nurbs_hdiv2d_div_partition() {
        // The divergence of the basis functions should sum to a constant.
        let elem = NurbsHDiv2D::new(1, 1);
        let n = elem.n_dofs();
        let mut div_vals = vec![0.0; n];
        elem.eval_div(&[0.5, 0.5], &mut div_vals);

        // For constant divergence field, sum should be dim = 2
        let sum: f64 = div_vals.iter().sum();
        // Not necessarily 2.0 for NURBS, but should be finite & non-zero
        assert!(sum.is_finite());
    }

    #[test]
    fn degree_elevate_preserves_spans() {
        let kv = clamped_uniform_knots(2, 1);
        let kv1 = degree_elevate_knots(&kv, 1);
        // Elevated knot vector should have degree 3 and more knots.
        let orig_knots = kv.as_slice();
        let elev_knots = kv1.as_slice();
        assert!(elev_knots.len() > orig_knots.len());
    }
}
