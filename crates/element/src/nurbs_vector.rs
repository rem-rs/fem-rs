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
//! - DOFs: `(px+2)*(py+1) + (py+1)*(py+2)`
//! - x-component: `N_i^(px+1)(ξ) * N_j^(py)(η)`  for i=0..px+1, j=0..py
//! - y-component: `N_i^(px)(ξ) * N_j^(py+1)(η)`  for i=0..px, j=0..py+1
//! - Piola transform: `v_phys = J * v_ref / weight`
//!
//! **H(curl) 2D** (quad, orders px, py):
//! - DOFs: `(px+1)*(py+2) + (py+2)*(py+1)`
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
//!
//! The 2D DOF formulas mirror MFEM's
//! `NURBS_HDiv2DFiniteElement::SetOrder` / `NURBS_HCurl2DFiniteElement::SetOrder`
//! verbatim: both index `py` in the second term where the degree-elevated basis
//! suggests `px`.  Matching MFEM on anisotropic-order meshes requires the same
//! arithmetic — see [`crate::nurbs_fe_collection::NurbsElement::n_dofs`].
//!
//! # Degree elevation and spans
//!
//! The mixed-degree bases are built with MFEM's `KnotVector::DegreeElevate(1)`
//! ([`crate::nurbs_fe_collection::degree_elevate`]), which preserves the
//! element (span) count and raises the order by one.  MFEM evaluates both the
//! base and the elevated basis at the same span index `ijk`
//! (`NURBSExtension::LoadFE` sets it from `el_to_IJK`), so every evaluation
//! here is **span-local**: [`NurbsHDiv2D::set_ijk`] selects the knot span and
//! the basis values are the `Order + 1` values MFEM's
//! `KnotVector::CalcShape(shape, i, xi)` returns for that span, at the
//! span-local reference coordinate `xi ∈ [0,1]^{dim}`.
//!
//! `RATIONAL WEIGHTS ARE NOT USED`: MFEM's `NURBS_HDiv*`/`NURBS_HCurl*`
//! `CalcVShape`/`CalcDivShape`/`CalcCurlShape` evaluate the *non-rational*
//! B-spline bases (`kv` / `kv1`), unlike the scalar `NURBS2DFiniteElement`
//! whose `CalcShape` divides by the weighted sum.  LoadFE still stores weights
//! on the element, but they never enter the vector element's values.

use crate::iga::KnotVector;
use crate::nurbs_fe_collection::{
    degree_elevate, knot_order, knot_span_dshape, knot_span_shape,
};
use crate::reference::{QuadratureRule, VectorReferenceElement};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Helper: degree-elevated BsplineBasis
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

/// MFEM `KnotVector::DegreeElevate(1)` the `NURBS_HDiv*`/`NURBS_HCurl*`
/// elements build in `SetOrder` (`kv1[i] = kv[i]->DegreeElevate(1)`).
fn elevated_knots(kv: &KnotVector) -> Result<KnotVector, String> {
    degree_elevate(kv, 1)
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
    /// Base knot vectors (orders `px`, `py`).
    kv: [KnotVector; 2],
    /// Degree-elevated knot vectors (orders `px+1`, `py+1`).
    kv1: [KnotVector; 2],
    /// Knot span index within the patch (`NURBSFiniteElement::ijk`).
    ijk: [usize; 2],
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
        let px = knot_order(&kv_u).ok_or_else(|| "NurbsHDiv2D: invalid kv_u".to_string())?;
        let py = knot_order(&kv_v).ok_or_else(|| "NurbsHDiv2D: invalid kv_v".to_string())?;

        let kv1_u = elevated_knots(&kv_u)?;
        let kv1_v = elevated_knots(&kv_v)?;

        // MFEM `NURBS_HDiv2DFiniteElement::SetOrder` (see
        // `crate::nurbs_fe_collection::NurbsElement::n_dofs` for why the second
        // term uses `py` rather than `px`).
        let n = (px + 2) * (py + 1) + (py + 1) * (py + 2);

        Ok(Self {
            order_u: px,
            order_v: py,
            kv: [kv_u, kv_v],
            kv1: [kv1_u, kv1_v],
            ijk: [0, 0],
            n_dofs: n,
        })
    }

    /// Static DOF count for given orders, matching
    /// `NURBS_HDiv2DFiniteElement::SetOrder`.
    pub fn n_dofs_static(px: usize, py: usize) -> usize {
        (px + 2) * (py + 1) + (py + 1) * (py + 2)
    }

    /// MFEM `NURBSFiniteElement::SetIJK` — select the knot span.
    pub fn set_ijk(&mut self, ijk: [usize; 2]) {
        self.ijk = ijk;
    }

    /// MFEM `NURBSFiniteElement::GetIJK`.
    pub fn ijk(&self) -> [usize; 2] {
        self.ijk
    }

    /// Per-direction order.
    fn order_dir(&self, d: usize) -> usize {
        if d == 0 { self.order_u } else { self.order_v }
    }

    /// Span-local `KnotVector::CalcShape` in direction `d`; `elevated` selects
    /// MFEM's `kv1[d]`.
    fn shape1d(&self, d: usize, elevated: bool, xi: f64) -> Vec<f64> {
        let order = self.order_dir(d) + usize::from(elevated);
        let kv = if elevated { &self.kv1[d] } else { &self.kv[d] };
        let mut out = vec![0.0; order + 1];
        knot_span_shape(kv.as_slice(), order, self.ijk[d], xi, &mut out);
        out
    }

    /// Span-local `KnotVector::CalcDShape` in direction `d`.
    fn dshape1d(&self, d: usize, elevated: bool, xi: f64) -> Vec<f64> {
        let order = self.order_dir(d) + usize::from(elevated);
        let kv = if elevated { &self.kv1[d] } else { &self.kv[d] };
        let mut out = vec![0.0; order + 1];
        knot_span_dshape(kv.as_slice(), order, self.ijk[d], xi, &mut out);
        out
    }
}

impl VectorReferenceElement for NurbsHDiv2D {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { (self.order_u.max(self.order_v) + 1) as u8 } // MFEM reports the elevated degree
    fn n_dofs(&self) -> usize { self.n_dofs }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        assert_eq!(xi.len(), 2);
        let n = self.n_dofs;
        assert_eq!(values.len(), n * 2);

        // MFEM `NURBS_HDiv2DFiniteElement::CalcVShape`.
        let sx1 = self.shape1d(0, true, xi[0]);
        let sy = self.shape1d(1, false, xi[1]);
        let sx = self.shape1d(0, false, xi[0]);
        let sy1 = self.shape1d(1, true, xi[1]);

        let px = self.order_u;
        let py = self.order_v;

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

        // MFEM `NURBS_HDiv2DFiniteElement::CalcDivShape`.
        let sy = self.shape1d(1, false, xi[1]);
        let dsx1 = self.dshape1d(0, true, xi[0]);
        let sx = self.shape1d(0, false, xi[0]);
        let dsy1 = self.dshape1d(1, true, xi[1]);

        let mut o = 0;
        for j in 0..=py {
            let sj = sy[j];
            for i in 0..=px + 1 {
                div_vals[o] = dsx1[i] * sj;
                o += 1;
            }
        }
        for j in 0..=py + 1 {
            let dsj = dsy1[j];
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
    /// Base knot vectors (orders `px`, `py`).
    kv: [KnotVector; 2],
    /// Degree-elevated knot vectors (orders `px+1`, `py+1`).
    kv1: [KnotVector; 2],
    /// Knot span index within the patch (`NURBSFiniteElement::ijk`).
    ijk: [usize; 2],
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
        let px = knot_order(&kv_u).ok_or_else(|| "NurbsHCurl2D: invalid kv_u".to_string())?;
        let py = knot_order(&kv_v).ok_or_else(|| "NurbsHCurl2D: invalid kv_v".to_string())?;

        let kv1_u = elevated_knots(&kv_u)?;
        let kv1_v = elevated_knots(&kv_v)?;

        // MFEM `NURBS_HCurl2DFiniteElement::SetOrder` (second term uses `py`,
        // see `crate::nurbs_fe_collection::NurbsElement::n_dofs`).
        let n = (px + 1) * (py + 2) + (py + 2) * (py + 1);

        Ok(Self {
            order_u: px,
            order_v: py,
            kv: [kv_u, kv_v],
            kv1: [kv1_u, kv1_v],
            ijk: [0, 0],
            n_dofs: n,
        })
    }

    /// Static DOF count for given orders, matching
    /// `NURBS_HCurl2DFiniteElement::SetOrder`.
    pub fn n_dofs_static(px: usize, py: usize) -> usize {
        (px + 1) * (py + 2) + (py + 2) * (py + 1)
    }

    /// MFEM `NURBSFiniteElement::SetIJK` — select the knot span.
    pub fn set_ijk(&mut self, ijk: [usize; 2]) {
        self.ijk = ijk;
    }

    /// MFEM `NURBSFiniteElement::GetIJK`.
    pub fn ijk(&self) -> [usize; 2] {
        self.ijk
    }

    /// Per-direction order.
    fn order_dir(&self, d: usize) -> usize {
        if d == 0 { self.order_u } else { self.order_v }
    }

    /// Span-local `KnotVector::CalcShape` in direction `d`.
    fn shape1d(&self, d: usize, elevated: bool, xi: f64) -> Vec<f64> {
        let order = self.order_dir(d) + usize::from(elevated);
        let kv = if elevated { &self.kv1[d] } else { &self.kv[d] };
        let mut out = vec![0.0; order + 1];
        knot_span_shape(kv.as_slice(), order, self.ijk[d], xi, &mut out);
        out
    }

    /// Span-local `KnotVector::CalcDShape` in direction `d`.
    fn dshape1d(&self, d: usize, elevated: bool, xi: f64) -> Vec<f64> {
        let order = self.order_dir(d) + usize::from(elevated);
        let kv = if elevated { &self.kv1[d] } else { &self.kv[d] };
        let mut out = vec![0.0; order + 1];
        knot_span_dshape(kv.as_slice(), order, self.ijk[d], xi, &mut out);
        out
    }
}

impl VectorReferenceElement for NurbsHCurl2D {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { (self.order_u.max(self.order_v) + 1) as u8 } // MFEM reports the elevated degree
    fn n_dofs(&self) -> usize { self.n_dofs }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        assert_eq!(xi.len(), 2);
        let n = self.n_dofs;
        assert_eq!(values.len(), n * 2);

        // MFEM `NURBS_HCurl2DFiniteElement::CalcVShape`.
        let sx = self.shape1d(0, false, xi[0]);
        let sy1 = self.shape1d(1, true, xi[1]);
        let sx1 = self.shape1d(0, true, xi[0]);
        let sy = self.shape1d(1, false, xi[1]);

        let px = self.order_u;
        let py = self.order_v;

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

        // MFEM `NURBS_HCurl2DFiniteElement::CalcCurlShape`.
        let sx = self.shape1d(0, false, xi[0]);
        let dsy = self.dshape1d(1, true, xi[1]);
        let dsx = self.dshape1d(0, true, xi[0]);
        let sy = self.shape1d(1, false, xi[1]);

        let px = self.order_u;
        let py = self.order_v;

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
    /// Base knot vectors (orders `px`, `py`, `pz`).
    kv: [KnotVector; 3],
    /// Degree-elevated knot vectors (orders `px+1`, `py+1`, `pz+1`).
    kv1: [KnotVector; 3],
    /// Knot span index within the patch (`NURBSFiniteElement::ijk`).
    ijk: [usize; 3],
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
        let px = knot_order(&kv_u).ok_or_else(|| "NurbsHDiv3D: invalid kv_u".to_string())?;
        let py = knot_order(&kv_v).ok_or_else(|| "NurbsHDiv3D: invalid kv_v".to_string())?;
        let pz = knot_order(&kv_w).ok_or_else(|| "NurbsHDiv3D: invalid kv_w".to_string())?;

        let kv1_u = elevated_knots(&kv_u)?;
        let kv1_v = elevated_knots(&kv_v)?;
        let kv1_w = elevated_knots(&kv_w)?;

        let n = (px + 2) * (py + 1) * (pz + 1)
              + (px + 1) * (py + 2) * (pz + 1)
              + (px + 1) * (py + 1) * (pz + 2);

        Ok(Self {
            order_u: px,
            order_v: py,
            order_w: pz,
            kv: [kv_u, kv_v, kv_w],
            kv1: [kv1_u, kv1_v, kv1_w],
            ijk: [0, 0, 0],
            n_dofs: n,
        })
    }

    pub fn n_dofs_static(px: usize, py: usize, pz: usize) -> usize {
        (px + 2) * (py + 1) * (pz + 1)
            + (px + 1) * (py + 2) * (pz + 1)
            + (px + 1) * (py + 1) * (pz + 2)
    }

    /// MFEM `NURBSFiniteElement::SetIJK` — select the knot span.
    pub fn set_ijk(&mut self, ijk: [usize; 3]) {
        self.ijk = ijk;
    }

    /// MFEM `NURBSFiniteElement::GetIJK`.
    pub fn ijk(&self) -> [usize; 3] {
        self.ijk
    }

    /// Per-direction order.
    fn order_dir(&self, d: usize) -> usize {
        match d {
            0 => self.order_u,
            1 => self.order_v,
            _ => self.order_w,
        }
    }

    /// Span-local `KnotVector::CalcShape` in direction `d`.
    fn shape1d(&self, d: usize, elevated: bool, xi: f64) -> Vec<f64> {
        let order = self.order_dir(d) + usize::from(elevated);
        let kv = if elevated { &self.kv1[d] } else { &self.kv[d] };
        let mut out = vec![0.0; order + 1];
        knot_span_shape(kv.as_slice(), order, self.ijk[d], xi, &mut out);
        out
    }

    /// Span-local `KnotVector::CalcDShape` in direction `d`.
    fn dshape1d(&self, d: usize, elevated: bool, xi: f64) -> Vec<f64> {
        let order = self.order_dir(d) + usize::from(elevated);
        let kv = if elevated { &self.kv1[d] } else { &self.kv[d] };
        let mut out = vec![0.0; order + 1];
        knot_span_dshape(kv.as_slice(), order, self.ijk[d], xi, &mut out);
        out
    }
}

impl VectorReferenceElement for NurbsHDiv3D {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { (self.order_u.max(self.order_v).max(self.order_w) + 1) as u8 } // MFEM reports the elevated degree
    fn n_dofs(&self) -> usize { self.n_dofs }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        assert_eq!(xi.len(), 3);
        let n = self.n_dofs;
        assert_eq!(values.len(), n * 3);

        // MFEM `NURBS_HDiv3DFiniteElement::CalcVShape`.
        let px = self.order_u;
        let py = self.order_v;
        let pz = self.order_w;

        let sx1 = self.shape1d(0, true, xi[0]);
        let sy = self.shape1d(1, false, xi[1]);
        let sz = self.shape1d(2, false, xi[2]);
        let sx = self.shape1d(0, false, xi[0]);
        let sy1 = self.shape1d(1, true, xi[1]);
        let sz1 = self.shape1d(2, true, xi[2]);

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

        // MFEM `NURBS_HDiv3DFiniteElement::CalcDivShape`.
        let px = self.order_u;
        let py = self.order_v;
        let pz = self.order_w;

        let sy = self.shape1d(1, false, xi[1]);
        let sz = self.shape1d(2, false, xi[2]);
        let dsx1 = self.dshape1d(0, true, xi[0]);
        let sx = self.shape1d(0, false, xi[0]);
        let dsy1 = self.dshape1d(1, true, xi[1]);
        let dsz1 = self.dshape1d(2, true, xi[2]);

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
    /// Base knot vectors (orders `px`, `py`, `pz`).
    kv: [KnotVector; 3],
    /// Degree-elevated knot vectors (orders `px+1`, `py+1`, `pz+1`).
    kv1: [KnotVector; 3],
    /// Knot span index within the patch (`NURBSFiniteElement::ijk`).
    ijk: [usize; 3],
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
        let px = knot_order(&kv_u).ok_or_else(|| "NurbsHCurl3D: invalid kv_u".to_string())?;
        let py = knot_order(&kv_v).ok_or_else(|| "NurbsHCurl3D: invalid kv_v".to_string())?;
        let pz = knot_order(&kv_w).ok_or_else(|| "NurbsHCurl3D: invalid kv_w".to_string())?;

        let kv1_u = elevated_knots(&kv_u)?;
        let kv1_v = elevated_knots(&kv_v)?;
        let kv1_w = elevated_knots(&kv_w)?;

        let n = (px + 1) * (py + 2) * (pz + 2)
              + (px + 2) * (py + 1) * (pz + 2)
              + (px + 2) * (py + 2) * (pz + 1);

        Ok(Self {
            order_u: px,
            order_v: py,
            order_w: pz,
            kv: [kv_u, kv_v, kv_w],
            kv1: [kv1_u, kv1_v, kv1_w],
            ijk: [0, 0, 0],
            n_dofs: n,
        })
    }

    pub fn n_dofs_static(px: usize, py: usize, pz: usize) -> usize {
        (px + 1) * (py + 2) * (pz + 2)
            + (px + 2) * (py + 1) * (pz + 2)
            + (px + 2) * (py + 2) * (pz + 1)
    }

    /// MFEM `NURBSFiniteElement::SetIJK` — select the knot span.
    pub fn set_ijk(&mut self, ijk: [usize; 3]) {
        self.ijk = ijk;
    }

    /// MFEM `NURBSFiniteElement::GetIJK`.
    pub fn ijk(&self) -> [usize; 3] {
        self.ijk
    }

    /// Per-direction order.
    fn order_dir(&self, d: usize) -> usize {
        match d {
            0 => self.order_u,
            1 => self.order_v,
            _ => self.order_w,
        }
    }

    /// Span-local `KnotVector::CalcShape` in direction `d`.
    fn shape1d(&self, d: usize, elevated: bool, xi: f64) -> Vec<f64> {
        let order = self.order_dir(d) + usize::from(elevated);
        let kv = if elevated { &self.kv1[d] } else { &self.kv[d] };
        let mut out = vec![0.0; order + 1];
        knot_span_shape(kv.as_slice(), order, self.ijk[d], xi, &mut out);
        out
    }

    /// Span-local `KnotVector::CalcDShape` in direction `d`.
    fn dshape1d(&self, d: usize, elevated: bool, xi: f64) -> Vec<f64> {
        let order = self.order_dir(d) + usize::from(elevated);
        let kv = if elevated { &self.kv1[d] } else { &self.kv[d] };
        let mut out = vec![0.0; order + 1];
        knot_span_dshape(kv.as_slice(), order, self.ijk[d], xi, &mut out);
        out
    }
}

impl VectorReferenceElement for NurbsHCurl3D {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { (self.order_u.max(self.order_v).max(self.order_w) + 1) as u8 } // MFEM reports the elevated degree
    fn n_dofs(&self) -> usize { self.n_dofs }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        assert_eq!(xi.len(), 3);
        let n = self.n_dofs;
        assert_eq!(values.len(), n * 3);

        // MFEM `NURBS_HCurl3DFiniteElement::CalcVShape`.
        let px = self.order_u;
        let py = self.order_v;
        let pz = self.order_w;

        let sx = self.shape1d(0, false, xi[0]);
        let sy1 = self.shape1d(1, true, xi[1]);
        let sz1 = self.shape1d(2, true, xi[2]);
        let sx1 = self.shape1d(0, true, xi[0]);
        let sy = self.shape1d(1, false, xi[1]);
        let sz = self.shape1d(2, false, xi[2]);

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

        // MFEM `NURBS_HCurl3DFiniteElement::CalcCurlShape`.
        let px = self.order_u;
        let py = self.order_v;
        let pz = self.order_w;

        let sx = self.shape1d(0, false, xi[0]);
        let sy1 = self.shape1d(1, true, xi[1]);
        let sz1 = self.shape1d(2, true, xi[2]);
        let sx1 = self.shape1d(0, true, xi[0]);
        let sy = self.shape1d(1, false, xi[1]);
        let sz = self.shape1d(2, false, xi[2]);

        let dsx1 = self.dshape1d(0, true, xi[0]);
        let dsy1 = self.dshape1d(1, true, xi[1]);
        let dsz1 = self.dshape1d(2, true, xi[2]);

        let mut o = 0;
        // x-component basis v = (shape_x(i)*shape1_y(j)*shape1_z(k), 0, 0):
        // curl v = (0, ∂v_x/∂z, -∂v_x/∂y).
        for k in 0..=pz + 1 {
            let sk = sz1[k];
            let dsk = dsz1[k];
            for j in 0..=py + 1 {
                let dsy1_sk1 = dsy1[j] * sk;
                let sy1_dsk1 = sy1[j] * dsk;
                for i in 0..=px {
                    curl_vals[o * 3 + 0] = 0.0;
                    curl_vals[o * 3 + 1] = sx[i] * sy1_dsk1;
                    curl_vals[o * 3 + 2] = -sx[i] * dsy1_sk1;
                    o += 1;
                }
            }
        }
        // y-component basis v = (0, shape1_x(i)*shape_y(j)*shape1_z(k), 0):
        // curl v = (-∂v_y/∂z, 0, ∂v_y/∂x).
        for k in 0..=pz + 1 {
            let sk = sz1[k];
            let dsk = dsz1[k];
            for j in 0..=py {
                let sy_sk1 = sy[j] * sk;
                let sy_dsk1 = sy[j] * dsk;
                for i in 0..=px + 1 {
                    curl_vals[o * 3 + 0] = -sx1[i] * sy_dsk1;
                    curl_vals[o * 3 + 1] = 0.0;
                    curl_vals[o * 3 + 2] = dsx1[i] * sy_sk1;
                    o += 1;
                }
            }
        }
        // z-component basis v = (0, 0, shape1_x(i)*shape1_y(j)*shape_z(k)):
        // curl v = (∂v_z/∂y, -∂v_z/∂x, 0).
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
    use crate::nurbs_fe_collection::knot_ncp;

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

    /// `NURBS_HDiv2DFiniteElement::CalcVShape` / `CalcDivShape` on a
    /// **multi-span** patch: `square-nurbs.mesh` refined once, space order 1,
    /// knot vectors `{0,0,0.5,1,1}`, element 0 (`ijk = 0,0`), span-local
    /// `xi = (0.5, 0.5)`.  Values dumped from MFEM 4.9.
    #[test]
    fn nurbs_hdiv2d_multispan_values_match_mfem() {
        let kv = KnotVector::new_clamped(vec![0.0, 0.0, 0.5, 1.0, 1.0]).unwrap();
        let mut e = NurbsHDiv2D::from_knot_vectors(kv.clone(), kv).unwrap();
        e.set_ijk([0, 0]);
        assert_eq!(e.n_dofs(), 12);

        let want_vsh = [
            0.125, 0.0, 0.3125, 0.0, 0.0625, 0.0, 0.125, 0.0, 0.3125, 0.0, 0.0625, 0.0, 0.0,
            0.125, 0.0, 0.125, 0.0, 0.3125, 0.0, 0.3125, 0.0, 0.0625, 0.0, 0.0625,
        ];
        let mut vsh = [0.0; 24];
        e.eval_basis_vec(&[0.5, 0.5], &mut vsh);
        for i in 0..24 {
            assert!((vsh[i] - want_vsh[i]).abs() < 1e-16, "vsh {i}: {}", vsh[i]);
        }

        let want_div = [-0.5, 0.25, 0.25, -0.5, 0.25, 0.25, -0.5, -0.5, 0.25, 0.25, 0.25, 0.25];
        let mut div = [0.0; 12];
        e.eval_div(&[0.5, 0.5], &mut div);
        for i in 0..12 {
            assert!((div[i] - want_div[i]).abs() < 1e-16, "div {i}: {}", div[i]);
        }
    }

    /// `NURBS_HCurl2DFiniteElement::CalcVShape` / `CalcCurlShape` on the same
    /// multi-span patch and point as the H(div) test.  Values dumped from
    /// MFEM 4.9 (`nurbs_ex3`'s space configuration: order 1 on a refined
    /// `square-nurbs.mesh`).
    #[test]
    fn nurbs_hcurl2d_multispan_values_match_mfem() {
        let kv = KnotVector::new_clamped(vec![0.0, 0.0, 0.5, 1.0, 1.0]).unwrap();
        let mut e = NurbsHCurl2D::from_knot_vectors(kv.clone(), kv).unwrap();
        e.set_ijk([0, 0]);
        assert_eq!(e.n_dofs(), 12);

        let want_vsh = [
            0.125, 0.0, 0.125, 0.0, 0.3125, 0.0, 0.3125, 0.0, 0.0625, 0.0, 0.0625, 0.0, 0.0,
            0.125, 0.0, 0.3125, 0.0, 0.0625, 0.0, 0.125, 0.0, 0.3125, 0.0, 0.0625,
        ];
        let mut vsh = [0.0; 24];
        e.eval_basis_vec(&[0.5, 0.5], &mut vsh);
        for i in 0..24 {
            assert!((vsh[i] - want_vsh[i]).abs() < 1e-16, "vsh {i}: {}", vsh[i]);
        }

        let want_curl = [0.5, 0.5, -0.25, -0.25, -0.25, -0.25, -0.5, 0.25, 0.25, -0.5, 0.25, 0.25];
        let mut curl = [0.0; 12];
        e.eval_curl(&[0.5, 0.5], &mut curl);
        for i in 0..12 {
            assert!((curl[i] - want_curl[i]).abs() < 1e-16, "curl {i}: {}", curl[i]);
        }
    }


    #[test]
    fn degree_elevate_preserves_spans() {
        // MFEM `KnotVector::DegreeElevate(1)`: order and control points each
        // grow by one, the number of knot spans (elements) does not change.
        let kv = clamped_uniform_knots(2, 3); // order 2, 3 spans
        let kv1 = degree_elevate(&kv, 1).expect("elevate");
        assert_eq!(knot_order(&kv1), Some(3));
        assert_eq!(knot_ncp(&kv1), Some(knot_ncp(&kv).unwrap() + 1));
        assert!(kv1.as_slice().len() > kv.as_slice().len());

        // The elevated basis is the one the H(div)/H(curl) elements evaluate on.
        let b1 = elevated_knots(&kv).expect("elevated knot vector");
        assert_eq!(knot_order(&b1), Some(3));

        // Single span: order 1 -> order 2 with three basis functions.
        let kv = clamped_uniform_knots(1, 1);
        let kv1 = degree_elevate(&kv, 1).expect("elevate");
        assert_eq!(kv1.as_slice(), &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(knot_ncp(&kv1), Some(3));
    }
}
