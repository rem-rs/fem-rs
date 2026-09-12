//! NURBS finite element *collections*.
//!
//! 1:1 port of MFEM's `NURBSFECollection`, `NURBS_HDivFECollection` and
//! `NURBS_HCurlFECollection` (C++ `fem/fe_coll.cpp`, declarations in
//! `fem/fe_coll.hpp`).  The per-element DOF counts live in
//! `NURBS*FiniteElement::SetOrder` (C++ `fem/fe/fe_nurbs.cpp`).
//!
//! # Why a separate layer?
//!
//! MFEM's NURBS elements are *stateful*: a collection only builds elements with
//! dummy unit knot vectors, and `NURBSExtension::LoadFE(i, FE)` re-binds the
//! element to the comprehensive knot vectors of topological element `i` (plus
//! the element DOF weights) right before each assembly kernel runs.  The
//! collection therefore determines only
//!
//! * which element class is used for each `Geometry::Type`,
//! * the *order* used to construct those elements, and
//! * the corresponding DOF counts.
//!
//! That is what this module provides, together with MFEM's
//! `KnotVector::DegreeElevate` (needed to build the H(div)/H(curl) elements)
//! and the MFEM `KnotVector` queries (order / control points / element count),
//! which `crate::iga::KnotVector` deliberately does not store.
//!
//! # Geometry codes
//!
//! The MFEM `Geometry::Type` codes understood by these collections are
//! `POINT = 0`, `SEGMENT = 1`, `SQUARE = 3`, `CUBE = 5`.
//!
//! # Related types
//!
//! * `crate::nurbs_vector` — H(div)/H(curl) NURBS elements.
//! * `crate::nurbs` — scalar NURBS patch elements (`NurbsPatch2D/3D`), which
//!   take the *patch* parameter in the knot range; the span-local reference
//!   coordinate used during assembly is produced by the caller via
//!   [`Nurbs1DFiniteElement::knot_location`]-style conversion.
//! * `crate::iga::{KnotVector, BsplineBasis}` — knot-sequence representation.

use crate::iga::KnotVector;
use crate::nurbs_vector::{NurbsHCurl2D, NurbsHCurl3D, NurbsHDiv2D, NurbsHDiv3D};
use crate::reference::VectorReferenceElement;

/// MFEM `NURBSFECollection::VariableOrder`.
pub const VARIABLE_ORDER: i32 = -1;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// KnotVector queries (MFEM `mesh/nurbs.hpp` class KnotVector)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// MFEM `KnotVector` order, recovered from the knot sequence.
///
/// MFEM stores the order explicitly; for the clamped knot vectors written in
/// NURBS mesh files it equals the multiplicity of the first knot minus one
/// (Piegl & Tiller, "The NURBS Book", 2nd ed., §2.2).  Returns `None` for a
/// malformed (non-clamped) sequence.
pub fn knot_order(kv: &KnotVector) -> Option<usize> {
    let knots = kv.as_slice();
    let first = *knots.first()?;
    let mult = knots
        .iter()
        .take_while(|&&k| (k - first).abs() < 1e-12)
        .count();
    // A clamped knot vector repeats its ends `Order + 1` times; that
    // repetition cannot exceed half of the vector.
    if mult < 2 || mult > knots.len() / 2 {
        return None;
    }
    Some(mult - 1)
}

/// MFEM `KnotVector::GetNCP` — number of control points, `Size() - Order - 1`.
///
/// MFEM allocates `knot.SetSize(NCP + Order + 1)`.
pub fn knot_ncp(kv: &KnotVector) -> Option<usize> {
    let order = knot_order(kv)?;
    Some(kv.as_slice().len() - order - 1)
}

/// MFEM `KnotVector::GetNKS` — number of knot spans to test with `is_element`.
pub fn knot_nks(kv: &KnotVector) -> Option<usize> {
    Some(knot_ncp(kv)? - knot_order(kv)?)
}

/// MFEM `KnotVector::isElement(i)` — whether the span starting at knot index
/// `Order + i` is non-empty.
pub fn knot_is_element(kv: &KnotVector, order: usize, i: usize) -> bool {
    let knots = kv.as_slice();
    match (knots.get(order + i), knots.get(order + i + 1)) {
        (Some(&a), Some(&b)) => a != b,
        _ => false,
    }
}

/// MFEM `KnotVector::GetElements()` — number of elements, i.e. the number of
/// non-empty knot spans.
pub fn knot_n_elements(kv: &KnotVector) -> Option<usize> {
    let order = knot_order(kv)?;
    let ncp = knot_ncp(kv)?;
    Some((order..ncp).filter(|&i| knot_is_element(kv, order, i - order)).count())
}

/// MFEM `KnotVector::DegreeElevate(t)` — C++ `mesh/nurbs.cpp:403`.
///
/// Builds an order-`Order + t` / `NCP + t` knot vector by repeating the end
/// knots `t` extra times and shifting the interior knots up.  This is the
/// operation the H(div)/H(curl) NURBS elements apply to each directional knot
/// vector (`kv1[i] = kv[i]->DegreeElevate(1)`).
///
/// Note that this is *not* classical Bezier degree elevation: MFEM leaves the
/// interior knot multiplicities unchanged, so the number of elements (and hence
/// the mesh) is preserved while the order rises.
pub fn degree_elevate(kv: &KnotVector, t: usize) -> Result<KnotVector, String> {
    let order = knot_order(kv).ok_or_else(|| "degree_elevate: invalid knot vector".to_string())?;
    let ncp = knot_ncp(kv).ok_or_else(|| "degree_elevate: invalid knot vector".to_string())?;
    let knots = kv.as_slice();
    let new_order = order + t;
    let new_ncp = ncp + t;

    let mut out = vec![0.0_f64; new_ncp + new_order + 1];
    let first = knots[0];
    let last = knots[knots.len() - 1];

    for i in 0..=new_order {
        out[i] = first;
    }
    for i in (new_order + 1)..new_ncp {
        out[i] = knots[i - t];
    }
    for i in 0..=new_order {
        out[new_ncp + i] = last;
    }
    KnotVector::new_clamped(out)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Geometry
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// The MFEM `Geometry::Type` values understood by the NURBS collections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NurbsGeometry {
    /// `Geometry::POINT = 0`
    Point,
    /// `Geometry::SEGMENT = 1`
    Segment,
    /// `Geometry::SQUARE = 3`
    Square,
    /// `Geometry::CUBE = 5`
    Cube,
}

impl NurbsGeometry {
    /// Decode an MFEM `Geometry::Type` code.
    pub fn from_mfem_code(code: i32) -> Option<Self> {
        match code {
            0 => Some(NurbsGeometry::Point),
            1 => Some(NurbsGeometry::Segment),
            3 => Some(NurbsGeometry::Square),
            5 => Some(NurbsGeometry::Cube),
            _ => None,
        }
    }

    /// The MFEM `Geometry::Type` code.
    pub fn mfem_code(self) -> i32 {
        match self {
            NurbsGeometry::Point => 0,
            NurbsGeometry::Segment => 1,
            NurbsGeometry::Square => 3,
            NurbsGeometry::Cube => 5,
        }
    }

    /// Topological dimension of the geometry.
    pub fn dim(self) -> usize {
        match self {
            NurbsGeometry::Point => 0,
            NurbsGeometry::Segment => 1,
            NurbsGeometry::Square => 2,
            NurbsGeometry::Cube => 3,
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Element kinds
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// The NURBS finite element classes a collection can hand out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NurbsElement {
    /// MFEM `PointFiniteElement` (1 DOF, geometry `POINT`).
    Point,
    /// MFEM `NURBS1DFiniteElement`.
    Nurbs1D,
    /// MFEM `NURBS2DFiniteElement`.
    Nurbs2D,
    /// MFEM `NURBS3DFiniteElement`.
    Nurbs3D,
    /// MFEM `NURBS_HDiv2DFiniteElement`.
    HDiv2D,
    /// MFEM `NURBS_HDiv3DFiniteElement`.
    HDiv3D,
    /// MFEM `NURBS_HCurl2DFiniteElement`.
    HCurl2D,
    /// MFEM `NURBS_HCurl3DFiniteElement`.
    HCurl3D,
}

impl NurbsElement {
    /// Whether the element is vector valued (H(div)/H(curl)).
    pub fn is_vector(self) -> bool {
        matches!(
            self,
            NurbsElement::HDiv2D
                | NurbsElement::HDiv3D
                | NurbsElement::HCurl2D
                | NurbsElement::HCurl3D
        )
    }

    /// MFEM `FiniteElement::GetDim()`.
    pub fn dim(self) -> usize {
        match self {
            NurbsElement::Point => 0,
            NurbsElement::Nurbs1D => 1,
            NurbsElement::Nurbs2D | NurbsElement::HDiv2D | NurbsElement::HCurl2D => 2,
            NurbsElement::Nurbs3D | NurbsElement::HDiv3D | NurbsElement::HCurl3D => 3,
        }
    }

    /// MFEM `FiniteElement::GetDof()` for a NURBS element, given the *base*
    /// knot-vector orders of the element (one per parametric direction).
    ///
    /// These are transcriptions of `NURBS*FiniteElement::SetOrder`
    /// (`fem/fe/fe_nurbs.cpp`).  Two of MFEM's 2D formulas index `orders[1]`
    /// where the symmetric expression would use `orders[0]`:
    ///
    /// ```text
    /// NURBS_HDiv2DFiniteElement : (o0+2)*(o1+1) + (o1+1)*(o1+2)
    /// NURBS_HCurl2DFiniteElement: (o0+1)*(o1+2) + (o1+2)*(o1+1)
    /// ```
    ///
    /// That is reproduced verbatim: matching MFEM on meshes with anisotropic
    /// orders (multi-patch `VariableOrder` meshes) requires the same
    /// arithmetic, even though the degree-elevated basis suggests the symmetric
    /// form.
    pub fn n_dofs(self, orders: &[usize]) -> usize {
        let o = |i: usize| orders[i];
        match self {
            NurbsElement::Point => 1,
            NurbsElement::Nurbs1D => o(0) + 1,
            NurbsElement::Nurbs2D => (o(0) + 1) * (o(1) + 1),
            NurbsElement::Nurbs3D => (o(0) + 1) * (o(1) + 1) * (o(2) + 1),
            NurbsElement::HDiv2D => (o(0) + 2) * (o(1) + 1) + (o(1) + 1) * (o(1) + 2),
            NurbsElement::HDiv3D => {
                (o(0) + 2) * (o(1) + 1) * (o(2) + 1)
                    + (o(0) + 1) * (o(1) + 2) * (o(2) + 1)
                    + (o(0) + 1) * (o(1) + 1) * (o(2) + 2)
            }
            NurbsElement::HCurl2D => (o(0) + 1) * (o(1) + 2) + (o(1) + 2) * (o(1) + 1),
            NurbsElement::HCurl3D => {
                (o(0) + 1) * (o(1) + 2) * (o(2) + 2)
                    + (o(0) + 2) * (o(1) + 1) * (o(2) + 2)
                    + (o(0) + 2) * (o(1) + 2) * (o(2) + 1)
            }
        }
    }

    /// MFEM `FiniteElement::GetOrder()` for a NURBS element, given the base
    /// knot-vector orders.
    ///
    /// The scalar elements report `max(orders)`; the H(div)/H(curl) elements
    /// mix degree-elevated directions, so they report `max(orders) + 1`.
    pub fn order(self, orders: &[usize]) -> usize {
        let m = *orders.iter().max().unwrap_or(&0);
        match self {
            NurbsElement::Point => 0,
            NurbsElement::HDiv2D
            | NurbsElement::HCurl2D
            | NurbsElement::HDiv3D
            | NurbsElement::HCurl3D => m + 1,
            _ => m,
        }
    }

    /// Number of knot vectors the element needs (one per parametric direction).
    pub fn n_knot_vectors(self) -> usize {
        match self {
            NurbsElement::Point => 0,
            NurbsElement::Nurbs1D => 1,
            NurbsElement::Nurbs2D | NurbsElement::HDiv2D | NurbsElement::HCurl2D => 2,
            NurbsElement::Nurbs3D | NurbsElement::HDiv3D | NurbsElement::HCurl3D => 3,
        }
    }

    /// MFEM class name, for diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            NurbsElement::Point => "PointFiniteElement",
            NurbsElement::Nurbs1D => "NURBS1DFiniteElement",
            NurbsElement::Nurbs2D => "NURBS2DFiniteElement",
            NurbsElement::Nurbs3D => "NURBS3DFiniteElement",
            NurbsElement::HDiv2D => "NURBS_HDiv2DFiniteElement",
            NurbsElement::HDiv3D => "NURBS_HDiv3DFiniteElement",
            NurbsElement::HCurl2D => "NURBS_HCurl2DFiniteElement",
            NurbsElement::HCurl3D => "NURBS_HCurl3DFiniteElement",
        }
    }

    /// Build the element for the given *base* knot vectors.
    ///
    /// For the H(div)/H(curl) kinds the degree-elevated
    /// (`DegreeElevate(1)`) vectors MFEM derives internally are built by the
    /// element constructor, exactly as `NURBS_HDiv*FECollection` does.
    pub fn build(self, kv: &[KnotVector]) -> Result<NurbsRefElement, String> {
        if kv.len() != self.n_knot_vectors() {
            return Err(format!(
                "NurbsElement::build: {} needs {} knot vectors, got {}",
                self.name(),
                self.n_knot_vectors(),
                kv.len()
            ));
        }
        Ok(match self {
            NurbsElement::Point => NurbsRefElement::Point,
            NurbsElement::Nurbs1D => {
                NurbsRefElement::Nurbs1D(Nurbs1DFiniteElement::new(kv[0].clone())?)
            }
            NurbsElement::Nurbs2D => NurbsRefElement::Nurbs2D(NurbsScalar2D::new(
                kv[0].clone(),
                kv[1].clone(),
            )?),
            NurbsElement::Nurbs3D => NurbsRefElement::Nurbs3D(NurbsScalar3D::new(
                kv[0].clone(),
                kv[1].clone(),
                kv[2].clone(),
            )?),
            NurbsElement::HDiv2D => NurbsRefElement::HDiv2D(NurbsHDiv2D::from_knot_vectors(
                kv[0].clone(),
                kv[1].clone(),
            )?),
            NurbsElement::HDiv3D => NurbsRefElement::HDiv3D(NurbsHDiv3D::from_knot_vectors(
                kv[0].clone(),
                kv[1].clone(),
                kv[2].clone(),
            )?),
            NurbsElement::HCurl2D => NurbsRefElement::HCurl2D(NurbsHCurl2D::from_knot_vectors(
                kv[0].clone(),
                kv[1].clone(),
            )?),
            NurbsElement::HCurl3D => NurbsRefElement::HCurl3D(NurbsHCurl3D::from_knot_vectors(
                kv[0].clone(),
                kv[1].clone(),
                kv[2].clone(),
            )?),
        })
    }
}

/// A concrete NURBS element bound to knot vectors.
#[derive(Debug, Clone)]
pub enum NurbsRefElement {
    /// MFEM `PointFiniteElement` — a single unit DOF.
    Point,
    /// MFEM `NURBS1DFiniteElement`.
    Nurbs1D(Nurbs1DFiniteElement),
    /// MFEM `NURBS2DFiniteElement` (scalar, span-local reference coordinate).
    Nurbs2D(NurbsScalar2D),
    /// MFEM `NURBS3DFiniteElement` (scalar, span-local reference coordinate).
    Nurbs3D(NurbsScalar3D),
    /// MFEM `NURBS_HDiv2DFiniteElement`.
    HDiv2D(NurbsHDiv2D),
    /// MFEM `NURBS_HDiv3DFiniteElement`.
    HDiv3D(NurbsHDiv3D),
    /// MFEM `NURBS_HCurl2DFiniteElement`.
    HCurl2D(NurbsHCurl2D),
    /// MFEM `NURBS_HCurl3DFiniteElement`.
    HCurl3D(NurbsHCurl3D),
}

impl NurbsRefElement {
    /// The element kind.
    pub fn kind(&self) -> NurbsElement {
        match self {
            NurbsRefElement::Point => NurbsElement::Point,
            NurbsRefElement::Nurbs1D(_) => NurbsElement::Nurbs1D,
            NurbsRefElement::Nurbs2D(_) => NurbsElement::Nurbs2D,
            NurbsRefElement::Nurbs3D(_) => NurbsElement::Nurbs3D,
            NurbsRefElement::HDiv2D(_) => NurbsElement::HDiv2D,
            NurbsRefElement::HDiv3D(_) => NurbsElement::HDiv3D,
            NurbsRefElement::HCurl2D(_) => NurbsElement::HCurl2D,
            NurbsRefElement::HCurl3D(_) => NurbsElement::HCurl3D,
        }
    }

    /// MFEM `FiniteElement::GetDof()`.
    pub fn n_dofs(&self) -> usize {
        match self {
            NurbsRefElement::Point => 1,
            NurbsRefElement::Nurbs1D(e) => e.n_dofs(),
            NurbsRefElement::Nurbs2D(e) => e.n_dofs(),
            NurbsRefElement::Nurbs3D(e) => e.n_dofs(),
            NurbsRefElement::HDiv2D(e) => VectorReferenceElement::n_dofs(e),
            NurbsRefElement::HDiv3D(e) => VectorReferenceElement::n_dofs(e),
            NurbsRefElement::HCurl2D(e) => VectorReferenceElement::n_dofs(e),
            NurbsRefElement::HCurl3D(e) => VectorReferenceElement::n_dofs(e),
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Scalar NURBS elements
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Convert an [`crate::iga::KnotVector`] (pure knot sequence) to the
/// MFEM `KnotVector::GetKnotLocation(xi, i + Order)`: the knot (patch)
/// parameter of the span-local reference coordinate `xi ∈ [0,1]` on the span
/// beginning at knot index `span + Order`.
pub(crate) fn knot_span_location(knots: &[f64], order: usize, span: usize, xi: f64) -> f64 {
    let ip = span + order;
    if ip + 1 < knots.len() {
        xi * knots[ip + 1] + (1.0 - xi) * knots[ip]
    } else {
        xi
    }
}

/// MFEM `KnotVector::CalcShape(shape, i, xi)` — the `Order + 1` non-vanishing
/// values at the span-local coordinate `xi`, ordered
/// `N_{i}, N_{i+1}, ..., N_{i+Order}` in the span enumeration.
pub fn knot_span_shape(knots: &[f64], order: usize, span: usize, xi: f64, shape: &mut [f64]) {
    let u = knot_span_location(knots, order, span, xi);
    span_local_basis(knots, order, span, u, shape);
}

/// MFEM `KnotVector::CalcDShape(grad, i, xi)` — the span-local first
/// derivatives w.r.t. `xi` (same ordering as [`knot_span_shape`]).
pub fn knot_span_dshape(knots: &[f64], order: usize, span: usize, xi: f64, grad: &mut [f64]) {
    let u = knot_span_location(knots, order, span, xi);
    span_local_basis_deriv(knots, order, span, u, grad);
}

/// First derivative of the span-local B-spline values with respect to the knot
/// **parameter** `u` (not the span-local `xi` of [`knot_span_dshape`], which
/// differs by the span length).
///
/// Uses [`span_local_deriv_raw`], whose value is `dN/du / Order` (MFEM recovers
/// `dN/dxi` from it by multiplying with `Order * span_length`).
///
/// The span-local coordinate `xi` in `[0,1]` maps to a knot parameter inside
/// `[knots[span+Order], knots[span+Order+1]]`, so this is the entry point for
/// evaluating a basis on a *sub-interval* of a span — the NURBS geometry path,
/// where a refined element's parameter interval `[a, b]` lies inside the
/// original, unrefined span.
pub fn knot_span_dparam(knots: &[f64], order: usize, span: usize, u: f64, grad: &mut [f64]) {
    span_local_deriv_raw(knots, order, span, u, grad);
    let p = order as f64;
    for g in grad.iter_mut() {
        *g *= p;
    }
}

/// MFEM `KnotVector::GetSpan(u)` — the raw knot index `ks` with
/// `knot(ks) <= u < knot(ks+1)` (binary search in `(Order, NCP)`).
pub fn knot_get_span(knots: &[f64], order: usize, u: f64) -> usize {
    let ncp = knots.len() - order - 1;
    if u == knots[knots.len() - 1] {
        return ncp - 1;
    }
    if u == knots[0] {
        return order;
    }
    debug_assert!(u > knots[0] && u < knots[knots.len() - 1], "Knot location outside of the range of the KnotVector");
    let mut low = order;
    let mut high = ncp;
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

/// MFEM `KnotVector::GetGreville(i)` — the average of the `Order` knots
/// `knot[i+1] .. knot[i+Order]`.
pub fn knot_greville(knots: &[f64], order: usize, i: usize) -> f64 {
    let mut sum = 0.0;
    for j in 1..=order {
        sum += knots[i + j];
    }
    sum / order as f64
}

/// MFEM `KnotVector::inSpan(u, ni)` — whether `knot(ni) <= u <= knot(ni+1)`.
pub fn knot_in_span(knots: &[f64], u: f64, ni: usize) -> bool {
    u >= knots[ni] && u <= knots[ni + 1]
}

/// MFEM `KnotVector::CalcDnShape(gradn, n, i, xi)` (Piegl & Tiller, "The NURBS
/// Book", 2nd ed., Algorithm A2.3 p. 72) — the `n`-th derivatives of the
/// `Order + 1` non-vanishing basis values at the span-local coordinate `xi`,
/// w.r.t. the span-local reference coordinate (the closing `temp` scaling of
/// A2.3 converts `d^n N/du^n` into `d^n N/dxi^n`).  `n` must be `>= 1`.
///
/// The `j2 + 1` index of the `r <= pk` tail branch reproduces the C `for`
/// loop's post-condition (`j == j2 + 1` after the loop) that MFEM relies on.
pub fn knot_span_dn_shape(
    knots: &[f64],
    order: usize,
    span: usize,
    xi: f64,
    n: usize,
    gradn: &mut [f64],
) {
    let p = order;
    let ip = span + p;
    let u = knot_span_location(knots, order, span, xi);

    // ndu[j][r] holds the denominators for column j, ndu[r][j] the values.
    let mut ndu = vec![vec![0.0; p + 1]; p + 1];
    let mut left = vec![0.0; p + 1];
    let mut right = vec![0.0; p + 1];
    ndu[0][0] = 1.0;
    for j in 1..=p {
        left[j] = u - knots[ip + 1 - j];
        right[j] = knots[ip + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            ndu[j][r] = right[r + 1] + left[j - r];
            let temp = if ndu[j][r] != 0.0 { ndu[r][j - 1] / ndu[j][r] } else { 0.0 };
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }

    let mut a = [[0.0_f64; 32]; 2];
    for r in 0..=p {
        a[0][0] = 1.0;
        let mut s1 = 0usize;
        let mut s2 = 1usize;
        for k in 1..=n {
            let mut d = 0.0;
            let (ri, pk) = (r as isize - k as isize, p as isize - k as isize);
            if r >= k {
                a[s2][0] = a[s1][0] / ndu[(pk + 1) as usize][ri as usize];
                d = a[s2][0] * ndu[ri as usize][pk as usize];
            }
            let j1 = if ri >= -1 { 1 } else { -(ri) as usize };
            let j2 = if (r as isize - 1) <= pk { k - 1 } else { p - r };
            let mut j = j1;
            while j <= j2 {
                a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[(pk + 1) as usize][(ri + j as isize) as usize];
                d += a[s2][j] * ndu[(ri + j as isize) as usize][pk as usize];
                j += 1;
            }
            if r as isize <= pk {
                a[s2][k] = -a[s1][k - 1] / ndu[(pk + 1) as usize][r];
                // `j == j2 + 1` when the loop ran, `j1` otherwise (C loop
                // post-condition).
                let jq = if j1 <= j2 { j2 + 1 } else { j1 };
                d += a[s2][jq] * ndu[(ri + jq as isize) as usize][pk as usize];
            }
            gradn[r] = d;
            std::mem::swap(&mut s1, &mut s2);
        }
    }

    // Closing scaling: `temp = p*(knot[ip+1]-knot[ip])` for n = 1, multiplied
    // by `(p-k)*span` for every further derivative.
    let span_len = knots[ip + 1] - knots[ip];
    let mut temp = p as f64 * span_len;
    for k in 1..n {
        temp *= (p - k) as f64 * span_len;
    }
    for g in gradn.iter_mut() {
        *g *= temp;
    }
}

/// MFEM `KnotVector::GetBotella(i)` — the knot location where the `i`-th shape
/// function attains its maximum ("Botella point", Botella & Shariff 2003),
/// found by Newton iteration on `dN_i/du = 0` starting from the Greville
/// point.  Reverts to the Greville point when the knots repeat.
pub fn knot_botella(kv: &KnotVector, i: usize) -> f64 {
    const ITERMAX: usize = 10;
    const TOL: f64 = 1e-8;

    let knots = kv.as_slice();
    let order = knot_order(kv).expect("validated knot vector");

    // Get initial guess
    let mut u = knot_greville(knots, order, i);

    // Check for a repeated knot -- include begin and end
    if knots[i + 1] == knots[i + order] {
        return u;
    }

    let mut grad = vec![0.0_f64; order + 1];
    let mut hess = vec![0.0_f64; order + 1];
    for _iter in 0..ITERMAX {
        let ks = knot_get_span(knots, order, u);
        let xi = (u - knots[ks]) / (knots[ks + 1] - knots[ks]);
        let o = order as isize - (ks as isize - i as isize);
        debug_assert!((0..=order as isize).contains(&o), "GetBotella: shape index out of span");

        knot_span_dshape(knots, order, ks - order, xi, &mut grad);
        knot_span_dn_shape(knots, order, ks - order, xi, 2, &mut hess);

        u -= (grad[o as usize] / hess[o as usize]) * (knots[ks + 1] - knots[ks]);

        if grad[o as usize].abs() < TOL {
            break;
        }
    }
    u
}

/// MFEM `NURBS1DFiniteElement` — rational 1D NURBS element on one knot span.
///
/// The element owns one knot vector, one weight per DOF
/// (`NURBSFiniteElement::weights`, filled by `NURBSExtension::LoadFE`), and a
/// selected knot-span index `ijk` (`NURBSFiniteElement::SetIJK`).
///
/// The reference coordinate handed to [`Self::calc_shape`] /
/// [`Self::calc_dshape`] is *span-local* (`xi ∈ [0,1]` over the selected span),
/// matching MFEM's `KnotVector::CalcShape(shape, i, xi)` convention; the
/// conversion uses `GetKnotLocation` (`xi*knot[ip+1] + (1-xi)*knot[ip]` with
/// `ip = i + Order`).
#[derive(Debug, Clone)]
pub struct Nurbs1DFiniteElement {
    kv: KnotVector,
    order: usize,
    /// Weights, one per DOF; unit by default (`unitweights`).
    weights: Vec<f64>,
    /// Knot-span index of the selected element (MFEM `ijk[0]`).
    ijk: usize,
}

impl Nurbs1DFiniteElement {
    /// Create with the given knot vector and unit weights.
    pub fn new(kv: KnotVector) -> Result<Self, String> {
        let order = knot_order(&kv)
            .ok_or_else(|| "Nurbs1DFiniteElement: invalid knot vector".to_string())?;
        let ncp = knot_ncp(&kv).expect("validated knot vector");
        Ok(Self {
            kv,
            order,
            weights: vec![1.0; ncp],
            ijk: 0,
        })
    }

    /// MFEM `NURBSFiniteElement::SetIJK` — select the knot span.
    pub fn set_ijk(&mut self, span: usize) {
        self.ijk = span;
    }

    /// MFEM `NURBSFiniteElement::GetIJK`.
    pub fn ijk(&self) -> usize {
        self.ijk
    }

    /// MFEM `NURBSFiniteElement::SetWeights` / `LoadFE`'s `GetSubVector`.
    pub fn set_weights(&mut self, weights: Vec<f64>) -> Result<(), String> {
        if weights.len() != self.n_dofs() {
            return Err(format!(
                "Nurbs1DFiniteElement::set_weights: expected {} weights, got {}",
                self.n_dofs(),
                weights.len()
            ));
        }
        self.weights = weights;
        Ok(())
    }

    /// The knot vector.
    pub fn knot_vector(&self) -> &KnotVector {
        &self.kv
    }

    /// MFEM `NURBS1DFiniteElement::GetOrder`.
    pub fn order(&self) -> usize {
        self.order
    }

    /// MFEM `NURBS1DFiniteElement::GetDof`.
    pub fn n_dofs(&self) -> usize {
        self.order + 1
    }

    /// Number of non-empty knot spans.
    pub fn n_elements(&self) -> usize {
        knot_n_elements(&self.kv).unwrap_or(0)
    }

    /// `GetKnotLocation(xi, i + Order)` — map a span-local coordinate to the
    /// knot (parameter) range.
    pub fn knot_location(&self, xi: f64, span: usize) -> f64 {
        let knots = self.kv.as_slice();
        let ip = span + self.order;
        if ip + 1 < knots.len() {
            xi * knots[ip + 1] + (1.0 - xi) * knots[ip]
        } else {
            xi
        }
    }

    /// MFEM `NURBS1DFiniteElement::CalcShape`.
    ///
    /// `shape` must have length `Order + 1`; on return `shape[o]` is the
    /// rational basis value of the `o`-th active DOF, i.e. of the global DOF
    /// `first + o` where `first = ijk` in the knot-span enumeration (the values
    /// are ordered as MFEM's `KnotVector::CalcShape` orders them:
    /// `N_{ijk}, N_{ijk+1}, ..., N_{ijk+Order}`).
    pub fn calc_shape(&self, xi: f64, shape: &mut [f64]) {
        let dof = self.n_dofs();
        assert_eq!(shape.len(), dof, "Nurbs1DFiniteElement::calc_shape size");

        let u = self.knot_location(xi, self.ijk);
        span_local_basis(self.kv.as_slice(), self.order, self.ijk, u, shape);

        let mut sum = 0.0;
        for o in 0..dof {
            shape[o] *= self.weights[o];
            sum += shape[o];
        }
        divide_by(shape, sum);
    }

    /// MFEM `NURBS1DFiniteElement::CalcDShape` — derivative of the rational
    /// basis with respect to the *parameter* `u`.
    pub fn calc_dshape(&self, xi: f64, dshape: &mut [f64]) {
        let dof = self.n_dofs();
        assert_eq!(dshape.len(), dof, "Nurbs1DFiniteElement::calc_dshape size");

        let u = self.knot_location(xi, self.ijk);
        let mut vals = vec![0.0; dof];
        let mut ders = vec![0.0; dof];
        span_local_basis(self.kv.as_slice(), self.order, self.ijk, u, &mut vals);
        span_local_basis_deriv(self.kv.as_slice(), self.order, self.ijk, u, &mut ders);

        let mut sum = 0.0;
        let mut dsum = 0.0;
        for o in 0..dof {
            vals[o] *= self.weights[o];
            ders[o] *= self.weights[o];
            sum += vals[o];
            dsum += ders[o];
        }
        let inv = 1.0 / (sum * sum);
        for o in 0..dof {
            dshape[o] = (ders[o] * sum - vals[o] * dsum) * inv;
        }
    }
}

/// MFEM `NURBS2DFiniteElement` — rational scalar element on one knot span of a
/// quadrilateral patch.
///
/// Mirrors `NURBSFiniteElement`: the element owns the two (unique) knot
/// vectors, the span index `ijk` (`SetIJK`, from `el_to_IJK`) and one weight
/// per *local* DOF (`LoadFE`'s `weights.GetSubVector(dofs, Weights())`).  The
/// reference coordinate of `calc_shape` / `calc_grad` is span-local
/// (`xi ∈ [0,1]²` over the selected span), as in MFEM's
/// `KnotVector::CalcShape(shape, i, xi)`.
///
/// The local DOF order is the tensor-product order of
/// `NURBS2DFiniteElement::CalcShape` (`o = i + (orders[0]+1)*j`), which is the
/// order `NURBSExtension::Generate2DElementDofTable` lists the element's DOFs
/// in.
#[derive(Debug, Clone)]
pub struct NurbsScalar2D {
    kv: [KnotVector; 2],
    orders: [usize; 2],
    ijk: [usize; 2],
    /// One weight per local DOF; unit until `SetWeights` (`LoadFE`).
    weights: Vec<f64>,
}

impl NurbsScalar2D {
    /// Create with unit weights.
    pub fn new(kv_u: KnotVector, kv_v: KnotVector) -> Result<Self, String> {
        let o0 = knot_order(&kv_u).ok_or_else(|| "Nurbs2DFiniteElement: bad kv_u".to_string())?;
        let o1 = knot_order(&kv_v).ok_or_else(|| "Nurbs2DFiniteElement: bad kv_v".to_string())?;
        Ok(Self {
            kv: [kv_u, kv_v],
            orders: [o0, o1],
            ijk: [0, 0],
            weights: vec![1.0; (o0 + 1) * (o1 + 1)],
        })
    }

    /// MFEM `NURBSFiniteElement::SetIJK`.
    pub fn set_ijk(&mut self, ijk: [usize; 2]) {
        self.ijk = ijk;
    }

    /// MFEM `NURBSFiniteElement::GetIJK`.
    pub fn ijk(&self) -> [usize; 2] {
        self.ijk
    }

    /// MFEM `NURBSFiniteElement::SetWeights` (`LoadFE`'s `GetSubVector`).
    pub fn set_weights(&mut self, weights: Vec<f64>) -> Result<(), String> {
        if weights.len() != self.n_dofs() {
            return Err(format!(
                "NurbsScalar2D::set_weights: expected {} weights, got {}",
                self.n_dofs(),
                weights.len()
            ));
        }
        self.weights = weights;
        Ok(())
    }

    /// MFEM `NURBS2DFiniteElement::GetOrder`.
    pub fn order(&self) -> usize {
        self.orders[0].max(self.orders[1])
    }

    /// Per-direction order.
    pub fn orders(&self) -> [usize; 2] {
        self.orders
    }

    /// MFEM `NURBS2DFiniteElement::GetDof`.
    pub fn n_dofs(&self) -> usize {
        (self.orders[0] + 1) * (self.orders[1] + 1)
    }

    /// MFEM `NURBS2DFiniteElement::CalcShape` (rational basis values);
    /// `values` must have length `n_dofs`.
    pub fn calc_shape(&self, xi: &[f64], values: &mut [f64]) {
        assert_eq!(xi.len(), 2, "NurbsScalar2D::calc_shape needs 2 coordinates");
        assert_eq!(values.len(), self.n_dofs(), "NurbsScalar2D::calc_shape size");
        let (ox, oy) = (self.orders[0], self.orders[1]);
        let mut shape_x = vec![0.0; ox + 1];
        let mut shape_y = vec![0.0; oy + 1];
        knot_span_shape(self.kv[0].as_slice(), ox, self.ijk[0], xi[0], &mut shape_x);
        knot_span_shape(self.kv[1].as_slice(), oy, self.ijk[1], xi[1], &mut shape_y);

        let mut sum = 0.0;
        let mut o = 0;
        for j in 0..=oy {
            let sy = shape_y[j];
            for i in 0..=ox {
                let v = shape_x[i] * sy * self.weights[o];
                values[o] = v;
                sum += v;
                o += 1;
            }
        }
        divide_by(values, sum);
    }

    /// MFEM `NURBS2DFiniteElement::CalcDShape` (gradients w.r.t. the span-local
    /// reference coordinates); `grads` must have length `2 * n_dofs`, ordered
    /// `[dR/dx, dR/dy]` per DOF.
    pub fn calc_grad(&self, xi: &[f64], grads: &mut [f64]) {
        assert_eq!(xi.len(), 2, "NurbsScalar2D::calc_grad needs 2 coordinates");
        assert_eq!(grads.len(), 2 * self.n_dofs(), "NurbsScalar2D::calc_grad size");
        let (ox, oy) = (self.orders[0], self.orders[1]);
        let mut shape_x = vec![0.0; ox + 1];
        let mut shape_y = vec![0.0; oy + 1];
        let mut dshape_x = vec![0.0; ox + 1];
        let mut dshape_y = vec![0.0; oy + 1];
        knot_span_shape(self.kv[0].as_slice(), ox, self.ijk[0], xi[0], &mut shape_x);
        knot_span_shape(self.kv[1].as_slice(), oy, self.ijk[1], xi[1], &mut shape_y);
        knot_span_dshape(self.kv[0].as_slice(), ox, self.ijk[0], xi[0], &mut dshape_x);
        knot_span_dshape(self.kv[1].as_slice(), oy, self.ijk[1], xi[1], &mut dshape_y);
        let mut u = vec![0.0; self.n_dofs()];
        let mut sum = 0.0;
        let mut dsum = [0.0_f64; 2];
        let mut o = 0;
        for j in 0..=oy {
            let (sy, dsy) = (shape_y[j], dshape_y[j]);
            for i in 0..=ox {
                let v = shape_x[i] * sy * self.weights[o];
                let gx = dshape_x[i] * sy * self.weights[o];
                let gy = shape_x[i] * dsy * self.weights[o];
                u[o] = v;
                sum += v;
                grads[2 * o] = gx;
                grads[2 * o + 1] = gy;
                dsum[0] += gx;
                dsum[1] += gy;
                o += 1;
            }
        }
        sum = 1.0 / sum;
        dsum[0] *= sum * sum;
        dsum[1] *= sum * sum;
        for o in 0..self.n_dofs() {
            grads[2 * o] = grads[2 * o] * sum - u[o] * dsum[0];
            grads[2 * o + 1] = grads[2 * o + 1] * sum - u[o] * dsum[1];
        }
    }
}

/// MFEM `NURBS3DFiniteElement` — rational scalar element on one knot span of a
/// hexahedral patch (3-D counterpart of [`NurbsScalar2D`], same conventions).
#[derive(Debug, Clone)]
pub struct NurbsScalar3D {
    kv: [KnotVector; 3],
    orders: [usize; 3],
    ijk: [usize; 3],
    /// One weight per local DOF; unit until `SetWeights` (`LoadFE`).
    weights: Vec<f64>,
}

impl NurbsScalar3D {
    /// Create with unit weights.
    pub fn new(kv_u: KnotVector, kv_v: KnotVector, kv_w: KnotVector) -> Result<Self, String> {
        let o = [
            knot_order(&kv_u).ok_or_else(|| "Nurbs3DFiniteElement: bad kv_u".to_string())?,
            knot_order(&kv_v).ok_or_else(|| "Nurbs3DFiniteElement: bad kv_v".to_string())?,
            knot_order(&kv_w).ok_or_else(|| "Nurbs3DFiniteElement: bad kv_w".to_string())?,
        ];
        Ok(Self {
            kv: [kv_u, kv_v, kv_w],
            orders: o,
            ijk: [0, 0, 0],
            weights: vec![1.0; (o[0] + 1) * (o[1] + 1) * (o[2] + 1)],
        })
    }

    /// MFEM `NURBSFiniteElement::SetIJK`.
    pub fn set_ijk(&mut self, ijk: [usize; 3]) {
        self.ijk = ijk;
    }

    /// MFEM `NURBSFiniteElement::GetIJK`.
    pub fn ijk(&self) -> [usize; 3] {
        self.ijk
    }

    /// MFEM `NURBSFiniteElement::SetWeights` (`LoadFE`'s `GetSubVector`).
    pub fn set_weights(&mut self, weights: Vec<f64>) -> Result<(), String> {
        if weights.len() != self.n_dofs() {
            return Err(format!(
                "NurbsScalar3D::set_weights: expected {} weights, got {}",
                self.n_dofs(),
                weights.len()
            ));
        }
        self.weights = weights;
        Ok(())
    }

    /// MFEM `NURBS3DFiniteElement::GetOrder`.
    pub fn order(&self) -> usize {
        self.orders[0].max(self.orders[1]).max(self.orders[2])
    }

    /// Per-direction order.
    pub fn orders(&self) -> [usize; 3] {
        self.orders
    }

    /// MFEM `NURBS3DFiniteElement::GetDof`.
    pub fn n_dofs(&self) -> usize {
        (self.orders[0] + 1) * (self.orders[1] + 1) * (self.orders[2] + 1)
    }

    /// Span-local B-spline values/derivatives in the three directions.
    fn local_1d(&self, xi: &[f64]) -> ([Vec<f64>; 3], [Vec<f64>; 3]) {
        let mut vals = [Vec::new(), Vec::new(), Vec::new()];
        let mut ders = [Vec::new(), Vec::new(), Vec::new()];
        for d in 0..3 {
            let n = self.orders[d] + 1;
            vals[d] = vec![0.0; n];
            ders[d] = vec![0.0; n];
            knot_span_shape(
                self.kv[d].as_slice(), self.orders[d], self.ijk[d], xi[d], &mut vals[d]);
            knot_span_dshape(
                self.kv[d].as_slice(), self.orders[d], self.ijk[d], xi[d], &mut ders[d]);
        }
        (vals, ders)
    }

    /// MFEM `NURBS3DFiniteElement::CalcShape` (rational basis values);
    /// `values` must have length `n_dofs`.
    pub fn calc_shape(&self, xi: &[f64], values: &mut [f64]) {
        assert_eq!(xi.len(), 3, "NurbsScalar3D::calc_shape needs 3 coordinates");
        assert_eq!(values.len(), self.n_dofs(), "NurbsScalar3D::calc_shape size");
        let (ox, oy, oz) = (self.orders[0], self.orders[1], self.orders[2]);
        let (shape, _) = self.local_1d(xi);

        let mut sum = 0.0;
        let mut o = 0;
        for k in 0..=oz {
            let sz = shape[2][k];
            for j in 0..=oy {
                let sy_sz = shape[1][j] * sz;
                for i in 0..=ox {
                    let v = shape[0][i] * sy_sz * self.weights[o];
                    values[o] = v;
                    sum += v;
                    o += 1;
                }
            }
        }
        divide_by(values, sum);
    }

    /// MFEM `NURBS3DFiniteElement::CalcDShape` (gradients w.r.t. the span-local
    /// reference coordinates); `grads` must have length `3 * n_dofs`, ordered
    /// `[dR/dx, dR/dy, dR/dz]` per DOF.
    pub fn calc_grad(&self, xi: &[f64], grads: &mut [f64]) {
        assert_eq!(xi.len(), 3, "NurbsScalar3D::calc_grad needs 3 coordinates");
        assert_eq!(grads.len(), 3 * self.n_dofs(), "NurbsScalar3D::calc_grad size");
        let (ox, oy, oz) = (self.orders[0], self.orders[1], self.orders[2]);
        let (shape, dshape) = self.local_1d(xi);

        let mut u = vec![0.0; self.n_dofs()];
        let mut sum = 0.0;
        let mut dsum = [0.0_f64; 3];
        let mut o = 0;
        for k in 0..=oz {
            let (sz, dsz) = (shape[2][k], dshape[2][k]);
            for j in 0..=oy {
                let (sy_sz, dsy_sz, sy_dsz) =
                    (shape[1][j] * sz, dshape[1][j] * sz, shape[1][j] * dsz);
                for i in 0..=ox {
                    let w = self.weights[o];
                    let v = shape[0][i] * sy_sz * w;
                    u[o] = v;
                    sum += v;
                    grads[3 * o] = dshape[0][i] * sy_sz * w;
                    grads[3 * o + 1] = shape[0][i] * dsy_sz * w;
                    grads[3 * o + 2] = shape[0][i] * sy_dsz * w;
                    dsum[0] += grads[3 * o];
                    dsum[1] += grads[3 * o + 1];
                    dsum[2] += grads[3 * o + 2];
                    o += 1;
                }
            }
        }
        sum = 1.0 / sum;
        for d in 0..3 {
            dsum[d] *= sum * sum;
        }
        for o in 0..self.n_dofs() {
            for d in 0..3 {
                grads[3 * o + d] = grads[3 * o + d] * sum - u[o] * dsum[d];
            }
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Span-local B-spline evaluation (MFEM `KnotVector::CalcShape`)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Evaluate the `order + 1` non-vanishing B-spline values on knot span
/// `Order + i` at the parameter `u`, ordered as MFEM's
/// `KnotVector::CalcShape(shape, i, xi)` orders them
/// (`N_{i}, N_{i+1}, ..., N_{i+Order}` in the span enumeration).
///
/// This is the Cox-de Boor recurrence specialised to one span (Piegl &
/// Tiller A2.2), which is what MFEM implements in `KnotVector::CalcShape`.
fn span_local_basis(knots: &[f64], order: usize, i: usize, u: f64, shape: &mut [f64]) {
    let ip = i + order;
    shape[0] = 1.0;
    let mut left = vec![0.0; order + 1];
    let mut right = vec![0.0; order + 1];
    for j in 1..=order {
        left[j] = u - knots[ip + 1 - j];
        right[j] = knots[ip + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            let denom = right[r + 1] + left[j - r];
            let tmp = if denom != 0.0 { shape[r] / denom } else { 0.0 };
            shape[r] = saved + right[r + 1] * tmp;
            saved = left[j - r] * tmp;
        }
        shape[j] = saved;
    }
}

/// MFEM `KnotVector::CalcDShape` *before* its final
/// `grad *= p*(knot[ip+1] - knot[ip])` scaling.
///
/// This is the Piegl & Tiller A2.3 first-derivative combination without the
/// NURBS book's closing `ders[k][j] *= r` factor (`r = p!/(p-k)!`), so the
/// result is `dN/du / Order` in the knot parameter `u`: MFEM divides by the
/// same `Order` through the `p*span` scaling below.  [`span_local_basis_deriv`]
/// adds that scaling; the NURBS geometry path (`fem-space`) uses this raw form
/// to differentiate with respect to a *different* reference interval.
fn span_local_deriv_raw(knots: &[f64], order: usize, i: usize, u: f64, grad: &mut [f64]) {
    let p = order;
    let ip = i + p;
    // ndu[j][r] holds the denominators for column j, ndu[r][j] the values.
    let mut ndu = vec![vec![0.0; p + 1]; p + 1];
    let mut left = vec![0.0; p + 1];
    let mut right = vec![0.0; p + 1];
    ndu[0][0] = 1.0;
    for j in 1..=p {
        left[j] = u - knots[ip + 1 - j];
        right[j] = knots[ip + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            ndu[j][r] = right[r + 1] + left[j - r];
            let tmp = if ndu[j][r] != 0.0 { ndu[r][j - 1] / ndu[j][r] } else { 0.0 };
            ndu[r][j] = saved + right[r + 1] * tmp;
            saved = left[j - r] * tmp;
        }
        ndu[j][j] = saved;
    }
    let pk = p.wrapping_sub(1);
    for r in 0..=p {
        let mut d = 0.0;
        if r >= 1 && p >= 1 {
            let rk = r - 1;
            d = if ndu[p][rk] != 0.0 { ndu[rk][pk] / ndu[p][rk] } else { 0.0 };
        }
        if r <= pk && p >= 1 {
            d -= if ndu[p][r] != 0.0 { ndu[r][pk] / ndu[p][r] } else { 0.0 };
        }
        grad[r] = d;
    }
}

/// First derivative of the span-local B-spline values with respect to the
/// span-local reference coordinate `xi` in `[0,1]` (Piegl & Tiller A2.3 as
/// MFEM implements it in `KnotVector::CalcDShape`, including the closing
/// `grad *= p*(knot[i+Order+1] - knot[i+Order])` scaling that converts
/// `dN/du` into `dN/dxi`).
fn span_local_basis_deriv(knots: &[f64], order: usize, i: usize, u: f64, grad: &mut [f64]) {
    span_local_deriv_raw(knots, order, i, u, grad);
    let ip = i + order;
    let scale = order as f64 * (knots[ip + 1] - knots[ip]);
    for g in grad.iter_mut() {
        *g *= scale;
    }
}

/// Divide `v` by `sum` in place (MFEM `shape /= sum`).
fn divide_by(v: &mut [f64], sum: f64) {
    if sum != 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Collections
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// The element a collection selects for a geometry, together with the order the
/// collection constructs it with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NurbsFEEntry {
    /// Element class.
    pub element: NurbsElement,
    /// Order passed to the element constructor (`VariableOrder` maps to 1).
    pub order: i32,
}

/// MFEM `NURBSFECollection` — the scalar H1 NURBS collection.
#[derive(Debug, Clone)]
pub struct NurbsFECollection {
    order: i32,
}

impl Default for NurbsFECollection {
    fn default() -> Self {
        Self::new(1)
    }
}

impl NurbsFECollection {
    /// MFEM `NURBSFECollection(int Order)`; `VARIABLE_ORDER` maps to 1.
    pub fn new(order: i32) -> Self {
        Self { order }
    }

    /// MFEM `FiniteElementCollection::GetOrder()`.
    pub fn order(&self) -> i32 {
        self.order
    }

    /// The `GetOrder() == VariableOrder` test.
    pub fn is_variable_order(&self) -> bool {
        self.order == VARIABLE_ORDER
    }

    /// MFEM `NURBSFECollection::name` — `"NURBS%i"` or `"NURBS"`.
    pub fn name(&self) -> String {
        if self.is_variable_order() {
            "NURBS".to_string()
        } else {
            format!("NURBS{}", self.order)
        }
    }

    /// MFEM `NURBSFECollection::FiniteElementForGeometry`.
    ///
    /// Returns `None` for geometries the collection does not know, mirroring
    /// the `error_mode == RETURN_NULL` behaviour (the default `error_mode`
    /// aborts).
    pub fn finite_element_for_geometry(&self, geom: NurbsGeometry) -> Option<NurbsFEEntry> {
        let order = if self.is_variable_order() { 1 } else { self.order };
        let element = match geom {
            NurbsGeometry::Point => NurbsElement::Point,
            NurbsGeometry::Segment => NurbsElement::Nurbs1D,
            NurbsGeometry::Square => NurbsElement::Nurbs2D,
            NurbsGeometry::Cube => NurbsElement::Nurbs3D,
        };
        Some(NurbsFEEntry { element, order })
    }

    /// `NURBSFECollection::DofForGeometry` calls `mfem_error` in MFEM: NURBS
    /// DOFs are not attached to vertices/edges/faces, so the query is
    /// meaningless.  Modelled as an explicit error.
    pub fn dof_for_geometry(&self, _geom: NurbsGeometry) -> Result<usize, String> {
        Err("NURBSFECollection::DofForGeometry".to_string())
    }
}

/// MFEM `NURBS_HDivFECollection` — divergence-conforming NURBS collection.
#[derive(Debug, Clone)]
pub struct NurbsHDivFECollection {
    order: i32,
    dim: i32,
}

impl NurbsHDivFECollection {
    /// MFEM `NURBS_HDivFECollection(int Order, const int dim)`.
    ///
    /// `dim` may be `-1`, in which case [`Self::set_dim`] must be called before
    /// the collection is used — MFEM leaves `sFE` / `qFE` / `hFE` null then.
    pub fn new(order: i32, dim: i32) -> Result<Self, String> {
        let mut c = Self { order, dim: -1 };
        if dim != -1 {
            c.set_dim(dim)?;
        }
        Ok(c)
    }

    /// MFEM `NURBS_HDivFECollection::SetDim`.
    pub fn set_dim(&mut self, dim: i32) -> Result<(), String> {
        if dim != 2 && dim != 3 {
            return Err(format!("NURBS_HDivFECollection: wrong dimension! {dim}"));
        }
        self.dim = dim;
        Ok(())
    }

    /// MFEM `FiniteElementCollection::GetOrder()`, with `VariableOrder` mapped
    /// to 1 for the element constructors.
    pub fn order(&self) -> i32 {
        if self.order == VARIABLE_ORDER { 1 } else { self.order }
    }

    /// The `GetOrder()` value as configured (may be `VariableOrder`).
    pub fn raw_order(&self) -> i32 {
        self.order
    }

    /// The dimension set by `set_dim`, or `-1` if unset.
    pub fn dim(&self) -> i32 {
        self.dim
    }

    /// MFEM `NURBS_HDivFECollection::name` — `"NURBS_HDiv%i"` / `"NURBS_HDiv"`.
    pub fn name(&self) -> String {
        if self.order == VARIABLE_ORDER {
            "NURBS_HDiv".to_string()
        } else {
            format!("NURBS_HDiv{}", self.order)
        }
    }

    /// MFEM `NURBS_HDivFECollection::FiniteElementForGeometry`.
    ///
    /// Mirrors the `sFE` / `qFE` / `hFE` selection done by `SetDim`:
    ///
    /// | dim | SEGMENT | SQUARE | CUBE |
    /// |-----|---------|--------|------|
    /// | 2 | `NURBS1D(Order)` | `NURBS_HDiv2D(Order)` | — |
    /// | 3 | — | `NURBS2D(Order)` | `NURBS_HDiv3D(Order)` |
    pub fn finite_element_for_geometry(&self, geom: NurbsGeometry) -> Option<NurbsFEEntry> {
        let order = self.order();
        let element = match (self.dim, geom) {
            (2, NurbsGeometry::Segment) => NurbsElement::Nurbs1D,
            (2, NurbsGeometry::Square) => NurbsElement::HDiv2D,
            (3, NurbsGeometry::Square) => NurbsElement::Nurbs2D,
            (3, NurbsGeometry::Cube) => NurbsElement::HDiv3D,
            // `dim == -1` leaves every pointer null.
            _ => return None,
        };
        Some(NurbsFEEntry { element, order })
    }

    /// `NURBS_HDivFECollection::DofForGeometry` aborts in MFEM.
    pub fn dof_for_geometry(&self, _geom: NurbsGeometry) -> Result<usize, String> {
        Err("NURBS_HDivFECollection::DofForGeometry".to_string())
    }
}

/// MFEM `NURBS_HCurlFECollection` — curl-conforming NURBS collection.
#[derive(Debug, Clone)]
pub struct NurbsHCurlFECollection {
    order: i32,
    dim: i32,
}

impl NurbsHCurlFECollection {
    /// MFEM `NURBS_HCurlFECollection(int Order, const int dim)`.
    pub fn new(order: i32, dim: i32) -> Result<Self, String> {
        let mut c = Self { order, dim: -1 };
        if dim != -1 {
            c.set_dim(dim)?;
        }
        Ok(c)
    }

    /// MFEM `NURBS_HCurlFECollection::SetDim`.
    pub fn set_dim(&mut self, dim: i32) -> Result<(), String> {
        if dim != 2 && dim != 3 {
            return Err(format!("NURBS_HCurlFECollection: wrong dimension! {dim}"));
        }
        self.dim = dim;
        Ok(())
    }

    /// MFEM `FiniteElementCollection::GetOrder()`, with `VariableOrder` mapped
    /// to 1 for the element constructors.
    pub fn order(&self) -> i32 {
        if self.order == VARIABLE_ORDER { 1 } else { self.order }
    }

    /// The `GetOrder()` value as configured (may be `VariableOrder`).
    pub fn raw_order(&self) -> i32 {
        self.order
    }

    /// The dimension set by `set_dim`, or `-1` if unset.
    pub fn dim(&self) -> i32 {
        self.dim
    }

    /// MFEM `NURBS_HCurlFECollection::name` — `"NURBS_HCurl%i"` / `"NURBS_HCurl"`.
    pub fn name(&self) -> String {
        if self.order == VARIABLE_ORDER {
            "NURBS_HCurl".to_string()
        } else {
            format!("NURBS_HCurl{}", self.order)
        }
    }

    /// MFEM `NURBS_HCurlFECollection::FiniteElementForGeometry`.
    ///
    /// The scalar helper elements use `Order + 1`, matching the constructor
    /// (`SegmentFE = new NURBS1DFiniteElement(order+1)`,
    /// `QuadrilateralFE = new NURBS2DFiniteElement(order+1)`), while the vector
    /// elements use `Order`.
    ///
    /// | dim | SEGMENT | SQUARE | CUBE |
    /// |-----|---------|--------|------|
    /// | 2 | `NURBS1D(Order+1)` | `NURBS_HCurl2D(Order)` | — |
    /// | 3 | — | `NURBS2D(Order+1)` | `NURBS_HCurl3D(Order)` |
    pub fn finite_element_for_geometry(&self, geom: NurbsGeometry) -> Option<NurbsFEEntry> {
        let order = self.order();
        let element = match (self.dim, geom) {
            (2, NurbsGeometry::Segment) => NurbsElement::Nurbs1D,
            (2, NurbsGeometry::Square) => NurbsElement::HCurl2D,
            (3, NurbsGeometry::Square) => NurbsElement::Nurbs2D,
            (3, NurbsGeometry::Cube) => NurbsElement::HCurl3D,
            _ => return None,
        };
        let order = match element {
            NurbsElement::Nurbs1D | NurbsElement::Nurbs2D => order + 1,
            _ => order,
        };
        Some(NurbsFEEntry { element, order })
    }

    /// `NURBS_HCurlFECollection::DofForGeometry` aborts in MFEM.
    pub fn dof_for_geometry(&self, _geom: NurbsGeometry) -> Result<usize, String> {
        Err("NURBS_HCurlFECollection::DofForGeometry".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clamped(knots: Vec<f64>) -> KnotVector {
        KnotVector::new_clamped(knots).expect("valid knots")
    }

    /// Knot vectors taken verbatim from `data/*.mesh`.
    mod mesh_kv {
        use super::clamped;
        use crate::iga::KnotVector;

        /// `square-nurbs.mesh`: `1 2 0 0 1 1`
        pub fn unit_linear() -> KnotVector {
            clamped(vec![0.0, 0.0, 1.0, 1.0])
        }
        /// `beam-quad-nurbs.mesh` KV0: `1 5 0 0 1 2 3 4 4`
        pub fn four_span_linear() -> KnotVector {
            clamped(vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0])
        }
        /// `disc-nurbs.mesh` KV0: `2 3 0 0 0 1 1 1`
        pub fn unit_quadratic() -> KnotVector {
            clamped(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        }
    }

    /// Span-local `KnotVector::CalcDShape` values, dumped from MFEM 4.9 for
    /// `KnotVector kv(2, 4); kv = {0,0,0,0.5,1,1,1};` on element 0 (the span
    /// `[knot[2], knot[3]] = [0, 0.5]`) at the span-local coordinates
    /// `xi = 0, 0.25, 0.5, 0.75, 1`.
    #[test]
    fn knot_vector_dshape_matches_mfem() {
        let kv = clamped(vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]);
        let want = [
            (0.0, [-2.0, 2.0, 0.0]),
            (0.25, [-1.5, 1.25, 0.25]),
            (0.5, [-1.0, 0.5, 0.5]),
            (0.75, [-0.5, -0.25, 0.75]),
            (1.0, [0.0, -1.0, 1.0]),
        ];
        for (xi, expected) in want {
            let u = 0.5 * xi; // GetKnotLocation(xi, 0 + 2)
            let mut dshape = [0.0; 3];
            span_local_basis_deriv(kv.as_slice(), 2, 0, u, &mut dshape);
            for o in 0..3 {
                assert!(
                    (dshape[o] - expected[o]).abs() < 1e-15,
                    "xi = {xi}, o = {o}: {} != {}",
                    dshape[o],
                    expected[o]
                );
            }
            assert!(dshape.iter().sum::<f64>().abs() < 1e-15);
        }
    }

    /// `NURBS2DFiniteElement::CalcShape` / `CalcDShape` on a **multi-span**
    /// patch: `square-nurbs.mesh` refined once, space order 2, knot vectors
    /// `{0,0,0,0.5,1,1,1}`, element 0 (`ijk = 0,0`), span-local
    /// `xi = (0.5, 0.5)`.  Values dumped from MFEM 4.9.
    #[test]
    fn nurbs2d_multispan_values_match_mfem() {
        let kv = clamped(vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]);
        let mut e = NurbsScalar2D::new(kv.clone(), kv).unwrap();
        e.set_ijk([0, 0]);
        assert_eq!(e.n_dofs(), 9);

        let want_sh = [
            0.0625, 0.15625, 0.03125, 0.15625, 0.390625, 0.078125, 0.03125, 0.078125, 0.015625,
        ];
        let mut sh = [0.0; 9];
        e.calc_shape(&[0.5, 0.5], &mut sh);
        for i in 0..9 {
            assert!((sh[i] - want_sh[i]).abs() < 1e-16, "i = {i}: {}", sh[i]);
        }

        let want_dsh = [
            -0.25, -0.25, 0.125, -0.625, 0.125, -0.125, -0.625, 0.125, 0.3125, 0.3125, 0.3125,
            0.0625, -0.125, 0.125, 0.0625, 0.3125, 0.0625, 0.0625,
        ];
        let mut dsh = [0.0; 18];
        e.calc_grad(&[0.5, 0.5], &mut dsh);
        for i in 0..18 {
            assert!((dsh[i] - want_dsh[i]).abs() < 1e-16, "i = {i}: {}", dsh[i]);
        }
    }

    /// `NURBS3DFiniteElement` on a multi-span patch: `cube-nurbs.mesh` refined
    /// once, space order 2, knot vectors `{0,0,0,0.5,1,1,1}` in all three
    /// directions, element 0 (`ijk = 0,0,0`), span-local
    /// `xi = (0.5, 0.11270166537925831, 0.11270166537925831)`.  Values dumped
    /// from MFEM 4.9 (the first `IntRules.Get(CUBE, 5)` point of the rule's
    /// second row).
    #[test]
    fn nurbs3d_multispan_values_match_mfem() {
        let kv = clamped(vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]);
        let mut e = NurbsScalar3D::new(kv.clone(), kv.clone(), kv).unwrap();
        e.set_ijk([0, 0, 0]);
        assert_eq!(e.n_dofs(), 27);

        let xi = [0.5, 0.11270166537925831, 0.11270166537925831];
        let want_sh = [
            0.15495966692414834, 0.38739916731037083, 0.077479833462074169, 0.040614916731037079,
            0.10153729182759269, 0.02030745836551854, 0.0012499999999999998, 0.0031249999999999993,
            0.0006249999999999999, 0.040614916731037079, 0.10153729182759269, 0.02030745836551854,
            0.010645166537925832, 0.026612916344814577, 0.0053225832689629158, 0.00032762490344437334,
            0.0008190622586109333, 0.00016381245172218667, 0.0012499999999999998, 0.0031249999999999993,
            0.0006249999999999999, 0.00032762490344437334, 0.0008190622586109333, 0.00016381245172218667,
            1.0083268962915572e-05, 2.5208172407288925e-05, 5.0416344814577859e-06,
        ];
        let mut sh = [0.0; 27];
        e.calc_shape(&xi, &mut sh);
        for i in 0..27 {
            assert!(
                (sh[i] - want_sh[i]).abs() < 1e-15,
                "shape i = {i}: {} != {}",
                sh[i],
                want_sh[i]
            );
        }

        let want_dsh = [
            -0.61983866769659335, -0.34928425057933377, -0.34928425057933371, 0.30991933384829667,
            -0.87321062644833425, -0.87321062644833414, 0.30991933384829667, -0.17464212528966688,
            -0.17464212528966686, -0.16245966692414832, 0.32710179221381519, -0.091547375096555625,
            0.081229833462074158, 0.81775448053453792, -0.22886843774138904, 0.081229833462074158,
            0.1635508961069076, -0.045773687548277812, -0.0049999999999999992, 0.022182458365518541,
            -0.0028175416344814572, 0.0024999999999999996, 0.055456145913796349, -0.0070438540862036419,
            0.0024999999999999996, 0.011091229182759271, -0.0014087708172407286, -0.16245966692414832,
            -0.091547375096555639, 0.32710179221381519, 0.081229833462074158, -0.2288684377413891,
            0.81775448053453792, 0.081229833462074158, -0.045773687548277819, 0.1635508961069076,
            -0.042580666151703327, 0.085733354472426151, 0.085733354472426151, 0.021290333075851663,
            0.21433338618106534, 0.21433338618106534, 0.021290333075851663, 0.042866677236213076,
            0.042866677236213076, -0.0013104996137774934, 0.0058140206241294751, 0.0026386042793148973,
            0.00065524980688874668, 0.014535051560323687, 0.0065965106982872421, 0.00065524980688874668,
            0.0029070103120647376, 0.0013193021396574486, -0.0049999999999999992, -0.0028175416344814576,
            0.022182458365518541, 0.0024999999999999996, -0.0070438540862036427, 0.055456145913796349,
            0.0024999999999999996, -0.0014087708172407288, 0.011091229182759271, -0.0013104996137774934,
            0.0026386042793148973, 0.0058140206241294751, 0.00065524980688874668, 0.0065965106982872421,
            0.014535051560323687, 0.00065524980688874668, 0.0013193021396574486, 0.0029070103120647376,
            -4.0333075851662287e-05, 0.00017893735516656001, 0.00017893735516656001,
            2.0166537925831144e-05, 0.00044734338791639996, 0.00044734338791639996,
            2.0166537925831144e-05, 8.9468677583280003e-05, 8.9468677583280003e-05,
        ];
        let mut dsh = [0.0; 81];
        e.calc_grad(&xi, &mut dsh);
        for i in 0..81 {
            assert!(
                (dsh[i] - want_dsh[i]).abs() < 1e-15,
                "grad i = {i}: {} != {}",
                dsh[i],
                want_dsh[i]
            );
        }
    }

    /// Rational weights enter exactly through MFEM's `shape *= weights / sum`.
    #[test]
    fn nurbs2d_rational_weights_are_a_partition_of_unity() {
        let kv = clamped(vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]);
        let mut e = NurbsScalar2D::new(kv.clone(), kv).unwrap();
        assert_eq!(e.set_weights(vec![1.0]).is_err(), true);
        let mut w = vec![1.0; 9];
        w[4] = 0.5;
        w[8] = 2.0;
        e.set_weights(w).unwrap();
        let mut sh = [0.0; 9];
        for q in [0.0, 0.25, 0.5, 0.75, 1.0] {
            e.set_ijk([0, 0]);
            e.calc_shape(&[q, 1.0 - q], &mut sh);
            let sum: f64 = sh.iter().sum();
            assert!((sum - 1.0).abs() < 1e-13, "sum = {sum}");
        }
    }

    #[test]
    fn knot_vector_queries_match_mfem() {
        let kv = mesh_kv::unit_linear();
        assert_eq!(knot_order(&kv), Some(1));
        assert_eq!(knot_ncp(&kv), Some(2));
        assert_eq!(knot_n_elements(&kv), Some(1));
        assert_eq!(knot_nks(&kv), Some(1));

        let kv = mesh_kv::four_span_linear();
        assert_eq!(knot_order(&kv), Some(1));
        assert_eq!(knot_ncp(&kv), Some(5));
        assert_eq!(knot_n_elements(&kv), Some(4));

        let kv = mesh_kv::unit_quadratic();
        assert_eq!(knot_order(&kv), Some(2));
        assert_eq!(knot_ncp(&kv), Some(3));
        assert_eq!(knot_n_elements(&kv), Some(1));
    }

    #[test]
    fn degree_elevate_matches_mfem() {
        // Single span, order 1 -> order 2 with 3 control points.
        let e = degree_elevate(&mesh_kv::unit_linear(), 1).unwrap();
        assert_eq!(e.as_slice(), &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(knot_order(&e), Some(2));
        assert_eq!(knot_ncp(&e), Some(3));

        // Multi span: MFEM shifts the interior knots and keeps the element count.
        let e = degree_elevate(&mesh_kv::four_span_linear(), 1).unwrap();
        assert_eq!(e.as_slice(), &[0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0]);
        assert_eq!(knot_order(&e), Some(2));
        assert_eq!(knot_ncp(&e), Some(6));
        assert_eq!(knot_n_elements(&e), Some(4));
    }

    #[test]
    fn scalar_collection_dispatch() {
        let c = NurbsFECollection::new(2);
        assert_eq!(c.name(), "NURBS2");
        assert_eq!(c.order(), 2);
        assert!(!c.is_variable_order());
        let e = c.finite_element_for_geometry(NurbsGeometry::Square).unwrap();
        assert_eq!(e.element, NurbsElement::Nurbs2D);
        assert_eq!(e.order, 2);
        assert_eq!(e.element.n_dofs(&[2, 2]), 9);

        let v = NurbsFECollection::new(VARIABLE_ORDER);
        assert_eq!(v.name(), "NURBS");
        assert!(v.is_variable_order());
        // VariableOrder falls back to order 1 for the element constructors.
        let e = v.finite_element_for_geometry(NurbsGeometry::Cube).unwrap();
        assert_eq!(e.order, 1);
        assert_eq!(e.element, NurbsElement::Nurbs3D);
        assert_eq!(e.element.n_dofs(&[1, 1, 1]), 8);

        assert_eq!(NurbsGeometry::from_mfem_code(5), Some(NurbsGeometry::Cube));
        assert_eq!(NurbsGeometry::Square.mfem_code(), 3);
        assert_eq!(NurbsGeometry::from_mfem_code(2), None);
    }

    #[test]
    fn hdiv_collection_dim_dispatch() {
        let mut c = NurbsHDivFECollection::new(2, -1).unwrap();
        assert_eq!(c.dim(), -1);
        // With dim unset every geometry returns a null element (`sFE` etc.).
        assert!(c.finite_element_for_geometry(NurbsGeometry::Square).is_none());
        c.set_dim(2).unwrap();
        assert_eq!(c.name(), "NURBS_HDiv2");
        assert_eq!(
            c.finite_element_for_geometry(NurbsGeometry::Segment).unwrap().element,
            NurbsElement::Nurbs1D
        );
        assert_eq!(
            c.finite_element_for_geometry(NurbsGeometry::Square).unwrap().element,
            NurbsElement::HDiv2D
        );
        assert!(c.finite_element_for_geometry(NurbsGeometry::Cube).is_none());

        let c = NurbsHDivFECollection::new(1, 3).unwrap();
        assert_eq!(
            c.finite_element_for_geometry(NurbsGeometry::Square).unwrap().element,
            NurbsElement::Nurbs2D
        );
        assert_eq!(
            c.finite_element_for_geometry(NurbsGeometry::Cube).unwrap().element,
            NurbsElement::HDiv3D
        );
        assert!(c.finite_element_for_geometry(NurbsGeometry::Segment).is_none());

        let mut bad = NurbsHDivFECollection::new(1, 2).unwrap();
        assert!(bad.set_dim(4).is_err());
    }

    #[test]
    fn hcurl_collection_uses_order_plus_one_for_scalar_helpers() {
        let c = NurbsHCurlFECollection::new(2, 3).unwrap();
        assert_eq!(c.name(), "NURBS_HCurl2");
        let sq = c.finite_element_for_geometry(NurbsGeometry::Square).unwrap();
        assert_eq!(sq.element, NurbsElement::Nurbs2D);
        assert_eq!(sq.order, 3);
        let cube = c.finite_element_for_geometry(NurbsGeometry::Cube).unwrap();
        assert_eq!(cube.element, NurbsElement::HCurl3D);
        assert_eq!(cube.order, 2);

        let c = NurbsHCurlFECollection::new(1, 2).unwrap();
        let seg = c.finite_element_for_geometry(NurbsGeometry::Segment).unwrap();
        assert_eq!(seg.element, NurbsElement::Nurbs1D);
        assert_eq!(seg.order, 2);
    }

    #[test]
    fn element_dof_counts_match_mfem_formulas() {
        // NURBS_HDiv*/NURBS_HCurl*::SetOrder, uniform order p.
        for p in 1..=4usize {
            assert_eq!(
                NurbsElement::HDiv2D.n_dofs(&[p, p]),
                (p + 2) * (p + 1) + (p + 1) * (p + 2)
            );
            assert_eq!(
                NurbsElement::HCurl2D.n_dofs(&[p, p]),
                (p + 1) * (p + 2) + (p + 2) * (p + 1)
            );
            assert_eq!(
                NurbsElement::HDiv3D.n_dofs(&[p, p, p]),
                3 * (p + 1) * (p + 1) * (p + 2)
            );
            assert_eq!(
                NurbsElement::HCurl3D.n_dofs(&[p, p, p]),
                3 * (p + 1) * (p + 2) * (p + 2)
            );
        }
        // The order reported by the vector elements is the elevated degree.
        assert_eq!(NurbsElement::HDiv2D.order(&[1, 1]), 2);
        assert_eq!(NurbsElement::HCurl3D.order(&[2, 2, 2]), 3);
        assert_eq!(NurbsElement::Nurbs2D.order(&[2, 2]), 2);
        assert_eq!(NurbsElement::HDiv2D.dim(), 2);
        assert_eq!(NurbsElement::HCurl3D.dim(), 3);
        assert!(NurbsElement::HDiv3D.is_vector());
        assert!(!NurbsElement::Nurbs3D.is_vector());
    }

    #[test]
    fn element_construction_from_mesh_knot_vectors() {
        // square-nurbs: order 1, single span -> 4 H1 DOFs per patch element.
        let kv = mesh_kv::unit_linear();
        let e = NurbsElement::Nurbs2D.build(&[kv.clone(), kv.clone()]).unwrap();
        assert_eq!(e.n_dofs(), 4);

        // disc-nurbs: order 2 -> 9 DOFs.
        let kv = mesh_kv::unit_quadratic();
        let e = NurbsElement::Nurbs2D.build(&[kv.clone(), kv.clone()]).unwrap();
        assert_eq!(e.n_dofs(), 9);

        // H(div): order 1 -> 12 DOFs, values finite and each component a
        // partition of unity on the mixed-degree bases.
        let kv = mesh_kv::unit_linear();
        let e = NurbsElement::HDiv2D.build(&[kv.clone(), kv.clone()]).unwrap();
        assert_eq!(e.n_dofs(), 12);
        let mut vals = vec![0.0; 12 * 2];
        if let NurbsRefElement::HDiv2D(el) = &e {
            el.eval_basis_vec(&[0.5, 0.5], &mut vals);
        }
        let sum_x: f64 = (0..12).map(|i| vals[2 * i]).sum();
        let sum_y: f64 = (0..12).map(|i| vals[2 * i + 1]).sum();
        assert!((sum_x - 1.0).abs() < 1e-12, "sum_x = {sum_x}");
        assert!((sum_y - 1.0).abs() < 1e-12, "sum_y = {sum_y}");

        // Wrong knot vector count is rejected.
        assert!(NurbsElement::HDiv3D.build(&[kv.clone(), kv]).is_err());
    }

    #[test]
    fn nurbs1d_rational_shape_is_partition_of_unity() {
        let mut e = Nurbs1DFiniteElement::new(mesh_kv::unit_quadratic()).unwrap();
        assert_eq!(e.order(), 2);
        assert_eq!(e.n_dofs(), 3);
        assert_eq!(e.n_elements(), 1);
        e.set_weights(vec![1.0, 0.5, 1.0]).unwrap();

        let mut shape = vec![0.0; 3];
        for &x in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            e.set_ijk(0);
            e.calc_shape(x, &mut shape);
            let sum: f64 = shape.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "x = {x}, sum = {sum}");
        }
        e.set_ijk(0);
        e.calc_shape(0.0, &mut shape);
        assert!((shape[0] - 1.0).abs() < 1e-12, "shape = {shape:?}");

        // Rational derivatives sum to zero (the basis is a partition of unity).
        let mut dshape = vec![0.0; 3];
        e.set_ijk(0);
        e.calc_dshape(0.5, &mut dshape);
        let sum: f64 = dshape.iter().sum();
        assert!(sum.abs() < 1e-11, "sum dshape = {sum}");

        // A rejected weight count.
        assert!(e.set_weights(vec![1.0, 1.0]).is_err());
    }

    #[test]
    fn nurbs1d_multi_span_shape_is_partition_of_unity() {
        // beam-quad-nurbs KV0: 4 spans; shape must be a partition of unity on
        // every span and the derivative must sum to zero.
        let e = Nurbs1DFiniteElement::new(mesh_kv::four_span_linear()).unwrap();
        assert_eq!(e.order(), 1);
        assert_eq!(e.n_dofs(), 2);
        assert_eq!(e.n_elements(), 4);
        assert_eq!(e.knot_location(0.0, 0), 0.0);
        assert_eq!(e.knot_location(1.0, 0), 1.0);
        assert_eq!(e.knot_location(0.5, 2), 2.5);
        let mut shape = vec![0.0; 2];
        for span in 0..4 {
            let mut e = e.clone();
            e.set_ijk(span);
            for &x in &[0.0, 0.3, 1.0] {
                e.calc_shape(x, &mut shape);
                let sum: f64 = shape.iter().sum();
                assert!((sum - 1.0).abs() < 1e-12, "span {span} x = {x}, sum = {sum}");
            }
        }
    }

    #[test]
    fn scalar_2d_matches_patch_element_in_parameter_space() {
        // With a single span the span-local and patch-parameter conventions
        // coincide (the span is the whole [0,1] interval and every DOF is
        // active), so the span-local element must reproduce the patch element.
        use crate::reference::ReferenceElement;
        let kv = mesh_kv::unit_quadratic();
        let mut e = NurbsScalar2D::new(kv.clone(), kv.clone()).unwrap();
        e.set_ijk([0, 0]);
        assert_eq!(e.order(), 2);
        assert_eq!(e.n_dofs(), 9);
        let legacy = |k: &KnotVector| crate::nurbs::KnotVector {
            knots: k.as_slice().to_vec(),
            degree: knot_order(k).expect("valid knot vector"),
        };
        let patch = crate::nurbs::NurbsPatch2D::uniform(legacy(&kv), legacy(&kv));
        let mut got = vec![0.0; 9];
        e.calc_shape(&[0.3, 0.7], &mut got);
        let mut want = vec![0.0; 9];
        patch.eval_basis(&[0.3, 0.7], &mut want);
        for i in 0..9 {
            assert!((got[i] - want[i]).abs() < 1e-14, "i = {i}");
        }
        let sum: f64 = got.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn variable_order_name_has_no_number() {
        assert_eq!(
            NurbsHDivFECollection::new(VARIABLE_ORDER, 2).unwrap().name(),
            "NURBS_HDiv"
        );
        assert_eq!(
            NurbsHCurlFECollection::new(VARIABLE_ORDER, 3).unwrap().name(),
            "NURBS_HCurl"
        );
        assert_eq!(
            NurbsHDivFECollection::new(VARIABLE_ORDER, 2)
                .unwrap()
                .finite_element_for_geometry(NurbsGeometry::Square)
                .unwrap()
                .order,
            1
        );
    }

    #[test]
    fn dof_for_geometry_is_an_error_like_mfem() {
        let c = NurbsFECollection::new(1);
        assert!(c.dof_for_geometry(NurbsGeometry::Point).is_err());
        assert!(NurbsHDivFECollection::new(1, 2)
            .unwrap()
            .dof_for_geometry(NurbsGeometry::Square)
            .is_err());
        assert!(NurbsHCurlFECollection::new(1, 3)
            .unwrap()
            .dof_for_geometry(NurbsGeometry::Cube)
            .is_err());
    }
}
