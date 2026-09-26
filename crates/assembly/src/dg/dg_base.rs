//! Common DG utilities — shared across all DG assembly modules.
//!
//! Extracted to eliminate duplicated copies of reference-element dispatch,
//! Jacobian computation, gradient transforms, and face-geometry helpers.
//!
//! Each function is individually documented so callers can find what they need
//! without reading the implementation.

use std::collections::HashMap;
use nalgebra::DMatrix;

use fem_element::{
    ReferenceElement,
    lagrange::{
        SegP1, SegP2, SegP3,
        TriP1,
        TetL2GL, TriL2GL,
        QuadQk,
        factory::{HexL2GL, QuadL2GL},
    },
};
use fem_mesh::{element_type::ElementType, topology::MeshTopology};
use fem_mesh::transformation::element_jacobian_at;

// ═══════════════════════════════════════════════════════════════════════════════
// Reference-element dispatch
// ═══════════════════════════════════════════════════════════════════════════════

/// Return the volume reference element for a given element type and polynomial
/// order.  Supports triangles (Tri3), quadrilaterals (Quad4), and tetrahedra
/// (Tet4) up to order 3, with dynamic-order QuadQk for Quad4 at order > 2.
///
/// All arms use the `L2_FECollection` **default `GaussLegendre`** nodal
/// placement (MFEM 4.10's `DG_FECollection` is a typedef of `L2_FECollection`):
/// open GL tensor nodes on Quad4 ([`QuadL2GL`], round-40 precedent) and open
/// barycentric GL nodes on Tri3/Tet4 ([`TriL2GL`]/[`TetL2GL`], D269).
// MFEM: FECollection::FiniteElementForGeometry
pub fn ref_elem_vol(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        // D269: simplex DG volume elements use the MFEM open barycentric GL
        // nodes (same precedent as the Quad4 arm, round 40).
        (ElementType::Tri3, 1) => Box::new(TriL2GL::new(1)),
        (ElementType::Tri3, 2) => Box::new(TriL2GL::new(2)),
        (ElementType::Tri3, 3) => Box::new(TriL2GL::new(3)),
        // Quad4 in the DG/L2 path uses the Gauss-Legendre tensor basis
        // (MFEM L2_FECollection / DG_FECollection default), on [0,1]².
        (ElementType::Quad4, 1) => Box::new(QuadL2GL::new(1)),
        (ElementType::Quad4, 2) => Box::new(QuadL2GL::new(2)),
        (ElementType::Quad4, 3) => Box::new(QuadL2GL::new(3)),
        (ElementType::Tet4, 1) => Box::new(TetL2GL::new(1)),
        (ElementType::Tet4, 2) => Box::new(TetL2GL::new(2)),
        (ElementType::Tet4, 3) => Box::new(TetL2GL::new(3)),
        // D814-1: hexahedral DG volume elements — MFEM `L2_HexahedronElement`
        // (open GL tensor nodes, lexicographic order; [`HexL2GL`]).  Before
        // this arm every DG path panicked on a hexahedral mesh.
        (ElementType::Hex8, o) if o >= 1 => Box::new(HexL2GL::new(o as usize)),
        _ => panic!("ref_elem_vol: unsupported ({et:?}, order={order})"),
    }
}

/// Dynamic-order volume reference element — wraps `ref_elem_vol` but falls
/// back to [`QuadQk`] for Quad4 at arbitrary order > 2.
// MFEM: FECollection::FiniteElementForGeometry
pub fn ref_elem_vol_dynamic(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Quad4, order) if order > 2 => Box::new(QuadQk::new_lex(order as usize)),
        _ => ref_elem_vol(et, order),
    }
}

/// Face reference element (Line2 for 2-D, Tri3/Quad4 for 3-D).
///
/// The DG face drivers consume this element **only for its quadrature rule**
/// (MFEM takes the rule from `IntRules.Get(Trans.GetGeometryType(), ...)`,
/// where the geometry is the *face's*: `SEGMENT` in 2-D, `TRIANGLE`/`SQUARE`
/// in 3-D).  D814-1: the `Quad4` arms make a hexahedral face take MFEM's
/// `[0,1]²` tensor Gauss rule ([`QuadL2GL::quadrature`] = `quad_rule_01`)
/// instead of panicking.
// MFEM: FECollection::FiniteElementForGeometry (face)
pub fn ref_elem_face(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Line2, 1) => Box::new(SegP1),
        (ElementType::Line2, 2) => Box::new(SegP2),
        (ElementType::Line2, 3) => Box::new(SegP3),
        (ElementType::Tri3, 1)  => Box::new(TriP1),
        // D815-1: higher-order triangular faces (the tetrahedron's faces in an
        // order-2/3 DG space).  The face element is consumed only for its
        // quadrature rule (`IntRules.Get(TRIANGLE, 2·max(o₁,o₂))`); TriP2/P3
        // carry exactly MFEM's triangle rules at those orders.
        (ElementType::Tri3, 2)  => Box::new(fem_element::lagrange::TriP2),
        (ElementType::Tri3, 3)  => Box::new(fem_element::lagrange::TriP3),
        (ElementType::Quad4, o) if o >= 1 => Box::new(QuadL2GL::new(o as usize)),
        _ => panic!("ref_elem_face: unsupported ({et:?}, order={order})"),
    }
}

/// The face reference-element type for a mesh face, chosen by the face's
/// **node count** (D814-1): 2 nodes → `Line2` (2-D edge), 3 → `Tri3`
/// (tetrahedron face), 4 → `Quad4` (hexahedron face).  The pre-D814 dispatch
/// keyed on `mesh.dim()` instead (`dim == 2 ? Line2 : Tri3`), which silently
/// gave every 3-D face the triangular rule — wrong for a hexahedron's
/// quadrilateral faces, both in point layout (`[0,1]` triangle vs `[0,1]²`
/// tensor) and point count.
// MFEM: FaceElementTransformations::GetGeometryType()
pub fn face_type_of(face_nodes: &[u32]) -> ElementType {
    match face_nodes.len() {
        2 => ElementType::Line2,
        3 => ElementType::Tri3,
        4 => ElementType::Quad4,
        n => panic!("face_type_of: unsupported face node count {n}"),
    }
}

/// Return a Crouzeix-Raviart reference element by type and order.
// MFEM: FECollection (Crouzeix-Raviart)
pub fn ref_elem_cr(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Tri3, 1) => Box::new(fem_element::CrTri1),
        (ElementType::Tri3, 2) => Box::new(fem_element::CrTri2),
        (ElementType::Tet4, 1) => Box::new(fem_element::CrTet1),
        (ElementType::Tet4, 2) => Box::new(fem_element::CrTet2),
        _ => panic!("ref_elem_cr: unsupported ({et:?}, order={order})"),
    }
}

/// Return a Q1_rot (Rannacher–Turek) reference element for Quad4.
// MFEM: FECollection (Rannacher-Turek Q1_rot)
pub fn ref_elem_q1rot(et: ElementType) -> Box<dyn ReferenceElement> {
    match et {
        ElementType::Quad4 => Box::new(fem_element::Q1RotRef),
        _ => panic!("ref_elem_q1rot: unsupported {et:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Jacobian helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Affine Jacobian of a simplex (Tri/Tet) or centroid Jacobian for a bilinear
/// quad, returned as a `DMatrix` together with its determinant.
///
/// For elements with >3 nodes (quadrilaterals in 2-D) the centroid Jacobian of
/// the bilinear mapping on `[-1,1]²` is computed, scaled by 0.5 to match the
/// `[0,1]` reference-domain convention used by `phys_to_ref`.
// MFEM: CalcJacobian
pub fn simplex_jac<M: MeshTopology>(
    mesh: &M,
    nodes: &[u32],
    dim: usize,
) -> (DMatrix<f64>, f64) {
    if nodes.len() > 3 {
        // Quadrilateral — centroid Jacobian of bilinear mapping on [-1,1]²,
        // then scaled by 0.5 for [0,1]-based reference coordinates.
        let x: Vec<f64> = (0..4).map(|k| mesh.node_coords(nodes[k.min(3)])[0]).collect();
        let y: Vec<f64> = (0..4).map(|k| mesh.node_coords(nodes[k.min(3)])[1]).collect();
        let dxi  = [-0.5,  0.5,  0.5, -0.5];
        let deta = [-0.5, -0.5,  0.5,  0.5];
        let mut j = DMatrix::<f64>::zeros(dim, dim);
        for k in 0..4 {
            j[(0, 0)] += dxi[k]  * x[k];
            j[(0, 1)] += deta[k] * x[k];
            j[(1, 0)] += dxi[k]  * y[k];
            j[(1, 1)] += deta[k] * y[k];
        }
        let det = j.determinant();
        return (j, det);
    }
    // Simplex: affine mapping from [0,1]^dim
    let x0 = mesh.node_coords(nodes[0]);
    let mut j = DMatrix::<f64>::zeros(dim, dim);
    for col in 0..dim {
        let xc = mesh.node_coords(nodes[col + 1]);
        for row in 0..dim {
            j[(row, col)] = xc[row] - x0[row];
        }
    }
    let det = j.determinant();
    (j, det)
}

/// Bilinear quad Jacobian at a reference point `(xi, eta)` in `[-1,1]²`.
// MFEM: CalcJacobian (quad)
pub fn quad_jac_at(x: &[f64], y: &[f64], xi: f64, eta: f64) -> (DMatrix<f64>, f64) {
    let dxi  = [-(1.0 - eta),  (1.0 - eta),  (1.0 + eta), -(1.0 + eta)];
    let deta = [-(1.0 - xi),  -(1.0 + xi),   (1.0 + xi),   (1.0 - xi)];
    let mut j = DMatrix::<f64>::zeros(2, 2);
    for k in 0..4 {
        j[(0, 0)] += dxi[k]  * x[k];
        j[(0, 1)] += deta[k] * x[k];
        j[(1, 0)] += dxi[k]  * y[k];
        j[(1, 1)] += deta[k] * y[k];
    }
    j *= 0.25;
    (j.clone(), j.determinant())
}

/// Per-point Jacobian of the bilinear quad map on the reference square `[0,1]²`
/// with the **topological** (CCW) node order `(0,0),(1,0),(1,1),(0,1)` —
/// the order MFEM's `ElementTransformation` uses for element nodes.  (The L2/// solution basis is lexicographic; geometry is always topological.)
pub fn quad_jac_at_01(x: &[f64], y: &[f64], xi: f64, eta: f64) -> (DMatrix<f64>, f64) {
    // N1=(1-x)(1-y)@(0,0), N2=x(1-y)@(1,0), N3=xy@(1,1), N4=(1-x)y@(0,1)
    let dxi  = [-(1.0 - eta),  (1.0 - eta),  eta, -eta];
    let deta = [-(1.0 - xi),  -xi,   xi,  (1.0 - xi)];
    let mut j = DMatrix::<f64>::zeros(2, 2);
    for k in 0..4 {
        j[(0, 0)] += dxi[k]  * x[k];
        j[(0, 1)] += deta[k] * x[k];
        j[(1, 0)] += dxi[k]  * y[k];
        j[(1, 1)] += deta[k] * y[k];
    }
    (j.clone(), j.determinant())
}

/// Exact inverse of the bilinear quad map on `[0,1]²` via Newton iteration
/// (the affine `phys_to_ref` is only an approximation for bilinear maps; MFEM
/// uses `ElementTransformation::TransformBack` which converges to machine
/// precision).  Node order matches [`quad_jac_at_01`].
pub fn phys_to_ref_quad_01(
    x: &[f64],
    y: &[f64],
    xp: &[f64],
    xi0: &[f64],
) -> Vec<f64> {
    let mut xi = vec![xi0[0], xi0[1]];
    for _ in 0..12 {
        // Bilinear map value at (xi, eta): X = Σ N_k(ξ,η) x_k, topological
        // node order N1=(1-x)(1-y)@0, N2=x(1-y)@1, N3=xy@2, N4=(1-x)y@3
        // (matching quad_jac_at_01).
        let (nx0, nx1) = (1.0 - xi[0], xi[0]);
        let (ny0, ny1) = (1.0 - xi[1], xi[1]);
        let xv = nx0 * ny0 * x[0] + nx1 * ny0 * x[1] + nx1 * ny1 * x[2] + nx0 * ny1 * x[3];
        let yv = nx0 * ny0 * y[0] + nx1 * ny0 * y[1] + nx1 * ny1 * y[2] + nx0 * ny1 * y[3];
        let (j, _d) = quad_jac_at_01(x, y, xi[0], xi[1]);
        let det = j[(0, 0)] * j[(1, 1)] - j[(0, 1)] * j[(1, 0)];
        // D696 batch 4: abs RETAINED — Newton-iteration degeneracy guard
        // (magnitude test, not a measure); the inverse below uses signed det.
        if det.abs() < 1e-16 {
            break;
        }
        let fx = xp[0] - xv;
        let fy = xp[1] - yv;
        if fx * fx + fy * fy < 1e-26 {
            break;
        }
        let dxi = (j[(1, 1)] * fx - j[(0, 1)] * fy) / det;
        let deta = (-j[(1, 0)] * fx + j[(0, 0)] * fy) / det;
        xi[0] += dxi;
        xi[1] += deta;
    }
    xi
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2-D face geometry by *reference composition* (MFEM FaceElementTransformations)
// ═══════════════════════════════════════════════════════════════════════════════

/// Everything a 2-D DG face term needs at one face quadrature point.
pub struct FacePointGeom {
    /// Reference point inside the element: MFEM `GetElement1IntPoint()`.
    pub eip: [f64; 2],
    /// MFEM `nor = CalcOrtho(Trans.Jacobian())` — outward from the element,
    /// magnitude = the edge arc-length derivative `|dX/dξ|`.
    pub nor: [f64; 2],
    /// MFEM `Trans.Elem1->Weight()` = `det(J)` of the element transformation
    /// at `eip` (signed).
    pub det_j: f64,
    /// `J^{-T}` of the element transformation at `eip`.
    pub jit: DMatrix<f64>,
    /// `Trans.Elem1->Transform(eip)`: the physical face point through the
    /// element's own isoparametric map — the point MFEM's coefficients
    /// (`u->Eval(vu, *Trans.Elem1, eip1)`) are evaluated at.  On a straight
    /// element it is bit-equal to the affine interpolation of the face's two
    /// corner nodes (D799-3: the chord parameterisation the pre-fix code
    /// built by hand).
    pub xp: [f64; 2],
}

/// Find the local edge of a 2-D element whose endpoint nodes are `{a, b}`.
///
/// Returns `(le, forward)` where the element's own local edge `le` runs from
/// its corner `le` to corner `le + 1 (mod nv)` and `forward` is `true` when
/// that direction runs from `a` to `b` (MFEM `FaceInfo::Elem1Inf`'s
/// orientation bit, decoded at the reference level).
// MFEM: Mesh::GetLocalFaceTransformation
pub fn find_local_edge<M: MeshTopology + ?Sized>(mesh: &M, elem: u32, a: u32, b: u32) -> (usize, bool) {
    let en = mesh.element_nodes(elem);
    let nv = en.len();
    for le in 0..nv {
        let (p, q) = (en[le], en[(le + 1) % nv]);
        if p == a && q == b {
            return (le, true);
        }
        if p == b && q == a {
            return (le, false);
        }
    }
    panic!("find_local_edge: element {elem} has no edge ({a}, {b})");
}

/// Reference-edge parameterisation of a 2-D element: the point and its tangent
/// at `s ∈ [0,1]` along local edge `le`, which runs from the element's corner
/// `le` to corner `le + 1 (mod nv)`.
///
/// Reference domains: `Quad4` on `[0,1]²` with CCW corners
/// `(0,0),(1,0),(1,1),(0,1)` and `Tri3` on `[0,1]²` with vertices
/// `(0,0),(1,0),(0,1)` — the same domains MFEM's `Geometry::SQUARE` /
/// `Geometry::TRIANGLE` and this crate's `Quad4`/`Tri3` reference elements use.
// MFEM: Mesh::GetLocalFaceTransformation
pub fn ref_edge_map(et: ElementType, le: usize, s: f64) -> ([f64; 2], [f64; 2]) {
    match (et, le) {
        (ElementType::Quad4, 0) => ([s, 0.0], [1.0, 0.0]),
        (ElementType::Quad4, 1) => ([1.0, s], [0.0, 1.0]),
        (ElementType::Quad4, 2) => ([1.0 - s, 1.0], [-1.0, 0.0]),
        (ElementType::Quad4, 3) => ([0.0, 1.0 - s], [0.0, -1.0]),
        (ElementType::Tri3, 0) => ([s, 0.0], [1.0, 0.0]),
        (ElementType::Tri3, 1) => ([1.0 - s, s], [-1.0, 1.0]),
        (ElementType::Tri3, 2) => ([0.0, 1.0 - s], [0.0, -1.0]),
        _ => panic!("ref_edge_map: unsupported ({et:?}, le={le})"),
    }
}

/// MFEM `FaceElementTransformations` geometry at the reference-face
/// coordinate `ξ ∈ [0,1]` of the 2-D edge whose endpoint nodes are `{a, b}`.
///
/// The face is parameterised from node `a` (`ξ = 0`) to node `b` (`ξ = 1`) and
/// the element reference point is obtained by **direct reference composition**
/// (`FaceElementTransformations::Loc1`), never by inverting the physical map —
/// so a curved element is sampled at the exact point of its own isoparametric
/// map, as MFEM's `Trans.SetAllIntPoints(&ip)` does.
///
/// The composed face Jacobian is
/// `J_face = J_elem(eip) · d(eip)/dξ` (chain rule, `Trans.Jacobian()`), and
/// MFEM's `CalcOrtho` in 2-D returns `(J_face(1,0), -J_face(0,0))`.  With the
/// element's own CCW edge direction (`find_local_edge`'s `forward` flag) that
/// is the *outward* normal scaled by `|dX/dξ|` — the exact analogue of MFEM's
/// `Elem1Inf` orientation bits.  On a straight mesh this reproduces the chord
/// (`|nor|` = edge length, the `[0,1]` reference segment); on a curved mesh it
/// follows the isoparametric edge (MFEM's `Mesh::GetFaceTransformation`
/// curved branch composes the face transformation through `Elem1`).
// MFEM: FaceElementTransformations::Jacobian + CalcOrtho + Elem1->Weight()
pub fn face_point_geom<M: MeshTopology + ?Sized>(
    mesh: &M,
    elem: u32,
    a: u32,
    b: u32,
    xi: f64,
) -> FacePointGeom {
    let et = mesh.element_type(elem);
    let (le, forward) = find_local_edge(mesh, elem, a, b);
    // ξ runs a → b (the face's own node order, MFEM `Loc1`'s parameterisation);
    // the element's own edge runs corner le → corner le+1.
    let s = if forward { xi } else { 1.0 - xi };
    let (eip, ds) = ref_edge_map(et, le, s);
    let (jac, xp) = element_jacobian_at(mesh, elem, &eip, 2);
    // J_face = J_elem · d(eip)/ds, with `ds` the element's **own** CCW edge
    // tangent — `ref_edge_map`'s derivative is constant in `s`, so it is the
    // same on either traversal direction.  Using the element's own direction
    // (not the a → b one) is what makes `nor` MFEM's, i.e. pointing *out of*
    // this element: MFEM fixes the sign through `FaceInfo`'s orientation bits,
    // independently of how `Loc1` parameterises the face.
    let tf = [
        jac[(0, 0)] * ds[0] + jac[(0, 1)] * ds[1],
        jac[(1, 0)] * ds[0] + jac[(1, 1)] * ds[1],
    ];
    let nor = [tf[1], -tf[0]];
    let det_j = jac.determinant();
    // Degeneracy guard (magnitude test only, D696 batch 4 precedent): a
    // singular element Jacobian falls back to the identity so that assembly
    // stays finite instead of panicking.  Never taken on a valid mesh.
    let jit = jac
        .try_inverse()
        .unwrap_or_else(|| {
            eprintln!("  warning: degenerate element {elem} in DG face assembly");
            DMatrix::identity(2, 2)
        })
        .transpose();
    FacePointGeom { eip, nor, det_j, jit, xp: [xp[0], xp[1]] }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3-D face geometry by *reference composition*
// ═══════════════════════════════════════════════════════════════════════════════

/// Everything a 3-D DG face term needs at one face quadrature point.
pub struct FacePointGeom3 {
    /// Reference point inside the element: MFEM `GetElement1IntPoint()`.
    pub eip: [f64; 3],
    /// MFEM `nor = CalcOrtho(Trans.Jacobian())` = the cross product of the two
    /// columns of the composed face Jacobian `J_elem(eip)·d(eip)/dξ`; `|nor|`
    /// is the face area element (`dA = |nor|·dξ₁dξ₂` for the `[0,1]²` face
    /// rule, whose weights sum to 1/2 on a triangle).
    pub nor: [f64; 3],
    /// MFEM `Trans.Elem1->Weight()` = `det(J)` of the element transformation
    /// at `eip` (signed).
    pub det_j: f64,
    /// `J^{-T}` of the element transformation at `eip`.
    pub jit: DMatrix<f64>,
    /// `Trans.Elem1->Transform(eip)`.
    pub xp: [f64; 3],
}

/// Owned 2-D or 3-D face-geometry result ([`FacePointGeom`] / [`FacePointGeom3`]),
/// so a dim-generic DG face loop can consume either without duplicating its
/// per-quadrature-point arithmetic (D814-1; the periodic-seam driver in
/// `dg_advection.rs` shares it).
pub(crate) enum FaceGeom {
    /// 2-D edge (`face_point_geom`).
    Face2(FacePointGeom),
    /// 3-D triangular or quadrilateral face (`face_point_geom_3d_face`).
    Face3(FacePointGeom3),
}

impl FaceGeom {
    /// `GetElement1IntPoint()` — the composed reference point in the element.
    pub fn eip(&self) -> &[f64] {
        match self {
            FaceGeom::Face2(g) => &g.eip,
            FaceGeom::Face3(g) => &g.eip,
        }
    }
    /// `Elem1->Transform(eip)`.
    pub fn xp(&self) -> &[f64] {
        match self {
            FaceGeom::Face2(g) => &g.xp,
            FaceGeom::Face3(g) => &g.xp,
        }
    }
    /// `CalcOrtho(Trans.Jacobian())` — unnormalised, outward from `Elem1`.
    pub fn nor(&self) -> &[f64] {
        match self {
            FaceGeom::Face2(g) => &g.nor,
            FaceGeom::Face3(g) => &g.nor,
        }
    }
    /// `Elem1->Weight()` (signed det J at `eip`).
    pub fn det_j(&self) -> f64 {
        match self {
            FaceGeom::Face2(g) => g.det_j,
            FaceGeom::Face3(g) => g.det_j,
        }
    }
    /// `J^{-T}` of the element map at `eip`.
    pub fn jit(&self) -> &DMatrix<f64> {
        match self {
            FaceGeom::Face2(g) => &g.jit,
            FaceGeom::Face3(g) => &g.jit,
        }
    }
}

/// Reference-tetrahedron vertex coordinates, MFEM `Geometry::TETRAHEDRON`
/// reference element (`[0,1]³` simplex).
const TET_REF_V: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

/// Reference-cube corners of a hexahedron in the crate's `Hex8` vertex order
/// (`fem_element::lagrange::factory::HEX_VERT_SIDES`, MFEM
/// `Geometry::Constants<Geometry::CUBE>::Vertices`): local vertices 0..3 are
/// the bottom ring (`z = 0`) counter-clockwise, 4..7 the top ring.
const HEX_REF_V: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];

/// D799-3 3-D counterpart of [`face_point_geom`]: MFEM
/// `FaceElementTransformations` geometry at the reference-face coordinate
/// `ξ ∈ [0,1]²` of the **triangular** face whose nodes are `(a, b, c)`.
///
/// The face is parameterised from node `a`: `ξ₁` runs `a → b` and `ξ₂` runs
/// `a → c` (the reference triangle `(0,0),(1,0),(0,1)` of MFEM's
/// `Geometry::TRIANGLE`), and the element reference point is obtained by the
/// **reference composition** `Loc1` — never by inverting the physical map.  The
/// composed face Jacobian is `J_face = J_elem(eip)·[d(eip)/dξ₁ | d(eip)/dξ₂]`
/// and MFEM's `CalcOrtho` in 3-D returns the **cross product of its two
/// columns** (`densemat.cpp:2760`, `n = c₀ × c₁`).
///
/// Orientation: MFEM's `FaceInfo` orientation bits order the face's vertices so
/// that `c₀ × c₁` points *out of* `Elem1`.  The same is achieved here at the
/// reference level: the sign of `nor` is flipped whenever `c₀ × c₁` would point
/// *into* the element (tested against the element's fourth vertex in the
/// reference tetrahedron, which is exact and mesh-independent), while the
/// `ξ ↦ eip` parameterisation stays the face's own `(a, b, c)` order — the same
/// point both neighbours are composed at, exactly as `Loc1`/`Loc2` receive the
/// same `ip`.  On a straight tetrahedron this reproduces the affine face
/// geometry bit for bit; on a curved one it follows the isoparametric map.
///
/// Only tetrahedral (triangular-face) elements are supported: the 3-D DG paths
/// take their face measure from the `Tri3` reference element and the face list
/// only covers tetrahedra (`build_face_elem_map`'s `(4,3)` arm).  For the
/// **quadrilateral** face of a hexahedron use [`face_point_geom_3d_quad`]
/// (D805-4).
// MFEM: FaceElementTransformations::Jacobian + CalcOrtho + Elem1->Weight()
pub fn face_point_geom_3d<M: MeshTopology + ?Sized>(
    mesh: &M,
    elem: u32,
    a: u32,
    b: u32,
    c: u32,
    xi: [f64; 2],
) -> FacePointGeom3 {
    let et = mesh.element_type(elem);
    if et != ElementType::Tet4 {
        panic!("face_point_geom_3d: unsupported element type {et:?} (triangular faces of Tet4 only)");
    }
    let en = mesh.element_nodes(elem);
    if en.len() != 4 {
        panic!("face_point_geom_3d: element {elem} has {} nodes, expected 4 (Tet4)", en.len());
    }
    let idx_of = |n: u32| -> usize {
        en.iter().position(|&m| m == n).unwrap_or_else(|| {
            panic!("face_point_geom_3d: element {elem} has no node {n}")
        })
    };
    let (ia, ib, ic) = (idx_of(a), idx_of(b), idx_of(c));
    let (ra, rb, rc) = (TET_REF_V[ia], TET_REF_V[ib], TET_REF_V[ic]);
    // The element's fourth reference vertex: the one index not in {a, b, c}.
    let id = (0..4).find(|&k| k != ia && k != ib && k != ic).unwrap();
    let rd = TET_REF_V[id];
    // Reference-face normal c₀ × c₁ with the face's own (a, b, c) order; the
    // sign is corrected below when it points into the element.
    let t1 = [rb[0] - ra[0], rb[1] - ra[1], rb[2] - ra[2]];
    let t2 = [rc[0] - ra[0], rc[1] - ra[1], rc[2] - ra[2]];
    let nref = [
        t1[1] * t2[2] - t1[2] * t2[1],
        t1[2] * t2[0] - t1[0] * t2[2],
        t1[0] * t2[1] - t1[1] * t2[0],
    ];
    let to_4th = [rd[0] - ra[0], rd[1] - ra[1], rd[2] - ra[2]];
    let outward = nref[0] * to_4th[0] + nref[1] * to_4th[1] + nref[2] * to_4th[2] < 0.0;
    let (x1, x2) = (xi[0], xi[1]);
    let eip = [
        ra[0] + x1 * (rb[0] - ra[0]) + x2 * (rc[0] - ra[0]),
        ra[1] + x1 * (rb[1] - ra[1]) + x2 * (rc[1] - ra[1]),
        ra[2] + x1 * (rb[2] - ra[2]) + x2 * (rc[2] - ra[2]),
    ];
    let (jac, xp) = element_jacobian_at(mesh, elem, &eip, 3);
    // Composed face Jacobian columns: J_elem · d(eip)/dξ_k, in the face's own
    // (a, b, c) parameterisation — the same ξ both neighbours are composed at.
    let mut col = [[0.0_f64; 3]; 2];
    for k in 0..2 {
        let tan = if k == 0 {
            [rb[0] - ra[0], rb[1] - ra[1], rb[2] - ra[2]]
        } else {
            [rc[0] - ra[0], rc[1] - ra[1], rc[2] - ra[2]]
        };
        for i in 0..3 {
            col[k][i] = jac[(i, 0)] * tan[0] + jac[(i, 1)] * tan[1] + jac[(i, 2)] * tan[2];
        }
    }
    // CalcOrtho (3-D): the cross product of the two Jacobian columns, with its
    // sign chosen so that `nor` points *out of* this element — MFEM's `FaceInfo`
    // orientation bits do the same at the reference level, independently of the
    // face's parameterisation.
    let (c0, c1) = (col[0], col[1]);
    let mut nor = [
        c0[1] * c1[2] - c0[2] * c1[1],
        c0[2] * c1[0] - c0[0] * c1[2],
        c0[0] * c1[1] - c0[1] * c1[0],
    ];
    if !outward {
        nor = [-nor[0], -nor[1], -nor[2]];
    }
    let det_j = jac.determinant();
    // Degeneracy guard (magnitude test only, D696 batch 4 precedent).
    let jit = jac
        .try_inverse()
        .unwrap_or_else(|| {
            eprintln!("  warning: degenerate element {elem} in 3-D DG face assembly");
            DMatrix::identity(3, 3)
        })
        .transpose();
    FacePointGeom3 { eip, nor, det_j, jit, xp: [xp[0], xp[1], xp[2]] }
}

/// D805-4: 3-D counterpart of [`face_point_geom_3d`] for the
/// **quadrilateral** face of a hexahedron, at the reference-face coordinate
/// `ξ ∈ [0,1]²` (MFEM's `Geometry::SQUARE` parameterisation).
///
/// Same contract as the triangular arm — the face is parameterised from node
/// `a`, the element reference point comes from the **reference composition**,
/// and the composed face Jacobian is `J_face = J_elem(eip)·[∂eip/∂ξ₁ | ∂eip/∂ξ₂]`
/// with MFEM's 3-D `CalcOrtho` cross product (`densemat.cpp:2760`) — but the
/// face map is the quad's own **bilinear** one.  With the face cycle
/// `(a, b, c, d)` *positively oriented* as seen from outside the element, the
/// reference square's corners `(0,0), (1,0), (1,1), (0,1)` are `(a, b, c, d)`,
/// so
///
/// ```text
/// eip(ξ₁, ξ₂) = ra + ξ₁(rb − ra) + ξ₂(rd − ra) + ξ₁ξ₂(rc − rd − rb + ra)
/// ∂eip/∂ξ₁    = (rb − ra) + ξ₂(rc − rd − rb + ra)
/// ∂eip/∂ξ₂    = (rd − ra) + ξ₁(rc − rd − rb + ra)
/// ```
///
/// which is the `Geometry::SQUARE` `H1_QuadrilateralElement(1)` map the
/// reference face of a hex uses (`Loc1`/`Loc2` for a hex face).  Note the
/// `(1,1)` corner is the cycle's **third** vertex `c`, not `d`: taking `c` as
/// the `ξ₂` direction is the transpose convention and is wrong on every face
/// whose cycle is not symmetric (measured against MFEM: `eip` and `nor` off by
/// 1.8 relative on `beam-hex`'s `z+` faces before this was fixed).  The
/// bilinear term matters too: on a *warped* hex face the two tangents vary
/// across the face, so both the measure `|c₀ × c₁|` and the physical point
/// differ from the affine (corner-only) interpolation — the defect class
/// D783/D808-4 closed for the element kernels.
///
/// Orientation is resolved at the **reference level**, and it has to be:
/// MFEM's face integration point `ξ` lives in the *canonical* local face's
/// reference square, and a mesh may store the face's four vertices in either
/// winding (`data/beam-hex.mesh` stores its `z−` faces as reversed cycles and
/// its `z+` faces as forward ones).  D814-1: MFEM's `Loc1`/`Loc2`
/// (`GetLocalQuadToHexTransformation`, mesh.cpp:889, with the
/// `quad_t::Orient`-permuted point matrix) interpolate the *canonical* face
/// corners at the face `ξ` on **both** neighbours — so both land on the same
/// physical point — and the only element-dependent quantity is the *sign* of
/// the composed normal.  The same is done here: the parameterisation always
/// uses the stored corner roles (origin, `ξ₁` direction, `(0,1)` corner),
/// and when `(rb−ra) × (rd−ra)` points *into* the element (tested against a
/// reference vertex off the face: exact and mesh-independent, like the
/// tetrahedron's fourth vertex) only `nor`'s **sign** is flipped.  The
/// pre-D814 version swapped the last two corners instead, which changed the
/// parameterisation — silently bending the face map on any ring whose stored
/// normal pointed inward (an interior face seen from its second element, in
/// particular; MFEM-written boundary sections never trigger it, which is why
/// D805-4's boundary-only fixture could not see it).
///
/// Both neighbours must call this with the same `(a, b, c, d)` — the mesh's own
/// face-node order — exactly as the triangular arm requires.
// MFEM: FaceElementTransformations::Jacobian + CalcOrtho + Elem1->Weight()
pub fn face_point_geom_3d_quad<M: MeshTopology + ?Sized>(
    mesh: &M,
    elem: u32,
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    xi: [f64; 2],
) -> FacePointGeom3 {
    let et = mesh.element_type(elem);
    if !matches!(et, ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27) {
        panic!("face_point_geom_3d_quad: unsupported element type {et:?} (quadrilateral faces of hexahedra only)");
    }
    let en = mesh.element_nodes(elem);
    let idx_of = |n: u32| -> usize {
        en.iter()
            .position(|&m| m == n)
            .unwrap_or_else(|| panic!("face_point_geom_3d_quad: element {elem} has no node {n}"))
    };
    let (ia, ib, ic0, id0) = (idx_of(a), idx_of(b), idx_of(c), idx_of(d));
    let (ra, rb, rc0, rd0) = (HEX_REF_V[ia], HEX_REF_V[ib], HEX_REF_V[ic0], HEX_REF_V[id0]);
    // A reference vertex off the face: the outward test's witness (exact and
    // mesh-independent at the reference level, like the tet's fourth vertex).
    let ie = (0..8)
        .find(|&k| k != ia && k != ib && k != ic0 && k != id0)
        .unwrap_or_else(|| {
            panic!("face_point_geom_3d_quad: element {elem} has no vertex off the face")
        });
    let re = HEX_REF_V[ie];
    // The reference-level normal of the composed map, used only to decide the
    // SIGN of `nor` below (never the parameterisation).
    let t1 = [rb[0] - ra[0], rb[1] - ra[1], rb[2] - ra[2]];
    let t2 = [rd0[0] - ra[0], rd0[1] - ra[1], rd0[2] - ra[2]];
    let nref = [
        t1[1] * t2[2] - t1[2] * t2[1],
        t1[2] * t2[0] - t1[0] * t2[2],
        t1[0] * t2[1] - t1[1] * t2[0],
    ];
    let to_off = [re[0] - ra[0], re[1] - ra[1], re[2] - ra[2]];
    let inward = nref[0] * to_off[0] + nref[1] * to_off[1] + nref[2] * to_off[2] > 0.0;
    // D814-1: the composition is ALWAYS the canonical ring's own roles — MFEM
    // builds `Loc1`/`Loc2` (`GetLocalQuadToHexTransformation`, mesh.cpp:889)
    // as the reference bilinear quad evaluated on the point matrix whose
    // column j is the local face vertex carrying canonical corner j
    // (`FaceVert[lf][Orient[o][j]]`), i.e. both neighbours interpolate the
    // canonical corners `(a,b,c,d)` at the same `ξ` and land on the same
    // physical point.  The pre-D814 swap of the last two corners changed the
    // parameterisation instead of the sign — it silently bent the face map
    // whenever the stored ring's normal pointed into the element (never the
    // case on MFEM-written boundary sections, which is why D805-4's
    // boundary-only fixture could not see it).
    let (x1, x2) = (xi[0], xi[1]);
    // Bilinear face map and its two tangents.
    let mut eip = [0.0_f64; 3];
    let mut tan = [[0.0_f64; 3]; 2];
    for i in 0..3 {
        let gx = rb[i] - ra[i];
        let gy = rd0[i] - ra[i];
        let bl = rc0[i] - rd0[i] - rb[i] + ra[i];
        eip[i] = ra[i] + x1 * gx + x2 * gy + x1 * x2 * bl;
        tan[0][i] = gx + x2 * bl;
        tan[1][i] = gy + x1 * bl;
    }
    // Push the reference tangents through the element map at `eip`; `c₀ × c₁`
    // is `CalcOrtho(Trans.Jacobian())` up to the outward sign, applied below.
    let (jac, xp) = element_jacobian_at(mesh, elem, &eip, 3);
    let mut col = [[0.0_f64; 3]; 2];
    for k in 0..2 {
        for i in 0..3 {
            col[k][i] = jac[(i, 0)] * tan[k][0] + jac[(i, 1)] * tan[k][1] + jac[(i, 2)] * tan[k][2];
        }
    }
    let (c0, c1) = (col[0], col[1]);
    let mut nor = [
        c0[1] * c1[2] - c0[2] * c1[1],
        c0[2] * c1[0] - c0[0] * c1[2],
        c0[0] * c1[1] - c0[1] * c1[0],
    ];
    // Sign-only orientation correction: `nor` must point out of the element
    // (MFEM's canonical rings are outward-oriented; a face stored the other
    // way round just flips the sign — the parameterisation is untouched).
    if inward {
        for v in &mut nor {
            *v = -*v;
        }
    }
    let det_j = jac.determinant();
    let jit = jac
        .try_inverse()
        .unwrap_or_else(|| {
            eprintln!("  warning: degenerate element {elem} in 3-D DG quad-face assembly");
            DMatrix::identity(3, 3)
        })
        .transpose();
    FacePointGeom3 { eip, nor, det_j, jit, xp: [xp[0], xp[1], xp[2]] }
}

/// D805-4: a 3-D face's geometry, dispatched on the face's node count — the
/// triangular arm ([`face_point_geom_3d`]) for a tetrahedron's face, the
/// bilinear quadrilateral arm ([`face_point_geom_3d_quad`]) for a
/// hexahedron's.  `face_nodes` must be the mesh's own face-node list
/// (`MeshTopology::face_nodes`), so both neighbours parameterise the face
/// identically.
pub fn face_point_geom_3d_face<M: MeshTopology + ?Sized>(
    mesh: &M,
    elem: u32,
    face_nodes: &[u32],
    xi: [f64; 2],
) -> FacePointGeom3 {
    match face_nodes.len() {
        3 => face_point_geom_3d(mesh, elem, face_nodes[0], face_nodes[1], face_nodes[2], xi),
        4 => face_point_geom_3d_quad(
            mesh,
            elem,
            face_nodes[0],
            face_nodes[1],
            face_nodes[2],
            face_nodes[3],
            xi,
        ),
        n => panic!(
            "face_point_geom_3d_face: unsupported face with {n} nodes \
             (triangles and quadrilaterals only)"
        ),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Gradient transforms// ═══════════════════════════════════════════════════════════════════════════════

/// Transform reference-element gradients to physical gradients:
/// `∇_phys = J^{-T} ∇_ref`.
// MFEM: TransformGrad
pub fn xform_grads(
    jit: &DMatrix<f64>,
    gr: &[f64],
    gp: &mut [f64],
    n: usize,
    dim: usize,
) {
    for i in 0..n {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim {
                s += jit[(j, k)] * gr[i * dim + k];
            }
            gp[i * dim + j] = s;
        }
    }
}

/// Map a physical point back to reference coordinates:
/// `ξ = J^{-1}(x_phys − x_0)`.
// MFEM: PointToRef
pub fn phys_to_ref(
    jac: &DMatrix<f64>,
    x0: &[f64],
    xp: &[f64],
    dim: usize,
) -> Vec<f64> {
    let j_inv = match jac.clone().try_inverse() {
        Some(inv) => inv,
        None => {
            eprintln!("warning: degenerate element in phys_to_ref, using identity");
            DMatrix::identity(dim, dim)
        }
    };
    let dx: Vec<f64> = (0..dim).map(|i| xp[i] - x0[i]).collect();
    let mut xi = vec![0.0_f64; dim];
    for i in 0..dim {
        for k in 0..dim {
            xi[i] += j_inv[(i, k)] * dx[k];
        }
    }
    xi
}

// ═══════════════════════════════════════════════════════════════════════════════
// Mesh helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a map from face ID → owning element (one element per face).
///
/// The strategy: for each volume element, iterate its local faces (vertex
/// tuples), sort each tuple, and register the element in a global map keyed
/// by the sorted tuple.  Then for each `mesh.face` look up its sorted node
/// key to recover the owning element.
///
/// Returns a `HashMap<face_id, elem_id>` for boundary-face iteration.
// MFEM: Mesh::FaceToElementTable
pub fn build_face_elem_map<M: MeshTopology>(
    mesh: &M,
    dim: usize,
) -> HashMap<u32, u32> {
    let local_faces = |npe: usize| -> Vec<Vec<usize>> {
        match (npe, dim) {
            (3, 2) => vec![vec![0, 1], vec![1, 2], vec![0, 2]],
            (4, 2) => vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![0, 3]],
            (4, 3) => vec![
                vec![1, 2, 3],
                vec![0, 2, 3],
                vec![0, 1, 3],
                vec![0, 1, 2],
            ],
            // D805-4: the six quadrilateral faces of a `Hex8`, in the crate's
            // `HEX_VERT_SIDES` order (`z−`, `z+`, `y−`, `y+`, `x−`, `x+` — the
            // same sets `crates/mesh/src/mesh_characteristics.rs::element_faces`
            // and `fem_space::HEX_QUAD_FACES` list).  Only the *set* matters
            // here (the key is sorted below); the parameterisation of a face at
            // assembly time comes from the mesh's own face-node order, which
            // both neighbours share.
            (8, 3) => vec![
                vec![0, 1, 2, 3],
                vec![4, 5, 6, 7],
                vec![0, 1, 5, 4],
                vec![3, 2, 6, 7],
                vec![0, 3, 7, 4],
                vec![1, 2, 6, 5],
            ],
            _ => vec![],
        }
    };

    let mut vol_face_map: HashMap<Vec<u32>, u32> = HashMap::new();
    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let npe = nodes.len();
        for lf in local_faces(npe) {
            let mut key: Vec<u32> = lf.iter().map(|&k| nodes[k]).collect();
            key.sort_unstable();
            vol_face_map.entry(key).or_insert(e);
        }
    }

    let mut result = HashMap::new();
    for f in mesh.face_iter() {
        let fnodes = mesh.face_nodes(f);
        let mut key: Vec<u32> = fnodes.to_vec();
        key.sort_unstable();
        if let Some(&elem) = vol_face_map.get(&key) {
            result.insert(f, elem);
        }
    }
    result
}

/// Find the element that owns a boundary face by scanning all elements.
/// A simpler but O(n_elem) alternative to `build_face_elem_map`.
///
/// Useful when only a single face lookup is needed.
///
/// D814-1: the owner must contain **every** face node.  The old heuristic —
/// "at least 2 of the face's nodes" — was exact only for the 2-node edges of
/// a 2-D mesh; on a hexahedron a quad face's four nodes are shared two-at-a-
/// time by neighbouring elements all over the mesh (the very first element
/// of `data/…`'s ordering already matches two nodes of a face it does not
/// own), so every 3-D boundary term landed on the wrong element.
// MFEM: Mesh::FaceToElement
pub fn find_face_elem<M: MeshTopology>(
    mesh: &M,
    _face_id: u32,
    face_nodes: &[u32],
) -> u32 {
    let mut fkey: Vec<u32> = face_nodes.to_vec();
    fkey.sort_unstable();
    for e in mesh.elem_iter() {
        let enodes = mesh.element_nodes(e);
        if enodes.len() < 3 {
            continue;
        }
        if fkey.iter().all(|&n| enodes.contains(&n)) {
            return e;
        }
    }
    0
}
