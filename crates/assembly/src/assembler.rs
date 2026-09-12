//! Global assembly loop.
//!
//! [`Assembler`] drives the element-by-element assembly of bilinear and linear
//! forms over the mesh.  It is stateless; all data comes from the [`FESpace`]
//! and integrators supplied at call time.

use nalgebra::DMatrix;

use fem_core::types::DofId;
use fem_element::{
    QuadratureRule, ReferenceElement, PrismPk, PyramidPk,
    lagrange::{SegP1, SegP2, TetP1, TetP2, TriP1,
                HexQ1},
    lagrange::factory::{TriPk, TetPk},
    quadrature::quad_rule_01,
};
use fem_element::lagrange::factory::{ref_elem as factory_ref_elem, ElemType as FactoryElemType};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{ElementTransformation, element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;
use fem_space::{L2Basis, SpaceType};

use crate::integrator::{BdQpData, BoundaryBilinearIntegrator, BoundaryLinearIntegrator, BilinearIntegrator, LinearIntegrator, QpData};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use std::sync::OnceLock;

#[cfg(feature = "parallel")]
use std::cell::RefCell;

/// Environment variable for [`assembly_parallel_min_elems`].
/// 
/// If set, overrides adaptive thresholding. Format: positive integer.
#[cfg(feature = "parallel")]
pub const FEM_ASSEMBLY_PARALLEL_MIN_ELEMS: &str = "FEM_ASSEMBLY_PARALLEL_MIN_ELEMS";

#[cfg(feature = "parallel")]
const DEFAULT_PARALLEL_MIN_ELEMS: usize = 64;

#[cfg(feature = "parallel")]
const MIN_PARALLEL_MIN_ELEMS: usize = 8;

#[cfg(feature = "parallel")]
static ASSEMBLY_PARALLEL_MIN_ELEMS: OnceLock<Option<usize>> = OnceLock::new();

/// Compute adaptive assembly parallelization threshold based on thread count.
/// 
/// Returns the minimum number of elements required before Rayon parallelization.
/// The default policy keeps the historical serial threshold for small machines,
/// but scales it down as more worker threads are available so medium meshes can
/// actually enter the parallel path.
/// 
/// Formula: `max(8, 64 >> floor(log2(n_threads)))`
/// 
/// - 1 thread:     threshold = 64
/// - 2-3 threads:  threshold = 32
/// - 4-7 threads:  threshold = 16
/// - 8+ threads:   threshold = 8
/// 
/// Override via environment variable [`FEM_ASSEMBLY_PARALLEL_MIN_ELEMS`] to disable
/// adaptive logic and use a fixed threshold instead.
#[cfg(feature = "parallel")]
fn adaptive_assembly_threshold_for_threads(n_threads: usize) -> usize {
    let threads = n_threads.max(1);
    let log_threads = threads.ilog2() as usize;
    (DEFAULT_PARALLEL_MIN_ELEMS >> log_threads).max(MIN_PARALLEL_MIN_ELEMS)
}

#[cfg(feature = "parallel")]
fn adaptive_assembly_threshold() -> usize {
    adaptive_assembly_threshold_for_threads(rayon::current_num_threads())
}

/// Minimum number of volume elements before using Rayon for domain assembly.
///
/// Supports two modes:
/// 1. **Adaptive (default)**: threshold calibrated by thread count (see [`adaptive_assembly_threshold`])
/// 2. **Fixed override**: set [`FEM_ASSEMBLY_PARALLEL_MIN_ELEMS`] environment variable
///    to a positive integer to use fixed threshold instead of adaptive logic.
/// 
/// Computed once per process (lazy static); subsequent calls are O(1).
#[cfg(feature = "parallel")]
#[inline]
pub fn assembly_parallel_min_elems() -> usize {
    match ASSEMBLY_PARALLEL_MIN_ELEMS.get_or_init(|| {
        std::env::var(FEM_ASSEMBLY_PARALLEL_MIN_ELEMS)
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
    }) {
        Some(threshold) => *threshold,
        None => adaptive_assembly_threshold(),
    }
}

// ─── P0 (constant) reference element ─────────────────────────────────────────

/// Constant (P0) reference element: 1 DOF, basis ≡ 1.0, gradient ≡ 0.
///
/// `dim` selects the reference domain: 2 → `[0,1]²` (tri/quad), 3 → `[-1,1]³`
/// (hex).  The quadrature must live on the same domain as the geometry element
/// used by the isoparametric Jacobian.
struct P0 {
    dim: u8,
}

/// Constant (P0) element on the standard tetrahedron reference domain
/// (volume 1/6): the generic [`P0`] with `dim: 3` uses the hex `[-1,1]³`
/// Gauss rule (weight sum 8), which scales the L2 volume integral by
/// `8/(1/6) = 48` on tets.  `tet_rule` has weight sum 1/6, matching the
/// simplex `ElementTransformation` reference volume.
struct P0Tet;

impl ReferenceElement for P0Tet {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { 0 }
    fn n_dofs(&self) -> usize { 1 }
    fn eval_basis(&self, _xi: &[f64], v: &mut [f64]) { v[0] = 1.0; }
    fn eval_grad_basis(&self, _xi: &[f64], g: &mut [f64]) {
        for x in g.iter_mut() { *x = 0.0; }
    }
    fn quadrature(&self, order: u8) -> QuadratureRule {
        fem_element::quadrature::tet_rule(order)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![vec![0.0; 3]]
    }
}

/// Constant (P0) element on the standard triangle reference domain
/// (area 1/2): the generic [`P0`] with `dim: 2` uses the square `[0,1]²`
/// Gauss rule (weight sum 1), which doubles every standard assembly
/// integral on triangles — the square-rule points fall outside the tri
/// reference domain where the affine map still has |detJ| = 2·Area.
/// `tri_rule` has weight sum 1/2 (MFEM `IntRules.Get(TRIANGLE, order)`
/// semantics), matching the simplex `ElementTransformation` reference area
/// (same defect class as [`P0Tet`]).
struct P0Tri;

impl ReferenceElement for P0Tri {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { 0 }
    fn n_dofs(&self) -> usize { 1 }
    fn eval_basis(&self, _xi: &[f64], v: &mut [f64]) { v[0] = 1.0; }
    fn eval_grad_basis(&self, _xi: &[f64], g: &mut [f64]) {
        for x in g.iter_mut() { *x = 0.0; }
    }
    fn quadrature(&self, order: u8) -> QuadratureRule {
        fem_element::quadrature::tri_rule(order)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![vec![0.0; 2]]
    }
}

impl ReferenceElement for P0 {
    fn dim(&self) -> u8 { self.dim }
    fn order(&self) -> u8 { 0 }
    fn n_dofs(&self) -> usize { 1 }
    fn eval_basis(&self, _xi: &[f64], v: &mut [f64]) { v[0] = 1.0; }
    fn eval_grad_basis(&self, _xi: &[f64], g: &mut [f64]) {
        for x in g.iter_mut() { *x = 0.0; }
    }
    fn quadrature(&self, order: u8) -> QuadratureRule {
        if self.dim == 2 {
            quad_rule_01(order)
        } else {
            fem_element::quadrature::hex_rule(order)
        }
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![vec![0.0; self.dim as usize]]
    }
}

/// Bilinear (Q1) geometry element on `[0,1]²` with MFEM's `BiLinear2DFiniteElement`
/// direct formulas (H1 topological DOF order: v0=(0,0), v1=(1,0), v2=(1,1),
/// v3=(0,1)).  Used as the *geometry* element for non-curved Quad4 meshes —
/// MFEM's mesh without `Nodes` uses a `LinearFECollection` for the element
/// transformation, whose `CalcDShape` is the direct bilinear formula (NOT the
/// barycentric path used by `QuadQk`), and whose node order is the H1 order
/// (NOT the lexicographic `L2_T1` order of curved `Nodes` fields).  Using the
/// wrong path introduces last-ulp differences in the Jacobian and hence in
/// every element matrix of non-axis-aligned quads.
struct BiLinearGeo2D;

impl ReferenceElement for BiLinearGeo2D {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { 1 }
    fn n_dofs(&self) -> usize { 4 }
    fn eval_basis(&self, xi: &[f64], v: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        v[0] = (1.0 - x) * (1.0 - y);
        v[1] = x * (1.0 - y);
        v[2] = x * y;
        v[3] = (1.0 - x) * y;
    }
    fn eval_grad_basis(&self, xi: &[f64], g: &mut [f64]) {
        // MFEM BiLinear2DFiniteElement::CalcDShape:
        //   dshape(0) = (-(1-y), -(1-x))   dshape(1) = ((1-y), -x)
        //   dshape(2) = (y, x)             dshape(3) = (-y, (1-x))
        let (x, y) = (xi[0], xi[1]);
        g[0] = -(1.0 - y); g[1] = -(1.0 - x);
        g[2] = 1.0 - y;    g[3] = -x;
        g[4] = y;          g[5] = x;
        g[6] = -y;         g[7] = 1.0 - x;
    }
    fn quadrature(&self, order: u8) -> QuadratureRule { quad_rule_01(order) }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![1.0, 1.0], vec![0.0, 1.0]]
    }
}

// ─── Reference element factory ───────────────────────────────────────────────

/// Return the solution reference element matching `elem_type` and polynomial
/// `order` for an **L2/DG** space: tensor elements (Quad/Hex) use MFEM's
/// lexicographic `L2_DOF_MAP` order with the `L2_FECollection` default
/// Gauss-Legendre nodes, all other element types keep the H1 topological
/// ordering (which MFEM's L2 spaces on simplices also use).
pub fn ref_elem_vol_l2(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match elem_type {
        ElementType::Quad4 => match order {
            0 => Box::new(P0 { dim: 2 }),
            // MFEM L2_FECollection uses Gauss-Legendre tensor-product basis
            // (BasisType::GaussLegendre), NOT the GLL basis of H1.  QuadL2GL
            // reproduces it bit-identically on [0,1]² with lexicographic DOFs.
            o => Box::new(fem_element::lagrange::QuadL2GL::new(o as usize)),
        },
        ElementType::Hex8 => match order {
            0 => Box::new(P0 { dim: 3 }),
            // MFEM L2_HexahedronElement(o, GaussLegendre): interior GL tensor
            // nodes with lexicographic DOFs.  HexL2GL keeps the fem-rs hex
            // reference domain [-1,1]³ (same as HexQk/hex_rule) so quadrature
            // and the isoparametric Jacobian stay on a common domain.
            o => Box::new(fem_element::lagrange::HexL2GL::new(o as usize)),
        },
        _ => ref_elem_vol(elem_type, order),
    }
}

/// Reference element for `space`: L2/DG spaces get the lexicographic
/// tensor-product DOF ordering ([`ref_elem_vol_l2`]), H1 spaces the
/// topological one ([`ref_elem_vol_h1`]).
pub(crate) fn ref_elem_vol_for_space<S: FESpace>(
    space: &S,
    elem_type: ElementType,
    order: u8,
) -> Box<dyn ReferenceElement> {
    if space.space_type() == SpaceType::L2 {
        // MFEM `DG_FECollection`/`L2_FECollection(btype)` with GaussLobatto
        // uses GLL nodes with lexicographic DOFs (`L2_DOF_MAP`) on tensor
        // elements — `QuadQk::new_lex`/`HexQk::new_lex`, NOT the GL-noded
        // `QuadL2GL`/`HexL2GL` (which match only `L2_FECollection`'s default
        // `GaussLegendre`).  Using the wrong basis silently changes every
        // element matrix (ex41 regression: M/S/K off by ~6×, IMEX diverged).
        let gll_lex = space.l2_basis() == Some(L2Basis::GaussLobatto)
            && matches!(elem_type, ElementType::Quad4 | ElementType::Hex8);
        if gll_lex && order >= 1 {
            return match elem_type {
                ElementType::Quad4 => Box::new(
                    fem_element::lagrange::factory::QuadQk::new_lex(order as usize),
                ),
                _ => Box::new(fem_element::lagrange::factory::HexQk::new_lex(order as usize)),
            };
        }
        ref_elem_vol_l2(elem_type, order)
    } else {
        ref_elem_vol_h1(elem_type, order)
    }
}

/// H1 solution reference element: MFEM `H1_FECollection` semantics
/// (`BasisType::GaussLobatto`).
///
/// Simplex elements of order ≥ 3 use [`H1TriPk`] (Gauss-Lobatto nodes) —
/// the fixed-order `TriPk`/`TriP3`/`TriP4` are *equispaced*, which matches
/// MFEM only at p ≤ 2 (the p=2 edge midpoints coincide with the GLL points).
/// Note this differs from the DG/L2 paths, which keep the equispaced
/// [`TriPk`] (see [`ref_elem_vol_l2`]).
pub(crate) fn ref_elem_vol_h1(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (elem_type, order) {
        (ElementType::Tri3 | ElementType::Tri6, 0) => Box::new(P0Tri),
        (ElementType::Tri3 | ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3 | ElementType::Tri6, 2) => Box::new(TriPk::new(2)),
        (ElementType::Tri3 | ElementType::Tri6, 3) => {
            Box::new(fem_element::lagrange::H1TriPk::new(3))
        }
        (ElementType::Tri3 | ElementType::Tri6, 4) => {
            Box::new(fem_element::lagrange::H1TriPk::new(4))
        }
        (ElementType::Tri3 | ElementType::Tri6, o) => {
            Box::new(fem_element::lagrange::H1TriPk::new(o as usize))
        }
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetPk::new(3)),
        (ElementType::Tet4, o) => Box::new(fem_element::lagrange::TetPk::new(o as usize)),
        (ElementType::Quad4, 0) => Box::new(P0 { dim: 2 }),
        // order 1..=2: QuadQk (Gauss-Lobatto nodes on [0,1]^2) — matches MFEM
        // H1_FECollection's default BasisType::GaussLobatto.  QuadQ1/Q2 were
        // historically on [-1,1]^2; affine-embedding equivalent for the
        // gradient (Diffusion) but NOT for the mass ∫φ² (4× off on [0,1]²),
        // so the reference domain must be [0,1]^2 for all orders.
        (ElementType::Quad4, 1) => Box::new(fem_element::lagrange::QuadQk::new(1)),
        (ElementType::Quad4, 2) => Box::new(fem_element::lagrange::QuadQk::new(2)),
        // order >= 3: Gauss-Lobatto-Legendre nodes on [0,1]^2 (matches MFEM
        // H1_FECollection's default BasisType::GaussLobatto); QuadQ3 is
        // equidistant on [-1,1]^2 and therefore NOT MFEM-compatible at p=3.
        (ElementType::Quad4, 3) => Box::new(fem_element::lagrange::QuadQk::new(3)),
        (ElementType::Quad4, o) => Box::new(fem_element::lagrange::QuadQk::new(o as usize)),
        (ElementType::Hex8, 0) => Box::new(P0 { dim: 3 }), // L2 P0 (constant) on hexes
        (ElementType::Hex8, 1) => Box::new(HexQ1),
        (ElementType::Hex8, o) => Box::new(fem_element::lagrange::HexQk::new(o as usize)),
        (ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18, _) => {
            Box::new(PrismPk::new(order as usize))
        }
        (ElementType::Pyramid5 | ElementType::Pyramid13, _) => {
            Box::new(PyramidPk::new(order as usize))
        }
        _ => panic!(
            "ref_elem_vol_h1: unsupported combination (element_type={elem_type:?}, order={order}). \
             Try using a different polynomial order or a simplex mesh."
        ),
    }
}

/// Return the solution reference element matching `elem_type` and polynomial `order`.
pub(crate) fn ref_elem_vol(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (elem_type, order) {
        (ElementType::Tri3 | ElementType::Tri6, 0) => Box::new(P0Tri),
        (ElementType::Tri3 | ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3 | ElementType::Tri6, 2) => Box::new(TriPk::new(2)),
        (ElementType::Tri3 | ElementType::Tri6, 3) => Box::new(TriPk::new(3)),
        (ElementType::Tri3 | ElementType::Tri6, 4) => Box::new(TriPk::new(4)),
        (ElementType::Tri3 | ElementType::Tri6, o) => Box::new(TriPk::new(o as usize)),
        (ElementType::Tet4, 0)                           => Box::new(P0Tet), // L2 P0 (constant) on tets
        (ElementType::Tet4, 1)                           => Box::new(TetP1),
        (ElementType::Tet4, 2)                           => Box::new(TetP2),
        (ElementType::Tet4, 3)                           => Box::new(TetPk::new(3)),
        (ElementType::Tet4, o)                           => Box::new(fem_element::lagrange::TetPk::new(o as usize)),
        (ElementType::Quad4, 0)                          => Box::new(P0 { dim: 2 }),
        // order 1..=2: QuadQk (Gauss-Lobatto nodes on [0,1]^2) — matches MFEM
        // H1_FECollection's default BasisType::GaussLobatto.  QuadQ1/Q2 were
        // historically on [-1,1]^2; affine-embedding equivalent for the
        // gradient (Diffusion) but NOT for the mass ∫φ² (4× off on [0,1]²),
        // so the reference domain must be [0,1]^2 for all orders.
        (ElementType::Quad4, 1)                          => Box::new(fem_element::lagrange::QuadQk::new(1)),
        (ElementType::Quad4, 2)                          => Box::new(fem_element::lagrange::QuadQk::new(2)),
        // order >= 3: Gauss-Lobatto-Legendre nodes on [0,1]^2 (matches MFEM
        // H1_FECollection's default BasisType::GaussLobatto); QuadQ3 is
        // equidistant on [-1,1]^2 and therefore NOT MFEM-compatible at p=3.
        (ElementType::Quad4, 3)                          => Box::new(fem_element::lagrange::QuadQk::new(3)),
        (ElementType::Quad4, o)                          => Box::new(fem_element::lagrange::QuadQk::new(o as usize)),
        (ElementType::Hex8, 0)                           => Box::new(P0 { dim: 3 }), // L2 P0 (constant) on hexes
        (ElementType::Hex8, 1)                           => Box::new(HexQ1),
        (ElementType::Hex8, o)                           => Box::new(fem_element::lagrange::HexQk::new(o as usize)),
        (ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18, _) => Box::new(PrismPk::new(order as usize)),
        (ElementType::Pyramid5 | ElementType::Pyramid13, _) => Box::new(PyramidPk::new(order as usize)),
        _ => panic!(
            "ref_elem_vol: unsupported combination (element_type={elem_type:?}, order={order}). \
             Try using a different polynomial order or a simplex mesh."
        ),
    }
}

/// Map a quadrature point from the solution basis domain to the geometry
/// element's reference domain.
///
/// For `Quad4` the *geometry* element is always `QuadQk` on `[0,1]^d`
/// (`geo_ref_elem`), while the *solution* basis lives on `[-1,1]^d` for the
/// fixed-order elements `QuadQ1/Q2/Q3` (order 1..=3) and on `[0,1]^d` for
/// `QuadQk` (order ≥ 4).  The quadrature points come from the solution
/// element, so:
/// - order 1..=3: map `[-1,1] → [0,1]` with `(x+1)/2`;
/// - order ≥ 4: already on `[0,1]`, identity.
///
/// Before this mapping was fixed, `[-1,1]` quadrature points were passed
/// straight to the `[0,1]` geometry element, whose `to_std` chain-rule map
/// pushed them outside the reference domain (`[-3,-1]`): the geometry
/// polynomial was extrapolated there, corrupting the Jacobian (and hence the
/// stiffness) on strongly curved cells (e.g. the ex27 hole regions).
#[inline]
pub(crate) fn geom_quad_point(_elem_type: ElementType, _order: u8, xi: &[f64]) -> Vec<f64> {
    // All Quad4 solution bases now live on [0,1]^d (QuadQk, order >= 1), and
    // simplex bases share their reference domain with the geometry element,
    // so quadrature points always arrive in the geometry's reference domain.
    xi.to_vec()
}

/// Return the solution reference element for a boundary face.
///
/// The face reference element fixes both the quadrature rule of
/// [`Assembler::assemble_boundary_linear`] / `assemble_boundary_bilinear` and
/// the *pairing* between a face basis function `φᵢ` and the `i`-th entry of
/// the caller's `face_dofs` list.  The helper that builds such lists for H¹
/// spaces is [`face_dofs_h1`] (every order and every face type; [`face_dofs_p1`]
/// and [`face_dofs_p2`] are the older vertices-only / quadratic 2-D variants).
///
/// The element is always the one MFEM's `FiniteElementSpace::GetBE` returns,
/// i.e. `fec->GetFE(mesh->GetBdrElementGeometry(be), order)` — the *trace*
/// element of the H¹ family on the face geometry:
///
/// * a segment face (2-D mesh) → `H1_SegmentElement(p)`, which is the trace of
///   the volume element on its edge ([`H1SegPk`]);
/// * a triangle face (tet mesh) → `H1_TriangleElement(p)`
///   ([`H1TetFacePk`]): MFEM's 2-D triangular DOF *order* over the **volume**
///   element's face **nodes**, so the face basis functions are the
///   restrictions of the space's own basis functions;
/// * a quad face (hex mesh) → `H1_QuadrilateralElement(p)` on `[0,1]²`, which
///   is exactly [`fem_element::lagrange::factory::QuadQk`] (`QuadQ1`/`QuadQ2`
///   live on `[-1,1]²` and are therefore not the boundary element).
///
/// The face element's DOF blocks are the 2-D topological ones (vertices →
/// edges → interior, with the face's own edge directions) because that is what
/// MFEM's `GetBdrElementDofs` returns and what [`face_dofs_h1`] pairs with.
fn ref_elem_face(face_elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (face_elem_type, order) {
        // ── Historical fixed-order entries (bit-identical for old callers) ──
        // `SegP1` and `SegP2` have exactly the trace nodes ([0,1] and the
        // closed Gauss-Lobatto points of p = 2, which are equidistant).
        (ElementType::Line2, 1) => Box::new(SegP1),
        (ElementType::Line2, 2) => Box::new(SegP2),
        // ── Order-generic entries ───────────────────────────────────────────
        (ElementType::Line2, o) if o >= 1 => Box::new(H1SegPk::new(o)),
        (ElementType::Tri3, o) if o >= 1 => Box::new(H1TetFacePk::new(o as usize)),
        (ElementType::Quad4, o) if o >= 1 => {
            Box::new(fem_element::lagrange::factory::QuadQk::new(o as usize))
        }
        _ => panic!(
            "ref_elem_face: unsupported (element_type={face_elem_type:?}, order={order})"
        ),
    }
}

/// MFEM `H1_TriangleElement(p)` **DOF order** over the reference tetrahedron
/// element's face **nodes** — the boundary element of a tetrahedral mesh.
///
/// MFEM's boundary element for a tetrahedron face is `H1_TriangleElement(p)`,
/// a 2-D element whose DOF order is `[v0, v1, v2, edge0…, edge1…, edge2…,
/// interior…]` (`fem/fe/fe_h1.cpp`), where edge 2 is traversed from `v2`
/// towards `v0`.  Its DOFs are the *same mesh entities* as the volume
/// element's face DOFs, so the two must agree to yield a basis that is the
/// trace of the space's basis — and they do in MFEM because both elements put
/// their nodes at the same parametric positions.
///
/// In fem-rs they do **not** in general: the H¹ tetrahedron basis the
/// assembler uses ([`fem_element::lagrange::factory::TetPk`]) is *equispaced*
/// while `H1_TriangleElement`'s nodes are the closed Gauss-Lobatto points
/// (D49).  At `p = 3` the face edge nodes are at `1/3, 2/3` versus
/// `0.2764, 0.7236`; using the Gauss-Lobatto element here would make the face
/// basis functions *different functions* from the volume basis restricted to
/// the face, so the boundary integral would be distributed onto the wrong DOFs
/// (the D46② failure mode, measured as a per-DOF error of ~0.1 with the sum
/// still exact).  This element therefore takes the `H1_TriangleElement` node
/// *rule* and evaluates it with the closed points `cp[k] = k/p`, i.e. the
/// volume element's own edge distribution, and then resolves every node
/// against the volume element by evaluation (`H1TetFacePk::new`).
///
/// If D49 is ever fixed on the space side (H¹ tetrahedra switched to
/// Gauss-Lobatto nodes) this element must be switched to the Gauss-Lobatto
/// points with it — one line, next to the volume element it mirrors.
struct H1TetFacePk {
    order: usize,
    /// The volume element whose face restriction this element is.
    vol: Box<dyn ReferenceElement>,
    /// Face reference coordinates of the DOFs, in `H1_TriangleElement` order.
    nodes: Vec<[f64; 2]>,
    /// Interior (volume) slot of each DOF: the index of the volume basis
    /// function that is `1` at that node and `0` at all the others.
    slots: Vec<usize>,
    /// Scratch for the volume evaluation (`eval_basis` takes `&self`);
    /// `Mutex` rather than `RefCell` because `ReferenceElement: Send + Sync`.
    scratch: std::sync::Mutex<Vec<f64>>,
}

impl H1TetFacePk {
    fn new(p: usize) -> Self {
        assert!(p >= 1, "H1TetFacePk: order must be >= 1");
        // The volume element whose face restriction this element is.
        let vol: Box<dyn ReferenceElement> =
            Box::new(fem_element::lagrange::factory::TetPk::new(p));
        let pf = p as f64;
        let cp = |k: usize| k as f64 / pf;

        // `H1_TriangleElement`'s node rule (see the type docs) with the
        // volume element's edge distribution.
        let mut nodes: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        for i in 1..p {
            nodes.push([cp(i), 0.0]);
        }
        for i in 1..p {
            nodes.push([cp(p - i), cp(i)]);
        }
        for i in 1..p {
            nodes.push([0.0, cp(p - i)]);
        }
        for j in 1..p {
            for i in 1..(p - j) {
                let w = cp(i) + cp(j) + cp(p - i - j);
                nodes.push([cp(i) / w, cp(j) / w]);
            }
        }

        // Resolve every node against the volume element.  The volume basis is
        // nodal at its own face nodes, so the DOF there is the unique slot
        // whose basis function is 1; the position check below also verifies
        // that the node rule above really is the volume element's.
        let vol_coords = vol.dof_coords();
        let mut vbuf = vec![0.0_f64; vol.n_dofs()];
        let mut slots = Vec::with_capacity(nodes.len());
        for nd in nodes.iter() {
            vol.eval_basis(&[nd[0], nd[1], 0.0], &mut vbuf);
            let (k, &v) = vbuf
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).expect("H1TetFacePk: NaN basis value"))
                .expect("H1TetFacePk: empty volume element");
            assert!(
                v > 0.5,
                "H1TetFacePk(p={p}): no volume basis function is nodal at {nd:?} \
                 (largest value {v}) — is the volume element's node distribution still \
                 the one this element's node rule assumes?"
            );
            let c = &vol_coords[k];
            let d2 = (c[0] - nd[0]) * (c[0] - nd[0])
                + (c[1] - nd[1]) * (c[1] - nd[1])
                + c[2] * c[2];
            assert!(
                d2 < 1e-24,
                "H1TetFacePk(p={p}): volume slot {k} sits at {c:?}, not at the face node \
                 {nd:?} (distance {:.3e})",
                d2.sqrt()
            );
            slots.push(k);
        }
        let n = vol.n_dofs();
        Self {
            order: p,
            vol,
            nodes,
            slots,
            scratch: std::sync::Mutex::new(vec![0.0; n]),
        }
    }

    /// Evaluate the volume basis at `(u, v, 0)` and pick this face element's
    /// DOFs: `out[k] = φ_{slots[k]}` (values), or the row-major `[n × 2]`
    /// gradient `(∂/∂u, ∂/∂v)` of the same functions (the trace is embedded
    /// affinely, so the face's own two partials are those of the volume
    /// coordinates).
    fn trace(&self, u: f64, v: f64, out: &mut [f64], grads: bool) {
        let n = self.nodes.len();
        let mut buf = self.scratch.lock().expect("H1TetFacePk scratch poisoned");
        if grads {
            buf.resize(3 * self.vol.n_dofs(), 0.0);
            self.vol.eval_grad_basis(&[u, v, 0.0], &mut buf);
            for (i, &k) in self.slots.iter().enumerate() {
                out[i * 2] = buf[k * 3];
                out[i * 2 + 1] = buf[k * 3 + 1];
            }
        } else {
            buf.resize(self.vol.n_dofs(), 0.0);
            self.vol.eval_basis(&[u, v, 0.0], &mut buf);
            for (i, &k) in self.slots.iter().enumerate() {
                out[i] = buf[k];
            }
        }
        debug_assert_eq!(out.len(), if grads { 2 * n } else { n });
    }
}

impl ReferenceElement for H1TetFacePk {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        self.nodes.len()
    }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        self.trace(xi[0], xi[1], values, false);
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        self.trace(xi[0], xi[1], grads, true);
    }
    fn quadrature(&self, order: u8) -> QuadratureRule {
        fem_element::quadrature::tri_rule(order)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.nodes.iter().map(|c| vec![c[0], c[1]]).collect()
    }
}

/// MFEM `H1_SegmentElement(p)` — the **trace** of the volume
/// `H1_FECollection` element on one of its edges, on the reference segment
/// `[0, 1]`.
///
/// Two properties matter for a boundary form:
///
/// 1. **DOF order.**  MFEM orders the segment DOFs topologically
///    (`fem/fe/fe_h1.cpp`: `Nodes.IntPoint(0).x = cp[0]`,
///    `Nodes.IntPoint(1).x = cp[p]`, then the interior nodes in increasing
///    order), i.e. `[v0, v1, interior ascending]`.  That differs from the
///    ascending-`ξ` order of `lagrange::factory::SegPk` and of
///    `SegP4`/`SegP5`/`SegP6` at `p ≥ 2`.
/// 2. **DOF positions.**  They are the *edge nodes of the volume element* —
///    the closed Gauss-Lobatto points (`Lagrange1D`, MFEM `ClosedPoints(p)`),
///    which are **not** equispaced for `p ≥ 3`.  An equispaced face basis
///    still satisfies `Σᵢ φᵢ = 1`, so a divergence-theorem check on the DOF
///    *sum* passes while the per-DOF distribution is wrong: measured on the
///    `navier_kovasznay` mesh (order 6) the per-DOF `g_bdr` was off by up to
///    0.1 against the exact trace, with the flux sum exact.
///
/// Both are obtained here by *restricting the volume element*: evaluating the
/// order-`p` tensor-product H¹ element (`QuadQk`, whose bottom-edge DOFs 0, 1
/// and `4..4+p-1` are the vertices and the interior nodes in topological
/// order — see that element's `pos_dof_map`) at `(ξ, 0)` and keeping those
/// DOFs.  The trace of any tensor-product H¹ element on an edge is the same
/// 1-D Gauss-Lobatto basis, so this is `H1_SegmentElement(p)` by construction
/// and is consistent with the volume basis used by the assembler — no
/// separate node table to keep in sync.
///
/// Hence the pairing contract of [`ref_elem_face`]: `face_dofs(f)` must list
/// the space DOFs of the face as `[dof(v0), dof(v1), interior along v0 → v1]`,
/// which is what [`face_dofs_h1`] produces.
struct H1SegPk {
    /// Volume H¹ element providing the trace (bottom edge only).
    vol: Box<dyn ReferenceElement>,
    /// Local indices of the bottom-edge DOFs of `vol`, in topological order.
    edge: Vec<usize>,
    /// Scratch for the volume evaluation (`eval_basis` takes `&self`);
    /// `Mutex` rather than `RefCell` because `ReferenceElement: Send + Sync`.
    scratch: std::sync::Mutex<Vec<f64>>,
}

impl H1SegPk {
    fn new(p: u8) -> Self {
        assert!(p >= 1, "H1SegPk: order must be >= 1");
        let vol: Box<dyn ReferenceElement> =
            Box::new(fem_element::lagrange::factory::QuadQk::new(p as usize));
        let mut edge = vec![0, 1];
        edge.extend(4..4 + (p as usize - 1));
        let n = vol.n_dofs();
        H1SegPk {
            vol,
            edge,
            scratch: std::sync::Mutex::new(vec![0.0; n]),
        }
    }

    /// Evaluate the volume basis at `ξ = (s, 0)` and pick the edge DOFs.
    fn trace(&self, s: f64, out: &mut [f64], grads: bool) {
        let mut buf = self.scratch.lock().expect("H1SegPk scratch poisoned");
        let n = self.vol.n_dofs();
        buf.resize(if grads { 2 * n } else { n }, 0.0);
        if grads {
            self.vol.eval_grad_basis(&[s, 0.0], &mut buf);
        } else {
            self.vol.eval_basis(&[s, 0.0], &mut buf);
        }
        for (i, &k) in self.edge.iter().enumerate() {
            out[i] = if grads { buf[k * 2] } else { buf[k] };
        }
    }
}

impl ReferenceElement for H1SegPk {
    fn dim(&self) -> u8 {
        1
    }
    fn order(&self) -> u8 {
        self.vol.order()
    }
    fn n_dofs(&self) -> usize {
        self.edge.len()
    }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        self.trace(xi[0], values, false);
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        self.trace(xi[0], grads, true);
    }
    fn quadrature(&self, order: u8) -> QuadratureRule {
        fem_element::quadrature::seg_rule(order)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let vol_coords = self.vol.dof_coords();
        // Bottom-edge DOFs: vertices first (in the face's own order), then the
        // interior nodes ascending along `v0 → v1`.
        let mut c = vec![vec![0.0], vec![1.0]];
        c.extend(self.edge[2..].iter().map(|&k| vec![vol_coords[k][0]]));
        c
    }
}

// ─── Jacobian helpers ─────────────────────────────────────────────────────────

/// Convert a mesh `ElementType` to the factory's `ElemType`.
fn mesh_type_to_factory(et: ElementType) -> FactoryElemType {
    match et {
        ElementType::Tri3 | ElementType::Tri6 => FactoryElemType::Tri,
        ElementType::Tet4 | ElementType::Tet10 => FactoryElemType::Tet,
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => FactoryElemType::Quad,
        ElementType::Hex8 | ElementType::Hex20 => FactoryElemType::Hex,
        ElementType::Line2 | ElementType::Line3 => FactoryElemType::Seg,
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => FactoryElemType::Prism,
        ElementType::Pyramid5 | ElementType::Pyramid13 => FactoryElemType::Pyramid,
        _ => panic!("mesh_type_to_factory: unsupported ElementType {et:?}"),
    }
}

/// Build the geometry reference element for isoparametric Jacobians.
///
/// Returns `Some` for non-affine elements (Quad/Hex with P1, or any element
/// with `geom_order > 1`), `None` for affine P1 simplex elements.
///
/// **Important:** For Quad elements the geometry element is on `[0,1]^d`
/// (`QuadQk`, including order 1): `geom_quad_point` maps the solution-basis
/// quadrature points (`[-1,1]^d` for orders 1..=3) onto `[0,1]^d` before
/// evaluating the Jacobian, so the geometry basis domain and the evaluation
/// points must agree.  Using QuadQ1 (on `[-1,1]^d`) here would evaluate the
/// geometry basis at the `[0,1]^d`-mapped points as if they were `[-1,1]^d`
/// coordinates, sampling the Jacobian at shifted points (the ex28 trapezoid
/// stiffness error).  HexQk lives on `[-1,1]^3` (its order-1 form coincides
/// with HexQ1) and `geom_quad_point` leaves hex points unmapped, so the hex
/// arm is unchanged in effect.
pub(crate) fn geo_ref_elem(mesh: &dyn MeshTopology, e: u32) -> Option<Box<dyn ReferenceElement>> {
    let et = mesh.element_type(e);
    let g = mesh.geom_order();
    let is_quad_hex = matches!(et,
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9
        | ElementType::Hex8 | ElementType::Hex20
        | ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18
        | ElementType::Pyramid5 | ElementType::Pyramid13);
    if g == 1 && !is_quad_hex { return None; } // affine P1 simplex
    match et {
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => {
            return if g <= 1 {
                // Non-curved quad geometry: MFEM's mesh without `Nodes`
                // transforms with a LinearFECollection (BiLinear2DFiniteElement
                // direct formulas, H1 topological node order) — NOT the
                // barycentric QuadQk path nor the lexicographic L2_T1 order.
                Some(Box::new(BiLinearGeo2D) as Box<dyn ReferenceElement>)
            } else {
                Some(factory_ref_elem(FactoryElemType::Quad, g))
            };
        }
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => {
            return if g <= 1 {
                Some(Box::new(fem_element::lagrange::factory::HexQk::new(1))
                         as Box<dyn ReferenceElement>)
            } else {
                Some(factory_ref_elem(FactoryElemType::Hex, g))
            };
        }
        _ => {}
    }
    // Curved tetrahedra: MFEM's mesh `Nodes` are a grid function of the H¹
    // collection, i.e. their parametric positions are the **closed
    // Gauss-Lobatto** points — the equispaced `factory::TetPk` would interpret
    // the same node values as belonging to different reference points (D49;
    // 7.5e-2 error on a curved P3 fixture).  Straight P1 tets never get here
    // (the affine fast path above), and `p = 2`'s closed points are the
    // midpoints, so this only changes `p ≥ 3`.
    if matches!(et, ElementType::Tet4 | ElementType::Tet10) && g > 1 {
        return Some(Box::new(fem_element::lagrange::factory::H1TetPk::new(g as usize))
                    as Box<dyn ReferenceElement>);
    }
    let order = if g > 1 { g } else { 1 };
    let ft = mesh_type_to_factory(et);
    Some(factory_ref_elem(ft, order))
}

/// Whether this element type has a constant (affine) Jacobian.
///
/// Affine if P1 simplex geometry (`geom_order == 1` and non-tensor-product).
/// Non-affine for curved simplex elements (geom_order > 1) and all tensor-product
/// elements (Quad/Hex) which use isoparametric mapping.
pub(crate) fn is_affine(et: ElementType, geom_order: u8) -> bool {
    if geom_order > 1 { return false; }
    matches!(et, ElementType::Tri3 | ElementType::Tet4 | ElementType::Line2)
}

/// [`MeshTopology`] view whose node coordinates come from the mesh's
/// **geometry** node table ([`MeshTopology::geom_coords_of`] /
/// [`MeshTopology::geometry_nodes`]) instead of the folded vertex table.
///
/// Geometrically periodic meshes keep the *pre-merge* per-element geometry in
/// that table (MFEM `MakePeriodic` snapshots the nodal `Nodes` grid function
/// before renumbering the vertices with `v2v`), so a face/edge that crosses a
/// periodic seam has its true (unfolded) coordinates only there.  Reading
/// `node_coords` with geometry node ids would be wrong — the geometry table
/// uses a different index range and can exceed the vertex table.
///
/// For meshes without per-element geometry this view is an exact pass-through
/// (`geom_coords_of`/`geometry_nodes` fall back to
/// `node_coords`/`element_nodes`), so it can be used unconditionally.
pub(crate) struct GeomCoordView<'a, M: MeshTopology + ?Sized>(pub &'a M);

impl<M: MeshTopology + ?Sized> MeshTopology for GeomCoordView<'_, M> {
    fn dim(&self) -> u8 { self.0.dim() }
    fn n_nodes(&self) -> usize { self.0.geom_n_nodes() }
    fn n_elements(&self) -> usize { self.0.n_elements() }
    fn n_boundary_faces(&self) -> usize { self.0.n_boundary_faces() }
    fn element_nodes(&self, elem: u32) -> &[u32] { self.0.geometry_nodes(elem) }
    fn element_type(&self, elem: u32) -> ElementType { self.0.element_type(elem) }
    fn element_tag(&self, elem: u32) -> i32 { self.0.element_tag(elem) }
    fn node_coords(&self, node: u32) -> &[f64] { self.0.geom_coords_of(node) }
    fn face_nodes(&self, face: u32) -> &[u32] { self.0.face_nodes(face) }
    fn face_tag(&self, face: u32) -> i32 { self.0.face_tag(face) }
    fn face_elements(&self, face: u32) -> (u32, Option<u32>) { self.0.face_elements(face) }
}

/// Affine simplex [`ElementTransformation`] built from the element's
/// **geometry** nodes.
///
/// This is the affine fast path's counterpart of the isoparametric branches,
/// which already resolve `geometry_nodes` + `geom_coords_of`: on a
/// geometrically periodic mesh the per-element geometry (not the merged
/// vertex table) is what defines the element shape, and Tri3/Tet4 elements
/// crossing a seam would otherwise be assembled with a folded, wrong Jacobian
/// (round-9 leftover: Quad/Hex were fixed, simplices were not).
///
/// Non-periodic, non-curved meshes are bit-identical to
/// `ElementTransformation::from_simplex_nodes(mesh, mesh.element_nodes(e))`.
pub(crate) fn simplex_transformation<M: MeshTopology + ?Sized>(
    mesh: &M,
    e: u32,
) -> ElementTransformation {
    ElementTransformation::from_simplex_nodes(&GeomCoordView(mesh), mesh.geometry_nodes(e))
}
///
/// `J_{ij}(ξ) = Σ_k x_k[i] · ∂φ_k/∂ξ_j`
///
/// where φ_k are the **geometry** basis functions (same as solution basis for
/// Q1 elements) and x_k are the physical node coordinates.
///
/// Returns `(J, det J, x_phys)`.
fn isoparametric_jacobian<M: MeshTopology>(
    mesh: &M,
    nodes: &[u32],
    geo_elem: &dyn ReferenceElement,
    xi: &[f64],
    dim: usize,
    use_lex: bool,
) -> (DMatrix<f64>, f64, Vec<f64>) {
    let n_geo = geo_elem.n_dofs();
    let mut grad_geo = vec![0.0_f64; n_geo * dim];
    let mut phi_geo  = vec![0.0_f64; n_geo];
    geo_elem.eval_grad_basis(xi, &mut grad_geo);
    geo_elem.eval_basis(xi, &mut phi_geo);

    // Curved quad geometry uses MFEM's L2_T1 mesh-nodes field (LEX tensor
    // order: v2=(0,1), v3=(1,1)); the solution/geometry element for
    // non-curved quads is the H1 topological order (BiLinearGeo2D), in which
    // case no reordering is applied.  Reorder both the geometry nodes and
    // the basis rows to lex so the Jacobian accumulation order is
    // bit-identical to MFEM's EvalJacobian.
    let (nodes_r, grad_r, phi_r): (Vec<u32>, Vec<f64>, Vec<f64>) =
        if use_lex && dim == 2 && nodes.len() == 4 {
            let mut grad_lex = vec![0.0; n_geo * dim];
            let mut phi_lex = vec![0.0; n_geo];
            for (li, &hi) in [0usize, 1, 3, 2].iter().enumerate() {
                grad_lex[li * dim..li * dim + dim]
                    .copy_from_slice(&grad_geo[hi * dim..hi * dim + dim]);
                phi_lex[li] = phi_geo[hi];
            }
            (
                vec![nodes[0], nodes[1], nodes[3], nodes[2]],
                grad_lex,
                phi_lex,
            )
        } else {
            (nodes.to_vec(), grad_geo, phi_geo)
        };

    let mut j = DMatrix::<f64>::zeros(dim, dim);
    let mut xp = vec![0.0_f64; dim];

    for k in 0..n_geo {
        let xk = mesh.geom_coords_of(nodes_r[k]);
        for i in 0..dim {
            // MFEM kernels::AddMult: `Adata[i+j*Aheight] += val * Bdata[...]`
            // with plain multiply-then-add — NOT FMA (serial g++ builds do not
            // fuse; rustc's mul_add would introduce last-ulp differences).
            xp[i] += phi_r[k] * xk[i];
            for d in 0..dim {
                j[(i, d)] += xk[i] * grad_r[k * dim + d];
            }
        }
    }
    // MFEM CalcDeterminant: 2D det = J00*J11 - J01*J10 (nalgebra's
    // DMatrix::determinant can differ by 1 ulp for 2x2).
    let det = if dim == 2 {
        j[(0, 0)] * j[(1, 1)] - j[(0, 1)] * j[(1, 0)]
    } else {
        j.determinant()
    };
    (j, det, xp)
}

/// Surface element Jacobian info for 2D-in-3D meshes.
///
/// For a 2D surface element embedded in 3D space:
/// - J is a 3×2 matrix (mapping from 2D reference to 3D physical), returned
///   column-major as a flat `Vec<f64>` of length `embed_dim * tdim`
///   (`j[i + d*embed_dim] = ∂x_i/∂ξ_d`)
/// - G = J^T·J is the 2×2 metric tensor; `ginv` is its inverse, row-major
///   `[[a, b], [b, c]]`
/// - measure = sqrt(det(G)) = |J₁ × J₂| (surface area element)
///
/// The true physical (tangential) gradient of a reference-gradient is
/// `∇_surf φ = J · G⁻¹ · ∇_ref φ` (3 components); the assembly uses that to
/// support 3×3 matrix coefficients (e.g. MFEM ex29's anisotropic σ).
///
/// Returns `(measure, j, ginv, x_phys_3d)`.
pub(crate) fn surface_jacobian<M: MeshTopology>(
    mesh: &M,
    nodes: &[u32],
    geo_elem: &dyn ReferenceElement,
    xi: &[f64],
    embed_dim: usize,  // = 3 for Mesh<3>
    tdim: usize,       // = 2 for surface
) -> (f64, Vec<f64>, [f64; 3], Vec<f64>) {
    let n_geo = geo_elem.n_dofs();
    let mut grad_geo = vec![0.0_f64; n_geo * tdim];
    let mut phi_geo  = vec![0.0_f64; n_geo];
    geo_elem.eval_grad_basis(xi, &mut grad_geo);
    geo_elem.eval_basis(xi, &mut phi_geo);

    // 3×2 Jacobian: J[i][d] = Σ_k x_k[i] · ∂φ_k/∂ξ_d  (column-major)
    let mut j = vec![0.0_f64; embed_dim * tdim]; // [col0(3), col1(3)]
    let mut xp = vec![0.0_f64; embed_dim];
    for k in 0..n_geo {
        let xk = mesh.geom_coords_of(nodes[k]);
        for i in 0..embed_dim {
            xp[i] += phi_geo[k] * xk[i];
            for d in 0..tdim {
                j[i + d * embed_dim] += xk[i] * grad_geo[k * tdim + d];
            }
        }
    }

    // Metric G = J^T·J (2×2)
    let g00 = j[0]*j[0] + j[1]*j[1] + j[2]*j[2]; // col0·col0
    let g01 = j[0]*j[3] + j[1]*j[4] + j[2]*j[5]; // col0·col1
    let g11 = j[3]*j[3] + j[4]*j[4] + j[5]*j[5]; // col1·col1

    let det_g = g00 * g11 - g01 * g01;
    let measure = det_g.sqrt();

    // G⁻¹ (2×2 inverse metric), row-major [[a, b], [b, c]]
    let inv_det = 1.0 / det_g;
    let a = g11 * inv_det;
    let b = -g01 * inv_det;
    let c = g00 * inv_det;

    (measure, j, [a, b, c], xp)
}

/// Transform reference gradients to physical gradients:
/// `grad_phys[i] = J^{−T} grad_ref[i]`.
fn transform_grads(
    j_inv_t: &DMatrix<f64>,
    grad_ref: &[f64],
    grad_phys: &mut [f64],
    n_ldofs: usize,
    dim: usize,
) {
    for i in 0..n_ldofs {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim {
                s += j_inv_t[(j, k)] * grad_ref[i * dim + k];
            }
            grad_phys[i * dim + j] = s;
        }
    }
}

/// MFEM `ElementTransformation::AdjugateJacobian()`: the classical adjugate
/// (cofactor matrix) of J.  2-D: adj(J) = [[J11, -J01], [-J10, J00]].
pub(crate) fn adjugate_2d(j: &DMatrix<f64>) -> DMatrix<f64> {
    let mut a = DMatrix::<f64>::zeros(2, 2);
    a[(0, 0)] = j[(1, 1)];
    a[(0, 1)] = -j[(0, 1)];
    a[(1, 0)] = -j[(1, 0)];
    a[(1, 1)] = j[(0, 0)];
    a
}

/// 3-D adjugate (cofactor) matrix of J — MFEM `AdjugateJacobian` for
/// hexahedra/tetrahedra.  Missing before, so any non-affine 3-D assembly
/// (e.g. ex34's SubMesh with curved/Hex geometry) hit an out-of-bounds
/// `adj[(k,j)]` in [`transform_grads_adj`].
pub(crate) fn adjugate_3d(j: &DMatrix<f64>) -> DMatrix<f64> {
    let j00 = j[(0, 0)]; let j01 = j[(0, 1)]; let j02 = j[(0, 2)];
    let j10 = j[(1, 0)]; let j11 = j[(1, 1)]; let j12 = j[(1, 2)];
    let j20 = j[(2, 0)]; let j21 = j[(2, 1)]; let j22 = j[(2, 2)];
    let mut a = DMatrix::<f64>::zeros(3, 3);
    a[(0, 0)] = j11 * j22 - j12 * j21;
    a[(0, 1)] = j02 * j21 - j01 * j22;
    a[(0, 2)] = j01 * j12 - j02 * j11;
    a[(1, 0)] = j12 * j20 - j10 * j22;
    a[(1, 1)] = j00 * j22 - j02 * j20;
    a[(1, 2)] = j02 * j10 - j00 * j12;
    a[(2, 0)] = j10 * j21 - j11 * j20;
    a[(2, 1)] = j01 * j20 - j00 * j21;
    a[(2, 2)] = j00 * j11 - j01 * j10;
    a
}

/// Transform reference gradients by the adjugate Jacobian (MFEM
/// `Mult(dshape, AdjugateJacobian, dshapedxt)`): grad_phys(i,j) =
/// Σ_k adj(J)(k,j)·grad_ref(i,k).  No det division — the diffusion weight
/// carries `1/det` instead (MFEM `w = ip.weight / Trans.Weight()`), which
/// keeps the floating-point path bit-identical (using J⁻¹ = adj/det and a
/// `×det` weight differs by ~1 ulp).
pub(crate) fn transform_grads_adj(
    adj: &DMatrix<f64>,
    grad_ref: &[f64],
    grad_phys: &mut [f64],
    n_ldofs: usize,
    dim: usize,
) {
    for i in 0..n_ldofs {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim {
                s += adj[(k, j)] * grad_ref[i * dim + k];
            }
            grad_phys[i * dim + j] = s;
        }
    }
}

// ─── Volume element kernels (serial; used by parallel driver via Rayon) ─────

// ─── Element scratch buffer ───────────────────────────────────────────────────

/// Per-thread reusable scratch storage for element-level assembly.
///
/// Allocating `k_elem`, `phi`, `grad_ref`, `grad_phys` fresh for every element
/// is the dominant allocation pressure in the assembly loop.  This struct holds
/// those buffers between element calls so that `Vec::resize` can reuse the
/// existing capacity (no heap allocation when the element type is uniform across
/// the mesh, which is the common case).
///
/// The serial assembler creates one `ElementScratch` before the element loop.
/// The parallel assembler carries one inside each Rayon fold closure
/// (one per thread), so no locking is needed.
struct ElementScratch {
    k_elem:    Vec<f64>,
    f_elem:    Vec<f64>,
    phi:       Vec<f64>,
    grad_ref:  Vec<f64>,
    grad_phys: Vec<f64>,
    global_dofs: Vec<usize>,  // reused per-element to avoid allocation
}

impl ElementScratch {
    fn new() -> Self {
        Self {
            k_elem:    Vec::new(),
            f_elem:    Vec::new(),
            phi:       Vec::new(),
            grad_ref:  Vec::new(),
            grad_phys: Vec::new(),
            global_dofs: Vec::new(),
        }
    }
}

fn accumulate_volume_bilinear_element<S: FESpace>(
    space: &S,
    e: u32,
    integrators: &[&dyn BilinearIntegrator],
    quad: &QuadratureRule,
    quad_order: u8,
    coo: &mut CooMatrix<f64>,
    scratch: &mut ElementScratch,
    ref_elem: &dyn ReferenceElement,
) {
    let mesh    = space.mesh();
    let edim    = mesh.dim() as usize;   // embedding dimension (2 or 3)
    let tdim    = mesh.topological_dim() as usize; // element dimension (2 for surface)
    let is_surface = edim != tdim;
    let dim     = if is_surface { edim } else { edim }; // surface: 3-component true gradients
    let order   = space.element_order(e);
    let order0  = space.element_order(0);

    // Mixed meshes (e.g. Tet4 + Prism6): the quadrature rule passed in is
    // built from the FIRST element type, which is invalid on other element
    // types (wrong reference domain / weights).  Variable-order (hp) meshes:
    // a different polynomial order means a different basis (QuadQk(p)), so
    // both the rule and the reference element must be re-derived per element.
    let elem_type0 = mesh.element_type(0);
    let elem_type  = mesh.element_type(e);
    let differs = elem_type != elem_type0 || order != order0;
    let quad_owned;
    let ref_elem_owned;
    let quad: &QuadratureRule = if !differs {
        quad
    } else {
        // Use the caller's quadrature order (like the uniform path).
        quad_owned = ref_elem_vol_for_space(space, elem_type, order).quadrature(quad_order);
        &quad_owned
    };
    // Mixed/variable-order meshes: also re-derive the reference element itself
    // (basis, n_dofs).
    let ref_elem: &dyn ReferenceElement = if !differs {
        ref_elem
    } else {
        ref_elem_owned = ref_elem_vol_for_space(space, elem_type, order);
        &*ref_elem_owned
    };

    // Use the caller-provided reference element (custom basis, e.g. Bernstein
    // QuadPosQk for MFEM's H1_FECollection BasisType::Positive).
    let n_ldofs   = ref_elem.n_dofs();

    let raw_dofs: &[DofId] = space.element_dofs(e);
    scratch.global_dofs.clear();
    scratch.global_dofs.extend(raw_dofs.iter().map(|&d| d as usize));
    let global_dofs = &scratch.global_dofs;
    let n_elem_dofs = global_dofs.len();
    let nodes = mesh.element_nodes(e);
    let elem_tag = mesh.element_tag(e);

    let g_order = mesh.geom_order();
    let affine = !is_surface && is_affine(elem_type, g_order);
    let geo_elem = geo_ref_elem(mesh, e);

    let affine_tr = if affine {
        Some(simplex_transformation(mesh, e))
    } else {
        None
    };

    // Reuse scratch buffers: resize zeroes new entries, existing capacity is kept.
    let k_size = n_elem_dofs * n_elem_dofs;
    scratch.k_elem.clear();
    scratch.k_elem.resize(k_size, 0.0);

    scratch.phi.resize(n_ldofs, 0.0);
    scratch.grad_ref.resize(n_ldofs * dim, 0.0);
    scratch.grad_phys.resize(n_ldofs * dim, 0.0);

    for (q, xi) in quad.points.iter().enumerate() {
        if is_surface {
            // ── Surface path (2D elements in 3D space) ─────────────────────────
            let geo_p1 = ref_elem_vol(elem_type, 1);
            let (geo, geo_nds): (&dyn ReferenceElement, &[u32]) =
                if let Some(ref ge) = geo_elem {
                    (ge.as_ref(), mesh.geometry_nodes(e))
                } else {
                    (geo_p1.as_ref(), nodes)
                };
            let xi_g = geom_quad_point(elem_type, order, xi);
            let (measure, j, ginv, xp) =
                surface_jacobian(mesh, geo_nds, geo, &xi_g, edim, tdim);
            let w = quad.weights[q] * measure;

            ref_elem.eval_basis(xi, &mut scratch.phi);
            ref_elem.eval_grad_basis(xi, &mut scratch.grad_ref);
            // True tangential gradients: ∇_surf φ_i = J · G⁻¹ · ∇_ref φ_i (3 comps)
            let (j00, j01, j10, j11, j20, j21) = (j[0], j[3], j[1], j[4], j[2], j[5]);
            let (gi00, gi01, gi11) = (ginv[0], ginv[1], ginv[2]);
            for i in 0..n_ldofs {
                let gr = &scratch.grad_ref[i * 2..i * 2 + 2];
                let t0 = gi00 * gr[0] + gi01 * gr[1];
                let t1 = gi01 * gr[0] + gi11 * gr[1];
                scratch.grad_phys[i * 3]     = j00 * t0 + j01 * t1;
                scratch.grad_phys[i * 3 + 1] = j10 * t0 + j11 * t1;
                scratch.grad_phys[i * 3 + 2] = j20 * t0 + j21 * t1;
            }

            let qp = QpData {
                n_dofs:    n_elem_dofs,
                dim,
                weight:    w,
                phys_weight: w,
                ref_weight: quad.weights[q],
                phi:       &scratch.phi,
                grad_phys: &scratch.grad_phys,
                x_phys:    &xp,
                elem_id:   e,
                elem_tag,
                elem_dofs: Some(&raw_dofs),
            };
            for integ in integrators {
                integ.add_to_element_matrix(&qp, &mut scratch.k_elem);
            }
            continue;
        }
        if affine {
            let tr = affine_tr.as_ref().unwrap();
            // MFEM has no affine fast path: `ElementTransformation::Weight()` and
            // `AdjugateJacobian()` are used for every element, and every
            // integrator is written against them (DiffusionIntegrator:
            // `dshapedxt = Mult(dshape, AdjugateJacobian)`, `w = ip.weight/Weight()`;
            // ConvectionIntegrator: same dshapedxt with the bare `ip.weight`).
            // This branch therefore keeps the *same* convention as the
            // isoparametric branch below — grad_phys = adjJᵀ∇φ (|detJ|-scaled)
            // and weight = ip.weight/|detJ| — so that both `weight × grad_phys`
            // and `ip.weight × grad_phys` integrator families are correct.
            // (Previously this branch stored the true J⁻ᵀ∇φ with weight
            // ip.weight·|detJ|, which is equivalent only for the `weight × grad`
            // family and silently dropped |detJ| for the `ip.weight × grad`
            // family, e.g. ConvectionIntegrator.)
            let w = quad.weights[q] / tr.det_j().abs();

            ref_elem.eval_basis(xi, &mut scratch.phi);
            ref_elem.eval_grad_basis(xi, &mut scratch.grad_ref);
            let adj = if dim == 3 {
                adjugate_3d(tr.jacobian())
            } else {
                adjugate_2d(tr.jacobian())
            };
            transform_grads_adj(&adj, &scratch.grad_ref, &mut scratch.grad_phys, n_ldofs, dim);

            let xp = tr.map_to_physical(xi);
            let qp = QpData {
                n_dofs:    n_elem_dofs,
                dim,
                weight:    w,
                // True physical measure (mass-type integrands), identical to the
                // isoparametric branch below.
                phys_weight: quad.weights[q] * tr.det_j().abs(),
                ref_weight: quad.weights[q],
                phi:       &scratch.phi,
                grad_phys: &scratch.grad_phys,
                x_phys:    &xp,
                elem_id:   e,
                elem_tag,
                elem_dofs: Some(&raw_dofs),
            };

            for integ in integrators {
                integ.add_to_element_matrix(&qp, &mut scratch.k_elem);
            }
            continue;
        } else {
            let geo = geo_elem.as_ref().unwrap();
            let geo_nds = if is_surface { nodes } else { mesh.geometry_nodes(e) };
            let xi_g = geom_quad_point(elem_type, order, xi);
            let (jac_qp, det_qp, xp_qp) = isoparametric_jacobian(
                mesh, geo_nds, geo.as_ref(), &xi_g, dim,
                mesh.geom_order() > 1, // curved → L2_T1 lexicographic order
            );
            // MFEM DiffusionIntegrator: w = ip.weight / Trans.Weight() where
            // Trans.Weight() = |det J| for square elements (not ×|det|).
            let w = quad.weights[q] / det_qp.abs();
            if det_qp.abs() < 1e-12 {
                if cfg!(debug_assertions) {
                    eprintln!("warning: degenerate element {} at quad point {}, det={:.3e}", e, q, det_qp);
                }
                continue;
            }
            // MFEM: dshapedxt = Mult(dshape, AdjugateJacobian, dshapedxt).
            let adj = if dim == 3 { adjugate_3d(&jac_qp) } else { adjugate_2d(&jac_qp) };
            ref_elem.eval_basis(xi, &mut scratch.phi);
            ref_elem.eval_grad_basis(xi, &mut scratch.grad_ref);
            transform_grads_adj(&adj, &scratch.grad_ref, &mut scratch.grad_phys, n_ldofs, dim);
            let w_phys = quad.weights[q] * det_qp.abs();

            let qp = QpData {
                n_dofs:    n_elem_dofs,
                dim,
                weight:    w,
                phys_weight: w_phys,
                ref_weight: quad.weights[q],
                phi:       &scratch.phi,
                grad_phys: &scratch.grad_phys,
                x_phys:    &xp_qp,
                elem_id:   e,
                elem_tag,
                elem_dofs: Some(&raw_dofs),
            };
            for integ in integrators {
                integ.add_to_element_matrix(&qp, &mut scratch.k_elem);
            }
            continue;
        }
    }

    coo.add_element_matrix(&global_dofs, &scratch.k_elem);
}

fn accumulate_volume_linear_element<S: FESpace>(
    space: &S,
    e: u32,
    integrators: &[&dyn LinearIntegrator],
    quad: &QuadratureRule,
    quad_order: u8,
    rhs: &mut [f64],
    scratch: &mut ElementScratch,
) {
    let mesh    = space.mesh();
    let edim    = mesh.dim() as usize;   // embedding dimension (2 or 3)
    let tdim    = mesh.topological_dim() as usize;
    let is_surface = edim != tdim;
    let dim     = if is_surface { edim } else { edim };
    let order   = space.element_order(e);

    let elem_type = mesh.element_type(e);
    let ref_elem  = ref_elem_vol_for_space(space, elem_type, order);
    let n_ldofs   = ref_elem.n_dofs();
    let quad_owned;
    // Mixed meshes: the passed-in rule is built from the FIRST element type;
    // re-derive it from this element's own reference element (same order).
    let quad: &QuadratureRule = if mesh.element_type(0) == elem_type {
        quad
    } else {
        quad_owned = ref_elem.quadrature(quad_order);
        &quad_owned
    };

    let raw_dofs: &[DofId] = space.element_dofs(e);
    scratch.global_dofs.clear();
    scratch.global_dofs.extend(raw_dofs.iter().map(|&d| d as usize));
    let global_dofs = &scratch.global_dofs;
    let nodes = mesh.element_nodes(e);
    let elem_tag = mesh.element_tag(e);

    let g_order = mesh.geom_order();
    let affine = !is_surface && is_affine(elem_type, g_order);
    let geo_elem = geo_ref_elem(mesh, e);

    let affine_tr = if affine {
        Some(simplex_transformation(mesh, e))
    } else {
        None
    };

    let n_elem_dofs = global_dofs.len();

    // Reuse scratch buffers.
    scratch.f_elem.clear();
    scratch.f_elem.resize(n_elem_dofs, 0.0);
    scratch.phi.resize(n_ldofs, 0.0);
    scratch.grad_ref.resize(n_ldofs * dim, 0.0);
    scratch.grad_phys.resize(n_ldofs * dim, 0.0);

    for (q, xi) in quad.points.iter().enumerate() {
        let (w, xp);
        if is_surface {
            let geo_p1 = ref_elem_vol(elem_type, 1);
            let (geo, geo_nds): (&dyn ReferenceElement, &[u32]) =
                if let Some(ref ge) = geo_elem {
                    (ge.as_ref(), mesh.geometry_nodes(e))
                } else {
                    (geo_p1.as_ref(), nodes)
                };
            let xi_g = geom_quad_point(elem_type, order, xi);
            let (measure, _j, _ginv, xp_surf) =
                surface_jacobian(mesh, geo_nds, geo, &xi_g, edim, tdim);
            w = quad.weights[q] * measure;
            ref_elem.eval_basis(xi, &mut scratch.phi);
            xp = xp_surf;
        } else if affine {
            let tr = affine_tr.as_ref().unwrap();
            w = quad.weights[q] * tr.det_j().abs();
            ref_elem.eval_basis(xi, &mut scratch.phi);
            ref_elem.eval_grad_basis(xi, &mut scratch.grad_ref);
            transform_grads(tr.jacobian_inv_t(), &scratch.grad_ref, &mut scratch.grad_phys, n_ldofs, dim);
            xp = tr.map_to_physical(xi);
        } else {
            let geo = geo_elem.as_ref().unwrap();
            let geo_nds = if is_surface { nodes } else { mesh.geometry_nodes(e) };
            let xi_g = geom_quad_point(elem_type, order, xi);
            let (jac_qp, det_qp, xp_qp) = isoparametric_jacobian(
                mesh, geo_nds, geo.as_ref(), &xi_g, dim,
                mesh.geom_order() > 1, // curved → L2_T1 lexicographic order
            );
            w = quad.weights[q] * det_qp.abs();
            if det_qp.abs() < 1e-12 {
                if cfg!(debug_assertions) {
                    eprintln!("warning: degenerate element {} at quad point {}, det={:.3e}", e, q, det_qp);
                }
                continue;
            }
            let jit = jac_qp.try_inverse().expect("invertible").transpose();
            ref_elem.eval_basis(xi, &mut scratch.phi);
            ref_elem.eval_grad_basis(xi, &mut scratch.grad_ref);
            transform_grads(&jit, &scratch.grad_ref, &mut scratch.grad_phys, n_ldofs, dim);
            xp = xp_qp;
        }

        let qp = QpData {
            n_dofs:    n_elem_dofs,
            dim,
            weight:    w,
            phys_weight: w,
            ref_weight: quad.weights[q],
            phi:       &scratch.phi,
            grad_phys: &scratch.grad_phys,
            x_phys:    &xp,
            elem_id:   e,
            elem_tag,
            elem_dofs: Some(&raw_dofs),
        };

        for integ in integrators {
            integ.add_to_element_vector(&qp, &mut scratch.f_elem);
        }
    }

    coo_add_element_vec(&global_dofs, &scratch.f_elem, rhs);
}

fn accumulate_boundary_linear_face(
    mesh: &(dyn MeshTopology + Sync),
    f: u32,
    face_dofs: &(dyn Fn(u32) -> Vec<DofId> + Sync),
    order: u8,
    integrators: &[&dyn BoundaryLinearIntegrator],
    quad_order: u8,
    rhs: &mut [f64],
) {
    let dim = mesh.dim() as usize;
    let fdofs: Vec<DofId> = face_dofs(f);
    let n_fdofs = fdofs.len();

    let face_type = match mesh.face_nodes(f).len() {
        2 => ElementType::Line2,
        3 => ElementType::Tri3,
        4 => ElementType::Quad4,
        _ => panic!("unsupported boundary face node count"),
    };
    let ref_elem = ref_elem_face(face_type, order);
    let quad = ref_elem.quadrature(quad_order);

    let face_nodes = mesh.face_nodes(f);
    let geom = boundary_face_geom(mesh, f, face_nodes, dim);
    let face_tag = mesh.face_tag(f);

    let mut phi = vec![0.0_f64; n_fdofs];
    let mut f_face = vec![0.0_f64; n_fdofs];

    for (q, xi) in quad.points.iter().enumerate() {
        let (face_j_mag, normal, xp) = geom.eval(xi);
        let w = quad.weights[q] * face_j_mag;
        ref_elem.eval_basis(xi, &mut phi);

        let qp = BdQpData {
            n_dofs: n_fdofs,
            dim,
            weight: w,
            phi: &phi,
            x_phys: &xp,
            normal: &normal,
            elem_id: 0,
            elem_tag: face_tag,
        };

        for integ in integrators {
            integ.add_to_face_vector(&qp, &mut f_face);
        }
    }

    let global: Vec<usize> = fdofs.iter().map(|&d| d as usize).collect();
    coo_add_element_vec(&global, &f_face, rhs);
}

fn accumulate_boundary_bilinear_face(
    mesh: &(dyn MeshTopology + Sync),
    f: u32,
    face_dofs: &(dyn Fn(u32) -> Vec<DofId> + Sync),
    order: u8,
    integrators: &[&dyn BoundaryBilinearIntegrator],
    quad_order: u8,
    coo: &mut CooMatrix<f64>,
) {
    let dim = mesh.dim() as usize;
    let fdofs: Vec<DofId> = face_dofs(f);
    let n_fdofs = fdofs.len();

    let face_type = match mesh.face_nodes(f).len() {
        2 => ElementType::Line2,
        3 => ElementType::Tri3,
        4 => ElementType::Quad4,
        _ => panic!("unsupported boundary face node count"),
    };
    let ref_elem = ref_elem_face(face_type, order);
    let quad = ref_elem.quadrature(quad_order);

    let face_nodes = mesh.face_nodes(f);
    let geom = boundary_face_geom(mesh, f, face_nodes, dim);
    let face_tag = mesh.face_tag(f);

    let mut phi = vec![0.0_f64; n_fdofs];
    let mut k_face = vec![0.0_f64; n_fdofs * n_fdofs];

    for (q, xi) in quad.points.iter().enumerate() {
        let (face_j_mag, normal, xp) = geom.eval(xi);
        let w = quad.weights[q] * face_j_mag;
        ref_elem.eval_basis(xi, &mut phi);

        let qp = BdQpData {
            n_dofs: n_fdofs,
            dim,
            weight: w,
            phi: &phi,
            x_phys: &xp,
            normal: &normal,
            elem_id: 0,
            elem_tag: face_tag,
        };

        for integ in integrators {
            integ.add_to_face_matrix(&qp, &mut k_face);
        }
    }

    let global: Vec<usize> = fdofs.iter().map(|&d| d as usize).collect();
    coo.add_element_matrix(&global, &k_face);
}

#[cfg(feature = "parallel")]
fn assemble_bilinear_volume_parallel<S: FESpace>(
    space: &S,
    integrators: &[&dyn BilinearIntegrator],
    quad: &QuadratureRule,
    quad_order: u8,
    ref_elem: &dyn ReferenceElement,
) -> CsrMatrix<f64> {
    let mesh = space.mesh();
    let n_dofs = space.n_dofs();

    // Thread-local pool: one CooMatrix per Rayon thread, reused across fold iterations.
    // This reduces allocations from O(n_elements) to O(n_threads).
    thread_local! {
        static TL_COO: RefCell<Option<CooMatrix<f64>>> = const { RefCell::new(None) };
    }

    mesh.elem_iter()
        .into_par_iter()
        .fold(
            || {
                let coo = TL_COO.with(|tl| {
                    let mut slot = tl.borrow_mut();
                    if let Some(mut coo) = slot.take() {
                        coo.nrows = n_dofs;
                        coo.ncols = n_dofs;
                        coo.clear();
                        coo
                    } else {
                        CooMatrix::<f64>::new(n_dofs, n_dofs)
                    }
                });
                (coo, ElementScratch::new())
            },
            |(mut local_coo, mut scratch), e| {
                accumulate_volume_bilinear_element(space, e, integrators, &quad, quad_order, &mut local_coo, &mut scratch, ref_elem);
                (local_coo, scratch)
            },
        )
        .reduce(
            || (CooMatrix::<f64>::new(n_dofs, n_dofs), ElementScratch::new()),
            |(mut a_coo, a_scratch), (b_coo, _)| {
                a_coo.append(b_coo);
                (a_coo, a_scratch)
            },
        )
        .0
        .into_csr()
}

#[cfg(feature = "parallel")]
fn assemble_linear_volume_parallel<S: FESpace>(
    space: &S,
    integrators: &[&dyn LinearIntegrator],
    quad: &QuadratureRule,
    quad_order: u8,
) -> Vec<f64> {
    let mesh = space.mesh();
    let n_dofs = space.n_dofs();
    mesh.elem_iter()
        .into_par_iter()
        .fold(
            || (vec![0.0_f64; n_dofs], ElementScratch::new()),
            |(mut local_rhs, mut scratch), e| {
                accumulate_volume_linear_element(space, e, integrators, quad, quad_order, &mut local_rhs, &mut scratch);
                (local_rhs, scratch)
            },
        )
        .reduce(
            || (vec![0.0_f64; n_dofs], ElementScratch::new()),
            |(mut a_rhs, a_scratch), (b_rhs, _)| {
                for i in 0..n_dofs {
                    a_rhs[i] += b_rhs[i];
                }
                (a_rhs, a_scratch)
            },
        )
        .0
}

#[cfg(feature = "parallel")]
fn assemble_boundary_linear_parallel(
    n_dofs: usize,
    mesh: &(dyn MeshTopology + Sync),
    face_ids: &[u32],
    face_dofs: &(dyn Fn(u32) -> Vec<DofId> + Sync),
    order: u8,
    integrators: &[&dyn BoundaryLinearIntegrator],
    quad_order: u8,
) -> Vec<f64> {
    face_ids
        .par_iter()
        .copied()
        .fold(
            || vec![0.0_f64; n_dofs],
            |mut local, f| {
                accumulate_boundary_linear_face(mesh, f, face_dofs, order, integrators, quad_order, &mut local);
                local
            },
        )
        .reduce(
            || vec![0.0_f64; n_dofs],
            |mut a, b| {
                for i in 0..n_dofs {
                    a[i] += b[i];
                }
                a
            },
        )
}

#[cfg(feature = "parallel")]
fn assemble_boundary_bilinear_parallel(
    n_dofs: usize,
    mesh: &(dyn MeshTopology + Sync),
    face_ids: &[u32],
    face_dofs: &(dyn Fn(u32) -> Vec<DofId> + Sync),
    order: u8,
    integrators: &[&dyn BoundaryBilinearIntegrator],
    quad_order: u8,
) -> CsrMatrix<f64> {
    face_ids
        .par_iter()
        .copied()
        .map(|f| {
            let mut local = CooMatrix::<f64>::new(n_dofs, n_dofs);
            accumulate_boundary_bilinear_face(mesh, f, face_dofs, order, integrators, quad_order, &mut local);
            local
        })
        .reduce(
            || CooMatrix::<f64>::new(n_dofs, n_dofs),
            |mut a, b| {
                a.append(b);
                a
            },
        )
        .into_csr()
}

// ─── Assembler ────────────────────────────────────────────────────────────────

/// Stateless assembly driver.
///
/// All methods are associated functions (no `self` needed) that take the
/// relevant space and integrators as arguments.
pub struct Assembler;

impl Assembler {
    // ── Volume bilinear form: K = Σ_e k_e ────────────────────────────────────

    /// Assemble the global stiffness matrix for a bilinear form.
    ///
    /// # Arguments
    /// * `space`       — finite element space (provides mesh + DOF map).
    /// * `integrators` — slice of bilinear-form contributions to accumulate.
    /// * `quad_order`  — polynomial order that the quadrature rule integrates exactly.
    ///
    /// # Returns
    /// Assembled `CsrMatrix<f64>` in CSR format.
    pub fn assemble_bilinear<S: FESpace>(
        space:       &S,
        integrators: &[&dyn BilinearIntegrator],
        quad_order:  u8,
    ) -> CsrMatrix<f64> {
        // MFEM semantics: each integrator may select its own quadrature order.
        // If any integrator requests an explicit order, assemble integrators
        // individually on their own quadrature rules and accumulate.
        let n_dofs = space.n_dofs();
        let space_order = space.element_order(0);
        if integrators.iter().any(|i| i.integration_order(space_order).is_some()) {
            let mut acc: Option<CsrMatrix<f64>> = None;
            for integ in integrators {
                let qo = integ.integration_order(space_order).unwrap_or(quad_order);
                let m = Self::assemble_bilinear_inner(space, &[*integ], qo, None, None);
                acc = Some(match acc {
                    None => m,
                    Some(a) => a.add(&m),
                });
            }
            return acc.unwrap_or_else(|| CsrMatrix::new_empty(n_dofs, n_dofs));
        }

        Self::assemble_bilinear_inner(space, integrators, quad_order, None, None)
    }

    /// Assemble a bilinear form where each integrator is applied on a subset
    /// of elements selected by an element-attribute marker (MFEM
    /// `BilinearForm::AddDomainIntegrator(integ, marker)`).
    ///
    /// # Arguments
    /// * `space`       — finite element space (provides mesh + DOF map).
    /// * `integrators` — `(integrator, marker)` pairs; a `None` marker means
    ///   the integrator applies to all elements.  A marker is an array of
    ///   length `max_attr` (the largest element attribute number in the mesh)
    ///   with `marker[attr-1] = 1` selecting attribute `attr` (MFEM
    ///   `AttributeSets::AttrToMarker` layout).
    /// * `quad_order`  — polynomial order that the quadrature rule integrates exactly.
    ///
    /// # Returns
    /// Assembled `CsrMatrix<f64>` in CSR format.
    pub fn assemble_bilinear_marked<S: FESpace>(
        space:       &S,
        integrators: &[(&dyn BilinearIntegrator, Option<&[i32]>)],
        quad_order:  u8,
    ) -> CsrMatrix<f64> {
        let n_dofs = space.n_dofs();
        let mut acc: Option<CsrMatrix<f64>> = None;
        for (integ, marker) in integrators {
            let m = Self::assemble_bilinear_inner(space, &[*integ], quad_order, None, *marker);
            acc = Some(match acc {
                None => m,
                Some(a) => a.add(&m),
            });
        }
        acc.unwrap_or_else(|| CsrMatrix::new_empty(n_dofs, n_dofs))
    }

    /// Assemble a bilinear form using an explicit reference element (custom
    /// basis) instead of the space's default one.
    ///
    /// Used to reproduce MFEM spaces with a non-default `BasisType`, e.g.
    /// `BasisType::Positive` (Bernstein) for the elasticity space of ex37:
    /// the DOF layout (H1 ordering) is unchanged, only the basis functions
    /// differ, so the same `FESpace` can be assembled with `QuadPosQk`.
    pub fn assemble_bilinear_with_ref<S: FESpace>(
        space:       &S,
        integrators: &[&dyn BilinearIntegrator],
        quad_order:  u8,
        ref_elem:    &dyn ReferenceElement,
    ) -> CsrMatrix<f64> {
        // MFEM semantics: each integrator may select its own quadrature order.
        // If any integrator requests an explicit order, assemble integrators
        // individually on their own quadrature rules and accumulate.
        let n_dofs = space.n_dofs();
        let space_order = space.element_order(0);
        if integrators.iter().any(|i| i.integration_order(space_order).is_some()) {
            let mut acc: Option<CsrMatrix<f64>> = None;
            for integ in integrators {
                let qo = integ.integration_order(space_order).unwrap_or(quad_order);
                let m = Self::assemble_bilinear_inner(space, &[*integ], qo, Some(ref_elem), None);
                acc = Some(match acc {
                    None => m,
                    Some(a) => a.add(&m),
                });
            }
            return acc.unwrap_or_else(|| CsrMatrix::new_empty(n_dofs, n_dofs));
        }

        Self::assemble_bilinear_inner(space, integrators, quad_order, Some(ref_elem), None)
    }

    /// Core assembly loop shared by [`Self::assemble_bilinear`]; see there for
    /// argument semantics.
    fn assemble_bilinear_inner<S: FESpace>(
        space:       &S,
        integrators: &[&dyn BilinearIntegrator],
        quad_order:  u8,
        ref_elem_override: Option<&dyn ReferenceElement>,
        elem_marker: Option<&[i32]>,
    ) -> CsrMatrix<f64> {
        let mesh   = space.mesh();
        let n_dofs = space.n_dofs();

        // Precompute quadrature rule from the first element's type (same for all).
        // Use the ACTUAL solution order: for Quad4 order >= 4 the basis (QuadQk)
        // lives on [0,1]^2 while order <= 3 (QuadQ1/Q2/Q3) lives on [-1,1]^2, and
        // the quadrature domain must match the basis domain.
        let elem_type = mesh.element_type(0);
        let owned_default;
        let ref_elem: &dyn ReferenceElement = match ref_elem_override {
            Some(r) => r,
            None => {
                owned_default = ref_elem_vol_for_space(space, elem_type, space.element_order(0));
                &*owned_default
            }
        };
        let quad = ref_elem.quadrature(quad_order);

        // Estimate raw nnz for COO pre-allocation.
        // Each element contributes `dofs_per_elem^2` triplets.
        // Use first element's n_dofs for uniform-order meshes (common case).
        let dofs_per_elem = ref_elem.n_dofs();
        let est_nnz  = mesh.n_elements() as usize * dofs_per_elem * dofs_per_elem;

        #[cfg(feature = "parallel")]
        {
            if elem_marker.is_none() && mesh.n_elements() >= assembly_parallel_min_elems() {
                return assemble_bilinear_volume_parallel(space, integrators, &quad, quad_order, ref_elem);
            }
        }

        let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
        coo.reserve(est_nnz.min(10_000_000)); // cap to avoid giant pre-allocs for hp meshes
        let mut scratch = ElementScratch::new();
        for e in mesh.elem_iter() {
            if let Some(marker) = elem_marker {
                let tag = mesh.element_tag(e);
                if tag <= 0 || (tag as usize) > marker.len() || marker[(tag - 1) as usize] == 0 {
                    continue;
                }
            }
            accumulate_volume_bilinear_element(space, e, integrators, &quad, quad_order, &mut coo, &mut scratch, ref_elem);
        }
        coo.into_csr()
    }

    /// Assemble the global matrix AND return per-element matrix data.
    ///
    /// Returns `(csr_matrix, elem_dofs, elem_mats, ldofs, n_elems)`:
    /// - `csr_matrix` — same as [`assemble_bilinear`]
    /// - `elem_dofs`  — flattened per-element DOFs: `elem_dofs[e * ld + i]`
    /// - `elem_mats`  — flattened per-element matrices: `elem_mats[e * ld² + i * ld + j]`
    ///
    /// The element matrices come from the **exact same** integration loop as the CSR,
    /// ensuring bitwise-identical element-level values.
    pub fn assemble_bilinear_with_elements<S: FESpace>(
        space:       &S,
        integrators: &[&dyn BilinearIntegrator],
        quad_order:  u8,
    ) -> (CsrMatrix<f64>, Vec<u32>, Vec<f64>, usize, usize) {
        let mesh    = space.mesh();
        let n_dofs  = space.n_dofs();
        let n_elems = mesh.n_elements();

        // Quadrature from the ACTUAL solution order (see assemble_bilinear).
        let elem_type = mesh.element_type(0);
        let owned_default = ref_elem_vol_for_space(space, elem_type, space.element_order(0));
        let quad = owned_default.quadrature(quad_order);

        let dofs_per_elem = owned_default.n_dofs();
        let est_nnz = n_elems as usize * dofs_per_elem * dofs_per_elem;

        let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
        coo.reserve(est_nnz.min(10_000_000));
        let mut scratch = ElementScratch::new();

        let mut all_dofs: Vec<u32> = Vec::with_capacity(n_elems * dofs_per_elem);
        let mut all_mats: Vec<f64> = Vec::with_capacity(n_elems * dofs_per_elem * dofs_per_elem);
        let mut ldofs = 0;

        for e in mesh.elem_iter() {
            accumulate_volume_bilinear_element(space, e, integrators, &quad, quad_order, &mut coo, &mut scratch, &*owned_default);
            let gd = &scratch.global_dofs;
            ldofs = gd.len();
            all_dofs.extend(gd.iter().map(|&d| d as u32));
            all_mats.extend_from_slice(&scratch.k_elem);
        }
        (coo.into_csr(), all_dofs, all_mats, ldofs, n_elems)
    }

    /// Assemble the global load vector for a linear form.
    pub fn assemble_linear<S: FESpace>(
        space:       &S,
        integrators: &[&dyn LinearIntegrator],
        quad_order:  u8,
    ) -> Vec<f64> {
        let n_dofs = space.n_dofs();

        // MFEM semantics: each integrator may select its own quadrature order
        // (see `assemble_bilinear`).
        let space_order = space.element_order(0);
        if integrators.iter().any(|i| i.integration_order(space_order).is_some()) {
            let mut acc = vec![0.0_f64; n_dofs];
            for integ in integrators {
                let qo = integ.integration_order(space_order).unwrap_or(quad_order);
                let v = Self::assemble_linear_inner(space, &[*integ], qo, None);
                for i in 0..n_dofs { acc[i] += v[i]; }
            }
            return acc;
        }

        Self::assemble_linear_inner(space, integrators, quad_order, None)
    }

    /// Assemble a linear form where each integrator is applied on a subset of
    /// elements selected by an element-attribute marker (MFEM
    /// `LinearForm::AddDomainIntegrator(integ, marker)`).
    ///
    /// Marker layout is the same as [`Self::assemble_bilinear_marked`]:
    /// length `max_attr` with `marker[attr-1] = 1` selecting attribute `attr`.
    pub fn assemble_linear_marked<S: FESpace>(
        space:       &S,
        integrators: &[(&dyn LinearIntegrator, Option<&[i32]>)],
        quad_order:  u8,
    ) -> Vec<f64> {
        let n_dofs = space.n_dofs();
        let mut acc = vec![0.0_f64; n_dofs];
        for (integ, marker) in integrators {
            let v = Self::assemble_linear_inner(space, &[*integ], quad_order, *marker);
            for i in 0..n_dofs { acc[i] += v[i]; }
        }
        acc
    }

    /// Core linear-form assembly loop shared by [`Self::assemble_linear`].
    fn assemble_linear_inner<S: FESpace>(
        space:       &S,
        integrators: &[&dyn LinearIntegrator],
        quad_order:  u8,
        elem_marker: Option<&[i32]>,
    ) -> Vec<f64> {
        let mesh   = space.mesh();
        let n_dofs = space.n_dofs();

        // Precompute quadrature rule from the ACTUAL solution order (see assemble_bilinear).
        let elem_type = mesh.element_type(0);
        let quad = ref_elem_vol_for_space(space, elem_type, space.element_order(0)).quadrature(quad_order);

        #[cfg(feature = "parallel")]
        {
            if elem_marker.is_none() && mesh.n_elements() >= assembly_parallel_min_elems() {
                return assemble_linear_volume_parallel(space, integrators, &quad, quad_order);
            }
        }

        let mut rhs = vec![0.0_f64; n_dofs];
        let mut scratch = ElementScratch::new();
        for e in mesh.elem_iter() {
            if let Some(marker) = elem_marker {
                let tag = mesh.element_tag(e);
                if tag <= 0 || (tag as usize) > marker.len() || marker[(tag - 1) as usize] == 0 {
                    continue;
                }
            }
            accumulate_volume_linear_element(space, e, integrators, &quad, quad_order, &mut rhs, &mut scratch);
        }
        rhs
    }

    // ── White Gaussian noise right-hand side ────────────────────────────────

    /// Element mass matrix in MFEM `DenseMatrix` column-major layout, as
    /// consumed by `WhiteGaussianNoiseDomainLFIntegrator`. Weights follow
    /// MFEM `MassIntegrator` (`Trans.Weight()` = |det J|); the element matrix
    /// is produced by the standard [`MassIntegrator`] so it is bit-identical
    /// to the global mass assembly.
    pub(crate) fn mass_element_matrix<S: FESpace>(space: &S, e: u32) -> Vec<f64> {
        use crate::standard::MassIntegrator;

        let mesh  = space.mesh();
        let edim  = mesh.dim() as usize;
        let tdim  = mesh.topological_dim() as usize;
        if edim != tdim {
            panic!("mass_element_matrix: surface elements not supported (white noise uses volume elements)");
        }
        let dim    = edim;
        let order  = space.element_order(e);
        let elem_type = mesh.element_type(e);
        let ref_elem  = ref_elem_vol_for_space(space, elem_type, order);
        let n_ldofs   = ref_elem.n_dofs();
        // MFEM MassIntegrator default rule: IntRules.Get(geom, 2p + OrderW()).
        // OrderW() = 0 for the straight-sided geometries supported here.
        let quad = ref_elem.quadrature(2 * order);

        let n = n_ldofs;
        let mut k_elem = vec![0.0_f64; n * n];
        let mass = MassIntegrator { rho: 1.0 };
        let mut phi = vec![0.0_f64; n_ldofs];

        let elem_tag = mesh.element_tag(e);
        let g_order = mesh.geom_order();
        let affine = is_affine(elem_type, g_order);
        let geo_elem = geo_ref_elem(mesh, e);

        let affine_tr = if affine {
            Some(simplex_transformation(mesh, e))
        } else {
            None
        };

        for (q, xi) in quad.points.iter().enumerate() {
            let (w_phys, xp);
            if affine {
                let tr = affine_tr.as_ref().unwrap();
                w_phys = quad.weights[q] * tr.det_j().abs();
                ref_elem.eval_basis(xi, &mut phi);
                xp = tr.map_to_physical(xi);
            } else {
                let geo = geo_elem.as_ref().unwrap();
                let geo_nds = mesh.geometry_nodes(e);
                let xi_g = geom_quad_point(elem_type, order, xi);
                let (_jac, det, xp_qp) = isoparametric_jacobian(
                    mesh, geo_nds, geo.as_ref(), &xi_g, dim,
                    mesh.geom_order() > 1, // curved → L2_T1 lexicographic order
                );
                w_phys = quad.weights[q] * det.abs();
                ref_elem.eval_basis(xi, &mut phi);
                xp = xp_qp;
            }
            let qp = QpData {
                n_dofs:    n,
                dim,
                weight:    w_phys,
                phys_weight: w_phys,
                ref_weight: quad.weights[q],
                phi:       &phi,
                grad_phys: &[],
                x_phys:    &xp,
                elem_id:   e,
                elem_tag,
                elem_dofs: None,
            };
            mass.add_to_element_matrix(&qp, &mut k_elem);
        }

        // Row-major k_elem -> column-major MFEM layout: data[i + j*n] = (i,j).
        let mut colmajor = vec![0.0_f64; n * n];
        for j in 0..n {
            for i in 0..n {
                colmajor[i + j * n] = k_elem[i * n + j];
            }
        }
        colmajor
    }

    /// Assemble a white Gaussian noise right-hand side `b` with
    /// `E[b bᵀ] = M` (the global mass matrix), MFEM
    /// `WhiteGaussianNoiseDomainLFIntegrator` + `LinearForm::Assemble` 1:1.
    ///
    /// Elements are traversed in mesh order; per element the integrator draws
    /// `n` normals from its seeded RNG chain and multiplies by the Cholesky
    /// factor of the element mass matrix.
    pub fn assemble_white_gaussian_noise<S: FESpace>(
        space: &S,
        integ: &mut crate::standard::WhiteGaussianNoiseDomainLFIntegrator,
    ) -> Vec<f64> {
        let mesh = space.mesh();
        let n_dofs = space.n_dofs();
        let mut rhs = vec![0.0_f64; n_dofs];
        for e in mesh.elem_iter() {
            let order = space.element_order(e);
            let n = ref_elem_vol_for_space(space, mesh.element_type(e), order).n_dofs();
            let raw_dofs: Vec<DofId> = space.element_dofs(e).to_vec();
            debug_assert_eq!(raw_dofs.len(), n, "constrained elements are not supported");
            let m_e = Self::mass_element_matrix(space, e);
            let elvect = integ.assemble_element_vector(&m_e, n);
            for (k, &d) in raw_dofs.iter().enumerate() {
                rhs[d as usize] += elvect[k];
            }
        }
        rhs
    }

    // ── Boundary linear form ──────────────────────────────────────────────────

    /// Assemble boundary contributions (e.g. Neumann BCs) into a load vector.
    ///
    /// # Arguments
    /// * `n_dofs`      — total number of global DOFs.
    /// * `mesh`        — mesh topology.
    /// * `face_dofs`   — closure: `face_id → &[global_dof_id]` for each boundary face.
    /// * `integrators` — boundary linear integrators to accumulate.
    /// * `tags`        — only process boundary faces whose tag is in this list.
    /// * `quad_order`  — quadrature accuracy order.
    ///
    /// The closure `face_dofs` lets you pass either a P1 or P2 DOF list depending
    /// on your space (see [`face_dofs_p1`] and [`face_dofs_p2`] helpers).
    pub fn assemble_boundary_linear(
        n_dofs:      usize,
        mesh:        &(dyn MeshTopology + Sync),
        face_dofs:   &(dyn Fn(u32) -> Vec<DofId> + Sync),
        order:       u8,
        integrators: &[&dyn BoundaryLinearIntegrator],
        tags:        &[i32],
        quad_order:  u8,
    ) -> Vec<f64> {
        let face_ids: Vec<u32> = mesh
            .face_iter()
            .filter(|&f| tags.contains(&mesh.face_tag(f)))
            .collect();

        #[cfg(feature = "parallel")]
        {
            if face_ids.len() >= assembly_parallel_min_elems() {
                return assemble_boundary_linear_parallel(
                    n_dofs,
                    mesh,
                    &face_ids,
                    face_dofs,
                    order,
                    integrators,
                    quad_order,
                );
            }
        }

        let mut rhs = vec![0.0_f64; n_dofs];
        for f in face_ids {
            accumulate_boundary_linear_face(mesh, f, face_dofs, order, integrators, quad_order, &mut rhs);
        }
        rhs
    }

    // ── Boundary bilinear form ───────────────────────────────────────────────

    /// Assemble a boundary bilinear form (e.g. boundary mass ∫_Γ α u v ds).
    ///
    /// # Arguments
    /// * `n_dofs`      — total number of global DOFs.
    /// * `mesh`        — mesh topology.
    /// * `face_dofs`   — closure: `face_id → &[global_dof_id]` for each boundary face.
    /// * `order`       — polynomial order of the face reference element.
    /// * `integrators` — boundary bilinear integrators to accumulate.
    /// * `tags`        — only process boundary faces whose tag is in this list.
    /// * `quad_order`  — quadrature accuracy order.
    pub fn assemble_boundary_bilinear(
        n_dofs:      usize,
        mesh:        &(dyn MeshTopology + Sync),
        face_dofs:   &(dyn Fn(u32) -> Vec<DofId> + Sync),
        order:       u8,
        integrators: &[&dyn BoundaryBilinearIntegrator],
        tags:        &[i32],
        quad_order:  u8,
    ) -> CsrMatrix<f64> {
        let face_ids: Vec<u32> = mesh
            .face_iter()
            .filter(|&f| tags.contains(&mesh.face_tag(f)))
            .collect();

        #[cfg(feature = "parallel")]
        {
            if face_ids.len() >= assembly_parallel_min_elems() {
                return assemble_boundary_bilinear_parallel(
                    n_dofs,
                    mesh,
                    &face_ids,
                    face_dofs,
                    order,
                    integrators,
                    quad_order,
                );
            }
        }

        let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
        for f in face_ids {
            accumulate_boundary_bilinear_face(mesh, f, face_dofs, order, integrators, quad_order, &mut coo);
        }
        coo.into_csr()
    }

    /// Assemble a bilinear form stiffness matrix on the GPU via wgpu.
    ///
    /// Supports:
    /// - `DiffusionIntegrator` on P1 Tri3, P2 Tri6, Q1 Quad4 (2D)
    /// - `DiffusionIntegrator` on P1 Tet4 (3D)
    ///   Requires the `gpu` feature.
    #[cfg(feature = "gpu")]
    pub fn assemble_bilinear_gpu<S: FESpace>(
        space: &S,
        integrators: &[&dyn BilinearIntegrator],
    ) -> Result<CsrMatrix<f64>, String> {
        Self::assemble_bilinear_gpu_impl(space, integrators, "diffusion")
    }

    /// Like [`assemble_bilinear_gpu`] but with explicit operator kind selection.
    ///
    /// `kind` must be one of `"diffusion"`, `"mass"`, or `"elasticity"`.
    /// For elasticity the parameters are taken from the integrator (lambda, mu).
    #[cfg(feature = "gpu")]
    pub fn assemble_bilinear_gpu_with_kind<S: FESpace>(
        space: &S,
        integrators: &[&dyn BilinearIntegrator],
        kind: &str,
    ) -> Result<CsrMatrix<f64>, String> {
        Self::assemble_bilinear_gpu_impl(space, integrators, kind)
    }

    #[cfg(feature = "gpu")]
    fn assemble_bilinear_gpu_impl<S: FESpace>(
        space: &S,
        integrators: &[&dyn BilinearIntegrator],
        kind: &str,
    ) -> Result<CsrMatrix<f64>, String> {
        use fem_linalg_gpu::GpuContext;
        use fem_mesh::element_type::ElementType;

        if integrators.len() != 1 { return Err("GPU assembly requires exactly 1 integrator".into()); }

        let mesh = space.mesh();
        let dim = mesh.dim();
        let etype = mesh.element_type(0);
        let order = space.order();

        // Determine element type and dispatch
        let assemble_fn: Box<dyn Fn(&GpuContext, &[f32], &[u32], usize) -> Vec<(u32, u32, f32)>> =
        match (kind, dim, &etype, order) {
            // Diffusion
            ("diffusion", 2, ElementType::Tri3, 1)  => Box::new(|g,n,d,ne| fem_linalg_gpu::assemble_poisson_2d_p1(g,n,d,ne)),
            ("diffusion", 2, ElementType::Tri6, 2)  => Box::new(|g,n,d,ne| fem_linalg_gpu::assemble_poisson_2d_p2(g,n,d,ne)),
            ("diffusion", 2, ElementType::Quad4, 1) => Box::new(|g,n,d,ne| fem_linalg_gpu::assemble_poisson_2d_q1(g,n,d,ne)),
            ("diffusion", 3, ElementType::Tet4 | ElementType::Tet10, 1) => Box::new(|g,n,d,ne| fem_linalg_gpu::assemble_poisson_3d_p1(g,n,d,ne)),
            ("diffusion", 3, ElementType::Hex8 | ElementType::Hex20, 1) => Box::new(|g,n,d,ne| fem_linalg_gpu::assemble_poisson_3d_hex8(g,n,d,ne)),
            // Mass
            ("mass", 2, ElementType::Tri3, 1)  => Box::new(|g,n,d,ne| fem_linalg_gpu::assemble_mass_2d_tri3(g,n,d,ne)),
            ("mass", 2, ElementType::Quad4, 1) => Box::new(|g,n,d,ne| fem_linalg_gpu::assemble_mass_2d_quad4(g,n,d,ne)),
            ("mass", 3, ElementType::Tet4 | ElementType::Tet10, 1) => Box::new(|g,n,d,ne| fem_linalg_gpu::assemble_mass_3d_tet4(g,n,d,ne)),
            ("mass", 3, ElementType::Hex8 | ElementType::Hex20, 1) => Box::new(|g,n,d,ne| fem_linalg_gpu::assemble_mass_3d_hex8(g,n,d,ne)),
            // Elasticity (default lame parameters; call with integrator for actual values)
            ("elasticity", 2, ElementType::Tri3, 1) => Box::new(|g,n,d,ne| fem_linalg_gpu::assemble_elasticity_2d_tri3(g,n,d,ne,1.0,1.0)),
            ("elasticity", 3, ElementType::Tet4 | ElementType::Tet10, 1) => Box::new(|g,n,d,ne| fem_linalg_gpu::assemble_elasticity_3d_tet4(g,n,d,ne,1.0,1.0)),
            ("elasticity", 3, ElementType::Hex8 | ElementType::Hex20, 1) => Box::new(|g,n,d,ne| fem_linalg_gpu::assemble_elasticity_3d_hex8(g,n,d,ne,1.0,1.0)),
            _ => return Err(format!(
                "GPU assembly: unsupported (kind={kind}, dim={dim}, type={etype:?}, order={order})"
            )),
        };

        let npe = match &etype {
            ElementType::Tri3 => 3,
            ElementType::Tri6 => 6,
            ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => 4,
            ElementType::Tet4 | ElementType::Tet10 => 4,
            ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => 8,
            ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => 6,
            ElementType::Pyramid5 | ElementType::Pyramid13 => 5,
            _ => return Err(format!("GPU assembly: unsupported element type {etype:?}")),
        };
        let npe_coords = npe * dim as usize;
        let dofs_per_elem = npe;

        let n_elem = mesh.n_elements();
        let n_dofs = space.n_dofs();

        let mut elem_nodes_f32 = Vec::with_capacity(n_elem * npe_coords);
        let mut elem_dofs_u32 = Vec::with_capacity(n_elem * dofs_per_elem);

        for e in 0..n_elem as u32 {
            let nodes = mesh.element_nodes(e);
            let dofs: Vec<u32> = space.element_dofs(e).to_vec();
            for kn in 0..npe {
                let c = mesh.node_coords(nodes[kn]);
            for d in 0..dim as usize {
                    elem_nodes_f32.push(c[d] as f32);
                }
            }
            elem_dofs_u32.extend_from_slice(&dofs[..dofs_per_elem]);
        }

        let gpu = GpuContext::new_sync()
            .map_err(|e| format!("GpuContext init: {e}"))?;

        let coo_triplets =
            assemble_fn(&gpu, &elem_nodes_f32, &elem_dofs_u32, n_elem);

        let mut coo = fem_linalg::CooMatrix::new(n_dofs, n_dofs);
        for (r, c, v) in coo_triplets {
            coo.add(r as usize, c as usize, v as f64);
        }
        Ok(coo.into_csr())
    }
}

// ─── Face Jacobian and normal (2-D edges and 3-D faces) ──────────────────────

/// Order-1 reference element of a boundary face's **geometry**: the segment
/// `[0,1]`, the unit triangle, or the unit square `[0,1]²`.
///
/// Its vertex order is the corner order of [`MeshTopology::face_nodes`], so
/// interpolating the face's corner coordinates with it reproduces MFEM's
/// boundary transformation (`Mesh::GetBdrElementTransformation`, whose
/// boundary element is the linear/bilinear map of the face corners on a
/// straight mesh).  The same element also transports the *face's* reference
/// coordinates into the owning volume element's reference element, which is
/// how [`face_dofs_h1`] locates a face DOF inside its owner.
pub(crate) fn face_geo_elem(face_type: ElementType) -> Box<dyn ReferenceElement> {
    match face_type {
        ElementType::Line2 => Box::new(SegP1),
        ElementType::Tri3 => Box::new(TriP1),
        // MFEM's boundary element for a hexahedron face is
        // `H1_QuadrilateralElement`, which lives on `[0,1]²`.  `QuadQ1`/`QuadQ2`
        // live on `[-1,1]²` and would evaluate the face geometry outside its
        // reference domain.
        ElementType::Quad4 => Box::new(fem_element::lagrange::factory::QuadQk::new(1)),
        _ => panic!("face_geo_elem: unsupported boundary face type {face_type:?}"),
    }
}

/// Geometry of one boundary face: the physical coordinates of its corners, in
/// the corner order of [`MeshTopology::face_nodes`] (the vertex order of
/// [`ref_elem_face`] and of [`face_geo_elem`]).
struct FaceGeom {
    /// Face geometry element: order 1 ([`face_geo_elem`]) on a straight face,
    /// order `geom_order` on a curved one (D59/D66).
    geo: Box<dyn ReferenceElement>,
    /// Physical control-point coordinates; the order-1 count (2 for a segment,
    /// 3 for a triangle, 4 for a quad) on a straight face, the geometry order's
    /// dof count on a curved one.
    pts: Vec<[f64; 3]>,
    /// Physical dimension of the embedding space (2 or 3).
    dim: usize,
}

impl FaceGeom {
    /// Evaluate the face at the **face** reference point `xi`:
    /// `(|J_face|, n, x_phys)`.
    ///
    /// * 2-D (a boundary edge): `t0 = ∂x/∂ξ` is the edge tangent,
    ///   `|J_face| = |t0|`, and the outward unit normal is `t0` rotated by −90°
    ///   (`(t0_y, −t0_x)/|t0|`) — MFEM's `CalcOrtho` for a 2×1 Jacobian.
    /// * 3-D (a boundary face): `t0 = ∂x/∂ξ₀` and `t1 = ∂x/∂ξ₁` are the two
    ///   face tangents, the *unnormalised* face normal is `t0 × t1`,
    ///   `|J_face| = |t0 × t1|` and the unit outward normal is it divided by
    ///   its length — MFEM's `CalcOrtho` for a 3×2 Jacobian.  Its orientation
    ///   follows from the corner order, which puts the element interior on the
    ///   left of `ξ₀` and below `ξ₁`.
    fn eval(&self, xi: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        let n = self.pts.len();
        let dim = self.dim;
        // Reference dimension of the face: 1 for a segment, 2 for a tri/quad.
        // It is the *stride* of `eval_grad_basis` (one gradient component per
        // reference coordinate), which is not `dim` for a 2-D boundary edge.
        let rdim = self.geo.dim() as usize;
        let mut phi = vec![0.0_f64; n];
        let mut grad = vec![0.0_f64; n * rdim];
        self.geo.eval_basis(xi, &mut phi);
        self.geo.eval_grad_basis(xi, &mut grad);

        let mut x = vec![0.0_f64; dim];
        let mut t0 = [0.0_f64; 3];
        let mut t1 = [0.0_f64; 3];
        for k in 0..n {
            let g0 = grad[k * rdim];
            let g1 = if rdim > 1 { grad[k * rdim + 1] } else { 0.0 };
            for i in 0..dim {
                x[i] += phi[k] * self.pts[k][i];
                t0[i] += self.pts[k][i] * g0;
                t1[i] += self.pts[k][i] * g1;
            }
        }

        if dim == 2 {
            // Historical *chord* interpolation of the physical point.  It
            // coincides with the interpolated `Σᵏ φᵏ pᵏ` for a straight edge but
            // rounds differently in the last ulp, so it is kept for order-1
            // geometry to leave every existing 2-D caller bit-identical.  A
            // curved edge (D66, `SegPk(q >= 2)` geometry) must use the
            // interpolated point — the chord lies off the curve.
            if self.geo.order() == 1 {
                let (p0, p1) = (self.pts[0], self.pts[1]);
                for i in 0..2 {
                    x[i] = p0[i] + (p1[i] - p0[i]) * xi[0];
                }
            }
            let j = (t0[0] * t0[0] + t0[1] * t0[1]).sqrt();
            (j, vec![t0[1] / j, -t0[0] / j], x)
        } else {
            let c = [
                t0[1] * t1[2] - t0[2] * t1[1],
                t0[2] * t1[0] - t0[0] * t1[2],
                t0[0] * t1[1] - t0[1] * t1[0],
            ];
            let j = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
            (j, vec![c[0] / j, c[1] / j, c[2] / j], x)
        }
    }
}

/// Build the [`FaceGeom`] of boundary face `f` (2-D edge or 3-D face).
///
/// The corner coordinates come from the mesh's per-element geometry when
/// present — [`MeshTopology::boundary_face_endpoints`] in 2-D, the owner's
/// order-1 geometry slots in 3-D — so a face spanning a geometrically periodic
/// seam is measured in its unfolded position instead of the folded vertex
/// chord (`geom_coords_of` resolves the snapshot).  Otherwise the folded vertex
/// table is used, which is bit-identical to it on every non-periodic mesh.
///
/// D59: on a 3-D mesh with **curved** geometry (`geom_order() >= 2`) the face
/// geometry is the boundary element's own curved mapping instead of the
/// corner-only bilinear one: the face's high-order geometry nodes (the owner
/// element's geometry slots that lie on the face, resolved by reference
/// position exactly like [`face_dofs_h1`]) are interpolated with the
/// order-`geom_order` face element — MFEM's `GetBdrElementTransformation`,
/// which transports the *volume* `Nodes` values of the face through the
/// boundary element of the same order.
///
/// D66: the same rule now applies in 2-D (see [`curved_boundary_edge_geom`]).
/// Before it, a curved 2-D boundary edge was measured as the affine chord of
/// its two vertices — the 2-D counterpart of D59.
fn boundary_face_geom(
    mesh: &dyn MeshTopology,
    f: u32,
    face_nodes: &[u32],
    dim: usize,
) -> FaceGeom {
    let face_type = match face_nodes.len() {
        2 => ElementType::Line2,
        3 => ElementType::Tri3,
        4 => ElementType::Quad4,
        n => panic!("boundary_face_geom: unsupported boundary face with {n} nodes"),
    };
    assert!(
        dim == 2 || dim == 3,
        "boundary_face_geom: unsupported embedding dimension {dim}"
    );

    // D66: a curved 2-D mesh must measure its boundary edges through the
    // boundary element's own order-`geom_order` mapping, not the corner chord.
    if dim == 2 && mesh.geom_order() > 1 {
        return curved_boundary_edge_geom(mesh, f, face_nodes, dim);
    }

    let pts = if dim == 2 {
        let (p0, p1) = match mesh.boundary_face_endpoints(f) {
            Some(e) => e,
            None => {
                let c0 = mesh.node_coords(face_nodes[0]);
                let c1 = mesh.node_coords(face_nodes[1]);
                ([c0[0], c0[1]], [c1[0], c1[1]])
            }
        };
        vec![[p0[0], p0[1], 0.0], [p1[0], p1[1], 0.0]]
    } else if mesh.geom_order() > 1 {
        return curved_boundary_face_geom(mesh, f, face_nodes, face_type, dim);
    } else {
        // 3-D: read the corners through the owner element so `geom_coords_of`
        // can resolve a per-element (unfolded) geometry snapshot.
        let owner = face_owner(mesh, face_nodes);
        let local = owner.map(|e| mesh.element_nodes(e));
        let mut out = Vec::with_capacity(face_nodes.len());
        for &n in face_nodes {
            // Only an *order-1* geometry table has one slot per mesh vertex;
            // higher-order tables index reference slots, not vertices, so they
            // must not be indexed by the vertex position.
            let c = match local
                .and_then(|en| en.iter().position(|&x| x == n))
                .filter(|_| owner.is_some() && mesh.geom_order() == 1)
            {
                Some(pos) => {
                    let gn = mesh.geometry_nodes(owner.expect("owner present"));
                    mesh.geom_coords_of(gn[pos])
                }
                None => mesh.node_coords(n),
            };
            out.push([c[0], c[1], *c.get(2).unwrap_or(&0.0)]);
        }
        out
    };

    assert_eq!(
        pts.len(),
        face_nodes.len(),
        "boundary_face_geom: corner count mismatch"
    );
    FaceGeom {
        geo: face_geo_elem(face_type),
        pts,
        dim,
    }
}

/// D59: [`FaceGeom`] of a 3-D boundary face on a **curved** mesh
/// (`geom_order() >= 2`).
///
/// The face geometry element is the order-`q` trace of the volume geometry
/// family (`QuadQk(q)` for a hex face, the `H1_TriangleElement` rule resolved
/// against the volume element for a tet face — the same element
/// [`ref_elem_face`] uses for the *space*, here applied at the geometry
/// order), and its control points are the owner element's **geometry nodes**
/// that lie on the face.  Each face dof is transported into the volume
/// reference element through the straight-faced corner map and resolved to the
/// nearest geometry slot by position — the identical mechanism
/// [`face_dofs_h1`] uses to pair space dofs, so the geometry and the space see
/// the same boundary transformation.
fn curved_boundary_face_geom(
    mesh: &dyn MeshTopology,
    f: u32,
    face_nodes: &[u32],
    face_type: ElementType,
    dim: usize,
) -> FaceGeom {
    use fem_element::lagrange::factory::{HexQk, QuadQk};

    let q = mesh.geom_order() as usize;
    let owner = face_owner(mesh, face_nodes).unwrap_or_else(|| {
        panic!("curved_boundary_face_geom: no element contains every node of face {f}")
    });
    let elem_type = mesh.element_type(owner);

    // The owner's geometry element: the same factory `set_curvature` used to
    // lay out the geometry node list.
    let geom_vol: Box<dyn ReferenceElement> = match elem_type {
        ElementType::Hex8 => Box::new(HexQk::new(q)),
        ElementType::Tet4 => Box::new(TetPk::new(q)),
        other => panic!(
            "curved_boundary_face_geom: unsupported curved owner element {other:?} (face {f})"
        ),
    };
    let geom_vol_coords = geom_vol.dof_coords();
    let gn = mesh.geometry_nodes(owner);
    assert_eq!(
        gn.len(),
        geom_vol_coords.len(),
        "curved_boundary_face_geom: element {owner} has {} geometry nodes but the \
         order-{q} {} geometry element has {}",
        gn.len(),
        match elem_type {
            ElementType::Hex8 => "HexQk",
            _ => "TetPk",
        },
        geom_vol_coords.len(),
    );

    // The face's corners as coordinates of the volume reference element, in
    // the face's corner order (corner dofs sit first, in vertex order).
    let elem_nodes = mesh.element_nodes(owner);
    let mut corner_ref: Vec<&[f64]> = Vec::with_capacity(face_nodes.len());
    for &n in face_nodes {
        let pos = elem_nodes.iter().position(|&en| en == n).unwrap_or_else(|| {
            panic!("curved_boundary_face_geom: node {n} of face {f} is not in element {owner}")
        });
        corner_ref.push(&geom_vol_coords[pos]);
    }

    // Face geometry element of the same order — MFEM's boundary element.
    let (face_geo, face_coords): (Box<dyn ReferenceElement>, Vec<Vec<f64>>) =
        match face_type {
            ElementType::Tri3 => {
                let e = H1TetFacePk::new(q);
                let c = e.dof_coords();
                (Box::new(e), c)
            }
            ElementType::Quad4 => {
                let e = QuadQk::new(q);
                let c = e.dof_coords();
                (Box::new(e), c)
            }
            other => panic!(
                "curved_boundary_face_geom: unsupported boundary face type {other:?} (face {f})"
            ),
        };

    // Control points: transport every face dof position into the volume
    // reference element through the straight corner map and resolve it to the
    // nearest geometry slot (distinct slots are >= 1e-2 apart, the map rounds
    // at <= 1e-15 — the face_dofs_h1 tolerances).
    let transport = face_geo_elem(face_type);
    let mut phi = vec![0.0_f64; face_nodes.len()];
    let mut pts = Vec::with_capacity(face_coords.len());
    for fc in face_coords.iter() {
        transport.eval_basis(fc, &mut phi);
        let mut x = [0.0_f64; 3];
        for (k, c) in corner_ref.iter().enumerate() {
            for i in 0..3 {
                x[i] += phi[k] * (*c.get(i).unwrap_or(&0.0));
            }
        }
        let mut best = 0usize;
        let mut best_d2 = f64::INFINITY;
        for (k, c) in geom_vol_coords.iter().enumerate() {
            let d2: f64 = (0..3).map(|i| (c.get(i).copied().unwrap_or(0.0) - x[i]).powi(2)).sum();
            if d2 < best_d2 {
                best_d2 = d2;
                best = k;
            }
        }
        assert!(
            best_d2 < 1e-20,
            "curved_boundary_face_geom: no geometry dof of element {owner} at {x:?} \
             (face {f} dof {fc:?}, nearest distance {:.3e})",
            best_d2.sqrt()
        );
        let c = mesh.geom_coords_of(gn[best]);
        pts.push([c[0], c[1], *c.get(2).unwrap_or(&0.0)]);
    }

    FaceGeom {
        geo: face_geo,
        pts,
        dim,
    }
}

/// D66: [`FaceGeom`] of a 2-D boundary edge on a **curved** mesh
/// (`geom_order() >= 2`) — the 2-D counterpart of
/// [`curved_boundary_face_geom`], and MFEM's `GetBdrElementTransformation` in
/// 2-D.
///
/// The edge geometry is the boundary *element*'s own order-`q` mapping: the
/// trace of the volume geometry family on the edge, i.e. `SegPk(q)`
/// (`H1_SegmentElement(q)`), whose control points are the owner element's
/// geometry nodes that lie on the edge.  Each edge dof is transported into the
/// volume reference element through the straight corner map and resolved to the
/// nearest geometry slot by position — the identical mechanism
/// [`curved_boundary_face_geom`] uses in 3-D, so the geometry and the space see
/// the same boundary transformation.
///
/// Without it the edge was collapsed to the affine chord of its two vertices
/// ([`MeshTopology::boundary_face_endpoints`]), so `∫ u·n ds` on a curved edge
/// used the chord length and the chord tangent while the volume assembly used
/// the curved mapping.
fn curved_boundary_edge_geom(
    mesh: &dyn MeshTopology,
    f: u32,
    face_nodes: &[u32],
    dim: usize,
) -> FaceGeom {
    use fem_element::lagrange::factory::{QuadQk, SegPk};

    let q = mesh.geom_order() as usize;
    let owner = face_owner(mesh, face_nodes).unwrap_or_else(|| {
        panic!("curved_boundary_edge_geom: no element contains every node of edge {f}")
    });
    let elem_type = mesh.element_type(owner);

    // The owner's geometry element: the same factory `set_curvature` used to
    // lay out the geometry node list.
    let geom_vol: Box<dyn ReferenceElement> = match elem_type {
        ElementType::Quad4 => Box::new(QuadQk::new(q)),
        ElementType::Tri3 => Box::new(TriPk::new(q)),
        other => panic!(
            "curved_boundary_edge_geom: unsupported curved owner element {other:?} (edge {f})"
        ),
    };
    let geom_vol_coords = geom_vol.dof_coords();
    let gn = mesh.geometry_nodes(owner);
    assert_eq!(
        gn.len(),
        geom_vol_coords.len(),
        "curved_boundary_edge_geom: element {owner} has {} geometry nodes but the \
         order-{q} {} geometry element has {}",
        gn.len(),
        match elem_type {
            ElementType::Quad4 => "QuadQk",
            _ => "TriPk",
        },
        geom_vol_coords.len(),
    );

    // The edge's corners as coordinates of the volume reference element, in the
    // edge's corner order (corner dofs sit first, in vertex order).
    let elem_nodes = mesh.element_nodes(owner);
    let mut corner_ref: Vec<&[f64]> = Vec::with_capacity(face_nodes.len());
    for &n in face_nodes {
        let pos = elem_nodes.iter().position(|&en| en == n).unwrap_or_else(|| {
            panic!("curved_boundary_edge_geom: node {n} of edge {f} is not in element {owner}")
        });
        corner_ref.push(&geom_vol_coords[pos]);
    }

    // Edge geometry element of the same order — MFEM's boundary element
    // (`H1_SegmentElement(q)`), on `[0,1]` like `face_geo_elem(Line2)`.
    let face_geo = SegPk::new(q);
    let face_coords = face_geo.dof_coords();

    // Control points: transport every edge dof position into the volume
    // reference element through the straight corner map and resolve it to the
    // nearest geometry slot (distinct slots are >= 1e-2 apart, the map rounds
    // at <= 1e-15 — the same tolerances as the 3-D path).
    let transport = face_geo_elem(ElementType::Line2);
    let mut phi = vec![0.0_f64; face_nodes.len()];
    let mut pts = Vec::with_capacity(face_coords.len());
    for fc in face_coords.iter() {
        transport.eval_basis(fc, &mut phi);
        let mut x = [0.0_f64; 2];
        for (k, c) in corner_ref.iter().enumerate() {
            x[0] += phi[k] * c[0];
            x[1] += phi[k] * c.get(1).copied().unwrap_or(0.0);
        }
        let mut best = 0usize;
        let mut best_d2 = f64::INFINITY;
        for (k, c) in geom_vol_coords.iter().enumerate() {
            let dx = c[0] - x[0];
            let dy = c.get(1).copied().unwrap_or(0.0) - x[1];
            let d2 = dx * dx + dy * dy;
            if d2 < best_d2 {
                best_d2 = d2;
                best = k;
            }
        }
        assert!(
            best_d2 < 1e-20,
            "curved_boundary_edge_geom: no geometry dof of element {owner} at {x:?} \
             (edge {f} dof {fc:?}, nearest distance {:.3e})",
            best_d2.sqrt()
        );
        let c = mesh.geom_coords_of(gn[best]);
        pts.push([c[0], c[1], 0.0]);
    }

    FaceGeom {
        geo: Box::new(face_geo),
        pts,
        dim,
    }
}

/// Reference coordinates of the volume element's DOFs, **in the order of
/// `space.element_dofs`**.
///
/// Normally this is just [`ref_elem_vol_for_space`]'s `dof_coords`, but
/// tetrahedra need one correction: the H¹ space's local tet DOF order comes
/// from `fem_space::DofManager`, which documents it as matching
/// `fem_element::lagrange::factory::TetPk`, and the fixed-order
/// `fem_element::lagrange::TetP2` publishes a `dof_coords` list **longer** than
/// its own `n_dofs` (20 coordinates for a 10-DOF element, with the edge DOFs at
/// 1/3 and 2/3 instead of the midpoints).  That list cannot locate a DOF, so
/// tets are always read from the factory element instead.
fn volume_dof_reference_coords<S: FESpace>(
    space: &S,
    elem_type: ElementType,
    order: u8,
) -> Vec<Vec<f64>> {
    if matches!(elem_type, ElementType::Tet4 | ElementType::Tet10) {
        return fem_element::lagrange::factory::TetPk::new(order as usize).dof_coords();
    }
    ref_elem_vol_for_space(space, elem_type, order).dof_coords()
}

/// The volume element that contains every node of boundary face `face_nodes`.
///
/// `MeshTopology::face_elements` is not consulted: it is stale on meshes whose
/// face→element map was never built and on non-conformingly refined meshes, so
/// the node-membership scan (over all elements) is both simpler and uniform with
/// [`face_dofs_h1`]/[`face_dofs_p2`], which already fall back to it.
fn face_owner(mesh: &dyn MeshTopology, face_nodes: &[u32]) -> Option<u32> {
    let contains_all = |e: u32| {
        let en = mesh.element_nodes(e);
        face_nodes.iter().all(|&n| en.contains(&n))
    };
    (0..mesh.n_elements() as u32).find(|&e| contains_all(e))
}

// ─── Scatter helper ───────────────────────────────────────────────────────────

/// Scatter `f_elem` into `rhs` at global DOF indices `dofs`.
#[inline]
fn coo_add_element_vec(dofs: &[usize], f_elem: &[f64], rhs: &mut [f64]) {
    for (&d, &v) in dofs.iter().zip(f_elem.iter()) {
        rhs[d] += v;
    }
}

// ─── Face DOF helpers ─────────────────────────────────────────────────────────

/// Build the face DOF list for a P1 space: face node indices only.
///
/// Use this as the `face_dofs` closure in [`Assembler::assemble_boundary_linear`]
/// for H1/P1 and L2/P0 or P1 spaces.
pub fn face_dofs_p1(mesh: &dyn MeshTopology) -> impl Fn(u32) -> Vec<DofId> + '_ {
    move |f| mesh.face_nodes(f).iter().map(|&n| n as DofId).collect()
}

/// Build the face DOF list for a P2 H1 space.
///
/// For each boundary face `f`, the face DOFs are the two vertex DOFs plus the
/// edge-midpoint DOF shared between them.  The edge-midpoint DOF is found by
/// looking at the element that owns the face and matching the edge in its DOF table.
///
/// # Panics
/// Panics if the face is not owned by any element or if the vertices cannot be
/// matched in the element's DOF table (programming error).
pub fn face_dofs_p2<S>(space: &S) -> impl Fn(u32) -> Vec<DofId> + '_
where
    S: FESpace,
    S::Mesh: MeshTopology,
{
    move |f| {
        let mesh = space.mesh();
        let fn_nodes = mesh.face_nodes(f);
        let nfn = fn_nodes.len();
        if nfn == 0 { return vec![]; }

        // Find the element that actually contains ALL face nodes.
        // In non-conformingly refined meshes face_elements(f) may report a
        // stale owner (or 0 as fallback) whose node list does not include the
        // face nodes, so we scan all elements to find a match.
        let (elem, _) = mesh.face_elements(f);
        let mut owner = elem;
        let elem_nodes = mesh.element_nodes(elem);

        // Quick path: the reported owner contains the face nodes.
        let owner_has_all = fn_nodes.iter().all(|&n| elem_nodes.contains(&n));
        if !owner_has_all {
            // Slow path: find an element that contains all face nodes.
            let n_elems = mesh.n_elements() as u32;
            let mut found = false;
            for e in 0..n_elems {
                let en = mesh.element_nodes(e);
                if fn_nodes.iter().all(|&n| en.contains(&n)) {
                    owner = e;
                    found = true;
                    break;
                }
            }
            if !found {
                // Last resort: fall back to P1 face DOFs on the reported owner.
                return fn_nodes.iter().map(|&n| n as DofId).collect();
            }
        }

        let elem_nodes = mesh.element_nodes(owner);
        let elem_dofs  = space.element_dofs(owner);

        // Find local vertex positions of the face nodes.
        let mut dofs: Vec<DofId> = Vec::with_capacity(nfn + 1);
        for &n in fn_nodes {
            if let Some(pos) = elem_nodes.iter().position(|&en| en == n) {
                dofs.push(elem_dofs[pos]);
            } else {
                // Should not happen after the owner search above.
                return fn_nodes.iter().map(|&n| n as DofId).collect();
            }
        }

        // For TriP2 the edge DOF positions relative to vertex positions are:
        //   edge(v0→v1) = dofs[3],  edge(v1→v2) = dofs[4],  edge(v0→v2) = dofs[5]
        if nfn >= 2 && elem_dofs.len() >= 6 {
            let pos_a = elem_nodes.iter().position(|&n| n == fn_nodes[0]);
            let pos_b = elem_nodes.iter().position(|&n| n == fn_nodes[1]);
            if let (Some(pa), Some(pb)) = (pos_a, pos_b) {
                let edge_dof = find_edge_dof(elem_nodes, elem_dofs, pa, pb);
                dofs.push(edge_dof);
            }
        }

        dofs
    }
}

/// Build the face DOF list for an H¹ space of **any** order, in 2-D and 3-D.
///
/// For each boundary face `f` this returns the space DOFs of the face in the
/// order MFEM's boundary element ([`ref_elem_face`]) lists them:
///
/// * 2-D, a segment face → `[dof(v0), dof(v1), interior…]`, where
///   `v0 = mesh.face_nodes(f)[0]` and `v1 = mesh.face_nodes(f)[1]`, with the
///   interior DOFs in increasing order along `v0 → v1`
///   (`H1_SegmentElement` order);
/// * 3-D, a triangle or quad face → `[vertices…, edge blocks…, interior…]` in
///   the face's own 2-D topology (`H1_TriangleElement` /
///   `H1_QuadrilateralElement` order): the corners in `face_nodes` order, then
///   each face edge's interior DOFs along that edge (edges
///   `(v0,v1), (v1,v2), (v2,v0)` for a triangle, `(v0,v1), (v1,v2), (v2,v3),
///   (v3,v0)` for a quad), then the face-interior DOFs in MFEM's `(j, i)`
///   nested order.
///
/// That is exactly the DOF order of the face reference element
/// [`ref_elem_face`] builds, so `<face_dofs_h1> + assemble_boundary_*`
/// reproduces MFEM's boundary-element assembly.  Use it with
/// [`Assembler::assemble_boundary_linear`] / `assemble_boundary_bilinear`
/// whenever the space order is above 1 — [`face_dofs_p1`] only covers the
/// vertices and [`face_dofs_p2`] only the quadratic 2-D case.
///
/// The face DOFs are located geometrically, not through a per-element-type
/// table: the face's own reference element gives the face coordinate of each of
/// its DOFs, [`face_geo_elem`] transports that coordinate into the owning
/// volume element's reference frame (affine for a segment/triangle face,
/// bilinear for a quad face), and the volume element's DOF at that point is the
/// answer ([`ref_elem_vol_for_space`]).  So this works for any H¹ basis whose
/// `dof_coords` match the space's `element_dofs` ordering.  Note it must be the
/// face's own reference element (and not the volume element restricted to the
/// face): the face element's DOF *order* is what the boundary integrator pairs
/// with, and MFEM's hex/tet face blocks are ordered by the 2-D entity, not by
/// the volume element's slots.
///
/// # Panics
/// Panics if no element contains all of the face's nodes, or if some face DOF
/// has no volume-element DOF at its reference position (which would mean the
/// space's `element_dofs` order disagrees with [`ref_elem_vol_for_space`]).
pub fn face_dofs_h1<S>(space: &S) -> impl Fn(u32) -> Vec<DofId> + '_
where
    S: FESpace,
    S::Mesh: MeshTopology,
{
    move |f| {
        let mesh = space.mesh();
        let fn_nodes = mesh.face_nodes(f);
        let nfn = fn_nodes.len();
        let face_type = match nfn {
            2 => ElementType::Line2,
            3 => ElementType::Tri3,
            4 => ElementType::Quad4,
            n => panic!(
                "face_dofs_h1: boundary face {f} has {n} nodes; only segment (2), \
                 triangle (3) and quad (4) faces are supported"
            ),
        };

        let owner = face_owner(mesh, fn_nodes).unwrap_or_else(|| {
            panic!("face_dofs_h1: no element contains every node of boundary face {f}")
        });

        let elem_nodes = mesh.element_nodes(owner);
        let elem_dofs = space.element_dofs(owner);
        let elem_type = mesh.element_type(owner);
        let vol = ref_elem_vol_for_space(space, elem_type, space.order());
        let vol_coords = volume_dof_reference_coords(space, elem_type, space.order());
        assert_eq!(
            vol_coords.len(),
            elem_dofs.len(),
            "face_dofs_h1: element {owner} is {elem_type:?} of order {} with {} element DOFs, \
             but the volume reference element ({:?} order {}) has {} coordinates",
            space.order(),
            elem_dofs.len(),
            std::any::type_name_of_val(vol.as_ref()),
            vol.order(),
            vol_coords.len(),
        );
        let dim = vol_coords[0].len();

        // The face's corners as coordinates of the volume reference element, in
        // the face's own corner order.
        let mut corners: Vec<&[f64]> = Vec::with_capacity(nfn);
        for &n in fn_nodes {
            let pos = elem_nodes.iter().position(|&en| en == n).unwrap_or_else(|| {
                panic!("face_dofs_h1: node {n} of face {f} is not in element {owner}")
            });
            corners.push(&vol_coords[pos]);
        }

        let face_ref = ref_elem_face(face_type, space.order());
        let geo = face_geo_elem(face_type);
        let face_coords = face_ref.dof_coords();

        let mut phi = vec![0.0_f64; nfn];
        let mut dofs = Vec::with_capacity(face_coords.len());
        for fc in face_coords.iter() {
            geo.eval_basis(fc, &mut phi);
            let mut x = vec![0.0_f64; dim];
            for (k, c) in corners.iter().enumerate() {
                for i in 0..dim {
                    x[i] += phi[k] * c[i];
                }
            }
            // Nearest volume DOF: the transported face coordinate is a face DOF
            // position of the same H¹ family, so the match is exact up to the
            // rounding of the map (≤ 1e-15), while distinct DOFs are at least
            // ~1e-2 apart in the reference element.
            let mut best = 0usize;
            let mut best_d2 = f64::INFINITY;
            for (k, c) in vol_coords.iter().enumerate() {
                let d2: f64 = (0..dim).map(|i| (c[i] - x[i]) * (c[i] - x[i])).sum();
                if d2 < best_d2 {
                    best_d2 = d2;
                    best = k;
                }
            }
            assert!(
                best_d2 < 1e-20,
                "face_dofs_h1: no DOF of element {owner} at {x:?} (face {f} DOF {fc:?}, \
                 nearest distance {:.3e}); does `element_dofs` match ref_elem_vol_for_space?",
                best_d2.sqrt()
            );
            dofs.push(elem_dofs[best]);
        }
        dofs
    }
}

/// Return the edge-midpoint DOF for the edge between local vertex positions `a` and `b`
/// in a TriP2 element (with 6 DOFs: 3 vertex + 3 edge).
///
/// NOTE: For TriP3 (10 DOFs, 2 interior DOFs per edge) this function is not sufficient;
/// Neumann/Robin assembly for P3 requires returning both edge DOFs.
fn find_edge_dof(elem_nodes: &[u32], elem_dofs: &[DofId], pos_a: usize, pos_b: usize) -> DofId {
    let (lo, hi) = if pos_a < pos_b { (pos_a, pos_b) } else { (pos_b, pos_a) };
    // TriP2 edge DOF mapping: (0,1)→3, (1,2)→4, (0,2)→5
    let _ = elem_nodes; // used via pos_a/pos_b
    let edge_local = match (lo, hi) {
        (0, 1) => 3,
        (1, 2) => 4,
        (0, 2) => 5,
        _ => panic!("find_edge_dof: unexpected vertex pair ({lo},{hi}) — only TriP2 supported"),
    };
    elem_dofs[edge_local]
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::{H1Space, fe_space::FESpace};
    use crate::standard::MassIntegrator;
    // Only the boundary-face tests need this: `SegP3` is the equispaced
    // segment element that `ref_elem_face` must *not* use (its nodes are not
    // the volume element's edge nodes).
    use fem_element::lagrange::SegP3;

    /// D46②: the order-generic face segment element is the *trace* of the
    /// volume element, so at order 2 it must reproduce the fixed-order `SegP2`
    /// exactly — and `SegP3` (equispaced nodes) must be **rejected**, because
    /// the volume element's edge nodes are the closed Gauss-Lobatto points.
    #[test]
    fn h1_seg_pk_is_the_volume_trace() {
        fn check_exact(order: u8, fixed: &dyn ReferenceElement) {
            let gen = H1SegPk::new(order);
            assert_eq!(gen.n_dofs(), fixed.n_dofs());
            assert_eq!(gen.dof_coords(), fixed.dof_coords());
            let mut a = vec![0.0_f64; gen.n_dofs()];
            let mut b = vec![0.0_f64; gen.n_dofs()];
            let mut ga = vec![0.0_f64; gen.n_dofs()];
            let mut gb = vec![0.0_f64; gen.n_dofs()];
            for s in [0.0, 0.25, 1.0 / 3.0, 0.5, 0.777, 1.0] {
                gen.eval_basis(&[s], &mut a);
                fixed.eval_basis(&[s], &mut b);
                gen.eval_grad_basis(&[s], &mut ga);
                fixed.eval_grad_basis(&[s], &mut gb);
                for i in 0..a.len() {
                    assert!(
                        (a[i] - b[i]).abs() < 1e-13,
                        "order {order} φ{i}({s}): {} vs {}",
                        a[i],
                        b[i]
                    );
                    assert!(
                        (ga[i] - gb[i]).abs() < 1e-12,
                        "order {order} φ{i}'({s}): {} vs {}",
                        ga[i],
                        gb[i]
                    );
                }
            }
        }
        check_exact(1, &SegP1);
        check_exact(2, &SegP2);

        // Order 3: the fixed-order `SegP3` puts its interior DOFs at 1/3 and
        // 2/3, the trace at the Gauss-Lobatto points (0.2764, 0.7236) — the
        // equispaced basis would distribute a boundary integral to the wrong
        // DOFs while keeping its sum (partition of unity) exact.
        let gen3 = H1SegPk::new(3);
        let gll = fem_element::lagrange::factory::QuadQk::new(3)
            .dof_coords()
            .iter()
            .take(6)
            .map(|c| c[0])
            .collect::<Vec<_>>();
        assert_eq!(gen3.dof_coords()[0], vec![0.0]);
        assert_eq!(gen3.dof_coords()[1], vec![1.0]);
        // Interior trace nodes == the volume element's bottom-edge nodes.
        for (k, &x) in gll[4..6].iter().enumerate() {
            assert_eq!(gen3.dof_coords()[2 + k], vec![x]);
        }
        assert!((gen3.dof_coords()[2][0] - 1.0 / 3.0).abs() > 1e-3);
        assert_eq!(SegP3.dof_coords()[2], vec![1.0 / 3.0]);

        // The ascending-order factory element has a different DOF *order* at
        // p >= 2 (its DOF 1 is an interior node, not the second vertex).
        let asc = fem_element::lagrange::factory::SegPk::new(3);
        assert_eq!(asc.dof_coords()[1], vec![1.0 / 3.0]);
        assert_eq!(gen3.dof_coords()[1], vec![1.0]);
    }

    /// D46②: `ref_elem_face` now serves every order, so an order-6 boundary
    /// form can be assembled by the kernel instead of panicking.
    #[test]
    fn ref_elem_face_high_order() {
        let re = ref_elem_face(ElementType::Line2, 6);
        assert_eq!(re.n_dofs(), 7);
        assert_eq!(re.order(), 6);
        // Topological DOF order with the volume element's Gauss-Lobatto nodes.
        assert_eq!(re.dof_coords()[0], vec![0.0]);
        assert_eq!(re.dof_coords()[1], vec![1.0]);
        let gll = fem_element::lagrange::factory::QuadQk::new(6).dof_coords();
        for (k, c) in gll[4..9].iter().enumerate() {
            assert_eq!(re.dof_coords()[2 + k], vec![c[0]]);
        }
        // Existing fixed-order entries keep working (ex21 relies on order 2).
        assert_eq!(ref_elem_face(ElementType::Line2, 1).n_dofs(), 2);
        assert_eq!(ref_elem_face(ElementType::Line2, 2).n_dofs(), 3);
        assert_eq!(ref_elem_face(ElementType::Tri3, 5).n_dofs(), 21);
    }

    /// The face DOF list produced by [`face_dofs_h1`] must be the one the
    /// order-6 face element pairs with: `[v0, v1, interior along v0 → v1]`,
    /// all of them global DOFs of the space whose reference nodes sit on the
    /// face's edge.
    #[test]
    fn face_dofs_h1_order_6_ordering() {
        let mesh = Mesh::<2>::make_cartesian_2d(1, 1, 1.0, 1.0);
        let space = H1Space::new(mesh, 6);
        let fdofs = face_dofs_h1(&space);
        let dm = space.dof_manager();
        for f in space.mesh().face_iter() {
            let n0 = space.mesh().face_nodes(f)[0];
            let n1 = space.mesh().face_nodes(f)[1];
            let p0 = space.mesh().node_coords(n0);
            let p1 = space.mesh().node_coords(n1);
            let d = fdofs(f);
            assert_eq!(d.len(), 7, "face {f}");
            // Vertices first, in the face's own node order…
            assert_eq!(d[0], n0, "face {f}: first DOF must be v0");
            assert_eq!(d[1], n1, "face {f}: second DOF must be v1");
            // …then the interior DOFs in increasing order along v0 → v1.
            let param = |dof: DofId| -> f64 {
                let c = dm.dof_coord(dof);
                let ab = [p1[0] - p0[0], p1[1] - p0[1]];
                let t = ((c[0] - p0[0]) * ab[0] + (c[1] - p0[1]) * ab[1])
                    / (ab[0] * ab[0] + ab[1] * ab[1]);
                // Every DOF of the list sits exactly on the edge.
                let ex = c[0] - (p0[0] + t * ab[0]);
                let ey = c[1] - (p0[1] + t * ab[1]);
                assert!(
                    (ex * ex + ey * ey).sqrt() < 1e-12,
                    "DOF {dof} is not on face {f}"
                );
                t
            };
            for k in 0..d.len() - 3 {
                assert!(
                    param(d[2 + k]) < param(d[3 + k]),
                    "face {f}: interior DOFs not sorted along v0 -> v1"
                );
            }
        }
    }

    /// D46②③ end to end: `∫_Γ (v·n) φᵢ ds` at order 6 via
    /// `assemble_boundary_linear` + `face_dofs_h1` on the unit square, checked
    /// against the divergence theorem and against the exact integral
    /// `∮ (v·n) ψ ds` for `ψ = x` (both `∫_Ω ∇·v` and `∮ (v·n) x` equal 1 for
    /// `v = (x, 0)`).
    #[test]
    fn boundary_normal_flux_order_6_divergence_theorem() {
        use crate::postproc::coefficient::{FnVectorCoeff, VectorCoeff};
        let mesh = Mesh::<2>::make_cartesian_2d(1, 1, 1.0, 1.0);
        let space = H1Space::new(mesh, 6);
        let integ = crate::standard::boundary_flux::VectorBoundaryNormalLFIntegrator {
            v: FnVectorCoeff(|x: &[f64], out: &mut [f64]| {
                out[0] = x[0];
                out[1] = 0.0;
            }),
        };
        // Sanity: the coefficient itself is a `VectorCoeff`.
        let _: &dyn VectorCoeff = &integ.v;

        let fdofs = face_dofs_h1(&space);
        let rhs = Assembler::assemble_boundary_linear(
            space.n_dofs(),
            space.mesh(),
            &fdofs,
            6,
            &[&integ],
            &[1, 2, 3, 4],
            7,
        );
        let total: f64 = rhs.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "∮ v·n ds = {total}, want 1");

        let dm = space.dof_manager();
        let mut weighted = 0.0_f64;
        for (i, &r) in rhs.iter().enumerate() {
            weighted += r * dm.dof_coord(i as u32)[0];
        }
        assert!(
            (weighted - 1.0).abs() < 1e-12,
            "∮ (v·n) x ds = {weighted}, want 1"
        );
    }

    /// `BdQpData::normal` must be the **outward** unit normal not only for a
    /// single-element mesh but also after uniform refinement — where the
    /// boundary faces are re-created and may not keep MFEM's "element on the
    /// left of the vertex order" convention.  With `v = (x, 0)` the divergence
    /// theorem gives `Σᵢ rhsᵢ = ∮ v·n ds = |Ω|` for every mesh; a flipped
    /// normal changes the sign of its own face's contribution.
    ///
    /// This is the test that catches the kovasznay regression: its
    /// `g_bdr = ∫_Γ (u_D·n) q ds` flips sign on every boundary edge whose
    /// stored orientation is reversed, which blows the pressure solve up from
    /// `err_p ≈ 1e-6` to `≈ 6`.
    #[test]
    fn boundary_normal_is_outward_after_refinement() {
        // (nx, ny, sx, sy, refinements) — `sx`/`sy` are the *domain sizes*
        // (MFEM `MakeCartesian2D`), so the area is `sx*sy` in every case.
        for (nx, ny, sx, sy, refs) in [
            (1usize, 1usize, 1.0, 1.0, 0usize),
            (2, 4, 1.5, 2.0, 0),
            (2, 4, 1.5, 2.0, 1),
            (3, 3, 2.0, 2.0, 2),
        ] {
            let mut mesh = Mesh::<2>::make_cartesian_2d(nx, ny, sx, sy);
            for _ in 0..refs {
                mesh = fem_mesh::refine_uniform(&mesh);
            }
            let space = H1Space::new(mesh, 6);
            let integ = crate::standard::boundary_flux::VectorBoundaryNormalLFIntegrator {
                v: crate::postproc::coefficient::FnVectorCoeff(|x: &[f64], out: &mut [f64]| {
                    out[0] = x[0];
                    out[1] = 0.0;
                }),
            };
            let tags: Vec<i32> = (1..=4).collect();
            let rhs = Assembler::assemble_boundary_linear(
                space.n_dofs(),
                space.mesh(),
                &face_dofs_h1(&space),
                6,
                &[&integ],
                &tags,
                7,
            );
            let total: f64 = rhs.iter().sum();
            let area = sx * sy;
            assert!(
                (total - area).abs() < 1e-12,
                "{nx}x{ny} on {sx}x{sy} + {refs} refinements: area {area}, got {total}"
            );
        }
    }

    /// Round-10 (round-9 leftover): the affine simplex fast path must read the
    /// element's per-element **geometry** (MFEM `Nodes` snapshot), not the
    /// merged vertex table — on a geometrically periodic mesh the seam-crossing
    /// Tri3 elements otherwise get a folded (degenerate/mirrored) Jacobian.
    ///
    /// Verification: on a periodic 2×2 triangular mesh every element has the
    /// same shape as in the un-periodised mesh, so the assembled element mass
    /// matrices must agree element-by-element (this is exactly "per-element
    /// detJ identical"), and `Σ M_ij = ∫_Ω 1 dx = 1`.
    #[test]
    fn periodic_tri_affine_path_keeps_element_geometry() {
        let straight = Mesh::<2>::make_cartesian_2d_tri(3, 3, 1.0, 1.0);
        let periodic = straight
            .clone()
            .make_periodic(&[(4, 2, [1.0, 0.0]), (1, 3, [0.0, 1.0])], 1e-10)
            .expect("make_periodic");
        assert_eq!(periodic.geom_order(), 1, "periodic snapshot is order-1 geometry");
        assert!(periodic.geometry.is_some(), "per-element geometry must be present");

        let sp_s = H1Space::new(straight.clone(), 1);
        let sp_p = H1Space::new(periodic.clone(), 1);
        let (_, _, k_s, ld, ne) = Assembler::assemble_bilinear_with_elements(
            &sp_s, &[&MassIntegrator { rho: 1.0 }], 3,
        );
        let (m_p, _, k_p, ld_p, _) = Assembler::assemble_bilinear_with_elements(
            &sp_p, &[&MassIntegrator { rho: 1.0 }], 3,
        );
        assert_eq!(ld, ld_p);
        assert_eq!(ne, periodic.n_elems() as usize);

        let mut max_dev = 0.0_f64;
        for e in 0..ne {
            for i in 0..ld * ld {
                max_dev = max_dev.max((k_s[e * ld * ld + i] - k_p[e * ld * ld + i]).abs());
            }
        }
        eprintln!("periodic vs straight per-element mass: max |Δ| = {max_dev:.3e}");
        assert!(
            max_dev < 1e-15,
            "per-element mass matrices differ (max |Δ| = {max_dev:.3e})"
        );

        // Global sanity: Σ_ij M_ij = ∫_Ω 1 dx = 1 for the (unit) periodic square.
        let ones = vec![1.0_f64; m_p.nrows];
        let mut y = vec![0.0_f64; m_p.nrows];
        m_p.spmv(&ones, &mut y);
        let total: f64 = y.iter().sum();
        eprintln!("periodic Σ M_ij = {total:.15}");
        assert!((total - 1.0).abs() < 1e-13, "Σ M_ij = {total} (expected 1)");

        // Document what the pre-fix (merged vertex table) path would produce:
        // every element of the straight mesh has |detJ| = h² (h = 1/3), while the
        // folded chord of a seam-crossing triangle does not.
        let h = 1.0 / 3.0;
        let n_folded_wrong = (0..periodic.n_elems() as u32)
            .filter(|&e| {
                let nds = periodic.element_nodes(e);
                let x0 = periodic.node_coords(nds[0]);
                let x1 = periodic.node_coords(nds[1]);
                let x2 = periodic.node_coords(nds[2]);
                let det = ((x1[0] - x0[0]) * (x2[1] - x0[1])
                    - (x1[1] - x0[1]) * (x2[0] - x0[0]))
                    .abs();
                (det - h * h).abs() > 1e-12
            })
            .count();
        eprintln!(
            "folded-vertex |detJ| ≠ {:.6} on {} of {} elements",
            h * h,
            n_folded_wrong,
            periodic.n_elems()
        );
        assert!(n_folded_wrong > 0, "test setup: the folded table should be wrong here");
    }

    #[test]
    fn assemble_bilinear_p1_returns_correct_size() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let n = space.n_dofs();
        // Diffusion integrator stub (adds nothing) — just test shape.
        struct Zero;
        impl BilinearIntegrator for Zero {
            fn add_to_element_matrix(&self, _: &QpData<'_>, _: &mut [f64]) {}
        }
        let mat = Assembler::assemble_bilinear(&space, &[&Zero], 2);
        assert_eq!(mat.nrows, n);
        assert_eq!(mat.ncols, n);
    }

    #[test]
    fn assemble_linear_p1_returns_correct_size() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let n = space.n_dofs();
        struct Zero;
        impl LinearIntegrator for Zero {
            fn add_to_element_vector(&self, _: &QpData<'_>, _: &mut [f64]) {}
        }
        let rhs = Assembler::assemble_linear(&space, &[&Zero], 2);
        assert_eq!(rhs.len(), n);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn assembly_parallel_min_elems_positive() {
        assert!(assembly_parallel_min_elems() >= 1);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn adaptive_threshold_scales_with_threads() {
        assert_eq!(adaptive_assembly_threshold_for_threads(1), 64);
        assert_eq!(adaptive_assembly_threshold_for_threads(2), 32);
        assert_eq!(adaptive_assembly_threshold_for_threads(3), 32);
        assert_eq!(adaptive_assembly_threshold_for_threads(4), 16);
        assert_eq!(adaptive_assembly_threshold_for_threads(8), 8);
        assert_eq!(adaptive_assembly_threshold_for_threads(32), 8);
    }

    /// D11 regression: tri-P0 standard assembly integrals must not be doubled.
    /// The old `P0 { dim: 2 }` element returned the square `[0,1]²` Gauss rule
    /// (weight sum 1) for triangle owners, so `∫ 1 dx` on the unit square
    /// evaluated to 2.0 (the affine tri map has |detJ| = 2·Area everywhere,
    /// including outside the reference triangle).  `P0Tri` uses the MFEM
    /// `IntRules.Get(TRIANGLE, ·)` rule (weight sum 1/2), so the element-mass
    /// diagonal sums to the domain area.
    #[test]
    fn tri_p0_mass_not_doubled() {
        use fem_space::L2Space;
        use crate::standard::MassIntegrator;

        let mesh = Mesh::<2>::unit_square_tri(3);
        let l2 = L2Space::new(mesh, 0);
        let m = Assembler::assemble_bilinear(&l2, &[&MassIntegrator { rho: 1.0 }], 2);
        let diag_sum: f64 = (0..m.nrows).map(|i| m.get(i, i)).sum();
        assert!(
            (diag_sum - 1.0).abs() < 1e-12,
            "Σ diag M for tri-P0 = {diag_sum} (expected 1.0; 2.0 means the square rule is back)"
        );
    }

    // ─── D50: 3-D boundary assembly ──────────────────────────────────────────

    /// The discrete divergence theorem, **per DOF**, for the *constant* test
    /// vector `v = e_x`:
    ///
    /// ```text
    /// ∮_Γ (e_x·n) φᵢ ds = ∫_Ω ∇·(e_x φᵢ) dV = ∫_Ω ∂φᵢ/∂x dV.
    /// ```
    ///
    /// The left side is [`Assembler::assemble_boundary_linear`] with
    /// [`VectorBoundaryNormalLFIntegrator`](crate::standard::boundary_flux::VectorBoundaryNormalLFIntegrator)
    /// — the 3-D face Jacobian (`|t0 × t1|`), the outward normal and the
    /// boundary element's shape functions.  The right side is
    /// [`Assembler::assemble_linear`] with [`WeakDivConstLF`], a test-local
    /// `∫ (e_x·∇φᵢ) dV` integrator running over the *volume* elements with
    /// isoparametric Jacobians.  Both sides are exact for the constant `v` and
    /// the degree-`p` basis, and they share no code, so comparing them DOF by
    /// DOF pins:
    ///
    /// * the face basis function paired with each `face_dofs_h1` entry (a wrong
    ///   face DOF order permutes the vector and destroys the equality);
    /// * the face basis *nodes* (a face basis whose nodes are not the volume
    ///   element's face nodes has the correct partition of unity but puts the
    ///   flux on the wrong DOFs — the D46② failure mode);
    /// * `|J_face|`, the outward normal and the physical quadrature points.
    ///
    /// A flipped face orientation shows up directly: the face at `x = 1` must
    /// contribute `+∫φᵢ` and the one at `x = 0` `−∫φᵢ`.
    ///
    /// Deliberately *not* used here: `K·u` with `u_i = dof_coord(i)[0]`.  It
    /// would be the same identity through a third path, but it needs
    /// `DofManager::dof_coord`, which on tetrahedra only stores *approximate*
    /// coordinates for face and volume DOFs (see the DofManager's
    /// `build_pk`), so `u` would not be the interpolant of `x`.
    struct WeakDivConstLF {
        v: [f64; 3],
    }

    impl LinearIntegrator for WeakDivConstLF {
        fn add_to_element_vector(&self, qp: &QpData<'_>, f_elem: &mut [f64]) {
            for k in 0..qp.n_dofs {
                let g = &qp.grad_phys[k * qp.dim..(k + 1) * qp.dim];
                let s: f64 = (0..qp.dim).map(|d| self.v[d] * g[d]).sum();
                f_elem[k] += qp.weight * s;
            }
        }
    }

    fn check_divergence_theorem_3d(mesh: Mesh<3>, order: u8, label: &str) {
        use crate::postproc::coefficient::FnVectorCoeff;

        let space = H1Space::new(mesh, order);
        // `order + 3` integrates both sides exactly: the volume integrand is
        // `(e_x·∇φᵢ)·detJ` (degree `p - 1`, `detJ` constant on these straight
        // meshes) and the face integrand is `φᵢ·|J_face|` where the bilinear
        // face map makes `|J_face|` degree ≤ 2.
        let quad_order = order + 3;

        let vol = Assembler::assemble_linear(
            &space,
            &[&WeakDivConstLF { v: [1.0, 0.0, 0.0] }],
            quad_order,
        );

        let integ = crate::standard::boundary_flux::VectorBoundaryNormalLFIntegrator {
            v: FnVectorCoeff(|_x: &[f64], out: &mut [f64]| {
                out[0] = 1.0;
                out[1] = 0.0;
                out[2] = 0.0;
            }),
        };
        let tags: Vec<i32> = (1..=6).collect();
        let rhs = Assembler::assemble_boundary_linear(
            space.n_dofs(),
            space.mesh(),
            &face_dofs_h1(&space),
            order,
            &[&integ],
            &tags,
            quad_order,
        );

        let mut max_dev = 0.0_f64;
        let mut worst = 0usize;
        for i in 0..rhs.len() {
            let d = (rhs[i] - vol[i]).abs();
            if d > max_dev {
                max_dev = d;
                worst = i;
            }
        }
        eprintln!(
            "{label} order {order}: max |∮(v·n)φᵢ − ∫(v·∇φᵢ)| = {max_dev:.3e} (dof {worst}/{}), \
             Σ∮ = {:.12}",
            rhs.len(),
            rhs.iter().sum::<f64>()
        );
        assert!(
            max_dev < 1e-11,
            "{label} order {order}: DOF {worst} of {}: ∮(v·n)φᵢ = {} but ∫(v·∇φᵢ) = {} \
             (max |Δ| = {max_dev:.3e})",
            rhs.len(),
            rhs[worst],
            vol[worst],
        );
    }

    /// D50: `∫_Ω ∇·u dV = ∮_∂Ω u·n dS` for `u = (x, y, z)` — the divergence
    /// theorem in the *global* sense, `Σᵢ rhsᵢ = 3|Ω|`.  Any face whose outward
    /// normal points inward changes the sign of its own contribution, so this
    /// is also the 3-D counterpart of
    /// [`boundary_normal_is_outward_after_refinement`].
    fn check_divergence_theorem_3d_total(mesh: Mesh<3>, order: u8, label: &str) {
        use crate::postproc::coefficient::FnVectorCoeff;

        let space = H1Space::new(mesh.clone(), order);
        let integ = crate::standard::boundary_flux::VectorBoundaryNormalLFIntegrator {
            v: FnVectorCoeff(|x: &[f64], out: &mut [f64]| {
                out[0] = x[0];
                out[1] = x[1];
                out[2] = x[2];
            }),
        };
        let tags: Vec<i32> = (1..=6).collect();
        let rhs = Assembler::assemble_boundary_linear(
            space.n_dofs(),
            space.mesh(),
            &face_dofs_h1(&space),
            order,
            &[&integ],
            &tags,
            2 * order + 2,
        );
        let total: f64 = rhs.iter().sum();
        let want = 3.0 * volume_of(space.mesh());
        eprintln!("{label} order {order}: Σ∮ (x,y,z)·n ds = {total:.12} (3|Ω| = {want:.12})");
        assert!(
            (total - want).abs() < 1e-11,
            "{label} order {order}: Σ∮ = {total}, want 3|Ω| = {want}"
        );
    }

    /// `|Ω|` of a Cartesian box; the 3-D fixtures are built by
    /// `make_cartesian_3d`, whose domain is `[0,sx]×[0,sy]×[0,sz]`.
    fn volume_of(mesh: &Mesh<3>) -> f64 {
        let (lo, hi) = mesh.bounding_box();
        (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])
    }

    /// D50 acceptance: 3-D boundary assembly on hexahedra and tetrahedra
    /// satisfies the discrete (per-DOF) and global divergence theorem, on
    /// straight and uniformly refined meshes.
    #[test]
    fn boundary_assembly_3d_divergence_theorem() {
        // (nx, ny, nz, sx, sy, sz, refinements, orders)
        for etype in [ElementType::Hex8, ElementType::Tet4] {
            for (nx, ny, nz, sx, sy, sz, refs, orders) in [
                (1usize, 1usize, 1usize, 1.0, 1.0, 1.0, 0usize, &[1u8, 2, 3, 4, 5, 6][..]),
                (2, 1, 3, 2.0, 1.5, 0.75, 0, &[1, 2, 3, 4, 5, 6][..]),
                (2, 2, 2, 1.0, 1.0, 1.0, 1, &[2, 6][..]),
            ] {
                let mut mesh =
                    Mesh::<3>::make_cartesian_3d(nx, ny, nz, etype, sx, sy, sz, false);
                for _ in 0..refs {
                    mesh = fem_mesh::refine_uniform_3d(&mesh);
                }
                let label = format!("{etype:?} {nx}x{ny}x{nz} + {refs}");
                for &order in orders {
                    check_divergence_theorem_3d(mesh.clone(), order, &label);
                }
                check_divergence_theorem_3d_total(mesh, 6, &label);
            }
        }
    }

    /// The face DOF list produced by [`face_dofs_h1`] in 3-D must be the one
    /// the face reference element ([`ref_elem_face`]) pairs with: the vertices
    /// first in `face_nodes` order, then the edge DOFs of the face's own 2-D
    /// topology in increasing parameter, then the face interior.
    #[test]
    fn face_dofs_h1_3d_ordering() {
        let mesh = Mesh::<3>::make_cartesian_3d(1, 1, 1, ElementType::Hex8, 1.0, 1.0, 1.0, false);
        for order in [1u8, 2, 3, 6] {
            let space = H1Space::new(mesh.clone(), order);
            let fdofs = face_dofs_h1(&space);
            let dm = space.dof_manager();
            let np1 = order as usize + 1;
            for f in space.mesh().face_iter() {
                let d = fdofs(f);
                assert_eq!(d.len(), np1 * np1, "order {order} face {f}");
                let corners = space.mesh().face_nodes(f);
                // Vertices first, in the face's own node order.
                for (k, &n) in corners.iter().enumerate() {
                    let c = space.mesh().node_coords(n);
                    let dc = dm.dof_coord(d[k]);
                    for i in 0..3 {
                        assert!(
                            (c[i] - dc[i]).abs() < 1e-12,
                            "order {order} face {f}: DOF {k} must be vertex {n}"
                        );
                    }
                }
                if order < 2 {
                    continue;
                }
                // Then, for each face edge, its interior DOFs in increasing
                // parameter measured from the edge's first corner.
                let mut at = corners.len();
                for e in 0..corners.len() {
                    let (a, b) = (corners[e], corners[(e + 1) % corners.len()]);
                    let (pa, pb) = (space.mesh().node_coords(a), space.mesh().node_coords(b));
                    let ab: Vec<f64> = (0..3).map(|i| pb[i] - pa[i]).collect();
                    let prev = param(&dm.dof_coord(d[at - 1]), pa, &ab);
                    for k in 0..(order as usize - 1) {
                        let t = param(&dm.dof_coord(d[at + k]), pa, &ab);
                        assert!(
                            t > prev && t > 0.0 && t < 1.0,
                            "order {order} face {f} edge {e}: DOF at {t} out of order"
                        );
                    }
                    at += order as usize - 1;
                }
                assert!(
                    at <= d.len(),
                    "order {order} face {f}: {} edge DOFs but only {} total",
                    at,
                    d.len()
                );
            }
        }

        fn param(c: &[f64], a: &[f64], ab: &[f64]) -> f64 {
            let ab2: f64 = ab.iter().map(|x| x * x).sum();
            (0..3).map(|i| (c[i] - a[i]) * ab[i]).sum::<f64>() / ab2
        }
    }

    /// D50: the 3-D face reference element must be MFEM's boundary element —
    /// `H1_TriangleElement` / `H1_QuadrilateralElement` DOF order, with the
    /// **volume** element's face nodes (so the face basis is the restriction of
    /// the space's basis), and not `QuadQ1`/`QuadQ2` on `[-1,1]²`.
    #[test]
    fn ref_elem_face_3d_is_mfem_boundary_element() {
        // Quad face: MFEM `H1_QuadrilateralElement` on `[0,1]²` — the hex
        // volume element (`HexQk`) is itself Gauss-Lobatto, so the trace
        // positions *are* the closed points.
        let gll = fem_element::quadrature::gauss_lobatto_arbitrary(4).0;
        let gll01 = |k: usize| 0.5 * (gll[k] + 1.0);
        let quad = ref_elem_face(ElementType::Quad4, 3);
        assert_eq!(quad.n_dofs(), 16);
        assert_eq!(quad.dof_coords()[0], vec![0.0, 0.0]);
        assert_eq!(quad.dof_coords()[1], vec![1.0, 0.0]);
        assert_eq!(quad.dof_coords()[2], vec![1.0, 1.0]);
        assert_eq!(quad.dof_coords()[3], vec![0.0, 1.0]);
        // Bottom edge left → right, right edge bottom → top, top edge
        // right → left, left edge top → bottom (H1_DOF_MAP).
        assert_eq!(quad.dof_coords()[4], vec![gll01(1), 0.0]);
        assert_eq!(quad.dof_coords()[6], vec![1.0, gll01(1)]);
        assert_eq!(quad.dof_coords()[8], vec![gll01(2), 1.0]);
        assert_eq!(quad.dof_coords()[10], vec![0.0, gll01(2)]);
        // Interior, `(j, i)` nested with i fastest.
        let base = 4 + 4 * 2;
        assert_eq!(quad.dof_coords()[base], vec![gll01(1), gll01(1)]);
        assert_eq!(quad.dof_coords()[base + 1], vec![gll01(2), gll01(1)]);
        assert_eq!(quad.dof_coords()[base + 2], vec![gll01(1), gll01(2)]);
        // `H1_DOF_MAP`'s `QuadQ1`/`QuadQ2` live on `[-1,1]²` — still the face
        // geometry element's domain, which is what the transport needs.
        assert_eq!(face_geo_elem(ElementType::Quad4).dof_coords()[2], vec![1.0, 1.0]);
        assert_eq!(face_geo_elem(ElementType::Tri3).dof_coords()[1], vec![1.0, 0.0]);
        assert_eq!(face_geo_elem(ElementType::Line2).dof_coords()[1], vec![1.0]);

        // Triangle face: MFEM's `H1_TriangleElement` DOF order
        // [v0, v1, v2, edge0…, edge1…, edge2…, interior…] over the *tet*
        // volume element's face nodes.  The tet H¹ basis is equispaced
        // (`factory::TetPk`), so the trace nodes are at `k/p` (see
        // [`H1TetFacePk`]) — the DOF *order* is what must match MFEM, and the
        // nodal property below pins that the elements are the volume
        // element's traces.
        for p in 1..=6usize {
            let face = ref_elem_face(ElementType::Tri3, p as u8);
            let vol = fem_element::lagrange::factory::TetPk::new(p);
            let fcoords = face.dof_coords();
            assert_eq!(fcoords.len(), (p + 1) * (p + 2) / 2, "p = {p}");
            let vol_coords = vol.dof_coords();
            let pp = p as f64;
            let close = |a: &[f64], b: [f64; 2]| {
                (a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12
            };
            // Vertices.
            assert!(close(&fcoords[0], [0.0, 0.0]));
            assert!(close(&fcoords[1], [1.0, 0.0]));
            assert!(close(&fcoords[2], [0.0, 1.0]));
            // Edge blocks, in MFEM's order and directions.
            for i in 1..p {
                assert!(close(&fcoords[3 + i - 1], [i as f64 / pp, 0.0]), "p {p} e0 {i}");
            }
            for i in 1..p {
                assert!(
                    close(&fcoords[3 + (p - 1) + i - 1], [(p - i) as f64 / pp, i as f64 / pp]),
                    "p {p} e1 {i}"
                );
            }
            for i in 1..p {
                assert!(
                    close(&fcoords[3 + 2 * (p - 1) + i - 1], [0.0, (p - i) as f64 / pp]),
                    "p {p} e2 {i}"
                );
            }
            // Interior, `(j, i)` nested with i fastest.
            let mut at = 3 + 3 * (p - 1);
            for j in 1..p {
                for i in 1..(p - j) {
                    assert!(
                        close(&fcoords[at], [i as f64 / pp, j as f64 / pp]),
                        "p {p} interior {i},{j}"
                    );
                    at += 1;
                }
            }
            assert_eq!(at, fcoords.len(), "p = {p}");

            // Every node lies on a volume DOF, and the face basis is *nodal*
            // there — i.e. the face basis functions really are the volume
            // basis functions restricted to the face (a wrong slot resolution
            // would break either of these).
            let n = fcoords.len();
            let mut vals = vec![0.0_f64; n];
            for (k, fc) in fcoords.iter().enumerate() {
                let on_face = vol_coords
                    .iter()
                    .any(|c| (c[0] - fc[0]).abs() < 1e-12 && (c[1] - fc[1]).abs() < 1e-12 && c[2].abs() < 1e-12);
                assert!(on_face, "p {p}: face DOF {k} at {fc:?} is not a volume DOF");
                face.eval_basis(fc, &mut vals);
                for (l, &v) in vals.iter().enumerate() {
                    let want = if l == k { 1.0 } else { 0.0 };
                    assert!(
                        (v - want).abs() < 1e-12,
                        "p {p}: φ_{l}(node {k}) = {v}, want {want}"
                    );
                }
            }
        }
    }

    /// D50 acceptance against **MFEM itself**: per boundary element and per
    /// local face DOF, the 3-D boundary assembly must reproduce MFEM's
    /// `BoundaryNormalLFIntegrator` (`∫_Γ (e_x·n) φ_k ds`).
    ///
    /// Reference data: serial MFEM 4.10 on `Mesh::MakeCartesian3D(1,1,1)`
    /// (hexahedra and tetrahedra), orders 1…6, assembled with the explicit rule
    /// `IntRules.Get(face_geom, p + 3)` — the same rule family and order that
    /// `assemble_boundary_linear` is called with here (fem-rs mirrors MFEM's
    /// point/weight layout, so the sums agree to round-off).  Harness:
    /// `tests/data/bdr3d.cpp`, output `tests/data/bdr_3d_cpp.txt`.
    ///
    /// Per boundary face the test checks
    /// * the attribute and the face vertices (i.e. that the two meshes and
    ///   their boundary-element order coincide, which the whole comparison
    ///   rests on);
    /// * the face element's reference **DOF positions**, which pin the DOF
    ///   order and the node placement;
    /// * every entry `k` of the local vector.
    ///
    /// Hexahedron faces agree to round-off at every order.  Tetrahedron faces
    /// agree only for `p ≤ 2`: at `p ≥ 3` MFEM's `H1_TriangleElement` places
    /// the face nodes at the closed Gauss-Lobatto points while the fem-rs H¹
    /// tetrahedron basis (`factory::TetPk`) is *equispaced*, so the two
    /// assemblers integrate genuinely different basis functions (D49 — see
    /// [`H1TetFacePk`]).  For those orders the test asserts the face node
    /// positions are exactly the equispaced trace of the volume element (which
    /// documents the gap) and reports the value difference.
    #[test]
    fn boundary_assembly_3d_matches_mfem_reference() {
        use crate::postproc::coefficient::FnVectorCoeff;

        const DUMP: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/bdr_3d_cpp.txt"
        ));

        struct Face {
            tag: i32,
            verts: Vec<u32>,
            nodes: Vec<[f64; 2]>,
            vals: Vec<f64>,
        }
        struct Case {
            tet: bool,
            p: u8,
            faces: Vec<Face>,
        }
        let mut cases: Vec<Case> = Vec::new();
        for line in DUMP.lines() {
            let t: Vec<&str> = line.split_whitespace().collect();
            match t.first().copied() {
                Some("case") => {
                    let p = t[2].trim_start_matches("p=").parse().unwrap();
                    cases.push(Case { tet: t[1] == "tet", p, faces: Vec::new() });
                }
                Some("be") => {
                    let tag: i32 = t[3].parse().unwrap();
                    let nv: usize = t[7].parse().unwrap();
                    let verts = t[11..11 + nv].iter().map(|v| v.parse().unwrap()).collect();
                    cases.last_mut().unwrap().faces.push(Face {
                        tag,
                        verts,
                        nodes: Vec::new(),
                        vals: Vec::new(),
                    });
                }
                Some("k") => {
                    let f = cases.last_mut().unwrap().faces.last_mut().unwrap();
                    assert_eq!(f.nodes.len(), t[1].parse::<usize>().unwrap());
                    f.nodes.push([t[3].parse().unwrap(), t[4].parse().unwrap()]);
                    f.vals.push(t[8].parse().unwrap());
                }
                Some(other) => panic!("unexpected dump token {other:?}"),
                None => {}
            }
        }
        assert_eq!(cases.len(), 12, "6 orders x 2 element types");

        for case in &cases {
            let etype = if case.tet { ElementType::Tet4 } else { ElementType::Hex8 };
            let label = format!("{etype:?} p={}", case.p);
            let agrees = !case.tet || case.p <= 2;
            let mesh = Mesh::<3>::make_cartesian_3d(1, 1, 1, etype, 1.0, 1.0, 1.0, false);
            let space = H1Space::new(mesh, case.p);
            let fdofs = face_dofs_h1(&space);
            let quad_order = case.p + 3;
            let face_elt = ref_elem_face(
                if case.tet { ElementType::Tri3 } else { ElementType::Quad4 },
                case.p,
            );
            let integ = crate::standard::boundary_flux::VectorBoundaryNormalLFIntegrator {
                v: FnVectorCoeff(|_x: &[f64], out: &mut [f64]| {
                    out[0] = 1.0;
                    out[1] = 0.0;
                    out[2] = 0.0;
                }),
            };
            assert_eq!(space.mesh().face_iter().len(), case.faces.len(), "{label}: faces");

            let mut max_pos = 0.0_f64;
            let mut max_val = 0.0_f64;
            for (f, want) in space.mesh().face_iter().zip(case.faces.iter()) {
                let fnodes = space.mesh().face_nodes(f);
                assert_eq!(want.tag, space.mesh().face_tag(f), "{label}: face {f} tag");
                assert_eq!(
                    fnodes.to_vec(),
                    want.verts,
                    "{label}: face {f} vertices (the mesh and its boundary order must \
                     match MFEM for this comparison to mean anything)"
                );

                let d = fdofs(f);
                assert_eq!(d.len(), want.vals.len(), "{label}: face {f} DOF count");
                for (k, (c, w)) in face_elt.dof_coords().iter().zip(want.nodes.iter()).enumerate()
                {
                    let dp = (c[0] - w[0]).abs().max((c[1] - w[1]).abs());
                    max_pos = max_pos.max(dp);
                    if agrees {
                        assert!(
                            dp < 1e-15,
                            "{label}: face {f} DOF {k} node {c:?} != MFEM's {w:?}"
                        );
                    }
                }

                // The local face vector: a single-face accumulation read back at
                // the face's own DOFs.
                let mut local = vec![0.0_f64; space.n_dofs()];
                accumulate_boundary_linear_face(
                    space.mesh(),
                    f,
                    &fdofs,
                    case.p,
                    &[&integ],
                    quad_order,
                    &mut local,
                );
                for (k, &g) in d.iter().enumerate() {
                    let dv = (local[g as usize] - want.vals[k]).abs();
                    max_val = max_val.max(dv);
                    if agrees {
                        assert!(
                            dv < 1e-12,
                            "{label}: face {f} DOF {k}: ∫(e_x·n)φ_k = {} but MFEM has {} \
                             (|Δ| = {dv:.3e})",
                            local[g as usize],
                            want.vals[k]
                        );
                    }
                }
            }
            eprintln!(
                "{label}: max |Δ node| = {max_pos:.3e}, max |Δ value| = {max_val:.3e}{}",
                if agrees {
                    ""
                } else {
                    "   (D49: MFEM's Gauss-Lobatto face nodes vs fem-rs's equispaced tet basis)"
                }
            );

            if !agrees {
                // The face nodes are the *equispaced* trace of the volume
                // element — `k/p` along each face edge — while MFEM's are the
                // closed Gauss-Lobatto points, and the two differ visibly.
                let pp = case.p as f64;
                for i in 1..case.p as usize {
                    let c = &face_elt.dof_coords()[3 + i - 1];
                    assert!(
                        (c[0] - i as f64 / pp).abs() < 1e-12 && c[1].abs() < 1e-12,
                        "p = {}: face edge node {i} at {c:?}, expected {}/{}",
                        case.p,
                        i,
                        case.p
                    );
                }
                let gll =
                    fem_element::quadrature::gauss_lobatto_arbitrary(case.p as usize + 1).0;
                let mfem_node = 0.5 * (gll[1] + 1.0);
                assert!(
                    (mfem_node - 1.0 / pp).abs() > 1e-3,
                    "p = {}: Gauss-Lobatto and equispaced edge nodes should differ here",
                    case.p
                );
            }
        }
    }

}
