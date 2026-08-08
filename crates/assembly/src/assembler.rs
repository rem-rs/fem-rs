//! Global assembly loop.
//!
//! [`Assembler`] drives the element-by-element assembly of bilinear and linear
//! forms over the mesh.  It is stateless; all data comes from the [`FESpace`]
//! and integrators supplied at call time.

use nalgebra::DMatrix;

use fem_core::types::DofId;
use fem_element::{
    QuadratureRule, ReferenceElement, PrismPk, PyramidPk, VectorReferenceElement,
    lagrange::{SegP1, SegP2, SegP3, SegP4, TetP1, TetP2, TetP3, TriP1, TriP2, TriP3, TriP4,
                QuadQ1, QuadQ2, QuadQ4, HexQ1},
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

/// Constant (P0) reference element on the reference `[-1,1]²` domain.
/// 1 DOF, basis ≡ 1.0, gradient ≡ 0.
struct P0;

impl ReferenceElement for P0 {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { 0 }
    fn n_dofs(&self) -> usize { 1 }
    fn eval_basis(&self, _xi: &[f64], v: &mut [f64]) { v[0] = 1.0; }
    fn eval_grad_basis(&self, _xi: &[f64], g: &mut [f64]) { g[0] = 0.0; g[1] = 0.0; }
    fn quadrature(&self, order: u8) -> QuadratureRule { quad_rule_01(order) }
    fn dof_coords(&self) -> Vec<Vec<f64>> { vec![vec![0.0, 0.0]] }
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
/// `order` for an **L2/DG** space: quad DOFs use MFEM's lexicographic tensor
/// order (`DG_FECollection`), all other element types keep the H1 topological
/// ordering (which MFEM's L2 spaces on simplices also use).
pub fn ref_elem_vol_l2(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    if elem_type == ElementType::Quad4 {
        match order {
            0 => Box::new(P0),
            // MFEM L2_FECollection uses Gauss-Legendre tensor-product basis
            // (BasisType::GaussLegendre), NOT the GLL basis of H1.  QuadL2GL
            // reproduces it bit-identically on [0,1]² with lexicographic DOFs.
            o => Box::new(fem_element::lagrange::QuadL2GL::new(o as usize)),
        }
    } else {
        ref_elem_vol(elem_type, order)
    }
}

/// Reference element for `space`: L2/DG spaces get the lexicographic quad
/// DOF ordering ([`ref_elem_vol_l2`]), H1 spaces the topological one
/// ([`ref_elem_vol_h1`]).
pub(crate) fn ref_elem_vol_for_space<S: FESpace>(
    space: &S,
    elem_type: ElementType,
    order: u8,
) -> Box<dyn ReferenceElement> {
    if space.space_type() == SpaceType::L2 {
        if space.l2_basis() == Some(L2Basis::GaussLobatto) && elem_type == ElementType::Quad4 {
            // MFEM `DG_FECollection(..., BasisType::GaussLobatto)` uses GLL
            // nodes with lexicographic DOFs — `QuadQk::new_lex`, NOT the
            // GL-noded `QuadL2GL` (which matches only `L2_FECollection`'s
            // default `GaussLegendre`).  Using the wrong basis silently
            // changes every element matrix (ex41 regression: M/S/K off by
            // ~6×, IMEX diverged).
            Box::new(fem_element::lagrange::factory::QuadQk::new_lex(order as usize))
        } else {
            ref_elem_vol_l2(elem_type, order)
        }
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
        (ElementType::Tri3 | ElementType::Tri6, 0) => Box::new(P0),
        (ElementType::Tri3 | ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3 | ElementType::Tri6, 2) => Box::new(TriP2),
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
        (ElementType::Tet4, 3) => Box::new(TetP3),
        (ElementType::Tet4, o) => Box::new(fem_element::lagrange::TetPk::new(o as usize)),
        (ElementType::Quad4, 0) => Box::new(P0),
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
        (ElementType::Tri3 | ElementType::Tri6, 0) => Box::new(P0),
        (ElementType::Tri3 | ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3 | ElementType::Tri6, 2) => Box::new(TriP2),
        (ElementType::Tri3 | ElementType::Tri6, 3) => Box::new(TriP3),
        (ElementType::Tri3 | ElementType::Tri6, 4) => Box::new(TriP4),
        (ElementType::Tet4, 1)                           => Box::new(TetP1),
        (ElementType::Tet4, 2)                           => Box::new(TetP2),
        (ElementType::Tet4, 3)                           => Box::new(TetP3),
        (ElementType::Tet4, o)                           => Box::new(fem_element::lagrange::TetPk::new(o as usize)),
        (ElementType::Quad4, 0)                          => Box::new(P0),
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
fn geom_quad_point(_elem_type: ElementType, _order: u8, xi: &[f64]) -> Vec<f64> {
    // All Quad4 solution bases now live on [0,1]^d (QuadQk, order >= 1), and
    // simplex bases share their reference domain with the geometry element,
    // so quadrature points always arrive in the geometry's reference domain.
    xi.to_vec()
}

/// Return the solution reference element for a boundary face.
fn ref_elem_face(face_elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (face_elem_type, order) {
        (ElementType::Line2, 1) => Box::new(SegP1),
        (ElementType::Line2, 2) => Box::new(SegP2),
        (ElementType::Line2, 3) => Box::new(SegP3),
        (ElementType::Line2, 4) => Box::new(SegP4),
        (ElementType::Tri3,  1) => Box::new(TriP1),
        (ElementType::Tri3,  2) => Box::new(TriP2),
        (ElementType::Tri3,  3) => Box::new(TriP3),
        (ElementType::Tri3,  4) => Box::new(TriP4),
        (ElementType::Quad4, 1) => Box::new(QuadQ1),
        (ElementType::Quad4, 2) => Box::new(QuadQ2),
        _ => panic!("ref_elem_face: unsupported (element_type={face_elem_type:?}, order={order})"),
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
fn geo_ref_elem(mesh: &dyn MeshTopology, e: u32) -> Option<Box<dyn ReferenceElement>> {
    let et = mesh.element_type(e);
    let g = mesh.geom_order();
    let is_quad_hex = matches!(et,
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9
        | ElementType::Hex8 | ElementType::Hex20
        | ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18
        | ElementType::Pyramid5 | ElementType::Pyramid13);
    if g == 1 && !is_quad_hex { return None; } // affine P1 simplex
    use fem_element::lagrange::factory::QuadQk;
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
    let order = if g > 1 { g } else { 1 };
    let ft = mesh_type_to_factory(et);
    Some(factory_ref_elem(ft, order))
}

/// Whether this element type has a constant (affine) Jacobian.
///
/// Affine if P1 simplex geometry (`geom_order == 1` and non-tensor-product).
/// Non-affine for curved simplex elements (geom_order > 1) and all tensor-product
/// elements (Quad/Hex) which use isoparametric mapping.
fn is_affine(et: ElementType, geom_order: u8) -> bool {
    if geom_order > 1 { return false; }
    matches!(et, ElementType::Tri3 | ElementType::Tet4 | ElementType::Line2)
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
fn adjugate_2d(j: &DMatrix<f64>) -> DMatrix<f64> {
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
fn adjugate_3d(j: &DMatrix<f64>) -> DMatrix<f64> {
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
fn transform_grads_adj(
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

    // Mixed meshes (e.g. Tet4 + Prism6): the quadrature rule passed in is
    // built from the FIRST element type, which is invalid on other element
    // types (wrong reference domain / weights).  Re-derive the rule from
    // this element's own reference element whenever the type differs.
    let elem_type0 = mesh.element_type(0);
    let elem_type  = mesh.element_type(e);
    let quad_owned;
    let ref_elem_owned;
    let quad: &QuadratureRule = if elem_type == elem_type0 {
        quad
    } else {
        // Mixed meshes: use the caller's quadrature order (like the uniform
        // path) instead of reverse-engineering it from the first element's
        // point count.
        quad_owned = ref_elem_vol_for_space(space, elem_type, order).quadrature(quad_order);
        &quad_owned
    };
    // Mixed meshes: also re-derive the reference element itself (basis,
    // n_dofs) when the element type differs from the first element's.
    let ref_elem: &dyn ReferenceElement = if elem_type == elem_type0 {
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
        Some(ElementTransformation::from_simplex_nodes(mesh, nodes))
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
            let w = quad.weights[q] * tr.det_j().abs();

            ref_elem.eval_basis(xi, &mut scratch.phi);
            ref_elem.eval_grad_basis(xi, &mut scratch.grad_ref);
            transform_grads(tr.jacobian_inv_t(), &scratch.grad_ref, &mut scratch.grad_phys, n_ldofs, dim);

            let xp = tr.map_to_physical(xi);
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
        Some(ElementTransformation::from_simplex_nodes(mesh, nodes))
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
        _ => panic!("unsupported boundary face node count"),
    };
    let ref_elem = ref_elem_face(face_type, order);
    let quad = ref_elem.quadrature(quad_order);

    let face_nodes = mesh.face_nodes(f);
    let (face_j_mag, normal) = face_jacobian_and_normal(mesh, face_nodes, dim);
    let face_tag = mesh.face_tag(f);

    let mut phi = vec![0.0_f64; n_fdofs];
    let mut f_face = vec![0.0_f64; n_fdofs];
    let x0 = mesh.node_coords(face_nodes[0]);

    for (q, xi) in quad.points.iter().enumerate() {
        let w = quad.weights[q] * face_j_mag;
        ref_elem.eval_basis(xi, &mut phi);

        let xp: Vec<f64> = (0..dim)
            .map(|i| {
                let x1 = mesh.node_coords(face_nodes[1]);
                x0[i] + (x1[i] - x0[i]) * xi[0]
            })
            .collect();

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
        _ => panic!("unsupported boundary face node count"),
    };
    let ref_elem = ref_elem_face(face_type, order);
    let quad = ref_elem.quadrature(quad_order);

    let face_nodes = mesh.face_nodes(f);
    let (face_j_mag, normal) = face_jacobian_and_normal(mesh, face_nodes, dim);
    let face_tag = mesh.face_tag(f);

    let mut phi = vec![0.0_f64; n_fdofs];
    let mut k_face = vec![0.0_f64; n_fdofs * n_fdofs];
    let x0 = mesh.node_coords(face_nodes[0]);

    for (q, xi) in quad.points.iter().enumerate() {
        let w = quad.weights[q] * face_j_mag;
        ref_elem.eval_basis(xi, &mut phi);

        let xp: Vec<f64> = (0..dim)
            .map(|i| {
                let x1 = mesh.node_coords(face_nodes[1]);
                x0[i] + (x1[i] - x0[i]) * xi[0]
            })
            .collect();

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
        let mesh   = space.mesh();
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
        let mesh   = space.mesh();
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
        let mesh   = space.mesh();
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

// ─── Face Jacobian and normal (2-D) ──────────────────────────────────────────

/// Compute the face Jacobian magnitude and outward unit normal for a 2-D boundary edge.
///
/// Returns `(|J_face|, n)` where `|J_face|` is the edge length and `n` is the
/// unit outward normal (rotated 90° from the edge tangent, pointing away from
/// the interior by convention `n = (dy, -dx) / |J_face|`).
fn face_jacobian_and_normal(
    mesh:       &dyn MeshTopology,
    face_nodes: &[u32],
    dim:        usize,
) -> (f64, Vec<f64>) {
    assert_eq!(dim, 2, "face_jacobian_and_normal currently only supports 2-D meshes");
    let x0 = mesh.node_coords(face_nodes[0]);
    let x1 = mesh.node_coords(face_nodes[1]);
    let dx = x1[0] - x0[0];
    let dy = x1[1] - x0[1];
    let len = (dx * dx + dy * dy).sqrt();
    // Outward normal convention: rotate tangent (dx,dy) by -90° → (dy, -dx)
    let normal = vec![dy / len, -dx / len];
    (len, normal)
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
        let (elem, _) = mesh.face_elements(f);
        let elem_nodes = mesh.element_nodes(elem);
        let elem_dofs  = space.element_dofs(elem);

        // Find local vertex positions of the two face nodes.
        let pos_a = elem_nodes.iter().position(|&n| n == fn_nodes[0])
            .expect("face node 0 not in element");
        let pos_b = elem_nodes.iter().position(|&n| n == fn_nodes[1])
            .expect("face node 1 not in element");

        let dof_a = elem_dofs[pos_a];
        let dof_b = elem_dofs[pos_b];

        // For TriP2 the edge DOF positions relative to vertex positions are:
        //   edge(v0→v1) = dofs[3],  edge(v1→v2) = dofs[4],  edge(v0→v2) = dofs[5]
        // Generalised: edge DOF for sorted (min_pos, max_pos) in {(0,1),(1,2),(0,2)}.
        let edge_dof = find_edge_dof(elem_nodes, elem_dofs, pos_a, pos_b);

        vec![dof_a, dof_b, edge_dof]
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
}
