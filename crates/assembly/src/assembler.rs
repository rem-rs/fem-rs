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
                QuadQ1, QuadQ2, QuadQ3, QuadQ4, HexQ1},
};
use fem_element::lagrange::factory::{ref_elem as factory_ref_elem, ElemType as FactoryElemType};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{ElementTransformation, element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;

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
    fn quadrature(&self, order: u8) -> QuadratureRule { QuadQ1.quadrature(order) }
    fn dof_coords(&self) -> Vec<Vec<f64>> { vec![vec![0.0, 0.0]] }
}

// ─── Reference element factory ───────────────────────────────────────────────

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
        (ElementType::Quad4, 1)                          => Box::new(QuadQ1),
        (ElementType::Quad4, 2)                          => Box::new(QuadQ2),
        (ElementType::Quad4, 3)                          => Box::new(QuadQ3),
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
fn geom_quad_point(elem_type: ElementType, order: u8, xi: &[f64]) -> Vec<f64> {
    match elem_type {
        ElementType::Quad4 if order >= 4 => xi.to_vec(), // QuadQk on [0,1]^d
        ElementType::Quad4 => xi.iter().map(|x| 0.5 * (x + 1.0)).collect(), // QuadQ1..3 on [-1,1]^d
        _ => xi.to_vec(),
    }
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
/// **Important:** For Quad/Hex elements, uses QuadQ1/HexQ1 on [-1,1]^d to
/// match the quadrature domain of the scalar/vector FE reference elements.
/// Using QuadQk::new(1) / HexQk::new(1) (on [0,1]^d) would give a reference
/// domain mismatch with the FE basis functions and quadrature rules.
fn geo_ref_elem(mesh: &dyn MeshTopology, e: u32) -> Option<Box<dyn ReferenceElement>> {
    let et = mesh.element_type(e);
    let g = mesh.geom_order();
    let is_quad_hex = matches!(et,
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9
        | ElementType::Hex8 | ElementType::Hex20
        | ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18
        | ElementType::Pyramid5 | ElementType::Pyramid13);
    if g == 1 && !is_quad_hex { return None; } // affine P1 simplex
    // Quad/Hex elements use [-1,1]^d reference domain for FE basis and quadrature.
    // The geometry element must match this domain — use QuadQ1/HexQ1 for P1 geometry.
    use fem_element::lagrange::{QuadQ1, HexQ1};
    match et {
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => {
            return if g <= 1 {
                Some(Box::new(QuadQ1) as Box<dyn ReferenceElement>)
            } else {
                Some(factory_ref_elem(FactoryElemType::Quad, g))
            };
        }
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => {
            return if g <= 1 {
                Some(Box::new(HexQ1) as Box<dyn ReferenceElement>)
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
) -> (DMatrix<f64>, f64, Vec<f64>) {
    let n_geo = geo_elem.n_dofs();
    let mut grad_geo = vec![0.0_f64; n_geo * dim];
    let mut phi_geo  = vec![0.0_f64; n_geo];
    geo_elem.eval_grad_basis(xi, &mut grad_geo);
    geo_elem.eval_basis(xi, &mut phi_geo);

    let mut j = DMatrix::<f64>::zeros(dim, dim);
    let mut xp = vec![0.0_f64; dim];

    for k in 0..n_geo {
        let xk = mesh.geom_coords_of(nodes[k]);
        for i in 0..dim {
            xp[i] += phi_geo[k] * xk[i];
            for d in 0..dim {
                j[(i, d)] += xk[i] * grad_geo[k * dim + d];
            }
        }
    }
    let det = j.determinant();
    (j, det, xp)
}

/// Surface element Jacobian info for 2D-in-3D meshes.
///
/// For a 2D surface element embedded in 3D space:
/// - J is a 3×2 matrix (mapping from 2D reference to 3D physical)
/// - G = J^T·J is the 2×2 metric tensor
/// - measure = sqrt(det(G)) = |J₁ × J₂| (surface area element)
/// - chol_Ginv is the lower-triangular Cholesky factor of G⁻¹,
///   so that `(chol_Ginv · ∇_ref φ) · (chol_Ginv · ∇_ref ψ) = ∇_ref φ · G⁻¹ · ∇_ref ψ`
///   which gives the correct surface gradient dot product in the assembly.
///
/// Returns `(measure, chol_Ginv_2x2, x_phys_3d)`.
fn surface_jacobian<M: MeshTopology>(
    mesh: &M,
    nodes: &[u32],
    geo_elem: &dyn ReferenceElement,
    xi: &[f64],
    embed_dim: usize,  // = 3 for Mesh<3>
    tdim: usize,       // = 2 for surface
) -> (f64, DMatrix<f64>, Vec<f64>) {
    let n_geo = geo_elem.n_dofs();
    let mut grad_geo = vec![0.0_f64; n_geo * tdim];
    let mut phi_geo  = vec![0.0_f64; n_geo];
    geo_elem.eval_grad_basis(xi, &mut grad_geo);
    geo_elem.eval_basis(xi, &mut phi_geo);

    // 3×2 Jacobian: J[i][d] = Σ_k x_k[i] · ∂φ_k/∂ξ_d
    let mut j = vec![0.0_f64; embed_dim * tdim]; // column-major: [col0(3), col1(3)]
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

    // G⁻¹ (2×2 inverse metric)
    let inv_det = 1.0 / det_g;
    let a = g11 * inv_det;  // G⁻¹[0][0]
    let b = -g01 * inv_det; // G⁻¹[0][1] = G⁻¹[1][0]
    let c = g00 * inv_det;  // G⁻¹[1][1]

    // Cholesky factor L of G⁻¹ (lower triangular: L·L^T = G⁻¹)
    // L = [[l00, 0], [l10, l11]]
    let l00 = a.sqrt();
    let l10 = b / l00;
    let l11 = (c - b * b / a).sqrt();

    let mut chol = DMatrix::<f64>::zeros(tdim, tdim);
    chol[(0, 0)] = l00;
    chol[(1, 0)] = l10;
    chol[(1, 1)] = l11;

    (measure, chol, xp)
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
    coo: &mut CooMatrix<f64>,
    scratch: &mut ElementScratch,
) {
    let mesh    = space.mesh();
    let edim    = mesh.dim() as usize;   // embedding dimension (2 or 3)
    let tdim    = mesh.topological_dim() as usize; // element dimension (2 for surface)
    let is_surface = edim != tdim;
    let dim     = if is_surface { tdim } else { edim }; // assembly dimension
    let order   = space.element_order(e);

    let elem_type = mesh.element_type(e);
    let ref_elem  = ref_elem_vol(elem_type, order);
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
            let (measure, chol_ginv, xp) =
                surface_jacobian(mesh, geo_nds, geo, &xi_g, edim, tdim);
            let w = quad.weights[q] * measure;

            ref_elem.eval_basis(xi, &mut scratch.phi);
            ref_elem.eval_grad_basis(xi, &mut scratch.grad_ref);
            transform_grads(&chol_ginv, &scratch.grad_ref, &mut scratch.grad_phys, n_ldofs, dim);

            let qp = QpData {
                n_dofs:    n_elem_dofs,
                dim,
                weight:    w,
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
            let (jac_qp, det_qp, xp_qp) =
                isoparametric_jacobian(mesh, geo_nds, geo.as_ref(), &xi_g, dim);
            let w = quad.weights[q] * det_qp.abs();
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

            let qp = QpData {
                n_dofs:    n_elem_dofs,
                dim,
                weight:    w,
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
    rhs: &mut [f64],
    scratch: &mut ElementScratch,
) {
    let mesh    = space.mesh();
    let edim    = mesh.dim() as usize;
    let tdim    = mesh.topological_dim() as usize;
    let is_surface = edim != tdim;
    let dim     = if is_surface { tdim } else { edim };
    let order   = space.element_order(e);

    let elem_type = mesh.element_type(e);
    let ref_elem  = ref_elem_vol(elem_type, order);
    let n_ldofs   = ref_elem.n_dofs();

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
            let (measure, _chol, xp_surf) =
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
            let (jac_qp, det_qp, xp_qp) =
                isoparametric_jacobian(mesh, geo_nds, geo.as_ref(), &xi_g, dim);
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
                accumulate_volume_bilinear_element(space, e, integrators, &quad, &mut local_coo, &mut scratch);
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
) -> Vec<f64> {
    let mesh = space.mesh();
    let n_dofs = space.n_dofs();
    mesh.elem_iter()
        .into_par_iter()
        .fold(
            || (vec![0.0_f64; n_dofs], ElementScratch::new()),
            |(mut local_rhs, mut scratch), e| {
                accumulate_volume_linear_element(space, e, integrators, quad, &mut local_rhs, &mut scratch);
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
        let mesh   = space.mesh();
        let n_dofs = space.n_dofs();

        // Precompute quadrature rule from the first element's type (same for all).
        // Use the ACTUAL solution order: for Quad4 order >= 4 the basis (QuadQk)
        // lives on [0,1]^2 while order <= 3 (QuadQ1/Q2/Q3) lives on [-1,1]^2, and
        // the quadrature domain must match the basis domain.
        let elem_type = mesh.element_type(0);
        let quad = ref_elem_vol(elem_type, space.element_order(0).max(1)).quadrature(quad_order);

        // Estimate raw nnz for COO pre-allocation.
        // Each element contributes `dofs_per_elem^2` triplets.
        // Use first element's n_dofs for uniform-order meshes (common case).
        let elem0    = mesh.element_type(0);
        let ref0     = ref_elem_vol(elem0, space.element_order(0));
        let dofs_per_elem = ref0.n_dofs();
        let est_nnz  = mesh.n_elements() as usize * dofs_per_elem * dofs_per_elem;

        #[cfg(feature = "parallel")]
        {
            if mesh.n_elements() >= assembly_parallel_min_elems() {
                return assemble_bilinear_volume_parallel(space, integrators, &quad);
            }
        }

        let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
        coo.reserve(est_nnz.min(10_000_000)); // cap to avoid giant pre-allocs for hp meshes
        let mut scratch = ElementScratch::new();
        for e in mesh.elem_iter() {
            accumulate_volume_bilinear_element(space, e, integrators, &quad, &mut coo, &mut scratch);
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
        let quad = ref_elem_vol(elem_type, space.element_order(0).max(1)).quadrature(quad_order);

        let elem0   = mesh.element_type(0);
        let ref0    = ref_elem_vol(elem0, space.element_order(0));
        let dofs_per_elem = ref0.n_dofs();
        let est_nnz = n_elems as usize * dofs_per_elem * dofs_per_elem;

        let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
        coo.reserve(est_nnz.min(10_000_000));
        let mut scratch = ElementScratch::new();

        let mut all_dofs: Vec<u32> = Vec::with_capacity(n_elems * dofs_per_elem);
        let mut all_mats: Vec<f64> = Vec::with_capacity(n_elems * dofs_per_elem * dofs_per_elem);
        let mut ldofs = 0;

        for e in mesh.elem_iter() {
            accumulate_volume_bilinear_element(space, e, integrators, &quad, &mut coo, &mut scratch);
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

        // Precompute quadrature rule from the ACTUAL solution order (see assemble_bilinear).
        let elem_type = mesh.element_type(0);
        let quad = ref_elem_vol(elem_type, space.element_order(0).max(1)).quadrature(quad_order);

        #[cfg(feature = "parallel")]
        {
            if mesh.n_elements() >= assembly_parallel_min_elems() {
                return assemble_linear_volume_parallel(space, integrators, &quad);
            }
        }

        let mut rhs = vec![0.0_f64; n_dofs];
        let mut scratch = ElementScratch::new();
        for e in mesh.elem_iter() {
            accumulate_volume_linear_element(space, e, integrators, &quad, &mut rhs, &mut scratch);
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
