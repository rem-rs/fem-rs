//! DG time-domain operator for hyperbolic conservation laws.
//!
//! Provides [`FluxFunction`] trait, [`EulerFlux`], [`RusanovFlux`],
//! and [`DgHyperbolicConservationLaws`].
//!
//! ## Reference
//! MFEM examples/ex18.hpp — DGHyperbolicConservationLaws

use nalgebra as na;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use fem_element::reference::ReferenceElement;
use fem_element::quadrature::gauss_legendre_01;
use fem_element::lagrange::factory::TriPk;
use fem_element::lagrange::tri::{TriP1};
use fem_element::lagrange::QuadL2GL;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::transformation::element_jacobian_at;

use super::dg_base::face_point_geom;

/// Element shape for dispatching Tri3 vs Quad4 code paths.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ElemShape { Tri, Quad }

/// Physical flux function for hyperbolic conservation laws.
pub trait FluxFunction: Send + Sync {
    fn num_equations(&self) -> usize;
    fn compute_flux(&self, state: &[f64], point: &[f64], flux_out: &mut [f64]);
    fn max_speed(&self, state: &[f64], normal: &[f64]) -> f64;
    fn numerical_flux(&self, ql: &[f64], qr: &[f64], normal: &[f64]) -> Vec<f64>;
}

// ─── EulerFlux ──────────────────────────────────────────────────────────────────

/// Convert conserved variables to primitive variables.
fn cons_to_prim(q: &[f64], gamma: f64) -> (f64, f64, f64, f64) {
    let rho = q[0].max(1e-14);
    let u = q[1] / rho;
    let v = q[2] / rho;
    let ke = 0.5 * rho * (u * u + v * v);
    let p = ((gamma - 1.0) * (q[3] - ke)).max(1e-14);
    (rho, u, v, p)
}

/// Convert primitive variables to conserved variables.
#[allow(dead_code)]
fn prim_to_cons(rho: f64, u: f64, v: f64, p: f64, gamma: f64) -> [f64; 4] {
    let e = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);
    [rho, rho * u, rho * v, e]
}

/// 2-D compressible Euler flux (4 equations).
///
/// Conserved variables: [ρ, ρu, ρv, E]
/// γ (specific heat ratio) defaults to 1.4 (air).
pub struct EulerFlux {
    pub gamma: f64,
}

impl Default for EulerFlux {
    fn default() -> Self {
        Self { gamma: 1.4 }
    }
}

impl FluxFunction for EulerFlux {
    fn num_equations(&self) -> usize {
        4
    }

    fn compute_flux(&self, state: &[f64], _point: &[f64], flux_out: &mut [f64]) {
        let (rho, u, v, p) = cons_to_prim(state, self.gamma);
        // flux_out interleaved by dim then equation:
        //   [F_x[ρ], F_y[ρ], F_x[ρu], F_y[ρu], F_x[ρv], F_y[ρv], F_x[E], F_y[E]]
        let E = state[3];
        flux_out[0] = state[1];                         // F_x[ρ]:  ρu
        flux_out[1] = state[2];                         // F_y[ρ]:  ρv
        flux_out[2] = rho * u * u + p;                  // F_x[ρu]: ρu² + p
        flux_out[3] = rho * u * v;                      // F_y[ρu]: ρuv
        flux_out[4] = rho * u * v;                      // F_x[ρv]: ρuv
        flux_out[5] = rho * v * v + p;                  // F_y[ρv]: ρv² + p
        flux_out[6] = u * (E + p);                      // F_x[E]:  u(E + p)
        flux_out[7] = v * (E + p);                      // F_y[E]:  v(E + p)
    }

    fn max_speed(&self, state: &[f64], normal: &[f64]) -> f64 {
        let (rho, u, v, p) = cons_to_prim(state, self.gamma);
        let a = (self.gamma * p / rho).sqrt();
        // MFEM EulerFlux::ComputeFluxDotN (fem/hyperbolic.cpp): the maximum
        // characteristic speed is the NORMAL fluid speed |u·n|/|n| plus the
        // sound speed — NOT the full speed |u| + a.
        let un = u * normal[0] + v * normal[1];
        let nnorm = (normal[0] * normal[0] + normal[1] * normal[1]).sqrt();
        let speed = un.abs() / nnorm;
        speed + a
    }

    fn numerical_flux(&self, ql: &[f64], qr: &[f64], normal: &[f64]) -> Vec<f64> {
        let mut fl = [0.0_f64; 8];
        let mut fr = [0.0_f64; 8];
        self.compute_flux(ql, &[0.0, 0.0], &mut fl);
        self.compute_flux(qr, &[0.0, 0.0], &mut fr);
        // F_n = F_x · n_x + F_y · n_y  (component-wise for each equation)
        let mut fnl = [0.0_f64; 4];
        let mut fnr = [0.0_f64; 4];
        for eq in 0..4 {
            fnl[eq] = fl[eq * 2] * normal[0] + fl[eq * 2 + 1] * normal[1];
            fnr[eq] = fr[eq * 2] * normal[0] + fr[eq * 2 + 1] * normal[1];
        }
        let c = self.max_speed(ql, normal).max(self.max_speed(qr, normal));
        // ½(F_n(L) + F_n(R)) - ½·c·(qR - qL)
        let mut f = vec![0.0_f64; 4];
        for eq in 0..4 {
            f[eq] = 0.5 * (fnl[eq] + fnr[eq]) - 0.5 * c * (qr[eq] - ql[eq]);
        }
        f
    }
}

// ─── RusanovFlux ────────────────────────────────────────────────────────────────

/// Rusanov (local Lax-Friedrichs) numerical flux.
///
/// Wraps any `FluxFunction` — delegates `compute_flux` and `max_speed`
/// to the inner function, and `numerical_flux` calls `inner.numerical_flux`.
pub struct RusanovFlux<F: FluxFunction> {
    pub inner: F,
}

impl<F: FluxFunction> FluxFunction for RusanovFlux<F> {
    fn num_equations(&self) -> usize {
        self.inner.num_equations()
    }

    fn compute_flux(&self, state: &[f64], point: &[f64], flux_out: &mut [f64]) {
        self.inner.compute_flux(state, point, flux_out);
    }

    fn max_speed(&self, state: &[f64], normal: &[f64]) -> f64 {
        self.inner.max_speed(state, normal)
    }

    fn numerical_flux(&self, ql: &[f64], qr: &[f64], normal: &[f64]) -> Vec<f64> {
        self.inner.numerical_flux(ql, qr, normal)
    }
}

// ─── InteriorFace ─────────────────────────────────────────────────────────────

/// Face data for an interior face shared by two elements.
///
/// D799-3: the geometry is the **isoparametric** face geometry of MFEM's
/// `HyperbolicFormIntegrator::AssembleFaceVector` (hyperbolic.cpp:177) — per
/// face quadrature point, `Tr.SetAllIntPoints(&ip)` +
/// `nor = CalcOrtho(Tr.Jacobian())`.  `nor_qp` holds the *unit* normals
/// outward from `elem_l` and `qp_weights` the matching measures `ip.weight·|nor|`
/// (MFEM carries `|nor|` inside `nor` and uses the bare `ip.weight`; the product
/// `qp_weights[q]·nor_qp[q]` is exactly `ip.weight·nor`), so a straight edge
/// gives the same numbers as the pre-fix chord route.
#[allow(dead_code)]
struct InteriorFace {
    elem_l: usize,
    elem_r: usize,
    /// Unit outward normal from `elem_l` at each face quadrature point.
    nor_qp: Vec<[f64; 2]>,
    qp_ref_l: Vec<[f64; 2]>,
    qp_ref_r: Vec<[f64; 2]>,
    qp_weights: Vec<f64>,
    basis_l: Vec<Vec<f64>>,
    basis_r: Vec<Vec<f64>>,
}

// ─── BoundaryFace ─────────────────────────────────────────────────────────────

/// Face data for a boundary face (same isoparametric convention as
/// `InteriorFace`; the mirror-wall reflection needs a *unit* normal).
#[allow(dead_code)]
struct BoundaryFace {
    elem: usize,
    /// Unit outward normal from `elem` at each face quadrature point.
    nor_qp: Vec<[f64; 2]>,
    qp_ref: Vec<[f64; 2]>,
    qp_weights: Vec<f64>,
    basis: Vec<Vec<f64>>,
}

// ─── DgHyperbolicConservationLaws ─────────────────────────────────────────────

/// DG time-domain operator for hyperbolic conservation laws on triangle meshes.
///
/// Provides element-wise inverse mass matrix, weak divergence, and face-based
/// flux assembly for hyperbolic systems (e.g., Euler equations).
pub struct DgHyperbolicConservationLaws {
    n_elems: usize,
    dofs_per_elem: usize,
    n_eq: usize,
    dim: usize,
    total_dofs: usize,
    invmass: Vec<na::DMatrix<f64>>,
    /// Preassembled weak divergence ∫φ_i·∇φ_j (trial space ByDim layout:
    /// column j*dim+d). Used by `mult` when `preassemble_weakdiv` is set
    /// (MFEM ex18.hpp ComputeWeakDivergence / AddMult_a_ABt path).
    weakdiv: Vec<na::DMatrix<f64>>,
    /// Per-element, per-QP **isoparametric** volume geometry `(det J, J⁻ᵀ)` for
    /// the matrix-free volume term (MFEM `Tr.Weight()` + `CalcPhysDShape`).
    /// Empty when `preassemble_weakdiv` (that path never touches it).
    vol_qp_geom: Vec<Vec<(f64, [f64; 4])>>,
    ref_elem: Box<dyn ReferenceElement>,
    elem_shape: ElemShape,
    flux: Box<dyn FluxFunction>,
    interior_faces: Vec<InteriorFace>,
    boundary_faces: Vec<BoundaryFace>,
    max_char_speed: std::cell::Cell<f64>,
    z: RefCell<Vec<f64>>,
    preassemble_weakdiv: bool,
}

// ─── Helper functions ─────────────────────────────────────────────────────────

fn make_ref_elem(mesh: &dyn MeshTopology, order: u8) -> (Box<dyn ReferenceElement>, ElemShape) {
    let shape = if mesh.element_type(0) == ElementType::Quad4 {
        ElemShape::Quad
    } else {
        ElemShape::Tri
    };
    match shape {
        ElemShape::Quad => {
            assert_eq!(order, 1, "Quad4 only supports order=1 currently");
            // MFEM DG_FECollection(order, dim, BasisType::GaussLegendre) uses
            // the Gauss-Legendre nodal basis on [0,1]² — NOT the equally
            // spaced QuadQ1.  With GL nodes the mass matrix is diagonal
            // (C++ invmass = 36·I on this mesh; QuadQ1 gave a full 144/-72
            // matrix → 10× larger dudt → NaN).
            (Box::new(QuadL2GL::new(1)), shape)
        }
        ElemShape::Tri => {
            match order {
                1 => (Box::new(TriP1), shape),
                2 => (Box::new(TriPk::new(2)), shape),
                3 => (Box::new(TriPk::new(3)), shape),
                _ => (Box::new(TriP1), shape),
            }
        }
    }
}

/// Element Jacobian at a quadrature point, **SIGNED** — D696 batch 4: the
/// returned det is the quadrature measure consumed as `w_q·detJ` (MFEM
/// hyperbolic.cpp:103/165 `ip.weight * Tr.Weight()`, signed).
///
/// D799-3: MFEM's `Tr.Weight()`/`Tr.Jacobian()` are the element's
/// **isoparametric** transformation — the order-`g` curved map on a curved
/// mesh, the P1/bilinear map on a straight one.  The pre-fix helpers built the
/// Jacobian from raw corner differences (`tri3_jac_at_qp`) or from a single
/// centroid bilinear value (`quad4_jac_at_qp` + `elem_centroid_jac`), which
/// agreed with MFEM only on affine tets and parallelograms.  Returns
/// `(detJ, J^{-T})`.
// MFEM: ElementTransformation::Weight + Jacobian
fn elem_jac_at_qp(mesh: &dyn MeshTopology, elem: u32, xi: &[f64], dim: usize) -> (f64, [f64; 4]) {
    let (jac, _xp) = element_jacobian_at(mesh, elem, xi, dim);
    let det = jac.determinant();
    let inv_det = 1.0 / det;
    // J^{-1} = (1/det)·[[J22,-J12],[-J21,J11]] for the 2×2 case, then transpose.
    (det, [jac[(1, 1)] * inv_det, -jac[(1, 0)] * inv_det, -jac[(0, 1)] * inv_det, jac[(0, 0)] * inv_det])
}

/// Compute element-wise inverse mass matrix M_e⁻¹ in physical space.
/// M_e[i,j] = Σ_q w_q · detJ(ξ_q) · φ_i(ξ_q) · φ_j(ξ_q)   (detJ signed — D696)
fn compute_inv_mass(mesh: &dyn MeshTopology, ref_elem: &dyn ReferenceElement, n_elems: usize, shape: ElemShape) -> Vec<na::DMatrix<f64>> {
    let dp = ref_elem.n_dofs();
    let dim = mesh.dim() as usize;
    let _ = shape;
    let q_order = 2 * ref_elem.order();
    let qr = ref_elem.quadrature(q_order);
    let n_qp = qr.n_points();
    let mut phi = vec![0.0; dp];
    let mut invmass = Vec::with_capacity(n_elems);
    for e in 0..n_elems {
        let mut m = na::DMatrix::<f64>::zeros(dp, dp);
        for q in 0..n_qp {
            let xi = &qr.points[q];
            let det_j = elem_jac_at_qp(mesh, e as u32, xi, dim).0;
            let w = qr.weights[q] * det_j;
            ref_elem.eval_basis(xi, &mut phi);
            for i in 0..dp {
                for j in 0..dp {
                    m[(i, j)] += w * phi[i] * phi[j];
                }
            }
        }
        let chol = m.cholesky().expect("Mass matrix must be SPD");
        invmass.push(chol.inverse());
    }
    invmass
}

/// Compute element-wise weak divergence matrix in physical space.
/// weakdiv[e][i, j*dim + d] = Σ_q w_q · detJ(ξ_q) · φ_i(ξ_q) · (J^{-T}(ξ_q) · ∇ξ_φ_j)_d
fn compute_weak_div(mesh: &dyn MeshTopology, ref_elem: &dyn ReferenceElement, n_elems: usize, shape: ElemShape) -> Vec<na::DMatrix<f64>> {
    let dp = ref_elem.n_dofs();
    let dim = mesh.dim() as usize;
    let _ = shape;
    let q_order = 2 * ref_elem.order();
    let qr = ref_elem.quadrature(q_order);
    let n_qp = qr.n_points();
    let mut phi = vec![0.0; dp];
    let mut gphi = vec![0.0; dp * dim];
    let mut weakdiv = Vec::with_capacity(n_elems);
    for e in 0..n_elems {
        let mut wd = na::DMatrix::<f64>::zeros(dp, dp * dim);
        for q in 0..n_qp {
            let xi = &qr.points[q];
            let (det_j, jit) = elem_jac_at_qp(mesh, e as u32, xi, dim);
            let w = qr.weights[q] * det_j;
            ref_elem.eval_basis(xi, &mut phi);
            ref_elem.eval_grad_basis(xi, &mut gphi);
            for i in 0..dp {
                let mut gphys_i = [0.0; 2];
                for d in 0..dim {
                    for k in 0..dim {
                        gphys_i[d] += jit[d * dim + k] * gphi[i * dim + k];
                    }
                }
                for j in 0..dp {
                    for d in 0..dim {
                        wd[(i, j * dim + d)] += w * phi[j] * gphys_i[d];
                    }
                }
            }
        }
        weakdiv.push(wd);
    }
    weakdiv
}

/// Tri3 face patterns: [local_node_a, local_node_b] for faces 0, 1, 2 —
/// listed counter-clockwise, so `(a, b)` runs along the element's own edge.
const TRI3_FACES: [[usize; 2]; 3] = [[0, 1], [1, 2], [2, 0]];

/// Quad4 face patterns: [local_node_a, local_node_b] for faces 0, 1, 2, 3 —
/// counter-clockwise, like [`TRI3_FACES`].
const QUAD4_FACES: [[usize; 2]; 4] = [[0, 1], [1, 2], [2, 3], [3, 0]];

/// Local face patterns of the element's shape.
fn shape_faces(shape: ElemShape) -> &'static [[usize; 2]] {
    match shape {
        ElemShape::Quad => &QUAD4_FACES,
        ElemShape::Tri => &TRI3_FACES,
    }
}

/// Per-quadrature-point **isoparametric** face geometry of element `elem`'s
/// local face `lf`, composed through that element's own map — MFEM's
/// `FaceElementTransformations` (`Tr.SetAllIntPoints(&ip)` +
/// `nor = CalcOrtho(Tr.Jacobian())`, hyperbolic.cpp:232-255).
///
/// Returns `(unit outward normals, measures ipw·|nor|, element reference
/// points)`.  MFEM keeps `|nor|` inside its `nor` and integrates with the bare
/// `ip.weight`; folding the magnitude into the weight (`ipw·|nor|`, with the
/// `[0,1]` face rule) is the same number and leaves a unit normal for the
/// flux and for the reflecting-wall reflection.  `reverse` flips the face
/// parameterisation (the neighbour's edge runs the other way), keeping both
/// elements sampled at the **same** physical face point.
///
/// On a straight edge `|nor|` is the chord length, so every value below is the
/// pre-fix one; on a curved edge it follows the isoparametric edge.
// MFEM: FaceElementTransformations::Jacobian + CalcOrtho
fn face_qp_geom(
    mesh: &dyn MeshTopology,
    elem: u32,
    lf: usize,
    pts: &[f64],
    wts: &[f64],
    reverse: bool,
) -> (Vec<[f64; 2]>, Vec<f64>, Vec<[f64; 2]>) {
    let shape = if mesh.element_nodes(elem).len() == 4 { ElemShape::Quad } else { ElemShape::Tri };
    let en = mesh.element_nodes(elem);
    let (ia, ib) = (shape_faces(shape)[lf][0], shape_faces(shape)[lf][1]);
    let (na, nb) = (en[ia], en[ib]);
    let mut nors = Vec::with_capacity(pts.len());
    let mut ws = Vec::with_capacity(pts.len());
    let mut eips = Vec::with_capacity(pts.len());
    for q in 0..pts.len() {
        let t = if reverse { 1.0 - pts[q] } else { pts[q] };
        let g = face_point_geom(mesh, elem, na, nb, t);
        let nrm = (g.nor[0] * g.nor[0] + g.nor[1] * g.nor[1]).sqrt().max(1e-30);
        nors.push([g.nor[0] / nrm, g.nor[1] / nrm]);
        ws.push(wts[q] * nrm);
        eips.push(g.eip);
    }
    (nors, ws, eips)
}

/// Build interior and boundary face structures.
/// Supports Tri3 (3 faces) and Quad4 (4 faces) meshes.
/// Detects periodic pairs from boundary faces with opposite normals.
fn build_faces(mesh: &dyn MeshTopology, ref_elem: &dyn ReferenceElement) -> (Vec<InteriorFace>, Vec<BoundaryFace>) {
    let dp = ref_elem.n_dofs();
    let n_elems = mesh.n_elements() as u32;
    let n_qp = ((2 * ref_elem.order() + 1) as usize).min(4).max(1);
    let (face_pts, face_wts) = gauss_legendre_01(n_qp);

    // Detect element type from first element's node count
    let shape = if mesh.element_nodes(0).len() == 4 { ElemShape::Quad } else { ElemShape::Tri };
    let face_patterns: &[[usize; 2]] = shape_faces(shape);

    // Record all element edges
    struct ElemEdge {
        nodes: [u32; 2],
        elem: u32,
        local_face: u8,
    }
    let mut edge_list: Vec<ElemEdge> = Vec::new();
    for e in 0..n_elems {
        let enodes = mesh.element_nodes(e);
        for (lf, &[na, nb]) in face_patterns.iter().enumerate() {
            let (n0, n1) = (enodes[na].min(enodes[nb]), enodes[na].max(enodes[nb]));
            edge_list.push(ElemEdge {
                nodes: [n0, n1],
                elem: e,
                local_face: lf as u8,
            });
        }
    }

    // Group by sorted node pair
    let mut edge_map: HashMap<(u32, u32), Vec<(u32, u8)>> = HashMap::new();
    for ee in &edge_list {
        edge_map
            .entry((ee.nodes[0], ee.nodes[1]))
            .or_default()
            .push((ee.elem, ee.local_face));
    }

    let mut interior = Vec::new();
    let mut boundary = Vec::new();
    let mut visited = HashSet::new();
    let mut unpaired: Vec<(u32, u8)> = Vec::new();

    for ee in &edge_list {
        let key = (ee.nodes[0], ee.nodes[1]);
        if visited.contains(&(ee.elem, ee.local_face, key)) {
            continue;
        }
        let entries = &edge_map[&key];
        if entries.len() == 2 {
            let (e0, f0) = entries[0];
            let (e1, f1) = entries[1];
            visited.insert((e0, f0, key));
            visited.insert((e1, f1, key));
            let (elem_l, face_l, elem_r, face_r) = if e0 < e1 {
                (e0 as usize, f0, e1 as usize, f1)
            } else {
                (e1 as usize, f1, e0 as usize, f0)
            };

            // D799-3: the per-QP isoparametric face geometry of each side.  The
            // flux uses Elem1's (the left element's) `nor`, exactly as MFEM's
            // `AssembleFaceVector` does with `Tr` built from Elem1.
            let (nor_l, w_l, ref_l) =
                face_qp_geom(mesh, elem_l as u32, face_l as usize, &face_pts, &face_wts, false);
            // Orientation of the neighbour's edge relative to the left's: the
            // neighbour's own local face starts at a different node when the two
            // edges run in opposite directions.
            let l_first = mesh.element_nodes(elem_l as u32)[face_patterns[face_l as usize][0]];
            let r_first = mesh.element_nodes(elem_r as u32)[face_patterns[face_r as usize][0]];
            let reverse_r = l_first != r_first;
            let (_nor_r, _w_r, ref_r) =
                face_qp_geom(mesh, elem_r as u32, face_r as usize, &face_pts, &face_wts, reverse_r);

            let mut basis_l = Vec::with_capacity(n_qp);
            let mut basis_r = Vec::with_capacity(n_qp);
            let mut phi = vec![0.0; dp];
            for q in 0..n_qp {
                ref_elem.eval_basis(&ref_l[q], &mut phi);
                basis_l.push(phi.clone());
                ref_elem.eval_basis(&ref_r[q], &mut phi);
                basis_r.push(phi.clone());
            }

            interior.push(InteriorFace {
                elem_l,
                elem_r,
                nor_qp: nor_l,
                qp_ref_l: ref_l,
                qp_ref_r: ref_r,
                qp_weights: w_l,
                basis_l,
                basis_r,
            });
        } else {
            let (elem, lf) = entries[0];
            if !visited.contains(&(elem, lf, key)) {
                visited.insert((elem, lf, key));
                unpaired.push((elem, lf));
            }
        }
    }

    // Boundary face geometry for all unpaired edges.  `pair_normal` is the
    // chord normal of the edge's two **corner** nodes — used only by the
    // periodic-pairing heuristic below (a direction test), never as a measure;
    // the assembled flux uses the isoparametric per-QP normals.
    struct BoundInfo { elem: u32, lf: u8, pair_normal: [f64; 2] }
    let mut bound_info: Vec<BoundInfo> = Vec::new();
    for &(elem, lf) in &unpaired {
        let enodes = mesh.element_nodes(elem);
        let lfp = face_patterns[lf as usize];
        let (na, nb) = (enodes[lfp[0]], enodes[lfp[1]]);
        let pa = mesh.node_coords(na);
        let pb = mesh.node_coords(nb);
        let (dx, dy) = (pb[0] - pa[0], pb[1] - pa[1]);
        let length = (dx * dx + dy * dy).sqrt().max(1e-30);
        bound_info.push(BoundInfo { elem, lf, pair_normal: [dy / length, -dx / length] });
    }

    // Detect periodic pairs among boundary faces
    let periodic_idx: Vec<(usize, usize)> = {
        let unpaired_with_normals: Vec<(u32, u8, [f64;2])> = bound_info.iter()
            .map(|b| (b.elem, b.lf, b.pair_normal)).collect();
        detect_periodic_pairs(&unpaired_with_normals)
    };

    // Track which bound_info entries are consumed by periodic pairing
    let mut used_by_periodic = vec![false; bound_info.len()];
    for &(i, j) in &periodic_idx {
        used_by_periodic[i] = true;
        used_by_periodic[j] = true;
        // Create periodic interior face from pair (i, j)
        let bi = &bound_info[i];
        let bj = &bound_info[j];
        let (elem_l, elem_r, face_l, face_r) = {
            // Use the element with smaller index as L, normal outward from L
            if bi.elem < bj.elem {
                (bi.elem as usize, bj.elem as usize, bi.lf, bj.lf)
            } else {
                (bj.elem as usize, bi.elem as usize, bj.lf, bi.lf)
            }
        };
        let (nor_l, w_l, ref_l) =
            face_qp_geom(mesh, elem_l as u32, face_l as usize, &face_pts, &face_wts, false);
        // For a periodic pair the two faces are opposite sides of the domain, so
        // the neighbour's face quadrature points are mirrored.
        let (_nor_r, _w_r, ref_r) =
            face_qp_geom(mesh, elem_r as u32, face_r as usize, &face_pts, &face_wts, true);
        let mut basis_l = Vec::with_capacity(n_qp);
        let mut basis_r = Vec::with_capacity(n_qp);
        let mut phi_l = vec![0.0; dp];
        let mut phi_r = vec![0.0; dp];
        for q in 0..n_qp {
            ref_elem.eval_basis(&ref_l[q], &mut phi_l);
            basis_l.push(phi_l.clone());
            ref_elem.eval_basis(&ref_r[q], &mut phi_r);
            basis_r.push(phi_r.clone());
        }
        interior.push(InteriorFace {
            elem_l, elem_r,
            nor_qp: nor_l,
            qp_ref_l: ref_l, qp_ref_r: ref_r, qp_weights: w_l,
            basis_l, basis_r,
        });
    }

    // Remaining unpaired boundary faces become boundary faces
    for (idx, bi) in bound_info.iter().enumerate() {
        if used_by_periodic[idx] { continue; }
        let (nor_qp, qp_w, qp_ref) =
            face_qp_geom(mesh, bi.elem, bi.lf as usize, &face_pts, &face_wts, false);
        let mut basis = Vec::with_capacity(n_qp);
        let mut phi = vec![0.0; dp];
        for q in 0..n_qp {
            ref_elem.eval_basis(&qp_ref[q], &mut phi);
            basis.push(phi.clone());
        }
        boundary.push(BoundaryFace {
            elem: bi.elem as usize,
            nor_qp,
            qp_ref,
            qp_weights: qp_w,
            basis,
        });
    }

    (interior, boundary)
}

/// Detect periodic face pairs from a list of boundary face candidates.
///
/// Groups boundary faces by their normal direction, then pairs faces from
/// opposite sides (normals that are negatives of each other).  This handles
/// the standard periodic-square.mesh case.
fn detect_periodic_pairs(unpaired: &[(u32, u8, [f64;2])]) -> Vec<(usize, usize)> {
    // Group by normal direction (quantized to ±x, ±y).
    // For a square mesh periodic in both directions, opposite sides
    // have normals that are exact opposites.
    let eps = 1e-10_f64;
    let mut pairs = Vec::new();
    let n = unpaired.len();
    let mut used = vec![false; n];
    for i in 0..n {
        if used[i] { continue; }
        let ni = &unpaired[i].2;
        for j in (i+1)..n {
            if used[j] { continue; }
            let nj = &unpaired[j].2;
            // Check if normals are opposites: ni ≈ -nj
            if (ni[0] + nj[0]).abs() < eps && (ni[1] + nj[1]).abs() < eps {
                pairs.push((i, j));
                used[i] = true;
                used[j] = true;
                break;
            }
        }
    }
    pairs
}


// ─── DgHyperbolicConservationLaws impl ────────────────────────────────────────

impl DgHyperbolicConservationLaws {
    /// Construct a new DG hyperbolic conservation law operator.
    pub fn new(
        mesh: &dyn MeshTopology,
        order: u8,
        flux_fn: Box<dyn FluxFunction>,
        preassemble_weakdiv: bool,
    ) -> Self {
        let n_elems = mesh.n_elements();
        let dim = mesh.dim() as usize;
        let n_eq = flux_fn.num_equations();
        let (ref_elem, elem_shape) = make_ref_elem(mesh, order);
        let dofs_per_elem = ref_elem.n_dofs();
        let total_dofs = n_elems * dofs_per_elem * n_eq;
        let invmass = compute_inv_mass(mesh, &*ref_elem, n_elems, elem_shape);
        let weakdiv = if preassemble_weakdiv {
            compute_weak_div(mesh, &*ref_elem, n_elems, elem_shape)
        } else {
            Vec::new()
        };
        let (interior_faces, boundary_faces) = build_faces(mesh, &*ref_elem);
        // Matrix-free volume term: per-QP isoparametric geometry, built once.
        let vol_qp_geom = if preassemble_weakdiv {
            Vec::new()
        } else {
            let qr = ref_elem.quadrature(2 * ref_elem.order());
            (0..n_elems as u32)
                .map(|e| {
                    qr.points
                        .iter()
                        .map(|xi| elem_jac_at_qp(mesh, e, xi, dim))
                        .collect()
                })
                .collect()
        };
        Self {
            n_elems,
            dofs_per_elem,
            n_eq,
            dim,
            total_dofs,
            invmass,
            weakdiv,
            vol_qp_geom,
            ref_elem,
            elem_shape,
            flux: flux_fn,
            interior_faces,
            boundary_faces,
            max_char_speed: std::cell::Cell::new(0.0),
            z: RefCell::new(vec![0.0; total_dofs]),
            preassemble_weakdiv,
        }
    }

    /// Total number of degrees of freedom.
    pub fn n_dofs(&self) -> usize {
        self.total_dofs
    }

    /// Maximum characteristic speed across all faces (updated during Mult).
    pub fn max_char_speed(&self) -> f64 {
        self.max_char_speed.get()
    }

    /// Compute the DG update: dudt = M⁻¹ (face_fluxes - Div·F(u)).
    ///
    /// `u` is the solution vector in **byNODES (DOF-major)** layout:
    /// `index = e * dp * nq + i * nq + eq`
    /// where `e` = element, `i` = local DOF, `eq` = equation index.
    ///
    /// Algorithm:
    /// 1. Reset workspace `z` and `max_char_speed`.
    /// 2. Interior faces: add numerical flux contribution (±f_hat).
    /// 3. Boundary faces: reflecting wall BC (mirror normal velocity).
    /// 4. Volume term: `z += weakdiv[e] · F_col` (if preassembled).
    /// 5. Apply inverse mass: `dudt_e = invmass[e] · z_e`.
    pub fn mult(&self, u: &[f64], dudt: &mut [f64]) {
        let dp = self.dofs_per_elem;
        let nq = self.n_eq;
        let dim = self.dim;

        // 1. Reset workspace
        let mut z = self.z.borrow_mut();
        z.fill(0.0);
        self.max_char_speed.set(0.0);

        // 2. Interior face flux contributions.
        // MFEM `HyperbolicFormIntegrator::AssembleFaceVector`
        // (hyperbolic.cpp:177-270) per face quadrature point:
        //   Tr.SetAllIntPoints(&ip); CalcOrtho(Tr.Jacobian(), nor);
        //   fluxN = numFlux.Eval(state1, state2, nor, Tr)   // nor NOT unit
        //   elvect1 -= ip.weight·sign·shape1·fluxN ;  elvect2 += ip.weight·sign·shape2·fluxN
        // with the `[0,1]` face rule (`Σw = 1/2` on a triangle).  `nor` enters
        // the flux linearly, so carrying a unit normal plus `qp_weights = ipw·|nor|`
        // is the same product; `nor_qp` is that unit normal, per quadrature
        // point and outward from Elem1.
        let mut uL = vec![0.0; nq];
        let mut uR = vec![0.0; nq];
        for face in &self.interior_faces {
            let baseL = face.elem_l * dp * nq;
            let baseR = face.elem_r * dp * nq;
            for q in 0..face.qp_weights.len() {
                uL.fill(0.0);
                uR.fill(0.0);
                for eq in 0..nq {
                    for i in 0..dp {
                        uL[eq] += face.basis_l[q][i] * u[baseL + i * nq + eq];
                        uR[eq] += face.basis_r[q][i] * u[baseR + i * nq + eq];
                    }
                }
                let nor = &face.nor_qp[q];
                let cL = self.flux.max_speed(&uL, nor);
                let cR = self.flux.max_speed(&uR, nor);
                let c = cL.max(cR);
                if c > self.max_char_speed.get() { self.max_char_speed.set(c); }
                let f_hat = self.flux.numerical_flux(&uL, &uR, nor);
                let w = face.qp_weights[q];
                // Form 2: face = -ĝ·[[v]] = +ĝ·v_L - ĝ·v_R
                for eq in 0..nq {
                    let fw = w * f_hat[eq];
                    for i in 0..dp {
                        z[baseL + i * nq + eq] -= fw * face.basis_l[q][i];
                        z[baseR + i * nq + eq] += fw * face.basis_r[q][i];
                    }
                }
            }
        }

        // 3. Boundary faces (reflecting wall BC).  The mirror is built from the
        // **unit** normal (a scaled normal would not reflect), which is why the
        // face data keeps `nor_qp` normalised and the measure separate.
        let mut u_mirror = vec![0.0; nq];
        for face in &self.boundary_faces {
            let base = face.elem * dp * nq;
            for q in 0..face.qp_weights.len() {
                uL.fill(0.0);
                for eq in 0..nq {
                    for i in 0..dp {
                        uL[eq] += face.basis[q][i] * u[base + i * nq + eq];
                    }
                }
                let nor = &face.nor_qp[q];
                let nx = nor[0];
                let ny = nor[1];
                let vn = uL[1] * nx + uL[2] * ny;
                u_mirror[0] = uL[0];
                u_mirror[1] = uL[1] - 2.0 * vn * nx;
                u_mirror[2] = uL[2] - 2.0 * vn * ny;
                u_mirror[3] = uL[3];
                let c = self.flux.max_speed(&uL, nor)
                    .max(self.flux.max_speed(&u_mirror, nor));
                if c > self.max_char_speed.get() { self.max_char_speed.set(c); }
                let f_hat = self.flux.numerical_flux(&uL, &u_mirror, nor);
                let w = face.qp_weights[q];
                // Form 2: boundary face = -ĝ·v (only L side, no R element)
                for eq in 0..nq {
                    let fw = w * f_hat[eq];
                    for i in 0..dp {
                        z[base + i * nq + eq] -= fw * face.basis[q][i];
                    }
                }
            }
        }

        // 4. Volume term (Form 2: +∫ F·∇v).
        if self.preassemble_weakdiv {
            // Preassembled weak divergence × node flux — MFEM
            // DGHyperbolicConservationLaws::Mult weakdiv path
            // (ex18.hpp ComputeWeakDivergence): F(u) is evaluated at each
            // DOF *node* j, then contracted with the preassembled
            // ∫φ_i·∇φ_j matrix (trial space ByDim):
            //   z(r, eq) += Σ_j Σ_d weakdiv[r, j*dim+d] · F(u_j)[eq, d]
            // (cf. mfem::AddMult_a_ABt(1.0, weakdiv[i], flux, current_zmat)).
            // For nonlinear fluxes this is NOT equivalent to quadrature of
            // F(u(x_q)) — on periodic meshes the per-QP form fails to cancel
            // against the face fluxes and goes NaN.
            let mut flux_node = vec![0.0; nq * dim];
            for e in 0..self.n_elems {
                let base = e * dp * nq;
                let wd = &self.weakdiv[e];
                let mut state = vec![0.0; nq];
                for j in 0..dp {
                    for eq in 0..nq {
                        state[eq] = u[base + j * nq + eq];
                    }
                    self.flux.compute_flux(&state, &[0.0, 0.0], &mut flux_node);
                    for eq in 0..nq {
                        for d in 0..dim {
                            let f = flux_node[eq * dim + d];
                            if f != 0.0 {
                                let col = j * dim + d;
                                for r in 0..dp {
                                    z[base + r * nq + eq] += f * wd[(r, col)];
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Matrix-free fallback (MFEM non-preassembled path): direct
            // quadrature of F(u(x_q))·∇φ_i.
            let q_order = 2 * self.ref_elem.order();
            let qr = self.ref_elem.quadrature(q_order);
            let n_vol_qp = qr.n_points();
            let mut phi = vec![0.0; dp];
            let mut gphi = vec![0.0; dp * dim];
            let mut state_qp = vec![0.0; nq];
            let mut flux_qp = vec![0.0; nq * dim];
            for e in 0..self.n_elems {
                let base = e * dp * nq;
                let geom = &self.vol_qp_geom[e];
                for q in 0..n_vol_qp {
                    // D799-3: MFEM's `AssembleElementVector` uses the element's
                    // **isoparametric** transformation per quadrature point
                    // (`Tr.SetIntPoint(&ip)`, hyperbolic.cpp:44-105:
                    // `w = ip.weight·Tr.Weight()`, `dshape = CalcPhysDShape(Tr)`
                    // = J⁻ᵀ∇_ref), not one centroid Jacobian for the whole
                    // element.  The table was built in `new`.
                    let xi = &qr.points[q];
                    let (det_j, jit) = geom[q];
                    let w = qr.weights[q] * det_j;
                    self.ref_elem.eval_basis(xi, &mut phi);
                    // Interpolate u to QP
                    state_qp.fill(0.0);
                    for eq in 0..nq {
                        for i in 0..dp {
                            state_qp[eq] += phi[i] * u[base + i * nq + eq];
                        }
                    }
                    // Compute physical flux at QP
                    self.flux.compute_flux(&state_qp, &[0.0, 0.0], &mut flux_qp);
                    // Evaluate physical gradient of test functions at this QP
                    self.ref_elem.eval_grad_basis(xi, &mut gphi);
                    for i in 0..dp {
                        // ∇x_φ_i = J^{-T} · ∇ξ_φ_i
                        let gx = jit[0] * gphi[i * dim] + jit[1] * gphi[i * dim + 1];
                        let gy = jit[2] * gphi[i * dim] + jit[3] * gphi[i * dim + 1];
                        // z[e,i,eq] += w * detJ * (F_x * gx + F_y * gy)  (signed — D696)
                        for eq in 0..nq {
                            z[base + i * nq + eq] += w * (flux_qp[eq*dim] * gx + flux_qp[eq*dim+1] * gy);
                        }
                    }
                }
            }
        }

        // 5. Apply inverse mass matrix
        for e in 0..self.n_elems {
            let base = e * dp * nq;
            let inv = &self.invmass[e];
            for eq in 0..nq {
                let mut zcol = na::DVector::<f64>::zeros(dp);
                for i in 0..dp {
                    zcol[i] = z[base + i * nq + eq];
                }
                let ycol = inv * zcol;
                for i in 0..dp {
                    dudt[base + i * nq + eq] = ycol[i];
                }
            }
        }
    }
}
