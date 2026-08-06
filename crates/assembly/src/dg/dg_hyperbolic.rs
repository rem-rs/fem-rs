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
use fem_element::lagrange::tri::{TriP1, TriP2, TriP3};
use fem_element::lagrange::QuadL2GL;
use fem_core::types::NodeId;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;

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

    fn max_speed(&self, state: &[f64], _normal: &[f64]) -> f64 {
        let (rho, u, v, p) = cons_to_prim(state, self.gamma);
        let a = (self.gamma * p / rho).sqrt();
        // MFEM EulerFlux::ComputeFlux returns the FULL fluid speed |u| plus
        // sound speed (hyperbolic.cpp), not the normal component.
        let speed = (u * u + v * v).sqrt();
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
#[allow(dead_code)]
struct InteriorFace {
    elem_l: usize,
    elem_r: usize,
    normal: [f64; 2],
    length: f64,
    qp_ref_l: Vec<[f64; 2]>,
    qp_ref_r: Vec<[f64; 2]>,
    qp_weights: Vec<f64>,
    basis_l: Vec<Vec<f64>>,
    basis_r: Vec<Vec<f64>>,
}

// ─── BoundaryFace ─────────────────────────────────────────────────────────────

/// Face data for a boundary face.
#[allow(dead_code)]
struct BoundaryFace {
    elem: usize,
    normal: [f64; 2],
    length: f64,
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
    #[allow(dead_code)]
    weakdiv: Vec<na::DMatrix<f64>>,
    ref_elem: Box<dyn ReferenceElement>,
    elem_shape: ElemShape,
    // Stored mesh for volume term direct quadrature
    mesh_elem_nodes: Vec<Vec<u32>>,       // element → [n0, n1, ...] (3 for Tri3, 4 for Quad4)
    mesh_node_coords: Vec<[f64; 2]>,      // node → [x, y]
    elem_det_j: Vec<f64>,                  // per-element |detJ| (constant for Tri3, centroid for Quad4)
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
                2 => (Box::new(TriP2), shape),
                3 => (Box::new(TriP3), shape),
                _ => (Box::new(TriP1), shape),
            }
        }
    }
}

/// Bilinear (Q1) Jacobian and J^{-T} at quadrature point (ξ, η) ∈ [-1,1]².
/// Returns (detJ, [Jit00, Jit01, Jit10, Jit11]).
fn quad4_jac_at_qp(p: &[[f64; 2]; 4], xi: f64, eta: f64) -> (f64, [f64; 4]) {
    // Q1 shape derivatives on [0,1]² (GL nodal basis):
    // N0=(1-ξ)(1-η), N1=ξ(1-η), N2=ξη, N3=(1-ξ)η
    let dxi = [-(1.0 - eta), (1.0 - eta), eta, -eta];
    let deta = [-(1.0 - xi), -xi, xi, (1.0 - xi)];
    let j11 = dxi[0]*p[0][0] + dxi[1]*p[1][0] + dxi[2]*p[2][0] + dxi[3]*p[3][0];
    let j12 = deta[0]*p[0][0] + deta[1]*p[1][0] + deta[2]*p[2][0] + deta[3]*p[3][0];
    let j21 = dxi[0]*p[0][1] + dxi[1]*p[1][1] + dxi[2]*p[2][1] + dxi[3]*p[3][1];
    let j22 = deta[0]*p[0][1] + deta[1]*p[1][1] + deta[2]*p[2][1] + deta[3]*p[3][1];
    let det = j11 * j22 - j12 * j21;
    let inv_det = 1.0 / det;
    (det.abs(), [j22*inv_det, -j21*inv_det, -j12*inv_det, j11*inv_det])
}

/// Helper: per-element geometry coordinates (uses the mesh "nodes" section,
/// i.e. `geometry_nodes`/`geom_coords_of` — for geometrically periodic meshes
/// like periodic-square.mesh the same vertex index maps to different physical
/// positions in different elements, and only the per-element geometry is
/// meaningful; MFEM's element transforms use the nodes field too).
fn get_quad_nodes(mesh: &dyn MeshTopology, elem: u32) -> [[f64; 2]; 4] {
    let nodes = mesh.geometry_nodes(elem);
    let c = |n: &NodeId| {
        let p = mesh.geom_coords_of(*n);
        [p[0], p[1]]
    };
    [c(&nodes[0]), c(&nodes[1]), c(&nodes[2]), c(&nodes[3])]
}

/// Tri3 constant Jacobian (per-element geometry coordinates).
fn tri3_jac_at_qp(mesh: &dyn MeshTopology, elem: u32) -> (f64, [f64; 4]) {
    let nodes = mesh.geometry_nodes(elem);
    let p0 = mesh.geom_coords_of(nodes[0]);
    let p1 = mesh.geom_coords_of(nodes[1]);
    let p2 = mesh.geom_coords_of(nodes[2]);
    let j11 = p1[0] - p0[0]; let j12 = p2[0] - p0[0];
    let j21 = p1[1] - p0[1]; let j22 = p2[1] - p0[1];
    let det = j11 * j22 - j12 * j21;
    let inv_det = 1.0 / det;
    (det.abs(), [j22*inv_det, -j21*inv_det, -j12*inv_det, j11*inv_det])
}

/// Compute element-wise inverse mass matrix M_e⁻¹ in physical space.
/// M_e[i,j] = Σ_q w_q · |detJ(ξ_q)| · φ_i(ξ_q) · φ_j(ξ_q)
fn compute_inv_mass(mesh: &dyn MeshTopology, ref_elem: &dyn ReferenceElement, n_elems: usize, shape: ElemShape) -> Vec<na::DMatrix<f64>> {
    let dp = ref_elem.n_dofs();
    let q_order = 2 * ref_elem.order();
    let qr = ref_elem.quadrature(q_order);
    let n_qp = qr.n_points();
    let mut phi = vec![0.0; dp];
    let mut invmass = Vec::with_capacity(n_elems);
    for e in 0..n_elems {
        let mut m = na::DMatrix::<f64>::zeros(dp, dp);
        for q in 0..n_qp {
            let xi = &qr.points[q];
            let det_j = match shape {
                ElemShape::Tri => tri3_jac_at_qp(mesh, e as u32).0,
                ElemShape::Quad => {
                    let nodes = mesh.element_nodes(e as u32);
                    let p = get_quad_nodes(mesh, e as u32);
                    let (det, _) = quad4_jac_at_qp(&p, xi[0], xi[1]);
                    det
                }
            };
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

/// Element Jacobian at centroid for pre-stored elem_det_j (constant Tri3 or centroid Quad4).
fn elem_centroid_jac(mesh: &dyn MeshTopology, elem: u32, shape: ElemShape) -> f64 {
    match shape {
        ElemShape::Tri => tri3_jac_at_qp(mesh, elem).0,
        ElemShape::Quad => {
            let nodes = mesh.element_nodes(elem);
            let p = get_quad_nodes(mesh, elem);
            quad4_jac_at_qp(&p, 0.0, 0.0).0  // centroid (ξ=0, η=0)
        }
    }
}

/// Compute element-wise weak divergence matrix in physical space.
/// weakdiv[e][i, j*dim + d] = Σ_q w_q · |detJ(ξ_q)| · φ_i(ξ_q) · (J^{-T}(ξ_q) · ∇ξ_φ_j)_d
fn compute_weak_div(mesh: &dyn MeshTopology, ref_elem: &dyn ReferenceElement, n_elems: usize, shape: ElemShape) -> Vec<na::DMatrix<f64>> {
    let dp = ref_elem.n_dofs();
    let dim = mesh.dim() as usize;
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
            let (det_j, jit) = match shape {
                ElemShape::Tri => tri3_jac_at_qp(mesh, e as u32),
                ElemShape::Quad => {
                    let nodes = mesh.element_nodes(e as u32);
                    let p = get_quad_nodes(mesh, e as u32);
                    quad4_jac_at_qp(&p, xi[0], xi[1])
                }
            };
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

/// Tri3 face patterns: [local_node_a, local_node_b] for faces 0, 1, 2.
const TRI3_FACES: [[usize; 2]; 3] = [[0, 1], [1, 2], [2, 0]];

/// Quad4 face patterns: [local_node_a, local_node_b] for faces 0, 1, 2, 3.
const QUAD4_FACES: [[usize; 2]; 4] = [[0, 1], [1, 2], [2, 3], [3, 0]];

/// Get the two nodes of a face for a given element type.
fn face_nodes(shape: ElemShape, enodes: &[u32], face: u8) -> (u32, u32) {
    match shape {
        ElemShape::Tri => match face {
            0 => (enodes[0], enodes[1]),
            1 => (enodes[1], enodes[2]),
            2 => (enodes[2], enodes[0]),
            _ => unreachable!(),
        },
        ElemShape::Quad => match face {
            0 => (enodes[0], enodes[1]),
            1 => (enodes[1], enodes[2]),
            2 => (enodes[2], enodes[3]),
            3 => (enodes[3], enodes[0]),
            _ => unreachable!(),
        },
    }
}

/// Map a face-local coordinate `t ∈ [0,1]` to reference-triangle coordinates.
fn tri_face_ref(face: u8, t: f64, reverse: bool) -> [f64; 2] {
    let t1 = if reverse { 1.0 - t } else { t };
    match face {
        0 => [t1, 0.0],
        1 => [1.0 - t1, t1],
        2 => [0.0, 1.0 - t1],
        _ => panic!("Tri3 faces 0-2"),
    }
}

/// Map a face-local coordinate `t ∈ [0,1]` to reference-quadrilateral `[-1,1]²` coordinates.
fn quad_face_ref(face: u8, t: f64, reverse: bool) -> [f64; 2] {
    let t1 = if reverse { 1.0 - t } else { t };
    // QuadL2GL reference domain is [0,1]².
    match face {
        0 => [t1, 0.0],        // bottom: η = 0
        1 => [1.0, t1],        // right:  ξ = 1
        2 => [1.0 - t1, 1.0],  // top:    η = 1
        3 => [0.0, 1.0 - t1],  // left:   ξ = 0
        _ => panic!("Quad4 faces 0-3"),
    }
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
    let face_patterns: &[[usize; 2]] = match shape {
        ElemShape::Quad => &QUAD4_FACES,
        ElemShape::Tri  => &TRI3_FACES,
    };

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

            // Normal outward from L (per-element geometry coordinates —
            // periodic meshes: the same vertex index can map to different
            // physical positions in different elements).
            let l_nodes = mesh.geometry_nodes(elem_l as u32);
            let (na, nb) = face_nodes(shape, &l_nodes, face_l as u8);
            let pa = mesh.geom_coords_of(na);
            let pb = mesh.geom_coords_of(nb);
            let (dx, dy) = (pb[0] - pa[0], pb[1] - pa[1]);
            let length = (dx * dx + dy * dy).sqrt();
            // Unit outward normal.  MFEM CalcOrtho returns the outward
            // normal scaled by h/2 and the [-1,1] face quadrature weights
            // sum to 2, so the net face contribution is h·F̂(unit normal);
            // here qp_weights already carry the length, so normal is unit.
            // MFEM CalcOrtho (2D): n = (dy, -dx) — the CW rotation
            // of the face tangent, i.e. the OUTWARD normal.  The old
            // [-dy, dx] was the CCW/inward rotation: every face flux got the
            // wrong sign, so the face and volume terms ADDED instead of
            // cancelling (ex18: z L1 1321 vs C++ 63).
            let normal = [dy / length, -dx / length];

            // Check orientation for R element
            let r_nodes = mesh.geometry_nodes(elem_r as u32);
            let (r_na, _) = face_nodes(shape, &r_nodes, face_r as u8);
            let reverse_r = na != r_na;

            let mut qp_ref_l = Vec::with_capacity(n_qp);
            let mut qp_ref_r = Vec::with_capacity(n_qp);
            let mut qp_w = Vec::with_capacity(n_qp);
            let mut basis_l = Vec::with_capacity(n_qp);
            let mut basis_r = Vec::with_capacity(n_qp);
            let mut phi = vec![0.0; dp];

            for q in 0..n_qp {
                let t = face_pts[q];
                let w = face_wts[q];
                qp_w.push(w * length);
                let rl = match shape {
                    ElemShape::Tri => tri_face_ref(face_l as u8, t, false),
                    ElemShape::Quad => quad_face_ref(face_l as u8, t, false),
                };
                qp_ref_l.push(rl);
                ref_elem.eval_basis(&rl, &mut phi);
                basis_l.push(phi.clone());
                let rr = match shape {
                    ElemShape::Tri => tri_face_ref(face_r as u8, t, reverse_r),
                    ElemShape::Quad => quad_face_ref(face_r as u8, t, reverse_r),
                };
                qp_ref_r.push(rr);
                ref_elem.eval_basis(&rr, &mut phi);
                basis_r.push(phi.clone());
            }

            interior.push(InteriorFace {
                elem_l,
                elem_r,
                normal,
                length,
                qp_ref_l,
                qp_ref_r,
                qp_weights: qp_w,
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

    // Build boundary face geometry for all unpaired edges
    struct BoundInfo { elem: u32, lf: u8, normal: [f64; 2], length: f64 }
    let mut bound_info: Vec<BoundInfo> = Vec::new();
    for &(elem, lf) in &unpaired {
        let enodes = mesh.element_nodes(elem);
        let (na, nb) = face_nodes(shape, &enodes, lf as u8);
        let pa = mesh.node_coords(na);
        let pb = mesh.node_coords(nb);
        let (dx, dy) = (pb[0] - pa[0], pb[1] - pa[1]);
        let length = (dx * dx + dy * dy).sqrt();
        let normal = [dy / length, -dx / length];
        bound_info.push(BoundInfo { elem, lf, normal, length });
    }

    // Detect periodic pairs among boundary faces
    let periodic_idx: Vec<(usize, usize)> = {
        let unpaired_with_normals: Vec<(u32, u8, [f64;2])> = bound_info.iter()
            .map(|b| (b.elem, b.lf, b.normal)).collect();
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
        let (elem_l, elem_r, normal, face_l, face_r) = {
            // Use the element with smaller index as L, normal outward from L
            if bi.elem < bj.elem {
                (bi.elem as usize, bj.elem as usize, bi.normal, bi.lf, bj.lf)
            } else {
                (bj.elem as usize, bi.elem as usize, bj.normal, bj.lf, bi.lf)
            }
        };
        let length = bi.length;  // both have same length for periodic square
        let mut qp_ref_l = Vec::with_capacity(n_qp);
        let mut qp_ref_r = Vec::with_capacity(n_qp);
        let mut qp_w = Vec::with_capacity(n_qp);
        let mut basis_l = Vec::with_capacity(n_qp);
        let mut basis_r = Vec::with_capacity(n_qp);
        let mut phi_l = vec![0.0; dp];
        let mut phi_r = vec![0.0; dp];
        for q in 0..n_qp {
            let t = face_pts[q];
            let w = face_wts[q];
            qp_w.push(w * length);
            let rl = match shape { ElemShape::Tri => tri_face_ref(face_l as u8, t, false), ElemShape::Quad => quad_face_ref(face_l as u8, t, false) };
            qp_ref_l.push(rl);
            ref_elem.eval_basis(&rl, &mut phi_l);
            basis_l.push(phi_l.clone());
            // For periodic pair, the other element's face QP uses the opposite
            // face orientation (reverse = true) since it's the opposite side
            let rr = match shape { ElemShape::Tri => tri_face_ref(face_r as u8, t, true), ElemShape::Quad => quad_face_ref(face_r as u8, t, true) };
            qp_ref_r.push(rr);
            ref_elem.eval_basis(&rr, &mut phi_r);
            basis_r.push(phi_r.clone());
        }
        interior.push(InteriorFace {
            elem_l, elem_r,
            normal, length,
            qp_ref_l, qp_ref_r, qp_weights: qp_w,
            basis_l, basis_r,
        });
    }

    // Remaining unpaired boundary faces become boundary faces
    for (idx, bi) in bound_info.iter().enumerate() {
        if used_by_periodic[idx] { continue; }
        let mut qp_ref = Vec::with_capacity(n_qp);
        let mut qp_w = Vec::with_capacity(n_qp);
        let mut basis = Vec::with_capacity(n_qp);
        let mut phi = vec![0.0; dp];
        for q in 0..n_qp {
            let t = face_pts[q];
            let w = face_wts[q];
            qp_w.push(w * bi.length);
            let r = match shape { ElemShape::Tri => tri_face_ref(bi.lf, t, false), ElemShape::Quad => quad_face_ref(bi.lf, t, false) };
            qp_ref.push(r);
            ref_elem.eval_basis(&r, &mut phi);
            basis.push(phi.clone());
        }
        boundary.push(BoundaryFace {
            elem: bi.elem as usize,
            normal: bi.normal,
            length: bi.length,
            qp_ref,
            qp_weights: qp_w,
            basis,
        });
    }

    (interior, boundary)
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
        // Store mesh data for direct quadrature volume term
        let mut mesh_elem_nodes = Vec::with_capacity(n_elems);
        let mut elem_det_j = Vec::with_capacity(n_elems);
        for e in 0..n_elems as u32 {
            let nodes = mesh.element_nodes(e);
            mesh_elem_nodes.push(nodes.to_vec());
            elem_det_j.push(elem_centroid_jac(mesh, e, elem_shape));
        }
        let mut mesh_node_coords = Vec::with_capacity(mesh.n_nodes());
        for n in 0..mesh.n_nodes() as u32 {
            let c = mesh.node_coords(n);
            mesh_node_coords.push([c[0], c[1]]);
        }
        Self {
            n_elems,
            dofs_per_elem,
            n_eq,
            dim,
            total_dofs,
            invmass,
            weakdiv,
            ref_elem,
            elem_shape,
            flux: flux_fn,
            interior_faces,
            boundary_faces,
            max_char_speed: std::cell::Cell::new(0.0),
            z: RefCell::new(vec![0.0; total_dofs]),
            preassemble_weakdiv,
            mesh_elem_nodes,
            mesh_node_coords,
            elem_det_j,
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

        // 2. Interior face flux contributions
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
                let cL = self.flux.max_speed(&uL, &face.normal);
                let cR = self.flux.max_speed(&uR, &face.normal);
                let c = cL.max(cR);
                if c > self.max_char_speed.get() { self.max_char_speed.set(c); }
                let f_hat = self.flux.numerical_flux(&uL, &uR, &face.normal);
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

        // 3. Boundary faces (reflecting wall BC)
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
                let nx = face.normal[0];
                let ny = face.normal[1];
                let vn = uL[1] * nx + uL[2] * ny;
                u_mirror[0] = uL[0];
                u_mirror[1] = uL[1] - 2.0 * vn * nx;
                u_mirror[2] = uL[2] - 2.0 * vn * ny;
                u_mirror[3] = uL[3];
                let c = self.flux.max_speed(&uL, &face.normal)
                    .max(self.flux.max_speed(&u_mirror, &face.normal));
                if c > self.max_char_speed.get() { self.max_char_speed.set(c); }
                let f_hat = self.flux.numerical_flux(&uL, &u_mirror, &face.normal);
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

        // 4. Volume term: direct quadrature (Form 2: +∫ F·∇v)
        if self.preassemble_weakdiv {
        //    Interpolate u to QP, compute flux F(u_qp), dot with∇φ_i.
        let q_order = 2 * self.ref_elem.order();
        let qr = self.ref_elem.quadrature(q_order);
        let n_vol_qp = qr.n_points();
        let mut phi = vec![0.0; dp];
        let mut gphi = vec![0.0; dp * dim];
        let mut state_qp = vec![0.0; nq];
        let mut flux_qp = vec![0.0; nq * dim];
        for e in 0..self.n_elems {
            let base = e * dp * nq;
            let det_j = self.elem_det_j[e];
            // Precompute physical gradients of all test functions at each QP
            // ∇x_φ_i(q) = J_e^{-T} · ∇ξ_φ_i(q)
            let (_, jit) = match self.elem_shape {
                ElemShape::Tri => {
                    let en = &self.mesh_elem_nodes[e];
                    let p0 = &self.mesh_node_coords[en[0] as usize];
                    let p1 = &self.mesh_node_coords[en[1] as usize];
                    let p2 = &self.mesh_node_coords[en[2] as usize];
                    let j11 = p1[0] - p0[0]; let j12 = p2[0] - p0[0];
                    let j21 = p1[1] - p0[1]; let j22 = p2[1] - p0[1];
                    let det = j11 * j22 - j12 * j21;
                    let inv_det = 1.0 / det;
                    (det.abs(), [j22*inv_det, -j21*inv_det, -j12*inv_det, j11*inv_det])
                }
                // Quad: use the full 4-node isoparametric Jacobian.  The old
                // code built J from only (p0,p1,p2) — for a quad the η
                // direction is p3−p0, NOT p2−p0 (p2 is the opposite corner),
                // which corrupted det(J) and produced NaN in ex18's volume
                // term on quad meshes.
                ElemShape::Quad => {
                    let en = &self.mesh_elem_nodes[e];
                    let p: [[f64; 2]; 4] = [
                        self.mesh_node_coords[en[0] as usize],
                        self.mesh_node_coords[en[1] as usize],
                        self.mesh_node_coords[en[2] as usize],
                        self.mesh_node_coords[en[3] as usize],
                    ];
                    quad4_jac_at_qp(&p, 0.0, 0.0)
                }
            };
            for q in 0..n_vol_qp {
                let xi = &qr.points[q];
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
                    // z[e,i,eq] += w * |detJ| * (F_x * gx + F_y * gy)
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
