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
use fem_mesh::topology::MeshTopology;

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
        let vn = u * normal[0] + v * normal[1];
        let nlen = (normal[0] * normal[0] + normal[1] * normal[1]).sqrt();
        (if nlen > 0.0 { (vn / nlen).abs() } else { 0.0 }) + a
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
    weakdiv: Vec<na::DMatrix<f64>>,
    ref_elem: Box<dyn ReferenceElement>,
    flux: Box<dyn FluxFunction>,
    interior_faces: Vec<InteriorFace>,
    boundary_faces: Vec<BoundaryFace>,
    max_char_speed: std::cell::Cell<f64>,
    z: RefCell<Vec<f64>>,
    preassemble_weakdiv: bool,
}

// ─── Helper functions ─────────────────────────────────────────────────────────

fn make_ref_elem(_mesh: &dyn MeshTopology, order: u8) -> Box<dyn ReferenceElement> {
    match order {
        1 => Box::new(TriP1),
        2 => Box::new(TriP2),
        3 => Box::new(TriP3),
        _ => Box::new(TriP1), // fallback for P0
    }
}

/// Compute element-wise inverse mass matrix M_e⁻¹ in physical space.
/// M_e[i,j] = Σ_q w_q · |detJ| · φ_i(xi_q) · φ_j(xi_q)
fn compute_inv_mass(mesh: &dyn MeshTopology, ref_elem: &dyn ReferenceElement, n_elems: usize) -> Vec<na::DMatrix<f64>> {
    let dp = ref_elem.n_dofs();
    let dim = mesh.dim() as usize;
    let q_order = 2 * ref_elem.order();
    let qr = ref_elem.quadrature(q_order);
    let n_qp = qr.n_points();
    let mut phi = vec![0.0; dp];
    let mut invmass = Vec::with_capacity(n_elems);
    for e in 0..n_elems {
        let det_j = element_det_j(mesh, e as u32);
        let mut m = na::DMatrix::<f64>::zeros(dp, dp);
        for q in 0..n_qp {
            let w = qr.weights[q] * det_j;
            ref_elem.eval_basis(&qr.points[q], &mut phi);
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

/// Compute element Jacobian determinant for a Tri3 element.
fn element_det_j(mesh: &dyn MeshTopology, elem: u32) -> f64 {
    let nodes = mesh.element_nodes(elem);
    let p0 = mesh.node_coords(nodes[0]);
    let p1 = mesh.node_coords(nodes[1]);
    let p2 = mesh.node_coords(nodes[2]);
    let j11 = p1[0] - p0[0]; let j12 = p2[0] - p0[0];
    let j21 = p1[1] - p0[1]; let j22 = p2[1] - p0[1];
    (j11 * j22 - j12 * j21).abs()
}

/// Compute element Jacobian and its inverse-transpose for a Tri3 element.
/// Returns (detJ, inv_jac_transpose) where inv_jac_transpose is flattened [2×2]: [Jit00, Jit01, Jit10, Jit11].
fn element_jac_inv_transpose(mesh: &dyn MeshTopology, elem: u32) -> (f64, [f64; 4]) {
    let nodes = mesh.element_nodes(elem);
    let p0 = mesh.node_coords(nodes[0]);
    let p1 = mesh.node_coords(nodes[1]);
    let p2 = mesh.node_coords(nodes[2]);
    let j11 = p1[0] - p0[0]; let j12 = p2[0] - p0[0];
    let j21 = p1[1] - p0[1]; let j22 = p2[1] - p0[1];
    let det = j11 * j22 - j12 * j21;
    let inv_det = 1.0 / det;
    // J^{-T} = 1/det * [j22, -j21; -j12, j11]
    (det.abs(), [j22 * inv_det, -j21 * inv_det, -j12 * inv_det, j11 * inv_det])
}

/// Compute element-wise weak divergence matrix in physical space.
/// weakdiv[e][i, j*dim + d] = Σ_q w_q · |detJ| · φ_i(xi_q) · (J^{-T} · ∇ξ_φ_j)_d
fn compute_weak_div(mesh: &dyn MeshTopology, ref_elem: &dyn ReferenceElement, n_elems: usize) -> Vec<na::DMatrix<f64>> {
    let dp = ref_elem.n_dofs();
    let dim = mesh.dim() as usize;
    let q_order = 2 * ref_elem.order();
    let qr = ref_elem.quadrature(q_order);
    let n_qp = qr.n_points();
    let mut phi = vec![0.0; dp];
    let mut gphi = vec![0.0; dp * dim];
    let mut weakdiv = Vec::with_capacity(n_elems);
    for e in 0..n_elems {
        let (det_j, jit) = element_jac_inv_transpose(mesh, e as u32);
        let mut wd = na::DMatrix::<f64>::zeros(dp, dp * dim);
        for q in 0..n_qp {
            let w = qr.weights[q] * det_j;
            ref_elem.eval_basis(&qr.points[q], &mut phi);
            ref_elem.eval_grad_basis(&qr.points[q], &mut gphi);
            for i in 0..dp {
                for j in 0..dp {
                    for d in 0..dim {
                        // Transform gradient: grad_phys = J^{-T} * grad_ref
                        let mut grad_phys = 0.0;
                        for k in 0..dim {
                            grad_phys += jit[d * dim + k] * gphi[j * dim + k];
                        }
                        wd[(i, j * dim + d)] += w * phi[i] * grad_phys;
                    }
                }
            }
        }
        weakdiv.push(wd);
    }
    weakdiv
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

/// Build interior and boundary face structures for a triangle mesh.
/// Detects periodic pairs from boundary faces with opposite normals.
fn build_faces(mesh: &dyn MeshTopology, ref_elem: &dyn ReferenceElement) -> (Vec<InteriorFace>, Vec<BoundaryFace>) {
    let dp = ref_elem.n_dofs();
    let n_elems = mesh.n_elements() as u32;
    let n_qp = ((2 * ref_elem.order() + 1) as usize).min(4).max(1);
    let (face_pts, face_wts) = gauss_legendre_01(n_qp);

    // Record all element edges
    struct ElemEdge {
        nodes: [u32; 2],
        elem: u32,
        local_face: u8,
    }
    let mut edge_list: Vec<ElemEdge> = Vec::new();
    for e in 0..n_elems {
        let enodes = mesh.element_nodes(e);
        for (lf, &[na, nb]) in [[0, 1], [1, 2], [2, 0]].iter().enumerate() {
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

            // Normal outward from L
            let l_nodes = mesh.element_nodes(elem_l as u32);
            let (na, nb) = match face_l {
                0 => (l_nodes[0], l_nodes[1]),
                1 => (l_nodes[1], l_nodes[2]),
                2 => (l_nodes[2], l_nodes[0]),
                _ => unreachable!(),
            };
            let pa = mesh.node_coords(na);
            let pb = mesh.node_coords(nb);
            let (dx, dy) = (pb[0] - pa[0], pb[1] - pa[1]);
            let normal = [-dy, dx];
            let length = (dx * dx + dy * dy).sqrt();

            // Check orientation for R element
            let r_nodes = mesh.element_nodes(elem_r as u32);
            let (r_na, _) = match face_r {
                0 => (r_nodes[0], r_nodes[1]),
                1 => (r_nodes[1], r_nodes[2]),
                2 => (r_nodes[2], r_nodes[0]),
                _ => unreachable!(),
            };
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
                let rl = tri_face_ref(face_l as u8, t, false);
                qp_ref_l.push(rl);
                ref_elem.eval_basis(&rl, &mut phi);
                basis_l.push(phi.clone());
                let rr = tri_face_ref(face_r as u8, t, reverse_r);
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
        let (na, nb) = match lf {
            0 => (enodes[0], enodes[1]),
            1 => (enodes[1], enodes[2]),
            2 => (enodes[2], enodes[0]),
            _ => unreachable!(),
        };
        let pa = mesh.node_coords(na);
        let pb = mesh.node_coords(nb);
        let (dx, dy) = (pb[0] - pa[0], pb[1] - pa[1]);
        let normal = [-dy, dx];
        let length = (dx * dx + dy * dy).sqrt();
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
            let rl = tri_face_ref(face_l as u8, t, false);
            qp_ref_l.push(rl);
            ref_elem.eval_basis(&rl, &mut phi_l);
            basis_l.push(phi_l.clone());
            // For periodic pair, the other element's face QP uses the opposite
            // face orientation (reverse = true) since it's the opposite side
            let rr = tri_face_ref(face_r as u8, t, true);
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
            let r = tri_face_ref(bi.lf, t, false);
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
        let ref_elem = make_ref_elem(mesh, order);
        let dofs_per_elem = ref_elem.n_dofs();
        let total_dofs = n_elems * dofs_per_elem * n_eq;
        let invmass = compute_inv_mass(mesh, &*ref_elem, n_elems);
        let weakdiv = if preassemble_weakdiv {
            compute_weak_div(mesh, &*ref_elem, n_elems)
        } else {
            Vec::new()
        };
        let (interior_faces, boundary_faces) = build_faces(mesh, &*ref_elem);
        // Face counts for debugging: interior={}, boundary={}
        let _ = (interior_faces.len(), boundary_faces.len());
        Self {
            n_elems,
            dofs_per_elem,
            n_eq,
            dim,
            total_dofs,
            invmass,
            weakdiv,
            ref_elem,
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
                // Form 2: face = -ĝ·[[v]] = -ĝ·v_L + ĝ·v_R
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
                let nx = face.normal[0] / face.length;
                let ny = face.normal[1] / face.length;
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

        // 4. Volume term: weak divergence of F(u)
        if self.preassemble_weakdiv {
            let mut state = vec![0.0; nq];
            let mut flux_qp = vec![0.0; nq * dim];
            for e in 0..self.n_elems {
                let base = e * dp * nq;
                // f_all[(j * dim + d) * nq + eq] = F_d(u_j)[eq]
                let mut f_all = vec![0.0; dp * dim * nq];
                for j in 0..dp {
                    for eq in 0..nq { state[eq] = u[base + j * nq + eq]; }
                    self.flux.compute_flux(&state, &[0.0, 0.0], &mut flux_qp);
                    for eq in 0..nq {
                        for d in 0..dim {
                            f_all[(j * dim + d) * nq + eq] = flux_qp[eq * dim + d];
                        }
                    }
                }
                let wd = &self.weakdiv[e];
                for eq in 0..nq {
                    let mut f_col = na::DVector::<f64>::zeros(dp * dim);
                    for j in 0..dp {
                        for d in 0..dim {
                            f_col[j * dim + d] = f_all[(j * dim + d) * nq + eq];
                        }
                    }
                    let zcol = wd * f_col;
                    for i in 0..dp {
                        z[base + i * nq + eq] += zcol[i];
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
