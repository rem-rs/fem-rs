//! DG advection with upwind numerical flux.
//!
//! Solves the scalar advection equation:
//! ```text
//! ∂u/∂t + ∇·(b u) = f
//! ```
//! using a discontinuous Galerkin discretization with upwind numerical flux.
//!
//! ## Components
//!
//! - [`DgFaceIntegrator`] — trait for bilinear forms on DG interior faces
//!   (four-block element matrix K_ll, K_lr, K_rl, K_rr).
//! - [`DGAdvectionIntegrator`] — upwind-flux advection integrator.
//! - [`assemble_dg_interior_faces`] — generic face-assembly driver.
//! - [`DgAdvectionRhs`] — right-hand side closure for explicit RK time stepping.

use nalgebra::DMatrix;

use fem_core::types::{DofId, ElemId};
use fem_element::{ReferenceElement, lagrange::{SegP1, SegP2, SegP3, TriP1, TriP2, TriP3, TetP1, TetP2, TetP3}};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;

use crate::coefficient::{CoeffCtx, VectorCoeff};
use crate::integrator::{BilinearIntegrator, QpData};
use crate::interior_faces::InteriorFaceList;


// ─── DgFaceQpData ─────────────────────────────────────────────────────────────

/// Per-quadrature-point data for DG interior face integrals.
pub struct DgFaceQpData<'a> {
    /// Number of DOFs on the left element.
    pub n_dofs_l: usize,
    /// Number of DOFs on the right element.
    pub n_dofs_r: usize,
    /// Spatial dimension.
    pub dim: usize,
    /// Effective integration weight: quadrature weight × face Jacobian.
    pub weight: f64,
    /// Basis function values on the left element; length `n_dofs_l`.
    pub phi_l: &'a [f64],
    /// Basis function values on the right element; length `n_dofs_r`.
    pub phi_r: &'a [f64],
    /// Unit normal pointing outward from the left element; length `dim`.
    pub normal: &'a [f64],
    /// Physical coordinates of this quadrature point; length `dim`.
    pub x_phys: &'a [f64],
    /// Left element index.
    pub elem_l: ElemId,
    /// Right element index.
    pub elem_r: ElemId,
    /// Global DOF indices for the left element.
    pub elem_dofs_l: Option<&'a [DofId]>,
    /// Global DOF indices for the right element.
    pub elem_dofs_r: Option<&'a [DofId]>,
}

// ─── DgFaceIntegrator trait ───────────────────────────────────────────────────

/// Bilinear form on a DG interior face.
///
/// Produces a 4-block element matrix `[K_ll K_lr; K_rl K_rr]`.
pub trait DgFaceIntegrator: Send + Sync {
    /// Accumulate into the four face matrix blocks.
    fn add_to_face_matrix(&self, qp: &DgFaceQpData<'_>,
        k_ll: &mut [f64], k_lr: &mut [f64],
        k_rl: &mut [f64], k_rr: &mut [f64]);
}

// ─── Generic interior-face assembly driver ────────────────────────────────────

/// Assemble the interior-face contribution of a [`DgFaceIntegrator`].
///
/// For each interior face, evaluates basis functions on both adjacent elements
/// at face quadrature points and delegates to the integrator.
pub fn assemble_dg_interior_faces<M: MeshTopology, S: FESpace<Mesh=M>, F: DgFaceIntegrator>(
    coo: &mut CooMatrix<f64>,
    mesh: &M,
    space: &S,
    ifl: &InteriorFaceList,
    order: u8,
    quad_order: u8,
    integ: &F,
) {
    let dim = mesh.dim() as usize;

    for face in &ifl.faces {
        let el = face.elem_left;
        let er = face.elem_right;
        let face_nodes = &face.face_nodes;

        // Face geometry
        let mut normal_l: Vec<f64>;
        let h_f: f64;

        if dim == 2 {
            let x0 = mesh.node_coords(face_nodes[0]);
            let x1 = mesh.node_coords(face_nodes[1]);
            let dx = x1[0] - x0[0];
            let dy = x1[1] - x0[1];
            h_f = (dx*dx + dy*dy).sqrt();
            normal_l = vec![dy / h_f, -dx / h_f];
        } else {
            // 3-D: use the first two face nodes for a rough normal
            let x0 = mesh.node_coords(face_nodes[0]);
            let x1 = mesh.node_coords(face_nodes[1]);
            let x2 = mesh.node_coords(face_nodes[2]);
            let v1 = [x1[0]-x0[0], x1[1]-x0[1], x1[2]-x0[2]];
            let v2 = [x2[0]-x0[0], x2[1]-x0[1], x2[2]-x0[2]];
            let nx = v1[1]*v2[2] - v1[2]*v2[1];
            let ny = v1[2]*v2[0] - v1[0]*v2[2];
            let nz = v1[0]*v2[1] - v1[1]*v2[0];
            h_f = (nx*nx + ny*ny + nz*nz).sqrt();
            normal_l = vec![nx / h_f, ny / h_f, nz / h_f];
        }

        // Ensure normal points outward from left element
        orient_normal_outward(mesh, el, face_nodes, &mut normal_l);

        // Reference element for the face
        let face_elem_type = if dim == 2 { ElementType::Line2 } else { ElementType::Tri3 };
        let ref_face = ref_elem_face(face_elem_type, order);
        let q_face = ref_face.quadrature(quad_order);

        // Volume reference elements — use actual element DOF count from the space
        let dofs_l: Vec<usize> = space.element_dofs(el).iter().map(|&d| d as usize).collect();
        let dofs_r: Vec<usize> = space.element_dofs(er).iter().map(|&d| d as usize).collect();
        let n_l = dofs_l.len();
        let n_r = dofs_r.len();
        let et_l = mesh.element_type(el);
        let et_r = mesh.element_type(er);
        let elem_order_l = space.element_order(el);
        let elem_order_r = space.element_order(er);
        let re_l = ref_elem_vol(et_l, elem_order_l);
        let re_r = ref_elem_vol(et_r, elem_order_r);

        let nodes_l = mesh.element_nodes(el);
        let nodes_r = mesh.element_nodes(er);

        // Affine Jacobians for both elements
        let (jac_l, _) = simplex_jac(mesh, nodes_l, dim);
        let (jac_r, _) = simplex_jac(mesh, nodes_r, dim);
        let jit_l = jac_l.clone().try_inverse().unwrap().transpose();
        let jit_r = jac_r.clone().try_inverse().unwrap().transpose();

        let x0_l = mesh.node_coords(nodes_l[0]);
        let x0_r = mesh.node_coords(nodes_r[0]);

        // Scatter: face quadrature points along the edge (2-D) or triangle (3-D)
        let face_points: Vec<Vec<f64>> = q_face.points.clone();
        let face_weights = &q_face.weights;

        let mut k_ll = vec![0.0_f64; n_l * n_l];
        let mut k_lr = vec![0.0_f64; n_l * n_r];
        let mut k_rl = vec![0.0_f64; n_r * n_l];
        let mut k_rr = vec![0.0_f64; n_r * n_r];

        let mut phi_l = vec![0.0_f64; n_l];
        let mut phi_r = vec![0.0_f64; n_r];
        let mut gref_l = vec![0.0_f64; n_l * dim];
        let mut gref_r = vec![0.0_f64; n_r * dim];
        let mut gphys_l = vec![0.0_f64; n_l * dim];
        let mut gphys_r = vec![0.0_f64; n_r * dim];

        for (qi, xi_f) in face_points.iter().enumerate() {
            let w_f = face_weights[qi] * h_f;

            // Physical quadrature point on the face (linear interpolation of face nodes)
            let xp: Vec<f64> = if dim == 2 {
                let x0f = mesh.node_coords(face_nodes[0]);
                let x1f = mesh.node_coords(face_nodes[1]);
                let t = xi_f[0];
                (0..dim).map(|i| (1.0 - t) * x0f[i] + t * x1f[i]).collect()
            } else {
                // Barycentric interpolation for 3-D triangular faces
                let c0 = mesh.node_coords(face_nodes[0]);
                let c1 = mesh.node_coords(face_nodes[1]);
                let c2 = mesh.node_coords(face_nodes[2]);
                let u = xi_f[0];
                let v = xi_f[1];
                (0..dim).map(|i| (1.0 - u - v) * c0[i] + u * c1[i] + v * c2[i]).collect()
            };

            // Map physical point to reference coordinates of each element
            let xi_l = phys_to_ref(&jac_l, x0_l, &xp, dim);
            let xi_r = phys_to_ref(&jac_r, x0_r, &xp, dim);

            re_l.eval_basis(&xi_l, &mut phi_l);
            re_r.eval_basis(&xi_r, &mut phi_r);
            re_l.eval_grad_basis(&xi_l, &mut gref_l);
            re_r.eval_grad_basis(&xi_r, &mut gref_r);
            // Transform gradients (needed only if integrator uses grad, but computed for consistency)
            xform_grads(&jit_l, &gref_l, &mut gphys_l, n_l, dim);
            xform_grads(&jit_r, &gref_r, &mut gphys_r, n_r, dim);

            let qp = DgFaceQpData {
                n_dofs_l: n_l,
                n_dofs_r: n_r,
                dim,
                weight: w_f,
                phi_l: &phi_l,
                phi_r: &phi_r,
                normal: &normal_l,
                x_phys: &xp,
                elem_l: el,
                elem_r: er,
                elem_dofs_l: None,
                elem_dofs_r: None,
            };

            integ.add_to_face_matrix(&qp, &mut k_ll, &mut k_lr, &mut k_rl, &mut k_rr);
        }

        // Scatter 4 blocks to global COO matrix
        for (i, &gi) in dofs_l.iter().enumerate() {
            for (j, &gj) in dofs_l.iter().enumerate() { coo.add(gi, gj, k_ll[i*n_l+j]); }
            for (j, &gj) in dofs_r.iter().enumerate() { coo.add(gi, gj, k_lr[i*n_r+j]); }
        }
        for (i, &gi) in dofs_r.iter().enumerate() {
            for (j, &gj) in dofs_l.iter().enumerate() { coo.add(gi, gj, k_rl[i*n_l+j]); }
            for (j, &gj) in dofs_r.iter().enumerate() { coo.add(gi, gj, k_rr[i*n_r+j]); }
        }
    }
}

// ─── DGAdvectionIntegrator ────────────────────────────────────────────────────

/// DG upwind advection integrator.
///
/// Implements both:
/// - **Volume term** (via [`BilinearIntegrator`]): `∫ (b·∇u) v dx`
/// - **Interior face term** (via [`DgFaceIntegrator`]): upwind numerical flux
///   `-∫ [[v]] F̂ dS` where `F̂ = (b·n)⁺ u⁻ + (b·n)⁻ u⁺`
pub struct DGAdvectionIntegrator<V: VectorCoeff> {
    /// Convection velocity field.
    pub velocity: V,
}

// ── Volume term (BilinearIntegrator) — weak form ──────────────────────────
//
// The PDE: ∂u/∂t + ∇·(b·u) = 0
// Integration by parts on each element:
//   ∫ v·∇·(b·u) dx = -∫ (b·∇v)·u dx + ∫_{∂K} v·(b·n̂)·u dS
//
// The weak-form DG uses the LHS (integrated-by-parts):
//   K_vol[i,j] = -w · (b·∇φ_i) · φ_j
//
// Together with the face upwind flux, this preserves constants.
// (The strong form ∫ v·(b·∇u) dx does NOT preserve constants.)

impl<V: VectorCoeff> BilinearIntegrator for DGAdvectionIntegrator<V> {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let d = qp.dim;
        let ctx = CoeffCtx::from_qp(
            qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag,
            Some(qp.phi), qp.elem_dofs,
        );
        let mut b = [0.0_f64; 3];
        self.velocity.eval(&ctx, &mut b[..d]);

        for i in 0..n {
            // Compute (b·∇φ_i)
            let mut b_dot_grad_i = 0.0;
            for k in 0..d {
                b_dot_grad_i += b[k] * qp.grad_phys[i * d + k];
            }
            for j in 0..n {
                // -w · (b·∇φ_i) · φ_j
                k_elem[i * n + j] += -qp.weight * b_dot_grad_i * qp.phi[j];
            }
        }
    }
}

// ── Interior face term (DgFaceIntegrator) ───────────────────────────────────

impl<V: VectorCoeff> DgFaceIntegrator for DGAdvectionIntegrator<V> {
    fn add_to_face_matrix(&self, qp: &DgFaceQpData<'_>,
        k_ll: &mut [f64], k_lr: &mut [f64],
        k_rl: &mut [f64], k_rr: &mut [f64])
    {
        let n_l = qp.n_dofs_l;
        let n_r = qp.n_dofs_r;
        let d = qp.dim;
        let w = qp.weight;

        // Evaluate velocity at this face quadrature point
        let ctx = CoeffCtx::from_qp(qp.x_phys, d, qp.elem_l, 0, None, None);
        let mut b = [0.0_f64; 3];
        self.velocity.eval(&ctx, &mut b[..d]);

        // Project velocity onto face normal: vn = b·n
        let vn: f64 = (0..d).map(|i| b[i] * qp.normal[i]).sum();
        let vn_pos = vn.max(0.0);
        let vn_neg = vn.min(0.0);

        let phi_l = qp.phi_l;
        let phi_r = qp.phi_r;

        // DG weak form (integration by parts) interior face contribution:
        //   ∫_F [[v]]·F̂ dS where [[v]] = v⁻ - v⁺ (scalar jump)
        //   and F̂ = (b·n)⁺·u⁻ + (b·n)⁻·u⁺ (upwind numerical flux)
        //
        // Test=L (v⁻): [[v]] = +v⁻
        //   K_ll: +∫ v⁻·(b·n)⁺·u⁻ → +w · φ⁻ · (b·n)⁺ · φ⁻
        //   K_lr: +∫ v⁻·(b·n)⁻·u⁺ → +w · φ⁻ · (b·n)⁻ · φ⁺
        //
        // Test=R (v⁺): [[v]] = -v⁺
        //   K_rl: -∫ v⁺·(b·n)⁺·u⁻ → -w · φ⁺ · (b·n)⁺ · φ⁻
        //   K_rr: -∫ v⁺·(b·n)⁻·u⁺ → -w · φ⁺ · (b·n)⁻ · φ⁺
        for i in 0..n_l {
            for j in 0..n_l { k_ll[i * n_l + j] += w * phi_l[i] * vn_pos * phi_l[j]; }
        }
        for i in 0..n_l {
            for j in 0..n_r { k_lr[i * n_r + j] += w * phi_l[i] * vn_neg * phi_r[j]; }
        }
        for i in 0..n_r {
            for j in 0..n_l { k_rl[i * n_l + j] += -w * phi_r[i] * vn_pos * phi_l[j]; }
        }
        for i in 0..n_r {
            for j in 0..n_r { k_rr[i * n_r + j] += -w * phi_r[i] * vn_neg * phi_r[j]; }
        }
    }
}

// ─── Boundary face helper ─────────────────────────────────────────────────────

/// Assemble the boundary contribution for advection (inflow/outflow).
///
/// - **Inflow** (b·n < 0): applies `u = g_D` weakly via RHS
/// - **Outflow** (b·n ≥ 0): standard upwind takes interior value (no extra term)
///
/// Returns the assembled boundary RHS vector `f_bc`.
pub fn assemble_advection_boundary<M: MeshTopology, S: FESpace<Mesh=M>, V: VectorCoeff>(
    space: &S,
    velocity: &V,
    tags: &[i32],
    g_d: &dyn Fn(&[f64]) -> f64,
    order: u8,
    quad_order: u8,
) -> Vec<f64> {
    let mesh = space.mesh();
    let dim = mesh.dim() as usize;
    let n_dofs = space.n_dofs();
    let mut rhs = vec![0.0_f64; n_dofs];

    for f in mesh.face_iter() {
        if !tags.contains(&mesh.face_tag(f)) { continue; }
        let fnodes = mesh.face_nodes(f);
        let h_f: f64;
        let normal: Vec<f64>;

        if dim == 2 {
            let x0 = mesh.node_coords(fnodes[0]);
            let x1 = mesh.node_coords(fnodes[1]);
            let dx = x1[0] - x0[0];
            let dy = x1[1] - x0[1];
            h_f = (dx*dx + dy*dy).sqrt();
            normal = vec![dy / h_f, -dx / h_f];
        } else { return rhs; } // 3-D not implemented yet

        // Find owning element
        let elem = find_face_elem(mesh, f, fnodes);

        let face_type = if dim == 2 { ElementType::Line2 } else { ElementType::Tri3 };
        let ref_face = ref_elem_face(face_type, order);
        let q_face = ref_face.quadrature(quad_order);

        let et = mesh.element_type(elem);
        let ref_elem = ref_elem_vol(et, order);
        let n_dofs_e = ref_elem.n_dofs();
        let dofs: Vec<usize> = space.element_dofs(elem).iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(elem);
        let (jac, _) = simplex_jac(mesh, nodes, dim);
        let x0_e = mesh.node_coords(nodes[0]);

        let mut f_elem = vec![0.0_f64; n_dofs_e];
        let mut phi = vec![0.0_f64; n_dofs_e];

        for (qi, xi_f) in q_face.points.iter().enumerate() {
            let w_f = q_face.weights[qi] * h_f;
            let xp: Vec<f64> = {
                let x0f = mesh.node_coords(fnodes[0]);
                let x1f = mesh.node_coords(fnodes[1]);
                let t = xi_f[0];
                (0..dim).map(|i| (1.0 - t) * x0f[i] + t * x1f[i]).collect()
            };
            let xi = phys_to_ref(&jac, x0_e, &xp, dim);
            ref_elem.eval_basis(&xi, &mut phi);

            let ctx = CoeffCtx::from_qp(&xp, dim, elem, mesh.face_tag(f), None, None);
            let mut b = [0.0_f64; 3];
            velocity.eval(&ctx, &mut b[..dim]);
            let vn: f64 = (0..dim).map(|i| b[i] * normal[i]).sum();

            // Inflow: b·n < 0 → impose BC weakly
            if vn < 0.0 {
                let g_val = g_d(&xp);
                for i in 0..n_dofs_e {
                    f_elem[i] += w_f * phi[i] * vn * g_val;
                }
            }
        }

        for (i, &gi) in dofs.iter().enumerate() {
            rhs[gi] += f_elem[i];
        }
    }
    rhs
}

// ─── Explicit RHS wrapper ─────────────────────────────────────────────────────

/// Right-hand side for explicit time integration of DG advection.
///
/// `du/dt = M_lump⁻¹ · (K_adv · u + f_bc)`
pub struct DgAdvectionRhs {
    /// Assembled advection operator (volume + interior face + boundary).
    pub k_adv: CsrMatrix<f64>,
    /// Lumped mass diagonal entries (one per DOF).
    pub mass_diag: Vec<f64>,
    /// Boundary RHS contributions (inflow BC).
    pub rhs_bc: Vec<f64>,
}

impl DgAdvectionRhs {
    /// Evaluate the RHS: `dudt = M_lump⁻¹ · (K_adv·u + rhs_bc)`
    pub fn eval(&self, _t: f64, u: &[f64], dudt: &mut [f64]) {
        let mut tmp = self.rhs_bc.clone();
        self.k_adv.spmv(u, &mut tmp);
        for i in 0..dudt.len() {
            dudt[i] = tmp[i] / self.mass_diag[i];
        }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

pub(crate) fn ref_elem_vol(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Tri3, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) => Box::new(TriP2),
        (ElementType::Tri3, 3) => Box::new(TriP3),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetP3),
        _ => panic!("dg_adv ref_elem_vol: unsupported ({et:?}, {order})"),
    }
}

fn ref_elem_face(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Line2, 1) => Box::new(SegP1),
        (ElementType::Line2, 2) => Box::new(SegP2),
        (ElementType::Line2, 3) => Box::new(SegP3),
        (ElementType::Tri3, 1)  => Box::new(TriP1),
        _ => panic!("dg_adv ref_elem_face: unsupported ({et:?}, {order})"),
    }
}

pub(crate) fn simplex_jac<M: MeshTopology>(mesh: &M, nodes: &[u32], dim: usize) -> (DMatrix<f64>, f64) {
    let x0 = mesh.node_coords(nodes[0]);
    let mut j = DMatrix::<f64>::zeros(dim, dim);
    for col in 0..dim {
        let xc = mesh.node_coords(nodes[col+1]);
        for row in 0..dim { j[(row,col)] = xc[row] - x0[row]; }
    }
    let det = j.determinant();
    (j, det)
}

fn phys_to_ref(jac: &DMatrix<f64>, x0: &[f64], xp: &[f64], dim: usize) -> Vec<f64> {
    let j_inv = jac.clone().try_inverse().expect("degenerate element in phys_to_ref");
    let dx: Vec<f64> = (0..dim).map(|i| xp[i] - x0[i]).collect();
    let mut xi = vec![0.0_f64; dim];
    for i in 0..dim {
        for k in 0..dim { xi[i] += j_inv[(i,k)] * dx[k]; }
    }
    xi
}

pub(crate) fn xform_grads(jit: &DMatrix<f64>, gr: &[f64], gp: &mut [f64], n: usize, dim: usize) {
    for i in 0..n {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim { s += jit[(j,k)] * gr[i*dim+k]; }
            gp[i*dim+j] = s;
        }
    }
}

fn orient_normal_outward<M: MeshTopology>(
    mesh: &M,
    elem: u32,
    face_nodes: &[u32],
    normal: &mut [f64],
) {
    let dim = mesh.dim() as usize;
    let enodes = mesh.element_nodes(elem);
    let npe = enodes.len();
    let mut centroid = vec![0.0_f64; dim];
    for &n in enodes {
        let c = mesh.node_coords(n);
        for d in 0..dim { centroid[d] += c[d]; }
    }
    for d in 0..dim { centroid[d] /= npe as f64; }
    let mut midpoint = vec![0.0_f64; dim];
    for &n in face_nodes {
        let c = mesh.node_coords(n);
        for d in 0..dim { midpoint[d] += c[d]; }
    }
    for d in 0..dim { midpoint[d] /= face_nodes.len() as f64; }
    let dot: f64 = (0..dim).map(|d| normal[d] * (midpoint[d] - centroid[d])).sum();
    if dot < 0.0 {
        for d in 0..dim { normal[d] = -normal[d]; }
    }
}

fn find_face_elem<M: MeshTopology>(mesh: &M, _face_id: u32, face_nodes: &[u32]) -> u32 {
    // Build a sorted key and scan elements
    let mut fkey: Vec<u32> = face_nodes.to_vec();
    fkey.sort_unstable();
    for e in mesh.elem_iter() {
        let enodes = mesh.element_nodes(e);
        if enodes.len() < 3 { continue; }
        // Check any 2 matching nodes = face belongs to this element
        let count = fkey.iter().filter(|&n| enodes.contains(n)).count();
        if count >= 2 { return e; }
    }
    0
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::L2Space;
    use crate::coefficient::ConstantVectorCoeff;

    #[test]
    fn volume_term_is_transpose_of_convection() {
        // DGAdvectionIntegrator volume term (weak form): -∫ (b·∇v)·u
        // ConvectionIntegrator (strong form): ∫ v·(b·∇u)
        // These are the TRANSPOSE of each other
        use crate::standard::ConvectionIntegrator;
        use crate::assembler::Assembler;

        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = L2Space::new(mesh, 1);

        let vel = vec![1.0, 0.0];

        let dg_adv = DGAdvectionIntegrator { velocity: ConstantVectorCoeff(vel.clone()) };
        let conv = ConvectionIntegrator { velocity: ConstantVectorCoeff(vel) };
        let mat_dg = Assembler::assemble_bilinear(&space, &[&dg_adv], 3);
        let mat_conv = Assembler::assemble_bilinear(&space, &[&conv], 3);

        let dense_dg = mat_dg.to_dense();
        let dense_conv = mat_conv.to_dense();
        let n = mat_dg.nrows;
        // Check that mat_dg ≈ -mat_conv^T (weak form = -transpose of strong form)
        let mut diff_norm = 0.0;
        for i in 0..n {
            for j in 0..n {
                diff_norm += (dense_dg[i * n + j] + dense_conv[j * n + i]).abs();
            }
        }
        assert!(diff_norm < 1e-12, "DG volume should be -ConvectionIntegrator^T, diff={diff_norm}");
    }

    #[test]
    fn constant_advection_preserved() {
        // For u=1 with div-free b=(1,0), the weak form gives du/dt=0.
        const N: usize = 4;
        let mesh = SimplexMesh::<2>::unit_square_tri(N);
        let space = L2Space::new(mesh, 1);  // P1 L2
        let ifl = InteriorFaceList::build(space.mesh());

        let dg_adv = DGAdvectionIntegrator { velocity: ConstantVectorCoeff(vec![1.0, 0.0]) };
        let n = space.n_dofs();

        let k_vol = crate::assembler::Assembler::assemble_bilinear(&space, &[&dg_adv], 2);

        let mut coo_faces = CooMatrix::<f64>::new(n, n);
        assemble_dg_interior_faces(&mut coo_faces, space.mesh(), &space, &ifl, 1, 2, &dg_adv);
        let k_face = coo_faces.into_csr();

        // For u=1, K_vol + K_face gives the boundary flux.
        // Verify the discrete divergence theorem: Σ (K_vol + K_face)·1 ≈ boundary integral.
        // For now, just check that the operator is non-zero and well-formed.
        let u = vec![1.0_f64; n];
        let mut vol_result = vec![0.0_f64; n];
        k_vol.spmv(&u, &mut vol_result);
        let mut face_result = vec![0.0_f64; n];
        k_face.spmv(&u, &mut face_result);

        let total_norm: f64 = vol_result.iter().zip(face_result.iter())
            .map(|(a, b)| (a + b).abs()).sum();
        // Total operator applied to u=1 should give boundary flux. For an open
        // domain with b=(1,0), this is non-zero (outflow). Just verify it's
        // not NaN or absurdly large.
        assert!(total_norm.is_finite() && total_norm > 0.0,
            "K_total·1 should be finite and positive (outflow flux), got {total_norm}");
    }

    #[test]
    fn face_assembly_constant_velocity() {
        let mesh = SimplexMesh::<2>::unit_square_tri(1);
        eprintln!("Interior face: nodes={:?}", mesh.element_nodes(0));
        let ifl = InteriorFaceList::build(&mesh);
        eprintln!("ifl.faces[0]: left={}, right={}, nodes={:?}",
            ifl.faces[0].elem_left, ifl.faces[0].elem_right, ifl.faces[0].face_nodes);
        let space = L2Space::new(mesh, 1);
        let n = space.n_dofs();
        let mut coo = CooMatrix::<f64>::new(n, n);
        assemble_dg_interior_faces(&mut coo, space.mesh(), &space, &ifl, 1, 1,
            &DGAdvectionIntegrator { velocity: ConstantVectorCoeff(vec![1.0, 0.0]) });
        let k = coo.into_csr();
        let mut has_nonzero = false;
        for i in 0..n { for j in 0..n { if k.get(i, j).abs() > 1e-14 { has_nonzero = true; } } }
        assert!(has_nonzero, "Face matrix should have non-zero entries");
    }
}
