//! H(curl) finite element space for Nédélec edge elements.
//!
//! ## DOF association
//!
//! Each DOF corresponds to a unique mesh edge.  The DOF functional is the
//! tangential line integral: `DOF_e(u) = ∫_e u · t̂ ds`.
//!
//! For lowest-order Nédélec (ND1):
//! - **2-D triangles**: 3 edge DOFs per element, `n_dofs = n_unique_edges`
//! - **3-D tetrahedra**: 6 edge DOFs per element, `n_dofs = n_unique_edges`
//!
//! ## Sign convention
//!
//! A global edge orientation is defined as "from smaller to larger vertex
//! index."  When a local edge traverses vertices in this same direction the
//! sign is +1; otherwise the sign is −1.  The assembler multiplies each
//! basis-function value by its sign to guarantee tangential continuity
//! across elements.

use std::collections::HashMap;

use fem_core::types::DofId;
use fem_element::quadrature::gauss_legendre_01;
use fem_linalg::Vector;
use fem_mesh::{topology::MeshTopology, ElementTransformation, ElementType};

use crate::dof_manager::{EdgeKey, FaceKey, QuadFaceKey};
use crate::fe_space::{FESpace, SpaceType};

// ─── Local edge tables ──────────────────────────────────────────────────────

/// Local edge vertex pairs for 2-D triangles (TriND1 ordering).
const TRI_EDGES: [(usize, usize); 3] = [(0, 1), (1, 2), (0, 2)];

/// Local edge vertex pairs for 2-D quadrilaterals (QuadND1 ordering).
const QUAD_EDGES: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];

/// Local edge vertex pairs for 3-D tetrahedra (TetND1 ordering).
const TET_EDGES: [(usize, usize); 6] = [
    (0, 1), (0, 2), (0, 3),
    (1, 2), (1, 3), (2, 3),
];

/// Local edge vertex pairs for 3-D hexahedra (Hex8 ordering).
///
/// Ordering MUST match the HexNDk basis function ordering:
///   edges 0..3:  x-edges (bottom-front, bottom-back, top-back, top-front)
///   edges 4..7:  y-edges (left-front, right-front, right-back, left-back)
///   edges 8..11: z-edges (left-front, right-front, right-back, left-back)
const HEX_EDGES: [(usize, usize); 12] = [
    (0, 1), (3, 2), (7, 6), (4, 5),   // x-edges (y0=-1,+1,+1,-1; z0=-1,-1,+1,+1)
    (0, 3), (1, 2), (5, 6), (4, 7),   // y-edges (x0=-1,+1,+1,-1; z0=-1,-1,+1,+1)
    (0, 4), (1, 5), (2, 6), (3, 7),   // z-edges (x0=-1,+1,+1,-1; y0=-1,-1,+1,+1)
];

/// Local edge vertex pairs for prism (Prism6 ordering).
/// Ordering MUST match PrismND1 element EDGES table.
const PRISM_EDGES: [(usize, usize); 9] = [
    (0, 1), (0, 2), (1, 2), // bottom triangle
    (3, 4), (3, 5), (4, 5), // top triangle
    (0, 3), (1, 4), (2, 5), // vertical
];

/// Local edge vertex pairs for pyramid (Pyramid5 ordering).
const PYRAMID_EDGES: [(usize, usize); 8] = [
    (0, 1), (1, 2), (2, 3), (3, 0), // base quad
    (0, 4), (1, 4), (2, 4), (3, 4), // apex edges
];

/// Local face definitions for 3-D tetrahedra (TetND2 ordering).
const TET_FACES: [(usize, usize, usize); 4] = [
    (1, 2, 3),
    (0, 2, 3),
    (0, 1, 3),
    (0, 1, 2),
];

/// Local quad-face definitions for 3-D hexahedra (Quad4 face ordering).
pub(crate) const HEX_QUAD_FACES: [(usize, usize, usize, usize); 6] = [
    (0, 1, 2, 3), // z=-1 (bottom)
    (4, 5, 6, 7), // z= 1 (top)
    (0, 1, 5, 4), // y=-1 (front)
    (2, 3, 7, 6), // y= 1 (back)
    (0, 3, 7, 4), // x=-1 (left)
    (1, 2, 6, 5), // x= 1 (right)
];

/// Local triangular faces for prism (Prism6 ordering): bottom + top.
const PRISM_TRI_FACES: [(usize, usize, usize); 2] = [
    (0, 1, 2),
    (3, 4, 5),
];

/// Local quad faces for prism (Prism6 ordering).
pub(crate) const PRISM_QUAD_FACES: [(usize, usize, usize, usize); 3] = [
    (0, 1, 4, 3),
    (1, 2, 5, 4),
    (0, 2, 5, 3),
];

/// Local triangular faces for pyramid (Pyramid5 ordering): 4 apex triangles.
const PYRAMID_TRI_FACES: [(usize, usize, usize); 4] = [
    (0, 1, 4),
    (1, 2, 4),
    (2, 3, 4),
    (3, 0, 4),
];

/// Local base quad face for pyramid.
pub(crate) const PYRAMID_QUAD_FACE: [(usize, usize, usize, usize); 1] = [
    (0, 1, 2, 3),
];

// ─── HCurlSpace ─────────────────────────────────────────────────────────────

/// H(curl) finite element space using Nédélec edge elements.
///
/// Constructed from a [`MeshTopology`] with triangular or tetrahedral elements.
/// Currently supports order 1 (ND1).
// MFEM: ND_FECollection (Nedelec)
pub struct HCurlSpace<M: MeshTopology> {
    mesh: M,
    order: u8,
    n_dofs: usize,
    /// Flat global DOF indices: `[elem0_dof0, elem0_dof1, ..., elem1_dof0, ...]`
    dofs_flat: Vec<DofId>,
    /// Orientation signs (±1.0), same layout as `dofs_flat`.
    signs_flat: Vec<f64>,
    /// CSR-like offsets into `dofs_flat` / `signs_flat`, length `n_elems + 1`.
    /// `dofs_flat[offsets[e]..offsets[e+1]]` are the DOFs for element `e`.
    elem_offsets: Vec<usize>,
    /// Edge → global DOF map (for boundary queries and interpolation).
    edge_to_dof: HashMap<EdgeKey, DofId>,
    /// Face → first global DOF map for 3D ND2 (second = first + 1).
    face_to_dof: HashMap<FaceKey, DofId>,
    /// Quad-face → first global DOF for hex NDk (2k(k-1) DOFs per face).
    quad_face_to_dof: HashMap<QuadFaceKey, DofId>,
    /// Spatial dimension.
    dim: usize,
    /// Cell type used by this space.
    cell_type: ElementType,
}

impl<M: MeshTopology> HCurlSpace<M> {
    /// Construct an H(curl) space of the given order on `mesh`.
    ///
    /// Supports ND1 (order 1) and NDk (order k >= 2) for Tri3/Tri6, Quad4/Quad8, Tet4/Tet10, Hex8/Hex20.
    pub fn new(mesh: M, order: u8) -> Self {
        assert!(order >= 1, "HCurlSpace: order must be >= 1");
        let dim = mesh.dim() as usize;
        assert!(mesh.n_elements() > 0, "HCurlSpace: mesh must contain at least one element");
        let k = order as usize;
        let dofs_per_edge = k;
        let n_elem = mesh.n_elements();

        let mut edge_to_dof: HashMap<EdgeKey, DofId> = HashMap::new();
        let mut face_to_dof: HashMap<FaceKey, DofId> = HashMap::new();
        let mut quad_face_to_dof: HashMap<QuadFaceKey, DofId> = HashMap::new();
        let mut next_dof: DofId = 0;
        let mut dofs_flat = Vec::new();
        let mut signs_flat = Vec::new();
        let mut elem_offsets = Vec::with_capacity(n_elem + 1);
        elem_offsets.push(0);
        let first_cell_type = mesh.element_type(0);

        for e in 0..n_elem as u32 {
            let cell_type = mesh.element_type(e);
            let verts = mesh.element_nodes(e);

            // Per-element-type local edges.
            let local_edges: &[(usize, usize)] = match cell_type {
                ElementType::Tri3 | ElementType::Tri6 => &TRI_EDGES,
                ElementType::Quad4 | ElementType::Quad8 => &QUAD_EDGES,
                ElementType::Tet4 | ElementType::Tet10 => &TET_EDGES,
                ElementType::Hex8 | ElementType::Hex20 => &HEX_EDGES,
                ElementType::Prism6 => &PRISM_EDGES,
                ElementType::Pyramid5 => &PYRAMID_EDGES,
                _ => panic!("HCurlSpace: unsupported element type {cell_type:?}"),
            };

            // Edge DOFs.
            for &(li, lj) in local_edges {
                let (gi, gj) = (verts[li], verts[lj]);
                let key = EdgeKey::new(gi, gj);
                let sign = if gi < gj { 1.0 } else { -1.0 };
                let nd = dofs_per_edge as u32;
                let first_dof = *edge_to_dof.entry(key).or_insert_with(|| {
                    let d = next_dof; next_dof += nd; d
                });
                for m in 0..nd as usize {
                    dofs_flat.push(first_dof + m as u32);
                    signs_flat.push(sign);
                }
            }

            // Face DOFs (NDk, k>=2).
            if k >= 2 && dim == 3 {
                let ndf = k * (k - 1);
                match cell_type {
                    ElementType::Tet4 | ElementType::Tet10 => {
                        for &(la, lb, lc) in &TET_FACES {
                            let key = FaceKey::new(verts[la], verts[lb], verts[lc]);
                            let first_dof = *face_to_dof.entry(key).or_insert_with(|| {
                                let d = next_dof; next_dof += ndf as u32; d
                            });
                            for m in 0..ndf { dofs_flat.push(first_dof + m as u32); signs_flat.push(1.0); }
                        }
                    }
                    ElementType::Hex8 | ElementType::Hex20 => {
                        let ndf_quad = 2 * k * (k - 1);
                        for &(la, lb, lc, ld) in &HEX_QUAD_FACES {
                            let key = QuadFaceKey::new(verts[la], verts[lb], verts[lc], verts[ld]);
                            let first_dof = *quad_face_to_dof.entry(key).or_insert_with(|| {
                                let d = next_dof; next_dof += ndf_quad as u32; d
                            });
                            for m in 0..ndf_quad { dofs_flat.push(first_dof + m as u32); signs_flat.push(1.0); }
                        }
                    }
                    ElementType::Prism6 => {
                        // Tri face DOFs
                        for &(la, lb, lc) in &PRISM_TRI_FACES {
                            let key = FaceKey::new(verts[la], verts[lb], verts[lc]);
                            let first_dof = *face_to_dof.entry(key).or_insert_with(|| {
                                let d = next_dof; next_dof += ndf as u32; d
                            });
                            for m in 0..ndf { dofs_flat.push(first_dof + m as u32); signs_flat.push(1.0); }
                        }
                        // Quad face DOFs
                        let ndf_quad = 2 * k * (k - 1);
                        for &(la, lb, lc, ld) in &PRISM_QUAD_FACES {
                            let key = QuadFaceKey::new(verts[la], verts[lb], verts[lc], verts[ld]);
                            let first_dof = *quad_face_to_dof.entry(key).or_insert_with(|| {
                                let d = next_dof; next_dof += ndf_quad as u32; d
                            });
                            for m in 0..ndf_quad { dofs_flat.push(first_dof + m as u32); signs_flat.push(1.0); }
                        }
                    }
                    ElementType::Pyramid5 => {
                        for &(la, lb, lc) in &PYRAMID_TRI_FACES {
                            let key = FaceKey::new(verts[la], verts[lb], verts[lc]);
                            let first_dof = *face_to_dof.entry(key).or_insert_with(|| {
                                let d = next_dof; next_dof += ndf as u32; d
                            });
                            for m in 0..ndf { dofs_flat.push(first_dof + m as u32); signs_flat.push(1.0); }
                        }
                        let (la, lb, lc, ld) = PYRAMID_QUAD_FACE[0];
                        let ndf_quad = 2 * k * (k - 1);
                        let key = QuadFaceKey::new(verts[la], verts[lb], verts[lc], verts[ld]);
                        let first_dof = *quad_face_to_dof.entry(key).or_insert_with(|| {
                            let d = next_dof; next_dof += ndf_quad as u32; d
                        });
                        for m in 0..ndf_quad { dofs_flat.push(first_dof + m as u32); signs_flat.push(1.0); }
                    }
                    _ => {}
                }
            }

            // Interior DOFs (NDk, k>=3 for Tet, k>=2 for others).
            let interior_count: u32 = match (dim, cell_type) {
                (2, ElementType::Tri3 | ElementType::Tri6) if k >= 2 => (k * (k - 1)) as u32,
                (2, ElementType::Quad4 | ElementType::Quad8) if k >= 2 => (2 * k * (k - 1)) as u32,
                (3, ElementType::Tet4 | ElementType::Tet10) if k >= 3 => (k * (k - 1) * (k - 2) / 2) as u32,
                (3, ElementType::Hex8 | ElementType::Hex20) if k >= 2 => (3 * k * (k - 1) * (k - 1)) as u32,
                (3, ElementType::Prism6) if k >= 2 => (k * (k - 1) * (k - 1)) as u32,
                (3, ElementType::Pyramid5) if k >= 2 => (k * (k - 1) * (k - 1)) as u32,
                _ => 0,
            };
            for _ in 0..interior_count {
                dofs_flat.push(next_dof);
                next_dof += 1;
                signs_flat.push(1.0);
            }

            elem_offsets.push(dofs_flat.len());
        }

        HCurlSpace {
            mesh,
            order,
            n_dofs: next_dof as usize,
            dofs_flat,
            signs_flat,
            elem_offsets,
            edge_to_dof,
            face_to_dof,
            quad_face_to_dof,
            dim,
            cell_type: first_cell_type,
        }
    }

    /// Return the polynomial order of this space.
    pub fn order(&self) -> u8 { self.order }

    /// Orientation signs (±1.0) for the DOFs on element `elem`.
    ///
    /// `signs[i]` multiplies basis function `i` on this element so that the
    /// tangential trace is consistent with the global edge orientation.
    pub fn element_signs(&self, elem: u32) -> &[f64] {
        let start = self.elem_offsets[elem as usize];
        let end = self.elem_offsets[elem as usize + 1];
        &self.signs_flat[start..end]
    }

    /// Look up the global DOF index for a given edge (by canonical key).
    pub fn edge_dof(&self, edge: EdgeKey) -> Option<DofId> {
        self.edge_to_dof.get(&edge).copied()
    }

    /// Look up all global DOFs associated with a given edge.
    pub fn edge_dofs(&self, edge: EdgeKey) -> Option<Vec<DofId>> {
        self.edge_to_dof.get(&edge).map(|&first| {
            (0..self.order as DofId).map(|m| first + m).collect()
        })
    }

    /// Number of unique edges in the mesh (== `n_dofs` for ND1).
    pub fn n_edges(&self) -> usize {
        self.edge_to_dof.len()
    }

    /// Number of unique faces in 3D ND2 mode.
    pub fn n_faces(&self) -> usize {
        self.face_to_dof.len()
    }

    /// Number of unique quad faces for hex NDk.
    pub fn n_quad_faces(&self) -> usize {
        self.quad_face_to_dof.len()
    }

    /// Look up the first global DOF for a triangular face (Tet NDk, k≥2).
    /// Returns `None` for ND1 or if the face is not found.
    pub fn face_dof(&self, face: FaceKey) -> Option<DofId> {
        self.face_to_dof.get(&face).copied()
    }

    /// Look up all global DOFs associated with a quad face (hex NDk, k≥2).
    pub fn quad_face_dofs(&self, key: QuadFaceKey) -> Option<Vec<DofId>> {
        if self.order < 2 { return None; }
        let ndf = 2 * self.order as DofId * (self.order as DofId - 1);
        self.quad_face_to_dof.get(&key).map(|&first| {
            (0..ndf).map(|m| first + m).collect()
        })
    }

    /// Vector-valued interpolation via the Nédélec DOF functional.
    ///
    /// ## ND1 (order 1)
    /// Midpoint-evaluated tangential moment per edge.
    ///
    /// ## NDk (k >= 2)
    /// k-point Gauss-Legendre edge moments. Tri interior/Tet face moments for k >= 2.
    pub fn interpolate_vector(&self, f: &dyn Fn(&[f64]) -> Vec<f64>) -> Vector<f64> {
        let mut result = Vector::zeros(self.n_dofs);
        let k = self.order as usize;

        if k == 1 {
            // ND1: midpoint rule per edge.
            for (&EdgeKey(a, b), &dof) in &self.edge_to_dof {
                let pa = self.mesh.node_coords(a);
                let pb = self.mesh.node_coords(b);
                let mid: Vec<f64> = (0..self.dim).map(|d| 0.5 * (pa[d] + pb[d])).collect();
                let tangent: Vec<f64> = (0..self.dim).map(|d| pb[d] - pa[d]).collect();
                let fval = f(&mid);
                let dot: f64 = fval.iter().zip(&tangent).map(|(fi, ti)| fi * ti).sum();
                result.as_slice_mut()[dof as usize] = dot;
            }
            return result;
        }

        // NDk (k >= 2): edge moments via Gauss quadrature.
        // Use 3-point rule for k=2 (preserves existing de Rham tests),
        // k-point rule for k > 2.
        let (gl_pts, gl_wts) = if k == 2 {
            let sq_3_5: f64 = (3.0_f64 / 5.0).sqrt();
            (vec![0.5 * (1.0 - sq_3_5), 0.5, 0.5 * (1.0 + sq_3_5)],
             vec![5.0_f64 / 18.0, 4.0 / 9.0, 5.0 / 18.0])
        } else {
            gauss_legendre_01(k)
        };

        // Step 1 — edge DOFs.
        let npts = gl_pts.len();
        for (&EdgeKey(a, b), &first_dof) in &self.edge_to_dof {
            let pa = self.mesh.node_coords(a);
            let pb = self.mesh.node_coords(b);
            let dim = self.dim;
            let tangent: Vec<f64> = (0..dim).map(|d| pb[d] - pa[d]).collect();

            let mut moments = vec![0.0_f64; k];
            for ki in 0..npts {
                let t = gl_pts[ki];
                let w = gl_wts[ki];
                let pt: Vec<f64> = (0..dim).map(|d| pa[d] + t * tangent[d]).collect();
                let fval = f(&pt);
                let flux: f64 = fval.iter().zip(&tangent).map(|(fi, ti)| fi * ti).sum();
                for m in 0..k { moments[m] += w * flux * t.powi(m as i32); }
            }
            let r = result.as_slice_mut();
            for m in 0..k { r[first_dof as usize + m] = moments[m]; }
        }

        // Step 2 — interior/face DOFs.
        if self.dim == 2 {
            if k >= 2 && matches!(self.cell_type, ElementType::Tri3 | ElementType::Tri6) {
                if k == 2 {
                    // ND2: two vector-component interior moments (matching original discrete_op tests).
                    let qr = fem_element::quadrature::tri_rule(4);
                    let n_elem = self.mesh.n_elements();
                    for e in 0..n_elem as u32 {
                        let dofs = self.element_dofs(e);
                        let bub0 = dofs[dofs.len() - 2] as usize;
                        let bub1 = dofs[dofs.len() - 1] as usize;
                        let nodes = self.mesh.element_nodes(e);
                        let transform = ElementTransformation::from_simplex_nodes(&self.mesh, nodes);
                        let det_j = transform.det_j().abs();
                        let x0 = self.mesh.node_coords(nodes[0]);
                        let x1 = self.mesh.node_coords(nodes[1]);
                        let x2 = self.mesh.node_coords(nodes[2]);
                        let j00 = x1[0]-x0[0]; let j01 = x2[0]-x0[0];
                        let j10 = x1[1]-x0[1]; let j11 = x2[1]-x0[1];
                        let mut int_x = 0.0_f64; let mut int_y = 0.0_f64;
                        for (xi, &w) in qr.points.iter().zip(qr.weights.iter()) {
                            let xp = [x0[0]+j00*xi[0]+j01*xi[1], x0[1]+j10*xi[0]+j11*xi[1]];
                            let fv = f(&xp);
                            int_x += w * fv[0]; int_y += w * fv[1];
                        }
                        let r = result.as_slice_mut();
                        r[bub0] = int_x * det_j;
                        r[bub1] = int_y * det_j;
                    }
                } else {
                    // NDk (k >= 3): monomial-weighted tangential moments.
                    let n_interior = k * (k - 1);
                    let qr = fem_element::quadrature::tri_rule((2 * k) as u8);
                    let n_elem = self.mesh.n_elements();
                    for e in 0..n_elem as u32 {
                        let dofs = self.element_dofs(e);
                        let b_start = dofs.len() - n_interior;
                        let nodes = self.mesh.element_nodes(e);
                        let transform = ElementTransformation::from_simplex_nodes(&self.mesh, nodes);
                        let det_j = transform.det_j().abs();
                        let jit = transform.jacobian_inv_t();
                        let x0 = self.mesh.node_coords(nodes[0]);
                        let x1 = self.mesh.node_coords(nodes[1]);
                        let x2 = self.mesh.node_coords(nodes[2]);
                        let j00 = x1[0]-x0[0]; let j10 = x1[1]-x0[1];
                        let j01 = x2[0]-x0[0]; let j11 = x2[1]-x0[1];
                        let mut row = 0usize;
                        for p in 0..k { for q in 0..(k - 1) { if p + q < k {
                            let a = p; let b = q;
                            let mut moment = 0.0;
                            for (xi, &w) in qr.points.iter().zip(qr.weights.iter()) {
                                let xp = [x0[0]+j00*xi[0]+j01*xi[1], x0[1]+j10*xi[0]+j11*xi[1]];
                                let fv = f(&xp);
                                let ur0 = det_j * (jit[(0,0)]*fv[0] + jit[(1,0)]*fv[1]);
                                let ur1 = det_j * (jit[(0,1)]*fv[0] + jit[(1,1)]*fv[1]);
                                moment += w * (ur0 * xi[0].powi(a as i32) * xi[1].powi(b as i32)
                                             + ur1 * xi[0].powi(b as i32) * xi[1].powi(a as i32));
                            }
                            result.as_slice_mut()[dofs[b_start + row] as usize] = moment;
                            row += 1;
                        }}}
                    }
                }
            }
        } else if self.dim == 3 && k >= 2 && matches!(self.cell_type, ElementType::Tet4 | ElementType::Tet10) {
            // Tet NDk face DOFs.
            let nf = k * (k - 1);
            if nf > 0 {
                if k == 2 {
                    // ND2 face DOFs: two tangential moments (constant weighting).
                    let qr_face = fem_element::quadrature::tri_rule(4);
                    for (&FaceKey(a, b, c), &first_dof) in &self.face_to_dof {
                        let pa = self.mesh.node_coords(a);
                        let pb = self.mesh.node_coords(b);
                        let pc = self.mesh.node_coords(c);
                        let ds = [pb[0]-pa[0], pb[1]-pa[1], pb[2]-pa[2]];
                        let dt = [pc[0]-pa[0], pc[1]-pa[1], pc[2]-pa[2]];
                        let cross = [ds[1]*dt[2]-ds[2]*dt[1], ds[2]*dt[0]-ds[0]*dt[2], ds[0]*dt[1]-ds[1]*dt[0]];
                        let jac_area = (cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]).sqrt();
                        let mut m0 = 0.0_f64; let mut m1 = 0.0_f64;
                        for (xi, &w) in qr_face.points.iter().zip(qr_face.weights.iter()) {
                            let (s, t) = (xi[0], xi[1]);
                            let pt = [pa[0]+s*ds[0]+t*dt[0], pa[1]+s*ds[1]+t*dt[1], pa[2]+s*ds[2]+t*dt[2]];
                            let fv = f(&pt);
                            let d_sigma = w * jac_area;
                            m0 += d_sigma * (fv[0]*ds[0] + fv[1]*ds[1] + fv[2]*ds[2]);
                            m1 += d_sigma * (fv[0]*dt[0] + fv[1]*dt[1] + fv[2]*dt[2]);
                        }
                        let r = result.as_slice_mut();
                        r[first_dof as usize] = m0;
                        r[first_dof as usize + 1] = m1;
                    }
                } else {
                    // NDk (k >= 3): polynomial-weighted tangential moments.
                    let qr_face = fem_element::quadrature::tri_rule((2 * k) as u8);
                    for (&FaceKey(a, b, c), &first_dof) in &self.face_to_dof {
                        let pa = self.mesh.node_coords(a);
                        let pb = self.mesh.node_coords(b);
                        let pc = self.mesh.node_coords(c);
                        let ds = [pb[0]-pa[0], pb[1]-pa[1], pb[2]-pa[2]];
                        let dt = [pc[0]-pa[0], pc[1]-pa[1], pc[2]-pa[2]];
                        let cross = [ds[1]*dt[2]-ds[2]*dt[1], ds[2]*dt[0]-ds[0]*dt[2], ds[0]*dt[1]-ds[1]*dt[0]];
                        let jac_area = (cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]).sqrt();
                        let mut moments = vec![0.0_f64; nf];
                        for (xi, &w) in qr_face.points.iter().zip(qr_face.weights.iter()) {
                            let (s, t) = (xi[0], xi[1]);
                            let pt = [pa[0]+s*ds[0]+t*dt[0], pa[1]+s*ds[1]+t*dt[1], pa[2]+s*ds[2]+t*dt[2]];
                            let fv = f(&pt);
                            let d_sigma = w * jac_area;
                            let mut idx = 0usize;
                            for p in 0..k-1 { for q in 0..k-1-p {
                                moments[idx] += d_sigma * (fv[0]*ds[0]+fv[1]*ds[1]+fv[2]*ds[2]) * s.powi(p as i32) * t.powi(q as i32);
                                idx += 1;
                                moments[idx] += d_sigma * (fv[0]*dt[0]+fv[1]*dt[1]+fv[2]*dt[2]) * s.powi(p as i32) * t.powi(q as i32);
                                idx += 1;
                            }}
                        }
                        let r = result.as_slice_mut();
                        for m in 0..nf { r[first_dof as usize + m] = moments[m]; }
                    }
                }
            }
        }
        result
    }
}

impl<M: MeshTopology> FESpace for HCurlSpace<M> {
    type Mesh = M;

    fn mesh(&self) -> &M { &self.mesh }

    fn n_dofs(&self) -> usize { self.n_dofs }

    fn element_dofs(&self, elem: u32) -> &[DofId] {
        let start = self.elem_offsets[elem as usize];
        let end = self.elem_offsets[elem as usize + 1];
        &self.dofs_flat[start..end]
    }

    fn interpolate(&self, _f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        Vector::zeros(self.n_dofs)
    }

    fn space_type(&self) -> SpaceType { SpaceType::HCurl }

    fn order(&self) -> u8 { self.order }

    fn element_signs(&self, elem: u32) -> Option<&[f64]> {
        Some(self.element_signs(elem))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use fem_core::{ElemId, FaceId, NodeId};
    use fem_mesh::Mesh;

    #[derive(Clone)]
    struct OneQuadMesh {
        nodes: Vec<[f64; 2]>,
        elem: [NodeId; 4],
        bfaces: Vec<[NodeId; 2]>,
        btags: Vec<i32>,
    }

    impl OneQuadMesh {
        fn unit() -> Self {
            Self {
                nodes: vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
                elem: [0, 1, 2, 3],
                bfaces: vec![[0, 1], [1, 2], [2, 3], [3, 0]],
                btags: vec![1, 2, 3, 4],
            }
        }
    }

    impl MeshTopology for OneQuadMesh {
        fn dim(&self) -> u8 { 2 }
        fn n_nodes(&self) -> usize { self.nodes.len() }
        fn n_elements(&self) -> usize { 1 }
        fn n_boundary_faces(&self) -> usize { self.bfaces.len() }
        fn element_nodes(&self, _elem: ElemId) -> &[NodeId] { &self.elem }
        fn element_type(&self, _elem: ElemId) -> ElementType { ElementType::Quad4 }
        fn element_tag(&self, _elem: ElemId) -> i32 { 1 }
        fn node_coords(&self, node: NodeId) -> &[f64] { &self.nodes[node as usize] }
        fn face_nodes(&self, face: FaceId) -> &[NodeId] { &self.bfaces[face as usize] }
        fn face_tag(&self, face: FaceId) -> i32 { self.btags[face as usize] }
        fn face_elements(&self, _face: FaceId) -> (ElemId, Option<ElemId>) { (0, None) }
    }

    #[derive(Clone)]
    struct OneHexMesh {
        nodes: Vec<[f64; 3]>,
        elem: [NodeId; 8],
        bfaces: Vec<[NodeId; 4]>,
        btags: Vec<i32>,
    }

    impl OneHexMesh {
        fn unit() -> Self {
            Self {
                nodes: vec![
                    [0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,0.0],
                    [0.0,0.0,1.0],[1.0,0.0,1.0],[1.0,1.0,1.0],[0.0,1.0,1.0],
                ],
                elem: [0,1,2,3,4,5,6,7],
                bfaces: vec![[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]],
                btags: vec![1,2,3,4,5,6],
            }
        }
    }

    impl MeshTopology for OneHexMesh {
        fn dim(&self) -> u8 { 3 }
        fn n_nodes(&self) -> usize { self.nodes.len() }
        fn n_elements(&self) -> usize { 1 }
        fn n_boundary_faces(&self) -> usize { self.bfaces.len() }
        fn element_nodes(&self, _elem: ElemId) -> &[NodeId] { &self.elem }
        fn element_type(&self, _elem: ElemId) -> ElementType { ElementType::Hex8 }
        fn element_tag(&self, _elem: ElemId) -> i32 { 1 }
        fn node_coords(&self, node: NodeId) -> &[f64] { &self.nodes[node as usize] }
        fn face_nodes(&self, face: FaceId) -> &[NodeId] { &self.bfaces[face as usize] }
        fn face_tag(&self, face: FaceId) -> i32 { self.btags[face as usize] }
        fn face_elements(&self, _face: FaceId) -> (ElemId, Option<ElemId>) { (0, None) }
    }

    #[test]
    fn hcurl_dof_count_tri() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);
        assert_eq!(space.element_dofs(0).len(), 3);
        // Each triangle has 3 edges, 32 triangles, but edges are shared.
        // Expected: 56 unique edges.
        assert_eq!(space.n_dofs(), 56, "n_dofs should equal number of unique edges");
    }

    #[test]
    fn hcurl_shared_edge_dof() {
        // 1×1 mesh → 2 triangles sharing the diagonal edge.
        let mesh = Mesh::<2>::unit_square_tri(1);
        let space = HCurlSpace::new(mesh, 1);
        assert_eq!(space.mesh().n_elements(), 2);

        let dofs0 = space.element_dofs(0);
        let dofs1 = space.element_dofs(1);

        // At least one DOF must be shared between the two elements.
        let shared: Vec<_> = dofs0.iter().filter(|d| dofs1.contains(d)).collect();
        assert!(!shared.is_empty(), "adjacent triangles must share at least one edge DOF");
    }

    #[test]
    fn hcurl_signs_consistent_on_shared_edge() {
        // Two triangles sharing an edge: verify signs are well-defined (±1)
        // and that both elements reference the same global DOF.
        // Note: signs are NOT necessarily opposite — they are both relative
        // to the global edge orientation (min→max vertex ID).
        let mesh = Mesh::<2>::unit_square_tri(1);
        let space = HCurlSpace::new(mesh, 1);

        let dofs0 = space.element_dofs(0);
        let signs0 = space.element_signs(0);
        let dofs1 = space.element_dofs(1);
        let signs1 = space.element_signs(1);

        // All signs must be ±1.
        for s in signs0.iter().chain(signs1.iter()) {
            assert!((s.abs() - 1.0).abs() < 1e-14, "sign must be ±1, got {s}");
        }

        // At least one shared DOF.
        let shared: Vec<_> = dofs0.iter().filter(|d| dofs1.contains(d)).collect();
        assert!(!shared.is_empty(), "adjacent triangles must share at least one edge DOF");
    }

    #[test]
    fn hcurl_space_type() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let space = HCurlSpace::new(mesh, 1);
        assert_eq!(space.space_type(), SpaceType::HCurl);
    }

    #[test]
    fn hcurl_dof_count_quad_nd1() {
        let mesh = OneQuadMesh::unit();
        let space = HCurlSpace::new(mesh, 1);
        assert_eq!(space.element_dofs(0).len(), 4);
        assert_eq!(space.n_dofs(), 4);
    }

    #[test]
    fn hcurl_dof_count_hex_nd1() {
        let mesh = OneHexMesh::unit();
        let space = HCurlSpace::new(mesh, 1);
        assert_eq!(space.element_dofs(0).len(), 12);
        assert_eq!(space.n_dofs(), 12);
    }

    #[test]
    fn hcurl_dof_count_quad_nd2() {
        let mesh = OneQuadMesh::unit();
        let space = HCurlSpace::new(mesh, 2);
        assert_eq!(space.element_dofs(0).len(), 12, "QuadND2: 8 edge + 4 interior");
        assert_eq!(space.n_dofs(), 12);
    }

    #[test]
    fn hcurl_dof_count_hex_nd2() {
        let mesh = OneHexMesh::unit();
        let space = HCurlSpace::new(mesh, 2);
        assert_eq!(space.element_dofs(0).len(), 54, "HexND2: 24 edge + 30 face/interior");
        assert_eq!(space.n_dofs(), 54);
    }

    #[test]
    fn hcurl_interpolate_vector_constant_quad_nd1() {
        let mesh = OneQuadMesh::unit();
        let space = HCurlSpace::new(mesh, 1);
        let v = space.interpolate_vector(&|_x| vec![1.0, 0.0]);

        let vals = v.as_slice();
        assert_eq!(vals.len(), 4);
        assert!((vals[0] - 2.0).abs() < 1e-12);
        assert!(vals[1].abs() < 1e-12);
        assert!((vals[2] + 2.0).abs() < 1e-12);
        assert!(vals[3].abs() < 1e-12);
    }

    #[test]
    fn hcurl_interpolate_vector_constant() {
        // Interpolate a constant vector field F = (1, 0).
        // DOF value on each edge = F · tangent = tangent_x.
        let mesh = Mesh::<2>::unit_square_tri(2);
        let space = HCurlSpace::new(mesh, 1);
        let v = space.interpolate_vector(&|_x| vec![1.0, 0.0]);
        // All DOF values should be finite and within the range of edge lengths.
        for &val in v.as_slice() {
            assert!(val.is_finite(), "interpolated value should be finite");
        }
    }

    #[test]
    fn hcurl_nd2_tet_local_dof_layout() {
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let space = HCurlSpace::new(mesh, 2);

        assert_eq!(space.element_dofs(0).len(), 20, "TetND2 should have 20 local DOFs");
        assert_eq!(space.element_signs(0).len(), 20, "TetND2 sign array length mismatch");
    }

    #[test]
    fn hcurl_nd2_tet_global_dof_count_matches_edges_faces() {
        let mesh = Mesh::<3>::unit_cube_tet(2);

        let mut edges: HashSet<EdgeKey> = HashSet::new();
        let mut faces: HashSet<FaceKey> = HashSet::new();
        for e in 0..mesh.n_elements() as u32 {
            let ns = mesh.element_nodes(e);
            for &(i, j) in &TET_EDGES {
                edges.insert(EdgeKey::new(ns[i], ns[j]));
            }
            for &(i, j, k) in &TET_FACES {
                faces.insert(FaceKey::new(ns[i], ns[j], ns[k]));
            }
        }

        let space = HCurlSpace::new(mesh, 2);
        let expected = 2 * edges.len() + 2 * faces.len();
        assert_eq!(space.n_dofs(), expected, "ND2 3D global DOF count should be 2*n_edges + 2*n_faces");
    }

    // ─── ND3+ tests ───────────────────────────────────────────────────────────

    #[test]
    fn hcurl_nd3_quad_dof_count() {
        let mesh = OneQuadMesh::unit();
        let space = HCurlSpace::new(mesh, 3);
        assert_eq!(space.element_dofs(0).len(), 24, "QuadND3: 12 edge + 12 interior");
        assert_eq!(space.n_dofs(), 24);
    }

    #[test]
    fn hcurl_nd3_hex_dof_count() {
        let mesh = OneHexMesh::unit();
        let space = HCurlSpace::new(mesh, 3);
        assert_eq!(space.element_dofs(0).len(), 144, "HexND3: 36 edge + 108 face/interior");
        assert_eq!(space.n_dofs(), 144);
    }

    #[test]
    fn hcurl_nd3_tet_dof_count() {
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let space = HCurlSpace::new(mesh, 3);
        assert_eq!(space.element_dofs(0).len(), 45, "TetND3: k*(k+2)*(k+3)/2 = 45");
    }

    #[test]
    fn hcurl_nd3_tri_dof_count() {
        let mesh = Mesh::<2>::unit_square_tri(1);
        let space = HCurlSpace::new(mesh, 3);
        assert_eq!(space.element_dofs(0).len(), 15, "TriND3: k*(k+2) = 15");
    }

    #[test]
    fn hcurl_nd4_hex_dof_count() {
        let mesh = OneHexMesh::unit();
        let space = HCurlSpace::new(mesh, 4);
        assert_eq!(space.element_dofs(0).len(), 300, "HexND4: 48 edge + 252 face/interior");
        assert_eq!(space.n_dofs(), 300);
    }

    #[test]
    fn hcurl_nd3_interpolate_linear_field() {
        let mesh = OneQuadMesh::unit();
        let space = HCurlSpace::new(mesh, 3);
        let v = space.interpolate_vector(&|_x| vec![1.0, 0.0]);
        assert_eq!(v.as_slice().len(), 24, "QuadND3: 24 DOFs");
        for &val in v.as_slice() { assert!(val.is_finite()); }
    }
}
