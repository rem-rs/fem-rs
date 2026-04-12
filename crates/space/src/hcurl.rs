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
use fem_element::{TriND2, VectorReferenceElement};
use fem_linalg::Vector;
use fem_mesh::{topology::MeshTopology, ElementTransformation, ElementType};

use crate::dof_manager::{EdgeKey, FaceKey};
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
const HEX_EDGES: [(usize, usize); 12] = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
];

/// Local face definitions for 3-D tetrahedra (TetND2 ordering).
const TET_FACES: [(usize, usize, usize); 4] = [
    (1, 2, 3),
    (0, 2, 3),
    (0, 1, 3),
    (0, 1, 2),
];

// ─── HCurlSpace ─────────────────────────────────────────────────────────────

/// H(curl) finite element space using Nédélec edge elements.
///
/// Constructed from a [`MeshTopology`] with triangular or tetrahedral elements.
/// Currently supports order 1 (ND1).
pub struct HCurlSpace<M: MeshTopology> {
    mesh: M,
    order: u8,
    n_dofs: usize,
    /// Flat global DOF indices: `[elem0_dof0, elem0_dof1, ..., elem1_dof0, ...]`
    dofs_flat: Vec<DofId>,
    /// Orientation signs (±1.0), same layout as `dofs_flat`.
    signs_flat: Vec<f64>,
    /// Number of DOFs per element.
    dofs_per_elem: usize,
    /// Edge → global DOF map (for boundary queries and interpolation).
    edge_to_dof: HashMap<EdgeKey, DofId>,
    /// Face -> first global DOF map for 3D ND2 (second = first + 1).
    face_to_dof: HashMap<FaceKey, DofId>,
    /// Spatial dimension.
    dim: usize,
    /// Cell type used by this space.
    cell_type: ElementType,
}

impl<M: MeshTopology> HCurlSpace<M> {
    /// Construct an H(curl) space of the given order on `mesh`.
    ///
    /// # Panics
    /// - If `order > 2` (only ND1 and ND2 are currently supported).
    /// - If the mesh is neither 2-D triangles nor 3-D tetrahedra.
    pub fn new(mesh: M, order: u8) -> Self {
        assert!((1..=2).contains(&order), "HCurlSpace: only orders 1 (ND1) and 2 (ND2) are supported");
        let dim = mesh.dim() as usize;

        assert!(mesh.n_elements() > 0, "HCurlSpace: mesh must contain at least one element");
        let cell_type = mesh.element_type(0);
        for e in 1..mesh.n_elements() as u32 {
            assert_eq!(
                mesh.element_type(e),
                cell_type,
                "HCurlSpace: mixed element types are not supported"
            );
        }

        match (cell_type, order) {
            (ElementType::Quad8, _) => {
                panic!("HCurlSpace: quadrilateral support is currently Quad4 only")
            }
            (ElementType::Hex20, _) => {
                panic!("HCurlSpace: hexahedral support is currently Hex8 only")
            }
            _ => {}
        }

        let local_edges: &[(usize, usize)] = match cell_type {
            ElementType::Tri3 | ElementType::Tri6 => &TRI_EDGES,
            ElementType::Quad4 => &QUAD_EDGES,
            ElementType::Tet4 | ElementType::Tet10 => &TET_EDGES,
            ElementType::Hex8 => &HEX_EDGES,
            _ => panic!("HCurlSpace: unsupported element type {cell_type:?}"),
        };

        // DOFs per element:
        //   ND1: 1 DOF per edge only
        //   ND2(2D): 2 DOFs per edge + 2 interior bubble DOFs
        //   ND2(3D): 2 DOFs per edge + 2 DOFs per face
        let dofs_per_edge = order as usize;
        let face_dofs_per_face = match (order, dim, cell_type) {
            (2, 3, ElementType::Tet4 | ElementType::Tet10) => 2,
            _ => 0,
        };
        let interior_dofs_per_elem = match (order, dim, cell_type) {
            (1, _, _) => 0,
            (2, 2, ElementType::Tri3 | ElementType::Tri6) => 2,  // TriND2: 2 interior DOFs
            (2, 2, ElementType::Quad4) => 0,
            (2, 3, ElementType::Tet4 | ElementType::Tet10) => 0, // TetND2 has no volume moments in current element definition
            (2, 3, ElementType::Hex8) => 0,
            _ => panic!("unsupported"),
        };
        let n_local_faces = match cell_type {
            ElementType::Tet4 | ElementType::Tet10 if dim == 3 => TET_FACES.len(),
            _ => 0,
        };
        let dofs_per_elem =
            local_edges.len() * dofs_per_edge + n_local_faces * face_dofs_per_face + interior_dofs_per_elem;
        let n_elem = mesh.n_elements();

        let mut edge_to_dof: HashMap<EdgeKey, DofId> = HashMap::new();
        let mut face_to_dof: HashMap<FaceKey, DofId> = HashMap::new();
        let mut next_dof: DofId = 0;
        let mut dofs_flat = Vec::with_capacity(n_elem * dofs_per_elem);
        let mut signs_flat = Vec::with_capacity(n_elem * dofs_per_elem);

        for e in 0..n_elem as u32 {
            let verts = mesh.element_nodes(e);
            for &(li, lj) in local_edges {
                let (gi, gj) = (verts[li], verts[lj]);
                let key = EdgeKey::new(gi, gj);
                let sign = if gi < gj { 1.0 } else { -1.0 };

                if dofs_per_edge == 1 {
                    // ND1: one DOF per edge
                    let dof = *edge_to_dof.entry(key).or_insert_with(|| { let d=next_dof; next_dof+=1; d });
                    dofs_flat.push(dof);
                    signs_flat.push(sign);
                } else {
                    // ND2: two DOFs per edge, stored as key → first_dof (second = first+1)
                    let first_dof = *edge_to_dof.entry(key).or_insert_with(|| {
                        let d = next_dof; next_dof += 2; d
                    });
                    dofs_flat.push(first_dof);
                    dofs_flat.push(first_dof + 1);
                    // Both edge DOFs share the same orientation sign.
                    signs_flat.push(sign);
                    signs_flat.push(sign);
                }
            }

            // 3D ND2 face DOFs (shared globally by canonical sorted face key).
            if face_dofs_per_face > 0 {
                for &(la, lb, lc) in &TET_FACES {
                    let key = FaceKey::new(verts[la], verts[lb], verts[lc]);
                    let first_dof = *face_to_dof.entry(key).or_insert_with(|| {
                        let d = next_dof;
                        next_dof += 2;
                        d
                    });
                    dofs_flat.push(first_dof);
                    dofs_flat.push(first_dof + 1);
                    // Face moments are stored in canonical (sorted-face) tangential basis.
                    signs_flat.push(1.0);
                    signs_flat.push(1.0);
                }
            }

            // Interior bubble DOFs (element-local, not shared)
            for _ in 0..interior_dofs_per_elem {
                dofs_flat.push(next_dof);
                next_dof += 1;
                signs_flat.push(1.0); // interior DOFs have no sign ambiguity
            }
        }

        HCurlSpace {
            mesh,
            order,
            n_dofs: next_dof as usize,
            dofs_flat,
            signs_flat,
            dofs_per_elem,
            edge_to_dof,
            face_to_dof,
            dim,
            cell_type,
        }
    }

    /// Orientation signs (±1.0) for the DOFs on element `elem`.
    ///
    /// `signs[i]` multiplies basis function `i` on this element so that the
    /// tangential trace is consistent with the global edge orientation.
    pub fn element_signs(&self, elem: u32) -> &[f64] {
        let start = elem as usize * self.dofs_per_elem;
        &self.signs_flat[start..start + self.dofs_per_elem]
    }

    /// Look up the global DOF index for a given edge (by canonical key).
    pub fn edge_dof(&self, edge: EdgeKey) -> Option<DofId> {
        self.edge_to_dof.get(&edge).copied()
    }

    /// Look up all global DOFs associated with a given edge.
    pub fn edge_dofs(&self, edge: EdgeKey) -> Option<Vec<DofId>> {
        self.edge_to_dof.get(&edge).map(|&first| {
            if self.order == 1 {
                vec![first]
            } else {
                vec![first, first + 1]
            }
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

    /// Vector-valued interpolation via the Nédélec DOF functional.
    ///
    /// ## ND1 (order 1)
    /// `DOF_e(F) = F(midpoint_e) · tangent_e` where `tangent_e` is
    /// the edge vector in global orientation (from smaller to larger vertex).
    /// Exact for affine vector fields.
    ///
    /// ## ND2 (order 2, 2D only)
    /// Each edge contributes two DOFs:
    /// - `DOF_0 = ∫₀¹ F(γ(t)) · τ dt`   (zero-th tangential moment)
    /// - `DOF_1 = ∫₀¹ F(γ(t)) · τ · t dt`  (first tangential moment)
    ///
    /// where `γ(t)` parametrises the edge a→b (a < b in global orientation)
    /// and `τ = b − a` is the global edge tangent vector.
    ///
    /// Interior (bubble) DOFs:
    /// - `DOF_6 = ∫_T F_x dA`  and  `DOF_7 = ∫_T F_y dA`
    ///
    /// Computed via 3-point Gauss-Legendre on each edge and a degree-4
    /// triangle quadrature for the interior.
    pub fn interpolate_vector(&self, f: &dyn Fn(&[f64]) -> Vec<f64>) -> Vector<f64> {
        let mut result = Vector::zeros(self.n_dofs);

        if self.order == 1 {
            // ND1: one zero-th tangential moment per edge, midpoint-evaluated.
            for (&EdgeKey(a, b), &dof) in &self.edge_to_dof {
                let pa = self.mesh.node_coords(a);
                let pb = self.mesh.node_coords(b);
                let mid: Vec<f64> = (0..self.dim).map(|d| 0.5 * (pa[d] + pb[d])).collect();
                let tangent: Vec<f64> = (0..self.dim).map(|d| pb[d] - pa[d]).collect();
                let fval = f(&mid);
                let dot: f64 = fval.iter().zip(&tangent).map(|(fi, ti)| fi * ti).sum();
                result.as_slice_mut()[dof as usize] = dot;
            }
        } else {
            // ND2: edge moments in all dimensions, plus 2D interior or 3D face moments.
            // 3-point Gauss-Legendre on [0,1] (exact for polynomials <= degree 5).
            let sq_3_5: f64 = (3.0_f64 / 5.0).sqrt();
            let gl_pts = [0.5 * (1.0 - sq_3_5), 0.5, 0.5 * (1.0 + sq_3_5)];
            let gl_wts = [5.0_f64 / 18.0, 4.0 / 9.0, 5.0 / 18.0];

            // Step 1 — edge DOFs.
            for (&EdgeKey(a, b), &first_dof) in &self.edge_to_dof {
                let pa = self.mesh.node_coords(a);
                let pb = self.mesh.node_coords(b);
                // Global tangent a→b (a < b by EdgeKey convention).
                let dim = self.dim;
                let tangent: Vec<f64> = (0..dim).map(|d| pb[d] - pa[d]).collect();

                let mut mom0 = 0.0_f64;
                let mut mom1 = 0.0_f64;
                for k in 0..3 {
                    let t = gl_pts[k];
                    let w = gl_wts[k];
                    let pt: Vec<f64> = (0..dim).map(|d| pa[d] + t * tangent[d]).collect();
                    let fval = f(&pt);
                    let flux: f64 = fval.iter().zip(&tangent).map(|(fi, ti)| fi * ti).sum();
                    mom0 += w * flux;
                    mom1 += w * flux * t;
                }
                let r = result.as_slice_mut();
                r[first_dof as usize]     = mom0;
                r[first_dof as usize + 1] = mom1;
            }

            if self.dim == 2 {
                if matches!(self.cell_type, ElementType::Tri3 | ElementType::Tri6) {
                    // TriND2 only: interior bubble DOFs (element-local).
                    let qr = TriND2.quadrature(4);
                    let n_elem = self.mesh.n_elements();
                    for e in 0..n_elem as u32 {
                        let dofs = self.element_dofs(e);
                        let nodes = self.mesh.element_nodes(e);
                        let transform = ElementTransformation::from_simplex_nodes(&self.mesh, nodes);
                        let det_j = transform.det_j().abs();

                        // Bubble DOFs are always the last 2 local DOFs for TriND2.
                        let bub0 = dofs[dofs.len() - 2] as usize;
                        let bub1 = dofs[dofs.len() - 1] as usize;

                        let x0 = self.mesh.node_coords(nodes[0]);
                        let x1 = self.mesh.node_coords(nodes[1]);
                        let x2 = self.mesh.node_coords(nodes[2]);
                        let j00 = x1[0] - x0[0];
                        let j10 = x1[1] - x0[1];
                        let j01 = x2[0] - x0[0];
                        let j11 = x2[1] - x0[1];

                        let mut int_x = 0.0_f64;
                        let mut int_y = 0.0_f64;
                        for (xi, &w) in qr.points.iter().zip(qr.weights.iter()) {
                            let xp = [x0[0] + j00 * xi[0] + j01 * xi[1], x0[1] + j10 * xi[0] + j11 * xi[1]];
                            let fval = f(&xp);
                            int_x += w * fval[0];
                            int_y += w * fval[1];
                        }
                        let r = result.as_slice_mut();
                        r[bub0] = int_x * det_j;
                        r[bub1] = int_y * det_j;
                    }
                }
            } else {
                // Step 2 (3D) - face moments, assembled once per unique global face.
                // DOF_f0 = int_f F.ds dA, DOF_f1 = int_f F.dt dA in canonical sorted-face basis.
                let qr_face = TriND2.quadrature(4);
                for (&FaceKey(a, b, c), &first_dof) in &self.face_to_dof {
                    let pa = self.mesh.node_coords(a);
                    let pb = self.mesh.node_coords(b);
                    let pc = self.mesh.node_coords(c);

                    let ds = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                    let dt = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
                    let cross = [
                        ds[1] * dt[2] - ds[2] * dt[1],
                        ds[2] * dt[0] - ds[0] * dt[2],
                        ds[0] * dt[1] - ds[1] * dt[0],
                    ];
                    let jac_area = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();

                    let mut m0 = 0.0_f64;
                    let mut m1 = 0.0_f64;
                    for (xi, &w) in qr_face.points.iter().zip(qr_face.weights.iter()) {
                        let s = xi[0];
                        let t = xi[1];
                        let pt = [
                            pa[0] + s * ds[0] + t * dt[0],
                            pa[1] + s * ds[1] + t * dt[1],
                            pa[2] + s * ds[2] + t * dt[2],
                        ];
                        let fv = f(&pt);
                        let d_sigma = w * jac_area;
                        m0 += d_sigma * (fv[0] * ds[0] + fv[1] * ds[1] + fv[2] * ds[2]);
                        m1 += d_sigma * (fv[0] * dt[0] + fv[1] * dt[1] + fv[2] * dt[2]);
                    }

                    let r = result.as_slice_mut();
                    r[first_dof as usize] = m0;
                    r[first_dof as usize + 1] = m1;
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
        let start = elem as usize * self.dofs_per_elem;
        &self.dofs_flat[start..start + self.dofs_per_elem]
    }

    fn interpolate(&self, _f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        // Scalar interpolation is meaningless for H(curl).
        // Use `interpolate_vector` instead.
        Vector::zeros(self.n_dofs)
    }

    fn space_type(&self) -> SpaceType { SpaceType::HCurl }

    fn order(&self) -> u8 { self.order }

    fn element_signs(&self, elem: u32) -> Option<&[f64]> {
        Some(self.element_signs(elem))
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use fem_core::{ElemId, FaceId, NodeId};
    use fem_mesh::SimplexMesh;

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
                nodes: vec![
                    [-1.0, -1.0],
                    [ 1.0, -1.0],
                    [ 1.0,  1.0],
                    [-1.0,  1.0],
                ],
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
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
                elem: [0, 1, 2, 3, 4, 5, 6, 7],
                bfaces: vec![
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [0, 1, 5, 4],
                    [1, 2, 6, 5],
                    [2, 3, 7, 6],
                    [3, 0, 4, 7],
                ],
                btags: vec![1, 2, 3, 4, 5, 6],
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
        // 4×4 unit-square mesh: 4×4 squares → 2×16=32 triangles, 5×5=25 nodes.
        // n_edges for a 4×4 grid = 3n²+2n = 3*16+8 = 56.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);
        assert_eq!(space.dofs_per_elem, 3);
        // Each triangle has 3 edges, 32 triangles, but edges are shared.
        // Expected: 56 unique edges.
        assert_eq!(space.n_dofs(), 56, "n_dofs should equal number of unique edges");
    }

    #[test]
    fn hcurl_shared_edge_dof() {
        // 1×1 mesh → 2 triangles sharing the diagonal edge.
        let mesh = SimplexMesh::<2>::unit_square_tri(1);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(1);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let space = HCurlSpace::new(mesh, 1);
        assert_eq!(space.space_type(), SpaceType::HCurl);
    }

    #[test]
    fn hcurl_dof_count_quad_nd1() {
        let mesh = OneQuadMesh::unit();
        let space = HCurlSpace::new(mesh, 1);
        assert_eq!(space.dofs_per_elem, 4);
        assert_eq!(space.n_dofs(), 4);
    }

    #[test]
    fn hcurl_dof_count_hex_nd1() {
        let mesh = OneHexMesh::unit();
        let space = HCurlSpace::new(mesh, 1);
        assert_eq!(space.dofs_per_elem, 12);
        assert_eq!(space.n_dofs(), 12);
    }

    #[test]
    fn hcurl_dof_count_quad_nd2() {
        let mesh = OneQuadMesh::unit();
        let space = HCurlSpace::new(mesh, 2);
        assert_eq!(space.dofs_per_elem, 8);
        assert_eq!(space.n_dofs(), 8);
    }

    #[test]
    fn hcurl_dof_count_hex_nd2() {
        let mesh = OneHexMesh::unit();
        let space = HCurlSpace::new(mesh, 2);
        assert_eq!(space.dofs_per_elem, 24);
        assert_eq!(space.n_dofs(), 24);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let space = HCurlSpace::new(mesh, 1);
        let v = space.interpolate_vector(&|_x| vec![1.0, 0.0]);
        // All DOF values should be finite and within the range of edge lengths.
        for &val in v.as_slice() {
            assert!(val.is_finite(), "interpolated value should be finite");
        }
    }

    #[test]
    fn hcurl_nd2_tet_local_dof_layout() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let space = HCurlSpace::new(mesh, 2);

        assert_eq!(space.element_dofs(0).len(), 20, "TetND2 should have 20 local DOFs");
        assert_eq!(space.element_signs(0).len(), 20, "TetND2 sign array length mismatch");
    }

    #[test]
    fn hcurl_nd2_tet_global_dof_count_matches_edges_faces() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);

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
}
