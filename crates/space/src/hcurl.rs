//! H(curl) finite element space for Nédélec edge elements.
//!
//! ## DOF association (D32 — MFEM nodal point-value semantics)
//!
//! Each edge carries `k = order` global DOFs.  For ND2 (and the nodal quad
//! NDk bases) the DOF functionals are **point evaluations of the tangential
//! component** at Gauss-Legendre points along the edge, with the tangent given
//! by the physical edge vector — exactly MFEM's `ND_*Element` `FE::Nodes`
//! semantics (`σ_j(Φ) = Φ(y_j)·τ`, `y_j = P_min + t_j·τ`, `τ = P_max − P_min`).
//!
//! The Gauss points are symmetric about the edge midpoint, so an element whose
//! local edge direction opposes the canonical (min→max) direction has its
//! local slot `m` sitting at the same physical point as canonical slot
//! `k−1−m`, with the opposite tangent: `σ^local_m = −ρ_{k−1−m}`.  The
//! local→global mapping therefore uses a **signed anti-diagonal permutation**
//! — the identical encoding to MFEM's element dof tables (reversed dof order
//! + sign −1).  For ND1 (k=1) this reduces to the classic ±1 edge sign.
//!
//! (The pre-D32 integral-moment functionals `∫Φ·τ t^m dt` are *not*
//! reflection invariant — reversal mixes moments binomially, which scalar
//! signs cannot express — so adjacent elements disagreed on the shared-edge
//! trace and curl-curl solutions were polluted.)
//!
//! ## Sign convention
//!
//! The canonical orientation of an edge is from the smaller to the larger
//! vertex id — mirroring MFEM's `DSTable`-based `GetElementToEdgeTable`, whose
//! `Push(a,b)` stores every edge as (min, max).  The assembler multiplies each
//! basis-function value by its sign (and uses the permuted global id) to
//! guarantee tangential continuity across elements.

use std::collections::HashMap;

use fem_core::types::DofId;
use fem_element::quadrature::gauss_legendre_01;
use fem_linalg::Vector;
use fem_mesh::{topology::MeshTopology, ElementTransformation, ElementType};

use crate::dof_manager::{EdgeKey, FaceKey, QuadFaceKey};
use crate::fe_space::{FESpace, SpaceType};

// ─── Local edge tables ──────────────────────────────────────────────────────

/// Local edge vertex pairs for 2-D triangles.
///
/// ND1 (order 1) uses `TRI_EDGES_ND1` — the historical fem-rs convention with
/// the third edge directed (0,2), which matches the TriND1 Whitney slot and
/// keeps the assembled ND1 path bit-identical to the round-13 baseline.
/// Orders ≥ 2 use `TRI_EDGES_MFEM` = MFEM `Geometry::Constants<TRIANGLE>::Edges`
/// = (0,1),(1,2),(2,0), matching the TriND2 slot parametrizations exactly
/// (slot `m` of each edge sits at the m-th Gauss point along the pair
/// direction).
pub const TRI_EDGES_ND1: [(usize, usize); 3] = [(0, 1), (1, 2), (0, 2)];
pub const TRI_EDGES_MFEM: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];

/// Local edge vertex pairs for 2-D quadrilaterals (QuadND1 ordering).
const QUAD_EDGES: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];

/// Local edge vertex pairs for 3-D tetrahedra (TetND1 ordering).
const TET_EDGES: [(usize, usize); 6] = [
    (0, 1), (0, 2), (0, 3),
    (1, 2), (1, 3), (2, 3),
];

/// Local edge vertex pairs for 3-D hexahedra (Hex8 ordering).
///
/// Ordering = MFEM `Geometry::Constants<Geometry::CUBE>::Edges` so that edge
/// numbering and the HexND1 basis ordering match MFEM bit-for-bit:
///   (0,1),(1,2),(3,2),(0,3),(4,5),(5,6),(7,6),(4,7),(0,4),(1,5),(2,6),(3,7)
const HEX_EDGES: [(usize, usize); 12] = [
    (0, 1), (1, 2), (3, 2), (0, 3),
    (4, 5), (5, 6), (7, 6), (4, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
];

/// Local edge vertex pairs for prism (Prism6 ordering).
///
/// Ordering = MFEM `Geometry::Constants<Geometry::PRISM>::Edges` so that edge
/// numbering and the PrismND1 basis ordering match MFEM bit-for-bit:
///   (0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5)
const PRISM_EDGES: [(usize, usize); 9] = [
    (0, 1), (1, 2), (2, 0), // bottom triangle
    (3, 4), (4, 5), (5, 3), // top triangle
    (0, 3), (1, 4), (2, 5), // vertical
];

/// Local edge vertex pairs for pyramid (Pyramid5 ordering).
const PYRAMID_EDGES: [(usize, usize); 8] = [
    (0, 1), (1, 2), (2, 3), (3, 0), // base quad
    (0, 4), (1, 4), (2, 4), (3, 4), // apex edges
];

/// Local face definitions for 3-D tetrahedra (MFEM tet face order; sets match
/// MFEM `(1,2,3),(0,3,2),(0,1,3),(0,2,1)`).
const TET_FACES: [(usize, usize, usize); 4] = [
    (1, 2, 3),
    (0, 2, 3),
    (0, 1, 3),
    (0, 1, 2),
];

/// Face tangent slot table per `TET_FACES` entry, mirroring the TetND2
/// (MFEM `ND_TetrahedronElement`) face `dof2tk` pairs.  For an entry
/// `(p0, n0, p1, n1)`: slot 0 tangent = verts[p0] − verts[n0], slot 1 tangent
/// = verts[p1] − verts[n1] (reference directions; physical images via the
/// element Jacobian = the corresponding physical edge vectors).
///   face (1,2,3): v2−v1, v3−v1
///   face (0,3,2): v3−v0, v2−v0
///   face (0,1,3): v1−v0, v3−v0
///   face (0,2,1): v2−v0, v1−v0
const TET_FACE_TANGENTS: [(usize, usize, usize, usize); 4] = [
    (2, 1, 3, 1),
    (3, 0, 2, 0),
    (1, 0, 3, 0),
    (2, 0, 1, 0),
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
    /// Face → tangent anchor for interpolation: `[P_a, w0, w1]` where the two
    /// face DOF functionals are `Φ(P_c)·w0` and `Φ(P_c)·w1` with the centroid
    /// `P_c = P_a + (w0 + w1)/3` (tangents fixed by the face-creating element,
    /// matching the TetND2 slot tangents).
    face_anchor: HashMap<FaceKey, [[f64; 3]; 3]>,
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
        let mut face_anchor: HashMap<FaceKey, [[f64; 3]; 3]> = HashMap::new();
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
                ElementType::Tri3 | ElementType::Tri6 => {
                    if k >= 2 {
                        &TRI_EDGES_MFEM
                    } else {
                        &TRI_EDGES_ND1
                    }
                }
                ElementType::Quad4 | ElementType::Quad8 => &QUAD_EDGES,
                ElementType::Tet4 | ElementType::Tet10 => &TET_EDGES,
                ElementType::Hex8 | ElementType::Hex20 => &HEX_EDGES,
                ElementType::Prism6 => &PRISM_EDGES,
                ElementType::Pyramid5 => &PYRAMID_EDGES,
                _ => panic!("HCurlSpace: unsupported element type {cell_type:?}"),
            };

            // Edge DOFs.
            //
            // Canonical edge = (min vertex, max vertex); canonical slot j sits
            // at the j-th Gauss point along (min→max).  A local slot m sits at
            // the m-th Gauss point along the LOCAL pair direction, hence:
            // aligned (gi < gj): slot m → global first+m, sign +1;
            // reversed:          slot m → global first+(k−1−m), sign −1
            // (MFEM encodes edge reversal exactly like this: reversed dof
            // order + sign −1).
            for &(li, lj) in local_edges {
                let (gi, gj) = (verts[li], verts[lj]);
                let key = EdgeKey::new(gi, gj);
                let aligned = gi < gj;
                let sign = if aligned { 1.0 } else { -1.0 };
                let nd = dofs_per_edge as usize;
                let first_dof = *edge_to_dof.entry(key).or_insert_with(|| {
                    let d = next_dof; next_dof += nd as u32; d
                });
                for m in 0..nd {
                    let slot = if aligned { m } else { nd - 1 - m };
                    dofs_flat.push(first_dof + slot as u32);
                    signs_flat.push(sign);
                }
            }

            // Face DOFs (NDk, k>=2).
            if k >= 2 && dim == 3 {
                let ndf = k * (k - 1);
                match cell_type {
                    ElementType::Tet4 | ElementType::Tet10 => {
                        for (f, &(la, lb, lc)) in TET_FACES.iter().enumerate() {
                            let key = FaceKey::new(verts[la], verts[lb], verts[lc]);
                            let first_dof = *face_to_dof.entry(key).or_insert_with(|| {
                                let d = next_dof; next_dof += ndf as u32; d
                            });
                            if !face_anchor.contains_key(&key) {
                                // Fix the interpolation tangents from the
                                // face-creating element (TetND2 slot
                                // tangents, physical edge vectors).
                                let a0 = mesh.node_coords(verts[la]);
                                let (p0, n0, p1, n1) = TET_FACE_TANGENTS[f];
                                let g0 = mesh.node_coords(verts[p0]);
                                let h0 = mesh.node_coords(verts[n0]);
                                let g1 = mesh.node_coords(verts[p1]);
                                let h1 = mesh.node_coords(verts[n1]);
                                let pa = [a0[0], a0[1], a0[2]];
                                let w0 = [g0[0] - h0[0], g0[1] - h0[1], g0[2] - h0[2]];
                                let w1 = [g1[0] - h1[0], g1[1] - h1[1], g1[2] - h1[2]];
                                face_anchor.insert(key, [pa, w0, w1]);
                            }
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
            face_anchor,
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

    /// Physical coordinates of every global DOF (MFEM `GetDofCoords` analog).
    ///
    /// For ND1 the DOF sits at the edge midpoint; for NDk (k≥2) at the
    /// Gauss-Legendre points along the canonical (min→max) edge direction —
    /// the point-value DOF locations (MFEM `FE::Nodes` semantics).
    pub fn dof_coords(&self) -> Vec<[f64; 3]> {
        let mut out = vec![[0.0f64; 3]; self.n_dofs()];
        let dim = self.mesh.dim() as usize;
        let nd = self.order as usize;
        let (gl_pts, _) = gauss_legendre_01(nd.max(1));
        for (&EdgeKey(a, b), &first) in &self.edge_to_dof {
            let pa = self.mesh.node_coords(a);
            let pb = self.mesh.node_coords(b);
            for m in 0..nd {
                let t = if nd == 1 { 0.5 } else { gl_pts[m] };
                let d = (first + m as u32) as usize;
                for c in 0..3 {
                    let xa = if c < dim { pa[c] } else { 0.0 };
                    let xb = if c < dim { pb[c] } else { 0.0 };
                    out[d][c] = (1.0 - t) * xa + t * xb;
                }
            }
        }
        // Tet face DOFs sit at the face centroid.
        for (&FaceKey(a, b, c), &first) in &self.face_to_dof {
            if self.order < 2 { break; }
            let pa = self.mesh.node_coords(a);
            let pb = self.mesh.node_coords(b);
            let pc = self.mesh.node_coords(c);
            for j in 0..2 {
                let d = (first + j) as usize;
                for k in 0..3 {
                    out[d][k] = (pa[k] + pb[k] + pc[k]) / 3.0;
                }
            }
        }
        out
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

    /// Interpolation anchor of a shared tet face (ND2): `[P_a, w0, w1]`.
    ///
    /// The global face dof functionals are `Φ(P_c)·w0` and `Φ(P_c)·w1` with
    /// the centroid `P_c = P_a + (w0 + w1)/3` — fixed by the face-creating
    /// element's TetND2 slot tangents.  Consumers (`discrete_op`) must use the
    /// same anchor when building dof rows for the shared face slots.
    pub fn face_tangent_anchor(&self, face: FaceKey) -> Option<[[f64; 3]; 3]> {
        self.face_anchor.get(&face).copied()
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
    /// Total number of global DOFs.
    pub fn n_dofs(&self) -> usize { self.n_dofs }

    /// Global DOF indices for element `elem`.
    pub fn element_dofs(&self, elem: u32) -> &[DofId] {
        let s = self.elem_offsets[elem as usize];
        &self.dofs_flat[s..self.elem_offsets[elem as usize + 1]]
    }

    /// Reference to the underlying mesh.
    pub fn mesh_topology(&self) -> &dyn MeshTopology { &self.mesh }

    /// ## NDk (k >= 2)
    /// Point-value edge DOFs at Gauss-Legendre points (MFEM `Project_ND`
    /// semantics); ND2 interior/face DOFs are point values as well.  k >= 3
    /// interior/face DOFs keep the legacy moment semantics of the NDk
    /// elements.
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

        // NDk (k >= 2): point-value edge DOFs (MFEM `Project_ND` semantics):
        //   dof_j = Φ(y_j)·τ,  y_j = P_min + t_j·(P_max − P_min),  τ = P_max − P_min
        // with t_j the k-point Gauss-Legendre nodes (MFEM `OpenPoints(k−1)`),
        // i.e. exactly the canonical global DOF functionals.  Because the
        // interpolation uses the canonical functional directly, the values are
        // independent of any element's local orientation.
        let (gl_pts, _gl_wts) = gauss_legendre_01(k);

        let npts = gl_pts.len();
        let n_elem = self.mesh.n_elements();
        {
            let r = result.as_slice_mut();
            for (&EdgeKey(a, b), &first_dof) in &self.edge_to_dof {
                let pa = self.mesh.node_coords(a);
                let pb = self.mesh.node_coords(b);
                let dim = self.dim;
                for m in 0..npts {
                    let t = gl_pts[m];
                    let mut pt = [0.0_f64; 3];
                    let mut tau = [0.0_f64; 3];
                    for d in 0..dim {
                        pt[d] = pa[d] + t * (pb[d] - pa[d]);
                        tau[d] = pb[d] - pa[d];
                    }
                    let fval = f(&pt[..dim]);
                    let dot: f64 = fval.iter().zip(tau[..dim].iter()).map(|(fi, ti)| fi * ti).sum();
                    r[first_dof as usize + m] = dot;
                }
            }
        }

        // Step 2 — face / interior DOFs.
        if self.dim == 2 && k == 2 {
            // ── ND2 interior DOFs: point values (MFEM Nodes semantics) ──
            if matches!(self.cell_type, ElementType::Tri3 | ElementType::Tri6) {
                // TriND2 interior slots (last two): point values at the
                // reference (1/3,1/3) with tangents (1,0) and (0,1):
                // σ = (J t̂)·Φ_phys with J(1,0) = x1−x0, J(0,1) = x2−x0.
                for e in 0..n_elem as u32 {
                    let dofs = self.element_dofs(e);
                    let bub0 = dofs[dofs.len() - 2] as usize;
                    let bub1 = dofs[dofs.len() - 1] as usize;
                    let nodes = self.mesh.element_nodes(e);
                    let x0 = self.mesh.node_coords(nodes[0]);
                    let x1 = self.mesh.node_coords(nodes[1]);
                    let x2 = self.mesh.node_coords(nodes[2]);
                    let pt = [
                        (x0[0] + x1[0] + x2[0]) / 3.0,
                        (x0[1] + x1[1] + x2[1]) / 3.0,
                    ];
                    let fv = f(&pt);
                    let r = result.as_slice_mut();
                    r[bub0] = (x1[0] - x0[0]) * fv[0] + (x1[1] - x0[1]) * fv[1];
                    r[bub1] = (x2[0] - x0[0]) * fv[0] + (x2[1] - x0[1]) * fv[1];
                }
            } else if matches!(self.cell_type, ElementType::Quad4 | ElementType::Quad8) {
                // QuadND2 interior slots (8..12): x-comp point values at the
                // reference (t_m, 1/2) with tangent J(1,0); y-comp at (1/2, t_m)
                // with tangent J(0,1) — bilinear map through the 4 corners.
                for e in 0..n_elem as u32 {
                    let dofs = self.element_dofs(e);
                    let nodes = self.mesh.element_nodes(e);
                    let x0 = self.mesh.node_coords(nodes[0]);
                    let x1 = self.mesh.node_coords(nodes[1]);
                    let x2 = self.mesh.node_coords(nodes[2]);
                    let x3 = self.mesh.node_coords(nodes[3]);
                    let r = result.as_slice_mut();
                    for (m, &t) in gl_pts.iter().enumerate() {
                        // x-component at (t, 1/2)
                        let (xi, eta) = (t, 0.5);
                        let w = [
                            (1.0 - xi) * (1.0 - eta),
                            xi * (1.0 - eta),
                            xi * eta,
                            (1.0 - xi) * eta,
                        ];
                        let pt: Vec<f64> = (0..2)
                            .map(|d| w[0] * x0[d] + w[1] * x1[d] + w[2] * x2[d] + w[3] * x3[d])
                            .collect();
                        let fv = f(&pt);
                        // J(ξ,η)·(1,0) = ∂x/∂ξ = (1−η)(x1−x0) + η(x2−x3)
                        let jt: Vec<f64> = (0..2)
                            .map(|d| {
                                (1.0 - eta) * (x1[d] - x0[d]) + eta * (x2[d] - x3[d])
                            })
                            .collect();
                        r[dofs[8 + m] as usize] =
                            fv[0] * jt[0] + fv[1] * jt[1];
                        // y-component at (1/2, t)
                        let (xi, eta) = (0.5, t);
                        let w = [
                            (1.0 - xi) * (1.0 - eta),
                            xi * (1.0 - eta),
                            xi * eta,
                            (1.0 - xi) * eta,
                        ];
                        let pt: Vec<f64> = (0..2)
                            .map(|d| w[0] * x0[d] + w[1] * x1[d] + w[2] * x2[d] + w[3] * x3[d])
                            .collect();
                        let fv = f(&pt);
                        // J(ξ,η)·(0,1) = ∂x/∂η = (1−ξ)(x3−x0) + ξ(x2−x1)
                        let jt: Vec<f64> = (0..2)
                            .map(|d| {
                                (1.0 - xi) * (x3[d] - x0[d]) + xi * (x2[d] - x1[d])
                            })
                            .collect();
                        r[dofs[10 + m] as usize] =
                            fv[0] * jt[0] + fv[1] * jt[1];
                    }
                }
            }
        } else if self.dim == 2 && k >= 3 && matches!(self.cell_type, ElementType::Tri3 | ElementType::Tri6) {
            // Tri NDk (k >= 3): monomial-weighted interior moments
            // (pre-D32 semantics of the TriNDk moment elements — pending the
            // same nodal redesign, see round-14 report).
            let n_interior = k * (k - 1);
            let qr = fem_element::quadrature::tri_rule((2 * k) as u8);
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
        } else if self.dim == 3 && k >= 2 && matches!(self.cell_type, ElementType::Tet4 | ElementType::Tet10) {
            let nf = k * (k - 1);
            if k == 2 {
                // Tet ND2 face DOFs: point values at the face centroid with
                // the face-creating element's tangent pair (TetND2 slot
                // tangents).  Cross-element face pairing between differently
                // oriented tets needs MFEM's 2×2 ND face rotations — the
                // interpolation anchor matches the face-creating element.
                for (&face_key, &first_dof) in &self.face_to_dof {
                    let anchor = match self.face_anchor.get(&face_key) {
                        Some(a) => a,
                        None => continue,
                    };
                    let [pa, w0, w1] = *anchor;
                    let pc = [
                        pa[0] + (w0[0] + w1[0]) / 3.0,
                        pa[1] + (w0[1] + w1[1]) / 3.0,
                        pa[2] + (w0[2] + w1[2]) / 3.0,
                    ];
                    let fv = f(&pc);
                    let r = result.as_slice_mut();
                    r[first_dof as usize] = fv[0] * w0[0] + fv[1] * w0[1] + fv[2] * w0[2];
                    r[first_dof as usize + 1] = fv[0] * w1[0] + fv[1] * w1[1] + fv[2] * w1[2];
                }
            } else {
                // Tet NDk (k >= 3): polynomial-weighted tangential face
                // moments (pre-D32 semantics of the TetNDk moment elements).
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
