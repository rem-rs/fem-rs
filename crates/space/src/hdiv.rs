//! H(div) finite element space for Raviart-Thomas face elements.
//!
//! ## DOF association
//!
//! Each DOF corresponds to a unique mesh face (edge in 2-D, face in 3-D).
//! The DOF functional is the normal flux integral:
//! `DOF_f(u) = ∫_f u · n̂ ds`.
//!
//! For lowest-order Raviart-Thomas (RT0):
//! - **2-D triangles**: 3 face (= edge) DOFs per element, `n_dofs = n_unique_edges`
//! - **2-D quadrilaterals**: 4 face DOFs per element, `n_dofs = n_unique_edges`
//! - **3-D tetrahedra**: 4 face DOFs per element, `n_dofs = n_unique_faces`
//! - **3-D hexahedra**: 6 face DOFs per element, `n_dofs = n_unique_faces`
//!
//! ## Sign convention
//!
//! Each face is given a *global* orientation.  In 2-D this is the canonical
//! edge direction (from smaller to larger vertex index).  In 3-D it is defined
//! by the sorted vertex triple.  The sign on an element is +1 when the local
//! outward normal agrees with the global normal, and −1 otherwise.

use std::collections::HashMap;

use fem_core::types::DofId;
use fem_element::{quadrature::gauss_legendre_01, TriRT1, VectorReferenceElement};
use fem_linalg::Vector;
use fem_mesh::{element_type::ElementType, topology::MeshTopology, ElementTransformation};

use crate::dof_manager::{EdgeKey, FaceKey};
use crate::fe_space::{FESpace, SpaceType};

// ─── Local face tables ──────────────────────────────────────────────────────

/// Local face definitions for 2-D triangles (TriRT0 ordering).
/// Face `i` is the edge opposite vertex `i`.
const TRI_FACES: [(usize, usize); 3] = [(1, 2), (0, 2), (0, 1)];

/// Local face definitions for 2-D quads (QuadRT0 ordering, CCW).
/// Face `i` is edge `(i, (i+1)%4)` of the quad.
const QUAD_FACES: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];

/// Local face definitions for 3-D tetrahedra (TetRT0 ordering).
/// Face `i` is the triangle opposite vertex `i`.
const TET_FACES: [(usize, usize, usize); 4] = [
    (1, 2, 3), // opposite v₀
    (0, 2, 3), // opposite v₁
    (0, 1, 3), // opposite v₂
    (0, 1, 2), // opposite v₃
];

/// Local face definitions for 3-D hexahedra (HexRT0 ordering).
/// Convention: (z=-1, z=1, y=-1, y=1, x=-1, x=1).
/// Each face is a 4-sided quadrilateral (vᵢ,vⱼ,vₖ,vₗ).
const HEX_FACES: [[usize; 4]; 6] = [
    [0, 1, 2, 3], // z=-1 (bottom)
    [4, 5, 6, 7], // z= 1 (top)
    [0, 1, 5, 4], // y=-1 (near)
    [2, 3, 7, 6], // y= 1 (far)
    [0, 3, 7, 4], // x=-1 (left)
    [1, 2, 6, 5], // x= 1 (right)
];

// ─── Face DOF map ───────────────────────────────────────────────────────────

/// Unified face-to-DOF lookup: edges in 2-D, triangular/quad faces in 3-D.
enum FaceDofMap {
    Edges(HashMap<EdgeKey, DofId>),
    Faces(HashMap<FaceKey, DofId>),
    QuadEdges(HashMap<EdgeKey, DofId>),
    HexFaces(HashMap<FaceKey, DofId>),
}

// ─── HDivSpace ──────────────────────────────────────────────────────────────

/// H(div) finite element space using Raviart-Thomas face elements.
///
/// Constructed from a [`MeshTopology`] with triangular, quadrilateral,
/// tetrahedral, or hexahedral elements.
/// Supports order 0 (RT0), 1 (RT1), and on **2-D triangles only** order 2 (RT2).
/// Hex: orders 0 (RT0, 6 DOFs/elem) and 1 (RT1, 36 DOFs/elem).
pub struct HDivSpace<M: MeshTopology> {
    mesh: M,
    order: u8,
    n_dofs: usize,
    dofs_flat: Vec<DofId>,
    signs_flat: Vec<f64>,
    dofs_per_elem: usize,
    face_map: FaceDofMap,
    /// Cached element type for dispatch.
    elem_type: ElementType,
    /// If true, use BDM elements instead of RT.
    #[allow(dead_code)]
    is_bdm: bool,
}

impl<M: MeshTopology> HDivSpace<M> {
    /// Construct an H(div) space of the given order on `mesh`.
    ///
    /// # Supported combinations
    /// | Mesh type | Order | Element | DOFs/elem |
    /// |-----------|-------|---------|-----------|
    /// | Tri3/Tri6 | 0 | TriRT0 | 3 |
    /// | Tri3/Tri6 | 1 | TriRT1 | 8 |
    /// | Tri3/Tri6 | 2 | TriRT2 | 15 |
    /// | Quad4     | 0 | QuadRT0 | 4 |
    /// | Tet4/Tet10 | 0 | TetRT0 | 4 |
    /// | Tet4/Tet10 | 1 | TetRT1 | 15 |
    /// | Hex8      | 0 | HexRT0 | 6 |
    ///
    /// # Panics
    /// - If the element type is not supported.
    pub fn new(mesh: M, order: u8) -> Self {
        let dim = mesh.dim() as usize;
        let elem_type = mesh.element_type(0);
        Self::validate_order(dim, &elem_type, order);
        Self::build(mesh, order, elem_type, false)
    }

    /// Construct an H(div) space using BDM (Brezzi-Douglas-Marini) elements.
    ///
    /// BDM_k has the same edge DOFs as RT_k but fewer interior DOFs,
    /// making it more economical while preserving optimal convergence.
    /// Supported: order ≥ 1 on Tri3/Tri6, order ≥ 1 on Tet4/Tet10.
    ///
    /// | Mesh type | Order | Element | DOFs/elem |
    /// |-----------|-------|---------|-----------|
    /// | Tri3/Tri6 | 1 | TriBDM1 | 6 |
    /// | Tri3/Tri6 | 2 | TriBDM2 | 12 |
    /// | Tet4/Tet10 | 1 | TetBDM1 | 12 |
    /// | Tet4/Tet10 | 2 | TetBDM2 | 30 |
    pub fn new_bdm(mesh: M, order: u8) -> Self {
        assert!(order >= 1, "BDM requires order ≥ 1");
        let dim = mesh.dim() as usize;
        let elem_type = mesh.element_type(0);
        if dim == 2 {
            assert!(matches!(elem_type, ElementType::Tri3 | ElementType::Tri6),
                "BDM on 2D only supports Tri3/Tri6");
        } else if dim == 3 {
            assert!(matches!(elem_type, ElementType::Tet4 | ElementType::Tet10),
                "BDM on 3D only supports Tet4/Tet10");
        }
        Self::build(mesh, order, elem_type, true)
    }

    fn validate_order(dim: usize, elem_type: &ElementType, order: u8) {
        match (dim, elem_type) {
            (2, ElementType::Tri3 | ElementType::Tri6) => assert!(
                order <= 2,
                "HDivSpace: Tri RT supports orders 0, 1, 2"
            ),
            (2, ElementType::Quad4) => assert!(
                order <= 1,
                "HDivSpace: Quad RT supports orders 0 and 1"
            ),
            (3, ElementType::Tet4 | ElementType::Tet10) => assert!(
                order <= 2,
                "HDivSpace: Tet RT supports orders 0, 1, and 2"
            ),
            (3, ElementType::Hex8) => assert!(
                order <= 2,
                "HDivSpace: Hex RT supports orders 0, 1, and 2"
            ),
            _ => panic!(
                "HDivSpace: unsupported (dim={dim}, elem_type={elem_type:?})"
            ),
        }
    }

    fn build(mesh: M, order: u8, elem_type: ElementType, is_bdm: bool) -> Self {
        match (mesh.dim(), &elem_type) {
            (2, ElementType::Tri3 | ElementType::Tri6) => Self::build_2d_tri(mesh, order, is_bdm),
            (2, ElementType::Quad4) => Self::build_2d_quad(mesh, order),
            (3, ElementType::Tet4 | ElementType::Tet10) => Self::build_3d_tet(mesh, order, elem_type, is_bdm),
            (3, ElementType::Hex8) => Self::build_3d_hex(mesh, order),
            _ => panic!("HDivSpace::build: unsupported (elem_type={elem_type:?})"),
        }
    }

    // ─── 2-D triangle construction ──────────────────────────────────────────

    fn build_2d_tri(mesh: M, order: u8, is_bdm: bool) -> Self {
        // RT0: 1 per edge + 0 interior; RT1: 2 per edge + 2 interior; RT2: 3 per edge + 6 interior.
        // BDM1: 2 per edge + 0 interior; BDM2: 3 per edge + 3 interior; BDMk: k²-1 interior.
        let dofs_per_face = (order as usize) + 1;
        let interior_dofs = if is_bdm {
            let k = order as usize;
            k * k - 1 // (k+1)(k+2) - 3(k+1) = k²-1
        } else {
            match order {
                0 => 0,
                1 => 2,
                2 => 6,
                _ => order as usize * (order as usize + 1), // k(k+1) for higher RT
            }
        };
        let dofs_per_elem = TRI_FACES.len() * dofs_per_face + interior_dofs;
        let n_elem = mesh.n_elements();

        let mut edge_map: HashMap<EdgeKey, DofId> = HashMap::new();
        let mut next_dof: DofId = 0;
        let mut dofs_flat = Vec::with_capacity(n_elem * dofs_per_elem);
        let mut signs_flat = Vec::with_capacity(n_elem * dofs_per_elem);

        for e in 0..n_elem as u32 {
            let verts = mesh.element_nodes(e);
            for (face_idx, &(li, lj)) in TRI_FACES.iter().enumerate() {
                let (gi, gj) = (verts[li], verts[lj]);
                let key = EdgeKey::new(gi, gj);
                let sign = Self::compute_sign_2d_tri(&mesh, verts, face_idx, gi, gj);

                if dofs_per_face == 1 {
                    let dof = *edge_map.entry(key).or_insert_with(|| { let d=next_dof; next_dof+=1; d });
                    dofs_flat.push(dof);
                    signs_flat.push(sign);
                } else {
                    let nd = dofs_per_face as u32;
                    let first = *edge_map.entry(key).or_insert_with(|| {
                        let d = next_dof;
                        next_dof += nd;
                        d
                    });
                    for k in 0..dofs_per_face {
                        dofs_flat.push(first + k as u32);
                        signs_flat.push(sign);
                    }
                }
            }
            // Interior bubble DOFs
            for _ in 0..interior_dofs {
                dofs_flat.push(next_dof);
                next_dof += 1;
                signs_flat.push(1.0);
            }
        }

        HDivSpace {
            mesh,
            order,
            n_dofs: next_dof as usize,
            dofs_flat,
            signs_flat,
            dofs_per_elem,
            face_map: FaceDofMap::Edges(edge_map),
            elem_type: ElementType::Tri3,
            is_bdm,
        }
    }

    /// Compute the orientation sign for a 2-D face (edge) on triangles.
    ///
    /// Global edge normal is the 90° CCW rotation of (p_max − p_min).
    /// Local outward normal points away from the opposite vertex.
    /// Sign = +1 if they agree, −1 otherwise.
    fn compute_sign_2d_tri(mesh: &M, verts: &[u32], face_idx: usize, gi: u32, gj: u32) -> f64 {
        let pa = mesh.node_coords(gi);
        let pb = mesh.node_coords(gj);
        // Edge tangent gi→gj
        let tx = pb[0] - pa[0];
        let ty = pb[1] - pa[1];
        // Normal of edge gi→gj (90° CCW rotation): (−ty, tx)
        let nx = -ty;
        let ny = tx;

        // Opposite vertex
        let opp_local = face_idx; // face i is opposite vertex i
        let opp_global = verts[opp_local];
        let po = mesh.node_coords(opp_global);

        // The outward normal should point AWAY from the opposite vertex.
        // Test: (midpoint_of_edge → opposite_vertex) · normal < 0 means
        // the normal already points away from the opposite vertex.
        let mx = 0.5 * (pa[0] + pb[0]);
        let my = 0.5 * (pa[1] + pb[1]);
        let to_opp_x = po[0] - mx;
        let to_opp_y = po[1] - my;
        let dot = nx * to_opp_x + ny * to_opp_y;

        // Global orientation: the canonical edge goes min→max.
        // If gi < gj, the edge tangent is in global direction, and the normal
        // (nx, ny) is the global normal.  If gi > gj, we need to flip.
        let global_flip = if gi < gj { 1.0 } else { -1.0 };

        // dot < 0 → normal already points away from opp → outward direction agrees
        // with the tangent-based normal direction.
        let outward_flip = if dot < 0.0 { 1.0 } else { -1.0 };

        global_flip * outward_flip
    }

    // ─── 3-D tetrahedron construction ──────────────────────────────────────

    fn build_3d_tet(mesh: M, order: u8, _elem_type: ElementType, is_bdm: bool) -> Self {
        // RT0: 1 DOF per face, 0 interior → 4 DOFs/elem
        // RT1: 3 DOFs per face, 3 interior → 15 DOFs/elem
        // RT2: 6 DOFs per face, 12 interior → 36 DOFs/elem
        // BDM_k on tet: (k+1)(k+2)/2 DOFs per face, no interior if k=1.
        let k = order as usize;
        let (dofs_per_face, interior_dofs) = if is_bdm {
            let f = (k + 1) * (k + 2) / 2;
            let total = (k + 1) * (k + 2) * (k + 3) / 2;
            (f, total.saturating_sub(4 * f))
        } else {
            let f = (k + 1) * (k + 2) / 2; // (k+1)(k+2)/2 DOFs per face for RTk
            let interior = k * (k + 1) * (k + 2) / 2; // k(k+1)(k+2)/2 interior DOFs for RTk
            (f, interior)
        };
        let dofs_per_elem = TET_FACES.len() * dofs_per_face + interior_dofs;
        let n_elem = mesh.n_elements();

        let mut face_map: HashMap<FaceKey, DofId> = HashMap::new();
        let mut next_dof: DofId = 0;
        let mut dofs_flat = Vec::with_capacity(n_elem * dofs_per_elem);
        let mut signs_flat = Vec::with_capacity(n_elem * dofs_per_elem);

        for e in 0..n_elem as u32 {
            let verts = mesh.element_nodes(e);
            for (face_idx, &(la, lb, lc)) in TET_FACES.iter().enumerate() {
                let (ga, gb, gc) = (verts[la], verts[lb], verts[lc]);
                let key = FaceKey::new(ga, gb, gc);
                let sign = Self::compute_sign_3d_tet(&mesh, verts, face_idx, &key);

                if dofs_per_face == 1 {
                    let dof = *face_map.entry(key).or_insert_with(|| { let d=next_dof; next_dof+=1; d });
                    dofs_flat.push(dof);
                    signs_flat.push(sign);
                } else {
                    // Multiple DOFs per face (3 for RT1, 3+ for BDM)
                    let first = *face_map.entry(key).or_insert_with(|| { let d=next_dof; next_dof+=dofs_per_face as DofId; d });
                    for k in 0..dofs_per_face as DofId {
                        dofs_flat.push(first + k);
                        signs_flat.push(sign);
                    }
                }
            }
            for _ in 0..interior_dofs {
                dofs_flat.push(next_dof); next_dof+=1; signs_flat.push(1.0);
            }
        }

        HDivSpace {
            mesh,
            order,
            n_dofs: next_dof as usize,
            dofs_flat,
            signs_flat,
            dofs_per_elem,
            face_map: FaceDofMap::Faces(face_map),
            elem_type: ElementType::Tet4,
            is_bdm,
        }
    }

    /// Compute the orientation sign for a 3-D face (triangle).
    ///
    /// The global face normal is defined by the cross product of edges
    /// of the sorted vertex triple.  The local outward normal points
    /// away from the opposite vertex.  Sign = +1 if they agree.
    fn compute_sign_3d_tet(mesh: &M, verts: &[u32], face_idx: usize, key: &FaceKey) -> f64 {
        let p0 = mesh.node_coords(key.0);
        let p1 = mesh.node_coords(key.1);
        let p2 = mesh.node_coords(key.2);

        // Global face normal: (p1−p0) × (p2−p0)
        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let n_global = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        // The outward direction is away from the opposite vertex.
        let opp_local = face_idx;
        let opp_global = verts[opp_local];
        let po = mesh.node_coords(opp_global);

        let centroid = [
            (p0[0] + p1[0] + p2[0]) / 3.0,
            (p0[1] + p1[1] + p2[1]) / 3.0,
            (p0[2] + p1[2] + p2[2]) / 3.0,
        ];
        // outward = centroid − opposite_vertex
        let outward = [
            centroid[0] - po[0],
            centroid[1] - po[1],
            centroid[2] - po[2],
        ];

        let dot = n_global[0] * outward[0]
            + n_global[1] * outward[1]
            + n_global[2] * outward[2];

        if dot > 0.0 { 1.0 } else { -1.0 }
    }

    // ─── 2-D quadrilateral construction ───────────────────────────────────

    fn build_2d_quad(mesh: M, order: u8) -> Self {
        let dofs_per_edge = (order as usize) + 1; // 1 for RT0, 2 for RT1
        let interior_dofs = if order == 0 { 0 } else { 4 };
        let dofs_per_elem = QUAD_FACES.len() * dofs_per_edge + interior_dofs;
        let n_elem = mesh.n_elements();

        let mut edge_map: HashMap<EdgeKey, DofId> = HashMap::new();
        let mut next_dof: DofId = 0;
        let mut dofs_flat = Vec::with_capacity(n_elem * dofs_per_elem);
        let mut signs_flat = Vec::with_capacity(n_elem * dofs_per_elem);

        for e in 0..n_elem as u32 {
            let verts = mesh.element_nodes(e);
            for &(li, lj) in &QUAD_FACES {
                let (gi, gj) = (verts[li], verts[lj]);
                let key = EdgeKey::new(gi, gj);
                let sign = Self::compute_sign_2d_quad(&mesh, verts, li, gi, gj);

                if dofs_per_edge == 1 {
                    let dof = *edge_map.entry(key).or_insert_with(|| { let d = next_dof; next_dof += 1; d });
                    dofs_flat.push(dof);
                    signs_flat.push(sign);
                } else {
                    let nd = dofs_per_edge as u32;
                    let first = *edge_map.entry(key).or_insert_with(|| {
                        let d = next_dof;
                        next_dof += nd;
                        d
                    });
                    for k in 0..dofs_per_edge {
                        dofs_flat.push(first + k as u32);
                        signs_flat.push(sign);
                    }
                }
            }
            // Interior bubble DOFs (QuadRT1: ∫ Φ_x, ∫ ξ·Φ_x, ∫ Φ_y, ∫ η·Φ_y)
            for _ in 0..interior_dofs {
                dofs_flat.push(next_dof);
                next_dof += 1;
                signs_flat.push(1.0);
            }
        }

        HDivSpace {
            mesh,
            order,
            n_dofs: next_dof as usize,
            dofs_flat,
            signs_flat,
            dofs_per_elem,
            face_map: FaceDofMap::QuadEdges(edge_map),
            elem_type: ElementType::Quad4,
            is_bdm: false,
        }
    }

    /// Compute the orientation sign for a 2-D face (edge) on quads.
    /// Global edge normal points outward from the element centroid.
    /// Sign = +1 if the canonical (gi→gj) normal agrees with outward.
    fn compute_sign_2d_quad(mesh: &M, verts: &[u32], _li: usize, gi: u32, gj: u32) -> f64 {
        let pa = mesh.node_coords(gi);
        let pb = mesh.node_coords(gj);
        let tx = pb[0] - pa[0];
        let ty = pb[1] - pa[1];
        // Normal to edge gi→gj (90° CCW): (−ty, tx)
        let nx = -ty;
        let ny = tx;

        // Element centroid for outward check.
        let mut cx = 0.0; let mut cy = 0.0;
        for &v in verts {
            let p = mesh.node_coords(v);
            cx += p[0]; cy += p[1];
        }
        cx /= verts.len() as f64;
        cy /= verts.len() as f64;

        // Edge midpoint
        let mx = 0.5 * (pa[0] + pb[0]);
        let my = 0.5 * (pa[1] + pb[1]);
        // outward = midpoint → centroid points INTO the element;
        // outward = centroid → midpoint (mx-cx) points OUTWARD.
        let outward_x = mx - cx;
        let outward_y = my - cy;
        let dot = nx * outward_x + ny * outward_y;

        // `normal = [-ty, tx]` is the 90° CCW rotation of edge a→b.
        // For a CCW element this always points INWARD.
        // outward_flip = -1 when the CCW normal disagrees with outward.
        // global_flip is not needed: the CCW normal is independent of edge orientation.
        if dot > 0.0 { 1.0 } else { -1.0 }
    }

    // ─── 3-D hexahedron construction ───────────────────────────────────────

    fn build_3d_hex(mesh: M, order: u8) -> Self {
        let dofs_per_face = (order as usize + 1) * (order as usize + 1);
        let interior_dofs = if order == 0 { 0 } else { 3 * order as usize * (order as usize + 1) * (order as usize + 1) };
        let dofs_per_elem = HEX_FACES.len() * dofs_per_face + interior_dofs;
        let n_elem = mesh.n_elements();

        let mut face_map: HashMap<FaceKey, DofId> = HashMap::new();
        let mut next_dof: DofId = 0;
        let mut dofs_flat = Vec::with_capacity(n_elem * dofs_per_elem);
        let mut signs_flat = Vec::with_capacity(n_elem * dofs_per_elem);

        for e in 0..n_elem as u32 {
            let verts = mesh.element_nodes(e);
            for face_verts in &HEX_FACES {
                let (a, b, c, d) = (
                    verts[face_verts[0]],
                    verts[face_verts[1]],
                    verts[face_verts[2]],
                    verts[face_verts[3]],
                );
                let key = FaceKey::new(a, b, c);
                let sign = Self::compute_sign_3d_hex(&mesh, verts, a, b, c, d);

                if dofs_per_face == 1 {
                    let dof = *face_map.entry(key).or_insert_with(|| { let d = next_dof; next_dof += 1; d });
                    dofs_flat.push(dof);
                    signs_flat.push(sign);
                } else {
                    let nd = dofs_per_face as u32;
                    let first = *face_map.entry(key).or_insert_with(|| {
                        let d = next_dof;
                        next_dof += nd;
                        d
                    });
                    for k in 0..dofs_per_face {
                        dofs_flat.push(first + k as u32);
                        signs_flat.push(sign);
                    }
                }
            }
            // Interior bubble DOFs
            for _ in 0..interior_dofs {
                dofs_flat.push(next_dof);
                next_dof += 1;
                signs_flat.push(1.0);
            }
        }

        HDivSpace {
            mesh,
            order,
            n_dofs: next_dof as usize,
            dofs_flat,
            signs_flat,
            dofs_per_elem,
            face_map: FaceDofMap::HexFaces(face_map),
            elem_type: ElementType::Hex8,
            is_bdm: false,
        }
    }

    /// Compute the orientation sign for a 3-D hex face (quadrilateral).
    fn compute_sign_3d_hex(mesh: &M, verts: &[u32], a: u32, b: u32, c: u32, d: u32) -> f64 {
        let pa = mesh.node_coords(a);
        let pb = mesh.node_coords(b);
        let pc = mesh.node_coords(c);
        let pd = mesh.node_coords(d);

        // Face centroid
        let cx_f = (pa[0] + pb[0] + pc[0] + pd[0]) / 4.0;
        let cy_f = (pa[1] + pb[1] + pc[1] + pd[1]) / 4.0;
        let cz_f = (pa[2] + pb[2] + pc[2] + pd[2]) / 4.0;

        // Element centroid
        let mut cx_e = 0.0; let mut cy_e = 0.0; let mut cz_e = 0.0;
        for &v in verts {
            let p = mesh.node_coords(v);
            cx_e += p[0]; cy_e += p[1]; cz_e += p[2];
        }
        let nv = verts.len() as f64;
        cx_e /= nv; cy_e /= nv; cz_e /= nv;

        // Face normal via first triangle (a,b,c)
        let e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let n_global = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        // outward = face_centroid → element_centroid
        let outward = [cx_e - cx_f, cy_e - cy_f, cz_e - cz_f];
        let dot = n_global[0] * outward[0] + n_global[1] * outward[1] + n_global[2] * outward[2];
        if dot > 0.0 { 1.0 } else { -1.0 }
    }

    // ─── Public API ─────────────────────────────────────────────────────────

    /// Orientation signs (±1.0) for the DOFs on element `elem`.
    pub fn element_signs(&self, elem: u32) -> &[f64] {
        let start = elem as usize * self.dofs_per_elem;
        &self.signs_flat[start..start + self.dofs_per_elem]
    }

    /// Look up the global DOF for a 2-D face (edge).
    pub fn edge_face_dof(&self, edge: EdgeKey) -> Option<DofId> {
        match &self.face_map {
            FaceDofMap::Edges(map) | FaceDofMap::QuadEdges(map) => map.get(&edge).copied(),
            FaceDofMap::Faces(_) | FaceDofMap::HexFaces(_) => None,
        }
    }

    /// Look up the global DOF for a 3-D face (triangle/quad).
    pub fn tri_face_dof(&self, face: FaceKey) -> Option<DofId> {
        match &self.face_map {
            FaceDofMap::Faces(map) | FaceDofMap::HexFaces(map) => map.get(&face).copied(),
            FaceDofMap::Edges(_) | FaceDofMap::QuadEdges(_) => None,
        }
    }

    /// Vector-valued interpolation via the RT DOF functional.
    ///
    /// ## RT0 (order 0)
    /// `DOF_f(F) = ∫_f F · n̂_global ds`, approximated with the midpoint rule
    /// (exact for constant fields; sufficient for P0 RT0).
    ///
    /// ## RT1 (order 1, 2D only)
    /// Each edge has two DOFs:
    /// - `DOF_0 = ∫₀¹ F(γ(t)) · n_global dt`  (zero-th normal moment)
    /// - `DOF_1 = ∫₀¹ F(γ(t)) · n_global · t dt`  (first normal moment)
    ///
    /// where `γ(t)` parametrises the edge from endpoint a to b, and
    /// `n_global` is the unnormalized global edge normal (length = edge length).
    ///
    /// Interior (bubble) DOFs:
    /// - `DOF_6 = ∫_T F_x dA`  and  `DOF_7 = ∫_T F_y dA`
    ///
    /// Computed via 3-point Gauss-Legendre on each edge and a degree-3
    /// triangle quadrature rule for the interior, giving exact results for
    /// all fields representable in RT1.
    ///
    /// ## RT2 (order 2, 2D only)
    /// Three **point** normal fluxes per edge at MFEM `OpenPoints(2)` on `[0,1]`, and
    /// six interior values matching MFEM’s `RT_TriangleElement` nodal duals: at each
    /// interior reference point, `−(det J)(J^{-1}F)_y` then `−(det J)(J^{-1}F)_x` for
    /// affine triangles (contravariant Piola pullback of `F` to the reference triangle).
    pub fn interpolate_vector(&self, f: &dyn Fn(&[f64]) -> Vec<f64>) -> Vector<f64> {
        let mut result = Vector::zeros(self.n_dofs);
        match &self.face_map {
            FaceDofMap::Edges(map) | FaceDofMap::QuadEdges(map) => {
                if self.order == 0 {
                    // RT0: 1 DOF per edge — zero-th normal moment via midpoint rule.
                    for (&EdgeKey(a, b), &dof) in map {
                        let pa = self.mesh.node_coords(a);
                        let pb = self.mesh.node_coords(b);
                        let mid = [0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1])];
                        // Global edge tangent a→b (a < b), normal = 90° CCW rotation.
                        let tx = pb[0] - pa[0];
                        let ty = pb[1] - pa[1];
                        let normal = [-ty, tx]; // length = edge length
                        let fval = f(&mid);
                        result.as_slice_mut()[dof as usize] =
                            fval[0] * normal[0] + fval[1] * normal[1];
                    }
                } else if self.order == 1 {
                    // RT1: 2 DOFs per edge + interior bubble DOFs.
                    // Edge DOFs (same for Tri and Quad).
                    let sq_3_5: f64 = (3.0_f64 / 5.0).sqrt();
                    let gl_pts = [0.5 * (1.0 - sq_3_5), 0.5, 0.5 * (1.0 + sq_3_5)];
                    let gl_wts = [5.0_f64 / 18.0, 4.0 / 9.0, 5.0 / 18.0];

                    for (&EdgeKey(a, b), &first_dof) in map {
                        let pa = self.mesh.node_coords(a);
                        let pb = self.mesh.node_coords(b);
                        let tx = pb[0] - pa[0];
                        let ty = pb[1] - pa[1];
                        let normal = [-ty, tx];

                        let mut mom0 = 0.0_f64;
                        let mut mom1 = 0.0_f64;
                        for k in 0..3 {
                            let t = gl_pts[k];
                            let w = gl_wts[k];
                            let pt = [pa[0] + t * tx, pa[1] + t * ty];
                            let fval = f(&pt);
                            let flux = fval[0] * normal[0] + fval[1] * normal[1];
                            mom0 += w * flux;
                            mom1 += w * flux * t;
                        }
                        let r = result.as_slice_mut();
                        r[first_dof as usize]     = mom0;
                        r[first_dof as usize + 1] = mom1;
                    }

                    // Interior bubble DOFs depend on element type.
                    let n_elem = self.mesh.n_elements();
                    if self.elem_type == ElementType::Quad4 {
                        // QuadRT1: 4 interior DOFs per element:
                        //   DOF 8: ∫ Φ_x dA, DOF 9: ∫ ξ·Φ_x dA
                        //   DOF 10: ∫ Φ_y dA, DOF 11: ∫ η·Φ_y dA
                        // on reference [-1,1]², mapped via Piola transform.
                        use fem_element::quadrature::quad_rule;
                        let qr = quad_rule(4);
                        let jac_ref = 4.0; // area of [-1,1]²
                        for e in 0..n_elem as u32 {
                            let dofs = self.element_dofs(e);
                            let nodes = self.mesh.element_nodes(e);
                            let x0 = self.mesh.node_coords(nodes[0]);
                            let x1 = self.mesh.node_coords(nodes[1]);
                            let x2 = self.mesh.node_coords(nodes[2]);
                            let x3 = self.mesh.node_coords(nodes[3]);
                            // Bilinear map: for affine quad, J is constant.
                            let j00 = 0.5 * (x1[0] - x0[0] + x2[0] - x3[0]);
                            let j01 = 0.5 * (x3[0] - x0[0] + x2[0] - x1[0]);
                            let j10 = 0.5 * (x1[1] - x0[1] + x2[1] - x3[1]);
                            let j11 = 0.5 * (x3[1] - x0[1] + x2[1] - x1[1]);
                            let det_j = (j00 * j11 - j01 * j10).abs();
                            // Piola: u_phys(x) = (1/det_J) * J * u_ref(ξ)
                            // So interior moments: ∫ u_phys_x dx = ∫ u_ref_x dξ  (Piola preserves flux)
                            let bub_x  = dofs[8] as usize;
                            let bub_xx = dofs[9] as usize;
                            let bub_y  = dofs[10] as usize;
                            let bub_yy = dofs[11] as usize;

                            let mut int_x  = 0.0_f64;
                            let mut int_xx = 0.0_f64;
                            let mut int_y  = 0.0_f64;
                            let mut int_yy = 0.0_f64;
                            for (xi, &w) in qr.points.iter().zip(qr.weights.iter()) {
                                let xi0 = xi[0];
                                let xi1 = xi[1];
                                // Physical coords via bilinear map
                                let xp = [
                                    x0[0] + (x1[0]-x0[0])*(xi0+1.0)/2.0 + (x3[0]-x0[0])*(xi1+1.0)/2.0
                                        + (x2[0]-x1[0]-x3[0]+x0[0])*(xi0+1.0)*(xi1+1.0)/4.0,
                                    x0[1] + (x1[1]-x0[1])*(xi0+1.0)/2.0 + (x3[1]-x0[1])*(xi1+1.0)/2.0
                                        + (x2[1]-x1[1]-x3[1]+x0[1])*(xi0+1.0)*(xi1+1.0)/4.0,
                                ];
                                let fval = f(&xp);
                                // Reference moments with Piola push-forward.
                                // For affine quad, ∫_phys F · v dx = ∫_ref (J^{-1} F) · v_ref det_J dξ
                                // = ∫_ref F_phys · (1/det_J J v_ref) det_J = ∫_ref (J^{-1} F) · v_ref det_J dξ
                                // Interior DOFs in ref coords: ∫ F_x_ref, ∫ ξ F_x_ref, ∫ F_y_ref, ∫ η F_y_ref
                                let f_ref_x = fval[0] * j11 / det_j - fval[1] * j01 / det_j;
                                let f_ref_y = fval[1] * j00 / det_j - fval[0] * j10 / det_j;
                                let w_j = w * jac_ref;
                                int_x  += w_j * f_ref_x;
                                int_xx += w_j * xi0 * f_ref_x;
                                int_y  += w_j * f_ref_y;
                                int_yy += w_j * xi1 * f_ref_y;
                            }
                            let r = result.as_slice_mut();
                            r[bub_x]  = int_x  * det_j;
                            r[bub_xx] = int_xx * det_j;
                            r[bub_y]  = int_y  * det_j;
                            r[bub_yy] = int_yy * det_j;
                        }
                    } else {
                        // TriRT1: 2 interior bubble DOFs per element.
                        let qr = TriRT1.quadrature(4);
                        for e in 0..n_elem as u32 {
                            let dofs  = self.element_dofs(e);
                            let nodes = self.mesh.element_nodes(e);
                            let transform = ElementTransformation::from_simplex_nodes(&self.mesh, nodes);
                            let det_j = transform.det_j().abs();

                            let bub0 = dofs[6] as usize;
                            let bub1 = dofs[7] as usize;

                            let x0 = self.mesh.node_coords(nodes[0]);
                            let x1 = self.mesh.node_coords(nodes[1]);
                            let x2 = self.mesh.node_coords(nodes[2]);
                            let j00 = x1[0] - x0[0]; let j10 = x1[1] - x0[1];
                            let j01 = x2[0] - x0[0]; let j11 = x2[1] - x0[1];

                            let mut int_x = 0.0_f64;
                            let mut int_y = 0.0_f64;
                            for (xi, &w) in qr.points.iter().zip(qr.weights.iter()) {
                                let xp = [x0[0] + j00 * xi[0] + j01 * xi[1],
                                          x0[1] + j10 * xi[0] + j11 * xi[1]];
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
                    // RT2: MFEM-style nodal flux on edges + interior Piola samples (see `TriRT2`).
                    let (bop, _) = gauss_legendre_01(3);
                    let (iop, _) = gauss_legendre_01(2);

                    for (&EdgeKey(a, b), &first_dof) in map {
                        let pa = self.mesh.node_coords(a);
                        let pb = self.mesh.node_coords(b);
                        let tx = pb[0] - pa[0];
                        let ty = pb[1] - pa[1];
                        let normal = [-ty, tx];
                        let r = result.as_slice_mut();
                        for k in 0..3 {
                            let t = bop[k];
                            let pt = [pa[0] + t * tx, pa[1] + t * ty];
                            let fval = f(&pt);
                            r[first_dof as usize + k] =
                                fval[0] * normal[0] + fval[1] * normal[1];
                        }
                    }

                    let p = 2usize;
                    let n_elem = self.mesh.n_elements();
                    for e in 0..n_elem as u32 {
                        let dofs = self.element_dofs(e);
                        let nodes = self.mesh.element_nodes(e);
                        let transform =
                            ElementTransformation::from_simplex_nodes(&self.mesh, nodes);
                        let det_j = transform.det_j();
                        let jit = transform.jacobian_inv_t();

                        let bub_start = dofs.len() - 6;
                        let mut interior_row = 0usize;
                        for j in 0..p {
                            for i in 0..(p - j) {
                                let wsum = iop[i] + iop[j] + iop[p - 1 - i - j];
                                let xi0 = iop[i] / wsum;
                                let xi1 = iop[j] / wsum;
                                let xp = transform.map_to_physical(&[xi0, xi1]);
                                let fval = f(&xp);
                                let f0 = fval[0];
                                let f1 = fval[1];
                                // `u_ref = det(J) J^{-1} u_phys` (inverse contravariant Piola).
                                let ur0 = det_j * (jit[(0, 0)] * f0 + jit[(1, 0)] * f1);
                                let ur1 = det_j * (jit[(0, 1)] * f0 + jit[(1, 1)] * f1);
                                let r = result.as_slice_mut();
                                r[dofs[bub_start + interior_row] as usize] = -ur1;
                                interior_row += 1;
                                r[dofs[bub_start + interior_row] as usize] = -ur0;
                                interior_row += 1;
                            }
                        }
                    }
                }
            }
            FaceDofMap::Faces(map) | FaceDofMap::HexFaces(map) => {
                if self.order == 0 {
                    // 3-D RT0: one flux DOF per face (midpoint rule).
                    for (&FaceKey(a, b, c), &dof) in map {
                        let pa = self.mesh.node_coords(a);
                        let pb = self.mesh.node_coords(b);
                        let pc = self.mesh.node_coords(c);
                        let centroid = [
                            (pa[0] + pb[0] + pc[0]) / 3.0,
                            (pa[1] + pb[1] + pc[1]) / 3.0,
                            (pa[2] + pb[2] + pc[2]) / 3.0,
                        ];
                        // Global face normal = (pb−pa) × (pc−pa)  (length = 2 × area)
                        let e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                        let e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
                        let normal = [
                            e1[1] * e2[2] - e1[2] * e2[1],
                            e1[2] * e2[0] - e1[0] * e2[2],
                            e1[0] * e2[1] - e1[1] * e2[0],
                        ];
                        let fval = f(&centroid);
                        let dot = fval[0] * normal[0] + fval[1] * normal[1] + fval[2] * normal[2];
                        result.as_slice_mut()[dof as usize] = dot;
                    }
                } else {
                    // 3-D RTk (k ≥ 1): (k+1)(k+2)/2 face moments per face
                    // + k(k+1)(k+2)/2 interior moments per element.
                    let k = self.order as usize;
                    let nf = (k + 1) * (k + 2) / 2; // face DOFs per face
                    let n_int = k * (k + 1) * (k + 2) / 2; // interior DOFs per element

                    // Step 1 — face moments, assembled once per unique global face.
                    // Quadrature degree 2*(k+1) is sufficient (uf is deg k, moments up to deg k).
                    let qr_face = fem_element::quadrature::tri_rule(2 * (k + 1) as u8);
                    for (&FaceKey(a, b, c), &first_dof) in map {
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
                        let n_unit = [cross[0] / jac_area, cross[1] / jac_area, cross[2] / jac_area];

                        let mut moments = vec![0.0_f64; nf];
                        for (xi, &w) in qr_face.points.iter().zip(qr_face.weights.iter()) {
                            let s = xi[0];
                            let t = xi[1];
                            let pt = [
                                pa[0] + s * ds[0] + t * dt[0],
                                pa[1] + s * ds[1] + t * dt[1],
                                pa[2] + s * ds[2] + t * dt[2],
                            ];
                            let fv = f(&pt);
                            let nflux = fv[0] * n_unit[0] + fv[1] * n_unit[1] + fv[2] * n_unit[2];
                            let d_sigma = w * jac_area;
                            let mut idx = 0usize;
                            for p in 0..=k {
                                for q in 0..=(k - p) {
                                    moments[idx] += d_sigma * nflux * s.powi(p as i32) * t.powi(q as i32);
                                    idx += 1;
                                }
                            }
                        }

                        let r = result.as_slice_mut();
                        for m in 0..nf {
                            r[first_dof as usize + m] = moments[m];
                        }
                    }

                    // Step 2 — element-local interior moments.
                    // RTk interior: ∫ u · w dV for w ∈ [P_{k-1}]³.
                    // For affine elements this simplifies to detJ · ∫ (J⁻¹·u_phys) · w_ref dξ.
                    // We compute the monomial moments against 1, ξ, η, ζ (for k=2) per component.
                    let qr_vol = fem_element::quadrature::tet_rule(2 * (k + 1) as u8);
                    let n_elem = self.mesh.n_elements();
                    for e in 0..n_elem as u32 {
                        let dofs = self.element_dofs(e);
                        let nodes = self.mesh.element_nodes(e);
                        let transform = ElementTransformation::from_simplex_nodes(&self.mesh, nodes);
                        let det_j = transform.det_j();
                        let j_inv_t = transform.jacobian_inv_t();

                        let x0 = self.mesh.node_coords(nodes[0]);
                        let x1 = self.mesh.node_coords(nodes[1]);
                        let x2 = self.mesh.node_coords(nodes[2]);
                        let x3 = self.mesh.node_coords(nodes[3]);
                        let j0 = [x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2]];
                        let j1 = [x2[0] - x0[0], x2[1] - x0[1], x2[2] - x0[2]];
                        let j2 = [x3[0] - x0[0], x3[1] - x0[1], x3[2] - x0[2]];

                        let mut interior = vec![0.0_f64; n_int];
                        for (xi, &w) in qr_vol.points.iter().zip(qr_vol.weights.iter()) {
                            let pt = [
                                x0[0] + j0[0] * xi[0] + j1[0] * xi[1] + j2[0] * xi[2],
                                x0[1] + j0[1] * xi[0] + j1[1] * xi[1] + j2[1] * xi[2],
                                x0[2] + j0[2] * xi[0] + j1[2] * xi[1] + j2[2] * xi[2],
                            ];
                            let fv = f(&pt);
                            // Piola contravariant pullback: u_ref = detJ · J⁻¹ · u_phys
                            let u_ref_0 = det_j * (j_inv_t[(0, 0)] * fv[0]
                                                  + j_inv_t[(1, 0)] * fv[1]
                                                  + j_inv_t[(2, 0)] * fv[2]);
                            let u_ref_1 = det_j * (j_inv_t[(0, 1)] * fv[0]
                                                  + j_inv_t[(1, 1)] * fv[1]
                                                  + j_inv_t[(2, 1)] * fv[2]);
                            let u_ref_2 = det_j * (j_inv_t[(0, 2)] * fv[0]
                                                  + j_inv_t[(1, 2)] * fv[1]
                                                  + j_inv_t[(2, 2)] * fv[2]);

                            // monomials for W in [P_{k-1}]³
                            // component 0 with monomials ξ^a η^b ζ^c, a+b+c ≤ k-1
                            let mut idx = 0usize;
                            let mons = {
                                let mut m = Vec::new();
                                let km1 = k.saturating_sub(1);
                                for a in 0..=km1 {
                                    for b in 0..=(km1 - a) {
                                        for c in 0..=(km1 - a - b) {
                                            m.push(xi[0].powi(a as i32)
                                                 * xi[1].powi(b as i32)
                                                 * xi[2].powi(c as i32));
                                        }
                                    }
                                }
                                m
                            };
                            for mm in &mons {
                                interior[idx] += w * u_ref_0 * mm; idx += 1;
                            }
                            for mm in &mons {
                                interior[idx] += w * u_ref_1 * mm; idx += 1;
                            }
                            for mm in &mons {
                                interior[idx] += w * u_ref_2 * mm; idx += 1;
                            }
                        }

                        let interior_start = dofs.len() - n_int;
                        let r = result.as_slice_mut();
                        for m in 0..n_int {
                            r[dofs[interior_start + m] as usize] = interior[m];
                        }
                    }
                }
            }
        }
        result
    }
}

impl<M: MeshTopology> FESpace for HDivSpace<M> {
    type Mesh = M;

    fn mesh(&self) -> &M { &self.mesh }

    fn n_dofs(&self) -> usize { self.n_dofs }

    fn element_dofs(&self, elem: u32) -> &[DofId] {
        let start = elem as usize * self.dofs_per_elem;
        &self.dofs_flat[start..start + self.dofs_per_elem]
    }

    fn interpolate(&self, _f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        // Scalar interpolation is meaningless for H(div).
        // Use `interpolate_vector` instead.
        Vector::zeros(self.n_dofs)
    }

    fn space_type(&self) -> SpaceType { SpaceType::HDiv }

    fn order(&self) -> u8 { self.order }

    fn element_signs(&self, elem: u32) -> Option<&[f64]> {
        Some(self.element_signs(elem))
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn hdiv_dof_count_tri_2d() {
        // 4×4 unit-square mesh: 32 triangles, 56 unique edges.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = HDivSpace::new(mesh, 0);
        assert_eq!(space.dofs_per_elem, 3);
        assert_eq!(space.n_dofs(), 56, "n_dofs should equal number of unique edges in 2-D");
    }

    #[test]
    fn hdiv_shared_face_dof_2d() {
        // 1×1 mesh → 2 triangles sharing the diagonal edge.
        let mesh = SimplexMesh::<2>::unit_square_tri(1);
        let space = HDivSpace::new(mesh, 0);
        assert_eq!(space.mesh().n_elements(), 2);

        let dofs0 = space.element_dofs(0);
        let dofs1 = space.element_dofs(1);

        let shared: Vec<_> = dofs0.iter().filter(|d| dofs1.contains(d)).collect();
        assert!(!shared.is_empty(), "adjacent triangles must share at least one face DOF");
    }

    #[test]
    fn hdiv_signs_opposite_on_shared_face_2d() {
        let mesh = SimplexMesh::<2>::unit_square_tri(1);
        let space = HDivSpace::new(mesh, 0);

        let dofs0 = space.element_dofs(0);
        let signs0 = space.element_signs(0);
        let dofs1 = space.element_dofs(1);
        let signs1 = space.element_signs(1);

        for (i, &d0) in dofs0.iter().enumerate() {
            for (j, &d1) in dofs1.iter().enumerate() {
                if d0 == d1 {
                    assert!(
                        (signs0[i] + signs1[j]).abs() < 1e-14,
                        "shared face DOF {d0}: signs {}, {} should be opposite",
                        signs0[i], signs1[j]
                    );
                }
            }
        }
    }

    #[test]
    fn hdiv_space_type() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let space = HDivSpace::new(mesh, 0);
        assert_eq!(space.space_type(), SpaceType::HDiv);
    }

    #[test]
    fn hdiv_dof_count_tet_3d() {
        // Unit-cube tet mesh.
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let space = HDivSpace::new(mesh, 0);
        assert_eq!(space.dofs_per_elem, 4);
        // Each tet has 4 faces; total unique faces > n_elements (interior faces shared).
        assert!(space.n_dofs() > 0);
        // For a 2×2×2 cube mesh: 48 tets, each with 4 faces, many shared.
        // The exact count depends on the mesh generator, but verify consistency:
        // total face references = n_elem × 4, all dof indices valid.
        for e in 0..space.mesh().n_elements() as u32 {
            for &d in space.element_dofs(e) {
                assert!((d as usize) < space.n_dofs(), "DOF {d} out of range");
            }
        }
    }

    #[test]
    fn hdiv_interpolate_vector_constant_2d() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let space = HDivSpace::new(mesh, 0);
        let v = space.interpolate_vector(&|_x| vec![1.0, 0.0]);
        for &val in v.as_slice() {
            assert!(val.is_finite(), "interpolated value should be finite");
        }
    }

    #[test]
    fn hdiv_interpolate_vector_constant_3d_rt1_moments() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let space = HDivSpace::new(mesh, 1);

        // Constant field F = (1,0,0).
        let v = space.interpolate_vector(&|_x| vec![1.0, 0.0, 0.0]);
        let vals = v.as_slice();
        assert!(vals.iter().all(|x| x.is_finite()));

        // One tetrahedron: 12 face DOFs + 3 interior DOFs.
        let ldofs = space.element_dofs(0);
        assert_eq!(ldofs.len(), 15);

        // For constant face flux, moments against s and t are exactly 1/3 of the zeroth moment.
        assert!((vals[ldofs[4] as usize] - vals[ldofs[0] as usize] / 3.0).abs() < 1e-12);
    }

    #[test]
    fn hdiv_bdm1_2d_n_dofs() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = HDivSpace::new_bdm(mesh, 1);
        let ldofs = space.element_dofs(0);
        assert_eq!(ldofs.len(), 6, "TriBDM1 should have 6 DOFs per element");
    }

    #[test]
    fn hdiv_bdm2_2d_n_dofs() {
        let mesh = SimplexMesh::<2>::unit_square_tri(3);
        let space = HDivSpace::new_bdm(mesh, 2);
        let ldofs = space.element_dofs(0);
        assert_eq!(ldofs.len(), 12, "TriBDM2 should have 12 DOFs per element");
    }

    #[test]
    fn hdiv_bdm1_3d_n_dofs() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let space = HDivSpace::new_bdm(mesh, 1);
        let ldofs = space.element_dofs(0);
        assert_eq!(ldofs.len(), 12, "TetBDM1 should have 12 DOFs per element");
    }
}
