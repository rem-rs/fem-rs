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
///
/// Ordering = MFEM `Geometry::Constants<Geometry::CUBE>::FaceVert` (bottom,
/// front, right, back, left, top) so that face numbering and the RT0 basis
/// ordering match MFEM bit-for-bit:
/// - 0: {3,2,1,0}  z=−1 (bottom)
/// - 1: {0,1,5,4}  y=−1 (front)
/// - 2: {1,2,6,5}  x=+1 (right)
/// - 3: {2,3,7,6}  y=+1 (back)
/// - 4: {3,0,4,7}  x=−1 (left)
/// - 5: {4,5,6,7}  z=+1 (top)
const HEX_FACES: [[usize; 4]; 6] = [
    [3, 2, 1, 0], // z=-1 (bottom)
    [0, 1, 5, 4], // y=-1 (front)
    [1, 2, 6, 5], // x=+1 (right)
    [2, 3, 7, 6], // y=+1 (back)
    [3, 0, 4, 7], // x=-1 (left)
    [4, 5, 6, 7], // z=+1 (top)
];

/// Prism faces: 2 tri + 3 quad; ordered for RT DOF mapping.
/// Each entry: list of local vertex indices on the face.
/// Tri faces use all 3; quad faces use the first 3 for FaceKey + the 4th for volume.
const PRISM_FACES: [[usize; 4]; 5] = [
    [0, 1, 2, 2],    // bottom (tri, repeat for padding)
    [3, 4, 5, 5],    // top (tri, repeat for padding)
    [0, 1, 4, 3],    // quad 0 (front)
    [1, 2, 5, 4],    // quad 1 (right)
    [0, 2, 5, 3],    // quad 2 (left)
];

/// Pyramid faces: 4 tri + 1 quad; ordered for RT DOF mapping.
const PYRAMID_FACES: [[usize; 4]; 5] = [
    [0, 1, 4, 4],    // tri face (apex)
    [1, 2, 4, 4],    // tri face (apex)
    [2, 3, 4, 4],    // tri face (apex)
    [3, 0, 4, 4],    // tri face (apex)
    [0, 1, 2, 3],    // base quad
];

// ─── MFEM canonical face orientation tables ─────────────────────────────────
//
// These mirror `Geometry::Constants<...>::FaceVert` in MFEM's fem/geom.cpp.
// MFEM's canonical face orientation is the local-face vertex ordering of the
// first element that owns the face (Elem1, orientation 0).  The RT
// DofTransformation sign for an element-face pair is obtained by comparing
// the element's local face ordering against this canonical ordering via
// `Mesh::GetTriOrientation` / `Mesh::GetQuadOrientation` and taking the
// parity of the orientation: `RT_FECollection::DofOrderForOrientation`
// returns a sign flip for every odd orientation (for all RT orders).

/// MFEM `Constants<Geometry::TETRAHEDRON>::FaceVert` (canonical ordering).
const TET_FACES_CANON: [[usize; 3]; 4] = [
    [1, 2, 3], // opposite v₀
    [0, 3, 2], // opposite v₁
    [0, 1, 3], // opposite v₂
    [0, 2, 1], // opposite v₃
];

/// MFEM `Constants<Geometry::PRISM>::FaceVert` (tri faces padded with a dummy 4th).
const PRISM_FACES_CANON: [[usize; 4]; 5] = [
    [0, 2, 1, 0], // bottom (tri)
    [3, 4, 5, 0], // top (tri)
    [0, 1, 4, 3], // quad 0 (front)
    [1, 2, 5, 4], // quad 1 (right)
    [2, 0, 3, 5], // quad 2 (left)
];

/// MFEM `Constants<Geometry::PYRAMID>::FaceVert` (tri faces padded).
/// Rust [`PYRAMID_FACES`] lists 4 tri faces first, then the base quad; MFEM
/// lists the base quad first.  Maps Rust face index → MFEM face index.
const PYRAMID_FACES_CANON: [[usize; 4]; 5] = [
    [3, 2, 1, 0], // base quad
    [0, 1, 4, 0], // tri (apex)
    [1, 2, 4, 0], // tri (apex)
    [2, 3, 4, 0], // tri (apex)
    [3, 0, 4, 0], // tri (apex)
];
const PYRAMID_MFEM_FACE_IDX: [usize; 5] = [1, 2, 3, 4, 0];

/// Canonical (Elem1) ordering of a face, tracked per `FaceKey` while building.
#[derive(Clone, Copy)]
enum FaceCanon {
    Tri([u32; 3]),
    Quad([u32; 4]),
}

/// MFEM `Mesh::GetTriOrientation(base, test)`: index of the permutation that
/// transforms `test` into `base` (`test[tri_orientation[j][i]] == base[i]`).
/// Orientations 1, 3, 5 are odd permutations (flip).
pub fn tri_orientation(base: [u32; 3], test: [u32; 3]) -> usize {
    if test[0] == base[0] {
        if test[1] == base[1] { 0 } else { 5 }
    } else if test[0] == base[1] {
        if test[1] == base[0] { 1 } else { 2 }
    } else {
        // test[0] == base[2]
        if test[1] == base[0] { 4 } else { 3 }
    }
}

/// MFEM `Mesh::GetQuadOrientation(base, test)` → orientation in 0..=7.
/// Odd orientations are flips.
pub fn quad_orientation(base: [u32; 4], test: [u32; 4]) -> usize {
    let mut i = 0;
    while test[i] != base[0] {
        i += 1;
    }
    if test[(i + 1) % 4] == base[1] { 2 * i } else { 2 * i + 1 }
}

/// RT `DofOrderForOrientation`: odd orientation flips the sign of the face DOFs
/// (for all RT orders — `RT_FECollection::InitFaces` puts a `-1-` prefix on
/// every odd-orientation row of `TriDofOrd`/`QuadDofOrd`).
pub fn rt_face_sign(orientation: usize) -> f64 {
    if orientation % 2 == 1 { -1.0 } else { 1.0 }
}

// ─── Face DOF map ───────────────────────────────────────────────────────────

/// Unified face-to-DOF lookup: edges in 2-D, triangular/quad faces in 3-D.
#[derive(Clone)]
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
#[derive(Clone)]
// MFEM: RT_FECollection (Raviart-Thomas)
pub struct HDivSpace<M: MeshTopology> {
    mesh: M,
    order: u8,
    n_dofs: usize,
    dofs_flat: Vec<DofId>,
    signs_flat: Vec<f64>,
    dofs_per_elem: usize,
    /// Per-element offsets into dofs_flat (non-empty for mixed meshes).
    elem_offsets: Vec<usize>,
    face_map: FaceDofMap,
    /// Canonical vertex order of each global face (first-seen element's
    /// MFEM FaceVert ordering).  Used by interpolate_vector to compute the
    /// RT0 face normal consistent with MFEM DofOrderForOrientation.
    face_canon_verts: std::collections::HashMap<FaceKey, Vec<u32>>,
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
        let first_type = mesh.element_type(0);
        let is_mixed = (1..mesh.n_elements() as u32).any(|e| mesh.element_type(e) != first_type);
        if !is_mixed {
            Self::validate_order(dim, &first_type, order);
            Self::build(mesh, order, first_type, false)
        } else {
            Self::build_mixed(mesh, order)
        }
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
                order <= 6,
                "HDivSpace: Quad RT supports orders 0..=6 (QuadRTk)"
            ),
            (3, ElementType::Tet4 | ElementType::Tet10) => assert!(
                order <= 2,
                "HDivSpace: Tet RT supports orders 0, 1, and 2"
            ),
            (3, ElementType::Hex8) => assert!(
                order <= 2,
                "HDivSpace: Hex RT supports orders 0, 1, and 2"
            ),
            (3, ElementType::Prism6) => assert!(
                order <= 1,
                "HDivSpace: Prism RTk supports orders 0 and 1 (higher orders pending Phase 1B.4)"
            ),
            (3, ElementType::Pyramid5) => assert!(
                order <= 1,
                "HDivSpace: Pyramid RTk supports orders 0 and 1 (higher orders pending Phase 1B.4)"
            ),
            _ => panic!(
                "HDivSpace: unsupported (dim={dim}, elem_type={elem_type:?})"
            ),
        }
    }

    /// Build an H(div) space for a 3-D mesh with mixed element types.
    fn build_mixed(mesh: M, order: u8) -> Self {
        let dofs_per_face = (order as usize) + 1;
        let n_elem = mesh.n_elements();
        let mut face_map: HashMap<FaceKey, DofId> = HashMap::new();
        // Tracks the MFEM canonical (Elem1) vertex ordering of each face so
        // that element-face RT signs can be computed topologically.
        let mut face_canon: HashMap<FaceKey, FaceCanon> = HashMap::new();
        let mut face_canon_verts: HashMap<FaceKey, Vec<u32>> = HashMap::new();
        let mut next_dof: DofId = 0;
        let mut dofs_flat = Vec::new();
        let mut signs_flat = Vec::new();
        let mut elem_offsets = Vec::with_capacity(n_elem + 1);
        elem_offsets.push(0);

        for e in 0..n_elem as u32 {
            let et = mesh.element_type(e);
            let verts = mesh.element_nodes(e);
            let nd = dofs_per_face as u32;

            match et {
                ElementType::Tet4 | ElementType::Tet10 => {
                    let interior = if order == 0 { 0 } else if order == 1 { 2 } else { 6 };
                    for lf in 0..4 {
                        // Canonical ordering = MFEM tet FaceVert (not the
                        // simple opposite-vertex triple used for the key).
                        let [la, lb, lc] = TET_FACES_CANON[lf];
                        let local = [verts[la], verts[lb], verts[lc]];
                        let key = FaceKey::new(local[0], local[1], local[2]);
                        let sign = match face_canon.get(&key) {
                            Some(FaceCanon::Tri(base)) => rt_face_sign(tri_orientation(*base, local)),
                            _ => { face_canon.insert(key, FaceCanon::Tri(local)); face_canon_verts.entry(key).or_insert_with(|| local.to_vec()); 1.0 }
                        };
                        if nd == 1 {
                            dofs_flat.push(*face_map.entry(key).or_insert_with(|| { let d = next_dof; next_dof += 1; d }));
                            signs_flat.push(sign);
                        } else {
                            let first = *face_map.entry(key).or_insert_with(|| { let d = next_dof; next_dof += nd; d });
                            for m in 0..dofs_per_face { dofs_flat.push(first + m as u32); signs_flat.push(sign); }
                        }
                    }
                    for _ in 0..interior { dofs_flat.push(next_dof); next_dof += 1; signs_flat.push(1.0); }
                }
                ElementType::Hex8 => {
                    let interior = if order == 0 { 0 } else { 12 };
                    for (hf, &fv) in HEX_FACES.iter().enumerate() {
                        // canonical key from all 4 sorted vertices (shared
                        // quad faces must map to one DOF)
                        let mut v4 = [verts[fv[0]], verts[fv[1]], verts[fv[2]], verts[fv[3]]];
                        v4.sort_unstable();
                        let key = FaceKey::new(v4[0], v4[1], v4[2]);
                        // Canonical ordering = MFEM hex FaceVert (HEX_FACES
                        // already follows that ordering).
                        let c = HEX_FACES[hf];
                        let local = [verts[c[0]], verts[c[1]], verts[c[2]], verts[c[3]]];
                        let sign = match face_canon.get(&key) {
                            Some(FaceCanon::Quad(base)) => rt_face_sign(quad_orientation(*base, local)),
                            _ => { face_canon.insert(key, FaceCanon::Quad(local)); face_canon_verts.entry(key).or_insert_with(|| local.to_vec()); 1.0 }
                        };
                        if nd == 1 {
                            dofs_flat.push(*face_map.entry(key).or_insert_with(|| { let d = next_dof; next_dof += 1; d }));
                            signs_flat.push(sign);
                        } else {
                            let first = *face_map.entry(key).or_insert_with(|| { let d = next_dof; next_dof += nd; d });
                            for m in 0..dofs_per_face { dofs_flat.push(first + m as u32); signs_flat.push(sign); }
                        }
                    }
                    for _ in 0..interior { dofs_flat.push(next_dof); next_dof += 1; signs_flat.push(1.0); }
                }
                ElementType::Prism6 => {
                    let interior = 0;
                    // 2 tri faces + 3 quad faces; PRISM_FACES pads tri faces
                    // with a repeated 4th vertex.
                    for i in 0..5 {
                        let fv = &PRISM_FACES[i];
                        let key = if fv[2] == fv[3] {
                            FaceKey::new(verts[fv[0]], verts[fv[1]], verts[fv[2]])
                        } else {
                            let mut v4 = [verts[fv[0]], verts[fv[1]], verts[fv[2]], verts[fv[3]]];
                            v4.sort_unstable();
                            FaceKey::new(v4[0], v4[1], v4[2])
                        };
                        // Canonical ordering = MFEM prism FaceVert.
                        let c = PRISM_FACES_CANON[i];
                        let sign = if fv[2] == fv[3] {
                            let local = [verts[c[0]], verts[c[1]], verts[c[2]]];
                            match face_canon.get(&key) {
                                Some(FaceCanon::Tri(base)) => rt_face_sign(tri_orientation(*base, local)),
                                _ => { face_canon.insert(key, FaceCanon::Tri(local)); face_canon_verts.entry(key).or_insert_with(|| local.to_vec()); 1.0 }
                            }
                        } else {
                            let local = [verts[c[0]], verts[c[1]], verts[c[2]], verts[c[3]]];
                            match face_canon.get(&key) {
                                Some(FaceCanon::Quad(base)) => rt_face_sign(quad_orientation(*base, local)),
                                _ => { face_canon.insert(key, FaceCanon::Quad(local)); face_canon_verts.entry(key).or_insert_with(|| local.to_vec()); 1.0 }
                            }
                        };
                        if nd == 1 {
                            dofs_flat.push(*face_map.entry(key).or_insert_with(|| { let d = next_dof; next_dof += 1; d }));
                            signs_flat.push(sign);
                        } else {
                            let first = *face_map.entry(key).or_insert_with(|| { let d = next_dof; next_dof += nd; d });
                            for m in 0..dofs_per_face { dofs_flat.push(first + m as u32); signs_flat.push(sign); }
                        }
                    }
                    for _ in 0..interior { dofs_flat.push(next_dof); next_dof += 1; signs_flat.push(1.0); }
                }
                _ => panic!("HDivSpace::build_mixed: unsupported {et:?}"),
            }
            elem_offsets.push(dofs_flat.len());
        }

        HDivSpace {
            mesh,
            order,
            n_dofs: next_dof as usize,
            dofs_flat,
            signs_flat,
            dofs_per_elem: 0,
            elem_offsets,
            face_map: FaceDofMap::Faces(face_map),
            face_canon_verts,
            elem_type: ElementType::Tet4,
            is_bdm: false,
        }
    }

    fn build(mesh: M, order: u8, elem_type: ElementType, is_bdm: bool) -> Self {
        match (mesh.dim(), &elem_type) {
            (2, ElementType::Tri3 | ElementType::Tri6) => Self::build_2d_tri(mesh, order, is_bdm),
            (2, ElementType::Quad4) => Self::build_2d_quad(mesh, order),
            (3, ElementType::Tet4 | ElementType::Tet10) => Self::build_3d_tet(mesh, order, elem_type, is_bdm),
            (3, ElementType::Hex8) => Self::build_3d_hex(mesh, order),
            (3, ElementType::Prism6) => Self::build_3d_prism(mesh, order),
            (3, ElementType::Pyramid5) => Self::build_3d_pyramid(mesh, order),
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
            elem_offsets: vec![],
            face_canon_verts: HashMap::new(),
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
        let mut face_canon: HashMap<FaceKey, FaceCanon> = HashMap::new();
        let mut face_canon_verts: HashMap<FaceKey, Vec<u32>> = HashMap::new();
        let mut next_dof: DofId = 0;
        let mut dofs_flat = Vec::with_capacity(n_elem * dofs_per_elem);
        let mut signs_flat = Vec::with_capacity(n_elem * dofs_per_elem);

        for e in 0..n_elem as u32 {
            let verts = mesh.element_nodes(e);
            for lf in 0..4 {
                // Canonical ordering = MFEM tet FaceVert.
                let [la, lb, lc] = TET_FACES_CANON[lf];
                let local = [verts[la], verts[lb], verts[lc]];
                let key = FaceKey::new(local[0], local[1], local[2]);
                let sign = match face_canon.get(&key) {
                    Some(FaceCanon::Tri(base)) => rt_face_sign(tri_orientation(*base, local)),
                    _ => { face_canon.insert(key, FaceCanon::Tri(local)); face_canon_verts.entry(key).or_insert_with(|| local.to_vec()); 1.0 }
                };

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
            elem_offsets: vec![],
            face_map: FaceDofMap::Faces(face_map),
            face_canon_verts,
            elem_type: ElementType::Tet4,
            is_bdm,
        }
    }

    // ─── 2-D quadrilateral construction ───────────────────────────────────

    fn build_2d_quad(mesh: M, order: u8) -> Self {
        let dofs_per_edge = (order as usize) + 1; // 1 for RT0, k+1 for RTk
        // Interior DOFs of RT_QuadrilateralElement(k): 2k(k+1) (k(k+1) per
        // component).  k=0 → 0, k=1 → 4, k=2 → 12.
        let interior_dofs = if order == 0 {
            0
        } else {
            2 * order as usize * (order as usize + 1)
        };
        let dofs_per_elem = QUAD_FACES.len() * dofs_per_edge + interior_dofs;
        let n_elem = mesh.n_elements();

        // Global edge numbering follows element-traversal order — exactly
        // how MFEM builds its mesh edge table — and MFEM assigns
        // `edge_id * dofs_per_edge` consecutive DOFs per edge, then all
        // interior DOFs (per element) after every edge DOF.
        let mut edge_index: HashMap<EdgeKey, u32> = HashMap::new();
        let mut n_edges: u32 = 0;
        for e in 0..n_elem as u32 {
            let verts = mesh.element_nodes(e);
            for &(li, lj) in &QUAD_FACES {
                let key = EdgeKey::new(verts[li], verts[lj]);
                if !edge_index.contains_key(&key) {
                    edge_index.insert(key, n_edges);
                    n_edges += 1;
                }
            }
        }
        let nd = dofs_per_edge as u32;
        let mut next_dof: DofId = n_edges * nd; // interior DOF base

        let mut dofs_flat = Vec::with_capacity(n_elem * dofs_per_elem);
        let mut signs_flat = Vec::with_capacity(n_elem * dofs_per_elem);

        for e in 0..n_elem as u32 {
            let verts = mesh.element_nodes(e);
            for &(li, lj) in &QUAD_FACES {
                let (gi, gj) = (verts[li], verts[lj]);
                let sign = Self::compute_sign_2d_quad(&mesh, verts, li, gi, gj);
                let idx = edge_index[&EdgeKey::new(gi, gj)];

                if dofs_per_edge == 1 {
                    dofs_flat.push(idx * nd);
                    signs_flat.push(sign);
                } else {
                    let first = idx * nd;
                    // MFEM `RT_FECollection::DofOrderForOrientation` reverses
                    // the per-edge DOF order when the element's local edge
                    // direction opposes the global canonical (min,max)
                    // direction (cor < 0); the assembled signs (below) then
                    // carry the −1 from `EncodeDof`.  Without the reversal the
                    // global matrix columns would be permuted relative to
                    // MFEM for RT1 (2 DOFs/edge).
                    let rev = sign < 0.0;
                    for k in 0..dofs_per_edge {
                        let kk = if rev { dofs_per_edge - 1 - k } else { k };
                        dofs_flat.push(first + kk as u32);
                        signs_flat.push(sign);
                    }
                }
            }
            // Interior bubble DOFs (QuadRT1: ∫ Φ_x, ∫ ξ·Φ_x, ∫ Φ_y, ∫ η·Φ_y)
            // — MFEM numbers all edge DOFs first, then interior DOFs per
            // element (elem_id * 4 + j).
            for _ in 0..interior_dofs {
                dofs_flat.push(next_dof);
                next_dof += 1;
                signs_flat.push(1.0);
            }
        }

        let edge_map: HashMap<EdgeKey, DofId> =
            edge_index.into_iter().map(|(k, i)| (k, i * nd)).collect();

        HDivSpace {
            mesh,
            order,
            n_dofs: next_dof as usize,
            dofs_flat,
            signs_flat,
            dofs_per_elem,
            elem_offsets: vec![],
            face_canon_verts: HashMap::new(),
            face_map: FaceDofMap::QuadEdges(edge_map),
            elem_type: ElementType::Quad4,
            is_bdm: false,
        }
    }

    /// Compute the orientation sign for a 2-D face (edge) on quads.
    ///
    /// MFEM RT convention: the sign is +1 when the element's local edge
    /// direction (gi→gj, in element traversal order) agrees with the global
    /// canonical edge direction (min,max), and −1 otherwise.  This matches
    /// MFEM's `FiniteElementSpace::GetElementVDofs` for RT spaces (the RT
    /// normal-moment DOF on an edge points along the edge's canonical
    /// direction; MFEM `DofOrdering` fixes the sign by comparing the local
    /// and global edge orientations).
    fn compute_sign_2d_quad(mesh: &M, verts: &[u32], _li: usize, gi: u32, gj: u32) -> f64 {
        let _ = (mesh, verts);
        if gi < gj { 1.0 } else { -1.0 }
    }

    // ─── 3-D hexahedron construction ───────────────────────────────────────

    fn build_3d_hex(mesh: M, order: u8) -> Self {
        let dofs_per_face = (order as usize + 1) * (order as usize + 1);
        let interior_dofs = if order == 0 { 0 } else { 3 * order as usize * (order as usize + 1) * (order as usize + 1) };
        let dofs_per_elem = HEX_FACES.len() * dofs_per_face + interior_dofs;
        let n_elem = mesh.n_elements();

        let mut face_map: HashMap<FaceKey, DofId> = HashMap::new();
        let mut face_canon: HashMap<FaceKey, FaceCanon> = HashMap::new();
        let mut face_canon_verts: HashMap<FaceKey, Vec<u32>> = HashMap::new();
        let mut next_dof: DofId = 0;
        let mut dofs_flat = Vec::with_capacity(n_elem * dofs_per_elem);
        let mut signs_flat = Vec::with_capacity(n_elem * dofs_per_elem);

        for e in 0..n_elem as u32 {
            let verts = mesh.element_nodes(e);
            for (hf, face_verts) in HEX_FACES.iter().enumerate() {
                let (a, b, c, d) = (
                    verts[face_verts[0]],
                    verts[face_verts[1]],
                    verts[face_verts[2]],
                    verts[face_verts[3]],
                );
                let key = {
                    // MFEM RT0 has one DOF per unique (geometric) face; using
                    // only the first 3 local vertices makes shared quad faces
                    // look different between neighbouring elements.  Sort all
                    // 4 vertices and take the first 3 for a canonical key.
                    let mut v4 = [a, b, c, d];
                    v4.sort_unstable();
                    FaceKey::new(v4[0], v4[1], v4[2])
                };
                // Canonical ordering = MFEM hex FaceVert (HEX_FACES already
                // follows that ordering).
                let c = HEX_FACES[hf];
                let local = [verts[c[0]], verts[c[1]], verts[c[2]], verts[c[3]]];
                let sign = match face_canon.get(&key) {
                    Some(FaceCanon::Quad(base)) => rt_face_sign(quad_orientation(*base, local)),
                    _ => { face_canon.insert(key, FaceCanon::Quad(local)); face_canon_verts.entry(key).or_insert_with(|| local.to_vec()); 1.0 }
                };

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
            elem_offsets: vec![],
            face_map: FaceDofMap::HexFaces(face_map),
            face_canon_verts,
            elem_type: ElementType::Hex8,
            is_bdm: false,
        }
    }

    // ─── 3-D prism construction (RT0/RT1) ────────────────────────────────

    fn build_3d_prism(mesh: M, order: u8) -> Self {
        let dofs_per_face = (order as usize) + 1;
        let interior_dofs = 0; // RT0/RT1 for prism has no interior DOFs
        let dofs_per_elem = PRISM_FACES.len() * dofs_per_face + interior_dofs;
        let n_elem = mesh.n_elements();

        let mut face_map: HashMap<FaceKey, DofId> = HashMap::new();
        let mut face_canon: HashMap<FaceKey, FaceCanon> = HashMap::new();
        let mut face_canon_verts: HashMap<FaceKey, Vec<u32>> = HashMap::new();
        let mut next_dof: DofId = 0;
        let mut dofs_flat = Vec::with_capacity(n_elem * dofs_per_elem);
        let mut signs_flat = Vec::with_capacity(n_elem * dofs_per_elem);

        for e in 0..n_elem as u32 {
            let verts = mesh.element_nodes(e);
            for i in 0..5 {
                let face_verts = &PRISM_FACES[i];
                let a = verts[face_verts[0]];
                let b = verts[face_verts[1]];
                let c = verts[face_verts[2]];
                // PRISM_FACES pads triangles with a repeated 4th vertex;
                // quad faces get a canonical key from all 4 sorted vertices.
                let key = if face_verts[2] == face_verts[3] {
                    FaceKey::new(a, b, c)
                } else {
                    let d = verts[face_verts[3]];
                    let mut v4 = [a, b, c, d];
                    v4.sort_unstable();
                    FaceKey::new(v4[0], v4[1], v4[2])
                };
                // Canonical ordering = MFEM prism FaceVert.
                let cf = PRISM_FACES_CANON[i];
                let sign = if face_verts[2] == face_verts[3] {
                    let local = [verts[cf[0]], verts[cf[1]], verts[cf[2]]];
                    match face_canon.get(&key) {
                        Some(FaceCanon::Tri(base)) => rt_face_sign(tri_orientation(*base, local)),
                        _ => { face_canon.insert(key, FaceCanon::Tri(local)); face_canon_verts.entry(key).or_insert_with(|| local.to_vec()); 1.0 }
                    }
                } else {
                    let local = [verts[cf[0]], verts[cf[1]], verts[cf[2]], verts[cf[3]]];
                    match face_canon.get(&key) {
                        Some(FaceCanon::Quad(base)) => rt_face_sign(quad_orientation(*base, local)),
                        _ => { face_canon.insert(key, FaceCanon::Quad(local)); face_canon_verts.entry(key).or_insert_with(|| local.to_vec()); 1.0 }
                    }
                };
                if dofs_per_face == 1 {
                    let dof = *face_map.entry(key).or_insert_with(|| { let d = next_dof; next_dof += 1; d });
                    dofs_flat.push(dof);
                    signs_flat.push(sign);
                } else {
                    let nd = dofs_per_face as u32;
                    let first = *face_map.entry(key).or_insert_with(|| {
                        let d = next_dof; next_dof += nd; d
                    });
                    for k in 0..dofs_per_face {
                        dofs_flat.push(first + k as u32);
                        signs_flat.push(sign);
                    }
                }
            }
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
            elem_offsets: vec![],
            face_map: FaceDofMap::HexFaces(face_map),
            face_canon_verts,
            elem_type: ElementType::Prism6,
            is_bdm: false,
        }
    }

    // ─── 3-D pyramid construction (RT0/RT1) ──────────────────────────────

    fn build_3d_pyramid(mesh: M, order: u8) -> Self {
        let dofs_per_face = (order as usize) + 1;
        let interior_dofs = 0;
        let dofs_per_elem = PYRAMID_FACES.len() * dofs_per_face + interior_dofs;
        let n_elem = mesh.n_elements();

        let mut face_map: HashMap<FaceKey, DofId> = HashMap::new();
        let mut face_canon: HashMap<FaceKey, FaceCanon> = HashMap::new();
        let mut face_canon_verts: HashMap<FaceKey, Vec<u32>> = HashMap::new();
        let mut next_dof: DofId = 0;
        let mut dofs_flat = Vec::with_capacity(n_elem * dofs_per_elem);
        let mut signs_flat = Vec::with_capacity(n_elem * dofs_per_elem);

        for e in 0..n_elem as u32 {
            let verts = mesh.element_nodes(e);
            for i in 0..5 {
                let face_verts = &PYRAMID_FACES[i];
                let a = verts[face_verts[0]];
                let b = verts[face_verts[1]];
                let c = verts[face_verts[2]];
                let key = FaceKey::new(a, b, c);
                // Canonical ordering = MFEM pyramid FaceVert (Rust lists the 4
                // tri faces first, then the base quad).
                let cf = PYRAMID_FACES_CANON[PYRAMID_MFEM_FACE_IDX[i]];
                let sign = if face_verts[2] == face_verts[3] {
                    let local = [verts[cf[0]], verts[cf[1]], verts[cf[2]]];
                    match face_canon.get(&key) {
                        Some(FaceCanon::Tri(base)) => rt_face_sign(tri_orientation(*base, local)),
                        _ => { face_canon.insert(key, FaceCanon::Tri(local)); face_canon_verts.entry(key).or_insert_with(|| local.to_vec()); 1.0 }
                    }
                } else {
                    let local = [verts[cf[0]], verts[cf[1]], verts[cf[2]], verts[cf[3]]];
                    match face_canon.get(&key) {
                        Some(FaceCanon::Quad(base)) => rt_face_sign(quad_orientation(*base, local)),
                        _ => { face_canon.insert(key, FaceCanon::Quad(local)); face_canon_verts.entry(key).or_insert_with(|| local.to_vec()); 1.0 }
                    }
                };
                if dofs_per_face == 1 {
                    let dof = *face_map.entry(key).or_insert_with(|| { let d = next_dof; next_dof += 1; d });
                    dofs_flat.push(dof);
                    signs_flat.push(sign);
                } else {
                    let nd = dofs_per_face as u32;
                    let first = *face_map.entry(key).or_insert_with(|| {
                        let d = next_dof; next_dof += nd; d
                    });
                    for k in 0..dofs_per_face {
                        dofs_flat.push(first + k as u32);
                        signs_flat.push(sign);
                    }
                }
            }
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
            elem_offsets: vec![],
            face_map: FaceDofMap::HexFaces(face_map),
            face_canon_verts,
            elem_type: ElementType::Pyramid5,
            is_bdm: false,
        }
    }

    // ─── Public API ─────────────────────────────────────────────────────────

    /// Orientation signs (±1.0) for the DOFs on element `elem`.
    pub fn element_signs(&self, elem: u32) -> &[f64] {
        if !self.elem_offsets.is_empty() {
            let s = self.elem_offsets[elem as usize];
            &self.signs_flat[s..self.elem_offsets[elem as usize + 1]]
        } else {
            let start = elem as usize * self.dofs_per_elem;
            &self.signs_flat[start..start + self.dofs_per_elem]
        }
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

    /// Total number of global DOFs.
    pub fn n_dofs(&self) -> usize { self.n_dofs }

    /// Global DOF indices for element `elem`.
    pub fn element_dofs(&self, elem: u32) -> &[DofId] {
        if !self.elem_offsets.is_empty() {
            let s = self.elem_offsets[elem as usize];
            &self.dofs_flat[s..self.elem_offsets[elem as usize + 1]]
        } else {
            let start = elem as usize * self.dofs_per_elem;
            &self.dofs_flat[start..start + self.dofs_per_elem]
        }
    }

    /// Reference to the underlying mesh.
    pub fn mesh_topology(&self) -> &dyn MeshTopology { &self.mesh }

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
        // The DOF value is the flux integral through the face in the face's
        // canonical direction (CCW normal of the sorted edge a→b).  This is
        // independent of element orientation — element signs are applied during
        // element-level assembly and reconstruction (via element_signs), not here.
        match &self.face_map {
            FaceDofMap::Edges(map) | FaceDofMap::QuadEdges(map) => {
                if self.order == 0 {
                    // RT0: 1 DOF per edge — zero-th normal moment via midpoint rule.
                    for (&EdgeKey(a, b), &dof) in map {
                        let pa = self.mesh.node_coords(a);
                        let pb = self.mesh.node_coords(b);
                        let mid = [0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1])];
                        let tx = pb[0] - pa[0];
                        let ty = pb[1] - pa[1];
                        let normal = [ty, -tx]; // CW of sorted a→b (MFEM RT0 edge normal convention)
                        let fval = f(&mid);
                        let flux = fval[0] * normal[0] + fval[1] * normal[1];
                        result.as_slice_mut()[dof as usize] = flux;
                    }
                } else if self.order == 1 {
                    // RT1: 2 DOFs per edge + interior bubble DOFs.
                    let sq_3_5: f64 = (3.0_f64 / 5.0).sqrt();
                    let gl_pts = [0.5 * (1.0 - sq_3_5), 0.5, 0.5 * (1.0 + sq_3_5)];
                    let gl_wts = [5.0_f64 / 18.0, 4.0 / 9.0, 5.0 / 18.0];

                    for (&EdgeKey(a, b), &first_dof) in map {
                        let pa = self.mesh.node_coords(a);
                        let pb = self.mesh.node_coords(b);
                        let tx = pb[0] - pa[0];
                        let ty = pb[1] - pa[1];
                        let normal = [ty, -tx]; // CW (MFEM RT0 edge normal)

                        let mut mom0 = 0.0_f64;
                        let mut mom1 = 0.0_f64;
                        for k in 0..3 {
                            let t = gl_pts[k];
                            let w = gl_wts[k];
                            let pt = [pa[0] + t * tx, pa[1] + t * ty];
                            let fval = f(&pt);
                            let flux = fval[0] * normal[0] + fval[1] * normal[1];
                            mom0 += w * flux;
                            mom1 += w * flux * (2.0 * t - 1.0); // MFEM-compatible moment 1
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
                        let _jac_ref = 4.0; // area of [-1,1]² (kept for documentation)
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
                                let w_ref = w; // weights already sum to area of [-1,1]² (=4)
                                int_x  += w_ref * f_ref_x;
                                int_xx += w_ref * xi0 * f_ref_x;
                                int_y  += w_ref * f_ref_y;
                                int_yy += w_ref * xi1 * f_ref_y;
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
                        let normal = [ty, -tx]; // CW (MFEM RT0 edge normal)
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
                    for (&key, &dof) in map {
                        // MFEM's RT0 face DOF direction is the CANONICAL face
                        // orientation (first-seen element's FaceVert order,
                        // as used by DofOrderForOrientation), NOT the sorted
                        // FaceKey order.  The sorted-key cross product gave the
                        // wrong sign on half the faces (ex22 3D p2: 64 dofs
                        // flipped).  For a quad face the canonical ordering is
                        // [c0,c1,c2,c3] with normal = (c1−c0)×(c2−c0).
                        let canon = self.face_canon_verts.get(&key);
                        let Some(canon) = canon else { continue };
                        if canon.len() < 3 { continue; }
                        let pa = self.mesh.node_coords(canon[0]);
                        let pb = self.mesh.node_coords(canon[1]);
                        let pc = self.mesh.node_coords(canon[2]);
                        // Evaluation point: the FACE centroid.  Tri faces use
                        // the 3-vertex centroid; quad faces MUST use the
                        // 4-vertex centroid — the triangle (c0,c1,c2) centroid
                        // lies off the quad centre (pex24 prob-2 error: RT0
                        // flux interpolant divergence off ~2× per face).
                        let centroid = if canon.len() == 4 {
                            let pd = self.mesh.node_coords(canon[3]);
                            [
                                (pa[0] + pb[0] + pc[0] + pd[0]) / 4.0,
                                (pa[1] + pb[1] + pc[1] + pd[1]) / 4.0,
                                (pa[2] + pb[2] + pc[2] + pd[2]) / 4.0,
                            ]
                        } else {
                            [
                                (pa[0] + pb[0] + pc[0]) / 3.0,
                                (pa[1] + pb[1] + pc[1]) / 3.0,
                                (pa[2] + pb[2] + pc[2]) / 3.0,
                            ]
                        };
                        // Global face normal = (pb−pa) × (pc−pa); for a planar
                        // quad (c0,c1,c2,c3) the triangle (c0,c1,c2) is half
                        // the face, so the cross product length equals the
                        // face area and the dof = ∫_face f·n̂ ds (midpoint rule).
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
        if !self.elem_offsets.is_empty() {
            let s = self.elem_offsets[elem as usize];
            &self.dofs_flat[s..self.elem_offsets[elem as usize + 1]]
        } else {
            let start = elem as usize * self.dofs_per_elem;
            &self.dofs_flat[start..start + self.dofs_per_elem]
        }
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
    use fem_mesh::Mesh;

    #[test]
    fn hdiv_dof_count_tri_2d() {
        // 4×4 unit-square mesh: 32 triangles, 56 unique edges.
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = HDivSpace::new(mesh, 0);
        assert_eq!(space.dofs_per_elem, 3);
        assert_eq!(space.n_dofs(), 56, "n_dofs should equal number of unique edges in 2-D");
    }

    #[test]
    fn hdiv_shared_face_dof_2d() {
        // 1×1 mesh → 2 triangles sharing the diagonal edge.
        let mesh = Mesh::<2>::unit_square_tri(1);
        let space = HDivSpace::new(mesh, 0);
        assert_eq!(space.mesh().n_elements(), 2);

        let dofs0 = space.element_dofs(0);
        let dofs1 = space.element_dofs(1);

        let shared: Vec<_> = dofs0.iter().filter(|d| dofs1.contains(d)).collect();
        assert!(!shared.is_empty(), "adjacent triangles must share at least one face DOF");
    }

    #[test]
    fn hdiv_signs_opposite_on_shared_face_2d() {
        let mesh = Mesh::<2>::unit_square_tri(1);
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
        let mesh = Mesh::<2>::unit_square_tri(2);
        let space = HDivSpace::new(mesh, 0);
        assert_eq!(space.space_type(), SpaceType::HDiv);
    }

    #[test]
    fn hdiv_dof_count_tet_3d() {
        // Unit-cube tet mesh.
        let mesh = Mesh::<3>::unit_cube_tet(2);
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
        let mesh = Mesh::<2>::unit_square_tri(2);
        let space = HDivSpace::new(mesh, 0);
        let v = space.interpolate_vector(&|_x| vec![1.0, 0.0]);
        for &val in v.as_slice() {
            assert!(val.is_finite(), "interpolated value should be finite");
        }
    }

    #[test]
    fn hdiv_interpolate_vector_constant_3d_rt1_moments() {
        let mesh = Mesh::<3>::unit_cube_tet(1);
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
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = HDivSpace::new_bdm(mesh, 1);
        let ldofs = space.element_dofs(0);
        assert_eq!(ldofs.len(), 6, "TriBDM1 should have 6 DOFs per element");
    }

    #[test]
    fn hdiv_bdm2_2d_n_dofs() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let space = HDivSpace::new_bdm(mesh, 2);
        let ldofs = space.element_dofs(0);
        assert_eq!(ldofs.len(), 12, "TriBDM2 should have 12 DOFs per element");
    }

    #[test]
    fn hdiv_bdm1_3d_n_dofs() {
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let space = HDivSpace::new_bdm(mesh, 1);
        let ldofs = space.element_dofs(0);
        assert_eq!(ldofs.len(), 12, "TetBDM1 should have 12 DOFs per element");
    }
}
