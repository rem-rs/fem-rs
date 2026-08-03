//! Rebuild boundary face data for a 3-D mesh after refinement.
//!
//! 3-D refinement functions (`refine_nonconforming_3d`, `refine_nonconforming_hex`,
//! `refine_prism6_uniform`) produce meshes without `face_conn` / `face_tags`.
//! This module provides `rebuild_3d_boundary` to reconstruct them from the
//! original mesh's boundary faces, matching MFEM's `UniformRefinement3D_base`
//! boundary-element generation **exactly**:
//!
//! - **order**: for each original boundary face (in order), its 4 child faces
//!   are emitted in MFEM's refinement-template order;
//! - **vertex order**: each child face uses MFEM's template vertex order
//!   (e.g. tri child 0 = `(v0, mid(e0), mid(e2))`).
//!
//! This makes the refined mesh's `face_conn` identical to MFEM's boundary
//! element order, which is required for 1:1 port-submesh extraction
//! (`SubMesh::CreateFromBoundary` traversal order) and hence for
//! `SubMesh::Transfer` dof-mapping equality (MFEM ex35).

use std::collections::HashMap;
use fem_core::{ElemId, NodeId};
use crate::{BoundaryTag, ElementType, Mesh};

/// Exact-bit coordinate key for refined-vertex lookup (midpoints / face
/// centers are computed with the same expressions as the refinement, so IEEE
/// equality holds).
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct CKey([u64; 3]);

impl CKey {
    fn of(c: &[f64; 3]) -> Self {
        CKey([c[0].to_bits(), c[1].to_bits(), c[2].to_bits()])
    }
}

/// Rebuild boundary faces for a refined 3-D mesh.
///
/// Generates the child faces of every original boundary face using MFEM's
/// `UniformRefinement3D_base` templates (mesh.cpp `new_boundary`):
///
/// Triangle `(v0,v1,v2)` with edge midpoints `m0=(v0v1), m1=(v1v2), m2=(v2v0)`:
/// ```text
///   ch0: (v0, m0, m2)      ch1: (m1, m2, m0)
///   ch2: (m0, v1, m1)      ch3: (m2, m1, v2)
/// ```
/// Quadrilateral `(v0..v3)` with edge midpoints `m0..m3` and face center `qf`:
/// ```text
///   ch0: (v0, m0, qf, m3)  ch1: (m0, v1, m1, qf)
///   ch2: (qf, m1, v2, m2)  ch3: (m3, qf, m2, v3)
/// ```
pub fn rebuild_3d_boundary(refined: &mut Mesh<3>, original: &Mesh<3>) {
    if refined.n_elems() == 0 { return; }

    // New vertices of the refined mesh: edge midpoints, quad-face centers and
    // hex body centers, appended after the old vertices (MFEM offsets oedge /
    // oface / oelem).  Look them up by exact coordinates.
    let mut by_coord: HashMap<CKey, NodeId> = HashMap::new();
    for v in original.n_nodes() as NodeId..refined.n_nodes() as NodeId {
        let c = refined.coords_of(v);
        by_coord.insert(CKey::of(&c), v);
    }
    let mid = |a: NodeId, b: NodeId| -> NodeId {
        let ca = original.coords_of(a);
        let cb = original.coords_of(b);
        let key = CKey::of(&[0.5 * (ca[0] + cb[0]), 0.5 * (ca[1] + cb[1]), 0.5 * (ca[2] + cb[2])]);
        by_coord
            .get(&key)
            .copied()
            .expect("rebuild_3d_boundary: refined edge-midpoint vertex not found")
    };

    let mut new_face_conn = Vec::<NodeId>::new();
    let mut new_face_tags = Vec::<BoundaryTag>::new();
    let mut new_face_types = Vec::<ElementType>::new();
    let mut new_face_offsets = Vec::<usize>::new();
    new_face_offsets.push(0);

    // Faces are emitted in original boundary order × child-template order,
    // exactly like MFEM's `new_boundary` loop over GetBdrElement(i).
    for f in 0..original.n_faces() as ElemId {
        let bfv = original.bface_nodes(f as u32);
        let nv = bfv.len();
        if nv != 3 && nv != 4 { continue; }
        let tag = original.face_tags[f as usize];

        let mut m = Vec::with_capacity(nv);
        for k in 0..nv {
            m.push(mid(bfv[k], bfv[(k + 1) % nv]));
        }

        let mut emit = |conn: &[NodeId], ftype: ElementType| {
            new_face_conn.extend_from_slice(conn);
            new_face_offsets.push(new_face_conn.len());
            new_face_types.push(ftype);
            new_face_tags.push(tag);
        };

        if nv == 3 {
            let (v0, v1, v2) = (bfv[0], bfv[1], bfv[2]);
            let (m0, m1, m2) = (m[0], m[1], m[2]);
            emit(&[v0, m0, m2], ElementType::Tri3);
            emit(&[m1, m2, m0], ElementType::Tri3);
            emit(&[m0, v1, m1], ElementType::Tri3);
            emit(&[m2, m1, v2], ElementType::Tri3);
        } else {
            let (v0, v1, v2, v3) = (bfv[0], bfv[1], bfv[2], bfv[3]);
            let (m0, m1, m2, m3) = (m[0], m[1], m[2], m[3]);
            // Quad face center: same expression as the refinement
            // (x/4.0 accumulation order must match refine_mixed_3d:
            //  sum then divide by 4).
            let mut s = [0.0_f64; 3];
            for &v in bfv {
                let p = original.coords_of(v);
                for k in 0..3 { s[k] += p[k]; }
            }
            let qf = by_coord[&CKey::of(&[s[0] / 4.0, s[1] / 4.0, s[2] / 4.0])];
            emit(&[v0, m0, qf, m3], ElementType::Quad4);
            emit(&[m0, v1, m1, qf], ElementType::Quad4);
            emit(&[qf, m1, v2, m2], ElementType::Quad4);
            emit(&[m3, qf, m2, v3], ElementType::Quad4);
        }
    }

    refined.face_conn = new_face_conn;
    refined.face_tags = new_face_tags;
    let all_same = new_face_types.len() <= 1 || new_face_types.iter().all(|&t| t == new_face_types[0]);
    refined.face_type = if new_face_types.is_empty() { ElementType::Tri3 } else { new_face_types[0] };
    refined.face_types = if all_same { None } else { Some(new_face_types) };
    refined.face_offsets = if new_face_offsets.len() > 1 { Some(new_face_offsets) } else { None };
    refined.face_to_elem = None;
}
