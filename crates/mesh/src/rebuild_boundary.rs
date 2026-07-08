//! Rebuild boundary face data for a 3-D mesh after refinement.
//!
//! 3-D refinement functions (`refine_nonconforming_3d`, `refine_nonconforming_hex`,
//! `refine_prism6_uniform`) produce meshes without `face_conn` / `face_tags`.
//! This module provides `rebuild_3d_boundary` to reconstruct them from the
//! element connectivity and the original mesh.

use std::collections::HashMap;
use fem_core::{ElemId, NodeId};
use crate::{BoundaryTag, ElementType, Mesh};

/// Rebuild boundary faces for a refined 3-D mesh using the original mesh's
/// boundary data for tag propagation.
///
/// Scans all elements in the refined mesh, identifies faces that belong to
/// exactly one element (boundary faces), and assigns boundary tags by matching
/// against the original mesh's `face_conn`.
pub fn rebuild_3d_boundary(refined: &mut Mesh<3>, original: &Mesh<3>) {
    if refined.n_elems() == 0 { return; }

    // Build a hash from sorted-vertex face key → parent boundary tag.
    // Also track which faces are boundary (non-shared between elements).
    let mut face_counts: HashMap<FaceKey3, (usize, i32, Vec<NodeId>)> = HashMap::new();

    for e in 0..refined.n_elems() as ElemId {
        let et = refined.element_type_at(e);
        let verts = refined.elem_nodes(e);
        let local_faces = local_faces_3d(et);
        for lfv in &local_faces {
            let mut sorted: Vec<u32> = lfv.iter().map(|&i| verts[i]).collect();
            sorted.sort_unstable();
            let key = FaceKey3(sorted.clone());
            face_counts
                .entry(key)
                .and_modify(|(cnt, _, _)| *cnt += 1)
                .or_insert((1, -1, sorted));
        }
    }

    // Boundary faces: those counted exactly once.
    let mut new_face_conn = Vec::<NodeId>::new();
    let mut new_face_tags = Vec::<BoundaryTag>::new();
    let mut new_face_types = Vec::<ElementType>::new();
    let mut new_face_offsets = Vec::<usize>::new();
    new_face_offsets.push(0);

    for (_, &(count, _, ref verts)) in &face_counts {
        if count != 1 { continue; } // internal or degenerate

        // Determine face type from vertex count.
        let ftype = match verts.len() {
            3 => ElementType::Tri3,
            4 => ElementType::Quad4,
            _ => continue,
        };

        // Find the parent boundary tag by matching against the original mesh.
        let tag = find_parent_tag(original, verts);

        for &v in verts {
            new_face_conn.push(v);
        }
        new_face_tags.push(tag as BoundaryTag);
        new_face_types.push(ftype);
        new_face_offsets.push(new_face_conn.len());
    }

    refined.face_conn = new_face_conn;
    refined.face_tags = new_face_tags;
    refined.face_type = if new_face_types.is_empty() { ElementType::Tri3 } else { new_face_types[0] };
    let all_same = new_face_types.len() <= 1 || new_face_types.iter().all(|&t| t == new_face_types[0]);
    refined.face_types = if all_same { None } else { Some(new_face_types) };
    refined.face_offsets = if new_face_offsets.len() > 1 { Some(new_face_offsets) } else { None };
    refined.face_to_elem = None;
}

/// Local face vertices for 3-D element types (vertex indices into element nodes).
fn local_faces_3d(elem_type: ElementType) -> Vec<Vec<usize>> {
    match elem_type {
        ElementType::Tet4 | ElementType::Tet10 => vec![
            vec![1, 2, 3], vec![0, 2, 3], vec![0, 1, 3], vec![0, 1, 2],
        ],
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => vec![
            vec![0, 1, 2, 3], vec![4, 5, 6, 7],
            vec![0, 1, 5, 4], vec![2, 3, 7, 6],
            vec![0, 3, 7, 4], vec![1, 2, 6, 5],
        ],
        ElementType::Prism6 | ElementType::Prism15 => vec![
            vec![0, 1, 2], vec![3, 4, 5],
            vec![0, 1, 4, 3], vec![1, 2, 5, 4], vec![0, 2, 5, 3],
        ],
        _ => vec![],
    }
}

/// Sorted face vertex key for deduplication.
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct FaceKey3(Vec<NodeId>);

/// Find the boundary tag from the original mesh for a face with vertices `verts`.
///
/// Uses set overlap rather than subset: a child face is considered part of a
/// parent face if they share at least 2 vertices (for tri faces) or 3 vertices
/// (for quad faces). This tolerates edge-midpoint vertices introduced by
/// refinement.
fn find_parent_tag(original: &Mesh<3>, verts: &[NodeId]) -> i32 {
    let face_set: std::collections::BTreeSet<u32> = verts.iter().copied().collect();
    let min_overlap = if verts.len() >= 4 { 3 } else { 2 };
    for f in 0..original.n_faces() {
        let bfv = original.bface_nodes(f as u32);
        let bf_set: std::collections::BTreeSet<u32> = bfv.iter().copied().collect();
        let overlap = face_set.intersection(&bf_set).count();
        if overlap >= min_overlap {
            return original.face_tags[f] as i32;
        }
    }
    1
}
