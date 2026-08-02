//! Rebuild boundary face data for a 3-D mesh after refinement.
//!
//! 3-D refinement functions (`refine_nonconforming_3d`, `refine_nonconforming_hex`,
//! `refine_prism6_uniform`) produce meshes without `face_conn` / `face_tags`.
//! This module provides `rebuild_3d_boundary` to reconstruct them from the
//! element connectivity and the original mesh.

use std::collections::HashMap;
use fem_core::{ElemId, NodeId, FaceId};
use crate::{BoundaryTag, ElementType, Mesh};

/// Helper: a parent boundary face's vertex set, tag and coordinate cover
/// (its vertices + edge midpoints + quad center) for exact child-face
/// matching.
struct ParentFace {
    verts: Vec<NodeId>,
    tag: i32,
    cover: Vec<[f64; 3]>,
}

/// Rebuild boundary faces for a refined 3-D mesh using the original mesh's
/// boundary data for tag propagation.
///
/// Scans all elements in the refined mesh, identifies faces that belong to
/// exactly one element (boundary faces), and assigns boundary tags by matching
/// against the original mesh's `face_conn`.
///
/// Uses vertex overlap matching (max overlap wins, any positive overlap accepted).
pub fn rebuild_3d_boundary(refined: &mut Mesh<3>, original: &Mesh<3>) {
    if refined.n_elems() == 0 { return; }

    // Build parent boundary face lookup: for each original boundary face,
    // store its vertex set, tag and coordinate cover (vertices + edge
    // midpoints + quad center).  A refined child face is a subdivision of
    // exactly one parent face, so matching every child vertex coordinate
    // against the parent cover is exact (the midpoint/center coordinates are
    // computed with the same 0.5*(a+b)/averaging expressions the refinement
    // used, so IEEE equality holds).
    let mut parent_faces: Vec<ParentFace> = Vec::new();
    for f in 0..original.n_faces() as FaceId {
        let bfv = original.bface_nodes(f);
        let mut cover: Vec<[f64; 3]> = bfv.iter().map(|&v| original.coords_of(v)).collect();
        let nv = bfv.len();
        for i in 0..nv {
            let a = original.coords_of(bfv[i]);
            let b = original.coords_of(bfv[(i + 1) % nv]);
            cover.push([0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])]);
        }
        if nv == 4 {
            let mut c = [0.0; 3];
            for &v in bfv.iter() {
                let p = original.coords_of(v);
                for k in 0..3 { c[k] += p[k] / 4.0; }
            }
            cover.push(c);
        }
        parent_faces.push(ParentFace {
            verts: bfv.to_vec(),
            tag: original.face_tags[f as usize] as i32,
            cover,
        });
    }

    // Count faces in the refined mesh.
    let mut face_counts: HashMap<FaceKey3, (usize, Vec<NodeId>)> = HashMap::new();

    for e in 0..refined.n_elems() as ElemId {
        let et = refined.element_type_at(e);
        let verts = refined.elem_nodes(e);
        let local_faces = local_faces_3d(et);
        for lfv in &local_faces {
            if lfv.iter().any(|&i| i >= verts.len()) { continue; }
            // Keep the ring order of the face (as seen from this element) —
            // sorting would destroy the cycle order that boundary edge
            // collection (boundary_dofs_hcurl) relies on.
            let ring: Vec<u32> = lfv.iter().map(|&i| verts[i]).collect();
            let mut sorted = ring.clone();
            sorted.sort_unstable();
            let key = FaceKey3(sorted);
            face_counts
                .entry(key)
                .and_modify(|(cnt, _)| *cnt += 1)
                .or_insert((1, ring));
        }
    }

    // Boundary faces: those counted exactly once.
    let mut new_face_conn = Vec::<NodeId>::new();
    let mut new_face_tags = Vec::<BoundaryTag>::new();
    let mut new_face_types = Vec::<ElementType>::new();
    let mut new_face_offsets = Vec::<usize>::new();
    new_face_offsets.push(0);

    for (_, &(count, ref ring)) in &face_counts {
        if count != 1 { continue; }

        let ftype = match ring.len() {
            3 => ElementType::Tri3,
            4 => ElementType::Quad4,
            _ => continue,
        };

        // Find the exact parent face this child face subdivides.
        let tag = find_exact_tag(ring, refined, &parent_faces);

        for &v in ring { new_face_conn.push(v); }
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

/// Match a refined child boundary face to the unique parent boundary face it
/// subdivides: every child vertex (original vertex, edge midpoint, quad
/// center) must lie in the parent face's coordinate cover.  Falls back to the
/// old overlap heuristic only when no unique match is found.
fn find_exact_tag(child: &[NodeId], refined: &Mesh<3>, parents: &[ParentFace]) -> i32 {
    let child_coords: Vec<[f64; 3]> = child.iter().map(|&v| refined.coords_of(v)).collect();
    let mut exact: Option<i32> = None;
    let mut n_exact = 0;
    for pf in parents {
        if child_coords.iter().all(|c| pf.cover.contains(c)) {
            n_exact += 1;
            exact = Some(pf.tag);
        }
    }
    if n_exact == 1 { return exact.unwrap(); }
    // Fallback: overlap heuristic (e.g. degenerate/curved meshes where the
    // coordinate identities are perturbed).
    find_best_tag(child, parents)
}

/// Find the parent boundary face with the most vertex overlap.
/// Accepts any overlap > 0.
fn find_best_tag(verts: &[NodeId], parents: &[ParentFace]) -> i32 {
    let set: std::collections::BTreeSet<u32> = verts.iter().copied().collect();
    // Use overlap FRACTION (overlap / parent.n_verts) as the metric.
    // This correctly prefers a small parent face (tri) over a large one (quad)
    // when both share the same number of vertices with the child.
    let mut best = (0.0f64, 1i32);
    for pf in parents {
        let pf_set: std::collections::BTreeSet<u32> = pf.verts.iter().copied().collect();
        let overlap = set.intersection(&pf_set).count();
        let pnv = pf.verts.len().max(1);
        let score = (overlap * 100) as f64 / pnv as f64; // percentage match
        if score > best.0 { best = (score, pf.tag); }
    }
    best.1
}

/// Local face vertices for 3-D element types.
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

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct FaceKey3(Vec<NodeId>);
