//! 2-D closure (conforming) refinement: given a set of marked Tri3 elements,
//! recursively refines neighbours to eliminate hanging nodes.
//!
//! The underlying bisection logic lives in the sibling `bisect` module.
//! This module only contains the closure-driver and its private helpers.

use std::collections::HashMap;
use fem_core::{NodeId, ElemId};
use crate::cad::{ProjectionConfig, project_boundary_to_cad};
use crate::element_type::ElementType;
use crate::simplex::Mesh;
use super::{edge_key, local_edges_tri};

// ─── Closure refinement ────────────────────────────────────────────────────────

/// Repeatedly refine marked Tri3 elements and their neighbours until no hanging
/// edges remain (conforming mesh closure).
///
/// Mirrors MFEM `Mesh::LocalRefinement` (used by `GeneralRefinement` in ex21):
/// 1. RED-refine every marked element (4 children);
/// 2. iterate: any element that has a hanging midpoint on one of its edges is
///    GREEN-refined (bisected into 2 children along that edge) until the mesh
///    is conforming.
pub fn closure_refine(
    mesh: &Mesh<2>,
    marked: &[ElemId],
    max_iter: usize,
    project_boundary: Option<&ProjectionConfig>,
) -> Mesh<2> {
    assert!(
        mesh.elem_type == ElementType::Tri3,
        "closure_refine: only Tri3 meshes are supported"
    );

    let mut current = mesh.clone();

    // ── 1. RED-refine all marked elements (no neighbour propagation) ─────────
    current = super::refine_marked(&current, marked);

    // ── 2. GREEN-refine hanging elements until conforming ────────────────────
    for _iter in 0..max_iter {
        let hanging = detect_hanging_edges(&current);
        if hanging.is_empty() { break; }
        current = green_refine(&current, &hanging);
    }

    if let Some(config) = project_boundary {
        current = project_boundary_to_cad(&current, config, 2);
    }
    current
}

/// Convenience overload with a default iteration limit (20).
pub fn closure_refine_default(
    mesh: &Mesh<2>,
    marked: &[ElemId],
    project_boundary: Option<&ProjectionConfig>,
) -> Mesh<2> {
    closure_refine(mesh, marked, 20, project_boundary)
}

// ─── Private helpers ───────────────────────────────────────────────────────────

/// Find elements that have a hanging midpoint on one of their edges (the edge
/// is split by a midpoint node, but the element does not contain it).  These
/// elements must be GREEN-refined.
fn detect_hanging_edges(mesh: &Mesh<2>) -> Vec<ElemId> {
    let n_elems = mesh.n_elems() as ElemId;
    let mut hanging: std::collections::HashSet<ElemId> = std::collections::HashSet::new();
    for e in 0..n_elems {
        let ns = mesh.elem_nodes(e);
        for &(ea, eb) in &local_edges_tri() {
            let (a, b) = (ns[ea], ns[eb]);
            let mid_coord = [
                0.5 * (mesh.coords_of(a)[0] + mesh.coords_of(b)[0]),
                0.5 * (mesh.coords_of(a)[1] + mesh.coords_of(b)[1]),
            ];
            // Does any node at the edge midpoint exist globally?
            let mut mid_exists = false;
            for k in 0..mesh.n_nodes() as NodeId {
                let nc = mesh.coords_of(k);
                if (nc[0] - mid_coord[0]).abs() < 1e-12 && (nc[1] - mid_coord[1]).abs() < 1e-12 {
                    mid_exists = true;
                    break;
                }
            }
            if mid_exists && !ns.contains(&(mid_node_at(mesh, &mid_coord).unwrap_or(0))) {
                // The edge is split but this element misses the midpoint → hanging.
                hanging.insert(e);
                break;
            }
        }
    }
    let mut result: Vec<ElemId> = hanging.into_iter().collect();
    result.sort();
    result
}

/// Locate the global node at a given coordinate (midpoint lookup).
fn mid_node_at(mesh: &Mesh<2>, coord: &[f64; 2]) -> Option<NodeId> {
    for k in 0..mesh.n_nodes() as NodeId {
        let nc = mesh.coords_of(k);
        if (nc[0] - coord[0]).abs() < 1e-12 && (nc[1] - coord[1]).abs() < 1e-12 {
            return Some(k);
        }
    }
    None
}

/// GREEN refinement (bisection) of the given Tri3 elements — MFEM
/// `Mesh::GreenRefinement` (== `Bisection`).  For an element whose edge
/// (v0, v1) carries a midpoint node `m`, it is split into
/// `[v2, v0, m]` and `[v1, v2, m]` where `v2` is the opposite vertex.
fn green_refine(mesh: &Mesh<2>, hanging: &[ElemId]) -> Mesh<2> {
    let mut coords = mesh.coords.clone();
    let mut next_node = mesh.n_nodes() as NodeId;
    let mut conn: Vec<NodeId> = Vec::new();
    let mut tags: Vec<i32> = Vec::new();

    for e in 0..mesh.n_elems() as ElemId {
        let ns = mesh.elem_nodes(e);
        let tag = mesh.elem_tags[e as usize];
        if hanging.contains(&e) {
            // Find the edge (local) that carries a midpoint.
            let mut done = false;
            for &(ea, eb) in &local_edges_tri() {
                let (a, b) = (ns[ea], ns[eb]);
                let mid_coord = [
                    0.5 * (mesh.coords_of(a)[0] + mesh.coords_of(b)[0]),
                    0.5 * (mesh.coords_of(a)[1] + mesh.coords_of(b)[1]),
                ];
                if let Some(m) = mid_node_at(mesh, &mid_coord) {
                    if ns.contains(&m) { continue; } // element already has it — not this edge
                    // Opposite vertex v2 = the third node.
                    let v2 = ns[0] ^ ns[1] ^ ns[2] ^ a ^ b; // xor over all four leaves a,b
                    // v2 = the node that is not a or b:
                    let v2 = ns.iter().copied().find(|&n| n != a && n != b).unwrap();
                    // MFEM Bisection: children [v2, v0, m] and [v1, v2, m]
                    conn.extend_from_slice(&[v2, a, m]);
                    conn.extend_from_slice(&[b, v2, m]);
                    tags.push(tag);
                    tags.push(tag);
                    done = true;
                    break;
                }
            }
            if !done {
                // No hanging edge found (should not happen); keep element as-is.
                conn.extend_from_slice(&ns);
                tags.push(tag);
            }
        } else {
            conn.extend_from_slice(&ns);
            tags.push(tag);
        }
    }

    // Rebuild boundary faces (bisected boundary edges get two children).
    let mut face_conn: Vec<NodeId> = Vec::new();
    let mut face_tags: Vec<i32> = Vec::new();
    for f in 0..mesh.face_conn.len() / 2 {
        let a = mesh.face_conn[f * 2];
        let b = mesh.face_conn[f * 2 + 1];
        let tag = mesh.face_tags[f];
        let mid_coord = [
            0.5 * (mesh.coords_of(a)[0] + mesh.coords_of(b)[0]),
            0.5 * (mesh.coords_of(a)[1] + mesh.coords_of(b)[1]),
        ];
        if let Some(m) = mid_node_at(mesh, &mid_coord) {
            face_conn.extend_from_slice(&[a, m]);
            face_conn.extend_from_slice(&[m, b]);
            face_tags.push(tag);
            face_tags.push(tag);
        } else {
            face_conn.extend_from_slice(&[a, b]);
            face_tags.push(tag);
        }
    }

    Mesh {
        coords,
        conn,
        elem_tags: tags,
        elem_type: ElementType::Tri3,
        face_conn,
        face_tags,
        face_type: ElementType::Line2,
        elem_types: None, elem_offsets: None,
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![], edge_to_elem: vec![], geometry: None, nc_vertex_view: None,
    }
}
