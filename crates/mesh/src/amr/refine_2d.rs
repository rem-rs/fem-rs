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
    let mut to_refine: Vec<ElemId> = marked.to_vec();
    let mut visited: std::collections::HashSet<ElemId> = std::collections::HashSet::new();

    for _iter in 0..max_iter {
        if to_refine.is_empty() { break; }

        // Deduplicate and skip already-refined elements
        let mut dedup: Vec<ElemId> = Vec::new();
        for &e in &to_refine {
            if e < current.n_elems() as ElemId && visited.insert(e) {
                dedup.push(e);
            }
        }
        if dedup.is_empty() { break; }

        // Refine the marked elements (delegate to bisect module)
        current = super::refine_marked(&current, &dedup);
        visited.clear(); // After refinement, element IDs shift — reset.

        // Build edge → elements map for the new mesh
        let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
        for e in 0..current.n_elems() as ElemId {
            let ns = current.elem_nodes(e);
            for &(ea, eb) in &local_edges_tri() {
                let key = edge_key(ns[ea], ns[eb]);
                edge_elems.entry(key).or_default().push(e);
            }
        }

        // Detect hanging edges and collect elements to refine
        to_refine = detect_hanging_edges(&current, &edge_elems);
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

/// Detect edges where a hanging node exists: the edge key has some elements that
/// have the midpoint node and others that do not.  Returns the set of coarser
/// elements (those missing the midpoint) that must be refined.
fn detect_hanging_edges(
    mesh: &Mesh<2>,
    edge_elems: &HashMap<(NodeId, NodeId), Vec<ElemId>>,
) -> Vec<ElemId> {
    let mut to_refine: std::collections::HashSet<ElemId> = std::collections::HashSet::new();
    for (&key, elems) in edge_elems {
        if elems.len() < 2 { continue; }
        let (a, b) = key;
        // Compute the expected midpoint for this edge.
        let mid_coord = [
            0.5 * (mesh.coords_of(a)[0] + mesh.coords_of(b)[0]),
            0.5 * (mesh.coords_of(a)[1] + mesh.coords_of(b)[1]),
        ];
        // Find midpoint node if it exists
        let mut mid_node = None;
        for &e in elems {
            let ns = mesh.elem_nodes(e);
            for &n in ns {
                let nc = mesh.coords_of(n);
                if (nc[0] - mid_coord[0]).abs() < 1e-12 && (nc[1] - mid_coord[1]).abs() < 1e-12 {
                    mid_node = Some(n);
                    break;
                }
            }
            if mid_node.is_some() { break; }
        }
        if let Some(mid) = mid_node {
            // If any element on this edge does NOT have the midpoint, it must be refined
            for &e in elems {
                let ns = mesh.elem_nodes(e);
                if !ns.contains(&mid) {
                    to_refine.insert(e);
                }
            }
        }
    }
    let mut result: Vec<ElemId> = to_refine.into_iter().collect();
    result.sort();
    result
}
