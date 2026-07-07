//! Convert non-conforming meshes to conforming by recursive bisection.
use std::collections::{HashMap, HashSet};
use fem_core::{ElemId, NodeId};
use crate::element_type::ElementType;
use crate::simplex::Mesh;
use super::HangingNodeConstraint;
use super::refine_marked;

/// Canonical edge key: sorted node pair.
fn edge_key(a: NodeId, b: NodeId) -> (NodeId, NodeId) {
    if a < b { (a, b) } else { (b, a) }
}

fn local_edges_tri() -> [(usize, usize); 3] {
    [(0, 1), (1, 2), (0, 2)]
}

/// Convert a non-conforming Tri3 mesh to fully conforming.
pub fn make_conforming_tri(
    mesh: &Mesh<2>,
    hanging: &[HangingNodeConstraint],
) -> Mesh<2> {
    assert_eq!(mesh.elem_type, ElementType::Tri3,
        "make_conforming_tri: only Tri3 supported");

    let mut current = mesh.clone();
    let mut remaining: Vec<&HangingNodeConstraint> = hanging.iter().collect();

    for _ in 0..10 {
        if remaining.is_empty() { break; }
        let mut edge_to_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
        for e in 0..current.n_elems() as ElemId {
            let ns = current.elem_nodes(e);
            for &(a, b) in &local_edges_tri() {
                let key = edge_key(ns[a], ns[b]);
                edge_to_elems.entry(key).or_default().push(e);
            }
        }
        let mut to_refine: HashSet<ElemId> = HashSet::new();
        let mut next_remaining: Vec<&HangingNodeConstraint> = Vec::new();
        for hc in &remaining {
            let pa = hc.parent_a as NodeId;
            let pb = hc.parent_b as NodeId;
            let mid = hc.constrained as NodeId;
            let key = edge_key(pa, pb);
            if let Some(elems) = edge_to_elems.get(&key) {
                if elems.iter().any(|&e| !current.elem_nodes(e).contains(&mid)) {
                    for &e in elems {
                        if !current.elem_nodes(e).contains(&mid) { to_refine.insert(e); }
                    }
                    next_remaining.push(hc);
                }
            }
        }
        if to_refine.is_empty() { break; }
        let marked: Vec<ElemId> = to_refine.into_iter().collect();
        current = refine_marked(&current, &marked);
        remaining = next_remaining;
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amr::refine_nonconforming;
    use crate::simplex::Mesh;

    #[test]
    fn tc_single_hanging() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let (nc, constraints) = refine_nonconforming(&mesh, &[0]);
        assert!(!constraints.is_empty());
        let c = make_conforming_tri(&nc, &constraints);
        assert!(c.n_nodes() > nc.n_nodes());
    }

    #[test]
    fn tc_no_hanging() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let (nc, cns) = refine_nonconforming(&mesh, &[]);
        assert!(cns.is_empty());
        assert_eq!(make_conforming_tri(&nc, &cns).n_nodes(), nc.n_nodes());
    }

    #[test]
    fn tc_all_refined() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
        let (nc, _) = refine_nonconforming(&mesh, &all);
        assert_eq!(make_conforming_tri(&nc, &[]).n_nodes(), nc.n_nodes());
    }
}
