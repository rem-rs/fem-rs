//! MFEM-style `GeneralRefinement` for non-conforming quad meshes (2D).
//!
//! Provides the [`Refinement`] struct and [`general_refinement_2d`] mirroring
//! MFEM's `Mesh::GeneralRefinement(const Array<Refinement> &)` for Quad4
//! meshes with arbitrary split scales (e.g. the 2/3 + 0.5 sequence used by
//! `ref321`'s 3:1 refinement).
//!
//! Unlike [`super::amr_inner::refine_nonconforming_quad_aniso`] (midpoint
//! splits only), each refinement splits one element along one axis at an
//! arbitrary `scale` position.
//!
//! Note: node ids of unrefined elements are preserved, so callers may
//! accumulate state across successive calls (as MFEM's NCMesh does).

use std::collections::HashMap;

use fem_core::{ElemId, NodeId};
use crate::element_type::ElementType;
use crate::simplex::Mesh;
use super::amr_inner::{HangingNodeConstraint, quad_edge_key, local_edges_quad};

/// MFEM-compatible refinement descriptor (`ncmesh.hpp`).
///
/// - `index`: element to refine
/// - `type_`: bitmask — X=1, Y=2 (matching MFEM `Refinement::X/Y/Z`)
/// - `scale`: split position along the refined direction (0.0..1.0, default 0.5)
#[derive(Debug, Clone, Copy)]
pub struct Refinement {
    /// Element index to refine.
    pub index: ElemId,
    /// Direction bitmask: X=1, Y=2.
    pub type_: u8,
    /// Split scale along the direction (0.5 = midpoint).
    pub scale: f64,
}

impl Refinement {
    /// Refinement with explicit scale.
    pub fn new(index: ElemId, type_: u8, scale: f64) -> Self {
        Self { index, type_, scale }
    }

    /// Refinement with default midpoint scale (0.5), as in MFEM's
    /// `Refinement(elem, type)` constructor.
    pub fn with_midpoint(index: ElemId, type_: u8) -> Self {
        Self { index, type_, scale: 0.5 }
    }

    /// Whether this refinement splits along X.
    pub fn split_x(&self) -> bool { self.type_ & 1 != 0 }
    /// Whether this refinement splits along Y.
    pub fn split_y(&self) -> bool { self.type_ & 2 != 0 }
}

/// Parent→child embedding table (MFEM `CoarseFineTransformations.embeddings`).
#[derive(Debug, Clone)]
pub struct RefinementTransforms {
    /// Per-element parent element index (`None` = not a child / unrefined).
    pub embeddings: Vec<Option<ElemId>>,
}

impl RefinementTransforms {
    /// All children of `parent` (MFEM ref321 `FindChildren`).
    pub fn find_children(&self, parent: ElemId) -> Vec<ElemId> {
        self.embeddings
            .iter()
            .enumerate()
            .filter_map(|(i, &p)| (p == Some(parent)).then_some(i as ElemId))
            .collect()
    }
}

/// One edge split created by a refinement call.
///
/// `node` was inserted on the edge `(from, to)` at parameter `scale`
/// measured from `from` (the marked element's local edge orientation).
#[derive(Debug, Clone, Copy)]
pub struct EdgeSplit {
    /// New (or reused) node at the split position.
    pub node: NodeId,
    /// Edge start (element-local orientation).
    pub from: NodeId,
    /// Edge end (element-local orientation).
    pub to: NodeId,
    /// Position along `from → to` in [0,1].
    pub scale: f64,
}

/// Result of a [`general_refinement_2d`] call.
#[derive(Debug, Clone)]
pub struct GeneralRefineResult2D {
    /// Refined mesh (node ids preserved, elements replaced in place).
    pub mesh: Mesh<2>,
    /// P1 hanging-node constraints introduced by *this* call (vertex-level).
    pub constraints: Vec<HangingNodeConstraint>,
    /// All edge splits performed by this call (for lineage bookkeeping).
    pub splits: Vec<EdgeSplit>,
    /// Parent embeddings for the new element layout.
    pub transforms: RefinementTransforms,
}

/// General non-conforming refinement for 2D Quad4 meshes with arbitrary scale.
///
/// Each refinement splits one element along one axis at the given scale.
/// Refinements within one call must refer to distinct elements (matching
/// MFEM `ref321`'s usage). When two marked elements share a cut edge, the
/// scales must agree (otherwise MFEM would create two hanging nodes on the
/// same edge — not needed by `ref321`, so we reject it).
pub fn general_refinement_2d(
    mesh: &Mesh<2>,
    refs: &[Refinement],
) -> GeneralRefineResult2D {
    assert!(
        mesh.elem_type == ElementType::Quad4,
        "general_refinement_2d: only Quad4 meshes are supported"
    );

    if refs.is_empty() {
        return GeneralRefineResult2D {
            mesh: mesh.clone(),
            constraints: Vec::new(),
            splits: Vec::new(),
            transforms: RefinementTransforms {
                embeddings: vec![None; mesh.n_elems()],
            },
        };
    }

    let n_elems = mesh.n_elems();
    let ref_map: HashMap<ElemId, Refinement> = refs.iter().map(|r| (r.index, *r)).collect();

    // ── Edge adjacency of the input mesh ─────────────────────────────────
    let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_quad() {
            edge_elems.entry(quad_edge_key(ns[a], ns[b])).or_default().push(e);
        }
    }

    // ── Split nodes on cut edges ─────────────────────────────────────────
    // X-split cuts the bottom edge (n0,n1) and top edge (n3,n2);
    // Y-split cuts the left edge (n0,n3) and right edge (n1,n2).
    // Coordinates are computed in the element-local edge orientation so the
    // constraint `u[c] = (1-s)·u[from] + s·u[to]` matches the geometry.
    let mut split_nodes: HashMap<(NodeId, NodeId), (NodeId, f64)> = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();
    let mut next_node = mesh.n_nodes() as NodeId;
    let mut splits: Vec<EdgeSplit> = Vec::new();

    let mut get_or_create = |key: (NodeId, NodeId),
                             from: NodeId,
                             to: NodeId,
                             s: f64,
                             split_nodes: &mut HashMap<(NodeId, NodeId), (NodeId, f64)>,
                             new_coords: &mut Vec<f64>,
                             next_node: &mut NodeId,
                             splits: &mut Vec<EdgeSplit>| -> NodeId {
        match split_nodes.get(&key) {
            Some(&(node, s0)) => {
                assert!(
                    (s0 - s).abs() < 1e-14,
                    "general_refinement_2d: edge split at two different scales in one call"
                );
                node
            }
            None => {
                let xa = mesh.coords_of(from);
                let xb = mesh.coords_of(to);
                let px = xa[0] + s * (xb[0] - xa[0]);
                let py = xa[1] + s * (xb[1] - xa[1]);
                // Reuse an existing node at the same coordinate (NC semantics,
                // e.g. a hanging node created by an earlier refinement round).
                let node = (0..mesh.n_nodes() as NodeId)
                    .find(|&n| {
                        let c = mesh.coords_of(n);
                        (c[0] - px).abs() < 1e-12 && (c[1] - py).abs() < 1e-12
                    })
                    .unwrap_or_else(|| {
                        new_coords.push(px);
                        new_coords.push(py);
                        let id = *next_node;
                        *next_node += 1;
                        id
                    });
                split_nodes.insert(key, (node, s));
                splits.push(EdgeSplit { node, from, to, scale: s });
                node
            }
        }
    };

    // (element, local_edge_index) → split node
    let mut elem_edge_split: HashMap<(ElemId, usize), NodeId> = HashMap::new();

    for &r in refs {
        let e = r.index;
        let ns = mesh.elem_nodes(e);
        let s = r.scale;

        if r.split_x() && !r.split_y() {
            let bottom = get_or_create(
                quad_edge_key(ns[0], ns[1]), ns[0], ns[1], s,
                &mut split_nodes, &mut new_coords, &mut next_node, &mut splits,
            );
            let top = get_or_create(
                quad_edge_key(ns[3], ns[2]), ns[3], ns[2], s,
                &mut split_nodes, &mut new_coords, &mut next_node, &mut splits,
            );
            elem_edge_split.insert((e, 0), bottom);
            elem_edge_split.insert((e, 2), top);
        } else if r.split_y() && !r.split_x() {
            let left = get_or_create(
                quad_edge_key(ns[0], ns[3]), ns[0], ns[3], s,
                &mut split_nodes, &mut new_coords, &mut next_node, &mut splits,
            );
            let right = get_or_create(
                quad_edge_key(ns[1], ns[2]), ns[1], ns[2], s,
                &mut split_nodes, &mut new_coords, &mut next_node, &mut splits,
            );
            elem_edge_split.insert((e, 3), left);
            elem_edge_split.insert((e, 1), right);
        } else {
            panic!("general_refinement_2d: only single-direction splits (X or Y) are supported");
        }
    }

    // ── New element connectivity ─────────────────────────────────────────
    let mut new_conn: Vec<NodeId> = Vec::new();
    let mut new_tags: Vec<i32> = Vec::new();
    let mut embeddings: Vec<Option<ElemId>> = Vec::with_capacity(n_elems * 2);

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let tag = mesh.elem_tags[e as usize];

        if let Some(&r) = ref_map.get(&e) {
            if r.split_x() && !r.split_y() {
                let bm = elem_edge_split[&(e, 0)];
                let tm = elem_edge_split[&(e, 2)];
                // Left child: [n0, bottom, top, n3] (CCW).
                new_conn.extend_from_slice(&[ns[0], bm, tm, ns[3]]);
                new_tags.push(tag);
                embeddings.push(Some(e));
                // Right child: [bottom, n1, n2, top].
                new_conn.extend_from_slice(&[bm, ns[1], ns[2], tm]);
                new_tags.push(tag);
                embeddings.push(Some(e));
            } else {
                let lm = elem_edge_split[&(e, 3)];
                let rm = elem_edge_split[&(e, 1)];
                // Bottom child: [n0, n1, right, left].
                new_conn.extend_from_slice(&[ns[0], ns[1], rm, lm]);
                new_tags.push(tag);
                embeddings.push(Some(e));
                // Top child: [left, right, n2, n3].
                new_conn.extend_from_slice(&[lm, rm, ns[2], ns[3]]);
                new_tags.push(tag);
                embeddings.push(Some(e));
            }
        } else {
            new_conn.extend_from_slice(ns);
            new_tags.push(tag);
            embeddings.push(None);
        }
    }

    // ── Rebuild boundary edges (split boundary cut edges) ────────────────
    let n_faces = mesh.n_faces();
    let mut new_face_conn: Vec<NodeId> = Vec::new();
    let mut new_face_tags: Vec<i32> = Vec::new();
    for f in 0..n_faces {
        let a = mesh.face_conn[2 * f];
        let b = mesh.face_conn[2 * f + 1];
        let tag = mesh.face_tags[f];
        match split_nodes.get(&quad_edge_key(a, b)) {
            Some(&(mid, _)) => {
                new_face_conn.extend_from_slice(&[a, mid]);
                new_face_tags.push(tag);
                new_face_conn.extend_from_slice(&[mid, b]);
                new_face_tags.push(tag);
            }
            None => {
                new_face_conn.extend_from_slice(&[a, b]);
                new_face_tags.push(tag);
            }
        }
    }

    let new_mesh = Mesh::<2>::uniform(
        new_coords,
        new_conn,
        new_tags,
        ElementType::Quad4,
        new_face_conn,
        new_face_tags,
        ElementType::Line2,
    );

    // ── P1 hanging-node constraints from this call ───────────────────────
    // A split node on an edge is hanging iff some other element shares that
    // edge without being refined (at the same scale).
    let mut constraints = Vec::new();
    for (&(e, edge_idx), &node) in &elem_edge_split {
        let ns = mesh.elem_nodes(e);
        let (a, b) = match edge_idx {
            0 => (ns[0], ns[1]), // bottom
            1 => (ns[1], ns[2]), // right
            2 => (ns[3], ns[2]), // top
            _ => (ns[0], ns[3]), // left
        };
        let (_, s) = split_nodes[&quad_edge_key(a, b)];
        if let Some(adj) = edge_elems.get(&quad_edge_key(a, b)) {
            let all_refined = adj.iter().all(|&n| n == e || ref_map.contains_key(&n));
            if !all_refined {
                constraints.push(HangingNodeConstraint {
                    constrained: node as usize,
                    parent_a: a as usize,
                    parent_b: b as usize,
                    coeff_a: 1.0 - s,
                    coeff_b: s,
                    extra: Vec::new(),
                });
            }
        }
    }
    constraints.sort_by_key(|c| c.constrained);
    constraints.dedup_by(|a, b| a.constrained == b.constrained);

    GeneralRefineResult2D {
        mesh: new_mesh,
        constraints,
        splits,
        transforms: RefinementTransforms { embeddings },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refine31_x_split_sequence() {
        // 2x2 Cartesian quad mesh; MFEM ref321-style 3:1 refinement of elem 0
        // along X: split at 2/3, then split the first child at 1/2.
        let m0 = Mesh::<2>::make_cartesian_2d(2, 2, 1.0, 1.0);
        assert_eq!(m0.n_elems(), 4);
        let n0 = m0.n_nodes();

        // Elem 0 = [0,1]x[0,1] quad (nodes 0,1,?,?) in MFEM ordering.
        let r1 = general_refinement_2d(&m0, &[Refinement::new(0, 1, 2.0 / 3.0)]);
        assert_eq!(r1.mesh.n_elems(), 5);
        // One new node at (2/3 on elem-0 edges) -> two nodes? bottom + top cut
        // share coordinates with existing? No: bottom edge y=0 x=2/3 and top
        // edge y=1 x=2/3 are two distinct new nodes.
        assert_eq!(r1.mesh.n_nodes(), n0 + 2);
        // Elem 0's right neighbor (elem 1, [1,2]x[0,1]) is not refined ->
        // hanging constraints on the right edge (x=1) are NOT created (that
        // edge is not cut); the cut edges are interior to elem 0 except the
        // top/bottom boundary? Elem 0 occupies [0,1]^2, its top edge y=1 is
        // boundary shared with elem 2 ([0,1]x[1,2])! So constraints exist.
        assert!(!r1.constraints.is_empty(), "top neighbor (elem 2) unrefined -> hanging");
        // Splits recorded with local orientation (from -> to, scale).
        assert_eq!(r1.splits.len(), 2);
        for s in &r1.splits {
            assert!((s.scale - 2.0 / 3.0).abs() < 1e-14);
            let (a, b) = (r1.mesh.coords_of(s.from), r1.mesh.coords_of(s.to));
            let c = r1.mesh.coords_of(s.node);
            let px = a[0] + s.scale * (b[0] - a[0]);
            let py = a[1] + s.scale * (b[1] - a[1]);
            assert!((c[0] - px).abs() < 1e-12 && (c[1] - py).abs() < 1e-12);
        }
        // Children of elem 0 = the two left-most elements.
        let children = r1.transforms.find_children(0);
        assert_eq!(children.len(), 2);

        // Second call: split the first child at 1/2 along X.
        let e1 = children[0];
        let r2 = general_refinement_2d(&r1.mesh, &[Refinement::with_midpoint(e1, 1)]);
        assert_eq!(r2.mesh.n_elems(), 6);
        // Elem 0 spans [0,1/2]; the 2/3 split put the cut at x=1/3, and the
        // midpoint split of the left child puts the new node at x=1/6.
        // Coordinate-based reuse must NOT pick any existing node.
        let new_node = r2.splits[0].node;
        let c = r2.mesh.coords_of(new_node);
        assert!((c[0] - 1.0 / 6.0).abs() < 1e-12, "node at x=1/6, got {}", c[0]);
        // Second split of the same sub-edge (top) must produce y=1 counterpart.
        assert_eq!(r2.splits.len(), 2);
    }

    #[test]
    fn shared_edge_same_scale_reuses_node() {
        // Both elems of a 1x2 mesh split along Y at 2/3: the shared vertical
        // edge is cut once (reused), producing a conforming refinement with
        // no hanging node on it.
        let m0 = Mesh::<2>::make_cartesian_2d(2, 1, 1.0, 1.0);
        let refs = vec![
            Refinement::new(0, 2, 2.0 / 3.0),
            Refinement::new(1, 2, 2.0 / 3.0),
        ];
        let r = general_refinement_2d(&m0, &refs);
        assert_eq!(r.mesh.n_elems(), 4);
        // Cut nodes at (0,2/3), (1/2,2/3) [shared, reused], (1,2/3).
        assert_eq!(r.mesh.n_nodes(), m0.n_nodes() + 3);
        assert_eq!(r.splits.len(), 3); // shared-edge cut is recorded once
        // The shared edge x=1 was cut at the same scale by both elements ->
        // conforming, no constraint on it.
        assert!(
            r.constraints.is_empty(),
            "no hanging nodes when both neighbors refine at the same scale"
        );
    }
}
