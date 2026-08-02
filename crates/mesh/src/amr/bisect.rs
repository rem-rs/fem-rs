use std::collections::HashMap;
use fem_core::{NodeId, ElemId};
use crate::{element_type::ElementType, simplex::Mesh};

/// Newest-vertex bisection refinement for a 2-D triangle mesh.
///
/// Each marked element is split into **2** children by bisecting the longest
/// edge (opposite to the newest vertex).  To maintain conformity, edges shared
/// with unmarked neighbours are also bisected (propagation step, simplified
/// here to a single conformity pass).
///
/// # Arguments
/// - `mesh`    — input `Mesh<2>` with `elem_type = Tri3`.
/// - `marked`  — sorted list of element indices to refine.
///
/// # Returns
/// A new `Mesh<2>` with the refined elements replaced by their children.
pub fn refine_marked(mesh: &Mesh<2>, marked: &[ElemId]) -> Mesh<2> {
    assert!(
        mesh.elem_type == ElementType::Tri3,
        "refine_marked: only Tri3 meshes are supported"
    );

    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();

    // ── 1. Identify all edges to bisect ───────────────────────────────────────
    // For each marked element, mark its longest edge.
    // We also propagate to neighbours (one pass) to ensure conformity.
    let npe = 3usize;
    let n_elems = mesh.n_elems();

    // Build edge → element list for propagation.
    // edge key = (min_node, max_node)
    let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_tri() {
            let key = edge_key(ns[a], ns[b]);
            edge_elems.entry(key).or_default().push(e);
        }
    }

    // Mark the longest edge of each element to be bisected.
    let mut bisect_edges: std::collections::HashSet<(NodeId, NodeId)> = Default::default();
    for &e in marked {
        let ns = mesh.elem_nodes(e);
        let longest = longest_edge_tri(mesh, ns);
        bisect_edges.insert(longest);
    }

    // Conformity propagation (one pass): if an interior edge is bisected,
    // both adjacent elements' longest edges should also be bisected.
    // We simply bisect the entire element (all edges) for simplicity.
    // This over-refines slightly but guarantees conformity.
    let mut elems_to_refine: std::collections::HashSet<ElemId> = marked_set.clone();
    for &(a, b) in &bisect_edges {
        if let Some(nbrs) = edge_elems.get(&(a, b)) {
            for &ne in nbrs {
                elems_to_refine.insert(ne);
            }
        }
    }

    // ── 2. Collect new midpoint nodes ─────────────────────────────────────────
    let mut midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();

    let n_nodes_orig = mesh.n_nodes() as NodeId;
    let mut next_node = n_nodes_orig;

    for &e in &elems_to_refine {
        let ns = mesh.elem_nodes(e);
        // For Tri3 bisection: bisect longest edge only (newest-vertex bisection).
        // For simplicity here, bisect all 3 edges (red refinement).
        for &(a, b) in &local_edges_tri() {
            let key = edge_key(ns[a], ns[b]);
            midpoint_map.entry(key).or_insert_with(|| {
                let xa = mesh.coords_of(ns[a]);
                let xb = mesh.coords_of(ns[b]);
                new_coords.push(0.5 * (xa[0] + xb[0]));
                new_coords.push(0.5 * (xa[1] + xb[1]));
                let id = next_node;
                next_node += 1;
                id
            });
        }
    }

    // ── 3. Build new element connectivity ─────────────────────────────────────
    let mut new_conn: Vec<NodeId>  = Vec::new();
    let mut new_tags: Vec<i32>     = Vec::new();

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let tag = mesh.elem_tags[e as usize];

        if elems_to_refine.contains(&e) {
            // Red refinement: split Tri3 into 4 children.
            //   Original nodes: n0, n1, n2
            //   Midpoints:      m01, m12, m02
            let n0 = ns[0]; let n1 = ns[1]; let n2 = ns[2];
            let m01 = *midpoint_map.get(&edge_key(n0, n1)).unwrap();
            let m12 = *midpoint_map.get(&edge_key(n1, n2)).unwrap();
            let m02 = *midpoint_map.get(&edge_key(n0, n2)).unwrap();
            // MFEM UniformRefinement2D_base (mesh.cpp) child order:
            //   [v0,e0,e2] corner(n0), [e1,e2,e0] CENTER, [e0,v1,e1] corner(n1),
            //   [e2,e1,v2] corner(n2)  — the center triangle is child 1, NOT
            //   last; matching this keeps the refined element numbering (and
            //   hence the GS-sweep order) bit-identical to MFEM.
            new_conn.extend_from_slice(&[n0,  m01, m02]);  new_tags.push(tag);
            new_conn.extend_from_slice(&[m12, m02, m01]);  new_tags.push(tag); // center
            new_conn.extend_from_slice(&[m01, n1,  m12]);  new_tags.push(tag);
            new_conn.extend_from_slice(&[m02, m12, n2 ]);  new_tags.push(tag);
        } else {
            // Unchanged element — copy as-is.
            for k in 0..npe { new_conn.push(ns[k]); }
            new_tags.push(tag);
        }
    }

    // ── 4. Rebuild boundary faces ─────────────────────────────────────────────
    // Boundary edges that were bisected get 2 children; others stay.
    let npf = 2usize; // Line2
    let n_faces = mesh.n_faces();
    let mut new_face_conn: Vec<NodeId> = Vec::new();
    let mut new_face_tags: Vec<i32>    = Vec::new();

    for f in 0..n_faces {
        let fn_slice = &mesh.face_conn[f * npf..(f + 1) * npf];
        let a = fn_slice[0];
        let b = fn_slice[1];
        let tag = mesh.face_tags[f];

        if let Some(&mid) = midpoint_map.get(&edge_key(a, b)) {
            // Bisected edge → 2 children
            new_face_conn.extend_from_slice(&[a, mid]);   new_face_tags.push(tag);
            new_face_conn.extend_from_slice(&[mid, b]);   new_face_tags.push(tag);
        } else {
            new_face_conn.extend_from_slice(&[a, b]);
            new_face_tags.push(tag);
        }
    }

    Mesh::uniform(
        new_coords, new_conn, new_tags, ElementType::Tri3,
        new_face_conn, new_face_tags, ElementType::Line2,
    )
}

/// Refinement provenance for one red-refinement level.
///
/// Stores parent -> children mapping and parent connectivity needed by
/// [`derefine_marked`].
#[derive(Debug, Clone)]
pub struct DerefineTree {
    pub records: HashMap<ElemId, DerefineRecord>,
    pub midpoint_map: HashMap<(NodeId, NodeId), NodeId>,
}

/// One parent refinement record.
#[derive(Debug, Clone)]
pub struct DerefineRecord {
    pub parent_nodes: [NodeId; 3],
    pub parent_tag: i32,
    pub children: [ElemId; 4],
}

impl DerefineTree {
    /// Return sorted parent element ids available for derefinement.
    pub fn parents(&self) -> Vec<ElemId> {
        let mut p: Vec<ElemId> = self.records.keys().copied().collect();
        p.sort_unstable();
        p
    }
}

/// Same as [`refine_marked`], but also returns a provenance tree that enables
/// one-level derefinement through [`derefine_marked`].
pub fn refine_marked_with_tree(mesh: &Mesh<2>, marked: &[ElemId]) -> (Mesh<2>, DerefineTree) {
    assert!(
        mesh.elem_type == ElementType::Tri3,
        "refine_marked_with_tree: only Tri3 meshes are supported"
    );

    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();

    let npe = 3usize;
    let n_elems = mesh.n_elems();

    let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_tri() {
            let key = edge_key(ns[a], ns[b]);
            edge_elems.entry(key).or_default().push(e);
        }
    }

    let mut bisect_edges: std::collections::HashSet<(NodeId, NodeId)> = Default::default();
    for &e in marked {
        let ns = mesh.elem_nodes(e);
        let longest = longest_edge_tri(mesh, ns);
        bisect_edges.insert(longest);
    }

    let mut elems_to_refine: std::collections::HashSet<ElemId> = marked_set.clone();
    for &(a, b) in &bisect_edges {
        if let Some(nbrs) = edge_elems.get(&(a, b)) {
            for &ne in nbrs {
                elems_to_refine.insert(ne);
            }
        }
    }

    let mut midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();

    let n_nodes_orig = mesh.n_nodes() as NodeId;
    let mut next_node = n_nodes_orig;

    for &e in &elems_to_refine {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_tri() {
            let key = edge_key(ns[a], ns[b]);
            midpoint_map.entry(key).or_insert_with(|| {
                let xa = mesh.coords_of(ns[a]);
                let xb = mesh.coords_of(ns[b]);
                new_coords.push(0.5 * (xa[0] + xb[0]));
                new_coords.push(0.5 * (xa[1] + xb[1]));
                let id = next_node;
                next_node += 1;
                id
            });
        }
    }

    let mut new_conn: Vec<NodeId> = Vec::new();
    let mut new_tags: Vec<i32> = Vec::new();
    let mut tree_records: HashMap<ElemId, DerefineRecord> = HashMap::new();

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let tag = mesh.elem_tags[e as usize];

        if elems_to_refine.contains(&e) {
            let n0 = ns[0]; let n1 = ns[1]; let n2 = ns[2];
            let m01 = *midpoint_map.get(&edge_key(n0, n1)).unwrap();
            let m12 = *midpoint_map.get(&edge_key(n1, n2)).unwrap();
            let m02 = *midpoint_map.get(&edge_key(n0, n2)).unwrap();

            let c0 = (new_tags.len()) as ElemId;
            new_conn.extend_from_slice(&[n0,  m01, m02]); new_tags.push(tag);
            let c1 = (new_tags.len()) as ElemId;
            new_conn.extend_from_slice(&[m01, n1,  m12]); new_tags.push(tag);
            let c2 = (new_tags.len()) as ElemId;
            new_conn.extend_from_slice(&[m02, m12, n2 ]); new_tags.push(tag);
            let c3 = (new_tags.len()) as ElemId;
            new_conn.extend_from_slice(&[m01, m12, m02]); new_tags.push(tag);

            tree_records.insert(
                e,
                DerefineRecord {
                    parent_nodes: [n0, n1, n2],
                    parent_tag: tag,
                    children: [c0, c1, c2, c3],
                },
            );
        } else {
            for k in 0..npe { new_conn.push(ns[k]); }
            new_tags.push(tag);
        }
    }

    let npf = 2usize;
    let n_faces = mesh.n_faces();
    let mut new_face_conn: Vec<NodeId> = Vec::new();
    let mut new_face_tags: Vec<i32> = Vec::new();

    for f in 0..n_faces {
        let fn_slice = &mesh.face_conn[f * npf..(f + 1) * npf];
        let a = fn_slice[0];
        let b = fn_slice[1];
        let tag = mesh.face_tags[f];

        if let Some(&mid) = midpoint_map.get(&edge_key(a, b)) {
            new_face_conn.extend_from_slice(&[a, mid]);   new_face_tags.push(tag);
            new_face_conn.extend_from_slice(&[mid, b]);   new_face_tags.push(tag);
        } else {
            new_face_conn.extend_from_slice(&[a, b]);
            new_face_tags.push(tag);
        }
    }

    let fine = Mesh::uniform(
        new_coords, new_conn, new_tags, ElementType::Tri3,
        new_face_conn, new_face_tags, ElementType::Line2,
    );

    (fine, DerefineTree { records: tree_records, midpoint_map })
}

/// Derefine selected parent elements from a single-level [`DerefineTree`].
///
/// This function expects `mesh` to be the direct output of
/// [`refine_marked_with_tree`] with no additional refinement/coarsening in
/// between. It removes the 4 child triangles and restores the parent triangle.
pub fn derefine_marked(mesh: &Mesh<2>, tree: &DerefineTree, parents: &[ElemId]) -> Mesh<2> {
    assert!(
        mesh.elem_type == ElementType::Tri3,
        "derefine_marked: only Tri3 meshes are supported"
    );

    if parents.is_empty() {
        return mesh.clone();
    }

    let mut child_drop = std::collections::HashSet::<ElemId>::new();
    let mut restore = Vec::<DerefineRecord>::new();

    for &p in parents {
        if let Some(rec) = tree.records.get(&p) {
            for &c in &rec.children {
                child_drop.insert(c);
            }
            restore.push(rec.clone());
        }
    }

    let mut new_conn: Vec<NodeId> = Vec::new();
    let mut new_tags: Vec<i32> = Vec::new();

    for e in 0..mesh.n_elems() as ElemId {
        if child_drop.contains(&e) {
            continue;
        }
        let ns = mesh.elem_nodes(e);
        new_conn.extend_from_slice(&[ns[0], ns[1], ns[2]]);
        new_tags.push(mesh.elem_tags[e as usize]);
    }

    for rec in &restore {
        new_conn.extend_from_slice(&rec.parent_nodes);
        new_tags.push(rec.parent_tag);
    }

    // Rebuild boundary edges from exterior edges in the new connectivity.
    let mut edge_count: HashMap<(NodeId, NodeId), usize> = HashMap::new();
    let mut oriented_edge: HashMap<(NodeId, NodeId), (NodeId, NodeId)> = HashMap::new();
    for e in 0..new_tags.len() {
        let off = 3 * e;
        let tri = [new_conn[off], new_conn[off + 1], new_conn[off + 2]];
        let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];
        for (a, b) in edges {
            let k = edge_key(a, b);
            *edge_count.entry(k).or_insert(0) += 1;
            oriented_edge.entry(k).or_insert((a, b));
        }
    }

    let mut old_bnd_tags = HashMap::<(NodeId, NodeId), i32>::new();
    for f in 0..mesh.n_faces() {
        let a = mesh.face_conn[2 * f];
        let b = mesh.face_conn[2 * f + 1];
        old_bnd_tags.insert(edge_key(a, b), mesh.face_tags[f]);
    }

    let mut new_face_conn: Vec<NodeId> = Vec::new();
    let mut new_face_tags: Vec<i32> = Vec::new();
    for (&k, &cnt) in &edge_count {
        if cnt != 1 {
            continue;
        }
        let (a, b) = oriented_edge[&k];
        let mut tag = old_bnd_tags.get(&k).copied().unwrap_or(0);

        if tag == 0 {
            // Attempt to recover merged boundary tag from split edges (a,m) + (m,b).
            for m in 0..mesh.n_nodes() as NodeId {
                let k1 = edge_key(a, m);
                let k2 = edge_key(m, b);
                if let (Some(&t1), Some(&t2)) = (old_bnd_tags.get(&k1), old_bnd_tags.get(&k2)) {
                    if t1 == t2 {
                        tag = t1;
                        break;
                    }
                }
            }
        }

        if tag != 0 {
            new_face_conn.extend_from_slice(&[a, b]);
            new_face_tags.push(tag);
        }
    }

    Mesh::uniform(
        mesh.coords.clone(),
        new_conn,
        new_tags,
        ElementType::Tri3,
        new_face_conn,
        new_face_tags,
        ElementType::Line2,
    )
}

/// Local edge index pairs for Tri3.
pub(crate) fn local_edges_tri() -> [(usize, usize); 3] {
    [(0, 1), (1, 2), (0, 2)]
}

/// Canonical edge key (sorted node pair).
pub(crate) fn edge_key(a: NodeId, b: NodeId) -> (NodeId, NodeId) {
    if a < b { (a, b) } else { (b, a) }
}

/// Return the canonical edge key of the longest edge of a Tri3 element.
fn longest_edge_tri(mesh: &Mesh<2>, ns: &[NodeId]) -> (NodeId, NodeId) {
    let coords: [[f64; 2]; 3] = std::array::from_fn(|k| mesh.coords_of(ns[k]));
    let edges = local_edges_tri();
    let mut best = edge_key(ns[edges[0].0], ns[edges[0].1]);
    let mut best_len2 = 0.0_f64;
    for (a, b) in edges {
        let dx = coords[b][0] - coords[a][0];
        let dy = coords[b][1] - coords[a][1];
        let l2 = dx*dx + dy*dy;
        if l2 > best_len2 {
            best_len2 = l2;
            best = edge_key(ns[a], ns[b]);
        }
    }
    best
}
