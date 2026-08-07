//! 2-D red-green conforming refinement — 1:1 port of MFEM `Mesh::LocalRefinement`
//! (2-D branch, mesh.cpp ~10603) with `UniformRefinement` (red, mesh.cpp ~11637)
//! and `Bisection` (green, mesh.cpp ~11391).
//!
//! MFEM semantics reproduced here:
//! * A fixed vertex-to-vertex edge table (`v_to_v`, insertion-ordered =
//!   element × local-edge order) over the **initial** mesh vertices.
//! * `edge1`/`edge2` per initial edge hold the (up to two) adjacent elements;
//!   `middle` holds the midpoint node once the edge has been split.
//! * Red refinement of every marked element (4 children; the parent slot
//!   becomes the center triangle, 3 corners are appended) and updates the
//!   tables.  The "other side" neighbour of each split edge goes to `edge1`.
//! * Green closure loop: iterate the initial edges in index order; any edge
//!   with `middle != -1 && edge1 != -1` triggers `Bisection(edge1)`, which
//!   bisects **that element's own** `(vert[0], vert[1])` edge (its refinement
//!   edge — elements are rotated so the longest edge sits at vert[0..1]),
//!   creating the midpoint if missing and promoting the far neighbour into
//!   `edge1`.  Repeat until a full pass finds nothing.
//! * Boundary segments are split where their initial edge has a midpoint.

use std::collections::HashMap;
use fem_core::{NodeId, ElemId};
use crate::cad::{ProjectionConfig, project_boundary_to_cad};
use crate::element_type::ElementType;
use crate::simplex::Mesh;
use super::{edge_key, local_edges_tri};

// ─── MFEM LocalRefinement state ──────────────────────────────────────────────

/// Mutable state of the MFEM `LocalRefinement` 2-D algorithm.
struct LocalRefinement {
    /// Node coordinates, flat `[x0, y0, x1, y1, …]` (grows with midpoints).
    coords: Vec<f64>,
    /// Flat Tri3 connectivity, `3` entries per element (in-place replacement
    /// of refined parents + appends of children, exactly like MFEM).
    conn: Vec<NodeId>,
    /// Per-element attribute (material) tags.
    tags: Vec<i32>,
    /// Initial-mesh edge table (index = insertion order).
    v_to_v: Vec<(NodeId, NodeId)>,
    /// `(min,max)` node pair → index into `v_to_v`.
    edge_map: HashMap<(NodeId, NodeId), usize>,
    /// First adjacent element of each initial edge (`None` = no pending
    /// refinement, mirroring MFEM's `-1`).
    edge1: Vec<Option<ElemId>>,
    /// Second adjacent element of each initial edge.
    edge2: Vec<Option<ElemId>>,
    /// Midpoint node of each initial edge (`None` = not split yet).
    middle: Vec<Option<NodeId>>,
    /// Number of initial (original) vertices — used to test whether an edge
    /// is an initial edge before touching the fixed table.
    n_vertices: NodeId,
}

impl LocalRefinement {
    fn new(mesh: &Mesh<2>) -> Self {
        let n_elems = mesh.n_elems();
        let n_vertices = mesh.n_nodes() as NodeId;

        // 1. Vertex-to-vertex table, insertion-ordered like MFEM's DSTable
        //    (Push returns the existing index for a known edge, else appends).
        let mut edge_map: HashMap<(NodeId, NodeId), usize> = HashMap::new();
        let mut v_to_v: Vec<(NodeId, NodeId)> = Vec::new();
        for e in 0..n_elems as ElemId {
            let ns = mesh.elem_nodes(e);
            for &(a, b) in &local_edges_tri() {
                let k = edge_key(ns[a], ns[b]);
                if !edge_map.contains_key(&k) {
                    edge_map.insert(k, v_to_v.len());
                    v_to_v.push(k);
                }
            }
        }
        let n_edges = v_to_v.len();

        // 2. edge1/edge2 per initial edge (in element × local-edge order).
        let mut edge1: Vec<Option<ElemId>> = vec![None; n_edges];
        let mut edge2: Vec<Option<ElemId>> = vec![None; n_edges];
        for e in 0..n_elems as ElemId {
            let ns = mesh.elem_nodes(e);
            for &(a, b) in &local_edges_tri() {
                let i = edge_map[&edge_key(ns[a], ns[b])];
                if edge1[i].is_none() { edge1[i] = Some(e); } else { edge2[i] = Some(e); }
            }
        }

        LocalRefinement {
            coords: mesh.coords.clone(),
            conn: mesh.conn.clone(),
            tags: mesh.elem_tags.clone(),
            v_to_v,
            edge_map,
            edge1,
            edge2,
            middle: vec![None; n_edges],
            n_vertices,
        }
    }

    /// Index of an initial edge from its (unsorted) endpoint pair.
    fn edge_index(&self, a: NodeId, b: NodeId) -> usize {
        self.edge_map[&edge_key(a, b)]
    }

    /// Append a new midpoint node halfway between `a` and `b` (both are
    /// original vertices for every call site: red midpoints and the green
    /// Bisection edge lie on initial edges).
    fn new_midpoint(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let id = (self.coords.len() / 2) as NodeId;
        let (xa, ya) = (self.coords[a as usize * 2], self.coords[a as usize * 2 + 1]);
        let (xb, yb) = (self.coords[b as usize * 2], self.coords[b as usize * 2 + 1]);
        self.coords.push(0.5 * (xa + xb));
        self.coords.push(0.5 * (ya + yb));
        id
    }

    /// MFEM `UniformRefinement` (red): split element `el` into 4 children.
    /// The parent slot becomes the center triangle; the 3 corner triangles are
    /// appended.  Updates `middle`/`edge1` for the 3 split edges.
    fn uniform_refine(&mut self, el: ElemId) {
        let off = el as usize * 3;
        let v = [self.conn[off], self.conn[off + 1], self.conn[off + 2]];
        let mut bisect = [0usize; 3];
        let mut v_new = [NodeId::MAX; 3];
        for j in 0..3 {
            bisect[j] = self.edge_index(v[j], v[(j + 1) % 3]);
        }
        for j in 0..3 {
            let b = bisect[j];
            match self.middle[b] {
                None => {
                    v_new[j] = self.new_midpoint(v[j], v[(j + 1) % 3]);
                    // The other-side neighbour is now the pending element.
                    if self.edge1[b] == Some(el) { self.edge1[b] = self.edge2[b]; }
                    self.middle[b] = Some(v_new[j]);
                }
                Some(m) => {
                    v_new[j] = m;
                    // This edge needs no further (green) refinement.
                    self.edge1[b] = None;
                }
            }
        }
        // 2. Children (MFEM order: parent slot = center, then 3 corners).
        //    center = [m12, m02, m01], corner0 = [v0, m01, m02],
        //    corner1 = [m01, v1, m12], corner2 = [m02, m12, v2]
        self.conn[off] = v_new[1];
        self.conn[off + 1] = v_new[2];
        self.conn[off + 2] = v_new[0];
        let tag = self.tags[el as usize];
        self.conn.extend_from_slice(&[v[0], v_new[0], v_new[2]]);
        self.tags.push(tag);
        self.conn.extend_from_slice(&[v_new[0], v[1], v_new[1]]);
        self.tags.push(tag);
        self.conn.extend_from_slice(&[v_new[2], v_new[1], v[2]]);
        self.tags.push(tag);
    }

    /// MFEM `Bisection` (green): bisect element `el` along its own refinement
    /// edge `(vert[0], vert[1])` — which is guaranteed to be an initial edge
    /// for every element reachable through `edge1`.  Children:
    /// `[v2, v0, m]` (in place) and `[v1, v2, m]` (appended).
    fn bisection(&mut self, el: ElemId) {
        let off = el as usize * 3;
        let v = [self.conn[off], self.conn[off + 1], self.conn[off + 2]];
        let bisect = self.edge_index(v[0], v[1]);
        let v_new;
        match self.middle[bisect] {
            None => {
                v_new = self.new_midpoint(v[0], v[1]);
                if self.edge1[bisect] == Some(el) { self.edge1[bisect] = self.edge2[bisect]; }
                self.middle[bisect] = Some(v_new);
            }
            Some(m) => {
                v_new = m;
                self.edge1[bisect] = None;
            }
        }
        // 2. Children.
        self.conn[off] = v[2];
        self.conn[off + 1] = v[0];
        self.conn[off + 2] = v_new;
        let tag = self.tags[el as usize];
        self.conn.extend_from_slice(&[v[1], v[2], v_new]);
        self.tags.push(tag);
        let new_el = (self.tags.len() - 1) as ElemId;
        // 3. The new element's refinement edge (v[1], v[2]) — only update the
        //    tables if both endpoints are original vertices (initial edge).
        if v[1] < self.n_vertices && v[2] < self.n_vertices {
            let b2 = self.edge_index(v[1], v[2]);
            if self.edge1[b2] == Some(el) {
                self.edge1[b2] = Some(new_el);
            } else if self.edge2[b2] == Some(el) {
                self.edge2[b2] = Some(new_el);
            }
        }
    }

    /// Run the green closure loop: repeat full passes over the initial edges
    /// until a pass refines nothing (MFEM's `do { … } while (need_refinement)`).
    fn green_closure(&mut self) {
        loop {
            let mut need = false;
            for i in 0..self.v_to_v.len() {
                if self.middle[i].is_some() && self.edge1[i].is_some() {
                    need = true;
                    let el = self.edge1[i].unwrap();
                    self.bisection(el);
                }
            }
            if !need { break; }
        }
    }

    /// 5. Update boundary elements: split every boundary segment whose initial
    ///    edge carries a midpoint.
    fn finish(self, mesh: &Mesh<2>) -> Mesh<2> {
        let n_faces = mesh.n_faces();
        let mut face_conn: Vec<NodeId> = Vec::new();
        let mut face_tags: Vec<i32> = Vec::new();
        for f in 0..n_faces {
            let a = mesh.face_conn[f * 2];
            let b = mesh.face_conn[f * 2 + 1];
            let tag = mesh.face_tags[f];
            let bisect = self.edge_map.get(&edge_key(a, b));
            match bisect.and_then(|&i| self.middle[i]) {
                Some(m) => {
                    face_conn.extend_from_slice(&[a, m]);
                    face_conn.extend_from_slice(&[m, b]);
                    face_tags.push(tag);
                    face_tags.push(tag);
                }
                None => {
                    face_conn.extend_from_slice(&[a, b]);
                    face_tags.push(tag);
                }
            }
        }
        Mesh::uniform(
            self.coords, self.conn, self.tags, ElementType::Tri3,
            face_conn, face_tags, ElementType::Line2,
        )
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// MFEM `Mesh::LocalRefinement` for a Tri3 mesh: red-refine every marked
/// element, then run the green (bisection) closure loop over the initial edge
/// table until the mesh is conforming.  Node numbering, element ordering and
/// boundary splitting are bit-identical to MFEM (ex21/ex30 trajectory).
pub fn local_refinement(mesh: &Mesh<2>, marked: &[ElemId]) -> Mesh<2> {
    assert!(
        mesh.elem_type == ElementType::Tri3,
        "local_refinement: only Tri3 meshes are supported"
    );
    let mut lr = LocalRefinement::new(mesh);
    for &el in marked {
        lr.uniform_refine(el);
    }
    lr.green_closure();
    lr.finish(mesh)
}

/// Repeatedly refine marked Tri3 elements and their neighbours until no hanging
/// edges remain (conforming mesh closure).
///
/// Mirrors MFEM `Mesh::LocalRefinement` (used by `GeneralRefinement` in ex21):
/// 1. RED-refine every marked element (4 children);
/// 2. GREEN-bisect hanging neighbours (element-wise closure loop) until the
///    mesh is conforming.
pub fn closure_refine(
    mesh: &Mesh<2>,
    marked: &[ElemId],
    _max_iter: usize,
    project_boundary: Option<&ProjectionConfig>,
) -> Mesh<2> {
    assert!(
        mesh.elem_type == ElementType::Tri3,
        "closure_refine: only Tri3 meshes are supported"
    );

    // MFEM's green closure loop converges by itself (each bisection removes a
    // hanging edge); `max_iter` is kept only for API compatibility.
    let current = local_refinement(mesh, marked);

    if let Some(config) = project_boundary {
        project_boundary_to_cad(&current, config, 2)
    } else {
        current
    }
}

/// Convenience overload with a default iteration limit (20).
pub fn closure_refine_default(
    mesh: &Mesh<2>,
    marked: &[ElemId],
    project_boundary: Option<&ProjectionConfig>,
) -> Mesh<2> {
    closure_refine(mesh, marked, 20, project_boundary)
}
