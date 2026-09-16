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
/// edges remain (conforming mesh closure), or run the nonconforming (hanging
/// nodes) quad refinement for Quad4 meshes.
///
/// * Tri3: mirrors MFEM `Mesh::LocalRefinement` (the conforming
///   `GeneralRefinement` path used by ex21): 1. RED-refine every marked
///   element (4 children); 2. GREEN-bisect hanging neighbours (element-wise
///   closure loop) until the mesh is conforming.
/// * Quad4: mirrors MFEM `Mesh::NonconformingRefinement` (the nonconforming
///   `GeneralRefinement(refs, -1, nclimit)` path of the toys mandel/mondrian,
///   whose `-ncl` default is 1): refine only the marked elements, allowing
///   hanging nodes, then propagate to neighbours that violate `nc_limit = 1`
///   (see [`general_refinement_quad`]).  The hanging-node constraints of the
///   result are available from `general_refinement_quad`/`detect_hanging_quad`
///   (this stateless entry point drops them, like MFEM drops them for meshes
///   without a space).
pub fn closure_refine(
    mesh: &Mesh<2>,
    marked: &[ElemId],
    _max_iter: usize,
    project_boundary: Option<&ProjectionConfig>,
) -> Mesh<2> {
    // MFEM's green closure loop converges by itself (each bisection removes a
    // hanging edge); `max_iter` is kept only for API compatibility.
    match mesh.elem_type {
        ElementType::Tri3 => {
            let current = local_refinement(mesh, marked);
            if let Some(config) = project_boundary {
                project_boundary_to_cad(&current, config, 2)
            } else {
                current
            }
        }
        ElementType::Quad4 => {
            // `nc_limit = 1` = the toys' `-ncl 1` default; MFEM's API default
            // is 0 (no limit) — pass it explicitly via `general_refinement_quad`.
            general_refinement_quad(mesh, marked, 1, project_boundary).0
        }
        _ => panic!(
            "closure_refine: unsupported element type {:?} (Tri3/Quad4 only)",
            mesh.elem_type
        ),
    }
}

/// MFEM `Mesh::GeneralRefinement(refs, nonconforming=-1, nc_limit)` for Quad4
/// meshes (D160): the nonconforming path `Mesh::NonconformingRefinement`
/// (mesh.cpp ~11330), i.e. NCMesh 2-D refinement with hanging nodes.
///
/// MFEM semantics reproduced here (all splits are iso — MFEM `ref_type`
/// masked with `0x3` gives `XY` for squares, ncmesh.cpp ~1814):
/// 1. Refine every marked element in place into the 4 children
///    `[n0,m01,c,m30], [m01,n1,m12,c], [c,m12,n2,m23], [m30,c,m23,n3]`
///    (`refine_nonconforming_quad`, midpoint nodes reused across batches like
///    MFEM's `GetMidEdgeNode` hash).  No neighbour is forced.
/// 2. If `nc_limit > 0`, loop MFEM `NCMesh::LimitNCLevel`: refine every leaf
///    whose edge split level exceeds `nc_limit` (`limit_nc_level_quad`,
///    geometric `EdgeSplitLevel`/`GetLimitRefinements` semantics) until a full
///    pass finds nothing — this is what forces coarse neighbours of
///    twice-refined regions and reproduces MFEM's element counts.
///
/// Returns the refined mesh plus the P1 hanging-node constraints of the final
/// mesh (see `detect_hanging_quad`); the Tri3 `closure_refine` path always
/// returns a conforming mesh, the Quad4 path does not — consumers that build a
/// constrained space must consume these (the toys only write meshes and drop
/// them).
pub fn general_refinement_quad(
    mesh: &Mesh<2>,
    marked: &[ElemId],
    nc_limit: u32,
    project_boundary: Option<&ProjectionConfig>,
) -> (Mesh<2>, Vec<super::amr_inner::HangingNodeConstraint>) {
    use super::amr_inner::{limit_nc_level_quad, refine_nonconforming_quad};

    assert!(
        mesh.elem_type == ElementType::Quad4,
        "general_refinement_quad: only Quad4 meshes are supported"
    );

    if marked.is_empty() {
        // MFEM `NonconformingRefinement`: empty refinements → `last_operation
        // = NONE`, mesh unchanged.
        let m = match project_boundary {
            Some(config) => project_boundary_to_cad(mesh, config, 2),
            None => mesh.clone(),
        };
        return (m, Vec::new());
    }

    // 1. `ncmesh->Refine(refinements)` — split the marked elements only.
    let (mut current, mut constraints) = refine_nonconforming_quad(mesh, marked, None);
    // 2. `if (nc_limit > 0) ncmesh->LimitNCLevel(nc_limit)` — fixpoint loop.
    //    Each `refine_nonconforming_quad` call re-detects the full hanging set
    //    of the refined mesh, so the last batch's constraints describe the
    //    final mesh.
    if nc_limit > 0 {
        loop {
            let extra = limit_nc_level_quad(&current, nc_limit);
            if extra.is_empty() { break; }
            let (m, c) = refine_nonconforming_quad(&current, &extra, None);
            current = m;
            constraints = c;
        }
    }

    if let Some(config) = project_boundary {
        current = project_boundary_to_cad(&current, config, 2);
    }
    (current, constraints)
}

/// Anisotropic variant of [`general_refinement_quad`] — MFEM
/// `Mesh::GeneralRefinement(refs, -1, nc_limit)` with per-element
/// `Refinement::ref_type` (D229): each marked quad is X-bisected
/// (`ref_type & 0x3 == 1`, midpoints of the horizontal edges), Y-bisected
/// (2) or iso-split (3; 7 is masked to 3 for squares, ncmesh.cpp:1818), and
/// `LimitNCLevel` propagates **directionally** through
/// [`limit_nc_level_quad_aniso`] (MFEM `NCMesh::Iso` semantics: while only
/// iso refinements have been applied the limit forces iso splits; the first
/// X/Y split clears the flag and later limit refinements use the directional
/// bits).
///
/// `iso_in` is the caller's current `Iso` state (`true` for a mesh that has
/// never seen an anisotropic refinement); the returned bool is the state
/// after this call — pass it back on the next batch.
///
/// Returns `(mesh, iso_out, P1 hanging-node constraints)`.
pub fn general_refinement_quad_aniso(
    mesh: &Mesh<2>,
    marked: &[(ElemId, u8)],
    nc_limit: u32,
    iso_in: bool,
    project_boundary: Option<&ProjectionConfig>,
) -> (Mesh<2>, bool, Vec<super::amr_inner::HangingNodeConstraint>) {
    use super::amr_inner::{limit_nc_level_quad_aniso, refine_nonconforming_quad_aniso};
    use super::amr_inner::QuadRefineDir;

    assert!(
        mesh.elem_type == ElementType::Quad4,
        "general_refinement_quad_aniso: only Quad4 meshes are supported"
    );

    let to_dirs = |refs: &[(ElemId, u8)]| {
        refs.iter()
            .map(|&(e, rt)| {
                let dir = match rt & 0x3 {
                    1 => QuadRefineDir::X,
                    2 => QuadRefineDir::Y,
                    _ => QuadRefineDir::Both, // 3 (or 7 masked to 3): iso split
                };
                (e, dir)
            })
            .collect::<Vec<_>>()
    };
    // `if (ref_type != Refinement::XY) { Iso = false; }` (ncmesh.cpp:1882):
    // an X or Y split anywhere clears the iso flag.
    let mut iso = iso_in;
    for &(_, rt) in marked {
        let masked = rt & 0x3;
        if masked != 0 && masked != 0x3 {
            iso = false;
        }
    }

    if marked.is_empty() {
        let m = match project_boundary {
            Some(config) => project_boundary_to_cad(mesh, config, 2),
            None => mesh.clone(),
        };
        return (m, iso, Vec::new());
    }

    // 1. `ncmesh->Refine(refinements)` — split the marked elements per their
    //    ref_type (no neighbour is forced; hanging nodes are legal in 2-D).
    let (mut current, mut constraints) =
        refine_nonconforming_quad_aniso(mesh, &to_dirs(marked), None);
    // 2. `if (nc_limit > 0) ncmesh->LimitNCLevel(nc_limit)` — directional
    //    fixpoint loop; X/Y limit splits also clear `Iso` (they go through
    //    `RefineElement` too).
    if nc_limit > 0 {
        loop {
            let extra = limit_nc_level_quad_aniso(&current, nc_limit, iso);
            if extra.is_empty() { break; }
            for &(_, rt) in &extra {
                let masked = rt & 0x3;
                if masked != 0 && masked != 0x3 {
                    iso = false;
                }
            }
            let (m, c) = refine_nonconforming_quad_aniso(&current, &to_dirs(&extra), None);
            current = m;
            constraints = c;
        }
    }

    if let Some(config) = project_boundary {
        current = project_boundary_to_cad(&current, config, 2);
    }
    (current, iso, constraints)
}

/// Convenience overload with a default iteration limit (20).
pub fn closure_refine_default(
    mesh: &Mesh<2>,
    marked: &[ElemId],
    project_boundary: Option<&ProjectionConfig>,
) -> Mesh<2> {
    closure_refine(mesh, marked, 20, project_boundary)
}
