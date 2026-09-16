//! 1:1 port of the SQUARE (2-D quad) subset of MFEM's `NCMesh`
//! (mfem 4.10, `mesh/ncmesh.cpp`): the nonconforming refinement **tree** with
//! per-element `ref_type` (X = 1, Y = 2, XY = 3), MFEM node-id assignment
//! (`HashTable::GetId` midpoint hashing), the directional
//! `GetLimitRefinements`/`LimitNCLevel` NC-level propagation, the Hilbert SFC
//! leaf order (`CollectLeafElements`) and the "MFEM NC mesh v1.0" writer
//! (`NCMesh::Print`).
//!
//! Why a tree (D229/D231): the stateless `general_refinement_quad` reproduces
//! MFEM's element/node *counts*, but (a) anisotropic refinement was not wired
//! (`-a` refined iso), and (b) MFEM saves NC-refined meshes as the full tree
//! (`elements` with ref_type + children, `vertex_parents`, `root_state`,
//! `coordinates`), which cannot be reconstructed from the flat leaf mesh.
//! This module carries the tree so that
//!
//! * refinement batches `&[(leaf, ref_type)]` split elements exactly like
//!   `NCMesh::RefineElement` (ncmesh.cpp ~1816: SQUARE `ref_type &= 0x3`;
//!   X = 2 children over midpoints of the *horizontal* edges (n0,n1)/(n2,n3),
//!   Y = 2 children over the *vertical* edges (n1,n2)/(n3,n0), XY = the iso
//!   4-split with the diagonal center `GetId(mid01, mid23)`),
//! * the `Iso` flag flips to `false` on the first X/Y split and then
//!   `GetLimitRefinements` emits directional ref_types
//!   (`splits[0] = max(elevel[0], elevel[2])` → bit X,
//!   `splits[1] = max(elevel[1], elevel[3])` → bit Y; while `Iso` is true it
//!   forces ref_type 7 = iso, ncmesh.cpp:6060),
//! * node/element ids are assigned in exactly MFEM's creation order, and
//! * `print_mfem_nc_v10` reproduces `NCMesh::Print` byte for byte
//!   (straight 2-D meshes; coordinates as `ostream` with precision 8 like the
//!   toys' `mesh_ofs.precision(8)`).
//!
//! Boundary (face) attributes: MFEM keeps a `Face` per edge whose attribute
//! is inherited by the child faces of a split (the `fa[...]`/`eattr[...]`
//! arguments of `NewQuadrilateral`); brand-new interior edges get -1.  Here
//! `face_attr` maps an edge (min,max) to its boundary attribute, with 0 =
//! interior, and child edges inherit per the same masks.

use std::collections::HashMap;
use std::fmt::Write as _;

use fem_core::{NodeId, ElemId};
use crate::element_type::ElementType;
use crate::simplex::Mesh;

/// MFEM `quad_hilbert_child_order` (mesh/ncmesh_tables.hpp).
const HILBERT_CHILD_ORDER: [[u8; 4]; 8] = [
    [0, 1, 2, 3], [0, 3, 2, 1], [1, 2, 3, 0], [1, 0, 3, 2],
    [2, 3, 0, 1], [2, 1, 0, 3], [3, 0, 1, 2], [3, 2, 1, 0],
];

/// MFEM `quad_hilbert_child_state` (mesh/ncmesh_tables.hpp).
const HILBERT_CHILD_STATE: [[u8; 4]; 8] = [
    [1, 0, 0, 5], [0, 1, 1, 4], [3, 2, 2, 7], [2, 3, 3, 6],
    [5, 4, 4, 1], [4, 5, 5, 0], [7, 6, 6, 3], [6, 7, 7, 2],
];

/// Refinement type bits (MFEM `Refinement::X/Y/Z`, squares use X|Y).
pub const REF_X: u8 = 1;
pub const REF_Y: u8 = 2;
/// Isotropic 4-split (MFEM `Refinement::XY`); bit 2 (Z) is ignored for
/// squares (`ref_type &= 0x3`), so 7 is an alias of 3.
pub const REF_XY: u8 = 3;

/// One tree element (MFEM `NCMesh::Element`, squares only).
#[derive(Debug, Clone)]
struct NcElem {
    /// Material/attribute tag.
    attr: i32,
    /// 0 = leaf; otherwise a bit mask of X/Y splits (XY = 3).
    ref_type: u8,
    /// Corner nodes (valid while `ref_type == 0`).
    node: [NodeId; 4],
    /// Child element ids (valid while `ref_type != 0`), -1 = unused.
    child: [i32; 4],
}

/// Nonconforming quad tree — 1:1 port of MFEM `NCMesh` (2-D squares).
///
/// Leaf indices used by the public API are indices into `leaf_elements`
/// (MFEM's Mesh element numbering); tree element ids are internal.
pub struct NcQuadTree {
    /// Tree elements in creation order (id = index).
    elems: Vec<NcElem>,
    /// Per node: its two parents `(p1, p2)` with `p1 <= p2`; top-level nodes
    /// satisfy `p1 == p2 == id` (MFEM `nodes.Alloc(id, id, id)`).
    node_parents: Vec<(NodeId, NodeId)>,
    /// Per node coordinates (top-level nodes keep the mesh coordinates;
    /// midpoints are exact `0.5·(a+b)` averages).
    coords: Vec<[f64; 2]>,
    /// `(p1, p2)` → midpoint node (MFEM `HashTable::GetId` hash).
    edge_mid: HashMap<(NodeId, NodeId), NodeId>,
    /// Edge → boundary attribute (0 = interior); child edges inherit the
    /// parent edge's attribute through the `NewQuadrilateral` masks.
    face_attr: HashMap<(NodeId, NodeId), i32>,
    /// Hilbert state of each root element (MFEM `InitRootState`).
    root_state: Vec<u8>,
    /// Leaf elements in MFEM Mesh order (Hilbert SFC after the first update).
    leaf_elements: Vec<usize>,
    /// MFEM `NCMesh::Iso`: true until the first anisotropic (X or Y) split.
    iso: bool,
    /// Number of top-level (root) vertices = coordinates-section size.
    n_root_vertices: usize,
}

impl NcQuadTree {
    /// Wrap a straight Quad4 mesh: its elements become the tree roots (MFEM
    /// `NCMesh::NCMesh(const Mesh*)`), its vertices the top-level nodes, and
    /// its boundary faces define the boundary face attributes.
    pub fn from_mesh(mesh: &Mesh<2>) -> Self {
        assert!(
            mesh.elem_type == ElementType::Quad4 && mesh.elem_types.is_none(),
            "NcQuadTree::from_mesh: only uniform Quad4 meshes are supported"
        );

        let n_root_vertices = mesh.n_nodes();
        let coords: Vec<[f64; 2]> =
            (0..n_root_vertices).map(|n| mesh.coords_of(n as NodeId)).collect();
        let node_parents: Vec<(NodeId, NodeId)> =
            (0..n_root_vertices as NodeId).map(|i| (i, i)).collect();

        let mut elems = Vec::with_capacity(mesh.n_elems());
        for e in 0..mesh.n_elems() as ElemId {
            let ns = mesh.elem_nodes(e);
            elems.push(NcElem {
                attr: mesh.elem_tags[e as usize],
                ref_type: 0,
                node: [ns[0], ns[1], ns[2], ns[3]],
                child: [-1; 4],
            });
        }

        // Boundary face attributes (MFEM: face->attribute = be->GetAttribute()).
        let mut face_attr: HashMap<(NodeId, NodeId), i32> = HashMap::new();
        for (f, &tag) in mesh.face_tags.iter().enumerate() {
            let a = mesh.face_conn[2 * f] as NodeId;
            let b = mesh.face_conn[2 * f + 1] as NodeId;
            face_attr.insert((a.min(b), a.max(b)), tag);
        }

        let mut tree = NcQuadTree {
            elems,
            node_parents,
            coords,
            edge_mid: HashMap::new(),
            face_attr,
            root_state: Vec::new(),
            leaf_elements: Vec::new(),
            iso: true,
            n_root_vertices,
        };
        // MFEM `NCMesh::NCMesh(const Mesh*)` (ncmesh.cpp:178): ReferenceElement
        // every root element.  Besides the vertex refcounts this **creates a
        // midpoint node for every element edge** (`nodes.Get(a,b)`, element ×
        // local-edge order) — these nodes occupy the id space right after the
        // top-level vertices and are what makes later `GetId` calls return
        // existing ids.  They are hanging-node candidates only (vert_refc = 0
        // until an edge split makes them corners), so they never show up in
        // the `coordinates`/`vertex_parents` sections.
        let roots = tree.elems.len();
        for i in 0..roots {
            tree.reference_element(i);
        }
        tree.root_state = tree.init_root_states();
        tree.update();
        tree
    }

    /// MFEM `NCMesh::InitRootState` (ncmesh.cpp:2654) for squares: pick each
    /// root's Hilbert state so the curve enters at the previous root's exit
    /// node and exits through a node shared with the next root.
    fn init_root_states(&self) -> Vec<u8> {
        let n_roots = self.elems.len();
        let mut states = vec![0u8; n_roots];
        let mut entry_node: i64 = -2;
        for i in 0..n_roots {
            let el = &self.elems[i];
            let v_in = if entry_node >= 0 {
                el.node.iter().position(|&n| n as i64 == entry_node)
            } else {
                None
            };
            let v_in = v_in.unwrap_or(0);

            // Which nodes are shared with the next root.
            let mut shared = [false; 4];
            if i + 1 < n_roots {
                let next = self.elems[i + 1].node;
                for &n in &next {
                    if let Some(p) = el.node.iter().position(|&m| m == n) {
                        shared[p] = true;
                    }
                }
            }

            let mut state = 2 * v_in; // Dim * v_in
            for j in 0..2usize {
                let exit = HILBERT_CHILD_ORDER[state + j][3] as usize;
                if shared[exit] {
                    state += j;
                    break;
                }
            }
            states[i] = state as u8;
            entry_node = el.node[HILBERT_CHILD_ORDER[state][3] as usize] as i64;
        }
        states
    }

    /// MFEM `HashTable::GetId(p1, p2)`: return the midpoint node of the edge
    /// `(a, b)`, creating it (with the exact midpoint coordinates) on first
    /// use.  Nodes are numbered in creation order like MFEM's HashTable.
    fn get_id(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let key = (a.min(b), a.max(b));
        if let Some(&m) = self.edge_mid.get(&key) {
            return m;
        }
        let id = self.coords.len() as NodeId;
        let ca = self.coords[a as usize];
        let cb = self.coords[b as usize];
        self.coords.push([0.5 * (ca[0] + cb[0]), 0.5 * (ca[1] + cb[1])]);
        self.node_parents.push(key);
        self.edge_mid.insert(key, id);
        id
    }

    /// Boundary attribute of an element edge (0 = interior).
    fn face_attr_of(&self, a: NodeId, b: NodeId) -> i32 {
        self.face_attr.get(&(a.min(b), a.max(b))).copied().unwrap_or(0)
    }

    /// Register a child element and its face attributes (MFEM
    /// `NewQuadrilateral` + the `eattr[...]` masks of `RefineElement`).
    fn add_quad(&mut self, node: [NodeId; 4], attr: i32, eattr: [i32; 4]) -> usize {
        let id = self.elems.len();
        for k in 0..4 {
            let (a, b) = (node[k], node[(k + 1) % 4]);
            self.face_attr.insert((a.min(b), a.max(b)), eattr[k]);
        }
        self.elems.push(NcElem { attr, ref_type: 0, node, child: [-1; 4] });
        id
    }

    /// MFEM `NCMesh::ReferenceElement` (ncmesh.cpp:367): bump the vertex
    /// refcounts and reference (creating, if missing, via `HashTable::Get`)
    /// the midpoint node of every edge — in local edge order
    /// `(0,1), (1,2), (2,3), (3,0)`.  The created nodes make the id space
    /// dense exactly like MFEM's; in a refine-only workflow a node's refcounts
    /// never drop to zero (an edge either survives — its mid keeps
    /// `edge_refc > 0` — or splits — its mid becomes a corner — so nothing is
    /// ever deleted and the `unused` id-recycling list stays empty), which is
    /// why plain creation here reproduces MFEM's numbering.
    fn reference_element(&mut self, elem: usize) {
        let n = self.elems[elem].node;
        for k in 0..4 {
            self.get_id(n[k], n[(k + 1) % 4]);
        }
    }

    /// MFEM `NCMesh::RefineElement` (ncmesh.cpp:1129), SQUARE branch
    /// (ncmesh.cpp:1816): `ref_type &= 0x3`, then the X / Y / XY splits with
    /// MFEM's child order and midpoint creation order.
    fn refine_element(&mut self, elem: usize, ref_type: u8) {
        let ref_type = ref_type & 0x3;
        if ref_type == 0 {
            return;
        }
        if self.elems[elem].ref_type != 0 {
            // Element refined already (duplicate/forced reference in a batch):
            // apply the remaining splits to the children (ncmesh.cpp:1146).
            let remaining = ref_type & !self.elems[elem].ref_type;
            if remaining != 0 {
                let children = self.elems[elem].child;
                for ch in children {
                    if ch >= 0 {
                        self.refine_element(ch as usize, remaining);
                    }
                }
            }
            return;
        }

        let el = &self.elems[elem];
        let n = el.node;
        let attr = el.attr;
        // Parent face attributes (ncmesh.cpp:1172: fa[i] = face->attribute).
        let fa0 = self.face_attr_of(n[0], n[1]);
        let fa1 = self.face_attr_of(n[1], n[2]);
        let fa2 = self.face_attr_of(n[2], n[3]);
        let fa3 = self.face_attr_of(n[3], n[0]);

        let mut child = [-1i32; 4];
        match ref_type {
            REF_X => {
                // X split (ncmesh.cpp:1820): midpoints of the two horizontal
                // edges; children = left/right halves.
                let mid01 = self.get_id(n[0], n[1]);
                let mid23 = self.get_id(n[2], n[3]);
                child[0] = self.add_quad(
                    [n[0], mid01, mid23, n[3]], attr, [fa0, 0, fa2, fa3],
                ) as i32;
                child[1] = self.add_quad(
                    [mid01, n[1], n[2], mid23], attr, [fa0, fa1, fa2, 0],
                ) as i32;
                self.iso = false;
            }
            REF_Y => {
                // Y split (ncmesh.cpp:1834): midpoints of the two vertical
                // edges; children = bottom/top halves.
                let mid12 = self.get_id(n[1], n[2]);
                let mid30 = self.get_id(n[3], n[0]);
                child[0] = self.add_quad(
                    [n[0], n[1], mid12, mid30], attr, [fa0, fa1, 0, fa3],
                ) as i32;
                child[1] = self.add_quad(
                    [mid30, mid12, n[2], n[3]], attr, [0, fa1, fa2, fa3],
                ) as i32;
                self.iso = false;
            }
            _ => {
                // XY iso split (ncmesh.cpp:1848); center = midpoint of the
                // (mid01, mid23) diagonal.
                let mid01 = self.get_id(n[0], n[1]);
                let mid12 = self.get_id(n[1], n[2]);
                let mid23 = self.get_id(n[2], n[3]);
                let mid30 = self.get_id(n[3], n[0]);
                let midel = self.get_id(mid01, mid23);
                child[0] = self.add_quad(
                    [n[0], mid01, midel, mid30], attr, [fa0, 0, 0, fa3],
                ) as i32;
                child[1] = self.add_quad(
                    [mid01, n[1], mid12, midel], attr, [fa0, fa1, 0, 0],
                ) as i32;
                child[2] = self.add_quad(
                    [midel, mid12, n[2], mid23], attr, [0, fa1, fa2, 0],
                ) as i32;
                child[3] = self.add_quad(
                    [mid30, midel, mid23, n[3]], attr, [0, 0, fa2, fa3],
                ) as i32;
                // ref_type == XY keeps `Iso` unchanged (ncmesh.cpp:1882).
            }
        }

        // MFEM: children first (`NewQuadrilateral` — creates no nodes), then
        // `ReferenceElement(child)` for each child in order (ncmesh.cpp:1911);
        // this creates the child edges' midpoint nodes (edge_refc bookkeeping).
        for ch in child {
            if ch >= 0 {
                self.reference_element(ch as usize);
            }
        }

        let el = &mut self.elems[elem];
        el.ref_type = ref_type;
        el.child = child;
    }

    /// MFEM `NCMesh::Refine`: refine the given leaves `(leaf index, ref_type)`
    /// in array order (MFEM pushes the batch on a LIFO stack in reverse, so it
    /// processes it in order), then rebuild the leaf/SFC structures.
    pub fn refine(&mut self, refinements: &[(usize, u8)]) {
        for &(leaf, ref_type) in refinements {
            let elem = self.leaf_elements[leaf];
            self.refine_element(elem, ref_type);
        }
        self.update();
    }

    /// Rebuild `leaf_elements` in MFEM's Hilbert SFC order
    /// (`CollectLeafElements`, ncmesh.cpp:2380) and the vertex-use flags
    /// (`Node::HasVertex` = the node is a corner of some leaf).
    fn update(&mut self) {
        // Roots are the first `root_state.len()` elements of `elems`.
        let mut leaves = Vec::new();
        for root in 0..self.root_state.len() {
            self.collect_leaves(root, self.root_state[root], &mut leaves);
        }
        self.leaf_elements = leaves;
    }

    fn collect_leaves(&self, ei: usize, state: u8, out: &mut Vec<usize>) {
        let el = &self.elems[ei];
        if el.ref_type == 0 {
            out.push(ei);
            return;
        }
        if el.ref_type == REF_XY {
            // Squares with an iso split follow the Hilbert curve.
            for k in 0..4 {
                let ch = HILBERT_CHILD_ORDER[state as usize][k] as usize;
                let st = HILBERT_CHILD_STATE[state as usize][k];
                let ci = el.child[ch];
                if ci >= 0 {
                    self.collect_leaves(ci as usize, st, out);
                }
            }
        } else {
            // No SFC tables for aniso splits: natural child order, same state.
            for &ci in &el.child {
                if ci >= 0 {
                    self.collect_leaves(ci as usize, state, out);
                }
            }
        }
    }

    /// `Node::HasVertex` flags: which nodes are corners of some leaf element.
    fn leaf_vertices(&self) -> Vec<bool> {
        let mut is_vtx = vec![false; self.node_parents.len()];
        for &ei in &self.leaf_elements {
            for &n in &self.elems[ei].node {
                is_vtx[n as usize] = true;
            }
        }
        is_vtx
    }

    /// MFEM `NCMesh::EdgeSplitLevel` (ncmesh.cpp:5913): depth of the midpoint
    /// chain along `(a, b)`, counting only midpoints that are leaf vertices.
    fn edge_split_level(&self, a: NodeId, b: NodeId, is_vtx: &[bool]) -> u32 {
        let key = (a.min(b), a.max(b));
        match self.edge_mid.get(&key) {
            Some(&m) if is_vtx[m as usize] => {
                1 + self.edge_split_level(a, m, is_vtx)
                    .max(self.edge_split_level(m, b, is_vtx))
            }
            _ => 0,
        }
    }

    /// MFEM `NCMesh::CountSplits` (ncmesh.cpp:6044) for squares:
    /// `(splits[0], splits[1])` = max level over the horizontal / vertical
    /// edges (`edges[0] = (0,1)`, `edges[1] = (1,2)`, `edges[2] = (2,3)`,
    /// `edges[3] = (3,0)` of the Quadrilateral).
    fn count_splits(&self, ei: usize, is_vtx: &[bool]) -> (u32, u32) {
        let n = self.elems[ei].node;
        let e0 = self.edge_split_level(n[0], n[1], is_vtx);
        let e1 = self.edge_split_level(n[1], n[2], is_vtx);
        let e2 = self.edge_split_level(n[2], n[3], is_vtx);
        let e3 = self.edge_split_level(n[3], n[0], is_vtx);
        (e0.max(e2), e1.max(e3))
    }

    /// MFEM `NCMesh::GetLimitRefinements` (ncmesh.cpp:6060): every leaf whose
    /// directional split level exceeds `max_level` is queued with the
    /// corresponding ref_type bit; while the mesh is still `Iso` the type is
    /// forced to 7 (= iso for squares).
    pub fn get_limit_refinements(&self, max_level: u32) -> Vec<(usize, u8)> {
        let is_vtx = self.leaf_vertices();
        let mut out = Vec::new();
        for (li, &ei) in self.leaf_elements.iter().enumerate() {
            let (s0, s1) = self.count_splits(ei, &is_vtx);
            let mut ref_type = 0u8;
            if s0 > max_level { ref_type |= REF_X; }
            if s1 > max_level { ref_type |= REF_Y; }
            if ref_type != 0 {
                if self.iso {
                    // iso meshes should only be modified by iso refinements
                    ref_type = 7;
                }
                out.push((li, ref_type));
            }
        }
        out
    }

    /// MFEM `NCMesh::LimitNCLevel` (ncmesh.cpp:6090): refine the violating
    /// leaves until a pass finds none.
    pub fn limit_nc_level(&mut self, max_nc_level: u32) {
        loop {
            let refinements = self.get_limit_refinements(max_nc_level);
            if refinements.is_empty() {
                break;
            }
            self.refine(&refinements);
        }
    }

    /// MFEM `Mesh::NonconformingRefinement` (mesh.cpp:11330): one refinement
    /// batch followed by `LimitNCLevel(nc_limit)` (skipped for `nc_limit == 0`).
    /// An empty batch leaves the mesh unchanged (MFEM `last_operation = NONE`).
    pub fn general_refinement(&mut self, refinements: &[(usize, u8)], nc_limit: u32) {
        if refinements.is_empty() {
            return;
        }
        self.refine(refinements);
        if nc_limit > 0 {
            self.limit_nc_level(nc_limit);
        }
    }

    /// Number of leaf elements (MFEM `Mesh::GetNE` after the refinement).
    pub fn leaf_count(&self) -> usize {
        self.leaf_elements.len()
    }

    /// Number of leaf-corner nodes (MFEM `Mesh::GetNV` — nodes with
    /// `vert_refc > 0`; the never-split construction edge midpoints and
    /// deeper hanging candidates are not counted).
    pub fn leaf_vertex_count(&self) -> usize {
        self.leaf_vertices().iter().filter(|&&v| v).count()
    }

    /// Total number of nodes ever created (MFEM node ids are 0-based).
    pub fn n_nodes(&self) -> usize {
        self.node_parents.len()
    }

    /// The `Iso` flag (true while only iso refinements have been applied).
    pub fn is_iso(&self) -> bool {
        self.iso
    }

    /// MFEM `NCMesh::SetAttribute`: set the attribute of leaf `i` (a Mesh
    /// element index).
    pub fn set_leaf_attribute(&mut self, i: usize, attr: i32) {
        let ei = self.leaf_elements[i];
        self.elems[ei].attr = attr;
    }

    /// Extract the leaf mesh in MFEM's Mesh element order (Hilbert SFC):
    /// all nodes (hanging nodes included), leaf quad connectivity, and the
    /// boundary segments of the leaves (SFC order, local face order).
    pub fn extract_mesh(&self) -> Mesh<2> {
        let mut conn = Vec::with_capacity(self.leaf_elements.len() * 4);
        let mut tags = Vec::with_capacity(self.leaf_elements.len());
        let mut face_conn = Vec::new();
        let mut face_tags = Vec::new();
        for &ei in &self.leaf_elements {
            let el = &self.elems[ei];
            conn.extend_from_slice(&el.node);
            tags.push(el.attr);
            for k in 0..4 {
                let (a, b) = (el.node[k], el.node[(k + 1) % 4]);
                let attr = self.face_attr_of(a, b);
                if attr > 0 {
                    face_conn.extend_from_slice(&[a, b]);
                    face_tags.push(attr);
                }
            }
        }
        let mut coords = Vec::with_capacity(self.coords.len() * 2);
        for c in &self.coords {
            coords.push(c[0]);
            coords.push(c[1]);
        }
        Mesh::uniform(
            coords, conn, tags, ElementType::Quad4,
            face_conn, face_tags, ElementType::Line2,
        )
    }

    /// Serialize the "MFEM NC mesh v1.0" format — 1:1 port of
    /// `NCMesh::Print` (ncmesh.cpp:6349) + `Mesh::Printer`'s NC branch
    /// (mesh.cpp:12484, the `mfem_mesh_end` trailer), for straight 2-D
    /// meshes without scaling (`using_scaling = false`, serial `rank = 0`).
    ///
    /// Sections: `elements` (the whole tree: rank/attr/geom/ref_type +
    /// nodes-or-children, creation order), `boundary` (leaves in creation
    /// order, `attr geom v0 v1`), `vertex_parents` (nodes with `p1 != p2`
    /// that are leaf vertices, id order), `root_state` (only if any state is
    /// nonzero) and `coordinates` (top-level vertex coordinates).
    ///
    /// Coordinates are printed like `ostream` with precision 8 (the toys'
    /// `mesh_ofs.precision(8)`); for the dyadic values of straight Cartesian
    /// refinements Rust's shortest-roundtrip `Display` matches C++ exactly.
    pub fn print_mfem_nc_v10(&self) -> String {
        let is_vtx = self.leaf_vertices();

        let mut s = String::new();
        s.push_str("MFEM NC mesh v1.0\n\n");
        s.push_str(
            "# NCMesh supported geometry types:\n\
             # SEGMENT     = 1\n\
             # TRIANGLE    = 2\n\
             # SQUARE      = 3\n\
             # TETRAHEDRON = 4\n\
             # CUBE        = 5\n\
             # PRISM       = 6\n\
             # PYRAMID     = 7\n",
        );
        let _ = writeln!(s, "\ndimension\n2");

        // ── elements: the whole tree in creation order ──
        let _ = writeln!(
            s, "\n# rank attr geom ref_type nodes/children\nelements\n{}",
            self.elems.len()
        );
        for el in &self.elems {
            // CollectLeafElements sets rank = -1 on non-leaves; serial leaves
            // are rank 0 (MyRank).
            let rank = if el.ref_type == 0 { 0 } else { -1 };
            let _ = write!(s, "{rank} {} 3 {}", el.attr, el.ref_type);
            if el.ref_type == 0 {
                for &n in &el.node {
                    let _ = write!(s, " {n}");
                }
            } else {
                for &ch in &el.child {
                    if ch >= 0 {
                        let _ = write!(s, " {ch}");
                    }
                }
            }
            s.push('\n');
        }

        // ── boundary: leaf faces with attribute > 0, creation order ──
        // (MFEM `PrintBoundary`, ncmesh.cpp:6222; 2-D faces print their two
        // edge endpoints, `deg = 2`.)
        let mut bdr: Vec<(i32, NodeId, NodeId)> = Vec::new();
        for el in &self.elems {
            if el.ref_type != 0 {
                continue;
            }
            for k in 0..4 {
                let (a, b) = (el.node[k], el.node[(k + 1) % 4]);
                let attr = self.face_attr_of(a, b);
                if attr > 0 {
                    bdr.push((attr, a, b));
                }
            }
        }
        if !bdr.is_empty() {
            let _ = writeln!(
                s, "\n# attr geom nodes\nboundary\n{}", bdr.len()
            );
            for &(attr, a, b) in &bdr {
                let _ = writeln!(s, "{attr} 1 {a} {b}");
            }
        }

        // ── vertex_parents: nodes with p1 != p2 that are leaf vertices ──
        // (MFEM `PrintVertexParents`, ncmesh.cpp:6106; HashTable iteration =
        // node-id order.)
        let nvp = (0..self.node_parents.len())
            .filter(|&id| {
                let (p1, p2) = self.node_parents[id];
                p1 != p2 && is_vtx[id]
            })
            .count();
        if nvp > 0 {
            let _ = writeln!(
                s, "\n# vert_id p1 p2\nvertex_parents\n{nvp}"
            );
            for id in 0..self.node_parents.len() {
                let (p1, p2) = self.node_parents[id];
                if p1 != p2 && is_vtx[id] {
                    let _ = writeln!(s, "{id} {p1} {p2}");
                }
            }
        }

        // ── root_state (optional, only when any state is nonzero) ──
        if !self.root_state.iter().all(|&st| st == 0) {
            let _ = writeln!(
                s, "\n# root element orientation\nroot_state\n{}",
                self.root_state.len()
            );
            for &st in &self.root_state {
                let _ = writeln!(s, "{st}");
            }
        }

        // ── coordinates: top-level vertices (MFEM `PrintCoordinates`) ──
        let _ = writeln!(
            s, "\n# top-level node coordinates\ncoordinates\n{}\n2",
            self.n_root_vertices
        );
        for c in &self.coords[..self.n_root_vertices] {
            let _ = writeln!(s, "{} {}", disp_f64(c[0]), disp_f64(c[1]));
        }

        s.push_str("\nmfem_mesh_end\n");
        s
    }
}

/// Format a coordinate like C++ `ostream <<` with precision 8 (general
/// format) on the dyadic values of straight Cartesian refinements: both
/// print the exact shortest decimal ("0", "0.25", "0.03125", "1").  (C++
/// switches to exponential notation below 1e-4 — unreachable for root
/// vertices of unit-grid refinements, which are the only values printed.)
fn disp_f64(x: f64) -> String {
    format!("{x}")
}
