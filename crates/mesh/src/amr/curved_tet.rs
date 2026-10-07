//! Curved (high-order) geometry support for uniform Tet4 refinement — the
//! tetrahedron analogue of [`super::curved_hex`], [`super::curved_prism`] and
//! [`super::curved_tri`].
//!
//! MFEM refines a curved 3-D mesh (`nodes` grid function present) in two steps
//! (`Mesh::UniformRefinement3D_base` + `UpdateNodes`):
//!
//! 1. the straight `vertices` array is averaged exactly like a linear mesh —
//!    for a tet this means the **6 edge midpoints and nothing else** (MFEM's
//!    8-child tet split uses no face centers and no body center, so a pure-tet
//!    mesh gains exactly one vertex per mesh edge, `oedge + e2v[E]`);
//! 2. because `update_nodes` is set for curved meshes, the refined `nodes`
//!    grid function replaces the vertex positions: for every *fine* element
//!    the child's embedding (`CoarseFineTransformations::point_matrices`
//!    `tet_children`, an affine map `parent_ref = origin + M·child_ref` with
//!    no mirroring) is applied to the child's dof reference points and the
//!    coarse element's own FE is evaluated there, so the fine geometry is the
//!    **per-child evaluation of the parent's nodal field**, with dofs shared
//!    between elements meeting on the same fine entity (MFEM's first-touch
//!    `mark` rule).
//!
//! The positional layout of the `(p+1)(p+2)(p+3)/6` geometry dofs is the
//! element's own table ([`H1TetPk`] — MFEM `H1_TetrahedronElement`'s
//! Gauss-Lobatto lattice in MFEM's slot order: vertices, the six edges, the
//! four faces, the interior).  That is the same table `Mesh::set_curvature`'s
//! tet path writes, `fem_io`'s `nodes` reader keys, and
//! `Mesh::element_jacobian`'s tet arm evaluates, so element, numbering and
//! this module cannot drift apart.
//!
//! Two things set the tet apart from the hex/wedge ports:
//!
//! * the interior children depend on the **refinement type `rt`** (MFEM's
//!   best-aspect-ratio octahedron split, `tet_select_rt_debug`): embedding
//!   matrix `4·(rt+1)+k` is the `k`-th interior child of type `rt`, while
//!   matrices `0..4` are the four corner children.  Under uniform refinement
//!   every parent's fine element order is corner 0..3 then the four interior
//!   children — the historical fem-rs tet order *is* MFEM's, so (unlike the
//!   wedge) no child-order switch is needed;
//! * MFEM's new-vertex ids are **not** first-touch: `UniformRefinement3D_base`
//!   re-maps the global edge ids through `e2v` (the `GetVertexToVertexTable`
//!   edges re-sorted per vertex row), so the fine vertex of an edge sits at
//!   `oedge + e2v[E]` (`MfemTetRefineIds`, called from `amr_inner`).  As with
//!   the hex/wedge, that numbering is used only where a written file pins it
//!   down: geometry present **and** uniform refinement.
//!
//! **Order-1 discontinuous geometry** (`L2_T1_*_P1` — the per-element table
//! `Mesh::SetCurvature(1, true)` writes and the folded encoding of periodic
//! tet meshes) is the degenerate end of the same mechanism (D816-2): the
//! refined table stays *fully discontinuous* (every fine element owns its own
//! 4 corner dofs — [`build_refined_l2_p1_tet_geometry`]), and the refined
//! `vertices` are replaced wholesale by the mean over the element references
//! of those folded values ([`super::curved_hex::set_vertices_from_nodes`]).

use std::collections::HashMap;

use fem_core::{ElemId, NodeId};
use fem_element::lagrange::H1TetPk;
use fem_element::ReferenceElement;

use crate::element_type::ElementType;
use crate::simplex::{GeometryData, Mesh};

/// Reference coordinates of the four vertices of a Tet4 in MFEM
/// `Geometry::Constants<TETRAHEDRON>` order: `0=(0,0,0) 1=(1,0,0) 2=(0,1,0)
/// 3=(0,0,1)`.
const VERTS: [[f64; 3]; 4] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Local edges of a tet as `(from, to)` vertex pairs, in MFEM
/// `Geometry::Constants<TETRAHEDRON>::Edges` order (the same order
/// `amr_inner::local_edges_tet` enumerates).
const EDGES: [(usize, usize); 6] =
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// Index of the edge joining local vertices `a < b` in [`EDGES`].
const EDGE_INDEX: [[usize; 4]; 4] = [
    // [a][b] for a < b; unused slots (a >= b) are 0 and never read.
    [0, 0, 1, 2],
    [0, 0, 3, 4],
    [1, 3, 0, 5],
    [2, 4, 5, 0],
];

/// Local faces of a tet as vertex triples, in MFEM
/// `Geometry::Constants<TETRAHEDRON>::FaceVert` order: face `f` is the
/// triangle **opposite local vertex `f`** (`{1,2,3}, {0,3,2}, {0,1,3},
/// {0,2,1}` — the order `Mesh::GenerateFaces` and `H1TetPk`'s face blocks
/// use).
const FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];

/// MFEM `UniformRefinement3D_base`'s `mv_all`: the four interior children of
/// each refinement type `rt`, as `EDGES` indices (`0..6` = the six edge
/// midpoints `e0..e5`).
const MV_ALL: [[[usize; 4]; 4]; 3] = [
    [[0, 5, 1, 2], [0, 5, 2, 4], [0, 5, 4, 3], [0, 5, 3, 1]], // rt = 0
    [[1, 0, 4, 2], [1, 2, 4, 5], [1, 5, 4, 3], [1, 3, 4, 0]], // rt = 1
    [[2, 0, 1, 3], [2, 1, 5, 3], [2, 5, 4, 3], [2, 4, 0, 3]], // rt = 2
];

/// The reference points of embedding `matrix` — the images of the child's
/// reference-tet vertices `0..4` in the parent's reference tet (MFEM
/// `tet_children`, `mesh/mesh.cpp`).  Matrices `0..4` are the corner children
/// (child `v` spans vertex `v` and the midpoints of its three edges, the
/// vertices in ascending partner order); matrix `4·(rt+1)+k` is the `k`-th
/// interior child of refinement type `rt` (`mv_all[rt][k]`).
pub(crate) fn child_points(matrix: usize) -> [[f64; 3]; 4] {
    let mid = |ei: usize| {
        let (a, b) = EDGES[ei];
        [
            0.5 * (VERTS[a][0] + VERTS[b][0]),
            0.5 * (VERTS[a][1] + VERTS[b][1]),
            0.5 * (VERTS[a][2] + VERTS[b][2]),
        ]
    };
    if matrix < 4 {
        let v = matrix;
        let mut pts = [[0.0_f64; 3]; 4];
        for (j, p) in pts.iter_mut().enumerate() {
            *p = if j == v {
                VERTS[v]
            } else {
                let (a, b) = (v.min(j), v.max(j));
                mid(EDGE_INDEX[a][b])
            };
        }
        pts
    } else {
        let rt = matrix / 4 - 1;
        let k = matrix % 4;
        MV_ALL[rt][k].map(mid)
    }
}

/// Affine map `parent_ref = origin + M·child_ref` of embedding `matrix`.
fn child_map(matrix: usize) -> ([f64; 3], [[f64; 3]; 3]) {
    let pts = child_points(matrix);
    let origin = pts[0];
    let mut m = [[0.0_f64; 3]; 3];
    for (k, o) in origin.iter().enumerate() {
        m[k][0] = pts[1][k] - o;
        m[k][1] = pts[2][k] - o;
        m[k][2] = pts[3][k] - o;
    }
    (origin, m)
}

/// Child index of an **unrefined** element copied through unchanged (its fine
/// element *is* the parent element: the reference map is the identity and
/// every evaluation lands exactly on a parent dof point).
pub(crate) const IDENTITY: u8 = 16;

/// MFEM `Geometries.GetCenter(GEOMETRY::TETRAHEDRON)` — the reference point
/// whose element-Jacobian `UniformRefinement3D_base` feeds the refinement-type
/// selection (`mesh/mesh.cpp`, `rt_algo = 1`).
pub(crate) const TET_CENTER: [f64; 3] = [0.25, 0.25, 0.25];

/// Read-only view of an order-`p` (`p ≥ 2`) tet [`GeometryData`] attached to a
/// Tet4 mesh (as built by `Mesh::set_curvature`'s tet path or `fem_io`'s MFEM
/// `nodes` reader — both place the dofs in [`H1TetPk`]'s slot order).
pub(crate) struct TetPkGeometry<'a> {
    geo: &'a GeometryData,
    fe: H1TetPk,
    /// Geometry order `p`.
    pub(crate) order: usize,
    /// Geometry dofs per element, `(p+1)(p+2)(p+3)/6` (`GeometryData::nodes_per_elem`).
    pub(crate) dpe: usize,
}

impl<'a> TetPkGeometry<'a> {
    /// Returns `Some` iff `mesh` is a Tet4 mesh carrying order-`p ≥ 2`
    /// geometry with `(p+1)(p+2)(p+3)/6` dofs per element in [`H1TetPk`]'s
    /// slot order.
    pub(crate) fn new(mesh: &'a Mesh<3>) -> Option<Self> {
        if mesh.elem_type != ElementType::Tet4 {
            return None;
        }
        let geo = mesh.geometry.as_ref()?;
        let order = geo.order as usize;
        if order < 2 {
            // Order 1 (or absent) geometry: the straight averaging the
            // refinement kernels already do *is* the interpolation.
            return None;
        }
        let dpe = (order + 1) * (order + 2) * (order + 3) / 6;
        if geo.nodes_per_elem != dpe || geo.conn.len() < mesh.n_elems() * dpe {
            return None;
        }
        let fe = H1TetPk::new(order);
        if fe.n_dofs() != dpe {
            return None;
        }
        Some(TetPkGeometry { geo, fe, order, dpe })
    }

    fn elem_dofs(&self, e: ElemId) -> &[NodeId] {
        let o = e as usize * self.dpe;
        &self.geo.conn[o..o + self.dpe]
    }

    /// Evaluate the parent element's order-`p` geometry field at the reference
    /// tet point `xi` — the interpolation of the element's geometry dofs with
    /// the element's own nodal basis (exact δ at the dof points, so every
    /// evaluation at a parent dof reads the stored value back bit for bit).
    pub(crate) fn eval_at(&self, e: ElemId, xi: [f64; 3]) -> [f64; 3] {
        let mut w = vec![0.0_f64; self.dpe];
        self.fe.eval_basis(&xi, &mut w);
        let dofs = self.elem_dofs(e);
        let mut out = [0.0_f64; 3];
        for (s, &w) in w.iter().enumerate() {
            if w != 0.0 {
                let d = dofs[s] as usize;
                out[0] += w * self.geo.coords[3 * d];
                out[1] += w * self.geo.coords[3 * d + 1];
                out[2] += w * self.geo.coords[3 * d + 2];
            }
        }
        out
    }

    /// Coordinate of the new fine vertex at the midpoint of local edge `li`
    /// (index into [`EDGES`]): the parent geometry field evaluated at the edge
    /// midpoint.  For an order-2 parent that point *is* the parent's mid-edge
    /// geometry dof and the evaluation reads it back exactly (all basis
    /// weights are 0/1) — MFEM's `SetVerticesFromNodes` pick.
    pub(crate) fn edge_pick(&self, e: ElemId, li: usize) -> [f64; 3] {
        let (a, b) = EDGES[li];
        let (ra, rb) = (VERTS[a], VERTS[b]);
        self.eval_at(e, [0.5 * (ra[0] + rb[0]), 0.5 * (ra[1] + rb[1]), 0.5 * (ra[2] + rb[2])])
    }
}

/// The mesh's geometry as a **discontinuous order-1 tet** table — the
/// `L2_T1_*_P1` layout the MFEM reader produces for `Mesh::SetCurvature(1,
/// true)` on a tet mesh and for folded periodic tets: 4 dofs per element in
/// element-major rows of *fresh, unshared* dof ids, the row slot order being
/// the mesh's own reference-tet vertex order (D816-2).
///
/// A *continuous* order-1 table does not match: its dofs are the vertices
/// (shared ids, `n_nodes == n_vertices`), and the plain vertex averaging the
/// refinement kernels already do is its exact transport.
pub(crate) fn l2_p1_tet_geometry(mesh: &Mesh<3>) -> Option<&GeometryData> {
    if mesh.elem_type != ElementType::Tet4 {
        return None;
    }
    let geo = mesh.geometry.as_ref()?;
    if geo.order != 1 || geo.nodes_per_elem != 4 {
        return None;
    }
    let n = mesh.n_elems() * 4;
    if geo.conn.len() != n || geo.n_nodes != n {
        return None;
    }
    for (e, row) in geo.conn.chunks_exact(4).enumerate() {
        for (i, &d) in row.iter().enumerate() {
            if d as usize != e * 4 + i {
                return None; // shared dof: not the discontinuous layout
            }
        }
    }
    Some(geo)
}

/// The **periodic snapshot layout** of a straight (order-1) tet mesh:
/// `Mesh::make_periodic`'s per-element own-side geometry snapshot
/// (`simplex.rs::periodic_geometry_snapshot` — rows addressing the *pre-merge*
/// node table, with dof ids shared between same-side neighbours).  MFEM's
/// `MakePeriodic` materializes exactly this information as discontinuous
/// `L2_T1_3D_P1` nodes (`SetCurvature(order, /*discont=*/true)` before the
/// `v2v` renumbering), and refining it keeps the fully discontinuous own-side
/// layout (D840-1, MFEM 4.10 probe `mfem_tetx`: x-folded 2×2×2 tet cube
/// `ND = 48·4 = 192` → refined `ND = 384·4 = 1536`, every element
/// `det J = +1/64`) — so the snapshot rides the same
/// [`build_refined_l2_p1_tet_geometry`] transport, which reads the parent's
/// own rows through its `conn` and is layout-agnostic on the parent side.
///
/// Recognized **only** on a [periodically merged](curved_hex::is_periodic_merged_3d)
/// mesh: a continuous order-1 table (rows = the element corners themselves)
/// has every corner in its row, so the predicate excludes it, and the plain
/// vertex averaging those meshes use stays bit-for-bit untouched.
pub(crate) fn periodic_p1_tet_geometry(mesh: &Mesh<3>) -> Option<&GeometryData> {
    if !super::curved_hex::is_periodic_merged_3d(mesh) {
        return None;
    }
    let geo = mesh.geometry.as_ref()?;
    if geo.order != 1 || geo.nodes_per_elem != 4 {
        return None;
    }
    if geo.conn.len() != mesh.n_elems() * 4 {
        return None;
    }
    if geo.coords.len() < geo.n_nodes * 3 {
        return None;
    }
    if geo.conn.iter().any(|&d| d as usize >= geo.n_nodes) {
        return None;
    }
    Some(geo)
}

/// Build the refined mesh's **order-1 discontinuous** [`GeometryData`] from
/// the parent's `L2_T1_*_P1` tet table (D816-2).
///
/// MFEM refines the `nodes` grid function through its refinement operator:
/// for every fine element the coarse element's own shape functions are
/// evaluated at the fine dof reference points mapped into the parent frame
/// (`FiniteElement::GetLocalInterpolation`).  For a linear parent that is the
/// barycentric interpolation of the parent's 4 corner dofs at the child
/// embedding's reference points ([`child_points`] — the `tet_children` point
/// matrices; for an unrefined element ([`IDENTITY`]) the parent dofs are read
/// back exactly).  Because the space is discontinuous the fine table stays
/// fully discontinuous: fine element `fe` owns the fresh dof ids `4·fe .. 4·fe
/// + 3`, **no dof sharing across the fine mesh** — a folded shared face keeps
/// a different coordinate on each side.
///
/// Requires the fine elements to be emitted in MFEM's own child order with
/// their embedding matrix recorded in `fine_parent` — for
/// `refine_nonconforming_3d` that is the uniform case (`mfem_ids` built);
/// the historical straight-side partial order mirrors the corner children, so
/// its rows would not line up with the matrices.
pub(crate) fn build_refined_l2_p1_tet_geometry(
    parent_geo: &GeometryData,
    fine_parent: &[(ElemId, u8)],
) -> GeometryData {
    const DPE: usize = 4;
    let n_fine = fine_parent.len();
    let mut conn = Vec::with_capacity(n_fine * DPE);
    let mut coords = Vec::with_capacity(n_fine * DPE * 3);
    for (fe, &(pe, child)) in fine_parent.iter().enumerate() {
        // Parent-frame reference point of each fine dof slot: the child's own
        // reference vertices mapped through the embedding (the identity for an
        // unrefined element, whose points are the reference tet's vertices).
        let pts: [[f64; 3]; DPE] = if child == IDENTITY {
            VERTS
        } else {
            child_points(child as usize)
        };
        for (k, p) in pts.iter().enumerate() {
            // Barycentric weights of `p` — the parent's own P1 shape
            // functions, accumulated in parent slot order.
            let w = [1.0 - p[0] - p[1] - p[2], p[0], p[1], p[2]];
            let mut xyz = [0.0_f64; 3];
            for (j, &wj) in w.iter().enumerate() {
                let dof = parent_geo.conn[pe as usize * DPE + j] as usize;
                for c in 0..3 {
                    xyz[c] += wj * parent_geo.coords[dof * 3 + c];
                }
            }
            conn.push((fe * DPE + k) as NodeId);
            coords.extend_from_slice(&xyz);
        }
    }
    GeometryData {
        order: 1,
        nodes_per_elem: DPE,
        conn,
        n_nodes: n_fine * DPE,
        coords,
    }
}

/// One slot of the [`H1TetPk`] table.
#[derive(Clone, Copy, Debug)]
enum SlotKind {
    Vertex(usize),
    /// Edge `0..6` ([`EDGES`]), GLL position `m` (1..p) counted from the
    /// edge's first vertex.
    Edge { e: usize, m: usize },
    /// Face `0..4` ([`FACES`]), with the dof's GLL barycentric indices
    /// `(u0, u1, u2)` at the face's own three corners (`u0 + u1 + u2 = p`).
    Face { f: usize, w: [usize; 3] },
    /// Element interior (the slot's index, private per fine element).
    Interior,
}

/// The [`SlotKind`] of every [`H1TetPk`] slot of an order-`p` tet, read off
/// the slot's integer barycentric label `(λ₀..λ₃)` (`λq` is the weight of
/// local vertex `q`, so the label sums to `p`): one nonzero → vertex; two →
/// the edge between them, positioned from the lower vertex; three → the face
/// opposite the zero position, with the weights in the face's own corner
/// order; four → interior.
fn slot_kinds(p: usize) -> Vec<SlotKind> {
    let mut out = Vec::with_capacity((p + 1) * (p + 2) * (p + 3) / 6);
    for l in H1TetPk::slot_labels(p) {
        let nz: Vec<usize> = (0..4).filter(|&q| l[q] > 0).collect();
        out.push(match nz.len() {
            1 => SlotKind::Vertex(nz[0]),
            2 => {
                let (a, b) = (nz[0], nz[1]);
                SlotKind::Edge { e: EDGE_INDEX[a][b], m: l[b] }
            }
            3 => {
                let f = (0..4).find(|&q| l[q] == 0).expect("three nonzeros");
                SlotKind::Face { f, w: [l[FACES[f][0]], l[FACES[f][1]], l[FACES[f][2]]] }
            }
            _ => SlotKind::Interior,
        });
    }
    out
}

/// Key of a shared fine edge dof: the fine edge (its sorted node pair) plus
/// the dof's GLL index within the edge, measured from its lower-id node (both
/// sides speak in `1..p−1`; the GLL set is symmetric, so the dof at position
/// `m` from one end sits at `p − m` from the other).
type EdgeKey = (NodeId, NodeId, usize);
/// Key of a shared fine face dof: the fine face (sorted corner nodes) plus the
/// dof's GLL barycentric indices at the face's canonical corners (canonical
/// corner `q` = the `q`-th smallest node id).
type FaceKey = ([NodeId; 3], usize, usize, usize);

/// Build the refined mesh's [`GeometryData`] from the parent's order-`p` tet
/// geometry.
///
/// `fine` is the refined mesh (connectivity + final vertex coordinates, where
/// every new vertex already carries the parent-field coordinate computed by
/// the refinement kernels).  `fine_parent` maps each fine element to its
/// parent element and embedding matrix — MFEM's `tet_children` matrix index
/// (corner children `0..4`, interior `4·(rt+1)+k`) *in fine element order*
/// ([`IDENTITY`] for an unrefined element).
///
/// The fine geometry dof ids occupy a fresh index space: ids `0..n_fine_nodes`
/// are the fine mesh nodes (fine vertex coordinates), followed by the
/// edge/face/interior dofs created on first touch, keyed by the fine node sets
/// so elements sharing an entity share the dof.  Values are the parent
/// order-`p` field evaluated at the corresponding reference point — read back
/// exactly wherever the point coincides with a parent dof (in particular every
/// dof of an identity element) and the standard order-`p` interpolation
/// otherwise, i.e. MFEM's `GetLocalInterpolation` with its first-touch `mark`
/// rule.
pub(crate) fn build_refined_tet_geometry(
    parent: &Mesh<3>,
    fine: &Mesh<3>,
    fine_parent: &[(ElemId, u8)],
) -> Option<GeometryData> {
    let pq = TetPkGeometry::new(parent)?;
    let p = pq.order;
    let dpe = pq.dpe;
    debug_assert_eq!(fine.n_elems(), fine_parent.len());

    // Gauss-Lobatto points on [0,1] — the 1-D lattice the slot labels live on
    // (the same points `H1TetPk` places its dofs at); only the debug ground
    // check below needs them.
    #[cfg(debug_assertions)]
    let (g, _w) = fem_element::quadrature::gauss_lobatto_arbitrary(p + 1);
    #[cfg(debug_assertions)]
    let cp: Vec<f64> = g.iter().map(|&x| 0.5 * (x + 1.0)).collect();

    let fe_coords: Vec<[f64; 3]> =
        pq.fe.dof_coords().iter().map(|c| [c[0], c[1], c[2]]).collect();
    debug_assert_eq!(fe_coords.len(), dpe);
    let kinds = slot_kinds(p);
    debug_assert_eq!(kinds.len(), dpe);
    // Ground check (debug builds): the classification must place every slot at
    // the coordinates `H1TetPk` itself reports, or the entity keys below would
    // be built on a misread lattice.
    #[cfg(debug_assertions)]
    {
        let labels = H1TetPk::slot_labels(p);
        for (s, kind) in kinds.iter().enumerate() {
            let r = &fe_coords[s];
            let want: [f64; 3] = match *kind {
                SlotKind::Vertex(v) => VERTS[v],
                SlotKind::Edge { e, m } => {
                    let (a, b) = EDGES[e];
                    let (ra, rb) = (VERTS[a], VERTS[b]);
                    let t = cp[m];
                    [
                        ra[0] + t * (rb[0] - ra[0]),
                        ra[1] + t * (rb[1] - ra[1]),
                        ra[2] + t * (rb[2] - ra[2]),
                    ]
                }
                SlotKind::Face { f, w } => {
                    // Barycentric GLL point of the face triangle: the
                    // cp-weighted average of the face's corners, normalised by
                    // the weight sum (MFEM's `cp[i]/w` node placement).
                    let ws: f64 = w.iter().map(|&u| cp[u]).sum();
                    let mut c = [0.0_f64; 3];
                    for (q, &u) in w.iter().enumerate() {
                        for d in 0..3 {
                            c[d] += cp[u] * VERTS[FACES[f][q]][d];
                        }
                    }
                    [c[0] / ws, c[1] / ws, c[2] / ws]
                }
                SlotKind::Interior => {
                    // The barycentric formula `H1TetPk` itself places interior
                    // dofs with (`cp[i]/w` over the (x, y, z) weights).
                    let l = labels[s];
                    let ws: f64 = l.iter().map(|&u| cp[u]).sum();
                    [cp[l[1]] / ws, cp[l[2]] / ws, cp[l[3]] / ws]
                }
            };
            for d in 0..3 {
                assert!(
                    (r[d] - want[d]).abs() < 1e-12,
                    "tet slot {s} classified {kind:?} but sits at {r:?} (expected {want:?})"
                );
            }
        }
    }

    let n_fine_nodes = fine.n_nodes() as NodeId;
    let mut geo_coords: Vec<f64> = fine.coords.clone();
    let mut next_dof = n_fine_nodes;
    let mut edge_dofs: HashMap<EdgeKey, NodeId> = HashMap::new();
    let mut face_dofs: HashMap<FaceKey, NodeId> = HashMap::new();
    let mut vol_dofs: HashMap<(ElemId, usize), NodeId> = HashMap::new();

    let mut conn: Vec<NodeId> = Vec::with_capacity(fine.n_elems() * dpe);
    for (fe_i, &(pe, child)) in fine_parent.iter().enumerate() {
        let fe = fe_i as ElemId;
        let ns = fine.elem_nodes(fe);
        let (origin, m) = if child == IDENTITY {
            ([0.0_f64; 3], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        } else {
            child_map(child as usize)
        };

        for (s, kind) in kinds.iter().enumerate() {
            let r = &fe_coords[s];
            let xi = [
                origin[0] + m[0][0] * r[0] + m[0][1] * r[1] + m[0][2] * r[2],
                origin[1] + m[1][0] * r[0] + m[1][1] * r[1] + m[1][2] * r[2],
                origin[2] + m[2][0] * r[0] + m[2][1] * r[1] + m[2][2] * r[2],
            ];
            let id = match *kind {
                SlotKind::Vertex(v) => ns[v],
                SlotKind::Edge { e, m: pos } => {
                    let (a, b) = EDGES[e];
                    // Measure the within-edge index from the lower-id endpoint
                    // so both elements name the same dof; the GLL set is
                    // symmetric, so the dof at GLL position `pos` from one
                    // end sits at position `p − pos` from the other.
                    let (n0, n1) = (ns[a], ns[b]);
                    let key = if n0 <= n1 { (n0, n1, pos) } else { (n1, n0, p - pos) };
                    *edge_dofs.entry(key).or_insert_with(|| {
                        let x = pq.eval_at(pe, xi);
                        geo_coords.extend_from_slice(&x);
                        let id = next_dof;
                        next_dof += 1;
                        id
                    })
                }
                SlotKind::Face { f, w } => {
                    // The dof's GLL barycentric indices are attached to the
                    // face's corner *nodes* (a physical identity), so sorting
                    // the (node, weight) pairs by node id gives both elements
                    // sharing the fine face the same canonical frame.
                    let face = FACES[f];
                    let mut trip =
                        [(ns[face[0]], w[0]), (ns[face[1]], w[1]), (ns[face[2]], w[2])];
                    trip.sort_unstable_by_key(|&(n, _)| n);
                    let key = ([trip[0].0, trip[1].0, trip[2].0], trip[0].1, trip[1].1, trip[2].1);
                    *face_dofs.entry(key).or_insert_with(|| {
                        let x = pq.eval_at(pe, xi);
                        geo_coords.extend_from_slice(&x);
                        let id = next_dof;
                        next_dof += 1;
                        id
                    })
                }
                SlotKind::Interior => *vol_dofs.entry((fe, s)).or_insert_with(|| {
                    let x = pq.eval_at(pe, xi);
                    geo_coords.extend_from_slice(&x);
                    let id = next_dof;
                    next_dof += 1;
                    id
                }),
            };
            conn.push(id);
        }
    }

    Some(GeometryData {
        order: p as u8,
        conn,
        nodes_per_elem: dpe,
        coords: geo_coords,
        n_nodes: next_dof as usize,
    })
}
