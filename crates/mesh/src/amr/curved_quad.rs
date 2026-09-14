//! Curved (high-order) geometry support for uniform Quad4 refinement — the
//! 2-D analogue of [`super::curved_hex`] (same MFEM mechanism, square case).
//!
//! MFEM refines a curved 2-D mesh (`nodes` grid function present) in two steps
//! (`Mesh::UniformRefinement2D_base` + `Mesh::UpdateNodes`):
//!
//! 1. the straight `vertices` array is averaged exactly like a linear mesh
//!    (vertex layout `[coarse vertices | edge midpoints | quad centers]` —
//!    `refine_uniform_quad4` already implements that numbering);
//! 2. because `update_nodes` is set for curved meshes, the refined `nodes`
//!    grid function replaces the vertex positions: for every *fine* element
//!    the child's embedded sub-cell (`origin ∈ {0, 0.5}²`, scale 1/2, no
//!    mirroring) is mapped into the parent and the coarse element's own FE is
//!    evaluated at the fine dof reference points, so the fine geometry is the
//!    **per-child evaluation of the parent's nodal field**, with dofs shared
//!    between elements meeting on the same fine entity (`GridFunction::Update`'s
//!    first-touch `mark` rule).
//!
//! The old behaviour — per-child *independent* bilinear quads (`order: 1`)
//! whose corners sit on the parent field — keeps the fine vertices curved but
//! silently drops the mesh's geometric order: the mid-edge and interior
//! geometry of every child is straight/bilinear, while MFEM keeps order `p`.
//!
//! The positional layout of the `(p+1)²` geometry dofs is the element's own
//! table (`QuadQk::new(p).dof_coords()`, crates/element) — the same table
//! `fem-io`'s `nodes` reader/writer uses, so element, numbering and this
//! module cannot drift apart.

use std::collections::HashMap;

use fem_core::{ElemId, NodeId};

use crate::element_type::ElementType;
use crate::simplex::{GeometryData, Mesh};

/// Reference coordinates ([0,1]²) of the 4 vertices of a Quad4 in MFEM
/// `Geometry::Constants<SQUARE>` order: `0=(0,0) 1=(1,0) 2=(1,1) 3=(0,1)`.
const MFEM_SQUARE_VERTS: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

/// Local edges of a quad as `(from, to)` vertex pairs, in MFEM
/// `Constants<SQUARE>::Edges` order: bottom, right, top, left.
const MFEM_SQUARE_EDGES: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];

/// The 1-D Lagrange basis of the ascending nodes `nodes` evaluated at `x`
/// (exact at the nodes: `L_i(x_i)` is a product of exactly-1 ratios and every
/// other `L_i` carries an exactly-zero factor).
fn lagrange(nodes: &[f64], i: usize, x: f64) -> f64 {
    let mut v = 1.0;
    for (j, &xj) in nodes.iter().enumerate() {
        if j != i {
            v *= (x - xj) / (nodes[i] - xj);
        }
    }
    v
}

/// Read-only view of an order-`p` (`p ≥ 2`) quad [`GeometryData`] attached to a
/// 2-D mesh.
pub(crate) struct QuadQkGeometry<'a> {
    geo: &'a GeometryData,
    /// Geometry order `p`.
    pub(crate) order: usize,
    /// Geometry dofs per element, `(p+1)²` (`GeometryData::nodes_per_elem`).
    dpe: usize,
    /// The `p+1` one-dimensional dof nodes on `[0,1]`, ascending.
    nodes: Vec<f64>,
    /// Tensor index `(ix, iy)` of each local dof position, in the geometry
    /// FE's own local order (`QuadQk::dof_coords`).
    idx: Vec<[usize; 2]>,
    /// Reference coordinates ([0,1]²) of each local dof position.
    pub(crate) ref01: Vec<[f64; 2]>,
}

impl<'a> QuadQkGeometry<'a> {
    /// Returns `Some` iff `mesh` is a Quad4 mesh carrying order-`p ≥ 2`
    /// geometry with `(p+1)²` dofs per element.
    pub(crate) fn new(mesh: &'a Mesh<2>) -> Option<Self> {
        let geo = mesh.geometry.as_ref()?;
        let order = geo.order as usize;
        if order < 2 {
            // Order 1 (or absent) geometry: the straight averaging the
            // refinement already does *is* the interpolation.
            return None;
        }
        let dpe = (order + 1).pow(2);
        if geo.nodes_per_elem != dpe || geo.conn.len() < mesh.n_elems() * dpe {
            return None;
        }
        // The single source of truth for the positional layout: the element's
        // own dof table (`QuadQk`'s dof lattice).  NOTE the asymmetry with
        // `HexQk` (see `curved_hex`): the factory stores `QuadQk` dof
        // coordinates directly on the `[0,1]²` reference square
        // (`QuadQk::new` builds `gll01` = `0.5·(x+1)`), so no remapping here.
        let fe = ElementType::Quad4.ref_elem(order as u8);
        let dof_coords = fe.dof_coords();
        if dof_coords.len() != dpe {
            return None;
        }
        let ref01: Vec<[f64; 2]> = dof_coords.iter().map(|c| [c[0], c[1]]).collect();
        debug_assert!(ref01.iter().all(|r| r[0] >= 0.0 && r[0] <= 1.0 && r[1] >= 0.0 && r[1] <= 1.0),
            "QuadQk dof coordinates are expected on the [0,1]² reference square");
        let mut nodes: Vec<f64> = ref01.iter().map(|r| r[0]).collect();
        nodes.sort_by(|a, b| a.partial_cmp(b).expect("finite dof nodes"));
        nodes.dedup();
        if nodes.len() != order + 1 {
            return None;
        }
        let mut idx = Vec::with_capacity(dpe);
        for r in &ref01 {
            let t = [
                nodes.iter().position(|&n| n == r[0]).expect("dof node must exist"),
                nodes.iter().position(|&n| n == r[1]).expect("dof node must exist"),
            ];
            idx.push(t);
        }
        Some(QuadQkGeometry { geo, order, dpe, nodes, idx, ref01 })
    }

    fn elem_dofs(&self, e: ElemId) -> &[NodeId] {
        let o = e as usize * self.dpe;
        &self.geo.conn[o..o + self.dpe]
    }

    fn dof_xy(&self, dof: NodeId) -> [f64; 2] {
        let o = dof as usize * 2;
        [self.geo.coords[o], self.geo.coords[o + 1]]
    }

    /// Evaluate the parent element's order-`p` geometry field at the `[0,1]²`
    /// reference point `xi` — the tensor Lagrange interpolation of the
    /// element's `(p+1)²` geometry dofs, accumulated in local dof order.
    pub(crate) fn eval_at(&self, e: ElemId, xi: [f64; 2]) -> [f64; 2] {
        let p = self.order;
        let bx: Vec<f64> = (0..=p).map(|i| lagrange(&self.nodes, i, xi[0])).collect();
        let by: Vec<f64> = (0..=p).map(|i| lagrange(&self.nodes, i, xi[1])).collect();
        let dofs = self.elem_dofs(e);
        let mut out = [0.0_f64; 2];
        for (o, t) in self.idx.iter().enumerate() {
            let w = bx[t[0]] * by[t[1]];
            let x = self.dof_xy(dofs[o]);
            out[0] += w * x[0];
            out[1] += w * x[1];
        }
        out
    }

    /// Coordinate of the new fine vertex at the midpoint of local edge `li`
    /// (index into [`MFEM_SQUARE_EDGES`]): the parent geometry field evaluated
    /// at the edge midpoint.  For an order-2 parent that point *is* the
    /// parent's mid-edge geometry dof and the evaluation reads it back exactly
    /// (all basis weights are 0/1) — MFEM's `SetVerticesFromNodes` pick.
    pub(crate) fn edge_pick(&self, e: ElemId, li: usize) -> [f64; 2] {
        let (a, b) = MFEM_SQUARE_EDGES[li];
        let (ra, rb) = (MFEM_SQUARE_VERTS[a], MFEM_SQUARE_VERTS[b]);
        self.eval_at(e, [0.5 * (ra[0] + rb[0]), 0.5 * (ra[1] + rb[1])])
    }

    /// Coordinate of the new fine vertex at the center of element `e`: the
    /// parent field at the reference square center.
    pub(crate) fn center_pick(&self, e: ElemId) -> [f64; 2] {
        self.eval_at(e, [0.5, 0.5])
    }
}

/// Key of a shared fine edge dof: the fine edge (its sorted node pair) plus
/// the dof's index within the edge, measured from its lower-id node.
type EdgeKey = (NodeId, NodeId, usize);

/// Build the refined mesh's [`GeometryData`] from the parent's order-`p` quad
/// geometry.
///
/// `fine` is the refined mesh (connectivity + final vertex coordinates, where
/// every new vertex already carries the parent-field coordinate computed by
/// the refinement kernels).  The children of coarse element `e` are the fine
/// elements `4e..4e+4`, in child order `LL, LR, UR, UL`, and the child's
/// fine-ref → parent-ref affine map is `parent_ref = origin + 0.5·child_ref`
/// with `origin = 0.5·Constants<SQUARE>::Vertices[child]` (no mirroring).
///
/// The fine geometry dof ids occupy a fresh index space: ids `0..n_fine_nodes`
/// are the fine mesh nodes (fine vertex coordinates), followed by the edge
/// dofs created on first touch (keyed by the fine node pair so elements
/// sharing an edge share the dof — MFEM's first-touch `mark` rule) and one
/// interior dof set per fine element.  Values are the parent order-`p` field
/// evaluated at the corresponding reference point — read back exactly wherever
/// the point coincides with a parent dof (in particular every dof of the child
/// vertices) and the standard order-`p` interpolation otherwise, i.e. MFEM's
/// `GetLocalInterpolation`.
pub(crate) fn build_refined_quad_geometry(parent: &Mesh<2>, fine: &Mesh<2>) -> Option<GeometryData> {
    let pq = QuadQkGeometry::new(parent)?;
    let p = pq.order;
    let dpe = pq.dpe;
    debug_assert_eq!(fine.n_elems(), 4 * parent.n_elems());

    // Index-space corner of each local vertex: (0,0), (p,0), (p,p), (0,p).
    let vix: [[usize; 2]; 4] = [[0, 0], [p, 0], [p, p], [0, p]];

    let n_fine_nodes = fine.n_nodes() as NodeId;
    let mut geo_coords: Vec<f64> = fine.coords.clone();
    let mut next_dof = n_fine_nodes;
    let mut edge_dofs: HashMap<EdgeKey, NodeId> = HashMap::new();

    let mut conn: Vec<NodeId> = Vec::with_capacity(fine.n_elems() * dpe);
    for fe in 0..fine.n_elems() as ElemId {
        let ns = fine.elem_nodes(fe);
        let pe = fe / 4;
        let child = fe % 4;
        // Fine-ref → parent-ref affine map (origin, scale 1/2).
        let v = MFEM_SQUARE_VERTS[child as usize];
        let origin = [0.5 * v[0], 0.5 * v[1]];

        for o in 0..dpe {
            let r = pq.ref01[o];
            let xi = [origin[0] + 0.5 * r[0], origin[1] + 0.5 * r[1]];
            let t = pq.idx[o];
            let on_bnd = [t[0] == 0 || t[0] == p, t[1] == 0 || t[1] == p];
            let n_bnd = on_bnd.iter().filter(|&&b| b).count();

            let id = if n_bnd == 2 {
                // Vertex: the fine mesh node itself.
                let v = (0..4).find(|&v| vix[v] == t).expect("quad vertex slot");
                ns[v]
            } else if n_bnd == 1 {
                // Edge dof: shared by every element meeting on that fine edge.
                // `av` is the axis the dof varies along; the fixed coordinate
                // selects which of the two local edges parallel to `av`.
                let av = usize::from(on_bnd[0]);
                let fixed = 1 - av;
                let li = match (av, t[fixed] == 0) {
                    (0, true) => 0,  // bottom (y = 0, varies x)
                    (0, false) => 2, // top    (y = p, varies x)
                    (_, false) => 1, // right  (x = p, varies y)
                    (_, true) => 3,  // left   (x = 0, varies y)
                };
                let (a, b) = MFEM_SQUARE_EDGES[li];
                // Measure the within-edge index from one of its fine nodes so
                // that both elements name the same dof; the 1-D node set is
                // symmetric, so index `m` from the low end is `p - m` from the
                // high end.
                let from_lo = vix[a][av] == 0;
                let (n0, n1) = if from_lo { (ns[a], ns[b]) } else { (ns[b], ns[a]) };
                let key = if n0 <= n1 {
                    (n0, n1, t[av])
                } else {
                    (n1, n0, p - t[av])
                };
                *edge_dofs.entry(key).or_insert_with(|| {
                    let x = pq.eval_at(pe, xi);
                    geo_coords.extend_from_slice(&x);
                    let id = next_dof;
                    next_dof += 1;
                    id
                })
            } else {
                // Interior dof: private to this fine element and position.
                let x = pq.eval_at(pe, xi);
                geo_coords.extend_from_slice(&x);
                let id = next_dof;
                next_dof += 1;
                id
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
