//! The **fused pyramid P1** `L2_T1_3D_P1` table — reading semantics and the
//! uniform-refinement transport (D817-3), the pyramid analogue of
//! [`super::curved_tet`] / [`super::curved_prism`] (D816-2).
//!
//! `L2_FECollection(1, 3, BasisType::GaussLobatto)` — what `Mesh::Print`
//! writes as `L2_T1_3D_P1` — gives `Geometry::PYRAMID` the
//! `L2_FuentesPyramidElement` (the "fused" Fuentes pyramid,
//! `fem/fe/fe_l2.cpp:926-1076`): **8** dofs per element at
//! `(op[i](1 − a·op[k]), op[j](1 − a·op[k]), a·op[k])`,
//! `o = k·4 + j·2 + i`, `op = {0, 1}`, `a = 0.78867513459481287`
//! (`Poly_1D::GetPoints(1, GaussLegendre)[1]`) — four base corners plus the
//! four cross-section corners at `z = a`.  The apex carries **no dof**.  This
//! is the table MFEM's own `Mesh::SetCurvature(1, true)` + `UniformRefinement`
//! pipeline produces on pyramid meshes, and it is *ragged the moment the mesh
//! is refined once*: each Pyramid5 parent splits into 6 Pyramid5 (8-dof rows)
//! + 4 Tet4 (4-dof rows) children.
//!
//! ## The transport (MFEM-arithmetic-exact)
//!
//! `Mesh::UniformRefinement` → `UpdateNodes` propagates the `nodes` grid
//! function through `FiniteElementSpace::RefinementOperator::Mult`
//! (`fem/fespace.cpp:1944`):
//!
//! 1. `localP[geom](matrix) = fine_fe.GetLocalInterpolation(identity ⊕
//!    pmats[geom](matrix))` (`FiniteElementSpace::GetLocalRefinementMatrices`,
//!    `fem/fespace.cpp:1788`): row `i` = the **coarse** FE's shape functions
//!    at the parent-frame image of fine dof point `i`, with `|v| < 1e-12`
//!    snapped to zero (`ScalarFiniteElement::NodalLocalInterpolation`,
//!    `fem/fe/fe_base.cpp:130-171`).  The fine and coarse FE of a geometry
//!    are the same `L2` element: the Fuentes P1 (8×8) for pyramids, the
//!    barycentric tet P1 (4×4) for tets.
//! 2. per fine element `k`: `values(k) = localP[geom_k](emb.matrix) · subX`,
//!    where `subX` is the assigned parent's dof values.  MFEM assigns the
//!    embeddings in `UniformRefinement3D_base` **with a pyramid-blind rule**
//!    (`mesh/mesh.cpp:11058-11070`): only children of *tet* parents get their
//!    true `(parent, corner/interior matrix)`; every other non-tet fine
//!    element gets `(k / 8, k % 8)` — wrong templates for the 2nd..6th
//!    pyramid parent — and a tet child of a *pyramid* parent is left at the
//!    zero-initialized `Embedding` default `(parent 0, matrix 0)`.  Ported
//!    verbatim (this quirk *is* MFEM's r2 geometry; see the round-83 report).
//! 3. `DenseMatrix::Mult` in a release build reads `width` entries of `subX`
//!    regardless of the parent's row length (the `MFEM_ASSERT` compiles out),
//!    and `Vector::SetSize` keeps the never-shrinking buffer (`linalg/
//!    vector.hpp:633-643`) — so a tet parent's 4 values are followed by 4
//!    **stale leftovers** of the previous fill.  Replayed with the persistent
//!    8-slot buffer [`RefineBuffer`].
//!
//! Finally `UpdateNodes` → `SetVerticesFromNodes` rebuilds every fine vertex
//! as the mean over its element references of the element's geometry value at
//! that vertex — the value being `GridFunction::GetNodalValues`' shape-row
//! dot product (`fem/gridfunc.cpp:377-419`): unit vectors at Fuentes dofs
//! `(0, 1, 3, 2)` for the four base vertices, the fixed apex combination
//! `CalcShape(0, 0, 1)` for the apex (which carries no dof!), the identity
//! for tets.
//!
//! All of this reproduces the MFEM 4.10 oracle (`tmp/d81a/pyrl2_*`,
//! probes `tmp/d83a/probe7.cpp`) **bit for bit**, including the r2 table rows
//! MFEM computes through stale buffer slots and scrambled templates.

use fem_core::{ElemId, NodeId};
use fem_element::lagrange::l2_fuentes_pyramid_p1_shapes_gauss_lobatto;
use fem_element::lagrange::pyramid_l2::l2_fuentes_pyramid_nodes;

use crate::element_type::ElementType;
use crate::simplex::{GeometryData, Mesh};

/// Geometry element kind of a fused-table row.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RowKind {
    Pyramid,
    Tet,
}

impl RowKind {
    /// Dofs per row: 8 for the Fuentes P1, 4 for the tet P1.
    fn row_dofs(self) -> usize {
        match self {
            RowKind::Pyramid => 8,
            RowKind::Tet => 4,
        }
    }

    fn of(et: ElementType) -> Option<Self> {
        match et {
            ElementType::Pyramid5 => Some(RowKind::Pyramid),
            ElementType::Tet4 => Some(RowKind::Tet),
            _ => None,
        }
    }
}

/// The mesh's geometry as a **fused order-1 pyramid** table — the ragged (or
/// pure-pyramid uniform) `L2_T1_3D_P1` layout the io reader attaches for
/// pyramid meshes: element-major *fresh, unshared* dof ids, pyramid rows of
/// 8 in the Fuentes dof order, tet rows of 4 in the reference-tet vertex
/// order.  Every element must be a Pyramid5 or Tet4 and at least one must be
/// a pyramid (a pure-tet fused table is the D816-2 tet arm's shape).
pub(crate) fn l2_p1_pyramid_fused_geometry(mesh: &Mesh<3>) -> Option<&GeometryData> {
    let geo = mesh.geometry.as_ref()?;
    if geo.order != 1 {
        return None;
    }
    let mut n_rows = 0usize;
    let mut has_pyr = false;
    for e in 0..mesh.n_elems() as ElemId {
        let kind = RowKind::of(mesh.element_type_at(e))?;
        n_rows += kind.row_dofs();
        has_pyr |= kind == RowKind::Pyramid;
    }
    if !has_pyr || geo.conn.len() != n_rows || geo.n_nodes != geo.conn.len() {
        return None;
    }
    // Fresh, unshared dof ids only.
    for (i, &d) in geo.conn.iter().enumerate() {
        if d as usize != i {
            return None;
        }
    }
    Some(geo)
}

/// `pyr_t` refinement templates — MFEM `pyr_children` (`mesh/mesh.cpp:11014`):
/// 10 children × 5 columns, each column the parent-reference image of a
/// reference-pyramid vertex (base cycle `(0,0,0) (1,0,0) (1,1,0) (0,1,0)`,
/// apex `(0,0,1)`).  `A = 0, B = 0.5, C = 1, D = −1`; the four tet rows pad
/// their fifth column with the `D` sentinel (never read — the tet transform
/// consumes 4 columns).
const PYR_CHILDREN: [[[f64; 3]; 5]; 10] = [
    [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
    [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5]],
    [[0.5, 0.5, 0.0], [1.0, 0.5, 0.0], [1.0, 1.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.5, 0.5]],
    [[0.0, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 0.5]],
    [[0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]],
    [[0.0, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.0, 0.5], [0.0, 0.0, 0.5], [0.5, 0.5, 0.0]],
    [[0.5, 0.0, 0.0], [0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [-1.0, -1.0, -1.0]],
    [[1.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0], [-1.0, -1.0, -1.0]],
    [[0.5, 1.0, 0.0], [0.5, 0.5, 0.5], [0.0, 0.5, 0.5], [0.5, 0.5, 0.0], [-1.0, -1.0, -1.0]],
    [[0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.5], [0.5, 0.5, 0.0], [-1.0, -1.0, -1.0]],
];

/// Reference-tet vertices, MFEM `Geometry::Constants<TETRAHEDRON>` order.
const TET_VERTS: [[f64; 3]; 4] =
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// `LinearPyramidFiniteElement::CalcShape` (`fem/fe/fe_fixed_order.cpp`) —
/// the transforming FE of `IsoparametricTransformation::
/// SetIdentityTransformation(PYRAMID)`, consumed by the point-matrix map of
/// every fine pyramid dof.
fn linear_pyr_shape(x: f64, y: f64, z: f64) -> [f64; 5] {
    // MFEM's apex limit: `oz <= 1e-6` collapses to the apex dof.
    if 1.0 - z <= 1e-6 {
        return [0.0, 0.0, 0.0, 0.0, 1.0];
    }
    let ox = 1.0 - x - z;
    let oy = 1.0 - y - z;
    let ozi = 1.0 / (1.0 - z);
    [ox * oy * ozi, x * oy * ozi, x * y * ozi, ox * y * ozi, z]
}

/// `Linear3DFiniteElement::CalcShape` — the tet's transforming FE:
/// barycentric coordinates.
fn linear_tet_shape(x: f64, y: f64, z: f64) -> [f64; 4] {
    [1.0 - x - y - z, x, y, z]
}

/// `IsoparametricTransformation::Transform` ⊕ `kernels::Mult` for a point
/// matrix given by its *columns* (the template points): the accumulation runs
/// column by column — ascending `j` with no initial zero add, which is the
/// row dot in MFEM's exact order.
fn point_mat_transform<const N: usize>(cols: &[[f64; 3]; N], shape: &[f64; N]) -> [f64; 3] {
    let mut out = [0.0_f64; 3];
    for (t, o) in out.iter_mut().enumerate() {
        let mut acc = shape[0] * cols[0][t];
        for j in 1..N {
            acc += shape[j] * cols[j][t];
        }
        *o = acc;
    }
    out
}

/// `ScalarFiniteElement::NodalLocalInterpolation`'s snap: `|v| < 1e-12 → 0`.
fn snap(v: f64) -> f64 {
    if v.abs() < 1e-12 { 0.0 } else { v }
}

/// `localP[PYRAMID](matrix)` — the 8×8 local interpolation matrix of child
/// template `matrix`; rows = fine Fuentes dofs, columns = coarse Fuentes
/// dofs, row-major.
fn local_pyr_interpolation(matrix: usize) -> [[f64; 8]; 8] {
    let cols = &PYR_CHILDREN[matrix];
    let fine_nodes = l2_fuentes_pyramid_nodes(1, true);
    let mut out = [[0.0_f64; 8]; 8];
    for (i, node) in fine_nodes.iter().enumerate() {
        let w = linear_pyr_shape(node[0], node[1], node[2]);
        let p = point_mat_transform(cols, &w);
        let shape = l2_fuentes_pyramid_p1_shapes_gauss_lobatto(p[0], p[1], p[2]);
        for (j, v) in shape.iter().enumerate() {
            out[i][j] = snap(*v);
        }
    }
    out
}

/// The reference-tet corner images of `tet_children` matrix `matrix`
/// (`mesh/mesh.cpp:10960-11000` — corner children `0..4`, interior
/// `4·(rt+1)+k`; the same dyadic templates
/// [`super::curved_tet::child_points`] builds).
fn tet_child_points(matrix: usize) -> [[f64; 3]; 4] {
    super::curved_tet::child_points(matrix)
}

/// `localP[TETRAHEDRON](matrix)` — the 4×4 local interpolation matrix of tet
/// child template `matrix` (barycentric in, barycentric out).
fn local_tet_interpolation(matrix: usize) -> [[f64; 4]; 4] {
    let cols = &tet_child_points(matrix);
    let mut out = [[0.0_f64; 4]; 4];
    for (i, node) in TET_VERTS.iter().enumerate() {
        let w = linear_tet_shape(node[0], node[1], node[2]);
        let p = point_mat_transform(cols, &w);
        let shape = linear_tet_shape(p[0], p[1], p[2]);
        for (j, v) in shape.iter().enumerate() {
            out[i][j] = snap(*v);
        }
    }
    out
}

/// MFEM `UniformRefinement3D_base`'s embedding assignment for a refined
/// pyramid-carrying mesh, verbatim (`mesh/mesh.cpp:11058-11070`): children of
/// tet parents keep their explicit `(parent, corner|interior matrix)`; a tet
/// child of a *pyramid* parent stays at the zero-initialized `Embedding`
/// default `(0, 0)`; every other fine element `k` gets the pyramid-blind
/// generic rule `(k / 8, k % 8)`.  `explicit_matrix` is `Some` for exactly
/// the tet-parent children.
pub(crate) fn assign_embeddings(
    fine_types: &[ElementType],
    fine_owner: &[ElemId],
    parent_types: &[ElementType],
    explicit_matrix: &[Option<usize>],
) -> Vec<(ElemId, usize)> {
    fine_types
        .iter()
        .enumerate()
        .map(|(k, &ft)| match ft {
            ElementType::Tet4 => {
                if parent_types[fine_owner[k] as usize] == ElementType::Tet4 {
                    (fine_owner[k], explicit_matrix[k].expect("tet child matrix"))
                } else {
                    // The zero-initialized default `Embedding` — mesh.cpp
                    // never assigns these (the generic loop skips tets).
                    (0, 0)
                }
            }
            _ => ((k / 8) as ElemId, k % 8),
        })
        .collect()
}

/// The reusable `subX` Vector of `RefinementOperator::Mult`: filled to the
/// assigned parent's row length per (element, component), read to the local
/// matrix's width — a width past the fill replays the previous fill's
/// leftovers (`Vector::SetSize` keeps the never-shrinking buffer).
struct RefineBuffer {
    slots: [f64; 8],
}

impl RefineBuffer {
    fn new() -> Self {
        // MFEM's first fill (fine element 0, whose assigned parent's row is
        // written before any read) initialises every slot this pipeline ever
        // reads past; the zero start stands in for that write.
        Self { slots: [0.0; 8] }
    }

    fn fill(&mut self, values: &[f64]) {
        self.slots[..values.len()].copy_from_slice(values);
    }

    /// `y = lP · slots[0..W]` in `kernels::Mult`'s exact ascending order.
    fn mult<const H: usize, const W: usize>(&self, lp: &[[f64; W]; H]) -> [f64; H] {
        let mut out = [0.0_f64; H];
        for (row, o) in out.iter_mut().enumerate() {
            let mut acc = self.slots[0] * lp[row][0];
            for j in 1..W {
                acc += self.slots[j] * lp[row][j];
            }
            *o = acc;
        }
        out
    }
}

/// Build the refined mesh's fused [`GeometryData`] from the parent's table.
///
/// `parent` is the coarse mesh carrying `parent_geo` (the fused table);
/// `fine_types` describes the fine mesh in element order; `assign` is
/// [`assign_embeddings`]' output.  The fine rows are element-major fresh dof
/// ids — pyramid rows in the Fuentes dof order, tet rows in the reference
/// vertex order — which is the file `L2` order the oracles dump.
pub(crate) fn build_refined_l2_p1_fused_geometry(
    parent: &Mesh<3>,
    parent_geo: &GeometryData,
    fine_types: &[ElementType],
    assign: &[(ElemId, usize)],
) -> GeometryData {
    // Parent row offsets (element-major fresh ids; row length by kind).
    let mut parent_row0 = vec![0usize; parent.n_elems()];
    let mut cursor = 0usize;
    for e in 0..parent.n_elems() as ElemId {
        parent_row0[e as usize] = cursor;
        cursor += RowKind::of(parent.element_type_at(e)).expect("fused parent kinds").row_dofs();
    }
    debug_assert_eq!(parent_geo.coords.len(), cursor * 3);

    let mut buf = RefineBuffer::new();
    let mut conn = Vec::with_capacity(fine_types.len() * 8);
    let mut coords: Vec<f64> = Vec::with_capacity(fine_types.len() * 8 * 3);
    let mut row_dof_cursor = 0usize;
    for (k, &ft) in fine_types.iter().enumerate() {
        let kind = RowKind::of(ft).expect("fused table children are Pyramid5/Tet4 only");
        let (pe, matrix) = assign[k];
        let prow0 = parent_row0[pe as usize];
        // The assigned parent's row length — a tet parent fills only 4 of the
        // buffer's 8 slots (the stale-slot replay of `RefineBuffer`).
        let plen = RowKind::of(parent.element_type_at(pe))
            .expect("fused parent kinds")
            .row_dofs();
        let mut out = [[0.0_f64; 3]; 8];
        for c in 0..3 {
            for (j, slot) in buf.slots.iter_mut().take(plen).enumerate() {
                *slot = parent_geo.coords[(prow0 + j) * 3 + c];
            }
            match kind {
                RowKind::Pyramid => {
                    let y = buf.mult(&local_pyr_interpolation(matrix));
                    for (i, v) in y.iter().enumerate() {
                        out[i][c] = *v;
                    }
                }
                RowKind::Tet => {
                    let y = buf.mult(&local_tet_interpolation(matrix));
                    for (i, v) in y.iter().enumerate() {
                        out[i][c] = *v;
                    }
                }
            }
        }
        for xyz in out.iter().take(kind.row_dofs()) {
            conn.push(row_dof_cursor as NodeId);
            coords.extend_from_slice(xyz);
            row_dof_cursor += 1;
        }
    }
    GeometryData {
        order: 1,
        conn,
        nodes_per_elem: 0,
        coords,
        n_nodes: row_dof_cursor,
    }
}

/// Rebuild the fine vertex table, MFEM `SetVerticesFromNodes` ⊕
/// `GridFunction::GetNodalValues`: every vertex is the mean over its element
/// references of the element's geometry value at that reference vertex — the
/// dot of the element's shape row at the vertex with the element's dof values
/// (ascending dof order), divided once at the end.
pub(crate) fn set_vertices_from_nodes_fused(mesh: &Mesh<3>, geo: &GeometryData) -> Vec<f64> {
    // Fixed shape rows per kind: pyramid base vertices are unit vectors at
    // Fuentes dofs (0, 1, 3, 2); the apex is `CalcShape(0, 0, 1)`; tets are
    // the identity.
    let apex_row = l2_fuentes_pyramid_p1_shapes_gauss_lobatto(0.0, 0.0, 1.0);
    const PYR_VERTEX_DOFS: [usize; 4] = [0, 1, 3, 2];

    let n = mesh.n_nodes();
    let mut coords = vec![0.0_f64; n * 3];
    let mut overlap = vec![0_usize; n];
    let mut row0 = 0usize;
    for e in 0..mesh.n_elems() as ElemId {
        let kind = RowKind::of(mesh.element_type_at(e)).expect("fused mesh elements");
        let ns = mesh.elem_nodes(e);
        match kind {
            RowKind::Pyramid => {
                for (k, &v) in ns.iter().enumerate() {
                    let v = v as usize;
                    for c in 0..3 {
                        let dot = if k < 4 {
                            geo.coords[(row0 + PYR_VERTEX_DOFS[k]) * 3 + c]
                        } else {
                            let mut acc = 0.0_f64;
                            for (d, &wd) in apex_row.iter().enumerate() {
                                acc += wd * geo.coords[(row0 + d) * 3 + c];
                            }
                            acc
                        };
                        coords[v * 3 + c] += dot;
                    }
                    overlap[v] += 1;
                }
                row0 += 8;
            }
            RowKind::Tet => {
                for (k, &v) in ns.iter().enumerate() {
                    for c in 0..3 {
                        coords[v as usize * 3 + c] += geo.coords[(row0 + k) * 3 + c];
                    }
                    overlap[v as usize] += 1;
                }
                row0 += 4;
            }
        }
    }
    for (v, &ov) in overlap.iter().enumerate() {
        if ov > 0 {
            for c in 0..3 {
                coords[v * 3 + c] /= ov as f64;
            }
        }
    }
    coords
}
