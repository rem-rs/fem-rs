//! H(curl) finite element space for Nédélec edge elements.
//!
//! ## DOF association (D32 — MFEM nodal point-value semantics)
//!
//! Each edge carries `k = order` global DOFs.  For ND2 (and the nodal quad
//! NDk bases) the DOF functionals are **point evaluations of the tangential
//! component** at Gauss-Legendre points along the edge, with the tangent given
//! by the physical edge vector — exactly MFEM's `ND_*Element` `FE::Nodes`
//! semantics (`σ_j(Φ) = Φ(y_j)·τ`, `y_j = P_min + t_j·τ`, `τ = P_max − P_min`).
//!
//! The Gauss points are symmetric about the edge midpoint, so an element whose
//! local edge direction opposes the canonical (min→max) direction has its
//! local slot `m` sitting at the same physical point as canonical slot
//! `k−1−m`, with the opposite tangent: `σ^local_m = −ρ_{k−1−m}`.  The
//! local→global mapping therefore uses a **signed anti-diagonal permutation**
//! — the identical encoding to MFEM's element dof tables (reversed dof order
//! + sign −1).  For ND1 (k=1) this reduces to the classic ±1 edge sign.
//!
//! (The pre-D32 integral-moment functionals `∫Φ·τ t^m dt` are *not*
//! reflection invariant — reversal mixes moments binomially, which scalar
//! signs cannot express — so adjacent elements disagreed on the shared-edge
//! trace and curl-curl solutions were polluted.)
//!
//! ## Sign convention
//!
//! The canonical orientation of an edge is from the smaller to the larger
//! vertex id — mirroring MFEM's `DSTable`-based `GetElementToEdgeTable`, whose
//! `Push(a,b)` stores every edge as (min, max).  The assembler multiplies each
//! basis-function value by its sign (and uses the permuted global id) to
//! guarantee tangential continuity across elements.

use std::collections::HashMap;

use fem_core::types::DofId;
use fem_element::nedelec::{HexNDk, PrismNDk, PyraNDk, QuadND, TetNDk, TriNDk};
use fem_element::quadrature::gauss_legendre_01;
use fem_element::reference::VectorReferenceElement;
use fem_linalg::Vector;
use fem_mesh::{topology::MeshTopology, ElementType};

use crate::dof_manager::{EdgeKey, FaceKey, QuadFaceKey};
use crate::fe_space::{FESpace, SpaceType};

// ─── Local edge tables ──────────────────────────────────────────────────────

/// Local edge vertex pairs for 2-D triangles.
///
/// ND1 (order 1) uses `TRI_EDGES_ND1` — the historical fem-rs convention with
/// the third edge directed (0,2), which matches the TriND1 Whitney slot and
/// keeps the assembled ND1 path bit-identical to the round-13 baseline.
/// Orders ≥ 2 use `TRI_EDGES_MFEM` = MFEM `Geometry::Constants<TRIANGLE>::Edges`
/// = (0,1),(1,2),(2,0), matching the TriND2 slot parametrizations exactly
/// (slot `m` of each edge sits at the m-th Gauss point along the pair
/// direction).
pub const TRI_EDGES_ND1: [(usize, usize); 3] = [(0, 1), (1, 2), (0, 2)];
pub const TRI_EDGES_MFEM: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];

/// Local edge vertex pairs for 2-D quadrilaterals (QuadND1 ordering).
const QUAD_EDGES: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];

/// Local edge vertex pairs for 3-D tetrahedra (TetND1 ordering).
const TET_EDGES: [(usize, usize); 6] = [
    (0, 1), (0, 2), (0, 3),
    (1, 2), (1, 3), (2, 3),
];

/// Local edge vertex pairs for 3-D hexahedra (Hex8 ordering).
///
/// Ordering = MFEM `Geometry::Constants<Geometry::CUBE>::Edges` so that edge
/// numbering and the HexND1 basis ordering match MFEM bit-for-bit:
///   (0,1),(1,2),(3,2),(0,3),(4,5),(5,6),(7,6),(4,7),(0,4),(1,5),(2,6),(3,7)
const HEX_EDGES: [(usize, usize); 12] = [
    (0, 1), (1, 2), (3, 2), (0, 3),
    (4, 5), (5, 6), (7, 6), (4, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
];

/// Local edge vertex pairs for prism (Prism6 ordering).
///
/// Ordering = MFEM `Geometry::Constants<Geometry::PRISM>::Edges` so that edge
/// numbering and the PrismND1 basis ordering match MFEM bit-for-bit:
///   (0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5)
const PRISM_EDGES: [(usize, usize); 9] = [
    (0, 1), (1, 2), (2, 0), // bottom triangle
    (3, 4), (4, 5), (5, 3), // top triangle
    (0, 3), (1, 4), (2, 5), // vertical
];

/// Local edge vertex pairs for pyramid (Pyramid5 ordering).
///
/// Ordering **and directions** = MFEM `Geometry::Constants<Geometry::PYRAMID>::
/// Edges` — note edge 2 is directed (3,2) and edge 3 (0,3) (D547: a (2,3)/
/// (3,0) reading flips the edge orientation sign and the ND≥2 anti-diagonal
/// slot placement of exactly those two edges, which shows up in assembly but
/// not in the entity-canonical interpolation).
const PYRAMID_EDGES: [(usize, usize); 8] = [
    (0, 1), (1, 2), (3, 2), (0, 3), // base quad
    (0, 4), (1, 4), (2, 4), (3, 4), // apex edges
];

/// Local face definitions for 3-D tetrahedra (MFEM tet face order; sets match
/// MFEM `(1,2,3),(0,3,2),(0,1,3),(0,2,1)`).
const TET_FACES: [(usize, usize, usize); 4] = [
    (1, 2, 3),
    (0, 2, 3),
    (0, 1, 3),
    (0, 1, 2),
];

/// Face tangent slot table per `TET_FACES` entry, mirroring the TetND2
/// (MFEM `ND_TetrahedronElement`) face `dof2tk` pairs.  For an entry
/// `(p0, n0, p1, n1)`: slot 0 tangent = verts[p0] − verts[n0], slot 1 tangent
/// = verts[p1] − verts[n1] (reference directions; physical images via the
/// element Jacobian = the corresponding physical edge vectors).
///   face (1,2,3): v2−v1, v3−v1
///   face (0,3,2): v3−v0, v2−v0
///   face (0,1,3): v1−v0, v3−v0
///   face (0,2,1): v2−v0, v1−v0
const TET_FACE_TANGENTS: [(usize, usize, usize, usize); 4] = [
    (2, 1, 3, 1),
    (3, 0, 2, 0),
    (1, 0, 3, 0),
    (2, 0, 1, 0),
];

/// Local quad-face definitions for 3-D hexahedra (Quad4 face ordering).
pub(crate) const HEX_QUAD_FACES: [(usize, usize, usize, usize); 6] = [
    (0, 1, 2, 3), // z=-1 (bottom)
    (4, 5, 6, 7), // z= 1 (top)
    (0, 1, 5, 4), // y=-1 (front)
    (2, 3, 7, 6), // y= 1 (back)
    (0, 3, 7, 4), // x=-1 (left)
    (1, 2, 6, 5), // x= 1 (right)
];

/// Local triangular faces for prism (Prism6 ordering): bottom + top.
const PRISM_TRI_FACES: [(usize, usize, usize); 2] = [
    (0, 1, 2),
    (3, 4, 5),
];

/// Local quad faces for prism (Prism6 ordering).
pub(crate) const PRISM_QUAD_FACES: [(usize, usize, usize, usize); 3] = [
    (0, 1, 4, 3),
    (1, 2, 5, 4),
    (0, 2, 5, 3),
];

/// Local triangular faces for pyramid (Pyramid5 ordering): 4 apex triangles.
const PYRAMID_TRI_FACES: [(usize, usize, usize); 4] = [
    (0, 1, 4),
    (1, 2, 4),
    (2, 3, 4),
    (3, 0, 4),
];

// ─── Hex ND face-interior DOF geometry ──────────────────────────────────────

/// Hex8 reference vertices (`fem-element` `HexQ1` order) on `[0,1]³`
/// (D721: the hex reference frame is MFEM's; the historical `[-1,1]³`
/// corners are gone).
const HEX8_REF: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];

/// Physical point and Jacobian of the trilinear hexahedron map at `xi` on
/// `[0,1]³` (Jacobian columns as `jac[component][derivative]`) — the same
/// `ox·oy·oz` formulas as MFEM `TriLinear3DFiniteElement` and
/// `fem_element::lagrange::HexQ1` (D721).
///
/// D121: published (`pub`) so the assembly crate's postprocessing and the
/// joule miniapp can map reference points through the same Q1 hex geometry
/// the H(curl) space itself uses (MFEM `Hex8` `ElementTransformation`
/// semantics), instead of each consumer re-deriving the trilinear map.
pub fn hex_trilinear_map(verts: &[[f64; 3]; 8], xi: &[f64]) -> ([f64; 3], [[f64; 3]; 3]) {
    let (x, y, z) = (xi[0], xi[1], xi[2]);
    let (ox, oy, oz) = (1.0 - x, 1.0 - y, 1.0 - z);
    let mut p = [0.0_f64; 3];
    let mut jac = [[0.0_f64; 3]; 3];
    for (i, v) in verts.iter().enumerate() {
        let r = HEX8_REF[i];
        // Nodal factor and its derivative per axis (r ∈ {0,1} selects the
        // low/high 1-D Lagrange function).
        let (fx, dx) = if r[0] == 1.0 { (x, 1.0) } else { (ox, -1.0) };
        let (fy, dy) = if r[1] == 1.0 { (y, 1.0) } else { (oy, -1.0) };
        let (fz, dz) = if r[2] == 1.0 { (z, 1.0) } else { (oz, -1.0) };
        let n = fx * fy * fz;
        let gx = dx * fy * fz;
        let gy = fx * dy * fz;
        let gz = fx * fy * dz;
        for d in 0..3 {
            p[d] += n * v[d];
            jac[d][0] += gx * v[d];
            jac[d][1] += gy * v[d];
            jac[d][2] += gz * v[d];
        }
    }
    (p, jac)
}

/// The 8 hexahedron vertices of element `e` (the corners come first in the
/// `Hex8`/`Hex20` node order).
fn hex8_verts<M: MeshTopology>(mesh: &M, e: u32) -> [[f64; 3]; 8] {
    let nodes = mesh.element_nodes(e);
    let mut v = [[0.0_f64; 3]; 8];
    for i in 0..8 {
        let c = mesh.node_coords(nodes[i]);
        for d in 0..3 {
            v[i][d] = c[d];
        }
    }
    v
}

/// `HEX_QUAD_FACES` index (z−, z+, y−, y+, x−, x+) → the element's face-block
/// index (D225: `HexNDk` face blocks follow MFEM `CUBE::FaceVert`, i.e.
/// z−, y−, x+, y+, x−, z+).
const HEX_QUAD_FACE_TO_ND_BLOCK: [usize; 6] = [0, 5, 1, 3, 4, 2];

/// Inverse of [`HEX_QUAD_FACE_TO_ND_BLOCK`]: element face-block index →
/// `HEX_QUAD_FACES` entry, so the element-slot table can be filled in block
/// order.
const HEX_ND_BLOCK_TO_QUAD_FACE: [usize; 6] = [0, 2, 5, 3, 4, 1];

// ─── Prism / pyramid ND face-interior DOF geometry (D525) ───────────────────

/// Trilinear prism map (MFEM `LinearWedgeFiniteElement` frame): reference
/// (x, y, z) with barycentric (1−x−y, x, y) over the bottom triangle
/// (V0,V1,V2) and the same over the top triangle (V3,V4,V5), z ∈ [0,1].
fn prism_trilinear_map(verts: &[[f64; 3]; 6], x: f64, y: f64, z: f64) -> [f64; 3] {
    let (l0, b) = (1.0 - x - y, 1.0 - z);
    let n = [l0 * b, x * b, y * b, l0 * z, x * z, y * z];
    let mut p = [0.0_f64; 3];
    for (i, v) in verts.iter().enumerate() {
        for d in 0..3 {
            p[d] += n[i] * v[d];
        }
    }
    p
}

/// Jacobian of [`prism_trilinear_map`], `J[d][c] = ∂x_d/∂ξ_c` (MFEM frame).
fn prism_map_jacobian(verts: &[[f64; 3]; 6], x: f64, y: f64, z: f64) -> [[f64; 3]; 3] {
    let l0 = 1.0 - x - y;
    // ∂N/∂(x, y, z) per vertex
    let dn = [
        [-(1.0 - z), -(1.0 - z), -l0],
        [1.0 - z, 0.0, -x],
        [0.0, 1.0 - z, -y],
        [-z, -z, l0],
        [z, 0.0, x],
        [0.0, z, y],
    ];
    let mut j = [[0.0_f64; 3]; 3];
    for (i, v) in verts.iter().enumerate() {
        for d in 0..3 {
            for c in 0..3 {
                j[d][c] += dn[i][c] * v[d];
            }
        }
    }
    j
}

/// The 6 prism vertices of element `e` (the corners are the whole `Prism6`
/// node list, bottom triangle V0..V2, top V3..V5).
fn prism6_verts<M: MeshTopology>(mesh: &M, e: u32) -> [[f64; 3]; 6] {
    let nodes = mesh.element_nodes(e);
    let mut v = [[0.0_f64; 3]; 6];
    for i in 0..6 {
        let c = mesh.node_coords(nodes[i]);
        for d in 0..3 {
            v[i][d] = c[d];
        }
    }
    v
}

/// The 5 pyramid vertices of element `e` (base V0..V3, apex V4).
fn pyramid5_verts<M: MeshTopology>(mesh: &M, e: u32) -> [[f64; 3]; 5] {
    let nodes = mesh.element_nodes(e);
    let mut v = [[0.0_f64; 3]; 5];
    for i in 0..5 {
        let c = mesh.node_coords(nodes[i]);
        for d in 0..3 {
            v[i][d] = c[d];
        }
    }
    v
}

/// Collapsed straight-sided pyramid map: `phys = (1−z)·B(x/(1−z), y/(1−z)) +
/// z·V4`, with `B` the bilinear base map over (V0..V3) — on straight pyramids
/// this is exactly MFEM's `LinearPyramidFiniteElement` interpolation and
/// restricts to the affine face map on every side face.
fn pyramid_map(verts: &[[f64; 3]; 5], x: f64, y: f64, z: f64) -> [f64; 3] {
    let w = 1.0 - z;
    if w <= 1e-30 {
        return verts[4];
    }
    let (u, v) = (x / w, y / w);
    let b = [(1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v];
    let mut p = [0.0_f64; 3];
    for d in 0..3 {
        p[d] = w
            * (b[0] * verts[0][d]
                + b[1] * verts[1][d]
                + b[2] * verts[2][d]
                + b[3] * verts[3][d])
            + z * verts[4][d];
    }
    p
}

/// Physical point and Jacobian of the collapsed straight-sided pyramid map
/// `phys = (1−z)·B(x/(1−z), y/(1−z)) + z·V4`, `J[d][c] = ∂x_d/∂ξ_c`.
fn pyramid_map_jacobian(verts: &[[f64; 3]; 5], x: f64, y: f64, z: f64) -> [[f64; 3]; 3] {
    let w = 1.0 - z;
    if w <= 1e-30 {
        return [[0.0; 3]; 3];
    }
    let (u, v) = (x / w, y / w);
    // bilinear base B over (V0..V3) and its ∂/∂u, ∂/∂v
    let b = [(1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v];
    let bu = [-(1.0 - v), 1.0 - v, v, -v];
    let bv = [-(1.0 - u), -u, u, 1.0 - u];
    let mut j = [[0.0_f64; 3]; 3];
    for d in 0..3 {
        // ∂/∂x = ∂B/∂u · V (the (1−z) factors cancel)
        j[d][0] = bu[0] * verts[0][d] + bu[1] * verts[1][d] + bu[2] * verts[2][d]
            + bu[3] * verts[3][d];
        j[d][1] = bv[0] * verts[0][d] + bv[1] * verts[1][d] + bv[2] * verts[2][d]
            + bv[3] * verts[3][d];
        let base = b[0] * verts[0][d] + b[1] * verts[1][d] + b[2] * verts[2][d]
            + b[3] * verts[3][d];
        j[d][2] =
            -base + (bu[0] * verts[0][d] + bu[1] * verts[1][d] + bu[2] * verts[2][d]
                + bu[3] * verts[3][d])
                * x / w
                + (bv[0] * verts[0][d] + bv[1] * verts[1][d] + bv[2] * verts[2][d]
                    + bv[3] * verts[3][d])
                * y / w
                + verts[4][d];
    }
    j
}

/// `J·t` (physical image of a reference tangent).
fn j_mul(j: &[[f64; 3]; 3], t: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|d| j[d][0] * t[0] + j[d][1] * t[1] + j[d][2] * t[2])
}

/// Physical points of the prism interior slot block
/// (`k(k−1)² + k(k−1)(k−2)/2` slots, D525).
fn prism_interior_nodes(verts: &[[f64; 3]; 6], k: usize, layout: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let off = 9 * k + 8 * k * (k - 1);
    layout[off..]
        .iter()
        .map(|p| prism_trilinear_map(verts, p[0], p[1], p[2]))
        .collect()
}

/// Physical points of the pyramid interior slot block (`3k(k−1)²`, D525).
fn pyramid_interior_nodes(verts: &[[f64; 3]; 5], k: usize, layout: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let off = 8 * k + 6 * k * (k - 1);
    layout[off..]
        .iter()
        .map(|p| pyramid_map(verts, p[0], p[1], p[2]))
        .collect()
}

// ─── Prism / pyramid face slot anchors (D546/D547) ──────────────────────────

/// `MFEM-frame` reference tangents of the `ND_WedgeElement` slots: the
/// element's own (fem-rs frame `(ξ,η,ζ) = (z,x,y)`) tangents rotated back
/// `(t_ξ,t_η,t_ζ) → (t_η,t_ζ,t_ξ)` so they pair with the MFEM-frame layout
/// points and the MFEM-frame prism Jacobian.
fn prism_tangents_mfem(k: usize) -> Vec<[f64; 3]> {
    PrismNDk::new(k)
        .dof_tangents()
        .iter()
        .map(|t| [t[1], t[2], t[0]])
        .collect()
}

/// Physical `(point, tangent-pair)` of one prism tri-face slot block
/// (`k(k−1)` slots, two per point, `f = 0` bottom / `f = 1` top): the
/// creating element's canonical shared-face functionals, `t = J·t̂`.
fn prism_tri_face_slots(
    verts: &[[f64; 3]; 6],
    k: usize,
    f: usize,
    layout: &[[f64; 3]],
    tks: &[[f64; 3]],
) -> (Vec<[f64; 3]>, Vec<[[f64; 3]; 2]>) {
    let nfd = k * (k - 1);
    let off = 9 * k + f * nfd;
    let mut pts = Vec::with_capacity(nfd / 2);
    let mut tans = Vec::with_capacity(nfd / 2);
    for i in 0..nfd / 2 {
        let xi = layout[off + 2 * i];
        let j = prism_map_jacobian(verts, xi[0], xi[1], xi[2]);
        pts.push(prism_trilinear_map(verts, xi[0], xi[1], xi[2]));
        tans.push([j_mul(&j, &tks[off + 2 * i]), j_mul(&j, &tks[off + 2 * i + 1])]);
    }
    (pts, tans)
}

/// Physical `(point, tangent)` list of one prism quad-face slot block
/// (`2k(k−1)` slots, `f = 0,1,2` = the MFEM faces (0,1,4,3), (1,2,5,4),
/// (2,0,3,5)).
fn prism_quad_face_slots(
    verts: &[[f64; 3]; 6],
    k: usize,
    f: usize,
    layout: &[[f64; 3]],
    tks: &[[f64; 3]],
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let nfd = 2 * k * (k - 1);
    let off = 9 * k + 2 * k * (k - 1) + f * nfd;
    let mut pts = Vec::with_capacity(nfd);
    let mut tans = Vec::with_capacity(nfd);
    for n in 0..nfd {
        let xi = layout[off + n];
        let j = prism_map_jacobian(verts, xi[0], xi[1], xi[2]);
        pts.push(prism_trilinear_map(verts, xi[0], xi[1], xi[2]));
        tans.push(j_mul(&j, &tks[off + n]));
    }
    (pts, tans)
}

/// Physical `(point, tangent-pair)` of one pyramid tri-face slot block
/// (`k(k−1)` slots, `f = 0..4` = the MFEM faces (0,1,4), (1,2,4), (2,3,4),
/// (3,0,4)).
fn pyramid_tri_face_slots(
    verts: &[[f64; 3]; 5],
    k: usize,
    f: usize,
    layout: &[[f64; 3]],
    tks: &[[f64; 3]],
) -> (Vec<[f64; 3]>, Vec<[[f64; 3]; 2]>) {
    let nfd = k * (k - 1);
    let off = 8 * k + 2 * k * (k - 1) + f * nfd;
    let mut pts = Vec::with_capacity(nfd / 2);
    let mut tans = Vec::with_capacity(nfd / 2);
    for i in 0..nfd / 2 {
        let xi = layout[off + 2 * i];
        let j = pyramid_map_jacobian(verts, xi[0], xi[1], xi[2]);
        pts.push(pyramid_map(verts, xi[0], xi[1], xi[2]));
        tans.push([j_mul(&j, &tks[off + 2 * i]), j_mul(&j, &tks[off + 2 * i + 1])]);
    }
    (pts, tans)
}

/// Physical `(point, tangent)` list of the pyramid base quad-face slot block
/// (`2k(k−1)` slots, D525).
fn pyramid_quad_face_slots(
    verts: &[[f64; 3]; 5],
    k: usize,
    layout: &[[f64; 3]],
    tks: &[[f64; 3]],
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let nfd = 2 * k * (k - 1);
    let off = 8 * k;
    let mut pts = Vec::with_capacity(nfd);
    let mut tans = Vec::with_capacity(nfd);
    for n in 0..nfd {
        let xi = layout[off + n];
        let j = pyramid_map_jacobian(verts, xi[0], xi[1], xi[2]);
        pts.push(pyramid_map(verts, xi[0], xi[1], xi[2]));
        tans.push(j_mul(&j, &tks[off + n]));
    }
    (pts, tans)
}

/// Physical `(point, tangent)` list of the prism interior slot block — the
/// element-owned `Project_ND` functionals (`t = J·t̂`).
fn prism_interior_slots(
    verts: &[[f64; 3]; 6],
    k: usize,
    layout: &[[f64; 3]],
    tks: &[[f64; 3]],
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let off = 9 * k + 8 * k * (k - 1);
    let mut pts = Vec::new();
    let mut tans = Vec::new();
    for n in off..layout.len() {
        let xi = layout[n];
        let j = prism_map_jacobian(verts, xi[0], xi[1], xi[2]);
        pts.push(prism_trilinear_map(verts, xi[0], xi[1], xi[2]));
        tans.push(j_mul(&j, &tks[n]));
    }
    (pts, tans)
}

/// Physical `(point, tangent)` list of the pyramid interior slot block
/// (`3k(k−1)²` slots).
fn pyramid_interior_slots(
    verts: &[[f64; 3]; 5],
    k: usize,
    layout: &[[f64; 3]],
    tks: &[[f64; 3]],
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let off = 8 * k + 6 * k * (k - 1);
    let mut pts = Vec::new();
    let mut tans = Vec::new();
    for n in off..layout.len() {
        let xi = layout[n];
        let j = pyramid_map_jacobian(verts, xi[0], xi[1], xi[2]);
        pts.push(pyramid_map(verts, xi[0], xi[1], xi[2]));
        tans.push(j_mul(&j, &tks[n]));
    }
    (pts, tans)
}

// ─── Hex ND face-interior DOF geometry ──────────────────────────────────────

/// Physical point and tangent of every local DOF of one hex NDk face block.
fn hex_face_slots(
    verts: &[[f64; 3]; 8],
    k: usize,
    lf: usize,
    coords: &[Vec<f64>],
    tangents: &[[f64; 3]],
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let ndf = 2 * k * (k - 1);
    let off = 12 * k + ndf * HEX_QUAD_FACE_TO_ND_BLOCK[lf];
    let mut xs = Vec::with_capacity(ndf);
    let mut ts = Vec::with_capacity(ndf);
    for n in 0..ndf {
        let xi = &coords[off + n];
        let (x, jac) = hex_trilinear_map(verts, xi);
        let tau = tangents[off + n];
        let mut t = [0.0_f64; 3];
        for d in 0..3 {
            t[d] = jac[d][0] * tau[0] + jac[d][1] * tau[1] + jac[d][2] * tau[2];
        }
        xs.push(x);
        ts.push(t);
    }
    (xs, ts)
}

/// Match one element-local face DOF `(x, t)` against a face's canonical DOF
/// list — the same physical point with a parallel (same or opposite)
/// physical tangent.  Returns `(canonical slot, sign)`.
///
/// This is the hex analogue of MFEM's `DofOrderForOrientation(QUARE, or)`
/// signed permutation: because the DOFs are nodal point-value functionals
/// with *unnormalized* tangents, an orientation change is exactly a signed
/// permutation of the face DOFs.
/// Absolute tolerance of a face-DOF point match: `1e-9` relative to the point's
/// distance from the origin (shared by [`match_face_dof`] and
/// [`nearest_face_point`]).
#[inline]
fn face_point_tol(x: [f64; 3]) -> f64 {
    1e-9 * (1.0 + (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt())
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

fn match_face_dof(
    nodes: &[[f64; 3]],
    tangents: &[[f64; 3]],
    x: [f64; 3],
    t: [f64; 3],
) -> (usize, f64) {
    match match_face_dof_soft(nodes, tangents, x, t) {
        Some(v) => v,
        None => panic!("HCurlSpace: no matching canonical hex face DOF at {x:?}"),
    }
}

/// [`match_face_dof`] without the panic: `None` when no canonical DOF sits at
/// `x` with a parallel tangent (D813-1's frame channel treats a miss as a
/// refusal — a frame that is not this facet's).
fn match_face_dof_soft(
    nodes: &[[f64; 3]],
    tangents: &[[f64; 3]],
    x: [f64; 3],
    t: [f64; 3],
) -> Option<(usize, f64)> {
    let tol = face_point_tol(x);
    let tn = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
    for (m, (an, at)) in nodes.iter().zip(tangents.iter()).enumerate() {
        let dx = x[0] - an[0];
        let dy = x[1] - an[1];
        let dz = x[2] - an[2];
        if (dx * dx + dy * dy + dz * dz).sqrt() > tol {
            continue;
        }
        let an_n = (at[0] * at[0] + at[1] * at[1] + at[2] * at[2]).sqrt();
        let c = (t[0] * at[0] + t[1] * at[1] + t[2] * at[2]) / (tn * an_n);
        if c.abs() > 1.0 - 1e-9 {
            return Some((m, if c > 0.0 { 1.0 } else { -1.0 }));
        }
    }
    None
}

/// Physical DOF points/tangents of one quad face's canonical ND DOF list,
/// fixed by the face-creating element ("Elem1") — the global functionals of
/// the face's shared DOFs (`σ_m(Φ) = Φ(x_m)·t_m`).
#[derive(Debug, Clone)]
struct QuadFaceAnchor {
    nodes: Vec<[f64; 3]>,
    tangents: Vec<[f64; 3]>,
}

/// Physical DOF points and their canonical tangent pairs on one shared
/// triangular face (Tet NDk, k≥2).
///
/// The face carries `k(k−1)/2` DOF points and two DOFs per point; the global
/// functionals are
///
/// ```text
/// σ^canon_{2p+0}(Φ) = Φ(x_p) · t[p][0]
/// σ^canon_{2p+1}(Φ) = Φ(x_p) · t[p][1]
/// ```
///
/// fixed by the face-creating element (MFEM's "Elem1") — the anchor every
/// other element's face DOFs are expressed against.
#[derive(Debug, Clone)]
pub struct TetFaceAnchor {
    pts: Vec<[f64; 3]>,
    tans: Vec<[[f64; 3]; 2]>,
}

impl TetFaceAnchor {
    /// Number of DOF points on the face (`k(k−1)/2`).
    pub fn n_points(&self) -> usize {
        self.pts.len()
    }
    /// Physical point of face-point `p`.
    pub fn point(&self, p: usize) -> [f64; 3] {
        self.pts[p]
    }
    /// The two canonical tangents at face-point `p`.
    pub fn tangents(&self, p: usize) -> [[f64; 3]; 2] {
        self.tans[p]
    }
}

/// One 2×2 face-DOF block of one element (D37).
///
/// The element-local DOF pair at slots `slot`, `slot+1` (indices into
/// [`HCurlSpace::element_dofs`]) carries the functionals `Σ_k s[n][k] ·
/// σ^canon` of the canonical pair `canon_dofs[0..2]`, i.e. `s`'s rows are the
/// element's local tangents expressed in the canonical tangent basis — MFEM's
/// `ND_DofTransformation::T(ori)` relation, which no scalar sign can express.
///
/// Consumers must apply it as `A_canon = Sᵀ·A_local·S` for the element matrix
/// and `b_canon = Sᵀ·b_local` for the load vector, because `S` maps canonical
/// DOFs to element-local ones (`u_local = S·u_canon`).  Equivalently, with
/// MFEM's `T = S⁻¹` (`v_t = T·v`, `A_t = T⁻ᵀ A T⁻¹`) this is the same
/// relation — `S` *is* the transformation MFEM calls `T⁻¹`.
#[derive(Debug, Clone, Copy)]
pub struct FaceDofBlock {
    /// First element-local DOF of the pair.
    pub slot: usize,
    /// The two canonical global DOFs the pair maps onto.
    pub canon_dofs: [DofId; 2],
    /// Rows: the element's local tangents in the canonical tangent basis.
    pub s: [[f64; 2]; 2],
}

/// Identity 2×2 block (the face-creating element's own convention).
const ID2: [[f64; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]];

/// One shared-face DOF **pair transform** of one element, in the form the
/// parallel DOF partition consumes it (D807-2).
///
/// This is the DP-facing view of one [`FaceDofBlock`]: the block's
/// element-local slot, the 2×2 map from the space's canonical (face-creating
/// element) pair onto the element's own pair (`u_local = s·u_canonical`), and
/// the precomputed, validated inverse.
///
/// The DP's global face basis is the **minimum-global-element-id** element's
/// own basis, so the transform it has to apply for the element `a` that anchors
/// a facet is exactly this element's block: `u_global = s·u_canonical`, hence
/// the dual (row/load) maps are `s⁻ᵀ` and `s⁻¹`.  The scalar
/// [`HCurlSpace::element_signs`] array is the diagonal (signed-permutation)
/// special case and is identically `+1.0` at every tri-face slot — which is why
/// the DP cannot express this relation with a scalar and needs the pair channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FacePairTransform {
    /// Element-local slot of the pair's first DOF: the pair occupies slots
    /// `slot` and `slot + 1` of `FESpace::element_dofs(elem)`.
    pub slot: u32,
    /// `u_local = s · u_canonical`.
    pub s: [[f64; 2]; 2],
    /// `s⁻¹` (panics on a singular pair, i.e. a degenerate face frame).
    pub s_inv: [[f64; 2]; 2],
}

/// Match one element-local face DOF point against the anchor's point list —
/// the shared-face physical point, with a symmetric barycentric GL set the
/// correspondence is a permutation.
fn match_face_point(anchor: &TetFaceAnchor, x: [f64; 3]) -> usize {
    let (best, best_d) = nearest_face_point(&anchor.pts, x);
    assert!(
        best_d <= face_point_tol(x),
        "HCurlSpace: no canonical face DOF point at {x:?} (closest {best_d:e})"
    );
    best
}

/// [`match_face_point`] without the assert: the nearest face point in `pts` and
/// its distance, for callers that treat a miss as a refusal rather than a panic
/// (D813-1's frame channel, where a miss means "this family's frame is not
/// face-local").
fn nearest_face_point(pts: &[[f64; 3]], x: [f64; 3]) -> (usize, f64) {
    let mut best = usize::MAX;
    let mut best_d = f64::INFINITY;
    for (p, a) in pts.iter().enumerate() {
        let d = ((x[0] - a[0]).powi(2) + (x[1] - a[1]).powi(2) + (x[2] - a[2]).powi(2)).sqrt();
        if d < best_d {
            best_d = d;
            best = p;
        }
    }
    (best, best_d)
}

/// Change of basis of one element's face tangent pair against a canonical
/// pair: returns `s` with `t_n = Σ_k s[n][k]·w_k` (in-plane least squares,
/// exact for the affine tet map), i.e. `sᵀ = (WᵀW)⁻¹ Wᵀ T`.
fn face_pair_change_of_basis(t: &[[f64; 3]; 2], w: &[[f64; 3]; 2], ctx: &str) -> [[f64; 2]; 2] {
    let dot = |a: &[f64; 3], b: &[f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let g = [[dot(&w[0], &w[0]), dot(&w[0], &w[1])], [dot(&w[1], &w[0]), dot(&w[1], &w[1])]];
    let b = [[dot(&w[0], &t[0]), dot(&w[0], &t[1])], [dot(&w[1], &t[0]), dot(&w[1], &t[1])]];
    let det = g[0][0] * g[1][1] - g[0][1] * g[1][0];
    assert!(det.abs() > 1e-30, "HCurlSpace: degenerate face tangent pair ({ctx})");
    // sᵀ = g⁻¹ b  (2×2)
    let st = [
        [(g[1][1] * b[0][0] - g[0][1] * b[1][0]) / det, (g[1][1] * b[0][1] - g[0][1] * b[1][1]) / det],
        [(-g[1][0] * b[0][0] + g[0][0] * b[1][0]) / det, (-g[1][0] * b[0][1] + g[0][0] * b[1][1]) / det],
    ];
    // s = (sᵀ)ᵀ
    [[st[0][0], st[1][0]], [st[0][1], st[1][1]]]
}

/// Physical DOF points and their two tangents of one local tet face block
/// (MFEM `ND_TetrahedronElement` `Nodes`/`dof2tk`).
///
/// `k = 2` keeps the historical pre-D38 arithmetic (`pa + (w0+w1)/3` centroid
/// with the `TET_FACE_TANGENTS` edge vectors) bit-for-bit; `k ≥ 3` pushes the
/// general TetNDk face layout through the affine element map (`x = P₀ + J·ξ`,
/// `t = J·t̂`), the same point-value functionals the element itself uses.
fn tet_face_slots<M: MeshTopology>(
    mesh: &M,
    verts: &[fem_core::types::NodeId],
    k: usize,
    f: usize,
) -> (Vec<[f64; 3]>, Vec<[[f64; 3]; 2]>) {
    // `MeshTopology::node_coords` is a slice (`len == dim`), so lift it to the
    // fixed 3-vector the tet geometry works in (every caller is 3-D).
    let p: [[f64; 3]; 4] = std::array::from_fn(|i| {
        let c = mesh.node_coords(verts[i]);
        std::array::from_fn(|d| c.get(d).copied().unwrap_or(0.0))
    });
    tet_face_slots_coords(&p, k, f)
}

/// [`tet_face_slots`] from the element's four **vertex coordinates** instead of
/// the mesh (D813-1: the anchor element's frame is rebuilt from its published
/// vertex list, so the element itself need not be local).
fn tet_face_slots_coords(
    p: &[[f64; 3]; 4],
    k: usize,
    f: usize,
) -> (Vec<[f64; 3]>, Vec<[[f64; 3]; 2]>) {
    let (la, _lb, _lc) = TET_FACES[f];
    if k == 2 {
        let a0 = p[la];
        let (p0, n0, p1, n1) = TET_FACE_TANGENTS[f];
        let g0 = p[p0];
        let h0 = p[n0];
        let g1 = p[p1];
        let h1 = p[n1];
        let w0 = [g0[0] - h0[0], g0[1] - h0[1], g0[2] - h0[2]];
        let w1 = [g1[0] - h1[0], g1[1] - h1[1], g1[2] - h1[2]];
        let pc = [
            a0[0] + (w0[0] + w1[0]) / 3.0,
            a0[1] + (w0[1] + w1[1]) / 3.0,
            a0[2] + (w0[2] + w1[2]) / 3.0,
        ];
        return (vec![pc], vec![[w0, w1]]);
    }
    let nda = TetNDk::new(k);
    let coords = nda.dof_coords();
    let tks = nda.dof_tangents();
    let nfd = 2 * (k * (k - 1) / 2);
    let base = 6 * k + f * nfd;
    let p0 = p[0];
    let mut jac = [[0.0_f64; 3]; 3];
    for (c, lv) in [1usize, 2, 3].iter().enumerate() {
        let q = p[*lv];
        for d in 0..3 {
            jac[d][c] = q[d] - p0[d];
        }
    }
    let mut pts = Vec::with_capacity(nfd / 2);
    let mut tans = Vec::with_capacity(nfd / 2);
    for i in 0..nfd / 2 {
        let xi = &coords[base + 2 * i];
        let mut x = [p0[0], p0[1], p0[2]];
        let mut t = [[0.0_f64; 3]; 2];
        for r in 0..3 {
            x[r] += jac[r][0] * xi[0] + jac[r][1] * xi[1] + jac[r][2] * xi[2];
            for c in 0..2 {
                let tk = tks[base + 2 * i + c];
                t[c][r] = jac[r][0] * tk[0] + jac[r][1] * tk[1] + jac[r][2] * tk[2];
            }
        }
        pts.push(x);
        tans.push(t);
    }
    (pts, tans)
}

/// Local base quad face for pyramid.
pub(crate) const PYRAMID_QUAD_FACE: [(usize, usize, usize, usize); 1] = [
    (0, 1, 2, 3),
];

// ─── D813-1: a facet frame from the anchor element's published vertices ──────

/// Sentinel coordinate for an anchor vertex slot its facet frame must **not**
/// read (D813-1).  Every family's facet frame is face-local, so the anchor's
/// off-facet vertices never enter; filling them with a far-away sentinel turns a
/// family that is *not* face-local into a failed DOF match (a loud refusal)
/// instead of a silently wrong global face basis.
const OFF_FACET_SENTINEL: [f64; 3] = [1.0e7, -1.0e7, 1.0e7];

/// One facet block of one 3-D element family, in the order `HCurlSpace` fills
/// the element's DOF slot table (`build`'s Pass 3): `TET_FACES` for tetrahedra,
/// `HEX_ND_BLOCK_TO_QUAD_FACE` over `HEX_QUAD_FACES` for hexahedra, prism
/// triangles-then-quads, pyramid base-quad-then-apex-triangles.
#[derive(Debug, Clone)]
struct FacetBlockDesc {
    /// Element-local vertex slots of the facet.
    slots: Vec<usize>,
    /// Index the family's geometry helper takes (`TET_FACES` / `PRISM_TRI_FACES`
    /// / `PRISM_QUAD_FACES` / `PYRAMID_TRI_FACES` entry, or a `HEX_QUAD_FACES`
    /// index).
    family_index: usize,
    /// Triangular facet (two DOFs per face point, a 2×2 pair relation) vs
    /// quadrilateral (one DOF per face point, a signed permutation).
    tri: bool,
    /// First DOF of the block inside `FESpace::element_dofs(e)`.
    dof_offset: usize,
}

/// Facet blocks of one 3-D `NDk` (`k ≥ 2`) element family, in element-slot
/// order.  `None` for every family the channel does not implement (2-D types
/// and the variable-node `Polygon`).
fn facet_block_table(et: ElementType, k: usize) -> Option<Vec<FacetBlockDesc>> {
    let ndf = k * (k - 1); // tri face: k(k−1)/2 points × 2 DOFs
    let ndfq = 2 * k * (k - 1); // quad face: one DOF per layout point
    let mk = |slots: &[usize], family_index: usize, tri: bool, dof_offset: usize| {
        FacetBlockDesc { slots: slots.to_vec(), family_index, tri, dof_offset }
    };
    match et {
        ElementType::Tet4 | ElementType::Tet10 => Some(
            TET_FACES
                .iter()
                .enumerate()
                .map(|(f, c)| mk(&[c.0, c.1, c.2], f, true, 6 * k + f * ndf))
                .collect(),
        ),
        ElementType::Hex8 | ElementType::Hex20 => Some(
            HEX_ND_BLOCK_TO_QUAD_FACE
                .iter()
                .enumerate()
                .map(|(i, &lf)| {
                    let c = HEX_QUAD_FACES[lf];
                    mk(&[c.0, c.1, c.2, c.3], lf, false, 12 * k + i * ndfq)
                })
                .collect(),
        ),
        ElementType::Prism6 => Some(
            PRISM_TRI_FACES
                .iter()
                .enumerate()
                .map(|(f, c)| mk(&[c.0, c.1, c.2], f, true, 9 * k + f * ndf))
                .chain(PRISM_QUAD_FACES.iter().enumerate().map(|(f, c)| {
                    mk(&[c.0, c.1, c.2, c.3], f, false, 9 * k + 2 * ndf + f * ndfq)
                }))
                .collect(),
        ),
        ElementType::Pyramid5 => {
            let c = PYRAMID_QUAD_FACE[0];
            Some(
                std::iter::once(mk(&[c.0, c.1, c.2, c.3], 0, false, 8 * k))
                    .chain(PYRAMID_TRI_FACES.iter().enumerate().map(|(f, c)| {
                        mk(&[c.0, c.1, c.2], f, true, 8 * k + ndfq + f * ndf)
                    }))
                    .collect(),
            )
        }
        _ => None,
    }
}

/// The facet's block index inside one element's `facet_block_table`, located by
/// the facet's **global** vertex set (the vertex ids the element's own slots
/// carry, e.g. through `MeshPartition::global_node`).
fn find_facet_block(
    blocks: &[FacetBlockDesc],
    elem_verts: &[u32],
    facet_verts: &[u32],
) -> Option<usize> {
    blocks.iter().position(|b| {
        b.slots.len() == facet_verts.len()
            && b.slots
                .iter()
                .all(|&s| elem_verts.get(s).is_some_and(|v| facet_verts.contains(v)))
    })
}

/// Physical DOF geometry of one facet block (see [`facet_block_geometry`]).
enum FacetGeom {
    /// Quadrilateral facet: one point and one tangent per DOF.
    Quad { nodes: Vec<[f64; 3]>, tangents: Vec<[f64; 3]> },
    /// Triangular facet: one point and two tangents per DOF pair.
    Tri { pts: Vec<[f64; 3]>, tans: Vec<[[f64; 3]; 2]> },
}

/// Physical DOF points/tangents of one facet block, from the element's vertex
/// coordinates **in slot order** — the same helpers `build`'s Pass 2/3 use, so
/// an element-local call reproduces the space's own slot geometry exactly.
fn facet_block_geometry(
    et: ElementType,
    verts: &[[f64; 3]],
    k: usize,
    block: &FacetBlockDesc,
) -> Option<FacetGeom> {
    match et {
        ElementType::Tet4 | ElementType::Tet10 => {
            let p: [[f64; 3]; 4] = verts.try_into().ok()?;
            let (pts, tans) = tet_face_slots_coords(&p, k, block.family_index);
            Some(FacetGeom::Tri { pts, tans })
        }
        ElementType::Hex8 | ElementType::Hex20 => {
            let p: [[f64; 3]; 8] = verts.try_into().ok()?;
            let hnd = HexNDk::new(k);
            let (nodes, tangents) = hex_face_slots(
                &p,
                k,
                block.family_index,
                &hnd.dof_coords(),
                &hnd.dof_tangents(),
            );
            Some(FacetGeom::Quad { nodes, tangents })
        }
        ElementType::Prism6 => {
            let p: [[f64; 3]; 6] = verts.try_into().ok()?;
            let layout = PrismNDk::new(k).mfem_layout_points();
            let tks = prism_tangents_mfem(k);
            if block.tri {
                let (pts, tans) = prism_tri_face_slots(&p, k, block.family_index, &layout, &tks);
                Some(FacetGeom::Tri { pts, tans })
            } else {
                let (nodes, tangents) =
                    prism_quad_face_slots(&p, k, block.family_index, &layout, &tks);
                Some(FacetGeom::Quad { nodes, tangents })
            }
        }
        ElementType::Pyramid5 => {
            let p: [[f64; 3]; 5] = verts.try_into().ok()?;
            let layout = PyraNDk::new(k).mfem_layout_points();
            let tks = PyraNDk::new(k).dof_tangents();
            if block.tri {
                let (pts, tans) =
                    pyramid_tri_face_slots(&p, k, block.family_index, &layout, &tks);
                Some(FacetGeom::Tri { pts, tans })
            } else {
                let (nodes, tangents) = pyramid_quad_face_slots(&p, k, &layout, &tks);
                Some(FacetGeom::Quad { nodes, tangents })
            }
        }
        _ => None,
    }
}

/// D813-1: one DOF of a *local* element's facet block, expressed against the
/// **published anchor element's** own block (see
/// [`HCurlSpace::facet_slots_against_published_anchor`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FacetSlotToAnchor {
    /// Index of the DOF inside `FESpace::element_dofs(e)` of the local element.
    pub slot: usize,
    /// Index of the same physical DOF inside the anchor element's own facet
    /// block — the parallel DOF partition's face-DOF label `pos`.
    pub anchor_slot: usize,
    /// The anchor element's own `element_signs` value at `anchor_slot`: the
    /// partition-level sign of D122-3 (always `+1.0` on a triangular facet,
    /// whose rotation is carried by [`Self::anchor_pair`]).
    pub anchor_sign: f64,
    /// Triangular facets: the anchor's own 2×2 pair transform at the matched
    /// face point (`u_anchor = s·u_canonical`, the same object
    /// [`HCurlSpace::element_face_pair_transforms`] returns for the anchor
    /// element) with `slot` set to the pair's first element-local slot **of the
    /// local element `e`**, so the caller permutes `e`'s own two DOFs.
    /// `None` on quadrilateral facets, whose relation is a signed permutation
    /// carried by [`Self::anchor_sign`].
    pub anchor_pair: Option<FacePairTransform>,
}

// ─── HCurlSpace ─────────────────────────────────────────────────────────────

/// H(curl) finite element space using Nédélec edge elements.
///
/// Constructed from a [`MeshTopology`] with triangular or tetrahedral elements.
/// Currently supports order 1 (ND1).
// MFEM: ND_FECollection (Nedelec)
pub struct HCurlSpace<M: MeshTopology> {
    mesh: M,
    order: u8,
    n_dofs: usize,
    /// Flat global DOF indices: `[elem0_dof0, elem0_dof1, ..., elem1_dof0, ...]`
    dofs_flat: Vec<DofId>,
    /// Orientation signs (±1.0), same layout as `dofs_flat`.
    signs_flat: Vec<f64>,
    /// CSR-like offsets into `dofs_flat` / `signs_flat`, length `n_elems + 1`.
    /// `dofs_flat[offsets[e]..offsets[e+1]]` are the DOFs for element `e`.
    elem_offsets: Vec<usize>,
    /// Edge → global DOF map (for boundary queries and interpolation).
    edge_to_dof: HashMap<EdgeKey, DofId>,
    /// Face → first global DOF map for 3D ND2 (second = first + 1).
    face_to_dof: HashMap<FaceKey, DofId>,
    /// Face → canonical (shared) DOF functional anchor for 3-D NDk
    /// (`k(k−1)/2` point-value pairs on tets; `k(k−1)` slots on prism /
    /// pyramid tri faces), fixed by the face-creating element.
    face_anchor: HashMap<FaceKey, TetFaceAnchor>,
    /// Per element: the 2×2 face-DOF block transforms into the canonical
    /// (face-creating element) basis — empty for spaces without shared face
    /// DOF pairs (2-D, hex, k = 1).  See [`FaceDofBlock`].
    elem_face_blocks: Vec<Vec<FaceDofBlock>>,
    /// Quad-face → first global DOF for hex NDk (2k(k-1) DOFs per face).
    quad_face_to_dof: HashMap<QuadFaceKey, DofId>,
    /// Quad-face → physical DOF points/tangents (`σ_m(Φ) = Φ(x_m)·t_m`) of
    /// the face's canonical DOF list, fixed by the face-creating element
    /// (hex D505, prism/pyramid D546).
    quad_face_anchor: HashMap<QuadFaceKey, QuadFaceAnchor>,
    /// Spatial dimension.
    dim: usize,
    /// Cell type used by this space.
    cell_type: ElementType,
    /// 2-D quad basis variant: `true` = MFEM's
    /// `ND_FECollection(o, 2, GaussLobatto, IntegratedGLL)` collection (the
    /// LOR-compatible basis pair, built by
    /// [`Self::new_gauss_lobatto_integrated_gll`]); `false` = the library
    /// default (`HCurlSpace::new`, GaussLegendre open modes).  The variant
    /// selects which reference element [`fem_assembly::VectorAssembler`]
    /// pairs with the (identical) dof/slot tables — see
    /// `quad_integrated_gll`.
    quad_igll: bool,
}

impl<M: MeshTopology> HCurlSpace<M> {
    /// Construct an H(curl) space of the given order on `mesh`.
    ///
    /// Supports ND1 (order 1) and NDk (order k >= 2) for Tri3/Tri6, Quad4/Quad8, Tet4/Tet10, Hex8/Hex20.
    pub fn new(mesh: M, order: u8) -> Self {
        Self::build(mesh, order, false)
    }

    /// Construct the 2-D **quad** H(curl) space of MFEM's LOR-compatible
    /// collection `ND_FECollection(order, 2, BasisType::GaussLobatto,
    /// BasisType::IntegratedGLL)` (`fem/fe_coll.hpp`).
    ///
    /// The global DOF numbering, slot tables and orientation signs are
    /// **identical** to [`Self::new`] (MFEM's per-edge
    /// `DofOrderForOrientation` rule does not depend on the 1-D basis); the
    /// variant changes
    ///
    /// * which reference element the assembler pairs with the tables — the
    ///   faithful `fem_element::nedelec::QuadND::new_integrated_gll`
    ///   (integrated Gerritsma open modes) instead of the legacy
    ///   GaussLegendre elements — via
    ///   [`fem_assembly::VectorAssembler::assemble_bilinear_quad_igll`], and
    /// * [`Self::interpolate_vector`], which becomes MFEM's
    ///   `ProjectIntegrated` (sub-cell line integrals) instead of point
    ///   values, because `ND_QuadrilateralElement::Project` dispatches to
    ///   `ProjectIntegrated` for the integrated type (`fe_nd.hpp:68`).
    ///
    /// On non-quad meshes the two constructors build the same space.
    pub fn new_gauss_lobatto_integrated_gll(mesh: M, order: u8) -> Self {
        Self::build(mesh, order, true)
    }

    /// Whether this space carries the `(GaussLobatto, IntegratedGLL)` quad
    /// basis pair.  The assembler and LOR entry points must key off the same
    /// variant as the space (D347 pattern): a variant space must be assembled
    /// with the `*_quad_igll` entries, a default space with the plain ones.
    pub fn quad_integrated_gll(&self) -> bool {
        self.quad_igll
    }

    fn build(mesh: M, order: u8, quad_igll: bool) -> Self {
        assert!(order >= 1, "HCurlSpace: order must be >= 1");
        let dim = mesh.dim() as usize;
        // D786: an empty local mesh (np > n_elements over-partitioning) must
        // construct a valid empty space instead of panicking — the entity
        // loops below simply run zero times.
        let k = order as usize;
        let dofs_per_edge = k;
        let n_elem = mesh.n_elements();
        let first_cell_type = (n_elem > 0)
            .then(|| mesh.element_type(0))
            .unwrap_or(ElementType::Hex8);

        let mut edge_to_dof: HashMap<EdgeKey, DofId> = HashMap::new();
        let mut face_to_dof: HashMap<FaceKey, DofId> = HashMap::new();
        let mut face_anchor: HashMap<FaceKey, TetFaceAnchor> = HashMap::new();
        let mut elem_face_blocks: Vec<Vec<FaceDofBlock>> = Vec::with_capacity(n_elem);
        let mut quad_face_to_dof: HashMap<QuadFaceKey, DofId> = HashMap::new();
        let mut quad_face_anchor: HashMap<QuadFaceKey, QuadFaceAnchor> = HashMap::new();

        // D158: MFEM's global numbering is **entity-major** — all edge DOFs
        // (mesh-edge index order = first-encounter order), then all face DOFs
        // (first-encounter face order), then the element-interior DOFs (in
        // element order).  The previous single pass assigned ids while walking
        // the elements, interleaving each element's faces/interiors right
        // after its edges, so coefficient vectors exchanged through VisIt DC
        // files were read with the wrong global ids.  Passes 1/2 enumerate
        // the entities; pass 3 fills the per-element slot tables.
        //
        // Pass 1: edges.
        let mut next_dof: DofId = 0;
        for e in 0..n_elem as u32 {
            let cell_type = mesh.element_type(e);
            let verts = mesh.element_nodes(e);
            let local_edges: &[(usize, usize)] = match cell_type {
                ElementType::Tri3 | ElementType::Tri6 => {
                    if k >= 2 {
                        &TRI_EDGES_MFEM
                    } else {
                        &TRI_EDGES_ND1
                    }
                }
                ElementType::Quad4 | ElementType::Quad8 => &QUAD_EDGES,
                ElementType::Tet4 | ElementType::Tet10 => &TET_EDGES,
                ElementType::Hex8 | ElementType::Hex20 => &HEX_EDGES,
                ElementType::Prism6 => &PRISM_EDGES,
                ElementType::Pyramid5 => &PYRAMID_EDGES,
                _ => panic!("HCurlSpace: unsupported element type {cell_type:?}"),
            };
            for &(li, lj) in local_edges {
                let (gi, gj) = (verts[li], verts[lj]);
                let key = EdgeKey::new(gi, gj);
                if let std::collections::hash_map::Entry::Vacant(vac) = edge_to_dof.entry(key) {
                    vac.insert(next_dof);
                    next_dof += dofs_per_edge as DofId;
                }
            }
        }
        let n_edge_dofs: DofId = next_dof;

        // Pass 2: faces (NDk, k >= 2, 3-D).  Register every unique face and
        // fix its canonical anchor from the face-creating element.
        let mut face_creators: std::collections::HashSet<(u32, usize)> =
            std::collections::HashSet::new();
        if k >= 2 && dim == 3 {
            let ndf = k * (k - 1);
            let (hex_coords, hex_tks) = {
                let hnd = HexNDk::new(k);
                (hnd.dof_coords(), hnd.dof_tangents())
            };
            // MFEM nodal layouts (`FE::Nodes`) of the wedge / Fuentes pyramid,
            // used to anchor the prism/pyramid face blocks (D525), and the
            // matching MFEM-frame slot tangents (D546/D547).
            let prism_layout = PrismNDk::new(k).mfem_layout_points();
            let pyra_layout = PyraNDk::new(k).mfem_layout_points();
            let prism_tks = prism_tangents_mfem(k);
            let pyra_tks = PyraNDk::new(k).dof_tangents();
            for e in 0..n_elem as u32 {
                let cell_type = mesh.element_type(e);
                let verts = mesh.element_nodes(e);
                match cell_type {
                    ElementType::Tet4 | ElementType::Tet10 => {
                        let nfp = k * (k - 1) / 2; // points per face
                        let nfd = 2 * nfp; // dofs per face
                        for (f, &(la, lb, lc)) in TET_FACES.iter().enumerate() {
                            let key = FaceKey::new(verts[la], verts[lb], verts[lc]);
                            if face_to_dof.contains_key(&key) {
                                continue;
                            }
                            face_to_dof.insert(key, next_dof);
                            next_dof += nfd as DofId;
                            face_creators.insert((e, f));
                            let (pts, tans) = tet_face_slots(&mesh, verts, k, f);
                            face_anchor.insert(
                                key,
                                TetFaceAnchor { pts: pts.clone(), tans: tans.clone() },
                            );
                        }
                    }
                    ElementType::Hex8 | ElementType::Hex20 => {
                        let ndf_quad = 2 * k * (k - 1);
                        let verts8 = hex8_verts(&mesh, e);
                        // D225: register in MFEM `FaceVert` block order so the
                        // global face-dof ranges follow MFEM's numbering.
                        for &lf in HEX_ND_BLOCK_TO_QUAD_FACE.iter() {
                            let (la, lb, lc, ld) = HEX_QUAD_FACES[lf];
                            let key = QuadFaceKey::new(verts[la], verts[lb], verts[lc], verts[ld]);
                            if quad_face_to_dof.contains_key(&key) {
                                continue;
                            }
                            quad_face_to_dof.insert(key, next_dof);
                            next_dof += ndf_quad as DofId;
                            face_creators.insert((e, lf));
                            let (xs, ts) = hex_face_slots(&verts8, k, lf, &hex_coords, &hex_tks);
                            quad_face_anchor.insert(
                                key,
                                QuadFaceAnchor { nodes: xs, tangents: ts },
                            );
                        }
                    }
                    ElementType::Prism6 => {
                        let verts6 = prism6_verts(&mesh, e);
                        for (f, &(la, lb, lc)) in PRISM_TRI_FACES.iter().enumerate() {
                            let key = FaceKey::new(verts[la], verts[lb], verts[lc]);
                            if face_to_dof.contains_key(&key) {
                                continue;
                            }
                            face_to_dof.insert(key, next_dof);
                            next_dof += ndf as DofId;
                            face_creators.insert((e, f));
                            let (pts, tans) =
                                prism_tri_face_slots(&verts6, k, f, &prism_layout, &prism_tks);
                            face_anchor.insert(
                                key,
                                TetFaceAnchor { pts: pts.clone(), tans: tans.clone() },
                            );
                        }
                        let ndf_quad = 2 * k * (k - 1);
                        for (f, &(la, lb, lc, ld)) in PRISM_QUAD_FACES.iter().enumerate() {
                            let key = QuadFaceKey::new(verts[la], verts[lb], verts[lc], verts[ld]);
                            if quad_face_to_dof.contains_key(&key) {
                                continue;
                            }
                            quad_face_to_dof.insert(key, next_dof);
                            next_dof += ndf_quad as DofId;
                            face_creators.insert((e, 100 + f));
                            let (xs, ts) =
                                prism_quad_face_slots(&verts6, k, f, &prism_layout, &prism_tks);
                            quad_face_anchor.insert(
                                key,
                                QuadFaceAnchor { nodes: xs, tangents: ts },
                            );
                        }
                    }
                    ElementType::Pyramid5 => {
                        let verts5 = pyramid5_verts(&mesh, e);
                        // Registration order = MFEM's mesh face numbering
                        // (`FaceVert` walk: the base quad is the element's
                        // first face, then the four apex tris) so the global
                        // face-dof blocks line up with MFEM's (D525).
                        let ndf_quad = 2 * k * (k - 1);
                        let (la, lb, lc, ld) = PYRAMID_QUAD_FACE[0];
                        let key = QuadFaceKey::new(verts[la], verts[lb], verts[lc], verts[ld]);
                        if !quad_face_to_dof.contains_key(&key) {
                            quad_face_to_dof.insert(key, next_dof);
                            next_dof += ndf_quad as DofId;
                            face_creators.insert((e, 100));
                            let (xs, ts) =
                                pyramid_quad_face_slots(&verts5, k, &pyra_layout, &pyra_tks);
                            quad_face_anchor.insert(
                                key,
                                QuadFaceAnchor { nodes: xs, tangents: ts },
                            );
                        }
                        for (f, &(la, lb, lc)) in PYRAMID_TRI_FACES.iter().enumerate() {
                            let key = FaceKey::new(verts[la], verts[lb], verts[lc]);
                            if face_to_dof.contains_key(&key) {
                                continue;
                            }
                            face_to_dof.insert(key, next_dof);
                            next_dof += ndf as DofId;
                            face_creators.insert((e, f));
                            let (pts, tans) =
                                pyramid_tri_face_slots(&verts5, k, f, &pyra_layout, &pyra_tks);
                            face_anchor.insert(
                                key,
                                TetFaceAnchor { pts: pts.clone(), tans: tans.clone() },
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
        let n_face_dofs: DofId = next_dof - n_edge_dofs;
        let interior_base: DofId = n_edge_dofs + n_face_dofs;

        // Total interior count for the final `n_dofs`.
        let total_interior: u32 = (0..n_elem as u32)
            .map(|e| {
                let cell_type = mesh.element_type(e);
                match (dim, cell_type) {
                    (2, ElementType::Tri3 | ElementType::Tri6) if k >= 2 => (k * (k - 1)) as u32,
                    (2, ElementType::Quad4 | ElementType::Quad8) if k >= 2 => {
                        (2 * k * (k - 1)) as u32
                    }
                    (3, ElementType::Tet4 | ElementType::Tet10) if k >= 3 => {
                        (k * (k - 1) * (k - 2) / 2) as u32
                    }
                    (3, ElementType::Hex8 | ElementType::Hex20) if k >= 2 => {
                        (3 * k * (k - 1) * (k - 1)) as u32
                    }
                    (3, ElementType::Prism6) if k >= 2 => {
                        // MFEM `ND_WedgeElement`: `p(p−1)²` tri⊗layer slots +
                        // `p(p−1)(p−2)/2` vertical slots (D525).
                        ((k * (k - 1) * (k - 1)) + (k * (k - 1) * (k - 2)) / 2) as u32
                    }
                    (3, ElementType::Pyramid5) if k >= 2 => {
                        // MFEM `ND_FuentesPyramidElement`: `3p(p−1)²` (D525).
                        (3 * k * (k - 1) * (k - 1)) as u32
                    }
                    _ => 0,
                }
            })
            .sum();
        let n_dofs_total: DofId = interior_base + total_interior as DofId;

        // Pass 3: per-element slot tables.
        let mut dofs_flat = Vec::new();
        let mut signs_flat = Vec::new();
        let mut elem_offsets = Vec::with_capacity(n_elem + 1);
        elem_offsets.push(0);
        let (hex_coords, hex_tks) = {
            let hnd = HexNDk::new(k);
            (hnd.dof_coords(), hnd.dof_tangents())
        };
        let mut interior_cursor: DofId = interior_base;
        for e in 0..n_elem as u32 {
            let cell_type = mesh.element_type(e);
            let verts = mesh.element_nodes(e);
            let mut elem_blocks: Vec<FaceDofBlock> = Vec::new();

            // Per-element-type local edges.
            let local_edges: &[(usize, usize)] = match cell_type {
                ElementType::Tri3 | ElementType::Tri6 => {
                    if k >= 2 {
                        &TRI_EDGES_MFEM
                    } else {
                        &TRI_EDGES_ND1
                    }
                }
                ElementType::Quad4 | ElementType::Quad8 => &QUAD_EDGES,
                ElementType::Tet4 | ElementType::Tet10 => &TET_EDGES,
                ElementType::Hex8 | ElementType::Hex20 => &HEX_EDGES,
                ElementType::Prism6 => &PRISM_EDGES,
                ElementType::Pyramid5 => &PYRAMID_EDGES,
                _ => panic!("HCurlSpace: unsupported element type {cell_type:?}"),
            };

            // Edge DOFs.
            //
            // Canonical edge = (min vertex, max vertex); canonical slot j sits
            // at the j-th Gauss point along (min→max).  A local slot m sits at
            // the m-th Gauss point along the LOCAL pair direction, hence:
            // aligned (gi < gj): slot m → global first+m, sign +1;
            // reversed:          slot m → global first+(k−1−m), sign −1
            // (MFEM encodes edge reversal exactly like this: reversed dof
            // order + sign −1).
            for &(li, lj) in local_edges {
                let (gi, gj) = (verts[li], verts[lj]);
                let key = EdgeKey::new(gi, gj);
                let aligned = gi < gj;
                let sign = if aligned { 1.0 } else { -1.0 };
                let nd = dofs_per_edge as usize;
                let first_dof = edge_to_dof[&key];
                for m in 0..nd {
                    let slot = if aligned { m } else { nd - 1 - m };
                    dofs_flat.push(first_dof + slot as DofId);
                    signs_flat.push(sign);
                }
            }

            // Face DOFs (NDk, k>=2).
            if k >= 2 && dim == 3 {
                let ndf = k * (k - 1);
                match cell_type {
                    ElementType::Tet4 | ElementType::Tet10 => {
                        // Nodal face DOFs (D38): the element's local face
                        // slots are the point-value functionals at the TetNDk
                        // face points (`k(k−1)/2` points × 2 tangents).  The
                        // face-creating element fixes the *canonical* shared
                        // functional list; every other element records, per
                        // face point, the 2×2 change of basis into that list
                        // (D37) — a full matrix, not a scalar sign.
                        let nfp = k * (k - 1) / 2; // points per face
                        let nfd = 2 * nfp; // dofs per face
                        let fbase = 6 * k; // local slot of the first face dof
                        for (f, &(la, lb, lc)) in TET_FACES.iter().enumerate() {
                            let key = FaceKey::new(verts[la], verts[lb], verts[lc]);
                            let first_dof = face_to_dof[&key];
                            let (pts, tans) = tet_face_slots(&mesh, verts, k, f);
                            if face_creators.contains(&(e, f)) {
                                for i in 0..nfp {
                                    elem_blocks.push(FaceDofBlock {
                                        slot: fbase + f * nfd + 2 * i,
                                        canon_dofs: [first_dof + 2 * i as u32, first_dof + 2 * i as u32 + 1],
                                        s: ID2,
                                    });
                                    dofs_flat.push(first_dof + 2 * i as u32);
                                    signs_flat.push(1.0);
                                    dofs_flat.push(first_dof + 2 * i as u32 + 1);
                                    signs_flat.push(1.0);
                                }
                            } else {
                                let anchor = &face_anchor[&key];
                                for i in 0..nfp {
                                    let p = match_face_point(anchor, pts[i]);
                                    let w = anchor.tangents(p);
                                    let s = face_pair_change_of_basis(
                                        &tans[i],
                                        &w,
                                        "tet face block transform",
                                    );
                                    elem_blocks.push(FaceDofBlock {
                                        slot: fbase + f * nfd + 2 * i,
                                        canon_dofs: [
                                            first_dof + 2 * p as u32,
                                            first_dof + 2 * p as u32 + 1,
                                        ],
                                        s,
                                    });
                                    dofs_flat.push(first_dof + 2 * p as u32);
                                    signs_flat.push(1.0);
                                    dofs_flat.push(first_dof + 2 * p as u32 + 1);
                                    signs_flat.push(1.0);
                                }
                            }
                        }
                    }
                    ElementType::Hex8 | ElementType::Hex20 => {
                        let ndf_quad = 2 * k * (k - 1);
                        let verts8 = hex8_verts(&mesh, e);
                        // D225: the element's face slots are grouped in MFEM
                        // `FaceVert` block order (`HEX_QUAD_FACE_TO_ND_BLOCK`
                        // maps this loop's block index to the `HEX_QUAD_FACES`
                        // entry), so the slot table is filled in block order.
                        for &lf in HEX_ND_BLOCK_TO_QUAD_FACE.iter() {
                            let (la, lb, lc, ld) = HEX_QUAD_FACES[lf];
                            let key = QuadFaceKey::new(verts[la], verts[lb], verts[lc], verts[ld]);
                            let first_dof = quad_face_to_dof[&key];
                            let (xs, ts) = hex_face_slots(&verts8, k, lf, &hex_coords, &hex_tks);
                            if face_creators.contains(&(e, lf)) {
                                for m in 0..ndf_quad {
                                    dofs_flat.push(first_dof + m as DofId);
                                    signs_flat.push(1.0);
                                }
                            } else {
                                // MFEM `DofOrderForOrientation`: the shared
                                // face's DOFs are the canonical
                                // (creating-element) list, re-indexed by the
                                // orientation of this element's local face
                                // cycle with the sign of the covariant
                                // tangent alignment.
                                let anchor = &quad_face_anchor[&key];
                                for n in 0..ndf_quad {
                                    let (m, s) = match_face_dof(
                                        &anchor.nodes,
                                        &anchor.tangents,
                                        xs[n],
                                        ts[n],
                                    );
                                    dofs_flat.push(first_dof + m as DofId);
                                    signs_flat.push(s);
                                }
                            }
                        }
                    }
                    ElementType::Prism6 => {
                        // Tri face DOFs (D546): the element's local face
                        // slots are the point-value functionals at the
                        // wedge-layout face points with the layout tangents;
                        // the face-creating element fixes the canonical list
                        // and every other element records the 2×2 change of
                        // basis into it (MFEM `ND_DofTransformation`).
                        let layout = PrismNDk::new(k).mfem_layout_points();
                        let tks = prism_tangents_mfem(k);
                        let verts6 = prism6_verts(&mesh, e);
                        let fbase = 9 * k;
                        for (f, &(la, lb, lc)) in PRISM_TRI_FACES.iter().enumerate() {
                            let key = FaceKey::new(verts[la], verts[lb], verts[lc]);
                            let first_dof = face_to_dof[&key];
                            let (pts, tans) =
                                prism_tri_face_slots(&verts6, k, f, &layout, &tks);
                            if face_creators.contains(&(e, f)) {
                                for i in 0..pts.len() {
                                    elem_blocks.push(FaceDofBlock {
                                        slot: fbase + f * ndf + 2 * i,
                                        canon_dofs: [
                                            first_dof + 2 * i as u32,
                                            first_dof + 2 * i as u32 + 1,
                                        ],
                                        s: ID2,
                                    });
                                    dofs_flat.push(first_dof + 2 * i as u32);
                                    signs_flat.push(1.0);
                                    dofs_flat.push(first_dof + 2 * i as u32 + 1);
                                    signs_flat.push(1.0);
                                }
                            } else {
                                let anchor = &face_anchor[&key];
                                for i in 0..pts.len() {
                                    let p = match_face_point(anchor, pts[i]);
                                    let w = anchor.tangents(p);
                                    let s = face_pair_change_of_basis(
                                        &tans[i],
                                        &w,
                                        "prism tri face block transform",
                                    );
                                    elem_blocks.push(FaceDofBlock {
                                        slot: fbase + f * ndf + 2 * i,
                                        canon_dofs: [
                                            first_dof + 2 * p as u32,
                                            first_dof + 2 * p as u32 + 1,
                                        ],
                                        s,
                                    });
                                    dofs_flat.push(first_dof + 2 * p as u32);
                                    signs_flat.push(1.0);
                                    dofs_flat.push(first_dof + 2 * p as u32 + 1);
                                    signs_flat.push(1.0);
                                }
                            }
                        }
                        // Quad face DOFs (D546): the shared face's DOFs are
                        // the canonical list re-indexed by the orientation of
                        // this element's local face cycle — the geometric
                        // form of MFEM's `ND_FECollection::QuadDofOrd[ori]`
                        // signed permutation.
                        let ndf_quad = 2 * k * (k - 1);
                        for (f, &(la, lb, lc, ld)) in PRISM_QUAD_FACES.iter().enumerate() {
                            let key =
                                QuadFaceKey::new(verts[la], verts[lb], verts[lc], verts[ld]);
                            let first_dof = quad_face_to_dof[&key];
                            let (xs, ts) =
                                prism_quad_face_slots(&verts6, k, f, &layout, &tks);
                            if face_creators.contains(&(e, 100 + f)) {
                                for m in 0..ndf_quad {
                                    dofs_flat.push(first_dof + m as DofId);
                                    signs_flat.push(1.0);
                                }
                            } else {
                                let anchor = &quad_face_anchor[&key];
                                for n in 0..ndf_quad {
                                    let (m, s) = match_face_dof(
                                        &anchor.nodes,
                                        &anchor.tangents,
                                        xs[n],
                                        ts[n],
                                    );
                                    dofs_flat.push(first_dof + m as DofId);
                                    signs_flat.push(s);
                                }
                            }
                        }
                    }
                    ElementType::Pyramid5 => {
                        let layout = PyraNDk::new(k).mfem_layout_points();
                        let tks = PyraNDk::new(k).dof_tangents();
                        let verts5 = pyramid5_verts(&mesh, e);
                        // Base quad face (registered as local face 100).
                        let ndf_quad = 2 * k * (k - 1);
                        {
                            let (la, lb, lc, ld) = PYRAMID_QUAD_FACE[0];
                            let key = QuadFaceKey::new(verts[la], verts[lb], verts[lc], verts[ld]);
                            let first_dof = quad_face_to_dof[&key];
                            let (xs, ts) = pyramid_quad_face_slots(&verts5, k, &layout, &tks);
                            if face_creators.contains(&(e, 100)) {
                                for m in 0..ndf_quad {
                                    dofs_flat.push(first_dof + m as DofId);
                                    signs_flat.push(1.0);
                                }
                            } else {
                                let anchor = &quad_face_anchor[&key];
                                for n in 0..ndf_quad {
                                    let (m, s) = match_face_dof(
                                        &anchor.nodes,
                                        &anchor.tangents,
                                        xs[n],
                                        ts[n],
                                    );
                                    dofs_flat.push(first_dof + m as DofId);
                                    signs_flat.push(s);
                                }
                            }
                        }
                        // Apex tri faces.
                        let fbase = 8 * k + 2 * k * (k - 1);
                        for (f, &(la, lb, lc)) in PYRAMID_TRI_FACES.iter().enumerate() {
                            let key = FaceKey::new(verts[la], verts[lb], verts[lc]);
                            let first_dof = face_to_dof[&key];
                            let (pts, tans) =
                                pyramid_tri_face_slots(&verts5, k, f, &layout, &tks);
                            if face_creators.contains(&(e, f)) {
                                for i in 0..pts.len() {
                                    elem_blocks.push(FaceDofBlock {
                                        slot: fbase + f * ndf + 2 * i,
                                        canon_dofs: [
                                            first_dof + 2 * i as u32,
                                            first_dof + 2 * i as u32 + 1,
                                        ],
                                        s: ID2,
                                    });
                                    dofs_flat.push(first_dof + 2 * i as u32);
                                    signs_flat.push(1.0);
                                    dofs_flat.push(first_dof + 2 * i as u32 + 1);
                                    signs_flat.push(1.0);
                                }
                            } else {
                                let anchor = &face_anchor[&key];
                                for i in 0..pts.len() {
                                    let p = match_face_point(anchor, pts[i]);
                                    let w = anchor.tangents(p);
                                    let s = face_pair_change_of_basis(
                                        &tans[i],
                                        &w,
                                        "pyramid tri face block transform",
                                    );
                                    elem_blocks.push(FaceDofBlock {
                                        slot: fbase + f * ndf + 2 * i,
                                        canon_dofs: [
                                            first_dof + 2 * p as u32,
                                            first_dof + 2 * p as u32 + 1,
                                        ],
                                        s,
                                    });
                                    dofs_flat.push(first_dof + 2 * p as u32);
                                    signs_flat.push(1.0);
                                    dofs_flat.push(first_dof + 2 * p as u32 + 1);
                                    signs_flat.push(1.0);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }

            // Interior DOFs (NDk, k>=3 for Tet, k>=2 for others).
            let interior_count: u32 = match (dim, cell_type) {
                (2, ElementType::Tri3 | ElementType::Tri6) if k >= 2 => (k * (k - 1)) as u32,
                (2, ElementType::Quad4 | ElementType::Quad8) if k >= 2 => (2 * k * (k - 1)) as u32,
                (3, ElementType::Tet4 | ElementType::Tet10) if k >= 3 => {
                    (k * (k - 1) * (k - 2) / 2) as u32
                }
                (3, ElementType::Hex8 | ElementType::Hex20) if k >= 2 => {
                    (3 * k * (k - 1) * (k - 1)) as u32
                }
                (3, ElementType::Prism6) if k >= 2 => {
                    ((k * (k - 1) * (k - 1)) + (k * (k - 1) * (k - 2)) / 2) as u32
                }
                (3, ElementType::Pyramid5) if k >= 2 => (3 * k * (k - 1) * (k - 1)) as u32,
                _ => 0,
            };
            for _ in 0..interior_count {
                dofs_flat.push(interior_cursor);
                interior_cursor += 1;
                signs_flat.push(1.0);
            }

            elem_offsets.push(dofs_flat.len());
            elem_face_blocks.push(elem_blocks);
        }

        debug_assert_eq!(interior_cursor, n_dofs_total);
        HCurlSpace {
            mesh,
            order,
            n_dofs: n_dofs_total as usize,
            dofs_flat,
            signs_flat,
            elem_offsets,
            edge_to_dof,
            face_to_dof,
            face_anchor,
            elem_face_blocks,
            quad_face_to_dof,
            quad_face_anchor,
            dim,
            cell_type: first_cell_type,
            quad_igll,
        }
    }

    /// Return the polynomial order of this space.
    pub fn order(&self) -> u8 { self.order }

    /// Orientation signs (±1.0) for the DOFs on element `elem`.
    ///
    /// `signs[i]` multiplies basis function `i` on this element so that the
    /// tangential trace is consistent with the global edge orientation.
    pub fn element_signs(&self, elem: u32) -> &[f64] {
        let start = self.elem_offsets[elem as usize];
        let end = self.elem_offsets[elem as usize + 1];
        &self.signs_flat[start..end]
    }

    /// Look up the global DOF index for a given edge (by canonical key).
    pub fn edge_dof(&self, edge: EdgeKey) -> Option<DofId> {
        self.edge_to_dof.get(&edge).copied()
    }

    /// Look up all global DOFs associated with a given edge.
    pub fn edge_dofs(&self, edge: EdgeKey) -> Option<Vec<DofId>> {
        self.edge_to_dof.get(&edge).map(|&first| {
            (0..self.order as DofId).map(|m| first + m).collect()
        })
    }

    /// Physical coordinates of every global DOF (MFEM `GetDofCoords` analog).
    ///
    /// For ND1 the DOF sits at the edge midpoint; for NDk (k≥2) at the
    /// Gauss-Legendre points along the canonical (min→max) edge direction —
    /// the point-value DOF locations (MFEM `FE::Nodes` semantics).
    ///
    /// Triangular (tet) and quadrilateral (hex) **face-interior** DOFs are the
    /// `k(k−1)` / 2·`k(k−1)` point-value sites of the face's canonical DOF
    /// list, so they take the face-creating element's physical anchor points
    /// (`face_anchor` / `quad_face_anchor`).  D505: the hex quad-face branch
    /// was missing entirely, leaving all `2k(k−1)` DOFs of every hex face at
    /// `[0,0,0]`; D525 extended the same anchor semantics to the prism /
    /// pyramid triangular faces (`tri_face_nodes`, from the MFEM
    /// `ND_WedgeElement` / `ND_FuentesPyramidElement` layouts) and their quad
    /// faces (`quad_face_nodes`).
    ///
    /// D525 (c): **element-interior** DOFs take the owning element's physical
    /// points, obtained by pushing the interior slots of the MFEM nodal
    /// layout through the element map (trilinear hex, affine tet, trilinear
    /// prism, collapsed pyramid; 2-D tri/quad likewise).
    pub fn dof_coords(&self) -> Vec<[f64; 3]> {
        let mut out = vec![[0.0f64; 3]; self.n_dofs()];
        let dim = self.mesh.dim() as usize;
        let nd = self.order as usize;
        let (gl_pts, _) = gauss_legendre_01(nd.max(1));
        for (&EdgeKey(a, b), &first) in &self.edge_to_dof {
            let pa = self.mesh.node_coords(a);
            let pb = self.mesh.node_coords(b);
            for m in 0..nd {
                let t = if nd == 1 { 0.5 } else { gl_pts[m] };
                let d = (first + m as u32) as usize;
                for c in 0..3 {
                    let xa = if c < dim { pa[c] } else { 0.0 };
                    let xb = if c < dim { pb[c] } else { 0.0 };
                    out[d][c] = (1.0 - t) * xa + t * xb;
                }
            }
        }
        // Triangular face DOFs (k ≥ 2, 3-D) sit at the canonical face slot
        // points of the face-creating element: tets through the tangent-pair
        // anchor (`face_anchor`), prisms/pyramids likewise (D525 layouts with
        // the D546 tangents).  The branch must cover every registered tri
        // face — a miss is a build bug, not a placeholder situation (D525).
        for (&key, &first) in &self.face_to_dof {
            if self.order < 2 {
                break;
            }
            if let Some(anchor) = self.face_anchor.get(&key) {
                for p in 0..anchor.n_points() {
                    for j in 0..2 {
                        let d = (first + 2 * p as u32 + j as u32) as usize;
                        out[d] = anchor.point(p);
                    }
                }
            } else {
                panic!("HCurlSpace::dof_coords: triangular face {key:?} has no canonical anchor");
            }
        }
        // Quadrilateral face DOFs (k ≥ 2, 3-D): hexes (D505), prisms and
        // pyramids (D525/D546) through the per-slot point lists of the
        // face-creating element (`quad_face_anchor`).
        for (&key, &first) in &self.quad_face_to_dof {
            if let Some(anchor) = self.quad_face_anchor.get(&key) {
                for (m, p) in anchor.nodes.iter().enumerate() {
                    let d = (first + m as u32) as usize;
                    out[d] = *p;
                }
            } else {
                panic!("HCurlSpace::dof_coords: quad face {key:?} has no canonical anchor");
            }
        }
        // Element-interior DOFs (D525c): element-owned, so their points come
        // from the owning element's map applied to the interior slots of the
        // MFEM nodal layout (the tail of each layout table).
        if self.order >= 2 {
            let mut hex_coords: Option<Vec<Vec<f64>>> = None;
            let mut tet_coords: Option<Vec<Vec<f64>>> = None;
            let mut tri_coords: Option<Vec<Vec<f64>>> = None;
            let mut quad_coords: Option<Vec<Vec<f64>>> = None;
            let mut prism_layout: Option<Vec<[f64; 3]>> = None;
            let mut pyra_layout: Option<Vec<[f64; 3]>> = None;
            for e in 0..self.mesh.n_elements() as u32 {
                let cell_type = self.mesh.element_type(e);
                let dofs = self.element_dofs(e);
                match (dim, cell_type) {
                    (3, ElementType::Hex8 | ElementType::Hex20) => {
                        let coords = hex_coords.get_or_insert_with(|| HexNDk::new(nd).dof_coords());
                        let verts8 = hex8_verts(&self.mesh, e);
                        let ndf_quad = 2 * nd * (nd - 1);
                        let off = 12 * nd + 6 * ndf_quad;
                        for (n, xi) in coords.iter().skip(off).enumerate() {
                            let x = hex_trilinear_map(&verts8, xi).0;
                            out[dofs[dofs.len() - coords.len() + off + n] as usize] = x;
                        }
                    }
                    (3, ElementType::Tet4 | ElementType::Tet10) if nd >= 3 => {
                        let n_interior = nd * (nd - 1) * (nd - 2) / 2;
                        let coords =
                            tet_coords.get_or_insert_with(|| TetNDk::new(nd).dof_coords());
                        let off = coords.len() - n_interior;
                        let nodes = self.mesh.element_nodes(e);
                        let p0 = self.mesh.node_coords(nodes[0]);
                        for (n, xi) in coords.iter().skip(off).enumerate() {
                            let mut x = [p0[0], p0[1], p0[2]];
                            for (c, lv) in [1usize, 2, 3].iter().enumerate() {
                                let p = self.mesh.node_coords(nodes[*lv]);
                                for d in 0..3 {
                                    x[d] += (p[d] - p0[d]) * xi[c];
                                }
                            }
                            out[dofs[dofs.len() - n_interior + n] as usize] = x;
                        }
                    }
                    (3, ElementType::Prism6) => {
                        let n_interior =
                            (nd * (nd - 1) * (nd - 1)) + (nd * (nd - 1) * (nd - 2)) / 2;
                        let layout = prism_layout
                            .get_or_insert_with(|| PrismNDk::new(nd).mfem_layout_points());
                        let verts6 = prism6_verts(&self.mesh, e);
                        for (n, p) in prism_interior_nodes(&verts6, nd, layout)
                            .iter()
                            .enumerate()
                        {
                            out[dofs[dofs.len() - n_interior + n] as usize] = *p;
                        }
                    }
                    (3, ElementType::Pyramid5) => {
                        let layout = pyra_layout
                            .get_or_insert_with(|| PyraNDk::new(nd).mfem_layout_points());
                        let verts5 = pyramid5_verts(&self.mesh, e);
                        for (n, p) in pyramid_interior_nodes(&verts5, nd, layout)
                            .iter()
                            .enumerate()
                        {
                            out[dofs[dofs.len() - 3 * nd * (nd - 1) * (nd - 1) + n] as usize] = *p;
                        }
                    }
                    (2, ElementType::Tri3 | ElementType::Tri6) => {
                        let n_interior = nd * (nd - 1);
                        let coords =
                            tri_coords.get_or_insert_with(|| TriNDk::new(nd).dof_coords());
                        let off = coords.len() - n_interior;
                        let nodes = self.mesh.element_nodes(e);
                        let x0 = self.mesh.node_coords(nodes[0]);
                        let x1 = self.mesh.node_coords(nodes[1]);
                        let x2 = self.mesh.node_coords(nodes[2]);
                        for (n, xi) in coords.iter().skip(off).enumerate() {
                            out[dofs[dofs.len() - n_interior + n] as usize] = [
                                x0[0] + (x1[0] - x0[0]) * xi[0] + (x2[0] - x0[0]) * xi[1],
                                x0[1] + (x1[1] - x0[1]) * xi[0] + (x2[1] - x0[1]) * xi[1],
                                0.0,
                            ];
                        }
                    }
                    (2, ElementType::Quad4 | ElementType::Quad8) => {
                        let n_interior = 2 * nd * (nd - 1);
                        let coords =
                            quad_coords.get_or_insert_with(|| QuadND::new(nd).dof_coords());
                        let off = coords.len() - n_interior;
                        let nodes = self.mesh.element_nodes(e);
                        let c: Vec<[f64; 2]> = (0..4)
                            .map(|i| {
                                let p = self.mesh.node_coords(nodes[i]);
                                [p[0], p[1]]
                            })
                            .collect();
                        for (n, xi) in coords.iter().skip(off).enumerate() {
                            let w = [
                                (1.0 - xi[0]) * (1.0 - xi[1]),
                                xi[0] * (1.0 - xi[1]),
                                xi[0] * xi[1],
                                (1.0 - xi[0]) * xi[1],
                            ];
                            out[dofs[dofs.len() - n_interior + n] as usize] = [
                                w[0] * c[0][0]
                                    + w[1] * c[1][0]
                                    + w[2] * c[2][0]
                                    + w[3] * c[3][0],
                                w[0] * c[0][1]
                                    + w[1] * c[1][1]
                                    + w[2] * c[2][1]
                                    + w[3] * c[3][1],
                                0.0,
                            ];
                        }
                    }
                    _ => {}
                }
            }
        }
        out
    }

    /// Number of unique edges in the mesh (== `n_dofs` for ND1).
    pub fn n_edges(&self) -> usize {
        self.edge_to_dof.len()
    }

    /// Number of unique faces in 3D ND2 mode.
    pub fn n_faces(&self) -> usize {
        self.face_to_dof.len()
    }

    /// Number of unique quad faces for hex NDk.
    pub fn n_quad_faces(&self) -> usize {
        self.quad_face_to_dof.len()
    }

    /// Look up the first global DOF for a triangular face (Tet NDk, k≥2).
    /// Returns `None` for ND1 or if the face is not found.
    pub fn face_dof(&self, face: FaceKey) -> Option<DofId> {
        self.face_to_dof.get(&face).copied()
    }

    /// Canonical (shared) DOF functional anchor of a tet face (NDk, k≥2).
    ///
    /// The face's global face DOFs are the point-value functionals
    /// `σ_{2p+c}(Φ) = Φ(x_p)·t[p][c]` of the face-creating element; consumers
    /// (`discrete_op`) must use the same anchor when building dof rows for the
    /// shared face slots, and [`Self::element_face_blocks`] to relate another
    /// element's local face DOFs to it.
    pub fn face_anchor(&self, face: FaceKey) -> Option<&TetFaceAnchor> {
        self.face_anchor.get(&face)
    }

    /// The element's face-DOF block transforms into the canonical
    /// (face-creating element) basis (D37; D546 adds the prism / pyramid
    /// tri faces).  Empty when the space has no shared tri-face DOF pairs
    /// (2-D spaces, hex/prism/pyramid quad faces use signed permutations, and
    /// k = 1).
    ///
    /// For element matrices assembled in the element's own (signed) local
    /// DOFs, the canonical representation is `A ← Tᵀ·A·T` and `b ← Tᵀ·b`, with
    /// `T` the block-diagonal map built from these blocks; the primal dof
    /// vector satisfies `u_local = T·u_canon`.
    ///
    /// D55 note (hex NDk stays block-free *by MFEM design*, not by omission):
    /// `ND_FECollection::DofTransformationForGeometry` returns NULL for
    /// tensor-product geometries, so `ND_HexahedronElement` never receives a
    /// `DofTransformation`, and `ND_DofTransformation::TransformPrimal`
    /// applies the 2×2 `T(ori)` matrices to **triangular** faces only.  MFEM
    /// relates hex quad-face DOFs across elements through the signed
    /// permutation `ND_FECollection::QuadDofOrd[ori]` baked into the dof
    /// table — the same functional-level signed permutation this space
    /// computes in `match_face_dof`.  Adding 2×2 blocks for hex faces would
    /// therefore *diverge* from MFEM; the beam-hex `-o 2` discrepancy that
    /// motivated D55 was traced to the curl-curl quadrature order (see
    /// `examples/mfem_ex3_maxwell_cavity.rs` `assemble_mat_mfem_rule`) and is
    /// covered by `crates/assembly/tests/d55_hex_nd2_system_regression.rs`.
    pub fn element_face_blocks(&self, e: u32) -> &[FaceDofBlock] {
        &self.elem_face_blocks[e as usize]
    }

    /// D807-2: the element's per-face-point-pair 2×2 transforms, keyed by the
    /// element-local slot of the pair's first DOF — the accessor the parallel
    /// DOF partition consumes (see [`FacePairTransform`]).
    ///
    /// One entry per face point pair of every shared triangular face of the
    /// element (tet / prism / pyramid `NDk`, k ≥ 2), in element slot order;
    /// empty for every element whose shared-face relation is a signed
    /// permutation — 2-D spaces, hex quad faces, prism/pyramid quad faces and
    /// k = 1 — where [`Self::element_signs`] is the exact description (which is
    /// why the pair channel stays empty, and bit-identical, on those).
    ///
    /// The pair's two DOF ids are `Self::element_dofs(e)[slot]` and `[slot + 1]`
    /// (equal to the block's `canon_dofs`), so this accessor needs no `DofId`
    /// plumbing: it pairs each transform with the element slot that has to be
    /// mixed.
    pub fn element_face_pair_transforms(&self, e: u32) -> Vec<FacePairTransform> {
        self.elem_face_blocks[e as usize]
            .iter()
            .map(|b| FacePairTransform {
                slot: b.slot as u32,
                s: b.s,
                s_inv: Self::inv2(b.s),
            })
            .collect()
    }

    // ─── D813-1: facets whose canonical anchor element is not in the local mesh ──

    /// The facet-DOF relation of the local element `e` against a **published
    /// anchor element** (D813-1) — the channel that lets the parallel ghost
    /// layer drop the facet's canonical (minimum-global-element-id) holder.
    ///
    /// The parallel DOF partition's global face basis is that anchor element's
    /// own basis (D412 / D122-3): its face DOFs are keyed by `(facet, pos)`,
    /// `pos` being the anchor's slot index, and D807-2's pair channel reads the
    /// anchor's 2×2 blocks.  Until D813-1 the partition therefore had to carry
    /// the anchor element *itself*.  This accessor rebuilds the anchor's facet
    /// frame from the anchor's `(ElementType, global vertex list)` — published
    /// by the mesh extraction
    /// (`fem_parallel::EntityOwnership::facet_anchor_element`).
    ///
    /// Arguments:
    /// * `e` — the local element carrying the facet;
    /// * `facet_verts` — the facet's global vertex ids (a set; order is
    ///   irrelevant);
    /// * `anchor_et`, `anchor_verts` — the anchor element's type and its global
    ///   vertex list **in the element's own slot order**;
    /// * `global_node` — local mesh node id → global id (locates the facet in
    ///   `e`'s own block);
    /// * `anchor_coords` — global vertex id → physical coordinates.  Only the
    ///   facet's own vertices are queried: the anchor's off-facet slots are
    ///   filled with [`OFF_FACET_SENTINEL`] and the returned match must be a
    ///   bijection onto the anchor's block, so a family whose frame is *not*
    ///   face-local is refused instead of producing a wrong global basis.
    ///
    /// Returns one [`FacetSlotToAnchor`] per facet DOF of `e` (in element slot
    /// order), or `None` when the channel cannot express the relation: 2-D or
    /// `k < 2`, an unregistered element family, a facet whose shape disagrees
    /// between the two elements, or a frame that does not match `e`'s slot
    /// geometry.  The caller must then keep requiring a local anchor — loudly.
    pub fn facet_slots_against_published_anchor(
        &self,
        e: u32,
        facet_verts: &[u32],
        anchor_et: ElementType,
        anchor_verts: &[u32],
        global_node: &dyn Fn(u32) -> u32,
        anchor_coords: &dyn Fn(u32) -> Option<[f64; 3]>,
    ) -> Option<Vec<FacetSlotToAnchor>> {
        let k = self.order as usize;
        if self.dim != 3 || k < 2 || facet_verts.is_empty() {
            return None;
        }
        if facet_verts.len() != 3 && facet_verts.len() != 4 {
            return None;
        }
        if anchor_verts.len() < anchor_et.nodes_per_element() {
            return None;
        }
        let et_e = self.mesh.element_type(e);
        let blocks_e = facet_block_table(et_e, k)?;
        let blocks_a = facet_block_table(anchor_et, k)?;
        let nodes_e = self.mesh.element_nodes(e);
        let gi_e: Vec<u32> = nodes_e.iter().map(|&n| global_node(n)).collect();
        let ib_e = find_facet_block(&blocks_e, &gi_e, facet_verts)?;
        let ib_a = find_facet_block(&blocks_a, anchor_verts, facet_verts)?;
        if blocks_e[ib_e].tri != blocks_a[ib_a].tri {
            return None;
        }
        // The local space's face tables are keyed by the *local* sub-mesh's
        // node ids (`build`'s Pass 2), while `facet_verts` is a global-id set —
        // so the creator lookup needs the facet's ids as this element sees
        // them, i.e. its own block's slots mapped through `nodes_e`.
        let fv_local: Vec<u32> = blocks_e[ib_e].slots.iter().map(|&s| nodes_e[s]).collect();
        // The local element is local: every one of its vertices has real
        // coordinates.
        let verts_e: Vec<[f64; 3]> = nodes_e
            .iter()
            .map(|&n| {
                let c = self.mesh.node_coords(n);
                std::array::from_fn(|d| c.get(d).copied().unwrap_or(0.0))
            })
            .collect();
        let geom_e = facet_block_geometry(et_e, &verts_e, k, &blocks_e[ib_e])?;
        // The anchor's frame: the facet's vertices only.
        let verts_a: Vec<[f64; 3]> = anchor_verts
            .iter()
            .map(|&g| {
                if facet_verts.contains(&g) {
                    anchor_coords(g).unwrap_or(OFF_FACET_SENTINEL)
                } else {
                    OFF_FACET_SENTINEL
                }
            })
            .collect();
        let geom_a = facet_block_geometry(anchor_et, &verts_a, k, &blocks_a[ib_a])?;

        // The local space's own canonical (face-creating element's) frame — the
        // basis both sides are expressed against.
        match (geom_e, geom_a) {
            (
                FacetGeom::Quad { nodes: xe, tangents: te },
                FacetGeom::Quad { nodes: xa, tangents: ta },
            ) => {
                let creator = self
                    .quad_face_anchor
                    .get(&QuadFaceKey::new(
                        fv_local[0], fv_local[1], fv_local[2], fv_local[3],
                    ))?;
                let mut seen = vec![false; xa.len()];
                let mut out = Vec::with_capacity(xe.len());
                for n in 0..xe.len() {
                    let (m, _s) = match match_face_dof_soft(&xa, &ta, xe[n], te[n]) {
                        Some(v) => v,
                        None => {
                            eprintln!(
                                "[d813dbg] quad match miss n={n} x_e={:?} t_e={:?}\n  x_a={:?}\n  t_a={:?}",
                                xe[n], te[n], xa, ta
                            );
                            return None;
                        }
                    };
                    // A frame that is not this facet's is a refusal, not a
                    // silently wrong basis (the match is a permutation).
                    if seen[m] {
                        return None;
                    }
                    seen[m] = true;
                    // The anchor element's own `element_signs` at `m`: the
                    // anchor's tangent against the local canonical list — the
                    // same rule `build`'s Pass 3 records for the anchor.
                    let (_q, anchor_sign) =
                        match_face_dof(&creator.nodes, &creator.tangents, xa[m], ta[m]);
                    out.push(FacetSlotToAnchor {
                        slot: blocks_e[ib_e].dof_offset + n,
                        anchor_slot: m,
                        anchor_sign,
                        anchor_pair: None,
                    });
                }
                Some(out)
            }
            (FacetGeom::Tri { pts: xe, tans: te }, FacetGeom::Tri { pts: xa, tans: ta }) => {
                let creator = self
                    .face_anchor
                    .get(&FaceKey::new(fv_local[0], fv_local[1], fv_local[2]))?;
                // `xa` holds one entry per face *point*; a point carries a DOF
                // pair, so the slot-space is twice as wide.
                let mut seen = vec![false; 2 * xa.len()];
                let mut out = Vec::with_capacity(2 * xe.len());
                // Tri-face geometry is **per point** (each entry carries the
                // point's tangent pair), while the element's DOF block is laid
                // out as consecutive pairs per point (`build`'s Pass 3) — so
                // point `i` owns the element slots `2i` and `2i + 1`.
                for i in 0..xe.len() {
                    let (p, d) = nearest_face_point(&xa, xe[i]);
                    if d > face_point_tol(xe[i]) {
                        return None;
                    }
                    // The local element's own tangents are ground truth: both
                    // must lie in the anchor frame's tangent plane, otherwise
                    // the frame did not come from this facet (a family whose
                    // frame is not face-local).  This is the guard the point
                    // match alone cannot give, because the 2×2 pair below has
                    // no matching test of its own.
                    let nrm = cross3(ta[p][0], ta[p][1]);
                    let nn = norm3(nrm);
                    if !(nn > 0.0) {
                        return None;
                    }
                    for c in 0..2 {
                        let e = te[i][c];
                        if dot3(e, nrm).abs() > 1e-9 * norm3(e) * nn {
                            return None;
                        }
                    }
                    let q = match_face_point(creator, xa[p]);
                    let s = face_pair_change_of_basis(
                        &ta[p],
                        &creator.tangents(q),
                        "D813-1 anchor facet frame",
                    );
                    let pair = FacePairTransform {
                        slot: (blocks_e[ib_e].dof_offset + 2 * i) as u32,
                        s,
                        s_inv: Self::inv2(s),
                    };
                    for c in 0..2 {
                        let anchor_slot = 2 * p + c;
                        if seen[anchor_slot] {
                            return None;
                        }
                        seen[anchor_slot] = true;
                        out.push(FacetSlotToAnchor {
                            slot: blocks_e[ib_e].dof_offset + 2 * i + c,
                            anchor_slot,
                            anchor_sign: 1.0,
                            anchor_pair: Some(pair),
                        });
                    }
                }
                Some(out)
            }
            _ => None,
        }
    }

    /// Inverse of a 2×2 matrix (D602 helper).
    fn inv2(m: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        assert!(det.abs() > 1e-300, "HCurlSpace: singular 2×2 face map");
        [
            [m[1][1] / det, -m[0][1] / det],
            [-m[1][0] / det, m[0][0] / det],
        ]
    }

    /// Product of two 2×2 matrices (D602 helper).
    fn mul2(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
        [
            [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
            [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
        ]
    }

    /// D602: cross-element frame view of one shared triangular face's DOF
    /// pairs — `R = S_last · S_first⁻¹`, the change of basis between the two
    /// adjacent elements' local face-pair frames, where "first" is the
    /// face-creating (first-encounter) element and "last" the other one.
    ///
    /// Concretely, with the canonical (global) pair values `u_canon` and the
    /// element-local values `u_loc` of the *second* element,
    /// `u_loc(last) = R · u_canon`.  In MFEM terms `R` is
    /// `T(fo)⁻¹ = TInv(Fo(Elem2))` of `ND_DofTransformation`
    /// (`fem/doftrans.hpp`), because the face-creating element is MFEM's
    /// `FaceInfo::Elem1No`, whose orientation is pinned to zero
    /// (`Mesh::AddTriangleFaceElement`: `Elem1Inf = 64*lf`, "orientation 0")
    /// — the same element this space anchors the canonical functionals to.
    ///
    /// **This is not a `.gf` file-storage map.**  The vector MFEM's
    /// `GridFunction::ProjectCoefficient` stores is, per face pair,
    /// `T(Fo(ElemE))·raw(E)` for the last writer `E`, and the writer
    /// independence identity `T(Fo(E))·S(E) = I` (both writers, verified
    /// 38/38 exactly on the D559 6-tet probe) means the stored values equal
    /// `u_canon` bit for bit — fem-rs's canonical `.gf` output is already
    /// MFEM's `.gf` format, no transformation applied or required (D602
    /// evidence: `tmp/d602/`).  `R` is the *element-frame* relation behind
    /// that identity, exposed for diagnostics and for pinning fem-rs's S
    /// blocks against MFEM's T table in tests.
    ///
    /// Returns `None` when `face` is not a registered shared tri face of
    /// exactly two elements (boundary faces, 2-D spaces, k = 1), or when the
    /// space carries no 2×2 face blocks for it.  For k ≥ 3 every DOF pair of
    /// a straight face shares the same map; the first pair's map is
    /// returned.
    pub fn face_pair_storage_map(&self, face: FaceKey) -> Option<[[f64; 2]; 2]> {
        if self.order < 2 || self.dim != 3 {
            return None;
        }
        let first_dof = self.face_to_dof.get(&face).copied()?;
        let nfd = (self.order as usize * (self.order as usize - 1)) as DofId;
        // Locate the face's adjacent elements by scanning every element's
        // triangular faces (the only faces carrying 2×2 blocks).  Interior
        // faces are found twice, boundary faces once (→ `None`).
        let mut e1 = None;
        let mut e2 = None;
        for e in self.mesh.elem_iter() {
            let verts = self.mesh.element_nodes(e);
            let tri_faces: &[(usize, usize, usize)] = match self.mesh.element_type(e) {
                ElementType::Tet4 | ElementType::Tet10 => &TET_FACES,
                ElementType::Prism6 => &PRISM_TRI_FACES,
                ElementType::Pyramid5 => &PYRAMID_TRI_FACES,
                _ => continue,
            };
            for &(la, lb, lc) in tri_faces {
                if FaceKey::new(verts[la], verts[lb], verts[lc]) == face {
                    if e1.is_none() {
                        e1 = Some(e);
                    } else {
                        e2 = Some(e);
                        break;
                    }
                }
            }
        }
        let (e1, e2) = match (e1, e2) {
            (Some(a), Some(b)) => (a, b),
            _ => return None,
        };
        if e1 == e2 {
            return None;
        }
        let (first, last) = if e1 < e2 { (e1, e2) } else { (e2, e1) };
        // Per element: the 2×2 blocks of *this* face, keyed by the pair's
        // first canonical DOF (both elements map onto the same global pair).
        let blocks = |e: u32| -> std::collections::HashMap<DofId, [[f64; 2]; 2]> {
            self.element_face_blocks(e)
                .iter()
                .filter(|b| b.canon_dofs[0] >= first_dof && b.canon_dofs[0] < first_dof + nfd)
                .map(|b| (b.canon_dofs[0], b.s))
                .collect()
        };
        let last_blocks = blocks(last);
        let mut map = None;
        for b in self.element_face_blocks(first) {
            if b.canon_dofs[0] < first_dof || b.canon_dofs[0] >= first_dof + nfd {
                continue;
            }
            let s_last = last_blocks.get(&b.canon_dofs[0])?;
            let r = Self::mul2(*s_last, Self::inv2(b.s));
            if map.is_none() {
                map = Some(r);
            }
        }
        map
    }

    /// Look up all global DOFs associated with a quad face (hex NDk, k≥2).
    pub fn quad_face_dofs(&self, key: QuadFaceKey) -> Option<Vec<DofId>> {
        if self.order < 2 { return None; }
        let ndf = 2 * self.order as DofId * (self.order as DofId - 1);
        self.quad_face_to_dof.get(&key).map(|&first| {
            (0..ndf).map(|m| first + m).collect()
        })
    }

    /// Vector-valued interpolation via the Nédélec DOF functional.
    ///
    /// ## ND1 (order 1)
    /// Midpoint-evaluated tangential moment per edge.
    ///
    /// Total number of global DOFs.
    pub fn n_dofs(&self) -> usize { self.n_dofs }

    /// Global DOF indices for element `elem`.
    pub fn element_dofs(&self, elem: u32) -> &[DofId] {
        let s = self.elem_offsets[elem as usize];
        &self.dofs_flat[s..self.elem_offsets[elem as usize + 1]]
    }

    /// Reference to the underlying mesh.
    pub fn mesh_topology(&self) -> &dyn MeshTopology { &self.mesh }

    /// ## NDk (k >= 2)
    ///
    /// Every DOF is the MFEM point-value functional `σ(Φ) = Φ(x_i)·t̂_i`: edge
    /// DOFs at the Gauss-Legendre points of the canonical (min→max) edge
    /// direction; tri interior / quad interior / tet face / tet interior DOFs
    /// at MFEM's `FE::Nodes` points with the `dof2tk` tangent pairs pushed
    /// through the element map (`J·t̂`) — quad interiors through the bilinear
    /// map of the paired `fem_element::nedelec::QuadND` (D765).  Tet face DOFs
    /// use the shared face's canonical (face-creating element) functional list,
    /// so the returned values are the canonical (global) dof values — see
    /// [`Self::element_face_blocks`] for the per-element 2×2 change of basis
    /// (D37).
    pub fn interpolate_vector(&self, f: &dyn Fn(&[f64]) -> Vec<f64>) -> Vector<f64> {
        let mut result = Vector::zeros(self.n_dofs);
        let k = self.order as usize;

        // (GaussLobatto, IntegratedGLL) quad variant: MFEM
        // `ND_QuadrilateralElement::Project` dispatches to `ProjectIntegrated`
        // for the integrated type (`fe_nd.hpp:68`), so every DOF is the
        // sub-cell line integral `∫ f·(Jᵀt) ds` — not a point value.
        if self.quad_igll && self.cell_type == ElementType::Quad4 {
            let el = fem_element::nedelec::QuadND::new_integrated_gll(k);
            let functionals = el.integrated_functionals();
            let r = result.as_slice_mut();
            for e in 0..self.mesh.n_elements() as u32 {
                let verts = self.mesh.element_nodes(e);
                let c: Vec<[f64; 2]> = (0..4)
                    .map(|i| {
                        let p = self.mesh.node_coords(verts[i]);
                        [p[0], p[1]]
                    })
                    .collect();
                let dofs = self.element_dofs(e);
                let signs = self.element_signs(e);
                for (i, fi) in functionals.iter().enumerate() {
                    let mut val = 0.0_f64;
                    for &([xi, eta], w) in &fi.samples {
                        // Bilinear quad map and its Jacobian.
                        let px = (1.0 - xi) * (1.0 - eta) * c[0][0]
                            + xi * (1.0 - eta) * c[1][0]
                            + xi * eta * c[2][0]
                            + (1.0 - xi) * eta * c[3][0];
                        let py = (1.0 - xi) * (1.0 - eta) * c[0][1]
                            + xi * (1.0 - eta) * c[1][1]
                            + xi * eta * c[2][1]
                            + (1.0 - xi) * eta * c[3][1];
                        let dx_dxi = [
                            (1.0 - eta) * (c[1][0] - c[0][0]) + eta * (c[2][0] - c[3][0]),
                            (1.0 - eta) * (c[1][1] - c[0][1]) + eta * (c[2][1] - c[3][1]),
                        ];
                        let dx_deta = [
                            (1.0 - xi) * (c[3][0] - c[0][0]) + xi * (c[2][0] - c[1][0]),
                            (1.0 - xi) * (c[3][1] - c[0][1]) + xi * (c[2][1] - c[1][1]),
                        ];
                        // MFEM: `tk^T J vk` = `f(x) · (t0 ∂x/∂ξ + t1 ∂x/∂η)`.
                        let j_t = [
                            fi.t[0] * dx_dxi[0] + fi.t[1] * dx_deta[0],
                            fi.t[0] * dx_dxi[1] + fi.t[1] * dx_deta[1],
                        ];
                        let fv = f(&[px, py]);
                        val += w * (fv[0] * j_t[0] + fv[1] * j_t[1]);
                    }
                    r[dofs[i] as usize] = signs[i] * val;
                }
            }
            return result;
        }

        if k == 1 {
            // ND1: midpoint rule per edge.
            for (&EdgeKey(a, b), &dof) in &self.edge_to_dof {
                let pa = self.mesh.node_coords(a);
                let pb = self.mesh.node_coords(b);
                let mid: Vec<f64> = (0..self.dim).map(|d| 0.5 * (pa[d] + pb[d])).collect();
                let tangent: Vec<f64> = (0..self.dim).map(|d| pb[d] - pa[d]).collect();
                let fval = f(&mid);
                let dot: f64 = fval.iter().zip(&tangent).map(|(fi, ti)| fi * ti).sum();
                result.as_slice_mut()[dof as usize] = dot;
            }
            return result;
        }

        // NDk (k >= 2): point-value edge DOFs (MFEM `Project_ND` semantics):
        //   dof_j = Φ(y_j)·τ,  y_j = P_min + t_j·(P_max − P_min),  τ = P_max − P_min
        // with t_j the k-point Gauss-Legendre nodes (MFEM `OpenPoints(k−1)`),
        // i.e. exactly the canonical global DOF functionals.  Because the
        // interpolation uses the canonical functional directly, the values are
        // independent of any element's local orientation.
        let (gl_pts, _gl_wts) = gauss_legendre_01(k);

        let npts = gl_pts.len();
        let n_elem = self.mesh.n_elements();
        {
            let r = result.as_slice_mut();
            for (&EdgeKey(a, b), &first_dof) in &self.edge_to_dof {
                let pa = self.mesh.node_coords(a);
                let pb = self.mesh.node_coords(b);
                let dim = self.dim;
                for m in 0..npts {
                    let t = gl_pts[m];
                    let mut pt = [0.0_f64; 3];
                    let mut tau = [0.0_f64; 3];
                    for d in 0..dim {
                        pt[d] = pa[d] + t * (pb[d] - pa[d]);
                        tau[d] = pb[d] - pa[d];
                    }
                    let fval = f(&pt[..dim]);
                    let dot: f64 = fval.iter().zip(tau[..dim].iter()).map(|(fi, ti)| fi * ti).sum();
                    r[first_dof as usize + m] = dot;
                }
            }
        }

        // Step 2 — face / interior DOFs.
        if self.dim == 2 && k == 2 {
            // ── ND2 interior DOFs: point values (MFEM Nodes semantics) ──
            if matches!(self.cell_type, ElementType::Tri3 | ElementType::Tri6) {
                // TriND2 interior slots (last two): point values at the
                // reference (1/3,1/3) with tangents (1,0) and (0,1):
                // σ = (J t̂)·Φ_phys with J(1,0) = x1−x0, J(0,1) = x2−x0.
                for e in 0..n_elem as u32 {
                    let dofs = self.element_dofs(e);
                    let bub0 = dofs[dofs.len() - 2] as usize;
                    let bub1 = dofs[dofs.len() - 1] as usize;
                    let nodes = self.mesh.element_nodes(e);
                    let x0 = self.mesh.node_coords(nodes[0]);
                    let x1 = self.mesh.node_coords(nodes[1]);
                    let x2 = self.mesh.node_coords(nodes[2]);
                    let pt = [
                        (x0[0] + x1[0] + x2[0]) / 3.0,
                        (x0[1] + x1[1] + x2[1]) / 3.0,
                    ];
                    let fv = f(&pt);
                    let r = result.as_slice_mut();
                    r[bub0] = (x1[0] - x0[0]) * fv[0] + (x1[1] - x0[1]) * fv[1];
                    r[bub1] = (x2[0] - x0[0]) * fv[0] + (x2[1] - x0[1]) * fv[1];
                }
            } else if matches!(self.cell_type, ElementType::Quad4 | ElementType::Quad8) {
                // QuadND2 interior slots (8..12): x-comp point values at the
                // reference (t_m, 1/2) with tangent J(1,0); y-comp at (1/2, t_m)
                // with tangent J(0,1) — bilinear map through the 4 corners.
                for e in 0..n_elem as u32 {
                    let dofs = self.element_dofs(e);
                    let nodes = self.mesh.element_nodes(e);
                    let x0 = self.mesh.node_coords(nodes[0]);
                    let x1 = self.mesh.node_coords(nodes[1]);
                    let x2 = self.mesh.node_coords(nodes[2]);
                    let x3 = self.mesh.node_coords(nodes[3]);
                    let r = result.as_slice_mut();
                    for (m, &t) in gl_pts.iter().enumerate() {
                        // x-component at (t, 1/2)
                        let (xi, eta) = (t, 0.5);
                        let w = [
                            (1.0 - xi) * (1.0 - eta),
                            xi * (1.0 - eta),
                            xi * eta,
                            (1.0 - xi) * eta,
                        ];
                        let pt: Vec<f64> = (0..2)
                            .map(|d| w[0] * x0[d] + w[1] * x1[d] + w[2] * x2[d] + w[3] * x3[d])
                            .collect();
                        let fv = f(&pt);
                        // J(ξ,η)·(1,0) = ∂x/∂ξ = (1−η)(x1−x0) + η(x2−x3)
                        let jt: Vec<f64> = (0..2)
                            .map(|d| {
                                (1.0 - eta) * (x1[d] - x0[d]) + eta * (x2[d] - x3[d])
                            })
                            .collect();
                        r[dofs[8 + m] as usize] =
                            fv[0] * jt[0] + fv[1] * jt[1];
                        // y-component at (1/2, t)
                        let (xi, eta) = (0.5, t);
                        let w = [
                            (1.0 - xi) * (1.0 - eta),
                            xi * (1.0 - eta),
                            xi * eta,
                            (1.0 - xi) * eta,
                        ];
                        let pt: Vec<f64> = (0..2)
                            .map(|d| w[0] * x0[d] + w[1] * x1[d] + w[2] * x2[d] + w[3] * x3[d])
                            .collect();
                        let fv = f(&pt);
                        // J(ξ,η)·(0,1) = ∂x/∂η = (1−ξ)(x3−x0) + ξ(x2−x1)
                        let jt: Vec<f64> = (0..2)
                            .map(|d| {
                                (1.0 - xi) * (x3[d] - x0[d]) + xi * (x2[d] - x1[d])
                            })
                            .collect();
                        r[dofs[10 + m] as usize] =
                            fv[0] * jt[0] + fv[1] * jt[1];
                    }
                }
            }
        } else if self.dim == 2 && k >= 3 && matches!(self.cell_type, ElementType::Tri3 | ElementType::Tri6) {
            // Tri NDk (k >= 3) interior DOFs: point values at the MFEM
            // `FE::Nodes` barycentric GL points with the reference tangents
            // (1,0) and (0,1) pushed through the affine map (`J·t̂`, MFEM's
            // `ND_TriangleElement` interior `dof2tk` pair) — the same nodal
            // semantics as the k = 2 path, minus the centroid-only shorthand.
            let n_interior = k * (k - 1);
            let tnd = TriNDk::new(k);
            let coords = tnd.dof_coords();
            let base = coords.len() - n_interior;
            for e in 0..n_elem as u32 {
                let dofs = self.element_dofs(e);
                let b_start = dofs.len() - n_interior;
                let nodes = self.mesh.element_nodes(e);
                let x0 = self.mesh.node_coords(nodes[0]);
                let x1 = self.mesh.node_coords(nodes[1]);
                let x2 = self.mesh.node_coords(nodes[2]);
                let j00 = x1[0] - x0[0];
                let j10 = x1[1] - x0[1];
                let j01 = x2[0] - x0[0];
                let j11 = x2[1] - x0[1];
                let r = result.as_slice_mut();
                for m in 0..n_interior / 2 {
                    let xi = &coords[base + 2 * m];
                    let pt = [x0[0] + j00 * xi[0] + j01 * xi[1], x0[1] + j10 * xi[0] + j11 * xi[1]];
                    let fv = f(&pt);
                    r[dofs[b_start + 2 * m] as usize] = j00 * fv[0] + j10 * fv[1];
                    r[dofs[b_start + 2 * m + 1] as usize] = j01 * fv[0] + j11 * fv[1];
                }
            }
        } else if self.dim == 2
            && k >= 3
            && matches!(self.cell_type, ElementType::Quad4 | ElementType::Quad8)
        {
            // Quad NDk (k >= 3) interior DOFs (D765): point values at the
            // MFEM `FE::Nodes` sites of `ND_QuadrilateralElement(k)` with each
            // slot's reference tangent pushed through the element's bilinear
            // map (`Φ(x)·(J·t̂)`, MFEM's `Project_ND`).  The paired element is
            // `fem_element::nedelec::QuadND` — the element the assembly
            // dispatcher selects for order >= 3 (`vec_ref_elem_choice`) and the
            // one whose `dof_coords()` the space's interior layout is built
            // from (`HCurlSpace::build`), so the slots and tangents below are
            // exactly the assemble-time functional list.  Before D765 this
            // branch did not exist at all: every 2-D quad NDk (k >= 3) interior
            // dof silently stayed 0.
            let n_interior = 2 * k * (k - 1);
            let qnd = QuadND::new(k);
            let coords = qnd.dof_coords();
            let tks = qnd.dof_tangents();
            let base = coords.len() - n_interior;
            for e in 0..n_elem as u32 {
                let dofs = self.element_dofs(e);
                let b_start = dofs.len() - n_interior;
                let nodes = self.mesh.element_nodes(e);
                let c: Vec<[f64; 2]> = (0..4)
                    .map(|i| {
                        let p = self.mesh.node_coords(nodes[i]);
                        [p[0], p[1]]
                    })
                    .collect();
                let r = result.as_slice_mut();
                for n in 0..n_interior {
                    let xi = &coords[base + n];
                    let t = tks[base + n];
                    let (x, y) = (xi[0], xi[1]);
                    // Bilinear quad map and its Jacobian at the dof site.
                    let w = [
                        (1.0 - x) * (1.0 - y),
                        x * (1.0 - y),
                        x * y,
                        (1.0 - x) * y,
                    ];
                    let pt = [
                        w[0] * c[0][0] + w[1] * c[1][0] + w[2] * c[2][0] + w[3] * c[3][0],
                        w[0] * c[0][1] + w[1] * c[1][1] + w[2] * c[2][1] + w[3] * c[3][1],
                    ];
                    let dx_dxi = [
                        (1.0 - y) * (c[1][0] - c[0][0]) + y * (c[2][0] - c[3][0]),
                        (1.0 - y) * (c[1][1] - c[0][1]) + y * (c[2][1] - c[3][1]),
                    ];
                    let dx_deta = [
                        (1.0 - x) * (c[3][0] - c[0][0]) + x * (c[2][0] - c[1][0]),
                        (1.0 - x) * (c[3][1] - c[0][1]) + x * (c[2][1] - c[1][1]),
                    ];
                    let fv = f(&pt);
                    // J·t with J = [dx/dξ dx/dη] column-wise.
                    let jt = [
                        t[0] * dx_dxi[0] + t[1] * dx_deta[0],
                        t[0] * dx_dxi[1] + t[1] * dx_deta[1],
                    ];
                    r[dofs[b_start + n] as usize] = fv[0] * jt[0] + fv[1] * jt[1];
                }
            }
        } else if self.dim == 3 && k >= 2 {
            // ── Shared tri-face DOFs (tet D38, prism/pyramid D547): point
            // values at the canonical (shared) face DOF points with the
            // canonical tangent pair — the face-creating element's
            // functionals, so the values are the canonical (global) dof
            // values directly.  Tet k ≥ 3 additionally owns interior dofs
            // (below); hex has no tri faces.
            for (&face_key, &first_dof) in &self.face_to_dof {
                let anchor = match self.face_anchor.get(&face_key) {
                    Some(a) => a,
                    None => continue,
                };
                let r = result.as_slice_mut();
                for p in 0..anchor.n_points() {
                    let x = anchor.point(p);
                    let [w0, w1] = anchor.tangents(p);
                    let fv = f(&x);
                    r[first_dof as usize + 2 * p] =
                        fv[0] * w0[0] + fv[1] * w0[1] + fv[2] * w0[2];
                    r[first_dof as usize + 2 * p + 1] =
                        fv[0] * w1[0] + fv[1] * w1[1] + fv[2] * w1[2];
                }
            }
            // ── Shared quad-face DOFs (hex, prism D547, pyramid D547): point
            // values at the canonical per-slot points/tangents of the
            // face-creating element.
            for (&key, &first_dof) in &self.quad_face_to_dof {
                let anchor = &self.quad_face_anchor[&key];
                let r = result.as_slice_mut();
                for (m, (x, t)) in anchor.nodes.iter().zip(anchor.tangents.iter()).enumerate() {
                    let fv = f(x);
                    r[first_dof as usize + m] = fv[0] * t[0] + fv[1] * t[1] + fv[2] * t[2];
                }
            }
            // ── Element-interior DOFs: element-owned, their functionals use
            // the owning element's map.
            if matches!(self.cell_type, ElementType::Tet4 | ElementType::Tet10) && k >= 3 {
                // Tet NDk (k ≥ 3) interior DOFs: point values at the MFEM
                // `FE::Nodes` barycentric GL points with the reference
                // tangents (1,0,0), (0,1,0), (0,0,1) pushed through the
                // element's affine map.
                let n_interior = k * (k - 1) * (k - 2) / 2;
                let tnd = TetNDk::new(k);
                let coords = tnd.dof_coords();
                let tks = tnd.dof_tangents();
                let off = coords.len() - n_interior;
                for e in 0..n_elem as u32 {
                    let nodes = self.mesh.element_nodes(e);
                    let dofs = self.element_dofs(e);
                    let base = dofs.len() - n_interior;
                    let p0 = self.mesh.node_coords(nodes[0]);
                    let mut jac = [[0.0_f64; 3]; 3];
                    for (c, lv) in [1usize, 2, 3].iter().enumerate() {
                        let p = self.mesh.node_coords(nodes[*lv]);
                        for d in 0..3 {
                            jac[d][c] = p[d] - p0[d];
                        }
                    }
                    let r = result.as_slice_mut();
                    for m in 0..n_interior {
                        let xi = &coords[off + m];
                        let mut x = [p0[0], p0[1], p0[2]];
                        for d in 0..3 {
                            x[d] += jac[d][0] * xi[0] + jac[d][1] * xi[1] + jac[d][2] * xi[2];
                        }
                        let fv = f(&x);
                        let tk = tks[off + m];
                        let t = [
                            jac[0][0] * tk[0] + jac[0][1] * tk[1] + jac[0][2] * tk[2],
                            jac[1][0] * tk[0] + jac[1][1] * tk[1] + jac[1][2] * tk[2],
                            jac[2][0] * tk[0] + jac[2][1] * tk[1] + jac[2][2] * tk[2],
                        ];
                        r[dofs[base + m] as usize] = fv[0] * t[0] + fv[1] * t[1] + fv[2] * t[2];
                    }
                }
            } else if matches!(self.cell_type, ElementType::Hex8 | ElementType::Hex20) {
                // Hex NDk interior DOFs: point values with the unnormalized
                // physical tangents `J t̂` (MFEM `Project_ND`, round-15 D36).
                let ndf_quad = 2 * k * (k - 1);
                let hnd = HexNDk::new(k);
                let coords = hnd.dof_coords();
                let tks = hnd.dof_tangents();
                let n_interior = 3 * k * (k - 1) * (k - 1);
                if n_interior > 0 {
                    let off = 12 * k + 6 * ndf_quad;
                    for e in 0..n_elem as u32 {
                        let verts8 = hex8_verts(&self.mesh, e);
                        let dofs = self.element_dofs(e);
                        let base = dofs.len() - n_interior;
                        let r = result.as_slice_mut();
                        for n in 0..n_interior {
                            let xi = &coords[off + n];
                            let (x, jac) = hex_trilinear_map(&verts8, xi);
                            let tau = tks[off + n];
                            let t = [
                                jac[0][0] * tau[0] + jac[0][1] * tau[1] + jac[0][2] * tau[2],
                                jac[1][0] * tau[0] + jac[1][1] * tau[1] + jac[1][2] * tau[2],
                                jac[2][0] * tau[0] + jac[2][1] * tau[1] + jac[2][2] * tau[2],
                            ];
                            let fv = f(&x);
                            r[dofs[base + n] as usize] =
                                fv[0] * t[0] + fv[1] * t[1] + fv[2] * t[2];
                        }
                    }
                }
            } else if self.cell_type == ElementType::Prism6 {
                // Prism NDk interior DOFs (D547): point values at the
                // MFEM `ND_WedgeElement` layout tail pushed through the
                // element's trilinear map, tangents `J·t̂` (MFEM frame).
                let n_interior = k * (k - 1) * (k - 1) + k * (k - 1) * (k - 2) / 2;
                if n_interior > 0 {
                    let pend = PrismNDk::new(k);
                    let layout = pend.mfem_layout_points();
                    let tks = prism_tangents_mfem(k);
                    let off = layout.len() - n_interior;
                    for e in 0..n_elem as u32 {
                        let verts6 = prism6_verts(&self.mesh, e);
                        let dofs = self.element_dofs(e);
                        let base = dofs.len() - n_interior;
                        let r = result.as_slice_mut();
                        for n in 0..n_interior {
                            let xi = layout[off + n];
                            let x = prism_trilinear_map(&verts6, xi[0], xi[1], xi[2]);
                            let j = prism_map_jacobian(&verts6, xi[0], xi[1], xi[2]);
                            let t = j_mul(&j, &tks[off + n]);
                            let fv = f(&x);
                            r[dofs[base + n] as usize] =
                                fv[0] * t[0] + fv[1] * t[1] + fv[2] * t[2];
                        }
                    }
                }
            } else if self.cell_type == ElementType::Pyramid5 {
                // Pyramid NDk interior DOFs (D547): the Fuentes layout tail
                // through the collapsed map, tangents `J·tk`.
                let n_interior = 3 * k * (k - 1) * (k - 1);
                if n_interior > 0 {
                    let pynd = PyraNDk::new(k);
                    let layout = pynd.mfem_layout_points();
                    let tks = pynd.dof_tangents();
                    let off = layout.len() - n_interior;
                    for e in 0..n_elem as u32 {
                        let verts5 = pyramid5_verts(&self.mesh, e);
                        let dofs = self.element_dofs(e);
                        let base = dofs.len() - n_interior;
                        let r = result.as_slice_mut();
                        for n in 0..n_interior {
                            let xi = layout[off + n];
                            let x = pyramid_map(&verts5, xi[0], xi[1], xi[2]);
                            let j = pyramid_map_jacobian(&verts5, xi[0], xi[1], xi[2]);
                            let t = j_mul(&j, &tks[off + n]);
                            let fv = f(&x);
                            r[dofs[base + n] as usize] =
                                fv[0] * t[0] + fv[1] * t[1] + fv[2] * t[2];
                        }
                    }
                }
            }
        }
        result
    }
}

impl<M: MeshTopology> FESpace for HCurlSpace<M> {
    type Mesh = M;

    fn mesh(&self) -> &M { &self.mesh }

    fn n_dofs(&self) -> usize { self.n_dofs }

    fn element_dofs(&self, elem: u32) -> &[DofId] {
        let start = self.elem_offsets[elem as usize];
        let end = self.elem_offsets[elem as usize + 1];
        &self.dofs_flat[start..end]
    }

    fn interpolate(&self, _f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        Vector::zeros(self.n_dofs)
    }

    fn space_type(&self) -> SpaceType { SpaceType::HCurl }

    fn order(&self) -> u8 { self.order }

    fn element_signs(&self, elem: u32) -> Option<&[f64]> {
        Some(self.element_signs(elem))
    }

    // D58: forward to the inherent method (tet NDk (k ≥ 2) face blocks; empty
    // for 2-D, hex and k = 1).  Calls the inherent fn explicitly so the trait
    // method does not recurse into itself.
    fn element_face_blocks(&self, elem: u32) -> &[FaceDofBlock] {
        HCurlSpace::element_face_blocks(self, elem)
    }

    fn element_face_pair_transforms(&self, elem: u32) -> Vec<FacePairTransform> {
        HCurlSpace::element_face_pair_transforms(self, elem)
    }

    fn facet_slots_against_published_anchor(
        &self,
        e: u32,
        facet_verts: &[u32],
        anchor_et: ElementType,
        anchor_verts: &[u32],
        global_node: &dyn Fn(u32) -> u32,
        anchor_coords: &dyn Fn(u32) -> Option<[f64; 3]>,
    ) -> Option<Vec<FacetSlotToAnchor>> {
        HCurlSpace::facet_slots_against_published_anchor(
            self,
            e,
            facet_verts,
            anchor_et,
            anchor_verts,
            global_node,
            anchor_coords,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use fem_core::{ElemId, FaceId, NodeId};
    use fem_mesh::Mesh;

    #[derive(Clone)]
    struct OneQuadMesh {
        nodes: Vec<[f64; 2]>,
        elem: [NodeId; 4],
        bfaces: Vec<[NodeId; 2]>,
        btags: Vec<i32>,
    }

    impl OneQuadMesh {
        fn unit() -> Self {
            Self {
                nodes: vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
                elem: [0, 1, 2, 3],
                bfaces: vec![[0, 1], [1, 2], [2, 3], [3, 0]],
                btags: vec![1, 2, 3, 4],
            }
        }
    }

    impl MeshTopology for OneQuadMesh {
        fn dim(&self) -> u8 { 2 }
        fn n_nodes(&self) -> usize { self.nodes.len() }
        fn n_elements(&self) -> usize { 1 }
        fn n_boundary_faces(&self) -> usize { self.bfaces.len() }
        fn element_nodes(&self, _elem: ElemId) -> &[NodeId] { &self.elem }
        fn element_type(&self, _elem: ElemId) -> ElementType { ElementType::Quad4 }
        fn element_tag(&self, _elem: ElemId) -> i32 { 1 }
        fn node_coords(&self, node: NodeId) -> &[f64] { &self.nodes[node as usize] }
        fn face_nodes(&self, face: FaceId) -> &[NodeId] { &self.bfaces[face as usize] }
        fn face_tag(&self, face: FaceId) -> i32 { self.btags[face as usize] }
        fn face_elements(&self, _face: FaceId) -> (ElemId, Option<ElemId>) { (0, None) }
    }

    #[derive(Clone)]
    struct OneHexMesh {
        nodes: Vec<[f64; 3]>,
        elem: [NodeId; 8],
        bfaces: Vec<[NodeId; 4]>,
        btags: Vec<i32>,
    }

    impl OneHexMesh {
        fn unit() -> Self {
            Self {
                nodes: vec![
                    [0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,0.0],
                    [0.0,0.0,1.0],[1.0,0.0,1.0],[1.0,1.0,1.0],[0.0,1.0,1.0],
                ],
                elem: [0,1,2,3,4,5,6,7],
                bfaces: vec![[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]],
                btags: vec![1,2,3,4,5,6],
            }
        }
    }

    impl MeshTopology for OneHexMesh {
        fn dim(&self) -> u8 { 3 }
        fn n_nodes(&self) -> usize { self.nodes.len() }
        fn n_elements(&self) -> usize { 1 }
        fn n_boundary_faces(&self) -> usize { self.bfaces.len() }
        fn element_nodes(&self, _elem: ElemId) -> &[NodeId] { &self.elem }
        fn element_type(&self, _elem: ElemId) -> ElementType { ElementType::Hex8 }
        fn element_tag(&self, _elem: ElemId) -> i32 { 1 }
        fn node_coords(&self, node: NodeId) -> &[f64] { &self.nodes[node as usize] }
        fn face_nodes(&self, face: FaceId) -> &[NodeId] { &self.bfaces[face as usize] }
        fn face_tag(&self, face: FaceId) -> i32 { self.btags[face as usize] }
        fn face_elements(&self, _face: FaceId) -> (ElemId, Option<ElemId>) { (0, None) }
    }

    #[test]
    fn hcurl_dof_count_tri() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);
        assert_eq!(space.element_dofs(0).len(), 3);
        // Each triangle has 3 edges, 32 triangles, but edges are shared.
        // Expected: 56 unique edges.
        assert_eq!(space.n_dofs(), 56, "n_dofs should equal number of unique edges");
    }

    #[test]
    fn hcurl_shared_edge_dof() {
        // 1×1 mesh → 2 triangles sharing the diagonal edge.
        let mesh = Mesh::<2>::unit_square_tri(1);
        let space = HCurlSpace::new(mesh, 1);
        assert_eq!(space.mesh().n_elements(), 2);

        let dofs0 = space.element_dofs(0);
        let dofs1 = space.element_dofs(1);

        // At least one DOF must be shared between the two elements.
        let shared: Vec<_> = dofs0.iter().filter(|d| dofs1.contains(d)).collect();
        assert!(!shared.is_empty(), "adjacent triangles must share at least one edge DOF");
    }

    #[test]
    fn hcurl_signs_consistent_on_shared_edge() {
        // Two triangles sharing an edge: verify signs are well-defined (±1)
        // and that both elements reference the same global DOF.
        // Note: signs are NOT necessarily opposite — they are both relative
        // to the global edge orientation (min→max vertex ID).
        let mesh = Mesh::<2>::unit_square_tri(1);
        let space = HCurlSpace::new(mesh, 1);

        let dofs0 = space.element_dofs(0);
        let signs0 = space.element_signs(0);
        let dofs1 = space.element_dofs(1);
        let signs1 = space.element_signs(1);

        // All signs must be ±1.
        for s in signs0.iter().chain(signs1.iter()) {
            assert!((s.abs() - 1.0).abs() < 1e-14, "sign must be ±1, got {s}");
        }

        // At least one shared DOF.
        let shared: Vec<_> = dofs0.iter().filter(|d| dofs1.contains(d)).collect();
        assert!(!shared.is_empty(), "adjacent triangles must share at least one edge DOF");
    }

    #[test]
    fn hcurl_space_type() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let space = HCurlSpace::new(mesh, 1);
        assert_eq!(space.space_type(), SpaceType::HCurl);
    }

    #[test]
    fn hcurl_dof_count_quad_nd1() {
        let mesh = OneQuadMesh::unit();
        let space = HCurlSpace::new(mesh, 1);
        assert_eq!(space.element_dofs(0).len(), 4);
        assert_eq!(space.n_dofs(), 4);
    }

    #[test]
    fn hcurl_dof_count_hex_nd1() {
        let mesh = OneHexMesh::unit();
        let space = HCurlSpace::new(mesh, 1);
        assert_eq!(space.element_dofs(0).len(), 12);
        assert_eq!(space.n_dofs(), 12);
    }

    #[test]
    fn hcurl_dof_count_quad_nd2() {
        let mesh = OneQuadMesh::unit();
        let space = HCurlSpace::new(mesh, 2);
        assert_eq!(space.element_dofs(0).len(), 12, "QuadND2: 8 edge + 4 interior");
        assert_eq!(space.n_dofs(), 12);
    }

    #[test]
    fn hcurl_dof_count_hex_nd2() {
        let mesh = OneHexMesh::unit();
        let space = HCurlSpace::new(mesh, 2);
        assert_eq!(space.element_dofs(0).len(), 54, "HexND2: 24 edge + 30 face/interior");
        assert_eq!(space.n_dofs(), 54);
    }

    #[test]
    fn hcurl_interpolate_vector_constant_quad_nd1() {
        let mesh = OneQuadMesh::unit();
        let space = HCurlSpace::new(mesh, 1);
        let v = space.interpolate_vector(&|_x| vec![1.0, 0.0]);

        let vals = v.as_slice();
        assert_eq!(vals.len(), 4);
        assert!((vals[0] - 2.0).abs() < 1e-12);
        assert!(vals[1].abs() < 1e-12);
        assert!((vals[2] + 2.0).abs() < 1e-12);
        assert!(vals[3].abs() < 1e-12);
    }

    #[test]
    fn hcurl_interpolate_vector_constant() {
        // Interpolate a constant vector field F = (1, 0).
        // DOF value on each edge = F · tangent = tangent_x.
        let mesh = Mesh::<2>::unit_square_tri(2);
        let space = HCurlSpace::new(mesh, 1);
        let v = space.interpolate_vector(&|_x| vec![1.0, 0.0]);
        // All DOF values should be finite and within the range of edge lengths.
        for &val in v.as_slice() {
            assert!(val.is_finite(), "interpolated value should be finite");
        }
    }

    #[test]
    fn hcurl_nd2_tet_local_dof_layout() {
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let space = HCurlSpace::new(mesh, 2);

        assert_eq!(space.element_dofs(0).len(), 20, "TetND2 should have 20 local DOFs");
        assert_eq!(space.element_signs(0).len(), 20, "TetND2 sign array length mismatch");
    }

    #[test]
    fn hcurl_nd2_tet_global_dof_count_matches_edges_faces() {
        let mesh = Mesh::<3>::unit_cube_tet(2);

        let mut edges: HashSet<EdgeKey> = HashSet::new();
        let mut faces: HashSet<FaceKey> = HashSet::new();
        for e in 0..mesh.n_elements() as u32 {
            let ns = mesh.element_nodes(e);
            for &(i, j) in &TET_EDGES {
                edges.insert(EdgeKey::new(ns[i], ns[j]));
            }
            for &(i, j, k) in &TET_FACES {
                faces.insert(FaceKey::new(ns[i], ns[j], ns[k]));
            }
        }

        let space = HCurlSpace::new(mesh, 2);
        let expected = 2 * edges.len() + 2 * faces.len();
        assert_eq!(space.n_dofs(), expected, "ND2 3D global DOF count should be 2*n_edges + 2*n_faces");
    }

    // ─── ND3+ tests ───────────────────────────────────────────────────────────

    #[test]
    fn hcurl_nd3_quad_dof_count() {
        let mesh = OneQuadMesh::unit();
        let space = HCurlSpace::new(mesh, 3);
        assert_eq!(space.element_dofs(0).len(), 24, "QuadND3: 12 edge + 12 interior");
        assert_eq!(space.n_dofs(), 24);
    }

    #[test]
    fn hcurl_nd3_hex_dof_count() {
        let mesh = OneHexMesh::unit();
        let space = HCurlSpace::new(mesh, 3);
        assert_eq!(space.element_dofs(0).len(), 144, "HexND3: 36 edge + 108 face/interior");
        assert_eq!(space.n_dofs(), 144);
    }

    #[test]
    fn hcurl_nd3_tet_dof_count() {
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let space = HCurlSpace::new(mesh, 3);
        assert_eq!(space.element_dofs(0).len(), 45, "TetND3: k*(k+2)*(k+3)/2 = 45");
    }

    #[test]
    fn hcurl_nd3_tri_dof_count() {
        let mesh = Mesh::<2>::unit_square_tri(1);
        let space = HCurlSpace::new(mesh, 3);
        assert_eq!(space.element_dofs(0).len(), 15, "TriND3: k*(k+2) = 15");
    }

    #[test]
    fn hcurl_nd4_hex_dof_count() {
        let mesh = OneHexMesh::unit();
        let space = HCurlSpace::new(mesh, 4);
        assert_eq!(space.element_dofs(0).len(), 300, "HexND4: 48 edge + 252 face/interior");
        assert_eq!(space.n_dofs(), 300);
    }

    #[test]
    fn hcurl_nd3_interpolate_linear_field() {
        let mesh = OneQuadMesh::unit();
        let space = HCurlSpace::new(mesh, 3);
        let v = space.interpolate_vector(&|_x| vec![1.0, 0.0]);
        assert_eq!(v.as_slice().len(), 24, "QuadND3: 24 DOFs");
        for &val in v.as_slice() { assert!(val.is_finite()); }
    }
}
