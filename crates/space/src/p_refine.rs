//! p-Refinement infrastructure for finite element spaces.
//!
//! Provides variable-order DOF management, p-refinement/derefinement
//! operations, constraint detection at mixed-order interfaces (MFEM
//! `FiniteElementSpace::VariableOrderMinimumRule` semantics), and order field
//! smoothing.
//!
//! ## Key concepts (MFEM variable-order "variant" scheme)
//!
//! In p-refinement, different elements can have different polynomial orders.
//! Every topological entity (edge/face) holds **one DOF set ("variant") per
//! distinct element order of its adjacent elements**, with the DOFs placed at
//! that order's own basis node positions. An element references
//! the variant matching its own order — exactly MFEM's
//! `var_edge_dofs`/`var_face_dofs` tables (MFEM keys variants by order only,
//! never per cell family — probe `tmp/d86a/out_p33_gl.txt`).
//!
//! The 1-D node set of a variant follows the **family of the elements that
//! use it** (D827-1): closed Gauss-Lobatto for the tri/quad/hex/prism/
//! pyramid bases (`H1TriPk`/`QuadQk`/`HexQk`/`H1PrismPk`/
//! `H1FuentesPyramidPk`), equispaced for the tet (`TetPk`) — the
//! first-encountering element of that (entity, order) donates the basis if
//! two families share one variant.  MFEM itself keeps a collection-global
//! entity basis (its hp constraints evaluate
//! `fec->GetFE(Geometry::SEGMENT/TRIANGLE, p)`, `fem/fespace.cpp:1094` —
//! probe `tmp/d86a/mixed_edge_probe.cpp`: flipping only the collection basis
//! flips every row GLL ↔ equispaced); fem-rs's per-family cell bases make
//! the variant-level dispatch the faithful translation.
//!
//! ## Constraint generation (MFEM `BuildConformingInterpolation`)
//!
//! * Conforming entities holding multiple variants (MFEM
//!   `VariableOrderMinimumRule`): the lowest-order variant is the master; every
//!   higher-order variant DOF is constrained to interpolate the lowest-order
//!   trace at the slave DOF's own node position: `u_i = Σ_j L_j^{(p0)}(t_i) u_j`
//!   with `L_j` the master variant's nodal basis (per its family, D827-1)
//!   evaluated at the higher-variant node position `t_i`.
//! * Non-conforming (hanging) 2D edges: a *slave* edge strictly inside a
//!   *master* edge has **all** its variant DOFs (and its hanging endpoint
//!   vertices) constrained to the master edge's lowest-variant interpolation
//!   evaluated at the slave DOF's global position on the master edge. This
//!   composes MFEM's `OrientedPointMatrix` + `GetTransferMatrix` aliasing in
//!   one step (identical conforming space).
//!
//! ## Supported geometries
//!
//! * 2D triangles/quadrilaterals: full support, including NC (hanging) quad
//!   meshes — this powers the hp-refinement loop.
//! * 3D tets/hexes: conforming variable-order (min-rule) on edges and faces.
//!   NC 3D is not supported (recorded limitation).
//! * 3D prisms (wedges): conforming variable-order on edges, triangular and
//!   quadrilateral faces, and interiors, in MFEM `H1_WedgeElement`'s entity
//!   order (the layout MFEM's variable-order `GetElementDofs`
//!   (`fem/fespace.cpp:3428`) produces — MFEM-verified by probe
//!   `tmp/d174_prism_p_probe.cpp`, which dumps `GetElementDofs` rows and the
//!   mixed-order conforming interpolation of a variable-order H1 space on an
//!   `EnsureNCMesh` wedge mesh).  The element rows pair with
//!   [`fem_element::lagrange::H1PrismPk`]'s slot order up to the space-level
//!   face orientation convention (see `build_variable_order_dof_manager`).
//! * 3D pyramids: conforming variable-order in MFEM's default
//!   `H1_FuentesPyramidElement` entity order — 8 edge blocks in
//!   `Geometry::Constants<PYRAMID>::Edges` order/direction, the base quad
//!   face (face 0) in Fuentes' `(cp[i], cp[p−j])` j-reversed layout, 4
//!   triangular side faces in `Faces` order at `H1TriPk` GLL barycentrics,
//!   `(p−1)³` interior dofs (`fe_h1.cpp:1070-1150`).  MFEM-verified by probe
//!   `tmp/d85c/pyr_hp_probe.cpp` (D824-A): rows, NDofs and the min-rule
//!   constraints of a mixed-order pyramid triple on `EnsureNCMesh`.

use std::collections::{BTreeMap, HashMap, HashSet};
use fem_core::types::{DofId, NodeId};
use fem_element::lagrange::{H1TriPk, PRISM_EDGES};
use fem_element::ReferenceElement;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use crate::dof_manager::{quadratic_row_corner_slots, DofManager, EdgeKey, FaceKey};

// ─── PRefineConstraint ────────────────────────────────────────────────────────

/// A constraint arising from p-refinement (MFEM "slave" DOF dependency).
///
/// Expresses a high-order (or hanging) DOF as a weighted combination of
/// lower-order DOFs on the same entity, maintaining C⁰ continuity across
/// mixed-order interfaces and non-conforming edges.
///
/// For a conforming edge shared by P3 and P2 elements, each P3-variant edge
/// DOF is: `u_extra = Σ_j L_j^{(2)}(t_extra) u_j` over the P2-variant
/// (vertex, mid, vertex) DOFs.
#[derive(Debug, Clone)]
pub struct PRefineConstraint {
    /// The DOF being constrained (belongs to a higher-order variant or a
    /// hanging entity).
    pub constrained: DofId,
    /// Parent DOFs and their weights (sparse representation).
    pub parents: Vec<(DofId, f64)>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Lagrange interpolation weights over the equispaced `p_low + 1` nodes
/// on [0, 1] evaluated at position `t` (kept for the equispaced TetPk path
/// and unit tests).
///
/// Nodes: t_j = j / p_low for j = 0, 1, ..., p_low.
fn lagrange_weights_1d(t: f64, p_low: u8) -> Vec<f64> {
    let p = p_low as usize;
    let mut weights = Vec::with_capacity(p + 1);
    for j in 0..=p {
        let tj = j as f64 / p as f64;
        let mut w = 1.0;
        for m in 0..=p {
            if m != j {
                let tm = m as f64 / p as f64;
                w *= (t - tm) / (tj - tm);
            }
        }
        weights.push(w);
    }
    weights
}

/// Lagrange interpolation weights over arbitrary ascending nodes (including
/// endpoints) evaluated at `t`.  Weights follow the input order.
fn lagrange_weights_at(nodes: &[f64], t: f64) -> Vec<f64> {
    let n = nodes.len();
    let mut weights = Vec::with_capacity(n);
    for j in 0..n {
        let mut w = 1.0;
        for m in 0..n {
            if m != j {
                w *= (t - nodes[m]) / (nodes[j] - nodes[m]);
            }
        }
        weights.push(w);
    }
    weights
}

/// Gauss-Lobatto node positions on [0,1] for order `p` (`p+1` points,
/// endpoints included, ascending).  Matches the H1 (Gauss-Lobatto) basis of
/// `QuadQk`/`HexQk`/`H1TriPk` (MFEM `H1_FECollection` default basis).
fn gll_positions_01(p: u8) -> Vec<f64> {
    let (pts, _) = fem_element::quadrature::gauss_lobatto_arbitrary(p as usize + 1);
    pts.iter().map(|&x| 0.5 * (x + 1.0)).collect()
}

/// Equispaced node positions on [0,1] for order `p` (`p+1` points).
fn equi_positions_01(p: u8) -> Vec<f64> {
    (0..=p as usize).map(|j| j as f64 / p as f64).collect()
}

/// Interior (non-vertex) node positions for the entity DOF set of order `p`.
fn interior_positions_1d(p: u8, gll: bool) -> Vec<f64> {
    let all = if gll { gll_positions_01(p) } else { equi_positions_01(p) };
    all[1..all.len() - 1].to_vec()
}

/// Whether the H¹ basis on `et` cells uses Gauss-Lobatto node sets (`true`
/// for 2D tri/quad and 3D hex/prism/pyramid — `H1TriPk`/`QuadQk`/`HexQk`/
/// `H1PrismPk`/`H1FuentesPyramidPk`) or equispaced (3D tet via `TetPk`).
/// Family-based since D824: a Quad8 row (8 nodes) is a SQUARE cell, not a
/// hex, and a Tet10 row is a TET.
///
/// The pyramid arm is D824-C (probe-pinned, not inherited): MFEM's default
/// pyramid H¹ element is `H1_FuentesPyramidElement`, whose node table is
/// built from `Poly1D::ClosedPoints(p, GaussLobatto)` (`fe_h1.cpp:1043`), and
/// the probe dump `tmp/d85c/out_p311.txt` shows p3 edge-interior nodes at
/// `0.5·(1 ∓ 1/√5)` — the closed GLL points, not the equispaced `1/3, 2/3`.
fn elem_uses_gll(et: ElementType) -> bool {
    match et {
        ElementType::Tet4 | ElementType::Tet10 => false,
        ElementType::Tri3 | ElementType::Tri6
        | ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9
        | ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27
        | ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18
        | ElementType::Pyramid5 | ElementType::Pyramid13 => true,
        _ => false,
    }
}

/// The row's corner nodes in local vertex order: the whole row for a linear
/// family, the family's corner slots ([`quadratic_row_corner_slots`]) for a
/// quadratic row (D824 — a Quad8/Quad9/Tet10/Hex20/Hex27/Prism15 row is
/// corner-prefixed, a Prism18 row's corners sit at slots `[0, 1, 2, 12,
/// 13, 14]` of its layer-major `PrismPk` layout).
fn elem_corners(et: ElementType, ns: &[NodeId], curved_geometry: bool) -> Vec<NodeId> {
    match quadratic_row_corner_slots(et, curved_geometry) {
        Some(slots) => slots.iter().map(|&s| ns[s]).collect(),
        None => ns.to_vec(),
    }
}

fn is_tri_row(et: ElementType) -> bool {
    matches!(et, ElementType::Tri3 | ElementType::Tri6)
}
fn is_quad_row(et: ElementType) -> bool {
    matches!(et, ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9)
}
fn is_tet_row(et: ElementType) -> bool {
    matches!(et, ElementType::Tet4 | ElementType::Tet10)
}
fn is_prism_row(et: ElementType) -> bool {
    matches!(et, ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18)
}
fn is_hex_row(et: ElementType) -> bool {
    matches!(et, ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27)
}
fn is_pyramid_row(et: ElementType) -> bool {
    matches!(et, ElementType::Pyramid5 | ElementType::Pyramid13)
}

/// Extract the local-orientation edges of a 2D triangle: (0,1), (1,2), (2,0)
/// (MFEM/H1TriPk edge order: the last edge runs from v2 to v0).
fn tri_edges(ns: &[NodeId]) -> Vec<(NodeId, NodeId)> {
    vec![(ns[0], ns[1]), (ns[1], ns[2]), (ns[2], ns[0])]
}

/// Extract the local-orientation edges of a 3D tetrahedron.
fn tet_edges(ns: &[NodeId]) -> Vec<(NodeId, NodeId)> {
    vec![
        (ns[0], ns[1]), (ns[0], ns[2]), (ns[0], ns[3]),
        (ns[1], ns[2]), (ns[1], ns[3]), (ns[2], ns[3]),
    ]
}

/// Local-orientation edges of any supported element, over the row's **corner**
/// nodes (`ns` = [`elem_corners`]; quadratic rows must not mint edges onto
/// their mid-node geometry dofs).
fn elem_local_edges(et: ElementType, ns: &[NodeId]) -> Vec<(NodeId, NodeId)> {
    match et.dim() {
        2 if is_quad_row(et) => {
            // Quad: bottom, right, top (v2→v3), left (v3→v0) — QuadQk order.
            vec![(ns[0], ns[1]), (ns[1], ns[2]), (ns[2], ns[3]), (ns[3], ns[0])]
        }
        2 => tri_edges(ns),
        3 if is_prism_row(et) => {
            // Prism: the 9 edges in MFEM `Geometry::Constants<PRISM>::Edges`
            // order (bottom tri 0-2, top tri 3-5, verticals 6-8) — the
            // `H1PrismPk` edge-block order.
            PRISM_EDGES.iter().map(|e| (ns[e[0]], ns[e[1]])).collect()
        }
        3 if is_hex_row(et) => {
            // Hex: 12 edges in HexQk order.
            vec![
                (ns[0], ns[1]), (ns[1], ns[2]), (ns[2], ns[3]), (ns[3], ns[0]),
                (ns[4], ns[5]), (ns[5], ns[6]), (ns[6], ns[7]), (ns[7], ns[4]),
                (ns[0], ns[4]), (ns[1], ns[5]), (ns[2], ns[6]), (ns[3], ns[7]),
            ]
        }
        3 if is_pyramid_row(et) => {
            // Pyramid: 8 edges in MFEM `Geometry::Constants<PYRAMID>::Edges`
            // order and direction (fem/geom.cpp:1076): base (0,1) (1,2)
            // (3,2) (0,3) — note the two reversed base edges — then the
            // laterals (0,4) (1,4) (2,4) (3,4); the `H1_FuentesPyramidElement`
            // edge-block order (fe_h1.cpp:1070-1102, probe
            // `tmp/d85c/out_p311.txt`: the p3 row lists the (3,2)-edge dofs
            // reversed against the stored ascending order).
            vec![
                (ns[0], ns[1]), (ns[1], ns[2]), (ns[3], ns[2]), (ns[0], ns[3]),
                (ns[0], ns[4]), (ns[1], ns[4]), (ns[2], ns[4]), (ns[3], ns[4]),
            ]
        }
        3 => tet_edges(ns),
        _ => panic!("elem_local_edges: unsupported dimension for {et:?}"),
    }
}

/// Extract the faces of a 3D tetrahedron (TetPk face order).
fn tet_faces(ns: &[NodeId]) -> Vec<(NodeId, NodeId, NodeId)> {
    vec![
        (ns[0], ns[1], ns[2]),
        (ns[0], ns[1], ns[3]),
        (ns[0], ns[2], ns[3]),
        (ns[1], ns[2], ns[3]),
    ]
}

/// Triangular faces of a prism in MFEM `H1_WedgeElement` face order
/// (face 0 = bottom, face 1 = top).
fn prism_tri_faces(ns: &[NodeId]) -> Vec<(NodeId, NodeId, NodeId)> {
    vec![(ns[0], ns[1], ns[2]), (ns[3], ns[4], ns[5])]
}

/// Quadrilateral side faces of a prism in the `H1_WedgeElement` face order
/// (faces 2..4): `(0,1,4,3) (1,2,5,4) (2,0,3,5)` in the element's local
/// parameterisation (MFEM `Geometry::Constants<PRISM>::Faces`).
fn prism_quad_faces(ns: &[NodeId]) -> Vec<[NodeId; 4]> {
    vec![
        [ns[0], ns[1], ns[4], ns[3]],
        [ns[1], ns[2], ns[5], ns[4]],
        [ns[2], ns[0], ns[3], ns[5]],
    ]
}

/// Base quad face of a pyramid (MFEM `Geometry::Constants<PYRAMID>` face 0),
/// listed in the element's own corner cycle `(v0, v1, v2, v3)` — the frame
/// the Fuentes base-face layout `(cp[i], cp[p−j])` lives in
/// (`fe_h1.cpp:1112-1121`).
fn pyramid_base_face(ns: &[NodeId]) -> [NodeId; 4] {
    [ns[0], ns[1], ns[2], ns[3]]
}

/// Triangular side faces of a pyramid in MFEM
/// `Geometry::Constants<PYRAMID>::Faces` order (faces 1..4):
/// `(0,1,4) (1,2,4) (2,3,4) (3,0,4)` — the `H1_FuentesPyramidElement`
/// tri-face block order (fe_h1.cpp:1123-1144).
fn pyramid_tri_faces(ns: &[NodeId]) -> Vec<(NodeId, NodeId, NodeId)> {
    vec![
        (ns[0], ns[1], ns[4]),
        (ns[1], ns[2], ns[4]),
        (ns[2], ns[3], ns[4]),
        (ns[3], ns[0], ns[4]),
    ]
}

/// Extract the 4-node faces of a hexahedron (ordered for factory HexQk).
fn hex_quad_faces(ns: &[NodeId]) -> Vec<[NodeId; 4]> {
    vec![
        [ns[0], ns[1], ns[2], ns[3]],  // bottom (z=0)
        [ns[4], ns[7], ns[6], ns[5]],  // top (z=1), reversed for outward normal
        [ns[0], ns[4], ns[5], ns[1]],  // front (y=0)
        [ns[2], ns[3], ns[7], ns[6]],  // back (y=1), reversed
        [ns[0], ns[3], ns[7], ns[4]],  // left (x=0)
        [ns[1], ns[5], ns[6], ns[2]],  // right (x=1)
    ]
}

/// Cyclically canonical orientation of a quadrilateral face.
///
/// Two elements sharing a quad face list it with opposite cycles, so any
/// orientation-dependent key (e.g. the sorted *first three* of the local
/// ordered list) would split the face into per-side DOF sets.  This picks the
/// lexicographically smallest `(f0, f1, f3)` over all 8 rotations of the
/// cycle and its reverse — a pure function of the vertex set, so both sides
/// of a shared face agree.
fn canon_quad_face(f: [NodeId; 4]) -> [NodeId; 4] {
    let mut best = f;
    for start in [&f, &[f[0], f[3], f[2], f[1]]] {
        for r in 0..4 {
            let rot = [start[r], start[(r + 1) % 4], start[(r + 2) % 4], start[(r + 3) % 4]];
            if (rot[0], rot[1], rot[3]) < (best[0], best[1], best[3]) {
                best = rot;
            }
        }
    }
    best
}

/// Rising-factorial basis L_n(t) = Π_{a=0}^{n-1} (t-a)/(n-a), with L₀=1.
fn rising_val(n: usize, t: f64) -> f64 {
    if n == 0 { return 1.0; }
    let mut val = 1.0;
    for a in 0..n {
        val *= (t - a as f64) / (n as f64 - a as f64);
    }
    val
}

/// 2D Lagrange interpolation weights on a reference triangle (equispaced
/// nodes, TetPk/TriPk factory DOF ordering).
///
/// Evaluates the p-th-order Lagrange basis on the reference triangle
/// at barycentric position (r, s) where r,s ≥ 0, r+s ≤ 1.
/// Returns a vector of weights for each DOF of a TriPk(p) element,
/// in the factory DOF ordering.
fn lagrange_weights_tri(r: f64, s: f64, p: u8) -> Vec<f64> {
    let p = p as usize;
    let pf = p as f64;
    let t0 = pf * r;
    let t1 = pf * s;
    let t2 = pf * (1.0 - r - s);
    let n_dofs = (p + 1) * (p + 2) / 2;
    let mut weights = vec![0.0; n_dofs];

    // Reuse DOF ordering from factory TriPk: (i, j, k) with i+j+k = p
    // Ordered: vertices, then edge 0-1, edge 1-2, edge 2-0, then face interior
    let mut idx = 0usize;
    // Vertex 0: (p, 0, 0)
    weights[idx] = rising_val(p, t0) * rising_val(0, t1) * rising_val(0, t2); idx += 1;
    // Vertex 1: (0, p, 0)
    weights[idx] = rising_val(0, t0) * rising_val(p, t1) * rising_val(0, t2); idx += 1;
    // Vertex 2: (0, 0, p)
    weights[idx] = rising_val(0, t0) * rising_val(0, t1) * rising_val(p, t2); idx += 1;
    if p > 1 {
        // Edge 0-1: (p-k, k, 0) for k=1..p-1
        for k in 1..p { let i = p - k; weights[idx] = rising_val(i, t0) * rising_val(k, t1) * rising_val(0, t2); idx += 1; }
        // Edge 1-2: (0, p-k, k) for k=1..p-1
        for k in 1..p { let j = p - k; weights[idx] = rising_val(0, t0) * rising_val(j, t1) * rising_val(k, t2); idx += 1; }
        // Edge 2-0: (k, 0, p-k) for k=1..p-1
        for k in 1..p { let i = k; weights[idx] = rising_val(i, t0) * rising_val(0, t1) * rising_val(p-k, t2); idx += 1; }
    }
    if p >= 3 {
        // Face-interior: (i, j, p-i-j) for i=1..p-2, j=1..p-1-i
        for j in 1..=p-2 {
            for i in 1..=p-1-j {
                let k = p - i - j;
                weights[idx] = rising_val(i, t0) * rising_val(j, t1) * rising_val(k, t2);
                idx += 1;
            }
        }
    }
    debug_assert_eq!(idx, n_dofs);
    weights
}

// ═══════════════════════════════════════════════════════════════════════════════
// Entity DOF counts
// ═══════════════════════════════════════════════════════════════════════════════

/// Number of interior DOFs for a 2D element (Tri bubble or Quad face) of order p.
/// Tri: (p-1)(p-2)/2 for p≥3
/// Quad: (p-1)² for p≥2
fn n_face_dofs_2d(et: ElementType, p: u8) -> usize {
    let p = p as usize;
    if is_quad_row(et) {
        // Quad interior DOFs
        if p >= 2 { (p - 1) * (p - 1) } else { 0 }
    } else {
        // Tri bubble DOFs
        if p >= 3 { (p - 1) * (p - 2) / 2 } else { 0 }
    }
}

/// Number of face-interior DOFs for a 3D face of order p.
/// Tri face: (p-1)(p-2)/2 for p≥3
/// Quad face: (p-1)² for p≥2
fn n_face_dofs_3d(ns_len: usize, p: u8) -> usize {
    let p = p as usize;
    if ns_len == 4 {
        // Quad face of a hex
        if p >= 2 { (p - 1) * (p - 1) } else { 0 }
    } else {
        // Tri face of a tet
        if p >= 3 { (p - 1) * (p - 2) / 2 } else { 0 }
    }
}

/// Volume-interior DOFs of a 3D element.
/// Tet: (p-1)(p-2)(p-3)/6 (p≥4)
/// Prism: (p-1)·(p-1)(p-2)/2 (p≥3)
/// Hex: (p-1)³ (p≥2)
/// Pyramid: (p-1)³ (p≥2, Fuentes' tensor bubble grid; Bergot would be
/// `(p-2)(p-1)(2p-3)/6` — the hp builder is Fuentes like the fixed-order
/// default, D347)
fn n_volume_dofs_3d(et: ElementType, p: u8) -> usize {
    let p = p as usize;
    if is_tet_row(et) {
        if p >= 4 { (p - 1) * (p - 2) * (p - 3) / 6 } else { 0 }
    } else if is_prism_row(et) {
        if p >= 3 { (p - 1) * (p - 1) * (p - 2) / 2 } else { 0 }
    } else if is_pyramid_row(et) {
        if p >= 2 { (p - 1).pow(3) } else { 0 }
    } else if is_hex_row(et) {
        if p >= 2 { (p - 1).pow(3) } else { 0 }
    } else {
        0
    }
}

/// Bubble (interior) DOFs of a 2D element.
fn n_bubble_dofs_2d(et: ElementType, p: u8) -> usize {
    n_face_dofs_2d(et, p)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Entity variant computation
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-edge state: for each edge, the distinct orders of its adjacent
/// elements (one DOF set per order) mapped to the edge's 1-D basis flag
/// (`true` = closed Gauss-Lobatto, `false` = equispaced) for that variant.
///
/// D827-1: the flag is tracked per (edge, ORDER) VARIANT — the
/// first-encountering element of that order on the edge donates its family's
/// basis — not once per edge and not from element 0.  On a mixed-family edge
/// (tet p3 × pyramid p2) the p3 variant's dofs sit where the tet's
/// `TetPk` equispaced basis interpolates them and the p2 variant's at the
/// pyramid's GLL midpoint.  A variant shared by two families at the SAME
/// order (tet p3 × pyramid p3) keeps the first-encountering family's basis:
/// MFEM stores one variant per order and never per family (probe
/// `tmp/d86a/out_p33_gl.txt`: variant orders "3", single dof set, no
/// constraints), so one cell basis has to win — same limitation as the
/// documented TetPk-basis divergence (d176).
type EdgeVariantSets = HashMap<EdgeKey, BTreeMap<u8, bool>>;

/// Per-face state: distinct orders of adjacent elements + face node count.
/// Per-face state: for each face, the distinct orders of its adjacent
/// elements mapped to the face's basis flag (`true` = closed Gauss-Lobatto:
/// `H1TriPk` barycentrics on tri faces, GLL tensor on quad faces; `false` =
/// equispaced `TetPk` tri faces) plus the face's corner count (3 or 4).
///
/// D827-1: like the edge flags, the tri-face flag is tracked per
/// (face, ORDER) VARIANT — the first-encountering element of that order on
/// the face donates its family's basis — so a mixed tet × pyramid face
/// places the tet variant's dofs on the equispaced barycentric grid and the
/// pyramid variant's on the `H1TriPk` GLL grid.  Quad faces only occur in
/// GLL families (hex/prism/pyramid), so their flag is always `true`.
type FaceVariantSets = HashMap<FaceKey, (BTreeMap<u8, bool>, usize)>;

/// Collect the order variants of every edge (the set of orders of adjacent
/// elements; each order gets its own DOF set) with the 1-D basis flag of
/// each (edge, order) variant — the first-encountering element of that
/// order donates its family's basis (D827-1).  Also consumed by
/// [`detect_p_constraints`] so the constraint node positions match the
/// builder's DOF coordinates variant for variant.
fn collect_edge_variants<M: MeshTopology>(mesh: &M, elem_orders: &[u8]) -> EdgeVariantSets {
    let mut sets: EdgeVariantSets = HashMap::new();
    for e in 0..mesh.n_elements() as u32 {
        let p = elem_orders[e as usize];
        let et = mesh.element_type(e);
        let ns = elem_corners(et, mesh.element_nodes(e), mesh.geom_order() > 1);
        let gll = elem_uses_gll(et);
        for (a, b) in elem_local_edges(et, &ns) {
            sets.entry(EdgeKey::new(a, b))
                .or_default()
                .entry(p)
                .or_insert(gll);
        }
    }
    sets
}

/// 2D NC hp variant-order propagation (MFEM `CalcEdgeFaceVarOrders`,
/// fespace.cpp): a master edge must hold a variant at the minimum order of
/// its slave edges, and every slave edge adopts the master's post-update
/// minimum order, iterated to fixpoint.  Shared by the builder and
/// [`detect_p_constraints`] so both see the SAME variant set per edge
/// (D827-1: the detector's per-variant basis lookup must cover the
/// propagated variants too).
fn propagate_nc_2d_edge_variants<M: MeshTopology>(mesh: &M, edge_sets: &mut EdgeVariantSets) {
    let nc = detect_nc_geometry_2d(mesh);
    let mut masters: Vec<EdgeKey> = nc.masters.iter().map(|m| m.key).collect();
    masters.sort();
    let mut slaves_of: HashMap<EdgeKey, Vec<EdgeKey>> = HashMap::new();
    for s in &nc.slaves {
        slaves_of.entry(s.master).or_default().push(s.key);
    }
    loop {
        let mut changed = false;
        for mkey in &masters {
            let Some(slaves) = slaves_of.get(mkey) else { continue };
            // min order over all slave edges of this master
            let min_slaves = slaves.iter()
                .map(|k| edge_sets[k].keys().next().copied().unwrap())
                .min()
                .expect("master with no slaves");
            let min_master = edge_sets[mkey].keys().next().copied().unwrap();
            if min_slaves < min_master {
                // The inserted low variant inherits the edge's existing
                // variant basis (2D tri/quad edges are all GLL, so this
                // only keeps the map well-formed).
                let inherit = *edge_sets[mkey].values().next().unwrap();
                edge_sets.get_mut(mkey).unwrap().entry(min_slaves).or_insert(inherit);
                changed = true;
            }
            // apply the master's post-update minimum order to all slave
            // edges (MFEM's unconditional `|=` of `min_mask`)
            let min_master = edge_sets[mkey].keys().next().copied().unwrap();
            for sk in slaves {
                let min_slave = edge_sets[sk].keys().next().copied().unwrap();
                if min_master < min_slave {
                    let inherit = *edge_sets[sk].values().next().unwrap();
                    edge_sets.get_mut(sk).unwrap().entry(min_master).or_insert(inherit);
                    changed = true;
                }
            }
        }
        if !changed { break; }
    }
}

/// Collect the order variants of every 3D face.
fn collect_face_variants<M: MeshTopology>(
    mesh: &M,
    elem_orders: &[u8],
) -> FaceVariantSets {
    let mut sets: FaceVariantSets = HashMap::new();
    for e in 0..mesh.n_elements() as u32 {
        let p = elem_orders[e as usize];
        let et = mesh.element_type(e);
        let ns = elem_corners(et, mesh.element_nodes(e), mesh.geom_order() > 1);
        let gll = elem_uses_gll(et);
        if is_hex_row(et) {
            for face4 in hex_quad_faces(&ns) {
                let face4 = canon_quad_face(face4);
                let key = FaceKey::new(face4[0], face4[1], face4[2]);
                let entry = sets.entry(key).or_insert_with(|| (BTreeMap::new(), 4));
                entry.0.entry(p).or_insert(gll);
            }
        } else if is_pyramid_row(et) {
            for face4 in [pyramid_base_face(&ns)] {
                let face4 = canon_quad_face(face4);
                let key = FaceKey::new(face4[0], face4[1], face4[2]);
                let entry = sets.entry(key).or_insert_with(|| (BTreeMap::new(), 4));
                entry.0.entry(p).or_insert(gll);
            }
            for (a, b, c) in pyramid_tri_faces(&ns) {
                let key = FaceKey::new(a, b, c);
                let entry = sets.entry(key).or_insert_with(|| (BTreeMap::new(), 3));
                entry.0.entry(p).or_insert(gll);
            }
        } else if is_prism_row(et) {
            for (a, b, c) in prism_tri_faces(&ns) {
                let key = FaceKey::new(a, b, c);
                let entry = sets.entry(key).or_insert_with(|| (BTreeMap::new(), 3));
                entry.0.entry(p).or_insert(gll);
            }
            for face4 in prism_quad_faces(&ns) {
                let face4 = canon_quad_face(face4);
                let key = FaceKey::new(face4[0], face4[1], face4[2]);
                let entry = sets.entry(key).or_insert_with(|| (BTreeMap::new(), 4));
                entry.0.entry(p).or_insert(gll);
            }
        } else if is_tet_row(et) {
            for (a, b, c) in tet_faces(&ns) {
                let key = FaceKey::new(a, b, c);
                let entry = sets.entry(key).or_insert_with(|| (BTreeMap::new(), 3));
                entry.0.entry(p).or_insert(gll);
            }
        } else {
            panic!("collect_face_variants: unsupported element geometry {et:?}");
        }
    }
    sets
}

// ═══════════════════════════════════════════════════════════════════════════════
// Variable-order DOF manager (MFEM variant scheme)
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a variable-order DOF manager from per-element polynomial orders.
///
/// Every edge (and 3D face) holds one DOF set per distinct order of its
/// adjacent elements (MFEM `var_edge_dofs`/`var_face_dofs`); an element
/// references the variant matching its own order.  Use
/// [`detect_p_constraints`] to obtain the mixed-order (and, in 2D, hanging
/// edge) constraints.
///
/// Supported element geometries: 2D tri/quad, 3D tet/prism/hex/pyramid
/// (prisms in MFEM `H1_WedgeElement`'s entity order — the layout MFEM's
/// variable-order `FiniteElementSpace::GetElementDofs` produces, probe
/// `tmp/d174_prism_p_probe.cpp`; pyramids in MFEM's default
/// `H1_FuentesPyramidElement` entity order — edges in
/// `Geometry::Constants<PYRAMID>::Edges` order/direction, base quad face
/// first then 4 tri faces in `Faces` order, probe
/// `tmp/d85c/pyr_hp_probe.cpp`).  The per-entity DOF lists use canonical
/// (first-encountering element) face orientation, MFEM's space-level
/// `var_face_dofs` convention: a consumer pairing element rows with a
/// reference element slot-for-slot must apply the face orientation
/// (`TriDofOrd`/`QuadDofOrd`) for elements whose local face order differs
/// from the canonical one — the same convention as the tet/hex paths here.
///
/// # Panics
/// Panics if `elem_orders.len() != mesh.n_elements()`, any order is 0, or the
/// mesh type is unsupported.
pub fn build_variable_order_dof_manager<M: MeshTopology>(
    mesh: &M,
    elem_orders: &[u8],
) -> DofManager {
    let n_elems = mesh.n_elements();
    assert_eq!(elem_orders.len(), n_elems,
        "elem_orders length {} != n_elements {}", elem_orders.len(), n_elems);
    let dim = mesh.dim() as usize;
    let n_nodes = mesh.n_nodes();
    assert!(elem_orders.iter().all(|&p| p >= 1), "element orders must be >= 1");
    let p_max = *elem_orders.iter().max().unwrap_or(&1);

    // 1. Per-entity variant order sets.
    let mut edge_sets = collect_edge_variants(mesh, elem_orders);
    let face_sets = if dim == 3 { collect_face_variants(mesh, elem_orders) } else { FaceVariantSets::new() };

    // 1b. 2D NC hp propagation (MFEM `CalcEdgeFaceVarOrders`, fespace.cpp):
    // a master edge must hold a variant at the minimum order of its slave
    // edges (so the master interpolation the slaves alias to has the lowest
    // degree on the interface), and every slave edge adopts the master's
    // minimum order.  Iterate until fixpoint (master/slave hierarchies
    // propagate upwards).
    //
    // MFEM applies the master's minimum **after** the master itself absorbed
    // the slaves' minimum (`min_mask = 1 << MinOrder(edge_orders[master])` is
    // read post-update); using the pre-update minimum here lets a mid-chain
    // slave inherit an order the master no longer holds, allocating variant
    // DOFs MFEM never creates (D136: 4 spurious order-2 variants at
    // `hpref -n 100`).
    if dim == 2 {
        propagate_nc_2d_edge_variants(mesh, &mut edge_sets);
    }

    // 2. Assign global DOF ids: vertices, then edge variants (ascending order
    //    within an edge), then face variants (3D).
    let mut next_dof = n_nodes as DofId;

    // edge key → [(order, dofs)] ascending in order.
    let mut edge_variants: HashMap<EdgeKey, Vec<(u8, Vec<DofId>)>> = HashMap::new();
    // MFEM numbers the edge-variant DOF blocks by **edge id**, where edge ids
    // come from the flat Mesh's DSTable enumeration (`Mesh::GetEdgeVertexTable`
    // → `GetVertexToVertexTable`): scan the elements in order, each element's
    // edges in local order, canonicalized (min, max) — the first encounter of a
    // vertex pair is its edge id (`mesh/mesh.cpp:8526`, `general/table.hpp`
    // `DSTable::Push`).  This is NOT lexicographic by node pair; matching it is
    // what keeps the global DOF numbering (and hence the GS/PCG iteration
    // path) aligned with MFEM's variable-order spaces (D136).
    let mut edge_list: Vec<EdgeKey> = Vec::with_capacity(edge_sets.len());
    let mut edge_seen: HashSet<EdgeKey> = HashSet::new();
    for e in 0..n_elems as u32 {
        let et = mesh.element_type(e);
        let ns = elem_corners(et, mesh.element_nodes(e), mesh.geom_order() > 1);
        for (a, b) in elem_local_edges(et, &ns) {
            let key = EdgeKey::new(a, b);
            if edge_seen.insert(key) {
                edge_list.push(key);
            }
        }
    }
    debug_assert_eq!(edge_list.len(), edge_sets.len(), "edge scan must find every variant edge");
    for key in &edge_list {
        let orders = edge_sets[key].keys().copied().collect::<Vec<u8>>();
        let mut variants: Vec<(u8, Vec<DofId>)> = Vec::with_capacity(orders.len());
        for &p in &orders {
            let n = (p - 1) as usize;
            let dofs: Vec<DofId> = (0..n).map(|_| { let d = next_dof; next_dof += 1; d }).collect();
            variants.push((p, dofs));
        }
        edge_variants.insert(*key, variants);
    }

    let mut face_variants: HashMap<FaceKey, Vec<(u8, Vec<DofId>)>> = HashMap::new();
    if dim == 3 {
        let mut face_list: Vec<FaceKey> = face_sets.keys().copied().collect();
        face_list.sort();
        for key in face_list {
            let (orders, nn) = &face_sets[&key];
            let mut variants: Vec<(u8, Vec<DofId>)> = Vec::with_capacity(orders.len());
            for &p in orders.keys() {
                let n = n_face_dofs_3d(*nn, p);
                if n == 0 { continue; }
                let dofs: Vec<DofId> = (0..n).map(|_| { let d = next_dof; next_dof += 1; d }).collect();
                variants.push((p, dofs));
            }
            face_variants.insert(key, variants);
        }
    }

    // 3. Element DOF lists (vertex dofs, edge/face variants of the element's
    //    own order, then bubble dofs) with elem_dof_offsets.
    let mut dofs_flat: Vec<DofId> = Vec::new();
    let mut elem_dof_offsets = Vec::with_capacity(n_elems + 1);
    elem_dof_offsets.push(0);

    for e in 0..n_elems as u32 {
        let p_e = elem_orders[e as usize];
        let et = mesh.element_type(e);
        assert!(
            (dim == 2 && (is_tri_row(et) || is_quad_row(et)))
                || (dim == 3 && (is_tet_row(et) || is_prism_row(et) || is_hex_row(et)
                    || is_pyramid_row(et))),
            "build_variable_order_dof_manager: unsupported element geometry \
             (dim {dim}, {et:?}; quadratic Tri6/Tet10/Hex20/Hex27/Prism15/Prism18/\
             Pyramid13 rows are supported since D824)"
        );
        // D824: the row's CORNER nodes are the vertex dofs — a quadratic row
        // (Quad8/Quad9/Tet10/Hex20/Hex27/Prism15/Prism18) must not donate its
        // mid-node geometry dofs (for Prism18 the corners are the non-prefix
        // slots [0, 1, 2, 12, 13, 14] of the layer-major row).
        let ns = elem_corners(et, mesh.element_nodes(e), mesh.geom_order() > 1);
        for &n in ns.iter() {
            dofs_flat.push(n);
        }

        // Edge DOFs: the element's own order variant, oriented from the
        // element's local first vertex to its second vertex.
        for (a, b) in elem_local_edges(et, &ns) {
            let key = EdgeKey::new(a, b);
            let variants = &edge_variants[&key];
            let dofs = &variants.iter().find(|(p, _)| *p == p_e)
                .unwrap_or_else(|| panic!("edge variant p{p_e} missing for edge {key:?}")).1;
            if a == key.0 {
                dofs_flat.extend_from_slice(dofs);
            } else {
                dofs_flat.extend(dofs.iter().rev());
            }
        }

        // Face DOFs (3D): the element's own order variant.
        if dim == 3 {
            if is_hex_row(et) {
                for face4 in hex_quad_faces(&ns) {
                    let face4 = canon_quad_face(face4);
                    let key = FaceKey::new(face4[0], face4[1], face4[2]);
                    if let Some((_, dofs)) = face_variants[&key].iter().find(|(p, _)| *p == p_e) {
                        dofs_flat.extend_from_slice(dofs);
                    }
                }
            } else if is_pyramid_row(et) {
                // MFEM `Geometry::Constants<PYRAMID>::Faces` order: the base
                // quad face (face 0) first, then the four triangular side
                // faces (probe `tmp/d85c/out_p311.txt`: the p3 row runs
                // `... | 4 base-face dofs | 4 tri-face dofs | 8 interior`).
                for face4 in [pyramid_base_face(&ns)] {
                    let face4 = canon_quad_face(face4);
                    let key = FaceKey::new(face4[0], face4[1], face4[2]);
                    if let Some((_, dofs)) = face_variants[&key].iter().find(|(p, _)| *p == p_e) {
                        dofs_flat.extend_from_slice(dofs);
                    }
                }
                for (a, b, c) in pyramid_tri_faces(&ns) {
                    let key = FaceKey::new(a, b, c);
                    if let Some((_, dofs)) = face_variants[&key].iter().find(|(p, _)| *p == p_e) {
                        dofs_flat.extend_from_slice(dofs);
                    }
                }
            } else if is_prism_row(et) {
                // MFEM `H1_WedgeElement` face order: bottom tri, top tri,
                // then the three side quads (each block in the canonical
                // face-variant order — see the fn doc for the orientation
                // convention).
                for (a, b, c) in prism_tri_faces(&ns) {
                    let key = FaceKey::new(a, b, c);
                    if let Some((_, dofs)) = face_variants[&key].iter().find(|(p, _)| *p == p_e) {
                        dofs_flat.extend_from_slice(dofs);
                    }
                }
                for face4 in prism_quad_faces(&ns) {
                    let face4 = canon_quad_face(face4);
                    let key = FaceKey::new(face4[0], face4[1], face4[2]);
                    if let Some((_, dofs)) = face_variants[&key].iter().find(|(p, _)| *p == p_e) {
                        dofs_flat.extend_from_slice(dofs);
                    }
                }
            } else {
                for (a, b, c) in tet_faces(&ns) {
                    let key = FaceKey::new(a, b, c);
                    if let Some((_, dofs)) = face_variants[&key].iter().find(|(p, _)| *p == p_e) {
                        dofs_flat.extend_from_slice(dofs);
                    }
                }
            }
        }

        // Bubble/Volume DOFs (element-private).
        let n_bubble = if dim == 2 {
            n_bubble_dofs_2d(et, p_e)
        } else {
            n_volume_dofs_3d(et, p_e)
        };
        for _ in 0..n_bubble {
            dofs_flat.push(next_dof);
            next_dof += 1;
        }

        elem_dof_offsets.push(dofs_flat.len());
    }

    let n_dofs = next_dof as usize;

    // 4. DOF coordinates.
    let mut dof_coords = vec![0.0_f64; n_dofs * dim];

    // Vertex coordinates.
    for n in 0..n_nodes as u32 {
        let c = mesh.node_coords(n);
        let base = n as usize * dim;
        dof_coords[base..base + dim].copy_from_slice(c);
    }

    // Edge DOFs: at the variant's own basis node positions along the edge
    // (D827-1: the basis flag is per (edge, order) variant, so on a
    // mixed-family edge each variant sits at its own family's node set).
    for key in &edge_list {
        let gll_of = &edge_sets[key];
        let ca = mesh.node_coords(key.0);
        let cb = mesh.node_coords(key.1);
        for (p, dofs) in &edge_variants[key] {
            let pos = interior_positions_1d(*p, gll_of[p]);
            for (k, &dof_id) in dofs.iter().enumerate() {
                let t = pos[k];
                let base = dof_id as usize * dim;
                for d in 0..dim {
                    dof_coords[base + d] = (1.0 - t) * ca[d] + t * cb[d];
                }
            }
        }
    }

    // 3D face DOF coordinates: bilinear (hex/prism quad faces), Fuentes
    // j-reversed bilinear (pyramid base quad faces), equispaced barycentric
    // (tet tri faces) or GLL barycentric (prism/pyramid tri faces)
    // interpolation at the variant's node positions.
    if dim == 3 {
        // Face-local node lists per face key (from the first element seen —
        // the canonical face ORIENTATION only).  The 1-D BASIS is per
        // (face, order) variant (D827-1), read off `face_sets`: quad faces —
        // always GLL tensor, `true` in the pyr-arm flag selecting the
        // Fuentes j-reversed layout `(cp[i], cp[p−j])` for pyramid base
        // faces; tri faces — GLL `H1TriPk` barycentrics for the
        // prism/pyramid variants, equispaced `TetPk` barycentrics for the
        // tet variants.
        let mut face_nodes4: HashMap<FaceKey, ([NodeId; 4], bool)> = HashMap::new();
        let mut face_nodes3: HashMap<FaceKey, [NodeId; 3]> = HashMap::new();
        for e in 0..n_elems as u32 {
            let et = mesh.element_type(e);
            let ns = elem_corners(et, mesh.element_nodes(e), mesh.geom_order() > 1);
            if is_hex_row(et) {
                for face4 in hex_quad_faces(&ns) {
                    let face4 = canon_quad_face(face4);
                    face_nodes4.entry(FaceKey::new(face4[0], face4[1], face4[2]))
                        .or_insert((face4, false));
                }
            } else if is_pyramid_row(et) {
                let face4 = canon_quad_face(pyramid_base_face(&ns));
                face_nodes4.entry(FaceKey::new(face4[0], face4[1], face4[2]))
                    .or_insert((face4, true));
                for (a, b, c) in pyramid_tri_faces(&ns) {
                    face_nodes3.entry(FaceKey::new(a, b, c)).or_insert([a, b, c]);
                }
            } else if is_prism_row(et) {
                for (a, b, c) in prism_tri_faces(&ns) {
                    face_nodes3.entry(FaceKey::new(a, b, c)).or_insert([a, b, c]);
                }
                for face4 in prism_quad_faces(&ns) {
                    let face4 = canon_quad_face(face4);
                    face_nodes4.entry(FaceKey::new(face4[0], face4[1], face4[2]))
                        .or_insert((face4, false));
                }
            } else {
                for (a, b, c) in tet_faces(&ns) {
                    face_nodes3.entry(FaceKey::new(a, b, c)).or_insert([a, b, c]);
                }
            }
        }
        for (key, variants) in &face_variants {
            for &(p, ref dofs) in variants {
                if dofs.is_empty() { continue; }
                if let Some(&(n4, pyr_face)) = face_nodes4.get(key) {
                    // Quad face: GLL tensor (iy outer, ix inner), bilinear
                    // map — with the y axis j-reversed for pyramid base
                    // faces (Fuentes `cp[p−j]`, `fe_h1.cpp:1112-1121`).
                    let pos = interior_positions_1d(p, true);
                    let c: Vec<[f64; 3]> = (0..4).map(|i| {
                        let c = mesh.node_coords(n4[i]); [c[0], c[1], c[2]]
                    }).collect();
                    let n1 = pos.len();
                    for (j, &dof) in dofs.iter().enumerate() {
                        let (ix, iy) = (j % n1, j / n1);
                        let s = if pyr_face { pos[n1 - 1 - iy] } else { pos[iy] };
                        let (r, t) = (pos[ix], s);
                        let base = dof as usize * dim;
                        for d in 0..dim {
                            dof_coords[base + d] = (1.0 - r) * (1.0 - t) * c[0][d]
                                + r * (1.0 - t) * c[1][d]
                                + r * t * c[2][d]
                                + (1.0 - r) * t * c[3][d];
                        }
                    }
                } else if let Some(&n3) = face_nodes3.get(key) {
                    // Tri face: barycentric interpolation, dof list in the
                    // `H1TriPk`/TriPk running (j outer, i inner) order, on
                    // the VARIANT's own basis (D827-1).
                    let pq = p as usize;
                    let gll_face = face_sets[key].0[&p];
                    let c: Vec<[f64; 3]> = (0..3).map(|i| {
                        let c = mesh.node_coords(n3[i]); [c[0], c[1], c[2]]
                    }).collect();
                    // Reference (λ1, λ2) of the running-order interior nodes.
                    let tri_pos: Vec<[f64; 2]> = if gll_face {
                        // GLL: `H1TriPk`'s interior block, same running order
                        // (prism side faces and Fuentes pyramid side faces —
                        // `fe_h1.cpp:1123-1144` normalises the same
                        // `cp[i]/w` barycentrics as `H1_TriangleElement`).
                        H1TriPk::new(pq).dof_coords()[3 * pq..3 * pq + dofs.len()]
                            .iter().map(|rc| [rc[0], rc[1]]).collect()
                    } else {
                        // Equispaced (TetPk convention).
                        (1..=pq.saturating_sub(2))
                            .flat_map(|j| (1..=pq - 1 - j).map(move |i| [i as f64 / pq as f64, j as f64 / pq as f64]))
                            .collect()
                    };
                    for (k, &dof) in dofs.iter().enumerate() {
                        let (r, s) = (tri_pos[k][0], tri_pos[k][1]);
                        let lam0 = 1.0 - r - s;
                        let base = dof as usize * dim;
                        for d in 0..3 {
                            dof_coords[base + d] = lam0 * c[0][d] + r * c[1][d] + s * c[2][d];
                        }
                    }
                }
            }
        }
    }

    // Bubble/Volume DOF coordinates (element-private: walk the tail of each
    // element's DOF list).
    // `H1TriPk` interior reference (λ1, λ2) per order (prism interiors).
    let mut prism_tri_int: HashMap<u8, Vec<[f64; 2]>> = HashMap::new();
    for e in 0..n_elems as u32 {        let p_e = elem_orders[e as usize];
        let et = mesh.element_type(e);
        let ns = elem_corners(et, mesh.element_nodes(e), mesh.geom_order() > 1);
        let n_vol = if dim == 2 {
            n_bubble_dofs_2d(et, p_e)
        } else {
            n_volume_dofs_3d(et, p_e)
        };
        if n_vol == 0 { continue; }
        let bubble_dofs = &dofs_flat[elem_dof_offsets[e as usize + 1] - n_vol
            ..elem_dof_offsets[e as usize + 1]];
        if dim == 2 && is_tri_row(et) {
            // Tri bubble: TriPk factory (equispaced) reference coordinates.
            let factory = fem_element::lagrange::factory::ref_elem(
                fem_element::lagrange::factory::ElemType::Tri, p_e);
            let rc = factory.dof_coords();
            let start = rc.len() - n_vol;
            for (k, &dof) in bubble_dofs.iter().enumerate() {
                let base = dof as usize * dim;
                let rck = &rc[start + k];
                let lam0 = 1.0 - rck[0] - rck[1];
                for d in 0..2 {
                    dof_coords[base + d] = lam0 * mesh.node_coords(ns[0])[d]
                        + rck[0] * mesh.node_coords(ns[1])[d]
                        + rck[1] * mesh.node_coords(ns[2])[d];
                }
            }
        } else if dim == 2 && is_quad_row(et) {
            // Quad interior: GLL tensor product (QuadQk layout: iy outer,
            // ix inner).
            let pos = interior_positions_1d(p_e, true);
            let mut idx = 0usize;
            for &s in &pos {
                for &r in &pos {
                    let base = bubble_dofs[idx] as usize * dim;
                    for d in 0..2 {
                        dof_coords[base + d] = (1.0 - r) * (1.0 - s) * mesh.node_coords(ns[0])[d]
                            + r * (1.0 - s) * mesh.node_coords(ns[1])[d]
                            + r * s * mesh.node_coords(ns[2])[d]
                            + (1.0 - r) * s * mesh.node_coords(ns[3])[d];
                    }
                    idx += 1;
                }
            }
        } else if dim == 3 && is_tet_row(et) {
            // Tet volume: TetPk factory (equispaced) reference coordinates.
            let factory = fem_element::lagrange::factory::ref_elem(
                fem_element::lagrange::factory::ElemType::Tet, p_e);
            let rc = factory.dof_coords();
            let start = rc.len() - n_vol;
            for (k, &dof) in bubble_dofs.iter().enumerate() {
                let base = dof as usize * dim;
                let rck = &rc[start + k];
                let lam0 = 1.0 - rck[0] - rck[1] - rck[2];
                for d in 0..3 {
                    dof_coords[base + d] = lam0 * mesh.node_coords(ns[0])[d]
                        + rck[0] * mesh.node_coords(ns[1])[d]
                        + rck[1] * mesh.node_coords(ns[2])[d]
                        + rck[2] * mesh.node_coords(ns[3])[d];
                }
            }
        } else if dim == 3 && is_prism_row(et) {
            // Prism interior: MFEM `H1_WedgeElement` interior order — layer
            // kk = 1..p-1 outer (Gauss-Lobatto ξ), triangle interior in the
            // `H1TriPk` running (j outer, i inner) order within a layer.
            let pq = p_e as usize;
            let tri_int = prism_tri_int.entry(p_e).or_insert_with(|| {
                H1TriPk::new(pq).dof_coords()[3 * pq..3 * pq + n_vol / (pq - 1)]
                    .iter().map(|rc| [rc[0], rc[1]]).collect()
            });
            let (g, _) = fem_element::quadrature::gauss_lobatto_arbitrary(pq + 1);
            let c: [[f64; 3]; 6] = std::array::from_fn(|k| {
                let x = mesh.node_coords(ns[k]); [x[0], x[1], x[2]]
            });
            let mut idx = 0usize;
            for &xi in &g[1..pq] {
                let t = 0.5 * (xi + 1.0); // [-1,1] → [0,1]
                for rc in tri_int.iter() {
                    let (r, s) = (rc[0], rc[1]);
                    let lam0 = 1.0 - r - s;
                    let base = bubble_dofs[idx] as usize * dim;
                    for d in 0..3 {
                        let bottom = lam0 * c[0][d] + r * c[1][d] + s * c[2][d];
                        let top = lam0 * c[3][d] + r * c[4][d] + s * c[5][d];
                        dof_coords[base + d] = (1.0 - t) * bottom + t * top;
                    }
                    idx += 1;
                }
            }
        } else if dim == 3 && is_hex_row(et) {
            // Hex volume: GLL tensor product.
            let pos = interior_positions_1d(p_e, true);
            let mut idx = 0usize;
            for &t in &pos {
                for &s in &pos {
                    for &r in &pos {
                        let base = bubble_dofs[idx] as usize * dim;
                        let c: Vec<[f64; 3]> = (0..8).map(|i| {
                            let c = mesh.node_coords(ns[i]);
                            [c[0], c[1], c[2]]
                        }).collect();
                        for d in 0..3 {
                            // Trilinear interpolation.
                            dof_coords[base + d] =
                                (1.0 - r) * (1.0 - s) * (1.0 - t) * c[0][d]
                              + r * (1.0 - s) * (1.0 - t) * c[1][d]
                              + r * s * (1.0 - t) * c[2][d]
                              + (1.0 - r) * s * (1.0 - t) * c[3][d]
                              + (1.0 - r) * (1.0 - s) * t * c[4][d]
                              + r * (1.0 - s) * t * c[5][d]
                              + r * s * t * c[6][d]
                              + (1.0 - r) * s * t * c[7][d];
                        }
                        idx += 1;
                    }
                }
            }
        } else if dim == 3 && is_pyramid_row(et) {
            // Pyramid interior: MFEM `H1_FuentesPyramidElement` bubble nodes
            // (`fe_h1.cpp:1146-1157`) — GLL layers `cp[k]` outer, then the
            // in-plane GLL tensor `(cp[i], cp[j])` with `i` fastest, all
            // homothetic toward the apex:
            // `P = (1−t)·bilinear_base(cp[i], cp[j]) + t·apex`.
            let pq = p_e as usize;
            let (g, _) = fem_element::quadrature::gauss_lobatto_arbitrary(pq + 1);
            let c: Vec<[f64; 3]> = (0..5).map(|k| {
                let x = mesh.node_coords(ns[k]);
                [x[0], x[1], x[2]]
            }).collect();
            let mut idx = 0usize;
            for &gk in &g[1..pq] {
                let t = 0.5 * (gk + 1.0); // cp[k] on [0, 1]
                for &gj in &g[1..pq] {
                    let v = 0.5 * (gj + 1.0);
                    for &gi in &g[1..pq] {
                        let u = 0.5 * (gi + 1.0);
                        let base = bubble_dofs[idx] as usize * dim;
                        for d in 0..3 {
                            let b = (1.0 - u) * (1.0 - v) * c[0][d]
                                + u * (1.0 - v) * c[1][d]
                                + u * v * c[2][d]
                                + (1.0 - u) * v * c[3][d];
                            dof_coords[base + d] = (1.0 - t) * b + t * c[4][d];
                        }
                        idx += 1;
                    }
                }
            }
        }
    }

    // 5. Flat maps consumed by boundary_dofs and legacy code paths: all
    //    variant DOFs concatenated (lowest variant first).
    let mut edge_pk_map: HashMap<EdgeKey, Vec<DofId>> = HashMap::new();
    for (key, variants) in &edge_variants {
        let mut all = Vec::new();
        for (_, dofs) in variants { all.extend_from_slice(dofs); }
        edge_pk_map.insert(*key, all);
    }
    let mut face_pk_map: HashMap<FaceKey, Vec<DofId>> = HashMap::new();
    for (key, variants) in &face_variants {
        let mut all = Vec::new();
        for (_, dofs) in variants { all.extend_from_slice(dofs); }
        face_pk_map.insert(*key, all);
    }

    DofManager {
        order: p_max,
        n_dofs,
        dofs_flat,
        dofs_per_elem: 0, // variable: use elem_dof_offsets
        elem_dof_offsets: Some(elem_dof_offsets),
        dof_coords,
        dim,
        n_vertex_dofs: n_nodes,
        edge_dof_map: HashMap::new(),
        edge_dof2_map: HashMap::new(),
        phys_to_vertex_dof: HashMap::new(),
        edge_pk_map,
        face_pk_map,
        quad_face_pk_map: HashMap::new(),
        bubble_dof_start: n_dofs,
        n_volume_dofs: 0,
        elem_orders: Some(elem_orders.to_vec()),
        edge_variants,
        face_variants,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Non-conforming (hanging) 2D edge geometry
// ═══════════════════════════════════════════════════════════════════════════════

/// A master (coarse) element edge with hanging nodes strictly inside it.
#[derive(Debug, Clone)]
pub struct NcMasterEdge {
    /// Canonical edge key (min node first).
    pub key: EdgeKey,
}

/// A slave element edge strictly contained in a master edge.
#[derive(Debug, Clone)]
pub struct NcSlaveEdge {
    /// Canonical key of the slave edge.
    pub key: EdgeKey,
    /// Canonical key of the containing master edge.
    pub master: EdgeKey,
    /// Parameter of `key.0` on the master edge (from `master.0`).
    pub t_a: f64,
    /// Parameter of `key.1` on the master edge.
    pub t_b: f64,
}

/// Geometric non-conformity of a 2D mesh: hanging nodes, master and slave
/// edges, detected from element connectivity and coordinates (flat-mesh
/// equivalent of MFEM's `NCList` edge masters/slaves).
#[derive(Debug, Clone, Default)]
pub struct NcGeometry2D {
    /// Master edges (each is the longest element edge strictly containing at
    /// least one hanging node).
    pub masters: Vec<NcMasterEdge>,
    /// Slave edges fully contained in a master edge.
    pub slaves: Vec<NcSlaveEdge>,
    /// `(node, master_key, t)` for every hanging node (strictly inside a
    /// master edge at parameter `t` measured from `master.0`).
    pub hanging_nodes: Vec<(NodeId, EdgeKey, f64)>,
}

/// Projection parameter of point `p` on segment `(a, b)`, or `None` if `p` is
/// not (within tolerance) on the segment.  `closed` accepts `t ∈ {0, 1}`.
fn proj_on_segment(p: &[f64], a: &[f64], b: &[f64], closed: bool) -> Option<f64> {
    let abx = b[0] - a[0];
    let aby = b[1] - a[1];
    let apx = p[0] - a[0];
    let apy = p[1] - a[1];
    let len2 = abx * abx + aby * aby;
    if len2 < 1e-30 { return None; }
    let t = (apx * abx + apy * aby) / len2;
    let cross = apx * aby - apy * abx;
    if cross * cross > 1e-22 * len2 { return None; }
    let eps = 1e-9;
    if closed {
        if t >= -eps && t <= 1.0 + eps { Some(t.clamp(0.0, 1.0)) } else { None }
    } else if t > eps && t < 1.0 - eps {
        Some(t)
    } else {
        None
    }
}

/// Detect the 2D non-conforming edge geometry (masters / slaves / hanging
/// nodes) of a flat quad mesh.  Deterministic; a node's master is the longest
/// element edge strictly containing it.
pub fn detect_nc_geometry_2d<M: MeshTopology>(mesh: &M) -> NcGeometry2D {
    debug_assert_eq!(mesh.dim(), 2, "detect_nc_geometry_2d: 2D meshes only");
    let mut geom = NcGeometry2D::default();

    // Unique element edges with squared lengths.
    let mut edges: Vec<(EdgeKey, f64)> = Vec::new();
    let mut edge_set: HashSet<EdgeKey> = HashSet::new();
    for e in 0..mesh.n_elements() as u32 {
        let et = mesh.element_type(e);
        let ns = elem_corners(et, mesh.element_nodes(e), mesh.geom_order() > 1);
        for (a, b) in elem_local_edges(et, &ns) {
            let key = EdgeKey::new(a, b);
            if edge_set.insert(key) {
                let ca = mesh.node_coords(key.0);
                let cb = mesh.node_coords(key.1);
                let len2 = (cb[0] - ca[0]).powi(2) + (cb[1] - ca[1]).powi(2);
                edges.push((key, len2));
            }
        }
    }

    // Hanging nodes: row-corner nodes strictly inside some element edge; the
    // master is the longest containing edge.  D824: only row CORNERS can be
    // hanging — a quadratic row's mid-node geometry nodes are not mesh
    // vertices (MFEM's vertex table is the corner set) and must never enter
    // the NC walk.
    let mut is_corner = vec![false; mesh.n_nodes()];
    for e in 0..mesh.n_elements() as u32 {
        let et = mesh.element_type(e);
        for n in elem_corners(et, mesh.element_nodes(e), mesh.geom_order() > 1) {
            is_corner[n as usize] = true;
        }
    }
    let mut master_of: HashMap<NodeId, (EdgeKey, f64)> = HashMap::new();
    for n in 0..mesh.n_nodes() as u32 {
        if !is_corner[n as usize] {
            continue;
        }
        let p = mesh.node_coords(n);
        let mut best: Option<(EdgeKey, f64, f64)> = None; // (key, len2, t)
        for &(key, len2) in &edges {
            if key.0 == n || key.1 == n { continue; }
            let cu = mesh.node_coords(key.0);
            let cv = mesh.node_coords(key.1);
            if let Some(t) = proj_on_segment(p, cu, cv, false) {
                if best.map(|(_, l, _)| len2 > l).unwrap_or(true) {
                    best = Some((key, len2, t));
                }
            }
        }
        if let Some((key, _, t)) = best {
            master_of.insert(n, (key, t));
        }
    }
    if master_of.is_empty() { return geom; }

    // Masters: unique longest containers.
    let mut master_keys: Vec<EdgeKey> = master_of.values().map(|(k, _)| *k).collect();
    master_keys.sort();
    master_keys.dedup();
    for key in &master_keys {
        geom.masters.push(NcMasterEdge { key: *key });
    }

    // Slave edges: element edges whose BOTH endpoints lie on a longer master
    // edge (the longest such master wins).
    for &(key, len2) in &edges {
        let mut best: Option<(EdgeKey, f64, f64, f64)> = None; // (master, mlen2, ta, tb)
        for &mkey in &master_keys {
            let Some(mlen2) = edges.iter().find(|(k, _)| *k == mkey).map(|(_, l)| *l) else {
                continue;
            };
            if mlen2 <= len2 { continue; }
            let cu = mesh.node_coords(mkey.0);
            let cv = mesh.node_coords(mkey.1);
            let ca = mesh.node_coords(key.0);
            let cb = mesh.node_coords(key.1);
            let (Some(ta), Some(tb)) = (
                proj_on_segment(ca, cu, cv, true),
                proj_on_segment(cb, cu, cv, true),
            ) else { continue };
            if best.map(|(_, l, _, _)| mlen2 > l).unwrap_or(true) {
                best = Some((mkey, mlen2, ta, tb));
            }
        }
        if let Some((mkey, _, ta, tb)) = best {
            geom.slaves.push(NcSlaveEdge { key, master: mkey, t_a: ta, t_b: tb });
        }
    }
    geom.slaves.sort_by_key(|s| s.key);

    // Hanging node list (sorted by node id).
    for (n, (key, t)) in &master_of {
        geom.hanging_nodes.push((*n, *key, *t));
    }
    geom.hanging_nodes.sort_by_key(|(n, _, _)| *n);

    geom
}

// ═══════════════════════════════════════════════════════════════════════════════
// Constraint detection
// ═══════════════════════════════════════════════════════════════════════════════

/// Detect p-refinement constraints at mixed-order interfaces.
///
/// Combines (MFEM `BuildConformingInterpolation`):
/// * the variable-order minimum rule on conforming edges/faces holding
///   multiple order variants (higher variants interpolate the lowest one), and
/// * 2D non-conforming hanging-edge constraints (all DOFs of a slave edge and
///   its hanging endpoint vertices interpolate the master edge's lowest
///   variant at the slave DOF's global position).
///
/// Currently supports 2D triangular/quadrilateral meshes (edge constraints +
/// NC hanging edges) and 3D tetrahedral/hexahedral meshes (edge + face
/// min-rule constraints; NC 3D is not supported).
pub fn detect_p_constraints<M: MeshTopology>(
    dm: &DofManager,
    mesh: &M,
    elem_orders: &[u8],
) -> Vec<PRefineConstraint> {
    let mut constraints: Vec<PRefineConstraint> = Vec::new();
    let dim = mesh.dim() as usize;
    assert_eq!(elem_orders.len(), mesh.n_elements(),
        "elem_orders length {} != n_elements {}",
        elem_orders.len(), mesh.n_elements());

    // D827-1: the 1-D basis of an edge variant is a property of the
    // (edge, order) VARIANT — the family that uses it — not of element 0.
    // Recompute the per-variant flags with the builder's own rule so the
    // constraint node positions match the DOF coordinates variant for
    // variant (mixed-family hp edges: tet p3 × pyramid p2 → equispaced
    // slave positions on the tet side, GLL master nodes on the pyramid
    // side).  MFEM's counterpart (`VariableOrderMinimumRule`,
    // fem/fespace.cpp:1094) evaluates `fec->GetFE(Geometry::SEGMENT, p/q)`
    // — the collection-level segment FE for both sides, independent of the
    // adjacent cell families (probe tmp/d86a/: flipping only the collection
    // basis flips every edge row GLL ↔ equispaced); fem-rs's per-family
    // cell bases (TetPk equispaced vs GLL elsewhere) make the variant-level
    // dispatch the faithful translation.
    let mut edge_gll = collect_edge_variants(mesh, elem_orders);
    if dim == 2 {
        // Mirror the builder's 2D NC propagation so the per-variant basis
        // lookup covers the propagated master/slave variants too.
        propagate_nc_2d_edge_variants(mesh, &mut edge_gll);
    }

    // 2D NC geometry (before the min-rule so slave edges can be excluded).
    let nc = if dim == 2 { Some(detect_nc_geometry_2d(mesh)) } else { None };
    let mut slave_keys: HashSet<EdgeKey> = HashSet::new();
    if let Some(ref nc) = nc {
        for s in &nc.slaves {
            slave_keys.insert(s.key);
        }
    }

    // ─── Min-rule: conforming edges with multiple variants ────────────────
    for (key, variants) in &dm.edge_variants {
        if variants.len() <= 1 || slave_keys.contains(key) {
            continue;
        }
        let gll_of = &edge_gll[key];
        let (p0, dofs0) = &variants[0];
        // Master node list: endpoints + variant-0 interior positions (at the
        // master variant's own family basis).
        let mut nodes0 = vec![0.0_f64];
        nodes0.extend_from_slice(&interior_positions_1d(*p0, gll_of[p0]));
        nodes0.push(1.0);
        let mut master_dofs: Vec<DofId> = vec![key.0 as DofId];
        master_dofs.extend_from_slice(dofs0);
        master_dofs.push(key.1 as DofId);

        for (q, dofs_q) in variants.iter().skip(1) {
            // Slave positions at the slave variant's own family basis.
            let pos_q = interior_positions_1d(*q, gll_of[q]);
            for (j, &dof) in dofs_q.iter().enumerate() {
                let weights = lagrange_weights_at(&nodes0, pos_q[j]);
                let parents: Vec<(DofId, f64)> = master_dofs.iter()
                    .zip(weights.iter())
                    .filter(|&(_, &w)| w.abs() > 1e-15)
                    .map(|(&d, &w)| (d, w))
                    .collect();
                constraints.push(PRefineConstraint { constrained: dof, parents });
            }
        }
    }

    // ─── Min-rule: 3D faces with multiple variants ─────────────────────────
    if dim == 3 {
        detect_face_variant_constraints(dm, mesh, &mut constraints);
    }

    // ─── 2D non-conforming hanging edges ───────────────────────────────────
    if let Some(nc) = nc {
        // Master edge variant-0 node lists (cached per master).
        let mut master_cache: HashMap<EdgeKey, (Vec<f64>, Vec<DofId>)> = HashMap::new();
        for m in &nc.masters {
            let key = m.key;
            let variants = dm.edge_variants.get(&key)
                .unwrap_or_else(|| panic!("edge_variants missing for master edge {key:?}"));
            let (p0, dofs0) = &variants[0];
            let gll_of = edge_gll.get(&key)
                .unwrap_or_else(|| panic!("edge basis missing for master edge {key:?}"));
            let mut nodes0 = vec![0.0_f64];
            nodes0.extend_from_slice(&interior_positions_1d(*p0, gll_of[p0]));
            nodes0.push(1.0);
            let mut master_dofs: Vec<DofId> = vec![key.0 as DofId];
            master_dofs.extend_from_slice(dofs0);
            master_dofs.push(key.1 as DofId);
            master_cache.insert(key, (nodes0, master_dofs));
        }

        // Hanging vertices: constrained to the master's lowest variant.
        let mut seen: HashSet<DofId> = HashSet::new();
        for &(node, mkey, t) in &nc.hanging_nodes {
            let (nodes0, master_dofs) = &master_cache[&mkey];
            let weights = lagrange_weights_at(nodes0, t);
            let parents: Vec<(DofId, f64)> = master_dofs.iter()
                .zip(weights.iter())
                .filter(|&(_, &w)| w.abs() > 1e-15)
                .map(|(&d, &w)| (d, w))
                .collect();
            let dof = node as DofId;
            if seen.insert(dof) {
                constraints.push(PRefineConstraint { constrained: dof, parents });
            }
        }

        // Slave edges: ALL variant DOFs constrained to the master's lowest
        // variant at the DOF's global position on the master edge.
        for s in &nc.slaves {
            let (nodes0, master_dofs) = &master_cache[&s.master];
            let variants = dm.edge_variants.get(&s.key)
                .unwrap_or_else(|| panic!("edge_variants missing for slave edge {:?}", s.key));
            let gll_of = edge_gll.get(&s.key)
                .unwrap_or_else(|| panic!("edge basis missing for slave edge {:?}", s.key));
            for (q, dofs_q) in variants {
                let pos_q = interior_positions_1d(*q, gll_of[q]);
                for (j, &dof) in dofs_q.iter().enumerate() {
                    // Local canonical position s_j → global master position.
                    let t = s.t_a + (s.t_b - s.t_a) * pos_q[j];
                    let weights = lagrange_weights_at(nodes0, t);
                    let parents: Vec<(DofId, f64)> = master_dofs.iter()
                        .zip(weights.iter())
                        .filter(|&(_, &w)| w.abs() > 1e-15)
                        .map(|(&d, &w)| (d, w))
                        .collect();
                    constraints.push(PRefineConstraint { constrained: dof, parents });
                }
            }
        }
    }

    // Deduplicate: a constrained DOF must appear exactly once (shared faces
    // are visited from both sides with identical parents/weights).
    constraints.sort_by_key(|c| c.constrained);
    constraints.dedup_by(|a, b| a.constrained == b.constrained);

    constraints
}

/// One quad-face min-rule constraint: dof `dof` (the `j`-th dof of the
/// order-`q` variant of quad face `face4`) interpolates the order-`p0` GLL
/// trace over the face closure — vertices, oriented edge runs, face-interior
/// dofs of the order-`p0` variant `dofs0` (empty when `p0 < 2`, where the
/// grid has no interior nodes).
///
/// `reversed_y` selects the intra-block layout: `false` — the ascending GLL
/// tensor `(iy outer, ix inner)` of hex/prism quad faces; `true` — the
/// Fuentes pyramid base-face layout `(cp[i], cp[p−j])`, j reversed
/// (`fe_h1.cpp:1112-1121`; probe `tmp/d85c/out_p311.txt` pins the p3
/// weights: face dof 0 at `(cp[1], cp[p−1])` carries `(1−x)(1−y) = 0.2` on
/// `v0`, not `0.5236`).
fn quad_face_variant_constraint(
    dm: &DofManager,
    face4: [NodeId; 4],
    master: (u8, &[DofId]),
    q: u8,
    dof: DofId,
    j: usize,
    reversed_y: bool,
    constraints: &mut Vec<PRefineConstraint>,
) {
    let (p0, dofs0) = master;
    let pos_q = interior_positions_1d(q, true);
    let n1 = pos_q.len(); // (q-1)
    // Layout: iy outer, ix inner (HexQk face convention).
    let iy = j / n1;
    let ix = j % n1;
    // Tensor-product weights over the p0 grid:
    // parents = vertices, 4 edges, face-interior dofs of p0.
    let sy = if reversed_y { pos_q[n1 - 1 - iy] } else { pos_q[iy] };
    let (r, s) = (pos_q[ix], sy);
    let wx = lagrange_weights_at(&gll_positions_01(p0), r);
    let wy = lagrange_weights_at(&gll_positions_01(p0), s);
    let mut parents: Vec<(DofId, f64)> = Vec::new();
    // Grid walk over the (p0+1)² nodes: map (a, b) to
    // vertex / edge / face-interior DOFs.
    for (b, &wyb) in wy.iter().enumerate() {
        for (a, &wxa) in wx.iter().enumerate() {
            let w = wxa * wyb;
            if w.abs() < 1e-15 { continue; }
            let pe = p0 as usize;
            let dof = if a == 0 && b == 0 {
                face4[0] as DofId
            } else if a == pe && b == 0 {
                face4[1] as DofId
            } else if a == pe && b == pe {
                face4[2] as DofId
            } else if a == 0 && b == pe {
                face4[3] as DofId
            } else if b == 0 {
                // Bottom edge (v0→v1), ascending from v0.
                dofs_edge(dm, EdgeKey::new(face4[0], face4[1]), p0, face4[0], a - 1)
            } else if a == pe {
                // Right edge (v1→v2), ascending from v1.
                dofs_edge(dm, EdgeKey::new(face4[1], face4[2]), p0, face4[1], b - 1)
            } else if b == pe {
                // Top edge: runs v3→v2; node (a, p0) sits
                // a-1 interior steps from the v3 end.
                dofs_edge(dm, EdgeKey::new(face4[2], face4[3]), p0, face4[3], a - 1)
            } else if a == 0 {
                // Left edge: runs v0→v3; node (0, b) sits
                // b-1 interior steps from the v0 end.
                dofs_edge(dm, EdgeKey::new(face4[0], face4[3]), p0, face4[0], b - 1)
            } else {
                // Face interior: (iy-1)*(p0-1) + (ix-1).
                let fi = (b - 1) * (pe - 1) + (a - 1);
                dofs0[fi]
            };
            parents.push((dof, w));
        }
    }
    constraints.push(PRefineConstraint { constrained: dof, parents });
}

/// One tri-face min-rule row set for the order-`q` slave variant of a face
/// whose canonical orientation is `canon`: every slave dof interpolates the
/// order-`p_low` master trace over the face closure — vertices, oriented
/// edge runs, the master variant's interior dofs `dofs0` (empty when the
/// p_low variant holds none) — with the MASTER basis selected by
/// `master_gll` (D827-1: the basis of the master VARIANT's family —
/// `H1TriPk` GLL barycentrics when the p_low variant belongs to a
/// prism/pyramid, equispaced `TetPk` factory nodes when it belongs to a
/// tet) evaluated at the SLAVE variant's own node positions (`slave_gll`).
///
/// Both bases are closed P_k nodal sets in the same slot order as the
/// parent closure `[c0, c1, c2 | e(c0,c1) from c0 | e(c1,c2) from c1 |
/// e(c2,c0) from c2 | interior]`, so the weight vectors zip with the
/// parents directly.  `tri_w_cache` memoizes the weight rows per
/// (p_low, master_gll, q, slave_gll).
// The ten parameters are the minimal description of a tri-face row: the
// canonical face, the master variant (order, family, dofs), the slave
// variant (order, family, dofs) and the two shared scratch buffers —
// splitting them into structs would not reduce the arity of the state.
#[allow(clippy::too_many_arguments)]
fn tri_face_variant_rows(
    dm: &DofManager,
    canon: [NodeId; 3],
    p_low: u8,
    master_gll: bool,
    dofs0: &[DofId],
    q: u8,
    slave_gll: bool,
    dofs_q: &[DofId],
    tri_w_cache: &mut HashMap<(u8, bool, u8, bool), Vec<Vec<f64>>>,
    constraints: &mut Vec<PRefineConstraint>,
) {
    let pq = q as usize;
    let ws = tri_w_cache.entry((p_low, master_gll, q, slave_gll)).or_insert_with(|| {
        // Slave (λ1, λ2) per dof, in the running (j outer, i inner) order
        // of the builder's face-block allocation.
        let slave_pos: Vec<[f64; 2]> = if slave_gll {
            H1TriPk::new(pq).dof_coords()[3 * pq..3 * pq + dofs_q.len()]
                .iter().map(|rc| [rc[0], rc[1]]).collect()
        } else {
            (1..=pq.saturating_sub(2))
                .flat_map(|j| {
                    (1..=pq - 1 - j).map(move |i| [i as f64 / pq as f64, j as f64 / pq as f64])
                })
                .collect()
        };
        (0..dofs_q.len())
            .map(|k| {
                let (l1, l2) = (slave_pos[k][0], slave_pos[k][1]);
                if master_gll {
                    let master = H1TriPk::new(p_low as usize);
                    let mut w = vec![0.0_f64; master.n_dofs()];
                    master.eval_basis(&[l1, l2], &mut w);
                    w
                } else {
                    // `lagrange_weights_tri` takes (λ0, λ1).
                    lagrange_weights_tri(1.0 - l1 - l2, l1, p_low)
                }
            })
            .collect()
    });
    // Master closure in the shared slot order (see above).
    let mut parent_dofs: Vec<DofId> = canon.iter().map(|&n| n as DofId).collect();
    if p_low >= 2 {
        for (va, vb) in [(canon[0], canon[1]), (canon[1], canon[2]), (canon[2], canon[0])] {
            for k in 0..(p_low as usize - 1) {
                parent_dofs.push(dofs_edge(dm, EdgeKey::new(va, vb), p_low, va, k));
            }
        }
    }
    parent_dofs.extend_from_slice(dofs0);
    for (j, &dof) in dofs_q.iter().enumerate() {
        let parents: Vec<(DofId, f64)> = parent_dofs.iter()
            .zip(ws[j].iter())
            .filter(|&(_, &w)| w.abs() > 1e-15)
            .map(|(&d, &w)| (d, w))
            .collect();
        if !parents.is_empty() {
            constraints.push(PRefineConstraint { constrained: dof, parents });
        }
    }
}

/// 3D face variant min-rule constraints: every higher-order face variant
/// interpolates the lowest-order variant's face trace (tet faces: equispaced
/// barycentric; hex and prism quad faces: GLL tensor product; prism and
/// pyramid tri faces: GLL barycentric via `H1TriPk`; pyramid base quad
/// faces: GLL tensor product in Fuentes' j-reversed layout).
///
/// All geometries follow MFEM's `VariableOrderMinimumRule` exactly: the
/// master variant is the face's lowest **adjacent element order** (MFEM
/// `MakeDofTable` keeps zero-dof low-order variants, so e.g. a p2/p3 tri
/// face or a p1/p2 quad face is master'd by the empty low variant whose
/// closure is the face vertices plus the low-order edge dofs —
/// MFEM-verified for prisms by probe `tmp/d174_prism_p_probe.cpp` and for
/// hexes/tets by probe `tmp/d176_hextet_p_probe.cpp`).
fn detect_face_variant_constraints<M: MeshTopology>(
    dm: &DofManager,
    mesh: &M,
    constraints: &mut Vec<PRefineConstraint>,
) {
    let n_elems = mesh.n_elements();

    // Per-(face, order) variant basis flags, recomputed with the builder's
    // rule (D827-1): the master weights and the slave node positions of a
    // mixed tet × pyramid/prism tri face follow each variant's own family.
    let dm_orders: Vec<u8> =
        (0..n_elems as u32).map(|e| dm.element_order(e)).collect();
    let face_gll = collect_face_variants(mesh, &dm_orders);

    // Canonical face orientation (first-encountering element, matching the
    // builder's coordinate walk) + lowest adjacent order per face key.
    let mut face_canon3: HashMap<FaceKey, [NodeId; 3]> = HashMap::new();
    let mut face_canon4: HashMap<FaceKey, [NodeId; 4]> = HashMap::new();
    let mut face_low: HashMap<FaceKey, u8> = HashMap::new();
    for e in 0..n_elems as u32 {
        let et = mesh.element_type(e);
        let ns = elem_corners(et, mesh.element_nodes(e), mesh.geom_order() > 1);
        let p = dm.element_order(e);
        if is_hex_row(et) {
            for face4 in hex_quad_faces(&ns) {
                let face4 = canon_quad_face(face4);
                let key = FaceKey::new(face4[0], face4[1], face4[2]);
                face_low.entry(key)
                    .and_modify(|v| { if p < *v { *v = p; } })
                    .or_insert(p);
            }
        } else if is_pyramid_row(et) {
            let face4 = canon_quad_face(pyramid_base_face(&ns));
            let key = FaceKey::new(face4[0], face4[1], face4[2]);
            face_canon4.entry(key).or_insert(face4);
            face_low.entry(key)
                .and_modify(|v| { if p < *v { *v = p; } })
                .or_insert(p);
            for (a, b, c) in pyramid_tri_faces(&ns) {
                let key = FaceKey::new(a, b, c);
                face_canon3.entry(key).or_insert([a, b, c]);
                face_low.entry(key)
                    .and_modify(|v| { if p < *v { *v = p; } })
                    .or_insert(p);
            }
        } else if is_prism_row(et) {
            for (a, b, c) in prism_tri_faces(&ns) {
                let key = FaceKey::new(a, b, c);
                face_canon3.entry(key).or_insert([a, b, c]);
                face_low.entry(key)
                    .and_modify(|v| { if p < *v { *v = p; } })
                    .or_insert(p);
            }
            for face4 in prism_quad_faces(&ns) {
                let face4 = canon_quad_face(face4);
                let key = FaceKey::new(face4[0], face4[1], face4[2]);
                face_canon4.entry(key).or_insert(face4);
                face_low.entry(key)
                    .and_modify(|v| { if p < *v { *v = p; } })
                    .or_insert(p);
            }
        } else {
            for (a, b, c) in tet_faces(&ns) {
                let key = FaceKey::new(a, b, c);
                face_canon3.entry(key).or_insert([a, b, c]);
                face_low.entry(key)
                    .and_modify(|v| { if p < *v { *v = p; } })
                    .or_insert(p);
            }
        }
    }

    // Master-basis weights per (p_low, master_gll, q, slave_gll) tri face
    // tuple: entry j is the order-`p_low` master basis (per the master
    // variant's family) evaluated at the order-`q` slave variant's j-th
    // interior node (per the slave variant's family).
    let mut tri_w_cache: HashMap<(u8, bool, u8, bool), Vec<Vec<f64>>> = HashMap::new();

    // Face keys of prism, pyramid, tet and hex faces.
    for e in 0..n_elems as u32 {
        let et = mesh.element_type(e);
        let ns = elem_corners(et, mesh.element_nodes(e), mesh.geom_order() > 1);
        if is_prism_row(et) || is_pyramid_row(et) {
            // GLL tri faces (`H1_WedgeElement` sides / Fuentes pyramid
            // sides) + quad faces (prism sides: ascending GLL tensor;
            // pyramid base: Fuentes j-reversed layout).
            let tri_faces = if is_prism_row(et) {
                prism_tri_faces(&ns)
            } else {
                pyramid_tri_faces(&ns)
            };
            let quad_faces: Vec<([NodeId; 4], bool)> = if is_prism_row(et) {
                prism_quad_faces(&ns).into_iter()
                    .map(|f| (canon_quad_face(f), false)).collect()
            } else {
                vec![(canon_quad_face(pyramid_base_face(&ns)), true)]
            };
            for (a, b, c) in tri_faces {
                let key = FaceKey::new(a, b, c);
                let Some(variants) = dm.face_variants.get(&key) else { continue };
                // MFEM stores zero-dof low-order variants (`MakeDofTable`),
                // so a face whose lowest adjacent order contributes no
                // interior dofs still carries several variants and its
                // higher variants must be constrained to that order's
                // closure.  Skip faces holding no stored dof (single low
                // order with a zero-dof variant) or none above `p_low`.
                let Some(first_order) = variants.first().map(|(q, _)| *q) else { continue };
                let p_low = face_low.get(&key).copied().unwrap_or(first_order);
                if variants.iter().all(|(q, _)| *q <= p_low) { continue; }
                let canon = face_canon3.get(&key).copied().unwrap_or([a, b, c]);
                let dofs0 = variants.iter().find(|(p, _)| *p == p_low)
                    .map(|(_, d)| d.clone())
                    .unwrap_or_default();
                // D827-1: master/slave bases per (face, order) variant —
                // a tet sharing this face contributes equispaced variants.
                let gll_of = face_gll.get(&key);
                let master_gll = gll_of.and_then(|m| m.0.get(&p_low)).copied().unwrap_or(true);
                for (q, dofs_q) in variants {
                    if *q <= p_low { continue; }
                    let slave_gll = gll_of.and_then(|m| m.0.get(q)).copied().unwrap_or(true);
                    tri_face_variant_rows(
                        dm, canon, p_low, master_gll, &dofs0, *q, slave_gll, dofs_q,
                        &mut tri_w_cache, constraints);
                }
            }
            for (face4, reversed_y) in quad_faces {
                let key = FaceKey::new(face4[0], face4[1], face4[2]);
                let Some(variants) = dm.face_variants.get(&key) else { continue };
                let Some(first_order) = variants.first().map(|(q, _)| *q) else { continue };
                let p_low = face_low.get(&key).copied().unwrap_or(first_order);
                if variants.iter().all(|(q, _)| *q <= p_low) { continue; }
                let canon4 = face_canon4.get(&key).copied().unwrap_or(face4);
                let dofs0 = variants.iter().find(|(p, _)| *p == p_low)
                    .map(|(_, d)| d.clone())
                    .unwrap_or_default();
                for (q, dofs_q) in variants {
                    if *q <= p_low { continue; }
                    for (j, &dof) in dofs_q.iter().enumerate() {
                        quad_face_variant_constraint(
                            dm, canon4, (p_low, &dofs0), *q, dof, j, reversed_y,
                            constraints);
                    }
                }
            }
        } else if is_hex_row(et) {
            for face4 in hex_quad_faces(&ns) {
                let face4 = canon_quad_face(face4);
                let key = FaceKey::new(face4[0], face4[1], face4[2]);
                let Some(variants) = dm.face_variants.get(&key) else { continue };
                // MFEM stores zero-dof low-order variants (`MakeDofTable`),
                // so the master is the lowest ADJACENT element order even
                // when that order's face variant holds no interior dofs
                // (a p1 quad face): the higher variants interpolate the low
                // order's closure — 4 vertices only (probe
                // `tmp/d176_hextet_p_probe.cpp`, hex [2,1] row
                // `constrained 29 <- 4 × 0.25`).
                let Some(first_order) = variants.first().map(|(q, _)| *q) else { continue };
                let p_low = face_low.get(&key).copied().unwrap_or(first_order);
                if variants.iter().all(|(q, _)| *q <= p_low) { continue; }
                let dofs0 = variants.iter().find(|(p, _)| *p == p_low)
                    .map(|(_, d)| d.clone())
                    .unwrap_or_default();
                for (q, dofs_q) in variants {
                    if *q <= p_low { continue; }
                    for (j, &dof) in dofs_q.iter().enumerate() {
                        quad_face_variant_constraint(
                            dm, face4, (p_low, &dofs0), *q, dof, j, false, constraints);
                    }
                }
            }
        } else if is_tet_row(et) {
            // Tet: triangular face constraints, mastered at the lowest
            // ADJACENT order like MFEM: a p2 tri face stores no interior dof
            // but still masters (probe `tmp/d176_hextet_p_probe.cpp`, tet
            // [3,2] rows `constrained 40/41`: −1/9 on the 3 face vertices,
            // +4/9 on the 3 p2 edge dofs).  D827-1: on a mixed tet ×
            // pyramid/prism face the master (and any higher GLL-family
            // variant) takes its own family's basis via
            // [`tri_face_variant_rows`], shared with the pyramid/prism arm
            // so both sides of the face emit identical rows.
            for (v0, v1, v2) in tet_faces(&ns) {
                let key = FaceKey::new(v0, v1, v2);
                let Some(variants) = dm.face_variants.get(&key) else { continue };
                let Some(first_order) = variants.first().map(|(q, _)| *q) else { continue };
                let p_low = face_low.get(&key).copied().unwrap_or(first_order);
                if variants.iter().all(|(q, _)| *q <= p_low) { continue; }
                let canon3 = face_canon3.get(&key).copied().unwrap_or([v0, v1, v2]);
                let dofs0 = variants.iter().find(|(p, _)| *p == p_low)
                    .map(|(_, d)| d.clone())
                    .unwrap_or_default();
                let gll_of = face_gll.get(&key);
                let master_gll = gll_of.and_then(|m| m.0.get(&p_low)).copied().unwrap_or(false);
                for (q, dofs_q) in variants {
                    if *q <= p_low || dofs_q.is_empty() { continue; }
                    let slave_gll = gll_of.and_then(|m| m.0.get(q)).copied().unwrap_or(false);
                    tri_face_variant_rows(
                        dm, canon3, p_low, master_gll, &dofs0, *q, slave_gll, dofs_q,
                        &mut tri_w_cache, constraints);
                }
            }
        }
    }
}

/// The `k`-th (0-based, near `a`) DOF of the order-`p` variant of edge
/// `(a, b)`; `a`/`b` must be the edge endpoints and define the direction.
fn dofs_edge(dm: &DofManager, key: EdgeKey, p: u8, from: NodeId, k: usize) -> DofId {
    let variants = dm.edge_variants.get(&key)
        .unwrap_or_else(|| panic!("edge_variants missing for edge {key:?}"));
    let dofs = &variants.iter().find(|(q, _)| *q == p)
        .unwrap_or_else(|| panic!("edge variant p{p} missing for edge {key:?}")).1;
    if from == key.0 {
        dofs[k]
    } else {
        dofs[dofs.len() - 1 - k]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Constraint application and recovery
// ═══════════════════════════════════════════════════════════════════════════════

/// Apply p-refinement constraints to the assembled system `(mat, rhs)`.
///
/// For each constraint, the constrained DOF is eliminated by static
/// condensation via Pᵀ·K·P and Pᵀ·f (same pattern as hanging-node
/// constraints).
///
/// After solving, call [`recover_p_values`] to fill in constrained DOFs.
pub fn apply_p_constraints(
    mat: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    constraints: &[PRefineConstraint],
) {
    if constraints.is_empty() { return; }

    let n = mat.nrows;

    // Build constraint map: constrained → [(parent, weight)]
    let mut constraint_map: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
    for c in constraints {
        constraint_map.insert(c.constrained as usize,
            c.parents.iter().map(|&(d, w)| (d as usize, w)).collect());
    }

    // Recursive expansion: express a DOF in terms of free (unconstrained) DOFs
    fn expand(
        dof: usize,
        weight: f64,
        map: &HashMap<usize, Vec<(usize, f64)>>,
        out: &mut Vec<(usize, f64)>,
        visited: &mut HashSet<usize>,
        depth: usize,
    ) {
        if depth > 50 { return; }
        if !visited.insert(dof) { return; } // cycle guard
        if let Some(parents) = map.get(&dof) {
            for &(p, w) in parents {
                expand(p, weight * w, map, out, visited, depth + 1);
            }
        } else {
            out.push((dof, weight));
        }
    }

    // Build K' = Pᵀ·K·P in COO
    let mut coo = CooMatrix::<f64>::new(n, n);

    for i in 0..n {
        let start = mat.row_ptr[i];
        let end = mat.row_ptr[i + 1];

        let mut i_targets: Vec<(usize, f64)> = Vec::new();
        expand(i, 1.0, &constraint_map, &mut i_targets, &mut HashSet::new(), 0);

        for p in start..end {
            let j = mat.col_idx[p] as usize;
            let v = mat.values[p];
            if v.abs() < 1e-30 { continue; }

            let mut j_targets: Vec<(usize, f64)> = Vec::new();
            expand(j, 1.0, &constraint_map, &mut j_targets, &mut HashSet::new(), 0);

            for &(ii, ai) in &i_targets {
                for &(jj, aj) in &j_targets {
                    coo.add(ii, jj, v * ai * aj);
                }
            }
        }
    }

    // Set identity rows for constrained DOFs
    for c in constraints {
        coo.add(c.constrained as usize, c.constrained as usize, 1.0);
    }

    // Build f' = Pᵀ·f
    let mut new_rhs = vec![0.0_f64; n];
    for i in 0..n {
        if rhs[i].abs() < 1e-30 { continue; }
        let mut targets: Vec<(usize, f64)> = Vec::new();
        expand(i, 1.0, &constraint_map, &mut targets, &mut HashSet::new(), 0);
        for &(t, w) in &targets {
            new_rhs[t] += w * rhs[i];
        }
    }
    for c in constraints {
        new_rhs[c.constrained as usize] = 0.0;
    }
    rhs.copy_from_slice(&new_rhs);

    let new_mat: CsrMatrix<f64> = coo.into_csr();
    *mat = new_mat;
}

/// Recover constrained DOF values after solving.
///
/// Sets `x[constrained] = Σ w_i · x[parent_i]` for each constraint.
/// Processes in topological order so chained constraints are resolved.
pub fn recover_p_values(
    x: &mut [f64],
    constraints: &[PRefineConstraint],
) {
    if constraints.is_empty() { return; }

    let constrained_set: HashSet<usize> =
        constraints.iter().map(|c| c.constrained as usize).collect();

    let mut remaining: Vec<&PRefineConstraint> = constraints.iter().collect();
    let mut resolved: HashSet<usize> = HashSet::new();

    for _ in 0..constraints.len() + 1 {
        let mut progress = false;
        remaining.retain(|c| {
            let all_free = c.parents.iter().all(|&(d, _)| {
                let d_usize = d as usize;
                !constrained_set.contains(&d_usize) || resolved.contains(&d_usize)
            });
            if all_free {
                let mut val = 0.0;
                for &(parent, w) in &c.parents {
                    val += w * x[parent as usize];
                }
                x[c.constrained as usize] = val;
                resolved.insert(c.constrained as usize);
                progress = true;
                false
            } else {
                true
            }
        });
        if remaining.is_empty() || !progress { break; }
    }

    // Fallback
    for c in remaining {
        let mut val = 0.0;
        for &(parent, w) in &c.parents {
            val += w * x[parent as usize];
        }
        x[c.constrained as usize] = val;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Order field smoothing
// ═══════════════════════════════════════════════════════════════════════════════

/// Smooth the order field to limit order jumps between adjacent elements.
///
/// After this operation, no two adjacent elements differ in order by
/// more than `max_jump`. Uses an iterative smoothing algorithm until
/// convergence.
pub fn smooth_order_field<M: MeshTopology>(
    elem_orders: &mut [u8],
    mesh: &M,
    max_jump: u8,
) {
    let n_elems = mesh.n_elements();
    if n_elems <= 1 { return; }
    if max_jump == 0 { return; }

    // Build an edge-to-element adjacency map (O(n) instead of O(n²)).
    let mut edge_to_elems: HashMap<EdgeKey, Vec<u32>> = HashMap::new();
    for e in 0..n_elems as u32 {
        let et = mesh.element_type(e);
        let ns = elem_corners(et, mesh.element_nodes(e), mesh.geom_order() > 1);
        for (a, b) in elem_local_edges(et, &ns) {
            edge_to_elems.entry(EdgeKey::new(a, b)).or_default().push(e);
        }
    }

    // Use a worklist-based iterative smoothing for faster convergence.
    let mut changed = true;
    while changed {
        changed = false;
        for e in 0..n_elems as u32 {
            let p_e = elem_orders[e as usize];
            let et = mesh.element_type(e);
            let ns = elem_corners(et, mesh.element_nodes(e), mesh.geom_order() > 1);

            // Collect neighbor orders via the edge adjacency map (O(1) per edge).
            let mut neighbor_orders: Vec<u8> = Vec::new();
            for (a, b) in elem_local_edges(et, &ns) {
                let ek = EdgeKey::new(a, b);
                if let Some(adj_elems) = edge_to_elems.get(&ek) {
                    for &f in adj_elems {
                        if f != e {
                            neighbor_orders.push(elem_orders[f as usize]);
                        }
                    }
                }
            }

            for &p_nb in &neighbor_orders {
                if p_nb > p_e + max_jump {
                    elem_orders[e as usize] = p_nb - max_jump;
                    changed = true;
                    break;
                }
                if p_nb + max_jump < p_e {
                    elem_orders[e as usize] = p_nb + max_jump;
                    changed = true;
                    break;
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Refine / derefine operations
// ═══════════════════════════════════════════════════════════════════════════════

/// Increase the polynomial order of specified elements (p-refinement).
///
/// Returns a new [`DofManager`] with the updated orders plus the
/// constraints needed at mixed-order interfaces.
pub fn refine_p<M: MeshTopology>(
    dm: &DofManager,
    mesh: &M,
    elem_orders: &[u8],
    elem_ids: &[u32],
    new_order: u8,
) -> (DofManager, Vec<PRefineConstraint>) {
    let n_elems = mesh.n_elements();
    let mut new_orders: Vec<u8> = if let Some(ref existing) = dm.elem_orders {
        existing.clone()
    } else {
        elem_orders.to_vec()
    };
    assert_eq!(new_orders.len(), n_elems);

    for &e in elem_ids {
        let e_usize = e as usize;
        assert!(e_usize < n_elems, "refine_p: elem_id {e} out of range");
        assert!(new_order >= new_orders[e_usize],
            "refine_p: new_order {new_order} < current order {} for elem {e}", new_orders[e_usize]);
        new_orders[e_usize] = new_order;
    }

    let new_dm = build_variable_order_dof_manager(mesh, &new_orders);
    let constraints = detect_p_constraints(&new_dm, mesh, &new_orders);
    (new_dm, constraints)
}

/// Decrease the polynomial order of specified elements (p-derefinement).
///
/// Returns a new [`DofManager`] with the updated orders plus the
/// constraints needed at mixed-order interfaces.
pub fn derefine_p<M: MeshTopology>(
    dm: &DofManager,
    mesh: &M,
    elem_orders: &[u8],
    elem_ids: &[u32],
    new_order: u8,
) -> (DofManager, Vec<PRefineConstraint>) {
    let n_elems = mesh.n_elements();
    let mut new_orders: Vec<u8> = if let Some(ref existing) = dm.elem_orders {
        existing.clone()
    } else {
        elem_orders.to_vec()
    };
    assert_eq!(new_orders.len(), n_elems);

    for &e in elem_ids {
        let e_usize = e as usize;
        assert!(e_usize < n_elems, "derefine_p: elem_id {e} out of range");
        assert!(new_order <= new_orders[e_usize],
            "derefine_p: new_order {new_order} > current order {} for elem {e}", new_orders[e_usize]);
        new_orders[e_usize] = new_order;
    }

    let new_dm = build_variable_order_dof_manager(mesh, &new_orders);
    let constraints = detect_p_constraints(&new_dm, mesh, &new_orders);
    (new_dm, constraints)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    // ─── Lagrange weights ──────────────────────────────────────────────────

    #[test]
    fn lagrange_weights_p1_interpolates_exactly() {
        // P1: nodes at t=0, t=1. Weights at t=0.5: w_0=0.5, w_1=0.5
        let w = lagrange_weights_1d(0.5, 1);
        assert_eq!(w.len(), 2);
        assert!((w[0] - 0.5).abs() < 1e-14);
        assert!((w[1] - 0.5).abs() < 1e-14);
    }

    #[test]
    fn lagrange_weights_p2_at_midpoint() {
        // P2: nodes at t=0, 0.5, 1. At t=0.5: only node 1 is active
        let w = lagrange_weights_1d(0.5, 2);
        assert_eq!(w.len(), 3);
        assert!((w[1] - 1.0).abs() < 1e-14, "w[1]={} should be 1", w[1]);
        assert!(w[0].abs() < 1e-14);
        assert!(w[2].abs() < 1e-14);
    }

    #[test]
    fn lagrange_weights_p2_at_quarter() {
        let w = lagrange_weights_1d(0.25, 2);
        assert!((w[0] - 0.375).abs() < 1e-14, "w[0]={}", w[0]);
        assert!((w[1] - 0.75).abs() < 1e-14, "w[1]={}", w[1]);
        assert!((w[2] - -0.125).abs() < 1e-14, "w[2]={}", w[2]);
    }

    #[test]
    fn lagrange_weights_sum_to_one() {
        for p in 1..=6u8 {
            for &t in &[0.0, 0.25, 0.5, 0.75, 1.0, 0.333, 0.666] {
                let w = lagrange_weights_1d(t, p);
                let sum: f64 = w.iter().sum();
                assert!((sum - 1.0).abs() < 1e-12,
                    "p={p} t={t}: sum={}", sum);
            }
        }
    }

    #[test]
    fn lagrange_weights_at_matches_equispaced() {
        // The generic node-set evaluation must agree with the equispaced one.
        for p in 1..=4u8 {
            let nodes = equi_positions_01(p);
            for &t in &[0.1, 0.5, 0.9] {
                let wa = lagrange_weights_at(&nodes, t);
                let wb = lagrange_weights_1d(t, p);
                for (a, b) in wa.iter().zip(wb.iter()) {
                    assert!((a - b).abs() < 1e-13);
                }
            }
        }
    }

    #[test]
    fn gll_positions_match_mfem_closed_points() {
        // GLL 1D positions for p=2 are (0, 0.5, 1); for p=3 they are the
        // endpoints plus the roots of P'3 (±1/sqrt(5)) — NOT the equispaced
        // ones.  Matches MFEM Poly1D ClosedPoints(p, GaussLobatto).
        let p2 = gll_positions_01(2);
        assert!((p2[1] - 0.5).abs() < 1e-14);
        let p3 = gll_positions_01(3);
        let s = 1.0 / 5.0_f64.sqrt();
        assert!((p3[1] - 0.5 * (1.0 - s)).abs() < 1e-14, "p3[1]={}", p3[1]);
        assert!((p3[2] - 0.5 * (1.0 + s)).abs() < 1e-14, "p3[2]={}", p3[2]);
    }

    // ─── Variable-order DofManager (variant scheme) ────────────────────────

    #[test]
    fn variable_order_uniform_equivalent_to_build_pk() {
        // Uniform p=3 through the variable-order path should be structurally
        // equivalent to build_pk (same n_dofs, same per-element DOF counts,
        // same vertex DOFs).
        let mesh = Mesh::<2>::unit_square_tri(2);
        let elem_orders = vec![3u8; mesh.n_elements()];
        let dm_var = build_variable_order_dof_manager(&mesh, &elem_orders);
        let dm_pk = DofManager::new(&mesh, 3);

        assert_eq!(dm_var.n_dofs, dm_pk.n_dofs,
            "n_dofs: var={}, pk={}", dm_var.n_dofs, dm_pk.n_dofs);
        assert_eq!(dm_var.n_vertex_dofs, dm_pk.n_vertex_dofs);
        for e in 0..mesh.n_elements() as u32 {
            let dofs_var = dm_var.element_dofs(e);
            let dofs_pk  = dm_pk.element_dofs(e);
            assert_eq!(dofs_var.len(), dofs_pk.len(),
                "elem {e}: var {} vs pk {}", dofs_var.len(), dofs_pk.len());
            let nodes = mesh.element_nodes(e);
            assert_eq!(&dofs_var[..nodes.len()], nodes,
                "elem {e}: vertex DOFs mismatch");
        }
    }

    #[test]
    fn variable_order_mixed_count() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let n_elems = mesh.n_elements();
        let mut elem_orders = vec![2u8; n_elems];
        elem_orders[0] = 3;

        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);
        let dm_p2 = DofManager::new(&mesh, 2);
        let dm_p3 = DofManager::new(&mesh, 3);
        assert!(dm.n_dofs > dm_p2.n_dofs,
            "mixed P2/P3 should have more DOFs than all-P2: {} vs {}",
            dm.n_dofs, dm_p2.n_dofs);
        assert!(dm.n_dofs < dm_p3.n_dofs,
            "mixed P2/P3 should have fewer DOFs than all-P3: {} vs {}",
            dm.n_dofs, dm_p3.n_dofs);
    }

    #[test]
    fn variable_order_variant_sets_match_adjacent_orders() {
        // Two P2 tris sharing the diagonal, one promoted to P3: the shared
        // (diagonal) edge must hold variants {2, 3}; other edges a single
        // variant matching their element's order.
        let mesh = Mesh::<2>::unit_square_tri(1);
        assert_eq!(mesh.n_elements(), 2);
        let elem_orders = vec![3u8, 2u8];
        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);

        let n_multi = dm.edge_variants.values().filter(|v| v.len() > 1).count();
        assert_eq!(n_multi, 1, "exactly the shared edge holds two variants");
        for (_key, variants) in &dm.edge_variants {
            if variants.len() > 1 {
                assert_eq!(variants[0].0, 2);
                assert_eq!(variants[1].0, 3);
                assert_eq!(variants[0].1.len(), 1); // p2: 1 interior edge dof
                assert_eq!(variants[1].1.len(), 2); // p3: 2 interior edge dofs
            }
        }
    }

    #[test]
    fn variable_order_uses_elem_dof_offsets() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let n_elems = mesh.n_elements();
        let mut elem_orders = vec![2u8; n_elems];
        elem_orders[0] = 3;

        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);
        assert!(dm.elem_dof_offsets.is_some(),
            "variable-order should have elem_dof_offsets");
        assert_eq!(dm.dofs_per_elem, 0,
            "variable-order should have dofs_per_elem = 0");

        let dofs0 = dm.element_dofs(0);
        let dofs1 = dm.element_dofs(1);
        assert!(dofs0.len() > dofs1.len(),
            "P3 element should have more DOFs than P2: {} vs {}",
            dofs0.len(), dofs1.len());
    }

    #[test]
    fn variable_order_vertex_dofs_correct() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let n_elems = mesh.n_elements();
        let mut elem_orders = vec![2u8; n_elems];
        elem_orders[0] = 3;

        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);
        for e in 0..n_elems as u32 {
            let dofs = dm.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            assert_eq!(&dofs[..nodes.len()], nodes,
                "elem {e}: vertex DOFs mismatch");
        }
    }

    #[test]
    fn variable_order_elem_orders_stored() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let n_elems = mesh.n_elements();
        let mut elem_orders = vec![2u8; n_elems];
        elem_orders[0] = 4;
        elem_orders[1] = 3;

        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);
        let stored = dm.elem_orders.clone().expect("elem_orders should be Some");
        assert_eq!(stored, elem_orders);
        assert_eq!(dm.element_order(0), 4);
        assert_eq!(dm.element_order(1), 3);
        assert_eq!(dm.element_order(2), 2);
    }

    // ─── Constraint detection ──────────────────────────────────────────────

    #[test]
    fn detect_p_constraints_p2_p3_interface() {
        let mesh = Mesh::<2>::unit_square_tri(1);
        assert_eq!(mesh.n_elements(), 2);
        let elem_orders = vec![3u8, 2u8]; // one P3, one P2
        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);
        let constraints = detect_p_constraints(&dm, &mesh, &elem_orders);

        // The shared edge holds P2- and P3-variants; both P3-variant edge
        // DOFs are constrained to the P2 interpolation.
        assert_eq!(constraints.len(), 2,
            "P3-variant edge DOFs on the shared edge should be constrained");
    }

    #[test]
    fn detect_constraints_none_for_uniform() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let elem_orders = vec![2u8; mesh.n_elements()];
        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);
        let constraints = detect_p_constraints(&dm, &mesh, &elem_orders);
        assert!(constraints.is_empty(),
            "uniform order should have no constraints");
    }

    #[test]
    fn p_constraint_parents_sum_to_one() {
        let mesh = Mesh::<2>::unit_square_tri(1);
        let elem_orders = vec![3u8, 2u8];
        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);
        let constraints = detect_p_constraints(&dm, &mesh, &elem_orders);

        for (i, c) in constraints.iter().enumerate() {
            let sum: f64 = c.parents.iter().map(|&(_, w)| w).sum();
            assert!((sum - 1.0).abs() < 1e-12,
                "constraint {i}: parent weights sum to {}, expected 1", sum);
        }
    }

    #[test]
    fn p_constraint_reproduces_lower_order_interpolant() {
        // MFEM VariableOrderMinimumRule semantics: a quadratic field's nodal
        // values must satisfy every P2 -> P3 edge constraint exactly (the
        // quadratic trace through the P2 nodes equals the P3 nodal values at
        // the P3-variant positions).
        let mesh = Mesh::<2>::unit_square_tri(1);
        let elem_orders = vec![3u8, 2u8];
        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);
        let constraints = detect_p_constraints(&dm, &mesh, &elem_orders);
        assert!(!constraints.is_empty());

        let g = |x: [f64; 2]| x[0] * x[0] - 2.0 * x[1] * x[0] + 0.5 * x[0] * x[1];
        let mut ug = vec![0.0_f64; dm.n_dofs];
        for d in 0..dm.n_dofs as u32 {
            let c = dm.dof_coord(d);
            ug[d as usize] = g([c[0], c[1]]);
        }
        for c in &constraints {
            let lhs = ug[c.constrained as usize];
            let rhs: f64 = c.parents.iter().map(|&(d, w)| w * ug[d as usize]).sum();
            assert!((lhs - rhs).abs() < 1e-12,
                "quadratic field must satisfy the p constraint exactly");
        }
    }

    // ─── Apply/recover constraints ─────────────────────────────────────────

    #[test]
    fn apply_p_constraints_modifies_matrix() {
        use fem_linalg::{CooMatrix, CsrMatrix};

        let mesh = Mesh::<2>::unit_square_tri(1);
        let elem_orders = vec![3u8, 2u8];
        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);
        let constraints = detect_p_constraints(&dm, &mesh, &elem_orders);

        let n = dm.n_dofs;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 2.0); }
        let mut mat: CsrMatrix<f64> = coo.into_csr();
        let mut rhs = vec![1.0_f64; n];

        let constrained_count = constraints.len();
        if constrained_count > 0 {
            apply_p_constraints(&mut mat, &mut rhs, &constraints);
            for c in &constraints {
                let d = c.constrained as usize;
                assert!((mat.get(d, d) - 1.0).abs() < 1e-14,
                    "constrained DOF {d} diagonal should be 1");
                assert!(rhs[d].abs() < 1e-14,
                    "constrained DOF {d} RHS should be 0");
            }
        }
    }

    #[test]
    fn recover_p_values_p2_p3_interface() {
        let mesh = Mesh::<2>::unit_square_tri(1);
        let elem_orders = vec![3u8, 2u8];
        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);
        let constraints = detect_p_constraints(&dm, &mesh, &elem_orders);

        if constraints.is_empty() { return; }

        let mut x: Vec<f64> = (0..dm.n_dofs).map(|i| (i as f64) * 0.1).collect();
        let before = x.clone();
        recover_p_values(&mut x, &constraints);

        for c in &constraints {
            let expected: f64 = c.parents.iter()
                .map(|&(d, w)| w * before[d as usize])
                .sum();
            assert!((x[c.constrained as usize] - expected).abs() < 1e-12,
                "constrained DOF {}: got {}, expected {}",
                c.constrained, x[c.constrained as usize], expected);
        }
    }

    // ─── refine_p / derefine_p ─────────────────────────────────────────────

    #[test]
    fn refine_p_increases_dofs() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let elem_orders = vec![2u8; mesh.n_elements()];
        let dm = DofManager::new(&mesh, 2);

        let (dm_refined, _) = refine_p(&dm, &mesh, &elem_orders, &[0, 1], 3);
        assert!(dm_refined.n_dofs > dm.n_dofs,
            "refine_p should increase n_dofs: {} vs {}",
            dm_refined.n_dofs, dm.n_dofs);
    }

    #[test]
    fn derefine_p_decreases_dofs() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let elem_orders = vec![3u8; mesh.n_elements()];
        let dm = DofManager::new(&mesh, 3);

        let (dm_derefined, _) = derefine_p(&dm, &mesh, &elem_orders, &[0], 2);
        assert!(dm_derefined.n_dofs < dm.n_dofs,
            "derefine_p should decrease n_dofs: {} vs {}",
            dm_derefined.n_dofs, dm.n_dofs);
    }

    // ─── smooth_order_field ────────────────────────────────────────────────

    #[test]
    fn smooth_order_field_clamps_jumps() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let n_elems = mesh.n_elements();
        let mut orders = vec![1u8; n_elems];
        orders[0] = 5;

        smooth_order_field(&mut orders, &mesh, 1);

        let dim = mesh.dim();
        for e in 0..n_elems as u32 {
            let orders_e = orders[e as usize];
            let ns = mesh.element_nodes(e);
            let edges: Vec<(NodeId, NodeId)> = if dim == 2 {
                tri_edges(ns)
            } else {
                tet_edges(ns)
            };
            for (a, b) in &edges {
                let ek = EdgeKey::new(*a, *b);
                for f in 0..n_elems as u32 {
                    if f == e { continue; }
                    let fnodes = mesh.element_nodes(f);
                    let fedges: Vec<(NodeId, NodeId)> = if dim == 2 {
                        tri_edges(fnodes)
                    } else {
                        tet_edges(fnodes)
                    };
                    for &(fa, fb) in &fedges {
                        if EdgeKey::new(fa, fb) == ek {
                            let orders_f = orders[f as usize];
                            let diff = if orders_e > orders_f
                                { orders_e - orders_f }
                                else { orders_f - orders_e };
                            assert!(diff <= 1,
                                "order jump too large: elem {e} (p={orders_e}) vs elem {f} (p={orders_f})");
                        }
                    }
                }
            }
        }
    }

    // ─── 3D tet ────────────────────────────────────────────────────────────

    #[test]
    fn variable_order_3d_tet() {
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let n_elems = mesh.n_elements();
        let mut elem_orders = vec![2u8; n_elems];
        elem_orders[0] = 3;

        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);
        assert!(dm.elem_dof_offsets.is_some());
        assert!(dm.n_dofs > 0);

        let dofs0 = dm.element_dofs(0);
        let dofs1 = dm.element_dofs(1);
        assert!(dofs0.len() > dofs1.len(),
            "3D P3 element should have more DOFs than P2");
    }

    #[test]
    fn face_constraints_3d_p2_p3_interface() {
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let n_elems = mesh.n_elements();
        let mut elem_orders = vec![2u8; n_elems];
        elem_orders[0] = 3; // P3 element 0, P2 elements

        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);
        let constraints = detect_p_constraints(&dm, &mesh, &elem_orders);

        let has_face_dofs = dm.face_variants.values()
            .any(|v| v.iter().any(|(p, dofs)| *p >= 3 && !dofs.is_empty()));
        assert!(has_face_dofs, "P3 should have face DOFs");

        assert!(!constraints.is_empty(),
            "P3-P2 interface should produce constraints, got 0");
    }

    #[test]
    fn face_dof_coords_3d_not_centroid() {
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let n_elems = mesh.n_elements();
        let elem_orders = vec![3u8; n_elems]; // all P3

        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);

        let mut face_dofs_found = false;
        for (_, dofs) in &dm.face_pk_map {
            for &dof in dofs {
                let x = dm.dof_coord(dof);
                if x[0].abs() > 1e-10 || x[1].abs() > 1e-10 || x[2].abs() > 1e-10 {
                    face_dofs_found = true;
                }
            }
        }
        assert!(face_dofs_found, "face DOF coordinates should be non-zero");
    }

    // ─── 2D Quad ──────────────────────────────────────────────────────────

    #[test]
    fn variable_order_quad() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let n_elems = mesh.n_elements();
        let elem_orders = vec![2u8; n_elems];
        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);

        for e in 0..n_elems as u32 {
            let dofs = dm.element_dofs(e);
            // Q2: 4 vertices + 4 edges × 1 + 1 interior = 9 DOFs
            assert_eq!(dofs.len(), 9,
                "Q2 element {e} should have 9 DOFs, got {}", dofs.len());
        }
        assert!(dm.n_dofs > 0, "Quad DM should have DOFs");
    }

    #[test]
    fn variable_order_quad_p3() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let n_elems = mesh.n_elements();
        let elem_orders = vec![3u8; n_elems]; // all Q3
        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);

        for e in 0..n_elems as u32 {
            let dofs = dm.element_dofs(e);
            // Q3: 4 vertices + 4 edges × 2 + interior (p-1)²=4 = 16 DOFs
            assert_eq!(dofs.len(), 16,
                "Q3 element {e} should have 16 DOFs, got {}", dofs.len());
        }
    }

    #[test]
    fn variable_order_quad_p2_p3_interface() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let n_elems = mesh.n_elements();
        let mut elem_orders = vec![2u8; n_elems];
        elem_orders[0] = 3; // Q3 element 0, Q2 element 1..3

        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);
        let constraints = detect_p_constraints(&dm, &mesh, &elem_orders);

        // Shared edges hold Q2- and Q3-variants; every Q3-variant edge DOF on
        // a mixed edge is constrained.
        assert!(!constraints.is_empty(),
            "Q3-Q2 interface should have edge constraints");
        for c in &constraints {
            let sum: f64 = c.parents.iter().map(|&(_, w)| w).sum();
            assert!((sum - 1.0).abs() < 1e-12);
        }
    }

    // ─── 3D Hex ───────────────────────────────────────────────────────────

    #[test]
    fn variable_order_hex() {
        let mesh = Mesh::<3>::unit_cube_hex(1);
        let n_elems = mesh.n_elements();
        let elem_orders = vec![2u8; n_elems]; // all Q2
        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);

        for e in 0..n_elems as u32 {
            let dofs = dm.element_dofs(e);
            // Q2 hex (tensor product): 8 vertices + 12 edges×1 + 6 faces×1 + 1 volume = 27
            assert_eq!(dofs.len(), 27,
                "Q2 hex element {e} should have 27 DOFs, got {}", dofs.len());
        }
    }

    #[test]
    fn variable_order_hex_p3() {
        let mesh = Mesh::<3>::unit_cube_hex(1);
        let n_elems = mesh.n_elements();
        let elem_orders = vec![3u8; n_elems]; // all Q3
        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);

        for e in 0..n_elems as u32 {
            let dofs = dm.element_dofs(e);
            // Q3 hex: 8 + 12×2 + 6×4 + 8 = 64 DOFs
            assert_eq!(dofs.len(), 64,
                "Q3 hex element {e} should have 64 DOFs, got {}", dofs.len());
        }
    }

    // ─── 2D NC (hanging) geometry + hp constraints ─────────────────────────

    /// Isotropic (XY) refinement of element `e` of a quad mesh at midpoint,
    /// matching MFEM's NCMesh child order [BL, BR, TR, TL] (local corners).
    fn refine_quad_isotropic(mesh: &Mesh<2>, e: u32) -> Mesh<2> {
        use fem_mesh::amr::general_refinement::{general_refinement_2d, Refinement};
        // Split along X (bottom (0,1) and top (3,2) edges), then Y both
        // children; children replace the parent in place.
        let r1 = general_refinement_2d(mesh, &[Refinement::with_midpoint(e, 1)]);
        let children = r1.transforms.find_children(e);
        assert_eq!(children.len(), 2);
        let r2 = general_refinement_2d(&r1.mesh, &[
            Refinement::with_midpoint(children[0], 2),
            Refinement::with_midpoint(children[1], 2),
        ]);
        // The second call splits both children in place: the four leaves of
        // the original element occupy slots [c0, c0+1, c1, c1+1] in the order
        // [BL, TL, BR, TR] (local corners); MFEM's XY split creates
        // [BL, BR, TR, TL] → swap slots 1 and 3.
        let mut mesh = r2.mesh;
        let (c0, c1) = (children[0] as usize, children[1] as usize);
        let (i, j) = (c0 + 1, c1 + 1);
        for k in 0..4 {
            mesh.conn.swap(i * 4 + k, j * 4 + k);
        }
        mesh.elem_tags.swap(i, j);
        mesh
    }

    #[test]
    fn nc_geometry_detects_master_slave_hanging() {
        // 2x2 quad mesh; isotropic refinement of element 0 creates hanging
        // nodes on the two edges shared with unrefined neighbors.
        let mesh0 = Mesh::<2>::make_cartesian_2d(2, 2, 1.0, 1.0);
        let mesh = refine_quad_isotropic(&mesh0, 0);
        assert_eq!(mesh.n_elems(), 7);

        let geom = detect_nc_geometry_2d(&mesh);
        assert!(!geom.masters.is_empty(), "master edges must exist");
        assert_eq!(geom.hanging_nodes.len(), 2,
            "one hanging node per INTERIOR master edge (top and right); boundary split nodes are not hanging");
        for (_n, mkey, _t) in &geom.hanging_nodes {
            assert!(geom.masters.iter().any(|m| &m.key == mkey));
        }
        // Slave edges: only the two segments per INTERIOR master edge (the
        // boundary split segments have no containing element edge).
        assert_eq!(geom.slaves.len(), 4,
            "two slave segments per interior master edge, two interior masters");
        for s in &geom.slaves {
            assert!((s.t_a - s.t_b).abs() - 0.5 < 1e-12,
                "slave segment covers half the master edge (t_a={}, t_b={})", s.t_a, s.t_b);
        }
    }

    #[test]
    fn nc_hp_constraints_count_matches_mfem() {
        // 2x2 quad, element 0 refined isotropically, all orders 1:
        // ndofs = nodes (P1); constrained = the 4 hanging vertices; MFEM's
        // GetTrueVSize for this configuration is 16 - 4 = 12.
        let mesh0 = Mesh::<2>::make_cartesian_2d(2, 2, 1.0, 1.0);
        let mesh = refine_quad_isotropic(&mesh0, 0);
        let orders = vec![1u8; mesh.n_elems()];
        let dm = build_variable_order_dof_manager(&mesh, &orders);
        let constraints = detect_p_constraints(&dm, &mesh, &orders);
        assert_eq!(dm.n_dofs, mesh.n_nodes(), "P1: dofs == nodes");
        assert_eq!(constraints.len(), 2, "2 hanging vertex constraints");
        let n_true = dm.n_dofs - constraints.len();
        assert_eq!(n_true, 12, "true dofs (MFEM GetTrueVSize) = 12");
    }
}
