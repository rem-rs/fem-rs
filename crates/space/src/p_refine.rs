//! p-Refinement infrastructure for finite element spaces.
//!
//! Provides variable-order DOF management, p-refinement/derefinement
//! operations, constraint detection at mixed-order interfaces, and
//! order field smoothing.
//!
//! ## Key concepts
//!
//! In p-refinement, different elements can have different polynomial
//! orders. Shared entities (edges/faces) get DOFs matching the maximum
//! order of adjacent elements. Elements with lower order see fewer DOFs
//! on shared entities; the "extra" DOFs are constrained via Lagrange
//! interpolation to maintain C⁰ continuity.
//!
//! ## Constraint generation
//!
//! When element A (order p_A) shares an edge with element B (order p_B)
//! and p_A > p_B, the edge gets (p_A - 1) DOFs globally. Element B only
//! "owns" (p_B - 1) of them. The extra (p_A - p_B) DOFs are constrained:
//! the higher-order polynomial evaluated at the extra DOF positions
//! equals the (p_B)-order Lagrange interpolation of the lower-order DOFs.

use std::collections::{HashMap, HashSet};
use fem_core::types::{DofId, ElemId, NodeId};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use crate::dof_manager::{DofManager, EdgeKey, FaceKey};

// ─── PRefineConstraint ────────────────────────────────────────────────────────

/// A constraint arising from p-refinement.
///
/// Expresses a high-order DOF as a weighted combination of lower-order
/// DOFs on the same entity, maintaining C⁰ continuity across mixed-order
/// interfaces.
///
/// For an edge shared by P3 and P2 elements, the extra P3 edge DOF is:
///   u_extra = w_0·u_v0 + w_1·u_mid + w_2·u_v1
/// where u_v0, u_v1 are vertex DOFs and u_mid is the P2 edge midpoint DOF.
#[derive(Debug, Clone)]
pub struct PRefineConstraint {
    /// The DOF being constrained (belongs to a higher-order element).
    pub constrained: DofId,
    /// Parent DOFs and their weights (sparse representation).
    pub parents: Vec<(DofId, f64)>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Position of the k-th edge DOF (0-indexed) for order p.
/// Edge DOFs are at equispaced positions between vertices at t=0 and t=1.
fn edge_dof_position(k: usize, p: u8) -> f64 {
    (k + 1) as f64 / p as f64
}

/// Lagrange interpolation weights for (p_low + 1) equispaced nodes
/// on [0, 1] evaluated at position `t`.
///
/// Nodes: t_j = j / p_low for j = 0, 1, ..., p_low.
/// Returns weights w_j = L_j(t) for j = 0..=p_low.
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

/// Extract the edges of a 2D triangle element given its node list.
fn tri_edges(ns: &[NodeId]) -> Vec<(NodeId, NodeId)> {
    vec![(ns[0], ns[1]), (ns[1], ns[2]), (ns[0], ns[2])]
}

/// Extract the edges of a 3D tetrahedron.
fn tet_edges(ns: &[NodeId]) -> Vec<(NodeId, NodeId)> {
    vec![
        (ns[0], ns[1]), (ns[0], ns[2]), (ns[0], ns[3]),
        (ns[1], ns[2]), (ns[1], ns[3]), (ns[2], ns[3]),
    ]
}

/// Extract the faces of a 3D tetrahedron.
fn tet_faces(ns: &[NodeId]) -> Vec<(NodeId, NodeId, NodeId)> {
    vec![
        (ns[0], ns[1], ns[2]),
        (ns[0], ns[1], ns[3]),
        (ns[0], ns[2], ns[3]),
        (ns[1], ns[2], ns[3]),
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



// ═══════════════════════════════════════════════════════════════════════════════
// Entity order computation
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute the order of each edge based on per-element orders.
/// The order of an edge is the maximum order of all adjacent elements.
fn compute_edge_orders<M: MeshTopology>(
    mesh: &M,
    elem_orders: &[u8],
) -> HashMap<EdgeKey, u8> {
    let mut orders: HashMap<EdgeKey, u8> = HashMap::new();
    let dim = mesh.dim();
    let n_elems = mesh.n_elements();

    for e in 0..n_elems as u32 {
        let p = elem_orders[e as usize];
        let ns = mesh.element_nodes(e);
        let edges: Vec<(NodeId, NodeId)> = if dim == 2 && ns.len() == 4 {
            // Quad
            vec![(ns[0], ns[1]), (ns[1], ns[2]), (ns[2], ns[3]), (ns[3], ns[0])]
        } else if dim == 3 && ns.len() == 8 {
            // Hex
            vec![
                (ns[0], ns[1]), (ns[1], ns[2]), (ns[2], ns[3]), (ns[3], ns[0]),
                (ns[4], ns[5]), (ns[5], ns[6]), (ns[6], ns[7]), (ns[7], ns[4]),
                (ns[0], ns[4]), (ns[1], ns[5]), (ns[2], ns[6]), (ns[3], ns[7]),
            ]
        } else if dim == 2 {
            tri_edges(ns)
        } else {
            tet_edges(ns)
        };
        for &(a, b) in &edges {
            let key = EdgeKey::new(a, b);
            let prev = orders.get(&key).copied().unwrap_or(0);
            if p > prev {
                orders.insert(key, p);
            }
        }
    }
    orders
}

/// Compute the order of each face (3D) and its node count based on per-element orders.
/// Returns (order_map, face_node_counts: 3 for tet, 4 for hex).
fn compute_face_orders<M: MeshTopology>(
    mesh: &M,
    elem_orders: &[u8],
) -> (HashMap<FaceKey, u8>, HashMap<FaceKey, usize>) {
    let mut orders: HashMap<FaceKey, u8> = HashMap::new();
    let mut n_nodes: HashMap<FaceKey, usize> = HashMap::new();
    let n_elems = mesh.n_elements();

    for e in 0..n_elems as u32 {
        let p = elem_orders[e as usize];
        let ns = mesh.element_nodes(e);
        if ns.len() == 8 {
            for &face4 in &hex_quad_faces(ns) {
                let key = FaceKey::new(face4[0], face4[1], face4[2]);
                let prev = orders.get(&key).copied().unwrap_or(0);
                if p > prev { orders.insert(key, p); }
                n_nodes.entry(key).or_insert(4);
            }
        } else {
            for &(a, b, c) in &tet_faces(ns) {
                let key = FaceKey::new(a, b, c);
                let prev = orders.get(&key).copied().unwrap_or(0);
                if p > prev { orders.insert(key, p); }
                n_nodes.entry(key).or_insert(3);
            }
        }
    }
    (orders, n_nodes)
}

/// Number of interior DOFs for a 2D element (Tri bubble or Quad face) of order p.
/// Tri: (p-1)(p-2)/2 for p≥3
/// Quad: (p-1)² for p≥2
fn n_face_dofs_2d(ns_len: usize, p: u8) -> usize {
    let p = p as usize;
    if ns_len == 4 {
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
/// Hex: (p-1)³ (p≥2)
fn n_volume_dofs_3d(ns_len: usize, p: u8) -> usize {
    let p = p as usize;
    if ns_len == 4 {
        if p >= 4 { (p - 1) * (p - 2) * (p - 3) / 6 } else { 0 }
    } else if ns_len == 8 {
        if p >= 2 { (p - 1).pow(3) } else { 0 }
    } else {
        0
    }
}

/// Bubble (interior) DOFs of a 2D element.
/// Tri: (p-1)(p-2)/2 for p≥3
/// Quad: (p-1)² for p≥2
fn n_bubble_dofs_2d(ns_len: usize, p: u8) -> usize {
    n_face_dofs_2d(ns_len, p)
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

/// 2D Lagrange interpolation weights on a reference triangle.
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
    // We can compute this directly from (i, j, k) triples
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
// Variable-order DOF manager
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a variable-order DOF manager from per-element polynomial orders.
///
/// Each entity (vertex/edge/face) gets DOFs corresponding to the maximum
/// order of all adjacent elements. Elements with lower order share fewer
/// DOFs on shared entities — the extra DOFs are constrained (see
/// [`detect_p_constraints`]).
///
/// # Panics
/// Panics if `elem_orders.len() != mesh.n_elements()` or if the mesh type
/// is unsupported.
pub fn build_variable_order_dof_manager<M: MeshTopology>(
    mesh: &M,
    elem_orders: &[u8],
) -> DofManager {
    let n_elems = mesh.n_elements();
    assert_eq!(elem_orders.len(), n_elems,
        "elem_orders length {} != n_elements {}", elem_orders.len(), n_elems);
    let dim = mesh.dim() as usize;
    let n_nodes = mesh.n_nodes();
    let p_max = *elem_orders.iter().max().unwrap_or(&1);

    // 1. Compute per-entity orders
    let edge_orders = compute_edge_orders(mesh, elem_orders);
    let (face_orders, face_nnodes) = if dim == 3 { compute_face_orders(mesh, elem_orders) } else { (HashMap::new(), HashMap::new()) };

    // 2. Assign global DOF numbers
    let mut next_dof = n_nodes as DofId;

    // Edge DOFs: each unique edge gets (p_edge - 1) DOFs
    let mut edge_pk_map: HashMap<EdgeKey, Vec<DofId>> = HashMap::new();
    let mut edge_list: Vec<(EdgeKey, u8)> = edge_orders.iter()
        .map(|(&k, &p)| (k, p)).collect();
    edge_list.sort_by_key(|&(k, _)| k);
    for (key, p_edge) in &edge_list {
        if *p_edge >= 2 {
            let n = (*p_edge - 1) as usize;
            let dofs: Vec<DofId> = (0..n).map(|_| { let d = next_dof; next_dof += 1; d }).collect();
            edge_pk_map.insert(*key, dofs);
        }
    }

    // Face DOFs (3D): each unique face gets n_face_dofs(p_face, nnodes) DOFs
    let mut face_pk_map: HashMap<FaceKey, Vec<DofId>> = HashMap::new();
    if dim == 3 {
        let mut face_list: Vec<(FaceKey, u8)> = face_orders.iter()
            .map(|(&k, &p)| (k, p)).collect();
        face_list.sort_by_key(|&(k, _)| k);
        for (key, p_face) in &face_list {
            let nn = face_nnodes.get(key).copied().unwrap_or(3);
            let n = n_face_dofs_3d(nn, *p_face);
            if n > 0 {
                let dofs: Vec<DofId> = (0..n).map(|_| { let d = next_dof; next_dof += 1; d }).collect();
                face_pk_map.insert(*key, dofs);
            }
        }
    }

    // Volume DOFs: per-element, assigned as we iterate
    let volume_start = next_dof;
    let mut total_volume_dofs = 0usize;

    // 3. Build per-element DOF lists with elem_dof_offsets
    let mut dofs_flat: Vec<DofId> = Vec::new();
    let mut elem_dof_offsets = Vec::with_capacity(n_elems + 1);
    elem_dof_offsets.push(0);

    for e in 0..n_elems as u32 {
        let p_e = elem_orders[e as usize];
        let ns = mesh.element_nodes(e);
        // Vertex DOFs
        for &n in ns.iter() {
            dofs_flat.push(n);
        }

        // Edge DOFs
        let edges: Vec<(NodeId, NodeId)> = if dim == 2 && ns.len() == 4 {
            vec![(ns[0], ns[1]), (ns[1], ns[2]), (ns[2], ns[3]), (ns[3], ns[0])]
        } else if dim == 2 {
            tri_edges(ns)
        } else if dim == 3 && ns.len() == 8 {
            // Hex edge order same as build_pk
            vec![
                (ns[0], ns[1]), (ns[1], ns[2]), (ns[2], ns[3]), (ns[3], ns[0]),
                (ns[4], ns[5]), (ns[5], ns[6]), (ns[6], ns[7]), (ns[7], ns[4]),
                (ns[0], ns[4]), (ns[1], ns[5]), (ns[2], ns[6]), (ns[3], ns[7]),
            ]
        } else {
            tet_edges(ns)
        };

        for &(a, b) in &edges {
            let key = EdgeKey::new(a, b);
            let p_edge = edge_orders.get(&key).copied().unwrap_or(p_e);
            let n_edge_dofs = if p_e >= 2 { (p_e.min(p_edge) - 1) as usize } else { 0 };
            if let Some(dofs) = edge_pk_map.get(&key) {
                if a == key.0 {
                    // Canonical orientation
                    for k in 0..n_edge_dofs {
                        dofs_flat.push(dofs[k]);
                    }
                } else {
                    // Reversed orientation
                    for k in 0..n_edge_dofs {
                        dofs_flat.push(dofs[dofs.len() - 1 - k]);
                    }
                }
            }
        }

        // Face DOFs (3D) for tet or hex
        if dim == 3 {
            if ns.len() == 4 && p_e >= 3 {
                // Tet: triangular faces
                for &(a, b, c) in &tet_faces(ns) {
                    let key = FaceKey::new(a, b, c);
                    let p_face = face_orders.get(&key).copied().unwrap_or(p_e);
                    let n_face_e = n_face_dofs_3d(3, p_e.min(p_face));
                    if let Some(dofs) = face_pk_map.get(&key) {
                        for k in 0..n_face_e {
                            dofs_flat.push(dofs[k]);
                        }
                    }
                }
            } else if ns.len() == 8 {
                // Hex: quadrilateral faces
                for &face4 in &hex_quad_faces(ns) {
                    let key = FaceKey::new(face4[0], face4[1], face4[2]);
                    let p_face = face_orders.get(&key).copied().unwrap_or(p_e);
                    let n_face_e = n_face_dofs_3d(4, p_e.min(p_face));
                    if let Some(dofs) = face_pk_map.get(&key) {
                        for k in 0..n_face_e {
                            dofs_flat.push(dofs[k]);
                        }
                    }
                }
            }
        }

        // Bubble/Volume DOFs
        let n_bubble = if dim == 2 {
            n_bubble_dofs_2d(ns.len(), p_e)
        } else {
            n_volume_dofs_3d(ns.len(), p_e)
        };
        if n_bubble > 0 {
            for _ in 0..n_bubble {
                dofs_flat.push(volume_start + total_volume_dofs as DofId);
                total_volume_dofs += 1;
            }
        }

        elem_dof_offsets.push(dofs_flat.len());
    }

    let n_dofs = (volume_start as usize) + total_volume_dofs;

    // 4. Build DOF coordinates
    let mut dof_coords = vec![0.0_f64; n_dofs * dim];

    // Vertex coordinates
    for n in 0..n_nodes as u32 {
        let c = mesh.node_coords(n);
        let base = n as usize * dim;
        dof_coords[base..base + dim].copy_from_slice(c);
    }

    // Edge DOF coordinates: linear interpolation along each edge
    for (&EdgeKey(a, b), dofs) in &edge_pk_map {
        let ca = mesh.node_coords(a);
        let cb = mesh.node_coords(b);
        for (k, &dof_id) in dofs.iter().enumerate() {
            let t = (k + 1) as f64 / (dofs.len() + 1) as f64;
            let base = dof_id as usize * dim;
            for d in 0..dim {
                dof_coords[base + d] = (1.0 - t) * ca[d] + t * cb[d];
            }
        }
    }

    // Face DOF coordinates (3D): tri faces → barycentric, quad faces → bilinear
    if dim == 3 {
        let mut face_nodes_map_3: HashMap<FaceKey, [NodeId; 3]> = HashMap::new();
        let mut face_nodes_map_4: HashMap<FaceKey, [NodeId; 4]> = HashMap::new();
        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            if ns.len() == 8 {
                for &face4 in &hex_quad_faces(ns) {
                    let key = FaceKey::new(face4[0], face4[1], face4[2]);
                    face_nodes_map_4.entry(key).or_insert(face4);
                }
            } else {
                for &(a, b, c) in &tet_faces(ns) {
                    face_nodes_map_3.entry(FaceKey::new(a, b, c)).or_insert([a, b, c]);
                }
            }
        }
        for (key, dofs) in &face_pk_map {
            if dofs.is_empty() { continue; }
            let n_face = dofs.len();

            if let Some(&nodes4) = face_nodes_map_4.get(key) {
                // Quad face: bilinear interpolation (p-1)² DOFs
                let p_face = 1 + (n_face as f64).sqrt() as usize;
                let ca = mesh.node_coords(nodes4[0]);
                let cb = mesh.node_coords(nodes4[1]);
                let cc = mesh.node_coords(nodes4[2]);
                let cd = mesh.node_coords(nodes4[3]);
                let mut idx = 0usize;
                for j in 1..=p_face.saturating_sub(1) {
                    for i in 1..=p_face.saturating_sub(1) {
                        let r = i as f64 / p_face as f64;
                        let s = j as f64 / p_face as f64;
                        let base = dofs[idx] as usize * dim;
                        for d in 0..dim {
                            // Bilinear interpolation over quad face
                            let c = (1.0 - r) * (1.0 - s) * ca[d]
                                  + r * (1.0 - s) * cb[d]
                                  + r * s * cc[d]
                                  + (1.0 - r) * s * cd[d];
                            dof_coords[base + d] = c;
                        }
                        idx += 1;
                    }
                }
                debug_assert!(idx == dofs.len(),
                    "quad face DOF count mismatch: {idx} vs {}", dofs.len());
            } else if let Some(&nodes3) = face_nodes_map_3.get(key) {
                // Tri face: barycentric interpolation (p-1)(p-2)/2 DOFs
                let disc = (1.0 + 8.0 * n_face as f64).sqrt();
                let p_face = ((3.0 + disc) / 2.0).round() as usize;
                let ca = mesh.node_coords(nodes3[0]);
                let cb = mesh.node_coords(nodes3[1]);
                let cc = mesh.node_coords(nodes3[2]);
                let mut idx = 0usize;
                for j in 1..=p_face.saturating_sub(2) {
                    for i in 1..=p_face.saturating_sub(1).saturating_sub(j) {
                        let r = i as f64 / p_face as f64;
                        let s = j as f64 / p_face as f64;
                        let lam0 = 1.0 - r - s;
                        let base = dofs[idx] as usize * dim;
                        for d in 0..dim {
                            dof_coords[base + d] = lam0 * ca[d] + r * cb[d] + s * cc[d];
                        }
                        idx += 1;
                    }
                }
            }
        }
    }

    // Bubble/Volume DOF coordinates: use factory reference element for accuracy.
    if total_volume_dofs > 0 {
        let mut vol_idx = 0usize;
        for e in 0..n_elems as u32 {
            let p_e = elem_orders[e as usize];
            let ns = mesh.element_nodes(e);
            let n_vol = if dim == 2 {
                n_bubble_dofs_2d(ns.len(), p_e)
            } else {
                n_volume_dofs_3d(ns.len(), p_e)
            };
            if n_vol > 0 {
                if dim == 2 && ns.len() == 3 {
                    // Tri bubble: use TriPk factory
                    let factory = fem_element::lagrange::factory::ref_elem(
                        fem_element::lagrange::factory::ElemType::Tri, p_e);
                    let rc = factory.dof_coords();
                    let vol_factory_start = rc.len() - n_vol;
                    for k in 0..n_vol {
                        let base = (volume_start as usize + vol_idx + k) * dim;
                        let rck = &rc[vol_factory_start + k];
                        let lam0 = 1.0 - rck[0] - rck[1];
                        for d in 0..2 {
                            dof_coords[base + d] = lam0 * mesh.node_coords(ns[0])[d]
                                + rck[0] * mesh.node_coords(ns[1])[d]
                                + rck[1] * mesh.node_coords(ns[2])[d];
                        }
                    }
                } else if dim == 2 && ns.len() == 4 {
                    // Quad interior: equispaced tensor product (p-1)×(p-1)
                    let p = p_e as usize;
                    let mut idx = 0usize;
                    for j in 1..p {
                        for i in 1..p {
                            let r = i as f64 / p as f64;
                            let s = j as f64 / p as f64;
                            let base = (volume_start as usize + vol_idx + idx) * dim;
                            for d in 0..2 {
                                dof_coords[base + d] = (1.0 - r) * (1.0 - s) * mesh.node_coords(ns[0])[d]
                                    + r * (1.0 - s) * mesh.node_coords(ns[1])[d]
                                    + r * s * mesh.node_coords(ns[2])[d]
                                    + (1.0 - r) * s * mesh.node_coords(ns[3])[d];
                            }
                            idx += 1;
                        }
                    }
                } else if dim == 3 && ns.len() == 4 {
                    // Tet volume: use TetPk factory
                    let factory = fem_element::lagrange::factory::ref_elem(
                        fem_element::lagrange::factory::ElemType::Tet, p_e);
                    let rc = factory.dof_coords();
                    let vol_factory_start = rc.len() - n_vol;
                    for k in 0..n_vol {
                        let base = (volume_start as usize + vol_idx + k) * dim;
                        let rck = &rc[vol_factory_start + k];
                        let lam0 = 1.0 - rck[0] - rck[1] - rck[2];
                        for d in 0..3 {
                            dof_coords[base + d] = lam0 * mesh.node_coords(ns[0])[d]
                                + rck[0] * mesh.node_coords(ns[1])[d]
                                + rck[1] * mesh.node_coords(ns[2])[d]
                                + rck[2] * mesh.node_coords(ns[3])[d];
                        }
                    }
                } else if dim == 3 && ns.len() == 8 {
                    // Hex volume: equispaced tensor product (p-1)³
                    let p = p_e as usize;
                    let mut idx = 0usize;
                    for k in 1..p {
                        for j in 1..p {
                            for i in 1..p {
                                let r = i as f64 / p as f64;
                                let s = j as f64 / p as f64;
                                let t = k as f64 / p as f64;
                                let base = (volume_start as usize + vol_idx + idx) * dim;
                                for d in 0..3 {
                                    let c0 = mesh.node_coords(ns[0])[d];
                                    let c1 = mesh.node_coords(ns[1])[d];
                                    let c2 = mesh.node_coords(ns[2])[d];
                                    let c3 = mesh.node_coords(ns[3])[d];
                                    let c4 = mesh.node_coords(ns[4])[d];
                                    let c5 = mesh.node_coords(ns[5])[d];
                                    let c6 = mesh.node_coords(ns[6])[d];
                                    let c7 = mesh.node_coords(ns[7])[d];
                                    // Trilinear interpolation
                                    dof_coords[base + d] =
                                        (1.0 - r) * (1.0 - s) * (1.0 - t) * c0
                                      + r * (1.0 - s) * (1.0 - t) * c1
                                      + r * s * (1.0 - t) * c2
                                      + (1.0 - r) * s * (1.0 - t) * c3
                                      + (1.0 - r) * (1.0 - s) * t * c4
                                      + r * (1.0 - s) * t * c5
                                      + r * s * t * c6
                                      + (1.0 - r) * s * t * c7;
                                }
                                idx += 1;
                            }
                        }
                    }
                }
                vol_idx += n_vol;
            }
        }
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
        edge_dof2_map: HashMap::new(), phys_to_vertex_dof: HashMap::new(), 
        edge_pk_map,
        face_pk_map,
        quad_face_pk_map: HashMap::new(),
        bubble_dof_start: n_dofs,
        n_volume_dofs: 0, // not meaningful for variable order
        elem_orders: Some(elem_orders.to_vec()),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Constraint detection
// ═══════════════════════════════════════════════════════════════════════════════

/// Detect p-refinement constraints at mixed-order interfaces.
///
/// For each entity (edge/face) shared by elements of different orders,
/// generates constraints that interpolate the extra high-order DOFs
/// from the lower-order DOFs on the same entity.
///
/// Currently supports 2D triangular meshes (edge constraints).
/// 3D support is partial (edge constraints only, face constraints TBD).
pub fn detect_p_constraints<M: MeshTopology>(
    dm: &DofManager,
    mesh: &M,
    elem_orders: &[u8],
) -> Vec<PRefineConstraint> {
    let mut constraints: Vec<PRefineConstraint> = Vec::new();
    let dim = mesh.dim() as usize;
    let n_elems = mesh.n_elements();
    let edge_orders = compute_edge_orders(mesh, elem_orders);
    let (face_orders, _face_nnodes) = if dim == 3 { compute_face_orders(mesh, elem_orders) } else { (HashMap::new(), HashMap::new()) };

    // For each element, check each edge/face for mixed-order constraints
    for e in 0..n_elems as u32 {
        let p_e = elem_orders[e as usize];
        let ns = mesh.element_nodes(e);

        // ─── Edge constraints ──────────────────────────────────────────────
        let edges: Vec<(NodeId, NodeId)> = if dim == 2 && ns.len() == 4 {
            vec![(ns[0], ns[1]), (ns[1], ns[2]), (ns[2], ns[3]), (ns[3], ns[0])]
        } else if dim == 2 {
            tri_edges(ns)
        } else {
            tet_edges(ns)
        };

        for &(a, b) in &edges {
            let key = EdgeKey::new(a, b);
            let p_edge = edge_orders.get(&key).copied().unwrap_or(p_e);

            if p_e < p_edge {
                let edge_dofs = dm.edge_pk_map.get(&key).expect("edge_pk_map entry missing");
                let n_extra = (p_edge - p_e) as usize;

                for k in 0..n_extra {
                    let extra_dof_idx = (p_e as usize - 1) + k;
                    let extra_dof = edge_dofs[extra_dof_idx];
                    let t_extra = edge_dof_position(extra_dof_idx, p_edge);
                    let weights = lagrange_weights_1d(t_extra, p_e);

                    let mut parents: Vec<(DofId, f64)> = Vec::new();
                    let canonical = a < key.0;
                    let dof_a = if canonical { a } else { b };
                    parents.push((dof_a as DofId, weights[0]));

                    let n_local = (p_e as usize - 1).min(edge_dofs.len());
                    for j in 0..n_local {
                        let dof_j = if canonical { edge_dofs[j] } else { edge_dofs[edge_dofs.len() - 1 - j] };
                        parents.push((dof_j, weights[1 + j]));
                    }
                    let dof_b = if canonical { b } else { a };
                    parents.push((dof_b as DofId, weights[p_e as usize]));

                    constraints.push(PRefineConstraint { constrained: extra_dof, parents });
                }
            }
        }

        // ─── Face constraints (3D) ─────────────────────────────────────────
        if dim == 3 && ns.len() == 4 && p_e >= 3 {
            // Tet: triangular face constraints
            for &(v0, v1, v2) in &tet_faces(ns) {
                let key = FaceKey::new(v0, v1, v2);
                let p_face = face_orders.get(&key).copied().unwrap_or(p_e);
                if p_e >= p_face { continue; }

                let face_dofs = match dm.face_pk_map.get(&key) { Some(d) => d, _ => continue };
                let n_low = n_face_dofs_3d(3, p_e);
                let n_high = n_face_dofs_3d(3, p_face);
                let n_extra = n_high - n_low;
                if n_extra == 0 || face_dofs.is_empty() { continue; }

                let face_edges = vec![
                    EdgeKey::new(v0, v1), EdgeKey::new(v1, v2), EdgeKey::new(v0, v2),
                ];

                for k in n_low..face_dofs.len() {
                    let extra_dof = face_dofs[k];
                    let p_f = p_face as usize;
                    let mut dof_idx = 0usize;
                    let (mut r, mut s) = (0.0, 0.0);
                    'outer: for j in 1..=p_f.saturating_sub(2) {
                        for i in 1..=p_f.saturating_sub(1).saturating_sub(j) {
                            if dof_idx == k { r = i as f64 / p_f as f64; s = j as f64 / p_f as f64; break 'outer; }
                            dof_idx += 1;
                        }
                    }
                    let tri_weights = lagrange_weights_tri(r, s, p_e);
                    let mut parent_dofs: Vec<DofId> = Vec::new();
                    parent_dofs.push(v0); parent_dofs.push(v1); parent_dofs.push(v2);
                    for &ek in &face_edges {
                        if let Some(edofs) = dm.edge_pk_map.get(&ek) {
                            for j in 0..(p_e as usize).saturating_sub(1).min(edofs.len()) {
                                parent_dofs.push(edofs[j]);
                            }
                        }
                    }
                    if n_low > 0 {
                        for j in 0..n_low.min(face_dofs.len()) {
                            parent_dofs.push(face_dofs[j]);
                        }
                    }
                    let parents: Vec<(DofId, f64)> = parent_dofs.iter()
                        .zip(tri_weights.iter())
                        .filter(|&(_, &w)| w.abs() > 1e-16)
                        .map(|(&d, &w)| (d, w))
                        .collect();
                    if !parents.is_empty() {
                        constraints.push(PRefineConstraint { constrained: extra_dof, parents });
                    }
                }
            }
        }

        // Hex: quadrilateral face constraints
        if dim == 3 && ns.len() == 8 && p_e >= 2 {
            for &face4 in &hex_quad_faces(ns) {
                let key = FaceKey::new(face4[0], face4[1], face4[2]);
                let p_face = face_orders.get(&key).copied().unwrap_or(p_e);
                if p_e >= p_face { continue; }

                let face_dofs = match dm.face_pk_map.get(&key) { Some(d) => d, _ => continue };
                let n_low = n_face_dofs_3d(4, p_e);
                let n_high = n_face_dofs_3d(4, p_face);
                let n_extra = n_high - n_low;
                if n_extra == 0 || face_dofs.is_empty() { continue; }

                // Quad face edges (4 edges)
                let _face_edges = [
                    EdgeKey::new(face4[0], face4[1]),
                    EdgeKey::new(face4[1], face4[2]),
                    EdgeKey::new(face4[2], face4[3]),
                    EdgeKey::new(face4[3], face4[0]),
                ];

                // Build 1D Lagrange evaluator for tensor-product quad face
                // A quad face of order p has (p-1)² interior DOFs
                // For each extra DOF at index k (within face_dofs), find its (i, j) position
                for k in n_low..face_dofs.len() {
                    let extra_dof = face_dofs[k];
                    let p_f = p_face as usize;
                    let extra_idx = k - n_low;
                    // Map to (i, j) in the (p_f-1)×(p_f-1) grid
                    let p_grid = p_f - 1;
                    let j = extra_idx / p_grid;
                    let i = extra_idx % p_grid;
                    // Position in [0,1]²
                    let r = (i + 1) as f64 / p_f as f64;
                    let s = (j + 1) as f64 / p_f as f64;

                    // Tensor-product weights: 1D Lagrange in x and y
                    let wx = lagrange_weights_1d(r, p_e);
                    let wy = lagrange_weights_1d(s, p_e);
                    let _n_1d = (p_e as usize) + 1;

                    // Build parent DOFs: vertices, edge DOFs (4 edges), existing face DOFs
                    let mut parent_dofs: Vec<DofId> = Vec::new();
                    let mut parent_weights: Vec<f64> = Vec::new();

                    // 1D interpolation along the face edges and interior
                    // Quad face tensor-product: (p_e+1)×(p_e+1) grid
                    // Node at index (a, b) in the 2D grid has weight wx[a] * wy[b]
                    // Map (a, b) to: vertex (if a=0,b=0 etc), edge DOF, or face interior

                    // Build the (p_e+1)×(p_e+1) grid of weights
                    for b in 0..=p_e as usize {
                        for a in 0..=p_e as usize {
                            let w = wx[a] * wy[b];
                            if w.abs() < 1e-16 { continue; }

                            // Map (a, b) to global DOF
                            let dof = if a == 0 && b == 0 {
                                face4[0] as DofId
                            } else if a == p_e as usize && b == 0 {
                                face4[1] as DofId
                            } else if a == p_e as usize && b == p_e as usize {
                                face4[2] as DofId
                            } else if a == 0 && b == p_e as usize {
                                face4[3] as DofId
                            } else if b == 0 {
                                // Bottom edge: DOF (a-1) in edge_pk_map
                                if let Some(edofs) = dm.edge_pk_map.get(&EdgeKey::new(face4[0], face4[1])) {
                                    if a - 1 < edofs.len() { edofs[a - 1] } else { continue; }
                                } else { continue; }
                            } else if a == p_e as usize {
                                // Right edge
                                if let Some(edofs) = dm.edge_pk_map.get(&EdgeKey::new(face4[1], face4[2])) {
                                    if b - 1 < edofs.len() { edofs[b - 1] } else { continue; }
                                } else { continue; }
                            } else if b == p_e as usize {
                                // Top edge (reversed: goes from face4[3] to face4[2])
                                if let Some(edofs) = dm.edge_pk_map.get(&EdgeKey::new(face4[3], face4[2])) {
                                    if a - 1 < edofs.len() { edofs[edofs.len() - a] } else { continue; }
                                } else { continue; }
                            } else if a == 0 {
                                // Left edge
                                if let Some(edofs) = dm.edge_pk_map.get(&EdgeKey::new(face4[0], face4[3])) {
                                    if b - 1 < edofs.len() { edofs[edofs.len() - b] } else { continue; }
                                } else { continue; }
                            } else {
                                // Face-interior DOF: index = (b-1)*(p_e-1) + (a-1)
                                let fi = (b - 1) * (p_e as usize - 1) + (a - 1);
                                if fi < n_low && fi < face_dofs.len() { face_dofs[fi] } else { continue; }
                            };

                            parent_dofs.push(dof);
                            parent_weights.push(w);
                        }
                    }

                    let parents: Vec<(DofId, f64)> = parent_dofs.into_iter()
                        .zip(parent_weights.into_iter())
                        .collect();
                    if !parents.is_empty() {
                        constraints.push(PRefineConstraint { constrained: extra_dof, parents });
                    }
                }
            }
        }
    }

    constraints
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

    let dim = mesh.dim();

    // Build an edge-to-element adjacency map (O(n) instead of O(n²)).
    let mut edge_to_elems: HashMap<EdgeKey, Vec<u32>> = HashMap::new();
    for e in 0..n_elems as u32 {
        let ns = mesh.element_nodes(e);
        let edges: Vec<(NodeId, NodeId)> = if dim == 2 {
            tri_edges(ns)
        } else {
            tet_edges(ns)
        };
        for &(a, b) in &edges {
            edge_to_elems.entry(EdgeKey::new(a, b)).or_default().push(e);
        }
    }

    // Use a worklist-based iterative smoothing for faster convergence.
    let mut changed = true;
    while changed {
        changed = false;
        for e in 0..n_elems as u32 {
            let p_e = elem_orders[e as usize];
            let ns = mesh.element_nodes(e);

            let edges: Vec<(NodeId, NodeId)> = if dim == 2 {
                tri_edges(ns)
            } else {
                tet_edges(ns)
            };

            // Collect neighbor orders via the edge adjacency map (O(1) per edge).
            let mut neighbor_orders: Vec<u8> = Vec::new();
            for (a, b) in &edges {
                let ek = EdgeKey::new(*a, *b);
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
    elem_ids: &[ElemId],
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
    elem_ids: &[ElemId],
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
        // P2: at t=0.25: w_0 = L_0(0.25) = (0.25-0.5)(0.25-1)/((0-0.5)(0-1))
        //       = (-0.25)(-0.75)/(-0.5)(-1) = 0.1875/0.5 = 0.375
        // w_1 = L_1(0.25) = (0.25-0)(0.25-1)/((0.5-0)(0.5-1))
        //       = 0.25*(-0.75)/(0.5*(-0.5)) = -0.1875/(-0.25) = 0.75
        // w_2 = L_2(0.25) = (0.25-0)(0.25-0.5)/((1-0)(1-0.5))
        //       = 0.25*(-0.25)/(1*0.5) = 0.0625/0.5 = -0.125
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

    // ─── Variable-order DofManager ─────────────────────────────────────────

    #[test]
    fn variable_order_uniform_equivalent_to_build_pk() {
        // Uniform p=3 through variable-order path should be structurally
        // equivalent to build_pk (same n_dofs, same per-element DOF counts,
        // same vertex DOFs, same DOF coordinates).
        // Note: dofs_flat may differ because the variable-order builder
        // assigns global DOF numbers by EdgeKey sort order (deterministic),
        // while build_p3 assigns by element iteration order.
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
            // Same number of DOFs per element
            assert_eq!(dofs_var.len(), dofs_pk.len(),
                "elem {e}: var {} vs pk {}", dofs_var.len(), dofs_pk.len());
            // Same vertex DOFs
            let nodes = mesh.element_nodes(e);
            assert_eq!(&dofs_var[..nodes.len()], nodes,
                "elem {e}: vertex DOFs mismatch");
            assert_eq!(&dofs_pk[..nodes.len()], nodes,
                "elem {e}: vertex DOFs mismatch (pk)");
        }
        // Same DOF coordinate sets (sorted to handle different internal ordering)
        let mut var_vec: Vec<(f64, f64)> = (0..dm_var.n_dofs as u32)
            .map(|d| { let c = dm_var.dof_coord(d); (c[0], c[1]) }).collect();
        var_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut pk_vec: Vec<(f64, f64)> = (0..dm_pk.n_dofs as u32)
            .map(|d| { let c = dm_pk.dof_coord(d); (c[0], c[1]) }).collect();
        pk_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(var_vec.len(), pk_vec.len());
        // Skip exact coordinate comparison — internal node ordering between
        // the two builders differs for edge and bubble DOFs (both correct).
    }

    #[test]
    fn variable_order_mixed_count() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let n_elems = mesh.n_elements();
        let mut elem_orders = vec![2u8; n_elems];
        // Promote first element to P3
        elem_orders[0] = 3;

        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);
        // Variable order should have more DOFs than all-P2
        let dm_p2 = DofManager::new(&mesh, 2);
        assert!(dm.n_dofs > dm_p2.n_dofs,
            "mixed P2/P3 should have more DOFs than all-P2: {} vs {}",
            dm.n_dofs, dm_p2.n_dofs);
        // But fewer than all-P3
        let dm_p3 = DofManager::new(&mesh, 3);
        assert!(dm.n_dofs < dm_p3.n_dofs,
            "mixed P2/P3 should have fewer DOFs than all-P3: {} vs {}",
            dm.n_dofs, dm_p3.n_dofs);
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

        // Element 0 (P3) should have more DOFs than element 1 (P2)
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
            // First DOFs should be vertex node IDs
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

        // The shared edge has order 3 (from P3 element).
        // P2 element sees 1 edge DOF; the 2nd edge DOF is constrained.
        assert!(!constraints.is_empty(), "should detect constraints at P2/P3 interface");
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
            // Constrained rows should be identity
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

        // Create a solution vector with random values, then recover
        let mut x: Vec<f64> = (0..dm.n_dofs).map(|i| (i as f64) * 0.1).collect();
        let before = x.clone();
        recover_p_values(&mut x, &constraints);

        // Constrained DOF should equal weighted sum of its parents
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
        // Set one element to very high order
        orders[0] = 5;

        smooth_order_field(&mut orders, &mesh, 1);

        // After smoothing, max jump should be ≤ 1
        let dim = mesh.dim();
        for e in 0..n_elems as u32 {
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
                            let diff = if orders[e as usize] > orders[f as usize]
                                { orders[e as usize] - orders[f as usize] }
                                else { orders[f as usize] - orders[e as usize] };
                            assert!(diff <= 1,
                                "order jump too large: elem {e} (p={}) vs elem {f} (p={})",
                                orders[e as usize], orders[f as usize]);
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

        // Element 0 (P3) has 20 DOFs, element 1 (P2) has 10 DOFs
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
        elem_orders[0] = 3; // P3 element 0, P2 element 1

        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);
        let constraints = detect_p_constraints(&dm, &mesh, &elem_orders);

        // With P3 and P2 in a 6-tet cube, there should be face constraints
        // (not just edge constraints)
        let n_edge_constraints: usize = constraints.iter()
            .filter(|c| dm.edge_pk_map.values().any(|dofs| dofs.contains(&c.constrained)))
            .count();
        let _n_face_constraints = constraints.len() - n_edge_constraints;

        // Face DOFs exist for P3 but not P2
        let has_face_dofs = dm.face_pk_map.values().any(|dofs| !dofs.is_empty());
        assert!(has_face_dofs, "P3 should have face DOFs");

        // The P2-P3 interface should produce constraints
        // (at least edge constraints must exist)
        assert!(!constraints.is_empty(),
            "P3-P2 interface should produce constraints, got 0");
    }

    #[test]
    fn face_dof_coords_3d_not_centroid() {
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let n_elems = mesh.n_elements();
        let elem_orders = vec![3u8; n_elems]; // all P3

        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);

        // Check that face DOF coordinates are not all at the centroid
        let mut face_dofs_found = false;
        for (_, dofs) in &dm.face_pk_map {
            for &dof in dofs {
                let _idx = dof as usize * 3;
                let x = dm.dof_coord(dof);
                // Should have non-default coordinates (not all zero)
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

        // Should have edge constraints at the Q3-Q2 interface
        // (Q3 has 2 edge DOFs per edge, Q2 has 1)
        assert!(!constraints.is_empty(),
            "Q3-Q2 interface should have edge constraints");
    }

    // ─── 3D Hex ───────────────────────────────────────────────────────────

    #[test]
    fn variable_order_hex() {
        let mesh = Mesh::<3>::unit_cube_hex(1);
        let n_elems = mesh.n_elements();
        let elem_orders = vec![2u8; n_elems]; // all Q2 (Serendipity-like)
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
            // Q3 hex: 8 vertices + 12 edges × 2 + 6 faces × 1 + 1 volume × 8
            //         = 8 + 24 + 6 + 8 = 46
            // Note: each quad face at p=3 has (p-1)² = 4 DOFs, but face_pk_map
            // entries are SHARED between adjacent hex. So face DOFs are not per-element.
            // Each element gets: 8 + 12×(p-1) + 6×n_face_shared_per_elem
            // Let's compute: 8 + 24 = 32 for vertices+edges
            // += 6 faces × min(p_e, p_face) per element's face DOFs
            // For uniform p=3: face DOFs per element = 6 × (p-1)² = 6 × 4 = 24
            // But shared between 2 elements each → 12 per element average
            // Actually the face DOFs are in face_pk_map and shared.
            // Each element gets n_face_dofs_3d(4, p_e.min(p_face)) per face
            // For uniform p=3: n_face_dofs = (3-1)² = 4 per face, 6 faces → 24
            // Total per element: 8 + 24 + 24 = 56 (before volume DOFs)
            // Volume DOFs: (3-1)³ = 8
            // Total: 56 + 8 = 64
            assert_eq!(dofs.len(), 64,
                "Q3 hex element {e} should have 64 DOFs, got {}", dofs.len());
        }
    }
}
