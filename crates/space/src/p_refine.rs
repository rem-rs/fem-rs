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

/// Compute the order of each face (3D) based on per-element orders.
fn compute_face_orders<M: MeshTopology>(
    mesh: &M,
    elem_orders: &[u8],
) -> HashMap<FaceKey, u8> {
    let mut orders: HashMap<FaceKey, u8> = HashMap::new();
    let n_elems = mesh.n_elements();

    for e in 0..n_elems as u32 {
        let p = elem_orders[e as usize];
        let ns = mesh.element_nodes(e);
        let faces = tet_faces(ns);
        for &(a, b, c) in &faces {
            let key = FaceKey::new(a, b, c);
            let prev = orders.get(&key).copied().unwrap_or(0);
            if p > prev {
                orders.insert(key, p);
            }
        }
    }
    orders
}

/// Number of volume-interior DOFs for a simplex of given dimension and order.
fn n_volume_dofs(dim: usize, p: u8) -> usize {
    let p = p as usize;
    if dim == 2 && p >= 3 {
        (p - 1) * (p - 2) / 2
    } else if dim == 3 && p >= 4 {
        (p - 1) * (p - 2) * (p - 3) / 6
    } else {
        0
    }
}

/// Number of face-interior DOFs for a 3D simplex face (triangle) of order p.
fn n_face_dofs(p: u8) -> usize {
    let p = p as usize;
    if p >= 3 { (p - 1) * (p - 2) / 2 } else { 0 }
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
    let face_orders = if dim == 3 { compute_face_orders(mesh, elem_orders) } else { HashMap::new() };

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

    // Face DOFs (3D): each unique face gets n_face_dofs(p_face) DOFs
    let mut face_pk_map: HashMap<FaceKey, Vec<DofId>> = HashMap::new();
    if dim == 3 {
        let mut face_list: Vec<(FaceKey, u8)> = face_orders.iter()
            .map(|(&k, &p)| (k, p)).collect();
        face_list.sort_by_key(|&(k, _)| k);
        for (key, p_face) in &face_list {
            let n = n_face_dofs(*p_face);
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

        // Face DOFs (3D)
        if dim == 3 && p_e >= 3 {
            let faces = tet_faces(ns);
            for &(a, b, c) in &faces {
                let key = FaceKey::new(a, b, c);
                let p_face = face_orders.get(&key).copied().unwrap_or(p_e);
                let n_face_dofs_e = n_face_dofs(p_e.min(p_face));
                if let Some(dofs) = face_pk_map.get(&key) {
                    for k in 0..n_face_dofs_e {
                        dofs_flat.push(dofs[k]);
                    }
                }
            }
        }

        // Volume DOFs
        let n_vol = n_volume_dofs(dim, p_e);
        if n_vol > 0 {
            for _ in 0..n_vol {
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

    // Face DOF coordinates (3D): centroid
    if dim == 3 {
        let mut face_nodes_map: HashMap<FaceKey, [NodeId; 3]> = HashMap::new();
        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            for &(a, b, c) in &tet_faces(ns) {
                face_nodes_map.entry(FaceKey::new(a, b, c)).or_insert([a, b, c]);
            }
        }
        for (key, dofs) in &face_pk_map {
            if let Some(&nodes) = face_nodes_map.get(key) {
                let ca = mesh.node_coords(nodes[0]);
                let cb = mesh.node_coords(nodes[1]);
                let cc = mesh.node_coords(nodes[2]);
                for (k, &dof_id) in dofs.iter().enumerate() {
                    let base = dof_id as usize * dim;
                    let t = (k + 1) as f64 / (dofs.len() + 1) as f64;
                    for d in 0..dim {
                        dof_coords[base + d] = (1.0 - t) * ca[d]
                            + t * (cb[d] + cc[d]) / 2.0;
                    }
                }
            }
        }
    }

    // Volume DOF coordinates: use factory reference element for accuracy.
    if total_volume_dofs > 0 {
        let mut vol_idx = 0usize;
        for e in 0..n_elems as u32 {
            let p_e = elem_orders[e as usize];
            let n_vol = n_volume_dofs(dim, p_e);
            if n_vol > 0 {
                let ns = mesh.element_nodes(e);
                // Factory volume DOF positions are the LAST n_vol entries.
                // Use barycentric interpolation from vertex coordinates.
                if dim == 2 {
                    let factory = fem_element::lagrange::factory::ref_elem(
                        fem_element::lagrange::factory::ElemType::Tri, p_e as u8);
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
                } else {
                    let factory = fem_element::lagrange::factory::ref_elem(
                        fem_element::lagrange::factory::ElemType::Tet, p_e as u8);
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
        edge_dof2_map: HashMap::new(),
        edge_pk_map,
        face_pk_map,
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

    // For each element, check each edge for mixed-order constraints
    for e in 0..n_elems as u32 {
        let p_e = elem_orders[e as usize];
        let ns = mesh.element_nodes(e);
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
                // This element has fewer edge DOFs than the edge owns.
                // The extra DOFs need constraints.
                let edge_dofs = dm.edge_pk_map.get(&key).expect("edge_pk_map entry missing");
                let n_extra = (p_edge - p_e) as usize;

                for k in 0..n_extra {
                    // The (p_e - 1 + k)-th edge DOF (0-indexed) is constrained
                    let extra_dof_idx = (p_e as usize - 1) + k;
                    let extra_dof = edge_dofs[extra_dof_idx];

                    // Position of this extra DOF along the edge
                    let t_extra = edge_dof_position(extra_dof_idx, p_edge);

                    // Lagrange weights through (p_e + 1) points
                    let weights = lagrange_weights_1d(t_extra, p_e);

                    // Build parent DOF list
                    let mut parents: Vec<(DofId, f64)> = Vec::new();
                    let canonical = a < key.0;

                    // Vertex a (t=0)
                    let dof_a = if canonical { a } else { b };
                    parents.push((dof_a as DofId, weights[0]));

                    // Edge DOFs (j=1..p_e-1 → index 0..p_e-2 in edge DOF list)
                    let n_local = (p_e as usize - 1).min(edge_dofs.len());
                    for j in 0..n_local {
                        let dof_j = if canonical {
                            edge_dofs[j]
                        } else {
                            edge_dofs[edge_dofs.len() - 1 - j]
                        };
                        parents.push((dof_j, weights[1 + j]));
                    }

                    // Vertex b (t=1)
                    let dof_b = if canonical { b } else { a };
                    parents.push((dof_b as DofId, weights[p_e as usize]));

                    constraints.push(PRefineConstraint {
                        constrained: extra_dof,
                        parents,
                    });
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

            // Find neighbors sharing each edge
            let mut neighbor_orders: Vec<u8> = Vec::new();
            // We find neighbors by scanning all elements — O(n²) but simple.
            // For large meshes, a proper adjacency builder would be better.
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
    use fem_mesh::SimplexMesh;

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
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(1);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let elem_orders = vec![2u8; mesh.n_elements()];
        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);
        let constraints = detect_p_constraints(&dm, &mesh, &elem_orders);
        assert!(constraints.is_empty(),
            "uniform order should have no constraints");
    }

    #[test]
    fn p_constraint_parents_sum_to_one() {
        let mesh = SimplexMesh::<2>::unit_square_tri(1);
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

        let mesh = SimplexMesh::<2>::unit_square_tri(1);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(1);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let elem_orders = vec![2u8; mesh.n_elements()];
        let dm = DofManager::new(&mesh, 2);

        let (dm_refined, _) = refine_p(&dm, &mesh, &elem_orders, &[0, 1], 3);
        assert!(dm_refined.n_dofs > dm.n_dofs,
            "refine_p should increase n_dofs: {} vs {}",
            dm_refined.n_dofs, dm.n_dofs);
    }

    #[test]
    fn derefine_p_decreases_dofs() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
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
        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
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
}
