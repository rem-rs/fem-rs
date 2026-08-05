use std::collections::HashMap;

use fem_core::types::DofId;
use fem_linalg::CsrMatrix;
use fem_mesh::amr::HangingNodeConstraint;

use crate::dof_manager::DofManager;

use super::hanging_2d::apply_hanging_constraints;

/// Identify pairs of DOFs that should be identified via periodic boundary conditions.
///
/// Given a `master_tag` boundary and a `slave_tag` boundary, finds pairs
/// `(slave_dof, master_dof)` such that `slave_coord + offset ≈ master_coord`.
///
/// Works for P1, P2, and P3 spaces:
/// - P1: vertex DOFs only
/// - P2: vertex DOFs + edge-midpoint DOFs (matched by pair of vertex pairs)
/// - P3: vertex DOFs + 2 edge DOFs per boundary edge + bubble DOFs are interior (skipped)
///
/// # Arguments
/// * `mesh`       — provides boundary face/node data
/// * `dm`         — DOF manager for the space
/// * `master_tag` — boundary tag of the "master" side
/// * `slave_tag`  — boundary tag of the "slave" side
/// * `offset`     — vector such that `x_slave + offset ≈ x_master`
/// * `tol`        — coordinate matching tolerance
///
/// # Returns
/// Sorted list of `(slave_dof, master_dof)` pairs.
pub fn identify_periodic_dof_pairs(
    mesh:       &dyn fem_mesh::topology::MeshTopology,
    dm:         &DofManager,
    master_tag: i32,
    slave_tag:  i32,
    offset:     &[f64],
    tol:        f64,
) -> Vec<(DofId, DofId)> {
    let dim = dm.dof_coord(0).len();

    // Collect master boundary nodes with their coordinates.
    let mut master_nodes: HashMap<u32, Vec<f64>> = HashMap::new();
    for f in 0..mesh.n_boundary_faces() as u32 {
        if mesh.face_tag(f) == master_tag {
            for &node in mesh.face_nodes(f) {
                let coords = mesh.node_coords(node).to_vec();
                master_nodes.insert(node, coords);
            }
        }
    }

    // For P1, pairs are just vertex node DOFs (node index == DOF index for P1).
    // Collect slave nodes and match to master by x_slave + offset ≈ x_master.
    let mut pairs: Vec<(DofId, DofId)> = Vec::new();

    // Map: master_node -> master_dof (for P1, node == dof).
    // For higher orders, we look up via dm.
    let find_master_dof = |master_node: u32| -> DofId { master_node as DofId };

    // Match slave vertex nodes to master vertex nodes.
    let mut slave_node_to_master_node: HashMap<u32, u32> = HashMap::new();

    for f in 0..mesh.n_boundary_faces() as u32 {
        if mesh.face_tag(f) == slave_tag {
            for &slave_node in mesh.face_nodes(f) {
                let sc = mesh.node_coords(slave_node);
                // shifted coordinates
                let shifted: Vec<f64> = (0..dim).map(|i| sc[i] + offset[i]).collect();

                // Find matching master node
                let mut best: Option<(u32, f64)> = None;
                for (&mn, mc) in &master_nodes {
                    let dist: f64 = (0..dim).map(|i| (shifted[i] - mc[i]).powi(2)).sum::<f64>().sqrt();
                    if dist < tol
                        && best.is_none_or(|(_, d)| dist < d) {
                            best = Some((mn, dist));
                        }
                }

                if let Some((master_node, _)) = best {
                    slave_node_to_master_node.insert(slave_node, master_node);
                    let slave_dof = slave_node as DofId;
                    let master_dof = find_master_dof(master_node);
                    if slave_dof != master_dof {
                        pairs.push((slave_dof, master_dof));
                    }
                }
            }
        }
    }

    // For P2: also match edge-midpoint DOFs.
    // An edge midpoint DOF on the slave side is matched to the edge midpoint DOF
    // on the master side where both endpoints of the slave edge are matched.
    if dm.order == 2 {
        let n_elems = dm.dofs_flat.len() / dm.dofs_per_elem;
        // Build set of slave boundary edges and master boundary edges.
        let mut slave_edges: HashMap<(u32, u32), DofId> = HashMap::new();
        let mut master_edges: HashMap<(u32, u32), DofId> = HashMap::new();

        for e in 0..n_elems as u32 {
            let dofs  = dm.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            let edge_list = [
                (nodes[0], nodes[1], dofs[3]),
                (nodes[1], nodes[2], dofs[4]),
                (nodes[0], nodes[2], dofs[5]),
            ];
            for (a, b, edge_dof) in edge_list {
                let key = if a < b { (a, b) } else { (b, a) };
                // Check if both nodes are on slave boundary
                let a_slave = slave_node_to_master_node.contains_key(&a);
                let b_slave = slave_node_to_master_node.contains_key(&b);
                if a_slave && b_slave {
                    slave_edges.insert(key, edge_dof);
                }
                // Check if both nodes are on master boundary
                let a_master = master_nodes.contains_key(&a);
                let b_master = master_nodes.contains_key(&b);
                if a_master && b_master {
                    master_edges.insert(key, edge_dof);
                }
            }
        }

        // Match slave edge to master edge: the master edge has endpoints
        // that correspond to the master nodes matched to the slave edge's endpoints.
        for ((sa, sb), slave_dof) in &slave_edges {
            let ma = slave_node_to_master_node.get(sa);
            let mb = slave_node_to_master_node.get(sb);
            if let (Some(&ma), Some(&mb)) = (ma, mb) {
                let master_key = if ma < mb { (ma, mb) } else { (mb, ma) };
                if let Some(&master_dof) = master_edges.get(&master_key) {
                    if *slave_dof != master_dof {
                        pairs.push((*slave_dof, master_dof));
                    }
                }
            }
        }
    }

    // For P3: match the 2 edge interior DOFs per boundary edge.
    if dm.order == 3 {
        let n_elems = dm.dofs_flat.len() / dm.dofs_per_elem;
        let mut slave_edges: HashMap<(u32, u32), [DofId; 2]> = HashMap::new();
        let mut master_edges: HashMap<(u32, u32), [DofId; 2]> = HashMap::new();

        for e in 0..n_elems as u32 {
            let dofs  = dm.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            // [near_a, near_b] for edge a→b in element order
            let edge_list: [(u32, u32, [DofId; 2]); 3] = [
                (nodes[0], nodes[1], [dofs[3], dofs[4]]),
                (nodes[1], nodes[2], [dofs[5], dofs[6]]),
                (nodes[0], nodes[2], [dofs[7], dofs[8]]),
            ];
            for (a, b, edge_dofs) in edge_list {
                let (key, canonical_dofs) = if a < b {
                    ((a, b), edge_dofs)   // [near_a, near_b]
                } else {
                    ((b, a), [edge_dofs[1], edge_dofs[0]])  // flip to canonical order [near_min, near_max]
                };
                let a_slave = slave_node_to_master_node.contains_key(&a);
                let b_slave = slave_node_to_master_node.contains_key(&b);
                if a_slave && b_slave {
                    slave_edges.insert(key, canonical_dofs);
                }
                let a_master = master_nodes.contains_key(&a);
                let b_master = master_nodes.contains_key(&b);
                if a_master && b_master {
                    master_edges.insert(key, canonical_dofs);
                }
            }
        }

        // Match slave P3 edge DOFs to master P3 edge DOFs.
        // The 1/3 point near slave_min matches the 1/3 point near master_min.
        for ((sa, sb), slave_dofs) in &slave_edges {
            let ma = slave_node_to_master_node.get(sa);
            let mb = slave_node_to_master_node.get(sb);
            if let (Some(&ma), Some(&mb)) = (ma, mb) {
                let master_key = if ma < mb { (ma, mb) } else { (mb, ma) };
                if let Some(&master_dofs) = master_edges.get(&master_key) {
                    // slave canonical [near_sa, near_sb] matches master canonical [near_ma, near_mb]
                    // We need to check if the mapping preserves orientation:
                    // If sa→ma and sb→mb in the same "increasing" direction, dofs match directly.
                    // If sa→mb and sb→ma (orientation flip), dofs are swapped.
                    let master_near_sa = if ma < mb { master_dofs[0] } else { master_dofs[1] };
                    let master_near_sb = if ma < mb { master_dofs[1] } else { master_dofs[0] };
                    if slave_dofs[0] != master_near_sa {
                        pairs.push((slave_dofs[0], master_near_sa));
                    }
                    if slave_dofs[1] != master_near_sb {
                        pairs.push((slave_dofs[1], master_near_sb));
                    }
                }
            }
        }
    }

    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// Apply periodic boundary conditions to the assembled system `(K, f)`.
///
/// Converts each `(slave_dof, master_dof)` pair into a
/// `HangingNodeConstraint { constrained: slave, parent_a: master, parent_b: master }`
/// (degenerate: both parents are the same, giving `u_slave = master`).
///
/// Delegates to [`apply_hanging_constraints`] for the actual constraint application.
pub fn apply_periodic(
    mat:   &mut CsrMatrix<f64>,
    rhs:   &mut [f64],
    pairs: &[(DofId, DofId)],
) {
    let constraints: Vec<HangingNodeConstraint> = pairs.iter()
        .map(|&(slave, master)| HangingNodeConstraint::new_weighted(
            slave as usize, master as usize, master as usize, 0.5, 0.5, vec![],
        ))
        .collect();
    apply_hanging_constraints(mat, rhs, &constraints);
}
