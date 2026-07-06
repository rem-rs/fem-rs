//! XFEM (eXtended Finite Element Method) infrastructure.
//!
//! Provides enrichment management and assembly for problems with
//! strong discontinuities (cracks) using level-set tracking.
//!
//! # Organization
//! - `xfem_level_set` — signed-distance functions and sub-cell triangulation
//! - This module — enrichment detection, DOF management, assembly

use fem_core::types::DofId;
use fem_mesh::topology::MeshTopology;

use super::xfem_level_set::{cut_triangle, CutResult};
pub use super::xfem_level_set::XfemLevelSet;

/// Type of enrichment applied to a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnrichmentType {
    /// Heaviside (step) enrichment: extra DOF = H(x) · N(x)
    Heaviside,
    /// Crack-tip enrichment: 4 extra DOFs with branch functions F₁…F₄
    TipBranch,
}

/// Describes the enrichment configuration for a problem.
#[derive(Debug, Clone)]
pub struct XfemEnrichment {
    /// For each enriched node: (node_index, enrichment_type)
    pub enriched_nodes: Vec<(usize, EnrichmentType)>,
    /// For each enriched node, the global DOF indices of its enrichment DOFs.
    /// Heaviside: 1 extra DOF per enriched node.
    /// TipBranch: 4 extra DOFs per enriched node.
    pub enrichment_dofs: Vec<Vec<DofId>>,
    /// Total number of standard DOFs
    pub n_std_dofs: usize,
    /// Total number of enrichment DOFs
    pub n_enr_dofs: usize,
}

impl XfemEnrichment {
    /// Total system size: standard + enrichment DOFs.
    pub fn n_total_dofs(&self) -> usize {
        self.n_std_dofs + self.n_enr_dofs
    }
}

/// Detect which nodes should be enriched based on a level set.
///
/// For a crack line:
/// - Nodes of elements intersected by the crack → Heaviside enrichment
/// - Nodes within `tip_radius` of the crack tips → TipBranch enrichment
///
/// `dofs_per_node` specifies how many DOFs each node has in the
/// standard FE space (1 for scalar, 2 for elasticity, etc.).
pub fn detect_enriched_nodes<M: MeshTopology>(
    mesh: &M,
    ls: &XfemLevelSet,
    tip_radius: f64,
    dofs_per_node: usize,
) -> XfemEnrichment {
    let ne = mesh.n_elements() as u32;
    let nn = mesh.n_nodes();
    let mut is_enriched_heaviside = vec![false; nn];
    let mut is_enriched_tip = vec![false; nn];
    let mut tip_positions: Vec<[f64; 2]> = Vec::new();

    // Extract crack tip positions (endpoints of the CrackLine)
    if let XfemLevelSet::CrackLine { x1, x2 } = ls {
        tip_positions.push(*x1);
        tip_positions.push(*x2);
    }

    // Pass 1: find elements cut by the crack → enrich their nodes
    for e in 0..ne {
        let nodes = mesh.element_nodes(e);
        if nodes.len() < 3 { continue; }
        let phys: Vec<[f64; 2]> = nodes.iter().map(|&n| {
            let c = mesh.node_coords(n);
            [c[0], c[1]]
        }).collect();

        let tri_nodes: [[f64; 2]; 3] = [phys[0], phys[1], phys[2]];
        if let CutResult::Cut(_) = cut_triangle(ls, &tri_nodes) {
            for &n in nodes {
                is_enriched_heaviside[n as usize] = true;
            }
        }
    }

    // Pass 2: find nodes within tip_radius of any crack tip → TipBranch
    if !tip_positions.is_empty() {
        for n in 0..nn as u32 {
            let c = mesh.node_coords(n);
            let x = [c[0], c[1]];
            for &tip in &tip_positions {
                let dx = x[0] - tip[0];
                let dy = x[1] - tip[1];
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < tip_radius {
                    is_enriched_tip[n as usize] = true;
                    break;
                }
            }
        }
    }

    // Build enrichment list
    let n_std_dofs = nn * dofs_per_node; // For P1 H1, each node = `dofs_per_node` DOFs
    let mut enriched_nodes = Vec::new();
    let mut enrichment_dofs = Vec::new();
    let mut next_enr_dof = n_std_dofs as DofId;

    for n in 0..nn {
        let mut has_heaviside = false;
        let mut has_tip = false;

        if is_enriched_tip[n] {
            has_tip = true;
        }
        if is_enriched_heaviside[n] && !is_enriched_tip[n] {
            has_heaviside = true;
        }

        if has_tip {
            enriched_nodes.push((n, EnrichmentType::TipBranch));
            let n_enr_dofs = 4 * dofs_per_node; // 4 branch functions × components
            let dofs: Vec<DofId> = (0..n_enr_dofs).map(|_| { let d = next_enr_dof; next_enr_dof += 1; d }).collect();
            enrichment_dofs.push(dofs);
        } else if has_heaviside {
            enriched_nodes.push((n, EnrichmentType::Heaviside));
            let dofs: Vec<DofId> = (0..dofs_per_node).map(|_| { let d = next_enr_dof; next_enr_dof += 1; d }).collect();
            enrichment_dofs.push(dofs);
        }
    }

    let n_enr_dofs = (next_enr_dof - n_std_dofs as DofId) as usize;
    XfemEnrichment { enriched_nodes, enrichment_dofs, n_std_dofs, n_enr_dofs }
}

/// Map from standard DOF to its enrichment DOF(s), if any.
pub struct EnrichmentMap {
    /// Indexed by standard DOF (node index): list of enrichment global DOF indices
    pub enr_dofs: Vec<Vec<usize>>,
    /// Type of enrichment for each standard DOF (node)
    pub enr_type: Vec<Option<EnrichmentType>>,
    /// Total number of DOFs (standard + enrichment)
    pub n_total: usize,
}

impl EnrichmentMap {
    pub fn from_enrichment(e: &XfemEnrichment) -> Self {
        let mut enr_dofs = vec![Vec::new(); e.n_std_dofs];
        let mut enr_type = vec![None; e.n_std_dofs];
        for (i, &(node, etype)) in e.enriched_nodes.iter().enumerate() {
            enr_dofs[node] = e.enrichment_dofs[i].iter().map(|&d| d as usize).collect();
            enr_type[node] = Some(etype);
        }
        EnrichmentMap { enr_dofs, enr_type, n_total: e.n_total_dofs() }
    }
}

/// Evaluate Heaviside function: H(x) = +1 if ψ(x) > 0, -1 if ψ(x) < 0.
pub fn heaviside(ls: &XfemLevelSet, x: [f64; 2]) -> f64 {
    if ls.eval(x) >= 0.0 { 1.0 } else { -1.0 }
}

/// Evaluate the 4 crack-tip branch functions at (r, θ) in the
/// crack-tip polar coordinate system.
///
/// F₁ = √r · sin(θ/2)   — displacement discontinuity across crack
/// F₂ = √r · cos(θ/2)
/// F₃ = √r · sin(θ/2)·sin(θ)
/// F₄ = √r · cos(θ/2)·sin(θ)
pub fn tip_branch_functions(r: f64, theta: f64) -> [f64; 4] {
    let sqrt_r = r.sqrt().max(0.0);
    let sin_half = (theta / 2.0).sin();
    let cos_half = (theta / 2.0).cos();
    let sin_theta = theta.sin();
    [
        sqrt_r * sin_half,
        sqrt_r * cos_half,
        sqrt_r * sin_half * sin_theta,
        sqrt_r * cos_half * sin_theta,
    ]
}

/// Compute polar coordinates (r, θ) of point `x` relative to crack tip `tip`,
/// with the crack extending in direction `crack_dir` (unit vector).
pub fn polar_coords(x: [f64; 2], tip: [f64; 2], crack_dir: [f64; 2]) -> (f64, f64) {
    let dx = x[0] - tip[0];
    let dy = x[1] - tip[1];
    let r = (dx * dx + dy * dy).sqrt();
    // θ measured from crack_dir, CCW positive
    let theta = (dx * crack_dir[1] - dy * crack_dir[0]).atan2(dx * crack_dir[0] + dy * crack_dir[1]);
    (r, theta)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn heaviside_sign() {
        let ls = XfemLevelSet::Halfspace { normal: [1.0, 0.0], offset: 0.5 };
        assert!((heaviside(&ls, [1.0, 0.0]) - 1.0).abs() < 1e-14);
        assert!((heaviside(&ls, [0.0, 0.0]) - (-1.0)).abs() < 1e-14);
    }

    #[test]
    fn tip_branch_values() {
        let f = tip_branch_functions(1.0, 0.0);
        assert!((f[0] - 0.0).abs() < 1e-14, "F₁ at θ=0 should be 0"); // √1·sin(0)=0
        assert!((f[1] - 1.0).abs() < 1e-14, "F₂ at θ=0 should be 1"); // √1·cos(0)=1
    }

    #[test]
    fn detect_crack_enrichment() {
        // Unit square with horizontal crack at y=0.5, from x=0 to x=0.5
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let crack_ls = XfemLevelSet::CrackLine {
            x1: [0.0, 0.5],
            x2: [0.5, 0.5],
        };
        let enr = detect_enriched_nodes(&mesh, &crack_ls, 0.2, 1);
        // Nodes near the crack should be enriched
        assert!(enr.enriched_nodes.len() > 0, "should have enriched nodes");
        // All enrichment DOFs should be past the standard ones
        for edofs in &enr.enrichment_dofs {
            for &d in edofs {
                assert!(d >= enr.n_std_dofs as u32, "enriched DOF should be past standard DOFs");
            }
        }
        eprintln!("Standard DOFs: {}, Enriched DOFs: {}, Total: {}",
            enr.n_std_dofs, enr.n_enr_dofs, enr.n_total_dofs());
        assert!(enr.n_enr_dofs > 0, "should have enrichment DOFs");
    }
}
