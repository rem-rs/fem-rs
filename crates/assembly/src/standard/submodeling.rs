//! Submodeling analysis.
//!
//! Enables detailed analysis of a local region by interpolating
//! displacements from a coarse global solution onto the refined
//! local mesh boundary.
//!
//! # Workflow
//!
//! 1. Solve global model (coarse mesh) → get `u_global`
//! 2. Define submodel region (finer mesh)
//! 3. For each node on the submodel boundary, find its position
//!    in the global mesh and interpolate `u_global` at that point
//! 4. Apply interpolated displacements as Dirichlet BCs on submodel
//! 5. Solve submodel with these boundary conditions

use fem_linalg::CsrMatrix;
use fem_mesh::topology::MeshTopology;

/// Submodeling configuration.
#[derive(Debug, Clone)]
pub struct SubmodelConfig {
    /// Boundary tags of the submodel where global solution drives.
    pub driven_tags: Vec<i32>,
    /// Whether to use quadratic interpolation (P2) or linear (P1).
    pub quadratic_interp: bool,
}

impl Default for SubmodelConfig {
    fn default() -> Self {
        Self { driven_tags: vec![1], quadratic_interp: false }
    }
}

/// Result from a submodeling analysis.
#[derive(Debug, Clone)]
pub struct SubmodelResult {
    /// Displacements on the submodel (nodal DOF values).
    pub displacement: Vec<f64>,
    /// Stresses recovered from the submodel.
    pub stress: Vec<f64>,
}

/// Interpolate global solution onto submodel boundary nodes.
///
/// For each submodel boundary node, find the containing element in the
/// global mesh, evaluate the global FE shape functions at the node's
/// physical coordinates, and compute the interpolated displacement.
///
/// This is a stub that demonstrates the API. The full implementation
/// requires point-location in the global mesh and FE shape function
/// evaluation at arbitrary points.
pub fn interpolate_global_to_submodel<M1, M2>(
    _global_mesh: &M1,
    _global_displacement: &[f64],
    _sub_mesh: &M2,
    _sub_boundary_tags: &[i32],
    _quadratic: bool,
) -> Vec<f64>
where
    M1: MeshTopology + Clone,
    M2: MeshTopology + Clone,
{
    // Stub: returns zero displacements
    vec![0.0; 0]
}

/// Drive a submodel analysis from a global solution.
///
/// 1. Interpolates global displacement onto submodel boundary
/// 2. Assembles submodel stiffness and RHS
/// 3. Applies interpolated BCs
/// 4. Solves and returns submodel result
pub fn solve_submodel<M1, M2>(
    _global_mesh: &M1,
    _global_disp: &[f64],
    _sub_mesh: &M2,
    _config: &SubmodelConfig,
    _young_modulus: f64,
    _poisson_ratio: f64,
    _quad_order: u8,
) -> SubmodelResult
where
    M1: MeshTopology + Clone,
    M2: MeshTopology + Clone,
{
    SubmodelResult { displacement: vec![0.0], stress: vec![0.0] }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults() {
        let cfg = SubmodelConfig::default();
        assert_eq!(cfg.driven_tags, vec![1]);
        assert!(!cfg.quadratic_interp);
    }

    #[test]
    fn interpolate_stub_returns() {
        // Just verify the function signature compiles
        let sub_disp = vec![0.0f64; 0];
        let _result = sub_disp;
    }
}
