//! # Example 3 (Hex8) �?3D Maxwell smoke on a single Hex8 cell
//!
//! Solves:
//!   curl curl E + E = 0
//! with full PEC boundary on a single hexahedron.
//! The exact solution is the trivial field E=0.

use fem_assembly::{VectorAssembler, standard::{CurlCurlIntegrator, VectorMassIntegrator}};
use fem_core::{ElemId, FaceId, NodeId};
use fem_mesh::{ElementType, topology::MeshTopology};
use fem_solver::{SolverConfig, solve_pcg_jacobi};
use fem_space::{HCurlSpace, constraints::{apply_dirichlet, boundary_dofs_hcurl}, fe_space::FESpace};

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
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            elem: [0, 1, 2, 3, 4, 5, 6, 7],
            bfaces: vec![
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7],
            ],
            btags: vec![1, 2, 3, 4, 5, 6],
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

struct SolveResult {
    n_dofs:         usize,
    converged:      bool,
    iterations:     usize,
    solution_norm:  f64,
}

fn solve_hex8() -> SolveResult {
    let mesh = OneHexMesh::unit();
    let space = HCurlSpace::new(mesh, 1);

    let mut mat = VectorAssembler::assemble_bilinear(
        &space,
        &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }],
        4,
    );
    let n_dofs = space.n_dofs();
    let mut rhs = vec![0.0_f64; n_dofs];

    let bnd = boundary_dofs_hcurl(space.mesh(), &space, &[1, 2, 3, 4, 5, 6]);
    let vals = vec![0.0_f64; bnd.len()];
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &vals);

    let mut u = vec![0.0_f64; n_dofs];
    let cfg = SolverConfig {
        rtol: 1e-10,
        atol: 0.0,
        max_iter: 4000,
        verbose: false,
        ..SolverConfig::default()
    };
    let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("hex8 ex3 solve failed");
    let solution_norm = u.iter().map(|v| v * v).sum::<f64>().sqrt();

    SolveResult { n_dofs, converged: res.converged, iterations: res.iterations, solution_norm }
}

fn main() {
    let r = solve_hex8();
    println!("=== fem-rs mfem_ex3_hex8_maxwell ===");
    println!("  DOFs: {}", r.n_dofs);
    println!("  Converged: {} in {} iterations", r.converged, r.iterations);
    println!("  ||u||_2 = {:.3e}", r.solution_norm);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PEC on all six faces with zero forcing: only trivial solution E=0 exists.
    /// The solver must converge and ||u||₂ must be essentially machine-zero.
    #[test]
    fn hex8_maxwell_pec_trivial_solution_is_zero() {
        let r = solve_hex8();
        assert!(r.converged, "PCG failed to converge for trivial PEC problem");
        assert!(
            r.solution_norm < 1e-12,
            "||u||₂ should be ≈0 for zero-source/zero-BC problem, got {:.3e}",
            r.solution_norm
        );
    }

    /// A single ND1 Hex8 element has 12 edges, each carrying one DOF.
    #[test]
    fn hex8_maxwell_nd1_dof_count_is_twelve() {
        let r = solve_hex8();
        assert_eq!(r.n_dofs, 12, "expected 12 edge DOFs for ND1 on a single Hex8");
    }

    /// Solver must terminate in a small number of iterations for this 12-DOF problem.
    #[test]
    fn hex8_maxwell_solver_terminates_quickly() {
        let r = solve_hex8();
        assert!(r.converged);
        assert!(r.iterations <= 50, "PCG should converge in ≤50 iterations for a 12-DOF smoke problem, used {}", r.iterations);
    }
}

