//! # Example 3 (Hex8) — 3D Maxwell smoke on a single Hex8 cell
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

fn main() {
    let mesh = OneHexMesh::unit();
    let space = HCurlSpace::new(mesh, 1);

    let mut mat = VectorAssembler::assemble_bilinear(
        &space,
        &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }],
        4,
    );
    let mut rhs = vec![0.0_f64; space.n_dofs()];

    let bnd = boundary_dofs_hcurl(space.mesh(), &space, &[1, 2, 3, 4, 5, 6]);
    let vals = vec![0.0_f64; bnd.len()];
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &vals);

    let mut u = vec![0.0_f64; space.n_dofs()];
    let cfg = SolverConfig {
        rtol: 1e-10,
        atol: 0.0,
        max_iter: 4000,
        verbose: false,
        ..SolverConfig::default()
    };
    let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("hex8 ex3 solve failed");
    let norm_u = u.iter().map(|v| v * v).sum::<f64>().sqrt();

    println!("=== fem-rs ex3_hex8_maxwell ===");
    println!("  DOFs: {}", space.n_dofs());
    println!("  Converged: {} in {} iterations", res.converged, res.iterations);
    println!("  ||u||_2 = {:.3e}", norm_u);
}
