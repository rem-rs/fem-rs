//! Integration tests for mixed-element mesh assembly.
//!
//! Verifies that the assembler correctly handles meshes that contain more than
//! one element type (e.g. Tri3 + Quad4), and that boundary DOF extraction
//! works correctly on such meshes.

use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_mesh::{
    boundary::BoundaryTag,
    element_type::ElementType,
    topology::MeshTopology,
    SimplexMesh,
};
use fem_space::{
    constraints::{apply_dirichlet, boundary_dofs},
    fe_space::FESpace,
    H1Space,
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Build a small 2-D mixed mesh on [0,1]²:
///
/// Nodes (5 total):
///   0 = (0,0)   1 = (1,0)   2 = (1,1)   3 = (0,1)   4 = (0.5,0.5)
///
/// Elements:
///   e0 = Tri3 {0,1,4}    (lower-left triangle)
///   e1 = Tri3 {1,2,4}    (lower-right triangle)
///   e2 = Quad4 {0,3,2,4} … NOT axis-aligned so we pick a nicer layout:
///
/// Let's use a cleaner 6-node layout:
///   0=(0,0)  1=(0.5,0)  2=(1,0)
///   3=(0,1)  4=(0.5,1)  5=(1,1)
///
/// Elements:
///   e0 = Tri3 {0, 2, 5}   — right triangle
///   e1 = Quad4 {0, 1, 4, 3}  — left square [0,0.5]×[0,1]
///   e2 = Quad4 {1, 2, 5, 4}  — right square [0.5,1]×[0,1]
///
/// Boundary faces (all edges on ∂Ω):
///   tag 1 (bottom): (0,1), (1,2)
///   tag 2 (right):  (2,5)
///   tag 3 (top):    (5,4), (4,3)
///   tag 4 (left):   (3,0)
fn build_mixed_mesh() -> SimplexMesh<2> {
    // 6 nodes
    #[rustfmt::skip]
    let coords = vec![
        0.0, 0.0,   // 0
        0.5, 0.0,   // 1
        1.0, 0.0,   // 2
        0.0, 1.0,   // 3
        0.5, 1.0,   // 4
        1.0, 1.0,   // 5
    ];

    // Mixed connectivity: Tri3{0,1,3} + Quad4{1,2,5,4} + Quad4{0,1,4,3}
    // Using the simpler 3-element layout:
    //   e0 = Tri3 {0, 1, 3}   triangle in lower-left
    //   e1 = Quad4 {1, 2, 5, 4}  right half square
    //   e2 = Quad4 {0, 1, 4, 3}  left half square (shares edge (1,4) with e1)
    // Wait – (0,1,3) and (0,1,4,3) overlap.  Let's be careful:
    //
    // We'll use nodes:
    //   0=(0,0)  1=(1,0)  2=(1,1)  3=(0,1)  4=(0.5,0.5)
    // Elements:
    //   Tri3:  {0, 1, 4}  lower-left-ish
    //   Tri3:  {1, 2, 4}  lower-right
    //   Tri3:  {2, 3, 4}  upper-right
    //   Quad4: {0, 1, 4, 3} — NOT valid quad (4 is inside)
    // A Quad4 needs 4 distinct boundary nodes in order.
    //
    // Simplest valid mixed mesh: one Tri3 + one Quad4 tiling [0,1]x[0,1]
    // cut diagonally by y = x:
    //   nodes: 0=(0,0) 1=(1,0) 2=(1,1) 3=(0,1)  (the 4 corners)
    //   e0 = Tri3  {0, 1, 3}   lower-left tri (vertices 0,1,3)
    //   e1 = Tri3  {1, 2, 3}   upper-right tri
    // That gives 2 tris.  For a Quad:
    //   e0 = Quad4 {0, 1, 2, 3}  the full square — 1 element
    //   then we can't add a tri without subdividing.
    //
    // Let's do a clean 5-node mixed mesh:
    //   nodes: 0=(0,0) 1=(0.5,0) 2=(1,0) 3=(1,1) 4=(0,1)
    //   e0 = Tri3  {0, 2, 3}   (right triangle)
    //   e1 = Quad4 {0, 1, … } — needs 4 nodes
    // Actually the simplest is:
    //   Left half = Quad4, right half = 2 triangles,  5 nodes total:
    //     0=(0,0) 1=(0.5,0) 2=(1,0) 3=(0.5,1) 4=(0,1)  5=(1,1)
    // but that's 6 nodes again.
    //
    // FINAL CHOICE — 5 nodes, 3 elements (1 Quad4 + 2 Tri3):
    //   0=(0,0)  1=(0.5,0)  2=(1,0)  3=(1,1)  4=(0,1)
    //   Quad4: {0, 1, 3, 4}   left rectangle [0,0.5]×[0,1] (NOT axis-aligned)
    //   Tri3:  {1, 2, 3}      right lower triangle
    //   Tri3:  {1, 3, 4}      -- but this overlaps with Quad!
    //
    // Use 6 nodes, 3 elements (1 Quad4 + 2 Tri3), no overlap:
    //   0=(0,0) 1=(0.5,0) 2=(1,0) 3=(0,1) 4=(0.5,1) 5=(1,1)
    //   Quad4: {0, 1, 4, 3}   [0,0.5]×[0,1]
    //   Tri3:  {1, 2, 5}      right lower triangle
    //   Tri3:  {1, 5, 4}      right upper triangle
    //
    // This gives a valid, non-overlapping mixed tiling of the unit square.
    // We use THESE nodes/elements below.

    // ← The actual data:
    #[rustfmt::skip]
    let coords_final = vec![
        0.0, 0.0,   // 0
        0.5, 0.0,   // 1
        1.0, 0.0,   // 2
        0.0, 1.0,   // 3
        0.5, 1.0,   // 4
        1.0, 1.0,   // 5
    ];
    let _ = coords; // suppress unused warning for the first definition

    // Mixed connectivity stored consecutively: Quad4(4 nodes) + Tri3(3) + Tri3(3)
    let conn: Vec<u32> = vec![
        0, 1, 4, 3, // Quad4
        1, 2, 5,    // Tri3
        1, 5, 4,    // Tri3
    ];
    let elem_offsets = vec![0usize, 4, 7, 10];
    let elem_types = vec![ElementType::Quad4, ElementType::Tri3, ElementType::Tri3];
    let elem_tags = vec![0i32; 3];

    // Boundary faces (Line2 edges on ∂Ω), tagged:
    //   tag 1 = bottom (y=0): edges (0,1), (1,2)
    //   tag 2 = right  (x=1): edge  (2,5)
    //   tag 3 = top    (y=1): edges (5,4), (4,3)
    //   tag 4 = left   (x=0): edge  (3,0)
    let face_conn: Vec<u32> = vec![
        0, 1,  // tag 1
        1, 2,  // tag 1
        2, 5,  // tag 2
        5, 4,  // tag 3
        4, 3,  // tag 3
        3, 0,  // tag 4
    ];
    let face_tags: Vec<BoundaryTag> = vec![1, 1, 2, 3, 3, 4];

    SimplexMesh::<2> {
        coords: coords_final,
        conn,
        elem_tags,
        elem_type: ElementType::Tri3, // primary (ignored when elem_types is set)
        face_conn,
        face_tags,
        face_type: ElementType::Line2,
        elem_types: Some(elem_types),
        elem_offsets: Some(elem_offsets),
        face_types: None,
        face_offsets: None,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[test]
fn mixed_mesh_is_mixed() {
    let mesh = build_mixed_mesh();
    assert!(mesh.is_mixed(), "mesh should be flagged as mixed");
    assert_eq!(mesh.n_elements(), 3);
    assert_eq!(mesh.n_nodes(), 6);
}

#[test]
fn mixed_mesh_elem_types_correct() {
    let mesh = build_mixed_mesh();
    assert_eq!(mesh.element_type(0), ElementType::Quad4);
    assert_eq!(mesh.element_type(1), ElementType::Tri3);
    assert_eq!(mesh.element_type(2), ElementType::Tri3);
}

#[test]
fn mixed_mesh_assembles_bilinear() {
    // Should not panic. Matrix dimensions should match DOF count.
    let mesh = build_mixed_mesh();
    let space = H1Space::new(mesh, 1);
    let ndofs = space.n_dofs();
    assert!(ndofs > 0);

    let mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
    assert_eq!(mat.nrows, ndofs);
    assert_eq!(mat.ncols, ndofs);
    // Non-zero count should be > 0
    assert!(mat.nnz() > 0);
}

#[test]
fn mixed_mesh_assembles_linear() {
    let mesh = build_mixed_mesh();
    let space = H1Space::new(mesh, 1);
    let ndofs = space.n_dofs();

    let src = DomainSourceIntegrator::new(|x: &[f64]| {
        2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
    });
    let rhs = Assembler::assemble_linear(&space, &[&src], 3);
    assert_eq!(rhs.len(), ndofs);
    // Not all entries are zero for a non-trivial source
    let nonzero = rhs.iter().any(|&v| v.abs() > 1e-15);
    assert!(nonzero, "RHS should have non-zero entries for non-trivial source");
}

#[test]
fn mixed_mesh_boundary_dofs() {
    let mesh = build_mixed_mesh();
    let space = H1Space::new(mesh.clone(), 1);
    let dm = space.dof_manager();

    // Tags 1,2,3,4 = all four walls — should give all boundary nodes
    let all_bnd = boundary_dofs(&mesh, dm, &[1, 2, 3, 4]);
    assert!(
        !all_bnd.is_empty(),
        "boundary DOFs should not be empty for a closed mesh"
    );

    // Partial boundary: left wall only (tag 4) should give fewer DOFs
    let left_bnd = boundary_dofs(&mesh, dm, &[4]);
    assert!(
        !left_bnd.is_empty(),
        "left-wall boundary DOFs should not be empty"
    );
    assert!(
        left_bnd.len() < all_bnd.len(),
        "partial boundary should have fewer DOFs than the full boundary"
    );
}

#[test]
fn mixed_mesh_poisson_solves() {
    // End-to-end: assemble and solve Poisson on the mixed mesh, check the
    // solve converges (residual is small) without panicking.
    let mesh = build_mixed_mesh();
    let space = H1Space::new(mesh.clone(), 1);
    let ndofs = space.n_dofs();

    let src = DomainSourceIntegrator::new(|x: &[f64]| {
        2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
    });
    let mut mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
    let mut rhs = Assembler::assemble_linear(&space, &[&src], 3);

    let dm = space.dof_manager();
    let bnd = boundary_dofs(&mesh, dm, &[1, 2, 3, 4]);
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &vec![0.0; bnd.len()]);

    let mut u = vec![0.0_f64; ndofs];
    let cfg = fem_solver::SolverConfig {
        rtol: 1e-10,
        max_iter: 1_000,
        verbose: false,
        ..fem_solver::SolverConfig::default()
    };
    let res = fem_solver::solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg)
        .expect("solver should converge on mixed mesh");

    assert!(
        res.converged,
        "PCG should converge on mixed mesh, final residual = {:.3e}",
        res.final_residual
    );
}
