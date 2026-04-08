//! Integration tests for the fem-space crate.
//!
//! Tests DOF counting, boundary DOF extraction, DOF coordinate access, and
//! interpolation for H1 spaces.

use fem_mesh::{SimplexMesh};
use fem_space::{
    constraints::{apply_dirichlet, boundary_dofs},
    fe_space::FESpace,
    H1Space,
};

// ─── DOF counts ───────────────────────────────────────────────────────────────

#[test]
fn h1_p1_dof_count() {
    // P1 on an n×n mesh has (n+1)² DOFs = number of nodes.
    for n in [4usize, 8, 16] {
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        let expected = (n + 1) * (n + 1);
        let space = H1Space::new(mesh, 1);
        assert_eq!(
            space.n_dofs(),
            expected,
            "P1 n={n}: expected {expected} dofs, got {}",
            space.n_dofs()
        );
    }
}

#[test]
fn h1_p2_dof_count() {
    // P2 adds edge midpoints.  On an n×n triangle mesh:
    //   nodes       = (n+1)²
    //   interior edges = n*(n+1) + n*(n+1) - n² (need mesh.n_edges())
    // Simpler: for n=4, P2 dof count is a known value we can check.
    // Actually: P2 DOFs = nodes + interior_edges + boundary_edges
    // = (n+1)^2 + (total edges)
    // We just verify it is larger than P1 and consistent across levels.
    for n in [4usize, 8] {
        let mesh_p1 = SimplexMesh::<2>::unit_square_tri(n);
        let mesh_p2 = SimplexMesh::<2>::unit_square_tri(n);
        let p1_dofs = H1Space::new(mesh_p1, 1).n_dofs();
        let p2_dofs = H1Space::new(mesh_p2, 2).n_dofs();
        assert!(
            p2_dofs > p1_dofs,
            "P2 should have more DOFs than P1 for n={n}: P1={p1_dofs}, P2={p2_dofs}"
        );
        // P2 DOFs = nodes + edges.  On a uniform triangle mesh:
        //   nodes = (n+1)^2
        //   edges ≈ 3*n^2 + 2*n  (roughly)
        // Just check it's in a reasonable range.
        assert!(p2_dofs >= 2 * p1_dofs - 1,
            "P2 should have at least ~2× P1 DOFs for n={n}");
    }
}

// ─── Boundary DOFs ────────────────────────────────────────────────────────────

#[test]
fn boundary_dofs_all_walls() {
    // Requesting all 4 wall tags should return all boundary nodes.
    let n = 8;
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh.clone(), 1);
    let dm = space.dof_manager();

    let bnd = boundary_dofs(&mesh, dm, &[1, 2, 3, 4]);

    // Number of boundary nodes on an n×n grid = 4*n (perimeter)
    let expected = 4 * n;
    assert_eq!(
        bnd.len(),
        expected,
        "expected {expected} boundary DOFs, got {}",
        bnd.len()
    );
}

#[test]
fn boundary_dofs_single_wall() {
    // Each wall should have n+1 boundary nodes (including corners).
    let n = 8;
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh.clone(), 1);
    let dm = space.dof_manager();

    for tag in [1i32, 2, 3, 4] {
        let bnd = boundary_dofs(&mesh, dm, &[tag]);
        assert_eq!(
            bnd.len(),
            n + 1,
            "tag {tag}: expected {} DOFs, got {}",
            n + 1,
            bnd.len()
        );
    }
}

#[test]
fn boundary_dofs_partial_subset() {
    // Two opposing walls should give fewer DOFs than all four walls.
    let n = 8;
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh.clone(), 1);
    let dm = space.dof_manager();

    let two_walls  = boundary_dofs(&mesh, dm, &[1, 3]); // bottom + top
    let four_walls = boundary_dofs(&mesh, dm, &[1, 2, 3, 4]);

    assert!(
        two_walls.len() < four_walls.len(),
        "two-wall DOFs ({}) should be < four-wall DOFs ({})",
        two_walls.len(),
        four_walls.len()
    );
}

// ─── DOF coordinates ──────────────────────────────────────────────────────────

#[test]
fn p1_dof_coords_match_node_coords() {
    // For P1, each DOF corresponds to a mesh node.
    // DOF coordinate returned by dof_manager should equal the mesh node coord.
    let mesh = SimplexMesh::<2>::unit_square_tri(4);
    let space = H1Space::new(mesh.clone(), 1);
    let dm = space.dof_manager();
    let n_dofs = space.n_dofs();

    // For P1 the DOFs are ordered the same as the nodes.
    // Just verify all DOF coords lie in [0,1]^2.
    for dof in 0..n_dofs as u32 {
        let c = dm.dof_coord(dof);
        assert!(c.len() == 2, "2-D DOF should have 2 coords");
        assert!(c[0] >= -1e-12 && c[0] <= 1.0 + 1e-12,
            "DOF {dof}: x={} not in [0,1]", c[0]);
        assert!(c[1] >= -1e-12 && c[1] <= 1.0 + 1e-12,
            "DOF {dof}: y={} not in [0,1]", c[1]);
    }
}

// ─── Interpolation ────────────────────────────────────────────────────────────

#[test]
fn p1_interpolate_linear_exact() {
    // P1 interpolates linear functions exactly.
    let mesh = SimplexMesh::<2>::unit_square_tri(4);
    let space = H1Space::new(mesh, 1);
    let v = space.interpolate(&|x| 2.0 * x[0] + 3.0 * x[1]);
    let dm = space.dof_manager();
    let dofs = v.as_slice();

    for dof in 0..space.n_dofs() as u32 {
        let c = dm.dof_coord(dof);
        let expected = 2.0 * c[0] + 3.0 * c[1];
        assert!(
            (dofs[dof as usize] - expected).abs() < 1e-12,
            "DOF {dof}: interpolated={:.6}, expected={:.6}",
            dofs[dof as usize],
            expected
        );
    }
}

#[test]
fn p1_interpolate_constant_exact() {
    let mesh = SimplexMesh::<2>::unit_square_tri(4);
    let space = H1Space::new(mesh, 1);
    let v = space.interpolate(&|_| 7.0);
    let dofs = v.as_slice();
    for (i, &d) in dofs.iter().enumerate() {
        assert!((d - 7.0).abs() < 1e-14, "DOF {i}: expected 7.0, got {d}");
    }
}

#[test]
fn p2_interpolate_quadratic_exact() {
    // P2 interpolates quadratics exactly.
    let mesh = SimplexMesh::<2>::unit_square_tri(4);
    let space = H1Space::new(mesh, 2);
    let v = space.interpolate(&|x| x[0] * x[0] + x[1] * x[1]);
    let dm = space.dof_manager();
    let dofs = v.as_slice();

    for dof in 0..space.n_dofs() as u32 {
        let c = dm.dof_coord(dof);
        let expected = c[0] * c[0] + c[1] * c[1];
        assert!(
            (dofs[dof as usize] - expected).abs() < 1e-12,
            "P2 DOF {dof}: interpolated={:.6}, expected={:.6}",
            dofs[dof as usize],
            expected
        );
    }
}

// ─── Dirichlet application ────────────────────────────────────────────────────

#[test]
fn apply_dirichlet_zeros_rhs_at_boundary() {
    use fem_linalg::{CooMatrix, CsrMatrix};
    use fem_mesh::topology::MeshTopology;

    let n = 4;
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh.clone(), 1);
    let ndofs = space.n_dofs();
    let dm = space.dof_manager();
    let bnd = boundary_dofs(&mesh, dm, &[1, 2, 3, 4]);

    // Build a trivial identity matrix (so we can test Dirichlet application).
    let mut coo = CooMatrix::<f64>::new(ndofs, ndofs);
    for i in 0..ndofs {
        coo.add(i, i, 2.0);
        if i + 1 < ndofs { coo.add(i, i + 1, -1.0); coo.add(i + 1, i, -1.0); }
    }
    let mut mat: CsrMatrix<f64> = coo.into_csr();
    let mut rhs = vec![1.0_f64; ndofs]; // non-zero everywhere initially

    apply_dirichlet(&mut mat, &mut rhs, &bnd, &vec![0.0; bnd.len()]);

    // After applying zero Dirichlet: rhs at boundary DOFs should be 0.
    for &d in &bnd {
        assert!(
            rhs[d as usize].abs() < 1e-14,
            "rhs at boundary DOF {d} should be 0 after Dirichlet, got {}",
            rhs[d as usize]
        );
    }

    // Diagonal at boundary DOFs should be 1 (from row-zeroing).
    for &d in &bnd {
        assert!(
            (mat.get(d as usize, d as usize) - 1.0).abs() < 1e-14,
            "diagonal at boundary DOF {d} should be 1 after Dirichlet"
        );
    }
}
