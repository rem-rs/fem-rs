//! D61 regression: DOF counts on geometrically periodic meshes with fewer
//! than three cells per direction.
//!
//! With a periodic direction carrying only 1-2 cells, distinct torus
//! edges/faces share one folded vertex set, so vertex-pair entity keys
//! (`EdgeKey`/`FaceKey`/`QuadFaceKey`) collide and the DOF count collapses
//! (2×2×2 hex Q2 used to count 34 instead of 64).  `DofManager::new` now
//! numbers DOFs on the un-merged connectivity and quotients them to torus
//! entities (see `DofManager::build_periodic`).
//!
//! Reference counts: MFEM cannot build these meshes at all (its
//! vertex-merge periodicity aborts in `GenerateFaces` on a face claimed by
//! three elements once a direction has 2 cells), so the pins are the
//! topological ground truth of the quotient complex — cell complex counts on
//! the flat torus: 2×2×2 hex → 8 vertices, 24 edges, 24 faces, 8 cells.

use fem_mesh::{Mesh, MeshTopology};
use fem_space::{FESpace, H1Space};

fn periodic_hex_mesh(n: usize) -> Mesh<3> {
    let base =
        Mesh::<3>::make_cartesian_3d(n, n, n, fem_mesh::ElementType::Hex8, 1.0, 1.0, 1.0, true);
    base.make_periodic(
        &[
            (5, 3, [1.0, 0.0, 0.0]),
            (2, 4, [0.0, 1.0, 0.0]),
            (1, 6, [0.0, 0.0, 1.0]),
        ],
        1e-10,
    )
    .unwrap()
}

#[test]
fn d61_hex_2x2x2_dof_counts_match_torus_topology() {
    let mesh = periodic_hex_mesh(2);
    assert_eq!(mesh.n_nodes(), 8, "vertices must merge to the 8 torus points");
    // Q1: 8 vertices.  Q2: + 24 edges + 24 face centres + 8 cell centres.
    // Q3: + 48 edge + 96 face + 64 cell interior DOFs.
    let counts: Vec<(u8, usize)> = [1u8, 2, 3]
        .iter()
        .map(|&o| (o, H1Space::new(mesh.clone(), o).n_dofs()))
        .collect();
    assert_eq!(counts[0], (1, 8), "Q1");
    assert_eq!(counts[1], (2, 64), "Q2 = 8 v + 24 e + 24 f + 8 c");
    assert_eq!(counts[2], (3, 216), "Q3 = 8 v + 48 e + 96 f + 64 c");
}

#[test]
fn d61_quad_2x2_dof_counts_match_torus_topology() {
    let base = Mesh::<2>::make_cartesian_2d(2, 2, 1.0, 1.0);
    let mesh = base
        .make_periodic(&[(4, 2, [1.0, 0.0]), (1, 3, [0.0, 1.0])], 1e-10)
        .unwrap();
    assert_eq!(mesh.n_nodes(), 4);
    // 4 vertices + 8 edges (Q2 adds 4 cell centres; Q3 adds 8 edge + 16
    // cell-interior DOFs).
    assert_eq!(H1Space::new(mesh.clone(), 2).n_dofs(), 16);
    assert_eq!(H1Space::new(mesh.clone(), 3).n_dofs(), 36);
}

#[test]
fn d61_tri_2x2_dof_counts_match_torus_topology() {
    let base = Mesh::<2>::unit_square_tri(2);
    let mesh = base
        .make_periodic(&[(4, 2, [1.0, 0.0]), (1, 3, [0.0, 1.0])], 1e-10)
        .unwrap();
    // The 2x2 tri torus: 4 vertices + 12 edges (8 grid + 4 diagonals).
    assert_eq!(H1Space::new(mesh.clone(), 2).n_dofs(), 16);
    assert_eq!(H1Space::new(mesh.clone(), 3).n_dofs(), 36);
}

/// Three cells per direction must keep the exact MFEM counts
/// (`H1_FECollection(p, 3)` on `MakeCartesian3D(3,3,3)` + `MakePeriodic`).
#[test]
fn d61_hex_3x3x2_counts_match_mfem() {
    for (n, q1, q2, q3) in [(3usize, 27, 216, 729), (4, 64, 512, 1728)] {
        let mesh = periodic_hex_mesh(n);
        assert_eq!(H1Space::new(mesh.clone(), 1).n_dofs(), q1);
        assert_eq!(H1Space::new(mesh.clone(), 2).n_dofs(), q2);
        assert_eq!(H1Space::new(mesh.clone(), 3).n_dofs(), q3);
    }
}

/// The DOF numbering on a periodic mesh must be consistent: every DOF is
/// referenced by at least one element and the dof coordinates of the seam
/// DOFs live on the mesh (finite, non-degenerate).
#[test]
fn d61_2x2x2_every_dof_referenced_and_coords_finite() {
    let mesh = periodic_hex_mesh(2);
    for order in [1u8, 2, 3] {
        let space = H1Space::new(mesh.clone(), order);
        let dm = space.dof_manager();
        let mut seen = vec![false; space.n_dofs()];
        for e in 0..mesh.n_elements() as u32 {
            for &d in dm.element_dofs(e) {
                seen[d as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "order {order}: unreferenced DOFs");
        for dof in 0..space.n_dofs() {
            assert!(dm.dof_coord(dof as u32).iter().all(|x| x.is_finite()));
        }
    }
}

/// n=4 order 4 must keep the exact MFEM count (4096).
#[test]
fn d61_hex_4x4x4_q4_matches_mfem() {
    let mesh = periodic_hex_mesh(4);
    assert_eq!(H1Space::new(mesh, 4).n_dofs(), 4096);
}

/// D56 semantics must hold on the collision mesh too: `interpolate` agrees
/// with the per-element projection (curved-general replica of the d56 suite)
/// on the fully periodic 2×2×2 hex mesh.
#[test]
fn d61_2x2x2_interpolate_matches_per_element_projection() {
    use fem_element::lagrange::factory::HexQk;
    use fem_element::ReferenceElement;

    let mesh = periodic_hex_mesh(2);
    let f = |x: &[f64]| {
        (2.0 * std::f64::consts::PI * x[0]).sin()
            * (3.0 * x[1] + 1.0).cos()
            * x[2]
    };
    for order in [1u8, 2, 3] {
        let space = H1Space::new(mesh.clone(), order);
        let dm = space.dof_manager();
        let mut v = vec![0.0_f64; space.n_dofs()];
        for e in 0..mesh.n_elements() as u32 {
            let rf = HexQk::new(order as usize);
            let q1 = HexQk::new(1);
            let ref_dofs = rf.dof_coords();
            let dofs = dm.element_dofs(e);
            assert_eq!(dofs.len(), ref_dofs.len());
            let gnodes = mesh.geometry_nodes(e);
            let mut phi = vec![0.0_f64; gnodes.len()];
            for (slot, rc) in ref_dofs.iter().enumerate() {
                q1.eval_basis(rc, &mut phi);
                let mut x = [0.0_f64; 3];
                for (k, &p) in phi.iter().enumerate() {
                    let ck = mesh.geom_coords_of(gnodes[k]);
                    for d in 0..3 {
                        x[d] += p * ck[d];
                    }
                }
                v[dofs[slot] as usize] = f(&x);
            }
        }
        let interp = space.interpolate(&f);
        let max_diff = interp
            .as_slice()
            .iter()
            .zip(v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-12,
            "order {order}: interpolate vs per-element projection max |diff| = {max_diff:.3e}"
        );
    }
}
