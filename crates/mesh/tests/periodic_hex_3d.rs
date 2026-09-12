//! 3-D periodic hex-mesh construction (round 19, MFEM `MakePeriodic` on a
//! Cartesian cube): `Mesh::make_cartesian_3d` + `Mesh::make_periodic` over all
//! three directions, the mesh-level counterpart of MFEM's
//! `data/periodic-cube.mesh` used by `navier_tgv`.
//!
//! Checks the topology (vertex merge counts), the MFEM per-element geometry
//! snapshot (D56: every element keeps the coordinates of its *own* side of a
//! periodic seam), and the interplay with `Mesh::transform` (the C++ miniapp
//! scales `*nodes *= M_PI` *after* the periodic identification).

use fem_mesh::{ElementType, Mesh, MeshTopology};

/// The tags `make_cartesian_3d` assigns to the box sides:
/// bottom `z=0` → 1, front `y=0` → 2, right `x=sx` → 3, back `y=sy` → 4,
/// left `x=0` → 5, top `z=sz` → 6.
const PERIODIC_PAIRS: [(i32, i32, [f64; 3]); 3] = [
    (5, 3, [1.0, 0.0, 0.0]), // left  → right, +x
    (2, 4, [0.0, 1.0, 0.0]), // front → back,  +y
    (1, 6, [0.0, 0.0, 1.0]), // bottom → top,  +z
];

/// MFEM `Mesh::MakePeriodic(MakeCartesian3D(n,n,n,…), translations)` on the
/// unit cube: an `n×n×n` torus with `n³` merged vertices and no boundary
/// elements left.
fn periodic_cube(n: usize) -> Mesh<3> {
    let base = Mesh::<3>::make_cartesian_3d(n, n, n, ElementType::Hex8, 1.0, 1.0, 1.0, false);
    base.make_periodic(&PERIODIC_PAIRS, 1e-10).expect("make_periodic")
}

/// The vertex merge collapses the `(n+1)³` lattice to the `n³` torus, keeps
/// all `n³` hexes, and removes every boundary face (all six tags are periodic).
#[test]
fn periodic_cube_topology() {
    for n in [3usize, 8] {
        let pm = periodic_cube(n);
        assert_eq!(pm.n_nodes(), n * n * n, "n={n}: merged vertex count");
        assert_eq!(pm.n_elements(), n * n * n, "n={n}: element count");
        assert_eq!(pm.n_boundary_faces(), 0, "n={n}: periodic faces removed");
        assert_eq!(pm.face_tags.len(), 0, "n={n}: no boundary tags left");
    }
}

/// MFEM periodic-mesh semantics (mesh.cpp `MakePeriodic` + D56): the merged
/// mesh carries an order-1 *per-element* geometry snapshot of the pre-merge
/// mesh, so a seam element's corner coordinates stay on its own side of the
/// seam (`x = 1` for the last element, `x = 0` for the first) even though both
/// now reference the same topological vertex.
#[test]
fn periodic_seam_elements_keep_own_coordinates() {
    let pm = periodic_cube(3);
    // The snapshot exists, is order 1, and holds the pre-merge node table.
    assert_eq!(pm.geom_order(), 1);
    assert_ne!(pm.geom_n_nodes(), pm.n_nodes());
    assert_eq!(pm.geom_n_nodes(), 4 * 4 * 4, "pre-merge lattice size");

    // First element: corners at x = 0. Last element: corners at x = 1.
    // (lexicographic element order: e = x + 3*(y + 3*z))
    for e in [0u32, 2, 26] {
        let nodes = pm.geometry_nodes(e);
        let xs: Vec<f64> = nodes.iter().map(|&n| pm.geom_coords_of(n)[0]).collect();
        let (xmin, xmax) = (
            xs.iter().cloned().fold(f64::INFINITY, f64::min),
            xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
        let span = 1.0 / 3.0;
        assert!(
            (xmax - xmin - span).abs() < 1e-12,
            "e={e}: x-span {xmax}-{xmin}"
        );
    }
    // Element 0 spans x ∈ [0, 1/3]; element 2 spans x ∈ [2/3, 1] — the seam
    // element keeps its replica coordinates (not folded to [0, 1/3]).
    let e0 = pm.geometry_nodes(0);
    let e0_xmin = e0.iter().map(|&n| pm.geom_coords_of(n)[0]).fold(f64::INFINITY, f64::min);
    assert!((e0_xmin - 0.0).abs() < 1e-12, "e0 xmin = {e0_xmin}");
    let e2 = pm.geometry_nodes(2);
    let e2_xmax = e2.iter().map(|&n| pm.geom_coords_of(n)[0]).fold(f64::NEG_INFINITY, f64::max);
    assert!((e2_xmax - 1.0).abs() < 1e-12, "e2 xmax = {e2_xmax}");

    // Corner vertex (0,0,0) is shared by 8 elements; through element 2's own
    // geometry it sits at (1,1,1)... through element 0 at (0,0,0).  Both are
    // reachable: element 26 (x=2/3,y=2/3,z=2/3 cell) has corner vertex id 0
    // (the merged lattice wraps (1,1,1)→(0,0,0)) at coordinates (1,1,1).
    let e26 = pm.geometry_nodes(26);
    let max_x = e26.iter().map(|&n| pm.geom_coords_of(n)[0]).fold(f64::NEG_INFINITY, f64::max);
    assert!((max_x - 1.0).abs() < 1e-12);
}

/// The C++ `navier_tgv` scales the periodic mesh *after* identification
/// (`*nodes *= M_PI` on a `[-1,1]³` cube); `Mesh::transform` must scale the
/// vertex table *and* the per-element geometry snapshot together.
#[test]
fn transform_scales_vertex_and_geometry_tables() {
    // `make_cartesian_3d` builds `[0,2]³`; `x ↦ (x−1)·π` maps it onto the
    // C++ `[-1,1]³ · π = [-π,π]³` domain in one affine map.
    let mut pm = periodic_cube(3);
    pm.transform(|p| [(p[0] - 1.0) * std::f64::consts::PI, (p[1] - 1.0) * std::f64::consts::PI, (p[2] - 1.0) * std::f64::consts::PI]);
    // Vertex table: [-π, π] torus lattice.
    let v0 = pm.node_coords(0);
    for d in 0..3 {
        assert!((v0[d] + std::f64::consts::PI).abs() < 1e-12);
    }
    // Geometry snapshot: seam element keeps its replica side, transformed
    // (its pre-transform xmax 1.0 maps to (1−1)·π = 0, the +π side of the
    // torus being folded onto the vertex table's −π).
    let e2 = pm.geometry_nodes(2);
    let e2_xmax = e2.iter().map(|&n| pm.geom_coords_of(n)[0]).fold(f64::NEG_INFINITY, f64::max);
    assert!(e2_xmax.abs() < 1e-12, "e2 xmax = {e2_xmax}");
}

/// Every element of the merged mesh is a non-degenerate axis-aligned cube of
/// side `1/n` (a bad vertex merge would fold a seam element flat or duplicate
/// coordinates inside it).
#[test]
fn periodic_elements_are_unit_cubes() {
    let n = 8;
    let pm = periodic_cube(n);
    let h = 1.0 / n as f64;
    for e in 0..pm.n_elements() as u32 {
        let nodes = pm.geometry_nodes(e);
        assert_eq!(nodes.len(), 8);
        let c0 = pm.geom_coords_of(nodes[0]);
        // The element diagonal endpoint is the local vertex 6 of the MFEM hex
        // layout: opposite corner at +h in every direction.
        let c6 = pm.geom_coords_of(nodes[6]);
        for d in 0..3 {
            assert!((c6[d] - c0[d] - h).abs() < 1e-12, "e={e} d={d}");
        }
        // |det J| = (h/2)³ at the element centre: `element_jacobian_at` maps
        // the [-1,1]³ reference hex (MFEM convention) to physical space.
        use fem_mesh::element_jacobian_at;
        let pt = [0.5f64; 3];
        let (jac, _xp) = element_jacobian_at(&pm, e, &pt, 3);
        let det = jac[(0, 0)] * (jac[(1, 1)] * jac[(2, 2)] - jac[(1, 2)] * jac[(2, 1)])
            - jac[(0, 1)] * (jac[(1, 0)] * jac[(2, 2)] - jac[(1, 2)] * jac[(2, 0)])
            + jac[(0, 2)] * (jac[(1, 0)] * jac[(2, 1)] - jac[(1, 1)] * jac[(2, 0)]);
        let expect = (h / 2.0) * (h / 2.0) * (h / 2.0);
        assert!((det - expect).abs() < 1e-12, "e={e}: det J = {det}");
    }
}

/// The merged mesh is a genuine torus: every node is a vertex of exactly 8
/// elements (as on the unmerged lattice) and every element touches only nodes
/// of the merged `n³` table — a wrong merge would strand nodes on one side of
/// a seam or double-count them.
#[test]
fn merged_mesh_is_a_torus() {
    let n = 3;
    let pm = periodic_cube(n);
    let mut uses = vec![0u32; pm.n_nodes()];
    for e in 0..pm.n_elements() as u32 {
        let mut seen = std::collections::HashSet::new();
        for &nd in pm.element_nodes(e) {
            assert!(seen.insert(nd), "e={e}: repeated node {nd} in conn");
            uses[nd as usize] += 1;
        }
    }
    assert!(uses.iter().all(|&u| u == 8), "node uses = {uses:?}");
}
