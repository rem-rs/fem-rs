//! D306 — curved-pyramid geometry must be **isoparametric** in the
//! vector / Q-space geometry path.
//!
//! `vector_assembler::geo_ref_elem_from_mesh` is the geometry-element entry
//! point of the vector assembler, `qspace`, the DPG weakforms, `hdiv_error`
//! and the vector boundary assembler.  Its pyramid arm was guarded by
//! `ElementType::Pyramid5 if g <= 1`, so a pyramid with `geom_order > 1` fell
//! through to `_ => return None`:
//!
//! * `qspace::map_element_quadrature_points` treated it as an affine P1
//!   simplex and **panicked** (`unsupported element Pyramid5`);
//! * the other callers silently ran a vertex-only (non-isoparametric)
//!   geometry, i.e. every high-order geometry node was ignored.
//!
//! The fix adds the curved arm, using the *layer-order* `PyramidPk(g)` — the
//! same geometry element `assembler::geo_ref_elem` and
//! `Mesh::element_jacobian` use, because `Mesh::set_curvature_pyramid5` writes
//! the geometry DOFs in layer-slot order (the frozen D191 contract).
//!
//! The assertions below cross-check the geometry element against
//! `Mesh::element_jacobian` (an independent implementation of the same
//! isoparametric Jacobian, `crates/mesh/src/simplex.rs`) and against the
//! analytic volume of the unit pyramid.  Measured on the *curved* fixture: the
//! two Jacobians agree to ≤ 1e-15 and the volume moves off 1/3 exactly when a
//! high-order geometry node is displaced — a vertex-only (P1) map cannot see
//! that displacement at all.

use fem_assembly::geo_ref_elem_from_mesh;
use fem_element::quadrature::pyramid_rule;
use fem_element::ReferenceElement;
use fem_mesh::{element_type::ElementType, Mesh, MeshTopology};

/// Unit pyramid: base quad (0,0,0),(1,0,0),(1,1,0),(0,1,0) + apex (0,0,1),
/// vertices in MFEM/Gmsh order.
fn unit_pyramid() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            1.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0,
        ],
        vec![0, 1, 2, 3, 4],
        vec![1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// `Σ_q w_q |det J(ξ_q)|` of element 0, using the geometry element under test
/// over `mesh.geometry_nodes(0)` (an independent re-implementation of the map,
/// deliberately not calling into `qspace`).
fn volume_with_geo(mesh: &Mesh<3>, geo: &dyn ReferenceElement, order: u8) -> f64 {
    let rule = pyramid_rule(order);
    let nodes = mesh.geometry_nodes(0);
    let n = geo.n_dofs();
    assert_eq!(nodes.len(), n, "geometry table and element disagree on n_dofs");
    let mut grad = vec![0.0_f64; n * 3];
    let mut totally = 0.0_f64;
    for (q, xi) in rule.points.iter().enumerate() {
        geo.eval_grad_basis(xi, &mut grad);
        let mut j = [[0.0_f64; 3]; 3];
        for k in 0..n {
            let x = mesh.geom_coords_of(nodes[k]);
            for i in 0..3 {
                for d in 0..3 {
                    j[i][d] += x[i] * grad[k * 3 + d];
                }
            }
        }
        let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
            - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
            + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
        totally += rule.weights[q] * det.abs();
    }
    totally
}

/// Volume of element 0 via the mesh's own isoparametric Jacobian.
fn volume_with_mesh(mesh: &Mesh<3>, order: u8) -> f64 {
    let rule = pyramid_rule(order);
    rule.points
        .iter()
        .zip(rule.weights.iter())
        .map(|(xi, w)| w * mesh.element_jacobian(0, xi).1.abs())
        .sum()
}

#[test]
fn d306_curved_pyramid_geometry_element_is_isoparametric() {
    let mut mesh = unit_pyramid();
    mesh.set_curvature(2);
    let geo = geo_ref_elem_from_mesh(&mesh, 0)
        .expect("curved pyramid must have a geometry element (D306)");
    assert_eq!(geo.order(), 2);
    assert_eq!(geo.dim(), 3);
    // D347: `Mesh::set_curvature` writes MFEM `SetCurvature`'s default family
    // for pyramids — the **Fuentes** pyramid (`pyr_type =
    // ScalarPyramid::DefaultType = 1`), `p(p²+3)+1 = 15` nodes at p = 2 (the
    // pre-D347 layer-order `PyramidPk(2)` table had 14).
    assert_eq!(geo.n_dofs(), 15);
    assert_eq!(geo.n_dofs(), mesh.geometry_nodes(0).len());

    // MFEM 4.10 ground truth for the table itself: `tmp/d347/probe.cpp` run as
    // `./probe B <mfem>/data` prints `SetCurvature(2, false, -1, byNODES, 1)`
    // on the same unit pyramid as `GGEOM 0 ndof=15 | <reference positions>`
    // (dump `crates/space/tests/data/fuentes_pyramid_geometry_mfem.txt`,
    // `GEOM unit g=2 pyr_type=1`); these are exactly its 15 entries.
    let cpp: [[f64; 3]; 15] = [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0], [1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5], [0.5, 0.5, 0.5], [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.0],      // base-quad centre
        [0.25, 0.25, 0.5],    // Fuentes interior bubble node
    ];
    let coords = geo.dof_coords();
    for (s, want) in cpp.iter().enumerate() {
        for d in 0..3 {
            assert!(
                (coords[s][d] - want[d]).abs() < 1e-15,
                "geometry slot {s}: {:?} != MFEM {want:?}",
                coords[s],
            );
        }
    }

    // Nodal identity: the isoparametric map reproduces every geometry node at
    // its own reference coordinate (fails if the table were paired with a
    // different family's basis, the D304/d331 defect class).
    let nodes = mesh.geometry_nodes(0);
    let n = geo.n_dofs();
    let mut phi = vec![0.0_f64; n];
    let mut worst = 0.0_f64;
    for d in 0..n {
        geo.eval_basis(&coords[d], &mut phi);
        let mut xp = [0.0_f64; 3];
        for k in 0..n {
            let x = mesh.geom_coords_of(nodes[k]);
            for i in 0..3 {
                xp[i] += phi[k] * x[i];
            }
        }
        let xd = mesh.geom_coords_of(nodes[d]);
        for i in 0..3 {
            worst = worst.max((xp[i] - xd[i]).abs());
        }
    }
    eprintln!("D306 curved-pyramid nodal identity: worst = {worst:.3e}");
    assert!(worst < 1e-14, "nodal identity broken: {worst:.3e}");

    // Cross-check the geometry element against the mesh's own Jacobian, and
    // against the analytic volume of the unit pyramid (a layer-order mismatch
    // would give ~0.169 instead of 1/3 — see D304/D331).
    let v_geo = volume_with_geo(&mesh, geo.as_ref(), 8);
    let v_mesh = volume_with_mesh(&mesh, 8);
    eprintln!("D306 curved-pyramid volume: geo = {v_geo:.17}, mesh = {v_mesh:.17}");
    assert!(
        (v_geo - v_mesh).abs() < 1e-14,
        "geo element and Mesh::element_jacobian disagree: {v_geo} vs {v_mesh}"
    );
    assert!((v_geo - 1.0 / 3.0).abs() < 1e-13, "unit pyramid volume = {v_geo}");
}

/// The map must actually *see* the high-order geometry: displacing a
/// second-order-only node (a base edge midpoint, layer k = 0) leaves the 5
/// vertex coordinates untouched, so a vertex-only/P1 geometry would still
/// integrate to 1/3.  The isoparametric map reproduces the displacement
/// exactly and reports the changed volume, and it still agrees with
/// `Mesh::element_jacobian`.
#[test]
fn d306_curved_pyramid_map_follows_high_order_nodes() {
    let mut mesh = unit_pyramid();
    mesh.set_curvature(2);
    let geo = geo_ref_elem_from_mesh(&mesh, 0).expect("curved pyramid geometry element");
    let base_v = 1.0 / 3.0;

    // Displace the base-quad centre — slot 13 of the Fuentes table, the
    // `(p−1)²` face block's single p = 2 entry (in the pre-D347 layer-order
    // `PyramidPk(2)` table the same point sat at slot 4's neighbourhood, which
    // is now the apex and *reuses* the mesh vertex, so displacing it would be
    // a no-op) — straight up by 0.1: a pure second-order feature, no vertex
    // moves, the P1 map is unchanged.
    let nodes = mesh.geometry_nodes(0).to_vec();
    let moved = nodes[13];
    let before = mesh.geom_coords_of(moved).to_vec();
    {
        let g = mesh.geometry.as_mut().expect("curvature created a geometry table");
        let off = moved as usize * 3;
        g.coords[off + 2] += 0.1;
    }
    mesh.invalidate_locators();
    let after = mesh.geom_coords_of(moved).to_vec();
    assert!((after[2] - before[2] - 0.1).abs() < 1e-16, "displacement not applied");

    // Nodal identity still holds at the displaced node (isoparametric).
    let coords = geo.dof_coords();
    let n = geo.n_dofs();
    let mut phi = vec![0.0_f64; n];
    let mut worst = 0.0_f64;
    for d in 0..n {
        geo.eval_basis(&coords[d], &mut phi);
        let mut xp = [0.0_f64; 3];
        for k in 0..n {
            let x = mesh.geom_coords_of(nodes[k]);
            for i in 0..3 {
                xp[i] += phi[k] * x[i];
            }
        }
        let xd = mesh.geom_coords_of(nodes[d]);
        for i in 0..3 {
            worst = worst.max((xp[i] - xd[i]).abs());
        }
    }
    assert!(worst < 1e-14, "nodal identity after displacement: {worst:.3e}");

    let v_geo = volume_with_geo(&mesh, geo.as_ref(), 8);
    let v_mesh = volume_with_mesh(&mesh, 8);
    eprintln!(
        "D306 displaced-node volume: geo = {v_geo:.17}, mesh = {v_mesh:.17}, straight = {base_v:.17}"
    );
    assert!(
        (v_geo - v_mesh).abs() < 1e-14,
        "geo element and Mesh::element_jacobian disagree after displacement"
    );
    assert!(
        (v_geo - base_v).abs() > 1e-4,
        "the map ignored the high-order geometry node (non-isoparametric): {v_geo}"
    );
}
