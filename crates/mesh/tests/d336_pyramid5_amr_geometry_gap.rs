//! D336 — Pyramid5 AMR drops the high-order geometry table (characterisation).
//!
//! `amr_inner::refine_nonconforming_pyramid_internal` builds its 16 Tet4
//! children with `Mesh::uniform(..)` — `geometry: None` — and derives every new
//! vertex from `mesh.coords_of(..)`, i.e. from the **vertex** table.  Since
//! D472 the *uniform* path (`refine_pyramid5_uniform` → `refine_mixed_3d`)
//! follows MFEM's PYRAMID branch instead (6 Pyramid5 + 4 Tet4, straight
//! children, `geometry: None` as well) — either way every new vertex comes
//! from the vertex table, and the curved gap below is untouched.
//!
//! Consequences for a **curved** pyramid parent, both measured below:
//! * `geom_order()` of the refined mesh is 1 (the curvature is silently dropped);
//! * the children's total volume is that of the *straight* pyramid with the same
//!   five corners, not the parent's isoparametric volume.
//!
//! This is the same class of documented gap as the Hex20/Hex27 refinement
//! (`refine_uniform_3d`'s "Documented gap (D166 follow-up)"), and the fix has the
//! same shape as the hex one: create each new node by evaluating the parent's
//! isoparametric map at that node's *reference* position (all of them lie on
//! parent edges / the parent's base-quad center, so the positions are known in
//! the local slot frame), and attach an order-`g` geometry table to the
//! children.  **Not implemented in this round** — the numbers below are the
//! gap's exact size, and the straight-path behaviour is pinned so a future fix
//! cannot regress it.
//!
//! Evidence: `tmp/d343/EVIDENCE.md` §5.

use fem_mesh::element_type::ElementType;
use fem_mesh::transformation::element_jacobian_at;
use fem_mesh::{refine_uniform_3d, Mesh, MeshTopology};

const UNIT_VERTICES: [[f64; 3]; 5] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

/// The unit pyramid with its 6 Tri3 boundary faces (the base quad as two
/// triangles, plus the four sides).
fn unit_pyramid() -> Mesh<3> {
    let coords: Vec<f64> = UNIT_VERTICES.iter().flatten().copied().collect();
    let faces: Vec<u32> = vec![0, 1, 2, 0, 2, 3, 0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4];
    Mesh::<3>::uniform(
        coords,
        vec![0, 1, 2, 3, 4],
        vec![1],
        ElementType::Pyramid5,
        faces,
        vec![1; 6],
        ElementType::Tri3,
    )
}

/// Total volume of a mesh from the isoparametric Jacobian of every element.
fn volume(mesh: &Mesh<3>) -> f64 {
    let rule = fem_element::quadrature::pyramid_rule(6);
    let mut vol = 0.0;
    for e in 0..mesh.n_elems() as u32 {
        let et = mesh.element_type(e);
        let (pts, wts): (Vec<Vec<f64>>, Vec<f64>) = match et {
            ElementType::Pyramid5 => (
                rule.points.clone(),
                rule.weights.clone(),
            ),
            ElementType::Tet4 => {
                // 4-point rule is exact for the linear map (det J constant).
                let r = fem_element::quadrature::tet_rule(3);
                (r.points, r.weights)
            }
            other => panic!("unexpected child element type {other:?}"),
        };
        for (xi, w) in pts.iter().zip(wts.iter()) {
            let (jac, _x) = element_jacobian_at(mesh, e, xi, 3);
            vol += w * jac.determinant().abs();
        }
    }
    vol
}

/// Straight pyramids refine geometrically exact: since D472 the uniform path
/// follows MFEM's PYRAMID branch — 6 Pyramid5 + 4 Tet4 children (oracle
/// `tmp/d472/probe_pyr1.out`); it used to be fem-rs's own 16-Tet4 split.  No
/// geometry table is needed, same volume (this is what the AMR regression pins).
#[test]
fn d336_straight_pyramid_refinement_is_geometrically_exact() {
    let mesh = unit_pyramid();
    let before = volume(&mesh);
    let refined = refine_uniform_3d(&mesh);
    assert_eq!(refined.n_elems(), 10);
    assert_eq!(refined.element_type(0), ElementType::Pyramid5);
    assert!(refined.geometry.is_none(), "straight children need no geometry table");
    let after = volume(&refined);
    eprintln!("D336 straight: parent = {before:.15}, children = {after:.15}");
    assert!((before - 1.0 / 3.0).abs() < 1e-13);
    assert!((after - before).abs() < 1e-13, "straight refinement must preserve the volume");
    assert!(!refined.face_conn.is_empty());
}

/// A **curved** pyramid loses its geometry: `geom_order` drops 2 → 1 and the
/// children describe the straight pyramid with the same corners.
#[test]
fn d336_curved_pyramid_loses_its_geometry() {
    let mut mesh = unit_pyramid();
    mesh.set_curvature(2);
    // Push the base-centre quadratic node (Fuentes slot 13) up in z: the parent
    // is then genuinely curved and its isoparametric volume is < 1/3.
    let node = mesh.geometry_nodes(0)[13] as usize;
    {
        let geo = mesh.geometry.as_mut().expect("geometry table");
        geo.coords[node * 3 + 2] += 0.25;
    }
    let parent = volume(&mesh);
    assert_eq!(mesh.geom_order(), 2);
    assert!((parent - 0.222222222222222).abs() < 1e-12, "parent volume {parent:.15}");

    let refined = refine_uniform_3d(&mesh);
    // D472: the uniform path now produces MFEM's 6 Pyramid5 + 4 Tet4 children
    // (still straight-sided); the curved-geometry gap below is unchanged.
    assert_eq!(refined.n_elems(), 10);
    assert_eq!(refined.element_type(0), ElementType::Pyramid5);
    let child = volume(&refined);

    eprintln!(
        "D336 curved: geom_order {} -> {}, volume {parent:.15} -> {child:.15}",
        mesh.geom_order(),
        refined.geom_order()
    );
    assert_eq!(
        refined.geom_order(),
        1,
        "the refined pyramid mesh keeps no geometry table (open D336 gap)"
    );
    assert!(refined.geometry.is_none());
    assert!(
        (child - 1.0 / 3.0).abs() < 1e-12,
        "the children describe the *straight* pyramid ({child:.15})"
    );
    assert!(
        (child - parent).abs() > 1e-3,
        "if this ever passes, the curved-pyramid refinement has been implemented \
         and this test should be replaced by an isoparametric acceptance test"
    );
}
