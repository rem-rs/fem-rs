//! D632 — `refine_curved_3d_nc_general` pyramid arm family/slot mismatch.
//!
//! The pyramid branch picked its geometry element through
//! `factory::ref_elem(Pyramid, g)` = the **equispaced** `PyramidPk` (14 dofs at
//! order 2), while every fem-rs curved-pyramid geometry table is laid out in
//! the **Fuentes** slot order (`set_curvature_pyramid5`, D347;
//! `p(p²+3)+1 = 15` slots at order 2, the MFEM `SetCurvature` contract, D613).
//! Reading a Fuentes-15 table with that 14-dof element silently corrupts the
//! re-interpolated child geometry:
//!
//! * the output table loses the 15th slot (`nodes_per_elem = 14`) and can no
//!   longer be read by the Fuentes family at all (`curved_pyramid_geometry`
//!   refuses, everything silently falls back);
//! * with the fallback the corners are evaluated in the *layer* basis against
//!   a *vertex*-ordered table (the un-permuted D191 corner swap), so the unit
//!   pyramid's `∫|det J|` collapses to ≈ 0.1738 instead of 1/3.
//!
//! The hex/prism arms are self-consistent (`factory::ref_elem` IS their
//! geometry family); the fix routes the pyramid arm (and the `CurvedMesh`
//! geometry evaluators the re-interpolation samples the parent map through)
//! through `curved_geometry_ref_elem_3d`, their actual table family.
//!
//! Oracle: identity-map roundtrip.  `refine_curved_3d_nc_general` with an
//! empty mark list re-tables every parent; on the straight unit pyramid the
//! Fuentes-15 table must survive the roundtrip bit-for-bit and `∫|det J|`
//! must stay 1/3 (d334: the Fuentes interpolation of the trivial map is the
//! identity, `det J ≡ 1`).

use fem_element::QuadratureRule;
use fem_mesh::element_type::ElementType;
use fem_mesh::transformation::element_jacobian_at;
use fem_mesh::{CurvedMesh, Mesh, refine_curved_3d_nc_general};

fn unit_pyramid() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        vec![0, 1, 2, 3, 4],
        vec![1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// Pack a `set_curvature` geometry table into a `CurvedMesh` (the pyramid
/// labels are connectivity types: the 15-slot Fuentes table rides on
/// `nodes_per_elem`, D613).
fn curved_pyramid(mesh: &Mesh<3>) -> CurvedMesh<3> {
    let g = mesh.geometry.as_ref().expect("set_curvature table");
    CurvedMesh {
        coords: g.coords.clone(),
        geom_conn: g.conn.clone(),
        geom_order: g.order,
        nodes_per_elem: g.nodes_per_elem,
        elem_type: ElementType::Pyramid13,
        n_elems: mesh.n_elems(),
        n_nodes: g.n_nodes,
        face_conn: mesh.face_conn.clone(),
        face_tags: mesh.face_tags.clone(),
        face_type: mesh.face_type,
        elem_tags: mesh.elem_tags.clone(),
    }
}

/// `∫|det J|` of the curved-pyramid `CurvedMesh` element `e`, through the
/// family-true reader (`curved_pyramid_geometry` → Fuentes order-g element).
fn pyramid_volume(mesh: &CurvedMesh<3>, e: u32, quad_order: u8) -> f64 {
    let rule: QuadratureRule = fem_element::quadrature::pyramid_rule(quad_order);
    rule.points
        .iter()
        .zip(rule.weights.iter())
        .map(|(xi, w)| w * element_jacobian_at(mesh, e, xi, 3).0.determinant().abs())
        .sum()
}

#[test]
fn d632_nc_pyramid_roundtrip_keeps_fuentes_table_and_volume() {
    let mut lin = unit_pyramid();
    lin.set_curvature(2);
    assert_eq!(lin.geom_order(), 2);
    assert_eq!(lin.geometry.as_ref().unwrap().nodes_per_elem, 15,
        "MFEM SetCurvature(2) pyramid table = 15 Fuentes slots");
    let curved = curved_pyramid(&lin);

    // Parent sanity: the Fuentes table reads back the identity map.
    let v0 = pyramid_volume(&curved, 0, 4);
    assert!(
        (v0 - 1.0 / 3.0).abs() < 1e-12,
        "parent unit pyramid ∫|det J| = {v0} (Fuentes identity: 1/3)"
    );

    // Empty mark list: a pure re-table roundtrip — the geometry must survive.
    let fine = refine_curved_3d_nc_general(&curved, &[]);
    assert_eq!(fine.n_elems, 1);

    let v = pyramid_volume(&fine, 0, 4);
    assert!(
        (v - 1.0 / 3.0).abs() < 1e-12,
        "roundtrip ∫|det J| = {v} — the Fuentes-15 table was silently \
         re-read through the 14-dof equispaced PyramidPk (D632)"
    );

    assert_eq!(
        fine.nodes_per_elem, 15,
        "order-2 curved pyramid table must keep the MFEM 15-slot contract (D613), got {}",
        fine.nodes_per_elem
    );
    assert_eq!(fine.elem_type, ElementType::Pyramid13);
    assert_eq!(fine.geom_order, 2);

    // Every re-interpolated non-vertex slot must sit exactly where the parent
    // table had it (identity map, nodal sampling).
    for j in 5..15 {
        let xf = fine.node_coords_arr(fine.geom_conn[j]);
        let xp = curved.node_coords_arr(curved.geom_conn[j]);
        let d: f64 = xf.iter().zip(xp.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(d < 1e-10, "slot {j}: roundtrip node moved by {d}");
    }
}
