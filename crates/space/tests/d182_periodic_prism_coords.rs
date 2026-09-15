//! D182 regression: `DofManager::rebuild_dof_coords_periodic`'s **prism field
//! arm** must evaluate at `H1PrismPk`'s reference points — the MFEM entity
//! slot order `build_prism_h1` numbers `element_dofs` in — while the
//! **geometry** arm stays on `PrismPk`, the layer-major order
//! `Mesh::set_curvature_prism6` writes its geometry table in
//! (`crates/mesh/tests/d152_prism_curvature.rs`).
//!
//! The pre-fix code used layer-major `PrismPk::new(p)` for the *field* too:
//! the two elements share the same Gauss-Lobatto lattice and the same dof
//! count, so the "slot-count mismatch ⇒ keep fold-based coordinates" guard
//! never fired and the wrong slot→dof mapping silently produced wrong
//! coordinates for every periodic prism mesh at order ≥ 2 (slot 3 of
//! `PrismPk` is the bottom-edge e01 node, slot 3 of `H1PrismPk` is the top
//! vertex v3 — from there on the two tables enumerate different entities).
//!
//! The check is the exact `ProjectCoefficient`-coordinate replica: per
//! element, map each H1 slot's reference point through the element's own
//! (periodic) geometry snapshot, overwrite the global dof entry, last writer
//! wins — then require `DofManager` to agree slot-for-slot.

use fem_element::lagrange::{H1PrismPk, PrismPk};
use fem_element::ReferenceElement;
use fem_mesh::extrusion::extrude_tri3_to_prisms;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::dof_manager::DofManager;

/// Two-prism mesh (unit square, one triangle split, 1 extrusion layer),
/// periodic in x: left boundary (tag 4) merged into right (tag 2) under the
/// translation (1, 0, 0).  The periodic merge keeps the per-element geometry
/// snapshot, so seam vertices/edges/faces are shared DOFs whose coordinates
/// only the per-element rebuild can place.
fn periodic_prism_mesh() -> Mesh<3> {
    let base = extrude_tri3_to_prisms(&Mesh::<2>::unit_square_tri(1), 1, 1.0);
    base.make_periodic(&[(4, 2, [1.0, 0.0, 0.0])], 1e-10)
        .expect("periodic merge")
}

/// Same mesh with a curved (order-2) geometry snapshot taken *before* the
/// periodic merge — MFEM `MakePeriodic` materializes `Nodes` first and keeps
/// it, so the per-element geometry is the curved one.
fn curved_periodic_prism_mesh() -> Mesh<3> {
    let mut base = extrude_tri3_to_prisms(&Mesh::<2>::unit_square_tri(1), 1, 1.0);
    base.set_curvature(2);
    base.make_periodic(&[(4, 2, [1.0, 0.0, 0.0])], 1e-10)
        .expect("periodic merge")
}

/// The MFEM `ProjectCoefficient` replica for coordinates: for every element,
/// map each **field** slot's reference point (`H1PrismPk`, the order
/// `build_prism_h1` numbers) through the element's own geometry
/// (`PrismPk`-ordered table, `set_curvature_prism6`'s contract) and overwrite
/// the global entry — last writer wins.  Returns the expected flat coordinate
/// table (`n_dofs × 3`).
fn coordinate_replica(mesh: &Mesh<3>, dm: &DofManager) -> Vec<f64> {
    let dim = 3usize;
    let geom_order = mesh.geom_order() as usize;
    let mut expected = vec![f64::NAN; dm.n_dofs * dim];
    for e in 0..mesh.n_elements() as u32 {
        let p = dm.element_order(e) as usize;
        let field = H1PrismPk::new(p);
        let geom = PrismPk::new(geom_order);
        let dofs = dm.element_dofs(e).to_vec();
        let ref_points = field.dof_coords();
        assert_eq!(
            dofs.len(),
            ref_points.len(),
            "element {e}: build_prism_h1 slot count vs H1PrismPk"
        );
        let gnodes = mesh.geometry_nodes(e);
        let corners: Vec<&[f64]> = gnodes.iter().map(|&g| mesh.geom_coords_of(g)).collect();
        let mut phi = vec![0.0_f64; corners.len()];
        for (slot, rc) in ref_points.iter().enumerate() {
            geom.eval_basis(rc, &mut phi);
            let mut x = [0.0_f64; 3];
            for (k, &phik) in phi.iter().enumerate() {
                if phik == 0.0 {
                    continue;
                }
                let ck = corners[k];
                for d in 0..dim {
                    x[d] += phik * ck[d];
                }
            }
            let b = dofs[slot] as usize * dim;
            expected[b..b + dim].copy_from_slice(&x);
        }
    }
    expected
}

/// (worst |diff|, worst dof) between `DofManager`'s table and the replica.
fn coordinate_max_diff(mesh: &Mesh<3>, dm: &DofManager) -> (f64, u32) {
    let expected = coordinate_replica(mesh, dm);
    let mut worst = 0.0_f64;
    let mut worst_dof = 0u32;
    for dof in 0..dm.n_dofs as u32 {
        let got = dm.dof_coord(dof);
        let b = dof as usize * 3;
        for d in 0..3 {
            let diff = (got[d] - expected[b + d]).abs();
            if diff > worst {
                worst = diff;
                worst_dof = dof;
            }
        }
    }
    (worst, worst_dof)
}

/// Every periodic-prism dof coordinate must be the per-element image of the
/// dof's own `H1PrismPk` reference point (≤ 1e-13; the mapping is piecewise
/// polynomial in double precision).
#[test]
fn periodic_prism_dof_coords_follow_h1_prism_pk_slots() {
    for (name, mesh) in [("affine", periodic_prism_mesh()), ("curved", curved_periodic_prism_mesh())]
    {
        for order in [2u8, 3, 4] {
            let dm = DofManager::new(&mesh, order);
            let (worst, worst_dof) = coordinate_max_diff(&mesh, &dm);
            assert!(
                worst <= 1e-13,
                "{name} order {order}: dof {worst_dof} coordinate off by {worst:.3e} — \
                 the periodic prism field arm is not evaluating at H1PrismPk slots"
            );
        }
    }
}

/// End to end: `H1Space::interpolate` on the curved periodic prism mesh must
/// equal the projection replica evaluated at the corrected coordinates.
#[test]
fn h1_interpolate_matches_projection_on_periodic_prism_mesh() {
    use fem_space::FESpace;
    use fem_space::H1Space;
    let mesh = curved_periodic_prism_mesh();
    for order in [2u8, 3] {
        let space = H1Space::new(mesh.clone(), order);
        let f = |x: &[f64]| {
            (2.0 * std::f64::consts::PI * x[0]).sin() * x[1] * (1.0 + x[2])
        };
        let v = space.interpolate(&f);
        // Replica: the projection must evaluate f at the *replica* coordinate
        // table (independent of DofManager's own table).
        let expected_coords = coordinate_replica(&mesh, space.dof_manager());
        let mut worst = 0.0_f64;
        for dof in 0..space.dof_manager().n_dofs {
            let c = &expected_coords[dof * 3..dof * 3 + 3];
            let diff = (v.as_slice()[dof] - f(c)).abs();
            worst = worst.max(diff);
        }
        assert!(
            worst < 1e-12,
            "order {order}: interpolate vs per-element projection max |diff| = {worst:.3e}"
        );
    }
}
