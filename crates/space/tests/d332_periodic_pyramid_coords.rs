//! D332 — `DofManager::rebuild_dof_coords_periodic`'s **pyramid field arm**
//! (`crates/space/src/dof_manager.rs`).
//!
//! The pyramid arm must place the field slots at the MFEM Bergot
//! `H1PyramidPk` reference positions (the entity-order table `build_pyramid_pk`
//! numbers `element_dofs` in, D191/D299) and evaluate the *geometry* through
//! the element's own periodic geometry table.  The pre-fix arm built the field
//! slot positions from the **integer label grid** `(i, j, k)/p` instead: that
//! lattice coincides with the Gauss–Lobatto one for `p ≤ 2` and drifts from
//! `p = 3` (measured max 0.068670 at `p = 3`, 0.199682 at `p = 4`), and the
//! `dofs.len() == ref_dofs.len()` guard could not see it because both tables
//! have the same length.
//!
//! Second half of the same defect surface (D331's release path): a *straight*
//! pyramid keeps its geometry table in MFEM **vertex** order, so the
//! layer-ordered `PyramidPk(1)` basis must be fed the `P1_SLOT_VERTEX`
//! permuted corner list (`assembler::GeoPyrP1`, D304) — the old arm evaluated
//! it against the unpermuted list and swapped the two base corners of every
//! straight periodicity snapshot.
//!
//! The oracle is the independent replica in `coordinate_replica` below
//! (`H1PyramidPk::dof_coords()` for the field slots, the permuted layer basis
//! for the geometry).  On an element whose geometry *is* the reference pyramid
//! the replica collapses to exactly `H1PyramidPk::dof_coords()` — pinned by
//! `replica_of_the_unit_pyramid_is_the_h1_pyramid_pk_lattice`.  On the merged
//! periodic mesh the shared (identified) vertex dofs carry the last writer's
//! coordinate, exactly as MFEM's `SetCurvature` + `ProjectCoefficient` does,
//! so the per-element replica — not a global equality with the reference
//! lattice — is the well-posed criterion there.

use fem_element::lagrange::pyramid::{H1PyramidPk, PyramidPk};
use fem_element::ReferenceElement;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::dof_manager::DofManager;

/// `PyramidPk(1)`'s layer slot `k` carries the shape function of mesh vertex
/// `P1_SLOT_VERTEX[k]` (D191).
const P1_SLOT_VERTEX: [usize; 5] = [0, 1, 3, 2, 4];

/// Straight unit pyramid `(0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1)`.
fn unit_pyramid() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 1.],
        vec![0, 1, 2, 3, 4],
        vec![1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// Unit cube `[0,1]³` split into its 6 faces-as-base pyramids (apex = the cube
/// centre), periodic in x: the `x = 0` face (tag 4) merges into `x = 1`
/// (tag 2) under the translation `(1, 0, 0)`.
///
/// Nodes: `0..7` = the cube corners (`z = 0` then `z = 1`), `8` = the centre.
/// The periodic merge identifies `v0 → v1` and `v3 → v2`, so the elements keep
/// a per-element geometry snapshot (D56) while the mesh topology is the
/// quotient.
fn periodic_cube_pyramids() -> Mesh<3> {
    let coords = vec![
        0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., // z = 0
        0., 0., 1., 1., 0., 1., 1., 1., 1., 0., 1., 1., // z = 1
        0.5, 0.5, 0.5, // 8: apex of every pyramid
    ];
    let conn = vec![
        0, 1, 2, 3, 8, // bottom  (z = 0)
        4, 5, 6, 7, 8, // top     (z = 1)
        0, 1, 5, 4, 8, // front   (y = 0)
        1, 2, 6, 5, 8, // right   (x = 1)
        2, 3, 7, 6, 8, // back    (y = 1)
        3, 0, 4, 7, 8, // left    (x = 0)
    ];
    // Two triangles per cube face; tags 4 (x = 0) and 2 (x = 1) are the
    // periodic pair, every other face keeps tag 1.
    let faces = vec![
        0, 1, 2, 0, 2, 3, // z = 0
        4, 5, 6, 4, 6, 7, // z = 1
        0, 1, 5, 0, 5, 4, // y = 0
        2, 3, 7, 2, 7, 6, // y = 1
        3, 0, 4, 3, 4, 7, // x = 0  (tag 4)
        1, 2, 6, 1, 6, 5, // x = 1  (tag 2)
    ];
    let face_tags = vec![1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 2, 2];
    assert_eq!(faces.len() / 3, face_tags.len());
    let m = Mesh::<3>::uniform(
        coords,
        conn,
        vec![1; 6],
        ElementType::Pyramid5,
        faces,
        face_tags,
        ElementType::Tri3,
    );
    m.make_periodic(&[(4, 2, [1.0, 0.0, 0.0])], 1e-10)
        .expect("periodic merge of the pyramid cube")
}

/// The MFEM `ProjectCoefficient` replica for pyramid dof coordinates: for
/// every element, map each **field** slot's reference point
/// (`H1PyramidPk::dof_coords()`, the table `build_pyramid_pk` numbers) through
/// the element's own geometry snapshot and overwrite the global entry — last
/// writer wins.  Straight pyramids (`geom_order == 1`) keep the snapshot in
/// MFEM vertex order, so the layer-ordered `PyramidPk(1)` basis is fed the
/// `P1_SLOT_VERTEX`-permuted corner list.
fn coordinate_replica(mesh: &Mesh<3>, dm: &DofManager) -> Vec<f64> {
    let dim = 3usize;
    let geom_order = mesh.geom_order() as usize;
    let mut expected = vec![f64::NAN; dm.n_dofs * dim];
    for e in 0..mesh.n_elements() as u32 {
        let p = dm.element_order(e) as usize;
        let field_rc = H1PyramidPk::new(p).dof_coords();
        let geom = PyramidPk::new(geom_order);
        let dofs = dm.element_dofs(e).to_vec();
        assert_eq!(
            dofs.len(),
            field_rc.len(),
            "element {e}: build_pyramid_pk slot count vs H1PyramidPk"
        );
        let gnodes = mesh.geometry_nodes(e);
        let straight = geom_order == 1;
        let corner = |k: usize| -> &[f64] {
            let idx = if straight { P1_SLOT_VERTEX[k] } else { k };
            mesh.geom_coords_of(gnodes[idx])
        };
        let mut phi = vec![0.0_f64; gnodes.len()];
        for (slot, rc) in field_rc.iter().enumerate() {
            geom.eval_basis(rc, &mut phi);
            let mut x = [0.0_f64; 3];
            for (k, &phik) in phi.iter().enumerate() {
                if phik == 0.0 {
                    continue;
                }
                let ck = corner(k);
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

/// (worst |diff|, worst dof, worst component) between `DofManager`'s table and
/// the replica.
fn coordinate_max_diff(mesh: &Mesh<3>, dm: &DofManager) -> (f64, u32) {
    let expected = coordinate_replica(mesh, dm);
    for (i, v) in expected.iter().enumerate() {
        assert!(
            v.is_finite(),
            "the replica never reached dof {} (NaN) — a dof is not covered by any element",
            i / 3,
        );
    }
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

/// Oracle sanity: on an element whose geometry is the reference pyramid the
/// replica returns exactly `H1PyramidPk::dof_coords()` — so the periodic-mesh
/// checks below really are "the field dof coordinates are the `H1PyramidPk`
/// lattice mapped by each element's own (linear) pyramid geometry".
#[test]
fn replica_of_the_unit_pyramid_is_the_h1_pyramid_pk_lattice() {
    let mesh = unit_pyramid();
    for p in 2..=4usize {
        let dm = DofManager::new(&mesh, p as u8);
        let dofs = dm.element_dofs(0).to_vec();
        let want = H1PyramidPk::new(p).dof_coords();
        let replica = coordinate_replica(&mesh, &dm);
        for (slot, rc) in want.iter().enumerate() {
            let b = dofs[slot] as usize * 3;
            for d in 0..3 {
                assert!(
                    (replica[b + d] - rc[d]).abs() <= 1e-15,
                    "p{p} slot {slot}: replica {:?} != H1PyramidPk::dof_coords() {rc:?}",
                    &replica[b..b + 3],
                );
            }
        }
    }
}

/// Every dof coordinate on a periodic straight pyramid mesh must be the
/// per-element image of its own `H1PyramidPk` slot position (≤ 1e-13),
/// `p = 2..4`.
#[test]
fn periodic_pyramid_dof_coords_follow_h1_pyramid_pk_slots() {
    let mesh = periodic_cube_pyramids();
    assert_eq!(mesh.n_elements(), 6, "mesh sanity: 6 pyramids");
    for order in [2u8, 3, 4] {
        let dm = DofManager::new(&mesh, order);
        let (worst, worst_dof) = coordinate_max_diff(&mesh, &dm);
        assert!(
            worst <= 1e-13,
            "order {order}: dof {worst_dof} coordinate off by {worst:.3e} (got {:?}) — the \
             periodic pyramid field arm is not evaluating at H1PyramidPk slots through the \
             vertex-ordered geometry snapshot",
            dm.dof_coord(worst_dof),
        );
    }
}

/// End to end: `H1Space::interpolate` on the periodic pyramid mesh must equal
/// the projection replica evaluated at the corrected coordinates.
#[test]
fn h1_interpolate_matches_projection_on_periodic_pyramid_mesh() {
    use fem_space::FESpace;
    use fem_space::H1Space;
    let mesh = periodic_cube_pyramids();
    for order in [2u8, 3] {
        let space = H1Space::new(mesh.clone(), order);
        let f = |x: &[f64]| (2.0 * std::f64::consts::PI * x[0]).sin() * x[1] + 0.5 * x[2];
        let v = space.interpolate(&f);
        let expected_coords = coordinate_replica(&mesh, space.dof_manager());
        let mut worst = 0.0_f64;
        for dof in 0..space.dof_manager().n_dofs {
            let c = &expected_coords[dof * 3..dof * 3 + 3];
            worst = worst.max((v.as_slice()[dof] - f(c)).abs());
        }
        assert!(
            worst < 1e-12,
            "order {order}: interpolate vs per-element projection max |diff| = {worst:.3e}"
        );
    }
}
