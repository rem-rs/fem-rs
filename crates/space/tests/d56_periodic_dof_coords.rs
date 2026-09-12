//! D56 regression: on geometrically periodic meshes the DOF coordinates of
//! `DofManager` (and hence `H1Space::interpolate` / `VectorH1Space::interpolate_vec`)
//! were built from the **folded** vertex table, so DOFs on seam elements land at
//! the wrong physical position and any non-constant nodal interpolation is wrong.
//!
//! MFEM semantics (the authority): a periodic mesh keeps per-element geometry
//! (each element's own `Nodes` values), and `GridFunction::ProjectCoefficient`
//! evaluates the coefficient **per element** at each local DOF's nodal point in
//! that element's own transform, writing shared DOFs last-writer-wins.  These
//! tests replicate that reference projection and require `interpolate` to agree.

use fem_element::lagrange::factory::{HexQk, QuadQk};
use fem_element::lagrange::{H1TriPk, TetPk};
use fem_element::ReferenceElement;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::dof_manager::DofManager;
use fem_space::{H1Space, VectorH1Space};

/// The MFEM `ProjectCoefficient` replica: for every element, map each local
/// slot's reference dof position through the element's **own** geometry
/// (order-`geom_order` snapshot on periodic meshes: affine for order-1,
/// the high-order geometry basis for curved periodic meshes) and evaluate
/// `f` there; shared DOFs are overwritten in element order (last writer
/// wins).
fn project_coefficient_replica<M: MeshTopology>(
    mesh: &M,
    dm: &DofManager,
    f: &dyn Fn(&[f64]) -> f64,
) -> Vec<f64> {
    use fem_element::lagrange::factory::{HexQk, QuadQk};

    let dim = mesh.dim() as usize;
    let geom_order = mesh.geom_order() as usize;
    let mut v = vec![0.0_f64; dm.n_dofs];
    for e in 0..mesh.n_elements() as u32 {
        let p = dm.element_order(e) as usize;
        let (rf, gf): (Box<dyn ReferenceElement>, Box<dyn ReferenceElement>) =
            match mesh.element_type(e) {
                fem_mesh::ElementType::Quad4 => (
                    Box::new(QuadQk::new(p)),
                    Box::new(QuadQk::new(geom_order)),
                ),
                fem_mesh::ElementType::Hex8 => (
                    Box::new(HexQk::new(p)),
                    Box::new(HexQk::new(geom_order)),
                ),
                fem_mesh::ElementType::Tri3 => (
                    Box::new(H1TriPk::new(p)),
                    Box::new(H1TriPk::new(geom_order)),
                ),
                fem_mesh::ElementType::Tet4 => (
                    Box::new(TetPk::new(p)),
                    Box::new(TetPk::new(geom_order)),
                ),
                t => panic!("replica: unsupported element type {t:?} order {p}"),
            };
        let gnodes = mesh.geometry_nodes(e);
        let corners: Vec<Vec<f64>> = gnodes
            .iter()
            .map(|&g| mesh.geom_coords_of(g).to_vec())
            .collect();
        let ref_dofs = rf.dof_coords();
        let dofs = dm.element_dofs(e);
        assert_eq!(dofs.len(), ref_dofs.len(), "slot count vs factory");
        let mut phi = vec![0.0_f64; corners.len()];
        for (slot, rc) in ref_dofs.iter().enumerate() {
            gf.eval_basis(rc, &mut phi);
            let mut x = vec![0.0_f64; dim];
            for (k, &p) in phi.iter().enumerate() {
                for d in 0..dim {
                    x[d] += p * corners[k][d];
                }
            }
            v[dofs[slot] as usize] = f(&x);
        }
    }
    v
}

fn elem_type_of(mesh: &Mesh<2>) -> fem_mesh::ElementType {
    mesh.element_type(0)
}

/// Every DOF value of `interpolate` on a periodic quad mesh must match the
/// per-element projection (currently fails at seam DOFs with O(1) error).
#[test]
fn h1_interpolate_matches_per_element_projection_on_periodic_quad_mesh() {
    let base = Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0);
    let mesh = base
        .make_periodic(&[(4, 2, [1.0, 0.0]), (1, 3, [0.0, 1.0])], 1e-10)
        .unwrap();
    assert_eq!(elem_type_of(&mesh), fem_mesh::ElementType::Quad4);

    for order in [1u8, 2, 3, 4] {
        let space = H1Space::new(mesh.clone(), order);
        let f = |x: &[f64]| (2.0 * std::f64::consts::PI * x[0]).sin()
            * (2.0 * std::f64::consts::PI * x[1]).sin();
        let v = space.interpolate(&f);
        let replica = project_coefficient_replica(&mesh, space.dof_manager(), &f);
        let mut max_diff = 0.0_f64;
        for (a, b) in v.as_slice().iter().zip(replica.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff < 1e-12,
            "order {order}: interpolate vs per-element projection max |diff| = {max_diff:.3e}"
        );
    }
}

/// Same on the **unfolded** mesh (no periodic merge): interpolate must stay
/// exactly the nodal interpolant (guards the fix against regressing the
/// non-periodic path).
#[test]
fn h1_interpolate_matches_per_element_projection_on_plain_quad_mesh() {
    let mesh = Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0);
    for order in [1u8, 2, 3, 4] {
        let space = H1Space::new(mesh.clone(), order);
        let f = |x: &[f64]| (3.0 * x[0] + 1.0).sin() * x[1];
        let v = space.interpolate(&f);
        let replica = project_coefficient_replica(&mesh, space.dof_manager(), &f);
        for (a, b) in v.as_slice().iter().zip(replica.iter()) {
            assert!((a - b).abs() < 1e-12, "order {order}: |{a} - {b}|");
        }
    }
}

/// Triangles, periodic in x: seam vertex/edge DOFs.
#[test]
fn h1_interpolate_matches_per_element_projection_on_periodic_tri_mesh() {
    let base = Mesh::<2>::unit_square_tri(4);
    // unit_square_tri tags: 1 = ?, 2 = ?, 3 = bottom, 4 = top — use the same
    // pairs as the quad mesh (left/right = 1/2? here we probe with the pair
    // list that the mesh actually exposes).
    let mesh = base
        .make_periodic(&[(4, 2, [1.0, 0.0]), (1, 3, [0.0, 1.0])], 1e-10)
        .unwrap();
    for order in [1u8, 2, 3] {
        let space = H1Space::new(mesh.clone(), order);
        let f = |x: &[f64]| (2.0 * std::f64::consts::PI * x[0]).sin()
            * (2.0 * std::f64::consts::PI * x[1]).sin();
        let v = space.interpolate(&f);
        let replica = project_coefficient_replica(&mesh, space.dof_manager(), &f);
        let mut max_diff = 0.0_f64;
        for (a, b) in v.as_slice().iter().zip(replica.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff < 1e-12,
            "tri order {order}: interpolate vs per-element projection max |diff| = {max_diff:.3e}"
        );
    }
}

/// `VectorH1Space::interpolate_vec` — the entry navier_shear needs.
#[test]
fn vector_h1_interpolate_vec_matches_per_element_projection_on_periodic_mesh() {
    let base = Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0);
    let mesh = base
        .make_periodic(&[(4, 2, [1.0, 0.0]), (1, 3, [0.0, 1.0])], 1e-10)
        .unwrap();
    for order in [1u8, 2, 4] {
        let space = VectorH1Space::new(mesh.clone(), order, 2);
        let f = |x: &[f64]| {
            vec![
                (2.0 * std::f64::consts::PI * x[0]).sin() * (2.0 * std::f64::consts::PI * x[1]).cos(),
                (2.0 * std::f64::consts::PI * x[0]).cos() * (2.0 * std::f64::consts::PI * x[1]).sin(),
            ]
        };
        let v = space.interpolate_vec(&f);
        let n_scalar = space.n_scalar_dofs();
        let fx = |x: &[f64]| f(x)[0];
        let fy = |x: &[f64]| f(x)[1];
        let rx = project_coefficient_replica(&mesh, space.scalar_dof_manager(), &fx);
        let ry = project_coefficient_replica(&mesh, space.scalar_dof_manager(), &fy);
        for dof in 0..n_scalar {
            assert!(
                (v.as_slice()[dof] - rx[dof]).abs() < 1e-12,
                "order {order}: x-component dof {dof}"
            );
            assert!(
                (v.as_slice()[n_scalar + dof] - ry[dof]).abs() < 1e-12,
                "order {order}: y-component dof {dof}"
            );
        }
    }
}

/// The seam itself: on the 4x1 periodic strip the right boundary column is
/// merged into the left one; the H1 dof sitting there must be evaluated at the
/// physical x = 1 image by at least one element (i.e. the folded table must not
/// be the only coordinate source).  With f(x) = x the correct projection value
/// at the merged column is 1.0 or 0.0 (both are true images); the old folded
/// table instead placed *interior* seam dofs at chord midpoints (x = 0.5-ish),
/// which no element can see.
#[test]
fn seam_element_interior_dofs_do_not_land_on_folded_chords() {
    let base = Mesh::<2>::make_cartesian_2d(4, 1, 1.0, 1.0);
    let mesh = base.make_periodic(&[(4, 2, [1.0, 0.0])], 1e-10).unwrap();
    let order = 3u8;
    let space = H1Space::new(mesh.clone(), order);
    let f = |x: &[f64]| x[0];
    let v = space.interpolate(&f);
    let replica = project_coefficient_replica(&mesh, space.dof_manager(), &f);
    for (a, b) in v.as_slice().iter().zip(replica.iter()) {
        assert!((a - b).abs() < 1e-12, "|{a} - {b}|");
    }
}

// ─── D62: periodic + curved geometry ─────────────────────────────────────────

/// Build a **curved periodic** quad mesh: a 4×4 grid whose order-2 geometry
/// is displaced by a smooth periodic bump, then made periodic.  The periodic
/// merge keeps the pre-merge high-order geometry as the per-element snapshot,
/// so seam DOFs must be evaluated through the curved mapping, not the folded
/// vertex table.
fn curved_periodic_quad_mesh() -> Mesh<2> {
    let mut base = Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0);
    base.set_curvature(2);
    // Displace the geometry nodes periodically (the displacement vanishes on
    // the domain boundary, so seam images are displaced consistently and the
    // merged mesh is genuinely continuous across the seam).
    {
        let geo = base.geometry.as_mut().unwrap();
        for n in 0..geo.n_nodes {
            let x = geo.coords[n * 2];
            let y = geo.coords[n * 2 + 1];
            geo.coords[n * 2] =
                x + 0.05 * (2.0 * std::f64::consts::PI * x).sin() * (2.0 * std::f64::consts::PI * y).sin();
            geo.coords[n * 2 + 1] =
                y + 0.04 * (2.0 * std::f64::consts::PI * x).sin() * (2.0 * std::f64::consts::PI * y).cos();
        }
    }
    let mesh = base
        .make_periodic(&[(4, 2, [1.0, 0.0]), (1, 3, [0.0, 1.0])], 1e-10)
        .unwrap();
    assert_eq!(mesh.geom_order(), 2, "curved periodic mesh keeps its geometry order");
    assert!(mesh.n_nodes() < 25, "vertices must be merged across the seams");
    mesh
}

/// D62: on a curved periodic mesh the DOF coordinates must be evaluated
/// through the element's own **curved** geometry (order-2 basis over the full
/// geometry node list) — the old gate (`geom_order == 1`) skipped the rebuild
/// entirely and left seam DOFs at the collapsed folded positions.
#[test]
fn h1_interpolate_matches_curved_per_element_projection_on_periodic_curved_mesh() {
    let mesh = curved_periodic_quad_mesh();
    let f = |x: &[f64]| (2.0 * std::f64::consts::PI * x[0]).sin() * (3.0 * x[1] + 1.0).cos();
    for order in [1u8, 2, 3, 4] {
        let space = H1Space::new(mesh.clone(), order);
        let v = space.interpolate(&f);
        let replica = project_coefficient_replica(&mesh, space.dof_manager(), &f);
        let mut max_diff = 0.0_f64;
        for (a, b) in v.as_slice().iter().zip(replica.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff < 1e-12,
            "order {order}: interpolate vs curved per-element projection max |diff| = {max_diff:.3e}"
        );
    }
}

/// The D62 gate is load-bearing: the curved displacement actually moves the
/// DOF positions away from the straight grid — i.e. without the rebuild the
/// coordinates would silently be the un-displaced folded values, and the
/// interpolation test above would fail with O(0.05) error.
#[test]
fn d62_curved_periodic_dof_positions_are_displaced() {
    let mesh = curved_periodic_quad_mesh();
    let space = H1Space::new(mesh.clone(), 2);
    let dm = space.dof_manager();
    let mut max_disp = 0.0_f64;
    for dof in 0..space.n_dofs() as u32 {
        let c = dm.dof_coord(dof);
        // Distance to the straight-grid lattice (spacing 0.25).
        for (x, gx) in c.iter().zip([0.25_f64; 2]) {
            let nearest = (x / gx).round() * gx;
            max_disp = max_disp.max((x - nearest).abs());
        }
    }
    assert!(max_disp > 1.0e-3, "DOFs are still on the straight grid: {max_disp:.3e}");
}
