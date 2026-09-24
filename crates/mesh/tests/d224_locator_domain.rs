//! D224 — locator reference-domain convention (library-level fix).
//!
//! Before this round `MeshTopology::locate` used the affine-simplex
//! `findpts::FindPoints` for every mesh: on quad meshes it only covered the
//! lower triangle of each cell (upper-triangle points returned `None`), and
//! on hex meshes the affine corner Jacobian is singular (columns
//! `v1-v0, v2-v0, v3-v0`), so *every* hex query failed.  The fix routes
//! tensor/prism meshes through the isoparametric `GslibFindPoints` and
//! translates the MFEM-canonical `[0, 1]^D` result to the fem_element
//! **factory domain** once at the exit (hex: `[-1, 1]^3`), so
//! `GridFunction::get_value` can evaluate the bases directly.
//!
//! Acceptance (task round 40 ②):
//! - `GridFunction::get_value` recovers known linear / quadratic / bilinear
//!   fields pointwise on hex and quad meshes to <= 1e-14 (measured below);
//! - simplex (tri/tet) paths keep the legacy affine-simplex search and its
//!   results bit-for-bit;
//! - `transformation::find_points` reports hex coordinates in `[-1, 1]^3`.

use fem_assembly::postproc::grid_function::GridFunction;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::transformation::{element_jacobian_at, find_points, ElementTransformation};
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// Max |get_value - f| over `pts`; counts located points.
fn max_error_3d(
    gf: &GridFunction<'_, H1Space<Mesh<3>>>,
    f: &dyn Fn(&[f64]) -> f64,
    pts: &[[f64; 3]],
) -> (usize, f64) {
    let mut n_found = 0;
    let mut max_err = 0.0_f64;
    for p in pts {
        if let Some(v) = gf.get_value(p) {
            n_found += 1;
            max_err = max_err.max((v - f(p)).abs());
        }
    }
    (n_found, max_err)
}

/// 2-D counterpart of [`max_error_3d`].
fn max_error_2d(
    gf: &GridFunction<'_, H1Space<Mesh<2>>>,
    f: &dyn Fn(&[f64]) -> f64,
    pts: &[[f64; 2]],
) -> (usize, f64) {
    let mut n_found = 0;
    let mut max_err = 0.0_f64;
    for p in pts {
        if let Some(v) = gf.get_value(p) {
            n_found += 1;
            max_err = max_err.max((v - f(p)).abs());
        }
    }
    (n_found, max_err)
}

/// n×n×n grid over the unit cube, values at the cell walls included — the
/// shared faces / corners exercise the multi-element containment choice.
fn cube_grid(n: usize) -> Vec<[f64; 3]> {
    let mut pts = Vec::with_capacity((n + 1) * (n + 1) * (n + 1));
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                pts.push([i as f64 / n as f64, j as f64 / n as f64, k as f64 / n as f64]);
            }
        }
    }
    pts
}

fn square_grid(n: usize) -> Vec<[f64; 2]> {
    let mut pts = Vec::with_capacity((n + 1) * (n + 1));
    for j in 0..=n {
        for i in 0..=n {
            pts.push([i as f64 / n as f64, j as f64 / n as f64]);
        }
    }
    pts
}

#[test]
fn d224_hex_get_value_recovers_linear_field_p1() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let space = H1Space::new(mesh, 1);
    let f = |x: &[f64]| 1.0 + 2.0 * x[0] - 3.0 * x[1] + 5.0 * x[2];
    let dofs = space.interpolate(&f);
    let gf = GridFunction::new(&space, dofs.as_slice().to_vec());

    // Old behavior: `locate` was None on every hex point (singular affine
    // Jacobian), so none of the points below was recoverable.
    let pts = cube_grid(6); // 7×7×7 incl. faces/corners
    let (n_found, max_err) = max_error_3d(&gf, &f, &pts);
    assert_eq!(n_found, pts.len(), "every grid point must be located");
    assert!(max_err <= 1e-14, "linear recovery error {max_err:e}");
}

#[test]
fn d224_hex_get_value_recovers_quadratic_field_p2() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let space = H1Space::new(mesh, 2);
    // Tensor degree (2,1,1) ⊂ Q2 on straight hexes: exact interpolation.
    let f = |x: &[f64]| 1.0 + x[0] * x[0] - 3.0 * x[1] * x[2];
    let dofs = space.interpolate(&f);
    let gf = GridFunction::new(&space, dofs.as_slice().to_vec());

    let pts = cube_grid(9); // 10×10×10
    let (n_found, max_err) = max_error_3d(&gf, &f, &pts);
    assert_eq!(n_found, pts.len(), "every grid point must be located");
    assert!(max_err <= 1e-14, "quadratic recovery error {max_err:e}");
}

#[test]
fn d224_hex_near_shared_face_points() {
    // Points epsilon-away from the interior wall x = 0.5 of unit_cube_hex(2):
    // the two sides sit in different elements and both must recover the field.
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let space = H1Space::new(mesh, 1);
    let f = |x: &[f64]| 2.0 * x[0] - x[2] + 0.5;
    let dofs = space.interpolate(&f);
    let gf = GridFunction::new(&space, dofs.as_slice().to_vec());

    for dx in [-1e-9_f64, 1e-9] {
        for p in [[0.5 + dx, 0.2, 0.3], [0.5 + dx, 0.49, 0.01], [0.5 + dx, 0.5, 0.5]] {
            let v = gf.get_value(&p).expect("near-face point must be located");
            let err = (v - f(&p)).abs();
            assert!(err <= 1e-14, "point {p:?}: error {err:e}");
        }
    }
}

#[test]
fn d224_quad_get_value_recovers_bilinear_field() {
    let mesh = Mesh::<2>::unit_square_quad(3);
    let space = H1Space::new(mesh, 1);
    let f = |x: &[f64]| 1.0 + 2.0 * x[0] - 3.0 * x[1] + 4.0 * x[0] * x[1];
    let dofs = space.interpolate(&f);
    let gf = GridFunction::new(&space, dofs.as_slice().to_vec());

    let mut pts = square_grid(9);
    // Explicit upper-triangle points: the pre-D224 affine search could not
    // locate any of them (only the corner triangle of each cell was covered).
    pts.push([0.3, 0.25]);
    pts.push([0.42, 0.41]);

    let (n_found, max_err) = max_error_2d(&gf, &f, &pts);
    assert_eq!(n_found, pts.len(), "every grid point must be located");
    assert!(max_err <= 1e-14, "bilinear recovery error {max_err:e}");
}

#[test]
fn d224_tet_get_value_unchanged_linear_recovery() {
    // All-simplex meshes keep the legacy affine search path (bit-identical
    // coordinates); this test pins the end-to-end semantics.
    let mesh = Mesh::<3>::unit_cube_tet(3);
    let space = H1Space::new(mesh, 1);
    let f = |x: &[f64]| 1.0 - 2.0 * x[0] + 3.0 * x[1] - 4.0 * x[2];
    let dofs = space.interpolate(&f);
    let gf = GridFunction::new(&space, dofs.as_slice().to_vec());

    let pts = cube_grid(6);
    let (n_found, max_err) = max_error_3d(&gf, &f, &pts);
    assert_eq!(n_found, pts.len());
    assert!(max_err <= 1e-12, "tet linear recovery error {max_err:e}");
}

#[test]
fn d224_tri_get_value_unchanged_linear_recovery() {
    let mesh = Mesh::<2>::unit_square_tri(4);
    let space = H1Space::new(mesh, 1);
    // A P1 space on a fixed triangulation reproduces *any* affine field.
    let f = |x: &[f64]| 0.5 + 2.0 * x[0] - 3.0 * x[1];
    let dofs = space.interpolate(&f);
    let gf = GridFunction::new(&space, dofs.as_slice().to_vec());

    let pts = square_grid(8);
    let (n_found, max_err) = max_error_2d(&gf, &f, &pts);
    assert_eq!(n_found, pts.len());
    assert!(max_err <= 1e-12, "tri linear recovery error {max_err:e}");
}

/// MFEM 4.10 pointwise comparison (probe `tmp/d230/d224_probe.cpp`, WSL):
/// same mesh MakeCartesian3D(2,1,1,HEX), same P1/P2 fields, same points.
/// Run with --ignored --nocapture and diff against `d224_probe_out.txt`.
#[test]
#[ignore = "comparison probe; run explicitly with --ignored --nocapture"]
fn d224_rs_eval_probe() {
    for (order, f) in [
        (1_usize, Box::new(|x: &[f64]| 1.0 + 2.0 * x[0] - 3.0 * x[1] + 5.0 * x[2]) as Box<dyn Fn(&[f64]) -> f64>),
        (2, Box::new(|x: &[f64]| 1.0 + x[0] * x[0] - 3.0 * x[1] * x[2])),
    ] {
        let mesh = Mesh::<3>::make_cartesian_3d(2, 1, 1, ElementType::Hex8, 1.0, 1.0, 1.0, true);
        let space = H1Space::new(mesh, order as u8);
        let dofs = space.interpolate(&f);
        let gf = GridFunction::new(&space, dofs.as_slice().to_vec());
        let pts: [[f64; 3]; 10] = [
            [0.1, 0.1, 0.1], [0.25, 0.25, 0.25], [0.4, 0.3, 0.2],
            [0.5 - 1e-9, 0.2, 0.3], [0.5 + 1e-9, 0.2, 0.3],
            [0.5, 0.5, 0.5], [0.75, 0.25, 0.1], [0.6, 0.7, 0.8],
            [0.85, 0.55, 0.35], [1.0 - 1e-12, 1.0 - 1e-12, 1.0 - 1e-12]];
        println!("H1-P{order} Eval (MakeCartesian3D(2,1,1,HEX)):");
        for p in pts {
            let val = gf.get_value(&p).expect("point must be located");
            println!("  pt ({}, {}, {}) -> val {val:.17}", p[0], p[1], p[2]);
        }
    }
}

#[test]
#[ignore = "temporary probe retained for future locator debugging"]
fn d224_tri_probe() {
    let mesh = Mesh::<2>::unit_square_tri(4);
    let mt: &dyn MeshTopology = &mesh;
    let space = H1Space::new(mesh.clone(), 1);
    let f = |x: &[f64]| 0.5 + 2.0 * x[0] - 3.0 * x[1];
    let dofs = space.interpolate(&f);
    let gf = GridFunction::new(&space, dofs.as_slice().to_vec());
    for j in 0..=8 {
        for i in 0..=8 {
            let p = [i as f64 / 8.0, j as f64 / 8.0];
            if let Some((e, xi)) = mt.locate(&p, 1e-10) {
                let v = gf.get_value(&p).unwrap();
                if (v - f(&p)).abs() > 1e-10 {
                    println!("p {p:?} -> elem {e} xi {xi:?} v {v} f {}", f(&p));
                }
            } else {
                println!("p {p:?} -> NOT FOUND");
            }
        }
    }
}

#[test]
fn d224_find_points_hex_factory_domain() {
    // D721: `transformation::find_points` reports hex coordinates in
    // MFEM's `[0,1]^3` frame.  Single-cell mesh so the local coordinates are
    // the physical ones.
    let mesh = Mesh::<3>::make_cartesian_3d(1, 1, 1, ElementType::Hex8, 1.0, 1.0, 1.0, true);
    let pts = vec![0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.25, 0.25, 0.25];
    let (ids, xis) = find_points(&mesh, &pts, 3);
    assert!(ids.iter().all(|&e| e >= 0));
    for (i, xi) in xis.iter().enumerate() {
        assert_eq!(xi.len(), 3);
        for d in 0..3 {
            let expected = pts[i * 3 + d];
            assert!(
                (xi[d] - expected).abs() < 1e-12,
                "point {i} axis {d}: {} vs {expected}",
                xi[d]
            );
        }
        // Roundtrip: the physical map at the factory coordinates reproduces
        // the query point (isoparametric straight-geometry path).
        let (_j, xp) = element_jacobian_at(&mesh, ids[i] as u32, xi, 3);
        for d in 0..3 {
            assert!((xp[d] - pts[i * 3 + d]).abs() < 1e-12);
        }
    }

    // Multi-element mesh: every coordinate is local to the found element.
    let mesh2 = Mesh::<3>::unit_cube_hex(2);
    let pts2 = vec![0.1, 0.2, 0.3, 0.6, 0.7, 0.8];
    let (ids2, xis2) = find_points(&mesh2, &pts2, 2);
    assert!(ids2.iter().all(|&e| e >= 0));
    for (i, xi) in xis2.iter().enumerate() {
        let ns = mesh2.elem_nodes(ids2[i] as u32);
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for &n in ns {
            let c = mesh2.coords_of(n);
            for d in 0..3 {
                lo[d] = lo[d].min(c[d]);
                hi[d] = hi[d].max(c[d]);
            }
        }
        let (_j, xp) = element_jacobian_at(&mesh2, ids2[i] as u32, xi, 3);
        for d in 0..3 {
            assert!((xp[d] - pts2[i * 3 + d]).abs() < 1e-12, "pt {i} axis {d}");
            // D721: hex factory coordinates are MFEM's [0,1]^3 locals.
            assert!(xi[d] >= -1e-9 && xi[d] <= 1.0 + 1e-9);
            let expected = (pts2[i * 3 + d] - lo[d]) / (hi[d] - lo[d]);
            assert!((xi[d] - expected).abs() < 1e-12, "pt {i} axis {d} local");
        }
    }
}

#[test]
fn d224_locate_outside_is_none() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let mt: &dyn MeshTopology = &mesh;
    assert!(mt.locate(&[1.5, 0.2, 0.3], 1e-10).is_none());
    assert!(mt.locate(&[0.2, -0.3, 0.3], 1e-10).is_none());
    let inside = mt.locate(&[0.2, 0.3, 0.4], 1e-10).expect("inside");
    // D721: hex factory domain is MFEM's [0, 1]^3.
    for v in &inside.1 {
        assert!(*v >= -1e-9 && *v <= 1.0 + 1e-9, "hex xi {inside:?}");
    }
}

#[test]
fn d224_from_simplex_quad_bilinear_identity() {
    // D230 companion: ElementTransformation on Quad4 is the true bilinear map
    // (the warped-element proof lives in d230_quad_bilinear.rs).  On the unit
    // square the bilinear map is the identity.
    let mesh = Mesh::<2>::make_cartesian_2d(1, 1, 1.0, 1.0);
    let tr = ElementTransformation::from_simplex(&mesh, 0);
    let x = tr.map_to_physical(&[0.5, 0.25]);
    assert!((x[0] - 0.5).abs() < 1e-15 && (x[1] - 0.25).abs() < 1e-15);
}
