//! D787 — **consumer-level** evidence: the L² error estimators that route
//! through `fem_mesh::transformation::element_jacobian_at` on curved simplices.
//!
//! `compute_l2_error_l2` (and its `_hcurl`/`_hdiv` siblings) selects its
//! geometry map with a `use_iso` list of cell types
//! (`grid_function.rs:2444`): `Quad4/Quad8/Quad9`, `Hex8/Hex20/Hex27`,
//! `Prism6/Prism15`, `Pyramid5`.  `Tri3/Tri6` — and, in the sibling arms,
//! `Tet4/Tet10` — are **not** in the list, so those cells take the
//! `element_jacobian_at` fall-through.  Before D787 that entry point had no
//! isoparametric branch for simplices, so a curved triangle/tet was integrated
//! over its *straight* corner map while the field itself was placed with the
//! curved one — exactly the geometry/measure mismatch `compute_l2_error_l2`'s
//! own doc comment warns about ("the corner-difference fallback is exact for
//! affine simplices" — it is not for curved ones).
//!
//! Fixture and truth: the same `set_curvature(2)` + D319 quadratic warp mesh as
//! `tests/d787_curved_simplex_geometry.rs`, with the exact field
//! `u = 1 + x + 2y (+ 3z)` and an L² P0 field that is identically 1.  MFEM's
//! side is `L2_FECollection(0, dim)` + `GridFunction gf = 1.0` +
//! `ComputeL2Error(u, irs)` with `irs[geom] = IntRules.Get(geom, 12)` from
//! `tmp/d787/d787_probe.cpp` — so both sides integrate the identical
//! polynomial integrand `(u − 1)² |det J|` with exact rules, and any residual
//! disagreement is the geometry map alone.  (P0 is deliberate: with one DOF per
//! element the two codes' L² DOF lattices and orderings cannot enter the
//! comparison.)
//!
//! Numbers (`tmp/d787/red_pre_fix_clean.log` / `green_post_fix.log`):
//!
//! | case | before D787 | after | MFEM 4.10 |
//! |------|-------------|-------|-----------|
//! | curved tet P2 | `0.72718635850791347` (+5.3 %) | `0.69081382746219200` | `0.69081382746219189` |
//! | curved tri P2 | `0.88543774484714621` (+1.1 %) | `0.87562549072077600` | `0.87562549072077578` |
//!
//! Both consumers use the one-element fixture in
//! `tests/d787_curved_simplex_geometry.rs` (the curved tet) and its 2-D twin.

use fem_assembly::postproc::grid_function::compute_l2_error_l2;
use fem_mesh::element_type::ElementType;
use fem_mesh::Mesh;
use fem_space::L2Space;

/// MFEM 4.10 truth: `mfem_truth_tet.txt` (`l2err_p0`, rule order 12).
const MFEM_TET_P2_L2ERR_P0: f64 = 0.6908138274621919;
/// MFEM 4.10 truth: `mfem_truth_tri.txt` (`l2err_p0`, rule order 12).
const MFEM_TRI_P2_L2ERR_P0: f64 = 0.8756254907207758;

const TOL: f64 = 1e-13;

fn g3(x: [f64; 3]) -> [f64; 3] {
    [
        x[0] + 0.1 * x[1] * x[2] + 0.2 * x[0] * x[0],
        x[1] + 0.05 * x[0] * x[2],
        x[2] + 0.1 * x[0] * x[1],
    ]
}

fn g2(x: [f64; 2]) -> [f64; 2] {
    [
        x[0] + 0.2 * x[0] * x[0] + 0.1 * x[0] * x[1],
        x[1] + 0.05 * x[0] * x[1],
    ]
}

/// Warp every geometry node and vertex — the D787 fixture (`set_curvature(2)`
/// + the same quadratic warp MFEM's probe applies).
fn warp_mesh<const D: usize>(mesh: &mut Mesh<D>, f: fn([f64; D]) -> [f64; D]) {
    let n_geom = mesh.geometry.as_ref().expect("curved mesh keeps a table").coords.len() / D;
    {
        let geo = mesh.geometry.as_mut().expect("curved table");
        for k in 0..n_geom {
            let mut x = [0.0_f64; D];
            x.copy_from_slice(&geo.coords[k * D..(k + 1) * D]);
            let y = f(x);
            geo.coords[k * D..(k + 1) * D].copy_from_slice(&y);
        }
    }
    for k in 0..mesh.n_nodes() {
        let mut x = [0.0_f64; D];
        x.copy_from_slice(&mesh.coords[k * D..(k + 1) * D]);
        let y = f(x);
        mesh.coords[k * D..(k + 1) * D].copy_from_slice(&y);
    }
}

fn curved_unit_tet() -> Mesh<3> {
    let mut m = Mesh::<3>::uniform(
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        vec![0, 1, 2, 3],
        vec![1],
        ElementType::Tet4,
        vec![0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3],
        vec![1; 4],
        ElementType::Tri3,
    );
    m.set_curvature(2);
    warp_mesh(&mut m, g3);
    m
}

fn curved_unit_tri() -> Mesh<2> {
    let mut m = Mesh::<2>::uniform(
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        vec![0, 1, 2],
        vec![1],
        ElementType::Tri3,
        vec![0, 1, 1, 2, 2, 0],
        vec![1; 3],
        ElementType::Line2,
    );
    m.set_curvature(2);
    warp_mesh(&mut m, g2);
    m
}

/// `u_h ≡ 1` (the single L² P0 dof) against `u = 1 + x + 2y (+ 3z)`.
#[test]
fn d787_compute_l2_error_l2_on_a_curved_tet_matches_mfem() {
    let mesh = curved_unit_tet();
    assert_eq!(mesh.geom_order(), 2);
    let space = L2Space::new(mesh, 0);
    let dofs = vec![1.0_f64];
    let exact: &(dyn Fn(&[f64]) -> f64 + Send + Sync) =
        &|x: &[f64]| 1.0 + x[0] + 2.0 * x[1] + 3.0 * x[2];
    let err = compute_l2_error_l2(&dofs, &space, exact, 12, None);
    eprintln!("D787 consumer compute_l2_error_l2, curved tet P2: {err:.17e}");
    let rel = (err - MFEM_TET_P2_L2ERR_P0).abs() / MFEM_TET_P2_L2ERR_P0;
    assert!(
        rel <= TOL,
        "curved tet L2 error: got {err:.17e}, MFEM {MFEM_TET_P2_L2ERR_P0:.17e} \
         (rel {rel:.3e}) — the estimator is integrating a different geometry map"
    );
}

#[test]
fn d787_compute_l2_error_l2_on_a_curved_tri_matches_mfem() {
    let mesh = curved_unit_tri();
    assert_eq!(mesh.geom_order(), 2);
    let space = L2Space::new(mesh, 0);
    let dofs = vec![1.0_f64];
    let exact: &(dyn Fn(&[f64]) -> f64 + Send + Sync) = &|x: &[f64]| 1.0 + x[0] + 2.0 * x[1];
    let err = compute_l2_error_l2(&dofs, &space, exact, 12, None);
    eprintln!("D787 consumer compute_l2_error_l2, curved tri P2: {err:.17e}");
    let rel = (err - MFEM_TRI_P2_L2ERR_P0).abs() / MFEM_TRI_P2_L2ERR_P0;
    assert!(
        rel <= TOL,
        "curved tri L2 error: got {err:.17e}, MFEM {MFEM_TRI_P2_L2ERR_P0:.17e} (rel {rel:.3e})"
    );
}
