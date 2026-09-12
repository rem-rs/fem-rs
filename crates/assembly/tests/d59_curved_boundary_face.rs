//! D59 regression: on 3-D meshes with curved geometry the boundary integrals
//! must use the boundary element's own **curved** mapping, not the
//! corner-only bilinear face.
//!
//! Fixture: a hex (or tet) mesh of the unit cube whose bottom face is the
//! graph z = h(x,y) with h(x,y) = 0.3·x(1−x) + 0.2·y(1−y).  h is quadratic,
//! so the order-2 isoparametric geometry reproduces it *exactly* and the
//! surface measure is analytic: ∫√(1+|∇h|²) over the unit square.  h vanishes
//! at the four corners, so the corner-only bilinear geometry measures a flat
//! unit square (area 1) and misses the curved area by O(10⁻²).

use fem_assembly::standard::NeumannIntegrator;
use fem_assembly::Assembler;
use fem_element::quadrature::gauss_legendre_01;
use fem_mesh::{Mesh, MeshTopology};
use fem_space::{FESpace, H1Space};

/// h(x, y) — quadratic bump vanishing at the corners.
fn bump(x: f64, y: f64) -> f64 {
    0.3 * x * (1.0 - x) + 0.2 * y * (1.0 - y)
}

fn build_curved_cube(elem: fem_mesh::ElementType, n: usize) -> Mesh<3> {
    let mut mesh = Mesh::<3>::make_cartesian_3d(n, n, n, elem, 1.0, 1.0, 1.0, true);
    mesh.set_curvature(2);
    // Displace the bottom-layer geometry nodes onto the graph of `bump`.
    let geo = mesh.geometry.as_mut().expect("set_curvature built geometry");
    let d = 3usize;
    for k in 0..geo.n_nodes {
        let z = geo.coords[k * d + 2];
        if z < 0.5 {
            let (x, y) = (geo.coords[k * d], geo.coords[k * d + 1]);
            geo.coords[k * d + 2] = z + bump(x, y);
        }
    }
    mesh
}

/// ∫₀¹∫₀¹ √(1+|∇h|²) dx dy by 64-point Gauss–Legendre (machine precision).
fn analytic_bottom_area() -> f64 {
    let (pts, wts) = gauss_legendre_01(64);
    let mut total = 0.0_f64;
    for (i, &xi) in pts.iter().enumerate() {
        for (jdx, &eta) in pts.iter().enumerate() {
            let x = 0.5 * (xi + 1.0);
            let y = 0.5 * (eta + 1.0);
            let hx = 0.3 - 0.6 * x;
            let hy = 0.2 - 0.4 * y;
            let j = (1.0 + hx * hx + hy * hy).sqrt();
            total += wts[i] * wts[jdx] * j;
        }
    }
    total
}

fn run_boundary_area(elem: fem_mesh::ElementType) -> f64 {
    let mesh = build_curved_cube(elem, 2);
    let space = H1Space::new(mesh.clone(), 1);
    // P1 space: the face dofs are the face's corner nodes themselves.
    let face_dofs = |f: u32| mesh.face_nodes(f).to_vec();
    // Boundary tag 1 = bottom face (z = 0).
    let load = Assembler::assemble_boundary_linear(
        space.n_dofs(),
        &mesh,
        &face_dofs,
        1,
        &[&NeumannIntegrator::new(|_x, _n| 1.0)],
        &[1],
        16,
    );
    load.as_slice().iter().sum()
}

#[test]
fn d59_hex_curved_bottom_face_area_matches_analytic() {
    let expected = analytic_bottom_area();
    let measured = run_boundary_area(fem_mesh::ElementType::Hex8);
    // The corner-only geometry would measure the flat unit square.
    assert!(
        (measured - 1.0).abs() > 1.0e-2,
        "probe: curved area must visibly differ from the corner-only value 1.0 (got {measured})"
    );
    assert!(
        (measured - expected).abs() < 1.0e-12,
        "curved boundary area {measured:.16} vs analytic {expected:.16}"
    );
}

#[test]
fn d59_tet_curved_bottom_face_area_matches_analytic() {
    let expected = analytic_bottom_area();
    let measured = run_boundary_area(fem_mesh::ElementType::Tet4);
    assert!(
        (measured - 1.0).abs() > 1.0e-2,
        "probe: curved area must visibly differ from the corner-only value 1.0 (got {measured})"
    );
    assert!(
        (measured - expected).abs() < 1.0e-12,
        "curved boundary area {measured:.16} vs analytic {expected:.16}"
    );
}
