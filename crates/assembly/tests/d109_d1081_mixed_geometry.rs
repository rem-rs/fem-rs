//! d109 / D1081 red-green pins for the geometry path of
//! `fem_assembly::mixed::assemble_hcurl_h1_mixed`.
//!
//! # The debt
//!
//! The kernel built its element geometry **unconditionally** through
//! `ElementTransformation::from_simplex_nodes`:
//!
//! * on any **hex** (8 corners; `col_of = [1,2,3]` corner-difference Jacobian,
//!   `fem_mesh::transformation.rs:419-434`) the Jacobian is singular →
//!   `panic!("ElementTransformation: degenerate simplex element")` — even on a
//!   perfectly straight hex mesh;
//! * on a **surface** mesh (2-D elements in 3-D) there is no path at all;
//! * a **curved simplex** (order-`g` geometry table, `geom_order > 1`) is
//!   silently assembled on its straight corner simplex — `O(h)`-wrong, no
//!   signal.
//!
//! Fix (the siblings' contract — `assemble_hdiv_l2_mixed`, `mixed/mod.rs`
//! D667/D614, and the D1052 gradient twin): `use_iso = geom_order() > 1 ||
//! non-simplex`; the isoparametric arm reads the mesh's geometry table row
//! (`mesh.geometry_nodes(e)`) through `geo_ref_elem_from_mesh` +
//! `isoparametric_jacobian`, only straight simplices keep the affine
//! `from_simplex_nodes` fast path (bit-identical — the d1051 Whitney pins
//! stay green).
//!
//! # Pins
//!
//! Each pin compares the kernel entry-wise against a hand-rolled isoparametric
//! reference built from public primitives only (`geo_ref_elem_from_mesh`,
//! `isoparametric_jacobian`, `ref_elem_vec`, `h1_field_element`, the space's
//! own `element_signs`, and the physical-curl transform `2-D curl̂/detJ`,
//! `3-D (J·curl̂)/detJ` of the D1051 contract):
//!
//! 1. **straight hex** (`data/cylinder-hex.mesh`) — pre-fix this panics
//!    ("degenerate simplex element"); post-fix it must match the mirror
//!    (guard: `geometry_nodes(e) ≡ element_nodes(e)`, Hex8, reversed edges);
//! 2. **curved hex** (`tests/data/ball-quad.mesh`, 27-node Q2 geometry) —
//!    pre-fix panic; post-fix entry-wise against the isoparametric mirror
//!    (guard: `geom_order() > 1`, geometry row longer than the corner row);
//! 3. **curved simplex** (warped order-2 triangle) — pre-fix *silently*
//!    straight (no panic!); post-fix entry-wise against the isoparametric
//!    mirror (guard: `geom_order() == 2`, warped ≠ straight corners).

use fem_assembly::mixed::{
    assemble_hcurl_h1_mixed, ref_elem_vec, ref_elem_vol_with_pyramid_basis, HCurlH1CurlIntegrator,
};
use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology as _;
use fem_mesh::Mesh;
use fem_space::fe_space::{FESpace, SpaceType};
use fem_space::{HCurlSpace, H1Space};

fn has_reversed_edges<const D: usize>(mesh: &Mesh<D>, order: u8) -> bool {
    let nd = HCurlSpace::new(mesh.clone(), order);
    (0..mesh.n_elements() as u32).map(|e| nd.element_signs(e)).flatten().any(|&s| s < 0.0)
}

fn max_dev(a: &CsrMatrix<f64>, dense: &[f64], ncols: usize) -> f64 {
    let mut m = 0.0_f64;
    for row in 0..a.nrows {
        for col in 0..a.ncols {
            m = m.max((a.get(row, col) - dense[row * ncols + col]).abs());
        }
    }
    m
}

/// Hand-rolled isoparametric reference for the H¹×H(curl) curl coupling
/// `M[j,i] = ∫ φ_j · curl(E_i) dx` at ND/P order 1 through public primitives:
/// per-QP isoparametric Jacobian (geometry table row), physical curl
/// (`2-D curl̂/detJ`, `3-D (J·curl̂)/detJ`), per-dof signs, signed weight.
fn mixed_curl_reference<const D: usize>(
    mesh: &Mesh<D>,
    h1: &H1Space<Mesh<D>>,
    nd: &HCurlSpace<Mesh<D>>,
    qo: u8,
) -> Vec<f64> {
    let dim = D;
    let n_r = h1.n_dofs();
    let n_c = nd.n_dofs();
    let mut r = vec![0.0_f64; n_r * n_c];

    for e in 0..mesh.n_elements() as u32 {
        let et = mesh.element_type(e);
        // The kernel's own row element + quadrature dispatch (order 1).
        let ref_r =
            ref_elem_vol_with_pyramid_basis(et, h1.order(), h1.pyramid_basis()).unwrap();
        let n_r_loc = ref_r.n_dofs();
        let ref_c = ref_elem_vec(et, nd.order(), SpaceType::HCurl).unwrap();
        let n_c_loc = ref_c.n_dofs();
        let quad = ref_r.quadrature(qo);

        let geo = geo_ref_elem_from_mesh(mesh, e)
            .expect("mixed_curl_reference: isoparametric geometry element");
        let gnodes = mesh.geometry_nodes(e).to_vec();
        let signs = nd.element_signs(e);
        let h1_dofs: Vec<usize> = h1.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nd_dofs: Vec<usize> = nd.element_dofs(e).iter().map(|&d| d as usize).collect();

        let mut phi_r = vec![0.0_f64; n_r_loc];
        let mut curl_ref = vec![0.0_f64; n_c_loc * 3];

        for (xi, wq) in quad.points.iter().zip(quad.weights.iter()) {
            let (jac, det, _xp) = isoparametric_jacobian(mesh, &gnodes, geo.as_ref(), xi, dim);
            // D696 family verdict: signed weight.
            let w = wq * det;
            ref_r.eval_basis(xi, &mut phi_r);
            ref_c.eval_curl(xi, &mut curl_ref);
            // Physical curl per dof (D1051 layout: [n_dofs × dim], z last).
            for i in 0..n_c_loc {
                let s = signs.get(i).copied().unwrap_or(1.0);
                for j in 0..n_r_loc {
                    let cj = if dim == 2 {
                        curl_ref[i] / det
                    } else {
                        (0..dim).map(|k| jac[(dim - 1, k)] * curl_ref[i * 3 + k]).sum::<f64>() / det
                    };
                    r[h1_dofs[j] * n_c + nd_dofs[i]] += s * w * phi_r[j] * cj;
                }
            }
        }
    }
    r
}

// ─── Pin 1: straight hex (pre-fix: degenerate-simplex panic) ─────────────────

#[test]
fn d1081_mixed_kernel_straight_hex_oracle() {
    let (name, rel, qo, order) = ("cylinder-hex", "../../data/cylinder-hex.mesh", 4u8, 1u8);
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let mfem = read_mfem_file(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
    let mesh = mfem.mesh3d.unwrap_or_else(|| panic!("{name}: 3-D mesh"));
    let e0 = 0u32;
    assert_eq!(mesh.element_type(e0), ElementType::Hex8, "{name}: hex mesh");
    // Control-mesh guard: straight ⇒ geometry row equals the corner row.
    assert!(
        mesh.geometry_nodes(e0).len() == mesh.element_nodes(e0).len(),
        "{name}: mesh must be straight (equal node rows)"
    );
    assert!(
        has_reversed_edges(&mesh, order),
        "{name}: expected reversed H(curl) edges (sign coverage)"
    );

    let h1 = H1Space::new(mesh.clone(), order);
    let nd = HCurlSpace::new(mesh.clone(), order);

    // Pre-fix this call panicked: "ElementTransformation: degenerate simplex
    // element" (the hex corner-difference Jacobian is singular).
    let m = assemble_hcurl_h1_mixed(&h1, &nd, &[&HCurlH1CurlIntegrator], qo);
    assert_eq!(m.nrows, h1.n_dofs());
    assert_eq!(m.ncols, nd.n_dofs());

    let oracle = mixed_curl_reference(&mesh, &h1, &nd, qo);
    let dev = max_dev(&m, &oracle, nd.n_dofs());
    println!("{name}: max |M − isoparametric reference| = {dev:.3e}");
    assert!(
        dev < 1e-11,
        "{name}: mixed curl kernel deviates from the straight-hex reference by \
         {dev:.3e} (D1081: hexes must take the isoparametric geometry path, \
         not the corner-simplex map)"
    );
}

// ─── Pin 2: curved hex (27-node geometry table; pre-fix: panic) ──────────────

#[test]
fn d1081_mixed_kernel_curved_hex_oracle() {
    let (name, rel, qo, order) = ("ball-quad", "tests/data/ball-quad.mesh", 7u8, 1u8);
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let mfem = read_mfem_file(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
    let mesh = mfem.mesh3d.unwrap_or_else(|| panic!("{name}: 3-D mesh"));
    let e0 = 0u32;
    assert!(mesh.geom_order() > 1, "{name}: pin needs a curved mesh");
    assert_eq!(mesh.element_type(e0), ElementType::Hex8, "{name}: hex mesh");
    assert!(
        mesh.geometry_nodes(e0).len() > mesh.element_nodes(e0).len(),
        "{name}: geometry table row ({}) must exceed the corner row ({}) — \
         re-key the pin",
        mesh.geometry_nodes(e0).len(),
        mesh.element_nodes(e0).len()
    );
    assert!(
        has_reversed_edges(&mesh, order),
        "{name}: expected reversed H(curl) edges (sign coverage)"
    );

    let h1 = H1Space::new(mesh.clone(), order);
    let nd = HCurlSpace::new(mesh.clone(), order);

    // Pre-fix: the same degenerate-simplex panic as the straight hex.
    let m = assemble_hcurl_h1_mixed(&h1, &nd, &[&HCurlH1CurlIntegrator], qo);
    let oracle = mixed_curl_reference(&mesh, &h1, &nd, qo);
    let dev = max_dev(&m, &oracle, nd.n_dofs());
    println!("{name}: max |M − isoparametric reference| = {dev:.3e}");
    assert!(
        dev < 1e-10,
        "{name}: mixed curl kernel deviates from the curved-hex reference by \
         {dev:.3e} (D1081: the iso arm must read mesh.geometry_nodes(e) through \
         geo_ref_elem_from_mesh)"
    );
}

// ─── Pin 3: curved simplex (contract control — see the cancellation note) ────

fn warp2(x: [f64; 2]) -> [f64; 2] {
    [x[0] + 0.2 * x[0] * x[0] + 0.1 * x[0] * x[1], x[1] + 0.05 * x[0] * x[1]]
}

/// Contract control for the `geom_order() > 1` **simplex** arm: a curved
/// (order-2 geometry) triangle mesh must assemble through the isoparametric
/// route and agree with the isoparametric mirror.
///
/// *No teeth — by arithmetic, not by luck*: with the D1051 contract (signed
/// weight `w·det` × physical curl `curl̂/det` using the *same* per-QP det) the
/// det cancels and the curl-coupling kernel is invariant to which affine/
/// isoparametric map produced it — the pre-fix corner-simplex route already
/// matched the isoparametric one on this mesh (measured: equal to 1e-11, red
/// log `tmp/d109b/`).  The arm matters only for consumers that read the QP's
/// physical point (coefficients — none in this kernel today) or element types
/// whose corner row is not a simplex (D243 trap) — and of course for the
/// *panic* arm pinned above.  The pin keeps the new `use_iso` gate on
/// simplices from regressing.

#[test]
fn d1081_mixed_kernel_curved_simplex_isoparametric() {
    let name = "warped order-2 triangle";
    let mut mesh = Mesh::<2>::unit_square_tri(2);
    mesh.set_curvature(2);
    // Warp every geometry dof (and vertex copy) — the d831 curved-tri recipe.
    let n_geom = mesh.geometry.as_ref().expect("curved table").coords.len() / 2;
    {
        let geo = mesh.geometry.as_mut().expect("curved table");
        for k in 0..n_geom {
            let mut x = [0.0_f64; 2];
            x.copy_from_slice(&geo.coords[k * 2..(k + 1) * 2]);
            let y = warp2(x);
            geo.coords[k * 2..(k + 1) * 2].copy_from_slice(&y);
        }
    }
    for k in 0..mesh.n_nodes() {
        let mut x = [0.0_f64; 2];
        x.copy_from_slice(&mesh.coords[k * 2..(k + 1) * 2]);
        let y = warp2(x);
        mesh.coords[k * 2..(k + 1) * 2].copy_from_slice(&y);
    }

    // Guards: curved simplex — the exact arm the old code served straight.
    assert_eq!(mesh.geom_order(), 2, "{name}: pin needs geom_order 2");
    assert_eq!(mesh.element_type(0), ElementType::Tri3, "{name}: tri mesh");
    assert!(
        has_reversed_edges(&mesh, 1),
        "{name}: expected reversed H(curl) edges (sign coverage)"
    );

    let h1 = H1Space::new(mesh.clone(), 1);
    let nd = HCurlSpace::new(mesh.clone(), 1);

    // Pre-fix this already matched (the det cancels — see the doc above);
    // the pin guards the new use_iso gate on curved simplices.
    let m = assemble_hcurl_h1_mixed(&h1, &nd, &[&HCurlH1CurlIntegrator], 4);
    let oracle = mixed_curl_reference(&mesh, &h1, &nd, 4);
    let dev = max_dev(&m, &oracle, nd.n_dofs());
    println!("{name}: max |M − isoparametric reference| = {dev:.3e}");
    assert!(
        dev < 1e-11,
        "{name}: mixed curl kernel deviates from the curved-simplex reference \
         by {dev:.3e} (D1081: geom_order > 1 simplices must take the \
         isoparametric map, not the silent corner-simplex one)"
    );
}
