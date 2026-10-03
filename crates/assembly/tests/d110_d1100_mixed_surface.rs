//! d110 / D1100 red-green pins for the **surface arm** of
//! `fem_assembly::mixed::assemble_hcurl_h1_mixed` (2-D elements embedded in
//! 3-D, `mesh.dim() != mesh.topological_dim()`).
//!
//! # The debt
//!
//! The kernel had no path for surface meshes: it read `mesh.dim()` (= 3 on a
//! `Mesh<3>` of triangles) and built geometry through
//! `ElementTransformation::from_simplex_nodes`, whose 3×3 corner-difference
//! Jacobian is singular for the 3 vertices of a triangle — every surface
//! mesh panicked with "degenerate simplex element" (`transformation.rs`).
//! Even ignoring the geometry, the curl columns were wrong by construction:
//! `TriNDk::eval_curl` writes `n_c` *reference scalars* into a buffer the
//! kernel indexed as `[n_c × 3]`, so the 3-D normalization branch read
//! garbage.  (D1081 fixed the hex/curved volume arms; the surface arm was
//! left open as D1100.)
//!
//! # Fix (the `assemble_hcurl_h1_gradient` sibling's surface contract)
//!
//! `is_surface = dim != topological_dim()`; geometry through
//! `crate::assembler::surface_jacobian` — the 3×2 isoparametric (order-`g`
//! geometry table, `H1TriPk(g)`) or affine-P1 surface map with measure
//! `√det(JᵀJ)`; the physical **surface** curl of the covariantly mapped
//! Nédélec basis is `curl̂_i/measure` (the exact surface analogue of the
//! planar 2-D `curl̂/detJ`: with `E = J·G⁻¹·û` the covariant Piola on a
//! surface, `(∇_S×E)·n = curl̂ û / |x_ξ×x_η|`, because the Piola property
//! `E·x_ξ = û₁`, `E·x_η = û₂` collapses the surface-curl formula
//! `(∇_S×E)·(x_ξ×x_η) = ∂_ξ(E·x_η) − ∂_η(E·x_ξ)`).  Per-dof orientation
//! signs fold into the scalar; the scalar sits in the **last** slot of each
//! `[n_dofs × dim]` dof chunk (the D1051 layout — what
//! `HCurlH1CurlIntegrator` reads).
//!
//! # Pins
//!
//! 1. **flat twin** (`z=0` 2-triangle `Mesh<3>` vs the same patch as a planar
//!    `Mesh<2>`): the surface arm must reproduce the planar kernel
//!    entry-wise (measures: `|det J₂|` vs `√det(JᵀJ)` agree exactly on the
//!    axis-aligned patch).  Pre-fix: panic.
//! 2. **tent surface, closed-form oracle** (4 warped triangles, straight):
//!    the pairing weight `q·measure` cancels the `1/measure` of the physical
//!    surface curl *exactly*, so every entry equals the geometry-free
//!    reference integral `Σ_q q_w·φ̂_j·s_i·curl̂_i` — an oracle that also has
//!    teeth: a forgotten `1/measure` normalization leaves a
//!    geometry-scaled, wrong matrix (every tent measure ≠ 1).  Pre-fix: panic.
//! 3. **curved surface** (the same tent at `geom_order == 2` through the
//!    isoparametric `H1TriPk(2)` route): same oracle (the measure cancels on
//!    curved maps too — the reference integrand is polynomial).  Pre-fix: panic.
//!
//! Geometry guards (d108 convention): every pin mesh must carry reversed
//! H(curl) edges, and pin 3 must be genuinely curved.

use fem_assembly::mixed::{
    assemble_hcurl_h1_mixed, ref_elem_vec, ref_elem_vol_with_pyramid_basis, HCurlH1CurlIntegrator,
};
use fem_element::QuadratureRule;
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

// ─── fixtures (the d806 embedded-surface family) ─────────────────────────────

/// `Mesh::<2>::unit_square_tri(n)` lifted verbatim into the `z = 0` plane —
/// same vertex numbering, same triangles, same boundary edges — so the flat
/// twin differs from the planar mesh **only** in the embedding dimension.
/// (`unit_square_tri(2)` carries reversed H(curl) edges; the 2-triangle
/// diagonal patch of the d806 fixture does not.)
fn flat_lifted(n: usize) -> Mesh<3> {
    let np = n + 1;
    let mut coords = Vec::with_capacity(np * np * 3);
    for j in 0..np {
        for i in 0..np {
            coords.extend_from_slice(&[i as f64 / n as f64, j as f64 / n as f64, 0.0]);
        }
    }
    let nid = |i: usize, j: usize| -> u32 { (j * np + i) as u32 };
    let mut conn = Vec::new();
    let mut elem_tags = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let (n0, n1, n2, n3) = (nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1));
            conn.extend_from_slice(&[n0, n1, n3]);
            elem_tags.push(1);
            conn.extend_from_slice(&[n1, n2, n3]);
            elem_tags.push(1);
        }
    }
    let mut face_conn = Vec::new();
    let mut face_tags = Vec::new();
    let mut edge = |a: u32, b: u32, t: i32| {
        face_conn.push(a);
        face_conn.push(b);
        face_tags.push(t);
    };
    for i in 0..n {
        edge(nid(i, 0), nid(i + 1, 0), 1);
        edge(nid(n, i), nid(n, i + 1), 2);
        edge(nid(i + 1, n), nid(i, n), 3);
        edge(nid(0, i + 1), nid(0, i), 4);
    }
    Mesh::<3>::uniform(
        coords,
        conn,
        elem_tags,
        ElementType::Tri3,
        face_conn,
        face_tags,
        ElementType::Line2,
    )
}

/// A non-planar "tent": 5 vertices, 4 triangles around the apex `z = 0.75`.
fn tent_surface() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            1.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.5, 0.5, 0.75,
        ],
        vec![0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4],
        vec![1, 1, 1, 1],
        ElementType::Tri3,
        vec![0, 1, 1, 2, 2, 3, 3, 0],
        vec![1, 2, 3, 4],
        ElementType::Line2,
    )
}

/// The D793 quadratic warp on every geometry node and vertex.
fn g3(x: [f64; 3]) -> [f64; 3] {
    [
        x[0] + 0.1 * x[1] * x[2] + 0.2 * x[0] * x[0],
        x[1] + 0.05 * x[0] * x[2],
        x[2] + 0.1 * x[0] * x[1],
    ]
}

fn warp_mesh(mesh: &mut Mesh<3>) {
    if let Some(ref mut geo) = mesh.geometry {
        for k in 0..geo.coords.len() / 3 {
            let mut x = [0.0_f64; 3];
            x.copy_from_slice(&geo.coords[k * 3..(k + 1) * 3]);
            let y = g3(x);
            geo.coords[k * 3..(k + 1) * 3].copy_from_slice(&y);
        }
    }
    for k in 0..mesh.n_nodes() {
        let mut x = [0.0_f64; 3];
        x.copy_from_slice(&mesh.coords[k * 3..(k + 1) * 3]);
        let y = g3(x);
        mesh.coords[k * 3..(k + 1) * 3].copy_from_slice(&y);
    }
}

/// The tent at order-2 geometry (`SetCurvature(2)`) + warp — a genuinely
/// curved 2-D manifold in 3-D (the d806 `curved_surface`).
fn curved_surface() -> Mesh<3> {
    let mut m = tent_surface();
    m.set_curvature(2);
    warp_mesh(&mut m);
    m
}

// ─── the closed-form oracle ──────────────────────────────────────────────────

/// Geometry-free oracle for the surface curl coupling at ND1 × P1:
/// `M[j, i] = Σ_q q_w · φ̂_j(ξ_q) · s_i · curl̂_i(ξ_q)`.
///
/// Exact (not a mirror of convenience): the pairing weight `q_w·measure`
/// cancels the physical surface curl's `1/measure` on **any** map — straight
/// or curved — so the assembled entry is the reference integral of the
/// polynomial `φ̂_j·curl̂_i`.  A kernel that forgets the `1/measure` (or the
/// signs) deviates by `O(measure)` — the tent measures are far from 1.
fn surface_mixed_curl_reference(
    mesh: &Mesh<3>,
    h1: &H1Space<Mesh<3>>,
    nd: &HCurlSpace<Mesh<3>>,
    qo: u8,
) -> Vec<f64> {
    let n_r = h1.n_dofs();
    let n_c = nd.n_dofs();
    let mut r = vec![0.0_f64; n_r * n_c];

    for e in 0..mesh.n_elements() as u32 {
        let et = mesh.element_type(e);
        let ref_r = ref_elem_vol_with_pyramid_basis(et, h1.order(), h1.pyramid_basis()).unwrap();
        let n_r_loc = ref_r.n_dofs();
        let ref_c = ref_elem_vec(et, nd.order(), SpaceType::HCurl).unwrap();
        let n_c_loc = ref_c.n_dofs();
        let quad: QuadratureRule = ref_r.quadrature(qo);

        let signs = nd.element_signs(e);
        let h1_dofs: Vec<usize> = h1.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nd_dofs: Vec<usize> = nd.element_dofs(e).iter().map(|&d| d as usize).collect();

        let mut phi_r = vec![0.0_f64; n_r_loc];
        let mut curl_ref = vec![0.0_f64; n_c_loc];
        for (xi, wq) in quad.points.iter().zip(quad.weights.iter()) {
            ref_r.eval_basis(xi, &mut phi_r);
            ref_c.eval_curl(xi, &mut curl_ref);
            for i in 0..n_c_loc {
                let s = signs.get(i).copied().unwrap_or(1.0);
                for j in 0..n_r_loc {
                    r[h1_dofs[j] * n_c + nd_dofs[i]] += wq * phi_r[j] * s * curl_ref[i];
                }
            }
        }
    }
    r
}

// ─── Pin 1: flat surface ≡ planar twin (pre-fix: degenerate-simplex panic) ───

#[test]
fn d1100_mixed_surface_flat_matches_planar_twin() {
    let m3 = flat_lifted(2);
    let m2 = Mesh::<2>::unit_square_tri(2);
    assert_eq!(m3.topological_dim(), 2, "flat_lifted must be a surface mesh");
    assert!(has_reversed_edges(&m3, 1), "expected reversed H(curl) edges (pin key)");
    assert!(has_reversed_edges(&m2, 1), "expected reversed H(curl) edges (pin key)");

    let h1_3 = H1Space::new(m3.clone(), 1);
    let nd_3 = HCurlSpace::new(m3.clone(), 1);
    let h1_2 = H1Space::new(m2.clone(), 1);
    let nd_2 = HCurlSpace::new(m2.clone(), 1);
    assert_eq!(h1_3.n_dofs(), h1_2.n_dofs(), "twin dof counts must agree");
    assert_eq!(nd_3.n_dofs(), nd_2.n_dofs(), "twin dof counts must agree");

    // Pre-fix this call panicked: "ElementTransformation: degenerate simplex
    // element" (3 vertices cannot span a 3×3 corner Jacobian).
    let m_surf = assemble_hcurl_h1_mixed(&h1_3, &nd_3, &[&HCurlH1CurlIntegrator], 4);
    let m_plan = assemble_hcurl_h1_mixed(&h1_2, &nd_2, &[&HCurlH1CurlIntegrator], 4);
    assert_eq!(m_surf.nrows, m_plan.nrows);
    assert_eq!(m_surf.ncols, m_plan.ncols);

    let mut dev = 0.0_f64;
    for row in 0..m_surf.nrows {
        for col in 0..m_surf.ncols {
            dev = dev.max((m_surf.get(row, col) - m_plan.get(row, col)).abs());
        }
    }
    println!("flat twin: max |surface − planar| = {dev:.3e}");
    assert!(
        dev < 1e-14,
        "surface kernel deviates from the planar twin by {dev:.3e} (D1100: on a \
         flat patch the surface curl pairing must reproduce the planar one)"
    );
}

// ─── Pin 2: tent surface vs the closed-form oracle (pre-fix: panic) ──────────

#[test]
fn d1100_mixed_surface_tent_matches_closed_form_oracle() {
    let name = "tent surface (straight)";
    let mesh = tent_surface();
    assert_eq!(mesh.topological_dim(), 2, "{name}: surface mesh");
    assert_eq!(mesh.geom_order(), 1, "{name}: straight fixture");
    assert!(has_reversed_edges(&mesh, 1), "{name}: expected reversed H(curl) edges");

    let h1 = H1Space::new(mesh.clone(), 1);
    let nd = HCurlSpace::new(mesh.clone(), 1);

    // Pre-fix: the same degenerate-simplex panic as the flat patch.
    let m = assemble_hcurl_h1_mixed(&h1, &nd, &[&HCurlH1CurlIntegrator], 4);
    let oracle = surface_mixed_curl_reference(&mesh, &h1, &nd, 4);
    let dev = max_dev(&m, &oracle, nd.n_dofs());
    println!("{name}: max |M − closed-form oracle| = {dev:.3e}");
    assert!(
        dev < 1e-14,
        "{name}: surface curl coupling deviates from the closed-form oracle by \
         {dev:.3e} (D1100: the q·measure weight must cancel the physical \
         surface curl's 1/measure, and the per-dof signs must reach the scalar)"
    );
}

// ─── Pin 3: curved surface through the isoparametric route (pre-fix: panic) ──

#[test]
fn d1100_mixed_surface_curved_matches_closed_form_oracle() {
    let name = "curved surface (geom_order 2)";
    let mesh = curved_surface();
    assert_eq!(mesh.topological_dim(), 2, "{name}: surface mesh");
    assert_eq!(mesh.geom_order(), 2, "{name}: pin needs a curved mesh");
    assert_eq!(mesh.element_type(0), ElementType::Tri3, "{name}: tri mesh");
    assert!(
        mesh.geometry_nodes(0).len() > mesh.element_nodes(0).len(),
        "{name}: geometry table row must exceed the corner row — re-key the pin"
    );
    assert!(has_reversed_edges(&mesh, 1), "{name}: expected reversed H(curl) edges");

    let h1 = H1Space::new(mesh.clone(), 1);
    let nd = HCurlSpace::new(mesh.clone(), 1);

    // Pre-fix: the same degenerate-simplex panic.  Post-fix the isoparametric
    // surface map (H1TriPk(2) over the geometry table row) serves the measure,
    // which cancels against the 1/measure of the physical surface curl — the
    // closed-form oracle still holds entry-wise.
    let m = assemble_hcurl_h1_mixed(&h1, &nd, &[&HCurlH1CurlIntegrator], 4);
    let oracle = surface_mixed_curl_reference(&mesh, &h1, &nd, 4);
    let dev = max_dev(&m, &oracle, nd.n_dofs());
    println!("{name}: max |M − closed-form oracle| = {dev:.3e}");
    assert!(
        dev < 1e-14,
        "{name}: surface curl coupling deviates from the closed-form oracle by \
         {dev:.3e} (D1100: the isoparametric surface route must preserve the \
         reference-integral identity)"
    );
}
