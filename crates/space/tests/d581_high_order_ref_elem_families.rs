//! D581: the high-order cell families (`Hex20`/`Hex27`/`Prism15`/`Prism18`/
//! `Pyramid13`) in `fem_space::ref_elem`'s dispatchers — the truth source the
//! postproc consumers (`ref_elem_vol` / `geom_jacobian` / vertex-sampled
//! estimators) delegate to.
//!
//! MFEM 4.10 truth probe (`tmp/d581/d581_probe.cpp`, serial build in
//! `$HOME/work/d581`): straight unit meshes, `SetCurvature(2)` (the H1
//! order-2 geometry table: `H1_HexahedronElement(2)` / `H1_WedgeElement(2)` /
//! `H1_FuentesPyramidElement(2)` lattices), every geometry node pushed through
//! an analytic polynomial displacement `G`, then ∫1 dK with the EXPLICIT
//! `IntRules.Get(geom, order)` rule (the D235 discipline — `GetElementVolume`'s
//! single-point `OrderJ()` rule is only exact on straight cells).  `G` is
//! polynomial so `det J` is a polynomial and both sides' rules integrate it
//! exactly.
//!
//! Reference families (the contract the arms below pin):
//! * Hex20/Hex27 → `HexQk` on `[-1,1]³` (the same family
//!   `crates/mesh/src/curved.rs` reads Hex27 geometry tables with and
//!   `ElementType::Hex27::ref_elem` hands out; `HexQk::new(2)` = 27 dofs).
//! * Prism15/Prism18 → the prism tensor family: H¹ slots `H1PrismPk`
//!   (MFEM `H1_WedgeElement` entity order, D168), geometry tables `PrismPk`
//!   (equispaced layer-major) — 18 dofs at order 2, matching MFEM's curved
//!   wedge.  (A Gmsh 15-node prism carries no 15-dof MFEM family; fem-rs
//!   reads its geometry with the shared prism family, slot count from the
//!   mesh table.)
//! * Pyramid13 → the Fuentes H¹ family (`h1_pyramid_element`,
//!   `pyr_type = 1`): `p(p²+3)+1 = 15` dofs at order 2 — MFEM 4.10's own
//!   quadratic pyramid geometry table has 15 nodes, NOT 13.

use fem_element::lagrange::PyramidBasisType;
use fem_element::ReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_mesh::amr;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::l2::l2_pyramid_element;
use fem_space::ref_elem::{
    equispaced_prism, fixed_order_tensor, gll_tensor, geometry_node_element, h1_field_element,
    h1_prism_slots, h1_pyramid_slots, l2_field_element, legacy_equispaced_element,
};
use fem_space::L2Basis;

// ─── Displacement fields G (must mirror the MFEM probe exactly) ─────────────

fn warp_hex(t: [f64; 3]) -> [f64; 3] {
    [
        t[0] + 0.08 * t[1] * (1.0 - t[1]),
        t[1] + 0.06 * t[2] * (1.0 - t[2]),
        t[2] + 0.05 * t[0] * (1.0 - t[0]),
    ]
}

fn warp_wedge(t: [f64; 3]) -> [f64; 3] {
    [
        t[0] + 0.07 * t[2] * (1.0 - t[2]),
        t[1] + 0.05 * t[2] * (1.0 - t[2]),
        t[2] + 0.06 * t[0] * (1.0 - t[0]) * t[1] * (1.0 - t[1]),
    ]
}

fn warp_pyr(t: [f64; 3]) -> [f64; 3] {
    [
        t[0] + 0.025 * t[1] * (1.0 - t[1]),
        t[1] + 0.025 * t[2] * (1.0 - t[2]),
        t[2] + 0.020 * t[0] * (1.0 - t[0]),
    ]
}

// ─── Volume machinery (the estimator formula: Σ_q w_q·|det J(x_q)|) ─────────

fn det3(j: [[f64; 3]; 3]) -> f64 {
    j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
        - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
        + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0])
}

/// ∫ 1 over the isoparametric element whose geometry table (family slot →
/// physical coordinate) is `table`, using the family's own quadrature.
fn element_volume(family: &dyn ReferenceElement, table: &[[f64; 3]], quad_order: u8) -> f64 {
    let n = family.n_dofs();
    assert_eq!(n, table.len(), "family/table slot count mismatch");
    let q = family.quadrature(quad_order);
    let mut grad = vec![0.0_f64; n * 3];
    let mut vol = 0.0;
    for (qi, xi) in q.points.iter().enumerate() {
        family.eval_grad_basis(xi, &mut grad);
        let mut j = [[0.0_f64; 3]; 3];
        for k in 0..n {
            for d in 0..3 {
                let g = grad[k * 3 + d];
                for c in 0..3 {
                    j[c][d] += table[k][c] * g;
                }
            }
        }
        let det = det3(j);
        assert!(det > 0.0, "folded geometry: det J = {det:e} at quad point {qi}");
        vol += q.weights[qi] * det;
    }
    vol
}

// ─── Arm shape tests (every new arm, every dispatcher) ──────────────────────

#[test]
fn d581_hex27_hex20_gll_tensor_and_h1_arms() {
    for et in [ElementType::Hex20, ElementType::Hex27] {
        // gll_tensor: HexQk on [-1,1]^3 — 27 dofs at order 2, 8 at order 1.
        let e2 = gll_tensor(et, 2);
        assert_eq!(e2.n_dofs(), 27, "{et:?} gll_tensor order 2");
        let dc = e2.dof_coords();
        assert_eq!(dc[0], vec![-1.0, -1.0, -1.0], "{et:?} HexQk frame origin");
        assert!(dc.iter().any(|c| c[0] == 1.0), "{et:?} HexQk frame max");
        assert_eq!(gll_tensor(et, 1).n_dofs(), 8, "{et:?} gll_tensor order 1");
        // quadrature on the fem-rs cube frame: weight sum = 8 (2^3).
        let q = e2.quadrature(10);
        let wsum: f64 = q.weights.iter().sum();
        assert!((wsum - 8.0).abs() < 1e-14, "{et:?} hex rule weight sum {wsum}");

        // h1_field_element: order 0 → P0 cube, order 1 → trilinear HexQ1,
        // order ≥ 2 → the HexQk GLL family (MFEM H1_FECollection semantics).
        assert_eq!(h1_field_element(et, 0, PyramidBasisType::default()).n_dofs(), 1);
        assert_eq!(h1_field_element(et, 1, PyramidBasisType::default()).n_dofs(), 8);
        assert_eq!(h1_field_element(et, 2, PyramidBasisType::default()).n_dofs(), 27);
        assert_eq!(h1_field_element(et, 3, PyramidBasisType::default()).n_dofs(), 64);
    }
}

#[test]
fn d581_hex27_hex20_l2_legacy_geometry_arms() {
    for et in [ElementType::Hex20, ElementType::Hex27] {
        // L² GaussLegendre (DG default): open GL tensor nodes, 27 dofs at order 2.
        assert_eq!(l2_field_element(et, 2, L2Basis::GaussLegendre).n_dofs(), 27);
        // L² GaussLobatto: lexicographic GLL (HexQk::new_lex).
        assert_eq!(l2_field_element(et, 2, L2Basis::GaussLobatto).n_dofs(), 27);
        // Legacy equispaced dispatch (the historical ref_elem_vol fallback).
        assert_eq!(legacy_equispaced_element(et, 2).n_dofs(), 27);
        assert_eq!(legacy_equispaced_element(et, 0).n_dofs(), 1);
        // Geometry-node dispatch (the curved-mesh table reader).
        assert_eq!(geometry_node_element(et, 2).n_dofs(), 27);
        assert_eq!(geometry_node_element(et, 1).n_dofs(), 8);
    }
}

#[test]
fn d581_prism15_prism18_pyramid13_arms() {
    // H¹ wedge slots: 18 dofs at order 2 for BOTH quadratic prism cell types
    // (MFEM's curved wedge = H1_WedgeElement(2); a Gmsh 15-node prism shares
    // the same 18-dof family — there is no 15-dof MFEM wedge element).
    for et in [ElementType::Prism15, ElementType::Prism18] {
        assert_eq!(h1_field_element(et, 2, PyramidBasisType::default()).n_dofs(), 18);
        assert_eq!(geometry_node_element(et, 2).n_dofs(), 18);
        assert_eq!(legacy_equispaced_element(et, 2).n_dofs(), 18);
    }
    assert_eq!(h1_prism_slots(2).n_dofs(), 18);
    assert_eq!(equispaced_prism(2).n_dofs(), 18);

    // Pyramid13: the Fuentes family of the default pyr_type — MFEM 4.10's
    // quadratic pyramid has p(p²+3)+1 = 15 dofs, not 13.
    assert_eq!(
        h1_field_element(ElementType::Pyramid13, 2, PyramidBasisType::default()).n_dofs(),
        15
    );
    assert_eq!(geometry_node_element(ElementType::Pyramid13, 2).n_dofs(), 15);
    assert_eq!(h1_pyramid_slots(2, PyramidBasisType::Fuentes).n_dofs(), 15);
    // Bergot keeps its own count (the pyr_type switch is honoured).
    let _bergot = h1_pyramid_slots(2, PyramidBasisType::Bergot);
    // L² pyramid arm keeps its own family (order 0 → P0 pyramid rule).
    assert_eq!(
        l2_field_element(ElementType::Pyramid13, 2, L2Basis::GaussLegendre).n_dofs(),
        l2_pyramid_element(2, L2Basis::GaussLegendre).n_dofs()
    );
    assert_eq!(
        l2_field_element(ElementType::Pyramid13, 0, L2Basis::GaussLegendre).n_dofs(),
        1
    );
}

// ─── MFEM 4.10 volume parity ────────────────────────────────────────────────

#[test]
fn d581_hex27_warped_volume_matches_mfem() {
    // MFEM probe (tmp/d581/probe_out.log):
    //   HEX27 straight volume = 1.00000000000000133e+00  (IntRules CUBE 10)
    //   HEX27 warped   volume = 9.99999999999999889e-01  (IntRules CUBE 10)
    // fem-rs: HexQk(2) on [-1,1]^3, geometry table = G((xi+1)/2) at the family
    // dof lattice (MFEM's H1_HexahedronElement(2) lattice is the same
    // {0, 1/2, 1}^3 points; the slot order differs — fem-rs keeps its legacy
    // p=2 hex slots, hex.rs doc — but the (slot ↔ reference point) pairing is
    // each family's own, so the interpolated geometry is identical).
    let family = geometry_node_element(ElementType::Hex27, 2);
    let dc = family.dof_coords();
    let table: Vec<[f64; 3]> = dc
        .iter()
        .map(|c| warp_hex([(c[0] + 1.0) / 2.0, (c[1] + 1.0) / 2.0, (c[2] + 1.0) / 2.0]))
        .collect();
    let straight: Vec<[f64; 3]> = dc
        .iter()
        .map(|c| [(c[0] + 1.0) / 2.0, (c[1] + 1.0) / 2.0, (c[2] + 1.0) / 2.0])
        .collect();

    let vol_straight = element_volume(family.as_ref(), &straight, 10);
    assert!(
        (vol_straight - 1.0).abs() < 1e-14,
        "Hex27 straight: {vol_straight:.17e}"
    );
    let cpp_warped = 9.99999999999999889e-01;
    let vol = element_volume(family.as_ref(), &table, 10);
    // Evidence line (tmp/d581/probe_parity.log): fem-rs vs MFEM 4.10.
    eprintln!("D581 Hex27 warped: fem-rs {vol:.17e} vs MFEM {cpp_warped:.17e} (|Δ| = {:.1e})", (vol - cpp_warped).abs());
    assert!(
        (vol - cpp_warped).abs() < 1e-13,
        "Hex27 warped: {vol:.17e} vs MFEM {cpp_warped:.17e}"
    );

    // The h1 arm (the space's own family) integrates the same geometry equally.
    let h1 = h1_field_element(ElementType::Hex27, 2, PyramidBasisType::default());
    let vol_h1 = element_volume(h1.as_ref(), &table, 10);
    assert!((vol_h1 - cpp_warped).abs() < 1e-13, "Hex27 warped (h1 arm): {vol_h1:.17e}");
}

#[test]
fn d581_prism18_warped_volume_matches_mfem() {
    // MFEM probe:
    //   PRISM18 straight volume = 5.00000000000000444e-01  (IntRules PRISM 10)
    //   PRISM18 warped   volume = 4.99999999999999722e-01  (IntRules PRISM 10)
    // Unit wedge → the straight map is the identity; the geometry table is
    // G(dof_coords) in each family's own slot pairing.  The fem-rs reference
    // prism is the axes permutation of MFEM's wedge frame ((ξ segment) ×
    // ((η,ζ) triangle) vs (base triangle x,y) × (vertical z)); G is applied
    // in the family's own coordinates, and the parity is permutation-tight —
    // |det D(G∘P)| = |det DG∘P| pointwise and the domain is the same unit
    // prism, so the integrals agree to rounding.
    let cpp_warped = 4.99999999999999722e-01;

    // H¹ slot family (H1PrismPk(2), MFEM entity order = the slots MFEM's
    // nodes GF is written in).
    let h1 = h1_field_element(ElementType::Prism18, 2, PyramidBasisType::default());
    let table: Vec<[f64; 3]> = h1
        .dof_coords()
        .iter()
        .map(|c| warp_wedge([c[0], c[1], c[2]]))
        .collect();
    let vol = element_volume(h1.as_ref(), &table, 10);
    eprintln!("D581 Prism18 warped (h1): fem-rs {vol:.17e} vs MFEM {cpp_warped:.17e} (|Δ| = {:.1e})", (vol - cpp_warped).abs());
    assert!(
        (vol - cpp_warped).abs() < 1e-13,
        "Prism18 warped (h1): {vol:.17e} vs {cpp_warped:.17e}"
    );

    // Geometry-reader family (PrismPk(2), equispaced layer-major) — the same
    // interpolant, a different slot layout.
    let geo = geometry_node_element(ElementType::Prism18, 2);
    let table_g: Vec<[f64; 3]> = geo
        .dof_coords()
        .iter()
        .map(|c| warp_wedge([c[0], c[1], c[2]]))
        .collect();
    let vol_g = element_volume(geo.as_ref(), &table_g, 10);
    assert!((vol_g - cpp_warped).abs() < 1e-13, "Prism18 warped (geo): {vol_g:.17e}");

    // Prism15 shares the family (no 15-dof MFEM wedge exists).
    let p15 = h1_field_element(ElementType::Prism15, 2, PyramidBasisType::default());
    assert_eq!(p15.n_dofs(), h1.n_dofs());
}

#[test]
fn d581_pyramid13_fuentes_geometry_warped_volume_matches_mfem() {
    // MFEM probe on data/d235_pyramid_2x2x2.mesh (48 straight pyramids):
    //   PYR13 straight e0/total = 2.08333333333333079e-02 / 9.99999999999997780e-01
    //   PYR13 warped   e0/total = 2.08333902994791952e-02 / 1.00000000000000155e+00
    //   PYR13 element-0 n_dofs  = 15  (H1_FuentesPyramidElement(2))
    // fem-rs: per element the straight collapsed-linear map through the 5
    // mesh vertices (h1_pyramid_element(1) = MFEM LinearPyramidElement),
    // geometry table = G(that map at the Fuentes(2) dof lattice), measure via
    // the Fuentes(2) family + the MFEM-cloned pyramid rule.
    let path = format!(
        "{}/../../{}",
        env!("CARGO_MANIFEST_DIR"),
        "data/d235_pyramid_2x2x2.mesh"
    );
    let mfem = read_mfem_file(&path).expect("read d235 pyramid mesh");
    let mesh = mfem.mesh3d.expect("3-D pyramid mesh");
    assert_eq!(mesh.element_type(0), ElementType::Pyramid5);

    let p1 = h1_pyramid_slots(1, PyramidBasisType::default());
    assert_eq!(p1.n_dofs(), 5);
    let geo = h1_pyramid_slots(2, PyramidBasisType::default());
    assert_eq!(geo.n_dofs(), 15);
    let dc = geo.dof_coords();
    let shapes: Vec<Vec<f64>> = dc
        .iter()
        .map(|xi| {
            let mut v = vec![0.0_f64; 5];
            p1.eval_basis(xi, &mut v);
            v
        })
        .collect();

    let mut rs_e0 = 0.0;
    let mut total = 0.0;
    for e in 0..mesh.n_elements() as u32 {
        let verts = mesh.element_nodes(e);
        assert_eq!(verts.len(), 5);
        let v: Vec<[f64; 3]> = (0..5)
            .map(|i| {
                let c = mesh.node_coords(verts[i]);
                [c[0], c[1], c[2]]
            })
            .collect();
        let table: Vec<[f64; 3]> = shapes
            .iter()
            .map(|phi| {
                let mut p = [0.0_f64; 3];
                for (k, phik) in phi.iter().enumerate() {
                    for c in 0..3 {
                        p[c] += phik * v[k][c];
                    }
                }
                warp_pyr(p)
            })
            .collect();
        let vol = element_volume(geo.as_ref(), &table, 12);
        total += vol;
        if e == 0 {
            rs_e0 = vol;
        }
    }

    let mfem_e0 = 2.08333902994791952e-02;
    let mfem_total = 1.00000000000000155e+00;
    eprintln!(
        "D581 Pyr13 (Fuentes-15) warped e0/total: fem-rs {rs_e0:.17e}/{total:.17e} vs MFEM {mfem_e0:.17e}/{mfem_total:.17e} (|Δe0| = {:.1e}, |Δtot| = {:.1e})",
        (rs_e0 - mfem_e0).abs(),
        (total - mfem_total).abs()
    );
    assert!(
        (rs_e0 - mfem_e0).abs() < 1e-12,
        "Pyr13 warped e0: {rs_e0:.17e} vs MFEM {mfem_e0:.17e}"
    );
    assert!(
        (total - mfem_total).abs() < 1e-11,
        "Pyr13 warped total: {total:.17e} vs MFEM {mfem_total:.17e}"
    );
}

// ─── Consumer contract: vertex sampling on quadratic cells ─────────────────

#[test]
fn d581_prism15_corner_sampling_true_corners_distinct() {
    // The Tet10 lesson (D235): a vertex-sampled estimator must take the TRUE
    // corner slots, never the connectivity node count (Prism15/18 cells list
    // 15/18 nodes; only 6 are corners; Pyramid13 lists 13; only 5 are).
    //
    // Contract pinned here, for both prism slot conventions.  The fem-rs
    // reference prism frame is (ξ segment) × ((η, ζ) triangle) — the axes
    // permutation of MFEM's (base-triangle x,y; vertical z) wedge — so the 6
    // reference corners are ξ ∈ {0,1} × {(η,ζ) = (0,0),(1,0),(0,1)}:
    // * the H¹ family (`H1PrismPk`, MFEM entity order) keeps the 6 corners at
    //   the FIRST six slots — a vertex sampler reading space DOF slots can
    //   take `dof_coords()[..6]` directly;
    // * the geometry-reader family (`PrismPk`, equispaced layer-major) does
    //   NOT (its layer-0 block holds 3 vertices + 3 edge mids), so a sampler
    //   must find the corners by matching dof_coords against the 6 reference
    //   corner points — coordinate lookup, never slot arithmetic;
    // * sampling the warped order-2 geometry through the 6 true corners
    //   yields 6 distinct physical points (the old `_ => {}` origin fallback
    //   would have produced 6 coincident centroid samples).
    let ref_corners: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0],
        vec![1.0, 1.0, 0.0],
        vec![1.0, 0.0, 1.0],
    ];

    let h1 = h1_field_element(ElementType::Prism15, 2, PyramidBasisType::default());
    let h1_dc = h1.dof_coords();
    for (k, rc) in ref_corners.iter().enumerate() {
        assert_eq!(&h1_dc[k], rc, "H1PrismPk slot {k} must be corner {rc:?}");
    }

    let geo = geometry_node_element(ElementType::Prism18, 2);
    let dc = geo.dof_coords();
    let corner_slots: Vec<usize> = ref_corners
        .iter()
        .map(|rc| {
            dc.iter()
                .position(|c| c == rc)
                .unwrap_or_else(|| panic!("geometry family misses corner {rc:?}"))
        })
        .collect();
    let table: Vec<[f64; 3]> = dc
        .iter()
        .map(|c| warp_wedge([c[0], c[1], c[2]]))
        .collect();
    // Evaluate the geometry map at each corner reference point.
    let mut phi = vec![0.0_f64; geo.n_dofs()];
    let mut pts = Vec::with_capacity(6);
    for rc in &ref_corners {
        geo.eval_basis(rc, &mut phi);
        let mut p = [0.0_f64; 3];
        for (k, phik) in phi.iter().enumerate() {
            for c in 0..3 {
                p[c] += phik * table[k][c];
            }
        }
        pts.push(p);
    }
    for (a, pa) in pts.iter().enumerate() {
        for b in (a + 1)..pts.len() {
            let d: f64 = (0..3)
                .map(|c| (pa[c] - pts[b][c]).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(d > 1e-6, "corner samples {a} and {b} coincide (d = {d:e})");
        }
    }
    // The corner slots exist in the geometry family too (distinct slots).
    let mut s = corner_slots.clone();
    s.sort_unstable();
    s.dedup();
    assert_eq!(s.len(), 6, "corner slots must be distinct geometry slots");
}

#[test]
fn d581_hex27_p1_geometry_of_prefined_cells_integrates_unit_cube() {
    // Mesh-side convention note (D612, registered from this lane): the
    // p_refine chain Hex8 → Hex20 → Hex27 writes the 12 edge mids in VTK
    // order (bottom, top, vertical) and the 6 face centres as
    // (ζ−, ζ+, η−, η+, ξ−, ξ+), while the `HexQk::new(2)` family slots carry
    // (vertical, y, x) edge mids and (ξ−, ξ+, η−, η+, ζ−, ζ+) face centres —
    // a 27-slot isoparametric read of a p_refine table deforms the geometry
    // (det J = -2.42e-1 on the unit cube; see tmp/d581/d612_hex27_conn.md).
    // What IS pinned across every convention — and what this test guards — is
    // the order-1 geometry: slots 0..8 are the cube corners in Hex8 order, so
    // the P1 map of a p-refined Hex27 cell integrates the unit volume exactly.
    let hex8 = Mesh::<3>::unit_cube_hex(1);
    let all: Vec<u32> = (0..hex8.n_elements() as u32).collect();
    let (hex20, _) = amr::p_refine_hex8_to_hex20(&hex8, &all);
    let (hex27, _) = amr::p_refine_hex20_to_hex27(&hex20, &all);
    assert_eq!(hex27.element_type(0), ElementType::Hex27);

    let p1 = geometry_node_element(ElementType::Hex27, 1);
    assert_eq!(p1.n_dofs(), 8);
    let mut total = 0.0;
    for e in 0..hex27.n_elements() as u32 {
        let conn = hex27.element_nodes(e);
        let table: Vec<[f64; 3]> = conn
            .iter()
            .take(8)
            .map(|&nid| {
                let c = hex27.node_coords(nid);
                [c[0], c[1], c[2]]
            })
            .collect();
        total += element_volume(p1.as_ref(), &table, 10);
    }
    assert!(
        (total - 1.0).abs() < 1e-14,
        "Hex27 P1 geometry total volume: {total:.17e}"
    );

    // And the family the geometry readers use for order-2 tables remains the
    // full-tensor HexQk (27 dofs), whose own lattice integrates the unit cube
    // exactly — the truth source for the D612 convention reconciliation.
    let family = geometry_node_element(ElementType::Hex27, 2);
    let dc = family.dof_coords();
    let table: Vec<[f64; 3]> = dc
        .iter()
        .map(|c| [(c[0] + 1.0) / 2.0, (c[1] + 1.0) / 2.0, (c[2] + 1.0) / 2.0])
        .collect();
    let vol = element_volume(family.as_ref(), &table, 10);
    assert!((vol - 1.0).abs() < 1e-14, "HexQk(2) identity lattice: {vol:.17e}");
}

// ─── fixed_order_tensor extension (order-1 Hex20/Hex27 arm) ─────────────────

#[test]
fn d581_fixed_order_tensor_hex20_hex27_order1() {
    for et in [ElementType::Hex20, ElementType::Hex27] {
        let e = fixed_order_tensor(et, 1);
        assert_eq!(e.n_dofs(), 8, "{et:?} trilinear arm");
        assert_eq!(e.dof_coords()[0], vec![-1.0, -1.0, -1.0]);
    }
}
