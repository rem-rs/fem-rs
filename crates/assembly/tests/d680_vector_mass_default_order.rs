//! D680: `VectorMassIntegrator`'s **default** quadrature order must reproduce
//! MFEM's `VectorFEMassIntegrator` per-element rule
//! `Trans.OrderW() + 2·el.GetOrder()` with the family-aware FE order —
//! `GetOrder() = p + 1` for every RT family member (`fe/fe_rt.cpp`
//! constructors) and `= p` for ND (`fe/fe_nd.cpp`).
//!
//! # Why the family must reach the integrator
//!
//! The previous blind default `2k + 3` (affine-quad value) under-integrated
//! hex RT by one 1D point: affine-hex `OrderW() = 2` gives MFEM's RT0 hex
//! default order 4 → a 3×3×3 (27-point) rule, where `2k + 3` picked 2×2×2.
//! Both rules integrate the RT0 mass polynomial exactly, but their summation
//! round-off differs and leaks into noise-level solve outputs (ex24 `-p 1`
//! iteration residuals, D700).  A family-blind value cannot be right for both
//! families on hexes (RT needs `2p + 4`, ND `2p + 2`), so the assembler now
//! passes the space family through `integration_order_for_space`.
//!
//! # Truth provenance
//!
//! `tmp/d680/` + `data/d680_vector_mass_mfem.txt`: MFEM 4.10 serial probe
//! (`tmp/d680/../d700/probe_d680_fixtures.cpp`, compiled
//! `g++ -std=c++17 -O2 -I$HOME/mfem410_ser … $HOME/mfem410_ser/libmfem.a`)
//! dumping `VectorFEMassIntegrator::AssembleElementMatrix` on the single
//! affine hex `data/d680_unit_hex_mfem.mesh` (`Mesh::MakeCartesian3D(1,1,1)`)
//! for RT0 / RT1 / ND1, each at its own default rule, plus the rule point
//! counts.

use fem_assembly::standard::VectorMassIntegrator;
use fem_assembly::VectorAssembler;
use fem_assembly::VectorBilinearIntegrator;
use fem_io::mfem::read_mfem_file;
use fem_linalg::CooMatrix;
use fem_mesh::element_type::ElementType;
use fem_space::fe_space::SpaceType;
use fem_space::{HCurlSpace, HDivSpace};

const FIXTURES: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/d680_vector_mass_mfem.txt"
));
const MESH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/d680_unit_hex_mfem.mesh");

/// Parse the probe dump: `NAME rows cols` followed by `rows` lines of values.
fn parse_fixture() -> std::collections::HashMap<String, Vec<Vec<f64>>> {
    let mut out = std::collections::HashMap::new();
    let mut lines = FIXTURES.lines().peekable();
    while let Some(line) = lines.next() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() != 3 || tokens[1].parse::<usize>().is_err() {
            continue;
        }
        let name = tokens[0].to_string();
        let rows: usize = tokens[1].parse().unwrap();
        let cols: usize = tokens[2].parse().unwrap();
        let mut mat = Vec::with_capacity(rows);
        for _ in 0..rows {
            let vals: Vec<f64> = lines
                .next()
                .unwrap()
                .split_whitespace()
                .map(|t| t.parse().unwrap())
                .collect();
            assert_eq!(vals.len(), cols);
            mat.push(vals);
        }
        out.insert(name, mat);
    }
    out
}

/// Assemble the global matrix on the single-element mesh through the
/// top-level assembler, so the integrator's **own default rule** governs the
/// quadrature (the fallback `quad_order` argument is overridden) — then
/// unmangle the space-interface signs the D667 way.  On the single-element
/// mesh the global matrix IS the element matrix.
fn default_rule_global_matrix<S: fem_space::fe_space::FESpace>(
    space: &S,
    integ: &dyn VectorBilinearIntegrator,
) -> Vec<Vec<f64>> {
    let n = space.n_dofs();
    let csr = VectorAssembler::assemble_bilinear(space, &[integ], 1);
    let signs = space.element_signs(0).unwrap_or(&[1.0]).to_vec();
    let dofs: Vec<usize> = space.element_dofs(0).iter().map(|&d| d as usize).collect();
    let mut mat = vec![vec![0.0; n]; n];
    for (i, &gi) in dofs.iter().enumerate() {
        for (j, &gj) in dofs.iter().enumerate() {
            mat[i][j] = signs[i] * signs[j] * csr.get(gi, gj);
        }
    }
    mat
}

#[test]
fn d680_mfem_default_rule_policy_table() {
    let integ = VectorMassIntegrator { alpha: 1.0 };
    // (family, collection order k, geometry) → MFEM OrderW + 2·GetOrder.
    let cases: &[(SpaceType, u8, ElementType, u8)] = &[
        // Hex: OrderW = 2 (affine), 5 (curved P2).
        (SpaceType::HDiv, 0, ElementType::Hex8, 4),
        (SpaceType::HDiv, 1, ElementType::Hex8, 6),
        (SpaceType::HDiv, 2, ElementType::Hex8, 8),
        (SpaceType::HDiv, 1, ElementType::Hex27, 9),
        (SpaceType::HCurl, 0, ElementType::Hex8, 2),
        (SpaceType::HCurl, 1, ElementType::Hex8, 4),
        (SpaceType::HCurl, 1, ElementType::Hex27, 7),
        // Quad: OrderW = 1 (affine) → the historical blind value survives.
        (SpaceType::HDiv, 0, ElementType::Quad4, 3),
        (SpaceType::HDiv, 1, ElementType::Quad4, 5),
        (SpaceType::HCurl, 1, ElementType::Quad4, 3),
        // Simplex: OrderW = (g−1)·dim.
        (SpaceType::HDiv, 0, ElementType::Tri3, 2),
        (SpaceType::HDiv, 1, ElementType::Tri6, 6),
        (SpaceType::HDiv, 0, ElementType::Tet4, 2),
        (SpaceType::HCurl, 1, ElementType::Tri3, 2),
        (SpaceType::HCurl, 2, ElementType::Tet10, 7),
        // Prism (Qk, OrderW = 2/5) and pyramid (Uk, OrderW = 2).
        (SpaceType::HDiv, 0, ElementType::Prism6, 4),
        (SpaceType::HDiv, 1, ElementType::Prism15, 9),
        (SpaceType::HDiv, 0, ElementType::Pyramid5, 4),
    ];
    for &(family, k, et, want) in cases {
        let got = integ
            .integration_order_for_space(family, k, et)
            .unwrap_or_else(|| panic!("{family:?} k={k} {et:?}: no default"));
        assert_eq!(got, want, "{family:?} k={k} {et:?}");
    }
    // The family-blind legacy hook keeps its affine-quad compromise.
    assert_eq!(integ.integration_order(0), Some(3));
    assert_eq!(integ.integration_order(1), Some(5));

    // D690 geom-aware rows: a curved (P2) map lifts OrderW to 3g−1 on the
    // Qk/Uk geometries even when the connectivity stays linear — the
    // multidomain_rt cylinder case (RT1 hex, g = 2 → order 9, D667 probe).
    assert_eq!(
        integ.integration_order_for_space_geom(SpaceType::HDiv, 1, ElementType::Hex8, 2),
        Some(9)
    );
    assert_eq!(
        integ.integration_order_for_space_geom(SpaceType::HDiv, 0, ElementType::Hex8, 2),
        Some(7)
    );
    assert_eq!(
        integ.integration_order_for_space_geom(SpaceType::HCurl, 1, ElementType::Hex8, 2),
        Some(7)
    );
    // Affine meshes reproduce the 3-arg table bit-for-bit.
    assert_eq!(
        integ.integration_order_for_space_geom(SpaceType::HDiv, 0, ElementType::Hex8, 1),
        Some(4)
    );
    assert_eq!(
        integ.integration_order_for_space_geom(SpaceType::HDiv, 0, ElementType::Hex8, 0),
        Some(4)
    );
}

#[test]
fn d680_rt0_rt1_nd1_hex_mass_matches_mfem_default_rule() {
    let fixtures = parse_fixture();
    let mfem = read_mfem_file(MESH).expect("read d680 unit hex");
    let mesh3 = mfem.mesh3d.expect("3d mesh");

    // (name, collection order)
    let cases: &[(&str, u8)] = &[("RT0", 0), ("RT1", 1)];
    for &(name, order) in cases {
        let space = HDivSpace::new(mesh3.clone(), order);
        let integ = VectorMassIntegrator { alpha: 1.0 };
        let got = default_rule_global_matrix(&space, &integ);
        let want = &fixtures[&format!("{name}_HexAFF")];
        assert_eq!(got.len(), want.len());
        for (i, (grow, wrow)) in got.iter().zip(want.iter()).enumerate() {
            for (j, (&g, &w)) in grow.iter().zip(wrow.iter()).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-11,
                    "{name} hex RT mass ({i},{j}): {g} vs MFEM {w}"
                );
            }
        }
    }

    let nd = HCurlSpace::new(mesh3.clone(), 1);
    let integ = VectorMassIntegrator { alpha: 1.0 };
    let got = default_rule_global_matrix(&nd, &integ);
    let want = &fixtures["ND1_HexAFF"];
    for (i, (grow, wrow)) in got.iter().zip(want.iter()).enumerate() {
        for (j, (&g, &w)) in grow.iter().zip(wrow.iter()).enumerate() {
            assert!(
                (g - w).abs() <= 1e-11,
                "ND1 hex mass ({i},{j}): {g} vs MFEM {w}"
            );
        }
    }
}

#[test]
fn d680_rule_point_counts_match_mfem_picks() {
    // The 1D point count behind each default: order 4 → 3/dim (27 pts),
    // order 6 → 4/dim (64 pts) — matching the probe's rule dumps.
    let read_tag = |tag: &str| -> usize {
        FIXTURES
            .lines()
            .find(|l| l.starts_with(tag))
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap()
    };
    assert_eq!(read_tag("RT0_RULE"), 27);
    assert_eq!(read_tag("RT1_RULE"), 64);
    assert_eq!(read_tag("ND1_RULE"), 27);
}
