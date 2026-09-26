//! D808-4 (residual, closed in round 78) — `build_tet4_pa_data`'s geometry on a
//! **curved** tet mesh.
//!
//! # The debt
//!
//! Round 76 gave the hex/quad PA kernels the mesh's order-`g` isoparametric
//! geometry (`pa::curved::curved_jacobian`, `geom_order <= 1` keeping the
//! verbatim straight path) but left `tet4` registered: its `PaData` was a
//! **single centroid quadrature point** (`PaData::new(n_elems, 1, 3)`), so even
//! substituting the curved Jacobian at that point could not represent a varying
//! `det J`/`J⁻ᵀJ⁻¹` — the fix is a rule-shaped layout, not a one-line swap.
//!
//! `build_tet4_pa_data` now takes two branches (the same `geom_order >= 2`
//! gate): a straight mesh keeps `nqp = 1` and the verbatim pre-fix arithmetic,
//! a curved one takes `tet_rule(2p+1)` (`p = 1`) with a per-QP
//! `(J⁻ᵀ, |det J|, κ(x_q))` — *exactly* the rule and the map
//! `Assembler::assemble_bilinear(.., 2p+1)` integrates.
//!
//! # What this file pins
//!
//! * [`d808r78_tet4_straight_meshes_are_byte_identical`] — the straight branch
//!   reproduces the **round-77 release's raw bits** on three straight fixtures
//!   (unit-cube tet mesh at two resolutions, one sheared).  The gold file is the
//!   verbatim pre-fix dump (`tmp/d78c/before_tet4_straight.txt`); the post-fix
//!   dump diffs against it byte for byte.
//! * [`d808r78_tet4_curved_pa_matches_assembly`] — on a curved multi-element
//!   fixture the PA apply reproduces the assembled SpMV to round-off, while the
//!   **pre-fix** route (the same fixture with the geometry table dropped, which
//!   *is* the single-centroid vertex-geometry data source) is `O(1e-1)` off.
//!   Teeth built inside the test, as in `d808r76_pa_curved_geometry.rs`.
//! * [`d808r78_tet4_layout_follows_geom_order`] — the layout gate itself.

use fem_assembly::pa::types::PaData;
use fem_assembly::pa::{build_tet4_pa_data, pa_apply_tet4};
use fem_assembly::standard::DiffusionIntegrator;
use fem_assembly::Assembler;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// The verbatim pre-round-78 dump: `[name] DATA_BITS <66> Y_BITS <8> Y_VALUES`.
const GOLDEN: &str = include_str!("data/d808r78_tet4_straight_bits.txt");

/// The pre-fix dump's κ field: a non-constant coefficient, so the QP physical
/// point is exercised too.
fn kappa(x: &[f64]) -> f64 {
    1.0 + 0.25 * x[0] - 0.125 * x[1] + 0.5 * x[2]
}

fn x_vector(n: usize) -> Vec<f64> {
    (0..n).map(|i| 1.0 + 0.125 * (i % 7) as f64).collect()
}

/// `(name, data bits, y bits)` blocks of the gold file.
fn golden_blocks(s: &str) -> Vec<(String, Vec<u64>, Vec<u64>)> {
    let mut out: Vec<(String, Vec<u64>, Vec<u64>)> = Vec::new();
    let mut cur: Option<(String, Vec<u64>, Vec<u64>)> = None;
    let mut section = 0u8;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("=== ") {
            if let Some(b) = cur.take() {
                out.push(b);
            }
            cur = Some((rest.split(':').next().unwrap().to_string(), Vec::new(), Vec::new()));
            section = 0;
        } else if line == "DATA_BITS" {
            section = 1;
        } else if line == "Y_BITS" {
            section = 2;
        } else if line == "Y_VALUES" {
            section = 3;
        } else if let Some(hex) = line.strip_prefix("0x") {
            let v = u64::from_str_radix(hex, 16).expect("hex bit pattern");
            match (section, cur.as_mut()) {
                (1, Some((_, d, _))) => d.push(v),
                (2, Some((_, _, y))) => y.push(v),
                _ => {}
            }
        }
    }
    if let Some(b) = cur.take() {
        out.push(b);
    }
    out
}

/// The three straight fixtures the pre-fix dump used, in the same order.
fn straight_fixtures() -> Vec<(String, Mesh<3>)> {
    let mut sheared = Mesh::<3>::unit_cube_tet(1);
    for k in 0..sheared.n_nodes() {
        let c: [f64; 3] = std::array::from_fn(|d| sheared.coords[k * 3 + d]);
        sheared.coords[k * 3] = c[0] + 0.3 * c[1];
        sheared.coords[k * 3 + 1] = c[1] - 0.2 * c[2];
        sheared.coords[k * 3 + 2] = 1.25 * c[2];
    }
    vec![
        ("unit_cube_tet(1)".to_string(), Mesh::<3>::unit_cube_tet(1)),
        ("unit_cube_tet(2)".to_string(), Mesh::<3>::unit_cube_tet(2)),
        ("sheared unit_cube_tet(1)".to_string(), sheared),
    ]
}

/// The straight-mesh case is **bit-identical** to the round-77 release: same
/// `PaData` (66 values on the 6-element fixture: `J⁻ᵀ`, `|det J|`, κ per
/// element) and same apply output, compared as raw `f64::to_bits`.
#[test]
fn d808r78_tet4_straight_meshes_are_byte_identical() {
    let gold = golden_blocks(GOLDEN);
    let fixtures = straight_fixtures();
    assert_eq!(gold.len(), fixtures.len(), "gold fixture count");

    let mut bad = Vec::new();
    for ((gname, gdata, gy), (fname, mesh)) in gold.iter().zip(fixtures.iter()) {
        assert_eq!(gname, fname, "fixture order");
        let pd = build_tet4_pa_data(mesh, &kappa);
        assert_eq!(pd.nqp, 1, "{fname}: a straight mesh must keep the single-QP layout");
        if pd.data.len() != gdata.len() {
            bad.push(format!("{fname}: PaData length {} vs gold {}", pd.data.len(), gdata.len()));
            continue;
        }
        for (i, (a, b)) in pd.data.iter().zip(gdata.iter()).enumerate() {
            if a.to_bits() != *b {
                bad.push(format!(
                    "{fname}: PaData[{i}] = {a:.17e} ({:#018x}) vs gold {:#018x}",
                    a.to_bits(),
                    b
                ));
            }
        }
        let space = H1Space::new(mesh.clone(), 1);
        let elem_dofs: Vec<Vec<u32>> =
            (0..mesh.n_elems() as u32).map(|e| space.element_dofs(e).to_vec()).collect();
        let x = x_vector(space.n_dofs());
        let mut y = vec![0.0_f64; space.n_dofs()];
        pa_apply_tet4(&pd, &elem_dofs, &x, &mut y);
        assert_eq!(y.len(), gy.len(), "{fname}: apply length");
        for (i, (a, b)) in y.iter().zip(gy.iter()).enumerate() {
            if a.to_bits() != *b {
                bad.push(format!("{fname}: y[{i}] = {a:.17e} vs gold {:#018x}", b));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "D808-4-r78: the straight branch moved ({} values):\n  {}",
        bad.len(),
        bad.iter().take(8).cloned().collect::<Vec<_>>().join("\n  ")
    );
}

// ─── The curved fixture ────────────────────────────────────────────────────

/// The fixture's smooth, vertex-vanishing bulge: only the **non-vertex**
/// geometry nodes move, so every element's corners stay exact while the P1
/// vertex map and the order-2 table map become different functions.  Amplitude
/// small enough that `det J > 0` everywhere (`assert_no_fold`).
fn bulge3(c: [f64; 3]) -> [f64; 3] {
    let s = (std::f64::consts::PI * c[0]).sin()
        * (std::f64::consts::PI * c[1]).sin()
        * (std::f64::consts::PI * c[2]).sin();
    [0.040 * s, 0.035 * s, 0.045 * s]
}

fn curved_tet_mesh() -> Mesh<3> {
    let mut mesh = Mesh::<3>::unit_cube_tet(2);
    let n_vertices = mesh.n_nodes();
    mesh.set_curvature(2);
    {
        let g = mesh.geometry.as_mut().expect("set_curvature(2) attaches a table");
        for node in n_vertices..g.n_nodes {
            let off = node * 3;
            let c = [g.coords[off], g.coords[off + 1], g.coords[off + 2]];
            let d = bulge3(c);
            for k in 0..3 {
                g.coords[off + k] += d[k];
            }
        }
    }
    mesh
}

/// The curved fixture with the geometry table dropped: the straight-edged P1
/// approximation the pre-round-78 PA always used (single centroid QP).
fn straightened_tet_mesh() -> Mesh<3> {
    let mut mesh = curved_tet_mesh();
    mesh.geometry = None;
    mesh
}

/// `|det J|` and the signed determinant must agree at every point the kernels
/// sample: the PA data stores `|det J|` while the assembled path carries the
/// signed determinant (D679), so a folded fixture would compare two different
/// quantities (round 76's Finding 4).
fn assert_no_fold(mesh: &Mesh<3>, name: &str) {
    let (pts, _w) = fem_element::quadrature::gauss_legendre_01(5);
    let mut min_det = f64::INFINITY;
    let mut at = (0u32, [0.0_f64; 3]);
    for e in 0..mesh.n_elems() as u32 {
        for &a in &pts {
            for &b in &pts {
                for &c in &pts {
                    let xi = [a, b, c];
                    let (jac, _) =
                        fem_mesh::transformation::element_jacobian_at(mesh, e, &xi, 3);
                    let det = jac.determinant();
                    if det < min_det {
                        min_det = det;
                        at = (e, xi);
                    }
                }
            }
        }
    }
    assert!(
        min_det > 0.0,
        "{name}: the curved fixture folds — min det J = {min_det:.6e} at element {} xi {:?}",
        at.0,
        at.1
    );
}

/// `(max |PA·x − A·x| / max |A·x|, DOFs shared by elements 0 and 1)`.
fn pa_vs_assembled(pd_mesh: &Mesh<3>, space_mesh: &Mesh<3>, order: u8) -> (f64, usize) {
    let space = H1Space::new(space_mesh.clone(), 1);
    let n = space.n_dofs();
    let a = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], order);
    let pd = build_tet4_pa_data(pd_mesh, &|_| 1.0);
    let elem_dofs: Vec<Vec<u32>> =
        (0..space_mesh.n_elems() as u32).map(|e| space.element_dofs(e).to_vec()).collect();
    let shared = elem_dofs[0].iter().filter(|d| elem_dofs[1].contains(d)).count();
    let x = x_vector(n);
    let mut y_pa = vec![0.0; n];
    pa_apply_tet4(&pd, &elem_dofs, &x, &mut y_pa);
    let mut y_asm = vec![0.0; n];
    a.spmv(&x, &mut y_asm);
    let num = y_pa.iter().zip(y_asm.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
    let den = y_asm.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    (num / den.max(1e-300), shared)
}

/// PA vs assembly on the curved fixture, at the rule the PA data carries
/// (`tet_rule(2p+1) = tet_rule(3)`), with the pre-fix route as the witness.
#[test]
fn d808r78_tet4_curved_pa_matches_assembly() {
    let curved = curved_tet_mesh();
    let straight = straightened_tet_mesh();
    assert_eq!(curved.geom_order(), 2, "the fixture must be curved");
    assert_eq!(straight.geom_order(), 1, "the witness must be straight");

    let mut results: Vec<(&str, f64, usize)> = Vec::new();
    for (name, pd_mesh, space_mesh) in [
        ("curved", &curved, &curved),
        ("pre-fix (vertex geoms, 1 centroid QP)", &straight, &curved),
    ] {
        let (rel, shared) = pa_vs_assembled(pd_mesh, space_mesh, 3);
        println!("tet4 {name}: shared dofs(e0,e1)={shared}  rel={rel:.3e}");
        results.push((name, rel, shared));
    }
    let (_, rel_curved, shared) = results[0];
    let (_, rel_prefix, _) = results[1];
    assert!(shared > 0, "the fixture has no DOFs shared by elements 0/1");
    assert!(
        rel_prefix > 1e-3,
        "the pre-fix witness differs by only {rel_prefix:.3e} — the pin has no teeth"
    );
    assert!(
        rel_curved < 1e-11,
        "curved tet4 PA vs assembled = {rel_curved:.3e} (pre-fix would be {rel_prefix:.3e})"
    );
}

/// The layout gate: `nqp == 1` for a straight mesh, the rule's point count for
/// a curved one.
#[test]
fn d808r78_tet4_layout_follows_geom_order() {
    let straight = Mesh::<3>::unit_cube_tet(2);
    assert_eq!(straight.geom_order(), 1);
    assert_eq!(build_tet4_pa_data(&straight, &|_| 1.0).nqp, 1);

    let curved = curved_tet_mesh();
    let pd: PaData = build_tet4_pa_data(&curved, &|_| 1.0);
    assert_eq!(curved.geom_order(), 2);
    assert_eq!(
        pd.nqp,
        fem_element::quadrature::tet_rule(3).points.len(),
        "the curved layout must be the assembled path's own rule"
    );
}

/// The curved fixture is genuinely curved (its order-2 map is not the vertex
/// map) and does not fold — the two preconditions of the comparison above.
#[test]
fn d808r78_tet4_curved_fixture_is_curved_and_unfolded() {
    let curved = curved_tet_mesh();
    assert_no_fold(&curved, "tet4 curved fixture");
    let g = curved.geometry.as_ref().expect("geometry table");
    assert!(
        g.n_nodes > curved.n_nodes(),
        "curved fixture: geometry nodes {} vs vertices {}",
        g.n_nodes,
        curved.n_nodes()
    );
    // The vertex map and the table map differ at the element centre.
    let pd_curved = build_tet4_pa_data(&curved, &|_| 1.0);
    let pd_straight = build_tet4_pa_data(&straightened_tet_mesh(), &|_| 1.0);
    let d = pd_curved.data[9] - pd_straight.data[9];
    assert!(d.abs() > 1e-6, "the table must change |det J| at the centroid: {d:.3e}");
}
