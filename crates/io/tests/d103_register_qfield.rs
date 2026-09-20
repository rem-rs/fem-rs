//! D104a (round 54): `DataCollection::RegisterQField` — the QP-side grid
//! function registration/save chain.
//!
//! C++ anchors:
//! * `VisItDataCollection::RegisterQField` (`fem/datacollection.cpp:454-478`):
//!   basis pattern `QF_{ORDER}_{VDIM}`, root `assoc: "quadrature"`, LOD =
//!   max over elements of `GeometryRefiner::GetRefinementLevelFromElems`
//!   (`fem/geom.cpp:1972`) starting at -1.
//! * `QuadratureSpace::Save` (`fem/qspace.cpp:187`) +
//!   `QuadratureFunction::Save` (`fem/qfunction.cpp:50`): the slice header is
//!   `QuadratureSpace / Type: default_quadrature / Order: N / VDim: N` and the
//!   values always print `vdim` per line (`Vector::Print(os, vdim)`).
//! * `VisItDataCollection::LoadFields` (`fem/datacollection.cpp:633-664`):
//!   load dispatches on `assoc` — `"nodes"` rebuilds a `GridFunction`,
//!   `"quadrature"` (or legacy `"elements"`) a `QuadratureFunction`.
//!
//! No compiled MFEM is available in this environment, so the expectations are
//! structural assertions derived from those source lines (the same method the
//! byte-exact `nodes` tests in `data_collection.rs` used for their original
//! verification).

use fem_io::data_collection::{
    qfield_lod, read_gf_slice, read_visit_root, DcAssoc, DcField, DcGeom, VisItCollection,
};

/// One-hex 3-D MFEM mesh slice (same fixture style as `data_collection_load`).
const HEX_MESH_TXT: &str = concat!(
    "MFEM mesh v1.0\n",
    "\n",
    "dimension\n",
    "3\n",
    "\n",
    "elements\n",
    "1\n",
    "1 5 0 1 2 3 4 5 6 7\n",
    "\n",
    "boundary\n",
    "6\n",
    "1 3 0 1 2 3\n",
    "1 3 1 5 6 2\n",
    "1 3 0 4 5 1\n",
    "1 3 3 7 6 2\n",
    "1 3 0 4 7 3\n",
    "1 3 4 5 6 7\n",
    "\n",
    "vertices\n",
    "8\n",
    "3\n",
    "0 0 0\n",
    "1 0 0\n",
    "1 1 0\n",
    "0 1 0\n",
    "0 0 1\n",
    "1 0 1\n",
    "1 1 1\n",
    "0 1 1\n",
);

/// `GeometryRefiner::GetRefinementLevelFromElems` (geom.cpp:1972-2006):
/// tensor-product point counts give `LOD = n - 1` for `n^dim` points,
/// segments take the point count directly, points are -1, and non-square
/// counts (e.g. a simplex rule with 6 points on a triangle) find no `n`
/// and return -1.
#[test]
fn qfield_lod_matches_get_refinement_level_from_elems() {
    // SQUARE / TRIANGLE: n*n == n_qp -> n-1.
    assert_eq!(qfield_lod(DcGeom::Square, 1), 0);
    assert_eq!(qfield_lod(DcGeom::Square, 4), 1);
    assert_eq!(qfield_lod(DcGeom::Square, 9), 2);
    assert_eq!(qfield_lod(DcGeom::Square, 16), 3);
    assert_eq!(qfield_lod(DcGeom::Triangle, 9), 2);
    // No n < 15 with n*n == 6 (a 2nd-order triangle rule): -1, as in C++.
    assert_eq!(qfield_lod(DcGeom::Triangle, 6), -1);
    assert_eq!(qfield_lod(DcGeom::Square, 6), -1);
    // CUBE / TETRAHEDRON / PRISM: n*n*n == n_qp -> n-1.
    assert_eq!(qfield_lod(DcGeom::Cube, 1), 0);
    assert_eq!(qfield_lod(DcGeom::Cube, 8), 1);
    assert_eq!(qfield_lod(DcGeom::Cube, 27), 2);
    assert_eq!(qfield_lod(DcGeom::Cube, 64), 3);
    assert_eq!(qfield_lod(DcGeom::Prism, 8), 1);
    // A 1st-order tet rule has 4 points; no n^3 matches -> -1.
    assert_eq!(qfield_lod(DcGeom::Tetrahedron, 4), -1);
    assert_eq!(qfield_lod(DcGeom::Tetrahedron, 8), 1);
    // SEGMENT: the point count itself; POINT: -1.
    assert_eq!(qfield_lod(DcGeom::Segment, 4), 4);
    assert_eq!(qfield_lod(DcGeom::Point, 1), -1);
}

/// Pyramids `MFEM_ABORT` in `GetRefinementLevelFromElems` (geom.cpp:1986).
#[test]
#[should_panic(expected = "Reference element type is not supported!")]
fn qfield_lod_pyramid_aborts_like_cpp() {
    let _ = qfield_lod(DcGeom::Pyramid, 8);
}

/// The `RegisterQField` save chain: the root carries `assoc: "quadrature"`
/// with the `QF_{ORDER}_{VDIM}` basis and the LOD derived from the element
/// quadrature-point counts; the slice file carries the `QuadratureSpace`
/// header with `vdim` values per line.  A nodal field registered alongside
/// keeps its own `assoc: "nodes"` root tag (one `fields` map, as in C++).
#[test]
fn register_qfield_root_and_slice_format() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut dc = VisItCollection::new("QF");
    dc.set_prefix_path(dir.path().to_str().unwrap());
    dc.set_cycle(0);
    dc.spatial_dim = 3;
    dc.topo_dim = 3;

    // Quadrature field on the single hex: order-2 tensor rule -> 3^3 = 27
    // QPs, so `GetRefinementLevelFromElems(CUBE, 27)` = 3 - 1 = 2 and the QF
    // holds 27 values.
    let lod = qfield_lod(DcGeom::Cube, 27).max(0) as u32;
    assert_eq!(lod, 2);
    let w_values: Vec<f64> = (0..27).map(|i| 0.5 + i as f64).collect();
    dc.register_qfield(DcField::quadrature("W", 2, 1, w_values).with_lod(lod));

    // A nodal field in the same collection: 8 H1 P1 DOFs.
    dc.register_field(DcField::nodes("u", "H1_3D_P1", 1, 1, vec![0.0; 8]));

    dc.save(0, HEX_MESH_TXT).expect("save failed");
    let step = dir.path().join("QF_000000");
    assert!(step.join("mesh.000000").exists());
    assert!(step.join("W.000000").exists());
    assert!(step.join("u.000000").exists());
    assert!(dir.path().join("QF_000000.mfem_root").exists());

    // Root: both fields share the `fields` map; only the assoc tags differ.
    let root_json = std::fs::read_to_string(dir.path().join("QF_000000.mfem_root")).unwrap();
    let w_pos = root_json.find("\"W\"").expect("W entry");
    let u_pos = root_json.find("\"u\"").expect("u entry");
    let w_entry = &root_json[w_pos..u_pos.min(w_pos + 500)];
    assert!(w_entry.contains("\"assoc\": \"quadrature\""), "{w_entry}");
    assert!(w_entry.contains("\"basis\": \"QF_2_1\""), "{w_entry}");
    assert!(w_entry.contains("\"lod\": \"2\""), "{w_entry}");
    assert!(w_entry.contains("\"order\": \"2\""), "{w_entry}");
    let u_entry = &root_json[u_pos..];
    assert!(u_entry.contains("\"assoc\": \"nodes\""), "{u_entry}");
    assert!(u_entry.contains("\"basis\": \"H1_3D_P1\""), "{u_entry}");

    // Slice: `QuadratureSpace::Save` + `QuadratureFunction::Save` layout —
    // no FiniteElementSpace block, no Ordering line, VDim values per line.
    let w_txt = std::fs::read_to_string(step.join("W.000000")).unwrap();
    let w_lines: Vec<String> = (0..27).map(|i| format!("{}", 0.5 + i as f64)).collect();
    assert_eq!(
        w_txt,
        format!(
            "QuadratureSpace\nType: default_quadrature\nOrder: 2\nVDim: 1\n\n{}\n",
            w_lines.join("\n")
        )
    );
    // The nodal slice keeps the FiniteElementSpace header.
    let u_txt = std::fs::read_to_string(step.join("u.000000")).unwrap();
    assert!(u_txt.starts_with("FiniteElementSpace\n"), "{u_txt}");
}

/// `LoadFields` dispatch equivalence: a saved collection whose quadrature
/// field round-trips through `read_visit_root` (assoc/basis/order/lod tags)
/// and `read_gf_slice` (the `QuadratureSpace` header parses back to the
/// `QF_{ORDER}_{VDIM}` basis with the values intact).
#[test]
fn qfield_roundtrip_through_load_chain() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut dc = VisItCollection::new("QFRT");
    dc.set_prefix_path(dir.path().to_str().unwrap());
    dc.set_cycle(7);
    dc.set_time(0.125);
    dc.spatial_dim = 3;
    dc.topo_dim = 3;

    // vdim 3 (a vector-valued QF): 27 QPs x 3 comps = 81 values, printed 3
    // per line.  Values are chosen to survive precision-6 formatting exactly.
    let sigma_values: Vec<f64> = (0..81).map(|i| 1.0 + (i % 7) as f64 * 0.25).collect();
    let lod = qfield_lod(DcGeom::Cube, 27).max(0) as u32;
    dc.register_qfield(DcField::quadrature("sigma", 2, 3, sigma_values).with_lod(lod));
    dc.save(0, HEX_MESH_TXT).expect("save failed");

    // Root metadata: assoc dispatch tag + QF basis + derived LOD.
    let root = dir.path().join("QFRT_000007.mfem_root");
    let (cycle, domains, fields) = read_visit_root(&root).expect("read root failed");
    assert_eq!(cycle, 7);
    assert_eq!(domains, 1);
    assert_eq!(fields.len(), 1);
    assert_eq!(fields[0].name, "sigma");
    assert_eq!(fields[0].assoc, DcAssoc::Quadrature, "assoc tag must dispatch to the QF arm");
    assert_eq!(fields[0].basis, "QF_2_3");
    assert_eq!(fields[0].order, 2);
    assert_eq!(fields[0].lod, 2);
    assert_eq!(fields[0].vdim, 3);

    // Slice: values round-trip at stream precision 6 (3 per line).
    let (basis, vdim, values) =
        read_gf_slice(&dir.path().join("QFRT_000007/sigma.000000")).expect("read slice failed");
    assert_eq!(basis, "QF_2_3", "the QuadratureSpace header maps back to the QF basis pattern");
    assert_eq!(vdim, 3);
    let expected: Vec<f64> = (0..81).map(|i| 1.0 + (i % 7) as f64 * 0.25).collect();
    assert_eq!(values, expected);
}

/// A vdim-2 quadrature field prints two values per line, and the slice reader
/// accepts the file in place (`LoadFields` would rebuild a QuadratureFunction
/// from exactly this layout, `QuadratureFunction(Mesh*, istream)`).
#[test]
fn qfield_vector_values_print_vdim_per_line() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut dc = VisItCollection::new("QFV");
    dc.set_prefix_path(dir.path().to_str().unwrap());
    dc.set_cycle(0);
    dc.spatial_dim = 3;
    dc.topo_dim = 3;
    dc.register_qfield(DcField::quadrature(
        "q",
        2,
        2,
        vec![1.0, 2.0, 3.0, 4.0],
    ));
    dc.save(0, HEX_MESH_TXT).expect("save failed");

    let txt = std::fs::read_to_string(dir.path().join("QFV_000000/q.000000")).unwrap();
    assert_eq!(
        txt,
        "QuadratureSpace\nType: default_quadrature\nOrder: 2\nVDim: 2\n\n1 2\n3 4\n"
    );
    let (basis, vdim, values) =
        read_gf_slice(&dir.path().join("QFV_000000/q.000000")).expect("read slice failed");
    assert_eq!(basis, "QF_2_2");
    assert_eq!(vdim, 2);
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
}
