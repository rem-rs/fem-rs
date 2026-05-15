//! Integration tests for fem-io: GMSH reader and VTK writer.

use fem_io::{
    abaqus::read_abaqus_inp,
    gmsh::read_msh,
    netgen::{read_netgen_vol, write_netgen_vol},
    vtk::{DataArray, VtkWriter},
};
use fem_mesh::{topology::MeshTopology, SimplexMesh};

fn gmsh_v2_tri6_unit_triangle() -> &'static str {
    "$MeshFormat\n\
     2.2 0 8\n\
     $EndMeshFormat\n\
     $PhysicalNames\n\
     2\n\
     1 1 \"boundary\"\n\
     2 2 \"domain\"\n\
     $EndPhysicalNames\n\
     $Nodes\n\
     6\n\
     1 0.0 0.0 0.0\n\
     2 1.0 0.0 0.0\n\
     3 0.0 1.0 0.0\n\
     4 0.5 0.0 0.0\n\
     5 0.5 0.5 0.0\n\
     6 0.0 0.5 0.0\n\
     $EndNodes\n\
     $Elements\n\
     4\n\
     1 8 2 1 1 1 4 2\n\
     2 8 2 1 1 2 5 3\n\
     3 8 2 1 1 3 6 1\n\
     4 9 2 2 1 1 2 3 4 5 6\n\
     $EndElements\n"
}

// ---------------------------------------------------------------------------
// GMSH reader tests
// ---------------------------------------------------------------------------

/// Parse the minimal unit-square fixture with 8 triangles.
#[test]
fn gmsh_unit_square_parse() {
    let msh_src = include_str!("fixtures/unit_square.msh");
    let msh = read_msh(msh_src.as_bytes()).unwrap();
    let mesh = msh.into_2d().unwrap();

    // 9 nodes (4 corners + 4 edge midpoints + 1 centre)
    assert_eq!(mesh.n_nodes(), 9, "expected 9 nodes");
    // 8 triangles
    assert_eq!(mesh.n_elems(), 8, "expected 8 triangles");
    // 8 boundary edges
    assert_eq!(mesh.n_faces(), 8, "expected 8 boundary edges");

    // All node coords should be in [0,1]
    for n in 0..mesh.n_nodes() as u32 {
        let c = mesh.node_coords(n);
        for &x in c {
            assert!((-1e-12..=1.0 + 1e-12).contains(&x), "coord out of range: {x}");
        }
    }
    // Check passes internal consistency
    mesh.check().unwrap();
}

#[test]
fn gmsh_tag_names_populated() {
    let msh_src = include_str!("fixtures/unit_square.msh");
    let msh = read_msh(msh_src.as_bytes()).unwrap();
    assert!(!msh.tag_names.is_empty(), "tag_names should be non-empty");
    assert!(msh.tag_names.values().any(|n| n == "domain"), "expected 'domain' tag");
}

#[test]
fn gmsh_physical_groups_populated() {
    let msh_src = include_str!("fixtures/unit_square.msh");
    let msh = read_msh(msh_src.as_bytes()).unwrap();
    assert_eq!(msh.physical_groups.len(), 5, "5 physical groups expected");
}

#[test]
fn gmsh_tri6_named_attributes_and_vtk_result_workflow() {
    let msh = read_msh(gmsh_v2_tri6_unit_triangle().as_bytes()).expect("parse Tri6 fixture");

    let named = msh.named_attribute_registry();
    let domain = named.get("domain").expect("missing domain attribute set");
    let boundary = named.get("boundary").expect("missing boundary attribute set");
    assert!(domain.has_element_tag(2), "domain should map to element tag 2");
    assert!(boundary.has_boundary_tag(1), "boundary should map to boundary tag 1");

    let curved = msh.curved2d.as_ref().expect("missing curved Tri6 mesh");
    assert_eq!(curved.geom_order, 2, "Tri6 import should produce order-2 geometry");
    assert_eq!(curved.n_nodes, 6, "Tri6 curved mesh should keep all 6 geometry nodes");
    assert_eq!(curved.face_tags, vec![1, 1, 1], "Line3 boundary tags should be preserved");

    let mesh = msh.mesh2d.as_ref().expect("missing working 2D mesh");
    assert_eq!(mesh.elem_tags, vec![2], "volume material tag should be preserved");
    assert_eq!(mesh.face_tags, vec![1, 1, 1], "boundary tags should be preserved on linear mesh");

    let temp: Vec<f64> = (0..mesh.n_nodes())
        .map(|i| {
            let coords = mesh.node_coords(i as u32);
            let x = coords[0];
            let y = coords[1];
            x + 2.0 * y
        })
        .collect();
    let material_id = vec![mesh.elem_tags[0] as f64];

    let mut writer = VtkWriter::new(mesh);
    writer.add_point_data(DataArray::scalars("temperature", temp));
    writer.add_cell_data(DataArray::scalars("material_id", material_id));

    let mut buf = Vec::<u8>::new();
    writer.write(&mut buf).expect("write vtu");
    let xml = String::from_utf8(buf).expect("utf8 vtk xml");

    assert!(xml.contains(r#"Name="temperature""#), "point result field should be exported");
    assert!(xml.contains(r#"Name="material_id""#), "cell material field should be exported");
    assert!(xml.contains("NumberOfPoints=\"6\""), "linearized Tri6 working mesh should export 6 nodes");
    assert!(xml.contains("NumberOfCells=\"1\""), "single imported domain element should export one cell");
}

#[test]
fn netgen_unit_tet_parse() {
    let vol_src = include_str!("fixtures/unit_tet.vol");
    let mesh = read_netgen_vol(vol_src.as_bytes()).unwrap();

    assert_eq!(mesh.n_nodes(), 4, "expected 4 nodes");
    assert_eq!(mesh.n_elems(), 1, "expected 1 tet");
    assert_eq!(mesh.n_faces(), 4, "expected 4 boundary faces");
    assert_eq!(mesh.elem_tags, vec![1]);
    assert!(mesh.face_tags.iter().all(|&t| t == 1));
    mesh.check().unwrap();
}

#[test]
fn netgen_dimension_2_is_rejected() {
    let src = r#"
dimension
2
points
3
0 0 0
1 0 0
0 1 0
volumeelements
1
1 3 1 2 3
"#;
    let err = read_netgen_vol(src.as_bytes()).expect_err("2D .vol should be rejected in Tet baseline");
    assert!(format!("{err}").contains("dimension=3"));
}

#[test]
fn netgen_write_then_read_roundtrip() {
    let mesh = SimplexMesh::<3>::unit_cube_tet(1);
    let mut buf = Vec::new();
    write_netgen_vol(&mesh, &mut buf).unwrap();

    let parsed = read_netgen_vol(buf.as_slice()).unwrap();
    assert_eq!(parsed.n_nodes(), mesh.n_nodes());
    assert_eq!(parsed.n_elems(), mesh.n_elems());
    assert_eq!(parsed.elem_tags.len(), mesh.n_elems());
    assert_eq!(parsed.n_faces(), 12, "unit cube tet(1) should expose 12 boundary triangles");
    parsed.check().unwrap();
}

#[test]
fn netgen_mixed_tet_hex_parse() {
    let vol_src = include_str!("fixtures/mixed_tet_hex.vol");
    let mesh = read_netgen_vol(vol_src.as_bytes()).unwrap();

    assert_eq!(mesh.n_nodes(), 8);
    assert_eq!(mesh.n_elems(), 2);
    assert_eq!(mesh.elem_tags, vec![1, 2]);
    assert!(mesh.elem_types.is_some(), "elem_types should be populated for mixed mesh");
    assert!(mesh.elem_offsets.is_some(), "elem_offsets should be populated for mixed mesh");
    assert!(mesh.face_types.is_some(), "face_types should be populated for mixed boundary faces");
    assert!(mesh.face_offsets.is_some(), "face_offsets should be populated for mixed boundary faces");
    assert_eq!(mesh.n_faces(), 10, "tet(4) + hex(6) boundary faces expected in fixture");
    mesh.check().unwrap();
}

#[test]
fn abaqus_unit_tet_parse() {
    let inp_src = include_str!("fixtures/unit_tet.inp");
    let mesh = read_abaqus_inp(inp_src.as_bytes()).unwrap();

    assert_eq!(mesh.n_nodes(), 4);
    assert_eq!(mesh.n_elems(), 1);
    assert_eq!(mesh.n_faces(), 4);
    assert_eq!(mesh.elem_tags, vec![1]);
    mesh.check().unwrap();
}

#[test]
fn abaqus_mixed_element_types_parse() {
    let inp_src = include_str!("fixtures/mixed_c3d4_c3d8.inp");
    let mesh = read_abaqus_inp(inp_src.as_bytes())
        .expect("mixed C3D4/C3D8 should parse in mixed-element baseline");

    assert_eq!(mesh.n_nodes(), 8);
    assert_eq!(mesh.n_elems(), 2);
    assert_eq!(mesh.elem_tags.len(), 2);
    assert!(mesh.elem_types.is_some(), "elem_types should be populated for mixed mesh");
    assert!(mesh.elem_offsets.is_some(), "elem_offsets should be populated for mixed mesh");
    assert!(mesh.face_types.is_some(), "face_types should be populated for mixed boundary faces");
    assert!(mesh.face_offsets.is_some(), "face_offsets should be populated for mixed boundary faces");
    assert_eq!(mesh.n_faces(), 10, "tet(4) + hex(6) boundary faces expected in fixture");
    mesh.check().unwrap();
}

// ---------------------------------------------------------------------------
// VTK writer tests
// ---------------------------------------------------------------------------

/// Write a mesh to a buffer and verify the XML is syntactically valid.
#[test]
fn vtk_write_2d_mesh() {
    let mesh = SimplexMesh::<2>::unit_square_tri(4);
    let n    = mesh.n_nodes();
    let e    = mesh.n_elems();
    let w    = VtkWriter::new(&mesh);
    let mut buf = Vec::<u8>::new();
    w.write(&mut buf).unwrap();
    let xml = String::from_utf8(buf).unwrap();
    assert!(xml.contains(&format!("NumberOfPoints=\"{n}\"")));
    assert!(xml.contains(&format!("NumberOfCells=\"{e}\"")));
    assert!(xml.contains("</VTKFile>"));
}

#[test]
fn vtk_write_3d_mesh() {
    let mesh = SimplexMesh::<3>::unit_cube_tet(2);
    let w    = VtkWriter::new(&mesh);
    let mut buf = Vec::<u8>::new();
    w.write(&mut buf).unwrap();
    let xml = String::from_utf8(buf).unwrap();
    let n = mesh.n_nodes();
    assert!(xml.contains(&format!("NumberOfPoints=\"{n}\"")));
}

/// Write a mesh + scalar solution field, then verify the field appears.
#[test]
fn vtk_write_poisson_solution() {
    use std::f64::consts::PI;
    let mesh  = SimplexMesh::<2>::unit_square_tri(8);
    let n     = mesh.n_nodes();

    // "Exact" solution values at nodes.
    let u: Vec<f64> = (0..n).map(|i| {
        let x = mesh.node_coords(i as u32)[0];
        let y = mesh.node_coords(i as u32)[1];
        (PI * x).sin() * (PI * y).sin()
    }).collect();

    // Element-wise pressure (cell data).
    let p = vec![1.0_f64; mesh.n_elems()];

    let mut w = VtkWriter::new(&mesh);
    w.add_point_data(DataArray::scalars("u", u.clone()));
    w.add_cell_data(DataArray::scalars("pressure", p));

    let mut buf = Vec::<u8>::new();
    w.write(&mut buf).unwrap();
    let xml = String::from_utf8(buf).unwrap();

    assert!(xml.contains(r#"Name="u""#));
    assert!(xml.contains(r#"Name="pressure""#));
    assert!(xml.contains("<PointData>"));
    assert!(xml.contains("<CellData>"));

    // Verify the first DOF value appears in the output (node 0 at origin → sin(0)=0).
    assert!(xml.contains("0.0000000000e0") || xml.contains("0.0000000000e"),
        "expected zero value for node at origin");
}

/// Write to a temp file and read it back to confirm non-empty.
#[test]
fn vtk_write_file_roundtrip() {
    let mesh = SimplexMesh::<2>::unit_square_tri(4);
    let w    = VtkWriter::new(&mesh);

    let tmp = std::env::temp_dir().join("fem_rs_test_output.vtu");
    w.write_file(&tmp).unwrap();

    let content = std::fs::read_to_string(&tmp).unwrap();
    assert!(content.contains("UnstructuredGrid"));
    std::fs::remove_file(&tmp).ok();
}
