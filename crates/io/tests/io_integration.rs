//! Integration tests for fem-io: GMSH reader and VTK writer.

use fem_io::{
    gmsh::read_msh,
    vtk::{DataArray, VtkWriter},
};
use fem_mesh::{topology::MeshTopology, SimplexMesh};

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
            assert!(x >= -1e-12 && x <= 1.0 + 1e-12, "coord out of range: {x}");
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
