//! Integration tests for fem-io: GMSH reader and VTK writer.

use fem_io::{
    abaqus::read_abaqus_inp,
    gmsh::read_msh,
    netgen::{read_netgen_vol, write_netgen_vol},
    vtk::{DataArray, VtkWriter},
    FIELD_MATERIAL_ID,
    FIELD_TEMPERATURE,
    vtk_nodal_workflow_fields,
};
use fem_mesh::{topology::MeshTopology, SimplexMesh};

#[cfg(feature = "hdf5")]
use fem_io::{
    abaqus::read_abaqus_inp_full,
    read_mesh_and_fields,
    write_mesh_and_fields,
    write_xdmf,
    vtk_abaqus_solution_fields,
    vtk_imported_mask_fields,
    vtk_named_boundary_fields,
    vtk_named_attribute_solution_fields,
    xdmf_abaqus_solution_fields,
    xdmf_imported_mask_workflow_fields,
    xdmf_named_boundary_fields,
    xdmf_named_attribute_solution_fields,
    xdmf_nodal_workflow_fields,
    DATASET_ELEM_TAGS,
    FIELD_BOUNDARY_MASK,
    FIELD_DRIVE_MASK,
    FIELD_FIXED_MASK,
    FIELD_FLUID_ID,
    FIELD_INLET_MASK,
    FIELD_KAPPA,
    FIELD_LINER_ID,
    FIELD_SOLUTION,
    FIELD_MERGED_BOUNDARY_MASK,
    FIELD_OUTLET_MASK,
    FIELD_SOURCE_STRENGTH,
    Hdf5WriteOptions,
};

#[cfg(feature = "hdf5")]
use fem_mesh::NamedAttributeRegistry;

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

#[cfg(feature = "hdf5")]
fn gmsh_named_attribute_solver_square() -> &'static str {
    "$MeshFormat\n\
    2.2 0 8\n\
    $EndMeshFormat\n\
    $PhysicalNames\n\
    4\n\
    2 1 \"fluid\"\n\
    2 2 \"liner\"\n\
    1 4 \"inlet\"\n\
    1 2 \"outlet\"\n\
    $EndPhysicalNames\n\
    $Nodes\n\
    9\n\
    1 0 0 0\n\
    2 0.5 0 0\n\
    3 1 0 0\n\
    4 0 0.5 0\n\
    5 0.5 0.5 0\n\
    6 1 0.5 0\n\
    7 0 1 0\n\
    8 0.5 1 0\n\
    9 1 1 0\n\
    $EndNodes\n\
    $Elements\n\
    16\n\
    1 1 2 1 1 1 2\n\
    2 1 2 1 1 2 3\n\
    3 1 2 2 2 3 6\n\
    4 1 2 2 2 6 9\n\
    5 1 2 3 3 9 8\n\
    6 1 2 3 3 8 7\n\
    7 1 2 4 4 7 4\n\
    8 1 2 4 4 4 1\n\
    9 2 2 1 1 1 2 5\n\
    10 2 2 1 1 1 5 4\n\
    11 2 2 2 2 2 3 6\n\
    12 2 2 2 2 2 6 5\n\
    13 2 2 1 1 4 5 8\n\
    14 2 2 1 1 4 8 7\n\
    15 2 2 2 2 5 6 9\n\
    16 2 2 2 2 5 9 8\n\
    $EndElements\n"
}

fn vtk_xml_with_fields<const D: usize>(
    mesh: &SimplexMesh<D>,
    point_data: Vec<DataArray>,
    cell_data: Vec<DataArray>,
) -> String {
    let mut writer = VtkWriter::new(mesh);
    for field in point_data {
        writer.add_point_data(field);
    }
    for field in cell_data {
        writer.add_cell_data(field);
    }

    let mut buf = Vec::<u8>::new();
    writer.write(&mut buf).expect("write vtu");
    String::from_utf8(buf).expect("utf8 vtk xml")
}

#[cfg(feature = "hdf5")]
fn point_mask_from_node_ids<const D: usize>(mesh: &SimplexMesh<D>, node_ids: &[u32]) -> Vec<f64> {
    let mut mask = vec![0.0; mesh.n_nodes()];
    for &node in node_ids {
        mask[node as usize] = 1.0;
    }
    mask
}

#[cfg(feature = "hdf5")]
fn unique_nodes_for_named_boundary_set(
    mesh: &SimplexMesh<2>,
    registry: &NamedAttributeRegistry,
    name: &str,
) -> Vec<u32> {
    let faces = mesh
        .face_ids_for_named_set(registry, name)
        .expect("missing named boundary set");
    let mut nodes = std::collections::BTreeSet::new();
    for face in faces {
        for &node in mesh.bface_nodes(face) {
            nodes.insert(node);
        }
    }
    nodes.into_iter().collect()
}

#[cfg(feature = "hdf5")]
fn nodal_mask_for_boundary_set(
    mesh: &SimplexMesh<2>,
    registry: &NamedAttributeRegistry,
    name: &str,
) -> Vec<f64> {
    point_mask_from_node_ids(mesh, &unique_nodes_for_named_boundary_set(mesh, registry, name))
}

#[cfg(feature = "hdf5")]
fn cell_mask_for_named_region(
    mesh: &SimplexMesh<2>,
    registry: &NamedAttributeRegistry,
    name: &str,
) -> Vec<f64> {
    let elems = mesh
        .element_ids_for_named_set(registry, name)
        .expect("missing named region set");
    let mut mask = vec![0.0; mesh.n_elems()];
    for elem in elems {
        mask[elem as usize] = 1.0;
    }
    mask
}

#[cfg(feature = "hdf5")]
fn merged_boundary_mask(mesh: &SimplexMesh<2>, registry: &NamedAttributeRegistry) -> Vec<f64> {
    nodal_mask_for_boundary_set(mesh, registry, "inlet")
        .into_iter()
        .zip(nodal_mask_for_boundary_set(mesh, registry, "outlet"))
        .map(|(inlet, outlet)| if inlet > 0.0 || outlet > 0.0 { 1.0 } else { 0.0 })
        .collect()
}

#[cfg(feature = "hdf5")]
fn cell_values_from_tags<const D: usize>(
    mesh: &SimplexMesh<D>,
    values_by_tag: &[(i32, f64)],
    default: f64,
) -> Vec<f64> {
    let by_tag: std::collections::BTreeMap<i32, f64> = values_by_tag.iter().copied().collect();
    mesh.elem_tags
        .iter()
        .map(|tag| by_tag.get(tag).copied().unwrap_or(default))
        .collect()
}

#[cfg(feature = "hdf5")]
fn cell_values_from_named_regions(
    mesh: &SimplexMesh<2>,
    registry: &NamedAttributeRegistry,
    values_by_name: &[(&str, f64)],
) -> Vec<f64> {
    let mut out = vec![0.0; mesh.n_elems()];
    for &(name, value) in values_by_name {
        let elems = mesh
            .element_ids_for_named_set(registry, name)
            .unwrap_or_else(|_| panic!("missing named region set: {name}"));
        for elem in elems {
            out[elem as usize] = value;
        }
    }
    out
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

    let (point_data, cell_data) = vtk_nodal_workflow_fields(FIELD_TEMPERATURE, temp, material_id);

    let xml = vtk_xml_with_fields(mesh, point_data, cell_data);

    assert!(xml.contains(&format!(r#"Name="{}""#, FIELD_TEMPERATURE)), "point result field should be exported");
    assert!(xml.contains(&format!(r#"Name="{}""#, FIELD_MATERIAL_ID)), "cell material field should be exported");
    assert!(xml.contains("NumberOfPoints=\"6\""), "linearized Tri6 working mesh should export 6 nodes");
    assert!(xml.contains("NumberOfCells=\"1\""), "single imported domain element should export one cell");
}

#[cfg(feature = "hdf5")]
#[test]
fn hdf5_xdmf_imported_mesh_preserves_tags_and_result_metadata_workflow() {
    let msh_src = include_str!("fixtures/unit_square.msh");
    let msh = read_msh(msh_src.as_bytes()).expect("parse unit_square.msh");
    let mesh = msh.into_2d().expect("extract 2D mesh");

    let temperature: Vec<f64> = (0..mesh.n_nodes())
        .map(|i| {
            let coords = mesh.node_coords(i as u32);
            coords[0] + coords[1]
        })
        .collect();

    let tmp = tempfile::tempdir().expect("temp dir");
    let h5_path = tmp.path().join("imported_mesh_workflow.h5");
    let xmf_path = tmp.path().join("imported_mesh_workflow.xmf");

    let fields = [(FIELD_TEMPERATURE, temperature.as_slice(), "H1")];
    write_mesh_and_fields(&h5_path, &mesh, &fields, &Hdf5WriteOptions::default())
        .expect("write mesh + fields to hdf5");

    let (mesh2, fields2) = read_mesh_and_fields::<2>(&h5_path).expect("read mesh + fields from hdf5");
    assert_eq!(mesh2.elem_tags, mesh.elem_tags, "material tags should round-trip through HDF5");
    assert_eq!(mesh2.face_tags, mesh.face_tags, "boundary tags should round-trip through HDF5");
    assert_eq!(fields2.len(), 1, "one nodal result field should round-trip");
    assert_eq!(fields2[0].0, FIELD_TEMPERATURE);
    assert_eq!(fields2[0].1, temperature);
    assert_eq!(fields2[0].2, "H1");

    let h5_path_text = h5_path.to_string_lossy().to_string();
    write_xdmf(
        &xmf_path,
        1,
        mesh.elem_type,
        2,
        &[mesh.n_nodes()],
        &[mesh.n_elems()],
        &h5_path_text,
        &xdmf_nodal_workflow_fields(FIELD_TEMPERATURE, &h5_path_text),
    )
    .expect("write xdmf sidecar");

    let xmf = std::fs::read_to_string(&xmf_path).expect("read xdmf sidecar");
    assert!(xmf.contains(&format!(r#"Name="{}""#, FIELD_TEMPERATURE)), "XDMF should expose nodal result field");
    assert!(xmf.contains(r#"Center="Node""#), "nodal field should be node-centered");
    assert!(xmf.contains("/fields/temperature/values"), "XDMF should point to the HDF5 temperature dataset");
    assert!(xmf.contains(&format!(r#"Name="{}""#, FIELD_MATERIAL_ID)), "XDMF should expose cell material field");
    assert!(xmf.contains(r#"Center="Cell""#), "material field should be cell-centered");
    assert!(xmf.contains(DATASET_ELEM_TAGS), "XDMF should point to the HDF5 material-tag dataset");
}

#[cfg(feature = "hdf5")]
#[test]
fn named_attribute_solution_export_workflow() {
    let msh = read_msh(gmsh_named_attribute_solver_square().as_bytes()).expect("parse named-attribute solver fixture");
    let registry = msh.named_attribute_registry();
    let mesh = msh.into_2d().expect("extract 2D named-attribute mesh");

    let solution: Vec<f64> = (0..mesh.n_nodes())
        .map(|i| mesh.node_coords(i as u32)[1])
        .collect();
    let inlet_mask = nodal_mask_for_boundary_set(&mesh, &registry, "inlet");
    let outlet_mask = nodal_mask_for_boundary_set(&mesh, &registry, "outlet");
    let merged_mask = merged_boundary_mask(&mesh, &registry);
    let material_id: Vec<f64> = mesh.elem_tags.iter().map(|&tag| tag as f64).collect();
    let kappa = cell_values_from_tags(&mesh, &[(1, 10.0), (2, 1.0)], 1.0);
    let source_strength = cell_values_from_named_regions(&mesh, &registry, &[("fluid", 2.0), ("liner", 0.0)]);
    let fluid_id = cell_mask_for_named_region(&mesh, &registry, "fluid");
    let liner_id = cell_mask_for_named_region(&mesh, &registry, "liner");

    let tmp = tempfile::tempdir().expect("temp dir");
    let h5_path = tmp.path().join("named_attribute_solution_workflow.h5");
    let xmf_path = tmp.path().join("named_attribute_solution_workflow.xmf");

    let fields = [
        (FIELD_SOLUTION, solution.as_slice(), "H1"),
        (FIELD_INLET_MASK, inlet_mask.as_slice(), "H1"),
        (FIELD_OUTLET_MASK, outlet_mask.as_slice(), "H1"),
        (FIELD_MERGED_BOUNDARY_MASK, merged_mask.as_slice(), "H1"),
        (FIELD_KAPPA, kappa.as_slice(), "L2"),
        (FIELD_SOURCE_STRENGTH, source_strength.as_slice(), "L2"),
        (FIELD_FLUID_ID, fluid_id.as_slice(), "L2"),
        (FIELD_LINER_ID, liner_id.as_slice(), "L2"),
    ];
    write_mesh_and_fields(&h5_path, &mesh, &fields, &Hdf5WriteOptions::default())
        .expect("write named-attribute mesh + fields to hdf5");

    let (mesh2, fields2) = read_mesh_and_fields::<2>(&h5_path).expect("read named-attribute mesh + fields from hdf5");
    assert_eq!(mesh2.elem_tags, mesh.elem_tags, "material tags should round-trip through HDF5");
    assert_eq!(mesh2.face_tags, mesh.face_tags, "boundary tags should round-trip through HDF5");
    assert_eq!(fields2.len(), 8);
    let fields_by_name: std::collections::HashMap<_, _> = fields2
        .into_iter()
        .map(|(name, values, space)| (name, (values, space)))
        .collect();
    assert_eq!(fields_by_name.get(FIELD_SOLUTION), Some(&(solution.clone(), "H1".to_string())));
    assert_eq!(fields_by_name.get(FIELD_INLET_MASK), Some(&(inlet_mask.clone(), "H1".to_string())));
    assert_eq!(fields_by_name.get(FIELD_OUTLET_MASK), Some(&(outlet_mask.clone(), "H1".to_string())));
    assert_eq!(fields_by_name.get(FIELD_MERGED_BOUNDARY_MASK), Some(&(merged_mask.clone(), "H1".to_string())));
    assert_eq!(fields_by_name.get(FIELD_KAPPA), Some(&(kappa.clone(), "L2".to_string())));
    assert_eq!(fields_by_name.get(FIELD_SOURCE_STRENGTH), Some(&(source_strength.clone(), "L2".to_string())));
    assert_eq!(fields_by_name.get(FIELD_FLUID_ID), Some(&(fluid_id.clone(), "L2".to_string())));
    assert_eq!(fields_by_name.get(FIELD_LINER_ID), Some(&(liner_id.clone(), "L2".to_string())));

    let (point_data, cell_data) = vtk_named_attribute_solution_fields(
        solution.clone(),
        inlet_mask.clone(),
        outlet_mask.clone(),
        Some(merged_mask.clone()),
        material_id,
        kappa.clone(),
        source_strength.clone(),
        fluid_id.clone(),
        Some(liner_id.clone()),
    );
    let vtk = vtk_xml_with_fields(&mesh, point_data, cell_data);
    assert!(vtk.contains(&format!(r#"Name="{}""#, FIELD_SOLUTION)), "VTK should expose the nodal solution field");
    assert!(vtk.contains(&format!(r#"Name="{}""#, FIELD_INLET_MASK)), "VTK should expose the inlet mask field");
    assert!(vtk.contains(&format!(r#"Name="{}""#, FIELD_OUTLET_MASK)), "VTK should expose the outlet mask field");
    assert!(vtk.contains(&format!(r#"Name="{}""#, FIELD_MERGED_BOUNDARY_MASK)), "VTK should expose the merged boundary field");
    assert!(vtk.contains(&format!(r#"Name="{}""#, FIELD_MATERIAL_ID)), "VTK should expose the material field");
    assert!(vtk.contains(&format!(r#"Name="{}""#, FIELD_KAPPA)), "VTK should expose the material coefficient field");
    assert!(vtk.contains(&format!(r#"Name="{}""#, FIELD_SOURCE_STRENGTH)), "VTK should expose the source field");
    assert!(vtk.contains(&format!(r#"Name="{}""#, FIELD_FLUID_ID)), "VTK should expose the fluid mask field");
    assert!(vtk.contains(&format!(r#"Name="{}""#, FIELD_LINER_ID)), "VTK should expose the liner mask field");

    let h5_path_text = h5_path.to_string_lossy().to_string();
    write_xdmf(
        &xmf_path,
        1,
        mesh.elem_type,
        2,
        &[mesh.n_nodes()],
        &[mesh.n_elems()],
        &h5_path_text,
        &xdmf_named_attribute_solution_fields(&h5_path_text, true, true),
    )
    .expect("write named-attribute xdmf sidecar");

    let xmf = std::fs::read_to_string(&xmf_path).expect("read named-attribute xdmf sidecar");
    assert!(xmf.contains(&format!(r#"Name="{}""#, FIELD_SOLUTION)), "XDMF should expose the nodal solution field");
    assert!(xmf.contains(&format!(r#"Name="{}""#, FIELD_INLET_MASK)), "XDMF should expose the inlet mask field");
    assert!(xmf.contains(&format!(r#"Name="{}""#, FIELD_OUTLET_MASK)), "XDMF should expose the outlet mask field");
    assert!(xmf.contains(&format!(r#"Name="{}""#, FIELD_MERGED_BOUNDARY_MASK)), "XDMF should expose the merged boundary field");
    assert!(xmf.contains(&format!(r#"Name="{}""#, FIELD_MATERIAL_ID)), "XDMF should expose the material field");
    assert!(xmf.contains(&format!(r#"Name="{}""#, FIELD_KAPPA)), "XDMF should expose the material coefficient field");
    assert!(xmf.contains(&format!(r#"Name="{}""#, FIELD_SOURCE_STRENGTH)), "XDMF should expose the source field");
    assert!(xmf.contains(&format!(r#"Name="{}""#, FIELD_FLUID_ID)), "XDMF should expose the fluid mask field");
    assert!(xmf.contains(&format!(r#"Name="{}""#, FIELD_LINER_ID)), "XDMF should expose the liner mask field");
    assert!(xmf.contains("/fields/u/values"), "XDMF should point to the solution dataset");
    assert!(xmf.contains("/fields/inlet_mask/values"), "XDMF should point to the inlet-mask dataset");
    assert!(xmf.contains("/fields/outlet_mask/values"), "XDMF should point to the outlet-mask dataset");
    assert!(xmf.contains("/fields/merged_boundary_mask/values"), "XDMF should point to the merged-boundary dataset");
    assert!(xmf.contains("/fields/kappa/values"), "XDMF should point to the kappa dataset");
    assert!(xmf.contains("/fields/source_strength/values"), "XDMF should point to the source dataset");
    assert!(xmf.contains("/fields/fluid_id/values"), "XDMF should point to the fluid-mask dataset");
    assert!(xmf.contains("/fields/liner_id/values"), "XDMF should point to the liner-mask dataset");
    assert!(xmf.contains(DATASET_ELEM_TAGS), "XDMF should point to the material tag dataset");
}

#[cfg(feature = "hdf5")]
#[test]
fn named_boundary_export_workflow() {
    let msh = read_msh(gmsh_named_attribute_solver_square().as_bytes()).expect("parse named-boundary fixture");
    let registry = msh.named_attribute_registry();
    let mesh = msh.into_2d().expect("extract 2D named-boundary mesh");

    let inlet_mask = nodal_mask_for_boundary_set(&mesh, &registry, "inlet");
    let outlet_mask = nodal_mask_for_boundary_set(&mesh, &registry, "outlet");
    let merged_mask = merged_boundary_mask(&mesh, &registry);
    let fluid_id = cell_mask_for_named_region(&mesh, &registry, "fluid");

    let tmp = tempfile::tempdir().expect("temp dir");
    let h5_path = tmp.path().join("named_boundary_workflow.h5");
    let xmf_path = tmp.path().join("named_boundary_workflow.xmf");

    let fields = [
        (FIELD_INLET_MASK, inlet_mask.as_slice(), "H1"),
        (FIELD_OUTLET_MASK, outlet_mask.as_slice(), "H1"),
        (FIELD_MERGED_BOUNDARY_MASK, merged_mask.as_slice(), "H1"),
        (FIELD_FLUID_ID, fluid_id.as_slice(), "L2"),
    ];
    write_mesh_and_fields(&h5_path, &mesh, &fields, &Hdf5WriteOptions::default())
        .expect("write named-boundary mesh + fields to hdf5");

    let (mesh2, fields2) = read_mesh_and_fields::<2>(&h5_path).expect("read named-boundary mesh + fields from hdf5");
    assert_eq!(mesh2.elem_tags, mesh.elem_tags, "material tags should round-trip through HDF5");
    assert_eq!(mesh2.face_tags, mesh.face_tags, "boundary tags should round-trip through HDF5");
    assert_eq!(fields2.len(), 4);
    let fields_by_name: std::collections::HashMap<_, _> = fields2
        .into_iter()
        .map(|(name, values, space)| (name, (values, space)))
        .collect();
    assert_eq!(fields_by_name.get(FIELD_INLET_MASK), Some(&(inlet_mask.clone(), "H1".to_string())));
    assert_eq!(fields_by_name.get(FIELD_OUTLET_MASK), Some(&(outlet_mask.clone(), "H1".to_string())));
    assert_eq!(fields_by_name.get(FIELD_MERGED_BOUNDARY_MASK), Some(&(merged_mask.clone(), "H1".to_string())));
    assert_eq!(fields_by_name.get(FIELD_FLUID_ID), Some(&(fluid_id.clone(), "L2".to_string())));

    let (point_data, cell_data) = vtk_named_boundary_fields(
        inlet_mask.clone(),
        outlet_mask.clone(),
        Some(merged_mask.clone()),
        fluid_id.clone(),
    );
    let vtk = vtk_xml_with_fields(&mesh, point_data, cell_data);
    assert!(vtk.contains(&format!(r#"Name="{}""#, FIELD_INLET_MASK)), "VTK should expose the inlet mask field");
    assert!(vtk.contains(&format!(r#"Name="{}""#, FIELD_OUTLET_MASK)), "VTK should expose the outlet mask field");
    assert!(vtk.contains(&format!(r#"Name="{}""#, FIELD_MERGED_BOUNDARY_MASK)), "VTK should expose the merged boundary field");
    assert!(vtk.contains(&format!(r#"Name="{}""#, FIELD_FLUID_ID)), "VTK should expose the fluid mask field");

    let h5_path_text = h5_path.to_string_lossy().to_string();
    write_xdmf(
        &xmf_path,
        1,
        mesh.elem_type,
        2,
        &[mesh.n_nodes()],
        &[mesh.n_elems()],
        &h5_path_text,
        &xdmf_named_boundary_fields(&h5_path_text, true),
    )
    .expect("write named-boundary xdmf sidecar");

    let xmf = std::fs::read_to_string(&xmf_path).expect("read named-boundary xdmf sidecar");
    assert!(xmf.contains(&format!(r#"Name="{}""#, FIELD_INLET_MASK)), "XDMF should expose the inlet mask field");
    assert!(xmf.contains(&format!(r#"Name="{}""#, FIELD_OUTLET_MASK)), "XDMF should expose the outlet mask field");
    assert!(xmf.contains(&format!(r#"Name="{}""#, FIELD_MERGED_BOUNDARY_MASK)), "XDMF should expose the merged boundary field");
    assert!(xmf.contains(&format!(r#"Name="{}""#, FIELD_FLUID_ID)), "XDMF should expose the fluid mask field");
    assert!(xmf.contains("/fields/inlet_mask/values"), "XDMF should point to the inlet-mask dataset");
    assert!(xmf.contains("/fields/outlet_mask/values"), "XDMF should point to the outlet-mask dataset");
    assert!(xmf.contains("/fields/merged_boundary_mask/values"), "XDMF should point to the merged-boundary dataset");
    assert!(xmf.contains("/fields/fluid_id/values"), "XDMF should point to the fluid-mask dataset");
}

#[cfg(feature = "hdf5")]
#[test]
fn abaqus_named_sets_and_result_export_workflow() {
    let inp = r#"*Heading
** Abaqus user-closure fixture with one material set and boundary node sets
*Node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
4, 0.0, 0.0, 1.0
*Element, type=C3D4, elset=MAT_A
1, 1, 2, 3, 4
*Nset, nset=FIXED
1, 2, 3
*Nset, nset=DRIVE
4
"#;

    let data = read_abaqus_inp_full(inp.as_bytes()).expect("parse abaqus full input");
    let fixed = data.node_sets.get("FIXED").expect("missing FIXED node set");
    let drive = data.node_sets.get("DRIVE").expect("missing DRIVE node set");
    let mat_a = data.elem_sets.get("MAT_A").expect("missing MAT_A element set");
    assert_eq!(fixed, &vec![0, 1, 2], "FIXED node set should preserve 0-based node indices");
    assert_eq!(drive, &vec![3], "DRIVE node set should preserve 0-based node indices");
    assert_eq!(mat_a, &vec![0], "MAT_A should contain the single tetrahedron");

    let mesh = &data.mesh;
    assert_eq!(mesh.elem_tags, vec![1], "Abaqus elset should map to a non-zero material tag");

    let mut fixed_mask = vec![0.0; mesh.n_nodes()];
    for &node in fixed {
        fixed_mask[node as usize] = 1.0;
    }
    let mut drive_mask = vec![0.0; mesh.n_nodes()];
    for &node in drive {
        drive_mask[node as usize] = 1.0;
    }
    let solution = drive_mask.clone();

    let tmp = tempfile::tempdir().expect("temp dir");
    let h5_path = tmp.path().join("abaqus_user_workflow.h5");
    let xmf_path = tmp.path().join("abaqus_user_workflow.xmf");

    let fields = [
        (FIELD_SOLUTION, solution.as_slice(), "H1"),
        (FIELD_FIXED_MASK, fixed_mask.as_slice(), "H1"),
        (FIELD_DRIVE_MASK, drive_mask.as_slice(), "H1"),
    ];
    write_mesh_and_fields(&h5_path, mesh, &fields, &Hdf5WriteOptions::default())
        .expect("write abaqus mesh + fields to hdf5");

    let (mesh2, fields2) = read_mesh_and_fields::<3>(&h5_path).expect("read abaqus mesh + fields from hdf5");
    assert_eq!(mesh2.elem_tags, mesh.elem_tags, "material tags should round-trip through HDF5");
    assert_eq!(mesh2.face_tags, mesh.face_tags, "reconstructed boundary tags should round-trip through HDF5");
    assert_eq!(fields2.len(), 3);
    let fields_by_name: std::collections::HashMap<_, _> = fields2
        .into_iter()
        .map(|(name, values, space)| (name, (values, space)))
        .collect();
    assert_eq!(fields_by_name.get(FIELD_SOLUTION), Some(&(solution.clone(), "H1".to_string())));
    assert_eq!(fields_by_name.get(FIELD_FIXED_MASK), Some(&(fixed_mask.clone(), "H1".to_string())));
    assert_eq!(fields_by_name.get(FIELD_DRIVE_MASK), Some(&(drive_mask.clone(), "H1".to_string())));

    let (point_data, cell_data) = vtk_abaqus_solution_fields(
        drive_mask.clone(),
        fixed_mask.clone(),
        drive_mask.clone(),
        mesh.elem_tags.iter().map(|&tag| tag as f64).collect(),
    );
    let vtk = vtk_xml_with_fields(mesh, point_data, cell_data);
    assert!(vtk.contains(&format!(r#"Name="{}""#, FIELD_SOLUTION)), "VTK should expose the nodal solution field");
    assert!(vtk.contains(&format!(r#"Name="{}""#, FIELD_FIXED_MASK)), "VTK should expose the boundary-config node mask");
    assert!(vtk.contains(&format!(r#"Name="{}""#, FIELD_DRIVE_MASK)), "VTK should expose the drive node mask");
    assert!(vtk.contains(&format!(r#"Name="{}""#, FIELD_MATERIAL_ID)), "VTK should expose the material cell field");

    let h5_path_text = h5_path.to_string_lossy().to_string();
    write_xdmf(
        &xmf_path,
        1,
        mesh.elem_type,
        3,
        &[mesh.n_nodes()],
        &[mesh.n_elems()],
        &h5_path_text,
        &xdmf_abaqus_solution_fields(&h5_path_text),
    )
    .expect("write abaqus xdmf sidecar");

    let xmf = std::fs::read_to_string(&xmf_path).expect("read abaqus xdmf sidecar");
    assert!(xmf.contains(&format!(r#"Name="{}""#, FIELD_SOLUTION)), "XDMF should expose the nodal solution field");
    assert!(xmf.contains(&format!(r#"Name="{}""#, FIELD_FIXED_MASK)), "XDMF should expose the boundary-config node mask");
    assert!(xmf.contains(&format!(r#"Name="{}""#, FIELD_DRIVE_MASK)), "XDMF should expose the drive node mask");
    assert!(xmf.contains("/fields/u/values"), "XDMF should point to the solution dataset");
    assert!(xmf.contains("/fields/fixed_mask/values"), "XDMF should point to the node-mask dataset");
    assert!(xmf.contains("/fields/drive_mask/values"), "XDMF should point to the drive-mask dataset");
    assert!(xmf.contains(&format!(r#"Name="{}""#, FIELD_MATERIAL_ID)), "XDMF should expose the material cell field");
    assert!(xmf.contains(DATASET_ELEM_TAGS), "XDMF should point to the element-tag dataset");
}

#[cfg(feature = "hdf5")]
#[test]
fn netgen_surfaceelements_boundary_mask_and_result_export_workflow() {
    let vol_src = r#"
dimension
3

points
4
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

volumeelements
1
1 4 1 2 3 4

surfaceelements
4
3 3 1 2 3
5 3 1 2 4
3 3 1 3 4
3 3 2 3 4
"#;

    let mesh = read_netgen_vol(vol_src.as_bytes()).expect("parse netgen surfaceelements workflow mesh");
    assert_eq!(mesh.elem_tags, vec![1], "volume material tag should be preserved");
    assert_eq!(mesh.face_tags, vec![3, 5, 3, 3], "surfaceelements tags should preserve explicit boundary IDs");

    let mut boundary_mask = vec![0.0; mesh.n_nodes()];
    for (face_index, &tag) in mesh.face_tags.iter().enumerate() {
        if tag == 5 {
            for &node in &mesh.face_conn[3 * face_index..3 * (face_index + 1)] {
                boundary_mask[node as usize] = 1.0;
            }
        }
    }
    assert_eq!(boundary_mask, vec![1.0, 1.0, 0.0, 1.0], "tag-5 boundary should activate exactly its face nodes");

    let tmp = tempfile::tempdir().expect("temp dir");
    let h5_path = tmp.path().join("netgen_surfaceelements_workflow.h5");
    let xmf_path = tmp.path().join("netgen_surfaceelements_workflow.xmf");

    let fields = [(FIELD_BOUNDARY_MASK, boundary_mask.as_slice(), "H1")];
    write_mesh_and_fields(&h5_path, &mesh, &fields, &Hdf5WriteOptions::default())
        .expect("write netgen workflow mesh + fields to hdf5");

    let (mesh2, fields2) = read_mesh_and_fields::<3>(&h5_path).expect("read netgen workflow mesh + fields from hdf5");
    assert_eq!(mesh2.elem_tags, mesh.elem_tags, "material tags should round-trip through HDF5");
    assert_eq!(mesh2.face_tags, mesh.face_tags, "surface boundary tags should round-trip through HDF5");
    assert_eq!(fields2.len(), 1);
    assert_eq!(fields2[0].0, FIELD_BOUNDARY_MASK);
    assert_eq!(fields2[0].1, boundary_mask);
    assert_eq!(fields2[0].2, "H1");

    let (point_data, cell_data) = vtk_imported_mask_fields(
        FIELD_BOUNDARY_MASK,
        boundary_mask.clone(),
        mesh.elem_tags.iter().map(|&tag| tag as f64).collect(),
    );
    let vtk = vtk_xml_with_fields(&mesh, point_data, cell_data);
    assert!(vtk.contains(&format!(r#"Name="{}""#, FIELD_BOUNDARY_MASK)), "VTK should expose the boundary mask field");
    assert!(vtk.contains(&format!(r#"Name="{}""#, FIELD_MATERIAL_ID)), "VTK should expose the material cell field");

    let h5_path_text = h5_path.to_string_lossy().to_string();
    write_xdmf(
        &xmf_path,
        1,
        mesh.elem_type,
        3,
        &[mesh.n_nodes()],
        &[mesh.n_elems()],
        &h5_path_text,
        &xdmf_imported_mask_workflow_fields(FIELD_BOUNDARY_MASK, &h5_path_text),
    )
    .expect("write netgen workflow xdmf sidecar");

    let xmf = std::fs::read_to_string(&xmf_path).expect("read netgen workflow xdmf sidecar");
    assert!(xmf.contains(&format!(r#"Name="{}""#, FIELD_BOUNDARY_MASK)), "XDMF should expose the boundary mask field");
    assert!(xmf.contains("/fields/boundary_mask/values"), "XDMF should point to the boundary mask dataset");
    assert!(xmf.contains(&format!(r#"Name="{}""#, FIELD_MATERIAL_ID)), "XDMF should expose the material cell field");
    assert!(xmf.contains("/mesh/elem_tags"), "XDMF should point to the material tag dataset");
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
