use crate::vtk::DataArray;
use crate::xdmf::{XdmfCenter, XdmfField};

pub const FIELD_MATERIAL_ID: &str = "material_id";
pub const FIELD_FIXED_MASK: &str = "fixed_mask";
pub const FIELD_DRIVE_MASK: &str = "drive_mask";
pub const FIELD_BOUNDARY_MASK: &str = "boundary_mask";
pub const FIELD_INLET_MASK: &str = "inlet_mask";
pub const FIELD_OUTLET_MASK: &str = "outlet_mask";
pub const FIELD_MERGED_BOUNDARY_MASK: &str = "merged_boundary_mask";
pub const FIELD_FLUID_ID: &str = "fluid_id";
pub const FIELD_LINER_ID: &str = "liner_id";
pub const FIELD_KAPPA: &str = "kappa";
pub const FIELD_SOURCE_STRENGTH: &str = "source_strength";
pub const FIELD_SOLUTION: &str = "u";
pub const FIELD_TEMPERATURE: &str = "temperature";
pub const DATASET_ELEM_TAGS: &str = "/mesh/elem_tags";

pub fn hdf5_field_values_path(field_name: &str) -> String {
    format!("/fields/{field_name}/values")
}

pub fn vtk_scalar_field(name: &'static str, values: Vec<f64>) -> DataArray {
    DataArray::scalars(name, values)
}

pub fn vtk_nodal_workflow_fields(
    field_name: &'static str,
    values: Vec<f64>,
    material_id: Vec<f64>,
) -> (Vec<DataArray>, Vec<DataArray>) {
    (
        vec![vtk_scalar_field(field_name, values)],
        vec![vtk_scalar_field(FIELD_MATERIAL_ID, material_id)],
    )
}

pub fn vtk_imported_mask_fields(
    primary_mask_name: &'static str,
    primary_mask: Vec<f64>,
    material_id: Vec<f64>,
) -> (Vec<DataArray>, Vec<DataArray>) {
    (
        vec![vtk_scalar_field(primary_mask_name, primary_mask)],
        vec![vtk_scalar_field(FIELD_MATERIAL_ID, material_id)],
    )
}

pub fn vtk_named_boundary_point_fields(
    inlet_mask: Vec<f64>,
    outlet_mask: Vec<f64>,
    merged_boundary_mask: Option<Vec<f64>>,
) -> Vec<DataArray> {
    let mut point_data = vec![
        vtk_scalar_field(FIELD_INLET_MASK, inlet_mask),
        vtk_scalar_field(FIELD_OUTLET_MASK, outlet_mask),
    ];
    if let Some(mask) = merged_boundary_mask {
        point_data.push(vtk_scalar_field(FIELD_MERGED_BOUNDARY_MASK, mask));
    }
    point_data
}

pub fn vtk_named_boundary_cell_fields(fluid_id: Vec<f64>) -> Vec<DataArray> {
    vec![vtk_scalar_field(FIELD_FLUID_ID, fluid_id)]
}

pub fn vtk_named_boundary_fields(
    inlet_mask: Vec<f64>,
    outlet_mask: Vec<f64>,
    merged_boundary_mask: Option<Vec<f64>>,
    fluid_id: Vec<f64>,
) -> (Vec<DataArray>, Vec<DataArray>) {
    (
        vtk_named_boundary_point_fields(inlet_mask, outlet_mask, merged_boundary_mask),
        vtk_named_boundary_cell_fields(fluid_id),
    )
}

pub fn vtk_named_attribute_cell_fields(
    material_id: Vec<f64>,
    kappa: Vec<f64>,
    source_strength: Vec<f64>,
    fluid_id: Vec<f64>,
    liner_id: Option<Vec<f64>>,
) -> Vec<DataArray> {
    let mut cell_data = vec![
        vtk_scalar_field(FIELD_MATERIAL_ID, material_id),
        vtk_scalar_field(FIELD_KAPPA, kappa),
        vtk_scalar_field(FIELD_SOURCE_STRENGTH, source_strength),
    ];
    cell_data.extend(vtk_named_boundary_cell_fields(fluid_id));
    if let Some(mask) = liner_id {
        cell_data.push(vtk_scalar_field(FIELD_LINER_ID, mask));
    }
    cell_data
}

pub fn vtk_named_attribute_solution_fields(
    solution: Vec<f64>,
    inlet_mask: Vec<f64>,
    outlet_mask: Vec<f64>,
    merged_boundary_mask: Option<Vec<f64>>,
    material_id: Vec<f64>,
    kappa: Vec<f64>,
    source_strength: Vec<f64>,
    fluid_id: Vec<f64>,
    liner_id: Option<Vec<f64>>,
) -> (Vec<DataArray>, Vec<DataArray>) {
    let mut point_data =
        vtk_named_boundary_point_fields(inlet_mask, outlet_mask, merged_boundary_mask);
    point_data.insert(0, vtk_scalar_field(FIELD_SOLUTION, solution));
    (
        point_data,
        vtk_named_attribute_cell_fields(material_id, kappa, source_strength, fluid_id, liner_id),
    )
}

pub fn vtk_abaqus_solution_point_fields(
    solution: Vec<f64>,
    fixed_mask: Vec<f64>,
    drive_mask: Vec<f64>,
) -> Vec<DataArray> {
    vec![
        vtk_scalar_field(FIELD_SOLUTION, solution),
        vtk_scalar_field(FIELD_FIXED_MASK, fixed_mask),
        vtk_scalar_field(FIELD_DRIVE_MASK, drive_mask),
    ]
}

pub fn vtk_abaqus_solution_cell_fields(material_id: Vec<f64>) -> Vec<DataArray> {
    vec![vtk_scalar_field(FIELD_MATERIAL_ID, material_id)]
}

pub fn vtk_abaqus_solution_fields(
    solution: Vec<f64>,
    fixed_mask: Vec<f64>,
    drive_mask: Vec<f64>,
    material_id: Vec<f64>,
) -> (Vec<DataArray>, Vec<DataArray>) {
    (
        vtk_abaqus_solution_point_fields(solution, fixed_mask, drive_mask),
        vtk_abaqus_solution_cell_fields(material_id),
    )
}

pub fn xdmf_nodal_scalar_field(name: &str, hdf5_path: &str) -> XdmfField {
    XdmfField {
        name: name.into(),
        hdf5_path: hdf5_path.into(),
        dataset_path: hdf5_field_values_path(name),
        center: XdmfCenter::Node,
    }
}

pub fn xdmf_cell_scalar_field(name: &str, hdf5_path: &str, dataset_path: &str) -> XdmfField {
    XdmfField {
        name: name.into(),
        hdf5_path: hdf5_path.into(),
        dataset_path: dataset_path.into(),
        center: XdmfCenter::Cell,
    }
}

pub fn xdmf_material_id_field(hdf5_path: &str) -> XdmfField {
    xdmf_cell_scalar_field(FIELD_MATERIAL_ID, hdf5_path, DATASET_ELEM_TAGS)
}

pub fn xdmf_nodal_workflow_fields(field_name: &str, hdf5_path: &str) -> [XdmfField; 2] {
    [
        xdmf_nodal_scalar_field(field_name, hdf5_path),
        xdmf_material_id_field(hdf5_path),
    ]
}

pub fn xdmf_imported_mask_workflow_fields(field_name: &str, hdf5_path: &str) -> [XdmfField; 2] {
    xdmf_nodal_workflow_fields(field_name, hdf5_path)
}

pub fn xdmf_named_boundary_point_fields(
    hdf5_path: &str,
    include_merged_boundary: bool,
) -> Vec<XdmfField> {
    let mut fields = vec![
        xdmf_nodal_scalar_field(FIELD_INLET_MASK, hdf5_path),
        xdmf_nodal_scalar_field(FIELD_OUTLET_MASK, hdf5_path),
    ];
    if include_merged_boundary {
        fields.push(xdmf_nodal_scalar_field(FIELD_MERGED_BOUNDARY_MASK, hdf5_path));
    }
    fields
}

pub fn xdmf_named_boundary_cell_fields(hdf5_path: &str) -> Vec<XdmfField> {
    vec![xdmf_cell_scalar_field(
        FIELD_FLUID_ID,
        hdf5_path,
        &hdf5_field_values_path(FIELD_FLUID_ID),
    )]
}

pub fn xdmf_named_boundary_fields(
    hdf5_path: &str,
    include_merged_boundary: bool,
) -> Vec<XdmfField> {
    let mut fields = xdmf_named_boundary_point_fields(hdf5_path, include_merged_boundary);
    fields.extend(xdmf_named_boundary_cell_fields(hdf5_path));
    fields
}

pub fn xdmf_abaqus_solution_point_fields(hdf5_path: &str) -> Vec<XdmfField> {
    vec![
        xdmf_nodal_scalar_field(FIELD_SOLUTION, hdf5_path),
        xdmf_nodal_scalar_field(FIELD_FIXED_MASK, hdf5_path),
        xdmf_nodal_scalar_field(FIELD_DRIVE_MASK, hdf5_path),
    ]
}

pub fn xdmf_abaqus_solution_cell_fields(hdf5_path: &str) -> Vec<XdmfField> {
    vec![xdmf_material_id_field(hdf5_path)]
}

pub fn xdmf_abaqus_solution_fields(hdf5_path: &str) -> Vec<XdmfField> {
    let mut fields = xdmf_abaqus_solution_point_fields(hdf5_path);
    fields.extend(xdmf_abaqus_solution_cell_fields(hdf5_path));
    fields
}

pub fn xdmf_named_attribute_solution_fields(
    hdf5_path: &str,
    include_merged_boundary: bool,
    include_liner: bool,
) -> Vec<XdmfField> {
    let mut fields = vec![xdmf_nodal_scalar_field(FIELD_SOLUTION, hdf5_path)];
    fields.extend(xdmf_named_boundary_point_fields(hdf5_path, include_merged_boundary));
    fields.extend(xdmf_named_attribute_cell_fields(hdf5_path, include_liner));
    fields
}

pub fn xdmf_named_attribute_cell_fields(
    hdf5_path: &str,
    include_liner: bool,
) -> Vec<XdmfField> {
    let mut fields = vec![
        xdmf_material_id_field(hdf5_path),
        xdmf_cell_scalar_field(FIELD_KAPPA, hdf5_path, &hdf5_field_values_path(FIELD_KAPPA)),
        xdmf_cell_scalar_field(
            FIELD_SOURCE_STRENGTH,
            hdf5_path,
            &hdf5_field_values_path(FIELD_SOURCE_STRENGTH),
        ),
    ];
    fields.extend(xdmf_named_boundary_cell_fields(hdf5_path));
    if include_liner {
        fields.push(xdmf_cell_scalar_field(FIELD_LINER_ID, hdf5_path, &hdf5_field_values_path(FIELD_LINER_ID)));
    }
    fields
}
