//! Minimal VTK UnstructuredGrid (`.vtu`) XML reader.
//!
//! Reads named point-data arrays from ASCII-encoded `.vtu` files produced by
//! [`VtkWriter`](crate::vtk::VtkWriter).
//!
//! This is intentionally simple: it handles the ASCII format we write, not the
//! full VTK specification.

use std::collections::HashMap;
use std::io::Read;

use fem_core::{FemError, FemResult};

/// Read all point-data arrays from a `.vtu` file.
///
/// Returns a map from array name to `(n_components, values)`.
///
/// # Errors
/// Returns `FemError::Io` on file read errors or `FemError::Mesh` on parse errors.
pub fn read_vtu_point_data(
    path: impl AsRef<std::path::Path>,
) -> FemResult<HashMap<String, (usize, Vec<f64>)>> {
    let mut content = String::new();
    std::fs::File::open(path)?.read_to_string(&mut content)?;
    parse_point_data(&content)
}

/// Read point-data arrays from a `.vtu` XML string.
pub fn read_vtu_point_data_str(xml: &str) -> FemResult<HashMap<String, (usize, Vec<f64>)>> {
    parse_point_data(xml)
}

fn parse_point_data(xml: &str) -> FemResult<HashMap<String, (usize, Vec<f64>)>> {
    let mut result = HashMap::new();

    // Find the <PointData> ... </PointData> section.
    let pd_start = match xml.find("<PointData>") {
        Some(pos) => pos,
        None => return Ok(result), // no point data
    };
    let pd_end = xml[pd_start..].find("</PointData>")
        .ok_or_else(|| FemError::Mesh("malformed VTU: unclosed <PointData>".into()))?
        + pd_start;
    let pd_section = &xml[pd_start..pd_end];

    // Extract each <DataArray ...> ... </DataArray> within PointData.
    let mut cursor = 0;
    while let Some(da_start) = pd_section[cursor..].find("<DataArray") {
        let abs_start = cursor + da_start;
        let tag_end = pd_section[abs_start..].find('>')
            .ok_or_else(|| FemError::Mesh("malformed DataArray tag".into()))?
            + abs_start;

        let tag = &pd_section[abs_start..=tag_end];

        // Parse Name attribute
        let name = extract_attr(tag, "Name")
            .ok_or_else(|| FemError::Mesh("DataArray missing Name attribute".into()))?;

        // Parse NumberOfComponents (default 1)
        let n_comp: usize = extract_attr(tag, "NumberOfComponents")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        // Parse the float data between > and </DataArray>
        let da_close = pd_section[tag_end..].find("</DataArray>")
            .ok_or_else(|| FemError::Mesh("unclosed DataArray".into()))?
            + tag_end;
        let data_text = &pd_section[tag_end + 1..da_close];

        let values: Vec<f64> = data_text
            .split_whitespace()
            .filter_map(|s| s.parse::<f64>().ok())
            .collect();

        result.insert(name, (n_comp, values));
        cursor = da_close + "</DataArray>".len();
    }

    Ok(result)
}

/// Extract an XML attribute value: `Name="foo"` → `"foo"`.
fn extract_attr(tag: &str, attr_name: &str) -> Option<String> {
    let pattern = format!("{attr_name}=\"");
    let start = tag.find(&pattern)? + pattern.len();
    let end = tag[start..].find('"')? + start;
    Some(tag[start..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vtk::{DataArray, VtkWriter};
    use fem_mesh::SimplexMesh;

    #[test]
    fn roundtrip_scalar_point_data() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();

        // Write
        let mut w = VtkWriter::new(&mesh);
        w.add_point_data(DataArray::scalars("temperature", u.clone()));
        let mut buf = Vec::<u8>::new();
        w.write(&mut buf).unwrap();
        let xml = String::from_utf8(buf).unwrap();

        // Read back
        let data = parse_point_data(&xml).unwrap();
        assert!(data.contains_key("temperature"));
        let (nc, vals) = &data["temperature"];
        assert_eq!(*nc, 1);
        assert_eq!(vals.len(), n);
        for (i, (&orig, &loaded)) in u.iter().zip(vals.iter()).enumerate() {
            assert!(
                (orig - loaded).abs() < 1e-8,
                "mismatch at node {i}: wrote {orig}, read {loaded}"
            );
        }
    }

    #[test]
    fn roundtrip_vector_point_data() {
        let mesh = SimplexMesh::<2>::unit_square_tri(3);
        let n = mesh.n_nodes();
        let uv: Vec<f64> = (0..n * 2).map(|i| i as f64 * 0.01).collect();

        let mut w = VtkWriter::new(&mesh);
        w.add_point_data(DataArray::vectors("displacement", 2, uv.clone()));
        let mut buf = Vec::<u8>::new();
        w.write(&mut buf).unwrap();
        let xml = String::from_utf8(buf).unwrap();

        let data = parse_point_data(&xml).unwrap();
        let (nc, vals) = &data["displacement"];
        assert_eq!(*nc, 2);
        assert_eq!(vals.len(), n * 2);
    }

    #[test]
    fn empty_point_data() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let w = VtkWriter::new(&mesh);
        let mut buf = Vec::<u8>::new();
        w.write(&mut buf).unwrap();
        let xml = String::from_utf8(buf).unwrap();

        let data = parse_point_data(&xml).unwrap();
        assert!(data.is_empty());
    }
}
