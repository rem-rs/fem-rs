//! VTK UnstructuredGrid (`.vtu`) XML reader — mesh + field data.
//!
//! Reads ASCII-encoded `.vtu` files produced by [`VtkWriter`](crate::vtk::VtkWriter)
//! and similar tools.  Returns both the mesh topology and point/cell data arrays.

use std::collections::HashMap;
use std::io::Read;

use fem_core::{FemError, FemResult};
use fem_mesh::{element_type::ElementType, simplex::Mesh};

/// Read result: mesh + named point/cell data arrays.
pub struct VtuData {
    pub mesh: Mesh<3>,
    pub point_data: HashMap<String, (usize, Vec<f64>)>,
    pub cell_data: HashMap<String, (usize, Vec<f64>)>,
}

/// Read a `.vtu` file returning mesh and data arrays.
pub fn read_vtu(path: impl AsRef<std::path::Path>) -> FemResult<VtuData> {
    let mut content = String::new();
    std::fs::File::open(path)?.read_to_string(&mut content)?;
    parse_vtu(&content)
}

/// Read a `.vtu` file, returning only mesh connectivity (no field data).
pub fn read_vtu_mesh(path: impl AsRef<std::path::Path>) -> FemResult<Mesh<3>> {
    let mut content = String::new();
    std::fs::File::open(path)?.read_to_string(&mut content)?;
    parse_vtu(&content).map(|d| d.mesh)
}

/// Read mesh + data from a VTU XML string.
pub fn read_vtu_str(xml: &str) -> FemResult<VtuData> {
    parse_vtu(xml)
}

fn parse_vtu(xml: &str) -> FemResult<VtuData> {
    let piece = extract_section(xml, "<Piece", "</Piece>")
        .ok_or_else(|| FemError::Mesh("VTU: missing <Piece>".into()))?;

    // Points
    let points_xml = extract_section(piece, "<Points>", "</Points>")
        .ok_or_else(|| FemError::Mesh("VTU: missing <Points>".into()))?;
    let coords = parse_data_array_f64(points_xml, None)?;
    let _n_nodes = coords.len() / 3;

    // Cells
    let cells_xml = extract_section(piece, "<Cells>", "</Cells>")
        .ok_or_else(|| FemError::Mesh("VTU: missing <Cells>".into()))?;
    let conn = parse_data_array_usize(cells_xml, Some("connectivity"))?;
    let offsets = parse_data_array_usize(cells_xml, Some("offsets"))?;
    let types = parse_data_array_u8(cells_xml, Some("types"))?;
    let n_elems = types.len();

    // Decompose connectivity & element types
    let mut elem_conn: Vec<Vec<u32>> = Vec::with_capacity(n_elems);
    let mut elem_types: Vec<ElementType> = Vec::with_capacity(n_elems);
    let elem_tags: Vec<i32> = vec![1; n_elems];
    let mut ci = 0usize;
    for e in 0..n_elems {
        let off = offsets[e];
        let npe = off - ci;
        let vtk_type = types[e];
        let et = vtk_to_elem(vtk_type, npe)
            .ok_or_else(|| FemError::Mesh(format!("VTU: unsupported cell type {vtk_type} with {npe} nodes")))?;
        let nodes: Vec<u32> = conn[ci..off].iter().map(|&n| n as u32).collect();
        elem_conn.push(nodes);
        elem_types.push(et);
        ci = off;
    }

    // Face info: no embedded boundary in VTU; build empty
    let n_face_type = if n_elems > 0 {
        elem_types[0].boundary_type().unwrap_or(ElementType::Tri3)
    } else {
        ElementType::Tri3
    };

    let uniform_type = if elem_types.iter().all(|&t| t == elem_types[0]) {
        Some(elem_types[0])
    } else {
        None
    };

    let flat_elem: Vec<u32> = elem_conn.into_iter().flatten().collect();

    let mesh = Mesh {
        coords,
        conn: flat_elem,
        elem_tags,
        elem_type: uniform_type.unwrap_or(ElementType::Tet4),
        face_conn: Vec::new(),
        face_tags: Vec::new(),
        face_type: n_face_type,
        elem_types: if uniform_type.is_some() { None } else { Some(elem_types) },
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![], edge_to_elem: vec![],
        geometry: None, nc_vertex_view: None,
    };

    let point_data = parse_point_data_array(piece, "<PointData>", "</PointData>")?;
    let cell_data = parse_point_data_array(piece, "<CellData>", "</CellData>")?;

    Ok(VtuData { mesh, point_data, cell_data })
}

fn vtk_to_elem(vtk_type: u8, npe: usize) -> Option<ElementType> {
    Some(match (vtk_type, npe) {
        ( 3, _) => ElementType::Line2,
        ( 5, 3) => ElementType::Tri3,
        ( 5, 6) => ElementType::Tri6,
        ( 9, 4) => ElementType::Quad4,
        ( 9, 8) => ElementType::Quad8,
        (10, 4) => ElementType::Tet4,
        (10,10) => ElementType::Tet10,
        (12, 8) => ElementType::Hex8,
        (12,20) => ElementType::Hex20,
        (13, 6) => ElementType::Prism6,
        (13,15) => ElementType::Prism15,
        (14, 5) => ElementType::Pyramid5,
        (14,13) => ElementType::Pyramid13,
        _ => return None,
    })
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn extract_section<'a>(xml: &'a str, open_tag: &str, close_tag: &str) -> Option<&'a str> {
    let start = xml.find(open_tag)?;
    let content_start = xml[start..].find('>')? + start + 1;
    let end = xml[content_start..].find(close_tag)? + content_start;
    Some(&xml[content_start..end])
}

fn extract_attr(tag: &str, name: &str) -> Option<String> {
    let p = format!("{name}=\"");
    let s = tag.find(&p)? + p.len();
    let e = tag[s..].find('"')? + s;
    Some(tag[s..e].to_string())
}

fn data_array_section<'a>(xml: &'a str, type_filter: Option<&str>, name_filter: Option<&str>) -> Option<&'a str> {
    let mut cursor = 0;
    loop {
        let s = xml[cursor..].find("<DataArray")? + cursor;
        let tag_end = xml[s..].find('>')? + s;
        let tag = &xml[s..=tag_end];

        if let Some(tf) = type_filter {
            if extract_attr(tag, "type").as_deref() != Some(tf) { cursor = tag_end + 1; continue; }
        }
        if let Some(nf) = name_filter {
            if extract_attr(tag, "Name").as_deref() != Some(nf) { cursor = tag_end + 1; continue; }
        }
        let close = xml[tag_end..].find("</DataArray>")? + tag_end;
        return Some(&xml[tag_end + 1..close]);
    }
}

fn parse_data_array_f64(xml: &str, name_filter: Option<&str>) -> FemResult<Vec<f64>> {
    let section = data_array_section(xml, Some("Float64"), name_filter)
        .ok_or_else(|| FemError::Mesh("VTU: missing Float64 DataArray".into()))?;
    section.split_whitespace()
        .map(|s: &str| s.parse().map_err(|_| FemError::Mesh(format!("VTU: bad float: {s}"))))
        .collect()
}

fn parse_data_array_usize(xml: &str, name_filter: Option<&str>) -> FemResult<Vec<usize>> {
    for type_name in &["Int64", "Int32", "UInt32"] {
        if let Some(section) = data_array_section(xml, Some(type_name), name_filter) {
            return section.split_whitespace()
                .map(|s: &str| s.parse().map_err(|_| FemError::Mesh(format!("VTU: bad int: {s}"))))
                .collect()
        }
    }
    // Try without type filter
    let section = data_array_section(xml, None, name_filter)
        .ok_or_else(|| FemError::Mesh("VTU: missing integer DataArray".into()))?;
    section.split_whitespace()
        .map(|s: &str| s.parse().map_err(|_| FemError::Mesh(format!("VTU: bad int: {s}"))))
        .collect()
}

fn parse_data_array_u8(xml: &str, name_filter: Option<&str>) -> FemResult<Vec<u8>> {
    for type_name in &["UInt8", "Int8"] {
        if let Some(section) = data_array_section(xml, Some(type_name), name_filter) {
            return section.split_whitespace()
                .map(|s| s.parse().map_err(|_| FemError::Mesh(format!("VTU: bad u8: {s}"))))
                .collect();
        }
    }
    // Fallback: parse as usize and truncate
    let section = data_array_section(xml, None, name_filter)
        .ok_or_else(|| FemError::Mesh("VTU: missing UInt8 DataArray".into()))?;
    section.split_whitespace()
        .map(|s: &str| s.parse::<usize>().map(|v| v as u8).map_err(|_| FemError::Mesh(format!("VTU: bad u8: {s}"))))
        .collect()
}

fn parse_point_data_array(xml: &str, open_tag: &str, close_tag: &str) -> FemResult<HashMap<String, (usize, Vec<f64>)>> {
    let mut result = HashMap::new();
    let section = match extract_section(xml, open_tag, close_tag) {
        Some(s) => s,
        None => return Ok(result),
    };
    let mut cursor = 0;
    while let Some(da_start) = section[cursor..].find("<DataArray") {
        let abs = cursor + da_start;
        let tag_end = section[abs..].find('>').ok_or_else(|| FemError::Mesh("VTU: bad DataArray".into()))? + abs;
        let tag = &section[abs..=tag_end];
        let name = extract_attr(tag, "Name")
            .ok_or_else(|| FemError::Mesh("VTU: DataArray missing Name".into()))?;
        let n_comp: usize = extract_attr(tag, "NumberOfComponents")
            .and_then(|s| s.parse().ok()).unwrap_or(1);
        let close = section[tag_end..].find("</DataArray>")
            .ok_or_else(|| FemError::Mesh("VTU: unclosed DataArray".into()))? + tag_end;
        let data: Vec<f64> = section[tag_end + 1..close]
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        result.insert(name, (n_comp, data));
        cursor = close + "</DataArray>".len();
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vtk::{DataArray, VtkWriter};
    use fem_mesh::Mesh;

    #[test]
    fn roundtrip_tet_mesh() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let w = VtkWriter::new(&mesh);
        let mut buf = Vec::new();
        w.write(&mut buf).unwrap();
        let xml = String::from_utf8(buf).unwrap();
        let vtu = parse_vtu(&xml).unwrap();
        assert_eq!(vtu.mesh.n_nodes(), mesh.n_nodes());
        assert_eq!(vtu.mesh.n_elems(), mesh.n_elems());
    }

    #[test]
    fn roundtrip_hex_mesh() {
        let mesh = Mesh::<3>::unit_cube_hex(2);
        let w = VtkWriter::new(&mesh);
        let mut buf = Vec::new();
        w.write(&mut buf).unwrap();
        let xml = String::from_utf8(buf).unwrap();
        let vtu = parse_vtu(&xml).unwrap();
        assert_eq!(vtu.mesh.n_elems(), mesh.n_elems());
    }

    #[test]
    fn roundtrip_point_data() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let mut w = VtkWriter::new(&mesh);
        w.add_point_data(DataArray::scalars("u", u.clone()));
        let mut buf = Vec::new();
        w.write(&mut buf).unwrap();
        let xml = String::from_utf8(buf).unwrap();
        let vtu = parse_vtu(&xml).unwrap();
        assert!(vtu.point_data.contains_key("u"));
        let (nc, vals) = &vtu.point_data["u"];
        assert_eq!(*nc, 1);
        assert_eq!(vals.len(), n);
    }

    #[test]
    fn missing_point_data_ok() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let w = VtkWriter::new(&mesh);
        let mut buf = Vec::new();
        w.write(&mut buf).unwrap();
        let xml = String::from_utf8(buf).unwrap();
        let vtu = parse_vtu(&xml).unwrap();
        assert!(vtu.point_data.is_empty());
    }
}
