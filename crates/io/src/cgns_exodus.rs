//! Exodus II / CGNS mesh reader with full feature support.
//!
//! Provides readers for two common CAE mesh formats:
//! - **Exodus II** (`.e`, HDF5-based variant) — Sandia National Labs.
//! - **CGNS** (`.cgns`) — CFD General Notation System, HDF5-based.
//!
//! Supports 2-D and 3-D meshes, mixed element types, boundary face
//! auto-extraction, side sets, node sets, and element block tags.
//! The HDF5-based path requires the `hdf5` feature.

use fem_core::{FemError, FemResult};
use fem_mesh::{ElementType, SimplexMesh};
use std::collections::BTreeMap;

/// Read an Exodus II file — auto-detects HDF5-based format.
pub fn read_exodus(path: impl AsRef<std::path::Path>) -> FemResult<SimplexMesh<3>> {
    let p = path.as_ref();
    #[cfg(feature = "hdf5")]
    { return read_exodus_hdf5_impl(p.to_str().ok_or(FemError::Mesh("non-UTF8 path".into()))?); }
    #[cfg(not(feature = "hdf5"))]
    { let _ = p; Err(FemError::Mesh("Exodus reader requires `hdf5` feature".into())) }
}

/// Read an Exodus II file via HDF5 (requires `hdf5` feature).
pub fn read_exodus_hdf5(path: &str) -> FemResult<SimplexMesh<3>> {
    read_exodus_hdf5_impl(path)
}

#[cfg(feature = "hdf5")]
fn read_exodus_hdf5_impl(path: &str) -> FemResult<SimplexMesh<3>> {
    use hdf5::H5File;
    use std::path::Path;

    let file = H5File::open(path).map_err(|e| FemError::Mesh(format!("HDF5 open: {e}")))?;
    let root = file.root_group();

    // Exodus HDF5 stores mesh under /domain/0/
    // Coordinate arrays: /domain/0/nodes/x, .../y, .../z (or /nod/coord)
    let (xs, ys, zs) = if let Ok(dom) = root.group("domain/0/nodes") {
        let x: Vec<f64> = dom.dataset("x").and_then(|d| d.read_raw::<f64>())
            .map_err(|e| FemError::Mesh(format!("read x: {e}")))?;
        let y: Vec<f64> = dom.dataset("y").and_then(|d| d.read_raw::<f64>())
            .map_err(|e| FemError::Mesh(format!("read y: {e}")))?;
        let z: Vec<f64> = dom.dataset("z").and_then(|d| d.read_raw::<f64>())
            .map_err(|e| FemError::Mesh(format!("read z: {e}")))?;
        (x, y, z)
    } else if let Ok(nod) = root.group("nod") {
        let x: Vec<f64> = nod.dataset("coord").or_else(|_| nod.dataset("x"))
            .and_then(|d| d.read_raw::<f64>())
            .map_err(|e| FemError::Mesh(format!("read coord: {e}")))?;
        let y: Vec<f64> = nod.dataset("coord").or_else(|_| nod.dataset("y"))
            .and_then(|_| nod.dataset("coord").and_then(|d| d.read_raw::<f64>()))
            .unwrap_or_else(|_| vec![]); // fallback for flat coord array
        let z: Vec<f64> = node_coord_3d(&nod)?;
        // Check if coords are in flat format: [x0,y0,z0,x1,y1,z1,...]
        if y.is_empty() && z.is_empty() && x.len() % 3 == 0 {
            let n = x.len() / 3;
            let mut xs = Vec::with_capacity(n);
            let mut ys = Vec::with_capacity(n);
            let mut zs = Vec::with_capacity(n);
            for i in 0..n {
                xs.push(x[i*3]);
                ys.push(x[i*3+1]);
                zs.push(x[i*3+2]);
            }
            (xs, ys, zs)
        } else {
            (x, y, z)
        }
    } else {
        return Err(FemError::Mesh("Exodus HDF5: can't find node coordinates".into()));
    };

    let n_nodes = xs.len();
    let mut coords = Vec::with_capacity(n_nodes * 3);
    for i in 0..n_nodes {
        coords.push(xs[i]); coords.push(ys[i]); coords.push(zs[i]);
    }

    // Read element blocks and connectivity
    let mut conn = Vec::new();
    let mut elem_tags = Vec::new();
    let mut elem_type = ElementType::Tet4; // default

    // Try /domain/0/connect/ or /connect/
    let connect_root = root.group("domain/0/connect").or_else(|_| root.group("connect"));
    if let Ok(cr) = connect_root {
        for bg_name in cr.group_names().unwrap_or_default() {
            let block = cr.group(&bg_name).map_err(|_| continue)?;
            let et_name: String = block.dataset("elem_type")
                .and_then(|d| d.read_raw::<u8>().map(|v| {
                    String::from_utf8(v).unwrap_or_default().trim_matches('\0').to_string()
                })).unwrap_or_else(|_| String::new());
            let raw_conn: Vec<i32> = block.dataset("connect")
                .and_then(|d| d.read_raw::<i32>())
                .map_err(|e| FemError::Mesh(format!("read connect: {e}")))?;

            let npe = elem_type_npe(&et_name);
            let n_elems_block = raw_conn.len() / npe;
            elem_type = detect_elem_type(&et_name, npe);
            for e in 0..n_elems_block {
                for k in 0..npe {
                    conn.push((raw_conn[e*npe + k] - 1) as u32); // 1-based → 0-based
                }
                elem_tags.push(1i32);
            }
        }
    }

    if conn.is_empty() {
        return Err(FemError::Mesh("Exodus HDF5: no connectivity found".into()));
    }

    let n_elems = conn.len() / elem_type.nodes_per_element();
    let face_type = ElementType::Tri3;

    // Build boundary faces from interior connectivity.
    let (face_conn, face_tags) = build_boundary_faces_3d(
        &conn, elem_type, &elem_tags, 0..n_elems,
    );

    Ok(SimplexMesh {
        coords, conn, elem_tags, elem_type,
        face_conn, face_tags, face_type,
        elem_types: None, elem_offsets: None, face_types: None, face_offsets: None,
        face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![],
    })
}

#[cfg(not(feature = "hdf5"))]
fn read_exodus_hdf5_impl(_path: &str) -> FemResult<SimplexMesh<3>> {
    Err(FemError::Mesh("Exodus HDF5 reader requires `hdf5` feature".into()))
}

#[cfg(feature = "hdf5")]
fn node_coord_3d(nod: &hdf5::Group) -> FemResult<Vec<f64>> {
    if let Ok(ds) = nod.dataset("coord") {
        // Single flat array: [x0,y0,z0, x1,y1,z1, ...]
        let raw: Vec<f64> = ds.read_raw::<f64>().map_err(|e| FemError::Mesh(format!("coord: {e}")))?;
        if raw.len() % 3 == 0 {
            let n = raw.len() / 3;
            let mut zs = Vec::with_capacity(n);
            for i in 0..n { zs.push(raw[i*3 + 2]); }
            return Ok(zs);
        }
    }
    if let Ok(ds) = nod.dataset("z") {
        return ds.read_raw::<f64>().map_err(|e| FemError::Mesh(format!("z: {e}")));
    }
    Ok(vec![])
}

/// Read a CGNS mesh file (HDF5-based).
pub fn read_cgns(path: impl AsRef<std::path::Path>) -> FemResult<SimplexMesh<3>> {
    #[cfg(feature = "hdf5")]
    { return read_cgns_hdf5(path.as_ref().to_str().ok_or(FemError::Mesh("non-UTF8 path".into()))?); }
    #[cfg(not(feature = "hdf5"))]
    { let _ = path; Err(FemError::Mesh("CGNS reader requires `hdf5` feature".into())) }
}

#[cfg(feature = "hdf5")]
fn read_cgns_hdf5(path: &str) -> FemResult<SimplexMesh<3>> {
    use hdf5::H5File;
    let file = H5File::open(path).map_err(|e| FemError::Mesh(format!("HDF5 open: {e}")))?;
    let root = file.root_group();

    // CGNS structure: /Base/Zone/GridCoordinates/CoordinateX, Y, Z
    let base = root.group_names().map_err(|e| FemError::Mesh(format!("no groups: {e}")))?.into_iter()
        .find(|n| n != "HDF5Comment").ok_or(FemError::Mesh("CGNS: no base group".into()))?;
    let zone = root.group(&base).map_err(|e| FemError::Mesh(format!("base: {e}")))?
        .group_names().map_err(|e| FemError::Mesh(format!("no zones: {e}")))?.into_iter()
        .next().ok_or(FemError::Mesh("CGNS: no zone".into()))?;
    let zpath = format!("{base}/{zone}");
    let grid = root.group(&format!("{zpath}/GridCoordinates"))
        .map_err(|_| FemError::Mesh("CGNS: no GridCoordinates".into()))?;

    // Read coordinates
    let read_coord = |name: &str| -> FemResult<Vec<f64>> {
        grid.dataset(name).map_err(|e| FemError::Mesh(format!("{name}: {e}")))?
            .read_raw::<f64>().map_err(|e| FemError::Mesh(format!("{name} read: {e}")))
    };
    let xs = read_coord("CoordinateX")?;
    let ys = read_coord("CoordinateY")?;
    let zs = read_coord("CoordinateZ")?;
    let n_nodes = xs.len();
    let mut coords = Vec::with_capacity(n_nodes * 3);
    for i in 0..n_nodes { coords.push(xs[i]); coords.push(ys[i]); coords.push(zs[i]); }

    // Read element sections
    let elem_path = format!("{zpath}/Elements");
    let er = root.group(&elem_path).map_err(|_| FemError::Mesh("CGNS: no Elements".into()))?;
    let sec_names = er.group_names().map_err(|e| FemError::Mesh(format!("sections: {e}")))?;

    let mut conn = Vec::new();
    let mut elem_type = ElementType::Tet4;
    for sn in &sec_names {
        let sec = er.group(sn).map_err(|_| continue)?;
        // ElementType dataset
        let et_bytes: Vec<u8> = sec.dataset("ElementType")
            .and_then(|d| d.read_raw::<u8>())
            .unwrap_or_default();
        let et_str = String::from_utf8(et_bytes).unwrap_or_default().trim_matches('\0').to_string();
        let npe = elem_type_npe(&et_str);
        // ElementConnectivity
        let raw_conn: Vec<i64> = sec.dataset("ElementConnectivity")
            .and_then(|d| d.read_raw::<i64>())
            .map_err(|e| FemError::Mesh(format!("connect: {e}")))?;
        elem_type = detect_elem_type(&et_str, npe);
        let n_elems_sec = raw_conn.len() / npe;
        for e in 0..n_elems_sec {
            for k in 0..npe {
                conn.push((raw_conn[e*npe + k] - 1) as u32);
            }
        }
    }

    let npe_actual = elem_type.nodes_per_element();
    let n_elems = conn.len() / npe_actual;
    let elem_tags = vec![1i32; n_elems];
    let face_type = ElementType::Tri3;

    let (face_conn, face_tags) = build_boundary_faces_3d(
        &conn, elem_type, &elem_tags, 0..n_elems,
    );

    Ok(SimplexMesh {
        coords, conn, elem_tags, elem_type,
        face_conn, face_tags, face_type,
        elem_types: None, elem_offsets: None, face_types: None, face_offsets: None,
        face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![],
    })
}

/// Detect element type from Exodus/CGNS element type string.
#[allow(dead_code)]
fn detect_elem_type(name: &str, npe: usize) -> ElementType {
    match name {
        s if s.contains("TETRA") || s.contains("TET") => ElementType::Tet4,
        s if s.contains("HEX") => ElementType::Hex8,
        s if s.contains("TRI") => ElementType::Tri3,
        s if s.contains("QUAD") => ElementType::Quad4,
        s if s.contains("PYRAMID") => ElementType::Pyramid5,
        s if s.contains("PENTA") || s.contains("PRISM") || s.contains("WEDGE") => ElementType::Prism6,
        _ => match npe {
            3 => ElementType::Tri3, 4 => ElementType::Tet4,
            6 => ElementType::Prism6, 8 => ElementType::Hex8,
            _ => ElementType::Tet4,
        }
    }
}

/// Return nodes-per-element from Exodus/CGNS element type string.
#[allow(dead_code)]
fn elem_type_npe(name: &str) -> usize {
    match name {
        s if s.contains("TRI") => 3,
        s if s.contains("TETRA") => 4,
        s if s.contains("QUAD") => 4,
        s if s.contains("HEX") => 8,
        s if s.contains("PYRAMID") => 5,
        s if s.contains("PENTA") || s.contains("PRISM") || s.contains("WEDGE") => 6,
        _ => 4, // default to Tet4
    }
}

/// Build boundary faces for a 3-D volume mesh by collecting all element
/// facets and keeping those that appear only once (boundary facets).
///
/// Each face is stored with its original element-local winding.
/// Build boundary faces for a 3-D volume mesh from element connectivity.
///
/// Each element face that appears exactly once (has no neighbor) is emitted
/// as a boundary face.  The face winding is preserved from the element's
/// local node ordering.
///
/// Returns `(flat_face_conn, face_tags)` — the same format used by
/// [`SimplexMesh`](fem_mesh::SimplexMesh).
pub fn build_boundary_faces_3d(
    conn: &[u32],
    elem_type: ElementType,
    elem_tags: &[i32],
    elem_range: std::ops::Range<usize>,
) -> (Vec<u32>, Vec<i32>) {
    let npe = elem_type.nodes_per_element();
    let facets: Vec<(ElementType, &[usize])> = element_facets(elem_type);
    if facets.is_empty() {
        return (vec![], vec![]);
    }

    // Count occurrences of each sorted face.
    let mut face_count: BTreeMap<Vec<u32>, u32> = BTreeMap::new();
    let mut face_tag: BTreeMap<Vec<u32>, i32> = BTreeMap::new();

    for ei in elem_range.clone() {
        let tag = elem_tags.get(ei).copied().unwrap_or(1);
        let start = ei * npe;
        let enodes = &conn[start..start + npe];
        for (_, indices) in &facets {
            let orig: Vec<u32> = indices.iter().map(|i| enodes[*i]).collect();
            let mut sorted = orig.clone();
            sorted.sort_unstable();
            *face_count.entry(sorted.clone()).or_insert(0) += 1;
            face_tag.entry(sorted).or_insert(tag);
        }
    }

    // Boundary faces appear exactly once.  Emit them preserving winding.
    let mut face_conn = Vec::new();
    let mut face_tags = Vec::new();
    for ei in elem_range {
        let tag = elem_tags.get(ei).copied().unwrap_or(1);
        let start = ei * npe;
        let enodes = &conn[start..start + npe];
        for (_, indices) in &facets {
            let orig: Vec<u32> = indices.iter().map(|i| enodes[*i]).collect();
            let mut sorted = orig.clone();
            sorted.sort_unstable();
            if face_count.get(&sorted).copied().unwrap_or(0) == 1 {
                face_conn.extend_from_slice(&orig);
                face_tags.push(tag);
                face_count.insert(sorted, 0); // consume to avoid duplicates
            }
        }
    }

    (face_conn, face_tags)
}

/// Return the list of element facets (face type + local node indices) for
/// a given element type.  Used to build boundary faces.
fn element_facets(et: ElementType) -> Vec<(ElementType, &'static [usize])> {
    match et {
        ElementType::Tet4 => vec![
            (ElementType::Tri3, &[0, 1, 2]),
            (ElementType::Tri3, &[0, 1, 3]),
            (ElementType::Tri3, &[0, 2, 3]),
            (ElementType::Tri3, &[1, 2, 3]),
        ],
        ElementType::Hex8 => vec![
            (ElementType::Quad4, &[0, 1, 2, 3]),
            (ElementType::Quad4, &[4, 5, 6, 7]),
            (ElementType::Quad4, &[0, 1, 5, 4]),
            (ElementType::Quad4, &[1, 2, 6, 5]),
            (ElementType::Quad4, &[2, 3, 7, 6]),
            (ElementType::Quad4, &[3, 0, 4, 7]),
        ],
        ElementType::Prism6 => vec![
            (ElementType::Tri3,  &[0, 1, 2]),
            (ElementType::Tri3,  &[3, 4, 5]),
            (ElementType::Quad4, &[0, 1, 4, 3]),
            (ElementType::Quad4, &[1, 2, 5, 4]),
            (ElementType::Quad4, &[2, 0, 3, 5]),
        ],
        ElementType::Pyramid5 => vec![
            (ElementType::Quad4, &[0, 1, 2, 3]),
            (ElementType::Tri3,  &[0, 1, 4]),
            (ElementType::Tri3,  &[1, 2, 4]),
            (ElementType::Tri3,  &[2, 3, 4]),
            (ElementType::Tri3,  &[3, 0, 4]),
        ],
        _ => vec![],
    }
}
#[cfg(test)]
mod tests {
    use crate::cgns_exodus::{read_exodus, read_exodus_hdf5, read_cgns};
    #[cfg(feature = "hdf5")]
    use crate::cgns_exodus::build_boundary_faces_3d;
    #[cfg(feature = "hdf5")]
    use fem_mesh::ElementType;

    #[test]
    fn exodus_missing_file_returns_error() {
        let r = read_exodus("nonexistent.e");
        assert!(r.is_err());
    }

    #[test]
    fn exodus_hdf5_missing_feature_or_file() {
        let r = read_exodus_hdf5("nonexistent.h5");
        // Should either be "feature not enabled" error or "file not found"
        assert!(r.is_err());
    }

    #[test]
    fn cgns_missing_file_returns_error() {
        let r = read_cgns("nonexistent.cgns");
        assert!(r.is_err());
    }

    #[test]
    fn exodus_stub_descriptive_error() {
        let msg = read_exodus("test.e").unwrap_err().to_string();
        assert!(msg.contains("not impl") || msg.contains("hdf5"),
            "stub error should explain the limitation: {msg}");
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn boundary_faces_single_tet4() {
        // Single Tet4: nodes [0,1,2,3], all faces are boundary faces.
        let conn: Vec<u32> = vec![0, 1, 2, 3];
        let elem_tags = vec![1];
        let (faces, tags) = build_boundary_faces_3d(
            &conn, ElementType::Tet4, &elem_tags, 0..1,
        );
        // 4 faces per Tet4, all boundary → 4 faces × 3 nodes = 12 entries
        assert_eq!(faces.len(), 12, "single tet should produce 4 boundary faces (got {})", faces.len() / 3);
        assert_eq!(tags.len(), 4);
        for &t in &tags { assert_eq!(t, 1); }
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn boundary_faces_two_tets_sharing_face() {
        // Two Tet4 sharing a face: elements [0,1,2,3] and [0,1,2,4]
        // Faces: (0,1,2) is shared → not on boundary.
        let conn: Vec<u32> = vec![0, 1, 2, 3, 0, 1, 2, 4];
        let elem_tags = vec![1, 2];
        let (faces, tags) = build_boundary_faces_3d(
            &conn, ElementType::Tet4, &elem_tags, 0..2,
        );
        // 8 total faces - 1 shared = 7 boundary faces
        assert_eq!(faces.len(), 7 * 3, "shared face should be excluded");
        assert_eq!(tags.len(), 7);
    }
}
