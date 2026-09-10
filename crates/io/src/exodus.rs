//! Exodus II (Genesis) mesh writer — port of MFEM `Mesh::PrintExodusII`
//! (`mesh/exodus_writer.cpp`).
//!
//! Writes a Cubit/Genesis NetCDF classic file (see [`super::netcdf`]) with:
//! - global attributes: `title`, `version`, `api_version`,
//!   `floating_point_word_size`, `file_size`, `maximum_name_length`,
//!   `maximum_line_length` (as in MFEM's `WriteExodusIIFileInformation`),
//! - element blocks grouped by attribute (first-appearance order), each with
//!   `num_el_in_blk{id}`, `num_nod_per_el{id}`, `num_edg_per_el{id}`,
//!   `num_fac_per_el{id}` dimensions and a `connect{id}` variable carrying an
//!   `elem_type` attribute (`Hex8`, `TETRA4`, `WEDGE6`, `PYRAMID5`),
//! - side sets derived from the exterior boundary faces, with Exodus II side
//!   ids obtained by matching face node sets (`cubit_side_map_*`),
//! - `eb_prop1` / `ss_prop1` id variables, `coordx/coordy/coordz` and the
//!   `dummy_var`/`time_step` placeholders MFEM writes.
//!
//! Notes (upstream quirks kept for 1:1 fidelity):
//! - MFEM labels the per-block/per-sideset dimensions and variables with the
//!   *attribute value* (block id), while MFEM's *reader* looks them up by
//!   sequential index; round-tripping through MFEM therefore requires block
//!   attributes `1..=num_el_blk` in ascending order.  This writer reproduces
//!   the MFEM-writer labeling.
//! - Higher-order meshes are rejected (MFEM's order-2 path requires a nodal
//!   H1 grid function projection).

use std::collections::BTreeMap;
use std::io::Write;

use fem_core::{FemError, FemResult, NodeId};
use fem_mesh::element_type::ElementType;
use fem_mesh::simplex::Mesh;

use super::netcdf::{NcOutput, NcValue};
use super::cubit::side_map_nodes;

fn exodus_err(msg: impl Into<String>) -> FemError {
    FemError::Mesh(format!("exodus: {}", msg.into()))
}

/// Number of edges per linear element (`Element::GetNEdges`).
fn num_edges(et: ElementType) -> usize {
    match et {
        ElementType::Tet4 => 6,
        ElementType::Hex8 => 12,
        ElementType::Prism6 => 9,
        ElementType::Pyramid5 => 8,
        _ => 0,
    }
}

/// Number of faces per linear element (`Element::GetNFaces`).
fn num_faces(et: ElementType) -> usize {
    match et {
        ElementType::Tet4 => 4,
        ElementType::Hex8 => 6,
        ElementType::Prism6 => 5,
        ElementType::Pyramid5 => 5,
        _ => 0,
    }
}

/// Number of Exodus II sides per element type.
fn num_sides(et: ElementType) -> i32 {
    match et {
        ElementType::Tet4 => 4,
        ElementType::Hex8 => 6,
        ElementType::Prism6 | ElementType::Pyramid5 => 5,
        _ => 0,
    }
}

/// Exodus II element type string (MFEM `WriteElementBlockParameters`).
fn exodus_element_type(et: ElementType) -> Option<&'static str> {
    match et {
        ElementType::Hex8 => Some("Hex8"),
        ElementType::Tet4 => Some("TETRA4"),
        ElementType::Prism6 => Some("WEDGE6"),
        ElementType::Pyramid5 => Some("PYRAMID5"),
        _ => None,
    }
}

/// Write a 3-D linear mesh to an Exodus II file stream.
pub fn write_exodus(mesh: &Mesh<3>, mut writer: impl Write) -> FemResult<()> {
    let bytes = serialize_exodus(mesh)?;
    writer.write_all(&bytes)?;
    Ok(())
}

/// Convenience wrapper: write an `.exo`/`.gen` file by path.
pub fn write_exodus_file(mesh: &Mesh<3>, path: impl AsRef<std::path::Path>) -> FemResult<()> {
    let bytes = serialize_exodus(mesh)?;
    std::fs::File::create(path)?.write_all(&bytes)?;
    Ok(())
}

/// Build the complete NetCDF-3 file contents.
fn serialize_exodus(mesh: &Mesh<3>) -> FemResult<Vec<u8>> {
    if mesh.geometry.is_some() {
        return Err(exodus_err(
            "higher-order meshes are not supported by the Exodus II writer \
             (MFEM requires a second-order H1 nodal grid function)",
        ));
    }

    let n_elems = mesh.n_elems();
    let n_nodes = mesh.n_nodes();

    // ── GenerateExodusIIElementBlocks: group by attribute, first-appearance ──
    let mut block_ids: Vec<i32> = Vec::new();
    let mut elems_for_block: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    let mut etype_for_block: BTreeMap<i32, ElementType> = BTreeMap::new();
    for e in 0..n_elems {
        let eid = e as u32;
        let block_id = mesh.elem_tags.get(e).copied().unwrap_or(1);
        let et = mesh.element_type_at(eid);
        if !elems_for_block.contains_key(&block_id) {
            block_ids.push(block_id);
            etype_for_block.insert(block_id, et);
        } else if etype_for_block[&block_id] != et {
            return Err(exodus_err(format!(
                "multiple element types are defined for block {block_id}"
            )));
        }
        elems_for_block.entry(block_id).or_default().push(e);
    }
    for &block_id in &block_ids {
        if exodus_element_type(etype_for_block[&block_id]).is_none() {
            return Err(exodus_err(format!(
                "unsupported element type {et:?} in block {block_id}",
                et = etype_for_block[&block_id]
            )));
        }
    }

    // ── GenerateExodusIIBoundaryInfo: exterior faces → (element, exodus side)
    let mut face_set_to_index: BTreeMap<Vec<u32>, (usize, i32)> = BTreeMap::new();
    let mut face_multiplicity: BTreeMap<Vec<u32>, u32> = BTreeMap::new();
    for e in 0..n_elems {
        let eid = e as u32;
        let et = mesh.element_type_at(eid);
        let verts = element_nodes(mesh, eid);
        let verts1: Vec<i32> = verts.iter().map(|&v| v as i32 + 1).collect();
        for side in 1..=num_sides(et) {
            let nodes1 = side_map_nodes(et, side, &verts1)
                .ok_or_else(|| exodus_err("invalid side id"))?;
            let mut key: Vec<u32> = nodes1.iter().map(|&n| (n - 1) as u32).collect();
            key.sort_unstable();
            face_set_to_index
                .entry(key.clone())
                .or_insert((e, side));
            *face_multiplicity.entry(key).or_insert(0) += 1;
        }
    }
    // Boundary faces in file order: match their node sets.
    let mut boundary_ids: Vec<i32> = Vec::new();
    let mut elems_for_boundary: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
    let mut sides_for_boundary: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
    for f in 0..mesh.n_faces() {
        let mut key = face_nodes(mesh, f as u32).to_vec();
        key.sort_unstable();
        // Skip faces shared by two elements (interior faces, MFEM FaceIsInterior).
        if face_multiplicity.get(&key).copied().unwrap_or(0) != 1 {
            continue;
        }
        let Some(&(elem, side)) = face_set_to_index.get(&key) else {
            continue;
        };
        let tag = mesh.face_tags.get(f).copied().unwrap_or(1);
        if !elems_for_boundary.contains_key(&tag) {
            boundary_ids.push(tag);
        }
        elems_for_boundary.entry(tag).or_default().push(elem as i32 + 1);
        sides_for_boundary.entry(tag).or_default().push(side);
    }
    boundary_ids.sort_unstable();

    // ── Build the NetCDF file ────────────────────────────────────────────────
    let mut nc = NcOutput::new();

    // File information (WriteExodusIIFileInformation).
    nc.add_global_attr("title", NcValue::Char(b"MFEM mesh".to_vec()));
    nc.add_global_attr("version", NcValue::Double(vec![4.72]));
    nc.add_global_attr("api_version", NcValue::Double(vec![4.72]));
    nc.add_global_attr(
        "floating_point_word_size",
        NcValue::Int(vec![std::mem::size_of::<f64>() as i32]),
    );
    nc.add_global_attr("file_size", NcValue::Int(vec![1]));
    nc.add_global_attr("maximum_name_length", NcValue::Int(vec![80]));
    nc.add_global_attr("maximum_line_length", NcValue::Int(vec![80]));

    // Dimensions (WriteMeshDimension / WriteNumOfElements / WriteTimesteps /
    // WriteNodeSets / WriteDummyVariable / boundaries / blocks).
    let d_dim = nc.add_dim("num_dim", 3);
    let d_elem = nc.add_dim("num_elem", n_elems as u32);
    let _d_time = nc.add_dim("time_step", 1);
    let d_nodes = nc.add_dim("num_nodes", n_nodes as u32);
    let d_blk = nc.add_dim("num_el_blk", block_ids.len() as u32);
    let d_blk_dim = nc.add_dim("block_dim", block_ids.len() as u32);
    let d_nbdr = nc.add_dim("boundary_ids_dim", boundary_ids.len() as u32);
    let d_nss = nc.add_dim("num_side_sets", boundary_ids.len() as u32);
    let d_nns = nc.add_dim("num_node_sets", 0);
    let d_dummy = nc.add_dim("dummy_var_dim", 1);
    let mut d_per_block: BTreeMap<i32, [u32; 4]> = BTreeMap::new();
    let mut d_connect: BTreeMap<i32, u32> = BTreeMap::new();
    for &block_id in &block_ids {
        let et = etype_for_block[&block_id];
        let ne = elems_for_block[&block_id].len() as u32;
        d_per_block.insert(
            block_id,
            [
                nc.add_dim(&format!("num_el_in_blk{block_id}"), ne),
                nc.add_dim(
                    &format!("num_nod_per_el{block_id}"),
                    et.nodes_per_element() as u32,
                ),
                nc.add_dim(&format!("num_edg_per_el{block_id}"), num_edges(et) as u32),
                nc.add_dim(&format!("num_fac_per_el{block_id}"), num_faces(et) as u32),
            ],
        );
        // MFEM flattens the connectivity into a 1-D variable over its own
        // "connect{id}_dim" (ne * npe) dimension.
        d_connect.insert(
            block_id,
            nc.add_dim(
                &format!("connect{block_id}_dim"),
                ne * et.nodes_per_element() as u32,
            ),
        );
    }
    let mut d_per_ss: BTreeMap<i32, u32> = BTreeMap::new();
    for &bid in &boundary_ids {
        let n = elems_for_boundary[&bid].len() as u32;
        d_per_ss.insert(bid, nc.add_dim(&format!("num_side_ss{bid}"), n));
        nc.add_dim(&format!("side_ss{bid}_dim"), n);
        nc.add_dim(&format!("elem_ss{bid}_dim"), n);
    }

    // Variables.
    let _ = (d_dim, d_elem, d_nodes, d_blk, d_blk_dim, d_nbdr, d_nss, d_nns, d_dummy);
    nc.add_var_i32("dummy_var", &[d_dummy], &[1]);

    // Nodal coordinates (WriteNodalCoordinates).
    nc.add_var_f64(
        "coordx",
        &[d_nodes],
        &(0..n_nodes).map(|i| mesh.coords[i * 3]).collect::<Vec<_>>(),
    );
    nc.add_var_f64(
        "coordy",
        &[d_nodes],
        &(0..n_nodes).map(|i| mesh.coords[i * 3 + 1]).collect::<Vec<_>>(),
    );
    nc.add_var_f64(
        "coordz",
        &[d_nodes],
        &(0..n_nodes).map(|i| mesh.coords[i * 3 + 2]).collect::<Vec<_>>(),
    );

    // Element blocks (WriteElementBlocks).
    nc.add_var_i32("eb_prop1", &[d_blk], &block_ids);
    for &block_id in &block_ids {
        let et = etype_for_block[&block_id];
        let d_conn = d_connect[&block_id];
        let conn: Vec<i32> = elems_for_block[&block_id]
            .iter()
            .flat_map(|&e| element_nodes(mesh, e as u32).iter().map(|&v| v as i32 + 1))
            .collect();
        let vi = nc.add_var_i32(&format!("connect{block_id}"), &[d_conn], &conn);
        nc.add_var_attr(
            vi,
            "elem_type",
            NcValue::Char(exodus_element_type(et).unwrap().as_bytes().to_vec()),
        );
    }

    // Boundaries (WriteBoundaries).
    nc.add_var_i32("ss_prop1", &[d_nss], &boundary_ids);
    for &bid in &boundary_ids {
        let dd = d_per_ss[&bid];
        nc.add_var_i32(&format!("side_ss{bid}"), &[dd], &sides_for_boundary[&bid]);
        nc.add_var_i32(&format!("elem_ss{bid}"), &[dd], &elems_for_boundary[&bid]);
    }

    Ok(nc.finish())
}

/// Connectivity of element `e`, honoring mixed-element meshes.
fn element_nodes(mesh: &Mesh<3>, e: u32) -> &[NodeId] {
    if let Some(ref offsets) = mesh.elem_offsets {
        let e = e as usize;
        &mesh.conn[offsets[e]..offsets[e + 1]]
    } else {
        let npe = mesh.elem_type.nodes_per_element();
        let e = e as usize;
        &mesh.conn[npe * e..npe * (e + 1)]
    }
}

/// Boundary face `f` node ids (0-based).
fn face_nodes(mesh: &Mesh<3>, f: u32) -> &[NodeId] {
    if let Some(ref offsets) = mesh.face_offsets {
        let f = f as usize;
        &mesh.face_conn[offsets[f]..offsets[f + 1]]
    } else {
        let npf = mesh.face_type.nodes_per_element();
        let f = f as usize;
        &mesh.face_conn[npf * f..npf * (f + 1)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubit::{read_cubit, CubitMesh};

    /// 1 x 1 x 1 hex cube, material 1, side set 1 on the top face (z = 1).
    fn cube_mesh() -> Mesh<3> {
        let mut coords = Vec::new();
        for z in [0.0_f64, 1.0] {
            for y in [0.0_f64, 1.0] {
                for x in [0.0_f64, 1.0] {
                    coords.extend_from_slice(&[x, y, z]);
                }
            }
        }
        // Node order: 0(0,0,0) 1(1,0,0) 2(0,1,0) 3(1,1,0) 4(0,0,1) 5(1,0,1)
        // 6(0,1,1) 7(1,1,1) → MFEM Hex8 ordering (bottom 0-3, top 4-7).
        let order = [0usize, 1, 3, 2, 4, 5, 7, 6];
        let conn: Vec<NodeId> = order.iter().map(|&i| i as NodeId).collect();
        Mesh {
            coords,
            conn,
            elem_tags: vec![1],
            elem_type: ElementType::Hex8,
            face_conn: vec![4, 5, 7, 6],
            face_tags: vec![1],
            face_type: ElementType::Quad4,
            elem_types: None,
            elem_offsets: None,
            face_types: None,
            face_offsets: None,
            face_to_elem: None,
            edge_conn: vec![],
            edge_to_elem: vec![],
            geometry: None,
            nc_vertex_view: None,
            vertex_parents: vec![],
        }
    }

    #[test]
    fn write_then_read_roundtrip() {
        let mesh = cube_mesh();
        let bytes = serialize_exodus(&mesh).expect("write failed");

        let nc = crate::netcdf::NetCdfFile::from_bytes(bytes).expect("netcdf parse failed");
        let read = match read_cubit(&nc).expect("cubit parse failed") {
            CubitMesh::D3(m) => *m,
            CubitMesh::D2(_) => panic!("expected 3-D mesh"),
        };

        assert_eq!(read.n_nodes(), 8);
        assert_eq!(read.n_elems(), 1);
        assert_eq!(read.n_faces(), 1);
        assert_eq!(read.elem_tags, vec![1]);
        assert_eq!(read.face_tags, vec![1]);
        // Connectivity preserved (vertex ids unchanged).
        assert_eq!(&read.conn[0..8], &mesh.conn[0..8]);
        // Side set 2 = exodus side 6 of the hex = top face (z = 1).
        assert_eq!(&read.face_conn[0..4], &[4, 5, 7, 6]);
        for i in 0..8 {
            assert_eq!(read.coords_of(i), mesh.coords_of(i));
        }
    }

    #[test]
    fn tet_mesh_side_set_roundtrip() {
        // Tet with vertices 0(0,0,0) 1(1,0,0) 2(0,1,0) 3(0,0,1); side set 1 on
        // exodus side 2 = {2,3,4} 1-based → 0-based {1,2,3}.
        let coords = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mesh = Mesh {
            coords,
            conn: vec![0, 1, 2, 3],
            elem_tags: vec![1],
            elem_type: ElementType::Tet4,
            face_conn: vec![1, 2, 3],
            face_tags: vec![1],
            face_type: ElementType::Tri3,
            elem_types: None,
            elem_offsets: None,
            face_types: None,
            face_offsets: None,
            face_to_elem: None,
            edge_conn: vec![],
            edge_to_elem: vec![],
            geometry: None,
            nc_vertex_view: None,
            vertex_parents: vec![],
        };
        let bytes = serialize_exodus(&mesh).expect("write failed");
        let nc = crate::netcdf::NetCdfFile::from_bytes(bytes).expect("netcdf parse failed");
        let read = match read_cubit(&nc).expect("cubit parse failed") {
            CubitMesh::D3(m) => *m,
            CubitMesh::D2(_) => panic!("expected 3-D mesh"),
        };
        assert_eq!(read.elem_type, ElementType::Tet4);
        assert_eq!(read.face_tags, vec![1]);
        assert_eq!(read.face_type, ElementType::Tri3);
        assert_eq!(&read.face_conn[0..3], &[1, 2, 3]);
    }

    #[test]
    fn rejects_higher_order() {
        let mut mesh = cube_mesh();
        mesh.geometry = Some(fem_mesh::simplex::GeometryData {
            order: 2,
            conn: vec![],
            nodes_per_elem: 27,
            coords: vec![],
            n_nodes: 0,
        });
        let err = serialize_exodus(&mesh).unwrap_err();
        assert!(err.to_string().contains("higher-order"), "got: {err}");
    }
}
