//! Cubit/Genesis (Exodus II) `.exo` / `.gen` reader.
//!
//! 1:1 port of the linear path of MFEM's `Mesh::ReadCubit`
//! (`mesh_readers.cpp`, namespace `cubit`): block/element/side-set data is
//! read from a NetCDF classic file (see [`super::netcdf`]) and converted to a
//! fem-rs mesh.
//!
//! Supported element blocks (first order, as in MFEM's
//! `CubitElement::GetElementType`): Tri3/Quad4 in 2-D, Tet4/Hex8/Wedge6/
//! Pyramid5 in 3-D.  Second-order blocks (Tri6, Quad9, Tet10, Hex27,
//! Wedge18, Pyramid14) require MFEM's quadratic `nodes` projection and are
//! rejected with a clear error.
//!
//! Boundary side-sets use the Exodus II side-id maps `cubit_side_map_*`.

use std::collections::BTreeMap;

use fem_core::{FemError, FemResult, NodeId};
use fem_mesh::element_type::ElementType;
use fem_mesh::simplex::Mesh;

use super::netcdf::NetCdfFile;

fn cubit_err(msg: impl Into<String>) -> FemError {
    FemError::Mesh(format!("cubit: {}", msg.into()))
}

/// A mesh read from a Cubit Genesis file (dimension taken from the file).
#[derive(Debug)]
pub enum CubitMesh {
    /// 2-D mesh (`num_dim == 2`).
    D2(Box<Mesh<2>>),
    /// 3-D mesh (`num_dim == 3`).
    D3(Box<Mesh<3>>),
}

impl CubitMesh {
    /// Spatial dimension of the mesh (2 or 3).
    pub fn dimension(&self) -> usize {
        match self {
            CubitMesh::D2(_) => 2,
            CubitMesh::D3(_) => 3,
        }
    }
}

/// Read a Cubit Genesis file from disk.
pub fn read_cubit_file(path: impl AsRef<std::path::Path>) -> FemResult<CubitMesh> {
    let f = NetCdfFile::open(path)?;
    read_cubit(&f)
}

/// Build the mesh from an already-parsed NetCDF file.
pub fn read_cubit(nc: &NetCdfFile) -> FemResult<CubitMesh> {
    // ReadCubitDimensions
    let num_dim = nc.dimension("num_dim")? as usize;
    let num_nodes = nc.dimension("num_nodes")? as usize;
    let num_elem = nc.dimension("num_elem")? as usize;
    let num_el_blk = nc.dimension("num_el_blk")? as usize;
    let num_side_sets = if nc.has_dimension("num_side_sets") {
        nc.dimension("num_side_sets")? as usize
    } else {
        0
    };
    if num_dim != 2 && num_dim != 3 {
        return Err(cubit_err(format!("unsupported num_dim {num_dim}")));
    }

    // BuildCubitBlockIDs: variable `eb_prop1`.
    let block_ids = nc.read_var_i32("eb_prop1")?;
    if block_ids.len() != num_el_blk {
        return Err(cubit_err("eb_prop1 length != num_el_blk"));
    }

    // ReadCubitNumElementsInBlock + ReadCubitBlocks: dims per block.
    let mut num_elements_for_block: BTreeMap<i32, usize> = BTreeMap::new();
    let mut elem_type_for_block: BTreeMap<i32, ElementType> = BTreeMap::new();
    for (i, &block_id) in block_ids.iter().enumerate() {
        let dim_name = format!("num_el_in_blk{}", i + 1);
        let n = nc.dimension(&dim_name)? as usize;
        num_elements_for_block.insert(block_id, n);

        let nod_name = format!("num_nod_per_el{}", i + 1);
        let npe = nc.dimension(&nod_name)? as usize;
        let et = element_type_for(npe, num_dim)
            .ok_or_else(|| cubit_err(format!("unsupported {num_dim}-D element with {npe} nodes")))?;
        if is_second_order(&et) {
            return Err(cubit_err(
                "second-order Cubit blocks are not supported (MFEM projects them through a \
                 quadratic nodal grid function); export a linear mesh instead",
            ));
        }
        elem_type_for_block.insert(block_id, et);
    }

    // BuildElementIDsForBlockID: element ids are contiguous from 1 across blocks.
    let mut element_ids_for_block: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    let mut block_id_for_element: BTreeMap<usize, i32> = BTreeMap::new();
    {
        let mut element_id = 1usize;
        for &block_id in &block_ids {
            let n = num_elements_for_block[&block_id];
            let mut ids = Vec::with_capacity(n);
            for _ in 0..n {
                ids.push(element_id);
                block_id_for_element.insert(element_id, block_id);
                element_id += 1;
            }
            element_ids_for_block.insert(block_id, ids);
        }
    }

    // ReadCubitElementBlocks: variable `connect{i}` per block.
    let mut node_ids_for_element: BTreeMap<usize, Vec<i32>> = BTreeMap::new();
    for (i, &block_id) in block_ids.iter().enumerate() {
        let et = elem_type_for_block[&block_id];
        let elem_ids = &element_ids_for_block[&block_id];
        let npe = et.nodes_per_element();
        let data = nc.read_var_i32(&format!("connect{}", i + 1))?;
        if data.len() != elem_ids.len() * npe {
            return Err(cubit_err(format!(
                "connect{} has {} entries, expected {}",
                i + 1,
                data.len(),
                elem_ids.len() * npe
            )));
        }
        for (ei, &element_id) in elem_ids.iter().enumerate() {
            node_ids_for_element.insert(
                element_id,
                data[ei * npe..(ei + 1) * npe].to_vec(),
            );
        }
    }

    // ReadCubitBoundaryIDs: variable `ss_prop1` (optional).
    let boundary_ids: Vec<i32> = if num_side_sets > 0 {
        nc.read_var_i32("ss_prop1")?
    } else {
        Vec::new()
    };

    // ReadCubitBoundaries: `num_side_ss{i}` dim, `elem_ss{i}` / `side_ss{i}` vars.
    let mut element_ids_for_boundary: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    let mut side_ids_for_boundary: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
    for (i, &boundary_id) in boundary_ids.iter().enumerate() {
        let n = nc.dimension(&format!("num_side_ss{}", i + 1))? as usize;
        let elems = nc.read_var_i32(&format!("elem_ss{}", i + 1))?;
        let sides = nc.read_var_i32(&format!("side_ss{}", i + 1))?;
        if elems.len() != n || sides.len() != n {
            return Err(cubit_err(format!("side set {boundary_id} data length mismatch")));
        }
        element_ids_for_boundary.insert(
            boundary_id,
            elems.into_iter().map(|e| e as usize).collect(),
        );
        side_ids_for_boundary.insert(boundary_id, sides);
    }

    // BuildBoundaryNodeIDs: nodes of each side via the Exodus side maps.
    // node_ids_for_boundary[boundary_id] = Vec<face node ids (1-based exodus)>.
    let mut node_ids_for_boundary: BTreeMap<i32, Vec<Vec<i32>>> = BTreeMap::new();
    for &boundary_id in &boundary_ids {
        let bdr_elems = &element_ids_for_boundary[&boundary_id];
        let bdr_sides = &side_ids_for_boundary[&boundary_id];
        let mut faces = Vec::with_capacity(bdr_elems.len());
        for (&element_global_id, &side) in bdr_elems.iter().zip(bdr_sides.iter()) {
            let block_id = block_id_for_element
                .get(&element_global_id)
                .ok_or_else(|| cubit_err("side set references unknown element"))?;
            let et = elem_type_for_block[block_id];
            let elem_nodes = &node_ids_for_element[&element_global_id];
            let face = side_map_nodes(et, side, elem_nodes)
                .ok_or_else(|| cubit_err(format!("invalid side id {side} for element type {et:?}")))?;
            faces.push(face);
        }
        node_ids_for_boundary.insert(boundary_id, faces);
    }

    // BuildUniqueVertexIDs: sorted unique 1-based node ids used as vertices.
    let mut unique_vertex_ids: Vec<i32> = Vec::with_capacity(num_nodes);
    for &block_id in &block_ids {
        let et = elem_type_for_block[&block_id];
        let nverts = num_vertices(et);
        for &element_id in &element_ids_for_block[&block_id] {
            for &n in &node_ids_for_element[&element_id][..nverts] {
                unique_vertex_ids.push(n);
            }
        }
    }
    unique_vertex_ids.sort_unstable();
    unique_vertex_ids.dedup();

    // BuildCubitToMFEMVertexMap: 1-based exodus id -> contiguous 1-based index.
    let mut cubit_to_mfem: BTreeMap<i32, usize> = BTreeMap::new();
    for (i, &id) in unique_vertex_ids.iter().enumerate() {
        cubit_to_mfem.insert(id, i + 1);
    }

    // ReadCubitNodeCoordinates.
    let coordx = nc.read_var_f64("coordx")?;
    let coordy = nc.read_var_f64("coordy")?;
    let coordz = if num_dim == 3 { nc.read_var_f64("coordz")? } else { Vec::new() };
    if coordx.len() < num_nodes || coordy.len() < num_nodes {
        return Err(cubit_err("coordinate arrays shorter than num_nodes"));
    }
    if num_dim == 3 && coordz.len() < num_nodes {
        return Err(cubit_err("coordz shorter than num_nodes"));
    }

    // BuildCubitVertices: coords via the original 1-based node ids.
    let mut coords = Vec::with_capacity(unique_vertex_ids.len() * num_dim);
    for &id in &unique_vertex_ids {
        let idx = (id - 1) as usize;
        coords.push(coordx[idx]);
        coords.push(coordy[idx]);
        if num_dim == 3 {
            coords.push(coordz[idx]);
        }
    }

    // BuildCubitElements: renumber vertices and emit elements block by block.
    let mut conn: Vec<NodeId> = Vec::with_capacity(num_elem * 8);
    let mut elem_tags: Vec<i32> = Vec::with_capacity(num_elem);
    let mut elem_types: Vec<ElementType> = Vec::with_capacity(num_elem);
    let mut elem_offsets: Vec<usize> = vec![0];
    for &block_id in &block_ids {
        let et = elem_type_for_block[&block_id];
        let nverts = num_vertices(et);
        for &element_id in &element_ids_for_block[&block_id] {
            for &n in &node_ids_for_element[&element_id][..nverts] {
                let mfem1 = cubit_to_mfem
                    .get(&n)
                    .ok_or_else(|| cubit_err("element references node outside mesh"))?;
                conn.push((mfem1 - 1) as NodeId);
            }
            elem_types.push(et);
            elem_offsets.push(conn.len());
            elem_tags.push(block_id); // attribute = block id (MFEM BuildElement)
        }
    }

    // BuildCubitBoundaries: face connectivity + side-set id tags.
    let mut face_conn: Vec<NodeId> = Vec::new();
    let mut face_tags: Vec<i32> = Vec::new();
    let mut face_types: Vec<ElementType> = Vec::new();
    let mut face_offsets: Vec<usize> = vec![0];
    for &boundary_id in &boundary_ids {
        let bdr_elems = &element_ids_for_boundary[&boundary_id];
        let bdr_sides = &side_ids_for_boundary[&boundary_id];
        let faces = &node_ids_for_boundary[&boundary_id];
        for (jelement, &_side) in bdr_sides.iter().enumerate() {
            let element_global_id = bdr_elems[jelement];
            let block_id = block_id_for_element[&element_global_id];
            let et = elem_type_for_block[&block_id];
            let face_ft = face_element_type(et);
            for &n in &faces[jelement] {
                let mfem1 = cubit_to_mfem
                    .get(&n)
                    .ok_or_else(|| cubit_err("side set references node outside mesh"))?;
                face_conn.push((mfem1 - 1) as NodeId);
            }
            face_tags.push(boundary_id);
            face_types.push(face_ft);
            face_offsets.push(face_conn.len());
        }
    }

    let mixed_elems = elem_types.iter().any(|&t| t != elem_types[0]);
    let face_type0 = face_types.first().copied().unwrap_or(match num_dim {
        2 => ElementType::Line2,
        _ => ElementType::Quad4,
    });
    let mixed_faces = face_types.iter().any(|&t| t != face_type0);

    let parts = MeshParts {
        coords,
        conn,
        elem_tags,
        elem_type: elem_types[0],
        face_conn,
        face_tags,
        face_type: face_type0,
        elem_types: if mixed_elems { Some(elem_types) } else { None },
        elem_offsets: if mixed_elems { Some(elem_offsets) } else { None },
        face_types: if mixed_faces { Some(face_types) } else { None },
        face_offsets: face_offsets,
    };

    if num_dim == 2 {
        Ok(CubitMesh::D2(Box::new(build_mesh::<2>(parts)?)))
    } else {
        Ok(CubitMesh::D3(Box::new(build_mesh::<3>(parts)?)))
    }
}

/// Deconstructed mesh fields handed to [`build_mesh`].
struct MeshParts {
    coords: Vec<f64>,
    conn: Vec<NodeId>,
    elem_tags: Vec<i32>,
    elem_type: ElementType,
    face_conn: Vec<NodeId>,
    face_tags: Vec<i32>,
    face_type: ElementType,
    elem_types: Option<Vec<ElementType>>,
    elem_offsets: Option<Vec<usize>>,
    face_types: Option<Vec<ElementType>>,
    face_offsets: Vec<usize>,
}

/// Assemble a [`Mesh`] from parsed Cubit data.
fn build_mesh<const D: usize>(parts: MeshParts) -> FemResult<Mesh<D>> {
    if parts.coords.len() % D != 0 {
        return Err(cubit_err("coordinate count is not a multiple of the dimension"));
    }
    Ok(Mesh {
        coords: parts.coords,
        conn: parts.conn,
        elem_tags: parts.elem_tags,
        elem_type: parts.elem_type,
        face_conn: parts.face_conn,
        face_tags: parts.face_tags,
        face_type: parts.face_type,
        elem_types: parts.elem_types,
        elem_offsets: parts.elem_offsets,
        face_types: parts.face_types,
        face_offsets: Some(parts.face_offsets),
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None,
        nc_vertex_view: None,
        vertex_parents: vec![],
    })
}

/// `CubitElement::GetElementType` for first/second order blocks.
fn element_type_for(num_nodes: usize, dimension: usize) -> Option<ElementType> {
    match dimension {
        2 => match num_nodes {
            3 => Some(ElementType::Tri3),
            6 => Some(ElementType::Tri6),
            4 => Some(ElementType::Quad4),
            8 => Some(ElementType::Quad8),
            9 => Some(ElementType::Quad9),
            _ => None,
        },
        3 => match num_nodes {
            4 => Some(ElementType::Tet4),
            10 => Some(ElementType::Tet10),
            8 => Some(ElementType::Hex8),
            20 => Some(ElementType::Hex20),
            27 => Some(ElementType::Hex27),
            6 => Some(ElementType::Prism6),
            15 => Some(ElementType::Prism15),
            18 => Some(ElementType::Prism18),
            5 => Some(ElementType::Pyramid5),
            13 | 14 => Some(ElementType::Pyramid13),
            _ => None,
        },
        _ => None,
    }
}

fn is_second_order(et: &ElementType) -> bool {
    matches!(
        et,
        ElementType::Tri6
            | ElementType::Quad8
            | ElementType::Quad9
            | ElementType::Tet10
            | ElementType::Hex20
            | ElementType::Hex27
            | ElementType::Prism15
            | ElementType::Prism18
            | ElementType::Pyramid13
    )
}

/// Number of linear vertices of a block type (`CubitElement::GetNumVertices`).
fn num_vertices(et: ElementType) -> usize {
    match et {
        ElementType::Tri3 | ElementType::Tri6 => 3,
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => 4,
        ElementType::Tet4 | ElementType::Tet10 => 4,
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => 8,
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => 6,
        ElementType::Pyramid5 | ElementType::Pyramid13 => 5,
        other => other.nodes_per_element(),
    }
}

/// Boundary element type for a volume element (`CubitElement::GetFaceType`).
fn face_element_type(et: ElementType) -> ElementType {
    match et {
        ElementType::Tri3 | ElementType::Tri6 => ElementType::Line2,
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => ElementType::Line2,
        ElementType::Tet4 | ElementType::Tet10 => ElementType::Tri3,
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => ElementType::Quad4,
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => ElementType::Quad4,
        ElementType::Pyramid5 | ElementType::Pyramid13 => ElementType::Tri3,
        other => other,
    }
}

/// Exodus II side maps (`cubit_side_map_*`, 1-based local node ids; 0 marks an
/// unused slot in tri/prism/pyramid faces).  Returns the global 1-based node
/// ids of the given side.
pub(crate) fn side_map_nodes(et: ElementType, side: i32, elem_nodes: &[i32]) -> Option<Vec<i32>> {
    let map: &[&[i32]] = match et {
        ElementType::Tri3 | ElementType::Tri6 => &[&[1, 2], &[2, 3], &[3, 1]],
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => {
            &[&[1, 2], &[2, 3], &[3, 4], &[4, 1]]
        }
        ElementType::Tet4 | ElementType::Tet10 => {
            &[&[1, 2, 4], &[2, 3, 4], &[1, 4, 3], &[1, 3, 2]]
        }
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => {
            &[
                &[1, 2, 6, 5],
                &[2, 3, 7, 6],
                &[3, 4, 8, 7],
                &[1, 5, 8, 4],
                &[1, 4, 3, 2],
                &[5, 6, 7, 8],
            ]
        }
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => {
            &[&[1, 2, 5, 4], &[2, 3, 6, 5], &[3, 1, 4, 6], &[1, 3, 2, 0], &[4, 5, 6, 0]]
        }
        ElementType::Pyramid5 | ElementType::Pyramid13 => {
            &[&[1, 2, 5, 0], &[2, 3, 5, 0], &[3, 4, 5, 0], &[1, 5, 4, 0], &[1, 4, 3, 2]]
        }
        _ => return None,
    };
    let row = map.get(side.checked_sub(1)? as usize)?;
    Some(
        row.iter()
            .filter(|&&i| i != 0)
            .map(|&i| elem_nodes[(i - 1) as usize])
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netcdf::NcOutput as NcWriter;

    /// Build a minimal 3-D Genesis file: one Hex8 block (id 1, a unit cube)
    /// plus one side set (id 2) on the top face (side 6 = {5,6,7,8}).
    fn cube_genesis() -> Vec<u8> {
        let mut w = NcWriter::new();
        let d_nodes = w.add_dim("num_nodes", 8);
        let _d_dim = w.add_dim("num_dim", 3);
        let _d_elem = w.add_dim("num_elem", 1);
        let d_blk = w.add_dim("num_el_blk", 1);
        let d_ss = w.add_dim("num_side_sets", 1);
        let d_npb = w.add_dim("num_el_in_blk1", 1);
        let d_npe = w.add_dim("num_nod_per_el1", 8);
        let d_nss = w.add_dim("num_side_ss1", 1);

        w.add_var_i32("eb_prop1", &[d_blk], &[1]);
        w.add_var_i32("connect1", &[d_npb, d_npe], &[1, 2, 3, 4, 5, 6, 7, 8]);
        w.add_var_i32("ss_prop1", &[d_ss], &[2]);
        w.add_var_i32("elem_ss1", &[d_nss], &[1]);
        w.add_var_i32("side_ss1", &[d_nss], &[6]);
        w.add_var_f64(
            "coordx",
            &[d_nodes],
            &[0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        );
        w.add_var_f64(
            "coordy",
            &[d_nodes],
            &[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        );
        w.add_var_f64(
            "coordz",
            &[d_nodes],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        );
        w.finish()
    }

    #[test]
    fn reads_hex8_cube_with_side_set() {
        let bytes = cube_genesis();
        let nc = NetCdfFile::from_bytes(bytes).expect("netcdf parse failed");
        let mesh = match read_cubit(&nc).expect("cubit parse failed") {
            CubitMesh::D3(m) => *m,
            CubitMesh::D2(_) => panic!("expected 3-D mesh"),
        };
        assert_eq!(mesh.n_nodes(), 8);
        assert_eq!(mesh.n_elems(), 1);
        assert_eq!(mesh.n_faces(), 1);
        assert_eq!(mesh.elem_type, ElementType::Hex8);
        // Element: block id 1 becomes the attribute; vertices keep file order.
        assert_eq!(mesh.elem_tags, vec![1]);
        assert_eq!(&mesh.conn[0..8], &[0, 1, 2, 3, 4, 5, 6, 7]);
        // Corner (1,1,1) is exodus node 7 -> 0-based 6.
        assert_eq!(mesh.coords_of(6), [1.0, 1.0, 1.0]);
        // Side set 2 on side 6 (top face): exodus nodes 5,6,7,8 -> 0-based 4..8.
        assert_eq!(mesh.face_tags, vec![2]);
        assert_eq!(mesh.face_type, ElementType::Quad4);
        assert_eq!(&mesh.face_conn[0..4], &[4, 5, 6, 7]);
    }

    #[test]
    fn reads_2d_quad_mesh() {
        let mut w = NcWriter::new();
        let d_nodes = w.add_dim("num_nodes", 4);
        let _d_dim = w.add_dim("num_dim", 2);
        let _d_elem = w.add_dim("num_elem", 1);
        let d_blk = w.add_dim("num_el_blk", 1);
        let d_ss = w.add_dim("num_side_sets", 1);
        let d_npb = w.add_dim("num_el_in_blk1", 1);
        let d_npe = w.add_dim("num_nod_per_el1", 4);
        let d_nss = w.add_dim("num_side_ss1", 1);
        w.add_var_i32("eb_prop1", &[d_blk], &[3]);
        w.add_var_i32("connect1", &[d_npb, d_npe], &[1, 2, 3, 4]);
        w.add_var_i32("ss_prop1", &[d_ss], &[5]);
        w.add_var_i32("elem_ss1", &[d_nss], &[1]);
        w.add_var_i32("side_ss1", &[d_nss], &[3]); // side 3 = {3,4}
        w.add_var_f64("coordx", &[d_nodes], &[0.0, 1.0, 1.0, 0.0]);
        w.add_var_f64("coordy", &[d_nodes], &[0.0, 0.0, 1.0, 1.0]);
        let bytes = w.finish();

        let nc = NetCdfFile::from_bytes(bytes).expect("netcdf parse failed");
        let mesh = match read_cubit(&nc).expect("cubit parse failed") {
            CubitMesh::D2(m) => *m,
            CubitMesh::D3(_) => panic!("expected 2-D mesh"),
        };
        assert_eq!(mesh.n_nodes(), 4);
        assert_eq!(mesh.n_elems(), 1);
        assert_eq!(mesh.elem_type, ElementType::Quad4);
        assert_eq!(mesh.elem_tags, vec![3]);
        // Boundary: side set 5 on side 3 → nodes 3,4 → 0-based [2,3], Line2.
        assert_eq!(mesh.n_faces(), 1);
        assert_eq!(mesh.face_tags, vec![5]);
        assert_eq!(mesh.face_type, ElementType::Line2);
        assert_eq!(&mesh.face_conn[0..2], &[2, 3]);
    }

    #[test]
    fn rejects_second_order_blocks() {
        let mut w = NcWriter::new();
        let d_nodes = w.add_dim("num_nodes", 10);
        w.add_dim("num_dim", 3);
        let _d_elem = w.add_dim("num_elem", 1);
        let d_blk = w.add_dim("num_el_blk", 1);
        let d_npb = w.add_dim("num_el_in_blk1", 1);
        let d_npe = w.add_dim("num_nod_per_el1", 10); // Tet10
        w.add_var_i32("eb_prop1", &[d_blk], &[1]);
        w.add_var_i32("connect1", &[d_npb, d_npe], &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        w.add_var_f64("coordx", &[d_nodes], &[0.0; 10]);
        w.add_var_f64("coordy", &[d_nodes], &[0.0; 10]);
        w.add_var_f64("coordz", &[d_nodes], &[0.0; 10]);
        let bytes = w.finish();

        let nc = NetCdfFile::from_bytes(bytes).expect("netcdf parse failed");
        let err = read_cubit(&nc).unwrap_err();
        assert!(
            err.to_string().contains("second-order"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn missing_coord_variable_errors() {
        let mut w = NcWriter::new();
        let d_nodes = w.add_dim("num_nodes", 8);
        w.add_dim("num_dim", 3);
        let _d_elem = w.add_dim("num_elem", 1);
        let d_blk = w.add_dim("num_el_blk", 1);
        let d_npb = w.add_dim("num_el_in_blk1", 1);
        let d_npe = w.add_dim("num_nod_per_el1", 8);
        w.add_var_i32("eb_prop1", &[d_blk], &[1]);
        w.add_var_i32("connect1", &[d_npb, d_npe], &[1, 2, 3, 4, 5, 6, 7, 8]);
        w.add_var_f64("coordx", &[d_nodes], &[0.0; 8]);
        w.add_var_f64("coordy", &[d_nodes], &[0.0; 8]);
        // coordz missing → error.
        let bytes = w.finish();
        let nc = NetCdfFile::from_bytes(bytes).expect("netcdf parse failed");
        let err = read_cubit(&nc).unwrap_err();
        assert!(err.to_string().contains("coordz"), "unexpected error: {err}");
    }
}
