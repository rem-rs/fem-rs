//! Binary serialisation of sub-mesh + partition for streaming distribution.
//!
//! Uses a fixed-layout header followed by flat arrays.  No serde dependency
//! needed — matches the byte-level transport used by [`CommBackend`].
//!
//! ## Wire format
//!
//! ```text
//! [SubMeshHeader : 56 bytes, repr(C)]
//! [coords        : f64 × n_nodes × dim]
//! [conn          : u32 × conn_len]
//! [elem_tags     : i32 × n_elems]
//! [face_conn     : u32 × face_conn_len]
//! [face_tags     : i32 × n_faces]
//! [global_node_ids : u32 × (n_owned + n_ghost)]
//! [node_owner    : i32 × (n_owned + n_ghost)]
//! [global_elem_ids : u32 × n_local_elems]
//! ```

use fem_core::{ElemId, NodeId, Rank};
use fem_mesh::{ElementType, SimplexMesh};

use crate::MeshPartition;

// ── ElementType ↔ u32 ────────────────────────────────────────────────────────

fn element_type_to_u32(et: ElementType) -> u32 {
    match et {
        ElementType::Point1   =>  0,
        ElementType::Line2    =>  1,
        ElementType::Line3    =>  2,
        ElementType::Tri3     =>  3,
        ElementType::Tri6     =>  4,
        ElementType::Quad4    =>  5,
        ElementType::Quad8    =>  6,
        ElementType::Tet4     =>  7,
        ElementType::Tet10    =>  8,
        ElementType::Hex8     =>  9,
        ElementType::Hex20    => 10,
        ElementType::Prism6   => 11,
        ElementType::Pyramid5 => 12,
    }
}

fn u32_to_element_type(v: u32) -> Result<ElementType, String> {
    match v {
         0 => Ok(ElementType::Point1),
         1 => Ok(ElementType::Line2),
         2 => Ok(ElementType::Line3),
         3 => Ok(ElementType::Tri3),
         4 => Ok(ElementType::Tri6),
         5 => Ok(ElementType::Quad4),
         6 => Ok(ElementType::Quad8),
         7 => Ok(ElementType::Tet4),
         8 => Ok(ElementType::Tet10),
         9 => Ok(ElementType::Hex8),
        10 => Ok(ElementType::Hex20),
        11 => Ok(ElementType::Prism6),
        12 => Ok(ElementType::Pyramid5),
         _ => Err(format!("unknown ElementType discriminant: {v}")),
    }
}

// ── Header ───────────────────────────────────────────────────────────────────

/// Fixed-size header: 14 × u32 = 56 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct SubMeshHeader {
    dim:           u32,
    n_nodes:       u32,
    n_elems:       u32,
    n_faces:       u32,
    elem_type:     u32,
    face_type:     u32,
    conn_len:      u32,
    face_conn_len: u32,
    n_owned_nodes: u32,
    n_ghost_nodes: u32,
    n_local_elems: u32,
    /// Reserved for future use / alignment padding.
    _pad0:         u32,
    _pad1:         u32,
    _pad2:         u32,
}

const HEADER_SIZE: usize = std::mem::size_of::<SubMeshHeader>();
const _: () = assert!(HEADER_SIZE == 56);

// ── Encode ───────────────────────────────────────────────────────────────────

/// Encode a sub-mesh and its partition descriptor into a flat byte buffer.
pub fn encode_submesh<const D: usize>(
    mesh:      &SimplexMesh<D>,
    partition: &MeshPartition,
) -> Vec<u8> {
    let n_nodes = mesh.n_nodes();
    let n_elems = mesh.n_elems();
    let n_faces = mesh.n_faces();

    let header = SubMeshHeader {
        dim:           D as u32,
        n_nodes:       n_nodes as u32,
        n_elems:       n_elems as u32,
        n_faces:       n_faces as u32,
        elem_type:     element_type_to_u32(mesh.elem_type),
        face_type:     element_type_to_u32(mesh.face_type),
        conn_len:      mesh.conn.len() as u32,
        face_conn_len: mesh.face_conn.len() as u32,
        n_owned_nodes: partition.n_owned_nodes as u32,
        n_ghost_nodes: partition.n_ghost_nodes as u32,
        n_local_elems: partition.n_local_elems as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    // Pre-compute total size.
    let total = HEADER_SIZE
        + mesh.coords.len() * 8   // f64
        + mesh.conn.len() * 4     // u32
        + mesh.elem_tags.len() * 4 // i32
        + mesh.face_conn.len() * 4
        + mesh.face_tags.len() * 4
        + partition.global_node_ids.len() * 4
        + partition.node_owner.len() * 4
        + partition.global_elem_ids.len() * 4;

    let mut buf = Vec::with_capacity(total);

    // Header (safe: all fields are POD-like primitives).
    buf.extend_from_slice(unsafe {
        std::slice::from_raw_parts(
            &header as *const SubMeshHeader as *const u8,
            HEADER_SIZE,
        )
    });

    // coords: f64[]
    buf.extend_from_slice(bytemuck::cast_slice::<f64, u8>(&mesh.coords));

    // conn: u32[]
    buf.extend_from_slice(bytemuck::cast_slice::<u32, u8>(&mesh.conn));

    // elem_tags: i32[]
    buf.extend_from_slice(bytemuck::cast_slice::<i32, u8>(&mesh.elem_tags));

    // face_conn: u32[]
    buf.extend_from_slice(bytemuck::cast_slice::<u32, u8>(&mesh.face_conn));

    // face_tags: i32[]
    buf.extend_from_slice(bytemuck::cast_slice::<i32, u8>(&mesh.face_tags));

    // partition: global_node_ids
    buf.extend_from_slice(bytemuck::cast_slice::<NodeId, u8>(&partition.global_node_ids));

    // partition: node_owner (i32)
    buf.extend_from_slice(bytemuck::cast_slice::<Rank, u8>(&partition.node_owner));

    // partition: global_elem_ids
    buf.extend_from_slice(bytemuck::cast_slice::<ElemId, u8>(&partition.global_elem_ids));

    debug_assert_eq!(buf.len(), total);
    buf
}

// ── Decode ───────────────────────────────────────────────────────────────────

/// Decode a sub-mesh and partition descriptor from a byte buffer produced by
/// [`encode_submesh`].
pub fn decode_submesh<const D: usize>(buf: &[u8]) -> Result<(SimplexMesh<D>, MeshPartition), String> {
    if buf.len() < HEADER_SIZE {
        return Err(format!("buffer too short for header: {} < {HEADER_SIZE}", buf.len()));
    }

    // Read header.
    let header: SubMeshHeader = unsafe {
        std::ptr::read_unaligned(buf.as_ptr() as *const SubMeshHeader)
    };

    if header.dim != D as u32 {
        return Err(format!("dimension mismatch: header.dim={} but D={D}", header.dim));
    }

    let elem_type = u32_to_element_type(header.elem_type)?;
    let face_type = u32_to_element_type(header.face_type)?;

    let n_nodes      = header.n_nodes as usize;
    let n_elems      = header.n_elems as usize;
    let n_faces      = header.n_faces as usize;
    let conn_len     = header.conn_len as usize;
    let face_conn_len = header.face_conn_len as usize;
    let n_owned      = header.n_owned_nodes as usize;
    let n_ghost      = header.n_ghost_nodes as usize;
    let n_local_elems = header.n_local_elems as usize;
    let total_part_nodes = n_owned + n_ghost;

    // Read arrays sequentially from the buffer.
    let mut offset = HEADER_SIZE;

    let coords = read_f64_vec(buf, &mut offset, n_nodes * D)?;
    let conn = read_u32_vec(buf, &mut offset, conn_len)?;
    let elem_tags = read_i32_vec(buf, &mut offset, n_elems)?;
    let face_conn = read_u32_vec(buf, &mut offset, face_conn_len)?;
    let face_tags = read_i32_vec(buf, &mut offset, n_faces)?;
    let global_node_ids = read_u32_vec(buf, &mut offset, total_part_nodes)?;
    let node_owner = read_i32_vec(buf, &mut offset, total_part_nodes)?;
    let global_elem_ids = read_u32_vec(buf, &mut offset, n_local_elems)?;

    let mesh = SimplexMesh::uniform(
        coords, conn, elem_tags, elem_type,
        face_conn, face_tags, face_type,
    );

    let partition = MeshPartition::from_raw(
        n_owned,
        n_ghost,
        global_node_ids,
        node_owner,
        global_elem_ids,
    );

    Ok((mesh, partition))
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn read_f64_vec(buf: &[u8], offset: &mut usize, count: usize) -> Result<Vec<f64>, String> {
    let byte_len = count * 8;
    let end = *offset + byte_len;
    if end > buf.len() {
        return Err(format!("buffer underflow at f64 read: need {end}, have {}", buf.len()));
    }
    let slice: &[f64] = bytemuck::cast_slice(&buf[*offset..end]);
    let v = slice.to_vec();
    *offset = end;
    Ok(v)
}

fn read_u32_vec(buf: &[u8], offset: &mut usize, count: usize) -> Result<Vec<u32>, String> {
    let byte_len = count * 4;
    let end = *offset + byte_len;
    if end > buf.len() {
        return Err(format!("buffer underflow at u32 read: need {end}, have {}", buf.len()));
    }
    let slice: &[u32] = bytemuck::cast_slice(&buf[*offset..end]);
    let v = slice.to_vec();
    *offset = end;
    Ok(v)
}

fn read_i32_vec(buf: &[u8], offset: &mut usize, count: usize) -> Result<Vec<i32>, String> {
    let byte_len = count * 4;
    let end = *offset + byte_len;
    if end > buf.len() {
        return Err(format!("buffer underflow at i32 read: need {end}, have {}", buf.len()));
    }
    let slice: &[i32] = bytemuck::cast_slice(&buf[*offset..end]);
    let v = slice.to_vec();
    *offset = end;
    Ok(v)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn round_trip_serial_mesh() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let partition = MeshPartition::new_serial(mesh.n_nodes(), mesh.n_elems());

        let buf = encode_submesh(&mesh, &partition);
        let (mesh2, part2) = decode_submesh::<2>(&buf).expect("decode failed");

        assert_eq!(mesh.coords, mesh2.coords);
        assert_eq!(mesh.conn, mesh2.conn);
        assert_eq!(mesh.elem_tags, mesh2.elem_tags);
        assert_eq!(mesh.elem_type, mesh2.elem_type);
        assert_eq!(mesh.face_conn, mesh2.face_conn);
        assert_eq!(mesh.face_tags, mesh2.face_tags);
        assert_eq!(mesh.face_type, mesh2.face_type);

        assert_eq!(partition.n_owned_nodes, part2.n_owned_nodes);
        assert_eq!(partition.n_ghost_nodes, part2.n_ghost_nodes);
        assert_eq!(partition.global_node_ids, part2.global_node_ids);
        assert_eq!(partition.node_owner, part2.node_owner);
        assert_eq!(partition.global_elem_ids, part2.global_elem_ids);
    }

    #[test]
    fn round_trip_partitioned_mesh() {
        // Simulate a partition where rank 1 has some ghost nodes.
        let owned_global: Vec<NodeId> = vec![3, 4, 5, 6];
        let ghost_global: Vec<(NodeId, Rank)> = vec![(0, 0), (1, 0), (2, 0)];
        let local_elems: Vec<ElemId> = vec![2, 3, 4];

        let partition = MeshPartition::from_partitioner(
            &owned_global,
            &ghost_global,
            &local_elems,
            1,
        );

        // Build a minimal local mesh matching the partition.
        let n_local_nodes = owned_global.len() + ghost_global.len(); // 7
        let mesh = SimplexMesh::<2>::uniform(
            vec![0.0; n_local_nodes * 2],
            vec![0, 1, 2,  3, 4, 5,  4, 5, 6],
            vec![1, 1, 1],
            ElementType::Tri3,
            vec![0, 1],
            vec![1],
            ElementType::Line2,
        );

        let buf = encode_submesh(&mesh, &partition);
        let (mesh2, part2) = decode_submesh::<2>(&buf).expect("decode failed");

        assert_eq!(mesh.coords, mesh2.coords);
        assert_eq!(mesh.conn, mesh2.conn);
        assert_eq!(partition.n_owned_nodes, part2.n_owned_nodes);
        assert_eq!(partition.n_ghost_nodes, part2.n_ghost_nodes);
        assert_eq!(partition.global_node_ids, part2.global_node_ids);
        assert_eq!(partition.node_owner, part2.node_owner);
        assert_eq!(partition.global_elem_ids, part2.global_elem_ids);
    }

    #[test]
    fn dimension_mismatch_detected() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let partition = MeshPartition::new_serial(mesh.n_nodes(), mesh.n_elems());
        let buf = encode_submesh(&mesh, &partition);
        let result = decode_submesh::<3>(&buf);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("dimension mismatch"));
    }

    #[test]
    fn empty_buffer_rejected() {
        let result = decode_submesh::<2>(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn round_trip_3d_mesh() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let partition = MeshPartition::new_serial(mesh.n_nodes(), mesh.n_elems());

        let buf = encode_submesh(&mesh, &partition);
        let (mesh2, part2) = decode_submesh::<3>(&buf).expect("decode failed");

        assert_eq!(mesh.coords, mesh2.coords);
        assert_eq!(mesh.conn, mesh2.conn);
        assert_eq!(mesh.elem_type, mesh2.elem_type);
        assert_eq!(partition.n_owned_nodes, part2.n_owned_nodes);
    }
}
