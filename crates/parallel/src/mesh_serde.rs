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
//!
//! **Wire format v2** (`wire_format = 2`): optional tail after `global_elem_ids`
//! for mixed volume / mixed boundary topology (prism, pyramid, tet+hex, …):
//!
//! ```text
//! [flags : u32]  // bit0 = elem_offsets+elem_types present; bit1 = face_offsets+face_types
//! [elem_offsets : u32 × (n_elems+1)]  // if bit0
//! [elem_types   : u32 × n_elems]     // if bit0
//! [face_offsets : u32 × (n_faces+1)] // if bit1
//! [face_types   : u32 × n_faces]     // if bit1
//! ```

use fem_core::{ElemId, NodeId, Rank};
use fem_mesh::{ElementType, Mesh};

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
        ElementType::Quad9    => 13,
        ElementType::Prism15  => 14,
        ElementType::Prism18  => 15,
        ElementType::Pyramid13 => 16,
        ElementType::Hex27    => 17,
        ElementType::Polygon  => 18,
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
        13 => Ok(ElementType::Quad9),
        14 => Ok(ElementType::Prism15),
        15 => Ok(ElementType::Prism18),
        16 => Ok(ElementType::Pyramid13),
        17 => Ok(ElementType::Hex27),
        18 => Ok(ElementType::Polygon),
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
    n_local_elems:  u32,
    /// Number of owned elements (subset of n_local_elems). If 0, all local elements are owned.
    n_owned_elems:  u32,
    /// `0` = legacy payload only; `2` = mixed-topology extension follows (see module doc).
    wire_format:    u32,
    /// `1` = local node ids are the global ids (identity partition mode).
    node_id_identity: u32,
    _pad:           u32,
}

const HEADER_SIZE: usize = std::mem::size_of::<SubMeshHeader>();
const _: () = assert!(HEADER_SIZE == 60);

// ── Encode ───────────────────────────────────────────────────────────────────

/// Encode a sub-mesh and its partition descriptor into a flat byte buffer.
pub fn encode_submesh<const D: usize>(
    mesh:      &Mesh<D>,
    partition: &MeshPartition,
) -> Vec<u8> {
    let n_nodes = mesh.n_nodes();
    let n_elems = mesh.n_elems();
    let n_faces = mesh.n_faces();

    let mut mix_flags: u32 = 0;
    let mut ext_tail: usize = 0;
    if let Some(elem_offsets) = &mesh.elem_offsets {
        debug_assert_eq!(elem_offsets.len(), n_elems + 1);
        debug_assert_eq!(mesh.elem_types.as_ref().map(|t| t.len()), Some(n_elems));
        mix_flags |= 1;
        ext_tail += (n_elems + 1) * 4 + n_elems * 4;
    }
    if let Some(face_offsets) = &mesh.face_offsets {
        debug_assert_eq!(face_offsets.len(), n_faces + 1);
        debug_assert_eq!(mesh.face_types.as_ref().map(|t| t.len()), Some(n_faces));
        mix_flags |= 2;
        ext_tail += (n_faces + 1) * 4 + n_faces * 4;
    }
    if let Some(g) = &mesh.geometry {
        // Per-element (possibly geometrically periodic) geometry table:
        // order + nodes_per_elem + n_nodes + conn (u32[]) + coords (f64[]).
        mix_flags |= 4;
        ext_tail += 4 + 4 + 4 + g.conn.len() * 4 + g.coords.len() * 8;
    }
    let wire_format: u32 = if mix_flags != 0 { 2 } else { 0 };
    if mix_flags != 0 {
        ext_tail += 4; // mix_flags word
    }

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
        n_local_elems: (partition.n_owned_elems + partition.n_ghost_elems) as u32,
        n_owned_elems: partition.n_owned_elems as u32,
        wire_format,
        node_id_identity: partition.node_id_identity as u32,
        _pad: 0,
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
        + partition.global_elem_ids.len() * 4
        + partition.elem_owner.len() * 4
        + ext_tail;

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

    // partition: elem_owner (i32) — must be transmitted so that non-root
    // ranks can tell owned elements from ghosts (decode used to default
    // ghosts to rank 1, breaking rank>0 owned-element queries).
    buf.extend_from_slice(bytemuck::cast_slice::<Rank, u8>(&partition.elem_owner));

    if wire_format == 2 {
        buf.extend_from_slice(&mix_flags.to_le_bytes());
        if mix_flags & 1 != 0 {
            let eo = mesh.elem_offsets.as_ref().unwrap();
            let et = mesh.elem_types.as_ref().unwrap();
            for &x in eo {
                buf.extend_from_slice(&(x as u32).to_le_bytes());
            }
            for t in et {
                buf.extend_from_slice(&element_type_to_u32(*t).to_le_bytes());
            }
        }
        if mix_flags & 2 != 0 {
            let fo = mesh.face_offsets.as_ref().unwrap();
            let ft = mesh.face_types.as_ref().unwrap();
            for &x in fo {
                buf.extend_from_slice(&(x as u32).to_le_bytes());
            }
            for t in ft {
                buf.extend_from_slice(&element_type_to_u32(*t).to_le_bytes());
            }
        }
        if mix_flags & 4 != 0 {
            let g = mesh.geometry.as_ref().unwrap();
            buf.extend_from_slice(&(g.order as u32).to_le_bytes());
            buf.extend_from_slice(&(g.nodes_per_elem as u32).to_le_bytes());
            buf.extend_from_slice(&(g.n_nodes as u32).to_le_bytes());
            for &x in &g.conn {
                buf.extend_from_slice(&x.to_le_bytes());
            }
            for &x in &g.coords {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
    }

    debug_assert_eq!(buf.len(), total);
    buf
}

// ── Decode ───────────────────────────────────────────────────────────────────

/// Decode a sub-mesh and partition descriptor from a byte buffer produced by
/// [`encode_submesh`].
pub fn decode_submesh<const D: usize>(buf: &[u8]) -> Result<(Mesh<D>, MeshPartition), String> {
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
    let n_owned_elems = header.n_owned_elems as usize;
    let n_ghost_elems = if n_owned_elems > 0 {
        n_local_elems - n_owned_elems
    } else {
        0 // backward compat: all local elements are owned
    };
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
    let elem_owner = read_i32_vec(buf, &mut offset, n_local_elems)?;

    let mut mesh = Mesh::uniform(
        coords, conn, elem_tags, elem_type,
        face_conn, face_tags, face_type,
    );

    match header.wire_format {
        0 => {}
        2 => {
            let mix_flags = read_u32_at(buf, &mut offset)?;
            if mix_flags & 1 != 0 {
                let mut eo = Vec::with_capacity(n_elems + 1);
                for _ in 0..=n_elems {
                    eo.push(read_u32_at(buf, &mut offset)? as usize);
                }
                let mut et = Vec::with_capacity(n_elems);
                for _ in 0..n_elems {
                    et.push(u32_to_element_type(read_u32_at(buf, &mut offset)?)?);
                }
                mesh.elem_offsets = Some(eo);
                mesh.elem_types = Some(et);
            }
            if mix_flags & 2 != 0 {
                let mut fo = Vec::with_capacity(n_faces + 1);
                for _ in 0..=n_faces {
                    fo.push(read_u32_at(buf, &mut offset)? as usize);
                }
                let mut ft = Vec::with_capacity(n_faces);
                for _ in 0..n_faces {
                    ft.push(u32_to_element_type(read_u32_at(buf, &mut offset)?)?);
                }
                mesh.face_offsets = Some(fo);
                mesh.face_types = Some(ft);
            }
            if mix_flags & 4 != 0 {
                let order = read_u32_at(buf, &mut offset)? as u8;
                let nodes_per_elem = read_u32_at(buf, &mut offset)? as usize;
                let n_nodes = read_u32_at(buf, &mut offset)? as usize;
                // `conn` holds one `nodes_per_elem`-row per LOCAL element
                // (n_local_elems × npe), NOT n_nodes × npe — n_nodes is the
                // deduplicated geometry-node count (coords.len() / D).
                let conn = read_u32_vec(buf, &mut offset, n_local_elems * nodes_per_elem)?;
                let coords = read_f64_vec(buf, &mut offset, n_nodes * D)?;
                mesh.geometry = Some(fem_mesh::simplex::GeometryData {
                    order,
                    conn,
                    nodes_per_elem,
                    coords,
                    n_nodes,
                });
            }
        }
        other => {
            return Err(format!("unsupported submesh wire_format: {other}"));
        }
    }

    if offset != buf.len() {
        return Err(format!(
            "submesh buffer size mismatch: consumed {offset} bytes, buffer length {}",
            buf.len()
        ));
    }

    let mut partition = MeshPartition::from_raw(
        n_owned, n_ghost,
        n_owned_elems, n_ghost_elems,
        global_node_ids, node_owner,
        global_elem_ids, elem_owner,
    );
    partition.node_id_identity = header.node_id_identity != 0;

    Ok((mesh, partition))
}

// ── Helpers ──────────────────────────────────────────────────────────────────

#[inline]
fn read_u32_at(buf: &[u8], offset: &mut usize) -> Result<u32, String> {
    let end = *offset + 4;
    if end > buf.len() {
        return Err(format!("buffer underflow at u32 read: need {end}, have {}", buf.len()));
    }
    let v = u32::from_le_bytes(buf[*offset..end].try_into().unwrap());
    *offset = end;
    Ok(v)
}

fn read_f64_vec(buf: &[u8], offset: &mut usize, count: usize) -> Result<Vec<f64>, String> {
    let byte_len = count * 8;
    let end = *offset + byte_len;
    if end > buf.len() {
        return Err(format!("buffer underflow at f64 read: need {end}, have {}", buf.len()));
    }
    // Manual parse — `bytemuck::cast_slice` requires 8-byte alignment of the
    // slice start, which is not guaranteed when the header size isn't a
    // multiple of 8.
    let mut v = Vec::with_capacity(count);
    for i in 0..count {
        let start = *offset + i * 8;
        v.push(f64::from_le_bytes(buf[start..start + 8].try_into().unwrap()));
    }
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
    use fem_mesh::{ElementType, Mesh};

    #[test]
    fn round_trip_serial_mesh() {
        let mesh = Mesh::<2>::unit_square_tri(4);
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
        assert_eq!(mesh.elem_offsets, mesh2.elem_offsets);
        assert_eq!(mesh.elem_types, mesh2.elem_types);
        assert_eq!(mesh.face_offsets, mesh2.face_offsets);
        assert_eq!(mesh.face_types, mesh2.face_types);

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

        let ghost_elems: Vec<(ElemId, Rank)> = Vec::new();
        let partition = MeshPartition::from_partitioner(
            &owned_global,
            &ghost_global,
            &local_elems,
            &ghost_elems,
            1,
        );

        // Build a minimal local mesh matching the partition.
        let n_local_nodes = owned_global.len() + ghost_global.len(); // 7
        let mesh = Mesh::<2>::uniform(
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
        let mesh = Mesh::<2>::unit_square_tri(2);
        let partition = MeshPartition::new_serial(mesh.n_nodes(), mesh.n_elems());
        let buf = encode_submesh(&mesh, &partition);
        let result = decode_submesh::<3>(&buf);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("dimension mismatch"));
    }

    #[test]
    fn round_trip_per_element_geometry() {
        // Geometrically periodic mesh (periodic-hexagon-style): each Quad4
        // element owns an independent 4-node geometry table whose coords are
        // NOT the folded vertex coords.  Encode/decode must preserve it.
        let coords = vec![
            -0.5, -0.866, 0.0, -0.866, 0.25, -0.433, -0.25, -0.433, // elem 0
            0.25, -0.433, 0.75, -0.433, 1.0, 0.0, 0.5, 0.0, // elem 1
        ];
        let conn = vec![0u32, 1, 2, 3, 1, 4, 5, 2];
        let elem_tags = vec![1i32, 1];
        let mut mesh = Mesh::<2>::uniform(
            vec![0.0; 5 * 2], // folded vertex coords (unused by geometry path)
            conn,
            elem_tags,
            ElementType::Quad4,
            Vec::new(),
            Vec::new(),
            ElementType::Line2,
        );
        mesh.geometry = Some(fem_mesh::simplex::GeometryData {
            order: 1,
            conn: vec![0u32, 1, 2, 3, 4, 5, 6, 7],
            nodes_per_elem: 4,
            coords,
            n_nodes: 8,
        });

        let partition = MeshPartition::new_serial(mesh.n_nodes(), mesh.n_elems());
        let buf = encode_submesh(&mesh, &partition);
        let (mesh2, _) = decode_submesh::<2>(&buf).expect("decode failed");

        let g = mesh.geometry.as_ref().expect("source geometry");
        let g2 = mesh2.geometry.as_ref().expect("decoded geometry");
        assert_eq!(g.order, g2.order);
        assert_eq!(g.nodes_per_elem, g2.nodes_per_elem);
        assert_eq!(g.n_nodes, g2.n_nodes);
        assert_eq!(g.conn, g2.conn);
        assert_eq!(g.coords, g2.coords);
    }

    #[test]
    fn empty_buffer_rejected() {
        let result = decode_submesh::<2>(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn round_trip_3d_mesh() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let partition = MeshPartition::new_serial(mesh.n_nodes(), mesh.n_elems());

        let buf = encode_submesh(&mesh, &partition);
        let (mesh2, part2) = decode_submesh::<3>(&buf).expect("decode failed");

        assert_eq!(mesh.coords, mesh2.coords);
        assert_eq!(mesh.conn, mesh2.conn);
        assert_eq!(mesh.elem_type, mesh2.elem_type);
        assert_eq!(partition.n_owned_nodes, part2.n_owned_nodes);
    }

    /// Single `Prism6` with mixed Tri3 + Quad4 boundary (cylinder-like extrusion).
    fn unit_prism_mixed_boundary() -> Mesh<3> {
        let coords: Vec<f64> = vec![
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
        ];
        let conn = vec![0u32, 1, 2, 3, 4, 5];
        let elem_tags = vec![1i32];
        let face_conn: Vec<u32> = vec![
            0, 1, 2,
            3, 4, 5,
            0, 1, 4, 3,
            1, 2, 5, 4,
            2, 0, 3, 5,
        ];
        let face_tags = vec![1i32, 2, 3, 3, 3];
        let face_types = vec![
            ElementType::Tri3,
            ElementType::Tri3,
            ElementType::Quad4,
            ElementType::Quad4,
            ElementType::Quad4,
        ];
        let face_offsets = vec![0usize, 3, 6, 10, 14, 18];
        let mut m = Mesh::uniform(
            coords,
            conn,
            elem_tags,
            ElementType::Prism6,
            face_conn,
            face_tags,
            ElementType::Tri3,
        );
        m.face_types = Some(face_types);
        m.face_offsets = Some(face_offsets);
        m
    }

    #[test]
    fn round_trip_prism_mixed_boundary() {
        let mesh = unit_prism_mixed_boundary();
        mesh.check().expect("fixture mesh");
        let partition = MeshPartition::new_serial(mesh.n_nodes(), mesh.n_elems());

        let buf = encode_submesh(&mesh, &partition);
        let (mesh2, _) = decode_submesh::<3>(&buf).expect("decode failed");

        assert_eq!(mesh.n_faces(), 5, "fixture: 2 triangles + 3 quads");
        assert_eq!(mesh2.n_faces(), mesh.n_faces());
        assert_eq!(mesh2.face_offsets, mesh.face_offsets);
        assert_eq!(mesh2.face_types, mesh.face_types);
        assert_eq!(mesh.face_conn, mesh2.face_conn);
        mesh2.check().expect("decoded mesh");
    }
}
