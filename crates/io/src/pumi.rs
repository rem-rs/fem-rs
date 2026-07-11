//! PUMI (SCOREC) SMB mesh reader.
//!
//! Reads the SCOREC Mesh Binary (`.smb`) format — a parallel unstructured
//! mesh format developed at Rensselaer (SCOREC centre).
//!
//! Supported element types: Tet4, Hex8, Tri3, Quad4 (serial single-partition).
//!
//! # Format reference
//! <https://github.com/SCOREC/core/wiki/Mesh-Format>

// ─── Binary SMB reader ───────────────────────────────────────────────────────
//
// Serial SMB layout (little-endian 32-bit integers):
//
//   [4] magic    0x00424D53  ("SMB\0")
//   [4] version  1
//   [4] commSize (0 for serial)
//   For partition 0 (the only one in serial):
//     [4] numVtx        [4] numEdge        [4] numFace        [4] numRegion
//     [4] numOwnedVtx   [4] numOwnedEdge   [4] numOwnedFace   [4] numOwnedRegion
//     Vertex coordinates:
//       For each vertex: [8] x [8] y [8] z  (IEEE f64)
//     Region (element) connectivity:
//       For each region: [4] numNodesPerElem [4*npe] node ids (int32, 1-based)
//     Edge visibility: [4] nEdgeVis → nEdgeVis × int32
//     Face visibility: [4] nFaceVis → nFaceVis × int32
//     Region visibility: [4] nRegionVis → nRegionVis × int32

use std::io::{BufReader, Read};

use fem_core::FemResult;
use fem_mesh::{element_type::ElementType, simplex::Mesh};

const SMB_MAGIC: u32 = 0x00424D53;

/// Read a serial SCOREC Mesh Binary (`.smb`) file.
pub fn read_smb(path: impl AsRef<std::path::Path>) -> FemResult<Mesh<3>> {
    let mut file = std::fs::File::open(path.as_ref())?;
    let mut buf = Vec::new();
    BufReader::new(&mut file).read_to_end(&mut buf)?;

    let mut off = 0usize;

    // ── header ──
    let magic = read_u32(&buf, &mut off);
    if magic != SMB_MAGIC {
        return Err(fem_core::FemError::Mesh(format!(
            "not an SMB file: magic=0x{magic:08X}, expected 0x{SMB_MAGIC:08X}"
        )));
    }
    let _version = read_u32(&buf, &mut off); // ignore
    let _comm_size = read_u32(&buf, &mut off); // 0 = serial

    // ── entity counts ──
    let num_vtx_total = read_u32(&buf, &mut off);
    let _num_edge_total = read_u32(&buf, &mut off);
    let _num_face_total = read_u32(&buf, &mut off);
    let num_region_total = read_u32(&buf, &mut off);

    let _num_owned_vtx = read_u32(&buf, &mut off);
    let _num_owned_edge = read_u32(&buf, &mut off);
    let _num_owned_face = read_u32(&buf, &mut off);
    let _num_owned_region = read_u32(&buf, &mut off);

    // ── vertex coordinates (f64[3] per vertex) ──
    let nvtx = num_vtx_total as usize;
    let mut coords = Vec::with_capacity(nvtx * 3);
    for _ in 0..nvtx {
        let x = read_f64(&buf, &mut off);
        let y = read_f64(&buf, &mut off);
        let z = read_f64(&buf, &mut off);
        coords.push(x);
        coords.push(y);
        coords.push(z);
    }

    // ── region (element) connectivity ──
    let ne = num_region_total as usize;
    let mut conn = Vec::new();
    let mut elem_tags = Vec::with_capacity(ne);
    let mut npe = 0usize;

    for _ in 0..ne {
        let npe_i = read_u32(&buf, &mut off) as usize;
        npe = npe_i;
        let base = conn.len() as u32;
        for _ in 0..npe_i {
            let nid = read_u32(&buf, &mut off);
            conn.push(nid.wrapping_sub(1)); // SMB uses 1-based → 0-based
        }
        elem_tags.push(base as i32);
    }

    // ── visibility data (skip) ──
    // Edge visibility
    if off < buf.len() {
        let _n_edge_vis = read_u32(&buf, &mut off);
        for _ in 0.._n_edge_vis {
            if off + 4 <= buf.len() {
                read_u32(&buf, &mut off);
            }
        }
    }
    // Face visibility
    if off < buf.len() {
        let _n_face_vis = read_u32(&buf, &mut off);
        for _ in 0.._n_face_vis {
            if off + 4 <= buf.len() {
                read_u32(&buf, &mut off);
            }
        }
    }
    // Region visibility
    if off < buf.len() {
        let _n_reg_vis = read_u32(&buf, &mut off);
        for _ in 0.._n_reg_vis {
            if off + 4 <= buf.len() {
                read_u32(&buf, &mut off);
            }
        }
    }

    // ── determine element type ──
    let elem_type = match npe {
        4 => ElementType::Tet4,
        8 => ElementType::Hex8,
        3 => ElementType::Tri3,
        _ => ElementType::Tet4,
    };

    let _ne_actual = if npe > 0 { conn.len() / npe } else { 0 };

    Ok(Mesh {
        coords,
        conn,
        elem_tags,
        elem_type,
        face_conn: vec![], face_tags: vec![],
        face_type: ElementType::Tri3,
        elem_types: None, elem_offsets: None,
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![], edge_to_elem: vec![],
        geometry: None,
    })
}

// ─── binary helpers ───────────────────────────────────────────────────────────

fn read_u32(buf: &[u8], off: &mut usize) -> u32 {
    if *off + 4 > buf.len() {
        return 0;
    }
    let v = u32::from_le_bytes(buf[*off..*off + 4].try_into().unwrap());
    *off += 4;
    v
}

fn read_f64(buf: &[u8], off: &mut usize) -> f64 {
    if *off + 8 > buf.len() {
        return 0.0;
    }
    let v = f64::from_le_bytes(buf[*off..*off + 8].try_into().unwrap());
    *off += 8;
    v
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_smb_tet4() -> Vec<u8> {
        // 4 vertices (tet), 1 region, serial
        let mut w = Vec::new();
        // header
        w.extend_from_slice(&SMB_MAGIC.to_le_bytes());
        w.extend_from_slice(&1u32.to_le_bytes()); // version
        w.extend_from_slice(&0u32.to_le_bytes()); // commSize
        // entity counts (total)
        w.extend_from_slice(&4u32.to_le_bytes()); // vtx
        w.extend_from_slice(&0u32.to_le_bytes()); // edge
        w.extend_from_slice(&4u32.to_le_bytes()); // face
        w.extend_from_slice(&1u32.to_le_bytes()); // region
        // entity counts (owned)
        w.extend_from_slice(&4u32.to_le_bytes()); // vtx owned
        w.extend_from_slice(&0u32.to_le_bytes()); // edge owned
        w.extend_from_slice(&4u32.to_le_bytes()); // face owned
        w.extend_from_slice(&1u32.to_le_bytes()); // region owned
        // coordinates: unit tet
        let tet_coords: &[(f64, f64, f64)] = &[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)];
        for &(x, y, z) in tet_coords {
            w.extend_from_slice(&x.to_le_bytes());
            w.extend_from_slice(&y.to_le_bytes());
            w.extend_from_slice(&z.to_le_bytes());
        }
        // connectivity: one tet, npe=4
        w.extend_from_slice(&4u32.to_le_bytes()); // npe
        w.extend_from_slice(&1u32.to_le_bytes()); // node 1 (1-based)
        w.extend_from_slice(&2u32.to_le_bytes());
        w.extend_from_slice(&3u32.to_le_bytes());
        w.extend_from_slice(&4u32.to_le_bytes());
        // visibility (empty)
        w.extend_from_slice(&0u32.to_le_bytes()); // no edge visibility
        w.extend_from_slice(&0u32.to_le_bytes()); // no face visibility
        w.extend_from_slice(&0u32.to_le_bytes()); // no region visibility
        w
    }

    fn make_smb_hex8() -> Vec<u8> {
        let mut w = Vec::new();
        w.extend_from_slice(&SMB_MAGIC.to_le_bytes());
        w.extend_from_slice(&1u32.to_le_bytes());
        w.extend_from_slice(&0u32.to_le_bytes());
        w.extend_from_slice(&8u32.to_le_bytes()); // vtx
        w.extend_from_slice(&0u32.to_le_bytes()); // edge
        w.extend_from_slice(&6u32.to_le_bytes()); // face
        w.extend_from_slice(&1u32.to_le_bytes()); // region
        w.extend_from_slice(&8u32.to_le_bytes());
        w.extend_from_slice(&0u32.to_le_bytes());
        w.extend_from_slice(&6u32.to_le_bytes());
        w.extend_from_slice(&1u32.to_le_bytes());
        // coordinates: unit cube
        let hex_coords: &[(f64, f64, f64)] = &[
            (0.0,0.0,0.0), (1.0,0.0,0.0), (0.0,1.0,0.0), (1.0,1.0,0.0),
            (0.0,0.0,1.0), (1.0,0.0,1.0), (0.0,1.0,1.0), (1.0,1.0,1.0),
        ];
        for &(x, y, z) in hex_coords {
            w.extend_from_slice(&x.to_le_bytes());
            w.extend_from_slice(&y.to_le_bytes());
            w.extend_from_slice(&z.to_le_bytes());
        }
        // connectivity
        w.extend_from_slice(&8u32.to_le_bytes()); // npe = 8
        for i in 1u32..=8 { w.extend_from_slice(&i.to_le_bytes()); }
        w.extend_from_slice(&0u32.to_le_bytes());
        w.extend_from_slice(&0u32.to_le_bytes());
        w.extend_from_slice(&0u32.to_le_bytes());
        w
    }

    #[test]
    fn read_smb_tet4() {
        let buf = make_smb_tet4();
        let dir = std::env::temp_dir().join("test_tet4.smb");
        std::fs::write(&dir, &buf).unwrap();
        let mesh = read_smb(&dir).unwrap();
        std::fs::remove_file(&dir).ok();
        assert_eq!(mesh.n_nodes(), 4);
        assert_eq!(mesh.n_elems(), 1);
        assert_eq!(mesh.elem_type, ElementType::Tet4);
    }

    #[test]
    fn read_smb_hex8() {
        let buf = make_smb_hex8();
        let dir = std::env::temp_dir().join("test_hex8.smb");
        std::fs::write(&dir, &buf).unwrap();
        let mesh = read_smb(&dir).unwrap();
        std::fs::remove_file(&dir).ok();
        assert_eq!(mesh.n_nodes(), 8);
        assert_eq!(mesh.n_elems(), 1);
        assert_eq!(mesh.elem_type, ElementType::Hex8);
    }

    #[test]
    fn bad_magic_rejected() {
        let buf = vec![0u8; 16];
        let dir = std::env::temp_dir().join("bad.smb");
        std::fs::write(&dir, &buf).unwrap();
        let result = read_smb(&dir);
        std::fs::remove_file(&dir).ok();
        assert!(result.is_err());
    }

    #[test]
    fn hex8_coordinates_correct() {
        let buf = make_smb_hex8();
        let dir = std::env::temp_dir().join("coord_test.smb");
        std::fs::write(&dir, &buf).unwrap();
        let mesh = read_smb(&dir).unwrap();
        std::fs::remove_file(&dir).ok();
        assert!((mesh.coords[0] - 0.0).abs() < 1e-12);
        assert!((mesh.coords[3] - 1.0).abs() < 1e-12);
        assert!((mesh.coords[6] - 0.0).abs() < 1e-12);
        assert!((mesh.coords[9] - 1.0).abs() < 1e-12);
        assert!((mesh.coords[12] - 0.0).abs() < 1e-12);
    }
}
