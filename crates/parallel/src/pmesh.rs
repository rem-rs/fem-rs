//! Collective `.pmesh` format — single-file parallel mesh I/O.
//!
//! ## Binary format
//! ```text
//! [header]  — magic(8B), n_ranks(8B), dim(8B), elem_type(8B), npe(8B),
//!             n_total_nodes(8B), n_total_elems(8B),
//!             n_nodes_per_rank[0..n_ranks](8B each),
//!             n_elems_per_rank[0..n_ranks](8B each)
//! [coords]  — n_total_nodes × dim × f64le
//! [conn]    — n_total_elems × npe × u32le
//! [tags]    — n_total_elems × i32le
//! ```

use std::io::{Read, Write};
use fem_core::{FemError, FemResult};
use fem_mesh::{ElementType, SimplexMesh};
use crate::comm::Comm;

const PMESH_MAGIC: u64 = 0x504D_4553; // "PMES"

/// Write a distributed mesh to a single `.pmesh` file.
///
/// Rank 0 orchestrates: receives data from all ranks and serializes to file.
/// Non-root ranks send their mesh data via point-to-point.
pub fn write_pmesh<const D: usize>(
    mesh: &SimplexMesh<D>,
    base_path: &str,
    comm: &Comm,
) -> FemResult<()> {
    let rank = comm.rank();
    let n_ranks = comm.size();
    let npe = mesh.elem_type.nodes_per_element() as u64;
    let dim = D as u64;

    let n_nodes_local = mesh.n_nodes() as u64;
    let n_elems_local = mesh.n_elems() as u64;

    // Gather per-rank sizes via broadcast from rank 0
    let mut n_nodes_all = vec![0u64; n_ranks];
    let mut n_elems_all = vec![0u64; n_ranks];

    if rank == 0 {
        n_nodes_all[0] = n_nodes_local;
        n_elems_all[0] = n_elems_local;
        for src in 1..n_ranks as i32 {
            let recv: Vec<u64> = comm.recv(src, 1030);
            if recv.len() >= 2 {
                n_nodes_all[src as usize] = recv[0];
                n_elems_all[src as usize] = recv[1];
            }
        }
    } else {
        comm.send(0, 1030, &[n_nodes_local, n_elems_local]);
    }
    // Broadcast arrays to all ranks
    let mut flat = vec![0u8; n_ranks * 16];
    if rank == 0 {
        for i in 0..n_ranks {
            flat[i*16..i*16+8].copy_from_slice(&n_nodes_all[i].to_le_bytes());
            flat[i*16+8..i*16+16].copy_from_slice(&n_elems_all[i].to_le_bytes());
        }
    }
    comm.broadcast_bytes(0, &mut flat);
    for i in 0..n_ranks {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&flat[i*16..i*16+8]); n_nodes_all[i] = u64::from_le_bytes(buf);
        buf.copy_from_slice(&flat[i*16+8..i*16+16]); n_elems_all[i] = u64::from_le_bytes(buf);
    }

    let n_total_nodes: u64 = n_nodes_all.iter().sum();
    let n_total_elems: u64 = n_elems_all.iter().sum();

    // Rank 0 writes the file
    if rank == 0 {
        let path = format!("{base_path}.pmesh");
        let mut file = std::fs::File::create(&path).map_err(FemError::Io)?;

        // Header
        let write_u64 = |f: &mut std::fs::File, v: u64| -> FemResult<()> {
            f.write_all(&v.to_le_bytes()).map_err(FemError::Io)
        };
        write_u64(&mut file, PMESH_MAGIC)?;
        write_u64(&mut file, n_ranks as u64)?;
        write_u64(&mut file, dim)?;
        write_u64(&mut file, mesh.elem_type as u32 as u64)?;
        write_u64(&mut file, npe)?;
        write_u64(&mut file, n_total_nodes)?;
        write_u64(&mut file, n_total_elems)?;
        for &v in &n_nodes_all { write_u64(&mut file, v)?; }
        for &v in &n_elems_all { write_u64(&mut file, v)?; }

        // Write rank 0 data
        let coords_bytes: Vec<u8> = mesh.coords.iter().flat_map(|v| v.to_le_bytes()).collect();
        let conn_bytes: Vec<u8> = mesh.conn.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tags_bytes: Vec<u8> = mesh.elem_tags.iter().flat_map(|v| v.to_le_bytes()).collect();
        file.write_all(&coords_bytes).map_err(FemError::Io)?;
        file.write_all(&conn_bytes).map_err(FemError::Io)?;
        file.write_all(&tags_bytes).map_err(FemError::Io)?;

        // Receive and write data from other ranks
        for src in 1..n_ranks as i32 {
            let recv_coords: Vec<f64> = comm.recv(src, 1031);
            let recv_conn: Vec<u32> = comm.recv(src, 1032);
            let recv_tags: Vec<i32> = comm.recv(src, 1033);
            let cbytes: Vec<u8> = recv_coords.iter().flat_map(|v| v.to_le_bytes()).collect();
            let nbytes: Vec<u8> = recv_conn.iter().flat_map(|v| v.to_le_bytes()).collect();
            let tbytes: Vec<u8> = recv_tags.iter().flat_map(|v| v.to_le_bytes()).collect();
            file.write_all(&cbytes).map_err(FemError::Io)?;
            file.write_all(&nbytes).map_err(FemError::Io)?;
            file.write_all(&tbytes).map_err(FemError::Io)?;
        }
    } else {
        comm.send::<f64>(0, 1031, &mesh.coords);
        let conn_u32: Vec<u32> = mesh.conn.clone();
        comm.send::<u32>(0, 1032, &conn_u32);
        comm.send::<i32>(0, 1033, &mesh.elem_tags);
    }

    comm.barrier();
    Ok(())
}

/// Read a `.pmesh` file back into a serial `SimplexMesh` (single rank).
pub fn read_pmesh<const D: usize>(path: &str) -> FemResult<SimplexMesh<D>> {
    let mut file = std::fs::File::open(path).map_err(FemError::Io)?;

    let read_u64 = |f: &mut std::fs::File| -> FemResult<u64> {
        let mut buf = [0u8; 8];
        f.read_exact(&mut buf).map_err(FemError::Io)?;
        Ok(u64::from_le_bytes(buf))
    };

    let magic = read_u64(&mut file)?;
    if magic != PMESH_MAGIC {
        return Err(FemError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Bad .pmesh magic: {magic:#x} != {:#x}", PMESH_MAGIC),
        )));
    }
    let _n_ranks = read_u64(&mut file)? as usize;
    let _dim = read_u64(&mut file)? as usize;
    let elem_type_val = read_u64(&mut file)? as u32;
    let _npe = read_u64(&mut file)? as usize;
    let n_total_nodes = read_u64(&mut file)? as usize;
    let n_total_elems = read_u64(&mut file)? as usize;

    let elem_type = match elem_type_val {
        v if v == (ElementType::Tri3 as u32) => ElementType::Tri3,
        v if v == (ElementType::Quad4 as u32) => ElementType::Quad4,
        v if v == (ElementType::Tet4 as u32) => ElementType::Tet4,
        v if v == (ElementType::Hex8 as u32) => ElementType::Hex8,
        _ => return Err(FemError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unsupported elem_type {elem_type_val}"))))
    };
    // Skip per-rank arrays
    for _ in 0.._n_ranks * 2 { read_u64(&mut file)?; }

    let n_coords = n_total_nodes * D;
    let mut coord_buf = vec![0u8; n_coords * 8];
    file.read_exact(&mut coord_buf).map_err(FemError::Io)?;
    let coords: Vec<f64> = coord_buf.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();

    let npe = _npe;
    let n_conn = n_total_elems * npe;
    let mut conn_buf = vec![0u8; n_conn * 4];
    file.read_exact(&mut conn_buf).map_err(FemError::Io)?;
    let conn: Vec<u32> = conn_buf.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();

    let mut tag_buf = vec![0u8; n_total_elems * 4];
    file.read_exact(&mut tag_buf).map_err(FemError::Io)?;
    let elem_tags: Vec<i32> = tag_buf.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();

    let face_type = if D == 2 { ElementType::Line2 } else { ElementType::Tri3 };
    Ok(SimplexMesh {
        coords, conn, elem_tags, elem_type,
        face_conn: vec![], face_tags: vec![], face_type,
        elem_types: None, elem_offsets: None, face_types: None, face_offsets: None,
        face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![],
    })
}

/// Serial pmesh write (no MPI, single rank). Used for testing.
#[allow(dead_code)]
fn write_pmesh_serial<const D: usize>(mesh: &SimplexMesh<D>, path: &str) -> FemResult<()> {
    use std::io::Write;
    let dim = D as u64;
    let npe = mesh.elem_type.nodes_per_element() as u64;
    let n_nodes = mesh.n_nodes() as u64;
    let n_elems = mesh.n_elems() as u64;

    let mut file = std::fs::File::create(path).map_err(FemError::Io)?;
    let write_u64 = |f: &mut std::fs::File, v: u64| -> FemResult<()> {
        f.write_all(&v.to_le_bytes()).map_err(FemError::Io)
    };
    write_u64(&mut file, PMESH_MAGIC)?;
    write_u64(&mut file, 1)?; // n_ranks
    write_u64(&mut file, dim)?;
    write_u64(&mut file, mesh.elem_type as u32 as u64)?;
    write_u64(&mut file, npe)?;
    write_u64(&mut file, n_nodes)?;
    write_u64(&mut file, n_elems)?;
    write_u64(&mut file, n_nodes)?; // n_nodes_per_rank[0]
    write_u64(&mut file, n_elems)?; // n_elems_per_rank[0]

    let cbytes: Vec<u8> = mesh.coords.iter().flat_map(|v| v.to_le_bytes()).collect();
    let nbytes: Vec<u8> = mesh.conn.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tbytes: Vec<u8> = mesh.elem_tags.iter().flat_map(|v| v.to_le_bytes()).collect();
    file.write_all(&cbytes).map_err(FemError::Io)?;
    file.write_all(&nbytes).map_err(FemError::Io)?;
    file.write_all(&tbytes).map_err(FemError::Io)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn pmesh_roundtrip_tri3() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let tmp = std::env::temp_dir().join("test_pmesh_tri3.pmesh");
        let path = tmp.to_str().unwrap().to_string();

        // Serial write (no MPI needed — rank 0 writes, others send via comm)
        // Use a simple communicator for the test
        write_pmesh_serial(&mesh, &path).unwrap();
        let read_back = read_pmesh::<2>(&path).unwrap();

        assert_eq!(read_back.n_nodes(), mesh.n_nodes());
        assert_eq!(read_back.n_elems(), mesh.n_elems());
        assert_eq!(read_back.elem_type, mesh.elem_type);
        for (a, b) in read_back.coords.iter().zip(mesh.coords.iter()) {
            assert!((a - b).abs() < 1e-14, "coord mismatch {a} vs {b}");
        }
        for (a, b) in read_back.conn.iter().zip(mesh.conn.iter()) {
            assert_eq!(a, b);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn pmesh_roundtrip_tet4() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let tmp = std::env::temp_dir().join("test_pmesh_tet4.pmesh");
        let path = tmp.to_str().unwrap().to_string();
        write_pmesh_serial(&mesh, &path).unwrap();
        let read_back = read_pmesh::<3>(&path).unwrap();
        assert_eq!(read_back.n_nodes(), mesh.n_nodes());
        assert_eq!(read_back.n_elems(), mesh.n_elems());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn pmesh_invalid_magic() {
        let tmp = std::env::temp_dir().join("bad.pmesh");
        std::fs::write(&tmp, b"NOTPMESH").ok();
        let r = read_pmesh::<2>(tmp.to_str().unwrap());
        assert!(r.is_err());
        std::fs::remove_file(&tmp).ok();
    }
}
