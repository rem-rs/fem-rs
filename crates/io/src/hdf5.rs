//! # HDF5 mesh + solution I/O — pure-Rust backend (`rust-hdf5` crate)
//!
//! Feature-gated behind `features = ["hdf5"]`.
//!
//! ## File layout
//! ```text
//! /mesh/
//!   dim          — scalar u64 dataset
//!   n_nodes      — scalar u64 dataset
//!   n_elems      — scalar u64 dataset
//!   n_faces      — scalar u64 dataset
//!   coords       — [n_nodes × dim] f64 dataset, row-major
//!   conn         — [total] i64 dataset, padded with -1 for mixed
//!   conn_npe     — [n_elems] u32 dataset
//!   elem_tags    — [n_elems] i32 dataset
//!   elem_type    — scalar u8 dataset (integer code)
//!   face_conn    — [total] i64 dataset, padded with -1 for mixed
//!   face_conn_npf — [n_faces] u32 dataset
//!   face_tags    — [n_faces] i64 dataset
//!   face_type    — scalar u8 dataset (integer code)
//! /fields/<name>/
//!   values       — [n_dofs] f64 dataset
//!   space_type   — VarLenUnicode string attr on `values` dataset
//! /metadata/
//!   version_attr — VarLenUnicode string attr on `marker` dataset
//! ```

use std::path::Path;

use fem_core::NodeId;
use fem_mesh::{BoundaryTag, ElementType, SimplexMesh};

use rust_hdf5::{H5File, H5Group, VarLenUnicode};

/// Public result type for HDF5 I/O operations.
pub type H5Result<T> = Result<T, Box<dyn std::error::Error + Send + Sync + 'static>>;

// ──────────────────────────────────────────────────────────────
// Options
// ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Hdf5WriteOptions {
    /// GZIP compression level 0–9. 0 = none.
    pub compression: u8,
    /// Overwrite existing file.
    pub overwrite: bool,
}

impl Default for Hdf5WriteOptions {
    fn default() -> Self {
        Self { compression: 6, overwrite: true }
    }
}

// ──────────────────────────────────────────────────────────────
// Writer
// ──────────────────────────────────────────────────────────────

pub fn write_mesh_and_fields<const D: usize>(
    path: impl AsRef<Path>,
    mesh: &SimplexMesh<D>,
    fields: &[(&str, &[f64], &str)], // (name, values, space_type)
    options: &Hdf5WriteOptions,
) -> H5Result<()> {
    let path = path.as_ref();
    if options.overwrite && path.exists() {
        std::fs::remove_file(path).ok();
    }
    let path_str = path.to_str().ok_or("non-UTF8 path")?;

    let file = H5File::create(path_str)?;
    let root = file.root_group();
    let mg = root.create_group("mesh")?;

    // scalars (stored as 1-element datasets)
    write_scalar_u64(&mg, "dim", D as u64)?;
    write_scalar_u64(&mg, "n_nodes", mesh.n_nodes() as u64)?;
    write_scalar_u64(&mg, "n_elems", mesh.n_elems() as u64)?;
    write_scalar_u64(&mg, "n_faces", mesh.n_faces() as u64)?;

    // elem / face type codes
    write_scalar_u8(&mg, "elem_type", elem_type_to_code(mesh.elem_type))?;
    write_scalar_u8(&mg, "face_type", elem_type_to_code(mesh.face_type))?;

    // coordinates [n_nodes × D]
    let nn = mesh.n_nodes();
    let len = nn * D;
    let ds = mg.new_dataset::<f64>().shape(&[len]).create("coords")?;
    ds.write_raw(mesh.coords.as_slice())?;

    // connectivity
    write_conn(&mg, mesh, options)?;

    // element tags
    let ds = mg.new_dataset::<i32>().shape(&[mesh.n_elems()]).create("elem_tags")?;
    ds.write_raw(mesh.elem_tags.as_slice())?;

    // face connectivity
    write_face_conn(&mg, mesh, options)?;

    // face tags
    let ft: Vec<i64> = mesh.face_tags.iter().map(|t| *t as i64).collect();
    if !ft.is_empty() {
        let ds = mg.new_dataset::<i64>().shape(&[mesh.n_faces()]).create("face_tags")?;
        ds.write_raw(ft.as_slice())?;
    }

    // ── fields ──
    if !fields.is_empty() {
        let fg = root.create_group("fields")?;
        for (name, values, space_type) in fields {
            let fgrp = fg.create_group(*name)?;
            let nd = values.len();
            let ds = fgrp
                .new_dataset::<f64>()
                .shape(&[nd])
                .create("values")?;
            ds.write_raw(*values)?;
            // space_type stored as VarLenUnicode attribute on the `values` dataset
            let attr = ds.new_attr::<VarLenUnicode>().shape(()).create("space_type")?;
            attr.write_string(space_type)?;
        }
    }

    // metadata — store version as attr on a marker dataset
    let meta = root.create_group("metadata")?;
    let marker = meta.new_dataset::<u8>().shape(&[1]).create("marker")?;
    marker.write_raw(&[0u8])?;
    let attr = marker.new_attr::<VarLenUnicode>().shape(()).create("fem-rs_version")?;
    attr.write_string(env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

fn write_conn<const D: usize>(
    mg: &H5Group,
    mesh: &SimplexMesh<D>,
    _opts: &Hdf5WriteOptions,
) -> H5Result<()> {
    let ne = mesh.n_elems();
    if ne == 0 {
        return Ok(());
    }
    let npe_ref = mesh.elem_type.nodes_per_element();

    let (npe_vec, conn_padded): (Vec<u32>, Vec<i64>) =
        if let Some(ref types) = mesh.elem_types {
            let max_npe = types
                .iter()
                .map(|t| t.nodes_per_element())
                .max()
                .unwrap_or(npe_ref);
            let mut npe = Vec::with_capacity(ne);
            let mut padded = Vec::with_capacity(ne * max_npe);
            if let Some(ref offs) = mesh.elem_offsets {
                for e in 0..ne {
                    let start = offs[e];
                    let end = offs[e + 1];
                    let n = (end - start) as u32;
                    npe.push(n);
                    for i in start..end {
                        padded.push(mesh.conn[i] as i64);
                    }
                    for _ in n as usize..max_npe {
                        padded.push(-1i64);
                    }
                }
            }
            (npe, padded)
        } else {
            let p: Vec<i64> = mesh.conn.iter().map(|&n| n as i64).collect();
            (vec![npe_ref as u32; ne], p)
        };

    let ds = mg.new_dataset::<i64>().shape(&[conn_padded.len()]).create("conn")?;
    ds.write_raw(conn_padded.as_slice())?;

    let ds = mg.new_dataset::<u32>().shape(&[ne]).create("conn_npe")?;
    ds.write_raw(npe_vec.as_slice())?;

    Ok(())
}

fn write_face_conn<const D: usize>(
    mg: &H5Group,
    mesh: &SimplexMesh<D>,
    _opts: &Hdf5WriteOptions,
) -> H5Result<()> {
    let nf = mesh.n_faces();
    if nf == 0 {
        return Ok(());
    }
    let npf_ref = mesh.face_type.nodes_per_element();

    let (npf_vec, face_padded): (Vec<u32>, Vec<i64>) =
        if let Some(ref types) = mesh.face_types {
            let max_npf = types
                .iter()
                .map(|t| t.nodes_per_element())
                .max()
                .unwrap_or(npf_ref);
            let mut npf = Vec::with_capacity(nf);
            let mut padded = Vec::with_capacity(nf * max_npf);
            if let Some(ref offs) = mesh.face_offsets {
                for f in 0..nf {
                    let start = offs[f];
                    let end = offs[f + 1];
                    let n = (end - start) as u32;
                    npf.push(n);
                    for i in start..end {
                        padded.push(mesh.face_conn[i] as i64);
                    }
                    for _ in n as usize..max_npf {
                        padded.push(-1i64);
                    }
                }
            }
            (npf, padded)
        } else {
            let p: Vec<i64> = mesh.face_conn.iter().map(|&n| n as i64).collect();
            (vec![npf_ref as u32; nf], p)
        };

    let ds = mg.new_dataset::<i64>().shape(&[face_padded.len()]).create("face_conn")?;
    ds.write_raw(face_padded.as_slice())?;

    let ds = mg.new_dataset::<u32>().shape(&[nf]).create("face_conn_npf")?;
    ds.write_raw(npf_vec.as_slice())?;

    Ok(())
}

// ──────────────────────────────────────────────────────────────
// Reader
// ──────────────────────────────────────────────────────────────

pub fn read_mesh_and_fields<const D: usize>(
    path: impl AsRef<Path>,
) -> H5Result<(SimplexMesh<D>, Vec<(String, Vec<f64>, String)>)> {
    let path_str = path.as_ref().to_str().ok_or("non-UTF8 path")?;
    let file = H5File::open(path_str)?;
    let root = file.root_group();

    // dimension sanity check
    let dim = read_scalar_u64(&file, "mesh/dim")?;
    assert_eq!(dim as usize, D, "dimension mismatch");

    let n_elems = read_scalar_u64(&file, "mesh/n_elems")? as usize;
    let n_faces = read_scalar_u64(&file, "mesh/n_faces")? as usize;

    // coords
    let coords: Vec<f64> = file.dataset("mesh/coords")?.read_raw::<f64>()?;

    // element type
    let elem_type = read_elem_type_code(&file, "mesh/elem_type")?;

    // connectivity
    let (conn, elem_types, elem_offsets) = read_conn(&file, "mesh", n_elems, elem_type)?;

    // element tags
    let elem_tags: Vec<i32> = file.dataset("mesh/elem_tags")?.read_raw::<i32>()?;

    // face type
    let face_type = read_elem_type_code(&file, "mesh/face_type")?;

    // face connectivity
    let (face_conn, face_types, face_offsets) =
        read_face_conn(&file, "mesh", n_faces, face_type)?;

    // face tags
    let face_tags: Vec<BoundaryTag> = if n_faces > 0 {
        file.dataset("mesh/face_tags")?
            .read_raw::<i64>()?
            .into_iter()
            .map(|t| t as BoundaryTag)
            .collect()
    } else {
        vec![]
    };

    let mesh = SimplexMesh {
        coords,
        conn,
        elem_tags,
        elem_type,
        face_conn,
        face_tags,
        face_type,
        elem_types,
        elem_offsets,
        face_types,
        face_offsets,
    };

    // fields
    let mut fields = Vec::new();
    if let Ok(fg) = root.group("fields") {
        for name in fg.group_names()? {
            let values: Vec<f64> = file
                .dataset(&format!("fields/{name}/values"))?
                .read_raw::<f64>()?;
            let ds = file.dataset(&format!("fields/{name}/values"))?;
            let space_type = ds.attr("space_type")?.read_string()?;
            fields.push((name, values, space_type));
        }
    }

    Ok((mesh, fields))
}

fn read_conn(
    file: &H5File,
    prefix: &str,
    n_elems: usize,
    default_type: ElementType,
) -> H5Result<(
    Vec<NodeId>,
    Option<Vec<ElementType>>,
    Option<Vec<usize>>,
)> {
    if n_elems == 0 {
        return Ok((vec![], None, None));
    }
    let conn_raw: Vec<i64> = file.dataset(&format!("{prefix}/conn"))?.read_raw::<i64>()?;
    let conn_npe: Vec<u32> = file.dataset(&format!("{prefix}/conn_npe"))?.read_raw::<u32>()?;

    let first = conn_npe.first().copied().unwrap_or(0);
    let npe_ref = default_type.nodes_per_element() as u32;
    let uniform = conn_npe.iter().all(|&n| n == first) && first == npe_ref;

    if uniform {
        let conn: Vec<NodeId> = conn_raw.into_iter().map(|n| n as NodeId).collect();
        Ok((conn, None, None))
    } else {
        let mut conn = Vec::new();
        let mut types = Vec::with_capacity(n_elems);
        let mut offsets = Vec::with_capacity(n_elems + 1);
        offsets.push(0);
        let dims = file.dataset(&format!("{prefix}/conn"))?.shape();
        let stride = dims.get(1).copied().unwrap_or(0);
        for e in 0..n_elems {
            let npe = conn_npe[e] as usize;
            let base = e * stride;
            for j in 0..npe {
                conn.push(conn_raw[base + j] as NodeId);
            }
            offsets.push(conn.len());
            types.push(default_type);
        }
        Ok((conn, Some(types), Some(offsets)))
    }
}

fn read_face_conn(
    file: &H5File,
    prefix: &str,
    n_faces: usize,
    default_type: ElementType,
) -> H5Result<(
    Vec<NodeId>,
    Option<Vec<ElementType>>,
    Option<Vec<usize>>,
)> {
    if n_faces == 0 {
        return Ok((vec![], None, None));
    }
    let face_raw: Vec<i64> = file.dataset(&format!("{prefix}/face_conn"))?.read_raw::<i64>()?;
    let face_npf: Vec<u32> = file.dataset(&format!("{prefix}/face_conn_npf"))?.read_raw::<u32>()?;

    let first = face_npf.first().copied().unwrap_or(0);
    let npf_ref = default_type.nodes_per_element() as u32;
    let uniform = face_npf.iter().all(|&n| n == first) && first == npf_ref;

    if uniform {
        let conn: Vec<NodeId> = face_raw.into_iter().map(|n| n as NodeId).collect();
        Ok((conn, None, None))
    } else {
        let mut conn = Vec::new();
        let mut types = Vec::with_capacity(n_faces);
        let mut offsets = Vec::with_capacity(n_faces + 1);
        offsets.push(0);
        let dims = file.dataset(&format!("{prefix}/face_conn"))?.shape();
        let stride = dims.get(1).copied().unwrap_or(0);
        for f in 0..n_faces {
            let npf = face_npf[f] as usize;
            let base = f * stride;
            for j in 0..npf {
                conn.push(face_raw[base + j] as NodeId);
            }
            offsets.push(conn.len());
            types.push(default_type);
        }
        Ok((conn, Some(types), Some(offsets)))
    }
}

// ──────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────

fn elem_type_to_code(t: ElementType) -> u8 {
    match t {
        ElementType::Line2 => 1,
        ElementType::Tri3  => 2,
        ElementType::Quad4 => 3,
        ElementType::Tet4  => 4,
        ElementType::Hex8  => 5,
        other => panic!("unsupported element type for HDF5: {other:?}"),
    }
}

fn code_to_elem_type(code: u8) -> ElementType {
    match code {
        1 => ElementType::Line2,
        2 => ElementType::Tri3,
        3 => ElementType::Quad4,
        4 => ElementType::Tet4,
        5 => ElementType::Hex8,
        other => panic!("unknown element type code: {other}"),
    }
}

fn read_elem_type_code(file: &H5File, path: &str) -> H5Result<ElementType> {
    let v = file.dataset(path)?.read_raw::<u8>()?;
    Ok(code_to_elem_type(v[0]))
}

fn write_scalar_u8(grp: &H5Group, name: &str, val: u8) -> H5Result<()> {
    let ds = grp.new_dataset::<u8>().shape(&[1]).create(name)?;
    ds.write_raw(&[val])?;
    Ok(())
}

fn write_scalar_u64(grp: &H5Group, name: &str, val: u64) -> H5Result<()> {
    let ds = grp.new_dataset::<u64>().shape(&[1]).create(name)?;
    ds.write_raw(&[val])?;
    Ok(())
}

fn read_scalar_u64(file: &H5File, path: &str) -> H5Result<u64> {
    let v = file.dataset(path)?.read_raw::<u64>()?;
    Ok(v[0])
}

// ──────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn tri_mesh_2d() -> SimplexMesh<2> {
        SimplexMesh {
            coords: vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            conn: vec![
                0u32, 1u32, 2u32,
                1u32, 3u32, 2u32,
            ],
            elem_tags: vec![1, 1],
            elem_type: ElementType::Tri3,
            face_conn: vec![
                0u32, 1u32,
                1u32, 3u32,
                2u32, 0u32,
                3u32, 2u32,
            ],
            face_tags: vec![10, 20, 10, 30],
            face_type: ElementType::Line2,
            elem_types: None,
            elem_offsets: None,
            face_types: None,
            face_offsets: None,
        }
    }

    #[test]
    fn test_hdf5_roundtrip_tri2d() {
        let mesh = tri_mesh_2d();
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.h5");

        let fields: [(&str, &[f64], &str); 1] =
            [("temperature", &[0.0, 0.5, 0.5, 1.0, 0.5, 0.0], "H1")];

        write_mesh_and_fields(&path, &mesh, &fields, &Hdf5WriteOptions::default())
            .expect("write");

        let (mesh2, fields2) = read_mesh_and_fields::<2>(&path).expect("read");

        assert_eq!(mesh2.n_nodes(), 4);
        assert_eq!(mesh2.n_elems(), 2);
        assert_eq!(mesh2.n_faces(), 4);
        assert_eq!(mesh2.coords, mesh.coords);
        assert_eq!(mesh2.elem_tags, mesh.elem_tags);
        assert_eq!(mesh2.face_tags, mesh.face_tags);
        for (&a, &b) in mesh2.conn.iter().zip(mesh.conn.iter()) {
            assert_eq!(a, b);
        }

        assert_eq!(fields2.len(), 1);
        assert_eq!(fields2[0].0, "temperature");
        assert_eq!(fields2[0].1, &[0.0, 0.5, 0.5, 1.0, 0.5, 0.0]);
        assert_eq!(fields2[0].2, "H1");
    }

    #[test]
    fn test_hdf5_roundtrip_tet3d() {
        let mesh: SimplexMesh<3> = SimplexMesh {
            coords: vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            ],
            conn: vec![
                0u32, 1u32, 2u32, 3u32,
            ],
            elem_tags: vec![1],
            elem_type: ElementType::Tet4,
            face_conn: vec![
                1u32, 2u32, 3u32,
                0u32, 3u32, 2u32,
                0u32, 1u32, 3u32,
                0u32, 2u32, 1u32,
            ],
            face_tags: vec![10, 20, 20, 20],
            face_type: ElementType::Tri3,
            elem_types: None,
            elem_offsets: None,
            face_types: None,
            face_offsets: None,
        };

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("tet.h5");
        write_mesh_and_fields(&path, &mesh, &[], &Hdf5WriteOptions::default())
            .expect("write tet");
        let (mesh2, _) = read_mesh_and_fields::<3>(&path).expect("read tet");

        assert_eq!(mesh2.n_nodes(), 4);
        assert_eq!(mesh2.n_elems(), 1);
        assert_eq!(mesh2.n_faces(), 4);
        assert_eq!(mesh2.coords, mesh.coords);
    }
}
