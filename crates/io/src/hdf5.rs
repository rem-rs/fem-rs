//! # HDF5 mesh + solution I/O
//!
//! Feature-gated behind `features = ["hdf5"]`.
//!
//! ## File layout
//! ```text
//! /mesh/
//!   dim          — scalar u32, spatial dimension
//!   n_nodes      — scalar u64
//!   n_elems      — scalar u64
//!   n_faces      — scalar u64
//!   coords       — [n_nodes × dim] f64, row-major
//!   conn         — [n_elems × max_npe] i64, padded with -1 for mixed
//!   conn_npe     — [n_elems] u32
//!   elem_tags    — [n_elems] i32
//!   elem_type    — string attr
//!   face_conn    — [n_faces × max_npf] i64, padded with -1
//!   face_tags    — [n_faces] i64
//!   face_type    — string attr
//! /fields/<name>/
//!   values       — [n_dofs] f64
//!   space_type   — string attr
//! /metadata/
//!   fem-rs_version — string attr
//! ```

use std::path::Path;

use fem_core::NodeId;
use fem_mesh::{BoundaryTag, ElementType, SimplexMesh};

use std::str::FromStr;
use hdf5::{
    types::VarLenUnicode, File, H5Type, Result as H5Result,
};

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

    let file = File::create(path)?;
    let mg = file.create_group("mesh")?;

    // scalars
    write_scalar(&mg, "dim", D as u32)?;
    write_scalar(&mg, "n_nodes", mesh.n_nodes() as u64)?;
    write_scalar(&mg, "n_elems", mesh.n_elems() as u64)?;
    write_scalar(&mg, "n_faces", mesh.n_faces() as u64)?;

    // coordinates [n_nodes × D]
    let nn = mesh.n_nodes();
    let len = nn * D;
    let dset = mg
        .new_dataset::<f64>()
        .shape((len,))
        .create("coords")?;
    dset.write(mesh.coords.as_slice())?;

    // connectivity
    write_conn(&mg, mesh, options)?;

    // element tags
    let dset = mg
        .new_dataset::<i32>()
        .shape((mesh.n_elems(),))
        .create("elem_tags")?;
    dset.write(mesh.elem_tags.as_slice())?;

    // face connectivity
    write_face_conn(&mg, mesh, options)?;

    // face tags
    let ft: Vec<i64> = mesh.face_tags.iter().map(|t| *t as i64).collect();
    if !ft.is_empty() {
        let dset = mg
            .new_dataset::<i64>()
            .shape((mesh.n_faces(),))
            .create("face_tags")?;
        dset.write(ft.as_slice())?;
    }

    // element type as attribute (VarLenUnicode for portable read)
    {
        let s = VarLenUnicode::from_str(elem_type_str(mesh.elem_type)).unwrap();
        let attr = mg.new_attr::<VarLenUnicode>().shape(()).create("elem_type")?;
        attr.write_scalar(&s)?;
    }
    {
        let s = VarLenUnicode::from_str(elem_type_str(mesh.face_type)).unwrap();
        let attr = mg.new_attr::<VarLenUnicode>().shape(()).create("face_type")?;
        attr.write_scalar(&s)?;
    }

    // ── fields ──
    if !fields.is_empty() {
        let fg = file.create_group("fields")?;
        for (name, values, space_type) in fields {
            let fgrp = fg.create_group(*name)?;
            let nd = values.len();
            let dset = fgrp
                .new_dataset::<f64>()
                .shape((nd,))
                .chunk([(4096.min(nd)).max(1)])
                .deflate(options.compression)
                .create("values")?;
            dset.write(values)?;
            let s = VarLenUnicode::from_str(space_type).unwrap();
            let attr = fgrp.new_attr::<VarLenUnicode>().shape(()).create("space_type")?;
            attr.write_scalar(&s)?;
        }
    }

    // metadata
    let meta = file.create_group("metadata")?;
    {
        let s = VarLenUnicode::from_str(env!("CARGO_PKG_VERSION")).unwrap();
        let attr = meta.new_attr::<VarLenUnicode>().shape(()).create("fem-rs_version")?;
        attr.write_scalar(&s)?;
    }

    Ok(())
}

fn write_conn<const D: usize>(
    mg: &hdf5::Group,
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

    let dset = mg
        .new_dataset::<i64>()
        .shape((conn_padded.len(),))
        .create("conn")?;
    dset.write(conn_padded.as_slice())?;

    let dset = mg
        .new_dataset::<u32>()
        .shape((ne,))
        .create("conn_npe")?;
    dset.write(npe_vec.as_slice())?;

    Ok(())
}

fn write_face_conn<const D: usize>(
    mg: &hdf5::Group,
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

    let dset = mg
        .new_dataset::<i64>()
        .shape((face_padded.len(),))
        .create("face_conn")?;
    dset.write(face_padded.as_slice())?;

    let dset = mg
        .new_dataset::<u32>()
        .shape((nf,))
        .create("face_conn_npf")?;
    dset.write(npf_vec.as_slice())?;

    Ok(())
}

// ──────────────────────────────────────────────────────────────
// Reader
// ──────────────────────────────────────────────────────────────

pub fn read_mesh_and_fields<const D: usize>(
    path: impl AsRef<Path>,
) -> H5Result<(SimplexMesh<D>, Vec<(String, Vec<f64>, String)>)> {
    let file = File::open(path)?;
    let mg = file.group("mesh")?;

    // dimension sanity check
    let dim: u32 = read_scalar(&mg, "dim")?;
    assert_eq!(dim as usize, D, "dimension mismatch");

    let _n_nodes: u64 = read_scalar(&mg, "n_nodes")?;
    let n_elems: u64 = read_scalar(&mg, "n_elems")?;
    let n_faces: u64 = read_scalar(&mg, "n_faces")?;

    // coords
    let coords: Vec<f64> = mg.dataset("coords")?.read_1d()?.to_vec();

    // element type
    let elem_type = read_elem_type_attr(&mg, "elem_type")?;

    // connectivity
    let (conn, elem_types, elem_offsets) =
        read_conn(&mg, n_elems as usize, elem_type)?;

    // element tags
    let elem_tags: Vec<i32> = mg.dataset("elem_tags")?.read_1d()?.to_vec();

    // face type
    let face_type = read_elem_type_attr(&mg, "face_type")?;

    // face connectivity
    let (face_conn, face_types, face_offsets) =
        read_face_conn(&mg, n_faces as usize, face_type)?;

    // face tags
    let face_tags: Vec<BoundaryTag> = if n_faces > 0 {
        let raw: Vec<i64> = mg.dataset("face_tags")?.read_1d()?.to_vec();
        raw.into_iter().map(|t| t as BoundaryTag).collect()
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
    if let Ok(fg) = file.group("fields") {
        for name in fg.member_names()? {
            let fgrp = fg.group(&name)?;
            let values: Vec<f64> = fgrp.dataset("values")?.read_1d()?.to_vec();
            let space_type: String = {
                let attr = fgrp.attr("space_type")?;
                let v: VarLenUnicode = attr.as_reader().read_scalar()?;
            let s = v.to_string();
                s.to_string()
            };
            fields.push((name, values, space_type));
        }
    }

    Ok((mesh, fields))
}

fn read_conn(
    mg: &hdf5::Group,
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
    let conn_raw: Vec<i64> = mg.dataset("conn")?.read_1d()?.to_vec();
    let conn_npe: Vec<u32> = mg.dataset("conn_npe")?.read_1d()?.to_vec();

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
        let dims = mg.dataset("conn")?.shape();
        let stride = dims.get(1).copied().unwrap_or(0) as usize;
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
    mg: &hdf5::Group,
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
    let face_raw: Vec<i64> = mg.dataset("face_conn")?.read_1d()?.to_vec();
    let face_npf: Vec<u32> = mg.dataset("face_conn_npf")?.read_1d()?.to_vec();

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
        let dims = mg.dataset("face_conn")?.shape();
        let stride = dims.get(1).copied().unwrap_or(0) as usize;
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

fn elem_type_str(t: ElementType) -> &'static str {
    match t {
        ElementType::Line2 => "Line2",
        ElementType::Tri3   => "Tri3",
        ElementType::Quad4  => "Quad4",
        ElementType::Tet4   => "Tet4",
        ElementType::Hex8   => "Hex8",
        other => panic!("unsupported element type for HDF5: {other:?}"),
    }
}

fn read_elem_type_attr(mg: &hdf5::Group, name: &str) -> H5Result<ElementType> {
    let attr = mg.attr(name)?;
    let v: VarLenUnicode = attr.as_reader().read_scalar()?;
    let s = v.to_string();
    Ok(match s.as_str() {
        "Line2" => ElementType::Line2,
        "Tri3"  => ElementType::Tri3,
        "Quad4" => ElementType::Quad4,
        "Tet4"  => ElementType::Tet4,
        "Hex8"  => ElementType::Hex8,
        other => panic!("unknown element type: {other}"),
    })
}

fn write_scalar<T: H5Type + Copy>(grp: &hdf5::Group, name: &str, val: T) -> H5Result<()> {
    let dset = grp.new_dataset::<T>().shape((1,)).create(name)?;
    dset.write(&[val])?;
    Ok(())
}

fn read_scalar<T: H5Type + Copy>(grp: &hdf5::Group, name: &str) -> H5Result<T> {
    let dset = grp.dataset(name)?;
    let data: Vec<T> = dset.as_reader().read_raw().map_err(|e| {
        eprintln!("read_scalar({name}) failed: {e}");
        e
    })?;
    Ok(data[0])
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
