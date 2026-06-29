//! MFEM `.mesh` format reader (v1.0 / v1.2).
//!
//! Supports linear elements in 2D and 3D:
//! Segment, Triangle, Quadrilateral, Tetrahedron, Hexahedron, Wedge, Pyramid.
//! Format reference: https://mfem.org/mesh-format/

use std::io::{BufRead, BufReader, Read};

use fem_core::{FemError, FemResult};
use fem_mesh::{element_type::ElementType, simplex::SimplexMesh};

fn mfem_elem_type(code: u32) -> Option<ElementType> {
    Some(match code {
        1 => ElementType::Line2,
        2 => ElementType::Tri3,
        3 => ElementType::Quad4,
        4 => ElementType::Tet4,
        5 => ElementType::Hex8,
        6 => ElementType::Prism6,
        7 => ElementType::Pyramid5,
        8 => ElementType::Line3,
        9 => ElementType::Tri6,
        10 => ElementType::Quad8,
        11 => ElementType::Tet10,
        12 => ElementType::Hex20,
        13 => ElementType::Prism15,
        14 => ElementType::Pyramid13,
        _ => return None,
    })
}

/// Parsed MFEM mesh data (supports both 2D and 3D).
pub struct MfemFile {
    pub mesh2d: Option<SimplexMesh<2>>,
    pub mesh3d: Option<SimplexMesh<3>>,
}

/// Read an MFEM `.mesh` file from a `BufRead` source.
pub fn read_mfem<R: Read>(reader: R) -> FemResult<MfemFile> {
    let mut r = BufReader::new(reader);
    let mut line = String::new();

    r.read_line(&mut line)?;
    if !line.trim().starts_with("MFEM mesh") {
        return Err(FemError::Mesh(format!("expected 'MFEM mesh' header, got: {line}")));
    }

    // skip section keyword, then read value
    read_line(&mut r)?;  // "dimension"
    let dim = read_uint(&mut r)?;
    if dim != 2 && dim != 3 {
        return Err(FemError::Mesh(format!("MFEM: dim={dim} unsupported")));
    }

    read_line(&mut r)?;  // "elements"
    let n_elem = read_uint(&mut r)?;
    let mut elem_conn: Vec<Vec<u32>> = Vec::with_capacity(n_elem);
    let mut elem_tags: Vec<i32> = Vec::with_capacity(n_elem);
    let mut elem_types: Vec<ElementType> = Vec::with_capacity(n_elem);
    let mut uniform_type: Option<ElementType> = None;
    for _ in 0..n_elem {
        let vals = read_uint_line(&mut r)?;
        if vals.len() < 3 { return Err(FemError::Mesh("MFEM: invalid element line".into())); }
        let et = mfem_elem_type(vals[0] as u32)
            .ok_or_else(|| FemError::Mesh(format!("MFEM: unknown elem type {}", vals[0])))?;
        let npe = et.nodes_per_element();
        if vals.len() != 2 + npe {
            return Err(FemError::Mesh(format!("MFEM: elem type {} expects {npe} nodes, got {}", vals[0], vals.len() - 2)));
        }
        elem_types.push(et);
        elem_tags.push(vals[1] as i32);
        elem_conn.push(vals[2..].iter().map(|&v| (v - 1) as u32).collect());
        if n_elem == 1 { uniform_type = Some(et); }
    }
    if n_elem > 0 {
        let first = elem_types[0];
        uniform_type = if elem_types.iter().all(|&t| t == first) { Some(first) } else { None };
    }

    read_line(&mut r)?;  // "boundary"
    let n_bdr = read_uint(&mut r)?;
    let mut bdr_types: Vec<ElementType> = Vec::with_capacity(n_bdr);
    let mut face_conn: Vec<Vec<u32>> = Vec::with_capacity(n_bdr);
    let mut face_tags: Vec<i32> = Vec::with_capacity(n_bdr);
    for _ in 0..n_bdr {
        let vals = read_uint_line(&mut r)?;
        if vals.len() < 3 { return Err(FemError::Mesh("MFEM: invalid boundary line".into())); }
        let et = mfem_elem_type(vals[0] as u32)
            .ok_or_else(|| FemError::Mesh(format!("MFEM: unknown boundary type {}", vals[0])))?;
        let npe = et.nodes_per_element();
        if vals.len() != 2 + npe {
            return Err(FemError::Mesh(format!("MFEM: bdr type {} expects {npe} nodes", vals[0])));
        }
        bdr_types.push(et);
        face_tags.push(vals[1] as i32);
        face_conn.push(vals[2..].iter().map(|&v| (v - 1) as u32).collect());
    }

    read_line(&mut r)?;  // "vertices"
    let n_vert = read_uint(&mut r)?;
    // vertex dimension (often same as mesh dim)
    let _vdim = read_uint(&mut r)?;
    let mut coords = Vec::with_capacity(n_vert * dim);
    for _ in 0..n_vert {
        let v = read_f64_line(&mut r)?;
        if v.len() < dim { return Err(FemError::Mesh("MFEM: invalid vertex line".into())); }
        coords.extend_from_slice(&v[..dim]);
    }

    let flat_elem = elem_conn.into_iter().flatten().collect();
    let flat_face = face_conn.into_iter().flatten().collect();

    let face_type_from_file = if n_bdr > 0 {
        let first = bdr_types[0];
        if bdr_types.iter().all(|&t| t == first) { first } else { ElementType::Line2 }
    } else if uniform_type.is_some() {
        uniform_type.unwrap().boundary_type().unwrap_or(ElementType::Tri3)
    } else if !elem_types.is_empty() {
        elem_types[0].boundary_type().unwrap_or(ElementType::Tri3)
    } else {
        ElementType::Tri3
    };
    let use_mixed_faces = !bdr_types.is_empty() && !bdr_types.iter().all(|&t| t == bdr_types[0]);
    let face_types_opt = if use_mixed_faces { Some(bdr_types) } else { None };

    if dim == 2 {
        let mesh = SimplexMesh {
            coords,
            conn: flat_elem,
            elem_tags,
            elem_type: uniform_type.unwrap_or(ElementType::Tri3),
            face_conn: flat_face,
                face_tags: face_tags.into_iter().map(|t| t as fem_mesh::BoundaryTag).collect(),
                face_type: face_type_from_file,
                elem_types: if uniform_type.is_some() { None } else { Some(elem_types) },
                elem_offsets: None,
                face_types: face_types_opt.clone(),
                face_offsets: None,
                face_to_elem: None,
                edge_conn: vec![], edge_to_elem: vec![],
            };
            Ok(MfemFile { mesh2d: Some(mesh), mesh3d: None })
        } else {
        let mesh = SimplexMesh {
            coords,
            conn: flat_elem,
            elem_tags,
            elem_type: uniform_type.unwrap_or(ElementType::Tet4),
            face_conn: flat_face,
            face_tags: face_tags.into_iter().map(|t| t as fem_mesh::BoundaryTag).collect(),
            face_type: face_type_from_file,
            elem_types: if uniform_type.is_some() { None } else { Some(elem_types) },
            elem_offsets: None,
            face_types: face_types_opt,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![], edge_to_elem: vec![],
    };
        Ok(MfemFile { mesh2d: None, mesh3d: Some(mesh) })
    }
}

/// Convenience: read MFEM file from disk.
pub fn read_mfem_file(path: impl AsRef<std::path::Path>) -> FemResult<MfemFile> {
    read_mfem(std::fs::File::open(path)?)
}

fn skip_comment(mut line: &str) -> &str {
    loop {
        let trimmed = line.trim();
        if trimmed.starts_with('#') || trimmed.is_empty() { return ""; }
        if let Some(idx) = trimmed.find('#') { return trimmed[..idx].trim(); }
        return trimmed;
    }
}

fn read_line(r: &mut impl BufRead) -> FemResult<String> {
    let mut line = String::new();
    loop {
        line.clear();
        if r.read_line(&mut line)? == 0 {
            return Err(FemError::Mesh("MFEM: unexpected EOF".into()));
        }
        let t = skip_comment(&line);
        if !t.is_empty() { return Ok(t.to_owned()); }
    }
}

fn read_uint(r: &mut impl BufRead) -> FemResult<usize> {
    let l = read_line(r)?;
    l.parse().map_err(|_| FemError::Mesh(format!("MFEM: expected integer, got: {l}")))
}

fn read_uint_line(r: &mut impl BufRead) -> FemResult<Vec<usize>> {
    let l = read_line(r)?;
    l.split_whitespace().map(|s| s.parse().map_err(|_| FemError::Mesh(format!("MFEM: bad int: {s}")))).collect()
}

fn read_f64_line(r: &mut impl BufRead) -> FemResult<Vec<f64>> {
    let l = read_line(r)?;
    l.split_whitespace().map(|s| s.parse().map_err(|_| FemError::Mesh(format!("MFEM: bad float: {s}")))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_2d_square() {
        let data = "\
MFEM mesh v1.0

dimension
2

elements\n1\n3 1 1 2 3 4\n\nboundary\n4\n1 1 1 2\n1 1 2 3\n1 1 3 4\n1 1 4 1\n\nvertices\n4\n2\n0.0 0.0\n1.0 0.0\n1.0 1.0\n0.0 1.0
";
        let mfem = read_mfem(data.as_bytes()).unwrap();
        let mesh = mfem.mesh2d.unwrap();
        assert_eq!(mesh.n_nodes(), 4);
        assert_eq!(mesh.n_elems(), 1);
        assert_eq!(mesh.n_faces(), 4);
    }

    #[test]
    fn read_3d_cube() {
        let data = "\
MFEM mesh v1.0

dimension
3

elements\n1\n5 1 1 2 3 4 5 6 7 8\n\nboundary\n6\n3 1 1 2 3 4\n3 1 5 6 7 8\n3 1 1 2 6 5\n3 1 3 4 8 7\n3 1 1 4 8 5\n3 1 2 3 7 6\n\nvertices\n8\n3\n0.0 0.0 0.0\n1.0 0.0 0.0\n1.0 1.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\n1.0 0.0 1.0\n1.0 1.0 1.0\n0.0 1.0 1.0
";
        let mfem = read_mfem(data.as_bytes()).unwrap();
        let mesh = mfem.mesh3d.unwrap();
        assert_eq!(mesh.n_nodes(), 8);
        assert_eq!(mesh.n_elems(), 1);
        assert_eq!(mesh.n_faces(), 6);
    }
}
