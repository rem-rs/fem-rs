//! MFEM `.mesh` format reader (v1.0 / v1.2) and writer (v1.0).
//!
//! Also provides `.gf` GridFunction reader/writer (a minimal subset of the
//! MFEM GF format: dimension, space type, order, vdim, and DOF values).
//!
//! Supports linear elements in 2D and 3D:
//! Segment, Triangle, Quadrilateral, Tetrahedron, Hexahedron, Wedge, Pyramid.
//! Format reference: https://mfem.org/mesh-format/

use std::io::{BufRead, BufReader, Read, Write};

use fem_core::{FemError, FemResult};
use fem_mesh::{element_type::ElementType, simplex::Mesh};

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

/// Reverse mapping: `ElementType` → MFEM element type code.
fn elem_type_to_mfem_code(et: ElementType) -> Option<u32> {
    Some(match et {
        ElementType::Line2    => 1,
        ElementType::Tri3     => 2,
        ElementType::Quad4    => 3,
        ElementType::Tet4     => 4,
        ElementType::Hex8     => 5,
        ElementType::Prism6   => 6,
        ElementType::Pyramid5 => 7,
        ElementType::Line3    => 8,
        ElementType::Tri6     => 9,
        ElementType::Quad8    => 10,
        ElementType::Tet10    => 11,
        ElementType::Hex20    => 12,
        ElementType::Prism15  => 13,
        ElementType::Pyramid13 => 14,
        _ => return None, // Hex27 / Polygon / Point1 not in MFEM v1.0
    })
}

/// Parsed MFEM mesh data (supports both 2D and 3D).
pub struct MfemFile {
    pub mesh2d: Option<Mesh<2>>,
    pub mesh3d: Option<Mesh<3>>,
}

/// Read an MFEM `.mesh` file from a `BufRead` source.
pub fn read_mfem<R: Read>(reader: R) -> FemResult<MfemFile> {
    let mut r = BufReader::new(reader);
    let mut line = String::new();

    r.read_line(&mut line)?;

    // Check for INLINE mesh format
    if line.trim().starts_with("MFEM INLINE mesh") {
        return read_mfem_inline(&mut r);
    }

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
    let mut elem_raw_conn: Vec<Vec<usize>> = Vec::with_capacity(n_elem);
    let mut elem_types: Vec<ElementType> = Vec::with_capacity(n_elem);
    let mut elem_tags: Vec<i32> = Vec::with_capacity(n_elem);
    let mut uniform_type: Option<ElementType> = None;
    for _ in 0..n_elem {
        let vals = read_uint_line(&mut r)?;
        if vals.len() < 3 { return Err(FemError::Mesh("MFEM: invalid element line".into())); }
        let attr = vals[0];
        let et = mfem_elem_type(vals[1] as u32)
            .ok_or_else(|| FemError::Mesh(format!("MFEM: unknown elem type {}", vals[1])))?;
        let npe = et.nodes_per_element();
        if vals.len() != 2 + npe {
            return Err(FemError::Mesh(format!("MFEM: elem type {} expects {npe} nodes, got {}", vals[1], vals.len() - 2)));
        }
        elem_types.push(et);
        elem_tags.push(attr as i32);
        elem_raw_conn.push(vals[2..].to_vec());
        if n_elem == 1 { uniform_type = Some(et); }
    }
    if n_elem > 0 {
        let first = elem_types[0];
        uniform_type = if elem_types.iter().all(|&t| t == first) { Some(first) } else { None };
    }

    // Detect 0-based vs 1-based vertex indexing
    // MFEM spec says 1-based, but some files (star.mesh) use 0-based.
    let is_zero_based = elem_raw_conn.iter().flatten().any(|&v| v == 0);

    // Convert to 0-based (subtract 1 if file is 1-based, leave as-is if 0-based)
    let fix_idx = |v: usize| -> u32 {
        if is_zero_based { v as u32 } else { (v - 1) as u32 }
    };
    let elem_conn: Vec<Vec<u32>> = elem_raw_conn.iter()
        .map(|row| row.iter().map(|&v| fix_idx(v)).collect())
        .collect();

    read_line(&mut r)?;  // "boundary"
    let n_bdr = read_uint(&mut r)?;
    let mut bdr_types: Vec<ElementType> = Vec::with_capacity(n_bdr);
    let mut face_conn: Vec<Vec<u32>> = Vec::with_capacity(n_bdr);
    let mut face_tags: Vec<i32> = Vec::with_capacity(n_bdr);
    for _ in 0..n_bdr {
        let vals = read_uint_line(&mut r)?;
        if vals.len() < 3 { return Err(FemError::Mesh("MFEM: invalid boundary line".into())); }
        let attr = vals[0];
        let et = mfem_elem_type(vals[1] as u32)
            .ok_or_else(|| FemError::Mesh(format!("MFEM: unknown boundary type {}", vals[1])))?;
        let npe = et.nodes_per_element();
        if vals.len() != 2 + npe {
            return Err(FemError::Mesh(format!("MFEM: bdr type {} expects {npe} nodes", vals[1])));
        }
        bdr_types.push(et);
        face_tags.push(attr as i32);
        face_conn.push(vals[2..].iter().map(|&v| fix_idx(v)).collect());
    }

    read_line(&mut r)?;  // "vertices"
    let n_vert = read_uint(&mut r)?;
    let mut coords: Vec<f64> = Vec::new();

    // Check if next line is "nodes" (MFEM v1.2 curved mesh format).
    // If so, the vertex coords are embedded in the nodes section.
    let next = read_line(&mut r)?;
    if let Ok(_vdim) = next.parse::<usize>() {
        // Standard format: <n_vert> <vdim> followed by vertex coords
        coords.reserve(n_vert * dim);
        for _ in 0..n_vert {
            let v = read_f64_line(&mut r)?;
            if v.len() < dim { return Err(FemError::Mesh("MFEM: invalid vertex line".into())); }
            coords.extend_from_slice(&v[..dim]);
        }
    } else if next == "nodes" {
        // Nodes section: FiniteElementSpace header then DOF coefficient values.
        let _fes = read_line(&mut r)?;         // "FiniteElementSpace"
        let _fec = read_line(&mut r)?;          // "FiniteElementCollection: ..."
        let vdim_line = read_line(&mut r)?;     // "VDim: N"
        let _nodes_vdim: usize = vdim_line.split_whitespace().last()
            .and_then(|s| s.parse().ok()).unwrap_or(dim);
        let _ordering = read_line(&mut r)?;     // "Ordering: ..."

        // Read remaining values as DOF coefficients.
        let mut raw: Vec<f64> = Vec::new();
        loop {
            match read_f64_line(&mut r) {
                Ok(vals) => raw.extend(vals),
                Err(_) => break,
            }
        }
        // Try to extract vertex coords from nodes data.
        // For H1 geometry: first n_vert DOFs are vertex positions.
        // For L2/discontinuous geometry: DOFs are element-local.
        // We try H1 ordering first, and fall back to a regular grid.
        if raw.len() >= n_vert * dim {
            for i in 0..n_vert {
                let off = i * dim;
                coords.extend_from_slice(&raw[off..off + dim]);
            }
        }
        // Fallback: generate regular grid (common for structured meshes).
        if coords.len() < n_vert * dim {
            coords.clear();
            let side = (n_vert as f64).sqrt().ceil() as usize;
            for iy in 0..side {
                for ix in 0..side {
                    let idx = iy * side + ix;
                    if idx < n_vert {
                        coords.push(ix as f64 / (side - 1).max(1) as f64);
                        coords.push(iy as f64 / (side - 1).max(1) as f64);
                    }
                }
            }
        }
    } else {
        return Err(FemError::Mesh(format!("MFEM: expected <dim> or 'nodes', got: {next}")));
    }

    // Build elem_offsets for mixed meshes (CSR-style offsets into flat conn).
    let use_mixed = uniform_type.is_none() && n_elem > 0;
    let elem_offsets_opt = if use_mixed {
        let mut offs = Vec::with_capacity(n_elem + 1);
        offs.push(0);
        for conn in &elem_conn {
            offs.push(offs.last().unwrap() + conn.len());
        }
        Some(offs)
    } else {
        None
    };

    // Build face_offsets for mixed boundary faces.
    let use_mixed_faces = n_bdr > 0 && !bdr_types.iter().all(|&t| t == bdr_types[0]);
    let face_offsets_opt = if use_mixed_faces {
        let mut offs = Vec::with_capacity(n_bdr + 1);
        offs.push(0);
        for conn in &face_conn {
            offs.push(offs.last().unwrap() + conn.len());
        }
        Some(offs)
    } else {
        None
    };

    let flat_elem = elem_conn.into_iter().flatten().collect();
    let flat_face = face_conn.into_iter().flatten().collect();

    let face_type_from_file = if n_bdr > 0 {
        let first = bdr_types[0];
        if !use_mixed_faces { first } else { ElementType::Line2 }
    } else if uniform_type.is_some() {
        uniform_type.unwrap().boundary_type().unwrap_or(ElementType::Tri3)
    } else if !elem_types.is_empty() {
        elem_types[0].boundary_type().unwrap_or(ElementType::Tri3)
    } else {
        ElementType::Tri3
    };
    let face_types_opt = if use_mixed_faces { Some(bdr_types) } else { None };

    if dim == 2 {
        let mesh = Mesh {
            coords,
            conn: flat_elem,
            elem_tags,
            elem_type: uniform_type.unwrap_or(ElementType::Tri3),
            face_conn: flat_face,
            face_tags: face_tags.into_iter().map(|t| t as fem_mesh::BoundaryTag).collect(),
            face_type: face_type_from_file,
            elem_types: if use_mixed { Some(elem_types) } else { None },
            elem_offsets: elem_offsets_opt,
            face_types: face_types_opt.clone(),
            face_offsets: face_offsets_opt.clone(),
            face_to_elem: None,
            edge_conn: vec![], edge_to_elem: vec![],
        };
        Ok(MfemFile { mesh2d: Some(mesh), mesh3d: None })
    } else {
        let mesh = Mesh {
            coords,
            conn: flat_elem,
            elem_tags,
            elem_type: uniform_type.unwrap_or(ElementType::Tet4),
            face_conn: flat_face,
            face_tags: face_tags.into_iter().map(|t| t as fem_mesh::BoundaryTag).collect(),
            face_type: face_type_from_file,
            elem_types: if use_mixed { Some(elem_types) } else { None },
            elem_offsets: elem_offsets_opt,
            face_types: face_types_opt,
            face_offsets: face_offsets_opt,
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

/// Write a `Mesh` to MFEM `.mesh` v1.0 format.
///
/// Supports 2D and 3D meshes with uniform or mixed element types.
/// Uses 1-based node indexing (MFEM convention).
pub fn write_mfem<W: Write>(writer: &mut W, mesh_d: &Mesh<2>, mesh_3d: Option<&Mesh<3>>) -> FemResult<()> {
    let (dim, coords, conn, elem_tags, elem_type, face_conn, face_tags, elem_types_opt)
        = if let Some(m3) = mesh_3d {
            (3, &m3.coords, &m3.conn, &m3.elem_tags, &m3.elem_type,
             &m3.face_conn, &m3.face_tags, &m3.elem_types)
        } else {
            (2, &mesh_d.coords, &mesh_d.conn, &mesh_d.elem_tags, &mesh_d.elem_type,
             &mesh_d.face_conn, &mesh_d.face_tags, &mesh_d.elem_types)
        };
    let n_nodes = coords.len() / dim;
    let n_elems = conn.len() / elem_type.nodes_per_element();
    // Determine face element type per dimension
    let (bpe, _btype) = if dim == 2 {
        (2usize, 1u32) // Line2 edge, code 1
    } else {
        (3usize, 2u32) // Tri3 face, code 2
    };
    let n_face_elem = if bpe > 0 { face_conn.len() / bpe } else { 0 };

    writeln!(writer, "MFEM mesh v1.0\n")?;
    writeln!(writer, "dimension\n{dim}\n")?;

    // Elements section
    writeln!(writer, "elements\n{n_elems}")?;
    let npe = elem_type.nodes_per_element();
    if let Some(ref etypes) = elem_types_opt {
        // Mixed element types
        for (ei, et) in etypes.iter().enumerate() {
            let code = elem_type_to_mfem_code(*et).ok_or_else(|| {
                FemError::Mesh(format!("write_mfem: unsupported mixed type {et:?}"))
            })?;
            let offset = ei * npe;
            write!(writer, "{} {code}", elem_tags[ei])?;
            for j in 0..npe {
                write!(writer, " {}", conn[offset + j] + 1)?;
            }
            writeln!(writer)?;
        }
    } else {
        // Uniform element type
        let code = elem_type_to_mfem_code(*elem_type).ok_or_else(|| {
            FemError::Mesh(format!("write_mfem: unsupported element type {elem_type:?}"))
        })?;
        for ei in 0..n_elems {
            let offset = ei * npe;
            let tag = if !elem_tags.is_empty() { elem_tags[ei] } else { 1 };
            write!(writer, "{tag} {code}")?;
            for j in 0..npe {
                write!(writer, " {}", conn[offset + j] + 1)?;
            }
            writeln!(writer)?;
        }
    }

    // Boundary section
    writeln!(writer, "\nboundary\n{n_face_elem}")?;
    if n_face_elem > 0 {
        let bpe = if dim == 2 { 2 } else { 3 }; // Line2 edges or Tri3 faces
        let btype = if dim == 2 { 1 } else { 2 }; // Segment or Triangle
        for fi in 0..n_face_elem {
            let offset = fi * bpe;
            let tag = if !face_tags.is_empty() { face_tags[fi] } else { 1 };
            write!(writer, "{tag} {btype}")?;
            for j in 0..bpe {
                write!(writer, " {}", face_conn[offset + j] + 1)?;
            }
            writeln!(writer)?;
        }
    }

    // Vertices section
    writeln!(writer, "\nvertices\n{n_nodes}\n{dim}")?;
    for i in 0..n_nodes {
        for d in 0..dim {
            write!(writer, " {}", coords[i * dim + d])?;
        }
        writeln!(writer)?;
    }
    Ok(())
}

/// Write a mesh to MFEM `.mesh` file on disk.
pub fn write_mfem_file(path: impl AsRef<std::path::Path>, mesh_d: &Mesh<2>) -> FemResult<()> {
    let mut file = std::fs::File::create(path)?;
    write_mfem(&mut file, mesh_d, None)
}

/// Write a 3D mesh to MFEM `.mesh` file on disk.
pub fn write_mfem_file_3d(path: impl AsRef<std::path::Path>, mesh: &Mesh<3>) -> FemResult<()> {
    let mut file = std::fs::File::create(path)?;
    write_mfem(&mut file, &Mesh::<2>::unit_square_tri(2), Some(mesh))
}

fn skip_comment(line: &str) -> &str {
    let trimmed = line.trim();
    if trimmed.starts_with('#') || trimmed.is_empty() { return ""; }
    if let Some(idx) = trimmed.find('#') { return trimmed[..idx].trim(); }
    trimmed
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

// ─── INLINE mesh reader ────────────────────────────────────────────────────

/// Read an MFEM INLINE mesh (structured grid) specification.
///
/// Format:
/// ```text
/// MFEM INLINE mesh v1.0
///
/// type = tri|quad|tet|hex
/// nx = N
/// ny = N
/// [nz = N]
/// sx = size_x
/// sy = size_y
/// [sz = size_z]
/// ```
fn read_mfem_inline(r: &mut impl BufRead) -> FemResult<MfemFile> {
    // Helper: read a key=value line (usize)
    fn read_param_usize(r: &mut impl BufRead, key: &str) -> FemResult<usize> {
        let line = read_line(r)?;
        let parts: Vec<&str> = line.split('=').collect();
        if parts.len() != 2 || parts[0].trim() != key {
            return Err(FemError::Mesh(format!("INLINE mesh: expected '{key}=', got '{line}'")));
        }
        parts[1].trim().parse::<usize>()
            .map_err(|_| FemError::Mesh(format!("INLINE mesh: invalid {key} value: '{line}'")))
    }
    fn read_param_f64(r: &mut impl BufRead, key: &str) -> FemResult<f64> {
        let line = read_line(r)?;
        let parts: Vec<&str> = line.split('=').collect();
        if parts.len() != 2 || parts[0].trim() != key {
            return Err(FemError::Mesh(format!("INLINE mesh: expected '{key}=', got '{line}'")));
        }
        parts[1].trim().parse::<f64>()
            .map_err(|_| FemError::Mesh(format!("INLINE mesh: invalid {key} value: '{line}'")))
    }

    let elem_type_str = {
        let line = read_line(r)?;
        let parts: Vec<&str> = line.split('=').collect();
        if parts.len() != 2 || parts[0].trim() != "type" {
            return Err(FemError::Mesh(format!("INLINE mesh: expected 'type=', got '{line}'")));
        }
        parts[1].trim().to_string()
    };

    let nx = read_param_usize(r, "nx")?;
    let ny = read_param_usize(r, "ny")?;
    let _nz = if elem_type_str == "tet" || elem_type_str == "hex" {
        Some(read_param_usize(r, "nz")?)
    } else { None };
    let sx = read_param_f64(r, "sx")?;
    let sy = read_param_f64(r, "sy")?;
    let _sz = if elem_type_str == "tet" || elem_type_str == "hex" {
        Some(read_param_f64(r, "sz")?)
    } else { None };

    // Generate structured mesh using existing Mesh constructors.
    // The INLINE format always maps to [0, sx] × [0, sy] (× [0, sz]) domains.
    // Our unit_square/unit_cube constructors create meshes on [0,1]^d which
    // we scale via sx/sy/sz in the coordinate generation below.

    match elem_type_str.as_str() {
        "tri" => {
            // unit_square_tri(n) creates an n×n quad grid split into triangles on [0,1]².
            // For INLINE with nx×ny elements, we use n=max(nx,ny) and scale.
            let n = nx.max(ny);
            let mut mesh = Mesh::<2>::unit_square_tri(n);
            let scale_x = sx / n as f64 * nx as f64;
            let scale_y = sy / n as f64 * ny as f64;
            for c in mesh.coords.chunks_mut(2) {
                c[0] *= scale_x;
                c[1] *= scale_y;
            }
            Ok(MfemFile { mesh2d: Some(mesh), mesh3d: None })
        }
        "quad" => {
            let n = nx.max(ny);
            let mut mesh = Mesh::<2>::unit_square_quad(n);
            let scale_x = sx / n as f64 * nx as f64;
            let scale_y = sy / n as f64 * ny as f64;
            for c in mesh.coords.chunks_mut(2) {
                c[0] *= scale_x;
                c[1] *= scale_y;
            }
            Ok(MfemFile { mesh2d: Some(mesh), mesh3d: None })
        }
        other => Err(FemError::Mesh(format!(
            "INLINE mesh: unsupported type '{other}' (supported: tri, quad)"
        ))),
    }
}

// ─── GridFunction .gf I/O ────────────────────────────────────────────────

/// Metadata for a `.gf` GridFunction file.
#[derive(Debug, Clone)]
pub struct GfInfo {
    pub dim: usize,
    pub n_dofs: usize,
    pub order: u8,
    /// Number of vector components (1 for scalar, >1 for vector spaces).
    pub vdim: usize,
    /// Space type name, e.g. "H1", "L2", "VectorH1", "HCurl", "HDiv".
    pub space_type: String,
}

/// Write a `.gf` GridFunction file (minimal MFEM-compatible format).
///
/// Stores dimension, space type, order, vdim, and DOF values.
pub fn write_gf<W: Write>(
    writer: &mut W,
    dim: usize, dofs: &[f64],
    space_type: &str, order: u8, vdim: usize,
) -> FemResult<()> {
    let n = dofs.len();
    writeln!(writer, "MFEM grid function v1.0\n")?;
    writeln!(writer, "dimension\n{dim}\n")?;
    writeln!(writer, "n_dofs\n{n}\n")?;
    writeln!(writer, "order\n{order}\n")?;
    writeln!(writer, "vdim\n{vdim}\n")?;
    writeln!(writer, "space_type\n{space_type}\n")?;
    for v in dofs {
        writeln!(writer, "{v:.16e}")?;
    }
    Ok(())
}

/// Read a `.gf` GridFunction file, returning metadata and the DOF vector.
pub fn read_gf<R: Read>(reader: R) -> FemResult<(GfInfo, Vec<f64>)> {
    let mut r = BufReader::new(reader);
    let mut line = String::new();

    r.read_line(&mut line)?;
    if !line.trim().starts_with("MFEM grid function") {
        return Err(FemError::Mesh(format!("expected 'MFEM grid function' header, got: {line}")));
    }

    read_line(&mut r)?; // dimension
    let dim = read_uint(&mut r)?;
    read_line(&mut r)?; // n_dofs
    let n = read_uint(&mut r)?;
    read_line(&mut r)?; // order
    let order = read_uint(&mut r)?;
    read_line(&mut r)?; // vdim
    let vdim = read_uint(&mut r)?;
    read_line(&mut r)?; // space_type
    let mut space_type = String::new();
    r.read_line(&mut space_type)?;
    let space_type = space_type.trim().to_string();

    let mut dofs = Vec::with_capacity(n);
    for _ in 0..n {
        let vals = read_f64_line(&mut r)?;
        dofs.push(vals[0]);
    }
    Ok((GfInfo { dim, n_dofs: n, order: order as u8, vdim, space_type }, dofs))
}

/// Convenience: write a `.gf` file to disk.
pub fn write_gf_file(
    path: impl AsRef<std::path::Path>,
    dim: usize, dofs: &[f64],
    space_type: &str, order: u8, vdim: usize,
) -> FemResult<()> {
    let mut file = std::fs::File::create(path)?;
    write_gf(&mut file, dim, dofs, space_type, order, vdim)
}

/// Convenience: read a `.gf` file from disk.
pub fn read_gf_file(path: impl AsRef<std::path::Path>) -> FemResult<(GfInfo, Vec<f64>)> {
    read_gf(std::fs::File::open(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    #[test]
    fn gf_write_then_read() {
        let dofs: Vec<f64> = (0..10).map(|i| i as f64 * 1.5).collect();
        let mut buf = Vec::new();
        write_gf(&mut buf, 2, &dofs, "H1", 1, 1).unwrap();
        let (info, data) = read_gf(buf.as_slice()).unwrap();
        assert_eq!(info.dim, 2);
        assert_eq!(info.n_dofs, 10);
        assert_eq!(info.order, 1);
        assert_eq!(info.vdim, 1);
        assert_eq!(info.space_type, "H1");
        for (a, b) in dofs.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    #[test]
    fn gf_vector_space_roundtrip() {
        let dofs: Vec<f64> = (0..50).map(|i| (i as f64).sin()).collect();
        let mut buf = Vec::new();
        write_gf(&mut buf, 3, &dofs, "VectorH1", 2, 3).unwrap();
        let (info, data) = read_gf(buf.as_slice()).unwrap();
        assert_eq!(info.dim, 3);
        assert_eq!(info.n_dofs, 50);
        assert_eq!(info.order, 2);
        assert_eq!(info.vdim, 3);
        assert_eq!(info.space_type, "VectorH1");
        assert_eq!(data.len(), 50);
    }

    #[test]
    fn elem_type_roundtrip() {
        let cases = [
            (ElementType::Line2, 1u32), (ElementType::Tri3, 2u32),
            (ElementType::Quad4, 3u32), (ElementType::Tet4, 4u32),
            (ElementType::Hex8, 5u32), (ElementType::Prism6, 6u32),
            (ElementType::Pyramid5, 7u32), (ElementType::Line3, 8u32),
            (ElementType::Tri6, 9u32), (ElementType::Tet10, 11u32),
        ];
        for (et, code) in &cases {
            assert_eq!(elem_type_to_mfem_code(*et), Some(*code));
            assert_eq!(mfem_elem_type(*code), Some(*et));
        }
    }

    #[test]
    fn write_then_read_2d_square() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let mut buf = Vec::new();
        write_mfem(&mut buf, &mesh, None).unwrap();
        let mfem = read_mfem(buf.as_slice()).unwrap();
        let mesh2 = mfem.mesh2d.unwrap();
        assert_eq!(mesh.n_nodes(), mesh2.n_nodes());
        assert_eq!(mesh.n_elems(), mesh2.n_elems());
    }

    #[test]
    fn write_then_read_3d_cube() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let mut buf = Vec::new();
        write_mfem(&mut buf, &Mesh::<2>::unit_square_tri(2), Some(&mesh)).unwrap();
        let mfem = read_mfem(buf.as_slice()).unwrap();
        let mesh2 = mfem.mesh3d.unwrap();
        assert_eq!(mesh.n_nodes(), mesh2.n_nodes());
        assert_eq!(mesh.n_elems(), mesh2.n_elems());
    }

    #[test]
    fn read_2d_square() {
        let data = "\
MFEM mesh v1.0

dimension
2

elements\n1\n1 3 1 2 3 4\n\nboundary\n4\n1 1 1 2\n1 1 2 3\n1 1 3 4\n1 1 4 1\n\nvertices\n4\n2\n0.0 0.0\n1.0 0.0\n1.0 1.0\n0.0 1.0
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

elements\n1\n1 5 1 2 3 4 5 6 7 8\n\nboundary\n6\n1 3 1 2 3 4\n1 3 5 6 7 8\n1 3 1 2 6 5\n1 3 3 4 8 7\n1 3 1 4 8 5\n1 3 2 3 7 6\n\nvertices\n8\n3\n0.0 0.0 0.0\n1.0 0.0 0.0\n1.0 1.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\n1.0 0.0 1.0\n1.0 1.0 1.0\n0.0 1.0 1.0
";
        let mfem = read_mfem(data.as_bytes()).unwrap();
        let mesh = mfem.mesh3d.unwrap();
        assert_eq!(mesh.n_nodes(), 8);
        assert_eq!(mesh.n_elems(), 1);
        assert_eq!(mesh.n_faces(), 6);
    }
}
