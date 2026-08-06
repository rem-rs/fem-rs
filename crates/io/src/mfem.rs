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
use fem_mesh::{
    element_type::ElementType,
    simplex::{GeometryData, Mesh},
    topology::MeshTopology,
};

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

    let is_nurbs = line.trim().starts_with("MFEM NURBS mesh");

    if !is_nurbs && !line.trim().starts_with("MFEM mesh") {
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

    {
        let next = read_line(&mut r)?;  // "edges" or "vertices"
        if next.trim() == "edges" {
            let n_edges = read_uint(&mut r)?;
            for _ in 0..n_edges { read_uint_line(&mut r)?; }
            read_line(&mut r)?;  // "vertices"
        } // else already "vertices"
    }
    let n_vert = read_uint(&mut r)?;
    let mut coords: Vec<f64> = Vec::new();
    // Per-element high-order geometry (MFEM `nodes` section).  For L2
    // (discontinuous) node spaces each element owns `nodes_per_elem`
    // independent geometry nodes — this is how geometrically periodic meshes
    // (e.g. `periodic-square.mesh`) encode per-element geometry.
    let mut geometry: Option<GeometryData> = None;

    // Check if next line is "nodes" (MFEM v1.2 curved mesh format),
    // a dimension number (standard format), or a NURBS keyword (skip).
    let next = read_line(&mut r)?;
    if next.trim() == "knotvectors" || next.trim() == "knots" || next.trim().starts_with("FiniteElement") {
        // NURBS or IGA format — read through remaining sections to extract vertex coords.
        // The element/boundary/edges sections provide topology; NURBS data provides geometry.
        if is_nurbs {
            if next.trim() == "knotvectors" || next.trim() == "knots" {
                // Read knot vectors section
                let n_kv = read_uint(&mut r)?;
                for _ in 0..n_kv {
                    let _ = read_f64_line(&mut r)?;
                }
                // Read "weights" header then weight values until "FiniteElementSpace"
                let _weights_header = read_line(&mut r)?;
                loop {
                    let line = read_line(&mut r)?;
                    if line.starts_with("FiniteElementSpace") {
                        break;
                    }
                }
            }
            // else: already at "FiniteElementSpace" (next.trim() starts with it)

            // Read FiniteElementCollection line
            let _fec = read_line(&mut r)?;  // "FiniteElementCollection: NURBS<N>"
            // Read VDim line
            let vdim_line = read_line(&mut r)?;  // "VDim: N"
            let _vdim: usize = vdim_line.split_whitespace().last()
                .and_then(|s| s.parse().ok()).unwrap_or(dim);
            // Read Ordering line
            let _ordering = read_line(&mut r)?;  // "Ordering: 1"

            // Read remaining values as control point coordinates
            let mut raw: Vec<f64> = Vec::new();
            loop {
                match read_f64_line(&mut r) {
                    Ok(vals) => raw.extend(vals),
                    Err(_) => break,
                }
            }

            // Extract vertex coordinates: first n_vert * dim values
            if raw.len() >= n_vert * dim {
                for i in 0..n_vert {
                    let off = i * dim;
                    coords.extend_from_slice(&raw[off..off + dim]);
                }
            } else {
                // Fallback: generate a regular grid
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
        }
        // else: non-NURBS mesh with unexpected keyword — ignore, coords stays empty
    } else if let Ok(_vdim) = next.parse::<usize>() {
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
        let fec_line = read_line(&mut r)?;     // "FiniteElementCollection: ..."
        let vdim_line = read_line(&mut r)?;     // "VDim: N"
        let _nodes_vdim: usize = vdim_line.split_whitespace().last()
            .and_then(|s| s.parse().ok()).unwrap_or(dim);
        let _ordering = read_line(&mut r)?;     // "Ordering: ..."

        // Read remaining values as DOF coefficient values.
        let mut raw: Vec<f64> = Vec::new();
        loop {
            match read_f64_line(&mut r) {
                Ok(vals) => raw.extend(vals),
                Err(_) => break,
            }
        }
        let fec_name = fec_line.split(':').nth(1).unwrap_or("").trim().to_string();
        let is_l2_nodes = fec_name.starts_with("L2_");
        if is_l2_nodes && n_elem > 0 && raw.len() >= n_elem * dim {
            // Discontinuous (L2) geometry: every element owns an independent
            // set of `nodes_per_elem` geometry nodes (MFEM L2_T1_2D_P1 etc.).
            // This is how geometrically periodic meshes (periodic-square.mesh,
            // periodic-hexagon.mesh, ...) encode per-element geometry — the
            // same vertex index can map to different physical positions in
            // different elements, and the element-to-element face pairing is
            // done purely by (periodically identified) vertex indices.
            let npe = raw.len() / (n_elem * dim);
            if npe >= 2 && raw.len() % (n_elem * dim) == 0 {
                // 1) Folded vertex coordinates: for each vertex, take the
                //    position it has in the first element that references it.
                //    This mirrors MFEM's `Mesh::vertices` array (used only by
                //    the face transformations; element assembly uses the
                //    per-element geometry below).
                coords = vec![0.0_f64; n_vert * dim];
                for v in 0..n_vert {
                    'outer: for e in 0..n_elem {
                        for k in 0..elem_conn[e].len() {
                            if elem_conn[e][k] as usize == v {
                                // `k` is the vertex index in the element's
                                // connectivity (H1 order); the nodes section
                                // stores them in lexicographic (L2) order, so
                                // map k -> lex index (P1: swap 2<->3).
                                let kl = if npe == 4 && dim == 2 {
                                    match k {
                                        2 => 3,
                                        3 => 2,
                                        _ => k,
                                    }
                                } else {
                                    k
                                };
                                for c in 0..dim {
                                    coords[v * dim + c] = raw[(e * npe + kl) * dim + c];
                                }
                                break 'outer;
                            }
                        }
                    }
                }
                // 2) Per-element geometry table (non-shared nodes).  The node
                //    order matches the element connectivity (H1 vertex order:
                //    LL, LR, UR, UL), which is what the QuadQk assembly basis
                //    and the mesh topology expect.  The MFEM `nodes` section
                //    stores them in lexicographic (L2) order, so for P1 quad
                //    we swap the last two entries.
                let mut geo_conn: Vec<u32> = Vec::with_capacity(n_elem * npe);
                let mut geo_coords: Vec<f64> = Vec::with_capacity(n_elem * npe * dim);
                let perm: Vec<usize> = if npe == 4 && dim == 2 {
                    vec![0, 1, 3, 2]
                } else {
                    (0..npe).collect()
                };
                for e in 0..n_elem {
                    for i in 0..npe {
                        geo_conn.push((e * npe + i) as u32);
                        let k = perm[i];
                        for c in 0..dim {
                            geo_coords.push(raw[(e * npe + k) * dim + c]);
                        }
                    }
                }
                geometry = Some(GeometryData {
                    order: 1,
                    conn: geo_conn,
                    nodes_per_elem: npe,
                    coords: geo_coords,
                    n_nodes: n_elem * npe,
                });
            }
        } else if raw.len() >= n_vert * dim {
            // Continuous (H1) geometry: first n_vert DOFs are the vertex
            // positions.  Build unique coordinate list (dedup by rounded
            // values) for the remaining DOFs (edge/face/interior nodes).
            let tol = 1e-10;
            for chunk in raw.chunks(dim) {
                if chunk.len() < dim { break; }
                let mut dup = false;
                for j in (0..coords.len()).step_by(dim) {
                    let mut dist_sq = 0.0;
                    for c in 0..dim {
                        let d = coords[j + c] - chunk[c];
                        dist_sq += d * d;
                    }
                    if dist_sq < tol {
                        dup = true;
                        break;
                    }
                }
                if !dup {
                    for &v in chunk.iter().take(dim) {
                        coords.push(v);
                    }
                }
                if coords.len() >= n_vert * dim { break; }
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
            nc_vertex_view: None,
            geometry,
        };
        Ok(MfemFile { mesh2d: Some(mesh), mesh3d: None })
    } else {
        let mut mesh = Mesh {
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
            nc_vertex_view: None,
            geometry,
        };
        // MFEM's Mesh(filename, 1, 1) finalizes tetrahedral meshes with
        // refine=1 → MarkTetMeshForRefinement (vertex rotation so the longest
        // edge is (v0,v1)).  Apply the same so element/vertex numbering and
        // the GS-sweep order match MFEM bit-for-bit.
        fem_mesh::mark_tet_mesh_for_refinement(&mut mesh);
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
    let n_elems = if dim == 3 {
        mesh_3d.map_or(conn.len() / elem_type.nodes_per_element(), |m| m.n_elems())
    } else if let Some(ref offsets) = mesh_d.elem_offsets {
        offsets.len() - 1
    } else {
        conn.len() / elem_type.nodes_per_element()
    };
    let n_face_elem = if dim == 3 { mesh_3d.map_or(0, |m| m.n_faces()) }
        else { face_conn.len() / 2 };

    writeln!(writer, "MFEM mesh v1.0\n")?;
    writeln!(writer, "dimension\n{dim}\n")?;

    // Elements section
    writeln!(writer, "elements\n{n_elems}")?;
    let npe = elem_type.nodes_per_element();
    if let Some(ref etypes) = elem_types_opt {
        // Mixed element types - use elem_offsets if available, else uniform stride
        let offsets = if dim == 3 {
            mesh_3d.and_then(|m| m.elem_offsets.as_ref())
        } else {
            mesh_d.elem_offsets.as_ref()
        };
        for ei in 0..n_elems {
            let et = &etypes[ei];
            let code = elem_type_to_mfem_code(*et).ok_or_else(|| {
                FemError::Mesh(format!("write_mfem: unsupported mixed type {et:?}"))
            })?;
            let npe_local = et.nodes_per_element();
            let offset = offsets.map(|offs| offs[ei]).unwrap_or(ei * npe);
            write!(writer, "{} {code}", elem_tags[ei])?;
            for j in 0..npe_local {
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
    for fi in 0..n_face_elem as u32 {
        let (offset, nvf, btype) = if let Some(ref m3) = mesh_3d {
            let (off, nv) = if let Some(ref fo) = m3.face_offsets {
                (fo[fi as usize], fo[fi as usize + 1] - fo[fi as usize])
            } else {
                (fi as usize * 3, 3usize)
            };
            let code = if nv == 3 { 2u32 } else { 3u32 }; // 2=Triangle, 3=Quad
            (off, nv, code)
        } else {
            (fi as usize * 2, 2usize, 1u32)
        };
        let tag = if !face_tags.is_empty() { face_tags[fi as usize] } else { 1 };
        write!(writer, "{tag} {btype}")?;
        for j in 0..nvf {
            write!(writer, " {}", face_conn[offset + j] + 1)?;
        }
        writeln!(writer)?;
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

/// Write a 2D mesh with custom vertex coordinates (e.g. displaced nodes).
///
/// `coords` is interleaved `[x0, y0, x1, y1, ...]`, length `n_nodes × dim`.
/// Uses the mesh's topology (elements, boundaries) but replaces vertex positions.
pub fn write_mfem_file_with_coords(
    path: impl AsRef<std::path::Path>,
    mesh: &Mesh<2>,
    coords: &[f64],
) -> FemResult<()> {
    let mut displaced = mesh.clone();
    let n = coords.len().min(displaced.coords.len());
    displaced.coords[..n].copy_from_slice(&coords[..n]);
    write_mfem_file(path, &displaced)
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
            // MFEM's INLINE quad mesh uses Hilbert space-filling-curve element
            // ordering (ReadInlineMesh → Make2D(..., sfc_ordering=true) →
            // NCMesh::GridSfcOrdering2D), NOT row-major.  The element
            // numbering must match for bit-identical assembly/GS-sweep order.
            let nxv = nx + 1;
            let nyv = ny + 1;
            let mut coords = Vec::with_capacity(nxv * nyv * 2);
            for j in 0..nyv {
                for i in 0..nxv {
                    coords.push(i as f64 / nx as f64 * sx);
                    coords.push(j as f64 / ny as f64 * sy);
                }
            }
            let mut sfc: Vec<(i32, i32)> = Vec::new();
            hilbert_sfc_2d(0, 0, nx as i32, 0, 0, ny as i32, &mut sfc);
            let id = |x: i32, y: i32| (y * nxv as i32 + x) as u32;
            let mut conn = Vec::with_capacity(sfc.len() * 4);
            let mut elem_tags = Vec::with_capacity(sfc.len());
            for &(i, j) in &sfc {
                conn.extend([id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)]);
                elem_tags.push(1);
            }
            // Boundary segments (MFEM Make2D): bottom attr 1, right attr 2,
            // top attr 3, left attr 4 — directions follow MFEM.
            let mut face_conn = Vec::with_capacity(2 * (nx + ny) * 2);
            let mut face_tags = Vec::with_capacity(2 * (nx + ny));
            for i in 0..nx {
                face_conn.extend([id(i as i32, 0), id(i as i32 + 1, 0)]);
                face_tags.push(1);
            }
            for j in 0..ny {
                face_conn.extend([id(nx as i32, j as i32), id(nx as i32, j as i32 + 1)]);
                face_tags.push(2);
            }
            for i in 0..nx {
                face_conn.extend([id(i as i32 + 1, ny as i32), id(i as i32, ny as i32)]);
                face_tags.push(3);
            }
            for j in 0..ny {
                face_conn.extend([id(0, j as i32 + 1), id(0, j as i32)]);
                face_tags.push(4);
            }
            let mesh = Mesh::uniform(
                coords, conn, elem_tags, ElementType::Quad4,
                face_conn, face_tags, ElementType::Line2,
            );
            Ok(MfemFile { mesh2d: Some(mesh), mesh3d: None })
        }
        "hex" => {
            // MFEM INLINE hex: Make3D(nx,ny,nz, HEX, sx,sy,sz, sfc_ordering=true)
            // — elements follow the 3-D Hilbert SFC (NCMesh::GridSfcOrdering3D),
            // NOT row-major.  The old `unit_cube_hex` row-major order misnumbered
            // elements (elem1 = x+1 vs MFEM z+1), which scrambled the RT0/ND
            // face-DOF numbering (ex22 3D p2: Re error 3.3× off).
            let nz = _nz.unwrap_or(1);
            let nxv = nx as i32 + 1;
            let nyv = ny as i32 + 1;
            let nzv = nz as i32 + 1;
            let mut coords = Vec::with_capacity((nxv * nyv * nzv) as usize * 3);
            for k in 0..nzv {
                for j in 0..nyv {
                    for i in 0..nxv {
                        coords.push(i as f64 / nx as f64 * sx);
                        coords.push(j as f64 / ny as f64 * sy);
                        coords.push(k as f64 / nz as f64 * _sz.unwrap_or(1.0));
                    }
                }
            }
            let vtx = |x: i32, y: i32, z: i32| (x + (y + z * nyv) * nxv) as u32;
            let sfc = grid_sfc_ordering_3d(nx, ny, nz);
            let mut conn = Vec::with_capacity(sfc.len() * 8);
            let mut elem_tags = Vec::with_capacity(sfc.len());
            for &(x, y, z) in &sfc {
                conn.extend([
                    vtx(x, y, z), vtx(x + 1, y, z), vtx(x + 1, y + 1, z), vtx(x, y + 1, z),
                    vtx(x, y, z + 1), vtx(x + 1, y, z + 1), vtx(x + 1, y + 1, z + 1), vtx(x, y + 1, z + 1),
                ]);
                elem_tags.push(1);
            }
            // Boundary faces (MFEM Make3D attr order):
            // bottom 1, front 2, right 3, back 4, left 5, top 6.
            let mut face_conn = Vec::with_capacity(2 * 4 * (nx * ny + ny * nz + nx * nz));
            let mut face_tags = Vec::with_capacity(2 * 4 * (nx * ny + ny * nz + nx * nz));
            let mut quad = |f: [u32; 4], tag: i32| { face_conn.extend_from_slice(&f); face_tags.push(tag); };
            for y in 0..ny as i32 {
                for x in 0..nx as i32 {
                    quad([vtx(x, y, 0), vtx(x, y + 1, 0), vtx(x + 1, y + 1, 0), vtx(x + 1, y, 0)], 1);
                }
            }
            for y in 0..ny as i32 {
                for x in 0..nx as i32 {
                    quad([vtx(x, y, nz as i32), vtx(x + 1, y, nz as i32), vtx(x + 1, y + 1, nz as i32), vtx(x, y + 1, nz as i32)], 6);
                }
            }
            for z in 0..nz as i32 {
                for y in 0..ny as i32 {
                    quad([vtx(0, y, z), vtx(0, y, z + 1), vtx(0, y + 1, z + 1), vtx(0, y + 1, z)], 5);
                }
            }
            for z in 0..nz as i32 {
                for y in 0..ny as i32 {
                    quad([vtx(nx as i32, y, z), vtx(nx as i32, y + 1, z), vtx(nx as i32, y + 1, z + 1), vtx(nx as i32, y, z + 1)], 3);
                }
            }
            for z in 0..nz as i32 {
                for x in 0..nx as i32 {
                    quad([vtx(x, 0, z), vtx(x + 1, 0, z), vtx(x + 1, 0, z + 1), vtx(x, 0, z + 1)], 2);
                }
            }
            for z in 0..nz as i32 {
                for x in 0..nx as i32 {
                    quad([vtx(x, ny as i32, z), vtx(x, ny as i32, z + 1), vtx(x + 1, ny as i32, z + 1), vtx(x + 1, ny as i32, z)], 4);
                }
            }
            let mesh = Mesh::<3>::uniform(
                coords, conn, elem_tags, ElementType::Hex8,
                face_conn, face_tags, ElementType::Quad4,
            );
            Ok(MfemFile { mesh2d: None, mesh3d: Some(mesh) })
        }
        "tet" => {
            let n = nx.max(ny).max(_nz.unwrap_or(1));
            let mut mesh = Mesh::<3>::unit_cube_tet(n);
            let scale_x = sx / n as f64 * nx as f64;
            let scale_y = sy / n as f64 * ny as f64;
            let scale_z = _sz.unwrap_or(1.0) / n as f64 * _nz.unwrap_or(1) as f64;
            for c in mesh.coords.chunks_mut(3) {
                c[0] *= scale_x;
                c[1] *= scale_y;
                c[2] *= scale_z;
            }
            Ok(MfemFile { mesh2d: None, mesh3d: Some(mesh) })
        }
        other => Err(FemError::Mesh(format!(
            "INLINE mesh: unsupported type '{other}' (supported: tri, quad, hex, tet)"
        ))),
    }
}

/// Sign function (MFEM ncmesh.cpp `sgn`).
fn sfc_sgn(x: i32) -> i32 {
    if x < 0 { -1 } else if x > 0 { 1 } else { 0 }
}

/// Hilbert space-filling curve in 2-D — 1:1 port of MFEM's
/// `NCMesh::HilbertSfc2D` (ncmesh.cpp).  Appends `(x, y)` grid coordinates
/// in Hilbert-curve order; used by `GridSfcOrdering2D` for INLINE quad meshes.
fn hilbert_sfc_2d(
    x: i32, y: i32, ax: i32, ay: i32, bx: i32, by: i32,
    coords: &mut Vec<(i32, i32)>,
) {
    let w = (ax + ay).abs();
    let h = (bx + by).abs();
    let dax = sfc_sgn(ax);
    let day = sfc_sgn(ay);
    let dbx = sfc_sgn(bx);
    let dby = sfc_sgn(by);

    if h == 1 {
        // trivial row fill
        let (mut x, mut y) = (x, y);
        for _ in 0..w {
            coords.push((x, y));
            x += dax;
            y += day;
        }
        return;
    }
    if w == 1 {
        // trivial column fill
        let (mut x, mut y) = (x, y);
        for _ in 0..h {
            coords.push((x, y));
            x += dbx;
            y += dby;
        }
        return;
    }

    let mut ax2 = ax / 2;
    let mut ay2 = ay / 2;
    let mut bx2 = bx / 2;
    let mut by2 = by / 2;
    let w2 = (ax2 + ay2).abs();
    let h2 = (bx2 + by2).abs();

    if 2 * w > 3 * h {
        // long case: split in two parts only
        if (w2 & 1) != 0 && w > 2 {
            ax2 += dax;
            ay2 += day; // prefer even steps
        }
        hilbert_sfc_2d(x, y, ax2, ay2, bx, by, coords);
        hilbert_sfc_2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by, coords);
    } else {
        // standard case: one step up, one long horizontal step, one step down
        if (h2 & 1) != 0 && h > 2 {
            bx2 += dbx;
            by2 += dby; // prefer even steps
        }
        hilbert_sfc_2d(x, y, bx2, by2, ax2, ay2, coords);
        hilbert_sfc_2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2, coords);
        hilbert_sfc_2d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            -bx2, -by2,
            -(ax - ax2), -(ay - ay2),
            coords,
        );
    }
}

/// Hilbert space-filling curve in 3-D — 1:1 port of MFEM's
/// `NCMesh::HilbertSfc3D` (ncmesh.cpp).  Appends `(x, y, z)` grid
/// coordinates in Hilbert-curve order.
fn hilbert_sfc_3d(
    x: i32, y: i32, z: i32,
    ax: i32, ay: i32, az: i32,
    bx: i32, by: i32, bz: i32,
    cx: i32, cy: i32, cz: i32,
    coords: &mut Vec<(i32, i32, i32)>,
) {
    let w = (ax + ay + az).abs();
    let h = (bx + by + bz).abs();
    let d = (cx + cy + cz).abs();
    let dax = sfc_sgn(ax); let day = sfc_sgn(ay); let daz = sfc_sgn(az);
    let dbx = sfc_sgn(bx); let dby = sfc_sgn(by); let dbz = sfc_sgn(bz);
    let dcx = sfc_sgn(cx); let dcy = sfc_sgn(cy); let dcz = sfc_sgn(cz);

    // trivial row/column fills
    if h == 1 && d == 1 {
        let (mut x, mut y, mut z) = (x, y, z);
        for _ in 0..w {
            coords.push((x, y, z));
            x += dax; y += day; z += daz;
        }
        return;
    }
    if w == 1 && d == 1 {
        let (mut x, mut y, mut z) = (x, y, z);
        for _ in 0..h {
            coords.push((x, y, z));
            x += dbx; y += dby; z += dbz;
        }
        return;
    }
    if w == 1 && h == 1 {
        let (mut x, mut y, mut z) = (x, y, z);
        for _ in 0..d {
            coords.push((x, y, z));
            x += dcx; y += dcy; z += dcz;
        }
        return;
    }

    let mut ax2 = ax / 2; let mut ay2 = ay / 2; let mut az2 = az / 2;
    let mut bx2 = bx / 2; let mut by2 = by / 2; let mut bz2 = bz / 2;
    let mut cx2 = cx / 2; let mut cy2 = cy / 2; let mut cz2 = cz / 2;
    let w2 = (ax2 + ay2 + az2).abs();
    let h2 = (bx2 + by2 + bz2).abs();
    let d2 = (cx2 + cy2 + cz2).abs();

    // prefer even steps
    if (w2 & 0x1) != 0 && w > 2 { ax2 += dax; ay2 += day; az2 += daz; }
    if (h2 & 0x1) != 0 && h > 2 { bx2 += dbx; by2 += dby; bz2 += dbz; }
    if (d2 & 0x1) != 0 && d > 2 { cx2 += dcx; cy2 += dcy; cz2 += dcz; }

    // wide case, split in w only
    if 2 * w > 3 * h && 2 * w > 3 * d {
        hilbert_sfc_3d(x, y, z, ax2, ay2, az2, bx, by, bz, cx, cy, cz, coords);
        hilbert_sfc_3d(x + ax2, y + ay2, z + az2, ax - ax2, ay - ay2, az - az2, bx, by, bz, cx, cy, cz, coords);
    }
    // do not split in d
    else if 3 * h > 4 * d {
        hilbert_sfc_3d(x, y, z, bx2, by2, bz2, cx, cy, cz, ax2, ay2, az2, coords);
        hilbert_sfc_3d(x + bx2, y + by2, z + bz2, ax, ay, az, bx - bx2, by - by2, bz - bz2, cx, cy, cz, coords);
        hilbert_sfc_3d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            z + (az - daz) + (bz2 - dbz),
            -bx2, -by2, -bz2,
            cx, cy, cz,
            -(ax - ax2), -(ay - ay2), -(az - az2),
            coords,
        );
    }
    // do not split in h
    else if 3 * d > 4 * h {
        hilbert_sfc_3d(x, y, z, cx2, cy2, cz2, ax2, ay2, az2, bx, by, bz, coords);
        hilbert_sfc_3d(x + cx2, y + cy2, z + cz2, ax, ay, az, bx, by, bz, cx - cx2, cy - cy2, cz - cz2, coords);
        hilbert_sfc_3d(
            x + (ax - dax) + (cx2 - dcx),
            y + (ay - day) + (cy2 - dcy),
            z + (az - daz) + (cz2 - dcz),
            -cx2, -cy2, -cz2,
            -(ax - ax2), -(ay - ay2), -(az - az2),
            bx, by, bz,
            coords,
        );
    }
    // regular case, split in all w/h/d
    else {
        hilbert_sfc_3d(x, y, z, bx2, by2, bz2, cx2, cy2, cz2, ax2, ay2, az2, coords);
        hilbert_sfc_3d(x + bx2, y + by2, z + bz2, cx, cy, cz, ax2, ay2, az2, bx - bx2, by - by2, bz - bz2, coords);
        hilbert_sfc_3d(
            x + (bx2 - dbx) + (cx - dcx),
            y + (by2 - dby) + (cy - dcy),
            z + (bz2 - dbz) + (cz - dcz),
            ax, ay, az,
            -bx2, -by2, -bz2,
            -(cx - cx2), -(cy - cy2), -(cz - cz2),
            coords,
        );
        hilbert_sfc_3d(
            x + (ax - dax) + bx2 + (cx - dcx),
            y + (ay - day) + by2 + (cy - dcy),
            z + (az - daz) + bz2 + (cz - dcz),
            -cx, -cy, -cz,
            -(ax - ax2), -(ay - ay2), -(az - az2),
            bx - bx2, by - by2, bz - bz2,
            coords,
        );
        hilbert_sfc_3d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            z + (az - daz) + (bz2 - dbz),
            -bx2, -by2, -bz2,
            cx2, cy2, cz2,
            -(ax - ax2), -(ay - ay2), -(az - az2),
            coords,
        );
    }
}

/// MFEM `NCMesh::GridSfcOrdering3D`: Hilbert-curve element order for INLINE
/// hex/tet meshes (`Make3D(..., sfc_ordering=true)`).
fn grid_sfc_ordering_3d(nx: usize, ny: usize, nz: usize) -> Vec<(i32, i32, i32)> {
    let mut coords = Vec::with_capacity(nx * ny * nz);
    let (w, h, d) = (nx as i32, ny as i32, nz as i32);
    if w >= h && w >= d {
        hilbert_sfc_3d(0, 0, 0, w, 0, 0, 0, h, 0, 0, 0, d, &mut coords);
    } else if h >= w && h >= d {
        hilbert_sfc_3d(0, 0, 0, 0, h, 0, w, 0, 0, 0, 0, d, &mut coords);
    } else {
        hilbert_sfc_3d(0, 0, 0, 0, 0, d, w, 0, 0, 0, h, 0, &mut coords);
    }
    coords
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

/// Write a `.gf` file in MFEM's native FiniteElementSpace format.
///
/// Produces files compatible with GLVis (`glvis -m mesh.mesh -g sol.gf`).
/// The format matches MFEM's `GridFunction::Save(std::ostream &)` output:
///
/// ```text
/// FiniteElementSpace
/// FiniteElementCollection: H1_<dim>D_P<order>
/// VDim: <vdim>
/// Ordering: <ordering>
/// <value 1>
/// <value 2>
/// ...
/// ```
///
/// A `precision` of 8 reproduces the C++ `ostream::precision(8)` setting.
/// fem-rs vector FE spaces use `fem_space::Ordering::ByNodes` (= MFEM
/// `Ordering::byNODES`, block layout: all component-0 DOFs, then component-1,
/// …; `vdof = dof + ndofs*vd`), so the ordering line is always `0` —
/// matching MFEM `GridFunction::Save`.
/// Format a `f64` exactly like C's `printf("%.16g", x)` — which is what
/// MFEM `Vector::Print` (via the default `std::ostream` `floatfield` with
/// precision 16) produces for `GridFunction::Save`.  Used to make `.gf`
/// output text-identical to the C++ reference.
///
/// Rules (matching `%.16g`): at most 16 significant digits; trailing zeros
/// stripped; fixed notation for decimal exponent in `[-4, 16)`, scientific
/// notation otherwise (exponent sign always present, at least two digits).
fn c_printf_g16(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    if !x.is_finite() {
        return x.to_string();
    }
    // 16 significant digits via scientific notation with 15 decimals.
    let s = format!("{:.15e}", x);
    let (mant, exp_str) = s.split_once('e').expect("scientific format");
    let exp: i32 = exp_str.parse().expect("exponent");
    let neg = mant.starts_with('-');
    let digits: String = mant
        .chars()
        .filter(|c| c.is_ascii_digit())
        .collect();
    // Strip trailing zeros like %g.
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    if (-4..16).contains(&exp) {
        // Fixed notation: decimal point sits after digit index `exp`.
        let mut out = String::new();
        if neg && digits != "0" {
            out.push('-');
        }
        let dot = 1 + exp; // 0-based position of the decimal point
        if dot <= 0 {
            out.push_str("0.");
            for _ in 0..-dot {
                out.push('0');
            }
            out.push_str(digits);
        } else if dot as usize >= digits.len() {
            out.push_str(digits);
            for _ in 0..(dot as usize - digits.len()) {
                out.push('0');
            }
        } else {
            out.push_str(&digits[..dot as usize]);
            out.push('.');
            out.push_str(&digits[dot as usize..]);
        }
        out
    } else {
        // Scientific notation, exponent with sign and ≥ 2 digits.
        let mut m = digits.to_string();
        if m.len() > 1 {
            m.insert(1, '.');
        }
        let sign = if exp >= 0 { "+" } else { "-" };
        let e = format!("{}{:02}", sign, exp.abs());
        format!("{}{}e{}", if neg { "-" } else { "" }, m, e)
    }
}

/// Write a `.gf` file in MFEM's native FiniteElementSpace format.
///
/// Produces files compatible with GLVis (`glvis -m mesh.mesh -g sol.gf`).
/// The format matches MFEM's `GridFunction::Save(std::ostream &)` output:
///
/// ```text
/// FiniteElementSpace
/// FiniteElementCollection: H1_<dim>D_P<order>
/// VDim: <vdim>
/// Ordering: <ordering>
/// <value 1>
/// <value 2>
/// ...
/// ```
///
/// `precision` controls the value formatting:
/// - `precision >= 16`: `printf("%.16g")` style (16 significant digits,
///   defaultfloat) — text-identical to MFEM `Vector::Print` at precision 16.
/// - `precision < 16`: `{:.prec$e}` scientific notation with `precision`
///   significant digits.
pub fn write_mfem_gf_file(
    path: impl AsRef<std::path::Path>,
    dim: usize, dofs: &[f64],
    space_type: &str, order: u8, vdim: usize,
    precision: usize,
) -> FemResult<()> {
    let mut file = std::fs::File::create(path)?;
    // FiniteElementSpace header (matching MFEM GridFunction::Save)
    writeln!(file, "FiniteElementSpace")?;
    writeln!(file, "FiniteElementCollection: {space_type}_{dim}D_P{order}")?;
    writeln!(file, "VDim: {vdim}")?;
    writeln!(file, "Ordering: 0")?;
    writeln!(file)?;
    // Values: precision controls total significant digits (C++ precision(8) → 8 sf)
    // Use {:.prec$e} where prec = precision - 1 gives precision total significant digits
    // (e.g. prec=7 gives 8 sf: "4.2830810e+01" for value 42.83081)
    let sig_digits = precision.saturating_sub(1).max(0);
    for v in dofs {
        if precision >= 16 {
            writeln!(file, "{}", c_printf_g16(*v))?;
        } else {
            writeln!(file, "{:.prec$e}", v, prec = sig_digits)?;
        }
    }
    Ok(())
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
