//! MFEM `.mesh` format reader (v1.0 / v1.2) and writer (v1.0).
//!
//! Also provides `.gf` GridFunction reader/writer (a minimal subset of the
//! MFEM GF format: dimension, space type, order, vdim, and DOF values).
//!
//! Supports linear elements in 2D and 3D:
//! Segment, Triangle, Quadrilateral, Tetrahedron, Hexahedron, Wedge, Pyramid.
//! Format reference: https://mfem.org/mesh-format/

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};

use fem_core::{FemError, FemResult, NodeId};
use fem_element::ReferenceElement;
use fem_mesh::{
    element_type::ElementType,
    simplex::{GeometryData, Mesh},
    topology::MeshTopology,
};
use fem_space::dof_manager::DofManager;

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
    // H1-continuous `nodes` section payload: (nodal order, values, ordering).
    let mut h1_nodes: Option<(u8, Vec<f64>, usize)> = None;

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
        let ordering_line = read_line(&mut r)?; // "Ordering: ..."

        // Read remaining values as DOF coefficient values.
        let mut raw: Vec<f64> = Vec::new();
        loop {
            match read_f64_line(&mut r) {
                Ok(vals) => raw.extend(vals),
                Err(_) => break,
            }
        }
        let fec_name = fec_line.split(':').nth(1).unwrap_or("").trim().to_string();
        let nodes_ordering: usize = ordering_line.split(':').nth(1)
            .and_then(|s| s.trim().parse().ok()).unwrap_or(0);
        let is_l2_nodes = fec_name.starts_with("L2_");
        if !is_l2_nodes {
            // Continuous (H1) geometry: remember the nodal order so the
            // high-order GeometryData can be attached once the mesh topology
            // is built (the DOF numbering needs it).
            if let Some(p) = parse_nodal_fec_order(&fec_name) {
                h1_nodes = Some((p, raw.clone(), nodes_ordering));
            }
        }
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
            // Continuous (H1) geometry: vertex `i` is dof `i` of the `nodes`
            // grid function (MFEM `Mesh::Loader` → `SetVerticesFromNodes`).
            // The dof values are stored with the section's ordering:
            //   Ordering: 0 (byNODES) — raw = [x of all dofs, y of all, z …],
            //   Ordering: 1 (byVDIM)  — raw = interleaved [x y z] per dof.
            // Reading interleaved triples from a byNODES stream (the previous
            // behavior) scrambles the vertex coordinates of curved meshes
            // (e.g. `cube.mesh`: vertex 0 became (0, 0.5, 1) instead of the
            // origin, which then corrupts refinement midpoints).
            if nodes_ordering == 0 {
                let ndof = raw.len() / dim;
                coords.clear();
                coords.resize(n_vert * dim, 0.0);
                for v in 0..n_vert {
                    for c in 0..dim {
                        coords[v * dim + c] = raw[c * ndof + v];
                    }
                }
            } else {
                coords.clear();
                coords.reserve(n_vert * dim);
                for i in 0..n_vert {
                    coords.extend_from_slice(&raw[i * dim..i * dim + dim]);
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
            vertex_parents: vec![],
            elem_offsets: elem_offsets_opt,
            face_types: face_types_opt.clone(),
            face_offsets: face_offsets_opt.clone(),
            face_to_elem: None,
            edge_conn: vec![], edge_to_elem: vec![],
            nc_vertex_view: None,
            geometry,
        };
        let mut mesh = mesh;
        if mesh.geometry.is_none() {
            if let Some((p, raw, ord)) = &h1_nodes {
                mesh.geometry = build_h1_geometry(&mesh, *p, raw, *ord, dim, None);
            }
        }
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
            vertex_parents: vec![],
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
        //
        // D43: the rotation changes the H1 numbering, and MFEM's
        // `PrepareNodeReorder`/`DoNodeReorder` renumber the `nodes` grid
        // function along with it so the geometry is preserved.  Reproduce that
        // by evaluating the slot map on the *pre-rotation* mesh (whose element
        // order is exactly the file's) and re-attaching the file's node values
        // to the same physical slots afterwards.
        let tet_file_slots: Option<TetFileSlots> = match &h1_nodes {
            Some((p, _, _)) if mesh.geometry.is_none() => {
                tet_slot_map(&mesh, *p as usize).ok().map(|(conn, keys, _)| TetFileSlots { conn, keys })
            }
            _ => None,
        };
        fem_mesh::mark_tet_mesh_for_refinement(&mut mesh);
        if mesh.geometry.is_none() {
            if let Some((p, raw, ord)) = &h1_nodes {
                mesh.geometry = build_h1_geometry(&mesh, *p, raw, *ord, dim, tet_file_slots.as_ref());
            }
        }
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
///
/// For 3D meshes containing tetrahedra, the mesh is cloned and normalized
/// with `mark_tet_mesh_for_refinement` before writing, so that programmatically
/// created meshes round-trip with the same canonical tet orientation that
/// `read_mfem` produces (longest edge = (v0,v1)).
pub fn write_mfem<W: Write>(writer: &mut W, mesh_d: &Mesh<2>, mesh_3d: Option<&Mesh<3>>) -> FemResult<()> {
    // D2: tet io round-trip orientation normalization.
    // read_mfem applies mark_tet_mesh_for_refinement (MarkTetMeshForRefinement)
    // on read to canonicalize tet vertex order.  write_mfem must apply the
    // same normalization so meshes created programmatically round-trip.
    let needs_normalization = mesh_3d.map_or(false, has_tet4);
    let tet_normalized: Option<Mesh<3>> = if needs_normalization {
        let mut clone = (*mesh_3d.unwrap()).clone();
        fem_mesh::mark_tet_mesh_for_refinement(&mut clone);
        Some(clone)
    } else {
        None
    };
    // If the 3D mesh contained tets, use the normalized clone; otherwise
    // fall back to the original mesh reference.
    let mesh_3d: Option<&Mesh<3>> = match tet_normalized.as_ref() {
        Some(n) => Some(n),
        None => mesh_3d,
    };
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

/// Returns `true` if the 3D mesh contains any Tet4 elements (uniform or mixed).
fn has_tet4(mesh: &Mesh<3>) -> bool {
    if let Some(ref etypes) = mesh.elem_types {
        etypes.iter().any(|et| *et == ElementType::Tet4)
    } else {
        mesh.elem_type == ElementType::Tet4
    }
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

/// Parse the polynomial order from an MFEM nodal FEC name, e.g.
/// `Linear_2D` → 1, `Quadratic3D` → 2, `Cubic_2D` → 3, `H1_2D_P4` → 4.
fn parse_nodal_fec_order(fec: &str) -> Option<u8> {
    let f = fec.trim();
    if f.starts_with("Linear_") || f.starts_with("LinearF") {
        return Some(1);
    }
    if f.starts_with("Quadratic") {
        return Some(2);
    }
    if f.starts_with("Cubic") {
        return Some(3);
    }
    if let Some(pos) = f.rfind("_P") {
        if let Ok(p) = f[pos + 2..].parse::<u8>() {
            if (1..=9).contains(&p) {
                return Some(p);
            }
        }
    }
    None
}

// ─── D41: MFEM-faithful H1 hexahedron geometry ───────────────────────────────
//
// MFEM numbers the DOFs of the H1 `nodes` grid function (fem/fespace.cpp,
// `FiniteElementSpace::GetElementDofs`) as
//
//     [ vertices | edge blocks | face blocks | interior blocks ]
//     vertex v            -> dof v
//     mesh edge  E, slot t -> dof NV + E*(p-1) + t
//     mesh face  F, slot o -> dof NV + NE*(p-1) + F*(p-1)^2 + o
//     element e, slot o    -> dof NV + NE*(p-1) + NF*(p-1)^2 + e*(p-1)^3 + o
//
// with the mesh edge/face indices assigned by `Mesh::FinalizeTopology`
// (`GetElementToEdgeTable` / `GenerateFaces`): element traversal order, local
// entities in `Geometry::CUBE::Edges` / `FaceVert` order, first encounter wins.
// Within an entity:
//   * an edge slot counts from the element end vertex with the **smaller mesh
//     vertex id** (`Mesh::GetElementEdges` sets `cor = v[e0] < v[e1] ? 1 : -1`
//     and `H1_FECollection`'s `SegDofOrd[0]` is the identity);
//   * a face slot `o = a + b*(p-1)` counts `a` along the face's `FaceVert`
//     edge 0→1 and `b` along edge 0→3 of the **first** element that created
//     the face (`QuadDofOrd[0]` is the identity), and every other element
//     re-indexes through `QuadDofOrd[orientation]`.
//
// The fem-rs assembly bases (`QuadQk`/`HexQk`) use their own slot order, so
// the file's dof ids cannot be handed to them directly (D31/D41).  The mapping
// below is therefore built geometrically: every slot of the reference element
// is classified from its own reference coordinate (vertex / edge / face /
// interior, and where on that entity), then translated into MFEM's numbering.
// This keeps `GeometryData::conn` in the reference element's slot order (which
// is what the assembler indexes) without duplicating any slot table.

/// MFEM `Constants<Geometry::CUBE>::Edges`: local edge `k` runs from local
/// vertex `EDGES[k][0]` to `EDGES[k][1]`.
const HEX_EDGES: [[usize; 2]; 12] = [
    [0, 1], [1, 2], [3, 2], [0, 3], [4, 5], [5, 6], [7, 6], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7],
];
/// MFEM `Constants<Geometry::CUBE>::FaceVert`: local face `f` lists its four
/// local vertices in canonical order (reference square `0,0 → 1,0 → 1,1 → 0,1`).
const HEX_FACES: [[usize; 4]; 6] = [
    [3, 2, 1, 0], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7],
];
/// Reference-cube corners (`{0,1}³`) of the 8 local vertices, in
/// `Geometry::CUBE::Vertices` order — the same order `HexQk` uses for its
/// vertex slots.
const HEX_CORNERS: [[usize; 3]; 8] = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
];
/// Square corners in the canonical (stored) face parameterisation `(a, b)`.
const QUAD_CORNERS: [[i32; 2]; 4] = [[0, 0], [1, 0], [1, 1], [0, 1]];

/// Outcome of the D41 hex geometry pass.
enum HexGeom {
    /// The mesh is not an all-Hex8 mesh — this path does not apply.
    NotHex,
    /// All-Hex8 mesh, but the `nodes` section cannot be mapped faithfully.
    Unsupported(&'static str),
    /// Faithful geometry table.
    Built(GeometryData),
}

/// D41: reproduce MFEM's H1 `nodes` numbering for an all-Hex8 mesh and return
/// the geometry table in the reference element's ([`HexQk`]) slot order.
///
/// `raw` is the `nodes` dof vector as stored in the file (`ordering` 0 =
/// byNODES, 1 = byVDIM).  Returns [`HexGeom::NotHex`] for meshes that are not
/// uniformly hexahedral so the caller can fall back to the simplex path.
fn build_h1_hex_geometry<M: MeshTopology>(
    mesh: &M,
    order: u8,
    raw: &[f64],
    ordering: usize,
) -> HexGeom {
    let p = order as usize;
    let e = p - 1; // dofs per edge (and per face row/column)
    let n_elems = mesh.n_elements();
    let n_vert = mesh.n_nodes();
    if n_elems == 0 || n_vert == 0 {
        return HexGeom::NotHex;
    }
    let mut n_hex = 0usize;
    for el in 0..n_elems as u32 {
        if mesh.element_nodes(el).len() == 8 {
            n_hex += 1;
        }
    }
    if n_hex == 0 {
        return HexGeom::NotHex; // pure simplex/other mesh: not our business
    }
    if n_hex != n_elems {
        // A mixed mesh containing hexahedra: the `nodes` dof blocks are sized
        // per element geometry, so neither this mapper nor the uniform
        // DofManager fallback can describe it.
        return HexGeom::Unsupported("mixed-element mesh containing hexahedra");
    }

    // Mesh edges/faces in MFEM's enumeration (element traversal, then local
    // entity order, first encounter wins).
    let mut elems: Vec<[u32; 8]> = Vec::with_capacity(n_elems);
    let mut edge_ids: HashMap<[u32; 2], u32> = HashMap::new();
    let mut face_ids: HashMap<[u32; 4], u32> = HashMap::new();
    let mut face_verts: Vec<[u32; 4]> = Vec::new();
    for el in 0..n_elems as u32 {
        let ns = mesh.element_nodes(el);
        let mut n8 = [0u32; 8];
        n8.copy_from_slice(ns);
        for &[la, lb] in HEX_EDGES.iter() {
            let (a, b) = (n8[la], n8[lb]);
            let key = if a < b { [a, b] } else { [b, a] };
            let next = edge_ids.len() as u32;
            edge_ids.entry(key).or_insert(next);
        }
        for fv in HEX_FACES.iter() {
            let mut key = [n8[fv[0]], n8[fv[1]], n8[fv[2]], n8[fv[3]]];
            key.sort_unstable();
            let next = face_ids.len() as u32;
            face_ids.entry(key).or_insert_with(|| {
                face_verts.push([n8[fv[0]], n8[fv[1]], n8[fv[2]], n8[fv[3]]]);
                next
            });
        }
        elems.push(n8);
    }
    let n_edges = edge_ids.len();
    let n_faces = face_ids.len();

    let n_dofs = n_vert + n_edges * e + n_faces * e * e + n_elems * e * e * e;
    if raw.len() < 3 * n_dofs {
        return HexGeom::Unsupported("nodes section too short for the H1 hex space");
    }
    let mut coords = vec![0.0f64; n_dofs * 3];
    match ordering {
        0 => {
            for c in 0..3 {
                for g in 0..n_dofs {
                    coords[g * 3 + c] = raw[c * n_dofs + g];
                }
            }
        }
        1 => coords.copy_from_slice(&raw[..3 * n_dofs]),
        _ => return HexGeom::Unsupported("unknown nodes ordering"),
    }

    // Local-edge lookup: (varying axis, side of the two fixed axes) -> edge id.
    let mut edge_lut: HashMap<(usize, usize, usize), usize> = HashMap::new();
    for (k, &[la, lb]) in HEX_EDGES.iter().enumerate() {
        let (ca, cb) = (HEX_CORNERS[la], HEX_CORNERS[lb]);
        let av = match (0..3).find(|&d| ca[d] != cb[d]) {
            Some(d) => d,
            None => return HexGeom::Unsupported("degenerate HEX_EDGES table"),
        };
        let bnd: Vec<usize> = (0..3).filter(|&d| d != av).collect();
        edge_lut.insert((av, ca[bnd[0]], ca[bnd[1]]), k);
    }

    let ref_elem = fem_element::lagrange::factory::HexQk::new(p);
    let ref_coords = ref_elem.dof_coords();
    let npe = ref_coords.len();
    if npe != 8 + 12 * e + 6 * e * e + e * e * e {
        return HexGeom::Unsupported("reference hex element is not the H1 order-p tensor basis");
    }
    // The 1-D GLL nodes of the reference basis: the tensor index of a slot is
    // the position of its coordinate in this table (`HexQk` builds its 1-D
    // basis from the same function, so the values match bit-for-bit).
    let gll = fem_element::quadrature::gauss_lobatto_arbitrary(p + 1).0;
    if gll.len() != p + 1 {
        return HexGeom::Unsupported("unexpected Gauss-Lobatto node count");
    }

    let edge_base = n_vert;
    let face_base = edge_base + n_edges * e;
    let interior_base = face_base + n_faces * e * e;
    let mut conn: Vec<NodeId> = Vec::with_capacity(n_elems * npe);
    for (el, n8) in elems.iter().enumerate() {
        let mut interior_seen = 0usize;
        for c in ref_coords.iter() {
            // Tensor index of the slot along each axis (GLL nodes, 0..=p).
            let mut idx = [0usize; 3];
            for d in 0..3 {
                let k = match gll.iter().position(|&x| (x - c[d]).abs() < 1e-12) {
                    Some(k) => k,
                    None => {
                        return HexGeom::Unsupported("reference slot is not on the GLL tensor grid")
                    }
                };
                idx[d] = k;
            }
            let on_bnd = [
                idx[0] == 0 || idx[0] == p,
                idx[1] == 0 || idx[1] == p,
                idx[2] == 0 || idx[2] == p,
            ];
            let nb = on_bnd.iter().filter(|&&b| b).count();
            let g: usize = match nb {
                // Vertex slot: the file stores vertex `v` as dof `v`.
                3 => {
                    let side = [idx[0] / p, idx[1] / p, idx[2] / p];
                    match (0..8).find(|&k| HEX_CORNERS[k] == side) {
                        Some(lv) => n8[lv] as usize,
                        None => return HexGeom::Unsupported("bad vertex slot"),
                    }
                }
                // Edge slot: shared, canonical direction = ascending vertex id.
                2 => {
                    let bnd: Vec<usize> = (0..3).filter(|&d| on_bnd[d]).collect();
                    let av = match (0..3).find(|&d| !on_bnd[d]) {
                        Some(d) => d,
                        None => return HexGeom::Unsupported("bad edge slot"),
                    };
                    let key = (av, idx[bnd[0]] / p, idx[bnd[1]] / p);
                    let k = match edge_lut.get(&key) {
                        Some(&k) => k,
                        None => return HexGeom::Unsupported("unmatched edge slot"),
                    };
                    let [la, lb] = HEX_EDGES[k];
                    let t_local = if HEX_CORNERS[la][av] == 0 {
                        idx[av] - 1
                    } else {
                        p - 1 - idx[av]
                    };
                    let (a, b) = (n8[la], n8[lb]);
                    let ekey = if a < b { [a, b] } else { [b, a] };
                    let ei = match edge_ids.get(&ekey) {
                        Some(&ei) => ei as usize,
                        None => return HexGeom::Unsupported("unmatched mesh edge"),
                    };
                    let t = if a < b { t_local } else { e - 1 - t_local };
                    edge_base + ei * e + t
                }
                // Face slot: shared, canonical parameterisation = the first
                // element's `FaceVert` order.
                1 => {
                    let ax = match (0..3).find(|&d| on_bnd[d]) {
                        Some(d) => d,
                        None => return HexGeom::Unsupported("bad face slot"),
                    };
                    let side = idx[ax] / p;
                    let f = match (0..6).find(|&f| HEX_FACES[f].iter().all(|&v| HEX_CORNERS[v][ax] == side)) {
                        Some(f) => f,
                        None => return HexGeom::Unsupported("unmatched local face"),
                    };
                    let [l0, l1, _, l3] = HEX_FACES[f];
                    // In-face indices (from the local vertex `l0`).
                    let mut in_face = [0usize; 2];
                    for (s, &lk) in [l1, l3].iter().enumerate() {
                        let d = match (0..3).find(|&d| HEX_CORNERS[l0][d] != HEX_CORNERS[lk][d]) {
                            Some(d) => d,
                            None => return HexGeom::Unsupported("degenerate face slot"),
                        };
                        if d == ax {
                            return HexGeom::Unsupported("degenerate face slot");
                        }
                        in_face[s] = if HEX_CORNERS[l0][d] == 0 { idx[d] } else { p - idx[d] };
                        if in_face[s] == 0 || in_face[s] >= p {
                            return HexGeom::Unsupported("face slot on a face edge");
                        }
                    }
                    let mut fkey = [0u32; 4];
                    for (k, &lv) in HEX_FACES[f].iter().enumerate() {
                        fkey[k] = n8[lv];
                    }
                    fkey.sort_unstable();
                    let fi = match face_ids.get(&fkey) {
                        Some(&fi) => fi as usize,
                        None => return HexGeom::Unsupported("unmatched mesh face"),
                    };
                    // Map the local face parameterisation onto the canonical
                    // one (corner matching: at most a rotation/reflection).
                    let cv = face_verts[fi];
                    let corner_of = |v: u32| (0..4).find(|&i| cv[i] == v);
                    let p0 = match corner_of(n8[l0]) {
                        Some(i) => i,
                        None => return HexGeom::Unsupported("face corner mismatch"),
                    };
                    let pu = match corner_of(n8[l1]) {
                        Some(i) => i,
                        None => return HexGeom::Unsupported("face corner mismatch"),
                    };
                    let pv = match corner_of(n8[l3]) {
                        Some(i) => i,
                        None => return HexGeom::Unsupported("face corner mismatch"),
                    };
                    let du = [
                        QUAD_CORNERS[pu][0] - QUAD_CORNERS[p0][0],
                        QUAD_CORNERS[pu][1] - QUAD_CORNERS[p0][1],
                    ];
                    let dv = [
                        QUAD_CORNERS[pv][0] - QUAD_CORNERS[p0][0],
                        QUAD_CORNERS[pv][1] - QUAD_CORNERS[p0][1],
                    ];
                    // Canonical parameter position of the slot, in GLL index
                    // units along the stored face's own (a, b) axes.
                    let u = QUAD_CORNERS[p0][0] * p as i32 + in_face[0] as i32 * du[0]
                        + in_face[1] as i32 * dv[0];
                    let v = QUAD_CORNERS[p0][1] * p as i32 + in_face[0] as i32 * du[1]
                        + in_face[1] as i32 * dv[1];
                    if u <= 0 || u >= p as i32 || v <= 0 || v >= p as i32 {
                        return HexGeom::Unsupported("face slot outside the canonical face");
                    }
                    let o = (u - 1) as usize + (v - 1) as usize * e;
                    face_base + fi * e * e + o
                }
                // Interior slot: private to the element; the file orders them
                // per element, so keep the reference element's own order.
                _ => {
                    let g = interior_base + el * e * e * e + interior_seen;
                    interior_seen += 1;
                    g
                }
            };
            conn.push(g as NodeId);
        }
        if interior_seen != e * e * e {
            return HexGeom::Unsupported("unexpected interior slot count");
        }
    }

    HexGeom::Built(GeometryData {
        order,
        conn,
        nodes_per_elem: npe,
        coords,
        n_nodes: n_dofs,
    })
}

// ─── D43: MFEM-faithful H1 tetrahedron geometry ──────────────────────────────

/// MFEM `Constants<Geometry::TETRAHEDRON>::Edges`: local edge `k` runs from
/// local vertex `EDGES[k][0]` to `EDGES[k][1]` (both already ascending).
const TET_EDGES: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
/// MFEM `Constants<Geometry::TETRAHEDRON>::FaceVert`: local face `f` lists its
/// three local vertices (`f` is the vertex the face is *opposite*; the three
/// are ordered as in `Mesh::GenerateFaces`).
const TET_FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];
/// MFEM `Constants<Geometry::TRIANGLE>::Orient`: `Orient[o][i]` is the local
/// vertex that ends up at position `i` under orientation `o`.
const TRI_ORIENT: [[usize; 3]; 6] = [
    [0, 1, 2],
    [1, 0, 2],
    [2, 0, 1],
    [2, 1, 0],
    [1, 2, 0],
    [0, 2, 1],
];

/// MFEM `Mesh::GetTriOrientation(base, test)`: the orientation index `o` with
/// `test[TRI_ORIENT[o][i]] == base[i]` (0 if the two orderings coincide).
fn tri_orientation(base: &[u32; 3], test: &[u32; 3]) -> Option<usize> {
    (0..6).find(|&o| (0..3).all(|i| test[TRI_ORIENT[o][i]] == base[i]))
}

/// Index, within the interior block of MFEM's `H1_TriangleElement`, of the
/// face DOF at in-face integer coordinates `(a, b)` (measured from face vertex
/// 0 towards vertices 1 and 2, so `a, b ≥ 1` and `a + b ≤ p - 1`).
///
/// This is the `o` of `H1_FECollection`'s `TriDofOrd` construction
/// (`fem/fe_coll.cpp`): `TriDof - ((p-1-j)(p-2-j))/2 + i` with
/// `(i, j) = (a-1, b-1)`.
fn tri_face_index(p: usize, a: usize, b: usize) -> Option<usize> {
    if p < 3 || a < 1 || b < 1 || a + b > p - 1 {
        return None;
    }
    let (i, j) = (a - 1, b - 1);
    let tri_dof = (p - 1) * (p - 2) / 2;
    let (pm1, pm2) = (p - 1, p - 2);
    if i + j >= pm2 {
        return None;
    }
    Some(tri_dof - ((pm1 - j) * (pm2 - j)) / 2 + i)
}

/// MFEM `H1_FECollection::DofOrderForOrientation(TRIANGLE, or)[j]` — the
/// canonical (stored) face DOF index for local face DOF `j`.
fn tri_dof_ord(p: usize, orient: usize, j: usize) -> Option<usize> {
    if p < 3 || orient > 5 {
        return None;
    }
    let (pm1, pm2, pm3) = (p - 1, p - 2, p - 3);
    let tri_dof = (p - 1) * (p - 2) / 2;
    if j >= tri_dof {
        return None;
    }
    for jj in 0..pm2 {
        for ii in 0..(pm2 - jj) {
            let o = tri_dof - ((pm1 - jj) * (pm2 - jj)) / 2 + ii;
            if o != j {
                continue;
            }
            let k = pm3 - jj - ii;
            return Some(match orient {
                0 => o,
                1 => tri_dof - ((pm1 - jj) * (pm2 - jj)) / 2 + k,
                2 => tri_dof - ((pm1 - ii) * (pm2 - ii)) / 2 + k,
                3 => tri_dof - ((pm1 - k) * (pm2 - k)) / 2 + ii,
                4 => tri_dof - ((pm1 - k) * (pm2 - k)) / 2 + jj,
                _ => tri_dof - ((pm1 - ii) * (pm2 - ii)) / 2 + jj,
            });
        }
    }
    None
}

/// Outcome of the D43 tetrahedron geometry pass.
enum TetGeom {
    /// The mesh is not an all-Tet4 mesh — this path does not apply.
    NotTet,
    /// All-Tet4 mesh, but the `nodes` section cannot be mapped faithfully.
    Unsupported(&'static str),
    /// Faithful geometry table.
    Built(GeometryData),
}

/// D43: reproduce MFEM's H1 `nodes` numbering for an all-Tet4 mesh and return
/// the geometry table in the reference element's slot order.
///
/// Same construction as [`build_h1_hex_geometry`], with the tetrahedron's
/// entity tables (`TET_EDGES`, `TET_FACES`) and MFEM's triangle DOF ordering
/// (`TriDofOrd`) for the shared face blocks.  Until D43 this element type fell
/// through to fem-rs's own `DofManager` numbering, which silently gave 11 of
/// the 42 elements of `data/escher-p2.mesh` another element's edge dofs.
///
/// The reference element is the one the assembler uses for tet geometry
/// (`fem_element::lagrange::factory::TetPk`, equispaced nodes on the unit
/// tetrahedron), so its slots are classified by the *integer barycentric
/// coordinates* of their node positions.
fn build_h1_tet_geometry<M: MeshTopology>(
    mesh: &M,
    order: u8,
    raw: &[f64],
    ordering: usize,
    file_slots: Option<&TetFileSlots>,
) -> TetGeom {
    let p = order as usize;
    let n_elems = mesh.n_elements();
    let n_vert = mesh.n_nodes();
    if n_elems == 0 || n_vert == 0 {
        return TetGeom::NotTet;
    }
    let mut n_tet = 0usize;
    let mut n_other = 0usize;
    for el in 0..n_elems as u32 {
        match mesh.element_nodes(el).len() {
            4 => n_tet += 1,
            _ => n_other += 1,
        }
    }
    if n_tet == 0 {
        return TetGeom::NotTet;
    }
    if n_other != 0 {
        // Any mixture (including hexes, which the D41 pass refuses separately)
        // has per-element `nodes` blocks that no single numbering describes.
        return TetGeom::Unsupported("mixed-element mesh containing tetrahedra");
    }

    let (conn, keys, n_dofs) = match tet_slot_map(mesh, p) {
        Ok(c) => c,
        Err(why) => return TetGeom::Unsupported(why),
    };
    let npe = conn.len() / n_elems;
    if raw.len() < 3 * n_dofs {
        return TetGeom::Unsupported("nodes section too short for the H1 tet space");
    }
    // The file's `nodes` dof vector, in the file's *own* numbering (which is
    // the numbering of the mesh as recorded in the file's `elements` section).
    let mut coords_file = vec![0.0f64; n_dofs * 3];
    match ordering {
        0 => {
            for c in 0..3 {
                for g in 0..n_dofs {
                    coords_file[g * 3 + c] = raw[c * n_dofs + g];
                }
            }
        }
        1 => coords_file.copy_from_slice(&raw[..3 * n_dofs]),
        _ => return TetGeom::Unsupported("unknown nodes ordering"),
    }
    // A curved mesh read with MFEM's `refine = 1` (which fem-rs mirrors with
    // `mark_tet_mesh_for_refinement`) is renumbered: `PrepareNodeReorder` /
    // `DoNodeReorder` permute the nodes grid function so the *geometry* is
    // preserved under the rotated element vertex order.  Every geometry node
    // belongs to a physical entity (a vertex, an edge, a face, or the element
    // interior) at a definite position, which is exactly what the slot `keys`
    // encode; transferring the file's node values through the keys therefore
    // reproduces MFEM's renumbering.
    let mut coords = coords_file.clone();
    if let Some(fs) = file_slots {
        if fs.conn.len() != conn.len() {
            return TetGeom::Unsupported("file slot map size mismatch");
        }
        let mut value_of_key: HashMap<(usize, [usize; 4]), [f64; 3]> = HashMap::with_capacity(conn.len());
        for (i, k) in fs.keys.iter().enumerate() {
            let g = fs.conn[i] as usize;
            value_of_key.insert(*k, [coords_file[g * 3], coords_file[g * 3 + 1], coords_file[g * 3 + 2]]);
        }
        for (i, k) in keys.iter().enumerate() {
            let v = match value_of_key.get(k) {
                Some(v) => *v,
                None => return TetGeom::Unsupported("no file node for a physical slot"),
            };
            let g = conn[i] as usize;
            coords[g * 3] = v[0];
            coords[g * 3 + 1] = v[1];
            coords[g * 3 + 2] = v[2];
        }
    }

    TetGeom::Built(GeometryData {
        order,
        conn,
        nodes_per_elem: npe,
        coords,
        n_nodes: n_dofs,
    })
}

/// The file's own slot map for the D43 tetrahedral path.
///
/// `conn[i]` is the dof the file's `nodes` vector uses for slot `i` (with the
/// element connectivity exactly as listed in the file), and `keys[i]` is the
/// slot's *physical* key `(element, pattern)`: its integer barycentric
/// coordinates in the labelling of the element's four vertices **sorted by
/// global id**.  Two slots of the same element with the same key address the
/// same physical node (the same vertex, or the same edge/face position
/// measured from the mesh vertex with the smaller id), which is what makes the
/// key usable to transfer the file's node values across MFEM's `refine = 1`
/// renumbering (the rotation permutes an element's vertex *list* but keeps its
/// vertex *set*).
struct TetFileSlots {
    conn: Vec<NodeId>,
    keys: Vec<(usize, [usize; 4])>,
}

/// Per-element slot map for a uniform Tet4 mesh: the geometry dof index MFEM's
/// H1 numbering assigns to each reference-element slot, in
/// `fem_element::lagrange::factory::TetPk`'s slot order, the slot's physical
/// key (see [`TetFileSlots`]), and the total number of geometry dofs.  `Err`
/// carries the reason the mesh cannot be mapped faithfully.
///
/// The entity enumeration is MFEM's: mesh edges/faces are numbered by element
/// traversal, then local entity order (`TET_EDGES` / `TET_FACES`), first
/// encounter wins, and the stored face parameterisation is the first
/// encountering element's `TET_FACES` order (`Mesh::AddTriangleFaceElement`
/// stores the face verbatim from elem1, elem2 gets the orientation).
fn tet_slot_map<M: MeshTopology>(
    mesh: &M,
    p: usize,
) -> Result<(Vec<NodeId>, Vec<(usize, [usize; 4])>, usize), &'static str> {
    let e = p - 1; // dofs per edge
    let nf = if p >= 3 { (p - 1) * (p - 2) / 2 } else { 0 }; // dofs per face
    let nb = if p >= 4 { (p - 1) * (p - 2) * (p - 3) / 6 } else { 0 };
    let n_elems = mesh.n_elements();
    let n_vert = mesh.n_nodes();
    if n_elems == 0 || n_vert == 0 {
        return Err("empty mesh");
    }

    let mut elems: Vec<[u32; 4]> = Vec::with_capacity(n_elems);
    let mut edge_ids: HashMap<[u32; 2], u32> = HashMap::new();
    let mut face_ids: HashMap<[u32; 3], u32> = HashMap::new();
    let mut face_verts: Vec<[u32; 3]> = Vec::new();
    for el in 0..n_elems as u32 {
        let ns = mesh.element_nodes(el);
        if ns.len() != 4 {
            return Err("mesh is not uniformly tetrahedral");
        }
        let mut n4 = [0u32; 4];
        n4.copy_from_slice(ns);
        for &[la, lb] in TET_EDGES.iter() {
            let (a, b) = (n4[la], n4[lb]);
            let key = if a < b { [a, b] } else { [b, a] };
            let next = edge_ids.len() as u32;
            edge_ids.entry(key).or_insert(next);
        }
        for fv in TET_FACES.iter() {
            let mut key = [n4[fv[0]], n4[fv[1]], n4[fv[2]]];
            key.sort_unstable();
            let next = face_ids.len() as u32;
            face_ids.entry(key).or_insert_with(|| {
                face_verts.push([n4[fv[0]], n4[fv[1]], n4[fv[2]]]);
                next
            });
        }
        elems.push(n4);
    }
    let n_edges = edge_ids.len();
    let n_faces = face_ids.len();

    // The reference element the assembler uses for tet geometry; its node
    // positions are the equispaced barycentric grid, so the slot's integer
    // barycentric coordinates are exactly `p·λ`.
    let ref_elem = fem_element::lagrange::factory::TetPk::new(p);
    let ref_coords = ref_elem.dof_coords();
    let npe = ref_coords.len();
    if npe != 4 + 6 * e + 4 * nf + nb {
        return Err("reference tet element is not the H1 order-p basis");
    }

    let edge_base = n_vert;
    let face_base = edge_base + n_edges * e;
    let interior_base = face_base + n_faces * nf;
    let mut conn: Vec<NodeId> = Vec::with_capacity(n_elems * npe);
    let mut keys: Vec<(usize, [usize; 4])> = Vec::with_capacity(n_elems * npe);
    for (el, n4) in elems.iter().enumerate() {
        // Local vertex indices ordered by global vertex id: the key's position
        // `m` always refers to the `m`-th smallest mesh vertex of the element.
        let mut order = [0usize, 1, 2, 3];
        order.sort_by_key(|&m| n4[m]);
        let mut interior_seen = 0usize;
        for c in ref_coords.iter() {
            let lam = [1.0 - c[0] - c[1] - c[2], c[0], c[1], c[2]];
            let mut idx = [0usize; 4];
            for m in 0..4 {
                let t = p as f64 * lam[m];
                let r = t.round();
                if (t - r).abs() > 1e-9 {
                    return Err("slot is not on the integer barycentric grid");
                }
                idx[m] = r as usize;
            }
            if idx.iter().sum::<usize>() != p {
                return Err("slot barycentric coordinates do not sum to the order");
            }
            let key = [idx[order[0]], idx[order[1]], idx[order[2]], idx[order[3]]];
            keys.push((el, key));
            let on_bnd: Vec<usize> = (0..4).filter(|&m| idx[m] > 0).collect();
            let g: usize = match on_bnd.len() {
                // Vertex slot: the file stores vertex `v` as dof `v`.
                1 => n4[on_bnd[0]] as usize,
                // Edge slot: shared, slot `t` counted from the local edge's
                // first vertex, and from the end vertex with the smaller mesh
                // vertex id (MFEM `cor = v[e0] < v[e1] ? 1 : -1`).
                2 => {
                    let (la, lb) = (on_bnd[0], on_bnd[1]);
                    if !TET_EDGES.iter().any(|&e| e == [la, lb]) {
                        return Err("unmatched local edge slot");
                    }
                    let t_local = idx[lb] - 1;
                    let (a, b) = (n4[la], n4[lb]);
                    let ekey = if a < b { [a, b] } else { [b, a] };
                    let ei = match edge_ids.get(&ekey) {
                        Some(&ei) => ei as usize,
                        None => return Err("unmatched mesh edge"),
                    };
                    let t = if a < b { t_local } else { e - 1 - t_local };
                    edge_base + ei * e + t
                }
                // Face slot: shared, canonical parameterisation = the first
                // element's `TET_FACES` order; the other elements re-key
                // through MFEM's `TriDofOrd[orientation]`.
                3 => {
                    if nf == 0 {
                        return Err("face slot at order < 3");
                    }
                    let mis = match (0..4).find(|&m| idx[m] == 0) {
                        Some(m) => m,
                        None => return Err("bad face slot"),
                    };
                    let [l0, l1, l2] = TET_FACES[mis];
                    let (a, b) = (idx[l1], idx[l2]);
                    let j = match tri_face_index(p, a, b) {
                        Some(j) => j,
                        None => return Err("face slot outside the reference face"),
                    };
                    let mut fkey = [n4[l0], n4[l1], n4[l2]];
                    fkey.sort_unstable();
                    let fi = match face_ids.get(&fkey) {
                        Some(&fi) => fi as usize,
                        None => return Err("unmatched mesh face"),
                    };
                    let test = [n4[l0], n4[l1], n4[l2]];
                    let orient = match tri_orientation(&face_verts[fi], &test) {
                        Some(o) => o,
                        None => return Err("face corner mismatch"),
                    };
                    let canon = match tri_dof_ord(p, orient, j) {
                        Some(v) => v,
                        None => return Err("unmatched face dof ordering"),
                    };
                    face_base + fi * nf + canon
                }
                // Interior slot: private to the element; MFEM's
                // `H1_TetrahedronElement` enumerates them in the same order as
                // the reference element, so keep the running index.
                _ => {
                    if interior_seen >= nb {
                        return Err("unexpected interior slot count");
                    }
                    let g = interior_base + el * nb + interior_seen;
                    interior_seen += 1;
                    g
                }
            };
            conn.push(g as NodeId);
        }
        if interior_seen != nb {
            return Err("unexpected interior slot count");
        }
    }

    Ok((conn, keys, n_vert + n_edges * e + n_faces * nf + n_elems * nb))
}

/// Build the high-order `GeometryData` for an H1-continuous `nodes` section:
/// the file stores one coordinate triple per DOF of the order-`p` H1 space and
/// the per-element geometry tables are the element DOF lists (H1 topological
/// order, matching the `QuadQk::new(p)`/`HexQk::new(p)` assembly bases).
///
/// D41: the H1 DOF numbering of the file is *MFEM's*, not fem-rs's.  For
/// hexahedra the two differ (HexQk orders its edge/face blocks differently)
/// and the difference is silent on load, so 3D hex meshes are routed through
/// [`build_h1_hex_geometry`], which reproduces MFEM's numbering exactly.  If
/// that fails the mesh is left without high-order geometry *and* a warning is
/// printed — never a silently scrambled mapping (see `D41` notes below).
fn build_h1_geometry<M: MeshTopology>(
    mesh: &M,
    order: u8,
    raw: &[f64],
    ordering: usize,
    dim: usize,
    tet_file_slots: Option<&TetFileSlots>,
) -> Option<GeometryData> {
    if order < 2 {
        return None; // linear geometry needs no table
    }
    if dim == 3 {
        match build_h1_hex_geometry(mesh, order, raw, ordering) {
            HexGeom::Built(g) => return Some(g),
            HexGeom::NotHex => {
                // D43: all-tetrahedron meshes get the same faithful treatment.
                // Meshes with neither hexahedra nor tetrahedra still fall
                // through to fem-rs's own numbering below (with a warning).
                match build_h1_tet_geometry(mesh, order, raw, ordering, tet_file_slots) {
                    TetGeom::Built(g) => return Some(g),
                    TetGeom::Unsupported(why) => {
                        eprintln!(
                            "warning (D43): refusing to build high-order geometry for a \
                             tetrahedral mesh ({why}); the mesh is read as straight-sided \
                             (geometric order 1)"
                        );
                        return None;
                    }
                    TetGeom::NotTet => {
                        eprintln!(
                            "warning (D41): high-order `nodes` geometry on a 3D mesh without \
                             hexahedra is read with fem-rs's own H1 numbering, which is not \
                             verified against MFEM for this element type"
                        );
                    }
                }
            }
            HexGeom::Unsupported(why) => {
                // D41: accepting the (wrong) DofManager slot order here would
                // silently scramble the geometry of every curved hex mesh —
                // the Jacobians, volumes and quadrature maps would all be
                // built for a different isoparametric element.  Refuse instead
                // (the mesh then degrades to its straight-line vertices, which
                // are always correct).
                eprintln!(
                    "warning (D41): refusing to build high-order geometry for a \
                     hexahedral mesh ({why}); the mesh is read as straight-sided \
                     (geometric order 1)"
                );
                return None;
            }
        }
    }
    let dm = DofManager::new(mesh, order);
    let n_dofs = dm.n_dofs;
    if raw.len() < dim * n_dofs {
        return None;
    }
    let mut dof_coords = vec![0.0f64; n_dofs * dim];
    match ordering {
        0 => {
            // byNODES: [x of all dofs, y of all dofs, ...]
            for c in 0..dim {
                for d in 0..n_dofs {
                    dof_coords[d * dim + c] = raw[c * n_dofs + d];
                }
            }
        }
        1 => {
            // byVDIM: [x y (z)] per dof.
            dof_coords.copy_from_slice(&raw[..dim * n_dofs]);
        }
        _ => return None,
    }
    let mut conn: Vec<NodeId> = Vec::new();
    let npe = dm.element_dofs(0).len();
    for e in 0..mesh.n_elements() {
        let dofs = dm.element_dofs(e as u32);
        if dofs.len() != npe {
            return None; // mixed mesh: not supported by GeometryData layout
        }
        conn.extend(dofs.iter().copied());
    }
    Some(GeometryData {
        order,
        conn,
        nodes_per_elem: npe,
        coords: dof_coords,
        n_nodes: n_dofs,
    })
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
    // 3-D inline types (tet/hex/wedge/pyramid) also carry nz/sz
    // (MFEM ReadInlineMesh, mesh_readers.cpp).
    let is_3d = matches!(elem_type_str.as_str(), "tet" | "hex" | "wedge" | "pyramid");
    let _nz = if is_3d {
        Some(read_param_usize(r, "nz")?)
    } else { None };
    let sx = read_param_f64(r, "sx")?;
    let sy = read_param_f64(r, "sy")?;
    let _sz = if is_3d {
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
            // Boundary segments — MFEM Make2D order (mesh/mesh.cpp):
            //   boundary[i]            = (i, i+1)          bottom, attr 1
            //   boundary[nx+i]         = (m+i+1, m+i)      top,    attr 3
            //   boundary[2*nx+j]       = ((j+1)*m, j*m)    left,   attr 4
            //   boundary[2*nx+ny+j]    = (j*m+nx, (j+1)*m+nx) right, attr 2
            // (m = nxv).  The boundary-face order matters for the assembly
            // column order (e.g. ex41 BlockILU MDF reordering).
            let mut face_conn = Vec::with_capacity(2 * (nx + ny) * 2);
            let mut face_tags = Vec::with_capacity(2 * (nx + ny));
            for i in 0..nx {
                face_conn.extend([id(i as i32, 0), id(i as i32 + 1, 0)]);
                face_tags.push(1);
            }
            for i in 0..nx {
                face_conn.extend([id(i as i32 + 1, ny as i32), id(i as i32, ny as i32)]);
                face_tags.push(3);
            }
            for j in 0..ny {
                face_conn.extend([id(0, j as i32 + 1), id(0, j as i32)]);
                face_tags.push(4);
            }
            for j in 0..ny {
                face_conn.extend([id(nx as i32, j as i32), id(nx as i32, j as i32 + 1)]);
                face_tags.push(2);
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
        "wedge" => {
            // MFEM ReadInlineMesh → Make3D(nx, ny, nz, WEDGE, ...) — row-major
            // hex grid, each hex split into 2 prisms (AddHexAsWedges:
            // {0,1,2,4,5,6}, {0,2,3,4,6,7}; prism vertices = bottom tri then
            // top tri).
            let nz = _nz.unwrap_or(1);
            let sz = _sz.unwrap_or(1.0);
            let (nxv, nyv, nzv) = (nx + 1, ny + 1, nz + 1);
            let mut coords = Vec::with_capacity(nxv * nyv * nzv * 3);
            for k in 0..nzv {
                for j in 0..nyv {
                    for i in 0..nxv {
                        coords.push(i as f64 / nx as f64 * sx);
                        coords.push(j as f64 / ny as f64 * sy);
                        coords.push(k as f64 / nz as f64 * sz);
                    }
                }
            }
            let id = |x: usize, y: usize, z: usize| {
                ((z * nyv + y) * nxv + x) as u32
            };
            const HEX_TO_WDG: [[usize; 6]; 2] = [
                [0, 1, 2, 4, 5, 6],
                [0, 2, 3, 4, 6, 7],
            ];
            let mut conn = Vec::with_capacity(nx * ny * nz * 2 * 6);
            let mut elem_tags = Vec::with_capacity(nx * ny * nz * 2);
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let hex = [
                            id(x, y, z), id(x + 1, y, z), id(x + 1, y + 1, z), id(x, y + 1, z),
                            id(x, y, z + 1), id(x + 1, y, z + 1), id(x + 1, y + 1, z + 1),
                            id(x, y + 1, z + 1),
                        ];
                        for w in HEX_TO_WDG {
                            conn.extend(w.map(|k| hex[k]));
                            elem_tags.push(1);
                        }
                    }
                }
            }
            // Boundary triangles (bottom attr 1, top attr 6; sides 2–5 like
            // Make3D's AddBdrQuadAsTriangles layout, simplified per face).
            let mut face_conn = Vec::new();
            let mut face_tags = Vec::new();
            let mut bdr_tri = |quad: [u32; 4], tag: i32| {
                face_conn.extend([quad[0], quad[1], quad[2]]);
                face_tags.push(tag);
                face_conn.extend([quad[0], quad[2], quad[3]]);
                face_tags.push(tag);
            };
            for y in 0..ny {
                for x in 0..nx {
                    bdr_tri([id(x, y, 0), id(x, y + 1, 0), id(x + 1, y + 1, 0), id(x + 1, y, 0)], 1);
                    bdr_tri([id(x, y, nz), id(x + 1, y, nz), id(x + 1, y + 1, nz), id(x, y + 1, nz)], 6);
                }
            }
            for z in 0..nz {
                for y in 0..ny {
                    bdr_tri([id(0, y, z), id(0, y, z + 1), id(0, y + 1, z + 1), id(0, y + 1, z)], 5);
                    bdr_tri([id(nx, y, z + 1), id(nx, y, z), id(nx, y + 1, z), id(nx, y + 1, z + 1)], 2);
                }
            }
            for z in 0..nz {
                for x in 0..nx {
                    bdr_tri([id(x, 0, z + 1), id(x, 0, z), id(x + 1, 0, z), id(x + 1, 0, z + 1)], 4);
                    bdr_tri([id(x, ny, z), id(x, ny, z + 1), id(x + 1, ny, z + 1), id(x + 1, ny, z)], 3);
                }
            }
            let mesh = Mesh::uniform(
                coords, conn, elem_tags, ElementType::Prism6,
                face_conn, face_tags, ElementType::Tri3,
            );
            Ok(MfemFile { mesh2d: None, mesh3d: Some(mesh) })
        }
        other => Err(FemError::Mesh(format!(
            "INLINE mesh: unsupported type '{other}' (supported: tri, quad, hex, tet, wedge)"
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

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    #[test]
    fn write_gf_emits_legacy_header_and_metadata() {
        // The custom "MFEM grid function v1.0" writer is still used by
        // ex23/ex32/ex34; pin its exact line layout so it cannot regress.
        let mut buf = Vec::new();
        write_gf(&mut buf, 2, &[1.0, 2.0, 3.0], "H1", 1, 1).unwrap();
        let text = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].starts_with("MFEM grid function v1.0"));
        assert!(lines.contains(&"dimension"));
        assert!(lines.contains(&"n_dofs"));
        assert!(lines.contains(&"order"));
        assert!(lines.contains(&"vdim"));
        assert!(lines.contains(&"space_type"));
        // DOF values are written with 16 significant digits.
        assert!(lines.iter().any(|l| l.starts_with("1.0000000000000000e")));
        assert_eq!(lines[lines.len() - 1], "3.0000000000000000e0");
    }

    #[test]
    fn elem_type_roundtrip() {
        let cases = [
            (ElementType::Line2, 1u32), (ElementType::Tri3, 2u32),            (ElementType::Quad4, 3u32), (ElementType::Tet4, 4u32),
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

    /// D41: the hex `nodes` mapper binds each geometry slot to MFEM's dof id
    /// (vertices, mesh-edge blocks, mesh-face blocks, per-element interior),
    /// and refuses — with a warning, never a scrambled table — when the
    /// section cannot be mapped.
    #[test]
    fn d41_hex_nodes_mapper_counts_and_safety_net() {
        let mesh = Mesh::<3>::unit_cube_hex(2); // 27 vertices, 8 hexes
        for p in 2..=4u8 {
            let n_dofs = DofManager::new(&mesh, p).n_dofs;
            let raw = vec![0.0f64; 3 * n_dofs];
            match build_h1_hex_geometry(&mesh, p, &raw, 0) {
                HexGeom::Built(g) => {
                    // MFEM's H1 dof count: NV + NE*(p-1) + NF*(p-1)^2 + NE*(p-1)^3.
                    let e = (p - 1) as usize;
                    let expected = 27 + 54 * e + 36 * e * e + 8 * e * e * e;
                    assert_eq!(g.n_nodes, expected, "p={p}");
                    assert_eq!(g.order, p);
                    assert_eq!(g.nodes_per_elem, (p as usize + 1).pow(3));
                    assert_eq!(g.conn.len(), 8 * g.nodes_per_elem);
                    // Every slot of an element resolves to a dof of its own
                    // space, and the shared entities are shared.
                    let mut seen = vec![false; g.n_nodes];
                    for &n in g.conn.iter() {
                        assert!((n as usize) < g.n_nodes);
                        seen[n as usize] = true;
                    }
                    assert!(seen.iter().all(|&s| s), "p={p}: unused geometry dof");
                }
                _ => panic!("p={p}: full-length nodes section must map"),
            }
            // One value short: refuse instead of building a partial table.
            let short = vec![0.0f64; 3 * n_dofs - 1];
            assert!(
                matches!(build_h1_hex_geometry(&mesh, p, &short, 0), HexGeom::Unsupported(_)),
                "p={p}: truncated section must be rejected"
            );
        }
        // Pure simplex meshes are not this mapper's business …
        let tets = Mesh::<3>::unit_cube_tet(1);
        assert!(matches!(
            build_h1_hex_geometry(&tets, 2, &vec![0.0; 300], 0),
            HexGeom::NotHex
        ));
        // … but a mixed mesh containing hexahedra cannot be numbered by either
        // path, so it is refused rather than silently mis-numbered.
        let mut mixed = Mesh::<3>::unit_cube_hex(1);
        mixed.conn.extend_from_slice(&[0, 1, 2, 4]);
        mixed.elem_types = Some(vec![ElementType::Hex8, ElementType::Tet4]);
        mixed.elem_offsets = Some(vec![0, 8, 12]);
        mixed.elem_tags.push(1);
        assert!(matches!(
            build_h1_hex_geometry(&mixed, 2, &vec![0.0; 300], 0),
            HexGeom::Unsupported(_)
        ));
    }
}
