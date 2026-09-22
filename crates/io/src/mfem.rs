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
use fem_element::lagrange::{H1PrismSlot, PRISM_EDGES};
use fem_element::ReferenceElement;
use fem_mesh::{
    element_type::ElementType,
    simplex::{GeometryData, Mesh},
    topology::MeshTopology,
};
use fem_space::dof_manager::DofManager;

use crate::data_collection::format_g;

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

    // Vertex indices are read raw here; the 0-based / 1-based decision needs
    // `n_vert` (see below), so the conversion happens once it is known.
    read_line(&mut r)?;  // "boundary"
    let n_bdr = read_uint(&mut r)?;
    let mut bdr_types: Vec<ElementType> = Vec::with_capacity(n_bdr);
    let mut face_raw: Vec<Vec<usize>> = Vec::with_capacity(n_bdr);
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
        face_raw.push(vals[2..].to_vec());
    }

    // ── "vertices" header (D589) ────────────────────────────────────────────
    //
    // MFEM 4.10 reads this region as whitespace-separated *tokens*, not lines
    // (`Mesh::ReadMFEMMesh`, `mesh/mesh_readers.cpp:100-118`):
    //
    //   input >> ident;          // 'vertices'
    //   input >> NumOfVertices;
    //   input >> ws >> ident;    // the NEXT token decides the reading:
    //   ident != "nodes" → spaceDim = atoi(ident), then NV*spaceDim coord tokens
    //   ident == "nodes" → curved = 1, a nodes GridFunction section follows
    //
    // `Mesh::Printer` writes `<NV>` and `<spaceDim>` on separate lines for
    // straight meshes and `<NV>` followed by a `nodes` line for curved ones
    // (`mesh/mesh.cpp:12560-12578`), so `vertices / 9 / 3 / <coords>` is the
    // canonical straight form — MFEM's own `data/star.mesh` carries
    // `vertices / 31 / 2`.  The token rule also makes "vertices 9 3" (one
    // line) and "vertices 9 nodes" equivalent layouts.  This reader stays
    // line-based, so trailing tokens of the keyword / count lines are carried
    // in `vert_tail` and consumed before new lines are read.  Coordinates
    // themselves are still read one vertex per line (the writer's layout; a
    // fully free token stream is not reproduced).
    //
    // Two deliberate divergences from MFEM (D600/D601):
    //   * MFEM drops vertices unreferenced by any element at this point
    //     (`RemoveUnusedVertices`, mesh_readers.cpp:38/130, default on) and
    //     renumbers; this reader keeps the file's vertex table, so `GetNV()`
    //     can differ for slack-vertex files.
    //   * the 0-based / 1-based index detection below silently repairs
    //     1-based files that MFEM would read verbatim (and garble).
    let mut vert_tail: Vec<String> = Vec::new();
    {
        let next = read_line(&mut r)?;  // "edges" or "vertices" (+ trailing)
        let mut toks = next.split_whitespace();
        let kw = toks.next().unwrap_or("");
        vert_tail.extend(toks.map(str::to_owned));
        if kw == "edges" {
            let n_edges = if vert_tail.is_empty() {
                read_uint(&mut r)?
            } else {
                take_tail_uint(&mut vert_tail)?
            };
            for _ in 0..n_edges { read_uint_line(&mut r)?; }
            if vert_tail.is_empty() {
                let vl = read_line(&mut r)?;  // "vertices" (+ trailing)
                vert_tail.extend(vl.split_whitespace().skip(1).map(str::to_owned));
            }
        } // else: the consumed line was the "vertices" keyword
    }
    let n_vert = if vert_tail.is_empty() {
        // The <NV> line may carry trailing tokens ("9 3", "9 nodes") — MFEM's
        // token reader accepts those layouts (mesh_readers.cpp:104-108).
        let l = read_line(&mut r)?;
        let mut toks = l.split_whitespace();
        let first = toks.next().unwrap_or("");
        vert_tail.extend(toks.map(str::to_owned));
        first.parse()
            .map_err(|_| FemError::Mesh(format!("MFEM: expected integer, got: {l}")))?
    } else {
        take_tail_uint(&mut vert_tail)?
    };

    // Detect 0-based vs 1-based vertex indexing.
    //
    // MFEM's own `Mesh::PrintElement` / `Mesh::ReadElementWithoutAttr` pass the
    // vertex ids through verbatim, so MFEM files are 0-based (every file in
    // `data/` contains a vertex `0`).  Older / foreign converters wrote
    // 1-based files instead, which is what the format note claims.  Decide by:
    //   * any index == 0                → 0-based (unambiguous);
    //   * max index + 1 == n_vert       → 0-based (the file addresses exactly
    //     the vertices it declares, which a 1-based file never does);
    //   * otherwise                     → 1-based (the historical default).
    let max_idx = elem_raw_conn
        .iter()
        .chain(face_raw.iter())
        .flatten()
        .copied()
        .max()
        .unwrap_or(0);
    let has_zero = elem_raw_conn
        .iter()
        .chain(face_raw.iter())
        .flatten()
        .any(|&v| v == 0);
    let is_zero_based = has_zero || max_idx + 1 == n_vert;

    // Convert to 0-based (subtract 1 if the file is 1-based, leave as-is if 0-based)
    let fix_idx = |v: usize| -> u32 {
        if is_zero_based { v as u32 } else { (v - 1) as u32 }
    };
    let elem_conn: Vec<Vec<u32>> = elem_raw_conn.iter()
        .map(|row| row.iter().map(|&v| fix_idx(v)).collect())
        .collect();
    let face_conn: Vec<Vec<u32>> = face_raw.iter()
        .map(|row| row.iter().map(|&v| fix_idx(v)).collect())
        .collect();

    let mut coords: Vec<f64> = Vec::new();
    // Per-element high-order geometry (MFEM `nodes` section).  For L2
    // (discontinuous) node spaces each element owns `nodes_per_elem`
    // independent geometry nodes — this is how geometrically periodic meshes
    // (e.g. `periodic-square.mesh`) encode per-element geometry.
    let mut geometry: Option<GeometryData> = None;
    // H1-continuous `nodes` section payload: (nodal order, values, ordering,
    // legacy closed-uniform node family — see `parse_nodal_fec`).
    let mut h1_nodes: Option<(u8, Vec<f64>, usize, bool)> = None;
    // `VDim` of the `nodes` section (D112b): the component stride.  Defaults to
    // the mesh dimension, which is what a conforming `nodes` section has.
    let mut nodes_vdim: usize = dim;

    // The token after <NV> decides straight vs curved (D589,
    // `mesh/mesh_readers.cpp:104-118`): "nodes" → the section is the
    // curved-mesh nodes GridFunction (`Mesh::Loader`, `mesh/mesh.cpp:5291-5300`
    // then sets spaceDim from the GF's VDim and overwrites the vertex table
    // from the first NV dofs — `SetVerticesFromNodes`); anything else is the
    // vertex space dimension (MFEM applies `atoi`) and NV coordinates follow.
    let next = if vert_tail.is_empty() {
        read_line(&mut r)?
    } else {
        vert_tail.remove(0)
    };
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
        // Straight format: the token after <NV> is the vertex space dimension
        // (MFEM `atoi`s it into spaceDim, mesh_readers.cpp:112) followed by
        // NV coordinate lines.  One vertex per line, first `dim` components
        // (MFEM reads NV*spaceDim tokens instead; the layouts coincide for
        // everything `Mesh::Printer` writes).
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
        nodes_vdim = vdim_line.split_whitespace().last()
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
        // `nodes_vdim` is the number of *components* per DOF, `dim` the
        // topological dimension of the mesh.  A conforming `nodes` section has
        // `VDim == dim`; `VDim > dim` means a mesh with `spaceDim > dim` (a
        // surface embedded in 3-D, MFEM's `Mesh::SetSpaceDim` case), which the
        // geometry path here cannot represent — see the warning below (D112b).
        if nodes_vdim != dim {
            eprintln!(
                "warning (D112b): `nodes` section has VDim={nodes_vdim} but the mesh is {dim}-dimensional; \
                 a `dim < spaceDim` (surface) mesh is not supported — the geometry is read with only its \
                 first {dim} components and its measure is the {dim}-dimensional one"
            );
        }
        if !is_l2_nodes {
            // Continuous (H1) geometry: remember the nodal order so the
            // high-order GeometryData can be attached once the mesh topology
            // is built (the DOF numbering needs it), plus whether the
            // collection is one of MFEM's legacy closed-uniform families
            // (D112) — those store values at the equispaced nodes.
            if let Some(fec) = parse_nodal_fec(&fec_name) {
                h1_nodes = Some((fec.order, raw.clone(), nodes_ordering, fec.closed_uniform));
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
                // `L2_T1_<dim>D_P<p>`: the polynomial order and the DOF lattice
                // come from the collection's name (D153 — the order used to be
                // hard-coded to 1, which made every P>1 L2 mesh a geometry with
                // `order = 1` but `(p+1)^dim` nodes per element).
                let l2_order = parse_nodal_fec(&fec_name).map(|f| f.order as usize).unwrap_or(1);
                let et0 = elem_types[0];
                // The file stores the values in MFEM's L2 element order, while
                // `GeometryData` must be in the slot order of the *mesh's* own
                // reference element (the one the assembler evaluates the
                // geometry with).  `perm[i]` is therefore the file's L2 index
                // of reference slot `i` — the inverse of the writer's
                // `lex_slot_permutation(factory, mfem_l2)` (`nodes_dof_values`).
                let perm: Vec<usize> = match (
                    l2_geometry_slots(et0, l2_order),
                    mfem_l2_slots(et0, l2_order),
                ) {
                    (Some(factory), Some(mfem)) => {
                        lex_slot_permutation(&mfem, &factory).unwrap_or_else(|| {
                            eprintln!(
                                "warning (D153): the {fec_name} `nodes` section of this mesh does \
                                 not sit on the {et0:?} P{l2_order} node lattice — the geometry is \
                                 read with the file's own DOF order, which is likely scrambled"
                            );
                            (0..npe).collect()
                        })
                    }
                    _ => {
                        eprintln!(
                            "warning (D153): no verified `{fec_name}` slot mapping for {et0:?} \
                             P{l2_order}; the geometry is read with the file's own DOF order, \
                             which is only correct when the two orderings coincide"
                        );
                        (0..npe).collect()
                    }
                };
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
                                // connectivity, which is also its reference
                                // slot (the vertices come first in every
                                // element family), so the file's L2 index of
                                // that vertex is `perm[k]`.
                                let kl = perm[k];
                                for c in 0..dim {
                                    coords[v * dim + c] = raw[(e * npe + kl) * dim + c];
                                }
                                break 'outer;
                            }
                        }
                    }
                }
                // 2) Per-element geometry table (non-shared nodes), in the
                //    mesh reference element's own slot order.
                let mut geo_conn: Vec<u32> = Vec::with_capacity(n_elem * npe);
                let mut geo_coords: Vec<f64> = Vec::with_capacity(n_elem * npe * dim);
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
                    order: l2_order as u8,
                    conn: geo_conn,
                    nodes_per_elem: npe,
                    coords: geo_coords,
                    n_nodes: n_elem * npe,
                });
            }
        } else if raw.len() >= n_vert * nodes_vdim {
            // Continuous (H1) geometry: vertex `i` is dof `i` of the `nodes`
            // grid function (MFEM `Mesh::Loader` → `SetVerticesFromNodes`).
            // The dof values are stored with the section's ordering:
            //   Ordering: 0 (byNODES) — raw = [x of all dofs, y of all, z …],
            //   Ordering: 1 (byVDIM)  — raw = interleaved [x y z] per dof.
            // Reading interleaved triples from a byNODES stream (the previous
            // behavior) scrambles the vertex coordinates of curved meshes
            // (e.g. `cube.mesh`: vertex 0 became (0, 0.5, 1) instead of the
            // origin, which then corrupts refinement midpoints).
            //
            // The component *stride* is `VDim`, not the mesh dimension: a
            // `dim < spaceDim` mesh stores `VDim = spaceDim` components per DOF
            // (see the D112b warning above), and using `dim` here crossed the
            // components of the whole vertex table.
            if nodes_ordering == 0 {
                let ndof = raw.len() / nodes_vdim;
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
                    coords.extend_from_slice(&raw[i * nodes_vdim..i * nodes_vdim + dim]);
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
            if let Some((p, raw, ord, _)) = &h1_nodes {
                mesh.geometry =
                    build_h1_geometry(&mesh, *p, raw, *ord, dim, nodes_vdim, None, None);
            }
        }
        repair_legacy_geometry(&mut mesh, &h1_nodes);
        // D530: MFEM's reader fixes negatively oriented elements at load time
        // (`Mesh::Mesh(filename, generate_edges, fix_geometry)` →
        // `FinalizeTriMesh` → `CheckElementOrientation(fix)`); without it a CW
        // triangle silently corrupts the HDiv/HCurl dof gauges.
        mesh.check_element_orientation(true);
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
            Some((p, _, _, _)) if mesh.geometry.is_none() => {
                tet_slot_map(&mesh, *p as usize).ok().map(|(conn, keys, _)| TetFileSlots { conn, keys })
            }
            _ => None,
        };
        // D624: for a *mixed* mesh the mixed geometry is attached after the
        // rotation through frame-invariant slot keys — capture the file-frame
        // values now (the dof ids below are the file's own).
        let mixed_values: Option<MixedFileValues> = match &h1_nodes {
            Some((p, raw, ord, _)) if mesh.geometry.is_none() && dim == nodes_vdim => {
                let mixed = {
                    let mut fams: Vec<ElementType> = Vec::new();
                    for e in 0..mesh.n_elements() as u32 {
                        let f = mixed_family_type(mesh.element_type(e));
                        if !fams.contains(&f) {
                            fams.push(f);
                        }
                    }
                    fams.len() > 1
                };
                if !mixed {
                    None
                } else {
                    let rows_mesh = MixedMeshRows {
                        rows: (0..mesh.n_elements() as u32)
                            .map(|e| (mesh.element_type(e), mesh.element_nodes(e).to_vec()))
                            .collect(),
                        n_nodes: mesh.n_nodes(),
                        dim,
                    };
                    let sdim = dim;
                    let ndof = raw.len().checked_div(nodes_vdim).unwrap_or(0);
                    let (raw_c, vd) = (raw.clone(), nodes_vdim);
                    let dof_xyz = move |d: usize| -> [f64; 3] {
                        let mut out = [0.0f64; 3];
                        match *ord {
                            0 => {
                                for c in 0..sdim {
                                    out[c] = raw_c.get(c * ndof + d).copied().unwrap_or(0.0);
                                }
                            }
                            _ => {
                                for c in 0..sdim {
                                    out[c] = raw_c.get(d * vd + c).copied().unwrap_or(0.0);
                                }
                            }
                        }
                        out
                    };
                    MixedFileValues::capture(&rows_mesh, *p as usize, dof_xyz).ok()
                }
            }
            _ => None,
        };
        // MFEM `Mesh::Loader` marks the mesh for refinement only through the
        // `meshgen & 1` simplex bit (`mesh/mesh.cpp:3111`, bitmask derived at
        // `mesh/mesh.cpp:5052-5098`: TETRAHEDRON/TRIANGLE/SEGMENT/POINT set
        // bit 1, HEXAHEDRON/QUADRILATERAL bit 2, WEDGE bit 4, PYRAMID bit 8).
        // A tet-free 3-D mesh is therefore *never* marked — its boundary
        // triangles keep the file's vertex order (the C++ re-save of
        // `wedge_curved2.mesh` equals the fixture byte for byte).  Marking
        // unconditionally here rotated every wedge mesh's end-cap triangles;
        // D295's `Mesh::Save` byte acceptance caught it.
        if (0..mesh.n_elements() as u32).any(|e| {
            matches!(
                mesh.element_type(e),
                fem_mesh::element_type::ElementType::Tet4
                    | fem_mesh::element_type::ElementType::Tet10
            )
        }) {
            fem_mesh::mark_tet_mesh_for_refinement(&mut mesh);
        }
        if mesh.geometry.is_none() {
            if let Some((p, raw, ord, _)) = &h1_nodes {
                mesh.geometry = build_h1_geometry(
                    &mesh,
                    *p,
                    raw,
                    *ord,
                    dim,
                    nodes_vdim,
                    tet_file_slots.as_ref(),
                    mixed_values.as_ref(),
                );
            }
        }
        repair_legacy_geometry(&mut mesh, &h1_nodes);
        // D530: `FinalizeTetMesh(generate_edges, refine, fix_orientation)` →
        // `CheckElementOrientation(fix)` — flipped tets are repaired by
        // `Swap(vi[0], vi[1])` before the refinement marking.
        mesh.check_element_orientation(true);
        Ok(MfemFile { mesh2d: None, mesh3d: Some(mesh) })
    }
}


/// Convenience: read MFEM file from disk.
pub fn read_mfem_file(path: impl AsRef<std::path::Path>) -> FemResult<MfemFile> {
    read_mfem(std::fs::File::open(path)?)
}

/// D126: self-check the element tables of a mesh that is about to be written.
///
/// The writer derives the number of volume elements and each element's node
/// count from `elem_type` / `elem_types` and then indexes `conn` with that
/// stride.  A mesh whose tables disagree with `conn` (e.g. `elem_type = Hex8`
/// while `conn` holds 6-node wedges — the `toroid` miniapp bug) used to be
/// written out as a *different, larger* set of elements read from the same
/// buffer, producing a file that neither MFEM nor `read_mfem` could read back.
/// Fail with the offending element instead.
fn check_element_tables<const D: usize>(mesh: &Mesh<D>) -> FemResult<()> {
    let n_conn = mesh.conn.len();
    if let Some(ref types) = mesh.elem_types {
        let n_elems = types.len();
        let offsets = mesh.elem_offsets.as_ref().ok_or_else(|| {
            FemError::Mesh(
                "write_mfem: elem_types is set but elem_offsets is None, so a mixed-element \
                 connectivity cannot be located".to_string(),
            )
        })?;
        if offsets.len() != n_elems + 1 {
            return Err(FemError::Mesh(format!(
                "write_mfem: elem_types has {n_elems} entries but elem_offsets has {} entries \
                 (expected {})",
                offsets.len(),
                n_elems + 1
            )));
        }
        for e in 0..n_elems {
            let et = types[e];
            let npe = et.nodes_per_element();
            if npe == 0 {
                return Err(FemError::Mesh(format!(
                    "write_mfem: element {e} of {n_elems} has element type {et:?} with 0 nodes \
                     per element"
                )));
            }
            let got = offsets[e + 1] - offsets[e];
            if got != npe {
                return Err(FemError::Mesh(format!(
                    "write_mfem: element {e} of {n_elems} has element type {et:?} ({npe} nodes) \
                     but elem_offsets gives {got} nodes"
                )));
            }
        }
        if offsets[n_elems] != n_conn {
            return Err(FemError::Mesh(format!(
                "write_mfem: elem_offsets ends at {} but conn has {n_conn} entries",
                offsets[n_elems]
            )));
        }
        return Ok(());
    }

    let npe = mesh.elem_type.nodes_per_element();
    if npe == 0 {
        return Err(FemError::Mesh(format!(
            "write_mfem: element type {:?} has 0 nodes per element",
            mesh.elem_type
        )));
    }
    if let Some(ref offsets) = mesh.elem_offsets {
        let n_elems = offsets.len() - 1;
        if offsets[n_elems] != n_conn {
            return Err(FemError::Mesh(format!(
                "write_mfem: elem_offsets ends at {} but conn has {n_conn} entries",
                offsets[n_elems]
            )));
        }
        return Ok(());
    }
    if n_conn % npe != 0 {
        return Err(FemError::Mesh(format!(
            "write_mfem: element type {:?} needs {npe} nodes per element, but conn has {n_conn} \
             entries — this is not a whole number of elements",
            mesh.elem_type
        )));
    }
    Ok(())
}

/// D126: per-face node count of a mesh's boundary section, or a `FemError`
/// naming the first face whose declared geometry does not fit the data.
///
/// The MFEM boundary records carry the face's *geometric* element type, so the
/// node count of a face must come from `Mesh::face_type_at` — never from a
/// hard-coded "3 nodes ⇒ TRIANGLE" guess, which turned quadrilateral faces into
/// truncated triangles (and re-read as a different mesh).  Both the per-face
/// stride and the total `face_conn` length are verified, so a mesh that would
/// write a corrupt file fails loudly instead.
fn check_boundary_tables<const D: usize>(mesh: &Mesh<D>) -> FemResult<Vec<usize>> {
    let n_faces = mesh.n_faces();
    if let Some(ref ft) = mesh.face_types {
        if ft.len() != n_faces {
            return Err(FemError::Mesh(format!(
                "write_mfem: face_types has {} entries but the mesh has {n_faces} boundary faces",
                ft.len()
            )));
        }
    }
    if !mesh.face_tags.is_empty() && mesh.face_tags.len() != n_faces {
        return Err(FemError::Mesh(format!(
            "write_mfem: face_tags has {} entries but the mesh has {n_faces} boundary faces",
            mesh.face_tags.len()
        )));
    }
    if let Some(ref fo) = mesh.face_offsets {
        if fo.len() != n_faces + 1 {
            return Err(FemError::Mesh(format!(
                "write_mfem: face_offsets has {} entries but the mesh has {n_faces} boundary \
                 faces (expected {})",
                fo.len(),
                n_faces + 1
            )));
        }
    }

    let mut counts = Vec::with_capacity(n_faces);
    let mut total = 0usize;
    for f in 0..n_faces {
        let et = mesh.face_type_at(f as u32);
        let nv = et.nodes_per_element();
        if nv == 0 {
            return Err(FemError::Mesh(format!(
                "write_mfem: boundary face {f} of {n_faces} has element type {et:?} with 0 nodes \
                 per element"
            )));
        }
        if let Some(ref fo) = mesh.face_offsets {
            let got = fo[f + 1] - fo[f];
            if got != nv {
                return Err(FemError::Mesh(format!(
                    "write_mfem: boundary face {f} of {n_faces} has element type {et:?} ({nv} \
                     nodes) but face_offsets gives {got} nodes"
                )));
            }
        } else if let Some(ref ft) = mesh.face_types {
            // Per-face types without an offsets table: the implicit
            // `f * face_type.nodes_per_element()` stride is only correct when
            // every face really has the uniform `face_type`.
            if ft[f] != mesh.face_type {
                return Err(FemError::Mesh(format!(
                    "write_mfem: boundary face {f} of {n_faces} has element type {:?} but \
                     face_offsets is None (the mesh's uniform face_type is {:?}), so the face's \
                     nodes cannot be located",
                    ft[f], mesh.face_type
                )));
            }
        }
        let remaining = mesh.face_conn.len().saturating_sub(total);
        if nv > remaining {
            return Err(FemError::Mesh(format!(
                "write_mfem: boundary face {f} of {n_faces} needs {nv} nodes (element type \
                 {et:?}) but only {remaining} of the {} face_conn entries remain",
                mesh.face_conn.len()
            )));
        }
        total += nv;
        counts.push(nv);
    }
    if total != mesh.face_conn.len() {
        return Err(FemError::Mesh(format!(
            "write_mfem: the {n_faces} boundary face types account for {total} nodes but \
             face_conn has {} entries",
            mesh.face_conn.len()
        )));
    }
    Ok(counts)
}

/// D126: validate both the element and the boundary tables of the mesh that is
/// about to be written.  Called by [`write_mfem`] (before it emits anything)
/// and by the `write_mfem_file*` helpers (before they create the file, so a
/// rejected mesh leaves no empty file behind).
fn validate_mesh_for_write(mesh_d: &Mesh<2>, mesh_3d: Option<&Mesh<3>>) -> FemResult<()> {
    if let Some(m3) = mesh_3d {
        check_element_tables(m3)?;
        check_boundary_tables(m3)?;
    } else {
        check_element_tables(mesh_d)?;
        check_boundary_tables(mesh_d)?;
    }
    Ok(())
}

/// Write a `Mesh` to MFEM `.mesh` v1.0 format.
///
/// Supports 2D and 3D meshes with uniform or mixed element types.
///
/// **Node indexing is 0-based** (D126): MFEM's `Mesh::PrintElement` writes
/// `v[j]` verbatim and `Mesh::ReadElementWithoutAttr` reads it verbatim
/// (`mesh/mesh.cpp`), so the vertex indices in a `.mesh` file are straight
/// indices into the `vertices` array — every file under `data/` (e.g.
/// `star.mesh`: `1 3 0 11 26 14`) uses vertex `0`.  Writing 1-based indices
/// (the previous behaviour here, following the format note "1-based") shifted
/// every connectivity entry by one, which made MFEM either abort with
/// `Invalid mesh topology` or overrun its vertex array.
///
/// **High-order geometry**: when the mesh carries a curved geometry table
/// (`Mesh::geom_order() > 1`, set by `Mesh::set_curvature`) the file gets an
/// MFEM `nodes` section and — exactly as `Mesh::Printer` does — the `vertices`
/// section holds only the vertex count, with the *space dimension* moving into
/// the section's `VDim` line (`mesh/mesh_readers.cpp:105-110`).  The continuous
/// (`H1_<dim>D_P<p>`) numbering is implemented for hexahedra, tetrahedra,
/// prisms (wedges), 2-D quadrilaterals and 2-D triangles and the discontinuous
/// (`L2_T1_<dim>D_P<p>`) one for hexahedra, quads, triangles and prisms; any
/// other combination is refused with an error instead of silently writing a
/// straight-sided mesh.  See [`NodesSpace`] and [`write_mfem_nodes`].
///
/// For 3D meshes containing tetrahedra *without* high-order geometry, the mesh
/// is cloned and normalized with `mark_tet_mesh_for_refinement` before writing,
/// so that programmatically created meshes round-trip with the same canonical
/// tet orientation that `read_mfem` produces (longest edge = (v0,v1)).  A mesh
/// that writes a `nodes` section keeps its own vertex order (the normalization
/// permutes element vertex lists and does not carry the geometry table).
pub fn write_mfem<W: Write>(writer: &mut W, mesh_d: &Mesh<2>, mesh_3d: Option<&Mesh<3>>) -> FemResult<()> {
    write_mfem_nodes(writer, mesh_d, mesh_3d, NodesSpace::Continuous)
}

/// [`write_mfem`] with an explicit continuity for the high-order `nodes`
/// section (MFEM `Mesh::SetCurvature`'s `discont` argument).
pub fn write_mfem_nodes<W: Write>(
    writer: &mut W,
    mesh_d: &Mesh<2>,
    mesh_3d: Option<&Mesh<3>>,
    space: NodesSpace,
) -> FemResult<()> {
    // D126: never write a mesh whose element/face tables contradict each other.
    // This runs before a single byte is emitted so a failure cannot leave a
    // half-written (or silently corrupt) `.mesh` behind.
    validate_mesh_for_write(mesh_d, mesh_3d)?;
    // The `nodes` payload is resolved before anything is emitted: a mesh whose
    // high-order geometry has no faithful MFEM numbering must fail *without*
    // leaving a file behind that silently drops the curvature.
    let nodes: Option<(u8, usize, Vec<f64>)> = if let Some(m3) = mesh_3d {
        nodes_dof_values(m3, space)?
    } else {
        nodes_dof_values(mesh_d, space)?
    };
    // D2: tet io round-trip orientation normalization.
    // read_mfem applies mark_tet_mesh_for_refinement (MarkTetMeshForRefinement)
    // on read to canonicalize tet vertex order.  write_mfem must apply the
    // same normalization so meshes created programmatically round-trip.
    //
    // The normalization only permutes each element's *vertex list*; a curved
    // mesh's geometry table is keyed by reference slot and is not carried
    // through it, so a mesh that writes a `nodes` section keeps its own vertex
    // order (the file is then self-consistent, and `read_mfem`'s geometry
    // table — which is built from the file's own element order — comes back
    // unchanged).
    let needs_normalization = nodes.is_none() && mesh_3d.map_or(false, has_tet4);
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
    let (dim, coords, conn, elem_tags, elem_type, elem_types_opt)
        = if let Some(m3) = mesh_3d {
            (3, &m3.coords, &m3.conn, &m3.elem_tags, &m3.elem_type, &m3.elem_types)
        } else {
            (2, &mesh_d.coords, &mesh_d.conn, &mesh_d.elem_tags, &mesh_d.elem_type,
             &mesh_d.elem_types)
        };
    // The `dimension` line is the *topological* dimension (MFEM's `Dim`), not
    // the coordinate count: a surface mesh is a `Mesh<3>` of `Quad4` elements
    // (`topological_dim() == 2`) whose `nodes` section carries the space
    // dimension separately in `VDim` (`Mesh::Printer` writes `Dim = 2` and
    // `spaceDim` only through the section).  For every volume mesh
    // `topological_dim() == dim`, so nothing changes there.
    let topo = if let Some(m3) = mesh_3d {
        m3.topological_dim() as usize
    } else {
        mesh_d.topological_dim() as usize
    };
    let n_nodes = coords.len() / dim;
    let n_elems = if dim == 3 {
        mesh_3d.map_or(conn.len() / elem_type.nodes_per_element(), |m| m.n_elems())
    } else if let Some(ref offsets) = mesh_d.elem_offsets {
        offsets.len() - 1
    } else {
        conn.len() / elem_type.nodes_per_element()
    };
    // D126: the number of boundary faces and each face's node count come from
    // the mesh's own face tables (`face_type_at`), validated up front by
    // `validate_mesh_for_write`.  The per-face counts are returned by the
    // check, so the write loop below cannot walk off the connectivity.
    let face_nv: Vec<usize> = if let Some(m3) = mesh_3d {
        check_boundary_tables(m3)?
    } else {
        check_boundary_tables(mesh_d)?
    };

    writeln!(writer, "MFEM mesh v1.0\n")?;
    // `Mesh::Printer` always prefixes the serial conforming format with the
    // geometry-type comment block (`mesh/mesh.cpp:12521-12531`), whatever the
    // element mix — the block is a fixed 12-line constant, not a summary of
    // the mesh's own types.  D294: emitted here so a written `.mesh` is
    // byte-identical to C++ `Mesh::Save` (the consumers that splice a
    // parallel header around this body — `tools/gridfunction_bounds.rs` —
    // pass the block through instead of adding their own).
    write!(
        writer,
        "#\n# MFEM Geometry Types (see fem/geom.hpp):\n#\n\
         # POINT       = 0\n\
         # SEGMENT     = 1\n\
         # TRIANGLE    = 2\n\
         # SQUARE      = 3\n\
         # TETRAHEDRON = 4\n\
         # CUBE        = 5\n\
         # PRISM       = 6\n\
         # PYRAMID     = 7\n\
         #\n"
    )?;
    // `Mesh::Printer` writes `"\ndimension\n" << Dim` — the blank line after
    // the geometry block belongs to this write (`mesh/mesh.cpp:12533`).
    writeln!(writer, "\ndimension\n{topo}\n")?;

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
                write!(writer, " {}", conn[offset + j])?;
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
                write!(writer, " {}", conn[offset + j])?;
            }
            writeln!(writer)?;
        }
    }

    // Boundary section
    //
    // Each record is `<attr> <mfem geometry code> <n1> ... <nn>`, with the code
    // and the node count taken from the face's own geometric type (D126).
    if let Some(m3) = mesh_3d {
        write_boundary_section(writer, m3, &face_nv)?;
    } else {
        write_boundary_section(writer, mesh_d, &face_nv)?;
    }

    // Vertices section
    //
    // MFEM writes the vertex *coordinates* only for a straight-sided mesh; a
    // mesh with a `nodes` grid function emits `vertices / <n>` followed by the
    // `nodes` section instead (`Mesh::Printer`), because the geometry is then
    // fully described by the node dof values and even the space dimension is
    // carried by the section's `VDim` (`mesh/mesh_readers.cpp:105-110`).
    writeln!(writer, "\nvertices\n{n_nodes}")?;
    if let Some((order, n_dofs, values)) = nodes.as_ref() {
        // The section's collection name counts the *topological* dimension
        // (`H1_2D_P3` for a surface, even though the values carry 3
        // components); `VDim` below carries the space dimension.  The
        // straight-sided branch stays keyed by the coordinate count `dim`.
        write_nodes_section(writer, space, *order, topo, *n_dofs, values)?;
    } else {
        // Straight-sided: `<space dim>` then one coordinate row per vertex —
        // `Mesh::Printer` (`mesh/mesh.cpp:12544`): `os << vertices[i](0)`
        // first (no leading blank), the remaining components each prefixed by
        // `' '`, values rendered at the stream precision (`Mesh::Save`'s
        // default, `mesh/mesh.hpp:2616`: `precision = 16`).
        writeln!(writer, "{dim}")?;
        for i in 0..n_nodes {
            for d in 0..dim {
                if d > 0 {
                    write!(writer, " ")?;
                }
                write!(writer, "{}", format_g(coords[i * dim + d], MFEM_SAVE_PRECISION))?;
            }
            writeln!(writer)?;
        }
    }
    Ok(())
}

// ─── High-order `nodes` section (writer) ─────────────────────────────────────

/// Continuity of the high-order `nodes` grid function a written `.mesh` file
/// declares in its `FiniteElementSpace` header — MFEM `Mesh::SetCurvature`'s
/// `discont` argument (`mesh/mesh.cpp:7211`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NodesSpace {
    /// `SetCurvature(order, false, …)`: the node dofs are a `H1_FECollection`
    /// space (`H1_<dim>D_P<p>`), one dof per mesh vertex / edge / face /
    /// element interior.
    #[default]
    Continuous,
    /// `SetCurvature(order, true, …)`: the node dofs are an `L2_FECollection`
    /// with Gauss-Lobatto points (`L2_T1_<dim>D_P<p>`), `(p+1)^dim` private
    /// dofs per element for the tensor families (`(p+1)²(p+2)/2` for the
    /// wedge).
    Discontinuous,
}

/// The `nodes` section header and payload, exactly as MFEM's
/// `Mesh::Printer` (`mesh/mesh.cpp:12551`) + `GridFunction::Save`
/// (`fem/gridfunc.cpp:4305`) write them for the ordering the nodal space
/// carries.  MFEM 4.10's `Mesh::SetCurvature` defaults to `ordering = 1`
/// (`byVDIM`, `mesh/mesh.hpp:2439`), and `Printer` round-trips whatever the
/// read file declared, so the writer emits `Ordering: 1`:
///
/// ```text
/// nodes
/// FiniteElementSpace
/// FiniteElementCollection: <fec>
/// VDim: <space dim>
/// Ordering: 1
/// <blank>
/// <x y [z]> of dof 0
/// <x y [z]> of dof 1
/// …
/// ```
///
/// Values go through `Vector::Print` (`linalg/vector.cpp:870`): each one is
/// `ZeroSubnormal`-flushed and printed with a single separating space, one
/// newline after every `VDim` values, one final newline, at the stream
/// precision — `Mesh::Save`'s default `precision = 16`
/// (`mesh/mesh.hpp:2616`).  (A mesh read back from an `Ordering: 0` file is
/// re-emitted as `Ordering: 1`: the reader normalises the geometry table to
/// slots and the file's ordering bit is not carried in memory — the same
/// normalisation `Mesh::SetCurvature` applies on the C++ side.)
///
/// `values[d * sdim + c]` is component `c` of dof `d`.  `dim` is the mesh's
/// *topological* dimension (the `P<dim>D` of the collection name) while `sdim`
/// is the space dimension (`VDim`); the two agree for every mesh this writer
/// accepts, because `fem_mesh::Mesh<D>` stores `D` coordinate components.
fn write_nodes_section<W: Write>(
    writer: &mut W,
    space: NodesSpace,
    order: u8,
    dim: usize,
    n_dofs: usize,
    values: &[f64],
) -> FemResult<()> {
    // `H1_%dD_P%d` (`H1_FECollection`'s constructor, `fem/fe_coll.cpp:1760`) or
    // `L2_T1_%dD_P%d` (`L2_FECollection` with `Quadrature1D` type 1 = the
    // Gauss-Lobatto points `Mesh::SetCurvature` selects).
    let family = match space {
        NodesSpace::Continuous => "H1",
        NodesSpace::Discontinuous => "L2_T1",
    };
    let sdim = values.len() / n_dofs;
    writeln!(
        writer,
        "\nnodes\nFiniteElementSpace\nFiniteElementCollection: {family}_{dim}D_P{order}\n\
         VDim: {sdim}\nOrdering: 1\n"
    )?;
    // `GridFunction::Save` calls `Vector::Print(os, fes->GetVDim())` for a
    // byVDIM space: `ZeroSubnormal`-flushed values separated by a single
    // space, one newline after every `VDim` values, and one final newline
    // (`linalg/vector.cpp:870`), printed at the stream precision
    // (`Mesh::Save` default: 16 significant digits).
    for d in 0..n_dofs {
        for c in 0..sdim {
            if c > 0 {
                write!(writer, " ")?;
            }
            let v = zero_subnormal(values[d * sdim + c]);
            write!(writer, "{}", format_g(v, MFEM_SAVE_PRECISION))?;
        }
        writeln!(writer)?;
    }
    Ok(())
}

/// `Mesh::Save`'s default stream precision (`mesh/mesh.hpp:2616`:
/// `virtual void Save(const std::string &fname, int precision = 16)`).
/// `Mesh::Print(os)` itself inherits the *caller's* precision — MFEM harnesses
/// that print at `os.precision(17)` produce one more digit round-trip; the
/// file-writer default that `Mesh::Save`/`r31_save` exercise is 16.
const MFEM_SAVE_PRECISION: usize = 16;

/// MFEM `Vector::Print`'s `ZeroSubnormal` (`linalg/vector.cpp:857`): a
/// subnormal magnitude (either sign) is flushed to `+0.0` before printing.
fn zero_subnormal(s: f64) -> f64 {
    if s.abs() < f64::MIN_POSITIVE {
        0.0
    } else {
        s
    }
}

/// The `nodes` dof values of `mesh`, laid out for [`write_nodes_section`], plus
/// the element order and the dof count.
///
/// Returns `Ok(None)` for a straight-sided mesh (`geom_order() == 1`), which
/// MFEM stores as a plain `vertices` coordinate block.
///
/// **The numbering is MFEM's, not fem-rs's.**  For a continuous space the
/// per-slot maps are the ones the reader uses ([`hex_slot_map`] for D41,
/// [`tet_slot_map`] for D43, [`quad2d_slot_map`] / [`tri2d_slot_map`] for the
/// 2-D families) — this direction is their exact inverse, and it
/// is available because both element families place their dofs on the *same*
/// nodal lattice as MFEM's `H1_HexahedronElement` / `H1_TetrahedronElement`,
/// so a value read at one slot can be handed to the matching dof unchanged.
/// For a discontinuous space the dof of slot `d` of element `e` is simply
/// `e * npe + d`, with `d` indexing MFEM's `L2FECollection` element ordering
/// (the lexicographic tensor order for the tensor families, the layer ×
/// L2-triangle order for the wedge), which is a permutation of the mesh's own
/// slot order.
fn nodes_dof_values<const D: usize>(
    mesh: &Mesh<D>,
    space: NodesSpace,
) -> FemResult<Option<(u8, usize, Vec<f64>)>> {
    let order = mesh.geom_order();
    if order <= 1 {
        return Ok(None);
    }
    let dim = mesh.topological_dim() as usize;
    let et = mesh.element_type_at(0);
    if dim != D {
        // The one faithful surface representation (`Mesh::MakeCartesian2D` +
        // `SetCurvature(order, …, 3, Ordering::byVDIM)`, as in the
        // mobius-strip / klein-bottle miniapps): a dimension-2 mesh whose
        // elements are the intrinsically 2-D `Quad4`, stored in 3-D
        // coordinates.  The 2-D `Quad4` numbering below is element-local, so
        // it applies unchanged; only the node values then carry `D = 3`
        // components (`VDim: 3`).  Everything else (a "2-D hex", a 1-D
        // element in 2-D, …) has no faithful MFEM numbering here and stays
        // refused rather than silently mis-numbered.
        if !(D == 3 && dim == 2 && et == ElementType::Quad4) {
            return Err(FemError::Mesh(format!(
                "write_mfem: `nodes` section for a {dim}-dimensional mesh in {D}-D space \
                 (spaceDim > dim) is not supported"
            )));
        }
    }
    let sdim = D;
    let geo = mesh.geometry.as_ref().ok_or_else(|| {
        FemError::Mesh(
            "write_mfem: mesh reports geometric order > 1 but carries no geometry table".into(),
        )
    })?;
    let n_elems = mesh.n_elements();
    let npe = geo.nodes_per_elem;
    if npe == 0 {
        // D624/D627: a ragged (mixed-mesh) table has per-family rows; the
        // writer's entity numbering below assumes one uniform stride.  (The
        // mixed renumbering is registered as D627.)
        return Err(FemError::Mesh(
            "write_mfem: the mesh carries a ragged (mixed-element) high-order geometry table; \
             writing a `nodes` section for a mixed mesh is not supported yet (D627)"
                .into(),
        ));
    }
    if geo.conn.len() != n_elems * npe || geo.n_nodes == 0 {
        return Err(FemError::Mesh(format!(
            "write_mfem: geometry table is inconsistent with the mesh ({} elements, \
             {npe} nodes per element, {} connectivity entries, {} nodes)",
            n_elems,
            geo.conn.len(),
            geo.n_nodes
        )));
    }
    for e in 1..n_elems as u32 {
        if mesh.element_type_at(e) != et {
            return Err(FemError::Mesh(format!(
                "write_mfem: a `nodes` section needs a uniform element type, but the mesh \
                 mixes {et:?} and {:?}",
                mesh.element_type_at(e)
            )));
        }
    }
    let src = |slot: usize| -> &[f64] {
        let n = geo.conn[slot] as usize;
        &geo.coords[n * sdim..n * sdim + sdim]
    };
    match space {
        NodesSpace::Continuous => {
            let (slots, n_dofs): (Vec<NodeId>, usize) = match et {
                ElementType::Hex8 => {
                    let (slots, n_dofs) = hex_slot_map(mesh, order as usize).map_err(|e| match e {
                        HexSlotErr::NotHex => FemError::Mesh(
                            "write_mfem: no MFEM H1 `nodes` numbering for this mesh".into(),
                        ),
                        HexSlotErr::Unsupported(why) => FemError::Mesh(format!(
                            "write_mfem: cannot write the hexahedral `nodes` section: {why}"
                        )),
                    })?;
                    (slots, n_dofs)
                }
                ElementType::Tet4 => {
                    // Only the slot -> dof map is needed here; the extra
                    // physical keys the reader keeps for MFEM's `refine = 1`
                    // renumbering are not part of the file's numbering.
                    let (slots, _, n_dofs) = tet_slot_map(mesh, order as usize).map_err(|why| {
                        FemError::Mesh(format!(
                            "write_mfem: cannot write the tetrahedral `nodes` section: {why}"
                        ))
                    })?;
                    (slots, n_dofs)
                }
                ElementType::Quad4 => {
                    let (slots, n_dofs) = quad2d_slot_map(mesh, order as usize).map_err(|e| match e {
                        HexSlotErr::NotHex => FemError::Mesh(
                            "write_mfem: no MFEM H1 `nodes` numbering for this mesh".into(),
                        ),
                        HexSlotErr::Unsupported(why) => FemError::Mesh(format!(
                            "write_mfem: cannot write the 2-D quadrilateral `nodes` section: {why}"
                        )),
                    })?;
                    (slots, n_dofs)
                }
                // D151: the prism table pairs MFEM's entity-ordered H1 wedge
                // slots with the mesh's layer-major `PrismPk` slots, so it goes
                // through the interpolation matrix instead of a direct
                // permutation (round 34, D164: the two lattices are both GLL,
                // so the matrix is the identity on nodes).
                ElementType::Prism6 => {
                    let (n_dofs, v) = prism_nodes_dof_values(mesh, geo, order as usize, sdim)?;
                    return Ok(Some((order, n_dofs, v)));
                }
                // D178: the 2-D triangle analogue of `quad2d_slot_map` —
                // `H1TriPk` and MFEM's `H1_TriangleElement` place their dofs
                // on the same Gauss-Lobatto lattice in the same slot order
                // (probe-verified point-for-point), so this is a pure
                // renumbering.
                ElementType::Tri3 => {
                    let (slots, n_dofs) = tri2d_slot_map(mesh, order as usize).map_err(|e| match e {
                        HexSlotErr::NotHex => FemError::Mesh(
                            "write_mfem: no MFEM H1 `nodes` numbering for this mesh".into(),
                        ),
                        HexSlotErr::Unsupported(why) => FemError::Mesh(format!(
                            "write_mfem: cannot write the 2-D triangular `nodes` section: {why}"
                        )),
                    })?;
                    (slots, n_dofs)
                }
                other => {
                    return Err(FemError::Mesh(format!(
                        "write_mfem: no MFEM-faithful continuous `nodes` numbering for \
                         {other:?} (only Hex8, Tet4, Prism6, Quad4 and Tri3 are implemented); \
                         no `nodes` section was written"
                    )))
                }
            };
            let mut values = vec![0.0f64; n_dofs * sdim];
            let mut filled = vec![false; n_dofs];
            // MFEM's `SetNodalFESpace` fills a continuous `nodes` section via
            // `GridFunction::ProjectCoefficient`, which processes the elements
            // **in order** and writes each element's dof values wholesale — a
            // shared dof keeps the *last* element's value.  For a volume mesh
            // the geometry is continuous, so every writer agrees and the
            // conflict check below is a genuine consistency guard.  For the
            // `Dim = 2, spaceDim = 3` surface path, though, disagreement is
            // MFEM's own semantic: the miniapps identify vertices *after*
            // `SetCurvature`, so the seam elements still carry their
            // pre-identification samples and `ProjectCoefficient` resolves the
            // shared edge/vertex dofs last-writer-wins (measured on the MFEM
            // 4.10 mobius-strip / klein-bottle outputs).  Reproducing that
            // requires the overwrite, not the error.
            let strict_continuity = dim == D;
            for e in 0..n_elems {
                for s in 0..npe {
                    let g = slots[e * npe + s] as usize;
                    let v = src(e * npe + s);
                    if filled[g] {
                        // A dof shared by several elements must describe one
                        // physical point; a disagreement means the mesh's
                        // geometry is not actually continuous.
                        if strict_continuity
                            && (0..sdim).any(|c| !approx_eq(values[g * sdim + c], v[c]))
                        {
                            return Err(FemError::Mesh(format!(
                                "write_mfem: geometry dof {g} is shared by two elements with \
                                 different coordinates — the mesh geometry is not continuous"
                            )));
                        }
                        values[g * sdim..g * sdim + sdim].copy_from_slice(v);
                    } else {
                        values[g * sdim..g * sdim + sdim].copy_from_slice(v);
                        filled[g] = true;
                    }
                }
            }
            if let Some(g) = filled.iter().position(|f| !*f) {
                return Err(FemError::Mesh(format!(
                    "write_mfem: no element claims the continuous `nodes` dof {g}"
                )));
            }
            Ok(Some((order, n_dofs, values)))
        }
        NodesSpace::Discontinuous => {
            // MFEM's `L2_FECollection(p, dim, 1)` element enumeration, given as
            // a permutation of the mesh's own geometry slot order (both are the
            // same nodal lattice, so the slots are matched by their reference
            // coordinates).
            let (factory, mfem) = (
                l2_geometry_slots(et, order as usize),
                mfem_l2_slots(et, order as usize),
            );
            let (factory, mfem) = match (factory, mfem) {
                (Some(f), Some(m)) => (f, m),
                _ => {
                    return Err(FemError::Mesh(format!(
                        "write_mfem: no MFEM-faithful discontinuous `nodes` numbering for \
                         {et:?} (only Hex8, Quad4, Tri3 and Prism6 are implemented); no \
                         `nodes` section was written"
                    )))
                }
            };
            let perm: Vec<usize> = lex_slot_permutation(&factory, &mfem).ok_or_else(|| {
                FemError::Mesh(
                    "write_mfem: the mesh's geometry slots are not on the element's own node \
                     lattice, so they cannot be re-ordered into MFEM's L2 numbering"
                        .into(),
                )
            })?;
            let n_dofs = n_elems * npe;
            let mut values = vec![0.0f64; n_dofs * sdim];
            for e in 0..n_elems {
                for (d, &s) in perm.iter().enumerate() {
                    values[(e * npe + d) * sdim..(e * npe + d + 1) * sdim]
                        .copy_from_slice(src(e * npe + s));
                }
            }
            Ok(Some((order, n_dofs, values)))
        }
    }
}

/// The reference-element DOF coordinates a mesh's own geometry table is stored
/// in, for the element families whose L2 (`nodes`) numbering this module knows
/// — the element the assembler uses to evaluate the geometry
/// (`element_jacobian` / `CurvedMesh` both go through the `lagrange::factory`
/// element of the mesh's type and geometric order).
///
/// All of them place their DOFs on the **closed Gauss-Lobatto** points:
/// `QuadQk`/`HexQk` tensor GLL, `H1TriPk` the `w`-normalised barycentric GLL
/// nodes of `H1_TriangleElement`.
fn l2_geometry_slots(et: ElementType, p: usize) -> Option<Vec<Vec<f64>>> {
    use fem_element::lagrange::factory::{HexQk, QuadQk};
    if p == 0 {
        return None;
    }
    Some(match et {
        ElementType::Quad4 => QuadQk::new(p).dof_coords(),
        ElementType::Hex8 => HexQk::new(p).dof_coords(),
        ElementType::Tri3 => fem_element::lagrange::H1TriPk::new(p).dof_coords(),
        ElementType::Prism6 => fem_element::lagrange::PrismPk::new(p).dof_coords(),
        _ => return None,
    })
}

/// MFEM's `L2_T1_<dim>D_P<p>` node reference coordinates **in the element's own
/// DOF order** (`fem/fe/fe_l2.cpp`):
///
/// * `L2_QuadrilateralElement` / `L2_HexahedronElement` are
///   `NodalTensorFiniteElement`s with `L2_DOF_MAP`, which leaves `dof_map`
///   empty — so the nodes are simply lexicographic (`ix` fastest,
///   `ix + iy·(p+1) + iz·(p+1)²`), exactly what [`QuadQk::new_lex`] /
///   [`HexQk::new_lex`] enumerate (already pinned against MFEM's own
///   `L2_T1_3D_P3` output by `crates/io/tests/nodes_writer.rs`);
/// * `L2_TriangleElement` enumerates its `(p+1)(p+2)/2` nodes as
///   `for (j = 0..p) for (i = 0..p-j)` over the `w`-normalised Gauss-Lobatto
///   barycentric points `(op[i]/w, op[j]/w)`, `w = op[i]+op[j]+op[p-i-j]` —
///   the *same point set* as `H1TriPk`, in a different order;
/// * `L2_WedgeElement` (`fe_l2.cpp:839`) composes `L2_TriangleElement ×
///   L2_SegmentElement`: its `t_dof`/`s_dof` fill loop leaves
///   `t_dof[m] = m mod T`, `s_dof[m] = m / T` (`T = (p+1)(p+2)/2` — the
///   inner `for (j) for (i<=j)` double loop only counts to `T`, and `l`
///   increments alongside `m`), so node `m = k·T + l` sits at
///   `(t_Nodes[l].x, t_Nodes[l].y, s_Nodes[k].x)`: **layer-major** over the
///   closed Gauss-Lobatto points of the segment, with the *triangle's own L2
///   order* inside every layer (verified against MFEM 4.10's
///   `L2_WedgeElement(p).GetNodes()` and its `flatprism_l2` `-dm` files).
///   The point set coincides with `PrismPk`'s GLL lattice (both build on
///   `Poly_1D`'s `BasisType::GaussLobatto` closed points — classical
///   Gauss-Lobatto-Legendre, `p = 3`: `{0, 0.276393…, 0.723607…, 1}`).
/// * every other family (tets, pyramids) is left out: its L2 ordering has not
///   been verified against MFEM here.
fn mfem_l2_slots(et: ElementType, p: usize) -> Option<Vec<Vec<f64>>> {
    use fem_element::lagrange::factory::{HexQk, QuadQk};
    if p == 0 {
        return None;
    }
    Some(match et {
        ElementType::Quad4 => QuadQk::new_lex(p).dof_coords(),
        ElementType::Hex8 => HexQk::new_lex(p).dof_coords(),
        ElementType::Tri3 => {
            let gll: Vec<f64> = fem_element::quadrature::gauss_lobatto_arbitrary(p + 1)
                .0
                .iter()
                .map(|&x| 0.5 * (x + 1.0))
                .collect();
            let mut slots = Vec::with_capacity((p + 1) * (p + 2) / 2);
            for j in 0..=p {
                for i in 0..=(p - j) {
                    let w = gll[i] + gll[j] + gll[p - i - j];
                    slots.push(vec![gll[i] / w, gll[j] / w]);
                }
            }
            slots
        }
        ElementType::Prism6 => {
            // `L2_WedgeElement`: node `k·T + l` = layer `gll[k]` of the
            // segment × triangle node `l` (`L2_TriangleElement` order), in
            // `PrismPk`'s `[z, x, y]` reference convention.
            let gll: Vec<f64> = fem_element::quadrature::gauss_lobatto_arbitrary(p + 1)
                .0
                .iter()
                .map(|&x| 0.5 * (x + 1.0))
                .collect();
            let tri = (p + 1) * (p + 2) / 2;
            let mut slots = Vec::with_capacity((p + 1) * tri);
            for k in 0..=p {
                for j in 0..=p {
                    for i in 0..=(p - j) {
                        let w = gll[i] + gll[j] + gll[p - i - j];
                        slots.push(vec![gll[k], gll[i] / w, gll[j] / w]);
                    }
                }
            }
            slots
        }
        _ => return None,
    })
}

/// Permutation taking `mesh_slots` (the order the mesh's geometry element
/// enumerates its nodes) to `target_slots` (the order MFEM's element uses for
/// the same lattice), matched by reference coordinate.  `None` when the two are
/// not the same set of points (then no faithful re-ordering exists).
fn lex_slot_permutation(mesh_slots: &[Vec<f64>], target_slots: &[Vec<f64>]) -> Option<Vec<usize>> {
    if mesh_slots.len() != target_slots.len() {
        return None;
    }
    let same = |a: &[f64], b: &[f64]| {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12)
    };
    let mut perm = Vec::with_capacity(target_slots.len());
    let mut used = vec![false; mesh_slots.len()];
    for t in target_slots {
        let found = mesh_slots
            .iter()
            .enumerate()
            .position(|(i, m)| !used[i] && same(m, t))?;
        used[found] = true;
        perm.push(found);
    }
    Some(perm)
}

/// Relative comparison for two dof coordinates that must describe the same
/// physical point (bit-exact agreement is not required: the two elements
/// compute the shared node through different parametrisations).
fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
}

/// Returns `true` if the 3D mesh contains any Tet4 elements (uniform or mixed).
fn has_tet4(mesh: &Mesh<3>) -> bool {
    if let Some(ref etypes) = mesh.elem_types {
        etypes.iter().any(|et| *et == ElementType::Tet4)
    } else {
        mesh.elem_type == ElementType::Tet4
    }
}

/// Write the `.mesh` `boundary` section for a 2-D or 3-D mesh.
///
/// `face_nv[f]` is the validated node count of face `f` (from
/// [`check_boundary_tables`]); `face_conn` is walked with those counts, so the
/// records can never overlap or overrun.  The MFEM geometry code comes from the
/// face's own type via [`elem_type_to_mfem_code`].
fn write_boundary_section<W: Write, const D: usize>(
    writer: &mut W,
    mesh: &Mesh<D>,
    face_nv: &[usize],
) -> FemResult<()> {
    writeln!(writer, "\nboundary\n{}", face_nv.len())?;
    let mut off = 0usize;
    for (fi, &nvf) in face_nv.iter().enumerate() {
        let et = mesh.face_type_at(fi as u32);
        let code = elem_type_to_mfem_code(et).ok_or_else(|| {
            FemError::Mesh(format!("write_mfem: unsupported boundary face type {et:?}"))
        })?;
        let tag = if !mesh.face_tags.is_empty() { mesh.face_tags[fi] } else { 1 };
        write!(writer, "{tag} {code}")?;
        for j in 0..nvf {
            write!(writer, " {}", mesh.face_conn[off + j])?;
        }
        writeln!(writer)?;
        off += nvf;
    }
    Ok(())
}

/// Write a mesh to MFEM `.mesh` file on disk.
///
/// The mesh is validated (D126) *before* the file is created, so a rejected
/// mesh leaves no empty file behind.
pub fn write_mfem_file(path: impl AsRef<std::path::Path>, mesh_d: &Mesh<2>) -> FemResult<()> {
    write_mfem_bytes_to(path, &mut |w: &mut Vec<u8>| write_mfem(w, mesh_d, None))
}

/// Write a 3D mesh to MFEM `.mesh` file on disk.
///
/// The file is only created once the whole payload has been produced (the mesh
/// tables are validated by D126 and a high-order `nodes` section must have a
/// faithful MFEM numbering), so a rejected mesh leaves no empty file behind.
pub fn write_mfem_file_3d(path: impl AsRef<std::path::Path>, mesh: &Mesh<3>) -> FemResult<()> {
    write_mfem_file_3d_nodes(path, mesh, NodesSpace::Continuous)
}

/// [`write_mfem_file_3d`] with an explicit continuity for the `nodes` section.
pub fn write_mfem_file_3d_nodes(
    path: impl AsRef<std::path::Path>,
    mesh: &Mesh<3>,
    space: NodesSpace,
) -> FemResult<()> {
    write_mfem_bytes_to(path, &mut |w: &mut Vec<u8>| {
        write_mfem_nodes(w, &Mesh::<2>::unit_square_tri(2), Some(mesh), space)
    })
}

/// Render a mesh into a buffer and only then create `path`, so a failure to
/// serialize leaves no partially written file behind.
fn write_mfem_bytes_to(
    path: impl AsRef<std::path::Path>,
    render: &mut dyn FnMut(&mut Vec<u8>) -> FemResult<()>,
) -> FemResult<()> {
    let mut buf: Vec<u8> = Vec::new();
    render(&mut buf)?;
    let mut file = std::fs::File::create(path)?;
    file.write_all(&buf)?;
    Ok(())
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

/// The polynomial order of a nodal `FiniteElementCollection` name together with
/// its DOF *node family* (D112).
struct NodalFec {
    order: u8,
    /// `true` for MFEM's legacy fixed-order collections (`Linear`,
    /// `Quadratic`/`QuadraticPos`, `Cubic`), whose DOFs sit at the
    /// **closed-uniform** (equispaced) points instead of the closed
    /// Gauss-Lobatto points `H1_FECollection` uses.
    closed_uniform: bool,
}

/// Classify the `FiniteElementCollection` name of a `nodes` section (D112).
///
/// MFEM's legacy collections are named by their order alone (`Linear`,
/// `Quadratic`, `Cubic` — `fem/fe_coll.hpp`), while a modern collection carries
/// the family and the dimension (`H1_2D_P3`, `L2_3D_P2`, …).  The legacy
/// family is exactly the one whose 1-D DOF nodes are `BasisType::ClosedUniform`
/// (`Lagrange1DFiniteElement`, `BiCubic2DFiniteElement`, `Cubic2DFiniteElement`,
/// `Cubic3DFiniteElement`, `LagrangeHexFiniteElement` all place their DOFs at
/// `i/degree`), so the distinction decides whether the stored values may be
/// handed to the Gauss-Lobatto-based geometry elements unchanged.
///
/// The *discontinuous* collections never describe a mesh's `nodes` grid
/// function and are excluded so that a stray `LinearDiscont2D` cannot be
/// mistaken for `Linear`.
fn parse_nodal_fec(fec: &str) -> Option<NodalFec> {
    let f = fec.trim();
    let legacy = if f.starts_with("LinearDiscont") || f.starts_with("QuadraticDiscont") {
        None
    } else if f.starts_with("Linear") || f.starts_with("Quadratic") || f.starts_with("Cubic") {
        // `Linear`/`LinearF` → 1, `Quadratic`/`QuadraticPos` → 2, `Cubic` → 3
        // (the same prefixes the order parser accepts).
        Some(parse_nodal_fec_order(f)?)
    } else {
        None
    };
    match legacy {
        Some(order) => Some(NodalFec { order, closed_uniform: true }),
        None => parse_nodal_fec_order(f).map(|order| NodalFec { order, closed_uniform: false }),
    }
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

/// Failure of the D41 hexahedral `nodes` map.  `NotHex` (the mesh has no
/// hexahedra at all) is a *routing* answer — the caller falls back to the
/// simplex paths — while `Unsupported` means the mesh is hexahedral but its
/// `nodes` numbering cannot be reproduced faithfully.
enum HexSlotErr {
    NotHex,
    Unsupported(&'static str),
}

impl HexSlotErr {
    /// Shorthand for the `Unsupported` variant as a function result.
    fn unsupported(why: &'static str) -> Result<(Vec<NodeId>, usize), HexSlotErr> {
        Err(HexSlotErr::Unsupported(why))
    }
}

/// D41: MFEM's H1 `nodes` numbering for an all-Hex8 mesh, as the map
/// `conn[e * npe + s] = <file dof of reference slot s of element e>` plus the
/// total number of geometry dofs.  Slots are in the reference element's
/// ([`HexQk`]) order.
///
/// This is the single source of truth for the hexahedral numbering: the reader
/// uses it to place a file's dof values into the geometry table, and the
/// writer uses it to emit a mesh's geometry at the dofs MFEM expects (the two
/// directions differ only in which side supplies the values).
fn hex_slot_map<M: MeshTopology>(
    mesh: &M,
    p: usize,
) -> Result<(Vec<NodeId>, usize), HexSlotErr> {
    let e = p - 1; // dofs per edge (and per face row/column)
    let n_elems = mesh.n_elements();
    let n_vert = mesh.n_nodes();
    if n_elems == 0 || n_vert == 0 {
        return Err(HexSlotErr::NotHex);
    }
    let mut n_hex = 0usize;
    for el in 0..n_elems as u32 {
        if mesh.element_nodes(el).len() == 8 {
            n_hex += 1;
        }
    }
    if n_hex == 0 {
        return Err(HexSlotErr::NotHex); // pure simplex/other mesh: not our business
    }
    if n_hex != n_elems {
        // A mixed mesh containing hexahedra: the `nodes` dof blocks are sized
        // per element geometry, so neither this mapper nor the uniform
        // DofManager fallback can describe it.
        return HexSlotErr::unsupported("mixed-element mesh containing hexahedra");
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

    // Local-edge lookup: (varying axis, side of the two fixed axes) -> edge id.
    let mut edge_lut: HashMap<(usize, usize, usize), usize> = HashMap::new();
    for (k, &[la, lb]) in HEX_EDGES.iter().enumerate() {
        let (ca, cb) = (HEX_CORNERS[la], HEX_CORNERS[lb]);
        let av = match (0..3).find(|&d| ca[d] != cb[d]) {
            Some(d) => d,
            None => return HexSlotErr::unsupported("degenerate HEX_EDGES table"),
        };
        let bnd: Vec<usize> = (0..3).filter(|&d| d != av).collect();
        edge_lut.insert((av, ca[bnd[0]], ca[bnd[1]]), k);
    }

    let ref_elem = fem_element::lagrange::factory::HexQk::new(p);
    let ref_coords = ref_elem.dof_coords();
    let npe = ref_coords.len();
    if npe != 8 + 12 * e + 6 * e * e + e * e * e {
        return HexSlotErr::unsupported("reference hex element is not the H1 order-p tensor basis");
    }
    // The 1-D GLL nodes of the reference basis: the tensor index of a slot is
    // the position of its coordinate in this table (`HexQk` builds its 1-D
    // basis from the same function, so the values match bit-for-bit).
    let gll = fem_element::quadrature::gauss_lobatto_arbitrary(p + 1).0;
    if gll.len() != p + 1 {
        return HexSlotErr::unsupported("unexpected Gauss-Lobatto node count");
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
                        return HexSlotErr::unsupported("reference slot is not on the GLL tensor grid")
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
                        None => return HexSlotErr::unsupported("bad vertex slot"),
                    }
                }
                // Edge slot: shared, canonical direction = ascending vertex id.
                2 => {
                    let bnd: Vec<usize> = (0..3).filter(|&d| on_bnd[d]).collect();
                    let av = match (0..3).find(|&d| !on_bnd[d]) {
                        Some(d) => d,
                        None => return HexSlotErr::unsupported("bad edge slot"),
                    };
                    let key = (av, idx[bnd[0]] / p, idx[bnd[1]] / p);
                    let k = match edge_lut.get(&key) {
                        Some(&k) => k,
                        None => return HexSlotErr::unsupported("unmatched edge slot"),
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
                        None => return HexSlotErr::unsupported("unmatched mesh edge"),
                    };
                    let t = if a < b { t_local } else { e - 1 - t_local };
                    edge_base + ei * e + t
                }
                // Face slot: shared, canonical parameterisation = the first
                // element's `FaceVert` order.
                1 => {
                    let ax = match (0..3).find(|&d| on_bnd[d]) {
                        Some(d) => d,
                        None => return HexSlotErr::unsupported("bad face slot"),
                    };
                    let side = idx[ax] / p;
                    let f = match (0..6).find(|&f| HEX_FACES[f].iter().all(|&v| HEX_CORNERS[v][ax] == side)) {
                        Some(f) => f,
                        None => return HexSlotErr::unsupported("unmatched local face"),
                    };
                    let [l0, l1, _, l3] = HEX_FACES[f];
                    // In-face indices (from the local vertex `l0`).
                    let mut in_face = [0usize; 2];
                    for (s, &lk) in [l1, l3].iter().enumerate() {
                        let d = match (0..3).find(|&d| HEX_CORNERS[l0][d] != HEX_CORNERS[lk][d]) {
                            Some(d) => d,
                            None => return HexSlotErr::unsupported("degenerate face slot"),
                        };
                        if d == ax {
                            return HexSlotErr::unsupported("degenerate face slot");
                        }
                        in_face[s] = if HEX_CORNERS[l0][d] == 0 { idx[d] } else { p - idx[d] };
                        if in_face[s] == 0 || in_face[s] >= p {
                            return HexSlotErr::unsupported("face slot on a face edge");
                        }
                    }
                    let mut fkey = [0u32; 4];
                    for (k, &lv) in HEX_FACES[f].iter().enumerate() {
                        fkey[k] = n8[lv];
                    }
                    fkey.sort_unstable();
                    let fi = match face_ids.get(&fkey) {
                        Some(&fi) => fi as usize,
                        None => return HexSlotErr::unsupported("unmatched mesh face"),
                    };
                    // Map the local face parameterisation onto the canonical
                    // one (corner matching: at most a rotation/reflection).
                    let cv = face_verts[fi];
                    let corner_of = |v: u32| (0..4).find(|&i| cv[i] == v);
                    let p0 = match corner_of(n8[l0]) {
                        Some(i) => i,
                        None => return HexSlotErr::unsupported("face corner mismatch"),
                    };
                    let pu = match corner_of(n8[l1]) {
                        Some(i) => i,
                        None => return HexSlotErr::unsupported("face corner mismatch"),
                    };
                    let pv = match corner_of(n8[l3]) {
                        Some(i) => i,
                        None => return HexSlotErr::unsupported("face corner mismatch"),
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
                        return HexSlotErr::unsupported("face slot outside the canonical face");
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
            return HexSlotErr::unsupported("unexpected interior slot count");
        }
    }

    Ok((conn, n_dofs))
}

// ─── D151: MFEM's H1 `nodes` numbering for 2-D quadrilateral meshes ─────────

/// MFEM `Constants<Geometry::SQUARE>::Edges` (`fem/geom.cpp:960`): local edge
/// `k` runs from local vertex `EDGES[k][0]` to `EDGES[k][1]`.
const QUAD_EDGES: [[usize; 2]; 4] = [[0, 1], [1, 2], [2, 3], [3, 0]];
/// The four corners of the reference square in `QuadQk`'s (and MFEM's
/// `Geometry::SQUARE`) vertex order.
const QUAD_VERT_CORNERS: [[usize; 2]; 4] = [[0, 0], [1, 0], [1, 1], [0, 1]];

/// D151: MFEM's H1 `nodes` numbering for an all-Quad4 **2-D** mesh, as the map
/// `conn[e * npe + s] = <file dof of reference slot s of element e>` plus the
/// total number of geometry dofs — the 2-D analogue of [`hex_slot_map`], with
/// the same slot (that is, `QuadQk`) order.
///
/// MFEM's numbering (`FiniteElementSpace::GetElementDofs`,
/// `fem/fespace.cpp`; entity enumeration from `Mesh::FinalizeTopology`):
///
/// ```text
///   vertex v            -> dof v
///   mesh edge E, slot t -> dof NV + E*(p-1) + t
///   element e, slot o   -> dof NV + NE*(p-1) + e*(p-1)^2 + o
/// ```
///
/// where the interior slot `o = (i-1) + (j-1)*(p-1)` follows
/// `H1_QuadrilateralElement`'s `dof_map` (`i` fastest) and an edge's slot `t`
/// counts from the edge end with the **smaller mesh vertex id**
/// (`SegDofOrd[0]` is the identity and `Mesh::GetElementEdges` orders the edge's
/// vertices by ascending id).
fn quad2d_slot_map<M: MeshTopology>(
    mesh: &M,
    p: usize,
) -> Result<(Vec<NodeId>, usize), HexSlotErr> {
    let e = p - 1; // dofs per edge, and per interior row/column
    let n_elems = mesh.n_elements();
    let n_vert = mesh.n_nodes();
    if n_elems == 0 || n_vert == 0 {
        return Err(HexSlotErr::NotHex);
    }
    let mut n_quad = 0usize;
    for el in 0..n_elems as u32 {
        if mesh.element_nodes(el).len() == 4 {
            n_quad += 1;
        }
    }
    if n_quad == 0 {
        return Err(HexSlotErr::NotHex);
    }
    if n_quad != n_elems {
        return HexSlotErr::unsupported("mixed-element mesh containing quadrilaterals");
    }

    // Mesh edges in MFEM's enumeration (element traversal, local edge order,
    // first encounter wins).
    let mut elems: Vec<[u32; 4]> = Vec::with_capacity(n_elems);
    let mut edge_ids: HashMap<[u32; 2], u32> = HashMap::new();
    for el in 0..n_elems as u32 {
        let ns = mesh.element_nodes(el);
        let mut n4 = [0u32; 4];
        n4.copy_from_slice(ns);
        for &[la, lb] in QUAD_EDGES.iter() {
            let (a, b) = (n4[la], n4[lb]);
            let key = if a < b { [a, b] } else { [b, a] };
            let next = edge_ids.len() as u32;
            edge_ids.entry(key).or_insert(next);
        }
        elems.push(n4);
    }
    let n_edges = edge_ids.len();
    let n_dofs = n_vert + n_edges * e + n_elems * e * e;

    let ref_elem = fem_element::lagrange::factory::QuadQk::new(p);
    let ref_coords = ref_elem.dof_coords();
    let npe = ref_coords.len();
    if npe != 4 + 4 * e + e * e {
        return HexSlotErr::unsupported("reference quad element is not the H1 order-p tensor basis");
    }
    // `QuadQk`'s 1-D basis lives on `[0,1]` (`Lagrange1D` mapped by
    // `0.5*(x+1)`), so the tensor index of a slot is the position of its
    // coordinate in the `[0,1]` Gauss-Lobatto table.
    let gll: Vec<f64> = fem_element::quadrature::gauss_lobatto_arbitrary(p + 1)
        .0
        .iter()
        .map(|&x| 0.5 * (x + 1.0))
        .collect();
    if gll.len() != p + 1 {
        return HexSlotErr::unsupported("unexpected Gauss-Lobatto node count");
    }

    let edge_base = n_vert;
    let interior_base = edge_base + n_edges * e;
    let mut conn: Vec<NodeId> = Vec::with_capacity(n_elems * npe);
    for (el, n4) in elems.iter().enumerate() {
        let mut interior_seen = 0usize;
        for c in ref_coords.iter() {
            let mut idx = [0usize; 2];
            for d in 0..2 {
                let k = match gll.iter().position(|&x| (x - c[d]).abs() < 1e-12) {
                    Some(k) => k,
                    None => {
                        return HexSlotErr::unsupported("reference slot is not on the GLL tensor grid")
                    }
                };
                idx[d] = k;
            }
            let on_bnd = [idx[0] == 0 || idx[0] == p, idx[1] == 0 || idx[1] == p];
            let nb = on_bnd.iter().filter(|&&b| b).count();
            let g: usize = match nb {
                // Vertex slot: the file stores vertex `v` as dof `v`.
                2 => {
                    let side = [idx[0] / p, idx[1] / p];
                    match (0..4).find(|&k| QUAD_VERT_CORNERS[k] == side) {
                        Some(lv) => n4[lv] as usize,
                        None => return HexSlotErr::unsupported("bad vertex slot"),
                    }
                }
                // Edge slot: shared; its slot counts from the edge end with the
                // smaller mesh vertex id.
                1 => {
                    // The varying axis (1 for the x = 0 / x = 1 edges, 0 for the
                    // y = 0 / y = 1 ones) and the local edge whose side it is.
                    let (av, k) = if on_bnd[0] {
                        (1usize, if idx[0] == 0 { 3 } else { 1 })
                    } else {
                        (0usize, if idx[1] == 0 { 0 } else { 2 })
                    };
                    if idx[av] == 0 || idx[av] == p {
                        return HexSlotErr::unsupported("edge slot at a corner");
                    }
                    let [la, lb] = QUAD_EDGES[k];
                    // `t_local` counts from `la` (the edge's first vertex).
                    let t_local = if QUAD_VERT_CORNERS[la][av] == 0 {
                        idx[av] - 1
                    } else {
                        p - 1 - idx[av]
                    };
                    let (a, b) = (n4[la], n4[lb]);
                    let ekey = if a < b { [a, b] } else { [b, a] };
                    let ei = match edge_ids.get(&ekey) {
                        Some(&ei) => ei as usize,
                        None => return HexSlotErr::unsupported("unmatched mesh edge"),
                    };
                    let t = if a < b { t_local } else { e - 1 - t_local };
                    edge_base + ei * e + t
                }
                // Interior slot: private to the element, `(i-1) + (j-1)*(p-1)`.
                _ => {
                    let o = (idx[0] - 1) + (idx[1] - 1) * e;
                    debug_assert_eq!(o, interior_seen);
                    interior_seen += 1;
                    interior_base + el * e * e + o
                }
            };
            conn.push(g as NodeId);
        }
        if interior_seen != e * e {
            return HexSlotErr::unsupported("unexpected interior slot count");
        }
    }

    Ok((conn, n_dofs))
}

/// D178: MFEM's H1 `nodes` numbering for an all-`Tri3` **2-D** mesh, as the map
/// `conn[e * npe + s] = <file dof of reference slot s of element e>` plus the
/// total number of geometry dofs — the triangle analogue of [`quad2d_slot_map`],
/// with the same slot (that is, `H1TriPk`) order.
///
/// MFEM's numbering (`FiniteElementSpace::GetElementDofs`, `fem/fespace.cpp`;
/// entity enumeration from `Mesh::FinalizeTopology`; node positions from
/// `H1_TriangleElement`, `fem/fe/fe_h1.cpp:451`; verified against MFEM 4.10's
/// own `H1_2D_P2/P3/P4` output for `MakeCartesian2D` triangle grids, probe
/// `tmp/r36/tri2d_nodes.cpp`):
///
/// ```text
///   vertex v            -> dof v
///   mesh edge E, slot t -> dof NV + E*(p-1) + t
///   element e, slot o   -> dof NV + NE*(p-1) + e*(p-1)(p-2)/2 + o
/// ```
///
/// The local edge order is `(0,1) (1,2) (2,0)` and an edge's slot `t` counts
/// from the edge end with the **smaller mesh vertex id** — the same rule the
/// quadrilateral map pins down (`SegDofOrd[0]` is the identity and the edge
/// table stores `(min, max)`; measured on the p = 3 fixture, where e.g. the
/// top-edge dofs count from the *left* end because the left vertex id is
/// smaller).  The interior slot `o` follows `H1_TriangleElement`'s interior
/// enumeration (`j`-outer, `i`-inner over the `w`-normalised barycentric
/// Gauss-Lobatto points), which is exactly `H1TriPk`'s interior block order,
/// so the interior slots are simply counted per element.
fn tri2d_slot_map<M: MeshTopology>(mesh: &M, p: usize) -> Result<(Vec<NodeId>, usize), HexSlotErr> {
    /// MFEM `Geometry::Constants<Geometry::TRIANGLE>::Edges`: local edge `k`
    /// runs from vertex `TRI_EDGES[k][0]` to vertex `TRI_EDGES[k][1]`.
    const TRI_EDGES: [[usize; 2]; 3] = [[0, 1], [1, 2], [2, 0]];
    let e = p - 1; // dofs per edge
    let n_int = if p >= 3 { (p - 1) * (p - 2) / 2 } else { 0 };
    let n_elems = mesh.n_elements();
    let n_vert = mesh.n_nodes();
    if p < 2 || n_elems == 0 || n_vert == 0 {
        return Err(HexSlotErr::NotHex); // order 1 needs no numbering
    }
    let mut n_tri = 0usize;
    for el in 0..n_elems as u32 {
        if mesh.element_nodes(el).len() == 3 {
            n_tri += 1;
        }
    }
    if n_tri == 0 {
        return Err(HexSlotErr::NotHex);
    }
    if n_tri != n_elems {
        return HexSlotErr::unsupported("mixed-element mesh containing triangles");
    }

    // Mesh edges in MFEM's enumeration (element traversal, local edge order,
    // first encounter wins).
    let mut elems: Vec<[u32; 3]> = Vec::with_capacity(n_elems);
    let mut edge_ids: HashMap<[u32; 2], u32> = HashMap::new();
    for el in 0..n_elems as u32 {
        let ns = mesh.element_nodes(el);
        let mut n3 = [0u32; 3];
        n3.copy_from_slice(ns);
        for &[la, lb] in TRI_EDGES.iter() {
            let (a, b) = (n3[la], n3[lb]);
            let key = if a < b { [a, b] } else { [b, a] };
            let next = edge_ids.len() as u32;
            edge_ids.entry(key).or_insert(next);
        }
        elems.push(n3);
    }
    let n_edges = edge_ids.len();
    let n_dofs = n_vert + n_edges * e + n_elems * n_int;

    // `H1TriPk`'s slots follow MFEM `H1_TriangleElement`'s own dof order
    // (vertices -> edges (0→1, 1→2, 2→0) -> interior) on the *same* closed
    // Gauss-Lobatto lattice (probe: MFEM's `GetNodes()` and `H1TriPk`'s
    // `dof_coords()` agree point-for-point for p = 2..4), so the slot's
    // reference coordinates identify its entity and its along-edge position.
    let ref_coords = fem_element::lagrange::H1TriPk::new(p).dof_coords();
    let npe = ref_coords.len();
    if npe != 3 + 3 * e + n_int {
        return HexSlotErr::unsupported("reference tri element is not the H1 order-p basis");
    }
    // `H1TriPk`'s 1-D points live on `[0,1]` (`Lagrange1D` mapped by `0.5*(x+1)`),
    // the same closed Gauss-Lobatto table MFEM's `Poly_1D::ClosedPoints` uses.
    let gll: Vec<f64> = fem_element::quadrature::gauss_lobatto_arbitrary(p + 1)
        .0
        .iter()
        .map(|&x| 0.5 * (x + 1.0))
        .collect();
    if gll.len() != p + 1 {
        return HexSlotErr::unsupported("unexpected Gauss-Lobatto node count");
    }
    let tol = 1e-12;
    let gll_index = |x: f64| -> Option<usize> { gll.iter().position(|&g| (g - x).abs() < tol) };

    let edge_base = n_vert;
    let interior_base = edge_base + n_edges * e;
    // `along` is the edge parameter in [0,1] counted from the local first
    // vertex; its Gauss-Lobatto index minus one is the local edge slot.  The
    // global slot counts from the edge end with the smaller mesh vertex id.
    let edge_dof = |n3: &[u32; 3], la: usize, lb: usize, along: f64| -> Result<usize, ()> {
        let k = gll_index(along).ok_or(())?;
        let t_local = k - 1;
        let (a, b) = (n3[la], n3[lb]);
        let key = if a < b { [a, b] } else { [b, a] };
        let ei = *edge_ids.get(&key).ok_or(())?;
        let t = if a < b { t_local } else { e - 1 - t_local };
        Ok(edge_base + ei as usize * e + t)
    };
    let bad_slot = || {
        HexSlotErr::Unsupported("reference slot is not on the Gauss-Lobatto lattice")
    };
    let mut conn: Vec<NodeId> = Vec::with_capacity(n_elems * npe);
    for (el, n3) in elems.iter().enumerate() {
        let mut interior_seen = 0usize;
        for c in ref_coords.iter() {
            let (x, y) = (c[0], c[1]);
            let at_lo = |v: f64| v.abs() < tol;
            let at_hi = |v: f64| (v - 1.0).abs() < tol;
            let g: usize = if at_lo(x) && at_lo(y) {
                // Vertex slot: the file stores vertex `v` as dof `v`.
                n3[0] as usize
            } else if at_hi(x) && at_lo(y) {
                n3[1] as usize
            } else if at_lo(x) && at_hi(y) {
                n3[2] as usize
            } else if at_lo(y) {
                // Edge 0 (v0→v1): the parameter counts from v0.
                edge_dof(n3, 0, 1, x).map_err(|()| bad_slot())?
            } else if at_hi(x + y) {
                // Edge 1 (v1→v2): `x + y == 1`, the parameter counts from v1.
                edge_dof(n3, 1, 2, y).map_err(|()| bad_slot())?
            } else if at_lo(x) {
                // Edge 2 (v2→v0): the parameter counts from v2, i.e. `1 - y`.
                edge_dof(n3, 2, 0, 1.0 - y).map_err(|()| bad_slot())?
            } else {
                // Interior slot: private to the element.  `H1TriPk` enumerates
                // its interior block in `H1_TriangleElement`'s own
                // `j`-outer/`i`-inner order, so the running count *is* the
                // interior slot `o`.
                let o = interior_seen;
                interior_seen += 1;
                interior_base + el * n_int + o
            };
            conn.push(g as NodeId);
        }
        if interior_seen != n_int {
            return HexSlotErr::unsupported("unexpected interior slot count");
        }
    }

    Ok((conn, n_dofs))
}

// ─── D151: MFEM's H1 `nodes` numbering for prism (wedge) meshes ──────────────
//
// MFEM's `H1_WedgeElement` (`fem/fe/fe_h1.cpp:863`) is `H1_TriangleElement ×
// H1_SegmentElement`, i.e. its nodes sit on the **closed Gauss-Lobatto**
// points of both factors.  fem-rs's prism geometry element `PrismPk`
// (`crates/element/src/lagrange/prism.rs`) used to be a tensor product of the
// *equispaced* triangle and segment bases — the same *family split* this
// project had already hit for tetrahedra (D49/D112/D152) and triangles (D138),
// measured as `|fem-rs − C++| = 7.2e-5` on the default `toroid -o 3` wedge.
// **Round 34 (D164) moved `PrismPk` to the Gauss-Lobatto lattice** (GLL
// triangle factor ⊗ GLL segment factor, layer-major slot order kept), so the
// two lattices now coincide for every `p` — historically they only agreed at
// `p ≤ 2` and diverged from `p = 3` (`0.276393202250021` /
// `0.723606797749979` vs `1/3` / `2/3`).
//
// The file's `nodes` section is read by MFEM as `H1_WedgeElement` nodal values,
// so dof `g` must carry the *physical position of the mesh's own geometry map*
// at the GLL point `ξ_g` of that dof.  The writer still goes through the
// interpolation matrix
//
//     B[i][s] = φ_s^{PrismPk}(ξ_i),      ξ_i = the GLL point of H1 dof `i`
//
// built once per order — which is now the identity map on GLL nodes (it was a
// genuine equispaced→GLL re-interpolation before round 34) — and evaluates
// `x_e(ξ_i) = Σ_s B[i][s]·X_s` per element.  The
// result is *the mesh's own geometry function*, sampled at MFEM's nodes: the
// `nodes` section then describes the same curved mesh fem-rs assembles with,
// and the read-back file MFEM produces is a fixed point of this writer.
//
// (For a *straight-sided* mesh the two lattices describe the same polynomial —
// the affine prism map — so the written nodes match MFEM's own file
// bit-for-bit; `crates/io/tests/prism_nodes_writer.rs` pins that against a
// mesh file produced by MFEM 4.10's `SetCurvature`.)

/// MFEM `Geometry::Constants<Geometry::PRISM>::FaceVert` (`fem/geom.cpp:1061`):
/// local faces 0/1 are the bottom/top triangles, 2..4 the quadrilateral sides.
/// Every quadrilateral entry is a valid square parameterisation
/// (`FaceVert[f][2] = FaceVert[f][1] + FaceVert[f][3] - FaceVert[f][0]` in
/// reference coordinates), so `QUAD_CORNERS` describes the stored face too.
const PRISM_FACES: [[usize; 4]; 5] = [
    [0, 2, 1, usize::MAX], [3, 4, 5, usize::MAX], [0, 1, 4, 3], [1, 2, 5, 4],
    [2, 0, 3, 5],
];

/// One local dof of `H1_WedgeElement`: its reference point in `PrismPk`'s
/// convention (the points `PrismPk::eval_basis` is evaluated at) and the
/// entity it belongs to.
///
/// The entity is the element crate's [`H1PrismSlot`] — the single source of
/// truth for MFEM's `H1_WedgeElement` node table (D168/D191 ground truth,
/// `fem/fe/fe_h1.cpp:863`).  D315 removed this file's second copy of that
/// table (`PrismSlotEntity` + a local `t_dof`/`s_dof` reconstruction): the
/// writers now consume `H1PrismPk`'s slot labels *and* node positions, so the
/// io tables can no longer drift from the ones the solvers number spaces with.
struct PrismSlot {
    xi: [f64; 3],
    entity: H1PrismSlot,
}

/// MFEM's `H1_WedgeElement` node table (`fem/fe/fe_h1.cpp:863`) as
/// [`H1PrismPk`]: entity label per slot plus the slot's node position, in slot
/// order.
///
/// The constructor does not build the node positions from the entity structure
/// directly: it keeps two index tables `t_dof`/`s_dof` into the nodes of
/// `H1_TriangleElement` and `H1_SegmentElement` and takes
/// `(t_Nodes[t_dof[i]].x, .y, s_Nodes[s_dof[i]].x)` as dof `i`'s position.  The
/// element crate reproduces those tables verbatim (including the triangular-
/// face index arithmetic, which is *not* the plain running index) and pins
/// them against MFEM (`crates/element/tests`); this function only pairs them
/// with the node positions [`H1PrismPk::dof_coords`] reports, so the io and
/// element sides cannot silently disagree.
fn prism_h1_slots(p: usize) -> Result<Vec<PrismSlot>, &'static str> {
    if p < 1 {
        return Err("order < 1");
    }
    let labels = fem_element::lagrange::H1PrismPk::slot_labels(p);
    let coords = fem_element::lagrange::H1PrismPk::new(p).dof_coords();
    if labels.len() != coords.len() {
        return Err("the H1 wedge slot labels and node positions disagree");
    }
    Ok(labels
        .into_iter()
        .zip(coords)
        .map(|(entity, c)| PrismSlot { xi: [c[0], c[1], c[2]], entity })
        .collect())
}


/// The canonical in-face Gauss-Lobatto index `(u, v)` (1-based, as they index
/// `QuadDofOrd`) of the slot at local in-face indices `(a, b)` (1-based from
/// the local face's vertex `l0` towards `l1` and `l3`), measured in the
/// **stored** face's parameterisation — the first encountering element's
/// `FaceVert` order (`Mesh::GenerateFaces` →
/// `AddQuadFaceElement(j, ef[j], i, v[fv[0]], …)`).
fn quad_face_canonical_slot(
    stored: &[u32; 4],
    l0: u32,
    l1: u32,
    l3: u32,
    in_face: [usize; 2],
    p: usize,
) -> Option<usize> {
    let e = p - 1;
    let corner_of = |v: u32| (0..4).find(|&i| stored[i] == v);
    let p0 = corner_of(l0)?;
    let pu = corner_of(l1)?;
    let pv = corner_of(l3)?;
    let du = [
        QUAD_CORNERS[pu][0] - QUAD_CORNERS[p0][0],
        QUAD_CORNERS[pu][1] - QUAD_CORNERS[p0][1],
    ];
    let dv = [
        QUAD_CORNERS[pv][0] - QUAD_CORNERS[p0][0],
        QUAD_CORNERS[pv][1] - QUAD_CORNERS[p0][1],
    ];
    let u = QUAD_CORNERS[p0][0] * p as i32 + in_face[0] as i32 * du[0] + in_face[1] as i32 * dv[0];
    let v = QUAD_CORNERS[p0][1] * p as i32 + in_face[0] as i32 * du[1] + in_face[1] as i32 * dv[1];
    if u <= 0 || u >= p as i32 || v <= 0 || v >= p as i32 {
        return None;
    }
    Some((u - 1) as usize + (v - 1) as usize * e)
}

/// D151: the `nodes` dof values of an all-`Prism6` mesh, in MFEM's H1 wedge
/// numbering.  Returns `(n_dofs, values)` with `values[d * sdim + c]`.
///
/// Unlike the hexahedral/tetrahedral tables (which permute the mesh's geometry
/// slots into MFEM's numbering), the prism table has to *re-evaluate* the
/// mesh's geometry: `PrismPk`'s nodes and the GLL nodes of `H1_WedgeElement`
/// are different points (see the module note above), so a value read at one
/// lattice cannot be handed over unchanged.
fn prism_nodes_dof_values<M: MeshTopology>(
    mesh: &M,
    geo: &GeometryData,
    p: usize,
    sdim: usize,
) -> FemResult<(usize, Vec<f64>)> {
    let slots = prism_h1_slots(p)
        .map_err(|why| FemError::Mesh(format!("write_mfem: bad prism `nodes` table: {why}")))?;
    let npe = slots.len();
    let n_elems = mesh.n_elements();
    let n_vert = mesh.n_nodes();
    let e_per_edge = p - 1;
    let nf_tri = if p >= 3 { (p - 1) * (p - 2) / 2 } else { 0 };
    let nf_quad = e_per_edge * e_per_edge;
    let nb = nf_tri * e_per_edge;

    // MFEM's mesh tables: edges and faces are numbered by element traversal,
    // then the element's local entity order, first encounter wins
    // (`Mesh::FinalizeTopology`), and the stored face parameterisation is the
    // first encountering element's `FaceVert` order.
    let mut elems: Vec<[u32; 6]> = Vec::with_capacity(n_elems);
    let mut edge_ids: HashMap<[u32; 2], usize> = HashMap::new();
    let mut face_ids: HashMap<Vec<u32>, usize> = HashMap::new();
    let mut face_verts: Vec<[u32; 4]> = Vec::new();
    let mut face_dofs: Vec<usize> = Vec::new();
    for el in 0..n_elems as u32 {
        let ns = mesh.element_nodes(el);
        if ns.len() != 6 {
            return Err(FemError::Mesh(
                "write_mfem: the prism `nodes` table needs a uniform all-Prism6 mesh".into(),
            ));
        }
        let n6 = [ns[0], ns[1], ns[2], ns[3], ns[4], ns[5]];
        for &[la, lb] in PRISM_EDGES.iter() {
            let (a, b) = (n6[la], n6[lb]);
            let key = if a < b { [a, b] } else { [b, a] };
            let next = edge_ids.len();
            edge_ids.entry(key).or_insert(next);
        }
        for (f, fv) in PRISM_FACES.iter().enumerate() {
            let mut key: Vec<u32> = fv
                .iter()
                .filter(|&&v| v != usize::MAX)
                .map(|&v| n6[v])
                .collect();
            key.sort_unstable();
            let next = face_ids.len();
            face_ids.entry(key).or_insert_with(|| {
                let mut verts = [u32::MAX; 4];
                for (i, &v) in fv.iter().enumerate() {
                    if v != usize::MAX {
                        verts[i] = n6[v];
                    }
                }
                face_verts.push(verts);
                face_dofs.push(if f < 2 { nf_tri } else { nf_quad });
                next
            });
        }
        elems.push(n6);
    }
    let n_edges = edge_ids.len();
    let mut face_base: Vec<usize> = Vec::with_capacity(face_dofs.len());
    let mut acc = 0usize;
    for &d in &face_dofs {
        face_base.push(acc);
        acc += d;
    }
    let n_face_dofs = acc;
    let n_dofs = n_vert + n_edges * e_per_edge + n_face_dofs + n_elems * nb;

    // `B[i][s] = φ_s^{PrismPk}(ξ_i)`: the mesh's own geometry element
    // (Gauss-Lobatto lattice since D164, round 34) evaluated at the GLL points
    // of MFEM's H1 wedge — the identity on nodes, kept as a matrix so the
    // pairing stays explicit.
    if geo.nodes_per_elem != npe || geo.conn.len() != n_elems * npe {
        return Err(FemError::Mesh(format!(
            "write_mfem: the prism geometry table has {} nodes per element, the H1 \
             wedge needs {npe}",
            geo.nodes_per_elem
        )));
    }
    let prism = fem_element::lagrange::PrismPk::new(p);
    if prism.n_dofs() != npe {
        return Err(FemError::Mesh(
            "write_mfem: `PrismPk` and the H1 wedge do not describe the same order".into(),
        ));
    }
    let mut basis = vec![0.0f64; npe];
    let mut interp = vec![0.0f64; npe * npe];
    for (i, slot) in slots.iter().enumerate() {
        prism.eval_basis(&slot.xi, &mut basis);
        interp[i * npe..(i + 1) * npe].copy_from_slice(&basis);
    }

    let mut values = vec![0.0f64; n_dofs * sdim];
    let mut filled = vec![false; n_dofs];
    for (el, n6) in elems.iter().enumerate() {
        let base = n_vert + n_edges * e_per_edge + n_face_dofs + el * nb;
        for (i, slot) in slots.iter().enumerate() {
            let g: usize = match slot.entity {
                H1PrismSlot::Vertex(v) => n6[v] as usize,
                H1PrismSlot::Edge(k, j) => {
                    let [la, lb] = PRISM_EDGES[k];
                    let (a, b) = (n6[la], n6[lb]);
                    let key = if a < b { [a, b] } else { [b, a] };
                    let ei = match edge_ids.get(&key) {
                        Some(&ei) => ei,
                        None => {
                            return Err(FemError::Mesh(
                                "write_mfem: unmatched prism edge".into(),
                            ))
                        }
                    };
                    // `Mesh::GetElementEdges` orients the edge by ascending
                    // vertex id and `H1_FECollection`'s `SegDofOrd[1][j] =
                    // p-2-j`, so a reversed edge reverses the slot positions.
                    let t = if a < b { j } else { e_per_edge - 1 - j };
                    n_vert + ei * e_per_edge + t
                }
                H1PrismSlot::TriFace(f, k) => {
                    let fv = PRISM_FACES[f];
                    let test = [n6[fv[0]], n6[fv[1]], n6[fv[2]]];
                    let mut key = test.to_vec();
                    key.sort_unstable();
                    let fi = match face_ids.get(&key) {
                        Some(&fi) => fi,
                        None => {
                            return Err(FemError::Mesh(
                                "write_mfem: unmatched prism triangular face".into(),
                            ))
                        }
                    };
                    let stored = [
                        face_verts[fi][0],
                        face_verts[fi][1],
                        face_verts[fi][2],
                    ];
                    let orient = match tri_orientation(&stored, &test) {
                        Some(o) => o,
                        None => {
                            return Err(FemError::Mesh(
                                "write_mfem: prism face corner mismatch".into(),
                            ))
                        }
                    };
                    let off = match tri_dof_ord(p, orient, k) {
                        Some(o) => o,
                        None => {
                            return Err(FemError::Mesh(
                                "write_mfem: unmatched prism face dof ordering".into(),
                            ))
                        }
                    };
                    n_vert + n_edges * e_per_edge + face_base[fi] + off
                }
                H1PrismSlot::QuadFace(f, a, b) => {
                    let fv = PRISM_FACES[f];
                    let mut key = vec![n6[fv[0]], n6[fv[1]], n6[fv[2]], n6[fv[3]]];
                    key.sort_unstable();
                    let fi = match face_ids.get(&key) {
                        Some(&fi) => fi,
                        None => {
                            return Err(FemError::Mesh(
                                "write_mfem: unmatched prism quadrilateral face".into(),
                            ))
                        }
                    };
                    let off = match quad_face_canonical_slot(
                        &face_verts[fi],
                        n6[fv[0]],
                        n6[fv[1]],
                        n6[fv[3]],
                        [a, b],
                        p,
                    ) {
                        Some(o) => o,
                        None => {
                            return Err(FemError::Mesh(
                                "write_mfem: prism quad face slot outside the stored face".into(),
                            ))
                        }
                    };
                    n_vert + n_edges * e_per_edge + face_base[fi] + off
                }
                H1PrismSlot::Interior(m) => base + m,
            };
            if g >= n_dofs {
                return Err(FemError::Mesh(
                    "write_mfem: prism `nodes` dof index out of range".into(),
                ));
            }
            // The value at MFEM's GLL node: the mesh's own geometry map there.
            let mut v = [0.0f64; 3];
            for s in 0..npe {
                let w = interp[i * npe + s];
                if w != 0.0 {
                    let c = geo.conn[el * npe + s] as usize;
                    for d in 0..sdim {
                        v[d] += w * geo.coords[c * sdim + d];
                    }
                }
            }
            if filled[g] {
                if (0..sdim).any(|d| !approx_eq(values[g * sdim + d], v[d])) {
                    return Err(FemError::Mesh(format!(
                        "write_mfem: prism geometry dof {g} is shared by two elements with \
                         different coordinates — the mesh geometry is not continuous"
                    )));
                }
            } else {
                values[g * sdim..g * sdim + sdim].copy_from_slice(&v[..sdim]);
                filled[g] = true;
            }
        }
    }
    if let Some(g) = filled.iter().position(|f| !*f) {
        return Err(FemError::Mesh(format!(
            "write_mfem: no element claims the prism `nodes` dof {g}"
        )));
    }
    Ok((n_dofs, values))
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
    let (conn, n_dofs) = match hex_slot_map(mesh, order as usize) {
        Ok(v) => v,
        Err(HexSlotErr::NotHex) => return HexGeom::NotHex,
        Err(HexSlotErr::Unsupported(why)) => return HexGeom::Unsupported(why),
    };
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
    let npe = conn.len() / mesh.n_elements();
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
/// The reference element is the one the assembler uses for tet geometry —
/// `fem_element::lagrange::factory::H1TetPk`, MFEM's `H1_TetrahedronElement`
/// (closed Gauss-Lobatto nodes, MFEM's DOF order; D49) — and its slots carry
/// their *integer barycentric coordinates* (`H1TetPk::slot_labels`).
///
/// **Keep the two in step:** `geo_ref_elem` (`crates/assembly/src/assembler.rs`)
/// builds the matching geometry element for the transform, and
/// `vector_assembler::geo_ref_elem_from_mesh` returns the same families for the
/// vector-assembly path — tet → `H1TetPk` (GLL, D49), prism → `PrismPk`
/// (Gauss-Lobatto since round 34's D164) — so a curved mesh is transformed with
/// the same slot order this table was built in.
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
/// `fem_element::lagrange::factory::H1TetPk`'s slot order (MFEM's
/// `H1_TetrahedronElement` DOF order), the slot's physical key (see
/// [`TetFileSlots`]), and the total number of geometry dofs.  `Err` carries the
/// reason the mesh cannot be mapped faithfully.
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

    // The reference element the assembler uses for tet geometry: MFEM's
    // `H1_TetrahedronElement(p)` (closed Gauss-Lobatto nodes, D49).  Its slots
    // carry their integer barycentric coordinates directly (see
    // `h1_tet_slot_labels`), which is what MFEM keys a geometric DOF by, so a
    // slot is classified exactly instead of by rounding a node coordinate.
    let labels = fem_element::lagrange::factory::H1TetPk::slot_labels(p);
    let npe = labels.len();
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
        for idx in labels.iter() {
            let idx = *idx;
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

// ─── D117/D118: mixed-mesh H1 `nodes` geometry ───────────────────────────────
//
// MFEM's `nodes` grid function numbers its dofs over the whole mesh regardless
// of element mixture (`FiniteElementSpace::GetElementDofs`, `fem/fespace.cpp`):
// vertices, then per-edge blocks of `p-1` dofs, then per-face blocks sized by
// the *face* geometry (tri `(p-1)(p-2)/2`, quad `(p-1)²` — MFEM's
// `var_face_dofs` table for mixed faces, `fespace.cpp:2855`), then
// per-element interior blocks sized by the *element* geometry.  This module
// reproduces that numbering for tri/quad (2-D) and tet/hex/prism (3-D)
// mixtures at any order (`mixed_h1_slot_map`), verified slot-for-slot and
// bit-for-bit against MFEM 4.10 probes on `data/llnl-p3.mesh`,
// `data/star-mixed-p2.mesh` and `data/fichera-mixed-p2.mesh`
// (`crates/io/tests/d117_mixed_h1_geometry.rs`; probe `tmp/d117/d117_dump.cpp`).
//
// Storage (D624): `GeometryData` (`crates/mesh/src/simplex.rs`) stores a
// mixed table **ragged** — `nodes_per_elem == 0` with per-element rows of
// `h1_family_dofs(element family, order)` nodes concatenated in element order
// and addressed by `Mesh::geometry_row_range`.  `build_h1_geometry` attaches
// that table for every fully-supported mixture; a mesh mixing in an
// unverified family (e.g. a non-pyramid type outside the engine's tables) is
// still refused loudly — never again the silent wrong table the pre-D117
// generic arm produced (it attached `DofManager::build_pk`'s 10-dof simplex
// rows to llnl-p3's 16-dof quads with no warning at all).

/// One local slot of a mixed-mesh geometry row: the entity it lives on and the
/// position within it (MFEM `GetElementDofs`'s entity-blocked layout).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MixedSlot {
    /// Local vertex.
    Vertex(usize),
    /// (family edge index, position from the edge's first vertex).
    Edge(usize, usize),
    /// (family tri-face index, barycentric `(a, b)` from the face's vertex 0
    /// towards vertices 1 / 2; both `≥ 1` with `a + b ≤ p - 1`).
    TriFace(usize, usize, usize),
    /// (family quad-face index, 1-based in-face GLL indices from the face's
    /// vertex 0 along vertex0→1 and vertex0→3).
    QuadFace(usize, usize, usize),
    /// Element-private interior dof.
    Interior,
}

/// The geometry families the mixed engine covers.
const MIXED_FAMILIES: [ElementType; 6] = [
    ElementType::Tri3,
    ElementType::Quad4,
    ElementType::Tet4,
    ElementType::Hex8,
    ElementType::Prism6,
    ElementType::Pyramid5,
];

/// Collapse a mesh element type to its geometry family.
fn mixed_family_type(et: ElementType) -> ElementType {
    match et {
        ElementType::Tri3 | ElementType::Tri6 => ElementType::Tri3,
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => ElementType::Quad4,
        ElementType::Tet4 | ElementType::Tet10 => ElementType::Tet4,
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => ElementType::Hex8,
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => ElementType::Prism6,
        ElementType::Pyramid5 | ElementType::Pyramid13 => ElementType::Pyramid5,
        other => other,
    }
}

/// The 1-D closed Gauss-Lobatto points on `[0, 1]` (MFEM
/// `Poly_1D::ClosedPoints`; the table every `[0,1]`-domain family builds its
/// lattice from).
fn gll01(p: usize) -> Vec<f64> {
    fem_element::quadrature::gauss_lobatto_arbitrary(p + 1)
        .0
        .iter()
        .map(|&x| 0.5 * (x + 1.0))
        .collect()
}

/// Position of `x` in a 1-D node table (exact up to floating point; the
/// reference lattices are well separated).
fn gll_index(table: &[f64], x: f64) -> Option<usize> {
    table.iter().position(|&g| (g - x).abs() < 1e-12)
}

/// Inverse of the `H1_TriangleElement` interior enumeration: the running index
/// `k` → barycentric `(a, b)` (j-outer, i-inner).
fn tri_interior_ab(p: usize, k: usize) -> (usize, usize) {
    let mut run = 0usize;
    for j in 1..p {
        for i in 1..(p - j) {
            if run == k {
                return (i, j);
            }
            run += 1;
        }
    }
    unreachable!("tri interior index {k} out of range for p = {p}")
}

/// One element geometry for the mixed engine: MFEM's entity tables plus the
/// slot descriptors **in the consumer's slot order** — the element-layer
/// reference element family every in-memory consumer evaluates the table with
/// (`H1TriPk` / `QuadQk` / `H1TetPk` / `HexQk` / `PrismPk`), so the engine's
/// rows need no second permutation at consumption time.
struct MixedFamily {
    edges: &'static [[usize; 2]],
    tri_faces: &'static [[usize; 3]],
    quad_faces: &'static [[usize; 4]],
    /// The family's faces in **MFEM `FaceVert` row order** — the order its
    /// face-dof blocks are enumerated in (`GetElementDofs` walks local faces
    /// row by row; the pyramid's base quad comes *before* its tri faces).
    /// `(is_quad, index into tri_faces/quad_faces)`.
    face_enum: Vec<(bool, usize)>,
    slots: Vec<MixedSlot>,
    interior: usize,
}

/// MFEM `Geometry::Constants<Geometry::TRIANGLE>::Edges` (fn-local in
/// `tri2d_slot_map`; module-level here for the mixed families).
const MIXED_TRI_EDGES: [[usize; 2]; 3] = [[0, 1], [1, 2], [2, 0]];

/// MFEM `Geometry::Constants<Geometry::PRISM>::FaceVert` split by geometry:
/// faces 0/1 are the bottom/top triangles, 2..4 the quadrilateral sides.
const MIXED_PRISM_TRI_FACES: [[usize; 3]; 2] = [[0, 2, 1], [3, 4, 5]];
const MIXED_PRISM_QUAD_FACES: [[usize; 4]; 3] = [[0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]];

/// MFEM `Geometry::Constants<Geometry::PYRAMID>::Edges` (`fem/geom.cpp:1076`):
/// base `{0,1},{1,2},{3,2},{0,3}`, laterals `{0,4},{1,4},{2,4},{3,4}`.
const MIXED_PYRAMID_EDGES: [[usize; 2]; 8] = [
    [0, 1],
    [1, 2],
    [3, 2],
    [0, 3],
    [0, 4],
    [1, 4],
    [2, 4],
    [3, 4],
];

/// MFEM `Geometry::Constants<Geometry::PYRAMID>::FaceVert` split by geometry:
/// face 0 is the base quadrilateral (corners listed reversed), faces 1..4 the
/// triangular sides (`fem/geom.cpp:1086-1087`).
const MIXED_PYRAMID_QUAD_FACES: [[usize; 4]; 1] = [[3, 2, 1, 0]];
const MIXED_PYRAMID_TRI_FACES: [[usize; 3]; 4] = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]];

/// MFEM `Geometry::Constants<Geometry::CUBE>::FaceVert` **verbatim**
/// (`fem/geom.cpp:1032`): the face enumeration order (z0, y0, x1, y1, x0, z1)
/// numbers the mesh's face blocks, so the mixed engine must use MFEM's own
/// table.  (This file's [`HEX_FACES`] carries the same six faces in another
/// order with equivalent corner parameterisations — good enough for
/// `hex_slot_map`'s key-based face lookup on all-hex meshes, where any
/// consistent face renumbering yields the same physical geometry — but not
/// for a mixed mesh, whose face-block bases interleave with the other
/// families'.  See D626.)
const CUBE_FACEVERT: [[usize; 4]; 6] = [
    [3, 2, 1, 0],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [4, 5, 6, 7],
];

impl MixedFamily {
    fn tri(p: usize) -> Self {
        // Consumer family: `H1TriPk` (MFEM `H1_TriangleElement`); slots are
        // classified from `dof_coords` by GLL-lattice matching — the same
        // derivation `tri2d_slot_map` pins against MFEM (D178).
        let e = p - 1;
        let n_int = if p >= 3 { e * (p - 2) / 2 } else { 0 };
        let tri = fem_element::lagrange::factory::H1TriPk::new(p);
        let coords = tri.dof_coords();
        let gll = gll01(p);
        let mut slots = Vec::with_capacity(coords.len());
        for c in &coords {
            let (x, y) = (c[0], c[1]);
            let lam = [1.0 - x - y, x, y];
            let zeros: Vec<usize> = (0..3).filter(|&m| lam[m].abs() < 1e-12).collect();
            match zeros.len() {
                2 => {
                    let v = (0..3).find(|&m| lam[m].abs() > 1e-12).unwrap();
                    slots.push(MixedSlot::Vertex(v));
                }
                1 => {
                    // local edge (a, b) with λ_a, λ_b > 0; the position from
                    // the edge's first vertex is the GLL index of λ at the
                    // *second* vertex, minus one.
                    let z = zeros[0];
                    let k = MIXED_TRI_EDGES
                        .iter()
                        .position(|&[a, b]| a != z && b != z)
                        .expect("tri edge slot");
                    let b = MIXED_TRI_EDGES[k][1];
                    let t = gll_index(&gll, lam[b]).expect("slot on the GLL lattice") - 1;
                    slots.push(MixedSlot::Edge(k, t));
                }
                _ => slots.push(MixedSlot::Interior),
            }
        }
        debug_assert_eq!(slots.len(), 3 + 3 * e + n_int);
        Self {
            edges: &MIXED_TRI_EDGES,
            tri_faces: &[],
            quad_faces: &[],
            face_enum: vec![],
            slots,
            interior: n_int,
        }
    }

    fn quad(p: usize) -> Self {
        // Consumer family: `QuadQk` on `[0,1]²` (MFEM `H1_QuadrilateralElement`
        // lattice, D151).
        let e = p - 1;
        let quad = fem_element::lagrange::factory::QuadQk::new(p);
        let coords = quad.dof_coords();
        let gll = gll01(p);
        let mut slots = Vec::with_capacity(coords.len());
        for c in &coords {
            let i = gll_index(&gll, c[0]).expect("slot on the GLL tensor grid");
            let j = gll_index(&gll, c[1]).expect("slot on the GLL tensor grid");
            let bnd = [i == 0 || i == p, j == 0 || j == p];
            match bnd.iter().filter(|&&b| b).count() {
                2 => {
                    let side = [i / p, j / p];
                    let v = QUAD_VERT_CORNERS
                        .iter()
                        .position(|&s| s == side)
                        .expect("corner slot");
                    slots.push(MixedSlot::Vertex(v));
                }
                1 => {
                    let (av, k) = if bnd[0] {
                        (1usize, if i == 0 { 3 } else { 1 })
                    } else {
                        (0usize, if j == 0 { 0 } else { 2 })
                    };
                    let vary = if av == 0 { i } else { j };
                    let la = QUAD_EDGES[k][0];
                    let t = if QUAD_VERT_CORNERS[la][av] == 0 {
                        vary - 1
                    } else {
                        p - 1 - vary
                    };
                    slots.push(MixedSlot::Edge(k, t));
                }
                _ => slots.push(MixedSlot::Interior),
            }
        }
        debug_assert_eq!(slots.len(), 4 + 4 * e + e * e);
        Self {
            edges: &QUAD_EDGES,
            tri_faces: &[],
            quad_faces: &[],
            face_enum: vec![],
            slots,
            interior: e * e,
        }
    }

    fn tet(p: usize) -> Self {
        // Consumer family: `H1TetPk` (MFEM `H1_TetrahedronElement`, D43/D49);
        // the integer barycentric slot labels are the published truth source.
        let e = p - 1;
        let nf = if p >= 3 { e * (p - 2) / 2 } else { 0 };
        let nb = if p >= 4 { e * (p - 2) * (p - 3) / 6 } else { 0 };
        let labels = fem_element::lagrange::factory::H1TetPk::slot_labels(p);
        let mut slots = Vec::with_capacity(labels.len());
        for idx in &labels {
            let zeros: Vec<usize> = (0..4).filter(|&m| idx[m] == 0).collect();
            match zeros.len() {
                3 => {
                    let v = (0..4).find(|&m| idx[m] > 0).unwrap();
                    slots.push(MixedSlot::Vertex(v));
                }
                2 => {
                    let k = TET_EDGES
                        .iter()
                        .position(|&[a, b]| idx[a] > 0 && idx[b] > 0)
                        .expect("tet edge slot");
                    let lb = TET_EDGES[k][1];
                    slots.push(MixedSlot::Edge(k, idx[lb] - 1));
                }
                1 => {
                    let m = zeros[0];
                    let fv = TET_FACES[m];
                    slots.push(MixedSlot::TriFace(m, idx[fv[1]], idx[fv[2]]));
                }
                _ => slots.push(MixedSlot::Interior),
            }
        }
        debug_assert_eq!(slots.len(), 4 + 6 * e + 4 * nf + nb);
        Self {
            edges: &TET_EDGES,
            tri_faces: &TET_FACES,
            quad_faces: &[],
            face_enum: (0..4).map(|f| (false, f)).collect(),
            slots,
            interior: nb,
        }
    }

    fn hex(p: usize) -> Self {
        // Consumer family: `HexQk` on `[-1,1]³` (MFEM `H1_HexahedronElement`
        // tensor lattice, D41).
        let e = p - 1;
        let hex = fem_element::lagrange::factory::HexQk::new(p);
        let coords = hex.dof_coords();
        let gll = fem_element::quadrature::gauss_lobatto_arbitrary(p + 1).0;
        let mut slots = Vec::with_capacity(coords.len());
        for c in &coords {
            let mut idx = [0usize; 3];
            for (d, &x) in c.iter().enumerate() {
                idx[d] = gll_index(&gll, x).expect("slot on the GLL tensor grid");
            }
            let on_bnd = [
                idx[0] == 0 || idx[0] == p,
                idx[1] == 0 || idx[1] == p,
                idx[2] == 0 || idx[2] == p,
            ];
            match on_bnd.iter().filter(|&&b| b).count() {
                3 => {
                    let side = [idx[0] / p, idx[1] / p, idx[2] / p];
                    let v = HEX_CORNERS.iter().position(|&s| s == side).expect("corner slot");
                    slots.push(MixedSlot::Vertex(v));
                }
                2 => {
                    let bnd: Vec<usize> = (0..3).filter(|&d| on_bnd[d]).collect();
                    let av = (0..3).find(|&d| !on_bnd[d]).unwrap();
                    let key = (av, idx[bnd[0]] / p, idx[bnd[1]] / p);
                    let k = (0..12)
                        .position(|k| {
                            let (ca, cb) =
                                (HEX_CORNERS[HEX_EDGES[k][0]], HEX_CORNERS[HEX_EDGES[k][1]]);
                            let a = match (0..3).find(|&d| ca[d] != cb[d]) {
                                Some(d) => d,
                                None => return false,
                            };
                            let f: Vec<usize> = (0..3).filter(|&d| d != a).collect();
                            (a, ca[f[0]], ca[f[1]]) == key
                        })
                        .expect("hex edge slot");
                    let la = HEX_EDGES[k][0];
                    let t = if HEX_CORNERS[la][av] == 0 {
                        idx[av] - 1
                    } else {
                        p - 1 - idx[av]
                    };
                    slots.push(MixedSlot::Edge(k, t));
                }
                1 => {
                    let ax = (0..3).find(|&d| on_bnd[d]).unwrap();
                    let side = idx[ax] / p;
                    let f = (0..6)
                        .find(|&f| CUBE_FACEVERT[f].iter().all(|&v| HEX_CORNERS[v][ax] == side))
                        .expect("hex face slot");
                    let fv = CUBE_FACEVERT[f];
                    let (l0, l1, l3) = (fv[0], fv[1], fv[3]);
                    let mut in_face = [0usize; 2];
                    for (s, &lk) in [l1, l3].iter().enumerate() {
                        let d = (0..3)
                            .find(|&d| HEX_CORNERS[l0][d] != HEX_CORNERS[lk][d])
                            .expect("face axis");
                        assert!(d != ax, "degenerate face slot");
                        in_face[s] = if HEX_CORNERS[l0][d] == 0 { idx[d] } else { p - idx[d] };
                        assert!(in_face[s] > 0 && in_face[s] < p, "slot on a face edge");
                    }
                    slots.push(MixedSlot::QuadFace(f, in_face[0], in_face[1]));
                }
                _ => slots.push(MixedSlot::Interior),
            }
        }
        debug_assert_eq!(slots.len(), 8 + 12 * e + 6 * e * e + e * e * e);
        Self {
            edges: &HEX_EDGES,
            tri_faces: &[],
            quad_faces: &CUBE_FACEVERT,
            face_enum: (0..6).map(|f| (true, f)).collect(),
            slots,
            interior: e * e * e,
        }
    }

    fn prism(p: usize) -> Self {
        // Consumer family: `PrismPk` (layer-major Gauss-Lobatto wedge — the
        // element `Mesh::element_jacobian` and `geo_ref_elem` evaluate); the
        // MFEM entity layout comes from `H1PrismPk`'s slot labels (D168/D191
        // ground truth) paired with its node positions (`prism_h1_slots`).
        // Rows are re-laid into the layer-major order by lattice-point
        // matching — the same re-lay D295's `layer_perm` performs on the
        // all-prism reader path.
        let e = p - 1;
        let nf_tri = if p >= 3 { e * (p - 2) / 2 } else { 0 };
        let h1_slots = prism_h1_slots(p).expect("prism_h1_slots");
        let mut h1: Vec<(MixedSlot, [f64; 3])> = Vec::with_capacity(h1_slots.len());
        for slot in &h1_slots {
            let s = match slot.entity {
                H1PrismSlot::Vertex(v) => MixedSlot::Vertex(v),
                H1PrismSlot::Edge(k, j) => MixedSlot::Edge(k, j),
                // `TriFace`'s running index is the `H1_TriangleElement`
                // interior order (j-outer i-inner) of the local face; its face
                // index is already the local face number (0/1).
                H1PrismSlot::TriFace(f, k) => {
                    let (a, b) = tri_interior_ab(p, k);
                    MixedSlot::TriFace(f, a, b)
                }
                // `QuadFace`'s face index is the local face number 2..4; the
                // quad-face table here is the quad-only list, so re-base it.
                H1PrismSlot::QuadFace(f, i, j) => MixedSlot::QuadFace(f - 2, i, j),
                H1PrismSlot::Interior(_) => MixedSlot::Interior,
            };
            h1.push((s, slot.xi));
        }
        let layer_coords = fem_element::lagrange::PrismPk::new(p).dof_coords();
        let mut slots = Vec::with_capacity(layer_coords.len());
        for c in &layer_coords {
            let m = h1
                .iter()
                .position(|(_, x)| (0..3).all(|d| (x[d] - c[d]).abs() < 1e-12))
                .expect("PrismPk slot not on the H1 wedge lattice");
            slots.push(h1[m].0);
        }
        let n_int = nf_tri * e;
        debug_assert_eq!(slots.len(), 6 + 9 * e + 3 * e * e);
        Self {
            edges: &PRISM_EDGES,
            tri_faces: &MIXED_PRISM_TRI_FACES,
            quad_faces: &MIXED_PRISM_QUAD_FACES,
            face_enum: vec![(false, 0), (false, 1), (true, 0), (true, 1), (true, 2)],
            slots,
            interior: n_int,
        }
    }

    /// MFEM's H1 pyramid, `pyr_type = 1` — the **Fuentes** element
    /// (`H1_FECollection`'s default; `H1_FuentesPyramidElement`,
    /// `p(p²+3)+1` dofs).  Consumer family:
    /// `h1_pyramid_element(p, PyramidBasisType::default())` — the element
    /// `Mesh::element_jacobian` evaluates — whose slot order *is* MFEM's own
    /// entity order, so no re-lay is needed (unlike the prism's layer-major
    /// consumer).
    ///
    /// The slots are generated by mirroring the element-layer truth
    /// [`h1_fuentes_pyramid_nodes`] block by block (5 vertices, 8 edge blocks
    /// in `Constants<PYRAMID>::Edges` order, the base-quad face block, the
    /// four tri-face blocks, the `(p−1)³` interior) and `debug_assert`ed
    /// position-for-position against that table — no second hand-copy of the
    /// node lattice.
    fn pyr(p: usize) -> Self {
        let e = p - 1;
        let (g_raw, _) = fem_element::quadrature::gauss_lobatto_arbitrary(p + 1);
        let cp: Vec<f64> = g_raw.iter().map(|&x| 0.5 * (x + 1.0)).collect();
        let mut slots: Vec<MixedSlot> =
            Vec::with_capacity(fem_element::lagrange::fuentes_pyramid_n_dofs(p));
        for v in 0..5 {
            slots.push(MixedSlot::Vertex(v));
        }
        if p >= 2 {
            // Edge blocks in `Constants<PYRAMID>::Edges` order; `t` counted
            // from the edge's first vertex (apex edges run base→apex).
            for k in 0..8 {
                for i in 1..p {
                    slots.push(MixedSlot::Edge(k, i - 1));
                }
            }
            // Base quad face (FaceVert row 0, corners {3,2,1,0}): the block's
            // node (cp[i], cp[p−j], 0) sits at in-face index `i` along 3→2
            // (+x) and index `j` along 3→0 (−y).
            for j in 1..p {
                for i in 1..p {
                    slots.push(MixedSlot::QuadFace(0, i, j));
                }
            }
            // Tri faces (FaceVert rows 1..4): the block node labelled (i, j)
            // has barycentric (p−i−j, i, j)/p w.r.t. the row's corner list.
            for f in 0..4 {
                for j in 1..p {
                    for i in 1..p - j {
                        slots.push(MixedSlot::TriFace(f + 1, i, j));
                    }
                }
            }
            // Interior: Fuentes' (p−1)³ bubble tensor grid.
            for _k in 1..p {
                for _j in 1..p {
                    for _i in 1..p {
                        slots.push(MixedSlot::Interior);
                    }
                }
            }
        }
        // Truth-source pin: the generated layout must reproduce the
        // element-layer node table — count, vertices, edge-block positions.
        #[cfg(debug_assertions)]
        {
            let truth = fem_element::lagrange::h1_fuentes_pyramid_nodes(p);
            assert_eq!(slots.len(), truth.len(), "pyramid slot count");
            let corner_idx = [
                [0usize, 0, 0],
                [p, 0, 0],
                [p, p, 0],
                [0, p, 0],
                [0, 0, p],
            ];
            let pos_of = |k: usize, t: usize| -> [f64; 3] {
                let (ea, eb) = (MIXED_PYRAMID_EDGES[k][0], MIXED_PYRAMID_EDGES[k][1]);
                let q = (t + 1) as f64 / p as f64;
                std::array::from_fn(|d| {
                    cp[corner_idx[ea][d]] * (1.0 - q) + cp[corner_idx[eb][d]] * q
                })
            };
            let mut m = 0usize;
            for (s, node) in slots.iter().zip(truth.iter()) {
                match (s, m) {
                    (MixedSlot::Vertex(v), _) => {
                        let want: [f64; 3] = std::array::from_fn(|d| cp[corner_idx[*v][d]]);
                        assert!(
                            (0..3).all(|d| (want[d] - node[d]).abs() < 1e-12),
                            "pyramid vertex slot {v}"
                        );
                    }
                    (MixedSlot::Edge(k, t), _) if m >= 5 => {
                        let want = pos_of(*k, *t);
                        assert!(
                            (0..3).all(|d| (want[d] - node[d]).abs() < 1e-12),
                            "pyramid edge slot ({k},{t})"
                        );
                    }
                    _ => {}
                }
                m += 1;
            }
        }
        Self {
            edges: &MIXED_PYRAMID_EDGES,
            tri_faces: &MIXED_PYRAMID_TRI_FACES,
            quad_faces: &MIXED_PYRAMID_QUAD_FACES,
            face_enum: vec![(true, 0), (false, 0), (false, 1), (false, 2), (false, 3)],
            slots,
            interior: e * e * e,
        }
    }
}

/// The element rows of a mesh to map (mixed-engine input).
pub(crate) struct MixedMeshRows {
    rows: Vec<(ElementType, Vec<NodeId>)>,
    n_nodes: usize,
    dim: usize,
}

impl MixedMeshRows {
    fn families(&self, p: usize) -> Result<HashMap<ElementType, MixedFamily>, String> {
        let mut fams: HashMap<ElementType, MixedFamily> = HashMap::new();
        for (et, _) in &self.rows {
            let fam_t = mixed_family_type(*et);
            if fams.contains_key(&fam_t) {
                continue;
            }
            let f = match fam_t {
                ElementType::Tri3 => MixedFamily::tri(p),
                ElementType::Quad4 => MixedFamily::quad(p),
                ElementType::Tet4 => MixedFamily::tet(p),
                ElementType::Hex8 => MixedFamily::hex(p),
                ElementType::Prism6 => MixedFamily::prism(p),
                ElementType::Pyramid5 => MixedFamily::pyr(p),
                other => return Err(format!("unsupported element family {other:?}")),
            };
            fams.insert(fam_t, f);
        }
        Ok(fams)
    }
}

/// D118: the mixed-mesh H1 `nodes` slot map — for every element, the file dof
/// of every reference slot (in the consumer's slot order), plus the global dof
/// count.
///
/// MFEM's numbering (`FiniteElementSpace::GetElementDofs`, `fem/fespace.cpp`):
/// vertex dofs, per-edge blocks of `p-1`, per-face blocks sized by the *face*
/// geometry (MFEM's `var_face_dofs` for mixed faces), then per-element
/// interior blocks sized by the *element* geometry; edges/faces are enumerated
/// by element traversal × local entity order (first encounter wins) and each
/// face is stored with its first-encountering parameterisation.
fn mixed_h1_slot_map(
    mesh: &MixedMeshRows,
    p: usize,
) -> Result<(Vec<Vec<NodeId>>, usize), String> {
    let (rows, _, n_dofs) = mixed_h1_engine(mesh, p)?;
    Ok((rows, n_dofs))
}

// ─── D624: frame-invariant slot keys (tet-rotation value transfer) ──────────
//
// A 3-D mesh containing tets is `mark_tet_mesh_for_refinement`-rotated before
// `build_h1_geometry` runs (MFEM parity).  The mixed slot map is therefore
// computed on the *pre-rotation* mesh — whose numbering is the file's — and
// every slot carries a **frame-invariant physical key**; after the rotation
// the post-frame rows are re-derived and their values recovered through the
// keys:
//
// * `Vertex` — the global node id;
// * `Edge` — the sorted endpoint pair + the position along the ascending
//   direction (orientation canonicalised, so a rotated element's edge block
//   maps onto the same key);
// * `TriFace` — the sorted corner list as the canonical frame: the tri dof
//   machinery accepts *any* corner permutation (`tri_orientation` finds it),
//   so a rotated tet's listing maps onto the same key;
// * `QuadFace` — the slot's offset in the face's **stored** (first-encounter)
//   frame: only hexes/prisms/pyramids own quad faces and none of them rotate,
//   so the stored frame is rotation-invariant (the sorted-corner trick above
//   is unusable here — the quad dof machinery needs a cyclic corner order);
// * `Interior` — element-private.  A tet's interior lattice is permuted by
//   the rotation, so its key is the barycentric label re-expressed in the
//   *sorted-vertex* frame; every other family is never rotated and keys on
//   `(element, slot)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum MixedSlotKey {
    V(NodeId),
    E([NodeId; 2], usize),
    T([NodeId; 3], usize),
    Q([NodeId; 4], usize),
    /// Packed interior key: bits 62-63 tag the flavour, bits 32.. the
    /// element, low byte(s) the barycentric label (tet) or slot (others).
    I(u64),
}

/// The mixed slot map **and** the frame-invariant keys, in one enumeration
/// (the keys need the same entity tables the dof ids are built from).
fn mixed_h1_engine(
    mesh: &MixedMeshRows,
    p: usize,
) -> Result<(Vec<Vec<NodeId>>, Vec<Vec<MixedSlotKey>>, usize), String> {
    if p < 2 {
        return Err("order < 2".into());
    }
    let e_dof = p - 1;
    let nf_tri = if p >= 3 { e_dof * (p - 2) / 2 } else { 0 };
    let nf_quad = e_dof * e_dof;
    let n_elems = mesh.rows.len();
    let n_vert = mesh.n_nodes;
    let dim = mesh.dim;

    let fams = mesh.families(p)?;
    let fam_of =
        |et: &ElementType| -> &MixedFamily { fams.get(&mixed_family_type(*et)).expect("checked") };

    // MFEM's entity enumeration: element traversal × local entity order,
    // first encounter wins.
    let mut edge_ids: HashMap<[u32; 2], u32> = HashMap::new();
    let mut face_index: HashMap<Vec<u32>, usize> = HashMap::new();
    // (b't' tri / b'q' quad, stored corner vertex ids)
    let mut faces: Vec<(u8, Vec<u32>)> = Vec::new();
    for (et, ns) in &mesh.rows {
        let fam = fam_of(et);
        for &[la, lb] in fam.edges {
            let (a, b) = (ns[la], ns[lb]);
            let key = if a < b { [a, b] } else { [b, a] };
            let next = edge_ids.len() as u32;
            edge_ids.entry(key).or_insert(next);
        }
        if dim == 3 {
            // Faces in the family's MFEM `FaceVert` row order — the pyramid's
            // base quad precedes its tri faces.
            for &(is_quad, f) in &fam.face_enum {
                let (g, fv): (u8, &[usize]) = if is_quad {
                    (b'q', &fam.quad_faces[f])
                } else {
                    (b't', &fam.tri_faces[f])
                };
                let mut key: Vec<u32> = fv.iter().map(|&v| ns[v]).collect();
                key.sort_unstable();
                let next = faces.len();
                face_index.entry(key).or_insert_with(|| {
                    faces.push((g, fv.iter().map(|&v| ns[v]).collect()));
                    next
                });
            }
        }
    }
    let n_edges = edge_ids.len();
    // Per-face dof blocks in face order (MFEM's `var_face_dofs`).
    let mut face_base: Vec<usize> = Vec::with_capacity(faces.len());
    let mut acc = 0usize;
    for (g, _) in &faces {
        face_base.push(acc);
        acc += if *g == b't' { nf_tri } else { nf_quad };
    }
    let n_face_dofs = acc;

    // Per-element interior offsets (MFEM's `bdofs`).
    let mut interior_base: Vec<usize> = Vec::with_capacity(n_elems);
    acc = 0;
    for (et, _) in &mesh.rows {
        interior_base.push(acc);
        acc += fam_of(et).interior;
    }
    let n_interior = acc;

    let edge_base = n_vert;
    let face_dof_base = edge_base + n_edges * e_dof;
    let int_base = face_dof_base + n_face_dofs;
    let n_dofs = int_base + n_interior;

    // Tet interior barycentric labels (the only interior block whose element
    // frame the reader ever rotates).
    let tet_labels = if fams.contains_key(&ElementType::Tet4) {
        Some(fem_element::lagrange::factory::H1TetPk::slot_labels(p))
    } else {
        None
    };

    let mut rows: Vec<Vec<NodeId>> = Vec::with_capacity(n_elems);
    let mut keys: Vec<Vec<MixedSlotKey>> = Vec::with_capacity(n_elems);
    for (el, (et, ns)) in mesh.rows.iter().enumerate() {
        let fam = fam_of(et);
        let mut row = Vec::with_capacity(fam.slots.len());
        let mut krow: Vec<MixedSlotKey> = Vec::with_capacity(fam.slots.len());
        let mut interior_seen = 0usize;
        for (m, s) in fam.slots.iter().enumerate() {
            let (g, k): (usize, MixedSlotKey) = match *s {
                MixedSlot::Vertex(v) => (ns[v] as usize, MixedSlotKey::V(ns[v])),
                MixedSlot::Edge(k, t_local) => {
                    let (a, b) = (ns[fam.edges[k][0]], ns[fam.edges[k][1]]);
                    let key = if a < b { [a, b] } else { [b, a] };
                    let ei = *edge_ids.get(&key).expect("edge enumerated") as usize;
                    let t = if a < b { t_local } else { e_dof - 1 - t_local };
                    let kk = MixedSlotKey::E(key, t);
                    (edge_base + ei * e_dof + t, kk)
                }
                MixedSlot::TriFace(f, a, b) => {
                    let j = tri_face_index(p, a, b).expect("tri face slot in range");
                    let fv = fam.tri_faces[f];
                    let test = [ns[fv[0]], ns[fv[1]], ns[fv[2]]];
                    let mut key: Vec<u32> = test.to_vec();
                    key.sort_unstable();
                    let fi = *face_index.get(&key).expect("face enumerated");
                    let stored: [u32; 3] = match &faces[fi] {
                        (b't', verts) => verts[..].try_into().unwrap(),
                        _ => return Err("tri face slot matched a quadrilateral face".into()),
                    };
                    // dof id: position in the face's stored frame
                    let orient = tri_orientation(&stored, &test).expect("face corner mismatch");
                    let canon = tri_dof_ord(p, orient, j).expect("tri face dof order");
                    // key: position in the (rotation-invariant) sorted frame
                    let sorted: [u32; 3] = key.clone().try_into().unwrap();
                    let orient_s = tri_orientation(&sorted, &test).expect("face corner mismatch");
                    let canon_s = tri_dof_ord(p, orient_s, j).expect("tri face dof order");
                    let kk = MixedSlotKey::T(
                        [
                            key[0] as NodeId,
                            key[1] as NodeId,
                            key[2] as NodeId,
                        ],
                        canon_s,
                    );
                    (face_dof_base + face_base[fi] + canon, kk)
                }
                MixedSlot::QuadFace(f, i, j) => {
                    let fv = fam.quad_faces[f];
                    let mut key: Vec<u32> = fv.iter().map(|&v| ns[v]).collect();
                    key.sort_unstable();
                    let fi = *face_index.get(&key).expect("face enumerated");
                    let stored: [u32; 4] = match &faces[fi] {
                        (b'q', verts) => verts[..].try_into().unwrap(),
                        _ => return Err("quad face slot matched a triangular face".into()),
                    };
                    let off = quad_face_canonical_slot(
                        &stored,
                        ns[fv[0]],
                        ns[fv[1]],
                        ns[fv[3]],
                        [i, j],
                        p,
                    )
                    .ok_or("quad face slot outside the stored face")?;
                    let kk = MixedSlotKey::Q(
                        [
                            key[0] as NodeId,
                            key[1] as NodeId,
                            key[2] as NodeId,
                            key[3] as NodeId,
                        ],
                        off,
                    );
                    (face_dof_base + face_base[fi] + off, kk)
                }
                MixedSlot::Interior => {
                    let g = int_base + interior_base[el] + interior_seen;
                    interior_seen += 1;
                    let kk = match &tet_labels {
                        Some(labels) if mixed_family_type(*et) == ElementType::Tet4 => {
                            // Barycentric label re-expressed in the
                            // sorted-vertex frame (rotation-invariant).
                            let label = labels[m];
                            let mut order: Vec<NodeId> = ns.to_vec();
                            order.sort_unstable();
                            let mut packed: u64 = 1 << 62;
                            for (r, &gn) in order.iter().enumerate() {
                                let local =
                                    ns.iter().position(|&n| n == gn).expect("vertex present");
                                packed |= (label[local] as u64) << (r * 8);
                            }
                            packed |= (el as u64) << 32;
                            MixedSlotKey::I(packed)
                        }
                        _ => MixedSlotKey::I((2 << 62) | ((el as u64) << 32) | (m as u64)),
                    };
                    (g, kk)
                }
            };
            row.push(g as NodeId);
            krow.push(k);
        }
        debug_assert_eq!(interior_seen, fam.interior);
        rows.push(row);
        keys.push(krow);
    }
    Ok((rows, keys, n_dofs))
}

/// The file's slot values keyed for the rotation transfer
/// ([`build_h1_geometry`]'s `mixed_values` argument): for every physical slot
/// key, the coordinate the *file* stores at the slot's dof.
pub(crate) struct MixedFileValues {
    map: std::collections::HashMap<MixedSlotKey, [f64; 3]>,
}

impl MixedFileValues {
    /// Capture the file-frame values of every mixed slot: `slot_xyz` maps a
    /// global dof id to its coordinate triple.
    pub(crate) fn capture(
        mesh: &MixedMeshRows,
        p: usize,
        dof_xyz: impl Fn(usize) -> [f64; 3],
    ) -> Result<MixedFileValues, String> {
        let (rows, keys, n_dofs) = mixed_h1_engine(mesh, p)?;
        let mut map = std::collections::HashMap::with_capacity(n_dofs);
        for (row, krow) in rows.iter().zip(keys.iter()) {
            for (&g, &k) in row.iter().zip(krow.iter()) {
                map.entry(k).or_insert_with(|| dof_xyz(g as usize));
            }
        }
        Ok(MixedFileValues { map })
    }
}

/// One geometry slot of a mixed-mesh table row
/// ([`read_mfem_mixed_h1_geometry_file`]).
pub struct MixedGeoSlot {
    /// The file dof this slot maps to (MFEM's H1 numbering).
    pub dof: NodeId,
    /// The file's coordinate at that dof (`sdim` components + zero padding).
    pub xyz: [f64; 3],
}

/// The mixed-mesh high-order geometry a file's H1 `nodes` section describes
/// ([`read_mfem_mixed_h1_geometry_file`]).
pub struct MixedH1Geometry {
    pub order: u8,
    /// Components per dof (the mesh's topological dimension).
    pub sdim: usize,
    /// MFEM's global scalar H1 dof count (`FiniteElementSpace::GetNDofs`).
    pub n_dofs: usize,
    /// Per element: (element type, row in the consumer's slot order).
    pub elems: Vec<(ElementType, Vec<MixedGeoSlot>)>,
}

/// D118 probe API: read an MFEM `.mesh` file and return the **mixed-mesh**
/// high-order geometry its H1 `nodes` section describes, or `Ok(None)` when
/// the mesh is uniform (its geometry is attached by [`read_mfem`]/[`read_mfem_file`]
/// directly) or the file carries no H1 `nodes` section.
///
/// This exists because [`GeometryData`](fem_mesh::simplex::GeometryData) is
/// uniform-stride (one `nodes_per_elem` for every element), so the reader
/// cannot attach a mixed table yet (D624); the engine's table is validated
/// against MFEM 4.10 in `crates/io/tests/d117_mixed_h1_geometry.rs`.  The
/// parse here is the minimal slice of the MFEM v1.0 grammar the engine needs
/// (dimension, elements, vertices/nodes header) — the main reader stays
/// `read_mfem`.
pub fn read_mfem_mixed_h1_geometry_file(
    path: impl AsRef<std::path::Path>,
) -> FemResult<Option<MixedH1Geometry>> {
    let text = std::fs::read_to_string(path.as_ref()).map_err(FemError::from)?;
    parse_mixed_h1_geometry(&text)
        .map_err(|e| FemError::Mesh(format!("read_mfem_mixed_h1_geometry_file: {e}")))
}

fn parse_mixed_h1_geometry(text: &str) -> Result<Option<MixedH1Geometry>, String> {
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'));
    let _magic = lines.next().ok_or("empty file")?;
    let mut dim = 0usize;
    let mut elem_rows: Vec<(ElementType, Vec<NodeId>)> = Vec::new();
    let mut n_vert = 0usize;
    let mut order = 0u8;
    let mut closed_uniform = false;
    let mut vdim = 0usize;
    let mut ordering = 0usize;
    let mut raw: Vec<f64> = Vec::new();
    let mut seen_vertices = false;
    while let Some(line) = lines.next() {
        match line {
            "dimension" => {
                dim = lines
                    .next()
                    .ok_or("no dimension")?
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("bad dimension")?;
            }
            "elements" => {
                let n: usize = lines
                    .next()
                    .ok_or("no element count")?
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("bad element count")?;
                for _ in 0..n {
                    let row = lines.next().ok_or("short elements")?;
                    let t: Vec<&str> = row.split_whitespace().collect();
                    let code: usize = t[1].parse().map_err(|_| "bad geometry code")?;
                    let et = mfem_elem_type(code as u32).ok_or("unsupported geometry code")?;
                    elem_rows.push((et, t[2..].iter().map(|v| v.parse().unwrap()).collect()));
                }
            }
            "boundary" | "edges" => {
                let n: usize = lines
                    .next()
                    .unwrap_or("0")
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
                for _ in 0..n {
                    lines.next();
                }
            }
            "vertices" => {
                seen_vertices = true;
                n_vert = lines
                    .next()
                    .ok_or("no vertex count")?
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("bad vertex count")?;
                // The next token is the space dimension (straight vertices
                // follow) or the `nodes` keyword (curved).
                let next = lines.next().ok_or("short vertices")?;
                if next == "nodes" {
                    let _fes = lines.next().ok_or("short nodes")?;
                    let fec = lines.next().ok_or("short nodes")?;
                    let name = fec.split(':').nth(1).unwrap_or("").trim();
                    vdim = lines
                        .next()
                        .ok_or("short nodes")?
                        .split_whitespace()
                        .last()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(dim);
                    ordering = lines
                        .next()
                        .ok_or("short nodes")?
                        .split_whitespace()
                        .last()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    let parsed = parse_nodal_fec(name).ok_or("no H1 `nodes` collection")?;
                    order = parsed.order;
                    closed_uniform = parsed.closed_uniform;
                    for l in lines.by_ref() {
                        for v in l.split_whitespace() {
                            if let Ok(x) = v.parse::<f64>() {
                                raw.push(x);
                            }
                        }
                    }
                }
                // straight vertices: the coordinates are not needed (the
                // values come from the `nodes` section), only `n_vert`.
            }
            _ => {}
        }
    }
    if dim == 0 {
        return Err("no dimension".into());
    }
    if !seen_vertices {
        return Err("no vertices section".into());
    }
    if order < 2 || closed_uniform || raw.is_empty() {
        return Ok(None); // no high-order H1 `nodes` section
    }
    // 0-based vs 1-based vertex ids (the reader's heuristic).
    let max_idx = elem_rows
        .iter()
        .flat_map(|(_, ns)| ns.iter())
        .copied()
        .max()
        .unwrap_or(0);
    let has_zero = elem_rows
        .iter()
        .flat_map(|(_, ns)| ns.iter())
        .any(|&v| v == 0);
    if !(has_zero || max_idx as usize + 1 == n_vert) {
        for (_, ns) in elem_rows.iter_mut() {
            for v in ns.iter_mut() {
                *v -= 1;
            }
        }
    }
    let distinct: usize = {
        let mut fams: Vec<ElementType> = Vec::new();
        for (et, _) in &elem_rows {
            let f = mixed_family_type(*et);
            if !fams.contains(&f) {
                fams.push(f);
            }
        }
        // a mixture of *raw* types within one supported family (Tri3+Tri6,
        // Quad4+Quad8, ...) still needs per-type rows
        let mut raw_types: Vec<ElementType> = Vec::new();
        for (et, _) in &elem_rows {
            if !raw_types.contains(et) {
                raw_types.push(*et);
            }
        }
        fams.len().max(if raw_types.len() > 1 { 2 } else { 1 })
    };
    if distinct <= 1 {
        return Ok(None); // uniform: the normal reader attaches the geometry
    }
    let mesh = MixedMeshRows { rows: elem_rows, n_nodes: n_vert, dim };
    let (rows, n_dofs) = mixed_h1_slot_map(&mesh, order as usize)?;
    let sdim = dim.min(vdim.max(1));
    let ndof = raw.len() / vdim.max(1);
    let dof_xyz = |d: usize| -> [f64; 3] {
        let mut out = [0.0f64; 3];
        match ordering {
            0 => {
                for c in 0..sdim {
                    out[c] = raw[c * ndof + d];
                }
            }
            _ => {
                for c in 0..sdim {
                    out[c] = raw[d * vdim + c];
                }
            }
        }
        out
    };
    let mut elems = Vec::with_capacity(rows.len());
    for (i, row) in rows.into_iter().enumerate() {
        let et = mesh.rows[i].0;
        elems.push((
            et,
            row.into_iter()
                .map(|d| MixedGeoSlot { dof: d, xyz: dof_xyz(d as usize) })
                .collect(),
        ));
    }
    Ok(Some(MixedH1Geometry { order, sdim, n_dofs, elems }))
}

/// Build the high-order `GeometryData` for an H1-continuous `nodes` section:
/// the file stores one coordinate triple per DOF of the order-`p` H1 space and
/// the per-element geometry tables are the element DOF lists (H1 topological
/// order, matching the `QuadQk::new(p)`/`HexQk::new(p)` assembly bases).
///
/// D41: the H1 DOF numbering of the file is *MFEM's*, not fem-rs's.  The
/// faithful routes by element type:
/// - hexahedra → [`build_h1_hex_geometry`] (D41; the `DofManager` slot order
///   differs and the difference is silent on load);
/// - tetrahedra → [`build_h1_tet_geometry`] (D43);
/// - prisms (wedges) → the generic `DofManager` arm, whose
///   `build_prism_h1` reproduces MFEM's entity-ordered numbering since D177
///   (pinned for orders 2-4 in `crates/space/tests/
///   d177_prism_h1_mfem_numbering.rs`, round-tripped on a curved 2-prism
///   mesh by `tests/d190_wedge_curved_nodes_roundtrip.rs`); D295 re-lays the
///   all-`Prism6` rows from MFEM's entity slot order into `PrismPk`'s
///   layer-major order via `H1PrismPk::layer_perm`, because the geometry
///   table every in-memory consumer evaluates is contracted to `PrismPk`'s
///   slot order (the dof *ids* stay MFEM's file numbering);
/// - mixed meshes → the D117 gate below: the numbering itself is mapped by
///   [`mixed_h1_slot_map`] (verified against MFEM 4.10), but a mixed table
///   needs per-element row lengths which the uniform-stride `GeometryData`
///   cannot store (D624), so the mesh is read straight-sided *with a precise
///   warning* — never the silent wrong table the generic arm produced before
///   D117 (it gave llnl-p3's 16-dof quads 10-dof simplex rows with no
///   warning), and never an unverified numbering for pyramid-containing
///   mixtures;
/// - anything else (uniform pyramids): fem-rs's own numbering, *with* a
///   warning — never a silently scrambled mapping (see `D41` notes below).
/// D624: attach the mixed-mesh high-order geometry as a **ragged**
/// `GeometryData` (`nodes_per_elem == 0`; per-family rows concatenated in
/// element order, addressed by `Mesh::geometry_row_range`).
///
/// Values come either straight from the file's dof ids (no rotation happened:
/// 2-D, or 3-D without tets) or — when the reader rotated tets before this
/// call (`mark_tet_mesh_for_refinement`) — through the frame-invariant slot
/// keys captured pre-rotation ([`MixedFileValues`]).
fn build_h1_geometry_mixed<M: MeshTopology>(
    mesh: &M,
    order: u8,
    raw: &[f64],
    ordering: usize,
    dim: usize,
    vdim: usize,
    mixed_values: Option<&MixedFileValues>,
) -> Option<GeometryData> {
    let rows_mesh = MixedMeshRows {
        rows: (0..mesh.n_elements() as u32)
            .map(|e| (mesh.element_type(e), mesh.element_nodes(e).to_vec()))
            .collect(),
        n_nodes: mesh.n_nodes(),
        dim,
    };
    let (rows, keys, n_dofs) = match mixed_h1_engine(&rows_mesh, order as usize) {
        Ok(v) => v,
        Err(why) => {
            eprintln!(
                "warning (D624): refusing high-order `nodes` geometry for a mixed mesh \
                 ({why}); the mesh is read as straight-sided (geometric order 1)"
            );
            return None;
        }
    };
    if raw.len() < vdim * n_dofs {
        return None;
    }
    let mut coords = vec![0.0f64; n_dofs * dim];
    match ordering {
        0 => {
            for c in 0..dim {
                for d in 0..n_dofs {
                    coords[d * dim + c] = raw[c * n_dofs + d];
                }
            }
        }
        1 => {
            for d in 0..n_dofs {
                coords[d * dim..(d + 1) * dim]
                    .copy_from_slice(&raw[d * vdim..d * vdim + dim]);
            }
        }
        _ => return None,
    }
    if let Some(mv) = mixed_values {
        // Rotated frame: the dof ids above are this frame's; recover every
        // slot's *value* through its frame-invariant key.
        let mut filled = vec![false; n_dofs];
        for (row, krow) in rows.iter().zip(keys.iter()) {
            for (&g, &k) in row.iter().zip(krow.iter()) {
                let g = g as usize;
                let v = match mv.map.get(&k) {
                    Some(v) => *v,
                    None => {
                        eprintln!(
                            "warning (D624): no file value for a mixed slot key after the \
                             tet renumbering; the mesh is read as straight-sided"
                        );
                        return None;
                    }
                };
                coords[g * dim..(g + 1) * dim].copy_from_slice(&v[..dim]);
                filled[g] = true;
            }
        }
        if filled.iter().any(|&f| !f) {
            eprintln!(
                "warning (D624): mixed geometry has unfilled dofs after the tet renumbering; \
                 the mesh is read as straight-sided"
            );
            return None;
        }
    }
    let mut conn: Vec<NodeId> = Vec::with_capacity(rows.iter().map(|r| r.len()).sum());
    for row in &rows {
        conn.extend_from_slice(row);
    }
    Some(GeometryData {
        order,
        conn,
        nodes_per_elem: 0,
        coords,
        n_nodes: n_dofs,
    })
}

fn build_h1_geometry<M: MeshTopology>(
    mesh: &M,
    order: u8,
    raw: &[f64],
    ordering: usize,
    dim: usize,
    vdim: usize,
    tet_file_slots: Option<&TetFileSlots>,
    mixed_values: Option<&MixedFileValues>,
) -> Option<GeometryData> {
    if order < 2 {
        return None; // linear geometry needs no table
    }
    // D117/D624: mixed-element meshes.  `mixed_h1_slot_map` reproduces MFEM's
    // mixed numbering (tri/quad/tet/hex/prism/pyramid; verified against MFEM
    // 4.10), and since D624 `GeometryData` stores a **ragged** table
    // (`nodes_per_elem == 0`, per-family row lengths — MFEM's own `nodes`
    // contract), so a verified mixed table is attached here.  Meshes mixing
    // in a family the engine does not cover (D41) are still refused loudly:
    // never again the silent wrong table the generic arm below used to
    // produce (it gave llnl-p3's 16-dof quads 10-dof simplex rows with no
    // warning at all), and never an unverified numbering.
    if mesh.n_elements() > 1 {
        let mut fams: Vec<ElementType> = Vec::new();
        let mut unsup: Vec<ElementType> = Vec::new();
        for e in 0..mesh.n_elements() as u32 {
            let f = mixed_family_type(mesh.element_type(e));
            if MIXED_FAMILIES.contains(&f) {
                if !fams.contains(&f) {
                    fams.push(f);
                }
            } else if !unsup.contains(&f) {
                unsup.push(f);
            }
        }
        if fams.len() + unsup.len() > 1 {
            if !unsup.is_empty() {
                eprintln!(
                    "warning (D41): refusing high-order `nodes` geometry for a mixed mesh \
                     containing {}: no MFEM-verified numbering for every element type — the \
                     mesh is read as straight-sided (geometric order 1)",
                    unsup.iter().map(|f| format!("{f:?}")).collect::<Vec<_>>().join(" + ")
                );
                return None;
            }
            if dim != vdim {
                // D117 残余: a mixed *surface* (dim < spaceDim) needs the
                // sdim-projected store validated first; stay straight-sided.
                eprintln!(
                    "warning (D117/D624): mixed-element surface mesh with a high-order `nodes` \
                     section ({}): the surface-mixed geometry store is not validated; the mesh \
                     is read as straight-sided (geometric order 1)",
                    fams.iter().map(|f| format!("{f:?}")).collect::<Vec<_>>().join(" + ")
                );
                return None;
            }
            return build_h1_geometry_mixed(
                mesh,
                order,
                raw,
                ordering,
                dim,
                vdim,
                mixed_values,
            );
        }
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
                        // D190: of the remaining 3-D types, the prism (wedge)
                        // numbering IS MFEM's — `DofManager::build_prism_h1`
                        // reproduces `GetElementDofs` entity-for-entity since
                        // D177 (pinned for orders 2-4 in
                        // `crates/space/tests/d177_prism_h1_mfem_numbering.rs`
                        // and round-tripped on a curved 2-prism mesh by
                        // `tests/d190_wedge_curved_nodes_roundtrip.rs`), so an
                        // all-prism mesh reads faithfully and stays silent.
                        // Pyramids and prism/pyramid mixes still route through
                        // fem-rs's own numbering without an MFEM comparison —
                        // those keep the warning.
                        let has_unverified = (0..mesh.n_elements() as u32).any(|e| {
                            !matches!(
                                mesh.element_type(e),
                                ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18
                            )
                        });
                        if has_unverified {
                            eprintln!(
                                "warning (D41): high-order `nodes` geometry on a 3-D mesh of \
                                 pyramids (or a prism/pyramid mix) is read with fem-rs's own H1 \
                                 numbering, which is not verified against MFEM for these element \
                                 types (hexahedra, tetrahedra and prisms are verified)"
                            );
                        }
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
    // Component stride = `VDim` (`build_h1_geometry`'s caller checked that the
    // section is readable); for a `dim < spaceDim` surface mesh only the first
    // `dim` components are kept (D112b).
    if raw.len() < vdim * n_dofs {
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
            for d in 0..n_dofs {
                dof_coords[d * dim..(d + 1) * dim]
                    .copy_from_slice(&raw[d * vdim..d * vdim + dim]);
            }
        }
        _ => return None,
    }
    let mut conn: Vec<NodeId> = Vec::new();
    let npe = dm.element_dofs(0).len();
    // D295: an all-`Prism6` mesh's `DofManager` rows are numbered in MFEM's
    // entity slot order (`build_prism_h1` → `H1PrismPk`), but the
    // geometry-table contract every in-memory consumer evaluates the table
    // with is `PrismPk`'s **layer-major** order (`Mesh::set_curvature`'s
    // prism path, frozen by `crates/mesh/tests/d152_prism_curvature.rs`;
    // `element_jacobian`, `crates/assembly`'s `geo_ref_elem`,
    // `curved::CurvedMesh`, AMR's `curved_prism` and fem-io's own
    // `prism_nodes_dof_values` all evaluate `PrismPk` over these rows).
    // Storing the H1 slot order here made read → write mutually exclusive
    // with the programmatic path (the writer's shared-dof consistency check
    // rejected every curved wedge read from a file).  `H1PrismPk::layer_perm`
    // is the published slot correspondence (`perm[m]` = the layer slot of H1
    // slot `m`, same dof), so the row is re-laid out under the file's own dof
    // ids — the numbering stays MFEM's, only the row order follows the
    // reference element, exactly like the hex/tet tables above.
    let prism_perm: Option<Vec<usize>> =
        if dim == 3
            && (0..mesh.n_elements() as u32).all(|e| mesh.element_type(e) == ElementType::Prism6)
        {
            Some(fem_element::lagrange::H1PrismPk::new(order as usize)
                .layer_perm()
                .to_vec())
        } else {
            None
        };
    for e in 0..mesh.n_elements() {
        let dofs = dm.element_dofs(e as u32);
        if dofs.len() != npe {
            // D41/D112c: a mixed mesh has no single `nodes_per_elem`, so the
            // flat geometry layout cannot describe it.  Refuse loudly rather
            // than keep a table whose element rows are misaligned.
            eprintln!(
                "warning (D112c): refusing to build high-order geometry for a mixed mesh \
                 (element {} has {} geometry DOFs, element 0 has {npe}); the mesh is read as \
                 straight-sided (geometric order 1)",
                e, dofs.len()
            );
            return None;
        }
        if let Some(perm) = &prism_perm {
            debug_assert_eq!(perm.len(), dofs.len(), "prism slot permutation size");
            let mut row = vec![0 as NodeId; dofs.len()];
            for (m, &g) in dofs.iter().enumerate() {
                row[perm[m]] = g;
            }
            conn.extend(row);
        } else {
            conn.extend(dofs.iter().copied());
        }
    }
    Some(GeometryData {
        order,
        conn,
        nodes_per_elem: npe,
        coords: dof_coords,
        n_nodes: n_dofs,
    })
}

// ─── D112: legacy (`closed-uniform`) `nodes` sections ────────────────────────
//
// MFEM's legacy finite element collections — `Linear`, `Quadratic` and `Cubic`
// (`fem/fe_coll.hpp`) — are built from fixed-order elements whose DOFs sit at
// the **closed-uniform** points:
//
//   * `Lagrange1DFiniteElement(degree)` (`fem/fe/fe_fixed_order.cpp`) places
//     DOF `0` at `0`, DOF `1` at `1` and DOF `i+1` at `i/degree`;
//   * `LagrangeHexFiniteElement` is its tensor product, driven by a
//     hand-written `I`/`J`/`K` tensor-index table (vertices, edges and faces
//     sit at `i/degree`; the *interior* block of that table is not the
//     `H1_HexahedronElement` enumeration — see
//     `fem_element::lagrange::legacy`);
//   * `BiCubic2DFiniteElement`, `Cubic2DFiniteElement` and
//     `Cubic3DFiniteElement` place their edge/face/interior DOFs on the
//     equispaced lattice (`1/3`, `2/3`, …).
//
// `H1_FECollection` instead uses `BasisType::GaussLobatto`, i.e. the closed
// Gauss-Lobatto points (`0`, `(1−1/√5)/2`, `(1+1/√5)/2`, `1` at `p = 3`).  Both
// families therefore describe the *same* polynomial space with the *same* DOF
// layout — `CubicFECollection::DofForGeometry` returns exactly the `H1` counts
// (`SEGMENT 2`, `SQUARE 4`, `CUBE 8`, `TRIANGLE 1`, `TETRAHEDRON 0`) and its
// `DofOrderForOrientation` tables are the `H1` ones (`sq_ind[8][4]` is
// `QuadDofOrd[8][4]`, `{0,1}`/`{1,0}` is `SegDofOrd`) — but they disagree on
// **where the DOFs are**, so a geometry table built for the Gauss-Lobatto
// element reads the stored values as belonging to different physical points.
//
// `p <= 2` cannot show the difference: the closed-uniform and closed
// Gauss-Lobatto points coincide (`{0,1}` and `{0,½,1}`), which is why only the
// `p = 3` legacy meshes (`fichera-q3`, `star-q3`, `escher-p3`,
// `square-disc-p3`, …) were mis-read.  Before D112 the loader simply dropped
// the family and every such mesh came back with a scrambled (even inverted)
// isoparametric map and no warning at all.
//
// The repair is a re-interpolation, not a re-numbering: `conn` is already
// MFEM's (the numbering depends only on `DofForGeometry` +
// `DofOrderForOrientation`, which agree), so only the *node values* move.  For
// each element the stored values `c_j` are the nodal values of the legacy
// polynomial `L(ξ) = Σ_j c_j ψ_j(ξ)` at the closed-uniform nodes; tabulating
// `L` at the Gauss-Lobatto nodes `ξ_i` of the same slot gives
//
//     v_i = L(ξ_i) = Σ_j B[i][j]·c_j,   B[i][j] = ψ_j(ξ_i),
//
// and `Σ_i L(ξ_i)·φ^GLL_i ≡ L` (both sides are the same polynomial: the
// Gauss-Lobatto basis is nodal at `ξ_i`).  The conversion is therefore exact —
// a purely algebraic change of basis — and `GeometryData` keeps its
// "Gauss-Lobatto semantics" contract, so `assembler::geo_ref_elem` and
// `vector_assembler::geo_ref_elem_from_mesh` need no change (method A).

/// The (Gauss-Lobatto, closed-uniform) reference element pair for a geometric
/// element type at order `p`, or `None` for element types the legacy
/// collections do not carry a `nodes` section for.
fn legacy_element_pair(
    et: ElementType,
    p: usize,
) -> Option<(Box<dyn fem_element::ReferenceElement>, Box<dyn fem_element::ReferenceElement>)> {
    use fem_element::lagrange::factory::{HexQk, QuadQk, H1TetPk};
    use fem_element::lagrange::legacy::LegacyHexQ3;
    use fem_element::lagrange::H1TriPk;
    match et {
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => Some((
            Box::new(QuadQk::new(p)),
            Box::new(QuadQk::new_closed_uniform(p)),
        )),
        // The hexahedron is the one geometry whose legacy element is not "the
        // H1 slot layout with equispaced nodes": `LagrangeHexFiniteElement`'s
        // hand-written table permutes the interior block (see
        // `fem_element::lagrange::legacy`), so it is reproduced verbatim.
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 if p == 3 => Some((
            Box::new(HexQk::new(p)),
            Box::new(LegacyHexQ3::new()),
        )),
        ElementType::Tri3 | ElementType::Tri6 => Some((
            Box::new(H1TriPk::new(p)),
            Box::new(H1TriPk::new_closed_uniform(p)),
        )),
        ElementType::Tet4 | ElementType::Tet10 => Some((
            Box::new(H1TetPk::new(p)),
            Box::new(H1TetPk::new_closed_uniform(p)),
        )),
        _ => None,
    }
}

/// `B[i][j] = ψ_j(ξ_i)`: the closed-uniform basis of the legacy element
/// evaluated at the Gauss-Lobatto nodes `ξ_i` of the element fem-rs actually
/// uses ([`legacy_element_pair`]'s first item).  Empty if the two elements do
/// not have the same DOF count.
fn legacy_change_of_basis(
    gll: &dyn fem_element::ReferenceElement,
    legacy: &dyn fem_element::ReferenceElement,
) -> Vec<f64> {
    let n = gll.n_dofs();
    if legacy.n_dofs() != n {
        return Vec::new();
    }
    let coords = gll.dof_coords();
    let mut b = vec![0.0_f64; n * n];
    for i in 0..n {
        legacy.eval_basis(&coords[i], &mut b[i * n..(i + 1) * n]);
    }
    b
}

/// D112: re-interpolate a `nodes` geometry table written with a legacy
/// (closed-uniform) collection onto the Gauss-Lobatto nodes the rest of the
/// library assumes.  See the block comment above for why this is exact.
///
/// Only the node *values* change; `conn`, `nodes_per_elem`, `order` and
/// `n_nodes` stay as they are.  Node values are shared between the elements
/// that meet at a mesh entity, and the legacy polynomial's trace on a shared
/// edge/face is the same from either side (as it must be for the file's
/// conforming H1 geometry), so the first element that reaches a node fixes its
/// value.
fn rewrite_legacy_nodes<M: MeshTopology>(
    mesh: &M,
    geom: &mut GeometryData,
    dim: usize,
) {
    let p = geom.order as usize;
    if p < 3 || geom.n_nodes == 0 || geom.nodes_per_elem == 0 || mesh.n_elements() == 0 {
        return; // p <= 2: closed-uniform and Gauss-Lobatto nodes coincide
    }
    type Pair = (
        Box<dyn fem_element::ReferenceElement>,
        Box<dyn fem_element::ReferenceElement>,
        Vec<f64>,
    );
    let npe = geom.nodes_per_elem;
    let mut cache: HashMap<ElementType, Option<Pair>> = HashMap::new();
    let mut rewritten = vec![0.0_f64; geom.coords.len()];
    let mut assigned = vec![false; geom.n_nodes];
    for e in 0..mesh.n_elements() {
        let et = mesh.element_type(e as u32);
        let entry = cache.entry(et).or_insert_with(|| {
            legacy_element_pair(et, p).map(|(gll, legacy)| {
                let b = legacy_change_of_basis(&*gll, &*legacy);
                (gll, legacy, b)
            })
        });
        let Some((_, _, b)) = entry else { continue };
        if b.len() != npe * npe {
            continue; // element type the legacy pair does not describe
        }
        let slots = &geom.conn[e * npe..(e + 1) * npe];
        for i in 0..npe {
            let node = slots[i] as usize;
            if assigned[node] {
                continue;
            }
            for d in 0..dim {
                let mut acc = 0.0;
                for j in 0..npe {
                    acc += b[i * npe + j] * geom.coords[slots[j] as usize * dim + d];
                }
                rewritten[node * dim + d] = acc;
            }
            assigned[node] = true;
        }
    }
    for n in 0..geom.n_nodes {
        if assigned[n] {
            geom.coords[n * dim..(n + 1) * dim]
                .copy_from_slice(&rewritten[n * dim..(n + 1) * dim]);
        }
    }
}

/// D112: apply [`rewrite_legacy_nodes`] to a freshly built geometry table when
/// the `nodes` section came from a legacy (closed-uniform) collection.
fn repair_legacy_geometry<const D: usize>(
    mesh: &mut Mesh<D>,
    h1_nodes: &Option<(u8, Vec<f64>, usize, bool)>,
) {
    if !matches!(h1_nodes, Some((_, _, _, true))) {
        return;
    }
    if let Some(mut g) = mesh.geometry.take() {
        rewrite_legacy_nodes(mesh, &mut g, D);
        mesh.geometry = Some(g);
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

/// D589: pop the first carried-over token of the `vertices` header and parse
/// it as `usize` (MFEM's token rule lets `<NV>` / `<spaceDim>` / `nodes`
/// straddle or share lines; `mesh/mesh_readers.cpp:100-118`).
fn take_tail_uint(tail: &mut Vec<String>) -> FemResult<usize> {
    let t = tail.remove(0);
    t.parse().map_err(|_| FemError::Mesh(format!("MFEM: expected integer, got: {t}")))
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
            // MFEM `ReadInlineMesh` → `Make2D(nx, ny, TRIANGLE, sx, sy, true,
            // true)`, and `Make2D`'s *triangle* branch is **not** the quad one:
            // it ignores `sfc_ordering` entirely (only the quadrilateral branch
            // calls `NCMesh::GridSfcOrdering2D`) and numbers the elements
            // row-major, splitting each cell of the structured grid along its
            // main diagonal (v0,v2):
            //   elem[2k]   = { i + j*m,   i+1 + (j+1)*m,  i + (j+1)*m   }  (v0,v2,v3)
            //   elem[2k+1] = { i + j*m,   i+1 + j*m,      i+1 + (j+1)*m }  (v0,v1,v2)
            // with `m = nx+1`.  The previous `Mesh::unit_square_tri(max(nx,ny))`
            // split every cell along the *anti*-diagonal and disregarded the
            // nx != ny case, so both the element order and each element's local
            // vertex order differed from MFEM (`data/inline-tri.mesh`: MFEM
            // elem 0 = {0,6,5}, unit_square_tri gave {0,1,5}).  The target P3
            // node set of `gslib_field_interp` depends on it, so the written
            // `interpolated.gf` differed from the C++ miniapp.
            let nxu = nx as u32;
            let nyu = ny as u32;
            let m = nxu + 1;
            let nxv = nx + 1;
            let nyv = ny + 1;
            let mut coords = Vec::with_capacity(nxv * nyv * 2);
            for j in 0..nyv {
                for i in 0..nxv {
                    coords.push(i as f64 / nx as f64 * sx);
                    coords.push(j as f64 / ny as f64 * sy);
                }
            }
            let id = |x: u32, y: u32| y * m + x;
            let mut conn = Vec::with_capacity(2 * nx * ny * 3);
            let mut elem_tags = Vec::with_capacity(2 * nx * ny);
            for j in 0..nyu {
                for i in 0..nxu {
                    conn.extend([id(i, j), id(i + 1, j + 1), id(i, j + 1)]);
                    elem_tags.push(1);
                    conn.extend([id(i, j), id(i + 1, j), id(i + 1, j + 1)]);
                    elem_tags.push(1);
                }
            }
            // Boundary segments — the same `Make2D` layout as the quad branch
            // (the tri branch has the identical bdr code): bottom attr 1, top
            // attr 3, left attr 4, right attr 2, with `m = (nx+1)*ny` for the
            // top/bottom rows and `m = nx+1` for the left/right columns.
            let mut face_conn = Vec::with_capacity(2 * (nx + ny) * 2);
            let mut face_tags = Vec::with_capacity(2 * (nx + ny));
            for i in 0..nxu {
                face_conn.extend([id(i, 0), id(i + 1, 0)]);
                face_tags.push(1);
            }
            let m_top = (nxu + 1) * nyu;
            for i in 0..nxu {
                face_conn.extend([m_top + i + 1, m_top + i]);
                face_tags.push(3);
            }
            for j in 0..nyu {
                face_conn.extend([(j + 1) * m, j * m]);
                face_tags.push(4);
            }
            for j in 0..nyu {
                face_conn.extend([j * m + nxu, (j + 1) * m + nxu]);
                face_tags.push(2);
            }
            let mut mesh = Mesh::uniform(
                coords, conn, elem_tags, ElementType::Tri3,
                face_conn, face_tags, ElementType::Line2,
            );
            // `Mesh(mesh_file, 1, 1)` finalizes a 2-D triangle mesh through
            // `FinalizeTriMesh(..., refine = 1, ...)` →
            // `MarkTriMeshForRefinement()` (`mesh/mesh.cpp:2588`): every
            // triangle is rotated so that its longest edge is (v0,v1).  That
            // permutes each element's *local* vertex order (and with it the H1
            // dof numbering of any space built on the mesh), so it is part of
            // being 1:1 with MFEM and is applied here too.
            fem_mesh::amr::mark_tri_mesh_for_refinement(&mut mesh);
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
            // front, bdr attribute 2 — MFEM Make3D loops `for (x) for (z)`.
            for x in 0..nx as i32 {
                for z in 0..nz as i32 {
                    quad([vtx(x, 0, z), vtx(x + 1, 0, z), vtx(x + 1, 0, z + 1), vtx(x, 0, z + 1)], 2);
                }
            }
            // back, bdr attribute 4 — same `for (x) for (z)` nesting.
            for x in 0..nx as i32 {
                for z in 0..nz as i32 {
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
            // MFEM ReadInlineMesh → Make3D(nx,ny,nz, TETRAHEDRON, sx,sy,sz,
            // true): row-major cells (the SFC ordering applies to HEXAHEDRON
            // only), each hex split by `AddHexAsTets`' hex_to_tet table in
            // element-local vertex order (mesh.cpp:2243).  `Mesh::Load`
            // = Loader + Finalize(refine=1, fix_orientation=true)
            // (mesh.hpp:823), so the tet table is rotated afterwards by
            // MarkTetMeshForRefinement — mirrored below.
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
            let id = |x: usize, y: usize, z: usize| ((z * nyv + y) * nxv + x) as u32;
            // AddHexAsTets (element-LOCAL indices 0..7 of the cell's hex).
            const HEX_TO_TET: [[usize; 4]; 6] = [
                [0, 1, 2, 6],
                [0, 5, 1, 6],
                [0, 4, 5, 6],
                [0, 2, 3, 6],
                [0, 3, 7, 6],
                [0, 7, 4, 6],
            ];
            let mut conn = Vec::with_capacity(nx * ny * nz * 6 * 4);
            let mut elem_tags = Vec::with_capacity(nx * ny * nz * 6);
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let vi = [
                            id(i, j, k),
                            id(i + 1, j, k),
                            id(i + 1, j + 1, k),
                            id(i, j + 1, k),
                            id(i, j, k + 1),
                            id(i + 1, j, k + 1),
                            id(i + 1, j + 1, k + 1),
                            id(i, j + 1, k + 1),
                        ];
                        for t in HEX_TO_TET {
                            conn.extend(t.map(|v| vi[v]));
                            elem_tags.push(1);
                        }
                    }
                }
            }
            // Boundary: Make3D's six loops, each quad split by
            // `AddBdrQuadAsTriangles` = {(0,1,2), (0,2,3)} (mesh.cpp:2510).
            let mut face_conn = Vec::with_capacity(2 * 4 * (nx * ny + ny * nz + nx * nz) * 3);
            let mut face_tags = Vec::with_capacity(2 * 4 * (nx * ny + ny * nz + nx * nz));
            let mut bdr_tri = |q: [u32; 4], tag: i32| {
                face_conn.extend([q[0], q[1], q[2]]);
                face_conn.extend([q[0], q[2], q[3]]);
                face_tags.push(tag);
                face_tags.push(tag);
            };
            for j in 0..ny {
                for i in 0..nx {
                    bdr_tri([id(i, j, 0), id(i, j + 1, 0), id(i + 1, j + 1, 0), id(i + 1, j, 0)], 1);
                }
            }
            for j in 0..ny {
                for i in 0..nx {
                    bdr_tri([id(i, j, nz), id(i + 1, j, nz), id(i + 1, j + 1, nz), id(i, j + 1, nz)], 6);
                }
            }
            for k in 0..nz {
                for j in 0..ny {
                    bdr_tri([id(0, j, k), id(0, j, k + 1), id(0, j + 1, k + 1), id(0, j + 1, k)], 5);
                }
            }
            for k in 0..nz {
                for j in 0..ny {
                    bdr_tri([id(nx, j, k), id(nx, j + 1, k), id(nx, j + 1, k + 1), id(nx, j, k + 1)], 3);
                }
            }
            for i in 0..nx {
                for k in 0..nz {
                    bdr_tri([id(i, 0, k), id(i + 1, 0, k), id(i + 1, 0, k + 1), id(i, 0, k + 1)], 2);
                }
            }
            for i in 0..nx {
                for k in 0..nz {
                    bdr_tri([id(i, ny, k), id(i, ny, k + 1), id(i + 1, ny, k + 1), id(i + 1, ny, k)], 4);
                }
            }
            let mut mesh = Mesh::<3>::uniform(
                coords, conn, elem_tags, ElementType::Tet4,
                face_conn, face_tags, ElementType::Tri3,
            );
            // `Mesh::Load` = Loader + Finalize(refine=1): the tet table is
            // rotated by MarkTetMeshForRefinement (longest edge → (v0,v1)),
            // exactly like the `tri` arm's marking above (D154).
            fem_mesh::mark_tet_mesh_for_refinement(&mut mesh);
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
            // Boundary: Make3D's six loops in Make3D order.  Only the bottom
            // (attr 1) and top (attr 6) quads are split into triangles for
            // WEDGE (`AddBdrQuadAsTriangles`); the four side faces remain
            // quadrilaterals (`AddBdrQuad`) — a MIXED boundary, so the mesh
            // carries per-face types + CSR offsets like the .mesh reader.
            let mut face_conn = Vec::new();
            let mut face_tags = Vec::new();
            let mut face_kinds: Vec<ElementType> = Vec::new();
            let mut bdr = |quad: [u32; 4], tag: i32, split: bool| {
                face_conn.extend([quad[0], quad[1], quad[2]]);
                face_tags.push(tag);
                if split {
                    face_kinds.push(ElementType::Tri3);
                    face_conn.extend([quad[0], quad[2], quad[3]]);
                    face_tags.push(tag);
                    face_kinds.push(ElementType::Tri3);
                } else {
                    face_kinds.push(ElementType::Quad4);
                    face_conn.push(quad[3]);
                }
            };
            for j in 0..ny {
                for i in 0..nx {
                    bdr([id(i, j, 0), id(i, j + 1, 0), id(i + 1, j + 1, 0), id(i + 1, j, 0)], 1, true);
                }
            }
            for j in 0..ny {
                for i in 0..nx {
                    bdr([id(i, j, nz), id(i + 1, j, nz), id(i + 1, j + 1, nz), id(i, j + 1, nz)], 6, true);
                }
            }
            for k in 0..nz {
                for j in 0..ny {
                    bdr([id(0, j, k), id(0, j, k + 1), id(0, j + 1, k + 1), id(0, j + 1, k)], 5, false);
                }
            }
            for k in 0..nz {
                for j in 0..ny {
                    bdr([id(nx, j, k), id(nx, j + 1, k), id(nx, j + 1, k + 1), id(nx, j, k + 1)], 3, false);
                }
            }
            for i in 0..nx {
                for k in 0..nz {
                    bdr([id(i, 0, k), id(i + 1, 0, k), id(i + 1, 0, k + 1), id(i, 0, k + 1)], 2, false);
                }
            }
            for i in 0..nx {
                for k in 0..nz {
                    bdr([id(i, ny, k), id(i, ny, k + 1), id(i + 1, ny, k + 1), id(i + 1, ny, k)], 4, false);
                }
            }
            let mut mesh = Mesh::uniform(
                coords, conn, elem_tags, ElementType::Prism6,
                face_conn, face_tags, ElementType::Tri3,
            );
            // Mixed tri/quad boundary: per-face types + CSR offsets (same
            // representation the .mesh reader uses).
            let mut offs = Vec::with_capacity(face_kinds.len() + 1);
            offs.push(0);
            for t in &face_kinds {
                offs.push(offs.last().unwrap() + t.nodes_per_element());
            }
            mesh.face_types = Some(face_kinds);
            mesh.face_offsets = Some(offs);
            Ok(MfemFile { mesh2d: None, mesh3d: Some(mesh) })
        }
        "pyramid" => {
            // MFEM ReadInlineMesh → Make3D(nx,ny,nz, PYRAMID, ...): row-major
            // cells, each hex split by `AddHexAsPyramids` (mesh.cpp:2280) into
            // 6 pyramids with a NEW apex vertex at the cell centre
            // (`VTXP(x,y,z)` = grid vertices + (z*ny+y)*nx+x, coords at the
            // cell centre, mesh.cpp:3882).  The boundary stays quadrilateral
            // (`AddBdrQuad`), in Make3D's six-loop order.
            let nz = _nz.unwrap_or(1);
            let sz = _sz.unwrap_or(1.0);
            let (nxv, nyv, nzv) = (nx + 1, ny + 1, nz + 1);
            let mut coords = Vec::with_capacity((nxv * nyv * nzv + nx * ny * nz) * 3);
            for k in 0..nzv {
                for j in 0..nyv {
                    for i in 0..nxv {
                        coords.push(i as f64 / nx as f64 * sx);
                        coords.push(j as f64 / ny as f64 * sy);
                        coords.push(k as f64 / nz as f64 * sz);
                    }
                }
            }
            // Cell-centre apex vertices, Make3D vertex-loop order (z, y, x).
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        coords.push((i as f64 + 0.5) / nx as f64 * sx);
                        coords.push((j as f64 + 0.5) / ny as f64 * sy);
                        coords.push((k as f64 + 0.5) / nz as f64 * sz);
                    }
                }
            }
            let id = |x: usize, y: usize, z: usize| ((z * nyv + y) * nxv + x) as u32;
            let apex = |x: usize, y: usize, z: usize| {
                (nxv * nyv * nzv + (z * ny + y) * nx + x) as u32
            };
            // AddHexAsPyramids (element-LOCAL hex indices 0..7 + apex 8).
            const HEX_TO_PYR: [[usize; 5]; 6] = [
                [0, 1, 2, 3, 8],
                [0, 4, 5, 1, 8],
                [1, 5, 6, 2, 8],
                [2, 6, 7, 3, 8],
                [3, 7, 4, 0, 8],
                [7, 6, 5, 4, 8],
            ];
            let mut conn = Vec::with_capacity(nx * ny * nz * 6 * 5);
            let mut elem_tags = Vec::with_capacity(nx * ny * nz * 6);
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let vi = [
                            id(i, j, k),
                            id(i + 1, j, k),
                            id(i + 1, j + 1, k),
                            id(i, j + 1, k),
                            id(i, j, k + 1),
                            id(i + 1, j, k + 1),
                            id(i + 1, j + 1, k + 1),
                            id(i, j + 1, k + 1),
                            apex(i, j, k),
                        ];
                        for p in HEX_TO_PYR {
                            conn.extend(p.map(|v| vi[v]));
                            elem_tags.push(1);
                        }
                    }
                }
            }
            // Boundary: Make3D's six loops, unsplit quads (`AddBdrQuad`).
            let mut face_conn = Vec::with_capacity(2 * 4 * (nx * ny + ny * nz + nx * nz));
            let mut face_tags = Vec::with_capacity(2 * 4 * (nx * ny + ny * nz + nx * nz));
            let mut quad = |f: [u32; 4], tag: i32| { face_conn.extend_from_slice(&f); face_tags.push(tag); };
            for j in 0..ny {
                for i in 0..nx {
                    quad([id(i, j, 0), id(i, j + 1, 0), id(i + 1, j + 1, 0), id(i + 1, j, 0)], 1);
                }
            }
            for j in 0..ny {
                for i in 0..nx {
                    quad([id(i, j, nz), id(i + 1, j, nz), id(i + 1, j + 1, nz), id(i, j + 1, nz)], 6);
                }
            }
            for k in 0..nz {
                for j in 0..ny {
                    quad([id(0, j, k), id(0, j, k + 1), id(0, j + 1, k + 1), id(0, j + 1, k)], 5);
                }
            }
            for k in 0..nz {
                for j in 0..ny {
                    quad([id(nx, j, k), id(nx, j + 1, k), id(nx, j + 1, k + 1), id(nx, j, k + 1)], 3);
                }
            }
            for i in 0..nx {
                for k in 0..nz {
                    quad([id(i, 0, k), id(i + 1, 0, k), id(i + 1, 0, k + 1), id(i, 0, k + 1)], 2);
                }
            }
            for i in 0..nx {
                for k in 0..nz {
                    quad([id(i, ny, k), id(i, ny, k + 1), id(i + 1, ny, k + 1), id(i + 1, ny, k)], 4);
                }
            }
            let mesh = Mesh::<3>::uniform(
                coords, conn, elem_tags, ElementType::Pyramid5,
                face_conn, face_tags, ElementType::Quad4,
            );
            Ok(MfemFile { mesh2d: None, mesh3d: Some(mesh) })
        }
        other => Err(FemError::Mesh(format!(
            "INLINE mesh: unsupported type '{other}' (supported: tri, quad, hex, tet, wedge, pyramid; \
             'segment' needs a 1-D MfemFile container, see D602 report)"
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

/// Parsed contents of an MFEM `.gf` file in the native `FiniteElementSpace`
/// format written by `GridFunction::Save` (`fem/gridfunc.cpp`).
#[derive(Debug, Clone)]
pub struct MfemGf {
    /// `FiniteElementCollection` name, e.g. `"ND_3D_P2"`.
    pub collection: String,
    /// `VDim` header line.
    pub vdim: usize,
    /// `Ordering` header line (`0` = byNODES, `vdof = dof + ndofs*vd` —
    /// the only ordering fem-rs spaces use / write).
    pub ordering: i32,
    /// Coefficient values in the file's own (byNODES) layout.
    ///
    /// D602: for vector `ND`/`RT` spaces on shared triangular faces this is
    /// **already fem-rs's canonical convention** — MFEM pins each face's
    /// storage frame to its face-creating element
    /// (`Mesh::AddTriangleFaceElement`: `Elem1Inf = 64*lf`, "orientation 0",
    /// i.e. `Fo(Elem1) = 0` and `T(0) = I`), which is exactly the element
    /// fem-rs anchors the canonical functionals to.  Verified 74/74 against
    /// MFEM `ProjectCoefficient` output on the D559 tet ND2 probe
    /// (`tmp/d602/`); no per-face transform is applied when reading.
    pub values: Vec<f64>,
}

/// Read an MFEM `.gf` file in the native `FiniteElementSpace` format:
///
/// ```text
/// FiniteElementSpace
/// FiniteElementCollection: ND_3D_P2
/// VDim: 1
/// Ordering: 0
///
/// <value 1>
/// ...
/// ```
///
/// Everything after the `Ordering` line is parsed as whitespace-separated
/// `f64` values (the `Vector::Load` rule), so wrapped/aligned layouts parse
/// the same as MFEM's one-value-per-line output.  The legacy
/// "MFEM grid function v1.0" container written by [`write_gf`] is *not*
/// this format and is rejected.
pub fn read_mfem_gf<R: Read>(reader: R) -> FemResult<MfemGf> {
    let mut r = BufReader::new(reader);
    let head = read_line(&mut r)?;
    if head.trim() != "FiniteElementSpace" {
        return Err(FemError::Mesh(format!(
            "MFEM gf: expected 'FiniteElementSpace' header, got: {head}"
        )));
    }
    let coll = read_line(&mut r)?;
    let collection = coll
        .strip_prefix("FiniteElementCollection:")
        .map(str::trim)
        .ok_or_else(|| FemError::Mesh(format!("MFEM gf: missing collection line: {coll}")))?
        .to_string();
    let vdim_line = read_line(&mut r)?;
    let vdim = vdim_line
        .strip_prefix("VDim:")
        .and_then(|t| t.trim().parse().ok())
        .ok_or_else(|| FemError::Mesh(format!("MFEM gf: bad VDim line: {vdim_line}")))?;
    let ord_line = read_line(&mut r)?;
    let ordering = ord_line
        .strip_prefix("Ordering:")
        .and_then(|t| t.trim().parse().ok())
        .ok_or_else(|| FemError::Mesh(format!("MFEM gf: bad Ordering line: {ord_line}")))?;
    let mut rest = String::new();
    r.read_to_string(&mut rest)?;
    let mut values = Vec::new();
    for tok in rest.split_whitespace() {
        let v: f64 = tok
            .parse()
            .map_err(|_| FemError::Mesh(format!("MFEM gf: bad value token: {tok}")))?;
        values.push(v);
    }
    Ok(MfemGf { collection, vdim, ordering, values })
}

/// Convenience: read an MFEM `.gf` file from disk.
pub fn read_mfem_gf_file(path: impl AsRef<std::path::Path>) -> FemResult<MfemGf> {
    let file = std::fs::File::open(path)?;
    read_mfem_gf(std::io::BufReader::new(file))
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
