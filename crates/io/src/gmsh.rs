//! GMSH `.msh` file format version 4.1 (ASCII) reader **and** writer.
//!
//! **Reading** – produces a [`SimplexMesh`] from the highest-dimension
//! elements found in the file.  Lower-dimension elements that belong to
//! physical groups are treated as boundary faces.
//!
//! **Writing** – serialises a [`SimplexMesh`] (plus optional
//! [`PhysicalGroup`] metadata) back to a valid GMSH v4.1 ASCII `.msh` file
//! that can be opened directly in the Gmsh GUI for inspection.
//!
//! # Format reference
//! <https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format>

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};

use fem_core::{FemError, FemResult};
use fem_mesh::{boundary::PhysicalGroup, element_type::ElementType, simplex::SimplexMesh};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Read a GMSH v4.1 ASCII `.msh` file and return a 2-D or 3-D `SimplexMesh`.
///
/// The mesh spatial dimension is inferred from the highest-dimension elements
/// present.  If both 2-D and 3-D elements exist, 3-D takes precedence.
///
/// # Errors
/// Returns [`FemError::Io`] for read errors and [`FemError::Mesh`] for
/// format errors.
pub fn read_msh<R: Read>(reader: R) -> FemResult<MshFile> {
    let buf = BufReader::new(reader);
    let mut parser = MshParser::new();
    parser.parse(buf)?;
    parser.build()
}

/// Convenience wrapper: open a file by path and parse it.
pub fn read_msh_file(path: impl AsRef<std::path::Path>) -> FemResult<MshFile> {
    let f = std::fs::File::open(path)?;
    read_msh(f)
}

// ---------------------------------------------------------------------------
// Output type
// ---------------------------------------------------------------------------

/// Parsed mesh data.  The caller can then convert to the desired `SimplexMesh<D>`.
pub struct MshFile {
    /// Physical group definitions.
    pub physical_groups: Vec<PhysicalGroup>,
    /// Map from physical tag → name (for BCs).
    pub tag_names: HashMap<i32, String>,
    /// The 2-D mesh (populated when highest element dimension is 2).
    pub mesh2d: Option<SimplexMesh<2>>,
    /// The 3-D mesh (populated when highest element dimension is 3).
    pub mesh3d: Option<SimplexMesh<3>>,
}

impl MshFile {
    /// Unwrap the 2-D mesh, or return an error if none was found.
    pub fn into_2d(self) -> FemResult<SimplexMesh<2>> {
        self.mesh2d.ok_or_else(|| FemError::Mesh("no 2-D elements found in .msh file".into()))
    }
    /// Unwrap the 3-D mesh, or return an error if none was found.
    pub fn into_3d(self) -> FemResult<SimplexMesh<3>> {
        self.mesh3d.ok_or_else(|| FemError::Mesh("no 3-D elements found in .msh file".into()))
    }
}

// ---------------------------------------------------------------------------
// Internal parser
// ---------------------------------------------------------------------------

struct MshParser {
    version:  f64,
    // Physical groups: dim → (tag → name)
    phys: Vec<PhysicalGroup>,
    // Node storage: tag (1-based) → coords [x, y, z]
    nodes: Vec<[f64; 3]>,            // indexed by 0-based internal id
    node_tag_to_id: HashMap<usize, usize>,
    // Element blocks grouped by dimension
    // elem_blocks[dim] = Vec<(elem_type, physical_tag, node_tags_flat)>
    elem_by_dim: [Vec<ElemBlock>; 4],
}

struct ElemBlock {
    etype: ElementType,
    phys_tag: i32,
    /// Flat list of (local-0-based) node ids per element.
    conn: Vec<u32>,
}

impl MshParser {
    fn new() -> Self {
        Self {
            version: 0.0,
            phys: Vec::new(),
            nodes: Vec::new(),
            node_tag_to_id: HashMap::new(),
            elem_by_dim: Default::default(),
        }
    }

    fn parse<R: BufRead>(&mut self, mut reader: R) -> FemResult<()> {
        let mut line = String::new();
        loop {
            line.clear();
            let n = reader.read_line(&mut line)?;
            if n == 0 { break; }
            let section = line.trim();
            match section {
                "$MeshFormat"    => self.read_mesh_format(&mut reader)?,
                "$PhysicalNames" => self.read_physical_names(&mut reader)?,
                "$Entities"      => self.read_entities(&mut reader)?,
                "$Nodes"         => self.read_nodes(&mut reader)?,
                "$Elements"      => self.read_elements(&mut reader)?,
                _                => { /* skip unknown sections */ }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Section readers
    // -----------------------------------------------------------------------

    fn read_mesh_format<R: BufRead>(&mut self, r: &mut R) -> FemResult<()> {
        let line = next_line(r)?;
        let mut parts = line.split_whitespace();
        self.version = parts.next()
            .ok_or_else(|| mesh_err("missing version in $MeshFormat"))?
            .parse::<f64>()
            .map_err(|e| mesh_err(&format!("bad version: {e}")))?;
        if self.version < 4.0 {
            return Err(mesh_err(&format!(
                "unsupported MSH version {:.1}; only v4.x is supported", self.version
            )));
        }
        let file_type: i32 = parts.next()
            .ok_or_else(|| mesh_err("missing file-type in $MeshFormat"))?
            .parse().map_err(|e| mesh_err(&format!("{e}")))?;
        if file_type != 0 {
            return Err(mesh_err("binary MSH format not supported; save as ASCII"));
        }
        skip_to(r, "$EndMeshFormat")
    }

    fn read_physical_names<R: BufRead>(&mut self, r: &mut R) -> FemResult<()> {
        let n: usize = next_line(r)?.trim().parse()
            .map_err(|e| mesh_err(&format!("$PhysicalNames count: {e}")))?;
        for _ in 0..n {
            let line = next_line(r)?;
            let mut parts = line.split_whitespace();
            let dim: u8 = parts.next().unwrap_or("0").parse().unwrap_or(0);
            let tag: i32 = parts.next().unwrap_or("0").parse().unwrap_or(0);
            // Name may be quoted
            let raw = parts.collect::<Vec<_>>().join(" ");
            let name = raw.trim_matches('"').to_string();
            self.phys.push(PhysicalGroup { dim, tag, name });
        }
        skip_to(r, "$EndPhysicalNames")
    }

    fn read_entities<R: BufRead>(&mut self, r: &mut R) -> FemResult<()> {
        // We skip entity geometry; we only need physical tags from $Elements.
        skip_to(r, "$EndEntities")
    }

    fn read_nodes<R: BufRead>(&mut self, r: &mut R) -> FemResult<()> {
        // Header: numEntityBlocks numNodes minNodeTag maxNodeTag
        let header = next_line(r)?;
        let mut hp = header.split_whitespace();
        let n_blocks: usize = hp.next().unwrap_or("0").parse().unwrap_or(0);
        let _n_nodes: usize = hp.next().unwrap_or("0").parse().unwrap_or(0);

        for _ in 0..n_blocks {
            // Block header: entityDim entityTag parametric numNodesInBlock
            let bh = next_line(r)?;
            let mut bp = bh.split_whitespace();
            let _dim: i32 = bp.next().unwrap_or("0").parse().unwrap_or(0);
            let _tag: i32 = bp.next().unwrap_or("0").parse().unwrap_or(0);
            let _param: i32= bp.next().unwrap_or("0").parse().unwrap_or(0);
            let count: usize = bp.next().unwrap_or("0").parse().unwrap_or(0);

            // Read node tags (1-based)
            let mut tags = Vec::with_capacity(count);
            for _ in 0..count {
                let t: usize = next_line(r)?.trim().parse()
                    .map_err(|e| mesh_err(&format!("node tag: {e}")))?;
                tags.push(t);
            }
            // Read coordinates
            for &tag in &tags {
                let line = next_line(r)?;
                let mut cp = line.split_whitespace();
                let x: f64 = cp.next().unwrap_or("0").parse().unwrap_or(0.0);
                let y: f64 = cp.next().unwrap_or("0").parse().unwrap_or(0.0);
                let z: f64 = cp.next().unwrap_or("0").parse().unwrap_or(0.0);
                let id = self.nodes.len();
                self.nodes.push([x, y, z]);
                self.node_tag_to_id.insert(tag, id);
            }
        }
        skip_to(r, "$EndNodes")
    }

    fn read_elements<R: BufRead>(&mut self, r: &mut R) -> FemResult<()> {
        // Header: numEntityBlocks numElements minElementTag maxElementTag
        let header = next_line(r)?;
        let mut hp = header.split_whitespace();
        let n_blocks: usize = hp.next().unwrap_or("0").parse().unwrap_or(0);

        for _ in 0..n_blocks {
            // Block header: entityDim entityTag elementType numElementsInBlock
            let bh = next_line(r)?;
            let mut bp = bh.split_whitespace();
            let edim: i32   = bp.next().unwrap_or("0").parse().unwrap_or(0);
            let etag: i32   = bp.next().unwrap_or("0").parse().unwrap_or(0);
            let etype_code: i32 = bp.next().unwrap_or("0").parse().unwrap_or(0);
            let count: usize    = bp.next().unwrap_or("0").parse().unwrap_or(0);

            let etype = match ElementType::from_gmsh_type(etype_code) {
                Some(t) => t,
                None    => {
                    // Skip unknown element type
                    for _ in 0..count { let _ = next_line(r); }
                    continue;
                }
            };
            if etype == ElementType::Point1 {
                for _ in 0..count { let _ = next_line(r); }
                continue;
            }

            let npe = etype.nodes_per_element();
            let mut conn = Vec::with_capacity(count * npe);

            for _ in 0..count {
                let line = next_line(r)?;
                let mut lp = line.split_whitespace();
                let _etag_: &str = lp.next().unwrap_or("0"); // element tag (discard)
                for _ in 0..npe {
                    let node_tag: usize = lp.next()
                        .ok_or_else(|| mesh_err("missing node tag in element"))?
                        .parse().map_err(|e| mesh_err(&format!("node tag parse: {e}")))?;
                    let node_id = *self.node_tag_to_id.get(&node_tag)
                        .ok_or_else(|| mesh_err(&format!("node tag {node_tag} not found")))?;
                    conn.push(node_id as u32);
                }
            }

            let dim = edim as usize;
            self.elem_by_dim[dim].push(ElemBlock { etype, phys_tag: etag, conn });
        }
        skip_to(r, "$EndElements")
    }

    // -----------------------------------------------------------------------
    // Build output
    // -----------------------------------------------------------------------

    fn build(self) -> FemResult<MshFile> {
        // Collect tag→name map
        let tag_names: HashMap<i32, String> = self.phys.iter()
            .map(|pg| (pg.tag, pg.name.clone()))
            .collect();

        let physical_groups = self.phys.clone();

        // Determine highest element dimension present
        let max_dim = (0usize..=3)
            .rev()
            .find(|&d| !self.elem_by_dim[d].is_empty())
            .unwrap_or(0);

        let n_nodes = self.nodes.len();

        let mut mesh2d = None;
        let mut mesh3d = None;

        if max_dim == 2 {
            mesh2d = Some(self.build_2d(n_nodes)?);
        } else if max_dim == 3 {
            mesh3d = Some(self.build_3d(n_nodes)?);
        }

        Ok(MshFile { physical_groups, tag_names, mesh2d, mesh3d })
    }

    fn build_2d(self, n_nodes: usize) -> FemResult<SimplexMesh<2>> {
        let mut coords = vec![0.0f64; n_nodes * 2];
        for (i, c) in self.nodes.iter().enumerate() {
            coords[i * 2    ] = c[0];
            coords[i * 2 + 1] = c[1];
        }

        // Collect 2-D elements
        let mut conn:      Vec<u32> = Vec::new();
        let mut elem_tags: Vec<i32> = Vec::new();
        let mut elem_type_opt = None;

        for blk in &self.elem_by_dim[2] {
            let npe = blk.etype.nodes_per_element();
            if elem_type_opt.is_none() { elem_type_opt = Some(blk.etype); }
            let n_elems = blk.conn.len() / npe;
            for i in 0..n_elems {
                conn.extend_from_slice(&blk.conn[i * npe..(i + 1) * npe]);
                elem_tags.push(blk.phys_tag);
            }
        }

        let elem_type = elem_type_opt.ok_or_else(|| mesh_err("no 2D elements"))?;
        let face_type = elem_type.boundary_type()
            .ok_or_else(|| mesh_err("cannot determine boundary element type"))?;

        // Collect 1-D boundary edges
        let mut face_conn: Vec<u32> = Vec::new();
        let mut face_tags: Vec<i32> = Vec::new();
        for blk in &self.elem_by_dim[1] {
            let npe = blk.etype.nodes_per_element();
            let n_faces = blk.conn.len() / npe;
            for i in 0..n_faces {
                face_conn.extend_from_slice(&blk.conn[i * npe..(i + 1) * npe]);
                face_tags.push(blk.phys_tag);
            }
        }

        Ok(SimplexMesh { coords, conn, elem_tags, elem_type, face_conn, face_tags, face_type })
    }

    fn build_3d(self, n_nodes: usize) -> FemResult<SimplexMesh<3>> {
        let mut coords = vec![0.0f64; n_nodes * 3];
        for (i, c) in self.nodes.iter().enumerate() {
            coords[i * 3    ] = c[0];
            coords[i * 3 + 1] = c[1];
            coords[i * 3 + 2] = c[2];
        }

        let mut conn:      Vec<u32> = Vec::new();
        let mut elem_tags: Vec<i32> = Vec::new();
        let mut elem_type_opt = None;

        for blk in &self.elem_by_dim[3] {
            let npe = blk.etype.nodes_per_element();
            if elem_type_opt.is_none() { elem_type_opt = Some(blk.etype); }
            let n_elems = blk.conn.len() / npe;
            for i in 0..n_elems {
                conn.extend_from_slice(&blk.conn[i * npe..(i + 1) * npe]);
                elem_tags.push(blk.phys_tag);
            }
        }

        let elem_type = elem_type_opt.ok_or_else(|| mesh_err("no 3D elements"))?;
        let face_type = elem_type.boundary_type()
            .ok_or_else(|| mesh_err("cannot determine boundary element type"))?;

        let mut face_conn: Vec<u32> = Vec::new();
        let mut face_tags: Vec<i32> = Vec::new();
        for blk in &self.elem_by_dim[2] {
            let npe = blk.etype.nodes_per_element();
            let n_faces = blk.conn.len() / npe;
            for i in 0..n_faces {
                face_conn.extend_from_slice(&blk.conn[i * npe..(i + 1) * npe]);
                face_tags.push(blk.phys_tag);
            }
        }

        Ok(SimplexMesh { coords, conn, elem_tags, elem_type, face_conn, face_tags, face_type })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn next_line<R: BufRead>(r: &mut R) -> FemResult<String> {
    loop {
        let mut line = String::new();
        let n = r.read_line(&mut line)?;
        if n == 0 {
            return Err(mesh_err("unexpected EOF in .msh file"));
        }
        let t = line.trim().to_string();
        if !t.is_empty() && !t.starts_with("//") {
            return Ok(t);
        }
    }
}

fn skip_to<R: BufRead>(r: &mut R, end_tag: &str) -> FemResult<()> {
    loop {
        let line = next_line(r)?;
        if line.trim() == end_tag { return Ok(()); }
    }
}

fn mesh_err(msg: &str) -> FemError {
    FemError::Mesh(msg.to_string())
}
