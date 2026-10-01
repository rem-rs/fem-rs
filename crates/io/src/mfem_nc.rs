//! MFEM **non-conforming** mesh writer — `MFEM NC mesh v1.0`.
//!
//! This is the format `NCMesh::Print` (`mfem/mesh/ncmesh.cpp:6349`) emits for
//! a `Mesh` that carries an `NCMesh` — i.e. any mesh with hanging vertices
//! (MFEM converts a mesh built with `Mesh::AddVertexParents` into an NC mesh
//! in `Mesh::FinalizeTopology`, `mesh/mesh.cpp:3679`).  It is a genuinely
//! different container from the conforming `MFEM mesh v1.0` the crate's
//! [`crate::mfem::write_mfem`] writes: element records carry a `rank` and a
//! `ref_type` and reference NC *node* ids, hanging vertices travel in a
//! `vertex_parents` section (`NCMesh::PrintVertexParents`), and the geometry
//! is either a straight `coordinates` section (all NC nodes, node-id order,
//! `NCMesh::PrintCoordinates`) or a curved `nodes` GridFunction appended by
//! `Mesh::Printer` (`mesh/mesh.cpp:12493-12506`).
//!
//! Scope (D104 round, honest subset — exactly what the `polar-nc` miniapp
//! needs): 2-D records with `ref_type = 0` (a mesh built from scratch with
//! hanging vertices has *every* element a root: `NCMesh::NCMesh(const Mesh*)`
//! creates one root per mesh element, so `ref_type` is always 0 and there is
//! no `root_state` section — `ZeroRootStates()` keeps it out), serial
//! `rank = 0` (the `rank` section itself is serial-skipped, `ncmesh.cpp:6379`),
//! and either the `coordinates` or the `nodes` payload.  Full refinement-tree
//! NC meshes (`ref_type != 0`, `children`, anisotropic roots) are **not**
//! claimable from this writer and it refuses them rather than emitting
//! something MFEM would misread.
//!
//! Section order and separators byte-match `NCMesh::Print` +
//! `Mesh::Printer` (verified against the MFEM 4.10 `polar-nc` miniapp
//! products in `fem-rs/tmp/d104pnc/gold/`).

use std::io::Write;

use fem_core::{FemError, FemResult};

use crate::data_collection::format_g;

/// One record of the NC `elements` section: `rank attr geom ref_type nodes…`.
#[derive(Debug, Clone)]
pub struct NcElement {
    /// Owner rank (`NCMesh::Element::rank`); serial meshes are all 0.
    pub rank: i32,
    /// Element attribute.
    pub attr: i32,
    /// NC geometry code (`NCMesh`'s own numbering: SEGMENT=1, TRIANGLE=2,
    /// SQUARE=3, TETRAHEDRON=4, CUBE=5, PRISM=6, PYRAMID=7).
    pub geom: u8,
    /// Refinement bit-mask; 0 for every element of a from-scratch NC mesh.
    pub ref_type: u8,
    /// Node ids in the element's local vertex order.
    pub nodes: Vec<i32>,
}

/// One record of the NC `boundary` section: `attr geom nodes…`.
#[derive(Debug, Clone)]
pub struct NcBoundary {
    /// Boundary attribute.
    pub attr: i32,
    /// NC geometry code (2-D boundary segments: SEGMENT = 1).
    pub geom: u8,
    /// Node ids (`NCMesh::PrintBoundary` prints them in the *owning element's*
    /// local edge direction, not the direction the `Mesh`'s segment had).
    pub nodes: Vec<i32>,
}

/// The geometry payload printed after `vertex_parents`.
#[derive(Debug, Clone)]
pub enum NcGeometry {
    /// Straight mesh (`Mesh::Nodes == NULL`): `NCMesh::Print` closes the file
    /// with the `coordinates` section — **all** NC nodes (top-level vertices
    /// *and* hanging ones) in node-id order, `spaceDim` components per row.
    /// `coords` is the flat row-major table (`nv * space_dim` entries).
    Coordinates { space_dim: usize, coords: Vec<f64> },
    /// Curved mesh (`Mesh::Nodes != NULL`): `Mesh::Printer` swaps the NC
    /// coordinates out and appends a `nodes` GridFunction section instead
    /// (`mfem/mesh/mesh.cpp:12496-12503`).
    NodesGf {
        /// `FiniteElementCollection` name, e.g. `H1_2D_P2` — printed verbatim.
        collection: String,
        /// `VDim` of the saved GridFunction (mesh spaceDim for `nodes`).
        vdim: usize,
        /// `Ordering` of the saved GridFunction (MFEM's `SetCurvature` spaces
        /// are `byVDIM` = 1).
        ordering: u8,
        /// Values in the file's own (byVDIM) layout, `vdim` per dof row.
        values: Vec<f64>,
    },
}

/// A serial `MFEM NC mesh v1.0` document — the plain-data input of
/// [`write_nc_mesh`].  The caller (e.g. the `polar-nc` miniapp, which mirrors
/// the C++ generate → `FinalizeMesh` → `NCMesh` conversion pipeline) supplies
/// the sections in the order MFEM derives them.
#[derive(Debug, Clone)]
pub struct NcMeshV1 {
    /// Mesh dimension (`dimension` section).
    pub dim: i32,
    /// `elements` section, in file order.
    pub elements: Vec<NcElement>,
    /// `boundary` section, in file order (element order × local face order —
    /// `NCMesh::PrintBoundary`'s iteration; only refcount-1 faces appear).
    pub boundary: Vec<NcBoundary>,
    /// `vertex_parents` records `(child, p1, p2)`, in NC node-id order
    /// (`HashTable` iterates its `BlockArray` in allocation-id order, so
    /// MFEM's emission order is simply ascending child id).
    pub vertex_parents: Vec<[i32; 3]>,
    /// Trailing geometry payload (straight `coordinates` or curved `nodes`).
    pub geometry: NcGeometry,
    /// Stream precision the C++ harness had set on its `ofstream`
    /// (`polar-nc` uses `ofs.precision(8)`); renders every float as C++
    /// `operator<<(double)` — `%.*g` with this many significant digits.
    /// MFEM's own `Mesh::Save` default is 16.
    pub precision: usize,
}

impl NcMeshV1 {
    /// Sanity-check the parts this writer claims to encode (D104 subset).
    /// `NCMesh::Print` would happily write richer trees; we refuse them so a
    /// caller can't smuggle an under-described mesh past the writer.
    fn validate(&self) -> FemResult<()> {
        if !(1..=3).contains(&self.dim) {
            return Err(FemError::Mesh(format!(
                "MFEM NC: dimension {} out of range 1..=3",
                self.dim
            )));
        }
        if self.precision == 0 || self.precision > 17 {
            return Err(FemError::Mesh(format!(
                "MFEM NC: stream precision {} outside 1..=17",
                self.precision
            )));
        }
        for (i, el) in self.elements.iter().enumerate() {
            if el.ref_type != 0 {
                return Err(FemError::Mesh(format!(
                    "MFEM NC: element {i} has ref_type {} — refined NC trees are \
                     outside this writer's D104 subset (from-scratch NC meshes are \
                     all-roots: NCMesh::NCMesh(const Mesh*) makes every element a \
                     root with ref_type 0)",
                    el.ref_type
                )));
            }
            if el.rank != 0 {
                return Err(FemError::Mesh(format!(
                    "MFEM NC: element {i} has rank {} — serial NC writer (rank \
                     section is skipped in serial builds, ncmesh.cpp:6379)",
                    el.rank
                )));
            }
            let nfv = nc_geom_vertex_count(el.geom)
                .ok_or_else(|| FemError::Mesh(format!("MFEM NC: bad geom {}", el.geom)))?;
            if el.nodes.len() != nfv {
                return Err(FemError::Mesh(format!(
                    "MFEM NC: element {i} geom {} expects {nfv} nodes, got {}",
                    el.geom,
                    el.nodes.len()
                )));
            }
        }
        for (i, b) in self.boundary.iter().enumerate() {
            let nfv = nc_geom_vertex_count(b.geom)
                .ok_or_else(|| FemError::Mesh(format!("MFEM NC: bad bdr geom {}", b.geom)))?;
            if b.nodes.len() != nfv {
                return Err(FemError::Mesh(format!(
                    "MFEM NC: boundary {i} geom {} expects {nfv} nodes, got {}",
                    b.geom,
                    b.nodes.len()
                )));
            }
        }
        match &self.geometry {
            NcGeometry::Coordinates { space_dim, coords } => {
                let nv: usize = self
                    .elements
                    .iter()
                    .flat_map(|e| e.nodes.iter())
                    .copied()
                    .max()
                    .map(|m| m as usize + 1)
                    .unwrap_or(0);
                if coords.len() != nv * space_dim {
                    return Err(FemError::Mesh(format!(
                        "MFEM NC: coordinates table has {} entries, expected \
                         {nv} nodes × {space_dim} = {} (PrintCoordinates prints \
                         every node id the elements reference)",
                        coords.len(),
                        nv * space_dim
                    )));
                }
            }
            NcGeometry::NodesGf { vdim, ordering, values, .. } => {
                if *ordering > 1 {
                    return Err(FemError::Mesh(format!(
                        "MFEM NC: nodes Ordering {ordering} — only 0 (byNODES) / \
                         1 (byVDIM) exist"
                    )));
                }
                if values.len() % *vdim != 0 {
                    return Err(FemError::Mesh(format!(
                        "MFEM NC: nodes values len {} not a multiple of VDim {vdim}",
                        values.len()
                    )));
                }
            }
        }
        Ok(())
    }
}

/// `NCMesh`'s geometry-code → vertex-count table (the `MaxElemNodes` prefix of
/// every `elements` record; `NCMesh::Print` stops at the first negative node).
pub fn nc_geom_vertex_count(geom: u8) -> Option<usize> {
    match geom {
        1 => Some(2), // SEGMENT
        2 => Some(3), // TRIANGLE
        3 => Some(4), // SQUARE
        4 => Some(4), // TETRAHEDRON
        5 => Some(8), // CUBE
        6 => Some(6), // PRISM
        7 => Some(5), // PYRAMID
        _ => None,
    }
}

/// Write the document to `writer` byte-for-byte as MFEM 4.10's
/// `NCMesh::Print` + `Mesh::Printer` produce it for this subset.
pub fn write_nc_mesh<W: Write>(writer: &mut W, mesh: &NcMeshV1) -> FemResult<()> {
    mesh.validate()?;
    let prec = mesh.precision;

    // `os << "MFEM NC mesh v1.0\n\n"` (ncmesh.cpp:6362 — the v1.1 header only
    // appears for scaled NC meshes, which this subset does not emit).
    writer.write_all(b"MFEM NC mesh v1.0\n\n")?;

    writer.write_all(
        b"# NCMesh supported geometry types:\n\
          # SEGMENT     = 1\n\
          # TRIANGLE    = 2\n\
          # SQUARE      = 3\n\
          # TETRAHEDRON = 4\n\
          # CUBE        = 5\n\
          # PRISM       = 6\n\
          # PYRAMID     = 7\n",
    )?;

    writeln!(writer, "\ndimension\n{}", mesh.dim)?;

    // Serial: the `rank` section is skipped (`MyRank == 0`,
    // ncmesh.cpp:6379-6384), but every element record still leads with its
    // (0) rank.
    writeln!(
        writer,
        "\n# rank attr geom ref_type nodes/children\nelements\n{}",
        mesh.elements.len()
    )?;
    for el in &mesh.elements {
        write!(writer, "{} {} {} {}", el.rank, el.attr, el.geom, el.ref_type)?;
        for n in &el.nodes {
            write!(writer, " {n}")?;
        }
        writeln!(writer)?;
    }

    // `PrintBoundary(NULL)` counts first; the section only appears when
    // non-empty (ncmesh.cpp:6403-6410).
    if !mesh.boundary.is_empty() {
        writeln!(
            writer,
            "\n# attr geom nodes\nboundary\n{}",
            mesh.boundary.len()
        )?;
        for b in &mesh.boundary {
            write!(writer, "{} {}", b.attr, b.geom)?;
            for n in &b.nodes {
                write!(writer, " {n}")?;
            }
            writeln!(writer)?;
        }
    }

    // `PrintVertexParents(NULL)` counts hanging-vertex nodes; likewise
    // conditional (ncmesh.cpp:6412-6419).
    if !mesh.vertex_parents.is_empty() {
        writeln!(
            writer,
            "\n# vert_id p1 p2\nvertex_parents\n{}",
            mesh.vertex_parents.len()
        )?;
        for vp in &mesh.vertex_parents {
            writeln!(writer, "{} {} {}", vp[0], vp[1], vp[2])?;
        }
    }

    match &mesh.geometry {
        NcGeometry::Coordinates { space_dim, coords } => {
            // `os << "\n# top-level node coordinates\ncoordinates\n"` then
            // `PrintCoordinates`: count, spaceDim, rows (ncmesh.cpp:6438-6444,
            // 6301-6317).  Rows are `x[0] x[1] …` for the first `space_dim`
            // components of each stored (3-wide) coordinate slot — here the
            // caller's row-major `nv * space_dim` table.
            let nv = if *space_dim > 0 { coords.len() / space_dim } else { 0 };
            writeln!(writer, "\n# top-level node coordinates\ncoordinates")?;
            writeln!(writer, "{nv}")?;
            if nv == 0 {
                return Ok(());
            }
            writeln!(writer, "{space_dim}")?;
            for i in 0..nv {
                write!(writer, "{}", format_g(coords[i * space_dim], prec))?;
                for c in 1..*space_dim {
                    write!(writer, " {}", format_g(coords[i * space_dim + c], prec))?;
                }
                writeln!(writer)?;
            }
        }
        NcGeometry::NodesGf { collection, vdim, ordering, values } => {
            // `Mesh::Printer` (mesh.cpp:12500-12502): NC-specific comment (no
            // trailing newline of its own — the GridFunction tail's leading
            // `"\nnodes"` provides it) + the shared GridFunction tail.
            write!(writer, "\n# mesh curvature GridFunction")?;
            write_gf_section(writer, collection, *vdim, *ordering, values, prec)?;
        }
    }

    // `os << "\nmfem_mesh_end" << endl` (mesh.cpp:12505).
    writer.write_all(b"\nmfem_mesh_end\n")?;
    Ok(())
}

/// [`write_nc_mesh`] into a freshly created file.
pub fn write_nc_mesh_file(path: impl AsRef<std::path::Path>, mesh: &NcMeshV1) -> FemResult<()> {
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    write_nc_mesh(&mut writer, mesh)?;
    writer.flush()?;
    Ok(())
}

/// A plain **conforming** `MFEM mesh v1.0` document — the container
/// `Mesh::Printer` falls back to when a generated mesh carries **no**
/// `AddVertexParents` triples (`Mesh::FinalizeTopology` only promotes a mesh
/// to `NCMesh` when `tmp_vertex_parents` is non-empty, mesh.cpp:3679).  For
/// the `polar-nc` miniapp this is the degenerate `-n 1` run: one ring of
/// triangles, printed exactly like any straight-from-elements conforming mesh
/// (`mesh/mesh.cpp:12509-12587`, golden `tmp/d104pnc/gold/n1.mesh`).
///
/// This lives beside the NC writer because it is the NC *miniapp*'s fallback
/// branch, not a general conforming-mesh writer (the crate's general one is
/// [`crate::mfem::write_mfem`], driven by a [`fem_mesh::Mesh`]).
#[derive(Debug, Clone)]
pub struct ConformingMeshV1 {
    /// Mesh dimension (`dimension` section).
    pub dim: i32,
    /// `elements` records `(attr, geom, vertices)` in file order.  `geom`
    /// uses MFEM's *conforming* element codes — identical numbers to the NC
    /// codes for the supported set (TRIANGLE=2, SQUARE=3, …).
    pub elements: Vec<(i32, u8, Vec<i32>)>,
    /// `boundary` records `(attr, geom, vertices)` in file order, exactly as
    /// the segments were declared (no per-element re-derivation here — the
    /// conforming printer stores them verbatim).
    pub boundary: Vec<(i32, u8, Vec<i32>)>,
    /// Vertex count for the `vertices` section header.
    pub n_vertices: usize,
    /// Geometry payload — straight: coordinates per vertex (the `vertices`
    /// section carries them); curved: the `nodes` GridFunction appended after
    /// a bare vertex count.
    pub geometry: NcGeometry,
    /// Stream precision (as in [`NcMeshV1::precision`]).
    pub precision: usize,
}

/// Write the conforming document (`Mesh::Printer`'s non-NC branch).  No
/// `mfem_mesh_end` delimiter — that only exists for v1.2+ (attribute-set
/// names), and `Mesh::Print` on a plain v1.0 mesh ends after the last
/// section.
pub fn write_conforming_mesh_v1<W: Write>(writer: &mut W, mesh: &ConformingMeshV1) -> FemResult<()> {
    if !(1..=3).contains(&mesh.dim) {
        return Err(FemError::Mesh(format!(
            "MFEM conforming: dimension {} out of range 1..=3",
            mesh.dim
        )));
    }
    if mesh.precision == 0 || mesh.precision > 17 {
        return Err(FemError::Mesh(format!(
            "MFEM conforming: stream precision {} outside 1..=17",
            mesh.precision
        )));
    }

    writer.write_all(b"MFEM mesh v1.0\n\n")?;
    writer.write_all(
        b"#\n# MFEM Geometry Types (see fem/geom.hpp):\n#\n\
          # POINT       = 0\n\
          # SEGMENT     = 1\n\
          # TRIANGLE    = 2\n\
          # SQUARE      = 3\n\
          # TETRAHEDRON = 4\n\
          # CUBE        = 5\n\
          # PRISM       = 6\n\
          # PYRAMID     = 7\n#\n",
    )?;

    writeln!(writer, "\ndimension\n{}", mesh.dim)?;

    writeln!(writer, "\nelements\n{}", mesh.elements.len())?;
    for (attr, geom, verts) in &mesh.elements {
        write!(writer, "{attr} {geom}")?;
        for v in verts {
            write!(writer, " {v}")?;
        }
        writeln!(writer)?;
    }

    writeln!(writer, "\nboundary\n{}", mesh.boundary.len())?;
    for (attr, geom, verts) in &mesh.boundary {
        write!(writer, "{attr} {geom}")?;
        for v in verts {
            write!(writer, " {v}")?;
        }
        writeln!(writer)?;
    }

    match &mesh.geometry {
        NcGeometry::Coordinates { space_dim, coords } => {
            if coords.len() != mesh.n_vertices * space_dim {
                return Err(FemError::Mesh(format!(
                    "MFEM conforming: vertices table has {} entries, expected {} × {space_dim}",
                    coords.len(),
                    mesh.n_vertices
                )));
            }
            writeln!(writer, "\nvertices\n{}\n{}", mesh.n_vertices, space_dim)?;
            let prec = mesh.precision;
            for v in 0..mesh.n_vertices {
                write!(writer, "{}", format_g(coords[v * space_dim], prec))?;
                for c in 1..*space_dim {
                    write!(writer, " {}", format_g(coords[v * space_dim + c], prec))?;
                }
                writeln!(writer)?;
            }
        }
        NcGeometry::NodesGf { collection, vdim, ordering, values } => {
            // `os << "\nvertices\n" << NumOfVertices << '\n'` then the nodes
            // GridFunction (bare count first — the curved mesh's vertex
            // positions live in the field).
            writeln!(writer, "\nvertices\n{}", mesh.n_vertices)?;
            write_gf_section(writer, collection, *vdim, *ordering, values, mesh.precision)?;
        }
    }

    Ok(())
}

/// The shared `nodes` GridFunction tail (`GridFunction::Save`: `FES::Save` +
/// `'\n'` + `Vector::Print(os, VDim)`), used by both the NC and the
/// conforming writer.
fn write_gf_section<W: Write>(
    writer: &mut W,
    collection: &str,
    vdim: usize,
    ordering: u8,
    values: &[f64],
    prec: usize,
) -> FemResult<()> {
    if ordering > 1 {
        return Err(FemError::Mesh(format!(
            "MFEM nodes: Ordering {ordering} — only 0 (byNODES) / 1 (byVDIM) exist"
        )));
    }
    if values.len() % vdim != 0 {
        return Err(FemError::Mesh(format!(
            "MFEM nodes: values len {} not a multiple of VDim {vdim}",
            values.len()
        )));
    }
    writeln!(
        writer,
        "\nnodes\nFiniteElementSpace\nFiniteElementCollection: {collection}\n\
         VDim: {vdim}\nOrdering: {ordering}\n"
    )?;
    let ndofs = values.len() / vdim;
    for d in 0..ndofs {
        write!(writer, "{}", format_g(zero_subnormal_is_flunnable(values[d * vdim]), prec))?;
        for c in 1..vdim {
            write!(writer, " {}", format_g(zero_subnormal_is_flunnable(values[d * vdim + c]), prec))?;
        }
        writeln!(writer)?;
    }
    Ok(())
}

/// MFEM `Vector::Print`'s `ZeroSubnormal` flush (`linalg/vector.cpp:857`):
/// subnormal magnitudes print as `0`.  Re-exported behaviour twin of
/// `mfem::zero_subnormal` (kept private there); re-implemented here so this
/// module stays independent of that file's private helpers.
fn zero_subnormal_is_flunnable(v: f64) -> f64 {
    if v != 0.0 && v.abs() < f64::MIN_POSITIVE {
        0.0
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One center triangle + two hanging-split quads, straight (`coordinates`),
    /// pinned against `NCMesh::Print`'s separators.
    #[test]
    fn minimal_document_matches_ncmesh_print_bytes() {
        let mesh = NcMeshV1 {
            dim: 2,
            elements: vec![
                NcElement { rank: 0, attr: 1, geom: 2, ref_type: 0, nodes: vec![0, 1, 2] },
                NcElement { rank: 0, attr: 1, geom: 3, ref_type: 0, nodes: vec![3, 5, 6, 2] },
                NcElement { rank: 0, attr: 1, geom: 3, ref_type: 0, nodes: vec![1, 4, 5, 3] },
            ],
            boundary: vec![
                NcBoundary { attr: 1, geom: 1, nodes: vec![0, 1] },
                NcBoundary { attr: 2, geom: 1, nodes: vec![2, 0] },
                NcBoundary { attr: 2, geom: 1, nodes: vec![6, 3] },
                NcBoundary { attr: 1, geom: 1, nodes: vec![4, 5] },
            ],
            vertex_parents: vec![[3, 1, 2]],
            geometry: NcGeometry::Coordinates {
                space_dim: 2,
                coords: vec![
                    0.0, 0.0, //
                    0.1, 0.0, //
                    6.123233995736766e-18, 0.1, //
                    0.07071067811865476, 0.07071067811865474, //
                    0.2, 0.0, //
                    0.1414213562373095, 0.14142135623730948, //
                    1.2246467991473532e-17, 0.2, //
                ],
            },
            precision: 8,
        };
        let mut out = Vec::new();
        write_nc_mesh(&mut out, &mesh).unwrap();
        let text = String::from_utf8(out).unwrap();
        let expected = "\
MFEM NC mesh v1.0

# NCMesh supported geometry types:
# SEGMENT     = 1
# TRIANGLE    = 2
# SQUARE      = 3
# TETRAHEDRON = 4
# CUBE        = 5
# PRISM       = 6
# PYRAMID     = 7

dimension
2

# rank attr geom ref_type nodes/children
elements
3
0 1 2 0 0 1 2
0 1 3 0 3 5 6 2
0 1 3 0 1 4 5 3

# attr geom nodes
boundary
4
1 1 0 1
2 1 2 0
2 1 6 3
1 1 4 5

# vert_id p1 p2
vertex_parents
1
3 1 2

# top-level node coordinates
coordinates
7
2
0 0
0.1 0
6.123234e-18 0.1
0.070710678 0.070710678
0.2 0
0.14142136 0.14142136
1.2246468e-17 0.2

mfem_mesh_end
";
        assert_eq!(text, expected);
    }

    /// A refined-tree record (`ref_type != 0`) is refused, not mis-written.
    #[test]
    fn refined_tree_is_refused() {
        let mesh = NcMeshV1 {
            dim: 2,
            elements: vec![NcElement {
                rank: 0,
                attr: 1,
                geom: 3,
                ref_type: 7,
                nodes: vec![0, 1, 2, 3],
            }],
            boundary: vec![],
            vertex_parents: vec![],
            geometry: NcGeometry::Coordinates {
                space_dim: 2,
                coords: vec![0.0; 8],
            },
            precision: 8,
        };
        let err = write_nc_mesh(&mut Vec::new(), &mesh).unwrap_err();
        assert!(err.to_string().contains("ref_type"), "{err}");
    }
}
