//! Legacy (non-XML) VTK `.vtk` mesh reader — 1:1 port of MFEM's
//! `Mesh::ReadVTKMesh` + `Mesh::CreateVTKMesh` + `Mesh::Finalize` pipeline
//! (`mesh/mesh_readers.cpp:1177-1309` and `:397-680`,
//! `mesh/mesh.cpp:3766-3833`, MFEM 4.10).
//!
//! The reader reproduces the state a C++ `Mesh mesh(file, 1, 1)` constructor
//! produces, so that `write_mfem` output is byte-identical to C++
//! `Mesh::Print()`:
//!
//! 1. Parse `POINTS` / `CELLS` / `CELL_TYPES` / optional `CELL_DATA`
//!    `SCALARS material|attribute`; missing attributes default to 1 (MFEM's
//!    `Element` default).
//! 2. Build elements.  VTK vertex order equals MFEM's for every geometry
//!    *except prisms*, which go through `VTKGeometry::PrismMap`
//!    (`{0,2,1,3,5,4}`).
//! 3. Linear meshes: every VTK point becomes a vertex.  Quadratic meshes:
//!    only the corner points become vertices, renumbered by first appearance
//!    in the point array (`CreateVTKMesh`'s `pts_dof` loop); the remaining
//!    points become a second-order geometry table (the mesh's `Nodes`).
//! 4. First-encounter faces / boundary elements with attribute 1
//!    (`GenerateFaces` + `GenerateBoundaryElements`), the state
//!    `FinalizeTopology` leaves a VTK mesh in.
//! 5. `CheckElementOrientation(true)` (shared `Mesh` method), then the
//!    refinement marking of `Mesh::Finalize(refine = 1)`:
//!    * 3-D tetrahedra: [`fem_mesh::mark_tet_mesh_for_refinement`] (also
//!      rotates the triangular boundary cycles, like MFEM's boundary
//!      `MarkEdge`),
//!    * 2-D triangles: `Mesh::MarkTriMeshForRefinement` — a per-element
//!      longest-edge rotation ported from `mesh/triangle.cpp:53`
//!      (`Triangle::MarkEdge`).  MFEM builds the 2-D boundary *before* this
//!      marking and re-aligns each boundary segment with its owner's (possibly
//!      rotated) local edge afterwards, so the reader does the same.
//!    Hex / prism / pyramid meshes never mark (`meshgen` bit 1 unset).
//!
//! # The `Finalize(refine, fix_orientation)` knobs
//!
//! The `read_vtk_mesh_with` / `read_vtk_mesh_file_with` variants expose the
//! `Mesh mesh(file, refine, fix_orientation)` constructor overloads
//! (`mesh.hpp:823-827`, `Mesh::Finalize` at `mesh.cpp:3766`):
//!
//! * `refine = false` skips `MarkForRefinement()` — no longest-edge rotation
//!   of elements, boundary faces or geometry rows.
//! * `fix_orientation = false` turns `Finalize`'s `CheckElementOrientation`
//!   into a check-only pass.  This is the only orientation pass a *quadratic*
//!   mesh gets; *linear* meshes are oriented unconditionally inside
//!   `CreateVTKMesh` (`mesh_readers.cpp:488`), so for them the knob is moot.
//! * The trailing `CheckBdrElementOrientation()` is unconditional in MFEM;
//!   its 2-D realignment stays on in every combination (it is an identity
//!   unless an owner's vertex cycle changed).
//!
//! `(true, true)` — the [`read_vtk_mesh`] / [`read_vtk_mesh_file`] defaults —
//! is the historical (and byte-parity-pinned) behaviour.  The C++ trimmer's
//! `Mesh mesh(mesh_file, 0, 0)` is `(refine, fix_orientation) = (false, true)`
//! (the constructor's second parameter is `generate_edges`, `fix_orientation`
//! keeps its `true` default — `mesh.hpp:813`).
//!
//! # Supported cell types
//!
//! | VTK code | type                               | MFEM geometry  |
//! |----------|------------------------------------|----------------|
//! | 5        | `TRIANGLE`                         | triangle (o1)  |
//! | 9        | `QUAD`                             | square (o1)    |
//! | 10       | `TETRA`                            | tet (o1)       |
//! | 12       | `HEXAHEDRON`                       | cube (o1)      |
//! | 13       | `WEDGE`                            | prism (o1)     |
//! | 14       | `PYRAMID`                          | pyramid (o1)   |
//! | 22       | `QUADRATIC_TRIANGLE` (6 pts)       | triangle (o2)  |
//! | 24       | `QUADRATIC_TETRAHEDRON` (10 pts)   | tet (o2)       |
//! | 28       | `BIQUADRATIC_SQUARE` (9 pts)       | square (o2)    |
//! | 29       | `TRIQUADRATIC_HEXAHEDRON` (27 pts) | cube (o2)      |
//! | 32       | `BIQUADRATIC_QUADRATIC_PRISM` (18) | prism (o2)     |
//!
//! Anything else is rejected, listing the offending type number(s).  The
//! quadratic *serendipity* types 23 (`QUADRATIC_QUAD`, 8 pts), 25
//! (`QUADRATIC_HEXAHEDRON`, 20 pts) and 34 (`BIQUADRATIC_TRIANGLE`, 10 pts)
//! are rejected because MFEM's legacy quadratic branch fills an entity-ordered
//! 9/27/6-dof row from the file's point list, so those types read past the
//! element's own data (fewer points than dofs).  Type 27
//! (`QUADRATIC_PYRAMID`, 13 pts) is rejected because MFEM itself aborts on it
//! (`QuadraticFECollection` has no pyramid element — probe-verified on MFEM
//! 4.10).  The Lagrange types (68-75) need `CreateVTKElementConnectivity` +
//! lexicographic orderings and are not implemented yet.
//!
//! # Not yet reproduced
//!
//! None for the supported types: quadratic tetrahedra (24) get the full
//! `MarkTetMeshForRefinement` treatment including the geometry-row rotation
//! that MFEM performs through `PrepareNodeReorder`/`DoNodeReorder`
//! (probe-verified).

use std::collections::HashMap;
use std::io::Read;

use fem_core::{FemError, FemResult, NodeId};
use fem_element::ReferenceElement;
use fem_mesh::element_type::ElementType;
use fem_mesh::simplex::GeometryData;
use fem_mesh::simplex::Mesh;
use fem_mesh::{BoundaryTag, mark_tet_mesh_for_refinement};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// A legacy-VTK mesh, split by spatial dimension exactly like
/// [`crate::mfem::MfemFile`].
pub struct VtkMeshFile {
    /// `Some` when the mesh's space dimension is 2.
    pub mesh2d: Option<Mesh<2>>,
    /// `Some` when the mesh's space dimension is 3.
    pub mesh3d: Option<Mesh<3>>,
}

/// Read a legacy VTK mesh from a stream (MFEM `Mesh::Load`'s VTK branch),
/// with MFEM's default `Load(..., refine = 1, fix_orientation = true)`.
pub fn read_vtk_mesh<R: Read>(mut reader: R) -> FemResult<VtkMeshFile> {
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    read_vtk_mesh_str(&text, true, true)
}

/// [`read_vtk_mesh`] with explicit `Mesh::Finalize(refine, fix_orientation)`
/// knobs — the `Mesh mesh(file, refine, fix_orientation)` constructor overload
/// (`mesh.hpp:823-827`): `refine = false` skips `MarkForRefinement`
/// (the read-side longest-edge marking), `fix_orientation = false` turns
/// `CheckElementOrientation` into a check-only pass.  The trailing
/// `CheckBdrElementOrientation()` runs unconditionally in MFEM, so boundary
/// realignment stays on in every combination.
pub fn read_vtk_mesh_with<R: Read>(
    mut reader: R,
    refine: bool,
    fix_orientation: bool,
) -> FemResult<VtkMeshFile> {
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    read_vtk_mesh_str(&text, refine, fix_orientation)
}

/// Read a legacy VTK mesh file (MFEM defaults, `Mesh(file, 1, 1)`).
pub fn read_vtk_mesh_file(path: impl AsRef<std::path::Path>) -> FemResult<VtkMeshFile> {
    read_vtk_mesh_file_with(path, true, true)
}

/// [`read_vtk_mesh_file`] with explicit `Finalize(refine, fix_orientation)`
/// knobs — e.g. `(false, true)` reproduces the C++ trimmer's
/// `Mesh mesh(mesh_file, 0, 0)` load exactly.
pub fn read_vtk_mesh_file_with(
    path: impl AsRef<std::path::Path>,
    refine: bool,
    fix_orientation: bool,
) -> FemResult<VtkMeshFile> {
    let text = std::fs::read_to_string(path)?;
    read_vtk_mesh_str(&text, refine, fix_orientation)
}

/// Write `mesh` in MFEM format exactly the way MFEM prints a mesh that was
/// *loaded from VTK* (`CreateVTKMesh`): the `nodes` section declares
/// `FiniteElementCollection: Quadratic` and `Ordering: 0` (byNODES), whereas
/// [`crate::mfem::write_mfem`] emits the `Mesh::SetCurvature` convention
/// (`H1_2D_P2`/`H1_3D_P2`, `Ordering: 1` byVDIM).  The dof numbering and the
/// coordinate values are identical between the two conventions — this only
/// relabels the collection and transposes the value block (the float strings
/// are reused verbatim, so the bytes stay formatter-identical).
pub fn write_mfem_vtk_load_style_file(
    path: impl AsRef<std::path::Path>,
    mesh2d: Option<&Mesh<2>>,
    mesh3d: Option<&Mesh<3>>,
) -> FemResult<()> {
    let mut buf: Vec<u8> = Vec::new();
    match (mesh2d, mesh3d) {
        (Some(m2), _) => crate::mfem::write_mfem(&mut buf, m2, mesh3d)?,
        (None, Some(m3)) => crate::mfem::write_mfem(&mut buf, &Mesh::<2>::unit_square_tri(2), Some(m3))?,
        (None, None) => return Err(vtk_err("no mesh to write")),
    }
    let text = String::from_utf8(buf)
        .map_err(|e| vtk_err(format!("internal: writer produced non-UTF-8 output: {e}")))?;
    let out = rewrite_nodes_section_vtk_load_style(&text)?;
    std::fs::write(path, out)?;
    Ok(())
}

/// Relabel + transpose the `nodes` section (see
/// [`write_mfem_vtk_load_style_file`]).  Layout of the section the writer
/// emits:
///
/// ```text
/// nodes
/// FiniteElementSpace
/// FiniteElementCollection: H1_3D_P2
/// VDim: 3
/// Ordering: 1
///
/// <n_dofs * vdim values, one per line, vdim-major>
/// ```
fn rewrite_nodes_section_vtk_load_style(text: &str) -> FemResult<String> {
    const MARKER: &str = "FiniteElementCollection: ";
    const ORDERING: &str = "Ordering: ";
    let coll_pos = match text.rfind(MARKER) {
        Some(p) => p,
        // A linear mesh has no `nodes` section; the plain writer output already
        // equals C++ `Mesh::Print()` byte for byte.
        None => return Ok(text.to_string()),
    };
    let line_end = text[coll_pos..]
        .find('\n')
        .map(|p| coll_pos + p)
        .unwrap_or(text.len());
    let coll = &text[coll_pos + MARKER.len()..line_end];
    // Only the order-2 H1 names come from this reader's meshes; anything else
    // would mean the mesh was built elsewhere.
    let relabeled = match coll {
        "H1_2D_P2" | "H1_3D_P2" => "Quadratic",
        other => {
            return Err(vtk_err(format!(
                "internal: unexpected nodes collection `{other}` for a VTK-loaded mesh"
            )));
        }
    };
    let rest = &text[line_end..];
    let ord_pos = rest
        .find(ORDERING)
        .ok_or_else(|| vtk_err("internal: writer output has no `Ordering:` line"))?;
    let ord_line_end = rest[ord_pos..]
        .find('\n')
        .map(|p| ord_pos + p)
        .unwrap_or(rest.len());
    // `VDim: <d>` sits between the collection and the ordering lines.
    let vdim_head = &rest[..ord_pos];
    let vdim_pos = vdim_head
        .rfind("VDim: ")
        .ok_or_else(|| vtk_err("internal: writer output has no `VDim:` line"))?;
    let vdim_end = rest[vdim_pos..]
        .find('\n')
        .map(|p| vdim_pos + p)
        .unwrap_or(rest.len());
    let vdim_line: usize = rest[vdim_pos + "VDim: ".len()..vdim_end]
        .trim()
        .parse()
        .map_err(|_| vtk_err("internal: bad `VDim:` in writer output"))?;
    let tokens: Vec<&str> = rest[ord_line_end..].split_whitespace().collect();
    if tokens.len() % vdim_line != 0 {
        return Err(vtk_err("internal: nodes value count is not a multiple of VDim"));
    }
    let nd = tokens.len() / vdim_line;
    // Transpose: `write_mfem`'s `Ordering: 1` block is node-major (the vdim
    // components of dof d consecutively), while MFEM's `Ordering: 0`
    // (byNODES) is component-major (all x, then all y, …).  Emit the same
    // strings in MFEM's `GridFunction::Save` packing: one value per line.
    let mut out = String::with_capacity(text.len() * 2);
    out.push_str(&text[..coll_pos + MARKER.len()]);
    out.push_str(relabeled);
    out.push_str(&rest[..ord_pos]);
    out.push_str(ORDERING);
    out.push_str("0\n\n");
    for c in 0..vdim_line {
        for d in 0..nd {
            out.push_str(tokens[d * vdim_line + c]);
            out.push('\n');
        }
    }
    Ok(out)
}

fn vtk_err(msg: impl Into<String>) -> FemError {
    FemError::Mesh(format!("vtk: {}", msg.into()))
}

// ---------------------------------------------------------------------------
// Parsing (MFEM `Mesh::ReadVTKMesh`)
// ---------------------------------------------------------------------------

/// Whitespace tokenizer mirroring MFEM's `istream >> buff`, tracking the byte
/// offset after the last consumed token so the trailing attribute section can
/// be scanned *line by line* like MFEM's `getline` loop.
struct Tokens<'a> {
    toks: Vec<(&'a str, usize)>,
    pos: usize,
}

impl<'a> Tokens<'a> {
    fn new(text: &'a str) -> Self {
        let mut toks: Vec<(&'a str, usize)> = Vec::new();
        let mut search_from = 0usize;
        for t in text.split_whitespace() {
            let pos = text[search_from..].find(t).unwrap() + search_from;
            toks.push((t, pos));
            search_from = pos + t.len();
        }
        Tokens { toks, pos: 0 }
    }
    fn peek(&self) -> Option<&'a str> {
        self.toks.get(self.pos).map(|&(t, _)| t)
    }
    fn next(&mut self) -> Option<&'a str> {
        let t = self.toks.get(self.pos).map(|&(t, _)| t);
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
    /// Byte offset of the first unconsumed token (`usize::MAX` at EOF).
    fn rest_offset(&self) -> usize {
        self.toks.get(self.pos).map(|&(_, o)| o).unwrap_or(usize::MAX)
    }
    fn skip_until(&mut self, kw: &str, what: &str) -> FemResult<()> {
        loop {
            match self.next() {
                Some(t) if t == kw => return Ok(()),
                Some(_) => {}
                None => return Err(vtk_err(format!("missing {what} (expected `{kw}`)"))),
            }
        }
    }
    fn parse_i64(&mut self, what: &str) -> FemResult<i64> {
        self.next()
            .ok_or_else(|| vtk_err(format!("unexpected end of file reading {what}")))?
            .parse::<i64>()
            .map_err(|_| vtk_err(format!("bad integer reading {what}")))
    }
    fn parse_f64(&mut self, what: &str) -> FemResult<f64> {
        self.next()
            .ok_or_else(|| vtk_err(format!("unexpected end of file reading {what}")))?
            .parse::<f64>()
            .map_err(|_| vtk_err(format!("bad number reading {what}")))
    }
}

struct VtkParsed {
    /// `3 * np` coordinates, point-major.
    points: Vec<f64>,
    /// Cell connectivity, all cells concatenated.
    cell_data: Vec<i64>,
    /// Cumulative offsets: cell `i` occupies `cell_data[end[i-1]..end[i]]`.
    cell_ends: Vec<usize>,
    /// VTK cell type per cell.
    cell_types: Vec<i64>,
    /// Element attributes (empty = none found in the file).
    cell_attributes: Vec<i64>,
}

fn parse_vtk(text: &str) -> FemResult<VtkParsed> {
    // MFEM's `Mesh::Load` dispatcher recognises the file by the first line.
    let header = text.lines().next().unwrap_or("");
    if !header.trim_start().starts_with("# vtk DataFile Version") {
        return Err(vtk_err(
            "not a legacy VTK mesh (first line must be `# vtk DataFile Version ...`)",
        ));
    }
    let mut tk = Tokens::new(text);
    tk.skip_until("ASCII", "the `ASCII` line")?;
    tk.skip_until("DATASET", "the `DATASET` keyword")?;
    if tk.next() != Some("UNSTRUCTURED_GRID") {
        return Err(vtk_err("VTK mesh is not UNSTRUCTURED_GRID"));
    }
    tk.skip_until("POINTS", "the `POINTS` section")?;
    let np = tk.parse_i64("the point count")? as usize;
    let _dtype = tk.next(); // "double" / "float"
    let mut points = Vec::with_capacity(3 * np);
    for _ in 0..3 * np {
        points.push(tk.parse_f64("the POINTS array")?);
    }

    tk.skip_until("CELLS", "the `CELLS` section")?;
    let ncells = tk.parse_i64("the cell count")? as usize;
    let nvals = tk.parse_i64("the connectivity size")? as usize;
    let mut cell_ends = Vec::with_capacity(ncells);
    let mut cell_data = Vec::with_capacity(nvals.saturating_sub(ncells));
    for _ in 0..ncells {
        let nv = tk.parse_i64("a cell node count")? as usize;
        cell_ends.push(cell_data.len() + nv);
        for _ in 0..nv {
            cell_data.push(tk.parse_i64("cell connectivity")?);
        }
    }
    // MFEM reads one token and requires `CELL_TYPES`.
    match tk.next() {
        Some("CELL_TYPES") => {}
        other => {
            return Err(vtk_err(format!(
                "expected `CELL_TYPES` after the CELLS section, found `{}`",
                other.unwrap_or("<eof>")
            )));
        }
    }
    let ntypes = tk.parse_i64("the cell-type count")? as usize;
    let mut cell_types = Vec::with_capacity(ntypes);
    for _ in 0..ntypes {
        cell_types.push(tk.parse_i64("cell types")?);
    }

    // `while (input.good() && buff != "CELL_DATA") input >> buff;`
    loop {
        match tk.peek() {
            Some("CELL_DATA") | None => break,
            Some(_) => {
                tk.next();
            }
        }
    }
    // The attribute scan is a *line* loop from here (MFEM's `getline`).
    let mut cell_attributes: Vec<i64> = Vec::new();
    let rest = match tk.rest_offset() {
        usize::MAX => "",
        off => &text[off..],
    };
    for line in rest.lines() {
        let line = line.trim_end_matches('\r');
        if line.starts_with("POINT_DATA") {
            break;
        }
        if line.starts_with("SCALARS material") || line.starts_with("SCALARS attribute") {
            // MFEM: the next getline must yield `LOOKUP_TABLE default`, then
            // the values follow.
            let mut vt = Tokens::new(&rest[rest.find(line).unwrap() + line.len()..]);
            if vt.peek() == Some("LOOKUP_TABLE") {
                vt.next();
                vt.next(); // default
            }
            for _ in 0..ntypes {
                cell_attributes.push(vt.parse_i64("cell attributes")?);
            }
            break;
        }
    }
    if cell_attributes.len() != ncells {
        cell_attributes.clear(); // MFEM: no material array → default attribute
    }
    Ok(VtkParsed { points, cell_data, cell_ends, cell_types, cell_attributes })
}

// ---------------------------------------------------------------------------
// Cell-type tables
// ---------------------------------------------------------------------------

/// Corner (= MFEM vertex) count of each supported VTK cell type.
fn corners_of(ct: i64) -> Option<usize> {
    Some(match ct {
        5 => 3,
        9 => 4,
        10 => 4,
        12 => 8,
        13 => 6,
        14 => 5,
        22 => 3,
        24 => 4,
        28 => 4,
        29 => 8,
        32 => 6,
        _ => return None,
    })
}

/// Total VTK point count of each supported type.
fn points_of(ct: i64) -> Option<usize> {
    Some(match ct {
        5 | 9 | 10 | 12 | 13 | 14 => corners_of(ct)?,
        22 => 6,
        24 => 10,
        28 => 9,
        29 => 27,
        32 => 18,
        _ => return None,
    })
}

/// The mesh's *corner* element type for each supported VTK cell type.
fn corner_type_of(ct: i64) -> Option<ElementType> {
    Some(match ct {
        5 | 22 => ElementType::Tri3,
        9 | 28 => ElementType::Quad4,
        10 | 24 => ElementType::Tet4,
        12 | 29 => ElementType::Hex8,
        13 | 32 => ElementType::Prism6,
        14 => ElementType::Pyramid5,
        _ => return None,
    })
}

/// Geometry family key for the face tables and the geometry rows.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Family {
    Tri,
    Quad,
    Tet,
    Cube,
    Prism,
    Pyramid,
}

fn family_of(ct: i64) -> Option<Family> {
    Some(match ct {
        5 | 22 => Family::Tri,
        9 | 28 => Family::Quad,
        10 | 24 => Family::Tet,
        12 | 29 => Family::Cube,
        13 | 32 => Family::Prism,
        14 => Family::Pyramid,
        _ => return None,
    })
}

fn family_dim(f: Family) -> usize {
    match f {
        Family::Tri | Family::Quad => 2,
        Family::Tet | Family::Cube | Family::Prism | Family::Pyramid => 3,
    }
}

fn family_corner_type(f: Family) -> ElementType {
    match f {
        Family::Tri => ElementType::Tri3,
        Family::Quad => ElementType::Quad4,
        Family::Tet => ElementType::Tet4,
        Family::Cube => ElementType::Hex8,
        Family::Prism => ElementType::Prism6,
        Family::Pyramid => ElementType::Pyramid5,
    }
}

/// `VTKGeometry::PrismMap` — VTK prism vertex order → MFEM.
const PRISM_MAP: [usize; 6] = [0, 2, 1, 3, 5, 4];

/// `Mesh::vtk_quadratic_tet` — VTK point j sits at MFEM dof `TABLE[j]`.
const VTK_QUADRATIC_TET: [usize; 10] = [0, 1, 2, 3, 4, 7, 5, 6, 8, 9];
/// `Mesh::vtk_quadratic_wedge` (a quadratic prism has 18 dofs).
const VTK_QUADRATIC_WEDGE: [usize; 18] = [
    0, 2, 1, 3, 5, 4, 8, 7, 6, 11, 10, 9, 12, 14, 13, 17, 16, 15,
];
/// `Mesh::vtk_quadratic_hex` (TRIANGLE/SQUARE use its identity prefix).
const VTK_QUADRATIC_HEX: [usize; 27] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 24,
    22, 21, 23, 20, 25, 26,
];

/// MFEM's `vtk_mfem` table for the supported quadratic types: VTK point j →
/// MFEM local dof (`CreateVTKMesh`'s switch; TRIANGLE/SQUARE share the
/// identity prefix of `vtk_quadratic_hex`).
fn vtk_mfem_table(ct: i64) -> &'static [usize] {
    match ct {
        22 => &VTK_QUADRATIC_HEX[..6],
        24 => &VTK_QUADRATIC_TET,
        28 => &VTK_QUADRATIC_HEX[..9],
        29 => &VTK_QUADRATIC_HEX,
        32 => &VTK_QUADRATIC_WEDGE,
        _ => unreachable!("vtk_mfem_table on unsupported type {ct}"),
    }
}

/// `Geometry::Constants<T>::FaceVert` — local face cycles of the 3-D *corner*
/// types, first-encounter orientation donor (`fem/geom.cpp`: tet 987, hex
/// 1032, prism 1061, pyramid 1086; `-1` padding dropped).  2-D families use
/// their local edges (`Element::GetEdgeVertices` order).
fn local_faces(f: Family) -> Vec<Vec<usize>> {
    match f {
        Family::Tri => vec![vec![0, 1], vec![1, 2], vec![2, 0]],
        Family::Quad => vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 0]],
        Family::Tet => vec![vec![1, 2, 3], vec![0, 3, 2], vec![0, 1, 3], vec![0, 2, 1]],
        Family::Cube => vec![
            vec![3, 2, 1, 0],
            vec![0, 1, 5, 4],
            vec![1, 2, 6, 5],
            vec![2, 3, 7, 6],
            vec![3, 0, 4, 7],
            vec![4, 5, 6, 7],
        ],
        Family::Prism => vec![
            vec![0, 2, 1],
            vec![3, 4, 5],
            vec![0, 1, 4, 3],
            vec![1, 2, 5, 4],
            vec![2, 0, 3, 5],
        ],
        Family::Pyramid => vec![
            vec![3, 2, 1, 0],
            vec![0, 1, 4],
            vec![1, 2, 4],
            vec![2, 3, 4],
            vec![3, 0, 4],
        ],
    }
}

// ---------------------------------------------------------------------------
// VTK reference-node coordinates (order-2 lattices, half-integer keys)
// ---------------------------------------------------------------------------

/// Reference position of each VTK node of a supported quadratic type, in the
/// family's unit reference frame, in half-integer steps (coordinate × 2).
/// The layouts are exactly the ones MFEM's own `BarycentricToVTKTriangle` /
/// `BarycentricToVTKTetra` / `CartesianToVTKPrism` / `CartesianToVTKTensor`
/// enumerations produce (`mesh/vtk.cpp`).
fn vtk_half_coords(ct: i64) -> Vec<[i64; 3]> {
    match ct {
        // Corners, then edge mids of (0,1), (1,2), (2,0).
        22 => vec![[0, 0, 0], [2, 0, 0], [0, 2, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        // Corners, then edge mids of (0,1), (1,2), (2,0), (0,3), (1,3), (2,3).
        24 => vec![
            [0, 0, 0],
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
        ],
        // Corners (0,0),(1,0),(1,1),(0,1); edge mids (0,1),(1,2),(2,3),(3,0);
        // center.
        28 => vec![
            [0, 0, 0],
            [2, 0, 0],
            [2, 2, 0],
            [0, 2, 0],
            [1, 0, 0],
            [2, 1, 0],
            [1, 2, 0],
            [0, 1, 0],
            [1, 1, 0],
        ],
        // 8 corners, 12 edge mids in VTK hex edge order, 6 face centers in
        // VTK face order, body center.
        29 => {
            let c: [[i64; 3]; 8] = [
                [0, 0, 0],
                [2, 0, 0],
                [2, 2, 0],
                [0, 2, 0],
                [0, 0, 2],
                [2, 0, 2],
                [2, 2, 2],
                [0, 2, 2],
            ];
            let mut v = c.to_vec();
            for e in [
                [0, 1],
                [1, 2],
                [3, 2],
                [0, 3],
                [4, 5],
                [5, 6],
                [7, 6],
                [4, 7],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ] {
                v.push(std::array::from_fn(|i| (c[e[0]][i] + c[e[1]][i]) / 2));
            }
            // Face centers, in the layout MFEM 4.10's PrintVTK writes and
            // LoadVtk expects (probe-verified with a single curved hex whose
            // CELLS line lists the dof ids: x=0, x=1, y=0, y=1, z=0, z=1 —
            // NOT the VTK-spec face order).
            v.push([0, 1, 1]); // x = 0
            v.push([2, 1, 1]); // x = 1
            v.push([1, 0, 1]); // y = 0
            v.push([1, 2, 1]); // y = 1
            v.push([1, 1, 0]); // z = 0
            v.push([1, 1, 2]); // z = 1
            v.push([1, 1, 1]); // body center
            v
        }
        // 6 corners (bottom 0-2, top 3-5), bottom / top / vertical edge mids,
        // then the three quadrilateral face centers — the layout
        // `CartesianToVTKPrism(i, j, k, 2)` enumerates.
        32 => {
            let c: [[i64; 3]; 6] = [
                [0, 0, 0],
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 2],
                [2, 0, 2],
                [0, 2, 2],
            ];
            let mut v = c.to_vec();
            v.push([1, 0, 0]);
            v.push([1, 1, 0]);
            v.push([0, 1, 0]); // bottom edge mids
            v.push([1, 0, 2]);
            v.push([1, 1, 2]);
            v.push([0, 1, 2]); // top edge mids
            v.push([0, 0, 1]);
            v.push([2, 0, 1]);
            v.push([0, 2, 1]); // vertical edge mids
            v.push([1, 0, 1]);
            v.push([1, 1, 1]);
            v.push([0, 1, 1]); // quad face centers
            v
        }
        _ => unreachable!("vtk_half_coords on unsupported type {ct}"),
    }
}

/// fem-rs geometry-row slot of each VTK point of quadratic cell type `ct`,
/// matched by reference coordinates (half-integer keys, exact at order 2).
/// This fuses MFEM's `vtk_mfem[j]` (VTK point → MFEM dof) with the MFEM dof →
/// fem-rs slot correspondence of the ported H1 element layer, whose element
/// factories place their dofs on the same closed Gauss-Lobatto lattices.
fn vtk_slots_for(ct: i64) -> FemResult<Vec<usize>> {
    let vtk = vtk_half_coords(ct);
    // Unit-frame VTK half-integers → the fem-rs reference frame of the family
    // (`HexQk` lives on `[-1,1]³`: u ∈ {0,½,1} ↦ 2u−1, quantized 2U−2;
    // `PrismPk` puts the triangle on the (y, z) axes with the layer on x; the
    // tri/quad/tet frames are the unit frames).
    let to_fem: Box<dyn Fn([i64; 3]) -> [i64; 3]> = match ct {
        29 => Box::new(|[a, b, c]| [2 * a - 2, 2 * b - 2, 2 * c - 2]),
        32 => Box::new(|[a, b, c]| [c, b, a]),
        _ => Box::new(|c| c),
    };
    let fem = fem_slot_half_coords(corner_type_of(ct).unwrap(), 2);
    let mut slots = vec![usize::MAX; vtk.len()];
    for (j, &v) in vtk.iter().enumerate() {
        let v = to_fem(v);
        match fem.iter().position(|&f| f == v) {
            Some(s) => slots[j] = s,
            None => {
                return Err(vtk_err(format!(
                    "internal: VTK node {j} of cell type {ct} at {v:?} has no \
                     matching fem-rs geometry slot"
                )));
            }
        }
    }
    let mut unique = slots.clone();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != vtk.len() {
        return Err(vtk_err(format!(
            "internal: the VTK slot match for cell type {ct} is not a bijection"
        )));
    }
    Ok(slots)
}

/// fem-rs geometry-row slot reference coordinates, quantized like
/// [`vtk_half_coords`].  The pyramid stays in its own (MFEM) frame — its
/// coordinates are integers, so scaling by 2 is a no-op.
fn fem_slot_half_coords(corner_et: ElementType, order: u8) -> Vec<[i64; 3]> {
    let p = order as usize;
    let coords: Vec<Vec<f64>> = match corner_et {
        ElementType::Tri3 => fem_element::lagrange::factory::H1TriPk::new(p).dof_coords(),
        ElementType::Quad4 => fem_element::lagrange::factory::QuadQk::new(p).dof_coords(),
        ElementType::Tet4 => fem_element::lagrange::factory::H1TetPk::new(p).dof_coords(),
        ElementType::Hex8 => fem_element::lagrange::factory::HexQk::new(p).dof_coords(),
        ElementType::Prism6 => fem_element::lagrange::PrismPk::new(p).dof_coords(),
        ElementType::Pyramid5 => fem_element::lagrange::h1_pyramid_element(
            p,
            fem_element::lagrange::PyramidBasisType::default(),
        )
        .dof_coords(),
        other => unreachable!("fem_slot_half_coords on {other:?}"),
    };
    coords
        .into_iter()
        .map(|c| {
            let q = |i: usize| (c.get(i).copied().unwrap_or(0.0) * 2.0).round() as i64;
            [q(0), q(1), q(2)]
        })
        .collect()
}

/// Geometry-row dof count of a family at `order` (`h1_family_dofs`'s MFEM
/// contract, keyed by the family's corner type).
fn h1_dof_count(f: Family, order: u8) -> usize {
    fem_mesh::simplex::h1_family_dofs(family_corner_type(f), order)
}

// ---------------------------------------------------------------------------
// Mesh construction (MFEM `Mesh::CreateVTKMesh` + `Finalize`)
// ---------------------------------------------------------------------------

/// Everything `CreateVTKMesh` produces, before the face generation and the
/// refinement marking of `Finalize`.
struct VtkCells {
    /// Space dimension (2 or 3, MFEM's `spaceDim`).
    space_dim: usize,
    /// Corner-vertex coordinates (quadratic meshes: corners only).
    coords: Vec<f64>,
    /// Corner connectivity, uniform- or CSR-layout per `elem_types`.
    conn: Vec<NodeId>,
    elem_tags: Vec<i32>,
    elem_types: Vec<ElementType>,
    fams: Vec<Family>,
    quadratic: bool,
    /// Per-element geometry rows in fem-rs slot order (quadratic only).
    geo_rows: Vec<Vec<NodeId>>,
    /// Coordinates of the non-corner geometry nodes, in point order, with
    /// `space_dim` components each.
    extra_coords: Vec<f64>,
}

fn build_cells(p: VtkParsed) -> FemResult<VtkCells> {
    let ncells = p.cell_types.len();
    if ncells != p.cell_ends.len() {
        return Err(vtk_err(format!(
            "CELL_TYPES has {} entries but CELLS has {}",
            ncells,
            p.cell_ends.len()
        )));
    }
    if ncells == 0 {
        return Err(vtk_err("mesh has no cells"));
    }
    let np = p.points.len() / 3;

    // Per-cell geometry with MFEM's consistency checks (`CreateVTKMesh`).
    let mut fams: Vec<Family> = Vec::with_capacity(ncells);
    let mut order = -1i32;
    let mut elem_dim = -1i32;
    for i in 0..ncells {
        let ct = p.cell_types[i];
        let Some(fam) = family_of(ct) else {
            return Err(unsupported_types_error(&p.cell_types));
        };
        let d = family_dim(fam) as i32;
        let o = if points_of(ct) > corners_of(ct) { 2 } else { 1 };
        if elem_dim != -1 && elem_dim != d {
            return Err(vtk_err("Elements with different dimensions are not supported"));
        }
        if order != -1 && order != o {
            return Err(vtk_err("Elements with different orders are not supported"));
        }
        elem_dim = d;
        order = o;
        fams.push(fam);
    }
    if elem_dim == 1 {
        return Err(vtk_err(
            "1-D VTK meshes (SEGMENT/QUADRATIC_SEGMENT cells) are not supported; \
             the fem-rs mesh layer has no 1-D representation",
        ));
    }
    let quadratic = order == 2;

    // spaceDim: `CreateVTKMesh`'s min/max scan over d = 3..1.
    let mut space_dim = 0usize;
    if np > 0 {
        for d in (1..=3).rev() {
            let mut min = p.points[d - 1];
            let mut max = p.points[d - 1];
            for i in 1..np {
                let v = p.points[3 * i + d - 1];
                min = min.min(v);
                max = max.max(v);
                if min != max {
                    space_dim = d;
                    break;
                }
            }
            if space_dim > 0 {
                break;
            }
        }
    }
    if space_dim == 0 {
        space_dim = elem_dim as usize; // `FinalizeTopology`
    }
    match space_dim {
        2 | 3 => {}
        other => {
            return Err(vtk_err(format!(
                "unsupported space dimension {other} (only 2-D and 3-D are supported)"
            )));
        }
    }

    // ── Vertices ────────────────────────────────────────────────────────────
    // Linear: every point is a vertex (`NumOfVertices = np`).  Quadratic:
    // corners only, renumbered by first appearance in the point array.
    let mut pts_dof = vec![-1i64; np];
    let n_vertices;
    let coords: Vec<f64>;
    if quadratic {
        for i in 0..ncells {
            let nv = corners_of(p.cell_types[i]).unwrap();
            let start = if i > 0 { p.cell_ends[i - 1] } else { 0 };
            for &pt in &p.cell_data[start..start + nv] {
                pts_dof[pt as usize] = 0;
            }
        }
        let mut nv_count = 0usize;
        for m in pts_dof.iter_mut() {
            if *m != -1 {
                *m = nv_count as i64;
                nv_count += 1;
            }
        }
        n_vertices = nv_count;
        let mut c = vec![0.0; n_vertices * space_dim];
        for (pt, &m) in pts_dof.iter().enumerate() {
            if m == -1 {
                continue;
            }
            for d in 0..space_dim {
                c[m as usize * space_dim + d] = p.points[3 * pt + d];
            }
        }
        coords = c;
    } else {
        pts_dof.iter_mut().enumerate().for_each(|(i, m)| *m = i as i64);
        n_vertices = np;
        // Linear: `coords` is the point array as-is — compacted from the
        // file's 3 components to the mesh's spaceDim (a planar mesh in 3-D
        // space keeps 3 components per point here; the Mesh<D> below is
        // chosen by spaceDim).
        if space_dim == 3 {
            coords = p.points.clone();
        } else {
            coords = (0..np)
                .flat_map(|i| p.points[3 * i..3 * i + 2].to_vec())
                .collect();
        }
    }

    // ── Element connectivity (+PrismMap) and geometry rows ──────────────────
    let mut conn: Vec<NodeId> = Vec::with_capacity(ncells * 4);
    let mut elem_types: Vec<ElementType> = Vec::with_capacity(ncells);
    let mut elem_tags: Vec<i32> = Vec::with_capacity(ncells);
    let mut fam_dofs: HashMap<Family, usize> = HashMap::new();
    if quadratic {
        for &f in &fams {
            fam_dofs.entry(f).or_insert_with(|| h1_dof_count(f, 2));
        }
    }
    // fem-rs slot of each VTK point, cached per cell type.
    let mut slot_cache: HashMap<i64, Vec<usize>> = HashMap::new();
    // Geometry-node id of every non-corner point.
    let mut extra_id: Vec<usize> = vec![usize::MAX; np];
    let mut n_extra = 0usize;
    let mut extra_coords: Vec<f64> = Vec::new();
    let mut geo_rows: Vec<Vec<NodeId>> = Vec::with_capacity(if quadratic { ncells } else { 0 });

    for i in 0..ncells {
        let ct = p.cell_types[i];
        let nv = corners_of(ct).unwrap();
        let npt = points_of(ct).unwrap();
        let start = if i > 0 { p.cell_ends[i - 1] } else { 0 };
        let pts = &p.cell_data[start..start + npt];

        // Corner connectivity (VTK order == MFEM, prisms permuted).
        if fams[i] == Family::Prism {
            for &m in PRISM_MAP.iter() {
                conn.push(pts_dof[pts[m] as usize] as u32);
            }
        } else {
            for &pt in pts.iter().take(nv) {
                conn.push(pts_dof[pt as usize] as u32);
            }
        }
        elem_types.push(corner_type_of(ct).unwrap());
        elem_tags.push(if p.cell_attributes.is_empty() {
            1
        } else {
            p.cell_attributes[i] as i32
        });

        if quadratic {
            // Resolve every VTK point to a geometry-node id, then place it at
            // its fem-rs slot.  A non-corner point that doubles as another
            // element's corner is already a vertex (`pts_dof != -1`) — MFEM's
            // `pts_dof[cell_data[offset+j]] != -1` branch.
            let table_len = vtk_mfem_table(ct).len();
            debug_assert_eq!(table_len, npt);
            let slots = match slot_cache.get(&ct) {
                Some(s) => s.clone(),
                None => {
                    let s = vtk_slots_for(ct)?;
                    slot_cache.insert(ct, s.clone());
                    s
                }
            };
            let mut row = vec![0u32; fam_dofs[&fams[i]]];
            for (j, &slot) in slots.iter().enumerate() {
                let pt = pts[j] as usize;
                let node = if j < nv || pts_dof[pt] != -1 {
                    pts_dof[pt] as u32
                } else if extra_id[pt] != usize::MAX {
                    extra_id[pt] as u32
                } else {
                    let id = n_vertices + n_extra;
                    extra_id[pt] = id;
                    n_extra += 1;
                    extra_coords.extend_from_slice(&p.points[3 * pt..3 * pt + space_dim]);
                    id as u32
                };
                row[slot] = node;
            }
            geo_rows.push(row);
        }
    }

    Ok(VtkCells {
        space_dim,
        coords,
        conn,
        elem_tags,
        elem_types,
        fams,
        quadratic,
        geo_rows,
        extra_coords,
    })
}

fn unsupported_types_error(types: &[i64]) -> FemError {
    let mut bad: Vec<i64> = types.to_vec();
    bad.retain(|ct| corners_of(*ct).is_none());
    bad.sort_unstable();
    bad.dedup();
    vtk_err(format!(
        "unsupported VTK cell type(s) {bad:?}: this reader supports the linear \
         types 5 (triangle), 9 (quad), 10 (tet), 12 (hex), 13 (wedge), 14 \
         (pyramid) and the quadratic types 22, 24, 28, 29, 32 only.  The \
         serendipity quadratics 23/25/34 are rejected because MFEM's own \
         legacy reader mis-fills their rows (fewer points than dofs); the \
         quadratic pyramid 27 is rejected because MFEM itself aborts on it \
         (QuadraticFECollection has no pyramid element); the Lagrange types \
         68-75 are not implemented yet."
    ))
}

// ---------------------------------------------------------------------------
// Shared face/boundary generation (MFEM `GenerateFaces` +
// `GenerateBoundaryElements`; boundary attribute 1)
// ---------------------------------------------------------------------------

/// First-encounter faces: `(vertex cycle, owner element, second element)`.
fn generate_faces(
    n_elems: usize,
    fams: &[Family],
    elem_nodes: impl Fn(u32) -> Vec<u32>,
) -> Vec<(Vec<u32>, u32, Option<u32>)> {
    let mut faces: Vec<(Vec<u32>, u32, Option<u32>)> = Vec::new();
    let mut lookup: HashMap<Vec<u32>, usize> = HashMap::new();
    for e in 0..n_elems {
        let enodes = elem_nodes(e as u32);
        for fv in local_faces(fams[e]) {
            let cyc: Vec<u32> = fv.iter().map(|&i| enodes[i]).collect();
            let mut key = cyc.clone();
            key.sort_unstable();
            match lookup.get(&key) {
                Some(&f) => faces[f].2 = Some(e as u32),
                None => {
                    lookup.insert(key, faces.len());
                    faces.push((cyc, e as u32, None));
                }
            }
        }
    }
    faces
}

/// Element corner slice, honouring the uniform / CSR layouts.
fn elem_slice<'a>(
    conn: &'a [NodeId],
    e: u32,
    uniform: bool,
    npe: usize,
    offsets: Option<&Vec<usize>>,
) -> &'a [u32] {
    if uniform {
        let off = e as usize * npe;
        &conn[off..off + npe]
    } else {
        let offs = offsets.unwrap();
        &conn[offs[e as usize]..offs[e as usize + 1]]
    }
}

/// `(per-element types, CSR offsets)` for a possibly mixed element list;
/// `(None, None)` when every element shares one type.
fn mixed_tables(elem_types: &[ElementType]) -> (Option<Vec<ElementType>>, Option<Vec<usize>>) {
    if elem_types.iter().all(|&t| t == elem_types[0]) {
        (None, None)
    } else {
        let mut offs = Vec::with_capacity(elem_types.len() + 1);
        offs.push(0);
        for &t in elem_types {
            offs.push(offs.last().unwrap() + t.nodes_per_element());
        }
        (Some(elem_types.to_vec()), Some(offs))
    }
}

/// `Mesh::MarkTriMeshForRefinement` → `Triangle::MarkEdge`
/// (`mesh/triangle.cpp:53`): rotate each triangle so its longest edge is
/// (v0, v1).  Returns the applied shift (0/1/2) per element.
fn mark_tri_mesh_for_refinement(mesh: &mut Mesh<2>) -> Vec<u8> {
    let uniform = mesh.elem_types.is_none();
    let npe = mesh.elem_type.nodes_per_element();
    let mut shifts = Vec::with_capacity(mesh.n_elems());
    for e in 0..mesh.n_elems() {
        let start = if uniform {
            e * npe
        } else {
            mesh.elem_offsets.as_ref().unwrap()[e]
        };
        let n: [u32; 3] = [mesh.conn[start], mesh.conn[start + 1], mesh.conn[start + 2]];
        let p0 = mesh.coords_of(n[0]);
        let p1 = mesh.coords_of(n[1]);
        let p2 = mesh.coords_of(n[2]);
        let d2 = |a: [f64; 2], b: [f64; 2]| {
            (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])
        };
        let d = [d2(p0, p1), d2(p1, p2), d2(p0, p2)];
        // `Triangle::MarkEdge`'s exact tie-breaking.
        let shift = if d[0] >= d[1] {
            if d[0] >= d[2] {
                0
            } else {
                2
            }
        } else if d[1] >= d[2] {
            1
        } else {
            2
        };
        if shift == 1 {
            // (v0,v1,v2) -> (v1,v2,v0)
            mesh.conn[start..start + 3].copy_from_slice(&[n[1], n[2], n[0]]);
        } else if shift == 2 {
            // (v0,v1,v2) -> (v2,v0,v1)
            mesh.conn[start..start + 3].copy_from_slice(&[n[2], n[0], n[1]]);
        }
        shifts.push(shift);
    }
    mesh.invalidate_locators();
    shifts
}

/// A reference-frame rotation of one element's geometry row: `new[k]` reads
/// `old[perm[k]]`.
enum RowRotation<'a> {
    /// `Triangle::MarkEdge`'s cyclic shift (0/1/2).
    Tri(u8),
    /// `Tetrahedron::MarkEdge`'s corner permutation: `sigma[i]` = old local
    /// slot of the corner now at new slot `i`.
    Tet(&'a [usize; 4]),
    /// `CheckElementOrientation`'s curved-triangle `Swap(v0, v1)`.
    TriSwap01,
    /// `CheckElementOrientation`'s curved-quadrilateral `Swap(v1, v3)`.
    QuadSwap13,
}

/// Slot permutation of the order-2 triangle for `Triangle::MarkEdge`'s
/// cyclic shift: corner slots cycle with the vertices, edge slots follow
/// their (midpoint) edges.
fn tri_mark_perm(shift: u8) -> [usize; 6] {
    const ROT: [[usize; 6]; 3] = [
        [0, 1, 2, 3, 4, 5], // shift 0: identity
        [1, 2, 0, 4, 5, 3], // shift 1: (c0,c1,c2) -> (c1,c2,c0)
        [2, 0, 1, 5, 3, 4], // shift 2: (c0,c1,c2) -> (c2,c0,c1)
    ];
    ROT[shift as usize]
}

/// Slot permutation of the order-2 tetrahedron for the corner permutation
/// `sigma` (`sigma[i]` = old local slot of the corner now at new slot `i`):
/// corner slots follow the corners, midedge slots follow their edges
/// (midpoint orientation is irrelevant at order 2).
fn tet_mark_perm(sigma: &[usize; 4]) -> [usize; 10] {
    const TET_EDGES: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
    let mut perm = [0usize; 10];
    for (i, &s) in sigma.iter().enumerate() {
        perm[i] = s;
    }
    for (k, &[a, b]) in TET_EDGES.iter().enumerate() {
        perm[4 + k] = 4
            + TET_EDGES
                .iter()
                .position(|&[ca, cb]| {
                    (sigma[a], sigma[b]) == (ca, cb) || (sigma[a], sigma[b]) == (cb, ca)
                })
                .expect("corner permutation lost a tet edge");
    }
    perm
}

/// Slot permutation of the order-2 quadrilateral for `Swap(v1, v3)`:
/// corners `[0,3,2,1]`, edges follow `(0,1),(1,2),(2,3),(3,0) ->
/// (0,3),(3,2),(2,1),(1,0)`, center stays.
const QUAD_SWAP13_PERM: [usize; 9] = [0, 3, 2, 1, 7, 6, 5, 4, 8];

/// Slot permutation of the order-2 triangle for `Swap(v0, v1)`: corners
/// `[1,0,2]`, edges `(0,1),(1,2),(2,0) -> (1,0),(0,2),(2,1)`.
const TRI_SWAP01_PERM: [usize; 6] = [1, 0, 2, 3, 5, 4];

/// Apply `rot` to element `e`'s geometry row in place (uniform and ragged
/// tables).
fn rotate_row_in_place<const D: usize>(mesh: &mut Mesh<D>, e: usize, rot: &RowRotation) {
    let g = mesh.geometry.as_ref().unwrap();
    let (start, len) = if g.nodes_per_elem > 0 {
        (e * g.nodes_per_elem, g.nodes_per_elem)
    } else {
        // Ragged: prefix sum of the family dof counts in element order.
        let mut start = 0usize;
        let mut len = 0usize;
        for e2 in 0..=e {
            let t = mesh.element_type_at(e2 as u32);
            len = fem_mesh::simplex::h1_family_dofs(t, g.order);
            if e2 < e {
                start += len;
            }
        }
        (start, len)
    };
    let perm: Vec<usize> = match rot {
        RowRotation::Tri(shift) => tri_mark_perm(*shift).to_vec(),
        RowRotation::Tet(sigma) => tet_mark_perm(sigma).to_vec(),
        RowRotation::TriSwap01 => TRI_SWAP01_PERM.to_vec(),
        RowRotation::QuadSwap13 => QUAD_SWAP13_PERM.to_vec(),
    };
    let rotated: Vec<NodeId> = perm.iter().map(|&k| g.conn[start + k]).collect();
    let g = mesh.geometry.as_mut().unwrap();
    g.conn[start..start + len].copy_from_slice(&rotated);
}

/// MFEM `CheckElementOrientation(fix = true)`'s *curved* branch
/// (`mesh/mesh.cpp:7370-7390`): the Jacobian at the MFEM geometry center is
/// checked and negatively oriented 2-D triangles (`Swap(v0, v1)`), 2-D
/// quadrilaterals (`Swap(v1, v3)`) and 3-D tetrahedra (`Swap(v0, v1)`) are
/// repaired; wedges and hexes are counted but never fixed.  The shared
/// `Mesh::check_element_orientation` fixes linear geometry only, so this pass
/// reproduces the curved half for the reader (with the geometry-row rotation
/// `DoNodeReorder` would perform).  Quadratic pyramids cannot occur (rejected
/// at parse time).
fn fix_curved_orientation<const D: usize>(mesh: &mut Mesh<D>) {
    let n_elems = mesh.n_elems();
    for e in 0..n_elems {
        let et = mesh.element_type_at(e as u32);
        let (rot, swap): (RowRotation, [usize; 2]) = match (D, et) {
            (2, ElementType::Tri3) => (RowRotation::TriSwap01, [0, 1]),
            (2, ElementType::Quad4) => (RowRotation::QuadSwap13, [1, 3]),
            (3, ElementType::Tet4) => (RowRotation::Tet(&[1, 0, 2, 3]), [0, 1]),
            // Wedges/hexes: counted but not fixed by MFEM either.
            _ => continue,
        };
        let center: &[f64] = match et {
            ElementType::Tri3 => &[1.0 / 3.0, 1.0 / 3.0],
            ElementType::Quad4 => &[0.5, 0.5],
            ElementType::Tet4 => &[0.25, 0.25, 0.25],
            _ => continue,
        };
        if mesh.element_jacobian(e as u32, center).1 >= 0.0 {
            continue;
        }
        let uniform = mesh.elem_types.is_none();
        let npe = mesh.elem_type.nodes_per_element();
        let start = if uniform {
            e * npe
        } else {
            mesh.elem_offsets.as_ref().unwrap()[e]
        };
        mesh.conn.swap(start + swap[0], start + swap[1]);
        rotate_row_in_place(mesh, e, &rot);
    }
}

/// Overwrite each 2-D boundary segment with its owner's *current* local edge
/// cycle (MFEM `CheckBdrElementOrientation`'s `bv[0] == fv[0]` rule).
fn realign_segments_2d(mesh: &mut Mesh<2>, bdr_owner: &[u32]) {
    let uniform = mesh.elem_types.is_none();
    let npe = mesh.elem_type.nodes_per_element();
    let mut new_face_conn = Vec::with_capacity(mesh.face_conn.len());
    for (seg, &owner) in mesh.face_conn.chunks(2).zip(bdr_owner.iter()) {
        let enodes = elem_slice(&mesh.conn, owner, uniform, npe, mesh.elem_offsets.as_ref());
        let fam = if npe == 3 { Family::Tri } else { Family::Quad };
        let mut cyc = [seg[0], seg[1]];
        for f in local_faces(fam) {
            let a = enodes[f[0]];
            let b = enodes[f[1]];
            if (a == seg[0] && b == seg[1]) || (a == seg[1] && b == seg[0]) {
                cyc = [a, b];
                break;
            }
        }
        new_face_conn.extend_from_slice(&cyc);
    }
    mesh.face_conn = new_face_conn;
}

/// Attach the second-order geometry table.  `geo_rows` holds fem-rs-slot
/// ordered node ids per element; the geometry coordinates are the mesh's
/// corner coordinates followed by the non-corner node coordinates.
fn attach_geometry<const D: usize>(
    mesh: &mut Mesh<D>,
    geo_rows: Vec<Vec<NodeId>>,
    extra_coords: &[f64],
) {
    let uniform_len = geo_rows.first().map(|r| r.len()).unwrap_or(0);
    let ragged = geo_rows.iter().any(|r| r.len() != uniform_len);
    let (conn, nodes_per_elem) = if ragged {
        let conn = geo_rows.iter().flatten().copied().collect();
        (conn, 0) // `nodes_per_elem == 0` marks the ragged layout
    } else {
        let conn = geo_rows.into_iter().flatten().collect();
        (conn, uniform_len)
    };
    let mut coords = mesh.coords.clone();
    coords.extend_from_slice(extra_coords);
    mesh.geometry = Some(GeometryData {
        order: 2,
        conn,
        nodes_per_elem,
        coords,
        n_nodes: mesh.n_nodes() + extra_coords.len() / D,
    });
}

fn build_mesh_2d(cells: VtkCells, refine: bool, fix_orientation: bool) -> FemResult<Mesh<2>> {
    let VtkCells {
        coords,
        conn,
        elem_tags,
        elem_types,
        fams,
        quadratic,
        geo_rows,
        extra_coords,
        ..
    } = cells;
    let n_elems = elem_types.len();
    let uniform = elem_types.iter().all(|&t| t == elem_types[0]);
    let elem_type = elem_types[0];
    let npe = elem_type.nodes_per_element();
    let (elem_types_opt, elem_offsets_opt) = mixed_tables(&elem_types);
    let faces = generate_faces(n_elems, &fams, |e| {
        elem_slice(&conn, e, uniform, npe, elem_offsets_opt.as_ref()).to_vec()
    });
    let mut face_conn: Vec<NodeId> = Vec::new();
    let mut bdr_owner: Vec<u32> = Vec::new();
    for (cyc, e1, e2) in &faces {
        if e2.is_none() {
            face_conn.extend_from_slice(cyc);
            bdr_owner.push(*e1);
        }
    }
    let face_tags: Vec<BoundaryTag> = vec![1; bdr_owner.len()];
    let mut mesh = Mesh::<2> {
        coords,
        conn,
        elem_tags,
        elem_type,
        face_conn,
        face_tags,
        face_type: ElementType::Line2,
        elem_types: elem_types_opt,
        elem_offsets: elem_offsets_opt,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None,
        nc_vertex_view: None,
        vertex_parents: vec![],
    };
    // MFEM orients a *loaded VTK* mesh in two places:
    //   * `CreateVTKMesh` (mesh_readers.cpp:488): `CheckElementOrientation(true)`
    //     — **unconditional** for linear meshes, whatever the knobs are;
    //   * `Mesh::Finalize(refine, fix_orientation)` (mesh.cpp:3793):
    //     `CheckElementOrientation(fix_orientation)` — knob-controlled, and the
    //     only orientation pass a quadratic (curved) mesh ever gets.
    // The geometry attaches so the curved branch (mirrored by
    // `fix_curved_orientation`) and the marking's row rotation can run; the
    // extra linear pass under `fix_orientation` is the reader's defensive
    // split of MFEM's single curved-mesh check (same corner swaps).
    if quadratic {
        attach_geometry(&mut mesh, geo_rows, &extra_coords);
        if fix_orientation {
            fix_curved_orientation(&mut mesh);
            mesh.check_element_orientation(true);
        }
    } else {
        mesh.check_element_orientation(true);
    }
    // `MarkForRefinement` (`meshgen` bit 1 → 2-D triangles), gated on the
    // `refine` knob like `Mesh::Finalize`.  The trailing
    // `CheckBdrElementOrientation()` is unconditional in MFEM: its realignment
    // is mirrored below and is an identity unless the owners' vertex cycles
    // changed (marking or an orientation fix).
    if refine && fams.iter().any(|&f| f == Family::Tri) {
        let shifts = mark_tri_mesh_for_refinement(&mut mesh);
        if quadratic {
            for (e, &shift) in shifts.iter().enumerate() {
                if shift != 0 {
                    rotate_row_in_place(&mut mesh, e, &RowRotation::Tri(shift));
                }
            }
        }
    }
    if fams.iter().any(|&f| f == Family::Tri) {
        // MFEM built the boundary before the marking; `Finalize`'s
        // `CheckBdrElementOrientation` re-aligns every segment with its
        // owner's (possibly rotated) local edge (`bv[0] == fv[0]`).
        realign_segments_2d(&mut mesh, &bdr_owner);
    }
    Ok(mesh)
}

fn build_mesh_3d(cells: VtkCells, refine: bool, fix_orientation: bool) -> FemResult<Mesh<3>> {
    let VtkCells {
        coords,
        conn,
        elem_tags,
        elem_types,
        fams,
        quadratic,
        geo_rows,
        extra_coords,
        ..
    } = cells;
    let n_elems = elem_types.len();
    let uniform = elem_types.iter().all(|&t| t == elem_types[0]);
    let elem_type = elem_types[0];
    let npe = elem_type.nodes_per_element();
    let (elem_types_opt, elem_offsets_opt) = mixed_tables(&elem_types);
    let faces = generate_faces(n_elems, &fams, |e| {
        elem_slice(&conn, e, uniform, npe, elem_offsets_opt.as_ref()).to_vec()
    });
    // Boundary records with their per-face type; a tet+hex mesh has both
    // triangular and quadrilateral faces (MFEM's `faces[i]->Duplicate`).
    let mut face_conn: Vec<NodeId> = Vec::new();
    let mut face_type_list: Vec<ElementType> = Vec::new();
    let mut face_offsets: Vec<usize> = vec![0];
    for (cyc, _, e2) in &faces {
        if e2.is_none() {
            face_conn.extend_from_slice(cyc);
            face_type_list.push(if cyc.len() == 3 { ElementType::Tri3 } else { ElementType::Quad4 });
            face_offsets.push(face_offsets.last().unwrap() + cyc.len());
        }
    }
    let n_bdr = face_type_list.len();
    let mixed_faces = face_type_list.iter().any(|&t| t != face_type_list[0]);
    let face_type = face_type_list.first().copied().unwrap_or(ElementType::Tri3);
    let (face_types, face_offsets) = if mixed_faces {
        (Some(face_type_list), Some(face_offsets))
    } else {
        (None, None)
    };
    let face_tags: Vec<BoundaryTag> = vec![1; n_bdr];
    let mut mesh = Mesh::<3> {
        coords,
        conn,
        elem_tags,
        elem_type,
        face_conn,
        face_tags,
        face_type,
        elem_types: elem_types_opt,
        elem_offsets: elem_offsets_opt,
        face_types,
        face_offsets,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None,
        nc_vertex_view: None,
        vertex_parents: vec![],
    };
    // MFEM orients a *loaded VTK* mesh in two places (same split as the 2-D
    // path): `CreateVTKMesh`'s unconditional `CheckElementOrientation(true)`
    // for linear meshes, and — for quadratic meshes only — the
    // knob-controlled `CheckElementOrientation(fix_orientation)` inside
    // `Mesh::Finalize`.  Then `MarkForRefinement` gated on the `refine` knob.
    // Tetrahedral meshes mark (`meshgen` bit 1) — elements and boundary
    // triangles via `mark_tet_mesh_for_refinement`; quadratic tet rows follow
    // the corner rotation (MFEM's `PrepareNodeReorder`/`DoNodeReorder` keeps
    // `Nodes` attached to the rotated frame; the writer re-derives the global
    // dof numbering from the topology, so only the per-element row permutation
    // is needed here).  The trailing `CheckBdrElementOrientation()` is an
    // identity in this pipeline: the boundary triangles carry the same
    // first-encounter owner cycles as the face table (and, when marked, were
    // rotated along with the elements).
    if quadratic {
        attach_geometry(&mut mesh, geo_rows, &extra_coords);
        if fix_orientation {
            fix_curved_orientation(&mut mesh);
            mesh.check_element_orientation(true);
        }
    } else {
        mesh.check_element_orientation(true);
    }
    if refine && fams.iter().any(|&f| f == Family::Tet) {
        let old_corners: Vec<[u32; 4]> = (0..n_elems as u32)
            .map(|e| {
                let mut c = [0u32; 4];
                c.copy_from_slice(elem_slice(
                    &mesh.conn,
                    e,
                    uniform,
                    npe,
                    mesh.elem_offsets.as_ref(),
                ));
                c
            })
            .collect();
        mark_tet_mesh_for_refinement(&mut mesh);
        if quadratic {
            for (e, old) in old_corners.iter().enumerate() {
                let new: [u32; 4] = {
                    let c = elem_slice(
                        &mesh.conn,
                        e as u32,
                        uniform,
                        npe,
                        mesh.elem_offsets.as_ref(),
                    );
                    [c[0], c[1], c[2], c[3]]
                };
                if new != *old {
                    let mut sigma = [0usize; 4];
                    for (i, &nv) in new.iter().enumerate() {
                        sigma[i] = old.iter().position(|&ov| ov == nv).unwrap_or_else(|| {
                            panic!("vtk: tet {e} corner {nv} lost by the marking rotation")
                        });
                    }
                    rotate_row_in_place(&mut mesh, e, &RowRotation::Tet(&sigma));
                }
            }
        }
    }
    Ok(mesh)
}

fn read_vtk_mesh_str(text: &str, refine: bool, fix_orientation: bool) -> FemResult<VtkMeshFile> {
    let cells = build_cells(parse_vtk(text)?)?;
    match cells.space_dim {
        2 => Ok(VtkMeshFile {
            mesh2d: Some(build_mesh_2d(cells, refine, fix_orientation)?),
            mesh3d: None,
        }),
        3 => Ok(VtkMeshFile {
            mesh2d: None,
            mesh3d: Some(build_mesh_3d(cells, refine, fix_orientation)?),
        }),
        other => Err(vtk_err(format!("internal: space dimension {other}"))),
    }
}
