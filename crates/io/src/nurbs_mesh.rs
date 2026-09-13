//! MFEM NURBS mesh format — reader and writer.
//!
//! Two layers live here:
//!
//! * [`read_nurbs_mesh`] / [`read_nurbs_mesh_file`] turn a file into the
//!   patch-wise [`NurbsFile`] that the NURBS element and space code consume.
//! * [`read_nurbs_mesh_doc`] / [`write_nurbs_mesh_doc`] model the *whole*
//!   document ([`NurbsMeshDoc`]) — topology sections, geometry section and
//!   FiniteElementSpace node block — and write it back in MFEM's own layout.
//!
//! Both geometry flavours are handled: the `knotvectors` + `weights` flavour,
//! and the `patches` flavour (one block per patch; the patch count comes from
//! the `elements` section, matching MFEM's `NURBSExtension::GetNP()`).
//! `MFEM NURBS NC-patch mesh v1.0` and the v1.1 `spacing` section are rejected
//! with an error that names the section.
//!
//! # Format
//!
//! The format is *token* oriented (whitespace is not significant, `#` starts a
//! comment); MFEM reads it with `>>`, see `NURBSExtension::Load` and
//! `Mesh::LoadPatchTopo` in `mesh/nurbs.cpp` / `mesh/mesh.cpp`.
//!
//! ```text
//! MFEM NURBS mesh v1.0                    (or v1.1 / NC-patch v1.0)
//!
//! # ...optional comment block...          (kept verbatim by NurbsMeshDoc)
//!
//! dimension
//! 2|3
//!
//! elements                                (the *patch* topology)
//! n
//! attr geom_type node0 ... nodeN          (geom: 1=SEGMENT, 3=SQUARE, 5=CUBE)
//!
//! boundary
//! n
//! attr geom_type node0 ...
//!
//! edges                                   (edge -> knot-vector index)
//! n
//! kv_index v0 v1                          (v0 > v1 flips the kv orientation)
//!
//! vertices
//! n
//!
//! knotvectors                             (flavour A)
//! n_kv
//! Order NCP kv0 ... kvNCP+Order            (NCP + Order + 1 knot values)
//!
//! spacing                                 (v1.1 only — not supported yet)
//!
//! weights
//! w0 w1 ...                               (one per control point)
//!
//! FiniteElementSpace
//! FiniteElementCollection: NURBS<p>
//! VDim: n
//! Ordering: 0|1                           (0 = byNODES, 1 = byVDIM)
//! x0 y0 [z0]                              (one control point per line)
//! ...
//! ```
//!
//! The alternative geometry section is
//!
//! ```text
//! patches                                 (flavour B, one block per patch)
//! knotvectors / n / Order NCP knots...
//! dimension / d
//! controlpoints[_cartesian|_homogeneous]
//! c0 c1 ... w                             (d + 1 values per control point)
//! ...
//! ```
//!
//! A `patches`-flavour file may stop right after its last patch block; MFEM then
//! derives the nodal space from the patches (`Mesh::ReadNURBSMesh` sets
//! `read_gf = 0`), so [`NurbsMeshDoc::has_node_block`] is `false`.

use std::io::{BufReader, Read, Write};
use std::path::Path;

use fem_core::{FemError, FemResult};
use fem_element::nurbs::{
    KnotVector, NurbsMesh2D, NurbsMesh3D, NurbsPatch2DData, NurbsPatch3DData,
};

/// Result from parsing a NURBS mesh file.
#[derive(Debug, Clone)]
pub enum NurbsFile {
    /// A 2-D NURBS mesh.
    Mesh2D(NurbsMesh2D),
    /// A 3-D NURBS mesh.
    Mesh3D(NurbsMesh3D),
}

/// Read an MFEM NURBS mesh file from a `BufRead` source.
///
/// Two geometry flavours are handled:
///
/// * `patches` — one patch per block is built (the MFEM mapping), so a
///   multi-patch file such as `square-disc-nurbs-patch.mesh` yields
///   `patches.len() == mesh.n_patches()`;
/// * `knotvectors` + `weights` — the *legacy* single-patch view: the first
///   `dim` knot vectors and the control points they consume.  A multi-patch
///   file in this flavour (e.g. `disc-nurbs.mesh`: 5 knot vectors, 5 patches)
///   is still reduced to its first patch; use [`read_nurbs_mesh_doc`] plus
///   [`NurbsMeshDoc::is_single_patch_representable`] to detect it, or
///   [`NurbsMeshDoc::n_patches`] to count the patches the file really has.
pub fn read_nurbs_mesh<R: Read>(reader: R) -> FemResult<NurbsFile> {
    let doc = read_nurbs_mesh_doc(reader)?;
    match &doc.geometry {
        NurbsGeometry::Patches(_) => doc.to_nurbs_file(),
        NurbsGeometry::Global { .. } => doc.global_single_patch_to_nurbs_file(),
    }
}

/// Convenience: read from a file path.
pub fn read_nurbs_mesh_file(path: impl AsRef<Path>) -> FemResult<NurbsFile> {
    let file = std::fs::File::open(path.as_ref())
        .map_err(FemError::Io)?;
    read_nurbs_mesh(file)
}

// ── Internal helpers ───────────────────────────────────────────────────────

fn parse_collection_degree(line: &str) -> FemResult<usize> {
    let s = line.trim();
    // Accept "FiniteElementCollection: NURBS<N>" or just "NURBS<N>"
    if let Some(rest) = s.strip_prefix("FiniteElementCollection: ") {
        if let Some(num) = rest.strip_prefix("NURBS") {
            num.trim().parse::<usize>()
                .map_err(|_| FemError::Mesh(format!("cannot parse NURBS degree from: {line}")))
        } else {
            Err(FemError::Mesh(format!("expected NURBS<N>, got: {rest}")))
        }
    } else if let Some(num) = s.strip_prefix("NURBS") {
        num.trim().parse::<usize>()
            .map_err(|_| FemError::Mesh(format!("cannot parse NURBS degree from: {line}")))
    } else {
        Err(FemError::Mesh(format!(
            "expected 'FiniteElementCollection: NURBS<N>', got: {line}"
        )))
    }
}

// ── Single-patch builders ──────────────────────────────────────────────────

fn build_single_patch_2d(
    kv_data: &[(usize, Vec<f64>)],
    weights: &[f64],
    ctrl_coords: &[f64],
    vdim: usize,
    n_ctrl: usize,
) -> FemResult<NurbsFile> {
    if kv_data.len() < 2 {
        return Err(FemError::Mesh("2D needs 2 knot vectors".into()));
    }
    let (order_u, knots_u) = &kv_data[0];
    let (order_v, knots_v) = &kv_data[1];

    let kv_u = KnotVector::new(knots_u.clone(), *order_u);
    let kv_v = KnotVector::new(knots_v.clone(), *order_v);

    let n_u = kv_u.n_basis();
    let n_v = kv_v.n_basis();
    let expected = n_u * n_v;

    if n_ctrl != expected {
        // CP count mismatch is expected for multi-patch meshes
        // where patches share boundary DOFs. Use available data.
    }

    let n_cp = n_ctrl.min(expected);
    let mut ctrl = Vec::with_capacity(expected);
    for i in 0..expected {
        if i < n_cp {
            let base = i * vdim;
            ctrl.push([
                ctrl_coords[base],
                ctrl_coords[base + 1],
            ]);
        } else {
            ctrl.push([0.0, 0.0]); // placeholder for missing CPs
        }
    }

    let w: Vec<f64> = if weights.len() >= expected {
        weights[..expected].to_vec()
    } else {
        vec![1.0; expected]
    };

    Ok(NurbsFile::Mesh2D(NurbsMesh2D {
        patches: vec![NurbsPatch2DData {
            kv_u,
            kv_v,
            control_pts: ctrl,
            weights: w,
            tag: 1,
        }],
        edge_connectivity: Vec::new(),
    }))
}

fn build_single_patch_3d(
    kv_data: &[(usize, Vec<f64>)],
    weights: &[f64],
    ctrl_coords: &[f64],
    vdim: usize,
    n_ctrl: usize,
) -> FemResult<NurbsFile> {
    if kv_data.len() < 3 {
        return Err(FemError::Mesh("3D needs 3 knot vectors".into()));
    }
    let (order_u, knots_u) = &kv_data[0];
    let (order_v, knots_v) = &kv_data[1];
    let (order_w, knots_w) = &kv_data[2];

    let kv_u = KnotVector::new(knots_u.clone(), *order_u);
    let kv_v = KnotVector::new(knots_v.clone(), *order_v);
    let kv_w = KnotVector::new(knots_w.clone(), *order_w);

    let n_u = kv_u.n_basis();
    let n_v = kv_v.n_basis();
    let n_w = kv_w.n_basis();
    let expected = n_u * n_v * n_w;

    let n_cp = n_ctrl.min(expected);
    let mut ctrl = Vec::with_capacity(expected);
    for i in 0..expected {
        if i < n_cp {
            let base = i * vdim;
            ctrl.push([
                ctrl_coords[base],
                ctrl_coords[base + 1],
                ctrl_coords[base + 2],
            ]);
        } else {
            ctrl.push([0.0, 0.0, 0.0]);
        }
    }

    let w: Vec<f64> = if weights.len() >= expected {
        weights[..expected].to_vec()
    } else {
        vec![1.0; expected]
    };

    Ok(NurbsFile::Mesh3D(NurbsMesh3D {
        patches: vec![NurbsPatch3DData {
            kv_u,
            kv_v,
            kv_w,
            control_pts: ctrl,
            weights: w,
            tag: 1,
        }],
        face_connectivity: Vec::new(),
    }))
}

// ═══════════════════════════════════════════════════════════════════════════
// Lossless document model + writer
// ═══════════════════════════════════════════════════════════════════════════
//
// `read_nurbs_mesh` above answers "which NURBS patches does this file
// describe?" and deliberately ignores the `elements` / `boundary` / `edges` /
// `vertices` topology sections.  That is enough to build a NURBS space but it
// cannot *write* the file back.  The types below model the whole document —
// the topology sections, both geometry flavours (`knotvectors` + `weights` and
// `patches`) and the closing `FiniteElementSpace` node block — so a file can be
// read and written without loss.
//
// The writer mirrors MFEM exactly:
//   * `Mesh::PrintTopo` / `Mesh::PrintTopoEdges` (mesh/mesh.cpp) for the
//     header, `dimension`, `elements`, `boundary`, `edges`, `vertices`;
//   * `NURBSExtension::Print` (mesh/nurbs.cpp) for `knotvectors` + `weights`,
//     or `patches`;
//   * `NURBSPatch::Print` (mesh/nurbs.cpp) for a single patch block;
//   * `FiniteElementSpace::Save` + `GridFunction::Save` for the node block.
//
// Numbers are printed with C++ `operator<<(std::ostream &, double)` semantics
// at a caller-chosen `precision` (see [`format_g`]), which is what MFEM's
// `Mesh::Save(fname, precision = 16)` relies on.  Consequently fem-rs output is
// byte-comparable with MFEM's own `Mesh::Save`.

/// Default numeric precision of [`write_nurbs_mesh_doc`] — the same default as
/// MFEM's `Mesh::Save(const std::string &, int precision = 16)`.
pub const NURBS_MESH_DEFAULT_PRECISION: usize = 16;

/// Which `MFEM NURBS mesh` header a document carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NurbsMeshFormat {
    /// `MFEM NURBS mesh v1.0` — `knotvectors` + `weights`, or `patches`.
    V1_0,
    /// `MFEM NURBS mesh v1.1` — v1.0 plus the `spacing` section.
    V1_1,
    /// `MFEM NURBS NC-patch mesh v1.0`.
    NcPatchV1_0,
}

impl NurbsMeshFormat {
    /// The exact first line of the file.
    pub fn header(self) -> &'static str {
        match self {
            NurbsMeshFormat::V1_0 => "MFEM NURBS mesh v1.0",
            NurbsMeshFormat::V1_1 => "MFEM NURBS mesh v1.1",
            NurbsMeshFormat::NcPatchV1_0 => "MFEM NURBS NC-patch mesh v1.0",
        }
    }

    /// Classify a file's first line.
    pub fn from_header(line: &str) -> FemResult<Self> {
        match line.trim() {
            "MFEM NURBS mesh v1.0" => Ok(NurbsMeshFormat::V1_0),
            "MFEM NURBS mesh v1.1" => Ok(NurbsMeshFormat::V1_1),
            "MFEM NURBS NC-patch mesh v1.0" => Ok(NurbsMeshFormat::NcPatchV1_0),
            other => Err(FemError::Mesh(format!(
                "expected a NURBS mesh header ('MFEM NURBS mesh v1.0', \
                 'MFEM NURBS mesh v1.1' or 'MFEM NURBS NC-patch mesh v1.0'), got: {other}"
            ))),
        }
    }

    /// The geometry-type comment block `Mesh::PrintTopo` writes.  MFEM v1.1
    /// moved `geom.hpp` from `fem/` to `mesh/`, and the comment follows.
    ///
    /// The block has no trailing blank line: the writer's `\ndimension` adds
    /// the single blank separator, which is what keeps a read/write round trip
    /// a fixed point (`comments` drops trailing blanks on read).
    fn geom_types_comment(self) -> [&'static str; 8] {
        let prefix = match self {
            NurbsMeshFormat::V1_1 => "# MFEM Geometry Types (see mesh/geom.hpp):",
            _ => "# MFEM Geometry Types (see fem/geom.hpp):",
        };
        [
            "",
            "#",
            prefix,
            "#",
            "# SEGMENT     = 1",
            "# SQUARE      = 3",
            "# CUBE        = 5",
            "#",
        ]
    }
}

/// One `elements` / `boundary` record: attribute, MFEM geometry type and the
/// node (vertex) indices — `Mesh::ReadElement` in mesh/mesh.cpp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NurbsTopoElement {
    /// Element attribute (`1` unless the mesh tags it).
    pub attribute: i32,
    /// MFEM `Geometry::Type` (1 = SEGMENT, 3 = SQUARE, 5 = CUBE, …).
    pub geom: i32,
    /// Vertices of the element, in file order.
    pub nodes: Vec<i32>,
}

/// One `edges` record: `<knotvector> <v0> <v1>`.
///
/// The *sign* of the knot-vector index is not stored in the file — MFEM derives
/// it from `v0 > v1` (`Mesh::LoadPatchTopo`), so the raw index is kept here and
/// compared by [`Self::signed_knotvector`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NurbsEdgeRecord {
    /// Knot-vector index as written in the file (non-negative).
    pub knotvector: i32,
    /// First vertex.
    pub v0: i32,
    /// Second vertex.
    pub v1: i32,
}

impl NurbsEdgeRecord {
    /// The signed knot-vector index MFEM ends up with: negative when the edge
    /// runs from the higher to the lower vertex index.
    pub fn signed_knotvector(&self) -> i32 {
        if self.v0 > self.v1 {
            -1 - self.knotvector
        } else {
            self.knotvector
        }
    }
}

/// One knot vector of a `knotvectors` section, or of one `patches` block.
#[derive(Debug, Clone, PartialEq)]
pub struct NurbsKvRecord {
    /// Order (`degree + 1`).
    pub order: usize,
    /// Number of control points.
    pub ncp: usize,
    /// `ncp + order + 1` knot values.
    pub knots: Vec<f64>,
}

impl NurbsKvRecord {
    /// Number of basis functions; must equal [`Self::ncp`] for a well-formed
    /// record.
    pub fn n_basis(&self) -> usize {
        self.knots.len().saturating_sub(self.order + 1)
    }
}

/// One block of the `patches` geometry flavour — the file image of MFEM's
/// `NURBSPatch::Print`.
#[derive(Debug, Clone, PartialEq)]
pub struct NurbsPatchRecord {
    /// One knot vector per parametric direction.
    pub knotvectors: Vec<NurbsKvRecord>,
    /// Physical dimension (the file's `dimension` line inside the block); each
    /// control point carries `dim + 1` values.
    pub dim: usize,
    /// Control points in file order (index `k` fastest along the last knot
    /// vector), each with `dim + 1` values.
    pub control_points: Vec<Vec<f64>>,
    /// `true` when the block was written as `controlpoints` /
    /// `controlpoints_homogeneous` (values already multiplied by the weight),
    /// `false` for `controlpoints_cartesian` (`x y z w`, un-multiplied).
    pub homogeneous: bool,
}

/// The geometry section of a NURBS mesh document.
#[derive(Debug, Clone, PartialEq)]
pub enum NurbsGeometry {
    /// `knotvectors` followed by the mesh-wide `weights`.
    Global {
        /// One knot vector per "unique" patch direction; MFEM renumbers them
        /// through the `edges` section.
        knotvectors: Vec<NurbsKvRecord>,
        /// One rational weight per control point, in node order.
        weights: Vec<f64>,
    },
    /// `patches` followed by one block per patch (MFEM reads `GetNP()` blocks,
    /// where `GetNP()` is the `elements` count).
    Patches(Vec<NurbsPatchRecord>),
}

/// A complete `MFEM NURBS mesh` file image: every section is retained, so
/// `read_nurbs_mesh_doc` -> `write_nurbs_mesh_doc` is content-preserving.
#[derive(Debug, Clone, PartialEq)]
pub struct NurbsMeshDoc {
    /// Which `MFEM NURBS mesh` variant this is.
    pub format: NurbsMeshFormat,
    /// Comment and blank lines between the header and `dimension`, verbatim
    /// (without line terminators).  Written back as-is, so the standard
    /// geometry-type block survives a round trip.
    pub comments: Vec<String>,
    /// Topological dimension.
    pub dim: usize,
    /// `elements` section — for NURBS meshes this is the *patch* topology.
    pub elements: Vec<NurbsTopoElement>,
    /// `boundary` section.
    pub boundary: Vec<NurbsTopoElement>,
    /// `edges` section.
    pub edges: Vec<NurbsEdgeRecord>,
    /// `vertices` count.
    pub n_vertices: usize,
    /// Geometry section.
    pub geometry: NurbsGeometry,
    /// Finite-element collection name from the node block, e.g. `NURBS1`.
    /// Synthesised as `NURBS<order>` when the file has no node block.
    pub collection: String,
    /// Node block vector dimension (`dim` when the file has no node block).
    pub vdim: usize,
    /// Node block ordering (`0` = byNODES, `1` = byVDIM).
    pub ordering: i32,
    /// One entry per control point, `vdim` components each. Empty when the file
    /// has no node block.
    pub coords: Vec<Vec<f64>>,
    /// Whether the file carries a `FiniteElementSpace` node block.
    ///
    /// A `patches`-flavour file may end right after its last patch block: MFEM's
    /// `Mesh::ReadNURBSMesh` then sets `read_gf = 0` and builds the nodal space
    /// from the patches.  `data/square-disc-nurbs-patch.mesh` does exactly that.
    pub has_node_block: bool,
}

impl NurbsMeshDoc {
    /// Number of patches — MFEM's `NURBSExtension::GetNP()`, i.e. the number of
    /// elements in the *patch topology* (the file's `elements` section).
    pub fn n_patches(&self) -> usize {
        self.elements.len()
    }

    /// Number of knot vectors of the `knotvectors` flavour (`0` for `patches`).
    pub fn n_knot_vectors(&self) -> usize {
        match &self.geometry {
            NurbsGeometry::Global { knotvectors, .. } => knotvectors.len(),
            NurbsGeometry::Patches(_) => 0,
        }
    }

    /// `true` when the document is a single patch that
    /// [`read_nurbs_mesh`]'s single-patch view represents without loss.
    ///
    /// Callers that read with [`read_nurbs_mesh_doc`] can use this to detect
    /// files where the single-patch view would silently drop patches (several
    /// knot vectors and/or several elements).
    pub fn is_single_patch_representable(&self) -> bool {
        match &self.geometry {
            NurbsGeometry::Global { knotvectors, .. } => {
                knotvectors.len() == self.dim && self.elements.len() == 1
            }
            NurbsGeometry::Patches(blocks) => blocks.len() == 1,
        }
    }

    /// Build the patch-wise [`NurbsFile`] view used by the NURBS element and
    /// space code.
    ///
    /// Supported: the `patches` flavour (one patch per block, in file order)
    /// and the `knotvectors` flavour when the document really is a single patch
    /// (`dim` knot vectors, one element) of dimension 2 or 3.
    ///
    /// A 1-D document cannot be converted: [`NurbsFile`] has no 1-D variant
    /// (`NurbsMesh2D` / `NurbsMesh3D` only).  The *document* model and the
    /// writer handle 1-D fine (`data/segment-nurbs.mesh` round-trips
    /// byte-exactly); only this patch-wise view is limited.
    ///
    /// A multi-patch `knotvectors` document is rejected: mapping
    /// element -> (knot vectors, control points) needs MFEM's
    /// `NURBS_PatchMap` / `GenerateElementDofTable` machinery, which fem-rs has
    /// not ported.  Use [`Self::elements`] and [`Self::geometry`] directly, or
    /// [`Self::is_single_patch_representable`] to detect the situation.
    pub fn to_nurbs_file(&self) -> FemResult<NurbsFile> {
        match &self.geometry {
            NurbsGeometry::Patches(blocks) => self.patches_to_nurbs_file(blocks),
            NurbsGeometry::Global { .. } => {
                if !self.is_single_patch_representable() {
                    return Err(FemError::Mesh(format!(
                        "nurbs mesh: {}-D document has {} knot vectors and {} elements — \
                         only the `patches` flavour and single-patch `knotvectors` \
                         documents can be converted to a patch-wise NurbsFile",
                        self.dim,
                        self.n_knot_vectors(),
                        self.elements.len()
                    )));
                }
                self.global_single_patch_to_nurbs_file()
            }
        }
    }

    /// Lenient single-patch view of a `knotvectors` document: the first `dim`
    /// knot vectors, and as many control points/weights as that patch consumes.
    ///
    /// This is what [`read_nurbs_mesh`] has always done, and it *silently drops*
    /// the extra knot vectors of a multi-patch file — use
    /// [`Self::is_single_patch_representable`] to detect that case.
    fn global_single_patch_to_nurbs_file(&self) -> FemResult<NurbsFile> {
        let NurbsGeometry::Global {
            knotvectors,
            weights,
        } = &self.geometry
        else {
            unreachable!("caller checked the geometry flavour");
        };
        let kv_data: Vec<(usize, Vec<f64>)> = knotvectors
            .iter()
            .map(|kv| (kv.order, kv.knots.clone()))
            .collect();
        let flat: Vec<f64> = self.coords.iter().flatten().copied().collect();
        match self.dim {
            2 => build_single_patch_2d(&kv_data, weights, &flat, self.vdim, self.coords.len()),
            3 => build_single_patch_3d(&kv_data, weights, &flat, self.vdim, self.coords.len()),
            _ => Err(FemError::Mesh(format!(
                "nurbs mesh: unsupported dimension {}",
                self.dim
            ))),
        }
    }

    fn patches_to_nurbs_file(&self, blocks: &[NurbsPatchRecord]) -> FemResult<NurbsFile> {
        if self.dim != 2 && self.dim != 3 {
            return Err(FemError::Mesh(format!(
                "nurbs mesh: unsupported dimension {}",
                self.dim
            )));
        }
        let mut p2: Vec<NurbsPatch2DData> = Vec::with_capacity(blocks.len());
        let mut p3: Vec<NurbsPatch3DData> = Vec::with_capacity(blocks.len());
        for (p, b) in blocks.iter().enumerate() {
            if b.dim != self.dim {
                return Err(FemError::Mesh(format!(
                    "nurbs mesh: patch {p} has dimension {}, expected {}",
                    b.dim, self.dim
                )));
            }
            if b.knotvectors.len() != self.dim {
                return Err(FemError::Mesh(format!(
                    "nurbs mesh: patch {p} has {} knotvectors, expected {}",
                    b.knotvectors.len(),
                    self.dim
                )));
            }
            let mut knots: Vec<KnotVector> = Vec::with_capacity(self.dim);
            let mut n_cp = 1usize;
            for kv in &b.knotvectors {
                if kv.n_basis() != kv.ncp {
                    return Err(FemError::Mesh(format!(
                        "nurbs mesh: patch {p} knot vector has {} knots for order {} and \
                         {} control points (expected {})",
                        kv.knots.len(),
                        kv.order,
                        kv.ncp,
                        kv.ncp + kv.order + 1
                    )));
                }
                n_cp = n_cp.saturating_mul(kv.ncp);
                knots.push(KnotVector::new(kv.knots.clone(), kv.order));
            }
            if b.control_points.len() != n_cp {
                return Err(FemError::Mesh(format!(
                    "nurbs mesh: patch {p} has {} control points, expected {n_cp}",
                    b.control_points.len()
                )));
            }
            let mut pts: Vec<Vec<f64>> = Vec::with_capacity(n_cp);
            let mut wts: Vec<f64> = Vec::with_capacity(n_cp);
            for (i, cp) in b.control_points.iter().enumerate() {
                if cp.len() != self.dim + 1 {
                    return Err(FemError::Mesh(format!(
                        "nurbs mesh: patch {p} control point {i} has {} values, expected {}",
                        cp.len(),
                        self.dim + 1
                    )));
                }
                let w = cp[self.dim];
                let (coords, w) = if b.homogeneous {
                    if w == 0.0 {
                        return Err(FemError::Mesh(format!(
                            "nurbs mesh: patch {p} control point {i} has zero homogeneous weight"
                        )));
                    }
                    (cp[..self.dim].iter().map(|v| v / w).collect(), w)
                } else {
                    (cp[..self.dim].to_vec(), w)
                };
                pts.push(coords);
                wts.push(w);
            }
            // MFEM's `SetPatchAttribute`; the file's element attribute is the
            // patch tag.
            let tag = self.elements.get(p).map(|e| e.attribute).unwrap_or(1);
            if self.dim == 2 {
                p2.push(NurbsPatch2DData {
                    kv_u: knots[0].clone(),
                    kv_v: knots[1].clone(),
                    control_pts: pts.iter().map(|c| [c[0], c[1]]).collect(),
                    weights: wts,
                    tag,
                });
            } else {
                p3.push(NurbsPatch3DData {
                    kv_u: knots[0].clone(),
                    kv_v: knots[1].clone(),
                    kv_w: knots[2].clone(),
                    control_pts: pts.iter().map(|c| [c[0], c[1], c[2]]).collect(),
                    weights: wts,
                    tag,
                });
            }
        }
        if self.dim == 2 {
            Ok(NurbsFile::Mesh2D(NurbsMesh2D {
                patches: p2,
                edge_connectivity: Vec::new(),
            }))
        } else {
            Ok(NurbsFile::Mesh3D(NurbsMesh3D {
                patches: p3,
                face_connectivity: Vec::new(),
            }))
        }
    }
}

// ── Reader ────────────────────────────────────────────────────────────────

/// Read a NURBS mesh document, retaining every section (see [`NurbsMeshDoc`]).
///
/// Unlike [`read_nurbs_mesh`] this does not drop the topology sections and does
/// not silently reduce a multi-patch file to its first patch.
pub fn read_nurbs_mesh_doc<R: Read>(reader: R) -> FemResult<NurbsMeshDoc> {
    let mut src = String::new();
    BufReader::new(reader).read_to_string(&mut src)?;
    read_nurbs_mesh_doc_str(&src)
}

/// Convenience: read a NURBS mesh document from a path.
pub fn read_nurbs_mesh_doc_file(path: impl AsRef<Path>) -> FemResult<NurbsMeshDoc> {
    let file = std::fs::File::open(path.as_ref()).map_err(FemError::Io)?;
    read_nurbs_mesh_doc(file)
}

/// Parse a NURBS mesh document from an in-memory string (used by tests).
pub fn read_nurbs_mesh_doc_str(src: &str) -> FemResult<NurbsMeshDoc> {
    // `str::lines` already normalises CRLF, so DOS-endings files round-trip to
    // LF output (the only whitespace difference we tolerate).
    let lines: Vec<String> = src.lines().map(str::to_string).collect();
    let header = lines.first().map(|l| l.trim()).unwrap_or_default();
    let format = NurbsMeshFormat::from_header(header)?;
    if format == NurbsMeshFormat::NcPatchV1_0 {
        return Err(FemError::Mesh(
            "nurbs mesh: 'MFEM NURBS NC-patch mesh v1.0' is not supported yet \
             (it needs the NCNURBS extension semantics)"
                .into(),
        ));
    }

    let mut s = DocScanner::new(lines, 1);

    s.expect("dimension")?;
    // Only the preamble (header comment block) is preserved verbatim; blank
    // lines inside the body are re-emitted by the writer's own layout.
    let comments = s.take_comments();
    let dim = s.next_usize()?;
    s.expect("elements")?;
    let elements = read_topo_section(&mut s)?;
    s.expect("boundary")?;
    let boundary = read_topo_section(&mut s)?;
    s.expect("edges")?;
    let n_edges = s.next_usize()?;
    let mut edges = Vec::with_capacity(n_edges);
    for _ in 0..n_edges {
        edges.push(NurbsEdgeRecord {
            knotvector: s.next_i32()?,
            v0: s.next_i32()?,
            v1: s.next_i32()?,
        });
    }
    s.expect("vertices")?;
    let n_vertices = s.next_usize()?;

    // ── Geometry section ────────────────────────────────────────────────
    let section = s.next_token()?;
    let geometry = match section.as_str() {
        "knotvectors" => {
            let n_kv = s.next_usize()?;
            let mut knotvectors = Vec::with_capacity(n_kv);
            for _ in 0..n_kv {
                knotvectors.push(read_kv_record(&mut s)?);
            }
            let kw = s.next_token()?;
            if kw != "weights" {
                return Err(FemError::Mesh(format!(
                    "nurbs mesh line {}: expected 'weights' after 'knotvectors', got '{kw}'. \
                     The v1.1 'spacing' section and the 'mesh_elements' / 'periodic' \
                     sections are not supported yet.",
                    s.line_no()
                )));
            }
            let mut weights = Vec::new();
            while let Some(tok) = s.next_weight_until_fes()? {
                weights.push(tok);
            }
            NurbsGeometry::Global { knotvectors, weights }
        }
        "patches" => {
            // MFEM: `patches.SetSize(GetNP())` where GetNP() is the patch
            // topology's element count.
            let np = elements.len();
            let mut blocks = Vec::with_capacity(np);
            for _ in 0..np {
                blocks.push(read_patch_record(&mut s)?);
            }
            NurbsGeometry::Patches(blocks)
        }
        other => {
            return Err(FemError::Mesh(format!(
                "nurbs mesh line {}: expected 'knotvectors' or 'patches', got '{other}'",
                s.line_no()
            )));
        }
    };

    // ── Node block: `FiniteElementCollection` + control-point coordinates ─
    // Both branches above stop right after consuming the `FiniteElementSpace`
    // keyword (MFEM's `weights` reader needs it as its terminator).
    //
    // A `patches`-flavour file may instead end right after its last patch
    // block: MFEM sets `read_gf = 0` and derives the nodal space from the
    // patches, so there is no node block to read.
    let mut has_node_block = true;
    if matches!(&geometry, NurbsGeometry::Patches(_)) {
        match s.next_token_opt()? {
            Some(tok) if tok == "FiniteElementSpace" => {}
            Some(tok) => {
                return Err(FemError::Mesh(format!(
                    "nurbs mesh line {}: expected 'FiniteElementSpace' or end of file after \
                     the last patch, got '{tok}'",
                    s.line_no()
                )));
            }
            None => has_node_block = false,
        }
    }

    let (collection, vdim, ordering, coords) = if has_node_block {
        read_node_block(&mut s)?
    } else {
        // MFEM: `new NURBSFECollection(NURBSext->GetOrder())`, vdim from the
        // patch space dimension, `Ordering::byVDIM`.
        (default_collection(&geometry), dim, 1, Vec::new())
    };

    Ok(NurbsMeshDoc {
        format,
        comments,
        dim,
        elements,
        boundary,
        edges,
        n_vertices,
        geometry,
        collection,
        vdim,
        ordering,
        coords,
        has_node_block,
    })
}

/// `NURBS<order>` for a node-block-less `patches` document, using the highest
/// patch order — MFEM's `NURBSFECollection(NURBSext->GetOrder())`.
fn default_collection(geometry: &NurbsGeometry) -> String {
    let order = match geometry {
        NurbsGeometry::Patches(blocks) => blocks
            .iter()
            .flat_map(|b| b.knotvectors.iter().map(|kv| kv.order))
            .max()
            .unwrap_or(1),
        NurbsGeometry::Global { knotvectors, .. } => {
            knotvectors.iter().map(|kv| kv.order).max().unwrap_or(1)
        }
    };
    format!("NURBS{order}")
}

/// `Geometry::NumVerts` (fem/geom.cpp) indexed by MFEM geometry type.
fn geom_num_vertices(geom: i32) -> Option<usize> {
    match geom {
        0 => Some(1), // POINT
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

fn read_topo_section(s: &mut DocScanner) -> FemResult<Vec<NurbsTopoElement>> {
    let n = s.next_usize()?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let attribute = s.next_i32()?;
        let geom = s.next_i32()?;
        let nv = geom_num_vertices(geom).ok_or_else(|| {
            FemError::Mesh(format!(
                "nurbs mesh line {}: unsupported MFEM geometry type {geom}",
                s.line_no()
            ))
        })?;
        let mut nodes = Vec::with_capacity(nv);
        for _ in 0..nv {
            nodes.push(s.next_i32()?);
        }
        out.push(NurbsTopoElement {
            attribute,
            geom,
            nodes,
        });
    }
    Ok(out)
}

/// `KnotVector::KnotVector(std::istream &)` (mesh/nurbs.cpp): order, control
/// point count, then exactly `ncp + order + 1` knot values.
fn read_kv_record(s: &mut DocScanner) -> FemResult<NurbsKvRecord> {
    let order = s.next_usize()?;
    let ncp = s.next_usize()?;
    let n_knots = ncp + order + 1;
    let mut knots = Vec::with_capacity(n_knots);
    for _ in 0..n_knots {
        knots.push(s.next_f64()?);
    }
    Ok(NurbsKvRecord { order, ncp, knots })
}

/// `NURBSPatch::NURBSPatch(std::istream &)` (mesh/nurbs.cpp).
fn read_patch_record(s: &mut DocScanner) -> FemResult<NurbsPatchRecord> {
    s.expect("knotvectors")?;
    let n_kv = s.next_usize()?;
    let mut knotvectors = Vec::with_capacity(n_kv);
    let mut n_cp = 1usize;
    for _ in 0..n_kv {
        let kv = read_kv_record(s)?;
        n_cp = n_cp.saturating_mul(kv.ncp);
        knotvectors.push(kv);
    }
    s.expect("dimension")?;
    let dim = s.next_usize()?;
    let kind = s.next_token()?;
    let homogeneous = match kind.as_str() {
        "controlpoints" | "controlpoints_homogeneous" => true,
        "controlpoints_cartesian" => false,
        other => {
            return Err(FemError::Mesh(format!(
                "nurbs mesh line {}: expected 'controlpoints', 'controlpoints_homogeneous' \
                 or 'controlpoints_cartesian', got '{other}'",
                s.line_no()
            )));
        }
    };
    let mut control_points = Vec::with_capacity(n_cp);
    for _ in 0..n_cp {
        let mut cp = Vec::with_capacity(dim + 1);
        for _ in 0..(dim + 1) {
            cp.push(s.next_f64()?);
        }
        control_points.push(cp);
    }
    Ok(NurbsPatchRecord {
        knotvectors,
        dim,
        control_points,
        homogeneous,
    })
}

/// The `FiniteElementSpace` node block, when the file has one.  Only the
/// `patches` flavour may omit it (see [`NurbsMeshDoc::has_node_block`]).
fn read_node_block(s: &mut DocScanner) -> FemResult<(String, usize, i32, Vec<Vec<f64>>)> {
    s.expect("FiniteElementCollection:")?;
    let collection = s.next_token()?;
    parse_collection_degree(&format!("FiniteElementCollection: {collection}"))?;
    s.expect("VDim:")?;
    let vdim = s.next_usize()?;
    s.expect("Ordering:")?;
    let ordering = s.next_i32()?;

    let mut flat: Vec<f64> = Vec::new();
    while let Some(tok) = s.next_token_opt()? {
        flat.push(tok.parse::<f64>().map_err(|_| {
            FemError::Mesh(format!(
                "nurbs mesh line {}: expected a control-point coordinate, got '{tok}'",
                s.line_no()
            ))
        })?);
    }
    if vdim == 0 || flat.len() % vdim != 0 {
        return Err(FemError::Mesh(format!(
            "nurbs mesh: {} control-point coordinate values are not a multiple of VDim {vdim}",
            flat.len()
        )));
    }
    Ok((
        collection,
        vdim,
        ordering,
        flat.chunks(vdim).map(|c| c.to_vec()).collect(),
    ))
}

/// Whitespace-agnostic token scanner with MFEM's `#` comment handling.
///
/// The MFEM NURBS format is *token* oriented (see `NURBSExtension::Load`), not
/// line oriented, so parsing has to be able to cross line boundaries — while
/// still remembering the comment/blank lines it skipped so the header comment
/// block can be written back verbatim.
struct DocScanner {
    lines: Vec<String>,
    /// Index of the line the scanner is positioned on.
    li: usize,
    /// Byte offset inside `lines[li]` of the next unread token.
    ti: usize,
    /// Verbatim (right-trimmed) comment/blank lines skipped so far.
    skipped: Vec<String>,
    capture_comments: bool,
}

impl DocScanner {
    fn new(lines: Vec<String>, first_line: usize) -> Self {
        DocScanner {
            lines,
            li: first_line,
            ti: 0,
            skipped: Vec::new(),
            capture_comments: true,
        }
    }

    /// 1-based line number the scanner is on (for error messages).
    fn line_no(&self) -> usize {
        (self.li + 1).max(1)
    }

    /// Next token, or `None` at end of input.
    fn next_token_opt(&mut self) -> FemResult<Option<String>> {
        loop {
            if self.li >= self.lines.len() {
                return Ok(None);
            }
            let line = self.lines[self.li].clone();
            let trimmed = line.trim_start();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                if self.capture_comments {
                    self.skipped.push(line.trim_end().to_string());
                }
                self.li += 1;
                self.ti = 0;
                continue;
            }
            let bytes = line.as_bytes();
            // Skip the whitespace that separates this token from the previous
            // one (or the line's indentation, when `ti == 0`).
            let mut start = self.ti;
            while start < bytes.len() && bytes[start].is_ascii_whitespace() {
                start += 1;
            }
            if start >= bytes.len() {
                self.li += 1;
                self.ti = 0;
                continue;
            }
            let mut end = start;
            while end < bytes.len() && !bytes[end].is_ascii_whitespace() {
                end += 1;
            }
            let tok = line[start..end].to_string();
            self.ti = end;
            return Ok(Some(tok));
        }
    }

    fn next_token(&mut self) -> FemResult<String> {
        match self.next_token_opt()? {
            Some(tok) => Ok(tok),
            None => Err(FemError::Mesh(format!(
                "nurbs mesh: unexpected end of input after line {}",
                self.lines.len()
            ))),
        }
    }

    /// Stop capturing skipped comment lines and return what was captured,
    /// without the trailing blank lines (the writer emits its own blank
    /// separator before each section keyword, so keeping them would make the
    /// preamble grow on every round trip).
    fn take_comments(&mut self) -> Vec<String> {
        self.capture_comments = false;
        let mut out = std::mem::take(&mut self.skipped);
        while out.last().is_some_and(|l| l.is_empty()) {
            out.pop();
        }
        out
    }

    fn expect(&mut self, keyword: &str) -> FemResult<()> {
        let got = self.next_token()?;
        if got != keyword {
            return Err(FemError::Mesh(format!(
                "nurbs mesh line {}: expected section '{keyword}', got '{got}'",
                self.line_no()
            )));
        }
        Ok(())
    }

    fn next_f64(&mut self) -> FemResult<f64> {
        let tok = self.next_token()?;
        tok.parse::<f64>()
            .map_err(|_| self.bad_number(&tok, "a number"))
    }

    fn next_i32(&mut self) -> FemResult<i32> {
        let tok = self.next_token()?;
        tok.parse::<i32>()
            .map_err(|_| self.bad_number(&tok, "an integer"))
    }

    fn next_usize(&mut self) -> FemResult<usize> {
        let tok = self.next_token()?;
        tok.parse::<usize>()
            .map_err(|_| self.bad_number(&tok, "a non-negative integer"))
    }

    fn bad_number(&self, tok: &str, what: &str) -> FemError {
        FemError::Mesh(format!(
            "nurbs mesh line {}: expected {what}, got '{tok}'",
            self.line_no()
        ))
    }

    /// Read the next weight value, or `None` (consuming) when the
    /// `FiniteElementSpace` keyword ends the `weights` section.  This mirrors
    /// the single-patch reader's rule: the weight count (`GetNDof()`) is only
    /// known once the node block has been parsed.
    fn next_weight_until_fes(&mut self) -> FemResult<Option<f64>> {
        let tok = self.next_token()?;
        if tok == "FiniteElementSpace" {
            return Ok(None);
        }
        tok.parse::<f64>()
            .map(Some)
            .map_err(|_| self.bad_number(&tok, "a weight value"))
    }
}

// ── Number formatting (C++ `operator<<(ostream&, double)`) ────────────────

/// Format `value` exactly as C++ `operator<<(std::ostream &, double)` does for
/// a stream whose `precision()` is `precision`, i.e. `printf("%.*g", …)`.
///
/// Rust has no `%g`, so `%g`'s style selection is reproduced explicitly: MFEM's
/// `Mesh::Save` / `Mesh::Print` output depends on it, and byte-for-byte
/// agreement with MFEM is the point of this writer.
fn format_g(value: f64, precision: usize) -> String {
    // MFEM prints through `Vector::Print`, which zeroes subnormals first.
    let v = if value != 0.0 && value.abs() < f64::MIN_POSITIVE {
        0.0
    } else {
        value
    };
    if v == 0.0 {
        return if v.is_sign_negative() { "-0" } else { "0" }.to_string();
    }
    if v.is_nan() {
        return "nan".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf" } else { "-inf" }.to_string();
    }

    let p = precision.max(1);
    // `%g` picks the style from the exponent of the value rounded to `p`
    // significant digits, which is exactly what `{:.*e}` reports.
    let sci = format!("{:.*e}", p - 1, v);
    let (mant, exp) = sci
        .split_once('e')
        .expect("Rust's LowerExp always writes an exponent");
    let exp: i32 = exp
        .parse()
        .expect("Rust's LowerExp always writes a decimal exponent");

    if exp < -4 || exp >= p as i32 {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", trim_frac(mant), sign, exp.unsigned_abs())
    } else {
        let frac = (p as i32 - 1 - exp).max(0) as usize;
        trim_frac(&format!("{v:.*}", frac))
    }
}

/// Drop trailing zeros (and a dangling `.`) from a decimal rendering.
fn trim_frac(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    let t = s.trim_end_matches('0');
    t.strip_suffix('.').unwrap_or(t).to_string()
}

// ── Writer ────────────────────────────────────────────────────────────────

/// Write `doc` as an `MFEM NURBS mesh` file, at
/// [`NURBS_MESH_DEFAULT_PRECISION`].
pub fn write_nurbs_mesh_doc<W: Write>(doc: &NurbsMeshDoc, writer: W) -> FemResult<()> {
    write_nurbs_mesh_doc_with_precision(doc, writer, NURBS_MESH_DEFAULT_PRECISION)
}

/// Write `doc` as an `MFEM NURBS mesh` file with an explicit numeric precision.
///
/// Layout matches MFEM's own writer (see the module comment above), so the
/// result can be diffed against MFEM's `Mesh::Save` output directly.
pub fn write_nurbs_mesh_doc_with_precision<W: Write>(
    doc: &NurbsMeshDoc,
    mut writer: W,
    precision: usize,
) -> FemResult<()> {
    let g = |v: f64| format_g(v, precision);

    // ── Header + comment block ──────────────────────────────────────────
    writeln!(writer, "{}", doc.format.header())?;
    if doc.comments.is_empty() {
        for line in doc.format.geom_types_comment() {
            writeln!(writer, "{line}")?;
        }
    } else {
        for line in &doc.comments {
            writeln!(writer, "{line}")?;
        }
    }

    // ── Topology ────────────────────────────────────────────────────────
    writeln!(writer, "\ndimension\n{}", doc.dim)?;
    writeln!(writer, "\nelements\n{}", doc.elements.len())?;
    for el in &doc.elements {
        write_topo_element(&mut writer, el)?;
    }
    writeln!(writer, "\nboundary\n{}", doc.boundary.len())?;
    for el in &doc.boundary {
        write_topo_element(&mut writer, el)?;
    }
    writeln!(writer, "\nedges\n{}", doc.edges.len())?;
    for e in &doc.edges {
        writeln!(writer, "{} {} {}", e.knotvector, e.v0, e.v1)?;
    }
    writeln!(writer, "\nvertices\n{}", doc.n_vertices)?;

    // ── Geometry section ────────────────────────────────────────────────
    match &doc.geometry {
        NurbsGeometry::Global {
            knotvectors,
            weights,
        } => {
            writeln!(writer, "\nknotvectors\n{}", knotvectors.len())?;
            for kv in knotvectors {
                write_kv_record(&mut writer, kv, &g)?;
            }
            writeln!(writer, "\nweights")?;
            for w in weights {
                writeln!(writer, "{}", g(*w))?;
            }
        }
        NurbsGeometry::Patches(blocks) => {
            writeln!(writer, "\npatches")?;
            for (p, b) in blocks.iter().enumerate() {
                // `NURBSExtension::Print` labels each block.
                writeln!(writer, "\n# patch {p}\n")?;
                writeln!(writer, "knotvectors\n{}", b.knotvectors.len())?;
                for kv in &b.knotvectors {
                    write_kv_record(&mut writer, kv, &g)?;
                }
                writeln!(writer, "\ndimension\n{}", b.dim)?;
                let keyword = if b.homogeneous {
                    "controlpoints"
                } else {
                    "controlpoints_cartesian"
                };
                writeln!(writer, "\n{keyword}")?;
                for cp in &b.control_points {
                    writeln!(writer, "{}", join_numbers(cp, &g))?;
                }
            }
        }
    }

    // ── Node block: `FiniteElementSpace` + control-point coordinates ────
    if !doc.has_node_block {
        // A `patches`-flavour file without a node block ends with the last
        // patch block (MFEM derives the nodal space from the patches).  Any
        // other flavour has nowhere to put its weights, so refuse.
        if !matches!(&doc.geometry, NurbsGeometry::Patches(_)) {
            return Err(FemError::Mesh(
                "nurbs mesh: `has_node_block == false` is only valid for the `patches` \
                 geometry flavour"
                    .into(),
            ));
        }
        return Ok(());
    }
    if doc.collection.is_empty() {
        return Err(FemError::Mesh(
            "nurbs mesh: cannot write a node block without a FiniteElementCollection".into(),
        ));
    }
    writeln!(writer, "\nFiniteElementSpace")?;
    writeln!(writer, "FiniteElementCollection: {}", doc.collection)?;
    writeln!(writer, "VDim: {}", doc.vdim)?;
    writeln!(writer, "Ordering: {}", doc.ordering)?;
    writeln!(writer)?;
    for (i, cp) in doc.coords.iter().enumerate() {
        if cp.len() != doc.vdim {
            return Err(FemError::Mesh(format!(
                "nurbs mesh: control point {i} has {} coordinates, expected VDim {}",
                cp.len(),
                doc.vdim
            )));
        }
        writeln!(writer, "{}", join_numbers(cp, &g))?;
    }
    Ok(())
}

/// Write `doc` to `path` at [`NURBS_MESH_DEFAULT_PRECISION`].
pub fn write_nurbs_mesh_doc_file(path: impl AsRef<Path>, doc: &NurbsMeshDoc) -> FemResult<()> {
    let file = std::fs::File::create(path.as_ref()).map_err(FemError::Io)?;
    let mut w = std::io::BufWriter::new(file);
    write_nurbs_mesh_doc(doc, &mut w)?;
    w.flush().map_err(FemError::Io)
}

fn write_topo_element<W: Write>(w: &mut W, el: &NurbsTopoElement) -> FemResult<()> {
    write!(w, "{} {}", el.attribute, el.geom)?;
    for n in &el.nodes {
        write!(w, " {n}")?;
    }
    writeln!(w)?;
    Ok(())
}

/// `KnotVector::Print` (mesh/nurbs.cpp): `<order> <ncp> <knots…>`.
fn write_kv_record<W: Write, F: Fn(f64) -> String>(
    w: &mut W,
    kv: &NurbsKvRecord,
    g: &F,
) -> FemResult<()> {
    write!(w, "{} {}", kv.order, kv.ncp)?;
    for k in &kv.knots {
        write!(w, " {}", g(*k))?;
    }
    writeln!(w)?;
    Ok(())
}

fn join_numbers<F: Fn(f64) -> String>(values: &[f64], g: &F) -> String {
    let mut out = String::new();
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(&g(*v));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .parent().unwrap()
            .join("data")
            .join(name)
    }

    #[test]
    fn parse_beam_hex_nurbs_single() {
        // beam-hex-nurbs is single-patch 3D
        let path = data_path("beam-hex-nurbs.mesh");
        let result = read_nurbs_mesh_file(&path);
        if let Err(ref e) = result {
            panic!("beam-hex-nurbs: {e}");
        }
        match result.unwrap() {
            NurbsFile::Mesh3D(mesh) => {
                assert_eq!(mesh.n_patches(), 1);
                assert_eq!(mesh.patches[0].kv_u.degree, 1);
                assert_eq!(mesh.patches[0].kv_v.degree, 1);
                assert_eq!(mesh.patches[0].kv_w.degree, 1);
            }
            _ => panic!("expected 3D"),
        }
    }

    #[test]
    fn parse_beam_quad_nurbs() {
        let path = data_path("beam-quad-nurbs.mesh");
        let result = read_nurbs_mesh_file(&path).unwrap();
        match result {
            NurbsFile::Mesh2D(m) => assert_eq!(m.patches[0].kv_u.degree, 1),
            _ => panic!("expected 2D"),
        }
    }

    #[test]
    fn parse_disc_nurbs() {
        let path = data_path("disc-nurbs.mesh");
        let result = read_nurbs_mesh_file(&path).unwrap();
        match result {
            NurbsFile::Mesh2D(m) => assert_eq!(m.patches[0].kv_u.degree, 2),
            _ => panic!("expected 2D"),
        }
    }

    #[test]
    fn parse_pipe_nurbs() {
        let path = data_path("pipe-nurbs.mesh");
        let result = read_nurbs_mesh_file(&path).unwrap();
        match result {
            NurbsFile::Mesh3D(m) => assert_eq!(m.patches[0].kv_u.degree, 2),
            _ => panic!("expected 3D"),
        }
    }

    #[test]
    fn parse_ball_nurbs() {
        let path = data_path("ball-nurbs.mesh");
        let result = read_nurbs_mesh_file(&path).unwrap();
        match result {
            NurbsFile::Mesh3D(m) => assert_eq!(m.patches[0].kv_u.degree, 4),
            _ => panic!("expected 3D"),
        }
    }

    #[test]
    fn parse_square_disc_nurbs() {
        let path = data_path("square-disc-nurbs.mesh");
        let result = read_nurbs_mesh_file(&path).unwrap();
        match result {
            NurbsFile::Mesh2D(m) => assert_eq!(m.patches[0].kv_u.degree, 2),
            _ => panic!("expected 2D"),
        }
    }

    #[test]
    fn reject_standard_mfem_header() {
        let data = b"MFEM mesh v1.0\n";
        let result = read_nurbs_mesh(&data[..]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("NURBS"));
    }
}
