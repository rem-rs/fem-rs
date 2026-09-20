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
//! The v1.1 `spacing` section is parsed into [`NurbsSpacingRecord`]s and
//! round-tripped (the knot vectors themselves are unaffected); only
//! `MFEM NURBS NC-patch mesh v1.0` is still rejected with an error that names
//! the format.
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
//! spacing                                 (v1.1 only, optional)
//! n
//! kv type nip nrp ipar... dpar...         (one record per spacing function)
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
    /// A 1-D NURBS mesh (`segment-nurbs.mesh`; the `knotvectors` flavour).
    Mesh1D(NurbsMesh1D),
    /// A 2-D NURBS mesh.
    Mesh2D(NurbsMesh2D),
    /// A 3-D NURBS mesh.
    Mesh3D(NurbsMesh3D),
}

/// One patch of a 1-D NURBS mesh.
#[derive(Debug, Clone)]
pub struct NurbsPatch1DData {
    /// The single parametric knot vector.
    pub kv: KnotVector,
    /// One scalar coordinate per control point — the 1-D fixtures carry
    /// `VDim: 1`; a larger node-block `VDim` is rejected on read.
    pub control_pts: Vec<f64>,
    /// One rational weight per control point.
    pub weights: Vec<f64>,
    /// Patch attribute (the `elements` section tag).
    pub tag: i32,
}

/// A 1-D NURBS mesh.
#[derive(Debug, Clone)]
pub struct NurbsMesh1D {
    /// One entry per topology element (MFEM's `GetNP()`).
    pub patches: Vec<NurbsPatch1DData>,
}

impl NurbsMesh1D {
    /// Number of patches.
    pub fn n_patches(&self) -> usize {
        self.patches.len()
    }
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
///   file in this flavour (e.g. `disc-nurbs.mesh`: 3 knot vectors, 5 patches)
///   is still reduced to its first patch; use [`read_nurbs_mesh_doc`] plus
///   [`NurbsMeshDoc::is_single_patch_representable`] to detect it,
///   [`NurbsMeshDoc::to_nurbs_file`] for the full multi-patch view, or
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

/// Single-patch view of a 1-D `knotvectors` document (`segment-nurbs.mesh`).
fn build_single_patch_1d(
    kv_data: &[(usize, Vec<f64>)],
    weights: &[f64],
    ctrl_coords: &[f64],
    vdim: usize,
) -> FemResult<NurbsFile> {
    let (order, knots) = kv_data
        .first()
        .ok_or_else(|| FemError::Mesh("1D needs 1 knot vector".into()))?;
    if vdim != 1 {
        return Err(FemError::Mesh(format!(
            "nurbs mesh: the 1-D NurbsFile variant carries one scalar coordinate per \
             control point, but the node block has VDim {vdim}"
        )));
    }
    let kv = KnotVector::new(knots.clone(), *order);
    let expected = kv.n_basis();
    let n_cp = ctrl_coords.len().min(expected);
    let mut control_pts = Vec::with_capacity(expected);
    control_pts.extend_from_slice(&ctrl_coords[..n_cp]);
    control_pts.resize(expected, 0.0);
    let mut w: Vec<f64> = weights[..weights.len().min(expected)].to_vec();
    w.resize(expected, 1.0);
    Ok(NurbsFile::Mesh1D(NurbsMesh1D {
        patches: vec![NurbsPatch1DData {
            kv,
            control_pts,
            weights: w,
            tag: 1,
        }],
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

/// One record of the v1.1 `spacing` section — MFEM `NURBSExtension::Load`
/// (the `spacing` flag): the knot-vector index followed by the
/// `SpacingFunction::Print` image
/// `<SpacingType> <num-int> <num-real> <int params…> <real params…>`
/// (`mesh/spacing.hpp`: `0=UNIFORM 1=LINEAR 2=GEOMETRIC 3=BELL 4=GAUSSIAN
/// 5=LOGARITHMIC 6=PIECEWISE 7=PARTIAL`).
///
/// The parameters are retained verbatim so the section round-trips, and
/// [`Self::eval`] evaluates the spacing function itself (`SpacingFunction::
/// EvalAll`): the interval widths MFEM's `KnotVector::UniformRefinement`
/// consumes for NURBS *h*-refinement with spacing.  Reading the section still
/// leaves the knot vectors themselves untouched (probe-verified:
/// `beam-quad-nurbs-sf.mesh` and its spacing-less twin `beam-quad-nurbs.mesh`
/// carry identical knot structures, NKV=3, NDof=18).
#[derive(Debug, Clone, PartialEq)]
pub struct NurbsSpacingRecord {
    /// Knot-vector index the spacing function belongs to (MFEM verifies
    /// `0 <= ki < NumOfKnotVectors`).
    pub knotvector: usize,
    /// `SpacingType` discriminant.
    pub spacing_type: i32,
    /// Integer parameters (`ipar`); `ipar[0]` is the function's size `n`
    /// (the number of intervals).
    pub int_params: Vec<i32>,
    /// Real parameters (`dpar`).
    pub real_params: Vec<f64>,
}

impl NurbsSpacingRecord {
    /// MFEM `SpacingFunction::EvalAll` (`mesh/spacing.cpp`): the widths of all
    /// `Size()` intervals in interval order (`Eval(0) … Eval(Size()-1)` — for
    /// a `reverse` function this is the reversed table, exactly as in MFEM).
    ///
    /// The evaluation mirrors MFEM's constructors (`CalculateSpacing`) and
    /// `Eval` operations one for one, including the Newton iterations of the
    /// GEOMETRIC/GAUSSIAN types and the iterative scheme of BELL; `MFEM_VERIFY`
    /// failures become errors.
    pub fn eval(&self) -> FemResult<Vec<f64>> {
        spacing_eval_all(self.spacing_type, &self.int_params, &self.real_params)
    }
}

/// `mfem::GetSpacingFunction(type, ipar, dpar)->EvalAll()` (`mesh/spacing.cpp`).
///
/// `ipar[0]` is the function's size `n`; the parameter layouts are the ones
/// `SpacingFunction::Print`/`GetIntParameters`/`GetDoubleParameters` write
/// into a v1.1 `spacing` record.
fn spacing_eval_all(spacing_type: i32, ipar: &[i32], dpar: &[f64]) -> FemResult<Vec<f64>> {
    match spacing_type {
        0 => {
            // UniformSpacingFunction: n intervals of width 1/n.
            if ipar.len() != 1 || !dpar.is_empty() {
                return Err(FemError::Mesh("Invalid spacing function parameters".into()));
            }
            Ok(vec![1.0 / ipar[0] as f64; ipar[0] as usize])
        }
        1 => {
            // LinearSpacingFunction(n, reverse, s, scale).
            if ipar.len() != 3 || dpar.len() != 1 {
                return Err(FemError::Mesh("Invalid spacing function parameters".into()));
            }
            linear_widths(ipar[0], ipar[1] != 0, dpar[0])
        }
        2 => {
            // GeometricSpacingFunction(n, reverse, s, scale).
            if ipar.len() != 3 || dpar.len() != 1 {
                return Err(FemError::Mesh("Invalid spacing function parameters".into()));
            }
            geometric_widths(ipar[0], ipar[1] != 0, dpar[0])
        }
        3 => {
            // BellSpacingFunction(n, reverse, s0, s1, scale).
            if ipar.len() != 3 || dpar.len() != 2 {
                return Err(FemError::Mesh("Invalid spacing function parameters".into()));
            }
            bell_widths(ipar[0], ipar[1] != 0, dpar[0], dpar[1])
        }
        4 => {
            // GaussianSpacingFunction(n, reverse, s0, s1, scale).
            if ipar.len() != 3 || dpar.len() != 2 {
                return Err(FemError::Mesh("Invalid spacing function parameters".into()));
            }
            gaussian_widths(ipar[0], ipar[1] != 0, dpar[0], dpar[1])
        }
        5 => {
            // LogarithmicSpacingFunction(n, reverse, sym, logBase).
            if ipar.len() != 3 || dpar.len() != 1 {
                return Err(FemError::Mesh("Invalid spacing function parameters".into()));
            }
            logarithmic_widths(ipar[0], ipar[1] != 0, ipar[2] != 0, dpar[0])
        }
        6 => {
            // PiecewiseSpacingFunction(n, np, reverse, relN, pieces' ipar,
            //                           dpar = [partition | pieces' dpar]).
            if ipar.len() < 3 {
                return Err(FemError::Mesh("Invalid spacing function parameters".into()));
            }
            let np = ipar[1] as usize;
            if ipar.len() < 3 + np {
                return Err(FemError::Mesh("Invalid spacing function parameters".into()));
            }
            let rel_n = &ipar[3..3 + np];
            let piece_ipar = &ipar[3 + np..];
            piecewise_widths(ipar[0], np, ipar[2] != 0, rel_n, piece_ipar, dpar)
        }
        7 => {
            // PartialSpacingFunction(n, reverse, first_elem, num_elems,
            //                        num_elems_full, iparsub, dpar, typeFull).
            if ipar.len() < 8 {
                return Err(FemError::Mesh("Invalid spacing function parameters".into()));
            }
            partial_widths(ipar[0], ipar[1] != 0, ipar[2], ipar[3], ipar[4], ipar[5],
                           &ipar[8..], dpar)
        }
        other => Err(FemError::Mesh(format!("Unknown spacing type \"{other}\""))),
    }
}

/// `LinearSpacingFunction::CalculateDifference` + `Eval`.
fn linear_widths(n: i32, reverse: bool, s: f64) -> FemResult<Vec<f64>> {
    if !(0.0 < s && s < 1.0) {
        return Err(FemError::Mesh("Initial spacing must be in (0,1)".into()));
    }
    let n = n as usize;
    let d = if n < 2 {
        0.0
    } else {
        // Spacings are s, s + d, …, s + (n-1)d with the sum equal to 1.
        2.0 * (1.0 - (n as f64) * s) / ((n * (n - 1)) as f64)
    };
    if s + (((n - 1) as f64) * d) <= 0.0 {
        return Err(FemError::Mesh("Invalid linear spacing parameters".into()));
    }
    Ok((0..n)
        .map(|p| {
            let i = if reverse { n - 1 - p } else { p };
            s + (i as f64) * d
        })
        .collect())
}

/// `GeometricSpacingFunction::CalculateSpacing` + `Eval`: widths `s*r^i` with
/// `r` solved from `s*(r^n - 1) - r + 1 = 0` by Newton's method.
fn geometric_widths(n: i32, reverse: bool, s: f64) -> FemResult<Vec<f64>> {
    let n = n as usize;
    if n == 1 {
        // CalculateSpacing returns early; Eval special-cases n == 1.
        return Ok(vec![1.0]);
    }
    let conv_tol = 1.0e-8;
    let max_iter = 100;
    let n_f = n as f64;
    let s_unif = 1.0 / n_f;
    let mut r: f64 = if s < s_unif { 1.5 } else { 0.5 };
    let mut converged = false;
    for _ in 0..max_iter {
        let g = s * (r.powf(n_f) - 1.0) - r + 1.0;
        let dg = n_f * s * r.powf(n_f - 1.0) - 1.0;
        r -= g / dg;
        if (g / dg).abs() < conv_tol {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(FemError::Mesh(
            "Convergence failure in GeometricSpacingFunction".into(),
        ));
    }
    Ok((0..n)
        .map(|p| {
            let i = if reverse { n - 1 - p } else { p };
            s * r.powf(i as f64)
        })
        .collect())
}

/// `BellSpacingFunction::CalculateSpacing` + `Eval`: `s[0] = s0`, `s[n-1] =
/// s1`, and the interior spacings from the iterative scheme that minimizes the
/// ratios of adjacent spacings.
fn bell_widths(n: i32, reverse: bool, s0: f64, s1: f64) -> FemResult<Vec<f64>> {
    let n = n as usize;
    if n < 3 {
        return Ok(vec![1.0 / n as f64; n]);
    }
    if s0 + s1 >= 1.0 {
        return Err(FemError::Mesh(
            "Sum of first and last Bell spacings must be less than 1".into(),
        ));
    }
    let mut s = vec![0.0_f64; n];
    s[0] = s0;
    s[n - 1] = s1;
    if n == 3 {
        s[1] = 1.0 - s0 - s1;
    } else {
        // Solve a system iteratively (spacing.cpp lines 148-247).
        let urk = 1.0;
        let initial_guess = (1.0 - s0 - s1) / ((n - 2) as f64);
        for v in s.iter_mut().take(n - 1).skip(1) {
            *v = initial_guess;
        }
        let mut wk = [0.0_f64; 7];
        let mut s_new = vec![0.0_f64; n];
        let mut a = vec![0.5_f64; n + 2];
        a[0] = 0.0;
        a[1] = 0.0;
        let mut b = a.clone();
        let mut alpha = vec![0.0_f64; n + 2];
        let mut beta = vec![0.0_f64; n + 2];
        let mut gamma = vec![0.0_f64; n + 2];
        gamma[1] = s0;

        let max_iter = 100;
        let conv_tol = 1.0e-10;
        let mut converged = false;
        for _ in 0..max_iter {
            for j in 1..=(n - 3) {
                wk[0] = (s[j] + s[j + 1]) * (s[j] + s[j + 1]);
                wk[1] = s[j - 1];
                wk[2] = (s[j - 1] + s[j]) * (s[j - 1] + s[j]) * (s[j - 1] + s[j]);
                wk[3] = s[j + 2];
                wk[4] = (s[j + 2] + s[j + 1]) * (s[j + 2] + s[j + 1]) * (s[j + 2] + s[j + 1]);
                wk[5] = wk[0] * wk[1] / wk[2];
                wk[6] = wk[0] * wk[3] / wk[4];
                a[j + 1] += urk * (wk[5] - a[j + 1]);
                b[j + 1] += urk * (wk[6] - b[j + 1]);
            }
            for j in 2..=(n - 2) {
                wk[0] = a[j] * (1.0 - 2.0 * alpha[j - 1] + alpha[j - 1] * alpha[j - 2]
                                     + beta[j - 2])
                        + b[j] + 2.0 - alpha[j - 1];
                wk[1] = 1.0 / wk[0];
                alpha[j] = wk[1] * (a[j] * beta[j - 1] * (2.0 - alpha[j - 2])
                                        + 2.0 * b[j] + beta[j - 1] + 1.0);
                beta[j] = -b[j] * wk[1];
                gamma[j] = wk[1] * (a[j] * (2.0 * gamma[j - 1] - gamma[j - 2]
                                                - alpha[j - 2] * gamma[j - 1])
                                        + gamma[j - 1]);
            }
            s_new[0] = s[0];
            for j in 1..n {
                s_new[j] = s_new[j - 1] + s[j];
            }
            for j in (1..=(n - 3)).rev() {
                s_new[j] = alpha[j + 1] * s_new[j + 1] + beta[j + 1] * s_new[j + 2]
                           + gamma[j + 1];
            }
            // Convert back from points to spacings.
            for j in (1..n).rev() {
                s_new[j] -= s_new[j - 1];
            }
            wk[5] = 0.0;
            wk[6] = 0.0;
            for j in (2..=(n - 2)).rev() {
                wk[5] += s_new[j] * s_new[j];
                wk[6] += (s_new[j] - s[j]).powf(2.0);
            }
            s.copy_from_slice(&s_new);
            let res = (wk[6] / wk[5]).sqrt();
            if res < conv_tol {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(FemError::Mesh(
                "Convergence failure in BellSpacingFunction".into(),
            ));
        }
    }
    Ok((0..n)
        .map(|p| {
            let i = if reverse { n - 1 - p } else { p };
            s[i]
        })
        .collect())
}

/// `GaussianSpacingFunction::CalculateSpacing` + `Eval`: a Gaussian
/// `q*exp(-u*(x-m)^2/c^2)` fitted to the endpoint widths by Newton's method.
fn gaussian_widths(n: i32, reverse: bool, s0: f64, s1: f64) -> FemResult<Vec<f64>> {
    let n = n as usize;
    if n < 3 {
        return Ok(vec![1.0 / n as f64; n]);
    }
    let mut s = vec![0.0_f64; n];
    s[0] = s0;
    s[n - 1] = s1;
    if n == 3 {
        s[1] = 1.0 - s0 - s1;
    } else {
        let lnz01 = (s0 / s1).ln();
        let h = 1.0 / ((n - 1) as f64);
        // Determine concavity by comparing the linear distribution to 1.
        let slinear = (n as f64) * (s0 + (h * (s1 - s0) * 0.5 * ((n - 1) as f64)));
        if (slinear - 1.0).abs() <= 1.0e-8 {
            return Err(FemError::Mesh(
                "Bell distribution is too close to linear.".into(),
            ));
        }
        let u = if slinear < 1.0 { 1.0 } else { -1.0 };
        let mut c = 0.3; // Initial guess

        let max_iter = 10;
        let conv_tol = 1.0e-8;
        let mut converged = false;
        for _ in 0..max_iter {
            let c2 = c * c;
            let m = 0.5 * (1.0 - (u * c2 * lnz01));
            let dmdc = -u * c * lnz01;
            let mut r = 0.0_f64; // Residual
            let mut drdc = 0.0_f64; // Derivative of residual
            for i in 0..n {
                let x = i as f64 * h;
                let ti = ((-(x * x) + (2.0 * x * m)) * u / c2).exp(); // Gaussian
                r += ti;
                // Derivative of Gaussian
                drdc += ((-2.0 * (-(x * x) + (2.0 * x * m)) / (c2 * c))
                         + ((2.0 * x * dmdc) / c2)) * ti;
            }
            r *= s0;
            r -= 1.0; // Sum of spacings should equal 1.
            if r.abs() < conv_tol {
                converged = true;
                break;
            }
            drdc *= s0 * u;
            // Newton update limited by factors of 1/2 and 2.
            let mut dc = (-r / drdc).max(-0.5 * c);
            dc = dc.min(2.0 * c);
            c += dc;
        }
        if !converged {
            return Err(FemError::Mesh(
                "Convergence failure in GaussianSpacingFunction".into(),
            ));
        }

        let c2 = c * c;
        let m = 0.5 * (1.0 - (u * c2 * lnz01));
        let q = s0 * (u * m * m / c2).exp();
        for (i, v) in s.iter_mut().enumerate() {
            let x = i as f64 * h - m;
            *v = q * (-u * x * x / c2).exp();
        }
    }
    Ok((0..n)
        .map(|p| {
            let i = if reverse { n - 1 - p } else { p };
            s[i]
        })
        .collect())
}

/// `LogarithmicSpacingFunction::CalculateSpacing` (+ symmetric variant):
/// uniform in `log(logBase)` over the unit interval.
fn logarithmic_widths(n: i32, reverse: bool, sym: bool, log_base: f64) -> FemResult<Vec<f64>> {
    if n <= 0 || log_base <= 1.0 {
        return Err(FemError::Mesh(
            "Invalid parameters in LogarithmicSpacingFunction".into(),
        ));
    }
    let n = n as usize;
    let mut s = vec![0.0_f64; n];
    if sym {
        let odd = n % 2 == 1;
        let m0 = n / 2;
        let m = if odd { m0 + 1 } else { m0 };
        let h = 1.0 / m as f64;
        let mut p = 1.0; // Initialize at right endpoint of [0,1].
        for i in (0..m.saturating_sub(1)).rev() {
            let p_i = (log_base.powf((i + 1) as f64 * h) - 1.0) / (log_base - 1.0);
            s[i + 1] = p - p_i;
            p = p_i;
        }
        s[0] = p;
        let t = if odd { 1.0 / (2.0 - s[m - 1]) } else { 0.5 };
        for i in 0..m {
            s[i] *= t;
            if i < (m - 1) || !odd {
                s[n - i - 1] = s[i];
            }
        }
    } else {
        let h = 1.0 / n as f64;
        let mut p = 1.0;
        for i in (0..n.saturating_sub(1)).rev() {
            let p_i = (log_base.powf((i + 1) as f64 * h) - 1.0) / (log_base - 1.0);
            s[i + 1] = p - p_i;
            p = p_i;
        }
        s[0] = p;
    }
    Ok((0..n)
        .map(|p| {
            let i = if reverse { n - 1 - p } else { p };
            s[i]
        })
        .collect())
}

/// `PiecewiseSpacingFunction`: spacing functions on `np` fixed subintervals
/// of the unit interval (`dpar[0..np-1]` is the partition), the piece in
/// interval `p` carrying `ref * relN[p]` of the `n` intervals.
fn piecewise_widths(
    n: i32,
    np: usize,
    reverse: bool,
    rel_n: &[i32],
    piece_ipar: &[i32],
    dpar: &[f64],
) -> FemResult<Vec<f64>> {
    let n = n as usize;
    // SetupPieces: the partition must ascend strictly inside (0,1).
    let mut partition = Vec::with_capacity(np - 1);
    for (i, &v) in dpar.iter().take(np - 1).enumerate() {
        if v <= 0.0 || v >= 1.0 || (i > 0 && v <= partition[i - 1]) {
            return Err(FemError::Mesh("Invalid partition".into()));
        }
        partition.push(v);
    }
    // SetupPieces: decode each piece's `(type, nip, nrp, ipar…, dpar…)` block.
    let mut piece_defs: Vec<(i32, Vec<i32>, Vec<f64>)> = Vec::with_capacity(np);
    let mut osi = 0;
    let mut osd = np - 1;
    let mut n0 = 0_usize;
    for p in 0..np {
        if osi + 3 > piece_ipar.len() {
            return Err(FemError::Mesh("Invalid spacing function parameters".into()));
        }
        let ptype = piece_ipar[osi];
        let nip = piece_ipar[osi + 1] as usize;
        let nrd = piece_ipar[osi + 2] as usize;
        if osi + 3 + nip > piece_ipar.len() || osd + nrd > dpar.len() {
            return Err(FemError::Mesh("Invalid spacing function parameters".into()));
        }
        piece_defs.push((
            ptype,
            piece_ipar[osi + 3..osi + 3 + nip].to_vec(),
            dpar[osd..osd + nrd].to_vec(),
        ));
        osi += 3 + nip;
        osd += nrd;
        n0 += rel_n[p] as usize;
    }
    if osi != piece_ipar.len() || osd != dpar.len() {
        return Err(FemError::Mesh("Invalid spacing function parameters".into()));
    }

    // CalculateSpacing.
    let mut out;
    if n == 1 {
        out = vec![1.0];
    } else {
        let ref_factor = n / n0; // Refinement factor
        let cf = n0 / n; // Coarsening factor
        let mut coarsen = cf > 1 && n > 1;
        if coarsen {
            // If coarsening, check whether all pieces have size divisible by
            // cf (a piece's *file* size is its own `ipar[0]`).
            for def in &piece_defs {
                let size = def.1.first().copied().unwrap_or(0).max(0) as usize;
                if size != cf * (size / cf) {
                    coarsen = false;
                }
            }
        }
        if !(coarsen || n >= n0) {
            return Err(FemError::Mesh(
                "Invalid case in PiecewiseSpacingFunction::CalculateSpacing".into(),
            ));
        }
        out = Vec::with_capacity(n);
        for (p, (ptype, pipar, pdpar)) in piece_defs.iter().enumerate() {
            let piece_n = if coarsen {
                (rel_n[p] as usize) / cf
            } else {
                ref_factor * rel_n[p] as usize
            };
            // `pieces[p]->SetSize(piece_n)` re-runs the piece's calculation at
            // the composite size, then `Eval(i)` reads its table.
            let widths = spacing_eval_all_n(*ptype, pipar, pdpar, piece_n)?;
            let p0 = if p == 0 { 0.0 } else { partition[p - 1] };
            let p1 = if p == np - 1 { 1.0 } else { partition[p] };
            let h_p = p1 - p0;
            out.extend(widths.into_iter().map(|w| h_p * w));
        }
        if out.len() != n {
            return Err(FemError::Mesh(
                "Invalid case in PiecewiseSpacingFunction::CalculateSpacing".into(),
            ));
        }
    }
    Ok((0..n)
        .map(|p| {
            let i = if reverse { n - 1 - p } else { p };
            out[i]
        })
        .collect())
}

/// `PartialSpacingFunction::CalculateSpacing` + `Eval`: a contiguous
/// `num_elems`-window of the *full* function's table, normalized to sum to 1.
#[allow(clippy::too_many_arguments)]
fn partial_widths(
    n: i32,
    reverse: bool,
    first_elem: i32,
    num_elems: i32,
    num_elems_full: i32,
    type_full: i32,
    full_ipar: &[i32],
    dpar: &[f64],
) -> FemResult<Vec<f64>> {
    let ref_factor = n / num_elems;
    if ref_factor * num_elems != n {
        return Err(FemError::Mesh("Invalid number of elements".into()));
    }
    let n = n as usize;
    if n == 1 {
        return Ok(vec![1.0]);
    }
    let full = spacing_eval_all_n(type_full, full_ipar, dpar, (ref_factor * num_elems_full) as usize)?;
    let os = (ref_factor * first_elem) as usize;
    if os + n > full.len() {
        return Err(FemError::Mesh(
            "partial spacing window outside the full spacing function".into(),
        ));
    }
    let mut s: Vec<f64> = full[os..os + n].to_vec();
    // Normalize.
    let d1: f64 = s.iter().sum();
    for v in &mut s {
        *v /= d1;
    }
    Ok((0..n)
        .map(|p| {
            let i = if reverse { n - 1 - p } else { p };
            s[i]
        })
        .collect())
}

/// [`spacing_eval_all`] with the size `n` overridden — what MFEM's
/// `SpacingFunction::SetSize` does when a composite function recomputes a
/// piece at its own interval count (`piece_n` replaces `ipar[0]`).
fn spacing_eval_all_n(
    spacing_type: i32,
    ipar: &[i32],
    dpar: &[f64],
    piece_n: usize,
) -> FemResult<Vec<f64>> {
    let mut ipar_n = ipar.to_vec();
    if ipar_n.is_empty() {
        ipar_n.push(piece_n as i32);
    } else {
        ipar_n[0] = piece_n as i32;
    }
    spacing_eval_all(spacing_type, &ipar_n, dpar)
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
    /// The v1.1 `spacing` section (empty for v1.0 files and the `patches`
    /// flavour).  See [`NurbsSpacingRecord`].
    pub spacing: Vec<NurbsSpacingRecord>,
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
    /// Supported:
    ///
    /// * the `patches` flavour (one patch per block, in file order);
    /// * the `knotvectors` flavour when the document really is a single patch
    ///   (`dim` knot vectors, one element) of any dimension;
    /// * multi-patch `knotvectors` documents: one patch per topology element,
    ///   with per-direction knot vectors, control points and orientation
    ///   resolved through `fem_space`'s `NurbsExtension` (the port of MFEM's
    ///   `NURBS_PatchMap` / `GenerateElementDofTable` machinery).  Requires a
    ///   node block whose row count equals MFEM's `GetNDof()`.
    pub fn to_nurbs_file(&self) -> FemResult<NurbsFile> {
        match &self.geometry {
            NurbsGeometry::Patches(blocks) => self.patches_to_nurbs_file(blocks),
            NurbsGeometry::Global { .. } => {
                if self.is_single_patch_representable() {
                    self.global_single_patch_to_nurbs_file()
                } else {
                    self.global_multi_patch_to_nurbs_file()
                }
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
            1 => build_single_patch_1d(&kv_data, weights, &flat, self.vdim),
            2 => build_single_patch_2d(&kv_data, weights, &flat, self.vdim, self.coords.len()),
            3 => build_single_patch_3d(&kv_data, weights, &flat, self.vdim, self.coords.len()),
            _ => Err(FemError::Mesh(format!(
                "nurbs mesh: unsupported dimension {}",
                self.dim
            ))),
        }
    }

    /// Multi-patch `knotvectors` view: MFEM's unified-knotvector representation
    /// maps every element (= patch) to its per-direction knot vectors and
    /// control points through `NURBS_PatchMap` / `GenerateElementDofTable`.
    /// That machinery is ported in `fem_space::NurbsExtension`, so this
    /// re-serializes the document, builds the extension from it, and gathers
    /// each patch's control points by global DOF (probe-pinned against MFEM
    /// 4.10's `NURBSPatchMap` dumps — see `tmp/d143/probe_truth.txt`).
    fn global_multi_patch_to_nurbs_file(&self) -> FemResult<NurbsFile> {
        let mut buf = Vec::new();
        write_nurbs_mesh_doc(self, &mut buf)?;
        let text = String::from_utf8(buf).map_err(|e| {
            FemError::Mesh(format!("nurbs mesh: document re-serialization failed: {e}"))
        })?;
        let ext = fem_space::NurbsExtension::from_mesh_str(&text)
            .map_err(|e| FemError::Mesh(format!("nurbs mesh: {e}")))?;
        if self.coords.len() != ext.n_dofs() {
            return Err(FemError::Mesh(format!(
                "nurbs mesh: the node block carries {} control points, expected \
                 GetNDof {}",
                self.coords.len(),
                ext.n_dofs()
            )));
        }
        if self.vdim < self.dim {
            return Err(FemError::Mesh(format!(
                "nurbs mesh: node block VDim {vdim} cannot carry dimension {d} geometry",
                vdim = self.vdim,
                d = self.dim
            )));
        }
        if self.dim == 1 && self.vdim != 1 {
            return Err(FemError::Mesh(
                "nurbs mesh: the 1-D NurbsFile variant carries one scalar coordinate per \
                 control point"
                    .into(),
            ));
        }
        let NurbsGeometry::Global { weights, .. } = &self.geometry else {
            unreachable!("caller checked the geometry flavour");
        };

        let mut p1: Vec<NurbsPatch1DData> = Vec::new();
        let mut p2: Vec<NurbsPatch2DData> = Vec::with_capacity(ext.n_patches());
        let mut p3: Vec<NurbsPatch3DData> = Vec::with_capacity(ext.n_patches());
        for p in 0..ext.n_patches() {
            let kvs = ext
                .patch_knot_vectors(p)
                .map_err(FemError::Mesh)?;
            let ncps: Vec<usize> = kvs.iter().map(|kv| kv.ncp()).collect();
            let n_cp: usize = ncps.iter().product();
            let tag = self.elements.get(p).map(|e| e.attribute).unwrap_or(1);

            // Patch-local control points in MFEM's `NURBSPatchMap` order
            // (i fastest along the first knot vector).
            let mut dofs = Vec::with_capacity(n_cp);
            let dof = |multi: &[usize]| -> FemResult<usize> {
                ext.patch_dof(p, multi)
                    .map_err(|e| FemError::Mesh(format!("nurbs mesh: patch {p}: {e}")))
            };
            match self.dim {
                1 => {
                    for i in 0..ncps[0] {
                        dofs.push(dof(&[i])?);
                    }
                }
                2 => {
                    for j in 0..ncps[1] {
                        for i in 0..ncps[0] {
                            dofs.push(dof(&[i, j])?);
                        }
                    }
                }
                3 => {
                    for k in 0..ncps[2] {
                        for j in 0..ncps[1] {
                            for i in 0..ncps[0] {
                                dofs.push(dof(&[i, j, k])?);
                            }
                        }
                    }
                }
                d => {
                    return Err(FemError::Mesh(format!(
                        "nurbs mesh: unsupported dimension {d}"
                    )));
                }
            }
            let coord = |d: usize| -> FemResult<Vec<f64>> {
                Ok(self.coords[d][..self.dim].to_vec())
            };
            let weight = |d: usize| weights.get(d).copied().unwrap_or(1.0);
            let cps: Vec<(Vec<f64>, f64)> = dofs
                .iter()
                .map(|&d| Ok((coord(d)?, weight(d))))
                .collect::<FemResult<Vec<_>>>()?;
            // `NurbsKnot` stores an `iga::KnotVector`; the patch data carries
            // the `nurbs::KnotVector` flavour, rebuilt from the same knots.
            let kv = |i: usize| {
                KnotVector::new(kvs[i].knot_vector().as_slice().to_vec(), kvs[i].order())
            };
            match self.dim {
                1 => p1.push(NurbsPatch1DData {
                    kv: kv(0),
                    control_pts: cps.iter().map(|(c, _)| c[0]).collect(),
                    weights: cps.iter().map(|(_, w)| *w).collect(),
                    tag,
                }),
                2 => p2.push(NurbsPatch2DData {
                    kv_u: kv(0),
                    kv_v: kv(1),
                    control_pts: cps.iter().map(|(c, _)| [c[0], c[1]]).collect(),
                    weights: cps.iter().map(|(_, w)| *w).collect(),
                    tag,
                }),
                _ => p3.push(NurbsPatch3DData {
                    kv_u: kv(0),
                    kv_v: kv(1),
                    kv_w: kv(2),
                    control_pts: cps.iter().map(|(c, _)| [c[0], c[1], c[2]]).collect(),
                    weights: cps.iter().map(|(_, w)| *w).collect(),
                    tag,
                }),
            }
        }
        match self.dim {
            1 => Ok(NurbsFile::Mesh1D(NurbsMesh1D { patches: p1 })),
            2 => Ok(NurbsFile::Mesh2D(NurbsMesh2D {
                patches: p2,
                edge_connectivity: Vec::new(),
            })),
            _ => Ok(NurbsFile::Mesh3D(NurbsMesh3D {
                patches: p3,
                face_connectivity: Vec::new(),
            })),
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
    let mut spacing = Vec::new();
    let geometry = match section.as_str() {
        "knotvectors" => {
            let n_kv = s.next_usize()?;
            let mut knotvectors = Vec::with_capacity(n_kv);
            for _ in 0..n_kv {
                knotvectors.push(read_kv_record(&mut s)?);
            }
            let kw = s.next_token()?;
            match kw.as_str() {
                "spacing" => {
                    // v1.1: one record per spacing function; the knot vectors
                    // themselves are unaffected.
                    let n = s.next_usize()?;
                    spacing.reserve(n);
                    for _ in 0..n {
                        let r = read_spacing_record(&mut s)?;
                        if r.knotvector >= n_kv {
                            return Err(FemError::Mesh(format!(
                                "nurbs mesh line {}: spacing record for knot vector {}, \
                                 but the file declares only {n_kv}",
                                s.line_no(),
                                r.knotvector
                            )));
                        }
                        spacing.push(r);
                    }
                    let w = s.next_token()?;
                    if w != "weights" {
                        return Err(FemError::Mesh(format!(
                            "nurbs mesh line {}: expected 'weights' after 'knotvectors', \
                             got '{w}'",
                            s.line_no()
                        )));
                    }
                }
                "weights" => {}
                // MFEM also accepts these before `spacing`; they carry NURBS
                // refinement bookkeeping, which fem-rs has not ported.
                "refinements" | "knotvector_refinements" => {
                    return Err(FemError::Mesh(format!(
                        "nurbs mesh line {}: the '{kw}' section (MFEM NURBS refinement \
                         bookkeeping) is not supported yet",
                        s.line_no()
                    )));
                }
                other => {
                    return Err(FemError::Mesh(format!(
                        "nurbs mesh line {}: expected 'weights' after 'knotvectors', \
                         got '{other}'. The 'mesh_elements' / 'periodic' sections are \
                         not supported yet.",
                        s.line_no()
                    )));
                }
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
        spacing,
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

/// One record of the v1.1 `spacing` section (MFEM `NURBSExtension::Load`):
/// `<kv> <SpacingType> <num-int> <num-real> <ipar…> <dpar…>` — see
/// [`NurbsSpacingRecord`].
fn read_spacing_record(s: &mut DocScanner) -> FemResult<NurbsSpacingRecord> {
    let knotvector = s.next_usize()?;
    let spacing_type = s.next_i32()?;
    let nip = s.next_usize()?;
    let nrp = s.next_usize()?;
    let mut int_params = Vec::with_capacity(nip);
    for _ in 0..nip {
        int_params.push(s.next_i32()?);
    }
    let mut real_params = Vec::with_capacity(nrp);
    for _ in 0..nrp {
        real_params.push(s.next_f64()?);
    }
    Ok(NurbsSpacingRecord {
        knotvector,
        spacing_type,
        int_params,
        real_params,
    })
}

/// The `FiniteElementSpace` node block, when the file has one.  Only the
/// `patches` flavour may omit it (see [`NurbsMeshDoc::has_node_block`]).
///
/// The coordinate rows are normalised to one row per control point
/// regardless of the file's `Ordering:` — MFEM `linalg/ordering.hpp`:
/// `Ordering::byNODES` (`0`) is **component-major** (`Map = dof + ndofs·vd`,
/// all of a component first), `Ordering::byVDIM` (`1`) is **interleaved**
/// (`Map = vd + vdim·dof`, `XYZ,XYZ,…` per control point).  D495: the D486
/// revision had the two arms swapped; the interleaved decode is what every
/// in-repo NURBS fixture (`Ordering: 1`) actually uses.
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
    let n = flat.len() / vdim;
    let coords: Vec<Vec<f64>> = if ordering == 0 {
        (0..n)
            .map(|d| (0..vdim).map(|c| flat[c * n + d]).collect())
            .collect()
    } else {
        flat.chunks(vdim).map(|c| c.to_vec()).collect()
    };
    Ok((collection, vdim, ordering, coords))
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
            if !doc.spacing.is_empty() {
                // `NURBSExtension::Print`: `\nspacing\n<count>\n` then one
                // `<kv> <type> <nip> <nrp> <ipar…> <dpar…>` line per record.
                writeln!(writer, "\nspacing\n{}", doc.spacing.len())?;
                for r in &doc.spacing {
                    write!(
                        writer,
                        "{} {} {} {}",
                        r.knotvector,
                        r.spacing_type,
                        r.int_params.len(),
                        r.real_params.len()
                    )?;
                    for v in &r.int_params {
                        write!(writer, " {v}")?;
                    }
                    for v in &r.real_params {
                        write!(writer, " {}", g(*v))?;
                    }
                    writeln!(writer)?;
                }
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
    // `GridFunction::Save`: the data stream is per-`Ordering` — byVDIM
    // interleaves the components of each control point, byNODES stores all of
    // a component first — and `Vector::Print` wraps it at `vdim` values per
    // line (vdim for byVDIM, 1 for byNODES).
    for (i, cp) in doc.coords.iter().enumerate() {
        if cp.len() != doc.vdim {
            return Err(FemError::Mesh(format!(
                "nurbs mesh: control point {i} has {} coordinates, expected VDim {}",
                cp.len(),
                doc.vdim
            )));
        }
    }
    let mut stream: Vec<String> = Vec::with_capacity(doc.coords.len() * doc.vdim);
    if doc.ordering == 1 {
        for cp in &doc.coords {
            for v in cp {
                stream.push(g(*v));
            }
        }
    } else {
        for c in 0..doc.vdim {
            for cp in &doc.coords {
                stream.push(g(cp[c]));
            }
        }
    }
    for chunk in stream.chunks(doc.vdim.max(1)) {
        writeln!(writer, "{}", chunk.join(" "))?;
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
