//! MFEM `NURBSExtension` — NURBS patch topology and global DOF numbering.
//!
//! 1:1 port of the parts of MFEM's `NURBSExtension` (C++ `mesh/nurbs.{hpp,cpp}`)
//! that determine the NURBS finite element space: patch topology bookkeeping,
//! unique/comprehensive knot vectors, `GenerateOffsets`, and
//! `GenerateElementDofTable` with its per-element DOF table.
//!
//! # What is reproduced
//!
//! * `Mesh::LoadPatchTopo` + `NURBSExtension::Load` for both file variants:
//!   the `knotvectors` flavour (what most `data/*-nurbs.mesh` meshes use) and
//!   the `patches` flavour (`NURBSPatch` blocks, e.g.
//!   `square-disc-nurbs-patch.mesh`, from which the unique knot vectors are
//!   reconstructed through `GetPatchDirectionEdges` / `CheckKVDirection` /
//!   `KnotVector::Flip`): `dimension`, `elements`, `boundary`, `edges`,
//!   `vertices`, `knotvectors`, `spacing`, `weights`.
//! * Edge canonicalisation (`edge_to_ukv` sign flip when the file writes the
//!   pair in decreasing vertex order) — `Mesh::LoadPatchTopo`.
//! * `NURBSExtension::GenerateOffsets` / `GetPatchOffsets`: the mesh and space
//!   offsets for vertices, edges, faces and patches.
//! * `NURBSExtension::CountElements` / `CountBdrElements`.
//! * `NURBSExtension::GenerateElementDofTable` (1D/2D/3D) including
//!   `NURBSPatchMap::operator()`, `Or1D`/`Or2D`, `EC`, `FC`, `FCP`, the global
//!   face construction of `Mesh::GenerateFaces` and the element edge/face
//!   orientations of `Mesh::GetElementEdges` / `GetElementFaces`.
//! * The global-to-local DOF compaction of `GenerateElementDofTable`
//!   (`activeDof`, `NumOfActiveDofs`).
//!
//! # What is not (yet) reproduced
//!
//! Only the conforming, non-periodic, non-NC path is ported: `activeElem` is
//! all-true (no `mesh_elements` section), `activeVert` is trivial, the periodic
//! `d_to_d` map is the identity, `NCNURBSExtension` master edges/faces are
//! absent (as in a conforming mesh, where `IsMasterEdge`/`IsMasterFace` are
//! false), and the per-row boundary-element
//! DOF *table* (`{Self::boundary_sides}` reproduces the union it feeds into
//! `GetEssentialTrueDofs`, with the attribute of each row), B-net/patch
//! conversion, refinement and `Print`/`PrintSolution` are out of scope.  See
//! the module tests for the coverage boundary.

use fem_element::iga::KnotVector;
use fem_element::nurbs_fe_collection::{degree_elevate, knot_n_elements, knot_ncp, knot_order};

/// Control-point coordinates of a NURBS mesh (`NurbsExtension::parse_nodes`).
#[derive(Debug, Clone, PartialEq)]
pub struct NurbsNodes {
    /// Physical (or embedding) dimension of a control point.
    pub vdim: usize,
    /// One coordinate vector per control point, in DOF order.
    pub coords: Vec<Vec<f64>>,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MFEM element topology tables (C++ `fem/geom.cpp`)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// `Geometry::Constants<Geometry::SQUARE>::Edges`
const SQUARE_EDGES: [[usize; 2]; 4] = [[0, 1], [1, 2], [2, 3], [3, 0]];
/// `Geometry::Constants<Geometry::CUBE>::Edges`
const CUBE_EDGES: [[usize; 2]; 12] = [
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
];
/// `Geometry::Constants<Geometry::CUBE>::FaceVert`
const CUBE_FACE_VERT: [[usize; 4]; 6] = [
    [3, 2, 1, 0],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [4, 5, 6, 7],
];

/// The patch-boundary side of the `j`-th local edge of a quadrilateral, as
/// `(direction, low)`: `Geometry::Constants<SQUARE>::Edges` is
/// `(0,1), (1,2), (2,3), (3,0)` on the reference square `(0,0) (1,0) (1,1)
/// (0,1)`, so edge 0 is the `y = 0` side, edge 1 the `x = 1` side, edge 2 the
/// `y = 1` side and edge 3 the `x = 0` side.
fn quad_edge_side(j: usize) -> (usize, bool) {
    match j {
        0 => (1, true),
        1 => (0, false),
        2 => (1, false),
        3 => (0, true),
        _ => unreachable!("a quadrilateral has four edges"),
    }
}

/// The patch-boundary side of the `k`-th local face of a hexahedron, as
/// `(direction, low)`: `Geometry::Constants<CUBE>::FaceVert` is
/// `(3,2,1,0), (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7), (4,5,6,7)` on the
/// reference cube with `v0 = (0,0,0), v1 = (1,0,0), v2 = (1,1,0), v3 = (0,1,0),
/// v4 = (0,0,1), …`, i.e. bottom (`z = 0`), front (`y = 0`), right (`x = 1`),
/// back (`y = 1`), left (`x = 0`) and top (`z = 1`).
fn hex_face_side(k: usize) -> (usize, bool) {
    match k {
        0 => (2, true),
        1 => (1, true),
        2 => (0, false),
        3 => (1, false),
        4 => (0, true),
        5 => (2, false),
        _ => unreachable!("a hexahedron has six faces"),
    }
}

/// MFEM `Geometry::Type` codes (subset used by NURBS meshes).
const GEOM_POINT: i32 = 0;
const GEOM_SEGMENT: i32 = 1;
const GEOM_SQUARE: i32 = 3;
const GEOM_CUBE: i32 = 5;

/// Number of vertices of a geometry code.
fn geom_n_vertices(code: i32) -> Option<usize> {
    match code {
        GEOM_POINT => Some(1),
        GEOM_SEGMENT => Some(2),
        GEOM_SQUARE => Some(4),
        GEOM_CUBE => Some(8),
        _ => None,
    }
}
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Knot vectors
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A NURBS knot vector with MFEM's cached `Order` / `NumOfControlPoints` /
/// `NumOfElements` (`mesh/nurbs.hpp` class `KnotVector`).
#[derive(Debug, Clone)]
pub struct NurbsKnot {
    kv: KnotVector,
    order: usize,
    ncp: usize,
    n_elements: usize,
}

impl NurbsKnot {
    /// Build from a clamped knot sequence, computing `NCP` / `NE` like MFEM's
    /// `KnotVector(int order, int NCP)` + `GetElements()`.
    pub fn new(kv: KnotVector, order: usize) -> Result<Self, String> {
        let real_order = knot_order(&kv).ok_or_else(|| "NurbsKnot: invalid knot vector".to_string())?;
        if real_order != order {
            return Err(format!(
                "NurbsKnot: declared order {order} disagrees with the knot sequence ({real_order})"
            ));
        }
        let ncp = knot_ncp(&kv).ok_or_else(|| "NurbsKnot: invalid knot vector".to_string())?;
        let n_elements = knot_n_elements(&kv).ok_or_else(|| "NurbsKnot: invalid knot vector".to_string())?;
        Ok(Self {
            kv,
            order,
            ncp,
            n_elements,
        })
    }

    /// MFEM `KnotVector::GetOrder`.
    pub fn order(&self) -> usize {
        self.order
    }

    /// MFEM `KnotVector::GetNCP`.
    pub fn ncp(&self) -> usize {
        self.ncp
    }

    /// MFEM `KnotVector::GetNE`.
    pub fn n_elements(&self) -> usize {
        self.n_elements
    }

    /// MFEM `KnotVector::GetNKS`.
    pub fn nks(&self) -> usize {
        self.ncp - self.order
    }

    /// MFEM `KnotVector::isElement(i)`.
    pub fn is_element(&self, i: usize) -> bool {
        fem_element::nurbs_fe_collection::knot_is_element(&self.kv, self.order, i)
    }

    /// The underlying knot sequence.
    pub fn knot_vector(&self) -> &KnotVector {
        &self.kv
    }

    /// MFEM `KnotVector::GetRefPoint` / `GetKnotLocation` — the reference
    /// coordinate of parameter `u` in the element starting at knot index
    /// `ni`, and back.
    pub fn ref_point(&self, u: f64, ni: usize) -> f64 {
        let k = self.kv.as_slice();
        (u - k[ni]) / (k[ni + 1] - k[ni])
    }

    /// MFEM `KnotVector::GetKnotLocation`.
    pub fn knot_location(&self, xi: f64, ni: usize) -> f64 {
        let k = self.kv.as_slice();
        xi * k[ni + 1] + (1.0 - xi) * k[ni]
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// NURBS mesh file parsing (`Mesh::LoadPatchTopo` + `NURBSExtension::Load`)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Tokenised NURBS mesh file: `(section keyword, numeric payload)` pairs in
/// file order, comments (`#` to end of line) removed.
fn tokenize_mesh(text: &str) -> Result<Vec<(String, Vec<f64>)>, String> {
    let mut sections: Vec<(String, Vec<f64>)> = Vec::new();
    let mut current: Option<(String, Vec<f64>)> = None;
    let mut lines = text.lines();
    // Skip the MFEM banner ("MFEM NURBS mesh v1.0").
    let banner = lines.next().unwrap_or("");
    if !banner.contains("NURBS mesh") {
        return Err(format!("not an MFEM NURBS mesh file (first line: {banner:?})"));
    }
    for raw in lines {
        let line = match raw.find('#') {
            Some(i) => &raw[..i],
            None => raw,
        };
        for tok in line.split_whitespace() {
            if let Ok(v) = tok.parse::<f64>() {
                match current.as_mut() {
                    Some((_, vals)) => vals.push(v),
                    None => return Err("mesh file: number before the first section".to_string()),
                }
            } else {
                if let Some(done) = current.take() {
                    sections.push(done);
                }
                current = Some((tok.to_string(), Vec::new()));
            }
        }
    }
    if let Some(done) = current.take() {
        sections.push(done);
    }
    Ok(sections)
}

/// Fetch a section's payload.
fn section<'a>(
    sections: &'a [(String, Vec<f64>)],
    name: &str,
) -> Result<&'a [f64], String> {
    sections
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.as_slice())
        .ok_or_else(|| format!("mesh file: missing '{name}' section"))
}

/// Whether a section is present.
fn has_section(sections: &[(String, Vec<f64>)], name: &str) -> bool {
    sections.iter().any(|(k, _)| k == name)
}

/// MFEM `KnotVector::Flip` (mesh/nurbs.cpp): mirror the interior knots around
/// the midpoint of the parameter interval, `k -> k(0) + k(size-1) - k`.  The
/// clamped end knots and the order/NCP/element counts are unaffected.
fn flip_knot_vector(knots: &[f64], order: usize) -> Vec<f64> {
    let mut k = knots.to_vec();
    let ncp = k.len() - order - 1;
    let apb = k[0] + k[k.len() - 1];
    let ns = ncp.saturating_sub(order) / 2;
    for i in 1..=ns {
        let tmp = apb - k[order + i];
        k[order + i] = apb - k[ncp - i];
        k[ncp - i] = tmp;
    }
    k
}

/// One `patches`-flavour block — the file image of MFEM's `NURBSPatch`.
///
/// The extension consumes the block's knot vectors and its **homogeneous
/// weights** (`Nurbsextension::weights`); the Cartesian coordinates of the
/// control points are validated and skipped (the geometry arrives through the
/// mesh's node block).
struct PatchBlock {
    /// One `(order, knots)` pair per parametric direction
    /// (`NCP = knots.len() - order - 1`).
    knotvectors: Vec<(usize, Vec<f64>)>,
    /// Per-direction control-point counts (`KnotVector::GetNCP`).
    ncp: Vec<usize>,
    /// The homogeneous last component of every control point, in MFEM's
    /// `NURBSPatch` storage order `i + j*ncp[0] + k*ncp[0]*ncp[1]`
    /// (`NURBSPatch::NURBSPatch(std::istream&)` fills `data[dim]` that way for
    /// both the `controlpoints_homogeneous` and the `controlpoints_cartesian`
    /// flavour — the latter multiplies the first `dim` components by it).
    weights: Vec<f64>,
}

/// Whitespace token scanner over the raw file text, with MFEM's `#` comment
/// handling (`skip_comment_lines`).
struct PatchTokens {
    toks: Vec<String>,
    pos: usize,
}

impl PatchTokens {
    fn next(&mut self) -> Result<String, String> {
        let tok = self
            .toks
            .get(self.pos)
            .cloned()
            .ok_or_else(|| "patches: unexpected end of input".to_string())?;
        self.pos += 1;
        Ok(tok)
    }

    fn next_usize(&mut self, what: &str) -> Result<usize, String> {
        let tok = self.next()?;
        tok.parse::<usize>()
            .map_err(|_| format!("patches: expected {what}, got '{tok}'"))
    }

    fn next_f64(&mut self, what: &str) -> Result<f64, String> {
        let tok = self.next()?;
        tok.parse::<f64>()
            .map_err(|_| format!("patches: expected {what}, got '{tok}'"))
    }

    fn expect(&mut self, keyword: &str) -> Result<(), String> {
        let got = self.next()?;
        if got != keyword {
            return Err(format!("patches: expected '{keyword}', got '{got}'"));
        }
        Ok(())
    }
}

/// `NURBSPatch::NURBSPatch(std::istream &)` (mesh/nurbs.cpp): `np` blocks of
/// `knotvectors / dimension / controlpoints[_cartesian|_homogeneous]`.
fn parse_patch_blocks(text: &str, np: usize) -> Result<Vec<PatchBlock>, String> {
    let toks: Vec<String> = text
        .lines()
        .map(|l| match l.find('#') {
            Some(i) => &l[..i],
            None => l,
        })
        .flat_map(str::split_whitespace)
        .map(str::to_string)
        .collect();
    let pos = toks
        .iter()
        .position(|t| t == "patches")
        .ok_or_else(|| "patches: missing 'patches' section".to_string())?;
    let mut s = PatchTokens { toks, pos: pos + 1 };
    let mut blocks = Vec::with_capacity(np);
    for p in 0..np {
        s.expect("knotvectors").map_err(|e| format!("patch {p}: {e}"))?;
        let n_kv = s.next_usize("knot vector count")?;
        let mut knotvectors = Vec::with_capacity(n_kv);
        for _ in 0..n_kv {
            let order = s.next_usize("knot vector order")?;
            let ncp = s.next_usize("knot vector NCP")?;
            let n = ncp + order + 1;
            let mut knots = Vec::with_capacity(n);
            for _ in 0..n {
                knots.push(s.next_f64("knot value")?);
            }
            knotvectors.push((order, knots));
        }
        s.expect("dimension").map_err(|e| format!("patch {p}: {e}"))?;
        let d = s.next_usize("patch dimension")?;
        if d == 0 || d > 3 {
            return Err(format!("patch {p}: unsupported dimension {d}"));
        }
        let keyword = s.next()?;
        match keyword.as_str() {
            "controlpoints" | "controlpoints_homogeneous" | "controlpoints_cartesian" => {}
            other => {
                return Err(format!(
                    "patch {p}: expected 'controlpoints', 'controlpoints_homogeneous' or \
                     'controlpoints_cartesian', got '{other}'"
                ));
            }
        }
        let n_cp: usize = knotvectors.iter().map(|(o, k)| k.len() - o - 1).product();
        let ncp: Vec<usize> = knotvectors.iter().map(|(o, k)| k.len() - o - 1).collect();
        let mut weights = Vec::with_capacity(n_cp);
        for _ in 0..n_cp {
            let mut last = 0.0_f64;
            for _ in 0..(d + 1) {
                last = s.next_f64("control point value")?;
            }
            weights.push(last);
        }
        blocks.push(PatchBlock { knotvectors, ncp, weights });
    }
    Ok(blocks)
}

fn as_usize(v: f64, what: &str) -> Result<usize, String> {
    if v < 0.0 || v.fract() != 0.0 {
        return Err(format!("{what}: expected a non-negative integer, got {v}"));
    }
    Ok(v as usize)
}

/// Row-major strides of a tensor with one entry per direction
/// (`stride[0] = 1`, MFEM's `NURBSPatch` layout `i + j*ni + k*ni*nj`).
fn tensor_strides(ncp: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; ncp.len()];
    for i in 1..ncp.len() {
        s[i] = s[i - 1] * ncp[i - 1];
    }
    s
}

/// The flat index of `multi` in a tensor of shape `ncp` (the inverse of the
/// digit decomposition `multi_index_from`'s callers use).
fn multi_index_from(multi: &[usize], ncp: &[usize]) -> usize {
    debug_assert_eq!(multi.len(), ncp.len());
    multi.iter().zip(ncp).enumerate().map(|(d, (&m, &n))| {
        debug_assert!(m < n, "multi-index out of range in direction {d}");
        m * tensor_strides(ncp)[d]
    }).sum()
}

/// MFEM `KnotVector::GetSpan` (`mesh/nurbs.cpp:1187`) — the knot-span index of
/// the parameter `u` (`knots.len() == ncp + order + 1`).  Note the two exact
/// endpoint shortcuts and the `[order, ncp]` search window: they matter, because
/// `NURBSPatch::KnotInsert` uses this span to *start* its A5.5 loop.
fn get_span(knots: &[f64], order: usize, ncp: usize, u: f64) -> usize {
    if u == knots[ncp + order] {
        return ncp - 1;
    }
    if u == knots[0] {
        return order;
    }
    let (mut low, mut high) = (order, ncp);
    let mut mid = (low + high) / 2;
    while u < knots[mid] || u >= knots[mid + 1] {
        if u < knots[mid] {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }
    mid
}

/// MFEM `NURBSPatch::KnotInsert(dir, const Vector &knot)` (`mesh/nurbs.cpp:1767`,
/// NURBS Book **A5.5** as MFEM implements it) applied to one control polygon:
/// insert **all** of `knots_in` in a single backward pass.
///
/// This is deliberately not textbook A5.1.  MFEM's loop runs `j` from the last
/// inserted knot down to the first, carrying `i` (old index) and `k` (new index)
/// downward, and computes the blend factor from the **new** knot vector,
/// `alfa = (newkv[k+l] - u_j) / (newkv[k+l] - oldkv[i-pl+l])`, applying
/// `Q[ind-1] = alfa*Q[ind-1] + (1-alfa)*Q[ind]`.  The result differs from A5.1's
/// `[p0, p1, (p1+p2)/2, ...]` at the low end: for a single inserted knot in a
/// uniformly single-span order-4 patch MFEM produces
/// `[p0, (p0+p1)/2, (p1+p2)/2, (p2+p3)/2, (p3+p4)/2, p4]`.  Verified against the
/// MFEM 4.10 `weights` of every refined `ball-nurbs.mesh` element
/// (`crates/space/tests/d516_nurbs_weights.rs`).
///
/// `pl` is the MFEM order (= degree + 1) and `p` the `ml` control values.
fn knot_insert_line(kv_old: &[f64], pl: usize, p: &[f64], knots_in: &[f64]) -> Vec<f64> {
    let ml = p.len();
    let rr = knots_in.len() - 1;
    let a = get_span(kv_old, pl, ml, knots_in[0]);
    let b = get_span(kv_old, pl, ml, knots_in[rr]);
    let mut newkv = vec![0.0_f64; kv_old.len() + knots_in.len()];
    let mut q = vec![0.0_f64; ml + knots_in.len()];

    for j in 0..=a {
        newkv[j] = kv_old[j];
    }
    for j in (b + pl)..=(ml + pl) {
        newkv[j + rr + 1] = kv_old[j];
    }
    if a >= pl {
        for k in 0..=(a - pl) {
            q[k] = p[k];
        }
    }
    for k in (b - 1)..ml {
        q[k + rr + 1] = p[k];
    }

    let mut i = (b + pl - 1) as isize;
    let mut k = (b + pl + rr) as isize;
    for j in (0..=rr).rev() {
        while knots_in[j] <= kv_old[i as usize] && i > a as isize {
            newkv[k as usize] = kv_old[i as usize];
            q[(k - pl as isize - 1) as usize] = p[(i - pl as isize - 1) as usize];
            k -= 1;
            i -= 1;
        }
        q[(k - pl as isize - 1) as usize] = q[(k - pl as isize) as usize];
        for l in 1..=pl {
            let ind = (k - pl as isize + l as isize) as usize;
            let mut alfa = newkv[k as usize + l] - knots_in[j];
            if alfa == 0.0 {
                q[ind - 1] = q[ind];
            } else {
                alfa /= newkv[k as usize + l] - kv_old[(i - pl as isize + l as isize) as usize];
                q[ind - 1] = alfa * q[ind - 1] + (1.0 - alfa) * q[ind];
            }
        }
        newkv[k as usize] = knots_in[j];
        k -= 1;
    }
    q
}

/// [`knot_insert_line`] applied to every line of a tensor along direction `d`:
/// `w` is a tensor of shape `ncp`, the result a tensor of shape `ncp` with
/// `ncp[d] + knots_in.len()` entries along `d`.
fn insert_knot_direction(
    w: &[f64],
    ncp: &[usize],
    d: usize,
    knots: &[f64],
    order: usize,
    knots_in: &[f64],
) -> Vec<f64> {
    let dim = ncp.len();
    let old_strides = tensor_strides(ncp);
    let mut nnew = ncp.to_vec();
    nnew[d] += knots_in.len();
    let new_strides = tensor_strides(&nnew);
    let mut out = vec![0.0_f64; nnew.iter().product()];
    let outer: usize = ncp
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != d)
        .map(|(_, &v)| v)
        .product();
    let mut idx = vec![0usize; dim];
    for _ in 0..outer {
        let base: usize = (0..dim).map(|i| idx[i] * old_strides[i]).sum();
        let line: Vec<f64> = (0..ncp[d]).map(|i| w[base + i * old_strides[d]]).collect();
        let q = knot_insert_line(knots, order, &line, knots_in);
        let base_new: usize = (0..dim).map(|i| idx[i] * new_strides[i]).sum();
        for (i, &v) in q.iter().enumerate() {
            out[base_new + i * new_strides[d]] = v;
        }
        for i in 0..dim {
            if i == d {
                continue;
            }
            idx[i] += 1;
            if idx[i] < ncp[i] {
                break;
            }
            idx[i] = 0;
        }
    }
    out
}

/// A topology element (patch or boundary element).
#[derive(Debug, Clone)]
struct TopoElement {
    attr: i32,
    geom: i32,
    verts: Vec<usize>,
}

fn read_elements(payload: &[f64], what: &str) -> Result<Vec<TopoElement>, String> {
    let n = as_usize(*payload.first().unwrap_or(&0.0), what)?;
    let mut out = Vec::with_capacity(n);
    let mut i = 1;
    for e in 0..n {
        if i + 1 >= payload.len() {
            return Err(format!("{what}: truncated at element {e}"));
        }
        let attr = payload[i] as i32;
        let geom = payload[i + 1] as i32;
        let nv = geom_n_vertices(geom)
            .ok_or_else(|| format!("{what}: unsupported geometry code {geom}"))?;
        let verts = payload
            .get(i + 2..i + 2 + nv)
            .ok_or_else(|| format!("{what}: truncated element {e}"))?
            .iter()
            .map(|&v| v as usize)
            .collect();
        out.push(TopoElement { attr, geom, verts });
        i += 2 + nv;
    }
    Ok(out)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// NURBSExtension
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// MFEM `NURBSExtension` — the conforming NURBS patch topology plus the global
/// NURBS DOF numbering of a `NURBSFECollection` H1 space.
///
/// The element DOF table [`Self::element_dofs`] is MFEM's `el_dof` table, i.e.
/// the map `element -> global DOF` that `NURBSExtension::GetElementDofTable`
/// exposes and `FiniteElementSpace` consumes.  DOF numbering is bit-identical
/// to MFEM: verified against `NURBSExtension::GetElementDofTable` dumps for the
/// ten NURBS meshes in the test suite.
#[derive(Debug, Clone)]
pub struct NurbsExtension {
    dim: usize,
    /// Per-knot-vector orders (MFEM `mOrders`).
    orders: Vec<usize>,
    /// The overall order, or `None` when `mOrders` disagree (`VariableOrder`).
    order: Option<usize>,
    /// Unique knot vectors (MFEM `knotVectors`).
    knot_vectors: Vec<NurbsKnot>,
    /// Signed unique-knot-vector index per global edge (MFEM `edge_to_ukv`).
    edge_to_ukv: Vec<i32>,
    /// Patch topology elements (MFEM `patchTopo` elements).
    elements: Vec<TopoElement>,
    /// Patch topology boundary elements.
    boundary: Vec<TopoElement>,
    /// For every entry of `boundary`, the patch-boundary entity it lies on
    /// (see [`BdrSide`]).  Filled by [`Self::compute_bdr_sides`].
    bdr_sides: Vec<BdrSide>,
    /// Vertices per global edge (MFEM `edge_vertex`, canonicalised min/max).
    edge_vertex: Vec<(usize, usize)>,
    /// Global face vertex cycles (MFEM `Mesh::faces`), first-encounter order.
    faces: Vec<[usize; 4]>,
    /// Element local edge -> global edge (MFEM `el_to_edge`).
    el_edges: Vec<Vec<usize>>,
    /// Element local edge -> orientation, `+1`/`-1`
    /// (MFEM `Mesh::GetElementEdges`'s `cor`).
    el_edge_sign: Vec<Vec<i32>>,
    /// Element local face -> global face (MFEM `el_to_face`).
    el_faces: Vec<Vec<usize>>,
    /// Element local face -> orientation (MFEM `Mesh::GetElementFaces`'s `ori`,
    /// i.e. `faces_info[f].Elem{1,2}Inf % 64`).
    el_face_ori: Vec<Vec<i32>>,
    /// Space offsets per vertex (MFEM `v_spaceOffsets`).
    v_space_offsets: Vec<usize>,
    /// Space offsets per edge (MFEM `e_spaceOffsets`).
    e_space_offsets: Vec<usize>,
    /// Space offsets per face (MFEM `f_spaceOffsets`).
    f_space_offsets: Vec<usize>,
    /// Space offsets per patch (MFEM `p_spaceOffsets`).
    p_space_offsets: Vec<usize>,
    /// Element DOF table, globally numbered (MFEM `el_dof`).
    el_dof: Vec<Vec<usize>>,
    /// Element -> patch (MFEM `el_to_patch`).
    el_to_patch: Vec<usize>,
    /// Element -> knot-span indices `(i, j, k)` (MFEM `el_to_IJK`).
    el_to_ijk: Vec<[usize; 3]>,
    /// Total DOFs before compaction (MFEM `GetNTotalDof`).
    n_total_dofs: usize,
    /// Active DOFs (MFEM `GetNDof`).
    n_dofs: usize,
    /// Total elements over all patches (MFEM `GetGNE`).
    n_elements: usize,
    /// Total boundary elements (MFEM `GetGNBE`).
    n_bdr_elements: usize,
    /// DOF weights (MFEM `weights`).
    weights: Vec<f64>,
    /// Active vertices (MFEM `GetNV` / `NumOfActiveVertices`).
    n_vertices: usize,
    /// Mesh-offset count of `GenerateOffsets` (MFEM `GetGNV`).
    n_global_vertices: usize,
    /// Patch topology vertex count (`Mesh::FinalizeTopology`).
    n_topo_vertices: usize,
    /// Mesh offsets (MFEM `v_meshOffsets` … `p_meshOffsets`).
    v_mesh_offsets: Vec<usize>,
    e_mesh_offsets: Vec<usize>,
    f_mesh_offsets: Vec<usize>,
    p_mesh_offsets: Vec<usize>,
    /// Periodic-BC DOF map (MFEM `d_to_d`), empty while the extension has no
    /// connected boundaries; see [`Self::dof_map`] and
    /// [`Self::connect_boundaries`].
    d_to_d: Vec<usize>,
    /// Raw-DOF → compacted-DOF map (MFEM `activeDof` after the finalize pass
    /// of `GenerateElementDofTable`): the slot of a raw DOF — in the
    /// [`Self::dof_map`]-mapped numbering — in the compacted element-table
    /// numbering, or `usize::MAX` for raw slots no element reaches.  The only
    /// conforming case with such slots is 1-D, where the interior slot of a
    /// unique edge duplicates the owning patch's interior slot and only the
    /// patch's is ever addressed (`NURBSPatchMap::operator()(i)` case 1 goes
    /// to `pOffset`); 2-D/3-D conforming offsets are fully used, so the map is
    /// the identity there.
    active_dof: Vec<usize>,
}

/// Which FE space a boundary DOF table is generated for — MFEM
/// `NURBSExtension::Mode`.  The mode changes *which* boundary DOFs exist and
/// their sign, never the element DOF table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BdrDofMode {
    /// `Mode::H_1` — scalar (or DG) space: every control point of the boundary
    /// entity, no sign.
    H1,
    /// `Mode::H_DIV` — divergence-conforming space: only the DOF block of the
    /// component normal to the entity survives, and it is **negated on the low
    /// side** of its direction.
    HDiv,
    /// `Mode::H_CURL` — curl-conforming space: only the DOF block of the
    /// component tangential to the entity survives; signs are all `+`.
    HCurl,
}

/// MFEM `NURBSPatchMap`'s per-boundary-element orientation data, as
/// `NURBSExtension::GenerateBdrElementDofTable` uses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BdrSide {
    /// Owning patch (MFEM `bel_to_patch`).
    pub patch: usize,
    /// The entity's **normal** direction: 0/1 in 2-D, 0/1/2 in 3-D.
    pub dir: usize,
    /// `true` on the minimum-parameter side of `dir`.
    pub low: bool,
    /// Mesh boundary attribute (`Mesh::GetBdrAttribute`).
    pub attr: i32,
    /// Local entity index of `dir` inside the owning patch element
    /// (`Mesh::GetBdrElementFaceIndex`, i.e. the local edge in 2-D and the
    /// local face in 3-D).
    pub local: usize,
}

/// A signed boundary DOF table row entry: MFEM encodes a DOF whose basis
/// function enters with the opposite sign as `-1 - dof`
/// (`Vector::AddElementVector`, `NURBSExtension::GenerateBdrElementDofTable`).
pub fn unsign_dof(d: i64) -> (usize, i64) {
    if d < 0 { ((-1 - d) as usize, -1) } else { (d as usize, 1) }
}

/// Which `NURBSPatchMap` mode to use: MFEM builds the patch map twice, once for
/// the mesh vertices (`SetPatchVertexMap`, `I = GetNE() - 1`, mesh offsets) and
/// once for the NURBS space DOFs (`SetPatchDofMap`, `I = GetNCP() - 2`, space
/// offsets).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MapMode {
    /// `NURBSPatchMap::SetPatchVertexMap` — mesh offsets, `NE - 1` interior.
    Vertex,
    /// `NURBSPatchMap::SetPatchDofMap` — space offsets, `NCP - 2` interior.
    Dof,
}

/// `NURBSPatchMap::SetBdrPatchDofMap` for a 2-D patch, whose boundary patch is a
/// single segment: the state its `operator()(i)` consumes.
///
/// `edgeMaster` is empty for a conforming extension (`NURBSExtension::
/// IsMasterEdge` returns false), so `EC` always takes the `edges[0] + …` offset
/// path; [`NurbsExtension::bdr_seg_dof`] is `operator()(i)` under that
/// assumption.
struct BdrSegDofMap<'a> {
    /// `dir`-endpoint control points of the boundary segment.
    verts: [usize; 2],
    /// `e_spaceOffsets[edges[0]]`.
    p_offset: usize,
    /// `p2g.nx()` = `I + 1` = `GetNCP() - 1`.
    nx: isize,
    /// `I` = `GetNCP() - 2`.
    i_cap: isize,
    /// `KnotVec(edges[0], oedge[0])`.
    kv: &'a NurbsKnot,
    /// `KnotSign(edge) * oedge[0]`.
    okv: i32,
    /// `cor[0]` of `Mesh::GetBdrElementEdges`.
    oedge: i32,
}

/// `NURBSPatchMap::SetBdrPatchDofMap` for a 3-D patch, whose boundary patch is a
/// quadrilateral: the state its `operator()(i, j)` consumes.
///
/// As for [`BdrSegDofMap`], `edgeMaster`/`faceMaster` are empty for a conforming
/// extension, so `EC`/`FC` take the offset path and `FCP` the `pOffset` one;
/// [`BdrQuadDofMap::dof`] implements exactly that.
struct BdrQuadDofMap<'a> {
    /// `v_spaceOffsets` of the boundary element's four vertices.
    verts: [usize; 4],
    /// `e_spaceOffsets` of the boundary element's four local edges.
    edges: [usize; 4],
    /// `cor[j]` of `Mesh::GetBdrElementEdges`.
    oedge: [i32; 4],
    /// `p2g.nx()`/`p2g.ny()` = `I + 1`/`J + 1` = `GetNCP() - 1` per direction.
    nx: isize,
    ny: isize,
    /// `I`/`J` = `GetNCP() - 2` per direction.
    i_cap: isize,
    j_cap: isize,
    /// `KnotVec(edges[0/1], oedge[0/1])`.
    kvs: [&'a NurbsKnot; 2],
    /// `KnotSign(edges[0/1]) * oedge[0/1]`.
    okv: [i32; 2],
    /// `Mesh::GetBdrElementFace`'s `o` — the boundary element's orientation
    /// w.r.t. the face it lies on (`Mesh::GetQuadOrientation`).
    opatch: i32,
    /// `f_spaceOffsets[face]`.
    p_offset: usize,
}

impl BdrQuadDofMap<'_> {
    /// `NURBSPatchMap::operator()(i, j)`, i.e. the global DOF of boundary-patch
    /// multi-index `(i, j)` with `0 <= i <= nx` and `0 <= j <= ny` (the extreme
    /// values address the boundary patch's vertices).
    fn dof(&self, i: usize, j: usize) -> usize {
        let f = |n: isize, big_n: isize| -> usize {
            if n < 0 {
                0
            } else if n >= big_n {
                2
            } else {
                1
            }
        };
        let or1d = |n: isize, big_n: isize, or: i32| -> usize {
            if or > 0 {
                n as usize
            } else {
                (big_n - 1 - n) as usize
            }
        };
        let or2d = |m: isize, n: isize, big_m: isize, big_n: isize, or: i32| -> usize {
            let (m, big_m, n, big_n) =
                (m as usize, big_m as usize, n as usize, big_n as usize);
            match or {
                0 => m + n * big_m,
                1 => n + m * big_n,
                2 => n + (big_m - 1 - m) * big_n,
                3 => (big_m - 1 - m) + n * big_m,
                4 => (big_m - 1 - m) + (big_n - 1 - n) * big_m,
                5 => (big_n - 1 - n) + (big_m - 1 - m) * big_n,
                6 => (big_n - 1 - n) + m * big_n,
                _ => m + (big_n - 1 - n) * big_m,
            }
        };
        let ec = |e: usize, m: isize, big_n: isize, s: i32| -> usize {
            self.edges[e] + or1d(m, big_n, s * self.oedge[e])
        };
        let i1 = i as isize - 1;
        let j1 = j as isize - 1;
        match 3 * f(j1, self.j_cap) + f(i1, self.i_cap) {
            0 => self.verts[0],
            1 => ec(0, i1, self.i_cap, 1),
            2 => self.verts[1],
            3 => ec(3, j1, self.j_cap, -1),
            4 => self.p_offset + or2d(i1, j1, self.i_cap, self.j_cap, self.opatch),
            5 => ec(1, j1, self.j_cap, 1),
            6 => self.verts[3],
            7 => ec(2, i1, self.i_cap, -1),
            _ => self.verts[2],
        }
    }
}

impl NurbsExtension {
    /// Read a NURBS mesh file (`NURBSExtension(std::istream&)`).  Both
    /// geometry flavours of `NURBSExtension::Load` are supported: the
    /// `knotvectors` variant and the `patches` variant (`NURBSPatch` blocks,
    /// from which the unique knot vectors are reconstructed).
    pub fn from_mesh_str(text: &str) -> Result<Self, String> {
        let sections = tokenize_mesh(text)?;

        // ── patch topology (`Mesh::LoadPatchTopo`) ────────────────────────────
        let dim = as_usize(section(&sections, "dimension")?[0], "dimension")?;
        if dim == 0 || dim > 3 {
            return Err(format!("NurbsExtension: unsupported dimension {dim}"));
        }
        let elements = read_elements(section(&sections, "elements")?, "elements")?;
        let boundary = read_elements(section(&sections, "boundary")?, "boundary")?;

        let edge_payload = section(&sections, "edges")?;
        let n_edges = as_usize(edge_payload[0], "edges")?;
        let mut edge_vertex = Vec::with_capacity(n_edges);
        let mut raw_ukv = Vec::with_capacity(n_edges);
        for j in 0..n_edges {
            let base = 1 + 3 * j;
            let kv = edge_payload[base] as i32;
            let mut v0 = edge_payload[base + 1] as usize;
            let mut v1 = edge_payload[base + 2] as usize;
            // MFEM keeps the edge direction increasing and flips the knot
            // vector sign (`FlipIndexSign`, i.e. `-1 - ukv`) when the file
            // writes it the other way round.
            let kv = if v0 > v1 {
                std::mem::swap(&mut v0, &mut v1);
                -kv - 1
            } else {
                kv
            };
            edge_vertex.push((v0, v1));
            raw_ukv.push(kv);
        }

        // `Mesh::LoadPatchTopo` reads the vertex count and then drops the vertex
        // data; `Mesh::FinalizeTopology` re-derives it as one past the largest
        // vertex index used by the elements/boundary elements.  Cross-check the
        // file's number for format sanity.
        let declared_vertices = as_usize(section(&sections, "vertices")?[0], "vertices")?;
        let max_used = elements
            .iter()
            .chain(boundary.iter())
            .flat_map(|e| e.verts.iter())
            .copied()
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        let n_topo_vertices = declared_vertices.max(max_used);

        // 1D: edge indices are patch indices, the sign encodes orientation.
        let edge_to_ukv = if n_edges == 0 && dim == 1 {
            let mut e2u = vec![0i32; elements.len()];
            for (p, el) in elements.iter().enumerate() {
                e2u[p] = if el.verts[1] > el.verts[0] {
                    p as i32
                } else {
                    -(p as i32) - 1
                };
            }
            e2u
        } else {
            raw_ukv
        };

        // ── unique knot vectors (`NURBSExtension::Load`) ──────────────────────
        // The `patches` variant reconstructs them from the patch blocks once
        // the patch topology (per-patch direction edges) exists; see
        // `fill_knot_vectors_from_patches`.
        let patches_variant = has_section(&sections, "patches");
        let knot_vectors: Vec<NurbsKnot> = if patches_variant {
            Vec::new()
        } else {
            let kv_payload = section(&sections, "knotvectors")?;
            let n_kv = as_usize(kv_payload[0], "knotvectors")?;
            let mut knot_vectors = Vec::with_capacity(n_kv);
            let mut i = 1;
            for k in 0..n_kv {
                let order = as_usize(kv_payload[i], "knotvectors order")?;
                let ncp = as_usize(kv_payload[i + 1], "knotvectors NCP")?;
                let size = ncp + order + 1;
                let knots: Vec<f64> = kv_payload
                    .get(i + 2..i + 2 + size)
                    .ok_or_else(|| format!("knotvectors: truncated knot vector {k}"))?
                    .to_vec();
                let kv = KnotVector::new_clamped(knots)?;
                knot_vectors.push(NurbsKnot::new(kv, order)?);
                i += 2 + size;
            }
            knot_vectors
        };

        let mut ext = Self {
            dim,
            orders: Vec::new(),
            order: None,
            knot_vectors,
            edge_to_ukv,
            elements,
            boundary,
            bdr_sides: Vec::new(),
            edge_vertex,
            faces: Vec::new(),
            el_edges: Vec::new(),
            el_edge_sign: Vec::new(),
            el_faces: Vec::new(),
            el_face_ori: Vec::new(),
            v_space_offsets: Vec::new(),
            e_space_offsets: Vec::new(),
            f_space_offsets: Vec::new(),
            p_space_offsets: Vec::new(),
            el_dof: Vec::new(),
            el_to_patch: Vec::new(),
            el_to_ijk: Vec::new(),
            n_total_dofs: 0,
            n_dofs: 0,
            n_elements: 0,
            n_bdr_elements: 0,
            weights: Vec::new(),
            n_vertices: 0,
            n_global_vertices: 0,
            n_topo_vertices,
            v_mesh_offsets: Vec::new(),
            e_mesh_offsets: Vec::new(),
            f_mesh_offsets: Vec::new(),
            p_mesh_offsets: Vec::new(),
            d_to_d: Vec::new(),
            active_dof: Vec::new(),
        };

        ext.build_patch_topology()?;

        if patches_variant {
            // `NURBSExtension::Load`, `patches` branch.
            ext.fill_knot_vectors_from_patches(text)?;
        }

        // `SetOrdersFromKnotVectors` + `SetOrderFromOrders`.
        ext.orders = ext.knot_vectors.iter().map(|k| k.order()).collect();
        ext.order = {
            let mut o = ext.orders.first().copied();
            for &x in &ext.orders[1..] {
                if Some(x) != o {
                    o = None;
                    break;
                }
            }
            o
        };

        if ext.boundary.is_empty() {
            ext.generate_boundary_elements();
        }
        ext.rebuild()?;

        // ── weights ───────────────────────────────────────────────────────────
        if patches_variant {
            // `Mesh::ReadNURBSMesh` → `NURBSext->SetCoordsFromPatches(*Nodes,
            // vdim)` → `NURBSExtension::Set{1,2,3}DSolutionVector`.  MFEM's
            // `NURBSExtension::Load` guards the `weights` section with
            // `if (patches.Size() == 0)`, so the `patches` flavour never reads
            // one: its rational weights are the homogeneous last component of
            // the patch control points.
            ext.fill_weights_from_patches(text)?;
        } else {
            // `NURBSExtension::Load` does `weights.Load(input, GetNDof())`, and
            // `Vector::Load(std::istream &in, int Size)` reads **exactly**
            // `Size` values sequentially from the stream — there is no count
            // and no length check, and anything the section carries beyond the
            // first `GetNDof()` values is simply never consumed (the stream
            // position is left on it; `weights` is the last section of the
            // v1.0 format, so nothing else reads it either).  The mesh's
            // rational weights are therefore the **first `GetNDof()` numbers**
            // after the keyword — `ball-nurbs.mesh` is the file that exercises
            // this: its section holds 517 values for `GetNDof() == 517`, and
            // the trailing `FiniteElementSpace` node block (a further 1560
            // tokens) is *not* part of it.
            //
            // A section shorter than `GetNDof()` cannot be reproduced
            // faithfully (MFEM would keep reading and consume the following
            // block's tokens), so it is rejected loudly rather than padded or
            // silently replaced by unit weights.
            match section(&sections, "weights") {
                Ok(w) if w.len() >= ext.n_dofs => {
                    ext.weights = w[..ext.n_dofs].to_vec();
                }
                Ok(w) => {
                    return Err(format!(
                        "weights: MFEM's `weights.Load(input, GetNDof())` reads {} values, \
                         the section holds {}",
                        ext.n_dofs,
                        w.len()
                    ));
                }
                Err(_) => {
                    // `unitweights` / `autoweights`:
                    // `weights.SetSize(GetNDof()); weights = 1.0;`
                    ext.weights = ext.unit_weights();
                }
            }
        }

        Ok(ext)
    }

    /// Read a NURBS mesh file from disk.
    pub fn from_mesh_file(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let text = std::fs::read_to_string(path.as_ref())
            .map_err(|e| format!("NurbsExtension::from_mesh_file: {e}"))?;
        Self::from_mesh_str(&text)
    }

    // ── the `patches` mesh-file variant (`NURBSExtension::Load`) ─────────────

    /// MFEM `NURBSExtension::Load`, `patches` branch: parse the `GetNP()`
    /// `NURBSPatch` blocks from the raw file text and reconstruct the unique
    /// knot vectors, each stored in the canonical orientation
    /// (`KnotVector::Flip` when the patch direction runs against its edge).
    ///
    /// Must be called after [`Self::build_patch_topology`] — the per-patch
    /// direction edges (`GetPatchDirectionEdges`) come from it.
    fn fill_knot_vectors_from_patches(&mut self, text: &str) -> Result<(), String> {
        let blocks = parse_patch_blocks(text, self.elements.len())?;
        let n_kv = (0..self.edge_to_ukv.len())
            .map(|e| self.knot_ind(e) + 1)
            .max()
            .unwrap_or(0);
        let dim = self.dim;
        let mut filled: Vec<Option<NurbsKnot>> = (0..n_kv).map(|_| None).collect();
        for (p, block) in blocks.iter().enumerate() {
            let dir_edges: Vec<usize> = match dim {
                1 => vec![self.el_edges[p][0]],
                2 => vec![self.el_edges[p][0], self.el_edges[p][1]],
                3 => vec![self.el_edges[p][0], self.el_edges[p][3], self.el_edges[p][8]],
                d => return Err(format!("patches: unsupported dimension {d}")),
            };
            if block.knotvectors.len() != dim {
                return Err(format!(
                    "patch {p}: {} knot vectors, expected {dim}",
                    block.knotvectors.len()
                ));
            }
            let kvdir = self.check_kv_direction(p)?;
            for (d, &edge) in dir_edges.iter().enumerate() {
                let kv = self.knot_ind(edge);
                if filled[kv].is_some() {
                    continue;
                }
                let (order, knots) = &block.knotvectors[d];
                let knots = if kvdir[d] == -1 {
                    flip_knot_vector(knots, *order)
                } else {
                    knots.clone()
                };
                let kvec = KnotVector::new_clamped(knots)?;
                filled[kv] = Some(NurbsKnot::new(kvec, *order)?);
            }
        }
        self.knot_vectors = filled
            .into_iter()
            .enumerate()
            .map(|(i, k)| {
                k.ok_or_else(|| {
                    format!("patches: knot vector {i} is not defined by any patch block")
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(())
    }

    /// MFEM `Mesh::ReadNURBSMesh`'s `NURBSext->SetCoordsFromPatches(*Nodes,
    /// vdim)` → `NURBSExtension::Set{1,2,3}DSolutionVector`: the rational
    /// weights of a `patches`-flavour mesh are the **homogeneous last
    /// component** of the patch control points,
    ///
    /// ```text
    ///   weights(p2g(i,j,k)) = patch(i,j,k,vdim)      (nurbs.cpp:5496/5556/5620)
    /// ```
    ///
    /// with `p2g` MFEM's `NURBSPatchMap::operator()` — [`Self::patch_dof`]
    /// here.  MFEM writes every patch's block unconditionally (the
    /// `dof2patch` guard only fires for the non-conforming extension), so a
    /// control point shared by two patches takes its weight from the last one
    /// to reach it; the knot-insertion invariance of the shared net makes both
    /// values equal, and the completeness check below keeps a partially
    /// covered net from passing silently.
    ///
    /// Must be called after the knot vectors and the DOF numbering are final
    /// (`rebuild`).
    fn fill_weights_from_patches(&mut self, text: &str) -> Result<(), String> {
        let blocks = parse_patch_blocks(text, self.elements.len())?;
        let dim = self.dim;
        let mut weights = vec![0.0_f64; self.n_dofs];
        let mut covered = vec![false; self.n_dofs];
        for (p, block) in blocks.iter().enumerate() {
            if block.ncp.len() != dim {
                return Err(format!(
                    "patch {p}: {} knot vectors, expected {dim}",
                    block.ncp.len()
                ));
            }
            let n_cp: usize = block.ncp.iter().product();
            if block.weights.len() != n_cp {
                return Err(format!(
                    "patch {p}: {} control-point weights for {n_cp} control points",
                    block.weights.len()
                ));
            }
            let mut multi = vec![0usize; dim];
            for (flat, &w) in block.weights.iter().enumerate() {
                // `NURBSPatch` layout `i + j*ncp[0] + k*ncp[0]*ncp[1]`
                // (`mesh/nurbs.hpp:1364`).
                let mut rem = flat;
                for d in 0..dim {
                    multi[d] = rem % block.ncp[d];
                    rem /= block.ncp[d];
                }
                let g = self.patch_dof(p, &multi)?;
                weights[g] = w;
                covered[g] = true;
            }
        }
        if let Some(g) = covered.iter().position(|&c| !c) {
            return Err(format!(
                "patches: control point {g} is not covered by any patch block \
                 ({} DOFs)",
                self.n_dofs
            ));
        }
        self.weights = weights;
        Ok(())
    }

    /// MFEM `NURBSExtension::CheckKVDirection` — the per-direction orientation
    /// (`+1` / `-1`) of patch `p`'s knot vectors, derived by comparing each
    /// direction edge's vertices with the patch's first vertices.
    ///
    /// In 1D the sign of `edge_to_ukv` *is* the orientation.
    fn check_kv_direction(&self, p: usize) -> Result<Vec<i32>, String> {
        let dim = self.dim;
        if dim == 1 {
            return Ok(vec![self.knot_sign(self.el_edges[p][0])]);
        }
        let patchvert = &self.elements[p].verts;
        let mut kvdir = vec![0i32; dim];
        for &e in &self.el_edges[p] {
            let (ev0, ev1) = self.edge_vertex[e];
            let ks = self.knot_sign(e);
            // First side (direction 0 runs along patchvert[0] -> patchvert[1]).
            if ev0 == patchvert[0] && ev1 == patchvert[1] {
                kvdir[0] = ks;
            }
            if ev0 == patchvert[1] && ev1 == patchvert[0] {
                kvdir[0] = -ks;
            }
            // Second side (direction 1 along patchvert[0] -> patchvert[3]).
            if ev0 == patchvert[0] && ev1 == patchvert[3] {
                kvdir[1] = ks;
            }
            if ev0 == patchvert[3] && ev1 == patchvert[0] {
                kvdir[1] = -ks;
            }
            if dim == 3 {
                // Third side (direction 2 along patchvert[0] -> patchvert[4]).
                if ev0 == patchvert[0] && ev1 == patchvert[4] {
                    kvdir[2] = ks;
                }
                if ev0 == patchvert[4] && ev1 == patchvert[0] {
                    kvdir[2] = -ks;
                }
            }
        }
        if kvdir.contains(&0) {
            return Err(format!(
                "patch {p}: could not find the direction of a knot vector"
            ));
        }
        Ok(kvdir)
    }

    // ── derived data (`SetOrdersFromKnotVectors` … `GenerateElementDofTable`) ─

    /// Recompute everything derived from the knot vectors: the per-knot-vector
    /// orders (`SetOrdersFromKnotVectors` + `SetOrderFromOrders`),
    /// `GenerateOffsets`, `CountElements` / `CountBdrElements`,
    /// `GenerateActiveVertices` and `GenerateElementDofTable`.
    ///
    /// Called after the knot vectors change ([`Self::with_orders`],
    /// [`Self::uniform_refinement`]); the patch topology (elements, boundary
    /// elements, edges, faces) is unaffected by knot insertion and degree
    /// elevation, exactly as in MFEM's `NURBSUniformRefinement` /
    /// `NURBSExtension(parent, order)`.
    fn rebuild(&mut self) -> Result<(), String> {
        self.orders = self.knot_vectors.iter().map(|k| k.order()).collect();
        self.order = {
            let mut o = self.orders.first().copied();
            for &x in &self.orders[1..] {
                if Some(x) != o {
                    o = None;
                    break;
                }
            }
            o
        };
        self.generate_offsets();
        self.count_elements();
        self.count_bdr_elements();
        self.generate_active_vertices()?;
        self.generate_element_dof_table()?;
        self.compute_bdr_sides();
        Ok(())
    }

    /// Unit weights for every DOF — MFEM's `NURBSExtension(parent, …)`
    /// constructors do `weights.SetSize(GetNDof()); weights = 1.0;`, so the
    /// **analysis space** is the polynomial B-spline space even when the mesh
    /// geometry is rational.
    fn unit_weights(&self) -> Vec<f64> {
        vec![1.0; self.n_dofs]
    }

    /// The analysis extension of MFEM's **`NURBSext == NULL`** finite element
    /// space — `FiniteElementSpace::Constructor` (`fem/fespace.cpp:2557-2567`):
    ///
    /// ```cpp
    /// const NURBSFECollection *nurbs_fec = dynamic_cast<...>(fec_);
    /// if (nurbs_fec)
    /// {
    ///    MFEM_VERIFY(mesh_->NURBSext, "NURBS FE space requires a NURBS mesh.");
    ///    if (NURBSext_ == NULL) { NURBSext = mesh_->NURBSext; own_ext = 0; }
    ///    else                   { NURBSext = NURBSext_;      own_ext = 1; }
    /// ```
    ///
    /// With `NURBSext_ == NULL` — the `nurbs_patch_ex1` configuration
    /// (`FiniteElementSpace fespace(&mesh, fec)`) — the space's extension *is*
    /// the mesh's, so it inherits the mesh's **rational** weights, and
    /// `NURBSFiniteElement::CalcShape` (`fem/fe/fe_nurbs.cpp:38-42`) normalizes
    /// the B-spline values by them.  That is the opposite of
    /// [`Self::with_orders`], which mirrors `new NURBSExtension(parent, order)`
    /// (`nurbs_ex1 -o ≥ 1`) and resets the weights to one.
    ///
    /// The target orders therefore have to *equal* the mesh's orders: MFEM
    /// never elevates through this path (the collection comes from the mesh's
    /// own `Nodes` grid function), and a raised order would need a refined
    /// control net that this extension does not carry.  Both cases are rejected
    /// loudly instead of silently dropping the rational weighting.
    pub fn with_orders_keeping_weights(&self, orders: &[usize]) -> Result<Self, String> {
        if orders.len() != self.knot_vectors.len() {
            return Err(format!(
                "NurbsExtension::with_orders_keeping_weights: {} orders for {} knot vectors",
                orders.len(),
                self.knot_vectors.len()
            ));
        }
        let mismatched: Vec<usize> = (0..orders.len())
            .filter(|&i| orders[i] != self.knot_vectors[i].order())
            .collect();
        if !mismatched.is_empty() {
            return Err(format!(
                "NurbsExtension::with_orders_keeping_weights: target orders {:?} differ from the \
                 mesh orders {:?} at knot vectors {mismatched:?}; MFEM's \
                 `FiniteElementSpace(mesh, fec)` has `NURBSext == mesh->NURBSext`, so it cannot \
                 elevate — use `with_orders` for `nurbs_ex1`'s \
                 `NURBSExtension(mesh->NURBSext, order)`",
                orders, self.orders
            ));
        }
        if self.weights.len() != self.n_dofs {
            return Err(format!(
                "NurbsExtension::with_orders_keeping_weights: {} weights for {} DOFs — the mesh \
                 extension's weights must cover its control net",
                self.weights.len(),
                self.n_dofs
            ));
        }
        Ok(self.clone())
    }

    /// MFEM `NURBSExtension(NURBSExtension *parent, const Array<int> &newOrders)`
    /// (and the single-order form used by `nurbs_ex1`/`nurbs_ex3`).
    ///
    /// Every knot vector is degree elevated to its target order (unchanged when
    /// the target is not larger, exactly as MFEM), the DOF numbering and the
    /// element DOF table are regenerated, and the weights are reset to one.
    pub fn with_orders(&self, orders: &[usize]) -> Result<Self, String> {
        if orders.len() != self.knot_vectors.len() {
            return Err(format!(
                "NurbsExtension::with_orders: {} orders for {} knot vectors",
                orders.len(),
                self.knot_vectors.len()
            ));
        }
        let mut ext = self.clone();
        for (i, &target) in orders.iter().enumerate() {
            let current = ext.knot_vectors[i].order();
            ext.knot_vectors[i] = if target > current {
                NurbsKnot::new(
                    degree_elevate(ext.knot_vectors[i].knot_vector(), target - current)?,
                    target,
                )?
            } else {
                ext.knot_vectors[i].clone()
            };
        }
        ext.rebuild()?;
        ext.weights = ext.unit_weights();
        Ok(ext)
    }

    /// MFEM `Mesh::NURBSUniformRefinement` at the extension level:
    /// `KnotVector::UniformRefinement(new_knots, rf)` inserts `rf - 1` equally
    /// spaced knots into every non-empty span of every unique knot vector, and
    /// `NURBSPatch::KnotInsert` re-derives the control net by knot insertion in
    /// **homogeneous** form — so the refined weights are A5.1 blends of the old
    /// ones, not the old values repeated.
    ///
    /// The refined *geometry* is not stored: knot insertion leaves the
    /// parametric surface invariant, so [`crate::NurbsFESpace`] evaluates the
    /// original control net over the refined parameter intervals (see
    /// [`crate::NurbsFESpace::geometry`]).  The refined **weights** do have to
    /// be materialised, because they are what
    /// `NURBSFiniteElement::CalcShape` normalizes by on the refined mesh
    /// (`NURBSExtension::LoadFE` copies `weights.GetSubVector(el_dofs)`).
    pub fn uniform_refinement(&mut self, rf: usize) -> Result<(), String> {
        if rf < 2 {
            return Err(format!(
                "NurbsExtension::uniform_refinement: refinement factor must be >= 2, got {rf}"
            ));
        }
        let old = self.clone();
        for k in self.knot_vectors.iter_mut() {
            let knots = k.knot_vector().as_slice();
            // `KnotVector::UniformRefinement`: for every non-empty span
            // [knot[i], knot[i+1]] insert the values
            // (1 - m/rf)*knot[i] + (m/rf)*knot[i+1], m = 1..rf, which sorts into
            // the existing sequence (the inserted values are strictly interior).
            let mut refined: Vec<f64> = Vec::with_capacity(knots.len() + knots.len() * rf);
            refined.push(knots[0]);
            for w in knots.windows(2) {
                if w[0] != w[1] {
                    for m in 1..rf {
                        let t = m as f64 / rf as f64;
                        refined.push((1.0 - t) * w[0] + t * w[1]);
                    }
                }
                refined.push(w[1]);
            }
            *k = NurbsKnot::new(KnotVector::new_clamped(refined)?, k.order())?;
        }
        self.rebuild()?;
        self.weights = old.refined_weights(self, rf)?;
        Ok(())
    }

    /// The rational weights of the uniformly refined control net:
    /// per patch, `NURBSPatch::KnotInsert(dir, knot)` (Piegl & Tiller A5.5 as
    /// MFEM implements it) applied to the **homogeneous** weight tensor,
    /// direction by direction.
    ///
    /// `self` is the pre-refinement extension (source of the old weights),
    /// `new` the post-refinement one (source of the new DOF numbering).  The
    /// projective form of a rational patch is invariant under knot insertion,
    /// so inserting one knot `u` along direction `d` turns a weight line `w[i]`
    /// into a blend with the *new* knot vector's factors; the control points
    /// shared by neighbouring patches get the same value (both blend the same
    /// projective net), and the disagreement check keeps a damaged net from
    /// passing silently.
    ///
    /// The local control point ⇄ global DOF correspondence is taken from the
    /// **element DOF tables** ([`Self::patch_local_dofs`]); [`Self::patch_dof`]
    /// produces the same numbering over its full index domain (pinned by the
    /// D539 conformance test), and the table route additionally covers meshes
    /// whose elements do not reach every control point.
    fn refined_weights(&self, new: &NurbsExtension, rf: usize) -> Result<Vec<f64>, String> {
        let mut comps = self.refined_components(new, rf, std::slice::from_ref(&self.weights))?;
        Ok(comps.remove(0))
    }

    /// The per-component tensors of the uniformly refined control net — the
    /// shared body of [`Self::refined_weights`] (one component: the weights)
    /// and [`Self::refined_control_points`] (`vdim` homogeneous coordinates
    /// plus the weight): per patch, `NURBSPatch::KnotInsert(dir, knot)` applied
    /// direction by direction to every component tensor with the same blend
    /// factors (`NURBSPatch::KnotInsert`'s `slice(k, ll)` loop runs the A5.5
    /// recursion over the components of one control polygon line, which is the
    /// same per-component arithmetic as blending each component's line
    /// separately).
    ///
    /// `comps[c][g]` is component `c` at old-DOF `g`; the returned tensors are
    /// in `new`'s DOF numbering.  A control point shared by several patches is
    /// written once and cross-checked against the other patches' values
    /// (knot-insertion invariance makes them agree).
    fn refined_components(
        &self,
        new: &NurbsExtension,
        rf: usize,
        comps: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, String> {
        let dim = self.dim;
        let mut out = vec![vec![0.0_f64; new.n_dofs]; comps.len()];
        let mut set = vec![false; new.n_dofs];
        for p in 0..self.n_patches() {
            let pkv = self.patch_knot_vectors(p)?;
            let mut ncp: Vec<usize> = pkv.iter().map(|k| k.ncp()).collect();
            let mut tensors = vec![vec![0.0_f64; ncp.iter().product()]; comps.len()];
            for (multi, g) in self.patch_local_dofs(p)? {
                let flat = multi_index_from(&multi, &ncp);
                for (t, c) in tensors.iter_mut().zip(comps) {
                    t[flat] = c[g];
                }
            }
            for d in 0..dim {
                let order = pkv[d].order();
                let knots = pkv[d].knot_vector().as_slice();
                let mut inserted: Vec<f64> = Vec::new();
                for pair in knots.windows(2) {
                    if pair[0] != pair[1] {
                        for m in 1..rf {
                            let t = m as f64 / rf as f64;
                            inserted.push((1.0 - t) * pair[0] + t * pair[1]);
                        }
                    }
                }
                if inserted.is_empty() {
                    continue;
                }
                for t in tensors.iter_mut() {
                    *t = insert_knot_direction(t, &ncp, d, knots, order, &inserted);
                }
                ncp[d] += inserted.len();
            }
            for (multi, g) in new.patch_local_dofs(p)? {
                let flat = multi_index_from(&multi, &ncp);
                for (o, t) in out.iter_mut().zip(&tensors) {
                    let v = t[flat];
                    if set[g] && (o[g] - v).abs() > 1e-12 * o[g].abs().max(1.0) {
                        return Err(format!(
                            "NurbsExtension::uniform_refinement: control point {g} of patch {p} \
                             disagrees with the value another patch wrote ({} vs {v})",
                            o[g]
                        ));
                    }
                    o[g] = v;
                }
                set[g] = true;
            }
        }
        if let Some(g) = set.iter().position(|&s| !s) {
            return Err(format!(
                "NurbsExtension::uniform_refinement: refined control point {g} is not covered \
                 by any patch"
            ));
        }
        Ok(out)
    }

    /// The control-point **coordinates** of the uniformly refined control net —
    /// the coordinate half of MFEM's `NURBSUniformRefinement`
    /// (`Mesh::RefineNURBS`): `NURBSExtension::ConvertToPatches` homogenizes
    /// the mesh's `Nodes` (`Patch(...,d) = coords(l·vdim+d)·weights(l)`), every
    /// direction runs `NURBSPatch::KnotInsert` on the homogeneous tensor, and
    /// `NURBSExtension::Set{1,2,3}DSolutionVector` divides the refined
    /// components back by the refined weights
    /// (`coords(l·vdim+d) = patch(...,d)/patch(...,vdim)`).
    ///
    /// `self` is the pre-refinement extension (with the mesh's rational
    /// weights), `new` the post-refinement one, `coords` the original control
    /// points (`vdim` components per DOF, as [`NurbsNodes::coords`]), and `rf`
    /// the refinement factor [`Self::uniform_refinement`] ran with.  Returns
    /// the refined Cartesian control points in `new`'s DOF numbering.
    pub fn refined_control_points(
        &self,
        new: &NurbsExtension,
        coords: &[Vec<f64>],
        rf: usize,
    ) -> Result<Vec<Vec<f64>>, String> {
        let vdim = coords
            .first()
            .map(|c| c.len())
            .ok_or_else(|| "NurbsExtension::refined_control_points: no control points".to_string())?;
        if vdim < self.dim {
            return Err(format!(
                "NurbsExtension::refined_control_points: {vdim} components for dimension {}",
                self.dim
            ));
        }
        if coords.len() != self.n_dofs() {
            return Err(format!(
                "NurbsExtension::refined_control_points: {} control points for {} DOFs",
                coords.len(),
                self.n_dofs()
            ));
        }
        if self.weights.len() != self.n_dofs() {
            return Err(format!(
                "NurbsExtension::refined_control_points: {} weights for {} DOFs — the rational \
                 weights must cover the control net before homogenizing",
                self.weights.len(),
                self.n_dofs()
            ));
        }
        let mut comps: Vec<Vec<f64>> = Vec::with_capacity(vdim + 1);
        for d in 0..vdim {
            comps.push(
                coords
                    .iter()
                    .zip(&self.weights)
                    .map(|(c, &w)| c[d] * w)
                    .collect(),
            );
        }
        comps.push(self.weights.clone());
        let mut refined = self.refined_components(new, rf, &comps)?;
        let weights = refined.pop().expect("the weight component");
        // Dehomogenize componentwise: `coords(l·vdim+d) = patch(...,d) /
        // patch(...,vdim)` — every dof of every coordinate component is
        // divided by the refined weight **at that dof**.
        for c in refined.iter_mut() {
            for (v, &w) in c.iter_mut().zip(weights.iter()) {
                *v /= w;
            }
        }
        // Transpose the component-major tensors (`comps[c][g]`) into the
        // dof-major convention of [`NurbsNodes::coords`] (one coordinate vector
        // per DOF).
        let mut out = vec![vec![0.0_f64; vdim]; new.n_dofs()];
        for (d, comp) in refined.iter().enumerate() {
            for (g, &v) in comp.iter().enumerate() {
                out[g][d] = v;
            }
        }
        Ok(out)
    }

    /// The patch-local control point ⇄ global DOF correspondence of patch `p`,
    /// as `(local multi-index, DOF)` pairs — read off the **element DOF
    /// tables**, which is the same data the finite element space itself uses.
    ///
    /// MFEM deep-copies an element's `(i, j, k)` and its `el_dof` row in
    /// `NURBSExtension::LoadFE`, and `NURBSPatchMap::SetPatchDofMap` reads the
    /// same numbering, so the two views agree: for an element whose patch-local
    /// span indices are `spans`, local DOF `o` of `element_dofs(e)` belongs to
    /// the control point `spans[d] + o_d`, where `o_d` is the `d`-th digit of `o`
    /// in the `(order_d + 1)`-per-direction tensor layout (x fastest — the
    /// `NurbsScalar{1,2,3}D` dof order).
    ///
    /// A control point on a patch boundary appears in several elements and is
    /// assigned once per global DOF; the returned pairs are unique per `multi`
    /// and per `dof`.  Public for the D539 conformance tests, which pin
    /// [`Self::patch_dof`] against this table over the full index domain.
    pub fn patch_local_dofs(&self, p: usize) -> Result<Vec<(Vec<usize>, usize)>, String> {
        let dim = self.dim;
        let kvs = self.patch_knot_vectors(p)?;
        let nloc: Vec<usize> = kvs.iter().map(|k| k.order() + 1).collect();
        let ncp: Vec<usize> = kvs.iter().map(|k| k.ncp()).collect();
        let total: usize = ncp.iter().product();
        let mut pairs: Vec<(Vec<usize>, usize)> = Vec::with_capacity(total);
        let mut seen = vec![false; total];
        for e in 0..self.n_elements() {
            if self.element_patch(e) != p {
                continue;
            }
            let spans = self.element_ijk(e);
            for (o, &g) in self.element_dofs(e).iter().enumerate() {
                let mut rem = o;
                let mut multi = vec![0usize; dim];
                for d in 0..dim {
                    if spans[d] + rem % nloc[d] >= ncp[d] {
                        return Err(format!(
                            "NurbsExtension::patch_local_dofs: element {e} local DOF {o} leaves \
                             patch {p}'s net in direction {d}"
                        ));
                    }
                    multi[d] = spans[d] + rem % nloc[d];
                    rem /= nloc[d];
                }
                let flat = multi_index_from(&multi, &ncp);
                if !seen[flat] {
                    seen[flat] = true;
                    pairs.push((multi, g));
                }
            }
        }
        if let Some(flat) = seen.iter().position(|&s| !s) {
            return Err(format!(
                "NurbsExtension::patch_local_dofs: patch {p} control point {flat} is not reached \
                 by any element"
            ));
        }
        Ok(pairs)
    }

    /// MFEM `NURBSExtension::GetPatchDofs` — the global (compacted) DOF index
    /// of the patch multi-index `multi` in the `(x, y, z)` direction order,
    /// `0 <= multi[d] < NCP[d]`.
    ///
    /// C++ evaluates `DofMap(NURBSPatchMap::operator()(...))`; the element DOF
    /// table additionally pushes the result through the `activeDof` compaction
    /// of `GenerateElementDofTable`, and the consumers of `GetPatchVDofs`
    /// (the patch-wise `BilinearForm` assembly) index compacted vectors with
    /// it — so this port applies both maps.  For conforming 2-D/3-D meshes the
    /// compaction is the identity and the result is MFEM's raw value; the 1-D
    /// refined extension is where they differ (the raw value can alias the
    /// inactive interior slot of a unique edge, which MFEM itself never
    /// consumes through `GetPatchDofs` — D539).
    pub fn patch_dof(&self, patch: usize, multi: &[usize]) -> Result<usize, String> {
        if multi.len() != self.dim {
            return Err(format!(
                "NurbsExtension::patch_dof: expected {} indices, got {}",
                self.dim,
                multi.len()
            ));
        }
        let merged = self.dof_map(self.patch_map_mode(patch, multi, MapMode::Dof)?);
        match self.active_dof.get(merged) {
            Some(&dof) if dof != usize::MAX => Ok(dof),
            Some(_) => Err(format!(
                "NurbsExtension::patch_dof: patch {patch} control point {multi:?} addresses \
                 raw DOF {merged}, which no element reaches (inactive slot)"
            )),
            None => Err(format!(
                "NurbsExtension::patch_dof: raw DOF {merged} is outside the offset table"
            )),
        }
    }

    /// The control-point coordinates of a NURBS mesh file: the
    /// `FiniteElementSpace` / `VDim: <n>` / `Ordering: <0|1>` block that MFEM
    /// reads into the mesh's `Nodes` grid function (the B-spline control net).
    ///
    /// The file's `Ordering:` line is honoured exactly as MFEM's
    /// `FiniteElementSpace`/`GridFunction` pair does
    /// (`linalg/ordering.hpp::Ordering::Map`): `1` = byVDIM interleaves the
    /// components of each control point (`XYZ,XYZ,...`, i.e. `chunks(vdim)`),
    /// `0` = byNODES stores the data component-major (`XXX...,YYY...`, i.e.
    /// `values[d*n_dofs + i]`).  An MFEM 4.10 probe (`tmp/d487/probe.cpp`)
    /// dumps identical control points and element geometry for
    /// `disc-nurbs.mesh` and for a transposed `Ordering: 0` image of it,
    /// pinning this interpretation.
    ///
    /// `n_dofs` is the expected number of control points (`GetNDof`), so a
    /// truncated or mismatched block is rejected rather than silently accepted.
    pub fn parse_nodes(text: &str, n_dofs: usize) -> Result<NurbsNodes, String> {
        let mut lines = text.lines();
        let banner = lines.next().unwrap_or("");
        if !banner.contains("NURBS mesh") {
            return Err(format!("not an MFEM NURBS mesh file (first line: {banner:?})"));
        }
        let mut vdim = None;
        let mut ordering = 1;
        let mut values: Vec<f64> = Vec::new();
        let mut in_block = false;
        for raw in lines {
            let line = match raw.find('#') {
                Some(i) => &raw[..i],
                None => raw,
            };
            let trimmed = line.trim();
            if !in_block {
                if trimmed == "FiniteElementSpace" {
                    in_block = true;
                }
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("VDim:") {
                vdim = Some(
                    rest.trim()
                        .parse::<usize>()
                        .map_err(|_| format!("FiniteElementSpace: bad VDim {rest:?}"))?,
                );
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("Ordering:") {
                ordering = rest
                    .trim()
                    .parse::<i32>()
                    .map_err(|_| format!("FiniteElementSpace: bad Ordering {rest:?}"))?;
                if ordering != 0 && ordering != 1 {
                    return Err(format!(
                        "FiniteElementSpace: Ordering must be 0 (byNODES) or 1 (byVDIM), \
                         got {ordering}"
                    ));
                }
                continue;
            }
            // Header lines (`FiniteElementCollection: …`) carry the only other
            // non-numeric columns; the coordinate rows are plain numbers.
            if trimmed.contains(':') {
                continue;
            }
            for tok in trimmed.split_whitespace() {
                if let Ok(v) = tok.parse::<f64>() {
                    values.push(v);
                }
            }
        }
        let vdim = vdim.ok_or_else(|| "mesh file: no 'VDim:' line".to_string())?;
        if vdim == 0 {
            return Err("FiniteElementSpace: VDim must be positive".to_string());
        }
        if values.len() != n_dofs * vdim {
            return Err(format!(
                "FiniteElementSpace: expected {} control point values ({n_dofs} x vdim {vdim}), \
                 found {}",
                n_dofs * vdim,
                values.len()
            ));
        }
        // `Ordering::Map`: byVDIM = `vd + vdim*dof` (a control point's
        // components are contiguous), byNODES = `dof + ndofs*vd`
        // (component-major blocks).
        let coords: Vec<Vec<f64>> = if ordering == 1 {
            values.chunks(vdim).map(|c| c.to_vec()).collect()
        } else {
            (0..n_dofs)
                .map(|i| (0..vdim).map(|d| values[d * n_dofs + i]).collect())
                .collect()
        };
        Ok(NurbsNodes { vdim, coords })
    }

    // ── topology construction (`Mesh::FinalizeTopology` / `GenerateFaces`) ────

    /// Resolve per-element edges and (3D) faces, building the global edge/face
    /// numbering exactly like MFEM's `Mesh::FinalizeTopology`.
    fn build_patch_topology(&mut self) -> Result<(), String> {
        let n_el = self.elements.len();

        // In 1D the edges are not stored in the mesh file: the edge index is the
        // patch index (`NURBSExtension::GenerateOffsets` uses `KnotVec(p)`).
        if self.dim == 1 {
            self.el_edges = vec![vec![0usize]; n_el];
            self.el_edge_sign = vec![vec![0i32]; n_el];
            for (p, el) in self.elements.iter().enumerate() {
                self.el_edges[p][0] = p;
                self.el_edge_sign[p][0] = if el.verts[1] > el.verts[0] { 1 } else { -1 };
            }
            self.el_faces = vec![Vec::new(); n_el];
            self.el_face_ori = vec![Vec::new(); n_el];
            self.faces = Vec::new();
            return Ok(());
        }

        // Vertex-pair -> global edge (the edge numbering comes from the file).
        let mut edge_of_pair: Vec<((usize, usize), usize)> = self
            .edge_vertex
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, i))
            .collect();
        edge_of_pair.sort_by_key(|&(p, _)| p);
        let lookup_edge = |a: usize, b: usize| -> Result<usize, String> {
            let key = if a < b { (a, b) } else { (b, a) };
            edge_of_pair
                .binary_search_by_key(&key, |&(p, _)| p)
                .map(|i| edge_of_pair[i].1)
                .map_err(|_| format!("patch topology: no global edge for vertex pair {key:?}"))
        };

        self.el_edges = Vec::with_capacity(n_el);
        self.el_edge_sign = Vec::with_capacity(n_el);
        self.el_faces = Vec::with_capacity(n_el);
        self.el_face_ori = Vec::with_capacity(n_el);

        let mut faces: Vec<[usize; 4]> = Vec::new();
        let mut face_of_key: Vec<(Vec<usize>, usize)> = Vec::new();
        let mut face_owner: Vec<usize> = Vec::new();

        for (p, el) in self.elements.iter().enumerate() {
            let v = &el.verts;
            let edge_table: &[[usize; 2]] = match el.geom {
                GEOM_SQUARE => &SQUARE_EDGES,
                GEOM_CUBE => &CUBE_EDGES,
                g => return Err(format!("patch topology: unsupported geometry {g}")),
            };

            let mut edges = Vec::with_capacity(edge_table.len());
            let mut signs = Vec::with_capacity(edge_table.len());
            for &[a, b] in edge_table {
                if v[a] == v[b] {
                    return Err(format!("element {p}: degenerate edge"));
                }
                edges.push(lookup_edge(v[a], v[b])?);
                // MFEM `Mesh::GetElementEdges`: cor = +1 when the element's local
                // edge runs from the smaller to the larger global vertex.
                signs.push(if v[a] < v[b] { 1 } else { -1 });
            }
            self.el_edges.push(edges);
            self.el_edge_sign.push(signs);

            let mut efaces = Vec::new();
            let mut eori = Vec::new();
            if el.geom == GEOM_CUBE {
                for fv in CUBE_FACE_VERT.iter() {
                    let tuple = [v[fv[0]], v[fv[1]], v[fv[2]], v[fv[3]]];
                    let mut key = tuple.to_vec();
                    key.sort_unstable();
                    let gf = match face_of_key.binary_search_by(|(k, _)| k.as_slice().cmp(&key)) {
                        Ok(idx) => face_of_key[idx].1,
                        Err(pos) => {
                            let gf = faces.len();
                            faces.push(tuple);
                            face_owner.push(p);
                            face_of_key.insert(pos, (key, gf));
                            gf
                        }
                    };
                    // MFEM `Mesh::GetElementFaces`: the face's storing element
                    // sees orientation 0, the other element's orientation is
                    // `Mesh::GetQuadOrientation(face_verts, local_verts)`.
                    let ori = if face_owner[gf] == p {
                        0
                    } else {
                        Self::quad_orientation(&faces[gf], &tuple)
                    };
                    efaces.push(gf);
                    eori.push(ori);
                }
            }
            self.el_faces.push(efaces);
            self.el_face_ori.push(eori);
        }

        self.faces = faces;
        Ok(())
    }

    /// MFEM `Mesh::GetQuadOrientation(base, test)`.
    ///
    /// Returns the rotation/flip index `oo ∈ 0..8` used by
    /// `NURBSExtension::NURBSPatchMap::Or2D`, i.e. `2*i` when `test` traverses
    /// the quad in the same sense as `base` and `2*i + 1` when it is reversed,
    /// with `i` the position of `base[0]` inside `test`.
    fn quad_orientation(base: &[usize; 4], test: &[usize; 4]) -> i32 {
        let i = match test.iter().position(|&t| t == base[0]) {
            Some(i) => i,
            // MFEM aborts here; the meshes in the test suite never hit it.
            None => return 0,
        };
        if test[(i + 1) % 4] == base[1] {
            2 * i as i32
        } else {
            2 * i as i32 + 1
        }
    }

    // ── patch knot vectors ────────────────────────────────────────────────────

    /// MFEM `NURBSExtension::GetPatchDirectionEdges` — the (unique) knot vector
    /// index in each parametric direction for patch `p`.
    ///
    /// The comprehensive knot vectors of `CreateComprehensiveKV` differ from the
    /// unique ones only by a possible `Flip`, which leaves `Order`/`NCP`/`NE`
    /// unchanged, so the unique index is enough for every DOF-counting purpose.
    pub fn patch_direction_kv(&self, p: usize) -> Result<Vec<usize>, String> {
        let e = self
            .el_edges
            .get(p)
            .ok_or_else(|| format!("patch_direction_kv: no patch {p}"))?;
        let idx = match self.dim {
            1 => vec![e[0]],
            2 => vec![e[0], e[1]],
            3 => vec![e[0], e[3], e[8]],
            d => return Err(format!("patch_direction_kv: bad dimension {d}")),
        };
        Ok(idx.iter().map(|&e| self.knot_ind(e)).collect())
    }

    /// MFEM `NURBSExtension::KnotInd(edge)` — `UnsignIndex(edge_to_ukv[edge])`:
    /// `x` for `x >= 0` and `-1 - x` for the `FlipIndexSign` encoding.
    pub fn knot_ind(&self, edge: usize) -> usize {
        let v = self.edge_to_ukv[edge];
        if v >= 0 {
            v as usize
        } else {
            (-1 - v) as usize
        }
    }

    /// MFEM `NURBSExtension::KnotSign(edge)`.
    pub fn knot_sign(&self, edge: usize) -> i32 {
        if self.edge_to_ukv[edge] >= 0 {
            1
        } else {
            -1
        }
    }

    /// MFEM `NURBSExtension::GetPatchKnotVectors` (comprehensive vectors, by
    /// unique index).
    pub fn patch_knot_vectors(&self, p: usize) -> Result<Vec<&NurbsKnot>, String> {
        self.patch_direction_kv(p)?
            .into_iter()
            .map(|i| {
                self.knot_vectors
                    .get(i)
                    .ok_or_else(|| format!("patch {p}: knot vector {i} out of range"))
            })
            .collect()
    }

    /// The knot-span indices of patch `p`'s elements, one list per direction —
    /// MFEM's `for (i = 0; i < kv[d]->GetNKS(); i++) if (kv[d]->isElement(i))`
    /// span loop of `Generate{1,2,3}DElementDofTable`.
    ///
    /// These raw indices are what `NURBSExtension::el_to_IJK` stores, so they
    /// are **not** consecutive when a patch's knot vector repeats an interior
    /// knot (`pipe-nurbs.mesh`'s direction-2 knot vector is
    /// `{0, 0, 0, 0.5, 0.5, 1, 1, 1}`, whose two elements sit at indices 0 and 2).
    pub fn patch_element_spans(&self, p: usize) -> Result<Vec<Vec<usize>>, String> {
        let kvs = self.patch_knot_vectors(p)?;
        Ok(kvs
            .iter()
            .map(|kv| (0..kv.nks()).filter(|&i| kv.is_element(i)).collect())
            .collect())
    }

    // ── offsets and counts ────────────────────────────────────────────────────

    /// MFEM `NURBSExtension::GenerateOffsets` + `GetPatchOffsets`.
    ///
    /// Computes both the mesh offsets (`v/e/f/p_meshOffsets`, whose final mesh
    /// counter is `GetGNV()`) and the space offsets (`GetNTotalDof`).
    fn generate_offsets(&mut self) {
        let nv = self.n_topo_vertices;
        self.v_mesh_offsets = (0..nv).collect();
        self.v_space_offsets = (0..nv).collect();
        let mut mesh = nv;
        let mut space = nv;

        // Edges.
        let n_e = self.edge_vertex.len();
        self.e_mesh_offsets = Vec::with_capacity(n_e);
        self.e_space_offsets = Vec::with_capacity(n_e);
        for e in 0..n_e {
            self.e_mesh_offsets.push(mesh);
            self.e_space_offsets.push(space);
            let k = &self.knot_vectors[self.knot_ind(e)];
            mesh += k.n_elements() - 1;
            space += k.ncp() - 2;
        }

        // Faces (3D only: a 2D patch topology has no faces).
        self.f_mesh_offsets = Vec::with_capacity(self.faces.len());
        self.f_space_offsets = Vec::with_capacity(self.faces.len());
        for f in 0..self.faces.len() {
            self.f_mesh_offsets.push(mesh);
            self.f_space_offsets.push(space);
            let e = self.face_edges(f);
            let (a, b) = (self.knot_ind(e[0]), self.knot_ind(e[1]));
            mesh += (self.knot_vectors[a].n_elements() - 1)
                * (self.knot_vectors[b].n_elements() - 1);
            space += (self.knot_vectors[a].ncp() - 2) * (self.knot_vectors[b].ncp() - 2);
        }

        // Patches.
        self.p_mesh_offsets = Vec::with_capacity(self.elements.len());
        self.p_space_offsets = Vec::with_capacity(self.elements.len());
        for p in 0..self.elements.len() {
            self.p_mesh_offsets.push(mesh);
            self.p_space_offsets.push(space);
            let e = self.el_edges[p].clone();
            let k = |i: usize| &self.knot_vectors[self.knot_ind(i)];
            match self.dim {
                1 => {
                    mesh += k(e[0]).n_elements() - 1;
                    space += k(e[0]).ncp() - 2;
                }
                2 => {
                    mesh += (k(e[0]).n_elements() - 1) * (k(e[1]).n_elements() - 1);
                    space += (k(e[0]).ncp() - 2) * (k(e[1]).ncp() - 2);
                }
                3 => {
                    mesh += (k(e[0]).n_elements() - 1)
                        * (k(e[3]).n_elements() - 1)
                        * (k(e[8]).n_elements() - 1);
                    space += (k(e[0]).ncp() - 2) * (k(e[3]).ncp() - 2) * (k(e[8]).ncp() - 2);
                }
                _ => unreachable!(),
            }
        }

        self.n_global_vertices = mesh;
        self.n_total_dofs = space;
    }

    /// MFEM `NURBSExtension::GenerateActiveVertices`: count the mesh-offset
    /// slots that the vertex patch map reaches (`GetNV`).
    fn generate_active_vertices(&mut self) -> Result<(), String> {
        let mut active = vec![false; self.n_global_vertices];
        let d = self.dim;
        for p in 0..self.elements.len() {
            let kvs = self.patch_knot_vectors(p)?;
            // `NURBSPatchMap::nx()` is `I + 1 = GetNE()`.
            let n: Vec<usize> = kvs.iter().map(|k| k.n_elements()).collect();
            for kk in 0..if d == 3 { n[2] } else { 1 } {
                for jj in 0..if d >= 2 { n[1] } else { 1 } {
                    for ii in 0..n[0] {
                        // MFEM enumerates the mesh element's corners in its own
                        // vertex order (`NURBSExtension::GenerateActiveVertices`).
                        let corners: Vec<[usize; 3]> = match d {
                            1 => vec![[ii, 0, 0], [ii + 1, 0, 0]],
                            2 => vec![
                                [ii, jj, 0],
                                [ii + 1, jj, 0],
                                [ii + 1, jj + 1, 0],
                                [ii, jj + 1, 0],
                            ],
                            _ => vec![
                                [ii, jj, kk],
                                [ii + 1, jj, kk],
                                [ii + 1, jj + 1, kk],
                                [ii, jj + 1, kk],
                                [ii, jj, kk + 1],
                                [ii + 1, jj, kk + 1],
                                [ii + 1, jj + 1, kk + 1],
                                [ii, jj + 1, kk + 1],
                            ],
                        };
                        for c in corners {
                            let g = self.patch_map_mode(p, &c[..d], MapMode::Vertex)?;
                            active[g] = true;
                        }
                    }
                }
            }
        }
        self.n_vertices = active.iter().filter(|&&a| a).count();
        Ok(())
    }

    /// MFEM `Mesh::GetFaceEdges(f)` — the edge vertex pairs of a face's stored
    /// vertex cycle, in global edge numbering.
    fn face_edges(&self, f: usize) -> [usize; 4] {
        let v = &self.faces[f];
        let mut out = [0usize; 4];
        for j in 0..4 {
            let (a, b) = (v[j], v[(j + 1) % 4]);
            let key = if a < b { (a, b) } else { (b, a) };
            out[j] = self
                .edge_vertex
                .iter()
                .position(|&p| p == key)
                .unwrap_or_else(|| panic!("face {f}: no global edge for {key:?}"));
        }
        out
    }

    /// MFEM `NURBSExtension::CountElements`.
    fn count_elements(&mut self) {
        let mut total = 0;
        for p in 0..self.elements.len() {
            let kv = self.patch_knot_vectors(p).expect("patch knot vectors");
            let mut ne = kv[0].n_elements();
            for k in &kv[1..] {
                ne *= k.n_elements();
            }
            total += ne;
        }
        self.n_elements = total;
    }

    /// MFEM `Mesh::GenerateBoundaryElements` — used when the mesh file's
    /// `boundary` section is empty (`boundary 0`, as in `pipe-nurbs.mesh`):
    /// every edge/face that belongs to exactly one element becomes a boundary
    /// element, written in that element's local edge/face order.
    fn generate_boundary_elements(&mut self) {
        let d = self.dim;
        match d {
            1 => {
                // A 1D mesh gets one POINT at each end of the single element.
                for el in &self.elements {
                    // D170: MFEM's GenerateBoundaryElements duplicates the
                    // face elements, whose default attribute is 1 (`Element`
                    // ctor) — the parent patch attribute is NOT inherited
                    // (tmp/d170/d170_probe.cpp: pipe-nurbs.mesh, boundary 0
                    // => NBE=24, bdr_attributes={1}).
                    self.boundary.push(TopoElement {
                        attr: 1,
                        geom: GEOM_POINT,
                        verts: vec![el.verts[0]],
                    });
                    self.boundary.push(TopoElement {
                        attr: 1,
                        geom: GEOM_POINT,
                        verts: vec![el.verts[1]],
                    });
                }
            }
            2 => {
                let mut count = vec![0usize; self.edge_vertex.len()];
                for edges in &self.el_edges {
                    for &e in edges {
                        count[e] += 1;
                    }
                }
                for (p, el) in self.elements.iter().enumerate() {
                    for (j, &[a, b]) in SQUARE_EDGES.iter().enumerate() {
                        if count[self.el_edges[p][j]] == 1 {
                            // D170: generated boundary elements carry MFEM's
                            // default attribute 1, not the patch attribute.
                            self.boundary.push(TopoElement {
                                attr: 1,
                                geom: GEOM_SEGMENT,
                                verts: vec![el.verts[a], el.verts[b]],
                            });
                        }
                    }
                }
            }
            _ => {
                let mut count = vec![0usize; self.faces.len()];
                for faces in &self.el_faces {
                    for &f in faces {
                        count[f] += 1;
                    }
                }
                for el in &self.elements {
                    for fv in CUBE_FACE_VERT.iter() {
                        let tuple = [el.verts[fv[0]], el.verts[fv[1]], el.verts[fv[2]], el.verts[fv[3]]];
                        let mut key = tuple.to_vec();
                        key.sort_unstable();
                        let f = self
                            .faces
                            .iter()
                            .position(|face| {
                                let mut k = face.to_vec();
                                k.sort_unstable();
                                k == key
                            })
                            .expect("face must exist");
                        if count[f] == 1 {
                            self.boundary.push(TopoElement {
                                // D170: MFEM's generated boundary elements
                                // carry the default attribute 1, not the
                                // patch attribute (tmp/d170/d170_probe.cpp).
                                attr: 1,
                                geom: GEOM_SQUARE,
                                verts: tuple.to_vec(),
                            });
                        }
                    }
                }
            }
        }
    }

    /// MFEM `NURBSExtension::CountBdrElements` with `GetBdrPatchKnotVectors`.
    fn count_bdr_elements(&mut self) {
        let mut total = 0;
        for (bp, _) in self.boundary.iter().enumerate() {
            let kv = self.bdr_patch_knot_vectors(bp);
            let mut ne = 1;
            for k in &kv {
                ne *= k.n_elements();
            }
            total += ne;
        }
        self.n_bdr_elements = total;
    }

    /// MFEM `NURBSExtension::GetBdrPatchKnotVectors` (unique indices).
    ///
    /// `Mesh::GetBdrElementEdges` gives the boundary element's edges in its own
    /// vertex cycle order, so the first two edges are `(v0,v1)` and `(v1,v2)`.
    pub fn bdr_patch_knot_vectors(&self, bp: usize) -> Vec<&NurbsKnot> {
        let be = &self.boundary[bp];
        match self.dim {
            // 1D boundary elements are points: `CountBdrElements` contributes 1
            // per boundary patch and no knot vector is involved.
            1 => Vec::new(),
            2 => vec![&self.knot_vectors[self.knot_ind(self.find_edge(be.verts[0], be.verts[1]))]],
            _ => vec![
                &self.knot_vectors[self.knot_ind(self.find_edge(be.verts[0], be.verts[1]))],
                &self.knot_vectors[self.knot_ind(self.find_edge(be.verts[1], be.verts[2]))],
            ],
        }
    }

    fn find_edge(&self, a: usize, b: usize) -> usize {
        let key = if a < b { (a, b) } else { (b, a) };
        self.edge_vertex
            .iter()
            .position(|&p| p == key)
            .unwrap_or_else(|| panic!("no global edge for vertex pair {key:?}"))
    }

    // ── element DOF table ─────────────────────────────────────────────────────

    /// MFEM `NURBSExtension::GenerateElementDofTable` for a conforming mesh
    /// whose elements are all active.
    fn generate_element_dof_table(&mut self) -> Result<(), String> {
        let mut el_dof: Vec<Vec<usize>> = Vec::new();
        let mut el_to_patch = Vec::new();
        let mut el_to_ijk = Vec::new();

        // `activeDof[glob] = 1` for every DOF touched by an element, compacted
        // afterwards exactly as `GenerateElementDofTable` does.
        let mut active = vec![false; self.n_total_dofs];

        for p in 0..self.elements.len() {
            let kv_idx = self.patch_direction_kv(p)?;
            let kvs: Vec<NurbsKnot> = kv_idx.iter().map(|&i| self.knot_vectors[i].clone()).collect();
            let d = self.dim;

            let ord: Vec<usize> = kvs.iter().map(|k| k.order()).collect();

            // Nested span loop in MFEM's order: for 3D `(k, j, i)` with the
            // first direction innermost, for 2D `(j, i)`, for 1D `i`.  The
            // indices are the *raw* knot-span indices of `NURBSFiniteElement::ijk`.
            let ranges: Vec<Vec<usize>> = self.patch_element_spans(p)?;

            // The span loops mirror MFEM's nesting: 3D is `(k, j, i)` with `i`
            // innermost, 2D is `(j, i)` and 1D is just `i`.
            let n_k = if d == 3 { ranges[2].len() } else { 1 };
            let n_j = if d >= 2 { ranges[1].len() } else { 1 };
            for kk in 0..n_k {
                for jj in 0..n_j {
                    for ii in 0..ranges[0].len() {
                        let idx = if d == 3 {
                            [ranges[0][ii], ranges[1][jj], ranges[2][kk]]
                        } else if d == 2 {
                            [ranges[0][ii], ranges[1][jj], 0]
                        } else {
                            [ranges[0][ii], 0, 0]
                        };

                        let mut dofs = Vec::new();
                        // MFEM iterates the multi-index with the *first*
                        // direction innermost, each running `0..=order`.
                        let counters: Vec<usize> = (0..d).map(|dd| ord[dd] + 1).collect();
                        let mut c = vec![0usize; d];
                        loop {
                            let multi: Vec<usize> = (0..d).map(|dd| idx[dd] + c[dd]).collect();
                            let g = self.dof_map(self.patch_map_mode(p, &multi, MapMode::Dof)?);
                            active[g] = true;
                            dofs.push(g);
                            // Increment the innermost (first-direction) counter.
                            let mut carry = 0;
                            while carry < d {
                                c[carry] += 1;
                                if c[carry] < counters[carry] {
                                    break;
                                }
                                c[carry] = 0;
                                carry += 1;
                            }
                            if carry == d {
                                break;
                            }
                        }

                        el_to_patch.push(p);
                        el_to_ijk.push(idx);
                        el_dof.push(dofs);
                    }
                }
            }
        }

        // Compact: `activeDof[d] = ++NumOfActiveDofs` for every active DOF.
        let mut map = vec![usize::MAX; self.n_total_dofs];
        let mut n_active = 0;
        for (d, a) in active.iter().enumerate() {
            if *a {
                n_active += 1;
                map[d] = n_active - 1;
            }
        }
        for row in el_dof.iter_mut() {
            for g in row.iter_mut() {
                *g = map[*g];
            }
        }

        self.el_dof = el_dof;
        self.el_to_patch = el_to_patch;
        self.el_to_ijk = el_to_ijk;
        self.n_dofs = n_active;
        self.active_dof = map;
        Ok(())
    }

    /// MFEM `NURBSPatchMap::operator()(i)` / `(i, j)` / `(i, j, k)` — the
    /// patch-local knot-multi-index to global DOF map, evaluated with `f = p`
    /// the owning patch.
    ///
    /// `multi` holds the multi-index in `(x, y, z)` order; the values are
    /// `0..=NCP[d]` with the boundary slots `0` / `NCP-1` addressing vertices.
    fn patch_map_mode(&self, p: usize, multi: &[usize], mode: MapMode) -> Result<usize, String> {
        let d = self.dim;
        let kvs = self.patch_knot_vectors(p)?;
        let e = &self.el_edges[p];
        let (v_off, e_off, f_off, p_off) = match mode {
            MapMode::Vertex => (
                &self.v_mesh_offsets,
                &self.e_mesh_offsets,
                &self.f_mesh_offsets,
                &self.p_mesh_offsets,
            ),
            MapMode::Dof => (
                &self.v_space_offsets,
                &self.e_space_offsets,
                &self.f_space_offsets,
                &self.p_space_offsets,
            ),
        };
        let verts: Vec<usize> = self.elements[p].verts.iter().map(|&v| v_off[v]).collect();
        let edges: Vec<usize> = e.iter().map(|&x| e_off[x]).collect();
        let faces: Vec<usize> = self.el_faces[p].iter().map(|&x| f_off[x]).collect();
        let p_offset = p_off[p];

        // MFEM `NURBSPatchMap::SetPatchVertexMap` uses `I = GetNE() - 1`;
        // `SetPatchDofMap` uses `I = GetNCP() - 2`.
        let n: Vec<usize> = match mode {
            MapMode::Vertex => kvs.iter().map(|k| k.n_elements() - 1).collect(),
            MapMode::Dof => kvs.iter().map(|k| k.ncp() - 2).collect(),
        };

        // `F(n, N)` classifies an index relative to the interior range.
        let f = |m: isize, nn: isize| -> usize {
            if m < 0 {
                0
            } else if m >= nn {
                2
            } else {
                1
            }
        };
        // `Or1D(n, N, Or)`.
        let or1d = |m: isize, nn: isize, or: i32| -> usize {
            if or > 0 {
                m as usize
            } else {
                (nn - 1 - m) as usize
            }
        };
        // `Or2D(n1, n2, N1, N2, Or)`.
        let or2d = |m: isize, nn: isize, n1: isize, n2: isize, or: i32| -> usize {
            let (m, nn) = (m as usize, nn as usize);
            let (n1u, n2u) = (n1 as usize, n2 as usize);
            match or {
                0 => m + nn * n1u,
                1 => nn + m * n2u,
                2 => nn + (n1u - 1 - m) * n2u,
                3 => (n1u - 1 - m) + nn * n1u,
                4 => (n1u - 1 - m) + (n2u - 1 - nn) * n1u,
                5 => (n2u - 1 - nn) + (n1u - 1 - m) * n2u,
                6 => (n2u - 1 - nn) + m * n2u,
                _ => m + (n2u - 1 - nn) * n1u,
            }
        };

        // `EC(e, n, N, s)` with `s = 1` (conforming: `edgeMaster` is false).
        let ec = |e_local: usize, m: isize, nn: isize, s: i32| -> usize {
            let oedge = self.el_edge_sign[p][e_local];
            edges[e_local] + or1d(m, nn, s * oedge)
        };
        // `FC(f, m, n, M, N)` for a conforming mesh (`faceMaster` is false).
        let fc = |f_local: usize, m: isize, nn: isize, m1: isize, n2: isize| -> usize {
            let oface = self.el_face_ori[p][f_local];
            faces[f_local] + or2d(m, nn, m1, n2, oface)
        };
        // `FCP(f, m, n, M, N)` — `faceMaster.Size() == 0` takes the pOffset path.
        let fcp = |_f_local: usize, m: isize, nn: isize, m1: isize, n2: isize| -> usize {
            p_offset + or2d(m, nn, m1, n2, 0)
        };

        let (i, j, k) = (multi[0] as isize, multi.get(1).copied().unwrap_or(0) as isize, multi.get(2).copied().unwrap_or(0) as isize);
        let (ni, nj, nk) = (
            n.first().copied().unwrap_or(0) as isize,
            n.get(1).copied().unwrap_or(0) as isize,
            n.get(2).copied().unwrap_or(0) as isize,
        );

        let out = if d == 1 {
            let i1 = i - 1;
            match f(i1, ni) {
                0 => verts[0],
                1 => p_offset + or1d(i1, ni, 0),
                _ => verts[1],
            }
        } else if d == 2 {
            let (i1, j1) = (i - 1, j - 1);
            match 3 * f(j1, nj) + f(i1, ni) {
                0 => verts[0],
                1 => ec(0, i1, ni, 1),
                2 => verts[1],
                3 => ec(3, j1, nj, -1),
                4 => fcp(0, i1, j1, ni, nj),
                5 => ec(1, j1, nj, 1),
                6 => verts[3],
                7 => ec(2, i1, ni, -1),
                _ => verts[2],
            }
        } else {
            let (i1, j1, k1) = (i - 1, j - 1, k - 1);
            match 3 * (3 * f(k1, nk) + f(j1, nj)) + f(i1, ni) {
                0 => verts[0],
                1 => ec(0, i1, ni, 1),
                2 => verts[1],
                3 => ec(3, j1, nj, 1),
                4 => fc(0, i1, nj - 1 - j1, ni, nj),
                5 => ec(1, j1, nj, 1),
                6 => verts[3],
                7 => ec(2, i1, ni, 1),
                8 => verts[2],
                9 => ec(8, k1, nk, 1),
                10 => fc(1, i1, k1, ni, nk),
                11 => ec(9, k1, nk, 1),
                12 => fc(4, nj - 1 - j1, k1, nj, nk),
                13 => {
                    p_offset
                        + ni as usize
                            * (nj as usize * k1 as usize + j1 as usize)
                        + i1 as usize
                }
                14 => fc(2, j1, k1, nj, nk),
                15 => ec(11, k1, nk, 1),
                16 => fc(3, ni - 1 - i1, k1, ni, nk),
                17 => ec(10, k1, nk, 1),
                18 => verts[4],
                19 => ec(4, i1, ni, 1),
                20 => verts[5],
                21 => ec(7, j1, nj, 1),
                22 => fc(5, i1, j1, ni, nj),
                23 => ec(5, j1, nj, 1),
                24 => verts[7],
                25 => ec(6, i1, ni, 1),
                _ => verts[6],
            }
        };
        Ok(out)
    }

    // ── accessors (MFEM `NURBSExtension` public API) ──────────────────────────

    /// MFEM `NURBSExtension::Dimension` — the patch topology dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// MFEM `NURBSExtension::GetOrder` — the common knot-vector order, or `None`
    /// for `NURBSFECollection::VariableOrder`.
    pub fn order(&self) -> Option<usize> {
        self.order
    }

    /// MFEM `NURBSExtension::GetOrders`.
    pub fn orders(&self) -> &[usize] {
        &self.orders
    }

    /// MFEM `NURBSExtension::GetNKV`.
    pub fn n_knot_vectors(&self) -> usize {
        self.knot_vectors.len()
    }

    /// MFEM `NURBSExtension::GetKnotVector(i)`.
    pub fn knot_vector(&self, i: usize) -> &NurbsKnot {
        &self.knot_vectors[i]
    }

    /// MFEM `NURBSExtension::GetNP` — number of patches.
    pub fn n_patches(&self) -> usize {
        self.elements.len()
    }

    /// MFEM `NURBSExtension::GetNBP` — number of boundary patches.
    pub fn n_bdr_patches(&self) -> usize {
        self.boundary.len()
    }

    /// MFEM `NURBSExtension::GetGNE` — total elements over all patches.
    pub fn n_global_elements(&self) -> usize {
        self.n_elements
    }

    /// MFEM `NURBSExtension::GetNE` — active elements.
    pub fn n_elements(&self) -> usize {
        self.n_elements
    }

    /// MFEM `NURBSExtension::GetGNBE`.
    pub fn n_global_bdr_elements(&self) -> usize {
        self.n_bdr_elements
    }

    /// MFEM `NURBSExtension::GetNBE`.
    pub fn n_bdr_elements(&self) -> usize {
        self.n_bdr_elements
    }

    /// MFEM `mesh->bdr_attributes.Max()` — the largest boundary attribute, i.e.
    /// the number of entries of the `ess_bdr`/`neu_bdr`/`per_bdr` marker arrays
    /// (`nurbs_ex1` prints those arrays).
    pub fn max_bdr_attribute(&self) -> i32 {
        self.boundary.iter().map(|b| b.attr).max().unwrap_or(0)
    }

    /// For every boundary element, the patch-boundary entity it lies on:
    /// `(patch, direction, low, attribute)` with `low` marking the
    /// minimum-parameter side and `attribute` the mesh boundary attribute
    /// (`GetBdrAttribute`).
    ///
    /// This is the `[direction, side]` an element's `NURBSFiniteElement` carries
    /// in MFEM (`NURBSPatchMap::SetBdrPatchVertexMap`'s orientation): a boundary
    /// element of patch `p` on the `low`/`high` side of direction `d` spans the
    /// control points `multi[d] == 0` / `multi[d] == NCP_d - 1`, which is exactly
    /// the information `GetEssentialTrueDofs` needs.  The attribute is what
    /// `GetEssentialVDofs` tests against `bdr_attr_is_ess[GetBdrAttribute(i)-1]`,
    /// i.e. what a *partial* essential mask needs.
    pub fn boundary_sides(&self) -> &[BdrSide] {
        &self.bdr_sides
    }

    /// The boundary element's own vertices *after*
    /// `Mesh::CheckBdrElementOrientation` — MFEM mutates a boundary element so
    /// that its vertex cycle matches the face it lies on, and
    /// `NURBSPatchMap`/`GetBdrElementEdges` read that corrected order.
    ///
    /// 2-D: the element's face is the edge, whose stored cycle is the owning
    /// element's local edge vertex pair; 3-D: the boundary element is flipped
    /// (`bv[0] <-> bv[2]`) when its orientation w.r.t. the face is odd.
    pub fn bdr_element_vertices(&self, bp: usize) -> Vec<usize> {
        let be = &self.boundary[bp];
        if self.dim == 2 {
            let side = &self.bdr_sides[bp];
            let [a, b] = SQUARE_EDGES[side.local];
            let el = &self.elements[side.patch];
            return vec![el.verts[a], el.verts[b]];
        }
        let mut bv = be.verts.clone();
        if self.dim == 3 && bv.len() == 4 {
            let face = self.find_face(&bv).expect("boundary element face");
            // `CheckBdrElementOrientation`: odd orientation w.r.t. the face.
            if Self::quad_orientation(&self.faces[face], &[bv[0], bv[1], bv[2], bv[3]]) % 2 != 0 {
                bv.swap(0, 2);
            }
        }
        bv
    }

    /// The global face whose vertex set matches `bv` (MFEM `be_to_face` for a
    /// boundary element, evaluated by vertex set).
    fn find_face(&self, bv: &[usize]) -> Option<usize> {
        let mut key = bv.to_vec();
        key.sort_unstable();
        self.faces.iter().position(|f| {
            let mut k = f.to_vec();
            k.sort_unstable();
            k == key
        })
    }

    /// [`Self::boundary_sides`] from the boundary elements' own edges/faces: a
    /// boundary element is a mesh edge (2-D) or face (3-D), and the element that
    /// contains it fixes the local entity index, hence the direction and side.
    /// This works for boundary elements read from the file *and* for the ones
    /// [`Self::generate_boundary_elements`] synthesises, exactly as MFEM's
    /// `NURBSExtension::GenerateBdrElementDofTable` derives the boundary patch
    /// from the boundary element's own vertices.
    fn compute_bdr_sides(&mut self) {
        self.bdr_sides.clear();
        let d = self.dim;
        if d == 1 {
            // A 1-D boundary element is a point at one end of its patch's
            // segment: the `low`/`high` side of direction 0 is decided by which
            // end of the patch element's own vertex pair it is (MFEM's
            // `SetBdrPatchDofMap` maps the point through `v_spaceOffsets`, so
            // `NURBSPatchMap::operator()(0)` returns exactly that endpoint's
            // control point; `Generate1DBdrElementDofTable` then records it as
            // the boundary element's single DOF).
            for be in &self.boundary {
                if be.verts.len() != 1 {
                    continue;
                }
                let v = be.verts[0];
                for (p, el) in self.elements.iter().enumerate() {
                    if el.verts.len() == 2 && (el.verts[0] == v || el.verts[1] == v) {
                        self.bdr_sides.push(BdrSide {
                            patch: p,
                            dir: 0,
                            low: el.verts[0] == v,
                            attr: be.attr,
                            local: 0,
                        });
                        break;
                    }
                }
            }
        } else if d == 2 {
            // Element local edge -> the element that owns it (every mesh edge
            // reaches at most one element here; interior edges are skipped).
            let mut owner: Vec<Option<(usize, usize)>> = vec![None; self.edge_vertex.len()];
            for (p, edges) in self.el_edges.iter().enumerate() {
                for (j, &e) in edges.iter().enumerate() {
                    owner[e].get_or_insert((p, j));
                }
            }
            for be in &self.boundary {
                if be.verts.len() != 2 {
                    continue;
                }
                let e = self.find_edge(be.verts[0], be.verts[1]);
                if let Some((p, j)) = owner[e] {
                    let (dir, low) = quad_edge_side(j);
                    self.bdr_sides.push(BdrSide { patch: p, dir, low, attr: be.attr, local: j });
                }
            }
        } else if d == 3 {
            let mut key_to_face: Vec<(Vec<usize>, usize)> = self
                .faces
                .iter()
                .enumerate()
                .map(|(f, vs)| {
                    let mut k = vs.to_vec();
                    k.sort_unstable();
                    (k, f)
                })
                .collect();
            key_to_face.sort_unstable();
            // Global face -> (element, local face index), first owner wins.
            let mut owner: Vec<Option<(usize, usize)>> = vec![None; self.faces.len()];
            for (p, faces) in self.el_faces.iter().enumerate() {
                for (k, &f) in faces.iter().enumerate() {
                    owner[f].get_or_insert((p, k));
                }
            }
            for be in &self.boundary {
                if be.verts.len() != 4 {
                    continue;
                }
                let mut key = be.verts.clone();
                key.sort_unstable();
                let Ok(pos) = key_to_face.binary_search_by(|(k, _)| k.as_slice().cmp(&key)) else {
                    continue;
                };
                let f = key_to_face[pos].1;
                if let Some((p, k)) = owner[f] {
                    let (dir, low) = hex_face_side(k);
                    self.bdr_sides.push(BdrSide { patch: p, dir, low, attr: be.attr, local: k });
                }
            }
        }
    }

    /// MFEM `NURBSExtension::GenerateBdrElementDofTable` —
    /// `GetBdrElementDofTable` for the given space `mode`, one row per *mesh*
    /// boundary element (the `(patch-topology boundary entity, knot span)` pair
    /// in MFEM's enumeration order), each entry the global DOF (`>= 0`) or the
    /// negated-DOF encoding `-1 - dof` (see [`unsign_dof`]).
    ///
    /// Mirrors `Generate{1,2,3}DBdrElementDofTable` followed by its
    /// sign/compaction pass: `dof.i = activeDof[i] - 1` for `i >= 0`,
    /// `dof.i = -(activeDof[FlipIndexSign(i)])` otherwise — which for a compact
    /// `0..NDof` numbering is exactly `FlipIndexSign(dof)`.
    ///
    /// `mode` selects MFEM's `NURBSExtension::Mode` semantics:
    ///
    /// * `H1` — every control point of the boundary entity, all signs `+`.
    /// * `HDiv` — only the DOF block of the component **normal** to the entity,
    ///   negated on the **low** side of that component's direction.  MFEM tests
    ///   `fn == 0 || fn == 2` (2-D, `fn = be_to_face`, the index of the boundary
    ///   element's edge in the mesh file's `edges` table) resp.
    ///   `fn ∈ {0, 1, 4}` (3-D, `fn` the local face index), where `fn` for a
    ///   single-patch mesh is by construction the low side of each direction;
    ///   the probe comparison in `tests/nurbs_bdr_dofs.rs` verifies the two
    ///   formulations agree row by row on every usable mesh.
    /// * `HCurl` — only the DOF block of the component **tangential** to the
    ///   entity: MFEM drops a component's block when the entity's own
    ///   (tangential) knot-vector order equals the extension's largest order
    ///   (`ord0 == mOrders.Max()` in 2-D, `ord0 == ord1` in 3-D), which for
    ///   `GetCurlExtension` is exactly "this is not the entity's direction".
    pub fn boundary_dof_table(&self, mode: BdrDofMode) -> Vec<Vec<i64>> {
        match self.dim {
            1 => self.bdr_dof_table_1d(),
            2 => self.bdr_dof_table_2d(mode),
            _ => self.bdr_dof_table_3d(mode),
        }
    }

    /// The boundary FE's own reference directions of the boundary element with
    /// local knot spans `spans` (one entry per reference direction, in the
    /// boundary element's own order) on the patch-boundary entity `bp`, as
    /// `(patch direction, signed span index)`.
    ///
    /// This is the `SetIJK(bel_to_IJK.GetRow(i))` half of
    /// `NURBSExtension::LoadBE`: MFEM's `NURBSPatchMap::SetBdrPatchDofMap` maps
    /// the boundary element's own edges through `KnotVec(edge, oedge, &okv)`
    /// and `Generate{2,3}DBdrElementDofTable` stores
    /// `bel_to_IJK(j) = (okv_j >= 0) ? i_j : FlipIndexSign(i_j)` — the *signed*
    /// span index the boundary `NURBSFiniteElement` runs
    /// `KnotVector::CalcShape(shape, i, xi)` on.  A negative `i` mirrors the
    /// reference coordinate (`1 - xi`) and selects the same knot span
    /// (`ip = -1 - i + Order`), which is what makes the boundary element's
    /// parameterization follow its own vertex cycle.
    ///
    /// Only meaningful for a single-patch mesh, where a knot-vector index
    /// determines its patch direction (the H(div)/H(curl) NURBS spaces are
    /// single-patch only anyway).
    pub fn bdr_element_span(&self, bp: usize, spans: &[usize]) -> Vec<(usize, i64)> {
        let bv = self.bdr_element_vertices(bp);
        let dir_kvs =
            self.patch_direction_kv(self.bdr_sides[bp].patch).expect("patch knot vectors");
        (0..spans.len())
            .map(|j| {
                let v0 = bv[j];
                let v1 = bv[(j + 1) % bv.len()];
                let e = self.find_edge(v0, v1);
                let dir = dir_kvs
                    .iter()
                    .position(|&k| k == self.knot_ind(e))
                    .expect("boundary knot vector is a patch direction");
                let oedge = if v0 < v1 { 1 } else { -1 };
                let okv = self.knot_sign(e) * oedge;
                let i = spans[j] as i64;
                (dir, if okv >= 0 { i } else { -1 - i })
            })
            .collect()
    }

    /// MFEM `Generate1DBdrElementDofTable`: one DOF per boundary point, no
    /// mode-dependent filtering or sign.
    fn bdr_dof_table_1d(&self) -> Vec<Vec<i64>> {
        let mut rows = Vec::with_capacity(self.boundary.len());
        for be in &self.boundary {
            let v = be.verts[0];
            let mut row: Vec<i64> = Vec::new();
            for (p, el) in self.elements.iter().enumerate() {
                if el.verts.len() == 2 && (el.verts[0] == v || el.verts[1] == v) {
                    // `DofMap(p2g[0])`: `operator()(0)` is `verts[0]`, i.e. the
                    // patch vertex the boundary point coincides with.
                    let _ = p;
                    row.push(self.dof_map(self.v_space_offsets[v]) as i64);
                    break;
                }
            }
            rows.push(row);
        }
        rows
    }

    /// MFEM `Generate2DBdrElementDofTable`.
    fn bdr_dof_table_2d(&self, mode: BdrDofMode) -> Vec<Vec<i64>> {
        let mut rows: Vec<Vec<i64>> = Vec::new();
        let max_order = *self.orders.iter().max().expect("orders");
        for bp in 0..self.boundary.len() {
            let side = &self.bdr_sides[bp];
            let m = self.bdr_seg_dof_map(bp);
            let (nx, ord) = (m.nx, m.kv.order());
            let (add_dofs, s) = match mode {
                BdrDofMode::H1 => (true, 1),
                BdrDofMode::HDiv => (ord != max_order, if side.low { -1 } else { 1 }),
                BdrDofMode::HCurl => (ord != max_order, 1),
            };
            for i in 0..m.kv.nks() {
                if !m.kv.is_element(i) {
                    continue;
                }
                let mut row = Vec::new();
                if add_dofs {
                    for ii in 0..=ord {
                        let j =
                            if m.okv >= 0 { i + ii } else { (nx - i as isize - ii as isize) as usize };
                        let g = self.dof_map(self.bdr_seg_dof(&m, j));
                        row.push(if s < 0 { -1 - g as i64 } else { g as i64 });
                    }
                }
                rows.push(row);
            }
        }
        rows
    }

    /// `NURBSPatchMap::SetBdrPatchDofMap(bp, …)` for a 2-D patch, whose boundary
    /// patch is a single segment read with `operator()(i)` (`BdrSegDofMap::dof`).
    fn bdr_seg_dof_map(&self, bp: usize) -> BdrSegDofMap<'_> {
        let bv = self.bdr_element_vertices(bp);
        debug_assert_eq!(bv.len(), 2, "a 2-D boundary element is a segment");
        let edge = self.find_edge(bv[0], bv[1]);
        // `KnotVec(edge, oedge, &okv)` with `oedge = cor[0]`.
        let oedge = if bv[0] < bv[1] { 1 } else { -1 };
        let kv = &self.knot_vectors[self.knot_ind(edge)];
        BdrSegDofMap {
            verts: [self.v_space_offsets[bv[0]], self.v_space_offsets[bv[1]]],
            p_offset: self.e_space_offsets[edge],
            nx: kv.ncp() as isize - 1,
            i_cap: kv.ncp() as isize - 2,
            kv,
            okv: self.knot_sign(edge) * oedge,
            oedge,
        }
    }

    /// `NURBSPatchMap::operator()(i)` for a boundary *segment* whose patch map
    /// was set up by `SetBdrPatchDofMap` (`edgeMaster` is empty for a conforming
    /// extension, so the offset path is always taken).
    fn bdr_seg_dof(&self, m: &BdrSegDofMap<'_>, j: usize) -> usize {
        let i1 = j as isize - 1;
        let f = |n: isize, big_n: isize| -> usize {
            if n < 0 {
                0
            } else if n >= big_n {
                2
            } else {
                1
            }
        };
        match f(i1, m.i_cap) {
            0 => m.verts[0],
            1 => {
                let or = if m.oedge > 0 { i1 } else { m.i_cap - 1 - i1 };
                m.p_offset + or as usize
            }
            _ => m.verts[1],
        }
    }

    /// MFEM `Generate3DBdrElementDofTable`.
    fn bdr_dof_table_3d(&self, mode: BdrDofMode) -> Vec<Vec<i64>> {
        let mut rows: Vec<Vec<i64>> = Vec::new();
        for bp in 0..self.boundary.len() {
            let side = &self.bdr_sides[bp];
            let m = self.bdr_quad_dof_map(bp);
            let (ord0, ord1) = (m.kvs[0].order(), m.kvs[1].order());
            // `add_dofs` is false when the entity's two knot vectors have
            // different orders (`H_DIV`) resp. the same one (`H_CURL`).
            let add_dofs = match mode {
                BdrDofMode::H1 => true,
                BdrDofMode::HDiv => ord0 == ord1,
                BdrDofMode::HCurl => ord0 != ord1,
            };
            let s = match mode {
                BdrDofMode::HDiv if side.low => -1,
                _ => 1,
            };
            let (nxs, nys) = (m.nx, m.ny);
            for j in 0..m.kvs[1].nks() {
                if !m.kvs[1].is_element(j) {
                    continue;
                }
                for i in 0..m.kvs[0].nks() {
                    if !m.kvs[0].is_element(i) {
                        continue;
                    }
                    let mut row = Vec::new();
                    if add_dofs {
                        for jj in 0..=ord1 {
                            let jj_ = if m.okv[1] >= 0 {
                                j + jj
                            } else {
                                (nys - j as isize - jj as isize) as usize
                            };
                            for ii in 0..=ord0 {
                                let ii_ = if m.okv[0] >= 0 {
                                    i + ii
                                } else {
                                    (nxs - i as isize - ii as isize) as usize
                                };
                                let g = self.dof_map(m.dof(ii_, jj_));
                                row.push(if s < 0 { -1 - g as i64 } else { g as i64 });
                            }
                        }
                    }
                    rows.push(row);
                }
            }
        }
        rows
    }

    /// `NURBSPatchMap::SetBdrPatchDofMap(bp, …)` for a 3-D patch, whose boundary
    /// patch is a quadrilateral read with `operator()(i, j)`.
    ///
    /// `edgeMaster`/`faceMaster` are empty for a conforming extension —
    /// `NURBSExtension::IsMasterEdge` returns false — so `EC`/`FC` take the
    /// offset path and `FCP` the `pOffset` one; `BdrQuadDofMap::dof` implements
    /// exactly that.
    fn bdr_quad_dof_map(&self, bp: usize) -> BdrQuadDofMap<'_> {
        let bv = self.bdr_element_vertices(bp);
        debug_assert_eq!(bv.len(), 4, "a 3-D boundary element is a quadrilateral");
        let face = self.find_face(&bv).expect("boundary element face");
        let opatch = Self::quad_orientation(&self.faces[face], &[bv[0], bv[1], bv[2], bv[3]]);
        // `Mesh::GetBdrElementEdges`: the boundary element's own local edges and
        // their `cor` orientations.
        let bdr_edges = [0usize, 1, 2, 3].map(|j| self.find_edge(bv[j], bv[(j + 1) % 4]));
        let oedge: [i32; 4] = [0, 1, 2, 3].map(|j| if bv[j] < bv[(j + 1) % 4] { 1 } else { -1 });
        // `KnotVec(edges[j], oedge[j], &okv[j])`, `j = 0, 1`.
        let kvs = [0usize, 1].map(|j| &self.knot_vectors[self.knot_ind(bdr_edges[j])]);
        let okv = [0usize, 1].map(|j| self.knot_sign(bdr_edges[j]) * oedge[j]);
        let verts = [0usize, 1, 2, 3].map(|j| self.v_space_offsets[bv[j]]);
        let edges = [0usize, 1, 2, 3].map(|j| self.e_space_offsets[bdr_edges[j]]);
        BdrQuadDofMap {
            verts,
            edges,
            oedge,
            // `p2g.nx()/ny()` are `I + 1` = `GetNCP() - 1`.
            nx: kvs[0].ncp() as isize - 1,
            ny: kvs[1].ncp() as isize - 1,
            i_cap: kvs[0].ncp() as isize - 2,
            j_cap: kvs[1].ncp() as isize - 2,
            kvs,
            okv,
            opatch,
            p_offset: self.f_space_offsets[face],
        }
    }

    /// MFEM `NURBSExtension::ConnectBoundaries(Array<int>&, Array<int>&)` — the
    /// `-pm`/`-ps` periodic boundary conditions of `nurbs_ex1`.
    ///
    /// Every pair `(master[i], slave[i])` names two *mesh boundary attributes*
    /// whose boundary patches are identified DOF by DOF, so that the space
    /// becomes periodic across them.  The pairing is resolved through
    /// `patchTopo->GetBdrAttribute` exactly as MFEM does — including that the
    /// **last** boundary element carrying the attribute wins, and that an
    /// attribute with no boundary element aborts (`Bdr N not found`).
    ///
    /// MFEM's compaction is mirrored literally: `d_to_d` values are resolved in
    /// increasing order of the *target* index, so the surviving DOFs keep their
    /// relative order, and the element/boundary DOF tables are regenerated with
    /// [`Self::dof_map`] in force.  `weights` keeps its pre-merge length, as in
    /// MFEM (`LoadFE` indexes it by the merged DOFs, which are `<=` the old
    /// count).
    pub fn connect_boundaries(
        &mut self,
        master: &[i32],
        slave: &[i32],
    ) -> Result<(), String> {
        if master.len() != slave.len() {
            return Err(
                "NURBSExtension::ConnectBoundaries() boundary lists not of equal size"
                    .to_string(),
            );
        }
        if master.is_empty() {
            return Ok(());
        }

        // Initialize d_to_d.  It is indexed by the **un-compacted** DOF numbering
        // of `NURBSPatchMap` — MFEM sizes it by `NumOfDofs`, which is
        // `GetNTotalDof()` at this point in the constructor, not by the active
        // count `GetNDof()`.
        let mut d_to_d: Vec<usize> = (0..self.n_total_dofs).collect();

        for i in 0..master.len() {
            let (mut bnd0, mut bnd1) = (None, None);
            for b in 0..self.boundary.len() {
                if master[i] == self.boundary[b].attr {
                    bnd0 = Some(b);
                }
                if slave[i] == self.boundary[b].attr {
                    bnd1 = Some(b);
                }
            }
            let bnd0 = bnd0.ok_or_else(|| "Bdr 0 not found".to_string())?;
            let bnd1 = bnd1.ok_or_else(|| "Bdr 1 not found".to_string())?;

            match self.dim {
                1 => self.connect_boundaries_1d(&mut d_to_d, bnd0, bnd1),
                2 => self.connect_boundaries_2d(&mut d_to_d, bnd0, bnd1)?,
                _ => self.connect_boundaries_3d(&mut d_to_d, bnd0, bnd1)?,
            }
        }

        // Clean d_to_d: compact the target indices in increasing order.
        let mut tmp = vec![0i32; d_to_d.len() + 1];
        for &d in d_to_d.iter() {
            tmp[d] = 1;
        }
        let mut cnt = 0usize;
        for t in tmp.iter_mut() {
            if *t == 1 {
                *t = cnt as i32;
                cnt += 1;
            }
        }
        for d in d_to_d.iter_mut() {
            *d = tmp[*d] as usize;
        }

        self.d_to_d = d_to_d;
        // Finalize: `GenerateElementDofTable()` + `GenerateBdrElementDofTable()`;
        // the latter is evaluated on demand by `boundary_dof_table`, which now
        // routes through `dof_map`.
        self.generate_element_dof_table()?;
        Ok(())
    }

    /// MFEM `NURBSExtension::ConnectBoundaries1D`: the two boundary points
    /// coincide with one control point each (`NURBSPatchMap::operator()(0)` is
    /// `verts[0]`, the `I = 0` vertex case).
    fn connect_boundaries_1d(&self, d_to_d: &mut [usize], bnd0: usize, bnd1: usize) {
        let p0 = self.v_space_offsets[self.boundary[bnd0].verts[0]];
        let p1 = self.v_space_offsets[self.boundary[bnd1].verts[0]];
        d_to_d[p0] = d_to_d[p1];
    }

    /// MFEM `NURBSExtension::ConnectBoundaries2D`: walk the boundary segment's
    /// knot spans and pair the control points, honouring each boundary patch's
    /// own orientation (`okv`).
    ///
    /// `nx` is `p2g0.nx()` and MFEM uses it for **both** maps (it is `I + 1`,
    /// with `I` from the first map only) — mirrored here.
    fn connect_boundaries_2d(
        &self,
        d_to_d: &mut [usize],
        bnd0: usize,
        bnd1: usize,
    ) -> Result<(), String> {
        let s0 = self.bdr_seg_dof_map(bnd0);
        let s1 = self.bdr_seg_dof_map(bnd1);
        let nx = s0.nx;

        for i in 0..s0.kv.nks() {
            if !s0.kv.is_element(i) {
                continue;
            }
            if !s1.kv.is_element(i) {
                return Err("isElement does not match".to_string());
            }
            for ii in 0..=s0.kv.order() {
                let ii0 = if s0.okv >= 0 { i + ii } else { (nx - i as isize - ii as isize) as usize };
                let ii1 = if s1.okv >= 0 { i + ii } else { (nx - i as isize - ii as isize) as usize };
                let a = self.bdr_seg_dof(&s0, ii0);
                let b = self.bdr_seg_dof(&s1, ii1);
                d_to_d[a] = d_to_d[b];
            }
        }
        Ok(())
    }

    /// MFEM `NURBSExtension::ConnectBoundaries3D`: the two boundary
    /// quadrilaterals are paired span by span in both directions, again through
    /// each map's own orientation; `nx`/`ny` come from the first map.
    fn connect_boundaries_3d(
        &self,
        d_to_d: &mut [usize],
        bnd0: usize,
        bnd1: usize,
    ) -> Result<(), String> {
        let m0 = self.bdr_quad_dof_map(bnd0);
        let m1 = self.bdr_quad_dof_map(bnd1);
        let (nx, ny) = (m0.nx, m0.ny);

        for j in 0..m0.kvs[1].nks() {
            if !m0.kvs[1].is_element(j) {
                continue;
            }
            if !m1.kvs[1].is_element(j) {
                return Err("isElement does not match #1".to_string());
            }
            for i in 0..m0.kvs[0].nks() {
                if !m0.kvs[0].is_element(i) {
                    continue;
                }
                if !m1.kvs[0].is_element(i) {
                    return Err("isElement does not match #0".to_string());
                }
                for jj in 0..=m0.kvs[1].order() {
                    let jj0 =
                        if m0.okv[1] >= 0 { j + jj } else { (ny - j as isize - jj as isize) as usize };
                    let jj1 =
                        if m1.okv[1] >= 0 { j + jj } else { (ny - j as isize - jj as isize) as usize };
                    for ii in 0..=m0.kvs[0].order() {
                        let ii0 = if m0.okv[0] >= 0 {
                            i + ii
                        } else {
                            (nx - i as isize - ii as isize) as usize
                        };
                        let ii1 = if m1.okv[0] >= 0 {
                            i + ii
                        } else {
                            (nx - i as isize - ii as isize) as usize
                        };
                        let a = m0.dof(ii0, jj0);
                        let b = m1.dof(ii1, jj1);
                        d_to_d[a] = d_to_d[b];
                    }
                }
            }
        }
        Ok(())
    }

    /// MFEM `NURBSExtension::GetNTotalDof`.
    pub fn n_total_dofs(&self) -> usize {
        self.n_total_dofs
    }

    /// MFEM `NURBSExtension::DofMap` — the *un-compacted* DOF index that
    /// `NURBSPatchMap::operator()` returns, mapped onto the merged numbering of
    /// [`Self::connect_boundaries`].  While no boundary has been connected this
    /// is the identity (MFEM: `d_to_d.Size() == 0`), so every caller that walks
    /// `NURBSPatchMap::operator()` in MFEM — the element DOF table, the boundary
    /// DOF tables and `GetEssentialTrueDofs` — has to route through here.
    ///
    /// `dof` must be the *raw* `NURBSPatchMap::operator()` value (as consumed
    /// by the element/boundary DOF tables and [`Self::patch_dof`], which apply
    /// `dof_map` and the `activeDof` compaction internally); an index of the
    /// *compacted* table (e.g. an entry of [`Self::element_dof_table`]) is
    /// already in the final numbering and must not be mapped again.
    pub fn dof_map(&self, dof: usize) -> usize {
        if self.d_to_d.is_empty() {
            dof
        } else {
            self.d_to_d[dof]
        }
    }

    /// MFEM `NURBSExtension::GetNDof` — the number of finite element unknowns.
    pub fn n_dofs(&self) -> usize {
        self.n_dofs
    }

    /// MFEM `NURBSExtension::GetNV` — active vertices.
    pub fn n_vertices(&self) -> usize {
        self.n_vertices
    }

    /// MFEM `NURBSExtension::GetGNV` — the mesh-offset count of
    /// `GenerateOffsets` (real vertices plus the interior mesh offsets of
    /// edges, faces and patches).
    pub fn n_global_vertices(&self) -> usize {
        self.n_global_vertices
    }

    /// MFEM `NURBSExtension::GetWeights`.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// MFEM `NURBSExtension::GetElementDofTable` row access — the DOFs of
    /// element `e` in the element's own (tensor) order.
    pub fn element_dofs(&self, e: usize) -> &[usize] {
        &self.el_dof[e]
    }

    /// The whole element DOF table (MFEM `el_dof`), `n_elements` rows.
    pub fn element_dof_table(&self) -> &[Vec<usize>] {
        &self.el_dof
    }

    /// MFEM `NURBSExtension::GetElementPatch`.
    pub fn element_patch(&self, e: usize) -> usize {
        self.el_to_patch[e]
    }

    /// MFEM `NURBSExtension::GetElementIJK` — the knot-span indices of element
    /// `e` in its patch (MFEM's `el_to_IJK`).
    pub fn element_ijk(&self, e: usize) -> [usize; 3] {
        self.el_to_ijk[e]
    }

    /// The mesh-boundary-element vertex list of boundary element `b` (MFEM
    /// `Mesh::GetBdrElementVertices`).
    pub fn boundary_vertices(&self, b: usize) -> &[usize] {
        &self.boundary[b].verts
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod flip_tests {
    use super::flip_knot_vector;

    #[test]
    fn mirrors_interior_knots_like_mfem() {
        // Hand-computed from `KnotVector::Flip`: apb = k(0)+k(size-1) = 1,
        // ns = (NCP-Order)/2, k(Order+i) <-> apb - k(NCP-i).  For the
        // asymmetric interior knot 0.2 the flip mirrors it to 0.8.
        let kv = [0.0, 0.0, 0.2, 1.0, 1.0];
        assert_eq!(flip_knot_vector(&kv, 1), vec![0.0, 0.0, 0.8, 1.0, 1.0]);
        // A symmetric knot vector is its own flip.
        let sym = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0];
        assert_eq!(flip_knot_vector(&sym, 1), sym.to_vec());
        // Clamped quadratic with no interior knots is unchanged.
        let clamp = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        assert_eq!(flip_knot_vector(&clamp, 2), clamp.to_vec());
    }
}
