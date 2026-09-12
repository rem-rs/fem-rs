//! MFEM `NURBSExtension` — NURBS patch topology and global DOF numbering.
//!
//! 1:1 port of the parts of MFEM's `NURBSExtension` (C++ `mesh/nurbs.{hpp,cpp}`)
//! that determine the NURBS finite element space: patch topology bookkeeping,
//! unique/comprehensive knot vectors, `GenerateOffsets`, and
//! `GenerateElementDofTable` with its per-element DOF table.
//!
//! # What is reproduced
//!
//! * `Mesh::LoadPatchTopo` + `NURBSExtension::Load` for the `knotvectors` file
//!   variant (the one every `data/*-nurbs.mesh` mesh uses except
//!   `square-disc-nurbs-patch.mesh`): `dimension`, `elements`, `boundary`,
//!   `edges`, `vertices`, `knotvectors`, `weights`.
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
//! false), and the `patches` mesh-file variant, boundary-element DOF tables,
//! B-net/patch conversion, refinement and `Print`/`PrintSolution` are out of
//! scope.  See the module tests for the coverage boundary.

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

fn as_usize(v: f64, what: &str) -> Result<usize, String> {
    if v < 0.0 || v.fract() != 0.0 {
        return Err(format!("{what}: expected a non-negative integer, got {v}"));
    }
    Ok(v as usize)
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
    /// For every entry of `boundary`, the patch-boundary entity it lies on:
    /// `(patch, direction, low)`, where `low` marks the minimum-parameter side.
    /// Filled by [`Self::generate_boundary_elements`] (dimensions 2 and 3).
    bdr_sides: Vec<(usize, usize, bool)>,
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

impl NurbsExtension {
    /// Read a NURBS mesh file (`NURBSExtension(std::istream&)` with the
    /// `knotvectors` variant).
    pub fn from_mesh_str(text: &str) -> Result<Self, String> {
        let sections = tokenize_mesh(text)?;

        if has_section(&sections, "patches") {
            return Err(
                "NurbsExtension: the 'patches' mesh-file variant (NURBSPatch data) is not \
                 supported yet; use a 'knotvectors' mesh"
                    .to_string(),
            );
        }

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
            // vector sign when the file writes it the other way round.
            let kv = if v0 > v1 {
                std::mem::swap(&mut v0, &mut v1);
                -kv
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

        // ── unique knot vectors (`NURBSExtension::Load`) ──────────────────────
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
        };

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

        ext.build_patch_topology()?;
        if ext.boundary.is_empty() {
            ext.generate_boundary_elements();
        }
        ext.rebuild()?;
        ext.weights = ext.unit_weights();

        // ── weights ───────────────────────────────────────────────────────────
        // `NURBSExtension::Load` does `weights.Load(input, GetNDof())`: the
        // section carries exactly `GetNDof()` values and no count.
        if let Ok(w) = section(&sections, "weights") {
            if w.len() != ext.n_dofs {
                return Err(format!(
                    "weights: expected {} values (GetNDof), found {}",
                    ext.n_dofs,
                    w.len()
                ));
            }
            ext.weights = w.to_vec();
        } else {
            // `unitweights` / `autoweights`.
            ext.weights = vec![1.0; ext.n_dofs];
        }

        Ok(ext)
    }

    /// Read a NURBS mesh file from disk.
    pub fn from_mesh_file(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let text = std::fs::read_to_string(path.as_ref())
            .map_err(|e| format!("NurbsExtension::from_mesh_file: {e}"))?;
        Self::from_mesh_str(&text)
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
    /// spaced knots into every non-empty span of every unique knot vector.
    ///
    /// The mesh's *control points* are re-derived by MFEM's
    /// `NURBSPatch::UniformRefinement`; knot insertion leaves the geometry
    /// invariant, so [`crate::NurbsFESpace`] keeps the original control net and
    /// evaluates it over the refined parameter intervals instead.
    pub fn uniform_refinement(&mut self, rf: usize) -> Result<(), String> {
        if rf < 2 {
            return Err(format!(
                "NurbsExtension::uniform_refinement: refinement factor must be >= 2, got {rf}"
            ));
        }
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
        self.rebuild()
    }

    /// MFEM `NURBSPatchMap::operator()(i, j, k)` with `MapMode::Dof` — the
    /// global (compacted) DOF index of the patch multi-index `multi` in the
    /// `(x, y, z)` direction order, `0 <= multi[d] < NCP[d]`.
    pub fn patch_dof(&self, patch: usize, multi: &[usize]) -> Result<usize, String> {
        if multi.len() != self.dim {
            return Err(format!(
                "NurbsExtension::patch_dof: expected {} indices, got {}",
                self.dim,
                multi.len()
            ));
        }
        self.patch_map_mode(patch, multi, MapMode::Dof)
    }

    /// The control-point coordinates of a NURBS mesh file: the
    /// `FiniteElementSpace` / `VDim: <n>` / `Ordering: 1` block that MFEM reads
    /// into the mesh's `Nodes` grid function (the B-spline control net).
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
            // Header lines (`FiniteElementCollection: …`, `Ordering: …`) carry
            // the only non-numeric columns; the coordinate rows are plain
            // numbers.
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
        let coords = values.chunks(vdim).map(|c| c.to_vec()).collect();
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

    /// MFEM `NURBSExtension::KnotInd(edge)`.
    pub fn knot_ind(&self, edge: usize) -> usize {
        self.edge_to_ukv[edge].unsigned_abs() as usize
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
                    self.boundary.push(TopoElement {
                        attr: el.attr,
                        geom: GEOM_POINT,
                        verts: vec![el.verts[0]],
                    });
                    self.boundary.push(TopoElement {
                        attr: el.attr,
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
                            self.boundary.push(TopoElement {
                                attr: el.attr,
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
                                attr: el.attr,
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
                            let g = self.patch_map_mode(p, &multi, MapMode::Dof)?;
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
    /// `(patch, direction, low)` with `low` marking the minimum-parameter side.
    ///
    /// This is the `[direction, side]` an element's `NURBSFiniteElement` carries
    /// in MFEM (`NURBSPatchMap::SetBdrPatchVertexMap`'s orientation): a boundary
    /// element of patch `p` on the `low`/`high` side of direction `d` spans the
    /// control points `multi[d] == 0` / `multi[d] == NCP_d - 1`, which is exactly
    /// the information `GetEssentialTrueDofs` needs.  Empty for 1-D meshes (a
    /// 1-D boundary element is a point and carries no side).
    pub fn boundary_sides(&self) -> &[(usize, usize, bool)] {
        &self.bdr_sides
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
        if d == 2 {
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
                    self.bdr_sides.push((p, dir, low));
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
                    self.bdr_sides.push((p, dir, low));
                }
            }
        }
    }

    /// MFEM `NURBSExtension::GetNTotalDof`.
    pub fn n_total_dofs(&self) -> usize {
        self.n_total_dofs
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
}

#[cfg(test)]
mod tests;
