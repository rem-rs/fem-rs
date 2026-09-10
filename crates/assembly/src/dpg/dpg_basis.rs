//! DPG element/face basis evaluation engine.
//!
//! Ported from the element machinery implicit in MFEM's
//! `miniapps/dpg/util/weakform.cpp` (`DPGWeakForm::Assemble`): per element the
//! weak form needs the *element-local* basis of each broken test space
//! (MFEM `FiniteElementCollection::GetFE(geom, order)`) and of each trial
//! space, plus the face basis of each trace (skeleton) space.
//!
//! This module provides that machinery over fem-rs types:
//!
//! * [`VolKind`] — element-local space kind: scalar Lagrange (L2/H1-broken),
//!   vector Lagrange (L2 with vdim, MFEM `byNODES` expansion), H(div) (RT) and
//!   H(curl) (ND).
//! * [`VolVals`] — physical-space values of one basis at one quadrature point
//!   (`phi`, `grad`, `div`, `curl`) including the Piola transforms
//!   (`contravariant` for H(div), `covariant` + `J/detJ` curl for H(curl)) and
//!   the DOF orientation signs (`FiniteElementSpace::element_signs`).
//! * [`SkeletonSpace`] — the trace (mesh skeleton) space with Lagrange face
//!   bases; serves both `H1_Trace` (nodal) and `RT_Trace` (orientation via the
//!   ±1 `scale` in the trace integrators, cf. `TraceIntegrator` /
//!   `NormalTraceIntegrator` in MFEM `fem/bilininteg.cpp`).
//!
//! Reference-element selection mirrors MFEM's families:
//! Tri/Tet: `TriPk`/`TetPk`; Quad/Hex L2: Gauss-Legendre nodal (`QuadL2GL`…);
//! RT: `TriRTk`/`QuadRTk`/`TetRTk`/`HexRTk`; ND: `TriNDk`/`QuadNDk`/`TetNDk`/
//! `HexNDk`.

use fem_element::{
    ReferenceElement, VectorReferenceElement,
    lagrange::{factory::{TetPk, TriPk}, HexL2GL, QuadL2GL},
    quadrature::{gauss_legendre_01, quad_rule_01, tet_rule, tri_rule},
    raviart_thomas::{HexRTk, QuadRTk, TetRTk, TriRTk},
    nedelec::{HexNDk, QuadNDk, TetNDk, TriNDk},
};
use fem_mesh::{element_type::ElementType, topology::MeshTopology};

// ─── Volume (element-local) space kinds ──────────────────────────────────────

/// Element-local (broken) space kind for DPG test spaces and volume trial spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolKind {
    /// Scalar Lagrange element (MFEM L2/H1 element-local basis).
    Scalar,
    /// Scalar Lagrange element expanded `vdim` times, `byNODES` layout
    /// (MFEM `FiniteElementSpace(mesh, fec, vdim)` default ordering):
    /// DOFs are `[comp0 dofs, comp1 dofs, ...]`.
    Vector {
        /// Number of vector components.
        vdim: usize,
    },
    /// H(div) element (MFEM RT_FECollection), contravariant Piola transform.
    HDiv,
    /// H(curl) element (MFEM ND_FECollection), covariant Piola transform.
    HCurl,
}

impl VolKind {
    /// Number of DOFs per element for geometry `et` and order `order`.
    pub fn n_dofs_per_elem(&self, et: ElementType, order: u8) -> usize {
        let n = scalar_ref_elem(et, order).n_dofs();
        match self {
            VolKind::Scalar => n,
            VolKind::Vector { vdim } => n * vdim,
            VolKind::HDiv => vector_ref_elem(et, order).n_dofs(),
            VolKind::HCurl => hcurl_ref_elem(et, order).n_dofs(),
        }
    }
}

/// Constant P0 reference element (order-0 broken spaces).
struct P0Elem {
    dim: usize,
}

impl ReferenceElement for P0Elem {
    fn n_dofs(&self) -> usize {
        1
    }
    fn eval_basis(&self, _xi: &[f64], values: &mut [f64]) {
        values[0] = 1.0;
    }
    fn eval_grad_basis(&self, _xi: &[f64], grads: &mut [f64]) {
        grads[..self.dim].fill(0.0);
    }
    fn dim(&self) -> u8 {
        self.dim as u8
    }
    fn order(&self) -> u8 {
        0
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![vec![0.5; self.dim]]
    }
    fn quadrature(&self, _order: u8) -> fem_element::QuadratureRule {
        fem_element::QuadratureRule {
            points: vec![vec![0.5; self.dim]],
            weights: vec![1.0],
        }
    }
}

/// Scalar (Lagrange) reference element, matching fem-space `L2Space` bases
/// (Gauss-Legendre nodal on tensor elements, equispaced on simplices).
pub fn scalar_ref_elem(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (_, 0) => Box::new(P0Elem { dim: match et {
            ElementType::Tri3 | ElementType::Tri6 | ElementType::Quad4 => 2,
            _ => 3,
        } }),
        (ElementType::Tri3 | ElementType::Tri6, p) => Box::new(TriPk::new(p as usize)),
        (ElementType::Tet4, p) => Box::new(TetPk::new(p as usize)),
        (ElementType::Quad4, p) => Box::new(QuadL2GL::new(p as usize)),
        (ElementType::Hex8, p) => Box::new(HexL2GL::new(p as usize)),
        _ => panic!("dpg_basis::scalar_ref_elem: unsupported {et:?} order {order}"),
    }
}

/// Vector (RT/ND) reference element of degree `order`.
pub fn vector_ref_elem(et: ElementType, order: u8) -> Box<dyn VectorReferenceElement> {
    let p = order as usize;
    match et {
        ElementType::Tri3 | ElementType::Tri6 => Box::new(TriRTk::new(p)),
        ElementType::Quad4 => Box::new(QuadRTk::new(p)),
        ElementType::Tet4 => Box::new(TetRTk::new(p)),
        ElementType::Hex8 => Box::new(HexRTk::new(p)),
        _ => panic!("dpg_basis::vector_ref_elem: unsupported {et:?}"),
    }
}

/// H(curl) reference element of order `order`.
pub fn hcurl_ref_elem(et: ElementType, order: u8) -> Box<dyn VectorReferenceElement> {
    let p = order as usize;
    match et {
        ElementType::Tri3 | ElementType::Tri6 => Box::new(TriNDk::new(p)),
        ElementType::Quad4 => Box::new(QuadNDk::new(p)),
        ElementType::Tet4 => Box::new(TetNDk::new(p)),
        ElementType::Hex8 => Box::new(HexNDk::new(p)),
        _ => panic!("dpg_basis::hcurl_ref_elem: unsupported {et:?}"),
    }
}

/// Volume quadrature rule for the element geometry (reference domain matched
/// to the fem-element families: `[0,1]^dim`).
pub fn vol_quadrature(et: ElementType, order: u8) -> (Vec<Vec<f64>>, Vec<f64>) {
    match et {
        ElementType::Tri3 | ElementType::Tri6 => {
            let r = tri_rule(order);
            (r.points, r.weights)
        }
        ElementType::Quad4 => {
            let r = quad_rule_01(order);
            (r.points, r.weights)
        }
        ElementType::Tet4 => {
            let r = tet_rule(order);
            (r.points, r.weights)
        }
        ElementType::Hex8 => {
            // Tensor-product Gauss-Legendre on [0,1]^3.
            let (pts1d, wts1d) = gauss_legendre_01(order as usize);
            let mut points = Vec::new();
            let mut weights = Vec::new();
            for (iz, &wz) in wts1d.iter().enumerate() {
                for (iy, &wy) in wts1d.iter().enumerate() {
                    for (ix, &wx) in wts1d.iter().enumerate() {
                        points.push(vec![pts1d[ix], pts1d[iy], pts1d[iz]]);
                        weights.push(wx * wy * wz);
                    }
                }
            }
            (points, weights)
        }
        _ => panic!("dpg_basis::vol_quadrature: unsupported {et:?}"),
    }
}

// ─── Per-quadrature-point values ─────────────────────────────────────────────

/// Physical-space values of one space's basis at one quadrature point.
///
/// For [`VolKind::Vector`] the arrays are `byNODES`-expanded: entry `i` of
/// `phi` corresponds to global component-major DOF `i`
/// (`[comp0: n scalars][comp1: n scalars]...`); `grad` is row-major
/// `[n_expanded × dim]` with the same expansion.
#[derive(Debug, Default, Clone)]
pub struct VolVals {
    /// Basis values, length `n_expanded` (`n_dofs` incl. vdim expansion).
    pub phi: Vec<f64>,
    /// Physical gradients, row-major `[n_expanded × dim]`.
    pub grad: Vec<f64>,
    /// Physical divergence, length `n_scalar` (HDiv only; NOT expanded).
    pub div: Vec<f64>,
    /// Physical curl: length `n_expanded` (2-D vector curl of scalar basis /
    /// scalar curl of ND, when `curl_is_vec2`) or `[n × curl_dim]` (HCurl 3-D).
    pub curl: Vec<f64>,
    /// Expanded DOF count (`n_scalar * vdim` for Vector, else `n_scalar`).
    pub n_expanded: usize,
    /// Number of scalar (per-component) DOFs.
    pub n_scalar: usize,
    /// vdim expansion (1 unless [`VolKind::Vector`]).
    pub vdim: usize,
    /// `true` when `curl` holds a 2-component curl per scalar DOF
    /// (2-D ND scalar curl pre-expansion: length `n_scalar`).
    pub curl_dim: usize,
}

/// Evaluate one element-local space at quadrature point `xi`.
///
/// `jac`/`det_j`/`jit` are the element Jacobian, its determinant and
/// `J^{-T}` at `xi` (dim × dim, row-major via nalgebra).  `signs` are the
/// DOF orientation signs (HDiv/HCurl), pass `None` for scalar spaces.
pub fn eval_vol_space(
    kind: VolKind,
    order: u8,
    et: ElementType,
    dim: usize,
    jac: &nalgebra::DMatrix<f64>,
    det_j: f64,
    jit: &nalgebra::DMatrix<f64>,
    xi: &[f64],
    signs: Option<&[f64]>,
    out: &mut VolVals,
) {
    let (n_scalar, vdim) = match kind {
        VolKind::Scalar => (scalar_ref_elem(et, order).n_dofs(), 1),
        VolKind::Vector { vdim } => (scalar_ref_elem(et, order).n_dofs(), vdim),
        VolKind::HDiv => (vector_ref_elem(et, order).n_dofs(), 1),
        VolKind::HCurl => (hcurl_ref_elem(et, order).n_dofs(), 1),
    };

    out.phi.clear();
    out.grad.clear();
    out.div.clear();
    out.curl.clear();
    out.n_scalar = n_scalar;
    out.vdim = vdim;
    out.n_expanded = n_scalar * vdim;
    out.curl_dim = match kind {
        VolKind::HCurl if dim == 3 => 3,
        VolKind::HCurl => 1,
        _ => dim, // scalar-basis vector curl in 2-D: dim components
    };

    match kind {
        VolKind::Scalar | VolKind::Vector { .. } => {
            let fe = scalar_ref_elem(et, order);
            let mut phi = vec![0.0_f64; n_scalar];
            let mut dphi = vec![0.0_f64; n_scalar * dim];
            fe.eval_basis(xi, &mut phi);
            fe.eval_grad_basis(xi, &mut dphi);
            // covariant transform: grad_phys = J^{-T} grad_ref
            let mut gphys = vec![0.0_f64; n_scalar * dim];
            for i in 0..n_scalar {
                for r in 0..dim {
                    let mut s = 0.0;
                    for c in 0..dim {
                        s += jit[(r, c)] * dphi[i * dim + c];
                    }
                    gphys[i * dim + r] = s;
                }
            }
            // 2-D vector curl of the scalar basis, MFEM
            // `GradToVectorCurl2D`: curl φ = (∂φ/∂y, −∂φ/∂x).  Consumed by
            // `DpgCurl2dPairingIntegrator` (Maxwell ultraweak pairing).
            if dim == 2 {
                out.curl.resize(n_scalar * 2, 0.0);
                for i in 0..n_scalar {
                    out.curl[i * 2] = gphys[i * 2 + 1];
                    out.curl[i * 2 + 1] = -gphys[i * 2];
                }
            }
            // expand byNODES
            for _c in 0..vdim {
                out.phi.extend_from_slice(&phi);
                out.grad.extend_from_slice(&gphys);
            }
        }
        VolKind::HDiv => {
            let fe = vector_ref_elem(et, order);
            let mut vref = vec![0.0_f64; n_scalar * dim];
            let mut dref = vec![0.0_f64; n_scalar];
            fe.eval_basis_vec(xi, &mut vref);
            fe.eval_div(xi, &mut dref);
            let inv_det = 1.0 / det_j;
            out.phi.resize(n_scalar * dim, 0.0);
            out.grad.resize(n_scalar * dim, 0.0); // grad unused for HDiv
            out.div.resize(n_scalar, 0.0);
            for i in 0..n_scalar {
                let s = signs.map(|sg| sg[i]).unwrap_or(1.0);
                for r in 0..dim {
                    let mut v = 0.0;
                    for c in 0..dim {
                        v += jac[(r, c)] * vref[i * dim + c];
                    }
                    out.phi[i * dim + r] = s * v * inv_det;
                }
                out.div[i] = s * dref[i] * inv_det;
            }
            // Mark: phi is component-interleaved for HDiv (each dof is a
            // vector).  n_expanded = n_scalar for layout purposes.
        }
        VolKind::HCurl => {
            let fe = hcurl_ref_elem(et, order);
            let mut vref = vec![0.0_f64; n_scalar * dim];
            let mut cref = vec![0.0_f64; n_scalar * if dim == 3 { 3 } else { 1 }];
            fe.eval_basis_vec(xi, &mut vref);
            fe.eval_curl(xi, &mut cref);
            let inv_det = 1.0 / det_j;
            out.phi.resize(n_scalar * dim, 0.0);
            out.grad.resize(n_scalar * dim, 0.0); // grad unused for HCurl
            out.curl.resize(n_scalar * out.curl_dim, 0.0);
            for i in 0..n_scalar {
                let s = signs.map(|sg| sg[i]).unwrap_or(1.0);
                for r in 0..dim {
                    let mut v = 0.0;
                    for c in 0..dim {
                        v += jit[(r, c)] * vref[i * dim + c];
                    }
                    out.phi[i * dim + r] = s * v;
                }
                if dim == 3 {
                    for c in 0..3 {
                        let mut v = 0.0;
                        for r in 0..3 {
                            v += jac[(c, r)] * cref[i * 3 + r];
                        }
                        out.curl[i * 3 + c] = s * v * inv_det;
                    }
                } else {
                    out.curl[i] = s * cref[i] * inv_det;
                }
            }
        }
    }
}

// ─── Face quadrature ─────────────────────────────────────────────────────────

/// Quadrature points/weights on the reference face (parameter domain).
///
/// 2-D edges: `[0,1]` segment; 3-D triangular faces: `[0,1]²` reference
/// triangle (`(0,0),(1,0),(0,1)`); 3-D quadrilateral faces: `[0,1]²`.
pub fn face_quadrature(dim: usize, is_quad_face: bool, order: u8) -> (Vec<Vec<f64>>, Vec<f64>) {
    if dim == 2 {
        let (pts, wts) = gauss_legendre_01(order as usize);
        (pts.iter().map(|&p| vec![p]).collect(), wts)
    } else if is_quad_face {
        let r = quad_rule_01(order);
        (r.points, r.weights)
    } else {
        let r = tri_rule(order);
        (r.points, r.weights)
    }
}

// ─── Skeleton (trace) space ──────────────────────────────────────────────────

/// Adjacency of one skeleton face.
#[derive(Debug, Clone)]
pub enum SkeletonFaceInfo {
    /// Boundary face (single adjacent element).
    Boundary {
        /// Adjacent element.
        elem: u32,
        /// Local face index within the element.
        local_face: usize,
    },
    /// Interior face; `elem_first` is the element that first saw the face
    /// (defines the reference face-node order / normal direction).
    Interior {
        /// Element that first encountered the face ("Elem1").
        elem_first: u32,
        /// The other element ("Elem2").
        elem_second: u32,
        /// Local face index within `elem_first`.
        local_first: usize,
        /// Local face index within `elem_second`.
        local_second: usize,
    },
}

/// Mesh-skeleton (trace) space with Lagrange face bases.
///
/// Port of MFEM's trace FE spaces (`H1_Trace_FECollection`, and the scalar
/// `RT_Trace_FECollection` whose face trace is the full `P_k` space on the
/// face — same Lagrange DOFs; the RT orientation sign is applied by the trace
/// integrators via the ±1 `scale`, cf. `TraceIntegrator` in MFEM
/// `fem/bilininteg.cpp`).
///
/// DOF layout:
/// * 2-D edges: `order + 1` DOFs per edge.
/// * 3-D triangular faces: `(order+1)(order+2)/2`; quadrilateral faces:
///   `(order+1)²`.
/// * Faces are numbered in first-seen order while walking the elements
///   (matching `DpgTraceSpace` and MFEM's edge-table walk); a face's DOFs are
///   consecutive, ordered along the face's **canonical direction**: in 2-D
///   this is MFEM's edge-table direction `(min vertex, max vertex)` (MFEM
///   `DSTable::Push` canonicalises every edge that way, so the global edge
///   direction is not always the first-seen element's local direction); in
///   3-D it is the generating (first-seen) element's local face vertex order
///   (MFEM `FaceInfo`: Elem1 generates the face).  Trace bases and
///   `face_geo_at` normals follow this canonical direction for BOTH adjacent
///   elements, exactly like MFEM's `GetFaceElement` /
///   `GetFaceElementTransformations`; the element side enters only through
///   the ±1 scale (`rt_trace_face_sign(elem_face_orientation(..))`).
/// * `element_dofs(elem)` concatenates the DOFs of all faces of the element in
///   the element's local face order.
pub struct SkeletonSpace<M: MeshTopology> {
    mesh: M,
    order: u8,
    dim: usize,
    face_node_ids: Vec<Vec<u32>>,
    face_info: Vec<SkeletonFaceInfo>,
    elem_local_faces: Vec<Vec<usize>>, // elem -> [global face ids in local order]
    face_dof_offsets: Vec<usize>,
    elem_dofs: Vec<Vec<usize>>,
    n_dofs: usize,
    is_quad_face: Vec<bool>,
    /// Per-face dof ids in `eval_face_lagrange` node order.  For the
    /// discontinuous mode this is the contiguous offset range; for the
    /// continuous H1-trace mode the corner dofs are the shared vertex dofs.
    face_dof_lists: Vec<Vec<usize>>,
    /// `true` for H1-trace (vertex-continuous) spaces: the two endpoint dofs
    /// of each face are shared with the adjacent faces through the mesh
    /// vertex they sit on (MFEM `H1_Trace_FECollection` semantics); `false`
    /// for RT-trace-style spaces (all face dofs face-local).
    continuous: bool,
}

impl<M: MeshTopology + Clone> SkeletonSpace<M> {
    /// Build the (face-discontinuous) skeleton space of face order `order`.
    pub fn new(mesh: M, order: u8) -> Self {
        Self::build(mesh, order, false)
    }

    /// Build the vertex-continuous H1-trace skeleton space (MFEM
    /// `H1_Trace_FECollection`): endpoint dofs of each face are shared
    /// through their mesh vertex, so the trace is continuous across the
    /// skeleton.  The face dof order still matches [`eval_face_lagrange`]
    /// (node `k` at parameter `k/p` along the face's canonical direction).
    pub fn new_h1(mesh: M, order: u8) -> Self {
        Self::build(mesh, order, true)
    }

    fn build(mesh: M, order: u8, continuous: bool) -> Self {
        let dim = mesh.dim() as usize;
        // Enumerate faces in first-seen order.
        let mut face_map: std::collections::HashMap<Vec<u32>, usize> =
            std::collections::HashMap::new();
        let mut face_node_ids: Vec<Vec<u32>> = Vec::new();
        let mut face_info: Vec<SkeletonFaceInfo> = Vec::new();
        let mut elem_local_faces: Vec<Vec<usize>> = vec![Vec::new(); mesh.n_elements()];
        let mut is_quad_face: Vec<bool> = Vec::new();

        for e in mesh.elem_iter() {
            let en = mesh.element_nodes(e);
            let lfs = local_face_table(en, dim);
            for (li, lf) in lfs.iter().enumerate() {
                let unsorted: Vec<u32> = lf.iter().map(|&k| en[k]).collect();
                // Canonical face direction.  2-D: MFEM stores every edge
                // directed (min vertex, max vertex) — the edge table is a
                // `DSTable` (`Mesh::GetElementToEdgeTable`,
                // `DSTable::Push(a,b) := Push_(min, max)`) — so the global
                // edge direction is NOT always the first-seen element's
                // local direction.  3-D: faces keep the generating
                // (first-seen) element's local face vertex order (MFEM
                // `FaceInfo`: Elem1 generates the face).
                let mut key = unsorted.clone();
                key.sort_unstable();
                // Canonical face direction.  2-D: the (min vertex, max
                // vertex) direction — MFEM's global edge orientation
                // (`DSTable::Push(a,b) := Push_(min,max)`,
                // `GetEdgeVertices`: "the two vertices are sorted ... and
                // consistent with the global edge orientation").  3-D: the
                // generating (first-seen) element's local face vertex order
                // (MFEM `FaceInfo`: Elem1 generates the face).  Elements
                // whose local face cycle runs opposite contribute through
                // the orientation sign (`elem_face_orientation` →
                // `rt_trace_face_sign`) and the reversed element-side
                // parametrisation in the trace assembly.
                let mut canonical = unsorted.clone();
                if dim == 2 && canonical[0] > canonical[1] {
                    canonical.swap(0, 1);
                }
                match face_map.get(&key) {
                    Some(&fid) => {
                        if let SkeletonFaceInfo::Boundary { elem: fe, local_face: fl } =
                            face_info[fid]
                        {
                            face_info[fid] = SkeletonFaceInfo::Interior {
                                elem_first: fe,
                                elem_second: e,
                                local_first: fl,
                                local_second: li,
                            };
                        }
                        elem_local_faces[e as usize].push(fid);
                    }
                    None => {
                        let fid = face_info.len();
                        face_map.insert(key, fid);
                        face_node_ids.push(canonical);
                        face_info.push(SkeletonFaceInfo::Boundary { elem: e, local_face: li });
                        is_quad_face.push(lf.len() == 4);
                        elem_local_faces[e as usize].push(fid);
                    }
                }
            }
        }

        // DOF counts per face
        let dofs_per_face = |is_quad: bool| -> usize {
            let p = order as usize;
            if dim == 2 {
                p + 1
            } else if is_quad {
                (p + 1) * (p + 1)
            } else {
                (p + 1) * (p + 2) / 2
            }
        };

        let (face_dof_offsets, n_dofs, elem_dofs, face_dof_lists) = if !continuous {
            let mut face_dof_offsets = Vec::with_capacity(face_info.len() + 1);
            face_dof_offsets.push(0);
            for f in 0..face_info.len() {
                let n = dofs_per_face(is_quad_face[f]);
                face_dof_offsets.push(face_dof_offsets[f] + n);
            }
            let n_dofs = face_dof_offsets[face_info.len()];
            let mut elem_dofs = vec![Vec::new(); mesh.n_elements()];
            for e in mesh.elem_iter() {
                let ei = e as usize;
                for &fid in &elem_local_faces[ei] {
                    elem_dofs[ei].extend(face_dof_offsets[fid]..face_dof_offsets[fid + 1]);
                }
            }
            let face_dof_lists: Vec<Vec<usize>> = (0..face_info.len())
                .map(|f| (face_dof_offsets[f]..face_dof_offsets[f + 1]).collect())
                .collect();
            (face_dof_offsets, n_dofs, elem_dofs, face_dof_lists)
        } else {
            // Vertex-continuous H1 trace.  DOF layout (MFEM
            // `FiniteElementSpace::GetFaceDofs` with an H1 fec): corner dofs
            // are the MESH VERTEX ids themselves (one dof per vertex, shared
            // by all touching faces), interior face dofs follow after the
            // vertex-dof count.  The face dof list is ordered to match
            // `eval_face_lagrange`: node k sits at parameter k/p along the
            // face's canonical direction, so the first/last nodes are the
            // face's first/last corner vertex and the interior nodes follow.
            let p = order as usize;
            let n_vdofs = mesh.n_nodes();
            let vertex_dof = |v: u32| v as usize;
            let interior_per_face = if dim == 2 {
                (p + 1).saturating_sub(2)
            } else if is_quad_face.iter().any(|&q| q) {
                // quad faces: interior dofs per 1-D direction = p − 1,
                // total (p−1)² per face (p ≥ 1)
                let pi = p.saturating_sub(1);
                pi * pi
            } else {
                // tri faces: interior dofs = (p−1)(p−2)/2
                let pi = p.saturating_sub(1);
                pi.max(0) * pi.saturating_sub(2).max(0) / 2
            };
            let mut face_dof_offsets = Vec::with_capacity(face_info.len() + 1);
            face_dof_offsets.push(0);
            for f in 0..face_info.len() {
                let n = dofs_per_face(is_quad_face[f]);
                face_dof_offsets.push(face_dof_offsets[f] + n);
            }
            // Interior dofs start AFTER the vertex dofs (MFEM
            // `H1_Trace_FECollection`: face dofs live on mesh vertices and
            // face-interior entities, so the global count is
            // n_vertices_touched + n_faces × interior_per_face — NOT the
            // per-face dof-count sum, which would leave phantom dofs).
            let base = n_vdofs;
            // Per-face dofs: corner vertex dofs + shared interior range.
            let mut face_dofs: Vec<Vec<usize>> = Vec::with_capacity(face_node_ids.len());
            if dim == 2 {
                for f in 0..face_node_ids.len() {
                    let mut dofs = Vec::with_capacity(p + 1);
                    dofs.push(vertex_dof(face_node_ids[f][0]));
                    for k in 1..p {
                        dofs.push(base + f * interior_per_face + (k - 1));
                    }
                    if p >= 1 {
                        dofs.push(vertex_dof(face_node_ids[f][1]));
                    }
                    face_dofs.push(dofs);
                }
            } else {
                // 3-D: only the per-face corner/interior mapping is needed by
                // the current miniapps; order-1 H1 traces are vertex dofs.
                for f in 0..face_node_ids.len() {
                    let mut dofs = Vec::with_capacity(dofs_per_face(is_quad_face[f]));
                    for &v in &face_node_ids[f] {
                        dofs.push(vertex_dof(v));
                    }
                    let b2 = base + f * interior_per_face;
                    for k in 0..interior_per_face {
                        dofs.push(b2 + k);
                    }
                    face_dofs.push(dofs);
                }
            }
            let n_dofs = base + face_node_ids.len() * interior_per_face;
            let mut elem_dofs = vec![Vec::new(); mesh.n_elements()];
            for e in mesh.elem_iter() {
                let ei = e as usize;
                for &fid in &elem_local_faces[ei] {
                    elem_dofs[ei].extend_from_slice(&face_dofs[fid]);
                }
            }
            // `face_dof_offsets` carries the per-face dof-count prefix sums
            // (used only by the discontinuous `face_dofs()` accessor); in the
            // continuous mode it does not describe global dof ids.
            (face_dof_offsets, n_dofs, elem_dofs, face_dofs)
        };

        SkeletonSpace {
            mesh,
            order,
            dim,
            face_node_ids,
            face_info,
            elem_local_faces,
            face_dof_offsets,
            elem_dofs,
            n_dofs,
            is_quad_face,
            face_dof_lists,
            continuous,
        }
    }

    /// Face order (polynomial degree on each face).
    pub fn order(&self) -> u8 {
        self.order
    }

    /// Total number of skeleton DOFs.
    pub fn n_dofs(&self) -> usize {
        self.n_dofs
    }

    /// Number of faces (boundary + interior).
    pub fn n_faces(&self) -> usize {
        self.face_info.len()
    }

    /// Number of boundary faces (indices `0..n_boundary_faces` are not
    /// guaranteed contiguous — use [`Self::face_dofs`] with the face
    /// adjacency to identify boundary faces).
    pub fn is_boundary_face(&self, f: usize) -> bool {
        matches!(self.face_info[f], SkeletonFaceInfo::Boundary { .. })
    }

    /// Global DOFs of face `f` (consecutive range).  Only meaningful for the
    /// discontinuous (RT-trace style) spaces — use [`Self::face_dof_list`]
    /// for the continuous H1-trace mode.
    pub fn face_dofs(&self, f: usize) -> std::ops::Range<usize> {
        self.face_dof_offsets[f]..self.face_dof_offsets[f + 1]
    }

    /// Global DOF ids of face `f` in `eval_face_lagrange` node order
    /// (valid in both modes; for the continuous H1-trace mode the corner
    /// dofs are the shared skeleton-vertex dofs).
    pub fn face_dof_list(&self, f: usize) -> &[usize] {
        &self.face_dof_lists[f]
    }

    /// Face adjacency info.
    pub fn face_info(&self, f: usize) -> &SkeletonFaceInfo {
        &self.face_info[f]
    }

    /// Physical coordinates of the face nodes in first-seen direction order.
    pub fn face_nodes(&self, f: usize) -> &Vec<u32> {
        &self.face_node_ids[f]
    }

    /// Whether face `f` is a quadrilateral (3-D only).
    pub fn is_quad_face(&self, f: usize) -> bool {
        self.is_quad_face[f]
    }

    /// DOFs per face of `f`.
    pub fn dofs_per_face(&self, f: usize) -> usize {
        self.face_dofs(f).len()
    }

    /// Element DOF list: concatenation of the element's faces' DOFs in local
    /// face order (the trace block's element vdofs).
    pub fn element_dofs(&self, elem: u32) -> &[usize] {
        &self.elem_dofs[elem as usize]
    }

    /// Global face id of the element's `local_index`-th local face.
    pub fn elem_face_id(&self, elem: u32, local_index: usize) -> usize {
        self.elem_local_faces[elem as usize][local_index]
    }

    /// Orientation parity of the element's `local_index`-th local face
    /// against the canonical (global face storage) direction.
    ///
    /// MFEM stores every interior face directed along the local face of the
    /// element that generated it (`Mesh::FaceInfo`: "Elem1No always refers
    /// to the element that generated the face"), which is exactly this
    /// skeleton's canonical direction.  Returns `+1` when the element's
    /// local face cycle matches the canonical directed cycle (the element is
    /// MFEM's `Elem1`) and `−1` when it is the reverse cycle (`Elem2` — the
    /// element's outward normal is opposite to the canonical face normal).
    /// This is the input to `fem_space::dof_transformation::
    /// rt_trace_face_sign` (the RT/trace orientation sign applied by MFEM's
    /// trace integrators) and fixes the element-side reference
    /// parametrisation for trace quadrature.
    ///
    /// Panics for non-manifold faces (node cycles that match neither
    /// direction).
    pub fn elem_face_orientation(&self, elem: u32, local_index: usize) -> i32 {
        let fid = self.elem_local_faces[elem as usize][local_index];
        let canonical = &self.face_node_ids[fid];
        let en = self.mesh.element_nodes(elem);
        let lfs = local_face_table(en, self.dim);
        let local: Vec<u32> = lfs[local_index].iter().map(|&k| en[k]).collect();
        let n = canonical.len();
        debug_assert_eq!(n, local.len());
        if n == 2 {
            // A 2-cycle's "cyclic rotation" is its reversal: compare exactly.
            return if canonical[0] == local[0] { 1 } else { -1 };
        }
        let start = canonical
            .iter()
            .position(|&x| x == local[0])
            .unwrap_or_else(|| panic!("elem_face_orientation: face node mismatch"));
        if (0..n).all(|k| canonical[(start + k) % n] == local[k]) {
            return 1;
        }
        if (0..n).all(|k| canonical[(start + n - k % n) % n] == local[k]) {
            return -1;
        }
        panic!(
            "elem_face_orientation: non-manifold face {fid} (element {elem}, local {local_index})"
        );
    }

    /// Mesh reference.
    pub fn mesh(&self) -> &M {
        &self.mesh
    }

    /// Dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Local face table (node indices into the element node list), matching
/// MFEM's `Geometry::Constants` face-vertex tables for the fem-element
/// reference domains.
pub fn local_face_table(elem_nodes: &[u32], dim: usize) -> Vec<Vec<usize>> {
    match (elem_nodes.len(), dim) {
        (3, 2) => vec![vec![0, 1], vec![1, 2], vec![2, 0]],
        (4, 2) => vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 0]],
        (4, 3) => vec![
            vec![0, 1, 2],
            vec![0, 1, 3],
            vec![0, 2, 3],
            vec![1, 2, 3],
        ],
        (8, 3) => vec![
            vec![0, 1, 2, 3],
            vec![4, 5, 6, 7],
            vec![0, 1, 5, 4],
            vec![1, 2, 6, 5],
            vec![2, 3, 7, 6],
            vec![3, 0, 4, 7],
        ],
        _ => panic!(
            "local_face_table: unsupported (npe={}, dim={})",
            elem_nodes.len(),
            dim
        ),
    }
}

// ─── Face parametrization ────────────────────────────────────────────────────

/// Reference-domain node coordinates of the element geometries.
pub fn ref_node_coords(et: ElementType) -> Vec<[f64; 3]> {
    match et {
        ElementType::Tri3 | ElementType::Tri6 => vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ElementType::Quad4 => vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        ElementType::Tet4 => vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        ElementType::Hex8 => vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        _ => panic!("ref_node_coords: unsupported {et:?}"),
    }
}

/// Map a reference-face parameter to element reference coordinates.
///
/// * 2-D edge (`s ∈ [0,1]`): straight-edge interpolation of the two local
///   face nodes.
/// * 3-D triangular face (`(s,t)` barycentric on the ref triangle):
///   affine interpolation of the three local face nodes.
/// * 3-D quad face: bilinear interpolation of the four local face nodes.
pub fn face_param_to_elem_ref(
    et: ElementType,
    local_face: &[usize],
    is_quad: bool,
    param: &[f64],
) -> Vec<f64> {
    let rn = ref_node_coords(et);
    let dim = if rn[0][2] != 0.0 || matches!(et, ElementType::Tet4 | ElementType::Hex8) {
        3
    } else {
        2
    };
    let mut x = vec![0.0_f64; dim];
    if local_face.len() == 2 {
        let (a, b) = (&rn[local_face[0]], &rn[local_face[1]]);
        let s = param[0];
        for d in 0..dim {
            x[d] = a[d] * (1.0 - s) + b[d] * s;
        }
    } else if is_quad {
        let s = param[0];
        let t = param[1];
        let c = [
            &rn[local_face[0]],
            &rn[local_face[1]],
            &rn[local_face[2]],
            &rn[local_face[3]],
        ];
        for d in 0..dim {
            x[d] = c[0][d] * (1.0 - s) * (1.0 - t)
                + c[1][d] * s * (1.0 - t)
                + c[2][d] * s * t
                + c[3][d] * (1.0 - s) * t;
        }
    } else {
        let s = param[0];
        let t = param[1];
        let a = &rn[local_face[0]];
        let b = &rn[local_face[1]];
        let c = &rn[local_face[2]];
        for d in 0..dim {
            x[d] = a[d] * (1.0 - s - t) + b[d] * s + c[d] * t;
        }
    }
    x
}

// ─── Face Lagrange basis ─────────────────────────────────────────────────────

/// Tensor/Lagrange DOF count on a reference face of order `p`.
pub fn face_basis_dofs(dim: usize, is_quad: bool, p: usize) -> usize {
    if dim == 2 {
        p + 1
    } else if is_quad {
        (p + 1) * (p + 1)
    } else {
        (p + 1) * (p + 2) / 2
    }
}

/// Evaluate the Lagrange basis of degree `p` at point `x` in `[0,1]`.
fn lagrange_1d(p: usize, x: f64, out: &mut [f64]) {
    if p == 0 {
        out[0] = 1.0;
        return;
    }
    for k in 0..=p {
        let xk = k as f64 / p as f64;
        let mut v = 1.0;
        for j in 0..=p {
            if j == k {
                continue;
            }
            let xj = j as f64 / p as f64;
            v *= (x - xj) / (xk - xj);
        }
        out[k] = v;
    }
}

/// Evaluate the scalar Lagrange face basis of degree `p` at a reference-face
/// parameter point.
///
/// 2-D edges: 1-D Lagrange `P_p(s)`; 3-D quad faces: tensor `Q_p(s) Q_p(t)`
/// (row-major over `(t, s)`: index `t*(p+1) + s`); 3-D tri faces: equispaced
/// nodal `P_p` (index `row*(p+1) - row*(row-1)/2 + col`, node
/// `(a,b)` with `a+b ≤ p`, ordered lex by `(a, b)`).
pub fn eval_face_lagrange(dim: usize, is_quad: bool, p: usize, param: &[f64], out: &mut [f64]) {
    if dim == 2 {
        lagrange_1d(p, param[0], out);
    } else if is_quad {
        let mut la = vec![0.0; p + 1];
        let mut lb = vec![0.0; p + 1];
        lagrange_1d(p, param[0], &mut la);
        lagrange_1d(p, param[1], &mut lb);
        let mut k = 0;
        for t in 0..=p {
            for s in 0..=p {
                out[k] = lb[t] * la[s];
                k += 1;
            }
        }
    } else {
        // Equispaced barycentric nodes on the reference triangle, evaluated
        // through the collapsed coordinate x = s/(1−t):
        //   L_{a,b}(s,t) = ℓ^{(p−b)}_a(s/(1−t)) · ℓ^{(p)}_b(t)
        // where ℓ^{(n)}_m is the 1-D Lagrange basis of degree n for the node
        // m/n ∈ [0,1].  This spans exactly P_p and satisfies partition of
        // unity.
        let s = param[0];
        let t = param[1];
        let x = if (1.0 - t).abs() < 1e-30 { s / (1.0 - 1e-30) } else { s / (1.0 - t) };
        let mut k = 0;
        for row in 0..=p {
            for a in 0..=row {
                let b = row - a;
                out[k] = lagrange_1d_at(p - b, a, x) * lagrange_1d_at(p, b, t);
                k += 1;
            }
        }
    }
}

/// 1-D Lagrange basis `ℓ^{(n)}_m(x)`: degree-`n` basis for the node `m/n`.
fn lagrange_1d_at(n: usize, m: usize, x: f64) -> f64 {
    let mut v = 1.0;
    for j in 0..=n {
        if j == m {
            continue;
        }
        let xj = j as f64 / n as f64;
        let xm = m as f64 / n as f64;
        v *= (x - xj) / (xm - xj);
    }
    v
}

/// Face dof index layout for tri faces must match `eval_face_lagrange`.
/// (Node `(a, b)` with `a + b ≤ p` at index `row*(p+1) - row*(row-1)/2 + a`
/// where `row = a + b`.)  This helper returns the index of `(a, b)`.
pub fn tri_face_dof_index(a: usize, b: usize, p: usize) -> usize {
    let row = a + b;
    row * (p + 1) - row * (row - 1) / 2 + a
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    /// Finite-difference consistency: the reference curl reported by the
    /// ND element matches curl of its reference vector basis.
    #[test]
    fn nd_curl_matches_basis_fd() {
        for et in [ElementType::Quad4, ElementType::Tri3] {
            for order in [1u8, 2u8] {
                let fe = hcurl_ref_elem(et, order);
                let n = fe.n_dofs();
                let mut v = vec![0.0_f64; n * 2];
                let mut c = vec![0.0_f64; n];
                let mut vl = vec![0.0_f64; n * 2];
                let mut vr = vec![0.0_f64; n * 2];
                let mut vd = vec![0.0_f64; n * 2];
                let mut vu = vec![0.0_f64; n * 2];
                let h = 1e-6;
                let pts: Vec<Vec<f64>> = match et {
                    ElementType::Quad4 => {
                        vec![vec![0.3, 0.4], vec![0.7, 0.6], vec![0.5, 0.2]]
                    }
                    _ => vec![vec![0.3, 0.3], vec![0.4, 0.2]],
                };
                for xi in &pts {
                    fe.eval_basis_vec(xi, &mut v);
                    fe.eval_curl(xi, &mut c);
                    let mut xl = xi.clone();
                    let mut xr = xi.clone();
                    xl[0] -= h;
                    xr[0] += h;
                    fe.eval_basis_vec(&xl, &mut vl);
                    fe.eval_basis_vec(&xr, &mut vr);
                    let mut yl = xi.clone();
                    let mut yr = xi.clone();
                    yl[1] -= h;
                    yr[1] += h;
                    fe.eval_basis_vec(&yl, &mut vd);
                    fe.eval_basis_vec(&yr, &mut vu);
                    for i in 0..n {
                        // ref curl = dv_y/dx - dv_x/dy
                        let curl_fd = (vr[i * 2 + 1] - vl[i * 2 + 1]) / (2.0 * h)
                            - (vu[i * 2] - vd[i * 2]) / (2.0 * h);
                        assert!(
                            (curl_fd - c[i]).abs() < 1e-5,
                            "curl mismatch {et:?} order {order} dof {i}: {curl_fd} vs {}",
                            c[i]
                        );
                        let _ = v;
                    }
                }
            }
        }
    }

    #[test]
    fn skeleton_2d_counts() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let sk = SkeletonSpace::new(mesh, 1);
        assert_eq!(sk.n_faces(), 16);
        assert_eq!(sk.n_dofs(), 16 * 2);
        for e in 0..sk.mesh().n_elements() as u32 {
            assert_eq!(sk.element_dofs(e).len(), 6);
        }
    }

    #[test]
    fn face_lagrange_partition_of_unity() {
        let mut out = vec![0.0; 6];
        eval_face_lagrange(3, false, 2, &[1.0 / 3.0, 1.0 / 3.0], &mut out);
        let sum: f64 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn face_lagrange_quad_partition_of_unity() {
        let mut out = vec![0.0; 9];
        eval_face_lagrange(3, true, 2, &[0.3, 0.6], &mut out);
        let sum: f64 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn vol_vals_scalar_expansion() {
        let mesh = Mesh::<2>::unit_square_quad(1);
        let e = 0u32;
        let et = mesh.element_type(e);
        let mut vals = VolVals::default();
        let tr = fem_mesh::ElementTransformation::from_simplex_nodes(&mesh, mesh.element_nodes(e));
        // Quads take the isoparametric path in the weak form; here simply
        // check the scalar path on a triangle mesh instead.
        let mesh_tri = Mesh::<2>::unit_square_tri(1);
        let et_tri = mesh_tri.element_type(0);
        let tr_tri =
            fem_mesh::ElementTransformation::from_simplex_nodes(&mesh_tri, mesh_tri.element_nodes(0));
        let jac = tr_tri.jacobian().clone();
        let jit = tr_tri.jacobian_inv_t().clone();
        eval_vol_space(
            VolKind::Scalar,
            2,
            et_tri,
            2,
            &jac,
            tr_tri.det_j(),
            &jit,
            &[0.25, 0.25],
            None,
            &mut vals,
        );
        assert_eq!(vals.n_scalar, 6);
        assert_eq!(vals.n_expanded, 6);
        let _ = et; // quad type checked elsewhere
        let _ = tr;
    }

    #[test]
    fn vector_expansion_by_nodes() {
        let mesh_tri = Mesh::<2>::unit_square_tri(1);
        let et = mesh_tri.element_type(0);
        let tr = fem_mesh::ElementTransformation::from_simplex_nodes(&mesh_tri, mesh_tri.element_nodes(0));
        let jac = tr.jacobian().clone();
        let jit = tr.jacobian_inv_t().clone();
        let mut vals = VolVals::default();
        eval_vol_space(
            VolKind::Vector { vdim: 2 },
            1,
            et,
            2,
            &jac,
            tr.det_j(),
            &jit,
            &[0.3, 0.3],
            None,
            &mut vals,
        );
        assert_eq!(vals.n_scalar, 3);
        assert_eq!(vals.n_expanded, 6);
        // byNODES: phi[i] == phi[i + 3]
        for i in 0..3 {
            assert!((vals.phi[i] - vals.phi[i + 3]).abs() < 1e-15);
        }
    }
}
