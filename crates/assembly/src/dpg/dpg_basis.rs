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
/// to the fem-element families: `[0,1]^dim` for simplices and quads,
/// `[−1,1]³` for hexahedra — see [`ref_node_coords`]).
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
            // Tensor-product Gauss-Legendre on `[−1,1]³`.  The hexahedral
            // fem-element family (`HexL2GL`/`HexQk`/`HexRTk`/`HexNDk` and the
            // `HexQ1` geometry element, all of which `eval_vol_space` and
            // `geo_ref_elem_from_mesh` use) is defined on `[−1,1]³`, exactly
            // like the rest of fem-rs's 3-D kernels; the DPG quadrature and
            // reference geometry must use the same domain, otherwise
            // `x(ξ)` only covers a sub-box of each element and the measure is
            // off by `2^dim`.
            let (pts1d, wts1d) = fem_element::quadrature::gauss_legendre_arbitrary(order as usize);
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
            let mut face_dof_offsets = Vec::with_capacity(face_info.len() + 1);
            face_dof_offsets.push(0);
            for f in 0..face_info.len() {
                let n = dofs_per_face(is_quad_face[f]);
                face_dof_offsets.push(face_dof_offsets[f] + n);
            }
            let (n_dofs, face_dofs) = if dim == 2 {
                let interior_per_face = (p + 1).saturating_sub(2);
                // Interior dofs start AFTER the vertex dofs (MFEM
                // `H1_Trace_FECollection`: face dofs live on mesh vertices and
                // face-interior entities, so the global count is
                // n_vertices_touched + n_faces × interior_per_face — NOT the
                // per-face dof-count sum, which would leave phantom dofs).
                let base = n_vdofs;
                let mut face_dofs: Vec<Vec<usize>> = Vec::with_capacity(face_node_ids.len());
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
                (base + face_node_ids.len() * interior_per_face, face_dofs)
            } else {
                // ── 3-D H1 trace = MFEM `H1_Trace_FECollection(p, 3)`
                // (`H1_FECollection(p, 2)` on the skeleton): 1 dof per mesh
                // vertex, `p − 1` per mesh EDGE (lines of the 3-D mesh, shared
                // by every face meeting at the edge), and `(p−1)²` / `(p−1)(p−2)/2`
                // face-interior dofs per quadrilateral / triangular face.
                //
                // The face dof list is built in the **trace-basis node order**
                // of `eval_face_lagrange`: index `k` there is the tensor
                // Lagrange node `(s,t) = (k%(p+1)/p, k/(p+1)/p)` on a quad
                // face and the equispaced node `(a/p, b/p)`, `a+b ≤ p`, on a
                // triangle.  Every node is mapped to the skeleton entity that
                // carries it, so shared vertices/edges use a single global dof
                // (the tensor order differs from MFEM's `[v0 v1 v2 v3 | edges |
                // interior]` face-dof order, but the *association*
                // basis-function ↔ global dof is what must be consistent, and
                // this one is).
                let n_edof = p.saturating_sub(1);
                // Global edge table: first-seen order over (element, local
                // edge), each edge stored in the canonical (min, max) vertex
                // direction — same convention as `TraceSpace` / MFEM's
                // `DSTable::Push(min, max)`.
                let mut edge_map: std::collections::HashMap<(u32, u32), usize> =
                    std::collections::HashMap::new();
                let mut n_edges = 0usize;
                for e in mesh.elem_iter() {
                    let et = mesh.element_type(e);
                    let en = mesh.element_nodes(e);
                    for ev in mfem_local_edges(et) {
                        let (a, b) = (en[ev[0]], en[ev[1]]);
                        let key = if a < b { (a, b) } else { (b, a) };
                        edge_map.entry(key).or_insert_with(|| {
                            n_edges += 1;
                            n_edges - 1
                        });
                    }
                }
                let e_base = n_vdofs;
                let f_base = n_vdofs + n_edges * n_edof;
                // Interior-dof base of each face (prefix sums, face order).
                let mut if_base = Vec::with_capacity(face_info.len());
                let mut acc = f_base;
                for f in 0..face_info.len() {
                    if_base.push(acc);
                    acc += interior_per_face_of(p, is_quad_face[f]);
                }
                let mut face_dofs: Vec<Vec<usize>> = Vec::with_capacity(face_info.len());
                for f in 0..face_info.len() {
                    let cyc = &face_node_ids[f];
                    let is_quad = is_quad_face[f];
                    let mut dofs = Vec::with_capacity(dofs_per_face(is_quad));
                    // Global dof of the face-local edge `ei` at face-local
                    // parameter `u ∈ [0,1]` (measured from the edge's first
                    // reference-face vertex); `u·p` must be an integer in
                    // `1..=p−1`.
                    let edge_dof = |ei: usize, u: f64| -> usize {
                        let evert = mfem_face_edges(is_quad)[ei];
                        let (a, b) = (cyc[evert[0]], cyc[evert[1]]);
                        let key = if a < b { (a, b) } else { (b, a) };
                        let g = *edge_map
                            .get(&key)
                            .expect("3-D H1 trace: face edge missing from edge table");
                        let pos = if a < b { u } else { 1.0 - u };
                        let m = (pos * p as f64).round();
                        debug_assert!(m >= 1.0 && m <= (p - 1) as f64, "edge node out of range");
                        e_base + g * n_edof + (m as usize - 1)
                    };
                    if is_quad {
                        let mut ic = 0usize;
                        for t in 0..=p {
                            for s in 0..=p {
                                let d = if (s == 0 || s == p) && (t == 0 || t == p) {
                                    let ci = match (s == p, t == p) {
                                        (false, false) => 0,
                                        (true, false) => 1,
                                        (true, true) => 2,
                                        (false, true) => 3,
                                    };
                                    vertex_dof(cyc[ci])
                                } else if s == 0 {
                                    edge_dof(3, 1.0 - t as f64 / p as f64)
                                } else if s == p {
                                    edge_dof(1, t as f64 / p as f64)
                                } else if t == 0 {
                                    edge_dof(0, s as f64 / p as f64)
                                } else if t == p {
                                    edge_dof(2, 1.0 - s as f64 / p as f64)
                                } else {
                                    let d = if_base[f] + ic;
                                    ic += 1;
                                    d
                                };
                                dofs.push(d);
                            }
                        }
                        debug_assert_eq!(ic, interior_per_face_of(p, true));
                    } else {
                        let mut ic = 0usize;
                        for row in 0..=p {
                            for a in 0..=row {
                                let b = row - a;
                                let d = if a == 0 && b == 0 {
                                    vertex_dof(cyc[0])
                                } else if a == p {
                                    vertex_dof(cyc[1])
                                } else if b == p {
                                    vertex_dof(cyc[2])
                                } else if a + b == p {
                                    // hypotenuse (c1, c2): the node is at
                                    // (s,t) = (a/p, b/p), i.e. `b/p` of the way
                                    // from c1 = (1,0) to c2 = (0,1)
                                    edge_dof(1, b as f64 / p as f64)
                                } else if b == 0 {
                                    // (c0, c1), parameter from c0
                                    edge_dof(0, a as f64 / p as f64)
                                } else if a == 0 {
                                    // (c2, c0), parameter from c2
                                    edge_dof(2, 1.0 - b as f64 / p as f64)
                                } else {
                                    let d = if_base[f] + ic;
                                    ic += 1;
                                    d
                                };
                                dofs.push(d);
                            }
                        }
                        debug_assert_eq!(ic, interior_per_face_of(p, false));
                    }
                    debug_assert_eq!(dofs.len(), dofs_per_face(is_quad));
                    face_dofs.push(dofs);
                }
                (acc, face_dofs)
            };
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
/// Local face vertex cycles, verbatim from MFEM
/// `Geometry::Constants<ElemType>::FaceVert` — the same tables as
/// [`mfem_local_faces`], keyed here off the element's node count (`local_face_table`
/// is the node-index form used by the trace assemblers, `mfem_local_faces` the
/// static `ElementType` form used by [`TraceSpace`]).
///
/// The winding matters: MFEM's tables list every face so that
/// `J_s × J_t` (with `J_s` towards the 2nd and `J_t` towards the 3rd listed
/// vertex) points **out of the element** — this is what makes the ±1
/// element-face orientation of the trace assembly equal to the sign relating
/// the canonical face normal to the element's outward normal.  An
/// "outward-inconsistent" table entry (e.g. the lexicographic cycle
/// `[0,1,2,3]` for a hexahedron's bottom face) silently flips the trace
/// contribution of every element that meets the face through that local face.
pub fn local_face_table(elem_nodes: &[u32], dim: usize) -> Vec<Vec<usize>> {
    match (elem_nodes.len(), dim) {
        (3, 2) => vec![vec![0, 1], vec![1, 2], vec![2, 0]],
        (4, 2) => vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 0]],
        (4, 3) => vec![
            vec![1, 2, 3],
            vec![0, 3, 2],
            vec![0, 1, 3],
            vec![0, 2, 1],
        ],
        (8, 3) => vec![
            vec![3, 2, 1, 0],
            vec![0, 1, 5, 4],
            vec![1, 2, 6, 5],
            vec![2, 3, 7, 6],
            vec![3, 0, 4, 7],
            vec![4, 5, 6, 7],
        ],
        _ => panic!(
            "local_face_table: unsupported (npe={}, dim={})",
            elem_nodes.len(),
            dim
        ),
    }
}

// ─── Face parametrization ────────────────────────────────────────────────────

/// Element-local face vertex order matching the canonical (stored) face
/// cycle — MFEM's `Loc1`/`Loc2` point-matrix convention
/// (`Mesh::GetLocalQuadToHexTransformation` / `GetLocalTriToTetTransformation`
/// build the point matrix with column `j` = the element-reference vertex of
/// the local face vertex that coincides with canonical face vertex `j`,
/// i.e. `hv[qo[j]]` / `tv[to[j]]`).
///
/// Feeding this order to [`face_param_to_elem_ref`] reproduces MFEM's
/// `FaceElementTransformations::SetAllIntPoints` chaining exactly (the
/// element-side reference image of a canonical face parameter), and — unlike
/// a Newton-refined seed — keeps the reference face-plane coordinates exact
/// (`x_ref = ±1`).  The exact plane coordinate matters: the ND/RT reference
/// bases are evaluated with boundary special cases at the face plane, so a
/// point drifted to `−1 + 2e−16` by Newton falls into the interior branch and
/// corrupts the assembled trace block.
pub fn local_face_canonical_order(nodes: &[u32], lf: &[usize], canonical: &[u32]) -> Vec<usize> {
    canonical
        .iter()
        .map(|&c| {
            lf.iter()
                .copied()
                .find(|&k| nodes[k as usize] == c)
                .unwrap_or_else(|| {
                    panic!(
                        "local_face_canonical_order: local face {lf:?} does not carry canonical \
                         vertex {c}"
                    )
                })
        })
        .collect()
}

/// Reference-domain node coordinates of the element geometries.
///
/// Simplices and quads use the fem-element `[0,1]^dim` convention; the
/// hexahedral family (`HexQ1`/`HexQk`/`HexL2GL`/`HexRTk`/`HexNDk`) is defined
/// on `[−1,1]³`, so the hex entries below are the `[−1,1]³` corners in the
/// same (MFEM) vertex order.  [`vol_quadrature`] and the element bases must
/// share this convention.
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
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
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

/// Face-interior dof count of an order-`p` H1 trace face:
/// `(p−1)²` on a quadrilateral, `(p−1)(p−2)/2` on a triangle (MFEM
/// `H1_FECollection(p, 2)`).
fn interior_per_face_of(p: usize, is_quad: bool) -> usize {
    let q = p.saturating_sub(1);
    if is_quad {
        q * q
    } else {
        q * p.saturating_sub(2) / 2
    }
}

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

// ─── MFEM reference-interval point sets ──────────────────────────────────────

/// MFEM `Poly_1D::ClosedPoints(n)`: the `n + 1` Gauss-Lobatto points on
/// `[0,1]`, ascending.  (`Polynomial degree n`, endpoints included.)
pub fn lobatto_points_01(n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![0.0];
    }
    fem_element::quadrature::gauss_lobatto_01_arbitrary(n + 1).0
}

/// MFEM `Poly_1D::OpenPoints(n)`: the `n + 1` Gauss-Legendre points on
/// `[0,1]`, ascending (endpoints excluded).
pub fn legendre_points_01(n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![0.5];
    }
    fem_element::quadrature::gauss_legendre_01(n + 1).0
}

/// Lagrange basis at the nodes `pts` (ascending), evaluated at `x`.
fn lagrange_at(pts: &[f64], x: f64, out: &mut [f64]) {
    let n = pts.len();
    for j in 0..n {
        let mut v = 1.0;
        for k in 0..n {
            if k != j {
                v *= (x - pts[k]) / (pts[j] - pts[k]);
            }
        }
        out[j] = v;
    }
}

// ─── MFEM ND face (trace) elements ───────────────────────────────────────────

/// Row-major dense inverse (Gauss–Jordan).  Matrices here are the small
/// Vandermonde systems of the ND face elements (`n ≤ 30`).
fn dense_inverse(a: &[f64], n: usize) -> Vec<f64> {
    let mut m = vec![0.0_f64; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            m[i * 2 * n + j] = a[i * n + j];
        }
        m[i * 2 * n + n + i] = 1.0;
    }
    for c in 0..n {
        let mut piv = c;
        let mut bv = m[c * 2 * n + c].abs();
        for r in (c + 1)..n {
            let v = m[r * 2 * n + c].abs();
            if v > bv {
                bv = v;
                piv = r;
            }
        }
        assert!(bv > 1e-14, "dense_inverse: singular Vandermonde");
        if piv != c {
            for k in 0..2 * n {
                m.swap(c * 2 * n + k, piv * 2 * n + k);
            }
        }
        let p = m[c * 2 * n + c];
        for k in 0..2 * n {
            m[c * 2 * n + k] /= p;
        }
        for r in 0..n {
            if r != c {
                let f = m[r * 2 * n + c];
                if f != 0.0 {
                    for k in 0..2 * n {
                        m[r * 2 * n + k] -= f * m[c * 2 * n + k];
                    }
                }
            }
        }
    }
    let mut inv = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = m[i * 2 * n + n + j];
        }
    }
    inv
}

/// Reference DOF count of the ND trace face element — MFEM
/// `ND_TriangleElement::GetDof()` / `ND_QuadrilateralElement::GetDof()`
/// (`p(p+2)` resp. `2p(p+1)`), split into edge and face-interior parts.
pub fn nd_face_dofs(p: usize, is_quad: bool) -> usize {
    let n_edges = if is_quad { 4 } else { 3 };
    n_edges * p + nd_face_interior_dofs(p, is_quad)
}

/// Face-interior DOF count of the ND trace face element (MFEM
/// `ND_dof[TRIANGLE] = p(p−1)`, `ND_dof[SQUARE] = 2p(p−1)`).
pub fn nd_face_interior_dofs(p: usize, is_quad: bool) -> usize {
    let pm1 = p.saturating_sub(1);
    if is_quad {
        2 * p * pm1
    } else {
        p * pm1
    }
}

/// Reference coordinates of the DOFs of MFEM `ND_TriangleElement(p)` /
/// `ND_QuadrilateralElement(p)` in the element's DOF order.
pub fn nd_face_dof_nodes(p: usize, is_quad: bool) -> Vec<[f64; 2]> {
    let mut out = Vec::with_capacity(nd_face_dofs(p, is_quad));
    let eop = legendre_points_01(p - 1); // p points
    if is_quad {
        let cp = lobatto_points_01(p); // p+1 points
        for i in 0..p {
            out.push([eop[i], cp[0]]);
        }
        for j in 0..p {
            out.push([cp[p], eop[j]]);
        }
        for i in 0..p {
            out.push([eop[p - 1 - i], cp[p]]);
        }
        for j in 0..p {
            out.push([cp[0], eop[p - 1 - j]]);
        }
        for j in 1..p {
            for i in 0..p {
                out.push([eop[i], cp[j]]);
            }
        }
        for j in 0..p {
            for i in 1..p {
                out.push([cp[i], eop[j]]);
            }
        }
    } else {
        for i in 0..p {
            out.push([eop[i], 0.0]);
        }
        for i in 0..p {
            out.push([eop[p - 1 - i], eop[i]]);
        }
        for i in 0..p {
            out.push([0.0, eop[p - 1 - i]]);
        }
        if p >= 2 {
            let iop = legendre_points_01(p - 2); // p-1 points
            let pm2 = p - 2;
            for j in 0..=pm2 {
                for i in 0..=(pm2 - j) {
                    let w = iop[i] + iop[j] + iop[pm2 - i - j];
                    out.push([iop[i] / w, iop[j] / w]);
                    out.push([iop[i] / w, iop[j] / w]);
                }
            }
        }
    }
    out
}

/// Reference tangent direction `tk` of every DOF of MFEM
/// `ND_TriangleElement(p)` / `ND_QuadrilateralElement(p)`, in the element's
/// DOF order (matching [`nd_face_dof_nodes`] / [`eval_face_nd`]).
///
/// Used by the ND-trace interpolation of boundary data — MFEM
/// `VectorFiniteElement::Project_ND`: `dof_k = v(x_k) · (J tk_k)` with `J` the
/// canonical face Jacobian at the DOF node `x_k`.  The directions follow the
/// reference-face cycle: on a quad, edges 0/1 run +s/+t while edges 2/3 run
/// the cycle direction (−s/−t) — MFEM `dof2tk` = 0,1,2,3 with
/// `tk = {±e_s, ±e_t}`; interior x-/y-dofs use +s/+t.  On a triangle the
/// directions are MFEM's `tk = {e_s, −e_s+e_t, e_t, e_t}` per edge/interior.
pub fn nd_face_dof_tangents(p: usize, is_quad: bool) -> Vec<[f64; 2]> {
    let mut out = Vec::with_capacity(nd_face_dofs(p, is_quad));
    if is_quad {
        for _ in 0..p {
            out.push([1.0, 0.0]); // edge 0: +s
        }
        for _ in 0..p {
            out.push([0.0, 1.0]); // edge 1: +t
        }
        for _ in 0..p {
            out.push([-1.0, 0.0]); // edge 2: cycle direction (−s)
        }
        for _ in 0..p {
            out.push([0.0, -1.0]); // edge 3: cycle direction (−t)
        }
        for _ in 1..p {
            for _ in 0..p {
                out.push([1.0, 0.0]); // interior x-dofs
            }
        }
        for _ in 0..p {
            for _ in 1..p {
                out.push([0.0, 1.0]); // interior y-dofs
            }
        }
    } else {
        for _ in 0..p {
            out.push(ND_TRI_TK[0]);
        }
        for _ in 0..p {
            out.push(ND_TRI_TK[1]);
        }
        for _ in 0..p {
            out.push(ND_TRI_TK[2]);
        }
        if p >= 2 {
            let pm2 = p - 2;
            for _j in 0..=pm2 {
                for _i in 0..=(pm2 - _j) {
                    out.push(ND_TRI_TK[0]);
                    out.push(ND_TRI_TK[3]);
                }
            }
        }
    }
    out
}

/// The `tk` direction vectors of MFEM `ND_TriangleElement`
/// (`tk = {1,0, −1,1, 0,−1, 0,1}`).
const ND_TRI_TK: [[f64; 2]; 4] = [[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0], [0.0, 1.0]];

/// `dof2tk` of MFEM `ND_TriangleElement(p)` (index into [`ND_TRI_TK`]).
fn nd_tri_dof2tk(p: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(nd_face_dofs(p, false));
    for _ in 0..p {
        out.push(0);
    }
    for _ in 0..p {
        out.push(1);
    }
    for _ in 0..p {
        out.push(2);
    }
    if p >= 2 {
        let pm2 = p - 2;
        for j in 0..=pm2 {
            for i in 0..=(pm2 - j) {
                let _ = (i, j);
                out.push(0);
                out.push(3);
            }
        }
    }
    out
}

/// MFEM `Poly_1D::CalcBasis(p, x, u)` = `CalcChebyshev(p, x, u)`: the
/// hierarchical Chebyshev basis `u_k = T_k(2x − 1)`, `k = 0..=p`.
fn chebyshev_basis(p: usize, x: f64, out: &mut [f64]) {
    out[0] = 1.0;
    if p == 0 {
        return;
    }
    let z = 2.0 * x - 1.0;
    out[1] = z;
    for n in 1..p {
        out[n + 1] = 2.0 * z * out[n] - out[n - 1];
    }
}

/// The polynomial expansion functions of MFEM `ND_TriangleElement(p)`
/// (before the Vandermonde inversion), row-major `[n_basis × 2]`.
fn nd_tri_expansion(p: usize, x: f64, y: f64) -> Vec<f64> {
    let pm1 = p - 1;
    let mut lx = vec![0.0; pm1 + 1];
    let mut ly = vec![0.0; pm1 + 1];
    let mut ll = vec![0.0; pm1 + 1];
    chebyshev_basis(pm1, x, &mut lx);
    chebyshev_basis(pm1, y, &mut ly);
    chebyshev_basis(pm1, 1.0 - x - y, &mut ll);
    let c = 1.0 / 3.0;
    let n_basis = 2 * (pm1 + 1) * (pm1 + 2) / 2 + (pm1 + 1);
    let mut out = vec![0.0_f64; n_basis * 2];
    let mut n = 0;
    for j in 0..=pm1 {
        for i in 0..=(pm1 - j) {
            let s = lx[i] * ly[j] * ll[pm1 - i - j];
            out[n * 2] = s;
            out[n * 2 + 1] = 0.0;
            n += 1;
            out[n * 2] = 0.0;
            out[n * 2 + 1] = s;
            n += 1;
        }
    }
    for j in 0..=pm1 {
        let s = lx[pm1 - j] * ly[j];
        out[n * 2] = s * (y - c);
        out[n * 2 + 1] = -s * (x - c);
        n += 1;
    }
    out
}

/// Evaluate the reference basis of MFEM `ND_TriangleElement(p)` at `(x, y)`;
/// `out` is row-major `[n_dofs × 2]` in the element's DOF order.
pub fn nd_tri_basis(p: usize, x: f64, y: f64, out: &mut [f64]) {
    assert!(p >= 1, "nd_tri_basis requires p >= 1");
    let pm1 = p - 1;
    let n_basis = 2 * (pm1 + 1) * (pm1 + 2) / 2 + (pm1 + 1);
    let nodes = nd_face_dof_nodes(p, false);
    let d2t = nd_tri_dof2tk(p);
    let n_dof = nodes.len();
    assert_eq!(out.len(), n_dof * 2);
    // Vandermonde T[n_basis × n_dof]: T[k][m] = expansion_k(node_m) · tk_m
    let mut t = vec![0.0_f64; n_basis * n_dof];
    for m in 0..n_dof {
        let e = nd_tri_expansion(p, nodes[m][0], nodes[m][1]);
        let tk = ND_TRI_TK[d2t[m]];
        for k in 0..n_basis {
            t[k * n_dof + m] = e[k * 2] * tk[0] + e[k * 2 + 1] * tk[1];
        }
    }
    let tinv = dense_inverse(&t, n_basis); // [n_dof × n_basis]
    let u = nd_tri_expansion(p, x, y);
    for m in 0..n_dof {
        let mut vx = 0.0;
        let mut vy = 0.0;
        for k in 0..n_basis {
            let c = tinv[m * n_basis + k];
            vx += c * u[k * 2];
            vy += c * u[k * 2 + 1];
        }
        out[m * 2] = vx;
        out[m * 2 + 1] = vy;
    }
}

/// Evaluate the reference basis of MFEM `ND_QuadrilateralElement(p)` at
/// `(s, t)`; `out` is row-major `[n_dofs × 2]` in the element's DOF order.
pub fn nd_quad_basis(p: usize, s: f64, t: f64, out: &mut [f64]) {
    assert!(p >= 1, "nd_quad_basis requires p >= 1");
    let cp = lobatto_points_01(p); // p+1
    let op = legendre_points_01(p - 1); // p
    let mut lcx = vec![0.0; p + 1];
    let mut lcy = vec![0.0; p + 1];
    let mut lox = vec![0.0; p];
    let mut loy = vec![0.0; p];
    lagrange_at(&cp, s, &mut lcx);
    lagrange_at(&cp, t, &mut lcy);
    lagrange_at(&op, s, &mut lox);
    lagrange_at(&op, t, &mut loy);
    assert_eq!(out.len(), nd_face_dofs(p, true) * 2);
    let mut k = 0;
    for i in 0..p {
        out[k * 2] = lox[i] * lcy[0];
        out[k * 2 + 1] = 0.0;
        k += 1;
    }
    for j in 0..p {
        out[k * 2] = 0.0;
        out[k * 2 + 1] = lcx[p] * loy[j];
        k += 1;
    }
    for i in 0..p {
        out[k * 2] = -lox[p - 1 - i] * lcy[p];
        out[k * 2 + 1] = 0.0;
        k += 1;
    }
    for j in 0..p {
        out[k * 2] = 0.0;
        out[k * 2 + 1] = -lcx[0] * loy[p - 1 - j];
        k += 1;
    }
    for j in 1..p {
        for i in 0..p {
            out[k * 2] = lox[i] * lcy[j];
            out[k * 2 + 1] = 0.0;
            k += 1;
        }
    }
    for j in 0..p {
        for i in 1..p {
            out[k * 2] = 0.0;
            out[k * 2 + 1] = lcx[i] * loy[j];
            k += 1;
        }
    }
}

/// Reference ND face basis (dispatcher over the face geometry).
pub fn eval_face_nd(p: usize, is_quad: bool, param: &[f64], out: &mut [f64]) {
    if is_quad {
        nd_quad_basis(p, param[0], param[1], out);
    } else {
        nd_tri_basis(p, param[0], param[1], out);
    }
}

/// Covariant (tangential) map of a reference face 2-vector to physical space —
/// MFEM `VectorFiniteElement::CalcVShape_ND(Trans, shape)` with `Trans` the
/// face transformation: `w = J (JᵀJ)⁻¹ v` (equivalently `v · J⁺` with the
/// left inverse `J⁺ = (JᵀJ)⁻¹Jᵀ`).
///
/// `jac` holds the two surface-tangent columns `∂x/∂s`, `∂x/∂t`.
pub fn map_face_nd_to_phys(jac: &[[f64; 3]; 2], v: &[f64; 2]) -> [f64; 3] {
    let j0 = jac[0];
    let j1 = jac[1];
    let g00 = j0[0] * j0[0] + j0[1] * j0[1] + j0[2] * j0[2];
    let g01 = j0[0] * j1[0] + j0[1] * j1[1] + j0[2] * j1[2];
    let g11 = j1[0] * j1[0] + j1[1] * j1[1] + j1[2] * j1[2];
    let det = g00 * g11 - g01 * g01;
    let y0 = (g11 * v[0] - g01 * v[1]) / det;
    let y1 = (g00 * v[1] - g01 * v[0]) / det;
    [
        j0[0] * y0 + j1[0] * y1,
        j0[1] * y0 + j1[1] * y1,
        j0[2] * y0 + j1[2] * y1,
    ]
}

// ─── MFEM mesh-entity tables ─────────────────────────────────────────────────

/// Local edge vertex pairs, verbatim from MFEM
/// `Geometry::Constants<ElemType>::Edges`.
pub fn mfem_local_edges(et: ElementType) -> &'static [[usize; 2]] {
    match et {
        ElementType::Tri3 | ElementType::Tri6 => &[[0, 1], [1, 2], [2, 0]],
        ElementType::Quad4 => &[[0, 1], [1, 2], [2, 3], [3, 0]],
        ElementType::Tet4 => &[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        ElementType::Hex8 => &[
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
        ],
        _ => panic!("mfem_local_edges: unsupported {et:?}"),
    }
}

/// Local face vertex cycles, verbatim from MFEM
/// `Geometry::Constants<ElemType>::FaceVert` (2-D: the element's edges).
pub fn mfem_local_faces(et: ElementType, dim: usize) -> &'static [&'static [usize]] {
    match (et, dim) {
        (ElementType::Tri3 | ElementType::Tri6, 2) => &[&[0, 1], &[1, 2], &[2, 0]],
        (ElementType::Quad4, 2) => &[&[0, 1], &[1, 2], &[2, 3], &[3, 0]],
        (ElementType::Tet4, 3) => &[&[1, 2, 3], &[0, 3, 2], &[0, 1, 3], &[0, 2, 1]],
        (ElementType::Hex8, 3) => &[
            &[3, 2, 1, 0],
            &[0, 1, 5, 4],
            &[1, 2, 6, 5],
            &[2, 3, 7, 6],
            &[3, 0, 4, 7],
            &[4, 5, 6, 7],
        ],
        _ => panic!("mfem_local_faces: unsupported {et:?} dim {dim}"),
    }
}

/// Face-local edge vertex pairs — MFEM `Triangle::GetEdgeVertices` /
/// `Quadrilateral::GetEdgeVertices` (the reference face's edge cycle).
pub fn mfem_face_edges(is_quad: bool) -> &'static [[usize; 2]] {
    if is_quad {
        &[[0, 1], [1, 2], [2, 3], [3, 0]]
    } else {
        &[[0, 1], [1, 2], [2, 0]]
    }
}

/// MFEM `FiniteElementSpace::EncodeDof`: sign-encodes a negative local index
/// into a global DOF id.
pub fn encode_dof(base: usize, idx: i32) -> i32 {
    if idx >= 0 {
        base as i32 + idx
    } else {
        -1 - (base as i32 + (-1 - idx))
    }
}

// ─── 3-D face geometry (canonical face parametrisation) ─────────────────────

/// Physical coordinates of the canonical face parametrisation of face `f`:
/// affine on triangles (reference `(0,0),(1,0),(0,1)`), bilinear on quads
/// (reference `[0,1]²`, vertices in the face's canonical cycle order) — the
/// same map MFEM's `GetFaceTransformation` builds.
pub fn face_point_3d<M: MeshTopology>(tr: &TraceSpace<M>, f: usize, param: &[f64]) -> [f64; 3] {
    let c = face_vertex_coords(tr, f);
    let (s, t) = (param[0], param[1]);
    let mut x = [0.0; 3];
    if tr.is_quad_face(f) {
        for d in 0..3 {
            x[d] = (1.0 - s) * (1.0 - t) * c[0][d]
                + s * (1.0 - t) * c[1][d]
                + s * t * c[2][d]
                + (1.0 - s) * t * c[3][d];
        }
    } else {
        for d in 0..3 {
            x[d] = (1.0 - s - t) * c[0][d] + s * c[1][d] + t * c[2][d];
        }
    }
    x
}

/// Physical coordinates of the canonical face vertices.
fn face_vertex_coords<M: MeshTopology>(tr: &TraceSpace<M>, f: usize) -> Vec<[f64; 3]> {
    tr.face_nodes(f)
        .iter()
        .map(|&n| {
            let p = tr.mesh().node_coords(n);
            [p[0], p[1], p[2]]
        })
        .collect()
}

/// Face Jacobian columns `(∂x/∂s, ∂x/∂t)` of the canonical face
/// parametrisation at `param` (constant on affine triangles, bilinear on
/// quads).
pub fn face_jacobian_3d<M: MeshTopology>(
    tr: &TraceSpace<M>,
    f: usize,
    param: &[f64],
) -> [[f64; 3]; 2] {
    let c = face_vertex_coords(tr, f);
    let (s, t) = (param[0], param[1]);
    let mut jac = [[0.0; 3]; 2];
    if tr.is_quad_face(f) {
        for d in 0..3 {
            jac[0][d] = (c[1][d] - c[0][d]) * (1.0 - t) + (c[2][d] - c[3][d]) * t;
            jac[1][d] = (c[3][d] - c[0][d]) * (1.0 - s) + (c[2][d] - c[1][d]) * s;
        }
    } else {
        for d in 0..3 {
            jac[0][d] = c[1][d] - c[0][d];
            jac[1][d] = c[2][d] - c[0][d];
        }
    }
    jac
}

/// MFEM `CalcOrtho(J_face)` for a 3-D face: the cross product `J_s × J_t`
/// (column-major `d` order of the MFEM helper).  Note the MFEM convention:
/// for a triangular face the length is `2 · area` (the reference triangle has
/// area 1/2), for a quad face it is the reference-square-normalised surface
/// Jacobian — NOT the Euclidean face measure.  Paired with MFEM's
/// reference-domain quadrature weights this integrates exactly like
/// `Trans.Weight()`.
pub fn face_normal_3d(jac: &[[f64; 3]; 2]) -> [f64; 3] {
    let (a, b) = (jac[0], jac[1]);
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Euclidean norm.
pub fn norm3(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

// ─── Trace (skeleton) spaces ─────────────────────────────────────────────────

/// Trace-space family — 1:1 with MFEM's `*_Trace_FECollection` classes:
///
/// | kind | MFEM collection | per-vertex | per-edge | per tri face | per quad face |
/// |------|-----------------|-----------|----------|--------------|---------------|
/// | H1 | `H1_Trace_FECollection(p,dim)` = `H1_FECollection(p, dim−1)` | 1 | p−1 | (p−1)(p−2)/2 | (p−1)² |
/// | Rt | `RT_Trace_FECollection(p,dim)` = `RT_FECollection(p, dim)` restricted to faces | 0 | 0 | (p+1)(p+2)/2 | (p+1)² |
/// | Nd | `ND_Trace_FECollection(p,dim)` = `ND_FECollection(p, dim−1)` | 0 | p | p(p−1) | 2p(p−1) |
///
/// For `Rt` the collection is built through MFEM's `InitFaces(p, dim, …)`
/// (`fe_coll.cpp`), which populates ONLY the face geometries of the given
/// dimension — so a 3-D `RT_Trace` space has face-interior DOFs only (its
/// face element is `L2_TriangleElement(p)` / `L2_QuadrilateralElement(p)`),
/// while a 2-D one has edge DOFs only (`L2_SegmentElement(p)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceKind {
    /// Continuous nodal trace (`H1_Trace_FECollection`).
    H1,
    /// Normal-moment trace (`RT_Trace_FECollection`); orientation handled by
    /// the ±1 scale of the trace integrators (MFEM `TraceIntegrator`).
    Rt,
    /// Tangential-vector trace (`ND_Trace_FECollection`); the edge DOFs are
    /// shared between the faces meeting at that edge.
    Nd,
}

/// Mesh-skeleton trace space with MFEM-compatible entity tables and global
/// DOF numbering.
///
/// The entity enumeration follows MFEM exactly so that, on a mesh whose node
/// and element numbering matches MFEM's, the global trace DOF ids agree
/// DOF-for-DOF:
///
/// * **edges** are numbered in first-seen order over `(element, local edge)`
///   pairs (MFEM `Mesh::GetVertexToVertexTable` + `DSTable::Push(min,max)`
///   canonicalises each edge to its `(min vertex, max vertex)` direction);
/// * **faces** are numbered in first-seen order over `(element, local face)`
///   pairs, where the local face list is MFEM
///   `Geometry::Constants<>::FaceVert`; the face keeps the generating
///   ("Elem1") element's local vertex cycle verbatim (MFEM
///   `Mesh::GenerateFaces` stores `FaceInfo` with Elem1 as generator);
/// * the global DOF numbering is MFEM's `Base` layout: all vertex DOFs, then
///   all edge DOFs, then the face-interior DOFs face by face;
/// * an edge DOF list runs along the edge's **global** direction
///   `(min → max)`, and a face's DOF list is
///   `[vertex dofs | edge dofs (face-local edge order, oriented) | interior]`
///   — exactly MFEM `FiniteElementSpace::GetFaceDofs`.
///
/// Signed (sign-encoded) ids as returned by MFEM are produced by
/// [`Self::face_signed_dofs`] / [`Self::element_signed_dofs`] /
/// [`Self::element_trace_signed_dofs`]; the unsigned ids used by the DPG
/// trace assembly are [`Self::face_dof_list`] /
/// [`Self::element_trace_dof_list`] (with the orientation signs left to the
/// trace integrators' ±1 `scale`, as in MFEM's `AssembleTraceFaceMatrix`).
pub struct TraceSpace<M: MeshTopology> {
    mesh: M,
    dim: usize,
    order: u8,
    kind: TraceKind,
    /// Canonical face vertex cycles (generating element's local face).
    faces: Vec<Vec<u32>>,
    /// Per face: `(global edge id, orientation of the face-local edge
    /// direction against the global edge direction)`.
    face_edge_list: Vec<Vec<(usize, i32)>>,
    /// Generating element ("Elem1") and the other element ("Elem2") of a face.
    face_elems: Vec<(u32, Option<u32>)>,
    /// Element faces in MFEM local face order.
    elem_faces: Vec<Vec<usize>>,
    /// Element face orientation parity (±1) against the canonical face cycle.
    elem_face_ori: Vec<Vec<i32>>,
    /// Global edge table, each entry `(min vertex, max vertex)`.
    edges: Vec<[u32; 2]>,
    n_vdof: usize,
    n_edof: usize,
    /// Interior DOFs per face.
    face_nf: Vec<usize>,
    /// First global face-interior DOF of each face.
    face_fbase: Vec<usize>,
    n_dofs: usize,
    /// Per face: DOF ids in the face element's DOF order (unsigned).
    face_dof_list: Vec<Vec<usize>>,
    /// Per face: MFEM sign-encoded DOF ids (`GetFaceDofs`).
    face_signed: Vec<Vec<i32>>,
    /// Per element: the DPG trace block = concatenation of the element's
    /// faces' [`Self::face_dof_list`] in local face order (unsigned) — this is
    /// what MFEM's `DPGWeakForm::Assemble` builds from `GetFaceVDofs`.
    elem_trace_dof_list: Vec<Vec<usize>>,
}

impl<M: MeshTopology> TraceSpace<M> {
    /// Continuous nodal trace (MFEM `H1_Trace_FECollection(p, dim)`).
    pub fn new_h1(mesh: M, p: u8) -> Self {
        Self::build(mesh, p, TraceKind::H1)
    }

    /// Normal-moment trace (MFEM `RT_Trace_FECollection(p, dim)`).
    pub fn new_rt(mesh: M, p: u8) -> Self {
        Self::build(mesh, p, TraceKind::Rt)
    }

    /// Tangential-vector trace (MFEM `ND_Trace_FECollection(p, dim)`).
    pub fn new_nd(mesh: M, p: u8) -> Self {
        Self::build(mesh, p, TraceKind::Nd)
    }

    fn build(mesh: M, order: u8, kind: TraceKind) -> Self {
        let dim = mesh.dim() as usize;
        assert!(dim == 2 || dim == 3, "TraceSpace: dim {dim} unsupported");

        // ── entity tables (MFEM ordering) ───────────────────────────────────
        let mut edge_map: std::collections::HashMap<(u32, u32), usize> =
            std::collections::HashMap::new();
        let mut edges: Vec<[u32; 2]> = Vec::new();
        let mut face_map: std::collections::HashMap<Vec<u32>, usize> =
            std::collections::HashMap::new();
        let mut faces: Vec<Vec<u32>> = Vec::new();
        let mut face_elems: Vec<(u32, Option<u32>)> = Vec::new();
        let mut elem_faces: Vec<Vec<usize>> = vec![Vec::new(); mesh.n_elements()];
        let mut elem_face_ori: Vec<Vec<i32>> = vec![Vec::new(); mesh.n_elements()];

        for e in 0..mesh.n_elements() as u32 {
            let et = mesh.element_type(e);
            let en = mesh.element_nodes(e);
            for ev in mfem_local_edges(et) {
                let (a, b) = (en[ev[0]], en[ev[1]]);
                let key = if a < b { (a, b) } else { (b, a) };
                edge_map.entry(key).or_insert_with(|| {
                    edges.push([key.0, key.1]);
                    edges.len() - 1
                });
            }
            let lfs = mfem_local_faces(et, dim);
            for lf in lfs.iter() {
                let cyc: Vec<u32> = lf.iter().map(|&k| en[k]).collect();
                let mut key = cyc.clone();
                key.sort_unstable();
                let fid = match face_map.get(&key) {
                    Some(&f) => {
                        let cur = face_elems[f];
                        debug_assert!(cur.1.is_none());
                        face_elems[f] = (cur.0, Some(e));
                        f
                    }
                    None => {
                        let f = faces.len();
                        face_map.insert(key, f);
                        faces.push(cyc.clone());
                        face_elems.push((e, None));
                        f
                    }
                };
                elem_faces[e as usize].push(fid);
                elem_face_ori[e as usize].push(cycle_orientation(&faces[fid], &cyc));
            }
        }

        // ── per-entity DOF counts ───────────────────────────────────────────
        let p = order as usize;
        let (n_vdof, n_edof) = match kind {
            TraceKind::H1 => (1usize, p.saturating_sub(1)),
            TraceKind::Rt => (0usize, 0usize),
            TraceKind::Nd => (0usize, p),
        };
        let face_nf: Vec<usize> = faces
            .iter()
            .map(|v| {
                let is_quad = v.len() == 4;
                match kind {
                    TraceKind::H1 => {
                        let q = p.saturating_sub(1);
                        if is_quad {
                            q * q
                        } else {
                            q.saturating_sub(1) * q / 2
                        }
                    }
                    TraceKind::Rt => {
                        if is_quad {
                            (p + 1) * (p + 1)
                        } else {
                            (p + 1) * (p + 2) / 2
                        }
                    }
                    TraceKind::Nd => nd_face_interior_dofs(p, is_quad),
                }
            })
            .collect();

        // ── face-local edge lists (reference face edge cycle, oriented) ─────
        let face_edge_list: Vec<Vec<(usize, i32)>> = faces
            .iter()
            .map(|v| {
                let is_quad = v.len() == 4;
                mfem_face_edges(is_quad)
                    .iter()
                    .map(|ev| {
                        let a = v[ev[0]];
                        let b = v[ev[1]];
                        let key = if a < b { (a, b) } else { (b, a) };
                        let eid = *edge_map
                            .get(&key)
                            .expect("TraceSpace: face edge missing from edge table");
                        (eid, if a < b { 1 } else { -1 })
                    })
                    .collect()
            })
            .collect();

        // ── global DOF numbering (MFEM Base layout) ─────────────────────────
        let n_dofs_v = mesh.n_nodes() * n_vdof;
        let n_dofs_e = edges.len() * n_edof;
        let mut face_fbase = Vec::with_capacity(faces.len());
        let mut acc = n_dofs_v + n_dofs_e;
        for f in 0..faces.len() {
            face_fbase.push(acc);
            acc += face_nf[f];
        }
        let n_dofs = acc;

        // ── per-face DOF lists ──────────────────────────────────────────────
        let mut face_dof_list: Vec<Vec<usize>> = Vec::with_capacity(faces.len());
        let mut face_signed: Vec<Vec<i32>> = Vec::with_capacity(faces.len());
        for f in 0..faces.len() {
            let mut dofs: Vec<usize> = Vec::new();
            let mut signed: Vec<i32> = Vec::new();
            if n_vdof > 0 {
                for &v in &faces[f] {
                    for j in 0..n_vdof {
                        dofs.push(v as usize * n_vdof + j);
                        signed.push((v as usize * n_vdof + j) as i32);
                    }
                }
            }
            if n_edof > 0 {
                for &(eid, ori) in &face_edge_list[f] {
                    let base = n_dofs_v + eid * n_edof;
                    for j in 0..n_edof {
                        if ori > 0 {
                            dofs.push(base + j);
                            signed.push(encode_dof(base, j as i32));
                        } else {
                            // Face-local edge reversed w.r.t. the global edge
                            // direction: MFEM's `DofOrderForOrientation`
                            // (`SegDofOrd[1][j] = -1 - (n-1-j)` for ND, `n-1-j`
                            // for H1).
                            let idx = n_edof - 1 - j;
                            dofs.push(base + idx);
                            signed.push(if kind == TraceKind::Nd {
                                encode_dof(base, -1 - (n_edof - 1 - j) as i32)
                            } else {
                                encode_dof(base, idx as i32)
                            });
                        }
                    }
                }
            }
            for j in 0..face_nf[f] {
                dofs.push(face_fbase[f] + j);
                signed.push((face_fbase[f] + j) as i32);
            }
            let n_face_edges = if faces[f].len() == 4 { 4 } else { 3 };
            assert_eq!(
                dofs.len(),
                faces[f].len() * n_vdof + n_face_edges * n_edof + face_nf[f]
            );
            face_dof_list.push(dofs);
            face_signed.push(signed);
        }

        // ── element trace blocks (concatenated per-face DOF lists) ──────────
        let mut elem_trace_dof_list: Vec<Vec<usize>> = vec![Vec::new(); mesh.n_elements()];
        for e in 0..mesh.n_elements() as u32 {
            let mut trace_dofs: Vec<usize> = Vec::new();
            for &fid in &elem_faces[e as usize] {
                trace_dofs.extend_from_slice(&face_dof_list[fid]);
            }
            elem_trace_dof_list[e as usize] = trace_dofs;
        }

        TraceSpace {
            mesh,
            dim,
            order,
            kind,
            faces,
            face_edge_list,
            face_elems,
            elem_faces,
            elem_face_ori,
            edges,
            n_vdof,
            n_edof,
            face_nf,
            face_fbase,
            n_dofs,
            face_dof_list,
            face_signed,
            elem_trace_dof_list,
        }
    }

    /// Trace family.
    pub fn kind(&self) -> TraceKind {
        self.kind
    }

    /// Face-element order (`p`).
    pub fn order(&self) -> u8 {
        self.order
    }

    /// Mesh dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Total number of skeleton DOFs.
    pub fn n_dofs(&self) -> usize {
        self.n_dofs
    }

    /// Number of faces (2-D: edges).
    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }

    /// Number of mesh edges (2-D: equal to [`Self::n_faces`]).
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// DOFs per vertex / per edge / per face-interior.
    pub fn dofs_per_entity(&self) -> (usize, usize) {
        (self.n_vdof, self.n_edof)
    }

    /// Canonical vertex cycle of face `f` (generating element's local face).
    pub fn face_nodes(&self, f: usize) -> &[u32] {
        &self.faces[f]
    }

    /// Whether face `f` is a quadrilateral.
    pub fn is_quad_face(&self, f: usize) -> bool {
        self.faces[f].len() == 4
    }

    /// Face-local edges of face `f`: `(global edge id, orientation)`.
    pub fn face_edges(&self, f: usize) -> &[(usize, i32)] {
        &self.face_edge_list[f]
    }

    /// `(Elem1, Option<Elem2>)` of face `f`.
    pub fn face_elements(&self, f: usize) -> (u32, Option<u32>) {
        self.face_elems[f]
    }

    /// `true` when face `f` has a single adjacent element.
    pub fn is_boundary_face(&self, f: usize) -> bool {
        self.face_elems[f].1.is_none()
    }

    /// Interior DOFs of face `f`.
    pub fn face_interior_dofs(&self, f: usize) -> usize {
        self.face_nf[f]
    }

    /// DOFs of face `f` (unsigned, face-element order).
    pub fn face_dof_list(&self, f: usize) -> &[usize] {
        &self.face_dof_list[f]
    }

    /// DOFs of face `f`, MFEM sign-encoded (`GetFaceDofs`).
    pub fn face_signed_dofs(&self, f: usize) -> &[i32] {
        &self.face_signed[f]
    }

    /// Global face id of the element's `li`-th local face.
    pub fn elem_face_id(&self, e: u32, li: usize) -> usize {
        self.elem_faces[e as usize][li]
    }

    /// Orientation parity (±1) of the element's `li`-th local face against the
    /// canonical face cycle (the generating element is `+1`).
    pub fn elem_face_orientation(&self, e: u32, li: usize) -> i32 {
        self.elem_face_ori[e as usize][li]
    }

    /// Trace block of element `e`: the concatenation of its faces' DOF lists in
    /// local face order (unsigned) — MFEM's `DPGWeakForm::Assemble` trace
    /// layout (`GetFaceVDofs` over `GetElementFaces`).
    pub fn element_trace_dof_list(&self, e: u32) -> &[usize] {
        &self.elem_trace_dof_list[e as usize]
    }

    /// Sign-encoded trace block of element `e` (same layout), i.e. MFEM's
    /// concatenation of `GetFaceVDofs(face)` over the element's local faces.
    pub fn element_trace_signed_dofs(&self, e: u32) -> Vec<i32> {
        let mut out = Vec::new();
        for li in 0..self.elem_faces[e as usize].len() {
            let fid = self.elem_faces[e as usize][li];
            out.extend_from_slice(&self.face_signed[fid]);
        }
        out
    }

    /// Mesh reference.
    pub fn mesh(&self) -> &M {
        &self.mesh
    }
}

/// Orientation parity of a face's local vertex cycle against the canonical
/// cycle: `+1` for rotations (same directed cycle), `−1` for reflections.
fn cycle_orientation(canonical: &[u32], local: &[u32]) -> i32 {
    let n = canonical.len();
    debug_assert_eq!(n, local.len());
    if n == 2 {
        return if canonical[0] == local[0] { 1 } else { -1 };
    }
    let start = canonical
        .iter()
        .position(|&x| x == local[0])
        .unwrap_or_else(|| panic!("cycle_orientation: face node mismatch"));
    if (0..n).all(|k| canonical[(start + k) % n] == local[k]) {
        return 1;
    }
    if (0..n).all(|k| canonical[(start + n - k % n) % n] == local[k]) {
        return -1;
    }
    panic!("cycle_orientation: non-manifold face cycle {canonical:?} vs {local:?}");
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

/// 3-D trace-space (`*_Trace_FECollection`) port checks against hard data
/// dumped from MFEM 4.10 (`tmp/dpg3d/ndtrace_dump.txt`, produced by
/// `tmp/dpg3d/ndtrace_dump.cpp`).
#[cfg(test)]
mod trace_tests {
    use super::*;
    use fem_mesh::Mesh;

    /// MFEM `Mesh::MakeCartesian3D(1,1,1,HEXAHEDRON)` — one element on the
    /// unit cube, MFEM node/element numbering.
    fn mfem_hex_1x1x1() -> Mesh<3> {
        let mut coords = Vec::new();
        for k in 0..2 {
            for j in 0..2 {
                for i in 0..2 {
                    coords.push(i as f64);
                    coords.push(j as f64);
                    coords.push(k as f64);
                }
            }
        }
        let conn: Vec<u32> = vec![0, 1, 3, 2, 4, 5, 7, 6];
        let bfaces: [[u32; 4]; 6] = [
            [3, 2, 1, 0],
            [0, 1, 5, 4],
            [1, 3, 7, 5],
            [3, 2, 6, 7],
            [2, 0, 4, 6],
            [4, 5, 7, 6],
        ];
        let mut face_conn = Vec::new();
        let mut face_tags = Vec::new();
        for (i, f) in bfaces.iter().enumerate() {
            face_conn.extend_from_slice(f);
            face_tags.push(i as i32 + 1);
        }
        Mesh::<3>::uniform(
            coords,
            conn,
            vec![1],
            ElementType::Hex8,
            face_conn,
            face_tags,
            ElementType::Quad4,
        )
    }

    /// MFEM `Mesh::MakeCartesian3D(2,1,1,HEXAHEDRON)`: identical node and
    /// element numbering (node id = i + 3j + 6k, hex vertex order
    /// `(i,j,k),(i+1,j,k),(i+1,j+1,k),(i,j+1,k),+k`).
    fn mfem_hex_2x1x1() -> Mesh<3> {
        let mut coords = Vec::new();
        for k in 0..2 {
            for j in 0..2 {
                for i in 0..3 {
                    coords.push(i as f64 * 0.5);
                    coords.push(j as f64);
                    coords.push(k as f64);
                }
            }
        }
        let conn: Vec<u32> = vec![
            0, 1, 4, 3, 6, 7, 10, 9, //
            1, 2, 5, 4, 7, 8, 11, 10,
        ];
        // Boundary faces (MFEM's stored cycles, single-element faces).
        let bfaces: [[u32; 4]; 10] = [
            [3, 4, 1, 0],
            [0, 1, 7, 6],
            [4, 3, 9, 10],
            [3, 0, 6, 9],
            [6, 7, 10, 9],
            [4, 5, 2, 1],
            [1, 2, 8, 7],
            [2, 5, 11, 8],
            [5, 4, 10, 11],
            [7, 8, 11, 10],
        ];
        let mut face_conn = Vec::new();
        let mut face_tags = Vec::new();
        for (i, f) in bfaces.iter().enumerate() {
            face_conn.extend_from_slice(f);
            face_tags.push(i as i32 + 1);
        }
        Mesh::<3>::uniform(
            coords,
            conn,
            vec![1, 1],
            ElementType::Hex8,
            face_conn,
            face_tags,
            ElementType::Quad4,
        )
    }

    /// MFEM `Mesh::MakeCartesian3D(1,1,1,TETRAHEDRON)`: 8 nodes, 6 tets split
    /// around the main diagonal `0–7` (Kuhn/Freudenthal, MFEM's ordering).
    fn mfem_tet_1x1x1() -> Mesh<3> {
        let mut coords = Vec::new();
        for k in 0..2 {
            for j in 0..2 {
                for i in 0..2 {
                    coords.push(i as f64);
                    coords.push(j as f64);
                    coords.push(k as f64);
                }
            }
        }
        let conn: Vec<u32> = vec![
            7, 0, 3, 1, //
            7, 0, 1, 5, //
            7, 0, 5, 4, //
            7, 0, 2, 3, //
            7, 0, 6, 2, //
            7, 0, 4, 6,
        ];
        // MFEM's stored boundary triangle cycles (all faces are boundary).
        let bfaces: [[u32; 3]; 18] = [
            [0, 3, 1],
            [7, 1, 3],
            [7, 0, 1],
            [7, 3, 0],
            [0, 1, 5],
            [7, 5, 1],
            [7, 0, 5],
            [0, 5, 4],
            [7, 4, 5],
            [7, 0, 4],
            [0, 2, 3],
            [7, 3, 2],
            [7, 2, 0],
            [0, 6, 2],
            [7, 2, 6],
            [7, 6, 0],
            [0, 4, 6],
            [7, 6, 4],
        ];
        let mut face_conn = Vec::new();
        let mut face_tags = Vec::new();
        for (i, f) in bfaces.iter().enumerate() {
            face_conn.extend_from_slice(f);
            face_tags.push(i as i32 + 1);
        }
        Mesh::<3>::uniform(
            coords,
            conn,
            vec![1; 6],
            ElementType::Tet4,
            face_conn,
            face_tags,
            ElementType::Tri3,
        )
    }

    /// DOF counts (`ndofs`) dumped from MFEM for the two reference meshes.
    #[test]
    fn trace_dof_counts_match_cpp() {
        let hex = mfem_hex_2x1x1();
        let tet = mfem_tet_1x1x1();
        // (kind, order) -> hex ndofs / tet ndofs (from ndtrace_dump.txt)
        let expected: &[(TraceKind, u8, usize, usize)] = &[
            (TraceKind::Nd, 1, 20, 19),
            (TraceKind::Nd, 2, 84, 74),
            (TraceKind::Nd, 3, 192, 165),
            (TraceKind::H1, 1, 12, 8),
            (TraceKind::H1, 2, 43, 27),
            (TraceKind::H1, 3, 96, 64),
            (TraceKind::Rt, 1, 44, 54),
            (TraceKind::Rt, 2, 99, 108),
            (TraceKind::Rt, 3, 176, 180),
        ];
        for &(kind, p, n_hex, n_tet) in expected {
            let sh = match kind {
                TraceKind::Nd => TraceSpace::new_nd(hex.clone(), p),
                TraceKind::H1 => TraceSpace::new_h1(hex.clone(), p),
                TraceKind::Rt => TraceSpace::new_rt(hex.clone(), p),
            };
            let st = match kind {
                TraceKind::Nd => TraceSpace::new_nd(tet.clone(), p),
                TraceKind::H1 => TraceSpace::new_h1(tet.clone(), p),
                TraceKind::Rt => TraceSpace::new_rt(tet.clone(), p),
            };
            assert_eq!(sh.n_dofs(), n_hex, "{kind:?} p{p} hex ndofs");
            assert_eq!(st.n_dofs(), n_tet, "{kind:?} p{p} tet ndofs");
            assert_eq!(sh.n_faces(), 11);
            assert_eq!(sh.n_edges(), 20);
            assert_eq!(st.n_faces(), 18);
            assert_eq!(st.n_edges(), 19);
        }
    }

    /// The vertex-continuous 3-D H1 trace skeleton
    /// ([`SkeletonSpace::new_h1`]) must have MFEM's
    /// `H1_Trace_FECollection(p, 3)` = `H1_FECollection(p, 2)` dof count
    /// (1/vertex + `p−1`/edge + face-interior) and — crucially — a *consistent*
    /// face-dof association: two faces that share a vertex/edge node must
    /// reference the same global dof at the same physical point.
    #[test]
    fn skeleton_3d_h1_trace_counts_and_sharing() {
        for (mesh, nv, ne, nf, counts) in [
            (mfem_hex_2x1x1(), 12, 20, 11, [12usize, 43, 96]),
            (mfem_tet_1x1x1(), 8, 19, 18, [8, 27, 64]),
        ] {
            assert_eq!(mesh.n_nodes(), nv);
            for (i, p) in (1u8..=3).enumerate() {
                let sk = SkeletonSpace::new_h1(mesh.clone(), p);
                assert_eq!(sk.n_dofs(), counts[i], "H1 trace p={p} ndofs");
                assert_eq!(sk.n_faces(), nf);
                // Map every global dof to the physical point(s) it is used at.
                let mut owner: std::collections::HashMap<usize, ([f64; 3], usize)> =
                    std::collections::HashMap::new();
                for f in 0..sk.n_faces() {
                    let is_qf = sk.is_quad_face(f);
                    let nfd = sk.dofs_per_face(f);
                    let mut phi = vec![0.0; nfd];
                    for (k, &d) in sk.face_dof_list(f).iter().enumerate() {
                        // The basis must be a nodal Kronecker delta at the dof
                        // nodes: evaluate at node k and require phi = e_k.
                        let node =
                            crate::dpg_weakform::face_dof_params(3, is_qf, p as usize, k);
                        eval_face_lagrange(3, is_qf, p as usize, &node, &mut phi);
                        for (j, &v) in phi.iter().enumerate() {
                            let want = if j == k { 1.0 } else { 0.0 };
                            assert!(
                                (v - want).abs() < 1e-12,
                                "p={p} face {f}: basis not nodal at index {k}"
                            );
                        }
                        let (xp, _, _) = face_point_3d_local(&sk, f, &node);
                        match owner.get(&d) {
                            None => {
                                owner.insert(d, (xp, k));
                            }
                            Some(&(prev, _)) => {
                                let dist: f64 = (0..3).map(|c| (prev[c] - xp[c]).abs()).sum();
                                assert!(
                                    dist < 1e-12,
                                    "p={p}: shared dof {d} sits at two different points \
                                     ({prev:?} vs {xp:?})"
                                );
                            }
                        }
                        if d < mesh.n_nodes() {
                            let c = mesh.node_coords(d as u32);
                            let dist: f64 = (0..3).map(|c2| (c[c2] - xp[c2]).abs()).sum();
                            assert!(
                                dist < 1e-12,
                                "p={p}: corner dof {d} must be the mesh vertex at {xp:?}"
                            );
                        }
                    }
                }
                assert_eq!(owner.len(), sk.n_dofs(), "every dof must be used");
            }
            let _ = (ne, nf);
        }
    }

    /// Physical point of the canonical face parametrisation (local helper;
    /// `face_point_3d` takes the `TraceSpace`, this takes the skeleton).
    fn face_point_3d_local<M: MeshTopology + Clone>(
        sk: &SkeletonSpace<M>,
        f: usize,
        param: &[f64],
    ) -> ([f64; 3], f64, f64) {
        let c: Vec<[f64; 3]> = sk
            .face_nodes(f)
            .iter()
            .map(|&n| {
                let p = sk.mesh().node_coords(n);
                [p[0], p[1], p[2]]
            })
            .collect();
        let (s, t) = (param[0], param[1]);
        let mut x = [0.0; 3];
        if sk.is_quad_face(f) {
            for d in 0..3 {
                x[d] = (1.0 - s) * (1.0 - t) * c[0][d]
                    + s * (1.0 - t) * c[1][d]
                    + s * t * c[2][d]
                    + (1.0 - s) * t * c[3][d];
            }
        } else {
            for d in 0..3 {
                x[d] = (1.0 - s - t) * c[0][d] + s * c[1][d] + t * c[2][d];
            }
        }
        (x, 0.0, 0.0)
    }

    /// The node-index face table used by the trace assemblers
    /// ([`local_face_table`]) must be MFEM's `FaceVert` verbatim, i.e. the same
    /// cycles as the static [`mfem_local_faces`] table — and each cycle must
    /// have the outward winding (`J_s × J_t` points out of the reference
    /// element), which is exactly what makes the ±1 element-face orientation
    /// in the trace assembly equal to the sign relating the canonical face
    /// normal to the element's outward normal.
    #[test]
    fn local_face_table_matches_mfem_facevert_and_is_outward() {
        for (et, npe) in [
            (ElementType::Tri3, 3usize),
            (ElementType::Quad4, 4),
            (ElementType::Tet4, 4),
            (ElementType::Hex8, 8),
        ] {
            let dim = if npe == 3 || et == ElementType::Quad4 { 2 } else { 3 };
            let dyn_tab = local_face_table(&vec![0u32; npe], dim);
            let static_tab = mfem_local_faces(et, dim);
            assert_eq!(dyn_tab.len(), static_tab.len(), "{et:?}");
            for (i, (a, b)) in dyn_tab.iter().zip(static_tab.iter()).enumerate() {
                assert_eq!(a.as_slice(), *b, "{et:?} local face {i}");
            }
            // Outward winding on the reference element: the face centroid
            // direction relative to the reference centroid must agree with
            // `J_s × J_t` of the cycle (2-D: the (dy, −dx) edge normal).
            let rn = ref_node_coords(et);
            let n_vert = if dim == 2 { 3 } else { rn.len() };
            let cen = |idxs: &[usize]| -> Vec<f64> {
                (0..3)
                    .map(|d| idxs.iter().map(|&k| rn[k][d]).sum::<f64>() / idxs.len() as f64)
                    .collect()
            };
            let all: Vec<usize> = (0..n_vert).collect();
            let center = cen(&all);
            for cyc in &dyn_tab {
                let fcen = cen(cyc);
                if dim == 2 {
                    let a = rn[cyc[0]];
                    let b = rn[cyc[1]];
                    let n = [b[1] - a[1], -(b[0] - a[0])];
                    let dot = n[1] * (fcen[1] - center[1]) + n[0] * (fcen[0] - center[0]);
                    assert!(dot > 0.0, "{et:?} face {cyc:?}: 2-D edge normal not outward");
                } else {
                    let a = rn[cyc[0]];
                    let c1 = rn[cyc[1]];
                    let c2 = rn[cyc[2]];
                    let s = [c1[0] - a[0], c1[1] - a[1], c1[2] - a[2]];
                    let t = [c2[0] - a[0], c2[1] - a[1], c2[2] - a[2]];
                    let n = [
                        s[1] * t[2] - s[2] * t[1],
                        s[2] * t[0] - s[0] * t[2],
                        s[0] * t[1] - s[1] * t[0],
                    ];
                    let dot: f64 =
                        (0..3).map(|d| n[d] * (fcen[d] - center[d])).sum();
                    assert!(dot > 0.0, "{et:?} face {cyc:?}: 3-D face normal not outward");
                }
            }
        }
    }

    /// Per-face entity tables (canonical vertex cycles, face-local edges with
    /// orientations) against MFEM's `GetFaceVertices` / `GetFaceEdges`.
    #[test]
    fn trace_face_tables_match_cpp() {
        let hex = mfem_hex_2x1x1();
        let tr = TraceSpace::new_nd(hex, 1);
        let want_verts: [[u32; 4]; 11] = [
            [3, 4, 1, 0],
            [0, 1, 7, 6],
            [1, 4, 10, 7],
            [4, 3, 9, 10],
            [3, 0, 6, 9],
            [6, 7, 10, 9],
            [4, 5, 2, 1],
            [1, 2, 8, 7],
            [2, 5, 11, 8],
            [5, 4, 10, 11],
            [7, 8, 11, 10],
        ];
        let want_edges: [[(usize, i32); 4]; 11] = [
            [(2, 1), (1, -1), (0, -1), (3, 1)],
            [(0, 1), (9, 1), (4, -1), (8, -1)],
            [(1, 1), (10, 1), (5, -1), (9, -1)],
            [(2, -1), (11, 1), (6, 1), (10, -1)],
            [(3, -1), (8, 1), (7, 1), (11, -1)],
            [(4, 1), (5, 1), (6, -1), (7, -1)],
            [(14, 1), (13, -1), (12, -1), (1, 1)],
            [(12, 1), (18, 1), (15, -1), (9, -1)],
            [(13, 1), (19, 1), (16, -1), (18, -1)],
            [(14, -1), (10, 1), (17, 1), (19, -1)],
            [(15, 1), (16, 1), (17, -1), (5, -1)],
        ];
        for f in 0..11 {
            assert_eq!(tr.face_nodes(f), &want_verts[f], "face {f} verts");
            assert_eq!(tr.face_edges(f), &want_edges[f], "face {f} edges");
        }
    }

    fn orthonormal_basis(cols: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut out: Vec<Vec<f64>> = Vec::new();
        for c in cols {
            let mut w = c.clone();
            let c0: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            for _sweep in 0..2 {
                for b in out.iter() {
                    let d: f64 = w.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                    for (k, bk) in b.iter().enumerate() {
                        w[k] -= d * bk;
                    }
                }
            }
            let rn: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if rn > 1e-11 * c0.max(1e-30) {
                for x in w.iter_mut() {
                    *x /= rn;
                }
                out.push(w);
            }
        }
        out
    }

    /// The fem-rs `HexNDk(p)` element (used as the DPG 3-D H(curl) test space
    /// via [`VolKind::HCurl`]) must span exactly MFEM's `ND_HexahedronElement`
    /// space, the tensor Nédélec space
    ///
    /// ```text
    ///     Q_{p−1,p,p} × Q_{p,p−1,p} × Q_{p,p,p−1}
    /// ```
    ///
    /// (MFEM builds that element as a tensor product of the *closed*
    /// (Gauss-Lobatto) 1-D basis of degree `p` in the two transversal
    /// directions and the *open* (Gauss-Legendre) basis of degree `p−1` in the
    /// tangential one; both 1-D bases span the full 1-D polynomial space of
    /// their degree).  fem-rs's element uses a different node set and the
    /// `[−1,1]³` parametrisation, which is a basis (not a space) difference —
    /// this test pins that so the DPG element normal equations, which are
    /// invariant under a test-basis change, stay comparable with MFEM's.
    #[test]
    fn nd_hex_span_is_tensor_nedelec() {
        let ident = nalgebra::DMatrix::<f64>::identity(3, 3);
        let quad = vol_quadrature(ElementType::Hex8, 4);
        for p in 1u8..=3 {
            let pu = p as usize;
            let n = hcurl_ref_elem(ElementType::Hex8, p).n_dofs();
            assert_eq!(n, 3 * pu * (pu + 1) * (pu + 1));
            // Monomial columns: (component, a, b, c) with the tensor Nédélec
            // degree bounds; sample rows are (quadrature point, component).
            let mut mono: Vec<Vec<f64>> = Vec::new();
            for d in 0..3 {
                let (ea, eb, ec) = match d {
                    0 => (pu - 1, pu, pu),
                    1 => (pu, pu - 1, pu),
                    _ => (pu, pu, pu - 1),
                };
                for a in 0..=ea {
                    for b in 0..=eb {
                        for c in 0..=ec {
                            let mut col = vec![0.0_f64; 3 * quad.0.len()];
                            for (q, pts) in quad.0.iter().enumerate() {
                                col[3 * q + d] =
                                    pts[0].powi(a as i32) * pts[1].powi(b as i32) * pts[2].powi(c as i32);
                            }
                            mono.push(col);
                        }
                    }
                }
            }
            let mono_dim = 3 * pu * (pu + 1) * (pu + 1);
            assert_eq!(mono.len(), mono_dim);
            // Orthonormal basis of the monomial span.
            let ob = orthonormal_basis(&mono);
            assert_eq!(ob.len(), mono_dim, "p{p}: monomial basis must be independent");
            // Every fem-rs basis sample must lie in that span, and the samples
            // must be independent (=> the spans are equal).
            let npts = quad.0.len();
            let mut samples: Vec<Vec<f64>> = Vec::new();
            let mut vals = VolVals::default();
            for i in 0..n {
                let mut row = vec![0.0_f64; 3 * npts];
                for (q, pts) in quad.0.iter().enumerate() {
                    eval_vol_space(
                        VolKind::HCurl,
                        p,
                        ElementType::Hex8,
                        3,
                        &ident,
                        1.0,
                        &ident,
                        pts,
                        None,
                        &mut vals,
                    );
                    for d in 0..3 {
                        row[3 * q + d] = vals.phi[i * 3 + d];
                    }
                }
                // residual after projection onto the monomial span
                let mut w = row.clone();
                for b in ob.iter() {
                    let dp: f64 = w.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                    for (k, bk) in b.iter().enumerate() {
                        w[k] -= dp * bk;
                    }
                }
                let rn: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
                let r0: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
                assert!(
                    rn <= 1e-10 * r0.max(1e-30),
                    "p{p}: HexNDk dof {i} sample outside the tensor Nedelec space ({rn} vs {r0})"
                );
                samples.push(row);
            }
            assert_eq!(orthonormal_basis(&samples).len(), n, "p{p}: HexNDk samples rank");
        }
    }

    /// The fem-rs `HexRTk(p)` element (used as the DPG 3-D H(div) test space
    /// via [`VolKind::HDiv`]) must span MFEM's `RT_HexahedronElement` space,
    /// the tensor Raviart–Thomas space
    ///
    /// ```text
    ///     Q_{p+1,p,p} × Q_{p,p+1,p} × Q_{p,p,p+1}
    /// ```
    ///
    /// (MFEM builds it from a *closed* GLL factor of degree `p+1` in the
    /// normal direction and *open* Gauss–Legendre factors of degree `p` in the
    /// two tangential ones, all on `[0,1]`; `HexRTk` uses the `[−1,1]` pullback
    /// and different node sets).  Ranking the fem-rs samples against the
    /// monomial span is what decides whether the `[−1,1]` vs `[0,1]`
    /// parametrisation is only a *basis* difference (span preserved) or a
    /// genuine *space* mismatch — the latter would silently break the 3-D DPG
    /// test norm.
    #[test]
    fn rt_hex_span_is_tensor_rt() {
        let ident = nalgebra::DMatrix::<f64>::identity(3, 3);
        let quad = vol_quadrature(ElementType::Hex8, 4);
        for p in 1u8..=2 {
            let pu = p as usize;
            let n = vector_ref_elem(ElementType::Hex8, p).n_dofs();
            assert_eq!(n, 3 * (pu + 1) * (pu + 1) * (pu + 2));
            let mut mono: Vec<Vec<f64>> = Vec::new();
            for d in 0..3 {
                let (ea, eb, ec) = match d {
                    0 => (pu + 1, pu, pu),
                    1 => (pu, pu + 1, pu),
                    _ => (pu, pu, pu + 1),
                };
                for a in 0..=ea {
                    for b in 0..=eb {
                        for c in 0..=ec {
                            let mut col = vec![0.0_f64; 3 * quad.0.len()];
                            for (q, pts) in quad.0.iter().enumerate() {
                                col[3 * q + d] = pts[0].powi(a as i32)
                                    * pts[1].powi(b as i32)
                                    * pts[2].powi(c as i32);
                            }
                            mono.push(col);
                        }
                    }
                }
            }
            let mono_dim = 3 * (pu + 1) * (pu + 1) * (pu + 2);
            assert_eq!(mono.len(), mono_dim);
            let ob = orthonormal_basis(&mono);
            assert_eq!(ob.len(), mono_dim, "p{p}: monomial basis must be independent");
            let npts = quad.0.len();
            let mut samples: Vec<Vec<f64>> = Vec::new();
            let mut vals = VolVals::default();
            for i in 0..n {
                let mut row = vec![0.0_f64; 3 * npts];
                for (q, pts) in quad.0.iter().enumerate() {
                    eval_vol_space(
                        VolKind::HDiv,
                        p,
                        ElementType::Hex8,
                        3,
                        &ident,
                        1.0,
                        &ident,
                        pts,
                        None,
                        &mut vals,
                    );
                    for d in 0..3 {
                        row[3 * q + d] = vals.phi[i * 3 + d];
                    }
                }
                let mut w = row.clone();
                for b in ob.iter() {
                    let dp: f64 = w.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                    for (k, bk) in b.iter().enumerate() {
                        w[k] -= dp * bk;
                    }
                }
                let rn: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
                let r0: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
                assert!(
                    rn <= 1e-10 * r0.max(1e-30),
                    "p{p}: HexRTk dof {i} sample outside the tensor RT space ({rn} vs {r0})"
                );
                samples.push(row);
            }
            assert_eq!(orthonormal_basis(&samples).len(), n, "p{p}: HexRTk samples rank");
        }
    }

    fn trace_face_dofs_match_cpp_hex() {
        let hex = mfem_hex_2x1x1();
        let cases: &[(TraceKind, u8, &[&[i32]])] = &[
            (
                TraceKind::Nd,
                1,
                &[
                    &[2, -2, -1, 3],
                    &[0, 9, -5, -9],
                    &[1, 10, -6, -10],
                    &[-3, 11, 6, -11],
                    &[-4, 8, 7, -12],
                    &[4, 5, -7, -8],
                    &[14, -14, -13, 1],
                    &[12, 18, -16, -10],
                    &[13, 19, -17, -19],
                    &[-15, 10, 17, -20],
                    &[15, 16, -18, -6],
                ],
            ),
            (
                TraceKind::Nd,
                2,
                &[
                    &[4, 5, -4, -3, -2, -1, 6, 7, 40, 41, 42, 43],
                    &[0, 1, 18, 19, -10, -9, -18, -17, 44, 45, 46, 47],
                    &[2, 3, 20, 21, -12, -11, -20, -19, 48, 49, 50, 51],
                    &[-6, -5, 22, 23, 12, 13, -22, -21, 52, 53, 54, 55],
                    &[-8, -7, 16, 17, 14, 15, -24, -23, 56, 57, 58, 59],
                    &[8, 9, 10, 11, -14, -13, -16, -15, 60, 61, 62, 63],
                    &[28, 29, -28, -27, -26, -25, 2, 3, 64, 65, 66, 67],
                    &[24, 25, 36, 37, -32, -31, -20, -19, 68, 69, 70, 71],
                    &[26, 27, 38, 39, -34, -33, -38, -37, 72, 73, 74, 75],
                    &[-30, -29, 20, 21, 34, 35, -40, -39, 76, 77, 78, 79],
                    &[30, 31, 32, 33, -36, -35, -12, -11, 80, 81, 82, 83],
                ],
            ),
            (
                TraceKind::H1,
                1,
                &[
                    &[3, 4, 1, 0],
                    &[0, 1, 7, 6],
                    &[1, 4, 10, 7],
                    &[4, 3, 9, 10],
                    &[3, 0, 6, 9],
                    &[6, 7, 10, 9],
                    &[4, 5, 2, 1],
                    &[1, 2, 8, 7],
                    &[2, 5, 11, 8],
                    &[5, 4, 10, 11],
                    &[7, 8, 11, 10],
                ],
            ),
            (
                TraceKind::Rt,
                2,
                &[
                    &[0, 1, 2, 3, 4, 5, 6, 7, 8],
                    &[9, 10, 11, 12, 13, 14, 15, 16, 17],
                    &[18, 19, 20, 21, 22, 23, 24, 25, 26],
                    &[27, 28, 29, 30, 31, 32, 33, 34, 35],
                    &[36, 37, 38, 39, 40, 41, 42, 43, 44],
                    &[45, 46, 47, 48, 49, 50, 51, 52, 53],
                    &[54, 55, 56, 57, 58, 59, 60, 61, 62],
                    &[63, 64, 65, 66, 67, 68, 69, 70, 71],
                    &[72, 73, 74, 75, 76, 77, 78, 79, 80],
                    &[81, 82, 83, 84, 85, 86, 87, 88, 89],
                    &[90, 91, 92, 93, 94, 95, 96, 97, 98],
                ],
            ),
        ];
        for (kind, p, want) in cases {
            let tr = match kind {
                TraceKind::Nd => TraceSpace::new_nd(hex.clone(), *p),
                TraceKind::H1 => TraceSpace::new_h1(hex.clone(), *p),
                TraceKind::Rt => TraceSpace::new_rt(hex.clone(), *p),
            };
            for (f, w) in want.iter().enumerate() {
                assert_eq!(tr.face_signed_dofs(f), *w, "{kind:?} p{p} face {f}");
                assert_eq!(tr.face_dof_list(f).len(), w.len());
            }
        }
    }

    /// MFEM `GetFaceDofs` for the tet mesh, all three families / orders.
    #[test]
    fn trace_face_dofs_match_cpp_tet() {
        let tet = mfem_tet_1x1x1();
        // ND_Trace p=2 (from ndtrace_dump.txt lines 3547ff)
        let tr = TraceSpace::new_nd(tet.clone(), 2);
        let want: [&[i32]; 18] = [
            &[6, 7, -12, -11, -10, -9, 38, 39],
            &[-6, -5, 10, 11, 2, 3, 40, 41],
            &[-2, -1, 8, 9, 4, 5, 42, 43],
            &[-4, -3, -8, -7, 0, 1, 44, 45],
            &[8, 9, 16, 17, -16, -15, 46, 47],
            &[-14, -13, -18, -17, 4, 5, 48, 49],
            &[-2, -1, 14, 15, 12, 13, 50, 51],
            &[14, 15, -24, -23, -22, -21, 52, 53],
            &[-20, -19, 22, 23, 12, 13, 54, 55],
            &[-2, -1, 20, 21, 18, 19, 56, 57],
            &[26, 27, 28, 29, -8, -7, 58, 59],
            &[-4, -3, -30, -29, 24, 25, 60, 61],
            &[-26, -25, -28, -27, 0, 1, 62, 63],
            &[32, 33, -36, -35, -28, -27, 64, 65],
            &[-26, -25, 34, 35, 30, 31, 66, 67],
            &[-32, -31, -34, -33, 0, 1, 68, 69],
            &[20, 21, 36, 37, -34, -33, 70, 71],
            &[-32, -31, -38, -37, 18, 19, 72, 73],
        ];
        for (f, w) in want.iter().enumerate() {
            assert_eq!(tr.face_signed_dofs(f), *w, "ND p2 tet face {f}");
        }
        // H1_Trace p=1: one DOF per mesh vertex, in the face's vertex order.
        let h1 = TraceSpace::new_h1(tet, 1);
        assert_eq!(h1.n_dofs(), 8);
        assert_eq!(h1.face_signed_dofs(0), &[0, 3, 1]);
        assert_eq!(h1.face_signed_dofs(1), &[7, 1, 3]);
    }

    /// MFEM `ND_TriangleElement` / `ND_QuadrilateralElement` reference basis
    /// values, verbatim from the harness dump (hex face 0, quad; tet face 0,
    /// triangle).
    #[test]
    fn nd_face_basis_values_match_cpp() {
        // Quad, p = 2, at (0.25, 0.25) — hex mesh face 0.
        let want_quad_p2: [[f64; 2]; 12] = [
            [0.34987976320958225, 0.0],
            [0.025120236790417763, 0.0],
            [0.0, -0.11662658773652743],
            [0.0, -0.0083734122634725877],
            [0.0083734122634725877, 0.0],
            [0.11662658773652743, 0.0],
            [0.0, -0.025120236790417763],
            [0.0, -0.34987976320958225],
            [0.6997595264191645, 0.0],
            [0.050240473580835526, 0.0],
            [0.0, 0.6997595264191645],
            [0.0, 0.050240473580835526],
        ];
        let mut out = vec![0.0; 24];
        nd_quad_basis(2, 0.25, 0.25, &mut out);
        for (k, w) in want_quad_p2.iter().enumerate() {
            assert!(
                (out[k * 2] - w[0]).abs() < 1e-14,
                "quad p2 dof {k} x: {} vs {}",
                out[k * 2],
                w[0]
            );
            assert!(
                (out[k * 2 + 1] - w[1]).abs() < 1e-14,
                "quad p2 dof {k} y: {} vs {}",
                out[k * 2 + 1],
                w[1]
            );
        }
        // Quad, p = 2, at (0.3, 0.7).
        let want_quad_p2_b: [[f64; 2]; 12] = [
            [-0.10156921938165304, 0.0],
            [-0.018430780618346947, 0.0],
            [0.0, -0.018430780618346954],
            [0.0, -0.10156921938165306],
            [-0.043005154776142869, 0.0],
            [-0.23699484522385705, 0.0],
            [0.0, -0.23699484522385716],
            [0.0, -0.04300515477614289],
            [0.71098453567157149, 0.0],
            [0.12901546432842864, 0.0],
            [0.0, 0.12901546432842864],
            [0.0, 0.71098453567157138],
        ];
        nd_quad_basis(2, 0.3, 0.7, &mut out);
        for (k, w) in want_quad_p2_b.iter().enumerate() {
            assert!(
                (out[k * 2] - w[0]).abs() < 1e-14,
                "quad p2 (0.3,0.7) dof {k} x: {} vs {}",
                out[k * 2],
                w[0]
            );
            assert!(
                (out[k * 2 + 1] - w[1]).abs() < 1e-14,
                "quad p2 (0.3,0.7) dof {k} y: {} vs {}",
                out[k * 2 + 1],
                w[1]
            );
        }
        // Triangle, p = 2, at (0.2, 0.3) — tet mesh face 0.
        let want_tri_p2: [f64; 16] = [
            0.21686533479473216,
            0.061961524227066286,
            -0.14686533479473216,
            -0.041961524227066331,
            0.10098076211353299,
            -0.067320508075688734,
            0.049019237886466727,
            -0.03267949192431116,
            -0.0080384757729336856,
            -0.021435935394489822,
            -0.11196152422706616,
            -0.2985640646055101,
            1.0799999999999998,
            -0.1199999999999999,
            -0.26999999999999963,
            0.78000000000000003,
        ];
        let mut out = vec![0.0; 16];
        nd_tri_basis(2, 0.2, 0.3, &mut out);
        for (k, &w) in want_tri_p2.iter().enumerate() {
            assert!((out[k] - w).abs() < 1e-14, "tri p2 entry {k}: {} vs {w}", out[k]);
        }
        // Nodal property of the reference basis: the DOF functionals
        // `u ↦ ⟨u(node_k), tk_k⟩` reproduce the Kronecker delta, i.e.
        // `⟨φ_m(node_k), tk_k⟩ = δ_mk`.  `tk_k` is MFEM's direction vector for
        // DOF `k` (`tk = {1,0, −1,1, 0,−1, 0,1}` for triangles, the
        // sign-encoded edge/interior components for quads).
        for (p, is_quad) in [(1usize, true), (2, true), (3, true), (1, false), (2, false), (3, false)] {
            let nodes = nd_face_dof_nodes(p, is_quad);
            let nd = nodes.len();
            let tk: Vec<[f64; 2]> = if is_quad {
                let mut t = Vec::new();
                t.extend(std::iter::repeat([1.0, 0.0]).take(p));
                t.extend(std::iter::repeat([0.0, 1.0]).take(p));
                t.extend(std::iter::repeat([-1.0, 0.0]).take(p));
                t.extend(std::iter::repeat([0.0, -1.0]).take(p));
                for j in 1..p {
                    for _i in 0..p {
                        t.push([1.0, 0.0]);
                        let _ = j;
                    }
                }
                for _j in 0..p {
                    for _i in 1..p {
                        t.push([0.0, 1.0]);
                    }
                }
                t
            } else {
                let mut t = Vec::new();
                t.extend(std::iter::repeat([1.0, 0.0]).take(p));
                t.extend(std::iter::repeat([-1.0, 1.0]).take(p));
                t.extend(std::iter::repeat([0.0, -1.0]).take(p));
                if p >= 2 {
                    let pm2 = p - 2;
                    for j in 0..=pm2 {
                        for i in 0..=(pm2 - j) {
                            let _ = (i, j);
                            t.push([1.0, 0.0]);
                            t.push([0.0, 1.0]);
                        }
                    }
                }
                t
            };
            assert_eq!(tk.len(), nd);
            for k in 0..nd {
                let mut v = vec![0.0; nd * 2];
                eval_face_nd(p, is_quad, &[nodes[k][0], nodes[k][1]], &mut v);
                for m in 0..nd {
                    let dot = v[m * 2] * tk[k][0] + v[m * 2 + 1] * tk[k][1];
                    let want = if m == k { 1.0 } else { 0.0 };
                    assert!(
                        (dot - want).abs() < 1e-11,
                        "p{p} quad={is_quad} dof functional {k}(φ_{m}) = {dot}, want {want}"
                    );
                }
            }
        }
    }

    /// Physical (tangential covariant) map of the ND face basis against the
    /// harness dump: hex face 0 at (0.3, 0.7), p = 2.
    #[test]
    fn nd_face_physical_map_matches_cpp() {
        let hex = mfem_hex_2x1x1();
        let tr = TraceSpace::new_nd(hex, 2);
        let jac = face_jacobian_3d(&tr, 0, &[0.3, 0.7]);
        let mut ref_vals = vec![0.0; tr.face_dof_list(0).len() * 2];
        eval_face_nd(2, tr.is_quad_face(0), &[0.3, 0.7], &mut ref_vals);
        let want: [[f64; 3]; 12] = [
            [-0.20313843876330609, 0.0, 0.0],
            [-0.036861561236693895, 0.0, 0.0],
            [0.0, 0.018430780618346954, 0.0],
            [0.0, 0.10156921938165306, 0.0],
            [-0.086010309552285738, 0.0, 0.0],
            [-0.47398969044771411, 0.0, 0.0],
            [0.0, 0.23699484522385716, 0.0],
            [0.0, 0.04300515477614289, 0.0],
            [1.421969071343143, 0.0, 0.0],
            [0.25803092865685728, 0.0, 0.0],
            [0.0, -0.12901546432842864, 0.0],
            [0.0, -0.71098453567157138, 0.0],
        ];
        for (k, w) in want.iter().enumerate() {
            let p = map_face_nd_to_phys(&jac, &[ref_vals[k * 2], ref_vals[k * 2 + 1]]);
            for c in 0..3 {
                assert!(
                    (p[c] - w[c]).abs() < 1e-14,
                    "dof {k} phys[{c}]: {} vs {}",
                    p[c],
                    w[c]
                );
            }
        }
    }
}

