//! Constraint-matrix (`Cᵀ`) construction for [`Hybridization`]
//! (`crates/assembly/src/hybridization.rs`).
//!
//! This is the port of the `ConstructC()` step of MFEM
//! `fem/hybridization.cpp`: for every interior face, the constraint
//! integrator's `AssembleFaceMatrix(face_fe, trial_fe1, trial_fe2, FTr,
//! elmat)` produces an `(nd1 + nd2) × n_face_dofs` block that is added into
//! the global `Cᵀ` matrix (rows = "hat" element dofs, columns = trace-space
//! dofs).
//!
//! MFEM keeps the trace space (`c_fes`) and the constraint integrator
//! (`c_bfi`) fully generic — the caller picks them (ex4 Darcy uses
//! `DG_Interface_FECollection(order-1)` + `NormalTraceJumpIntegrator`).
//! This port offers the same choice through two enums:
//!
//! * [`TraceSpaceKind`] — the trace space: per-face discontinuous `Pₖ`
//!   (MFEM `DG_Interface_FECollection(k)`) or the H1-style trace made of
//!   the vertex + edge dofs lying on the face (what MFEM's
//!   `FiniteElementSpace::GetFaceVDofs` returns for an `H1_FECollection`
//!   `c_fes` on a 2-D mesh).
//! * [`ConstraintIntegratorKind`] — `NormalTraceJumpIntegrator` (H(div);
//!   exact port of `fem/bilininteg.cpp:NormalTraceJumpIntegrator::
//!   AssembleFaceMatrix`) and a 2-D tangential-trace jump for H(curl)
//!   (MFEM 4.10 ships no built-in H(curl) jump integrator — its
//!   `Hybridization` accepts any user integrator — so this port fixes the
//!   natural convention `C[u, φ] = ∫_F (u₁ − u₂)·τ φ ds`, reported as a
//!   port-level convention).
//!
//! Supported geometries: 2-D `Tri3` and `Quad4` meshes (straight edges;
//! the quadrature-point maps and Piola transforms mirror the element
//! geometry used by `VectorAssembler`).  `Tri6`/curved meshes are rejected.

use fem_element::quadrature::gauss_legendre_01;
use fem_element::reference::VectorReferenceElement;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::{FESpace, SpaceType};

use crate::interior_faces::InteriorFaceList;
use crate::vector_assembler::vec_ref_elem;

/// The trace ("constraint") space `c_fes` of the hybridization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceSpaceKind {
    /// Discontinuous polynomial trace, one polynomial per face (MFEM
    /// `DG_Interface_FECollection(order)`): `order + 1` dofs per 2-D face,
    /// all private to the face.  MFEM ex4 hybridization uses
    /// `DG_Interface_FECollection(order - 1)`.
    FaceDG {
        /// Polynomial order of the face trace element.
        order: u8,
    },
    /// H1-style trace: the dofs of an `H1_FECollection(order)` space that
    /// lie *on* the face — in 2-D the two endpoint vertex dofs plus
    /// `order − 1` edge-interior dofs (this is exactly what MFEM's
    /// `FiniteElementSpace::GetFaceVDofs` returns for an H1 collection on a
    /// 2-D mesh).  Adjacent faces share their endpoint dofs, so the
    /// hybridized matrix `H` couples neighbouring face traces.
    H1 {
        /// Polynomial order of the H1 trace space.
        order: u8,
    },
}

impl TraceSpaceKind {
    /// Polynomial order of the trace element on a face.
    fn order(self) -> u8 {
        match self {
            TraceSpaceKind::FaceDG { order } | TraceSpaceKind::H1 { order } => order,
        }
    }
}

/// The constraint face integrator (MFEM `c_bfi`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintIntegratorKind {
    /// `NormalTraceJumpIntegrator`: `C[u, φ] = ∫_F (u₁·n₁ − u₂·n₂) φ ds`
    /// with the outward normal `nᵢ` of each side.  For H(div) trial spaces.
    NormalTraceJump,
    /// 2-D tangential-trace jump for H(curl) trial spaces:
    /// `C[u, φ] = ∫_F (u₁ − u₂)·τ φ ds` with `τ` the unit tangent of the
    /// canonical (min-node → max-node) face direction — the same vector on
    /// both sides, so the block is a genuine jump.
    TangentialTraceJump2D,
}

/// Build the global constraint matrix `Cᵀ`
/// (port of `Hybridization::ConstructC`, serial, interior faces only).
///
/// `hat_offsets` is the `NE + 1` array of per-element hat-dof offsets
/// (computed by `Hybridization::init`).  Returns `(Ct, n_trace_dofs)` where
/// `Ct` is `num_hat_dofs × n_trace`.  Boundary-face constraint integrators
/// (MFEM `AddBdrConstraintIntegrator` / `AssembleBdrMatrix`) are not ported:
/// ex4's hybridization route imposes `F·n = 0` boundary conditions through
/// the essential-dof list instead.
pub(crate) fn construct_ct<M, S>(
    fespace: &S,
    hat_offsets: &[usize],
    trace: TraceSpaceKind,
    integrator: ConstraintIntegratorKind,
) -> (CsrMatrix<f64>, usize)
where
    M: MeshTopology,
    S: FESpace<Mesh = M>,
{
    let mesh = fespace.mesh();
    let dim = mesh.dim() as usize;
    assert!(dim == 2, "hybridization: only 2-D meshes are supported");
    let faces = InteriorFaceList::build(mesh);

    // ── Trace dof numbering ─────────────────────────────────────────────
    // FaceDG: interior face f owns a private block of (k+1) dofs.
    // H1: vertex dofs are the mesh node ids; edge-interior dofs are
    // numbered n_nodes + edge_index·(k−1) + m (canonical-edge order of
    // first appearance).
    let k = trace.order() as usize;
    let n_trace = match trace {
        TraceSpaceKind::FaceDG { .. } => faces.len() * (k + 1),
        TraceSpaceKind::H1 { .. } => {
            let mut seen = std::collections::HashSet::new();
            for f in &faces.faces {
                seen.insert(canonical_edge(&f.face_nodes));
            }
            mesh.n_nodes() + seen.len() * (k - 1).max(0)
        }
    };

    // Canonical-edge → running index (H1 edge-interior dofs).
    let mut edge_index: std::collections::HashMap<(u32, u32), usize> =
        std::collections::HashMap::new();
    if matches!(trace, TraceSpaceKind::H1 { .. }) && k >= 2 {
        let mut next = 0usize;
        for f in &faces.faces {
            let key = canonical_edge(&f.face_nodes);
            if !edge_index.contains_key(&key) {
                edge_index.insert(key, next);
                next += 1;
            }
        }
    }

    // Per interior face: global trace dof ids and their Lagrange node
    // parameters t ∈ [0,1] on the canonical (min-node → max-node) axis.
    let mut face_trace_dofs: Vec<Vec<usize>> = Vec::with_capacity(faces.len());
    let mut face_trace_nodes: Vec<Vec<f64>> = Vec::with_capacity(faces.len());
    for (f, face) in faces.faces.iter().enumerate() {
        let (a, b) = canonical_edge(&face.face_nodes);
        let mut dofs = Vec::with_capacity(k + 1);
        let mut nodes = Vec::with_capacity(k + 1);
        match trace {
            TraceSpaceKind::FaceDG { .. } => {
                let base = f * (k + 1);
                if k == 0 {
                    // P0 trace: one constant shape function.
                    dofs.push(base);
                    nodes.push(0.0);
                } else {
                    for i in 0..=k {
                        dofs.push(base + i);
                        nodes.push(i as f64 / k as f64);
                    }
                }
            }
            TraceSpaceKind::H1 { .. } => {
                dofs.push(a as usize);
                nodes.push(0.0);
                dofs.push(b as usize);
                nodes.push(1.0);
                if k >= 2 {
                    let idx = edge_index[&(a, b)];
                    let base = mesh.n_nodes() + idx * (k - 1);
                    for m in 1..k {
                        dofs.push(base + m - 1);
                        nodes.push(m as f64 / k as f64);
                    }
                }
            }
        }
        face_trace_dofs.push(dofs);
        face_trace_nodes.push(nodes);
    }

    // ── Assemble the face blocks into Cᵀ ────────────────────────────────
    let num_hat_dofs = *hat_offsets.last().expect("empty hat_offsets");
    let mut ct = CooMatrix::<f64>::new(num_hat_dofs, n_trace);
    let mtol = 1e-12; // MFEM ConstructC threshold (MFEM_USE_DOUBLE)
    for (f, face) in faces.faces.iter().enumerate() {
        let n_t = face_trace_dofs[f].len();

        // Quadrature: MFEM uses
        //   order = max(order_fe1, order_fe2) − 1 + order_face_fe
        // points = IntRules.Get(SEGMENT, order) → (order+2)/2 for ≥ 0.
        let ord_total = fespace.order() as i32 - 1 + trace.order() as i32;
        let np = (((ord_total + 2).max(1)) / 2).max(1) as usize;
        let (xq, wq) = gauss_legendre_01(np);

        // Per-side data: flux[i][q] = (outward flux component of basis i at
        // face quadrature point q), plus the s→t parametrization flip.
        let side_elems = [face.elem_left, face.elem_right];
        let mut flux: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        let mut ndofs = [0usize; 2];
        let mut flipped = [false; 2];
        for (side, &el) in side_elems.iter().enumerate() {
            let (fx, nd, flip) =
                side_flux(fespace, el, &face.face_nodes, integrator, np, &xq);
            flux[side] = fx;
            ndofs[side] = nd;
            flipped[side] = flip;
        }

        let n_rows = ndofs[0] + ndofs[1];
        let mut elmat = vec![0.0_f64; n_rows * n_t];
        let mut row0 = 0usize;
        for side in 0..2 {
            let sign = if side == 0 { 1.0 } else { -1.0 };
            for (q, &s) in xq.iter().enumerate() {
                let t = if flipped[side] { 1.0 - s } else { s };
                let w = wq[q];
                for (j, _) in face_trace_nodes[f].iter().enumerate() {
                    let psi_j = lagrange_1d(&face_trace_nodes[f], j, t);
                    for i in 0..ndofs[side] {
                        elmat[(row0 + i) * n_t + j] +=
                            sign * w * flux[side][q * ndofs[side] + i] * psi_j;
                    }
                }
            }
            row0 += ndofs[side];
        }

        // Threshold tiny entries (MFEM `elmat.Threshold(mtol·MaxMaxNorm)`)
        // and add the block into Cᵀ.  Rows are the hat dofs of both sides.
        let maxmax = elmat.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        let tol = mtol * maxmax;
        let mut row0 = 0usize;
        for (side, &el) in side_elems.iter().enumerate() {
            let hat_base = hat_offsets[el as usize];
            for i in 0..ndofs[side] {
                for j in 0..n_t {
                    let v = elmat[(row0 + i) * n_t + j];
                    if v.abs() > tol {
                        ct.add(hat_base + i, face_trace_dofs[f][j], v);
                    }
                }
            }
            row0 += ndofs[side];
        }
    }

    (ct.into_csr(), n_trace)
}

/// Canonical (min, max) node pair of a 2-D face.
fn canonical_edge(nodes: &[u32]) -> (u32, u32) {
    assert_eq!(nodes.len(), 2, "hybridization: 2-D faces must be segments");
    if nodes[0] <= nodes[1] {
        (nodes[0], nodes[1])
    } else {
        (nodes[1], nodes[0])
    }
}

/// Lagrange basis function `j` at parameter `t` for nodes `ts`.
fn lagrange_1d(ts: &[f64], j: usize, t: f64) -> f64 {
    let mut v = 1.0;
    for (m, &tm) in ts.iter().enumerate() {
        if m != j {
            v *= (t - tm) / (ts[j] - tm);
        }
    }
    v
}

/// Local face table: element-local (corner index pairs), matching the
/// reference-element face order used by the H(div)/H(curl) space builders
/// (`HDivSpace::build_2d_tri` uses `TRI_FACES = [(1,2), (0,2), (0,1)]`;
/// `build_2d_quad` uses `QUAD_FACES = [(0,1), (1,2), (2,3), (3,0)]`).
fn local_faces(elem_type: ElementType) -> &'static [(usize, usize)] {
    match elem_type {
        ElementType::Tri3 | ElementType::Tri6 => &[(1, 2), (0, 2), (0, 1)],
        ElementType::Quad4 => &[(0, 1), (1, 2), (2, 3), (3, 0)],
        _ => panic!("hybridization: unsupported 2-D element type {elem_type:?}"),
    }
}

/// Reference-domain corner coordinates (fem-element convention: the RT/ND
/// reference elements live on [0,1]², vertex k maps to element node k).
fn ref_corner(elem_type: ElementType, i: usize) -> [f64; 2] {
    match elem_type {
        ElementType::Tri3 | ElementType::Tri6 => match i {
            0 => [0.0, 0.0],
            1 => [1.0, 0.0],
            _ => [0.0, 1.0],
        },
        _ => match i {
            0 => [0.0, 0.0],
            1 => [1.0, 0.0],
            2 => [1.0, 1.0],
            _ => [0.0, 1.0],
        },
    }
}

/// Evaluate the side's contribution to the face block.
///
/// Returns `(flux, n_dofs, s_to_t_flip)` where `flux[q * n_dofs + i]` is
/// the integrator's flux component (normal or tangential, with the MFEM
/// `CalcOrtho` length convention already applied for the normal case) of
/// the signed physical basis function `i` at face quadrature point `q`.
fn side_flux<M, S>(
    fespace: &S,
    el: u32,
    face_nodes: &[u32],
    integrator: ConstraintIntegratorKind,
    np: usize,
    xq: &[f64],
) -> (Vec<f64>, usize, bool)
where
    M: MeshTopology,
    S: FESpace<Mesh = M>,
{
    let mesh = fespace.mesh();
    let elem_type = mesh.element_type(el);
    if elem_type != ElementType::Tri3 && elem_type != ElementType::Quad4 {
        panic!(
            "hybridization: element type {elem_type:?} is not supported \
             (only straight-sided Tri3/Quad4)"
        );
    }
    let verts = mesh.element_nodes(el);
    let p = |i: usize| -> [f64; 2] {
        let c = mesh.node_coords(verts[i]);
        [c[0], c[1]]
    };

    // Locate the local face (li, lj) matching the mesh face node set.
    let faces = local_faces(elem_type);
    let (a, b) = canonical_edge(face_nodes);
    let mut found = None;
    for &(li, lj) in faces {
        let (gi, gj) = (verts[li], verts[lj]);
        let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
        if (lo, hi) == (a, b) {
            found = Some((li, lj));
            break;
        }
    }
    let (li, lj) = found.unwrap_or_else(|| {
        panic!("hybridization: face ({a},{b}) not found in element {el}")
    });
    // Local edge param s goes verts[li] → verts[lj]; the canonical face
    // param t goes a → b.
    let flip = verts[li] != a;

    // Reference edge point ξ(s) = r_li + s (r_lj − r_li).
    let (ri, rj) = (ref_corner(elem_type, li), ref_corner(elem_type, lj));

    // Geometry Jacobian at a reference point (affine tri / bilinear quad).
    let geo_jac = |xi: [f64; 2]| -> ([[f64; 2]; 2], f64) {
        match elem_type {
            ElementType::Tri3 => {
                let p0 = p(0);
                let j = [[p(1)[0] - p0[0], p(2)[0] - p0[0]],
                         [p(1)[1] - p0[1], p(2)[1] - p0[1]]];
                let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
                (j, det)
            }
            ElementType::Quad4 => {
                // Q1: N_i(ξ,η) = L_i(ξ) L_i(η), L = (1−t, t).
                let (lx, ly) = ([1.0 - xi[0], xi[0]], [1.0 - xi[1], xi[1]]);
                let gx = [
                    ly[0] * (p(1)[0] - p(0)[0]) + ly[1] * (p(2)[0] - p(3)[0]),
                    ly[0] * (p(1)[1] - p(0)[1]) + ly[1] * (p(2)[1] - p(3)[1]),
                ];
                let gy = [
                    lx[0] * (p(3)[0] - p(0)[0]) + lx[1] * (p(2)[0] - p(1)[0]),
                    lx[0] * (p(3)[1] - p(0)[1]) + lx[1] * (p(2)[1] - p(1)[1]),
                ];
                let j = [[gx[0], gy[0]], [gx[1], gy[1]]];
                let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
                (j, det)
            }
            _ => unreachable!(),
        }
    };

    // Reference element (vector basis).
    let stype = fespace.space_type();
    let ref_elem: Box<dyn VectorReferenceElement> =
        vec_ref_elem(stype, elem_type, 2, fespace.order());
    let n_ref = ref_elem.n_dofs();
    let n_loc = fespace.element_dofs(el).len();
    assert_eq!(
        n_ref, n_loc,
        "hybridization: reference element covers all {n_loc} element dofs, \
         got {n_ref} (higher-order quad RT/ND interior bubbles are not \
         supported by the C-matrix builder)"
    );
    let signs = fespace.element_signs(el);

    let mut ref_phi = vec![0.0_f64; n_ref * 2];
    let mut phi = vec![0.0_f64; n_ref * 2]; // physical basis
    let mut flux = vec![0.0_f64; np * n_ref];

    // Face tangent (canonical direction) and its unit vector.  `a`/`b`
    // are global mesh node ids — take their coordinates directly.
    let (pa, pb) = (mesh.node_coords(a), mesh.node_coords(b));
    let tx = pb[0] - pa[0];
    let ty = pb[1] - pa[1];
    let tlen = (tx * tx + ty * ty).sqrt();

    for q in 0..np {
        let s = xq[q];
        let xi = [ri[0] + s * (rj[0] - ri[0]), ri[1] + s * (rj[1] - ri[1])];
        let (j, det) = geo_jac(xi);
        assert!(
            det.abs() > 1e-300,
            "hybridization: degenerate element {el}"
        );

        ref_elem.eval_basis_vec(&xi, &mut ref_phi);
        // Contravariant Piola: φ_phys = J φ_ref / det (H(div)).
        // Covariant Piola: φ_phys = J⁻ᵀ φ_ref (H(curl)).
        for i in 0..n_ref {
            let (p0, p1) = (ref_phi[i * 2], ref_phi[i * 2 + 1]);
            match stype {
                SpaceType::HDiv => {
                    phi[i * 2] = (j[0][0] * p0 + j[0][1] * p1) / det;
                    phi[i * 2 + 1] = (j[1][0] * p0 + j[1][1] * p1) / det;
                }
                SpaceType::HCurl => {
                    let inv = 1.0 / det;
                    // J⁻ᵀ = (1/det)·[[j11, −j10], [−j01, j00]]
                    phi[i * 2] = (j[1][1] * p0 - j[1][0] * p1) * inv;
                    phi[i * 2 + 1] = (-j[0][1] * p0 + j[0][0] * p1) * inv;
                }
                _ => panic!("hybridization: unsupported space type {stype:?}"),
            }
            if let Some(sg) = signs {
                let sg = sg[i];
                phi[i * 2] *= sg;
                phi[i * 2 + 1] *= sg;
            }
        }

        match integrator {
            ConstraintIntegratorKind::NormalTraceJump => {
                // The jump uses ONE normal for the face — the normal of the
                // canonical (min-node → max-node) direction, MFEM
                // CalcOrtho convention n = (τ_y, −τ_x) whose length equals
                // |τ| (the face measure).  Side 2 is subtracted in
                // `construct_ct`, so the constraint enforces
                // (u₁ − u₂)·n = 0, i.e. normal continuity.
                let tau = [tx, ty];
                let n = [tau[1], -tau[0]];
                for i in 0..n_ref {
                    flux[q * n_ref + i] = phi[i * 2] * n[0] + phi[i * 2 + 1] * n[1];
                }
            }
            ConstraintIntegratorKind::TangentialTraceJump2D => {
                // Unit tangent of the canonical face direction — the same
                // vector on both sides, so the assembled block is a jump.
                let tau = [tx / tlen, ty / tlen];
                for i in 0..n_ref {
                    flux[q * n_ref + i] = phi[i * 2] * tau[0] + phi[i * 2 + 1] * tau[1];
                }
            }
        }
    }

    (flux, n_ref, flip)
}
