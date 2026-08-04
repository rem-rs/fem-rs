//! MFEM-style DG interior-face integrators for IMEX advection–diffusion
//! (MFEM ex41): `NonconservativeDGTraceIntegrator` (upwind advection trace,
//! implemented as the transpose of `DGTraceIntegrator`) and
//! `DGDiffusionIntegrator` (SIPG, `sigma`/`kappa`).
//!
//! Geometry is taken from the mesh's *per-element* geometry table
//! (`Mesh::geometry` in per-element mode), which is how geometrically periodic
//! meshes (e.g. `periodic-square.mesh`) are assembled: the two sides of a
//! periodic seam live at different physical locations and each element's
//! shape/velocity evaluation uses its own geometry, while the face
//! transformation (normal) uses the `Elem1` trace, exactly like MFEM's
//! `Mesh::GetFaceTransformation` "L2 Nodes" branch.

use fem_core::types::{ElemId, FaceId, NodeId};
use fem_element::lagrange::factory::QuadQk;
use fem_element::ReferenceElement;
use fem_mesh::{element_type::ElementType, simplex::Mesh, topology::MeshTopology};
use fem_linalg::CooMatrix;
use fem_space::fe_space::FESpace;
use fem_space::L2Space;

/// Per-face local data: which local (H1) edge of each adjacent element the
/// face corresponds to, and whether the face direction is flipped w.r.t. the
/// second element's edge direction.
#[derive(Debug, Clone, Copy)]
pub struct FaceLoc {
    /// First element (Elem1 — face transformation uses its trace).
    pub e1: ElemId,
    /// Second element (Elem2, only for interior faces).
    pub e2: ElemId,
    /// Local H1 edge id of the face in `e1` (0=bottom, 1=right, 2=top, 3=left).
    pub e1_inf: usize,
    /// Local H1 edge id of the face in `e2`.
    pub e2_inf: usize,
    /// True when the face direction (Elem1 edge direction) is opposite to
    /// Elem2's edge direction (MFEM `orientation == 1`).
    pub e2_flip: bool,
}

/// Per-boundary-face data: the adjacent (interior) element and its local edge.
#[derive(Debug, Clone, Copy)]
pub struct BdrFaceLoc {
    /// Adjacent element (Elem1 — face transformation uses its trace).
    pub e1: ElemId,
    /// Local H1 edge id of the face in `e1`.
    pub e1_inf: usize,
}

/// Build the per-face local data from the element connectivity.  The face
/// direction is the edge direction of the first element that registered the
/// edge (same order as MFEM's `Mesh::GenerateFaces`/`FindFaces`), so
/// `face_dir == e1` edge direction (no flip on Elem1 side).
///
/// Faces are emitted in MFEM's face-numbering order: the order in which each
/// edge is *first* registered while scanning elements (MFEM `GenerateFaces`
/// numbers faces by `el_to_edge` row order = first-registration order), NOT
/// the order in which the second element is encountered.  This ordering
/// determines the CSR column order of the assembled matrices, which in turn
/// determines the floating-point accumulation order of the BlockILU block
/// graph — bit-identical MDF reordering requires it.
///
/// Only *interior* faces are returned (each edge registered twice); edges
/// registered once (mesh boundary) are skipped — use [`build_bdr_face_locs`]
/// for those.
pub fn build_face_locs(mesh: &Mesh<2>) -> Vec<FaceLoc> {
    const QUAD_EDGES: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];
    // Registered face: (e1, e1_inf, face direction start vertex, reg order).
    let mut face_map: std::collections::HashMap<(NodeId, NodeId), (ElemId, usize, NodeId, usize)> =
        std::collections::HashMap::new();
    let mut next_reg = 0usize;
    // (registration order, FaceLoc) — sorted at the end to match MFEM's
    // face numbering (first-registration order).
    let mut out = Vec::new();
    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        for (li, &(a, b)) in QUAD_EDGES.iter().enumerate() {
            let va = nodes[a];
            let vb = nodes[b];
            let key = (va.min(vb), va.max(vb));
            match face_map.remove(&key) {
                None => {
                    // First element registers the face; face direction = its
                    // edge direction.
                    face_map.insert(key, (e, li, va, next_reg));
                    next_reg += 1;
                }
                Some((e1, e1_inf, face_start, reg)) => {
                    // Second element: find its local edge matching this vertex
                    // pair and the direction relative to the face direction.
                    let mut e2_inf = 0usize;
                    let mut flip = false;
                    for (lj, &(ca, cb)) in QUAD_EDGES.iter().enumerate() {
                        if (nodes[ca].min(nodes[cb]), nodes[ca].max(nodes[cb])) == key {
                            e2_inf = lj;
                            // face direction: face_start -> other endpoint;
                            // e2 edge direction: nodes[ca] -> nodes[cb].
                            flip = face_start != nodes[ca];
                            break;
                        }
                    }
                    out.push((
                        reg,
                        FaceLoc {
                            e1,
                            e2: e,
                            e1_inf,
                            e2_inf,
                            e2_flip: flip,
                        },
                    ));
                }
            }
        }
    }
    // Remaining edges in face_map are boundary edges (registered once).
    let _ = face_map;
    out.sort_by_key(|(reg, _)| *reg);
    out.into_iter().map(|(_, f)| f).collect()
}

/// Build the per-boundary-face data: for each mesh boundary edge, the
/// adjacent element and its local edge id (MFEM `GetBdrElementAdjacentElement`).
///
/// The returned order matches the mesh's `face_conn` (boundary) order, which
/// mirrors MFEM's `boundary` array order.
pub fn build_bdr_face_locs(mesh: &Mesh<2>) -> Vec<BdrFaceLoc> {
    const QUAD_EDGES: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];
    // (min,max) vertex pair -> (elem, local edge).
    let mut edge_map: std::collections::HashMap<(NodeId, NodeId), (ElemId, usize)> =
        std::collections::HashMap::new();
    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        for (li, &(a, b)) in QUAD_EDGES.iter().enumerate() {
            let key = (nodes[a].min(nodes[b]), nodes[a].max(nodes[b]));
            edge_map.insert(key, (e, li));
        }
    }
    let mut out = Vec::new();
    for f in 0..mesh.n_faces() as FaceId {
        let ns = mesh.bface_nodes(f);
        if ns.len() != 2 {
            continue;
        }
        let key = (ns[0].min(ns[1]), ns[0].max(ns[1]));
        if let Some(&(e1, e1_inf)) = edge_map.get(&key) {
            out.push(BdrFaceLoc { e1, e1_inf });
        }
    }
    out
}

// ─── Reference geometry helpers ────────────────────────────────────────────

/// Bilinear map [0,1]² -> physical using the element's 4 geometry nodes in
/// H1 order (LL, LR, UR, UL).
fn bilinear_map(g: &[[f64; 2]; 4], xi: f64, eta: f64) -> [f64; 2] {
    let omx = 1.0 - xi;
    let omy = 1.0 - eta;
    [
        omx * omy * g[0][0] + xi * omy * g[1][0] + xi * eta * g[2][0] + omx * eta * g[3][0],
        omx * omy * g[0][1] + xi * omy * g[1][1] + xi * eta * g[2][1] + omx * eta * g[3][1],
    ]
}

/// Jacobian matrix J = [dp/dxi, dp/deta] (2x2) at a reference point.
fn jacobian(g: &[[f64; 2]; 4], xi: f64, eta: f64) -> [[f64; 2]; 2] {
    let omy = 1.0 - eta;
    let omx = 1.0 - xi;
    // dp/dxi
    let dxi = [
        omy * (g[1][0] - g[0][0]) + eta * (g[2][0] - g[3][0]),
        omy * (g[1][1] - g[0][1]) + eta * (g[2][1] - g[3][1]),
    ];
    // dp/deta
    let deta = [
        omx * (g[3][0] - g[0][0]) + xi * (g[2][0] - g[1][0]),
        omx * (g[3][1] - g[0][1]) + xi * (g[2][1] - g[1][1]),
    ];
    [[dxi[0], deta[0]], [dxi[1], deta[1]]]
}

fn det2(j: &[[f64; 2]; 2]) -> f64 {
    j[0][0] * j[1][1] - j[0][1] * j[1][0]
}

/// Map a face reference coordinate `xi` in [0,1] to the element reference
/// domain [0,1]² for local H1 edge `edge` (MFEM `GetLocalSegToQuadTransformation`).
fn loc_map(edge: usize, flip: bool, xi: f64) -> (f64, f64) {
    let t = if flip { 1.0 - xi } else { xi };
    match edge {
        0 => (t, 0.0),     // bottom: (0,0)->(1,0)
        1 => (1.0, t),     // right: (1,0)->(1,1)
        2 => (1.0 - t, 1.0), // top: (1,1)->(0,1)
        3 => (0.0, 1.0 - t), // left: (0,1)->(0,0)
        _ => unreachable!(),
    }
}

/// Face transformation (Elem1 trace): the face endpoint coordinates for local
/// edge `edge` of `g` (MFEM `GetFaceTransformation` L2 branch uses Elem1).
fn face_edge(g: &[[f64; 2]; 4], edge: usize) -> ([f64; 2], [f64; 2]) {
    match edge {
        0 => (g[0], g[1]), // bottom
        1 => (g[1], g[2]), // right
        2 => (g[2], g[3]), // top
        3 => (g[3], g[0]), // left
        _ => unreachable!(),
    }
}

/// CalcOrtho of the 2x1 face Jacobian: n = (dy, -dx), not normalised (the
/// length encodes the face measure, like MFEM).
fn face_normal(p0: &[f64; 2], p1: &[f64; 2]) -> [f64; 2] {
    [p1[1] - p0[1], -(p1[0] - p0[0])]
}

// ─── Assembly ───────────────────────────────────────────────────────────────

/// Assemble the MFEM-ex41 boundary-face contributions (the `ndof2 == 0`
/// branch of `AddBdrFaceIntegrator`):
///   * advection: `NonconservativeDGTraceIntegrator(velocity, alpha)` —
///     only the `(1,1)` block `w = ipw*(a+b)`, `a = 0.5*(-alpha)*un`,
///     `b = 0.5*alpha*|un|` survives (no Elem2 block);
///   * diffusion: `DGDiffusionIntegrator(diff, sigma, kappa)` — `w` is *not*
///     divided by 2, `wq` gets only the Elem1 contribution, and `jmat` fills
///     only the `(1,1)` lower-triangle block; the final combination
///     `elmat := -elmat + sigma*elmat^T + jmat` acts on the single block.
///
/// `space` must be an L2 (discontinuous) space; each element contributes
/// `dofs_per_elem` DOFs.  `vel` is evaluated at the *Elem1* physical point of
/// each face quadrature point, exactly like MFEM's `DGTraceIntegrator`.
pub fn assemble_ex41_bdr_faces<F>(
    coo: &mut CooMatrix<f64>,
    mesh: &Mesh<2>,
    space: &L2Space<Mesh<2>>,
    bdr_faces: &[BdrFaceLoc],
    vel: &F,
    alpha: f64,
    diff: f64,
    sigma: f64,
    kappa: f64,
) where
    F: Fn(f64, f64) -> [f64; 2],
{
    let order: u8 = space.order();
    debug_assert_eq!(mesh.element_nodes(0).len(), 4, "ex41 face assembly requires Quad4 meshes");
    if let Some(g) = mesh.geometry.as_ref() {
        debug_assert_eq!(g.nodes_per_elem, 4);
    }
    let re = QuadQk::new(order as usize);
    let n_dofs = re.n_dofs();
    let mut phi1 = vec![0.0f64; n_dofs];
    let mut dphi1 = vec![0.0f64; n_dofs * 2];

    // Element geometry (H1 order).
    let mut g1 = [[0.0f64; 2]; 4];

    // Quadrature rules on [0,1] (MFEM IntRules.Get(SEGMENT, order)):
    // advection trace: order = OrderW(Elem1) + 2*o1 (+1 Pk) = 1 + 6 + 1 = 8
    //                 -> 5 Gauss points (same as interior).
    // diffusion:      order = o1 = 3 -> 2*o1 = 6 -> 4 Gauss points.
    let (gp5, gw5) = fem_element::quadrature::gauss_legendre_01(5);
    let (gp4, gw4) = fem_element::quadrature::gauss_legendre_01(4);

    for f in bdr_faces {
        let e1 = f.e1 as usize;
        let gn1 = mesh.geometry_nodes(f.e1);
        for k in 0..4 {
            let c = mesh.geom_coords_of(gn1[k]);
            g1[k] = [c[0], c[1]];
        }

        let dofs1 = space.element_dofs(f.e1);
        let nd1 = dofs1.len();
        debug_assert_eq!(nd1, n_dofs);

        // Face endpoints (Elem1 trace) and normal — same convention as the
        // interior-face Elem1 side.
        let (fp0, fp1) = face_edge(&g1, f.e1_inf);
        let nor = face_normal(&fp0, &fp1);

        // Per-face block accumulator for the advection trace.
        let mut k_ll = vec![0.0f64; nd1 * nd1];

        // ── Advection trace (NonconservativeDGTraceIntegrator), (1,1) block ──
        for (qi, &xi) in gp5.iter().enumerate() {
            let ipw = gw5[qi];
            let (r1x, r1y) = loc_map(f.e1_inf, false, xi);
            re.eval_basis(&[r1x, r1y], &mut phi1);
            // Velocity evaluated at Elem1's physical point.
            let p1 = bilinear_map(&g1, r1x, r1y);
            let vu = vel(p1[0], p1[1]);
            let un = vu[0] * nor[0] + vu[1] * nor[1];
            // Inner integrator: DGTraceIntegrator(u, -alpha, 0.5*alpha).
            //   a = 0.5*alpha_in*un; b = beta_in*|un|
            // For the boundary face only the (1,1) block survives:
            //   w = ipw*(a+b); elmat(i,j) += w*s1(i)*s1(j)
            let a_in = 0.5 * (-alpha) * un;
            let b_in = 0.5 * alpha * un.abs();
            let w = ipw * (a_in + b_in);
            if w != 0.0 {
                for i in 0..nd1 {
                    for j in 0..nd1 {
                        k_ll[i * nd1 + j] += w * phi1[i] * phi1[j];
                    }
                }
            }
        }

        // ── Diffusion (DGDiffusionIntegrator), (1,1) block ──
        // elmat = -elmat + sigma*elmat^T + jmat (single block).
        let mut el = vec![0.0f64; nd1 * nd1];
        let mut jm = vec![0.0f64; nd1 * nd1];
        for (qi, &xi) in gp4.iter().enumerate() {
            let ipw = gw4[qi];
            let (r1x, r1y) = loc_map(f.e1_inf, false, xi);
            re.eval_basis(&[r1x, r1y], &mut phi1);
            re.eval_grad_basis(&[r1x, r1y], &mut dphi1);

            let j1 = jacobian(&g1, r1x, r1y);
            let det1 = det2(&j1);
            // adjugate of J1: [J11 -J01; -J10 J00]
            let adj1 = [[j1[1][1], -j1[0][1]], [-j1[1][0], j1[0][0]]];
            // Boundary face: NO division by 2 (ndof2 == 0).
            let mut w = ipw / det1;
            // Q == diffusion coefficient (MFEM ConstantCoefficient diff).
            w *= diff;
            // ni = w * nor
            let ni = [w * nor[0], w * nor[1]];
            // nh = adjJ * ni
            let nh = [
                adj1[0][0] * ni[0] + adj1[0][1] * ni[1],
                adj1[1][0] * ni[0] + adj1[1][1] * ni[1],
            ];
            // dshape1dn[j] = dshape1(j,:) . nh
            let mut dshape1dn = vec![0.0f64; nd1];
            for j in 0..nd1 {
                dshape1dn[j] = dphi1[j * 2] * nh[0] + dphi1[j * 2 + 1] * nh[1];
            }
            let wq = ni[0] * nor[0] + ni[1] * nor[1];
            // elmat (1,1): s1 * dshape1dn^T
            for i in 0..nd1 {
                for j in 0..nd1 {
                    el[i * nd1 + j] += phi1[i] * dshape1dn[j];
                }
            }
            if kappa != 0.0 {
                let wqk = wq * kappa;
                // only the lower-triangular part of the (1,1) block
                for i in 0..nd1 {
                    let wsi = wqk * phi1[i];
                    for j in 0..=i {
                        jm[i * nd1 + j] += wsi * phi1[j];
                    }
                }
            }
        }
        // elmat := -elmat + sigma*elmat^T + jmat (single (1,1) block)
        for i in 0..nd1 {
            for j in 0..i {
                let aij = el[i * nd1 + j];
                let aji = el[j * nd1 + i];
                let mij = jm[i * nd1 + j];
                el[i * nd1 + j] = sigma * aji - aij + mij;
                el[j * nd1 + i] = sigma * aij - aji + mij;
            }
            el[i * nd1 + i] = (sigma - 1.0) * el[i * nd1 + i] + jm[i * nd1 + i];
        }

        // Scatter the diffusion block.
        for i in 0..nd1 {
            for j in 0..nd1 {
                // MFEM AddSubMatrix(skip_zeros=0): keep structural zeros.
                coo.add(dofs1[i] as usize, dofs1[j] as usize, el[i * nd1 + j]);
            }
        }
        // Scatter the advection trace (1,1) block.
        for i in 0..nd1 {
            for j in 0..nd1 {
                coo.add(dofs1[i] as usize, dofs1[j] as usize, k_ll[i * nd1 + j]);
            }
        }
    }
}

/// Assemble the MFEM-ex41 interior-face contributions:
///   * advection: `NonconservativeDGTraceIntegrator(velocity, alpha)` =
///     transpose of `DGTraceIntegrator(velocity, -alpha, 0.5*alpha)`;
///   * diffusion: `DGDiffusionIntegrator(diff, sigma, kappa)`.
///
/// `space` must be an L2 (discontinuous) space; each element contributes
/// `dofs_per_elem` DOFs.  `vel` is evaluated at the *Elem1* physical point of
/// each face quadrature point, exactly like MFEM's `DGTraceIntegrator`.
pub fn assemble_ex41_interior_faces<F>(
    coo: &mut CooMatrix<f64>,
    mesh: &Mesh<2>,
    space: &L2Space<Mesh<2>>,
    faces: &[FaceLoc],
    vel: &F,
    alpha: f64,
    diff: f64,
    sigma: f64,
    kappa: f64,
) where
    F: Fn(f64, f64) -> [f64; 2],
{
    let order: u8 = space.order();
    let npe = mesh.element_nodes(0).len();
    debug_assert_eq!(npe, 4, "ex41 face assembly requires Quad4 meshes");
    if let Some(g) = mesh.geometry.as_ref() {
        debug_assert_eq!(g.nodes_per_elem, 4);
    }
    let re = QuadQk::new(order as usize);
    let n_dofs = re.n_dofs();
    let mut phi1 = vec![0.0f64; n_dofs];
    let mut phi2 = vec![0.0f64; n_dofs];
    let mut dphi1 = vec![0.0f64; n_dofs * 2];
    let mut dphi2 = vec![0.0f64; n_dofs * 2];

    // Element geometry (H1 order).
    let mut g1 = [[0.0f64; 2]; 4];
    let mut g2 = [[0.0f64; 2]; 4];

    // Quadrature rules on [0,1] (MFEM IntRules.Get(SEGMENT, order)):
    // advection trace: order = min(OrderW1, OrderW2) + 2*max(o1,o2) + 1 (Pk)
    //                 = 1 + 2*3 + 1 = 8 -> 5 Gauss points.
    // diffusion:      order = 2*max(o1,o2) = 6 -> 4 Gauss points.
    let (gp5, gw5) = fem_element::quadrature::gauss_legendre_01(5);
    let (gp4, gw4) = fem_element::quadrature::gauss_legendre_01(4);

    for f in faces {
        let e1 = f.e1 as usize;
        let e2 = f.e2 as usize;
        let gn1 = mesh.geometry_nodes(f.e1);
        let gn2 = mesh.geometry_nodes(f.e2);
        for k in 0..4 {
            let c = mesh.geom_coords_of(gn1[k]);
            g1[k] = [c[0], c[1]];
            let c = mesh.geom_coords_of(gn2[k]);
            g2[k] = [c[0], c[1]];
        }

        let dofs1 = space.element_dofs(f.e1);
        let dofs2 = space.element_dofs(f.e2);
        let nd1 = dofs1.len();
        let nd2 = dofs2.len();
        debug_assert_eq!(nd1, n_dofs);
        debug_assert_eq!(nd2, n_dofs);

        // Face endpoints (Elem1 trace) and normal.
        let (fp0, fp1) = face_edge(&g1, f.e1_inf);
        let nor = face_normal(&fp0, &fp1);

        // Per-face block accumulators: [k_ll k_lr; k_rl k_rr].
        let mut k_ll = vec![0.0f64; nd1 * nd1];
        let mut k_lr = vec![0.0f64; nd1 * nd2];
        let mut k_rl = vec![0.0f64; nd2 * nd1];
        let mut k_rr = vec![0.0f64; nd2 * nd2];

        // ── Advection trace (NonconservativeDGTraceIntegrator) ──
        for (qi, &xi) in gp5.iter().enumerate() {
            let ipw = gw5[qi];
            let (r1x, r1y) = loc_map(f.e1_inf, false, xi);
            let (r2x, r2y) = loc_map(f.e2_inf, f.e2_flip, xi);
            re.eval_basis(&[r1x, r1y], &mut phi1);
            re.eval_basis(&[r2x, r2y], &mut phi2);
            // Velocity evaluated at Elem1's physical point.
            let p1 = bilinear_map(&g1, r1x, r1y);
            let vu = vel(p1[0], p1[1]);
            let un = vu[0] * nor[0] + vu[1] * nor[1];
            // Inner integrator: DGTraceIntegrator(u, -alpha, 0.5*alpha).
            //   alpha_in = -alpha, beta_in = 0.5*alpha
            //   a = 0.5*alpha_in*un; b = beta_in*|un|
            // then the whole element matrix is transposed.
            let a_in = 0.5 * (-alpha) * un;
            let b_in = 0.5 * alpha * un.abs();
            let w1 = ipw * (a_in + b_in);
            let w2 = ipw * (b_in - a_in);
            // DGTrace blocks M (pre-transpose):
            //   M11 += w1 s1 s1^T ; M21 -= w1 s2 s1^T
            //   M22 += w2 s2 s2^T ; M12 -= w2 s1 s2^T
            // Transposed blocks (what we scatter):
            //   K11 = M11 ; K12 = M21^T ; K21 = M12^T ; K22 = M22
            if w1 != 0.0 {
                for i in 0..nd1 {
                    for j in 0..nd1 {
                        k_ll[i * nd1 + j] += w1 * phi1[i] * phi1[j];
                    }
                }
                // K12(r,c) = -w1 * s2[c] * s1[r]  (r = Elem1 dof row, c = Elem2 dof col)
                for r in 0..nd1 {
                    for c in 0..nd2 {
                        k_lr[r * nd2 + c] += -w1 * phi2[c] * phi1[r];
                    }
                }
            }
            if w2 != 0.0 {
                for i in 0..nd2 {
                    for j in 0..nd2 {
                        k_rr[i * nd2 + j] += w2 * phi2[i] * phi2[j];
                    }
                }
                // K21(r,c) = -w2 * s1[c] * s2[r]  (r = Elem2 dof row, c = Elem1 dof col)
                for r in 0..nd2 {
                    for c in 0..nd1 {
                        k_rl[r * nd1 + c] += -w2 * phi1[c] * phi2[r];
                    }
                }
            }
        }

        // ── Diffusion (DGDiffusionIntegrator) ──
        // elmat = -elmat + sigma*elmat^T + jmat
        let mut el = vec![0.0f64; (nd1 + nd2) * (nd1 + nd2)];
        let mut jm = vec![0.0f64; (nd1 + nd2) * (nd1 + nd2)];
        for (qi, &xi) in gp4.iter().enumerate() {
            let ipw = gw4[qi];
            let (r1x, r1y) = loc_map(f.e1_inf, false, xi);
            let (r2x, r2y) = loc_map(f.e2_inf, f.e2_flip, xi);
            re.eval_basis(&[r1x, r1y], &mut phi1);
            re.eval_basis(&[r2x, r2y], &mut phi2);
            re.eval_grad_basis(&[r1x, r1y], &mut dphi1);
            re.eval_grad_basis(&[r2x, r2y], &mut dphi2);

            let j1 = jacobian(&g1, r1x, r1y);
            let det1 = det2(&j1);
            // adjugate of J1: [J11 -J01; -J10 J00]
            let adj1 = [[j1[1][1], -j1[0][1]], [-j1[1][0], j1[0][0]]];
            let mut w = ipw / det1;
            if nd2 > 0 {
                w /= 2.0;
            }
            // Q == diffusion coefficient (MFEM ConstantCoefficient diff).
            w *= diff;
            // ni = w * nor  (Q == 1 constant diffusion coefficient)
            let ni = [w * nor[0], w * nor[1]];
            // nh = adjJ * ni
            let nh = [
                adj1[0][0] * ni[0] + adj1[0][1] * ni[1],
                adj1[1][0] * ni[0] + adj1[1][1] * ni[1],
            ];
            // dshape1dn[j] = dshape1(j,:) . nh
            let mut dshape1dn = vec![0.0f64; nd1];
            for j in 0..nd1 {
                dshape1dn[j] = dphi1[j * 2] * nh[0] + dphi1[j * 2 + 1] * nh[1];
            }
            let mut wq = ni[0] * nor[0] + ni[1] * nor[1];
            // elmat block (1,1): s1 * dshape1dn^T
            for i in 0..nd1 {
                for j in 0..nd1 {
                    el[i * (nd1 + nd2) + j] += phi1[i] * dshape1dn[j];
                }
            }
            if nd2 > 0 {
                let j2 = jacobian(&g2, r2x, r2y);
                let det2_ = det2(&j2);
                let adj2 = [[j2[1][1], -j2[0][1]], [-j2[1][0], j2[0][0]]];
                let w2_ = ipw / 2.0 / det2_ * diff;
                let ni2 = [w2_ * nor[0], w2_ * nor[1]];
                let nh2 = [
                    adj2[0][0] * ni2[0] + adj2[0][1] * ni2[1],
                    adj2[1][0] * ni2[0] + adj2[1][1] * ni2[1],
                ];
                let mut dshape2dn = vec![0.0f64; nd2];
                for j in 0..nd2 {
                    dshape2dn[j] = dphi2[j * 2] * nh2[0] + dphi2[j * 2 + 1] * nh2[1];
                }
                wq += ni2[0] * nor[0] + ni2[1] * nor[1];
                // elmat (1,2): s1 * dshape2dn^T
                for i in 0..nd1 {
                    for j in 0..nd2 {
                        el[i * (nd1 + nd2) + (nd1 + j)] += phi1[i] * dshape2dn[j];
                    }
                }
                // elmat (2,1): - s2 * dshape1dn^T
                for i in 0..nd2 {
                    for j in 0..nd1 {
                        el[(nd1 + i) * (nd1 + nd2) + j] -= phi2[i] * dshape1dn[j];
                    }
                }
                // elmat (2,2): - s2 * dshape2dn^T
                for i in 0..nd2 {
                    for j in 0..nd2 {
                        el[(nd1 + i) * (nd1 + nd2) + (nd1 + j)] -= phi2[i] * dshape2dn[j];
                    }
                }
            }
            if kappa != 0.0 {
                wq *= kappa;
                for i in 0..nd1 {
                    for j in 0..nd1 {
                        jm[i * (nd1 + nd2) + j] += wq * phi1[i] * phi1[j];
                    }
                }
                if nd2 > 0 {
                    for i in 0..nd2 {
                        for j in 0..nd1 {
                            jm[(nd1 + i) * (nd1 + nd2) + j] -= wq * phi2[i] * phi1[j];
                        }
                        for j in 0..nd2 {
                            jm[(nd1 + i) * (nd1 + nd2) + (nd1 + j)] += wq * phi2[i] * phi2[j];
                        }
                    }
                }
            }
        }
        // elmat := -elmat + sigma*elmat^T + jmat
        // NOTE: MFEM's `jmat` only stores the lower triangle (the (2,1)
        // block is nonzero, the (1,2) block is zero); in the final
        // combination `elmat(i,j) = sigma*aji - aij + mij` the *same* lower
        // value `mij = jmat(min(i,j), max(i,j))` is used for both (i,j) and
        // (j,i).  So the (1,2) position gets the (2,1) jmat value.
        let n = nd1 + nd2;
        for i in 0..n {
            for j in 0..n {
                // jmat lower-triangle value (symmetric diagonal blocks; the
                // (1,2) block mirrors the (2,1) block).
                let (r, c) = if i > j { (i, j) } else { (j, i) };
                let v = -el[i * n + j] + sigma * el[j * n + i] + jm[r * n + c];
                let (gi, gj) = if i < nd1 {
                    (dofs1[i], if j < nd1 { dofs1[j] } else { dofs2[j - nd1] })
                } else {
                    (dofs2[i - nd1], if j < nd1 { dofs1[j] } else { dofs2[j - nd1] })
                };
                // MFEM AddSubMatrix(skip_zeros=0): structural zeros are kept
                // (they matter for the block pattern of BlockILU).
                coo.add(gi as usize, gj as usize, v);
            }
        }

        // Scatter the advection trace blocks.
        for i in 0..nd1 {
            for j in 0..nd1 {
                coo.add(dofs1[i] as usize, dofs1[j] as usize, k_ll[i * nd1 + j]);
            }
            for j in 0..nd2 {
                coo.add(dofs1[i] as usize, dofs2[j] as usize, k_lr[i * nd2 + j]);
            }
        }
        for i in 0..nd2 {
            for j in 0..nd1 {
                coo.add(dofs2[i] as usize, dofs1[j] as usize, k_rl[i * nd1 + j]);
            }
            for j in 0..nd2 {
                coo.add(dofs2[i] as usize, dofs2[j] as usize, k_rr[i * nd2 + j]);
            }
        }
    }
}
