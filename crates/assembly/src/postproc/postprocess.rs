//! Post-processing utilities: element-wise gradient, curl, divergence, and
//! nodal gradient recovery (Zienkiewicz-Zhu style).

use nalgebra::DMatrix;

use fem_element::nedelec::{
    HexNDk, PrismND1, PrismNDk, QuadND2, QuadNDk, TetND2, TetNDk, TriND2, TriNDk,
};
use fem_element::raviart_thomas::{
    HexRTk, PrismRTk, QuadRT1, QuadRTk, TetRT1, TetRT2, TetRTk, TriRT1, TriRT2, TriRTk,
};
use fem_element::reference::VectorReferenceElement;
use fem_element::ReferenceElement;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::ElementTransformation;
use fem_space::fe_space::{FESpace, SpaceType};

// ─── Reference element factories ───────────────────────────────────────────

/// D364: delegated to the single source of truth.  Arm-for-arm identical to
/// the historical local table: H¹-slot simplices (D185/D157/D202 — pairs with
/// `space.element_dofs`) and the D265 hex arm (`HexQk` GLL slots on `[-1,1]³`,
/// `o.max(1)`).  D614: the hex arm covers every hexahedral cell label
/// (`Hex20`/`Hex27` share the one `HexQk` CUBE family), and the wedge /
/// pyramid arms route the H¹ families the DofManager numbers the spaces in
/// (`H1PrismPk` MFEM entity order, Fuentes pyramid).
fn ref_elem_vol(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match elem_type {
        ElementType::Tri3 | ElementType::Tri6 | ElementType::Tet4 => {
            fem_space::ref_elem::h1_simplex_slots(elem_type, order)
        }
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => {
            fem_space::ref_elem::gll_tensor(elem_type, order.max(1))
        }
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => {
            fem_space::ref_elem::h1_prism_slots(order.max(1))
        }
        ElementType::Pyramid5 | ElementType::Pyramid13 => fem_space::ref_elem::h1_pyramid_slots(
            order.max(1),
            fem_element::lagrange::PyramidBasisType::default(),
        ),
        _ => panic!("ref_elem_vol: unsupported (element_type={elem_type:?}, order={order})"),
    }
}

/// Reference-frame evaluation point of the element centroid.
///
/// Simplex (and prism) bases evaluate at `(1/(d+1), …)` in their `[0,1]^d`
/// frames; the hex bases (`HexNDk`/`HexRTk`/`HexQk`) live on `[-1,1]³`, whose
/// center is the origin.  Quad bases live on `[0,1]²` — same `1/(d+1)` rule
/// as the simplices.
fn ref_centroid(elem_type: ElementType, dim: usize) -> Vec<f64> {
    match elem_type {
        // The hex bases (`HexNDk`/`HexRTk`/`HexQk`) live on `[-1,1]³` —
        // centre = origin.  D614: `Hex20`/`Hex27` share the frame.
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => vec![0.0; 3],
        // D614: the fem-rs reference wedge is (ξ segment) × ((η, ζ) triangle),
        // centre (½, ⅓, ⅓); the reference pyramid's centroid is (½, ½, ¼)
        // (base ring at ζ = 0, apex at (0, 0, 1)).
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => {
            vec![0.5, 1.0 / 3.0, 1.0 / 3.0]
        }
        ElementType::Pyramid5 | ElementType::Pyramid13 => vec![0.5, 0.5, 0.25],
        _ => vec![1.0 / (dim as f64 + 1.0); dim],
    }
}

/// D265: whether the element must go through the isoparametric geometry
/// instead of the corner-difference simplex transform.  Straight (and curved)
/// Quad4/Hex8 elements: their bases live on `[0,1]²`/`[-1,1]³`, a corner
/// difference is only exact for parallelepipeds, and the hex corner triple
/// (nodes 0,1,2) is even coplanar — the simplex transform degenerates.  Same
/// recipe as the assembler and the D242/D250 error arms:
/// `geo_ref_elem_from_mesh` + `isoparametric_jacobian`.
/// D265: whether the element must go through the isoparametric geometry
/// instead of the corner-difference simplex transform.  Straight (and curved)
/// Quad4/Hex8 elements: their bases live on `[0,1]²`/`[-1,1]³`, a corner
/// difference is only exact for parallelepipeds, and the hex corner triple
/// (nodes 0,1,2) is even coplanar — the simplex transform degenerates.  Same
/// recipe as the assembler and the D242/D250 error arms:
/// `geo_ref_elem_from_mesh` + `isoparametric_jacobian`.  D614: every 3-D
/// non-simplex cell label joins (`Hex20`/`Hex27`/`Prism15`/`Prism18`/
/// `Pyramid13`) — the wedge/pyramid corner triples are equally coplanar and
/// their geometry families are served by `geo_ref_elem_from_mesh`.
fn is_iso_elem(elem_type: ElementType, dim: usize) -> bool {
    match elem_type {
        ElementType::Quad4 => dim == 2,
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => dim == 3,
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => dim == 3,
        ElementType::Pyramid5 | ElementType::Pyramid13 => dim == 3,
        _ => false,
    }
}

/// Isoparametric Jacobian `(J, det J, x_phys)` at reference point `xi`,
/// built from the mesh's geometry element/table (handles `geom_order > 1`
/// through `mesh.geometry_nodes`).
fn iso_jacobian<M: MeshTopology>(
    mesh: &M,
    e: u32,
    xi: &[f64],
    dim: usize,
) -> (DMatrix<f64>, f64, Vec<f64>) {
    let ge = crate::vector_assembler::geo_ref_elem_from_mesh(mesh, e)
        .expect("D265: missing geometry reference element for quad/hex postprocessing");
    let geo_nds = mesh.geometry_nodes(e);
    crate::vector_assembler::isoparametric_jacobian(mesh, geo_nds, ge.as_ref(), xi, dim)
}

/// Element-level copy of `vector_assembler::vec_ref_elem` for the element-wise
/// postprocessing kernels (curl / divergence at the centroid).
///
/// D245 sync: the hex H(div) arm selects the MFEM-default GaussLegendre
/// nodal-open variant (`HexRTk::new_gauss_legendre`, `RT_FECollection(p, 3)`)
/// exactly like the assembler dispatcher, so every consumer of
/// [`compute_element_divergence`] sees the same physical basis the assembly
/// used.  (The LOR stack keeps the IntegratedGLL pair via `lor_factory`.)
fn vec_ref_elem(
    space_type: SpaceType,
    elem_type: ElementType,
    dim: usize,
    order: u8,
) -> Box<dyn VectorReferenceElement> {
    match (space_type, elem_type, dim, order) {
        (SpaceType::HCurl, ElementType::Tri3 | ElementType::Tri6, 2, 1) => Box::new(TriNDk::new(1)),
        (SpaceType::HCurl, ElementType::Tri3 | ElementType::Tri6, 2, 2) => Box::new(TriND2),
        (SpaceType::HCurl, ElementType::Tri3 | ElementType::Tri6, 2, o) if o >= 3 => {
            Box::new(TriNDk::new(o as usize))
        }
        // 2-D surface elements embedded in 3-D: same 2-D Nédélec basis.
        (SpaceType::HCurl, ElementType::Tri3 | ElementType::Tri6, 3, 1) => Box::new(TriNDk::new(1)),
        (SpaceType::HCurl, ElementType::Tri3 | ElementType::Tri6, 3, 2) => Box::new(TriND2),
        (SpaceType::HCurl, ElementType::Tri3 | ElementType::Tri6, 3, o) if o >= 3 => {
            Box::new(TriNDk::new(o as usize))
        }
        (SpaceType::HCurl, ElementType::Quad4, 2, 1) => Box::new(QuadNDk::new(1)),
        (SpaceType::HCurl, ElementType::Quad4, 2, 2) => Box::new(QuadND2),
        (SpaceType::HCurl, ElementType::Quad4, 2, o) if o >= 3 => Box::new(QuadNDk::new(o as usize)),
        (SpaceType::HCurl, ElementType::Quad4, 3, 1) => Box::new(QuadNDk::new(1)),
        (SpaceType::HCurl, ElementType::Quad4, 3, 2) => Box::new(QuadND2),
        (SpaceType::HCurl, ElementType::Quad4, 3, o) if o >= 3 => Box::new(QuadNDk::new(o as usize)),
        (SpaceType::HCurl, ElementType::Tet4 | ElementType::Tet10, 3, 1) => Box::new(TetNDk::new(1)),
        (SpaceType::HCurl, ElementType::Tet4 | ElementType::Tet10, 3, 2) => Box::new(TetND2),
        (SpaceType::HCurl, ElementType::Tet4 | ElementType::Tet10, 3, o) if o >= 3 => {
            Box::new(TetNDk::new(o as usize))
        }
        (SpaceType::HCurl, ElementType::Hex8, 3, 1) => Box::new(HexNDk::new(1)),
        (SpaceType::HCurl, ElementType::Hex8, 3, 2) => Box::new(HexNDk::new(2)),
        (SpaceType::HCurl, ElementType::Hex8, 3, o) if o >= 3 => Box::new(HexNDk::new(o as usize)),
        (SpaceType::HDiv, ElementType::Quad4, 2, 0) => Box::new(QuadRTk::new(0)),
        (SpaceType::HDiv, ElementType::Quad4, 2, 1) => Box::new(QuadRT1),
        (SpaceType::HDiv, ElementType::Quad4, 2, o) if o >= 2 => {
            Box::new(fem_element::raviart_thomas::QuadRTk::new(o as usize))
        }
        (SpaceType::HDiv, ElementType::Tri3 | ElementType::Tri6, 2, 0) => Box::new(TriRTk::new(0)),
        (SpaceType::HDiv, ElementType::Tri3 | ElementType::Tri6, 2, 1) => Box::new(TriRT1),
        (SpaceType::HDiv, ElementType::Tri3 | ElementType::Tri6, 2, 2) => Box::new(TriRT2),
        // D245: MFEM-default GaussLegendre nodal-open RT on hexes (assembly
        // parity); the IntegratedGLL LOR pair is pinned in `lor_factory`.
        (SpaceType::HDiv, ElementType::Hex8, 3, 0) => Box::new(HexRTk::new_gauss_legendre(0)),
        (SpaceType::HDiv, ElementType::Hex8, 3, 1) => Box::new(HexRTk::new_gauss_legendre(1)),
        (SpaceType::HDiv, ElementType::Hex8, 3, o) if o >= 2 => {
            Box::new(HexRTk::new_gauss_legendre(o as usize))
        }
        (SpaceType::HDiv, ElementType::Tet4 | ElementType::Tet10, 3, 0) => Box::new(TetRTk::new(0)),
        (SpaceType::HDiv, ElementType::Tet4 | ElementType::Tet10, 3, 1) => Box::new(TetRT1),
        (SpaceType::HDiv, ElementType::Tet4 | ElementType::Tet10, 3, 2) => Box::new(TetRT2),
        (SpaceType::HDiv, ElementType::Prism6, 3, 0) => Box::new(PrismRTk::new(0)),
        (SpaceType::HDiv, ElementType::Prism6, 3, 1) => Box::new(PrismRTk::new(1)),
        (SpaceType::HCurl, ElementType::Prism6, 3, 1) => Box::new(PrismND1),
        (SpaceType::HCurl, ElementType::Prism6, 3, o) if o >= 2 => {
            Box::new(PrismNDk::new(o as usize))
        }
        _ => panic!(
            "vec_ref_elem: unsupported (space_type={space_type:?}, elem_type={elem_type:?}, dim={dim}, order={order})"
        ),
    }
}

// ─── Jacobian helpers ──────────────────────────────────────────────────────

fn simplex_jacobian<M: MeshTopology>(
    mesh: &M,
    geo_nodes: &[u32],
) -> (DMatrix<f64>, f64) {
    let tr = ElementTransformation::from_simplex_nodes(mesh, geo_nodes);
    (tr.jacobian().clone(), tr.det_j())
}

fn transform_grads(
    j_inv_t: &DMatrix<f64>,
    grad_ref: &[f64],
    grad_phys: &mut [f64],
    n_ldofs: usize,
    dim: usize,
) {
    for i in 0..n_ldofs {
        for d in 0..dim {
            let mut s = 0.0;
            for k in 0..dim {
                s += j_inv_t[(d, k)] * grad_ref[i * dim + k];
            }
            grad_phys[i * dim + d] = s;
        }
    }
}

// ─── Element-wise gradient (scalar H1/L2 spaces) ──────────────────────────

/// Compute element-wise gradient of a scalar FE solution.
///
/// Evaluates the gradient at the element centroid.
/// Returns a `Vec` of length `n_elements`, each entry is `[du/dx, du/dy, ...]`.
pub fn compute_element_gradients<S: FESpace>(space: &S, dofs: &[f64]) -> Vec<Vec<f64>> {
    let mesh = space.mesh();
    let dim = mesh.dim() as usize;
    let order = space.order();

    let mut result = Vec::with_capacity(mesh.n_elements());

    for e in mesh.elem_iter() {
        let elem_type = mesh.element_type(e);
        let ref_elem = ref_elem_vol(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();

        let elem_dofs = space.element_dofs(e);
        let nodes = mesh.element_nodes(e);

        // D265: quad/hex elements go through the isoparametric geometry
        // (corner-difference transform is wrong there); simplices keep the
        // affine path bit-identical.
        let (jac, _det_j) = if is_iso_elem(elem_type, dim) {
            let xi = ref_centroid(elem_type, dim);
            let (j, d, _xp) = iso_jacobian(mesh, e, &xi, dim);
            (j, d)
        } else {
            simplex_jacobian(mesh, nodes)
        };
        let j_inv_t = jac.try_inverse().expect("degenerate element").transpose();

        // Evaluate at centroid: (1/3, 1/3) for tri, (1/4, 1/4, 1/4) for tet,
        // the origin for hex ([-1,1]³ frame).
        let xi: Vec<f64> = ref_centroid(elem_type, dim);

        let mut grad_ref = vec![0.0; n_ldofs * dim];
        let mut grad_phys = vec![0.0; n_ldofs * dim];
        ref_elem.eval_grad_basis(&xi, &mut grad_ref);
        transform_grads(&j_inv_t, &grad_ref, &mut grad_phys, n_ldofs, dim);

        let mut grad = vec![0.0; dim];
        for i in 0..n_ldofs {
            let c = dofs[elem_dofs[i] as usize];
            for d in 0..dim {
                grad[d] += c * grad_phys[i * dim + d];
            }
        }
        result.push(grad);
    }

    result
}

// ─── H1 seminorm error ────────────────────────────────────────────────────

/// Compute the H¹ seminorm error `‖∇u_exact − ∇u_h‖_{L²(Ω)}`.
///
/// Integrates `|∇u_exact − ∇u_h|²` over every element using Gaussian
/// quadrature (not just the centroid), and returns the square root.
///
/// # Arguments
/// * `space`      — finite element space (H¹, scalar)
/// * `dofs`       — FE solution coefficient vector (length = `space.n_dofs()`)
/// * `grad_exact` — closure mapping physical coordinates `x` to the exact
///   gradient vector (length = mesh dimension)
/// * `quad_order` — quadrature accuracy order.  Use `order * 2 + 2` or higher
///   for P1/P2 solutions.
///
/// # Returns
/// `sqrt( ∫_Ω |∇u_exact - ∇u_h|² dΩ )`
pub fn compute_h1_error<S: FESpace>(
    space: &S,
    dofs: &[f64],
    grad_exact: impl Fn(&[f64]) -> Vec<f64>,
    quad_order: u8,
) -> f64 {
    let mesh = space.mesh();
    let dim = mesh.dim() as usize;
    let order = space.order();

    let mut err2 = 0.0_f64;

    for e in mesh.elem_iter() {
        let elem_type = mesh.element_type(e);
        let ref_elem = ref_elem_vol(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();

        let elem_dofs = space.element_dofs(e);
        let nodes = mesh.element_nodes(e);

        // ── Jacobian for this element ────────────────────────────────────
        // D265: quad/hex geometry is isoparametric per quadrature point
        // (element-level affine machinery unused); simplices keep the affine
        // path bit-identical.
        let use_iso = is_iso_elem(elem_type, dim);
        let (jac, det_j) = if use_iso {
            (None, 0.0_f64)
        } else {
            let (j, d) = simplex_jacobian(mesh, nodes);
            (Some(j), d)
        };
        let j_inv_t = jac.as_ref().and_then(|j| {
            j.clone()
                .try_inverse()
                .map(|inv| inv.transpose())
        });

        // Cache vertex coordinates for physical-point mapping.
        // For a 2-D simplex: x = x0 + J * xi (J built from x1-x0, x2-x0).
        let x0: Vec<f64> = mesh.node_coords(nodes[0]).to_vec();
        let jac_cols: Vec<Vec<f64>> = (1..=dim)
            .map(|k| {
                let xk = mesh.node_coords(nodes[k]);
                (0..dim).map(|d| xk[d] - x0[d]).collect()
            })
            .collect();

        // ── Quadrature ───────────────────────────────────────────────────
        let quad = ref_elem.quadrature(quad_order);
        let mut grad_ref = vec![0.0_f64; n_ldofs * dim];
        let mut grad_phys = vec![0.0_f64; n_ldofs * dim];

        for (qi, xi) in quad.points.iter().enumerate() {
            // Reference → physical coordinates and quadrature weight.
            // D265: hex/quad use the isoparametric map (the quadrature and
            // the geometry share the element's reference frame, so `xi` is
            // used directly); simplices keep the affine x0 + J·xi map.
            let (w, x_phys, j_inv_t) = if use_iso {
                let (j_iso, det_iso, xp_iso) = iso_jacobian(mesh, e, xi, dim);
                let jt = j_iso
                    .try_inverse()
                    .expect("degenerate element in compute_h1_error")
                    .transpose();
                (quad.weights[qi] * det_iso.abs(), xp_iso, jt)
            } else {
                let mut xp: Vec<f64> = x0.clone();
                for k in 0..dim {
                    for d in 0..dim {
                        xp[d] += jac_cols[k][d] * xi[k];
                    }
                }
                let jt = j_inv_t
                    .clone()
                    .expect("simplex arm must carry an affine Jacobian");
                (quad.weights[qi] * det_j.abs(), xp, jt)
            };

            // Basis gradients in reference space, then transform to physical.
            ref_elem.eval_grad_basis(xi, &mut grad_ref);
            transform_grads(&j_inv_t, &grad_ref, &mut grad_phys, n_ldofs, dim);

            // FE gradient: ∇u_h = Σ_i c_i ∇φ_i(x)
            let mut grad_h = vec![0.0_f64; dim];
            for i in 0..n_ldofs {
                let c = dofs[elem_dofs[i] as usize];
                for d in 0..dim {
                    grad_h[d] += c * grad_phys[i * dim + d];
                }
            }

            // Exact gradient at this physical point.
            let grad_ex = grad_exact(&x_phys);

            // Accumulate ‖∇u_exact − ∇u_h‖² weighted by quadrature weight.
            let diff2: f64 = (0..dim).map(|d| (grad_ex[d] - grad_h[d]).powi(2)).sum();
            err2 += w * diff2;
        }
    }

    err2.sqrt()
}

// ─── Kelly error indicators ───────────────────────────────────────────────────

/// Compute element-wise Kelly error indicators for a scalar H¹ solution.
///
/// The Kelly indicator for element `K` is:
/// ```text
///   η_K = sqrt( Σ_{E ∈ ∂K ∩ Ω_int} h_E * |[∇u_h · n_E]|² )
/// ```
/// where the sum is over interior edges, `h_E` is the edge length, and
/// `[∇u_h · n_E]` is the normal gradient jump across the edge.
///
/// This is a simplified (facet-jump) variant without volume residuals.
/// Suitable as a refinement indicator for AMR marking (Dörfler / maximum strategy).
///
/// # Arguments
/// * `space`          — finite element space (H¹, scalar)
/// * `dofs`           — FE solution coefficient vector
/// * `interior_faces` — pre-computed interior face list (from [`InteriorFaceList::build`])
///
/// # Returns
/// A `Vec<f64>` of length `n_elements` where `result[e]` is `η_e²`
/// (take square root for the actual indicator).
pub fn compute_kelly_indicators<S: FESpace>(
    space: &S,
    dofs: &[f64],
    interior_faces: &crate::InteriorFaceList,
) -> Vec<f64> {
    let mesh = space.mesh();
    let n_elems = mesh.n_elements();

    // Pre-compute element-wise gradients (centroid evaluation).
    let grads = compute_element_gradients(space, dofs);

    let mut indicators = vec![0.0_f64; n_elems];

    for face in &interior_faces.faces {
        let na = face.face_nodes[0];
        let nb = face.face_nodes[1];

        let (h_e, nx, ny) = edge_length_and_normal_2d(mesh, na, nb);

        let gl = &grads[face.elem_left as usize];
        let gr = &grads[face.elem_right as usize];

        // Normal gradient jump: [∇u_h · n]
        let jump = (gl[0] - gr[0]) * nx + (gl[1] - gr[1]) * ny;
        let contrib = h_e * jump * jump;

        indicators[face.elem_left  as usize] += contrib;
        indicators[face.elem_right as usize] += contrib;
    }

    indicators
}

/// Compute the 2-D edge length and outward unit normal for the edge `(na, nb)`.
/// The normal is perpendicular to the edge direction, rotated 90° counterclockwise.
fn edge_length_and_normal_2d<M: MeshTopology>(
    mesh: &M,
    na: u32,
    nb: u32,
) -> (f64, f64, f64) {
    let xa = mesh.node_coords(na);
    let xb = mesh.node_coords(nb);
    let dx = xb[0] - xa[0];
    let dy = xb[1] - xa[1];
    let h = (dx * dx + dy * dy).sqrt();
    // Unit normal: rotate edge vector 90° CCW, then normalise.
    // (The sign convention doesn't matter for jump² magnitude.)
    (h, -dy / h, dx / h)
}

// ─── Element-wise curl (H(curl) spaces) ────────────────────────────────────

/// Compute element-wise curl of an H(curl) FE solution.
///
/// Evaluates at the element centroid.
/// For 2D: each entry is a `Vec<f64>` of length 1 (scalar curl).
/// For 3D: each entry is a `Vec<f64>` of length 3 (vector curl).
pub fn compute_element_curl<S: FESpace>(space: &S, dofs: &[f64]) -> Vec<Vec<f64>> {
    let mesh = space.mesh();
    let dim = mesh.dim() as usize;
    let stype = space.space_type();

    let curl_dim = if dim == 2 { 1 } else { 3 };

    let mut result = Vec::with_capacity(mesh.n_elements());

    for e in mesh.elem_iter() {
        // D245: dispatch per element so mixed meshes pick the right basis
        // (and hex H(div) no longer mis-dispatches onto the tet element).
        let elem_type = mesh.element_type(e);
        let ref_elem = vec_ref_elem(stype, elem_type, dim, space.order());
        let n_ldofs = ref_elem.n_dofs();

        let mut ref_curl = vec![0.0; n_ldofs * curl_dim];
        let mut phys_curl = vec![0.0; n_ldofs * curl_dim];

        let elem_dofs = space.element_dofs(e);
        let signs = space.element_signs(e);
        // D58 canonical reconstruction: with face blocks (tet NDk, k ≥ 2) the
        // global vector is canonical → `u_local = S·u_canon` and the curl
        // basis is used unsigned; empty blocks → historical signed path.
        let blocks = space.element_face_blocks(e);
        let uloc = if blocks.is_empty() {
            None
        } else {
            Some(crate::vector_assembler::element_local_dofs_canonical(
                space, e, dofs,
            ))
        };
        let nodes = mesh.element_nodes(e);

        // D265: quad/hex elements evaluate through the isoparametric
        // geometry; the centroid sits at the hex frame origin ([-1,1]³).
        let xi = ref_centroid(elem_type, dim);
        let (jac, det_j) = if is_iso_elem(elem_type, dim) {
            let (j, d, _xp) = iso_jacobian(mesh, e, &xi, dim);
            (j, d)
        } else {
            simplex_jacobian(mesh, nodes)
        };

        ref_elem.eval_curl(&xi, &mut ref_curl);

        // H(curl) curl transform: curl_phys = J * curl_ref / det_j (3D)
        //                          curl_phys = curl_ref / det_j (2D scalar)
        let inv_det = 1.0 / det_j;
        if dim == 2 {
            for i in 0..n_ldofs {
                phys_curl[i] = ref_curl[i] * inv_det;
            }
        } else {
            for i in 0..n_ldofs {
                for r in 0..3 {
                    let mut s = 0.0;
                    for c in 0..3 {
                        s += jac[(r, c)] * ref_curl[i * 3 + c];
                    }
                    phys_curl[i * 3 + r] = s * inv_det;
                }
            }
        }

        // Apply orientation signs.
        if uloc.is_none() {
            if let Some(s) = signs {
                for i in 0..n_ldofs {
                    for c in 0..curl_dim {
                        phys_curl[i * curl_dim + c] *= s[i];
                    }
                }
            }
        }

        // Sum contributions: curl(u_h) = Σ_i c_i curl(φ_i)
        let mut curl_val = vec![0.0; curl_dim];
        for i in 0..n_ldofs {
            let c = match &uloc {
                Some(u) => u[i],
                None => dofs[elem_dofs[i] as usize],
            };
            for d in 0..curl_dim {
                curl_val[d] += c * phys_curl[i * curl_dim + d];
            }
        }
        result.push(curl_val);
    }

    result
}

// ─── Element-wise divergence (H(div) spaces) ──────────────────────────────

/// Compute element-wise divergence of an H(div) FE solution.
///
/// Evaluates at the element centroid.
/// Returns a `Vec<f64>` of length `n_elements`.
pub fn compute_element_divergence<S: FESpace>(space: &S, dofs: &[f64]) -> Vec<f64> {
    let mesh = space.mesh();
    let dim = mesh.dim() as usize;
    let stype = space.space_type();

    let mut result = Vec::with_capacity(mesh.n_elements());

    for e in mesh.elem_iter() {
        // D245: dispatch per element (see compute_element_curl).
        let elem_type = mesh.element_type(e);
        let ref_elem = vec_ref_elem(stype, elem_type, dim, space.order());
        let n_ldofs = ref_elem.n_dofs();

        let mut ref_div = vec![0.0; n_ldofs];
        let mut phys_div = vec![0.0; n_ldofs];

        let elem_dofs = space.element_dofs(e);
        let signs = space.element_signs(e);
        let nodes = mesh.element_nodes(e);

        // D265: quad/hex elements evaluate through the isoparametric
        // geometry; the centroid sits at the hex frame origin ([-1,1]³).
        let xi = ref_centroid(elem_type, dim);
        let (_jac, det_j) = if is_iso_elem(elem_type, dim) {
            let (j, d, _xp) = iso_jacobian(mesh, e, &xi, dim);
            (j, d)
        } else {
            simplex_jacobian(mesh, nodes)
        };

        ref_elem.eval_div(&xi, &mut ref_div);

        // H(div) divergence transform: div_phys = div_ref / det_j
        let inv_det = 1.0 / det_j;
        for i in 0..n_ldofs {
            phys_div[i] = ref_div[i] * inv_det;
        }

        // Apply orientation signs.
        if let Some(s) = signs {
            for i in 0..n_ldofs {
                phys_div[i] *= s[i];
            }
        }

        // Sum: div(u_h) = Σ_i c_i div(φ_i)
        let mut div_val = 0.0;
        for i in 0..n_ldofs {
            div_val += dofs[elem_dofs[i] as usize] * phys_div[i];
        }
        result.push(div_val);
    }

    result
}

// ─── Nodal gradient recovery (Zienkiewicz-Zhu) ────────────────────────────

/// Nodal gradient recovery via area-weighted averaging (Zienkiewicz-Zhu style).
///
/// For a scalar H1 solution, computes the smoothed gradient at each mesh node
/// by averaging the element-wise constant gradients weighted by element area.
///
/// Returns a `Vec` of `dim` vectors, each of length `n_nodes`:
/// `result[d][node]` = d-th component of the recovered gradient at `node`.
pub fn recover_gradient_nodal<S: FESpace>(space: &S, dofs: &[f64]) -> Vec<Vec<f64>> {
    let mesh = space.mesh();
    let dim = mesh.dim() as usize;
    let n_nodes = mesh.n_nodes();
    let order = space.order();

    // Accumulate area-weighted gradient at each node.
    let mut grad_accum = vec![vec![0.0; n_nodes]; dim];
    let mut area_accum = vec![0.0; n_nodes];

    for e in mesh.elem_iter() {
        let elem_type = mesh.element_type(e);
        let ref_elem = ref_elem_vol(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();

        let elem_dofs = space.element_dofs(e);
        let nodes = mesh.element_nodes(e);

        // D265: hexes evaluate through the isoparametric geometry.
        let use_iso = is_iso_elem(elem_type, dim);
        let xi = ref_centroid(elem_type, dim);
        let (jac, det_j) = if use_iso {
            let (j, d, _xp) = iso_jacobian(mesh, e, &xi, dim);
            (j, d)
        } else {
            simplex_jacobian(mesh, nodes)
        };
        let j_inv_t = jac.try_inverse().expect("degenerate element").transpose();

        // Element area/volume: for a simplex |det_j| / d!; for a hex the
        // integral of |det J| over the [-1,1]³ frame (the corner-difference
        // determinant has no simplex meaning there).
        let elem_area = if use_iso {
            let vol_order = (2 * mesh.geom_order().max(1)).max(2);
            let gq = ref_elem.quadrature(vol_order);
            let mut vol = 0.0_f64;
            for (gx, gw) in gq.points.iter().zip(gq.weights.iter()) {
                let (_jg, dg, _xp) = iso_jacobian(mesh, e, gx, dim);
                vol += gw * dg.abs();
            }
            vol
        } else {
            det_j.abs() / match dim {
                2 => 2.0,
                3 => 6.0,
                _ => 1.0,
            }
        };

        // Gradient at centroid.
        let mut grad_ref = vec![0.0; n_ldofs * dim];
        let mut grad_phys = vec![0.0; n_ldofs * dim];
        ref_elem.eval_grad_basis(&xi, &mut grad_ref);
        transform_grads(&j_inv_t, &grad_ref, &mut grad_phys, n_ldofs, dim);

        let mut grad = vec![0.0; dim];
        for i in 0..n_ldofs {
            let c = dofs[elem_dofs[i] as usize];
            for d in 0..dim {
                grad[d] += c * grad_phys[i * dim + d];
            }
        }

        // Distribute to element vertices: simplices use the first dim+1
        // nodes; hexes all 8 corners (MFEM `CUBE::Vertices` order — the
        // first 8 slots of the element node list).  D614: the higher-order
        // cell labels distribute over their TRUE corner counts
        // (Hex20/Hex27 → 8, wedges → 6, pyramids → 5), never the conn
        // length (the Tet10/D235 lesson).
        let n_verts = match elem_type {
            ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => 8,
            ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => 6,
            ElementType::Pyramid5 | ElementType::Pyramid13 => 5,
            _ => dim + 1,
        };
        for v in 0..n_verts {
            let node = nodes[v] as usize;
            area_accum[node] += elem_area;
            for d in 0..dim {
                grad_accum[d][node] += elem_area * grad[d];
            }
        }
    }

    // Divide by total accumulated weight.
    for node in 0..n_nodes {
        if area_accum[node] > 0.0 {
            for d in 0..dim {
                grad_accum[d][node] /= area_accum[node];
            }
        }
    }

    grad_accum
}

// ─── Element-wise scalar integration ────────────────────────────────────────

/// Integrate piecewise-constant element values against a test basis on a 2-D
/// mesh, returning one integrated scalar per element.
///
/// Each entry in `elem_values` is interpreted as a constant source over the
/// corresponding element.  The result is the L²-type projection
/// `result[e] = ∫_K s_e · w dx / ∫_K 1 dx` (element-average of the source
/// against the FE test basis `w`).
///
/// Used by multiphysics template examples (Joule, EM-thermal-stress) for
/// post-processing per-element diagnostics.
pub fn integrate_element_scalar_2d<S: FESpace>(
    mesh: &fem_mesh::Mesh<2>,
    space: &S,
    elem_values: &[f64],
) -> Vec<f64> {
    use fem_mesh::element_type::ElementType;
    let qo = (2 * space.order() + 1).max(4);
    let ne = mesh.n_elements();
    assert_eq!(elem_values.len(), ne, "integrate_element_scalar_2d: elem_values length mismatch");
    let mut result = vec![0.0_f64; ne];
    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        let ok = matches!(et, ElementType::Tri3 | ElementType::Tri6 | ElementType::Quad4);
        if !ok { continue; }
        let ref_e = ref_elem_vol(et, space.order());
        let quad = ref_e.quadrature(qo);
        let nodes = mesh.element_nodes(e);
        let tr = fem_mesh::ElementTransformation::from_simplex_nodes(mesh, nodes);
        let s = elem_values[e as usize];
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let n_ldofs = ref_e.n_dofs();
        let mut phi = vec![0.0; n_ldofs];
        for (qi, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[qi] * tr.det_j().abs();
            ref_e.eval_basis(xi, &mut phi);
            for j in 0..n_ldofs {
                result[dofs[j]] += w * s * phi[j];
            }
        }
    }
    result
}

/// Integrate piecewise-constant element values on a 3-D Tet4 mesh (same logic
/// as [`integrate_element_scalar_2d`] but for 3-D).
pub fn integrate_element_scalar_3d<S: FESpace>(
    mesh: &fem_mesh::Mesh<3>,
    space: &S,
    elem_values: &[f64],
) -> Vec<f64> {
    use fem_mesh::element_type::ElementType;
    let qo = (2 * space.order() + 1).max(4);
    let ne = mesh.n_elements();
    assert_eq!(elem_values.len(), ne, "integrate_element_scalar_3d: elem_values length mismatch");
    let mut result = vec![0.0_f64; ne];
    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        if et != ElementType::Tet4 { continue; }
        let ref_e = ref_elem_vol(et, space.order());
        let quad = ref_e.quadrature(qo);
        let nodes = mesh.element_nodes(e);
        let tr = fem_mesh::ElementTransformation::from_simplex_nodes(mesh, nodes);
        let s = elem_values[e as usize];
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let n_ldofs = ref_e.n_dofs();
        let mut phi = vec![0.0; n_ldofs];
        for (qi, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[qi] * tr.det_j().abs();
            ref_e.eval_basis(xi, &mut phi);
            for j in 0..n_ldofs {
                result[dofs[j]] += w * s * phi[j];
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::{H1Space, HCurlSpace, HDivSpace, fe_space::FESpace};

    #[test]
    fn element_gradients_linear_function() {
        // u(x,y) = 3x - 2y → ∇u = [3, -2] everywhere (constant per element).
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let v = space.interpolate(&|x| 3.0 * x[0] - 2.0 * x[1]);
        let dofs = v.as_slice();

        let grads = compute_element_gradients(&space, dofs);
        assert_eq!(grads.len(), space.mesh().n_elements());

        for (e, g) in grads.iter().enumerate() {
            assert!(
                (g[0] - 3.0).abs() < 1e-10,
                "elem {e}: ∂u/∂x = {}, expected 3.0",
                g[0]
            );
            assert!(
                (g[1] + 2.0).abs() < 1e-10,
                "elem {e}: ∂u/∂y = {}, expected -2.0",
                g[1]
            );
        }
    }

    #[test]
    fn recover_gradient_nodal_linear() {
        // For u(x,y) = 3x - 2y, the recovered gradient at every node should
        // be [3, -2] exactly (since all contributing elements have the same gradient).
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let v = space.interpolate(&|x| 3.0 * x[0] - 2.0 * x[1]);
        let dofs = v.as_slice();

        let grad = recover_gradient_nodal(&space, dofs);
        assert_eq!(grad.len(), 2); // dim = 2
        assert_eq!(grad[0].len(), space.mesh().n_nodes());

        for node in 0..space.mesh().n_nodes() {
            assert!(
                (grad[0][node] - 3.0).abs() < 1e-10,
                "node {node}: ∂u/∂x = {}, expected 3.0",
                grad[0][node]
            );
            assert!(
                (grad[1][node] + 2.0).abs() < 1e-10,
                "node {node}: ∂u/∂y = {}, expected -2.0",
                grad[1][node]
            );
        }
    }

    #[test]
    fn element_curl_hcurl_basic() {
        // Sanity check: compute element curls on a simple mesh, verify the
        // output has the right length.
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);
        let n = space.n_dofs();
        let dofs = vec![0.0; n]; // zero field → curl should be zero

        let curls = compute_element_curl(&space, &dofs);
        assert_eq!(curls.len(), space.mesh().n_elements());

        for (e, c) in curls.iter().enumerate() {
            assert_eq!(c.len(), 1, "2D curl should be scalar");
            assert!(
                c[0].abs() < 1e-12,
                "elem {e}: curl of zero field should be 0, got {}",
                c[0]
            );
        }
    }

    #[test]
    fn element_divergence_hdiv_basic() {
        // Sanity check: compute element divergences for zero field.
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = HDivSpace::new(mesh, 0);
        let n = space.n_dofs();
        let dofs = vec![0.0; n];

        let divs = compute_element_divergence(&space, &dofs);
        assert_eq!(divs.len(), space.mesh().n_elements());

        for (e, &d) in divs.iter().enumerate() {
            assert!(
                d.abs() < 1e-12,
                "elem {e}: div of zero field should be 0, got {d}"
            );
        }
    }

    // ── H1 error tests ────────────────────────────────────────────────────────

    #[test]
    fn h1_error_linear_function() {
        // u = 3x - 2y → ∇u = [3, -2] exactly for P1.
        // So H1 seminorm error should be essentially machine-zero.
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let v = space.interpolate(&|x| 3.0 * x[0] - 2.0 * x[1]);
        let dofs = v.as_slice();

        let err = compute_h1_error(&space, dofs, |_| vec![3.0, -2.0], 5);
        assert!(
            err < 1e-10,
            "H1 error for linear u should be ~0, got {err:.3e}"
        );
    }

    #[test]
    fn h1_error_poisson_p1_convergence() {
        // P1 Poisson: u = sin(πx)sin(πy), ∇u_exact = [π cos(πx)sin(πy), π sin(πx)cos(πy)]
        // H1 seminorm error should converge at O(h¹) for P1 (rate ≈ 1).
        use std::f64::consts::PI;
        use crate::assembler::Assembler;
        use crate::standard::{DiffusionIntegrator, DomainSourceIntegrator};
        use fem_space::constraints::{apply_dirichlet, boundary_dofs};

        let ns = [8usize, 16, 32];
        let mut prev: Option<(f64, f64)> = None;

        for &n in &ns {
            let mesh = Mesh::<2>::unit_square_tri(n);
            let space = H1Space::new(mesh, 1);
            let ndofs = space.n_dofs();

            let mut mat = Assembler::assemble_bilinear(
                &space,
                &[&DiffusionIntegrator { kappa: 1.0 }],
                3,
            );
            let src = DomainSourceIntegrator::new(|x: &[f64]| {
                2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
            });
            let mut rhs = Assembler::assemble_linear(&space, &[&src], 3);

            let dm = space.dof_manager();
            let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
            apply_dirichlet(&mut mat, &mut rhs, &bnd, &vec![0.0; bnd.len()]);

            let mut u = vec![0.0_f64; ndofs];
            fem_solver::solve_pcg_jacobi(&mat, &rhs, &mut u, &fem_solver::SolverConfig {
                rtol: 1e-12, max_iter: 10_000, verbose: false, ..fem_solver::SolverConfig::default()
            }).unwrap();

            let err = compute_h1_error(&space, &u, |x| {
                vec![
                    PI * (PI * x[0]).cos() * (PI * x[1]).sin(),
                    PI * (PI * x[0]).sin() * (PI * x[1]).cos(),
                ]
            }, 5);

            let h = 1.0 / n as f64;
            if let Some((e0, h0)) = prev {
                let rate = (err / e0).ln() / (h / h0).ln();
                assert!(
                    rate > 0.85,
                    "P1 H1 convergence rate = {rate:.3}, expected ≥ 0.85 (n={n})"
                );
            }
            prev = Some((err, h));
        }
    }

    #[test]
    fn h1_error_poisson_p2_convergence() {
        // P2 Poisson: H1 seminorm error should converge at O(h²) (rate ≈ 2).
        use std::f64::consts::PI;
        use crate::assembler::Assembler;
        use crate::standard::{DiffusionIntegrator, DomainSourceIntegrator};
        use fem_space::constraints::{apply_dirichlet, boundary_dofs};

        let ns = [4usize, 8, 16];
        let mut prev: Option<(f64, f64)> = None;

        for &n in &ns {
            let mesh = Mesh::<2>::unit_square_tri(n);
            let space = H1Space::new(mesh, 2);
            let ndofs = space.n_dofs();

            let mut mat = Assembler::assemble_bilinear(
                &space,
                &[&DiffusionIntegrator { kappa: 1.0 }],
                5,
            );
            let src = DomainSourceIntegrator::new(|x: &[f64]| {
                2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
            });
            let mut rhs = Assembler::assemble_linear(&space, &[&src], 5);

            let dm = space.dof_manager();
            let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
            apply_dirichlet(&mut mat, &mut rhs, &bnd, &vec![0.0; bnd.len()]);

            let mut u = vec![0.0_f64; ndofs];
            fem_solver::solve_pcg_jacobi(&mat, &rhs, &mut u, &fem_solver::SolverConfig {
                rtol: 1e-12, max_iter: 10_000, verbose: false, ..fem_solver::SolverConfig::default()
            }).unwrap();

            let err = compute_h1_error(&space, &u, |x| {
                vec![
                    PI * (PI * x[0]).cos() * (PI * x[1]).sin(),
                    PI * (PI * x[0]).sin() * (PI * x[1]).cos(),
                ]
            }, 7);

            let h = 1.0 / n as f64;
            if let Some((e0, h0)) = prev {
                let rate = (err / e0).ln() / (h / h0).ln();
                assert!(
                    rate > 1.8,
                    "P2 H1 convergence rate = {rate:.3}, expected ≥ 1.8 (n={n})"
                );
            }
            prev = Some((err, h));
        }
    }

    // ── Kelly indicator tests ─────────────────────────────────────────────────

    #[test]
    fn kelly_zero_for_linear_solution() {
        // u = 3x - 2y: P1 reproduces this exactly → constant gradient per element
        // → no gradient jump across any interior edge → all indicators = 0.
        use crate::InteriorFaceList;
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let v = space.interpolate(&|x| 3.0 * x[0] - 2.0 * x[1]);
        let dofs = v.as_slice();

        let ifl = InteriorFaceList::build(space.mesh());
        let indicators = compute_kelly_indicators(&space, dofs, &ifl);

        assert_eq!(indicators.len(), space.mesh().n_elements());
        for (e, &ind) in indicators.iter().enumerate() {
            assert!(
                ind.abs() < 1e-20,
                "elem {e}: Kelly indicator should be 0 for linear u, got {ind:.3e}"
            );
        }
    }

    #[test]
    fn kelly_nonzero_for_nonlinear_solution() {
        // For a Poisson solution on a coarse mesh, gradient jumps between adjacent
        // elements are non-zero → total indicators > 0.
        use crate::{Assembler, InteriorFaceList, standard::{DiffusionIntegrator, DomainSourceIntegrator}};
        use fem_space::constraints::{apply_dirichlet, boundary_dofs};
        use std::f64::consts::PI;

        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh.clone(), 1);
        let ndofs = space.n_dofs();

        let mut mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
        let src = DomainSourceIntegrator::new(|x: &[f64]| {
            2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
        });
        let mut rhs = Assembler::assemble_linear(&space, &[&src], 3);
        let dm = space.dof_manager();
        let bnd = boundary_dofs(&mesh, dm, &[1, 2, 3, 4]);
        apply_dirichlet(&mut mat, &mut rhs, &bnd, &vec![0.0; bnd.len()]);

        let mut u = vec![0.0_f64; ndofs];
        fem_solver::solve_pcg_jacobi(&mat, &rhs, &mut u, &fem_solver::SolverConfig {
            rtol: 1e-12, max_iter: 10_000, verbose: false,
            ..fem_solver::SolverConfig::default()
        }).unwrap();

        let ifl = InteriorFaceList::build(space.mesh());
        let indicators = compute_kelly_indicators(&space, &u, &ifl);

        assert_eq!(indicators.len(), space.mesh().n_elements());
        let total: f64 = indicators.iter().sum();
        assert!(total > 0.0, "total Kelly indicator should be > 0, got {total}");
    }

    // ── D614: the high-order-cell labels in the postprocess table ───────────

    /// D614: every hexahedral/wedge/pyramid cell label lands on its H¹
    /// family in the postprocess `ref_elem_vol` table (one CUBE / one wedge
    /// / one Fuentes pyramid family), so the element-wise kernels sample the
    /// slots `DofManager` numbers.
    #[test]
    fn d614_postprocess_ref_elem_vol_high_order_labels() {
        for et in [ElementType::Hex8, ElementType::Hex20, ElementType::Hex27] {
            assert_eq!(ref_elem_vol(et, 2).n_dofs(), 27, "{et:?}");
            assert_eq!(ref_elem_vol(et, 1).n_dofs(), 8, "{et:?}");
            assert!(is_iso_elem(et, 3), "{et:?} routes through the iso geometry");
        }
        for et in [ElementType::Prism6, ElementType::Prism15, ElementType::Prism18] {
            assert_eq!(ref_elem_vol(et, 2).n_dofs(), 18, "{et:?}");
            assert_eq!(ref_elem_vol(et, 1).n_dofs(), 6, "{et:?}");
            assert!(is_iso_elem(et, 3), "{et:?} routes through the iso geometry");
        }
        for et in [ElementType::Pyramid5, ElementType::Pyramid13] {
            assert_eq!(ref_elem_vol(et, 2).n_dofs(), 15, "{et:?}");
            assert_eq!(ref_elem_vol(et, 1).n_dofs(), 5, "{et:?}");
            assert!(is_iso_elem(et, 3), "{et:?} routes through the iso geometry");
        }
        // Frame anchors: hex centre at the origin, wedge centroid in the
        // (ξ segment)×(triangle) frame, pyramid centroid (½, ½, ¼).
        assert_eq!(ref_centroid(ElementType::Hex27, 3), vec![0.0, 0.0, 0.0]);
        assert_eq!(
            ref_centroid(ElementType::Prism18, 3),
            vec![0.5, 1.0 / 3.0, 1.0 / 3.0]
        );
        assert_eq!(ref_centroid(ElementType::Pyramid13, 3), vec![0.5, 0.5, 0.25]);
    }
}

// ─── D202: high-order tri/tet table arms ────────────────────────────────────
#[cfg(test)]
mod d202_high_order_tables {
    //! The local table must hand back, at every order, the element whose slots
    //! pair with the H¹ space's DOF numbering — i.e. the assembler's space
    //! dispatch choice (`ref_elem_vol_for_space`'s H1 branch,
    //! `assembler.rs::ref_elem_vol_h1`).  Before D202, orders >= 4 fell into
    //! the fail-fast panic arm.

    use super::ref_elem_vol;
    use crate::assembler::ref_elem_vol_h1;
    use fem_mesh::element_type::ElementType;

    #[test]
    fn tri_tet_o4_to_o6_match_space_slots() {
        for o in 4..=6u8 {
            let local = ref_elem_vol(ElementType::Tri3, o);
            let space = ref_elem_vol_h1(ElementType::Tri3, o);
            assert_eq!(local.n_dofs(), space.n_dofs(), "tri n_dofs at order {o}");
            assert_eq!(
                local.dof_coords(),
                space.dof_coords(),
                "tri dof_coords at order {o}"
            );
            let mut phi = vec![0.0; local.n_dofs()];
            local.eval_basis(&[0.3, 0.2], &mut phi);
            let sum: f64 = phi.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "tri partition of unity at order {o}: {sum}");

            let local = ref_elem_vol(ElementType::Tet4, o);
            let space = ref_elem_vol_h1(ElementType::Tet4, o);
            assert_eq!(local.n_dofs(), space.n_dofs(), "tet n_dofs at order {o}");
            assert_eq!(
                local.dof_coords(),
                space.dof_coords(),
                "tet dof_coords at order {o}"
            );
            let mut phi = vec![0.0; local.n_dofs()];
            local.eval_basis(&[0.25, 0.3, 0.2], &mut phi);
            let sum: f64 = phi.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "tet partition of unity at order {o}: {sum}");
        }
    }
}
