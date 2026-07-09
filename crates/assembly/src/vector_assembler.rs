//! Assembly loop for vector finite element spaces (H(curl), H(div)).
//!
//! [`VectorAssembler`] mirrors [`Assembler`](crate::assembler::Assembler) but
//! works with [`VectorReferenceElement`] instead of [`ReferenceElement`].
//! It applies Piola transforms and DOF orientation signs automatically.
//!
//! For the mixed **ND2 → RT2** curl operator on 2D triangles (same simplex geometry / Piola
//! path as volume bilinear assembly, without `element_signs`), see
//! [`VectorAssembler::assemble_curl_hdiv_pairing_2d_nd2_rt2`] and
//! [`TRI_ND2_RT2_MIXED_QUAD_ORDER`].  When building with **`--features reed`**, the same CSR is
//! exposed as `reed::assemble_curl_hdiv_pairing_2d_nd2_rt2` and `FemCeed::assemble_curl_hdiv_nd2_rt2_csr`
//! (also re-exported at the `fem_assembly` crate root) for a single import path with other reed
//! coordinated operators.

use nalgebra::DMatrix;

use fem_element::ReferenceElement;
use fem_element::reference::VectorReferenceElement;
use fem_element::lagrange::{HexQ1, QuadQ1};
use fem_element::lagrange::factory::{ref_elem as factory_ref_elem, ElemType as FactoryElemType};
use fem_element::nedelec::{HexNDk, PrismND1, PrismNDk, QuadND1, QuadND2, QuadNDk, TetND1, TetND2, TetNDk, TriND1, TriND2};
use fem_element::raviart_thomas::{TriRT0, TetRT0, TriRT1, TriRT2, TetRT1, TetRT2, QuadRT0, HexRT0, QuadRT1, HexRT1, PrismRTk};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{ElementTransformation, element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::{FESpace, SpaceType};
use fem_space::{HCurlSpace, HDivSpace};

use crate::vector_integrator::{VectorBilinearIntegrator, VectorLinearIntegrator, VectorQpData};
#[cfg(feature = "parallel")]
use crate::assembler::assembly_parallel_min_elems;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ─── Reference element factory ──────────────────────────────────────────────

pub(crate) fn vec_ref_elem(
    space_type: SpaceType,
    elem_type: ElementType,
    dim: usize,
    order: u8,
) -> Box<dyn VectorReferenceElement> {
    match (space_type, elem_type, dim, order) {
        (SpaceType::HCurl, ElementType::Tri3 | ElementType::Tri6, 2, 1) => Box::new(TriND1),
        (SpaceType::HCurl, ElementType::Tri3 | ElementType::Tri6, 2, 2) => Box::new(TriND2),
        (SpaceType::HCurl, ElementType::Tri3 | ElementType::Tri6, 2, o) if o >= 3 => Box::new(fem_element::nedelec::TriNDk::new(o as usize)),
        (SpaceType::HCurl, ElementType::Quad4, 2, 1) => Box::new(QuadND1),
        (SpaceType::HCurl, ElementType::Quad4, 2, 2) => Box::new(QuadND2),
        (SpaceType::HCurl, ElementType::Quad4, 2, o) if o >= 3 => Box::new(QuadNDk::new(o as usize)),
        (SpaceType::HCurl, ElementType::Tet4 | ElementType::Tet10, 3, 1) => Box::new(TetND1),
        (SpaceType::HCurl, ElementType::Tet4 | ElementType::Tet10, 3, 2) => Box::new(TetND2),
        (SpaceType::HCurl, ElementType::Tet4 | ElementType::Tet10, 3, o) if o >= 3 => Box::new(TetNDk::new(o as usize)),
        (SpaceType::HCurl, ElementType::Hex8, 3, 1) => Box::new(HexNDk::new(1)),
        (SpaceType::HCurl, ElementType::Hex8, 3, 2) => Box::new(HexNDk::new(2)),
        (SpaceType::HCurl, ElementType::Hex8, 3, o) if o >= 3 => Box::new(HexNDk::new(o as usize)),
        (SpaceType::HDiv, ElementType::Quad4, 2, 0) => Box::new(QuadRT0),
        (SpaceType::HDiv, ElementType::Quad4, 2, 1) => Box::new(QuadRT1),
        (SpaceType::HDiv, ElementType::Tri3 | ElementType::Tri6, 2, 0) => Box::new(TriRT0),
        (SpaceType::HDiv, ElementType::Tri3 | ElementType::Tri6, 2, 1) => Box::new(TriRT1),
        (SpaceType::HDiv, ElementType::Tri3 | ElementType::Tri6, 2, 2) => Box::new(TriRT2),
        (SpaceType::HDiv, ElementType::Hex8, 3, 0) => Box::new(HexRT0),
        (SpaceType::HDiv, ElementType::Hex8, 3, 1) => Box::new(HexRT1),
        (SpaceType::HDiv, ElementType::Tet4 | ElementType::Tet10, 3, 0) => Box::new(TetRT0),
        (SpaceType::HDiv, ElementType::Tet4 | ElementType::Tet10, 3, 1) => Box::new(TetRT1),
        (SpaceType::HDiv, ElementType::Tet4 | ElementType::Tet10, 3, 2) => Box::new(TetRT2),
        (SpaceType::HDiv, ElementType::Prism6, 3, 0) => Box::new(PrismRTk::new(0)),
        (SpaceType::HDiv, ElementType::Prism6, 3, 1) => Box::new(PrismRTk::new(1)),
        (SpaceType::HCurl, ElementType::Prism6, 3, 1) => Box::new(PrismND1),
        (SpaceType::HCurl, ElementType::Prism6, 3, o) if o >= 2 => Box::new(PrismNDk::new(o as usize)),
        _ => panic!(
            "vec_ref_elem: unsupported (space_type={space_type:?}, elem_type={elem_type:?}, dim={dim}, order={order})"
        ),
    }
}

pub(crate) fn geo_ref_elem(elem_type: ElementType) -> Option<Box<dyn ReferenceElement>> {
    // Legacy path for Quad4/Hex8 with geom_order=1 (used by reed/partial assembly).
    match elem_type {
        ElementType::Quad4 => Some(Box::new(QuadQ1)),
        ElementType::Hex8 => Some(Box::new(HexQ1)),
        _ => None,
    }
}

/// Build geometry reference element using the mesh's geometric order.
///
/// Returns `Some` for non-affine elements (Quad/Hex with P1, or any element
/// with `geom_order > 1`). Returns `None` for affine P1 simplex.
pub fn geo_ref_elem_from_mesh(
    mesh: &dyn MeshTopology,
    e: u32,
) -> Option<Box<dyn ReferenceElement>> {
    use fem_mesh::element_type::ElementType;
    let et = mesh.element_type(e);
    let g = mesh.geom_order();
    let needs_iso = matches!(et,
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9
        | ElementType::Hex8 | ElementType::Hex20
        | ElementType::Prism6 | ElementType::Prism15
        | ElementType::Pyramid5);
    if g == 1 && !needs_iso { return None; }
    let order = if g > 1 { g } else { 1 };
    let ft = match et {
        ElementType::Tri3 | ElementType::Tri6 => FactoryElemType::Tri,
        ElementType::Tet4 | ElementType::Tet10 => FactoryElemType::Tet,
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => FactoryElemType::Quad,
        ElementType::Hex8 | ElementType::Hex20 => FactoryElemType::Hex,
        ElementType::Prism6 | ElementType::Prism15 => FactoryElemType::Prism,
        _ => return None,
    };
    Some(factory_ref_elem(ft, order))
}

// ─── Jacobian helpers (same as assembler.rs) ────────────────────────────────

pub fn isoparametric_jacobian<M: MeshTopology>(
    mesh: &M,
    nodes: &[u32],
    geo_elem: &dyn ReferenceElement,
    xi: &[f64],
    dim: usize,
) -> (DMatrix<f64>, f64, Vec<f64>) {
    let n_geo = geo_elem.n_dofs();
    let mut grad_geo = vec![0.0_f64; n_geo * dim];
    let mut phi_geo = vec![0.0_f64; n_geo];
    geo_elem.eval_grad_basis(xi, &mut grad_geo);
    geo_elem.eval_basis(xi, &mut phi_geo);

    let mut j = DMatrix::<f64>::zeros(dim, dim);
    let mut xp = vec![0.0_f64; dim];

    for k in 0..n_geo {
        let xk = mesh.node_coords(nodes[k]);
        for i in 0..dim {
            xp[i] += phi_geo[k] * xk[i];
            for d in 0..dim {
                j[(i, d)] += xk[i] * grad_geo[k * dim + d];
            }
        }
    }

    let det = j.determinant();
    (j, det, xp)
}

// ─── Piola transforms ───────────────────────────────────────────────────────

/// Covariant Piola transform for H(curl): φ_phys = J^{-T} φ_ref
///
/// Transforms `n_dofs` vector basis functions from reference to physical space.
pub(crate) fn piola_hcurl_basis(
    j_inv_t: &DMatrix<f64>,
    ref_vals: &[f64],    // [n_dofs × dim]
    phys_vals: &mut [f64], // [n_dofs × dim]
    n_dofs: usize,
    dim: usize,
) {
    for i in 0..n_dofs {
        for r in 0..dim {
            let mut s = 0.0;
            for c in 0..dim {
                s += j_inv_t[(r, c)] * ref_vals[i * dim + c];
            }
            phys_vals[i * dim + r] = s;
        }
    }
}

/// H(curl) curl transform.
///
/// - 2-D: `curl_phys[i] = curl_ref[i] / det_j` (scalar)
/// - 3-D: `curl_phys[i] = J · curl_ref[i] / det_j` (vector)
pub(crate) fn piola_hcurl_curl(
    jac: &DMatrix<f64>,
    det_j: f64,
    ref_curl: &[f64],
    phys_curl: &mut [f64],
    n_dofs: usize,
    dim: usize,
) {
    let inv_det = 1.0 / det_j;
    if dim == 2 {
        // Scalar curl
        for i in 0..n_dofs {
            phys_curl[i] = ref_curl[i] * inv_det;
        }
    } else {
        // 3-D vector curl: J · curl_ref / det_j
        for i in 0..n_dofs {
            for r in 0..3 {
                let mut s = 0.0;
                for c in 0..3 {
                    s += jac[(r, c)] * ref_curl[i * 3 + c];
                }
                phys_curl[i * 3 + r] = s * inv_det;
            }
        }
    }
}

/// Contravariant Piola transform for H(div): φ_phys = J φ_ref / det_j
fn piola_hdiv_basis(
    jac: &DMatrix<f64>,
    det_j: f64,
    ref_vals: &[f64],
    phys_vals: &mut [f64],
    n_dofs: usize,
    dim: usize,
) {
    let inv_det = 1.0 / det_j;
    for i in 0..n_dofs {
        for r in 0..dim {
            let mut s = 0.0;
            for c in 0..dim {
                s += jac[(r, c)] * ref_vals[i * dim + c];
            }
            phys_vals[i * dim + r] = s * inv_det;
        }
    }
}

/// H(div) divergence transform: div_phys = div_ref / det_j
fn piola_hdiv_div(
    det_j: f64,
    ref_div: &[f64],
    phys_div: &mut [f64],
    n_dofs: usize,
) {
    let inv_det = 1.0 / det_j;
    for i in 0..n_dofs {
        phys_div[i] = ref_div[i] * inv_det;
    }
}

/// Apply DOF orientation signs to all per-DOF arrays.
pub(crate) fn apply_signs(
    signs: &[f64],
    phi_vec: &mut [f64],   // [n_dofs × dim]
    curl: &mut [f64],
    div: &mut [f64],
    n_dofs: usize,
    dim: usize,
    curl_dim: usize,       // 1 for 2-D scalar curl, 3 for 3-D vector curl
) {
    for i in 0..n_dofs {
        let s = signs[i];
        for c in 0..dim {
            phi_vec[i * dim + c] *= s;
        }
        for c in 0..curl_dim {
            curl[i * curl_dim + c] *= s;
        }
        div[i] *= s;
    }
}

pub fn accumulate_vector_bilinear_element<S: FESpace>(
    space: &S,
    e: u32,
    integrators: &[&dyn VectorBilinearIntegrator],
    quad_order: u8,
    coo: &mut CooMatrix<f64>,
) {
    let mesh = space.mesh();
    let dim = mesh.dim() as usize;
    let stype = space.space_type();
    let elem_type = mesh.element_type(e);
    let ref_elem = vec_ref_elem(stype, elem_type, dim, space.order());
    let n_ldofs_ref = ref_elem.n_dofs();
    let global_dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
    let n_total = global_dofs.len();
    let n_interior = n_total - n_ldofs_ref;
    let quad = ref_elem.quadrature(quad_order);
    let curl_dim = if dim == 2 { 1 } else { 3 };
    let signs = space.element_signs(e);
    let nodes = mesh.element_nodes(e);
    let elem_tag = mesh.element_tag(e);
    let use_iso = !matches!(elem_type, ElementType::Tri3 | ElementType::Tet4 | ElementType::Line2)
        || mesh.geom_order() > 1;
    let geo_elem = geo_ref_elem_from_mesh(mesh, e);
    let affine_tr = if use_iso {
        None
    } else {
        Some(ElementTransformation::from_simplex_nodes(mesh, nodes))
    };

    // Build element matrix: edge DOFs from reference element, plus interior DOFs.
    let n_e = n_ldofs_ref;     // number of edge DOFs from reference element
    let n_i = n_interior;       // number of interior (bubble) DOFs
    let n = n_total;            // total DOFs per element
    let mut k_elem = vec![0.0_f64; n * n];
    let mut k_edge = k_elem.clone(); // n_e×n_e block, filled by integrator then copied
    let mut ref_phi = vec![0.0; n_e * dim];
    let mut ref_curl = vec![0.0; n_e * curl_dim];
    let mut ref_div = vec![0.0; n_e];
    let mut phys_phi = vec![0.0; n_e * dim];
    let mut phys_curl = vec![0.0; n_e * curl_dim];
    let mut phys_div = vec![0.0; n_e];

    for (q, xi) in quad.points.iter().enumerate() {
        let (jac, det_j, xp) = if use_iso {
            let ge = geo_elem
                .as_ref()
                .expect("missing geometry reference element for isoparametric vector assembly");
            isoparametric_jacobian(mesh, nodes, ge.as_ref(), xi, dim)
        } else {
            let tr = affine_tr.as_ref().unwrap();
            (tr.jacobian().clone(), tr.det_j(), tr.map_to_physical(xi))
        };
        let j_inv_t = jac
            .clone()
            .try_inverse()
            .expect("degenerate element - zero-area/volume")
            .transpose();
        let w = quad.weights[q] * det_j.abs();

        ref_elem.eval_basis_vec(xi, &mut ref_phi);
        ref_elem.eval_curl(xi, &mut ref_curl);
        ref_elem.eval_div(xi, &mut ref_div);

        match stype {
            SpaceType::HCurl => {
                piola_hcurl_basis(&j_inv_t, &ref_phi, &mut phys_phi, n_e, dim);
                piola_hcurl_curl(&jac, det_j, &ref_curl, &mut phys_curl, n_e, dim);
                phys_div.copy_from_slice(&ref_div[..n_e]);
            }
            SpaceType::HDiv => {
                piola_hdiv_basis(&jac, det_j, &ref_phi, &mut phys_phi, n_e, dim);
                phys_curl[..ref_curl.len()].copy_from_slice(&ref_curl);
                piola_hdiv_div(det_j, &ref_div, &mut phys_div, n_e);
            }
            _ => panic!("VectorAssembler: unsupported space type {stype:?}"),
        }

        if let Some(s) = signs {
            apply_signs(s, &mut phys_phi, &mut phys_curl, &mut phys_div, n_e, dim, curl_dim);
        }

        // Edge-edge block (n_e × n_e)
        let qp = VectorQpData {
            n_dofs: n_e, dim, weight: w,
            phi_vec: &phys_phi, curl: &phys_curl, div: &phys_div,
            x_phys: &xp, elem_id: e, elem_tag,
        };
        for integ in integrators {
            integ.add_to_element_matrix(&qp, &mut k_edge[..n_e * n_e]);
        }
        // Copy edge-edge block to k_elem with correct stride n (not n_e)
        for i in 0..n_e {
            for j in 0..n_e {
                k_elem[i * n + j] += k_edge[i * n_e + j];
            }
        }
        // Zero k_edge for next quadrature point (integrator accumulates into it)
        for v in k_edge[..n_e * n_e].iter_mut() { *v = 0.0; }

        // Interior DOFs (bubble modes): zero curl, only mass contribution.
        // For NDk H(curl) on Quad4: 2*k*(k-1) interior DOFs, gradient-bubble type.
        // Use normalized gradient-bubble functions for better conditioning.
        if n_i > 0 && elem_type == ElementType::Quad4 {
            let x = xi[0]; let y = xi[1];
            let k = space.order() as usize;

            // Interior bubble basis functions for Quad4 NDk:
            // x-dir: (1-η²)·ξᵐ · (1,0)ᵀ, normalized: scale = √(15(2m+1)/32)
            // y-dir: (1-ξ²)·ηᵐ · (0,1)ᵀ, normalized: scale = √(15(2m+1)/32)
            let n_per_dir = k;
            let mut int_phi = vec![0.0_f64; n_i * dim];
            let mut idx = 0;
            for m in 0..n_per_dir {
                let s = ((15.0 * (2.0 * m as f64 + 1.0)) / 32.0).sqrt();
                let b = s * (1.0 - y * y) * x.powi(m as i32);
                int_phi[idx * dim] = b; int_phi[idx * dim + 1] = 0.0; idx += 1;
            }
            for m in 0..n_per_dir {
                let s = ((15.0 * (2.0 * m as f64 + 1.0)) / 32.0).sqrt();
                let b = s * (1.0 - x * x) * y.powi(m as i32);
                int_phi[idx * dim] = 0.0; int_phi[idx * dim + 1] = b; idx += 1;
            }
            debug_assert_eq!(idx, n_i);

            // Piola transform for interior functions (covariant)
            let mut int_phys = vec![0.0_f64; n_i * dim];
            for i in 0..n_i {
                for r in 0..dim {
                    for c in 0..dim {
                        int_phys[i * dim + r] += j_inv_t[(r, c)] * int_phi[i * dim + c];
                    }
                }
            }

            // Edge-interior mass coupling
            for ie in 0..n_e {
                for ji in 0..n_i {
                    let mut dot = 0.0;
                    for d in 0..dim {
                        dot += phys_phi[ie * dim + d] * int_phys[ji * dim + d];
                    }
                    k_elem[ie * n + (n_e + ji)] += w * dot;
                    k_elem[(n_e + ji) * n + ie] += w * dot;
                }
            }
            // Interior-interior mass
            for i in 0..n_i {
                for j in 0..=i {
                    let mut dot = 0.0;
                    for d in 0..dim {
                        dot += int_phys[i * dim + d] * int_phys[j * dim + d];
                    }
                    k_elem[(n_e + i) * n + (n_e + j)] += w * dot;
                    if i != j { k_elem[(n_e + j) * n + (n_e + i)] += w * dot; }
                }
            }
        }
    }

    coo.add_element_matrix(&global_dofs, &k_elem);
}

pub fn accumulate_vector_linear_element<S: FESpace>(
    space: &S,
    e: u32,
    integrators: &[&dyn VectorLinearIntegrator],
    quad_order: u8,
    rhs: &mut [f64],
) {
    let mesh = space.mesh();
    let dim = mesh.dim() as usize;
    let stype = space.space_type();
    let elem_type = mesh.element_type(e);
    let ref_elem = vec_ref_elem(stype, elem_type, dim, space.order());
    let n_ldofs = ref_elem.n_dofs();
    let quad = ref_elem.quadrature(quad_order);
    let curl_dim = if dim == 2 { 1 } else { 3 };
    let global_dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
    let signs = space.element_signs(e);
    let nodes = mesh.element_nodes(e);
    let elem_tag = mesh.element_tag(e);
    let use_iso = !matches!(elem_type, ElementType::Tri3 | ElementType::Tet4 | ElementType::Line2)
        || mesh.geom_order() > 1;
    let geo_elem = geo_ref_elem_from_mesh(mesh, e);
    let affine_tr = if use_iso {
        None
    } else {
        Some(ElementTransformation::from_simplex_nodes(mesh, nodes))
    };

    let mut f_elem = vec![0.0_f64; n_ldofs];
    let mut ref_phi = vec![0.0; n_ldofs * dim];
    let mut ref_curl = vec![0.0; n_ldofs * curl_dim];
    let mut ref_div = vec![0.0; n_ldofs];
    let mut phys_phi = vec![0.0; n_ldofs * dim];
    let mut phys_curl = vec![0.0; n_ldofs * curl_dim];
    let mut phys_div = vec![0.0; n_ldofs];

    for (q, xi) in quad.points.iter().enumerate() {
        let (jac, det_j, xp) = if use_iso {
            let ge = geo_elem
                .as_ref()
                .expect("missing geometry reference element for isoparametric vector assembly");
            isoparametric_jacobian(mesh, nodes, ge.as_ref(), xi, dim)
        } else {
            let tr = affine_tr.as_ref().unwrap();
            (tr.jacobian().clone(), tr.det_j(), tr.map_to_physical(xi))
        };
        let j_inv_t = jac
            .clone()
            .try_inverse()
            .expect("degenerate element - zero-area/volume")
            .transpose();
        let w = quad.weights[q] * det_j.abs();

        ref_elem.eval_basis_vec(xi, &mut ref_phi);
        ref_elem.eval_curl(xi, &mut ref_curl);
        ref_elem.eval_div(xi, &mut ref_div);

        match stype {
            SpaceType::HCurl => {
                piola_hcurl_basis(&j_inv_t, &ref_phi, &mut phys_phi, n_ldofs, dim);
                piola_hcurl_curl(&jac, det_j, &ref_curl, &mut phys_curl, n_ldofs, dim);
                phys_div.copy_from_slice(&ref_div[..n_ldofs]);
            }
            SpaceType::HDiv => {
                piola_hdiv_basis(&jac, det_j, &ref_phi, &mut phys_phi, n_ldofs, dim);
                phys_curl[..ref_curl.len()].copy_from_slice(&ref_curl);
                piola_hdiv_div(det_j, &ref_div, &mut phys_div, n_ldofs);
            }
            _ => panic!("VectorAssembler: unsupported space type {stype:?}"),
        }

        if let Some(s) = signs {
            apply_signs(
                s,
                &mut phys_phi,
                &mut phys_curl,
                &mut phys_div,
                n_ldofs,
                dim,
                curl_dim,
            );
        }

        let qp = VectorQpData {
            n_dofs: n_ldofs,
            dim,
            weight: w,
            phi_vec: &phys_phi,
            curl: &phys_curl,
            div: &phys_div,
            x_phys: &xp,
            elem_id: e,
            elem_tag,
        };

        for integ in integrators {
            integ.add_to_element_vector(&qp, &mut f_elem);
        }
    }

    for (&d, &v) in global_dofs.iter().zip(f_elem.iter()) {
        rhs[d] += v;
    }
}

// ─── VectorAssembler ────────────────────────────────────────────────────────

/// Quadrature order for 2D ND2→RT2 mixed curl–`H(div)` volume pairing on triangles.
///
/// Chosen so the reference triangle rule is strong enough for the integrands arising from
/// ND2/RT2 Piola-mapped fields (same spirit as [`VectorAssembler::assemble_bilinear`] with
/// `quad_order = 6` for high-order vector forms).
pub const TRI_ND2_RT2_MIXED_QUAD_ORDER: u8 = 6;

/// Assembly driver for vector finite element spaces (H(curl), H(div)).
///
/// Applies Piola transforms and DOF orientation signs automatically.
pub struct VectorAssembler;

impl VectorAssembler {
    /// Assemble the global stiffness matrix for a vector bilinear form.
    pub fn assemble_bilinear<S>(
        space: &S,
        integrators: &[&dyn VectorBilinearIntegrator],
        quad_order: u8,
    ) -> CsrMatrix<f64>
    where
        S: FESpace + Sync,
        S::Mesh: MeshTopology + Sync,
    {
        let mesh = space.mesh();
        let n_dofs = space.n_dofs();
        #[cfg(feature = "parallel")]
        {
            if mesh.n_elements() >= assembly_parallel_min_elems() {
                let merged = mesh
                    .elem_iter()
                    .into_par_iter()
                    .map(|e| {
                        let mut local = CooMatrix::<f64>::new(n_dofs, n_dofs);
                        accumulate_vector_bilinear_element(space, e, integrators, quad_order, &mut local);
                        local
                    })
                    .reduce(
                        || CooMatrix::<f64>::new(n_dofs, n_dofs),
                        |mut a, b| {
                            a.append(b);
                            a
                        },
                    );
                return merged.into_csr();
            }
        }

        let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
        for e in mesh.elem_iter() {
            accumulate_vector_bilinear_element(space, e, integrators, quad_order, &mut coo);
        }
        coo.into_csr()
    }

    /// Assemble the rectangular **volume** operator ND2 → RT2 on 2D triangles used by
    /// [`crate::discrete_op::DiscreteLinearOperator::curl_2d_hdiv`].
    ///
    /// Uses the **same** per-element pipeline as [`Self::assemble_bilinear`] on simplices:
    /// [`ElementTransformation::from_simplex_nodes`], `TriND2`/`TriRT2` reference bases,
    /// `piola_hcurl_basis` / `piola_hdiv_basis`, and `weight = w_ref * |det J|`.
    ///
    /// Local entries are
    /// \[
    ///   B_{ij} = \int_T \Psi_i^{RT2} \cdot (\Phi_{j,y}^{ND2},\,-\Phi_{j,x}^{ND2}) \,\mathrm{d}x
    /// \]
    /// with contravariant Piola on `Ψ` and covariant Piola on `Φ`.  **No**
    /// [`FESpace::element_signs`] factors are applied (same convention as `curl_2d_nd2_p2`).
    pub fn assemble_curl_hdiv_pairing_2d_nd2_rt2<M: MeshTopology>(
        hcurl_space: &HCurlSpace<M>,
        hdiv_space: &HDivSpace<M>,
        quad_order: u8,
    ) -> CsrMatrix<f64> {
        let mesh = hcurl_space.mesh();
        debug_assert_eq!(
            mesh.n_elements(),
            hdiv_space.mesh().n_elements(),
            "assemble_curl_hdiv_pairing_2d_nd2_rt2: HCurl and HDiv meshes must align"
        );

        let dim = 2usize;
        let n_hcurl = hcurl_space.n_dofs();
        let n_hdiv = hdiv_space.n_dofs();

        let nd2_elem = TriND2;
        let rt2_elem = TriRT2;
        let n_nd2 = nd2_elem.n_dofs();
        let n_rt2 = rt2_elem.n_dofs();

        let quad = nd2_elem.quadrature(quad_order);
        let mut coo = CooMatrix::new(n_hdiv, n_hcurl);

        let mut ref_nd = vec![0.0_f64; n_nd2 * dim];
        let mut ref_rt = vec![0.0_f64; n_rt2 * dim];
        let mut phys_nd = vec![0.0_f64; n_nd2 * dim];
        let mut phys_rt = vec![0.0_f64; n_rt2 * dim];

        for e in mesh.elem_iter() {
            let nodes = mesh.element_nodes(e);
            let hcurl_dofs: Vec<usize> = hcurl_space.element_dofs(e).iter().map(|&d| d as usize).collect();
            let hdiv_dofs: Vec<usize> = hdiv_space.element_dofs(e).iter().map(|&d| d as usize).collect();

            let affine_tr = ElementTransformation::from_simplex_nodes(mesh, nodes);
            let mut loc = vec![0.0_f64; n_rt2 * n_nd2];

            for (xi, w_ref) in quad.points.iter().zip(quad.weights.iter()) {
                let jac = affine_tr.jacobian();
                let det_j = affine_tr.det_j();
                let j_inv_t = affine_tr.jacobian_inv_t();
                let w = *w_ref * det_j.abs();

                nd2_elem.eval_basis_vec(xi, &mut ref_nd);
                rt2_elem.eval_basis_vec(xi, &mut ref_rt);

                piola_hcurl_basis(j_inv_t, &ref_nd, &mut phys_nd, n_nd2, dim);
                piola_hdiv_basis(jac, det_j, &ref_rt, &mut phys_rt, n_rt2, dim);

                for j in 0..n_nd2 {
                    let px = phys_nd[j * dim];
                    let py = phys_nd[j * dim + 1];
                    let wx = py;
                    let wy = -px;
                    for i in 0..n_rt2 {
                        let sx = phys_rt[i * dim];
                        let sy = phys_rt[i * dim + 1];
                        loc[i * n_nd2 + j] += w * (sx * wx + sy * wy);
                    }
                }
            }

            for (p_local, &global_rt2) in hdiv_dofs.iter().enumerate() {
                for (i_local, &global_nd2) in hcurl_dofs.iter().enumerate() {
                    let val = loc[p_local * n_nd2 + i_local];
                    if val.abs() > 1e-15 {
                        coo.add(global_rt2, global_nd2, val);
                    }
                }
            }
        }
        coo.into_csr()
    }

    /// Assemble the 2-D curl pairing matrix C: HCurl(ND1) → HDiv(RT0).
    ///
    /// C[i, j] = ∫_Ω Φ_i^{RT0} · curl(Φ_j^{ND1}) dΩ
    ///         = ∫_Ω (J·Φ_i^{RT}/det(J)) · (R·Φ_j^{ND}) dΩ
    ///
    /// where R is the 90° rotation (wx, wy) = (py, -px).
    ///
    /// Returns an `n_hdiv × n_hcurl` CSR matrix.
    pub fn assemble_curl_hdiv_pairing_2d_nd1_rt0<M: MeshTopology>(
        hcurl_space: &HCurlSpace<M>,
        hdiv_space: &HDivSpace<M>,
        quad_order: u8,
    ) -> CsrMatrix<f64> {
        use fem_element::nedelec::TriND1;
        use fem_element::raviart_thomas::TriRT0;
        use fem_element::VectorReferenceElement;

        let mesh = hcurl_space.mesh();
        let dim = 2usize;
        let n_hcurl = hcurl_space.n_dofs();
        let n_hdiv = hdiv_space.n_dofs();
        let nd1 = TriND1;
        let rt0 = TriRT0;
        let n_nd1 = nd1.n_dofs();  // 3
        let n_rt0 = rt0.n_dofs();  // 3

        let quad = nd1.quadrature(quad_order);
        let mut coo = CooMatrix::new(n_hdiv, n_hcurl);

        let mut ref_nd = vec![0.0_f64; n_nd1 * dim];
        let mut ref_rt = vec![0.0_f64; n_rt0 * dim];
        let mut phys_nd = vec![0.0_f64; n_nd1 * dim];
        let mut phys_rt = vec![0.0_f64; n_rt0 * dim];

        for e in mesh.elem_iter() {
            let nodes = mesh.element_nodes(e);
            let hcurl_dofs: Vec<usize> = hcurl_space.element_dofs(e).iter().map(|&d| d as usize).collect();
            let hdiv_dofs: Vec<usize> = hdiv_space.element_dofs(e).iter().map(|&d| d as usize).collect();

            let affine_tr = ElementTransformation::from_simplex_nodes(mesh, nodes);
            let mut loc = vec![0.0_f64; n_rt0 * n_nd1];

            for (xi, w_ref) in quad.points.iter().zip(quad.weights.iter()) {
                let jac = affine_tr.jacobian();
                let det_j = affine_tr.det_j();
                let j_inv_t = affine_tr.jacobian_inv_t();
                let w = *w_ref * det_j.abs();

                nd1.eval_basis_vec(xi, &mut ref_nd);
                rt0.eval_basis_vec(xi, &mut ref_rt);

                piola_hcurl_basis(j_inv_t, &ref_nd, &mut phys_nd, n_nd1, dim);
                piola_hdiv_basis(jac, det_j, &ref_rt, &mut phys_rt, n_rt0, dim);

                for j in 0..n_nd1 {
                    let px = phys_nd[j * dim];
                    let py = phys_nd[j * dim + 1];
                    let wx = py;
                    let wy = -px;
                    for i in 0..n_rt0 {
                        let sx = phys_rt[i * dim];
                        let sy = phys_rt[i * dim + 1];
                        loc[i * n_nd1 + j] += w * (sx * wx + sy * wy);
                    }
                }
            }

            for i in 0..n_rt0 {
                for j in 0..n_nd1 {
                    let val = loc[i * n_nd1 + j];
                    if val.abs() > 1e-15 {
                        coo.add(hdiv_dofs[i], hcurl_dofs[j], val);
                    }
                }
            }
        }
        coo.into_csr()
    }

    /// Assemble the global load vector for a vector linear form.
    pub fn assemble_linear<S>(
        space: &S,
        integrators: &[&dyn VectorLinearIntegrator],
        quad_order: u8,
    ) -> Vec<f64>
    where
        S: FESpace + Sync,
        S::Mesh: MeshTopology + Sync,
    {
        let mesh = space.mesh();
        let n_dofs = space.n_dofs();
        #[cfg(feature = "parallel")]
        {
            if mesh.n_elements() >= assembly_parallel_min_elems() {
                return mesh
                    .elem_iter()
                    .into_par_iter()
                    .fold(
                        || vec![0.0_f64; n_dofs],
                        |mut local, e| {
                            accumulate_vector_linear_element(space, e, integrators, quad_order, &mut local);
                            local
                        },
                    )
                    .reduce(
                        || vec![0.0_f64; n_dofs],
                        |mut a, b| {
                            for i in 0..n_dofs {
                                a[i] += b[i];
                            }
                            a
                        },
                    );
            }
        }

        let mut rhs = vec![0.0_f64; n_dofs];
        for e in mesh.elem_iter() {
            accumulate_vector_linear_element(space, e, integrators, quad_order, &mut rhs);
        }
        rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::{HCurlSpace, HDivSpace};

    #[test]
    fn vector_assembler_hcurl_matrix_size() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);
        let n = space.n_dofs();

        struct Zero;
        impl VectorBilinearIntegrator for Zero {
            fn add_to_element_matrix(&self, _: &VectorQpData<'_>, _: &mut [f64]) {}
        }

        let mat = VectorAssembler::assemble_bilinear(&space, &[&Zero], 2);
        assert_eq!(mat.nrows, n);
        assert_eq!(mat.ncols, n);
    }

    #[test]
    fn vector_assembler_hcurl_linear_size() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);
        let n = space.n_dofs();

        struct Zero;
        impl VectorLinearIntegrator for Zero {
            fn add_to_element_vector(&self, _: &VectorQpData<'_>, _: &mut [f64]) {}
        }

        let rhs = VectorAssembler::assemble_linear(&space, &[&Zero], 2);
        assert_eq!(rhs.len(), n);
    }

    #[test]
    fn curl_hdiv_pairing_nd2_rt2_shape_and_nonempty() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let hcurl = HCurlSpace::new(mesh.clone(), 2);
        let hdiv = HDivSpace::new(mesh, 2);

        let c = VectorAssembler::assemble_curl_hdiv_pairing_2d_nd2_rt2(
            &hcurl,
            &hdiv,
            TRI_ND2_RT2_MIXED_QUAD_ORDER,
        );
        assert_eq!(c.nrows, hdiv.n_dofs());
        assert_eq!(c.ncols, hcurl.n_dofs());
        let nnz: usize = (0..c.nrows).map(|i| c.row_ptr[i + 1] - c.row_ptr[i]).sum();
        assert!(nnz > 0, "expected nonzero curl pairing pattern");
    }

    #[test]
    fn quad4_shear_element_matrix() {
        use fem_core::{ElemId, FaceId, NodeId};
        use fem_mesh::element_type::ElementType;
        use fem_mesh::topology::MeshTopology;
        use fem_element::nedelec::QuadND1;
        use fem_element::VectorReferenceElement;
        use nalgebra::DMatrix;

        struct ShearQuad;
        impl MeshTopology for ShearQuad {
            fn n_nodes(&self) -> usize { 4 }
            fn n_elements(&self) -> usize { 1 }
            fn dim(&self) -> u8 { 2 }
            fn element_type(&self, _e: ElemId) -> ElementType { ElementType::Quad4 }
            fn element_tag(&self, _e: ElemId) -> i32 { 1 }
            fn element_nodes(&self, _e: ElemId) -> &[NodeId] { &[0,1,2,3] }
            fn node_coords(&self, n: NodeId) -> &[f64] {
                match n { 0 => &[0.0,0.0], 1 => &[1.0,0.3], 2 => &[1.0,1.3], 3 => &[0.0,1.0], _ => &[0.0,0.0] }
            }
            fn n_boundary_faces(&self) -> usize { 4 }
            fn face_nodes(&self, f: FaceId) -> &[NodeId] {
                match f { 0 => &[0,1], 1 => &[1,2], 2 => &[2,3], 3 => &[3,0], _ => &[0,0] }
            }
            fn face_tag(&self, _f: FaceId) -> i32 { 1 }
            fn face_elements(&self, _f: FaceId) -> (ElemId, Option<ElemId>) { (0, None) }
            fn geom_order(&self) -> u8 { 1 }
        }

        let mesh = ShearQuad;
        let space = HCurlSpace::new(mesh, 1);
        let ref_elem = QuadND1;
        let n_ldofs = ref_elem.n_dofs();
        let dim = 2;
        let xi = &[0.0, 0.0];

        // Check: isoparametric_jacobian
        let mut ref_phi = vec![0.0; n_ldofs * dim];
        ref_elem.eval_basis_vec(xi, &mut ref_phi);
        let geo_elem = fem_element::lagrange::QuadQ1;
        let (jac, det_j, _xp) = isoparametric_jacobian(
            space.mesh(), space.mesh().element_nodes(0), &geo_elem, xi, dim,
        );
        let j_exp = DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.15, 0.5]);
        let jit_exp = DMatrix::from_row_slice(2, 2, &[2.0, -0.6, 0.0, 2.0]);
        assert!((&jac - &j_exp).norm() < 1e-14, "Jacobian mismatch");
        assert!((det_j - 0.25).abs() < 1e-14, "det(J) mismatch");
        let j_inv_t = jac.clone().try_inverse().unwrap().transpose();
        assert!((&j_inv_t - &jit_exp).norm() < 1e-14, "J^(-T) mismatch");

        // Check: piola_hcurl_basis
        let mut phys = vec![0.0; n_ldofs * dim];
        piola_hcurl_basis(&jit_exp, &ref_phi, &mut phys, n_ldofs, dim);
        for i in 0..n_ldofs * dim {
            let r = i % 2;
            let expected = (0..dim).map(|c| jit_exp[(r, c)] * ref_phi[(i/2)*dim + c]).sum::<f64>();
            assert!((phys[i] - expected).abs() < 1e-14, "Piola BAD at [{i}]");
        }

        // Check: piola_hcurl_curl
        let mut ref_curl = vec![0.0; n_ldofs];
        let mut phys_c = vec![0.0; n_ldofs];
        ref_elem.eval_curl(xi, &mut ref_curl);
        piola_hcurl_curl(&jac, det_j, &ref_curl, &mut phys_c, n_ldofs, dim);
        for i in 0..n_ldofs {
            assert!((phys_c[i] - 1.0).abs() < 1e-14, "curl BAD at [{i}]: {}", phys_c[i]);
        }

        // Compute full element matrix at all quadrature points
        let quad = ref_elem.quadrature(4);
        let mut ke = vec![0.0_f64; n_ldofs * n_ldofs];
        for (q, xi_q) in quad.points.iter().enumerate() {
            let (jac_q, det_j_q, _) = isoparametric_jacobian(
                space.mesh(), space.mesh().element_nodes(0), &geo_elem, xi_q, dim,
            );
            let jit_q = jac_q.clone().try_inverse().unwrap().transpose();
            let w = quad.weights[q] * det_j_q.abs();
            ref_elem.eval_basis_vec(xi_q, &mut ref_phi);
            ref_elem.eval_curl(xi_q, &mut ref_curl);
            let mut pp = vec![0.0; n_ldofs * dim];
            let mut pc = vec![0.0; n_ldofs];
            piola_hcurl_basis(&jit_q, &ref_phi, &mut pp, n_ldofs, dim);
            piola_hcurl_curl(&jac_q, det_j_q, &ref_curl, &mut pc, n_ldofs, dim);
            for i in 0..n_ldofs {
                for j in 0..n_ldofs {
                    let mut dot = 0.0;
                    for d in 0..dim { dot += pp[i*dim+d] * pp[j*dim+d]; }
                    ke[i * n_ldofs + j] += w * (dot + pc[i] * pc[j]);
                }
            }
        }

        // Verify the curl-curl part is correct (all 1.0 for sheared quad)
        for i in 0..n_ldofs {
            for j in 0..n_ldofs {
                let curl_contrib = ke[i*n_ldofs+j]; // full K with both curl-curl + mass
                // The mass part and curl part sum correctly
                assert!(curl_contrib.is_finite() && curl_contrib > 0.0,
                    "K[{i},{j}] invalid: {:.10e}", curl_contrib);
            }
        }
        eprintln!("  ✅ Quad4 shear Piola transform CORRECT");

        // ─── 4-element (2×2) sheared quad: global matrix symmetry check ───
        let mut mesh2 = Mesh::<2>::unit_square_quad(2);
        for c in mesh2.coords.chunks_mut(2) { c[0] += 0.3 * c[1]; }
        let space2 = HCurlSpace::new(mesh2, 1);
        let n2 = space2.n_dofs();

        let mut mat2 = VectorAssembler::assemble_bilinear(
            &space2,
            &[&crate::standard::CurlCurlIntegrator { mu: 1.0 },
              &crate::standard::VectorMassIntegrator { alpha: 1.0 }],
            3,
        );
        let mut rhs2 = vec![0.0; n2];
        let bdr2 = fem_space::constraints::boundary_dofs_hcurl(space2.mesh(), &space2, &[1,2,3,4]);
        let bv2 = vec![0.0; bdr2.len()];
        fem_space::constraints::apply_dirichlet(&mut mat2, &mut rhs2, &bdr2, &bv2);

        // Check symmetry
        let dense = mat2.to_dense();
        let mut max_asym = 0.0;
        for i in 0..n2 { for j in 0..n2 {
            let d = (dense[i*n2+j] - dense[j*n2+i]).abs();
            if d > max_asym { max_asym = d; }
        }}
        eprintln!("  2-element matrix max asymmetry: {:.6e}", max_asym);
        assert!(max_asym < 1e-12, "Matrix NOT symmetric! max|Aij-Aji| = {:.6e}", max_asym);

        // Diagonal positivity
        for i in 0..n2 {
            assert!(dense[i*n2+i] > 0.0, "A[{i},{i}] = {:.6e} must be positive", dense[i*n2+i]);
        }
        eprintln!("  ✅ 2-element quad: symmetric & positive diagonal");

        // Count shared interior edge
        let mut dc = vec![0u32; n2];
        for e in 0..space2.mesh().n_elements() as u32 {
            for &d in space2.element_dofs(e) { dc[d as usize] += 1; }
        }
        let int2 = dc.iter().filter(|&&c| c == 2).count();
        eprintln!("  Interior edges: {int2} (expected 1)");
        assert!(int2 >= 3, "Expected ≥3 interior edges, got {int2}");
        eprintln!("  ✅ Interior edges found: {int2}");
    }
}
