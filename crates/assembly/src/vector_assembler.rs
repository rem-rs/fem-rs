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
use fem_element::nedelec::{HexND1, HexND2, QuadND1, QuadND2, TetND1, TetND2, TriND1, TriND2};
use fem_element::raviart_thomas::{TriRT0, TetRT0, TriRT1, TriRT2, TetRT1, QuadRT0, HexRT0};
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
        (SpaceType::HCurl, ElementType::Quad4, 2, 1) => Box::new(QuadND1),
        (SpaceType::HCurl, ElementType::Quad4, 2, 2) => Box::new(QuadND2),
        (SpaceType::HCurl, ElementType::Tet4 | ElementType::Tet10, 3, 1) => Box::new(TetND1),
        (SpaceType::HCurl, ElementType::Tet4 | ElementType::Tet10, 3, 2) => Box::new(TetND2),
        (SpaceType::HCurl, ElementType::Hex8, 3, 1) => Box::new(HexND1),
        (SpaceType::HCurl, ElementType::Hex8, 3, 2) => Box::new(HexND2),
        (SpaceType::HDiv, ElementType::Quad4, 2, 0) => Box::new(QuadRT0),
        (SpaceType::HDiv, ElementType::Tri3 | ElementType::Tri6, 2, 0) => Box::new(TriRT0),
        (SpaceType::HDiv, ElementType::Tri3 | ElementType::Tri6, 2, 1) => Box::new(TriRT1),
        (SpaceType::HDiv, ElementType::Tri3 | ElementType::Tri6, 2, 2) => Box::new(TriRT2),
        (SpaceType::HDiv, ElementType::Hex8, 3, 0) => Box::new(HexRT0),
        (SpaceType::HDiv, ElementType::Tet4 | ElementType::Tet10, 3, 0) => Box::new(TetRT0),
        (SpaceType::HDiv, ElementType::Tet4 | ElementType::Tet10, 3, 1) => Box::new(TetRT1),
        (SpaceType::HDiv, _, 2, 0) => Box::new(QuadRT0),
        (SpaceType::HDiv, _, 2, 1) => Box::new(TriRT1),
        (SpaceType::HDiv, _, 2, 2) => Box::new(TriRT2),
        (SpaceType::HDiv, _, 3, 0) => Box::new(HexRT0),
        (SpaceType::HDiv, _, 3, 1) => Box::new(TetRT1),
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
pub(crate) fn geo_ref_elem_from_mesh(
    mesh: &dyn MeshTopology,
    e: u32,
) -> Option<Box<dyn ReferenceElement>> {
    use fem_mesh::element_type::ElementType;
    let et = mesh.element_type(e);
    let g = mesh.geom_order();
    let is_quad_hex = matches!(et,
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9
        | ElementType::Hex8 | ElementType::Hex20);
    if g == 1 && !is_quad_hex { return None; }
    let order = if g > 1 { g } else { 1 };
    let ft = match et {
        ElementType::Tri3 | ElementType::Tri6 => FactoryElemType::Tri,
        ElementType::Tet4 | ElementType::Tet10 => FactoryElemType::Tet,
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => FactoryElemType::Quad,
        ElementType::Hex8 | ElementType::Hex20 => FactoryElemType::Hex,
        _ => return None,
    };
    Some(factory_ref_elem(ft, order))
}

// ─── Jacobian helpers (same as assembler.rs) ────────────────────────────────

pub(crate) fn isoparametric_jacobian<M: MeshTopology>(
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

fn accumulate_vector_bilinear_element<S: FESpace>(
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

    let mut k_elem = vec![0.0_f64; n_ldofs * n_ldofs];
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
            integ.add_to_element_matrix(&qp, &mut k_elem);
        }
    }

    coo.add_element_matrix(&global_dofs, &k_elem);
}

fn accumulate_vector_linear_element<S: FESpace>(
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
    use fem_mesh::SimplexMesh;
    use fem_space::{HCurlSpace, HDivSpace};

    #[test]
    fn vector_assembler_hcurl_matrix_size() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(3);
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
}
