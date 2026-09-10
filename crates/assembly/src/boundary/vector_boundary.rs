//! Boundary assembly for vector finite element spaces (H(curl) / H(div)).
//!
//! Provides [`VectorBoundaryAssembler`] for integrating bilinear and linear
//! forms over boundary faces (edges in 2-D, triangular faces in 3-D) for
//! H(curl) Nédélec and H(div) Raviart-Thomas elements.
//!
//! ## H(curl) boundary integrals
//!
//! The principal use case is the **tangential mass** operator:
//!
//! ```text
//! a(u, v) = ∫_Γ γ (n×u)·(n×v) dS
//! ```
//!
//! ## H(div) boundary integrals
//!
//! ```text
//! a(u, v) = ∫_Γ g (u·n)(v·n) dS     (bilinear)
//! F(v)    = ∫_Γ g (v·n) dS          (linear / natural BC)
//! ```
//!
//! ## Quadrature-point data
//!
//! The assembler evaluates the volume H(curl) basis functions restricted to
//! the boundary face, applies the covariant Piola transform, and packages the
//! result in [`VectorBdQpData`] for the integrators.
//!
//! ## 2-D boundary (edge) integration
//!
//! In 2-D the boundary "face" is an edge.  The H(curl) basis functions are
//! 2-D vectors; their tangential trace on the edge is a scalar
//! `u_t = u · t̂`.  The tangential crossing `n×u` in 2-D is also a scalar
//! `(n×u) = u_x n_y − u_y n_x`.
//!
//! ## 3-D boundary (face) integration
//!
//! In 3-D the boundary "face" is a triangle.  The tangential part of a 3-D
//! vector on the face is `u_t = u − (u·n̂) n̂`.  The quantity `n×u` is the
//! tangential trace rotated by 90° in the face plane:
//! `n×u = n × u_t` (the `n × n × u` contribution is zero since `n × n = 0`).

use nalgebra::DMatrix;

use fem_element::ReferenceElement;
use fem_element::reference::VectorReferenceElement;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{ElementTransformation, element_type::ElementType};
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::fe_space::{FESpace, SpaceType};
use fem_space::{EdgeKey, HDivSpace};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use crate::assembler::assembly_parallel_min_elems;
use crate::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian, vec_ref_elem as vol_ref_elem};

// ─── Quadrature-point data ───────────────────────────────────────────────────

/// Data available to H(curl) boundary integrators at each face quadrature point.
///
/// All basis-function data is Piola-transformed (covariant) and sign-corrected
/// by the assembler.  Integrators see physical-space values.
#[derive(Debug)]
pub struct VectorBdQpData<'a> {
    /// Number of local H(curl) DOFs on the parent element.
    pub n_dofs: usize,
    /// Spatial dimension of the embedding space.
    pub dim: usize,
    /// Effective integration weight: quadrature weight × face Jacobian.
    pub weight: f64,
    /// Physical basis function values (Piola + sign), layout `phi_vec[i*dim + c]`.
    pub phi_vec: &'a [f64],
    /// Outward unit normal to the face, length `dim`.
    pub normal: &'a [f64],
    /// Physical coordinates of this face quadrature point, length `dim`.
    pub x_phys: &'a [f64],
    /// Owning volume element id.
    pub elem_id: u32,
    /// Element material / region tag.
    pub elem_tag: i32,
}

// ─── Integrator traits ───────────────────────────────────────────────────────

/// Accumulate a bilinear-form contribution over a boundary face for H(curl).
///
/// `k_face` is row-major with shape `[n_dofs × n_dofs]`.
/// Implementations must **add** their contribution (not overwrite).
pub trait VectorBoundaryBilinearIntegrator: Send + Sync {
    fn add_to_face_matrix(&self, qp: &VectorBdQpData<'_>, k_face: &mut [f64]);
}

/// Accumulate a linear-form contribution over a boundary face for H(curl).
///
/// `f_face` has length `n_dofs`.
pub trait VectorBoundaryLinearIntegrator: Send + Sync {
    fn add_to_face_vector(&self, qp: &VectorBdQpData<'_>, f_face: &mut [f64]);
}

// ─── Integrator: TangentialMassIntegrator ────────────────────────────────────

/// Boundary bilinear integrator: `γ ∫_Γ (n×u)·(n×v) dS`.
///
/// Used for:
/// - Silver-Müller absorbing boundary conditions
/// - Impedance (Robin-type) boundary conditions for Maxwell
/// - First-order absorbing boundary conditions in time-domain Maxwell
///
/// # 2-D formula
///
/// In 2-D, `n×u` is the scalar `u_x n_y − u_y n_x`, so:
/// ```text
/// k_ij = γ ∫_Γ (n×φᵢ)(n×φⱼ) ds
/// ```
///
/// # 3-D formula
///
/// In 3-D, `n×u` is a 3-vector perpendicular to `n`:
/// ```text
/// k_ij = γ ∫_Γ (n×φᵢ)·(n×φⱼ) dS
/// ```
pub struct TangentialMassIntegrator {
    /// Boundary coefficient γ (e.g. admittance Y = 1/η₀ for ABC).
    pub gamma: f64,
}

impl VectorBoundaryBilinearIntegrator for TangentialMassIntegrator {
    fn add_to_face_matrix(&self, qp: &VectorBdQpData<'_>, k_face: &mut [f64]) {
        let n     = qp.n_dofs;
        let dim   = qp.dim;
        let w_gam = qp.weight * self.gamma;
        let normal = qp.normal;

        if dim == 2 {
            // n × φᵢ = φᵢ_x * n_y − φᵢ_y * n_x   (scalar in 2-D)
            let nx = normal[0];
            let ny = normal[1];
            for i in 0..n {
                let phi_ix = qp.phi_vec[i * 2];
                let phi_iy = qp.phi_vec[i * 2 + 1];
                let nxphi_i = phi_ix * ny - phi_iy * nx;
                for j in 0..n {
                    let phi_jx = qp.phi_vec[j * 2];
                    let phi_jy = qp.phi_vec[j * 2 + 1];
                    let nxphi_j = phi_jx * ny - phi_jy * nx;
                    k_face[i * n + j] += w_gam * nxphi_i * nxphi_j;
                }
            }
        } else {
            // n × φᵢ in 3-D: (n × φᵢ)[k] = n[l] φᵢ[m] - n[m] φᵢ[l]
            // via the cross product formula.
            let [nx, ny, nz] = [normal[0], normal[1], normal[2]];
            for i in 0..n {
                let [pix, piy, piz] = [
                    qp.phi_vec[i * 3],
                    qp.phi_vec[i * 3 + 1],
                    qp.phi_vec[i * 3 + 2],
                ];
                // n × φᵢ
                let cx_i = ny * piz - nz * piy;
                let cy_i = nz * pix - nx * piz;
                let cz_i = nx * piy - ny * pix;

                for j in 0..n {
                    let [pjx, pjy, pjz] = [
                        qp.phi_vec[j * 3],
                        qp.phi_vec[j * 3 + 1],
                        qp.phi_vec[j * 3 + 2],
                    ];
                    let cx_j = ny * pjz - nz * pjy;
                    let cy_j = nz * pjx - nx * pjz;
                    let cz_j = nx * pjy - ny * pjx;

                    let dot = cx_i * cx_j + cy_i * cy_j + cz_i * cz_j;
                    k_face[i * n + j] += w_gam * dot;
                }
            }
        }
    }
}

// ─── VectorBoundaryAssembler ─────────────────────────────────────────────────

/// Assembly driver for boundary integrals over H(curl) spaces.
///
/// Iterates over boundary faces (edges in 2-D, triangular faces in 3-D),
/// evaluates H(curl) basis functions at face quadrature points using the
/// covariant Piola transform, and delegates to [`VectorBoundaryBilinearIntegrator`]
/// or [`VectorBoundaryLinearIntegrator`] implementations.
pub struct VectorBoundaryAssembler;

/// ∫ g · (v·n) ds — H(div) natural BC flux integrator.
///
/// MFEM equivalent: `VectorFEBoundaryFluxLFIntegrator`.
/// Evaluates `∫ g (φᵢ·n) ds` for each boundary DOF `i`.
pub struct HdivNormalFluxIntegrator<F: Fn(&[f64]) -> f64 + Send + Sync> {
    pub g: F,
}

impl<F: Fn(&[f64]) -> f64 + Send + Sync> VectorBoundaryLinearIntegrator for HdivNormalFluxIntegrator<F> {
    fn add_to_face_vector(&self, qp: &VectorBdQpData, f_elem: &mut [f64]) {
        let g_val = (self.g)(&qp.x_phys);
        let dim = qp.normal.len();
        for i in 0..qp.n_dofs {
            let vn: f64 = (0..dim).map(|c| qp.phi_vec[i * dim + c] * qp.normal[c]).sum();
            f_elem[i] += g_val * vn * qp.weight;
        }
    }
}

impl VectorBoundaryAssembler {
    /// Assemble a boundary bilinear form over tagged boundary faces.
    ///
    /// Returns a `n_global_dofs × n_global_dofs` sparse matrix.
    ///
    /// # Arguments
    /// * `space`       — H(curl) FE space providing element DOFs and signs.
    /// * `integrators` — list of boundary bilinear integrators.
    /// * `tags`        — boundary face tags to integrate over.
    /// * `quad_order`  — quadrature accuracy order on each face.
    pub fn assemble_boundary_bilinear<S>(
        space:       &S,
        integrators: &[&dyn VectorBoundaryBilinearIntegrator],
        tags:        &[i32],
        quad_order:  u8,
    ) -> CsrMatrix<f64>
    where
        S: FESpace + Sync,
        S::Mesh: MeshTopology + Sync,
    {
        let mesh    = space.mesh();
        let dim     = mesh.dim() as usize;
        let n_dofs  = space.n_dofs();
        let stype = space.space_type();
        assert!(
            stype == SpaceType::HCurl || stype == SpaceType::HDiv,
            "VectorBoundaryAssembler: only H(curl) and H(div) spaces are supported"
        );

        let face_ids: Vec<u32> = mesh
            .face_iter()
            .filter(|&f| tags.contains(&mesh.face_tag(f)))
            .collect();

        #[cfg(feature = "parallel")]
        {
            if face_ids.len() >= assembly_parallel_min_elems() {
                return assemble_boundary_bilinear_parallel(
                    space,
                    integrators,
                    quad_order,
                    &face_ids,
                    n_dofs,
                    dim,
                );
            }
        }

        assemble_boundary_bilinear_serial(space, integrators, quad_order, &face_ids, n_dofs, dim)
    }

    /// Assemble a boundary linear form over tagged boundary faces.
    ///
    /// Returns a global load vector of length `space.n_dofs()`.
    pub fn assemble_boundary_linear<S>(
        space:       &S,
        integrators: &[&dyn VectorBoundaryLinearIntegrator],
        tags:        &[i32],
        quad_order:  u8,
    ) -> Vec<f64>
    where
        S: FESpace + Sync,
        S::Mesh: MeshTopology + Sync,
    {
        let mesh   = space.mesh();
        let dim    = mesh.dim() as usize;
        let n_dofs = space.n_dofs();
        let stype = space.space_type();
        assert!(
            stype == SpaceType::HCurl || stype == SpaceType::HDiv,
            "VectorBoundaryAssembler: only H(curl) and H(div) spaces are supported"
        );

        let face_ids: Vec<u32> = mesh
            .face_iter()
            .filter(|&f| tags.contains(&mesh.face_tag(f)))
            .collect();

        #[cfg(feature = "parallel")]
        {
            if face_ids.len() >= assembly_parallel_min_elems() {
                return assemble_boundary_linear_parallel(
                    space,
                    integrators,
                    quad_order,
                    &face_ids,
                    n_dofs,
                    dim,
                );
            }
        }

        assemble_boundary_linear_serial(space, integrators, quad_order, &face_ids, n_dofs, dim)
    }

    /// Assemble the RHS for an H(div) natural (flux) BC: ∫_Γ g(x) ds per face DOF.
    ///
    /// Each boundary face contributes `∫_Γ g(x) ds` to its associated DOF,
    /// computed via the midpoint rule (exact for linear g).  Unlike the generic
    /// Assemble the H(div) boundary flux RHS: `∫_∂Ω g (v·n) ds`.
    ///
    /// Used for natural boundary conditions in mixed Darcy / Poisson.
    /// Correctly handles DOF orientation signs (contravariant Piola sign)
    /// by looking up the adjacent element's orientation.
    pub fn assemble_hdiv_boundary_flux(
        space: &HDivSpace<Mesh<2>>,
        g: &dyn Fn(&[f64]) -> f64,
        tags: &[i32],
    ) -> Vec<f64> {
        let mesh = space.mesh();
        let n_dofs = space.n_dofs();
        let mut rhs = vec![0.0; n_dofs];
        for f in 0..mesh.n_boundary_faces() as u32 {
            if !tags.contains(&mesh.face_tag(f)) { continue; }
            let nodes = mesh.face_nodes(f);
            if nodes.len() < 2 { continue; }
            // Find the adjacent element to apply orientation signs.
            let elem = mesh.face_elements(f).first().copied().unwrap_or(0);
            let pa = mesh.node_coords(nodes[0]);
            let pb = mesh.node_coords(nodes[1]);
            let tx = pb[0] - pa[0]; let ty = pb[1] - pa[1];
            let mid = [0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1])];
            let edge_len = (tx * tx + ty * ty).sqrt();
            let flux_val = (g)(&mid) * edge_len;
            let ek = EdgeKey::new(nodes[0], nodes[1]);
            if let Some(global_dof) = space.edge_face_dof(ek) {
                // Apply orientation sign from the adjacent element.
                let elem_dofs = space.element_dofs(elem);
                let signs = space.element_signs(elem);
                let sign = signs.iter().zip(elem_dofs.iter())
                    .find(|(_, &gd)| gd == global_dof)
                    .map(|(s, _)| *s)
                    .unwrap_or(1.0);
                rhs[global_dof as usize] += sign * flux_val;
            }
        }
        rhs
    }
}

fn assemble_boundary_bilinear_serial<S: FESpace>(
    space: &S,
    integrators: &[&dyn VectorBoundaryBilinearIntegrator],
    quad_order: u8,
    face_ids: &[u32],
    n_dofs: usize,
    dim: usize,
) -> CsrMatrix<f64>
where
    S::Mesh: MeshTopology,
{
    let stype = space.space_type();
    let elem_type = if space.mesh().n_elements() > 0 { space.mesh().element_type(0) } else { panic!("empty mesh") };
    let vol_elem = vol_ref_elem(stype, elem_type, dim, space.order());
    let n_ldofs = vol_elem.n_dofs();
    let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
    let mut ref_phi = vec![0.0_f64; n_ldofs * dim];
    let mut phys_phi = vec![0.0_f64; n_ldofs * dim];

    for &f in face_ids {
        if let Some((global_dofs, k_face)) =
            assemble_face_bilinear_contrib(space, &*vol_elem, stype, n_ldofs, dim, f, integrators, quad_order, &mut ref_phi, &mut phys_phi)
        {
            coo.add_element_matrix(&global_dofs, &k_face);
        }
    }
    coo.into_csr()
}

fn assemble_boundary_linear_serial<S: FESpace>(
    space: &S,
    integrators: &[&dyn VectorBoundaryLinearIntegrator],
    quad_order: u8,
    face_ids: &[u32],
    n_dofs: usize,
    dim: usize,
) -> Vec<f64>
where
    S::Mesh: MeshTopology,
{
    let stype = space.space_type();
    let elem_type = if space.mesh().n_elements() > 0 { space.mesh().element_type(0) } else { panic!("empty mesh") };
    let vol_elem = vol_ref_elem(stype, elem_type, dim, space.order());
    let n_ldofs = vol_elem.n_dofs();
    let mut rhs = vec![0.0_f64; n_dofs];
    let mut ref_phi = vec![0.0_f64; n_ldofs * dim];
    let mut phys_phi = vec![0.0_f64; n_ldofs * dim];

    for &f in face_ids {
        if let Some((global_dofs, f_face)) =
            assemble_face_linear_contrib(space, &*vol_elem, stype, n_ldofs, dim, f, integrators, quad_order, &mut ref_phi, &mut phys_phi)
        {
            for (&d, &v) in global_dofs.iter().zip(f_face.iter()) {
                rhs[d] += v;
            }
        }
    }
    rhs
}

#[cfg(feature = "parallel")]
fn assemble_boundary_bilinear_parallel<S: FESpace + Sync>(
    space: &S,
    integrators: &[&dyn VectorBoundaryBilinearIntegrator],
    quad_order: u8,
    face_ids: &[u32],
    n_dofs: usize,
    dim: usize,
) -> CsrMatrix<f64>
where
    S::Mesh: MeshTopology + Sync,
{
    let stype = space.space_type();
    face_ids
        .par_iter()
        .copied()
        .filter_map(|f| {
            let et = space.mesh().element_type(0);
            let vol_elem = vol_ref_elem(stype, et, dim, space.order());
            let n_ldofs = vol_elem.n_dofs();
            let mut ref_phi = vec![0.0_f64; n_ldofs * dim];
            let mut phys_phi = vec![0.0_f64; n_ldofs * dim];
            assemble_face_bilinear_contrib(
                space,
                &*vol_elem,
                stype,
                n_ldofs,
                dim,
                f,
                integrators,
                quad_order,
                &mut ref_phi,
                &mut phys_phi,
            )
        })
        .map(|(global_dofs, k_face)| {
            let mut local = CooMatrix::<f64>::new(n_dofs, n_dofs);
            local.add_element_matrix(&global_dofs, &k_face);
            local
        })
        .reduce(
            || CooMatrix::<f64>::new(n_dofs, n_dofs),
            |mut a, b| {
                a.append(b);
                a
            },
        )
        .into_csr()
}

#[cfg(feature = "parallel")]
fn assemble_boundary_linear_parallel<S: FESpace + Sync>(
    space: &S,
    integrators: &[&dyn VectorBoundaryLinearIntegrator],
    quad_order: u8,
    face_ids: &[u32],
    n_dofs: usize,
    dim: usize,
) -> Vec<f64>
where
    S::Mesh: MeshTopology + Sync,
{
    let stype = space.space_type();
    face_ids
        .par_iter()
        .copied()
        .filter_map(|f| {
            let et = space.mesh().element_type(0);
            let vol_elem = vol_ref_elem(stype, et, dim, space.order());
            let n_ldofs = vol_elem.n_dofs();
            let mut ref_phi = vec![0.0_f64; n_ldofs * dim];
            let mut phys_phi = vec![0.0_f64; n_ldofs * dim];
            assemble_face_linear_contrib(
                space,
                &*vol_elem,
                stype,
                n_ldofs,
                dim,
                f,
                integrators,
                quad_order,
                &mut ref_phi,
                &mut phys_phi,
            )
        })
        .fold(
            || vec![0.0_f64; n_dofs],
            |mut local, (global_dofs, f_face)| {
                for (&d, &v) in global_dofs.iter().zip(f_face.iter()) {
                    local[d] += v;
                }
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
        )
}

/// Per-quadrature-point owner-element geometry for one boundary face.
///
/// `xi_ref` lives in the *owner element's* reference domain (the same domain
/// the solution [`VectorReferenceElement`] evaluates on); `jac`/`det_j` are
/// the owner's isoparametric (or affine simplex) Jacobian at that point.
struct FaceQpGeometry {
    weight: f64,
    normal: Vec<f64>,
    x_phys: Vec<f64>,
    xi_ref: Vec<f64>,
    jac: DMatrix<f64>,
    det_j: f64,
}

/// Resolve the owner element of boundary face `f` and map every face
/// quadrature point back to owner reference coordinates.
///
/// D13: the owner geometry must match the *volume* assembly — tensor-product
/// owners (Quad/Hex/...) and curved meshes go through the isoparametric
/// path (`geo_ref_elem_from_mesh` + `isoparametric_jacobian`, the same
/// dispatch as `assemble_hdiv_l2_mixed`/`VectorAssembler`), only affine
/// simplices use the direct `ElementTransformation::from_simplex_nodes`.
/// The previous simplex-only inverse silently produced wrong reference
/// coordinates (and hence wrong basis values) on quad owners, leaking DOF
/// contributions across the whole element row.
fn face_owner_geometry<M: MeshTopology>(
    mesh: &M,
    f: u32,
    quad_order: u8,
    dim: usize,
) -> Option<(u32, Vec<FaceQpGeometry>)> {
    let face_nodes = mesh.face_nodes(f);
    let owner_elem = find_owner_element(mesh, face_nodes)?;
    let elem_nodes = mesh.element_nodes(owner_elem);
    let use_iso = mesh.geom_order() > 1
        || !matches!(
            mesh.element_type(owner_elem),
            ElementType::Tri3 | ElementType::Tet4 | ElementType::Line2
        );
    let geo_elem = if use_iso {
        geo_ref_elem_from_mesh(mesh, owner_elem)
    } else {
        None
    };
    let affine_tr = if use_iso {
        None
    } else {
        Some(ElementTransformation::from_simplex_nodes(mesh, elem_nodes))
    };

    let (face_qp_phys, face_weights, face_normals) =
        face_quadrature(mesh, face_nodes, dim, quad_order);
    let mut qps = Vec::with_capacity(face_weights.len());
    for q in 0..face_weights.len() {
        let x_phys = face_qp_phys[q * dim..(q + 1) * dim].to_vec();
        let normal = face_normals[q * dim..(q + 1) * dim].to_vec();
        let (xi_ref, jac, det_j) = match geo_elem.as_deref() {
            Some(ge) => {
                let geo_nds = mesh.geometry_nodes(owner_elem);
                let xi = phys_to_ref_isoparametric(mesh, geo_nds, ge, &x_phys, dim);
                let (jac, det, _) = isoparametric_jacobian(mesh, geo_nds, ge, &xi, dim);
                (xi, jac, det)
            }
            None => {
                let tr = affine_tr.as_ref().unwrap();
                (
                    phys_to_ref(mesh, elem_nodes, &x_phys, dim),
                    tr.jacobian().clone(),
                    tr.det_j(),
                )
            }
        };
        qps.push(FaceQpGeometry {
            weight: face_weights[q],
            normal,
            x_phys,
            xi_ref,
            jac,
            det_j,
        });
    }
    Some((owner_elem, qps))
}

#[allow(clippy::too_many_arguments)]
fn assemble_face_bilinear_contrib<S: FESpace>(
    space: &S,
    vol_elem: &dyn VectorReferenceElement,
    stype: SpaceType,
    n_ldofs: usize,
    dim: usize,
    f: u32,
    integrators: &[&dyn VectorBoundaryBilinearIntegrator],
    quad_order: u8,
    ref_phi: &mut [f64],
    phys_phi: &mut [f64],
) -> Option<(Vec<usize>, Vec<f64>)>
where
    S::Mesh: MeshTopology,
{
    let mesh = space.mesh();
    let (owner_elem, qps) = face_owner_geometry(mesh, f, quad_order, dim)?;
    let global_dofs: Vec<usize> = space
        .element_dofs(owner_elem)
        .iter()
        .map(|&d| d as usize)
        .collect();
    let signs_opt = space.element_signs(owner_elem);
    let mut k_face = vec![0.0_f64; n_ldofs * n_ldofs];
    for qp in &qps {
        vol_elem.eval_basis_vec(&qp.xi_ref, ref_phi);
        match stype {
            SpaceType::HCurl => {
                let j_inv_t = qp
                    .jac
                    .clone()
                    .try_inverse()
                    .expect("degenerate boundary owner element")
                    .transpose();
                piola_hcurl_basis(&j_inv_t, ref_phi, phys_phi, n_ldofs, dim);
            }
            SpaceType::HDiv => {
                piola_hdiv_basis(&qp.jac, qp.det_j, ref_phi, phys_phi, n_ldofs, dim);
            }
            _ => unreachable!(),
        }
        if let Some(s) = signs_opt {
            for i in 0..n_ldofs {
                for c in 0..dim {
                    phys_phi[i * dim + c] *= s[i];
                }
            }
        }
        let qp_data = VectorBdQpData {
            n_dofs: n_ldofs,
            dim,
            weight: qp.weight,
            phi_vec: phys_phi,
            normal: &qp.normal,
            x_phys: &qp.x_phys,
            elem_id: owner_elem,
            elem_tag: mesh.face_tag(f),
        };
        for integ in integrators {
            integ.add_to_face_matrix(&qp_data, &mut k_face);
        }
    }
    Some((global_dofs, k_face))
}

#[allow(clippy::too_many_arguments)]
fn assemble_face_linear_contrib<S: FESpace>(
    space: &S,
    vol_elem: &dyn VectorReferenceElement,
    stype: SpaceType,
    n_ldofs: usize,
    dim: usize,
    f: u32,
    integrators: &[&dyn VectorBoundaryLinearIntegrator],
    quad_order: u8,
    ref_phi: &mut [f64],
    phys_phi: &mut [f64],
) -> Option<(Vec<usize>, Vec<f64>)>
where
    S::Mesh: MeshTopology,
{
    let mesh = space.mesh();
    let (owner_elem, qps) = face_owner_geometry(mesh, f, quad_order, dim)?;
    let global_dofs: Vec<usize> = space
        .element_dofs(owner_elem)
        .iter()
        .map(|&d| d as usize)
        .collect();
    let signs_opt = space.element_signs(owner_elem);
    let mut f_face = vec![0.0_f64; n_ldofs];
    for qp in &qps {
        vol_elem.eval_basis_vec(&qp.xi_ref, ref_phi);
        match stype {
            SpaceType::HCurl => {
                let j_inv_t = qp
                    .jac
                    .clone()
                    .try_inverse()
                    .expect("degenerate boundary owner element")
                    .transpose();
                piola_hcurl_basis(&j_inv_t, ref_phi, phys_phi, n_ldofs, dim);
            }
            SpaceType::HDiv => {
                piola_hdiv_basis(&qp.jac, qp.det_j, ref_phi, phys_phi, n_ldofs, dim);
            }
            _ => unreachable!(),
        }
        if let Some(s) = signs_opt {
            for i in 0..n_ldofs {
                for c in 0..dim {
                    phys_phi[i * dim + c] *= s[i];
                }
            }
        }
        let qp_data = VectorBdQpData {
            n_dofs: n_ldofs,
            dim,
            weight: qp.weight,
            phi_vec: phys_phi,
            normal: &qp.normal,
            x_phys: &qp.x_phys,
            elem_id: owner_elem,
            elem_tag: mesh.face_tag(f),
        };
        for integ in integrators {
            integ.add_to_face_vector(&qp_data, &mut f_face);
        }
    }
    Some((global_dofs, f_face))
}

// ─── Piola transforms (same as vector_assembler.rs) ──────────────────────────

fn piola_hcurl_basis(
    j_inv_t:  &DMatrix<f64>,
    ref_vals: &[f64],
    phys_vals: &mut [f64],
    n_dofs:   usize,
    dim:      usize,
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

/// Contravariant Piola transform for H(div): φ_phys = J · φ_ref / det_j
fn piola_hdiv_basis(
    jac:  &DMatrix<f64>,
    det_j: f64,
    ref_vals: &[f64],
    phys_vals: &mut [f64],
    n_dofs:   usize,
    dim:      usize,
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

/// Find the volume element that owns this boundary face (all face nodes are
/// a subset of the element's nodes).
fn find_owner_element<M: MeshTopology>(mesh: &M, face_nodes: &[u32]) -> Option<u32> {
    for e in mesh.elem_iter() {
        let enodes = mesh.element_nodes(e);
        if face_nodes.iter().all(|fn_| enodes.contains(fn_)) {
            return Some(e);
        }
    }
    None
}

/// Map a physical point `xp` back to reference coordinates for a simplex element.
///
/// Solves J ξ = (xp − x0) where J is the element Jacobian.
fn phys_to_ref<M: MeshTopology>(
    mesh:       &M,
    elem_nodes: &[u32],
    xp:         &[f64],
    dim:        usize,
) -> Vec<f64> {
    let tr = ElementTransformation::from_simplex_nodes(mesh, elem_nodes);
    let x0 = mesh.node_coords(elem_nodes[0]);
    let mut b = vec![0.0_f64; dim];
    for i in 0..dim { b[i] = xp[i] - x0[i]; }

    let j_inv = tr.jacobian().clone().try_inverse().expect("degenerate element");

    let mut xi = vec![0.0_f64; dim];
    for i in 0..dim {
        for k in 0..dim {
            xi[i] += j_inv[(i, k)] * b[k];
        }
    }
    xi
}

/// Invert the isoparametric element map `F(ξ) = xp` by Newton iteration.
///
/// `geo_elem` must be the same geometry reference element the volume
/// assembly uses for this element (`geo_ref_elem_from_mesh`).  The
/// iteration starts from the element centroid; straight-sided elements
/// (the common case for boundary faces) converge quadratically in a
/// handful of steps.
fn phys_to_ref_isoparametric<M: MeshTopology>(
    mesh:      &M,
    geo_nodes: &[u32],
    geo_elem:  &dyn ReferenceElement,
    xp:        &[f64],
    dim:       usize,
) -> Vec<f64> {
    let mut xi = vec![0.5_f64; dim];
    let scale: f64 = xp.iter().map(|v| v.abs()).sum::<f64>() + 1.0;
    for _ in 0..50 {
        let (jac, _, fx) = isoparametric_jacobian(mesh, geo_nodes, geo_elem, &xi, dim);
        let mut resid = vec![0.0_f64; dim];
        let mut resid2 = 0.0_f64;
        for i in 0..dim {
            resid[i] = xp[i] - fx[i];
            resid2 += resid[i] * resid[i];
        }
        if resid2.sqrt() < 1e-14 * scale {
            return xi;
        }
        let j_inv = jac.try_inverse().expect("degenerate boundary owner element");
        for i in 0..dim {
            for k in 0..dim {
                xi[i] += j_inv[(i, k)] * resid[k];
            }
        }
    }
    panic!(
        "phys_to_ref_isoparametric: Newton did not converge (xp={xp:?}, xi={xi:?})"
    );
}

/// Compute face quadrature points (in physical space), weights, and outward normals.
///
/// Returns `(xp_flat, weights, normals_flat)` where:
/// - `xp_flat` has length `n_qp * dim`  (row-major)
/// - `weights` has length `n_qp`
/// - `normals_flat` has length `n_qp * dim` (same normal repeated per QP)
fn face_quadrature<M: MeshTopology>(
    mesh:       &M,
    face_nodes: &[u32],
    dim:        usize,
    quad_order: u8,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    match dim {
        2 => face_quadrature_2d(mesh, face_nodes, quad_order),
        3 => face_quadrature_3d(mesh, face_nodes, quad_order),
        _ => panic!("face_quadrature: unsupported dim={dim}"),
    }
}

/// 2-D edge quadrature (1-D Gauss-Legendre on [0,1] → edge parametrisation).
fn face_quadrature_2d<M: MeshTopology>(
    mesh:       &M,
    face_nodes: &[u32],
    quad_order: u8,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let x0 = mesh.node_coords(face_nodes[0]);
    let x1 = mesh.node_coords(face_nodes[1]);
    let dx = x1[0] - x0[0];
    let dy = x1[1] - x0[1];
    let len = (dx * dx + dy * dy).sqrt();

    // Outward unit normal (pointing away from domain interior by convention).
    // Convention: rotate edge tangent by -90°: n = (dy, -dx) / len.
    let nx =  dy / len;
    let ny = -dx / len;

    // 1-D Gauss-Legendre points on [0, 1].
    let (gpts, gwts) = gauss_legendre_1d(quad_order);
    let n_qp = gpts.len();

    let mut xp_flat  = Vec::with_capacity(n_qp * 2);
    let mut weights  = Vec::with_capacity(n_qp);
    let mut normals  = Vec::with_capacity(n_qp * 2);

    for q in 0..n_qp {
        let t = gpts[q];
        xp_flat.push(x0[0] + t * dx);
        xp_flat.push(x0[1] + t * dy);
        weights.push(gwts[q] * len);
        normals.push(nx);
        normals.push(ny);
    }

    (xp_flat, weights, normals)
}

/// 3-D triangular face quadrature (reference triangle QP mapped to physical face).
fn face_quadrature_3d<M: MeshTopology>(
    mesh:       &M,
    face_nodes: &[u32],
    _quad_order: u8,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let pa = mesh.node_coords(face_nodes[0]);
    let pb = mesh.node_coords(face_nodes[1]);
    let pc = mesh.node_coords(face_nodes[2]);

    let ab = [pb[0]-pa[0], pb[1]-pa[1], pb[2]-pa[2]];
    let ac = [pc[0]-pa[0], pc[1]-pa[1], pc[2]-pa[2]];

    // Cross product → normal (not yet normalised).
    let cross = [
        ab[1]*ac[2] - ab[2]*ac[1],
        ab[2]*ac[0] - ab[0]*ac[2],
        ab[0]*ac[1] - ab[1]*ac[0],
    ];
    let area2 = (cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]).sqrt();
    let area  = 0.5 * area2;
    let nx    = cross[0] / area2;
    let ny    = cross[1] / area2;
    let nz    = cross[2] / area2;

    // 3-point centroid quadrature on reference triangle (order 2).
    // Points: (1/6,1/6), (2/3,1/6), (1/6,2/3) — weight 1/3 each.
    let ref_pts = [(1.0/6.0, 1.0/6.0), (2.0/3.0, 1.0/6.0), (1.0/6.0, 2.0/3.0)];
    let ref_w   = 1.0 / 3.0;
    let n_qp    = ref_pts.len();

    let mut xp_flat = Vec::with_capacity(n_qp * 3);
    let mut weights = Vec::with_capacity(n_qp);
    let mut normals = Vec::with_capacity(n_qp * 3);

    for (s, t) in ref_pts {
        xp_flat.push(pa[0] + s*ab[0] + t*ac[0]);
        xp_flat.push(pa[1] + s*ab[1] + t*ac[1]);
        xp_flat.push(pa[2] + s*ab[2] + t*ac[2]);
        weights.push(ref_w * area);
        normals.push(nx); normals.push(ny); normals.push(nz);
    }

    (xp_flat, weights, normals)
}

/// 1-D Gauss-Legendre quadrature on [0, 1] with `n` points (n = quad_order / 2 + 1).
fn gauss_legendre_1d(order: u8) -> (Vec<f64>, Vec<f64>) {
    // Map standard [-1,1] GL points to [0,1]: t = (xi + 1) / 2, w → w/2.
    match order {
        0 | 1 => (vec![0.5], vec![1.0]),
        2 | 3 => {
            let s = 1.0 / (3.0_f64).sqrt();
            (
                vec![0.5 * (1.0 - s), 0.5 * (1.0 + s)],
                vec![0.5, 0.5],
            )
        }
        4 | 5 => {
            let s = (3.0_f64 / 5.0).sqrt();
            (
                vec![0.5*(1.0-s), 0.5, 0.5*(1.0+s)],
                vec![5.0/18.0, 4.0/9.0, 5.0/18.0],
            )
        }
        _ => {
            // 4-point GL (exact up to degree 7).
            let s1 = ((3.0 - 2.0*(6.0_f64/5.0).sqrt())/7.0).sqrt();
            let s2 = ((3.0 + 2.0*(6.0_f64/5.0).sqrt())/7.0).sqrt();
            let w1 = 0.5 + (1.0_f64/6.0)*(5.0_f64/6.0).sqrt();
            let w2 = 0.5 - (1.0_f64/6.0)*(5.0_f64/6.0).sqrt();
            (
                vec![0.5*(1.0-s2), 0.5*(1.0-s1), 0.5*(1.0+s1), 0.5*(1.0+s2)],
                vec![0.5*w2, 0.5*w1, 0.5*w1, 0.5*w2],
            )
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::HCurlSpace;

    /// The tangential mass matrix over the full boundary of the unit square
    /// must be symmetric.
    #[test]
    fn tangential_mass_symmetric_2d() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);
        let n     = space.n_dofs();

        let integ = TangentialMassIntegrator { gamma: 1.0 };
        let mat   = VectorBoundaryAssembler::assemble_boundary_bilinear(
            &space, &[&integ], &[1, 2, 3, 4], 4,
        );

        let dense = mat.to_dense();
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i * n + j] - dense[j * n + i]).abs();
                assert!(diff < 1e-12, "B[{i},{j}] - B[{j},{i}] = {diff}");
            }
        }
    }

    /// The diagonal of the boundary tangential mass must be non-negative
    /// (PSD on boundary DOFs).
    #[test]
    fn tangential_mass_psd_2d() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);

        let integ = TangentialMassIntegrator { gamma: 1.0 };
        let mat   = VectorBoundaryAssembler::assemble_boundary_bilinear(
            &space, &[&integ], &[1, 2, 3, 4], 4,
        );

        for i in 0..mat.nrows {
            let d = mat.get(i, i);
            assert!(d >= -1e-14, "diagonal B[{i},{i}] = {d} is negative");
        }
    }

    /// Interior-only boundary tags should produce an all-zero matrix.
    #[test]
    fn tangential_mass_empty_tag_gives_zero() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);

        let integ = TangentialMassIntegrator { gamma: 1.0 };
        // Tag 99 does not exist → zero matrix.
        let mat = VectorBoundaryAssembler::assemble_boundary_bilinear(
            &space, &[&integ], &[99], 4,
        );
        let dense = mat.to_dense();
        for &v in &dense {
            assert!(v.abs() < 1e-15, "expected zero matrix, got {v}");
        }
    }

    /// Tensor curl-curl with identity tensor equals the scalar version.
    #[test]
    fn curl_curl_tensor_identity_matches_scalar_2d() {
        use crate::standard::{CurlCurlIntegrator, CurlCurlTensorIntegrator};
        use crate::vector_assembler::VectorAssembler;
        use crate::postproc::coefficient::ConstantMatrixCoeff;

        let mesh1 = Mesh::<2>::unit_square_tri(4);
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let sp1 = HCurlSpace::new(mesh1, 1);
        let sp2 = HCurlSpace::new(mesh2, 1);

        let scalar  = CurlCurlIntegrator { mu: 1.0_f64 };
        let tensor  = CurlCurlTensorIntegrator {
            mu: ConstantMatrixCoeff(vec![1.0, 0.0, 0.0, 1.0]),
        };

        let mat_s = VectorAssembler::assemble_bilinear(&sp1, &[&scalar], 4);
        let mat_t = VectorAssembler::assemble_bilinear(&sp2, &[&tensor], 4);

        let n = mat_s.nrows;
        let ds = mat_s.to_dense();
        let dt = mat_t.to_dense();
        for i in 0..n {
            for j in 0..n {
                let diff = (ds[i*n+j] - dt[i*n+j]).abs();
                assert!(diff < 1e-12,
                    "scalar vs tensor K[{i},{j}]: {} vs {}", ds[i*n+j], dt[i*n+j]);
            }
        }
    }

    /// Tensor vector-mass with identity tensor equals the scalar version.
    #[test]
    fn vector_mass_tensor_identity_matches_scalar_2d() {
        use crate::standard::{VectorMassIntegrator, VectorMassTensorIntegrator};
        use crate::vector_assembler::VectorAssembler;
        use crate::postproc::coefficient::ConstantMatrixCoeff;

        let mesh1 = Mesh::<2>::unit_square_tri(4);
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let sp1 = HCurlSpace::new(mesh1, 1);
        let sp2 = HCurlSpace::new(mesh2, 1);

        let scalar = VectorMassIntegrator { alpha: 1.0_f64 };
        let tensor = VectorMassTensorIntegrator {
            alpha: ConstantMatrixCoeff(vec![1.0, 0.0, 0.0, 1.0]),
        };

        let mat_s = VectorAssembler::assemble_bilinear(&sp1, &[&scalar], 4);
        let mat_t = VectorAssembler::assemble_bilinear(&sp2, &[&tensor], 4);

        let n = mat_s.nrows;
        let ds = mat_s.to_dense();
        let dt = mat_t.to_dense();
        for i in 0..n {
            for j in 0..n {
                let diff = (ds[i*n+j] - dt[i*n+j]).abs();
                assert!(diff < 1e-12,
                    "scalar vs tensor M[{i},{j}]: {} vs {}", ds[i*n+j], dt[i*n+j]);
            }
        }
    }

    // ── D13: boundary linear form on tri AND quad owners (analytic) ─────────

    /// H(div) boundary flux `b_i = ∫_Γ g (φ_i·n̂) ds` must satisfy
    /// `b · c = ∮_Γ g (v·n̂) ds` exactly whenever `v = Σ c_i φ_i` is a
    /// representable flux field.  With `g(x,y) = x` and `v = (1, 0)` (in RT0):
    /// only the right edge of the unit square contributes → `∮ = 1`.
    ///
    /// D13 regression: quad owners previously went through the simplex
    /// inverse map (`from_simplex_nodes` + affine `phys_to_ref`), producing
    /// wrong reference coordinates and leaking DOF contributions; the D13
    /// fix routes tensor-product owners through the isoparametric geometry.
    #[test]
    fn hdiv_boundary_flux_rt0_analytic_tri_and_quad() {
        use fem_space::HDivSpace;

        let g = |x: &[f64]| x[0];

        for (name, mesh) in [
            ("tri", Mesh::<2>::unit_square_tri(2)),
            ("quad", Mesh::<2>::unit_square_quad(2)),
        ] {
            let rt = HDivSpace::new(mesh.clone(), 0);
            let tags: Vec<i32> = (0..mesh.n_boundary_faces() as u32)
                .map(|f| mesh.face_tag(f))
                .collect();
            let b = VectorBoundaryAssembler::assemble_boundary_linear(
                &rt, &[&HdivNormalFluxIntegrator { g }], &tags, 4,
            );

            // Per-face analytic value: the RT0 normal trace of the owner-local
            // basis on its own edge is the CONSTANT 1/|E| (Piola preserves the
            // reference flux ∫ (φ·n̂) = 1), so the assembled (orientation-
            // signed) entry is exactly  b_d = s_d · (1/|E|) ∫_E x ds.
            let mut n_nonzero = 0_usize;
            for f in 0..mesh.n_boundary_faces() as u32 {
                let nds = mesh.face_nodes(f);
                let p0 = mesh.node_coords(nds[0]);
                let p1 = mesh.node_coords(nds[1]);
                let mean_g = 0.5 * (p0[0] + p1[0]); // (1/len)·∫ x ds on a segment
                // Owner = the element containing both face nodes.
                let owner = mesh.elem_iter().find(|&e| {
                    nds.iter().all(|n| mesh.element_nodes(e).contains(n))
                }).expect("boundary face owner");
                let dofs = rt.element_dofs(owner);
                let signs = rt.element_signs(owner);
                let dof = rt.edge_face_dof(EdgeKey::new(nds[0], nds[1]))
                    .expect("RT0 edge dof") as usize;
                let local = dofs.iter().position(|&d| d as usize == dof).unwrap();
                let s = if local < signs.len() { signs[local] } else { 1.0 };
                let got = b[dof];
                assert!(
                    (got - s * mean_g).abs() < 1e-12,
                    "{name} face {f}: b[{dof}] = {got} (expected {})",
                    s * mean_g
                );
                if got.abs() > 1e-14 {
                    n_nonzero += 1;
                }
            }
            // D13 leak check: with g = x the left edge (x = 0) is exactly
            // zero, so the nonzero count must be (#boundary faces − #left
            // faces) — the pre-D13 quad path leaked DOF contributions across
            // the whole owner row (188 nonzeros instead of 64).
            let n_left = (0..mesh.n_boundary_faces() as u32)
                .filter(|&f| {
                    let nds = mesh.face_nodes(f);
                    let p0 = mesh.node_coords(nds[0]);
                    let p1 = mesh.node_coords(nds[1]);
                    p0[0].abs() < 1e-14 && p1[0].abs() < 1e-14
                })
                .count();
            assert!(
                n_nonzero == mesh.n_boundary_faces() - n_left,
                "{name}: {n_nonzero} nonzeros (expected {})",
                mesh.n_boundary_faces() - n_left
            );
        }
    }

    /// Closed-boundary divergence identity on a single-element quad mesh
    /// (D13 quad-owner path: isoparametric geometry + Newton inverse map):
    /// with g ≡ 1 and the boundary = ∂K,
    ///   b_i = ∮_∂K Φ_i·n̂ ds = s_i·∮_∂K φ_i·n̂ ds = s_i·∫_K̂ div φ̂_i dξ̂,
    /// where the last step is the divergence theorem in the reference domain
    /// (the Piola transform preserves flux, |detJ| cancels).  The reference
    /// side uses the element-crate `QuadRT1` primitives directly.
    #[test]
    fn hdiv_boundary_flux_rt1_quad_closed_boundary_identity() {
        use fem_element::reference::VectorReferenceElement;
        use fem_element::raviart_thomas::QuadRT1;
        use fem_space::HDivSpace;

        let mesh = Mesh::<2>::unit_square_quad(1);
        let rt = HDivSpace::new(mesh.clone(), 1);
        let tags: Vec<i32> = (0..mesh.n_boundary_faces() as u32)
            .map(|f| mesh.face_tag(f))
            .collect();
        let b = VectorBoundaryAssembler::assemble_boundary_linear(
            &rt, &[&HdivNormalFluxIntegrator { g: |_| 1.0 }], &tags, 6,
        );

        // ∫_[0,1]² div φ̂_i dξ̂ with the element's own quadrature.
        let qr = QuadRT1.quadrature(6);
        let n_loc = QuadRT1.n_dofs();
        let mut ref_div = vec![0.0_f64; n_loc];
        let mut int_div = vec![0.0_f64; n_loc];
        for (xi, w) in qr.points.iter().zip(qr.weights.iter()) {
            QuadRT1.eval_div(xi, &mut ref_div);
            for (i, d) in ref_div.iter().enumerate() {
                int_div[i] += w * d;
            }
        }

        let owner = 0_u32;
        let dofs = rt.element_dofs(owner);
        let signs = rt.element_signs(owner);
        for (i, &gd) in dofs.iter().enumerate() {
            let s = if i < signs.len() { signs[i] } else { 1.0 };
            let got = b[gd as usize];
            let expected = s * int_div[i];
            assert!(
                (got - expected).abs() < 1e-12,
                "quad RT1 dof {i}: b = {got} (expected {expected})"
            );
        }
    }

    /// The boundary tangential mass on the quad unit square must be symmetric
    /// (the pre-D13 quad path produced a garbage/leaking matrix).
    #[test]
    fn tangential_mass_symmetric_2d_quad() {
        let mesh  = Mesh::<2>::unit_square_quad(4);
        let space = HCurlSpace::new(mesh, 1);
        let n     = space.n_dofs();

        let integ = TangentialMassIntegrator { gamma: 1.0 };
        let mat   = VectorBoundaryAssembler::assemble_boundary_bilinear(
            &space, &[&integ], &[1, 2, 3, 4], 4,
        );

        let dense = mat.to_dense();
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i * n + j] - dense[j * n + i]).abs();
                assert!(diff < 1e-12, "B[{i},{j}] - B[{j},{i}] = {diff}");
            }
        }
    }
}
