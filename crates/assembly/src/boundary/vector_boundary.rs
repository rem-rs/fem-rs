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
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::fe_space::{FESpace, SpaceType};
use fem_space::{EdgeKey, HDivSpace};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use crate::assembler::assembly_parallel_min_elems;
use crate::assembler::simplex_transformation;
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
/// the owner's isoparametric (or affine simplex) Jacobian at that point, and
/// `x_phys`/`weight`/`normal` are evaluated from the same owner geometry, so
/// curved boundary faces are integrated on the curve rather than on the
/// straight corner chord.
struct FaceQpGeometry {
    weight: f64,
    normal: Vec<f64>,
    x_phys: Vec<f64>,
    xi_ref: Vec<f64>,
    jac: DMatrix<f64>,
    det_j: f64,
}

/// Reference-domain coordinates, inside the owner's **geometry** reference
/// element, of the element's local nodes.
///
/// Entry `i` corresponds to position `i` in [`MeshTopology::element_nodes`]:
/// `vector_assembler::isoparametric_jacobian` pairs geometry node `k` with the
/// geometry basis row `k`, so the reference position of that node is
/// `dof_coords()[k]`.
fn owner_ref_coords(geo: &dyn ReferenceElement, n_local: usize) -> Vec<Vec<f64>> {
    let gd = geo.dof_coords();
    assert!(
        gd.len() >= n_local,
        "owner_ref_coords: geometry element has {} nodes but the element has {n_local}",
        gd.len()
    );
    gd[..n_local].to_vec()
}

/// `sqrt(det(J_faceᵀ J_face))` — the physical surface measure factor of the
/// face Jacobian `J_face` (`dim × fdim`, row-major).
fn face_measure(j_face: &[f64], dim: usize, fdim: usize) -> f64 {
    if fdim == 1 {
        let mut s = 0.0;
        for r in 0..dim {
            s += j_face[r] * j_face[r];
        }
        return s.sqrt();
    }
    let (mut g00, mut g01, mut g11) = (0.0, 0.0, 0.0);
    for r in 0..dim {
        let a = j_face[r * 2];
        let b = j_face[r * 2 + 1];
        g00 += a * a;
        g01 += a * b;
        g11 += b * b;
    }
    (g00 * g11 - g01 * g01).max(0.0).sqrt()
}

/// Outward unit normal of the face from its Jacobian columns.
///
/// Orientation convention (same as the previous straight-chord code, which
/// derived it from the face node order): in 3-D the first two face directions
/// are crossed (`(x_f1−x_f0) × (x_f2−x_f0)` for triangles,
/// `(x_f1−x_f0) × (x_f3−x_f0)` for quadrilaterals, both CCW seen from
/// outside); in 2-D the edge tangent is rotated by −90°, `n = (t_y, −t_x)`.
fn face_normal(j_face: &[f64], dim: usize, fdim: usize) -> Vec<f64> {
    if dim == 2 {
        let (tx, ty) = (j_face[0], j_face[1]);
        let len = (tx * tx + ty * ty).sqrt().max(1e-30);
        return vec![ty / len, -tx / len];
    }
    debug_assert_eq!(fdim, 2, "3-D boundary faces must be triangles/quads");
    let (a0, a1, a2) = (j_face[0], j_face[2], j_face[4]);
    let (b0, b1, b2) = (j_face[1], j_face[3], j_face[5]);
    let c = [
        a1 * b2 - a2 * b1,
        a2 * b0 - a0 * b2,
        a0 * b1 - a1 * b0,
    ];
    let len = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt().max(1e-30);
    vec![c[0] / len, c[1] / len, c[2] / len]
}

/// Resolve the owner element of boundary face `f` and evaluate the owner
/// geometry at every face quadrature point.
///
/// This mirrors MFEM's `Mesh::GetFaceElementTransformations`: the face rule is
/// defined on the *face's* reference element, and each face point is mapped
/// into the owner's reference domain by the (affine) face-to-element map built
/// from the reference positions of the face's local vertices — no Newton
/// inversion of a straight-chord physical point is involved, so the mapping is
/// exact for curved elements too.
///
/// The owner geometry itself follows the *volume* assembly dispatch: tensor
/// product owners (Quad/Hex/...) and curved meshes go through the
/// isoparametric path ([`geo_ref_elem_from_mesh`] + [`isoparametric_jacobian`]),
/// affine simplices through the geometry-node-aware affine transformation.  The
/// face measure and normal come from the owner Jacobian / face Jacobian, so
/// `geom_order > 1` faces are integrated on the curved surface.
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
        Some(simplex_transformation(mesh, owner_elem))
    };

    // ── Face reference element and rule ──────────────────────────────────────
    // The face-to-element map is affine in reference space, so the P1 face
    // element is the exact geometric map regardless of the solution order; the
    // rule order is the caller's `quad_order`.
    let face_type = match face_nodes.len() {
        2 => ElementType::Line2,
        3 => ElementType::Tri3,
        4 => ElementType::Quad4,
        n => panic!("boundary face with {n} nodes is not supported (only 2/3/4-node faces)"),
    };
    let face_ref = face_type.ref_elem(1);
    let fdim = face_ref.dim() as usize;
    assert_eq!(
        fdim + 1,
        dim,
        "boundary face dimension {fdim} does not match mesh dimension {dim}"
    );
    let n_face_nodes = face_ref.n_dofs();
    let fquad = face_ref.quadrature(quad_order);

    // ── Owner reference coordinates of the face's local vertices ─────────────
    let p1_ref;
    let owner_geo: &dyn ReferenceElement = match geo_elem.as_deref() {
        Some(ge) => ge,
        None => {
            p1_ref = mesh.element_type(owner_elem).ref_elem(1);
            &*p1_ref
        }
    };
    let rc = owner_ref_coords(owner_geo, elem_nodes.len());
    let mut face_rc: Vec<Vec<f64>> = Vec::with_capacity(face_nodes.len());
    for &n in face_nodes {
        let local = elem_nodes.iter().position(|&e| e == n)?;
        face_rc.push(rc[local].clone());
    }

    // ── Per-quadrature-point owner geometry ─────────────────────────────────
    let mut nphi = vec![0.0_f64; n_face_nodes];
    let mut ngrad = vec![0.0_f64; n_face_nodes * fdim];
    let mut qps = Vec::with_capacity(fquad.weights.len());
    for (q, xi_face) in fquad.points.iter().enumerate() {
        face_ref.eval_basis(xi_face, &mut nphi);
        face_ref.eval_grad_basis(xi_face, &mut ngrad);

        // ξ_owner = Σᵢ Nᵢ(ξ_face)·ξ_owner,i  and
        // M[c][j] = ∂ξ_owner[c]/∂ξ_face[j] = Σᵢ ∂Nᵢ/∂ξ_face[j]·ξ_owner,i[c]
        let mut xi_ref = vec![0.0_f64; dim];
        let mut m = vec![0.0_f64; dim * fdim];
        for i in 0..n_face_nodes {
            for c in 0..dim {
                xi_ref[c] += nphi[i] * face_rc[i][c];
                for j in 0..fdim {
                    m[c * fdim + j] += ngrad[i * fdim + j] * face_rc[i][c];
                }
            }
        }

        // Owner geometry: physical point, volume Jacobian, det(J).
        let (jac, det_j, x_phys) = match geo_elem.as_deref() {
            Some(ge) => {
                let geo_nds = mesh.geometry_nodes(owner_elem);
                isoparametric_jacobian(mesh, geo_nds, ge, &xi_ref, dim)
            }
            None => {
                let tr = affine_tr.as_ref().unwrap();
                (tr.jacobian().clone(), tr.det_j(), tr.map_to_physical(&xi_ref))
            }
        };

        // Face Jacobian J_face = J_owner · M, its measure and outward normal.
        let mut j_face = vec![0.0_f64; dim * fdim];
        for r in 0..dim {
            for j in 0..fdim {
                let mut s = 0.0;
                for c in 0..dim {
                    s += jac[(r, c)] * m[c * fdim + j];
                }
                j_face[r * fdim + j] = s;
            }
        }
        let measure = face_measure(&j_face, dim, fdim);
        let normal = face_normal(&j_face, dim, fdim);

        qps.push(FaceQpGeometry {
            weight: fquad.weights[q] * measure,
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

    // ── Round-10: hex quad faces and curved boundary measure ────────────────

    /// A single Hex8 element has 6 **quadrilateral** boundary faces.  For the
    /// RT0 DOF attached to face `F`, `b_d = ∫_F g (φ_d·n) dS`; the Piola
    /// transform preserves the reference flux, so with `g ≡ 1` every face
    /// gives `|b_d| = 1` regardless of its area/shape.
    ///
    /// Round-10 regression: `face_quadrature_3d` built 3-D faces from the first
    /// three corner nodes only, so a quad face silently became one of its two
    /// corner triangles (half the measure, wrong normal) and `|b_d| = 0.5`.
    #[test]
    fn hdiv_boundary_flux_hex_quad_faces_analytic() {
        use fem_space::HDivSpace;

        let mesh = Mesh::<3>::unit_cube_hex(1);
        assert_eq!(mesh.n_faces(), 6, "a single hex has 6 boundary quads");
        assert_eq!(mesh.element_type(0), ElementType::Hex8);
        for f in 0..mesh.n_boundary_faces() as u32 {
            assert_eq!(mesh.face_nodes(f).len(), 4, "face {f} must be a quad");
        }

        let rt = HDivSpace::new(mesh.clone(), 0);
        assert_eq!(rt.n_dofs(), 6, "RT0 on one hex → one dof per face");
        let tags: Vec<i32> = (0..mesh.n_boundary_faces() as u32)
            .map(|f| mesh.face_tag(f))
            .collect();
        let b = VectorBoundaryAssembler::assemble_boundary_linear(
            &rt, &[&HdivNormalFluxIntegrator { g: |_| 1.0 }], &tags, 4,
        );

        for (d, &v) in b.iter().enumerate() {
            assert!(
                (v.abs() - 1.0).abs() < 1e-14,
                "face dof {d}: |b| = {} (expected 1 = reference flux)",
                v.abs()
            );
        }
        // ... and the six unit-cube faces have total area 6.
        let total: f64 = b.iter().map(|v| v.abs()).sum();
        assert!((total - 6.0).abs() < 1e-13, "∮ 1 dS = {total} (expected 6)");
    }

    /// Linear measure probe: accumulates the effective quadrature weight
    /// (`qp.weight`, i.e. rule weight × face measure) into local DOF 0, so a
    /// single face yields exactly `∫_F 1 ds`.
    struct MeasureProbe;

    impl VectorBoundaryLinearIntegrator for MeasureProbe {
        fn add_to_face_vector(&self, qp: &VectorBdQpData<'_>, f: &mut [f64]) {
            f[0] += qp.weight;
        }
    }

    /// Curved boundary measure: with an order-2 geometry whose top edge is
    /// bowed into the parabola `y = 1 + a·x(1−x)`, `∫_F 1 ds` must be the ARC
    /// length, not the corner chord.
    ///
    /// Round-10 regression: the face points used to be generated on the
    /// straight corner chord (`face_quadrature_2d`) with the chord measure,
    /// giving 1.0 here instead of ≈1.0982.
    #[test]
    fn boundary_measure_curved_edge_is_arc_length() {
        use fem_space::HDivSpace;

        let mut mesh = Mesh::<2>::unit_square_quad(1);
        mesh.set_curvature(2);
        let d = 0.2_f64; // vertical displacement of the top edge midpoint
        {
            let g = mesh.geometry.as_mut().expect("order-2 geometry");
            let mut moved = 0;
            for n in 0..g.n_nodes {
                let c = &mut g.coords[2 * n..2 * n + 2];
                // Only the top-edge midpoint (the corners stay put, so the
                // curved mesh remains conforming).
                if (c[0] - 0.5).abs() < 1e-14 && (c[1] - 1.0).abs() < 1e-14 {
                    c[1] += d;
                    moved += 1;
                }
            }
            assert_eq!(moved, 1, "expected exactly one top-edge midpoint node");
        }
        assert_eq!(mesh.geom_order(), 2);

        // The top boundary face: both nodes at y = 1 (unchanged by the bowing).
        let top: Vec<u32> = (0..mesh.n_boundary_faces() as u32)
            .filter(|&f| {
                mesh.face_nodes(f)
                    .iter()
                    .all(|&n| (mesh.node_coords(n)[1] - 1.0).abs() < 1e-14)
            })
            .collect();
        assert_eq!(top.len(), 1, "exactly one boundary face is the top edge");
        let tag = mesh.face_tag(top[0]);

        let space = HDivSpace::new(mesh.clone(), 0);
        let b = VectorBoundaryAssembler::assemble_boundary_linear(
            &space, &[&MeasureProbe], &[tag], 6,
        );
        let got: f64 = b.iter().sum();

        // Q2 edge through (0,1), (0.5,1+d), (1,1) is y = 1 + 4d·x(1−x); the
        // arc length of y = 1 + a·x(1−x) over [0,1] is
        //   L = √(1+a²)/2 + asinh(a)/(2a).
        let a = 4.0 * d;
        let exact = 0.5 * (1.0 + a * a).sqrt() + (a + (1.0 + a * a).sqrt()).ln() / (2.0 * a);

        // Same-rule reference: the discrete arc length Σ w_i √(1+y'(x_i)²)
        // evaluated with the face rule the assembler used (order-6 → 4-point
        // Gauss on [0,1]).  This checks the *measure* (the implementation must
        // integrate the curve, not the chord) to machine precision; the
        // (non-polynomial) quadrature error vs the closed-form arc length is
        // ~1.5e-5 and is asserted separately below.
        let rule = fem_element::lagrange::factory::ref_elem(
            fem_element::lagrange::factory::ElemType::Seg,
            1,
        )
        .quadrature(6);
        let gauss_arc: f64 = rule
            .points
            .iter()
            .zip(rule.weights.iter())
            .map(|(s, w)| {
                let x = s[0];
                let dy = a * (1.0 - 2.0 * x);
                w * (1.0 + dy * dy).sqrt()
            })
            .sum();

        eprintln!(
            "curved edge ∫1 ds = {got:.12}, same-rule curve = {gauss_arc:.12}, \
             closed form = {exact:.12}, chord = 1.000000000000"
        );
        assert!(
            (got - gauss_arc).abs() < 1e-13,
            "∫_F 1 ds = {got} (same-rule curve integral = {gauss_arc})"
        );
        assert!(
            (got - exact).abs() < 2e-5,
            "∫_F 1 ds = {got} vs closed-form arc length {exact} \
             (chord measure would give 1.0, i.e. 9.8e-2 off)"
        );
    }

    /// Curved **hex** face measure: a single Hex8 whose order-2 geometry has a
    /// bump on the `z = 1` face,
    /// `z = 1 + b(1−x̃²)(1−ỹ²)` with `x̃ = 2x−1`, `ỹ = 2y−1`.
    ///
    /// The Q2 interpolant reproduces that (bi)quadratic surface exactly, so
    /// `∫_F 1 dS = ∫∫ √(1 + z_x² + z_y²) dx dy` has a known value at any given
    /// quadrature rule; the assembled measure must match it to machine
    /// precision.  (With the old chord/3-corner face quadrature the top face
    /// degenerated into half its area on a flat patch.)
    ///
    /// The geometry is built by hand: `Mesh::set_curvature` must not be used
    /// for this test — see the `set_curvature_hex8` note in the round-10
    /// report (it mis-assigns the order-2 geometry nodes of a Hex8, giving a
    /// degenerate owner Jacobian).
    #[test]
    fn hex_quad_face_measure_curved_isoparametric() {
        use fem_element::lagrange::factory::{ElemType as FEType, ref_elem};
        use fem_space::HDivSpace;

        let b = 0.2_f64;

        // ── single Hex8 topology (unit cube) ────────────────────────────────
        let mut mesh = Mesh::<3>::unit_cube_hex(1);
        let n_vert = mesh.n_nodes();

        // ── replace the geometry with a hand-built order-2 one ──────────────
        let hex2 = ref_elem(FEType::Hex, 2);
        let rc = hex2.dof_coords();
        let npe = rc.len();
        let mut conn = Vec::with_capacity(npe);
        let mut coords = mesh.coords.clone();
        for (d, r) in rc.iter().enumerate() {
            let x = 0.5 * (r[0] + 1.0);
            let y = 0.5 * (r[1] + 1.0);
            let xt = 2.0 * x - 1.0;
            let yt = 2.0 * y - 1.0;
            let z = if (r[2] - 1.0).abs() < 1e-12 {
                1.0 + b * (1.0 - xt * xt) * (1.0 - yt * yt)
            } else {
                0.5 * (r[2] + 1.0)
            };
            coords.extend_from_slice(&[x, y, z]);
            conn.push(n_vert as u32 + d as u32);
        }
        let n_geo = n_vert + npe;
        mesh.geometry = Some(fem_mesh::simplex::GeometryData {
            order: 2,
            conn,
            nodes_per_elem: npe,
            coords,
            n_nodes: n_geo,
        });
        assert_eq!(mesh.geom_order(), 2);

        // ── top face (z = 1, tag 2) ─────────────────────────────────────────
        let top: Vec<u32> = (0..mesh.n_boundary_faces() as u32)
            .filter(|&f| {
                mesh.face_nodes(f)
                    .iter()
                    .all(|&n| (mesh.node_coords(n)[2] - 1.0).abs() < 1e-14)
            })
            .collect();
        assert_eq!(top.len(), 1);
        let tag = mesh.face_tag(top[0]);

        let space = HDivSpace::new(mesh.clone(), 0);
        // quad_order 4 → 3×3 Gauss-Legendre on the face reference square.
        let bvec = VectorBoundaryAssembler::assemble_boundary_linear(
            &space, &[&MeasureProbe], &[tag], 4,
        );
        let got: f64 = bvec.iter().sum();

        // Same-rule reference on the analytic parametrisation (x = s, y = t).
        let rule = fem_element::lagrange::factory::ref_elem(FEType::Quad, 1).quadrature(4);
        let ref_int: f64 = rule
            .points
            .iter()
            .zip(rule.weights.iter())
            .map(|(p, w)| {
                let (xt, yt) = (2.0 * p[0] - 1.0, 2.0 * p[1] - 1.0);
                let dz_dx = -4.0 * b * xt * (1.0 - yt * yt);
                let dz_dy = -4.0 * b * yt * (1.0 - xt * xt);
                w * (1.0 + dz_dx * dz_dx + dz_dy * dz_dy).sqrt()
            })
            .sum();

        eprintln!(
            "curved hex face ∫1 dS = {got:.12}, same-rule curved surface = {ref_int:.12}, \
             flat face = 1.000000000000"
        );
        assert!(
            (got - ref_int).abs() < 1e-13,
            "∫_F 1 dS = {got} (same-rule curved surface integral = {ref_int})"
        );
        assert!(
            got > 1.01,
            "the measure must follow the curved face, got {got} (flat face = 1)"
        );
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
