//! Boundary assembly for vector finite element spaces (H(curl)).
//!
//! Provides [`VectorBoundaryAssembler`] for integrating bilinear and linear
//! forms over boundary faces (edges in 2-D, triangular faces in 3-D) for
//! H(curl) Nédélec elements.
//!
//! ## H(curl) boundary integrals
//!
//! The principal use case is the **tangential mass** operator:
//!
//! ```text
//! a(u, v) = ∫_Γ γ (n×u)·(n×v) dS
//! ```
//!
//! which arises in:
//! - **Silver-Müller absorbing BC**: `n × (μ⁻¹ curl E) + γ (n×E)×n = g`
//! - **Impedance BC**: `n × H = Y (n × E × n)`  with admittance Y
//! - **PML / 1st-order ABC** truncation conditions
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

use fem_element::nedelec::{TetND1, TetND2, TriND1, TriND2};
use fem_element::reference::VectorReferenceElement;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::ElementTransformation;
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::{FESpace, SpaceType};

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
        S: FESpace,
        S::Mesh: MeshTopology,
    {
        let mesh    = space.mesh();
        let dim     = mesh.dim() as usize;
        let n_dofs  = space.n_dofs();
        assert_eq!(
            space.space_type(), SpaceType::HCurl,
            "VectorBoundaryAssembler: only H(curl) spaces are supported"
        );

        // Volume reference element (to evaluate basis on face QPs).
        let vol_elem = vec_ref_elem_hcurl(dim, space.order());
        let n_ldofs  = vol_elem.n_dofs();

        let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);

        let mut ref_phi  = vec![0.0_f64; n_ldofs * dim];
        let mut phys_phi = vec![0.0_f64; n_ldofs * dim];

        for f in mesh.face_iter() {
            if !tags.contains(&mesh.face_tag(f)) { continue; }

            let face_nodes = mesh.face_nodes(f);

            // Find the owning element for this boundary face and compute Jacobian.
            let owner_elem = match find_owner_element(mesh, face_nodes) {
                Some(e) => e,
                None    => continue, // orphan face — skip
            };

            let elem_nodes = mesh.element_nodes(owner_elem);
            let tr = ElementTransformation::from_simplex_nodes(mesh, elem_nodes);
            let j_inv_t = tr.jacobian_inv_t().clone();

            // DOFs and signs for the owner element.
            let global_dofs: Vec<usize> = space
                .element_dofs(owner_elem)
                .iter()
                .map(|&d| d as usize)
                .collect();
            let signs_opt = space.element_signs(owner_elem);

            // Face quadrature on the boundary face (in face parameter space).
            let (face_qp_phys, face_weights, face_normals) =
                face_quadrature(mesh, face_nodes, dim, quad_order);

            let mut k_face = vec![0.0_f64; n_ldofs * n_ldofs];

            for q in 0..face_qp_phys.len() / dim {
                let xp   = &face_qp_phys[q * dim..(q + 1) * dim];
                let w    = face_weights[q];
                let norm = &face_normals[q * dim..(q + 1) * dim];

                // Map physical face QP to reference coordinates.
                let xi_ref = phys_to_ref(mesh, elem_nodes, xp, dim);

                // Evaluate volume basis at reference point.
                vol_elem.eval_basis_vec(&xi_ref, &mut ref_phi);

                // Covariant Piola transform.
                piola_hcurl_basis(&j_inv_t, &ref_phi, &mut phys_phi, n_ldofs, dim);

                // Apply DOF orientation signs.
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
                    weight: w,
                    phi_vec: &phys_phi,
                    normal: norm,
                    x_phys: xp,
                    elem_id: owner_elem,
                    elem_tag: mesh.face_tag(f),
                };

                for integ in integrators {
                    integ.add_to_face_matrix(&qp_data, &mut k_face);
                }
            }

            coo.add_element_matrix(&global_dofs, &k_face);
        }

        coo.into_csr()
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
        S: FESpace,
        S::Mesh: MeshTopology,
    {
        let mesh   = space.mesh();
        let dim    = mesh.dim() as usize;
        let n_dofs = space.n_dofs();
        assert_eq!(
            space.space_type(), SpaceType::HCurl,
            "VectorBoundaryAssembler: only H(curl) spaces are supported"
        );

        let vol_elem = vec_ref_elem_hcurl(dim, space.order());
        let n_ldofs  = vol_elem.n_dofs();

        let mut rhs = vec![0.0_f64; n_dofs];
        let mut ref_phi  = vec![0.0_f64; n_ldofs * dim];
        let mut phys_phi = vec![0.0_f64; n_ldofs * dim];

        for f in mesh.face_iter() {
            if !tags.contains(&mesh.face_tag(f)) { continue; }

            let face_nodes = mesh.face_nodes(f);
            let owner_elem = match find_owner_element(mesh, face_nodes) {
                Some(e) => e,
                None    => continue,
            };

            let elem_nodes = mesh.element_nodes(owner_elem);
            let tr = ElementTransformation::from_simplex_nodes(mesh, elem_nodes);
            let j_inv_t = tr.jacobian_inv_t().clone();

            let global_dofs: Vec<usize> = space
                .element_dofs(owner_elem)
                .iter()
                .map(|&d| d as usize)
                .collect();
            let signs_opt = space.element_signs(owner_elem);

            let (face_qp_phys, face_weights, face_normals) =
                face_quadrature(mesh, face_nodes, dim, quad_order);

            let mut f_face = vec![0.0_f64; n_ldofs];

            for q in 0..face_qp_phys.len() / dim {
                let xp   = &face_qp_phys[q * dim..(q + 1) * dim];
                let w    = face_weights[q];
                let norm = &face_normals[q * dim..(q + 1) * dim];

                let xi_ref = phys_to_ref(mesh, elem_nodes, xp, dim);
                vol_elem.eval_basis_vec(&xi_ref, &mut ref_phi);

                piola_hcurl_basis(&j_inv_t, &ref_phi, &mut phys_phi, n_ldofs, dim);

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
                    weight: w,
                    phi_vec: &phys_phi,
                    normal: norm,
                    x_phys: xp,
                    elem_id: owner_elem,
                    elem_tag: mesh.face_tag(f),
                };

                for integ in integrators {
                    integ.add_to_face_vector(&qp_data, &mut f_face);
                }
            }

            for (&d, &v) in global_dofs.iter().zip(f_face.iter()) {
                rhs[d] += v;
            }
        }

        rhs
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

fn vec_ref_elem_hcurl(dim: usize, order: u8) -> Box<dyn VectorReferenceElement> {
    match (dim, order) {
        (2, 1) => Box::new(TriND1),
        (2, 2) => Box::new(TriND2),
        (3, 1) => Box::new(TetND1),
        (3, 2) => Box::new(TetND2),
        _ => panic!("VectorBoundaryAssembler: unsupported (dim={dim}, order={order})"),
    }
}

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
    use fem_mesh::SimplexMesh;
    use fem_space::HCurlSpace;

    /// The tangential mass matrix over the full boundary of the unit square
    /// must be symmetric.
    #[test]
    fn tangential_mass_symmetric_2d() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
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
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
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
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
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
        use crate::coefficient::ConstantMatrixCoeff;

        let mesh1 = SimplexMesh::<2>::unit_square_tri(4);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
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
        use crate::coefficient::ConstantMatrixCoeff;

        let mesh1 = SimplexMesh::<2>::unit_square_tri(4);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
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
}
