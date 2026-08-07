//! Surface finite element method for solving PDEs on 2-D manifolds in 3-D.
//!
//! Provides integrators that work with `Mesh<3>` containing Tri3 elements
//! (2-D topology embedded in 3-D space).  Key differences from planar FEM:
//!
//! - Jacobian is `3×2` instead of `2×2`
//! - Gradient transform uses pseudo-inverse `(JᵀJ)⁻¹Jᵀ` instead of `J⁻ᵀ`
//! - Area element is `√det(JᵀJ)` instead of `|det(J)|`
//! - Surface normal `n = (J₀ × J�? / |J₀ × J₁|`
//!
//! # Example
//! ```ignore
//! use fem_mesh::Mesh;
//! use fem_space::H1Space;
//! use fem_assembly::surface::{SurfaceAssembler, SurfaceDiffusionIntegrator};
//!
//! let mesh = sphere_mesh(2);  // icosahedral sphere
//! let space = H1Space::new(mesh, 1);
//! let mat = SurfaceAssembler::assemble_bilinear(&space, &[&SurfaceDiffusionIntegrator]);
//! ```

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;

// ─── Surface Jacobian helpers ────────────────────────────────────────────────

/// Compute the 3×2 Jacobian for a triangle embedded in 3-D.
///
/// Returns `(J, G, sqrt_det_G, normal)` where:
/// - `J` is the 3×2 Jacobian matrix `[x1-x0, x2-x0]`
/// - `G = JᵀJ` is the 2×2 metric tensor
/// - `sqrt_det_G = √det(G)` is the surface area scaling (`dS = √det(G) dξ`)
/// - `normal` is the unit surface normal (3-D vector)
#[allow(clippy::type_complexity)]
fn surface_jacobian(x: &[[f64; 3]; 3]) -> ([[f64; 3]; 2], [[f64; 2]; 2], f64, [f64; 3]) {
    // J = [x1-x0, x2-x0]  �?3×2 matrix stored as 2 columns of 3 components
    let j0 = [x[1][0] - x[0][0], x[1][1] - x[0][1], x[1][2] - x[0][2]];
    let j1 = [x[2][0] - x[0][0], x[2][1] - x[0][1], x[2][2] - x[0][2]];
    let j = [j0, j1];

    // G = JᵀJ  �?2×2 metric
    let g00 = j0[0]*j0[0] + j0[1]*j0[1] + j0[2]*j0[2];
    let g01 = j0[0]*j1[0] + j0[1]*j1[1] + j0[2]*j1[2];
    let g10 = g01;
    let g11 = j1[0]*j1[0] + j1[1]*j1[1] + j1[2]*j1[2];
    let g = [[g00, g01], [g10, g11]];

    // det(G) and sqrt(det(G))
    let det_g = g00 * g11 - g01 * g01;
    let sqrt_det_g = det_g.sqrt().max(1e-30);

    // Unit normal n = (J₀ × J�? / |J₀ × J₁|
    let nx = j0[1]*j1[2] - j0[2]*j1[1];
    let ny = j0[2]*j1[0] - j0[0]*j1[2];
    let nz = j0[0]*j1[1] - j0[1]*j1[0];
    let n_len = (nx*nx + ny*ny + nz*nz).sqrt().max(1e-30);
    let normal = [nx/n_len, ny/n_len, nz/n_len];

    (j, g, sqrt_det_g, normal)
}

/// Compute the pseudo-inverse J_pinv = (JᵀJ)⁻¹Jᵀ  �?2×3 matrix.
///
/// Used for surface gradient: `∇_Γ u = J_pinvᵀ · ∇_ξ u = G⁻�?· Jᵀ · ∇_ξ u`
fn pseudo_inverse(j: &[[f64; 3]; 2], g: &[[f64; 2]; 2], det_g: f64) -> [[f64; 2]; 3] {
    let inv_det = 1.0 / det_g.max(1e-30);
    let g_inv_00 = g[1][1] * inv_det;
    let g_inv_01 = -g[0][1] * inv_det;
    let g_inv_10 = g_inv_01;
    let g_inv_11 = g[0][0] * inv_det;

    // J_pinv = G⁻�?· Jᵀ  (2×3)
    let mut p = [[0.0; 2]; 3]; // stored as 3 rows × 2 cols
    for r in 0..3 {
        p[r][0] = g_inv_00 * j[0][r] + g_inv_01 * j[1][r];
        p[r][1] = g_inv_10 * j[0][r] + g_inv_11 * j[1][r];
    }
    p
}

// ─── Surface integrators ─────────────────────────────────────────────────────

/// Surface diffusion (Laplace-Beltrami) bilinear form: `∫_Γ ∇_Γ u · ∇_Γ v dS`
pub struct SurfaceDiffusionIntegrator;

impl SurfaceDiffusionIntegrator {
    pub fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 3], k_elem: &mut [f64; 9]) {
        let (j, g, sqrt_det_g, _normal) = surface_jacobian(elem_nodes);
        let det_g = g[0][0]*g[1][1] - g[0][1]*g[1][0];
        let pinv = pseudo_inverse(&j, &g, det_g);

        // Reference gradients for P1 triangle: ∇φ₁=(-1,-1), ∇φ₂=(1,0), ∇φ₃=(0,1)
        let ref_grad = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]];

        // Surface gradients: ∇_Γ φ_i = J_pinvᵀ · ∇_ξ φ_i = (∇_ξ φ_i) · J_pinv
        let mut sg = [[0.0; 3]; 3]; // 3 DOFs × 3 components
        for i in 0..3 {
            for c in 0..3 {
                sg[i][c] = ref_grad[i][0] * pinv[c][0] + ref_grad[i][1] * pinv[c][1];
            }
        }

        // One-point quadrature at centroid (weight = 0.5 for reference triangle)
        // with surface measure dS = sqrt_det_G * area_ref = sqrt_det_G * 0.5
        let area_factor = 0.5 * sqrt_det_g; // det(G)^(1/2) * area of ref tri

        for i in 0..3 {
            for j in 0..3 {
                let mut dot = 0.0;
                for c in 0..3 { dot += sg[i][c] * sg[j][c]; }
                k_elem[i * 3 + j] += dot * area_factor;
            }
        }
    }
}

/// Surface mass bilinear form: `∫_Γ u v dS`
pub struct SurfaceMassIntegrator;

impl SurfaceMassIntegrator {
    pub fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 3], k_elem: &mut [f64; 9]) {
        let (_j, _g, sqrt_det_g, _normal) = surface_jacobian(elem_nodes);
        let area_factor = 0.5 * sqrt_det_g; // dS = sqrt(det(G)) * dξ (area of ref tri = 0.5)

        // Mass matrix for P1: M[i,j] = �?φ_i·φ_j dS
        // One-point quadrature at centroid (ξ=1/3, η=1/3): φ_i = 1/3 for all i
        let phi_qp = 1.0 / 3.0;
        let _val = phi_qp * phi_qp * area_factor;

        // 3-point quadrature for better accuracy
        // Using centroid + edge midpoints (stiffness accuracy is fine)
        // Structured quadrature: 3 points at (1/2,0), (0,1/2), (1/2,1/2) with weights 1/6 each
        let qpts = [[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]];
        let qwt = [1.0/6.0, 1.0/6.0, 1.0/6.0];

        for i in 0..3 {
            for j in 0..3 {
                let mut val = 0.0;
                for q in 0..3 {
                    let (xi, eta) = (qpts[q][0], qpts[q][1]);
                    let phi_i = [1.0 - xi - eta, xi, eta][i];
                    let phi_j = [1.0 - xi - eta, xi, eta][j];
                    val += phi_i * phi_j * qwt[q];
                }
                k_elem[i * 3 + j] += val * sqrt_det_g;
            }
        }
    }
}

/// Surface domain source linear form: `∫_Γ f(x) v(x) dS`
pub struct SurfaceDomainSourceIntegrator<'a> {
    pub f: &'a dyn Fn(&[f64; 3]) -> f64,
}

impl SurfaceDomainSourceIntegrator<'_> {
    pub fn add_to_element_vector(&self, elem_nodes: &[[f64; 3]; 3], f_elem: &mut [f64; 3]) {
        let (_j, _g, sqrt_det_g, _normal) = surface_jacobian(elem_nodes);

        // One-point quadrature at centroid
        let centroid = [
            (elem_nodes[0][0] + elem_nodes[1][0] + elem_nodes[2][0]) / 3.0,
            (elem_nodes[0][1] + elem_nodes[1][1] + elem_nodes[2][1]) / 3.0,
            (elem_nodes[0][2] + elem_nodes[1][2] + elem_nodes[2][2]) / 3.0,
        ];
        let f_val = (self.f)(&centroid);
        let area_factor = 0.5 * sqrt_det_g;

        // At centroid, φ_i = 1/3
        let phi_qp = 1.0 / 3.0;
        for i in 0..3 {
            f_elem[i] += f_val * phi_qp * area_factor;
        }
    }
}

// ─── Surface integrator traits (MFEM-aligned API) ────────────────────────────

/// Trait for Tri3 surface bilinear integrators (Laplace-Beltrami, mass, �?.
///
/// Matches MFEM's pattern where integrators implement a common interface
/// and are added to the form via `AddDomainIntegrator`.
pub trait SurfaceBilinearIntegrator {
    fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 3], k_elem: &mut [f64; 9]);
}

/// Trait for Tri3 surface linear integrators (source terms).
pub trait SurfaceLinearIntegrator {
    fn add_to_element_vector(&self, elem_nodes: &[[f64; 3]; 3], f_elem: &mut [f64; 3]);
}

/// Trait for Tri6 (P2) surface bilinear integrators — 6-node triangles.
pub trait SurfaceTri6BilinearIntegrator {
    fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 6], k_elem: &mut [f64; 36]);
}

/// Trait for Tri6 (P2) surface linear integrators.
pub trait SurfaceTri6LinearIntegrator {
    fn add_to_element_vector(&self, elem_nodes: &[[f64; 3]; 6], f_elem: &mut [f64; 6]);
}

/// Trait for Quad4 surface bilinear integrators.
pub trait SurfaceQuad4BilinearIntegrator {
    fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 4], k_elem: &mut [f64; 16]);
}

/// Trait for Quad4 surface linear integrators.
pub trait SurfaceQuad4LinearIntegrator {
    fn add_to_element_vector(&self, elem_nodes: &[[f64; 3]; 4], f_elem: &mut [f64; 4]);
}

// Implement traits for existing integrators.

impl SurfaceBilinearIntegrator for SurfaceDiffusionIntegrator {
    fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 3], k_elem: &mut [f64; 9]) {
        self.add_to_element_matrix(elem_nodes, k_elem);
    }
}

impl SurfaceBilinearIntegrator for SurfaceMassIntegrator {
    fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 3], k_elem: &mut [f64; 9]) {
        self.add_to_element_matrix(elem_nodes, k_elem);
    }
}

impl SurfaceQuad4BilinearIntegrator for SurfaceQuad4DiffusionIntegrator {
    fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 4], k_elem: &mut [f64; 16]) {
        self.add_to_element_matrix(elem_nodes, k_elem);
    }
}

impl SurfaceQuad4BilinearIntegrator for SurfaceQuad4MassIntegrator {
    fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 4], k_elem: &mut [f64; 16]) {
        self.add_to_element_matrix(elem_nodes, k_elem);
    }
}

// Linear integrator traits (source terms for Tri3 surfaces).

impl SurfaceLinearIntegrator for SurfaceDomainSourceIntegrator<'_> {
    fn add_to_element_vector(&self, elem_nodes: &[[f64; 3]; 3], f_elem: &mut [f64; 3]) {
        self.add_to_element_vector(elem_nodes, f_elem);
    }
}

impl SurfaceQuad4LinearIntegrator for SurfaceQuad4DomainSourceIntegrator<'_> {
    fn add_to_element_vector(&self, elem_nodes: &[[f64; 3]; 4], f_elem: &mut [f64; 4]) {
        self.add_to_element_vector(elem_nodes, f_elem);
    }
}

// ─── Surface Assembler ───────────────────────────────────────────────────────

/// Assemble a surface bilinear form using a surface integrator.
///
/// Works with `H1Space<Mesh<3>>` containing Tri3 surface elements.
pub struct SurfaceAssembler;

impl SurfaceAssembler {
    /// Assemble a surface bilinear form from a slice of integrators.
    ///
    /// Each integrator's contribution is accumulated into the same element
    /// matrix, matching MFEM's `BilinearForm::AddDomainIntegrator`.
    pub fn assemble_bilinear<S: FESpace>(
        space: &S,
        integrators: &[&dyn SurfaceBilinearIntegrator],
    ) -> CsrMatrix<f64> {
        let mesh = space.mesh();
        let n_dofs = space.n_dofs();
        let ne = mesh.n_elements() as u32;
        let mut coo = CooMatrix::new(n_dofs, n_dofs);

        for e in 0..ne {
            let dofs = space.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            if nodes.len() < 3 { continue; }
            let x: [[f64; 3]; 3] = [
                get_coord3(mesh, nodes[0]),
                get_coord3(mesh, nodes[1]),
                get_coord3(mesh, nodes[2]),
            ];
            let mut ke = [0.0; 9];
            for integ in integrators {
                integ.add_to_element_matrix(&x, &mut ke);
            }
            for i in 0..3 {
                for j in 0..3 {
                    coo.add(dofs[i] as usize, dofs[j] as usize, ke[i * 3 + j]);
                }
            }
        }
        coo.into_csr()
    }

    /// MFEM-style linear assembly: add all linear integrators in a slice.
    pub fn assemble_linear<S: FESpace>(
        space: &S,
        integrators: &[&dyn SurfaceLinearIntegrator],
    ) -> Vec<f64> {
        let mesh = space.mesh();
        let n_dofs = space.n_dofs();
        let ne = mesh.n_elements() as u32;
        let mut rhs = vec![0.0; n_dofs];

        for e in 0..ne {
            let dofs = space.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            if nodes.len() < 3 { continue; }
            let x: [[f64; 3]; 3] = [
                get_coord3(mesh, nodes[0]),
                get_coord3(mesh, nodes[1]),
                get_coord3(mesh, nodes[2]),
            ];
            let mut fe = [0.0; 3];
            for integ in integrators {
                integ.add_to_element_vector(&x, &mut fe);
            }
            for i in 0..3 {
                rhs[dofs[i] as usize] += fe[i];
            }
        }
        rhs
    }
}

pub(crate) fn get_coord3<M: MeshTopology>(mesh: &M, n: u32) -> [f64; 3] {
    let c = mesh.node_coords(n);
    [c[0], c[1], if c.len() > 2 { c[2] } else { 0.0 }]
}

// ─── Quad4 surface helpers ───────────────────────────────────────────────────

/// Surface Jacobian for a bilinear Quad4 (4-node quadrilateral) embedded in 3-D.
///
/// Evaluated at the centroid (ξ=0, η=0) of the reference square [-1,1]².
/// Returns `(J, sqrt_det_G, normal)` where J is the 3×2 Jacobian.
fn surface_jacobian_quad4(x: &[[f64; 3]; 4]) -> ([[f64; 3]; 2], f64, [f64; 3]) {
    // Bilinear shape function reference gradients at centroid:
    //   ∂N_i/∂�?= sign_ξ_i / 4,  ∂N_i/∂�?= sign_η_i / 4
    //   sign_ξ:  [-1,  1,  1, -1]
    //   sign_η:  [-1, -1,  1,  1]
    let dxi = [
        (-x[0][0] + x[1][0] + x[2][0] - x[3][0]) / 4.0,
        (-x[0][1] + x[1][1] + x[2][1] - x[3][1]) / 4.0,
        (-x[0][2] + x[1][2] + x[2][2] - x[3][2]) / 4.0,
    ];
    let deta = [
        (-x[0][0] - x[1][0] + x[2][0] + x[3][0]) / 4.0,
        (-x[0][1] - x[1][1] + x[2][1] + x[3][1]) / 4.0,
        (-x[0][2] - x[1][2] + x[2][2] + x[3][2]) / 4.0,
    ];
    let j = [dxi, deta];

    // Metric G = J^T * J (2×2)
    let g00 = dxi[0]*dxi[0] + dxi[1]*dxi[1] + dxi[2]*dxi[2];
    let g01 = dxi[0]*deta[0] + dxi[1]*deta[1] + dxi[2]*deta[2];
    let g11 = deta[0]*deta[0] + deta[1]*deta[1] + deta[2]*deta[2];
    let det_g = g00 * g11 - g01 * g01;
    let sqrt_det_g = det_g.sqrt().max(1e-30);

    // Unit normal n = (J_ξ × J_η) / |J_ξ × J_η|
    let nx = dxi[1]*deta[2] - dxi[2]*deta[1];
    let ny = dxi[2]*deta[0] - dxi[0]*deta[2];
    let nz = dxi[0]*deta[1] - dxi[1]*deta[0];
    let n_len = (nx*nx + ny*ny + nz*nz).sqrt().max(1e-30);
    let normal = [nx/n_len, ny/n_len, nz/n_len];

    (j, sqrt_det_g, normal)
}

/// Pseudo-inverse J_pinv = (J^T J)^{-1} J^T for a 3×2 Jacobian (Quad4 centroid).
fn pseudo_inverse_quad4(j: &[[f64; 3]; 2], det_g: f64) -> [[f64; 2]; 3] {
    let inv_det = 1.0 / det_g.max(1e-30);
    // G = J^T J, G^{-1} = [[j11, -j01], [-j10, j00]] / det
    let g00 = j[0][0]*j[0][0] + j[0][1]*j[0][1] + j[0][2]*j[0][2];
    let g01 = j[0][0]*j[1][0] + j[0][1]*j[1][1] + j[0][2]*j[1][2];
    let g11 = j[1][0]*j[1][0] + j[1][1]*j[1][1] + j[1][2]*j[1][2];
    let ginvt = [[g11 * inv_det, -g01 * inv_det], [-g01 * inv_det, g00 * inv_det]];

    // J_pinv = G^{-1} * J^T  (2×3) �?stored as 3 rows × 2 cols
    let mut p = [[0.0; 2]; 3];
    for r in 0..3 {
        p[r][0] = ginvt[0][0] * j[0][r] + ginvt[0][1] * j[1][r];
        p[r][1] = ginvt[1][0] * j[0][r] + ginvt[1][1] * j[1][r];
    }
    p
}

// ─── Quad4 surface integrators ───────────────────────────────────────────────

/// Surface diffusion (Laplace-Beltrami) bilinear form for Quad4: `∫_Γ ∇_Γ u · ∇_Γ v dS`
pub struct SurfaceQuad4DiffusionIntegrator;

impl SurfaceQuad4DiffusionIntegrator {
    pub fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 4], k_elem: &mut [f64; 16]) {
        let (j, sqrt_det_g, _normal) = surface_jacobian_quad4(elem_nodes);
        let det_g = {
            let g00 = j[0][0]*j[0][0] + j[0][1]*j[0][1] + j[0][2]*j[0][2];
            let g01 = j[0][0]*j[1][0] + j[0][1]*j[1][1] + j[0][2]*j[1][2];
            let g11 = j[1][0]*j[1][0] + j[1][1]*j[1][1] + j[1][2]*j[1][2];
            g00 * g11 - g01 * g01
        };
        let pinv = pseudo_inverse_quad4(&j, det_g);

        // Reference gradients of Q1 bilinear at centroid:
        //   ∇N₀ = (-¼, -¼),  ∇N�?= (¼, -¼),  ∇N�?= (¼, ¼),  ∇N�?= (-¼, ¼)
        let ref_grad = [[-0.25, -0.25], [0.25, -0.25], [0.25, 0.25], [-0.25, 0.25]];

        // Surface gradients: ∇_Γ N_i = J_pinv^T · ∇_ξ N_i
        let mut sg = [[0.0; 3]; 4];
        for i in 0..4 {
            for c in 0..3 {
                sg[i][c] = ref_grad[i][0] * pinv[c][0] + ref_grad[i][1] * pinv[c][1];
            }
        }

        // dS = sqrt(det(G)) * dξ dη  (area of ref quad = 4, weight = 4 at centroid)
        let area_factor = 4.0 * sqrt_det_g;

        for i in 0..4 {
            for j in 0..4 {
                let dot = sg[i][0]*sg[j][0] + sg[i][1]*sg[j][1] + sg[i][2]*sg[j][2];
                k_elem[i * 4 + j] += dot * area_factor;
            }
        }
    }
}

/// Surface mass bilinear form for Quad4: `∫_Γ u v dS`
pub struct SurfaceQuad4MassIntegrator;

impl SurfaceQuad4MassIntegrator {
    pub fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 4], k_elem: &mut [f64; 16]) {
        let (_j, sqrt_det_g, _normal) = surface_jacobian_quad4(elem_nodes);

        // 2×2 Gauss quadrature on [-1,1]²
        let qpts = [
            [-0.5773502691896257, -0.5773502691896257],
            [ 0.5773502691896257, -0.5773502691896257],
            [ 0.5773502691896257,  0.5773502691896257],
            [-0.5773502691896257,  0.5773502691896257],
        ];
        let qwt = [1.0, 1.0, 1.0, 1.0]; // weights sum to 4

        for i in 0..4 {
            for j in 0..4 {
                let mut val = 0.0;
                for q in 0..4 {
                    let (xi, eta) = (qpts[q][0], qpts[q][1]);
                    let phi_i = 0.25 * (1.0 + match i {
                        0 | 3 => -xi, _ => xi,
                    }) * (1.0 + match i {
                        0 | 1 => -eta, _ => eta,
                    });
                    let phi_j = 0.25 * (1.0 + match j {
                        0 | 3 => -xi, _ => xi,
                    }) * (1.0 + match j {
                        0 | 1 => -eta, _ => eta,
                    });
                    val += phi_i * phi_j * qwt[q];
                }
                k_elem[i * 4 + j] += val * sqrt_det_g;
            }
        }
    }
}

/// Surface domain source linear form for Quad4: `∫_Γ f(x) v(x) dS`
pub struct SurfaceQuad4DomainSourceIntegrator<'a> {
    pub f: &'a dyn Fn(&[f64; 3]) -> f64,
}

impl SurfaceQuad4DomainSourceIntegrator<'_> {
    pub fn add_to_element_vector(&self, elem_nodes: &[[f64; 3]; 4], f_elem: &mut [f64; 4]) {
        let (_j, sqrt_det_g, _normal) = surface_jacobian_quad4(elem_nodes);

        // Centroid quadrature
        let centroid = [
            (elem_nodes[0][0] + elem_nodes[1][0] + elem_nodes[2][0] + elem_nodes[3][0]) / 4.0,
            (elem_nodes[0][1] + elem_nodes[1][1] + elem_nodes[2][1] + elem_nodes[3][1]) / 4.0,
            (elem_nodes[0][2] + elem_nodes[1][2] + elem_nodes[2][2] + elem_nodes[3][2]) / 4.0,
        ];
        let f_val = (self.f)(&centroid);
        let area_factor = 4.0 * sqrt_det_g; // dS at centroid * area of ref quad

        // At centroid, φ_i = 1/4 for Q1
        let phi_qp = 1.0 / 4.0;
        for i in 0..4 {
            f_elem[i] += f_val * phi_qp * area_factor;
        }
    }
}

// ─── Quad4 Surface Assembler ─────────────────────────────────────────────────

/// Assemble a surface bilinear form on Quad4 elements.
pub struct SurfaceQuad4Assembler;

impl SurfaceQuad4Assembler {
    /// MFEM-style bilinear assembly for Quad4 surfaces.
    pub fn assemble_bilinear<S: FESpace>(
        space: &S,
        integrators: &[&dyn SurfaceQuad4BilinearIntegrator],
    ) -> CsrMatrix<f64> {
        let mesh = space.mesh();
        let n_dofs = space.n_dofs();
        let ne = mesh.n_elements() as u32;
        let mut coo = CooMatrix::new(n_dofs, n_dofs);

        for e in 0..ne {
            let dofs = space.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            if nodes.len() < 4 { continue; }
            let x: [[f64; 3]; 4] = [
                get_coord3(mesh, nodes[0]),
                get_coord3(mesh, nodes[1]),
                get_coord3(mesh, nodes[2]),
                get_coord3(mesh, nodes[3]),
            ];
            let mut ke = [0.0; 16];
            for integ in integrators {
                integ.add_to_element_matrix(&x, &mut ke);
            }
            for i in 0..4 {
                for j in 0..4 {
                    coo.add(dofs[i] as usize, dofs[j] as usize, ke[i * 4 + j]);
                }
            }
        }
        coo.into_csr()
    }

    /// MFEM-style linear assembly for Quad4 surfaces.
    pub fn assemble_linear<S: FESpace>(
        space: &S,
        integrators: &[&dyn SurfaceQuad4LinearIntegrator],
    ) -> Vec<f64> {
        let mesh = space.mesh();
        let n_dofs = space.n_dofs();
        let ne = mesh.n_elements() as u32;
        let mut rhs = vec![0.0; n_dofs];

        for e in 0..ne {
            let dofs = space.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            if nodes.len() < 4 { continue; }
            let x: [[f64; 3]; 4] = [
                get_coord3(mesh, nodes[0]),
                get_coord3(mesh, nodes[1]),
                get_coord3(mesh, nodes[2]),
                get_coord3(mesh, nodes[3]),
            ];
            let mut fe = [0.0; 4];
            for integ in integrators {
                integ.add_to_element_vector(&x, &mut fe);
            }
            for i in 0..4 {
                rhs[dofs[i] as usize] += fe[i];
            }
        }
        rhs
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Quad9 (Q2, 9-node) surface integrators — matches MFEM's SetCurvature(2)
// ═══════════════════════════════════════════════════════════════════════════

// ─── Q2 basis on [-1, 1]² ─────────────────────────────────────────────────

/// 1-D quadratic Lagrange basis on [-1, 1] with nodes at {-1, 0, +1}.
fn lagrange_q1d(x: f64) -> [f64; 3] {
    [x * (x - 1.0) / 2.0,   // ℓ₀, node at -1
     (1.0 - x) * (1.0 + x), // ℓ₁, node at  0
     x * (x + 1.0) / 2.0]   // ℓ₂, node at +1
}

fn lagrange_q1d_deriv(x: f64) -> [f64; 3] {
    [x - 0.5,   // ℓ₀′
     -2.0 * x,  // ℓ₁′
     x + 0.5]   // ℓ₂′
}

/// Evaluate all 9 Q2 Lagrangian basis functions at (ξ, η) on [-1, 1]².
///
/// Node ordering (tensor-product):
///   0: (-1,-1)  1: (+1,-1)  2: (+1,+1)  3: (-1,+1)
///   4: ( 0,-1)  5: (+1, 0)  6: ( 0,+1)  7: (-1, 0)
///   8: ( 0, 0)
/// Basis: φ_{i×3+j}(ξ,η) = ℓ_i(ξ) · ℓ_j(η)
pub fn q2_basis(xi: f64, eta: f64) -> [f64; 9] {
    let lx = lagrange_q1d(xi);
    let ly = lagrange_q1d(eta);
    // MFEM H1_FECollection order-2 (Q2) DOF order — vertex-edge-face:
    //   corners (ξ,η): (-1,-1), (1,-1), (1,1), (-1,1)
    //   edge mids:     (0,-1), (1,0), (0,1), (-1,0)   [bottom,right,top,left]
    //   center:        (0,0)
    // (NOT the lexicographic tensor order — that misaligns the element
    //  matrix with the vertex-first global DOF numbering and breaks A/B.)
    [
        lx[0] * ly[0], // 0: (-1,-1)
        lx[2] * ly[0], // 1: ( 1,-1)
        lx[2] * ly[2], // 2: ( 1, 1)
        lx[0] * ly[2], // 3: (-1, 1)
        lx[1] * ly[0], // 4: ( 0,-1)
        lx[2] * ly[1], // 5: ( 1, 0)
        lx[1] * ly[2], // 6: ( 0, 1)
        lx[0] * ly[1], // 7: (-1, 0)
        lx[1] * ly[1], // 8: ( 0, 0)
    ]
}

/// Derivatives ∂φ/∂ξ and ∂φ/∂η for the 9-node Lagrangian Q2 basis
/// (MFEM vertex-edge-face DOF order, see [`q2_basis`]).
fn q2_basis_derivs(xi: f64, eta: f64) -> ([f64; 9], [f64; 9]) {
    let lx  = lagrange_q1d(xi);
    let ly  = lagrange_q1d(eta);
    let dlx = lagrange_q1d_deriv(xi);
    let dly = lagrange_q1d_deriv(eta);
    // corner (-1,-1): dlx0*ly0, lx0*dly0
    let c00 = (dlx[0] * ly[0], lx[0] * dly[0]);
    let c10 = (dlx[2] * ly[0], lx[2] * dly[0]);
    let c11 = (dlx[2] * ly[2], lx[2] * dly[2]);
    let c01 = (dlx[0] * ly[2], lx[0] * dly[2]);
    // edge mids (0,-1),(1,0),(0,1),(-1,0)
    let e_b = (dlx[1] * ly[0], lx[1] * dly[0]);
    let e_r = (dlx[2] * ly[1], lx[2] * dly[1]);
    let e_t = (dlx[1] * ly[2], lx[1] * dly[2]);
    let e_l = (dlx[0] * ly[1], lx[0] * dly[1]);
    // center (0,0)
    let ce = (dlx[1] * ly[1], lx[1] * dly[1]);
    (
        [c00.0, c10.0, c11.0, c01.0, e_b.0, e_r.0, e_t.0, e_l.0, ce.0],
        [c00.1, c10.1, c11.1, c01.1, e_b.1, e_r.1, e_t.1, e_l.1, ce.1],
    )
}

/// Surface Jacobian for Q2 element at (ξ, η).
pub fn q2_jacobian_at(x: &[[f64; 3]; 9], xi: f64, eta: f64) -> [[f64; 3]; 2] {
    let (dxi, det) = q2_basis_derivs(xi, eta);
    let mut j0 = [0.0; 3]; let mut j1 = [0.0; 3];
    for i in 0..9 {
        for c in 0..3 {
            j0[c] += dxi[i] * x[i][c];
            j1[c] += det[i] * x[i][c];
        }
    }
    [j0, j1]
}

/// Metric tensor and surface area factor from a 2×3 Jacobian.
fn q2_metric_at(j: &[[f64; 3]; 2]) -> (f64, f64) {
    let g00 = j[0][0]*j[0][0] + j[0][1]*j[0][1] + j[0][2]*j[0][2];
    let g01 = j[0][0]*j[1][0] + j[0][1]*j[1][1] + j[0][2]*j[1][2];
    let g11 = j[1][0]*j[1][0] + j[1][1]*j[1][1] + j[1][2]*j[1][2];
    let det_g = g00 * g11 - g01 * g01;
    (det_g, det_g.sqrt().max(1e-30))
}

/// Pseudo-inverse for a 3×2 Jacobian (Quad9 variant).
fn pseudo_inverse_quad9(j: &[[f64; 3]; 2], det_g: f64) -> [[f64; 2]; 3] {
    let inv_det = 1.0 / det_g.max(1e-30);
    let g00 = j[0][0]*j[0][0] + j[0][1]*j[0][1] + j[0][2]*j[0][2];
    let g01 = j[0][0]*j[1][0] + j[0][1]*j[1][1] + j[0][2]*j[1][2];
    let g11 = j[1][0]*j[1][0] + j[1][1]*j[1][1] + j[1][2]*j[1][2];
    let ginvt = [[g11 * inv_det, -g01 * inv_det], [-g01 * inv_det, g00 * inv_det]];
    let mut p = [[0.0; 2]; 3];
    for r in 0..3 {
        p[r][0] = ginvt[0][0] * j[0][r] + ginvt[0][1] * j[1][r];
        p[r][1] = ginvt[1][0] * j[0][r] + ginvt[1][1] * j[1][r];
    }
    p
}

/// Surface gradients of Q2 basis at a point with pseudo-inverse.
fn q2_surface_gradients_at(pinv: &[[f64; 2]; 3], xi: f64, eta: f64) -> [[f64; 3]; 9] {
    let (dxi, det) = q2_basis_derivs(xi, eta);
    let mut sg = [[0.0; 3]; 9];
    for i in 0..9 {
        for c in 0..3 {
            sg[i][c] = dxi[i] * pinv[c][0] + det[i] * pinv[c][1];
        }
    }
    sg
}

// ─── Traits ───────────────────────────────────────────────────────────────

/// Bilinear form on Quad9: adds to a 9×9 element matrix.
pub trait SurfaceQuad9BilinearIntegrator {
    fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 9], k_elem: &mut [f64; 81]);
}

/// Linear form on Quad9: adds to a 9-element vector.
pub trait SurfaceQuad9LinearIntegrator {
    fn add_to_element_vector(&self, elem_nodes: &[[f64; 3]; 9], f_elem: &mut [f64; 9]);
}

// ─── 3×3 Gauss-Legendre points on [-1, 1]² ────────────────────────────────

const GL3: [[f64; 2]; 9] = [
    [-0.7745966692414834, -0.7745966692414834],
    [ 0.7745966692414834, -0.7745966692414834],
    [ 0.7745966692414834,  0.7745966692414834],
    [-0.7745966692414834,  0.7745966692414834],
    [ 0.0,                -0.7745966692414834],
    [ 0.7745966692414834,  0.0               ],
    [ 0.0,                 0.7745966692414834],
    [-0.7745966692414834,  0.0               ],
    [ 0.0,                 0.0               ],
];
const GL3_W: [f64; 3] = [0.5555555555555556, 0.8888888888888888, 0.5555555555555556];
pub fn q2_quad_weight(qi: usize) -> f64 {
    // GL3 point → (xi_idx, eta_idx):
    //   0:(-0.774,-0.774)→(0,0)  1:( 0.774,-0.774)→(2,0)  2:( 0.774, 0.774)→(2,2)
    //   3:(-0.774, 0.774)→(0,2)  4:( 0.0,  -0.774)→(1,0)  5:( 0.774, 0.0  )→(2,1)
    //   6:( 0.0,   0.774)→(1,2)  7:(-0.774, 0.0  )→(0,1)  8:( 0.0,   0.0  )→(1,1)
    let i: usize = match qi { 0|3|7 => 0, 4|6|8 => 1, 1|2|5 => 2, _ => 0 };
    let j: usize = match qi { 0|1|4 => 0, 5|7|8 => 1, 2|3|6 => 2, _ => 0 };
    GL3_W[i] * GL3_W[j]
}
pub fn q2_quad_point(qi: usize) -> (f64, f64) { (GL3[qi][0], GL3[qi][1]) }

// ─── 4×4 Gauss-Legendre on [-1,1]² (degree-7 accurate) ────────────────────

const GL4_X: [f64; 4] = [
    -0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526,
];
const GL4_W: [f64; 4] = [
    0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538,
];

pub fn q4_quad_point(qi: usize) -> (f64, f64) {
    (GL4_X[qi % 4], GL4_X[qi / 4])
}
pub fn q4_quad_weight(qi: usize) -> f64 {
    GL4_W[qi % 4] * GL4_W[qi / 4]
}

// ─── Diffusion (Laplace-Beltrami) ─────────────────────────────────────────

/// Surface diffusion for Q2 (Quad9): `∫_Γ ∇_Γ u · ∇_Γ v dS`
pub struct SurfaceQuad9DiffusionIntegrator;

impl SurfaceQuad9DiffusionIntegrator {
    pub fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 9], k_elem: &mut [f64; 81]) {
        for q in 0..9 {
            let (xi, eta) = q2_quad_point(q);
            let w = q2_quad_weight(q);
            let j = q2_jacobian_at(elem_nodes, xi, eta);
            let (det_g, sqrt_det_g) = q2_metric_at(&j);
            let pinv = pseudo_inverse_quad9(&j, det_g);
            let sg = q2_surface_gradients_at(&pinv, xi, eta);
            let area = w * sqrt_det_g;
            for i in 0..9 {
                for j in 0..9 {
                    let dot = sg[i][0]*sg[j][0] + sg[i][1]*sg[j][1] + sg[i][2]*sg[j][2];
                    k_elem[i * 9 + j] += dot * area;
                }
            }
        }
    }
}

impl SurfaceQuad9BilinearIntegrator for SurfaceQuad9DiffusionIntegrator {
    fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 9], k_elem: &mut [f64; 81]) {
        self.add_to_element_matrix(elem_nodes, k_elem);
    }
}

// ─── Mass ─────────────────────────────────────────────────────────────────

/// Surface mass for Q2 (Quad9): `∫_Γ u v dS`
pub struct SurfaceQuad9MassIntegrator;

impl SurfaceQuad9MassIntegrator {
    pub fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 9], k_elem: &mut [f64; 81]) {
        // MFEM MassIntegrator uses order = 2·GetOrder() + OrderW() = 7 on this
        // Q2 surface mesh → 4×4 Gauss-Legendre (16 pts), NOT the 3×3 rule used
        // by DiffusionIntegrator (order 5).  Using 3×3 here under-integrates
        // the mass block (0.4% error on ex7).
        for q in 0..16 {
            let (xi, eta) = q4_quad_point(q);
            let w = q4_quad_weight(q);
            let j = q2_jacobian_at(elem_nodes, xi, eta);
            let (_det_g, sqrt_det_g) = q2_metric_at(&j);
            let area = w * sqrt_det_g;
            let phi = q2_basis(xi, eta);
            for i in 0..9 {
                for j in 0..9 {
                    k_elem[i * 9 + j] += phi[i] * phi[j] * area;
                }
            }
        }
    }
}

impl SurfaceQuad9BilinearIntegrator for SurfaceQuad9MassIntegrator {
    fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 9], k_elem: &mut [f64; 81]) {
        self.add_to_element_matrix(elem_nodes, k_elem);
    }
}

// ─── Domain source ────────────────────────────────────────────────────────

/// Surface domain source for Q2 (Quad9): `∫_Γ f(x) v(x) dS`
pub struct SurfaceQuad9DomainSourceIntegrator<'a> {
    pub f: &'a dyn Fn(&[f64; 3]) -> f64,
}

impl SurfaceQuad9DomainSourceIntegrator<'_> {
    pub fn add_to_element_vector(&self, elem_nodes: &[[f64; 3]; 9], f_elem: &mut [f64; 9]) {
        for q in 0..9 {
            let (xi, eta) = q2_quad_point(q);
            let w = q2_quad_weight(q);
            let j = q2_jacobian_at(elem_nodes, xi, eta);
            let (_det_g, sqrt_det_g) = q2_metric_at(&j);
            let area = w * sqrt_det_g;
            let phi = q2_basis(xi, eta);
            let xp = [
                phi.iter().zip(elem_nodes.iter()).map(|(&p, &n)| p * n[0]).sum::<f64>(),
                phi.iter().zip(elem_nodes.iter()).map(|(&p, &n)| p * n[1]).sum::<f64>(),
                phi.iter().zip(elem_nodes.iter()).map(|(&p, &n)| p * n[2]).sum::<f64>(),
            ];
            let f_val = (self.f)(&xp);
            for i in 0..9 {
                f_elem[i] += f_val * phi[i] * area;
            }
        }
    }
}

impl SurfaceQuad9LinearIntegrator for SurfaceQuad9DomainSourceIntegrator<'_> {
    fn add_to_element_vector(&self, elem_nodes: &[[f64; 3]; 9], f_elem: &mut [f64; 9]) {
        self.add_to_element_vector(elem_nodes, f_elem);
    }
}

/// Build 9 Q2 nodal coordinates from 4 corner nodes of a Quad4 mesh.
/// Edge midpoints and centroid are projected onto the unit sphere
/// to match the snapping applied to DofManager Q2 DOF coordinates.
pub fn q2_coords_from_quad4<M: MeshTopology>(mesh: &M, nodes: &[u32]) -> [[f64; 3]; 9] {
    let x0 = get_coord3(mesh, nodes[0]);
    let x1 = get_coord3(mesh, nodes[1]);
    let x2 = get_coord3(mesh, nodes[2]);
    let x3 = get_coord3(mesh, nodes[3]);
    let snap = |mut p: [f64; 3]| { let r = (p[0]*p[0]+p[1]*p[1]+p[2]*p[2]).sqrt().max(1e-30);
                                   p[0]/=r; p[1]/=r; p[2]/=r; p };
    let midpoint = |a: [f64; 3], b: [f64; 3]| snap([(a[0]+b[0])/2.0, (a[1]+b[1])/2.0, (a[2]+b[2])/2.0]);
    let centre = snap([(x0[0]+x1[0]+x2[0]+x3[0])/4.0,
                       (x0[1]+x1[1]+x2[1]+x3[1])/4.0,
                       (x0[2]+x1[2]+x2[2]+x3[2])/4.0]);
    [x0, x1, x2, x3, midpoint(x0,x1), midpoint(x1,x2), midpoint(x2,x3), midpoint(x3,x0), centre]
}

// ─── Quad9 Surface Assembler ──────────────────────────────────────────────

/// Assemble surface forms for Q2 (9 DOF/element) on a Quad4 mesh with H1 order=2.
///
/// Reads 4 vertex nodes from `mesh.element_nodes(e)`, builds the 9 Q2
/// coordinates, and uses 9 DOFs from `space.element_dofs(e)` for global
/// indexing.  This matches DofManager's `build_q2_quad` numbering.
pub struct SurfaceQuad9Assembler;

impl SurfaceQuad9Assembler {
    /// Assemble a bilinear form over all elements.
    pub fn assemble_bilinear<S: FESpace>(
        space: &S,
        integrators: &[&dyn SurfaceQuad9BilinearIntegrator],
    ) -> CsrMatrix<f64> {
        let mesh = space.mesh();
        let n_dofs = space.n_dofs();
        let ne = mesh.n_elements() as u32;
        let mut coo = CooMatrix::new(n_dofs, n_dofs);
        for e in 0..ne {
            let dofs = space.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            if nodes.len() < 4 { continue; }
            let x = q2_coords_from_quad4(mesh, nodes);
            let mut ke = [0.0; 81];
            for integ in integrators {
                integ.add_to_element_matrix(&x, &mut ke);
            }
            for i in 0..9 {
                for j in 0..9 {
                    coo.add(dofs[i] as usize, dofs[j] as usize, ke[i * 9 + j]);
                }
            }
        }
        coo.into_csr()
    }

    /// Assemble a linear form over all elements.
    pub fn assemble_linear<S: FESpace>(
        space: &S,
        integrators: &[&dyn SurfaceQuad9LinearIntegrator],
    ) -> Vec<f64> {
        let mesh = space.mesh();
        let n_dofs = space.n_dofs();
        let ne = mesh.n_elements() as u32;
        let mut rhs = vec![0.0; n_dofs];
        for e in 0..ne {
            let dofs = space.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            if nodes.len() < 4 { continue; }
            let x = q2_coords_from_quad4(mesh, nodes);
            let mut fe = [0.0; 9];
            for integ in integrators {
                integ.add_to_element_vector(&x, &mut fe);
            }
            for i in 0..9 {
                rhs[dofs[i] as usize] += fe[i];
            }
        }
        rhs
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::{ElementType, Mesh};
    use fem_space::H1Space;
    use fem_solver::{solve_cg, SolverConfig};

    /// Unit sphere mesh: octahedron refined n times.
    fn sphere_mesh(n: u32) -> Mesh<3> {
        // Start with an octahedron (6 vertices, 8 faces)
        let t = 2.0_f64.sqrt() / 2.0; // = 1/�?
        let mut coords = vec![
            0.0, 0.0, -1.0,    // 0: south
            -t, 0.0, t,         // 1: front-left
            0.0, t, t,          // 2: front-top
            t, 0.0, t,          // 3: front-right
            0.0, -t, t,         // 4: front-bottom
            0.0, 0.0, 1.0,      // 5: north
        ];
        let mut conn = vec![
            0u32, 1, 2,  0, 2, 3,  0, 3, 4,  0, 4, 1,
            5, 2, 1,  5, 3, 2,  5, 4, 3,  5, 1, 4,
        ];

        // Uniform subdivision: split each triangle into 4
        for _ in 0..n {
            let old_n = conn.len() / 3;
            let mut new_conn = Vec::with_capacity(old_n * 12);
            let mut edge_map: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
            let mut next_node = coords.len() as u32 / 3;

            for t in 0..old_n {
                let i = t * 3;
                let (a, b, c) = (conn[i], conn[i+1], conn[i+2]);
                let edge = |x: u32, y: u32| if x < y { (x, y) } else { (y, x) };

                let ab = *edge_map.entry(edge(a, b)).or_insert_with(|| {
                    let j = next_node; next_node += 1;
                    let (xa, ya, za) = (coords[a as usize*3], coords[a as usize*3+1], coords[a as usize*3+2]);
                    let (xb, yb, zb) = (coords[b as usize*3], coords[b as usize*3+1], coords[b as usize*3+2]);
                    let len = ((xa-xb).powi(2)+(ya-yb).powi(2)+(za-zb).powi(2)).sqrt();
                    let cx = (xa+xb)/2.0; let cy = (ya+yb)/2.0; let cz = (za+zb)/2.0;
                    let r = 1.0 / (cx*cx+cy*cy+cz*cz).sqrt();
                    coords.extend_from_slice(&[cx*r, cy*r, cz*r]);
                    j
                });
                let ac = *edge_map.entry(edge(a, c)).or_insert_with(|| {
                    let j = next_node; next_node += 1;
                    let (xa, ya, za) = (coords[a as usize*3], coords[a as usize*3+1], coords[a as usize*3+2]);
                    let (xc, yc, zc) = (coords[c as usize*3], coords[c as usize*3+1], coords[c as usize*3+2]);
                    let cx = (xa+xc)/2.0; let cy = (ya+yc)/2.0; let cz = (za+zc)/2.0;
                    let r = 1.0 / (cx*cx+cy*cy+cz*cz).sqrt();
                    coords.extend_from_slice(&[cx*r, cy*r, cz*r]);
                    j
                });
                let bc = *edge_map.entry(edge(b, c)).or_insert_with(|| {
                    let j = next_node; next_node += 1;
                    let (xb, yb, zb) = (coords[b as usize*3], coords[b as usize*3+1], coords[b as usize*3+2]);
                    let (xc, yc, zc) = (coords[c as usize*3], coords[c as usize*3+1], coords[c as usize*3+2]);
                    let cx = (xb+xc)/2.0; let cy = (yb+yc)/2.0; let cz = (zb+zc)/2.0;
                    let r = 1.0 / (cx*cx+cy*cy+cz*cz).sqrt();
                    coords.extend_from_slice(&[cx*r, cy*r, cz*r]);
                    j
                });

                new_conn.extend_from_slice(&[a, ab, ac, b, bc, ab, c, ac, bc, ab, bc, ac]);
            }
            conn = new_conn;
        }

        let n_elem = conn.len() / 3;
        Mesh {
            coords, conn,
            elem_tags: vec![0; n_elem],
            elem_type: fem_mesh::element_type::ElementType::Tri3,
            face_conn: vec![], face_tags: vec![],
            face_type: fem_mesh::element_type::ElementType::Line2,
            elem_types: None, elem_offsets: None,
            face_types: None, face_offsets: None,
            face_to_elem: None,
            edge_conn: vec![], edge_to_elem: vec![], nc_vertex_view: None,
            geometry: None,
        }
    }

    #[test]
    fn sphere_mesh_basics() {
        let m = sphere_mesh(1); // ~32 elements
        assert!(m.n_elems() > 0);
        assert!(m.n_nodes() > 0);
        // Check that all nodes are on the unit sphere
        for n in 0..m.n_nodes() as u32 {
            let c = m.node_coords(n);
            let r2 = c[0]*c[0] + c[1]*c[1] + c[2]*c[2];
            assert!((r2 - 1.0).abs() < 1e-12, "node {n} not on sphere: r²={r2}");
        }
    }

    /// Solve Laplace-Beltrami on sphere: -Δ_Γ u = 0 with u = x at some boundary
    /// The exact solution is u = x (linear function restricted to sphere).
    /// For a closed surface (no boundary), the solution is non-unique (kernel = constants).
    /// Instead, we solve -Δ_Γ u + u = f and verify convergence.
    #[test]
    fn surface_laplace_beltrami_mass() {
        let mesh = sphere_mesh(2); // ~128 elements
        let space = H1Space::new(mesh, 1);

        // For u(x,y,z) = x on the unit sphere: Δ_Γ x = -2x
        // So -Δ_Γ x + x = 3x
        let f = &|x: &[f64; 3]| 3.0 * x[0];

        // Assemble: A = Diffusion (stiffness) + Mass
        let a = SurfaceAssembler::assemble_bilinear(&space, &[
            &SurfaceDiffusionIntegrator as &dyn SurfaceBilinearIntegrator,
            &SurfaceMassIntegrator,
        ]);

        let source = SurfaceDomainSourceIntegrator { f };
        let rhs = SurfaceAssembler::assemble_linear(&space, &[
            &source as &dyn SurfaceLinearIntegrator,
        ]);

        let mut u = vec![0.0; space.n_dofs()];
        let cfg = SolverConfig { rtol: 1e-10, max_iter: 5000, ..SolverConfig::default() };
        let res = solve_cg(&a, &rhs, &mut u, &cfg).expect("CG solve failed");
        eprintln!("Laplace-Beltrami+Mass: {}/{} iters, residual {:.3e}",
            res.iterations, cfg.max_iter, res.final_residual);
        assert!(res.converged, "CG did not converge");

        // Check solution against exact u_exact = x at mesh nodes.
        // For P1 H1, DOF i corresponds to node i.
        let m = space.mesh();
        let mut err2 = 0.0;
        for i in 0..space.n_dofs().min(m.n_nodes()) {
            let x = m.node_coords(i as u32);
            let expected = x[0]; // u = x
            let diff = u[i] - expected;
            err2 += diff * diff;
        }
        let err = err2.sqrt() / (space.n_dofs() as f64).sqrt();
        eprintln!("Normalized DOF error: {:.6e}", err);
        assert!(err < 0.1, "Surface PDE error too large: {err:.6e}");
    }

    /// Convergence test: refine sphere mesh and verify error decreases.
    #[test]
    fn surface_quad4_laplace_beltrami_mass() {
        // Cube inscribed in unit sphere: 8 vertices, 6 quad faces.
        let coords = vec![
            -0.57735, -0.57735, -0.57735,
             0.57735, -0.57735, -0.57735,
             0.57735,  0.57735, -0.57735,
            -0.57735,  0.57735, -0.57735,
            -0.57735, -0.57735,  0.57735,
             0.57735, -0.57735,  0.57735,
             0.57735,  0.57735,  0.57735,
            -0.57735,  0.57735,  0.57735,
        ];
        let conn = vec![
            3, 2, 1, 0,  0, 1, 5, 4,  1, 2, 6, 5,
            2, 3, 7, 6,  3, 0, 4, 7,  4, 5, 6, 7,
        ];
        let mesh: Mesh<3> = Mesh {
            coords, conn,
            elem_tags: vec![1; 6],
            elem_type: ElementType::Quad4,
            face_conn: vec![], face_tags: vec![],
            face_type: ElementType::Line2,
            elem_types: None, elem_offsets: None,
            face_types: None, face_offsets: None,
            face_to_elem: None,
            edge_conn: vec![], edge_to_elem: vec![], nc_vertex_view: None,
            geometry: None,
        };
        let space = H1Space::new(mesh, 1);

        // Solve -Δ_Γ u + u = 3x on the cube (u = x is exact for cube)
        let f = &|x: &[f64; 3]| 3.0 * x[0];

        let a = SurfaceQuad4Assembler::assemble_bilinear(&space, &[
            &SurfaceQuad4DiffusionIntegrator as &dyn SurfaceQuad4BilinearIntegrator,
            &SurfaceQuad4MassIntegrator,
        ]);

        let source = SurfaceQuad4DomainSourceIntegrator { f };
        let rhs = SurfaceQuad4Assembler::assemble_linear(&space, &[
            &source as &dyn SurfaceQuad4LinearIntegrator,
        ]);

        let mut u = vec![0.0; space.n_dofs()];
        let cfg = SolverConfig { rtol: 1e-10, max_iter: 5000, ..SolverConfig::default() };
        let res = solve_cg(&a, &rhs, &mut u, &cfg).expect("CG solve failed");
        assert!(res.converged, "Quad4 CG did not converge");

        let m = space.mesh();
        let mut err2 = 0.0;
        for i in 0..space.n_dofs().min(m.n_nodes()) {
            let x = m.node_coords(i as u32);
            let diff = u[i] - x[0];
            err2 += diff * diff;
        }
        let err = err2.sqrt() / (space.n_dofs() as f64).sqrt();
        eprintln!("Quad4 surface error: {:.6e}", err);
        assert!(err < 0.5, "Quad4 surface error too large: {err:.6e}");
    }

    #[test]
    fn surface_convergence() {
        let mut prev_err: Option<f64> = None;
        let mut prev_h: Option<f64> = None;

        for level in 1..=3 {
            let mesh = sphere_mesh(level);
            let space = H1Space::new(mesh, 1);
            let n = space.n_dofs();

            let a = SurfaceAssembler::assemble_bilinear(&space, &[
                &SurfaceDiffusionIntegrator as &dyn SurfaceBilinearIntegrator,
                &SurfaceMassIntegrator,
            ]);

            let f = &|x: &[f64; 3]| 3.0 * x[0];
            let source = SurfaceDomainSourceIntegrator { f };
            let rhs = SurfaceAssembler::assemble_linear(&space, &[
                &source as &dyn SurfaceLinearIntegrator,
            ]);

            let mut u = vec![0.0; n];
            let cfg = SolverConfig { rtol: 1e-10, max_iter: 5000, ..SolverConfig::default() };
            let res = solve_cg(&a, &rhs, &mut u, &cfg).expect("CG solve failed");
            assert!(res.converged, "CG did not converge at level {level}");

            let m = space.mesh();
            let mut err2 = 0.0;
            for i in 0..n.min(m.n_nodes()) {
                let x = m.node_coords(i as u32);
                let diff = u[i] - x[0];
                err2 += diff * diff;
            }
            let err = err2.sqrt() / (n as f64).sqrt();
            let h = 1.0 / (8 << level) as f64; // approx element size

            eprintln!("Surface convergence level={level} n={n} h={h:.4e} err={err:.6e} iters={}",
                res.iterations);

            if let (Some(pe), Some(ph)) = (prev_err, prev_h) {
                let rate = (pe / err.max(1e-30)).ln() / (ph / h.max(1e-30)).ln();
                eprintln!("  └─ observed order �?{rate:.2}");
                assert!(rate > 0.5, "expected convergence, got rate={rate:.2}");
            }

            prev_err = Some(err);
            prev_h = Some(h);
        }
    }
}
