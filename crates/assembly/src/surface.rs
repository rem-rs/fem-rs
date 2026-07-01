//! Surface finite element method for solving PDEs on 2-D manifolds in 3-D.
//!
//! Provides integrators that work with `SimplexMesh<3>` containing Tri3 elements
//! (2-D topology embedded in 3-D space).  Key differences from planar FEM:
//!
//! - Jacobian is `3×2` instead of `2×2`
//! - Gradient transform uses pseudo-inverse `(JᵀJ)⁻¹Jᵀ` instead of `J⁻ᵀ`
//! - Area element is `√det(JᵀJ)` instead of `|det(J)|`
//! - Surface normal `n = (J₀ × J₁) / |J₀ × J₁|`
//!
//! # Example
//! ```ignore
//! use fem_mesh::SimplexMesh;
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
fn surface_jacobian(x: &[[f64; 3]; 3]) -> ([[f64; 3]; 2], [[f64; 2]; 2], f64, [f64; 3]) {
    // J = [x1-x0, x2-x0]  — 3×2 matrix stored as 2 columns of 3 components
    let j0 = [x[1][0] - x[0][0], x[1][1] - x[0][1], x[1][2] - x[0][2]];
    let j1 = [x[2][0] - x[0][0], x[2][1] - x[0][1], x[2][2] - x[0][2]];
    let j = [j0, j1];

    // G = JᵀJ  — 2×2 metric
    let g00 = j0[0]*j0[0] + j0[1]*j0[1] + j0[2]*j0[2];
    let g01 = j0[0]*j1[0] + j0[1]*j1[1] + j0[2]*j1[2];
    let g10 = g01;
    let g11 = j1[0]*j1[0] + j1[1]*j1[1] + j1[2]*j1[2];
    let g = [[g00, g01], [g10, g11]];

    // det(G) and sqrt(det(G))
    let det_g = g00 * g11 - g01 * g01;
    let sqrt_det_g = det_g.sqrt().max(1e-30);

    // Unit normal n = (J₀ × J₁) / |J₀ × J₁|
    let nx = j0[1]*j1[2] - j0[2]*j1[1];
    let ny = j0[2]*j1[0] - j0[0]*j1[2];
    let nz = j0[0]*j1[1] - j0[1]*j1[0];
    let n_len = (nx*nx + ny*ny + nz*nz).sqrt().max(1e-30);
    let normal = [nx/n_len, ny/n_len, nz/n_len];

    (j, g, sqrt_det_g, normal)
}

/// Compute the pseudo-inverse J_pinv = (JᵀJ)⁻¹Jᵀ  — 2×3 matrix.
///
/// Used for surface gradient: `∇_Γ u = J_pinvᵀ · ∇_ξ u = G⁻¹ · Jᵀ · ∇_ξ u`
fn pseudo_inverse(j: &[[f64; 3]; 2], g: &[[f64; 2]; 2], det_g: f64) -> [[f64; 2]; 3] {
    let inv_det = 1.0 / det_g.max(1e-30);
    let g_inv_00 = g[1][1] * inv_det;
    let g_inv_01 = -g[0][1] * inv_det;
    let g_inv_10 = g_inv_01;
    let g_inv_11 = g[0][0] * inv_det;

    // J_pinv = G⁻¹ · Jᵀ  (2×3)
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
                k_elem[i * 3 + j] = dot * area_factor;
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

        // Mass matrix for P1: M[i,j] = ∫ φ_i·φ_j dS
        // One-point quadrature at centroid (ξ=1/3, η=1/3): φ_i = 1/3 for all i
        let phi_qp = 1.0 / 3.0;
        let val = phi_qp * phi_qp * area_factor;

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
                k_elem[i * 3 + j] = val * sqrt_det_g;
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
            f_elem[i] = f_val * phi_qp * area_factor;
        }
    }
}

// ─── Surface Assembler ───────────────────────────────────────────────────────

/// Assemble a surface bilinear form using a surface integrator.
///
/// Works with `H1Space<SimplexMesh<3>>` containing Tri3 surface elements.
pub struct SurfaceAssembler;

impl SurfaceAssembler {
    pub fn assemble_bilinear<S: FESpace>(
        space: &S,
        integrator: &dyn Fn(&[[f64; 3]; 3], &mut [f64; 9]),
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
            integrator(&x, &mut ke);
            for i in 0..3 {
                for j in 0..3 {
                    coo.add(dofs[i] as usize, dofs[j] as usize, ke[i * 3 + j]);
                }
            }
        }

        coo.into_csr()
    }

    pub fn assemble_linear<S: FESpace>(
        space: &S,
        integrator: &dyn Fn(&[[f64; 3]; 3], &mut [f64; 3]),
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
            integrator(&x, &mut fe);
            for i in 0..3 {
                rhs[dofs[i] as usize] += fe[i];
            }
        }

        rhs
    }
}

fn get_coord3<M: MeshTopology>(mesh: &M, n: u32) -> [f64; 3] {
    let c = mesh.node_coords(n);
    [c[0], c[1], if c.len() > 2 { c[2] } else { 0.0 }]
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;
    use fem_solver::{solve_cg, SolverConfig};

    /// Unit sphere mesh: octahedron refined n times.
    fn sphere_mesh(n: u32) -> SimplexMesh<3> {
        // Start with an octahedron (6 vertices, 8 faces)
        let t = (2.0_f64.sqrt() / 2.0); // = 1/√2
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
        SimplexMesh {
            coords, conn,
            elem_tags: vec![0; n_elem],
            elem_type: fem_mesh::element_type::ElementType::Tri3,
            face_conn: vec![], face_tags: vec![],
            face_type: fem_mesh::element_type::ElementType::Line2,
            elem_types: None, elem_offsets: None,
            face_types: None, face_offsets: None,
            face_to_elem: None,
            edge_conn: vec![], edge_to_elem: vec![],
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
        let stiffness = SurfaceAssembler::assemble_bilinear(&space, &|x, ke| {
            SurfaceDiffusionIntegrator.add_to_element_matrix(x, ke);
        });
        let mass = SurfaceAssembler::assemble_bilinear(&space, &|x, ke| {
            SurfaceMassIntegrator.add_to_element_matrix(x, ke);
        });
        let mut a = stiffness.clone();
        for i in 0..a.nrows {
            for jp in a.row_ptr[i]..a.row_ptr[i+1] {
                let j = a.col_idx[jp] as usize;
                a.values[jp] += mass.get(i, j);
            }
        }

        let rhs = SurfaceAssembler::assemble_linear(&space, &|x, fe| {
            SurfaceDomainSourceIntegrator { f }.add_to_element_vector(x, fe);
        });

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
    fn surface_convergence() {
        let mut prev_err: Option<f64> = None;
        let mut prev_h: Option<f64> = None;

        for level in 1..=3 {
            let mesh = sphere_mesh(level);
            let space = H1Space::new(mesh, 1);
            let n = space.n_dofs();

            let stiffness = SurfaceAssembler::assemble_bilinear(&space, &|x, ke| {
                SurfaceDiffusionIntegrator.add_to_element_matrix(x, ke);
            });
            let mass = SurfaceAssembler::assemble_bilinear(&space, &|x, ke| {
                SurfaceMassIntegrator.add_to_element_matrix(x, ke);
            });
            let mut a = stiffness.clone();
            for i in 0..a.nrows {
                for jp in a.row_ptr[i]..a.row_ptr[i+1] {
                    let j = a.col_idx[jp] as usize;
                    a.values[jp] += mass.get(i, j);
                }
            }

            let f = &|x: &[f64; 3]| 3.0 * x[0];
            let rhs = SurfaceAssembler::assemble_linear(&space, &|x, fe| {
                SurfaceDomainSourceIntegrator { f }.add_to_element_vector(x, fe);
            });

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
                eprintln!("  └─ observed order ≈ {rate:.2}");
                assert!(rate > 0.5, "expected convergence, got rate={rate:.2}");
            }

            prev_err = Some(err);
            prev_h = Some(h);
        }
    }
}
