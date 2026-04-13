//! # Surface FEM — Laplace-Beltrami baseline (toward MFEM ex7/ex29)
//!
//! Solves a reaction-diffusion PDE on the **unit sphere** surface:
//!
//! ```text
//!   −Δ_S u + u = f   on S² = { x ∈ R³ : |x| = 1 }
//! ```
//!
//! using P1 Lagrange finite elements on a triangulated sphere mesh
//! (icosahedron-based geodesic subdivision).
//!
//! ## Laplace-Beltrami operator
//!
//! For each triangle K embedded in R³ with nodes P₀, P₁, P₂, the
//! element Jacobian is
//!
//! ```text
//!   J = [P₁−P₀ | P₂−P₀]   (3×2 real matrix)
//! ```
//!
//! The surface metric tensor `G = Jᵀ J` (2×2 SPD) gives:
//! - Surface area element: `dS = √|G| dξ`
//! - Surface gradient:     `∇_S φᵢ = J G⁻¹ ∇_ref φᵢ`  (3-D vector)
//!
//! ## Manufactured solution
//!
//! Exact solution: `u(x,y,z) = z` (linear, a degree-1 spherical harmonic).
//! Using `Δ_S z = −2z` on the unit sphere:
//! ```text
//!   f = −Δ_S z + z = 2z + z = 3z
//! ```
//!
//! ## Usage
//! ```
//! cargo run --example mfem_surface_fem
//! cargo run --example mfem_surface_fem -- --subdiv 4
//! ```

use std::collections::HashMap;
use fem_element::lagrange::tri::TriP1;
use fem_element::reference::ReferenceElement;
use fem_linalg::CooMatrix;
use fem_solver::solve_sparse_cholesky;

fn main() {
    let args = parse_args();

    println!("=== fem-rs Surface FEM: Laplace-Beltrami on unit sphere ===");
    for &subdiv in &[1usize, 2, 3, args.subdiv] {
        let res = solve_surface_fem(subdiv);
        println!(
            "  subdiv={:2}  nodes={:5}  triangles={:5}  L2 err={:.3e}",
            subdiv, res.n_nodes, res.n_elements, res.l2_error
        );
    }
    println!();
    println!("Note: P1 Lagrange on a triangulated sphere; closed surface (no BCs).");
    println!("      Full surface FEM for open surfaces / MFEM ex29-style is still pending.");
}

// ─── CLI args ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Args {
    subdiv: usize,
}

fn parse_args() -> Args {
    let mut subdiv = 3usize;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if a == "--subdiv" {
            subdiv = it.next().unwrap_or_default().parse().unwrap_or(3);
        }
    }
    Args { subdiv }
}

// ─── result ───────────────────────────────────────────────────────────────────

struct SurfaceResult {
    n_nodes: usize,
    n_elements: usize,
    l2_error: f64,
}

// ─── manufactured solution ────────────────────────────────────────────────────

/// Exact solution on the unit sphere: u(x,y,z) = z.
fn exact_u(p: &[f64; 3]) -> f64 {
    p[2]
}

/// Source term: f = -Δ_S u + u = 2z + z = 3z.
fn source_f(p: &[f64; 3]) -> f64 {
    3.0 * p[2]
}

// ─── solver ───────────────────────────────────────────────────────────────────

fn solve_surface_fem(subdiv: usize) -> SurfaceResult {
    // Build the geodesic sphere mesh
    let (vertices, triangles) = build_sphere(subdiv);
    let n_nodes = vertices.len();
    let n_elements = triangles.len();

    // ── assemble global (K + M) and RHS ───────────────────────────────────────
    let mut coo = CooMatrix::<f64>::new(n_nodes, n_nodes);
    let mut rhs = vec![0.0_f64; n_nodes];

    // pre-fill diagonal so Cholesky has full structural support
    for i in 0..n_nodes {
        coo.add(i, i, 0.0);
    }

    let quad = TriP1.quadrature(4); // order 4 for accurate integration

    for tri in &triangles {
        let p0 = &vertices[tri[0]];
        let p1 = &vertices[tri[1]];
        let p2 = &vertices[tri[2]];
        assemble_surface_element(p0, p1, p2, tri, &quad, &mut coo, &mut rhs);
    }

    // ── solve (K + M) u = f  (SPD, no Dirichlet needed: α=1 mass term) ────────
    let mat = coo.into_csr();
    let sol = solve_sparse_cholesky(&mat, &rhs).expect("surface Cholesky solve failed");

    // ── L2 error on surface ───────────────────────────────────────────────────
    let l2_error = compute_surface_l2_error(&vertices, &triangles, &sol, &quad);

    SurfaceResult { n_nodes, n_elements, l2_error }
}

// ─── element assembly ─────────────────────────────────────────────────────────

/// Assemble local (stiffness + mass) and load contributions for one surface triangle.
///
/// `global_dofs = tri` (node indices, same as DOF indices for P1 nodal elements).
fn assemble_surface_element(
    p0: &[f64; 3],
    p1: &[f64; 3],
    p2: &[f64; 3],
    global_dofs: &[usize; 3],
    quad: &fem_element::reference::QuadratureRule,
    coo: &mut CooMatrix<f64>,
    rhs: &mut [f64],
) {
    // ── Surface Jacobian J (3×2) ──────────────────────────────────────────────
    // J[:,0] = P1 - P0,  J[:,1] = P2 - P0
    let j0 = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]];  // tangent along ξ₁
    let j1 = [p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]];  // tangent along ξ₂

    // ── Metric tensor G = Jᵀ J (2×2) ─────────────────────────────────────────
    let g00 = dot3(&j0, &j0);
    let g01 = dot3(&j0, &j1);
    let g10 = g01;
    let g11 = dot3(&j1, &j1);
    let det_g = g00 * g11 - g01 * g10;
    let sqrt_det_g = det_g.sqrt();

    // G⁻¹ = (1/det_g) [ g11, -g01; -g10, g00 ]
    let inv_det_g = 1.0 / det_g;
    let ginv00 =  g11 * inv_det_g;
    let ginv01 = -g01 * inv_det_g;
    let ginv10 = -g10 * inv_det_g;
    let ginv11 =  g00 * inv_det_g;

    // P1 reference gradients (constant on element):
    // ∇_ref φ_0 = (-1,-1),  ∇_ref φ_1 = (1,0),  ∇_ref φ_2 = (0,1)
    let ref_grads: [[f64; 2]; 3] = [[-1.0,-1.0], [1.0,0.0], [0.0,1.0]];

    // Surface gradients (3-D): ∇_S φ_i = J G⁻¹ ∇_ref φ_i
    // = J @ [ ginv00 * ∂φ/∂ξ₁ + ginv01 * ∂φ/∂ξ₂ ;
    //         ginv10 * ∂φ/∂ξ₁ + ginv11 * ∂φ/∂ξ₂ ]
    let mut surf_grads = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        let rg = ref_grads[i];
        let c0 = ginv00 * rg[0] + ginv01 * rg[1]; // coefficient for J[:,0]
        let c1 = ginv10 * rg[0] + ginv11 * rg[1]; // coefficient for J[:,1]
        for d in 0..3 {
            surf_grads[i][d] = c0 * j0[d] + c1 * j1[d];
        }
    }

    // ── Integrate using quadrature rule ───────────────────────────────────────
    let mut p1_vals = [0.0_f64; 3];
    for (pt, &w) in quad.points.iter().zip(quad.weights.iter()) {
        let xi = &[pt[0], pt[1]];
        TriP1.eval_basis(xi, &mut p1_vals);

        let wds = w * sqrt_det_g; // weight × dS factor

        // Physical point on surface (for source evaluation)
        let phys = [
            p0[0] + j0[0]*xi[0] + j1[0]*xi[1],
            p0[1] + j0[1]*xi[0] + j1[1]*xi[1],
            p0[2] + j0[2]*xi[0] + j1[2]*xi[1],
        ];
        let fval = source_f(&phys);

        for i in 0..3 {
            let gi = global_dofs[i];
            // RHS: f_i += w √|G| f(x) φ_i
            rhs[gi] += wds * fval * p1_vals[i];

            for j in 0..3 {
                let gj = global_dofs[j];
                // Stiffness: K_ij += w √|G| ∇_S φ_i · ∇_S φ_j
                let kij = wds * dot3(&surf_grads[i], &surf_grads[j]);
                // Mass: M_ij += w √|G| φ_i φ_j
                let mij = wds * p1_vals[i] * p1_vals[j];
                coo.add(gi, gj, kij + mij);
            }
        }
    }
}

// ─── L2 error ─────────────────────────────────────────────────────────────────

fn compute_surface_l2_error(
    vertices: &[[f64; 3]],
    triangles: &[[usize; 3]],
    sol: &[f64],
    quad: &fem_element::reference::QuadratureRule,
) -> f64 {
    let mut err_sq = 0.0_f64;
    let mut p1_vals = [0.0_f64; 3];

    for tri in triangles {
        let p0 = &vertices[tri[0]];
        let p1 = &vertices[tri[1]];
        let p2 = &vertices[tri[2]];

        let j0 = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]];
        let j1 = [p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]];
        let det_g = (dot3(&j0,&j0)*dot3(&j1,&j1) - dot3(&j0,&j1).powi(2)).sqrt();

        for (pt, &w) in quad.points.iter().zip(quad.weights.iter()) {
            let xi = &[pt[0], pt[1]];
            TriP1.eval_basis(xi, &mut p1_vals);

            let phys = [
                p0[0] + j0[0]*xi[0] + j1[0]*xi[1],
                p0[1] + j0[1]*xi[0] + j1[1]*xi[1],
                p0[2] + j0[2]*xi[0] + j1[2]*xi[1],
            ];
            // Interpolate FEM solution
            let uh = p1_vals[0]*sol[tri[0]] + p1_vals[1]*sol[tri[1]] + p1_vals[2]*sol[tri[2]];
            let diff = uh - exact_u(&phys);
            err_sq += w * det_g * diff * diff;
        }
    }
    err_sq.sqrt()
}

// ─── sphere mesh generator ────────────────────────────────────────────────────

/// Build a geodesic sphere mesh via icosahedron subdivision.
///
/// Returns `(vertices, triangles)` where all vertices lie on the unit sphere.
/// `subdiv` = number of subdivision steps (0 = base icosahedron with 20 faces).
fn build_sphere(subdiv: usize) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let (mut verts, mut tris) = icosahedron();
    // Normalize initial vertices (they should already be close to unit sphere,
    // but ensure exactness)
    for v in &mut verts {
        normalize3(v);
    }
    for _ in 0..subdiv {
        (verts, tris) = subdivide_sphere(verts, tris);
    }
    (verts, tris)
}

/// Base icosahedron: 12 vertices on the unit sphere, 20 triangular faces.
fn icosahedron() -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let t = (1.0 + 5.0_f64.sqrt()) / 2.0;  // golden ratio
    let s = 1.0 / (1.0 + t * t).sqrt();    // normalization factor
    let ts = t * s;

    let verts = vec![
        [-s,  ts,  0.0], [ s,  ts,  0.0], [-s, -ts,  0.0], [ s, -ts,  0.0],
        [ 0.0, -s,  ts], [ 0.0,  s,  ts], [ 0.0, -s, -ts], [ 0.0,  s, -ts],
        [ ts,  0.0, -s], [ ts,  0.0,  s], [-ts,  0.0, -s], [-ts,  0.0,  s],
    ];

    let tris = vec![
        [0, 11,  5], [0,  5,  1], [0,  1,  7], [0,  7, 10], [0, 10, 11],
        [1,  5,  9], [5, 11,  4], [11, 10,  2], [10,  7,  6], [7,  1,  8],
        [3,  9,  4], [3,  4,  2], [3,  2,  6], [3,  6,  8], [3,  8,  9],
        [4,  9,  5], [2,  4, 11], [6,  2, 10], [8,  6,  7], [9,  8,  1],
    ];

    (verts, tris)
}

/// One loop of mesh subdivision: each triangle → 4 child triangles.
///
/// New midpoint vertices are projected onto the unit sphere.
fn subdivide_sphere(verts: Vec<[f64; 3]>, tris: Vec<[usize; 3]>) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let mut new_verts = verts;
    let mut cache: HashMap<(usize, usize), usize> = HashMap::new();
    let mut new_tris: Vec<[usize; 3]> = Vec::with_capacity(tris.len() * 4);

    for tri in &tris {
        let [a, b, c] = *tri;
        let ab = midpoint_vertex(&mut new_verts, &mut cache, a, b);
        let bc = midpoint_vertex(&mut new_verts, &mut cache, b, c);
        let ca = midpoint_vertex(&mut new_verts, &mut cache, c, a);
        new_tris.push([a, ab, ca]);
        new_tris.push([b, bc, ab]);
        new_tris.push([c, ca, bc]);
        new_tris.push([ab, bc, ca]);
    }

    (new_verts, new_tris)
}

/// Get or create the midpoint vertex between nodes `a` and `b` projected onto unit sphere.
fn midpoint_vertex(
    verts: &mut Vec<[f64; 3]>,
    cache: &mut HashMap<(usize, usize), usize>,
    a: usize,
    b: usize,
) -> usize {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&idx) = cache.get(&key) {
        return idx;
    }
    let va = verts[a];
    let vb = verts[b];
    let mut mid = [
        (va[0] + vb[0]) * 0.5,
        (va[1] + vb[1]) * 0.5,
        (va[2] + vb[2]) * 0.5,
    ];
    normalize3(&mut mid);
    let idx = verts.len();
    verts.push(mid);
    cache.insert(key, idx);
    idx
}

// ─── helpers ──────────────────────────────────────────────────────────────────

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

fn normalize3(v: &mut [f64; 3]) {
    let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    v[0] /= len;
    v[1] /= len;
    v[2] /= len;
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn sphere_mesh_area_converges_to_4pi() {
        // Surface area of the unit sphere = 4π ≈ 12.566
        let area_exact = 4.0 * PI;
        for subdiv in 0..4 {
            let (verts, tris) = build_sphere(subdiv);
            let area = compute_mesh_area(&verts, &tris);
            let rel_err = ((area - area_exact) / area_exact).abs();
            assert!(
                rel_err < 0.02 + 0.3 / (subdiv as f64 + 1.0),
                "subdiv={}: mesh area={:.4} (exact={:.4}), rel_err={:.3e}",
                subdiv, area, area_exact, rel_err
            );
        }
    }

    #[test]
    fn surface_fem_l2_error_decreases_with_refinement() {
        let r1 = solve_surface_fem(1);
        let r2 = solve_surface_fem(2);
        let r3 = solve_surface_fem(3);

        // All errors should be finite
        assert!(r1.l2_error.is_finite());
        assert!(r2.l2_error.is_finite());

        // Error should decrease with refinement
        assert!(
            r2.l2_error < r1.l2_error,
            "L2 error should decrease: subdiv=1 {:.3e} → subdiv=2 {:.3e}",
            r1.l2_error, r2.l2_error
        );

        assert!(
            r3.l2_error < r2.l2_error,
            "L2 error should decrease: subdiv=2 {:.3e} → subdiv=3 {:.3e}",
            r2.l2_error, r3.l2_error
        );
    }

    #[test]
    fn surface_fem_l2_error_is_small() {
        // subdiv=3 gives 1280 triangles; baseline implementation is expected <1e-2
        let res = solve_surface_fem(3);
        assert!(
            res.l2_error < 1.0e-2,
            "L2 error too large: {} (threshold 1e-2)",
            res.l2_error
        );
    }

    fn compute_mesh_area(verts: &[[f64; 3]], tris: &[[usize; 3]]) -> f64 {
        let mut area = 0.0_f64;
        for tri in tris {
            let p0 = &verts[tri[0]];
            let p1 = &verts[tri[1]];
            let p2 = &verts[tri[2]];
            let j0 = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]];
            let j1 = [p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]];
            // cross product magnitude / 2
            let cx = j0[1]*j1[2] - j0[2]*j1[1];
            let cy = j0[2]*j1[0] - j0[0]*j1[2];
            let cz = j0[0]*j1[1] - j0[1]*j1[0];
            area += 0.5 * (cx*cx + cy*cy + cz*cz).sqrt();
        }
        area
    }
}
