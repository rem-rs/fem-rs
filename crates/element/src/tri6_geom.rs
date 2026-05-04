//! Geometric (isoparametric) tools for **Tri6** — the 6-node quadratic triangle.
//!
//! These functions provide the element-layer geometry kernel used both by
//! `fem-element` itself and by `fem-mesh::CurvedMesh` for the isoparametric
//! mapping `F: K̂ → K`.  Centralising the logic here avoids duplication between
//! the element and mesh crates.
//!
//! # DOF / node ordering (MFEM / Gmsh convention)
//! ```text
//!    2
//!    | \
//!    5   4
//!    |     \
//!    0 - 3 - 1
//!
//!  node 0 = (0, 0)      vertex
//!  node 1 = (1, 0)      vertex
//!  node 2 = (0, 1)      vertex
//!  node 3 = (0.5, 0)    mid-edge 0-1
//!  node 4 = (0.5, 0.5)  mid-edge 1-2
//!  node 5 = (0, 0.5)    mid-edge 0-2
//! ```
//!
//! All functions operate in the **reference coordinate system** `(ξ, η)` with
//! `0 ≤ ξ`, `0 ≤ η`, `ξ + η ≤ 1`.

// ─── Basis functions ──────────────────────────────────────────────────────────

/// Evaluate the 6 TriP2 Lagrange basis functions at `(xi[0], xi[1])`.
///
/// `phi` must have length ≥ 6.
///
/// Barycentric coordinates: `λ₁ = 1−ξ−η`, `λ₂ = ξ`, `λ₃ = η`.
#[inline]
pub fn eval_basis(xi: &[f64], phi: &mut [f64]) {
    let (x, y) = (xi[0], xi[1]);
    let l1 = 1.0 - x - y;
    let l2 = x;
    let l3 = y;
    phi[0] = l1 * (2.0 * l1 - 1.0);
    phi[1] = l2 * (2.0 * l2 - 1.0);
    phi[2] = l3 * (2.0 * l3 - 1.0);
    phi[3] = 4.0 * l1 * l2;
    phi[4] = 4.0 * l2 * l3;
    phi[5] = 4.0 * l1 * l3;
}

/// Evaluate the gradients of the 6 TriP2 Lagrange basis functions at `(xi[0], xi[1])`.
///
/// `grads` is row-major with shape `6 × 2`:
/// `grads[2*i]` = ∂φᵢ/∂ξ,  `grads[2*i+1]` = ∂φᵢ/∂η.
///
/// `grads` must have length ≥ 12.
#[inline]
pub fn eval_grad_basis(xi: &[f64], grads: &mut [f64]) {
    let (x, y) = (xi[0], xi[1]);
    // node 0: φ = λ₁(2λ₁−1)
    grads[0]  = 4.0 * x + 4.0 * y - 3.0;   // ∂/∂ξ
    grads[1]  = 4.0 * x + 4.0 * y - 3.0;   // ∂/∂η
    // node 1: φ = λ₂(2λ₂−1)
    grads[2]  = 4.0 * x - 1.0;
    grads[3]  = 0.0;
    // node 2: φ = λ₃(2λ₃−1)
    grads[4]  = 0.0;
    grads[5]  = 4.0 * y - 1.0;
    // node 3: φ = 4λ₁λ₂
    grads[6]  = 4.0 * (1.0 - 2.0 * x - y);
    grads[7]  = -4.0 * x;
    // node 4: φ = 4λ₂λ₃
    grads[8]  = 4.0 * y;
    grads[9]  = 4.0 * x;
    // node 5: φ = 4λ₁λ₃
    grads[10] = -4.0 * y;
    grads[11] = 4.0 * (1.0 - x - 2.0 * y);
}

// ─── Geometric mapping ────────────────────────────────────────────────────────

/// Compute the isoparametric Jacobian `J` and its determinant for a Tri6 element.
///
/// `nodes` is a flat array of 12 `f64` values: `[x0,y0, x1,y1, …, x5,y5]`
/// (coordinates of the 6 geometric nodes in the order above).
///
/// Returns `(j_flat, det)` where `j_flat = [J00, J01, J10, J11]` (row-major 2×2).
///
/// `J[i,j] = Σ_k x_k[i] · ∂φ_k/∂ξ_j`
#[inline]
pub fn jacobian(nodes_flat: &[f64], xi: &[f64]) -> ([f64; 4], f64) {
    let mut grads = [0.0f64; 12];
    eval_grad_basis(xi, &mut grads);

    let mut j = [0.0f64; 4]; // [J00, J01, J10, J11]
    for k in 0..6 {
        let xk = nodes_flat[2 * k    ];
        let yk = nodes_flat[2 * k + 1];
        let dphi_dxi = grads[2 * k    ];
        let dphi_det = grads[2 * k + 1];
        j[0] += xk * dphi_dxi;  // J[0,0]
        j[1] += xk * dphi_det;  // J[0,1]
        j[2] += yk * dphi_dxi;  // J[1,0]
        j[3] += yk * dphi_det;  // J[1,1]
    }
    let det = j[0] * j[3] - j[1] * j[2];
    (j, det)
}

/// Compute the inverse of a 2×2 Jacobian given its flattened representation and determinant.
///
/// `j_flat = [J00, J01, J10, J11]`.  Returns `[J⁻¹₀₀, J⁻¹₀₁, J⁻¹₁₀, J⁻¹₁₁]`.
///
/// # Panics
/// Panics (in debug) if `|det| < f64::EPSILON`.
#[inline]
pub fn inv_jacobian(j_flat: &[f64; 4], det: f64) -> [f64; 4] {
    let inv_det = 1.0 / det;
    [
         j_flat[3] * inv_det,   // J⁻¹₀₀ =  J₁₁ / det
        -j_flat[1] * inv_det,   // J⁻¹₀₁ = −J₀₁ / det
        -j_flat[2] * inv_det,   // J⁻¹₁₀ = −J₁₀ / det
         j_flat[0] * inv_det,   // J⁻¹₁₁ =  J₀₀ / det
    ]
}

/// Map a reference point `xi` to physical coordinates using the Tri6 isoparametric mapping.
///
/// `nodes_flat` has the same layout as in [`jacobian`].
#[inline]
pub fn ref_to_phys(nodes_flat: &[f64], xi: &[f64]) -> [f64; 2] {
    let mut phi = [0.0f64; 6];
    eval_basis(xi, &mut phi);
    let mut xp = [0.0f64; 2];
    for k in 0..6 {
        xp[0] += nodes_flat[2 * k    ] * phi[k];
        xp[1] += nodes_flat[2 * k + 1] * phi[k];
    }
    xp
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_tri6_nodes() -> [f64; 12] {
        // Vertices: (0,0), (1,0), (0,1); mid-edges: (0.5,0), (0.5,0.5), (0,0.5)
        [
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            0.5, 0.0,
            0.5, 0.5,
            0.0, 0.5,
        ]
    }

    #[test]
    fn partition_of_unity() {
        let pts: &[&[f64]] = &[
            &[1.0/3.0, 1.0/3.0],
            &[0.1, 0.2],
            &[0.5, 0.5],
            &[0.0, 0.0],
            &[1.0, 0.0],
            &[0.0, 1.0],
        ];
        for xi in pts {
            let mut phi = [0.0f64; 6];
            eval_basis(xi, &mut phi);
            let sum: f64 = phi.iter().sum();
            assert!((sum - 1.0).abs() < 1e-14,
                "POU failed at xi={xi:?}: sum={sum:.15e}");
        }
    }

    #[test]
    fn basis_at_vertices() {
        // φᵢ(node_i) = 1, φᵢ(node_j≠i) = 0 for vertex nodes 0,1,2.
        let pts = [&[0.0f64, 0.0][..], &[1.0, 0.0][..], &[0.0, 1.0][..]];
        for (v, xi) in pts.iter().enumerate() {
            let mut phi = [0.0f64; 6];
            eval_basis(xi, &mut phi);
            assert!((phi[v] - 1.0).abs() < 1e-14,
                "phi[{v}] at vertex {v} = {} (expected 1)", phi[v]);
            for j in 0..3 {
                if j != v {
                    assert!(phi[j].abs() < 1e-14,
                        "phi[{j}] at vertex {v} = {} (expected 0)", phi[j]);
                }
            }
        }
    }

    #[test]
    fn basis_at_midpoints() {
        // Mid-edge node 3 = (0.5, 0): φ₃ = 1, all others = 0.
        let mid_pts: &[(&[f64], usize)] = &[
            (&[0.5, 0.0], 3),
            (&[0.5, 0.5], 4),
            (&[0.0, 0.5], 5),
        ];
        for &(xi, idx) in mid_pts {
            let mut phi = [0.0f64; 6];
            eval_basis(xi, &mut phi);
            assert!((phi[idx] - 1.0).abs() < 1e-14,
                "phi[{idx}] at mid-node {idx} = {} (expected 1)", phi[idx]);
        }
    }

    #[test]
    fn jacobian_unit_triangle_is_identity() {
        // For the unit reference triangle, J should be the 2×2 identity.
        let nodes = unit_tri6_nodes();
        let (j, det) = jacobian(&nodes, &[1.0/3.0, 1.0/3.0]);
        assert!((j[0] - 1.0).abs() < 1e-13, "J00 = {}", j[0]);
        assert!((j[1]).abs()       < 1e-13, "J01 = {}", j[1]);
        assert!((j[2]).abs()       < 1e-13, "J10 = {}", j[2]);
        assert!((j[3] - 1.0).abs() < 1e-13, "J11 = {}", j[3]);
        assert!((det - 1.0).abs()  < 1e-13, "det = {}", det);
    }

    #[test]
    fn jacobian_positive_in_reference_triangle() {
        let nodes = unit_tri6_nodes();
        let pts: &[&[f64]] = &[
            &[1.0/3.0, 1.0/3.0],
            &[0.1, 0.1],
            &[0.7, 0.2],
            &[0.1, 0.7],
        ];
        for xi in pts {
            let (_, det) = jacobian(&nodes, xi);
            assert!(det > 0.0, "det ≤ 0 at xi={xi:?}: det={det}");
        }
    }

    #[test]
    fn inv_jacobian_is_identity_inverse() {
        let nodes = unit_tri6_nodes();
        let (j, det) = jacobian(&nodes, &[0.25, 0.25]);
        let jinv = inv_jacobian(&j, det);
        // J * J⁻¹ = I
        let a00 = j[0] * jinv[0] + j[1] * jinv[2];
        let a01 = j[0] * jinv[1] + j[1] * jinv[3];
        let a10 = j[2] * jinv[0] + j[3] * jinv[2];
        let a11 = j[2] * jinv[1] + j[3] * jinv[3];
        assert!((a00 - 1.0).abs() < 1e-13, "J·J⁻¹ [0,0] = {a00}");
        assert!(a01.abs()          < 1e-13, "J·J⁻¹ [0,1] = {a01}");
        assert!(a10.abs()          < 1e-13, "J·J⁻¹ [1,0] = {a10}");
        assert!((a11 - 1.0).abs() < 1e-13, "J·J⁻¹ [1,1] = {a11}");
    }

    #[test]
    fn ref_to_phys_at_vertices() {
        let nodes = unit_tri6_nodes();
        let pts: &[(&[f64], [f64; 2])] = &[
            (&[0.0, 0.0], [0.0, 0.0]),
            (&[1.0, 0.0], [1.0, 0.0]),
            (&[0.0, 1.0], [0.0, 1.0]),
        ];
        for &(xi, expected) in pts {
            let xp = ref_to_phys(&nodes, xi);
            assert!((xp[0] - expected[0]).abs() < 1e-13,
                "x mismatch at {xi:?}: got {}, expected {}", xp[0], expected[0]);
            assert!((xp[1] - expected[1]).abs() < 1e-13,
                "y mismatch at {xi:?}: got {}, expected {}", xp[1], expected[1]);
        }
    }

    #[test]
    fn ref_to_phys_at_midpoints() {
        let nodes = unit_tri6_nodes();
        let pts: &[(&[f64], [f64; 2])] = &[
            (&[0.5, 0.0], [0.5, 0.0]),
            (&[0.5, 0.5], [0.5, 0.5]),
            (&[0.0, 0.5], [0.0, 0.5]),
        ];
        for &(xi, expected) in pts {
            let xp = ref_to_phys(&nodes, xi);
            assert!((xp[0] - expected[0]).abs() < 1e-13);
            assert!((xp[1] - expected[1]).abs() < 1e-13);
        }
    }

    #[test]
    fn scaled_triangle_jacobian() {
        // Nodes for triangle [0,0], [2,0], [0,2]: J = [[2,0],[0,2]], det = 4.
        let nodes = [
            0.0f64, 0.0,
            2.0,    0.0,
            0.0,    2.0,
            1.0,    0.0,
            1.0,    1.0,
            0.0,    1.0,
        ];
        let (j, det) = jacobian(&nodes, &[1.0/3.0, 1.0/3.0]);
        assert!((j[0] - 2.0).abs() < 1e-13);
        assert!((j[3] - 2.0).abs() < 1e-13);
        assert!((det - 4.0).abs()  < 1e-13);
    }

    #[test]
    fn grad_basis_agrees_with_finite_difference() {
        let xi = &[0.2, 0.3];
        let h = 1e-7;
        let mut phi = [0.0f64; 6];
        let mut phi_dx = [0.0f64; 6];
        let mut phi_dy = [0.0f64; 6];
        eval_basis(xi, &mut phi);
        eval_basis(&[xi[0] + h, xi[1]], &mut phi_dx);
        eval_basis(&[xi[0], xi[1] + h], &mut phi_dy);
        let mut grads = [0.0f64; 12];
        eval_grad_basis(xi, &mut grads);
        for i in 0..6 {
            let fd_dx = (phi_dx[i] - phi[i]) / h;
            let fd_dy = (phi_dy[i] - phi[i]) / h;
            assert!((grads[2*i]   - fd_dx).abs() < 1e-6,
                "∂φ{i}/∂ξ: analytic={:.8}, FD={:.8}", grads[2*i], fd_dx);
            assert!((grads[2*i+1] - fd_dy).abs() < 1e-6,
                "∂φ{i}/∂η: analytic={:.8}, FD={:.8}", grads[2*i+1], fd_dy);
        }
    }
}
