//! Lagrange elements on the reference triangle `(0,0),(1,0),(0,1)`.
//!
//! Barycentric coordinates: λ₁ = 1−ξ−η,  λ₂ = ξ,  λ₃ = η

use crate::quadrature::tri_rule;
use crate::reference::{QuadratureRule, ReferenceElement};

// ─── P1 ───────────────────────────────────────────────────────────────────────

/// Linear Lagrange element on the reference triangle — 3 DOFs at vertices.
///
/// Basis:
/// - φ₀ = 1−ξ−η  (vertex 0: origin)
/// - φ₁ = ξ       (vertex 1: (1,0))
/// - φ₂ = η       (vertex 2: (0,1))
pub struct TriP1;

impl ReferenceElement for TriP1 {
    fn dim(&self)    -> u8    { 2 }
    fn order(&self)  -> u8    { 1 }
    fn n_dofs(&self) -> usize  { 3 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        values[0] = 1.0 - x - y;
        values[1] = x;
        values[2] = y;
    }

    fn eval_grad_basis(&self, _xi: &[f64], grads: &mut [f64]) {
        // row-major [3×2]: grads[i*2 + j]
        grads[0] = -1.0;  grads[1] = -1.0;  // ∇φ₀
        grads[2] =  1.0;  grads[3] =  0.0;  // ∇φ₁
        grads[4] =  0.0;  grads[5] =  1.0;  // ∇φ₂
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { tri_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]]
    }
}

// ─── P2 ───────────────────────────────────────────────────────────────────────

/// Quadratic Lagrange element on the reference triangle — 6 DOFs.
///
/// DOF ordering:
/// - 0: vertex (0,0)   — φ₀ = λ₁(2λ₁−1)
/// - 1: vertex (1,0)   — φ₁ = λ₂(2λ₂−1)
/// - 2: vertex (0,1)   — φ₂ = λ₃(2λ₃−1)
/// - 3: edge midpoint (0.5, 0)   — φ₃ = 4λ₁λ₂
/// - 4: edge midpoint (0.5, 0.5) — φ₄ = 4λ₂λ₃
/// - 5: edge midpoint (0, 0.5)   — φ₅ = 4λ₁λ₃
pub struct TriP2;

impl ReferenceElement for TriP2 {
    fn dim(&self)    -> u8    { 2 }
    fn order(&self)  -> u8    { 2 }
    fn n_dofs(&self) -> usize  { 6 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        let l1 = 1.0 - x - y;
        let l2 = x;
        let l3 = y;
        values[0] = l1 * (2.0 * l1 - 1.0);
        values[1] = l2 * (2.0 * l2 - 1.0);
        values[2] = l3 * (2.0 * l3 - 1.0);
        values[3] = 4.0 * l1 * l2;
        values[4] = 4.0 * l2 * l3;
        values[5] = 4.0 * l1 * l3;
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        // ∂φ₀/∂ξ = (4ξ+4η−3),  ∂φ₀/∂η = same
        grads[0]  = 4.0 * x + 4.0 * y - 3.0;
        grads[1]  = 4.0 * x + 4.0 * y - 3.0;
        // ∂φ₁/∂ξ = 4ξ−1,  ∂φ₁/∂η = 0
        grads[2]  = 4.0 * x - 1.0;
        grads[3]  = 0.0;
        // ∂φ₂/∂ξ = 0,  ∂φ₂/∂η = 4η−1
        grads[4]  = 0.0;
        grads[5]  = 4.0 * y - 1.0;
        // ∂φ₃/∂ξ = 4(1−2ξ−η),  ∂φ₃/∂η = −4ξ
        grads[6]  = 4.0 * (1.0 - 2.0 * x - y);
        grads[7]  = -4.0 * x;
        // ∂φ₄/∂ξ = 4η,  ∂φ₄/∂η = 4ξ
        grads[8]  = 4.0 * y;
        grads[9]  = 4.0 * x;
        // ∂φ₅/∂ξ = −4η,  ∂φ₅/∂η = 4(1−ξ−2η)
        grads[10] = -4.0 * y;
        grads[11] = 4.0 * (1.0 - x - 2.0 * y);
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { tri_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0],
            vec![0.5, 0.0], vec![0.5, 0.5], vec![0.0, 0.5],
        ]
    }
}

// ─── P3 ───────────────────────────────────────────────────────────────────────

/// Cubic Lagrange element on the reference triangle — 10 DOFs.
///
/// DOF ordering (barycentric coordinates λ₁=1−ξ−η, λ₂=ξ, λ₃=η):
/// - 0: vertex (0,0)         — φ₀ = λ₁(3λ₁−1)(3λ₁−2)/2
/// - 1: vertex (1,0)         — φ₁ = λ₂(3λ₂−1)(3λ₂−2)/2
/// - 2: vertex (0,1)         — φ₂ = λ₃(3λ₃−1)(3λ₃−2)/2
/// - 3: edge v0→v1 @ (1/3,0) — φ₃ = 9λ₁λ₂(3λ₁−1)/2
/// - 4: edge v0→v1 @ (2/3,0) — φ₄ = 9λ₁λ₂(3λ₂−1)/2
/// - 5: edge v1→v2 @ (2/3,1/3) — φ₅ = 9λ₂λ₃(3λ₂−1)/2
/// - 6: edge v1→v2 @ (1/3,2/3) — φ₆ = 9λ₂λ₃(3λ₃−1)/2
/// - 7: edge v0→v2 @ (0,1/3) — φ₇ = 9λ₁λ₃(3λ₁−1)/2
/// - 8: edge v0→v2 @ (0,2/3) — φ₈ = 9λ₁λ₃(3λ₃−1)/2
/// - 9: bubble @ (1/3,1/3)   — φ₉ = 27λ₁λ₂λ₃
pub struct TriP3;

impl ReferenceElement for TriP3 {
    fn dim(&self)    -> u8    { 2 }
    fn order(&self)  -> u8    { 3 }
    fn n_dofs(&self) -> usize  { 10 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        let l1 = 1.0 - x - y;  // λ₁
        let l2 = x;              // λ₂
        let l3 = y;              // λ₃

        // Vertex DOFs
        values[0] = 0.5 * l1 * (3.0 * l1 - 1.0) * (3.0 * l1 - 2.0);
        values[1] = 0.5 * l2 * (3.0 * l2 - 1.0) * (3.0 * l2 - 2.0);
        values[2] = 0.5 * l3 * (3.0 * l3 - 1.0) * (3.0 * l3 - 2.0);
        // Edge v0→v1 DOFs (3λ₁−1 at 1/3 point, 3λ₂−1 at 2/3 point)
        values[3] = 4.5 * l1 * l2 * (3.0 * l1 - 1.0);
        values[4] = 4.5 * l1 * l2 * (3.0 * l2 - 1.0);
        // Edge v1→v2 DOFs
        values[5] = 4.5 * l2 * l3 * (3.0 * l2 - 1.0);
        values[6] = 4.5 * l2 * l3 * (3.0 * l3 - 1.0);
        // Edge v0→v2 DOFs
        values[7] = 4.5 * l1 * l3 * (3.0 * l1 - 1.0);
        values[8] = 4.5 * l1 * l3 * (3.0 * l3 - 1.0);
        // Bubble DOF
        values[9] = 27.0 * l1 * l2 * l3;
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        // grads layout: [φᵢ_dξ, φᵢ_dη] for i = 0..10
        // ∂λ₁/∂ξ = -1, ∂λ₁/∂η = -1
        // ∂λ₂/∂ξ =  1, ∂λ₂/∂η =  0
        // ∂λ₃/∂ξ =  0, ∂λ₃/∂η =  1
        let (x, y) = (xi[0], xi[1]);
        let l1 = 1.0 - x - y;
        let l2 = x;
        let l3 = y;

        // φ₀ = 0.5 * l1*(3l1-1)*(3l1-2)
        // d/dξ = 0.5 * (-1)*(3*(3l1-1)*(3l1-2) + l1*3*(3l1-2)*3 + l1*(3l1-1)*3*(-1))
        // = 0.5 * (-1) * (27l1²−18l1+2) * 3  ... let's expand directly
        // φ₀ = 0.5*(9l1³ - 9l1² + 2l1) = (27l1³ - 9l1² + 2l1)/2 ...
        // Simpler: d(φ₀)/dξ = (∂φ₀/∂l1) * (∂l1/∂ξ)
        // ∂φ₀/∂l1 = 0.5*(3*(3l1-1)*(3l1-2) + l1*9*(3l1-2) + l1*(3l1-1)*9) / ...
        // Use product rule: d/dl1 [l1(3l1-1)(3l1-2)] = (3l1-1)(3l1-2) + l1*3*(3l1-2) + l1*(3l1-1)*3
        //   = 27l1²-18l1+2
        // so ∂φ₀/∂ξ = 0.5*(27l1²-18l1+2)*(−1)
        //    ∂φ₀/∂η = 0.5*(27l1²-18l1+2)*(−1)
        let dphi0_dl1 = 0.5 * (27.0 * l1 * l1 - 18.0 * l1 + 2.0);
        grads[0] = -dphi0_dl1;
        grads[1] = -dphi0_dl1;

        // φ₁ = 0.5*l2*(3l2-1)*(3l2-2), ∂φ₁/∂l2 = 0.5*(27l2²-18l2+2)
        let dphi1_dl2 = 0.5 * (27.0 * l2 * l2 - 18.0 * l2 + 2.0);
        grads[2] = dphi1_dl2 * 1.0;
        grads[3] = dphi1_dl2 * 0.0;

        // φ₂ = 0.5*l3*(3l3-1)*(3l3-2), ∂φ₂/∂l3 = 0.5*(27l3²-18l3+2)
        let dphi2_dl3 = 0.5 * (27.0 * l3 * l3 - 18.0 * l3 + 2.0);
        grads[4] = dphi2_dl3 * 0.0;
        grads[5] = dphi2_dl3 * 1.0;

        // φ₃ = 4.5*l1*l2*(3l1-1)
        // d/dξ = 4.5 * [(-1)*l2*(3l1-1) + l1*(1)*(3l1-1) + l1*l2*(-3)]
        //      = 4.5 * [-l2*(3l1-1) + l1*(3l1-1) - 3l1l2]
        //      = 4.5 * [(3l1-1)*(l1-l2) - 3l1l2]
        // d/dη = 4.5 * [(-1)*l2*(3l1-1) + l1*0*(3l1-1) + l1*l2*(-3)]
        //      = 4.5 * [-l2*(3l1-1) - 3l1l2]
        //      = 4.5 * [-l2*(3l1-1+3l1)] = 4.5 * [-l2*(6l1-1)]
        grads[6] = 4.5 * ((3.0*l1 - 1.0)*(l1 - l2) - 3.0*l1*l2);
        grads[7] = 4.5 * (-l2*(6.0*l1 - 1.0));

        // φ₄ = 4.5*l1*l2*(3l2-1)
        // d/dξ = 4.5 * [(-1)*l2*(3l2-1) + l1*(3l2-1) + l1*l2*3]
        //      = 4.5 * [(3l2-1)*(l1-l2) + 3l1l2]  ... wait, let's redo:
        // ∂/∂ξ [l1*l2*(3l2-1)] = (-1)*l2*(3l2-1) + l1*1*(3l2-1) + l1*l2*3*1
        //   = l2*(3l2-1)*(l1/l2 -1) ... no, just:
        //   = -l2*(3l2-1) + l1*(3l2-1) + 3l1l2
        //   = (3l2-1)*(l1-l2) + 3l1l2
        // ∂/∂η [l1*l2*(3l2-1)] = (-1)*l2*(3l2-1) + l1*0*(3l2-1) + l1*l2*3*0
        //   = -l2*(3l2-1)
        grads[8]  = 4.5 * ((3.0*l2 - 1.0)*(l1 - l2) + 3.0*l1*l2);
        grads[9]  = 4.5 * (-l2*(3.0*l2 - 1.0));

        // φ₅ = 4.5*l2*l3*(3l2-1)
        // ∂/∂ξ = 4.5 * [l3*(3l2-1) + l2*0 + l2*l3*3] = 4.5*l3*(3l2-1+3l2) = 4.5*l3*(6l2-1)
        // ∂/∂η = 4.5 * [0*l3*(3l2-1) + l2*1*(3l2-1) + l2*l3*0] = 4.5*l2*(3l2-1)
        grads[10] = 4.5 * l3 * (6.0*l2 - 1.0);
        grads[11] = 4.5 * l2 * (3.0*l2 - 1.0);

        // φ₆ = 4.5*l2*l3*(3l3-1)
        // ∂/∂ξ = 4.5 * [1*l3*(3l3-1) + l2*0*(3l3-1) + l2*l3*0] = 4.5*l3*(3l3-1)
        // ∂/∂η = 4.5 * [0*l3*(3l3-1) + l2*1*(3l3-1) + l2*l3*3] = 4.5*l2*(3l3-1+3l3) = 4.5*l2*(6l3-1)
        grads[12] = 4.5 * l3 * (3.0*l3 - 1.0);
        grads[13] = 4.5 * l2 * (6.0*l3 - 1.0);

        // φ₇ = 4.5*l1*l3*(3l1-1)
        // ∂/∂ξ = 4.5 * [(-1)*l3*(3l1-1) + l1*0*(3l1-1) + l1*l3*(-3)]
        //       = 4.5 * [-l3*(3l1-1) - 3l1l3] = 4.5*(-l3)*(3l1-1+3l1) = 4.5*(-l3)*(6l1-1)
        // ∂/∂η = 4.5 * [(-1)*l3*(3l1-1) + l1*1*(3l1-1) + l1*l3*(-3)]
        //       = 4.5 * [(3l1-1)*(l1-l3) - 3l1l3]  ... wait, ∂l3/∂η=1, ∂l1/∂η=-1
        // ∂/∂η [l1*l3*(3l1-1)] = (-1)*l3*(3l1-1) + l1*1*(3l1-1) + l1*l3*3*(-1)
        //   = (3l1-1)*(l1-l3) - 3l1*l3
        grads[14] = 4.5 * (-l3*(6.0*l1 - 1.0));
        grads[15] = 4.5 * ((3.0*l1 - 1.0)*(l1 - l3) - 3.0*l1*l3);

        // φ₈ = 4.5*l1*l3*(3l3-1)
        // ∂/∂ξ [l1*l3*(3l3-1)] = (-1)*l3*(3l3-1) + l1*0*(3l3-1) + l1*l3*0
        //   = -l3*(3l3-1)
        // ∂/∂η [l1*l3*(3l3-1)] = (-1)*l3*(3l3-1) + l1*1*(3l3-1) + l1*l3*3
        //   = (3l3-1)*(l1-l3) + 3l1l3  ... ∂l3/∂η=1, ∂l1/∂η=-1, ∂(3l3-1)/∂η=3
        // = (3l3-1)*(-l3 + l1) + 3l1l3
        grads[16] = 4.5 * (-l3*(3.0*l3 - 1.0));
        grads[17] = 4.5 * ((3.0*l3 - 1.0)*(l1 - l3) + 3.0*l1*l3);

        // φ₉ = 27*l1*l2*l3
        // ∂/∂ξ = 27*[(-1)*l2*l3 + l1*1*l3 + l1*l2*0] = 27*l3*(l1-l2)  ...
        //   wait: ∂l1/∂ξ=-1, ∂l2/∂ξ=1, ∂l3/∂ξ=0
        // = 27*(-l2*l3 + l1*l3) = 27*l3*(l1-l2)  ...
        // Hmm, need ∂(l1*l2*l3)/∂ξ = l2*l3*(−1) + l1*l3*(1) + l1*l2*(0) = l3*(l1-l2)
        // ∂/∂η = l2*l3*(−1) + l1*l3*(0) + l1*l2*(1) = -l2*l3 + l1*l2 = l2*(l1-l3)
        grads[18] = 27.0 * l3 * (l1 - l2);
        grads[19] = 27.0 * l2 * (l1 - l3);
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { tri_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let t = 1.0 / 3.0;
        let t2 = 2.0 / 3.0;
        vec![
            vec![0.0, 0.0],  // v0
            vec![1.0, 0.0],  // v1
            vec![0.0, 1.0],  // v2
            vec![t,   0.0],  // edge v0→v1 @ 1/3
            vec![t2,  0.0],  // edge v0→v1 @ 2/3
            vec![t2,  t  ],  // edge v1→v2 @ 1/3 from v1
            vec![t,   t2 ],  // edge v1→v2 @ 2/3 from v1
            vec![0.0, t  ],  // edge v0→v2 @ 1/3
            vec![0.0, t2 ],  // edge v0→v2 @ 2/3
            vec![t,   t  ],  // bubble
        ]
    }
}

// ─── P4 ───────────────────────────────────────────────────────────────────────

/// Quartic Lagrange element on the reference triangle — 15 DOFs.
///
/// Uses the rising-factorial barycentric basis for compactness and stability.
/// DOF ordering follows the TriPk / MFEM convention.
pub struct TriP4;

const TRIP4_IJK: [(usize, usize, usize); 15] = [
    (0,0,4), (4,0,0), (0,4,0),                         // vertices
    (1,0,3), (2,0,2), (3,0,1),                         // edge 0→1
    (3,1,0), (2,2,0), (1,3,0),                         // edge 1→2
    (0,1,3), (0,2,2), (0,3,1),                         // edge 2→0
    (2,1,1), (1,2,1), (1,1,2),                         // face interior
];

const TRIP4_COORDS: [[f64; 2]; 15] = [
    [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
    [0.25, 0.0], [0.5, 0.0],  [0.75, 0.0],
    [0.75, 0.25], [0.5, 0.5], [0.25, 0.75],
    [0.0, 0.25], [0.0, 0.5],  [0.0, 0.75],
    [0.5, 0.25], [0.25, 0.5], [0.25, 0.25],
];

impl ReferenceElement for TriP4 {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { 4 }
    fn n_dofs(&self) -> usize { 15 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let pf = 4.0_f64;
        let t0 = pf * xi[0];
        let t1 = pf * xi[1];
        let t2 = pf * (1.0 - xi[0] - xi[1]);
        for (dof_idx, &(i, j, k)) in TRIP4_IJK.iter().enumerate() {
            values[dof_idx] = rising_val(i, t0) * rising_val(j, t1) * rising_val(k, t2);
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let pf = 4.0_f64;
        let t0 = pf * xi[0];
        let t1 = pf * xi[1];
        let t2 = pf * (1.0 - xi[0] - xi[1]);
        for (dof_idx, &(i, j, k)) in TRIP4_IJK.iter().enumerate() {
            let vi = rising_val(i, t0);
            let vj = rising_val(j, t1);
            let vk = rising_val(k, t2);
            let di = rising_deriv(i, t0);
            let dj = rising_deriv(j, t1);
            let dk = rising_deriv(k, t2);
            grads[dof_idx * 2]     = pf * (di * vj * vk - vi * vj * dk);
            grads[dof_idx * 2 + 1] = pf * (vi * dj * vk - vi * vj * dk);
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { tri_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        TRIP4_COORDS.iter().map(|c| vec![c[0], c[1]]).collect()
    }
}

// ─── P5 ───────────────────────────────────────────────────────────────────────

/// Quintic Lagrange element on the reference triangle — 21 DOFs.
pub struct TriP5;

const TRIP5_IJK: [(usize, usize, usize); 21] = [
    (0,0,5), (5,0,0), (0,5,0),                         // vertices
    (1,0,4), (2,0,3), (3,0,2), (4,0,1),               // edge 0→1
    (4,1,0), (3,2,0), (2,3,0), (1,4,0),               // edge 1→2
    (0,1,4), (0,2,3), (0,3,2), (0,4,1),               // edge 2→0
    (1,1,3), (2,1,2), (3,1,1),                         // face interior j=1
    (1,2,2), (2,2,1),                                   // face interior j=2
    (1,3,1),                                             // face interior j=3
];

const TRIP5_COORDS: [[f64; 2]; 21] = [
    [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
    [0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0],
    [0.8, 0.2], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8],
    [0.0, 0.2], [0.0, 0.4], [0.0, 0.6], [0.0, 0.8],
    [0.2, 0.2], [0.4, 0.2], [0.6, 0.2],
    [0.2, 0.4], [0.4, 0.4],
    [0.2, 0.6],
];

impl ReferenceElement for TriP5 {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { 5 }
    fn n_dofs(&self) -> usize { 21 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let pf = 5.0_f64;
        let t0 = pf * xi[0];
        let t1 = pf * xi[1];
        let t2 = pf * (1.0 - xi[0] - xi[1]);
        for (dof_idx, &(i, j, k)) in TRIP5_IJK.iter().enumerate() {
            values[dof_idx] = rising_val(i, t0) * rising_val(j, t1) * rising_val(k, t2);
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let pf = 5.0_f64;
        let t0 = pf * xi[0];
        let t1 = pf * xi[1];
        let t2 = pf * (1.0 - xi[0] - xi[1]);
        for (dof_idx, &(i, j, k)) in TRIP5_IJK.iter().enumerate() {
            let vi = rising_val(i, t0);
            let vj = rising_val(j, t1);
            let vk = rising_val(k, t2);
            let di = rising_deriv(i, t0);
            let dj = rising_deriv(j, t1);
            let dk = rising_deriv(k, t2);
            grads[dof_idx * 2]     = pf * (di * vj * vk - vi * vj * dk);
            grads[dof_idx * 2 + 1] = pf * (vi * dj * vk - vi * vj * dk);
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { tri_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        TRIP5_COORDS.iter().map(|c| vec![c[0], c[1]]).collect()
    }
}

// ─── P6 ───────────────────────────────────────────────────────────────────────

/// Sextic Lagrange element on the reference triangle — 28 DOFs.
pub struct TriP6;

const TRIP6_IJK: [(usize, usize, usize); 28] = [
    (0,0,6), (6,0,0), (0,6,0),                         // vertices
    (1,0,5), (2,0,4), (3,0,3), (4,0,2), (5,0,1),       // edge 0→1
    (5,1,0), (4,2,0), (3,3,0), (2,4,0), (1,5,0),       // edge 1→2
    (0,1,5), (0,2,4), (0,3,3), (0,4,2), (0,5,1),       // edge 2→0
    (1,1,4), (2,1,3), (3,1,2), (4,1,1),                 // face j=1
    (1,2,3), (2,2,2), (3,2,1),                           // face j=2
    (1,3,2), (2,3,1),                                     // face j=3
    (1,4,1),                                               // face j=4
];

const TRIP6_COORDS: [[f64; 2]; 28] = [
    [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
    [1.0/6.0, 0.0], [2.0/6.0, 0.0], [3.0/6.0, 0.0], [4.0/6.0, 0.0], [5.0/6.0, 0.0],
    [5.0/6.0, 1.0/6.0], [4.0/6.0, 2.0/6.0], [3.0/6.0, 3.0/6.0], [2.0/6.0, 4.0/6.0], [1.0/6.0, 5.0/6.0],
    [0.0, 1.0/6.0], [0.0, 2.0/6.0], [0.0, 3.0/6.0], [0.0, 4.0/6.0], [0.0, 5.0/6.0],
    [1.0/6.0, 1.0/6.0], [2.0/6.0, 1.0/6.0], [3.0/6.0, 1.0/6.0], [4.0/6.0, 1.0/6.0],
    [1.0/6.0, 2.0/6.0], [2.0/6.0, 2.0/6.0], [3.0/6.0, 2.0/6.0],
    [1.0/6.0, 3.0/6.0], [2.0/6.0, 3.0/6.0],
    [1.0/6.0, 4.0/6.0],
];

impl ReferenceElement for TriP6 {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { 6 }
    fn n_dofs(&self) -> usize { 28 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let pf = 6.0_f64;
        let t0 = pf * xi[0];
        let t1 = pf * xi[1];
        let t2 = pf * (1.0 - xi[0] - xi[1]);
        for (dof_idx, &(i, j, k)) in TRIP6_IJK.iter().enumerate() {
            values[dof_idx] = rising_val(i, t0) * rising_val(j, t1) * rising_val(k, t2);
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let pf = 6.0_f64;
        let t0 = pf * xi[0];
        let t1 = pf * xi[1];
        let t2 = pf * (1.0 - xi[0] - xi[1]);
        for (dof_idx, &(i, j, k)) in TRIP6_IJK.iter().enumerate() {
            let vi = rising_val(i, t0);
            let vj = rising_val(j, t1);
            let vk = rising_val(k, t2);
            let di = rising_deriv(i, t0);
            let dj = rising_deriv(j, t1);
            let dk = rising_deriv(k, t2);
            grads[dof_idx * 2]     = pf * (di * vj * vk - vi * vj * dk);
            grads[dof_idx * 2 + 1] = pf * (vi * dj * vk - vi * vj * dk);
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { tri_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        TRIP6_COORDS.iter().map(|c| vec![c[0], c[1]]).collect()
    }
}

// ─── P7 ───────────────────────────────────────────────────────────────────────

/// Septic Lagrange element on the reference triangle — 36 DOFs.
pub struct TriP7;

const TRIP7_IJK: [(usize, usize, usize); 36] = [
    (0,0,7), (7,0,0), (0,7,0),                         // vertices
    (1,0,6), (2,0,5), (3,0,4), (4,0,3), (5,0,2), (6,0,1), // edge 0→1
    (6,1,0), (5,2,0), (4,3,0), (3,4,0), (2,5,0), (1,6,0), // edge 1→2
    (0,1,6), (0,2,5), (0,3,4), (0,4,3), (0,5,2), (0,6,1), // edge 2→0
    (1,1,5), (2,1,4), (3,1,3), (4,1,2), (5,1,1),           // face j=1
    (1,2,4), (2,2,3), (3,2,2), (4,2,1),                     // face j=2
    (1,3,3), (2,3,2), (3,3,1),                               // face j=3
    (1,4,2), (2,4,1),                                         // face j=4
    (1,5,1),                                                   // face j=5
];

const TRIP7_COORDS: [[f64; 2]; 36] = [
    [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
    [1.0/7.0, 0.0], [2.0/7.0, 0.0], [3.0/7.0, 0.0], [4.0/7.0, 0.0], [5.0/7.0, 0.0], [6.0/7.0, 0.0],
    [6.0/7.0, 1.0/7.0], [5.0/7.0, 2.0/7.0], [4.0/7.0, 3.0/7.0], [3.0/7.0, 4.0/7.0], [2.0/7.0, 5.0/7.0], [1.0/7.0, 6.0/7.0],
    [0.0, 1.0/7.0], [0.0, 2.0/7.0], [0.0, 3.0/7.0], [0.0, 4.0/7.0], [0.0, 5.0/7.0], [0.0, 6.0/7.0],
    [1.0/7.0, 1.0/7.0], [2.0/7.0, 1.0/7.0], [3.0/7.0, 1.0/7.0], [4.0/7.0, 1.0/7.0], [5.0/7.0, 1.0/7.0],
    [1.0/7.0, 2.0/7.0], [2.0/7.0, 2.0/7.0], [3.0/7.0, 2.0/7.0], [4.0/7.0, 2.0/7.0],
    [1.0/7.0, 3.0/7.0], [2.0/7.0, 3.0/7.0], [3.0/7.0, 3.0/7.0],
    [1.0/7.0, 4.0/7.0], [2.0/7.0, 4.0/7.0],
    [1.0/7.0, 5.0/7.0],
];

impl ReferenceElement for TriP7 {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { 7 }
    fn n_dofs(&self) -> usize { 36 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let pf = 7.0_f64;
        let t0 = pf * xi[0];
        let t1 = pf * xi[1];
        let t2 = pf * (1.0 - xi[0] - xi[1]);
        for (dof_idx, &(i, j, k)) in TRIP7_IJK.iter().enumerate() {
            values[dof_idx] = rising_val(i, t0) * rising_val(j, t1) * rising_val(k, t2);
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let pf = 7.0_f64;
        let t0 = pf * xi[0];
        let t1 = pf * xi[1];
        let t2 = pf * (1.0 - xi[0] - xi[1]);
        for (dof_idx, &(i, j, k)) in TRIP7_IJK.iter().enumerate() {
            let vi = rising_val(i, t0);
            let vj = rising_val(j, t1);
            let vk = rising_val(k, t2);
            let di = rising_deriv(i, t0);
            let dj = rising_deriv(j, t1);
            let dk = rising_deriv(k, t2);
            grads[dof_idx * 2]     = pf * (di * vj * vk - vi * vj * dk);
            grads[dof_idx * 2 + 1] = pf * (vi * dj * vk - vi * vj * dk);
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { tri_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        TRIP7_COORDS.iter().map(|c| vec![c[0], c[1]]).collect()
    }
}

// ─── P8 ───────────────────────────────────────────────────────────────────────

/// Octic Lagrange element on the reference triangle — 45 DOFs.
pub struct TriP8;

const TRIP8_IJK: [(usize, usize, usize); 45] = [
    (0,0,8), (8,0,0), (0,8,0),
    (1,0,7),(2,0,6),(3,0,5),(4,0,4),(5,0,3),(6,0,2),(7,0,1),
    (7,1,0),(6,2,0),(5,3,0),(4,4,0),(3,5,0),(2,6,0),(1,7,0),
    (0,1,7),(0,2,6),(0,3,5),(0,4,4),(0,5,3),(0,6,2),(0,7,1),
    (1,1,6),(2,1,5),(3,1,4),(4,1,3),(5,1,2),(6,1,1),
    (1,2,5),(2,2,4),(3,2,3),(4,2,2),(5,2,1),
    (1,3,4),(2,3,3),(3,3,2),(4,3,1),
    (1,4,3),(2,4,2),(3,4,1),
    (1,5,2),(2,5,1),
    (1,6,1),
];

const TRIP8_COORDS: [[f64; 2]; 45] = [
    [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
    [1./8., 0.], [2./8., 0.], [3./8., 0.], [4./8., 0.], [5./8., 0.], [6./8., 0.], [7./8., 0.],
    [7./8., 1./8.], [6./8., 2./8.], [5./8., 3./8.], [4./8., 4./8.], [3./8., 5./8.], [2./8., 6./8.], [1./8., 7./8.],
    [0., 1./8.], [0., 2./8.], [0., 3./8.], [0., 4./8.], [0., 5./8.], [0., 6./8.], [0., 7./8.],
    [1./8., 1./8.], [2./8., 1./8.], [3./8., 1./8.], [4./8., 1./8.], [5./8., 1./8.], [6./8., 1./8.],
    [1./8., 2./8.], [2./8., 2./8.], [3./8., 2./8.], [4./8., 2./8.], [5./8., 2./8.],
    [1./8., 3./8.], [2./8., 3./8.], [3./8., 3./8.], [4./8., 3./8.],
    [1./8., 4./8.], [2./8., 4./8.], [3./8., 4./8.],
    [1./8., 5./8.], [2./8., 5./8.],
    [1./8., 6./8.],
];

impl ReferenceElement for TriP8 {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { 8 }
    fn n_dofs(&self) -> usize { 45 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let pf = 8.0_f64;
        let t0 = pf * xi[0];
        let t1 = pf * xi[1];
        let t2 = pf * (1.0 - xi[0] - xi[1]);
        for (dof_idx, &(i, j, k)) in TRIP8_IJK.iter().enumerate() {
            values[dof_idx] = rising_val(i, t0) * rising_val(j, t1) * rising_val(k, t2);
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let pf = 8.0_f64;
        let t0 = pf * xi[0];
        let t1 = pf * xi[1];
        let t2 = pf * (1.0 - xi[0] - xi[1]);
        for (dof_idx, &(i, j, k)) in TRIP8_IJK.iter().enumerate() {
            let vi = rising_val(i, t0);
            let vj = rising_val(j, t1);
            let vk = rising_val(k, t2);
            let di = rising_deriv(i, t0);
            let dj = rising_deriv(j, t1);
            let dk = rising_deriv(k, t2);
            grads[dof_idx * 2]     = pf * (di * vj * vk - vi * vj * dk);
            grads[dof_idx * 2 + 1] = pf * (vi * dj * vk - vi * vj * dk);
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { tri_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        TRIP8_COORDS.iter().map(|c| vec![c[0], c[1]]).collect()
    }
}

// ─── P9 ───────────────────────────────────────────────────────────────────────

/// Nonic Lagrange element on the reference triangle — 55 DOFs.
pub struct TriP9;

const TRIP9_IJK: [(usize, usize, usize); 55] = [
    (0,0,9), (9,0,0), (0,9,0),
    (1,0,8),(2,0,7),(3,0,6),(4,0,5),(5,0,4),(6,0,3),(7,0,2),(8,0,1),
    (8,1,0),(7,2,0),(6,3,0),(5,4,0),(4,5,0),(3,6,0),(2,7,0),(1,8,0),
    (0,1,8),(0,2,7),(0,3,6),(0,4,5),(0,5,4),(0,6,3),(0,7,2),(0,8,1),
    (1,1,7),(2,1,6),(3,1,5),(4,1,4),(5,1,3),(6,1,2),(7,1,1),
    (1,2,6),(2,2,5),(3,2,4),(4,2,3),(5,2,2),(6,2,1),
    (1,3,5),(2,3,4),(3,3,3),(4,3,2),(5,3,1),
    (1,4,4),(2,4,3),(3,4,2),(4,4,1),
    (1,5,3),(2,5,2),(3,5,1),
    (1,6,2),(2,6,1),
    (1,7,1),
];

impl ReferenceElement for TriP9 {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { 9 }
    fn n_dofs(&self) -> usize { 55 }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let p = 9.0; let t0=p*xi[0]; let t1=p*xi[1]; let t2=p*(1.0-xi[0]-xi[1]);
        for (i, &(a,b,c)) in TRIP9_IJK.iter().enumerate() { values[i]=rising_val(a,t0)*rising_val(b,t1)*rising_val(c,t2); }
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let p=9.0; let t0=p*xi[0]; let t1=p*xi[1]; let t2=p*(1.0-xi[0]-xi[1]);
        for (i, &(a,b,c)) in TRIP9_IJK.iter().enumerate() {
            let (va,vb,vc)=(rising_val(a,t0),rising_val(b,t1),rising_val(c,t2));
            grads[i*2]=p*(rising_deriv(a,t0)*vb*vc - va*vb*rising_deriv(c,t2));
            grads[i*2+1]=p*(va*rising_deriv(b,t1)*vc - va*vb*rising_deriv(c,t2));
        }
    }
    fn quadrature(&self, o: u8) -> QuadratureRule { tri_rule(o) }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let p=9.0; (0..55).map(|i| { let (a,b,_)=TRIP9_IJK[i]; vec![a as f64/p, b as f64/p] }).collect()
    }
}

// ─── P10 ──────────────────────────────────────────────────────────────────────

/// Decic Lagrange element on the reference triangle — 66 DOFs.
pub struct TriP10;

const TRIP10_IJK: [(usize, usize, usize); 66] = [
    (0,0,10),(10,0,0),(0,10,0),
    (1,0,9),(2,0,8),(3,0,7),(4,0,6),(5,0,5),(6,0,4),(7,0,3),(8,0,2),(9,0,1),
    (9,1,0),(8,2,0),(7,3,0),(6,4,0),(5,5,0),(4,6,0),(3,7,0),(2,8,0),(1,9,0),
    (0,1,9),(0,2,8),(0,3,7),(0,4,6),(0,5,5),(0,6,4),(0,7,3),(0,8,2),(0,9,1),
    (1,1,8),(2,1,7),(3,1,6),(4,1,5),(5,1,4),(6,1,3),(7,1,2),(8,1,1),
    (1,2,7),(2,2,6),(3,2,5),(4,2,4),(5,2,3),(6,2,2),(7,2,1),
    (1,3,6),(2,3,5),(3,3,4),(4,3,3),(5,3,2),(6,3,1),
    (1,4,5),(2,4,4),(3,4,3),(4,4,2),(5,4,1),
    (1,5,4),(2,5,3),(3,5,2),(4,5,1),
    (1,6,3),(2,6,2),(3,6,1),
    (1,7,2),(2,7,1),
    (1,8,1),
];

impl ReferenceElement for TriP10 {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { 10 }
    fn n_dofs(&self) -> usize { 66 }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let p=10.0; let t0=p*xi[0]; let t1=p*xi[1]; let t2=p*(1.0-xi[0]-xi[1]);
        for (i, &(a,b,c)) in TRIP10_IJK.iter().enumerate() { values[i]=rising_val(a,t0)*rising_val(b,t1)*rising_val(c,t2); }
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let p=10.0; let t0=p*xi[0]; let t1=p*xi[1]; let t2=p*(1.0-xi[0]-xi[1]);
        for (i, &(a,b,c)) in TRIP10_IJK.iter().enumerate() {
            let (va,vb,vc)=(rising_val(a,t0),rising_val(b,t1),rising_val(c,t2));
            grads[i*2]=p*(rising_deriv(a,t0)*vb*vc - va*vb*rising_deriv(c,t2));
            grads[i*2+1]=p*(va*rising_deriv(b,t1)*vc - va*vb*rising_deriv(c,t2));
        }
    }
    fn quadrature(&self, o: u8) -> QuadratureRule { tri_rule(o) }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let p=10.0; (0..66).map(|i| { let (a,b,_)=TRIP10_IJK[i]; vec![a as f64/p, b as f64/p] }).collect()
    }
}

/// Rising-factorial basis L_n(t) = Π_{a=0}^{n-1} (t - a) / (n - a), with L_0 = 1.
fn rising_val(n: usize, t: f64) -> f64 {
    if n == 0 { return 1.0; }
    let mut val = 1.0;
    for a in 0..n { val *= (t - a as f64) / (n as f64 - a as f64); }
    val
}

/// Derivative L_n'(t) = Σ_{b=0}^{n-1} 1/(n-b) · Π_{a≠b} (t-a)/(n-a).
fn rising_deriv(n: usize, t: f64) -> f64 {
    if n == 0 { return 0.0; }
    let nf = n as f64;
    let mut sum = 0.0;
    for b in 0..n {
        let mut term = 1.0;
        for a in 0..n {
            if a != b { term *= (t - a as f64) / (nf - a as f64); }
        }
        sum += term / (nf - b as f64);
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_pou(elem: &dyn ReferenceElement) {
        let rule = elem.quadrature(5);
        let mut phi = vec![0.0_f64; elem.n_dofs()];
        for pt in &rule.points {
            elem.eval_basis(pt, &mut phi);
            let s: f64 = phi.iter().sum();
            assert!((s - 1.0).abs() < 1e-13, "POU failed sum={s}");
        }
    }

    fn check_grad_zero(elem: &dyn ReferenceElement) {
        let dim = elem.dim() as usize;
        let rule = elem.quadrature(5);
        let mut g = vec![0.0_f64; elem.n_dofs() * dim];
        for pt in &rule.points {
            elem.eval_grad_basis(pt, &mut g);
            for d in 0..dim {
                let s: f64 = (0..elem.n_dofs()).map(|i| g[i * dim + d]).sum();
                assert!(s.abs() < 1e-12, "grad sum d={d} = {s}");
            }
        }
    }

    #[test] fn tri_p1_pou()       { check_pou(&TriP1); }
    #[test] fn tri_p1_grad_zero() { check_grad_zero(&TriP1); }
    #[test] fn tri_p2_pou()       { check_pou(&TriP2); }
    #[test] fn tri_p2_grad_zero() { check_grad_zero(&TriP2); }
    #[test] fn tri_p3_pou()       { check_pou(&TriP3); }
    #[test] fn tri_p3_grad_zero() { check_grad_zero(&TriP3); }

    #[test]
    fn tri_p3_vertex_dofs() {
        let mut phi = vec![0.0_f64; 10];
        // φ₀ = 1 at vertex 0 (0,0)
        TriP3.eval_basis(&[0.0, 0.0], &mut phi);
        assert!((phi[0] - 1.0).abs() < 1e-14, "phi0 at v0 = {}", phi[0]);
        for i in 1..10 { assert!(phi[i].abs() < 1e-14, "phi[{i}] at v0 = {}", phi[i]); }

        // φ₁ = 1 at vertex 1 (1,0)
        TriP3.eval_basis(&[1.0, 0.0], &mut phi);
        assert!((phi[1] - 1.0).abs() < 1e-14, "phi1 at v1 = {}", phi[1]);
        for i in (0..10).filter(|&j| j != 1) {
            assert!(phi[i].abs() < 1e-14, "phi[{i}] at v1 = {}", phi[i]);
        }

        // φ₂ = 1 at vertex 2 (0,1)
        TriP3.eval_basis(&[0.0, 1.0], &mut phi);
        assert!((phi[2] - 1.0).abs() < 1e-14, "phi2 at v2 = {}", phi[2]);
        for i in (0..10).filter(|&j| j != 2) {
            assert!(phi[i].abs() < 1e-14, "phi[{i}] at v2 = {}", phi[i]);
        }
    }

    #[test]
    fn tri_p3_edge_dofs() {
        let mut phi = vec![0.0_f64; 10];
        let t = 1.0_f64 / 3.0;
        let t2 = 2.0_f64 / 3.0;

        // φ₃ = 1 at (1/3, 0) — edge v0→v1 near v0
        TriP3.eval_basis(&[t, 0.0], &mut phi);
        assert!((phi[3] - 1.0).abs() < 1e-14, "phi3 at (1/3,0) = {}", phi[3]);
        for i in (0..10).filter(|&j| j != 3) {
            assert!(phi[i].abs() < 1e-14, "phi[{i}] at (1/3,0) = {}", phi[i]);
        }

        // φ₄ = 1 at (2/3, 0) — edge v0→v1 near v1
        TriP3.eval_basis(&[t2, 0.0], &mut phi);
        assert!((phi[4] - 1.0).abs() < 1e-14, "phi4 at (2/3,0) = {}", phi[4]);
        for i in (0..10).filter(|&j| j != 4) {
            assert!(phi[i].abs() < 1e-14, "phi[{i}] at (2/3,0) = {}", phi[i]);
        }

        // φ₅ = 1 at (2/3, 1/3) — edge v1→v2 near v1
        TriP3.eval_basis(&[t2, t], &mut phi);
        assert!((phi[5] - 1.0).abs() < 1e-14, "phi5 at (2/3,1/3) = {}", phi[5]);
        for i in (0..10).filter(|&j| j != 5) {
            assert!(phi[i].abs() < 1e-14, "phi[{i}] at (2/3,1/3) = {}", phi[i]);
        }

        // φ₉ = 1 at (1/3, 1/3) — bubble
        TriP3.eval_basis(&[t, t], &mut phi);
        assert!((phi[9] - 1.0).abs() < 1e-14, "phi9 (bubble) at centroid = {}", phi[9]);
        for i in (0..10).filter(|&j| j != 9) {
            assert!(phi[i].abs() < 1e-14, "phi[{i}] at centroid = {}", phi[i]);
        }
    }

    #[test]
    fn tri_p1_vertex_dofs() {
        let mut phi = vec![0.0; 3];
        TriP1.eval_basis(&[0.0, 0.0], &mut phi);
        assert!((phi[0] - 1.0).abs() < 1e-14);
        assert!(phi[1].abs() < 1e-14);
        assert!(phi[2].abs() < 1e-14);

        TriP1.eval_basis(&[1.0, 0.0], &mut phi);
        assert!(phi[0].abs() < 1e-14);
        assert!((phi[1] - 1.0).abs() < 1e-14);
        assert!(phi[2].abs() < 1e-14);
    }

    #[test]
    fn tri_p2_vertex_and_edge_dofs() {
        let mut phi = vec![0.0; 6];
        // At vertex 0: φ₀=1, rest=0
        TriP2.eval_basis(&[0.0, 0.0], &mut phi);
        assert!((phi[0] - 1.0).abs() < 1e-14);
        for i in 1..6 { assert!(phi[i].abs() < 1e-14, "phi[{i}]={}", phi[i]); }
        // At edge midpoint (0.5, 0): φ₃=1, rest=0
        TriP2.eval_basis(&[0.5, 0.0], &mut phi);
        for i in [0, 1, 2, 4, 5] { assert!(phi[i].abs() < 1e-14, "phi[{i}]={}", phi[i]); }
        assert!((phi[3] - 1.0).abs() < 1e-14);
    }

    // ── TriP4 ─────────────────────────────────────────────────────────────

    #[test] fn tri_p4_pou()       { check_pou(&TriP4); }
    #[test] fn tri_p4_grad_zero() { check_grad_zero(&TriP4); }
    #[test] fn tri_p4_n_dofs()    { assert_eq!(TriP4.n_dofs(), 15); }

    #[test]
    fn tri_p4_nodal_interp() {
        let coords = TriP4.dof_coords();
        let mut phi = vec![0.0_f64; 15];
        for (i, coord) in coords.iter().enumerate() {
            TriP4.eval_basis(&[coord[0], coord[1]], &mut phi);
            for j in 0..15 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((phi[j] - expected).abs() < 1e-12,
                    "TriP4 node {i}, basis {j}: expected {expected}, got {}", phi[j]);
            }
        }
    }

    #[test]
    fn tri_p4_matches_tri_pk() {
        use crate::lagrange::factory::TriPk;
        assert_eq!(TriP4.n_dofs(), TriPk::new(4).n_dofs());
    }

    #[test]
    fn tri_p4_gradient_fd() {
        let h = 1e-7;
        let n = 15;
        let (mut vc, mut vx, mut vy, mut grads) = (vec![0.0;n], vec![0.0;n], vec![0.0;n], vec![0.0;n*2]);
        for &(x, y) in &[(0.15, 0.2), (0.3, 0.3), (0.1, 0.6)] {
            if x + y > 0.9 { continue; }
            TriP4.eval_basis(&[x, y], &mut vc);
            TriP4.eval_basis(&[x+h, y], &mut vx);
            TriP4.eval_basis(&[x, y+h], &mut vy);
            TriP4.eval_grad_basis(&[x, y], &mut grads);
            for i in 0..n {
                let fd_x = (vx[i] - vc[i]) / h;
                let fd_y = (vy[i] - vc[i]) / h;
                assert!((grads[i*2] - fd_x).abs() < 1e-5, "({x},{y}) i={i} gx");
                assert!((grads[i*2+1] - fd_y).abs() < 1e-5, "({x},{y}) i={i} gy");
            }
        }
    }

    // ── TriP5 ─────────────────────────────────────────────────────────────

    #[test] fn tri_p5_pou()       { check_pou(&TriP5); }
    #[test] fn tri_p5_grad_zero() { check_grad_zero(&TriP5); }
    #[test] fn tri_p5_n_dofs()    { assert_eq!(TriP5.n_dofs(), 21); }

    #[test]
    fn tri_p5_nodal_interp() {
        let coords = TriP5.dof_coords();
        let mut phi = vec![0.0_f64; 21];
        for (i, coord) in coords.iter().enumerate() {
            TriP5.eval_basis(&[coord[0], coord[1]], &mut phi);
            for j in 0..21 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((phi[j] - expected).abs() < 1e-12,
                    "TriP5 node {i}, basis {j}: expected {expected}, got {}", phi[j]);
            }
        }
    }

    #[test]
    fn tri_p5_matches_tri_pk() {
        use crate::lagrange::factory::TriPk;
        assert_eq!(TriP5.n_dofs(), TriPk::new(5).n_dofs());
    }

    #[test]
    fn tri_p5_gradient_fd() {
        let h = 1e-7;
        let n = 21;
        let (mut vc, mut vx, mut vy, mut grads) = (vec![0.0;n], vec![0.0;n], vec![0.0;n], vec![0.0;n*2]);
        for &(x, y) in &[(0.12, 0.15), (0.25, 0.25), (0.08, 0.5)] {
            if x + y > 0.85 { continue; }
            TriP5.eval_basis(&[x, y], &mut vc);
            TriP5.eval_basis(&[x+h, y], &mut vx);
            TriP5.eval_basis(&[x, y+h], &mut vy);
            TriP5.eval_grad_basis(&[x, y], &mut grads);
            for i in 0..n {
                let fd_x = (vx[i] - vc[i]) / h;
                let fd_y = (vy[i] - vc[i]) / h;
                assert!((grads[i*2] - fd_x).abs() < 1e-5, "({x},{y}) i={i} gx");
                assert!((grads[i*2+1] - fd_y).abs() < 1e-5, "({x},{y}) i={i} gy");
            }
        }
    }

    // ── TriP6 ─────────────────────────────────────────────────────────────

    #[test] fn tri_p6_pou()       { check_pou(&TriP6); }
    #[test] fn tri_p6_grad_zero() { check_grad_zero(&TriP6); }
    #[test] fn tri_p6_n_dofs()    { assert_eq!(TriP6.n_dofs(), 28); }

    #[test]
    fn tri_p6_nodal_interp() {
        let coords = TriP6.dof_coords();
        let mut phi = vec![0.0_f64; 28];
        for (i, coord) in coords.iter().enumerate() {
            TriP6.eval_basis(&[coord[0], coord[1]], &mut phi);
            for j in 0..28 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((phi[j] - expected).abs() < 1e-12,
                    "TriP6 node {i}, basis {j}: expected {expected}, got {}", phi[j]);
            }
        }
    }

    #[test]
    fn tri_p6_matches_tri_pk() {
        use crate::lagrange::factory::TriPk;
        assert_eq!(TriP6.n_dofs(), TriPk::new(6).n_dofs());
    }

    #[test]
    fn tri_p6_gradient_fd() {
        let h = 1e-7;
        let n = 28;
        let (mut vc, mut vx, mut vy, mut grads) = (vec![0.0;n], vec![0.0;n], vec![0.0;n], vec![0.0;n*2]);
        for &(x, y) in &[(0.1, 0.12), (0.2, 0.2), (0.07, 0.4)] {
            if x + y > 0.8 { continue; }
            TriP6.eval_basis(&[x, y], &mut vc);
            TriP6.eval_basis(&[x+h, y], &mut vx);
            TriP6.eval_basis(&[x, y+h], &mut vy);
            TriP6.eval_grad_basis(&[x, y], &mut grads);
            for i in 0..n {
                let fd_x = (vx[i] - vc[i]) / h;
                let fd_y = (vy[i] - vc[i]) / h;
                assert!((grads[i*2] - fd_x).abs() < 5e-5, "({x},{y}) i={i} gx");
                assert!((grads[i*2+1] - fd_y).abs() < 5e-5, "({x},{y}) i={i} gy");
            }
        }
    }

    // ── TriP7 ─────────────────────────────────────────────────────────────

    #[test] fn tri_p7_pou()       { check_pou(&TriP7); }
    #[test] fn tri_p7_grad_zero() { check_grad_zero(&TriP7); }
    #[test] fn tri_p7_n_dofs()    { assert_eq!(TriP7.n_dofs(), 36); }

    #[test]
    fn tri_p7_nodal_interp() {
        let coords = TriP7.dof_coords();
        let mut phi = vec![0.0_f64; 36];
        for (i, coord) in coords.iter().enumerate() {
            TriP7.eval_basis(&[coord[0], coord[1]], &mut phi);
            for j in 0..36 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((phi[j] - expected).abs() < 1e-12,
                    "TriP7 node {i}, basis {j}: expected {expected}, got {}", phi[j]);
            }
        }
    }

    #[test]
    fn tri_p7_matches_tri_pk() {
        use crate::lagrange::factory::TriPk;
        assert_eq!(TriP7.n_dofs(), TriPk::new(7).n_dofs());
    }

    #[test]
    fn tri_p7_gradient_fd() {
        let h = 1e-7;
        let n = 36;
        let (mut vc, mut vx, mut vy, mut grads) = (vec![0.0;n], vec![0.0;n], vec![0.0;n], vec![0.0;n*2]);
        for &(x, y) in &[(0.08, 0.1), (0.15, 0.15), (0.05, 0.3)] {
            if x + y > 0.75 { continue; }
            TriP7.eval_basis(&[x, y], &mut vc);
            TriP7.eval_basis(&[x+h, y], &mut vx);
            TriP7.eval_basis(&[x, y+h], &mut vy);
            TriP7.eval_grad_basis(&[x, y], &mut grads);
            for i in 0..n {
                let fd_x = (vx[i] - vc[i]) / h;
                let fd_y = (vy[i] - vc[i]) / h;
                assert!((grads[i*2] - fd_x).abs() < 5e-5, "({x},{y}) i={i} gx");
                assert!((grads[i*2+1] - fd_y).abs() < 5e-5, "({x},{y}) i={i} gy");
            }
        }
    }

    // ── TriP8 ─────────────────────────────────────────────────────────────

    #[test] fn tri_p8_pou()       { check_pou(&TriP8); }
    #[test] fn tri_p8_grad_zero() { check_grad_zero(&TriP8); }
    #[test] fn tri_p8_n_dofs()    { assert_eq!(TriP8.n_dofs(), 45); }

    #[test]
    fn tri_p8_nodal_interp() {
        let coords = TriP8.dof_coords();
        let mut phi = vec![0.0_f64; 45];
        for (i, coord) in coords.iter().enumerate() {
            TriP8.eval_basis(&[coord[0], coord[1]], &mut phi);
            for j in 0..45 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((phi[j] - expected).abs() < 1e-12,
                    "TriP8 node {i}, basis {j}: expected {expected}, got {}", phi[j]);
            }
        }
    }

    #[test]
    fn tri_p8_matches_tri_pk() {
        use crate::lagrange::factory::TriPk;
        assert_eq!(TriP8.n_dofs(), TriPk::new(8).n_dofs());
    }

    #[test]
    fn tri_p8_gradient_fd() {
        let h = 1e-7;
        let n = 45;
        let (mut vc, mut vx, mut vy, mut grads) = (vec![0.0;n], vec![0.0;n], vec![0.0;n], vec![0.0;n*2]);
        for &(x, y) in &[(0.06, 0.08), (0.12, 0.12), (0.04, 0.25)] {
            if x + y > 0.7 { continue; }
            TriP8.eval_basis(&[x, y], &mut vc);
            TriP8.eval_basis(&[x+h, y], &mut vx);
            TriP8.eval_basis(&[x, y+h], &mut vy);
            TriP8.eval_grad_basis(&[x, y], &mut grads);
            for i in 0..n {
                let fd_x = (vx[i] - vc[i]) / h;
                let fd_y = (vy[i] - vc[i]) / h;
                assert!((grads[i*2] - fd_x).abs() < 1e-4, "({x},{y}) i={i} gx");
                assert!((grads[i*2+1] - fd_y).abs() < 1e-4, "({x},{y}) i={i} gy");
            }
        }
    }

    // ── TriP9 / TriP10 ────────────────────────────────────────────────────

    #[test] fn tri_p9_n_dofs() { assert_eq!(TriP9.n_dofs(), 55); }
    #[test] fn tri_p10_n_dofs() { assert_eq!(TriP10.n_dofs(), 66); }
    #[test] fn tri_p9_pou() { check_pou(&TriP9); }
    #[test] fn tri_p10_pou() { check_pou(&TriP10); }
    #[test] fn tri_p9_grad_zero() { check_grad_zero(&TriP9); }
    #[test] fn tri_p10_grad_zero() { check_grad_zero(&TriP10); }

    #[test]
    fn tri_p9_nodal_interp() {
        let coords = TriP9.dof_coords();
        let mut phi = vec![0.0; 55];
        for (i, c) in coords.iter().enumerate() {
            TriP9.eval_basis(&[c[0], c[1]], &mut phi);
            for j in 0..55 { assert!((phi[j] - if i==j {1.0} else {0.0}).abs() < 1e-12); }
        }
    }

    #[test]
    fn tri_p10_nodal_interp() {
        let coords = TriP10.dof_coords();
        let mut phi = vec![0.0; 66];
        for (i, c) in coords.iter().enumerate() {
            TriP10.eval_basis(&[c[0], c[1]], &mut phi);
            for j in 0..66 { assert!((phi[j] - if i==j {1.0} else {0.0}).abs() < 1e-12); }
        }
    }
}
