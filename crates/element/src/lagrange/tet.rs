//! Lagrange elements on the reference tetrahedron `(0,0,0),(1,0,0),(0,1,0),(0,0,1)`.
//!
//! Barycentric coordinates: λ₁=1−ξ−η−ζ, λ₂=ξ, λ₃=η, λ₄=ζ

use crate::quadrature::tet_rule;
use crate::reference::{QuadratureRule, ReferenceElement};

// ─── P1 ───────────────────────────────────────────────────────────────────────

/// Linear Lagrange element on the reference tetrahedron — 4 DOFs at vertices.
///
/// Basis:
/// - φ₀ = 1−ξ−η−ζ  (vertex (0,0,0))
/// - φ₁ = ξ          (vertex (1,0,0))
/// - φ₂ = η          (vertex (0,1,0))
/// - φ₃ = ζ          (vertex (0,0,1))
pub struct TetP1;

impl ReferenceElement for TetP1 {
    fn dim(&self)    -> u8    { 3 }
    fn order(&self)  -> u8    { 1 }
    fn n_dofs(&self) -> usize  { 4 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        values[0] = 1.0 - x - y - z;
        values[1] = x;
        values[2] = y;
        values[3] = z;
    }

    fn eval_grad_basis(&self, _xi: &[f64], grads: &mut [f64]) {
        // row-major [4×3]: grads[i*3 + j]
        // ∇φ₀ = (-1,-1,-1)
        grads[0]  = -1.0;  grads[1]  = -1.0;  grads[2]  = -1.0;
        // ∇φ₁ = (1,0,0)
        grads[3]  =  1.0;  grads[4]  =  0.0;  grads[5]  =  0.0;
        // ∇φ₂ = (0,1,0)
        grads[6]  =  0.0;  grads[7]  =  1.0;  grads[8]  =  0.0;
        // ∇φ₃ = (0,0,1)
        grads[9]  =  0.0;  grads[10] =  0.0;  grads[11] =  1.0;
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { tet_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]
    }
}

// ─── P2 ───────────────────────────────────────────────────────────────────────

/// Quadratic Lagrange element on the reference tetrahedron — 10 DOFs.
///
/// Barycentric coordinates: λ₁=1−ξ−η−ζ, λ₂=ξ, λ₃=η, λ₄=ζ
///
/// DOF ordering:
/// - 0: vertex (0,0,0)   — φ₀ = λ₁(2λ₁−1)
/// - 1: vertex (1,0,0)   — φ₁ = λ₂(2λ₂−1)
/// - 2: vertex (0,1,0)   — φ₂ = λ₃(2λ₃−1)
/// - 3: vertex (0,0,1)   — φ₃ = λ₄(2λ₄−1)
/// - 4: edge midpoint (1/2,0,0)   — φ₄ = 4λ₁λ₂
/// - 5: edge midpoint (0,1/2,0)   — φ₅ = 4λ₁λ₃
/// - 6: edge midpoint (0,0,1/2)   — φ₆ = 4λ₁λ₄
/// - 7: edge midpoint (1/2,1/2,0) — φ₇ = 4λ₂λ₃
/// - 8: edge midpoint (1/2,0,1/2) — φ₈ = 4λ₂λ₄
/// - 9: edge midpoint (0,1/2,1/2) — φ₉ = 4λ₃λ₄
pub struct TetP2;

impl ReferenceElement for TetP2 {
    fn dim(&self)    -> u8    { 3 }
    fn order(&self)  -> u8    { 2 }
    fn n_dofs(&self) -> usize  { 10 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let l1 = 1.0 - x - y - z;
        let l2 = x;
        let l3 = y;
        let l4 = z;
        // Vertex DOFs
        values[0] = l1 * (2.0 * l1 - 1.0);
        values[1] = l2 * (2.0 * l2 - 1.0);
        values[2] = l3 * (2.0 * l3 - 1.0);
        values[3] = l4 * (2.0 * l4 - 1.0);
        // Edge DOFs
        values[4] = 4.0 * l1 * l2;
        values[5] = 4.0 * l1 * l3;
        values[6] = 4.0 * l1 * l4;
        values[7] = 4.0 * l2 * l3;
        values[8] = 4.0 * l2 * l4;
        values[9] = 4.0 * l3 * l4;
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        // grads layout: row-major [10×3], grads[i*3 + j]
        // ∂λ₁/∂ξ=-1, ∂λ₁/∂η=-1, ∂λ₁/∂ζ=-1
        // ∂λ₂/∂ξ= 1, ∂λ₂/∂η= 0, ∂λ₂/∂ζ= 0
        // ∂λ₃/∂ξ= 0, ∂λ₃/∂η= 1, ∂λ₃/∂ζ= 0
        // ∂λ₄/∂ξ= 0, ∂λ₄/∂η= 0, ∂λ₄/∂ζ= 1
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let l1 = 1.0 - x - y - z;
        let l2 = x;
        let l3 = y;
        let l4 = z;
        // φ₀ = l1(2l1-1): ∂/∂λ₁ = 4l1-1; all partials = (4l1-1)*(-1)
        let d0 = 4.0 * l1 - 1.0;
        grads[0] = -d0;  grads[1] = -d0;  grads[2] = -d0;
        // φ₁ = l2(2l2-1): ∂/∂λ₂ = 4l2-1; ξ only
        let d1 = 4.0 * l2 - 1.0;
        grads[3] = d1;   grads[4] = 0.0;  grads[5] = 0.0;
        // φ₂ = l3(2l3-1): η only
        let d2 = 4.0 * l3 - 1.0;
        grads[6] = 0.0;  grads[7] = d2;   grads[8] = 0.0;
        // φ₃ = l4(2l4-1): ζ only
        let d3 = 4.0 * l4 - 1.0;
        grads[9] = 0.0;  grads[10] = 0.0; grads[11] = d3;
        // φ₄ = 4l1l2: ∂/∂ξ=4(l1-l2), ∂/∂η=-4l2, ∂/∂ζ=-4l2
        grads[12] = 4.0 * (l1 - l2);   grads[13] = -4.0 * l2;          grads[14] = -4.0 * l2;
        // φ₅ = 4l1l3: ∂/∂ξ=-4l3, ∂/∂η=4(l1-l3), ∂/∂ζ=-4l3
        grads[15] = -4.0 * l3;          grads[16] = 4.0 * (l1 - l3);   grads[17] = -4.0 * l3;
        // φ₆ = 4l1l4: ∂/∂ξ=-4l4, ∂/∂η=-4l4, ∂/∂ζ=4(l1-l4)
        grads[18] = -4.0 * l4;          grads[19] = -4.0 * l4;          grads[20] = 4.0 * (l1 - l4);
        // φ₇ = 4l2l3: ∂/∂ξ=4l3, ∂/∂η=4l2, ∂/∂ζ=0
        grads[21] = 4.0 * l3;           grads[22] = 4.0 * l2;           grads[23] = 0.0;
        // φ₈ = 4l2l4: ∂/∂ξ=4l4, ∂/∂η=0, ∂/∂ζ=4l2
        grads[24] = 4.0 * l4;           grads[25] = 0.0;                grads[26] = 4.0 * l2;
        // φ₉ = 4l3l4: ∂/∂ξ=0, ∂/∂η=4l4, ∂/∂ζ=4l3
        grads[27] = 0.0;                grads[28] = 4.0 * l4;           grads[29] = 4.0 * l3;
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { tet_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let t = 1.0 / 3.0; let t2 = 2.0 / 3.0;
        vec![
            vec![0.0,0.0,0.0], vec![1.0,0.0,0.0], vec![0.0,1.0,0.0], vec![0.0,0.0,1.0],
            vec![t,0.0,0.0], vec![t2,0.0,0.0],
            vec![0.0,t,0.0], vec![0.0,t2,0.0],
            vec![0.0,0.0,t], vec![0.0,0.0,t2],
            vec![t2,t,0.0], vec![t,t2,0.0],
            vec![t2,0.0,t], vec![t,0.0,t2],
            vec![0.0,t2,t], vec![0.0,t,t2],
            vec![t,t,0.0], vec![t,0.0,t], vec![0.0,t,t],
            vec![t,t,t],
        ]
    }
}

// ─── Pk (fixed-order) — using rising-factorial basis ─────────────────────────

use std::sync::OnceLock;

/// Generate IJKL indices for tet order p, in the same order as TetPk factory.
fn tet_ijkl(p: usize) -> Vec<(usize, usize, usize, usize)> {
    let mut v = Vec::with_capacity((p+1)*(p+2)*(p+3)/6);
    v.push((0,0,0,p)); v.push((p,0,0,0)); v.push((0,p,0,0)); v.push((0,0,p,0));
    if p == 1 { return v; }
    for k in 1..p { v.push((k,0,0,p-k)); }
    for k in 1..p { v.push((0,k,0,p-k)); }
    for k in 1..p { v.push((0,0,k,p-k)); }
    for k in 1..p { v.push((p-k,k,0,0)); }
    for k in 1..p { v.push((p-k,0,k,0)); }
    for k in 1..p { v.push((0,p-k,k,0)); }
    for j in 1..=p.saturating_sub(2) { for i in 1..=p-1-j { v.push((i,j,0,p-i-j)); }}
    for k in 1..=p.saturating_sub(2) { for i in 1..=p-1-k { v.push((i,0,k,p-i-k)); }}
    for k in 1..=p.saturating_sub(2) { for j in 1..=p-1-k { v.push((0,j,k,p-j-k)); }}
    for k in 1..=p.saturating_sub(2) { for j in 1..=p-1-k { v.push((p-j-k,j,k,0)); }}
    for k in 1..=p.saturating_sub(3) { for j in 1..=p-2-k { for i in 1..=p-1-j-k { v.push((i,j,k,p-i-j-k)); }}}
    debug_assert_eq!(v.len(), (p+1)*(p+2)*(p+3)/6);
    v
}

use super::factory::{rising_val, rising_deriv};

macro_rules! impl_tet_pk_fixed {
    ($name:ident, $p:literal) => {
        impl ReferenceElement for $name {
            fn dim(&self) -> u8 { 3 }
            fn order(&self) -> u8 { $p }
            fn n_dofs(&self) -> usize { ($p as usize + 1) * ($p as usize + 2) * ($p as usize + 3) / 6 }
            fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
                static IJKL: OnceLock<Vec<(usize,usize,usize,usize)>> = OnceLock::new();
                let ijkl = IJKL.get_or_init(|| tet_ijkl($p));
                let pf = $p as f64; let t0=pf*xi[0]; let t1=pf*xi[1]; let t2=pf*xi[2]; let t3=pf*(1.0-xi[0]-xi[1]-xi[2]);
                for (i,&(a,b,c,d)) in ijkl.iter().enumerate() { values[i]=rising_val(a,t0)*rising_val(b,t1)*rising_val(c,t2)*rising_val(d,t3); }
            }
            fn eval_grad_basis(&self, xi:&[f64], grads:&mut[f64]) {
                static IJKL: OnceLock<Vec<(usize,usize,usize,usize)>> = OnceLock::new();
                let ijkl = IJKL.get_or_init(|| tet_ijkl($p));
                let pf=$p as f64; let t0=pf*xi[0]; let t1=pf*xi[1]; let t2=pf*xi[2]; let t3=pf*(1.0-xi[0]-xi[1]-xi[2]);
                for (i,&(a,b,c,d)) in ijkl.iter().enumerate() {
                    let (va,vb,vc,vd)=(rising_val(a,t0),rising_val(b,t1),rising_val(c,t2),rising_val(d,t3));
                    grads[i*3]=pf*(rising_deriv(a,t0)*vb*vc*vd - va*vb*vc*rising_deriv(d,t3));
                    grads[i*3+1]=pf*(va*rising_deriv(b,t1)*vc*vd - va*vb*vc*rising_deriv(d,t3));
                    grads[i*3+2]=pf*(va*vb*rising_deriv(c,t2)*vd - va*vb*vc*rising_deriv(d,t3));
                }
            }
            fn quadrature(&self, o: u8) -> QuadratureRule { tet_rule(o) }
            fn dof_coords(&self) -> Vec<Vec<f64>> {
                static IJKL: OnceLock<Vec<(usize,usize,usize,usize)>> = OnceLock::new();
                let ijkl = IJKL.get_or_init(|| tet_ijkl($p));
                let pf = $p as f64; ijkl.iter().map(|&(a,b,c,_)| vec![a as f64/pf, b as f64/pf, c as f64/pf]).collect()
            }
        }
    };
}

/// Quartic Lagrange element on the reference tetrahedron — 35 DOFs.
pub struct TetP4;
impl_tet_pk_fixed!(TetP4, 4);

/// Quintic Lagrange element on the reference tetrahedron — 56 DOFs.
pub struct TetP5;
impl_tet_pk_fixed!(TetP5, 5);

/// Sextic Lagrange element on the reference tetrahedron — 84 DOFs.
pub struct TetP6;
impl_tet_pk_fixed!(TetP6, 6);

// ─── P3 ───────────────────────────────────────────────────────────────────────

/// Cubic Lagrange element on the reference tetrahedron — 20 DOFs.
///
/// Barycentric coordinates: λ₁=1−ξ−η−ζ, λ₂=ξ, λ₃=η, λ₄=ζ
///
/// DOF ordering:
/// - 0: vertex (0,0,0)          — φ₀ = ½λ₁(3λ₁−1)(3λ₁−2)
/// - 1: vertex (1,0,0)          — φ₁ = ½λ₂(3λ₂−1)(3λ₂−2)
/// - 2: vertex (0,1,0)          — φ₂ = ½λ₃(3λ₃−1)(3λ₃−2)
/// - 3: vertex (0,0,1)          — φ₃ = ½λ₄(3λ₄−1)(3λ₄−2)
/// - 4,5:  edge(v0→v1) at 1/3,2/3 — 9/2·λ₁λ₂(3λ₁−1), 9/2·λ₁λ₂(3λ₂−1)
/// - 6,7:  edge(v0→v2) at 1/3,2/3 — 9/2·λ₁λ₃(3λ₁−1), 9/2·λ₁λ₃(3λ₃−1)
/// - 8,9:  edge(v0→v3) at 1/3,2/3 — 9/2·λ₁λ₄(3λ₁−1), 9/2·λ₁λ₄(3λ₄−1)
/// - 10,11: edge(v1→v2) at 1/3,2/3 — 9/2·λ₂λ₃(3λ₂−1), 9/2·λ₂λ₃(3λ₃−1)
/// - 12,13: edge(v1→v3) at 1/3,2/3 — 9/2·λ₂λ₄(3λ₂−1), 9/2·λ₂λ₄(3λ₄−1)
/// - 14,15: edge(v2→v3) at 1/3,2/3 — 9/2·λ₃λ₄(3λ₃−1), 9/2·λ₃λ₄(3λ₄−1)
/// - 16: face(v0,v1,v2) — 27λ₁λ₂λ₃
/// - 17: face(v0,v1,v3) — 27λ₁λ₂λ₄
/// - 18: face(v0,v2,v3) — 27λ₁λ₃λ₄
/// - 19: face(v1,v2,v3) — 27λ₂λ₃λ₄
pub struct TetP3;

impl ReferenceElement for TetP3 {
    fn dim(&self)    -> u8     { 3 }
    fn order(&self)  -> u8     { 3 }
    fn n_dofs(&self) -> usize  { 20 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let l1 = 1.0 - x - y - z;
        let l2 = x;
        let l3 = y;
        let l4 = z;

        // Vertex DOFs: ½λ(3λ−1)(3λ−2)
        values[0]  = 0.5 * l1 * (3.0*l1 - 1.0) * (3.0*l1 - 2.0);
        values[1]  = 0.5 * l2 * (3.0*l2 - 1.0) * (3.0*l2 - 2.0);
        values[2]  = 0.5 * l3 * (3.0*l3 - 1.0) * (3.0*l3 - 2.0);
        values[3]  = 0.5 * l4 * (3.0*l4 - 1.0) * (3.0*l4 - 2.0);

        // Edge DOFs: 9/2·λᵢλⱼ(3λᵢ−1) and 9/2·λᵢλⱼ(3λⱼ−1)
        // edge v0→v1
        values[4]  = 4.5 * l1 * l2 * (3.0*l1 - 1.0);
        values[5]  = 4.5 * l1 * l2 * (3.0*l2 - 1.0);
        // edge v0→v2
        values[6]  = 4.5 * l1 * l3 * (3.0*l1 - 1.0);
        values[7]  = 4.5 * l1 * l3 * (3.0*l3 - 1.0);
        // edge v0→v3
        values[8]  = 4.5 * l1 * l4 * (3.0*l1 - 1.0);
        values[9]  = 4.5 * l1 * l4 * (3.0*l4 - 1.0);
        // edge v1→v2
        values[10] = 4.5 * l2 * l3 * (3.0*l2 - 1.0);
        values[11] = 4.5 * l2 * l3 * (3.0*l3 - 1.0);
        // edge v1→v3
        values[12] = 4.5 * l2 * l4 * (3.0*l2 - 1.0);
        values[13] = 4.5 * l2 * l4 * (3.0*l4 - 1.0);
        // edge v2→v3
        values[14] = 4.5 * l3 * l4 * (3.0*l3 - 1.0);
        values[15] = 4.5 * l3 * l4 * (3.0*l4 - 1.0);

        // Face DOFs: 27λᵢλⱼλₖ
        values[16] = 27.0 * l1 * l2 * l3;   // face v0,v1,v2 (z=0)
        values[17] = 27.0 * l1 * l2 * l4;   // face v0,v1,v3 (y=0)
        values[18] = 27.0 * l1 * l3 * l4;   // face v0,v2,v3 (x=0)
        values[19] = 27.0 * l2 * l3 * l4;   // face v1,v2,v3 (l1=0)
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        // grads layout: row-major [20×3], grads[i*3 + j]
        // ∂λ₁ = (-1,-1,-1), ∂λ₂ = (1,0,0), ∂λ₃ = (0,1,0), ∂λ₄ = (0,0,1)
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let l1 = 1.0 - x - y - z;
        let l2 = x;
        let l3 = y;
        let l4 = z;

        // Helper: d/dxi [½λ(3λ-1)(3λ-2)] = ½(27λ²-18λ+2) × dλ
        macro_rules! dvert {
            ($l:expr) => { 0.5 * (27.0*$l*$l - 18.0*$l + 2.0) }
        }
        let (dv0, dv1, dv2, dv3) = (dvert!(l1), dvert!(l2), dvert!(l3), dvert!(l4));
        // φ₀: ∇ = dv0 × ∂λ₁ = dv0 × (-1,-1,-1)
        grads[0]  = -dv0; grads[1]  = -dv0; grads[2]  = -dv0;
        // φ₁: ∇ = dv1 × ∂λ₂ = dv1 × (1,0,0)
        grads[3]  =  dv1; grads[4]  =  0.0; grads[5]  =  0.0;
        // φ₂: ∇ = dv2 × ∂λ₃
        grads[6]  =  0.0; grads[7]  =  dv2; grads[8]  =  0.0;
        // φ₃: ∇ = dv3 × ∂λ₄
        grads[9]  =  0.0; grads[10] =  0.0; grads[11] =  dv3;

        // Edge DOFs: φ = c·λᵢ·λⱼ·f(λₖ) where c=9/2 and f is linear in one bary coord.
        // General formula: ∂/∂ξ [λᵢλⱼ(3λₖ-1)] = (∂λᵢ/∂ξ)λⱼ(3λₖ-1) + λᵢ(∂λⱼ/∂ξ)(3λₖ-1) + 3λᵢλⱼ(∂λₖ/∂ξ)
        // for DOF pair near vertex k=i first, k=j second.
        // Macro: edge_grad(li, lj; dli=(∂li/∂ξ,∂li/∂η,∂li/∂ζ), dlj=...; DOF: 9/2·li·lj·(3li-1))
        // We write out all 16 edge gradients explicitly.

        // φ₄ = 4.5·l1·l2·(3l1-1)
        // ∂/∂ξ = 4.5·[(-1)·l2·(3l1-1) + l1·1·(3l1-1) + l1·l2·3·(-1)]
        //       = 4.5·[(l1-l2)(3l1-1) - 3l1l2]
        // ∂/∂η = 4.5·[(-1)·l2·(3l1-1) + 0 + (-3)·l1·l2]
        //       = 4.5·[-l2(3l1-1) - 3l1l2] = 4.5·[-l2(3l1-1+3l1)] = 4.5·[-l2(6l1-1)]
        // ∂/∂ζ same as η (l1 and l2 don't depend on η vs ζ differently here)
        grads[12] = 4.5*((l1-l2)*(3.0*l1-1.0) - 3.0*l1*l2);
        grads[13] = 4.5*(-l2*(6.0*l1-1.0));
        grads[14] = 4.5*(-l2*(6.0*l1-1.0));

        // φ₅ = 4.5·l1·l2·(3l2-1)
        // ∂/∂ξ = 4.5·[(-1)·l2·(3l2-1) + l1·(3l2-1) + l1·l2·3]
        //       = 4.5·[(l1-l2)(3l2-1) + 3l1l2]
        // ∂/∂η = 4.5·[(-1)·l2·(3l2-1) + 0 + 0] = -4.5·l2(3l2-1)
        // ∂/∂ζ same
        grads[15] = 4.5*((l1-l2)*(3.0*l2-1.0) + 3.0*l1*l2);
        grads[16] = -4.5*l2*(3.0*l2-1.0);
        grads[17] = -4.5*l2*(3.0*l2-1.0);

        // φ₆ = 4.5·l1·l3·(3l1-1)
        // ∂/∂ξ = 4.5·[(-1)·l3·(3l1-1) + l1·0·(3l1-1) + l1·l3·(-3)]
        //       = 4.5·[-l3(3l1-1) - 3l1l3] = 4.5·[-l3(6l1-1)]
        // ∂/∂η = 4.5·[(-1)·l3·(3l1-1) + l1·1·(3l1-1) + l1·l3·(-3)]
        //       = 4.5·[(l1-l3)(3l1-1) - 3l1l3]
        // ∂/∂ζ = 4.5·[(-1)·l3·(3l1-1) + 0 + 0] = -4.5l3(3l1-1) ... wait same as ξ pattern
        // Actually ζ: ∂l1/∂ζ=-1, ∂l3/∂ζ=0
        // ∂/∂ζ = 4.5·[(-1)l3(3l1-1) + 0 + l1·l3·(-3)] = 4.5·[-l3(6l1-1)]
        grads[18] = 4.5*(-l3*(6.0*l1-1.0));
        grads[19] = 4.5*((l1-l3)*(3.0*l1-1.0) - 3.0*l1*l3);
        grads[20] = 4.5*(-l3*(6.0*l1-1.0));

        // φ₇ = 4.5·l1·l3·(3l3-1)
        // ∂/∂ξ = 4.5·[(-1)l3(3l3-1) + 0 + 0] = -4.5l3(3l3-1)
        // ∂/∂η = 4.5·[(-1)l3(3l3-1) + l1(3l3-1) + l1l3·3] = 4.5[(l1-l3)(3l3-1)+3l1l3]
        // ∂/∂ζ = -4.5l3(3l3-1)
        grads[21] = -4.5*l3*(3.0*l3-1.0);
        grads[22] = 4.5*((l1-l3)*(3.0*l3-1.0) + 3.0*l1*l3);
        grads[23] = -4.5*l3*(3.0*l3-1.0);

        // φ₈ = 4.5·l1·l4·(3l1-1)
        // ∂/∂ξ = 4.5·[(-1)l4(3l1-1) + 0 + l1l4(-3)] = 4.5[-l4(6l1-1)]
        // ∂/∂η = same = 4.5[-l4(6l1-1)]
        // ∂/∂ζ = 4.5·[(-1)l4(3l1-1) + l1·1·(3l1-1) + l1l4(-3)] = 4.5[(l1-l4)(3l1-1)-3l1l4]
        grads[24] = 4.5*(-l4*(6.0*l1-1.0));
        grads[25] = 4.5*(-l4*(6.0*l1-1.0));
        grads[26] = 4.5*((l1-l4)*(3.0*l1-1.0) - 3.0*l1*l4);

        // φ₉ = 4.5·l1·l4·(3l4-1)
        // ∂/∂ξ = 4.5·[(-1)l4(3l4-1) + 0 + 0] = -4.5l4(3l4-1)
        // ∂/∂η = same
        // ∂/∂ζ = 4.5·[(-1)l4(3l4-1) + l1(3l4-1) + l1l4·3] = 4.5[(l1-l4)(3l4-1)+3l1l4]
        grads[27] = -4.5*l4*(3.0*l4-1.0);
        grads[28] = -4.5*l4*(3.0*l4-1.0);
        grads[29] = 4.5*((l1-l4)*(3.0*l4-1.0) + 3.0*l1*l4);

        // φ₁₀ = 4.5·l2·l3·(3l2-1)
        // ∂/∂ξ = 4.5·[l3(3l2-1) + 0 + l2l3·3] = 4.5·l3(6l2-1)
        // ∂/∂η = 4.5·[0 + l2(3l2-1) + 0] = 4.5l2(3l2-1)
        // ∂/∂ζ = 0
        grads[30] = 4.5*l3*(6.0*l2-1.0);
        grads[31] = 4.5*l2*(3.0*l2-1.0);
        grads[32] = 0.0;

        // φ₁₁ = 4.5·l2·l3·(3l3-1)
        // ∂/∂ξ = 4.5·[l3(3l3-1) + 0 + 0] = 4.5l3(3l3-1)
        // ∂/∂η = 4.5·[0 + l2(3l3-1) + l2l3·3] = 4.5l2(6l3-1)
        // ∂/∂ζ = 0
        grads[33] = 4.5*l3*(3.0*l3-1.0);
        grads[34] = 4.5*l2*(6.0*l3-1.0);
        grads[35] = 0.0;

        // φ₁₂ = 4.5·l2·l4·(3l2-1)
        // ∂/∂ξ = 4.5·[l4(3l2-1) + 0 + l2l4·3] = 4.5l4(6l2-1)
        // ∂/∂η = 0
        // ∂/∂ζ = 4.5·[0 + l2(3l2-1) + 0] = 4.5l2(3l2-1)
        grads[36] = 4.5*l4*(6.0*l2-1.0);
        grads[37] = 0.0;
        grads[38] = 4.5*l2*(3.0*l2-1.0);

        // φ₁₃ = 4.5·l2·l4·(3l4-1)
        // ∂/∂ξ = 4.5·[l4(3l4-1) + 0 + 0] = 4.5l4(3l4-1)
        // ∂/∂η = 0
        // ∂/∂ζ = 4.5·[0 + l2(3l4-1) + l2l4·3] = 4.5l2(6l4-1)
        grads[39] = 4.5*l4*(3.0*l4-1.0);
        grads[40] = 0.0;
        grads[41] = 4.5*l2*(6.0*l4-1.0);

        // φ₁₄ = 4.5·l3·l4·(3l3-1)
        // ∂/∂ξ = 0
        // ∂/∂η = 4.5·[l4(3l3-1) + 0 + l3l4·3] = 4.5l4(6l3-1)
        // ∂/∂ζ = 4.5·[0 + l3(3l3-1) + 0] = 4.5l3(3l3-1)
        grads[42] = 0.0;
        grads[43] = 4.5*l4*(6.0*l3-1.0);
        grads[44] = 4.5*l3*(3.0*l3-1.0);

        // φ₁₅ = 4.5·l3·l4·(3l4-1)
        // ∂/∂ξ = 0
        // ∂/∂η = 4.5·[l4(3l4-1) + 0 + 0] = 4.5l4(3l4-1)
        // ∂/∂ζ = 4.5·[0 + l3(3l4-1) + l3l4·3] = 4.5l3(6l4-1)
        grads[45] = 0.0;
        grads[46] = 4.5*l4*(3.0*l4-1.0);
        grads[47] = 4.5*l3*(6.0*l4-1.0);

        // φ₁₆ = 27·l1·l2·l3
        // ∂/∂ξ = 27·[(-1)l2l3 + l1l3 + 0] = 27l3(l1-l2)
        // ∂/∂η = 27·[(-1)l2l3 + 0 + l1l2] = 27l2(l1-l3)
        // ∂/∂ζ = 27·[(-1)l2l3 + 0 + 0]   = -27l2l3
        grads[48] = 27.0*l3*(l1-l2);
        grads[49] = 27.0*l2*(l1-l3);
        grads[50] = -27.0*l2*l3;

        // φ₁₇ = 27·l1·l2·l4
        // ∂/∂ξ = 27·[(-1)l2l4 + l1l4 + 0] = 27l4(l1-l2)
        // ∂/∂η = 27·[(-1)l2l4 + 0 + 0]    = -27l2l4
        // ∂/∂ζ = 27·[(-1)l2l4 + 0 + l1l2] = 27l2(l1-l4)
        grads[51] = 27.0*l4*(l1-l2);
        grads[52] = -27.0*l2*l4;
        grads[53] = 27.0*l2*(l1-l4);

        // φ₁₈ = 27·l1·l3·l4
        // ∂/∂ξ = 27·[(-1)l3l4 + 0 + 0]    = -27l3l4
        // ∂/∂η = 27·[(-1)l3l4 + l1l4 + 0] = 27l4(l1-l3)
        // ∂/∂ζ = 27·[(-1)l3l4 + 0 + l1l3] = 27l3(l1-l4)
        grads[54] = -27.0*l3*l4;
        grads[55] = 27.0*l4*(l1-l3);
        grads[56] = 27.0*l3*(l1-l4);

        // φ₁₉ = 27·l2·l3·l4
        // ∂/∂ξ = 27·[l3l4 + 0 + 0]  = 27l3l4
        // ∂/∂η = 27·[0 + l2l4 + 0]  = 27l2l4
        // ∂/∂ζ = 27·[0 + 0 + l2l3]  = 27l2l3
        grads[57] = 27.0*l3*l4;
        grads[58] = 27.0*l2*l4;
        grads[59] = 27.0*l2*l3;
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { tet_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            // 4 vertices
            vec![0.0,       0.0,       0.0    ],
            vec![1.0,       0.0,       0.0    ],
            vec![0.0,       1.0,       0.0    ],
            vec![0.0,       0.0,       1.0    ],
            // edge v0→v1: DOF 4 near v0 (1/3,0,0), DOF 5 near v1 (2/3,0,0)
            vec![1.0/3.0,   0.0,       0.0    ],
            vec![2.0/3.0,   0.0,       0.0    ],
            // edge v0→v2: DOF 6 near v0 (0,1/3,0), DOF 7 near v2 (0,2/3,0)
            vec![0.0,       1.0/3.0,   0.0    ],
            vec![0.0,       2.0/3.0,   0.0    ],
            // edge v0→v3: DOF 8 near v0 (0,0,1/3), DOF 9 near v3 (0,0,2/3)
            vec![0.0,       0.0,       1.0/3.0],
            vec![0.0,       0.0,       2.0/3.0],
            // edge v1→v2: DOF 10 near v1 (2/3,1/3,0), DOF 11 near v2 (1/3,2/3,0)
            vec![2.0/3.0,   1.0/3.0,   0.0    ],
            vec![1.0/3.0,   2.0/3.0,   0.0    ],
            // edge v1→v3: DOF 12 near v1 (2/3,0,1/3), DOF 13 near v3 (1/3,0,2/3)
            vec![2.0/3.0,   0.0,       1.0/3.0],
            vec![1.0/3.0,   0.0,       2.0/3.0],
            // edge v2→v3: DOF 14 near v2 (0,2/3,1/3), DOF 15 near v3 (0,1/3,2/3)
            vec![0.0,       2.0/3.0,   1.0/3.0],
            vec![0.0,       1.0/3.0,   2.0/3.0],
            // 4 face centroids
            vec![1.0/3.0,   1.0/3.0,   0.0    ],  // DOF 16: face v0,v1,v2
            vec![1.0/3.0,   0.0,       1.0/3.0],  // DOF 17: face v0,v1,v3
            vec![0.0,       1.0/3.0,   1.0/3.0],  // DOF 18: face v0,v2,v3
            vec![1.0/3.0,   1.0/3.0,   1.0/3.0],  // DOF 19: face v1,v2,v3
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_pou(elem: &dyn ReferenceElement) {
        let rule = elem.quadrature(4);
        let mut phi = vec![0.0_f64; elem.n_dofs()];
        for pt in &rule.points {
            elem.eval_basis(pt, &mut phi);
            let s: f64 = phi.iter().sum();
            assert!((s - 1.0).abs() < 1e-13, "POU failed sum={s}");
        }
    }

    fn check_grad_zero(elem: &dyn ReferenceElement) {
        let dim = elem.dim() as usize;
        let rule = elem.quadrature(4);
        let mut g = vec![0.0_f64; elem.n_dofs() * dim];
        for pt in &rule.points {
            elem.eval_grad_basis(pt, &mut g);
            for d in 0..dim {
                let s: f64 = (0..elem.n_dofs()).map(|i| g[i * dim + d]).sum();
                assert!(s.abs() < 1e-13, "grad sum d={d} = {s}");
            }
        }
    }

    #[test] fn tet_p1_pou()       { check_pou(&TetP1); }
    #[test] fn tet_p1_grad_zero() { check_grad_zero(&TetP1); }

    #[test]
    fn tet_p1_vertex_dofs() {
        let mut phi = vec![0.0; 4];
        TetP1.eval_basis(&[0.0, 0.0, 0.0], &mut phi);
        assert!((phi[0] - 1.0).abs() < 1e-14);
        for i in 1..4 { assert!(phi[i].abs() < 1e-14); }

        TetP1.eval_basis(&[1.0, 0.0, 0.0], &mut phi);
        assert!(phi[0].abs() < 1e-14);
        assert!((phi[1] - 1.0).abs() < 1e-14);
    }

    #[test] fn tet_p2_pou()       { check_pou(&TetP2); }
    #[test] fn tet_p2_grad_zero() { check_grad_zero(&TetP2); }

    #[test]
    fn tet_p2_vertex_dofs() {
        let verts = [[0.0f64,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]];
        for (vi, vp) in verts.iter().enumerate() {
            let mut phi = vec![0.0; 10];
            TetP2.eval_basis(vp, &mut phi);
            assert!((phi[vi] - 1.0).abs() < 1e-14, "vertex {vi}: phi={}", phi[vi]);
            for j in 0..10 { if j != vi { assert!(phi[j].abs() < 1e-14, "vertex {vi}, phi[{j}]={}", phi[j]); } }
        }
    }

    #[test]
    fn tet_p2_edge_midpoint_dofs() {
        // Edge midpoints and their DOF indices (DOFs 4-9)
        let edges = [
            ([0.5f64, 0.0, 0.0], 4usize),  // λ₁λ₂ midpoint
            ([0.0, 0.5, 0.0],    5),
            ([0.0, 0.0, 0.5],    6),
            ([0.5, 0.5, 0.0],    7),
            ([0.5, 0.0, 0.5],    8),
            ([0.0, 0.5, 0.5],    9),
        ];
        for (pt, di) in &edges {
            let mut phi = vec![0.0; 10];
            TetP2.eval_basis(pt, &mut phi);
            assert!((phi[*di] - 1.0).abs() < 1e-14, "edge dof {di}: phi={}", phi[*di]);
            for j in 0..10 { if j != *di { assert!(phi[j].abs() < 1e-13, "edge dof {di}, phi[{j}]={}", phi[j]); } }
        }
    }

    #[test] fn tet_p3_pou()       { check_pou(&TetP3); }
    #[test] fn tet_p3_grad_zero() { check_grad_zero(&TetP3); }

    #[test]
    fn tet_p3_vertex_dofs() {
        let verts = [[0.0f64,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]];
        for (vi, vp) in verts.iter().enumerate() {
            let mut phi = vec![0.0; 20];
            TetP3.eval_basis(vp, &mut phi);
            assert!((phi[vi] - 1.0).abs() < 1e-13, "vertex {vi}: phi={}", phi[vi]);
            for j in 0..20 { if j != vi { assert!(phi[j].abs() < 1e-13, "vertex {vi}, phi[{j}]={}", phi[j]); } }
        }
    }

    #[test]
    fn tet_p3_edge_dofs() {
        let pts: [(usize, [f64; 3]); 12] = [
            (4,  [1.0/3.0, 0.0,     0.0    ]),
            (5,  [2.0/3.0, 0.0,     0.0    ]),
            (6,  [0.0,     1.0/3.0, 0.0    ]),
            (7,  [0.0,     2.0/3.0, 0.0    ]),
            (8,  [0.0,     0.0,     1.0/3.0]),
            (9,  [0.0,     0.0,     2.0/3.0]),
            (10, [2.0/3.0, 1.0/3.0, 0.0    ]),
            (11, [1.0/3.0, 2.0/3.0, 0.0    ]),
            (12, [2.0/3.0, 0.0,     1.0/3.0]),
            (13, [1.0/3.0, 0.0,     2.0/3.0]),
            (14, [0.0,     2.0/3.0, 1.0/3.0]),
            (15, [0.0,     1.0/3.0, 2.0/3.0]),
        ];
        for (di, pt) in &pts {
            let mut phi = vec![0.0; 20];
            TetP3.eval_basis(pt, &mut phi);
            assert!((phi[*di] - 1.0).abs() < 1e-12, "edge dof {di}: phi={}", phi[*di]);
            for j in 0..20 { if j != *di { assert!(phi[j].abs() < 1e-12, "edge dof {di}: phi[{j}]={}", phi[j]); } }
        }
    }

    #[test]
    fn tet_p3_face_dofs() {
        let pts: [(usize, [f64; 3]); 4] = [
            (16, [1.0/3.0, 1.0/3.0, 0.0    ]),
            (17, [1.0/3.0, 0.0,     1.0/3.0]),
            (18, [0.0,     1.0/3.0, 1.0/3.0]),
            (19, [1.0/3.0, 1.0/3.0, 1.0/3.0]),
        ];
        for (di, pt) in &pts {
            let mut phi = vec![0.0; 20];
            TetP3.eval_basis(pt, &mut phi);
            assert!((phi[*di] - 1.0).abs() < 1e-11, "face dof {di}: phi={}", phi[*di]);
            for j in 0..20 { if j != *di { assert!(phi[j].abs() < 1e-11, "face dof {di}: phi[{j}]={}", phi[j]); } }
        }
    }

    // ── TetP4 ─────────────────────────────────────────────────────────────

    #[test] fn tet_p4_pou() { check_pou(&TetP4); }
    #[test] fn tet_p4_grad_zero() { check_grad_zero(&TetP4); }
    #[test] fn tet_p4_n_dofs() { assert_eq!(TetP4.n_dofs(), 35); }

    #[test]
    fn tet_p4_nodal_interp() {
        let coords = TetP4.dof_coords(); let n = 35; let mut phi = vec![0.0; n];
        for (i,c) in coords.iter().enumerate() { TetP4.eval_basis(&[c[0],c[1],c[2]], &mut phi);
            for j in 0..n { assert!((phi[j]-if i==j{1.0}else{0.0}).abs()<1e-11); }
        }
    }

    #[test]
    fn tet_p4_matches_tet_pk() { use crate::lagrange::factory::TetPk; assert_eq!(TetPk::new(4).n_dofs(), 35); }

    // ── TetP5 ─────────────────────────────────────────────────────────────

    #[test] fn tet_p5_pou() { check_pou(&TetP5); }
    #[test] fn tet_p5_grad_zero() { check_grad_zero(&TetP5); }
    #[test] fn tet_p5_n_dofs() { assert_eq!(TetP5.n_dofs(), 56); }

    #[test]
    fn tet_p5_nodal_interp() {
        let coords = TetP5.dof_coords(); let n = 56; let mut phi = vec![0.0; n];
        for (i,c) in coords.iter().enumerate() { TetP5.eval_basis(&[c[0],c[1],c[2]], &mut phi);
            for j in 0..n { assert!((phi[j]-if i==j{1.0}else{0.0}).abs()<1e-11); }
        }
    }

    #[test]
    fn tet_p5_matches_tet_pk() { use crate::lagrange::factory::TetPk; assert_eq!(TetPk::new(5).n_dofs(), 56); }

    // ── TetP6 ─────────────────────────────────────────────────────────────

    #[test] fn tet_p6_pou() { check_pou(&TetP6); }
    #[test] fn tet_p6_grad_zero() { check_grad_zero(&TetP6); }
    #[test] fn tet_p6_n_dofs() { assert_eq!(TetP6.n_dofs(), 84); }

    #[test]
    fn tet_p6_nodal_interp() {
        let coords = TetP6.dof_coords(); let n = 84; let mut phi = vec![0.0; n];
        for (i,c) in coords.iter().enumerate() { TetP6.eval_basis(&[c[0],c[1],c[2]], &mut phi);
            for j in 0..n { assert!((phi[j]-if i==j{1.0}else{0.0}).abs()<1e-11); }
        }
    }

    #[test]
    fn tet_p6_matches_tet_pk() { use crate::lagrange::factory::TetPk; assert_eq!(TetPk::new(6).n_dofs(), 84); }
}
