//! Lagrange elements on the reference segment `[0, 1]`.

use crate::quadrature::seg_rule;
use crate::reference::{QuadratureRule, ReferenceElement};

// ─── P1 ───────────────────────────────────────────────────────────────────────

/// Linear Lagrange element on `[0, 1]` — 2 DOFs at the vertices.
///
/// Basis:  φ₀ = 1 − ξ,  φ₁ = ξ
pub struct SegP1;

impl ReferenceElement for SegP1 {
    fn dim(&self)   -> u8    { 1 }
    fn order(&self) -> u8    { 1 }
    fn n_dofs(&self) -> usize { 2 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let x = xi[0];
        values[0] = 1.0 - x;
        values[1] = x;
    }

    fn eval_grad_basis(&self, _xi: &[f64], grads: &mut [f64]) {
        // grads[i*1 + 0] = ∂φᵢ/∂ξ
        grads[0] = -1.0;
        grads[1] =  1.0;
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { seg_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![vec![0.0], vec![1.0]]
    }
}

// ─── P2 ───────────────────────────────────────────────────────────────────────

/// Quadratic Lagrange element on `[0, 1]` — 3 DOFs: two vertices + midpoint.
///
/// DOF order: 0 (ξ=0), 1 (ξ=1), 2 (ξ=½)
///
/// Basis:
/// - φ₀ = (1−ξ)(1−2ξ)
/// - φ₁ = ξ(2ξ−1)
/// - φ₂ = 4ξ(1−ξ)
pub struct SegP2;

impl ReferenceElement for SegP2 {
    fn dim(&self)   -> u8    { 1 }
    fn order(&self) -> u8    { 2 }
    fn n_dofs(&self) -> usize { 3 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let x = xi[0];
        values[0] = (1.0 - x) * (1.0 - 2.0 * x);
        values[1] = x * (2.0 * x - 1.0);
        values[2] = 4.0 * x * (1.0 - x);
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let x = xi[0];
        grads[0] = -3.0 + 4.0 * x;
        grads[1] =  4.0 * x - 1.0;
        grads[2] =  4.0 - 8.0 * x;
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { seg_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![vec![0.0], vec![1.0], vec![0.5]]
    }
}

// ─── P3 ───────────────────────────────────────────────────────────────────────

/// Cubic Lagrange element on `[0, 1]` — 4 DOFs: two vertices + two interior points.
///
/// DOF order:
/// - 0: ξ = 0     (vertex)
/// - 1: ξ = 1     (vertex)
/// - 2: ξ = 1/3   (interior)
/// - 3: ξ = 2/3   (interior)
///
/// Basis (Lagrange interpolation through 0, 1, 1/3, 2/3):
/// - φ₀ = −9/2·(ξ−1)(ξ−1/3)(ξ−2/3)  = −9ξ³/2 + 9ξ² − 11ξ/2 + 1
/// - φ₁ =  9/2·ξ(ξ−1/3)(ξ−2/3)       =  9ξ³/2 − 9ξ²/2 + ξ
/// - φ₂ =  27/2·ξ(ξ−1)(ξ−2/3)        =  27ξ³/2 − 45ξ²/2 + 9ξ
/// - φ₃ = −27/2·ξ(ξ−1)(ξ−1/3)        = −27ξ³/2 + 18ξ² − 9ξ/2
pub struct SegP3;

impl ReferenceElement for SegP3 {
    fn dim(&self)    -> u8     { 1 }
    fn order(&self)  -> u8     { 3 }
    fn n_dofs(&self) -> usize  { 4 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let x  = xi[0];
        let x2 = x * x;
        let x3 = x2 * x;
        values[0] = -4.5 * x3 + 9.0 * x2 - 5.5 * x + 1.0;
        values[1] =  4.5 * x3 - 4.5 * x2 + x;
        values[2] = 13.5 * x3 - 22.5 * x2 + 9.0 * x;
        values[3] = -13.5 * x3 + 18.0 * x2 - 4.5 * x;
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let x  = xi[0];
        let x2 = x * x;
        // grads[i] = ∂φᵢ/∂ξ  (dim=1, so grads[i*1+0] = grads[i])
        grads[0] = -13.5 * x2 + 18.0 * x - 5.5;
        grads[1] =  13.5 * x2 -  9.0 * x + 1.0;
        grads[2] =  40.5 * x2 - 45.0 * x + 9.0;
        grads[3] = -40.5 * x2 + 36.0 * x - 4.5;
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { seg_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![vec![0.0], vec![1.0], vec![1.0 / 3.0], vec![2.0 / 3.0]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_partition_of_unity(elem: &dyn ReferenceElement) {
        let rule = elem.quadrature(elem.order() * 2);
        let mut phi = vec![0.0_f64; elem.n_dofs()];
        for pt in &rule.points {
            elem.eval_basis(pt, &mut phi);
            let s: f64 = phi.iter().sum();
            assert!((s - 1.0).abs() < 1e-14, "POU failed at {:?}: sum={s}", pt);
        }
    }

    fn check_grad_sum_zero(elem: &dyn ReferenceElement) {
        let dim = elem.dim() as usize;
        let rule = elem.quadrature(elem.order() * 2);
        let mut g = vec![0.0_f64; elem.n_dofs() * dim];
        for pt in &rule.points {
            elem.eval_grad_basis(pt, &mut g);
            for d in 0..dim {
                let s: f64 = (0..elem.n_dofs()).map(|i| g[i * dim + d]).sum();
                assert!(s.abs() < 1e-13, "grad sum d={d} != 0: {s} at {:?}", pt);
            }
        }
    }

    #[test]
    fn seg_p1_partition_of_unity() { check_partition_of_unity(&SegP1); }
    #[test]
    fn seg_p1_grad_sum_zero()      { check_grad_sum_zero(&SegP1); }
    #[test]
    fn seg_p2_partition_of_unity() { check_partition_of_unity(&SegP2); }
    #[test]
    fn seg_p2_grad_sum_zero()      { check_grad_sum_zero(&SegP2); }

    #[test]
    fn seg_p2_recovers_linear() {
        // φ₂ at ξ=0.5 should be 1.0; vertex functions should be 0.
        let mut phi = vec![0.0; 3];
        SegP2.eval_basis(&[0.5], &mut phi);
        assert!((phi[0]).abs() < 1e-14);
        assert!((phi[1]).abs() < 1e-14);
        assert!((phi[2] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn seg_p3_partition_of_unity() { check_partition_of_unity(&SegP3); }
    #[test]
    fn seg_p3_grad_sum_zero()      { check_grad_sum_zero(&SegP3); }

    #[test]
    fn seg_p3_vertex_dofs() {
        // DOF 0 at ξ=0, DOF 1 at ξ=1 — all others should vanish.
        let mut phi = vec![0.0; 4];
        SegP3.eval_basis(&[0.0], &mut phi);
        assert!((phi[0] - 1.0).abs() < 1e-14, "phi0(0)={}", phi[0]);
        assert!(phi[1].abs() < 1e-14);
        assert!(phi[2].abs() < 1e-14);
        assert!(phi[3].abs() < 1e-14);

        SegP3.eval_basis(&[1.0], &mut phi);
        assert!(phi[0].abs() < 1e-14);
        assert!((phi[1] - 1.0).abs() < 1e-14, "phi1(1)={}", phi[1]);
        assert!(phi[2].abs() < 1e-14);
        assert!(phi[3].abs() < 1e-14);
    }

    #[test]
    fn seg_p3_interior_dofs() {
        // DOF 2 at ξ=1/3, DOF 3 at ξ=2/3 — Lagrange delta property.
        let mut phi = vec![0.0; 4];
        SegP3.eval_basis(&[1.0 / 3.0], &mut phi);
        assert!(phi[0].abs() < 1e-13);
        assert!(phi[1].abs() < 1e-13);
        assert!((phi[2] - 1.0).abs() < 1e-13, "phi2(1/3)={}", phi[2]);
        assert!(phi[3].abs() < 1e-13);

        SegP3.eval_basis(&[2.0 / 3.0], &mut phi);
        assert!(phi[0].abs() < 1e-13);
        assert!(phi[1].abs() < 1e-13);
        assert!(phi[2].abs() < 1e-13);
        assert!((phi[3] - 1.0).abs() < 1e-13, "phi3(2/3)={}", phi[3]);
    }
}

// ─── P4/P5/P6 — using factory SegPk ─────────────────────────────────────────

macro_rules! seg_fixed {
    ($name:ident, $p:literal) => {
        impl ReferenceElement for $name {
            fn dim(&self) -> u8 { 1 }
            fn order(&self) -> u8 { $p }
            fn n_dofs(&self) -> usize { $p as usize + 1 }
            fn eval_basis(&self, xi:&[f64], v:&mut[f64]) {
                use crate::lagrange::factory::{lagrange_val as lv, lagrange_deriv as _ld};
                let t=$p as f64*xi[0]; for i in 0..=$p as usize { v[i]=lv(i,$p as usize,t); }
            }
            fn eval_grad_basis(&self, xi:&[f64], g:&mut[f64]) {
                use crate::lagrange::factory::{lagrange_val as _lv, lagrange_deriv as ld};
                let t=$p as f64*xi[0]; let p=$p as f64;
                for i in 0..=$p as usize { g[i]=p*ld(i,$p as usize,t); }
            }
            fn quadrature(&self, o: u8) -> QuadratureRule { seg_rule(o) }
            fn dof_coords(&self) -> Vec<Vec<f64>> {
                (0..=$p as usize).map(|i| vec![i as f64/$p as f64]).collect()
            }
        }
    };
}

/// Quartic Lagrange on [0,1] — 5 DOFs.
pub struct SegP4;
seg_fixed!(SegP4, 4);

/// Quintic Lagrange on [0,1] — 6 DOFs.
pub struct SegP5;
seg_fixed!(SegP5, 5);

/// Sextic Lagrange on [0,1] — 7 DOFs.
pub struct SegP6;
seg_fixed!(SegP6, 6);

#[cfg(test)]
mod tests_p4 {
    use super::*;
    fn check_pou(e: &dyn ReferenceElement) { let r=e.quadrature(4); let mut p=vec![0.0;e.n_dofs()];
        for pt in &r.points { e.eval_basis(pt,&mut p); assert!((p.iter().sum::<f64>()-1.0).abs()<1e-13); } }
    fn check_grad(e: &dyn ReferenceElement) { let r=e.quadrature(4); let n=e.n_dofs(); let mut g=vec![0.0;n];
        for pt in &r.points { e.eval_grad_basis(pt,&mut g);
            assert!(g.iter().sum::<f64>().abs()<1e-13); }}
    #[test] fn seg_p4_pou() { check_pou(&SegP4); }
    #[test] fn seg_p5_pou() { check_pou(&SegP5); }
    #[test] fn seg_p6_pou() { check_pou(&SegP6); }
    #[test] fn seg_p4_grad() { check_grad(&SegP4); }
    #[test] fn seg_p5_grad() { check_grad(&SegP5); }
    #[test] fn seg_p6_grad() { check_grad(&SegP6); }
    #[test] fn seg_p4_nodal() { let c=SegP4.dof_coords(); let mut p=vec![0.0;5];
        for (i,coord) in c.iter().enumerate() { SegP4.eval_basis(&[coord[0]],&mut p);
            for j in 0..5 { assert!((p[j]-if i==j{1.0}else{0.0}).abs()<1e-13); }}}
    #[test] fn seg_p5_nodal() { let c=SegP5.dof_coords(); let mut p=vec![0.0;6];
        for (i,coord) in c.iter().enumerate() { SegP5.eval_basis(&[coord[0]],&mut p);
            for j in 0..6 { assert!((p[j]-if i==j{1.0}else{0.0}).abs()<1e-13); }}}
    #[test] fn seg_p6_nodal() { let c=SegP6.dof_coords(); let mut p=vec![0.0;7];
        for (i,coord) in c.iter().enumerate() { SegP6.eval_basis(&[coord[0]],&mut p);
            for j in 0..7 { assert!((p[j]-if i==j{1.0}else{0.0}).abs()<1e-13); }}}
}
