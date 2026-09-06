//! TMOP_Integrator - core integrator for Target-Matrix Optimization Paradigm.
//!
//! Ported from MFEM's `fem/tmop.hpp` (class TMOP_Integrator).
//!
//! The integrator computes:
//!   ∫ W(Jpt) dx  where  Jpt = Jpr * Jrt
//!   Jpr = ref→physical Jacobian
//!   Jrt = ref→target Jacobian (from TargetConstructor)
//!   W = quality metric

use crate::tmop::invariants::{InvariantsEvaluator2D, InvariantsEvaluator3D};
use crate::tmop::metrics::{TmopQualityMetric, TmopQualityMetric3D};
use crate::tmop::target::{TargetConstructor, ideal_shape_jac_2d, ideal_shape_jac_3d};

/// TMOP integrator for 2D meshes.
///
/// Computes element energy, gradient (1st derivative), and Hessian (2nd derivative)
/// for the TMOP quality metric.
pub struct TmopIntegrator2D {
    metric: Box<dyn TmopQualityMetric>,
    target_constructor: TargetConstructor,
    /// Integration order (default: 2*order + 3)
    integ_order: i32,
    /// Metric coefficient (optional, defaults to 1.0)
    metric_coeff: Option<f64>,
}

impl TmopIntegrator2D {
    /// Create a new 2D TMOP integrator.
    pub fn new(metric: Box<dyn TmopQualityMetric>, target_constructor: TargetConstructor) -> Self {
        Self {
            metric,
            target_constructor,
            integ_order: -1, // use default
            metric_coeff: None,
        }
    }

    /// Set the metric coefficient (scalar weight).
    pub fn set_metric_coeff(&mut self, coeff: f64) {
        self.metric_coeff = Some(coeff);
    }

    /// Set the integration order.
    pub fn set_integration_order(&mut self, order: i32) {
        self.integ_order = order;
    }

    /// Compute the element energy for a 2D element.
    ///
    /// `node_pos` is the element's node positions in column-major [dof*2].
    /// `quad_rule` is the integration rule (list of (weight, ref_point)).
    pub fn compute_element_energy(
        &self,
        node_pos: &[f64],
        quad_rule: &[(f64, [f64; 2])],
        ideal_jac: &[f64; 4],
    ) -> f64 {
        let n_dofs = node_pos.len() / 2;
        let mut energy = 0.0;

        for &(weight, ref_pt) in quad_rule {
            // Compute Jpr (ref→physical Jacobian)
            // For linear elements: Jpr = node_pos^T * dshape
            // Simplified: assume affine, Jpr is constant
            let jpr = compute_jpr_2d(node_pos, n_dofs, &ref_pt);

            // Compute Jtr (ref→target Jacobian)
            let jtr = self.target_constructor.compute_element_target_2d(
                ideal_jac,
                Some(node_pos),
                Some(jpr.determinant()),
            );

            // Jpt = Jpr * Jrt where Jrt = Jtr^{-1}
            let jrt = invert_2x2(&jtr);
            let jpt = matmul_2x2(&jpr, &jrt);

            // Metric evaluation
            let jpt_arr = [[jpt[0], jpt[2]], [jpt[1], jpt[3]]];
            let mut w = self.metric.eval_w(&jpt_arr);

            if let Some(coeff) = self.metric_coeff {
                w *= coeff;
            }

            // Weight includes det(Jtr) for integration over target
            let det_jtr = jtr[0] * jtr[3] - jtr[1] * jtr[2];
            energy += weight * det_jtr * w;
        }

        energy
    }

    /// Compute the element gradient (1st derivative) using finite differences.
    pub fn compute_element_gradient_fd(
        &self,
        node_pos: &[f64],
        quad_rule: &[(f64, [f64; 2])],
        ideal_jac: &[f64; 4],
        fd_h: f64,
    ) -> Vec<f64> {
        let n = node_pos.len();
        let mut grad = vec![0.0; n];
        let e0 = self.compute_element_energy(node_pos, quad_rule, ideal_jac);

        for i in 0..n {
            let mut pos_plus = node_pos.to_vec();
            pos_plus[i] += fd_h;
            let e_plus = self.compute_element_energy(&pos_plus, quad_rule, ideal_jac);
            grad[i] = (e_plus - e0) / fd_h;
        }

        grad
    }

    /// Compute the element Hessian (2nd derivative) using finite differences.
    pub fn compute_element_hessian_fd(
        &self,
        node_pos: &[f64],
        quad_rule: &[(f64, [f64; 2])],
        ideal_jac: &[f64; 4],
        fd_h: f64,
    ) -> Vec<Vec<f64>> {
        let n = node_pos.len();
        let mut hessian = vec![vec![0.0; n]; n];

        for i in 0..n {
            let mut pos_plus = node_pos.to_vec();
            pos_plus[i] += fd_h;
            let grad_plus = self.compute_element_gradient_fd(&pos_plus, quad_rule, ideal_jac, fd_h);

            let grad_0 = self.compute_element_gradient_fd(node_pos, quad_rule, ideal_jac, fd_h);

            for j in 0..n {
                hessian[i][j] = (grad_plus[j] - grad_0[j]) / fd_h;
            }
        }

        hessian
    }
}

/// TMOP integrator for 3D meshes.
pub struct TmopIntegrator3D {
    metric: Box<dyn TmopQualityMetric3D>,
    target_constructor: TargetConstructor,
    integ_order: i32,
    metric_coeff: Option<f64>,
}

impl TmopIntegrator3D {
    pub fn new(metric: Box<dyn TmopQualityMetric3D>, target_constructor: TargetConstructor) -> Self {
        Self {
            metric,
            target_constructor,
            integ_order: -1,
            metric_coeff: None,
        }
    }

    pub fn set_metric_coeff(&mut self, coeff: f64) {
        self.metric_coeff = Some(coeff);
    }

    pub fn compute_element_energy(
        &self,
        node_pos: &[f64],
        quad_rule: &[(f64, [f64; 3])],
        ideal_jac: &[f64; 9],
    ) -> f64 {
        let n_dofs = node_pos.len() / 3;
        let mut energy = 0.0;

        for &(weight, ref_pt) in quad_rule {
            let jpr = compute_jpr_3d(node_pos, n_dofs, &ref_pt);
            let jtr = self.target_constructor.compute_element_target_3d(
                ideal_jac,
                Some(node_pos),
                Some(jpr.determinant()),
            );
            let jrt = invert_3x3(&jtr);
            let jpt = matmul_3x3(&jpr, &jrt);

            let jpt_arr = [
                [jpt[0], jpt[3], jpt[6]],
                [jpt[1], jpt[4], jpt[7]],
                [jpt[2], jpt[5], jpt[8]],
            ];
            let mut w = self.metric.eval_w(&jpt_arr);

            if let Some(coeff) = self.metric_coeff {
                w *= coeff;
            }

            let det_jtr = jtr[0] * (jtr[4] * jtr[8] - jtr[5] * jtr[7])
                - jtr[1] * (jtr[3] * jtr[8] - jtr[5] * jtr[6])
                + jtr[2] * (jtr[3] * jtr[7] - jtr[4] * jtr[6]);
            energy += weight * det_jtr * w;
        }

        energy
    }

    pub fn compute_element_gradient_fd(
        &self,
        node_pos: &[f64],
        quad_rule: &[(f64, [f64; 3])],
        ideal_jac: &[f64; 9],
        fd_h: f64,
    ) -> Vec<f64> {
        let n = node_pos.len();
        let mut grad = vec![0.0; n];
        let e0 = self.compute_element_energy(node_pos, quad_rule, ideal_jac);

        for i in 0..n {
            let mut pos_plus = node_pos.to_vec();
            pos_plus[i] += fd_h;
            let e_plus = self.compute_element_energy(&pos_plus, quad_rule, ideal_jac);
            grad[i] = (e_plus - e0) / fd_h;
        }

        grad
    }
}

// ============================================================================
// Helper functions for 2D matrix operations
// ============================================================================

/// Compute 2x2 Jacobian from node positions (simplified for linear elements).
fn compute_jpr_2d(node_pos: &[f64], n_dofs: usize, _ref_pt: &[f64; 2]) -> nalgebra::Matrix2<f64> {
    // For linear Tri3: Jpr = [x1-x0, x2-x0; y1-y0, y2-y0]
    // node_pos is [x0, x1, x2, y0, y1, y2] (column-major)
    let x0 = node_pos[0];
    let x1 = node_pos[1 % n_dofs];
    let x2 = node_pos[2 % n_dofs];
    let y0 = node_pos[n_dofs];
    let y1 = node_pos[n_dofs + 1 % n_dofs];
    let y2 = node_pos[n_dofs + 2 % n_dofs];

    nalgebra::Matrix2::new(x1 - x0, x2 - x0, y1 - y0, y2 - y0)
}

/// Invert a 2x2 matrix (column-major [f64; 4]).
fn invert_2x2(m: &[f64; 4]) -> [f64; 4] {
    let det = m[0] * m[3] - m[1] * m[2];
    let inv_det = 1.0 / det;
    [m[3] * inv_det, -m[1] * inv_det, -m[2] * inv_det, m[0] * inv_det]
}

/// Multiply two 2x2 matrices (column-major).
fn matmul_2x2(a: &nalgebra::Matrix2<f64>, b: &[f64; 4]) -> [f64; 4] {
    let b_mat = nalgebra::Matrix2::new(b[0], b[2], b[1], b[3]);
    let c = a * b_mat;
    [c[(0, 0)], c[(1, 0)], c[(0, 1)], c[(1, 1)]]
}

// ============================================================================
// Helper functions for 3D matrix operations
// ============================================================================

fn compute_jpr_3d(node_pos: &[f64], n_dofs: usize, _ref_pt: &[f64; 3]) -> nalgebra::Matrix3<f64> {
    let x0 = node_pos[0];
    let x1 = node_pos[1 % n_dofs];
    let x2 = node_pos[2 % n_dofs];
    let x3 = node_pos[3 % n_dofs];
    let y0 = node_pos[n_dofs];
    let y1 = node_pos[n_dofs + 1 % n_dofs];
    let y2 = node_pos[n_dofs + 2 % n_dofs];
    let y3 = node_pos[n_dofs + 3 % n_dofs];
    let z0 = node_pos[2 * n_dofs];
    let z1 = node_pos[2 * n_dofs + 1 % n_dofs];
    let z2 = node_pos[2 * n_dofs + 2 % n_dofs];
    let z3 = node_pos[2 * n_dofs + 3 % n_dofs];

    nalgebra::Matrix3::new(
        x1 - x0, x2 - x0, x3 - x0,
        y1 - y0, y2 - y0, y3 - y0,
        z1 - z0, z2 - z0, z3 - z0,
    )
}

fn invert_3x3(m: &[f64; 9]) -> [f64; 9] {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7])
        - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    let inv_det = 1.0 / det;

    let mut result = [0.0; 9];
    result[0] = (m[4] * m[8] - m[5] * m[7]) * inv_det;
    result[1] = (m[2] * m[7] - m[1] * m[8]) * inv_det;
    result[2] = (m[1] * m[5] - m[2] * m[4]) * inv_det;
    result[3] = (m[5] * m[6] - m[3] * m[8]) * inv_det;
    result[4] = (m[0] * m[8] - m[2] * m[6]) * inv_det;
    result[5] = (m[2] * m[3] - m[0] * m[5]) * inv_det;
    result[6] = (m[3] * m[7] - m[4] * m[6]) * inv_det;
    result[7] = (m[1] * m[6] - m[0] * m[7]) * inv_det;
    result[8] = (m[0] * m[4] - m[1] * m[3]) * inv_det;
    result
}

fn matmul_3x3(a: &nalgebra::Matrix3<f64>, b: &[f64; 9]) -> [f64; 9] {
    let b_mat = nalgebra::Matrix3::new(b[0], b[3], b[6], b[1], b[4], b[7], b[2], b[5], b[8]);
    let c = a * b_mat;
    [
        c[(0, 0)], c[(1, 0)], c[(2, 0)],
        c[(0, 1)], c[(1, 1)], c[(2, 1)],
        c[(0, 2)], c[(1, 2)], c[(2, 2)],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tmop::metrics::{TmopMetric002, TmopMetric007};
    use crate::tmop::target::TargetType;

    #[test]
    fn test_integrator_2d_energy() {
        let metric = Box::new(TmopMetric002);
        let tc = TargetConstructor::new(TargetType::IdealShapeUnitSize);
        let integrator = TmopIntegrator2D::new(metric, tc);

        // Unit triangle: nodes at (0,0), (1,0), (0,1)
        let node_pos = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let ideal = ideal_shape_jac_2d("TRIANGLE");

        // Simple 1-point quadrature at centroid
        let quad_rule = vec![(1.0 / 6.0, [1.0 / 3.0, 1.0 / 3.0])];

        let energy = integrator.compute_element_energy(&node_pos, &quad_rule, &ideal);
        // For unit triangle with ideal shape, energy should be small
        assert!(energy >= 0.0, "Energy should be non-negative");
    }

    #[test]
    fn test_integrator_2d_gradient() {
        let metric = Box::new(TmopMetric007);
        let tc = TargetConstructor::new(TargetType::IdealShapeUnitSize);
        let integrator = TmopIntegrator2D::new(metric, tc);

        let node_pos = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let ideal = ideal_shape_jac_2d("TRIANGLE");
        let quad_rule = vec![(1.0 / 6.0, [1.0 / 3.0, 1.0 / 3.0])];

        let grad = integrator.compute_element_gradient_fd(&node_pos, &quad_rule, &ideal, 1e-6);
        assert_eq!(grad.len(), 6);
    }

    #[test]
    fn test_integrator_3d_energy() {
        let metric = Box::new(crate::tmop::metrics::TmopMetric301);
        let tc = TargetConstructor::new(TargetType::IdealShapeUnitSize);
        let integrator = TmopIntegrator3D::new(metric, tc);

        // Unit tet: nodes at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        let node_pos = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let ideal = ideal_shape_jac_3d("TETRAHEDRON");

        let quad_rule = vec![(1.0 / 24.0, [0.25, 0.25, 0.25])];

        let energy = integrator.compute_element_energy(&node_pos, &quad_rule, &ideal);
        assert!(energy >= 0.0, "Energy should be non-negative");
    }
}
