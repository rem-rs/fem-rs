//! Target-Matrix Optimization Paradigm (TMOP) for mesh quality.
//!
//! TMOP optimises node positions by comparing each element's Jacobian
//! matrix `A` against a **target** Jacobian `W`. The deformation gradient
//! `T = A · W⁻¹` measures how far the current element is from the ideal,
//! and a metric `μ(T)` drives the optimisation.
//!
//! # Metrics
//!
//! | Metric | Formula | Use case |
//! |--------|---------|----------|
//! | L2     | `|T|²` | Untangling, smoothing |
//! | Shape  | `|T|² / det(T)^{1/d} - d` | Shape-only (ignore size) |
//! | Size+Shape | `|T|² / det(T)^{1/d}` | Combined control |
//!
//! # Usage
//! ```rust,ignore
//! use fem_mesh::tmop::{TmopMetric, tmop_metric_2d, TmopObjective};
//! let obj = TmopObjective::new(mesh);
//! let grad = obj.gradient_with_metric(0, &TmopMetric::Shape);
//! ```

use std::collections::HashSet;
use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};
use crate::{SimplexMesh, topology::MeshTopology};

/// Available TMOP quality metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TmopMetric {
    /// Frobenius-squared norm of T: `μ = ∣T∣² = tr(T^T T)`.
    /// Minimising this tends toward the target shape.
    L2,
    /// Shape metric (size-independent): `μ = ∣T∣² / det(T)^{1/d} - d`.
    Shape,
    /// Combined size+shape: `μ = ∣T∣² / det(T)^{1/d}`.
    SizeShape,
}

/// Result of evaluating a TMOP metric on a 2-D element.
#[derive(Debug, Clone, Copy)]
pub struct TmopElementMetric2d {
    /// Metric value μ(T).
    pub value: f64,
    /// Derivative of μ w.r.t. the Jacobian matrix A: ∂μ/∂A (2×2).
    pub dmu_da: [[f64; 2]; 2],
}

/// Compute the deformation gradient `T = A · W⁻¹` for a 2-D element.
///
/// `A = [x₁-x₀, x₂-x₀]` is the physical Jacobian.
/// `W` is the target Jacobian (identity by default).
fn deformation_gradient_2d(a: &Matrix2<f64>, w: &Matrix2<f64>) -> Matrix2<f64> {
    a * w.try_inverse().unwrap_or(Matrix2::identity())
}

/// Frobenius norm-squared of a 2×2 matrix: `∣T∣² = Σᵢⱼ T_ij²`.
fn frobenius2_2d(t: &Matrix2<f64>) -> f64 {
    t.iter().map(|v| v * v).sum()
}

/// Evaluate a TMOP metric μ(T) and its derivative ∂μ/∂A for a 2-D element.
///
/// `a` is the current Jacobian (columns = edge vectors of the physical element).
/// `w` is the target Jacobian (identity for shape optimisation).
pub fn tmop_metric_2d(a: &Matrix2<f64>, w: &Matrix2<f64>, metric: &TmopMetric) -> TmopElementMetric2d {
    let t = deformation_gradient_2d(a, w);
    let ft2 = frobenius2_2d(&t); // ∣T∣²
    let det_t = t.determinant();
    let det_w = w.determinant().abs().max(1e-30);
    let winv = w.try_inverse().unwrap_or(Matrix2::identity());
    let d = 2usize; // dimension

    match metric {
        TmopMetric::L2 => {
            let value = ft2;
            // ∂μ/∂A = 2 * T * W^{-T}
            let mut dmu_da = Matrix2::zeros();
            for i in 0..2 {
                for j in 0..2 {
                    dmu_da[(i, j)] = 2.0
                        * (t[(i, 0)] * winv[(j, 0)] + t[(i, 1)] * winv[(j, 1)]);
                }
            }
            TmopElementMetric2d {
                value,
                dmu_da: [[dmu_da[(0, 0)], dmu_da[(0, 1)]], [dmu_da[(1, 0)], dmu_da[(1, 1)]]],
            }
        }
        TmopMetric::Shape | TmopMetric::SizeShape => {
            let det_t_abs = det_t.abs().max(1e-30);
            let (det_power, d_power_per_det) = if *metric == TmopMetric::Shape {
                // μ = |T|² / det(T) - d  (scale-invariant: T / det(T)^(1/d) with d=2 → |T|²/det(T)-2)
                (det_t_abs, 1.0)
            } else {
                // μ = |T|² / det(T)^{1/d}  (size+shape)
                (det_t_abs.powf(1.0 / d as f64), 1.0 / d as f64)
            };
            let value = if *metric == TmopMetric::Shape {
                ft2 / det_power - d as f64
            } else {
                ft2 / det_power
            };

            // ∂μ/∂T = 2*T / denom - ft2 * ∂denom/∂T / denom²
            // ∂denom/∂T = d_power_per_det * det_power * T^{-T}
            //       = p * det(T)^p * T^{-T}  where denom = det(T)^p
            let t_inv_t = t.try_inverse().map(|m| m.transpose()).unwrap_or(Matrix2::identity());
            let mut dmu_dt = Matrix2::zeros();
            for i in 0..2 {
                for j in 0..2 {
                    let ddenom_dt = d_power_per_det * det_power * t_inv_t[(i, j)] / det_t_abs;
                    dmu_dt[(i, j)] = 2.0 * t[(i, j)] / det_power
                        - ft2 * ddenom_dt / (det_power * det_power);
                }
            }

            let mut dmu_da = Matrix2::zeros();
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        dmu_da[(i, j)] += dmu_dt[(i, k)] * winv[(j, k)];
                    }
                }
            }

            TmopElementMetric2d {
                value,
                dmu_da: [[dmu_da[(0, 0)], dmu_da[(0, 1)]], [dmu_da[(1, 0)], dmu_da[(1, 1)]]],
            }
        }
    }
}

/// Objective function and gradient for 2-D TMOP mesh optimisation.
///
/// The objective is the sum of per-element metric values.
/// The gradient is dJ/dx_i for each free (interior) node.
pub struct TmopObjective2d {
    n_nodes: usize,
    coords: Vec<f64>,
    conn: Vec<u32>,
    elem_tags: Option<Vec<i32>>,
    free_nodes: Vec<usize>,
    targets: Vec<Matrix2<f64>>,
}

impl TmopObjective2d {
    /// Build from a 2-D triangular mesh.
    ///
    /// Interior nodes are optimised; boundary nodes are fixed.
    /// Target matrices default to identity for shape optimisation.
    pub fn new(mesh: &SimplexMesh<2>) -> Self {
        let n_nodes = mesh.n_nodes();
        let coords = mesh.coords.clone();
        let n_elem = mesh.n_elems();

        // Identify interior nodes (not on any boundary face)
        let mut on_boundary = vec![false; n_nodes];
        for f in 0..mesh.n_boundary_faces() as u32 {
            let nodes = mesh.bface_nodes(f);
            for &n in nodes {
                on_boundary[n as usize] = true;
            }
        }
        let free_nodes: Vec<usize> = (0..n_nodes).filter(|&i| !on_boundary[i]).collect();

        // Build target matrices (identity for each element)
        let targets = vec![Matrix2::identity(); n_elem];

        TmopObjective2d {
            n_nodes,
            coords,
            conn: mesh.conn.clone(),
            elem_tags: None,
            free_nodes,
            targets,
        }
    }

    /// Number of free (interior) nodes.
    pub fn n_free(&self) -> usize {
        self.free_nodes.len()
    }

    /// Get current interior node coordinates as a flat vector.
    pub fn get_x(&self) -> Vec<f64> {
        self.free_nodes.iter().flat_map(|&i| {
            vec![self.coords[2 * i], self.coords[2 * i + 1]]
        }).collect()
    }

    /// Set interior node coordinates from a flat vector.
    pub fn set_x(&mut self, x: &[f64]) {
        for (k, &ni) in self.free_nodes.iter().enumerate() {
            self.coords[2 * ni] = x[2 * k];
            self.coords[2 * ni + 1] = x[2 * k + 1];
        }
    }

    /// Element node coordinates (2-D).
    fn elem_nodes_2d(&self, e: u32) -> Vec<[f64; 2]> {
        let npe = 3; // Tri3
        let base = e as usize * npe;
        (0..npe).map(|k| {
            let ni = self.conn[base + k] as usize;
            [self.coords[2 * ni], self.coords[2 * ni + 1]]
        }).collect()
    }

    /// Jacobian matrix for element `e` (columns = edge vectors from node 0).
    fn elem_jacobian(&self, nodes: &[[f64; 2]; 3]) -> Matrix2<f64> {
        Matrix2::new(
            nodes[1][0] - nodes[0][0], nodes[2][0] - nodes[0][0],
            nodes[1][1] - nodes[0][1], nodes[2][1] - nodes[0][1],
        )
    }

    /// Evaluate the objective and its gradient w.r.t. interior node coordinates.
    pub fn value_and_gradient(&self, metric: &TmopMetric, obj: &mut f64, grad: &mut [f64]) {
        *obj = 0.0;
        for g in grad.iter_mut() { *g = 0.0; }

        let mut node_grad: Vec<Vec<f64>> = vec![vec![0.0, 0.0]; self.n_nodes];

        for e in 0..self.n_elem() as u32 {
            let n = self.elem_nodes_2d(e);
            let nodes = [n[0], n[1], n[2]];
            let a = self.elem_jacobian(&nodes);
            let w = self.targets[e as usize];

            let result = tmop_metric_2d(&a, &w, metric);
            *obj += result.value;

            // dμ/dx_i = dμ/dA : dA/dx_i
            // A = [x1-x0, x2-x0], dA/dx_0 = -I, dA/dx_1 = [I, 0], dA/dx_2 = [0, I]
            let dmu = [[result.dmu_da[0][0], result.dmu_da[0][1]],
                       [result.dmu_da[1][0], result.dmu_da[1][1]]];

            // Node 0: x0 affects all columns
            node_grad[n[0][0] as usize][0] -= dmu[0][0] + dmu[1][0];
            node_grad[n[0][0] as usize][1] -= dmu[0][1] + dmu[1][1];

            // Node 1: x1 affects column 0
            node_grad[n[1][0] as usize][0] += dmu[0][0];
            node_grad[n[1][0] as usize][1] += dmu[0][1];

            // Node 2: x2 affects column 1
            node_grad[n[2][0] as usize][0] += dmu[1][0];
            node_grad[n[2][0] as usize][1] += dmu[1][1];
        }

        // Copy free-node gradients
        for (k, &ni) in self.free_nodes.iter().enumerate() {
            grad[2 * k] = node_grad[ni][0];
            grad[2 * k + 1] = node_grad[ni][1];
        }
    }

    fn n_elem(&self) -> usize { self.conn.len() / 3 }

    /// Get all node coordinates (for output).
    pub fn coords(&self) -> &[f64] { &self.coords }
}

/// Run TMOP optimisation for a given number of iterations using gradient descent.
///
/// Returns the final node coordinate vector.
pub fn tmop_optimise_2d(
    mesh: &SimplexMesh<2>,
    metric: &TmopMetric,
    max_iter: usize,
    step_size: f64,
) -> Vec<f64> {
    let mut obj = TmopObjective2d::new(mesh);
    let n_free = obj.n_free();

    let mut x = obj.get_x();
    let mut grad = vec![0.0; 2 * n_free];
    let mut val = 0.0;

    for _iter in 0..max_iter {
        obj.set_x(&x);
        obj.value_and_gradient(metric, &mut val, &mut grad);

        // Gradient descent
        let mut g_max = 0.0;
        for i in 0..2 * n_free {
            g_max = f64::max(g_max, grad[i].abs());
        }
        if g_max < 1e-12 { break; }

        // Backtracking line search
        let mut alpha = step_size;
        for _ in 0..20 {
            let mut x_new = x.clone();
            for i in 0..2 * n_free {
                x_new[i] -= alpha * grad[i];
            }
            obj.set_x(&x_new);
            let mut new_val = 0.0;
            obj.value_and_gradient(metric, &mut new_val, &mut grad);
            if new_val < val || alpha < 1e-14 {
                x = x_new;
                val = new_val;
                break;
            }
            alpha *= 0.5;
        }
    }

    obj.set_x(&x);
    obj.coords().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SimplexMesh;

    fn unit_square_tri4() -> SimplexMesh<2> {
        SimplexMesh::<2>::unit_square_tri(4)
    }

    #[test]
    fn tmop_metric_l2_identity_is_minimal() {
        let a = Matrix2::identity();
        let w = Matrix2::identity();
        let res = tmop_metric_2d(&a, &w, &TmopMetric::L2);
        assert!((res.value - 2.0).abs() < 1e-12, "|I|² = 2, got {}", res.value);
    }

    #[test]
    fn tmop_metric_l2_derivative_finite_diff() {
        use nalgebra::Matrix2;
        let a = Matrix2::new(1.0, 0.5, 0.2, 1.2);
        let w = Matrix2::identity();
        let eps = 1e-6;

        let res = tmop_metric_2d(&a, &w, &TmopMetric::L2);

        for i in 0..2 {
            for j in 0..2 {
                let mut a_plus = a;
                a_plus[(i, j)] += eps;
                let r_plus = tmop_metric_2d(&a_plus, &w, &TmopMetric::L2);

                let mut a_minus = a;
                a_minus[(i, j)] -= eps;
                let r_minus = tmop_metric_2d(&a_minus, &w, &TmopMetric::L2);

                let fd = (r_plus.value - r_minus.value) / (2.0 * eps);
                assert!((fd - res.dmu_da[i][j]).abs() < 1e-6,
                    "finite diff mismatch at ({i},{j}): fd={fd:.6e}, analytic={:.6e}",
                    res.dmu_da[i][j]);
            }
        }
    }

    #[test]
    fn tmop_shape_metric_does_not_change_with_scaling() {
        // Shape metric should be scale-invariant: scaling A shouldn't change μ
        let a = Matrix2::new(1.0, 0.5, 0.2, 1.2);
        let w = Matrix2::identity();
        let r1 = tmop_metric_2d(&a, &w, &TmopMetric::Shape);
        let r2 = tmop_metric_2d(&(2.0 * a), &w, &TmopMetric::Shape);
        assert!((r1.value - r2.value).abs() < 1e-12,
            "Shape metric should be scale-invariant: {:.10} vs {:.10}", r1.value, r2.value);
    }

    #[test]
    fn tmop_optimise_improves_min_quality() {
        let mut mesh = unit_square_tri4();
        // Perturb interior nodes
        let n_nodes = mesh.n_nodes();
        for i in 0..n_nodes {
            let x = mesh.coords[2 * i];
            let y = mesh.coords[2 * i + 1];
            if x > 0.0 && x < 1.0 && y > 0.0 && y < 1.0 {
                mesh.coords[2 * i] += 0.1 * (x * std::f64::consts::PI).sin();
                mesh.coords[2 * i + 1] += 0.1 * (y * std::f64::consts::PI).cos();
            }
        }

        let result = tmop_optimise_2d(&mesh, &TmopMetric::Shape, 50, 0.1);

        // Compute min q before and after
        let min_q_initial = min_element_quality_2d(&mesh);
        let final_mesh = SimplexMesh::<2>::uniform(
            result.clone(), mesh.conn.clone(), mesh.elem_tags.clone(),
            mesh.elem_type, mesh.face_conn.clone(), mesh.face_tags.clone(),
            mesh.face_type,
        );
        let min_q_final = min_element_quality_2d(&final_mesh);

        assert!(min_q_final >= min_q_initial - 1e-10,
            "min quality should not decrease: initial={:.6}, final={:.6}", min_q_initial, min_q_final);
    }

    /// Mean-ratio metric for a triangle: q = 4*sqrt(3)*A / (l01² + l12² + l20²).
    fn element_quality_2d(mesh: &SimplexMesh<2>, e: u32) -> f64 {
        let ns = mesh.element_nodes(e);
        let p0 = mesh.node_coords(ns[0]);
        let p1 = mesh.node_coords(ns[1]);
        let p2 = mesh.node_coords(ns[2]);
        let l01 = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2)).sqrt();
        let l12 = ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2)).sqrt();
        let l20 = ((p0[0] - p2[0]).powi(2) + (p0[1] - p2[1]).powi(2)).sqrt();
        let a = 0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])).abs();
        4.0 * 3.0f64.sqrt() * a / (l01.powi(2) + l12.powi(2) + l20.powi(2))
    }

    fn min_element_quality_2d(mesh: &SimplexMesh<2>) -> f64 {
        (0..mesh.n_elems() as u32)
            .map(|e| element_quality_2d(mesh, e))
            .fold(f64::INFINITY, f64::min)
    }
}
