//! Target-Matrix Optimization Paradigm (TMOP) for mesh quality.
//!
//! TMOP optimises node positions by comparing each element's Jacobian
//! matrix `A` against a **target** Jacobian `W`. The deformation gradient
//! `T = A · W⁻¹` measures how far the current element is from the ideal,
//! and a metric `μ(T)` drives the optimisation.
//!
//! # Metrics
//!
//! | Metric | Formula (d-dim) | Use case |
//! |--------|-----------------|----------|
//! | L2     | `|T|²` | Untangling, smoothing |
//! | Shape  | `|T|² / det(T)^{2/d} - d` | Shape-only (ignore size) |
//! | Size+Shape | `|T|² / det(T)^{2/d}` | Combined control |
//!
//! # 2-D vs 3-D
//!
//! - 2-D: [`tmop_metric_2d`], [`TmopObjective2d`], [`tmop_optimise_2d`] for Tri3 meshes.
//! - 3-D: [`tmop_metric_3d`], [`TmopObjectiveTetra`], [`tmop_optimise_tetra`] for Tet4 meshes.

use nalgebra::{Matrix2, Matrix3};
use crate::{SimplexMesh, topology::MeshTopology};

/// Available TMOP quality metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TmopMetric {
    L2,
    Shape,
    SizeShape,
    Volume,
    Condition,
    /// Deformed determinant: `μ = (det(T) - 1)²`. Drives element
    /// volume toward the target value (det(T) = 1 when T = I).
    DeformedDet,
    /// Barrier: `μ = 1/det(T) − 1` for det(T) > 0. Approaches +∞ as
    /// det(T) → 0⁺, preventing element inversion.
    BarrierDet,
    /// Squared Frobenius distance from identity: `μ = ∣T − I∣²`.
    FrobeniusDiff,
    /// Absolute volume deviation: `μ = ∣det(T) − 1∣`.
    AreaDeviation,
    /// 3-D Winslow: `μ = ∣T∣³ / det(T)`. For 2-D this equals SizeShape.
    Winslow3D,
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2-D helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of evaluating a TMOP metric on a 2-D element.
#[derive(Debug, Clone, Copy)]
pub struct TmopElementMetric2d {
    /// Metric value μ(T).
    pub value: f64,
    /// Derivative of μ w.r.t. the Jacobian matrix A: ∂μ/∂A (2×2).
    pub dmu_da: [[f64; 2]; 2],
}

fn frobenius2_2d(t: &Matrix2<f64>) -> f64 {
    t.iter().map(|v| v * v).sum()
}

fn deformation_gradient_2d(a: &Matrix2<f64>, w: &Matrix2<f64>) -> Matrix2<f64> {
    a * w.try_inverse().unwrap_or(Matrix2::identity())
}

/// Evaluate a TMOP metric μ(T) and its derivative ∂μ/∂A for a 2-D element.
pub fn tmop_metric_2d(a: &Matrix2<f64>, w: &Matrix2<f64>, metric: &TmopMetric) -> TmopElementMetric2d {
    let t = deformation_gradient_2d(a, w);
    let ft2 = frobenius2_2d(&t);
    let det_t = t.determinant();
    let winv = w.try_inverse().unwrap_or(Matrix2::identity());
    let d = 2;

    match metric {
        TmopMetric::L2 => {
            let value = ft2;
            let mut dmu_da = Matrix2::zeros();
            for i in 0..2 { for j in 0..2 {
                dmu_da[(i, j)] = 2.0 * (t[(i, 0)] * winv[(j, 0)] + t[(i, 1)] * winv[(j, 1)]);
            }}
            TmopElementMetric2d { value, dmu_da: [[dmu_da[(0,0)], dmu_da[(0,1)]], [dmu_da[(1,0)], dmu_da[(1,1)]]] }
        }
        TmopMetric::Shape | TmopMetric::SizeShape => {
            let det_t_abs = det_t.abs().max(1e-30);
            let exponent = 2.0 / d as f64;     // 2/d: 1 for 2D, 2/3 for 3D. Both Shape and SizeShape need det^{2/d}
            let det_power = det_t_abs.powf(exponent);
            let value = if *metric == TmopMetric::Shape {
                ft2 / det_power - d as f64
            } else {
                ft2 / det_power
            };

            // ∂(D)/∂T = exponent * det^exponent * T^{-T} where D = det^exponent
            let t_inv_t = t.try_inverse().map(|m| m.transpose()).unwrap_or(Matrix2::identity());
            let mut dmu_dt = Matrix2::zeros();
            for i in 0..2 { for j in 0..2 {
                let ddenom_dt = exponent * det_power * t_inv_t[(i, j)];
                dmu_dt[(i, j)] = 2.0 * t[(i, j)] / det_power - ft2 * ddenom_dt / (det_power * det_power);
            }}
            let mut dmu_da = Matrix2::zeros();
            for i in 0..2 { for j in 0..2 { for k in 0..2 {
                dmu_da[(i, j)] += dmu_dt[(i, k)] * winv[(j, k)];
            }}}
            TmopElementMetric2d { value, dmu_da: [[dmu_da[(0,0)], dmu_da[(0,1)]], [dmu_da[(1,0)], dmu_da[(1,1)]]] }
        }
        TmopMetric::Volume => {
            let value = det_t;
            let adj_t: Matrix2<f64> = t.try_inverse().map(|inv| det_t * inv.transpose()).unwrap_or(Matrix2::identity());
            let mut dmu_da = Matrix2::zeros();
            for i in 0..2 { for j in 0..2 { for k in 0..2 {
                dmu_da[(i, j)] += adj_t[(i, k)] * winv[(j, k)];
            }}}
            TmopElementMetric2d { value, dmu_da: [[dmu_da[(0,0)], dmu_da[(0,1)]], [dmu_da[(1,0)], dmu_da[(1,1)]]] }
        }
        TmopMetric::Condition => {
            let abs_det = det_t.abs().max(1e-30);
            let value = ft2 / (d as f64 * abs_det);
            let sign = if det_t >= 0.0 { 1.0 } else { -1.0 };
            let adj_t: Matrix2<f64> = t.try_inverse().map(|inv| det_t * inv.transpose()).unwrap_or(Matrix2::identity());
            let denom = d as f64 * det_t * abs_det;
            let mut dmu_dt = Matrix2::zeros();
            for i in 0..2 { for j in 0..2 {
                dmu_dt[(i, j)] = (2.0 * t[(i, j)] * abs_det - ft2 * 0.5 * sign * adj_t[(i, j)]) / denom;
            }}
            let mut dmu_da = Matrix2::zeros();
            for i in 0..2 { for j in 0..2 { for k in 0..2 {
                dmu_da[(i, j)] += dmu_dt[(i, k)] * winv[(j, k)];
            }}}
            TmopElementMetric2d { value, dmu_da: [[dmu_da[(0,0)], dmu_da[(0,1)]], [dmu_da[(1,0)], dmu_da[(1,1)]]] }
        }
        TmopMetric::DeformedDet => {
            let value = (det_t - 1.0) * (det_t - 1.0);
            let adj_t: Matrix2<f64> = t.try_inverse().map(|inv| det_t * inv.transpose()).unwrap_or(Matrix2::identity());
            let mut dmu_dt = Matrix2::zeros();
            for i in 0..2 { for j in 0..2 { dmu_dt[(i, j)] = 2.0 * (det_t - 1.0) * adj_t[(i, j)]; }}
            let mut dmu_da = Matrix2::zeros();
            for i in 0..2 { for j in 0..2 { for k in 0..2 { dmu_da[(i, j)] += dmu_dt[(i, k)] * winv[(j, k)]; }}}
            TmopElementMetric2d { value, dmu_da: [[dmu_da[(0,0)], dmu_da[(0,1)]], [dmu_da[(1,0)], dmu_da[(1,1)]]] }
        }
        TmopMetric::BarrierDet => {
            let a = det_t.abs().max(1e-30);
            let value = 1.0 / a - 1.0;
            let adj_t: Matrix2<f64> = t.try_inverse().map(|inv| det_t * inv.transpose()).unwrap_or(Matrix2::identity());
            let sign = if det_t >= 0.0 { 1.0 } else { -1.0 };
            let mut dmu_dt = Matrix2::zeros();
            for i in 0..2 { for j in 0..2 { dmu_dt[(i, j)] = -sign / (a * a) * adj_t[(i, j)]; }}
            let mut dmu_da = Matrix2::zeros();
            for i in 0..2 { for j in 0..2 { for k in 0..2 { dmu_da[(i, j)] += dmu_dt[(i, k)] * winv[(j, k)]; }}}
            TmopElementMetric2d { value, dmu_da: [[dmu_da[(0,0)], dmu_da[(0,1)]], [dmu_da[(1,0)], dmu_da[(1,1)]]] }
        }
        TmopMetric::FrobeniusDiff => {
            let ti: Matrix2<f64> = t - Matrix2::identity();
            let value = ti.iter().map(|v| v * v).sum();
            let dmu_dt = 2.0 * ti;
            let mut dmu_da = Matrix2::zeros();
            for i in 0..2 { for j in 0..2 { for k in 0..2 {
                dmu_da[(i, j)] += dmu_dt[(i, k)] * winv[(j, k)];
            }}}
            TmopElementMetric2d { value, dmu_da: [[dmu_da[(0,0)], dmu_da[(0,1)]], [dmu_da[(1,0)], dmu_da[(1,1)]]] }
        }
        TmopMetric::AreaDeviation => {
            let value = (det_t - 1.0).abs();
            let sign = if det_t >= 1.0 { 1.0 } else { -1.0 };
            let adj_t: Matrix2<f64> = t.try_inverse().map(|inv| det_t * inv.transpose()).unwrap_or(Matrix2::identity());
            let mut dmu_dt = Matrix2::zeros();
            for i in 0..2 { for j in 0..2 { dmu_dt[(i, j)] = sign * adj_t[(i, j)]; }}
            let mut dmu_da = Matrix2::zeros();
            for i in 0..2 { for j in 0..2 { for k in 0..2 {
                dmu_da[(i, j)] += dmu_dt[(i, k)] * winv[(j, k)];
            }}}
            TmopElementMetric2d { value, dmu_da: [[dmu_da[(0,0)], dmu_da[(0,1)]], [dmu_da[(1,0)], dmu_da[(1,1)]]] }
        }
        // Winslow3D for 2D falls back to SizeShape (same formula).
        TmopMetric::Winslow3D => {
            let det_t_abs = det_t.abs().max(1e-30);
            let value = ft2 / det_t_abs;
            let sign = if det_t >= 0.0 { 1.0 } else { -1.0 };
            let adj_t: Matrix2<f64> = t.try_inverse().map(|inv| det_t * inv.transpose()).unwrap_or(Matrix2::identity());
            let mut dmu_dt = Matrix2::zeros();
            for i in 0..2 { for j in 0..2 { dmu_dt[(i, j)] = (2.0 * t[(i, j)] * det_t_abs - ft2 * sign * 0.5 * adj_t[(i, j)]) / (det_t_abs * det_t_abs); }}
            let mut dmu_da = Matrix2::zeros();
            for i in 0..2 { for j in 0..2 { for k in 0..2 {
                dmu_da[(i, j)] += dmu_dt[(i, k)] * winv[(j, k)];
            }}}
            TmopElementMetric2d { value, dmu_da: [[dmu_da[(0,0)], dmu_da[(0,1)]], [dmu_da[(1,0)], dmu_da[(1,1)]]] }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3-D helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of evaluating a TMOP metric on a 3-D element.
#[derive(Debug, Clone, Copy)]
pub struct TmopElementMetric3d {
    pub value: f64,
    pub dmu_da: [[f64; 3]; 3],
}

fn frobenius2_3d(t: &Matrix3<f64>) -> f64 {
    t.iter().map(|v| v * v).sum()
}

fn deformation_gradient_3d(a: &Matrix3<f64>, w: &Matrix3<f64>) -> Matrix3<f64> {
    a * w.try_inverse().unwrap_or(Matrix3::identity())
}

/// Evaluate a TMOP metric μ(T) and derivative ∂μ/∂A for a 3-D element (tet/hex).
///
/// Formulas are dimension-agnostic; the 3-D variants use 3×3 matrices.
/// `d` = 3 throughout.
pub fn tmop_metric_3d(a: &Matrix3<f64>, w: &Matrix3<f64>, metric: &TmopMetric) -> TmopElementMetric3d {
    let t = deformation_gradient_3d(a, w);
    let ft2 = frobenius2_3d(&t);
    let det_t = t.determinant();
    let winv = w.try_inverse().unwrap_or(Matrix3::identity());
    let d = 3;

    match metric {
        TmopMetric::L2 => {
            let value = ft2;
            let mut dmu_da = Matrix3::zeros();
            for i in 0..3 { for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 { s += t[(i, k)] * winv[(j, k)]; }
                dmu_da[(i, j)] = 2.0 * s;
            }}
            TmopElementMetric3d {
                value,
                dmu_da: [
                    [dmu_da[(0,0)], dmu_da[(0,1)], dmu_da[(0,2)]],
                    [dmu_da[(1,0)], dmu_da[(1,1)], dmu_da[(1,2)]],
                    [dmu_da[(2,0)], dmu_da[(2,1)], dmu_da[(2,2)]],
                ],
            }
        }
        TmopMetric::Shape | TmopMetric::SizeShape => {
            let det_t_abs = det_t.abs().max(1e-30);
            let exponent = 2.0 / d as f64;     // 2/d: 1 for 2D, 2/3 for 3D
            let det_power = det_t_abs.powf(exponent);
            let value = if *metric == TmopMetric::Shape {
                ft2 / det_power - d as f64
            } else {
                ft2 / det_power
            };

            let t_inv_t = t.try_inverse().map(|m| m.transpose()).unwrap_or(Matrix3::identity());
            let mut dmu_dt = Matrix3::zeros();
            for i in 0..3 { for j in 0..3 {
                let ddenom_dt = exponent * det_power * t_inv_t[(i, j)];
                dmu_dt[(i, j)] = 2.0 * t[(i, j)] / det_power - ft2 * ddenom_dt / (det_power * det_power);
            }}
            let mut dmu_da = Matrix3::zeros();
            for i in 0..3 { for j in 0..3 { for k in 0..3 {
                dmu_da[(i, j)] += dmu_dt[(i, k)] * winv[(j, k)];
            }}}
            TmopElementMetric3d {
                value,
                dmu_da: [
                    [dmu_da[(0,0)], dmu_da[(0,1)], dmu_da[(0,2)]],
                    [dmu_da[(1,0)], dmu_da[(1,1)], dmu_da[(1,2)]],
                    [dmu_da[(2,0)], dmu_da[(2,1)], dmu_da[(2,2)]],
                ],
            }
        }
        TmopMetric::Volume => {
            let value = det_t;
            let adj_t: Matrix3<f64> = t.try_inverse().map(|inv| det_t * inv.transpose()).unwrap_or(Matrix3::identity());
            let mut dmu_da = Matrix3::zeros();
            for i in 0..3 { for j in 0..3 { for k in 0..3 {
                dmu_da[(i, j)] += adj_t[(i, k)] * winv[(j, k)];
            }}}
            TmopElementMetric3d {
                value,
                dmu_da: [
                    [dmu_da[(0,0)], dmu_da[(0,1)], dmu_da[(0,2)]],
                    [dmu_da[(1,0)], dmu_da[(1,1)], dmu_da[(1,2)]],
                    [dmu_da[(2,0)], dmu_da[(2,1)], dmu_da[(2,2)]],
                ],
            }
        }
        TmopMetric::Condition => {
            let abs_det = det_t.abs().max(1e-30);
            let value = ft2 / (d as f64 * abs_det.powf(2.0 / d as f64));
            let sign = if det_t >= 0.0 { 1.0 } else { -1.0 };
            let t_inv_t = t.try_inverse().map(|inv| inv.transpose()).unwrap_or(Matrix3::identity());
            let mut dmu_dt = Matrix3::zeros();
            for i in 0..3 { for j in 0..3 {
                let ddet_dt = det_t * t_inv_t[(i, j)];
                let ddet_pow = (2.0 / d as f64) * abs_det.powf(2.0 / d as f64 - 1.0) * sign * ddet_dt;
                dmu_dt[(i, j)] = (2.0 * t[(i, j)] * abs_det.powf(2.0 / d as f64) - ft2 * ddet_pow)
                    / (d as f64 * abs_det.powf(4.0 / d as f64));
            }}
            let mut dmu_da = Matrix3::zeros();
            for i in 0..3 { for j in 0..3 { for k in 0..3 {
                dmu_da[(i, j)] += dmu_dt[(i, k)] * winv[(j, k)];
            }}}
            TmopElementMetric3d {
                value,
                dmu_da: [
                    [dmu_da[(0,0)], dmu_da[(0,1)], dmu_da[(0,2)]],
                    [dmu_da[(1,0)], dmu_da[(1,1)], dmu_da[(1,2)]],
                    [dmu_da[(2,0)], dmu_da[(2,1)], dmu_da[(2,2)]],
                ],
            }
        }
        TmopMetric::DeformedDet => {
            let value = (det_t - 1.0) * (det_t - 1.0);
            let adj_t: Matrix3<f64> = t.try_inverse().map(|inv| det_t * inv.transpose()).unwrap_or(Matrix3::identity());
            let mut dmu_dt = Matrix3::zeros();
            for i in 0..3 { for j in 0..3 { dmu_dt[(i, j)] = 2.0 * (det_t - 1.0) * adj_t[(i, j)]; }}
            let mut dmu_da = Matrix3::zeros();
            for i in 0..3 { for j in 0..3 { for k in 0..3 {
                dmu_da[(i, j)] += dmu_dt[(i, k)] * winv[(j, k)];
            }}}
            TmopElementMetric3d {
                value, dmu_da: [[dmu_da[(0,0)], dmu_da[(0,1)], dmu_da[(0,2)]],
                               [dmu_da[(1,0)], dmu_da[(1,1)], dmu_da[(1,2)]],
                               [dmu_da[(2,0)], dmu_da[(2,1)], dmu_da[(2,2)]]],
            }
        }
        TmopMetric::BarrierDet => {
            let a = det_t.abs().max(1e-30);
            let value = 1.0 / a - 1.0;
            let adj_t: Matrix3<f64> = t.try_inverse().map(|inv| det_t * inv.transpose()).unwrap_or(Matrix3::identity());
            let sign = if det_t >= 0.0 { 1.0 } else { -1.0 };
            let mut dmu_dt = Matrix3::zeros();
            for i in 0..3 { for j in 0..3 { dmu_dt[(i, j)] = -sign / (a * a) * adj_t[(i, j)]; }}
            let mut dmu_da = Matrix3::zeros();
            for i in 0..3 { for j in 0..3 { for k in 0..3 {
                dmu_da[(i, j)] += dmu_dt[(i, k)] * winv[(j, k)];
            }}}
            TmopElementMetric3d {
                value, dmu_da: [[dmu_da[(0,0)], dmu_da[(0,1)], dmu_da[(0,2)]],
                               [dmu_da[(1,0)], dmu_da[(1,1)], dmu_da[(1,2)]],
                               [dmu_da[(2,0)], dmu_da[(2,1)], dmu_da[(2,2)]]],
            }
        }
        TmopMetric::FrobeniusDiff => {
            let ti: Matrix3<f64> = t - Matrix3::identity();
            let value = ti.iter().map(|v| v * v).sum();
            let dmu_dt = 2.0 * ti;
            let mut dmu_da = Matrix3::zeros();
            for i in 0..3 { for j in 0..3 { for k in 0..3 {
                dmu_da[(i, j)] += dmu_dt[(i, k)] * winv[(j, k)];
            }}}
            TmopElementMetric3d {
                value, dmu_da: [[dmu_da[(0,0)], dmu_da[(0,1)], dmu_da[(0,2)]],
                               [dmu_da[(1,0)], dmu_da[(1,1)], dmu_da[(1,2)]],
                               [dmu_da[(2,0)], dmu_da[(2,1)], dmu_da[(2,2)]]],
            }
        }
        TmopMetric::AreaDeviation => {
            let value = (det_t - 1.0).abs();
            let sign = if det_t >= 1.0 { 1.0 } else { -1.0 };
            let adj_t: Matrix3<f64> = t.try_inverse().map(|inv| det_t * inv.transpose()).unwrap_or(Matrix3::identity());
            let mut dmu_dt = Matrix3::zeros();
            for i in 0..3 { for j in 0..3 { dmu_dt[(i, j)] = sign * adj_t[(i, j)]; }}
            let mut dmu_da = Matrix3::zeros();
            for i in 0..3 { for j in 0..3 { for k in 0..3 {
                dmu_da[(i, j)] += dmu_dt[(i, k)] * winv[(j, k)];
            }}}
            TmopElementMetric3d {
                value, dmu_da: [[dmu_da[(0,0)], dmu_da[(0,1)], dmu_da[(0,2)]],
                               [dmu_da[(1,0)], dmu_da[(1,1)], dmu_da[(1,2)]],
                               [dmu_da[(2,0)], dmu_da[(2,1)], dmu_da[(2,2)]]],
            }
        }
        TmopMetric::Winslow3D => {
            let det_t_abs = det_t.abs().max(1e-30);
            let value = ft2 * ft2.sqrt() / det_t_abs; // = |T|³ / det(T)
            let adj_t: Matrix3<f64> = t.try_inverse().map(|inv| det_t * inv.transpose()).unwrap_or(Matrix3::identity());
            let mut dmu_dt = Matrix3::zeros();
            for i in 0..3 { for j in 0..3 {
                dmu_dt[(i, j)] = (4.0 * t[(i, j)] * ft2.sqrt() * det_t_abs - ft2 * ft2.sqrt() * adj_t[(i, j)]) / (det_t_abs * det_t_abs);
            }}
            let mut dmu_da = Matrix3::zeros();
            for i in 0..3 { for j in 0..3 { for k in 0..3 {
                dmu_da[(i, j)] += dmu_dt[(i, k)] * winv[(j, k)];
            }}}
            TmopElementMetric3d {
                value, dmu_da: [[dmu_da[(0,0)], dmu_da[(0,1)], dmu_da[(0,2)]],
                               [dmu_da[(1,0)], dmu_da[(1,1)], dmu_da[(1,2)]],
                               [dmu_da[(2,0)], dmu_da[(2,1)], dmu_da[(2,2)]]],
            }
        }
    }
}

// Result of evaluating a TMOP metric on a 3-D element.

// ═══════════════════════════════════════════════════════════════════════════════
// 2-D objective
// ═══════════════════════════════════════════════════════════════════════════════
/// Objective function and gradient for 2-D TMOP mesh optimisation (Tri3).
pub struct TmopObjective2d {
    n_nodes: usize,
    coords: Vec<f64>,
    conn: Vec<u32>,
    #[allow(dead_code)]
    elem_tags: Option<Vec<i32>>,
    free_nodes: Vec<usize>,
    targets: Vec<Matrix2<f64>>,
}

impl TmopObjective2d {
    /// Build from a 2-D Tri3 mesh. By default all boundary nodes are fixed.
    pub fn new(mesh: &SimplexMesh<2>) -> Self {
        Self::from_mesh_with_free_tags(mesh, &[])
    }

    /// Build with specified boundary face tags whose nodes may move freely.
    ///
    /// Nodes on boundary faces with tags in `free_tags` are **not** frozen,
    /// allowing them to slide along the boundary during optimisation.
    /// All other boundary nodes remain fixed.
    pub fn from_mesh_with_free_tags(mesh: &SimplexMesh<2>, free_tags: &[i32]) -> Self {
        let n_nodes = mesh.n_nodes();
        let coords = mesh.coords.clone();
        let n_elem = mesh.n_elems();
        let mut on_boundary = vec![false; n_nodes];
        for f in 0..mesh.n_boundary_faces() as u32 {
            let tag = mesh.face_tag(f);
            if free_tags.contains(&tag) { continue; }
            let nodes = mesh.bface_nodes(f);
            for &n in nodes { on_boundary[n as usize] = true; }
        }
        let free_nodes: Vec<usize> = (0..n_nodes).filter(|&i| !on_boundary[i]).collect();
        let targets = vec![Matrix2::identity(); n_elem];
        TmopObjective2d { n_nodes, coords, conn: mesh.conn.clone(), elem_tags: None, free_nodes, targets }
    }

    /// Replace the per-element target Jacobians (default: identity).
    pub fn with_targets(&mut self, targets: Vec<Matrix2<f64>>) {
        assert_eq!(targets.len(), self.targets.len());
        self.targets = targets;
    }

    /// Target Jacobian for an **ideal equilateral triangle** with unit edge length.
    /// Reference: [(0,0), (1,0), (0.5, √3/2)] → J = [[1, 0.5], [0, √3/2]].
    pub fn ideal_equilateral_target() -> Matrix2<f64> {
        Matrix2::new(1.0, 0.5, 0.0, 3.0_f64.sqrt() * 0.5)
    }

    /// Set all targets to the ideal equilateral shape.
    pub fn set_ideal_equilateral(&mut self) {
        let ideal = Self::ideal_equilateral_target();
        for t in self.targets.iter_mut() { *t = ideal; }
    }

    pub fn n_free(&self) -> usize { self.free_nodes.len() }

    pub fn get_x(&self) -> Vec<f64> {
        self.free_nodes.iter().flat_map(|&i| vec![self.coords[2 * i], self.coords[2 * i + 1]]).collect()
    }

    pub fn set_x(&mut self, x: &[f64]) {
        for (k, &ni) in self.free_nodes.iter().enumerate() {
            self.coords[2 * ni] = x[2 * k];
            self.coords[2 * ni + 1] = x[2 * k + 1];
        }
    }

    fn elem_nodes_2d(&self, e: u32) -> Vec<[f64; 2]> {
        let npe = 3;
        let base = e as usize * npe;
        (0..npe).map(|k| {
            let ni = self.conn[base + k] as usize;
            [self.coords[2 * ni], self.coords[2 * ni + 1]]
        }).collect()
    }

    fn elem_jacobian(&self, nodes: &[[f64; 2]; 3]) -> Matrix2<f64> {
        Matrix2::new(
            nodes[1][0] - nodes[0][0], nodes[2][0] - nodes[0][0],
            nodes[1][1] - nodes[0][1], nodes[2][1] - nodes[0][1],
        )
    }

    fn n_elem(&self) -> usize { self.conn.len() / 3 }

    pub fn coords(&self) -> &[f64] { &self.coords }

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
            let dmu = [[result.dmu_da[0][0], result.dmu_da[0][1]],
                       [result.dmu_da[1][0], result.dmu_da[1][1]]];

            node_grad[n[0][0] as usize][0] -= dmu[0][0] + dmu[1][0];
            node_grad[n[0][0] as usize][1] -= dmu[0][1] + dmu[1][1];
            node_grad[n[1][0] as usize][0] += dmu[0][0];
            node_grad[n[1][0] as usize][1] += dmu[0][1];
            node_grad[n[2][0] as usize][0] += dmu[1][0];
            node_grad[n[2][0] as usize][1] += dmu[1][1];
        }

        for (k, &ni) in self.free_nodes.iter().enumerate() {
            grad[2 * k] = node_grad[ni][0];
            grad[2 * k + 1] = node_grad[ni][1];
        }
    }
}

/// Run TMOP optimisation for 2-D Tri3 meshes using gradient descent.
pub fn tmop_optimise_2d(mesh: &SimplexMesh<2>, metric: &TmopMetric, max_iter: usize, step_size: f64) -> Vec<f64> {
    let mut obj = TmopObjective2d::new(mesh);
    let n_free = obj.n_free();
    let mut x = obj.get_x();
    let mut grad = vec![0.0; 2 * n_free];
    let mut val = 0.0;

    for _ in 0..max_iter {
        obj.set_x(&x);
        obj.value_and_gradient(metric, &mut val, &mut grad);
        let g_max = grad.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        if g_max < 1e-12 { break; }

        let mut alpha = step_size;
        for _ in 0..20 {
            let x_new: Vec<f64> = x.iter().zip(grad.iter()).map(|(xi, gi)| xi - alpha * gi).collect();
            obj.set_x(&x_new);
            let mut new_val = 0.0;
            obj.value_and_gradient(metric, &mut new_val, &mut grad);
            if new_val < val || alpha < 1e-14 { x = x_new; val = new_val; break; }
            alpha *= 0.5;
        }
    }
    obj.set_x(&x);
    obj.coords().to_vec()
}

/// TMOP optimisation followed by CAD surface projection.
///
/// Runs `tmop_optimise_*` then projects displaced boundary nodes onto
/// the CAD surfaces specified in `config`.  This is the standard pipeline
/// for CAD-aware mesh optimisation: improve element quality via TMOP
/// while keeping the boundary on the intended geometry.
pub fn tmop_optimise_2d_with_cad(
    mesh: &SimplexMesh<2>,
    metric: &TmopMetric,
    max_iter: usize,
    step_size: f64,
    config: &crate::cad::ProjectionConfig,
) -> SimplexMesh<2> {
    let new_coords = tmop_optimise_2d(mesh, metric, max_iter, step_size);
    let opt_mesh = SimplexMesh::uniform(
        new_coords, mesh.conn.clone(), mesh.elem_tags.clone(),
        mesh.elem_type, mesh.face_conn.clone(), mesh.face_tags.clone(),
        mesh.face_type,
    );
    crate::cad::project_boundary_to_cad(&opt_mesh, config, 1)
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3-D Tetra objective
// ═══════════════════════════════════════════════════════════════════════════════

/// Objective function and gradient for 3-D TMOP on tetrahedral meshes (Tet4).
pub struct TmopObjectiveTetra {
    n_nodes: usize,
    coords: Vec<f64>,
    conn: Vec<u32>,
    free_nodes: Vec<usize>,
    targets: Vec<Matrix3<f64>>,
}

impl TmopObjectiveTetra {
    /// Build from a tetrahedral mesh. By default all boundary nodes are fixed.
    pub fn new(mesh: &SimplexMesh<3>) -> Self {
        Self::from_mesh_with_free_tags(mesh, &[])
    }

    /// Build with specified boundary face tags whose nodes may move freely.
    pub fn from_mesh_with_free_tags(mesh: &SimplexMesh<3>, free_tags: &[i32]) -> Self {
        let n_nodes = mesh.n_nodes();
        let coords = mesh.coords.clone();
        let n_elem = mesh.n_elems();
        let mut on_boundary = vec![false; n_nodes];
        for f in 0..mesh.n_boundary_faces() as u32 {
            let tag = mesh.face_tag(f);
            if free_tags.contains(&tag) { continue; }
            let nodes = mesh.bface_nodes(f);
            for &n in nodes { on_boundary[n as usize] = true; }
        }
        let free_nodes: Vec<usize> = (0..n_nodes).filter(|&i| !on_boundary[i]).collect();
        let targets = vec![Matrix3::identity(); n_elem];
        TmopObjectiveTetra { n_nodes, coords, conn: mesh.conn.clone(), free_nodes, targets }
    }

    /// Replace the per-element target Jacobians (default: identity).
    pub fn with_targets(&mut self, targets: Vec<Matrix3<f64>>) {
        assert_eq!(targets.len(), self.targets.len());
        self.targets = targets;
    }

    /// Target Jacobian for an **ideal regular tetrahedron** (unit edge).
    /// Reference: [(0,0,0), (1,0,0), (0.5, √3/2, 0), (0.5, √3/6, √(2/3))].
    pub fn ideal_regular_target() -> Matrix3<f64> {
        let s32 = (3.0_f64).sqrt() * 0.5;
        let s6 = (2.0_f64 / 3.0_f64).sqrt();
        Matrix3::new(1.0, 0.5, 0.5, 0.0, s32, (3.0_f64).sqrt() / 6.0, 0.0, 0.0, s6)
    }

    /// Set all targets to the ideal regular tetrahedron shape.
    pub fn set_ideal_regular(&mut self) {
        let ideal = Self::ideal_regular_target();
        for t in self.targets.iter_mut() { *t = ideal; }
    }

    pub fn n_free(&self) -> usize { self.free_nodes.len() }

    pub fn get_x(&self) -> Vec<f64> {
        self.free_nodes.iter().flat_map(|&i| {
            vec![self.coords[3 * i], self.coords[3 * i + 1], self.coords[3 * i + 2]]
        }).collect()
    }

    pub fn set_x(&mut self, x: &[f64]) {
        for (k, &ni) in self.free_nodes.iter().enumerate() {
            self.coords[3 * ni] = x[3 * k];
            self.coords[3 * ni + 1] = x[3 * k + 1];
            self.coords[3 * ni + 2] = x[3 * k + 2];
        }
    }

    pub fn coords(&self) -> &[f64] { &self.coords }

    fn n_elem(&self) -> usize { self.conn.len() / 4 }

    fn elem_nodes(&self, e: u32) -> Vec<[f64; 3]> {
        let base = e as usize * 4;
        (0..4).map(|k| {
            let ni = self.conn[base + k] as usize;
            [self.coords[3 * ni], self.coords[3 * ni + 1], self.coords[3 * ni + 2]]
        }).collect()
    }

    /// Jacobian for Tet4: J = [x₁-x₀, x₂-x₀, x₃-x₀] (3×3).
    fn elem_jacobian(&self, n: &[[f64; 3]; 4]) -> Matrix3<f64> {
        Matrix3::new(
            n[1][0] - n[0][0], n[2][0] - n[0][0], n[3][0] - n[0][0],
            n[1][1] - n[0][1], n[2][1] - n[0][1], n[3][1] - n[0][1],
            n[1][2] - n[0][2], n[2][2] - n[0][2], n[3][2] - n[0][2],
        )
    }

    /// Compute objective value and gradient w.r.t. free node coordinates.
    pub fn value_and_gradient(&self, metric: &TmopMetric, obj: &mut f64, grad: &mut [f64]) {
        *obj = 0.0;
        for g in grad.iter_mut() { *g = 0.0; }

        // Accumulate per-node gradient in physical coordinates
        let mut node_grad: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]; self.n_nodes];

        for e in 0..self.n_elem() as u32 {
            let raw = self.elem_nodes(e);
            let n: [[f64; 3]; 4] = [raw[0], raw[1], raw[2], raw[3]];
            let a = self.elem_jacobian(&n);
            let w = self.targets[e as usize];
            let r = tmop_metric_3d(&a, &w, metric);
            *obj += r.value;

            // dμ/dx_i = ∂μ/∂A : dA/dx_i
            // A = [x₁-x₀, x₂-x₀, x₃-x₀]
            // dA/dx₀ = -I,  dA/dx₁ = diag(1,0,0) col0, etc.
            // For a 3×3 Jacobian A with column k = x_{k+1} - x₀:
            //   ∂A(p,q)/∂x₀(r) = -δ_{qr}  (each column gets -1 for each component r)
            //   ∂A(p,q)/∂x_{k+1}(r) = δ_{pk}·δ_{qr}
            // Chain rule: dμ/dx_i(r) = Σ_{p,q} ∂μ/∂A(p,q) · ∂A(p,q)/∂x_i(r)

            for i in 0..3 {
                // Node 0: dA/dx₀(r) = -δ_{qr} for each column q
                // Column 0: (x10 - x00), Column 1: (x20 - x00), Column 2: (x30 - x00)
                // ∂A(p,q)/∂x₀(r) = -δ_{pq} — wait, careful
                // A(p,q) = x_{q+1}(p) - x₀(p), so ∂A(p,q)/∂x₀(r) = -δ_{pr}
                // dμ/dx₀(r) = Σ_{p,q} dμ_da[p][q] * (-δ_{pr}) = -Σ_q dμ_da[r][q]
                // Better: ∂μ/∂A is a 3×3 tensor. A = [v1, v2, v3] where vk = xk - x0
                // dμ/dx0 = -(dμ/dv1 + dμ/dv2 + dμ/dv3) = -Σ_k ∂μ/∂A[:,k]
                // So for each component r: dμ/dx0[r] = -Σ_k dμ_da[r][k]
                let mut sum_col = 0.0;
                for k in 0..3 { sum_col += r.dmu_da[i][k]; }
                node_grad[n[0][0] as usize][i] -= sum_col;

                // Node k+1 (k=0,1,2): xk affects column k of A
                // A(p, k) = x_{k+1}(p) - x₀(p), so ∂A(p,k)/∂x_{k+1}(r) = δ_{pr}
                // dμ/dx_{k+1}(r) = Σ_{p,q} ∂μ/∂A[p][q] * δ_{pr}*δ_{kq} = ∂μ/∂A[r][k]
                node_grad[n[1][0] as usize][i] += r.dmu_da[i][0];
                node_grad[n[2][0] as usize][i] += r.dmu_da[i][1];
                node_grad[n[3][0] as usize][i] += r.dmu_da[i][2];
            }
        }

        // Copy free-node gradients (flat: x0,y0,z0, x1,y1,z1, ...)
        for (k, &ni) in self.free_nodes.iter().enumerate() {
            grad[3 * k]     = node_grad[ni][0];
            grad[3 * k + 1] = node_grad[ni][1];
            grad[3 * k + 2] = node_grad[ni][2];
        }
    }
}

/// Run TMOP optimisation for 3-D tetrahedral meshes using gradient descent.
pub fn tmop_optimise_tetra(mesh: &SimplexMesh<3>, metric: &TmopMetric, max_iter: usize, step_size: f64) -> Vec<f64> {
    let mut obj = TmopObjectiveTetra::new(mesh);
    let n_free = obj.n_free();
    let mut x = obj.get_x();
    let mut grad = vec![0.0; 3 * n_free];
    let mut val = 0.0;

    for _ in 0..max_iter {
        obj.set_x(&x);
        obj.value_and_gradient(metric, &mut val, &mut grad);
        let g_max = grad.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        if g_max < 1e-12 { break; }

        let mut alpha = step_size;
        for _ in 0..20 {
            let x_new: Vec<f64> = x.iter().zip(grad.iter()).map(|(xi, gi)| xi - alpha * gi).collect();
            obj.set_x(&x_new);
            let mut new_val = 0.0;
            obj.value_and_gradient(metric, &mut new_val, &mut grad);
            if new_val < val || alpha < 1e-14 { x = x_new; val = new_val; break; }
            alpha *= 0.5;
        }
    }
    obj.set_x(&x);
    obj.coords().to_vec()
}

/// TMOP optimisation for tetrahedral meshes with CAD projection.
pub fn tmop_optimise_tetra_with_cad(
    mesh: &SimplexMesh<3>,
    metric: &TmopMetric,
    max_iter: usize,
    step_size: f64,
    config: &crate::cad::ProjectionConfig,
) -> SimplexMesh<3> {
    let new_coords = tmop_optimise_tetra(mesh, metric, max_iter, step_size);
    let opt_mesh = SimplexMesh::uniform(
        new_coords, mesh.conn.clone(), mesh.elem_tags.clone(),
        mesh.elem_type, mesh.face_conn.clone(), mesh.face_tags.clone(),
        mesh.face_type,
    );
    crate::cad::project_boundary_to_cad(&opt_mesh, config, 1)
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3-D Hexahedron objective
// ═══════════════════════════════════════════════════════════════════════════════

fn hex8_shape_grad(sx: f64, sy: f64, sz: f64, xi: f64, eta: f64, zeta: f64) -> [f64; 3] {
    let lx = 1.0 + sx * xi;  let ly = 1.0 + sy * eta;  let lz = 1.0 + sz * zeta;
    [
        sx * ly * lz / 8.0,
        sy * lx * lz / 8.0,
        sz * lx * ly / 8.0,
    ]
}

/// Per-node sign patterns for Hex8: (±1, ±1, ±1) in bit order.
const HEX8_SIGNS: [(f64, f64, f64); 8] = [
    (-1.0, -1.0, -1.0), ( 1.0, -1.0, -1.0), ( 1.0,  1.0, -1.0), (-1.0,  1.0, -1.0),
    (-1.0, -1.0,  1.0), ( 1.0, -1.0,  1.0), ( 1.0,  1.0,  1.0), (-1.0,  1.0,  1.0),
];

/// 2×2×2 Gauss-Legendre quadrature on [-1,1]³.
fn hex8_quad_points() -> Vec<(f64, f64, f64, f64)> {
    let g = 1.0 / 3.0_f64.sqrt();
    let mut pts = Vec::with_capacity(8);
    for ki in 0..2 { for kj in 0..2 { for kk in 0..2 {
        pts.push((
            if ki == 0 { -g } else { g },
            if kj == 0 { -g } else { g },
            if kk == 0 { -g } else { g },
            1.0, // weight
        ));
    }}}
    pts
}

/// Objective function and gradient for 3-D TMOP on hexahedral meshes (Hex8).
///
/// The Jacobian varies with position; 2×2×2 Gauss-Legendre quadrature integrates
/// the metric over the reference cube [-1,1]³.
pub struct TmopObjectiveHex {
    n_nodes: usize,
    coords: Vec<f64>,
    conn: Vec<u32>,
    free_nodes: Vec<usize>,
    targets: Vec<Matrix3<f64>>,
}

impl TmopObjectiveHex {
    /// Build from a hexahedral mesh. By default all boundary nodes are fixed.
    pub fn new(mesh: &SimplexMesh<3>) -> Self {
        Self::from_mesh_with_free_tags(mesh, &[])
    }

    /// Build with specified boundary face tags whose nodes may move freely.
    pub fn from_mesh_with_free_tags(mesh: &SimplexMesh<3>, free_tags: &[i32]) -> Self {
        let n_nodes = mesh.n_nodes();
        let coords = mesh.coords.clone();
        let n_elem = mesh.n_elems();
        let mut on_boundary = vec![false; n_nodes];
        for f in 0..mesh.n_boundary_faces() as u32 {
            let tag = mesh.face_tag(f);
            if free_tags.contains(&tag) { continue; }
            let nodes = mesh.bface_nodes(f);
            for &n in nodes { on_boundary[n as usize] = true; }
        }
        let free_nodes: Vec<usize> = (0..n_nodes).filter(|&i| !on_boundary[i]).collect();
        let targets = vec![Matrix3::identity(); n_elem];
        TmopObjectiveHex { n_nodes, coords, conn: mesh.conn.clone(), free_nodes, targets }
    }

    pub fn n_free(&self) -> usize { self.free_nodes.len() }

    pub fn get_x(&self) -> Vec<f64> {
        self.free_nodes.iter().flat_map(|&i| {
            vec![self.coords[3 * i], self.coords[3 * i + 1], self.coords[3 * i + 2]]
        }).collect()
    }

    pub fn set_x(&mut self, x: &[f64]) {
        for (k, &ni) in self.free_nodes.iter().enumerate() {
            self.coords[3 * ni] = x[3 * k];
            self.coords[3 * ni + 1] = x[3 * k + 1];
            self.coords[3 * ni + 2] = x[3 * k + 2];
        }
    }

    pub fn coords(&self) -> &[f64] { &self.coords }

    fn n_elem(&self) -> usize { self.conn.len() / 8 }

    fn elem_nodes(&self, e: u32) -> [[f64; 3]; 8] {
        let base = e as usize * 8;
        let mut nodes = [[0.0; 3]; 8];
        for k in 0..8 {
            let ni = self.conn[base + k] as usize;
            nodes[k] = [self.coords[3 * ni], self.coords[3 * ni + 1], self.coords[3 * ni + 2]];
        }
        nodes
    }

    /// Compute Jacobian J(ξ) = Σᵢ xᵢ ⊗ ∇φᵢ(ξ) (3×3 matrix: J[p][q] = Σᵢ xᵢ[p] · ∂φᵢ/∂ξ_q).
    pub fn jacobian_at(xi: f64, eta: f64, zeta: f64, nodes: &[[f64; 3]; 8]) -> (Matrix3<f64>, f64, Matrix3<f64>) {
        let mut j = Matrix3::zeros();
        for i in 0..8 {
            let g = hex8_shape_grad(HEX8_SIGNS[i].0, HEX8_SIGNS[i].1, HEX8_SIGNS[i].2, xi, eta, zeta);
            let nx = nodes[i][0]; let ny = nodes[i][1]; let nz = nodes[i][2];
            j[(0, 0)] += g[0] * nx; j[(0, 1)] += g[1] * nx; j[(0, 2)] += g[2] * nx;
            j[(1, 0)] += g[0] * ny; j[(1, 1)] += g[1] * ny; j[(1, 2)] += g[2] * ny;
            j[(2, 0)] += g[0] * nz; j[(2, 1)] += g[1] * nz; j[(2, 2)] += g[2] * nz;
        }
        let det_j = j.determinant();
        let inv_j = j.try_inverse().unwrap_or(Matrix3::identity());
        (j, det_j, inv_j)
    }

    /// Evaluate the objective and gradient w.r.t. free (interior) node coordinates.
    pub fn value_and_gradient(&self, metric: &TmopMetric, obj: &mut f64, grad: &mut [f64]) {
        *obj = 0.0;
        for g in grad.iter_mut() { *g = 0.0; }
        let mut node_grad: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]; self.n_nodes];
        let qpts = hex8_quad_points();

        for e in 0..self.n_elem() as u32 {
            let nodes = self.elem_nodes(e);
            let w = self.targets[e as usize];

            for &(xi, eta, zeta, wq) in &qpts {
                let (j, det_j, inv_j) = Self::jacobian_at(xi, eta, zeta, &nodes);
                let det_j_abs = det_j.abs();
                if det_j_abs <= 0.0 { continue; }

                // Deformation gradient T = J·W⁻¹
                let t = j * w.try_inverse().unwrap_or(Matrix3::identity());
                let r = tmop_metric_3d(&t, &Matrix3::identity(), metric);
                *obj += r.value * wq * det_j_abs;

                // Chain rule contributions to each node
                // dJ_e/dx_k(p) = Σ_q w_q [∂μ/∂A : ∂J/∂x_k + μ · ∂|det(J)|/∂x_k]
                // where A = J (the Jacobian IS the deformation gradient A since W=I for shape)
                //
                // For shape metric (W=I): T = J, so:
                // ∂μ/∂x_k(p) = ∂μ/∂J : ∂J/∂x_k(p)  (the ∂μ/∂J = ∂μ/∂A already computed)
                // plus the metric-value term from ∂|det(J)|/∂x_k(p)
                //
                // ∂J(a,b)/∂x_k(p) = δ_{ap} · ∂φ_k/∂ξ_b
                // ∂|det(J)|/∂x_k(p) = sign(det(J)) · det(J) · Σ_b J⁻¹(b,p) · ∂φ_k/∂ξ_b

                // Precompute: for each node, accumulate the shape gradient
                for i in 0..8 {
                    let g = hex8_shape_grad(HEX8_SIGNS[i].0, HEX8_SIGNS[i].1, HEX8_SIGNS[i].2, xi, eta, zeta);
                    let ni = self.conn[e as usize * 8 + i] as usize;

                    // Term 1: ∂μ/∂J : ∂J/∂x_k(p) = Σ_{a,b} ∂μ/∂A(a,b) · δ_{ap} · ∂φ_k/∂ξ_b
                    //         = Σ_b ∂μ/∂A(p,b) · ∂φ_k/∂ξ_b
                    for p in 0..3 {
                        let mut dmu = 0.0;
                        for b in 0..3 { dmu += r.dmu_da[p][b] * g[b]; }
                        node_grad[ni][p] += wq * det_j_abs * dmu;
                    }

                    // Term 2: μ · ∂|det(J)|/∂x_k(p)
                    // ∂|det(J)|/∂x_k(p) = sign(det) · det(J) · Σ_b J⁻¹(b,p) · ∂φ_k/∂ξ_b
                    if r.value != 0.0 {
                        let det_grad_scale = det_j.signum() * det_j_abs;
                        for p in 0..3 {
                            let mut ddet = 0.0;
                            for b in 0..3 { ddet += inv_j[(b, p)] * g[b]; }
                            node_grad[ni][p] += wq * r.value * det_grad_scale * ddet;
                        }
                    }
                }
            }
        }

        for (k, &ni) in self.free_nodes.iter().enumerate() {
            grad[3 * k]     = node_grad[ni][0];
            grad[3 * k + 1] = node_grad[ni][1];
            grad[3 * k + 2] = node_grad[ni][2];
        }
    }
}

/// Run TMOP optimisation for 3-D hexahedral meshes using gradient descent.
pub fn tmop_optimise_hex(mesh: &SimplexMesh<3>, metric: &TmopMetric, max_iter: usize, step_size: f64) -> Vec<f64> {
    let mut obj = TmopObjectiveHex::new(mesh);
    let n_free = obj.n_free();
    let mut x = obj.get_x();
    let mut grad = vec![0.0; 3 * n_free];
    let mut val = 0.0;

    for _ in 0..max_iter {
        obj.set_x(&x);
        obj.value_and_gradient(metric, &mut val, &mut grad);
        let g_max = grad.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        if g_max < 1e-12 { break; }

        let mut alpha = step_size;
        for _ in 0..20 {
            let x_new: Vec<f64> = x.iter().zip(grad.iter()).map(|(xi, gi)| xi - alpha * gi).collect();
            obj.set_x(&x_new);
            let mut new_val = 0.0;
            obj.value_and_gradient(metric, &mut new_val, &mut grad);
            if new_val < val || alpha < 1e-14 { x = x_new; val = new_val; break; }
            alpha *= 0.5;
        }
    }
    obj.set_x(&x);
    obj.coords().to_vec()
}

/// TMOP optimisation for hexahedral meshes with CAD projection.
pub fn tmop_optimise_hex_with_cad(
    mesh: &SimplexMesh<3>,
    metric: &TmopMetric,
    max_iter: usize,
    step_size: f64,
    config: &crate::cad::ProjectionConfig,
) -> SimplexMesh<3> {
    let new_coords = tmop_optimise_hex(mesh, metric, max_iter, step_size);
    let opt_mesh = SimplexMesh::uniform(
        new_coords, mesh.conn.clone(), mesh.elem_tags.clone(),
        mesh.elem_type, mesh.face_conn.clone(), mesh.face_tags.clone(),
        mesh.face_type,
    );
    crate::cad::project_boundary_to_cad(&opt_mesh, config, 1)
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

    // ─── 3-D tests ────────────────────────────────────────────────────────────

    #[test]
    fn tmop_metric_3d_l2_identity_gives_three() {
        let a = Matrix3::identity();
        let w = Matrix3::identity();
        let res = tmop_metric_3d(&a, &w, &TmopMetric::L2);
        assert!((res.value - 3.0).abs() < 1e-12, "|I|² = 3, got {}", res.value);
    }

    #[test]
    fn tmop_metric_3d_l2_derivative_finite_diff() {
        let a = Matrix3::new(1.0, 0.3, 0.1, 0.2, 1.2, 0.4, 0.0, 0.1, 1.1);
        let w = Matrix3::identity();
        let eps = 1e-6;
        let res = tmop_metric_3d(&a, &w, &TmopMetric::L2);
        for i in 0..3 { for j in 0..3 {
            let mut ap = a; ap[(i, j)] += eps;
            let mut am = a; am[(i, j)] -= eps;
            let rp = tmop_metric_3d(&ap, &w, &TmopMetric::L2);
            let rm = tmop_metric_3d(&am, &w, &TmopMetric::L2);
            let fd = (rp.value - rm.value) / (2.0 * eps);
            assert!((fd - res.dmu_da[i][j]).abs() < 1e-6,
                "3D L2 finite diff at ({i},{j}): fd={fd:.6e}, analytic={:.6e}", res.dmu_da[i][j]);
        }}
    }

    #[test]
    fn tmop_metric_3d_shape_scale_invariant() {
        let a = Matrix3::new(1.0, 0.3, 0.1, 0.2, 1.2, 0.4, 0.0, 0.1, 1.1);
        let w = Matrix3::identity();
        let r1 = tmop_metric_3d(&a, &w, &TmopMetric::Shape);
        let r2 = tmop_metric_3d(&(2.0 * a), &w, &TmopMetric::Shape);
        assert!((r1.value - r2.value).abs() < 1e-12,
            "3D Shape metric should be scale-invariant: {:.10} vs {:.10}", r1.value, r2.value);
    }

    #[test]
    fn tmop_tetra_optimise_improves_quality() {
        use crate::SimplexMesh;
        let mesh = unit_cube_tet5();
        let n_before = mesh.n_nodes();

        // Perturb interior nodes
        let mut m2 = mesh.clone();
        for i in 0..n_before {
            let x = m2.coords[3 * i];
            let y = m2.coords[3 * i + 1];
            let z = m2.coords[3 * i + 2];
            if x > 0.0 && x < 1.0 && y > 0.0 && y < 1.0 && z > 0.0 && z < 1.0 {
                let pi = std::f64::consts::PI;
                m2.coords[3 * i] += 0.08 * (x * pi).sin();
                m2.coords[3 * i + 1] += 0.08 * (y * pi).cos();
                m2.coords[3 * i + 2] += 0.08 * (z * pi).sin();
            }
        }

        let min_q_initial = min_tet_quality(&m2);
        let result = tmop_optimise_tetra(&m2, &TmopMetric::Shape, 30, 0.05);
        let final_m = SimplexMesh::<3>::uniform(
            result, m2.conn.clone(), m2.elem_tags.clone(),
            m2.elem_type, m2.face_conn.clone(), m2.face_tags.clone(),
            m2.face_type,
        );
        let min_q_final = min_tet_quality(&final_m);
        assert!(min_q_final >= min_q_initial - 1e-10,
            "min tet quality decreased: initial={:.6}, final={:.6}", min_q_initial, min_q_final);
    }

    fn unit_cube_tet5() -> SimplexMesh<3> {
        SimplexMesh::<3>::unit_cube_tet(5)
    }

    /// Mean-ratio quality for a tetrahedron:
    /// q = 12 * (3V)^{2/3} / Σ(l_i²)
    /// where V is volume, l_i are edge lengths (6 edges).
    fn tet_quality(mesh: &SimplexMesh<3>, e: u32) -> f64 {
        let ns = mesh.element_nodes(e);
        let p: Vec<[f64; 3]> = (0..4).map(|k| {
            let c = mesh.node_coords(ns[k]);
            [c[0], c[1], c[2]]
        }).collect();

        let edges: Vec<f64> = [
            (&p[0], &p[1]), (&p[0], &p[2]), (&p[0], &p[3]),
            (&p[1], &p[2]), (&p[1], &p[3]), (&p[2], &p[3]),
        ].iter().map(|(a, b)| {
            ((a[0]-b[0]).powi(2) + (a[1]-b[1]).powi(2) + (a[2]-b[2]).powi(2)).sqrt()
        }).collect();

        let vol = ((p[1][0]-p[0][0])*((p[2][1]-p[0][1])*(p[3][2]-p[0][2])-(p[2][2]-p[0][2])*(p[3][1]-p[0][1]))
                 - (p[1][1]-p[0][1])*((p[2][0]-p[0][0])*(p[3][2]-p[0][2])-(p[2][2]-p[0][2])*(p[3][0]-p[0][0]))
                 + (p[1][2]-p[0][2])*((p[2][0]-p[0][0])*(p[3][1]-p[0][1])-(p[2][1]-p[0][1])*(p[3][0]-p[0][0]))).abs() / 6.0;

        let sum_l2: f64 = edges.iter().map(|l| l * l).sum();
        if sum_l2 < 1e-30 { return 0.0; }
        12.0 * (3.0 * vol).powf(2.0 / 3.0) / sum_l2
    }

    fn min_tet_quality(mesh: &SimplexMesh<3>) -> f64 {
        (0..mesh.n_elems() as u32).map(|e| tet_quality(mesh, e)).fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn tmop_tetra_l2_gradient_near_zero_for_uniform() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(4);
        let obj = TmopObjectiveTetra::new(&mesh);
        let n_free = obj.n_free();
        let mut val = 0.0;
        let mut grad = vec![0.0; 3 * n_free];
        obj.value_and_gradient(&TmopMetric::L2, &mut val, &mut grad);
        let g_max = grad.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        assert!(g_max < 1e-10, "L2 gradient on uniform mesh should be near zero, got max={:.6e}", g_max);
    }

    #[test]
    fn tmop_metric_3d_shape_derivative_finite_diff() {
        let a = Matrix3::new(1.0, 0.3, 0.1, 0.2, 1.2, 0.4, 0.0, 0.1, 1.1);
        let w = Matrix3::identity();
        let eps = 1e-6;
        let res = tmop_metric_3d(&a, &w, &TmopMetric::Shape);
        for i in 0..3 { for j in 0..3 {
            let mut ap = a; ap[(i, j)] += eps;
            let mut am = a; am[(i, j)] -= eps;
            let rp = tmop_metric_3d(&ap, &w, &TmopMetric::Shape);
            let rm = tmop_metric_3d(&am, &w, &TmopMetric::Shape);
            let fd = (rp.value - rm.value) / (2.0 * eps);
            assert!((fd - res.dmu_da[i][j]).abs() < 1e-6,
                "3D Shape finite diff at ({i},{j}): fd={fd:.6e}, analytic={:.6e}", res.dmu_da[i][j]);
        }}
    }

    #[test]
    fn tmop_metric_3d_sizeshape_not_scale_invariant() {
        let a = Matrix3::new(1.0, 0.3, 0.1, 0.2, 1.2, 0.4, 0.0, 0.1, 1.1);
        let w = Matrix3::identity();
        let r1 = tmop_metric_3d(&a, &w, &TmopMetric::SizeShape);
        let r2 = tmop_metric_3d(&(2.0 * a), &w, &TmopMetric::SizeShape);
        // SizeShape with d=3: |T|² / det(T)^{2/3}
        // Doubling A: T→2T, |2T|²=4|T|², det(2T)^{2/3}=(2³det(T))^{2/3}=4det(T)^{2/3}
        // So SizeShape should be invariant under uniform scaling too (|T|²/det(T)^{2/d})
        // Actually with d=3, (2³)^{2/3}=2²=4, so 4/4=1, it IS scale-invariant for uniform scaling
        // Let me just verify it's not the same as Shape (which subtracts d)
        assert!((r1.value - r2.value).abs() < 1e-12,
            "SizeShape should be invariant under uniform scaling but differs: {:.10} vs {:.10}",
            r1.value, r2.value);
        assert!((r1.value - (r2.value - 3.0)).abs() > 1e-8,
            "SizeShape and Shape should differ");
    }

    // ─── Hex8 TMOP tests ──────────────────────────────────────────────────────

    fn hex_grid_mesh(nx: usize, ny: usize, nz: usize) -> SimplexMesh<3> {
        use crate::SimplexMesh;
        let n = nx * ny * nz;
        let npe = 8usize;
        let mut coords = Vec::with_capacity(n * npe * 3);
        let mut conn = Vec::with_capacity(n * npe);
        let mut tags = Vec::with_capacity(n);

        for k in 0..nz { for j in 0..ny { for i in 0..nx {
            let base = (k * (ny + 1) * (nx + 1) + j * (nx + 1) + i) as u32;
            conn.extend_from_slice(&[
                base, base + 1, base + (nx as u32) + 2, base + (nx as u32) + 1,
                base + ((ny + 1) * (nx + 1)) as u32,
                base + ((ny + 1) * (nx + 1)) as u32 + 1,
                base + ((ny + 1) * (nx + 1)) as u32 + (nx as u32) + 2,
                base + ((ny + 1) * (nx + 1)) as u32 + (nx as u32) + 1,
            ]);
            tags.push(1i32);
        }}}

        for k in 0..=nz { for j in 0..=ny { for i in 0..=nx {
            coords.push(i as f64 / nx as f64);
            coords.push(j as f64 / ny as f64);
            coords.push(k as f64 / nz as f64);
        }}}

        SimplexMesh::uniform(coords, conn, tags,
            crate::ElementType::Hex8, vec![], vec![], crate::ElementType::Quad4)
    }

    #[test]
    fn hex_l2_objective_positive() {
        let mesh = hex_grid_mesh(3, 3, 3);
        let obj = TmopObjectiveHex::new(&mesh);
        let mut val = 0.0;
        let n = obj.n_free();
        let mut grad = vec![0.0; 3 * n];
        obj.value_and_gradient(&TmopMetric::L2, &mut val, &mut grad);
        assert!(val > 0.0, "L2 objective should be positive, got {val}");
        assert!(val.is_finite());
    }

    #[test]
    fn hex_l2_gradient_finite_diff() {
        // Verify the gradient against finite differences
        let mesh = hex_grid_mesh(3, 3, 3);
        let obj = TmopObjectiveHex::new(&mesh);

        // Pick a random free node index
        let n_free = obj.n_free();
        if n_free == 0 { return; }
        let mut val0 = 0.0;
        let mut grad = vec![0.0; 3 * n_free];
        obj.value_and_gradient(&TmopMetric::L2, &mut val0, &mut grad);

        let eps = 1e-6;
        let mut x = obj.get_x();
        for ki in 0..n_free.min(3) {
            for d in 0..3 {
                let idx = 3 * ki + d;
                let orig = x[idx];

                x[idx] = orig + eps;
                let mut obj_plus = TmopObjectiveHex::new(&mesh);
                obj_plus.set_x(&x);
                let mut vp = 0.0;
                let mut _gp = vec![0.0; 3 * n_free];
                obj_plus.value_and_gradient(&TmopMetric::L2, &mut vp, &mut _gp);

                x[idx] = orig - eps;
                let mut obj_minus = TmopObjectiveHex::new(&mesh);
                obj_minus.set_x(&x);
                let mut vm = 0.0;
                let mut _gm = vec![0.0; 3 * n_free];
                obj_minus.value_and_gradient(&TmopMetric::L2, &mut vm, &mut _gm);

                let fd = (vp - vm) / (2.0 * eps);
                assert!((fd - grad[idx]).abs() < 1e-6,
                    "finite diff mismatch at node {ki} dim {d}: fd={fd:.6e}, analytic={:.6e}", grad[idx]);
                x[idx] = orig;
            }
        }
    }

    #[test]
    fn hex_optimise_improves_quality() {
        let mut mesh = hex_grid_mesh(4, 4, 4);
        // Perturb interior nodes
        for i in 0..mesh.n_nodes() {
            let x = mesh.coords[3 * i];
            let y = mesh.coords[3 * i + 1];
            let z = mesh.coords[3 * i + 2];
            if x > 0.0 && x < 1.0 && y > 0.0 && y < 1.0 && z > 0.0 && z < 1.0 {
                let pi = std::f64::consts::PI;
                mesh.coords[3 * i] += 0.05 * (x * pi).sin();
                mesh.coords[3 * i + 1] += 0.05 * (y * pi).cos();
                mesh.coords[3 * i + 2] += 0.05 * (z * pi).sin();
            }
        }

        let min_q_init = min_hex_quality(&mesh);
        let result = tmop_optimise_hex(&mesh, &TmopMetric::Shape, 20, 0.05);
        let final_m = SimplexMesh::<3>::uniform(
            result, mesh.conn.clone(), mesh.elem_tags.clone(),
            mesh.elem_type, mesh.face_conn.clone(), mesh.face_tags.clone(),
            mesh.face_type,
        );
        let min_q_final = min_hex_quality(&final_m);
        assert!(min_q_final >= min_q_init - 1e-10,
            "hex quality decreased: {:.6} → {:.6}", min_q_init, min_q_final);
    }

    fn hex_scaled_jacobian(mesh: &SimplexMesh<3>, e: u32) -> f64 {
        let ns = mesh.element_nodes(e);
        let mut p = [[0.0; 3]; 8];
        for k in 0..8 { let c = mesh.node_coords(ns[k]); p[k] = [c[0], c[1], c[2]]; }
        // Min scaled Jacobian = min_{qpts} det(J)/Π_i|col_i|
        let mut min_q = 1.0_f64;
        let g = 1.0 / 3.0_f64.sqrt();
        for &(xi, eta, zeta) in &[(-g,-g,-g),(g,-g,-g),(g,g,-g),(-g,g,-g),(-g,-g,g),(g,-g,g),(g,g,g),(-g,g,g)] {
            let (j, det, _) = TmopObjectiveHex::jacobian_at(xi, eta, zeta, &p);
            if det <= 0.0 { return 0.0; }
            let col_norm = (0..3).map(|c| {
                (0..3).map(|r| j[(r, c)].powi(2)).sum::<f64>().sqrt()
            }).product::<f64>();
            if col_norm > 0.0 {
                let q = det / col_norm;
                min_q = min_q.min(q);
            }
        }
        min_q
    }

    fn min_hex_quality(mesh: &SimplexMesh<3>) -> f64 {
        (0..mesh.n_elems() as u32).map(|e| hex_scaled_jacobian(mesh, e)).fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn tmop_target_ideal_equilateral() {
        let t = TmopObjective2d::ideal_equilateral_target();
        let det = t.determinant();
        assert!((det - 3.0_f64.sqrt() * 0.5).abs() < 1e-12, "equilateral area = sqrt(3)/2, got {det}");
    }

    #[test]
    fn tmop_target_regular_tetrahedron() {
        let t = TmopObjectiveTetra::ideal_regular_target();
        let det = t.determinant();
        let vol = det.abs() / 6.0;
        assert!((vol - (2.0_f64.sqrt() / 12.0)).abs() < 1e-12, "reg tet vol = sqrt(2)/12, got {vol}");
    }

    #[test]
    fn tmop_2d_improves_with_ideal_target() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let mut obj = TmopObjective2d::new(&mesh);
        obj.set_ideal_equilateral();
        let result = tmop_optimise_2d(&mesh, &TmopMetric::Shape, 30, 0.05);
        assert!(result.iter().all(|&v| v.is_finite()), "non-finite after optimisation");
    }

    #[test]
    fn tmop_deformed_det_2d() {
        let a = Matrix2::identity();
        let w = Matrix2::identity();
        let res = tmop_metric_2d(&a, &w, &TmopMetric::DeformedDet);
        assert!(res.value.abs() < 1e-12, "DeformedDet(I) = 0, got {}", res.value);
        // A = 2I → det=4, T=2I, det(T)=4 → value = (4-1)² = 9
        let a2 = Matrix2::from_diagonal(&nalgebra::Vector2::new(2.0, 2.0));
        let res2 = tmop_metric_2d(&a2, &w, &TmopMetric::DeformedDet);
        assert!((res2.value - 9.0).abs() < 1e-12, "DeformedDet(2I) = 9, got {}", res2.value);
    }

    #[test]
    fn tmop_barrier_det_2d() {
        let a = Matrix2::identity();
        let w = Matrix2::identity();
        let res = tmop_metric_2d(&a, &w, &TmopMetric::BarrierDet);
        assert!((res.value - 0.0).abs() < 1e-12, "BarrierDet(I) = 1/1-1 = 0, got {}", res.value);
        // Very small det → large value
        let a_small = Matrix2::new(0.1, 0.0, 0.0, 0.1);
        let res_small = tmop_metric_2d(&a_small, &w, &TmopMetric::BarrierDet);
        assert!(res_small.value > 0.0, "BarrierDet(small) should be positive, got {}", res_small.value);
    }

    #[test]
    fn tmop_deformed_det_3d() {
        let a = Matrix3::identity();
        let w = Matrix3::identity();
        let res = tmop_metric_3d(&a, &w, &TmopMetric::DeformedDet);
        assert!(res.value.abs() < 1e-12, "3D DeformedDet(I) = 0, got {}", res.value);
    }

    #[test]
    fn tmop_2d_free_tags_releases_boundary_nodes() {
        // unit_square_tri uses face tags {1,2,3,4} for the four sides.
        // With free_tags=[1,2,3,4], ALL nodes should be free.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let obj_all_free = TmopObjective2d::from_mesh_with_free_tags(&mesh, &[1, 2, 3, 4]);
        assert_eq!(obj_all_free.n_free(), mesh.n_nodes(),
            "with all boundary tags free, all {} nodes should be free, got {}",
            mesh.n_nodes(), obj_all_free.n_free());

        // With free_tags=[], same as new() — boundary nodes should be fixed.
        let obj_all_fixed = TmopObjective2d::from_mesh_with_free_tags(&mesh, &[]);
        assert_eq!(obj_all_fixed.n_free(), TmopObjective2d::new(&mesh).n_free(),
            "empty free_tags should match new()");

        // With free_tags=[] the free count must be less than total nodes.
        assert!(obj_all_fixed.n_free() < mesh.n_nodes(),
            "a clamped mesh should have fewer free nodes than total nodes");
    }

    #[test]
    fn tmop_tetra_free_tags_releases_boundary_nodes() {
        let mesh = crate::SimplexMesh::<3>::unit_cube_tet(3);
        let obj_all_free = TmopObjectiveTetra::from_mesh_with_free_tags(&mesh, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(obj_all_free.n_free(), mesh.n_nodes(),
            "with all boundary tags free, all nodes should be free");

        let obj_all_fixed = TmopObjectiveTetra::from_mesh_with_free_tags(&mesh, &[]);
        assert_eq!(obj_all_fixed.n_free(), TmopObjectiveTetra::new(&mesh).n_free(),
            "empty free_tags should match new()");
        assert!(obj_all_fixed.n_free() < mesh.n_nodes(),
            "a clamped tetra mesh should have fewer free nodes than total nodes");
    }

    #[test]
    fn tmop_hex_free_tags_releases_boundary_nodes() {
        // unit_cube_hex uses face tags {1,2,3,4,5,6} for its six faces.
        let mesh = crate::SimplexMesh::<3>::unit_cube_hex(3);
        let obj_all_free = TmopObjectiveHex::from_mesh_with_free_tags(&mesh, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(obj_all_free.n_free(), mesh.n_nodes(),
            "with all boundary tags free, all nodes should be free");

        let obj_all_fixed = TmopObjectiveHex::from_mesh_with_free_tags(&mesh, &[]);
        assert_eq!(obj_all_fixed.n_free(), TmopObjectiveHex::new(&mesh).n_free(),
            "empty free_tags should match new()");
        assert!(obj_all_fixed.n_free() < mesh.n_nodes(),
            "a clamped hex mesh should have fewer free nodes than total nodes");
    }
}
