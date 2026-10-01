//! TMOP quality metric implementations.
//!
//! Each metric provides:
//! - `eval_w(jpt)` — invariant-form evaluation W(J)
//! - `eval_w_matrix_form(jpt)` — matrix-form evaluation (for validation)
//! - `eval_p(jpt, p)` — 1st Piola-Kirchhoff stress (dW/dJ)
//! - `assemble_h(jpt, ds, weight, a)` — 2nd derivative assembly
//!
//! All metrics follow MFEM's convention: J = Jpt = target→physical Jacobian.

use crate::tmop::ad::{ad_grad, ad_hessian, default_assemble_h, Ad1, Ad2, AdScalar};
use crate::tmop::invariants::{InvariantsEvaluator2D, InvariantsEvaluator3D};
use std::cell::RefCell;

/// Trait for TMOP quality metrics (ported from MFEM's TMOP_QualityMetric).
pub trait TmopQualityMetric {
    /// Evaluate the metric in invariant form W(J).
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64;

    /// Evaluate the metric in matrix form (used for validation against invariant form).
    fn eval_w_matrix_form(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        self.eval_w(jpt)
    }

    /// Evaluate the 1st Piola-Kirchhoff stress P = dW/dJ.
    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]);

    /// Assemble the 2nd derivative contribution into local matrix A.
    /// A has size (ndof*2) x (ndof*2), stored in column-major with block layout:
    /// A(i + ndof*j, k + ndof*l) for i,k in [0,ndof), j,l in [0,2).
    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]);

    /// MFEM `TMOP_QualityMetric::SetTargetJacobian`: specify the
    /// reference-element -> target-element Jacobian for the point of interest.
    /// Only target-dependent (A-)metrics use it; T-metrics ignore it (the
    /// default no-op mirrors a metric without the `Jtr` member).
    fn set_target_jacobian(&self, _jtr: &[[f64; 2]; 2]) {}

    /// Metric ID.
    fn id(&self) -> i32 {
        0
    }
}

/// Trait for 3D TMOP quality metrics.
pub trait TmopQualityMetric3D {
    /// Evaluate the metric in invariant form W(J).
    fn eval_w(&self, jpt: &[[f64; 3]; 3]) -> f64;

    /// Evaluate the metric in matrix form.
    fn eval_w_matrix_form(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        self.eval_w(jpt)
    }

    /// Evaluate the 1st Piola-Kirchhoff stress P = dW/dJ.
    fn eval_p(&self, jpt: &[[f64; 3]; 3], p: &mut [[f64; 3]; 3]);

    /// Assemble the 2nd derivative contribution into local matrix A.
    fn assemble_h(&self, jpt: &[[f64; 3]; 3], ds: &[[f64; 3]], weight: f64, a: &mut [f64]);

    /// Metric ID.
    fn id(&self) -> i32 {
        0
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Compute Frobenius norm squared of a 2x2 matrix.
fn fnorm2_2x2(m: &[[f64; 2]; 2]) -> f64 {
    m[0][0] * m[0][0] + m[1][0] * m[1][0] + m[0][1] * m[0][1] + m[1][1] * m[1][1]
}

/// Compute determinant of a 2x2 matrix.
fn det_2x2(m: &[[f64; 2]; 2]) -> f64 {
    m[0][0] * m[1][1] - m[1][0] * m[0][1]
}

/// Compute Frobenius norm squared of a 3x3 matrix.
fn fnorm2_3x3(m: &[[f64; 3]; 3]) -> f64 {
    let mut s = 0.0;
    for j in 0..3 {
        for i in 0..3 {
            s += m[i][j] * m[i][j];
        }
    }
    s
}

/// Compute determinant of a 3x3 matrix.
fn det_3x3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
        - m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2])
        + m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2])
}

/// Compute inverse transpose of a 3x3 matrix.
fn calc_inverse_transpose_3x3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = det_3x3(m);
    let inv_det = 1.0 / det;
    let mut inv_t = [[0.0; 3]; 3];
    // Inverse transpose = adjugate / det
    inv_t[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * inv_det;
    inv_t[1][0] = (m[2][1] * m[0][2] - m[0][1] * m[2][2]) * inv_det;
    inv_t[2][0] = (m[0][1] * m[1][2] - m[1][1] * m[0][2]) * inv_det;
    inv_t[0][1] = (m[2][0] * m[1][2] - m[1][0] * m[2][2]) * inv_det;
    inv_t[1][1] = (m[0][0] * m[2][2] - m[2][0] * m[0][2]) * inv_det;
    inv_t[2][1] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * inv_det;
    inv_t[0][2] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * inv_det;
    inv_t[1][2] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * inv_det;
    inv_t[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * inv_det;
    inv_t
}

/// Convert 2x2 matrix to column-major array.
fn to_col_major_2x2(m: &[[f64; 2]; 2]) -> [f64; 4] {
    [m[0][0], m[1][0], m[0][1], m[1][1]]
}

/// Convert 3x3 matrix to column-major array.
fn to_col_major_3x3(m: &[[f64; 3]; 3]) -> [f64; 9] {
    [
        m[0][0], m[1][0], m[2][0],
        m[0][1], m[1][1], m[2][1],
        m[0][2], m[1][2], m[2][2],
    ]
}

/// Convert column-major array to 2x2 matrix.
fn from_col_major_2x2(c: &[f64; 4]) -> [[f64; 2]; 2] {
    [[c[0], c[2]], [c[1], c[3]]]
}

/// Convert column-major array to 3x3 matrix.
fn from_col_major_3x3(c: &[f64; 9]) -> [[f64; 3]; 3] {
    [
        [c[0], c[3], c[6]],
        [c[1], c[4], c[7]],
        [c[2], c[5], c[8]],
    ]
}

// ============================================================================
// 2D Metrics
// ============================================================================

/// TMOP_Metric_001: W = |J|² (2D non-barrier, no type)
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric001;

impl TmopQualityMetric for TmopMetric001 {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        fnorm2_2x2(jpt)
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        // P = dI1 = 2*J
        p[0][0] = 2.0 * jpt[0][0];
        p[1][0] = 2.0 * jpt[1][0];
        p[0][1] = 2.0 * jpt[0][1];
        p[1][1] = 2.0 * jpt[1][1];
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_2x2(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_2d(ds));
        ie.assemble_dd_i1(weight, a);
    }

    fn id(&self) -> i32 {
        1
    }
}

/// TMOP_Metric_002: W = 0.5 * |J|² / det(J) - 1 (2D barrier shape, polyconvex)
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric002;

impl TmopQualityMetric for TmopMetric002 {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        0.5 * ie.get_i1b() - 1.0
    }

    fn eval_w_matrix_form(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        0.5 * fnorm2_2x2(jpt) / det_2x2(jpt) - 1.0
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        let di1b = ie.get_di1b().clone();
        *p = from_col_major_2x2(&scale_array(&di1b, 0.5));
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_2x2(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_2d(ds));
        ie.assemble_dd_i1b(0.5 * weight, a);
    }

    fn id(&self) -> i32 {
        2
    }
}

/// TMOP_Metric_007: W = |J - J^{-t}|² (2D barrier shape+size)
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric007;

impl TmopQualityMetric for TmopMetric007 {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        let i1 = ie.get_i1();
        let i2 = ie.get_i2();
        i1 * (1.0 + 1.0 / i2) - 4.0
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        let i2 = ie.get_i2();
        let i1 = ie.get_i1();
        let di1 = ie.get_di1().clone();
        let di2 = ie.get_di2().clone();
        // P = (1 + 1/I2) dI1 - I1/I2² dI2
        let c1 = 1.0 + 1.0 / i2;
        let c2 = -i1 / (i2 * i2);
        let mut result = [0.0; 4];
        for i in 0..4 {
            result[i] = c1 * di1[i] + c2 * di2[i];
        }
        *p = from_col_major_2x2(&result);
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_2x2(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_2d(ds));
        let i2 = ie.get_i2();
        let i1 = ie.get_i1();
        let c1 = 1.0 / i2;
        let c2 = weight * c1 * c1;
        let c3 = i1 * c2;
        let di1 = ie.get_di1().clone();
        let di2 = ie.get_di2().clone();
        ie.assemble_dd_i1(weight * (1.0 + c1), a);
        ie.assemble_dd_i2(-c3, a);
        ie.assemble_tprod_xy(-c2, &di1, &di2, a);
        ie.assemble_tprod_xx(2.0 * c1 * c3, &di2, a);
    }

    fn id(&self) -> i32 {
        7
    }
}

/// TMOP_Metric_009: W = det(J) * |J - J^{-t}|² (2D barrier shape+size)
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric009;

impl TmopQualityMetric for TmopMetric009 {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        let i1 = ie.get_i1();
        let i2b = ie.get_i2b();
        let i1b = ie.get_i1b();
        (i1 - 4.0) * i2b + i1b
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        let i1 = ie.get_i1();
        let i2b = ie.get_i2b();
        let di1 = ie.get_di1().clone();
        let di2b = ie.get_di2b().clone();
        let di1b = ie.get_di1b().clone();
        // P = (I1 - 4) dI2b + I2b dI1 + dI1b
        let mut result = [0.0; 4];
        for i in 0..4 {
            result[i] = (i1 - 4.0) * di2b[i] + i2b * di1[i] + di1b[i];
        }
        *p = from_col_major_2x2(&result);
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_2x2(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_2d(ds));
        let i1 = ie.get_i1();
        let i2b = ie.get_i2b();
        let di1 = ie.get_di1().clone();
        let di2b = ie.get_di2b().clone();
        ie.assemble_tprod_xy(weight, &di1, &di2b, a);
        ie.assemble_dd_i2b(weight * (i1 - 4.0), a);
        ie.assemble_dd_i1(weight * i2b, a);
        ie.assemble_dd_i1b(weight, a);
    }

    fn id(&self) -> i32 {
        9
    }
}

/// TMOP_Metric_014: W = |J - I|² (2D non-barrier shape+size+orientation, polyconvex)
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric014;

impl TmopQualityMetric for TmopMetric014 {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        // W = |J - I|² = I1[J-I]
        let mut mat = *jpt;
        mat[0][0] -= 1.0;
        mat[1][1] -= 1.0;
        let jac = to_col_major_2x2(&mat);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        ie.get_i1()
    }

    fn eval_w_matrix_form(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        let mut mat = *jpt;
        mat[0][0] -= 1.0;
        mat[1][1] -= 1.0;
        fnorm2_2x2(&mat)
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        let mut jpt_minus_id = *jpt;
        jpt_minus_id[0][0] -= 1.0;
        jpt_minus_id[1][1] -= 1.0;
        let jac = to_col_major_2x2(&jpt_minus_id);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        let di1 = ie.get_di1().clone();
        *p = from_col_major_2x2(&di1);
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        let mut jpt_minus_id = *jpt;
        jpt_minus_id[0][0] -= 1.0;
        jpt_minus_id[1][1] -= 1.0;
        let jac = to_col_major_2x2(&jpt_minus_id);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_2d(ds));
        ie.assemble_dd_i1(weight, a);
    }

    fn id(&self) -> i32 {
        14
    }
}

/// TMOP_Metric_022: W = 0.5(|J|² - 2det(J)) / (det(J) - tau0) (2D shifted barrier)
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric022 {
    pub min_det_t: f64,
}

impl TmopQualityMetric for TmopMetric022 {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        let i1 = ie.get_i1();
        let i2b = ie.get_i2b();
        let mut d = i2b - self.min_det_t;
        if d < 0.0 && self.min_det_t == 0.0 {
            d = -i2b * 0.1;
        }
        (0.5 * i1 - i2b) / d
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        let i1 = ie.get_i1();
        let i2b = ie.get_i2b();
        let c1 = 1.0 / (i2b - self.min_det_t);
        let di1 = ie.get_di1().clone();
        let di2b = ie.get_di2b().clone();
        // P = 0.5/(I2b - tau0) dI1 + (tau0 - 0.5*I1)/(I2b - tau0)² dI2b
        let c2 = (self.min_det_t - i1 / 2.0) * c1 * c1;
        let mut result = [0.0; 4];
        for i in 0..4 {
            result[i] = c1 / 2.0 * di1[i] + c2 * di2b[i];
        }
        *p = from_col_major_2x2(&result);
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_2x2(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_2d(ds));
        let i1 = ie.get_i1();
        let i2b = ie.get_i2b();
        let c1 = 1.0 / (i2b - self.min_det_t);
        let c2 = weight * c1 / 2.0;
        let c3 = c1 * c2;
        let c4 = (2.0 * self.min_det_t - i1) * c3;
        let di1 = ie.get_di1().clone();
        let di2b = ie.get_di2b().clone();
        ie.assemble_tprod_xy(-c3, &di1, &di2b, a);
        ie.assemble_tprod_xx(-2.0 * c1 * c4, &di2b, a);
        ie.assemble_dd_i1(c2, a);
        ie.assemble_dd_i2b(c4, a);
    }

    fn id(&self) -> i32 {
        22
    }
}

/// TMOP_Metric_050: W = 0.5 |J^t J|² / det(J)² - 1 (2D barrier shape, polyconvex)
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric050;

impl TmopQualityMetric for TmopMetric050 {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        let i1b = ie.get_i1b();
        0.5 * i1b * i1b - 2.0
    }

    fn eval_w_matrix_form(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        // W = 0.5 * |J^t J|² / det(J)² - 1
        let jt_j = [
            [
                jpt[0][0] * jpt[0][0] + jpt[1][0] * jpt[1][0],
                jpt[0][0] * jpt[0][1] + jpt[1][0] * jpt[1][1],
            ],
            [
                jpt[0][1] * jpt[0][0] + jpt[1][1] * jpt[1][0],
                jpt[0][1] * jpt[0][1] + jpt[1][1] * jpt[1][1],
            ],
        ];
        let det = det_2x2(jpt);
        0.5 * fnorm2_2x2(&jt_j) / (det * det) - 1.0
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        let i1b = ie.get_i1b();
        let di1b = ie.get_di1b().clone();
        // P = I1b * dI1b
        *p = from_col_major_2x2(&scale_array(&di1b, i1b));
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_2x2(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_2d(ds));
        let i1b = ie.get_i1b();
        let di1b = ie.get_di1b().clone();
        ie.assemble_tprod_xx(weight, &di1b, a);
        ie.assemble_dd_i1b(weight * i1b, a);
    }

    fn id(&self) -> i32 {
        50
    }
}

/// TMOP_Metric_055: W = (det(J) - 1)² (2D non-barrier size)
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric055;

impl TmopQualityMetric for TmopMetric055 {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        let c1 = ie.get_i2b() - 1.0;
        c1 * c1
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        let i2b = ie.get_i2b();
        let di2b = ie.get_di2b().clone();
        // P = 2*(I2b - 1) dI2b
        *p = from_col_major_2x2(&scale_array(&di2b, 2.0 * (i2b - 1.0)));
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_2x2(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_2d(ds));
        let i2b = ie.get_i2b();
        let di2b = ie.get_di2b().clone();
        ie.assemble_tprod_xx(2.0 * weight, &di2b, a);
        ie.assemble_dd_i2b(2.0 * weight * (i2b - 1.0), a);
    }

    fn id(&self) -> i32 {
        55
    }
}

/// TMOP_Metric_056: W = 0.5 (det(J) + 1/det(J)) - 1 (2D barrier size, polyconvex)
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric056;

impl TmopQualityMetric for TmopMetric056 {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        let i2b = ie.get_i2b();
        0.5 * (i2b + 1.0 / i2b) - 1.0
    }

    fn eval_w_matrix_form(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        let d = det_2x2(jpt);
        0.5 * (d + 1.0 / d) - 1.0
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        let i2 = ie.get_i2();
        let di2b = ie.get_di2b().clone();
        // P = (0.5 - 0.5/I2) dI2b
        *p = from_col_major_2x2(&scale_array(&di2b, 0.5 - 0.5 / i2));
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_2x2(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_2d(ds));
        let i2 = ie.get_i2();
        let i2b = ie.get_i2b();
        let di2b = ie.get_di2b().clone();
        ie.assemble_tprod_xx(weight / (i2 * i2b), &di2b, a);
        ie.assemble_dd_i2b(weight * (0.5 - 0.5 / i2), a);
    }

    fn id(&self) -> i32 {
        56
    }
}

/// TMOP_Metric_058: W = |J^t J|² / det(J)² - 2|J|² / det(J) + 2 (2D barrier shape)
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric058;

impl TmopQualityMetric for TmopMetric058 {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        let i1b = ie.get_i1b();
        i1b * (i1b - 2.0)
    }

    fn eval_w_matrix_form(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        let jt_j = [
            [
                jpt[0][0] * jpt[0][0] + jpt[1][0] * jpt[1][0],
                jpt[0][0] * jpt[0][1] + jpt[1][0] * jpt[1][1],
            ],
            [
                jpt[0][1] * jpt[0][0] + jpt[1][1] * jpt[1][0],
                jpt[0][1] * jpt[0][1] + jpt[1][1] * jpt[1][1],
            ],
        ];
        let det = det_2x2(jpt);
        fnorm2_2x2(&jt_j) / (det * det) - 2.0 * fnorm2_2x2(jpt) / det + 2.0
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        let i1b = ie.get_i1b();
        let di1b = ie.get_di1b().clone();
        // P = (2*I1b - 2) dI1b
        *p = from_col_major_2x2(&scale_array(&di1b, 2.0 * i1b - 2.0));
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_2x2(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_2d(ds));
        let i1b = ie.get_i1b();
        let di1b = ie.get_di1b().clone();
        ie.assemble_tprod_xx(2.0 * weight, &di1b, a);
        ie.assemble_dd_i1b(weight * (2.0 * i1b - 2.0), a);
    }

    fn id(&self) -> i32 {
        58
    }
}

/// TMOP_Metric_094: balanced 2D shape+size combo, `mu_2 + 1.5 mu_56`
/// (MFEM `TMOP_Metric_094 : public TMOP_Combo_QualityMetric`, `tmop.hpp`).
///
/// The C++ constructor is `AddQualityMetric(new TMOP_Metric_002, 1.0)` +
/// `AddQualityMetric(new TMOP_Metric_056, 1.5)`, and the combo's `EvalW` /
/// `EvalP` / `AssembleH` are the weighted sums of the parts
/// (`TMOP_Combo_QualityMetric::*`, `tmop.cpp`). `SetTargetJacobian` broadcasts
/// to both parts, which for these two stateless metrics is a no-op here since
/// the callers apply the target Jacobian to `Jpt` before calling.
///
/// Note the driver-level `-bec` (``TMOP_Combo_QualityMetric::
/// ComputeBalancedWeights``) is *not* applied by default, so the weights stay
/// `(1.0, 1.5)` (`miniapps/meshing/mesh-optimizer.cpp`: `bal_expl_combo =
/// false`).
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric094 {
    /// `sh_metric` (weight 1.0).
    sh: TmopMetric002,
    /// `sz_metric` (weight 1.5).
    sz: TmopMetric056,
}

impl Default for TmopMetric094 {
    fn default() -> Self {
        Self::new()
    }
}

impl TmopMetric094 {
    /// MFEM `TMOP_Metric_094::TMOP_Metric_094`: `mu_2 + lambda mu_56` with
    /// `lambda = 1.5` ("1 <= lambda <= 2 should produce best asymptotic
    /// balance").
    pub const SZ_WEIGHT: f64 = 1.5;
    /// Weight of `mu_2` (`AddQualityMetric(sh_metric, 1.0)`).
    pub const SH_WEIGHT: f64 = 1.0;

    pub fn new() -> Self {
        Self {
            sh: TmopMetric002,
            sz: TmopMetric056,
        }
    }
}

impl TmopQualityMetric for TmopMetric094 {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        Self::SH_WEIGHT * self.sh.eval_w(jpt) + Self::SZ_WEIGHT * self.sz.eval_w(jpt)
    }

    fn eval_w_matrix_form(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        Self::SH_WEIGHT * self.sh.eval_w_matrix_form(jpt)
            + Self::SZ_WEIGHT * self.sz.eval_w_matrix_form(jpt)
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        let mut pt = [[0.0_f64; 2]; 2];
        self.sh.eval_p(jpt, &mut pt);
        for r in 0..2 {
            for c in 0..2 {
                p[r][c] = Self::SH_WEIGHT * pt[r][c];
            }
        }
        self.sz.eval_p(jpt, &mut pt);
        for r in 0..2 {
            for c in 0..2 {
                p[r][c] += Self::SZ_WEIGHT * pt[r][c];
            }
        }
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        // C++ combo: `AssembleH(Jpt, DS, weight*wt, At); A += At;` — both parts
        // accumulate into the same matrix in fem-rs, so the weights fold into
        // the single `weight` argument.
        self.sh.assemble_h(jpt, ds, weight * Self::SH_WEIGHT, a);
        self.sz.assemble_h(jpt, ds, weight * Self::SZ_WEIGHT, a);
    }

    fn id(&self) -> i32 {
        94
    }
}

/// TMOP_Metric_077: W = 0.5 (det(J)² + 1/det(J)²) - 1 (2D barrier size, polyconvex)
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric077;

impl TmopQualityMetric for TmopMetric077 {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        let i2 = ie.get_i2();
        0.5 * (i2 + 1.0 / i2) - 1.0
    }

    fn eval_w_matrix_form(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        let d = det_2x2(jpt);
        0.5 * (d * d + 1.0 / (d * d)) - 1.0
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        let i2 = ie.get_i2();
        let di2 = ie.get_di2().clone();
        // P = 0.5*(1 - 1/I2²) dI2
        *p = from_col_major_2x2(&scale_array(&di2, 0.5 * (1.0 - 1.0 / (i2 * i2))));
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_2x2(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_2d(ds));
        let i2 = ie.get_i2();
        let i2inv_sq = 1.0 / (i2 * i2);
        let di2 = ie.get_di2().clone();
        ie.assemble_dd_i2(weight * 0.5 * (1.0 - i2inv_sq), a);
        ie.assemble_tprod_xx(weight * i2inv_sq / i2, &di2, a);
    }

    fn id(&self) -> i32 {
        77
    }
}

// ============================================================================
// 2D metrics (round-102 additions)
// ============================================================================

/// TMOP_Metric_000: W = 0 (the zero metric; MFEM defines it inline in
/// `fem/tmop.hpp`: `EvalW = 0`, `EvalP = 0`, `AssembleH` = A = 0).
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric000;

impl TmopQualityMetric for TmopMetric000 {
    fn eval_w(&self, _jpt: &[[f64; 2]; 2]) -> f64 {
        0.0
    }

    fn eval_p(&self, _jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        // P = 0.0
        *p = [[0.0; 2]; 2];
    }

    fn assemble_h(&self, _jpt: &[[f64; 2]; 2], _ds: &[[f64; 2]], _weight: f64, a: &mut [f64]) {
        // A = 0.0
        for v in a.iter_mut() {
            *v = 0.0;
        }
    }

    fn id(&self) -> i32 {
        0
    }
}

/// TMOP_Metric_004: W = |J|² - 2 det(J) (2D non-barrier, untangler).
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric004;

impl TmopQualityMetric for TmopMetric004 {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        ie.get_i1() - 2.0 * ie.get_i2b()
    }

    fn eval_w_matrix_form(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        fnorm2_2x2(jpt) - 2.0 * det_2x2(jpt)
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        // P = dI1 - 2 dI2b
        let di1 = ie.get_di1().clone();
        let di2b = ie.get_di2b().clone();
        let mut result = [0.0; 4];
        for i in 0..4 {
            result[i] = di1[i] - 2.0 * di2b[i];
        }
        *p = from_col_major_2x2(&result);
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_2x2(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_2d(ds));
        ie.assemble_dd_i1(weight, a);
        ie.assemble_dd_i2b(-2.0 * weight, a);
    }

    fn id(&self) -> i32 {
        4
    }
}

/// TMOP_Metric_066: `(1-gamma) mu_4 + gamma mu_55`
/// (MFEM `TMOP_Metric_066 : public TMOP_Combo_QualityMetric`,
/// `TMOP_Metric_066(gamma)` with `AddQualityMetric(sh=004, 1.-gamma)` and
/// `AddQualityMetric(sz=055, gamma)`; mesh-optimizer instantiates it with
/// gamma = 0.5).
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric066 {
    sh: TmopMetric004,
    sz: TmopMetric055,
    gamma: f64,
}

impl TmopMetric066 {
    /// mesh-optimizer / tmop-check-metric default gamma.
    pub fn new(gamma: f64) -> Self {
        Self {
            sh: TmopMetric004,
            sz: TmopMetric055,
            gamma,
        }
    }
}

impl TmopQualityMetric for TmopMetric066 {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        (1.0 - self.gamma) * self.sh.eval_w(jpt) + self.gamma * self.sz.eval_w(jpt)
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        let mut pt = [[0.0_f64; 2]; 2];
        self.sh.eval_p(jpt, &mut pt);
        for r in 0..2 {
            for c in 0..2 {
                p[r][c] = (1.0 - self.gamma) * pt[r][c];
            }
        }
        self.sz.eval_p(jpt, &mut pt);
        for r in 0..2 {
            for c in 0..2 {
                p[r][c] += self.gamma * pt[r][c];
            }
        }
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        self.sh.assemble_h(jpt, ds, weight * (1.0 - self.gamma), a);
        self.sz.assemble_h(jpt, ds, weight * self.gamma, a);
    }

    fn id(&self) -> i32 {
        66
    }
}

/// TMOP_Metric_080: `(1-gamma) mu_2 + gamma mu_77`
/// (MFEM `TMOP_Metric_080(gamma)`; mesh-optimizer / tmop-check-metric use
/// gamma = 0.5).
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric080 {
    sh: TmopMetric002,
    sz: TmopMetric077,
    gamma: f64,
}

impl TmopMetric080 {
    pub fn new(gamma: f64) -> Self {
        Self {
            sh: TmopMetric002,
            sz: TmopMetric077,
            gamma,
        }
    }
}

impl TmopQualityMetric for TmopMetric080 {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        (1.0 - self.gamma) * self.sh.eval_w(jpt) + self.gamma * self.sz.eval_w(jpt)
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        let mut pt = [[0.0_f64; 2]; 2];
        self.sh.eval_p(jpt, &mut pt);
        for r in 0..2 {
            for c in 0..2 {
                p[r][c] = (1.0 - self.gamma) * pt[r][c];
            }
        }
        self.sz.eval_p(jpt, &mut pt);
        for r in 0..2 {
            for c in 0..2 {
                p[r][c] += self.gamma * pt[r][c];
            }
        }
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        self.sh.assemble_h(jpt, ds, weight * (1.0 - self.gamma), a);
        self.sz.assemble_h(jpt, ds, weight * self.gamma, a);
    }

    fn id(&self) -> i32 {
        80
    }
}

/// TMOP_Metric_085: W = |T - T'|², T' = |T| I/√2 (2D barrier Shape+Orientation;
/// polyconvex). MFEM 4.10 implements it via native AD (`mu85_ad`), mirrored
/// through [`crate::tmop::ad`].
#[derive(Debug, Default, Clone, Copy)]
pub struct TmopMetric085;

impl TmopQualityMetric for TmopMetric085 {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        let t = to_col_major_2x2(jpt);
        let w = [0.0; 4];
        mu85_ad(&t, &w)
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        let t = to_col_major_2x2(jpt);
        let w = [0.0; 4];
        let g = ad_grad(&mu85_ad::<Ad1>, &t, Some(&w[..]));
        *p = from_col_major_2x2(&[g[0], g[1], g[2], g[3]]);
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        let t = to_col_major_2x2(jpt);
        let w = [0.0; 4];
        let h = ad_hessian(&mu85_ad::<Ad2>, &t, Some(&w[..]));
        let ndof = ds.len();
        let mut work = vec![0.0; ndof * 2 * ndof * 2];
        default_assemble_h(&h, &flatten_2d(ds), ndof, 2, weight, &mut work);
        for (o, v) in work.iter().enumerate() {
            a[o] += v;
        }
    }

    fn id(&self) -> i32 {
        85
    }
}

/// TMOP_Metric_090: `mu_50 + lambda mu_77` with lambda = 2.5
/// (MFEM `TMOP_Metric_090` — "1 <= lambda <= 4 should produce best asymptotic
/// balance", `AddQualityMetric(sh=050, 1.0)`, `AddQualityMetric(sz=077, 2.5)`).
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric090 {
    sh: TmopMetric050,
    sz: TmopMetric077,
}

impl TmopMetric090 {
    /// MFEM `TMOP_Metric_090::TMOP_Metric_090`: `AddQualityMetric(sz_metric, 2.5)`.
    pub const SZ_WEIGHT: f64 = 2.5;

    pub fn new() -> Self {
        Self {
            sh: TmopMetric050,
            sz: TmopMetric077,
        }
    }
}

impl Default for TmopMetric090 {
    fn default() -> Self {
        Self::new()
    }
}

impl TmopQualityMetric for TmopMetric090 {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        self.sh.eval_w(jpt) + Self::SZ_WEIGHT * self.sz.eval_w(jpt)
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        let mut pt = [[0.0_f64; 2]; 2];
        self.sh.eval_p(jpt, &mut pt);
        for r in 0..2 {
            for c in 0..2 {
                p[r][c] = pt[r][c];
            }
        }
        self.sz.eval_p(jpt, &mut pt);
        for r in 0..2 {
            for c in 0..2 {
                p[r][c] += Self::SZ_WEIGHT * pt[r][c];
            }
        }
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        self.sh.assemble_h(jpt, ds, weight, a);
        self.sz.assemble_h(jpt, ds, weight * Self::SZ_WEIGHT, a);
    }

    fn id(&self) -> i32 {
        90
    }
}

/// TMOP_Metric_098: W = |T - I|² / det(T) (2D barrier Shape+Size).
/// MFEM 4.10 implements it via native AD (`mu98_ad`).
#[derive(Debug, Default, Clone, Copy)]
pub struct TmopMetric098;

impl TmopQualityMetric for TmopMetric098 {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        let t = to_col_major_2x2(jpt);
        let w = [0.0; 4];
        mu98_ad(&t, &w)
    }

    fn eval_w_matrix_form(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        let diff = [
            jpt[0][0] - 1.0,
            jpt[1][0],
            jpt[0][1],
            jpt[1][1] - 1.0,
        ];
        fnorm2_arr(&diff) / det_2x2(jpt)
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        let t = to_col_major_2x2(jpt);
        let w = [0.0; 4];
        let g = ad_grad(&mu98_ad::<Ad1>, &t, Some(&w[..]));
        *p = from_col_major_2x2(&[g[0], g[1], g[2], g[3]]);
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        let t = to_col_major_2x2(jpt);
        let w = [0.0; 4];
        let h = ad_hessian(&mu98_ad::<Ad2>, &t, Some(&w[..]));
        let ndof = ds.len();
        let mut work = vec![0.0; ndof * 2 * ndof * 2];
        default_assemble_h(&h, &flatten_2d(ds), ndof, 2, weight, &mut work);
        for (o, v) in work.iter().enumerate() {
            a[o] += v;
        }
    }

    fn id(&self) -> i32 {
        98
    }
}

/// Frobenius norm squared of a flat column-major 2x2 array.
fn fnorm2_arr(m: &[f64; 4]) -> f64 {
    m[0] * m[0] + m[1] * m[1] + m[2] * m[2] + m[3] * m[3]
}

/// TMOP_Metric_211: W = (det(J) - 1)² - det(J) + sqrt(det(J)² + eps)
/// = (I2b - 1)² - I2b + sqrt(I2b² + eps)  (2D untangling, default eps 1e-4).
///
/// MFEM 4.10 implements only `EvalW` (`EvalP`/`AssembleH` abort with
/// "Metric not implemented yet. Use metric mu_55 instead."); the gradient and
/// Hessian below are the exact 2D analogue of MFEM's fully implemented
/// TMOP_Metric_311 (the 3D version of this untangling metric), assembled from
/// dI2b/ddI2b. The fem-rs FD derivative check (crate::tmop::check) validates
/// them against [`TmopMetric211::eval_w`].
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric211 {
    pub eps: f64,
}

impl Default for TmopMetric211 {
    fn default() -> Self {
        Self { eps: 1e-4 }
    }
}

impl TmopQualityMetric for TmopMetric211 {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        let i2b = ie.get_i2b();
        (i2b - 1.0) * (i2b - 1.0) - i2b + (i2b * i2b + self.eps).sqrt()
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        let i2b = ie.get_i2b();
        // dW/dI2b = 2 (I2b - 1) - 1 + I2b / sqrt(I2b^2 + eps)
        let c = 2.0 * i2b - 3.0 + i2b / (i2b * i2b + self.eps).sqrt();
        let di2b = ie.get_di2b().clone();
        *p = from_col_major_2x2(&scale_array(&di2b, c));
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_2x2(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_2d(ds));
        let i2b = ie.get_i2b();
        let c0 = i2b * i2b + self.eps;
        let c1 = 2.0 + 1.0 / c0.sqrt() - i2b * i2b / c0.powf(1.5);
        let c2 = 2.0 * i2b - 3.0 + i2b / c0.sqrt();
        let di2b = ie.get_di2b().clone();
        ie.assemble_tprod_xx(weight * c1, &di2b, a);
        ie.assemble_dd_i2b(c2 * weight, a);
    }

    fn id(&self) -> i32 {
        211
    }
}

/// TMOP_Metric_252: W = 0.5 (det(J) - 1)² / (det(J) - tau0)
/// (2D shifted-barrier untangling form of metric 56; `tau0` is MFEM's
/// reference-bound `real_t &tau0` — the fixed-value twin of the solver-shared
/// [`crate::tmop::metrics`] wrapper in `fem_assembly::tmop_form`).
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric252 {
    pub tau0: f64,
}

impl TmopQualityMetric for TmopMetric252 {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        let i2b = ie.get_i2b();
        0.5 * (i2b - 1.0) * (i2b - 1.0) / (i2b - self.tau0)
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        let jac = to_col_major_2x2(jpt);
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        let i2b = ie.get_i2b();
        // P = (c - 0.5 c²) dI2b with c = (I2b - 1)/(I2b - tau0).
        let c = (i2b - 1.0) / (i2b - self.tau0);
        let di2b = ie.get_di2b().clone();
        *p = from_col_major_2x2(&scale_array(&di2b, c - 0.5 * c * c));
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        // dP = (1 - c)²/(I2b - tau0) (dI2b x dI2b) + (c - 0.5 c²) ddI2b.
        let jac = to_col_major_2x2(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_2d(ds));
        let i2b = ie.get_i2b();
        let c0 = 1.0 / (i2b - self.tau0);
        let c = c0 * (i2b - 1.0);
        let di2b = ie.get_di2b().clone();
        ie.assemble_tprod_xx(weight * c0 * (1.0 - c) * (1.0 - c), &di2b, a);
        ie.assemble_dd_i2b(weight * (c - 0.5 * c * c), a);
    }

    fn id(&self) -> i32 {
        252
    }
}

/// TMOP_AMetric_049: `(1-gamma) mu_2 + gamma nu_50`
/// (MFEM `TMOP_AMetric_049(gamma) : public TMOP_Combo_QualityMetric`, a
/// barrier Shape+Skew combination; "gamma is recommended to be in (0, 0.9)".
/// mesh-optimizer instantiates it with gamma = 0.9).
#[derive(Debug)]
pub struct TmopAMetric049 {
    sh: TmopMetric002,
    sk: TmopAMetric050,
    gamma: f64,
}

impl TmopAMetric049 {
    pub fn new(gamma: f64) -> Self {
        Self {
            sh: TmopMetric002,
            sk: TmopAMetric050::new(),
            gamma,
        }
    }
}

impl TmopQualityMetric for TmopAMetric049 {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        (1.0 - self.gamma) * self.sh.eval_w(jpt) + self.gamma * self.sk.eval_w(jpt)
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        let mut pt = [[0.0_f64; 2]; 2];
        self.sh.eval_p(jpt, &mut pt);
        for r in 0..2 {
            for c in 0..2 {
                p[r][c] = (1.0 - self.gamma) * pt[r][c];
            }
        }
        self.sk.eval_p(jpt, &mut pt);
        for r in 0..2 {
            for c in 0..2 {
                p[r][c] += self.gamma * pt[r][c];
            }
        }
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        self.sh.assemble_h(jpt, ds, weight * (1.0 - self.gamma), a);
        self.sk.assemble_h(jpt, ds, weight * self.gamma, a);
    }

    fn set_target_jacobian(&self, jtr: &[[f64; 2]; 2]) {
        // TMOP_Combo_QualityMetric::SetTargetJacobian broadcasts to the parts.
        self.sk.set_target_jacobian(jtr);
    }

    fn id(&self) -> i32 {
        49
    }
}

/// TMOP_AMetric_126: `(1-gamma) nu_11 + gamma nu_14`
/// (MFEM `TMOP_AMetric_126(gamma)`, a barrier Shape+Size A-metric combination;
/// tmop-check-metric instantiates it with gamma = 0.9).
#[derive(Debug)]
pub struct TmopAMetric126 {
    sh: TmopAMetric011,
    sz: TmopAMetric014,
    gamma: f64,
}

impl TmopAMetric126 {
    pub fn new(gamma: f64) -> Self {
        Self {
            sh: TmopAMetric011::new(),
            sz: TmopAMetric014::new(),
            gamma,
        }
    }
}

impl TmopQualityMetric for TmopAMetric126 {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        (1.0 - self.gamma) * self.sh.eval_w(jpt) + self.gamma * self.sz.eval_w(jpt)
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        let mut pt = [[0.0_f64; 2]; 2];
        self.sh.eval_p(jpt, &mut pt);
        for r in 0..2 {
            for c in 0..2 {
                p[r][c] = (1.0 - self.gamma) * pt[r][c];
            }
        }
        self.sz.eval_p(jpt, &mut pt);
        for r in 0..2 {
            for c in 0..2 {
                p[r][c] += self.gamma * pt[r][c];
            }
        }
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        self.sh.assemble_h(jpt, ds, weight * (1.0 - self.gamma), a);
        self.sz.assemble_h(jpt, ds, weight * self.gamma, a);
    }

    fn set_target_jacobian(&self, jtr: &[[f64; 2]; 2]) {
        self.sh.set_target_jacobian(jtr);
        self.sz.set_target_jacobian(jtr);
    }

    fn id(&self) -> i32 {
        0
    }
}

// ============================================================================
// AD metric templates (fem/tmop.cpp "Metric definitions")
// ============================================================================
//
// MFEM 4.10 implements metrics 085/098/342 and the A-metrics
// 011/014/036/050/051/107 through native forward AD over the generic scalar
// `type`; the functions below mirror those templates operation-for-operation
// (column-major 4/9-entry matrices, matching `DenseMatrix::GetData()`).

fn fnorm2_ad<S: AdScalar>(u: &[S]) -> S {
    // u[0]*u[0] + u[1]*u[1] + ... (left-to-right, as the C++ templates).
    let mut s = u[0].mul(u[0]);
    for uk in u.iter().skip(1) {
        s = s.add((*uk).mul(*uk));
    }
    s
}

fn det_2d_ad<S: AdScalar>(u: &[S]) -> S {
    u[0].mul(u[3]).sub(u[1].mul(u[2]))
}

fn det_3d_ad<S: AdScalar>(u: &[S]) -> S {
    u[0]
        .mul(u[4].mul(u[8]).sub(u[5].mul(u[7])))
        .sub(u[1].mul(u[3].mul(u[8]).sub(u[5].mul(u[6]))))
        .add(u[2].mul(u[3].mul(u[7]).sub(u[4].mul(u[6]))))
}

/// C++ `mult_2D(u, M, mat)`: mat = u*M (2x2, column-major).
fn mult_2d_ad<S: AdScalar>(u: &[S], m: &[S]) -> [S; 4] {
    [
        u[0].mul(m[0]).add(u[2].mul(m[1])),
        u[1].mul(m[0]).add(u[3].mul(m[1])),
        u[0].mul(m[2]).add(u[2].mul(m[3])),
        u[1].mul(m[2]).add(u[3].mul(m[3])),
    ]
}

/// C++ `mult_aTa_2D(in, outm)`: outm = in^t in (2x2, column-major).
fn mult_aTa_2d_ad<S: AdScalar>(a: &[S]) -> [S; 4] {
    [
        a[0].mul(a[0]),
        a[0].mul(a[2]).add(a[1].mul(a[3])),
        a[0].mul(a[2]).add(a[1].mul(a[3])),
        a[3].mul(a[3]),
    ]
}

/// C++ `adjoint_2D(in, outm)`.
fn adjoint_2d_ad<S: AdScalar>(a: &[S]) -> [S; 4] {
    [a[3], a[1].neg(), a[2].neg(), a[0]]
}

/// C++ `transpose_2D(in, outm)`.
fn transpose_2d_ad<S: AdScalar>(a: &[S]) -> [S; 4] {
    [a[0], a[2], a[1], a[3]]
}

/// C++ `add_2D(scalar, u, M, mat)` with a `real_t` scalar: mat = u + s*M
/// (M plain reals, MFEM's DenseMatrix overload).
fn add_2d_real<S: AdScalar>(s: f64, u: &[S], m: &[f64]) -> [S; 4] {
    [
        u[0].add_real(s * m[0]),
        u[1].add_real(s * m[1]),
        u[2].add_real(s * m[2]),
        u[3].add_real(s * m[3]),
    ]
}

/// C++ `add_2D(scalar, u, M, mat)` with a `type` scalar: mat = u + s*M.
fn add_2d_scalar<S: AdScalar>(s: S, u: &[S], m: &[S]) -> [S; 4] {
    [
        u[0].add(s.mul(m[0])),
        u[1].add(s.mul(m[1])),
        u[2].add(s.mul(m[2])),
        u[3].add(s.mul(m[3])),
    ]
}

/// C++ `add_2D(scalar, u, M, mat)` with a `real_t` scalar over a `type` M:
/// mat = u + s*M (s multiplies as `real_t * type`).
fn add_2d_scale<S: AdScalar>(s: f64, u: &[S], m: &[S]) -> [S; 4] {
    [
        u[0].add(m[0].scale(s)),
        u[1].add(m[1].scale(s)),
        u[2].add(m[2].scale(s)),
        u[3].add(m[3].scale(s)),
    ]
}

/// C++ `add_3D(scalar, u, M, mat)` with a `real_t` scalar (3x3).
fn add_3d_real<S: AdScalar>(s: f64, u: &[S], m: &[f64]) -> [S; 9] {
    let mut mat = [S::default(); 9];
    for k in 0..9 {
        mat[k] = u[k].add_real(s * m[k]);
    }
    mat
}

/// W = |T-T'|^2, T' = |T|*I/sqrt(2)  (mu_85, 2D).
fn mu85_ad<S: AdScalar>(t: &[S], _w: &[S]) -> S {
    let fnorm = fnorm2_ad(t).sqrt();
    let r2 = 2.0f64.sqrt();
    let t0 = t[0].sub(fnorm.div_real(r2));
    let t3 = t[3].sub(fnorm.div_real(r2));
    t[1]
        .mul(t[1])
        .add(t[2].mul(t[2]))
        .add(t0.mul(t0))
        .add(t3.mul(t3))
}

/// W = |T-I|^2 / det(T)  (mu_98, 2D).
fn mu98_ad<S: AdScalar>(t: &[S], _w: &[S]) -> S {
    // add_2D(real_t{-1.0}, T, &Id, Mat) — Mat = T - I.
    let id = [1.0, 0.0, 0.0, 1.0];
    let mat = add_2d_real(-1.0, t, &id);
    fnorm2_ad(&mat).div(det_2d_ad(t))
}

/// W = |T-I|^2 / sqrt(det(T))  (mu_342, 3D).
fn mu342_ad<S: AdScalar>(t: &[S], _w: &[S]) -> S {
    let id = [
        1.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, //
        0.0, 0.0, 1.0,
    ];
    let mat = add_3d_real(-1.0, t, &id);
    fnorm2_ad(&mat).div(det_3d_ad(t).sqrt())
}

/// (1/4α) |A - (adj A)^t W^t W/ω|², A = T·W  (nu_11, 2D).
fn nu11_ad<S: AdScalar>(t: &[S], w: &[S]) -> S {
    let a = mult_2d_ad(t, w); // T*W = A
    let alpha = det_2d_ad(&a);
    let omega = det_2d_ad(w);
    let adj_a = adjoint_2d_ad(&a);
    let adj_at = transpose_2d_ad(&adj_a);
    let wt_w = mult_aTa_2d_ad(w);
    let wrk = mult_2d_ad(&adj_at, &wt_w);
    // add_2D(-1.0/omega, A, WRK, WRK2) — the scalar is `type` here.
    let s = AdScalar::real_div(-1.0, omega);
    let wrk2 = add_2d_scalar(s, &a, &wrk);
    let fnorm = fnorm2_ad(&wrk2);
    AdScalar::real_div(0.25, alpha).mul(fnorm)
}

/// 0.5 ( sqrt(α/ω) - sqrt(ω/α) )², A = T·W  (nu_14, 2D).
fn nu14_ad<S: AdScalar>(t: &[S], w: &[S]) -> S {
    let a = mult_2d_ad(t, w);
    let sqalpha = det_2d_ad(&a).sqrt();
    let sqomega = det_2d_ad(w).sqrt();
    sqalpha
        .div(sqomega)
        .sub(sqomega.div(sqalpha))
        .powf(2.0)
        .scale(0.5)
}

/// (1/α) |A - W|², A = T·W  (nu_36, 2D).
fn nu36_ad<S: AdScalar>(t: &[S], w: &[S]) -> S {
    let a = mult_2d_ad(t, w);
    // add_2D(-1.0, A, W, AminusW) — scalar is `real_t`, so s*M is a scale.
    let a_minus_w = add_2d_scale(-1.0, &a, w);
    let fnorm = fnorm2_ad(&a_minus_w);
    AdScalar::real_div(1.0, det_2d_ad(&a)).mul(fnorm)
}

/// [1 - cos(phi_A - phi_W)] / (sin phi_A * sin phi_W), A = T·W  (nu_50, 2D).
fn nu50_ad<S: AdScalar>(t: &[S], w: &[S]) -> S {
    let a = mult_2d_ad(t, w);
    let (sin_a, cos_a) = sincos_2d_ad(&a);
    let (sin_w, cos_w) = sincos_2d_ad(w);
    let one = S::default();
    one.add_real(1.0)
        .sub(cos_a.mul(cos_w))
        .sub(sin_a.mul(sin_w))
        .div(sin_a.mul(sin_w))
}

/// [0.5 (ups_A/ups_W + ups_W/ups_A) - cos(phi_A - phi_W)] /
/// (sin phi_A * sin phi_W), ups = l1 l2 sin(phi), A = T·W  (nu_51, 2D).
fn nu51_ad<S: AdScalar>(t: &[S], w: &[S]) -> S {
    let a = mult_2d_ad(t, w);
    let (sin_a, cos_a) = sincos_2d_ad(&a);
    let (sin_w, cos_w) = sincos_2d_ad(w);
    // ups = l1 l2 sin(phi) = prod * sin.
    let ups_a = l1l2_2d_ad(&a).mul(sin_a);
    let ups_w = l1l2_2d_ad(w).mul(sin_w);
    let half = S::default();
    half.add_real(0.5)
        .mul(ups_a.div(ups_w).add(ups_w.div(ups_a)))
        .sub(cos_a.mul(cos_w))
        .sub(sin_a.mul(sin_w))
        .div(sin_a.mul(sin_w))
}

/// (1/2α) |A - (|A|/|W|) W|², A = T·W  (nu_107, 2D).
fn nu107_ad<S: AdScalar>(t: &[S], w: &[S]) -> S {
    let a = mult_2d_ad(t, w);
    let alpha = det_2d_ad(&a);
    let aw = fnorm2_ad(&a).sqrt().div(fnorm2_ad(w).sqrt());
    // add_2D(-aw, A, W, Mat) — scalar is `type`.
    let s = aw.neg();
    let mat = add_2d_scalar(s, &a, w);
    AdScalar::real_div(0.5, alpha).mul(fnorm2_ad(&mat))
}

/// Shared sine/cosine of the first-column angle for nu_50/nu_51.
fn sincos_2d_ad<S: AdScalar>(m: &[S]) -> (S, S) {
    let l1 = m[0].mul(m[0]).add(m[1].mul(m[1])).sqrt();
    let l2 = m[2].mul(m[2]).add(m[3].mul(m[3])).sqrt();
    let prod = l1.mul(l2);
    let det = m[0].mul(m[3]).sub(m[1].mul(m[2]));
    let sin = det.div(prod);
    let cos = m[0].mul(m[2]).add(m[1].mul(m[3])).div(prod);
    (sin, cos)
}

/// l1*l2 for nu_51's ups product.
fn l1l2_2d_ad<S: AdScalar>(m: &[S]) -> S {
    let l1 = m[0].mul(m[0]).add(m[1].mul(m[1])).sqrt();
    let l2 = m[2].mul(m[2]).add(m[3].mul(m[3])).sqrt();
    l1.mul(l2)
}

/// Shared plumbing of the MFEM native-AD A-metrics: the target Jacobian `Jtr`
/// (set per quadrature point) plus the plain/gradient/Hessian evaluations of
/// the metric template.
macro_rules! impl_ad_ametric_2d {
    ($ty:ident, $mu:path, $id:expr) => {
        impl Default for $ty {
            fn default() -> Self {
                Self::new()
            }
        }

        impl TmopQualityMetric for $ty {
            fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
                let t = to_col_major_2x2(jpt);
                let w = to_col_major_2x2(&self.jtr.borrow());
                $mu(&t, &w)
            }

            fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
                let t = to_col_major_2x2(jpt);
                let w = to_col_major_2x2(&self.jtr.borrow());
                let mu1 = |x: &[Ad1], y: &[Ad1]| $mu(x, y);
                let g = ad_grad(&mu1, &t, Some(&w[..]));
                *p = from_col_major_2x2(&[g[0], g[1], g[2], g[3]]);
            }

            fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
                let t = to_col_major_2x2(jpt);
                let w = to_col_major_2x2(&self.jtr.borrow());
                let mu2 = |x: &[Ad2], y: &[Ad2]| $mu(x, y);
                let h = ad_hessian(&mu2, &t, Some(&w[..]));
                let ndof = ds.len();
                let mut work = vec![0.0; ndof * 2 * ndof * 2];
                default_assemble_h(&h, &flatten_2d(ds), ndof, 2, weight, &mut work);
                for (o, v) in work.iter().enumerate() {
                    a[o] += v;
                }
            }

            fn set_target_jacobian(&self, jtr: &[[f64; 2]; 2]) {
                *self.jtr.borrow_mut() = *jtr;
            }

            fn id(&self) -> i32 {
                $id
            }
        }
    };
}

// The doc comments live on the structs themselves (macro-expanded impls do not
// carry doc attributes cleanly).

/// TMOP_AMetric_011: W = (1/4α) |A - (adj A)^t W^t W / ω|², A = T·W
/// (2D barrier Shape, polyconvex; target-dependent). MFEM 4.10 implements it
/// with native AD (`nu11_ad`, fem/tmop.cpp), mirrored through
/// [`crate::tmop::ad`] here. Note MFEM does not override `Id()` for the
/// A-metrics, so `id()` reports the base-class 0 (except TMOP_AMetric_049).
#[derive(Debug)]
pub struct TmopAMetric011 {
    /// MFEM `Jtr` — the target Jacobian, set per quadrature point.
    jtr: RefCell<[[f64; 2]; 2]>,
}

impl TmopAMetric011 {
    pub fn new() -> Self {
        Self {
            jtr: RefCell::new([[1.0, 0.0], [0.0, 1.0]]),
        }
    }
}

impl_ad_ametric_2d!(TmopAMetric011, nu11_ad, 0);

/// TMOP_AMetric_014: W = 0.5 ( sqrt(α/ω) - sqrt(ω/α) )², A = T·W
/// (2D barrier Size, polyconvex; target-dependent, AD `nu14_ad`).
#[derive(Debug)]
pub struct TmopAMetric014 {
    jtr: RefCell<[[f64; 2]; 2]>,
}

impl TmopAMetric014 {
    pub fn new() -> Self {
        Self {
            jtr: RefCell::new([[1.0, 0.0], [0.0, 1.0]]),
        }
    }
}

impl_ad_ametric_2d!(TmopAMetric014, nu14_ad, 0);

/// TMOP_AMetric_036: W = (1/α) |A - W|², A = T·W
/// (2D barrier Shape+Size+Orientation, polyconvex; AD `nu36_ad`).
#[derive(Debug)]
pub struct TmopAMetric036 {
    jtr: RefCell<[[f64; 2]; 2]>,
}

impl TmopAMetric036 {
    pub fn new() -> Self {
        Self {
            jtr: RefCell::new([[1.0, 0.0], [0.0, 1.0]]),
        }
    }
}

impl_ad_ametric_2d!(TmopAMetric036, nu36_ad, 0);

/// TMOP_AMetric_050: W = [1 - cos(φ_A - φ_W)] / (sin φ_A sin φ_W), A = T·W
/// (2D barrier Skew; AD `nu50_ad`).
#[derive(Debug)]
pub struct TmopAMetric050 {
    jtr: RefCell<[[f64; 2]; 2]>,
}

impl TmopAMetric050 {
    pub fn new() -> Self {
        Self {
            jtr: RefCell::new([[1.0, 0.0], [0.0, 1.0]]),
        }
    }
}

impl_ad_ametric_2d!(TmopAMetric050, nu50_ad, 0);

/// TMOP_AMetric_051: W = [0.5 (ups_A/ups_W + ups_W/ups_A) - cos(φ_A - φ_W)] /
/// (sin φ_A sin φ_W), A = T·W (2D barrier Size+Skew; AD `nu51_ad`).
#[derive(Debug)]
pub struct TmopAMetric051 {
    jtr: RefCell<[[f64; 2]; 2]>,
}

impl TmopAMetric051 {
    pub fn new() -> Self {
        Self {
            jtr: RefCell::new([[1.0, 0.0], [0.0, 1.0]]),
        }
    }
}

impl_ad_ametric_2d!(TmopAMetric051, nu51_ad, 0);

/// TMOP_AMetric_107: W = (1/2α) |A - (|A|/|W|) W|², A = T·W
/// (2D barrier Shape+Orientation, polyconvex; AD `nu107_ad`).
#[derive(Debug)]
pub struct TmopAMetric107 {
    jtr: RefCell<[[f64; 2]; 2]>,
}

impl TmopAMetric107 {
    pub fn new() -> Self {
        Self {
            jtr: RefCell::new([[1.0, 0.0], [0.0, 1.0]]),
        }
    }
}

impl_ad_ametric_2d!(TmopAMetric107, nu107_ad, 0);

// ============================================================================
// 3D Metrics
// ============================================================================

/// TMOP_Metric_301: W = 1/3 sqrt(I1b * I2b) - 1 (3D barrier shape, polyconvex & invex)
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric301;

impl TmopQualityMetric3D for TmopMetric301 {
    fn eval_w(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        (ie.get_i1b() * ie.get_i2b()).sqrt() / 3.0 - 1.0
    }

    fn eval_w_matrix_form(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        // W = 1/3 |J| |J^-1| - 1
        let inv = calc_inverse_transpose_3x3(jpt);
        let fnorm_j = fnorm2_3x3(jpt).sqrt();
        let fnorm_inv = fnorm2_3x3(&inv).sqrt();
        fnorm_j * fnorm_inv / 3.0 - 1.0
    }

    fn eval_p(&self, jpt: &[[f64; 3]; 3], p: &mut [[f64; 3]; 3]) {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        let i1b = ie.get_i1b();
        let i2b = ie.get_i2b();
        let a = 1.0 / (6.0 * (i1b * i2b).sqrt());
        let di1b = ie.get_di1b().clone();
        let di2b = ie.get_di2b().clone();
        // P = a*I2b dI1b + a*I1b dI2b
        let mut result = [0.0; 9];
        for i in 0..9 {
            result[i] = a * i2b * di1b[i] + a * i1b * di2b[i];
        }
        *p = from_col_major_3x3(&result);
    }

    fn assemble_h(&self, jpt: &[[f64; 3]; 3], ds: &[[f64; 3]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_3x3(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_3d(ds));
        let i1b = ie.get_i1b();
        let i2b = ie.get_i2b();
        let di1b = ie.get_di1b().clone();
        let di2b = ie.get_di2b().clone();
        let mut x_data = [0.0; 9];
        for i in 0..9 {
            x_data[i] = -i2b * di1b[i] + i1b * di2b[i];
        }
        let i1b_i2b = i1b * i2b;
        let coeff = weight / (6.0 * i1b_i2b.sqrt());
        ie.assemble_dd_i1b(coeff * i2b, a);
        ie.assemble_dd_i2b(coeff * i1b, a);
        ie.assemble_tprod_xx(-coeff / (2.0 * i1b_i2b), &x_data, a);
    }

    fn id(&self) -> i32 {
        301
    }
}

/// TMOP_Metric_302: W = |J|² |J^{-1}|² / 9 - 1 (3D barrier shape)
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric302;

impl TmopQualityMetric3D for TmopMetric302 {
    fn eval_w(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        ie.get_i1b() * ie.get_i2b() / 9.0 - 1.0
    }

    fn eval_w_matrix_form(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        let inv = calc_inverse_transpose_3x3(jpt);
        fnorm2_3x3(jpt) * fnorm2_3x3(&inv) / 9.0 - 1.0
    }

    fn eval_p(&self, jpt: &[[f64; 3]; 3], p: &mut [[f64; 3]; 3]) {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        let i1b = ie.get_i1b();
        let i2b = ie.get_i2b();
        let di1b = ie.get_di1b().clone();
        let di2b = ie.get_di2b().clone();
        // P = (I1b/9) dI2b + (I2b/9) dI1b
        let mut result = [0.0; 9];
        for i in 0..9 {
            result[i] = i1b / 9.0 * di2b[i] + i2b / 9.0 * di1b[i];
        }
        *p = from_col_major_3x3(&result);
    }

    fn assemble_h(&self, jpt: &[[f64; 3]; 3], ds: &[[f64; 3]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_3x3(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_3d(ds));
        let i1b = ie.get_i1b();
        let i2b = ie.get_i2b();
        let di1b = ie.get_di1b().clone();
        let di2b = ie.get_di2b().clone();
        let c1 = weight / 9.0;
        ie.assemble_tprod_xy(c1, &di1b, &di2b, a);
        ie.assemble_dd_i2b(c1 * i1b, a);
        ie.assemble_dd_i1b(c1 * i2b, a);
    }

    fn id(&self) -> i32 {
        302
    }
}

/// TMOP_Metric_303: W = |J|² / (3 det(J)^{2/3}) - 1 (3D barrier shape)
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric303;

impl TmopQualityMetric3D for TmopMetric303 {
    fn eval_w(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        ie.get_i1b() / 3.0 - 1.0
    }

    fn eval_w_matrix_form(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        fnorm2_3x3(jpt) / 3.0 / det_3x3(jpt).powf(2.0 / 3.0) - 1.0
    }

    fn eval_p(&self, jpt: &[[f64; 3]; 3], p: &mut [[f64; 3]; 3]) {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        let di1b = ie.get_di1b().clone();
        *p = from_col_major_3x3(&scale_array(&di1b, 1.0 / 3.0));
    }

    fn assemble_h(&self, jpt: &[[f64; 3]; 3], ds: &[[f64; 3]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_3x3(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_3d(ds));
        ie.assemble_dd_i1b(weight / 3.0, a);
    }

    fn id(&self) -> i32 {
        303
    }
}

/// TMOP_Metric_304: W = |J|³ / (3^{3/2} det(J)) - 1 (3D barrier shape)
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric304;

impl TmopQualityMetric3D for TmopMetric304 {
    fn eval_w(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        (ie.get_i1b() / 3.0).powf(1.5) - 1.0
    }

    fn eval_w_matrix_form(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        let fnorm = fnorm2_3x3(jpt).sqrt();
        fnorm.powi(3) / 3.0_f64.powf(1.5) / det_3x3(jpt) - 1.0
    }

    fn eval_p(&self, jpt: &[[f64; 3]; 3], p: &mut [[f64; 3]; 3]) {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        let i1b = ie.get_i1b();
        let di1b = ie.get_di1b().clone();
        // P = 0.5 * (I1b/3)^{1/2} dI1b
        *p = from_col_major_3x3(&scale_array(&di1b, 0.5 * (i1b / 3.0).sqrt()));
    }

    fn assemble_h(&self, jpt: &[[f64; 3]; 3], ds: &[[f64; 3]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_3x3(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_3d(ds));
        let i1b = ie.get_i1b();
        let di1b = ie.get_di1b().clone();
        ie.assemble_tprod_xx(weight / (12.0 * (i1b / 3.0).sqrt()), &di1b, a);
        ie.assemble_dd_i1b(weight * 0.5 * (i1b / 3.0).sqrt(), a);
    }

    fn id(&self) -> i32 {
        304
    }
}

/// TMOP_Metric_315: W = (det(J) - 1)² (3D size)
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric315;

impl TmopQualityMetric3D for TmopMetric315 {
    fn eval_w(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        let c1 = ie.get_i3b() - 1.0;
        c1 * c1
    }

    fn eval_p(&self, jpt: &[[f64; 3]; 3], p: &mut [[f64; 3]; 3]) {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        let i3b = ie.get_i3b();
        let di3b = ie.get_di3b().clone();
        *p = from_col_major_3x3(&scale_array(&di3b, 2.0 * (i3b - 1.0)));
    }

    fn assemble_h(&self, jpt: &[[f64; 3]; 3], ds: &[[f64; 3]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_3x3(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_3d(ds));
        let i3b = ie.get_i3b();
        let di3b = ie.get_di3b().clone();
        ie.assemble_tprod_xx(2.0 * weight, &di3b, a);
        ie.assemble_dd_i3b(2.0 * weight * (i3b - 1.0), a);
    }

    fn id(&self) -> i32 {
        315
    }
}

/// TMOP_Metric_316: W = 0.5 (det(J) + 1/det(J)) - 1 (3D size)
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric316;

impl TmopQualityMetric3D for TmopMetric316 {
    fn eval_w(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        let i3b = ie.get_i3b();
        0.5 * (i3b + 1.0 / i3b) - 1.0
    }

    fn eval_w_matrix_form(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        let d = det_3x3(jpt);
        0.5 * (d + 1.0 / d) - 1.0
    }

    fn eval_p(&self, jpt: &[[f64; 3]; 3], p: &mut [[f64; 3]; 3]) {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        let i3 = ie.get_i3();
        let di3b = ie.get_di3b().clone();
        // P = (0.5 - 0.5/I3) dI3b
        *p = from_col_major_3x3(&scale_array(&di3b, 0.5 - 0.5 / i3));
    }

    fn assemble_h(&self, jpt: &[[f64; 3]; 3], ds: &[[f64; 3]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_3x3(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_3d(ds));
        let i3 = ie.get_i3();
        let i3b = ie.get_i3b();
        let di3b = ie.get_di3b().clone();
        ie.assemble_tprod_xx(weight / (i3 * i3b), &di3b, a);
        ie.assemble_dd_i3b(weight * (0.5 - 0.5 / i3), a);
    }

    fn id(&self) -> i32 {
        316
    }
}

/// TMOP_Metric_318: W = 0.5 (det(J)² + 1/det(J)²) - 1 (3D size)
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric318;

impl TmopQualityMetric3D for TmopMetric318 {
    fn eval_w(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        let i3 = ie.get_i3();
        0.5 * (i3 + 1.0 / i3) - 1.0
    }

    fn eval_w_matrix_form(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        let d = det_3x3(jpt);
        0.5 * (d * d + 1.0 / (d * d)) - 1.0
    }

    fn eval_p(&self, jpt: &[[f64; 3]; 3], p: &mut [[f64; 3]; 3]) {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        let i3 = ie.get_i3();
        let di3 = ie.get_di3().clone();
        // P = (0.5 - 0.5/I3²) dI3
        *p = from_col_major_3x3(&scale_array(&di3, 0.5 - 0.5 / (i3 * i3)));
    }

    fn assemble_h(&self, jpt: &[[f64; 3]; 3], ds: &[[f64; 3]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_3x3(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_3d(ds));
        let i3 = ie.get_i3();
        let di3 = ie.get_di3().clone();
        ie.assemble_tprod_xx(weight / (i3 * i3 * i3), &di3, a);
        ie.assemble_dd_i3(weight * (0.5 - 0.5 / (i3 * i3)), a);
    }

    fn id(&self) -> i32 {
        318
    }
}

/// TMOP_Metric_321: W = |J - J^{-t}|² (3D barrier shape+size)
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric321;

impl TmopQualityMetric3D for TmopMetric321 {
    fn eval_w(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        let i1 = ie.get_i1();
        let i2 = ie.get_i2();
        let i3 = ie.get_i3();
        i1 + i2 / i3 - 6.0
    }

    fn eval_w_matrix_form(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        let inv_t = calc_inverse_transpose_3x3(jpt);
        let mut diff = [[0.0; 3]; 3];
        for j in 0..3 {
            for i in 0..3 {
                diff[i][j] = jpt[i][j] - inv_t[i][j];
            }
        }
        fnorm2_3x3(&diff)
    }

    fn eval_p(&self, jpt: &[[f64; 3]; 3], p: &mut [[f64; 3]; 3]) {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        let i2 = ie.get_i2();
        let i3 = ie.get_i3();
        let i3b = ie.get_i3b();
        let di2 = ie.get_di2().clone();
        let di3b = ie.get_di3b().clone();
        // P = dI1 + (1/I3) dI2 - (2*I2/I3b³) dI3b
        let mut result = ie.get_di1().clone();
        for i in 0..9 {
            result[i] += (1.0 / i3) * di2[i] - (2.0 * i2 / (i3 * i3b)) * di3b[i];
        }
        *p = from_col_major_3x3(&result);
    }

    fn assemble_h(&self, jpt: &[[f64; 3]; 3], ds: &[[f64; 3]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_3x3(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_3d(ds));
        let i2 = ie.get_i2();
        let i3b = ie.get_i3b();
        let di2 = ie.get_di2().clone();
        let di3b = ie.get_di3b().clone();
        // MFEM TMOP_Metric_321::AssembleH:
        //   c0 = 1/I3b, c1 = w/I3b^2, c2 = -2w/I3b^3, c3 = -2w I2/I3b^3;
        //   ddI1(w) + ddI2(c1) + ddI3b(c3) + TProd(c2, dI2, dI3b).
        let c0 = 1.0 / i3b;
        let c1 = weight * c0 * c0;
        let c2 = -2.0 * c0 * c1;
        let c3 = c2 * i2;
        ie.assemble_dd_i1(weight, a);
        ie.assemble_dd_i2(c1, a);
        ie.assemble_dd_i3b(c3, a);
        ie.assemble_tprod_xy(c2, &di2, &di3b, a);
        // MFEM fifth term: TProd(-3*c0*c3, dI3b) — the (6 I2/I3b^4)
        // (dI3b x dI3b) contribution the original port dropped.
        ie.assemble_tprod_xx(-3.0 * c0 * c3, &di3b, a);
    }

    fn id(&self) -> i32 {
        321
    }
}

/// TMOP_Metric_323: W = |J|³ - 3 sqrt(3) ln(det(J)) - 3 sqrt(3) (3D shape+size)
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric323;

impl TmopQualityMetric3D for TmopMetric323 {
    fn eval_w(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        ie.get_i1().powf(1.5) - 3.0 * 3.0_f64.sqrt() * (ie.get_i3b().ln() + 1.0)
    }

    fn eval_w_matrix_form(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        let fnorm = fnorm2_3x3(jpt).sqrt();
        fnorm.powi(3) - 3.0 * 3.0_f64.sqrt() * (det_3x3(jpt).ln() + 1.0)
    }

    fn eval_p(&self, jpt: &[[f64; 3]; 3], p: &mut [[f64; 3]; 3]) {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        let i1 = ie.get_i1();
        let i3b = ie.get_i3b();
        let di1 = ie.get_di1().clone();
        let di3b = ie.get_di3b().clone();
        // P = 1.5 * sqrt(I1) dI1 - 3*sqrt(3)/I3b dI3b
        let mut result = scale_array(&di1, 1.5 * i1.sqrt());
        for i in 0..9 {
            result[i] += -3.0 * 3.0_f64.sqrt() / i3b * di3b[i];
        }
        *p = from_col_major_3x3(&result);
    }

    fn assemble_h(&self, jpt: &[[f64; 3]; 3], ds: &[[f64; 3]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_3x3(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_3d(ds));
        let i1 = ie.get_i1();
        let i3b = ie.get_i3b();
        let di1 = ie.get_di1().clone();
        let di3b = ie.get_di3b().clone();
        ie.assemble_dd_i1(weight * 1.5 * i1.sqrt(), a);
        ie.assemble_tprod_xx(weight * 0.75 / i1.sqrt(), &di1, a);
        ie.assemble_dd_i3b(-weight * 3.0 * 3.0_f64.sqrt() / i3b, a);
        ie.assemble_tprod_xx(weight * 3.0 * 3.0_f64.sqrt() / (i3b * i3b), &di3b, a);
    }

    fn id(&self) -> i32 {
        323
    }
}

/// TMOP_Metric_360: W = |J|³ / 3^{3/2} - det(J) (3D shape)
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric360;

impl TmopQualityMetric3D for TmopMetric360 {
    fn eval_w(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        (ie.get_i1() / 3.0).powf(1.5) - ie.get_i3b()
    }

    fn eval_w_matrix_form(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        let fnorm = fnorm2_3x3(jpt).sqrt();
        fnorm.powi(3) / 3.0_f64.powf(1.5) - det_3x3(jpt)
    }

    fn eval_p(&self, jpt: &[[f64; 3]; 3], p: &mut [[f64; 3]; 3]) {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        let i1 = ie.get_i1();
        let di1 = ie.get_di1().clone();
        let di3b = ie.get_di3b().clone();
        // P = 0.5 * (I1/3)^{1/2} dI1 - dI3b
        let mut result = scale_array(&di1, 0.5 * (i1 / 3.0).sqrt());
        for i in 0..9 {
            result[i] -= di3b[i];
        }
        *p = from_col_major_3x3(&result);
    }

    fn assemble_h(&self, jpt: &[[f64; 3]; 3], ds: &[[f64; 3]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_3x3(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_3d(ds));
        let i1 = ie.get_i1();
        let di1 = ie.get_di1().clone();
        ie.assemble_tprod_xx(weight / (12.0 * (i1 / 3.0).sqrt()), &di1, a);
        ie.assemble_dd_i1(weight * 0.5 * (i1 / 3.0).sqrt(), a);
        ie.assemble_dd_i3b(-weight, a);
    }

    fn id(&self) -> i32 {
        360
    }
}

// ============================================================================
// 3D metrics (round-102 additions)
// ============================================================================

/// TMOP_Metric_311: W = (det(J) - 1)² - det(J) + sqrt(det(J)² + eps)
/// = (I3b - 1)² - I3b + sqrt(I3b² + eps)  (3D untangling, default eps 1e-4).
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric311 {
    pub eps: f64,
}

impl Default for TmopMetric311 {
    fn default() -> Self {
        Self { eps: 1e-4 }
    }
}

impl TmopQualityMetric3D for TmopMetric311 {
    fn eval_w(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        let i3b = ie.get_i3b();
        (i3b - 1.0) * (i3b - 1.0) - i3b + (i3b * i3b + self.eps).sqrt()
    }

    fn eval_p(&self, jpt: &[[f64; 3]; 3], p: &mut [[f64; 3]; 3]) {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        let i3b = ie.get_i3b();
        // P = c dI3b, c = 2 I3b - 3 + I3b / sqrt(I3b^2 + eps)
        let c = 2.0 * i3b - 3.0 + i3b / (i3b * i3b + self.eps).sqrt();
        let di3b = ie.get_di3b().clone();
        *p = from_col_major_3x3(&scale_array(&di3b, c));
    }

    fn assemble_h(&self, jpt: &[[f64; 3]; 3], ds: &[[f64; 3]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_3x3(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_3d(ds));
        let i3b = ie.get_i3b();
        let c0 = i3b * i3b + self.eps;
        let c1 = 2.0 + 1.0 / c0.sqrt() - i3b * i3b / c0.powf(1.5);
        let c2 = 2.0 * i3b - 3.0 + i3b / c0.sqrt();
        let di3b = ie.get_di3b().clone();
        ie.assemble_tprod_xx(weight * c1, &di3b, a);
        ie.assemble_dd_i3b(c2 * weight, a);
    }

    fn id(&self) -> i32 {
        311
    }
}

/// TMOP_Metric_313: W = 1/3 |J|² / [det(J) - tau0]^(-2/3)
/// = I1 (I3b - tau0)^(-2/3) / 3  (3D untangling version of 303; `min_detT` is
/// MFEM's reference-bound `real_t &min_detT`, fed from the shared min-det cell
/// exactly like TMOP_Metric_022).
///
/// MFEM 4.10 implements only `EvalW` (`EvalP`/`AssembleH` abort with
/// "Metric not implemented yet."); the derivatives below are derived exactly
/// from the invariant form
/// `W = I1/3 · d^(-2/3)`, `d = I3b - tau0`:
/// `P = (1/3) d^(-2/3) dI1 - (2/9) I1 d^(-5/3) dI3b`,
/// `dP = (1/3) d^(-2/3) ddI1 - (2/9) d^(-5/3) [dI1 x dI3b + dI3b x dI1]
///     + (10/27) I1 d^(-8/3) (dI3b x dI3b) - (2/9) I1 d^(-5/3) ddI3b`.
/// The fem-rs FD derivative check validates them against
/// [`TmopMetric313::eval_w`].
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric313 {
    pub min_det_t: f64,
}

impl TmopQualityMetric3D for TmopMetric313 {
    fn eval_w(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        let i3b = ie.get_i3b();
        let mut d = i3b - self.min_det_t;
        if d < 0.0 && self.min_det_t == 0.0 {
            // MFEM: untangled-mesh FD guard (comment in TMOP_Metric_313::EvalW).
            d = -i3b * 0.1;
        }
        ie.get_i1() * d.powf(-2.0 / 3.0) / 3.0
    }

    fn eval_p(&self, jpt: &[[f64; 3]; 3], p: &mut [[f64; 3]; 3]) {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        let i3b = ie.get_i3b();
        let mut d = i3b - self.min_det_t;
        if d < 0.0 && self.min_det_t == 0.0 {
            d = -i3b * 0.1;
        }
        let i1 = ie.get_i1();
        let di1 = ie.get_di1().clone();
        let di3b = ie.get_di3b().clone();
        // P = (1/3) d^(-2/3) dI1 - (2/9) I1 d^(-5/3) dI3b
        let c1 = (1.0 / 3.0) * d.powf(-2.0 / 3.0);
        let c2 = -(2.0 / 9.0) * i1 * d.powf(-5.0 / 3.0);
        let mut result = [0.0; 9];
        for i in 0..9 {
            result[i] = c1 * di1[i] + c2 * di3b[i];
        }
        *p = from_col_major_3x3(&result);
    }

    fn assemble_h(&self, jpt: &[[f64; 3]; 3], ds: &[[f64; 3]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_3x3(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_3d(ds));
        let i3b = ie.get_i3b();
        let mut d = i3b - self.min_det_t;
        if d < 0.0 && self.min_det_t == 0.0 {
            d = -i3b * 0.1;
        }
        let i1 = ie.get_i1();
        let di1 = ie.get_di1().clone();
        let di3b = ie.get_di3b().clone();
        ie.assemble_dd_i1(weight / 3.0 * d.powf(-2.0 / 3.0), a);
        ie.assemble_tprod_xy(-weight * (2.0 / 9.0) * d.powf(-5.0 / 3.0), &di1, &di3b, a);
        ie.assemble_tprod_xx(
            weight * (10.0 / 27.0) * i1 * d.powf(-8.0 / 3.0),
            &di3b,
            a,
        );
        ie.assemble_dd_i3b(-weight * (2.0 / 9.0) * i1 * d.powf(-5.0 / 3.0), a);
    }

    fn id(&self) -> i32 {
        313
    }
}

/// TMOP_Metric_322: W = |J - adj(J)^t|² / (6 det(J))
/// = I1b (I3b^(-1/3)) / 6 + I2b (I3b^(1/3)) / 6 - 1  (3D barrier shape).
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric322;

impl TmopQualityMetric3D for TmopMetric322 {
    fn eval_w(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        ie.get_i1b() / ie.get_i3b().powf(1.0 / 3.0) / 6.0
            + ie.get_i2b() * ie.get_i3b().powf(1.0 / 3.0) / 6.0
            - 1.0
    }

    fn eval_w_matrix_form(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        // mu_322 = 1/(6 det) |J - adj(J)^t|^2 (MFEM EvalWMatrixForm:
        // CalcAdjugateTranspose(Jpt, adj_J_t); adj_J_t *= -1; += Jpt).
        let adj_t = calc_adjugate_transpose_3x3(jpt);
        let mut diff = [[0.0f64; 3]; 3];
        for j in 0..3 {
            for i in 0..3 {
                diff[i][j] = jpt[i][j] - adj_t[i][j];
            }
        }
        fnorm2_3x3(&diff) / 6.0 / det_3x3(jpt)
    }

    fn eval_p(&self, jpt: &[[f64; 3]; 3], p: &mut [[f64; 3]; 3]) {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        let i1b = ie.get_i1b();
        let i2b = ie.get_i2b();
        let i3b = ie.get_i3b();
        let di1b = ie.get_di1b().clone();
        let di2b = ie.get_di2b().clone();
        let di3b = ie.get_di3b().clone();
        // P =   1/6 (I3b^-1/3) dI1b - 1/18 I1b (I3b^-4/3) dI3b
        //     + 1/6 (I3b^1/3) dI2b  + 1/18 I2b (I3b^-2/3) dI3b
        let mut result = scale_array(&di1b, 1.0 / 6.0 * i3b.powf(-1.0 / 3.0));
        for (i, r) in result.iter_mut().enumerate() {
            *r += -1.0 / 18.0 * i1b * i3b.powf(-4.0 / 3.0) * di3b[i];
        }
        for (i, r) in result.iter_mut().enumerate() {
            *r += 1.0 / 6.0 * i3b.powf(1.0 / 3.0) * di2b[i];
        }
        for (i, r) in result.iter_mut().enumerate() {
            *r += 1.0 / 18.0 * i2b * i3b.powf(-2.0 / 3.0) * di3b[i];
        }
        *p = from_col_major_3x3(&result);
    }

    fn assemble_h(&self, jpt: &[[f64; 3]; 3], ds: &[[f64; 3]], weight: f64, a: &mut [f64]) {
        let jac = to_col_major_3x3(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_3d(ds));
        let i1b = ie.get_i1b();
        let i2b = ie.get_i2b();
        let i3b = ie.get_i3b();
        let p13 = weight * i3b.powf(1.0 / 3.0);
        let m13 = weight * i3b.powf(-1.0 / 3.0);
        let m23 = weight * i3b.powf(-2.0 / 3.0);
        let m43 = weight * i3b.powf(-4.0 / 3.0);
        let m53 = weight * i3b.powf(-5.0 / 3.0);
        let m73 = weight * i3b.powf(-7.0 / 3.0);
        let di1b = ie.get_di1b().clone();
        let di2b = ie.get_di2b().clone();
        let di3b = ie.get_di3b().clone();
        ie.assemble_dd_i1b(1.0 / 6.0 * m13, a);
        // Combines - 1/18 (I3b^-4/3) (dI1b x dI3b) - 1/18 (I3b^-4/3) (dI3b x dI1b).
        ie.assemble_tprod_xy(-1.0 / 18.0 * m43, &di1b, &di3b, a);
        ie.assemble_dd_i3b(-1.0 / 18.0 * i1b * m43, a);
        ie.assemble_tprod_xx(2.0 / 27.0 * i1b * m73, &di3b, a);
        ie.assemble_dd_i2b(1.0 / 6.0 * p13, a);
        // Combines + 1/18 (I3b^-2/3) (dI2b x dI3b) + 1/18 (I3b^-2/3) (dI3b x dI2b).
        ie.assemble_tprod_xy(1.0 / 18.0 * m23, &di2b, &di3b, a);
        ie.assemble_dd_i3b(1.0 / 18.0 * i2b * m23, a);
        ie.assemble_tprod_xx(-1.0 / 27.0 * i2b * m53, &di3b, a);
    }

    fn id(&self) -> i32 {
        322
    }
}

/// MFEM `CalcAdjugateTranspose` (linalg/densemat.cpp, 3x3 branch): the
/// cofactor matrix, `adjt = det(A) A^{-t}`.
fn calc_adjugate_transpose_3x3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [
            m[1][1] * m[2][2] - m[2][1] * m[1][2],
            m[2][0] * m[1][2] - m[0][2] * m[2][0],
            m[0][1] * m[2][0] - m[1][0] * m[0][1],
        ],
        [
            m[2][1] * m[0][2] - m[0][1] * m[2][2],
            m[0][0] * m[2][2] - m[2][0] * m[0][2],
            m[1][0] * m[0][1] - m[0][0] * m[1][0],
        ],
        [
            m[0][1] * m[1][2] - m[1][1] * m[0][2],
            m[1][0] * m[0][2] - m[0][0] * m[1][2],
            m[0][0] * m[1][1] - m[1][0] * m[0][1],
        ],
    ]
}

/// TMOP_Metric_328: `lambda mu_301 + mu_316` with lambda = 0.75
/// (MFEM `TMOP_Metric_328` — "3/8 <= lambda <= 9/8 should produce best
/// asymptotic balance").
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric328 {
    sh: TmopMetric301,
    sz: TmopMetric316,
}

impl TmopMetric328 {
    /// MFEM `AddQualityMetric(sh_metric, 0.75)`.
    pub const SH_WEIGHT: f64 = 0.75;

    pub fn new() -> Self {
        Self {
            sh: TmopMetric301,
            sz: TmopMetric316,
        }
    }
}

impl Default for TmopMetric328 {
    fn default() -> Self {
        Self::new()
    }
}

impl TmopQualityMetric3D for TmopMetric328 {
    fn eval_w(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        Self::SH_WEIGHT * self.sh.eval_w(jpt) + self.sz.eval_w(jpt)
    }

    fn eval_p(&self, jpt: &[[f64; 3]; 3], p: &mut [[f64; 3]; 3]) {
        let mut pt = [[0.0_f64; 3]; 3];
        self.sh.eval_p(jpt, &mut pt);
        for r in 0..3 {
            for c in 0..3 {
                p[r][c] = Self::SH_WEIGHT * pt[r][c];
            }
        }
        self.sz.eval_p(jpt, &mut pt);
        for r in 0..3 {
            for c in 0..3 {
                p[r][c] += pt[r][c];
            }
        }
    }

    fn assemble_h(&self, jpt: &[[f64; 3]; 3], ds: &[[f64; 3]], weight: f64, a: &mut [f64]) {
        self.sh.assemble_h(jpt, ds, weight * Self::SH_WEIGHT, a);
        self.sz.assemble_h(jpt, ds, weight, a);
    }

    fn id(&self) -> i32 {
        328
    }
}

/// TMOP_Metric_332: `(1-gamma) mu_302 + gamma mu_315`
/// (MFEM `TMOP_Metric_332(gamma)`; mesh-optimizer / tmop-check-metric use
/// gamma = 0.5).
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric332 {
    sh: TmopMetric302,
    sz: TmopMetric315,
    gamma: f64,
}

impl TmopMetric332 {
    pub fn new(gamma: f64) -> Self {
        Self {
            sh: TmopMetric302,
            sz: TmopMetric315,
            gamma,
        }
    }
}

impl TmopQualityMetric3D for TmopMetric332 {
    fn eval_w(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        (1.0 - self.gamma) * self.sh.eval_w(jpt) + self.gamma * self.sz.eval_w(jpt)
    }

    fn eval_p(&self, jpt: &[[f64; 3]; 3], p: &mut [[f64; 3]; 3]) {
        let mut pt = [[0.0_f64; 3]; 3];
        self.sh.eval_p(jpt, &mut pt);
        for r in 0..3 {
            for c in 0..3 {
                p[r][c] = (1.0 - self.gamma) * pt[r][c];
            }
        }
        self.sz.eval_p(jpt, &mut pt);
        for r in 0..3 {
            for c in 0..3 {
                p[r][c] += self.gamma * pt[r][c];
            }
        }
    }

    fn assemble_h(&self, jpt: &[[f64; 3]; 3], ds: &[[f64; 3]], weight: f64, a: &mut [f64]) {
        self.sh.assemble_h(jpt, ds, weight * (1.0 - self.gamma), a);
        self.sz.assemble_h(jpt, ds, weight * self.gamma, a);
    }

    fn id(&self) -> i32 {
        332
    }
}

/// TMOP_Metric_333: `(1-gamma) mu_302 + gamma mu_316`
/// (MFEM `TMOP_Metric_333(gamma)`; gamma = 0.5 in the drivers).
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric333 {
    sh: TmopMetric302,
    sz: TmopMetric316,
    gamma: f64,
}

impl TmopMetric333 {
    pub fn new(gamma: f64) -> Self {
        Self {
            sh: TmopMetric302,
            sz: TmopMetric316,
            gamma,
        }
    }
}

impl TmopQualityMetric3D for TmopMetric333 {
    fn eval_w(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        (1.0 - self.gamma) * self.sh.eval_w(jpt) + self.gamma * self.sz.eval_w(jpt)
    }

    fn eval_p(&self, jpt: &[[f64; 3]; 3], p: &mut [[f64; 3]; 3]) {
        let mut pt = [[0.0_f64; 3]; 3];
        self.sh.eval_p(jpt, &mut pt);
        for r in 0..3 {
            for c in 0..3 {
                p[r][c] = (1.0 - self.gamma) * pt[r][c];
            }
        }
        self.sz.eval_p(jpt, &mut pt);
        for r in 0..3 {
            for c in 0..3 {
                p[r][c] += self.gamma * pt[r][c];
            }
        }
    }

    fn assemble_h(&self, jpt: &[[f64; 3]; 3], ds: &[[f64; 3]], weight: f64, a: &mut [f64]) {
        self.sh.assemble_h(jpt, ds, weight * (1.0 - self.gamma), a);
        self.sz.assemble_h(jpt, ds, weight * self.gamma, a);
    }

    fn id(&self) -> i32 {
        333
    }
}

/// TMOP_Metric_334: `(1-gamma) mu_303 + gamma mu_316`
/// (MFEM `TMOP_Metric_334(gamma)`; gamma = 0.5 in the drivers).
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric334 {
    sh: TmopMetric303,
    sz: TmopMetric316,
    gamma: f64,
}

impl TmopMetric334 {
    pub fn new(gamma: f64) -> Self {
        Self {
            sh: TmopMetric303,
            sz: TmopMetric316,
            gamma,
        }
    }
}

impl TmopQualityMetric3D for TmopMetric334 {
    fn eval_w(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        (1.0 - self.gamma) * self.sh.eval_w(jpt) + self.gamma * self.sz.eval_w(jpt)
    }

    fn eval_p(&self, jpt: &[[f64; 3]; 3], p: &mut [[f64; 3]; 3]) {
        let mut pt = [[0.0_f64; 3]; 3];
        self.sh.eval_p(jpt, &mut pt);
        for r in 0..3 {
            for c in 0..3 {
                p[r][c] = (1.0 - self.gamma) * pt[r][c];
            }
        }
        self.sz.eval_p(jpt, &mut pt);
        for r in 0..3 {
            for c in 0..3 {
                p[r][c] += self.gamma * pt[r][c];
            }
        }
    }

    fn assemble_h(&self, jpt: &[[f64; 3]; 3], ds: &[[f64; 3]], weight: f64, a: &mut [f64]) {
        self.sh.assemble_h(jpt, ds, weight * (1.0 - self.gamma), a);
        self.sz.assemble_h(jpt, ds, weight * self.gamma, a);
    }

    fn id(&self) -> i32 {
        334
    }
}

/// TMOP_Metric_338: `mu_302 + lambda mu_318` with
/// lambda = 0.5 (4/9 + 3) (MFEM `TMOP_Metric_338` — "4/9 <= lambda <= 3 should
/// produce best asymptotic balance").
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric338 {
    sh: TmopMetric302,
    sz: TmopMetric318,
}

impl TmopMetric338 {
    /// MFEM `AddQualityMetric(sz_metric, 0.5 * (4.0/9.0 + 3.0))`.
    pub const SZ_WEIGHT: f64 = 0.5 * (4.0 / 9.0 + 3.0);

    pub fn new() -> Self {
        Self {
            sh: TmopMetric302,
            sz: TmopMetric318,
        }
    }
}

impl Default for TmopMetric338 {
    fn default() -> Self {
        Self::new()
    }
}

impl TmopQualityMetric3D for TmopMetric338 {
    fn eval_w(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        self.sh.eval_w(jpt) + Self::SZ_WEIGHT * self.sz.eval_w(jpt)
    }

    fn eval_p(&self, jpt: &[[f64; 3]; 3], p: &mut [[f64; 3]; 3]) {
        let mut pt = [[0.0_f64; 3]; 3];
        self.sh.eval_p(jpt, &mut pt);
        for r in 0..3 {
            for c in 0..3 {
                p[r][c] = pt[r][c];
            }
        }
        self.sz.eval_p(jpt, &mut pt);
        for r in 0..3 {
            for c in 0..3 {
                p[r][c] += Self::SZ_WEIGHT * pt[r][c];
            }
        }
    }

    fn assemble_h(&self, jpt: &[[f64; 3]; 3], ds: &[[f64; 3]], weight: f64, a: &mut [f64]) {
        self.sh.assemble_h(jpt, ds, weight, a);
        self.sz.assemble_h(jpt, ds, weight * Self::SZ_WEIGHT, a);
    }

    fn id(&self) -> i32 {
        338
    }
}

/// TMOP_Metric_342: W = |T - I|² / sqrt(det(T)) (3D barrier Shape+Size+
/// Orientation). MFEM 4.10 implements it via native AD (`mu342_ad`).
#[derive(Debug, Default, Clone, Copy)]
pub struct TmopMetric342;

impl TmopQualityMetric3D for TmopMetric342 {
    fn eval_w(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        let t = to_col_major_3x3(jpt);
        let w = [0.0; 9];
        mu342_ad(&t, &w)
    }

    fn eval_p(&self, jpt: &[[f64; 3]; 3], p: &mut [[f64; 3]; 3]) {
        let t = to_col_major_3x3(jpt);
        let w = [0.0; 9];
        let g = ad_grad(&mu342_ad::<Ad1>, &t, Some(&w[..]));
        *p = from_col_major_3x3(&[g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8]]);
    }

    fn assemble_h(&self, jpt: &[[f64; 3]; 3], ds: &[[f64; 3]], weight: f64, a: &mut [f64]) {
        let t = to_col_major_3x3(jpt);
        let w = [0.0; 9];
        let h = ad_hessian(&mu342_ad::<Ad2>, &t, Some(&w[..]));
        let ndof = ds.len();
        let mut work = vec![0.0; ndof * 3 * ndof * 3];
        default_assemble_h(&h, &flatten_3d(ds), ndof, 3, weight, &mut work);
        for (o, v) in work.iter().enumerate() {
            a[o] += v;
        }
    }

    fn id(&self) -> i32 {
        342
    }
}

/// TMOP_Metric_347: `(1-gamma) mu_304 + gamma mu_316`
/// (MFEM `TMOP_Metric_347(gamma)`; gamma = 0.5 in the drivers).
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric347 {
    sh: TmopMetric304,
    sz: TmopMetric316,
    gamma: f64,
}

impl TmopMetric347 {
    pub fn new(gamma: f64) -> Self {
        Self {
            sh: TmopMetric304,
            sz: TmopMetric316,
            gamma,
        }
    }
}

impl TmopQualityMetric3D for TmopMetric347 {
    fn eval_w(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        (1.0 - self.gamma) * self.sh.eval_w(jpt) + self.gamma * self.sz.eval_w(jpt)
    }

    fn eval_p(&self, jpt: &[[f64; 3]; 3], p: &mut [[f64; 3]; 3]) {
        let mut pt = [[0.0_f64; 3]; 3];
        self.sh.eval_p(jpt, &mut pt);
        for r in 0..3 {
            for c in 0..3 {
                p[r][c] = (1.0 - self.gamma) * pt[r][c];
            }
        }
        self.sz.eval_p(jpt, &mut pt);
        for r in 0..3 {
            for c in 0..3 {
                p[r][c] += self.gamma * pt[r][c];
            }
        }
    }

    fn assemble_h(&self, jpt: &[[f64; 3]; 3], ds: &[[f64; 3]], weight: f64, a: &mut [f64]) {
        self.sh.assemble_h(jpt, ds, weight * (1.0 - self.gamma), a);
        self.sz.assemble_h(jpt, ds, weight * self.gamma, a);
    }

    fn id(&self) -> i32 {
        347
    }
}

/// TMOP_Metric_352: W = 0.5 (det(J) - 1)² / (det(J) - tau0)
/// (3D shifted-barrier untangling form of metric 316; `tau0` is MFEM's
/// reference-bound `real_t &tau0`).
#[derive(Debug, Clone, Copy)]
pub struct TmopMetric352 {
    pub tau0: f64,
}

impl TmopQualityMetric3D for TmopMetric352 {
    fn eval_w(&self, jpt: &[[f64; 3]; 3]) -> f64 {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        let i3b = ie.get_i3b();
        0.5 * (i3b - 1.0) * (i3b - 1.0) / (i3b - self.tau0)
    }

    fn eval_p(&self, jpt: &[[f64; 3]; 3], p: &mut [[f64; 3]; 3]) {
        let jac = to_col_major_3x3(jpt);
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        let i3b = ie.get_i3b();
        // P = (c - 0.5 c²) dI3b with c = (I3b - 1)/(I3b - tau0).
        let c = (i3b - 1.0) / (i3b - self.tau0);
        let di3b = ie.get_di3b().clone();
        *p = from_col_major_3x3(&scale_array(&di3b, c - 0.5 * c * c));
    }

    fn assemble_h(&self, jpt: &[[f64; 3]; 3], ds: &[[f64; 3]], weight: f64, a: &mut [f64]) {
        // dP = (1 - c)²/(I3b - tau0) (dI3b x dI3b) + (c - 0.5 c²) ddI3b.
        let jac = to_col_major_3x3(jpt);
        let ndof = ds.len();
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        ie.set_derivative_matrix(ndof, &flatten_3d(ds));
        let i3b = ie.get_i3b();
        let c0 = 1.0 / (i3b - self.tau0);
        let c = c0 * (i3b - 1.0);
        let di3b = ie.get_di3b().clone();
        ie.assemble_tprod_xx(weight * c0 * (1.0 - c) * (1.0 - c), &di3b, a);
        ie.assemble_dd_i3b(weight * (c - 0.5 * c * c), a);
    }

    fn id(&self) -> i32 {
        352
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Flatten a 2D derivative matrix (dof x 2) into column-major Vec.
fn flatten_2d(ds: &[[f64; 2]]) -> Vec<f64> {
    let ndof = ds.len();
    let mut result = vec![0.0; ndof * 2];
    for i in 0..ndof {
        result[i + ndof * 0] = ds[i][0];
        result[i + ndof * 1] = ds[i][1];
    }
    result
}

/// Flatten a 3D derivative matrix (dof x 3) into column-major Vec.
fn flatten_3d(ds: &[[f64; 3]]) -> Vec<f64> {
    let ndof = ds.len();
    let mut result = vec![0.0; ndof * 3];
    for i in 0..ndof {
        result[i + ndof * 0] = ds[i][0];
        result[i + ndof * 1] = ds[i][1];
        result[i + ndof * 2] = ds[i][2];
    }
    result
}

/// Scale a 4-element array by a scalar.
fn scale_array<const N: usize>(arr: &[f64; N], s: f64) -> [f64; N] {
    let mut result = [0.0; N];
    for i in 0..N {
        result[i] = arr[i] * s;
    }
    result
}

/// Type alias for boxed 2D metric function.
pub type TmopMetricFn = Box<dyn Fn() -> Box<dyn TmopQualityMetric>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_001_identity() {
        let m = TmopMetric001;
        let jpt = [[1.0, 0.0], [0.0, 1.0]];
        assert!((m.eval_w(&jpt) - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_metric_002_identity() {
        let m = TmopMetric002;
        let jpt = [[1.0, 0.0], [0.0, 1.0]];
        assert!((m.eval_w(&jpt) - 0.0).abs() < 1e-14);
        assert!((m.eval_w_matrix_form(&jpt) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn test_metric_007_identity() {
        let m = TmopMetric007;
        let jpt = [[1.0, 0.0], [0.0, 1.0]];
        assert!((m.eval_w(&jpt) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn test_metric_014_identity() {
        let m = TmopMetric014;
        let jpt = [[1.0, 0.0], [0.0, 1.0]];
        assert!((m.eval_w(&jpt) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn test_metric_055_identity() {
        let m = TmopMetric055;
        let jpt = [[1.0, 0.0], [0.0, 1.0]];
        assert!((m.eval_w(&jpt) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn test_metric_301_identity() {
        let m = TmopMetric301;
        let jpt = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!((m.eval_w(&jpt) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn test_metric_303_identity() {
        let m = TmopMetric303;
        let jpt = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!((m.eval_w(&jpt) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn test_metric_315_identity() {
        let m = TmopMetric315;
        let jpt = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!((m.eval_w(&jpt) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn test_metric_360_identity() {
        let m = TmopMetric360;
        let jpt = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!((m.eval_w(&jpt) - 0.0).abs() < 1e-14);
    }
}

#[cfg(test)]
mod ad_hessian_tests {
    use super::*;

    /// Round-102 gate: the forward-forward AD Hessian (used by the AD-based
    /// metrics 085/098/342 and the A-metrics) must match a central FD of the
    /// analytic AD gradient for a non-identity target W.
    fn check_ad_hessian(mu2: &dyn Fn(&[Ad2], &[Ad2]) -> Ad2, mu1: &dyn Fn(&[Ad1], &[Ad1]) -> Ad1) {
        let t = [1.3_f64, 0.21, -0.34, 0.92];
        let w = [0.86_f64, -0.31, 0.12, 1.24];
        let h = ad_hessian(mu2, &t, Some(&w));
        let hh = 1e-6;
        let mut worst = 0.0f64;
        let mut scale = 0.0f64;
        for k in 0..4 {
            let mut tp = t;
            let mut tm = t;
            tp[k] += hh;
            tm[k] -= hh;
            let gp = ad_grad(mu1, &tp, Some(&w));
            let gm = ad_grad(mu1, &tm, Some(&w));
            for s in 0..4 {
                let fd = (gp[s] - gm[s]) / (2.0 * hh);
                worst = worst.max((h[s * 4 + k] - fd).abs());
                scale = scale.max(fd.abs());
            }
        }
        assert!(worst / scale < 1e-6, "AD Hessian rel err {}", worst / scale);
    }

    #[test]
    fn ad_hessian_matches_fd_nu14() {
        check_ad_hessian(&|x, y| nu14_ad(x, y), &|x, y| nu14_ad(x, y));
    }

    #[test]
    fn ad_hessian_matches_fd_nu50() {
        check_ad_hessian(&|x, y| nu50_ad(x, y), &|x, y| nu50_ad(x, y));
    }

    #[test]
    fn ad_hessian_matches_fd_nu11() {
        check_ad_hessian(&|x, y| nu11_ad(x, y), &|x, y| nu11_ad(x, y));
    }

    #[test]
    fn ad_hessian_matches_fd_nu51() {
        check_ad_hessian(&|x, y| nu51_ad(x, y), &|x, y| nu51_ad(x, y));
    }
}
