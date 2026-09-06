//! TMOP quality metric implementations.
//!
//! Each metric provides:
//! - `eval_w(jpt)` — invariant-form evaluation W(J)
//! - `eval_w_matrix_form(jpt)` — matrix-form evaluation (for validation)
//! - `eval_p(jpt, p)` — 1st Piola-Kirchhoff stress (dW/dJ)
//! - `assemble_h(jpt, ds, weight, a)` — 2nd derivative assembly
//!
//! All metrics follow MFEM's convention: J = Jpt = target→physical Jacobian.

use crate::tmop::invariants::{InvariantsEvaluator2D, InvariantsEvaluator3D};

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
// 2D A-Metrics
// ============================================================================

/// TMOP_AMetric_014: W = 0.5 * (sqrt(alpha/omega) - sqrt(omega/alpha))²
/// (2D barrier size, polyconvex)
/// where alpha = det(A), omega = det(W), A = J * W^{-1}
/// This metric requires a target Jacobian W.
#[derive(Debug, Clone, Copy)]
pub struct TmopAMetric014;

impl TmopAMetric014 {
    /// Evaluate with target Jacobian W.
    pub fn eval_w_with_target(&self, jpt: &[[f64; 2]; 2], w: &[[f64; 2]; 2]) -> f64 {
        // A = J * W^{-1}
        let det_w = det_2x2(w);
        let w_inv = [
            [w[1][1] / det_w, -w[0][1] / det_w],
            [-w[1][0] / det_w, w[0][0] / det_w],
        ];
        let a = [
            [
                jpt[0][0] * w_inv[0][0] + jpt[0][1] * w_inv[1][0],
                jpt[0][0] * w_inv[0][1] + jpt[0][1] * w_inv[1][1],
            ],
            [
                jpt[1][0] * w_inv[0][0] + jpt[1][1] * w_inv[1][0],
                jpt[1][0] * w_inv[0][1] + jpt[1][1] * w_inv[1][1],
            ],
        ];
        let alpha = det_2x2(&a);
        let omega = det_2x2(w);
        let ratio = (alpha / omega).sqrt();
        0.5 * (ratio - 1.0 / ratio).powi(2)
    }
}

/// TMOP_AMetric_050: W = [1 - cos(phi_A - phi_W)] / (sin phi_A * sin phi_W)
/// (2D barrier skew)
/// Requires target Jacobian W.
#[derive(Debug, Clone, Copy)]
pub struct TmopAMetric050;

impl TmopAMetric050 {
    /// Evaluate with target Jacobian W.
    pub fn eval_w_with_target(&self, jpt: &[[f64; 2]; 2], w: &[[f64; 2]; 2]) -> f64 {
        // Compute angles of J and W
        let phi_j = angle_2x2(jpt);
        let phi_w = angle_2x2(w);
        let diff = phi_j - phi_w;
        let cos_diff = diff.cos();
        let sin_j = phi_j.sin();
        let sin_w = phi_w.sin();
        (1.0 - cos_diff) / (sin_j * sin_w)
    }
}

/// Compute the "angle" of a 2x2 matrix (angle of its first column).
fn angle_2x2(m: &[[f64; 2]; 2]) -> f64 {
    let col1 = [m[0][0], m[1][0]];
    col1[1].atan2(col1[0])
}

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
        let i1 = ie.get_i1();
        let i2 = ie.get_i2();
        let i3 = ie.get_i3();
        let i3b = ie.get_i3b();
        let di1 = ie.get_di1().clone();
        let di2 = ie.get_di2().clone();
        let di3b = ie.get_di3b().clone();
        // P = dI1 + (1/I3) dI2 - (2*I2/I3b³) dI3b
        // dP = ddI1 + (1/I3) ddI2 - (dI2 x dI3b) * (2*I2/I3b³) / I3 ...
        // Simplified: use the same pattern as MFEM
        ie.assemble_dd_i1(weight, a);
        ie.assemble_dd_i2(weight / i3, a);
        ie.assemble_tprod_xy(-2.0 * weight * i2 / (i3 * i3b * i3), &di2, &di3b, a);
        ie.assemble_dd_i3b(-2.0 * weight * i2 / (i3 * i3b), a);
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
