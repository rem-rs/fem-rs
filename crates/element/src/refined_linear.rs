//! RefinedLinear macro-elements: piecewise-linear functions on the *once
//! refined* reference simplex (MFEM `RefinedLinearFECollection`, `Name() ==
//! "RefinedLinear"`, `fem/fe_coll.hpp:1412`).
//!
//! The reference segment/triangle/tetrahedron is subdivided (segment in 2,
//! triangle in 4, tetrahedron in 8 conforming children) and the basis is the
//! plain P1 nodal basis of that refined mesh, re-indexed to the macro
//! element's nodes.  The functions are continuous across the internal
//! subdivision faces, but their gradients jump there, so evaluation is
//! *piecewise* and the sub-element is selected with closed conditions
//! (`L0 >= 1.0` first-match chain) exactly as in MFEM.
//!
//! Ported verbatim from MFEM 4.10 `fem/fe/fe_fixed_order.cpp`:
//! - [`RefinedLinear1D`] — `RefinedLinear1DFiniteElement` (:3286), 3 dofs
//! - [`RefinedLinear2D`] — `RefinedLinear2DFiniteElement` (:3332), 6 dofs
//! - [`RefinedLinear3D`] — `RefinedLinear3DFiniteElement` (:3456), 10 dofs
//!
//! Node coordinates (`Nodes.IntPoint`) and the branch conditions are
//! transcribed line by line; the dof numbering follows MFEM exactly.
//!
//! On `order()`: MFEM constructs these as
//! `NodalFiniteElement(dim, geom, dof, 4 or 5)` — the stored order (4, 5, 4)
//! is MFEM's conservative degree/`GetOrder()` for the piecewise-linear macro
//! basis (the *shape degree is 1 on each child*), so it is mirrored here for
//! parity with `FiniteElement::GetOrder()`.

use crate::quadrature;
use crate::reference::{QuadratureRule, ReferenceElement};

// ─── 1D (3 dofs: ends + midpoint of the halved segment) ─────────────────────

/// `RefinedLinear1DFiniteElement` — 3-dof macro element on the reference
/// segment `[0,1]` split at `x = 1/2` (MFEM order tag 4).
#[derive(Debug, Clone, Copy, Default)]
pub struct RefinedLinear1D;

impl ReferenceElement for RefinedLinear1D {
    fn dim(&self) -> u8 {
        1
    }
    /// MFEM declares `NodalFiniteElement(1, SEGMENT, 3, 4)`.
    fn order(&self) -> u8 {
        4
    }
    fn n_dofs(&self) -> usize {
        3
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let x = xi[0];
        if x <= 0.5 {
            values[0] = 1.0 - 2.0 * x;
            values[1] = 0.0;
            values[2] = 2.0 * x;
        } else {
            values[0] = 0.0;
            values[1] = 2.0 * x - 1.0;
            values[2] = 2.0 - 2.0 * x;
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        // grads[i * 1 + 0] = dphi_i/dx, MFEM dshape(i,0).
        let x = xi[0];
        if x <= 0.5 {
            grads[0] = -2.0;
            grads[1] = 0.0;
            grads[2] = 2.0;
        } else {
            grads[0] = 0.0;
            grads[1] = 2.0;
            grads[2] = -2.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        quadrature::seg_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![vec![0.0], vec![1.0], vec![0.5]]
    }
}

// ─── 2D (6 dofs; reference triangle split in 4 sub-triangles) ───────────────

/// `RefinedLinear2DFiniteElement` — 6-dof macro element on the reference
/// triangle split in 4 sub-triangles (MFEM order tag 5).
///
/// MFEM subdivision (dof numbering):
/// ```text
/// T0 - 0,3,5   T1 - 1,3,4
/// T2 - 2,4,5   T3 - 3,4,5
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct RefinedLinear2D;

impl ReferenceElement for RefinedLinear2D {
    fn dim(&self) -> u8 {
        2
    }
    /// MFEM declares `NodalFiniteElement(2, TRIANGLE, 6, 5)`.
    fn order(&self) -> u8 {
        5
    }
    fn n_dofs(&self) -> usize {
        6
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let l0 = 2.0 * (1.0 - xi[0] - xi[1]);
        let l1 = 2.0 * xi[0];
        let l2 = 2.0 * xi[1];

        for v in values.iter_mut() {
            *v = 0.0;
        }

        if l0 >= 1.0 {
            // T0
            values[0] = l0 - 1.0;
            values[3] = l1;
            values[5] = l2;
        } else if l1 >= 1.0 {
            // T1
            values[3] = l0;
            values[1] = l1 - 1.0;
            values[4] = l2;
        } else if l2 >= 1.0 {
            // T2
            values[5] = l0;
            values[4] = l1;
            values[2] = l2 - 1.0;
        } else {
            // T3
            values[3] = 1.0 - l2;
            values[4] = 1.0 - l0;
            values[5] = 1.0 - l1;
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        // grads[i * 2 + j] = dphi_i/dxi_j, MFEM dshape(i,j).
        let l0 = 2.0 * (1.0 - xi[0] - xi[1]);
        let l1 = 2.0 * xi[0];
        let l2 = 2.0 * xi[1];

        // DL0 = (-2,-2), DL1 = (2,0), DL2 = (0,2).
        const DL: [[f64; 2]; 3] = [[-2.0, -2.0], [2.0, 0.0], [0.0, 2.0]];

        for g in grads.iter_mut() {
            *g = 0.0;
        }

        if l0 >= 1.0 {
            // T0: d0 = DL0, d3 = DL1, d5 = DL2
            grads[0 * 2..0 * 2 + 2].copy_from_slice(&DL[0]);
            grads[3 * 2..3 * 2 + 2].copy_from_slice(&DL[1]);
            grads[5 * 2..5 * 2 + 2].copy_from_slice(&DL[2]);
        } else if l1 >= 1.0 {
            // T1: d3 = DL0, d1 = DL1, d4 = DL2
            grads[3 * 2..3 * 2 + 2].copy_from_slice(&DL[0]);
            grads[1 * 2..1 * 2 + 2].copy_from_slice(&DL[1]);
            grads[4 * 2..4 * 2 + 2].copy_from_slice(&DL[2]);
        } else if l2 >= 1.0 {
            // T2: d5 = DL0, d4 = DL1, d2 = DL2
            grads[5 * 2..5 * 2 + 2].copy_from_slice(&DL[0]);
            grads[4 * 2..4 * 2 + 2].copy_from_slice(&DL[1]);
            grads[2 * 2..2 * 2 + 2].copy_from_slice(&DL[2]);
        } else {
            // T3: d3 = -DL2, d4 = -DL0, d5 = -DL1
            for (j, g) in grads[3 * 2..3 * 2 + 2].iter_mut().enumerate() {
                *g = -DL[2][j];
            }
            for (j, g) in grads[4 * 2..4 * 2 + 2].iter_mut().enumerate() {
                *g = -DL[0][j];
            }
            for (j, g) in grads[5 * 2..5 * 2 + 2].iter_mut().enumerate() {
                *g = -DL[1][j];
            }
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        quadrature::tri_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.5, 0.0],
            vec![0.5, 0.5],
            vec![0.0, 0.5],
        ]
    }
}

// ─── 3D (10 dofs; reference tetrahedron split in 8 sub-tetrahedra) ──────────

/// Copy gradient row `dl` into dof row `row` of `grads` (`grads[i*3..i*3+3]`).
fn put3(grads: &mut [f64], row: usize, dl: &[f64; 3]) {
    grads[row * 3..row * 3 + 3].copy_from_slice(dl);
}

/// Copy `-dl` into dof row `row` of `grads`.
fn put3_neg(grads: &mut [f64], row: usize, dl: &[f64; 3]) {
    for (j, g) in grads[row * 3..row * 3 + 3].iter_mut().enumerate() {
        *g = -dl[j];
    }
}

/// `RefinedLinear3DFiniteElement` — 10-dof macro element on the reference
/// tetrahedron split in 8 sub-tetrahedra (MFEM order tag 4).
///
/// MFEM subdivision (dof numbering):
/// ```text
/// T0 - 0,4,5,6   T1 - 1,4,7,8
/// T2 - 2,5,7,9   T3 - 3,6,8,9
/// T4 - 4,5,6,8   T5 - 4,5,7,8
/// T6 - 5,6,8,9   T7 - 5,7,8,9
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct RefinedLinear3D;

impl ReferenceElement for RefinedLinear3D {
    fn dim(&self) -> u8 {
        3
    }
    /// MFEM declares `NodalFiniteElement(3, TETRAHEDRON, 10, 4)`.
    fn order(&self) -> u8 {
        4
    }
    fn n_dofs(&self) -> usize {
        10
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let l0 = 2.0 * (1.0 - xi[0] - xi[1] - xi[2]);
        let l1 = 2.0 * xi[0];
        let l2 = 2.0 * xi[1];
        let l3 = 2.0 * xi[2];
        let l4 = 2.0 * (xi[0] + xi[1]);
        let l5 = 2.0 * (xi[1] + xi[2]);

        for v in values.iter_mut() {
            *v = 0.0;
        }

        if l0 >= 1.0 {
            // T0
            values[0] = l0 - 1.0;
            values[4] = l1;
            values[5] = l2;
            values[6] = l3;
        } else if l1 >= 1.0 {
            // T1
            values[4] = l0;
            values[1] = l1 - 1.0;
            values[7] = l2;
            values[8] = l3;
        } else if l2 >= 1.0 {
            // T2
            values[5] = l0;
            values[7] = l1;
            values[2] = l2 - 1.0;
            values[9] = l3;
        } else if l3 >= 1.0 {
            // T3
            values[6] = l0;
            values[8] = l1;
            values[9] = l2;
            values[3] = l3 - 1.0;
        } else if l4 <= 1.0 && l5 <= 1.0 {
            // T4
            values[4] = 1.0 - l5;
            values[5] = l2;
            values[6] = 1.0 - l4;
            values[8] = 1.0 - l0;
        } else if l4 >= 1.0 && l5 <= 1.0 {
            // T5
            values[4] = 1.0 - l5;
            values[5] = 1.0 - l1;
            values[7] = l4 - 1.0;
            values[8] = l3;
        } else if l4 <= 1.0 && l5 >= 1.0 {
            // T6
            values[5] = 1.0 - l3;
            values[6] = 1.0 - l4;
            values[8] = l1;
            values[9] = l5 - 1.0;
        } else {
            // T7: (L4 >= 1) && (L5 >= 1)
            values[5] = l0;
            values[7] = l4 - 1.0;
            values[8] = 1.0 - l2;
            values[9] = l5 - 1.0;
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        // grads[i * 3 + j] = dphi_i/dxi_j, MFEM dshape(i,j).
        let l0 = 2.0 * (1.0 - xi[0] - xi[1] - xi[2]);
        let l1 = 2.0 * xi[0];
        let l2 = 2.0 * xi[1];
        let l3 = 2.0 * xi[2];
        let l4 = 2.0 * (xi[0] + xi[1]);
        let l5 = 2.0 * (xi[1] + xi[2]);

        // DL0..DL5, MFEM's table.
        const DL: [[f64; 3]; 6] = [
            [-2.0, -2.0, -2.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 2.0],
        ];

        for g in grads.iter_mut() {
            *g = 0.0;
        }

        if l0 >= 1.0 {
            // T0
            put3(grads, 0, &DL[0]);
            put3(grads, 4, &DL[1]);
            put3(grads, 5, &DL[2]);
            put3(grads, 6, &DL[3]);
        } else if l1 >= 1.0 {
            // T1
            put3(grads, 4, &DL[0]);
            put3(grads, 1, &DL[1]);
            put3(grads, 7, &DL[2]);
            put3(grads, 8, &DL[3]);
        } else if l2 >= 1.0 {
            // T2
            put3(grads, 5, &DL[0]);
            put3(grads, 7, &DL[1]);
            put3(grads, 2, &DL[2]);
            put3(grads, 9, &DL[3]);
        } else if l3 >= 1.0 {
            // T3
            put3(grads, 6, &DL[0]);
            put3(grads, 8, &DL[1]);
            put3(grads, 9, &DL[2]);
            put3(grads, 3, &DL[3]);
        } else if l4 <= 1.0 && l5 <= 1.0 {
            // T4
            put3_neg(grads, 4, &DL[5]);
            put3(grads, 5, &DL[2]);
            put3_neg(grads, 6, &DL[4]);
            put3_neg(grads, 8, &DL[0]);
        } else if l4 >= 1.0 && l5 <= 1.0 {
            // T5
            put3_neg(grads, 4, &DL[5]);
            put3_neg(grads, 5, &DL[1]);
            put3(grads, 7, &DL[4]);
            put3(grads, 8, &DL[3]);
        } else if l4 <= 1.0 && l5 >= 1.0 {
            // T6
            put3_neg(grads, 5, &DL[3]);
            put3_neg(grads, 6, &DL[4]);
            put3(grads, 8, &DL[1]);
            put3(grads, 9, &DL[5]);
        } else {
            // T7
            put3(grads, 5, &DL[0]);
            put3(grads, 7, &DL[4]);
            put3_neg(grads, 8, &DL[2]);
            put3(grads, 9, &DL[5]);
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        quadrature::tet_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.5, 0.0, 0.0],
            vec![0.0, 0.5, 0.0],
            vec![0.0, 0.0, 0.5],
            vec![0.5, 0.5, 0.0],
            vec![0.5, 0.0, 0.5],
            vec![0.0, 0.5, 0.5],
        ]
    }
}
