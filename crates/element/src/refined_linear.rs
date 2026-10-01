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
//! - [`RefinedBiLinear2D`] — `RefinedBiLinear2DFiniteElement` (:3688), 9 dofs
//!   (the collection's SQUARE arm: piecewise-*bilinear* on the 4 sub-squares)
//! - [`RefinedTriLinear3D`] — `RefinedTriLinear3DFiniteElement` (:3835), 27
//!   dofs (the collection's CUBE arm: piecewise-*trilinear* on the 8
//!   sub-cubes)
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

// ─── 2D SQUARE arm (9 dofs; reference square split in 4 sub-squares) ────────

/// `RefinedBiLinear2DFiniteElement` — the SQUARE arm of
/// `RefinedLinearFECollection` (`fem/fe_coll.hpp:1418`): piecewise-bilinear
/// functions on the reference square `[0,1]²` split in the 4 sub-squares
/// `[0,1/2]²`-style (uniform h-refinement), 9 dofs (4 corners + 4 edge
/// midpoints + center).
///
/// MFEM subdivision (dof numbering):
/// ```text
/// T0 - 0,4,7,8   T1 - 1,4,5,8
/// T2 - 2,5,6,8   T3 - 3,6,7,8
/// ```
/// The sub-square is selected with the closed first-match chain
/// `(x <= 1/2) && (y <= 1/2)` → … exactly as in MFEM, so boundary points on
/// the internal lines select the earliest matching branch (T0 wins on the
/// seam).  Ported verbatim from MFEM 4.10 `fem/fe/fe_fixed_order.cpp:3688`
/// (`CalcShape` :3711, `CalcDShape` :3762); dof numbering follows MFEM
/// `Nodes` order.
///
/// **Faithfully preserved upstream quirk** (`fe_fixed_order.cpp:3788-3789`):
/// in the T0 branch of `CalcDShape`, `dshape(7,0)` is assigned twice — the
/// second write (`2(Lx−1)`) wins and `dshape(7,1)` stays 0.  The gradient of
/// `φ₇` returned by MFEM in T0 is therefore `(2(Lx−1), 0)` rather than the
/// analytic `(−2(2−Ly), −2(Lx−1))`; this port reproduces that behaviour
/// bit-for-bit (the d103 probe pins it at e.g. `(1/4, 1/4)` → row 7 =
/// `(1, 0)`).
///
/// MFEM declares `NodalFiniteElement(2, SQUARE, 9, 1, FunctionSpace::rQk)`,
/// so [`ReferenceElement::order`] mirrors the stored order tag 1.
#[derive(Debug, Clone, Copy, Default)]
pub struct RefinedBiLinear2D;

impl ReferenceElement for RefinedBiLinear2D {
    fn dim(&self) -> u8 {
        2
    }
    /// MFEM declares `NodalFiniteElement(2, SQUARE, 9, 1, rQk)`.
    fn order(&self) -> u8 {
        1
    }
    fn n_dofs(&self) -> usize {
        9
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];
        let lx = 2.0 * (1.0 - x);
        let ly = 2.0 * (1.0 - y);

        for v in values.iter_mut() {
            *v = 0.0;
        }

        if x <= 0.5 && y <= 0.5 {
            // T0
            values[0] = (lx - 1.0) * (ly - 1.0);
            values[4] = (2.0 - lx) * (ly - 1.0);
            values[8] = (2.0 - lx) * (2.0 - ly);
            values[7] = (lx - 1.0) * (2.0 - ly);
        } else if x >= 0.5 && y <= 0.5 {
            // T1
            values[4] = lx * (ly - 1.0);
            values[1] = (1.0 - lx) * (ly - 1.0);
            values[5] = (1.0 - lx) * (2.0 - ly);
            values[8] = lx * (2.0 - ly);
        } else if x >= 0.5 && y >= 0.5 {
            // T2
            values[8] = lx * ly;
            values[5] = (1.0 - lx) * ly;
            values[2] = (1.0 - lx) * (1.0 - ly);
            values[6] = lx * (1.0 - ly);
        } else if x <= 0.5 && y >= 0.5 {
            // T3
            values[7] = (lx - 1.0) * ly;
            values[8] = (2.0 - lx) * ly;
            values[6] = (2.0 - lx) * (1.0 - ly);
            values[3] = (lx - 1.0) * (1.0 - ly);
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        // grads[i * 2 + j] = dphi_i/dxi_j, MFEM dshape(i,j).
        let x = xi[0];
        let y = xi[1];
        let lx = 2.0 * (1.0 - x);
        let ly = 2.0 * (1.0 - y);

        for g in grads.iter_mut() {
            *g = 0.0;
        }

        if x <= 0.5 && y <= 0.5 {
            // T0
            grads[0 * 2] = 2.0 * (1.0 - ly);
            grads[0 * 2 + 1] = 2.0 * (1.0 - lx);

            grads[4 * 2] = 2.0 * (ly - 1.0);
            grads[4 * 2 + 1] = -2.0 * (2.0 - lx);

            grads[8 * 2] = 2.0 * (2.0 - ly);
            grads[8 * 2 + 1] = 2.0 * (2.0 - lx);

            // MFEM quirk (:3788-3789): the first write is dead code there;
            // kept verbatim so dshape(7,·) matches MFEM bit-for-bit.
            grads[7 * 2] = -2.0 * (2.0 - ly);
            grads[7 * 2] = 2.0 * (lx - 1.0);
        } else if x >= 0.5 && y <= 0.5 {
            // T1
            grads[4 * 2] = -2.0 * (ly - 1.0);
            grads[4 * 2 + 1] = -2.0 * lx;

            grads[1 * 2] = 2.0 * (ly - 1.0);
            grads[1 * 2 + 1] = -2.0 * (1.0 - lx);

            grads[5 * 2] = 2.0 * (2.0 - ly);
            grads[5 * 2 + 1] = 2.0 * (1.0 - lx);

            grads[8 * 2] = -2.0 * (2.0 - ly);
            grads[8 * 2 + 1] = 2.0 * lx;
        } else if x >= 0.5 && y >= 0.5 {
            // T2
            grads[8 * 2] = -2.0 * ly;
            grads[8 * 2 + 1] = -2.0 * lx;

            grads[5 * 2] = 2.0 * ly;
            grads[5 * 2 + 1] = -2.0 * (1.0 - lx);

            grads[2 * 2] = 2.0 * (1.0 - ly);
            grads[2 * 2 + 1] = 2.0 * (1.0 - lx);

            grads[6 * 2] = -2.0 * (1.0 - ly);
            grads[6 * 2 + 1] = 2.0 * lx;
        } else if x <= 0.5 && y >= 0.5 {
            // T3
            grads[7 * 2] = -2.0 * ly;
            grads[7 * 2 + 1] = -2.0 * (lx - 1.0);

            grads[8 * 2] = 2.0 * ly;
            grads[8 * 2 + 1] = -2.0 * (2.0 - lx);

            grads[6 * 2] = 2.0 * (1.0 - ly);
            grads[6 * 2 + 1] = 2.0 * (2.0 - lx);

            grads[3 * 2] = -2.0 * (1.0 - ly);
            grads[3 * 2 + 1] = 2.0 * (lx - 1.0);
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        // The element lives on MFEM's [0,1]^2 square.
        quadrature::quad_rule_01(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
            vec![0.5, 0.0],
            vec![1.0, 0.5],
            vec![0.5, 1.0],
            vec![0.0, 0.5],
            vec![0.5, 0.5],
        ]
    }
}

// ─── 3D CUBE arm (27 dofs; reference cube split in 8 sub-cubes) ─────────────

/// Branch selection shared by `eval_basis`/`eval_grad_basis` of
/// [`RefinedTriLinear3D`] — the (identical) first-match chain of MFEM's
/// `CalcShape`/`CalcDShape`: returns the sub-cube's local `(Lx, Ly, Lz)`
/// vertex barycentres and the macro-dof corner table `N[0..8]`.  All
/// conditions are closed (`<=` / `>=`), so seam points select the earliest
/// matching branch (T0 wins on the 1/2-planes).
fn rtl3_branch(x: f64, y: f64, z: f64) -> ([f64; 3], [usize; 8]) {
    if x <= 0.5 && y <= 0.5 && z <= 0.5 {
        // T0
        (
            [1.0 - 2.0 * x, 1.0 - 2.0 * y, 1.0 - 2.0 * z],
            [0, 8, 20, 11, 16, 21, 26, 24],
        )
    } else if x >= 0.5 && y <= 0.5 && z <= 0.5 {
        // T1
        (
            [2.0 - 2.0 * x, 1.0 - 2.0 * y, 1.0 - 2.0 * z],
            [8, 1, 9, 20, 21, 17, 22, 26],
        )
    } else if x <= 0.5 && y >= 0.5 && z <= 0.5 {
        // T2 (note: MFEM uses Lx = 2 - 2x here even though x <= 1/2)
        (
            [2.0 - 2.0 * x, 2.0 - 2.0 * y, 1.0 - 2.0 * z],
            [20, 9, 2, 10, 26, 22, 18, 23],
        )
    } else if x >= 0.5 && y >= 0.5 && z <= 0.5 {
        // T3
        (
            [1.0 - 2.0 * x, 2.0 - 2.0 * y, 1.0 - 2.0 * z],
            [11, 20, 10, 3, 24, 26, 23, 19],
        )
    } else if x <= 0.5 && y <= 0.5 && z >= 0.5 {
        // T4
        (
            [1.0 - 2.0 * x, 1.0 - 2.0 * y, 2.0 - 2.0 * z],
            [16, 21, 26, 24, 4, 12, 25, 15],
        )
    } else if x >= 0.5 && y <= 0.5 && z >= 0.5 {
        // T5
        (
            [2.0 - 2.0 * x, 1.0 - 2.0 * y, 2.0 - 2.0 * z],
            [21, 17, 22, 26, 12, 5, 13, 25],
        )
    } else if x <= 0.5 && y >= 0.5 && z >= 0.5 {
        // T6
        (
            [2.0 - 2.0 * x, 2.0 - 2.0 * y, 2.0 - 2.0 * z],
            [26, 22, 18, 23, 25, 13, 6, 14],
        )
    } else {
        // T7
        (
            [1.0 - 2.0 * x, 2.0 - 2.0 * y, 2.0 - 2.0 * z],
            [24, 26, 23, 19, 15, 25, 14, 7],
        )
    }
}

/// `RefinedTriLinear3DFiniteElement` — the CUBE arm of
/// `RefinedLinearFECollection` (`fem/fe_coll.hpp:1420`): piecewise-trilinear
/// functions on the reference cube `[0,1]³` split in the 8 octant sub-cubes
/// (uniform h-refinement), 27 dofs (8 corners + 12 edge midpoints + 6 face
/// centers + center).
///
/// Within each sub-cube the basis is the plain Q1 trilinear nodal basis,
/// written through the local vertex barycentres `(Lx, Ly, Lz)` and the
/// per-branch macro-dof corner table (see [`rtl3_branch`]).  The functions
/// are continuous across the internal subdivision planes, but their
/// gradients jump there.  Ported verbatim from MFEM 4.10
/// `fem/fe/fe_fixed_order.cpp:3835` (`CalcShape` :3881, `CalcDShape`
/// :4024); dof numbering follows MFEM `Nodes` order (corners, edges, faces,
/// center).
///
/// **Faithfully preserved upstream quirk**: branches T2/T3/T6/T7 use the
/// "crossed" `Lx` assignment (`Lx = 2−2x` where `x <= 1/2`, `Lx = 1−2x`
/// where `x >= 1/2` — MFEM's actual table), so there the basis is not nodal
/// at its corner nodes (e.g. `phi_10(node 3) = 2`, `phi_2(node 3) = −1`);
/// the partition of unity still sums to 1 (the 8-corner template does for
/// any `(Lx,Ly,Lz)`).  Reproduced verbatim; see the d103 probe pins.
///
/// MFEM declares `NodalFiniteElement(3, CUBE, 27, 2, FunctionSpace::rQk)`,
/// so [`ReferenceElement::order`] mirrors the stored order tag 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct RefinedTriLinear3D;

impl ReferenceElement for RefinedTriLinear3D {
    fn dim(&self) -> u8 {
        3
    }
    /// MFEM declares `NodalFiniteElement(3, CUBE, 27, 2, rQk)`.
    fn order(&self) -> u8 {
        2
    }
    fn n_dofs(&self) -> usize {
        27
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let ([lx, ly, lz], n) = rtl3_branch(xi[0], xi[1], xi[2]);

        for v in values.iter_mut() {
            *v = 0.0;
        }

        // Verbatim MFEM trilinear corner template (operation order kept),
        // stored through the branch dof table exactly as the C++ does.
        let corner = [
            lx * ly * lz,
            (1.0 - lx) * ly * lz,
            (1.0 - lx) * (1.0 - ly) * lz,
            lx * (1.0 - ly) * lz,
            lx * ly * (1.0 - lz),
            (1.0 - lx) * ly * (1.0 - lz),
            (1.0 - lx) * (1.0 - ly) * (1.0 - lz),
            lx * (1.0 - ly) * (1.0 - lz),
        ];
        for (k, &nk) in n.iter().enumerate() {
            values[nk] = corner[k];
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        // grads[i * 3 + j] = dphi_i/dxi_j, MFEM dshape(i,j).
        let ([lx, ly, lz], n) = rtl3_branch(xi[0], xi[1], xi[2]);

        for g in grads.iter_mut() {
            *g = 0.0;
        }

        // Verbatim MFEM dshape corner template (same expressions, same
        // evaluation order), written through the branch dof table.
        let dcorner = [
            [-2.0 * ly * lz, -2.0 * lx * lz, -2.0 * lx * ly],
            [
                2.0 * ly * lz,
                -2.0 * (1.0 - lx) * lz,
                -2.0 * (1.0 - lx) * ly,
            ],
            [
                2.0 * (1.0 - ly) * lz,
                2.0 * (1.0 - lx) * lz,
                -2.0 * (1.0 - lx) * (1.0 - ly),
            ],
            [
                -2.0 * (1.0 - ly) * lz,
                2.0 * lx * lz,
                -2.0 * lx * (1.0 - ly),
            ],
            [
                -2.0 * ly * (1.0 - lz),
                -2.0 * lx * (1.0 - lz),
                2.0 * lx * ly,
            ],
            [
                2.0 * ly * (1.0 - lz),
                -2.0 * (1.0 - lx) * (1.0 - lz),
                2.0 * (1.0 - lx) * ly,
            ],
            [
                2.0 * (1.0 - ly) * (1.0 - lz),
                2.0 * (1.0 - lx) * (1.0 - lz),
                2.0 * (1.0 - lx) * (1.0 - ly),
            ],
            [
                -2.0 * (1.0 - ly) * (1.0 - lz),
                2.0 * lx * (1.0 - lz),
                2.0 * lx * (1.0 - ly),
            ],
        ];
        for (k, &nk) in n.iter().enumerate() {
            grads[nk * 3..nk * 3 + 3].copy_from_slice(&dcorner[k]);
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        quadrature::hex_rule(order)
    }

    /// MFEM `Nodes.IntPoint` order: 8 corners (CCW bottom, CCW top), 12 edge
    /// midpoints, 6 face centers, center.
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            // corners
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![0.0, 1.0, 1.0],
            // edges
            vec![0.5, 0.0, 0.0],
            vec![1.0, 0.5, 0.0],
            vec![0.5, 1.0, 0.0],
            vec![0.0, 0.5, 0.0],
            vec![0.5, 0.0, 1.0],
            vec![1.0, 0.5, 1.0],
            vec![0.5, 1.0, 1.0],
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.0, 0.5],
            vec![1.0, 0.0, 0.5],
            vec![1.0, 1.0, 0.5],
            vec![0.0, 1.0, 0.5],
            // faces
            vec![0.5, 0.5, 0.0],
            vec![0.5, 0.0, 0.5],
            vec![1.0, 0.5, 0.5],
            vec![0.5, 1.0, 0.5],
            vec![0.0, 0.5, 0.5],
            vec![0.5, 0.5, 1.0],
            // element center
            vec![0.5, 0.5, 0.5],
        ]
    }
}
