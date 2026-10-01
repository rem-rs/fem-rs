//! D102 — RefinedLinear (1D/2D/3D) + LinearNonConf3D hex element, pinned
//! against MFEM 4.10 ground truth (bit-exact).
//!
//! Truth source: the C++ probe `tmp/d102elc/d102_probe.cpp`, built against
//! `$HOME/mfem410_ser` (MFEM 4.10 serial) with
//! `g++ -std=c++17 -O2 -I$HOME/mfem410_ser d102_probe.cpp
//! $HOME/mfem410_ser/libmfem.a`, output archived as
//! `tmp/d102elc/d102_truth.txt` (315 lines).  The probe dumps, for MFEM's
//! `RefinedLinearFECollection` (`fem/fe_coll.hpp:1412`) and
//! `LinearNonConf3DFECollection` (`fem/fe_coll.hpp:1044`):
//! - `GetFE(geom, 1)` dof/order per geometry (`COLL` lines),
//! - element `GetDim/GetOrder/GetDof/GetGeomType` (`FE` lines),
//! - the `Nodes` table (`NODE` lines),
//! - `CalcShape` + row-major `CalcDShape` on a fixed dyadic point set (`PT`
//!   lines) that covers every piecewise sub-element branch interior *and*
//!   straddle points on both sides of every branch boundary (the RefinedLinear
//!   selectors are closed `L0 >= 1.0` / `L4 <= 1.0` chains).
//!
//! A one-off harness (`tmp/d102elc/compare.rs`, same point sets, 276 points
//! total) compared this port against the dump: worst |Δ| = 0.0 and worst
//! relative Δ = 0.0 over **all** shape/dshape entries — bit-exact, including
//! every boundary-band point.  The rows below pin representative points of
//! that run (one per branch per element, transcribed mechanically from
//! `d102_truth.txt`; formatting noise from the transcription helper is
//! normalized but every float literal is verbatim).

use fem_element::nonconforming::RotTriLinearHex;
use fem_element::refined_linear::{RefinedLinear1D, RefinedLinear2D, RefinedLinear3D};
use fem_element::ReferenceElement;

/// One probe point: reference coords, MFEM CalcShape, MFEM row-major
/// CalcDShape (`dshape[i*dim + j]`).
struct Pin {
    xi: &'static [&'static f64],
    shape: &'static [f64],
    dshape: &'static [f64],
}

fn check_pins(elem: &dyn ReferenceElement, pins: &[Pin]) {
    let dim = elem.dim() as usize;
    let nd = elem.n_dofs();
    for (k, p) in pins.iter().enumerate() {
        let xi: Vec<f64> = p.xi.iter().map(|s| **s).collect();
        assert_eq!(xi.len(), dim, "pin {k}: coord count");
        assert_eq!(p.shape.len(), nd, "pin {k}: shape len");
        assert_eq!(p.dshape.len(), nd * dim, "pin {k}: dshape len");
        let mut shape = vec![0.0; nd];
        let mut grads = vec![0.0; nd * dim];
        elem.eval_basis(&xi, &mut shape);
        elem.eval_grad_basis(&xi, &mut grads);
        for (i, (&a, &b)) in shape.iter().zip(p.shape.iter()).enumerate() {
            assert_eq!(a, b, "pin {k} xi={xi:?}: shape[{i}] rust={a} mfem={b}");
        }
        for (i, (&a, &b)) in grads.iter().zip(p.dshape.iter()).enumerate() {
            assert_eq!(a, b, "pin {k} xi={xi:?}: dshape[{i}] rust={a} mfem={b}");
        }
    }
}

fn check_nodes(elem: &dyn ReferenceElement, nodes: &[&[f64]]) {
    let dc = elem.dof_coords();
    assert_eq!(dc.len(), elem.n_dofs(), "dof_coords count");
    assert_eq!(dc.len(), nodes.len(), "nodes table count");
    for (i, n) in nodes.iter().enumerate() {
        assert_eq!(dc[i], n.to_vec(), "node {i} coords");
    }
}

// ── MFEM facts (probe `FE`/`COLL` lines) ────────────────────────────────────
// COLL RefinedLinear  geom: POINT dof=1, SEGMENT dof=3 order=4,
//                     TRIANGLE dof=6 order=5, SQUARE dof=9 order=1 (out of
//                     scope: RefinedBiLinear2D), TETRAHEDRON dof=10 order=4,
//                     CUBE dof=27 order=2 (out of scope: RefinedTriLinear3D)
// COLL LinearNonConf3D geom: TRIANGLE dof=1 (P0Triangle), SQUARE dof=1
//                     (P0Quad), TETRAHEDRON dof=4 (P1TetNonConf, not ported
//                     here — D911), CUBE dof=6 order=2 (this element)
// FE RL1 dim=1 order=4 ndof=3 ; FE RL2 dim=2 order=5 ndof=6 ;
// FE RL3 dim=3 order=4 ndof=10 ; FE RTX dim=3 order=2 ndof=6

#[test]
fn d102_refined_linear_1d_mfem_truth() {
    let e = RefinedLinear1D;
    assert_eq!((e.dim(), e.order(), e.n_dofs()), (1, 4, 3));
    check_nodes(
        &e,
        &[&[0.0], &[1.0], &[0.5]], // NODE RL1 0..2
    );
    let pins = [
        // T-left branch (x <= 1/2 interior)
        Pin {
            xi: &[&0.4375],
            shape: &[0.125, 0.0, 0.875],
            dshape: &[-2.0, 0.0, 2.0],
        },
        // x = 1/2 exactly: closed `x <= 0.5` branch must win
        Pin {
            xi: &[&0.5],
            shape: &[0.0, 0.0, 1.0],
            dshape: &[-2.0, 0.0, 2.0],
        },
        // T-right branch
        Pin {
            xi: &[&0.5625],
            shape: &[0.0, 0.125, 0.875],
            dshape: &[0.0, 2.0, -2.0],
        },
        // right endpoint
        Pin {
            xi: &[&1.0],
            shape: &[0.0, 1.0, 0.0],
            dshape: &[0.0, 2.0, -2.0],
        },
    ];
    check_pins(&e, &pins);
}

#[test]
fn d102_refined_linear_2d_mfem_truth() {
    let e = RefinedLinear2D;
    assert_eq!((e.dim(), e.order(), e.n_dofs()), (2, 5, 6));
    check_nodes(
        &e,
        &[
            &[0.0, 0.0],
            &[1.0, 0.0],
            &[0.0, 1.0],
            &[0.5, 0.0],
            &[0.5, 0.5],
            &[0.0, 0.5],
        ], // NODE RL2 0..5
    );
    let pins = [
        // T0 interior (L0 = 1.5 >= 1)
        Pin {
            xi: &[&0.125, &0.125],
            shape: &[0.5, 0.0, 0.0, 0.25, 0.0, 0.25],
            dshape: &[-2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        },
        // T1 interior (L1 = 1.25)
        Pin {
            xi: &[&0.625, &0.125],
            shape: &[0.0, 0.25, 0.0, 0.5, 0.25, 0.0],
            dshape: &[0.0, 0.0, 2.0, 0.0, 0.0, 0.0, -2.0, -2.0, 0.0, 2.0, 0.0, 0.0],
        },
        // T2 interior (L2 = 1.25)
        Pin {
            xi: &[&0.125, &0.625],
            shape: &[0.0, 0.0, 0.25, 0.0, 0.25, 0.5],
            dshape: &[0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, -2.0, -2.0],
        },
        // T3 interior (the center sub-triangle); note MFEM's signed zeros
        Pin {
            xi: &[&0.375, &0.375],
            shape: &[0.0, 0.0, 0.0, 0.25, 0.5, 0.25],
            dshape: &[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -2.0, 2.0, 2.0, -2.0, -0.0,
            ],
        },
        // T0 boundary: L0 = 2*(1 - 1/4 - 1/4) = 1.0, closed branch fires
        Pin {
            xi: &[&0.25, &0.25],
            shape: &[0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
            dshape: &[-2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        },
        // T1 boundary: L1 = 1.0 exactly (first branch chain skipped, T1 wins)
        Pin {
            xi: &[&0.5, &0.25],
            shape: &[0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
            dshape: &[0.0, 0.0, 2.0, 0.0, 0.0, 0.0, -2.0, -2.0, 0.0, 2.0, 0.0, 0.0],
        },
        // T2 boundary: L2 = 1.0 exactly
        Pin {
            xi: &[&0.25, &0.5],
            shape: &[0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
            dshape: &[0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, -2.0, -2.0],
        },
        // just past the y = 1/2 line (eps = 1/64): T2 still
        Pin {
            xi: &[&0.25, &0.515625],
            shape: &[0.0, 0.0, 0.03125, 0.0, 0.5, 0.46875],
            dshape: &[0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, -2.0, -2.0],
        },
    ];
    check_pins(&e, &pins);
}

#[test]
fn d102_refined_linear_3d_mfem_truth() {
    let e = RefinedLinear3D;
    assert_eq!((e.dim(), e.order(), e.n_dofs()), (3, 4, 10));
    check_nodes(
        &e,
        &[
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
            &[0.5, 0.0, 0.0],
            &[0.0, 0.5, 0.0],
            &[0.0, 0.0, 0.5],
            &[0.5, 0.5, 0.0],
            &[0.5, 0.0, 0.5],
            &[0.0, 0.5, 0.5],
        ], // NODE RL3 0..9
    );
    let pins = [
        // T0 boundary: L0 = 2*(1-1/2) = 1.0, closed branch fires
        Pin {
            xi: &[&0.25, &0.25, &0.0],
            shape: &[0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            dshape: &[
                -2.0, -2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
                2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        },
        // T1 boundary: L1 = 1.0 exactly
        Pin {
            xi: &[&0.5, &0.25, &0.125],
            shape: &[0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.5, 0.25, 0.0],
            dshape: &[
                0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
            ],
        },
        // T2 boundary: L2 = 1.0 exactly
        Pin {
            xi: &[&0.125, &0.5, &0.25],
            shape: &[0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.25, 0.0, 0.5],
            dshape: &[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0,
                -2.0, -2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0,
            ],
        },
        // T3 boundary: L3 = 1.0 exactly (shape(3) = L3 - 1 = 0 on the face)
        Pin {
            xi: &[&0.25, &0.125, &0.5],
            shape: &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.5, 0.25],
            dshape: &[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, -2.0, -2.0, -2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0,
            ],
        },
        // T4: L4 = L5 = 1.0 exactly — both `<=` branches close on the seam
        Pin {
            xi: &[&0.25, &0.25, &0.25],
            shape: &[0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0],
            dshape: &[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -2.0, -2.0,
                0.0, 2.0, 0.0, -2.0, -2.0, -0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0,
            ],
        },
        // T5: L4 = 1.375 >= 1, L5 = 1.0 <= 1 (closed upper seam)
        Pin {
            xi: &[&0.4375, &0.25, &0.25],
            shape: &[0.0, 0.0, 0.0, 0.0, 0.0, 0.125, 0.0, 0.375, 0.5, 0.0],
            dshape: &[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -2.0, -2.0,
                -2.0, -0.0, -0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
            ],
        },
        // T6: L4 = 0.734375 <= 1, L5 = 1.0 >= 1 (closed)
        Pin {
            xi: &[&0.125, &0.2421875, &0.2578125],
            shape: &[0.0, 0.0, 0.0, 0.0, 0.0, 0.484375, 0.265625, 0.0, 0.25, 0.0],
            dshape: &[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -2.0, -2.0,
                0.0, 2.0, 0.0, -2.0, -2.0, -0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0,
            ],
        },
        // T7: L4 = 1.390625 >= 1, L5 = 1.0 >= 1 (closed)
        Pin {
            xi: &[&0.4375, &0.2578125, &0.2421875],
            shape: &[0.0, 0.0, 0.0, 0.0, 0.0, 0.125, 0.0, 0.390625, 0.484375, 0.0],
            dshape: &[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -2.0, -2.0,
                -2.0, -0.0, -0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
            ],
        },
    ];
    check_pins(&e, &pins);
}

#[test]
fn d102_rot_tri_linear_hex_mfem_truth() {
    let e = RotTriLinearHex;
    assert_eq!((e.dim(), e.order(), e.n_dofs()), (3, 2, 6));
    check_nodes(
        &e,
        &[
            &[0.5, 0.5, 0.0], // bottom face center
            &[0.5, 0.0, 0.5], // front
            &[1.0, 0.5, 0.5], // right
            &[0.5, 1.0, 0.5], // back
            &[0.0, 0.5, 0.5], // left
            &[0.5, 0.5, 1.0], // top
        ], // NODE RTX 0..5
    );
    let pins = [
        // body center: all shapes 1/6, constant gradients
        Pin {
            xi: &[&0.25, &0.25, &0.25],
            shape: &[
                0.41666666666666663,
                0.41666666666666663,
                -0.083333333333333329,
                -0.083333333333333329,
                0.41666666666666663,
                -0.083333333333333329,
            ],
            dshape: &[
                0.33333333333333331,
                0.33333333333333331,
                -1.6666666666666665,
                0.33333333333333331,
                -1.6666666666666665,
                0.33333333333333331,
                0.33333333333333337,
                0.33333333333333331,
                0.33333333333333331,
                0.33333333333333331,
                0.33333333333333337,
                0.33333333333333331,
                -1.6666666666666665,
                0.33333333333333331,
                0.33333333333333331,
                0.33333333333333331,
                0.33333333333333331,
                0.33333333333333337,
            ],
        },
        // dof 0 node (bottom face center): nodal delta property, exactly 1
        Pin {
            xi: &[&0.5, &0.5, &0.0],
            shape: &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dshape: &[
                0.0,
                0.0,
                -2.333333333333333,
                0.0,
                -1.0,
                0.66666666666666663,
                1.0,
                0.0,
                0.66666666666666663,
                0.0,
                1.0,
                0.66666666666666663,
                -1.0,
                0.0,
                0.66666666666666663,
                0.0,
                0.0,
                -0.33333333333333326,
            ],
        },
        // body center of the cube (0.5,0.5,0.5)
        Pin {
            xi: &[&0.5, &0.5, &0.5],
            shape: &[
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
            ],
            dshape: &[
                0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0,
                0.0, 0.0, 1.0,
            ],
        },
        // asymmetric interior point
        Pin {
            xi: &[&0.75, &0.0, &0.0],
            shape: &[
                0.79166666666666663,
                0.79166666666666663,
                0.16666666666666666,
                -0.20833333333333331,
                -0.33333333333333331,
                -0.20833333333333331,
            ],
            dshape: &[
                -0.33333333333333331,
                0.66666666666666663,
                -2.333333333333333,
                -0.33333333333333331,
                -2.333333333333333,
                0.66666666666666663,
                1.6666666666666665,
                0.66666666666666663,
                0.66666666666666663,
                -0.33333333333333331,
                -0.33333333333333326,
                0.66666666666666663,
                -0.33333333333333337,
                0.66666666666666663,
                0.66666666666666663,
                -0.33333333333333331,
                0.66666666666666663,
                -0.33333333333333326,
            ],
        },
    ];
    check_pins(&e, &pins);
}

// ── Structural properties (beyond the probe points) ─────────────────────────

#[test]
fn d102_refined_linear_partition_of_unity() {
    // The macro basis sums to 1 inside the whole reference simplex, on every
    // branch (values on the internal faces agree from either side).
    let cases: [(&dyn ReferenceElement, Vec<Vec<f64>>); 3] = [
        (
            &RefinedLinear1D,
            vec![
                vec![0.0],
                vec![0.25],
                vec![0.5],
                vec![0.5 + 1e-13],
                vec![0.75],
                vec![1.0],
            ],
        ),
        (
            &RefinedLinear2D,
            vec![
                vec![0.25, 0.25],
                vec![0.5, 0.25],
                vec![0.25, 0.5],
                vec![0.125, 0.125],
                vec![0.375, 0.375],
                vec![0.25 + 1e-13, 0.5 - 1e-13],
            ],
        ),
        (
            &RefinedLinear3D,
            vec![
                vec![0.25, 0.25, 0.25],
                vec![0.4375, 0.25, 0.25],
                vec![0.125, 0.2421875, 0.2578125],
                vec![0.4375, 0.2578125, 0.2421875],
                vec![0.25, 0.25, 0.0],
                vec![0.5, 0.25, 0.125],
                vec![0.125, 0.5, 0.25],
                vec![0.25, 0.125, 0.5],
            ],
        ),
    ];
    for (elem, pts) in &cases {
        let n = elem.n_dofs();
        let mut v = vec![0.0; n];
        for p in pts {
            elem.eval_basis(p, &mut v);
            let s: f64 = v.iter().sum();
            assert!((s - 1.0).abs() < 1e-14, "POU at {p:?}: sum={s}");
        }
    }
}

#[test]
fn d102_rot_tri_linear_hex_face_center_delta() {
    // Nodal dof = face-center value: phi_i(node_j) = delta_ij exactly.
    let e = RotTriLinearHex;
    let nodes = e.dof_coords();
    let n = e.n_dofs();
    let mut v = vec![0.0; n];
    for (j, node) in nodes.iter().enumerate() {
        e.eval_basis(node, &mut v);
        for (i, x) in v.iter().enumerate() {
            let expect = if i == j { 1.0 } else { 0.0 };
            assert_eq!(*x, expect, "phi_{i} at node_{j} = {x}");
        }
    }
}

#[test]
fn d102_refined_linear_gradient_sum_zero() {
    // POU gradient: sum_i dphi_i = 0 at every point (each branch).
    let cases: [(&dyn ReferenceElement, Vec<Vec<f64>>); 3] = [
        (
            &RefinedLinear1D,
            vec![vec![0.3], vec![0.5], vec![0.5 + 1e-13], vec![0.9]],
        ),
        (
            &RefinedLinear2D,
            vec![
                vec![0.25, 0.25],
                vec![0.125, 0.125],
                vec![0.625, 0.125],
                vec![0.125, 0.625],
                vec![0.375, 0.375],
            ],
        ),
        (
            &RefinedLinear3D,
            vec![
                vec![0.25, 0.25, 0.25],
                vec![0.4375, 0.25, 0.25],
                vec![0.125, 0.2421875, 0.2578125],
                vec![0.4375, 0.2578125, 0.2421875],
                vec![0.25, 0.25, 0.0],
                vec![0.5, 0.25, 0.125],
                vec![0.125, 0.5, 0.25],
                vec![0.25, 0.125, 0.5],
            ],
        ),
    ];
    for (elem, pts) in &cases {
        let d = elem.dim() as usize;
        let n = elem.n_dofs();
        let mut g = vec![0.0; n * d];
        for p in pts {
            elem.eval_grad_basis(p, &mut g);
            for j in 0..d {
                let s: f64 = (0..n).map(|i| g[i * d + j]).sum();
                assert!((s - 0.0).abs() < 1e-14, "sum dphi at {p:?} dir {j}: {s}");
            }
        }
    }
}
