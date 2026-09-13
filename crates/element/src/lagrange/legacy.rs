//! MFEM's **legacy** (`closed-uniform`) reference elements.
//!
//! MFEM's fixed-order finite element collections — `Linear`, `Quadratic` and
//! `Cubic` (`fem/fe_coll.hpp`) — place their DOFs at the *closed-uniform*
//! (equispaced) points instead of the closed Gauss-Lobatto points
//! `H1_FECollection` uses, so a mesh whose `nodes` section was written with one
//! of those names cannot be handed to the Gauss-Lobatto elements unchanged.
//! This module provides the closed-uniform twins of the Gauss-Lobatto elements
//! the rest of the library uses, so `fem-io` can re-interpolate the stored
//! values with an exact change of basis (see
//! `fem_io::mfem::rewrite_legacy_nodes`, D112).
//!
//! For the quadrilateral, triangle and tetrahedron the legacy element is
//! *exactly* "the H1 slot layout with equispaced nodes", which the
//! `new_closed_uniform` constructors of [`QuadQk`], [`H1TriPk`] and
//! [`H1TetPk`] provide:
//!
//! * `BiQuadratic2DFiniteElement` / `BiCubic2DFiniteElement` enumerate their
//!   DOFs like `H1_QuadrilateralElement` (`DofForGeometry(SQUARE) = 4/9/16`,
//!   edges bottom → right → top → left, interior `ix` fastest) with nodes at
//!   `i/p`;
//! * `Quadratic2DFiniteElement` / `Cubic2DFiniteElement` enumerate theirs like
//!   `H1_TriangleElement` (edge 2 counted from its `v2` end) with nodes at
//!   `i/p`;
//! * `Quadratic3DFiniteElement` / `Cubic3DFiniteElement` enumerate theirs like
//!   `H1_TetrahedronElement` (`h1_tet_slot_labels`) with nodes on the integer
//!   barycentric lattice.
//!
//! The **hexahedron** is the exception: `LagrangeHexFiniteElement(3)` carries a
//! hand-written tensor-index table (`fem/fe/fe_fixed_order.cpp`, the
//! `degree == 3` branch) whose *interior* block is **not** the
//! `H1_HexahedronElement` enumeration — slots 58/59 and 62/63 carry the tensor
//! nodes `(2,2,1)`/`(1,2,1)` and `(2,2,2)`/`(1,2,2)` in the opposite order.
//! [`LegacyHexQ3`] therefore reproduces that table verbatim.

use crate::lagrange::factory::{HexQk, Lagrange1D};
use crate::reference::{QuadratureRule, ReferenceElement};

/// MFEM `LagrangeHexFiniteElement(3)`'s tensor-node table, in the reference
/// element's own DOF order (`fem/fe/fe_fixed_order.cpp`, `LagrangeHexFinite-
/// Element::LagrangeHexFiniteElement`, the `degree == 3` branch — the `I`, `J`,
/// `K` arrays).
///
/// Each entry is the *tensor index* (0..=3 = the equispaced node `i/3` on
/// `[0,1]`) of that DOF.  MFEM stores the same table as a permutation of the
/// 1-D `Lagrange1DFiniteElement(3)` node array, whose order is
/// `0, 1, 1/3, 2/3` — i.e. label `0 → 0`, `1 → 1`, `2 → 1/3`, `3 → 2/3` — hence
/// the label/index map `label ℓ → index [0,3,1,2][ℓ]` applied below.
pub const LAGRANGE_HEX_Q3_SLOTS: [[usize; 3]; 64] = [
    // vertices
    [0, 0, 0],
    [3, 0, 0],
    [3, 3, 0],
    [0, 3, 0],
    [0, 0, 3],
    [3, 0, 3],
    [3, 3, 3],
    [0, 3, 3],
    // edges: (0,1), (1,2), (3,2), (0,3), (4,5), (5,6), (7,6), (4,7), (0,4),
    // (1,5), (2,6), (3,7)
    [1, 0, 0],
    [2, 0, 0],
    [3, 1, 0],
    [3, 2, 0],
    [1, 3, 0],
    [2, 3, 0],
    [0, 1, 0],
    [0, 2, 0],
    [1, 0, 3],
    [2, 0, 3],
    [3, 1, 3],
    [3, 2, 3],
    [1, 3, 3],
    [2, 3, 3],
    [0, 1, 3],
    [0, 2, 3],
    [0, 0, 1],
    [0, 0, 2],
    [3, 0, 1],
    [3, 0, 2],
    [3, 3, 1],
    [3, 3, 2],
    [0, 3, 1],
    [0, 3, 2],
    // faces: (3,2,1,0), (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7), (4,5,6,7)
    [1, 2, 0],
    [2, 2, 0],
    [1, 1, 0],
    [2, 1, 0],
    [1, 0, 1],
    [2, 0, 1],
    [1, 0, 2],
    [2, 0, 2],
    [3, 1, 1],
    [3, 2, 1],
    [3, 1, 2],
    [3, 2, 2],
    [2, 3, 1],
    [1, 3, 1],
    [2, 3, 2],
    [1, 3, 2],
    [0, 2, 1],
    [0, 1, 1],
    [0, 2, 2],
    [0, 1, 2],
    [1, 1, 3],
    [2, 1, 3],
    [1, 2, 3],
    [2, 2, 3],
    // interior (note: NOT the H1_HexahedronElement enumeration)
    [1, 1, 1],
    [2, 1, 1],
    [2, 2, 1],
    [1, 2, 1],
    [1, 1, 2],
    [2, 1, 2],
    [2, 2, 2],
    [1, 2, 2],
];

/// MFEM `LagrangeHexFiniteElement(3)` — the geometry element of the legacy
/// `Cubic` finite element collection for hexahedra.
///
/// The reference domain is `[-1,1]³` (the domain of [`HexQk`], the element the
/// rest of the library uses for hexahedral geometry), so a value table built
/// for this element can be handed to `HexQk` after the change of basis in
/// `fem_io::mfem::rewrite_legacy_nodes`.
///
/// Only order 3 exists here: MFEM's *legacy* collections are fixed-order
/// (`Linear`/`Quadratic`/`Cubic`) and at `p ≤ 2` the closed-uniform and closed
/// Gauss-Lobatto points coincide, so no reinterpretation is ever needed there.
pub struct LegacyHexQ3 {
    lag1d: Lagrange1D,
    nodes: Vec<[f64; 3]>,
}

impl Default for LegacyHexQ3 {
    fn default() -> Self {
        Self::new()
    }
}

impl LegacyHexQ3 {
    pub fn new() -> Self {
        let lag1d = Lagrange1D::from_nodes(
            (0..4).map(|i| -1.0 + 2.0 * i as f64 / 3.0).collect(),
        );
        let nodes = LAGRANGE_HEX_Q3_SLOTS
            .iter()
            .map(|s| [lag1d.nodes[s[0]], lag1d.nodes[s[1]], lag1d.nodes[s[2]]])
            .collect();
        Self { lag1d, nodes }
    }
}

impl ReferenceElement for LegacyHexQ3 {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        3
    }
    fn n_dofs(&self) -> usize {
        64
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let lx = self.lag1d.val(xi[0]);
        let ly = self.lag1d.val(xi[1]);
        let lz = self.lag1d.val(xi[2]);
        for (k, s) in LAGRANGE_HEX_Q3_SLOTS.iter().enumerate() {
            values[k] = lx[s[0]] * ly[s[1]] * lz[s[2]];
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (lx, dlx) = self.lag1d.val_d(xi[0]);
        let (ly, dly) = self.lag1d.val_d(xi[1]);
        let (lz, dlz) = self.lag1d.val_d(xi[2]);
        for (k, s) in LAGRANGE_HEX_Q3_SLOTS.iter().enumerate() {
            grads[k * 3] = dlx[s[0]] * ly[s[1]] * lz[s[2]];
            grads[k * 3 + 1] = lx[s[0]] * dly[s[1]] * lz[s[2]];
            grads[k * 3 + 2] = lx[s[0]] * ly[s[1]] * dlz[s[2]];
        }
    }

    /// Quadrature on the same `[-1,1]³` domain as [`HexQk`]; only used when the
    /// element takes part in an integration (the D112 reinterpretation itself
    /// only needs `eval_basis`).
    fn quadrature(&self, order: u8) -> QuadratureRule {
        HexQk::new(3).quadrature(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.nodes.iter().map(|c| c.to_vec()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The closed-uniform basis must be cardinal at `dof_coords`.
    #[test]
    fn legacy_hex_q3_is_nodal() {
        let e = LegacyHexQ3::new();
        let coords = e.dof_coords();
        let mut v = vec![0.0_f64; e.n_dofs()];
        for (k, c) in coords.iter().enumerate() {
            e.eval_basis(c, &mut v);
            for (j, &x) in v.iter().enumerate() {
                let want = if j == k { 1.0 } else { 0.0 };
                assert!(
                    (x - want).abs() < 1e-13,
                    "slot {k} at {c:?}: phi[{j}] = {x}"
                );
            }
        }
    }

    /// The DOFs must sit on the equispaced lattice `i/3` — i.e. every
    /// coordinate is one of `{-1, -1/3, 1/3, 1}` — and every tensor node of the
    /// `4×4×4` grid must be used exactly once.
    #[test]
    fn legacy_hex_q3_slots_cover_the_lattice() {
        let e = LegacyHexQ3::new();
        let mut seen = [[[false; 4]; 4]; 4];
        for c in e.dof_coords() {
            let idx: Vec<usize> = c
                .iter()
                .map(|&x| {
                    let i = ((x + 1.0) * 1.5).round() as i64;
                    assert!(
                        (x - (-1.0 + 2.0 * i as f64 / 3.0)).abs() < 1e-14,
                        "coordinate {x} is not on the equispaced lattice"
                    );
                    assert!((0..4).contains(&i), "coordinate {x} leaves [-1,1]");
                    i as usize
                })
                .collect();
            assert!(!seen[idx[0]][idx[1]][idx[2]], "duplicate slot {c:?}");
            seen[idx[0]][idx[1]][idx[2]] = true;
        }
        assert!(seen.iter().flatten().flatten().all(|&s| s));
    }
}
