//! Hex PA slot-layout bridge (D77).
//!
//! Every hex PA kernel in this module tree numbers its element-local DOFs
//! exactly like the element layer — `HexQ1` / `HexQ2` / `HexQ3` / `HexQk` —
//! because that is the layout `DofManager` (`build_q2_hex` for `p == 2`,
//! `build_pk_hex` for `p >= 3`) numbers `H1Space::element_dofs` in, and the
//! PA kernels are applied to those very `elem_dofs` slices.
//!
//! Before D77 each kernel carried its *own* hard-coded node table
//! (`HEX_Q2_MAP`, `hex_q3_ixyz`, `dofs[ix + iy·n + iz·n²]`), which had drifted
//! into three mutually different orders.  Instead of re-pinning three copies,
//! the kernels now derive `(1-D nodes, slot → tensor index)` from
//! [`ReferenceElement::dof_coords`] at kernel start: the element layer stays
//! the single source of truth and any future `HexQk` layout change (e.g. the
//! pending `HexQk(2)` MFEM-order switch) is followed automatically.
//!
//! D82 moved the derivation itself into the element crate
//! ([`fem_element::lagrange::hex::hex_tensor_layout`]) so that the GPU shader
//! generator builds its slot tables from the same function instead of a second
//! transcription of it.

use fem_element::ReferenceElement;

/// `(1-D nodes, slot → tensor index)` of a tensor-product hex element — the
/// element layer's own derivation.
pub(crate) fn hex_slots(elem: &dyn ReferenceElement) -> (Vec<f64>, Vec<[usize; 3]>) {
    fem_element::lagrange::hex::hex_tensor_layout(elem)
}

/// Inverse of a slot → tensor map: `inv[ix][iy][iz]` = element-local slot.
///
/// The sum-factorized kernels contract on the tensor grid and therefore need
/// the tensor → slot direction as well.
pub(crate) fn tensor_slots(slots: &[[usize; 3]], n1d: usize) -> Vec<Vec<Vec<usize>>> {
    assert_eq!(slots.len(), n1d * n1d * n1d, "hex slot count");
    let mut inv = vec![vec![vec![usize::MAX; n1d]; n1d]; n1d];
    for (slot, t) in slots.iter().enumerate() {
        debug_assert_eq!(inv[t[0]][t[1]][t[2]], usize::MAX, "duplicate hex tensor node");
        inv[t[0]][t[1]][t[2]] = slot;
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_element::lagrange::factory::HexQk;
    use fem_element::lagrange::hex::{HexQ1, HexQ2, HexQ3};

    /// Round-trip pin: rebuilding the slot coordinates from
    /// `(axis_nodes, slot_tensor)` must reproduce `dof_coords()` **bit for
    /// bit**, and `tensor_slots` must be its exact inverse.
    fn check(elem: &dyn ReferenceElement) {
        let coords = elem.dof_coords();
        let (nodes, slots) = hex_slots(elem);
        assert_eq!(slots.len(), coords.len(), "{}: slot count", elem.order());
        for (slot, t) in slots.iter().enumerate() {
            for d in 0..3 {
                assert_eq!(
                    nodes[t[d]], coords[slot][d],
                    "{}: slot {slot} axis {d} must round-trip bit-exactly",
                    elem.order()
                );
            }
        }
        let n1d = nodes.len();
        assert_eq!(n1d * n1d * n1d, coords.len(), "hex element must be a full tensor grid");
        let inv = tensor_slots(&slots, n1d);
        for (slot, t) in slots.iter().enumerate() {
            assert_eq!(inv[t[0]][t[1]][t[2]], slot, "tensor_slots inverse");
        }
    }

    #[test]
    fn hex_layout_round_trips_element_layer() {
        check(&HexQ1);
        check(&HexQ2);
        check(&HexQ3);
        for p in 1..=5 {
            check(&HexQk::new(p));
        }
    }

    /// The 1-D nodes are the GLL points of MFEM's `H1_FECollection`
    /// (`BasisType::GaussLobatto`), not equispaced — the equispaced tables the
    /// pre-D77 kernels used were a second, silent divergence from the element
    /// layer (and therefore from the assembled matrix).
    #[test]
    fn hex_qk_nodes_are_gauss_lobatto() {
        for p in 1..=5 {
            let (nodes, _) = hex_slots(&HexQk::new(p));
            let (want, _) = fem_element::quadrature::gauss_lobatto_arbitrary(p + 1);
            assert_eq!(nodes.len(), want.len(), "p={p}: node count");
            for (i, (got, want)) in nodes.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    got, want,
                    "p={p}: 1-D node {i} must be bit-identical to gauss_lobatto_arbitrary"
                );
            }
        }
    }

    /// `HexQ2`/`HexQ3` are already pinned to `HexQk::new(2)`/`new(3)` in the
    /// element crate; re-pin here so the PA bridge (which may point at either)
    /// cannot silently diverge from what the space numbering follows.
    #[test]
    fn hex_q2_q3_slots_match_hex_qk() {
        let (n2, s2) = hex_slots(&HexQ2);
        let (k2, t2) = hex_slots(&HexQk::new(2));
        assert_eq!((n2, s2), (k2, t2), "HexQ2 vs HexQk(2)");
        let (n3, s3) = hex_slots(&HexQ3);
        let (k3, t3) = hex_slots(&HexQk::new(3));
        assert_eq!((n3, s3), (k3, t3), "HexQ3 vs HexQk(3)");
    }
}
