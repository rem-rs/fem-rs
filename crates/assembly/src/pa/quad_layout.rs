//! Quad PA slot-layout bridge (D87) — the 2-D mirror of
//! [`crate::pa::hex_layout`].
//!
//! Every quad PA kernel in this module tree numbers its element-local DOFs
//! exactly like the element layer — `QuadQ1` / `QuadQ2` / `QuadQk` — because
//! that is the layout `DofManager` (`build_q2_quad` for `p == 2`, `build_pk_quad`
//! for `p >= 3`) numbers `H1Space::element_dofs` in, and the PA kernels are
//! applied to those very `elem_dofs` slices.
//!
//! Before D87 `pa::quad_qk` carried its *own* equispaced node set **and** a
//! lexicographic slot table (`ix + iy·(p+1)`), neither of which is the element
//! layer's: `QuadQk` uses Gauss-Lobatto nodes on `[0,1]` in the H1 topological
//! slot order (vertices → edges → interior).  The hex kernels went through the
//! same migration in D77 (see [`crate::pa::hex_layout`]).
//!
//! **Reference domain:** quad is the odd one out — `QuadQk` (and therefore the
//! H1 space and the assembled matrix) lives on `[0,1]²`, while `QuadQ1`/`QuadQ2`
//! and the whole hex family use `[-1,1]^d`.  The nodes returned here are
//! therefore in `[0,1]` for `QuadQk`, which is exactly what
//! `pa::quad_qk`'s quadrature and Jacobian must be expressed in.

use fem_element::ReferenceElement;

/// `(1-D nodes, slot → tensor index)` of a tensor-product quad element — the
/// element layer's own derivation.
pub(crate) fn quad_slots(elem: &dyn ReferenceElement) -> (Vec<f64>, Vec<[usize; 2]>) {
    fem_element::lagrange::quad::quad_tensor_layout(elem)
}

/// Inverse of a slot → tensor map: `inv[ix][iy]` = element-local slot.
///
/// The sum-factorized kernels contract on the tensor grid and therefore need
/// the tensor → slot direction as well.  (The 3-D sibling is
/// [`crate::pa::hex_layout::tensor_slots`]; they are dimension-split rather than
/// duplicated.)
pub(crate) fn tensor_slots(slots: &[[usize; 2]], n1d: usize) -> Vec<Vec<usize>> {
    assert_eq!(slots.len(), n1d * n1d, "quad slot count");
    let mut inv = vec![vec![usize::MAX; n1d]; n1d];
    for (slot, t) in slots.iter().enumerate() {
        debug_assert_eq!(inv[t[0]][t[1]], usize::MAX, "duplicate quad tensor node");
        inv[t[0]][t[1]] = slot;
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_element::lagrange::factory::QuadQk;
    use fem_element::lagrange::quad::{QuadQ1, QuadQ2};

    /// Round-trip pin: rebuilding the slot coordinates from
    /// `(axis_nodes, slot_tensor)` must reproduce `dof_coords()` **bit for
    /// bit**, and `tensor_slots` must be its exact inverse.
    fn check(elem: &dyn ReferenceElement) {
        let coords = elem.dof_coords();
        let (nodes, slots) = quad_slots(elem);
        assert_eq!(slots.len(), coords.len(), "{}: slot count", elem.order());
        for (slot, t) in slots.iter().enumerate() {
            for d in 0..2 {
                assert_eq!(
                    nodes[t[d]],
                    coords[slot][d],
                    "{}: slot {slot} axis {d} must round-trip bit-exactly",
                    elem.order()
                );
            }
        }
        let n1d = nodes.len();
        assert_eq!(n1d * n1d, coords.len(), "quad element must be a full tensor grid");
        let inv = tensor_slots(&slots, n1d);
        for (slot, t) in slots.iter().enumerate() {
            assert_eq!(inv[t[0]][t[1]], slot, "tensor_slots inverse");
        }
    }

    #[test]
    fn quad_layout_round_trips_element_layer() {
        check(&QuadQ1);
        check(&QuadQ2);
        for p in 1..=5 {
            check(&QuadQk::new(p));
        }
    }

    /// `QuadQ1` / `QuadQ2` are on `[-1,1]²` while the H1 space's `QuadQk` is on
    /// `[0,1]²` — this bridge must pass each element's own domain through, not a
    /// single hard-coded one (that is precisely the D87 defect).
    #[test]
    fn quad_layout_domains_differ_between_families() {
        let (n1, _) = quad_slots(&QuadQ1);
        assert_eq!(n1, vec![-1.0, 1.0], "QuadQ1 is on [-1,1]");
        let (n2, _) = quad_slots(&QuadQ2);
        assert_eq!(n2, vec![-1.0, 0.0, 1.0], "QuadQ2 is on [-1,1]");
        let (nk, _) = quad_slots(&QuadQk::new(2));
        assert_eq!(nk, vec![0.0, 0.5, 1.0], "QuadQk is on [0,1]");
    }
}
