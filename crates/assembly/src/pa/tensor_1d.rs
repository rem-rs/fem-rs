//! 1-D Lagrange basis on a nodal set, evaluated *correctly at the nodes*.
//!
//! Every tensor-product PA kernel needs ℓ_i and ℓ_i' at its quadrature points.
//! The obvious implementation — the product formula
//! `ℓ_i(x) = Π_{j≠i}(x−x_j)/(x_i−x_j)` together with
//! `ℓ_i'(x) = ℓ_i(x)·Σ_{j≠i} 1/(x−x_j)` — is only valid **off** the nodes.  At
//! a node `x_k` the second formula silently returns 0 for every `i ≠ k`
//! (`ℓ_i(x_k) = 0` multiplies a finite sum), while the true derivative is
//! `ℓ_i'(x_k) = (w_i/w_k)/(x_k−x_i) ≠ 0`.
//!
//! That case is **reachable for every even degree**: the kernels integrate with
//! `p+1` Gauss–Legendre points, which contain ξ = 0 exactly when `p` is even,
//! and 0 is a Gauss–Lobatto node for even `p`.  Before D81 the `p = 2` and
//! `p = 4` element matrices were therefore wrong by O(1e-1) while `p = 3`
//! (no coincident point) matched the assembled matrix to 3e-15.
//!
//! The node branch below uses the element layer's own closed forms — the same
//! ones in `fem_element`'s `Lagrange1D::val_d`, which is what the assembler
//! evaluates — so the PA kernels and the assembled matrix agree by
//! construction:
//!
//! ```text
//! ℓ_i(x)   = Π_{j≠i}(x−x_j)/(x_i−x_j)
//! ℓ_i'(x)  = ℓ_i(x)·(Σ_j 1/(x−x_j) − 1/(x−x_i))          (x not a node)
//! ℓ_k'(x_k)= Σ_{j≠k} 1/(x_k−x_j)                         (x = x_k)
//! ℓ_i'(x_k)= (w_i/w_k)/(x_k−x_i)                         (x = x_k, i ≠ k)
//! w_i      = 1/Π_{j≠i}(x_i−x_j)
//! ```

/// Barycentric weights `w_i = 1/Π_{j≠i}(x_i−x_j)` of a nodal set.
///
/// Built exactly like `fem_element`'s `Lagrange1D::from_nodes` (plain products,
/// no reordering) so the weights are bit-identical to the element layer's.
pub(crate) fn barycentric_weights(nodes: &[f64]) -> Vec<f64> {
    let n = nodes.len();
    let mut w = vec![1.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            if j != i {
                w[i] *= nodes[i] - nodes[j];
            }
        }
        w[i] = 1.0 / w[i];
    }
    w
}

/// `(ℓ_i(x), ℓ_i'(x))` for every node `i`, with `w` the [`barycentric_weights`]
/// of `nodes`.
///
/// The tolerance for "`x` is a node" is `1e-14`, matching the element layer's
/// `Lagrange1D::val_d`, so the two agree on which formula is used.
pub(crate) fn lagrange_1d(x: f64, nodes: &[f64], w: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = nodes.len();
    if let Some(k) = nodes.iter().position(|&xj| (x - xj).abs() < 1e-14) {
        let mut vals = vec![0.0_f64; n];
        let mut ders = vec![0.0_f64; n];
        vals[k] = 1.0;
        for i in 0..n {
            ders[i] = if i == k {
                (0..n)
                    .filter(|&j| j != k)
                    .map(|j| 1.0 / (x - nodes[j]))
                    .sum()
            } else {
                w[i] / w[k] / (x - nodes[i])
            };
        }
        return (vals, ders);
    }
    let ell: f64 = nodes.iter().map(|&xj| x - xj).product();
    let sum_inv: f64 = nodes.iter().map(|&xj| 1.0 / (x - xj)).sum();
    let mut vals = vec![0.0_f64; n];
    let mut ders = vec![0.0_f64; n];
    for i in 0..n {
        vals[i] = ell * w[i] / (x - nodes[i]);
        ders[i] = vals[i] * (sum_inv - 1.0 / (x - nodes[i]));
    }
    (vals, ders)
}

/// `(values, derivatives)` of the 1-D basis at every point of `qpts`.
pub(crate) fn basis_at_points(
    nodes: &[f64],
    qpts: &[f64],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let w = barycentric_weights(nodes);
    let mut phi = Vec::with_capacity(qpts.len());
    let mut dphi = Vec::with_capacity(qpts.len());
    for &q in qpts {
        let (v, d) = lagrange_1d(q, nodes, &w);
        phi.push(v);
        dphi.push(d);
    }
    (phi, dphi)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 1-D table must reproduce the element layer's own basis, which is the
    /// object the assembler evaluates.  At a node `HexQk`'s 3-D basis reduces to
    /// a tensor product of the 1-D basis, so evaluating at
    /// `(x, nodes[0], nodes[0])` exposes the 1-D table directly on the slots
    /// whose `(iy, iz)` tensor index is 0 (their value is 1 there).
    ///
    /// This is the D81 **discriminating** assertion: with the old product-only
    /// derivative formula it fails at every even `p` (the `p+1` Gauss–Legendre
    /// points contain ξ = 0, which is a Gauss–Lobatto node for even `p`), while
    /// odd `p` passes either way.
    #[test]
    fn tensor_1d_matches_element_layer_at_nodes_and_quadrature_points() {
        use crate::pa::hex_layout::hex_slots;
        use fem_element::lagrange::factory::HexQk;
        use fem_element::ReferenceElement;

        for p in 1..=6usize {
            let (nodes, slots) = hex_slots(&HexQk::new(p));
            let (qpts, _) = fem_element::quadrature::gauss_legendre_arbitrary(p + 1);
            // Slots whose tensor index is (ix, 0, 0): their basis value at a
            // point (x, nodes[0], nodes[0]) is exactly ℓ_ix(x).
            let probes: Vec<usize> = (0..=p)
                .map(|ix| slots.iter().position(|t| t[1] == 0 && t[2] == 0 && t[0] == ix).unwrap())
                .collect();

            let elem = HexQk::new(p);
            let n = elem.n_dofs();
            let mut vals = vec![0.0_f64; n];
            let mut grads = vec![0.0_f64; n * 3];
            let mut points: Vec<f64> = qpts.clone();
            points.extend_from_slice(&nodes); // includes the at-node case
            points.push(0.5 * (nodes[0] + nodes[1]));

            let w = barycentric_weights(&nodes);
            for &x in &points {
                let (v, d) = lagrange_1d(x, &nodes, &w);
                elem.eval_basis(&[x, nodes[0], nodes[0]], &mut vals);
                elem.eval_grad_basis(&[x, nodes[0], nodes[0]], &mut grads);
                for (ix, &slot) in probes.iter().enumerate() {
                    assert!(
                        (v[ix] - vals[slot]).abs() < 1e-13,
                        "p={p} x={x}: ℓ_{ix} value {} vs element {}",
                        v[ix], vals[slot]
                    );
                    assert!(
                        (d[ix] - grads[slot * 3]).abs() < 1e-12,
                        "p={p} x={x}: ℓ'_{ix} {} vs element {}",
                        d[ix], grads[slot * 3]
                    );
                }
            }
        }
    }

    /// Every even degree has a quadrature point sitting exactly on a node —
    /// the coincidence the at-node branch exists for.  Documented as an
    /// assertion so the branch cannot be "simplified" away.
    #[test]
    fn even_orders_have_a_quadrature_point_on_a_node() {
        for p in [2usize, 4, 6] {
            use crate::pa::hex_layout::hex_slots;
            use fem_element::lagrange::factory::HexQk;
            let (nodes, _) = hex_slots(&HexQk::new(p));
            let (qpts, _) = fem_element::quadrature::gauss_legendre_arbitrary(p + 1);
            assert!(
                qpts.iter().any(|q| nodes.iter().any(|n| (q - n).abs() < 1e-14)),
                "p={p}: expected a quadrature point on a node ({qpts:?} vs {nodes:?})"
            );
        }
    }

    /// Node values are Kronecker deltas and the derivative row sums to zero
    /// (partition of unity), at nodes and off them.
    #[test]
    fn tensor_1d_partition_of_unity() {
        let nodes = vec![-1.0, -0.4, 0.25, 1.0];
        let w = barycentric_weights(&nodes);
        for x in [-1.0, -0.4, 0.25, 1.0, -0.7, 0.0, 0.9] {
            let (v, d) = lagrange_1d(x, &nodes, &w);
            let sv: f64 = v.iter().sum();
            let sd: f64 = d.iter().sum();
            assert!((sv - 1.0).abs() < 1e-14, "x={x}: Σℓ = {sv}");
            assert!(sd.abs() < 1e-13, "x={x}: Σℓ' = {sd}");
        }
    }
}
