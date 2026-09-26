//! Curved-element geometry for the partial-assembly kernels.
//!
//! Every PA kernel builds its per-quadrature-point geometry from the element's
//! **vertices** (`hex_vertices()`, a quad's four corners, `TetP1`'s affine map)
//! — an exact representation only while the mesh is straight.  On a mesh that
//! carries a high-order geometry table those vertices describe a *different*
//! map from the one the assembled path integrates (`assembler::geo_ref_elem` /
//! `geo_ref_elem_from_mesh` → `fem_mesh::transformation::element_jacobian_at`),
//! so PA and SpMV silently disagree by an `O(h)` amount.
//!
//! D783 fixed `pa::prism_pk` this way and D808-4 extends the same recipe to the
//! hex/quad family.  The gating is deliberate: `geom_order() <= 1` returns
//! `None` so the caller keeps its analytic vertex map **bit for bit** (all the
//! straight-mesh pins depend on that), and only `geom_order() >= 2` switches to
//! the mesh's order-`g` isoparametric table.

use fem_mesh::topology::MeshTopology;

/// The `[∂x_d/∂ξ_c]` Jacobian (**rows reference**, the convention
/// [`super::prism_pk`]'s `invert_3x3`/`ref_metric` and the hex/quad kernels'
/// `jit` are written in) and the physical point, at reference point `xi`.
///
/// `element_jacobian_at` hands back `J[i][j] = ∂x_i/∂ξ_j` (rows physical), so
/// the result is its transpose.  Returns `None` for a straight mesh
/// (`geom_order() <= 1`), where the caller keeps its analytic map unchanged.
pub(crate) fn curved_jacobian<const D: usize, M: MeshTopology + ?Sized>(
    mesh: &M,
    elem: u32,
    xi: &[f64; D],
) -> Option<([[f64; D]; D], [f64; D])> {
    if mesh.geom_order() < 2 {
        return None;
    }
    let (j, xp) = fem_mesh::transformation::element_jacobian_at(mesh, elem, xi, D);
    let mut jac = [[0.0_f64; D]; D];
    for c in 0..D {
        for d in 0..D {
            jac[c][d] = j[(d, c)];
        }
    }
    let mut x = [0.0_f64; D];
    x.copy_from_slice(&xp[..D]);
    Some((jac, x))
}

/// Determinant and inverse of a 3×3 Jacobian whose **rows** are `∂x/∂ξ_c`
/// (`curved_jacobian`'s convention: row `c` is the physical vector `∂x/∂ξ_c`).
///
/// Returns `(det, Jinv)` with `Jinv[r][c] = ∂ξ_r/∂x_c` (the plain matrix
/// inverse) — exactly the `∇_phys = Jinv·∇_ref` transform the simplex PA
/// kernels store (`prism_pk`'s reference metric takes these columns as the
/// physical gradients of the reference coordinates).  The pre-D770 prism code
/// clamped to `max(det, 1e-30)` and took `|det|`, which quietly produced `1e30`
/// entries for inverted prisms instead of the negative-signed operator the
/// assembled path assembles (D679's signed convention).
///
/// D808-4-r78: moved here from `prism_pk` (its only home until then) so that
/// [`super::tet4`]'s curved branch and `prism_pk` share one inversion.
pub(crate) fn invert_3x3(j: &[[f64; 3]; 3]) -> (f64, [[f64; 3]; 3]) {
    let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
        - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
        + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
    let inv = 1.0 / det;
    let c = [
        [
            (j[1][1] * j[2][2] - j[1][2] * j[2][1]) * inv,
            (j[0][2] * j[2][1] - j[0][1] * j[2][2]) * inv,
            (j[0][1] * j[1][2] - j[0][2] * j[1][1]) * inv,
        ],
        [
            (j[1][2] * j[2][0] - j[1][0] * j[2][2]) * inv,
            (j[0][0] * j[2][2] - j[0][2] * j[2][0]) * inv,
            (j[0][2] * j[1][0] - j[0][0] * j[1][2]) * inv,
        ],
        [
            (j[1][0] * j[2][1] - j[1][1] * j[2][0]) * inv,
            (j[0][1] * j[2][0] - j[0][0] * j[2][1]) * inv,
            (j[0][0] * j[1][1] - j[0][1] * j[1][0]) * inv,
        ],
    ];
    (det, c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    /// The gate itself: `None` on a straight mesh (the caller then runs its
    /// verbatim pre-D808-4 vertex code, so no straight-mesh number can move),
    /// `Some` once a geometry table is attached.
    #[test]
    fn d808_4_gate_is_geom_order_two() {
        let straight = Mesh::<3>::unit_cube_hex(2);
        assert_eq!(straight.geom_order(), 1);
        assert!(
            curved_jacobian(&straight, 0, &[0.5, 0.5, 0.5]).is_none(),
            "a straight mesh must take the analytic vertex path"
        );
        assert!(curved_jacobian(&Mesh::<2>::unit_square_quad(2), 0, &[0.5, 0.5]).is_none());

        let mut curved = Mesh::<3>::unit_cube_hex(2);
        curved.set_curvature(2);
        assert_eq!(curved.geom_order(), 2);
        let (jac, xp) = curved_jacobian(&curved, 0, &[0.5, 0.5, 0.5])
            .expect("geom_order >= 2 must take the isoparametric path");
        assert!(jac.iter().flatten().all(|v| v.is_finite()));
        assert!(xp.iter().all(|v| v.is_finite()));
        // The vertex (P1) map and the table map are genuinely different here:
        // moving only the non-vertex geometry nodes leaves the corners exact,
        // so the difference shows up away from them.  The displacement must be
        // *smooth* — a uniform shift of every non-vertex node is degenerate for
        // `J` at the element centre (`∂/∂ξ Σ_vertex φ = 0` there), which is
        // exactly the trap the smooth fixture bulge avoids.
        let mut moved = curved.clone();
        let g = moved.geometry.as_mut().unwrap();
        for node in curved.n_nodes()..g.n_nodes {
            let c: [f64; 3] = std::array::from_fn(|d| g.coords[node * 3 + d]);
            let s = (std::f64::consts::PI * c[0]).sin()
                * (std::f64::consts::PI * c[1]).sin()
                * (std::f64::consts::PI * c[2]).sin();
            g.coords[node * 3] += 0.05 * s;
            g.coords[node * 3 + 1] += 0.04 * s;
        }
        let (jac2, xp2) = curved_jacobian(&moved, 0, &[0.5, 0.5, 0.5]).unwrap();
        let d = jac
            .iter()
            .flatten()
            .zip(jac2.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let dx = xp.iter().zip(xp2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
        assert!(
            d > 1e-3 && dx > 1e-3,
            "the geometry table must drive the result: |ΔJ|={d:.3e}, |Δx|={dx:.3e}"
        );
    }
}
