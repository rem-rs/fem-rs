//! Kershaw mesh transformation (MFEM `miniapps/common/mesh_extras.hpp`,
//! `common::KershawTransformation`, `smooth = 1`).
//!
//! Maps the unit square/cube to a mesh with six layers of highly anisotropic
//! elements; `eps_y`/`eps_z` in (0, 1] control the anisotropy toward the
//! left (`eps → 0` = extreme) and the right (`eps = 1` = untransformed).
//! Used by the diag-smoothers miniapps (`-Ky` / `-Kz` options) and TMOP
//! benchmarks to test solver robustness on bad-quality meshes.
//!
//! The C++ class is a `VectorCoefficient` evaluated at the *physical* point
//! of the (identity) reference map, so applying it to every mesh vertex
//! reproduces `Mesh::Transform(kershawT)`.

/// 1D transformation toward the right boundary of the unit interval.
#[inline]
fn right(eps: f64, x: f64) -> f64 {
    if x <= 0.5 { (2.0 - eps) * x } else { 1.0 + eps * (x - 1.0) }
}

/// 1D transformation toward the left boundary of the unit interval.
#[inline]
fn left(eps: f64, x: f64) -> f64 {
    1.0 - right(eps, 1.0 - x)
}

/// Linear transition from `a` (at 0) to `b` (at 1); MFEM `smooth = 1`.
#[inline]
fn step(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        a
    } else if x >= 1.0 {
        b
    } else {
        a + (b - a) * x
    }
}

/// Evaluate the Kershaw map at physical point `x` (dimension 2 or 3).
///
/// `eps_y` must lie in (0, 1]; for 3D, `eps_z` as well.  In 2D `eps_z` is
/// ignored (MFEM forces `eps_z = 0`, i.e. the identity in z).
pub fn kershaw_map<const D: usize>(x: [f64; D], eps_y: f64, eps_z: f64) -> [f64; D] {
    assert!(D == 2 || D == 3, "Kershaw transformation only works for 2D and 3D meshes");
    let (x0, y0, z0) = (
        x[0],
        x[1],
        if D == 3 { x[2] } else { 0.0 },
    );

    // The x-range is split in 6 layers going from left-to-left, left-to-right,
    // right-to-left (2 layers), left-to-right and right-to-right yz-faces.
    let layer = (x0 * 6.0) as i32;
    let lambda = (x0 - layer as f64 / 6.0) * 6.0;

    let y = match layer {
        0 => left(eps_y, y0),
        1 | 4 => step(left(eps_y, y0), right(eps_y, y0), lambda),
        2 => step(right(eps_y, y0), left(eps_y, y0), lambda / 2.0),
        3 => step(right(eps_y, y0), left(eps_y, y0), (1.0 + lambda) / 2.0),
        _ => right(eps_y, y0),
    };

    let mut out = [0.0; D];
    out[0] = x0;
    out[1] = y;
    if D == 3 {
        let z = match layer {
            0 => left(eps_z, z0),
            1 | 4 => step(left(eps_z, z0), right(eps_z, z0), lambda),
            2 => step(right(eps_z, z0), left(eps_z, z0), lambda / 2.0),
            3 => step(right(eps_z, z0), left(eps_z, z0), (1.0 + lambda) / 2.0),
            _ => right(eps_z, z0),
        };
        out[2] = z;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The right face (x = 1) and the left face (x = 0) stay in place; with
    /// eps = 1 the map is the identity everywhere (mesh quality 1).
    #[test]
    fn kershaw_identity_at_eps_one_and_boundaries_fixed() {
        for p in [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.13, 0.77]] {
            let q = kershaw_map::<2>(p, 1.0, 0.0);
            for k in 0..2 {
                assert!((q[k] - p[k]).abs() < 1e-14, "eps=1 must be identity: {p:?} -> {q:?}");
            }
        }
        // Layer boundaries keep their x and their face y values.
        for xv in [0.0f64, 1.0] {
            for yv in [0.0f64, 1.0] {
                let q = kershaw_map::<2>([xv, yv], 0.3, 0.0);
                assert!((q[0] - xv).abs() < 1e-14);
                assert!((q[1] - yv).abs() < 1e-14);
            }
        }
    }

    /// eps < 1 pulls the interior toward the left face (anisotropy).
    #[test]
    fn kershaw_anisotropy_midpoint() {
        // x = 0.5 lies at the boundary between layers 2 and 3: left(eps, y)
        // transition; y = 0.5 midpoint maps to 1 - right(eps, 0.5) = 1/2.
        let q = kershaw_map::<2>([0.5, 0.5], 0.2, 0.0);
        assert!((q[0] - 0.5).abs() < 1e-14);
        assert!((q[1] - 0.5).abs() < 1e-14);
        // Just inside layer 0 (x in [0, 1/6)), y is compressed toward 0.
        let q = kershaw_map::<2>([0.05, 0.5], 0.2, 0.0);
        assert!(q[1] < 0.5 - 1e-3, "left layer must compress: {}", q[1]);
    }
}
