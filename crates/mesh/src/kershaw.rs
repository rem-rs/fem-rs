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
//! reproduces `Mesh::Transform(kershawT)` — **on a regular `[0,1]^D`
//! Cartesian mesh**, the input MFEM's own doc comment prescribes.  That
//! equivalence is pinned vertex-for-vertex against MFEM 4.10 by
//! `d388_regular_mesh_mfem_parity` below (probe
//! `tmp/d388/d388_probe.cpp`, dump `data/d388_kershaw_dump.txt`).
//!
//! D388 pin (non-regular meshes): MFEM's `Eval` reads the *current* physical
//! position through `T.Transform(ip, pos)` and forms `layer = x*6.0` from it.
//! On a mesh that has already been transformed (or refined after a first
//! transform, or is not the unit box at all), those positions leave
//! `[0,1]^D`, the layer index runs past the six documented layers and
//! `lambda` leaves `[0,1]` — the C++ map then folds the geometry onto
//! overlapping layers (self-intersecting cells, down to NaNs in extreme
//! cases).  That self-interacting behaviour is C++-side emergent behaviour,
//! **not** part of this port's contract: `kershaw_map` is the pure point map
//! of the reference transformation, which is the only piece the 1:1
//! miniapp drivers (`-Ky`/`-Kz` diag-smoothers, TMOP benchmarks) apply to a
//! fresh Cartesian mesh.

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

    /// D388: the `Mesh::Transform(kershawT)` equivalence, pinned
    /// vertex-for-vertex against MFEM 4.10 on regular Cartesian `[0,1]^D`
    /// meshes.  The fixture `data/d388_kershaw_dump.txt` was produced by the
    /// C++ probe `tmp/d388/d388_probe.cpp` (mfem 4.10 serial,
    /// `miniapps/common/mesh_extras.hpp`): `common::KershawTransformation`
    /// with `smooth = 1`, `eps_y = 0.3` (2-D: `eps_z = 0` ignored; 3-D:
    /// `eps_z = 0.2`), applied through `Mesh::Transform` to
    /// `MakeCartesian2D(12,4)`, `MakeCartesian2D(5,3)` (a grid that straddles
    /// the six layers) and `MakeCartesian3D(12,4,4)`.  Each section lists the
    /// transformed coordinates of vertex `i` in MFEM's `VTX` lattice order;
    /// this test rebuilds the *untransformed* lattice coordinate from `i`,
    /// evaluates [`kershaw_map`], and demands the same point.
    #[test]
    fn d388_regular_mesh_mfem_parity() {
        let path = format!("{}/../../data/d388_kershaw_dump.txt", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read {path}: {e}"));

        // Section state machine: "==" header → dims, data lines → compare.
        let mut dims = [0usize; 3];
        let mut dim = 0usize;
        let mut checked = 0usize;
        let mut total = 0usize;
        for line in text.lines() {
            if let Some(hdr) = line.strip_prefix("== ") {
                let toks: Vec<&str> = hdr.trim().split_whitespace().collect();
                dim = if toks[0].starts_with("3D") { 3 } else { 2 };
                let grid: Vec<usize> = toks[1].split('x').map(|t| t.parse().unwrap()).collect();
                dims = [grid[0], grid[1], if dim == 3 { grid[2] } else { 0 }];
                checked = 0;
                continue;
            }
            if dim == 0 || line.trim().is_empty() {
                continue;
            }
            let v: Vec<f64> = line
                .split_whitespace()
                .skip(1) // leading vertex id
                .map(|t| t.parse().unwrap())
                .collect();
            let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
            // MFEM VTX order: x fastest, then y, then z.
            let i = checked;
            let base: [f64; 3] = match dim {
                2 => [(i % (nx + 1)) as f64 / nx as f64,
                      (i / (nx + 1)) as f64 / ny as f64,
                      0.0],
                3 => [(i % (nx + 1)) as f64 / nx as f64,
                      (i / (nx + 1) % (dims[1] + 1)) as f64 / dims[1] as f64,
                      (i / ((nx + 1) * (dims[1] + 1))) as f64 / nz as f64],
                _ => unreachable!(),
            };
            let q = if dim == 2 {
                let q = kershaw_map::<2>([base[0], base[1]], 0.3, 0.0);
                [q[0], q[1], 0.0]
            } else {
                let q = kershaw_map::<3>([base[0], base[1], base[2]], 0.3, 0.2);
                [q[0], q[1], q[2]]
            };
            for k in 0..dim {
                assert!(
                    (q[k] - v[k]).abs() < 1e-14,
                    "vertex {i} (2D/3D={dim}) component {k}: rust {:.17e} vs mfem {:.17e} (base {base:?})",
                    q[k], v[k]
                );
            }
            checked += 1;
            total += 1;
        }
        // 65 + 24 + 325 vertices across the three sections.
        assert_eq!(total, 414, "fixture must carry 414 transformed vertices");
    }
}
