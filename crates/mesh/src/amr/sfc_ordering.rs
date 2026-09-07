//! MFEM `NCMesh::GridSfcOrdering2D` — Hilbert space-filling-curve element
//! ordering used by `Mesh::MakeCartesian2D(..., sfc_ordering = true)`.
//!
//! 1:1 port of `HilbertSfc2D` (mfem/mesh/ncmesh.cpp).

/// Emit `(i, j)` grid coordinates along a generalized Hilbert curve.
fn hilbert_sfc_2d(
    x: i32,
    y: i32,
    ax: i32,
    ay: i32,
    bx: i32,
    by: i32,
    coords: &mut Vec<(i32, i32)>,
) {
    let w = (ax + ay).abs();
    let h = (bx + by).abs();

    let dax = ax.signum(); // unit major direction ("right")
    let day = ay.signum();
    let dbx = bx.signum(); // unit orthogonal direction ("up")
    let dby = by.signum();

    if h == 1 {
        // trivial row fill
        let (mut x, mut y) = (x, y);
        for _ in 0..w {
            coords.push((x, y));
            x += dax;
            y += day;
        }
        return;
    }
    if w == 1 {
        // trivial column fill
        let (mut x, mut y) = (x, y);
        for _ in 0..h {
            coords.push((x, y));
            x += dbx;
            y += dby;
        }
        return;
    }

    let (mut ax2, mut ay2) = (ax / 2, ay / 2);
    let (mut bx2, mut by2) = (bx / 2, by / 2);

    let w2 = (ax2 + ay2).abs();
    let h2 = (bx2 + by2).abs();

    if 2 * w > 3 * h {
        // long case: split in two parts only
        if (w2 & 0x1) != 0 && w > 2 {
            ax2 += dax; // prefer even steps
            ay2 += day;
        }
        hilbert_sfc_2d(x, y, ax2, ay2, bx, by, coords);
        hilbert_sfc_2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by, coords);
    } else {
        // standard case: one step up, one long horizontal step, one step down
        if (h2 & 0x1) != 0 && h > 2 {
            bx2 += dbx; // prefer even steps
            by2 += dby;
        }
        hilbert_sfc_2d(x, y, bx2, by2, ax2, ay2, coords);
        hilbert_sfc_2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2, coords);
        hilbert_sfc_2d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            -bx2,
            -by2,
            -(ax - ax2),
            -(ay - ay2),
            coords,
        );
    }
}

/// MFEM `NCMesh::GridSfcOrdering2D`: `(i, j)` cell coordinates in SFC order
/// for a `width x height` grid (2*width*height entries in MFEM's flat array;
/// here as pairs).
pub fn grid_sfc_ordering_2d(width: i32, height: i32) -> Vec<(i32, i32)> {
    let mut coords = Vec::with_capacity((width * height) as usize);
    if width >= height {
        hilbert_sfc_2d(0, 0, width, 0, 0, height, &mut coords);
    } else {
        hilbert_sfc_2d(0, 0, 0, height, width, 0, &mut coords);
    }
    coords
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sfc_matches_mfem_reference() {
        // Reference outputs dumped from MFEM 4.10 GridSfcOrdering2D (C++).
        let cases: &[((i32, i32), Vec<(i32, i32)>)] = &[
            ((2, 2), vec![(0, 0), (0, 1), (1, 1), (1, 0)]),
            (
                (3, 2),
                vec![(0, 0), (0, 1), (1, 1), (2, 1), (2, 0), (1, 0)],
            ),
            (
                (4, 4),
                vec![
                    (0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (0, 3), (1, 3),
                    (1, 2), (2, 2), (2, 3), (3, 3), (3, 2), (3, 1), (2, 1),
                    (2, 0), (3, 0),
                ],
            ),
            (
                (5, 3),
                vec![
                    (0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0),
                    (2, 1), (2, 2), (3, 2), (4, 2), (4, 1), (3, 1), (3, 0),
                    (4, 0),
                ],
            ),
        ];
        for ((w, h), expect) in cases {
            assert_eq!(grid_sfc_ordering_2d(*w, *h), *expect, "{w}x{h}");
        }
    }

    #[test]
    fn sfc_covers_all_cells() {
        for (w, h) in [(3, 2), (4, 4), (5, 3), (7, 4), (8, 8)] {
            let mut sfc = grid_sfc_ordering_2d(w, h);
            assert_eq!(sfc.len(), (w * h) as usize);
            sfc.sort_unstable();
            sfc.dedup();
            assert_eq!(sfc.len(), (w * h) as usize, "duplicate cells for {w}x{h}");
        }
    }
}
