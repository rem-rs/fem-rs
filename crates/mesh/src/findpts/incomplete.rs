//! Isoparametric geometry for the *incomplete quadratic* element families
//! `Quad8` / `Hex20` / `Prism15` (D244).
//!
//! MFEM has no such geometry types: its Gmsh reader only accepts the complete
//! tensor-product high-order elements (QUAD9, HEX27, PRISM18 — the `types`
//! table in `mesh/gmsh.cpp`) and `FindPointsGSLIB::Setup` (`fem/gslib.cpp`)
//! feeds gslib a full `(p+1)^d` tensor node table per element.  fem-rs does
//! import the incomplete families (Gmsh types 16/17/16, VTK quadratic
//! cells), so the locator evaluates them from their own node tables: the
//! classical serendipity (boundary-of-tensor) interpolation spaces, which
//! are unisolvent on the corners + edge nodes.
//!
//! Each family is evaluated as a nodal (Kronecker) polynomial interpolant on
//! the family's **factory reference domain** (the same convention
//! `to_factory_coords` uses, i.e. what `MeshTopology::locate` reports):
//!
//! - Quad8: `[0,1]^2`; nodes = 4 corners CCW `(0,0),(1,0),(1,1),(0,1)`, then
//!   edge nodes on `(0,1),(1,2),(2,3),(3,0)` at their midpoints.  (Gmsh type
//!   16 and `VTK_QUADRATIC_QUAD` use this same order.)
//! - Hex20: `[-1,1]^3`; nodes = 8 corners (MFEM corner order), then 12 edge
//!   midpoints in **Gmsh order** (`CartesianToGmshHex`, ref = 2):
//!   `(0,1),(0,3),(0,4),(1,2),(1,5),(2,3),(2,6),(3,7),(4,5),(4,7),(5,6),(6,7)`.
//!   Note `VTK_QUADRATIC_HEX` walks the perimeter first — for straight
//!   elements every edge correction term vanishes and the map is the corner
//!   trilinear one regardless of edge order, so the two conventions agree
//!   exactly there; curved VTK-origin Hex20 meshes would need the reader to
//!   canonicalize the order first (debt D262).
//! - Prism15: axial-first `(v, a, b)`, `v ∈ [0,1]` axial, `(a, b)` the unit
//!   right triangle (`a,b ≥ 0, a+b ≤ 1`); nodes = 6 corners (bottom `0,1,2`,
//!   top `3,4,5`), then 9 edge midpoints in Gmsh order
//!   (`WedgeToGmshPrism`, ref = 2): bottom `(0,1),(1,2)`, vertical `(0,3)`,
//!   bottom `(2,0)`, verticals `(1,4),(2,5)`, top `(3,4),(4,5),(5,3)`.
//!   (VTK order differs — bottom edges, top edges, verticals; same
//!   straight-mesh equivalence argument as for Hex20.)
//!
//! The interpolation coefficients are the inverse of the small Vandermonde
//! matrix of the serendipity monomial basis at the node table; they are
//! computed once per family and cached.  Correctness (nodal delta, partition
//! of unity, straight-edge trilinearity) is asserted by the tests in this
//! module and in `tests/d244_incomplete_families.rs`.

use std::sync::OnceLock;

use crate::element_type::ElementType;

/// Monomial exponents of one serendipity family (3-D slots; 2-D families
/// leave the third exponent and coordinate at zero).
#[derive(Clone, Copy)]
pub(crate) struct IncompleteFamily {
    /// Node count (== monomial count, unisolvent system).
    pub n: usize,
    /// Reference node positions in connectivity order (factory domain).
    pub nodes: &'static [[f64; 3]],
    /// Monomial exponents `x^i y^j z^k` spanning the interpolation space.
    pub monos: &'static [[u32; 3]],
    /// Element dimension (2 for Quad8, 3 for Hex20/Prism15).
    pub dim: usize,
}

const QUAD8_CORNERS: [[f64; 3]; 4] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
const QUAD8_EDGES: [[f64; 3]; 4] = [[0.5, 0.0, 0.0], [1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.5, 0.0]];

/// Hexahedron corners on `[-1,1]^3` (MFEM corner order 0..7).
const HEX_CORNERS: [[f64; 3]; 8] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
];

/// Edge endpoint pairs of the hexahedron in Gmsh Hex20 order.
const HEX20_GMSH_EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [0, 3],
    [0, 4],
    [1, 2],
    [1, 5],
    [2, 3],
    [2, 6],
    [3, 7],
    [4, 5],
    [4, 7],
    [5, 6],
    [6, 7],
];

/// Prism15 corners: bottom triangle `0,1,2`, top `3,4,5`; factory coords
/// `(v, a, b)` axial-first.
const PRISM15_CORNERS: [[f64; 3]; 6] = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
];

/// Edge endpoint pairs of the prism in Gmsh Prism15 order.
const PRISM15_GMSH_EDGES: [[usize; 2]; 9] = [
    [0, 1], // bottom (0,1)
    [1, 2], // bottom (1,2)
    [0, 3], // vertical (0,3)
    [2, 0], // bottom (2,0)
    [1, 4], // vertical (1,4)
    [2, 5], // vertical (2,5)
    [3, 4], // top (3,4)
    [4, 5], // top (4,5)
    [5, 3], // top (5,3)
];

fn edge_mids(corners: &[[f64; 3]], edges: &[[usize; 2]]) -> Vec<[f64; 3]> {
    edges
        .iter()
        .map(|&[a, b]| {
            std::array::from_fn(|d| 0.5 * (corners[a][d] + corners[b][d]))
        })
        .collect()
}

/// Serendipity monomials for the Quad8 `[0,1]^2` interpolant.
const QUAD8_MONOS: [[u32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [2, 0, 0],
    [1, 1, 0],
    [0, 2, 0],
    [2, 1, 0],
    [1, 2, 0],
];

/// Serendipity monomials for the Hex20 `[-1,1]^3` interpolant: the Q2 tensor
/// monomials minus the 6 face modes and the 1 center mode (at most one
/// exponent equals 2).
const HEX20_MONOS: [[u32; 3]; 20] = {
    let mut m = [[0u32; 3]; 20];
    let mut n = 0;
    let mut i = 0;
    while i <= 2 {
        let mut j = 0;
        while j <= 2 {
            let mut k = 0;
            while k <= 2 {
                let two = (i == 2) as u32 + (j == 2) as u32 + (k == 2) as u32;
                if two <= 1 {
                    m[n] = [i, j, k];
                    n += 1;
                }
                k += 1;
            }
            j += 1;
        }
        i += 1;
    }
    m
};

/// Serendipity monomials for the Prism15 interpolant: `v^k · a^i b^j`
/// (`k ≤ 2`, `i+j ≤ 2`) minus the 3 quad-face modes `v · (i+j = 2)`.
const PRISM15_MONOS: [[u32; 3]; 15] = [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 2, 0],
    [0, 1, 1],
    [0, 0, 2], // v = 0 layer (6)
    [1, 0, 0],
    [1, 1, 0],
    [1, 0, 1], // verticals (v^1 · corners)
    [2, 0, 0],
    [2, 1, 0],
    [2, 0, 1], // v = 1 layer (6)
    [1, 2, 0],
    [1, 1, 1],
    [1, 0, 2], // v^1 · edge modes (3)
];

/// Quad8 family (node table in connectivity order).
pub(crate) const QUAD8: IncompleteFamily = IncompleteFamily {
    n: 8,
    dim: 2,
    nodes: &[
        QUAD8_CORNERS[0],
        QUAD8_CORNERS[1],
        QUAD8_CORNERS[2],
        QUAD8_CORNERS[3],
        QUAD8_EDGES[0],
        QUAD8_EDGES[1],
        QUAD8_EDGES[2],
        QUAD8_EDGES[3],
    ],
    monos: &QUAD8_MONOS,
};

/// Build Hex20 node table (const, corners + Gmsh edge mids).
const HEX20_NODES: [[f64; 3]; 20] = {
    let mut n = [[0.0f64; 3]; 20];
    let mut k = 0;
    while k < 8 {
        n[k] = HEX_CORNERS[k];
        k += 1;
    }
    // Edge midpoints (const fn context: no iterator methods).
    let mut e = 0;
    while e < 12 {
        let a = HEX_CORNERS[HEX20_GMSH_EDGES[e][0]];
        let b = HEX_CORNERS[HEX20_GMSH_EDGES[e][1]];
        n[8 + e] = [
            0.5 * (a[0] + b[0]),
            0.5 * (a[1] + b[1]),
            0.5 * (a[2] + b[2]),
        ];
        e += 1;
    }
    n
};

pub(crate) const HEX20: IncompleteFamily = IncompleteFamily {
    n: 20,
    dim: 3,
    nodes: &HEX20_NODES,
    monos: &HEX20_MONOS,
};

/// Build Prism15 node table (const, corners + Gmsh edge mids).
const PRISM15_NODES: [[f64; 3]; 15] = {
    let mut n = [[0.0f64; 3]; 15];
    let mut k = 0;
    while k < 6 {
        n[k] = PRISM15_CORNERS[k];
        k += 1;
    }
    let mut e = 0;
    while e < 9 {
        let a = PRISM15_CORNERS[PRISM15_GMSH_EDGES[e][0]];
        let b = PRISM15_CORNERS[PRISM15_GMSH_EDGES[e][1]];
        n[6 + e] = [
            0.5 * (a[0] + b[0]),
            0.5 * (a[1] + b[1]),
            0.5 * (a[2] + b[2]),
        ];
        e += 1;
    }
    n
};

pub(crate) const PRISM15: IncompleteFamily = IncompleteFamily {
    n: 15,
    dim: 3,
    nodes: &PRISM15_NODES,
    monos: &PRISM15_MONOS,
};

/// The incomplete family of `et`, or `None` for complete/other types.
pub(crate) fn family_of(et: ElementType) -> Option<IncompleteFamily> {
    match et {
        ElementType::Quad8 => Some(QUAD8),
        ElementType::Hex20 => Some(HEX20),
        ElementType::Prism15 => Some(PRISM15),
        _ => None,
    }
}

/// Vandermonde `A[r][c] = m_c(node_r)` inverse via Gauss-Jordan with partial
/// pivoting (row `i` of the inverse holds the coefficients of `φ_i`).
fn vandermonde_inverse(f: &IncompleteFamily) -> Option<Vec<f64>> {
    let n = f.n;
    let mut a = vec![0.0_f64; n * n];
    for (r, nr) in f.nodes.iter().take(n).enumerate() {
        for (c, mc) in f.monos.iter().take(n).enumerate() {
            let mut v = 1.0;
            for d in 0..3 {
                v *= nr[d].powi(mc[d] as i32);
            }
            a[r * n + c] = v;
        }
    }
    let mut inv = vec![0.0_f64; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for col in 0..n {
        let (piv, best) = (col..n).fold((col, 0.0_f64), |(bp, bv), r| {
            let v = a[r * n + col].abs();
            if v > bv {
                (r, v)
            } else {
                (bp, bv)
            }
        });
        if best < 1e-12 {
            return None;
        }
        if piv != col {
            for c in 0..n {
                a.swap(col * n + c, piv * n + c);
                inv.swap(col * n + c, piv * n + c);
            }
        }
        let p = a[col * n + col];
        let ip = 1.0 / p;
        for c in 0..n {
            a[col * n + c] *= ip;
            inv[col * n + c] *= ip;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let fac = a[r * n + col];
            if fac == 0.0 {
                continue;
            }
            for c in 0..n {
                a[r * n + c] -= fac * a[col * n + c];
                inv[r * n + c] -= fac * inv[col * n + c];
            }
        }
    }
    Some(inv)
}

fn coeffs(f: &IncompleteFamily, cell: &'static OnceLock<Vec<f64>>) -> &'static [f64] {
    cell.get_or_init(|| {
        let inv = vandermonde_inverse(f).expect("serendipity Vandermonde must be invertible");
        // φ_i = Σ_c C[i][c]·m_c with C = A^{-T}: store the *transposed*
        // inverse so `cs[i * n + c]` reads the coefficient of monomial c in
        // shape function i (same layout as fem_element's `build_coef2d`).
        let n = f.n;
        let mut cc = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                cc[i * n + j] = inv[j * n + i];
            }
        }
        cc
    })
}

static QUAD8_COEFFS: OnceLock<Vec<f64>> = OnceLock::new();
static HEX20_COEFFS: OnceLock<Vec<f64>> = OnceLock::new();
static PRISM15_COEFFS: OnceLock<Vec<f64>> = OnceLock::new();

fn coeffs_for(f: &IncompleteFamily) -> &'static [f64] {
    match f.n {
        8 => coeffs(f, &QUAD8_COEFFS),
        15 => coeffs(f, &PRISM15_COEFFS),
        20 => coeffs(f, &HEX20_COEFFS),
        _ => unreachable!("no coefficient cache for n = {}", f.n),
    }
}

/// Evaluate the family's nodal basis (and gradients) at the factory-domain
/// point `x = (x0, x1, x2)` (the third coordinate is ignored in 2-D).
///
/// `phi[i]` = value of the i-th (connectivity-order) shape function,
/// `grad[3 * i + d]` = `∂φ_i/∂x_d`.
pub(crate) fn eval_basis_with_grad(
    f: &IncompleteFamily,
    x: [f64; 3],
    phi: &mut [f64],
    grad: &mut [f64],
) {
    let n = f.n;
    let cs = coeffs_for(f);
    // Monomial values and derivatives at x.
    let mut mv = [1.0_f64; 20];
    let mut dv = [[0.0_f64; 3]; 20];
    for (c, mc) in f.monos.iter().enumerate() {
        let mut v = 1.0;
        for dd in 0..3 {
            v *= x[dd].powi(mc[dd] as i32);
        }
        // ∂m/∂x_dd = e_dd·x_dd^(e_dd−1) · ∏_{d'≠dd} x_d'^e_d' — the product
        // of the *other* coordinates' powers is part of the derivative.
        let mut d = [0.0_f64; 3];
        for dd in 0..3 {
            let e = mc[dd] as i32;
            if e == 0 {
                continue;
            }
            let mut others = 1.0;
            for d2 in 0..3 {
                if d2 != dd {
                    others *= x[d2].powi(mc[d2] as i32);
                }
            }
            d[dd] = (e as f64) * x[dd].powi(e - 1) * others;
        }
        mv[c] = v;
        dv[c] = d;
    }
    for i in 0..n {
        let mut v = 0.0;
        let mut g = [0.0_f64; 3];
        for c in 0..n {
            let w = cs[i * n + c];
            v += w * mv[c];
            for dd in 0..3 {
                g[dd] += w * dv[c][dd];
            }
        }
        phi[i] = v;
        for dd in 0..3 {
            grad[i * 3 + dd] = g[dd];
        }
    }
}

/// Isoparametric map of an incomplete-family element: given the element's
/// node coordinates (connectivity order, factory dimension `D`) and a
/// factory-domain reference point, return the Jacobian `J` (3-stride
/// row-major, i.e. `J[d * 3 + c] = ∂x_d/∂ξ_c`; only the first `D·D` entries
/// are meaningful), its determinant and the physical point `x(ξ)`.
pub(crate) fn eval_map<const D: usize>(
    f: &IncompleteFamily,
    node_coords: &[[f64; D]],
    xi: &[f64; D],
) -> ([f64; 9], f64, [f64; D]) {
    assert_eq!(D, f.dim, "incomplete family dimension mismatch");
    let mut x3 = [0.0_f64; 3];
    for d in 0..D {
        x3[d] = xi[d];
    }
    let mut phi = vec![0.0_f64; f.n];
    let mut grad = vec![0.0_f64; f.n * 3];
    eval_basis_with_grad(f, x3, &mut phi, &mut grad);
    let mut xp = [0.0_f64; D];
    let mut jac = [0.0_f64; 9];
    for (k, nc) in node_coords.iter().enumerate() {
        for d in 0..D {
            xp[d] += phi[k] * nc[d];
            for c in 0..D {
                jac[d * 3 + c] += nc[d] * grad[k * 3 + c];
            }
        }
    }
    let det = if D == 2 {
        jac[0] * jac[4] - jac[1] * jac[3]
    } else {
        jac[0] * (jac[4] * jac[8] - jac[5] * jac[7])
            - jac[1] * (jac[3] * jac[8] - jac[5] * jac[6])
            + jac[2] * (jac[3] * jac[7] - jac[4] * jac[6])
    };
    (jac, det, xp)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn nodal_delta_and_partition_of_unity() {
        for (name, f) in [("quad8", QUAD8), ("hex20", HEX20), ("prism15", PRISM15)] {
            let mut phi = vec![0.0_f64; f.n];
            let mut grad = vec![0.0_f64; f.n * 3];
            for (i, nr) in f.nodes.iter().enumerate() {
                eval_basis_with_grad(&f, *nr, &mut phi, &mut grad);
                for (k, v) in phi.iter().enumerate() {
                    let want = if k == i { 1.0 } else { 0.0 };
                    assert!(
                        approx(*v, want, 1e-10),
                        "{name}: phi[{k}]({node:?}) = {v} != {want}",
                        node = nr
                    );
                }
                let sum: f64 = phi.iter().sum();
                assert!(approx(sum, 1.0, 1e-12), "{name}: partition of unity");
            }
        }
    }

    #[test]
    fn straight_edges_reproduce_corner_map() {
        // A straight quad8 with edge nodes at geometric midpoints must map
        // exactly like the bilinear corner interpolant.
        let corners: Vec<[f64; 2]> = vec![[0.3, 0.1], [1.4, 0.25], [1.1, 1.35], [-0.2, 0.9]];
        let mut nodes = corners.clone();
        nodes.push([
            0.5 * (corners[0][0] + corners[1][0]),
            0.5 * (corners[0][1] + corners[1][1]),
        ]);
        nodes.push([
            0.5 * (corners[1][0] + corners[2][0]),
            0.5 * (corners[1][1] + corners[2][1]),
        ]);
        nodes.push([
            0.5 * (corners[2][0] + corners[3][0]),
            0.5 * (corners[2][1] + corners[3][1]),
        ]);
        nodes.push([
            0.5 * (corners[3][0] + corners[0][0]),
            0.5 * (corners[3][1] + corners[0][1]),
        ]);
        for i in 0..=10 {
            for j in 0..=10 {
                let xi = [i as f64 / 10.0, j as f64 / 10.0];
                let (_j2, _d, x) = eval_map::<2>(&QUAD8, &nodes, &xi);
                let (s, t) = (xi[0], xi[1]);
                let bl = [
                    (1.0 - s) * (1.0 - t) * corners[0][0]
                        + s * (1.0 - t) * corners[1][0]
                        + s * t * corners[2][0]
                        + (1.0 - s) * t * corners[3][0],
                    (1.0 - s) * (1.0 - t) * corners[0][1]
                        + s * (1.0 - t) * corners[1][1]
                        + s * t * corners[2][1]
                        + (1.0 - s) * t * corners[3][1],
                ];
                assert!(approx(x[0], bl[0], 1e-13) && approx(x[1], bl[1], 1e-13));
            }
        }
    }

    #[test]
    fn curved_quad8_interpolates_edge_nodes() {
        // Curved bottom edge: the mid node pulled up; the map must pass
        // through every node at its reference position (nodal basis).
        let mut nodes = QUAD8.nodes.iter().map(|p| [p[0], p[1]]).collect::<Vec<_>>();
        nodes[4][1] = 0.25; // bottom edge midpoint lifted
        // The map must pass through every data node at the node's REFERENCE
        // position (nodal/Kronecker basis).
        for (k, rk) in QUAD8.nodes.iter().enumerate() {
            let (_j, _d, x) = eval_map::<2>(&QUAD8, &nodes, &[rk[0], rk[1]]);
            assert!(
                approx(x[0], nodes[k][0], 1e-13) && approx(x[1], nodes[k][1], 1e-13),
                "node {k}: map {x:?} != data {:?}",
                nodes[k]
            );
        }
    }

    /// The Jacobian returned by [`eval_map`] must be the actual derivative of
    /// the map it returns (central finite differences), everywhere in the
    /// reference domain — on *curved* nodal data, not just straight elements.
    #[test]
    fn eval_map_jacobian_matches_finite_differences() {
        for (name, f) in [("quad8", QUAD8), ("hex20", HEX20), ("prism15", PRISM15)] {
            let dim = f.dim;
            // Arbitrary curved nodal data (distinct irrational-ish offsets so
            // that all second derivatives are active).
            let nodes: Vec<[f64; 3]> = f
                .nodes
                .iter()
                .enumerate()
                .map(|(k, r)| {
                    std::array::from_fn(|d| {
                        r[d] + 0.13 * ((k as f64 + 1.0) * 0.37 + d as f64 * 0.61).sin()
                    })
                })
                .collect();
            match dim {
                2 => {
                    let n2: Vec<[f64; 2]> =
                        nodes.iter().map(|p| [p[0], p[1]]).collect();
                    for i in 1..4 {
                        for j in 1..4 {
                            let xi = [0.2 * i as f64, 0.25 * j as f64];
                            let (jac, _det, x) = eval_map::<2>(&f, &n2, &xi);
                            let h = 1e-6_f64;
                            for kk in 0..2 {
                                let mut a = xi;
                                a[kk] += h;
                                let mut b = xi;
                                b[kk] -= h;
                                let (_jp, _dp, x1) = eval_map::<2>(&f, &n2, &a);
                                let (_jm, _dm, x0) = eval_map::<2>(&f, &n2, &b);
                                for d in 0..2 {
                                    let fd = (x1[d] - x0[d]) / (2.0 * h);
                                    assert!(
                                        (jac[d * 3 + kk] - fd).abs() < 1e-8,
                                        "{name}: dx[{d}]/dxi[{kk}] at {xi:?}: analytic {} vs fd {fd}",
                                        jac[d * 3 + kk]
                                    );
                                }
                            }
                            let _ = x;
                        }
                    }
                }
                3 => {
                    let n3: Vec<[f64; 3]> = nodes.clone();
                    for i in 1..4 {
                        for j in 1..4 {
                            for k in 1..4 {
                                let xi = [
                                    0.25 * i as f64,
                                    0.2 * j as f64,
                                    0.3 * k as f64,
                                ];
                                let (jac, _det, _x) = eval_map::<3>(&f, &n3, &xi);
                                let h = 1e-6_f64;
                                for kk in 0..3 {
                                    let mut a = xi;
                                    a[kk] += h;
                                    let mut b = xi;
                                    b[kk] -= h;
                                    let (_jp, _dp, x1) = eval_map::<3>(&f, &n3, &a);
                                    let (_jm, _dm, x0) = eval_map::<3>(&f, &n3, &b);
                                    for d in 0..3 {
                                        let fd = (x1[d] - x0[d]) / (2.0 * h);
                                        assert!(
                                            (jac[d * 3 + kk] - fd).abs() < 1e-8,
                                            "{name}: dx[{d}]/dxi[{kk}] at {xi:?}: analytic {} vs fd {fd}",
                                            jac[d * 3 + kk]
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                other => unreachable!("dim {other}"),
            }
        }
    }

    #[test]
    fn straight_hex20_is_trilinear() {
        let corners = HEX_CORNERS.to_vec();
        let mut nodes = corners.clone();
        for [a, b] in HEX20_GMSH_EDGES {
            nodes.push(std::array::from_fn(|d| 0.5 * (corners[a][d] + corners[b][d])));
        }
        for i in 0..=4 {
            for j in 0..=4 {
                for k in 0..=4 {
                    let xi = [
                        -1.0 + 0.5 * i as f64,
                        -1.0 + 0.5 * j as f64,
                        -1.0 + 0.5 * k as f64,
                    ];
                    let (_jac, _det, x) = eval_map::<3>(&HEX20, &nodes, &xi);
                    let trilin: Vec<f64> = (0..3)
                        .map(|d| {
                            let c: Vec<f64> = corners.iter().map(|p| p[d]).collect();
                            // trilinear interpolation of corner values
                            let (s, t, u) = (
                                (xi[0] + 1.0) * 0.5,
                                (xi[1] + 1.0) * 0.5,
                                (xi[2] + 1.0) * 0.5,
                            );
                            (1.0 - s)
                                * (1.0 - t)
                                * (1.0 - u)
                                * c[0]
                                + s * (1.0 - t) * (1.0 - u) * c[1]
                                + s * t * (1.0 - u) * c[2]
                                + (1.0 - s) * t * (1.0 - u) * c[3]
                                + (1.0 - s) * (1.0 - t) * u * c[4]
                                + s * (1.0 - t) * u * c[5]
                                + s * t * u * c[6]
                                + (1.0 - s) * t * u * c[7]
                        })
                        .collect();
                    for d in 0..3 {
                        assert!(approx(x[d], trilin[d], 1e-12));
                    }
                }
            }
        }
    }
}

