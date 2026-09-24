//! Lagrange elements on the reference hexahedron `[0,1]³`.
//!
//! D721: the hex family moved from the historical fem-rs `[-1,1]³` frame to
//! MFEM's natural `[0,1]³` (the quad's D364 migration, done for hexes).  The
//! slot orders, DOF counts and DOF positions (now MFEM's `[0,1]` values) are
//! unchanged; only the reference frame of the basis and its chain-rule
//! factors moved.

use crate::quadrature::hex_rule;
use crate::reference::{QuadratureRule, ReferenceElement};

// ─── Q1 ───────────────────────────────────────────────────────────────────────

/// Trilinear Lagrange element on the reference hex `[0,1]³` — 8 DOFs.
///
/// 1:1 with MFEM `TriLinear3DFiniteElement` (`fem/fe/fe_fixed_order.cpp:1597`)
/// — the element `Mesh::GetTransformationFEforElementType(Element::HEXAHEDRON)`
/// hands a *straight* hex mesh (`mesh/hexahedron.cpp:60`), i.e. **the hex
/// geometry element** (and the H¹ order-1 hex element).  Note this is the
/// naive `ox·oy·oz` form, not the barycentric `H1_HexahedronElement(1)` — the
/// two coincide on the quadrature points this library uses, and the naive
/// form is the one that makes the affine-hex Jacobian bit-identical to
/// MFEM's.
///
/// Node ordering: bottom face (z=0) then top face (z=1), each as a
/// counter-clockwise quad starting from (0,0).
///
/// | Index | (ξ, η, ζ) |
/// |-------|----------|
/// | 0     | (0, 0, 0) |
/// | 1     | (1, 0, 0) |
/// | 2     | (1, 1, 0) |
/// | 3     | (0, 1, 0) |
/// | 4     | (0, 0, 1) |
/// | 5     | (1, 0, 1) |
/// | 6     | (1, 1, 1) |
/// | 7     | (0, 1, 1) |
///
/// Basis: φᵢ = (ξ or 1−ξ)(η or 1−η)(ζ or 1−ζ) — MFEM's `ox·oy·oz` products.
pub struct HexQ1;

const Q1_NODES: [(f64, f64, f64); 8] = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
];

impl ReferenceElement for HexQ1 {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        1
    }
    fn n_dofs(&self) -> usize {
        8
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        // MFEM TriLinear3DFiniteElement::CalcShape, verbatim.
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let ox = 1.0 - x;
        let oy = 1.0 - y;
        let oz = 1.0 - z;
        values[0] = ox * oy * oz;
        values[1] = x * oy * oz;
        values[2] = x * y * oz;
        values[3] = ox * y * oz;
        values[4] = ox * oy * z;
        values[5] = x * oy * z;
        values[6] = x * y * z;
        values[7] = ox * y * z;
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        // MFEM TriLinear3DFiniteElement::CalcDShape, verbatim.
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let ox = 1.0 - x;
        let oy = 1.0 - y;
        let oz = 1.0 - z;
        grads[0] = -oy * oz;
        grads[1] = -ox * oz;
        grads[2] = -ox * oy;

        grads[3] = oy * oz;
        grads[4] = -x * oz;
        grads[5] = -x * oy;

        grads[6] = y * oz;
        grads[7] = x * oz;
        grads[8] = -x * y;

        grads[9] = -y * oz;
        grads[10] = ox * oz;
        grads[11] = -ox * y;

        grads[12] = -oy * z;
        grads[13] = -ox * z;
        grads[14] = ox * oy;

        grads[15] = oy * z;
        grads[16] = -x * z;
        grads[17] = x * oy;

        grads[18] = y * z;
        grads[19] = x * z;
        grads[20] = x * y;

        grads[21] = -y * z;
        grads[22] = ox * z;
        grads[23] = ox * y;
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        hex_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        Q1_NODES.iter().map(|&(x, y, z)| vec![x, y, z]).collect()
    }
}

// ─── Q2 ───────────────────────────────────────────────────────────────────────

/// Quadratic Lagrange element on the reference hex `[0,1]³` — 27 DOFs.
///
/// Slot order = [`crate::lagrange::factory::HexQk`]`::new(2)` = MFEM
/// `H1_HexahedronElement(2)` (D31 closed the pre-D31 fem-rs legacy order;
/// probe `tmp/d31/probe_h1_hex.cpp`, dump `tmp/d31/fe_nodes_cpp.txt`):
///
/// - 0..7:   8 vertices (bottom ring, then top ring)
/// - 8..11:  z-low ring edge mids (`CUBE::Edges[0..3]`: 0→1, 1→2, 3→2, 0→3)
/// - 12..15: z-high ring edge mids (`CUBE::Edges[4..7]`: 4→5, 5→6, 7→6, 4→7)
/// - 16..19: vertical edge mids (`CUBE::Edges[8..11]`: 0→4, 1→5, 2→6, 3→7)
/// - 20..25: face centres in `CUBE::FaceVert` order
///           (z-low, y-low, x-high, y-high, x-low, z-high)
/// - 26:     volume centre (1/2, 1/2, 1/2)
///
/// Basis: φᵢ = L_ix(ξᵢ)(ξ) · L_iy(ηᵢ)(η) · L_iz(ζᵢ)(ζ)
/// where L(0), L(1/2), L(1) are the quadratic 1-D Lagrange polynomials on the
/// `[0,1]` frame.
pub struct HexQ2;

const Q2_NODES_HEX: [(f64, f64, f64); 27] = {
    let mut n = [(0.0, 0.0, 0.0); 27];
    // vertices
    n[0] = (0.0, 0.0, 0.0);
    n[1] = (1.0, 0.0, 0.0);
    n[2] = (1.0, 1.0, 0.0);
    n[3] = (0.0, 1.0, 0.0);
    n[4] = (0.0, 0.0, 1.0);
    n[5] = (1.0, 0.0, 1.0);
    n[6] = (1.0, 1.0, 1.0);
    n[7] = (0.0, 1.0, 1.0);
    // edges: z-low ring (0→1, 1→2, 3→2, 0→3)
    n[8] = (0.5, 0.0, 0.0);
    n[9] = (1.0, 0.5, 0.0);
    n[10] = (0.5, 1.0, 0.0);
    n[11] = (0.0, 0.5, 0.0);
    // edges: z-high ring (4→5, 5→6, 7→6, 4→7)
    n[12] = (0.5, 0.0, 1.0);
    n[13] = (1.0, 0.5, 1.0);
    n[14] = (0.5, 1.0, 1.0);
    n[15] = (0.0, 0.5, 1.0);
    // edges: vertical (0→4, 1→5, 2→6, 3→7)
    n[16] = (0.0, 0.0, 0.5);
    n[17] = (1.0, 0.0, 0.5);
    n[18] = (1.0, 1.0, 0.5);
    n[19] = (0.0, 1.0, 0.5);
    // face centres: FaceVert order (z-low, y-low, x-high, y-high, x-low, z-high)
    n[20] = (0.5, 0.5, 0.0);
    n[21] = (0.5, 0.0, 0.5);
    n[22] = (1.0, 0.5, 0.5);
    n[23] = (0.5, 1.0, 0.5);
    n[24] = (0.0, 0.5, 0.5);
    n[25] = (0.5, 0.5, 1.0);
    // volume centre
    n[26] = (0.5, 0.5, 0.5);
    n
};

/// 1-D quadratic Lagrange basis on the `[0,1]` frame at the nodes
/// `{0, 1/2, 1}` — the historical `[-1,1]` form `[½x(x−1), 1−x², ½x(x+1)]`
/// re-expressed directly on `[0,1]` (D721; derivative factor 2 folded in).
fn hex_q2_1d(u: f64) -> ([f64; 3], [f64; 3]) {
    let vals = [
        (2.0 * u - 1.0) * (u - 1.0), // ℓ at 0
        4.0 * u * (1.0 - u),         // ℓ at 1/2
        u * (2.0 * u - 1.0),         // ℓ at 1
    ];
    let ders = [
        4.0 * u - 3.0,
        4.0 - 8.0 * u,
        4.0 * u - 1.0,
    ];
    (vals, ders)
}

fn coord_to_q2_idx(c: f64) -> usize {
    if c < 0.25 {
        0
    } else if c > 0.75 {
        2
    } else {
        1
    }
}

impl ReferenceElement for HexQ2 {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        2
    }
    fn n_dofs(&self) -> usize {
        27
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let (lx, _) = hex_q2_1d(x);
        let (ly, _) = hex_q2_1d(y);
        let (lz, _) = hex_q2_1d(z);
        for (i, &(xi_i, eta_i, zeta_i)) in Q2_NODES_HEX.iter().enumerate() {
            let ix = coord_to_q2_idx(xi_i);
            let iy = coord_to_q2_idx(eta_i);
            let iz = coord_to_q2_idx(zeta_i);
            values[i] = lx[ix] * ly[iy] * lz[iz];
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let (lx, dlx) = hex_q2_1d(x);
        let (ly, dly) = hex_q2_1d(y);
        let (lz, dlz) = hex_q2_1d(z);
        for (i, &(xi_i, eta_i, zeta_i)) in Q2_NODES_HEX.iter().enumerate() {
            let ix = coord_to_q2_idx(xi_i);
            let iy = coord_to_q2_idx(eta_i);
            let iz = coord_to_q2_idx(zeta_i);
            grads[i * 3] = dlx[ix] * ly[iy] * lz[iz];
            grads[i * 3 + 1] = lx[ix] * dly[iy] * lz[iz];
            grads[i * 3 + 2] = lx[ix] * ly[iy] * dlz[iz];
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        hex_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        Q2_NODES_HEX
            .iter()
            .map(|&(x, y, z)| vec![x, y, z])
            .collect()
    }
}

// ─── Tensor-product layout (element-layer export) ─────────────────────────────

/// `(ascending 1-D nodes, slot → tensor index)` of a tensor-product hex element.
///
/// This is the single derivation of "which tensor node `(ix, iy, iz)` does
/// element-local slot `s` carry": both the CPU partial-assembly kernels
/// (`fem-assembly`'s `pa::hex_layout`) and the GPU shader generator
/// (`fem-linalg-gpu`'s `generate_hex_qk_wgsl`) build their slot tables from it,
/// so a `HexQk`/`HexQ2`/`HexQ3` layout change cannot silently desynchronize
/// them.
///
/// [`ReferenceElement::dof_coords`] hands out the element's own 1-D nodes
/// bit-for-bit, so the distinct values of any axis coordinate *are* those nodes
/// — the matching below is exact (`dedup`/`position` compare without tolerance).
pub fn hex_tensor_layout(elem: &dyn ReferenceElement) -> (Vec<f64>, Vec<[usize; 3]>) {
    let coords: Vec<[f64; 3]> = elem
        .dof_coords()
        .into_iter()
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let mut nodes: Vec<f64> = coords.iter().map(|c| c[0]).collect();
    nodes.sort_by(|a, b| a.partial_cmp(b).expect("hex node coordinate is finite"));
    nodes.dedup();
    let slots = coords
        .iter()
        .map(|c| {
            let axis = |v: f64| {
                nodes.iter().position(|&n| n == v).unwrap_or_else(|| {
                    panic!("hex slot coordinate {v} is not one of the 1-D nodes {nodes:?}")
                })
            };
            [axis(c[0]), axis(c[1]), axis(c[2])]
        })
        .collect();
    (nodes, slots)
}

// ─── Q3 ───────────────────────────────────────────────────────────────────────

/// Cubic Lagrange element on the reference hex `[0,1]³` — 64 DOFs.
///
/// Slot order = MFEM `H1_HexahedronElement(3)` = [`crate::lagrange::factory::HexQk`]`::new(3)`
/// (converged, D31 stage A): 8 vertices in `CUBE::Vertices` order, then two
/// slots per edge in `CUBE::Edges` order (each block ascending along the
/// edge), then four slots per face in `CUBE::FaceVert` order (`j` outer,
/// `i` inner), then the 2×2×2 interior (`k` outer, `i` fastest).
///
/// The 1-D nodes are the Gauss-Lobatto points `{0, 1, (1±1/√5)/2}` — the
/// `[0,1]` image of `HexQk::new(3)`'s `Lagrange1D` nodes
/// (`quadrature::gauss_lobatto_01(4)`), matching MFEM `H1_FECollection`'s
/// `BasisType::GaussLobatto`.
pub struct HexQ3;

/// 1-D GLL nodes for `p = 3` on `[0,1]`.  Must stay bit-identical to the
/// `[0,1]` image of `quadrature::gauss_lobatto_arbitrary(4)` — `HexQk::new(3)`
/// builds its nodes the same way, and the slot-layout pin test compares
/// coordinates exactly.
fn q3_nodes_1d() -> [f64; 4] {
    let s = (1.0_f64 / 5.0).sqrt();
    [0.0, 0.5 * (1.0 - s), 0.5 * (1.0 + s), 1.0]
}

/// Slot → tensor-node index `(ix, iy, iz)` (node coordinate = `q3_nodes_1d()[i]`).
///
/// This is the MFEM `H1_HexahedronElement(3)` dof map, i.e. exactly
/// `factory::tests::mfem_h1_hex_slot_nodes(3)`; pinned slot-by-slot against
/// `HexQk::new(3)` by [`tests::hex_q3_layout_matches_hex_qk3`].
const Q3_SLOT_TENSOR: [[u8; 3]; 64] = [
    // vertices: bottom ring 0..3, top ring 4..7
    [0, 0, 0], [3, 0, 0], [3, 3, 0], [0, 3, 0],
    [0, 0, 3], [3, 0, 3], [3, 3, 3], [0, 3, 3],
    // edges, `CUBE::Edges` order, two slots per edge, ascending tensor index
    [1, 0, 0], [2, 0, 0], // 0-1
    [3, 1, 0], [3, 2, 0], // 1-2
    [1, 3, 0], [2, 3, 0], // 3-2
    [0, 1, 0], [0, 2, 0], // 0-3
    [1, 0, 3], [2, 0, 3], // 4-5
    [3, 1, 3], [3, 2, 3], // 5-6
    [1, 3, 3], [2, 3, 3], // 7-6
    [0, 1, 3], [0, 2, 3], // 4-7
    [0, 0, 1], [0, 0, 2], // 0-4
    [3, 0, 1], [3, 0, 2], // 1-5
    [3, 3, 1], [3, 3, 2], // 2-6
    [0, 3, 1], [0, 3, 2], // 3-7
    // faces, `CUBE::FaceVert` order, `j` outer / `i` inner
    [1, 2, 0], [2, 2, 0], [1, 1, 0], [2, 1, 0], // z low  (i, 3-j, 0)
    [1, 0, 1], [2, 0, 1], [1, 0, 2], [2, 0, 2], // y low  (i, 0, j)
    [3, 1, 1], [3, 2, 1], [3, 1, 2], [3, 2, 2], // x high (3, i, j)
    [2, 3, 1], [1, 3, 1], [2, 3, 2], [1, 3, 2], // y high (3-i, 3, j)
    [0, 2, 1], [0, 1, 1], [0, 2, 2], [0, 1, 2], // x low  (0, 3-i, j)
    [1, 1, 3], [2, 1, 3], [1, 2, 3], [2, 2, 3], // z high (i, j, 3)
    // interior: k outer, j middle, i fastest
    [1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1],
    [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2],
];

fn hex_q3_lagrange_1d(x: f64, nodes: &[f64; 4]) -> ([f64; 4], [f64; 4]) {
    let mut vals = [1.0_f64; 4];
    for i in 0..4 {
        for j in 0..4 {
            if j != i {
                vals[i] *= (x - nodes[j]) / (nodes[i] - nodes[j]);
            }
        }
    }

    let mut ders = [0.0_f64; 4];
    for i in 0..4 {
        let mut sum = 0.0;
        for m in 0..4 {
            if m == i {
                continue;
            }
            let mut term = 1.0 / (nodes[i] - nodes[m]);
            for j in 0..4 {
                if j != i && j != m {
                    term *= (x - nodes[j]) / (nodes[i] - nodes[j]);
                }
            }
            sum += term;
        }
        ders[i] = sum;
    }
    (vals, ders)
}

impl ReferenceElement for HexQ3 {
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
        let n1d = q3_nodes_1d();
        let (lx, _) = hex_q3_lagrange_1d(xi[0], &n1d);
        let (ly, _) = hex_q3_lagrange_1d(xi[1], &n1d);
        let (lz, _) = hex_q3_lagrange_1d(xi[2], &n1d);
        for (slot, t) in Q3_SLOT_TENSOR.iter().enumerate() {
            let (ix, iy, iz) = (t[0] as usize, t[1] as usize, t[2] as usize);
            values[slot] = lx[ix] * ly[iy] * lz[iz];
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let n1d = q3_nodes_1d();
        let (lx, dlx) = hex_q3_lagrange_1d(xi[0], &n1d);
        let (ly, dly) = hex_q3_lagrange_1d(xi[1], &n1d);
        let (lz, dlz) = hex_q3_lagrange_1d(xi[2], &n1d);
        for (slot, t) in Q3_SLOT_TENSOR.iter().enumerate() {
            let (ix, iy, iz) = (t[0] as usize, t[1] as usize, t[2] as usize);
            grads[slot * 3] = dlx[ix] * ly[iy] * lz[iz];
            grads[slot * 3 + 1] = lx[ix] * dly[iy] * lz[iz];
            grads[slot * 3 + 2] = lx[ix] * ly[iy] * dlz[iz];
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        hex_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let n1d = q3_nodes_1d();
        Q3_SLOT_TENSOR
            .iter()
            .map(|t| vec![n1d[t[0] as usize], n1d[t[1] as usize], n1d[t[2] as usize]])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_pou(elem: &dyn ReferenceElement) {
        let rule = elem.quadrature(4);
        let mut phi = vec![0.0_f64; elem.n_dofs()];
        for pt in &rule.points {
            elem.eval_basis(pt, &mut phi);
            let s: f64 = phi.iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "POU failed sum={s}");
        }
    }

    fn check_grad_zero(elem: &dyn ReferenceElement) {
        let dim = elem.dim() as usize;
        let rule = elem.quadrature(4);
        let mut g = vec![0.0_f64; elem.n_dofs() * dim];
        for pt in &rule.points {
            elem.eval_grad_basis(pt, &mut g);
            for d in 0..dim {
                let s: f64 = (0..elem.n_dofs()).map(|i| g[i * dim + d]).sum();
                assert!(s.abs() < 1e-11, "grad sum d={d} = {s}");
            }
        }
    }

    #[test]
    fn hex_q1_pou() {
        check_pou(&HexQ1);
    }
    #[test]
    fn hex_q1_grad_zero() {
        check_grad_zero(&HexQ1);
    }

    #[test]
    fn hex_q1_node_dofs() {
        let mut phi = vec![0.0; 8];
        for (i, &(x, y, z)) in Q1_NODES.iter().enumerate() {
            HexQ1.eval_basis(&[x, y, z], &mut phi);
            for j in 0..8 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (phi[j] - expected).abs() < 1e-13,
                    "node {i}, basis {j}: expected {expected}, got {}",
                    phi[j]
                );
            }
        }
    }

    // ── HexQ2 ─────────────────────────────────────────────────────────────

    #[test]
    fn hex_q2_pou() {
        check_pou(&HexQ2);
    }
    #[test]
    fn hex_q2_grad_zero() {
        check_grad_zero(&HexQ2);
    }
    #[test]
    fn hex_q2_n_dofs() {
        assert_eq!(HexQ2.n_dofs(), 27);
    }

    #[test]
    fn hex_q2_node_dofs() {
        let mut phi = vec![0.0; 27];
        for (i, &(x, y, z)) in Q2_NODES_HEX.iter().enumerate() {
            HexQ2.eval_basis(&[x, y, z], &mut phi);
            for j in 0..27 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (phi[j] - expected).abs() < 1e-13,
                    "node {i}, basis {j}: expected {expected}, got {}",
                    phi[j]
                );
            }
        }
    }

    #[test]
    fn hex_q2_n_dofs_matches_hex_qk() {
        use crate::lagrange::factory::HexQk;
        assert_eq!(HexQ2.n_dofs(), HexQk::new(2).n_dofs());
    }

    // ── HexQ3 ─────────────────────────────────────────────────────────────

    #[test]
    fn hex_q3_pou() {
        check_pou(&HexQ3);
    }
    #[test]
    fn hex_q3_grad_zero() {
        check_grad_zero(&HexQ3);
    }
    #[test]
    fn hex_q3_n_dofs() {
        assert_eq!(HexQ3.n_dofs(), 64);
    }

    #[test]
    fn hex_q3_node_dofs() {
        let n1d = q3_nodes_1d();
        let mut phi = vec![0.0; 64];
        for (slot, t) in Q3_SLOT_TENSOR.iter().enumerate() {
            let pt = [n1d[t[0] as usize], n1d[t[1] as usize], n1d[t[2] as usize]];
            HexQ3.eval_basis(&pt, &mut phi);
            for j in 0..64 {
                let expected = if slot == j { 1.0 } else { 0.0 };
                assert!(
                    (phi[j] - expected).abs() < 1e-13,
                    "node {slot}, basis {j}: expected {expected}, got {}",
                    phi[j]
                );
            }
        }
    }

    /// D31 convergence pin: `HexQ2`'s layout (slot order + node
    /// coordinates + basis values) is **slot-by-slot identical** to the
    /// order-generic `HexQk::new(2)` the space numbering (`build_q2_hex`)
    /// follows — both are now the MFEM `H1_HexahedronElement(2)` order
    /// (`tmp/d31/fe_nodes_cpp.txt`) and must stay in lockstep.
    #[test]
    fn hex_q2_layout_matches_hex_qk2() {
        assert_layout_converged(&HexQ2, &crate::lagrange::factory::HexQk::new(2));
    }

    /// D31 stage-A convergence pin: `HexQ3`'s layout (slot order + GLL node
    /// coordinates + basis values) is **slot-by-slot identical** to the
    /// order-generic `HexQk::new(3)` the space numbering (`build_pk_hex`)
    /// follows — the two must stay in lockstep.
    #[test]
    fn hex_q3_layout_matches_hex_qk3() {
        assert_layout_converged(&HexQ3, &crate::lagrange::factory::HexQk::new(3));
    }

    /// Slot-by-slot equality of two hex elements: `dof_coords` bit-exact,
    /// values/gradients at generic and nodal points to 1e-12 (the elements
    /// use different 1-D evaluation formulas, so values may differ in the
    /// last ulps only).
    fn assert_layout_converged(a: &dyn ReferenceElement, b: &dyn ReferenceElement) {
        assert_eq!(a.n_dofs(), b.n_dofs());
        let n = a.n_dofs();
        let (ca, cb) = (a.dof_coords(), b.dof_coords());
        for (s, (pa, pb)) in ca.iter().zip(cb.iter()).enumerate() {
            for d in 0..3 {
                assert_eq!(
                    pa[d], pb[d],
                    "slot {s} coord {d}: {pa:?} vs {pb:?} (must be bit-identical)"
                );
            }
        }
        let mut va = vec![0.0; n];
        let mut vb = vec![0.0; n];
        let mut ga = vec![0.0; n * 3];
        let mut gb = vec![0.0; n * 3];
        let mut pts: Vec<[f64; 3]> = vec![
            [0.5, 0.5, 0.5],
            [0.65, 0.25, 0.85],
            [0.05, 0.72, 0.56],
            [0.0, 1.0, 0.0],
        ];
        for c in &ca {
            pts.push([c[0], c[1], c[2]]);
        }
        for pt in &pts {
            a.eval_basis(pt, &mut va);
            b.eval_basis(pt, &mut vb);
            for (i, (x, y)) in va.iter().zip(vb.iter()).enumerate() {
                assert!((x - y).abs() < 1e-12, "pt {pt:?} slot {i}: {x} vs {y}");
            }
            a.eval_grad_basis(pt, &mut ga);
            b.eval_grad_basis(pt, &mut gb);
            for (i, (x, y)) in ga.iter().zip(gb.iter()).enumerate() {
                assert!((x - y).abs() < 1e-12, "pt {pt:?} grad {i}: {x} vs {y}");
            }
        }
    }

    #[test]
    fn hex_q3_n_dofs_matches_hex_qk() {
        use crate::lagrange::factory::HexQk;
        assert_eq!(HexQ3.n_dofs(), HexQk::new(3).n_dofs());
    }

    #[test]
    fn hex_q3_gradient_fd() {
        let h = 1e-7;
        let n = 64;
        let elem = HexQ3;
        let (mut vc, mut vx, mut vy, mut vz, mut grads) = (
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n * 3],
        );
        for &(x, y, z) in &[(0.65, 0.25, 0.85), (0.45, 0.6, 0.35)] {
            elem.eval_basis(&[x, y, z], &mut vc);
            elem.eval_basis(&[x + h, y, z], &mut vx);
            elem.eval_basis(&[x, y + h, z], &mut vy);
            elem.eval_basis(&[x, y, z + h], &mut vz);
            elem.eval_grad_basis(&[x, y, z], &mut grads);
            for i in 0..n {
                let fd_x = (vx[i] - vc[i]) / h;
                let fd_y = (vy[i] - vc[i]) / h;
                let fd_z = (vz[i] - vc[i]) / h;
                assert!((grads[i * 3] - fd_x).abs() < 1e-5, "({x},{y},{z}) i={i} gx");
                assert!(
                    (grads[i * 3 + 1] - fd_y).abs() < 1e-5,
                    "({x},{y},{z}) i={i} gy"
                );
                assert!(
                    (grads[i * 3 + 2] - fd_z).abs() < 1e-5,
                    "({x},{y},{z}) i={i} gz"
                );
            }
        }
    }

    /// D721 pin on the flipped reference domain: the `[0,1]³` hex geometry
    /// basis is now MFEM's `TriLinear3DFiniteElement` (`ox·oy·oz`) rather
    /// than the historical `(1±ξ)(1±η)(1±ζ)/8` on `[-1,1]³`, and its
    /// 8-node map on the unit-cube vertices is the identity up to the
    /// affine-hex FP debris MFEM itself carries (the `J = I + 2⁻⁶⁰·k` junk
    /// that the D680 gate pins against MFEM's own dump).
    #[test]
    fn hex_q1_is_mfem_trilinear_on_unit_cube() {
        // Vertex (0,0,0) basis at the cube centre: 1/8 for every vertex.
        let mut phi = vec![0.0; 8];
        HexQ1.eval_basis(&[0.5, 0.5, 0.5], &mut phi);
        for v in &phi {
            assert_eq!(*v, 0.125);
        }
        // The 8-node geometry map on the unit-cube vertices is the identity to
        // within the affine-hex cancellation debris (MFEM's dump: |junk| <=
        // 2.8e-17, detJ ≡ 1 exactly).  Sampled at the actual 27-point CUBE
        // rule points (MFEM's qp order: x fastest).
        let verts: [[f64; 3]; 8] = Q1_NODES.map(|(a, b, c)| [a, b, c]);
        let mut g = vec![0.0; 24];
        let (xs, _w) = crate::quadrature::gauss_legendre_01(3);
        for &zk in &xs {
            for &yj in &xs {
                for &xi in &xs {
                    HexQ1.eval_grad_basis(&[xi, yj, zk], &mut g);
                    for i in 0..3 {
                        for d in 0..3 {
                            let j: f64 = (0..8).map(|k| verts[k][i] * g[k * 3 + d]).sum();
                            let want = if i == d { 1.0 } else { 0.0 };
                            assert!(
                                (j - want).abs() < 1e-16,
                                "J({i},{d}) = {j} at ({xi},{yj},{zk}) must be {want} \
                                 up to affine-hex debris"
                            );
                        }
                    }
                }
            }
        }
    }
}
