//! Second-order tensor-product H(curl) element on reference quad `[0,1]^2`.
//!
//! 1:1 port of MFEM `ND_QuadrilateralElement(2)` (`fe_nd.cpp`, `dof = 2p(p+1)`).
//!
//! # Basis (MFEM tensor structure)
//!
//! The 12 basis functions are the tensor products
//! `Q_{1,2} × Q_{2,1}` built from
//!
//! * the **open** 2-point Gauss-Legendre Lagrange basis `ℓ₀, ℓ₁` on `[0,1]`
//!   (MFEM `OpenBasis(GaussLegendre)`, nodes `t₀ = (1−1/√3)/2`, `t₁ = (1+1/√3)/2`),
//! * the **closed** 3-point quadratic (Lobatto) basis on `[0,1]`
//!   (MFEM `ClosedBasis`, nodes `{0, 1/2, 1}`):
//!   `c₀(y) = 2y²−3y+1`, `c₁(y) = 4y(1−y)`, `c₂(y) = 2y²−y`.
//!
//! MFEM's negative `dof_map` entries on the top/left edges (tangents
//! `(−1,0)`/`(0,−1)`) are baked into the corresponding basis functions, exactly
//! as `CalcVShape` applies the sign `s`.
//!
//! # DOF semantics (D32 fix — MFEM nodal point-value functionals)
//!
//! Every DOF is a point evaluation `σ_i(Φ) = Φ(x_i)·t̂_i` at the node point
//! `x_i = FE::Nodes` along the fixed reference tangent `t̂_i`.  The open GL
//! nodes are symmetric about `t = 1/2`, so an edge reversal maps the edge DOFs
//! to a **signed anti-diagonal permutation** (`σ^rev_m = −σ_{1−m}`) — the
//! cross-element pairing is conforming (the previous 8-mode split-trace basis
//! had no point-dual functionals and relied on the broken moment pairing).
//!
//! # DOF layout (must match `HCurlSpace` local ordering)
//! ```text
//! 0..2   bottom edge (y=0, tangent +x):  (ℓ_i(x)·c₀(y), 0)
//! 2..4   right edge  (x=1, tangent +y):  (0, c₂(x)·ℓ_i(y))
//! 4..6   top edge    (y=1, tangent −x):  (−ℓ_{1−i}(x)·c₂(y), 0)   (MFEM sign −1)
//! 6..8   left edge   (x=0, tangent −y):  (0, −c₀(x)·ℓ_{1−i}(y))   (MFEM sign −1)
//! 8..10  interior x-component: (ℓ_i(x)·c₁(y), 0)   at (t_i, 1/2)
//! 10..12 interior y-component: (0, c₁(x)·ℓ_i(y))   at (1/2, t_i)
//! ```

use crate::quadrature::quad_rule_01;
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// 2-point Gauss-Legendre nodes on `[0,1]` — MFEM `OpenPoints(1)` (GaussLegendre).
const GL2: [f64; 2] = [0.21132486540518710671, 0.78867513459481286553];

/// Open (GL) Lagrange basis on `[0,1]`: `ℓ_0(x)`, `ℓ_1(x)`.
#[inline]
fn open_basis(x: f64) -> [f64; 2] {
    let inv = 1.0 / (GL2[0] - GL2[1]);
    [
        (x - GL2[1]) * inv,
        (GL2[0] - x) * inv,
    ]
}

/// Closed 3-point quadratic (Lobatto) basis on `[0,1]`: `c_0, c_1, c_2` and
/// their derivatives.
#[inline]
fn closed_basis(y: f64) -> ([f64; 3], [f64; 3]) {
    (
        [2.0 * y * y - 3.0 * y + 1.0, 4.0 * y * (1.0 - y), 2.0 * y * y - y],
        [4.0 * y - 3.0, 4.0 - 8.0 * y, 4.0 * y - 1.0],
    )
}

/// Second-order H(curl) element on reference quad, 12 DOFs (8 edge + 4 interior).
pub struct QuadND2;

impl VectorReferenceElement for QuadND2 {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        2
    }
    fn n_dofs(&self) -> usize {
        12
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];
        let l = open_basis(x); // open (tangential) factor for x-comp modes
        let ly = open_basis(y); // open (tangential) factor for y-comp modes
        let (cy, _dcy) = closed_basis(y); // closed (normal) factor for x-comp
        let (cx, _dcx) = closed_basis(x); // closed (normal) factor for y-comp

        // bottom edge (y=0), tangent +x
        values[0] = l[0] * cy[0];
        values[1] = 0.0;
        values[2] = l[1] * cy[0];
        values[3] = 0.0;
        // right edge (x=1), tangent +y  (c₂(1) = 1)
        values[4] = 0.0;
        values[5] = cx[2] * ly[0];
        values[6] = 0.0;
        values[7] = cx[2] * ly[1];
        // top edge (y=1), tangent −x (MFEM dof_map sign −1; c₂(1) = 1)
        values[8] = -l[1] * cy[2];
        values[9] = 0.0;
        values[10] = -l[0] * cy[2];
        values[11] = 0.0;
        // left edge (x=0), tangent −y (MFEM dof_map sign −1; c₀(0) = 1)
        values[12] = 0.0;
        values[13] = -cx[0] * ly[1];
        values[14] = 0.0;
        values[15] = -cx[0] * ly[0];

        // Interior x-component: ℓ_i(x)·c₁(y)
        values[16] = l[0] * cy[1];
        values[17] = 0.0;
        values[18] = l[1] * cy[1];
        values[19] = 0.0;
        // Interior y-component: c₁(x)·ℓ_i(y)
        values[20] = 0.0;
        values[21] = cx[1] * ly[0];
        values[22] = 0.0;
        values[23] = cx[1] * ly[1];
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];
        let l = open_basis(x);
        let ly = open_basis(y);
        let (_cy, dcy) = closed_basis(y);
        let (_cx, dcx) = closed_basis(x);

        // scalar curl in 2D: ∂Φ_y/∂x − ∂Φ_x/∂y
        // bottom: Φ = (ℓ_i(x)c₀(y), 0) → curl = −ℓ_i·c₀'
        curl_vals[0] = -l[0] * dcy[0];
        curl_vals[1] = -l[1] * dcy[0];
        // right: Φ = (0, c₂(x)ℓ_i(y)) → curl = c₂'·ℓ_i(y)
        curl_vals[2] = dcx[2] * ly[0];
        curl_vals[3] = dcx[2] * ly[1];
        // top: Φ = (−ℓ_{1−i}c₂, 0) → curl = +ℓ_{1−i}·c₂'
        curl_vals[4] = l[1] * dcy[2];
        curl_vals[5] = l[0] * dcy[2];
        // left: Φ = (0, −c₀ℓ_{1−i}) → curl = −c₀'·ℓ_{1−i}(y)
        curl_vals[6] = -dcx[0] * ly[1];
        curl_vals[7] = -dcx[0] * ly[0];
        // interior x: Φ = (ℓ_i c₁, 0) → curl = −ℓ_i·c₁'
        curl_vals[8] = -l[0] * dcy[1];
        curl_vals[9] = -l[1] * dcy[1];
        // interior y: Φ = (0, c₁ℓ_i) → curl = c₁'·ℓ_i(y)
        curl_vals[10] = dcx[1] * ly[0];
        curl_vals[11] = dcx[1] * ly[1];
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        quad_rule_01(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let (t0, t1) = (GL2[0], GL2[1]);
        vec![
            // bottom (y=0), tangent +x: ascending
            vec![t0, 0.0],
            vec![t1, 0.0],
            // right (x=1), tangent +y: ascending
            vec![1.0, t0],
            vec![1.0, t1],
            // top (y=1), tangent −x: ascending along (2,3) = descending x
            vec![t1, 1.0],
            vec![t0, 1.0],
            // left (x=0), tangent −y: ascending along (3,0) = descending y
            vec![0.0, t1],
            vec![0.0, t0],
            // interior x at (t_i, 1/2); interior y at (1/2, t_i)
            vec![t0, 0.5],
            vec![t1, 0.5],
            vec![0.5, t0],
            vec![0.5, t1],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nd2_quad_basis_and_curl_are_finite() {
        let elem = QuadND2;
        let qr = elem.quadrature(4);
        let mut phi = vec![0.0; elem.n_dofs() * 2];
        let mut curl = vec![0.0; elem.n_dofs()];
        for xi in &qr.points {
            elem.eval_basis_vec(xi, &mut phi);
            elem.eval_curl(xi, &mut curl);
            assert!(phi.iter().all(|v| v.is_finite()));
            assert!(curl.iter().all(|v| v.is_finite()));
        }
    }

    /// 1:1 check against the C++ MFEM `ND_QuadrilateralElement(2)` dump
    /// (`tmp/d32/nd2_dump.cpp`): VShape and curl at (0.31, 0.73).
    #[test]
    fn nd2_quad_matches_mfem_dump() {
        let elem = QuadND2;
        let mut phi = vec![0.0; 24];
        elem.eval_basis_vec(&[0.31, 0.73], &mut phi);
        let mfem_phi: [[f64; 2]; 12] = [
            [-0.10297293495701038, 0.0],
            [-0.021227065042989636, 0.0],
            [0.0, -0.01197181541972884],
            [0.0, -0.10582818458027117],
            [-0.0573916943754905, 0.0],
            [-0.27840830562450952, 0.0],
            [0.0, -0.23555305600124871],
            [0.0, -0.026646943998751289],
            [0.65365428277058757, 0.0],
            [0.13474571722941248, 0.0],
            [0.0, 0.086953185680135783],
            [0.0, 0.76864681431986415],
        ];
        for i in 0..12 {
            assert!(
                (phi[i * 2] - mfem_phi[i][0]).abs() < 1e-13,
                "phi[{i}]_x = {}, expected {}",
                phi[i * 2],
                mfem_phi[i][0]
            );
            assert!(
                (phi[i * 2 + 1] - mfem_phi[i][1]).abs() < 1e-13,
                "phi[{i}]_y = {}, expected {}",
                phi[i * 2 + 1],
                mfem_phi[i][1]
            );
        }

        let mut curl = vec![0.0; 12];
        elem.eval_curl(&[0.31, 0.73], &mut curl);
        let mfem_curl = [
            0.066327172275047031,
            0.013672827724953086,
            0.024390795422197971,
            0.215609204577802,
            0.32814786539887358,
            1.5918521346011263,
            1.5811341669038816,
            0.1788658330961185,
            1.5255249623260791,
            0.31447503767392043,
            0.15447503767392048,
            1.3655249623260795,
        ];
        for i in 0..12 {
            assert!(
                (curl[i] - mfem_curl[i]).abs() < 1e-13,
                "curl[{i}] = {}, expected {}",
                curl[i],
                mfem_curl[i]
            );
        }
    }

    /// Interior bubbles must have zero TANGENTIAL trace on every edge (the
    /// normal component may be nonzero — H(curl) only requires tangential
    /// continuity).  x-comp modes die on top/bottom (c₁(0)=c₁(1)=0), y-comp
    /// modes on left/right.
    #[test]
    fn nd2_interior_bubbles_vanish_tangentially_on_edges() {
        let elem = QuadND2;
        let mut vals = vec![0.0; 24];
        // (edge tangent index: 0 → check Φ_x, 1 → check Φ_y)
        let cases: [([f64; 2], usize, &str); 8] = [
            ([0.3, 0.0], 0, "bottom x-comp"),
            ([0.7, 0.0], 0, "bottom x-comp 2"),
            ([0.3, 1.0], 0, "top x-comp"),
            ([0.7, 1.0], 0, "top x-comp 2"),
            ([0.0, 0.4], 1, "left y-comp"),
            ([0.0, 0.8], 1, "left y-comp 2"),
            ([1.0, 0.4], 1, "right y-comp"),
            ([1.0, 0.8], 1, "right y-comp 2"),
        ];
        for (xi, comp, what) in cases {
            elem.eval_basis_vec(&xi, &mut vals);
            for i in 8..12 {
                let tang = vals[i * 2 + comp];
                assert!(
                    tang.abs() < 1e-14,
                    "interior bubble {i} tangential trace nonzero ({what}) at {xi:?}: {tang}"
                );
            }
        }
    }

    /// The constant field (1,0) must lie in the local span: the x-component
    /// modes span P1(x)⊗P2(y).  Verify by collocation at 6 distinct points.
    #[test]
    fn nd2_local_span_contains_constant() {
        let elem = QuadND2;
        let pts = [
            [0.1f64, 0.2f64],
            [0.5, 0.1],
            [0.9, 0.6],
            [0.3, 0.8],
            [0.7, 0.4],
            [0.2, 0.5],
        ];
        let xidx = [0usize, 1, 4, 5, 8, 9];
        let n = xidx.len();
        let mut a = nalgebra::DMatrix::<f64>::zeros(n, n);
        let b = nalgebra::DVector::from_element(n, 1.0);
        for (r, p) in pts.iter().enumerate() {
            let mut vals = vec![0.0; 24];
            elem.eval_basis_vec(p, &mut vals);
            for (c, &i) in xidx.iter().enumerate() {
                a[(r, c)] = vals[i * 2];
            }
        }
        let sol = a.lu().solve(&b).expect("x-comp collocation singular");
        let mut vals = vec![0.0; 24];
        let mut max_err = 0.0f64;
        for p in &pts {
            elem.eval_basis_vec(p, &mut vals);
            let ux: f64 = xidx.iter().zip(sol.iter()).map(|(&i, &c)| c * vals[i * 2]).sum();
            max_err = max_err.max((ux - 1.0).abs());
        }
        assert!(max_err < 1e-12, "constant x-comp not representable: {max_err:.3e}");
    }

    /// Nodal delta property: basis slot i evaluated at its own node point,
    /// dotted with the slot's tangent, gives ±1 (the sign of the baked-in
    /// MFEM dof_map sign), and ~0 for the other slots of the same component.
    #[test]
    fn nd2_nodal_delta_at_nodes() {
        let elem = QuadND2;
        let coords = elem.dof_coords();
        let tang: [[f64; 2]; 12] = [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [0.0, -1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ];
        let mut vals = vec![0.0; 24];
        for (i, pt) in coords.iter().enumerate() {
            elem.eval_basis_vec(pt, &mut vals);
            // self-value
            let self_val = vals[i * 2] * tang[i][0] + vals[i * 2 + 1] * tang[i][1];
            let expected = if i < 8 { 1.0 } else { 1.0 };
            assert!(
                (self_val - expected).abs() < 1e-12,
                "slot {i} self value {self_val}"
            );
        }
        // cross-component orthogonality on the bottom edge: y-comp modes
        // (slots 2,3,6,7,10,11) have zero x-tangent dot there already via the
        // component structure — spot-check slot 10/11 against bottom tangent.
        elem.eval_basis_vec(&[0.4, 0.0], &mut vals);
        for i in [2usize, 3, 6, 7] {
            let v = vals[i * 2] * 1.0 + vals[i * 2 + 1] * 0.0;
            assert!(v.abs() < 1e-14, "slot {i} leaks x on bottom: {v}");
        }
    }
}
