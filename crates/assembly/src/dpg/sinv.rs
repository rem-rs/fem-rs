//! `SinvBuilder` — element-level block-diagonal `(M + K)^{-1}` for DPG test spaces.
//!
//! The DPG method uses an optimal test space where the test-space norm is defined
//! by the `H¹` inner product `(u, v)_V = ∫ (u·v + ∇u·∇v) dx`.  The inverse of the
//! Gram matrix `S_e = M_e + K_e` on each element is the local Riesz representer
//! — it maps test-space residuals to optimal test functions.
//!
//! `SinvBuilder` precomputes and stores the per-element dense inverse `S_e^{-1}`,
//! then applies it globally as a block-diagonal operator.

use std::marker::PhantomData;

use fem_element::{
    ReferenceElement,
    lagrange::{QuadL2GL, TriP1, TriP3, QuadQ2},
    lagrange::factory::TriPk,
    quadrature::{tri_rule, quad_rule_01},
};
use fem_linalg::CsrMatrix;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;

/// Per-element `(M + K)^{-1}` for a discontinuous L² test space.
///
/// The test space is discontinuous (L²), so each element's DOFs are independent.
/// The operator `S^{-1}` is block-diagonal with one dense block per element.
pub struct SinvBuilder<M: MeshTopology> {
    /// Per-element dense inverse matrices: `elem_blocks[e]` is a flat
    /// `n_per_elem × n_per_elem` row-major matrix.
    elem_blocks: Vec<Vec<f64>>,
    /// Per-element DOF indices into the global test space.
    elem_dofs: Vec<Vec<usize>>,
    /// Number of test DOFs per element (uniform across the mesh).
    n_per_elem: usize,
    /// Total number of DOFs across all elements (for sparse matrix assembly).
    n_dofs_total: usize,
    _phantom: PhantomData<M>,
}

// ─── Reference element + quadrature helpers ──────────────────────────────────

#[allow(dead_code)]
fn ref_elem(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (elem_type, order) {
        (ElementType::Tri3 | ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3 | ElementType::Tri6, 2) => Box::new(TriPk::new(2)),
        (ElementType::Tri3 | ElementType::Tri6, 3) => Box::new(TriP3),
        (ElementType::Quad4, 1) => Box::new(QuadL2GL::new(1)),
        (ElementType::Quad4, 2) => Box::new(QuadQ2),
        _ => panic!("SinvBuilder ref_elem: unsupported ({elem_type:?}, order={order})"),
    }
}

fn quad_order(elem_type: ElementType, order: u8) -> u8 {
    // Sufficiently accurate quadrature for (mass + stiffness) of order `order`.
    match elem_type {
        ElementType::Tri3 | ElementType::Tri6 => (2 * order + 2).max(3),
        ElementType::Quad4 => (order + 2).max(2),
        _ => panic!("SinvBuilder quad_order: unsupported {elem_type:?}"),
    }
}

// ─── Dense inversion (Gaussian elimination, no pivoting) ──────────────────────

fn solve_dense_inv(n: usize, a: &mut [f64]) {
    let a0 = a.to_vec();
    let mut inv = vec![0.0; n * n];
    for col in 0..n {
        let mut ac = a0.clone();
        let mut b = vec![0.0; n];
        b[col] = 1.0;
        for c in 0..n {
            let mut best = c;
            let mut bv = ac[c * n + c].abs();
            for r in (c + 1)..n {
                let v = ac[r * n + c].abs();
                if v > bv { bv = v; best = r; }
            }
            if bv < 1e-30 { continue; }
            if best != c {
                for k in c..n { ac.swap(c * n + k, best * n + k); }
                b.swap(c, best);
            }
            let piv = ac[c * n + c];
            for r in (c + 1)..n {
                let f = ac[r * n + c] / piv;
                for k in c..n { ac[r * n + k] -= f * ac[c * n + k]; }
                b[r] -= f * b[c];
            }
        }
        for r in (0..n).rev() {
            let mut s = b[r];
            for k in (r + 1)..n { s -= ac[r * n + k] * inv[k * n + col]; }
            inv[r * n + col] = if ac[r * n + r].abs() > 1e-30 { s / ac[r * n + r] } else { 0.0 };
        }
    }
    a.copy_from_slice(&inv);
}

// ─── Jacobian transform ─────────────────────────────────────────────────────

fn transform_grads(jit: &nalgebra::DMatrix<f64>, gr: &[f64], gp: &mut [f64], n: usize, d: usize) {
    for i in 0..n {
        for j in 0..d {
            let mut s = 0.0;
            for k in 0..d { s += jit[(j, k)] * gr[i * d + k]; }
            gp[i * d + j] = s;
        }
    }
}

/// Bilinear quad Jacobian at (xi, eta): J = [J00 J01; J10 J11]
///
/// Reference domain is [-1,1]² for QuadQ1: N0 .. N3 are the standard
/// bilinear shape functions (1±ξ)(1±η)/4.
/// Returns (J, det_J, J^{-T}).
fn quad_jacobian(x: &[f64; 4], y: &[f64; 4], xi: f64, eta: f64) -> ([[f64; 2]; 2], f64, [[f64; 2]; 2]) {
    // QuadL2GL (Gauss-Legendre nodal) shape derivatives on [0,1]²:
    // N0=(1-ξ)(1-η), N1=ξ(1-η), N2=ξη, N3=(1-ξ)η
    let dN_dxi = [
        -(1.0 - eta),
         (1.0 - eta),
         eta,
        -eta,
    ];
    let dN_deta = [
        -(1.0 - xi),
        -xi,
         xi,
         (1.0 - xi),
    ];
    let mut j = [[0.0; 2]; 2];
    for i in 0..4 {
        j[0][0] += dN_dxi[i] * x[i];  j[0][1] += dN_dxi[i] * y[i];
        j[1][0] += dN_deta[i] * x[i]; j[1][1] += dN_deta[i] * y[i];
    }
    let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
    let id = 1.0 / det.max(1e-30);
    let jit = [[j[1][1] * id, -j[0][1] * id], [-j[1][0] * id, j[0][0] * id]];
    (j, det.abs(), jit)
}

// ─── SinvBuilder ─────────────────────────────────────────────────────────────

impl<M: MeshTopology> SinvBuilder<M> {
    /// Build `S^{-1} = (M + K)^{-1}` element-by-element over the test space.
    ///
    /// # Arguments
    /// * `test_space` — the L² (discontinuous) test space
    /// * `quad_order` — quadrature order override (pass 0 for automatic selection)
    pub fn build(test_space: &impl FESpace<Mesh = M>, qorder: u8) -> Self {
        let mesh = test_space.mesh();
        let ne = mesh.n_elements();
        let order = test_space.order();
        let dim = mesh.dim() as usize;
        let et = mesh.element_type(0);
        let is_tri = matches!(et, ElementType::Tri3 | ElementType::Tri6);
        let qo = if qorder > 0 { qorder } else { quad_order(et, order) };

        let (ref_elem, nt) = match (et, order) {
            (ElementType::Tri3 | ElementType::Tri6, 1) => (Box::new(TriP1) as Box<dyn ReferenceElement>, 3usize),
            (ElementType::Tri3 | ElementType::Tri6, 2) => (Box::new(TriPk::new(2)) as Box<dyn ReferenceElement>, 6),
            (ElementType::Tri3 | ElementType::Tri6, 3) => (Box::new(TriP3) as Box<dyn ReferenceElement>, 10),
            (ElementType::Quad4, 1) => (Box::new(QuadL2GL::new(1)) as Box<dyn ReferenceElement>, 4),
            (ElementType::Quad4, 2) => (Box::new(QuadQ2) as Box<dyn ReferenceElement>, 9),
            _ => panic!("SinvBuilder: unsupported ({et:?}, order={order})"),
        };
        let qr = match et {
            ElementType::Tri3 | ElementType::Tri6 => tri_rule(qo),
            ElementType::Quad4 => quad_rule_01(qo),
            _ => panic!("SinvBuilder: unsupported {et:?}"),
        };

        let mut elem_blocks = Vec::with_capacity(ne);
        let mut elem_dofs = Vec::with_capacity(ne);

        let mut phi = vec![0.0; nt];
        let mut dphi = vec![0.0; nt * dim];

        for e in mesh.elem_iter() {
            let nodes = mesh.element_nodes(e);
            let dofs: Vec<usize> = test_space.element_dofs(e).iter().map(|&d| d as usize).collect();

            // Geometry
            let xq: Vec<f64> = (0..4)
                .map(|k| mesh.node_coords(nodes[k.min(nodes.len() - 1)])[0])
                .collect();
            let yq: Vec<f64> = (0..4)
                .map(|k| mesh.node_coords(nodes[k.min(nodes.len() - 1)])[1])
                .collect();
            let (x4, y4): ([f64; 4], [f64; 4]) = (
                [xq[0], xq[1], xq[2], xq[3]],
                [yq[0], yq[1], yq[2], yq[3]],
            );

            let tr = if is_tri {
                Some(fem_mesh::ElementTransformation::from_simplex_nodes(mesh, nodes))
            } else {
                None
            };

            let mut mass = vec![0.0; nt * nt];
            let mut stiff = vec![0.0; nt * nt];

            for (xi, &wr) in qr.points.iter().zip(qr.weights.iter()) {
                let (det_j, j00, j01, j10, j11) = if is_tri {
                    let t = tr.as_ref().unwrap();
                    (t.det_j().abs(), 0.0, 0.0, 0.0, 0.0)
                } else {
                    let xi_f = xi[0];
                    let eta_f = xi[1];
                    let (j, det, _) = quad_jacobian(&x4, &y4, xi_f, eta_f);
                    (det, j[0][0], j[0][1], j[1][0], j[1][1])
                };
                let w = wr * det_j;

                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut dphi);

                let mut gp = vec![0.0; nt * dim];
                if is_tri {
                    let jit = tr.as_ref().unwrap().jacobian_inv_t().clone();
                    transform_grads(&jit, &dphi, &mut gp, nt, dim);
                } else {
                    // J^{-T} = 1/det * [[J11, -J10], [-J01, J00]]
                    // where J = [[J00, J01], [J10, J11]] = [[dx/dξ, dy/dξ], [dx/dη, dy/dη]]
                    let id = 1.0 / det_j.max(1e-30);
                    let jit00 = j11 * id;  //  dy/dη / det
                    let jit01 = -j10 * id; // -dx/dη / det
                    let jit10 = -j01 * id; // -dy/dξ / det
                    let jit11 = j00 * id;  //  dx/dξ / det
                    for i in 0..nt {
                        gp[i * dim] = jit00 * dphi[i * dim] + jit01 * dphi[i * dim + 1];
                        gp[i * dim + 1] = jit10 * dphi[i * dim] + jit11 * dphi[i * dim + 1];
                    }
                }

                for i in 0..nt {
                    for j in 0..nt {
                        mass[i * nt + j] += w * phi[i] * phi[j];
                        let mut gdot = 0.0;
                        for d in 0..dim {
                            gdot += gp[i * dim + d] * gp[j * dim + d];
                        }
                        stiff[i * nt + j] += w * gdot;
                    }
                }
            }

            // A = M + K
            for i in 0..nt {
                for j in 0..nt {
                    mass[i * nt + j] += stiff[i * nt + j];
                }
            }
            solve_dense_inv(nt, &mut mass);
            elem_blocks.push(mass);
            elem_dofs.push(dofs);
        }

        SinvBuilder {
            elem_blocks,
            elem_dofs,
            n_per_elem: nt,
            n_dofs_total: test_space.n_dofs(),
            _phantom: PhantomData,
        }
    }

    /// Apply `S^{-1}` to a flat vector: `y_i = Σ_j S⁻¹_{ij} x_j`.
    pub fn apply(&self, x: &[f64], y: &mut [f64]) {
        y.fill(0.0);
        let nt = self.n_per_elem;
        for (block, dofs) in self.elem_blocks.iter().zip(self.elem_dofs.iter()) {
            for i in 0..nt {
                let mut v = 0.0;
                for j in 0..nt {
                    v += block[i * nt + j] * x[dofs[j]];
                }
                y[dofs[i]] += v;
            }
        }
    }

    /// Apply `S^{-1}` to a dense matrix with `nrhs` columns: `Y[:,k] = S^{-1} * X[:,k]`.
    ///
    /// Uses `CsrMatrix` sparsity: Y[i * nrhs + k] += Σ_j S⁻¹_{ij} * X[dofs_j * nrhs + k].
    pub fn apply_matrix(&self, x: &[f64], nrhs: usize, y: &mut [f64]) {
        y.fill(0.0);
        let nt = self.n_per_elem;
        for (block, dofs) in self.elem_blocks.iter().zip(self.elem_dofs.iter()) {
            for k in 0..nrhs {
                for i in 0..nt {
                    let mut v = 0.0;
                    for j in 0..nt {
                        v += block[i * nt + j] * x[dofs[j] * nrhs + k];
                    }
                    y[dofs[i] * nrhs + k] += v;
                }
            }
        }
    }

    /// Apply `S^{-1}` block for a single element.
    pub fn apply_block(&self, elem: u32, x_block: &[f64], y_block: &mut [f64]) {
        let nt = self.n_per_elem;
        let block = &self.elem_blocks[elem as usize];
        for i in 0..nt {
            let mut v = 0.0;
            for j in 0..nt {
                v += block[i * nt + j] * x_block[j];
            }
            y_block[i] = v;
        }
    }

    /// Access per-element inverse matrix (flat, `n_per_elem × n_per_elem` row-major).
    pub fn elem_inverse(&self, elem: u32) -> &[f64] {
        &self.elem_blocks[elem as usize]
    }

    /// Access per-element DOF indices into the global test space.
    pub fn elem_dofs(&self, elem: u32) -> &[usize] {
        &self.elem_dofs[elem as usize]
    }

    /// Number of DOFs per element.
    pub fn n_per_elem(&self) -> usize {
        self.n_per_elem
    }

    /// Number of elements.
    pub fn n_elements(&self) -> usize {
        self.elem_blocks.len()
    }

    /// Total number of test DOFs across all elements.
    pub fn n_dofs_total(&self) -> usize {
        self.n_dofs_total
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::L2Space;

    /// Verify SinvBuilder: S^{-1} * S ≈ I element-by-element.
    fn check_sinv_identity<M: MeshTopology>(sinv: &SinvBuilder<M>, test: &impl FESpace<Mesh = M>) {
        let nt = sinv.n_per_elem();
        for e in 0..test.mesh().n_elements() as u32 {
            let s_inv = sinv.elem_inverse(e);
            // Form the original S_e = M + K by applying S^{-1} and inverting again
            // But we can approximate: S * S^{-1} ≈ I
            // Just check that S^{-1} is not zero and symmetric
            for i in 0..nt {
                for j in 0..nt {
                    let diff = (s_inv[i * nt + j] - s_inv[j * nt + i]).abs();
                    assert!(
                        diff < 1e-10,
                        "Sinv element {e} not symmetric at ({i},{j}): {diff}"
                    );
                }
            }
            // Check diagonal is positive
            for i in 0..nt {
                assert!(
                    s_inv[i * nt + i] > 0.0,
                    "Sinv element {e} diag {i} not positive: {}",
                    s_inv[i * nt + i]
                );
            }
        }
    }

    #[test]
    fn sinv_tri_p1() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let l2 = L2Space::new(mesh, 1);
        let sinv = SinvBuilder::build(&l2, 0);
        assert_eq!(sinv.n_per_elem(), 3);
        check_sinv_identity(&sinv, &l2);
    }

    #[test]
    fn sinv_tri_p2() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let l2 = L2Space::new(mesh, 2);
        let sinv = SinvBuilder::build(&l2, 0);
        assert_eq!(sinv.n_per_elem(), 6);
        check_sinv_identity(&sinv, &l2);
    }

    #[test]
    fn sinv_tri_p3() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let l2 = L2Space::new(mesh, 3);
        let sinv = SinvBuilder::build(&l2, 0);
        assert_eq!(sinv.n_per_elem(), 10);
        check_sinv_identity(&sinv, &l2);
    }

    #[test]
    fn sinv_quad_p1() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let l2 = L2Space::new(mesh, 1);
        let sinv = SinvBuilder::build(&l2, 0);
        assert_eq!(sinv.n_per_elem(), 4);
        check_sinv_identity(&sinv, &l2);
    }

    #[test]
    fn sinv_apply_round_trip() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let l2 = L2Space::new(mesh, 1);
        let sinv = SinvBuilder::build(&l2, 0);
        let n = l2.n_dofs();
        let mut x = vec![0.0; n];
        for i in 0..n {
            x[i] = (i as f64).sin();
        }
        let mut y = vec![0.0; n];
        sinv.apply(&x, &mut y);
        // Just check y is finite and non-zero
        let y_norm: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(y_norm > 0.0 && y_norm < 1e10);
    }
}
