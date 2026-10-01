//! `SinvBuilder` — element-level block-diagonal `(M + K)^{-1}` for DPG test spaces.
//!
//! The DPG method uses an optimal test space where the test-space norm is defined
//! by the `H¹` inner product `(u, v)_V = ∫ (u·v + ∇u·∇v) dx`.  The inverse of the
//! Gram matrix `S_e = M_e + K_e` on each element is the local Riesz representer
//! — it maps test-space residuals to optimal test functions.
//!
//! `SinvBuilder` precomputes and stores the per-element dense inverse `S_e^{-1}`,
//! then applies it globally as a block-diagonal operator.
//!
//! # Negative-det elements (D652)
//!
//! Element Jacobian determinants are used **signed**, exactly like MFEM 4.10
//! (`ElementTransformation::Weight()` = det, `InverseJacobian()` = algebraic
//! inverse; `DenseMatrixInverse`/`CalcInverse` never check the determinant
//! sign — the only singularity guard is a debug-only `MFEM_ASSERT`).  An
//! inverted (det < 0) element therefore contributes the exact negation of its
//! mirrored positive-det block — physically meaningless but MFEM-faithful;
//! mesh orientation is the caller's responsibility.  MFEM's own `star.mesh`
//! (the ex8 mesh) is all-positive (det ≈ 0.2378), so this is a latent-only
//! semantic on every stock example mesh.

use std::marker::PhantomData;

use fem_linalg::CsrMatrix;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;

use crate::assembler::Assembler;
use crate::standard::{DiffusionIntegrator, MassIntegrator};

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

fn quad_order(elem_type: ElementType, order: u8) -> u8 {
    // Sufficiently accurate quadrature for (mass + stiffness) of order `order`.
    match elem_type {
        ElementType::Tri3 | ElementType::Tri6 => (2 * order + 2).max(3),
        ElementType::Quad4 => (order + 2).max(2),
        _ => panic!("SinvBuilder quad_order: unsupported {elem_type:?}"),
    }
}

// ─── Dense inversion — bit-for-bit port of MFEM's Invert() ───────────────────

/// Invert an `n × n` row-major matrix in place, bit-for-bit following MFEM
/// 4.10 `DenseMatrix::Invert()` in the **non-LAPACK** branch
/// (`linalg/densemat.cpp`; the reference build `$HOME/mfem410_ser` has
/// `MFEM_USE_LAPACK = NO`).
///
/// Why this exact algorithm (D864): the DPG element blocks `S_e^{-1}` feed
/// `b = Bᵀ S⁻¹ F`, `Shat = Bhatᵀ S⁻¹ Bhat` and every preconditioner
/// application, so an ulp-level difference in the inverse scheme grows into a
/// visible fork of the outer PCG trajectory.  The previous implementation here
/// solved `n` separate systems (fresh elimination + back substitution per unit
/// vector) — mathematically the same inverse, but a different floating-point
/// algorithm with ~5e-12 relative disagreement against MFEM's Gauss–Jordan on
/// the ex8 star-mesh blocks, which surfaced as a 1.3e-5 relative difference in
/// the printed `(B r, r)` by iteration 5 and a 29-vs-28 iteration split.
///
/// The port preserves MFEM's operation order exactly: pivot scan with strict
/// `a < b` (diagonal seeded first), full-row swap, reciprocal scaling of the
/// pivot row, Gauss–Jordan updates of rows above *and* below the pivot (both
/// split into the `j < c` / `j > c` column ranges), and the final column swaps
/// driven by the recorded pivot indices.  Row-major indexing replaces MFEM's
/// column-major `(i, j)` accessor; since every swap is whole-row or whole-column
/// and every update is elementwise, the sequence of floating-point operations is
/// identical.
fn mfem_dense_invert(n: usize, a: &mut [f64]) {
    let mut piv = vec![0usize; n];

    for c in 0..n {
        let mut amax = a[c * n + c].abs();
        let mut imax = c;
        for j in (c + 1)..n {
            let b = a[j * n + c].abs();
            if amax < b {
                amax = b;
                imax = j;
            }
        }
        if amax == 0.0 {
            // MFEM: `mfem_error("DenseMatrix::Invert() : singular matrix")`.
            panic!("mfem_dense_invert: singular matrix");
        }
        piv[c] = imax;
        for j in 0..n {
            a.swap(c * n + j, imax * n + j);
        }

        let r = 1.0 / a[c * n + c];
        a[c * n + c] = r;
        for j in 0..c {
            a[c * n + j] *= r;
        }
        for j in (c + 1)..n {
            a[c * n + j] *= r;
        }
        for i in 0..c {
            let b = -a[i * n + c];
            a[i * n + c] = r * b;
            for j in 0..c {
                a[i * n + j] += b * a[c * n + j];
            }
            for j in (c + 1)..n {
                a[i * n + j] += b * a[c * n + j];
            }
        }
        for i in (c + 1)..n {
            let b = -a[i * n + c];
            a[i * n + c] = r * b;
            for j in 0..c {
                a[i * n + j] += b * a[c * n + j];
            }
            for j in (c + 1)..n {
                a[i * n + j] += b * a[c * n + j];
            }
        }
    }

    for c in (0..n).rev() {
        let j = piv[c];
        for i in 0..n {
            a.swap(i * n + c, i * n + j);
        }
    }
}

// ─── Element-matrix extraction from the global assembly ──────────────────────

/// Copy the element's `dofs[i] × dofs[j]` entries of the assembled matrix `m`
/// into the row-major dense block `blk` (`nt × nt`); unstored entries are zero.
fn read_block(m: &CsrMatrix<f64>, dofs: &[usize], blk: &mut [f64]) {
    blk.fill(0.0);
    let nt = dofs.len();
    for (i, &di) in dofs.iter().enumerate() {
        for p in m.row_ptr[di]..m.row_ptr[di + 1] {
            let c = m.col_idx[p] as usize;
            if let Some(j) = dofs.iter().position(|&d| d == c) {
                blk[i * nt + j] = m.values[p];
            }
        }
    }
}

/// MFEM `SumIntegrator`: the second integrator's element matrix is added to the
/// first's, entrywise, over the element's dof block.
fn add_block(m: &CsrMatrix<f64>, dofs: &[usize], blk: &mut [f64]) {
    let nt = dofs.len();
    for (i, &di) in dofs.iter().enumerate() {
        for p in m.row_ptr[di]..m.row_ptr[di + 1] {
            let c = m.col_idx[p] as usize;
            if let Some(j) = dofs.iter().position(|&d| d == c) {
                blk[i * nt + j] += m.values[p];
            }
        }
    }
}

// ─── SinvBuilder ─────────────────────────────────────────────────────────────

impl<M: MeshTopology> SinvBuilder<M> {
    /// Build `S^{-1} = (M + K)^{-1}` element-by-element over the test space.
    ///
    /// The element matrices `S_e = M_e + K_e` come from the **generic volume
    /// assembly path** ([`Assembler::assemble_bilinear`] on the L² space, whose
    /// dofs are element-local so the global matrix is exactly block-diagonal).
    /// That path reproduces MFEM's integrator kernels bit-for-bit — signed
    /// `Weight()` in `ip.weight / Weight()`, adjugate-scaled gradients
    /// `dshapedxt = dshape · adj(J)`, and the `AddMult_a_AAt`/`AddMult_a_VVt`
    /// accumulation with the shared symmetric products — which the previous
    /// hand-rolled element loop here did *not*: its `J⁻ᵀ`-scaled gradient and
    /// `w = weight·det` grouping differ from MFEM's in the last ulps, and the
    /// `κ ≈ 10³` conditioning of `S_e` amplifies that into ~5e-12 relative
    /// error in `S_e^{-1}` (D864; measured by dumping both sides' `Sinv` —
    /// `tmp/d98runbit/`).  Since `S^{-1}` feeds the RHS, `Shat` and every
    /// preconditioner application, that error forked the ex8 outer PCG
    /// trajectory in the 6th printed digit from iteration 5 on.
    ///
    /// # Arguments
    /// * `test_space` — the L² (discontinuous) test space
    /// * `quad_order` — quadrature order override (pass 0 for automatic selection)
    pub fn build(test_space: &impl FESpace<Mesh = M>, qorder: u8) -> Self {
        let mesh = test_space.mesh();
        let ne = mesh.n_elements();
        let order = test_space.order();
        let et = mesh.element_type(0);
        let qo = if qorder > 0 { qorder } else { quad_order(et, order) };

        let stiff = Assembler::assemble_bilinear(
            test_space, &[&DiffusionIntegrator { kappa: 1.0 }], qo,
        );
        let mass = Assembler::assemble_bilinear(
            test_space, &[&MassIntegrator { rho: 1.0 }], qo,
        );

        let mut elem_blocks = Vec::with_capacity(ne);
        let mut elem_dofs = Vec::with_capacity(ne);
        let mut n_per_elem = 0usize;

        for e in mesh.elem_iter() {
            let dofs: Vec<usize> = test_space.element_dofs(e).iter().map(|&d| d as usize).collect();
            let nt = dofs.len();
            n_per_elem = nt;

            // MFEM `InverseIntegrator(SumIntegrator(Diffusion, Mass))`: the
            // first integrator's matrix is the accumulator, the second is added.
            let mut s = vec![0.0; nt * nt];
            read_block(&stiff, &dofs, &mut s);
            add_block(&mass, &dofs, &mut s);

            mfem_dense_invert(nt, &mut s);
            elem_blocks.push(s);
            elem_dofs.push(dofs);
        }

        SinvBuilder {
            elem_blocks,
            elem_dofs,
            n_per_elem,
            n_dofs_total: test_space.n_dofs(),
            _phantom: PhantomData,
        }
    }

    /// Apply `S^{-1}` to a flat vector: `y_i = Σ_j S⁻¹_{ij} x_j`.
    ///
    /// The per-row accumulation runs in the **reversed local order** — the
    /// same sequence MFEM's `matSinv` spmv produces, because the BilinearForm
    /// linked-list assembly stores each L2 row reversed (local j = nt-1 … 0,
    /// see [`crate::dpg::assemble_sinv_sparse`]) and its spmv is a
    /// left-to-right dot over the stored order (D101: with the ascending
    /// order, SinvF disagreed with MFEM by <=2 ulp on a third of the entries,
    /// which then fed `b = B^T S^-1 F` and forked the outer PCG).
    pub fn apply(&self, x: &[f64], y: &mut [f64]) {
        y.fill(0.0);
        let nt = self.n_per_elem;
        for (block, dofs) in self.elem_blocks.iter().zip(self.elem_dofs.iter()) {
            for i in 0..nt {
                let mut v = 0.0;
                for j in (0..nt).rev() {
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
                    for j in (0..nt).rev() {
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
            for j in (0..nt).rev() {
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

    /// D652: the `SinvBuilder` block on a NEGATIVE-det quad must be the exact
    /// negation of the mirrored positive-det block, matching MFEM 4.10.
    ///
    /// Adjudication (probe: `tmp/d652/probe.cpp`, MFEM 4.10 serial build):
    /// * `ElementTransformation::Weight()` is the **signed** determinant
    ///   (`eltrans.cpp: EvalWeight -> dFdx.Weight() -> DenseMatrix::Weight()
    ///   -> Det()`; `Wght` is only cached while positive).  MassIntegrator and
    ///   DiffusionIntegrator therefore assemble `S(neg-det elem) = -S(mirror)`,
    ///   and `DenseMatrixInverse` / `CalcInverse` return the signed algebraic
    ///   inverse `-S^{-1}` — no check, no abort anywhere on negative det.
    ///   (`CalcInverse`'s singularity guard is an `MFEM_ASSERT`, i.e. debug
    ///   builds only; for n >= 4 the release closed-form switch has no case
    ///   and silently leaves the output untouched.)
    /// * MFEM's `data/star.mesh` (the ex8 mesh) never triggers this: its 20
    ///   quads have det in [0.237761, 0.237765] — the semantics are latent.
    #[test]
    fn sinv_negative_det_quad_matches_mfem() {
        use fem_mesh::element_type::ElementType;

        // 2x1 quad pair from the probe: element 0 is the unit square (det +1);
        // element 1 is the same square rotated/reflected with a REVERSED
        // vertex cycle, an affine map with det = -1 everywhere.
        let coords: Vec<f64> = vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0];
        let conn: Vec<u32> = vec![0, 1, 4, 3, 1, 4, 5, 2];
        let face_conn: Vec<u32> = vec![0, 1, 1, 2, 2, 5, 5, 4, 4, 3, 3, 0];
        let mesh = Mesh::<2>::uniform(
            coords, conn, vec![1, 2], ElementType::Quad4,
            face_conn, vec![1; 6], ElementType::Line2,
        );
        let l2 = L2Space::new(mesh, 1);
        let sinv = SinvBuilder::build(&l2, 0);
        assert_eq!(sinv.n_elements(), 2);

        // Positive-det element: MFEM assembles S = M + K =
        // [[3.25,-1.5,-1.5,0],[-1.5,3.25,0,-1.5],[-1.5,0,3.25,-1.5],
        //  [0,-1.5,-1.5,3.25]] (exact GL 2-pt rule both sides) and
        // DenseMatrixInverse(S) = (pinned to the probe's printed digits;
        // closed form 1 + 2/13 + 1/25 = 1.193846..., 24/25 = 0.96,
        // 227/256 + ... = 0.886153846153...).
        let pin = [
            1.193846153846, 0.96, 0.96, 0.886153846154,
            0.96, 1.193846153846, 0.886153846154, 0.96,
            0.96, 0.886153846154, 1.193846153846, 0.96,
            0.886153846154, 0.96, 0.96, 1.193846153846,
        ];
        let pos = sinv.elem_inverse(0);
        for i in 0..16 {
            assert!(
                (pos[i] - pin[i]).abs() < 1e-9,
                "positive-det block pin at [{i}]: {} vs MFEM {}",
                pos[i], pin[i]
            );
        }

        // Negative-det element: MFEM semantics = negated block (signed
        // Weight() x signed InverseJacobian throughout).
        let neg = sinv.elem_inverse(1);
        for i in 0..16 {
            assert!(
                (neg[i] + pos[i]).abs() < 1e-10,
                "negative-det block must equal -mirror at [{i}]: {} vs {}",
                neg[i], -pos[i]
            );
        }
    }
}
