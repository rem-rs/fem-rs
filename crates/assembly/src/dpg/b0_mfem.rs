//! MFEM-bitwise assembly of the DPG primal block `B0` — the mixed diffusion
//! form `B0(i, j) = ∫ ∇v_test,i · ∇u_trial,j dx` between a continuous trial
//! space and the enriched L² test space.
//!
//! This mirrors MFEM `DiffusionIntegrator::AssembleElementMatrix2`
//! (`fem/bilininteg.cpp`) **operation by operation**, which the generic
//! [`crate::mixed::MixedAssembler`] volume path does not: that path stores
//! `J⁻ᵀ`-scaled gradients with the physical measure as the weight (algebraically
//! equal, differently rounded — the ex8 star-mesh `B0` disagrees with MFEM by a
//! median of ~5 ulp with a 333-entry tail above 100 ulp, D864).  The MFEM
//! arithmetic is:
//!
//! * `J(i, d) = Σ_k x_k[i] · ∂N_k/∂ξ_d` over geometry nodes `k` ascending;
//! * `det = j00·j11 − j01·j10` (`DenseMatrix::Det` closed form, **not** a
//!   general LU-based determinant) and `w = ip.weight / det` (signed — MFEM
//!   `Trans.Weight()`);
//! * physical gradients `Mult(dshape, CalcAdjugate(J))`: per entry a dot
//!   product over `j` ascending with `adja = [[j11, −j01], [−j10, j00]]`
//!   (`CalcAdjugate`'s square 2×2 branch — this is `det·J⁻ᵀ`, so the gradients
//!   carry the measure factor and the weight divides it back out);
//! * the element matrix accumulates with `AddMultABt`'s **k-outer** loop
//!   (`elmat(i,j) += te_adj(i,k) · (w·tr_adj(j,k))`, k ascending, one `+=` per
//!   dimension per quadrature point — NOT a per-point dot product).
//!
//! Essential-trial-dof columns are zeroed by the caller (MFEM
//! `EliminateTrialEssentialBC` with a zero solution vector).

use fem_linalg::CsrMatrix;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;

use crate::assembler::ref_elem_vol_for_space;
use crate::vector_assembler::geo_ref_elem_from_mesh;

/// Assemble `B0 = ∫ ∇v · ∇u` (unit coefficient), row space = `test`,
/// column space = `trial`, with MFEM `AssembleElementMatrix2` arithmetic.
pub fn assemble_b0_mfem<M, ST, SR>(trial: &SR, test: &ST, quad_order: u8) -> CsrMatrix<f64>
where
    M: MeshTopology,
    ST: FESpace<Mesh = M>,
    SR: FESpace<Mesh = M>,
{
    let mesh = test.mesh();
    let dim = mesh.dim() as usize;
    let n_test = test.n_dofs();
    let n_trial = trial.n_dofs();
    // Hand-built CSR: every L2 test row is element-private and receives
    // exactly n_trial entries (one block), so the layout is fixed up front.
    let nnz = n_test * n_trial;
    let mut row_ptr = vec![0usize; n_test + 1];
    let mut col_idx = vec![0u32; nnz];
    let mut values = vec![0.0_f64; nnz];
    let mut pos = 0usize;

    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        let elem_type_straight =
            matches!(et, ElementType::Tri3 | ElementType::Tet4 | ElementType::Line2);

        let ref_c = ref_elem_vol_for_space(trial, et, trial.order());
        let ref_r = ref_elem_vol_for_space(test, et, test.order());
        let n_r = ref_r.n_dofs();
        let n_c = ref_c.n_dofs();

        let quad = if test.order() >= trial.order() {
            ref_r.quadrature(quad_order)
        } else {
            ref_c.quadrature(quad_order)
        };

        let global_rows: Vec<usize> =
            test.element_dofs(e).iter().map(|&d| d as usize).collect();
        let global_cols: Vec<usize> =
            trial.element_dofs(e).iter().map(|&d| d as usize).collect();

        let geo_elem = if elem_type_straight {
            None
        } else {
            geo_ref_elem_from_mesh(mesh, e)
        };
        let nodes = mesh.element_nodes(e);
        let geom_nodes: Vec<u32> = if mesh.geom_order() > 1 && geo_elem.is_some() {
            mesh.geometry_nodes(e).to_vec()
        } else {
            nodes.to_vec()
        };

        let mut m_elem = vec![0.0_f64; n_r * n_c];
        let mut grad_r = vec![0.0_f64; n_r * dim];
        let mut grad_c = vec![0.0_f64; n_c * dim];
        let mut adj_r = vec![0.0_f64; n_r * dim];
        let mut adj_c = vec![0.0_f64; n_c * dim];

        for (q, xi) in quad.points.iter().enumerate() {
            // Geometry: J entries, then MFEM's closed-form det.  (The physical
            // point is not needed by this integrand and is not computed.)
            let (j00, j01, j10, j11): (f64, f64, f64, f64) = if let Some(ref ge) = geo_elem {
                let n_geo = ge.n_dofs();
                let mut grad_geo = vec![0.0_f64; n_geo * dim];
                ge.eval_grad_basis(xi, &mut grad_geo);
                let mut j = [[0.0_f64; 2]; 2];
                for k in 0..n_geo {
                    let xk = mesh.geom_coords_of(geom_nodes[k]);
                    for i in 0..dim {
                        for d in 0..dim {
                            j[i][d] += xk[i] * grad_geo[k * dim + d];
                        }
                    }
                }
                (j[0][0], j[0][1], j[1][0], j[1][1])
            } else {
                let tr = fem_mesh::ElementTransformation::from_simplex_nodes(mesh, &nodes);
                let jac = tr.jacobian();
                (jac[(0, 0)], jac[(0, 1)], jac[(1, 0)], jac[(1, 1)])
            };
            let det = j00 * j11 - j01 * j10; // MFEM DenseMatrix::Det (2×2)
            let w = quad.weights[q] / det; // MFEM: ip.weight / Trans.Weight()

            // Physical gradients = dshape · CalcAdjugate(J): per entry a dot
            // product over the reference dimension, ascending.
            ref_r.eval_grad_basis(xi, &mut grad_r);
            ref_c.eval_grad_basis(xi, &mut grad_c);
            let adj = [[j11, -j01], [-j10, j00]];
            for i in 0..n_r {
                for k in 0..dim {
                    let mut s = 0.0;
                    for j in 0..dim {
                        s += grad_r[i * dim + j] * adj[j][k];
                    }
                    adj_r[i * dim + k] = s;
                }
            }
            for i in 0..n_c {
                for k in 0..dim {
                    let mut s = 0.0;
                    for j in 0..dim {
                        s += grad_c[i * dim + j] * adj[j][k];
                    }
                    adj_c[i * dim + k] = s;
                }
            }

            // AddMultABt(A = te_adj, B = w·tr_adj): k-outer, one `+=` per
            // dimension per quadrature point — not a per-point dot product.
            for k in 0..dim {
                for j in 0..n_c {
                    let bjk = w * adj_c[j * dim + k];
                    for i in 0..n_r {
                        m_elem[i * n_c + j] += adj_r[i * dim + k] * bjk;
                    }
                }
            }
        }

        // MFEM scatters the element matrix into the global SparseMatrix in
        // (test row, trial col) order through the linked list (head
        // insertion);  preserves the list order, so every
        // element-private L2 row stores its trial block REVERSED (local j =
        // n_c-1 … 0) — exactly like .  The entry VALUES
        // are unaffected, but / consume the stored order, and
        // the sorted order carried last-ulp noise into every outer-PCG
        // iteration (D101: the ex8 fork surfaced at iteration 9 only because
        // the print rounds it away before that).
        for (i, &gi) in global_rows.iter().enumerate() {
            row_ptr[gi as usize] = pos;
            for (j, &gj) in global_cols.iter().enumerate().rev() {
                let v = m_elem[i * n_c + j];
                if v.abs() > 1e-30 {
                    col_idx[pos] = gj as u32;
                    values[pos] = v;
                    pos += 1;
                }
            }
        }
    }

    row_ptr[n_test] = pos;
    return CsrMatrix { nrows: n_test, ncols: n_trial, row_ptr, col_idx, values };
}
