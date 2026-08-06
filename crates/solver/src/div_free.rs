//! Divergence-free projection for H(curl) vector fields.

use fem_linalg::{CooMatrix, CsrMatrix};

/// Project `rhs` (H(curl) DOF vector) onto the divergence-free subspace:
/// `jr = rhs − G·(G^T·M·G)⁻¹·G^T·M·rhs`
///
/// Uses mass lumping: `A_h1 = G^T · diag(M) · G` for the coarse operator.
/// This is spectrally equivalent to the full product and avoids expensive
/// sparse triple products.
///
/// `G` is the gradient matrix (n_nd × n_h1, rows=HCurl, cols=H1).
/// `M` is the H(curl) mass matrix (n_nd × n_nd).
pub fn project_divergence_free(
    rhs: &mut [f64],
    g: &CsrMatrix<f64>,
    m: &CsrMatrix<f64>,
    n_h1: usize,
    solve_h1: &dyn Fn(&CsrMatrix<f64>, &[f64], &mut [f64]),
) {
    let n_nd = rhs.len();

    // rhoh = G^T · M · rhs  (project RHS onto H1 coarse space)
    let mut m_rhs = vec![0.0; n_nd];
    m.spmv(rhs, &mut m_rhs);
    let mut rhoh = vec![0.0; n_h1];
    for nd in 0..n_nd {
        for r in g.row_ptr[nd]..g.row_ptr[nd + 1] {
            rhoh[g.col_idx[r] as usize] += g.values[r] * m_rhs[nd];
        }
    }

    // Build A_h1 = G^T · diag(M) · G using mass lumping.
    // For each HCurl DOF nd, find its diagonal M[nd,nd], then for each
    // H1 pair (i,j) connected via G[nd,i] and G[nd,j]:
    //   A_h1[i,j] += G[nd,i] * M[nd,nd] * G[nd,j]
    let mut coo = CooMatrix::new(n_h1, n_h1);
    for nd in 0..n_nd {
        // M[nd, nd] via find_entry
        let md = m.find_entry(nd, nd).map(|p| m.values[p]).unwrap_or(1.0);
        if md.abs() < 1e-30 {
            continue;
        }
        for ri in g.row_ptr[nd]..g.row_ptr[nd + 1] {
            let i = g.col_idx[ri] as usize;
            let gi = g.values[ri];
            for rj in g.row_ptr[nd]..g.row_ptr[nd + 1] {
                let j = g.col_idx[rj] as usize;
                let gj = g.values[rj];
                let val = gi * md * gj;
                if val.abs() > 1e-30 {
                    coo.add(i, j, val);
                }
            }
        }
    }
    let a_h1 = coo.into_csr();

    // Solve A_h1 · x = rhoh
    let mut x_h1 = vec![0.0; n_h1];
    solve_h1(&a_h1, &rhoh, &mut x_h1);

    // rhs −= G · x_h1
    let mut gx = vec![0.0; n_nd];
    g.spmv(&x_h1, &mut gx);
    for i in 0..n_nd {
        rhs[i] -= gx[i];
    }
}
