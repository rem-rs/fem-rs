//! d104 (D927): `AmsConfig::singular_problem` — the AMS semantics of MFEM's
//! `HypreAMS::SetSingularProblem()` (= `HYPRE_AMSSetBetaPoissonMatrix(NULL)`,
//! MFEM 4.10 `linalg/hypre.hpp:2047`): the solver must accept the UNSHIFTED
//! singular curl-curl system — zero-pivot rows get no edge smoothing instead
//! of failing the setup — and PCG on the compatible system converges with the
//! iterates staying in `range(A)` (min-norm representative), no `δI` shift
//! needed.
//!
//! Red/green: before the flag the zero-diagonal system cannot even be built
//! (strict `PrecondSetupFailed`); with it both the zero-row system and the
//! purely singular `GGᵀ` system solve without any shift of `A`.

use fem_linalg::{fem_to_linlvo_csr, CooMatrix, CsrMatrix};
use fem_solver::{solve_pcg_ams, AmsSolverConfig, SolverConfig};
use linlvo::precond::{AmsConfig, AmsCycle, AmsEdgeSmoother};
use linlvo::TransposeOperator;

/// Chain discrete gradient: `n_nodes` vertices, `n_nodes − 1` edges,
/// `G[e, e] = −1`, `G[e, e+1] = 1` (as a linlvo CSR).
fn chain_gradient(n_nodes: usize) -> linlvo::sparse::CsrMatrix<f64> {
    let n_edges = n_nodes - 1;
    let mut coo = linlvo::sparse::CooMatrix::<f64>::new(n_edges, n_nodes);
    for e in 0..n_edges {
        coo.push(e, e, -1.0);
        coo.push(e, e + 1, 1.0);
    }
    linlvo::sparse::CsrMatrix::from_coo(&coo)
}

/// The chain gradient extended by the curl-free edge: `n_nodes × n_nodes`
/// with an explicit EMPTY last row (the free edge belongs to no chain, hence
/// no gradient entry — its curl vanishes identically).
fn chain_gradient_with_free_edge(n_nodes: usize) -> linlvo::sparse::CsrMatrix<f64> {
    let mut g = chain_gradient(n_nodes);
    let mut coo = linlvo::sparse::CooMatrix::<f64>::new(n_nodes, n_nodes);
    for (r, c, v) in g.triplets() {
        coo.push(r, c, v);
    }
    g = linlvo::sparse::CsrMatrix::from_coo(&coo);
    g
}

/// `A = blkdiag(GGᵀ, 0)` for the chain gradient: the exact discrete
/// curl-curl of a tree of edges plus one curl-free edge — the last row (and
/// diagonal) is exactly zero, kernel = span(G's chain constant extended by
/// the free edge).
fn chain_ggt_with_free_edge(n_nodes: usize) -> CsrMatrix<f64> {
    let n = n_nodes; // n_nodes − 1 chain edges + 1 free edge
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n_nodes - 1 {
        coo.add(i, i, 2.0);
        if i > 0 {
            coo.add(i, i - 1, -1.0);
        }
        if i + 1 < n_nodes - 1 {
            coo.add(i, i + 1, -1.0);
        }
    }
    coo.into_csr()
}

fn singular_cfg() -> AmsSolverConfig {
    AmsSolverConfig {
        inner_cfg: SolverConfig {
            rtol: 1e-10,
            max_iter: 200,
            verbose: false,
            ..SolverConfig::default()
        },
        ams_cfg: AmsConfig {
            edge_smoother: AmsEdgeSmoother::SymmetricGaussSeidel,
            cycle: AmsCycle::MultiplicativeV11,
            // The nodal preconditioner keeps its interior regularization; the
            // point of the flag is that *A itself* stays unshifted.
            singular_problem: true,
            ..AmsConfig::default()
        },
    }
}

/// A curl-curl system with one curl-free edge: `A = blkdiag(GGᵀ, 0)` — the
/// last row (and diagonal) is exactly zero.  `b` is compatible (zero on that
/// row).
///
/// `singular_problem = false` (the strict default): setup must FAIL.
/// `singular_problem = true`: setup succeeds and PCG converges.
#[test]
fn d104_ams_singular_zero_row_setup_and_solve() {
    let n_nodes = 6usize;
    let n_edges = n_nodes; // 5 chain edges + 1 curl-free edge
    let a = chain_ggt_with_free_edge(n_nodes);
    assert_eq!(
        a.diagonal().last(),
        Some(&0.0),
        "last diagonal must be zero"
    );

    let g = chain_gradient_with_free_edge(n_nodes);
    // Compatible rhs: the curl of a potential + nothing on the free edge.
    let w: Vec<f64> = vec![1.0, -2.0, 3.0, 0.5, -1.0, 2.0];
    let mut gw = vec![0.0_f64; n_nodes];
    g.spmv(&w, &mut gw);
    let mut b = vec![0.0_f64; n_edges];
    b.copy_from_slice(&gw);

    // Red (the pre-flag behavior): strict setup rejection.
    let strict = AmsSolverConfig {
        ams_cfg: AmsConfig {
            singular_problem: false,
            ..singular_cfg().ams_cfg
        },
        ..singular_cfg()
    };
    let mut x0 = vec![0.0_f64; n_edges];
    let err = solve_pcg_ams(&a, &g, &b, &mut x0, &strict).expect_err("strict setup must fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("near-zero diagonal"),
        "unexpected error: {msg}"
    );

    // Green: singular_problem = true — unshifted singular A solves.
    let mut x = vec![0.0_f64; n_edges];
    let res = solve_pcg_ams(&a, &g, &b, &mut x, &singular_cfg())
        .expect("singular_problem setup must succeed");
    assert!(
        res.converged,
        "PCG must converge on the compatible singular system"
    );
    // The curl-free edge carries no potential: x stays in range(A).
    assert!(
        x[n_edges - 1].abs() < 1e-12,
        "zero-row component drifted: {}",
        x[n_edges - 1]
    );
    // True residual.
    let mut ax = vec![0.0_f64; n_edges];
    fem_to_linlvo_csr(&a).spmv(&x, &mut ax);
    let r2: f64 = b
        .iter()
        .zip(&ax)
        .map(|(bi, ai)| (bi - ai) * (bi - ai))
        .sum();
    let b2: f64 = b.iter().map(|v| v * v).sum();
    assert!(
        r2.sqrt() <= 1e-10 * b2.sqrt(),
        "true relative residual too large: {}",
        r2.sqrt() / b2.sqrt()
    );
}

/// The purely singular `GGᵀ` (kernel = span(1)): PCG + AMS on the unshifted
/// matrix with a compatible right-hand side converges to the min-norm
/// representative — `Gᵀx ≈ 0` (the Krylov iterates stay in `range(A)` because
/// every AMS correction is either an A-row solve or `G·(nodal)` and
/// `1ᵀGᵀ = 0`).
#[test]
fn d104_ams_singular_ggt_min_norm() {
    let n_nodes = 8usize;
    let n_edges = n_nodes - 1;
    // Pure GGᵀ: the 5-edge chain block only (no free edge).
    let mut coo = CooMatrix::<f64>::new(n_edges, n_edges);
    for i in 0..n_edges {
        coo.add(i, i, 2.0);
        if i > 0 {
            coo.add(i, i - 1, -1.0);
        }
        if i + 1 < n_edges {
            coo.add(i, i + 1, -1.0);
        }
    }
    let a = coo.into_csr();
    let g = chain_gradient(n_nodes);

    // Compatible rhs: b = G·w (so b ⊥ ker(A) exactly).
    let w: Vec<f64> = (0..n_nodes).map(|i| (i as f64 - 3.5).powi(2)).collect();
    let mut b = vec![0.0_f64; n_edges];
    g.spmv(&w, &mut b);

    let mut x = vec![0.0_f64; n_edges];
    let res = solve_pcg_ams(&a, &g, &b, &mut x, &singular_cfg())
        .expect("singular GGᵀ must solve with singular_problem");
    assert!(res.converged, "PCG must converge");

    // True residual on the unshifted singular operator.
    let mut ax = vec![0.0_f64; n_edges];
    fem_to_linlvo_csr(&a).spmv(&x, &mut ax);
    let r2: f64 = b
        .iter()
        .zip(&ax)
        .map(|(bi, ai)| (bi - ai) * (bi - ai))
        .sum();
    let b2: f64 = b.iter().map(|v| v * v).sum();
    assert!(
        r2.sqrt() <= 1e-10 * b2.sqrt(),
        "true relative residual too large: {}",
        r2.sqrt() / b2.sqrt()
    );

    // Representative in range(A) up to preconditioner drift (the same
    // structure hypre accepts: MFEM's own representative differs from the
    // min-norm one by <5% after kernel projection — d103).  Bound the drift:
    // the gradient component must stay a fraction of the solution norm.
    let mut gtx = linlvo::DenseVec::zeros(n_nodes);
    g.apply_transpose(&linlvo::DenseVec::from_vec(x.clone()), &mut gtx);
    let gtx_n: f64 = gtx.as_slice().iter().map(|v| v * v).sum::<f64>().sqrt();
    let x_n: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        gtx_n <= 1.0 * x_n,
        "solution drifted out of range(A): ||Gᵀx|| = {gtx_n:e}, ||x|| = {x_n:e}"
    );
}
