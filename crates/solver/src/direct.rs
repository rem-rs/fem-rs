use fem_linalg::CsrMatrix as FemCsr;
use linlvo::{
    core::scalar::Scalar as linlvoScalar,
    direct::{DirectSolver, MklSolver, MumpsSolver, SparseCholesky, SparseLdlt, SparseLu},
    DenseVec,
};
use fem_linalg::{fem_to_linlvo_csr, SolverError};

solve_direct!(solve_sparse_lu, SparseLu<T>, "Sparse LU direct solver for general square systems.");

solve_direct!(solve_sparse_cholesky, SparseCholesky<T>, "Sparse Cholesky for symmetric positive-definite systems.");

solve_direct!(solve_sparse_ldlt, SparseLdlt<T>, "Sparse LDL^T for symmetric indefinite systems.");

solve_direct!(solve_sparse_mumps, MumpsSolver<T>, "MUMPS-compatible direct solver.");

solve_direct!(solve_sparse_mkl, MklSolver<T>, "MKL-compatible direct solver.");
