//! 1D IGA Poisson example (Task 4).
//!
//! Solves:
//!   -u'' = 1  on (0, 1),  u(0)=u(1)=0
//! using [`Assembler::assemble_bilinear_iga_1d`](fem_assembly::Assembler::assemble_bilinear_iga_1d)
//! on an [`IgaFESpace1D`](fem_space::IgaFESpace1D) (B-spline basis, not Line2/3 Lagrange).

use fem_assembly::{Assembler, Iga1dBilinearItem};
use fem_linalg::CsrMatrix;
use fem_space::IgaFESpace1D;
use fem_space::iga::IgaSpace1D;

use fem_examples::solve_dirichlet_reduced;

const DEGREE: usize = 2;
const N_CTRL: usize = 16;
const QUAD_ORDER: u8 = 4;
const SOURCE_VALUE: f64 = 1.0;
const SOLVER_TOL: f64 = 1e-12;
const MAX_ITER: usize = 20_000;

fn main() {
    // 1) IGA space + FESpace bridge (p+1=3 for degree 2: Line3 connectivity).
    let iga = IgaSpace1D::new_uniform_clamped(DEGREE, N_CTRL)
        .expect("failed to build IgaSpace1D");
    let fe = IgaFESpace1D::new(iga.clone()).expect("IgaFESpace1D");

    // 2) Assemble diffusion and source via Assembler IGA entry points.
    let k = Assembler::assemble_bilinear_iga_1d(
        &fe,
        &[Iga1dBilinearItem::Diffusion { kappa: 1.0 }],
        QUAD_ORDER,
    )
    .expect("failed to assemble diffusion matrix");
    let rhs = Assembler::assemble_linear_iga_1d_parametric(&fe, |_u| SOURCE_VALUE, QUAD_ORDER)
        .expect("failed to assemble source vector");

    // 3) Build endpoint Dirichlet BC list defensively.
    let (left, right) = iga.boundary_dofs();
    let dirichlet = build_dirichlet_zero_bcs(iga.n_dofs(), &[left, right]);

    // 4) Solve using the existing reduced-system Dirichlet utility.
    let (u, iters, final_residual) =
        solve_dirichlet_reduced(&k, &rhs, &dirichlet, SOLVER_TOL, MAX_ITER);

    // 5) Enforce convergence, then print residual metric.
    let rel_res_free = relative_residual_free_dofs(&k, &u, &rhs, &dirichlet);
    let converged = iters < MAX_ITER && final_residual <= SOLVER_TOL && rel_res_free <= SOLVER_TOL;
    assert!(
        converged,
        "IGA Poisson solve did not converge: iters={iters}, max_iter={MAX_ITER}, reduced_res={final_residual:.3e}, free_rel_res={rel_res_free:.3e}, tol={SOLVER_TOL:.3e}"
    );
    let l2_u = u.iter().map(|v| v * v).sum::<f64>().sqrt();

    println!("=== 1D IGA Poisson (Task 4) ===");
    println!(
        "degree = {DEGREE}, n_ctrl = {N_CTRL}, n_dofs = {}",
        iga.n_dofs()
    );
    println!("PCG iterations = {iters}, reported residual = {final_residual:.3e}");
    println!("free-DOF ||Ku-f||/||f|| = {rel_res_free:.3e}");
    println!("||u||_2 = {l2_u:.3e}");
}

fn build_dirichlet_zero_bcs(n_dofs: usize, dofs: &[usize]) -> Vec<(usize, f64)> {
    let mut uniq = Vec::new();
    for &dof in dofs {
        if dof < n_dofs && !uniq.contains(&dof) {
            uniq.push(dof);
        }
    }
    uniq.sort_unstable();
    uniq.into_iter().map(|dof| (dof, 0.0)).collect()
}

fn relative_residual_free_dofs(
    mat: &CsrMatrix<f64>,
    x: &[f64],
    rhs: &[f64],
    dirichlet: &[(usize, f64)],
) -> f64 {
    let mut is_dirichlet = vec![false; rhs.len()];
    for &(dof, _) in dirichlet {
        if dof < rhs.len() {
            is_dirichlet[dof] = true;
        }
    }

    let mut ax = vec![0.0; rhs.len()];
    mat.spmv(x, &mut ax);
    let r_norm = ax
        .iter()
        .zip(rhs.iter())
        .enumerate()
        .filter(|(i, _)| !is_dirichlet[*i])
        .map(|(_, (ax_i, b_i))| {
            let ri = ax_i - b_i;
            ri * ri
        })
        .sum::<f64>()
        .sqrt();
    let b_norm = rhs
        .iter()
        .enumerate()
        .filter(|(i, _)| !is_dirichlet[*i])
        .map(|(_, v)| v * v)
        .sum::<f64>()
        .sqrt()
        .max(1e-30);
    r_norm / b_norm
}
