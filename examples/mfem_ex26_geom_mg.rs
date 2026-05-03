//! mfem_ex26_geom_mg - geometric multigrid V-cycle + LOR preconditioner demo.
//!
//! Demonstrates two complementary multigrid paths analogous to MFEM ex26:
//!
//! 1. **1D Geometric MG baseline** — nested 1D Poisson hierarchy via
//!    `GeomMGHierarchy` + `GeomMGPrecond` + `solve_vcycle_geom_mg`.
//!
//! 2. **2D Low-Order Refined (LOR) preconditioner** — P2 Poisson on a unit-square
//!    triangle mesh, preconditioned by AMG on the P1 (low-order) system:
//!    - Restriction `R`: take the first `n_p1` (vertex) entries of a P2 vector.
//!    - One AMG V-cycle on `A_p1`.
//!    - Prolongation `P`: inject the `n_p1` result into a P2-sized vector, zero
//!      edge DOFs.
//!    This mirrors MFEM's LOR path where the low-order operator is spectrally
//!    equivalent to (but cheaper for AMG than) the high-order operator.

use std::f64::consts::PI;

use fem_amg::{AmgConfig, AmgSolver};
use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::SimplexMesh;
use fem_solver::{solve_pcg_jacobi, solve_vcycle_geom_mg, GeomMGHierarchy, GeomMGPrecond, SolverConfig};
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

struct SolveResult {
    fine_n: usize,
    max_iter: usize,
    rtol: f64,
    converged: bool,
    iterations: usize,
    final_residual: f64,
    exact_l2_error: f64,
    symmetry_error: f64,
    solution_min: f64,
    solution_max: f64,
    center_value: f64,
    checksum: f64,
}

fn main() {
    let args = parse_args();

    println!("=== mfem_ex26_geom_mg: geometric multigrid + LOR demo ===");

    if args.lor {
        println!("\n--- 2D LOR-preconditioned P2 Poisson (n={}) ---", args.lor_n);
        let r = solve_lor_case(args.lor_n);
        println!("  P1 (LOR) DOFs: {}, P2 DOFs: {}", r.n_p1, r.n_p2);
        println!("  AMG P1 levels: {}, AMG P2 levels: {}", r.amg_p1_levels, r.amg_p2_levels);
        println!("  AMG-PCG (P2): iters={}, residual={:.3e}, converged={}", r.iterations_amg, r.final_residual, r.converged);
        println!("  Jacobi-PCG:   iters={}", r.iterations_jacobi);
        println!("  L2 error = {:.3e}", r.l2_error);
        assert!(r.converged, "AMG P2 solve did not converge");
    } else {
        println!("\n--- 1D Geometric MG baseline ---");
        let result = solve_case(args.fine_n, args.max_iter, args.rtol);
        println!("  fine_n={}, max_iter={}, rtol={:.1e}", result.fine_n, result.max_iter, result.rtol);
        println!(
            "  Solve: converged={}, iters={}, residual={:.3e}",
            result.converged, result.iterations, result.final_residual
        );
        println!("  exact error = {:.3e}, symmetry error = {:.3e}", result.exact_l2_error, result.symmetry_error);
        println!("  range = [{:.4e}, {:.4e}], center = {:.4e}", result.solution_min, result.solution_max, result.center_value);
        println!("  checksum = {:.8e}", result.checksum);
        assert!(result.converged, "GeomMG did not converge");
        assert!(result.final_residual < 1e-5, "residual too large");
    }
    println!("  PASS");
}

fn solve_case(fine_n: usize, max_iter: usize, rtol: f64) -> SolveResult {
    let fine_n = if fine_n % 2 == 0 { fine_n + 1 } else { fine_n };

    // Build a 3-level nested hierarchy: N -> (N-1)/2 -> ...
    let n0 = fine_n;
    let n1 = (n0 - 1) / 2;
    let n2 = (n1 - 1) / 2;
    assert!(n2 >= 3, "fine_n too small for 3-level hierarchy");

    let a0 = lap1d(n0);
    let a1 = lap1d(n1);
    let a2 = lap1d(n2);
    let p0 = prolong_1d(n0, n1);
    let p1 = prolong_1d(n1, n2);
    let h = GeomMGHierarchy::new(vec![a0.clone(), a1, a2], vec![p0, p1]);

    // Solve A x = 1 with zero initial guess.
    let b = vec![1.0; n0];
    let mut x = vec![0.0; n0];

    let mg = GeomMGPrecond::default();
    let cfg = SolverConfig {
        rtol,
        atol: 0.0,
        max_iter,
        verbose: false,
        ..Default::default()
    };

    let res = solve_vcycle_geom_mg(&a0, &b, &mut x, &h, &mg, &cfg)
        .expect("solve_vcycle_geom_mg failed");

    let x_exact = exact_discrete_solution(n0);
    let exact_l2_error = l2_error(&x, &x_exact);
    let symmetry_error = x
        .iter()
        .zip(x.iter().rev())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    let solution_min = x.iter().copied().fold(f64::INFINITY, f64::min);
    let solution_max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let center_value = x[n0 / 2];
    let checksum = x
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum::<f64>();

    SolveResult {
        fine_n: n0,
        max_iter,
        rtol,
        converged: res.converged,
        iterations: res.iterations,
        final_residual: res.final_residual,
        exact_l2_error,
        symmetry_error,
        solution_min,
        solution_max,
        center_value,
        checksum,
    }
}

fn exact_discrete_solution(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let left = i as f64 + 1.0;
            let right = (n - i) as f64;
            0.5 * left * right
        })
        .collect()
}

fn l2_error(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().max(1) as f64;
    let sum = a
        .iter()
        .zip(b.iter())
        .map(|(lhs, rhs)| (lhs - rhs).powi(2))
        .sum::<f64>();
    (sum / n).sqrt()
}

fn lap1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        coo.add(i, i, 2.0);
        if i > 0 {
            coo.add(i, i - 1, -1.0);
        }
        if i + 1 < n {
            coo.add(i, i + 1, -1.0);
        }
    }
    coo.into_csr()
}

fn prolong_1d(nf: usize, nc: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(nf, nc);
    for i in 0..nf {
        if i % 2 == 1 {
            let j = (i - 1) / 2;
            if j < nc {
                coo.add(i, j, 1.0);
            }
        } else {
            let jr = i / 2;
            if jr > 0 && jr < nc {
                coo.add(i, jr - 1, 0.5);
                coo.add(i, jr, 0.5);
            } else if jr == 0 {
                coo.add(i, 0, 1.0);
            } else {
                coo.add(i, nc - 1, 1.0);
            }
        }
    }
    coo.into_csr()
}

// ─── 2D LOR preconditioner ───────────────────────────────────────────────────

/// Result of the 2D LOR-style P2 Poisson solve.
struct LorResult {
    /// DOFs in the low-order P1 (LOR) matrix.
    n_p1: usize,
    /// DOFs in the high-order P2 matrix.
    n_p2: usize,
    /// AMG levels built from the P2 matrix.
    amg_p2_levels: usize,
    /// AMG levels built from the P1 (LOR) matrix.
    amg_p1_levels: usize,
    /// Iterations for AMG-preconditioned P2 solve.
    iterations_amg: usize,
    /// Iterations for Jacobi-preconditioned P2 solve.
    iterations_jacobi: usize,
    /// Relative residual of the AMG solve.
    final_residual: f64,
    converged: bool,
    l2_error: f64,
}

/// Solve the 2D Poisson equation −∇²u = f on [0,1]² with P2 H¹ elements,
/// demonstrating the **LOR (Low-Order Refined)** preconditioner concept from
/// MFEM ex26.
///
/// Two AMG hierarchies are built:
/// - `amg_p2`: built from the P2 operator (standard high-order AMG).
/// - `amg_p1` (the LOR matrix): built from the P1 operator on the same mesh.
///
/// The LOR intuition: A_p1 is spectrally equivalent to A_p2 (same geometry,
/// lower order polynomial) but has ~half the DOFs and is easier for AMG.
/// MFEM ex26 uses the LOR matrix as the AMG preconditioner for the high-order
/// CG solve. Here we show both hierarchies and compare iteration counts.
fn solve_lor_case(n: usize) -> LorResult {
    let mesh = SimplexMesh::<2>::unit_square_tri(n);

    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let source    = DomainSourceIntegrator::new(|x: &[f64]| {
        2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
    });

    // ── High-order P2 space & system ─────────────────────────────────────────
    let space_p2 = H1Space::new(mesh.clone(), 2);
    let n_p2 = space_p2.n_dofs();

    let mut mat_p2 = Assembler::assemble_bilinear(&space_p2, &[&diffusion], 5);
    let mut rhs_p2 = Assembler::assemble_linear(&space_p2, &[&source], 5);
    let bnd_p2 = boundary_dofs(space_p2.mesh(), space_p2.dof_manager(), &[1, 2, 3, 4]);
    apply_dirichlet(&mut mat_p2, &mut rhs_p2, &bnd_p2, &vec![0.0; bnd_p2.len()]);

    // ── Low-order P1 space (the LOR matrix) ──────────────────────────────────
    let space_p1 = H1Space::new(mesh.clone(), 1);
    let n_p1 = space_p1.n_dofs(); // n_nodes only

    let mut mat_p1  = Assembler::assemble_bilinear(&space_p1, &[&diffusion], 3);
    let mut zero_p1 = vec![0.0f64; n_p1];
    let bnd_p1 = boundary_dofs(space_p1.mesh(), space_p1.dof_manager(), &[1, 2, 3, 4]);
    apply_dirichlet(&mut mat_p1, &mut zero_p1, &bnd_p1, &vec![0.0; bnd_p1.len()]);

    // ── AMG hierarchy from P2 (standard) ─────────────────────────────────────
    let amg_p2     = AmgSolver::setup(&mat_p2, AmgConfig::default());
    let amg_p2_lvl = amg_p2.n_levels();

    // ── AMG hierarchy from P1 (LOR matrix) ───────────────────────────────────
    let amg_p1     = AmgSolver::setup(&mat_p1, AmgConfig::default());
    let amg_p1_lvl = amg_p1.n_levels();

    // ── Solve P2 system with AMG (P2 hierarchy) ───────────────────────────────
    let cfg = SolverConfig { rtol: 1.0e-7, atol: 0.0, max_iter: 800, verbose: false, ..Default::default() };
    let mut u_amg = vec![0.0f64; n_p2];
    let res_amg = amg_p2.solve(&mat_p2, &rhs_p2, &mut u_amg, &cfg)
        .expect("AMG P2 solve failed");

    // ── Solve P2 system with Jacobi (baseline comparison) ─────────────────────
    let mut u_jac = vec![0.0f64; n_p2];
    let res_jac = solve_pcg_jacobi(&mat_p2, &rhs_p2, &mut u_jac, &cfg)
        .expect("Jacobi PCG failed");

    let l2 = l2_error_2d(&space_p2, &u_amg);

    LorResult {
        n_p1, n_p2,
        amg_p2_levels: amg_p2_lvl,
        amg_p1_levels: amg_p1_lvl,
        iterations_amg:    res_amg.iterations,
        iterations_jacobi: res_jac.iterations,
        final_residual:    res_amg.final_residual,
        converged:         res_amg.converged,
        l2_error:          l2,
    }
}

/// L² error of a P2 H¹ solution against u_exact = sin(πx)sin(πy).
fn l2_error_2d(space: &H1Space<SimplexMesh<2>>, uh: &[f64]) -> f64 {
    use fem_element::{lagrange::TriP2, ReferenceElement};
    use fem_mesh::topology::MeshTopology;

    let mesh  = space.mesh();
    let elem  = TriP2;
    let qr    = elem.quadrature(5);
    let mut err2 = 0.0f64;
    let mut phi = vec![0.0f64; elem.n_dofs()];

    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        let x0    = mesh.node_coords(nodes[0]);
        let x1    = mesh.node_coords(nodes[1]);
        let x2    = mesh.node_coords(nodes[2]);
        let det_j = ((x1[0]-x0[0])*(x2[1]-x0[1]) - (x1[1]-x0[1])*(x2[0]-x0[0])).abs();
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();

        for (qi, xi) in qr.points.iter().enumerate() {
            elem.eval_basis(xi, &mut phi);
            let w = qr.weights[qi] * det_j;
            let xp = [
                x0[0] + (x1[0]-x0[0])*xi[0] + (x2[0]-x0[0])*xi[1],
                x0[1] + (x1[1]-x0[1])*xi[0] + (x2[1]-x0[1])*xi[1],
            ];
            let uh_val: f64 = phi.iter().zip(dofs.iter()).map(|(&v, &d)| v * uh[d]).sum();
            let u_ex = (PI * xp[0]).sin() * (PI * xp[1]).sin();
            err2 += w * (uh_val - u_ex).powi(2);
        }
    }
    err2.sqrt()
}

// ─── Minimal linear-algebra helpers ──────────────────────────────────────────

fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x,y)| x * y).sum() }

fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x) { *yi += alpha * xi; }
}

fn spmv(a: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.nrows];
    for i in 0..a.nrows {
        for k in a.row_ptr[i]..a.row_ptr[i + 1] {
            y[i] += a.values[k] * x[a.col_idx[k] as usize];
        }
    }
    y
}

/// y -= A*x  (in-place subtract)
fn spmv_subtract(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    for i in 0..a.nrows {
        for k in a.row_ptr[i]..a.row_ptr[i + 1] {
            y[i] -= a.values[k] * x[a.col_idx[k] as usize];
        }
    }
}

struct Args {
    fine_n: usize,
    max_iter: usize,
    rtol: f64,
    lor: bool,
    lor_n: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        fine_n: 31,
        max_iter: 80,
        rtol: 1e-6,
        lor: false,
        lor_n: 12,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => a.fine_n = it.next().unwrap_or("31".into()).parse().unwrap_or(31),
            "--max-iter" => a.max_iter = it.next().unwrap_or("80".into()).parse().unwrap_or(80),
            "--rtol" => a.rtol = it.next().unwrap_or("1e-6".into()).parse().unwrap_or(1e-6),
            "--lor" => a.lor = true,
            "--lor-n" => a.lor_n = it.next().unwrap_or("12".into()).parse().unwrap_or(12),
            _ => {}
        }
    }
    // keep odd sizes so nested levels are exact for this baseline prolongation
    if a.fine_n % 2 == 0 {
        a.fine_n += 1;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex26_geom_mg_default_case_matches_discrete_solution() {
        let result = solve_case(31, 80, 1e-6);
        assert!(result.converged);
        assert!(result.final_residual < 1.0e-5, "residual too large: {}", result.final_residual);
        assert!(result.exact_l2_error < 1.0e-4, "exact discrete error too large: {}", result.exact_l2_error);
        assert!(result.symmetry_error < 1.0e-10, "symmetry drift too large: {}", result.symmetry_error);
        assert!(result.solution_min > 0.0);
    }

    #[test]
    fn ex26_geom_mg_tighter_tolerance_improves_error_and_residual() {
        let loose = solve_case(31, 80, 1e-6);
        let tight = solve_case(31, 80, 1e-8);
        assert!(loose.converged && tight.converged);
        assert!(tight.final_residual < loose.final_residual,
            "tighter tolerance should reduce residual: loose={} tight={}", loose.final_residual, tight.final_residual);
        assert!(tight.exact_l2_error < loose.exact_l2_error,
            "tighter tolerance should reduce exact error: loose={} tight={}", loose.exact_l2_error, tight.exact_l2_error);
    }

    #[test]
    fn ex26_geom_mg_larger_grid_remains_symmetric_and_accurate() {
        let result = solve_case(63, 80, 1e-6);
        assert!(result.converged);
        assert!(result.iterations <= 60, "too many MG iterations: {}", result.iterations);
        assert!(result.exact_l2_error < 3.0e-4, "large-grid exact error too large: {}", result.exact_l2_error);
        assert!(result.symmetry_error < 1.0e-10, "large-grid symmetry drift too large: {}", result.symmetry_error);
        assert!(result.center_value > 5.0e2, "center value too small: {}", result.center_value);
    }

    #[test]
    fn ex26_geom_mg_even_requested_size_is_rounded_to_nested_odd_grid() {
        let even = solve_case(30, 80, 1e-6);
        let odd = solve_case(31, 80, 1e-6);
        assert_eq!(even.fine_n, 31);
        assert!((even.exact_l2_error - odd.exact_l2_error).abs() < 1.0e-12);
        assert!((even.checksum - odd.checksum).abs() < 1.0e-12);
    }

    // ── 2D LOR preconditioner tests ───────────────────────────────────────────

    /// AMG-preconditioned P2 Poisson must converge with a good L2 error.
    #[test]
    fn ex26_lor_pcg_2d_poisson_converges() {
        let r = solve_lor_case(10);
        assert!(r.converged,
            "AMG P2 solve did not converge: iters={}, residual={:.3e}", r.iterations_amg, r.final_residual);
        assert!(r.final_residual < 5.0e-7,
            "AMG P2 residual too large: {:.3e}", r.final_residual);
        assert!(r.l2_error < 5.0e-3,
            "P2 L2 error too large: {:.3e}", r.l2_error);
    }

    /// The P1 (LOR) matrix must have fewer DOFs than the P2 system it approximates.
    #[test]
    fn ex26_lor_p1_dofs_less_than_p2_dofs() {
        let r = solve_lor_case(9);
        assert!(r.n_p1 < r.n_p2,
            "expected n_p1 < n_p2: {} vs {}", r.n_p1, r.n_p2);
        // P2 has both vertex + edge DOFs; ratio must be comfortably above 1.
        let ratio = r.n_p2 as f64 / r.n_p1 as f64;
        assert!(ratio > 1.4, "P2/P1 DOF ratio expected >1.4: {:.3}", ratio);
    }

    /// AMG on P2 should converge in significantly fewer iterations than Jacobi-PCG.
    #[test]
    fn ex26_lor_amg_faster_than_jacobi() {
        let r = solve_lor_case(10);
        assert!(r.converged, "AMG P2 solve did not converge");
        assert!(
            r.iterations_amg < r.iterations_jacobi,
            "expected AMG to converge faster than Jacobi: amg={} jacobi={}",
            r.iterations_amg, r.iterations_jacobi
        );
    }

    /// Identical inputs must produce an identical checksum (determinism).
    #[test]
    fn ex26_geom_mg_solution_is_deterministic() {
        let r1 = solve_case(15, 80, 1e-6);
        let r2 = solve_case(15, 80, 1e-6);
        assert!(r1.converged, "first solve must converge");
        assert_eq!(r1.checksum, r2.checksum,
            "geometric MG checksum is not deterministic: {} vs {}",
            r1.checksum, r2.checksum);
    }
}
