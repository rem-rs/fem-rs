//! Regression: AMG as the Schur-complement preconditioner for the mixed
//! Darcy problem (MFEM `block-solvers`, `BDPMinresSolver`).
//!
//! The Schur complement `S = B·diag(M)⁻¹·Bᵀ` of the RT0×P0 Darcy system is a
//! symmetric positive-definite M-matrix (all off-diagonals ≤ 0) that is
//! strongly diagonally dominant.  CG with the AMG V-cycle as preconditioner
//! requires that V-cycle to be an SPD operator:
//!
//! * [`fem_amg::boomeramg_config`] (Ruge–Stüben coarsening + symmetric
//!   Gauss–Seidel smoothing — hypre BoomerAMG default semantics) satisfies
//!   this and converges in a small, refinement-bounded iteration count.
//! * The plain `AmgConfig::default()` (smoothed aggregation + weighted
//!   Jacobi ω = 2/3) *used to* fail on these matrices — CG stagnated at
//!   `max_iter`.  The cause was traced (round 11, D10) to the AMG
//!   coarsest-level solve rather than to the smoother or the coarsening:
//!   `linlvo`'s cycle solves the coarse operator with `SparseLu` under its
//!   default `Rcm` fill-reducing ordering, which returns a *permuted*
//!   solution, so the coarse-grid correction was garbage and the V-cycle
//!   operator was not symmetric (measured `max|B − Bᵀ| ≈ 7.8e-2` against
//!   `‖B‖_F ≈ 0.76`).  `fem-amg` now drives every hierarchy through
//!   `CorrectedAmgPrecond`, which factors the coarsest operator with
//!   `OrderingMethod::Natural` (exact for it), so **both** presets are valid
//!   SPD preconditioners here — see
//!   `schur_default_config_amg_cg_converges_across_refinements`.
//!   `darcy_solvers::SchurMode::Amg` still pins the BoomerAMG-aligned preset,
//!   which needs somewhat fewer iterations.

use fem_amg::{AmgConfig, AmgSolver, boomeramg_config};
use fem_assembly::mixed::{HDivL2DivIntegrator, assemble_hdiv_l2_mixed};
use fem_assembly::standard::VectorMassIntegrator;
use fem_assembly::VectorAssembler;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::Mesh;
use fem_space::{HDivSpace, L2Space};

/// Assemble `S = B·diag(M)⁻¹·Bᵀ` for RT0 × P0 on a quad mesh
/// (same assembly as `miniapps/solvers/block_solvers.rs`).
fn assemble_schur(mesh: &Mesh<2>) -> CsrMatrix<f64> {
    let u_sp = HDivSpace::new(mesh.clone(), 0);
    let p_sp = L2Space::new(mesh.clone(), 0);
    let qo = 2u8; // (2·0+1).max(2)
    let m = VectorAssembler::assemble_bilinear(
        &u_sp,
        &[&VectorMassIntegrator { alpha: 1.0 }],
        qo,
    );
    let mut b = assemble_hdiv_l2_mixed(&p_sp, &u_sp, &[&HDivL2DivIntegrator], qo);
    for v in &mut b.values {
        *v *= -1.0;
    }
    let n_u = u_sp.n_dofs();
    let n_p = p_sp.n_dofs();
    let bt = b.transpose();
    let mut minvbt = CooMatrix::<f64>::new(n_u, n_p);
    for i in 0..n_u {
        let inv_d = 1.0 / m.get(i, i).max(1e-300);
        for ptr in bt.row_ptr[i]..bt.row_ptr[i + 1] {
            let j = bt.col_idx[ptr] as usize;
            minvbt.add(i, j, bt.values[ptr] * inv_d);
        }
    }
    b.multiply(&minvbt.into_csr())
}

/// Plain PCG with an external preconditioner; returns (iterations, converged).
fn pcg_with_precond(
    a: &CsrMatrix<f64>,
    precond: &dyn Fn(&[f64], &mut [f64]),
    max_iter: usize,
) -> (usize, bool) {
    let n = a.nrows;
    let b = vec![1.0_f64; n];
    let mut x = vec![0.0_f64; n];
    let mut r = b.clone();
    let mut z = vec![0.0_f64; n];
    precond(&r, &mut z);
    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(ri, zi)| ri * zi).sum();
    for it in 1..=max_iter {
        let mut ap = vec![0.0_f64; n];
        a.spmv(&p, &mut ap);
        let pap: f64 = p.iter().zip(&ap).map(|(pi, ai)| pi * ai).sum();
        if !(pap > 0.0) {
            return (it, false);
        }
        let alpha = rz / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        if r.iter().map(|ri| ri * ri).sum::<f64>().sqrt() < 1e-10 {
            return (it, true);
        }
        precond(&r, &mut z);
        let rz_new: f64 = r.iter().zip(&z).map(|(ri, zi)| ri * zi).sum();
        if rz.abs() < 1e-300 || rz_new < 0.0 {
            return (it, false);
        }
        let beta = rz_new / rz;
        rz = rz_new;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }
    (max_iter, false)
}

/// AMG-CG (BoomerAMG preset) on the Darcy Schur complement converges with a
/// refinement-bounded iteration count (unit-square quad meshes, RT0×P0).
#[test]
fn schur_amg_cg_converges_across_refinements() {
    // n_elem × n_elem quads → n_p = n_elem² pressure dofs (rs0/rs1/rs2 class).
    for n_elem in [4usize, 8, 16] {
        let mesh = Mesh::<2>::unit_square_quad(n_elem);
        let s = assemble_schur(&mesh);
        let n_p = s.nrows;
        let solver = AmgSolver::setup(&s, boomeramg_config());
        assert!(
            solver.n_levels() >= 2,
            "expected a multilevel hierarchy for n_p = {n_p}"
        );
        let (iters, ok) = pcg_with_precond(&s, &|v: &[f64], w: &mut [f64]| {
            let z = solver.precond_apply(v);
            w.copy_from_slice(&z);
        }, 200);
        assert!(
            ok,
            "AMG-CG on Darcy Schur complement failed at n_p = {n_p} (n_elem = {n_elem})"
        );
        assert!(
            iters <= 40,
            "AMG-CG on Darcy Schur complement too slow at n_p = {n_p}: {iters} iterations"
        );
    }
}

/// D10 acceptance: the **default** configuration must also be a valid SPD CG
/// preconditioner on the Darcy Schur complement (same matrices as above).
///
/// Before the coarsest-solve fix the V-cycle operator was not a valid SPD
/// preconditioner on these matrices: on the dumped `star.mesh` Schur
/// complements the measured asymmetry was `max|B − Bᵀ| = 7.8e-2` against
/// `‖B‖_F = 0.76` for smoothed aggregation + weighted Jacobi, and CG either
/// stalled at `max_iter` (levels 0/1) or converged only to a loose residual
/// (level 2).  After the fix the asymmetry is at round-off (`≈1e-17`).
#[test]
fn schur_default_config_amg_cg_converges_across_refinements() {
    for n_elem in [4usize, 8, 16] {
        let mesh = Mesh::<2>::unit_square_quad(n_elem);
        let s = assemble_schur(&mesh);
        let n_p = s.nrows;
        let solver = AmgSolver::setup(&s, AmgConfig::default());
        assert!(
            solver.n_levels() >= 2,
            "expected a multilevel hierarchy for n_p = {n_p}"
        );
        let (iters, ok) = pcg_with_precond(&s, &|v: &[f64], w: &mut [f64]| {
            let z = solver.precond_apply(v);
            w.copy_from_slice(&z);
        }, 200);
        assert!(
            ok,
            "AMG-CG with AmgConfig::default() failed at n_p = {n_p} (n_elem = {n_elem})"
        );
        // Not worse than the RS+SGS preset by more than a small factor.
        let solver_rs = AmgSolver::setup(&s, boomeramg_config());
        let (iters_rs, ok_rs) = pcg_with_precond(&s, &|v: &[f64], w: &mut [f64]| {
            let z = solver_rs.precond_apply(v);
            w.copy_from_slice(&z);
        }, 200);
        assert!(ok_rs);
        println!(
            "n_p = {n_p:>4}: default(SA+WJ) {iters} iters, boomeramg(RS+SGS) {iters_rs} iters"
        );
        assert!(
            iters <= iters_rs * 3 + 5,
            "default config degraded AMG-CG at n_p = {n_p}: {iters} vs {iters_rs} iterations"
        );
        assert!(
            iters <= 60,
            "AMG-CG on Darcy Schur complement too slow at n_p = {n_p}: {iters} iterations"
        );
    }
}

// ─── Dumped-matrix diagnostics (offline; requires tmp/amg dumps) ─────────────

/// Workspace root (integration tests run with CWD = crate dir).
fn ws(rel: &str) -> String {
    format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel)
}

/// Read the COO text dump written by `block_solvers -dump-schur`
/// (1-based `i j v` lines).
fn read_coo(path: &str) -> CsrMatrix<f64> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let mut triples: Vec<(usize, usize, f64)> = Vec::new();
    let mut n = 0usize;
    for line in text.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 3 {
            continue;
        }
        let i: usize = parts[0].parse().unwrap();
        let j: usize = parts[1].parse().unwrap();
        let v: f64 = parts[2].parse().unwrap();
        n = n.max(i).max(j);
        triples.push((i - 1, j - 1, v));
    }
    let mut coo = CooMatrix::<f64>::new(n, n);
    for (i, j, v) in triples {
        coo.add(i, j, v);
    }
    coo.into_csr()
}

/// Same convergence check on the *actual* block-solvers Schur matrices
/// (star.mesh is irregular — not reproducible from structured constructors).
///
/// Regenerate dumps with:
/// `cargo run --release --example block_solvers -- -m data/star.mesh -rs <k> \
///  -solver bdp -dump-schur tmp/amg/S_star_rs<k>.coo`
#[test]
#[ignore = "requires tmp/amg dumps from block_solvers -dump-schur"]
fn schur_dumped_star_mesh_amg_cg_converges() {
    for k in 0..=2 {
        let path = ws(&format!("tmp/amg/S_star_rs{k}.coo"));
        if !std::path::Path::new(&path).exists() {
            continue;
        }
        let s = read_coo(&path);
        let solver = AmgSolver::setup(&s, boomeramg_config());
        let (iters, ok) = pcg_with_precond(&s, &|v: &[f64], w: &mut [f64]| {
            let z = solver.precond_apply(v);
            w.copy_from_slice(&z);
        }, 200);
        assert!(ok, "AMG-CG failed on dumped star rs{k} (n_p = {})", s.nrows);
        assert!(iters <= 40, "AMG-CG too slow on star rs{k}: {iters} iterations");
    }
}
